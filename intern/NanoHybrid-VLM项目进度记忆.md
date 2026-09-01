# NanoHybrid-VLM 项目进度记忆

> 更新时间：2026-09-01
> 当前分支：`main`  
> 最近提交：`1883065 trying to achieve prefix cache`
> 用途：记录实际完成、已经验证、当前断点和下一步。  
> V1 完成度：约 90%～95%；文本/单图 Hybrid Runtime 已接通。
> 扩展后总体完成度：约 70%～75%；V2 纯文本 GDN-aware Prefix State Cache 已完成联合 KV/GDN 命中、BF16 snapshot、共享块容量核算、LRU 和频率准入，并通过完整回归与污染压测；下一步进入 V3 Hybrid Decode CUDA Graph，V4 GDN Decode 算子尚未实现。

## 1. 项目目标与边界

基于 nano-vLLM 实现 Qwen3.5 Hybrid Runtime：

```text
PIL Image → Vision Transformer → Patch Merger → visual embeddings
                                                        ↓
text token embeddings ─────────────────────────→ 图文 embeddings
                                                        ↓
                         3 × Gated DeltaNet + 1 × Full Attention
                                                        ↓
                                                生成文本 token
```

当前主要开发模型：

```text
/workspace/models/Qwen3.5-9B
```

V1 范围：RTX 5090、BF16、TP=1、Eager；每个请求最多一张本地 PIL 图片。V1 不支持网络 URL、多图、视频、MTP、MoE、CUDA Graph、自研 GDN Kernel 和 TP>1。

扩展后的正式范围：

```text
V1：Qwen3.5-9B 文本/单图 Hybrid Runtime（当前主要功能已完成）
V2：纯文本 GDN-aware Prefix State Cache（已完成并通过回归与性能实验）
V3：Qwen3.5 Hybrid Decode-only CUDA Graph（已规划，未实现）
V4：状态感知 GDN Decode 融合算子（已规划，未实现；必须先完成真实 Profile）
```

MTP、MoE、TP>1、多图、视频、图文 Prefix Cache 和完整 Chunk Gated Delta Rule 自研 Kernel 仍不在当前正式范围内。

## 2. 已实现的核心技术

### 2.1 Config、Registry 和权重加载

- `Config` 区分 `hf_config`、`text_config`、`vision_config`。
- 从 `text_config.layer_types` 识别 GDN 与 Full Attention 层。
- 支持 `num_state_slots`、`gdn_state_memory_fraction`、`max_prefill_wait_ms`。
- 含 GDN 的 Qwen3.5 自动关闭 Prefix Cache，禁止只复用 KV、不复用对应 GDN state。
- Registry 根据 HF `model_type`/`architectures` 惰性选择 Qwen3 或 Qwen3.5。
- `ModelRunner` 已解除对 Qwen3 的硬编码。
- loader 支持 safetensors、packed shard、缺失/多余/未知权重检查。
- Vision 权重已纳入严格检查；当前只明确忽略 `mtp.*`。

### 2.2 Greedy 和 HF Golden

- `temperature=0` 执行真正的 `argmax` greedy decoding。
- 已保存文本 Golden：`artifacts/golden/qwen35_08b_text_greedy.json`。
- 一条文本请求在 `max_tokens=36, ignore_eos=True` 下，36/36 token 与 HF 一致，`first_mismatch=None`。

### 2.3 Qwen3.5 文本模型

`nanovllm/models/qwen3_5.py` 已实现：

- Full Attention、GQA、Q/K Norm、attention output sigmoid gate。
- `partial_rotary_factor=0.25` 的 Partial RoPE。
- Qwen3.5 RMSNorm、SwiGLU、两条残差连接。
- 按 `layer_types` 选择 GDN 或 Full Attention。
- 共享 Embedding/LM Head。
- `input_ids`/`inputs_embeds` 二选一前向接口。

### 2.4 Gated DeltaNet

`nanovllm/layers/gated_delta_net.py` 已实现：

- Q/K/V/Z、beta、gate 投影与 `mixed_qkv` 拆分。
- Depthwise causal convolution 和 `conv_state` 更新。
- FP32 `recurrent_state` 的 delta-rule 更新。
- FLA chunk 路径用于多 token Prefill。
- FLA fused recurrent 路径用于有旧状态的单 token Decode。
- PyTorch reference 路径用于公式验证和数值排障。

```text
conv_state：保存短卷积需要的最近局部历史。
recurrent_state：保存 Gated Delta Rule 压缩的长期键值映射。
```

### 2.5 Hybrid Cache 和状态池

`nanovllm/engine/hybrid_state.py` 已实现：

- `GDNLayerState`、`HybridCacheSpec`、`StateSlotAllocator`、`HybridStateManager`。
- 全局 Decoder 层编号到紧凑 GDN/Attention 缓存编号的映射。
- 只为 Full Attention 层分配 Paged KV Cache。
- 为 GDN 层分配 conv/recurrent state pool。
- state slot 初始化、重置、回收、复用和多请求 Gather/Scatter。
- KV block 与 state slot 的显存字节数计算。

```text
KV Cache:
[2, num_full_attention_layers, num_blocks,
 block_size, num_kv_heads, head_dim]

conv_state_pool:
[num_slots, num_gdn_layers, conv_dim, kernel_size]

recurrent_state_pool:
[num_slots, num_gdn_layers, value_heads, key_dim, value_dim]
```

`ModelRunner` 已通过 warmup 测量临时激活峰值，先分配 GDN state 预算，再将剩余显存用于紧凑 KV Cache，并把 K/V Tensor 绑定到对应 Full Attention 模块。

### 2.6 Decode-first 状态感知调度

`nanovllm/engine/scheduler.py` 已实现：

- `SchedulePlan` 同时记录 Decode/Prefill microbatch。
- 一轮 step 先 Decode，再用剩余 token/sequence budget 执行 Prefill。
- 新请求必须同时获得 active capacity、KV blocks 和 GDN state slot。
- Chunked Prefill/Decode 持续持有原来的 KV blocks 和 state slot。
- `max_prefill_wait_ms` 饥饿保护，为超时 Prefill 保留 slot 和 token budget。
- 抢占未进入本轮 Decode、且重算成本较低的请求。
- 抢占释放 KV/state，恢复时从 token 0 确定性重算。
- 统计 `num_preemptions`、`num_recomputed_tokens`。

`LLMEngine.step()` 已按以下顺序执行：

```text
Decode microbatch → postprocess → Prefill microbatch → postprocess
```

### 2.7 Variable-length Batched Prefill 和 Batched Decode

文本 Hybrid Prefill 已经不是逐 Sequence 整模型循环，而是：

```text
A chunk | B chunk | C chunk
          ↓ packed
一次 Qwen3.5 模型前向
```

`prepare_prefill()` 已构造 packed `input_ids/positions`、`cu_seqlens_q/k`、`gdn_cu_seqlens`、`prefill_seqlens`、`slot_mapping` 和需要时的 `block_tables`。

`run_hybrid_prefill()` 已实现批量 Gather 旧 GDN state、一次 Variable-length Prefill、每条请求取本轮最后 token 的 logits、批量 Scatter 新 state 和 Batch Sampling。

Decode 已实现 `[B]` 输入、`[B,1,H]` GDN 执行和 `[B,vocab_size]` logits。已验证：

```text
batch_vs_solo [True, True, True]
duplicate_equal True
```

### 2.8 Qwen3.5 Vision Tower

`nanovllm/models/qwen3_5_vision.py` 已实现：

- PatchEmbed、absolute position interpolation、Vision 2D RoPE。
- Vision Attention、MLP、Transformer Block。
- Patch Merger 和到文本 hidden size 的视觉投影。
- 完整 `Qwen3_5VisionModel`。

9B 视觉配置与实测：

```text
depth=27, hidden=1152, intermediate=4304
heads=16, head_dim=72
patch_size=16, temporal_patch_size=2
spatial_merge_size=2, output_hidden=4096

PatchEmbed:       [320,1536] → [320,1152]
Vision Attention: [320,1152], BF16, no NaN
Vision Block:     [320,1152], BF16, no NaN
VisionModel:      [320,1536] → [80,4096], BF16, no NaN
```

权重完整性：

```text
checkpoint vision weights = 333
local vision parameters   = 333
missing=0, unexpected=0, shape_mismatch=0
```

顶层模型已提供 `get_visual_embeddings(pixel_values, image_grid_thw)`。Vision 权重接入后，9B 纯文本三请求回归运行成功。

### 2.9 图文输入处理和 Sequence 基础字段

`nanovllm/inputs.py` 已实现：

- `PromptInput = str | list[int] | MultiModalPrompt`。
- `ProcessedPrompt` 和严格字段校验。
- 官方 `AutoProcessor`、官方 chat template、图片 resize/patch/special token 构造。
- 最多一张本地 PIL 图片，灰度/RGBA 转 RGB。
- 输出 `token_ids`、`mm_token_type_ids`、`pixel_values`、`image_grid_thw`。
- 校验 image placeholder 数、type=1 数和 Patch Merger 输出数一致。
- 等待期间图片 Tensor 保持在 CPU。

320×240 RGB 图片实测：

```text
token_ids length  = 96
pixel_values      = [320,1536], float32, CPU
image_grid_thw    = [[1,16,20]]
image token count = 80
mm types          = [0,1]
```

`LLMEngine` 已用 `InputProcessor` 创建请求；`Sequence` 已保存 `mm_token_type_ids/pixel_values/image_grid_thw`。Decode 追加 token 时同步追加文本 type 0；Prefill 序列化携带图像，Decode 不重复传整图。CPU Sequence 生命周期测试通过。

### 2.10 Multimodal RoPE 位置构造

`nanovllm/models/qwen3_5_mrope.py` 已实现：

- 按 `mm_token_type_ids` 分组文本/图像段。
- 文本使用三轴相同位置；图像使用 temporal/height/width 三轴位置。
- 使用 `spatial_merge_size` 得到合并后网格。
- 图像段位置空间前进 `max(H,W)`，而非 `H×W`。
- 计算 Decode 所需 `mrope_position_delta`。
- 明确拒绝首版不支持的视频。

`ProcessedPrompt` 已包含 `[3,L] mrope_position_ids` 和整数 `mrope_position_delta`。

2026-08-29 CPU 实测并与本地 Transformers 官方 `get_rope_index()` 一致：

```text
shape=[3,96], delta=-70, image_range=[4,83]
token 3  → [3,3,3]
token 4  → [4,4,4]
token 5  → [4,4,5]
token 83 → [4,11,13]
token 84 → [14,14,14]
token 95 → [25,25,25]
```

### 2.11 mRoPE 接入 Sequence 与模型执行

旧文档中记录的 Vision Part 9B 断点已完成：

- `Sequence` 已保存 `mrope_position_ids` 和 `mrope_position_delta`。
- `LLMEngine.add_request()` 已将 `ProcessedPrompt` 的 mRoPE 字段传入 `Sequence`。
- Decode 追加 token 时，使用 `old_num_tokens + mrope_position_delta` 追加三轴相同的文本位置。
- `prepare_prefill()` 已构造 packed `[3,total_tokens]` mRoPE positions。
- `prepare_decode()` 已根据每个 Sequence 的 token 索引和 delta 构造 `[3,B]` positions。
- `RotaryEmbedding`/Qwen3.5 Full Attention 已支持 interleaved multimodal RoPE。
- 抢占后从 token 0 重算时，token/type/position 元数据保持对齐。

### 2.12 Vision 执行、Embedding 替换和视觉缓存

`ModelRunner` 已接通完整图文 Prefill：

```text
pixel_values + image_grid_thw
        ↓
Qwen3.5 Vision Tower + Patch Merger
        ↓
visual_embeddings [num_visual_tokens, text_hidden_size]
        ↓ index_copy_
替换文本 embedding 中 image placeholder 的位置
        ↓
Qwen3.5 Hybrid Decoder
```

已实现：

- 纯文本请求使用 `input_ids`，图文 Prefill 使用替换后的 `inputs_embeds`。
- Variable-length Batched Prefill 中，每个图文 Sequence 只替换当前 chunk 覆盖的 image token 位置。
- visual embedding cache 以 `seq_id` 管理，避免 Chunked Prefill 重复执行 Vision Tower。
- 缓存统计包含 `visual_cache_bytes`/peak、forwards、hits 和 misses。
- 请求完成时释放视觉缓存。
- 请求被抢占时释放视觉缓存；恢复时重新执行 Vision Tower 和 Prefill。
- 文本、长文本和单图请求可以进入同一个 Variable-length Prefill microbatch。

### 2.13 图文正确性验证

已完成的端到端验证：

1. **9B 单图生成**
   - 自制红色正方形/蓝色圆形图片可正确生成描述。
   - 视觉缓存最终 `current bytes=0`，无泄漏。

2. **Chunked Prefill 视觉缓存**
   - 一次 Vision forward，后续 chunk 命中 cache。
   - 实测过 `vision forwards=1, cache misses=1, cache hits=1`。

3. **文本/图文混合批处理**
   - 3 条请求进入同一 Prefill microbatch。
   - `prefill microbatches=1, max prefill batch size=3, mixed prefill microbatches=1`。

4. **Full Prefill 与 Chunked Prefill 一致**
   - 两条路径的 greedy 生成 token 一致。

5. **Hugging Face 图文对齐**
   - Prompt 长度 124 tokens，Nano 与 HF prompt token IDs 完全一致。
   - 比较 64 个 greedy 生成 token，64/64 一致。

6. **抢占与确定性重算**
   - `preemptions=1`。
   - `recomputed tokens=124`。
   - 抢占前后生成 64 个图文 token 一致。
   - Vision Tower 执行 2 次：首次 Prefill 一次，抢占恢复重算一次。
   - 最终 KV blocks、GDN state slot 和 visual cache 全部释放。

### 2.14 请求级指标和 Benchmark 基础设施

`LLMEngine` 已实现 `RequestMetrics` 和 `StepStats`：

- preprocessing time。
- queue time。
- TTFT。
- 每个生成 token 的 `token_timestamps`。
- request-level TPOT。
- E2E latency。
- 每轮 Prefill/Decode token 数和执行时间。

已实现的 Benchmark 脚本：

```text
tests/benchmark_hybrid.py             固定工作负载 Benchmark
tests/run_benchmark_matrix.py         工作负载矩阵
tests/dynamic_scheduler_benchmark.py  长 Prefill 动态插入实验
tests/saturation_benchmark.py         持续闭环饱和压测
```

### 2.15 Decode-first 与 Prefill-first 实测

`Config.scheduler_policy` 已支持：

```text
decode_first
prefill_first
```

`prefill_first` 是严格对照基线：只要 waiting 中存在 Prefill，当轮就不执行 Decode。

#### 单次动态干扰实验

Token budget=2048 时：

```text
Prefill-first TPOT p95: 794.92 ms
Decode-first  TPOT p95: 403.76 ms
特定干扰窗口下降低约 49.2%

Prefill-first late TTFT p95: 721.74 ms
Decode-first  late TTFT p95: 762.44 ms
TTFT 退化约 5.6%
```

这是小样本、刻意构造的长 Prefill 干扰场景，**不能将 49.2% 表述为通用 p95 提升**。

#### 持续饱和压测：C8

96 条请求，并发 8，64 条 Decode-heavy + 16 条长文本 + 16 条图片：

```text
Decode-first output throughput: 332.59 tok/s
Prefill-first output throughput: 334.65 tok/s
吞吐变化: -0.62%

request TPOT p95: 25.21 → 23.60 ms，降低约 6.38%
TTFT p95:         570.31 → 626.42 ms，增加约 9.84%
max token stall:  607.43 → 308.17 ms，降低约 49.27%
```

#### 持续饱和压测：C12

96 条请求，并发 12，工作负载与 C8 相同：

```text
Decode-first output throughput: 425.47 tok/s
Prefill-first output throughput: 424.21 tok/s
吞吐变化: +0.30%

request TPOT p95: 26.49 → 26.32 ms，降低约 0.63%
TTFT p95:         871.13 → 842.58 ms，降低约 3.28%
max token stall:  1115.67 → 316.58 ms，降低约 71.62%
E2E p95:          6805.82 → 6777.22 ms，降低约 0.42%
```

C12 Decode-first 硬件状态：

```text
GPU utilization mean/p95: 90% / 100%
power mean/max/limit:      401 W / 514 W / 575 W
NVML memory used:          27.21 GiB
Torch peak allocated:      25.28 GiB
running/outstanding p50:   12 / 12
```

C12 属于软件工作队列持续饱和，但不是持续满 TDP。

需要同时记录的现象：

- Decode-first 将少数超大停顿拆成更多中等停顿。
- 因此 max stall 显著下降，但 C12 token-interval p99 从 24.51 ms 增加到 199.44 ms。
- 不能只用单个 p95/p99/max 描述调度效果，后续报告应同时说明整个延迟分布。
- C8/C12、两种调度策略的 96 条请求、17408 个生成 token 全部一致。
- 两种饱和实验均 `preemptions=0, recomputed_tokens=0, final visual cache bytes=0`。
- 上述性能数字是单次实验结果，未完成多次重复统计，暂不作为最终简历数字。

## 3. 当前完整执行链路

纯文本：

```text
prompt → tokenize → Sequence → Decode-first Scheduler
→ Variable-length Batched Prefill → KV/GDN state
→ Batched Decode → Sampling → decode text
```

图文：

```text
PIL Image + prompt
→ AutoProcessor
→ token_ids/pixel_values/grid/mm types
→ mRoPE positions/delta
→ ProcessedPrompt
→ LLMEngine/Sequence
→ Decode-first Scheduler
→ Variable-length Chunked Prefill
→ Vision Tower/Patch Merger
→ visual embedding cache
→ 替换 image placeholder embeddings
→ Qwen3.5 Hybrid Decoder
→ Full Attention Paged KV + GDN conv/recurrent state
→ Batched Decode + mRoPE delta
→ greedy/sampling
→ decode text
```

## 4. V2：GDN-aware Prefix State Cache（已完成）

### 4.1 为什么不能直接打开原 Prefix Cache

Qwen3.5 的历史状态由两部分组成：

```text
Full Attention 历史 → Paged KV blocks
Gated DeltaNet 历史 → conv_state + recurrent_state
```

如果请求 B 只命中请求 A 的 Attention KV，却没有恢复同一前缀边界的 GDN state，后续 Decode 会从错误历史继续，生成结果不可信。因此一次命中必须是联合、原子的：

```text
Prefix Hit
   ├── Full Attention KV 有效
   └── 所有 GDN state snapshot 有效
两者同时满足 → restore
任一缺失     → 整体 miss，重新 Prefill
```

### 4.2 首版范围

- 只支持纯文本共享前缀。
- 只在完整 token block 边界创建 Prefix Entry。
- 保存 KV block 引用以及对应的全部 GDN `conv_state/recurrent_state` snapshot。
- `conv_state` snapshot 保持模型 BF16 dtype；`recurrent_state` snapshot 支持 FP32 正确性模式和 BF16 压缩模式。
- BF16 只压缩休眠的 Prefix snapshot；恢复到 active state slot 时转回 FP32，后续 GDN 递推仍使用 FP32 recurrent state。
- 使用显存预算、checkpoint interval、引用计数和 LRU 控制开销。
- Prefix 命中后从该边界继续执行剩余 suffix Prefill。

图文 Prefix Cache 暂缓，因为缓存身份还必须包含图片内容、processor 结果、image grid、mRoPE layout 和视觉模型版本；只按 image placeholder token IDs 做 hash 会错误共享不同图片。

### 4.3 最终 checkpoint 与 Admission 策略

V2 不会为了每个 checkpoint 强制切断冷请求 Prefill，因为这会增加模型前向次数、Kernel Launch 和小 chunk 开销。最终采用 opportunistic commit：

- 只在 Prefill chunk 自然结束于完整 token block 边界时考虑提交。
- 边界还需满足 “prefix_checkpoint_interval_blocks” 的稀疏间隔。
- “checkpoint_tokens” 必须严格小于 “seq.num_prompt_tokens”，确保命中后至少执行一个 suffix token 重新产生 logits。
- 每个请求通过 “max_new_prefix_snapshots_per_request” 限制新建 Entry 数量。
- always 策略立即缓存合法边界；frequency 策略先在 “admission_history” 中观察 “PrefixKey”，达到 “prefix_admission_min_observations” 才分配 GPU snapshot 和 pin KV。
- CPU 候选表由 “prefix_admission_max_candidates” 限制，并使用独立 LRU；GPU resident Entry 则由容量预算和另一套 LRU 管理。

因此，最终的“热前缀”判断不是单看长度，而是同时满足：合法 checkpoint、重复 PrefixKey 达到观察阈值、当前 Entry 尚未 resident、单 Entry 可装入预算；具体变量和状态机见第 4.9～4.13 节。

### 4.4 Snapshot 精度模式

```text
FP32 mode
    conv snapshot: BF16
    recurrent snapshot: FP32
    GDN snapshot 约 49.5 MiB/Entry
    用于 Cache on/off 严格正确性基线

BF16 mode
    conv snapshot: BF16
    recurrent snapshot: BF16
    GDN snapshot 约 25.5 MiB/Entry
    恢复时转回 FP32 active recurrent state
```

BF16 模式约减少 48.5% 的 GDN snapshot 显存，但会在保存时引入一次精度损失，因此不预设逐 token 完全一致。必须单独报告 recurrent state/logits 最大与平均误差、top-1 一致率和 greedy token 一致率；若未通过正确性门槛，BF16 保持为实验模式，FP32 作为安全 fallback。

`model_namespace` 必须包含 model dtype、active recurrent dtype、snapshot recurrent dtype、block size 和 Prefix schema version，避免不同缓存布局误共享。

### 4.5 原规划 Part（现已全部实现）

1. Prefix Key：模型/配置身份、完整 block token hash 和前驱链。
2. Prefix Entry：联合持有 KV blocks 与 GDN snapshot。
3. snapshot commit：只在完整 block 边界原子提交。
4. lookup/restore：KV/GDN 同时命中才恢复到活跃 state slot。
5. 显存预算：`hybrid_prefix_cache_capacity_mib` 与 checkpoint interval。
6. 生命周期：引用计数、LRU、请求完成、抢占、失效和异常回滚。
7. 正确性：Cache on/off、Chunked Prefill、batch、抢占和淘汰后重算。
8. Benchmark：TTFT、跳过的 Prefill tokens、hit rate、snapshot 拷贝耗时和显存。

### 4.6 关键资源风险

9B 的单请求 GDN active state 约为 49.5 MiB。若每 256 tokens 保存一次 8K 前缀，粗略需要 32 份 checkpoint，单条前缀的 GDN snapshot 就可能约为：

```text
49.5 MiB × 32 ≈ 1.55 GiB
```

因此不能无界保存每个 block 的状态。V2 必须使用显存上限、稀疏 checkpoint、热度准入和 LRU，并把 snapshot 显存作为一等指标。BF16 snapshot 会缓解但不会消除该风险，KV blocks 仍然随前缀长度增长。

### 4.7 完成判定

- FP32 Cache on/off 的 greedy token 完全一致。
- FP32 命中恢复后的 KV 和全部 GDN state 与完整重算一致。
- BF16 单独报告 state/logits 误差、top-1 和 greedy token 一致率，不把压缩模式与 FP32 基线混为同一正确性结论。
- 实际 scheduled Prefill tokens 按命中长度减少。
- 引用、淘汰、抢占和请求完成后无 KV/state snapshot 泄漏。
- 报告冷 miss TTFT、热 hit TTFT、Prefill 吞吐、额外 Forward 次数、snapshot 保存/恢复成本、命中率与 FP32/BF16 显存代价。

### 4.8 最终实现总览

V2 最终实现的是纯文本 Qwen3.5 Hybrid Prefix Cache。一个可复用前缀不是单独的 Attention KV，而是以下两类历史状态组成的原子快照：

~~~text
同一 token 边界
├── Full Attention：物理 Paged KV block 引用
└── Gated DeltaNet：conv_state + recurrent_state snapshot
~~~

冷请求、提交和热命中的完整流程为：

~~~text
冷请求
tokenize → Waiting → lookup miss → Full/Chunked Prefill
→ KV block hash 完成 → 到达合法 checkpoint
→ Admission 判断 → snapshot GDN + pin KV → Prefix Entry resident

热请求
tokenize → Waiting → lookup_longest
→ 比较链式 Hash + 真实 block token IDs
→ 为命中 KV 增加 request owner → 写入 Sequence.block_table
→ 分配 state_slot → restore conv/recurrent snapshot
→ num_cached_tokens 跳到边界 → 只 Prefill suffix → Decode

淘汰/删除
Entry 从 LRU 删除 → unpin cache owner
→ 只有 block 总 ref_count 变成 0 时才回收到 free_block_ids
~~~

当前限制：只缓存纯文本前缀；图文请求不参与 Prefix Cache；Entry 边界必须按完整 “block_size” 对齐，并且严格小于 “num_prompt_tokens”，因为当前 Entry 不保存 checkpoint 边界的 next-token logits，命中后至少要真实计算一个 suffix token。

### 4.9 Config、Key 和 Entry 的关键变量

“nanovllm/config.py” 中与 V2 行为直接相关的变量：

- “hybrid_prefix_cache_mode”：disabled 关闭联合 Prefix Cache；opportunistic 只在 Prefill 自然结束于合法边界时创建 checkpoint，不为了冷请求强制额外切分模型前向。
- “prefix_checkpoint_interval_blocks”：两个可提交 checkpoint 之间相隔多少个 token block；checkpoint token 间隔等于该值乘 “block_size”。
- “prefix_recurrent_snapshot_dtype”：休眠 recurrent snapshot 使用 float32 或 bfloat16；恢复到 active slot 时仍转回 FP32。
- “max_new_prefix_snapshots_per_request”：一个请求最多创建多少个新 Entry，防止单个超长请求保存过密 checkpoint。
- “hybrid_prefix_cache_capacity_mib”：KV pin 容量与 GDN snapshot 容量合计的 GPU 显存上限。
- “prefix_admission_policy”：always 表示合法 checkpoint 立即提交；frequency 表示重复观察达到阈值后才提交。
- “prefix_admission_min_observations”：frequency 策略的最小观察次数；当前实验为两次观察后提交，所以第一次只记录候选，第二次完整 Prefill 后创建 Entry，第三次请求开始命中。
- “prefix_admission_max_candidates”：CPU Admission 候选表最大条目数；超限时淘汰最久未观察的候选，避免只换掉 GPU 污染而让 CPU 元数据无界增长。

“PrefixKey” 是 Entry 的缓存身份，关键字段为：

- “model_namespace”：隔离模型、dtype、block size、snapshot schema 等不兼容配置，避免不同模型或布局错误共享。
- “block_hash”：到当前边界为止的链式 token block Hash；它已经包含此前所有完整 block 的历史。
- “num_cached_tokens”：该 Entry 对应的 checkpoint token 边界。

“PrefixStateEntry” 是真正 resident 的联合缓存对象，关键字段为：

- “key”：上述 PrefixKey。
- “kv_block_ids”：按逻辑前缀顺序保存的物理 Paged KV block ID。
- “conv_state_snapshot”：全部 GDN 层在该 token 边界的短卷积历史。
- “recurrent_state_snapshot”：全部 GDN 层在该边界的矩阵状态；休眠时可为 FP32/BF16。
- “gdn_snapshot_bytes”：本 Entry 两类 GDN snapshot 的显存字节数。

Entry 不是 token 内容的唯一真相。命中时除 “block_hash” 外，还会调用 “prefix_blocks_have_same_tokens()” 对比每个 block 的真实 “token_ids”，从而把 Hash 只当快速索引，避免碰撞导致错误命中。

### 4.10 BlockManager：KV 所有权和共享物理块

每个物理 “Block” 的关键变量：

- “ref_count”：请求 owner 与 Prefix Cache owner 的总引用数。
- “cache_ref_count”：当前有多少 resident Prefix Entry pin 住该物理块。
- “request_ref_count = ref_count - cache_ref_count”：当前有多少活跃 Sequence 正在使用该物理块。
- “hash”：当前完整 token block 的链式 Hash。
- “token_ids”：用于碰撞保护的真实 block token 内容。

关键方法及职责：

- “pin_blocks(kv_block_ids)”：Entry commit 时增加 “cache_ref_count/ref_count”，让请求结束后 KV 仍驻留。
- “unpin_blocks(kv_block_ids)”：Entry discard/evict 时释放 cache owner；只有总 “ref_count == 0” 才真正回收物理块。
- “validate_prefix_blocks(...)”：验证 block ID、有序链式 Hash、token 数和 owner 状态。
- “prefix_blocks_have_same_tokens(...)”：逐 block 比较真实 token，防止 Hash collision。
- “prefix_metadata_at_boundary(...)”：从 Sequence 的 “block_table” 和逻辑 token blocks 取得 checkpoint 对应的 block IDs/Hash/token 元数据。
- “can_allocate_from_prefix(...)”：在不修改状态前检查命中后剩余 suffix 所需 KV 资源。
- “allocate_from_prefix(...)”：为命中块增加 request owner，并将物理 block IDs 安装到新 Sequence 的 “block_table”。

共享层级前缀示例：Entry A 保存 4096 tokens、16 个 KV blocks；Entry B 保存同一前缀扩展到 8192 tokens、32 个 KV blocks。B 的前 16 个物理块与 A 共享，因此不能把两个 Entry 的 16+32 个块都当独立容量。实测：

~~~text
Entry A：16 KV blocks + 25.5 MiB GDN = 153.5 MiB
Entry B：32 KV blocks + 25.5 MiB GDN
唯一 pinned KV blocks：32
正确总容量：256 MiB KV + 51 MiB GDN = 307 MiB
错误的逐 Entry 重复相加：435 MiB
~~~

删除 A 时，A/B 共享的前 16 个 KV blocks 仍被 B pin 住，因此只回收 A 的 25.5 MiB GDN snapshot；剩余容量为 281.5 MiB。

### 4.11 Sequence、Scheduler 和 Engine 的命中状态机

Sequence 上与 Prefix Cache 相关的关键变量：

- “prefix_lookup_completed”：保证同一轮 Waiting 生命周期只做一次 lookup，避免每次 schedule 重复累计 lookup/miss 指标。
- “prefix_cache_key”：保存实际命中的 Entry Key，后续 restore 使用这个稳定身份，而不是再次搜索可能已经变化的 LRU。
- “prefix_restore_pending”：KV 已 attach、state slot 已分配，但 GDN snapshot 还需在 suffix Prefill 前恢复。
- “num_prefix_hit_tokens”：本请求跳过的 Prompt tokens 数量。
- “num_prefix_snapshots_created”：本请求已创建的 checkpoint 数，用于执行每请求上限。
- “block_table”：逻辑 token block 到物理 KV block 的映射；命中时前缀块直接安装进来。
- “num_cached_tokens”：已经拥有正确 KV/GDN 历史、无需重算的 token 数；命中后直接推进到 checkpoint 边界。
- “num_scheduled_tokens”：本轮实际交给模型计算的 suffix token 数。
- “state_slot”：该请求在 active GDN state pool 中的槽位。

Scheduler 的 “_lookup_prefix_for_new_request()” 在新请求进入 Running 前执行：

~~~text
Waiting Sequence
→ lookup_longest(token_ids)
→ can_allocate_from_prefix()
→ allocate_from_prefix() 增加 KV request owner
→ 分配 GDN state_slot
→ prefix_restore_pending = True
→ num_cached_tokens = 命中边界
→ 本轮 num_scheduled_tokens 只计算 suffix
~~~

Scheduler 还累计：

- “num_prefix_hit_requests”：发生联合命中的请求数量。
- “num_prefix_hit_tokens”：所有命中请求实际跳过的 Prefill tokens 总数。

Engine 的 “_restore_pending_prefix_states()” 在真正执行 suffix Prefill 之前再次校验 “prefix_cache_key”、命中边界和 “block_table”，然后把 “conv_state_snapshot/recurrent_state_snapshot” 复制进 “state_slot”。如果休眠 recurrent 是 BF16，复制时转换为 active FP32；之后的 GDN 递推仍在 FP32 state 上进行。

Engine 的 “_commit_opportunistic_prefixes()” 位于模型前向与 Scheduler postprocess/free 之间。此时：

- Full Attention 已把本轮 K/V 写入 Paged KV。
- GDN active state 已更新到本轮末尾。
- 完整 block Hash 已可用。
- 请求 KV owner 和 “state_slot” 尚未释放。

提交边界使用：

~~~python
checkpoint_tokens = seq.num_cached_tokens + seq.num_scheduled_tokens
~~~

随后检查纯文本、完整 block 对齐、checkpoint interval、“checkpoint_tokens < seq.num_prompt_tokens” 和 “num_prefix_snapshots_created” 上限；通过 Admission 后才 clone GDN state、pin KV 并插入 Entry。

抢占时会释放请求持有的 KV/state slot，并重置该 Sequence 的命中恢复元数据；恢复依靠确定性 lookup 或从 token 0 重算，不保留指向已失效 active slot 的引用。

### 4.12 PrefixStateCache：LRU、容量和统计变量

“PrefixStateCache” 的关键容器和容量变量：

- “entries: OrderedDict[PrefixKey, PrefixStateEntry]”：GPU resident Entry 的 LRU；左侧是 LRU，右侧是 MRU。
- “admission_history: OrderedDict[PrefixKey, int]”：独立的 CPU 候选 LRU，value 是观察次数；它不占 GDN/KV GPU 容量。
- “capacity_bytes”：Prefix Cache 可使用的总 GPU 字节上限。
- “kv_block_bytes”：单个物理 Paged KV block 的字节数。
- “estimated_gdn_snapshot_bytes_per_entry”：提交前用于判断单 Entry 是否可能装入预算的 GDN 估算值。
- “current_gdn_snapshot_bytes”：所有 resident Entry 的 GDN snapshot 实际总字节数。
- “pinned_kv_block_ids”：所有 Entry pin 住的物理 KV block 去重集合。
- “num_unique_pinned_kv_blocks”：上述集合大小。
- “current_pinned_kv_capacity_bytes”：唯一 pinned KV blocks 数乘 “kv_block_bytes”。
- “current_prefix_cache_capacity_bytes”：前两者之和。
- “remaining_capacity_bytes”：总预算减当前使用容量。
- “capacity_utilization”：当前使用容量除以总容量。

关键容量方法：

- “_additional_capacity_bytes(kv_block_ids)”：只计算候选中 “cache_ref_count == 0” 的新增物理 KV blocks，再加一份 GDN snapshot；共享块不会重复收费。
- “reclaimable_capacity_bytes(entry)”：只统计删除该 Entry 后真正失去最后一个 cache owner 的 KV blocks，再加该 Entry 的 GDN snapshot。
- “commit()”：先做单候选 oversize 检查；若投影容量超限，循环淘汰 LRU，并在每次淘汰后重新计算共享关系；最后再原子插入。
- “lookup_longest()”：按最长合法 block 边界查找，命中后将 Entry touch 到 MRU。
- “restore_gdn_state()”：确认 Entry 仍 resident，再恢复 active slot。
- “discard()”：从索引移除 Entry，释放 GDN snapshot 引用并 unpin KV。

运行统计变量：

- “num_commits”、 “num_duplicate_commits”：新提交与重复提交次数。
- “num_lookups”、 “num_hits”、 “num_misses”：lookup 行为。
- “num_hash_collisions”：Hash 命中但真实 token 不一致的次数。
- “num_gdn_restores”：联合命中后完成 GDN state restore 的次数。
- “num_lru_touches”：命中或复用导致 Entry 移到 MRU 的次数。
- “num_evictions”、 “total_evicted_capacity_bytes”：容量淘汰次数与实际释放字节。
- “num_capacity_rejections”：单个候选本身超过总预算而被拒绝的次数。

### 4.13 Frequency Admission：如何判断热前缀

最终实现没有通过“长度够长就一定缓存”来判断热前缀。长度和完整 block 对齐只是合法性条件；热度由同一个 “PrefixKey” 被独立请求重复观察的次数决定。

“observe_and_should_admit(key)” 的关键行为：

~~~text
policy = always
→ 合法 checkpoint 直接允许 commit

policy = frequency, min_observations = 2
第 1 个请求到达边界：admission_history[key] = 1，defer
第 2 个请求到达边界：计数变 2，accept 并 commit
第 3 个请求：lookup resident Entry，直接 hit
~~~

相关变量和统计：

- “num_admission_candidates”：CPU 候选表当前大小。
- “admission_keys_lru_to_mru”：调试时查看候选从最旧到最新的顺序。
- “admission_observation_count(key)”：读取某个 PrefixKey 当前观察次数。
- “num_admission_observations”：总观察次数。
- “num_admission_accepts”：达到阈值并允许提交的次数。
- “num_admission_deferrals”：尚未达到阈值而推迟提交的次数。
- “num_admission_candidate_evictions”：CPU 候选表超限时的 LRU 淘汰数。
- “num_admission_hit_touches”：resident hit 对候选/热度状态的触碰次数。
- “record_admission_hit()”：命中 resident Entry 后更新热度相关状态，但不会重复创建 snapshot。

这种策略适合 Agent 系统提示词：固定 system prompt/tool schema 会在多个请求中重复出现，第二次观察后可驻留；随机的一次性长输入只留下小型 CPU 候选记录，不立即消耗几十到几百 MiB GPU Prefix Cache。

### 4.14 正确性、精度和压力测试结果

完整回归入口最终输出：

~~~text
ALL CURRENT PREFIX CACHE TESTS PASSED
~~~

覆盖的测试包括：

- “tests/test_prefix_commit.py”：原子 commit、snapshot 独立性、重复 Entry 去重和释放。
- “tests/test_prefix_hit.py”：联合 KV/GDN longest hit、suffix Prefill token 扣减和 greedy 一致性。
- “tests/test_prefix_admission.py”：两次观察准入、候选 LRU 上限和统计。
- “tests/prefix_dtype_probe.py”：FP32/BF16 recurrent snapshot 显存与 token 对比。
- “tests/prefix_stress.py”：4K/8K/16K 长前缀的 cold/hot 性能与一致性。
- “tests/test_prefix_lru_order.py”：命中后 LRU→MRU 顺序更新。
- “tests/test_prefix_lru_eviction.py”：预算不足时按 LRU 淘汰。
- “tests/test_prefix_capacity_rejection.py”：单 Entry 超预算时拒绝且不破坏已有缓存。
- “tests/test_prefix_shared_blocks.py”：层级 Entry 共享物理 KV、去重容量核算和最长前缀恢复。
- “tests/test_prefix_remaining.py”：active request ownership、shared LRU、Hash collision guard 和全量回归编排。

FP32/BF16 snapshot 实测：

~~~text
conv snapshot：                 1.5 MiB
recurrent snapshot FP32：      48.0 MiB
recurrent snapshot BF16：      24.0 MiB
GDN Entry FP32 总计：          49.5 MiB
GDN Entry BF16 总计：          25.5 MiB
GDN snapshot 显存降低：        48.5%
当前 64-token greedy 对比：    FP32 64/64，BF16 64/64
~~~

这里的 64/64 是当前测试输入上的实测，不代表所有输入都保证 BF16 与 FP32 逐 token 相同。BF16 在 snapshot 保存时会量化 recurrent state，正确表述是“当前用例完全一致，并保留 FP32 fallback”。

4K/8K/16K BF16 长前缀压力测试：

| 场景 | Prompt | Checkpoint | 实际 suffix Prefill | Cache 显存 | Cold E2E | Hot median E2E | 降低 | Greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 4609 | 4096 | 513 | 153.5 MiB | 3101.123 ms | 2417.572 ms | 22.04% | 128/128 |
| 8K | 8704 | 8192 | 512 | 281.5 MiB | 3439.454 ms | 2366.322 ms | 31.20% | 128/128 |
| 16K | 16915 | 16384 | 531 | 537.5 MiB | 4227.161 ms | 2393.907 ms | 43.37% | 128/128 |

Qwen3.5-9B 中一个 KV block 实测为 8 MiB，计算为：

~~~text
2(K/V) × 8 个 Full Attention 层 × block_size 256
× 4 个 KV heads × head_dim 256 × BF16 2 bytes
= 8 MiB
~~~

### 4.15 Admission 污染压测与结论

“tests/benchmark_prefix_admission.py” 构造受控 Agent 场景：GPU Prefix Cache 预算 320 MiB；一个会再次使用的 4K hot system prefix；中间插入六个只出现一次的 4K 长前缀；输出长度固定为 1 token；always 与 frequency 交替顺序各重复三次，并排除模型 warmup。

关键结果：

| 指标 | always | frequency |
|---|---:|---:|
| 污染后 hot probe hit rate | 0.5 | 1.0 |
| 污染阶段 commits / evictions | 6 / 5 | 0 / 0 |
| 污染后 hot Entry | 被淘汰 | 保持 resident |
| hot probe 实际 Prefill tokens | 4618 | 522 |
| hot probe median latency | 426.599 ms | 68.431 ms |
| 污染后 Prefix Cache 容量 | 307 MiB / 32 unique blocks | 153.5 MiB / 16 unique blocks |
| Admission deferrals | 0 | 6 |

在这个特定受控工作负载中，frequency 相对 always：

~~~text
跳过 Prefill tokens：4096
Prefill token 减少：4618 → 522，约 88.7%
hot probe latency 降低：约 83.96%
对应 speedup：约 6.23×
~~~

污染请求本身的总耗时几乎相同：always 约 2553～2559 ms，frequency 约 2559～2561 ms，因此当前实验没有观察到显著 Admission CPU 开销。

83.96% 不能写成通用性能提升：这是单 token 输出、4K 重复前缀、320 MiB 小预算和对抗性 one-hit burst 下的结果。Frequency 的代价是前两次仍需完整 Prefill，到第三次才真正命中；真实收益取决于重复率、前缀长度、输出长度、显存预算和并发模式。原始数据位于：

~~~text
artifacts/prefix_regression/*.json
artifacts/prefix_admission_benchmark/always_0.json ... always_2.json
artifacts/prefix_admission_benchmark/frequency_0.json ... frequency_2.json
artifacts/prefix_admission_benchmark/summary.json
~~~

## 5. V3：Hybrid Decode CUDA Graph（已纳入，未实现）

### 5.1 当前已有基础

- 原 nano-vLLM 在 `ModelRunner.capture_cudagraph()` 中有普通 Attention 模型的 CUDA Graph 骨架。
- 当前 Qwen3.5 Hybrid Runtime 已有稳定的 Batched Decode、紧凑 Paged KV、GDN state-slot Gather/Scatter 和三轴 mRoPE。
- 当前 Qwen3.5 Hybrid 路径仍要求 `enforce_eager=True`；旧 Graph 骨架不能直接证明 Hybrid Decode 可 capture。

### 5.2 技术目标

只 capture 高频、shape 相对稳定的 Decode 路径：

```text
Vision + Prefill：继续 Eager
Decode：
真实请求数据 → 静态输入 buffer → CUDA Graph replay
                           ↓ 不支持/失败
                       Eager fallback
```

必须纳入静态输入的主要数据：

```text
input_ids[B]
positions[3,B]
slot_mapping
context_lens
block_tables
state_slot_ids
```

Full Attention 通过 `block_tables/slot_mapping` 读写正确 KV block；GDN 通过 `state_slot_ids` 读写正确 conv/recurrent state。Graph 不能把这些地址或请求映射固化成上一轮的数据。

### 5.3 待实现 Part

1. Graph-safe 审计：定位动态分配、CPU 同步和数据相关分支。
2. 静态 Decode buffer：覆盖 token、三轴位置、KV 元数据和 GDN slot。
3. batch bucket：首版计划 `1/2/4/8/12`，其余形状回退 Eager。
4. Hybrid capture/replay：验证 FLA recurrent、causal-conv1d 和 Full Attention。
5. 正确性：Eager/Graph greedy token、padding、重复 replay 和状态串槽测试。
6. Benchmark：TPOT、Decode tokens/s、CPU launch overhead 和额外显存。

### 5.4 完成判定

只有满足以下条件才可以声称“实现 Qwen3.5 Hybrid CUDA Graph”：

- 至少一个真实 Qwen3.5 Hybrid Decode bucket 完成 capture 和稳定 replay。
- Eager/Graph greedy token 完全一致。
- 多请求、多轮 replay 后 KV block 与 GDN state slot 无串写。
- 不支持的 shape 能自动、安全回退 Eager。
- 有原始 Benchmark 和 Graph 额外显存数据。

## 6. V4：状态感知 GDN Decode 融合算子（已纳入，未实现）

### 6.1 项目定位

V4 不是孤立复现一个教学 Kernel，而是在 V2 Prefix restore 和 V3 CUDA Graph 接口稳定后，根据真实 Decode Profile 优化 GDN 状态访问。候选热点包括：

```text
GDN state Gather/Scatter
Depthwise causal-conv1d 单 token update
FLA fused recurrent update
FP32 recurrent state 的重复 HBM 往返
多个小 Kernel 的 Launch 开销
```

只有 Nsight Systems/PyTorch Profiler 证明其中至少一项是显著瓶颈后，才冻结最终算子边界。

### 6.2 首版范围与主要变量

只支持 `L=1` Decode、动态 batch 和任意不连续的：

```text
state_slot_ids[B]
```

第一子算子直接更新：

```text
mixed_qkv[B,conv_dim]
conv_state_pool[num_slots,num_gdn_layers,conv_dim,kernel_size]
```

第二子算子接收：

```text
q/k[B,H,K]、v[B,H,V]、beta/g
recurrent_state_pool[num_slots,num_gdn_layers,H,K,V] FP32
```

并融合执行：

```text
S = exp(g) · S
prediction = k^T S
delta = beta · (v - prediction)
S = S + k ⊗ delta
o = q^T S
```

Kernel 必须通过 `state_slot_ids[row]` 直接定位真实状态池并原地写回，不能先 Gather 整个 FP32 state 再 Scatter。

### 6.3 待实现 Part

1. Eager/Graph Decode 热点 Profile 与算子输入契约。
2. `torch/fla/custom` 微基准框架。
3. Naive state-aware causal-conv1d update。
4. Naive state-aware recurrent update/read。
5. `[K,V]` tiling、向量化、访存与融合优化。
6. HybridStateManager/ModelRunner 接入和 FLA fallback。
7. Prefix restore、slot 复用和 CUDA Graph custom replay 正确性。
8. Nsight Compute、微基准和端到端 Benchmark。

### 6.4 完成判定

- `torch/fla/custom` 的输出、conv state 和 FP32 recurrent state 对齐。
- 连续、离散、乱序 slot 和动态 batch 无状态串写。
- Prefix restore 后首次 Decode、抢占重算和 slot 复用保持正确。
- custom backend 可被 V3 CUDA Graph 正确 capture/replay，或对不安全形状明确 fallback。
- 同时报告 Kernel latency、HBM/L2/occupancy 与 TPOT/Decode 吞吐，不能把微基准加速直接写成端到端收益。
- 如果 custom 未超过 FLA，如实记录原因和适用边界，不宣称优化成功。

## 7. 当前开发断点：V2 Prefix Cache 已完成，下一步进入 V3 CUDA Graph

### 7.1 V2 最终完成状态

纯文本 GDN-aware Prefix Cache 已完成以下闭环：

- PrefixKey/PrefixStateEntry：联合表示同一 token 边界的 Full Attention KV 与 GDN snapshot。
- 写路径：自然 Chunked Prefill 边界进行 Admission、snapshot、KV pin 和原子 commit。
- 读路径：longest lookup、真实 token collision guard、KV request-owner attach、GDN restore 和 suffix-only Prefill。
- 精度：支持 FP32/BF16 recurrent snapshot，active recurrent state 保持 FP32。
- 生命周期：请求/cache 双 owner、active request ownership、抢占重算、discard 和异常回滚。
- 容量：全局 MiB budget、oversize rejection、共享 KV 去重计费和 LRU eviction。
- 热度准入：always/frequency 两种策略、有界 CPU candidate LRU 和两次观察阈值。
- 验证：完整回归、4K/8K/16K 压力测试、层级共享块测试和 Admission 污染 Benchmark。

关键完成信号：

~~~text
ALL CURRENT PREFIX CACHE TESTS PASSED
~~~

### 7.2 V2 复现入口

完整 Prefix Cache 回归：

~~~bash
cd /workspace/nano-vllm
source .venv/bin/activate
python tests/test_prefix_remaining.py --full
~~~

Admission 污染实验：

~~~bash
python tests/benchmark_prefix_admission.py
~~~

原始数据：

~~~text
artifacts/prefix_regression/
artifacts/prefix_admission_benchmark/
~~~

如果后续修改 Scheduler、BlockManager、HybridStateManager 或 Prefix Cache，必须先重新跑这套回归，重点观察：

- “num_prefix_hit_tokens” 是否等于 checkpoint 边界。
- 热命中的 “num_scheduled_tokens” 是否只包含 suffix。
- “num_gdn_restores” 是否随真正命中增长。
- “request_ref_count/cache_ref_count” 是否在请求结束和 Entry 淘汰后归零。
- “current_prefix_cache_capacity_bytes” 是否等于唯一 pinned KV 容量加 GDN snapshot 容量。
- “num_admission_deferrals/accepts” 是否符合两次观察策略。
- Cache on/off 或 cold/hot greedy token 是否满足对应精度模式的正确性要求。

### 7.3 下一步：V3 Hybrid Decode CUDA Graph

下一步不再继续扩展 Prefix Cache 功能，而是先冻结 V2 行为，进入 CUDA Graph 的 Graph-safe 审计。第一小步只做只读分析和最小实验：

1. 保持 “enforce_eager=True” 跑通当前 Qwen3.5 Hybrid Decode 基线。
2. 从 “ModelRunner.capture_cudagraph()” 和当前 “run_hybrid_decode()” 对照调用链。
3. 标出 Decode 中的动态 Tensor 分配、Python 分支、CPU-GPU 同步和 shape 变化。
4. 明确需要静态化的关键输入：“input_ids”、“positions”、“slot_mapping”、“context_lens”、“block_tables”、“state_slot_ids”。
5. 验证 FLA recurrent update、causal-conv1d 和 Paged Attention 是否能被 capture；失败路径保留 Eager fallback。
6. 先支持 batch bucket 1，再逐步扩展到 2/4/8/12，不在第一步同时处理所有动态 shape。

## 8. 剩余 Part

V2 Prefix State Cache 的核心实现和实验已经结束。剩余工作按以下顺序：

~~~text
V3 Hybrid Decode CUDA Graph             6 个大 Part
V4 状态感知 GDN Decode 自定义算子       8 个大 Part
V1/V2 文档、报告、简历与 Git 收尾       4 个大 Part
--------------------------------------------------
总计                                    18 个大 Part
~~~

如果只计算核心技术实现，不计最后的文档/Git 收尾，则还剩约 14 个大 Part。

V3 的 6 个 Part：

1. Graph-safe 调用链与算子兼容性审计。
2. 静态 Decode 输入/输出 buffer。
3. batch bucket、padding 与 Eager fallback。
4. Hybrid capture/replay 接入。
5. 多请求、多轮 replay、KV/GDN state 串槽正确性。
6. TPOT、Decode throughput、CPU launch overhead 与额外显存 Benchmark。

V4 的 8 个 Part：

1. Nsight Systems/PyTorch Profiler 定位真实 Decode 热点。
2. 固定 custom operator 输入契约。
3. torch/fla/custom 微基准。
4. state-aware single-token causal-conv update。
5. state-aware recurrent update/read。
6. tiling、向量化、融合与访存优化。
7. ModelRunner/HybridStateManager/CUDA Graph 接入和 fallback。
8. 状态级正确性、Kernel 指标和端到端 Benchmark。

当前正式顺序不变：

~~~text
V2 Prefix Cache（已完成）
→ V3 Hybrid Decode CUDA Graph
→ V4 状态感知 GDN Decode 算子
~~~

MTP、MoE、TP>1、多图、视频、图文 Prefix Cache 和完整 Chunk Gated Delta Rule 自研 Kernel 不计入上述 18 个 Part。

## 9. 当前可以和不能在面试中声称的内容

可以准确表述 V1：

> 基于 nano-vLLM 实现 Qwen3.5-9B 文本/单图 Hybrid Runtime：根据 layer_types 为 Full Attention 分配紧凑 Paged KV Cache，为 Gated DeltaNet 设计 state-slot 池管理 depthwise causal-convolution state 和 FP32 recurrent state；实现 FLA GDN Prefill/Decode、Variable-length Batched Prefill、Batched Decode、Chunked Prefill、Continuous Batching、Decode-first 双 microbatch 调度、Prefill 饥饿保护及 KV/GDN/视觉状态联合抢占重算。接入 Qwen3.5 Vision Transformer/Patch Merger、图像 embedding 替换、三轴 mRoPE 和请求级 visual embedding cache，并完成 HF 64-token greedy 对齐、文本/图文混合批处理、Chunked Prefill 和抢占重算一致性验证。

可以准确表述 V2：

> 为 Qwen3.5 的 Full Attention + Gated DeltaNet 混合状态实现纯文本 GDN-aware Prefix Cache：以 PrefixKey(model_namespace、链式 block_hash、num_cached_tokens) 标识完整 block checkpoint，PrefixStateEntry 联合持有物理 kv_block_ids、conv_state_snapshot 和 recurrent_state_snapshot；通过 request_ref_count/cache_ref_count 管理共享 KV 生命周期，通过 BF16 休眠 snapshot 将单 Entry 的 GDN 状态由 49.5 MiB 降至 25.5 MiB，通过唯一物理块计费、全局显存预算和 LRU 支持层级前缀共享与淘汰，并使用有界 CPU admission_history 对重复 PrefixKey 做 frequency admission，避免一次性长前缀污染 GPU 缓存。命中时 longest lookup 后安装 KV request owner、恢复 GDN state_slot、推进 num_cached_tokens，仅执行 suffix Prefill。

可以准确引用、但必须带工作负载限定的数字：

- 4K/8K/16K BF16 checkpoint 的当前压力测试中，热命中 E2E 分别降低 22.04%、31.20%、43.37%，128-token greedy 均为 128/128。
- 在“320 MiB 预算、一个 4K hot prefix、六个 one-hit 4K 前缀、输出 1 token”的受控污染实验中，frequency 将 hot probe Prefill 从 4618 tokens 降至 522 tokens，median latency 从 426.599 ms 降至 68.431 ms；这是特定工作负载结果，不是通用提升。
- BF16 休眠 GDN snapshot 从 49.5 MiB 降至 25.5 MiB，降幅 48.5%；当前测试为 64/64 greedy 一致，但不保证所有输入逐 token 一致。

目前不能声称：

- 已支持图文 Prefix Cache；当前 V2 只支持纯文本，图片身份、processor/mRoPE layout 和视觉状态尚未纳入 Key。
- 已实现 Hybrid CUDA Graph 或自研 GDN Decode Kernel；两者分别是下一阶段 V3/V4。
- 已实现 MTP 或 MoE。
- 已支持多图、视频或 TP>1。
- 已达到生产级 vLLM 的完整功能、稳定性和通用性能。
- Prefix Cache 在所有负载上都有 83.96% 提升；该数字仅来自明确限定的污染实验。
- BF16 snapshot 在所有 Prompt 上与 FP32 严格逐 token 等价。
- 当前 GDN Kernel 是完全自研；高性能路径仍使用 FLA/causal-conv1d，V4 才计划基于真实 Profile 开发状态感知算子。

完成 V3 后才可以增加：

> 为 Qwen3.5 Hybrid Decode 构建 batch-bucket CUDA Graph，通过静态三轴位置、Paged KV 元数据和 GDN state-slot buffer 完成 capture/replay，并为动态形状保留 Eager fallback；性能数字只填写实测结果。

完成 V4 后才可以增加：

> 基于 Nsight 定位 Qwen3.5 Decode 的 GDN 状态访问热点，实现支持离散 state_slot_ids 的状态感知融合 Kernel，原地更新 causal-conv state 与 FP32 recurrent state，并与 FLA 完成状态级对齐；性能数字同时报告微基准和端到端实测结果。

## 10. 关键文件地图

~~~text
nanovllm/config.py                   模型/资源、Prefix budget、snapshot dtype 和 Admission 参数
nanovllm/models/registry.py          Qwen3/Qwen3.5 自动选择
nanovllm/models/qwen3_5.py           Hybrid Decoder 和 Vision 顶层接口
nanovllm/models/qwen3_5_vision.py    Vision Transformer/Patch Merger
nanovllm/models/qwen3_5_mrope.py     LLM 三轴位置 IDs 和 delta
nanovllm/layers/gated_delta_net.py   GDN、短卷积、delta rule、FLA
nanovllm/layers/rotary_embedding.py  Partial RoPE 和 interleaved mRoPE
nanovllm/inputs.py                   AutoProcessor 和图文输入元数据
nanovllm/engine/sequence.py          token、block_table、state_slot 和 Prefix 命中元数据
nanovllm/engine/hybrid_state.py      GDN active pool、snapshot_slot/restore_slot
nanovllm/engine/block_manager.py     Paged KV、双 owner、Prefix attach/pin/unpin
nanovllm/engine/prefix_cache.py      Key/Entry、lookup/commit、LRU、capacity、Admission
nanovllm/engine/scheduler.py         Decode-first、Chunked Prefill、Prefix lookup/attach
nanovllm/engine/model_runner.py      Packed Tensor、GPU 前向和 Cache 读写
nanovllm/engine/llm_engine.py        输入、双 microbatch、Prefix restore/commit、采样和 decode
~~~

Prefix Cache 正确性与性能文件：

~~~text
tests/test_prefix_commit.py             commit、snapshot independence、duplicate/discard
tests/test_prefix_hit.py                联合 hit、suffix-only Prefill、greedy 对齐
tests/test_prefix_admission.py          frequency threshold 和 candidate LRU
tests/prefix_dtype_probe.py             FP32/BF16 snapshot
tests/prefix_stress.py                  4K/8K/16K 压力测试
tests/test_prefix_lru_order.py          LRU touch 顺序
tests/test_prefix_lru_eviction.py       容量淘汰
tests/test_prefix_capacity_rejection.py oversize rejection
tests/test_prefix_shared_blocks.py      层级 Entry 与共享物理 KV
tests/test_prefix_remaining.py          ownership/collision/full regression
tests/benchmark_prefix_admission.py     always/frequency 污染压测
~~~

V3 预计主要修改：

~~~text
nanovllm/config.py                   Graph bucket、fallback 和开关配置
nanovllm/engine/model_runner.py      静态 Decode buffer、capture/replay 和统计
nanovllm/utils/context.py            Graph replay 所需静态 Context 元数据
tests/                               Eager/Graph 对齐、bucket 和 Benchmark
~~~

V4 预计主要修改/新增：

~~~text
nanovllm/layers/gated_delta_net.py   torch/fla/custom backend 和调用边界
nanovllm/engine/hybrid_state.py      state pool 直接访问接口与 slot 校验
nanovllm/engine/model_runner.py      custom backend、Graph capture 和 fallback
nanovllm/kernels/                    state-aware conv/recurrent CUDA 扩展
tests/                               状态级对齐、动态 slot、微基准和端到端 Benchmark
~~~

## 11. Git 工作区与恢复入口

2026-09-01 本次核对时，最近提交为 “1883065 trying to achieve prefix cache”。Prefix Cache 源码、测试和实验产物仍包含未提交或新增文件；本次仅更新进度记忆，没有修改这些源码和测试。

恢复命令：

~~~bash
cd /workspace/nano-vllm
source .venv/bin/activate
git status --short
~~~

正式提交前应按类别检查：

1. Prefix Cache 源码：config、sequence、hybrid_state、block_manager、scheduler、llm_engine、prefix_cache。
2. 正确性测试：test_prefix_commit/hit/admission/LRU/capacity/shared/remaining。
3. Benchmark：prefix_dtype_probe、prefix_stress、benchmark_prefix_admission。
4. 小型 JSON/summary 是否需要提交；模型权重、大型 profiler 和临时产物不要提交。
5. 使用 “git diff --check” 检查空白错误，再分别暂存源码、测试、文档和必要结果。

恢复开发时从本文第 7.3 节开始：先做 V3 Graph-safe 审计，当前 Qwen3.5 Hybrid 路径继续保持 “enforce_eager=True”，确认 Eager 正确性基线后再开始 capture/replay。
