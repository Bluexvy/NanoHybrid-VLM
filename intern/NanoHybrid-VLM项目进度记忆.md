# NanoHybrid-VLM 项目进度记忆

> 更新时间：2026-08-29  
> 当前分支：`main`  
> 最近提交：`aa76678 benchmark`
> 用途：记录实际完成、已经验证、当前断点和下一步。  
> V1 完成度：约 90%～95%；文本/单图 Hybrid Runtime 已接通。
> 扩展后总体完成度：约 50%～55%；V2 GDN-aware Prefix State Cache、V3 Decode CUDA Graph 与 V4 状态感知 GDN Decode 算子已纳入正式方案，但均尚未实现。

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

V1 范围：RTX 5090、BF16、TP=1、Eager；每个请求最多一张本地 PIL 图片。V1 不支持网络 URL、多图、视频、Qwen3.5 Prefix Cache、MTP、MoE、CUDA Graph、自研 GDN Kernel 和 TP>1。

扩展后的正式范围：

```text
V1：Qwen3.5-9B 文本/单图 Hybrid Runtime（当前主要功能已完成）
V2：纯文本 GDN-aware Prefix State Cache（已规划，未实现；当前首先实施）
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

## 4. V2：GDN-aware Prefix State Cache（已纳入，未实现；当前首先实施）

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
- 使用显存预算、checkpoint interval、引用计数和 LRU 控制开销。
- Prefix 命中后从该边界继续执行剩余 suffix Prefill。

图文 Prefix Cache 暂缓，因为缓存身份还必须包含图片内容、processor 结果、image grid、mRoPE layout 和视觉模型版本；只按 image placeholder token IDs 做 hash 会错误共享不同图片。

### 4.3 待实现 Part

1. Prefix Key：模型/配置身份、完整 block token hash 和前驱链。
2. Prefix Entry：联合持有 KV blocks 与 GDN snapshot。
3. snapshot commit：只在完整 block 边界原子提交。
4. lookup/restore：KV/GDN 同时命中才恢复到活跃 state slot。
5. 显存预算：`prefix_state_cache_max_bytes` 与 checkpoint interval。
6. 生命周期：引用计数、LRU、请求完成、抢占、失效和异常回滚。
7. 正确性：Cache on/off、Chunked Prefill、batch、抢占和淘汰后重算。
8. Benchmark：TTFT、跳过的 Prefill tokens、hit rate、snapshot 拷贝耗时和显存。

### 4.4 关键资源风险

9B 的单请求 GDN active state 约为 49.5 MiB。若每 256 tokens 保存一次 8K 前缀，粗略需要 32 份 checkpoint，单条前缀的 GDN snapshot 就可能约为：

```text
49.5 MiB × 32 ≈ 1.55 GiB
```

因此不能无界保存每个 block 的状态。V2 必须使用显存上限、稀疏 checkpoint 和 LRU，并把 snapshot 显存作为一等指标。

### 4.5 完成判定

- Cache on/off 的 greedy token 完全一致。
- 命中恢复后的 KV 和全部 GDN state 与完整重算一致。
- 实际 scheduled Prefill tokens 按命中长度减少。
- 引用、淘汰、抢占和请求完成后无 KV/state snapshot 泄漏。
- 报告 TTFT 收益，同时报告 snapshot 保存/恢复成本与显存代价。

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

## 7. 当前开发断点：V1 基线冻结，准备进入 V2 Prefix Cache

V1 Runtime 的主要功能代码已接通。先把当前 Eager/Cache-off 行为冻结为后续优化的正确性基线：

1. 整理最终技术复习文档。
2. 整理 README：架构、安装、文本/单图用法、边界。
3. 整理实验报告，区分正确性结果、单次性能结果和未验证结论。
4. 编写简历描述和面试高频追问。
5. 检查 Git 工作区，排除模型权重、大型 profiler 文件和不需要的临时产物后提交。

随后进入 V2 第 1 个 Part：定义纯文本 `PrefixKey`。第一步先核对当前 `BlockManager` 的 block hash、前驱链和引用计数，再把模型/配置身份、完整 token block 和前驱 prefix hash 纳入稳定缓存身份；此时只建立 key/lookup 骨架，不直接保存 49.5 MiB 的 GDN snapshot。

## 8. 剩余 Part

V1 收尾剩余 4 个 Part：

1. 最终技术复习文档。
2. README 和可复现使用说明。
3. 实验报告、简历要点和面试问答。
4. Git 工作区清点、提交和推送。

V2 Prefix State Cache 共 8 个 Part，V3 CUDA Graph 共 6 个 Part，V4 GDN Decode 算子共 8 个 Part；连同 V1 收尾，总计还剩 26 个 Part。

当前正式扩展按以下顺序执行：

```text
V2 Prefix State Cache
→ V3 Hybrid Decode CUDA Graph
→ V4 状态感知 GDN Decode 算子
```

MTP、MoE、TP>1、多图、视频、图文 Prefix Cache 和完整 Chunk Gated Delta Rule 自研 Kernel 不计入这 26 个 Part。

## 9. 当前可以和不能在面试中声称的内容

可以准确表述：

> 基于 nano-vLLM 实现 Qwen3.5-9B 文本/单图 Hybrid Runtime：根据 `layer_types` 为 Full Attention 分配紧凑 Paged KV Cache，为 Gated DeltaNet 设计 state-slot 池管理 depthwise causal-convolution state 和 FP32 recurrent state；实现 FLA GDN Prefill/Decode、Variable-length Batched Prefill、Batched Decode、Chunked Prefill、Decode-first 双 microbatch 调度、Prefill 饥饿保护及 KV/GDN/视觉状态联合抢占重算。接入 Qwen3.5 Vision Transformer/Patch Merger、图像 embedding 替换、三轴 mRoPE 和请求级 visual embedding cache，并完成 HF 64-token greedy 对齐、文本/图文混合批处理、Chunked Prefill 和抢占重算一致性验证。

目前不能声称：

- 已实现 Qwen3.5 Prefix Cache、Hybrid CUDA Graph 或自研 GDN Decode Kernel；三者目前只是已完成设计并纳入正式路线。
- 已实现 MTP 或 MoE；两者仍不在当前正式范围内。
- 已支持多图、视频或 TP>1。
- 已达到生产级 vLLM 完整性或超过官方 vLLM 性能。
- 已获得可泛化的固定百分比性能提升；当前 Benchmark 是特定工作负载的单次实验。
- 所有 token interval 分位数都改善；实测表明 Decode-first 减少最大停顿的同时，可能增加中等停顿的次数。

完成 V2 后可以增加的表述：

> 实现 GDN-aware Prefix State Cache，在完整 block 边界原子复用 Full Attention KV 与 GDN conv/recurrent snapshot，通过显存预算、checkpoint interval、引用计数和 LRU 管理缓存生命周期；性能数字只填写实测结果。

完成 V3 后可以增加的表述：

> 为 Qwen3.5 Hybrid Decode 构建 batch-bucket CUDA Graph，通过静态三轴位置、Paged KV 元数据和 GDN state-slot buffer 完成 capture/replay，并为动态形状保留 Eager fallback；性能数字只填写实测结果。

完成 V4 后可以增加的表述：

> 基于 Nsight 定位 Qwen3.5 Decode 的 GDN 状态访问热点，实现支持离散 `state_slot_ids` 的状态感知融合 Kernel，原地更新 causal-conv state 与 FP32 recurrent state，并与 FLA 完成状态级对齐；性能数字必须同时填写微基准和端到端实测结果。

## 10. 关键文件地图

```text
config.py                 根/文本/视觉配置与资源参数
models/registry.py        Qwen3/Qwen3.5 自动选择
models/qwen3_5.py         Hybrid Decoder 和 Vision 顶层接口
models/qwen3_5_vision.py  Vision Transformer/Patch Merger
models/qwen3_5_mrope.py   LLM 三轴位置 IDs 和 delta
layers/gated_delta_net.py GDN、短卷积、delta rule、FLA
layers/rotary_embedding.py Partial RoPE 和 interleaved mRoPE
inputs.py                 AutoProcessor 和图文输入元数据
engine/sequence.py        token、block table、state slot、CPU 图像负载
engine/hybrid_state.py    Hybrid CacheSpec 和 GPU GDN state pool
engine/block_manager.py   Paged KV blocks
engine/scheduler.py       Decode-first、Chunked Prefill、抢占
engine/model_runner.py    Packed Tensor、GPU 前向和 Cache 读写
engine/llm_engine.py      输入接入、双 microbatch、采样和 decode
tests/compare_hf_vision.py       HF 图文 greedy 对齐
tests/compare_vision_prefill.py  Full/Chunked Prefill 一致性
tests/vision_preemption.py       图文抢占重算一致性
tests/benchmark_hybrid.py        固定工作负载 Benchmark
tests/dynamic_scheduler_benchmark.py 动态调度对照
tests/saturation_benchmark.py    持续闭环饱和压测
```

V2 预计主要修改：

```text
config.py                 Prefix snapshot 预算与 checkpoint interval
engine/block_manager.py   KV block 引用与 Prefix Entry 联动
engine/hybrid_state.py    GDN snapshot 保存、恢复与显存统计
engine/scheduler.py       Prefix lookup、admission 和 scheduled-token 扣减
engine/sequence.py        命中边界和恢复元数据
tests/                    Cache on/off、淘汰、抢占和 Benchmark
```

V3 预计主要修改：

```text
config.py                 Graph bucket、fallback 和开关配置
engine/model_runner.py    静态 Decode buffer、capture/replay 和统计
utils/context.py          Graph replay 所需静态 Context 元数据
tests/                    Eager/Graph 对齐、bucket 和 Benchmark
```

V4 预计主要修改/新增：

```text
layers/gated_delta_net.py torch/fla/custom backend 和调用边界
engine/hybrid_state.py    state pool 直接访问接口与 slot 校验
engine/model_runner.py    custom backend、Graph capture 和 fallback
nanovllm/kernels/         state-aware conv/recurrent CUDA 扩展
tests/                    状态级对齐、动态 slot、微基准和端到端 Benchmark
```

## 11. Git 工作区与恢复入口

2026-08-29 本次核对时 `git status --short` 显示：

```text
D  bench.py
D  hf_cache_probe.py
D  hf_gdn_layer_probe.py
D  hf_text_baseline.py
M  intern/NanoHybrid-VLM项目实施方案.md
M  intern/NanoHybrid-VLM项目进度记忆.md
M  nanovllm/config.py
M  nanovllm/engine/llm_engine.py
M  nanovllm/engine/scheduler.py
?? artifacts/bench/
?? tests/benchmark_hybrid.py
?? tests/dynamic_scheduler_benchmark.py
?? tests/run_benchmark_matrix.py
?? tests/saturation_benchmark.py
?? tests/vision_preemption.py
```

这些删除、修改和未跟踪文件尚未在本次文档更新中处理。Git 收尾时必须逐项确认，不能直接删除或全部提交。

恢复命令：

```bash
cd /workspace/nano-vllm
source .venv/bin/activate
git status
```

然后先按本文第 7 节冻结 V1 基线并提交可恢复节点，再进入第 4 节 V2 的第 1 个 Part：定义 Prefix Key、核对 block hash 与前驱链。
