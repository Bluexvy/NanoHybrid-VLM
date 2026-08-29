# NanoHybrid-VLM 项目进度记忆

> 更新时间：2026-08-29  
> 当前分支：`main`  
> 最近提交：`9943b4b 完成了text model部分`  
> 用途：记录实际完成、已经验证、当前断点和下一步。  
> 总体完成度：约 65%～70%；文本 Hybrid Runtime 约 90%，Vision 组件已完成，图文端到端尚未接通。

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

首版范围：RTX 5090、BF16、TP=1、Eager；每个请求最多一张本地 PIL 图片。首版不支持网络 URL、多图、视频、Qwen3.5 Prefix Cache、MTP、MoE、CUDA Graph 和 TP>1。

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

## 3. 当前准确执行链路

纯文本链路已经可运行：

```text
prompt → tokenize → Sequence → Decode-first Scheduler
→ Variable-length Batched Prefill → KV/GDN state
→ Batched Decode → Sampling → decode text
```

图文链路当前到达：

```text
PIL Image + prompt
→ AutoProcessor
→ token_ids/pixel_values/grid/mm types
→ 自实现 mRoPE positions/delta
→ ProcessedPrompt
→ LLMEngine/Sequence  ← 当前断点：mRoPE 字段尚未传入
→ Scheduler
→ ModelRunner          ← 后续接入 Vision/mRoPE 执行
```

当前 `Sequence` 已经接收基础图像字段，但尚未接收 `mrope_position_ids/delta`。ModelRunner 也尚未运行 Vision Tower、替换 image token embedding，或把三轴 mRoPE 送入 Full Attention。因此目前不能声称已完成端到端图文生成。

## 4. 当前开发断点：Vision Part 9B

Part 9A 已完成：mRoPE positions/delta 已实现并与 HF 对齐。

下一步 Part 9B：

1. `Sequence.__init__()` 增加 `mrope_position_ids`、`mrope_position_delta`。
2. `LLMEngine.add_request()` 从 `ProcessedPrompt` 传入两字段。
3. `Sequence.append_token()` 为 Decode token 追加三轴相同的位置：

```text
new_position = old_num_tokens + mrope_position_delta
```

4. Prefill 序列化传完整 positions；Decode 至少传 delta。
5. 保证 `token_ids/mm_token_type_ids/mrope_position_ids` 在生成和抢占重算后仍对齐。

## 5. 剩余 Part

核心项目从当前状态起约剩 10 个 Part：

1. mRoPE 接入 Sequence、Decode 追加和抢占重算。
2. `RotaryEmbedding` 实现 Qwen3.5 interleaved mRoPE。
3. `prepare_prefill/decode` 构造 packed 三轴 positions。
4. ModelRunner 执行 Vision Tower并替换 image token embeddings。
5. visual embedding cache 的创建、Chunked Prefill 复用、抢占失效和完成释放。
6. 跑通第一个 9B 单图端到端 greedy 生成。
7. 图文 Variable-length Batched Prefill 和文本/图文混合调度。
8. 图文 Chunked Prefill、跨 KV block、Decode、抢占重算一致性。
9. 与 HF 对齐 Vision、单层、logits 误差和 greedy token。
10. 9B Benchmark、显存统计、Profiler、回归、架构图、实验报告和简历描述。

跑通第一个正确单图请求预计还需前 5～6 个 Part。

## 6. 当前可以和不能在面试中声称的内容

可以准确表述：

> 基于 nano-vLLM 接入 Qwen3.5 Gated DeltaNet/Full Attention 混合模型，根据 layer types 为 Full Attention 分配紧凑 Paged KV Cache，并为 GDN 设计 state-slot 状态池，管理 depthwise causal convolution state 和 FP32 recurrent state；实现 Variable-length Batched Prefill、Batched Decode、GDN state Gather/Scatter、Decode-first 双 microbatch 调度、Prefill 饥饿保护与抢占重算。另已实现 Qwen3.5 Vision Transformer/Patch Merger、单图 AutoProcessor 输入和三轴 mRoPE 位置构造，Vision 权重/形状检查通过，mRoPE positions 与 Transformers 官方结果一致。

目前不能声称：

- 已完成端到端图文生成或 visual embedding cache 生命周期。
- 已完成图文 Chunked Prefill/抢占一致性。
- 已实现 Qwen3.5 Prefix Cache、MTP、MoE 或 CUDA Graph。
- 已达到生产级 vLLM 完整性或超过官方 vLLM 性能。
- 已获得固定百分比性能提升；必须等待 Benchmark 实测。

## 7. 关键文件地图

```text
config.py                 根/文本/视觉配置与资源参数
models/registry.py        Qwen3/Qwen3.5 自动选择
models/qwen3_5.py         Hybrid Decoder 和 Vision 顶层接口
models/qwen3_5_vision.py  Vision Transformer/Patch Merger
models/qwen3_5_mrope.py   LLM 三轴位置 IDs 和 delta
layers/gated_delta_net.py GDN、短卷积、delta rule、FLA
layers/rotary_embedding.py 当前 Partial RoPE；待扩展 mRoPE
inputs.py                 AutoProcessor 和图文输入元数据
engine/sequence.py        token、block table、state slot、CPU 图像负载
engine/hybrid_state.py    Hybrid CacheSpec 和 GPU GDN state pool
engine/block_manager.py   Paged KV blocks
engine/scheduler.py       Decode-first、Chunked Prefill、抢占
engine/model_runner.py    Packed Tensor、GPU 前向和 Cache 读写
engine/llm_engine.py      输入接入、双 microbatch、采样和 decode
```

## 8. Git 工作区与恢复入口

当前未全部提交：

```text
M  example.py
M  nanovllm/engine/llm_engine.py
M  nanovllm/engine/sequence.py
M  nanovllm/models/qwen3_5.py
M  pyproject.toml
?? nanovllm/inputs.py
?? nanovllm/models/qwen3_5_mrope.py
?? nanovllm/models/qwen3_5_vision.py
```

恢复命令：

```bash
cd /workspace/nano-vllm
source .venv/bin/activate
git status
```

然后从本文第 4 节的 Vision Part 9B 继续。
