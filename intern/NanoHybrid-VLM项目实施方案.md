# NanoHybrid-VLM：Qwen3.5 图文推理、混合状态缓存与执行优化

> 文档性质：项目设计与实施路线，不代表当前仓库已经实现这些功能。
>
> 建议周期：V1 Hybrid VLM Runtime 为 8～10 周；V2 Prefix State Cache 与 V3 CUDA Graph 另增加约 6 周；V4 状态感知 GDN Decode 算子另增加约 4～5 周。
>
> 开发设备：单张 RTX 5090 32GB，BF16，TP=1；V1 使用 Eager，V2 先实现联合 Prefix State Cache，V3 为 Decode 引入 CUDA Graph，V4 根据真实 Profile 开发状态感知 CUDA 算子。

## 1. 项目定位

本项目基于 nano-vLLM 这个轻量推理引擎，接入 Qwen3.5 图文模型，并围绕 Qwen3.5 的混合层结构实现一套专用推理 Runtime。

Qwen3.5 的文本骨干不是普通的纯 Transformer，而是按照下面的比例交替堆叠两类层：

```text
3 × Gated DeltaNet
1 × Full Attention
```

因此，它同时需要两种完全不同的历史状态：

```text
Full Attention 层：随 token 数增长的 Paged KV Cache
Gated DeltaNet 层：每个请求一份固定大小的卷积状态和递归状态
```

项目的核心不是“把一个新模型类复制进来”，而是解决下面这条完整链路：

```text
文本或单张图片
    ↓
Tokenizer / AutoProcessor
    ↓
Vision Tower + 文本 Embedding
    ↓
Qwen3.5 Hybrid Decoder
    ↓
Paged KV + GDN State 生命周期
    ↓
Chunked Prefill + Decode-first 调度
    ↓
KV/GDN 联合 Prefix State Cache
    ↓
Decode-only CUDA Graph
    ↓
状态感知 GDN Decode 融合算子
    ↓
生成文本与性能数据
```

项目对外应描述为：

> 基于轻量级推理引擎实现 Qwen3.5 Hybrid Runtime，完成异构状态管理、图文 Prefill/Decode、并发调度，并依次研究 KV/GDN 联合 Prefix State Cache、Hybrid Decode CUDA Graph 与状态感知 GDN Decode 融合算子。

不要描述成“实现了生产级 vLLM”或“重写了 vLLM”。

## 2. 为什么这个项目适合写进 AI Infra 实习简历

这个项目可以同时覆盖推理框架岗位常问的五类能力：

1. 模型结构：Transformer、GQA、RoPE、Vision Transformer、Gated DeltaNet。
2. 推理状态：KV Cache、递归状态、Chunked Prefill、Decode、抢占与重算。
3. 推理系统：Continuous Batching、token budget、显存准入和长短请求公平性。
4. 性能工程：GPU Kernel 接入、Profiler、TTFT/TPOT、吞吐和显存分析。
5. 执行与缓存优化：Prefix Cache、CUDA Graph 静态执行、一致性、显存预算和淘汰策略。
6. 算子优化：Profiler 驱动的瓶颈定位、动态 state-slot 访问、状态原地更新、Kernel 融合与端到端验证。

面试中的项目故事可以形成完整闭环：

```text
观察原引擎只支持纯 Attention
    ↓
分析 Qwen3.5 的异构层与状态需求
    ↓
重构模型注册、Cache 和 Scheduler 接口
    ↓
逐层与 Hugging Face 对齐
    ↓
用混合负载验证调度优化
    ↓
原子复用 Full Attention KV 与 GDN prefix state
    ↓
用 CUDA Graph 降低 Hybrid Decode 的 Kernel Launch 开销
    ↓
针对真实 Decode 热点实现状态感知 GDN 融合算子
    ↓
总结收益、退化场景与限制
```

## 3. 模型与范围

### 3.1 模型选择

- 开发与频繁调试：`Qwen/Qwen3.5-0.8B`。
- 最终正确性与性能验证：`Qwen/Qwen3.5-4B`。
- 当前仓库实际主验证模型：`/workspace/models/Qwen3.5-9B`。

0.8B 用来缩短迭代时间；4B 是原计划验证模型；实际开发阶段切换到 9B。三者的核心 Hybrid 架构一致，因此代码不能写死某一个模型的层数和维度。

| 配置 | Qwen3.5-0.8B | Qwen3.5-4B | Qwen3.5-9B |
| --- | ---: | ---: | ---: |
| Text hidden size | 1024 | 2560 | 4096 |
| Text layers | 24 | 32 | 32 |
| GDN layers | 18 | 24 | 24 |
| Full Attention layers | 6 | 8 | 8 |
| Full Q heads | 8 | 16 | 16 |
| Full KV heads | 2 | 4 | 4 |
| Full Attention head dim | 256 | 256 | 256 |
| GDN key heads | 16 | 16 | 16 |
| GDN value heads | 16 | 32 | 32 |
| GDN key/value head dim | 128 / 128 | 128 / 128 | 128 / 128 |
| Vision depth | 12 | 24 | 27 |

### 3.2 V1 必须完成

- Qwen3 和 Qwen3.5 模型注册。
- Qwen3.5 纯文本 Prefill、Decode 和生成。
- 单请求最多一张本地 PIL Image。
- Vision Tower、visual embedding 合并和 multimodal RoPE。
- PyTorch GDN reference 路径。
- FLA + causal-conv1d 快速路径。
- 只给 Full Attention 层分配 Paged KV Cache。
- 为每个活跃请求管理 GDN state slot。
- Chunked Prefill。
- Decode-first 的双 microbatch 调度。
- 抢占后释放状态并重新 Prefill。
- Hugging Face golden tests。
- 0.8B 和 4B 的可复现 Benchmark。

### 3.3 V1 明确不做

- 视频和多图输入。
- 网络 URL 图片下载。
- TP 大于 1。
- Sequence Parallel。
- CUDA Graph；转入 V3 单独实现。
- MTP 投机解码。
- FP8/INT8 KV Cache。
- Qwen3.5 Prefix Cache；转入 V2 单独实现。
- CPU Swap。
- 自己重写 Gated DeltaNet 底层 Kernel；转入 V4，且必须先通过真实 Profile 证明瓶颈。

这些限制不是缺点，而是为了保证 8～10 周能够完成一个正确、可测、能讲清楚的项目。

### 3.4 V2/V3/V4 新增范围

V2：GDN-aware Prefix State Cache。

- 首版只支持纯文本和完整 token block 边界。
- Prefix 命中必须同时具备 Full Attention KV blocks 和同一边界的 GDN conv/recurrent state snapshot。
- 使用独立显存预算和 LRU 淘汰 Prefix State。
- KV 与 GDN snapshot 必须联合命中、联合失效。
- Cache 命中后恢复到活跃 state slot，并从第一个未命中的 token 继续 Prefill。
- 图文 Prefix Cache、部分 block snapshot、CPU swap 和跨进程共享不属于 V2 首版。

V3：Hybrid Decode CUDA Graph。

- Prefill、Vision Tower 和图片预处理保持 Eager。
- 只捕获 Qwen3.5 Hybrid Decode。
- 为 input IDs、三轴 mRoPE、Attention metadata 和 GDN state 建立固定地址 Tensor。
- 按 batch bucket 捕获并支持 Eager fallback。
- Sampler 首版放在 Graph 外。
- 对比 Eager/Graph 的 token、KV、GDN state、TPOT、吞吐、Graph 显存和 Nsight 时间线。

V4：状态感知 GDN Decode 融合算子。

- 先用 Nsight Systems/PyTorch Profiler 证明 GDN Decode 的状态 Gather/Scatter、Depthwise causal-conv1d update、recurrent update 或 Kernel Launch 是真实热点。
- 首版只优化 `L=1` Decode，不重新实现完整 Chunk Gated Delta Rule，也不实现训练反向传播。
- 支持动态 batch 和任意、不连续的 `state_slot_ids`，直接访问 `conv_state_pool/recurrent_state_pool`。
- 第一子算子负责 state-aware Depthwise causal-conv1d update；第二子算子融合衰减、`k^T S`、Delta Rule 写入和 `q^T S` 读取。
- Q/K/V 使用 BF16，`recurrent_state` 保持 FP32，状态在池中原地更新。
- 保留 `torch`、`fla`、`custom` 三条 backend，对比组件误差、Kernel latency、HBM 流量和端到端 TPOT/吞吐。
- 自研算子必须满足 CUDA Graph-safe 条件，并在 V3 Graph 路径中重新 capture 和验证。

继续明确不做：

- MTP speculative decoding。
- MoE/Expert Parallel。
- TP>1。
- 多图和视频。
- 完整 Prefill/Chunk Gated Delta Rule 自研 Kernel。
- 通用 GEMM、训练反向传播和脱离 Runtime 的教学型算子项目。

## 4. 开工前的仓库策略

### 4.1 分支建议

```text
learning-notes
    保存当前带学习注释的 nano-vLLM

feature/qwen35-hybrid-vlm
    从干净 upstream/main 创建，只放功能代码
```

功能分支建议拆成下面几组提交：

```text
1. model registry and config adapter
2. greedy sampling and golden-test harness
3. Qwen3.5 GDN reference and FLA backend
4. full attention and hybrid cache
5. Qwen3.5 text runtime
6. vision tower and multimodal input
7. state-aware scheduler
8. benchmark, profiler and documentation
9. GDN-aware prefix state cache
10. hybrid decode CUDA Graph
11. GDN decode profiling and naive custom kernel
12. state-aware fused GDN decode kernel and runtime integration
13. extension benchmark and documentation
```

每次提交只解决一个问题，保证出现数值错误时可以快速定位或回退。

### 4.2 不应提交的内容

- 模型权重。
- `.venv/`。
- Hugging Face Cache。
- 完整 Nsight trace。
- 大型图片数据集。
- 临时调试输出。

仓库只提交：

- 源码。
- 单元测试。
- 环境版本文件。
- 小型测试图片。
- Benchmark 配置。
- 汇总后的 CSV/JSON 和图表。
- README 与设计文档。

## 5. 第 0 周：环境兼容性门禁

在修改模型代码前，先验证基础环境，避免写完代码才发现 Kernel 无法在 RTX 5090 上运行。

当前环境需要重点检查：

```text
Python
PyTorch + CUDA
Triton
Transformers 是否包含原生 Qwen3.5
FlashAttention
Flash Linear Attention
causal-conv1d
GPU compute capability 是否为 SM120
```

建议输出一张明确的表：

| 项目 | 版本 | 是否通过 | 失败原因 |
| --- | --- | --- | --- |
| PyTorch | 实测填写 | PASS/FAIL |  |
| Triton | 实测填写 | PASS/FAIL |  |
| Transformers Qwen3.5 | 实测填写 | PASS/FAIL |  |
| FlashAttention | 实测填写 | PASS/FAIL |  |
| FLA GDN | 实测填写 | PASS/FAIL |  |
| causal-conv1d | 实测填写 | PASS/FAIL |  |
| RTX 5090 / SM120 | 实测填写 | PASS/FAIL |  |

版本选择原则：

1. 不根据模型 `config.json` 里的 `transformers_version` 猜测兼容性，必须实际 import Qwen3.5 类。
2. FLA 先选择明确包含消费级 Blackwell/SM120 修复的稳定版本，再跑数值测试。
3. 不因为包能够 import 就认为成功，必须在 5090 上真实执行一次 GDN prefill 和 decode。
4. 锁版本前记录 PyTorch、CUDA、Triton 和 C++ ABI，避免二进制扩展不匹配。

第 3 天前的硬门槛：

```text
Transformers Qwen3.5 baseline 可以执行
FLA chunk GDN 可以执行
FLA recurrent GDN 可以执行
causal-conv1d prefill/update 可以执行
输出没有 NaN、非法显存访问或明显数值异常
```

若这个门槛失败，先解决依赖，不进入模型移植。

## 6. 第一部分：模型注册和配置适配

### 6.1 当前问题

当前 `ModelRunner` 直接构造 `Qwen3ForCausalLM`，因此模型类型被写死。

Qwen3 的配置字段直接位于根配置；Qwen3.5 图文模型则使用：

```text
root config
├── text_config
└── vision_config
```

如果仍然直接访问：

```python
hf_config.num_hidden_layers
hf_config.num_key_value_heads
```

Qwen3.5 就会读错或报错。

### 6.2 计划接口

新增模型注册表：

```python
MODEL_REGISTRY = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_5": Qwen3_5ForConditionalGeneration,
}
```

实际实现还应同时检查 `architectures`，并使用惰性 import，避免仅加载配置时就导入 GPU 依赖。

配置层统一提供：

```python
config.hf_config       # 完整根配置
config.text_config     # 文本骨干配置
config.vision_config   # 可选视觉配置
```

### 6.3 验收标准

- 原 Qwen3 模型仍可被正确解析。
- Qwen3.5 自动走新模型类。
- 未支持模型给出清晰错误。
- 缺少核心文本字段时在 GPU 初始化前失败。
- CPU 单测不需要下载模型权重。

## 7. 第二部分：采样与 Hugging Face Golden Test

### 7.1 Greedy decoding

当前 nano-vLLM 禁止 `temperature=0`。为了做逐 token 对齐，需要实现真正的 greedy：

```text
temperature = 0  → argmax(logits)
temperature > 0  → 保留原随机采样
```

greedy 路径不能调用随机数生成器，否则相同输入可能影响后续随机请求的 RNG 状态。

### 7.2 Golden Test 分层

不要一开始就比较最终生成文本。应从最小组件逐层比较：

```text
输入
→ 单个投影
→ 单层输出
→ 单层最终 state
→ Decoder block
→ 完整 text model logits
→ greedy token
→ 图文完整生成
```

每层记录：

- 最大绝对误差。
- 平均绝对误差。
- 最大相对误差。
- top-1 token 是否一致。
- 是否出现 NaN/Inf。

## 8. 第三部分：Qwen3.5 Gated DeltaNet

### 8.1 需要实现的投影

每个 GDN 层包含：

```text
in_proj_qkv
in_proj_z
in_proj_b
in_proj_a
depthwise causal conv1d
A_log
dt_bias
RMSNormGated
out_proj
```

投影关系：

```text
hidden_states
    ├── in_proj_qkv → Q、K、V
    ├── in_proj_z   → output gate
    ├── in_proj_b   → beta
    └── in_proj_a   → decay
```

### 8.2 短卷积状态

GDN 的 Q/K/V 在进入递归更新前需要深度可分离的因果卷积。

卷积输入宽度：

```text
key_width   = linear_num_key_heads × linear_key_head_dim
value_width = linear_num_value_heads × linear_value_head_dim
conv_dim    = 2 × key_width + value_width
```

每个请求、每个 GDN 层的卷积状态：

```text
[conv_dim, linear_conv_kernel_dim]
```

### 8.3 递归更新原理

用简化符号表示，每个 token 的核心更新是：

```text
S_t = decay_t × S_(t-1)
remembered_v = K_tᵀ × S_t
delta = beta_t × (V_t - remembered_v)
S_t = S_t + K_t × deltaᵀ
O_t = Q_tᵀ × S_t
```

其中 `S_t` 就是必须跨 Chunked Prefill 和 Decode 保存的 recurrent state。

### 8.4 两条实现路径

`torch` 路径：

- 用清晰的 PyTorch 循环实现。
- 所有递归运算使用 FP32。
- 重点是可读性和数值排障。
- 只用于小张量测试和 golden reference。

`fla` 路径：

- Prefill 使用 chunk Gated Delta Rule。
- 单 token Decode 使用 fused recurrent Gated Delta Rule。
- 短卷积使用 causal-conv1d。
- 作为实际 Benchmark 路径。

### 8.5 必须通过的测试

```text
整段 Prefill 输出
≈ 两段 Chunked Prefill 拼接输出
≈ 逐 token Decode 拼接输出
```

同时比较最终：

```text
conv_state
recurrent_state
```

第 3 周末的硬门槛：单个 GDN 层的输出与最终状态能够和 Hugging Face/FLA 对齐。

## 9. 第四部分：Full Attention 与 partial multimodal RoPE

Qwen3.5 Full Attention 与现有 Qwen3 Attention 不能直接共用全部实现。

关键差异：

1. `q_proj` 同时输出 Query 和 attention output gate。
2. Q/K 做 head-dim RMSNorm。
3. 使用 GQA。
4. 只对 `head_dim × partial_rotary_factor` 的维度应用 RoPE。
5. 图文输入使用三轴 multimodal RoPE。
6. Attention 输出先乘 `sigmoid(gate)`，再经过 `o_proj`。

0.8B 和 4B 的 `head_dim` 都是 256，`partial_rotary_factor=0.25`，因此旋转维度为：

```text
rotary_dim = 256 × 0.25 = 64
```

纯文本位置可以看作三个轴使用相同位置；图片 token 则需要时间、高度、宽度三个位置轴。

测试必须覆盖：

- 纯文本 position IDs。
- 图像网格 position IDs。
- partial rotary 后未旋转维度保持不变。
- Q/K norm。
- output gate。
- 单层 logits 与 Hugging Face 对齐。

## 10. 第五部分：Hybrid Cache

### 10.1 HybridCacheSpec

从 `text_config.layer_types` 自动推导：

```text
full_attention_layer_ids
gdn_layer_ids
```

不能写死“每 4 层一个 Attention”，即使当前两个模型恰好如此；Runtime 应以配置中的 `layer_types` 为准。

### 10.2 Paged KV Cache

只为 Full Attention 层分配：

```text
[2, num_full_attention_layers, num_blocks,
 block_size, num_kv_heads, head_dim]
```

具体形状：

```text
0.8B: [2, 6, B, 256, 2, 256] BF16
4B:   [2, 8, B, 256, 4, 256] BF16
```

每个 token 的 KV 成本：

```text
0.8B: 12 KiB/token，3 MiB/block
4B:   32 KiB/token，8 MiB/block
```

如果仍给所有 Decoder 层分配 KV，会浪费约 4 倍 KV 显存。

### 10.3 GDN State Pool

状态池：

```text
conv_state[
    sequence_slot,
    gdn_layer,
    conv_dim,
    conv_kernel_dim,
]

recurrent_state[
    sequence_slot,
    gdn_layer,
    num_value_heads,
    key_head_dim,
    value_head_dim,
]
```

具体形状：

```text
0.8B conv:      [S, 18, 6144, 4] BF16
0.8B recurrent: [S, 18, 16, 128, 128] FP32

4B conv:        [S, 24, 8192, 4] BF16
4B recurrent:   [S, 24, 32, 128, 128] FP32
```

每个活跃请求的 GDN 状态约为：

```text
0.8B: 18.84 MiB
4B:   49.50 MiB
```

这意味着 `max_num_seqs=512` 不能直接等同于 state slot 数。4B 如果为 512 个请求提前分配 recurrent state，单这一项就会超过 25GB。

因此需要独立配置或自动计算：

```text
max_num_state_slots
```

并在请求进入运行队列前同时检查：

```text
有空闲 state slot
且
有足够 KV blocks
```

### 10.4 状态生命周期

```text
请求进入 Prefill
    → 分配 state slot
    → state 全部清零

第一个 Prefill chunk
    → 从零状态开始
    → 保存最终状态

后续 Prefill chunk
    → 读取上一 chunk 状态
    → 更新并写回

Decode
    → 读取状态
    → 单 token 更新
    → 原位写回

请求完成
    → 释放 KV blocks
    → state slot 清零并回收到 free list

请求被抢占
    → 释放 KV 和 GDN state
    → num_cached_tokens 归零
    → 恢复时重新 Prefill
```

首版不做 CPU swap，因为跨设备保存几十 MiB 的每请求状态会显著增加复杂度；使用确定性重算更容易保证正确。

### 10.5 为什么关闭 Qwen3.5 Prefix Cache

GDN recurrent state 是整个历史前缀的压缩结果。

如果只复用 Full Attention 的 KV block，却没有恢复同一前缀位置对应的 GDN state，那么后续 token 会从错误状态继续计算，输出必然不正确。

所以首版策略是：

- Qwen3 保留原 Prefix Cache。
- Qwen3.5 明确禁用 Prefix Cache。
- 后续如果要做，需要同时缓存 KV 和 prefix boundary 的 GDN state checkpoint。

## 11. 第六部分：把 GDN 状态接进执行上下文

当前执行上下文只描述 Attention 所需的张量，例如：

```text
is_prefill
cu_seqlens_q
cu_seqlens_k
slot_mapping
context_lens
block_tables
```

Qwen3.5 还需要加入：

```text
sequence_ids 或 state_slot_ids
每条请求本轮的 query token 数
GDN state manager / state views
本轮是 chunked prefill 还是 decode
multimodal position IDs
```

Prefill batch 中不同请求长度不同，FLA 可以使用类似 `cu_seqlens` 的边界来区分各请求，并为每个请求读取/写回独立 recurrent state。

Decode 时每个请求通常只有一个新 token，因此输入形状近似：

```text
[batch_size, 1, hidden_size]
```

但每个 batch row 必须映射到正确的 state slot，不能根据 batch 下标直接索引状态，因为请求在不同 step 中的 batch 顺序会变化。

## 12. 第七部分：Qwen3.5 文本模型

建议实现顺序：

```text
Qwen3.5 RMSNorm
→ SwiGLU MLP
→ GDN layer
→ Full Attention layer
→ Decoder block
→ Text model
→ tied LM head
```

注意 Qwen3.5 的 RMSNorm 权重语义可能与现有 Qwen3 层不同，必须按照官方实现和 checkpoint 验证，不能只因为名字相同就直接复用。

完整文本前向要支持：

- 单请求整段 Prefill。
- 单请求 Chunked Prefill。
- 单请求 Decode。
- 多请求不同长度 Prefill。
- 多请求 Decode。
- GDN state 保存和恢复。
- Full Attention Paged KV 写入和读取。
- tied embedding/LM head。

第 5 周末硬门槛：

```text
Qwen3.5-0.8B 纯文本 greedy 逐 token 与 Hugging Face 对齐
```

如果整段 logits 存在 BF16 误差，至少必须报告：

- 最大/平均 logits 误差。
- top-1 一致率。
- 生成 token 第一次分叉的位置。
- 分叉前是哪一层开始误差明显放大。

## 13. 第八部分：严格权重加载

现有 loader 遇到未知参数时直接查找模型参数，但没有完整的加载后核对。

新 loader 应记录三类集合：

```text
loaded_keys
ignored_keys
unexpected_keys
```

Qwen3.5 官方 checkpoint 的主要根路径包括：

```text
model.language_model.*
model.visual.*
mtp.*
```

首版只允许显式忽略：

```text
mtp.*
```

其他无法识别的 key 必须报错，不能使用宽泛的 `try/except` 静默跳过。

加载结束后还要检查模型中是否存在未加载参数。

必须特别验证：

- tied embedding 不需要重复的 `lm_head.weight`。
- Full Attention 的 `q_proj` 同时含 Query 和 gate 通道。
- GDN 的 conv1d weight/bias shape。
- 根路径到本地模型路径的映射。
- 0.8B 与 4B 均不依赖手工权重转换。

## 14. 第九部分：单图输入和 Vision Tower

### 14.1 公共接口

保留文本输入，同时支持：

```python
prompts = [
    "Explain linear attention.",
    {
        "prompt": "Describe this image.",
        "multi_modal_data": {
            "image": pil_image,
        },
    },
]
```

校验规则：

- `prompt` 必须是字符串。
- `multi_modal_data` 只允许 `image`。
- image 必须是本地 PIL Image。
- 每个请求最多一张图。
- 不下载 URL。

### 14.2 处理流程

```text
PIL Image
    ↓ AutoProcessor
pixel_values + image_grid_thw + 特殊 token
    ↓
Patch Embedding
    ↓
Vision Transformer blocks
    ↓
Patch Merger / projection
    ↓
visual embeddings
    ↓
替换语言序列中的 image placeholder embeddings
```

使用官方 `AutoProcessor` 的原因是图片 resize、patch 排列、特殊 token 和 grid 规则比较容易出错；但 Vision Tower 前向和 embedding 合并仍由项目自己执行。

### 14.3 第一版视觉缓存

图片只在 Prefill 中处理一次。可以在请求对象中临时保存 processor 输出或 visual embeddings，但请求完成、抢占或异常后必须释放。

首版抢占后可以重新计算 Vision Tower，以换取清晰的生命周期；后续再研究 visual embedding cache。

### 14.4 视觉正确性门槛

按顺序比较：

1. Processor 输出 shape。
2. `image_grid_thw`。
3. visual token 数。
4. Patch Embedding。
5. 单个 Vision block。
6. Patch Merger。
7. 最终 visual embeddings。
8. 合并后的图文 embedding 序列。
9. multimodal position IDs。
10. 单图完整 logits 与 greedy token。

第 7 周末硬门槛：至少一个单图请求能够完整生成文本并通过关键数值检查。

## 15. 第十部分：Decode-first 状态感知调度

### 15.1 当前调度问题

当前 Scheduler 的 batch 要么全部 Prefill，要么全部 Decode，而且 Prefill 优先。

长文本或图片 Prefill 进入后，正在生成的请求可能长时间无法 Decode，导致 TPOT 尾延迟恶化。

### 15.2 SchedulePlan

将返回值从：

```python
(seqs, is_prefill)
```

改成逻辑计划：

```python
SchedulePlan(
    decode=DecodeMicroBatch(...),
    prefill=PrefillMicroBatch(...),
    num_scheduled_tokens=...,
)
```

一次逻辑 step 的执行顺序：

```text
先执行 Decode microbatch
    ↓
计算剩余 token budget
    ↓
再执行 Chunked Prefill microbatch
```

首版不把两者拼成一个 Tensor，而是调用两次 ModelRunner，优先保证正确性。

### 15.3 调度规则

1. 所有 running Decode 请求先各获得 1 token budget。
2. 剩余 token budget 给 waiting/chunked Prefill。
3. Prefill 请求只有同时拿到 state slot 和 KV blocks 才能 admission。
4. 长 prompt 自动按 budget 切 chunk。
5. 等待超过 `max_prefill_wait_ms` 的请求强制保留至少一个 prefill chunk。
6. 显存不足时，优先抢占已计算 token 较少、重算成本较低的请求。
7. 抢占时 KV 和 GDN state 必须一起释放。

### 15.4 为什么不能永远只顾 Decode

如果一直优先 Decode 且系统持续有生成请求，新来的 Prefill 可能永远无法进入模型。

`max_prefill_wait_ms` 或 aging 的作用是给等待时间设置上界，在 Decode 延迟和 Prefill 饥饿之间取得平衡。

### 15.5 调度指标

每轮记录：

```text
decode batch size
prefill batch size
prefill scheduled tokens
waiting/running queue length
每请求 queue wait
state slots used/free
KV blocks used/free
preemption count
recomputed tokens
longest wait
scheduler CPU time
```

## 16. 扩展技术设计：Hybrid Decode CUDA Graph（V3）

### 16.1 实现边界

V3 只捕获重复执行、形状相对稳定的 Decode：

```text
Tokenizer / AutoProcessor：CPU / Eager
Vision Tower：Eager
Variable-length Prefill：Eager
Hybrid Decode：CUDA Graph
Sampler：首版在 Graph 外
```

不捕获 Prefill 和 Vision 的原因：

- Prefill token 数和 `cu_seqlens` 动态变化。
- 图像分辨率、patch 数和 visual token 数动态变化。
- 为大量动态形状分别捕获 Graph 会增加启动时间和 Graph private pool 显存。
- Decode 每条请求通常只有一个输入 token，更适合静态 batch bucket。

### 16.2 静态输入和上下文

每个 batch bucket 预分配固定地址 Tensor：

```text
input_ids        [B]                 int64
positions        [3,B]               int64
slot_mapping     [B]                 int32
context_lens     [B]                 int32
block_tables     [B,max_num_blocks]  int32
state_slot_ids   [B]                 int64/int32
hidden_states    [B,hidden_size]      model dtype
```

普通 Qwen3 的旧 Graph 使用一维 `positions[B]`；Qwen3.5 图文 Decode 必须使用三轴 `positions[3,B]`。

每次 replay 前只允许把本轮数据复制进静态 Tensor，不能替换 Tensor 对象或改变内存地址。

### 16.3 GDN State 静态化

当前 Eager Decode：

```text
state pool
    ↓ read_batched_states()
临时 batched conv/recurrent state
    ↓ model forward
updated state
    ↓ write_batched_states()
state pool
```

CUDA Graph 要求固定地址，因此需要为每个 bucket 创建长生命周期的：

```text
graph_conv_states[layer]
graph_recurrent_states[layer]
graph_updated_conv_states[layer]
graph_updated_recurrent_states[layer]
```

首版流程：

```text
正式 state pool
    ↓ Graph 外 Gather
静态 Graph state input
    ↓ graph.replay()
静态 Graph state output
    ↓ Graph 外 Scatter
正式 state pool
```

优化版可以把根据静态 `state_slot_ids` 的 Gather/Scatter 一并捕获，但必须先证明 in-place 写回和 Graph replay 的地址、顺序及状态一致性。

### 16.4 Capture 与 Replay

Capture：

```text
创建 scratch KV blocks 和 scratch GDN slots
    ↓
在非默认 CUDA stream 上 warm-up
    ↓
使用固定输入执行一次
    ↓
torch.cuda.graph(...) 捕获完整 Hybrid Decode
    ↓
保存 Graph、Graph pool 和所有静态 Tensor
```

Replay：

```text
选择 batch bucket
    ↓
复制 input IDs / mRoPE / Attention metadata / state
    ↓
graph.replay()
    ↓
读取固定 hidden/state 输出
    ↓
写回正式 GDN state
    ↓
Graph 外 compute_logits + sampler
```

捕获使用的 KV block 和 GDN slot 必须是专用 scratch 资源。Capture/warm-up 会真实执行状态更新，不能污染活跃请求或 Prefix Cache。

### 16.5 Batch Bucket 与 Eager Fallback

初始 bucket：

```text
1 / 2 / 4 / 8 / 12
```

要求：

- `max_num_seqs=12` 时必须存在 batch 12 的 Graph，不能沿用只生成 1/2/4/8/16 的旧逻辑。
- 首版优先为实际 batch size 捕获精确 Graph。
- 未捕获的 batch size 回退 Eager，不能为了复用较大 Graph 随意写入真实 Cache。
- 后续如果采用 padding bucket，dummy row 必须使用隔离的 KV/state scratch 资源并保证输出不会参与采样。

### 16.6 Graph-safety 门禁

正式改造前分别验证：

1. FLA fused recurrent GDN。
2. causal-conv1d 单 token update。
3. Paged/Flash Attention Decode。
4. RMSNorm、SwiGLU 和 Linear。
5. 三轴 mRoPE。
6. GDN state Gather/Scatter。

Capture 区域内禁止：

- `.item()` 或其他 CPU/GPU 同步。
- 动态 shape。
- 根据 Tensor 值执行 Python 控制流。
- 新的 CUDA allocation。
- 生命周期短于 Graph 的输入/输出 Tensor。

### 16.7 正确性和性能验收

正确性：

- Eager/Graph greedy token 完全一致。
- Full Attention KV Cache 一致。
- 每层 conv state 和 recurrent state 一致。
- batch 1/2/4/8/12。
- 纯文本和图文请求完成 Prefill 后的 Decode。
- Chunked Prefill、跨 KV block、抢占恢复后进入 Graph Decode。
- 多次 replay 后无状态串扰和显存泄漏。

性能：

- Decode tokens/s。
- TPOT/ITL p50/p95/p99/max。
- Scheduler 和 ModelRunner CPU 时间。
- CUDA Kernel Launch 间隔。
- Graph capture 时间和额外显存。
- Eager/Graph 在并发 1/4/8/12 下的对比。
- Nsight Systems 中 CPU/GPU 空洞变化。

最后只报告实测收益；如果高并发下无收益，要区分 CPU launch overhead、模型计算、state copy 和 Graph private pool 的影响。

## 17. 扩展技术设计：GDN-aware Prefix State Cache（V2）

### 17.1 为什么必须联合缓存

一个 Qwen3.5 前缀的完整历史包含：

```text
Full Attention layers
    → Paged KV blocks

Gated DeltaNet layers
    → conv_state
    → recurrent_state
```

只命中 Full Attention KV、却从零初始化 GDN state，会让两类层从不同历史位置继续执行，生成结果错误。

因此 V2 的命中条件是：

```text
KV prefix blocks 有效
且
同一个 prefix boundary 的 GDN snapshot 有效
```

任何一侧缺失都必须当作完整 miss。

### 17.2 PrefixStateEntry

建议的数据结构：

```python
PrefixStateEntry(
    prefix_hash,
    num_cached_tokens,
    kv_block_ids,
    conv_state_snapshot,
    recurrent_state_snapshot,
    last_access_time,
    ref_count,
    allocated_bytes,
)
```

其中：

- `prefix_hash` 沿用按完整 token block 链式计算的 hash。
- `num_cached_tokens` 必须是 `block_size` 的整数倍。
- `kv_block_ids` 指向现有物理 KV blocks。
- GDN snapshot 表示处理完该完整前缀边界后的所有 GDN 层状态。
- Entry 只有在 KV 和 GDN snapshot 都提交成功后才可见。

### 17.3 命中和恢复流程

```text
新请求进入 waiting
    ↓
按 token block 计算最长 prefix hash
    ↓
联合检查 KV blocks + GDN snapshot
    ↓
为请求引用已有 KV blocks
    ↓
申请一个活跃 state slot
    ↓
把 snapshot 复制到该 slot
    ↓
seq.num_cached_tokens = entry.num_cached_tokens
    ↓
从第一个未命中 token 继续 Chunked Prefill
```

恢复动作必须是事务式的：

- KV 引用成功但 state slot/snapshot 恢复失败时，撤销 KV 引用。
- state slot 已申请但 KV 引用失败时，释放并清零 slot。
- Scheduler 只有在两类资源都可用时才允许 admission。

### 17.4 Snapshot 粒度和显存预算

9B 当前实测每条活跃请求完整 GDN state 约 49.5 MiB。

如果对一个 8K 前缀的每个 256-token block 都保存完整 snapshot：

```text
32 boundaries × 49.5 MiB ≈ 1.55 GiB
```

因此不能无界缓存。V2 必须提供：

```text
prefix_state_cache_max_bytes
prefix_state_checkpoint_interval_blocks
prefix_state_cache_eviction_policy = LRU
```

首版策略：

- 只保存完整 block 边界。
- 默认不是每个 block 都保存，可按固定 block 间隔 checkpoint。
- 优先缓存被多请求复用的最长边界。
- snapshot 使用模型要求的 dtype：conv state 保持原 dtype，recurrent state 首版保持 FP32。
- 不通过降低 recurrent dtype 换取表面显存收益，除非另行完成数值验证。

### 17.5 引用、淘汰与生命周期

Prefix Entry 状态：

```text
building
    → active/cached
    → evictable
    → evicted
```

生命周期要求：

- 正在被请求引用的 KV blocks/Prefix Entry 不能淘汰。
- 请求完成只释放活跃 state slot 和 KV 引用，不立即删除可复用 snapshot。
- KV block 被重新分配或 hash 失效时，对应 Prefix Entry 必须同步失效。
- GDN snapshot 被 LRU 淘汰时，不能留下“KV 命中但 state 缺失”的半命中状态。
- 抢占活跃请求时释放其活跃 KV/state；已提交的 Prefix Entry 仍按 LRU 生命周期管理。
- 异常构建的 Entry 不进入查询表。

需要记录：

```text
prefix lookups
prefix hits/misses
cached/skipped tokens
snapshot bytes
snapshot copies
evictions
restore failures
KV-only rejected hits
```

### 17.6 首版多模态边界

V2 首版只支持纯文本 Prefix State Cache。

图文前缀还需要把以下内容纳入缓存身份：

- 图片内容或 processor 输出的稳定 hash。
- image grid。
- image token layout。
- mRoPE position IDs/delta。
- visual embedding 与 Vision 模型/Processor 配置版本。

如果只按 token IDs 缓存，两个特殊 image token 相同但实际图片不同的请求可能错误共享状态。因此图文 Prefix Cache 留作后续独立扩展。

### 17.7 正确性和性能验收

构造：

```text
请求 A = 共享 4096-token prefix + suffix A
请求 B = 共享 4096-token prefix + suffix B
```

验证：

- B 命中预期数量的完整 blocks。
- B 的 Prefill scheduled tokens 相应减少。
- KV block IDs 确实共享且引用计数正确。
- 恢复后的所有 GDN state 与完整重算在容差内一致。
- B 的 greedy token 与关闭 Prefix Cache 时完全一致。
- Chunked Prefill、batch、抢占、跨 block 和 Entry 淘汰后仍正确。
- 请求完成后无活跃 KV/state slot 泄漏。
- Cache 淘汰后重新计算结果仍一致。

Benchmark：

- Prefix hit rate。
- TTFT。
- Prefill tokens/s 与实际执行 token 数。
- snapshot 保存/恢复耗时。
- KV、GDN active state、Prefix State 各自显存。
- 最大并发。
- 不同共享前缀长度和并发下的收益。

## 18. 第十三部分：状态感知 GDN Decode 融合算子（V4）

### 18.1 为什么不能先写 Kernel

V4 不是为了简历孤立复现一个教学算子，而是解决 NanoHybrid Runtime 中已经由 Profile 证明的真实 Decode 热点。正式实现前必须使用 Nsight Systems、Nsight Compute 和 PyTorch Profiler 分解：

```text
GDN state Gather/Scatter 时间
causal-conv1d 单 token update 时间
FLA fused recurrent 时间
每层/每 token 的 Kernel Launch 数量
recurrent state HBM 读写字节
端到端 Decode step 中各部分占比
```

只有状态整理、Depthwise causal-conv1d、recurrent update 或它们之间的 Launch/内存流量构成显著瓶颈时，才进入自研 Kernel。若真实热点不在这里，应依据 Profile 调整融合边界，而不是强行制造一个没有端到端价值的算子。

### 18.2 第一子算子：State-aware Causal Conv1d Update

首版只处理 Decode 的 `L=1`：

```text
mixed_qkv       [B, conv_dim]                         BF16
state_slot_ids  [B]                                   int32/int64
conv_state_pool [num_slots, num_gdn_layers,
                 conv_dim, kernel_size]               BF16
conv_weight     [conv_dim, kernel_size]                BF16
```

动态 batch 的 `state_slot_ids` 可能是 `[7,2,15,4]`，Kernel 不能把 batch row 当成状态槽编号，而要通过 `state_slot_ids[row]` 找到该请求、该 GDN 层的 `conv_state`。

逻辑操作：

```text
读取请求对应的最近 kernel_size-1 个 mixed_qkv 历史
    ↓
写入当前 token 的 mixed_qkv
    ↓
每个 channel 独立执行 Depthwise causal convolution
    ↓
原地保留下一 token 所需的 conv_state
```

不能真的每轮执行整段 `state[..., :-1] = state[..., 1:]`。优化设计使用 `conv_cursor` 环形缓冲区或等价的固定窗口布局，减少状态搬移。需要验证 channel 独立性、权重布局、边界补零、slot 回收复用和任意 batch 顺序。

### 18.3 第二子算子：State-aware Gated Delta Recurrent Update

核心输入和状态：

```text
q/k             [B, H, K]                             BF16
v               [B, H, V]                             BF16
beta/g          [B, H] 或配置对应的可广播形状          BF16/FP32
state_slot_ids  [B]                                   int32/int64
state_pool      [num_slots, num_gdn_layers, H, K, V]  FP32
output          [B, H, V]                             BF16
```

每个 token 的数学过程：

```text
S = exp(g) · S
prediction = k^T S
delta = beta · (v - prediction)
S = S + k ⊗ delta
o = q^T S
```

自研 Kernel 使用 `state_slot_ids` 直接定位 `state_pool`，按 `[K,V]` tile 加载状态，在一次融合路径中完成状态衰减、旧值预测、Delta Rule 修正、Query 读取和 FP32 状态写回。目标是消除或减少：

```text
Gather state
多个逐算子临时 Tensor
recurrent state 的重复 HBM 往返
Scatter state
多次 Kernel Launch
```

第一版不把 Linear Projection GEMM 融进来，因为 GEMM 应继续交给 cuBLAS/现有 Linear 层；融合边界集中在状态更新这种通用 GEMM 库无法表达的部分。

### 18.4 Backend 与 Runtime 接口

`gated_delta_net.py` 保留三条可切换路径：

```text
backend="torch"   公式 reference 和小张量排障
backend="fla"     当前生产基线
backend="custom"  V4 自研状态感知 Decode Kernel
```

`custom` 路径从 `HybridStateManager` 接收：

```text
state_slot_ids
gdn_layer_idx
conv_state_pool
recurrent_state_pool
```

不能先把整个 FP32 state Gather 到连续临时 Tensor、计算后再 Scatter，否则会丢失状态感知融合的主要价值。请求结束、抢占、Prefix restore 和 state slot 复用仍由现有 Runtime 管理，Kernel 只负责本轮合法 slot 的读取和原地更新。

V4 完成后还必须重新捕获 V3 CUDA Graph：Graph 内 `state_slot_ids`、状态池地址和 custom backend 输出地址必须稳定；custom 不满足 Graph-safe 条件时自动回退 `fla` 或 Eager。

### 18.5 分阶段优化路线

```text
V0：PyTorch reference
    ↓ 正确性基线
V1：当前 FLA/causal-conv1d baseline
    ↓ 真实 Profile
V2：Naive state-aware custom Kernel
    ↓ 保证任意 slot 和 FP32 state 正确
V3：Tiled/vectorized/fused Kernel
    ↓ 减少 HBM 往返和 Launch
V4：接入 Scheduler/ModelRunner/CUDA Graph
    ↓ 端到端 Benchmark
```

每一版都保留可复现实现与数据，不能只留下最终 Kernel，否则无法解释每个优化为什么有效。

### 18.6 正确性验收

- batch size `1/2/4/8/12/16/32`。
- 连续与不连续、乱序的 `state_slot_ids`。
- 同一请求连续 Decode `1/16/64/256` tokens。
- state slot 释放、清零和复用。
- Prefix Cache restore 后首次 Decode。
- 抢占重算后进入 custom backend。
- `torch/fla/custom` 的 conv output、最终 conv state、recurrent output 和最终 recurrent state 对齐。
- Eager custom 与 CUDA Graph custom greedy token 完全一致。
- 无越界访问、状态串槽、NaN/Inf 和显存泄漏。

### 18.7 性能验收

微基准必须比较：

```text
PyTorch reference
FLA + causal-conv1d
Naive custom
Optimized custom
```

记录：

- Kernel latency 和 launch 数量。
- DRAM throughput、L2 hit rate、occupancy、active warps 和 register 使用量。
- 估算/实测 state bytes 与中间 Tensor bytes。
- batch size、slot 连续性和连续 Decode 长度的影响。

端到端必须比较：

- Decode tokens/s、requests/s。
- TPOT/ITL p50/p95/p99/max。
- Eager 和 CUDA Graph 下的收益。
- Kernel 加速占整步 Decode 的比例。

不预先填写提升百分比；如果 custom 不能超过 FLA，必须报告失败原因、适用 shape 和端到端边界，不能只挑有利 case。

## 19. 正确性测试矩阵

### 19.1 组件级

- Processor 与 visual token 数。
- Vision Patch Embedding。
- Vision block 与 Patch Merger。
- Full Attention 输出。
- Q/K Norm、partial mRoPE、output gate。
- GDN 卷积输出。
- GDN recurrent 输出与最终 state。
- Decoder block。
- LM Head。

### 19.2 模型级

- 0.8B 纯文本完整前向。
- 0.8B 单图完整前向。
- 4B 文本 smoke test。
- 4B 单图 smoke test。
- greedy token 序列。

### 19.3 Runtime 级

- 整段 Prefill vs Chunked Prefill。
- Prefill vs 逐 token Decode。
- 单请求 vs batch。
- 不同 prompt 长度。
- 不同 visual token 数。
- 文本和图文混合 batch。
- EOS。
- `max_tokens`。
- 跨 KV block 边界。
- state slot 回收和重复使用。
- 抢占、释放、重新 Prefill 后结果一致。
- 异常退出后无状态泄漏。

CUDA Graph：

- Eager Decode 与 Graph Decode 的 greedy token 完全一致。
- 不同 batch bucket、实际 batch size 和 padding 下结果一致。
- 图文请求完成 Vision Prefill 后能够进入同一 Decode Graph 路径。
- bucket 不匹配、算子不满足 Graph-safe 条件时能够安全回退 Eager。
- 多次 replay 不覆盖错误的 KV block、GDN state slot 或请求元数据。

Prefix State Cache：

- Prefix Cache 开启/关闭时 greedy token 完全一致。
- 命中时同时恢复 Full Attention KV 与全部 GDN conv/recurrent state。
- 整段 Prefill、Chunked Prefill、命中恢复三条路径的最终状态一致。
- KV-only 命中、state 缺失、hash 冲突或 Entry 失效时拒绝复用并重新计算。
- 引用计数、LRU 淘汰、请求完成和抢占后无 KV/state snapshot 泄漏。

### 19.4 回归要求

Qwen3.5 的重构不能破坏原 Qwen3：

- Qwen3 原有生成仍能运行。
- Qwen3 Prefix Cache 行为不变。
- Qwen3 eager 路径结果不变。
- V2 完成前不宣称 Qwen3.5 支持 Prefix Cache；V3 完成前不宣称 Qwen3.5 支持 CUDA Graph；V4 完成前不宣称自研 GDN Decode Kernel。
- CUDA Graph 和 Prefix State Cache 的改动都必须保留 Eager/Cache-off 回退路径。
- 自研算子必须保留 FLA fallback，且不能破坏 Prefix restore 或 CUDA Graph replay。
- 不宣称支持 TP>1。

## 20. Benchmark 设计

### 20.1 工作负载

纯文本：

```text
prompt: 128 / 2K / 8K tokens
output: 64 / 256 tokens
```

图文：

```text
visual tokens: 约 256 / 1K
output: 64 / 256 tokens
```

并发：

```text
1 / 4 / 8 / 16
```

混合负载至少包括：

```text
短文本
长文本
低分辨率单图
高 visual-token 单图
```

### 20.2 对比组

1. Hugging Face Transformers + FLA。
2. nano-vLLM Qwen3.5 单请求、无 Chunked Prefill。
3. Hybrid Cache + Chunked Prefill。
4. Hybrid Cache + Decode-first Scheduler。
5. Hybrid Runtime Eager Decode vs CUDA Graph Decode。
6. Prefix State Cache 关闭 vs 开启，并按 prefix hit length/hit rate 分组。
7. GDN Decode `torch/fla/custom` 微基准与端到端对照。
8. 官方 vLLM 作为外部参考。

不要求超过官方 vLLM；官方框架的意义是帮助判断实现距离成熟系统还有多远。

### 20.3 指标

延迟：

- TTFT p50/p95/p99。
- TPOT/ITL p50/p95/p99。
- E2E p50/p95/p99。

吞吐：

- Prefill tokens/s。
- Decode tokens/s。
- requests/s。

显存：

- 模型权重。
- Paged KV。
- GDN conv state。
- GDN recurrent state。
- visual cache。
- 临时激活。
- 峰值显存。

调度：

- KV block 使用率。
- state slot 使用率。
- 抢占次数。
- 重算 token 数。
- 最长等待时间。
- Scheduler CPU 开销。

硬件：

- GPU utilization。
- Kernel 时间线。
- CPU/GPU 空洞。
- H2D/D2H 同步。

CUDA Graph：

- capture 次数、成功/失败次数和 Eager fallback 次数。
- 各 batch bucket 的 replay 次数。
- Decode CPU launch overhead、单步 latency 和 tokens/s。
- Graph pool/静态输入输出 buffer 的额外显存。

Prefix State Cache：

- lookup、hit、miss、hit rate 和实际跳过的 Prefill tokens。
- snapshot save/restore 次数及耗时。
- KV bytes、活跃 GDN state bytes 与 Prefix snapshot bytes。
- Entry 数量、LRU eviction 次数和 restore failure 次数。

GDN Decode 算子：

- 各 backend 的 Kernel latency、launch 数量和端到端 Decode 占比。
- DRAM throughput、L2 hit rate、occupancy、active warps 和 register 使用量。
- Gather/Scatter 与中间 Tensor 是否消除及对应 bytes。
- 连续/离散 `state_slot_ids`、batch size 和 Graph/Eager 对性能的影响。

### 20.4 测量规范

- 固定依赖版本和 GPU。
- 固定 prompt/output 分布。
- warm-up 后再测。
- 每组运行多次。
- CUDA Event 或显式同步保证计时正确。
- 报告中位数与尾延迟，不只挑最好的一次。
- 保留原始 CSV/JSON。
- Nsight trace 只保存关键截图和结论，不提交巨大 trace。
- CUDA Graph 必须分别报告 warm-up/capture 成本和稳定 replay 性能。
- Prefix Cache 必须分别报告冷启动、首次写入、稳定命中和淘汰后重算。
- 算子必须报告真实 Qwen3.5-9B shape，并同时给出微基准和端到端结果。

### 20.5 调度优化目标

目标是在混合长短请求下：

```text
p95 TPOT 相对原 Prefill-first 至少改善 15%
p95 TTFT 退化控制在 10% 内
```

这只是实验目标，不是简历中预先写好的结论。最后必须填写实测数据；如果没有达到，要解释瓶颈和退化场景。

### 20.6 Prefix Cache、CUDA Graph 与 GDN 算子优化目标

CUDA Graph 的目标不是改变模型数学结果，而是降低 Decode 阶段重复 Kernel launch 和 Python 调度开销。重点观察小 batch/短 Decode step；若算子内部仍有动态分配或同步，必须记录 fallback 原因，不能只报告成功场景。

Prefix State Cache 的目标是减少共享长前缀请求的实际 Prefill token 数和 TTFT。收益必须和 snapshot 显存、保存/恢复拷贝开销一起报告；不把高命中率等同于必然加速。

GDN 自研算子的目标是减少动态 state-slot Gather/Scatter、FP32 recurrent state 的重复 HBM 往返和 Kernel Launch。必须以 FLA 为强基线，并同时报告组件加速与端到端收益；不能把 Kernel 微基准加速直接写成模型吞吐提升。

## 21. V1 10 周与 V2/V3/V4 扩展实施路线

### 第 1 周：环境与基线

- 建立干净功能分支。
- 保存依赖版本。
- 验证 Transformers Qwen3.5。
- 验证 FLA 和 causal-conv1d 在 SM120 上运行。
- 建立 Hugging Face baseline。
- 保存 0.8B 固定输入和 golden 输出。

交付物：环境报告、最小 baseline 脚本、版本锁定草案。

### 第 2 周：Registry、配置、Loader、Greedy

- 模型 Registry。
- 嵌套 `text_config/vision_config`。
- 严格 Loader 框架。
- `temperature=0` greedy。
- CPU 单测与 golden-test 工具。

交付物：Qwen3 回归通过，Qwen3.5 配置能够被正确识别。

### 第 3 周：GDN reference 与 FLA

- PyTorch recurrent reference。
- causal convolution state。
- FLA chunk prefill。
- FLA recurrent decode。
- 单层输出与最终状态对齐。

硬门槛：单 GDN 层通过数值测试。

### 第 4 周：Full Attention 与 Hybrid Cache

- Qwen3.5 Full Attention。
- Q/K Norm。
- attention output gate。
- partial RoPE。
- HybridCacheSpec。
- compact Paged KV。
- GDN State Pool。

交付物：缓存形状、显存公式和生命周期测试。

### 第 5 周：0.8B 纯文本 Runtime

- Decoder layers。
- 整体 TextModel。
- Paged KV 绑定。
- GDN state 读写。
- Prefill/Decode。
- Chunked Prefill。
- greedy token 对齐。

硬门槛：0.8B 纯文本生成与 Hugging Face 对齐。

### 第 6 周：Vision Tower 与图文合并

- 公共单图输入接口。
- AutoProcessor。
- Vision Patch Embedding。
- Vision blocks。
- Patch Merger。
- visual embedding 替换。
- multimodal RoPE。

交付物：组件级视觉 golden tests。

### 第 7 周：完整图文生成

- 单图完整 Prefill/Decode。
- 文本与图文 batch。
- 不同 visual token 数。
- EOS/block boundary。
- visual cache/state slot/KV 泄漏测试。

硬门槛：至少一个单图请求完整生成。

### 第 8 周：状态感知调度

- SchedulePlan。
- Decode-first。
- 剩余 budget 做 Chunked Prefill。
- state-aware admission。
- aging/starvation guard。
- 抢占与确定性重算。
- 调度指标。

交付物：原 Prefill-first 与新策略的可切换对比。

### 第 9 周：4B、Benchmark 与 Profile

- 4B 文本/单图正确性。
- 完整 workload matrix。
- TTFT/TPOT/E2E。
- 显存拆分。
- PyTorch Profiler。
- Nsight Systems。
- 消融实验。

交付物：原始数据、图表、瓶颈分析。

### 第 10 周：收尾

- 全部回归测试。
- 安装与复现说明。
- 架构图。
- 设计决策记录。
- 已知限制。
- Demo。
- 简历描述和面试问答。

### 第 11 周：Prefix Entry 与联合状态快照

- 定义 Prefix Key、Prefix Entry 和完整 block 边界。
- 保存 Full Attention KV block 引用以及全部 GDN conv/recurrent state snapshot。
- 实现原子 lookup/restore：只有 KV 与 GDN state 同时有效才算命中。
- 首版限制为纯文本请求。

硬门槛：共享前缀命中后的 state 与完整 Prefill 重算在容差内一致。

### 第 12 周：Prefix 生命周期与显存预算

- 实现 `prefix_state_cache_max_bytes` 和 checkpoint interval。
- 实现引用计数、LRU 淘汰、Entry 失效和 KV/state 联动回收。
- 接入请求完成、抢占、重算和异常退出生命周期。
- 增加 lookup/hit/miss、snapshot bytes、save/restore 和 eviction 指标。

交付物：生命周期测试、显存公式和无泄漏证明。

### 第 13 周：Prefix 正确性、Benchmark 与收尾

- 验证 Cache on/off greedy token、Chunked Prefill、batch、抢占和跨 block。
- 测试不同共享前缀长度、并发、命中率和显存预算。
- 报告 TTFT、实际执行 Prefill tokens、snapshot 拷贝成本与显存代价。
- 更新 README、架构图、复习文档、实验报告和简历描述。

交付物：可复现 Prefix Cache 实验和 V1/V2 完整项目材料。

### 第 14 周：Decode Graph-safe 审计与静态接口

- 枚举 Qwen3.5 Decode 路径中的动态分配、CPU 同步和数据相关分支。
- 为 `input_ids`、三轴 `positions`、`slot_mapping`、`context_lens`、`block_tables` 和 `state_slot_ids` 建立静态 buffer。
- 检查 Full Attention、FLA recurrent GDN 和 causal-conv1d 的 capture/replay 安全性。
- 为不安全或不支持的形状保留 Eager fallback。

交付物：Graph-safety 清单、静态 Decode 输入接口和最小 capture probe。

### 第 15 周：Decode-only CUDA Graph

- 按 `1/2/4/8/12` batch bucket warm-up 和 capture。
- replay 前把真实请求数据拷入静态 buffer。
- replay 后只读取有效 batch 范围的输出。
- 图文请求的 Vision/Prefill 保持 Eager，完成 Prefill 后复用统一 Decode Graph。
- 增加 capture/replay/fallback 统计。

硬门槛：各 bucket 的 Eager/Graph greedy token 完全一致，重复 replay 无 KV/GDN state 串槽。

### 第 16 周：CUDA Graph Benchmark 与 Profile

- 对比 Eager Decode 和 Graph Decode 的 TPOT、Decode tokens/s 与 CPU launch overhead。
- 分析 batch size、输出长度和并发对 Graph 收益的影响。
- 统计 Graph pool 和静态 buffer 的显存增量。
- 用 Nsight Systems 验证 Kernel launch gap 是否缩短。

交付物：正确性报告、原始 Benchmark 数据、Profiler 时间线和收益边界。

### 第 17 周：GDN Decode 瓶颈画像与算子规格冻结

- 在 Eager 和 CUDA Graph 两条路径中分别采集 Decode 时间线。
- 分解 state Gather/Scatter、causal-conv1d update、FLA recurrent 与其他层的占比。
- 固定 Qwen3.5-9B 的真实 shape、dtype、动态 batch 和 `state_slot_ids` 输入契约。
- 根据 Profile 选择最终融合边界，建立 `torch/fla/custom` benchmark harness。

硬门槛：必须有数据证明选定部分是可优化热点；没有证据则暂停自研 Kernel 或调整目标。

### 第 18 周：Naive State-aware CUDA Kernel

- 实现 `L=1` state-aware causal-conv1d update。
- 实现直接按 `state_slot_ids` 访问 FP32 recurrent state pool 的基础版本。
- 覆盖连续、离散、乱序 slot，以及释放后复用。
- 与 torch/FLA 比较 output、conv state 和 recurrent state。

交付物：正确但不预设高性能的 custom baseline、数值报告和非法访存检查。

### 第 19 周：Tiling、向量化与融合优化

- 按 `[K,V]` 状态矩阵设计 program/block tile。
- 融合 decay、`k^T S`、Delta Rule 更新与 `q^T S`。
- 优化向量化加载、访存合并、寄存器占用和中间 Tensor。
- 使用 Nsight Compute 分析 DRAM/L2、occupancy、active warps 和 register pressure。

交付物：Naive/Optimized/FLA 微基准及逐步优化证据。

### 第 20 周：Runtime、CUDA Graph 与端到端验证

- 在 `gated_delta_net.py` 接入 `custom` backend 和 FLA fallback。
- 直接连接 HybridStateManager 的 state pools，避免整状态 Gather/Scatter。
- 重新 capture V3 CUDA Graph，并验证 Prefix restore 后首次 Decode。
- 测量 TPOT、Decode tokens/s、requests/s 和不同 batch/slot 布局。
- 更新架构图、实验报告、复习文档和简历描述。

交付物：可复现的状态感知 GDN Decode 算子及 V1/V2/V3/V4 完整项目材料。

## 22. 止损规则

### 门槛 1：第 3 天

FLA 或 causal-conv1d 无法在 SM120 正确运行：

- 暂停模型移植。
- 先定位版本、ABI 或 Kernel 支持。
- 不用纯 PyTorch 假装性能路径已经完成。

### 门槛 2：第 3 周末

单 GDN 层无法和参考实现对齐：

- 不进入完整模型。
- 缩小输入形状，逐张量比较 Q/K/V、conv、g、beta 和 state。

### 门槛 3：第 5 周末

0.8B 文本 greedy 无法对齐：

- 暂停 Vision。
- 先完成文本模型的逐层定位。

### 门槛 4：第 7 周末

单图仍无法通过：

- 主分支发布完整的文本 Hybrid Runtime。
- 视觉代码放实验分支。
- 不让主分支保留半成品接口。

文本 Hybrid Runtime 本身已经包含模型适配、GDN 状态、异构缓存和调度，依然可以作为完整简历项目。

### 门槛 5：CUDA Graph capture 阶段

如果 FLA recurrent、causal-conv1d 或注意力路径存在无法消除的 Graph-unsafe 行为：

- 不修改数学路径来强行 capture。
- 先缩小到 Full Attention-only 或可安全捕获的 bucket。
- 记录不安全算子和同步点，保留自动 Eager fallback。
- 不把“能够调用 capture API”当成 CUDA Graph 已完成。

### 门槛 6：Prefix State Cache 恢复阶段

如果 KV 命中但 GDN snapshot 无法严格对应同一 token 边界：

- 整个候选命中作废并重新 Prefill。
- 不允许只复用 KV 或只恢复部分 GDN 层。
- 若 snapshot 显存压垮并发，增大 checkpoint interval 或降低缓存预算，而不是降低 FP32 recurrent state 精度。
- 图文 Prefix 身份未验证前，V2 只发布纯文本 Prefix Cache。

### 门槛 7：GDN Decode 算子阶段

如果 Profile 证明状态整理和 GDN recurrent/conv 不是主要 Decode 热点，或者 custom 在真实 shape 下持续显著慢于 FLA：

- 不为了简历强行宣称算子优化成功。
- 保留 Profile、Naive 实现和失败分析，重新评估融合边界。
- 不通过降低 FP32 recurrent state 精度换取未经验证的性能数字。
- 不把只在极小人工 shape 上的加速写成 Qwen3.5-9B 端到端收益。
- custom 不满足 Graph-safe 时保留 FLA/Graph 或 custom/Eager 的安全 fallback。

## 23. 最终交付物清单

源码：

- 模型 Registry。
- Qwen3.5 text/vision 模型。
- GDN torch/FLA backend。
- Hybrid CacheSpec 和 StateManager。
- 调度器。
- Decode-only CUDA Graph 管理、静态 buffer 和 Eager fallback。
- GDN-aware Prefix Entry、状态快照、恢复和 LRU 管理。
- 状态感知 causal-conv1d update 与 GDN recurrent Decode Kernel。
- `torch/fla/custom` backend 选择和安全 fallback。
- 严格 Loader。
- Benchmark 工具。

测试：

- 组件 golden tests。
- 完整模型测试。
- Runtime 边界测试。
- 抢占与泄漏测试。
- Eager/Graph Decode 一致性与 bucket 回退测试。
- Prefix Cache on/off、联合恢复、淘汰和泄漏测试。
- torch/FLA/custom 的输出、conv state、recurrent state 与 greedy token 对齐测试。
- 动态 `state_slot_ids`、slot 复用、Prefix restore 和 Graph replay 测试。
- Qwen3 回归测试。

文档：

- README。
- 架构图。
- 环境与复现说明。
- 正确性报告。
- Benchmark 报告。
- Profiler 分析。
- 已知限制。

演示：

- 纯文本请求。
- 单图请求。
- 文本/图文混合并发。
- 调度时间线或关键指标面板。
- Eager/Graph Decode 对照。
- 两个共享长文本前缀请求的 Cache miss/hit 对照。
- FLA/custom GDN Decode 微基准与端到端对照。

## 24. 简历描述模板

最终数字必须替换成实测结果。

项目标题：

> NanoHybrid-VLM：面向 Qwen3.5 混合架构的轻量级图文推理 Runtime

简历要点示例：

- 基于 nano-vLLM 接入 Qwen3.5-9B 图文模型，实现 Vision Tower、partial multimodal RoPE、Gated DeltaNet 与 Full Attention 混合 Decoder，并通过 Hugging Face golden tests 验证 Prefill/Decode 数值正确性。
- 设计 Hybrid State Cache，仅为 Full Attention 层分配 Paged KV，并为 GDN 层维护每请求 FP32 recurrent/卷积状态，支持 Chunked Prefill、Decode、状态回收和抢占重算。
- 实现 Decode-first 双 microbatch 调度和 state-aware admission，在混合长短文本/单图负载下测量 TTFT、TPOT、吞吐、抢占与显存占用；最终填入真实 p95 改善数据。
- 接入 FLA 与 causal-conv1d GPU Kernel，使用 PyTorch reference、Profiler 和逐层误差分析验证 Kernel 正确性与性能边界。

以下三条只能在对应功能完成且有原始实验数据后使用，并按实际实施顺序排列：

- 实现 GDN-aware Prefix State Cache，在完整 token-block 边界原子复用 Full Attention KV 与 GDN conv/recurrent snapshot，通过显存预算、checkpoint interval、引用计数和 LRU 控制状态开销；填入实测 TTFT、跳过 Prefill token 数和缓存显存。
- 为 Qwen3.5 Hybrid Decode 构建 batch-bucket CUDA Graph，使用静态三轴位置、Paged KV 元数据和 GDN state-slot buffer 完成 replay，并通过 Eager fallback 覆盖动态形状；填入实测 TPOT、Decode 吞吐和 launch overhead 变化。
- 基于 Nsight 定位 Qwen3.5 Decode 的 GDN 状态访问热点，设计支持离散 `state_slot_ids` 的状态感知融合 Kernel，原地完成 causal-conv state 更新及 FP32 recurrent state 的衰减、Delta Rule 写入和 Query 读取；与 FLA 对齐并填入实测 Kernel latency、HBM 流量、TPOT 和 Decode 吞吐。

不要写：

- “实现生产级 vLLM”。
- “支持所有 Qwen3.5 模型”。
- “吞吐提升 XX%”但没有脚本和原始数据。
- “自研 Gated DeltaNet Kernel”但实际调用了 FLA。
- “端到端提升 XX%”但只有孤立 Kernel 微基准。

## 25. 面试时必须能回答的问题

1. Qwen3.5 为什么不能直接复用 Qwen3 的 KV Cache 分配？
2. GDN recurrent state 的 shape 和显存公式是什么？
3. 为什么 4B 每请求约需要 49.5MiB GDN 状态？
4. 为什么只复用 Attention KV 会让 Qwen3.5 Prefix Cache 出错？
5. Chunked Prefill 如何延续 conv state 和 recurrent state？
6. Decode 为什么只输入一个 token，却仍能利用完整历史？
7. 抢占时为什么要同时释放 KV 和 GDN state？
8. 为什么抢占后选择重算而不是 CPU swap？
9. 为什么 Decode-first 能改善 TPOT？
10. 如何防止长 Prefill 饥饿？
11. FLA prefill 和 recurrent decode 分别使用什么执行方式？
12. `partial_rotary_factor=0.25` 对张量形状意味着什么？
13. Qwen3.5 Full Attention 的 q_proj 为什么不能当成普通 Qwen3 q_proj？
14. 如何证明整段 Prefill 与 Chunked Prefill 一致？
15. 如何区分数值正确、生成 token 一致和任务质量一致？
16. 为什么 Benchmark 要报告 p95/p99，而不能只报平均吞吐？
17. 在哪些 workload 下新调度策略可能没有收益甚至退化？
18. 为什么 V1 先使用 Eager，而把 CUDA Graph 放到 Hybrid Runtime 正确性之后？
19. 为什么首版 CUDA Graph 只 capture Decode，不 capture Vision 和 Prefill？
20. Graph replay 为什么要求输入地址和 shape 稳定？batch bucket 如何解决动态 batch？
21. 哪些 Tensor 必须做成静态 buffer，为什么三轴 mRoPE positions 也必须包含在内？
22. 如何证明 replay 没有把请求 A 的 KV block 或 GDN state slot 写到请求 B？
23. 为什么 Qwen3.5 Prefix Cache 必须联合保存 Attention KV 和 GDN state？
24. Prefix Entry 为什么只在完整 block 边界提交？
25. Prefix snapshot 的显存如何估算？checkpoint interval 和命中长度如何权衡？
26. KV 命中但 GDN snapshot 缺失时为什么必须整体 miss？
27. Prefix Entry 的引用计数、LRU 淘汰、抢占和请求完成如何协同？
28. 为什么 V2 先支持纯文本 Prefix Cache，而不立即缓存图文前缀？
29. 为什么项目暂不同时做 MTP、KV 量化、MoE 和 TP>1？
30. 为什么先做 Prefix Cache，再做 CUDA Graph，最后才做自研算子？
31. `state_slot_ids` 为什么不能直接用 Decode batch row 代替？
32. 自研算子为什么只从 `L=1` Decode 开始，而不先重写 Chunk Gated Delta Rule？
33. 如何证明 Gather/Scatter、HBM 往返或 Kernel Launch 是真实瓶颈？
34. 为什么 recurrent state 保持 FP32，而 Q/K/V 可以是 BF16？
35. 融合 `exp(g)`、`k^T S`、Delta 更新和 `q^T S` 后如何控制寄存器与状态 Tile？
36. 为什么 Kernel 微基准加速不等于 TPOT 或吞吐等比例提升？
37. custom backend 如何与 Prefix restore、state slot 回收和 CUDA Graph replay 协同？

## 26. 推荐的实际开工顺序

第一次真正开始写功能代码时，不要直接写 Vision Tower。严格按下面顺序：

```text
环境 Gate
→ HF 0.8B baseline
→ Registry/Config
→ Greedy/Golden harness
→ 单层 GDN torch
→ 单层 GDN FLA
→ Full Attention
→ Hybrid state lifecycle
→ 0.8B text
→ Chunked Prefill
→ Vision
→ Scheduler
→ 4B/Benchmark
→ V1 正确性与性能基线冻结
→ Prefix Entry/联合状态快照
→ LRU/预算/生命周期
→ Prefix 正确性与 Benchmark
→ Decode Graph-safe 审计
→ Decode-only CUDA Graph
→ Eager/Graph 对齐与 Benchmark
→ GDN Decode 真实瓶颈 Profile
→ Naive state-aware custom Kernel
→ Tiled/fused custom Kernel
→ Runtime/Graph 接入与端到端 Benchmark
```

原因很简单：Vision、Scheduler 和 Hybrid Cache 最终都依赖文本骨干及 GDN state 是正确的；Prefix Cache 先把 KV/GDN 的联合边界、restore 和生命周期固定下来；CUDA Graph 再围绕稳定的 Decode/恢复接口建立静态执行；自研算子最后依据真实 Eager/Graph Profile 选择融合边界，并同时验证 Cache restore 与 Graph replay。按这个顺序可以避免 Kernel 接口反复推倒，也让每一层优化都有可比较基线。

## 27. 项目成功标准

这个项目成功不取决于代码行数，也不取决于是否超过官方 vLLM。

成功标准是：

1. 能解释 Qwen3.5 两类层分别保存什么历史状态。
2. 能独立接入官方 checkpoint，而不是依赖私有权重转换。
3. 能证明整段 Prefill、Chunked Prefill 和 Decode 的状态连续性。
4. 能正确管理 KV block 和 GDN state slot 的分配、释放与抢占。
5. 能跑通文本和单图请求。
6. 能用 golden test 证明正确性。
7. 能用可复现实验说明性能收益和退化边界。
8. 能在面试中从代码、张量形状、显存公式、调度策略和 Profile 五个角度讲清楚项目。
9. 能在固定 batch bucket 下正确 capture/replay Hybrid Decode，并在不支持的形状上安全回退 Eager。
10. 能证明 Prefix 命中原子恢复 Full Attention KV 与全部 GDN state，且结果等价于完整重算。
11. 能量化 CUDA Graph 的 launch-overhead/显存代价，以及 Prefix Cache 的 TTFT/snapshot 显存权衡。
12. 能用 Profile 证明自研 GDN Decode 算子针对真实热点，而不是孤立教学练习。
13. 能让 custom backend 正确处理动态 `state_slot_ids`、FP32 recurrent state、Prefix restore 和 CUDA Graph replay。
14. 能同时报告 Kernel 微基准与端到端收益，并解释两者不一致的原因。
