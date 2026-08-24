# NanoHybrid-VLM：Qwen3.5 图文推理与混合状态缓存

> 文档性质：项目设计与实施路线，不代表当前仓库已经实现这些功能。
>
> 建议周期：8～10 周。
>
> 开发设备：单张 RTX 5090 32GB，BF16，TP=1，Eager 模式。

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
生成文本与性能数据
```

项目对外应描述为：

> 基于轻量级推理引擎实现 Qwen3.5 Hybrid Runtime，并完成异构状态管理、图文 Prefill/Decode、并发调度和可复现性能分析。

不要描述成“实现了生产级 vLLM”或“重写了 vLLM”。

## 2. 为什么这个项目适合写进 AI Infra 实习简历

这个项目可以同时覆盖推理框架岗位常问的四类能力：

1. 模型结构：Transformer、GQA、RoPE、Vision Transformer、Gated DeltaNet。
2. 推理状态：KV Cache、递归状态、Chunked Prefill、Decode、抢占与重算。
3. 推理系统：Continuous Batching、token budget、显存准入和长短请求公平性。
4. 性能工程：GPU Kernel 接入、Profiler、TTFT/TPOT、吞吐和显存分析。

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
总结收益、退化场景与限制
```

## 3. 模型与范围

### 3.1 模型选择

- 开发与频繁调试：`Qwen/Qwen3.5-0.8B`。
- 最终正确性与性能验证：`Qwen/Qwen3.5-4B`。

0.8B 用来缩短迭代时间，但 0.8B 和 4B 的核心架构一致，因此代码不能写死某一个模型的层数和维度。

| 配置 | Qwen3.5-0.8B | Qwen3.5-4B |
| --- | ---: | ---: |
| Text hidden size | 1024 | 2560 |
| Text layers | 24 | 32 |
| GDN layers | 18 | 24 |
| Full Attention layers | 6 | 8 |
| Full Q heads | 8 | 16 |
| Full KV heads | 2 | 4 |
| Full Attention head dim | 256 | 256 |
| GDN key heads | 16 | 16 |
| GDN value heads | 16 | 32 |
| GDN key/value head dim | 128 / 128 | 128 / 128 |
| Vision depth | 12 | 24 |

### 3.2 首版必须完成

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

### 3.3 首版明确不做

- 视频和多图输入。
- 网络 URL 图片下载。
- TP 大于 1。
- Sequence Parallel。
- CUDA Graph。
- MTP 投机解码。
- FP8/INT8 KV Cache。
- Qwen3.5 Prefix Cache。
- CPU Swap。
- 自己重写 Gated DeltaNet 底层 Kernel。

这些限制不是缺点，而是为了保证 8～10 周能够完成一个正确、可测、能讲清楚的项目。

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

## 16. 正确性测试矩阵

### 16.1 组件级

- Processor 与 visual token 数。
- Vision Patch Embedding。
- Vision block 与 Patch Merger。
- Full Attention 输出。
- Q/K Norm、partial mRoPE、output gate。
- GDN 卷积输出。
- GDN recurrent 输出与最终 state。
- Decoder block。
- LM Head。

### 16.2 模型级

- 0.8B 纯文本完整前向。
- 0.8B 单图完整前向。
- 4B 文本 smoke test。
- 4B 单图 smoke test。
- greedy token 序列。

### 16.3 Runtime 级

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

### 16.4 回归要求

Qwen3.5 的重构不能破坏原 Qwen3：

- Qwen3 原有生成仍能运行。
- Qwen3 Prefix Cache 行为不变。
- Qwen3 eager 路径结果不变。
- 不宣称 Qwen3.5 支持 CUDA Graph 或 TP。

## 17. Benchmark 设计

### 17.1 工作负载

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

### 17.2 对比组

1. Hugging Face Transformers + FLA。
2. nano-vLLM Qwen3.5 单请求、无 Chunked Prefill。
3. Hybrid Cache + Chunked Prefill。
4. Hybrid Cache + Decode-first Scheduler。
5. 官方 vLLM 作为外部参考。

不要求超过官方 vLLM；官方框架的意义是帮助判断实现距离成熟系统还有多远。

### 17.3 指标

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

### 17.4 测量规范

- 固定依赖版本和 GPU。
- 固定 prompt/output 分布。
- warm-up 后再测。
- 每组运行多次。
- CUDA Event 或显式同步保证计时正确。
- 报告中位数与尾延迟，不只挑最好的一次。
- 保留原始 CSV/JSON。
- Nsight trace 只保存关键截图和结论，不提交巨大 trace。

### 17.5 调度优化目标

目标是在混合长短请求下：

```text
p95 TPOT 相对原 Prefill-first 至少改善 15%
p95 TTFT 退化控制在 10% 内
```

这只是实验目标，不是简历中预先写好的结论。最后必须填写实测数据；如果没有达到，要解释瓶颈和退化场景。

## 18. 10 周实施路线

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

## 19. 止损规则

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

## 20. 最终交付物清单

源码：

- 模型 Registry。
- Qwen3.5 text/vision 模型。
- GDN torch/FLA backend。
- Hybrid CacheSpec 和 StateManager。
- 调度器。
- 严格 Loader。
- Benchmark 工具。

测试：

- 组件 golden tests。
- 完整模型测试。
- Runtime 边界测试。
- 抢占与泄漏测试。
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

## 21. 简历描述模板

最终数字必须替换成实测结果。

项目标题：

> NanoHybrid-VLM：面向 Qwen3.5 混合架构的轻量级图文推理 Runtime

简历要点示例：

- 基于 nano-vLLM 接入 Qwen3.5-0.8B/4B 图文模型，实现 Vision Tower、partial multimodal RoPE、Gated DeltaNet 与 Full Attention 混合 Decoder，并通过 Hugging Face golden tests 验证 Prefill/Decode 数值正确性。
- 设计 Hybrid State Cache，仅为 Full Attention 层分配 Paged KV，并为 GDN 层维护每请求 FP32 recurrent/卷积状态，支持 Chunked Prefill、Decode、状态回收和抢占重算。
- 实现 Decode-first 双 microbatch 调度和 state-aware admission，在混合长短文本/单图负载下测量 TTFT、TPOT、吞吐、抢占与显存占用；最终填入真实 p95 改善数据。
- 接入 FLA 与 causal-conv1d GPU Kernel，使用 PyTorch reference、Profiler 和逐层误差分析验证 Kernel 正确性与性能边界。

不要写：

- “实现生产级 vLLM”。
- “支持所有 Qwen3.5 模型”。
- “吞吐提升 XX%”但没有脚本和原始数据。
- “自研 Gated DeltaNet Kernel”但实际调用了 FLA。

## 22. 面试时必须能回答的问题

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
18. 为什么项目首版不同时做 MTP、KV 量化和 CUDA Graph？

## 23. 推荐的实际开工顺序

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
```

原因很简单：Vision、Scheduler 和 Hybrid Cache 最终都依赖文本骨干及 GDN state 是正确的。如果第 5 周前文本路径没有对齐，继续向上叠功能只会让错误更难定位。

## 24. 项目成功标准

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
