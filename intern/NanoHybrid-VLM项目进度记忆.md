# NanoHybrid-VLM 项目进度记忆

> 更新时间：2026-08-27  
> 用途：记录项目已经实现和验证的技术内容、当前开发位置、剩余任务，以及中断后恢复工作的入口。  
> 总体完成度：约 45%。如果只计算文本 Hybrid Runtime，约完成 75%～80%。

## 1. 项目最终目标

基于 nano-vLLM 实现 Qwen3.5 Hybrid Runtime，使推理引擎能够支持 Qwen3.5 的混合层结构：

```text
3 × Gated DeltaNet + 1 × Full Attention
```

最终目标包括：

1. 支持 Qwen3 和 Qwen3.5 的模型注册与自动选择。
2. 支持 Qwen3.5-0.8B 和 Qwen3.5-4B。
3. 为 Full Attention 层维护 Paged KV Cache。
4. 为 GDN 层维护 depthwise causal convolution state 和 recurrent state。
5. 支持文本 Prefill、Chunked Prefill 和 Batched Decode。
6. 实现 Decode-first、状态感知的调度策略。
7. 接入 Qwen3.5 Vision Tower、图像 embedding 合并和 multimodal RoPE。
8. 支持单图图文请求。
9. 与 Hugging Face/Transformers 逐组件、逐 token 对齐。
10. 完成 4B 验证、Benchmark、Profiler 分析和最终实验报告。

首版范围：

- 单卡 RTX 5090。
- BF16、TP=1、Eager 模式。
- 每个请求最多一张本地 PIL Image。
- Qwen3.5 Prefix Cache、MTP、CUDA Graph、视频、多图和 TP>1 暂不支持。

## 2. 当前已经实现的技术内容

### 2.1 Hugging Face 基线和 Golden 数据

- 已经能够使用 Transformers 加载本地 Qwen3.5-0.8B。
- 已建立 HF 文本 greedy baseline，用于保存参考 token 输出。
- 已使用 HF cache probe 检查 Qwen3.5 的混合缓存结构。
- Golden 数据的作用是作为后续自研 Runtime 的正确性参考，而不是训练数据。

本地 Qwen3.5-0.8B 模型路径：

```text
/workspace/.cache/huggingface/hub/
models--Qwen--Qwen3.5-0.8B/snapshots/
2fc06364715b967f1860aea9cf38778875588b17
```

现有 Golden 文件：

```text
artifacts/golden/qwen35_08b_text_greedy.json
```

### 2.2 Config 配置适配

`nanovllm/config.py` 已实现：

- 区分完整 Hugging Face 根配置 `hf_config`。
- 读取语言模型配置 `text_config`。
- 读取视觉模型配置 `vision_config`。
- 从 `text_config.layer_types` 识别 GDN 层和 Full Attention 层。
- 增加 `num_state_slots`，控制 GPU 上能够同时保存多少条请求的 GDN 状态。
- 增加 `gdn_state_memory_fraction`，用于自动计算状态池显存预算。
- 对带 GDN 的 Qwen3.5 自动关闭 Prefix Cache。
- 如果用户强制为 GDN 模型开启 Prefix Cache，则直接报错，避免只复用 KV、没有复用对应 GDN state 而产生错误结果。

尚未加入：

```text
max_prefill_wait_ms
```

它将在 Decode-first Scheduler 阶段加入。

### 2.3 模型 Registry

`nanovllm/models/registry.py` 已实现模型注册和惰性导入：

- Qwen3 继续映射到原来的 `Qwen3ForCausalLM`。
- Qwen3.5 根据 Hugging Face 的 `model_type=qwen3_5` 和 architecture 自动映射到 `Qwen3_5ForConditionalGeneration`。
- 模型模块只在真正选择该模型时导入，避免启动时加载所有模型依赖。
- `ModelRunner` 不再硬编码只能创建 Qwen3。

### 2.4 真正的 greedy decoding

`SamplingParams` 已允许：

```python
temperature=0
```

Sampler 在温度为 0 时执行 `argmax`，从而实现真正的确定性 greedy decoding。该能力用于：

- 与 Hugging Face 逐 token 对齐。
- 检查 Batch 和单请求是否一致。
- 检查抢占重算是否改变生成结果。

### 2.5 Qwen3.5 文本模型结构

`nanovllm/models/qwen3_5.py` 已实现文本侧主要结构：

- Qwen3.5 Full Attention。
- GQA。
- Q/K Norm。
- Attention output sigmoid gate。
- Partial RoPE。
- Qwen3.5 RMSNorm 权重偏移语义。
- SwiGLU MLP。
- DecoderLayer 残差连接。
- 根据 `layer_types` 选择 GDN 或 Full Attention。
- 共享 Embedding/LM Head。
- Qwen3.5 顶层 Causal LM 包装。

当前只完成文本模型，尚未实现 Vision Tower 和三轴 multimodal RoPE。

### 2.6 Gated DeltaNet

`nanovllm/layers/gated_delta_net.py` 已实现：

- Q/K/V/Z、beta、gate 等投影。
- `mixed_qkv` 的切分与形状转换。
- Depthwise causal convolution。
- `conv_state` 的读取和更新。
- `recurrent_state` 的读取和更新。
- FLA chunk 路径，用于处理 Prefill 中的多个 token。
- FLA fused recurrent 路径，用于处理已有状态下的单 token Decode。
- PyTorch delta rule 路径，用于数值理解和排障。

GDN 的核心状态：

```text
conv_state
    保存短卷积所需的最近历史。

recurrent_state
    保存 Gated Delta Rule 压缩后的长期键值映射状态。
```

当前 recurrent state 按项目方案使用 FP32。

### 2.7 权重加载

`nanovllm/utils/loader.py` 已适配 Qwen3.5 checkpoint 权重命名，并实现：

- safetensors 权重加载。
- Packed projection shard 的加载跟踪。
- 缺失权重和未知权重检查。
- 明确忽略当前首版不使用的 Vision/MTP 权重前缀。
- 其他未知 checkpoint 权重仍然报错。

### 2.8 Hybrid CacheSpec 和状态池

`nanovllm/engine/hybrid_state.py` 已实现：

- `GDNLayerState`。
- `HybridCacheSpec`。
- `StateSlotAllocator`。
- `HybridStateManager`。

`HybridCacheSpec` 根据 `layer_types` 推导：

- 哪些全局层是 Full Attention。
- 哪些全局层是 GDN。
- 全局层编号到紧凑缓存编号的映射。
- Full Attention KV Cache 形状。
- GDN conv state 形状。
- GDN recurrent state 形状。
- 每个 state slot 的显存字节数。
- 每个 KV block 的显存字节数。

Qwen3.5-0.8B 当前推导结果约为：

```text
18 个 GDN 层
6 个 Full Attention 层
每个 GDN state slot 约 18.84 MiB
block_size=256 时，每个 KV block 约 3 MiB
```

GPU 状态池的逻辑形状：

```text
conv_state_pool:
[num_slots, num_gdn_layers, conv_dim, kernel_size]

recurrent_state_pool:
[num_slots, num_gdn_layers, value_heads, key_dim, value_dim]
```

状态池支持：

- 单请求状态读取和写回。
- slot 初始化标记。
- slot 重置和重复利用。
- 多请求状态 Gather。
- 多请求状态 Scatter。
- 重复 slot、错误形状、错误 dtype 和错误 device 检查。

### 2.9 Hybrid KV Cache 显存分配

`nanovllm/engine/model_runner.py` 已实现：

- 通过 warmup 测量运行时临时激活显存。
- 根据 `gpu_memory_utilization` 计算缓存预算。
- 先为固定数量的 GDN state slots 分配显存。
- 剩余显存用于 Paged KV Cache。
- Qwen3.5 只为 6 个 Full Attention 层分配 KV Cache，不再假设 24 层都需要 KV。
- 将紧凑 KV Cache 中对应的 K/V 张量绑定到具体 Full Attention 层。

### 2.10 Scheduler 的 Hybrid 资源生命周期

当前 `nanovllm/engine/scheduler.py` 已实现：

- 新请求必须同时能够获得 KV blocks 和 GDN state slot，才能进入 Prefill。
- Chunked Prefill 期间请求继续持有同一个 state slot。
- Decode 期间请求继续持有自己的 KV blocks 和 state slot。
- 请求完成时同时释放 KV blocks 和 state slot。
- 抢占时同时释放 KV blocks 和 state slot。
- 被抢占请求将 `num_cached_tokens` 重置为 0，恢复时从 token 0 重新 Prefill，以确定性重建 KV 和全部 GDN state。

当前 Scheduler 仍然是旧的 Prefill-first 策略，尚未实现 Decode-first 双 microbatch。

### 2.11 Qwen3.5 文本 Prefill/Decode

当前文本 Runtime 已能够完成：

```text
prompt tokenization
    ↓
创建 Sequence
    ↓
分配 KV blocks + GDN state slot
    ↓
Prefill
    ↓
保存 Full Attention KV 和 GDN 最终状态
    ↓
反复 Decode
    ↓
采样 token
    ↓
EOS/max_tokens 结束
    ↓
释放 KV blocks 和 state slot
```

EOS 已改为优先读取 `text_config.eos_token_id`，避免错误地直接使用 tokenizer 的另一个特殊 token 作为模型终止条件。

### 2.12 Batched Decode

当前已经实现真正的多请求 Batched Decode：

1. Scheduler 选出多条正在生成的 Sequence。
2. `prepare_decode()` 将每条请求的 last token 组成 `input_ids[B]`。
3. 为 Full Attention 构造每条请求独立的 `slot_mapping`、`context_lens` 和 `block_tables`。
4. 根据每条请求持有的 state slot，Gather GDN 状态。
5. DecoderLayer 将 Decode 输入从 `[B,H]` 转换成 GDN Kernel 所需的 `[B,1,H]`。
6. Qwen3.5 模型通过一次前向同时处理 B 条请求。
7. 得到 `[B,vocab_size]` logits，并为每条请求采样下一个 token。
8. 将更新后的 GDN state Scatter 回各请求的固定 state slot。

技术效果：

```text
旧路径：每轮 Decode 调用 B 次完整模型前向
新路径：每轮 Decode 调用 1 次带 batch 维的完整模型前向
```

## 3. 已完成的正确性验证

### 3.1 单请求 HF Golden 对齐

已经使用 Golden prompt token IDs 执行：

```text
temperature=0
max_tokens=36
ignore_eos=True
```

结果：

```text
36/36 generated token IDs 完全一致
first_mismatch=None
```

这证明当前至少有一条短文本请求的 Prefill 和重复 Decode 与 HF greedy top-1 输出一致。

它尚不能证明：

- 所有 logits 完全一致。
- 所有输入长度都一致。
- Chunked Prefill 一致。
- 跨 KV block 一致。
- 抢占重算一致。
- 图文模型一致。

### 3.2 Batched Decode 一致性

已使用三条请求进行 greedy 检查：

```text
你是什么模型？
写出100以内所有的素数，只要结果。
你是什么模型？
```

结果：

```text
batch_vs_solo [True, True, True]
duplicate_equal True
```

含义：

- 三条请求一起 Decode 的 token IDs，与三条请求分别单独 Decode 完全一致。
- Batch 中两条相同 prompt 的 greedy token IDs 完全一致。
- 当前测试没有发现 GDN state 串请求或 KV Cache 串请求。

`example.py` 当前使用 `temperature=0.6`，所以两条相同 prompt 生成不同文本属于随机采样的正常现象，不代表状态串线。

## 4. 当前代码执行方式的准确描述

### Prefill

当前 Scheduler 可以在一轮中选中多条 Prefill Sequence，但 Hybrid ModelRunner 仍然通过 Python 循环逐 Sequence 执行模型前向：

```text
Prefill A：一次模型前向
Prefill B：一次模型前向
Prefill C：一次模型前向
```

因此当前不是 Variable-length Batched Prefill。

### Decode

Decode 已经是真正的 Batch：

```text
Decode(A, B, C)：一次模型前向
```

如果某条请求结束，下一轮 batch 会自动缩小；长期状态仍由固定 state slot 标识，不依赖每轮变化的 batch index。

## 5. 当前正在进行的 Part

当前准备开始：

```text
Decode-first 状态感知调度
```

这一 Part 分成 4 个小节：

1. 定义 `SchedulePlan`，使一次逻辑调度同时记录 Decode 和 Prefill microbatch。
2. 重写 `Scheduler.schedule()`：先调度 Decode，再将剩余 token budget 分配给 Chunked Prefill。
3. 重写 `LLMEngine.step()`：在一次逻辑 step 中依次执行两个 microbatch。
4. 加入 `max_prefill_wait_ms` 饥饿保护、抢占策略和正确性验证。

截至本文档创建时：

- `SchedulePlan` 尚未写入 `scheduler.py`。
- `max_prefill_wait_ms` 尚未写入 `config.py`。
- `LLMEngine.step()` 仍只支持一次运行一种 microbatch。

下次继续时，从以下修改开始：

```text
nanovllm/engine/scheduler.py
    添加 SchedulePlan dataclass

nanovllm/config.py
    添加 max_prefill_wait_ms
```

## 6. 后续剩余任务

### 6.1 Decode-first Scheduler

- `SchedulePlan`。
- Decode microbatch 优先。
- 剩余 token budget 执行 Chunked Prefill。
- Prefill 最长等待时间和强制 chunk。
- KV blocks 与 state slots 联合 admission。
- 基于重算成本的抢占策略。
- 排队、抢占和重算统计。

### 6.2 Chunked Prefill 验证

- 整段 Prefill 与多个 chunk 的最终 logits 一致。
- 整段 Prefill 与多个 chunk 的 conv state 一致。
- 整段 Prefill 与多个 chunk 的 recurrent state 一致。
- chunk 跨越 KV block 边界时一致。

### 6.3 抢占与资源生命周期验证

- 抢占后确定性重算 token 一致。
- 请求完成后 KV blocks 全部释放。
- 请求完成后 state slot 全部释放。
- state slot 能被后续请求安全重复使用。
- 不发生视觉缓存、KV 或 GDN state 泄漏。

### 6.4 Vision 和图文输入

- 使用 AutoProcessor 处理本地 PIL Image。
- 实现 Qwen3.5 Vision Transformer。
- 实现 Patch Merger 和视觉投影。
- 将视觉 embedding 替换到图像 token 位置。
- 实现三轴 multimodal RoPE。
- 支持一条请求最多一张图片。
- 支持文本请求和图文请求混合调度。

### 6.5 4B 和性能实验

- Qwen3.5-4B 文本验证。
- Qwen3.5-4B 单图验证。
- HF、基础 nano-vLLM、Hybrid Cache、Chunked Prefill、Decode-first Scheduler 对比。
- TTFT、TPOT、E2E、吞吐和显存统计。
- GPU 利用率和 Scheduler CPU 开销。
- PyTorch Profiler/Nsight Systems 时间线。
- 只根据实测结果撰写性能提升数字。

## 7. 当前进度估计

| 模块 | 当前完成度 |
|---|---:|
| 环境、HF Golden 基线 | 90% |
| Config 与模型 Registry | 100% |
| Qwen3.5 文本模型 | 85% |
| GDN 执行路径 | 80% |
| Hybrid State Cache | 80% |
| 文本 Prefill/Decode | 80% |
| Batched Decode | 90% |
| Decode-first Scheduler | 20% |
| Chunked Prefill 完整验证 | 40% |
| 抢占与确定性重算 | 40% |
| Vision/图文输入 | 0% |
| Multimodal RoPE | 0% |
| Qwen3.5-4B 验证 | 0% |
| Benchmark/Profiler | 5% |
| 项目报告和简历材料 | 10% |

整体项目约完成 45%。

## 8. 当前可用于面试的技术总结

目前可以准确表述为：

> 基于 nano-vLLM 接入 Qwen3.5 的 Gated DeltaNet/Full Attention 混合文本模型，根据 layer types 为 Full Attention 层分配紧凑 Paged KV Cache，并为 GDN 层设计固定 state-slot 状态池，分别管理 depthwise causal convolution state 和 FP32 recurrent state。实现多请求 GDN 状态 Gather、`[B,1,H]` Batched Decode 和状态 Scatter 回写，将每轮请求级模型调用从 B 次合并为一次，并通过 greedy 模式验证 Batch 与逐请求执行的 token 级一致性。

暂时不能声称：

- 已完成图文推理。
- 已完成 Decode-first Scheduler。
- 已完成生产级调度。
- 已超过官方 vLLM 性能。
- 已经测得固定百分比的性能提升。

这些结论必须等对应实现和 Benchmark 完成以后再更新。
