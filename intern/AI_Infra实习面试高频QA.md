# AI Infra 推理与算子实习面试高频 Q&A

> 以 NanoHybrid-VLM 为项目主线，重点准备推理系统与 GPU/CUDA 算子，兼顾框架、C++ 和分布式。
>
> 当前项目处于设计/实施准备阶段。`[设计中]` 表示尚未验证，`[待实测]` 必须换成真实数据，只有具备代码、测试和复现材料后才能写 `[已验证]`。
>
> **2026-08-29 进度补充**：原有问答作为项目设计阶段的复习记录保留；当前已经跑通 Qwen3.5-9B 文本/单图 Hybrid Runtime、Variable-length Batched Prefill、Batched Decode、Decode-first 调度、视觉缓存、抢占重算和 Hugging Face 图文 64-token greedy 对齐。CUDA Graph 与 GDN-aware Prefix State Cache 已纳入后续方案，但仍未实现；MTP、MoE、TP>1、KV Cache 量化仍不能写成项目成果。下面新增的“面经补充”以当前真实代码为准。

## 使用方法

- `S`：必须脱稿并能追问两层；`A`：岗位门槛；`B`：定向加分。
- 项目题按“当前进度→问题→设计→验证→限制”回答。
- 每题统一关注：口述答案、原理/Shape、性能逻辑、项目联系、误区与追问。

---

## 一、NanoHybrid-VLM 项目深挖

### Q1（S）：60 秒介绍项目

**口述版**：`[设计中]` 我计划基于 nano-vLLM 接入 Qwen3.5-0.8B/4B 图文模型。它按 3 个 Gated DeltaNet 加 1 个 Full Attention 交替堆叠，不能只使用普通 KV Cache。我会只给 Full Attention 分配 Paged KV，为每个活跃请求维护 GDN 卷积状态和 FP32 recurrent state，再实现单图 Vision、partial multimodal RoPE、Chunked Prefill 和 Decode-first 双微批调度。正确性用 Hugging Face/FLA 逐层 golden test，性能报告 TTFT、TPOT、吞吐、显存、抢占和重算，当前数据均为 `[待实测]`。

**数据流**：`文本/单图→Processor/Vision→Hybrid Decoder→Paged KV+GDN State→Scheduler→Sampling`。

**误区/追问**：不能声称生产级 vLLM、自研 FLA Kernel 或虚构提升。核心贡献是异构状态生命周期与状态感知调度，而不是复制模型类。

### Q2（S）：3 分钟讲清项目

**口述版**：原 nano-vLLM 假设历史主要由 KV 表示；Qwen3.5 同时有随 token 增长的 KV 和每请求固定但很大的 GDN 状态。Chunk、Decode、batch 重排和抢占要同步维护两类状态，Prefill-first 还会阻塞在线 Decode。方案包括 Registry、PyTorch/FLA 双后端、Hybrid State Manager、状态感知 Admission、Decode-first 双微批和图文链路。验证从组件、单层、模型到 Runtime，再做固定负载的分位数实验。

**项目联系**：形成“模型结构→历史状态→资源准入→调度→指标”闭环。证明不是玩具要靠可审查提交、边界测试、原始数据、Profiler 和退化场景。

### Q3（S）：为什么选择 Qwen3.5 和 nano-vLLM？

**口述版**：普通模型适配多是注册与权重映射；Qwen3.5 的 GDN/Attention 混合结构真正改变 Cache 和调度。nano-vLLM 小而完整，适合读懂并改造核心链路；vLLM/SGLang 用作成熟设计参考与外部基线。

**误区/追问**：新模型本身不是创新；不能贬低成熟框架。0.8B 用于迭代，4B 用于最终验证，层类型必须从配置读取。

### Q4（S）：原引擎哪些假设失效？

**口述版**：原执行上下文主要是 Attention 的 `slot_mapping/context_lens/block_tables`，且历史可由 KV 表示。Qwen3.5 还需稳定 `state_slot_ids`、GDN state view、每请求本轮 token 数、chunk/decode 标记和多模态位置。

**性能/误区**：给所有层分 KV 浪费约 4 倍；按 batch index 找状态会因重排读错。只加一个模型类并不能完成 Runtime 适配。

### Q5（S）：Gated DeltaNet 如何更新？

```text
S_t = decay_t·S_(t-1)
remembered_v = K_t^T·S_t
delta = beta_t·(V_t-remembered_v)
S_t = S_t + K_t·delta^T
O_t = Q_t^T·S_t
```

**口述版**：GDN 用固定状态压缩历史；Q/K/V 前还有 causal conv1d，因此还需卷积窗口。Prefill 计划使用 FLA chunk path，Decode 使用 recurrent path。

**误区/追问**：不能遗漏 decay、beta、conv state 或 FP32 累积；复用 FLA 不能说成自研 Kernel。

### Q6（S）：GDN 状态 Shape 与显存怎么算？

```text
conv bytes = Lgdn×conv_dim×kernel_dim×2
recurrent bytes = Lgdn×Hv×Dk×Dv×4
0.8B: [18,6144,4] BF16 + [18,16,128,128] FP32 ≈ 18.84 MiB/request
4B:   [24,8192,4] BF16 + [24,32,128,128] FP32 ≈ 49.50 MiB/request
```

**性能/项目联系**：4B 预分配 512 slots 会超过 25 GB，必须独立设置 `max_num_state_slots`。FP32 是为抑制递推误差累积，降精度必须实测。

### Q7（S）：Paged KV 如何分配？

```text
[2, full_layers, blocks, block_size, kv_heads, head_dim]
0.8B: [2,6,B,256,2,256] → 12 KiB/token，3 MiB/block
4B:   [2,8,B,256,4,256] → 32 KiB/token，8 MiB/block
```

**口述版**：只为 Full Attention 分配。KV 随 token 增长，GDN state 随活跃请求数增长，所以 Admission 要同时检查 KV blocks 与 state slots。

**误区**：别漏 K/V 因子 2、层数、dtype；8 MiB 来自 `32 KiB×256`。

### Q8（S）：Hybrid State Manager 的不变量？

**口述版**：维护 BlockManager、State Pool、request→slot 映射和 free list。运行请求必须同时有合法 KV 与唯一 slot；完成、抢占、异常退出成对释放；slot 复用前清零。

**验证/追问**：循环压测后核对 free list、KV 引用计数、allocated/peak memory；必须测试异常、抢占、重复复用和分配回滚。

### Q9（S）：为什么首版关闭 Prefix Cache？

**口述版**：只恢复 Full Attention KV 而不恢复同一 prefix boundary 的 GDN conv/recurrent state，会从错误历史继续。正确缓存项必须原子关联 token hash、KV 引用和 GDN checkpoint。

**性能/误区**：checkpoint 很大，复制可能抵消命中收益。Prefix Cache 缓存的是可恢复历史状态，不只是 token。

### Q10（S）：Chunked Prefill 如何保持状态连续？

**口述版**：首 chunk 从零状态开始并写回，后续读取上一状态。整段、任意切分和逐 token 的输出以及最终 conv/recurrent state 应在容差内一致。

**性能/误区**：chunk 大阻塞 Decode，小则增加 launch/调度/state I/O。Chunked Prefill 通常不减少总 FLOPs，更不减少 Decode 计算。

### Q11（S）：Decode 为何只输入一个 token？

**口述版**：Full Attention 从 Paged KV 读历史；GDN 从 recurrent/conv state 读压缩历史。新 token 只产生新投影并更新状态。

**性能/项目联系**：Decode 常为小 GEMM、低并行、权重/Cache 访存和 launch bound；必须用稳定 `state_slot_ids` 找状态。

### Q12（S）：抢占为什么同时释放两类状态？

**口述版**：KV 与 GDN state 共同表示同一 token 边界，只保留一类会版本不一致。首版一起释放、cached token 归零，恢复时重 Prefill。

**取舍**：重算简单可靠；CPU swap 每请求要搬几十 MiB 并处理异步一致性。slot 释放后必须清零。

### Q13（S）：Full Attention 有何特殊之处？

**口述版**：`q_proj` 同时输出 Query 和 output gate；Q/K 做 head-dim RMSNorm；使用 GQA；仅部分维度 RoPE；图文用三轴 mRoPE；Attention 输出乘 `sigmoid(gate)` 后过 `o_proj`。

**Shape**：head_dim=256、partial factor=0.25，rotary_dim=64，其余 192 维不旋转。普通 Qwen3 q_proj 拆法会漏 gate。

### Q14（S）：Vision 与 mRoPE 链路？

**口述版**：AutoProcessor 负责 resize、patch、特殊 token 和 `image_grid_thw`；Vision Tower/Patch Merger 产生 visual embeddings 并替换 placeholders；图片 token 使用时间/高度/宽度三轴位置。

**验证**：processor→grid→visual token→patch→vision block→merger→合并 embedding→position→logits。图片仅在 Prefill 处理，首版抢占后重算。

### Q15（S）：Registry、配置与权重加载如何保证正确？

**口述版**：按 `model_type/architectures` 惰性选择模型，统一根/文本/视觉配置，层类型从配置推导。Loader 记录 `loaded/ignored/unexpected/missing`，首版只显式忽略 `mtp.*`，并检查 tied embedding、q_proj gate 和 GDN conv。

**误区**：不能写死维度、宽泛 try/except 或只查 unexpected keys。

### Q16（S）：Reference 与 fast path 如何分工？

**口述版**：PyTorch 小张量 FP32 循环做 oracle；FLA chunk/recurrent 和 causal-conv1d 做性能路径。二者比较投影、conv、decay/beta、逐步 state、输出和最终状态。

**误区**：只看最终文本不够；要找首次数值分叉。

### Q17（S）：Decode-first 双微批与饥饿保护？

**口述版**：`[设计中]` 每个逻辑 step 先给 running Decode 各 1 token，再用余量执行受限 Prefill chunk；首版两次调用 ModelRunner。waiting 超过 `max_prefill_wait_ms` 后强制预留 chunk，也可限制 Decode quantum。

**性能/验证**：保护 TPOT 但可能伤 TTFT；比较各延迟分位数、吞吐、最长等待、GPU 空洞和 scheduler CPU time，而不是声称绝对更快。

### Q18（S）：State-aware Admission 与抢占策略？

**口述版**：进入 running 前同时预测 KV blocks 并确认空闲 slot；失败则 waiting 或抢占，且原子回滚。首版优先抢占已计算少、重算成本低的请求，并记录 recomputed tokens。

**误区**：不能把 `max_num_seqs` 直接当 state slots。

### Q19（S）：怎样证明三种执行路径一致？

**口述版**：固定输入与 greedy，按组件、单层、模型、Runtime 分层；同一序列做整段、随机 chunk、逐 token，比较输出、最终 state、KV、logits、top-1 与首次分叉。

**指标**：max/mean absolute/relative error、NaN/Inf、top-1 一致率。文本相同不等于状态正确；数值对齐也不等于任务质量。

### Q20（S）：怎样证明 Benchmark 可信？

**口述版**：固定硬件、软件、模型、长度分布和种子；warm-up 后至少三轮；CUDA Event/同步计时；报告 P50/P95/P99、原始 CSV/JSON 和复现命令。

```text
环境：[待补充]  基线TPOT p95：[待实测]
优化后TPOT p95：[待实测]  TTFT变化：[待实测]
吞吐/Goodput/峰值显存：[待实测]
```

**误区**：只报 tokens/s、不报到达率或只挑最好一次。

### Q21（S）：哪些 workload 会退化？

**口述版**：低并发/纯 Prefill 无 Decode 可保护；短 prompt 的 chunk 收益小；持续 Decode 压力会伤 TTFT；slot 很少时瓶颈是 admission。

**追问**：按长度、visual tokens、并发和到达率分桶；用 Amdahl 与时间线解释 microbenchmark 未转化为端到端收益。

### Q22（S）：项目最难问题怎么回答？

**口述版**：`[设计中]` 预计最难的是两类历史在 chunk、重排、抢占和复用下保持同一 token 边界。先定义生命周期不变量，再用 reference/fast path 和故障测试验证。实施后必须换成真实“现象→假设→证据→修改→结果”故事，不能虚构 Bug。

### Q23（S）：哪些自己实现，哪些复用？

**口述版**：自己做 Registry/Config、权重映射、Hybrid CacheSpec、StateManager、执行上下文、调度、embedding 合并、golden/benchmark；复用 AutoProcessor、FlashAttention、FLA、causal-conv1d。

**误区**：既不能冒充自研，也不能只说“调包”；价值在正确集成、生命周期和测量。

### Q24（S）：为什么首版不做 MTP、量化、CUDA Graph？

**口述版**：它们分别改变解码状态机、数值格式和 Shape/地址稳定性，会放大交叉错误。先闭环混合状态与调度，再按 Profile 增量加入。

**追问**：Decode 为主可优先投机/Graph；容量限制并发再量化。checkpoint 有 MTP 权重不代表可直接启用。

### Q25（S）：没达到性能目标怎么办？

**口述版**：如实报告。先确认正确性与噪声，Systems 找端到端瓶颈，Compute 分析热点；说明收益 workload、退化、新开销与下一步。

**目标而非结果**：TPOT p95 改善≥15%、TTFT p95 退化≤10%，完成前只能写 `[待实测]`。

### Q26（S）：环境兼容性门禁是什么？

**口述版**：先真实执行 Transformers Qwen3.5、FLA chunk/recurrent、causal-conv1d，记录 Python/PyTorch/CUDA/Triton/ABI/SM120，不以“能 import”当成功。

**失败处理**：第 3 天门禁不过就先解决依赖，不用纯 PyTorch 假装性能路径完成。

### Q27（S）：如何做止损和最小交付？

**口述版**：GDN 单层不对齐就不做完整模型；0.8B 文本不对齐就暂停 Vision；单图失败则交付文本 Hybrid Runtime。止损不是放弃，而是保护正确的主分支和可解释范围。

### Q28（S）：如何写最终简历描述？

**口述版**：只写真实完成的模块和真实数字，注明基线、硬件、负载、P95 指标与限制。不能写“生产级”“支持全部模型”“自研 GDN Kernel”或无复现的提升。

### 面经补充 P1（S）：结合当前进度，重新完整介绍这个项目

**建议口述**：

> 我基于 nano-vLLM 实现了 Qwen3.5-9B 的文本和单图 Hybrid Runtime。Qwen3.5 的 32 层文本骨干中有 24 个 Gated DeltaNet 层和 8 个 Full Attention 层，因此历史状态不能只用 KV Cache 表示。我根据 `layer_types` 构建 `HybridCacheSpec`：只为 8 个 Full Attention 层分配紧凑 Paged KV Cache，同时通过 `HybridStateManager` 为每个活跃请求分配 `state_slot`，保存 24 层的 depthwise causal-convolution state 和 FP32 recurrent state。
>
> 执行上，我实现了 Variable-length Batched Prefill 和 Batched Decode。Prefill 把不同请求的 token chunk 打包，通过 `cu_seqlens_q/k` 和 `gdn_cu_seqlens` 描述边界；Decode 每条请求只输入 `last_token`，Full Attention 从 `block_tables` 读取 KV，GDN 从 `state_slot` 对应的状态池读取历史。调度器每轮先执行 Decode microbatch，再用剩余 `max_num_batched_tokens` 做 Chunked Prefill，并使用 `max_prefill_wait_ms` 防止 Prefill 饥饿。
>
> 多模态方面，我用官方 `AutoProcessor` 构造 `token_ids`、`pixel_values`、`image_grid_thw` 和 `mm_token_type_ids`，实现 Vision Tower、Patch Merger、三轴 mRoPE 和 visual embedding 替换。当前完成了文本/单图生成、文本/图文混合 batch、Full/Chunked Prefill 一致性、抢占重算一致性，以及 Nano 与 Hugging Face 64 个 greedy token 完全一致的验证。
>
> 性能上，我实现了请求级 TTFT、TPOT、E2E、Prefill/Decode throughput、抢占和视觉缓存统计，并比较了 Decode-first 与 Prefill-first。下一阶段是 Hybrid Decode CUDA Graph 和 GDN-aware Prefix State Cache；前者降低 Decode launch overhead，后者必须联合缓存 Attention KV 和 GDN state。

**必须主动说明的边界**：

- 当前主模型是 Qwen3.5-9B、单张 RTX 5090、BF16、TP=1。
- FLA、causal-conv1d 和 FlashAttention 是第三方 Kernel；自己的工作是模型接入、异构状态、调度、图文链路和验证。
- CUDA Graph、Qwen3.5 Prefix Cache、MTP、MoE、TP>1 尚未完成。
- 不把某次构造负载下的 49.2% 最大停顿改善说成通用提升。

### 面经补充 P2（S）：为什么选择 nano-vLLM？为什么适配 Qwen3.5？

nano-vLLM 小但仍包含 `LLMEngine`、`Scheduler`、`BlockManager`、`ModelRunner`、Paged KV、Continuous Batching 和 Prefix Cache，适合完整追踪一条请求，而不是只会调用成熟框架 API。

Qwen3 是纯 Full Attention Decoder，历史主要是每层 K/V；Qwen3.5 使用约 3:1 的 GDN/Full Attention 混合结构，还加入 Vision、partial multimodal RoPE、Q/K Norm 和 attention output gate。它迫使 Runtime 同时处理：

~~~text
随 token 数增长的 Paged KV
+ 每活跃请求固定大小的 GDN conv/recurrent state
+ 图像预处理与视觉 embedding
+ 三轴位置编码
~~~

所以这里不是只注册一个 architecture 字符串，而是修改 CacheSpec、资源准入、执行上下文、抢占和调度。

### 面经补充 P3（S）：从用户输入到最终生成 token，完整链路是什么？

~~~text
用户输入
→ InputProcessor.process()
→ ProcessedPrompt
→ LLMEngine.add_request()
→ Sequence(WAITING)
→ Scheduler.schedule()
→ KV blocks + GDN state_slot admission
→ Prefill microbatch
→ Hybrid Decoder 写 KV/GDN state
→ Sampler 生成首 token
→ Sequence(RUNNING)
→ 多轮 Decode
→ EOS/max_tokens
→ FINISHED，释放 KV/state/visual cache
~~~

`InputProcessor.process()` 对 `str` 做 tokenize，对 `list[int]` 做校验，对图文字典调用官方 `AutoProcessor`，产生 `token_ids`、`mm_token_type_ids`、`pixel_values`、`image_grid_thw`、`mrope_position_ids[3,L]` 和 `mrope_position_delta`。

`LLMEngine.add_request()` 创建 `Sequence`，保存 `seq_id/status/token_ids/num_cached_tokens/num_scheduled_tokens/block_table/state_slot` 以及多模态字段，然后 `scheduler.add(seq)` 放入 `waiting`，`waiting_since[seq_id]` 记录等待时间。

`Scheduler.schedule()` 返回 `SchedulePlan`，包含 `decode_seqs`、`prefill_seqs`、`num_decode_tokens`、`num_prefill_tokens` 和 `preempted_seq_ids`。新请求准入必须同时满足：

~~~python
num_active_seqs < max_num_seqs
num_cached_blocks != -1
can_allocate_state_slot(seq)
~~~

成功后 `BlockManager.allocate()` 写 `seq.block_table`，`allocate_state_slot()` 写 `seq.state_slot`。`prepare_prefill()` 用：

~~~python
start = seq.num_cached_tokens
end = start + seq.num_scheduled_tokens
~~~

提取当前 chunk，并构造 packed `input_ids/positions`、`cu_seqlens_q/k`、`gdn_cu_seqlens`、`slot_mapping` 和 `block_tables`。Full Attention 把 K/V 写到 `slot_mapping` 指定的物理位置，GDN 从 `state_slot` 对应状态开始并写回新状态。

Decode 中 `prepare_decode()` 只放入 `seq.last_token`；`context_lens=len(seq)`，当前 KV 写入位置为：

~~~python
seq.block_table[-1] * block_size
+ seq.last_block_num_tokens - 1
~~~

图文请求的后续文本位置为 `token_index + mrope_position_delta`。最后 `Scheduler.postprocess()` 推进 `num_cached_tokens`、追加 token；EOS 或 `max_tokens` 后释放 KV、state slot 和视觉缓存。

### 面经补充 P4（S）：WAITING、RUNNING、FINISHED 如何迁移？

~~~text
WAITING：尚未完成完整 Prefill，或抢占后等待重算
RUNNING：完整 Prefill 已完成，正在 Decode
FINISHED：遇到 EOS 或达到 max_tokens
~~~

~~~text
add_request → WAITING
WAITING → 一个或多个 Prefill chunk → RUNNING
RUNNING → 多轮 Decode → FINISHED
RUNNING → preempt() → WAITING
~~~

Chunked Prefill 请求即使仍是 `WAITING`，也可能已经持有 `block_table` 和 `state_slot`，所以 WAITING 不等于不占显存。`preempt()` 会把 `status` 改回 WAITING，释放 `block_table/state_slot`，使 `num_cached_tokens` 回到 0，恢复时从头重算，并累计 `num_recomputed_tokens`。

### 面经补充 P5（S）：ModelRunner 和 BlockManager 如何传递信息？allocate/deallocate 做什么？

两者不直接互调，中间载体是 `Sequence`：

~~~text
Scheduler 调 BlockManager
→ 物理 block_id 写进 seq.block_table
→ ModelRunner 读取 seq.block_table
→ 构造 block_tables/slot_mapping
→ Attention Kernel 读写 GPU KV
~~~

`BlockManager` 的构造参数是 `num_blocks`、`block_size` 和 `enable_prefix_cache`。主要成员有 `blocks`、`free_block_ids`、`used_block_ids`、`hash_to_block_id`；每个 `Block` 保存 `block_id/ref_count/hash/token_ids`。

- `can_allocate(seq)` 只检查，返回可复用完整 block 数，资源不足返回 `-1`。
- `allocate(seq, num_cached_blocks)` 真正占用块，把 ID 写入 `seq.block_table`，设置 `seq.num_cached_tokens`。
- `can_append()/may_append()` 处理 Decode 跨入新逻辑 block。
- `hash_blocks()` 给新完成的完整块建立 prefix hash。
- `deallocate(seq)` 对每个物理块执行 `ref_count -= 1`；只有变成 0 才放回 `free_block_ids`，最后清空 `block_table`。

token 到物理位置的公式：

~~~text
logical_block = token_index // block_size
offset = token_index % block_size
physical_block = seq.block_table[logical_block]
physical_slot = physical_block * block_size + offset
~~~

### 面经补充 P6（S）：适配 Qwen3.5 具体做了什么？最大困难是什么？

具体实现包括 Config/Registry、Qwen3.5 Full Attention、GDN 与 FLA、`HybridCacheSpec`、`HybridStateManager`、状态感知 Scheduler、Batched Gather/Scatter、Vision Tower、三轴 mRoPE、embedding 替换、视觉缓存和抢占重算。

最大困难是让两类历史始终对应同一 token 边界：

~~~text
seq.num_cached_tokens     本轮从哪里开始
seq.num_scheduled_tokens  本轮处理多少 token
seq.block_table           Full Attention KV 历史
seq.state_slot            GDN conv/recurrent 历史
~~~

一次执行成功后四者必须一起推进；抢占时必须一起回滚。只推进 `num_cached_tokens` 却没写回 GDN state，或者复用 KV 却清空 GDN state，都会产生静默错误。

验证证据包括 batch-vs-solo、Full-vs-Chunked Prefill、Nano-vs-HF 单图 64/64 greedy token、抢占前后 64 token 一致，以及结束后 KV/state/visual cache 全部释放。

### 面经补充 P7（S）：文本和图片如何融合？

融合发生在进入文本 Decoder 之前：

~~~text
token IDs → text embeddings [L,4096]

pixel_values [patches,1536]
→ Vision Tower [patches,1152]
→ Patch Merger/projection
→ visual embeddings [visual_tokens,4096]

visual embeddings 替换 text embeddings 的 image placeholder 行
→ inputs_embeds [L,4096]
→ Hybrid Decoder
~~~

只有 `mm_token_type_ids == 1` 且 token ID 等于 `image_token_id` 的行会被替换。

### 面经补充 P8（S）：Image Placeholder 数量如何确定？pixel_values 和 THW 去哪里？

`image_grid_thw=[T,H,W]`，`T×H×W` 是原始 patch 数。Patch Merger 按 `spatial_merge_size²` 合并，因此：

~~~python
raw_patch_counts = image_grid_thw.prod(dim=-1)
merge_area = spatial_merge_size ** 2
expected_image_tokens = (
    raw_patch_counts // merge_area
).sum()
~~~

代码同时统计 `placeholder_count` 和 `image_type_count`，要求：

~~~text
placeholder_count
= image_type_count
= expected_image_tokens
= visual_embeddings.shape[0]
~~~

`pixel_values` 等待时保存在 CPU 的 `Sequence.pixel_values`；第一次覆盖到 image token 的 Prefill chunk 中转到 GPU，传给 `model.get_visual_embeddings()`。`image_grid_thw` 一路传给 Vision Tower 描述 patch 网格，另一路传给 `build_qwen35_mrope_positions()` 生成三轴位置。Decode 不再处理整张图片。

### 面经补充 P9（S）：Image Embedding 怎么填回 Placeholder？

`prepare_multimodal_embeddings()` 根据 `start=num_cached_tokens` 和 `end=start+num_scheduled_tokens` 找当前 chunk 的 `local_image_positions`，再计算：

~~~text
visual_start：当前 chunk 前已有多少 image token
visual_end：visual_start + 当前 chunk image token 数
packed_image_positions：局部位置 + packed batch 偏移
~~~

先执行 `inputs_embeds=model.embed_input_ids(input_ids)`，再取得 `visual_embeddings[visual_start:visual_end]`，最后：

~~~python
inputs_embeds.index_copy_(
    0,
    destination_indices,
    selected_visual_embeddings,
)
~~~

因此图片跨多个 chunk 时只替换当前 chunk。`visual_embedding_cache[seq_id]` 保存完整视觉输出，后续 chunk 不重复运行 Vision Tower。

### 面经补充 P10（S）：RoPE 做什么？mRoPE 与普通 RoPE 什么关系？

RoPE 对 Q/K 成对通道施加由位置决定的二维旋转，使 `Q_p^T K_q` 中出现与 `p-q` 相关的相位差，从而把相对位置带入 Attention 内积。

mRoPE 将一个位置扩展为 `positions[3,L]`：temporal、height、width。文本 token 三轴相同 `[p,p,p]`；图像 token 按合并后的网格使用 `[t,h,w]`。图像占 `T×H×W` 个 token，但位置空间只前进 `max(H,W)`，所以通过：

~~~python
mrope_position_delta = (
    position_ids.max() + 1 - sequence_length
)
~~~

把后续文本 token index 映射到压缩后的 mRoPE 位置。Qwen3.5 还使用 partial RoPE；9B 的 `head_dim=256`、factor 为 0.25，所以只旋转 64 维，其余 192 维不旋转。

### 面经补充 P11（S）：Benchmark/压测怎样设计？实际结果如何解释？

测试矩阵应覆盖请求类型、prompt/output 长度、并发、token budget 和两种调度策略。`RequestMetrics` 记录 `arrival_time/enqueue_time/first_scheduled_time/first_token_time/finish_time/token_timestamps`，用于计算 preprocessing、queue、TTFT、ITL、TPOT 和 E2E；`StepStats` 记录 `num_decode_tokens/num_prefill_tokens/decode_elapsed/prefill_elapsed`。

动态干扰实验能观察长 Prefill 如何阻塞 Decode，但时序可能放大收益；闭环饱和压测维持固定 outstanding requests，更适合观察吞吐和尾延迟；真实 serving 还应补 open-loop Poisson 到达。

当前 C12、96 请求混合负载的一次结果：

~~~text
Decode-first / Prefill-first output throughput：
425.47 / 424.21 tok/s

request TPOT p95：26.49 → 26.32 ms
TTFT p95：871.13 → 842.58 ms
max token stall：1115.67 → 316.58 ms
~~~

但 token-interval p99 从 24.51 ms 增加到 199.44 ms，说明最大停顿下降的同时中等停顿可能增多。必须报告完整分布、重复多轮并保存 JSON/CSV，不能只挑一个数字。

### 面经补充 P12（S）：Block Hash 怎么算？如何映射到物理 KV Block？

普通 Qwen3 Prefix Cache 使用链式 xxHash：

~~~text
h0 = hash(block0_tokens)
h1 = hash(h0, block1_tokens)
h2 = hash(h1, block2_tokens)
~~~

`compute_hash(token_ids, prefix)` 先把前驱 hash 的 8 字节写入 xxHash，再写当前完整 block 的 token bytes；查表后还比较 `block.token_ids == token_ids`。命中时 `allocate()` 把物理 `block_id` 放入 `seq.block_table` 并增加 `ref_count`。

当前 Qwen3.5 自动关闭 `enable_prefix_cache`。未来必须把 token/hash identity、Attention KV block refs 和同边界的全部 GDN snapshot 放进同一个 Prefix Entry；任一缺失都整体 miss。

### 面经补充 P13（S）：使用 AI 时，你自己真正学到了什么？

**诚实回答模板**：

> 我用 AI 辅助源码导航、方案比较、测试脚本草拟和文档整理，但不把“代码能运行”当作掌握。我的验收标准是：每个简历动词能定位到具体文件和变量；能推导 Shape 和显存；能解释状态何时分配、写回和释放；能用 HF golden、batch-vs-solo、full-vs-chunk、抢占重算和泄漏测试否定错误实现；能脱离 AI 画出完整请求链路。

若被质疑代码由 AI 生成，不要争辩，直接解释这个字段为什么存在、去掉会错在哪里、生命周期中谁更新它、哪个测试能发现错误、当前实现还有什么边界。

### 面经补充 P14（A）：实验室方向不相关，为什么转 AI Infra？

> 实验室经历训练了我做工程实验、读论文源码和用数据验证假设的能力。我选择 AI Infra 不是只因为大模型热门，而是喜欢模型结构、系统资源和 GPU 性能的交叉问题。为了验证兴趣，我从 nano-vLLM 的一次请求链路开始，实现 Qwen3.5 Hybrid Cache、GDN state lifecycle、图文 Prefill/Decode 和调度 Benchmark，并完成 HF token 对齐和饱和压测。我发现自己更喜欢追问状态放在哪里、为什么慢、怎样证明正确。

不要贬低原方向，也不要只谈薪资。

### 面经补充 P15（A）：什么时候可以开始实习？

> 我最早可以在 `[具体日期]` 到岗，每周可以实习 `[4/5]` 天，预计持续 `[三个月/六个月]`。学校课程、导师和住宿安排已经 `[确认情况]`。如果需要，我可以提前远程完成入职材料，但正式到岗时间以上述日期为准。

不要只说“随时”，除非真的能立即到岗；不同轮次的日期必须一致。

### 面经补充 P16（S）：如何做一段适合 AI Infra 岗位的自我介绍？

控制在 60～90 秒，按“身份—证据—能力—岗位”组织：

> 面试官您好，我是 `[学校/专业/年级]` 的 `[姓名]`。我的研究/课程背景主要是 `[一句话]`，近期把重点转向大模型推理系统和 GPU 性能工程。项目上，我基于 nano-vLLM 实现了 Qwen3.5-9B 文本/单图 Hybrid Runtime，主要负责 Gated DeltaNet 与 Full Attention 的异构状态管理、Variable-length Batched Prefill、Decode-first 调度、Vision/mRoPE 接入和正确性 Benchmark；目前已经完成 Hugging Face 图文 64-token greedy 对齐、Chunked Prefill 和抢占重算一致性验证。我也在系统补 CUDA、GEMM、FlashAttention 和 Tensor Parallel，希望寻找 AI Infra/推理优化实习，把模型结构、Runtime 和 GPU 性能分析结合起来。

不要逐条念简历，也不要一上来讲十分钟技术细节。自我介绍的作用是主动把面试官引向你最熟悉、证据最完整的项目。

### 面经补充 P17（A）：什么时候开始找实习？为什么现在开始？

这道题考察求职时间线和稳定性，不要编造。使用自己的真实日期：

> 我从 `[年月]` 开始系统准备 AI Infra 实习。前期先补 Transformer 推理、CUDA 和 nano-vLLM 请求链路，之后从 `[年月]` 开始做 Qwen3.5 Hybrid Runtime，并在 `[年月]` 开始正式投递。现在开始找，是因为项目已经有可展示的端到端结果和正确性/性能数据，同时我的课程与导师安排允许连续实习 `[时长]`。

若准备时间较短，可以坦诚说明，但要用 commit、实验数据、代码理解和持续学习计划证明不是临时追热点。

---

## 二、推理引擎与框架

### Q29（S）：Decoder-only Block 数据流？
**口述**：RMSNorm→QKV→RoPE→Causal Attention→O→Residual→RMSNorm→SwiGLU→Residual。`Q:[T,Nq,D]`，`K/V:[T,Nkv,D]`。**性能/项目**：Prefill 大矩阵偏计算，Decode 小矩阵偏访存；Hybrid Block 还按 layer type 选择 GDN/Attention。**误区**：别漏 mask、residual、norm 顺序和 GQA 映射。

### Q30（S）：Prefill 与 Decode？
**口述**：Prefill 并行处理 prompt、建立历史并产生首 token；Decode 每轮新增一个 token、读取权重和历史。**性能**：Decode 串行、小 batch、访存/launch 敏感。**误区/追问**：Decode 仍计算并写新 K/V；投机解码用更多并行计算减少串行 step。

### Q31（S）：TTFT、TPOT、ITL、E2E、Goodput？
**口述**：TTFT 是到首 token；ITL 是相邻 token 间隔；`TPOT=(完成-首token)/(输出数-1)`；E2E 是总时延；Goodput 是满足 SLO 的有效吞吐。**项目**：Decode-first 主优化 TPOT 尾部，可能伤 TTFT。**误区**：区分排队、执行和网络。

### Q32（S）：在线压测怎么做？
**口述**：用 open-loop Poisson 到达，固定 prompt/output 分布与 SLO，逐步提高 arrival rate 找延迟拐点。**项目**：记录 slot/KV、queue、preemption、recompute 和 scheduler CPU。**误区**：closed-loop 会在服务变慢后自动降压；排除客户端瓶颈。

### Q33（S）：通用 KV 公式？
**口述**：`2×layers×cached_tokens×kv_heads×head_dim×dtype_bytes`，另计 batch/beam/TP 与 block 碎片。**项目**：Hybrid 只统计 Full Attention。**误区**：别漏 K/V 因子 2。

### Q34（S）：PagedAttention 与 BlockManager？
**口述**：固定物理块承载逻辑 KV，按需增长、共享和回收；BlockManager 管块表与引用。**性能**：块大内部碎片高，块小寻址开销高。**误区**：分页不消灭内部碎片；FlashAttention 优化 I/O 算法，PagedAttention 优化存储管理，可组合。

### Q35（S）：Continuous Batching？
**口述**：每个迭代边界重组 batch，完成请求退出、waiting 请求加入。**性能/项目**：提高利用率但增加调度、重排、抢占和公平性复杂度，稳定 state_slot 映射因此重要。**误区**：不等于动态 padding。

### Q36（S）：Chunked Prefill 与 Prefix Cache？
**口述**：前者改变长 prompt 调度粒度，后者复用已计算前缀、减少计算。**项目**：Qwen3.5 首版只做前者。**误区**：Chunked Prefill 不降低 Decode FLOPs；Prefix Cache 必须恢复完整历史。

### Q37（S）：MHA、MQA、GQA、MLA？
**口述**：MHA 每 Q head 独立 K/V；MQA 全 Q 共享一组；GQA 分组共享；MLA 用低维潜变量压缩 KV。**性能**：减少 KV 带宽。**误区**：MTP 是多 token 预测/投机方法，不是 Attention 类型。

### Q38（S）：FlashAttention 为什么快？
**口述**：tiling+online softmax，不把完整 `T×T` 中间矩阵写回 HBM，减少 I/O，仍是精确 Attention。**误区**：数学复杂度仍近 O(T²)，且不等于 PagedAttention。

### Q39（A）：Online Softmax？
**口述**：维护 running max `m` 和和 `l`：`m'=max(m,max(block))`，`l'=exp(m-m')l+Σexp(x-m')`。**性能**：适合分块融合，FP32 累积。**误区**：处理全 mask/`-inf`。

### Q40（A）：RoPE 与长上下文？
**口述**：成对通道按位置旋转，使 Q/K 内积携带相对位置；超训练长度会外推到未见频率。**项目**：partial mRoPE。**误区**：RoPE 不减少 KV，通常只施加于 Q/K。

### Q41（A）：Prefix Cache 哈希与引用？
**口述**：按完整 token block、模型/Adapter/多模态上下文做链式 hash，命中后增物理块引用。**项目**：混合模型还要原子关联 GDN checkpoint。**误区**：token 相同但模型、LoRA、position、图片不同不一定可复用。

### Q42（S）：投机解码状态机？
**口述**：Draft propose→Target 并行 verify→接受最长前缀→append/rollback。**性能**：减少串行 target step，收益取决于接受率、draft 和验证成本。**误区**：不能提交未接受 token 的 KV，采样还需分布修正。

### Q43（A）：Draft、n-gram、EAGLE、MTP？
**口述**：独立小模型、上下文匹配、利用 target 特征预测、多未来 token 预测头；都需 target 验证。**项目**：首版显式忽略 `mtp.*`。**误区**：MTP 不在 Attention 内。

### Q44（A）：量化方法与文件格式？
**口述**：区分权重/激活/KV、INT8/INT4/FP8、量化粒度与 scale；GPTQ 是逐层 PTQ，AWQ 保护显著权重，SmoothQuant 平衡激活与权重。**误区**：safetensors 是存储格式；省显存不保证加速，需对应 Kernel。

### Q45（S）：vLLM、SGLang、TensorRT-LLM？
**口述**：vLLM 是通用 serving；SGLang 兼顾高性能 runtime、RadixAttention 与结构化生成；TRT-LLM 面向 NVIDIA 图优化、专用 Kernel、量化和多 GPU。**误区**：固定模型、版本、精度、长度和并发再比，不能说谁绝对快。

### Q46（S）：推理框架执行链？
**口述**：API/Tokenizer→Scheduler→Block/State Manager→ModelRunner 元数据→模型/Kernel→Sampler→状态更新/释放。**项目**：主要改 State、Context、Scheduler。**误区**：只讲 forward 不算框架理解。

### Q47（A）：TP 的 Linear 怎样切？
**口述**：QKV 与 up/gate 常 column parallel 按输出维切；O/down 常 row parallel 按输入维切，局部 GEMM 后 AllReduce/ReduceScatter。**误区**：不要只背“按 head 切”，要说明后续是否可保持分片。

### Q48（A）：PD 分离？
**口述**：Prefill/Decode 放不同资源池，Prefill 后传 KV/必要状态。**性能/项目**：减少阶段干扰但增加传输、路由、一致性；Hybrid 还需传 GDN state。**误区**：必须算状态字节/带宽/额外 TTFT。

### Q49（A）：KV Cache 布局？
**口述**：layer、K/V、block、token、head、head_dim 排列决定合并访问、L2、向量化和分页寻址。**误区**：只看总字节数，不看步长、对齐和 block-table gather。

### Q50（B）：CP、Ring Attention、PCP/DCP？
**口述**：CP 按 token 维切；Ring Attention 环传 K/V 并 online softmax 累积；PCP/DCP 名称依系统而异，先确认定义。**误区**：`isend/irecv` 是异步 P2P API，不等于 TCP；Prefill 与 Decode 的通信模式不同。

### Q51（B）：DSA？
**口述**：缩写有歧义；若指 DeepSeek Sparse Attention，其目标是选择关键 token 降低长上下文 Attention。**性能**：选择、索引和不规则访存有成本。**误区**：不要与 MLA/MTP/DMA 混用。

### Q52（A）：vLLM 还能优化什么？
**口述**：先限定版本/workload，再从调度、Cache、模型特化、量化/投机、PD/多机、CPU 控制面和算子找证据，做基线→Profile→最小改动→边界分析。**误区**：没读源码、没 Profile 就提大改。

### 面经补充 I1（S）：Qwen3.5 和 Qwen3 的结构有什么区别？

**Qwen3** 可以概括为标准 Decoder-only Transformer：每层都有 Full Attention，历史主要由每层 K/V 表示。普通 nano-vLLM 因此可以给所有 Attention 层使用同构 Paged KV Cache。

**当前项目中的 Qwen3.5** 是多模态 Hybrid Decoder：

- 文本层按约 3 个 Gated DeltaNet 加 1 个 Full Attention 交替。
- GDN 层保存 `conv_state` 和 `recurrent_state`，而不是随长度保存完整 K/V。
- Full Attention 使用 GQA、Q/K RMSNorm、partial RoPE 和 attention output gate。
- 图像经过 Vision Tower/Patch Merger 后替换 image placeholders。
- 图文位置使用 temporal/height/width 三轴 mRoPE。

Runtime 影响：

~~~text
Qwen3：
每层 KV → 同构 Paged KV

Qwen3.5：
8 个 Full Attention 层 → Paged KV
24 个 GDN 层 → 每请求 state slot
~~~

所以 Qwen3.5 的准入条件、抢占、Chunked Prefill、Prefix Cache 和 CUDA Graph 都必须考虑两类状态。

### 面经补充 I2（A）：KV Cache 量化后，位置编码和 Sequence 长度怎样处理？

KV 量化只改变 K/V 的存储 dtype、scale/zero-point 和读取 Kernel，不应该改变逻辑 token 位置或 `context_lens`。

以 block-wise FP8/INT8 KV 为例，一个物理块可能保存：

~~~text
quantized_k
quantized_v
k_scale
v_scale
~~~

Decode 时依然使用原来的：

~~~text
context_lens：每条请求当前有效历史长度
block_tables：逻辑块到物理块
positions：当前 token 的逻辑位置或 mRoPE 三轴位置
~~~

RoPE 通常在把当前 K 写入 Cache 前完成；Cache 中保存已经带位置旋转的 K。量化不能把 token 位置压短，也不能因为一个量化 block 存了 256 个 token 就把 Sequence 长度改成 block 数。

如果采用滑动窗口或 KV eviction，`context_lens` 仍描述 Kernel 可访问的有效上下文，位置 ID 通常保持原始绝对/旋转位置，不能因删除旧 KV 就重新从 0 编号。不同框架对 RoPE-before-cache、scale 粒度和 window metadata 的实现会不同，回答时应先说明具体方案。

**项目边界**：当前 NanoHybrid 没有实现 KV 量化；不能把这段通用原理写成项目成果。

### 面经补充 I3（S）：KV Cache 量化为什么可能提高并发和 Decode 吞吐？精度代价是什么？

未量化 KV 字节数：

~~~text
2 × full_attention_layers × tokens
× num_kv_heads × head_dim × dtype_bytes
~~~

若从 BF16 的 2 bytes 降到 FP8/INT8 的 1 byte，理想情况下 KV 主体约减半，但还要加 scale、对齐和 block 元数据。显存释放后可以容纳更多 token 或请求，因此最大并发上升。

Decode Attention 经常受 KV 读取带宽限制。更小的 KV 可以减少 HBM bytes，所以在长上下文场景可能提高吞吐；但必须有融合的 dequant+attention Kernel。若先把整个 KV 反量化到 BF16 临时 Tensor，额外读写和 Kernel launch 可能抵消收益。

精度方面：

- K 误差会改变 attention score 和 softmax 分布；
- V 误差直接进入加权输出；
- 长上下文、异常值、低 bit 和粗粒度 scale 更敏感；
- 应比较 logits max/mean error、top-1 一致率、greedy token 和任务指标。

项目里只有 8 个 Full Attention 层使用 KV，因此量化只影响这部分；24 个 GDN 层的 FP32 recurrent state 需要另一套精度研究。

### 面经补充 I4（A）：大模型推理显存主要消耗在哪里？

至少拆成五部分：

1. **模型权重**：`参数量 × dtype bytes`；9B BF16 理论主体约 18 GB，另有对齐和模块开销。
2. **KV Cache**：随有效 cached tokens 增长。
3. **GDN state**：随活跃请求数增长，9B 每个 state slot 约 49.5 MiB。
4. **临时激活与 workspace**：Prefill、Vision、FlashAttention、FLA、cuBLAS workspace。
5. **Runtime 额外内存**：CUDA context、allocator reserved memory、CUDA Graph pool、视觉缓存、通信 buffer。

不能只看 `nvidia-smi`。项目中分别统计模型权重、Paged KV、GDN state、visual cache、Torch allocated/reserved 和峰值，才能解释为什么理论权重能放下但运行仍 OOM。

### 面经补充 I5（A）：单卡放不下模型时有哪些办法？

按问题类型选择：

- **权重太大**：INT8/INT4/FP8 权重量化、Tensor Parallel、Pipeline Parallel、CPU/NVMe offload。
- **KV 太大**：GQA/MQA/MLA、KV 量化、分页管理、Prefix 复用、滑动窗口、降低并发或上下文。
- **临时激活太大**：Chunked Prefill、缩小 token budget、FlashAttention、算子融合。
- **Hybrid state 太大**：限制 `num_state_slots`、抢占重算、状态压缩或 offload；但压缩 recurrent state 必须验证误差。

选择前先量化各部分字节数。如果 9B BF16 权重已经占主要显存，单纯减少 KV 不一定解决启动 OOM；如果权重能放下但高并发 OOM，量化权重可能不如压缩 KV/state 有效。

### 面经补充 I6（A）：Offload 是什么？工程中怎样实现？

Offload 是把暂时不用或无法常驻 GPU 的权重、KV、状态迁移到 CPU 内存或 NVMe，在使用前预取回来。常见对象有权重 offload、KV/state swap 和分层 Cache。

工程上至少需要：

~~~text
GPU/CPU buffer pool
→ pinned host memory
→ 独立 copy stream
→ CUDA Event 依赖
→ prefetch/evict 状态机
→ 请求版本和取消处理
~~~

传输时间近似 `bytes / PCIe有效带宽 + latency`。例如单个 49.5 MiB GDN state 经 PCIe 搬出再搬入，若不能和计算重叠，会直接恶化 TPOT/TTFT。抢占时还必须保证 KV 和 GDN state 对应同一 token boundary。

当前项目选择“释放后确定性重算”，因为实现简单且不会引入异步版本一致性；尚未实现 CPU swap。

### 面经补充 I7（A）：常见量化方法有哪些？如何选择？

先区分对象：

- **Weight-only**：W8A16、W4A16；主要省权重显存和带宽。
- **Weight+Activation**：W8A8、FP8；需要处理激活动态范围。
- **KV Quantization**：FP8/INT8/INT4 KV；主要省长上下文状态。
- **GDN state quantization**：针对 recurrent/conv state，递推误差风险更高。

再区分方法：

- GPTQ：逐层二阶近似的离线 PTQ。
- AWQ：保护少量显著权重通道。
- SmoothQuant：把激活难量化问题平滑转移到权重。
- 动态/静态量化：scale 是运行时计算还是校准得到。
- per-tensor/per-channel/per-group：粒度越细通常误差小，但元数据和 Kernel 更复杂。

量化是否加速取决于硬件和 Kernel；safetensors 只是文件格式。必须报告显存、吞吐、延迟和质量，而不是只说“模型从 16 bit 变 4 bit，所以快 4 倍”。

### 面经补充 I8（S）：Prefill 和 Decode 分别受什么瓶颈限制？

**Prefill** 一次处理大量 token，QKV/MLP 是大 GEMM，通常并行度高、算术强度高，更偏 compute-bound；长序列 Attention 还可能受 HBM I/O 和二次复杂度影响。优化包括 FlashAttention、较大 batch、Tensor Core、高效 GEMM 和 Chunked Prefill 控制峰值/干扰。

**Decode** 每请求每轮只有一个 token，GEMM 的 M 维很小，权重和 KV/state 读取占比高，经常 memory-bandwidth-bound 或 launch-bound。优化包括 Continuous Batching、CUDA Graph、Kernel fusion、KV 量化、GQA、投机解码和更好的调度。

Hybrid 模型还要补充：GDN Decode 读取固定 recurrent state，Full Attention Decode 读取随长度增长的 KV；两者的带宽行为不同。

### 面经补充 I9（S）：Continuous Batching 与 Chunked Prefill 怎么实现？

Continuous Batching 在每个逻辑 step 重新选择 batch，而不是等整批请求都完成。项目中的 `waiting` 和 `running` 是两个 `deque`；`schedule()` 每轮生成新的 `SchedulePlan`，完成请求退出，新请求在资源允许时加入。

Chunked Prefill 使用：

~~~python
num_tokens = seq.num_tokens - seq.num_cached_tokens
seq.num_scheduled_tokens = min(
    num_tokens,
    remaining_token_budget,
)
~~~

`num_cached_tokens` 是已经完成的前缀边界，`num_scheduled_tokens` 是本轮 chunk。执行后通过 `postprocess()` 推进。Full Attention 继续写同一 `block_table`，GDN 从原 `state_slot` 继续，因此 chunk 之间状态连续。

项目的 Decode-first 策略先给 `running` 每条请求 1 token，再用 `remaining_token_budget` 做 Prefill；`max_prefill_wait_ms` 超时后用 `reserved_seq_slots/reserved_prefill_tokens` 保证 Prefill 不饿死。

### 面经补充 I10（S）：FlashAttention 的核心过程是什么？Softmax 是什么？

标准 Attention 是：

~~~text
scores = QKᵀ / sqrt(d)
P = softmax(scores + mask)
O = PV
~~~

Softmax 把一行 logits 变成非负且和为 1 的权重：

~~~text
softmax(x_i) = exp(x_i - m) / Σ_j exp(x_j - m)
m = max_j x_j
~~~

减最大值用于数值稳定。

FlashAttention 按 Q/K/V tile 处理，不把完整 `T×T` scores 和 probability 矩阵写入 HBM。对每个 Q tile 维护 running max `m`、normalizer `l` 和部分输出 `O`；读入下一个 K/V tile 后，用 online softmax 修正旧累计量，再累加新贡献。最终只输出 `O`。

它主要解决 Attention 中间矩阵的 HBM I/O 和显存问题，数学上仍是精确 Attention，复杂度仍近 `O(T²)`。PagedAttention 解决 KV 的分页存储，两者不是替代关系。

### 面经补充 I11（A）：投机解码有哪些方案？EAGLE 系列有什么特点？

共同状态机是：

~~~text
Draft/Propose 多个 token
→ Target 一次并行 Verify
→ 接受最长合法前缀
→ 拒绝位置采样修正
→ KV/state commit 或 rollback
~~~

常见 Draft 来源：

- 独立小模型：实现直观，但额外权重和访存。
- self-speculative/layer skipping：复用主模型部分层。
- n-gram/lookahead：从上下文匹配候选，几乎无小模型成本。
- Medusa/MTP：多个预测头并行产生未来 token 候选。
- EAGLE：利用 Target 的高层特征配合轻量 Draft 模块预测下一步特征/token，通常比只基于 token 的 Draft 更接近 Target；不同 EAGLE 版本在 feature、tree drafting 和训练方式上不同。

收益取决于接受率、平均接受长度、Draft 成本、Verify batch 和内存开销。对 Hybrid 模型，未接受 token 对应的 Attention KV 与 GDN recurrent/conv state 都不能错误提交；这也是当前项目没有直接启用 checkpoint 中 `mtp.*` 权重的原因。

### 面经补充 I12（A）：大模型推理常见加速手段如何系统回答？

按层次回答比罗列名词更好：

1. **模型/精度**：权重、激活、KV 量化；GQA/MLA；蒸馏。
2. **减少串行 step**：投机解码、MTP、并行采样。
3. **状态与内存**：Paged KV、Prefix Cache、KV 压缩、offload。
4. **调度**：Continuous Batching、Chunked Prefill、Decode-first、PD 分离。
5. **执行**：CUDA Graph、Kernel fusion、FlashAttention、Triton/CUDA 专用 Kernel。
6. **并行**：TP、PP、EP、CP，多机拓扑感知通信。
7. **控制面**：异步 tokenizer、pinned memory、减少 Python 开销和高效队列。

最后必须回到 workload：低并发短输出可能最需要减少 launch；长上下文可能最需要 KV/Attention；模型放不下先解决容量；高并发 serving 再讨论调度和 Goodput。

---

## 三、CUDA 与算子优化

### Q53（A）：线程层级与 Warp 调度？
**答**：Grid→Block→Thread，32 线程组成 Warp；Block 驻留 SM 后 scheduler 从 ready warps 发射指令以隐藏延迟。活跃 warp 受寄存器、shared、threads 限制。**误区**：不是逐线程调度；高 occupancy 不保证快。

### Q54（A）：内存层次、Sector、Local Memory？
**答**：register→shared/L1→L2→global。Local memory 是线程私有地址空间但通常落在 global，常由 spill/动态数组产生；NVIDIA 常以 32B sector 统计事务。**误区**：别把 CPU 64B cache line 生搬到 GPU，local 不等于片上。

### Q55（A）：Global 搬到 Shared？
**答**：线程协同、合并且对齐地 load，普通路径后 `__syncthreads()`；可用 `cp.async`/pipeline 双缓冲并在消费前 wait/barrier。**误区**：异步不等于无需同步；边界 tile 要 mask。

### Q56（A）：Coalescing、Bank Conflict、Divergence？
**答**：分别是 warp 全局地址事务数、shared bank 串行冲突、warp 内分支路径分化。用布局/向量化、padding/swizzle、分支重排优化。**误区**：同地址广播通常不冲突；有分支不一定有严重 divergence。

### Q57（A）：Occupancy 与寄存器压力？
**答**：Occupancy 是活跃 warp/上限；寄存器、shared、block size 限制驻留。提高它可隐藏延迟，但减寄存器可能 spill、损失 ILP。**误区**：100% 不是目标。

### Q58（A）：Shuffle down/up/xor？
**答**：寄存器跨 lane 交换；down 常做 reduce，up 常做 scan，xor 做 butterfly reduction/all-reduce。跨 warp 仍需 shared/同步。**误区**：处理 active mask、非满 warp 与非 2 次幂。

### Q59（A）：Nsight Systems 与 Compute？
**答**：Systems 看 CPU/CUDA/stream/memcpy/通信/空洞；Compute 看单 Kernel 吞吐、访存、stall、occupancy、指令、Roofline。先 Systems 找热点再 Compute 深挖。采集本身会扰动。

### Q60（A）：Roofline 与算术强度？
**答**：`AI=FLOPs/DRAM bytes`，上限约 `min(peak FLOPS, AI×bandwidth)`。融合减少中间 I/O、提高 AI。**误区**：Roofline 不自动解释 launch、依赖、Cache 和指令瓶颈。

### Q61（A）：有效带宽和利用率？
**答**：`effective BW=(读+写字节)/时间`，`achieved FLOPS=操作数/时间`，再除相应 dtype/指令峰值。**误区**：必须说明字节、FLOPs 与峰值口径。

### Q62（A）：算子是否还值得优化？
**答**：先确认端到端热点，再看 Roofline 距离、带宽/吞吐、stall、spill、事务和参考库；用 Amdahl 算最大端到端收益。接近相关上界或占比很小时应停。

### Q63（A）：Triton 与 CUDA？
**答**：Triton 以 program instance/向量块表达并由编译器生成代码，开发和 autotune 快；CUDA 控制更细，适合复杂流水与极致优化。**误区**：Triton 仍需懂 coalescing、mask、occupancy 和稳定性。

### Q64（A）：Reduce Kernel？
**答**：线程局部累加→warp shuffle→跨 warp shared→单 warp 收尾；全局可两阶段/原子。向量化、合并访问、少同步、FP32 累积。**误区**：处理任意长度、空输入、非对齐，别假设 2 次幂。

### Q65（A）：Stable Softmax Kernel？
**答**：逐行 reduce max→`exp(x-max)`→reduce sum→归一化；长行分块/online。可融合 mask/scale，低精输入用 FP32 累积。**误区**：处理全 mask、`-inf`、尾元素与大值。

### Q66（A）：Tiled GEMM？
**答**：A/B tile 搬到 shared/register 复用，warp 分工算输出，双缓冲重叠加载与计算，适合时用 Tensor Core。**误区**：tile 大会压 occupancy，小则复用不足；要处理非整除并与 cuBLAS 比。

### Q67（A）：Kernel 正确性？
**答**：高精 reference；覆盖 FP16/BF16、非对齐/极端 Shape、不同 GQA、最小输入、随机种子、NaN/Inf、sanitizer。说明 atol/rtol，不能只用默认 `allclose`。

### Q68（A）：Microbenchmark 规范？
**答**：预分配、充分 warm-up、CUDA Event/同步、多轮中位数/分位数，固定 Shape/dtype/layout/版本，排除编译和 H2D。收益须超过噪声并验证端到端。

### Q69（A）：Kernel fusion 收益与代价？
**答**：减少 launch、中间 HBM 和同步；代价是寄存器压力、编译复杂和 Shape 覆盖下降。NanoHybrid 可候选 Norm+RoPE+cache write 或 residual+RMSNorm，但必须先 Profile。

### Q70（A）：Streams、Events、Pinned Memory？
**答**：Stream 内有序、不同 Stream 可并发；Event 建依赖/计时；Pinned host memory 支持高效 async copy 但应池化。**误区**：async 调用不保证重叠，还需 pinned、独立 stream、硬件引擎和无隐式同步。

### 面经补充 C1（A）：CUDA Kernel 从 Python/C++ 到 GPU 是怎样被调用的？

C++ CUDA launch 的形式是：

~~~cpp
kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
    input,
    output,
    n
);
~~~

Host 端先准备设备指针和参数，CUDA Runtime/Driver 把 launch command 放进指定 Stream。调用通常对 Host 异步返回；GPU 前端收到命令后，把 Grid 划分为 Thread Blocks，Block 被调度到有足够寄存器、shared memory 和 thread slots 的 SM，Warp Scheduler 再从 ready warps 发射指令。

同一个 Stream 内按序执行；不同 Stream 只有在资源和依赖允许时才可能并发。需要跨 Stream 顺序时用 CUDA Event，而不是默认调用 `cudaDeviceSynchronize()`。Kernel 完成后结果仍在 GPU，只有显式 D2H、`.item()/.tolist()` 或同步 API 才要求 Host 等待。

PyTorch/Triton 路径虽然语法不同，本质仍是准备 Tensor 指针、Shape/stride/constexpr，编译或选择 Kernel，然后向当前 CUDA Stream 发 launch。项目中的 `store_kvcache_kernel[(N,)](...)` 就是 Triton Grid launch。

### 面经补充 C2（A）：GEMM Kernel 常见优化有哪些？

从朴素 `C[M,N]=A[M,K]×B[K,N]` 开始，每个输出元素都重复从 Global Memory 读取 A/B，数据复用差。优化按层次展开：

1. **CTA Tiling**：一个 Thread Block 计算 `BM×BN` 输出 tile。
2. **K Tiling**：每轮只加载 `BK` 深度的 A/B tile。
3. **Coalesced/Vectorized Load**：连续线程读取连续、对齐地址，使用 `float4` 或合适向量宽度。
4. **Shared Memory Reuse**：A tile 被 BN 方向线程复用，B tile 被 BM 方向线程复用。
5. **Thread/Warp Tiling**：每线程计算多个 C 元素，把累加器放寄存器。
6. **Double Buffer/Pipeline**：计算当前 K tile 时用 `cp.async` 或异步流水预取下一 tile。
7. **Bank-conflict Avoidance**：padding、转置布局或 swizzle。
8. **Tensor Core**：使用 WMMA/MMA/WGMMA 等矩阵指令，并满足 dtype/layout/alignment。
9. **Epilogue Fusion**：融合 bias、activation、scale，减少额外 HBM 往返和 launch。
10. **Shape-aware Dispatch**：不同 M/N/K、对齐和 dtype 使用不同 tile/config。

代价是 tile 越大，shared memory 和寄存器越多，可能降低 occupancy；线程 tile 太大还会 spill。优化必须结合 Nsight Compute 的 DRAM throughput、Tensor Core 利用率、stall、occupancy 和 register 数。

**项目边界**：NanoHybrid 当前复用 PyTorch/FlashAttention/FLA，没有自研 GEMM Kernel。这个答案属于算子基础，不能放进项目贡献。

### 面经补充 C3（A）：现场手撕 Tiled GEMM 应怎样推导？

先声明 Shape 和布局：

~~~text
A: [M,K] row-major
B: [K,N] row-major
C: [M,N] row-major
~~~

一个 Block 负责 `BM×BN` 输出；循环 `k0=0..K step BK`：

~~~cpp
for each K tile:
    cooperative load A[BM,BK] to shared
    cooperative load B[BK,BN] to shared
    __syncthreads()

    for k in [0,BK):
        accum += As[row][k] * Bs[k][col]

    __syncthreads()
store accum to C
~~~

面试时必须补充：

- `row<M`、`col<N`、`k<K` 的边界 mask；
- K 不整除 BK 时越界元素填 0；
- load 要合并访问，shared 访问避免 bank conflict；
- accumulation 通常用 FP32；
- 不要每个乘加都同步，只在共享 tile 被覆盖前后同步；
- 测试非整除 Shape、极小 Shape、不同 dtype 和转置布局。

若继续优化，让每线程算 `TM×TN` 微块，累加器放寄存器；再分 Warp Tile，并把 load/compute 做双缓冲。最后说明为什么某组 `BM/BN/BK` 适合当前 Shape，而不是背固定数字。

### 面经补充 C4（A）：和 cuBLAS 做性能对比时怎样计时？

公平流程：

1. A/B/C 预先分配在 GPU，初始化和 H2D 不计入纯 GEMM Kernel 时间。
2. 自定义 Kernel 和 cuBLAS 使用相同 M/N/K、dtype、layout、转置和 accumulation 精度。
3. 先 warm-up，排除 CUDA context、Triton JIT、cuBLAS heuristic 和频率爬升。
4. 在同一 Stream 上用 CUDA Event 包住多次迭代：

~~~python
start.record()
for _ in range(iters):
    op()
end.record()
end.synchronize()
ms = start.elapsed_time(end) / iters
~~~

5. 多组重复，报告 median/p95，不只取最小值。
6. 先和高精 reference 比较 max/mean error、`atol/rtol`，再比较性能。
7. 报告：

~~~text
TFLOPS = 2*M*N*K / seconds / 1e12
speedup = cublas_time / custom_time
~~~

还要说明 cuBLAS 是否允许 TF32、是否使用 Tensor Core、workspace 上限以及是否把 epilogue 算在双方同一范围。小矩阵很容易被 launch overhead 主导，单次 Host wall-clock 计时不可靠。

---

## 四、C++、工程与现场编程

### Q71（A）：完整与不完整类型？
**答**：前向声明可用于指针/引用；需要大小、成员访问、继承或按值成员时必须完整。pImpl 的 `unique_ptr<T>` 析构点通常需完整 T。

### Q72（A）：指针在栈还是堆？
**答**：指针变量位置由声明决定；所指对象可在栈、堆、静态区或设备内存。`new` 决定对象动态分配，不决定指针本身位置。

### Q73（A）：析构抛异常？
**答**：栈展开期间再抛会 `std::terminate`；析构通常 `noexcept`，失败通过日志/显式 close 报告。GPU 资源封装用 RAII，但关键释放可提供可检查接口。

### Q74（A）：`make_unique`？
**答**：创建对象并立即交给 `unique_ptr`，异常安全且少写裸 new；本质是完美转发到 `new T(...)`。与 `make_shared` 的控制块合并不是一回事。

### Q75（A）：`move`、`forward`、万能引用？
**答**：`move` 只是转成将亡值；`forward<T>` 保留推导值类别；推导上下文 `T&&` 可为 forwarding reference。const move 常仍拷贝，返回局部通常别手动 move 破坏 NRVO。

### Q76（A）：`enable_if`/SFINAE？
**答**：模板替换失败时移除候选而非硬错误；`enable_if` 按条件启用重载，C++20 更推荐 concepts。SFINAE 不会吞掉函数体内所有错误。

### Q77（A）：Python GIL？
**答**：CPython 通常一次仅一线程执行 Python bytecode；I/O 或 C/CUDA 扩展可释放 GIL。GPU 异步不等于 Python 调度无开销，CPU 密集可多进程/本地扩展。

### Q78（A）：rebase、amend、PR？
**答**：rebase 把提交重放到新基线并改 hash；amend 修改最近提交；PR 是请求审查并合入分支的协作流程。不要随意 rebase 已共享分支；按功能拆可审查提交。

### Q79（A）：RAII、智能指针、虚函数、并发？
**答**：RAII 绑定资源生命周期；unique 独占、shared 计数、weak 破环；多态基类需虚析构；共享状态先减少共享，再选 mutex/atomic。shared_ptr 控制块线程安全不代表对象内容安全。

### Q80（A）：Cache/TLB 与控制面？
**答**：指针追逐、随机元数据、伪共享会增加 cache/TLB miss；连续数组、紧凑结构、分片队列可改善。Scheduler/BlockManager CPU 开销也会进入 ITL。

### 现场编程清单

1. LeetCode Medium：哈希、二分、堆、滑窗、图、回溯、二维 DP。
2. C++ 模板 Stable Softmax：减最大值、FP32 累加、空输入/大值。
3. CUDA Reduce：任意长度、shuffle、跨 warp shared、两阶段归约。
4. CUDA/Triton Softmax：mask、非 2 次幂、全 mask、容差与 benchmark。
5. Tiled GEMM：边界 tile、shared、同步、布局、与 cuBLAS 对照。
6. PyTorch MHA/GQA：Shape、head 映射、causal mask、稳定 softmax。
7. 简化 BlockManager/Scheduler：allocate/free/refcount、token budget、无饥饿单测。

---

## 五、分布式与加分方向

### Q81（A）：DP、TP、PP、EP、CP？
**答**：分别切样本、层内张量、层、MoE experts、长序列 token。选择由模型规模、拓扑、序列与 SLO 决定，常组合使用。

### Q82（A）：集合通信？
**答**：AllReduce 聚合并广播；ReduceScatter 聚合后分片；AllGather 拼接；All-to-All 互发不同分片。TP 常用 AR/RS+AG，EP 常用 A2A。

### Q83（A）：Ring AllReduce？
**答**：ReduceScatter+AllGather，各 rank 环传分块；每 rank 约传 `2(N-1)/N×data_size`。大消息带宽好，小消息未必优于树形。

### Q84（A）：通信计算重叠？
**答**：bucket/chunk、独立 stream、异步 collective/P2P、Event 依赖，在当前通信时计算下一块；多个小通信可融合降启动延迟。异步不保证自动重叠。

### Q85（A）：NCCL、NVLink、PCIe、RDMA？
**答**：NCCL 是 GPU 通信库；NVLink/NVSwitch 为节点内高带宽，PCIe 连接 CPU/GPU/NIC，RDMA/GPUDirect RDMA 减少跨节点 CPU staging。先按拓扑组通信组。

### Q86（B）：DDP、ZeRO、FSDP？
**答**：DDP 每卡完整参数/梯度/优化器；ZeRO-1/2/3 依次切优化器、梯度、参数；FSDP 接近 ZeRO-3、按模块聚合参数。推理方向掌握分片、通信、显存权衡即可。

### Q87（B）：Hopper、CUTLASS/CuTe、FP8？
**答**：Hopper SM90 关注 TMA、WGMMA、Cluster；CUTLASS 3.x 用 mainloop/epilogue collective，CuTe 用 layout algebra；FP8 需 scale 和累积精度。RTX 5090 是 SM120 Blackwell，不是 Hopper；没 H100 实测只能写“理解”。

### Q88（B）：RDMA/多机推理如何分析？
**答**：先算每 step 状态字节、理论传输时间和通信计算比，再测带宽、延迟、拓扑、拥塞与 overlap；端到端看路由、straggler、容错。PD 分离必须把 KV 和 GDN state 都计入。

### 面经补充 D1（A）：TP=4 时 Q、K、V、O 和 FFN 分别怎么切？

假设线性层权重按 PyTorch 习惯写成 `W[out_features,in_features]`。

**Q/K/V 使用 Column Parallel**：沿输出维切权重，每张卡保留完整输入 hidden 的副本，但只计算一部分 heads：

~~~text
Wq: [Nq*D, H] → 每卡 [(Nq/4)*D, H]
Wk: [Nkv*D,H] → 每卡 [(Nkv/4)*D,H]
Wv: [Nkv*D,H] → 每卡 [(Nkv/4)*D,H]
~~~

Qwen3.5-9B 的 `Nq=16`、`Nkv=4`，TP=4 时每卡处理 4 个 Q heads 和 1 个 KV head。每张卡在本地完成 RoPE、Q/K Norm、Attention，并只保存本 rank 的 KV heads，所以单卡 KV Cache 主体约缩到 1/4。

**O Projection 使用 Row Parallel**：Attention 输出按 heads/输入维分片，`Wo[H,Nq*D]` 沿输入维切。每卡算：

~~~text
partial_o_r = local_attention_r × Wo_r
~~~

四张卡的 `partial_o_r` 都是对完整 hidden 输出的一部分贡献，因此需要 AllReduce 求和：

~~~text
O = Σ_r partial_o_r
~~~

**FFN 的 gate/up 使用 Column Parallel**：

~~~text
W_gate/W_up: [I,H]
→ 每卡 [I/4,H]
→ 本地 SwiGLU [tokens,I/4]
~~~

gate 和 up 必须用相同切分，才能本地执行 `silu(gate) * up`。

**down_proj 使用 Row Parallel**：

~~~text
W_down: [H,I]
→ 每卡 [H,I/4]
→ 本地 partial hidden
→ AllReduce
~~~

RMSNorm、残差和小型逐元素算子通常每卡复制执行。Qwen3.5 GDN 的投影也需要根据输出 channel/head 分片，但 recurrent state 的 head 维和 causal conv channel 必须使用相同分片规则。

### 面经补充 D2（A）：TP=4 下哪些位置需要 AllReduce、AllGather 或 ReduceScatter？

在经典 Megatron 风格、激活不做 Sequence Parallel 时：

~~~text
输入 hidden：每卡都有完整副本
→ QKV Column Parallel：不通信
→ Attention：各卡本地
→ O Row Parallel：AllReduce
→ gate/up Column Parallel：不通信
→ SwiGLU：各卡本地
→ down Row Parallel：AllReduce
~~~

Column Parallel 后若下一算子能消费分片，就不应立刻 AllGather；Row Parallel 的局部结果需要求和，所以使用 AllReduce。

如果启用 Sequence Parallel，常把 Row Parallel 末尾的 AllReduce 拆成 ReduceScatter，使输出按 token 分片；进入下一个需要完整 hidden 的 Column Parallel 前再 AllGather。这样减少每卡激活显存，也为通信计算重叠创造机会。

具体实现还要区分：

- logits 是全量 Gather 还是分布式 top-k；
- KV heads 是否能被 TP 整除；
- GQA 中一个 KV head 对应哪些本地 Q heads；
- residual 是复制布局还是 sequence-sharded 布局。

不能只背“每层两次 AllReduce”，必须说明为什么局部结果需要求和，以及下一层是否允许继续保持分片。

### 面经补充 D3（A）：TP=4 下 Embedding 和 LM Head 如何切？

**Vocabulary Parallel Embedding** 按词表行切：

~~~text
embedding weight [V,H]
rank r 保存 [V/4,H]
~~~

每张卡判断 input token 是否落在自己的 vocab range；命中的卡做本地 lookup，其他卡输出 0。若后续 hidden 需要每卡完整副本，对四卡 embedding 输出做 AllReduce 求和。因为每个 token 只会在一个 vocab shard 命中，求和等价于选择正确向量。

**LM Head** 若与 Embedding tied，通常使用相同 vocab 分片。每卡只计算：

~~~text
local_logits [tokens,V/4]
~~~

最简单方案是 AllGather 成完整 `[tokens,V]` 后采样，但 vocab 很大时通信和显存开销高。更成熟的实现可以：

- 每卡计算 local max/top-k；
- 集合通信合并候选；
- greedy 时选全局最大 token；
- sampling 时还需正确处理全局 softmax normalizer 和随机采样分布。

因此“权重能按 vocab 切”不等于“采样完全不通信”。

### 面经补充 D4（A）：四张 GPU 之间通过什么通信？如何分析瓶颈？

框架通常通过 PyTorch distributed 调用 NCCL collective；NCCL 再根据拓扑选择 NVLink/NVSwitch、PCIe 或跨机网络。常用操作是 AllReduce、ReduceScatter、AllGather 和 All-to-All。

性能分析步骤：

1. 用 `nvidia-smi topo -m` 或拓扑 API 看 GPU 连接。
2. 计算每层通信 Tensor 字节数和每 token/step 次数。
3. 用 NCCL tests 测实际带宽和延迟。
4. 在 Nsight Systems 中查看 collective 是否形成 GPU 空洞、能否和 GEMM 重叠。
5. 小消息关注 launch/latency，大消息关注链路 bandwidth。

Ring AllReduce 对 N 张卡每 rank 传输量约：

~~~text
2*(N-1)/N * tensor_bytes
~~~

TP 的 Decode batch 很小时，GEMM 变小而 collective 次数不变，通信比例可能比 Prefill 更高。TP=4 能解决容量并不保证单请求延迟一定更低。

**项目边界**：当前 NanoHybrid 使用 TP=1；以上是必须掌握的通用并行原理，不能写成已完成的多卡成果。

---

## 六、投递前验收

### 项目真实性

- [ ] 每个简历动词与真实 commit 一致；所有占位数据已替换。
- [ ] 能区分自己实现、第三方复用和上游参考。
- [ ] README 有架构图、复现命令、测试、原始指标、Profiler 和已知限制。
- [ ] 能讲一个真实失败方案及证据。

### 必须现场推导

- [ ] 4B GDN state 约 49.50 MiB/request。
- [ ] 4B KV 为 32 KiB/token、8 MiB/block。
- [ ] 为什么只命中 Attention KV 会破坏混合模型 Prefix Cache。
- [ ] Prefill/Decode、TTFT/TPOT/Goodput、Paged/Flash Attention。
- [ ] MHA/MQA/GQA/MLA 与 MTP 的边界。
- [ ] Decode-first 的收益、代价、无饥饿与退化 workload。
- [ ] Roofline、有效带宽、Systems→Compute 的分析顺序。
- [ ] TP 精确切法、PD 分离状态传输成本。

### 高频红线

1. MTP 不是 Attention；Chunked Prefill 不减少 Decode 计算。
2. FlashAttention 与 PagedAttention解决的问题不同。
3. `isend/irecv` 是异步 P2P API，不是固定底层协议。
4. RTX 5090 是 SM120 Blackwell，不是 Hopper SM90。
5. 调用 FLA/CUTLASS/Triton 示例不等于有内核优化经验。
6. Kernel 加速未转化为端到端收益时，用 Amdahl 与时间线解释。
7. 平均吞吐不能替代 TTFT/TPOT P95/P99 与 Goodput。
8. 项目目标不是实验结果，未完成前保留 `[待实测]`。

## 七、结果填写区

```text
Git commit/tag：[待补充]
GPU/驱动/CUDA/PyTorch/Triton/FLA：[待补充]
模型、dtype、workload、arrival rate：[待补充]
Baseline TTFT/TPOT p50/p95/p99：[待实测]
Optimized TTFT/TPOT p50/p95/p99：[待实测]
吞吐/Goodput/峰值显存：[待实测]
Profiler 结论、收益与退化 workload：[待补充]
```
