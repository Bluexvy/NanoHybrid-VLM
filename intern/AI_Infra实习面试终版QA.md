# AI Infra 实习面试终版 QA

> 适用简历：NanoHybrid-VLM 最终版（CANN 比赛结果待补）  
> 使用方式：先背每题“口述回答”和“一句话记忆点”，再用“实现细节/追问链”做压力面试。  
> 事实优先级：当前代码与《NanoHybrid-VLM项目进度记忆》第 14 节 > 最终简历 > 历史 QA。

## 0. 简历事实表与红线

### 0.1 固定事实

| 项目 | 最终口径 |
|---|---|
| 模型与硬件 | Qwen3.5-9B、RTX 5090、Tensor Parallel=1、BF16 |
| Hybrid 层 | 32 层 Decoder：24 个 Gated DeltaNet（GDN）层、8 个 Full Attention 层 |
| 两类历史状态 | Full Attention 使用 Paged KV Cache；GDN 使用 Conv State Pool 与 FP32 Recurrent State Pool |
| Recurrent State Pool | `[num_slots, 24, 32, 128, 128]`，FP32 |
| Conv State Pool | `[num_slots, 24, 8192, 4]`，BF16 |
| Decode Graph | B=1/2/4/8/16 五个精确 Bucket；其他 Batch 自动回退 Eager |
| Prefix Entry | 同一 Token 边界联合保存物理 KV Blocks、Conv Snapshot、Recurrent Snapshot |
| Prefix Snapshot | 单 Entry 从 49.5 MiB 降到 25.5 MiB：48 MiB recurrent FP32→24 MiB BF16，另有 1.5 MiB conv BF16 |
| 16K Trace | 16K 上下文、12K 共享前缀、50% 命中率，平均 TTFT 降低 36.66% |
| Graph B=16 | 1087.37 tok/s，TPOT 14.71 ms |
| Conv 子路径 | 相对原 Gather/Scatter 路径约 3.21×，耗时降低约 68.8% |
| 完整 Eager | 自研 CUDA 后端相对 FLA Eager 吞吐提升 26.86%～32.13% |
| 双方 Graph | 自研 CUDA 后端相对 FLA Graph 吞吐提升 1.99%～18.31% |
| 正确性 | Greedy Token 对齐；B=1/2/4/8/16、多步状态、未触碰 Slot/Layer、非连续 Tensor Stride 与 Graph/Eager 状态一致性均验证 |

### 0.2 归属边界

- **本人实现并可以展开**：Qwen3.5 Hybrid 引擎适配、两类 State Pool/Slot、联合 Prefix Cache、Decode CUDA Graph、自研 State-Aware recurrent/causal-conv CUDA Kernel、PyTorch C++/CUDA Extension、测试与 Benchmark。
- **第三方复用**：Prefill 的 FLA `chunk_gated_delta_rule`、Full Attention 的 FlashAttention、Hugging Face/AutoProcessor 的预处理和参考输出。回答时必须说“调用/适配”，不能说“实现”。
- **理解但未实现**：TP>1、MoE、MTP/EAGLE、量化、Offload、KV 压缩、通用 Shape Kernel、Prefill CUDA Kernel。
- **Profile 边界**：已用 CUDA Event/PyTorch Profiler 做时间与调用链分析；理解 Nsight Systems/Compute/SASS 的作用，但尚无 Nsight Compute 硬件计数器结论。
- **多模态边界**：完成文本与单图输入；未声称多图、视频或图像 Prefix Cache。

### 0.3 绝对不能说

1. 不能说自己实现了 FlashAttention、FLA、AutoProcessor 或 Triton 最终方案。
2. 不能说完成了 TP、MTP、MoE、量化、Offload、KV 压缩、通用模型/通用 Shape 支持。
3. 不能把 Conv 子路径的 3.21× 写成完整模型加速，也不能把 CUDA+Graph 相对 FLA+Eager 的联合收益说成纯 Kernel 收益。
4. 不能把一次纯 Hot 请求约 94% 的 TTFT 降幅说成生产 Trace 收益；简历口径是 50% 命中率下平均降低 36.66%。
5. 不能说 CUDA Graph 只占 0.1259 MiB；这是静态 Workspace，捕获分配增量约 48.88 MiB，CUDA 私有池/Reserved 还会更大。

---

## 一、开场与个人问题（Q1～Q5）

### Q1. 请做一个 60 秒自我介绍

**30～60 秒口述回答**

面试官您好，我目前是重庆大学软件工程硕士，本科是重庆师范大学数据科学与大数据技术。我的研究经历训练了数学建模、实验设计和论文阅读能力，后来我把主要精力转向 AI Infra。最近我基于 nano-vLLM 做了 Qwen3.5-9B Hybrid 多模态推理引擎：适配了 Full Attention 与 Gated DeltaNet 两类状态，完成了联合 Prefix Cache、B=1/2/4/8/16 Decode CUDA Graph，并针对动态 State Pool 与 FLA 连续 Batched State 接口不匹配的问题，实现了 State-Aware GDN CUDA Kernel。最终在 RTX 5090 上，完整 Eager Decode 相对 FLA 基线吞吐提升 26.86%～32.13%。我希望在实习中继续做推理系统和算子优化。

**项目实现细节**

- 重点只讲一条主线：Hybrid 状态管理 → Cache/Graph → CUDA Kernel。
- 量化结果必须同时带上硬件、模型、Batch 范围和 Baseline。
- 论文/比赛只作为学习与协作背景，不挤占项目细节。

**连续追问链**

- 问：你最能体现个人贡献的部分？答：State-Aware Kernel 的问题定位与端到端接入，因为它同时改了状态接口、CUDA 实现和运行时路径。
- 问：项目最硬的证据？答：正确性、子路径、完整模型和 Profile 四层证据闭环，而不是只报一个微基准数字。
- 问：什么时候能实习？答：按真实可到岗时间回答，并与简历“5 天/周、三个月以上”保持一致。

**证据**

- 最终 Benchmark：Qwen3.5-9B、RTX 5090、B=1～16。
- 完整 Eager 吞吐提升 26.86%～32.13%，双方 Graph 后提升 1.99%～18.31%。

**边界与红线**

- 不把团队/AI/第三方库贡献归为个人原创。
- 不在开场堆砌没有实现的 TP、MTP、MoE。

**一句话记忆点**：我的主线不是“给 nano-vLLM 加功能”，而是解决 Hybrid 模型状态在 Cache、Graph 和 Kernel 三层的真实接口矛盾。

### Q2. 你的原研究方向和 AI Infra 不完全相关，为什么转方向？

**30～60 秒口述回答**

原方向让我熟悉了形式化分析、数据结构和可复现实验，但我发现自己更喜欢能从系统瓶颈一路追到代码和硬件的数据闭环。AI Infra 正好要求模型理解、运行时设计和 CUDA 优化结合。我没有把原方向丢掉，而是把其中“先建模、再验证、最后用数据证明”的方法迁移到了推理系统：例如先定位 FLA 接口导致的状态搬运，再用 Profile 分离数学 Kernel 与系统开销，最后通过完整模型 Benchmark 验证收益。

**项目实现细节**

- 问题建模：连续 Batched State 和动态 Slot State Pool 的布局冲突。
- 实验拆分：Reference correctness → CUDA Extension → 子路径 → 完整 Eager/Graph → Profile。

**连续追问链**

- 问：转方向做了哪些补课？答：按推理链路学习 PyTorch、Transformer、vLLM 调度、CUDA 编程和性能测量。
- 问：为什么不是只做模型算法？答：我更关注请求如何被调度、状态如何驻留、Kernel 如何落到 GPU。

**证据**

- 项目同时包含调度、内存管理、缓存、Graph 与 CUDA Extension，不是单点 Demo。

**边界与红线**

- 不贬低原研究方向，也不声称短期内覆盖了所有 AI Infra 技术。

**一句话记忆点**：我转的不是“热门方向”，而是从算法实验转向更适合自己的系统—Kernel 闭环。

### Q3. 为什么选择 nano-vLLM 和 Qwen3.5，而不是直接改 vLLM？

**30～60 秒口述回答**

nano-vLLM 保留了 LLM Engine、Scheduler、Block Manager、Model Runner 和 Paged KV Cache 的核心骨架，但代码规模足够小，适合把一次请求的生命周期真正看透。Qwen3.5 的价值在于它不是纯 Transformer：32 层里有 24 个 GDN 层和 8 个 Full Attention 层，同一请求同时需要 KV Cache、Conv State 和 Recurrent State。这迫使我处理真实的 Hybrid 状态一致性，而不只是注册一个新模型类，因此比普通文本模型适配更有系统深度。

**项目实现细节**

- Full Attention：物理 KV Block + `block_table`。
- GDN：`state_slot` 指向 Conv/Recurrent State Pool。
- Cache/抢占/Graph 都必须同时处理两类历史。

**连续追问链**

- 问：nano-vLLM 的局限？答：功能和生产完备性不及 vLLM，但便于理解并验证核心机制。
- 问：为什么 Qwen3.5 难？答：同一个 Sequence 的历史不再只有 token-indexed KV，还包含定长 recurrent/conv state。
- 问：换模型能直接用吗？答：引擎抽象可复用，但当前 CUDA Kernel 对 H=32、Dk=Dv=128、C=8192、K=4 特化。

**证据**

- 32 层混合结构、两类 State Pool、联合 Prefix Entry 和 Hybrid Decode Graph 均已跑通。

**边界与红线**

- 不把教学框架说成生产 vLLM 的完整替代品。

**一句话记忆点**：nano-vLLM 让我看清引擎，Qwen3.5 则让状态管理问题从“KV”升级成真正的 Hybrid State。

### Q4. 项目使用了 AI，你自己真正掌握了什么？

**30～60 秒口述回答**

我会诚实区分辅助和掌握。AI 帮我做过资料整理、测试脚手架和早期方案讨论，但最终写进简历的 CUDA 路径，我能够从 Tensor Shape、地址计算、Warp 映射、C++ Binding、current stream、Graph capture 到 Benchmark/Profile 完整解释，并能根据报错定位非连续 Stride、参数接口和状态 Shape 问题。早期探索过但我没有真正掌握的实现不会写进最终项目，例如我没有把 Triton 原型列为个人技术成果。

**项目实现细节**

- 自己必须能白板解释：`state_slot_ids`、`gdn_index`、两个 State Pool、Prefix Entry、Graph Workspace。
- 每项性能结论都能指出测试脚本、Baseline 和测量方法。

**连续追问链**

- 问：没有 AI 能否维护？答：可以从 Reference/单测和 Profile 入手定位，并沿 Python→C++→CUDA 调用链修改。
- 问：AI 生成代码如何验收？答：先检查接口和 Shape，再做多 Batch、多步、未触碰状态、非连续 Stride、Eager/Graph 与完整模型验证。
- 问：哪部分最初理解不足？答：CUDA 内存/线程映射和 Profile，后来通过独立 Kernel、对照实验和反汇编基础逐步补齐。

**证据**

- Recurrent/Conv 均覆盖 B=1/2/4/8/16、8 个递归步和 untouched state 检查。

**边界与红线**

- 不用“全是我从零独立完成”回避工具辅助；重点证明可解释、可复现、可维护。

**一句话记忆点**：AI 可以加速写代码，但我的验收标准是能解释、能修改、能用正确性和性能证据闭环。

### Q5. 讲一次失败经历，以及你如何协作

**30～60 秒口述回答**

项目中我曾直觉认为只要把 recurrent 数学写成 CUDA 就一定会比 FLA 快，但 Profile 显示自研数学 Kernel 是 32.27 μs/层，反而慢于 FLA 的 28.64 μs/层。继续拆解后发现系统真正的瓶颈是 FLA 接口前后的 Gather/Scatter，每个 Decode Step 约 2.459 ms。于是我没有包装微基准，而是把优化目标改成直接寻址 State Pool，最终完整模型获得稳定收益。协作时我会把问题拆成接口、正确性、性能三个验收项，用脚本和数据同步，而不是只描述“已经优化”。

**项目实现细节**

- 失败假设：数学 Kernel 必须单独更快。
- 修正结论：系统优化可以接受单个数学 Kernel 略慢，只要消除的状态搬运更大。

**连续追问链**

- 问：如果同伴不同意？答：先统一 workload、Baseline 和指标，再用可复现实验决策。
- 问：比赛中如何当队长？答：当前只可说负责任务拆分、进度同步和验收；具体技术与结果待比赛完成后补。
- 问：学到了什么？答：Profile 的作用是推翻直觉，不是给既定方案找证据。

**证据**

- FLA 状态搬运 19.669 ms/8 steps；完整 Eager 提升 26.86%～32.13%。

**边界与红线**

- CANN 比赛尚在进行，不虚构算子、排名或性能。

**一句话记忆点**：最有价值的失败是发现“Kernel 略慢但系统更快”，并用 Profile 找到了真正该消除的接口开销。

---

## 二、请求生命周期与 Hybrid 架构（Q6～Q19）

### Q6. 用三分钟完整介绍 NanoHybrid-VLM 项目

**30～60 秒口述回答**

这个项目以 nano-vLLM 为骨架，目标是让 Qwen3.5-9B Hybrid 模型拥有可用且可测的多模态推理能力。第一层是模型与引擎适配：8 个 Full Attention 层使用 Paged KV Cache，24 个 GDN 层使用按请求分配的 Conv/Recurrent State Slot，并支持 Variable-length Batched Prefill、Chunked Prefill、Continuous Batching 和抢占。第二层是联合 Prefix Cache，把同一 Token 边界的 KV Blocks 与 GDN Snapshot 一起缓存。第三层是 B=1/2/4/8/16 Decode CUDA Graph。最后针对 FLA Batched State 与动态 State Pool 不匹配造成的 Gather/Scatter，开发 State-Aware CUDA Kernel 直接按 Slot 寻址并原位回写。

**项目实现细节**

- 主链：`LLMEngine → Sequence → Scheduler → ModelRunner → model/layers/kernel`。
- 数据面：token/position/KV metadata + `state_slot_ids` + 两个 State Pool。
- 优化面：Prefix 复用历史、Graph 降 CPU launch、Kernel 消除状态重排。

**连续追问链**

- 问：三项优化是否重复？答：Prefix 优化 Prefill，Graph 优化 Decode launch，State-Aware Kernel 优化 GDN Decode 状态 I/O。
- 问：项目最大创新点？答：把 Hybrid State 作为一等资源贯穿 Scheduler、Cache、Graph 与 Kernel。
- 问：最先做什么？答：先保证模型/HF Token 对齐，再逐层加优化，避免用性能掩盖错误。

**证据**

- 16K Trace TTFT -36.66%；B16 Graph 1087.37 tok/s；完整 Eager +26.86%～32.13%。

**边界与红线**

- Prefill GDN 和 Full Attention Kernel 仍复用第三方实现。

**一句话记忆点**：一个 Hybrid 请求有三类历史，我的工作是让它们在调度、缓存、Graph 和 CUDA 中始终一致。

### Q7. 从用户输入到最终生成 Token，完整链路是什么？

**30～60 秒口述回答**

用户的 messages 和图像先由 Processor/Tokenizer 生成 `input_ids`、图像输入与 mRoPE 位置；`LLMEngine.add_request()` 将它们封装为 `Sequence` 并放入 Scheduler 的 Waiting 队列。Scheduler 每步按 Token Budget、KV Block 和 State Slot 资源选择请求，生成 scheduled sequences 与 `num_scheduled_tokens`。ModelRunner 准备 `input_ids`、positions、`slot_mapping`、`block_tables`、`context_lens` 和 `state_slot_ids`，执行 Prefill 或单 Token Decode。模型输出 hidden states，LM Head 得到 logits，Sampler 选下一个 token；Engine 将 token 追加到 Sequence，未结束的请求继续 Running，达到 EOS/长度限制后释放 KV 和 State Slot 并返回结果。

**项目实现细节**

- 核心对象：`LLMEngine`、`Sequence`、`Scheduler`、`BlockManager`、State Slot allocator、`ModelRunner`。
- Prefill 可能分 Chunk；Decode 每个活跃请求通常调度 1 token。
- GDN Kernel 通过当前 Context 读取 `gdn_state_slot_ids` 和两个 Pool。

**连续追问链**

- 问：谁决定本步跑多少 Token？答：Scheduler 结合预算和请求阶段生成 `num_scheduled_tokens`。
- 问：谁负责物理内存？答：BlockManager 管 KV Blocks，State Slot allocator 管 GDN Slot；ModelRunner 只消费元数据执行。
- 问：首个输出 Token 在哪产生？答：最后一个 Prefill Chunk 完成后由 LM Head/Sampler 产生，然后进入 Decode。

**证据**

- Variable-length Prefill、Chunked Prefill、Decode、抢占与 Graph 动态 Batch 均有回归测试。

**边界与红线**

- AutoProcessor/HF tokenizer 是调用，不是本人重新实现。

**一句话记忆点**：Engine 管请求，Scheduler 管本步工作，资源管理器管历史，ModelRunner 把元数据变成一次 GPU 执行。

### Q8. WAITING、RUNNING、FINISHED 分别表示什么，如何迁移？

**30～60 秒口述回答**

WAITING 表示请求已进入系统但本轮没有执行资格，可能是新请求，也可能是被抢占后等待重调度；RUNNING 表示已持有当前执行所需的 KV/State 资源，并被 Scheduler 选中持续 Prefill 或 Decode；FINISHED 表示命中 EOS、长度上限或终止条件，之后输出被收集，KV Block 引用和 GDN State Slot 都必须释放。迁移不是简单的 FIFO：WAITING→RUNNING 取决于 Token Budget 和两类状态资源，RUNNING 资源不足时可以回到 WAITING，完成后进入 FINISHED。

**项目实现细节**

- 进入 Running：分配/恢复 KV Blocks，分配/恢复 `state_slot`。
- 抢占：撤销本轮调度并释放请求资源，保留可重算的 token 状态。
- Finish：最终 deallocate，不能只释放 KV。

**连续追问链**

- 问：为什么抢占后不是 Finished？答：请求逻辑未结束，只是释放驻留状态，后续可恢复/重算。
- 问：Running 是否每轮都执行？答：不一定，受预算和策略影响；状态名与当前微步是否入 Batch 要区分。
- 问：漏释放 State Slot 会怎样？答：Slot 泄漏，后续并发下降，严重时读到旧请求状态。

**证据**

- 抢占与动态 Batch 测试检查了请求切换后状态不串扰。

**边界与红线**

- 不把状态机描述成只有一次 WAITING→RUNNING→FINISHED 的线性流程。

**一句话记忆点**：Hybrid 抢占必须同时管理“是否执行”和“KV/State 是否驻留”。

### Q9. `num_tokens`、`num_cached_tokens`、`num_scheduled_tokens` 有什么区别？

**30～60 秒口述回答**

`num_tokens` 是 Sequence 当前逻辑上拥有的 Token 总数，包括 Prompt 和已经生成的 Token；`num_cached_tokens` 表示这些 Token 中已经有可直接复用的物理历史状态的前缀长度，对 Hybrid 模型要求 KV 与 GDN Snapshot 在同一边界都有效；`num_scheduled_tokens` 是 Scheduler 在当前 Step 真正安排计算的 Token 数。三者分别回答“序列有多长”“前面多少不用重算”“这一轮算多少”。典型 Hot Prefill 是 `num_tokens` 很长、`num_cached_tokens` 很大，而本轮只调度剩余 suffix 的一部分。

**项目实现细节**

- 待计算区间可理解为从已缓存/已计算边界到当前逻辑长度。
- Chunked Prefill 进一步限制本轮 `num_scheduled_tokens`。
- Decode 时每个请求通常新增 1 个 token，但 `num_tokens` 会随步数增长。

**连续追问链**

- 问：缓存命中 12K、Prompt 16K，本轮一定算 4K 吗？答：逻辑上需算 4K，但可能按 1K Chunk 分四轮。
- 问：为什么不能只维护一个长度？答：逻辑长度、物理可复用边界和本轮计算预算是不同状态。
- 问：位置编码用哪个长度？答：用真实逻辑 position，不能因跳过计算把位置重置为零。

**证据**

- Prefix hit 测试验证 1165-token Prompt 命中 1024 后只 Prefill 141 tokens。

**边界与红线**

- 具体字段更新时机以当前 `Sequence/Scheduler` 代码为准，不套用其他 vLLM 版本命名。

**一句话记忆点**：总长度管语义，缓存长度管复用，本轮调度长度管计算。

### Q10. Qwen3 和 Qwen3.5 在这个项目里最关键的结构差异是什么？

**30～60 秒口述回答**

对引擎而言，最关键的差异不是名字或参数量，而是 Qwen3.5-9B 使用 Hybrid Decoder：32 层中 24 层是 Gated DeltaNet，只有 8 层是 Full Attention。纯 Qwen3 Transformer 的历史主要是每层随 Token 增长的 KV Cache；Qwen3.5 还要维护固定大小的 recurrent matrix state 和 causal-conv window state。于是 BlockManager 不能独自代表请求全部历史，Prefix Cache、抢占、Continuous Batching 和 CUDA Graph 都必须同时知道 KV Block 与 State Slot。

**项目实现细节**

- Full Attention history：token-indexed、分页、随上下文增长。
- GDN history：per-sequence/per-layer 定长状态，借助 `state_slot` 间接寻址。
- 32=24 GDN+8 Attention。

**连续追问链**

- 问：为什么 GDN 长上下文显存更稳？答：recurrent state 大小不随 Token 数线性增长。
- 问：那为什么还保留 Attention？答：Hybrid 架构兼顾线性状态传播与显式历史注意力能力。
- 问：项目实现了模型训练吗？答：没有，只做推理适配与优化。

**证据**

- 两类 Layer 走不同历史路径，完整 Greedy Token 与 Hugging Face 对齐。

**边界与红线**

- 不扩展到没有验证过的 Qwen3.5 其他尺寸/变体。

**一句话记忆点**：Qwen3 的历史主要是 KV；Qwen3.5 Hybrid 的历史是 KV 加两种 GDN State。

### Q11. 24 个 GDN 层和 8 个 Full Attention 层如何共同管理历史？

**30～60 秒口述回答**

我把历史按 Layer Type 分流，但由同一个 Sequence 生命周期统一管理。遇到 Full Attention 层时，ModelRunner 根据 `slot_mapping/block_tables/context_lens` 访问该层 Paged KV Cache；遇到 GDN 层时，根据当前层对应的 `gdn_index` 和该请求的 `state_slot_ids` 访问 Conv/Recurrent State Pool。两条路径每层更新各自历史，但请求完成、抢占、Prefix commit/restore 时必须作为一个事务共同处理，否则会出现 KV 在时刻 t、GDN State 在时刻 t′ 的不一致。

**项目实现细节**

- `gdn_index` 是 GDN 层在 24 个状态层中的稠密编号，不等同于全局 Decoder layer id。
- `state_slot_ids[B]` 将当前动态 Batch 行映射到全局 Slot。
- Prefix Entry 的边界必须同时满足完整 KV Block 和 GDN Snapshot 可恢复。

**连续追问链**

- 问：为什么需要稠密 `gdn_index`？答：State Pool 第二维只有 24 个 GDN 层，节省空间且索引直接。
- 问：两类状态更新有先后依赖吗？答：各层按模型顺序串行，单层只读写自身对应历史。
- 问：能只恢复 GDN 不恢复 KV 吗？答：不能作为同一 Prefix hit 使用，会破坏语义边界。

**证据**

- Joint Prefix hit 输出保持一致；Graph/Eager 下 KV、Conv、Recurrent 状态一致。

**边界与红线**

- “统一管理”不是把三种状态放进一个 Tensor，而是生命周期和 Token 边界统一。

**一句话记忆点**：Layer 内分流、Sequence 级合流，三类历史必须绑定同一 Token 时间点。

### Q12. `block_table` 与 `state_slot` 分别保存什么？

**30～60 秒口述回答**

`block_table` 是逻辑 Token Block 到物理 KV Block ID 的映射，一个 Sequence 会随上下文增长拥有多个 Block；`state_slot` 是一个整数句柄，指向这个 Sequence 在固定大小 GDN State Pool 中的一整套槽位。前者是一对多、长度随上下文增长，后者通常是一对一、状态大小固定。ModelRunner 会把多个 Sequence 的 Block Table padding 成批量元数据，同时把每个 Sequence 的 Slot 组成 `state_slot_ids[B]`。

**项目实现细节**

- Attention 使用：`block_tables[B,max_num_blocks]`、`slot_mapping`、`context_lens`。
- GDN 使用：`state_slot_ids[B]`、`gdn_index`、Pool base pointer。
- Slot 0/1 等物理位置与 Batch 第 0/1 行没有固定关系。

**连续追问链**

- 问：为什么不让 Batch index 直接作为 Slot？答：Continuous Batching 会插入、完成和抢占请求，Batch 顺序每步变化。
- 问：Block Table 为什么要 padding？答：批量 Kernel/Graph 需要规则元数据 Shape。
- 问：state slot 会随 Token 增长吗？答：不会，内部状态数值更新但 Shape 固定。

**证据**

- 动态 16→…→1 Batch 与乱序 Slot 测试未发生状态串扰。

**边界与红线**

- 不把 `state_slot` 说成 GDN State 本身，它只是索引句柄。

**一句话记忆点**：Block Table 找分页 KV，State Slot 找整套定长 GDN 状态。

### Q13. ModelRunner、BlockManager 和 State Slot Allocator 如何传递信息？

**30～60 秒口述回答**

Scheduler 面向 Sequence 做决策；BlockManager 根据需要计算的 Token 数检查并分配 KV Blocks，State Slot Allocator 为需要驻留 GDN 历史的请求分配 Slot。调度结果把每个 Sequence 的物理 Block IDs、上下文长度和 `state_slot` 传给 ModelRunner。ModelRunner 不决定所有权，而是将它们整理成 GPU Tensor 元数据；Attention Kernel 按 Block Table 访问 KV，GDN CUDA Kernel 按 `state_slot_ids/gdn_index` 访问 Pool。完成或抢占时，控制流再通知两个资源管理器回收。

**项目实现细节**

- 所有权层：BlockManager/Slot allocator。
- 执行层：ModelRunner/Context/Kernel。
- `allocate` 建立引用与映射，`deallocate` 下降引用并归还空闲资源。

**连续追问链**

- 问：为什么 ModelRunner 不直接分配？答：资源与调度策略耦合，应在 Engine/Scheduler 侧统一决策。
- 问：如何避免双重释放？答：明确 request 引用与 cache 引用，状态机只在合法迁移点回收。
- 问：Prefix hit 谁负责？答：Cache 找 Entry，BlockManager 恢复 KV 引用，State 管理器拷贝 Snapshot 到活动 Slot。

**证据**

- Prefix commit/hit/eviction/shared-block 与抢占回归覆盖引用生命周期。

**边界与红线**

- 类名和职责按当前项目，不能把新版 vLLM 的 Worker/Executor 体系硬套进来。

**一句话记忆点**：资源管理器给地址，Scheduler 给任务，ModelRunner 把两者整理成 Kernel 能消费的元数据。

### Q14. GDN 的 Recurrent State 在 Prefill 和 Decode 中如何更新？

**30～60 秒口述回答**

Prefill 一次处理多个 Token，继续调用第三方 FLA 的 `chunk_gated_delta_rule`，因为它擅长 Chunk 并行计算，并在 Chunk 末得到最终 recurrent state。Decode 的序列长度 L=1，每步只更新一个 Token；原路径要从动态 Pool Gather 成连续 Batched State，再交给 FLA，之后 Scatter 回 Pool。我的 CUDA 后端在 Decode 时直接接收 `state_slot_ids` 和 `gdn_index`，从 `[num_slots,24,32,128,128]` Pool 读取对应矩阵，融合 decay、Delta Rule、输出计算和原位回写。

**项目实现细节**

- Prefill：第三方 FLA，返回 Chunk 末状态。
- Decode：自研 State-Aware CUDA，BF16 Q/K/V/Beta、FP32 g/state/accumulation、BF16 output。
- 每层通过 `gdn_index` 选择 State Pool 第二维。

**连续追问链**

- 问：为什么不统一都用自研 Kernel？答：当前 Kernel 针对 L=1 Decode，Prefill 的序列并行算法完全不同。
- 问：Decode 为什么仍需要旧 State？答：recurrent state 是之前所有 Token 历史的压缩表示。
- 问：原位更新会不会影响输出？答：同一 Kernel 内先按 Delta Rule得到新状态，再用定义所需状态计算输出，并通过 Reference 校验。

**证据**

- 多步 recurrent test 最大 state error 约 5.59e-9、output error 约 1.91e-6。

**边界与红线**

- 不能说完成了 GDN Prefill CUDA Kernel。

**一句话记忆点**：Prefill 保留 FLA Chunk 算法，Decode 用 State-Aware Kernel 消掉接口重排。

### Q15. 为什么 GDN 除 Recurrent State 外还需要 Conv State？

**30～60 秒口述回答**

GDN 层在进入 recurrent Delta Rule 前包含短卷积，用最近 K=4 个位置做 depthwise causal convolution。Recurrent State 压缩长历史，Conv State 保存每个 channel 最近 4 个值，它们语义不同，不能互相替代。项目中 Conv Pool 是 `[num_slots,24,8192,4]` BF16；Decode 每步按 Slot 取出 4-tap window，移位加入当前 `x`，与 `[8192,4]` 权重逐 channel 点积，经过 SiLU 输出，再把新 window 原位写回。

**项目实现细节**

- `x [B,8192,1]`，`weight [8192,4]`。
- 每线程处理一个 `(batch_index, channel_index)`。
- BF16×4 使用 64-bit vector load/store。

**连续追问链**

- 问：为什么是 depthwise？答：每个 channel 有独立 4-tap filter，没有 channel 间归约。
- 问：Conv State 为什么用 BF16？答：与输入/模型精度一致，减小容量和带宽；累加可转 FP32。
- 问：请求结束如何处理？答：Slot 被释放，后续分配时必须初始化/覆盖，不能继承旧请求 window。

**证据**

- Conv 8-step 测试 state/untouched error=0，最大 output error 7.63e-6。

**边界与红线**

- 这是 K=4、C=8192 的特化 Decode Conv，不是通用 cuDNN 替代。

**一句话记忆点**：Recurrent State 记长历史，Conv State 记最近 4 个位置。

### Q16. Prefill 和 Decode 为什么需要不同执行路径？

**30～60 秒口述回答**

Prefill 的序列长度通常大，能在 Token 维形成并行，主要关注吞吐和大矩阵计算；Decode 每个请求每步只有一个新 Token，Batch 维有限，更容易受 Kernel launch、状态读写和小算子开销限制。项目因此在 Prefill 使用 Variable-length Batch、Chunked Prefill、FLA Chunk Kernel 和 FlashAttention；Decode 则使用固定 Batch Bucket CUDA Graph，以及直接访问 State Pool 的 L=1 CUDA Kernel。相同数学模型需要不同系统优化，不能用一个 Kernel 覆盖所有阶段。

**项目实现细节**

- Prefill metadata 包含变长 Token 段和 cumulative offsets。
- Decode Graph 输入是 B 个单 Token，加静态 metadata/workspace。
- Prefix Cache 命中主要减少 Prefill Token。

**连续追问链**

- 问：Prefill 一定 compute-bound 吗？答：通常更接近 compute-bound，但具体仍取决于模型、长度和算子。
- 问：Decode 一定 memory-bound 吗？答：常因权重/状态读取和小 Batch 利用率受限，也可能被 CPU launch 主导。
- 问：为什么 Graph 不捕获 Prefill？答：Prefill Shape 和长度动态性更强，当前只实现 Decode Bucket。

**证据**

- Graph 对小 Batch Decode 收益明显；Prefix Trace 显著降低 TTFT。

**边界与红线**

- 不用“Prefill 永远算力瓶颈、Decode 永远带宽瓶颈”这种绝对表述。

**一句话记忆点**：Prefill 优化 Token 并行，Decode 优化每步固定工作和状态访问。

### Q17. 适配 Qwen3.5 时最困难的地方是什么？

**30～60 秒口述回答**

最困难的是把 GDN State 从模型内部临时量提升为推理引擎的一等资源。它必须随 Sequence 创建、分配、抢占、恢复和释放；Prefix hit 要与物理 KV 在同一 Token 边界恢复；Continuous Batching 每步 Batch 顺序改变，Kernel 不能假设 Batch index 等于 Slot；CUDA Graph 又要求静态地址。最终我的做法是用稳定的 State Pool 加动态 `state_slot_ids` 间接寻址，在控制面维护生命周期，在 CUDA 路径直接读写物理 Pool。

**项目实现细节**

- 稳定对象：两个 State Pool 的 base address。
- 动态对象：`state_slot_ids[B]` 和每层 `gdn_index`。
- Cache 对象：独立只读 Snapshot，不与活动 Slot 混用。

**连续追问链**

- 问：为什么 Snapshot 不能直接引用活动 Slot？答：Slot 会被其他请求复用，缓存会被污染。
- 问：Graph 下动态请求怎么处理？答：更新静态 `state_slot_ids` 内容，地址不变，值可以变。
- 问：最危险的 Bug？答：不同请求或不同 Token 边界的状态串用，可能仍生成“看似合理”的错误输出。

**证据**

- Snapshot independence、duplicate entry、untouched slot/layer 和动态 Batch 测试均通过。

**边界与红线**

- 不把一次模型类注册描述为完整适配；难点是生命周期一致性。

**一句话记忆点**：真正的难点不是算出 GDN，而是让它的历史跟着请求正确地活、停、存、恢复和释放。

### Q18. 你如何保证请求抢占、Prefix 命中和 Graph Replay 下状态一致？

**30～60 秒口述回答**

我用“逻辑 Token 边界 + 物理资源所有权 + 执行元数据”三层约束。逻辑上，KV 与 GDN Snapshot 只在同一完整边界 commit/restore；所有权上，KV 区分 request 和 cache 引用，活动 GDN State 使用独占 Slot，缓存使用独立 Snapshot；执行上，每步 ModelRunner 重新生成或拷贝当前 `block_tables/context_lens/state_slot_ids`，Graph 只固定地址不固定内容。测试则同时比较输出 token、KV、Conv/Recurrent state 和未触碰资源。

**项目实现细节**

- Prefix `PrefixKey` 带 `num_cached_tokens`。
- Graph Workspace 静态地址，replay 前 `copy_` 新元数据。
- Finish/Preempt 同步释放 KV request refs 与 State Slot。

**连续追问链**

- 问：Hash 相同就一定安全？答：还要验证链式 block 元数据/边界，碰撞不能直接视为命中。
- 问：Graph replay 会缓存旧 Slot ID 吗？答：不会，静态 Tensor 内容在 replay 前更新。
- 问：如何检查没有写错其他请求？答：保存 untouched Slot/Layer 的 golden copy 并逐项比较。

**证据**

- Prefix collision guard、共享 Block、Graph continuous batching 和 kernel untouched tests。

**边界与红线**

- 当前未验证跨 GPU/TP 下的状态一致性。

**一句话记忆点**：边界绑定保证语义，引用/Slot 保证所有权，动态元数据保证每次执行找对物理状态。

### Q19. 这个项目哪些是你实现的，哪些复用了第三方？

**30～60 秒口述回答**

我实现的是引擎侧 Hybrid State 生命周期、联合 Prefix Cache、Decode CUDA Graph、State-Aware recurrent 与 causal-conv CUDA Kernel，以及 C++ Extension、正确性/性能/Profile 脚本。Prefill 的 GDN Chunk 计算复用 FLA，Full Attention 复用 FlashAttention，图像预处理和对齐参考使用 Hugging Face/AutoProcessor。我的贡献不是重新造所有数学库，而是发现第三方接口与动态推理引擎的布局矛盾，并通过新的运行时接口和 Kernel 消除系统开销。

**项目实现细节**

- 第三方 FLA Baseline：连续 Batched State + Gather/Scatter。
- 自研 CUDA：直接 `state_slot_ids/gdn_index` 访问 Pool。
- HF：Golden Token/多模态处理参考，不进入性能归因。

**连续追问链**

- 问：为什么不用第三方到底？答：数学实现已成熟，瓶颈来自接口重排，针对引擎布局定制更有价值。
- 问：是否实现 Triton？答：早期只做探索，最终简历和性能结论均基于本人能解释的 CUDA 路径。
- 问：是否实现 FlashAttention？答：没有，只完成调用适配和 KV metadata 对接。

**证据**

- Profile 同时列出 FLA/官方 Conv 与自研 Kernel 调用，归因清晰。

**边界与红线**

- 第三方库必须主动说明，避免被一次追问击穿可信度。

**一句话记忆点**：我没有重写成熟库，我重写的是不适合动态 State Pool 的 Decode 接口与执行路径。

---

## 三、调度、Paged KV 与抢占（Q20～Q27）

### Q20. 什么是 Continuous Batching？项目中为什么能动态加入新请求？

**30～60 秒口述回答**

Static Batching 要等整批请求全部完成，短请求结束后的计算槽位会空着；Continuous Batching 在每个推理 Step 重新调度，让完成的请求退出、新请求进入，因此 Batch 组成和顺序可以动态变化。项目能够这样做，是因为请求历史不依赖 Batch 行号：Attention 通过 `block_table` 找物理 KV，GDN 通过 `state_slot_ids` 找物理 State Pool。ModelRunner 每步为当前 Batch 重建元数据，所以新请求只需获得自己的资源句柄即可加入。

**项目实现细节**

- 调度粒度是 Step，不是整段请求。
- Decode 中每个活跃请求通常贡献 1 token。
- `state_slot_ids[b]` 解耦 Batch row `b` 与物理 slot。

**连续追问链**

- 问：为什么吞吐更高？答：完成请求留下的容量立即被新请求利用，减少 padding 和空槽。
- 问：动态 Batch 会影响 CUDA Graph 吗？答：只要实际 B 命中精确 Bucket，并在 replay 前更新静态元数据内容即可。
- 问：请求顺序改变会串状态吗？答：不会，物理访问由 block table/slot id 决定，不由 Batch position 决定。

**证据**

- Graph Continuous Batching 与 16→…→1 动态 Bucket 测试通过。

**边界与红线**

- Continuous Batching 是调度机制，不等于把任意 Shape 捕获进同一个 Graph。

**一句话记忆点**：Batch 行每步可以变，物理历史句柄不能跟着变错。

### Q21. Variable-length Batched Prefill 是怎么实现的？

**30～60 秒口述回答**

Variable-length Batched Prefill 不把每个 Prompt padding 到相同长度，而是把本轮各请求真正需要计算的 Token 段拼接成一个扁平 Token 流，同时维护每个 Sequence 的边界、position 和 KV 映射。模型用这些 offsets/metadata 区分不同请求，Attention 写入各自的物理 KV Blocks，GDN 最终状态写回各自 Slot。这样总工作量接近有效 Token 数之和，而不是 Batch size 乘最大长度。

**项目实现细节**

- 输入是多段不同长度的 scheduled tokens，而不是规则 `[B,Smax]` padding。
- 每段保留真实 position；不能因 flatten 改变 mRoPE/causal 语义。
- GDN Prefill 完成后只把每个请求的 final state 写回对应 `state_slot`。

**连续追问链**

- 问：如何防止不同请求互相 Attention？答：使用 sequence boundary/cumulative length 元数据建立各自 causal 区间。
- 问：与 Continuous Batching 的关系？答：前者解决一轮内变长 Prefill，后者解决跨 Step 动态请求组成。
- 问：为什么 Decode 不需要变长？答：常规 Decode 每个请求每步只有一个新 token。

**证据**

- 不同 Prompt 长度批处理与 HF Greedy Token 对齐。

**边界与红线**

- 不把第三方变长 Attention Kernel 说成自研。

**一句话记忆点**：把有效 Token 拼起来算，用边界元数据保住每个请求自己的因果关系。

### Q22. Chunked Prefill 是什么？项目里如何实现，代价是什么？

**30～60 秒口述回答**

Chunked Prefill 把长 Prompt 切成多个调度 Chunk，例如本项目实验使用过 1024-token Prefill Chunk。Scheduler 每轮只给长 Prefill 一部分 Token Budget，让它可以和 Decode 请求共享一个 Step，避免单个超长 Prompt 长时间阻塞在线请求。代价是 Chunk 间需要保存并恢复 KV/GDN 状态，调度和 Kernel 启动次数增加，而且 Chunk 边界、position、`num_scheduled_tokens` 必须正确衔接。

**项目实现细节**

- `num_scheduled_tokens = min(remaining_prefill, token_budget, chunk_limit)`。
- 每个 Chunk 追加 KV，并将 GDN final state 写回活动 Slot。
- 只有最后一个 Prefill Chunk 完成后才进入正常 Decode/产生后续 token。

**连续追问链**

- 问：Chunk 越小越好吗？答：不是；小 Chunk 改善延迟公平性，但增加启动和状态交接开销。
- 问：Prefix 命中后怎么 Chunk？答：从 `num_cached_tokens` 后的 suffix 开始继续分 Chunk。
- 问：抢占发生在 Chunk 中间怎么办？答：只在明确的已提交计算边界保留/重算，不能留下半更新状态。

**证据**

- Prefix hit 1024 后只计算 141-token suffix；16K Benchmark 使用 1024-token Chunk 配置验证长上下文。

**边界与红线**

- 当前没有给出“1024 是全局最优”的结论，它是实验配置。

**一句话记忆点**：Chunked Prefill 用更多调度轮次换取长 Prompt 不独占 GPU。

### Q23. Decode-first、Prefill-first 怎么选？如何防止饥饿？

**30～60 秒口述回答**

Decode-first 更照顾已经在服务中的请求，能降低 TPOT 和交互抖动；Prefill-first 能让新请求更快拿到首 Token，但长 Prompt 可能拖慢现有 Decode。我会采用 Token Budget 内的混合策略：优先保证一定 Decode 配额，再把剩余预算给 Chunked Prefill；同时根据等待时长或连续未调度轮数提高 Waiting 请求优先级，做饥饿保护。核心不是固定选一边，而是明确 SLO：在线服务通常先保 TPOT，再控制 TTFT 尾延迟。

**项目实现细节**

- 调度约束：token budget、KV blocks、free state slots、最大 Batch。
- 长 Prefill 通过 Chunk 化进入，而不是一次吞掉全部预算。
- victim/priority 应同时考虑阶段、资源成本和等待时间。

**连续追问链**

- 问：纯吞吐场景怎么选？答：可增大 Prefill Chunk/Batch，提高 GPU 利用率。
- 问：聊天场景怎么选？答：更强调 Decode-first 和 P99 TPOT。
- 问：项目是否实现复杂 SLO Scheduler？答：实现了基础 Continuous/Chunk/抢占，复杂多租户 SLO 属于扩展方向。

**证据**

- Benchmark 同时报 TTFT、TPOT、吞吐与 P99，避免只优化单一平均值。

**边界与红线**

- 不声称实现了生产级多租户公平调度器。

**一句话记忆点**：Decode-first 保交互，Chunked Prefill 保新请求，aging 防饥饿。

### Q24. KV Block 如何分配、追加、引用和释放？

**30～60 秒口述回答**

Prompt/Decode Token 先按 block size 划分逻辑 Block，BlockManager 从空闲池分配物理 KV Block，并把物理 ID 追加到 Sequence 的 `block_table`。Attention 用 `slot_mapping` 把本轮 Token 写到具体 Block offset。请求持有时增加 request reference；如果完整前缀被 Prefix Cache 固定，还增加 `cache_ref_count`。请求完成或抢占时释放 request reference；Entry 被逐出时释放 cache reference；只有总引用归零，物理 Block 才真正回到 free list。

**项目实现细节**

- `ref_count` 表示总引用，`cache_ref_count` 表示其中缓存持有部分。
- request refs 可由 `ref_count - cache_ref_count` 判断。
- 部分 Block 的 Token 边界与 Prefix commit 策略必须明确，避免缓存未完成内容。

**连续追问链**

- 问：为什么不能请求结束就清空 KV？答：Prefix Entry 可能仍持有缓存引用。
- 问：多个 Prefix Entry 共享 Block 怎么计费？答：按唯一物理 Block 计容量，不重复累计。
- 问：Block Hash 与物理 ID 相同吗？答：不同；Hash 表示内容链，物理 ID 表示当前缓存地址。

**证据**

- shared-block 测试中 4096/8192 层级 Entry 只固定 32 个唯一 Blocks，而非重复 48 个。

**边界与红线**

- Prefix Cache 是复用 KV，不是压缩或量化 KV。

**一句话记忆点**：Block 能否释放由总引用决定，请求结束不等于缓存也结束。

### Q25. 请求抢占时为什么必须同时释放 KV 和 GDN State？

**30～60 秒口述回答**

对 Hybrid 模型，KV 和 GDN State 共同定义请求历史。如果只释放 KV，State Slot 会泄漏，限制并发；如果只释放 State，重新调度时即使 KV 还在也无法从相同语义边界继续。抢占必须把它们作为一个资源事务处理：明确当前已提交 Token 边界，释放 request-held KV references 和活动 State Slot；后续通过可用的联合 Prefix Entry 恢复，或者从更早边界重算。

**项目实现细节**

- KV 是分页且随长度增长；GDN Slot 固定大约 49.5 MiB/请求（活动 FP32 recurrent+BF16 conv）。
- Cache Snapshot 与活动 Slot 独立，抢占不应删除仍被 Cache 持有的 Entry。
- 调度可运行性检查需要同时满足两类资源。

**连续追问链**

- 问：为什么 State Slot 更可能成为并发瓶颈？答：单 Slot 固定约 49.5 MiB，数量预分配且不可碎片化共享。
- 问：能把活动 State 压成 BF16 吗？答：当前活动 recurrent 保留 FP32 正确性，只有只读 Prefix Snapshot 用 BF16。
- 问：抢占一定保存完整快照吗？答：不一定，取决于 cache/admission；也可释放后重算。

**证据**

- Preemption 与 slot reuse 测试验证旧请求 Snapshot 不被新 Slot 使用污染。

**边界与红线**

- 不声称实现 CPU Offload 式抢占。

**一句话记忆点**：Hybrid 请求的可恢复点必须同时拥有 KV 和 GDN State。

### Q26. 资源不足时如何选择 victim？

**30～60 秒口述回答**

victim 选择要比较“释放收益”和“恢复成本”。可综合考虑优先级、到达时间、已等待时间、剩余长度、持有 KV Block 数、是否占 State Slot、以及是否存在可复用 Prefix Entry。一般避免抢占马上结束或恢复成本很高的请求，并给长期等待请求 aging。当前项目实现并验证的是基础抢占和资源一致性；如果面向生产，我会把 victim score 显式化，并分别评估 P99、重算 Token 和资源利用率。

**项目实现细节**

- 释放收益：KV bytes + 一个活动 GDN Slot。
- 恢复成本：需重算的 Token 数、是否命中联合 Prefix。
- 正确性优先：victim 释放必须原子覆盖两类资源。

**连续追问链**

- 问：总抢占最大请求最好吗？答：不一定，大请求释放多但重算也贵。
- 问：LRU 能直接当 victim 策略吗？答：LRU 是缓存逐出策略，不等于运行请求抢占策略。
- 问：如何验证策略？答：构造不同长短/到达间隔 Trace，比较 P50/P99、吞吐、抢占次数和重算量。

**证据**

- 当前证据覆盖状态正确性，不把尚未做的复杂策略包装成成果。

**边界与红线**

- 复杂 victim scoring 是设计建议，不是简历已完成项。

**一句话记忆点**：抢占不是只看谁占得多，而是释放收益减恢复成本。

### Q27. 单请求 KV Cache 和 GDN State 显存怎么计算？

**30～60 秒口述回答**

KV Cache 公式是 `tokens × attention_layers × 2(K,V) × num_kv_heads × head_dim × dtype_bytes`。本模型只有 8 个 Full Attention 层，若 `num_kv_heads=4、head_dim=256、BF16=2 bytes`，每 Token 是 32 KiB；block size=256 时一个物理 KV Block 是 8 MiB。GDN 活动 recurrent state 是 `24×32×128×128×4 bytes=48 MiB`，Conv state 是 `24×8192×4×2 bytes=1.5 MiB`，合计 49.5 MiB/Slot。

**项目实现细节**

- 16K 上下文单请求 Full Attention KV 约 `16384×32 KiB=512 MiB`，不含碎片/metadata。
- 12K Prefix 对应 48 个 256-token blocks，物理 KV 约 384 MiB；具体实验容量以实际 cached boundary 为准。
- BF16 Prefix Snapshot：24 MiB recurrent +1.5 MiB conv=25.5 MiB。

**连续追问链**

- 问：为什么 GDN State 不随上下文增长？答：它用固定矩阵/窗口递推压缩历史。
- 问：为什么活动 Slot 仍是 FP32 recurrent？答：控制递归误差；缓存快照才转 BF16 节省容量。
- 问：实际显存为何高于公式？答：还有权重、激活、Allocator Reserved、Graph pool、Workspace 和碎片。

**证据**

- 测试实测一个 KV Block 8 MiB；一个 FP32 GDN Snapshot 49.5 MiB、BF16 25.5 MiB。

**边界与红线**

- KV 公式中的 `num_kv_heads/head_dim` 要与实际 config 核对；不能把模型 Attention head 与 GDN H/Dk 混用。

**一句话记忆点**：KV 随 Token 线性增长，GDN Slot 固定 49.5 MiB；两者都要进入并发容量预算。

---

## 四、多模态与 mRoPE（Q28～Q33）

### Q28. 文本和单张图片从输入到模型是怎样融合的？

**30～60 秒口述回答**

Processor 将 messages 中的图片和文本一起模板化，文本侧生成包含 image placeholder 的 `input_ids`，视觉侧生成 `pixel_values` 和 `image_grid_thw`。Vision Encoder 把图片 Patch 编码，Patch Merger 得到与占位 Token 数对应的 Visual Embeddings。文本 Embedding 先处理整个 `input_ids`，再根据 image mask 把 placeholder 位置替换为 Visual Embeddings；融合后的序列进入同一个 Hybrid Decoder，并配合三轴 mRoPE position。Decode 阶段只追加文本 Token，不重复跑 Vision Encoder。

**项目实现细节**

- 文本路径：`input_ids → token embedding`。
- 图像路径：`pixel_values + image_grid_thw → vision encoder/merger`。
- 融合：按 `mm_token_type_ids`/image mask 将 visual embedding scatter/replace 到序列。

**连续追问链**

- 问：是 cross-attention 融合吗？答：不是，视觉特征被投影/合并为序列 embedding 后送入 Decoder。
- 问：图片原始像素直接进 LLM 吗？答：不，先经视觉编码器变成 embedding。
- 问：支持几张图？答：当前项目验证文本+单图，不声称多图/视频。

**证据**

- 单图输入与 Hugging Face Greedy Token 对齐。

**边界与红线**

- AutoProcessor 和 Vision 模块结构来自模型/Transformers，本人做的是引擎适配与数据流接入。

**一句话记忆点**：图像先变成与 placeholder 等长的 embeddings，再替换文本序列中的占位位置。

### Q29. Image Placeholder 的数量怎么确定？

**30～60 秒口述回答**

Placeholder 数量不能写死，它由图像预处理后的网格尺寸和 Vision Patch Merger 的压缩比例决定。Processor 根据 resize 后的 temporal-height-width 网格生成对应数量的 image tokens；Vision Encoder/Patch Merger 最终输出的 visual embedding 行数必须与 mask 选中的 placeholder 数严格一致，否则替换时 Shape 对不上，或位置编码与视觉 Token 错位。

**项目实现细节**

- `image_grid_thw` 描述每张图/视频的 T、H、W patch grid。
- merge 后 token 数与 merge factor 有关。
- 运行前检查 `num_image_placeholders == visual_embeddings.shape[0]`。

**连续追问链**

- 问：原图分辨率越大 placeholder 一定越多吗？答：通常是，但还受 Processor resize/min/max pixels 和 merge 规则影响。
- 问：为什么不能用固定特殊 Token 一个位置？答：一张图对应多行视觉特征，需要一一承载。
- 问：多个图怎么处理？答：应分图维护 grid 和边界，但当前未作为完成项。

**证据**

- 单图不同输入通过 processor/embedding shape 与 HF 输出对齐检查。

**边界与红线**

- 具体公式以当前 Qwen3.5 Processor 配置为准，不背一个跨模型通用常数。

**一句话记忆点**：placeholder 数量由预处理后的 patch grid 和 merger 决定，必须等于视觉 embedding 行数。

### Q30. `pixel_values`、`image_grid_thw`、`mm_token_type_ids` 分别传给谁？

**30～60 秒口述回答**

`pixel_values` 是归一化并组织后的视觉 Patch 输入，送入 Vision Encoder；`image_grid_thw` 描述视觉网格，用于 Vision Encoder/Patch Merger 恢复空间布局和计算视觉 position；`mm_token_type_ids` 或等价 image mask 与文本序列对齐，用于定位哪些 token 位置要被 Visual Embedding 替换，也帮助区分多模态位置规则。三者分别解决“图像内容”“图像几何”“文本序列中的落点”。

**项目实现细节**

- `pixel_values` 不传给普通 token embedding。
- `image_grid_thw` 不是最终 mRoPE position tensor，而是其构造输入之一。
- Decode 后续 step 不再携带整张图的 pixel tensor。

**连续追问链**

- 问：为什么 grid 不能从 pixel tensor Shape 直接推？答：动态 resize、flatten 和 batch packing 后需要显式保留原网格边界。
- 问：type ids 是 vocabulary token id 吗？答：不是，它是模态/位置辅助元数据。
- 问：visual cache 存什么？答：缓存 Vision Encoder 产生的视觉特征，避免相同图像重复编码。

**证据**

- 多模态输入 prepare 与 embedding replacement 测试覆盖这三类数据。

**边界与红线**

- 变量的精确命名可能随 Transformers 版本变化，回答职责而非死背 API。

**一句话记忆点**：pixels 给内容，THW 给几何，token type/mask 给融合落点。

### Q31. Image Embedding 最后如何填回文本 Placeholder？

**30～60 秒口述回答**

先对完整 `input_ids` 做 token embedding，得到 `[total_tokens, hidden_size]`；再用 image token id 或 `mm_token_type_ids` 生成布尔 mask，找到 placeholder 的扁平位置。Vision Encoder/Patch Merger 输出 `[num_visual_tokens, hidden_size]`，校验数量一致后，按序写入这些 mask 位置。这样 Decoder 看到的仍是一条统一的 embedding 序列，但视觉位置携带的是图像特征而不是特殊 token 的普通词向量。

**项目实现细节**

- 替换前必须对齐 dtype、device、hidden size 和 token count。
- Variable-length flatten 后 mask 仍对应全局扁平 Token 顺序。
- Prefix/position metadata 使用替换后的统一序列长度。

**连续追问链**

- 问：placeholder token embedding 会保留吗？答：被视觉 embedding 覆盖，不作为最终输入内容。
- 问：如果数量不等？答：立即报错，不能截断或广播，否则语义错位。
- 问：融合后还需要特殊 Attention mask 吗？答：仍按 Decoder 的因果序列规则，视觉 tokens 作为前缀位置参与。

**证据**

- HF Greedy Token 对齐证明替换顺序与位置处理一致。

**边界与红线**

- 不声称自研 Vision Encoder。

**一句话记忆点**：先嵌入整条文本，再把 image mask 选中的行等量替换成视觉特征。

### Q32. 普通 RoPE 和三轴 mRoPE 是什么关系？

**30～60 秒口述回答**

RoPE 通过与位置相关的旋转把相对位置信息注入 Q/K。纯文本通常只有一维 token position；多模态 mRoPE 将 head channel 的旋转维度分组，分别使用 temporal、height、width 三个位置轴。文本 token 没有二维空间结构，三个轴通常使用一致或按模型规则构造的位置；图像 token 则依据 patch grid 获得 H/W 坐标。它不是额外加三个 embedding，而是在不同通道段选择不同 position index 做 RoPE。

**项目实现细节**

- Graph Workspace 的 `positions` 是 `[3,max_B]`，体现三轴 decode position。
- Prefill 图像段根据 `image_grid_thw` 生成三轴位置。
- 文本/图像融合后必须维持连续且符合模型规范的 position delta。

**连续追问链**

- 问：为什么只旋转 Q/K？答：RoPE 用来改变注意力相似度中的位置信息，V 不需要同样旋转。
- 问：mRoPE 改变 attention complexity 吗？答：不改变 O(S²) 注意力结构，只改变位置编码。
- 问：GDN 层也直接做 Full Attention RoPE 吗？答：按模型各层实现走对应位置/算子路径，不能把 Full Attention 机制套到 GDN 数学。

**证据**

- 多模态与文本 Greedy Token 对齐；Graph 静态位置使用 3×B shape。

**边界与红线**

- 三轴通道切分比例必须以模型配置/实现为准，不凭印象给固定比例。

**一句话记忆点**：mRoPE 是把 RoPE 的位置从一维扩成 T/H/W 三轴，并分配给不同旋转通道。

### Q33. 为什么图像只在 Prefill 执行？如何做 Visual Cache 和 HF 对齐？

**30～60 秒口述回答**

图片作为 Prompt 的一部分，只需在 Prefill 时编码成视觉 Token 并写入模型历史；Decode 每步生成的是新文本 Token，过去视觉信息已经通过 KV/GDN State 影响后续计算，所以无需重复跑 Vision Encoder。Visual Embedding Cache 以图像内容和处理配置对应的 key 复用视觉特征。正确性上，我用同一 Processor、同一模型权重和 greedy sampling，与 Hugging Face 比较 token IDs，而不是只比较自然语言“看起来相似”。

**项目实现细节**

- Prefill：image encode → merge → placeholder replacement。
- Decode：只传新 token/position 和历史状态。
- Greedy 对齐要固定 sampling、prompt template、dtype 与停止条件。

**连续追问链**

- 问：Prefix Cache 能缓存图片 Prompt 吗？答：当前不能安全支持，Prefix Key 未包含图像 identity/layout。
- 问：Visual Cache 等于 Prefix Cache 吗？答：不是；前者复用视觉 encoder 输出，后者复用 Decoder 历史。
- 问：浮点误差会导致 token 不同吗？答：可能，因此逐 token greedy 对齐比只看 logits 误差更接近端到端语义验证。

**证据**

- 文本与单图路径均完成 HF Greedy Token 对齐。

**边界与红线**

- 不声称图像 Prefix Cache、多图或视频已完成。

**一句话记忆点**：视觉编码一次写入历史，Decode 只读历史；正确性用同配置逐 Token 对齐。

---

## 五、联合 Prefix Cache（Q34～Q41）

### Q34. 为什么普通 KV-only Prefix Cache 对 Hybrid 模型不正确？

**30～60 秒口述回答**

Prefix Cache 的含义是跳过前 N 个 Token 的计算，因此必须恢复模型在第 N 个 Token 后的全部历史。对纯 Attention 模型，这主要是 KV；Qwen3.5 Hybrid 还有 24 层 Conv/Recurrent State。如果只恢复 KV 并跳过 GDN Prefill，Attention 层处于 N 时刻，GDN 层却是零状态或旧请求状态，模型已经不是同一次前向。我的 Entry 因此在同一个 `num_cached_tokens` 边界联合保存物理 KV Blocks、Conv Snapshot 和 Recurrent Snapshot。

**项目实现细节**

- `PrefixStateEntry.kv_block_ids`
- `PrefixStateEntry.conv_state_snapshot`
- `PrefixStateEntry.recurrent_state_snapshot`
- 三者必须同边界 commit/restore。

**连续追问链**

- 问：只缓存 GDN 可以吗？答：也不行，8 个 Full Attention 层会缺 KV 历史。
- 问：为什么 Snapshot 不是每个 Token 一个？答：只在选定完整边界保存，控制容量和恢复成本。
- 问：命中后还算什么？答：只算 `num_cached_tokens` 之后的 suffix。

**证据**

- 1165-token Prompt 命中 1024 后只算 141 tokens，greedy output token 保持一致。

**边界与红线**

- 不把 KV-only Cache 描述为“性能差”，而是对 Hybrid 跳算在语义上不完整。

**一句话记忆点**：跳过多少 Token，就必须恢复这个边界的全部模型历史。

### Q35. `PrefixKey`、链式 Block Hash 和最长前缀查找如何工作？

**30～60 秒口述回答**

`PrefixKey` 由 `model_namespace`、当前 `block_hash` 和 `num_cached_tokens` 组成。Token 按 Block 切分，第 i 个 Block 的 Hash 不只包含本 Block tokens，还链入前一个 Block Hash，因此相同末块但不同历史不会被当成同一 Prefix。Lookup 沿请求的完整 Block 边界生成候选 Key，从最长边界向短边界查找，命中后还验证物理 Block 的链式 Hash/有效性，再返回最长安全前缀。

**项目实现细节**

- `model_namespace` 隔离不同模型/权重/配置。
- `block_hash_i = H(block_hash_{i-1}, tokens_i, relevant_metadata)` 的链式思想。
- `num_cached_tokens` 明确语义边界，避免相同 hash 条目混淆长度。

**连续追问链**

- 问：为什么从最长往短找？答：最大化跳过的 Prefill，同时短 Prefix 仍可兜底。
- 问：Hash 碰撞怎么办？答：不能只信 dict key，还要验证链式 block 元数据；碰撞测试要求拒绝错误命中。
- 问：为什么要 namespace？答：相同 Token 在不同模型权重下产生的状态不同。

**证据**

- `hash_collision` guard、longest-prefix 与 duplicate detection 测试通过。

**边界与红线**

- 当前 Key 不包含图像 identity/layout，因此只对纯文本开放 Prefix Cache。

**一句话记忆点**：链式 Hash 证明“从开头到这里都相同”，长度和 namespace 再限定语义空间。

### Q36. 一个 Token 最后如何映射到物理 KV Block？两类引用计数怎么工作？

**30～60 秒口述回答**

逻辑上，token position 先除以 block size 得到逻辑 block index，取模得到 block 内 offset；再通过 Sequence 的 `block_table[logical_block]` 得到物理 block id，最终由 Attention layout 计算 K/V 层、head 和 offset 地址。请求运行时持有 request reference；Prefix Entry 需要长期固定这些物理块时持有 cache reference。项目中 `ref_count` 是总引用，`cache_ref_count` 是缓存部分，所以请求引用可由二者之差判断。只有两类引用都归零才能回收。

**项目实现细节**

- `logical_block = token_position // block_size`
- `block_offset = token_position % block_size`
- `physical_block = block_table[logical_block]`
- request refs 与 cache refs 生命周期独立。

**连续追问链**

- 问：Entry A/B 共享前半段怎么办？答：同一物理 Block 的 cache ref 可被多个 Entry 持有，容量按唯一 Block 统计。
- 问：请求结束但 Entry 还在？答：只减 request ref，cache ref 保证 Block 不被复用。
- 问：逐出 Entry 会立即释放所有块吗？答：只减少它的 cache ref，仍被其他请求/Entry 引用的块保留。

**证据**

- 4096/8192 层级 Prefix 共享测试验证 unique pinned block 与引用回收。

**边界与红线**

- 物理 KV 的具体维度布局由 Attention 实现决定，Block Table 只解决分页映射。

**一句话记忆点**：除法找逻辑块、Block Table 找物理块、双引用决定何时可回收。

### Q37. Prefix Entry 是如何 Commit 和 Restore 的？

**30～60 秒口述回答**

Commit 只发生在满足缓存策略的完整 Token/Block 边界。先确认对应 KV Blocks 已完成并增加 cache refs，再从当前活动 `state_slot` 克隆 24 层 Conv/Recurrent State，按配置将 recurrent Snapshot 转 BF16，组成只读 Entry。Restore 时先让新请求取得这些 KV Blocks 的 request refs，再把 Snapshot 拷贝到它自己的活动 Slot，设置 `num_cached_tokens`，随后只调度 suffix。Snapshot 不能与活动 Slot 共用内存，因为 Slot 会被后续请求更新。

**项目实现细节**

- Commit 数据源：完成边界后的活动 state + 已写完的 KV blocks。
- Restore 目标：新分配的活动 state slot。
- duplicate key：复用已有 Entry，不重复 Snapshot/引用。

**连续追问链**

- 问：为什么要 clone？答：保证 Snapshot independence，活动 Slot 原位更新不影响缓存。
- 问：BF16 Restore 到 FP32 Pool 怎么办？答：拷贝时转换回 FP32 recurrent，Conv 本身为 BF16。
- 问：Commit 失败如何回滚？答：不能留下只增加一半引用或只有部分 Snapshot 的 Entry，应事务式清理。

**证据**

- Prefix commit 测试验证 Snapshot independence、duplicate detection 和 KV ownership release。

**边界与红线**

- Snapshot 的 BF16 是缓存容量优化，不代表活动 recurrent 也改为 BF16。

**一句话记忆点**：Commit 把活动状态冻结成独立快照，Restore 把快照复制到新请求自己的 Slot。

### Q38. 如何支持最长前缀和层级 Prefix 的物理 Block 共享？

**30～60 秒口述回答**

同一 Prompt 可以有 4096 和 8192 两个缓存边界，后者的前 16 个 KV Blocks 与前者完全相同，因此 Entry 不复制 KV 数据，只各自保存物理 Block ID 列表并增加引用。Lookup 从 8192 向 4096 搜索，优先恢复最长命中。容量统计对 KV 使用唯一物理 Block 集合，对每个边界的 GDN Snapshot 单独计费，因为 4096 和 8192 的 recurrent/conv state 数值不同、不能共享。

**项目实现细节**

- 4096 tokens=16 blocks，8192=32 blocks（block size 256）。
- 两 Entry 合计 unique KV blocks=32，而非 48。
- 删除短 Entry 只回收它独有的 Snapshot；共享 KV 仍被长 Entry 固定。

**连续追问链**

- 问：为什么 GDN Snapshot 不能像 KV 一样共享前半段？答：它是边界处压缩后的整体状态，不是逐 Token Block。
- 问：最长命中一定最好吗？答：计算复用最大，但仍需满足 Entry 有效性和资源恢复条件。
- 问：删除长 Entry 后短 Entry还有效吗？答：只要其 KV refs/Snapshot 仍在，就有效。

**证据**

- shared-block 测试：正确容量 307.0 MiB，而 naive duplicated 容量 435.0 MiB；8192 最长前缀恢复正确。

**边界与红线**

- “共享”指 Entry 引用同一物理 KV Blocks，不是跨不同 Token 内容做近似复用。

**一句话记忆点**：层级 Prefix 共享逐块 KV，但每个语义边界保留自己的 GDN Snapshot。

### Q39. LRU 和容量计费如何实现？为什么不能只按 Entry 数量限制？

**30～60 秒口述回答**

Entry 大小并不相等：Prefix 越长固定的 KV Blocks 越多，而且层级 Entry 还能共享物理块，所以按 Entry 数量限制会严重失真。我按“唯一 pinned KV bytes + 各 Entry Snapshot bytes”计容量，命中时更新 LRU recency；容量不足时从最久未使用且可安全逐出的 Entry 开始释放 cache refs 和 Snapshot，直到满足新 Entry。若单个候选本身超过总容量，则直接拒绝，不反复逐出整个 Cache。

**项目实现细节**

- LRU 只更新使用顺序，不改变 Hash 语义。
- reclaimable bytes 要考虑共享 Block：只计算逐出后引用真正归零的物理块。
- capacity rejection 与 eviction 是两条路径。

**连续追问链**

- 问：为什么不是 LFU？答：LRU 控制近期复用，frequency admission 已在进入前过滤一次性前缀。
- 问：逐出正在被请求用的 Entry？答：可删缓存所有权，但不能回收仍有 request ref 的物理块。
- 问：显存容量看 allocated 还是逻辑 bytes？答：Cache 策略用可解释的逻辑占用；整体 OOM 还受 allocator reserved/碎片影响。

**证据**

- LRU order、eviction、capacity rejection 和 shared capacity tests 全部通过。

**边界与红线**

- 不把逻辑 capacity 等同于 `nvidia-smi` 总变化。

**一句话记忆点**：LRU 决定淘汰谁，唯一物理块与快照字节决定到底占多少。

### Q40. Frequency Admission 如何判断热前缀并防止缓存污染？

**30～60 秒口述回答**

如果所有够长 Prompt 第一次出现就 Commit，一个 25.5 MiB Snapshot 加大量 KV 很容易被一次性请求污染。我的 admission history 先按候选 `PrefixKey` 记录观察次数，达到 `admission_min_observations` 才允许进入正式 Cache；候选历史由 `admission_max_candidates` 限制，避免元数据无限增长。正式 Cache 再由 LRU 管最近使用。也就是 frequency 决定“值不值得进”，LRU 决定“进来后谁先出”。

**项目实现细节**

- `admission_history[PrefixKey] → observation count/recency`。
- `admission_min_observations` 是热度阈值。
- `admission_max_candidates` 控制候选元数据上限。

**连续追问链**

- 问：阈值越高越好吗？答：不是；污染更少但首次/低频复用无法获益，需要用 Trace 调参。
- 问：Agent System Prompt 如何变热？答：不同请求反复出现相同链式 PrefixKey，观察次数自然累积。
- 问：历史也要 LRU 吗？答：需要容量/淘汰策略，否则只是不存 Snapshot，却让候选表膨胀。

**证据**

- Admission correctness test 与 Trace Benchmark 验证冷候选拒绝、热前缀进入及污染控制。

**边界与红线**

- 这是频率门槛+LRU，不是学习型缓存策略，也未声称参数对所有流量最优。

**一句话记忆点**：先用频率过滤一次性 Prompt，再用 LRU 管已经证明有复用价值的 Entry。

### Q41. Agent System Prompt 场景和 16K 实验如何解释？

**30～60 秒口述回答**

Agent 服务通常有稳定的 System Prompt、工具描述和策略说明，后面才拼接每个用户的私有输入，因此天然存在跨请求共享长前缀。我的 16K Trace 固定 12K 共享前缀并设置 50% 命中率，模拟冷热请求混合，而不是只比较一次 Cold 和一次 Hot。RTX 5090、Qwen3.5-9B 下平均 TTFT 降低 36.66%。它证明的是复用前缀能减少 Prefill 工作；收益小于纯 Hot 的跳算比例，是因为一半请求不命中，而且仍需计算 suffix、调度与首 Token。

**项目实现细节**

- Context=16K、shared prefix=12K、hit rate=50%。
- 命中只跳过 shared prefix，真实 position 和 suffix 仍正常计算。
- 评价重点是 Trace 平均 TTFT，也应观察容量、命中率和输出正确性。

**连续追问链**

- 问：为什么不是 94% TTFT 降幅？答：那是单次近全量 Hot Prefix 的机制上限，不是 50% 混合流量口径。
- 问：Prefix Cache 会降低 TPOT 吗？答：主要影响 Prefill/TTFT，稳定 Decode TPOT 不应被当作核心收益。
- 问：命中率更高收益一定线性吗？答：不严格线性，还受 suffix、Batch、调度和内存压力影响。

**证据**

- 最终简历口径：16K/12K/50%，平均 TTFT -36.66%；另有逐 Token correctness 与 capacity tests。

**边界与红线**

- 不把 Prefix 复用叫“KV 压缩”；不把纯 Hot 94.1% 结果替换为生产 Trace 结论。

**一句话记忆点**：Agent 的固定长 System Prompt 是热前缀，最终数据用 50% 命中 Trace 而不是理想化全 Hot。

---

## 六、Hybrid CUDA Graph（Q42～Q48）

### Q42. Eager 和 CUDA Graph 有什么区别？

**30～60 秒口述回答**

Eager 每个 Decode Step 都由 CPU 逐个发起 PyTorch/CUDA 算子，Kernel 本身很短时，Python、Dispatcher 和 launch latency 会成为明显开销。CUDA Graph 先把一段固定 GPU 工作捕获成图，之后只需更新静态输入内容并一次 replay，省掉重复的 CPU launch。它不改变模型数学，也不会自动让单个 Kernel 算得更快；收益主要来自降低每 Token 都重复发生的提交开销。

**项目实现细节**

- 捕获主体是 Hybrid Decoder 的 Decode 主干。
- LM Head/Sampling 保持 Graph 外执行，便于处理动态 token 选择。
- `ModelRunner.run_hybrid_decode()` 根据 Policy 选择 replay 或 Eager。

**连续追问链**

- 问：Graph 为什么更适合 Decode？答：Decode 每步 Shape 相近、重复次数高且小 Kernel 多。
- 问：Graph 会融合 Kernel 吗？答：不会自动融合，只复用已捕获的 launch DAG。
- 问：Graph 是否总更快？答：不一定；大 Kernel 已占主导、填充浪费或拷贝开销过大时收益会小。

**证据**

- B=16 自研 CUDA Eager 1048.69 tok/s，Graph 1087.37 tok/s；端到端 Graph correctness 通过。

**边界与红线**

- 不把 Graph 收益归因于数学算子加速。

**一句话记忆点**：Eager 每步重新发命令，Graph 把固定 Decode 命令录好后重复播放。

### Q43. 为什么 CUDA Graph 要求固定地址和相对静态的 Shape？

**30～60 秒口述回答**

捕获时 Graph Node 会记录 Kernel 参数、依赖关系和 Tensor 指针；replay 时不能随意换成另一块地址或另一种 launch 配置，否则图中记录的指针/Shape 就失效。解决方式不是让请求数据固定，而是预分配静态 Workspace：地址和容量固定，每步把新的 token、position、KV metadata 与 State Slot IDs 拷入同一 Tensor。值可以改变，地址和捕获 Shape 保持不变。

**项目实现细节**

- 静态地址：`HybridDecodeStaticWorkspace` 内各 Tensor。
- 动态内容：`input_ids/positions/block_tables/state_slot_ids` 等。
- 动态 B：用多个精确 Bucket，而不是改变同一 Graph 的 launch Shape。

**连续追问链**

- 问：模型权重为什么可直接用？答：推理期间权重地址稳定。
- 问：State Pool 可以原位更新吗？答：可以，它本身地址稳定，Slot ID 内容决定访问位置。
- 问：为什么不能每次新建 Tensor 再传进去？答：新 Tensor 指针可能变化，破坏捕获假设。

**证据**

- Workspace address/shape/padding 测试与 capture/replay smoke test 通过。

**边界与红线**

- “固定 Shape”是单个 Graph 的约束，不代表整个服务只支持一个 Batch。

**一句话记忆点**：Graph 固定的是地址和执行拓扑，不是每次请求的数据值。

### Q44. Graph 静态 Workspace 里有哪些关键 Tensor？

**30～60 秒口述回答**

`HybridDecodeStaticWorkspace` 把 Decode 主干会变化的输入预分配到最大 Bucket：`input_ids[max_B]`、三轴 `positions[3,max_B]`、`slot_mapping[max_B]`、`context_lens[max_B]`、`block_tables[max_B,max_num_blocks]`、`state_slot_ids[max_B]` 和 `hidden_states[max_B,hidden_size]`。Replay 前只覆盖前 B 行并按策略填充剩余区。Attention 依靠 KV metadata，GDN 依靠 State Slot，二者都能在固定地址条件下访问动态请求历史。

**项目实现细节**

- `input_ids/positions/state_slot_ids`：INT64。
- `slot_mapping/context_lens/block_tables`：按实现使用 INT32。
- `hidden_states`：模型 Decode dtype，通常 BF16。

**连续追问链**

- 问：为什么 position 是 3×B？答：Qwen3.5 多模态使用三轴 mRoPE。
- 问：为什么 Block Table 需要 max blocks？答：不同请求上下文长度不同，需要规则二维静态容量。
- 问：这些 Tensor 很占显存吗？答：B16 workspace 实测约 0.1259 MiB，Graph 私有池才是更大的额外项。

**证据**

- `test_hybrid_graph_workspace.py` 验证了 dtype、shape、复用与 padding。

**边界与红线**

- 0.1259 MiB 只能描述显式 Workspace，不能描述 Graph 总显存。

**一句话记忆点**：静态 Workspace 同时固定 Token、KV 元数据、GDN Slot 与 Hidden 输出四类接口。

### Q45. B=1/2/4/8/16 Bucket 如何捕获、选择和回退？

**30～60 秒口述回答**

初始化或首次使用时，`capture_hybrid_cudagraph()` 对 B=1、2、4、8、16 分别准备静态输入并 warm-up，再捕获各自的 Decoder DAG。运行时 `HybridDecodeGraphPolicy` 根据实际 Batch size 选择精确匹配的 `HybridDecodeGraphRoute`；例如 B=8 使用 B8 Graph。当前策略不把 B=3 padding 到 B4，因为假请求也涉及 KV/GDN 状态语义，所以非标准 B 明确回退 Eager，保证正确性和可解释性。

**项目实现细节**

- 精确 buckets：`(1,2,4,8,16)`。
- Graph key 至少区分 batch size，并绑定对应 workspace/graph output。
- B>16 或 B=3/5 等走 `run_hybrid_decode()` 的 Eager route。

**连续追问链**

- 问：为什么不只捕获 B16？答：小 Batch padding 到 B16 会做大量无效工作，还要构造安全 dummy state。
- 问：捕获五个 Graph 的代价？答：初始化更慢，Graph 私有池/Executable 占额外显存。
- 问：以后能支持任意 B 吗？答：可用向上 Bucket+安全 padding，或扩展更多 bucket，但要评估浪费和状态正确性。

**证据**

- Bucket matrix 覆盖 B=1/2/4/8/16，另验证 unbucketed fallback。

**边界与红线**

- 简历说的是五个 Bucket，不是任意 Batch Graph。

**一句话记忆点**：标准 B 精确命中 Graph，非标准 B 宁可 Eager 也不引入伪请求状态。

### Q46. Continuous Batching 下 Graph 如何保持 KV 和 GDN 状态正确？

**30～60 秒口述回答**

Continuous Batching 改变的是每步 Batch row 对应哪个 Sequence。Replay 前，ModelRunner 把当前请求的物理 `block_tables/context_lens/slot_mapping` 和 `state_slot_ids` 复制到静态 Workspace；Graph 内 Attention 按新 KV 元数据寻址，GDN Kernel 按新 Slot ID 直接寻址稳定的 State Pool。因此即使上一轮 Batch row 0 是请求 A、下一轮变成请求 D，Graph 地址不变，但里面的映射值已更新，不会继承 A 的历史。

**项目实现细节**

- `replay_hybrid_cudagraph()` 更新静态输入再 replay。
- Context 写入 `gdn_state_slot_ids/gdn_recurrent_state_pool/gdn_conv_state_pool`。
- Pool base address 固定，slot content 随请求原位演进。

**连续追问链**

- 问：请求完成后 row 如何处理？答：下一 Step 重建实际 Batch；如果换 Bucket 则选择另一个 Graph。
- 问：同一个 Slot 能同时给两个请求吗？答：不能，活动 Slot 所有权必须唯一。
- 问：元数据 copy 会抵消收益吗？答：有成本，但远小于重放整条 Eager launch 链，需 Benchmark 验证。

**证据**

- 连续批处理测试覆盖请求动态进入/退出和状态一致性。

**边界与红线**

- Graph 只保证执行复用，Slot 生命周期仍由 Scheduler/State allocator 保证。

**一句话记忆点**：Graph 地址静态、请求映射动态，间接寻址把两者接起来。

### Q47. 为什么优化 Gather/Scatter 后还需要 Graph Workspace copy？

**30～60 秒口述回答**

要区分大状态搬运和小元数据拷贝。原 FLA 路径把每层 `[B,H,Dk,Dv]` recurrent matrix 和 Conv window 从 State Pool Gather 出来，再把 final state Scatter 回去，24 层每步搬运大量数据。State-Aware Kernel 消除了这两次大状态中间量，直接读写 Pool。Graph replay 前仍要把少量 token、position、block table 和 slot ids 复制到固定 Workspace，这是 CUDA Graph 固定地址所需的 staging，规模远小于 recurrent state，不能把两者混为一谈。

**项目实现细节**

- 被消除：GDN state `index_select/index_copy_` 与 conv D2D temporary copy。
- 仍保留：动态输入 → 静态 Graph Workspace 的 metadata/hidden staging。
- State Pool 本体不复制进 Graph Workspace。

**连续追问链**

- 问：能连 metadata copy 也消除吗？答：可让上游直接写静态 buffer，但会扩大接口侵入，需要权衡。
- 问：Graph 内能直接 Gather 吗？答：能捕获但仍会付状态搬运成本，不能解决根因。
- 问：最大的被消除量是什么？答：FP32 recurrent matrices 的每层 Gather/Scatter。

**证据**

- Profile 中 FLA 状态搬运 19.669 ms/8 steps，自研路径对应 index_select/index_copy 调用为零。

**边界与红线**

- 不说“所有 copy 都消失了”；消失的是 GDN 大状态重排。

**一句话记忆点**：保留的是小而必要的 Graph 入参 staging，消除的是每层大状态 Gather/Scatter。

### Q48. CUDA Graph 的收益和显存代价怎么解释？

**30～60 秒口述回答**

最终 B16 自研 CUDA Graph 达到 1087.37 tok/s、TPOT 14.71 ms；相对自研 Eager 的 1048.69 tok/s约提升 3.69%，说明 Kernel 已融合后 Graph 仍能减少 launch 开销。双方都开 Graph 时，自研相对 FLA 提升 1.99%～18.31%，这是公平的后端比较。显存方面，显式 B16 Workspace 只有约 0.1259 MiB，但捕获 allocated 增量约 48.88 MiB，Reserved/Graph private pool 可能更大，所以我会同时报告性能、初始化和显存代价。

**项目实现细节**

- 小 Batch launch 占比高，但具体相对收益受两个后端 Kernel 数量影响。
- B16 P99 step latency：17.446→14.748 ms（自研 vs FLA Graph），降低 15.47%。
- 多 Bucket 的 Graph pool 是以显存换低 launch latency。

**连续追问链**

- 问：为什么 Graph 后自研相对优势范围变小？答：Graph 同样帮 FLA 降低了很多 Python/launch 开销，剩余差距更接近状态搬运和 Kernel 路径。
- 问：1087 tok/s 如何换算 TPOT？答：B16 每 step 生成约16 tokens，`16/0.01471≈1087 tok/s`。
- 问：何时禁用 Graph？答：显存紧张、Shape 高度动态或捕获不兼容时回退 Eager。

**证据**

- 最终 Benchmark 5 次重复、128 output tokens；B16 P99 与显存数据均有记录。

**边界与红线**

- CUDA+Graph 对 FLA+Eager 的 30.97%～46.28%是联合系统收益，不能叫纯 Graph 或纯 Kernel 收益。

**一句话记忆点**：Graph 用额外捕获显存换 Decode launch 复用，1087 tok/s 是 B16 端到端结果。

---

## 七、State-Aware GDN CUDA Kernel（Q49～Q59）

### Q49. 为什么要做 State-Aware Kernel？原 FLA 路径慢在哪里？

**30～60 秒口述回答**

FLA 的 Decode API 接收连续 `[B,H,Dk,Dv]` Batched State，但引擎为了 Continuous Batching 维护的是 `[num_slots,24,H,Dk,Dv]` 动态 State Pool，本轮请求的 Slot 可能是 `[7,2,11,…]`。原路径每个 GDN 层先 `index_select` Gather 成连续中间 Tensor，FLA 产生 final state 后再 `index_copy_` Scatter 回 Pool；Conv 也有类似搬运。我的 Kernel 把 `state_slot_ids` 和 `gdn_index` 作为输入，直接定位 Pool 中每个请求、每层的物理状态，融合数学计算和原位回写。

**项目实现细节**

- 原链：Pool → Gather batched state → FLA → final state → Scatter → Pool。
- 新链：Pool + slot ids → State-Aware CUDA → Pool in-place + output。
- 优化目标是接口/布局，而不只是重写同一数学 Kernel。

**连续追问链**

- 问：Batched State 是列表吗？答：不是，是规则连续 Tensor `[B,H,Dk,Dv]`。
- 问：final state 是中间量吗？答：原 FLA 路径是与 Batched State 同 Shape 的中间 Tensor，需要再 Scatter。
- 问：为什么不让 State Pool 永远按 Batch 顺序？答：Continuous Batching 每步请求进入退出，移动整池会更贵且破坏稳定 Slot。

**证据**

- Profile：FLA 每 8 steps 的状态搬运共 19.669 ms，即约 2.459 ms/step。

**边界与红线**

- FLA 数学 Kernel 本身很快；问题是它的连续接口与本引擎布局不匹配。

**一句话记忆点**：不是 FLA 算错或太慢，而是为了喂它连续 State，动态引擎每层都要重排大状态。

### Q50. `state_slot_ids` 和 `gdn_index` 如何完成直接寻址？

**30～60 秒口述回答**

对 Batch 行 `b`，先取 `slot = state_slot_ids[b]`；当前 Decoder 层已经知道它在 24 个 GDN 层中的稠密编号 `gdn_index`。Recurrent Pool 逻辑地址是 `pool[slot,gdn_index,head,key_dim,value_dim]`，Conv Pool 是 `pool[slot,gdn_index,channel,tap]`。Kernel 把多维坐标按 stride 展平到元素 offset。这样 Batch 顺序随调度改变时只需改变 `state_slot_ids` 的值，不复制整个状态。

**项目实现细节**

- Recurrent base offset：`((slot*24 + gdn_index)*H + h)*Dk*Dv`。
- Conv base offset：`((slot*24 + gdn_index)*C + c)*K`。
- `state_slot_ids` 是 INT64 `[B]`；`gdn_index∈[0,23]`。

**连续追问链**

- 问：为什么 `gdn_index` 不是 global layer index？答：Pool 只存 24 个 GDN 层，稠密编号避免给 8 个 Attention 层留空。
- 问：Slot ID 能重复吗？答：一个活动 Batch 中不同请求不应共享活动 Slot。
- 问：地址计算错了会怎样？答：可能污染别的请求/层，因此必须检查 untouched slot/layer。

**证据**

- 乱序 multi-batch Slot 与 untouched state 测试通过。

**边界与红线**

- Direct addressing 消除重排，不代表省掉读取/写回状态本身的必要显存流量。

**一句话记忆点**：Batch 行通过 Slot 找请求，`gdn_index` 再找该请求的 GDN 层。

### Q51. 两个 State Pool 与 Kernel 输入输出的 Shape 是什么？

**30～60 秒口述回答**

Recurrent Decode 输入 `query/key/value: [B,1,32,128]` BF16，`g` FP32、`beta` BF16，`state_slot_ids:[B]` INT64；Pool 是 `[num_slots,24,32,128,128]` FP32，输出 `[B,1,32,128]` BF16。Conv 输入 `x:[B,8192,1]` BF16、`weight:[8192,4]` BF16；Pool 是 `[num_slots,24,8192,4]` BF16，输出与 x 对应。这里的 H/Dk/Dv 是 GDN 内部维度，不能与 Full Attention 的 query/kv head 配置混淆。

**项目实现细节**

- Recurrent 活动 Slot：48 MiB。
- Conv 活动 Slot：1.5 MiB。
- Output 支持外部静态 workspace，以便 CUDA Graph 复用。

**连续追问链**

- 问：为什么 state 是 FP32、输入输出是 BF16？答：递归累积对误差敏感，输入/输出按模型精度节省带宽。
- 问：为什么 query 有长度维 1？答：当前 Kernel 专门处理单 Token Decode。
- 问：Dk 和 Dv 必须相同吗？答：数学上不必；当前模型和特化实现都是 128。

**证据**

- Extension shape/dtype、多 Batch、多步和输出 workspace tests 通过。

**边界与红线**

- 当前不是动态 H/Dk/Dv/C/K 的通用 Kernel。

**一句话记忆点**：Recurrent 是 FP32 五维大池，Conv 是 BF16 四维短窗池，Decode 输入长度固定为 1。

### Q52. 请推导单 Token Gated Delta Rule

**30～60 秒口述回答**

对一个 head，旧状态 `S∈R^{Dk×Dv}`。先用门控衰减得到 `S_decay = αS`，其中 α 由 g 转成 0～1 的衰减系数。Key 从当前状态预测 value：`v_hat = kᵀS_decay`；再得到校正量 `delta = beta·(v-v_hat)`。用外积写回 `S_new = S_decay + k·deltaᵀ`，最后输出 `o = qᵀS_new`（精确归一化/门控顺序按模型实现）。它相当于让 State 对当前 key-value 关联做一次带学习率 beta 的在线修正。

**项目实现细节**

- `S` shape `[128,128]`，k/q 沿 Dk，v/delta/output 沿 Dv。
- `v_hat[value]` 需要对 128 个 key 维做归约。
- 更新每个 `S[key,value]` 后，output[value] 再对 key 维归约。

**连续追问链**

- 问：为什么是外积？答：k 的每个 key 分量都要对整条 value correction 更新一行状态。
- 问：beta 做什么？答：控制新观测相对旧状态的修正强度。
- 问：g 做什么？答：控制旧状态随时间衰减，使模型选择性遗忘。

**证据**

- CUDA 与 PyTorch reference 在 8 个递归步上保持小误差。

**边界与红线**

- 面试时先讲结构，再以当前源码确认 α 的具体变换和 normalization，不能凭公式省掉模型细节。

**一句话记忆点**：先衰减旧 State，再用 key 预测 value，用预测误差做 rank-1 更新，最后用 query 读新 State。

### Q53. Warp-level Split-K 如何映射 H、Dk、Dv？

**30～60 秒口述回答**

当前 Shape 的一个核心工作单元是一个 `(batch,head)` 的 128×128 State。Block 中多个 Warp 沿 Dk=128 做 Split-K，每个 Warp 负责一段 key rows；lane 沿 Dv 方向以 float4 处理连续的 4 个 value 元素。每个 Warp 先对自己的 key 分片计算 `v_hat` 和 output 的部分和，再通过 Warp Shuffle 做 Warp 内归约，必要时用少量 Shared Memory 合并多个 Warp 的部分结果。这样既并行利用 Dk，又让每个 lane 的 Dv load/store 连续。

**项目实现细节**

- Grid 逻辑至少覆盖 `B×H`。
- `lane_id=threadIdx.x%32`，`warp_id=threadIdx.x/32`。
- Dv 连续布局使 float4 对 `value_index…value_index+3` 合并访存。

**连续追问链**

- 问：矩阵按行存，线程为什么看起来负责一列？答：线程按 key row 循环推进，在每行取相同的一组 value columns；单次 load 连续，跨行有 stride，最终负责的是列组输出。
- 问：Split-K 为什么需要归约？答：多个 Warp 分别得到同一 value output 的 key-dimension partial sum。
- 问：B 很小时如何增加并行度？答：把单个 head 的 Dk 工作拆给多个 Warp，而不是只用一个 Warp 串行 128 行。

**证据**

- Split-K 相对 Full-Warp 在多数 B 上更快，大规模 B=2048 仍约 1.08×。

**边界与红线**

- 这是针对 128×128 Shape 的映射，不声称对任意矩阵最优。

**一句话记忆点**：Warp 切 key rows，lane 持有连续 value 四元组，先算部分和再合并。

### Q54. 为什么最终选 Warp Shuffle 两级归约，而不是纯 Shared Memory Reduction？

**30～60 秒口述回答**

纯 Shared Memory Reduction 要把每个线程部分和写入 shared、同步、再分阶段读取归约，存在更多 shared traffic 和 block barrier。Warp Shuffle 让同一 Warp 的寄存器值通过 `__shfl_down_sync` 交换，不经过 shared，也不需要 block-wide sync；只有跨 Warp 时才让各 Warp leader 写少量 shared，再由一个 Warp 做第二级归约。我们的对照还试过 Block Reduction、Shared State Staging、thread tile 等，最终实测当前 Shape 下 Warp Shuffle 路径更稳定，因此保留测出来的方案。

**项目实现细节**

- 一级：Warp 内 offsets 16/8/4/2/1 shuffle-down。
- 二级：每 Warp 一个 partial 写 shared，`__syncthreads()` 后由首 Warp 合并。
- Shared Memory 只承载 Warp partial，而不是全 Block 每线程数据。

**连续追问链**

- 问：Warp 间通信有没有成本？答：有，所以用 Shared Memory+一次 barrier，但数据量从每线程降为每 Warp。
- 问：Shared Reduction 何时可能更好？答：归约规模/数据复用不同，或需要整个 Block 多次共享数据时。
- 问：为什么不只看理论？答：occupancy、寄存器、访存和架构共同作用，必须同 workload A/B。

**证据**

- Reduction benchmark 后最终记录采用 Warp Shuffle；替代方案未带来稳定收益。

**边界与红线**

- 不说 Warp Shuffle 永远优于 Shared Memory，只说当前特化 Shape/RTX 5090 的实测选择。

**一句话记忆点**：寄存器内完成 Warp 归约，Shared 只负责 Warp 之间的最后一小步。

### Q55. `float4` 128-bit 与 BF16×4 向量访存分别优化了什么？

**30～60 秒口述回答**

Recurrent State 是 FP32，Dk×Dv 的最内层 Dv 连续，因此把四个 float 作为 `float4` 一次 16-byte load/store，可减少内存指令并提高合并访问效率；每个 lane 处理一个连续 value 四元组。Conv State 每 channel 有连续 K=4 个 BF16，总共 8 bytes，因此用 BF16×4 的 64-bit 向量一次读写整个窗口。向量化要求起始地址对齐、最内层 stride=1，并处理好非整除边界；当前 Shape 恰好满足特化条件。

**项目实现细节**

- Recurrent：4×FP32=16 bytes，128-bit。
- Conv：4×BF16=8 bytes，64-bit。
- Q/K/V 的 mixed-QKV view 可能整体不连续，不能只靠 `data_ptr + b*logical_size`。

**连续追问链**

- 问：float4 一定减少显存字节吗？答：不减少必要字节，减少指令数并改善事务利用率。
- 问：地址不对齐怎么办？答：回退 scalar/较窄 vector，或确保 allocator/layout 对齐。
- 问：为什么 Conv 不用 float4？答：其 state 是 BF16，4 元素只有 8 bytes，类型和宽度不同。

**证据**

- ptxas 无 spill；BF16/FP32 correctness 与 vectorized benchmark 通过。

**边界与红线**

- “128-bit”描述一次访问宽度，不等于计算使用 128-bit 浮点精度。

**一句话记忆点**：沿最内层连续维打包：FP32 四个一组 16B，BF16 四个一组 8B。

### Q56. 非连续 mixed-QKV Tensor 为什么容易读错，如何支持真实 Stride？

**30～60 秒口述回答**

真实模型常先生成一个大的 mixed-QKV Tensor，再用 slice/view 得到 query、key、value。以 query 为例 Shape 是 `(2,1,32,128)`，但 stride 可为 `(12288,4096,128,1)`，它不是 contiguous；从 batch 0 到 batch 1 要跨过整段 mixed-QKV 的 12288 元素，而不是 query 自身的 4096。CUDA Launcher 因此把各维 stride 传给 Kernel，地址按 `b*stride_b + l*stride_l + h*stride_h + d*stride_d` 计算；output 则使用连续静态 Workspace。

**项目实现细节**

- 必须满足最内层 `stride(-1)==1` 才安全做当前向量访问。
- Query/Key/Value 各自传真实 stride，不能假设三者相同 base offset。
- `torch.empty(shape,dtype,device)` 创建连续 output；Graph 可显式传入静态 output。

**连续追问链**

- 问：为什么单 B 测试没暴露？答：B=1 时错误 batch stride 没有跨行访问。
- 问：直接 `.contiguous()` 可以吗？答：正确但新增整 Tensor copy，违背消除搬运目标。
- 问：任意 stride 都支持吗？答：当前支持真实 batch/head 等 stride，但向量化仍要求最后一维连续。

**证据**

- 打印并断言 non-contiguous query stride，B=1～16 Extension correctness 通过。

**边界与红线**

- 不声称支持任意负 stride/任意 layout；明确最后一维约束。

**一句话记忆点**：Shape 告诉有多少元素，Stride 才告诉下一个 Batch 的数据到底在哪里。

### Q57. State-Aware Causal Conv Kernel 具体融合了什么？

**30～60 秒口述回答**

Kernel 为每个 `(batch,channel)` 分配一个线程。线程用 `state_slot_ids[b]` 和 `gdn_index` 找到该 channel 的 4 个历史 BF16 值，一次向量读取后做左移并追加当前 x，和 `weight[channel,0:4]` 做 depthwise 4-tap 点积，FP32 累加后经过 SiLU，写入 BF16 output，同时把更新后的四元窗口原位写回 Conv Pool。它把原来的 Gather、状态 shift/copy、官方 causal-conv、SiLU 和 Scatter 串成一个 Kernel。

**项目实现细节**

- Shape：C=8192、K=4；block size 最终固定 256。
- 一维 work index 展开为 `batch_index=work_index/C`、`channel_index=work_index%C`。
- 无 channel 间 reduction，因此映射比 recurrent 更直接。

**连续追问链**

- 问：为什么不同 block size 差别不大？答：工作规则、每线程较轻，B1～16 下 launch/总线程数主导，64～512 都接近。
- 问：为什么能 3.21×？答：主要消除多个框架算子和状态 Gather/Scatter，不是 4-tap 点积本身复杂。
- 问：in-place 会有 race 吗？答：每个 `(slot,layer,channel)` 只由一个线程更新，活动 slot 唯一。

**证据**

- 子路径由约 355 μs 降到约 111 μs，3.21×；状态误差 0，最大输出误差 7.63e-6。

**边界与红线**

- 3.21×只属于 Conv 状态子路径，不是完整模型。

**一句话记忆点**：一个线程包办一个 channel 的读窗、移位、4-tap、SiLU 和原位回写。

### Q58. Python → C++ Binding → CUDA Launcher → current stream 的调用链是什么？

**30～60 秒口述回答**

Python 侧 `load_state_aware_gdn_cuda_extension()` 用 PyTorch C++ Extension 编译/加载 binding 与 `.cu`；模型调用 `state_aware_gdn_decode_cuda()` 或 `state_aware_causal_conv1d_cuda()`。Binding 通过 `PYBIND11_MODULE` 暴露 `state_aware_gdn` 等函数，进入 C++/CUDA launcher 后读取 Tensor data pointer、shape、stride 和 device，取得 `c10::cuda::getCurrentCUDAStream(device_index)`，最后用 `kernel<<<blocks,threads,shared_bytes,current_stream>>>` 启动。使用 current stream 才能与 PyTorch 前后算子及 Graph capture 保持正确依赖。

**项目实现细节**

- Python loader：`nanovllm/kernels/state_aware_gdn_cuda.py`。
- Binding：`state_aware_gdn_binding.cpp`。
- Kernel/launcher：`state_aware_gdn_kernel.cu`。
- Extension name：`nanovllm_state_aware_gdn_cuda_ext`。

**连续追问链**

- 问：为什么不用默认 stream？答：PyTorch 当前工作可能在其他 stream，错误 stream 会产生 race 或隐式同步。
- 问：CUDA Graph 能捕获 Extension 吗？答：只要 launch 在 capture current stream、无 capture 禁止操作且地址稳定即可。
- 问：`<<< >>>` 四个参数是什么？答：grid、block、dynamic shared bytes、stream。

**证据**

- Python→C++→CUDA Extension test 与 Hybrid Graph capture/replay 均通过。

**边界与红线**

- JIT load 是当前工程接入方式，不声称已做发布级 wheel/多架构预编译。

**一句话记忆点**：Python 管 Tensor，Binding 管 ABI，Launcher 管参数/stream，Kernel 管 GPU 计算。

### Q59. Kernel 如何同时接入 Continuous Batching 和 CUDA Graph？

**30～60 秒口述回答**

Continuous Batching 需要每步变化的请求映射，CUDA Graph 需要稳定地址。ModelRunner 在 `run_hybrid_decode()` 或 `run_hybrid_graph_body()` 中，把当前或静态 `state_slot_ids` 和 State Pool 写入 Context 的 `gdn_state_slot_ids`、`gdn_recurrent_state_pool`、`gdn_conv_state_pool`。GDN Layer 选择 CUDA backend 后从 Context 取这些字段并调用 Extension。Eager 时 IDs 是本步 Tensor；Graph 时 IDs 来自固定 Workspace，但内容在 replay 前更新。Kernel 始终直接访问同一物理 Pool，因此两种执行模式共享数学和状态语义。

**项目实现细节**

- Eager入口：`ModelRunner.run_hybrid_decode()`。
- Graph body：`run_hybrid_graph_body()`，捕获/重放由 Graph 辅助函数管理。
- Output 参数允许复用 Graph 静态 output buffer。

**连续追问链**

- 问：backend 在 Prefill 也启用吗？答：State-Aware CUDA 专用于 L=1 Decode，Prefill 仍走 FLA。
- 问：Graph replay 后如何取 hidden？答：读取对应 Graph 的静态 output/hidden workspace，再继续 Graph 外 LM Head/Sampling。
- 问：如何证明没有改变状态顺序？答：比较 Eager/Graph 的 tokens、KV、Conv 和 Recurrent state，并做动态 Batch。

**证据**

- Full-model eager greedy 对齐、Graph/Eager 状态一致、bucket matrix 与 continuous batching 全通过。

**边界与红线**

- 当前仅 TP=1、RTX 5090 和特化 Shape 完整验证。

**一句话记忆点**：Context 把调度产生的 Slot 映射送进 Layer，current-stream Extension 让同一 Kernel 同时服务 Eager 与 Graph。

---

## 八、Benchmark 与 Profile（Q60～Q65）

### Q60. TTFT、TPOT、吞吐、P99 和 E2E 分别是什么？

**30～60 秒口述回答**

TTFT 是请求进入到首个输出 Token 的时间，主要受排队和 Prefill 影响；TPOT 是首 Token 之后相邻输出 Token 的平均时间，主要衡量 Decode；吞吐是单位时间全系统生成的 Token 数，多请求时大致是 Batch tokens 除以 step time；E2E 是完整请求总耗时；P99 描述最慢 1% 的尾延迟，反映抖动和 SLO。Prefix Cache 应重点看 TTFT，CUDA Graph/Decode Kernel 应重点看 TPOT、吞吐和 step P99，不能用错指标。

**项目实现细节**

- B16：`throughput≈16/step_seconds`。
- 首个 Prefill token 与后续 127 Decode steps 分开统计。
- 平均值/median 衡量中心趋势，P99 衡量尾部。

**连续追问链**

- 问：为什么吞吐提高而单请求延迟可能不降？答：更大 Batch 可提高总 tokens/s，但每个 step 可能更长。
- 问：TTFT 包括排队吗？答：在线口径应包括；微基准若不包括必须明确。
- 问：TPOT 和 ITL 一样吗？答：概念接近，但统计区间/聚合方法要明确。

**证据**

- 最终报告同时包含吞吐、TPOT、P99、Eager/Graph 与 Prefix TTFT。

**边界与红线**

- 指标必须说明 workload 和统计口径，不能只报百分比。

**一句话记忆点**：TTFT 看首 Token，TPOT 看后续 Token，吞吐看系统，P99 看最坏体验。

### Q61. 为什么需要 Warm-up、重复实验、CUDA Event 和独立进程？

**30～60 秒口述回答**

首次运行混有 Extension 编译、CUDA Context、权重 lazy initialization、Allocator 扩容和 Graph capture，不能进入稳态数据；所以先 warm-up。GPU 异步执行，普通 CPU 计时若不同步只量到 launch，Kernel 级测量用 CUDA Event 在同一 stream 记录时间；端到端可用同步后的 wall clock。重复运行取 median 并报告 P99 能降低偶然抖动。不同 backend/配置用独立进程可避免 CUDA allocator、Graph pool 和缓存残留互相污染。

**项目实现细节**

- 最终 Benchmark：128 output tokens、5 repeats。
- Profile：外部 warmup 8，Profiler schedule wait=1、warmup=1、active=8。
- Kernel 微基准用 CUDA Event；完整模型同时保留端到端时钟。

**连续追问链**

- 问：为什么 Event 不需要每个 Kernel 后 synchronize？答：Event 在 stream 中排序，结束后一次同步并计算 elapsed。
- 问：为什么取 median？答：比 mean 更不受偶发 OS/初始化抖动影响。
- 问：为什么同进程切 backend 有问题？答：缓存、JIT、allocator reserved 和 Graph pool 使起点不同。

**证据**

- 最终结果来自重复独立配置，并校验预期 24×8=192 次 GDN Kernel 调用。

**边界与红线**

- 不能把包含编译/捕获的首次运行混入稳态吞吐。

**一句话记忆点**：先热身、异步用 Event、配置用独立进程、重复取稳健统计。

### Q62. Conv 子路径 3.21× 为什么不能和完整模型收益混写？

**30～60 秒口述回答**

3.21×测的是原 Conv 状态路径约 355 μs降到自研融合路径约111 μs，范围只包含 Gather/Scatter、4-tap causal conv、SiLU 和回写。完整 Qwen3.5 Decode 还有 embedding、线性层、24 个 recurrent、8 个 Full Attention、MLP、norm、LM Head 和调度，因此 Amdahl 定律决定局部 3.21×只会转化为较小的端到端收益。最终完整 Eager 的公平口径是相对 FLA Eager提升26.86%～32.13%。

**项目实现细节**

- 子路径速度提升：`355/111≈3.21×`。
- 端到端收益受该路径占总时长比例约束。
- Graph 比较要保证双方都开或都关。

**连续追问链**

- 问：能用局部加速推算总加速吗？答：需知道优化部分占比，使用 Amdahl 公式。
- 问：为什么仍报告子路径？答：它解释优化机制，但必须与完整模型结果并列而非替代。
- 问：CUDA+Graph vs FLA Eager 可以报吗？答：可作为联合系统收益，但不能归因于单一 Kernel。

**证据**

- Conv 3.21×；完整 Eager +26.86%～32.13%；双方 Graph +1.99%～18.31%。

**边界与红线**

- 简历必须带“Conv 状态子路径”限定语。

**一句话记忆点**：微基准解释哪里快，完整模型才回答用户最终快多少。

### Q63. 最终 Eager/Graph 性能数字是如何得到和解释的？

**30～60 秒口述回答**

在 RTX 5090、Qwen3.5-9B、TP=1、输出128 tokens、5次重复下，我比较四组同 workload：FLA Eager、CUDA Eager、FLA Graph、CUDA Graph。B=1/2/4/8/16时，自研 CUDA Eager 相对 FLA Eager吞吐提升26.86%～32.13%；双方启用 Graph 后提升1.99%～18.31%。B16 CUDA Graph为1087.37 tok/s、TPOT14.71 ms，P99相对FLA Graph从17.446降到14.748 ms。Graph使两边都减少launch，所以后端差距缩小是合理的。

**项目实现细节**

- B1 throughput：54.72/71.45/78.48/80.04。
- B16 throughput：800.70/1048.69/919.05/1087.37。
- 顺序均为 FLA Eager/CUDA Eager/FLA Graph/CUDA Graph。

**连续追问链**

- 问：为什么B1双方Graph只差1.99%？答：小B的launch占比极高，Graph掩盖了很多原接口调用差异。
- 问：为什么B16差18.31%？答：状态搬运与Kernel工作随Batch放大，Graph外公共开销占比下降。
- 问：峰值allocated变化？答：Eager从25838.58降到25079.93 MiB，约减少758.65 MiB或2.94%。

**证据**

- `artifacts/cuda_graph/benchmark/final_conv/` 的最终四组结果。

**边界与红线**

- 不沿用历史文档中的旧性能区间；不跨不同开关比较后声称纯后端收益。

**一句话记忆点**：公平比较是Eager对Eager、Graph对Graph，B16最终是1087 tok/s和14.71 ms。

### Q64. Profile 显示了什么？为什么数学 Kernel 略慢而系统仍更快？

**30～60 秒口述回答**

B16 Eager Profile中，FLA fused recurrent本体是28.64 μs/层，自研 recurrent是32.27 μs/层，单看数学Kernel自研约慢12.7%。但FLA为了连续输入，每步还要做24层recurrent/conv Gather、Scatter和Conv D2D copy，8 steps合计19.669 ms，也就是约2.459 ms/step；自研路径这些调用为零。自研Conv本体也从2.95降到2.36 μs/层。最终profiled decode step从约16.987降到13.792 ms，所以系统更快来自接口搬运消除，而不是声称每个Kernel都更强。

**项目实现细节**

- FLA recurrent `index_select`：8.981 ms；`index_copy_`：7.743 ms。
- Conv `index_select/index_copy_/D2D`：1.852/0.788/0.306 ms。
- 自研：recurrent 6.196 ms/192 calls；conv 0.453 ms/192 calls。

**连续追问链**

- 问：为什么不继续优化recurrent数学？答：可以，但当前先解决更大的2.459 ms/step接口开销；后续才看occupancy/带宽等。
- 问：Profile总时长能代替Benchmark吗？答：不能；Profiler有instrumentation开销，官方性能用独立Benchmark。
- 问：如何确认没有漏记Kernel？答：24层×8 active steps=192 calls，与记录一致。

**证据**

- Profile次数和每类调用累计时间均与理论调用数一致；完整Benchmark同方向提升。

**边界与红线**

- 必须主动承认 recurrent数学Kernel略慢于FLA，不能挑数据隐藏。

**一句话记忆点**：自研赢在少搬2.459 ms/step的大状态，不是赢在每个数学Kernel都更快。

### Q65. 项目性能评估还有哪些局限，下一步会怎么做？

**30～60 秒口述回答**

当前结果在单张RTX 5090、Qwen3.5-9B、TP=1和特化Shape上闭环，说明方案有效，但还不能代表多GPU或所有模型。下一步我会先用Nsight Compute量化recurrent Kernel的DRAM吞吐、L2 hit、occupancy、register和stall reason，再决定是否做更多Shape dispatch、调整Split-K或流水化；系统侧会增加真实到达分布、不同Prompt/输出长度和并发下的P50/P99。最后才扩展TP>1与更多模型Shape，并重新验证Graph capture和状态一致性。

**项目实现细节**

- 当前特化：H=32、Dk=Dv=128；C=8192、K=4。
- 当前无TP>1、自研Prefill、跨GPU和图像Prefix Cache。
- 当前有CUDA Event/PyTorch Profiler，无Nsight Compute硬件计数器结论。

**连续追问链**

- 问：第一优先级是什么？答：先Profile确定recurrent慢于FLA的微架构原因，不盲加技巧。
- 问：为什么不是马上上cp.async？答：先证明有可隐藏的global→shared复用和瓶颈，否则只增加复杂度。
- 问：如何支持多Shape？答：保留通用fallback，按真实模型高频Shape做少量dispatch。

**证据**

- 已有功能/性能证据严格限定在RTX5090、TP1、Qwen3.5-9B和B1～16。

**边界与红线**

- 下一步是规划，不写成已完成；不能声称已有Nsight硬件指标。

**一句话记忆点**：现阶段证明“特化路径端到端有效”，下一步用硬件计数器决定，而不是靠优化术语堆叠。

---

## 九、相邻高频八股（Q66～Q75）

### Q66. FlashAttention、PagedAttention 和 Prefix Cache 分别解决什么问题？

**30～60 秒口述回答**

FlashAttention 是 Attention Kernel 算法，利用 tiling 和 online softmax 减少 QK、概率矩阵在 HBM 的物化与读写，解决单次 Attention 计算的 IO 瓶颈。PagedAttention 是推理时 KV Cache 的内存组织和寻址机制，用逻辑 Block 到物理 Block 的映射减少连续大块分配和碎片。Prefix Cache 是跨请求复用已经算过的前缀历史，命中后跳过部分 Prefill。三者分别位于 Kernel 计算、运行时内存和请求级复用三个层次，可以同时使用。

**项目实现细节**

- 本项目调用第三方 FlashAttention。
- 本项目适配 Paged KV metadata，并实现 Hybrid 联合 Prefix Cache。
- Prefix Cache 同时恢复 8 层 KV 和 24 层 GDN State。

**连续追问链**

- 问：PagedAttention 会减少 KV 总字节吗？答：主要减少碎片并支持非连续物理分配，不改变每 Token 的理论 KV 内容。
- 问：Prefix Cache 是 KV 压缩吗？答：不是，它复用相同内容；本项目没有实现 KV 压缩。
- 问：KV 压缩后 position 怎么办？答：必须保留原逻辑 position/映射，不能把剩余 KV 简单重编号；但这不是本项目完成项。

**证据**

- 项目中三层职责分别对应 Attention 调用、Block Table 和联合 Prefix Entry。

**边界与红线**

- 不能说自己实现了 FlashAttention，也不能把 Prefix Cache 写成 KV Cache 压缩。

**一句话记忆点**：FlashAttention 少搬中间矩阵，PagedAttention 管物理 KV，Prefix Cache 少算重复前缀。

### Q67. Online Softmax 如何保证数值稳定？

**30～60 秒口述回答**

普通稳定 Softmax 用 `exp(x_i-max(x)) / sum_j exp(x_j-max(x))` 防止指数溢出，但分块 Attention 无法一开始知道全局最大值。Online Softmax 为每个 query row 维护 running max `m` 和 running denominator `l`。处理新块时令 `m_new=max(m_old,block_max)`，把旧累积按 `exp(m_old-m_new)` 重标定，再加入新块的指数和；输出向量累积也做同样重标定。最后除以 `l`，结果与完整 Softmax 等价，但无需物化完整 score matrix。

**项目实现细节**

- 状态三元组可理解为 `(m, l, acc)`。
- 更新：`l_new=l_old*exp(m_old-m_new)+sum(exp(score_block-m_new))`。
- causal mask 在 block score 进入 online update 前应用。

**连续追问链**

- 问：为什么减最大值？答：使指数输入不大于0，避免 overflow。
- 问：为什么旧 acc 要缩放？答：running max 变化后，新旧指数使用了不同基准。
- 问：FlashAttention 为什么省显存？答：score/probability tile 在片上处理，不把完整 `S×S` 矩阵写回 HBM。

**证据**

- 这是解释第三方 Attention 的相邻知识，不作为项目自研结果。

**边界与红线**

- 不声称自己实现/优化了 FlashAttention。

**一句话记忆点**：Online Softmax 每块更新最大值，并把旧分母与旧输出缩放到同一个指数基准。

### Q68. Prefill 和 Decode 分别受什么瓶颈限制？常见加速手段有哪些？

**30～60 秒口述回答**

Prefill 同时处理很多 Token，大 GEMM 和长序列 Attention 并行度高，通常更偏计算吞吐与 Attention IO；常用 FlashAttention、算子融合、Chunked Prefill、量化和并行。Decode 每步每请求只有一个 Token，频繁读取权重、KV/GDN State，小 Kernel 与 CPU launch 占比高，通常更偏访存和调度；常用 Continuous Batching、PagedAttention、Prefix Cache、CUDA Graph、Kernel fusion、量化和投机解码。具体瓶颈不能只按阶段下结论，要用 Profile 判断。

**项目实现细节**

- Prefix Cache/Chunked Prefill主要优化 TTFT/调度。
- State-Aware CUDA 和 Decode Graph主要优化 TPOT/吞吐。
- Hybrid GDN 固定 State 与 Full Attention 线性 KV 的带宽结构不同。

**连续追问链**

- 问：Decode 为什么 GEMM 利用率低？答：M维接近 Batch，小时难以充分占满GPU。
- 问：Batch 越大越好吗？答：吞吐可能提高，但step latency、显存和尾延迟也上升。
- 问：如何判断 compute-bound？答：结合算术强度、roofline和Nsight Compute的计算/带宽利用率。

**证据**

- 项目中Prefix、Graph与Kernel在不同指标上分别测量，未混为一个优化。

**边界与红线**

- “Prefill compute-bound、Decode memory-bound”是常见趋势，不是所有Shape的定律。

**一句话记忆点**：Prefill追求大Token并行，Decode追求少搬数据、少发Kernel并扩大动态Batch。

### Q69. TP=4 时 QKV、O、FFN、Embedding、LM Head 如何切分和通信？

**30～60 秒口述回答**

经典 Megatron Tensor Parallel 中，QKV 采用 column parallel，按输出 head/hidden 维切分，每卡计算本地 heads，Attention 本体在本地完成；O projection 采用 row parallel，每卡产生部分和，末尾 All-Reduce。FFN 的 gate/up 是 column parallel，中间激活分片保留；down projection 是 row parallel，末尾 All-Reduce。Embedding/LM Head 可以按 vocabulary 切分：Embedding 由每卡对本地词表查找、非本地置零后 All-Reduce；LM Head 得到分片 logits，采样可做分布式 top-k/max 或 All-Gather。

**项目实现细节**

- 若 Full Attention `Nq=16、Nkv=4`，TP4可使每卡4个Q heads、1个KV head，需满足可整除。
- Column parallel通常不在输出立即通信；row parallel在求和处通信。
- GDN State 如何按head切分并保持Graph/Slot一致，需要额外设计。

**连续追问链**

- 问：四卡通常用什么通信？答：节点内多用NCCL，经NVLink/PCIe传输。
- 问：RMSNorm需要All-Reduce吗？答：若hidden完整则不需；若保持hidden分片，统计/布局取决于并行策略。
- 问：本项目完成TP了吗？答：没有，最终正确性和性能均为TP=1。

**证据**

- 此题属于相邻高频理论，不进入项目性能归因。

**边界与红线**

- 不声称项目已完成TP、通信优化或多卡验证。

**一句话记忆点**：QKV/gate-up按输出切，O/down按输入切并All-Reduce，词表可切Embedding和LM Head。

### Q70. 大模型推理显存由什么组成？单卡放不下怎么办？

**30～60 秒口述回答**

显存主要由模型权重、KV Cache、Hybrid recurrent/conv state、临时激活/Workspace、CUDA Context、Allocator Reserved、Graph private pool和通信buffer组成。权重约为参数量乘dtype bytes；KV按 `tokens×layers×2×kv_heads×head_dim×bytes`；本项目还要为每个活动请求预算49.5 MiB GDN Slot。单卡放不下时先判断哪部分主导：权重可量化或TP/PP，KV可降低并发/上下文、用更低精度或KV策略，冷数据可Offload；同时保留CPU fallback不等于可接受的在线性能。

**项目实现细节**

- Qwen3.5-9B BF16权重理论约18 GB，另有运行时开销。
- 本项目8个Attention层使理论KV低于32层全Attention模型。
- 16K单请求KV约512 MiB，外加49.5 MiB活动GDN State。

**连续追问链**

- 问：`nvidia-smi` 与PyTorch allocated为何不同？答：reserved缓存、context、非PyTorch分配和Graph pool不都计入allocated。
- 问：先量化还是先TP？答：取决于精度、吞吐、硬件和通信；量化省单卡容量，TP扩容量但引入通信。
- 问：减小Batch能让权重放下吗？答：不能改变权重大小，只减少KV/State/activation。

**证据**

- 项目实测Eager peak allocated减少758.65 MiB；Graph显式workspace与capture增量分开记录。

**边界与红线**

- 未实现权重量化、KV量化、TP或Offload。

**一句话记忆点**：先把显存拆成权重、随请求历史、临时工作区和运行时池，再对主项下手。

### Q71. FP16、BF16、INT8 和 FP8 有什么区别？

**30～60 秒口述回答**

FP16有5位指数、10位尾数，精度较高但动态范围较小；BF16有8位指数、7位尾数，动态范围接近FP32，更适合训练/递归数值但尾数更粗。INT8是定点量化，需要scale/zero-point或对称scale，容量小但量化误差和dequant开销需管理。FP8仍是浮点，常见E4M3偏精度、E5M2偏范围，也需要scale策略。是否更快取决于硬件Tensor Core、Kernel和是否被其他FP32状态/带宽主导。

**项目实现细节**

- 模型输入输出/Conv State主要BF16。
- Recurrent活动State与累积FP32，避免递归误差放大。
- Prefix recurrent Snapshot存BF16，容量48→24 MiB，恢复时转回FP32。

**连续追问链**

- 问：为什么BF16未必比FP32 Kernel快？答：若主要流量是FP32 State且存在转换/同样计算，输入减半不是主瓶颈。
- 问：量化分per-tensor/per-channel有什么区别？答：后者scale更细、精度常更好但metadata和实现更复杂。
- 问：能把recurrent State改INT8吗？答：需专门校准多步误差和Kernel，当前没有实现。

**证据**

- 项目dtype benchmark中BF16输入对以FP32 State为主的Kernel收益接近零，支持“瓶颈决定收益”的结论。

**边界与红线**

- 仅实现BF16输入/输出和Snapshot存储，不声称完成INT8/FP8量化。

**一句话记忆点**：格式只改变表示能力，真正的速度由主流量、硬件指令和量化开销共同决定。

### Q72. Offload 是什么？工程中如何做？

**30～60 秒口述回答**

Offload 是把GPU暂时不用的权重、KV或状态移到CPU内存甚至NVMe，在需要前预取回来，用容量换PCIe/NVLink传输与调度复杂度。工程上需要分层缓存、pinned host memory、异步 `cudaMemcpyAsync`、独立copy stream、event依赖和prefetch；只有传输与当前计算重叠，且工作集有冷热性，才可能实用。对本项目可考虑冷Prefix Snapshot或被抢占请求状态，但49.5 MiB Slot频繁往返可能直接增加Decode/恢复延迟。

**项目实现细节**

- Offload对象必须绑定Token边界和所有权，恢复KV与GDN State仍需一致。
- pinned memory提高DMA效率，但也是有限CPU资源。
- 需要测PCIe带宽、预取命中率、TTFT/P99和GPU节省量。

**连续追问链**

- 问：Unified Memory算Offload吗？答：可实现迁移，但page fault不可控，在线推理通常偏向显式管理。
- 问：什么最适合Offload？答：访问稀疏、可预测、恢复不在关键路径的冷对象。
- 问：项目实现了吗？答：没有，只理解其在Hybrid State场景的设计约束。

**证据**

- 当前抢占使用释放/重算或Prefix恢复，不包含CPU/NVMe状态Offload。

**边界与红线**

- 不能把CPU保存测试数据或Snapshot说成生产Offload。

**一句话记忆点**：Offload不是“挪到CPU”四个字，而是冷热识别、异步预取和计算通信重叠。

### Q73. Draft、EAGLE 和 MTP 投机解码有什么区别？

**30～60 秒口述回答**

投机解码的共同目标是用较便宜的路径一次提出多个候选，再由目标模型并行验证，接受连续正确前缀，从而减少目标模型串行Decode步数。传统Draft通常是独立小模型，自回归提出tokens；EAGLE系列更倾向在目标模型特征层预测未来特征/token，减少独立模型差异；MTP是在模型训练时加入多个未来Token预测头，直接产生多步候选。加速取决于候选成本、接受率、验证效率和回滚成本。

**项目实现细节**

- Hybrid模型被拒绝的候选不能只回滚Token/KV，还要恢复Conv/Recurrent State。
- 可在verify前做状态快照，或让候选状态写到临时分支，accept后commit。
- CUDA Graph需为Draft/Verify不同Shape设计独立Graph。

**连续追问链**

- 问：为什么候选越多不一定越快？答：接受率下降、验证和临时状态成本会上升。
- 问：Target输出分布会改变吗？答：正确的accept/reject算法保持目标模型分布。
- 问：项目实现了吗？答：没有；这是根据现有Hybrid状态管理可推导的扩展。

**证据**

- 当前项目没有Draft/Verify/Accept-Reject代码或性能数据。

**边界与红线**

- 简历只能写“了解MTP/EAGLE”，不能写完成投机解码或状态回滚。

**一句话记忆点**：Draft/EAGLE/MTP差在候选从哪里来，Hybrid难点在拒绝后不仅回滚KV，还要回滚两类GDN State。

### Q74. CUDA Kernel 如何被调用？如果让你手写并优化 GEMM，会怎么回答？

**30～60 秒口述回答**

CUDA Kernel由host用 `kernel<<<grid,block,shared_bytes,stream>>>(args...)` 发起；Grid包含Blocks，Block包含Threads，32个线程组成Warp。手写GEMM我会先做每线程一个C元素的正确baseline，再用CUDA Event与cuBLAS在相同dtype/shape/transposition上对比；随后按Global Memory合并访问、Shared Memory CTA tiling、寄存器thread tiling、循环展开和向量化逐步优化。更深的 `cp.async`、双缓冲和Tensor Core必须由目标架构、Shape和Profile支持，不能只罗列名词。

**项目实现细节**

- 本项目launcher显式传current stream，recurrent与conv使用不同grid/block映射。
- CUDA Event应在同一stream记录，warm-up后重复；GFLOPS=`2MNK/time`。
- cuBLAS对照要包含相同epilogue，否则比较不公平。

**连续追问链**

- 问：Shared Memory tiling为何快？答：一个tile从HBM加载一次，被Block内多次复用。
- 问：Bank Conflict是什么？答：同一Warp访问同bank不同地址会串行；padding/布局变换可缓解。
- 问：为何本项目没有照搬GEMM优化？答：GDN有状态原位更新和两次Dk归约，数据复用/依赖不同。

**证据**

- 本项目实际使用Warp Shuffle、Split-K与向量访存；GEMM部分是相邻基础。

**边界与红线**

- 不声称实现了cp.async、Tensor Core GEMM或达到某个cuBLAS百分比。

**一句话记忆点**：先有可验证baseline，再围绕数据复用和Profile逐层优化，绝不把GEMM术语硬套到GDN。

### Q75. Nsight Systems、Nsight Compute 和 SASS 分别解决什么问题？

**30～60 秒口述回答**

Nsight Systems看系统时间线：CPU/Python、CUDA API、Kernel、Memcpy、stream间空洞，适合回答“时间花在哪、是否launch-bound、有没有意外同步”。Nsight Compute深入单个Kernel，查看occupancy、register、DRAM/L2吞吐、warp stall和指令统计，适合回答“这个Kernel为什么慢”。SASS是GPU实际执行的机器指令，可由 `cuobjdump`/`nvdisasm` 查看，用来确认编译器是否生成向量load/store、shuffle、是否有spill等。顺序通常是Systems找热点、Compute判瓶颈、SASS验证指令。

**项目实现细节**

- 已使用PyTorch Profiler/CUDA Event确认调用次数与状态搬运热点。
- ptxas显示recurrent/conv编译无spill，并记录register/shared memory。
- 下一步才会用Nsight Compute获得硬件计数器。

**连续追问链**

- 问：SASS与PTX区别？答：PTX是虚拟ISA中间表示，SASS是针对具体SM架构生成的机器码。
- 问：occupancy越高越好吗？答：不一定，只要足以隐藏延迟；过度追求可能牺牲寄存器复用。
- 问：看到long scoreboard常意味着什么？答：常与等待global/local memory依赖有关，需结合memory指标确认。

**证据**

- 当前能够用Profiler证明2.459 ms/step搬运，但没有把未采集的Nsight counter写成结论。

**边界与红线**

- 面试时明确“理解工具与基础流程，尚未形成Nsight Compute硬件指标报告”。

**一句话记忆点**：Systems看整条时间线，Compute看一个Kernel的瓶颈，SASS看GPU最终执行了什么指令。

---

## 十、CANN 算子挑战赛占位区

### 当前可说事实

- 身份：2026年华为CANN算子挑战赛队长。
- 当前职责：组织任务拆分、学习计划、代码评审、测试口径和进度同步。
- 当前阶段：比赛仍在进行，技术方案、目标Shape、Baseline与性能结果尚未形成可对外陈述的最终证据。

### 结果产生后必须补齐

1. 题目与算子数学定义、输入输出Shape、dtype和目标芯片。
2. Baseline来源、正确性误差阈值、性能测量方法。
3. 自己负责的代码边界与团队协作方式。
4. Profile瓶颈、实际采用的优化、失败方案和选择依据。
5. 最终耗时/带宽/吞吐、相对Baseline收益、比赛排名或官方结果。

### 当前红线

- 不提前声称使用了某种tiling、double buffer、流水线或指令级优化。
- 不填写未经复现的性能百分比、排名或“达到某库多少”。
- 面试官追问时可以讲队长工作与当前学习过程，但主动说明比赛仍在进行。

---

## 十一、快速背诵路线

1. **第一轮（15题）**：Q1、Q6、Q7、Q10、Q11、Q17、Q20、Q22、Q34、Q35、Q41、Q42、Q49、Q57、Q63。
2. **第二轮（变量）**：Q9、Q12、Q27、Q30、Q37、Q44、Q50、Q51、Q56、Q58。
3. **第三轮（性能防守）**：Q48、Q60～Q65，确保每个数字都能说出模型、GPU、Batch、Baseline和范围。
4. **第四轮（压力面）**：让同伴只问每题的“连续追问链”，回答后再反向指出证据与红线。

最终统一回答结构：

> 先说问题和结论 → 再说数据流/关键变量 → 给正确性与性能证据 → 主动交代边界。

## 十二、图片补充八股复盘路线

`intern/pic` 两张面试截图中的题目已纳入《NanoHybrid-VLM项目进度记忆》第 15 节。原图共能确认18题，编号为1～17、20；不补造图片中缺失的18、19。

复盘顺序调整为：

1. 原 Part 1～8：请求链路、调度、Paged KV/State Pool、多模态、联合 Prefix、CUDA Graph、State-Aware CUDA、Benchmark/Profile。
2. Part 9：nano-vLLM/vLLM、SGLang Replay SSM、PD 分离及 Hybrid KV/GDN State 传输。
3. Part 10：Dense/MoE、TP/DP/EP、TP+EP Device Mesh 与通信原语。
4. Part 11：ViT/Decode 瓶颈、GEMM Profile、`torch.compile`、算子融合、host gap 与 HBM/DDR。
5. Part 12：vLLM 常见优化和 Continuous Batching 综合压力面。

新增题仍使用本文统一模板：30～60秒口述回答、原理/公式、项目联系、连续追问、证据、边界红线、一句话记忆点与闭卷巩固题。软件机制题（尤其 SGLang Replay SSM、vLLM 当前能力和 `torch.compile`）在正式展开时应优先核对届时官方文档/源码；不能把相邻八股写成本项目已完成功能。
