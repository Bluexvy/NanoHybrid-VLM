# Qwen3.5 Native MTP 开发与验证

这份实现使用 checkpoint 中训练好的 Qwen3.5 MTP Head，不是 EAGLE 或随机初始化的草稿模型。
功能开发已覆盖多请求、TP、图文输入、文本 Prefix Cache、随机采样，以及草稿/验证 CUDA Graph。
完整 9B 权重的端到端 Token 对齐、Linux/NCCL Graph 联调和性能结果仍需在匹配环境上执行；不要把局部测试当成这些结果。

## 支持范围

| 功能 | 实现方式 |
| --- | --- |
| Dense Qwen3.5 | 共享目标模型 Embedding 和 LM Head，严格检查 MTP 权重完整性 |
| 多草稿 | 重复使用一个训练好的 MTP 层，配置 1～15 个候选 Token |
| 多请求 / Continuous Batching | 请求独立的草稿 KV、概率、Hidden State；目标模型一次 Packed 验证 |
| TP | Q/K/V 与 gate/up 列切分；O/down 行切分并 AllReduce；LM Head Gather 到 Rank 0 |
| 随机采样 | 同温度的 p/q 分布；min(1,p/q) 接受；拒绝从归一化 (p-q)+ 采样 |
| 图文输入 | 移位后的图片位置使用 Visual Embedding；保留 T/H/W 三轴位置与 Decode delta |
| 文本 Prefix Cache | 联合保存目标 KV、GDN 和 MTP 前缀；MTP 字节数纳入容量预算与 LRU |
| 草稿 Graph | 单 Token 的固定容量 KV Buffer，加动态有效长度 Mask；多 Token 教师推进用 Eager |
| 验证 Graph | 按 (Batch, 验证长度, 上下文容量) 懒捕获；固定地址承载动态 metadata 和 Slot IDs |
| 回退 | 不同验证长度、非标准 Batch、Graph 数量/显存预算不足时走 Eager |
| GDN 回滚 | 原状态池作为只读 checkpoint，重放接受前缀的紧凑输入，不重跑整模型 |
| 融合 CUDA | Conv 连续窗口更新/卷积/SiLU；Recurrent 连续状态更新、每 Token 输出与最终状态 |

TP 要求 Q/KV Head 和 intermediate_size 能被 TP 数整除；没有额外实现 KV Head 复制。
当前联合 Prefix Cache 继续沿用原框架的“纯文本前缀”策略，图片请求不会只凭 placeholder Token Hash 复用。
这里借鉴 ReplaySSM 的输入重放思路，没有宣称实现其完整环形缓冲、延迟 Flush 或 output-only 验证算法。

## 一轮执行的数据流

1. Scheduler 按 pending Token + 配置草稿数预留计算预算；Engine 再按输出余量、上下文和 KV 空闲容量确定每条请求真正的草稿数。
2. Drafter 用上轮确认的目标 Hidden State 和下一 Token Embedding 起草。继续起草时使用临时的 MTP KV/feature；不覆盖已提交缓存。
3. 把每条请求的 [pending, draft0, draft1, ...] 拼进 Packed Prefill，分别构造 cu_seqlens、block_table、slot_mapping 和三轴 positions。
4. 所有 Rank 执行目标模型，目标 GDN 从状态池副本/视图读取，产生新的临时状态。验证过程不修改原 checkpoint。
5. 每个 Rank 计算本地词表 Logits，Gather 到 Rank 0。只有 Rank 0 执行接受/拒绝与随机采样。
6. Rank 0 广播每条请求的输出长度、Token IDs 和接受数；所有 Rank 使用相同接受边界。
7. 全接受时提交验证所得状态。部分接受时仅用接受前缀的 conv_input、K、V、g、beta 重放 GDN。
8. 利用首次因果验证的有效 Hidden State 前缀，教师推进 MTP KV；调度器再追加真实输出、释放多余 KV Block、记录确认前缀的 Hash。

### 接受边界的具体例子

验证前已计算 100 个 Token，当前还有 pending y1；草稿为 y2、y3、y4。
目标模型一次处理 [y1,y2,y3,y4]，产生预测 [p2,p3,p4,p5]。

- 全接受：输出 [y2,y3,y4,p5]，已计算边界为 104；p5 是下一轮 pending，还未被目标模型计算。
- y2 接受、y3 拒绝：输出 [y2,p3]，只提交处理 [y1,y2] 后的状态，已计算边界为 102。
- y3、y4 对应的 KV 写入在有效长度之外；尾部 Block 按确认边界释放，不会给拒绝 Token 建立 Prefix Hash。
- GDN 不是把“大矩阵倒着减回来”，而是从原 checkpoint 重放 [y1,y2] 的状态更新。

## 为什么 MTP 前缀快照少保存一个配对位置

MTP 的配对是 h_t + embedding(x_{t+1})。

目标前缀长度 L 的最后一个配对已经使用 x_L，它属于下一段后缀。
不同请求共享前缀时，x_L 可能不同。因此保存：

- MTP KV 前 L-1 个配对位置；
- 目标模型最后一个前缀 Hidden State h_(L-1)。

命中后，先用当前请求自己的 x_L 完成这个配对，再处理 suffix。
快照始终只读，恢复到每个请求独立的缓存；淘汰时各 Rank 同步释放本地 MTP 快照。

## CUDA Kernel 的分工

源码位于 nanovllm/kernels/cuda/gdn_replay_kernel.cu。

### Recurrent

- 一个 Block 有 4 个 Warp，总计 128 个线程。
- Block 负责一个请求/Head 的 16 个 V 列；一个 Warp 负责 4 个完整 V 列。
- Lane 负责 lane + 32*r 的 4 个 K 行，所以一个 Warp 覆盖完整 K 维度。
- 对齐时沿 V 方向 float4 读取；每 Lane 保留 4×4=16 个 FP32 状态值。
- 跨多个 Token 保留寄存器状态；K/Q 归一化、remembered 和输出都在 Warp 内归约。
- Replay 只更新状态；Verification 额外读取 Q，并输出每个 Token 的结果。
- checkpoint 只读一次，最终状态只写一次；保留已有 [K,V] 物理布局。

### Conv

- 一个线程负责一个请求的一个通道。
- Verification 把四个历史值和四个权重放在寄存器，连续处理短前缀，融合窗口更新、FP32 乘加和 SiLU。
- Replay 不需要重复卷积，只恢复原始输入窗口；BF16/FP16、4-tap、单 Token 对齐路径用 64-bit 移位读写。
- 输入与 checkpoint 按实际 element stride 寻址，输出是独立连续 Tensor。

CUDA 专用范围：1<=Dk<=128，1<=短前缀长度<=16，FP32/FP16/BF16 输入，FP32 Recurrent State；验证 Conv 专用窗口为 4。
auto 对不支持的形状使用 Reference，但编译、加载和执行失败会报错，不会被悄悄隐藏。

## Graph 的边界

草稿单步图：

- 固定地址 Embedding、目标 Feature、三轴 Position、K/V Buffer 和有效历史长度。
- K/V Buffer 按上下文容量 bucket 分配，动态 Mask 屏蔽无效尾部。
- 只修改 Workspace 中的私有 KV，返回有效前缀的独立缓存，避免下一次重放覆盖上一请求。

目标验证图：

- 固定地址 input_ids、positions、cu_seqlens、block_table、slot_mapping、state_slot_ids。
- GDN checkpoint Gather、目标 Forward、临时更新状态和紧凑输入采集在图内。
- LM Head 的词表 Gather、接受判定、回滚与提交在图外。
- 不给虚拟请求填充 live Slot；不同验证长度直接 Eager 回退。

两类 Graph 共享有上限的显存预算和条目数。各 Rank 通过 MIN AllReduce 共同决定是否捕获，保持通信顺序一致。
Capture 前清理 warmup 的临时状态引用；运行时数据复制和 Graph Replay 使用当前 Stream。
TP Decode 只传 pending Token、逻辑偏移和 mRoPE delta，不重复发送整段 Token 历史；Worker 通过逻辑偏移构造验证后缀。
TP 的 CPU 命令发布还加入了消费确认，避免快速缓存释放命令覆盖 Worker 尚未读走的上一条命令。

## 配置

MTP 默认关闭。示例：

```python
llm = LLM(
    model_path,
    tensor_parallel_size=2,
    max_num_seqs=16,
    max_num_batched_tokens=1024,
    max_model_len=4096,
    enforce_eager=False,
    hybrid_cuda_graph_batch_sizes=(1, 2, 4, 8, 16),
    mtp_num_speculative_tokens=3,
    mtp_gdn_rollback="replay_inputs",
    mtp_gdn_replay_backend="auto",
    mtp_gdn_verification_backend="auto",
    hybrid_prefix_cache_mode="opportunistic",
)
```

- mtp_cuda_graph_max_entries：草稿与验证 Graph 合计最多保留多少张，默认 8。
- mtp_cuda_graph_memory_fraction：从模型显存预算中预留给 MTP Graph 的比例，默认 0.1。
- reference 后端可用于逐项对齐；cuda 强制专用 Kernel；auto 按形状选择。
- 一个 checkpoint 必须含完整的训练后 mtp.* 参数，否则在模型分配前报错。
- C++/CUDA Extension 在模型加载后、warmup/capture 前编译；要求 CUDA 版 PyTorch、匹配的 Toolkit 与 C++ 编译器。

## 本地验证记录：2026-10-15

| 验证 | 结果 | 能证明什么 |
| --- | --- | --- |
| CPU unittest 全套 | 51 项，48 通过，3 GPU 项在 CPU 环境跳过 | 接受/拒绝、采样分布、预算、Hash、状态、前缀、位置和控制命令 |
| CPU Gloo TP=2/4 | 包含在 CPU 全套中，通过 | 真实多进程的 Head 切分/GQA/单步 KV，及 Gather、广播和本地提交 |
| PyTorch CUDA 小模型集成 | 3 项通过，RTX 3060 | 验证 Graph 动态路由、接受/拒绝提交；草稿 Graph 动态历史长度和三种 dtype |
| NVRTC Replay + CUDA Graph | 8 个设备用例通过 | 原重放 Kernel 的精度、stride、16 Token 边界、动态内容与 checkpoint 不变性 |
| NVRTC 融合 Verification | 12 个设备用例通过 | Conv/Recurrent 输出和状态，普通/Graph 执行，三种 dtype 与非连续输入 |

CPU 环境：F:/anaconda3/python.exe。
PyTorch CUDA 环境：F:/anaconda3/envs/rag/python.exe，Python 3.11 / PyTorch 2.11 + CUDA 13.0。
NVRTC 使用本机 CUDA Toolkit 12.8。没有修改或安装这些环境。

**尚未执行的验证：**

- PyTorch C++ Binding 的正式编译与完整调用：本地 Toolkit 12.8 与 CUDA PyTorch 13.0 不匹配，且当前没有可用 cl 编译器。
- 带真实训练 MTP 权重的 Qwen3.5-9B 端到端 Greedy Token 对齐。
- A800 TP=2 的 NCCL CUDA Graph 联调、多模态真实图片对齐，以及接受率/吞吐/延迟测量。

所以目前能声明“功能开发与上述局部验证完成”，不能声明“真实 9B 全部对齐或获得某个加速比”。

## 复现验证

```text
python -m unittest discover -s tests/mtp -p "test_*.py"
python tests/mtp/test_mtp_gpu.py
python tests/mtp/test_gdn_replay_cuda.py --nvrtc --graph
python tests/mtp/test_gdn_verification_cuda.py
```

GPU unittest 需要 CUDA 版 PyTorch；NVRTC 用例只需要 CPU PyTorch + 已安装 Toolkit/Driver，绕过 C++ Binding。

完整模型验证矩阵（在匹配的 Linux/CUDA 与本地权重环境执行）：

```text
python tests/mtp/validate_mtp_model.py --model /path/to/qwen3.5-with-mtp --tp 2 --batch 4 --drafts 3 --require-graph
```

该入口依次运行 Baseline、MTP Eager、MTP Graph、MTP Prefix Hit，比较 Greedy Token。
可追加 --image /path/to/image.png 检查混合图文输入。每个配置使用独立进程，不自动下载权重。
报告默认写到 artifacts/mtp/model_validation.json；未运行时不会生成“通过”报告。

快速手动入口：

```text
python tests/mtp/example_mtp.py --model /path/to/qwen3.5-with-mtp --tp 2 --batch 4 --mtp-tokens 3 --graph
```

## 性能上的实际限制

功能正确不等于一定加速。草稿接受率、草稿 Head 成本、词表 Gather、状态搬运、连续草稿 KV 的复制和 Graph bucket 命中率都会影响收益。
当前目标验证仍使用功能性的批量状态读入与提交；没有宣称去掉全部 GDN Gather/Scatter。
需要真实 Profile 后再决定是否值得做直接从 State Pool 验证、Paged Draft KV 或完整 ReplaySSM 的延迟物化。
