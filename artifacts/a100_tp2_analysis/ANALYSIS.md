# A100 PCIe 双卡 TP=2 测试分析

数据来源：`final_a100_tp2_20260930_020332`，环境为 2×A100 PCIe 40GB、PyTorch 2.8.0+cu128、NCCL 2.27.3。两卡拓扑为 `PHB`，P2P 可用，但没有 NVLink。

## 1. 总结

- TP=2 功能链路确实跑通：模型初始化、TP2 Eager、TP2 CUDA Graph、动态批处理、多模态、FLA smoke、联合 Prefix Cache 均完成。
- 五组独立 Kernel 正确性测试全部通过，包括 Conv/GDN reference、Triton 和 CUDA Extension。
- 单卡 A100 上，自研 State-aware CUDA 相比 FLA：
  - B=1：Decode step 77.4223 ms → 64.2153 ms，延迟降低 17.06%，吞吐提高 20.51%。
  - B=16：Decode step 85.4791 ms → 74.0395 ms，延迟降低 13.38%，吞吐提高 15.45%。
- PCIe 双卡下，TP=2 Eager 并没有加速：吞吐约为 TP=1 Eager 的 0.78～0.82 倍。这说明该模型和当前 Eager 实现下，AllReduce/调度开销大于单卡计算节省。
- TP=2 CUDA Graph 相对 TP=2 Eager 提升 2.55～5.53 倍，说明 Decode 中大量细粒度 Kernel launch、Python 调度和 collective launch 开销非常明显。
- 联合 Prefix Cache 成功复用 1024-token 前缀：Cold Prefill 1165 tokens，Hot Prefill 141 tokens，恢复一次 GDN Snapshot；Rank 0 Snapshot 为 24.75 MiB。
- 128-token 严格逐 token 比较没有通过，但 32-token Quick 测试通过，三次 Full 重复各自完全稳定，差异只在几十步自回归后出现。当前更符合低精度/批形状/归约顺序造成的确定性数值漂移，而不是随机 race；仍不应宣称 128-token bitwise equivalence。

## 2. Full Suite 性能

| Case | TP1 Eager tok/s | TP2 Eager tok/s | TP2/TP1 | TP2 Graph tok/s | Graph/Eager |
|---|---:|---:|---:|---:|---:|
| p128_b1 | 15.90 | 12.79 | 0.804× | 69.62 | 5.44× |
| p128_b2 | 30.86 | 25.16 | 0.815× | 139.23 | 5.53× |
| p128_b4 | 64.88 | 51.01 | 0.786× | 259.97 | 5.10× |
| p128_b8 | 125.36 | 98.64 | 0.787× | 447.55 | 4.54× |
| p128_b16 | 242.27 | 190.24 | 0.785× | 669.72 | 3.52× |
| p2048_b1 | 16.39 | 12.78 | 0.780× | 69.87 | 5.47× |
| p2048_b4 | 56.29 | 45.76 | 0.813× | 184.58 | 4.03× |
| p2048_b8 | 103.45 | 81.96 | 0.792× | 208.67 | 2.55× |

注意：没有 TP=1 Graph 基线，因此不能把 `TP2 Graph / TP1 Eager` 的差值解释为纯 TP 加速；这里只能分别评价 TP Eager 扩展性和 TP2 Graph 收益。

Full Graph 统计包含 3090 次 Graph replay 和 188 次 eager fallback；fallback 原因都是 batch size 不在捕获 bucket 中。B=16 的 TPOT 从 TP2 Eager 约 84.5 ms 降至 Graph 约 25.2 ms。

## 3. PCIe/NCCL

NCCL 微基准：

| Payload | AllReduce 平均延迟 | 64 次估算 |
|---:|---:|---:|
| 8 KiB | 73.90 us | 4.73 ms |
| 128 KiB | 94.19 us | 6.03 ms |
| 1 MiB | 319.58 us | 20.45 ms |

这与 TP2 Eager 的 B=1 TPOT 比 TP1 多约 15 ms 相符：每层/子层的小 AllReduce 在 PCIe 上难以被计算收益覆盖。

Nsight Systems 的全进程统计中，NCCL AllReduce kernel 占 GPU kernel duration 的主要部分；但 NCCL kernel duration 会包含等待、同步和跨 Rank 不平衡，而且当前报告包含初始化和未过滤区间，因此不能把 88.9% 直接称为“纯数据传输占比”。更可靠的证据是 NCCL 微基准、端到端 TPOT/吞吐，以及时间线三者共同指向通信瓶颈。

## 4. 自研 State-aware CUDA 对比 FLA

Clean Benchmark：

| Batch | FLA step | CUDA step | 延迟降低 | FLA tok/s | CUDA tok/s | 吞吐提升 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77.4223 ms | 64.2153 ms | 17.06% | 12.92 | 15.57 | 20.51% |
| 16 | 85.4791 ms | 74.0395 ms | 13.38% | 187.18 | 216.10 | 15.45% |

CUDA Event 也给出一致趋势：

- B=1：FLA Decode 84.299 ms，State-aware CUDA 71.160 ms，降低约 15.6%。
- B=16：FLA Decode 87.447 ms，State-aware CUDA 76.990 ms，降低约 12.0%。
- FLA 路径显式出现一次 Gather 和一次 Scatter；B=1 两者合计约 2.88 ms/step，B=16 合计约 5.46 ms/step。
- State-aware 路径没有整池 Gather/Scatter，24 个 GDN 层直接根据 `state_slot_ids` 访问状态池。

CUDA Event 的语义区间存在嵌套，例如 `gdn_layer_forward` 包含 Conv/Recurrent，因此各项不能相加。最终性能结论应以无插桩 Clean Benchmark 为准，Event 主要用于定位开销来源。

## 5. 显存与状态分片

- TP1 Rank 0 模型参数：17947.80 MiB。
- TP2 Rank 0 模型参数：9409.05 MiB，减少 47.58%，接近理论减半。
- GDN State：792 MiB → 396 MiB，恰好减半。
- KV block：8 MiB → 4 MiB，单 Rank 分片减半。
- TP2 的总 KV Cache MiB 更高，是因为引擎按 `gpu_memory_utilization` 用释放出的显存扩充了 block 数量，不能理解为单 token KV 变大。

## 6. 正确性口径

可以声称：

- TP=2 模型能够稳定完成文本、多模态、动态批处理、Prefix Cache 和 CUDA Graph 推理。
- Kernel reference/CUDA/Triton 测试全部通过。
- 32-token 贪心回归通过；Full 重复运行确定稳定。

不要声称：

- TP1/TP2 在 128 个生成 token 上 bitwise 完全一致。
- TP2 Eager 在 PCIe A100 上获得吞吐加速。
- Nsight Systems 中 NCCL kernel 的 88.9% 等于纯 PCIe 传输占比。

## 7. 面试表达

“我在两张 A100 PCIe 40GB 上完整验证了 TP=2。模型权重和 GDN 状态在单 Rank 上分别由约 17.95 GiB、792 MiB 降到 9.41 GiB、396 MiB，说明分片生效。Eager 下 TP2 吞吐只有 TP1 的约 0.78～0.82 倍，NCCL 微基准和 Nsight Systems 都表明 PCIe 上频繁的小 AllReduce 是主要限制；这说明 TP 的首要价值在这里是容量扩展，而不是低 Batch 延迟加速。启用 Hybrid CUDA Graph 后，TP2 Decode 吞吐提升约 2.55～5.53 倍，B=16 达到约 670 tok/s。另一个独立结论是，自研 State-aware CUDA 在 A100 上相对 FLA 将 B=1/B=16 Decode step 分别降低约 17.1%/13.4%，证明消除状态 Gather/Scatter 在不同 GPU 上仍然有效。”

