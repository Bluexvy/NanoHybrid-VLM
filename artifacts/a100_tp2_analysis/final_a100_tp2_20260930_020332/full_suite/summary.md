# NanoHybrid-VLM TP=2 测试报告

## 环境

- PyTorch：`2.8.0+cu128`
- CUDA：`12.8`
- NCCL：`[2, 27, 3]`
- 可见 GPU：`2`
- GPU 0：NVIDIA A100-PCIE-40GB，空闲 39.08 GiB / 总计 39.49 GiB
- GPU 1：NVIDIA A100-PCIE-40GB，空闲 39.08 GiB / 总计 39.49 GiB
- P2P：`{'0->1': True, '1->0': True}`
- 模型维度：`{'num_attention_heads': {'value': 16, 'divisible': True}, 'num_key_value_heads': {'value': 4, 'divisible': True}, 'intermediate_size': {'value': 12288, 'divisible': True}, 'vocab_size': {'value': 248320, 'divisible': True}, 'linear_num_key_heads': {'value': 16, 'divisible': True}, 'linear_num_value_heads': {'value': 32, 'divisible': True}}`

## NCCL AllReduce

| Payload | 单次平均延迟 | 64 次估算 |
|---:|---:|---:|
| 8 KiB | 73.90 us | 4.730 ms |
| 128 KiB | 94.19 us | 6.028 ms |
| 1024 KiB | 319.58 us | 20.453 ms |

## 正确性

- TP1 Eager vs TP2 Eager：**失败**
- TP2 Eager vs TP2 CUDA Graph：**失败**
- TP2 联合 Prefix Cache：**通过**

## 性能

| Case | TP1 Eager tok/s | TP2 Eager tok/s | TP 加速比 | TP2 Graph tok/s | Graph 加速比 | TP1 TTFT ms | TP2 TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p128_b1 | 15.90 | 12.79 | 0.804x | 69.62 | 5.445x | 115.42 | 134.11 |
| p128_b16 | 242.27 | 190.24 | 0.785x | 669.72 | 3.520x | 481.85 | 564.18 |
| p128_b2 | 30.86 | 25.16 | 0.815x | 139.23 | 5.533x | 135.20 | 160.07 |
| p128_b4 | 64.88 | 51.01 | 0.786x | 259.97 | 5.096x | 141.86 | 151.59 |
| p128_b8 | 125.36 | 98.64 | 0.787x | 447.55 | 4.537x | 260.83 | 306.12 |
| p2048_b1 | 16.39 | 12.78 | 0.780x | 69.87 | 5.465x | 475.94 | 550.37 |
| p2048_b4 | 56.29 | 45.76 | 0.813x | 184.58 | 4.034x | 1765.19 | 1994.80 |
| p2048_b8 | 103.45 | 81.96 | 0.792x | 208.67 | 2.546x | 3235.60 | 3818.14 |

## TP2 Prefix Cache

- PrefixKey 边界：1024 tokens
- Cold Prefill：1165 tokens
- Hot Prefill：141 tokens
- 跳过计算：1024 tokens
- Rank 0 GDN Snapshot：24.75 MiB
- 命中数：1
- GDN 恢复数：1

## 阶段状态

```text
preflight	PASS	11s
nccl_smoke	PASS	7s
tp1_eager_cuda	PASS	261s
tp2_eager_cuda	PASS	326s
compare_tp1_tp2	FAIL(1)	0s
tp2_graph_cuda	PASS	135s
compare_eager_graph	FAIL(1)	0s
tp2_fla_smoke	PASS	32s
prefix_tp2	PASS	35s
```
