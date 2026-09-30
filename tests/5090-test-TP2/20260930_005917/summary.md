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
| 8 KiB | 67.99 us | 4.352 ms |
| 128 KiB | 90.97 us | 5.822 ms |
| 1024 KiB | 320.82 us | 20.532 ms |

## 正确性

- TP1 Eager vs TP2 Eager：**通过**
- TP2 Eager vs TP2 CUDA Graph：**未运行**
- TP2 联合 Prefix Cache：**未通过或未运行**

## 性能

| Case | TP1 Eager tok/s | TP2 Eager tok/s | TP 加速比 | TP2 Graph tok/s | Graph 加速比 | TP1 TTFT ms | TP2 TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p128_b1 | 15.22 | 12.42 | 0.816x | - | - | 127.04 | 142.72 |
| p128_b16 | 205.24 | 169.08 | 0.824x | - | - | 10548.28 | 10404.34 |
| p128_b4 | 60.55 | 49.73 | 0.821x | - | - | 144.51 | 167.97 |
| p2048_b1 | 15.05 | 12.32 | 0.818x | - | - | 515.82 | 632.14 |

## 阶段状态

```text
preflight	PASS	17s
nccl_smoke	PASS	6s
tp1_eager_cuda	PASS	286s
tp2_eager_cuda	PASS	148s
compare_tp1_tp2	PASS	0s
tp2_fla_smoke	PASS	34s
```
