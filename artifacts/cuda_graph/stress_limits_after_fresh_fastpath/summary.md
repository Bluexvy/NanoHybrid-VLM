# Hybrid CUDA Graph 极限压测

- Backend: `graph`
- Bucket mode: `base + candidate`
- 最大成功测试 Batch: `40`
- 首个失败测试 Batch: `48`
- 峰值吞吐 Batch: `40`
- 峰值 Decode 吞吐: `1550.20 tok/s`

## 探测结果

| B | 状态 | Decode tok/s | TPOT ms | Peak allocated MiB | Peak reserved MiB |
|---:|---|---:|---:|---:|---:|
| 32 | success | 1364.79 | 23.447 | 25924.50 | 29352.00 |
| 40 | success | 1550.20 | 25.803 | 26348.62 | 30600.00 |
| 48 | failed: OutOfMemoryError | - | - | - | - |

## 解释边界

这里的最大 Batch 只表示给定候选集合、模型、显存比例、上下文长度和 Graph bucket 配置下的最大成功点，不代表硬件或引擎的数学绝对上限。
