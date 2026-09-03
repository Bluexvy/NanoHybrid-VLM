# Hybrid CUDA Graph 极限压测

- Backend: `graph`
- Bucket mode: `base + candidate`
- 最大成功测试 Batch: `32`
- 首个失败测试 Batch: `48`
- 峰值吞吐 Batch: `32`
- 峰值 Decode 吞吐: `1363.47 tok/s`

## 探测结果

| B | 状态 | Decode tok/s | TPOT ms | Peak allocated MiB | Peak reserved MiB |
|---:|---|---:|---:|---:|---:|
| 16 | success | 918.10 | 17.427 | 25855.94 | 27596.00 |
| 24 | success | 1140.93 | 21.035 | 26697.62 | 29302.00 |
| 32 | success | 1363.47 | 23.469 | 27509.25 | 30928.00 |
| 48 | failed: OutOfMemoryError | - | - | - | - |

## 最大成功 Batch 持续压测

```json
{
  "status": "success",
  "backend": "graph",
  "batch_size": 32,
  "graph_buckets": [
    1,
    2,
    4,
    8,
    16,
    32
  ],
  "output_tokens": 512,
  "repeats": 3,
  "gpu_memory_utilization": 0.78,
  "max_model_len": 1024,
  "token_budget": 32768,
  "prompt_tokens": 22,
  "initialization_seconds": 4.832776383962482,
  "aggregate": {
    "batch_size": 32,
    "output_tokens": 512,
    "repeats": 3,
    "total_decode_tokens": 49056,
    "total_decode_steps": 1533,
    "decode_tokens_per_second": 1354.251322697629,
    "average_tpot_ms": 23.629292040311203,
    "step_latency_ms": {
      "mean": 23.629292040311203,
      "p50": 23.624549969099462,
      "p95": 23.804847383871675,
      "p99": 23.851141165941954,
      "max": 25.411178008653224
    },
    "mean_e2e_seconds": 12.170754280329371,
    "graph_replays": 1533,
    "eager_fallbacks": 0
  },
  "captured_buckets": [
    1,
    2,
    4,
    8,
    16,
    32
  ],
  "workspace_mib": 0.251953125,
  "capture_allocated_delta_mib": 57.1298828125,
  "fallback_reasons": {},
  "used_state_slots_after_run": 0,
  "used_kv_blocks_after_run": 0,
  "memory": {
    "current_allocated_mib": 24224.0009765625,
    "peak_allocated_mib": 27509.2490234375,
    "current_reserved_mib": 30928.0,
    "peak_reserved_mib": 30928.0,
    "free_mib": 447.625,
    "total_mib": 32110.9375
  },
  "return_code": 0,
  "log_path": "/workspace/nano-vllm/artifacts/cuda_graph/stress_limits/sustained_graph_b32.log",
  "json_path": "/workspace/nano-vllm/artifacts/cuda_graph/stress_limits/sustained_graph_b32.json"
}
```

## 解释边界

这里的最大 Batch 只表示给定候选集合、模型、显存比例、上下文长度和 Graph bucket 配置下的最大成功点，不代表硬件或引擎的数学绝对上限。
