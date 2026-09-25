# NanoHybrid-VLM TP=2 双卡测试套件

这套脚本只位于 `tests/5090-test-TP2`，不会修改推理引擎源码。它用于在租到两张 GPU 后，一次完成 TP=2 的环境预检、通信验证、正确性回归和性能测试，并保存可复查的原始数据。

## 一键运行

在 Linux 服务器上进入项目根目录：

```bash
cd /workspace/nano-vllm
chmod +x tests/5090-test-TP2/run_all.sh tests/5090-test-TP2/profile_nccl.sh

# 先用短流程排除环境和实现问题
SUITE_MODE=quick ./tests/5090-test-TP2/run_all.sh /workspace/models/Qwen3.5-9B

# quick 全部通过后跑正式数据
SUITE_MODE=full ./tests/5090-test-TP2/run_all.sh /workspace/models/Qwen3.5-9B
```

默认使用可见 GPU 0、1。需要选择其他卡时：

```bash
CUDA_VISIBLE_DEVICES=2,3 \
SUITE_MODE=quick \
./tests/5090-test-TP2/run_all.sh /path/to/Qwen3.5-9B
```

也可以保存固定配置：

```bash
cd tests/5090-test-TP2
cp config.example.sh config.sh
vim config.sh
./run_all.sh
```

`config.sh` 已加入 `.gitignore`，适合填写服务器上的模型路径和测试开关。

## 测试阶段

1. **环境预检**：检查两张可见 GPU、BF16、NCCL、P2P、固定端口 2333、残留共享内存，以及模型各维度能否被 TP=2 整除。
2. **NCCL 微基准**：验证双卡 AllReduce 的数值正确性，并测量 8 KiB、128 KiB 和 1 MiB 消息的延迟。
3. **TP1 Eager 基线**：运行自研 State-aware CUDA 后端，记录输出 Token、TTFT、TPOT、吞吐和显存。
4. **TP2 Eager**：使用相同输入测试 TP 分片，并与 TP1 做逐 Token 的贪心输出比较。
5. **TP2 CUDA Graph**：运行带 NCCL collective 的双卡 Graph，并与 TP2 Eager 做逐 Token 比较。
6. **动态 Batch**：先启动四条 Decode 请求，再插入两条不同长度的 Prefill 请求，覆盖 Continuous Batching、Chunked Prefill 和 GDN 状态隔离。
7. **多模态**：生成一张本地图像，验证 TP=2 下 Vision Encoder、mRoPE 和 Decode 数据链路。
8. **FLA 冒烟测试**：帮助区分问题位于自研 Kernel，还是位于通用 TP 分片和模型路径。
9. **联合 Prefix Cache**：Cold 请求在 1024 Token 处提交 KV/GDN 检查点；Hot 请求只计算 141 Token 后缀，并检查输出一致性、命中次数和状态恢复。
10. **结果汇总**：生成 `summary.md` 和 `summary.json`，汇总正确性、TP 加速比、Graph 加速比、NCCL 延迟和 Prefix 命中数据。

## Quick 与 Full

| 模式 | 测试组合 | 输出长度 | 重复次数 |
|---|---|---:|---:|
| `quick` | `128×B1/B4/B16`、`2048×B1` | 32 | 1 |
| `full` | `128×B1/B2/B4/B8/B16`、`2048×B1/B4/B8` | 128 | 3 |

2048 Token Prompt 配合每轮 512 Token Budget，用来覆盖 Chunked Prefill。B=1 主要观察小 Batch 的通信开销；B=8/16 用于观察吞吐扩展。

## 常用开关

```bash
RUN_TP1_BASELINE=0       # 单张卡无法容纳 TP1 时跳过
RUN_GRAPH=0              # 暂时只验证 Eager
RUN_PREFIX=0             # 暂时跳过 Prefix Cache
RUN_DYNAMIC=0            # 跳过动态请求测试
RUN_VISION=0             # 跳过单图测试
RUN_NSYS=1               # 已安装 Nsight Systems 时采集通信时间线
GPU_MEMORY_UTILIZATION=0.78
```

例如只定位 TP2 Eager：

```bash
SUITE_MODE=quick \
RUN_TP1_BASELINE=0 \
RUN_GRAPH=0 \
RUN_PREFIX=0 \
RUN_VISION=0 \
./tests/5090-test-TP2/run_all.sh /path/to/model
```

## 输出目录

每次运行会创建独立目录：

```text
tests/5090-test-TP2/results/YYYYMMDD_HHMMSS/
├── preflight.json
├── nccl_smoke.json
├── tp1_eager_cuda.json
├── tp2_eager_cuda.json
├── tp2_graph_cuda.json
├── tp2_fla_smoke.json
├── prefix_tp2.json
├── compare_tp1_tp2.json
├── compare_eager_graph.json
├── status.tsv
├── summary.json
└── summary.md
```

每个阶段还有独立的 `.log`。失败后先看 `status.tsv`，再打开对应阶段的日志。预检失败时，脚本会在加载模型前停止，避免浪费租卡时间。

## 如何解释结果

- TP1 与 TP2 输出逐 Token 一致，说明权重、Attention/GDN Head、MLP 和 LM Head 分片至少通过了贪心回归。
- TP2 Eager 与 Graph 输出一致，说明固定地址 Workspace、NCCL collective 顺序、Block Table 和 State Slot 路由保持一致。
- Prefix Hot 只执行 141 个 Prefill Token，说明两个 Rank 使用同一个逻辑 PrefixKey，并分别恢复了本 Rank 的 GDN 状态分片。
- B=1 可能因大量小 AllReduce 的启动延迟而变慢；B=8/16 更容易摊薄通信开销。
- 两张 4090 没有 NVLink，应结合 `nvidia_smi_topo.txt` 和 `nccl_smoke.json` 判断 PCIe/P2P 通信成本。

性能数字只能在 `summary.md` 所记录的 GPU、模型、Batch、Prompt、输出长度和后端范围内使用。如果单卡显存无法运行 TP1，可以报告 TP2 的功能与容量结果，但不能声称未经测量的 TP 加速比。
