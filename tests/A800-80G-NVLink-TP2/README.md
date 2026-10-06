# 2×A800 80GB NVLink 超负载测试套件

这个目录用于 Qwen3.5-9B / NanoHybrid-VLM 的双卡 A800 实测。它不是把旧的
RTX 5090 小 Batch 测试换个 GPU 名称，而是针对 80GB 显存和 NVLink 重新设计：

- 主吞吐矩阵：`P=512, B=16/32/64/128, O=256`；
- TP 扩展：同一矩阵比较 TP1 与 TP2；
- GDN 后端：比较 FLA Eager、State-Aware CUDA Eager 和 CUDA Graph；
- 长上下文：`16K×B4`、`32K×B2`、`64K×B1`，容量测试继续冲 `128K`；
- Prefix Cache：`64K` 上下文、`48K` 共享前缀、3 次命中/1 次冷启动；
- NVLink：从 8 KiB 到 256 MiB 的 NCCL AllReduce 延迟与带宽；
- 可选长稳态：仅在 `RUN_SOAK=1` 时运行混合负载，不纳入默认流程；
- 极限边界：最高 B256、128K×B2、64K×B4。极限项 OOM 会被记录为容量边界，
  不会抹掉已完成的主矩阵结果。

## 1. 运行前提

必须退出云平台的“无卡模式”，并确认以下命令正常：

```bash
nvidia-smi -L
nvidia-smi topo -m
nvcc --version

cd /root/autodl-tmp/nano-vllm
source .venv/bin/activate
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

`nvidia-smi topo -m` 的 GPU0/GPU1 交叉单元应显示 `NV#`，而不是 `PHB`、`PIX`
或 `SYS`。严格 preflight 还会检查两张卡均为约 80GB 的 A800、P2P 可用、NCCL
可用、模型维度可被 TP=2 切分，以及固定端口 2333 未被占用。

## 2. 配置

```bash
cd /root/autodl-tmp/nano-vllm/tests/A800-80G-NVLink-TP2
cp config.example.sh config.sh
sed -n '1,120p' config.sh
```

至少修改 `MODEL_PATH` 和 `PYTHON_BIN`。`config.sh` 已被 `.gitignore` 忽略，不会
意外提交服务器路径。

## 3. 一键运行

```bash
cd /root/autodl-tmp/nano-vllm
source .venv/bin/activate
chmod +x tests/A800-80G-NVLink-TP2/*.sh

tmux new -s a800
SUITE_MODE=full \
tests/A800-80G-NVLink-TP2/run_all.sh \
/root/autodl-tmp/models/Qwen3.5-9B
```

tmux 中按 `Ctrl-b`，松开后按 `d` 可退出但保持任务运行；恢复使用：

```bash
tmux attach -t a800
```

模式说明：

- `quick`：只验证链路及 B16/B64 主流程；
- `full`：完整主矩阵、64K/128K 容量、Prefix 和 B64 稳态；
- `extreme`：在 full 基础上继续冲 B256、128K×B2，并将稳态提高到 B128/50轮。

推荐先跑 `quick`，确认无环境错误后直接跑 `full`。`extreme` 应最后运行，因为它
有意寻找 OOM/超时边界。

## 4. 测试口径

所有性能比较都使用相同 prompt、batch、output tokens、repeats 和后端以外的引擎
配置。主矩阵预热后重复 3 次，报告均值。吞吐、TPOT 与 TTFT 来自引擎请求统计；
NCCL 使用 CUDA Event 计时；Graph 另外记录 replay 和 eager fallback。

简历数字只从主矩阵和 Prefix 产物中选择。容量项用于描述“支持/验证到某上下文或
Batch 边界”，不应把 OOM 点写成成功点。A800 与旧 RTX 5090 数据不混算百分比。

## 5. 结果目录

每次运行写入独立时间戳目录：

```text
tests/A800-80G-NVLink-TP2/results/YYYYmmdd_HHMMSS/
├── preflight.json
├── nccl_stress.json
├── headline_tp1_eager_cuda.json
├── headline_tp2_eager_cuda.json
├── headline_tp2_eager_fla.json
├── headline_tp2_graph_cuda.json
├── long_context_tp2_eager.json
├── prefix_64k_48k_tp2.json
├── soak_tp2_graph.json
├── capacity/
├── status.tsv
├── summary.md
├── summary.json
└── resume_candidates.md
```

`resume_candidates.md` 自动列出可用于更新简历的区间，但仍应查看 `summary.md`，
选择没有 fallback、无异常抖动且口径一致的数据。

## 6. 打包下载

```bash
cd /root/autodl-tmp/nano-vllm/tests/A800-80G-NVLink-TP2/results
LATEST="$(ls -dt 20* | head -1)"
tar -czf "/root/autodl-tmp/a800_nvlink_tp2_${LATEST}.tar.gz" "${LATEST}"
ls -lh "/root/autodl-tmp/a800_nvlink_tp2_${LATEST}.tar.gz"
```

下载 `.tar.gz` 后再关机。至少确认压缩包内存在 `status.tsv`、`summary.md`、
`resume_candidates.md` 以及四个 `headline_*.json`。

## 7. Nsight Systems / Nsight Compute（可选）

建议先完成主压测并保存结果，再单独运行完整 profiling，避免插桩影响性能数字：

```bash
cd /root/autodl-tmp/nano-vllm
source .venv/bin/activate

PROFILE_MODE=full \
RUN_NSYS=1 \
RUN_NCU=1 \
tests/A800-80G-NVLink-TP2/run_profiling_all.sh \
/root/autodl-tmp/models/Qwen3.5-9B
```

Nsight Systems 覆盖：

- TP1 State-Aware Eager，P512/B128；
- TP2 FLA Eager，P512/B128；
- TP2 State-Aware Eager 与 CUDA Graph，P512/B128；
- TP2 64K context；
- 双卡 NCCL AllReduce。

Nsight Compute 使用 `--set full`，分别采集自研 Recurrent 和 Causal Conv Kernel
在 B16/B64/B128 下的完整报告。若云平台禁止 Performance Counter，会生成
`ncu/PERMISSION_BLOCKED.txt`，并停止后续无意义重试；这不影响 CUDA Event、NCCL
和 Nsight Systems 数据。

产物分别位于：

```text
profiling_*/nsys/*.nsys-rep
profiling_*/ncu/*.ncu-rep
profiling_*/ncu/*_details.csv
profiling_*/profiling_summary.md
```

将 `.nsys-rep` 下载到本机，用不旧于服务器 CLI 版本的 **NVIDIA Nsight Systems**
图形界面打开；`.ncu-rep` 则使用 **NVIDIA Nsight Compute** 图形界面打开。二者是
不同应用，不能相互打开对方的报告。

