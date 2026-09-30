# A800 NVLink TP=2 完整复现实验手册

> 本文档记录从租用服务器、配置环境、下载源码与模型，到 TP=2 Quick/Full、Kernel、Profiler、Nsight、归档下载的完整流程。目标环境为 **2×A800 80GB NVLink**。

## 0. 实验目标

1. 证明两张 A800 真正通过 NVLink 互联，P2P 可用。
2. 证明 TP=2 能加载 Qwen3.5-9B 并正常生成。
3. 比较 TP=1 Eager、TP=2 Eager 和 TP=2 CUDA Graph 的端到端性能。
4. 完成 Prefix Cache、Dynamic Scheduling、Vision 和 State-Aware Kernel 测试。
5. 保存 PyTorch trace、CUDA Event、Nsight Systems 和可用时的 Nsight Compute 报告。
6. 将代码版本、软硬件环境、原始数据和摘要一次性打包下载。

> 仓库中的目录名仍是 `tests/5090-test-TP2`，但这套脚本不限于 RTX 5090，A800 也可以使用。


## 1. 租机器时怎么选

截图中的配置可以使用：

- GPU：A800 80GB NVLink ×2。
- 镜像：PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8。
- 驱动：580.126.09。宿主机显示 CUDA 13.0 没问题，只要驱动兼容镜像内 CUDA 12.8 的 PyTorch。
- CPU/RAM：36 核/240GB 足够。
- 数据盘：50GB 偏紧。Qwen3.5-9B 权重、Hugging Face 缓存、Nsight 报告和最终压缩包会同时占空间，建议扩容到 **100GB 或更大**。

模型、代码和报告都放在 `/root/autodl-tmp`，不要放进 30GB 系统盘。

## 2. 登录后验证硬件和 NVLink

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi topo -p2p p
nvidia-smi nvlink --status
```

重点看 `nvidia-smi topo -m`：GPU0 与 GPU1 的交叉位置应出现 `NV#`，例如 `NV4`。如果显示 `PHB`、`PIX` 或 `SYS`，就不是我们要测试的 NVLink 路径。`nvidia-smi topo -p2p p` 也应确认两卡之间 P2P 可用。

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("NCCL:", torch.cuda.nccl.version())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, "capability=", torch.cuda.get_device_capability(i),
          "memory_GiB=", round(p.total_memory / 1024**3, 2),
          "bf16=", torch.cuda.is_bf16_supported())
if torch.cuda.device_count() >= 2:
    print("GPU0 -> GPU1 peer access:", torch.cuda.can_device_access_peer(0, 1))
    print("GPU1 -> GPU0 peer access:", torch.cuda.can_device_access_peer(1, 0))
PY
```

A800 是 Ampere，Compute Capability 为 8.0：

```bash
export TORCH_CUDA_ARCH_LIST=8.0
```

## 3. 安装基础工具并使用 tmux

```bash
apt-get update
apt-get install -y git git-lfs rsync tmux wget curl ca-certificates \
  build-essential ninja-build pkg-config unzip zip jq pciutils lsof
git lfs install
tmux -V
```

所有长任务放在 tmux 中：

```bash
tmux new -s tp2
```

- 暂时退出但不停止任务：按 `Ctrl+B`，松开后按 `D`。
- 重新进入：`tmux attach -t tp2`。
- 查看会话：`tmux ls`。
- 不要在任务运行时直接输入 `exit`，否则会话会结束。

## 4. 获取项目代码

### 4.1 从 GitHub 克隆

```bash
cd /root/autodl-tmp
git clone git@github.com:Bluexvy/NanoHybrid-VLM.git nano-vllm
cd /root/autodl-tmp/nano-vllm
git status -sb
git rev-parse --short HEAD
```

如果服务器没有 GitHub SSH Key，可以使用 HTTPS，或者从本地同步。

### 4.2 从本地同步

在**本地终端**执行：

```bash
rsync -avP \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'tests/5090-test-TP2/results' \
  -e "ssh -p <SSH_PORT>" \
  /workspace/nano-vllm/ \
  root@<SERVER_HOST>:/root/autodl-tmp/nano-vllm/
```

替换 `<SSH_PORT>` 和 `<SERVER_HOST>`。不要上传本地 `.venv`；模型也让服务器自行下载。

## 5. 创建 Python 环境

镜像已经包含 PyTorch 2.8.0+cu128。为了避免 pip 重新安装 PyTorch，让虚拟环境继承镜像包：

```bash
cd /root/autodl-tmp/nano-vllm
python -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install -e . --no-deps
```

先看基础版本：

```bash
python - <<'PY'
import sys, torch, triton
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("triton:", triton.__version__)
PY
```

上一台 A100 PCIe 服务器跑通的关键版本如下：

| 包 | 版本 |
|---|---:|
| Python | 3.12 |
| PyTorch | 2.8.0+cu128 |
| Triton | 3.4.0 |
| transformers | 5.14.1 |
| flash-linear-attention | 0.5.1 |
| fla-core | 0.5.1 |
| causal-conv1d | 1.6.2.post1 |

```bash
python -m pip install \
  'transformers==5.14.1' \
  'flash-linear-attention==0.5.1' \
  'fla-core==0.5.1' \
  huggingface_hub safetensors sentencepiece pillow xxhash pytest

python -m pip install 'causal-conv1d==1.6.2.post1' \
  --no-build-isolation --no-cache-dir
```

`causal-conv1d` 会下载与 Python/PyTorch/CUDA/CXX ABI 对应的 wheel。本环境对应：

```text
causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

如果 GitHub 超时，在本地下载该 wheel 后上传到 `/root/autodl-tmp/wheels/`：

```bash
python -m pip install \
  /root/autodl-tmp/wheels/causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

如果 `flash_attn` 不存在再安装：

```bash
python -c 'import flash_attn; print(flash_attn.__version__)' || \
python -m pip install flash-attn --no-build-isolation
```

验证项目真正使用的依赖：

```bash
python - <<'PY'
import torch
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
import nanovllm
print("all dependencies OK")
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
PY
```

## 6. 在服务器下载 Qwen3.5-9B

```bash
source /root/autodl-tmp/nano-vllm/.venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface-cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=3600

mkdir -p /root/autodl-tmp/models/Qwen3.5-9B

hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir /root/autodl-tmp/models/Qwen3.5-9B \
  --max-workers 2
```

```bash
du -sh /root/autodl-tmp/models/Qwen3.5-9B
find /root/autodl-tmp/models/Qwen3.5-9B -maxdepth 1 -type f -printf '%f\n' | sort
```

固定 revision 是为了让 A800 NVLink 与之前 A100 PCIe 使用完全相同的权重。

## 7. 测试前统一设置并记录环境

```bash
cd /root/autodl-tmp/nano-vllm
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=8.0
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

chmod +x tests/5090-test-TP2/run_all.sh
chmod +x tests/5090-test-TP2/profile_nccl.sh

mkdir -p /root/autodl-tmp/a800_nvlink_environment
git status -sb | tee /root/autodl-tmp/a800_nvlink_environment/git_status.txt
git rev-parse HEAD | tee /root/autodl-tmp/a800_nvlink_environment/git_commit.txt
python -m pip freeze > /root/autodl-tmp/a800_nvlink_environment/pip_freeze.txt
nvidia-smi -q > /root/autodl-tmp/a800_nvlink_environment/nvidia_smi_q.txt
nvidia-smi topo -m > /root/autodl-tmp/a800_nvlink_environment/nvidia_smi_topo.txt
nvidia-smi nvlink --status > /root/autodl-tmp/a800_nvlink_environment/nvlink_status.txt
```

## 8. Quick 测试

先用 Quick 在几分钟内排除环境、NCCL、TP、Graph 和功能错误：

```bash
cd /root/autodl-tmp/nano-vllm
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 \
PYTHON_BIN=/root/autodl-tmp/nano-vllm/.venv/bin/python \
SUITE_MODE=quick \
RUN_TP1_BASELINE=1 \
RUN_GRAPH=1 \
RUN_PREFIX=1 \
RUN_DYNAMIC=1 \
RUN_VISION=1 \
RUN_NSYS=0 \
GPU_MEMORY_UTILIZATION=0.78 \
NCCL_DEBUG=WARN \
./tests/5090-test-TP2/run_all.sh \
/root/autodl-tmp/models/Qwen3.5-9B
```

## 9. Full 测试

```bash
cd /root/autodl-tmp/nano-vllm
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 \
PYTHON_BIN=/root/autodl-tmp/nano-vllm/.venv/bin/python \
SUITE_MODE=full \
RUN_TP1_BASELINE=1 \
RUN_GRAPH=1 \
RUN_PREFIX=1 \
RUN_DYNAMIC=1 \
RUN_VISION=1 \
RUN_NSYS=0 \
GPU_MEMORY_UTILIZATION=0.78 \
NCCL_DEBUG=WARN \
./tests/5090-test-TP2/run_all.sh \
/root/autodl-tmp/models/Qwen3.5-9B
```

Full 覆盖 Prompt 128 的 B=1/2/4/8/16、Prompt 2048 的 B=1/4/8、128 个输出 token 和 3 次重复，并包含 TP=1 Eager、TP=2 Eager、TP=2 Graph、NCCL、Prefix、Dynamic、Vision 和 FLA smoke。

```bash
RESULT_DIR=$(ls -dt tests/5090-test-TP2/results/* | head -1)
echo "$RESULT_DIR"
cat "$RESULT_DIR/status.tsv"
sed -n '1,240p' "$RESULT_DIR/summary.md"
find "$RESULT_DIR" -maxdepth 2 -type f -printf '%p %k KB\n' | sort
```

至少同时观察：TP=1/TP=2 的 tok/s、TPOT 和端到端耗时；TP=2 Eager/Graph 差异；B=1/B=16 差异；Prompt 128/2048 差异；NCCL All-Reduce 微基准。

### 9.1 关于严格 token 比较

Full 中 `compare_tp1_tp2` 或 `compare_eager_graph` 可能因长自回归中的极小浮点差异造成后续 token 分叉而失败。之前 A100 上 Quick 短序列比较通过，Full 在较后 token 才分叉，而各配置自身重复运行稳定。因此：

- 不要只凭这两个 strict compare 就断言 TP=2 没做成。
- 也不要隐藏差异；保留 FAIL、首个分叉位置和重复稳定性。
- 综合 Quick 对齐、Kernel 单元测试、生成可用性和 Full 内部重复稳定性判断正确性。

## 10. 独立运行 State-Aware Kernel 测试

这些文件是可执行脚本，不是标准 pytest 测试收集格式。之前使用 `pytest` 出现 `no tests ran` 并不代表测试通过或失败，而是 pytest 没有收集到测试。应直接逐个运行：

```bash
cd /root/autodl-tmp/nano-vllm
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=8.0

set -o pipefail
for TEST_SCRIPT in \
  tests/kernels/test_state_aware_causal_conv_reference.py \
  tests/kernels/test_state_aware_causal_conv_cuda.py \
  tests/kernels/test_state_aware_gdn_reference.py \
  tests/kernels/test_state_aware_gdn_triton.py \
  tests/kernels/test_state_aware_gdn_cuda_extension.py
do
  echo "========== $TEST_SCRIPT =========="
  python "$TEST_SCRIPT" || exit 1
done 2>&1 | tee /root/autodl-tmp/kernel_unit_tests.log
```

首次运行 CUDA Extension 时会用 `nvcc` 编译，几分钟内长时间停在 C++/CUDA 编译行通常是正常现象。编译缓存后再次运行会快很多。

## 11. 常见问题和处理方法

### 11.1 `Permission denied: ./tests/.../run_all.sh`

```bash
chmod +x tests/5090-test-TP2/run_all.sh
```

或者显式使用：

```bash
bash tests/5090-test-TP2/run_all.sh /root/autodl-tmp/models/Qwen3.5-9B
```

### 11.2 `TCP port 2333 is occupied`

ModelRunner 使用固定端口 2333。先查是谁占用，确认是前一次残留进程后再结束：

```bash
ss -ltnp | grep ':2333'
lsof -iTCP:2333 -sTCP:LISTEN
ps -fp <PID>
kill <PID>
sleep 2
ss -ltnp | grep ':2333' || true
```

只有普通 `kill` 无效且已经确认是残留进程时，才使用 `kill -9 <PID>`。不要使用 `pkill python`，它可能误杀下载、测试或其他人的进程。

### 11.3 共享内存残留

确保没有 nano-vLLM 进程仍在运行，再检查：

```bash
ls -l /dev/shm/nanovllm
```

只有确认它是上次异常退出留下的文件时才删除：

```bash
rm -f /dev/shm/nanovllm
```

### 11.4 `libgomp: Invalid value for OMP_NUM_THREADS`

```bash
export OMP_NUM_THREADS=1
```

不要把它设置成空字符串。

### 11.5 `causal-conv1d` 构建很久后失败

如果日志中出现 `Guessing wheel URL` 后网络超时，问题通常不是编译错误，而是 GitHub wheel 下载失败。使用第 5 节中的匹配 wheel 离线上传安装。

### 11.6 任务究竟还在不在跑

```bash
ps -ef | grep -E 'run_all|run_generation|profile_walkthrough' | grep -v grep
nvidia-smi
```

有 Python 进程、GPU 显存和利用率变化、终端未返回 shell 提示符，通常说明仍在运行。CUDA Extension 第一次编译时 GPU 可能暂时空闲，要同时看编译进程。

## 12. PyTorch Profiler、CUDA Event 和端到端 Benchmark

三个工具回答的问题不同：

- PyTorch Profiler trace：看一次 Decode 中 CPU 调度、算子、NVTX 区间和 CUDA Kernel 的完整时间线，适合发现 Gather/FLA/Scatter 等阶段关系。
- CUDA Event：只测指定 GPU 区间的实际设备时间，适合稳定比较 FLA 与 State-Aware CUDA 路径。
- 端到端 Benchmark：包含调度、模型执行、采样和同步，回答优化是否真的改善用户可见性能。

创建结果目录：

```bash
cd /root/autodl-tmp/nano-vllm
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=8.0

PROFILE_DIR=/root/autodl-tmp/profile_a800_$(date +%Y%m%d_%H%M%S)
mkdir -p "$PROFILE_DIR"
echo "$PROFILE_DIR"
```

### 12.1 生成 trace

分别采集未优化 FLA 和 State-Aware CUDA，B=1：

```bash
python tests/kernels/profile_walkthrough.py \
  --model /root/autodl-tmp/models/Qwen3.5-9B \
  --phase trace --backend fla --mode eager \
  --batch-size 1 --warmup 4 --steps 4 --repeats 1 \
  --output-dir "$PROFILE_DIR" \
  2>&1 | tee "$PROFILE_DIR/trace_fla_b1.log"

python tests/kernels/profile_walkthrough.py \
  --model /root/autodl-tmp/models/Qwen3.5-9B \
  --phase trace --backend state_aware_cuda --mode eager \
  --batch-size 1 --warmup 4 --steps 4 --repeats 1 \
  --output-dir "$PROFILE_DIR" \
  2>&1 | tee "$PROFILE_DIR/trace_cuda_b1.log"
```

生成的 `.json` trace 可下载后拖进 [Perfetto](https://ui.perfetto.dev/)；左侧可能同时看到 `python <PID>` 和 `python 0`，它们通常分别是 CPU/PyTorch 线程轨道与 CUDA 设备/流轨道，不是程序重复运行了两次。

### 12.2 CUDA Event

```bash
for BACKEND in fla state_aware_cuda; do
  for BATCH in 1 16; do
    python tests/kernels/profile_walkthrough.py \
      --model /root/autodl-tmp/models/Qwen3.5-9B \
      --phase event --backend "$BACKEND" --mode eager \
      --batch-size "$BATCH" --warmup 4 --steps 12 --repeats 1 \
      --output-dir "$PROFILE_DIR" \
      2>&1 | tee "$PROFILE_DIR/event_${BACKEND}_b${BATCH}.log"
  done
done
```

### 12.3 稳定 Benchmark

```bash
for BACKEND in fla state_aware_cuda; do
  for BATCH in 1 16; do
    python tests/kernels/profile_walkthrough.py \
      --model /root/autodl-tmp/models/Qwen3.5-9B \
      --phase benchmark --backend "$BACKEND" --mode eager \
      --batch-size "$BATCH" --warmup 4 --steps 16 --repeats 3 \
      --output-dir "$PROFILE_DIR" \
      2>&1 | tee "$PROFILE_DIR/benchmark_${BACKEND}_b${BATCH}.log"
  done
done
```

正确分析顺序是：trace 发现可疑阶段 → CUDA Event 定量 → 修改实现 → Event 复测 → 端到端 Benchmark 验证收益。不要只凭 GPU 利用率柱状图判断“在搬运”；要结合 Kernel 名称、Memcpy、NVTX 区间和事件时间。

## 13. 安装 `nvcc`、Nsight Systems 和 Nsight Compute

镜像里能运行 PyTorch 不等于包含完整 CUDA Toolkit；因此可能有 CUDA runtime，却没有 `nvcc`、`ncu` 和 `nsys`。

### 13.1 CUDA 12.8 Toolkit/NVCC

```bash
cd /root/autodl-tmp
wget -O cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring.deb
apt-get update
apt-get install -y cuda-nvcc-12-8

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
```

### 13.2 Nsight Compute CLI

```bash
NCU_PACKAGE=$(apt-cache search '^nsight-compute-[0-9]' | awk '{print $1}' | sort -V | tail -1)
echo "$NCU_PACKAGE"
apt-get install -y --no-install-recommends "$NCU_PACKAGE"
ncu --version
```

### 13.3 Nsight Systems CLI

```bash
wget -qO- https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/7fa2af80.pub \
  | gpg --dearmor \
  | tee /usr/share/keyrings/nvidia-devtools-keyring.gpg >/dev/null

echo 'deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' \
  | tee /etc/apt/sources.list.d/nvidia-devtools.list

apt-get update
apt-get install -y nsight-systems-cli
nsys --version
```

如果命令装好了但新终端找不到 `nvcc`，重新执行：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

## 14. Nsight Compute：看单个 Kernel 的硬件指标

先做最小 smoke test。云平台有时禁止容器访问 GPU Performance Counter，即使容器里是 root 也无法解除：

```bash
NCU_DIR=/root/autodl-tmp/ncu_a800_$(date +%Y%m%d_%H%M%S)
mkdir -p "$NCU_DIR"

CUDA_VISIBLE_DEVICES=0 ncu \
  --set basic \
  --launch-count 1 \
  --force-overwrite \
  --export "$NCU_DIR/smoke" \
  python -c 'import torch; x=torch.randn(1024,1024,device="cuda"); y=x@x; torch.cuda.synchronize(); print(y[0,0])' \
  2>&1 | tee "$NCU_DIR/smoke.log"
```

如果 smoke 成功，再抓 State-Aware recurrent kernel：

```bash
CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set basic \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*state_aware_gdn_bf16_kernel.*' \
  --launch-count 1 \
  --force-overwrite \
  --export "$NCU_DIR/gdn_recurrent" \
  /root/autodl-tmp/nano-vllm/.venv/bin/python \
  tests/kernels/profile_walkthrough.py \
  --model /root/autodl-tmp/models/Qwen3.5-9B \
  --phase benchmark --backend state_aware_cuda --mode eager \
  --batch-size 16 --warmup 2 --steps 2 --repeats 1 \
  --output-dir "$NCU_DIR" \
  2>&1 | tee "$NCU_DIR/gdn_recurrent.log"
```

成功时会生成 `gdn_recurrent.ncu-rep`。下载到本地后用 NVIDIA Nsight Compute GUI（`ncu-ui`）打开，可看：

- Kernel Duration。
- Compute Throughput 与 Memory Throughput。
- DRAM/L2 吞吐。
- Achieved Occupancy。
- Warp Stall 原因。
- 寄存器、Shared Memory 和 Block 配置。

如果出现：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

这不是项目错误，而是云平台宿主机没有开放计数器权限；容器内安装工具或使用 root 都不能解决。保存错误日志，跳过 NCU，继续 CUDA Event 和 Nsight Systems。之前 A100 PCIe 实验就遇到了这个限制。

## 15. Nsight Systems：看端到端时间线和 NCCL

### 15.1 先做 smoke test

```bash
NSYS_DIR=/root/autodl-tmp/nsys_a800_$(date +%Y%m%d_%H%M%S)
mkdir -p "$NSYS_DIR"

CUDA_VISIBLE_DEVICES=0 nsys profile \
  --force-overwrite=true \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda,nvtx \
  --output="$NSYS_DIR/smoke" \
  python -c 'import torch; x=torch.randn(1024,1024,device="cuda"); y=x@x; torch.cuda.synchronize(); print(y[0,0])' \
  2>&1 | tee "$NSYS_DIR/smoke.log"
```

应生成 `smoke.nsys-rep`。如果只有 JSON 而没有 `.nsys-rep`，先不要加 NVTX capture-range，使用上面的全进程采集确认工具链可用。

### 15.2 TP=2 Eager 与 Graph

```bash
cd /root/autodl-tmp/nano-vllm
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

for MODE in eager graph; do
  nsys profile \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda,nvtx \
    --output="$NSYS_DIR/tp2_${MODE}_b16" \
    python tests/5090-test-TP2/run_generation.py \
      --model /root/autodl-tmp/models/Qwen3.5-9B \
      --output "$NSYS_DIR/tp2_${MODE}_b16.json" \
      --label "tp2_${MODE}_b16" \
      --tp-size 2 \
      --mode "$MODE" \
      --backend state_aware_cuda \
      --cases 128:16 \
      --output-tokens 16 \
      --repeats 1 \
      --warmup-output-tokens 4 \
      --token-budget 512 \
      --gpu-memory-utilization 0.78 \
      --graph-buckets 16 \
    2>&1 | tee "$NSYS_DIR/tp2_${MODE}_b16.log"
done
```

导出汇总 CSV：

```bash
for REPORT in "$NSYS_DIR"/*.nsys-rep; do
  BASE=${REPORT%.nsys-rep}
  nsys stats \
    --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
    --format csv \
    --output "${BASE}_stats" \
    "$REPORT" \
    2>&1 | tee "${BASE}_stats.log"
done
```

`.nsys-rep` 下载后用本地 Nsight Systems GUI（`nsys-ui`）打开。重点看：

- 两张 GPU 的 Kernel 是否并行推进。
- 每层 GEMM 与 NCCL All-Reduce 的先后关系。
- NCCL Kernel 是否成为 Decode step 的长尾。
- CUDA Graph 后 CPU launch gap 是否明显缩短。
- GPU 上是否有大段没有 Kernel 的空白区间。

注意：NCCL kernel 的持续时间可能包含等待其他 rank 的时间，不能直接把整段都解释成“链路传输耗时”。应结合 NCCL 微基准、两卡时间线和端到端 TP=1/TP=2 一起判断。

## 16. 一次性采集剩余证据

完成 Quick 和 Full 后，可以把下面脚本保存为 `/root/autodl-tmp/nano-vllm/collect_a800_nvlink.sh`。它不会重跑耗时最长的 Full，而是复制最新 Full 结果，然后运行 Kernel、trace、Event、Benchmark、NCU smoke 和 TP=2 Nsys。

```bash
#!/usr/bin/env bash
set -u -o pipefail

ROOT=/root/autodl-tmp/nano-vllm
MODEL=/root/autodl-tmp/models/Qwen3.5-9B
PYTHON_BIN="$ROOT/.venv/bin/python"
STAMP=$(date +%Y%m%d_%H%M%S)
FINAL_DIR="/root/autodl-tmp/final_a800_nvlink_tp2_${STAMP}"
LOG_DIR="$FINAL_DIR/logs"
PROFILE_DIR="$FINAL_DIR/profile"
NSYS_DIR="$FINAL_DIR/nsys"
NCU_DIR="$FINAL_DIR/ncu"
STATUS="$FINAL_DIR/status.tsv"

mkdir -p "$LOG_DIR" "$PROFILE_DIR" "$NSYS_DIR" "$NCU_DIR" "$FINAL_DIR/environment"
printf 'stage\tstatus\texit_code\n' > "$STATUS"

cd "$ROOT" || exit 1
source .venv/bin/activate
export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=8.0
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

record_status() {
  local name=$1 code=$2
  local state=PASS
  if [ "$code" -ne 0 ]; then state=FAIL; fi
  printf '%s\t%s\t%s\n' "$name" "$state" "$code" | tee -a "$STATUS"
}

run_logged() {
  local name=$1
  shift
  echo "========== $name =========="
  set +e
  "$@" >"$LOG_DIR/${name}.log" 2>&1
  local code=$?
  set -e
  record_status "$name" "$code"
  return 0
}

set -e

# 1. 环境证据
git status -sb > "$FINAL_DIR/environment/git_status.txt" 2>&1 || true
git rev-parse HEAD > "$FINAL_DIR/environment/git_commit.txt" 2>&1 || true
"$PYTHON_BIN" -m pip freeze > "$FINAL_DIR/environment/pip_freeze.txt" 2>&1 || true
nvidia-smi -q > "$FINAL_DIR/environment/nvidia_smi_q.txt" 2>&1 || true
nvidia-smi topo -m > "$FINAL_DIR/environment/nvidia_smi_topo.txt" 2>&1 || true
nvidia-smi topo -p2p p > "$FINAL_DIR/environment/nvidia_smi_p2p.txt" 2>&1 || true
nvidia-smi nvlink --status > "$FINAL_DIR/environment/nvlink_status.txt" 2>&1 || true
nvcc --version > "$FINAL_DIR/environment/nvcc_version.txt" 2>&1 || true
ncu --version > "$FINAL_DIR/environment/ncu_version.txt" 2>&1 || true
nsys --version > "$FINAL_DIR/environment/nsys_version.txt" 2>&1 || true

# 2. 复制最新 Full 结果
LATEST_RESULT=$(ls -dt tests/5090-test-TP2/results/* 2>/dev/null | head -1 || true)
if [ -n "$LATEST_RESULT" ]; then
  cp -a "$LATEST_RESULT" "$FINAL_DIR/full_suite_result"
  record_status copy_full_result 0
else
  echo "No suite result found" > "$LOG_DIR/copy_full_result.log"
  record_status copy_full_result 1
fi

# 3. Kernel tests
set +e
(
  export CUDA_VISIBLE_DEVICES=0
  for TEST_SCRIPT in \
    tests/kernels/test_state_aware_causal_conv_reference.py \
    tests/kernels/test_state_aware_causal_conv_cuda.py \
    tests/kernels/test_state_aware_gdn_reference.py \
    tests/kernels/test_state_aware_gdn_triton.py \
    tests/kernels/test_state_aware_gdn_cuda_extension.py
  do
    echo "========== $TEST_SCRIPT =========="
    "$PYTHON_BIN" "$TEST_SCRIPT" || exit 1
  done
) > "$LOG_DIR/kernel_tests.log" 2>&1
record_status kernel_tests $?
set -e

# 4. PyTorch trace
for BACKEND in fla state_aware_cuda; do
  run_logged "trace_${BACKEND}_b1" env CUDA_VISIBLE_DEVICES=0 \
    "$PYTHON_BIN" tests/kernels/profile_walkthrough.py \
    --model "$MODEL" --phase trace --backend "$BACKEND" --mode eager \
    --batch-size 1 --warmup 4 --steps 4 --repeats 1 \
    --output-dir "$PROFILE_DIR"
done

# 5. CUDA Event 与 Benchmark
for BACKEND in fla state_aware_cuda; do
  for BATCH in 1 16; do
    run_logged "event_${BACKEND}_b${BATCH}" env CUDA_VISIBLE_DEVICES=0 \
      "$PYTHON_BIN" tests/kernels/profile_walkthrough.py \
      --model "$MODEL" --phase event --backend "$BACKEND" --mode eager \
      --batch-size "$BATCH" --warmup 4 --steps 12 --repeats 1 \
      --output-dir "$PROFILE_DIR"

    run_logged "benchmark_${BACKEND}_b${BATCH}" env CUDA_VISIBLE_DEVICES=0 \
      "$PYTHON_BIN" tests/kernels/profile_walkthrough.py \
      --model "$MODEL" --phase benchmark --backend "$BACKEND" --mode eager \
      --batch-size "$BATCH" --warmup 4 --steps 16 --repeats 3 \
      --output-dir "$PROFILE_DIR"
  done
done

# 6. NCU smoke；云平台禁用计数器时允许失败
if command -v ncu >/dev/null 2>&1; then
  run_logged ncu_smoke env CUDA_VISIBLE_DEVICES=0 ncu \
    --set basic --launch-count 1 --force-overwrite \
    --export "$NCU_DIR/smoke" \
    "$PYTHON_BIN" -c \
    'import torch; x=torch.randn(1024,1024,device="cuda"); y=x@x; torch.cuda.synchronize(); print(y[0,0])'
else
  echo "ncu not installed" > "$LOG_DIR/ncu_smoke.log"
  record_status ncu_smoke 127
fi

# 7. Nsight Systems：TP=2 Eager/Graph B=16
if command -v nsys >/dev/null 2>&1; then
  for MODE in eager graph; do
    run_logged "nsys_tp2_${MODE}_b16" env CUDA_VISIBLE_DEVICES=0,1 nsys profile \
      --force-overwrite=true --sample=none --cpuctxsw=none --trace=cuda,nvtx \
      --output="$NSYS_DIR/tp2_${MODE}_b16" \
      "$PYTHON_BIN" tests/5090-test-TP2/run_generation.py \
      --model "$MODEL" \
      --output "$NSYS_DIR/tp2_${MODE}_b16.json" \
      --label "tp2_${MODE}_b16" --tp-size 2 --mode "$MODE" \
      --backend state_aware_cuda --cases 128:16 \
      --output-tokens 16 --repeats 1 --warmup-output-tokens 4 \
      --token-budget 512 --gpu-memory-utilization 0.78 --graph-buckets 16
  done

  for REPORT in "$NSYS_DIR"/*.nsys-rep; do
    [ -f "$REPORT" ] || continue
    BASE=${REPORT%.nsys-rep}
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
      --format csv --output "${BASE}_stats" "$REPORT" \
      > "${BASE}_stats.log" 2>&1 || true
  done
else
  echo "nsys not installed" > "$LOG_DIR/nsys.log"
  record_status nsys 127
fi

# 8. 清单和压缩包
find "$FINAL_DIR" -type f -printf '%P\t%s bytes\n' | sort > "$FINAL_DIR/manifest.tsv"
ARCHIVE="/root/autodl-tmp/nanohybrid_a800_nvlink_tp2_${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C /root/autodl-tmp "$(basename "$FINAL_DIR")"
sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
printf '%s\n' "$ARCHIVE" | tee /root/autodl-tmp/LATEST_FINAL_ARCHIVE.txt

echo
echo "FINAL_DIR=$FINAL_DIR"
echo "ARCHIVE=$ARCHIVE"
echo "STATUS=$STATUS"
cat "$STATUS"
```

保存并运行：

```bash
cd /root/autodl-tmp/nano-vllm
chmod +x collect_a800_nvlink.sh
tmux new-session -d -s collect 'bash ./collect_a800_nvlink.sh'
tmux attach -t collect
```

如果已经在 tmux 内，不要再嵌套创建，直接执行 `bash ./collect_a800_nvlink.sh`。

## 17. 下载结果到本地

采集脚本最后会打印压缩包路径，也会写入：

```bash
cat /root/autodl-tmp/LATEST_FINAL_ARCHIVE.txt
```

在服务器检查压缩包和校验值：

```bash
ARCHIVE=$(cat /root/autodl-tmp/LATEST_FINAL_ARCHIVE.txt)
ls -lh "$ARCHIVE" "${ARCHIVE}.sha256"
sha256sum -c "${ARCHIVE}.sha256"
```

然后在**本地终端**下载：

```bash
mkdir -p /workspace/nano-vllm/artifacts/tp2_a800_nvlink

rsync -avP \
  -e "ssh -p <SSH_PORT>" \
  'root@<SERVER_HOST>:/root/autodl-tmp/nanohybrid_a800_nvlink_tp2_<TIMESTAMP>.tar.gz*' \
  /workspace/nano-vllm/artifacts/tp2_a800_nvlink/
```

本地验证：

```bash
cd /workspace/nano-vllm/artifacts/tp2_a800_nvlink
sha256sum -c nanohybrid_a800_nvlink_tp2_<TIMESTAMP>.tar.gz.sha256

tar -tzf nanohybrid_a800_nvlink_tp2_<TIMESTAMP>.tar.gz | head -100
```

AutoDL SSH 地址、端口经常变化，必须使用本次实例控制台显示的值。之前 rsync 失败并出现 `region-9.autodl.pro: command not found`，就是 `-e` 的引号/换行复制错误。正确结构始终是：

```bash
rsync -avP -e "ssh -p 端口" 本地路径 root@主机:远端路径
```

## 18. 与 A100 PCIe 结果如何比较

之前两张 A100 PCIe 的拓扑是 `PHB`，没有 NVLink；本次 A800 应显示 `NV#`。此前结果可作为参考，而不是直接当作严格基准：

| 指标 | 之前 A100 PCIe 结果 |
|---|---:|
| NCCL 8KiB All-Reduce | 73.90 μs |
| NCCL 128KiB All-Reduce | 94.19 μs |
| NCCL 1MiB All-Reduce | 319.58 μs |
| TP=2 Eager / TP=1 Eager | 0.78–0.82× |
| TP=2 Graph / TP=2 Eager | 2.55–5.53× |
| TP=2 Graph，B=16 | 669.72 tok/s，TPOT 25.2 ms |
| 单 A100 State-Aware vs FLA，B=1 | 77.4223 → 64.2153 ms，吞吐 +20.51% |
| 单 A100 State-Aware vs FLA，B=16 | 85.4791 → 74.0395 ms，吞吐 +15.45% |

这次最有价值的比较是：

1. A800 两卡拓扑确实为 NVLink。
2. 相同 payload 下 NCCL All-Reduce 延迟相对 A100 PCIe 是否明显下降。
3. TP=2 Eager 相对 TP=1 的比值是否从此前 0.78–0.82× 提高。
4. B=1 和 B=16 哪个更能摊薄 TP 通信和进程调度开销。
5. Graph 是否仍然主要改善小 batch 的 CPU launch 开销。
6. Nsys 中 NCCL 与 GEMM 的比例和空隙是否缩小。

不要预设“有 NVLink，TP=2 就一定比 TP=1 快”。Qwen3.5-9B 较小，Decode 的单步 GEMM 规模有限，而每层仍需通信；NVLink 只降低通信成本，不会消除 NCCL 同步、跨进程调度、Kernel launch 和模型较小造成的并行效率损失。另外 A800 与 A100 本身算力和频率也可能不同，所以报告里要同时给绝对数和相对比值。

## 19. 最终报告建议保留的表格

### 19.1 环境

| 项目 | 实际值 |
|---|---|
| Git commit |  |
| 模型 revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| GPU | 2×A800 80GB |
| Topology | 例如 `NV4` |
| Driver/CUDA/PyTorch |  |
| NCCL |  |
| GDN backend | `state_aware_cuda` |
| GPU memory utilization | `0.78` |

### 19.2 端到端性能

| Case | TP=1 Eager tok/s | TP=2 Eager tok/s | TP=2/TP=1 | TP=2 Graph tok/s | Graph/Eager |
|---|---:|---:|---:|---:|---:|
| P128 B1 |  |  |  |  |  |
| P128 B4 |  |  |  |  |  |
| P128 B8 |  |  |  |  |  |
| P128 B16 |  |  |  |  |  |
| P2048 B1 |  |  |  |  |  |
| P2048 B4 |  |  |  |  |  |
| P2048 B8 |  |  |  |  |  |

### 19.3 通信和 Kernel

| 项目 | B/Payload | 平均时间 | 备注 |
|---|---:|---:|---|
| NCCL All-Reduce | 8KiB |  |  |
| NCCL All-Reduce | 128KiB |  |  |
| NCCL All-Reduce | 1MiB |  |  |
| FLA Eager Decode | B1 |  | CUDA Event |
| State-Aware CUDA Decode | B1 |  | CUDA Event |
| FLA Eager Decode | B16 |  | CUDA Event |
| State-Aware CUDA Decode | B16 |  | CUDA Event |

## 20. 关机前最后检查

只有以下项目都确认后再关机：

- [ ] `nvidia_smi_topo.txt` 中两卡之间是 `NV#`。
- [ ] Quick 已运行，并保存结果目录。
- [ ] Full 已运行，`status.tsv` 和 `summary.md` 已保存。
- [ ] TP=1 Eager、TP=2 Eager、TP=2 Graph JSON 都在。
- [ ] NCCL smoke/microbenchmark 已运行。
- [ ] Prefix、Dynamic、Vision 状态已记录。
- [ ] 五个 State-Aware Kernel 脚本已直接运行。
- [ ] FLA/State-Aware 的 trace、Event、Benchmark 已保存。
- [ ] 至少生成 TP=2 Eager 和 Graph 的 `.nsys-rep`；若失败，错误日志已保存。
- [ ] NCU 成功生成 `.ncu-rep`，或 `ERR_NVGPUCTRPERM` 日志已保存。
- [ ] 最终 `.tar.gz` 已下载到本地。
- [ ] 本地 `sha256sum -c` 通过。
- [ ] 本地能列出或解压压缩包，关键文件不是 0 字节。

完成这些以后再在 AutoDL 控制台关机。只在服务器上生成压缩包但没有下载和校验，不能算实验资料已经保存。
