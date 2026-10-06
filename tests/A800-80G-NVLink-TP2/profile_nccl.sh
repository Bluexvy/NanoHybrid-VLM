#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${1:?Usage: profile_nccl.sh MODEL_PATH RESULT_DIR}"
RESULT_DIR="${2:?Usage: profile_nccl.sh MODEL_PATH RESULT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "Nsight Systems CLI (nsys) is not installed or not on PATH."
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "${RESULT_DIR}/nsys"

nsys profile \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,osrt \
    --capture-range=nvtx \
    --nvtx-capture=nano_tp2_measure \
    --capture-range-end=stop \
    --output="${RESULT_DIR}/nsys/tp2_b128" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
    --model "${MODEL_PATH}" \
    --label nsys_tp2_b128 \
    --tp-size 2 \
    --mode eager \
    --backend state_aware_cuda \
    --cases 512:128 \
    --output-tokens 256 \
    --repeats 1 \
    --warmup-output-tokens 8 \
    --token-budget 32768 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.82}" \
    --graph-buckets 128 \
    --output "${RESULT_DIR}/nsys/tp2_b128.json"

REPORT="${RESULT_DIR}/nsys/tp2_b128.nsys-rep"
if [[ -f "${REPORT}" ]]; then
    nsys stats \
        --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
        --format csv \
        --output "${RESULT_DIR}/nsys/tp2_b128_stats" \
        "${REPORT}" || true
fi
