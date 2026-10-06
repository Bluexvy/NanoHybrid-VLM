#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${1:?Usage: profile_nsys_all.sh MODEL_PATH OUTPUT_DIR}"
OUTPUT_DIR="${2:?Usage: profile_nsys_all.sh MODEL_PATH OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE_MODE="${PROFILE_MODE:-full}"
NSYS_TRACE_DOMAINS="${NSYS_TRACE_DOMAINS:-cuda,nvtx,osrt,nccl}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "Nsight Systems CLI (nsys) is not installed or not on PATH."
    exit 1
fi
if [[ "${PROFILE_MODE}" != "core" && "${PROFILE_MODE}" != "full" ]]; then
    echo "PROFILE_MODE must be core or full"
    exit 2
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

NSYS_DIR="${OUTPUT_DIR}/nsys"
mkdir -p "${NSYS_DIR}"
FAILURES=0

export_stats() {
    local name="$1"
    local report="${NSYS_DIR}/${name}.nsys-rep"
    [[ -f "${report}" ]] || return 0
    nsys stats \
        --force-export=true \
        --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
        --format csv \
        --output "${NSYS_DIR}/${name}_stats" \
        "${report}" > "${NSYS_DIR}/${name}_stats.log" 2>&1 || true
}

run_profile() {
    local name="$1"
    shift
    echo
    echo "========== nsys: ${name} =========="
    nsys profile \
        --force-overwrite=true \
        --sample=none \
        --cpuctxsw=none \
        --trace="${NSYS_TRACE_DOMAINS}" \
        --cuda-memory-usage=true \
        --capture-range=nvtx \
        --nvtx-capture=nano_tp2_measure \
        --capture-range-end=stop \
        --output="${NSYS_DIR}/${name}" \
        "$@" 2>&1 | tee "${NSYS_DIR}/${name}.console.log"
    local status="${PIPESTATUS[0]}"
    if [[ "${status}" -ne 0 || ! -f "${NSYS_DIR}/${name}.nsys-rep" ]]; then
        printf '%s\tFAIL(%s)_OR_REPORT_MISSING\n' "${name}" "${status}" | tee -a "${NSYS_DIR}/status.tsv"
        FAILURES=$((FAILURES + 1))
    else
        printf '%s\tPASS\n' "${name}" | tee -a "${NSYS_DIR}/status.tsv"
        export_stats "${name}"
    fi
}

run_model_profile() {
    local name="$1"
    local tp_size="$2"
    local mode="$3"
    local backend="$4"
    local case_spec="$5"
    local output_tokens="$6"
    local token_budget="$7"
    local graph_bucket="$8"
    run_profile "${name}" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
        --model "${MODEL_PATH}" --label "nsys_${name}" \
        --tp-size "${tp_size}" --mode "${mode}" --backend "${backend}" \
        --cases "${case_spec}" --output-tokens "${output_tokens}" \
        --repeats 1 --warmup-output-tokens 8 \
        --token-budget "${token_budget}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.82}" \
        --graph-buckets "${graph_bucket}" \
        --output "${NSYS_DIR}/${name}_benchmark.json"
}

: > "${NSYS_DIR}/status.tsv"
nsys --version > "${NSYS_DIR}/version.txt" 2>&1 || true
nvidia-smi topo -m > "${NSYS_DIR}/topology.txt" 2>&1 || true

run_model_profile tp1_eager_cuda_p512_b128 1 eager state_aware_cuda 512:128 64 32768 128
run_model_profile tp2_eager_fla_p512_b128 2 eager fla 512:128 64 32768 128
run_model_profile tp2_eager_cuda_p512_b128 2 eager state_aware_cuda 512:128 64 32768 128
run_model_profile tp2_graph_cuda_p512_b128 2 graph state_aware_cuda 512:128 64 32768 128

if [[ "${PROFILE_MODE}" == "full" ]]; then
    run_model_profile tp2_eager_cuda_p64k_b1 2 eager state_aware_cuda 65536:1 32 32768 1
fi

echo
echo "========== nsys: nccl_allreduce =========="
nsys profile \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace="${NSYS_TRACE_DOMAINS}" \
    --cuda-memory-usage=true \
    --output="${NSYS_DIR}/nccl_allreduce" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/nccl_stress.py" \
    --world-size 2 --warmup 2 --iterations 20 \
    --output "${NSYS_DIR}/nccl_allreduce.json" \
    2>&1 | tee "${NSYS_DIR}/nccl_allreduce.console.log"
NCCL_STATUS="${PIPESTATUS[0]}"
if [[ "${NCCL_STATUS}" -eq 0 && -f "${NSYS_DIR}/nccl_allreduce.nsys-rep" ]]; then
    printf 'nccl_allreduce\tPASS\n' | tee -a "${NSYS_DIR}/status.tsv"
    export_stats nccl_allreduce
else
    printf 'nccl_allreduce\tFAIL(%s)_OR_REPORT_MISSING\n' "${NCCL_STATUS}" | tee -a "${NSYS_DIR}/status.tsv"
    FAILURES=$((FAILURES + 1))
fi

find "${NSYS_DIR}" -maxdepth 1 -type f -printf '%f\t%k KiB\n' | sort > "${NSYS_DIR}/files.tsv"
echo "Nsight Systems reports: ${NSYS_DIR}"
exit "${FAILURES}"
