#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${SCRIPT_DIR}/config.sh" ]]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/config.sh"
fi

MODEL_PATH="${1:-${MODEL_PATH:-}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE_MODE="${SUITE_MODE:-full}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
RUN_TP1_BASELINE="${RUN_TP1_BASELINE:-1}"
RUN_GRAPH="${RUN_GRAPH:-1}"
RUN_PREFIX="${RUN_PREFIX:-1}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"
RUN_VISION="${RUN_VISION:-1}"
RUN_NSYS="${RUN_NSYS:-0}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "Usage: ./run_all.sh /path/to/Qwen3.5-9B"
    echo "Or copy config.example.sh to config.sh and set MODEL_PATH."
    exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model directory does not exist: ${MODEL_PATH}"
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/${TIMESTAMP}}"
mkdir -p "${RESULT_DIR}"
STATUS_FILE="${RESULT_DIR}/status.tsv"
: > "${STATUS_FILE}"

FAILURES=0

run_step() {
    local name="$1"
    shift
    echo
    echo "========== ${name} =========="
    local start_seconds
    start_seconds="$(date +%s)"
    "$@" 2>&1 | tee "${RESULT_DIR}/${name}.log"
    local command_status="${PIPESTATUS[0]}"
    local elapsed="$(( $(date +%s) - start_seconds ))"
    if [[ "${command_status}" -eq 0 ]]; then
        printf '%s\tPASS\t%ss\n' "${name}" "${elapsed}" | tee -a "${STATUS_FILE}"
    else
        printf '%s\tFAIL(%s)\t%ss\n' "${name}" "${command_status}" "${elapsed}" | tee -a "${STATUS_FILE}"
        FAILURES=$((FAILURES + 1))
    fi
    return 0
}

echo "Result directory: ${RESULT_DIR}"
echo "Model: ${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Suite mode: ${SUITE_MODE}"

{
    echo "date: $(date --iso-8601=seconds 2>/dev/null || date)"
    echo "hostname: $(hostname)"
    echo "kernel: $(uname -a)"
    echo "git_commit: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status:"
    git -C "${REPO_ROOT}" status --short 2>/dev/null || true
} > "${RESULT_DIR}/run_metadata.txt"

nvidia-smi -L > "${RESULT_DIR}/nvidia_smi_L.txt" 2>&1 || true
nvidia-smi topo -m > "${RESULT_DIR}/nvidia_smi_topo.txt" 2>&1 || true
nvidia-smi -q > "${RESULT_DIR}/nvidia_smi_q.txt" 2>&1 || true

run_step preflight \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
    --model "${MODEL_PATH}" \
    --tp-size 2 \
    --output "${RESULT_DIR}/preflight.json"

if [[ ! -f "${RESULT_DIR}/preflight.json" ]]; then
    echo "Preflight failed; stopping before model loading."
    exit 1
fi

run_step nccl_smoke \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/nccl_smoke.py" \
    --world-size 2 \
    --output "${RESULT_DIR}/nccl_smoke.json"

if [[ "${SUITE_MODE}" == "quick" ]]; then
    CASES=(128:1 128:4 128:16 2048:1)
    GRAPH_BUCKETS=(1 4 16)
    OUTPUT_TOKENS=32
    REPEATS=1
    WARMUP_TOKENS=4
else
    CASES=(128:1 128:2 128:4 128:8 128:16 2048:1 2048:4 2048:8)
    GRAPH_BUCKETS=(1 2 4 8 16)
    OUTPUT_TOKENS=128
    REPEATS=3
    WARMUP_TOKENS=8
fi

EXTRA_FEATURE_ARGS=()
if [[ "${RUN_DYNAMIC}" == "1" ]]; then
    EXTRA_FEATURE_ARGS+=(--run-dynamic)
fi
if [[ "${RUN_VISION}" == "1" ]]; then
    EXTRA_FEATURE_ARGS+=(--run-vision)
fi

COMMON_ARGS=(
    --model "${MODEL_PATH}"
    --backend state_aware_cuda
    --cases "${CASES[@]}"
    --output-tokens "${OUTPUT_TOKENS}"
    --repeats "${REPEATS}"
    --warmup-output-tokens "${WARMUP_TOKENS}"
    --token-budget 512
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --graph-buckets "${GRAPH_BUCKETS[@]}"
)

if [[ "${RUN_TP1_BASELINE}" == "1" ]]; then
    run_step tp1_eager_cuda \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_FEATURE_ARGS[@]}" \
        --label tp1_eager_cuda \
        --tp-size 1 \
        --mode eager \
        --output "${RESULT_DIR}/tp1_eager_cuda.json"
fi

run_step tp2_eager_cuda \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_FEATURE_ARGS[@]}" \
    --label tp2_eager_cuda \
    --tp-size 2 \
    --mode eager \
    --output "${RESULT_DIR}/tp2_eager_cuda.json"

if [[ -f "${RESULT_DIR}/tp1_eager_cuda.json" && -f "${RESULT_DIR}/tp2_eager_cuda.json" ]]; then
    run_step compare_tp1_tp2 \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_results.py" \
        --reference "${RESULT_DIR}/tp1_eager_cuda.json" \
        --candidate "${RESULT_DIR}/tp2_eager_cuda.json" \
        --label tp1_vs_tp2_greedy \
        --output "${RESULT_DIR}/compare_tp1_tp2.json"
fi

if [[ "${RUN_GRAPH}" == "1" ]]; then
    run_step tp2_graph_cuda \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_FEATURE_ARGS[@]}" \
        --label tp2_graph_cuda \
        --tp-size 2 \
        --mode graph \
        --output "${RESULT_DIR}/tp2_graph_cuda.json"

    if [[ -f "${RESULT_DIR}/tp2_eager_cuda.json" && -f "${RESULT_DIR}/tp2_graph_cuda.json" ]]; then
        run_step compare_eager_graph \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_results.py" \
            --reference "${RESULT_DIR}/tp2_eager_cuda.json" \
            --candidate "${RESULT_DIR}/tp2_graph_cuda.json" \
            --label tp2_eager_vs_graph_greedy \
            --output "${RESULT_DIR}/compare_eager_graph.json"
    fi
fi

run_step tp2_fla_smoke \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
    --model "${MODEL_PATH}" \
    --label tp2_fla_smoke \
    --tp-size 2 \
    --mode eager \
    --backend fla \
    --cases 64:1 \
    --output-tokens 8 \
    --repeats 1 \
    --warmup-output-tokens 2 \
    --token-budget 256 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --graph-buckets 1 \
    --output "${RESULT_DIR}/tp2_fla_smoke.json"

if [[ "${RUN_PREFIX}" == "1" ]]; then
    run_step prefix_tp2 \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_prefix_tp2.py" \
        --model "${MODEL_PATH}" \
        --tp-size 2 \
        --backend state_aware_cuda \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --output "${RESULT_DIR}/prefix_tp2.json"
fi

if [[ "${RUN_NSYS}" == "1" ]]; then
    run_step nsys_tp2 \
        bash "${SCRIPT_DIR}/profile_nccl.sh" \
        "${MODEL_PATH}" "${RESULT_DIR}"
fi

run_step summarize \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --result-dir "${RESULT_DIR}"

echo
echo "Report: ${RESULT_DIR}/summary.md"
echo "Full log directory: ${RESULT_DIR}"

if [[ "${FAILURES}" -ne 0 ]]; then
    echo "Suite completed with ${FAILURES} failed stage(s)."
    exit 1
fi

echo "All enabled stages passed."
