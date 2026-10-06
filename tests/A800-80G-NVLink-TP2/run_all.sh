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
RUN_FLA_BASELINE="${RUN_FLA_BASELINE:-1}"
RUN_GRAPH="${RUN_GRAPH:-1}"
RUN_LONG_CONTEXT="${RUN_LONG_CONTEXT:-1}"
RUN_PREFIX="${RUN_PREFIX:-1}"
RUN_CAPACITY="${RUN_CAPACITY:-1}"
RUN_SOAK="${RUN_SOAK:-0}"
RUN_VISION="${RUN_VISION:-1}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"
RUN_NSYS="${RUN_NSYS:-0}"
RUN_NCU="${RUN_NCU:-0}"

if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
    echo "Usage: ./run_all.sh /path/to/Qwen3.5-9B"
    echo "Model directory does not exist: ${MODEL_PATH:-<empty>}"
    exit 2
fi
if [[ "${SUITE_MODE}" != "quick" && "${SUITE_MODE}" != "full" && "${SUITE_MODE}" != "extreme" ]]; then
    echo "SUITE_MODE must be quick, full, or extreme"
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
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
    local started status elapsed
    started="$(date +%s)"
    "$@" 2>&1 | tee "${RESULT_DIR}/${name}.log"
    status="${PIPESTATUS[0]}"
    elapsed="$(( $(date +%s) - started ))"
    if [[ "${status}" -eq 0 ]]; then
        printf '%s\tPASS\t%ss\n' "${name}" "${elapsed}" | tee -a "${STATUS_FILE}"
    else
        printf '%s\tFAIL(%s)\t%ss\n' "${name}" "${status}" "${elapsed}" | tee -a "${STATUS_FILE}"
        FAILURES=$((FAILURES + 1))
    fi
    return 0
}

echo "Result directory: ${RESULT_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Suite mode: ${SUITE_MODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

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
nvidia-smi nvlink --status > "${RESULT_DIR}/nvidia_smi_nvlink.txt" 2>&1 || true
nvidia-smi -q > "${RESULT_DIR}/nvidia_smi_q.txt" 2>&1 || true
"${PYTHON_BIN}" -m pip freeze > "${RESULT_DIR}/pip_freeze.txt" 2>&1 || true

run_step preflight \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
    --model "${MODEL_PATH}" --tp-size 2 \
    --output "${RESULT_DIR}/preflight.json"
if [[ ! -f "${RESULT_DIR}/preflight.json" ]]; then
    echo "Preflight failed; no benchmark was started."
    exit 1
fi

run_step nccl_stress \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/nccl_stress.py" \
    --world-size 2 --output "${RESULT_DIR}/nccl_stress.json"

if [[ "${SUITE_MODE}" == "quick" ]]; then
    HEADLINE_CASES=(512:16 512:64)
    GRAPH_BUCKETS=(16 32 64)
    REPEATS=1
    OUTPUT_TOKENS=64
else
    HEADLINE_CASES=(512:16 512:32 512:64 512:128)
    GRAPH_BUCKETS=(8 16 32 64 128)
    REPEATS=3
    OUTPUT_TOKENS=256
fi

HEADLINE_ARGS=(
    --model "${MODEL_PATH}"
    --cases "${HEADLINE_CASES[@]}"
    --output-tokens "${OUTPUT_TOKENS}"
    --repeats "${REPEATS}"
    --warmup-output-tokens 8
    --token-budget 32768
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --graph-buckets "${GRAPH_BUCKETS[@]}"
)

if [[ "${RUN_TP1_BASELINE}" == "1" ]]; then
    run_step headline_tp1_eager_cuda \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
        "${HEADLINE_ARGS[@]}" --label headline_tp1_eager_cuda \
        --tp-size 1 --mode eager --backend state_aware_cuda \
        --output "${RESULT_DIR}/headline_tp1_eager_cuda.json"
fi

run_step headline_tp2_eager_cuda \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
    "${HEADLINE_ARGS[@]}" --label headline_tp2_eager_cuda \
    --tp-size 2 --mode eager --backend state_aware_cuda \
    --output "${RESULT_DIR}/headline_tp2_eager_cuda.json"

if [[ "${RUN_FLA_BASELINE}" == "1" ]]; then
    run_step headline_tp2_eager_fla \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
        "${HEADLINE_ARGS[@]}" --label headline_tp2_eager_fla \
        --tp-size 2 --mode eager --backend fla \
        --output "${RESULT_DIR}/headline_tp2_eager_fla.json"
fi

if [[ "${RUN_GRAPH}" == "1" ]]; then
    EXTRA_ARGS=()
    [[ "${RUN_DYNAMIC}" == "1" ]] && EXTRA_ARGS+=(--run-dynamic)
    [[ "${RUN_VISION}" == "1" ]] && EXTRA_ARGS+=(--run-vision)
    run_step headline_tp2_graph_cuda \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
        "${HEADLINE_ARGS[@]}" "${EXTRA_ARGS[@]}" \
        --label headline_tp2_graph_cuda --tp-size 2 --mode graph \
        --backend state_aware_cuda \
        --output "${RESULT_DIR}/headline_tp2_graph_cuda.json"
fi

if [[ "${RUN_LONG_CONTEXT}" == "1" && "${SUITE_MODE}" != "quick" ]]; then
    run_step long_context_tp2_eager \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
        --model "${MODEL_PATH}" --label long_context_tp2_eager \
        --tp-size 2 --mode eager --backend state_aware_cuda \
        --cases 16384:4 32768:2 65536:1 \
        --output-tokens 128 --repeats 2 --warmup-output-tokens 8 \
        --token-budget 32768 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --graph-buckets 1 2 4 --output "${RESULT_DIR}/long_context_tp2_eager.json"
fi

if [[ "${RUN_PREFIX}" == "1" && "${SUITE_MODE}" != "quick" ]]; then
    run_step prefix_64k_48k_tp2 \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_prefix_tp2.py" \
        --model "${MODEL_PATH}" --tp-size 2 --backend state_aware_cuda \
        --checkpoint-tokens 49152 --suffix-tokens 16384 --output-tokens 128 \
        --hot-repeats 3 --cache-capacity-mib 4096 \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --output "${RESULT_DIR}/prefix_64k_48k_tp2.json"
fi

if [[ "${RUN_SOAK}" == "1" && "${SUITE_MODE}" != "quick" ]]; then
    SOAK_BATCH=64
    SOAK_ROUNDS=20
    [[ "${SUITE_MODE}" == "extreme" ]] && SOAK_BATCH=128 && SOAK_ROUNDS=50
    run_step soak_tp2_graph \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_soak.py" \
        --model "${MODEL_PATH}" --tp-size 2 --mode graph --backend state_aware_cuda \
        --batch-size "${SOAK_BATCH}" --rounds "${SOAK_ROUNDS}" \
        --prompt-lengths 128,512,2048,8192,16384 --output-tokens 256 \
        --token-budget 32768 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --graph-buckets 8 16 32 64 128 --output "${RESULT_DIR}/soak_tp2_graph.json"
fi

if [[ "${RUN_CAPACITY}" == "1" && "${SUITE_MODE}" != "quick" ]]; then
    CAPACITY_PROFILE=full
    [[ "${SUITE_MODE}" == "extreme" ]] && CAPACITY_PROFILE=extreme
    run_step capacity_sweep \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_capacity_sweep.py" \
        --model "${MODEL_PATH}" --tp-size 2 --profile "${CAPACITY_PROFILE}" \
        --python-bin "${PYTHON_BIN}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --output-dir "${RESULT_DIR}/capacity"
fi

if [[ "${RUN_NSYS}" == "1" || "${RUN_NCU}" == "1" ]]; then
    PROFILING_MODE=core
    [[ "${SUITE_MODE}" == "extreme" ]] && PROFILING_MODE=full
    run_step profiling_all \
        env PROFILE_MODE="${PROFILING_MODE}" RUN_NSYS="${RUN_NSYS}" RUN_NCU="${RUN_NCU}" \
        bash "${SCRIPT_DIR}/run_profiling_all.sh" "${MODEL_PATH}" "${RESULT_DIR}/profiling"
fi

run_step summarize \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" --result-dir "${RESULT_DIR}"

echo
echo "Report: ${RESULT_DIR}/summary.md"
echo "Archive with: tar -czf a800_nvlink_tp2_${TIMESTAMP}.tar.gz -C $(dirname "${RESULT_DIR}") $(basename "${RESULT_DIR}")"
if [[ "${FAILURES}" -ne 0 ]]; then
    echo "Suite completed with ${FAILURES} failed stage(s); successful stages were preserved."
    exit 1
fi
echo "All enabled stages passed."
