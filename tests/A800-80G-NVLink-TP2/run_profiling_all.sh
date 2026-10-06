#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${1:-${MODEL_PATH:-}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NSYS="${RUN_NSYS:-1}"
RUN_NCU="${RUN_NCU:-1}"
PROFILE_MODE="${PROFILE_MODE:-full}"

if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
    echo "Usage: ./run_profiling_all.sh /path/to/Qwen3.5-9B [OUTPUT_DIR]"
    exit 2
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${2:-${PROFILING_DIR:-${SCRIPT_DIR}/results/profiling_${TIMESTAMP}}}"
mkdir -p "${OUTPUT_DIR}"
: > "${OUTPUT_DIR}/status.tsv"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PROFILE_MODE
FAILURES=0

run_step() {
    local name="$1"
    shift
    echo
    echo "========== ${name} =========="
    "$@" 2>&1 | tee "${OUTPUT_DIR}/${name}.log"
    local status="${PIPESTATUS[0]}"
    if [[ "${status}" -eq 0 ]]; then
        printf '%s\tPASS\n' "${name}" | tee -a "${OUTPUT_DIR}/status.tsv"
    else
        printf '%s\tFAIL_OR_BLOCKED(%s)\n' "${name}" "${status}" | tee -a "${OUTPUT_DIR}/status.tsv"
        FAILURES=$((FAILURES + 1))
    fi
    return 0
}

{
    echo "date: $(date --iso-8601=seconds 2>/dev/null || date)"
    echo "hostname: $(hostname)"
    echo "git_commit: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "profile_mode: ${PROFILE_MODE}"
} > "${OUTPUT_DIR}/metadata.txt"
nvidia-smi topo -m > "${OUTPUT_DIR}/topology.txt" 2>&1 || true

if [[ "${RUN_NSYS}" == "1" ]]; then
    run_step nsys_all bash "${SCRIPT_DIR}/profile_nsys_all.sh" "${MODEL_PATH}" "${OUTPUT_DIR}"
fi
if [[ "${RUN_NCU}" == "1" ]]; then
    run_step ncu_all bash "${SCRIPT_DIR}/profile_ncu_all.sh" "${OUTPUT_DIR}"
fi

run_step summarize_profiling \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_profiling.py" --profiling-dir "${OUTPUT_DIR}"

find "${OUTPUT_DIR}" -type f -printf '%P\t%s bytes\n' | sort > "${OUTPUT_DIR}/all_files.tsv"
echo
echo "Profiling directory: ${OUTPUT_DIR}"
echo "GUI reports:"
find "${OUTPUT_DIR}" -type f \( -name '*.nsys-rep' -o -name '*.ncu-rep' \) -print | sort
echo "Pack with: tar -czf a800_profiling_${TIMESTAMP}.tar.gz -C $(dirname "${OUTPUT_DIR}") $(basename "${OUTPUT_DIR}")"

if [[ "${FAILURES}" -ne 0 ]]; then
    echo "Profiling completed with ${FAILURES} failed or host-blocked stage(s); available reports were preserved."
    exit 1
fi
echo "All requested profiling stages passed."
