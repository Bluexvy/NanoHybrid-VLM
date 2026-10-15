#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PARENT="${BASH_SOURCE[0]%/*}"
if [[ "${SCRIPT_PARENT}" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_PARENT=.; fi
SCRIPT_DIR="$(cd "${SCRIPT_PARENT}" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${1:-${MODEL_PATH:-}}"
DRAFT_PATH="${2:-${DRAFT_PATH:-}}"
OUTPUT_DIR="${3:-${OUTPUT_DIR:-${REPO_ROOT}/artifacts/dflash-a800/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE_MODE="${SUITE_MODE:-smoke}"
BACKEND="${BACKEND:-auto}"
GDN_DECODE_BACKEND="${GDN_DECODE_BACKEND:-state_aware_cuda}"
DRY_RUN="${DRY_RUN:-0}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
DOWNLOAD_DRAFT="${DOWNLOAD_DRAFT:-0}"
DRAFT_REPO="${DRAFT_REPO:-z-lab/Qwen3.5-9B-DFlash}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

usage() {
    cat <<'EOF'
Usage: bash tests/dflash/run_a800.sh TARGET_DIR DFLASH_DIR [OUTPUT_DIR]

Environment:
  SUITE_MODE=smoke|full    smoke: B=1,4, drafts=3; full: B=1,4,16, drafts=3,7
  DOWNLOAD_DRAFT=1        download DRAFT_REPO first (default: z-lab/Qwen3.5-9B-DFlash)
  DRAFT_REPO=...          use incoai/Qwen3.8-27B-DFlash2 for Qwen3.8
  BACKEND=auto|cuda|reference
  GDN_DECODE_BACKEND=state_aware_cuda|fla|state_aware_triton  default: original custom CUDA Decode
  RUN_UNIT_TESTS=0        skip the small CPU/GPU regression tests
  RUN_BINDING_TESTS=0     skip C++ binding gate (default: enabled unless reference)
  BATCHES="1 4" DRAFTS="3 7" REPEATS=3 OUTPUT_TOKENS=64
  PROMPTS_JSONL=/path/to/prompts.jsonl   optional {id,task,prompt} rows
  THINKING=1             enable Qwen thinking in benchmark prompts
  ALLOW_TOKEN_MISMATCH=1  continue benchmark timing on token differences; consistency remains FAIL
  DRY_RUN=1              print commands only; no GPU/model/directory required

Activate the existing CUDA environment first. This script does not install packages.
Do not use torchrun: the engine starts its own TP=2 worker.
EOF
}

if [[ "${MODEL_PATH}" == "--help" || "${MODEL_PATH}" == "-h" ]]; then usage; exit 0; fi
if [[ -z "${MODEL_PATH}" || -z "${DRAFT_PATH}" ]]; then usage; exit 2; fi
if [[ "${SUITE_MODE}" != smoke && "${SUITE_MODE}" != full ]]; then echo "Invalid SUITE_MODE" >&2; exit 2; fi
if [[ "${BACKEND}" != auto && "${BACKEND}" != cuda && "${BACKEND}" != reference ]]; then echo "Invalid BACKEND" >&2; exit 2; fi
if [[ "${GDN_DECODE_BACKEND}" != state_aware_cuda && "${GDN_DECODE_BACKEND}" != fla && "${GDN_DECODE_BACKEND}" != state_aware_triton ]]; then echo "Invalid GDN_DECODE_BACKEND" >&2; exit 2; fi
if [[ "${SUITE_MODE}" == full ]]; then
    DEFAULT_BATCHES="1 4 16"; DEFAULT_DRAFTS="3 7"; DEFAULT_REPEATS=5; DEFAULT_TOKENS=128
else
    DEFAULT_BATCHES="1 4"; DEFAULT_DRAFTS="3"; DEFAULT_REPEATS=3; DEFAULT_TOKENS=64
fi
read -r -a BATCH_ARRAY <<< "${BATCHES:-${DEFAULT_BATCHES}}"
read -r -a DRAFT_ARRAY <<< "${DRAFTS:-${DEFAULT_DRAFTS}}"
if [[ "${BACKEND}" == reference && "${GDN_DECODE_BACKEND}" != state_aware_cuda ]]; then DEFAULT_BINDING=0; else DEFAULT_BINDING=1; fi
RUN_BINDING_TESTS="${RUN_BINDING_TESTS:-${DEFAULT_BINDING}}"

run() {
    printf '+ '; printf '%q ' "$@"; printf '\n'
    if [[ "${DRY_RUN}" != 1 ]]; then "$@"; fi
}

cd "${REPO_ROOT}"
if [[ "${DRY_RUN}" != 1 ]]; then
    mkdir -p "${OUTPUT_DIR}"
    exec > >(tee -a "${OUTPUT_DIR}/run.log") 2>&1
fi
trap 'status=$?; if (( status != 0 )); then printf "FAILED (exit %s). Logs: %s\n" "$status" "$OUTPUT_DIR" >&2; fi' EXIT

echo "DFlash TP=2 ${SUITE_MODE}; GDN Decode=${GDN_DECODE_BACKEND}; verify/replay=${BACKEND}; visible GPUs=${CUDA_VISIBLE_DEVICES}; results=${OUTPUT_DIR}"
if [[ "${DOWNLOAD_DRAFT}" == 1 ]]; then
    run env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "${PYTHON_BIN}" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' "${DRAFT_REPO}" "${DRAFT_PATH}"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
run "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight_a800.py" --model "${MODEL_PATH}" --draft "${DRAFT_PATH}" --tp 2 --backend "${BACKEND}" --gdn-decode-backend "${GDN_DECODE_BACKEND}" --report "${OUTPUT_DIR}/environment.json"
if command -v nvidia-smi >/dev/null 2>&1; then
    run nvidia-smi
    run nvidia-smi topo -m
fi
if [[ "${RUN_UNIT_TESTS}" == 1 ]]; then
    run "${PYTHON_BIN}" -m unittest discover -s tests/mtp -p 'test_*.py'
    # Binding test has its own gate below; exclude it from this small regression suite.
    run "${PYTHON_BIN}" -m unittest discover -s tests/dflash -p 'test_dflash_*.py'
    run "${PYTHON_BIN}" -m unittest discover -s tests/dflash -p 'test_benchmark_harness.py'
fi
if [[ "${RUN_BINDING_TESTS}" == 1 ]]; then
    if [[ "${GDN_DECODE_BACKEND}" == state_aware_cuda ]]; then
        run "${PYTHON_BIN}" tests/dflash/test_state_aware_binding.py
    fi
    run "${PYTHON_BIN}" tests/mtp/test_gdn_replay_cuda.py
    run "${PYTHON_BIN}" tests/dflash/test_gdn_binding.py
fi
ARGS=("${PYTHON_BIN}" "${SCRIPT_DIR}/run_suite.py" --model "${MODEL_PATH}" --draft "${DRAFT_PATH}" --output "${OUTPUT_DIR}/cases"
    --tp 2 --backend "${BACKEND}" --gdn-decode-backend "${GDN_DECODE_BACKEND}" --batches "${BATCH_ARRAY[@]}" --drafts "${DRAFT_ARRAY[@]}"
    --repeats "${REPEATS:-${DEFAULT_REPEATS}}" --warmups "${WARMUPS:-1}" --output-tokens "${OUTPUT_TOKENS:-${DEFAULT_TOKENS}}"
    --max-model-len "${MAX_MODEL_LEN:-1024}" --token-budget "${TOKEN_BUDGET:-256}"
    --memory-fraction "${GPU_MEMORY_UTILIZATION:-0.82}" --case-timeout "${CASE_TIMEOUT:-1200}")
if [[ -n "${PROMPTS_JSONL:-}" ]]; then ARGS+=(--prompts-jsonl "${PROMPTS_JSONL}"); fi
if [[ "${THINKING:-0}" == 1 ]]; then ARGS+=(--thinking); fi
if [[ "${ALLOW_TOKEN_MISMATCH:-0}" == 1 ]]; then ARGS+=(--allow-token-mismatch); fi
run "${ARGS[@]}"
echo "Finished. Read ${OUTPUT_DIR}/cases/summary.md and summary.json; failures are never converted to PASS."
