#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_PARENT="${BASH_SOURCE[0]%/*}"
if [[ "${SCRIPT_PARENT}" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_PARENT=.; fi
SCRIPT_DIR="$(cd "${SCRIPT_PARENT}" && pwd)"
export DRAFT_REPO="incoai/Qwen3.8-27B-DFlash2"
export MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3.8-27B}"
export DRAFT_PATH="${DRAFT_PATH:-/workspace/models/Qwen3.8-27B-DFlash2}"
# Conservative first run for a larger 64-layer target. Full retains B=1,4,16.
if [[ "${SUITE_MODE:-smoke}" == smoke ]]; then export BATCHES="${BATCHES:-1 2}"; fi
exec "${BASH}" "${SCRIPT_DIR}/run_a800.sh" "${1:-${MODEL_PATH}}" "${2:-${DRAFT_PATH}}" "${@:3}"
