#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${1:?Usage: profile_ncu_all.sh OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NCU_SET="${NCU_SET:-full}"

if ! command -v ncu >/dev/null 2>&1; then
    echo "Nsight Compute CLI (ncu) is not installed or not on PATH."
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

NCU_DIR="${OUTPUT_DIR}/ncu"
mkdir -p "${NCU_DIR}"
: > "${NCU_DIR}/status.tsv"
ncu --version > "${NCU_DIR}/version.txt" 2>&1 || true
FAILURES=0
PERMISSION_DENIED=0

profile_kernel() {
    local kernel="$1"
    local batch="$2"
    local symbol name report status
    if [[ "${kernel}" == "recurrent" ]]; then
        symbol="state_aware_gdn_bf16_kernel"
    else
        symbol="state_aware_causal_conv1d_bf16_kernel"
    fi
    name="${kernel}_b${batch}"
    report="${NCU_DIR}/${name}.ncu-rep"

    echo
    echo "========== ncu: ${name} =========="
    ncu \
        --force-overwrite \
        --target-processes all \
        --replay-mode kernel \
        --set "${NCU_SET}" \
        --clock-control none \
        --cache-control none \
        --kernel-name-base function \
        --kernel-name "regex:${symbol}" \
        --launch-skip 5 \
        --launch-count 1 \
        --export "${NCU_DIR}/${name}" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/ncu_kernel_driver.py" \
        --kernel "${kernel}" --batch-size "${batch}" \
        --warmup 5 --iterations 5 \
        --output "${NCU_DIR}/${name}_events.json" \
        2>&1 | tee "${NCU_DIR}/${name}.console.log"
    status="${PIPESTATUS[0]}"

    if grep -Eqi 'ERR_NVGPUCTRPERM|permission to access NVIDIA GPU Performance Counters' "${NCU_DIR}/${name}.console.log"; then
        printf '%s\tBLOCKED_GPU_COUNTER_PERMISSION\n' "${name}" | tee -a "${NCU_DIR}/status.tsv"
        printf '%s\n' \
            "The rental host blocks NVIDIA GPU performance counters." \
            "Nsight Systems and CUDA Event results remain valid; NCU full metrics require host configuration." \
            > "${NCU_DIR}/PERMISSION_BLOCKED.txt"
        PERMISSION_DENIED=1
        return 0
    fi

    if [[ "${status}" -ne 0 || ! -f "${report}" ]]; then
        printf '%s\tFAIL(%s)\n' "${name}" "${status}" | tee -a "${NCU_DIR}/status.tsv"
        FAILURES=$((FAILURES + 1))
        return 0
    fi

    printf '%s\tPASS\n' "${name}" | tee -a "${NCU_DIR}/status.tsv"
    ncu --import "${report}" --page details --csv \
        > "${NCU_DIR}/${name}_details.csv" \
        2> "${NCU_DIR}/${name}_import.log" || true
}

for kernel in recurrent conv; do
    for batch in 16 64 128; do
        if [[ "${PERMISSION_DENIED}" -eq 1 ]]; then
            printf '%s_b%s\tSKIPPED_AFTER_PERMISSION_FAILURE\n' "${kernel}" "${batch}" \
                | tee -a "${NCU_DIR}/status.tsv"
            continue
        fi
        profile_kernel "${kernel}" "${batch}"
    done
done

find "${NCU_DIR}" -maxdepth 1 -type f -printf '%f\t%k KiB\n' | sort > "${NCU_DIR}/files.tsv"
echo "Nsight Compute reports: ${NCU_DIR}"
if [[ "${PERMISSION_DENIED}" -eq 1 ]]; then
    exit 2
fi
exit "${FAILURES}"
