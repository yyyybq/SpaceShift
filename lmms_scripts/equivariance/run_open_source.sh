#!/usr/bin/env bash
# Run all open-source VLMs on the equivariance task.
#
# Reuses the per-model launchers in scripts/full_benchmark/*.sh, overriding
# TASK_NAME and OUTPUT_DIR so results land under
#   results/equivariance_analysis/model_runs/<model>_equivariance/
#
# Task YAML:   lmms_eval/tasks/sceneshift/equivariance.yaml
# Source data: /nas2/edwin/lmms-eval/data/equivariance.jsonl  (4395 rows, image-only)
#
# Usage:
#     bash run_open_source.sh                          # all models, sequentially
#     bash run_open_source.sh qwen3vl_8b               # single model (basename of full_benchmark/*.sh)
#     bash run_open_source.sh qwen3vl_2b qwen3vl_4b    # specific subset (any number of args)
#     SKIP_EXISTING=1 bash run_open_source.sh          # skip models that already wrote *_results.json
#     DRY_RUN=1       bash run_open_source.sh          # print plan only
#
# Env (optional):
#     GPUS            CUDA_VISIBLE_DEVICES forwarded to each launcher (default 0,1,2,3,4,5,6,7)
#     NUM_PROCESSES   accelerate processes per launcher (default 8)
#     BATCH_SIZE      lmms-eval batch size (default 1)
#     RESULTS_ROOT    parent of per-model output dirs
#                     (default <repo>/results/equivariance_analysis/model_runs)
#     SKIP_EXISTING   skip launcher if its OUTPUT_DIR already has *_results.json (default 1)
#     DRY_RUN         print queue and exit (default 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FB_DIR="${REPO_ROOT}/scripts/full_benchmark"

TASK_NAME="${TASK_NAME:-equivariance}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/equivariance_analysis/model_runs}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Model launcher basenames in scripts/full_benchmark/, run in size order.
ORDER=(
    qwen3vl_2b
    qwen3vl_4b
    qwen3vl_8b
    qwen2_5vl_3b
    qwen2_5vl_7b
    cambrians_1p5b
    cambrians_3b
    cambrians_7b
    internvl3p5_2b
    internvl3p5_8b
)

# Map launcher basename -> output subdir name (uses canonical model tags).
output_dir_for() {
    local name="$1"
    case "${name}" in
        llava_onevision_0.5b) echo "${RESULTS_ROOT}/llava_onevision_0p5b_${TASK_NAME}" ;;
        qwen2_5vl_3b)         echo "${RESULTS_ROOT}/qwen2_5_vl_3b_${TASK_NAME}" ;;
        qwen2_5vl_7b)         echo "${RESULTS_ROOT}/qwen2_5_vl_7b_${TASK_NAME}" ;;
        *)                    echo "${RESULTS_ROOT}/${name}_${TASK_NAME}" ;;
    esac
}

has_results() {
    local dir="$1"
    [[ -d "${dir}" ]] && find "${dir}" -mindepth 2 -maxdepth 2 -name '*_results.json' -print -quit | grep -q .
}

if [[ "$#" -gt 0 && "${1}" != "all" ]]; then
    ORDER=("$@")
fi

failed=()
for name in "${ORDER[@]}"; do
    script="${FB_DIR}/${name}.sh"
    if [[ ! -f "${script}" ]]; then
        echo "skip: ${name} (no launcher at ${script})" >&2
        continue
    fi

    out_dir="$(output_dir_for "${name}")"

    if [[ "${SKIP_EXISTING}" == "1" ]] && has_results "${out_dir}"; then
        echo "skip: ${name} (results already at ${out_dir})"
        continue
    fi

    echo "run: ${name} -> ${out_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi

    if ! env \
        TASK_NAME="${TASK_NAME}" \
        OUTPUT_DIR="${out_dir}" \
        GPUS="${GPUS}" \
        NUM_PROCESSES="${NUM_PROCESSES}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        bash "${script}"; then
        echo "fail: ${name} (continuing with remaining models)" >&2
        failed+=("${name}")
    fi
done

if [[ "${#failed[@]}" -gt 0 ]]; then
    echo "done with failures: ${failed[*]}" >&2
    exit 1
fi
echo "done: all open-source launchers completed"
