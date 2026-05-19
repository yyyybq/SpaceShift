#!/usr/bin/env bash
# Run equivariance evals.
#
# Usage:
#     bash submit.sh             # default: 10-model open-source lineup (see SELECTED)
#     bash submit.sh selected    # same as default
#     bash submit.sh oss         # all 12 open-source models in run_open_source.sh
#     bash submit.sh qwen        # only qwen3vl_8b (the original single-model launcher)
#     bash submit.sh gemini      # only gemini_3_1_pro_preview
#     bash submit.sh all         # selected + gemini
#
# Env (optional): GPUS, NUM_PROCESSES, BATCH_SIZE, SKIP_EXISTING, DRY_RUN,
#                 GOOGLE_API_KEY (required for gemini).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-selected}"

# 8-GPU lineup: Qwen3-VL (3) + Qwen2.5-VL (2) + InternVL3.5 (2, no third cached) + Cambrian-S (3) = 10.
SELECTED=(
    qwen3vl_2b
    qwen3vl_4b
    qwen3vl_8b
    qwen2_5vl_3b
    qwen2_5vl_7b
    internvl3p5_2b
    internvl3p5_8b
    cambrians_1p5b
    cambrians_3b
    cambrians_7b
)

run_qwen() {
    bash "${SCRIPT_DIR}/qwen3vl_8b.sh"
}

run_gemini() {
    bash "${SCRIPT_DIR}/gemini_3_1_pro_preview.sh"
}

run_oss() {
    bash "${SCRIPT_DIR}/run_open_source.sh"
}

run_selected() {
    bash "${SCRIPT_DIR}/run_open_source.sh" "${SELECTED[@]}"
}

case "$TARGET" in
    selected) run_selected ;;
    oss)      run_oss ;;
    qwen)     run_qwen ;;
    gemini)   run_gemini ;;
    all)      run_selected; run_gemini ;;
    *)        echo "unknown target: $TARGET (expected: selected, oss, qwen, gemini, all)" >&2; exit 2 ;;
esac
