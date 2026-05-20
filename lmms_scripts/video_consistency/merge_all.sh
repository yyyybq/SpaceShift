#!/bin/bash
# Merge sceneshift_0304 + interiorgs samples for each model that has both,
# then run aggregate_results to produce combined metrics.
#
# Usage: bash scripts/video_consistency/merge_all.sh

set -euo pipefail

RESULTS_DIR="/nas2/edwin/lmms-eval/results"

# Models that have both a sceneshift and interiorgs result directory.
# Format: "base_dir interiorgs_dir"
PAIRS=(
    "gpt5p2           gpt5p2_interiorgs"
    "qwen3vl_4b       qwen3vl_4b_interiorgs"
    "qwen3vl_8b       qwen3vl_8b_interiorgs"
)

find_jsonl() {
    find "$1" -name "*.jsonl" -type f | head -1
}

for pair in "${PAIRS[@]}"; do
    read -r base interiorgs <<< "$pair"

    base_dir="${RESULTS_DIR}/${base}"
    interiorgs_dir="${RESULTS_DIR}/${interiorgs}"
    merged_dir="${RESULTS_DIR}/${base}_merged"

    file1=$(find_jsonl "$interiorgs_dir")
    file2=$(find_jsonl "$base_dir")

    if [[ -z "$file1" ]]; then
        echo "SKIP ${base}: no JSONL in ${interiorgs_dir}"
        continue
    fi
    if [[ -z "$file2" ]]; then
        echo "SKIP ${base}: no JSONL in ${base_dir}"
        continue
    fi

    echo "========================================"
    echo "Merging: ${base}"
    echo "  file1 (interiorgs): ${file1}"
    echo "  file2 (sceneshift): ${file2}"
    echo "  output:             ${merged_dir}/"
    echo "========================================"

    mkdir -p "${merged_dir}"

    python -m lmms_eval.aggregate_results \
        --file1 "$file1" \
        --file2 "$file2" \
        --merged_out "${merged_dir}/merged_samples.jsonl" \
        --results_out "${merged_dir}/results.json"

    echo ""
done

echo "All done."
