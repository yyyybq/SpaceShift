#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"

MODEL_VERSION="${MODEL_VERSION:-gpt-5.2}"
TASK_NAME="${TASK_NAME:-scene_variation}"
OUTPUT_DIR="${OUTPUT_DIR:-/nas2/edwin/lmms-eval/results/gpt5p2_scene_variation}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_CONCURRENT="${NUM_CONCURRENT:-32}"

PYTHON="/nas2/edwin/miniconda/envs/lmms_eval/bin/python"

export HF_HOME="${HF_HOME:-/nas2/edwin/lmms-eval/.cache/huggingface}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="model_version=${MODEL_VERSION},num_concurrent=${NUM_CONCURRENT},max_retries=5,timeout=120,continual_mode=True,response_persistent_folder=${OUTPUT_DIR}/cache"

$PYTHON -m lmms_eval \
    --model openai \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --log_samples \
    --log_samples_suffix gpt5p2_scene_variation \
    --output_path "${OUTPUT_DIR}"
