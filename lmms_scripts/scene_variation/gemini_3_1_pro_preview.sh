#!/usr/bin/env bash
set -euo pipefail

: "${GOOGLE_API_KEY:?GOOGLE_API_KEY must be set}"

MODEL_VERSION="${MODEL_VERSION:-gemini-3.1-pro-preview}"
TASK_NAME="${TASK_NAME:-scene_variation}"
OUTPUT_DIR="${OUTPUT_DIR:-/nas2/edwin/lmms-eval/results/gemini_3_1_pro_preview_scene_variation}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TIMEOUT="${TIMEOUT:-120}"
NUM_CONCURRENT="${NUM_CONCURRENT:-16}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_S="${RETRY_BACKOFF_S:-1.0}"
CACHE_WRITE_EVERY_N="${CACHE_WRITE_EVERY_N:-20}"
THINKING_BUDGET="${THINKING_BUDGET:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=${MAX_NEW_TOKENS},temperature=0,top_p=1.0,num_beams=1,do_sample=false}"

PYTHON="/nas2/edwin/miniconda/envs/lmms_eval/bin/python"

export HF_HOME="${HF_HOME:-/nas2/edwin/lmms-eval/.cache/huggingface}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="model_version=${MODEL_VERSION},timeout=${TIMEOUT},continual_mode=True,response_persistent_folder=${OUTPUT_DIR}/cache,num_concurrent=${NUM_CONCURRENT},max_retries=${MAX_RETRIES},retry_backoff_s=${RETRY_BACKOFF_S},cache_write_every_n=${CACHE_WRITE_EVERY_N},thinking_budget=${THINKING_BUDGET}"

$PYTHON -m lmms_eval \
    --model gemini_api \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK_NAME}" \
    --gen_kwargs "${GEN_KWARGS}" \
    --batch_size "${BATCH_SIZE}" \
    --log_samples \
    --log_samples_suffix gemini_3_1_pro_preview_scene_variation \
    --output_path "${OUTPUT_DIR}"
