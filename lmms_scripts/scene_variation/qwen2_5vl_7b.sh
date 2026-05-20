#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
TASK_NAME="${TASK_NAME:-scene_variation}"
OUTPUT_DIR="${OUTPUT_DIR:-/nas2/edwin/lmms-eval/results/qwen2_5_vl_7b_scene_variation}"
GPUS="${CUDA_VISIBLE_DEVICES:-4,7}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_PIXELS="${MIN_PIXELS:-784}"
MAX_PIXELS="${MAX_PIXELS:-50176}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-8}"
FPS="${FPS:-1}"

PYTHON="/nas2/edwin/miniconda/envs/qwen2_5_vl_eval/bin/python"
CACHE_HOME="${LMMS_EVAL_HF_HOME:-/nas2/edwin/lmms-eval/.cache/huggingface}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export HF_HOME="${CACHE_HOME}"
export HF_HUB_CACHE="${CACHE_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export LMMS_EVAL_LAUNCHER="accelerate"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29700 + RANDOM % 1000))}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

source "${SCRIPT_DIR}/scripts/full_benchmark/_hf_local_model.sh"
MODEL_PATH="$(resolve_local_hf_model_path "${MODEL}")" || {
    echo "error: local model snapshot not found for ${MODEL} under ${HF_HUB_CACHE}" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MODEL_ARGS="pretrained=${MODEL_PATH},min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},max_num_frames=${MAX_NUM_FRAMES},fps=${FPS},attn_implementation=${ATTN_IMPL},interleave_visuals=False"

$PYTHON -m accelerate.commands.launch --num_processes="${NUM_PROCESSES}" --main_process_port="${MASTER_PORT}" -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --log_samples \
    --log_samples_suffix qwen2_5_vl_7b_scene_variation \
    --output_path "${OUTPUT_DIR}"
