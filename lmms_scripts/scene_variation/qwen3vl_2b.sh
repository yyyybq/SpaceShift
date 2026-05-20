#!/bin/bash
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
TASK_NAME="${TASK_NAME:-scene_variation}"
OUTPUT_DIR="/nas2/edwin/lmms-eval/results/qwen3vl_2b_${TASK_NAME}"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3,4,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-40}"
MIN_PIXELS="${MIN_PIXELS:-784}"
MAX_PIXELS="${MAX_PIXELS:-50176}"

PYTHON="/nas2/edwin/miniconda/envs/lmms_eval/bin/python"
export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1
export HF_HOME="/nas2/edwin/lmms-eval/.cache/huggingface"

LMMS_EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LMMS_EVAL_ROOT"
mkdir -p "$OUTPUT_DIR"

export MODEL
if [[ "${SKIP_HF_PREDOWNLOAD:-0}" != "1" ]]; then
  HF_HUB_OFFLINE=0 $PYTHON -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['MODEL'])"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

MODEL_ARGS="pretrained=${MODEL},min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},fps=1,device_map=auto"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000 + ($$ % 1000)))}"
export LMMS_EVAL_LAUNCHER="accelerate"

$PYTHON -m accelerate.commands.launch --num_processes="${NUM_PROCESSES}" --main_process_port="${MASTER_PORT}" -m lmms_eval \
    --model qwen3_vl --model_args "${MODEL_ARGS}" \
    --tasks "${TASK_NAME}" --batch_size "${BATCH_SIZE}" \
    --log_samples --log_samples_suffix "qwen3vl_2b_${TASK_NAME}" \
    --output_path "${OUTPUT_DIR}"
