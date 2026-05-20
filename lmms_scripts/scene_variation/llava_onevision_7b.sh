#!/bin/bash
MODEL="${MODEL:-lmms-lab/llava-onevision-qwen2-7b-ov}"
TASK_NAME="${TASK_NAME:-scene_variation}"
OUTPUT_DIR="/nas2/edwin/lmms-eval/results/llava_onevision_7b_${TASK_NAME}"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3,4,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"

PYTHON="/nas2/edwin/miniconda/envs/llava_onevision/bin/python"
export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1
export HF_HOME="/nas2/edwin/lmms-eval/.cache/huggingface"

LMMS_EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LMMS_EVAL_ROOT"
mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=${MODEL},conv_template=qwen_1_5,device_map=cuda:0"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000 + ($$ % 1000)))}"
export LMMS_EVAL_LAUNCHER="accelerate"

$PYTHON -m accelerate.commands.launch --num_processes="${NUM_PROCESSES}" --main_process_port="${MASTER_PORT}" -m lmms_eval \
    --model llava_onevision --model_args "${MODEL_ARGS}" \
    --tasks "${TASK_NAME}" --batch_size "${BATCH_SIZE}" \
    --log_samples --log_samples_suffix "llava_onevision_7b_${TASK_NAME}" \
    --output_path "${OUTPUT_DIR}"
