#!/bin/bash
# Run two task-spooler queues in parallel: each model gets 2 GPUs (NUM_PROCESSES=2),
# so two models execute simultaneously. Models are split into "small" (queue A on GPUs 2,3)
# and "large" (queue B on GPUs 4,7) to balance per-job runtime.
#
# Usage: ./submit_ts_parallel.sh
# Inspect:  ./ts_status_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/nas2/edwin/lmms-eval/results/scene_variation_ts_logs"
mkdir -p "$LOG_DIR"

SOCK_A="/tmp/ts_socket_edwin_scene_variation_A"
SOCK_B="/tmp/ts_socket_edwin_scene_variation_B"

GROUP_A=(qwen3vl_4b.sh internvl3p5_2b.sh cambrians_1p5b.sh cambrians_3b.sh)
GROUP_B=(qwen3vl_8b.sh internvl3p5_8b.sh cambrians_7b.sh)

queue() {
    local sock="$1"; local gpus="$2"; local base_port="$3"; shift 3
    export TS_SOCKET="$sock"
    ts -S 1 >/dev/null 2>&1 || true
    local i=0
    for entry in "$@"; do
        local script="${SCRIPT_DIR}/${entry}"
        chmod +x "$script" 2>/dev/null || true
        local label="$(basename "$script" .sh)"
        local log="${LOG_DIR}/${label}.log"
        local port=$((base_port + i))
        local job_id
        job_id=$(ts -L "$label" bash -c "
            export TASK_NAME=scene_variation
            export CUDA_VISIBLE_DEVICES=${gpus}
            export NUM_PROCESSES=2
            export BATCH_SIZE=4
            export MASTER_PORT=${port}
            exec >'${log}' 2>&1
            '${script}'
        ")
        echo "  [${sock##*_}/$job_id] $label  GPUs=${gpus}  port=${port}  ->  $log"
        i=$((i+1))
    done
}

echo "Queue A on GPUs 2,3 (small/medium models):"
queue "$SOCK_A" "2,3" 29500 "${GROUP_A[@]}"
echo
echo "Queue B on GPUs 4,7 (large models):"
queue "$SOCK_B" "4,7" 29600 "${GROUP_B[@]}"

echo
echo "Inspect: TS_SOCKET=$SOCK_A ts; TS_SOCKET=$SOCK_B ts"
