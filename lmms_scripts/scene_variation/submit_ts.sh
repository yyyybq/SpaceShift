#!/bin/bash
# Queue all open-source model evals for the scene_variation task on GPUs 2,3,4,7
# via task spooler (`ts`). One model runs at a time; each uses all 4 GPUs.
#
# Usage:
#     ./scripts/scene_variation/submit_ts.sh [model1.sh model2.sh ...]
#
# With no args: queues all default models below.
# With args:    queues only the listed scripts (paths relative to this dir or absolute).
#
# Inspect the queue:    ts
# Tail a job's output:  ts -t <job_id>
# Remove a job:         ts -r <job_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/nas2/edwin/lmms-eval/results/scene_variation_ts_logs"
mkdir -p "$LOG_DIR"

export TS_SOCKET="${TS_SOCKET:-/tmp/ts_socket_edwin_scene_variation}"
ts -S 1 >/dev/null 2>&1 || true

export TASK_NAME="${TASK_NAME:-scene_variation}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"

DEFAULT_MODELS=(
    qwen3vl_2b.sh
    qwen3vl_4b.sh
    qwen3vl_8b.sh
    internvl3p5_2b.sh
    internvl3p5_8b.sh
    llava_onevision_0p5b.sh
    llava_onevision_7b.sh
    cambrians_1p5b.sh
    cambrians_3b.sh
    cambrians_7b.sh
)

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

echo "Queuing ${#MODELS[@]} jobs on GPUs ${CUDA_VISIBLE_DEVICES} (TASK=${TASK_NAME})"
echo "Logs: ${LOG_DIR}/<model>.log"
echo "Queue socket: ${TS_SOCKET}"
echo

for entry in "${MODELS[@]}"; do
    if [[ "$entry" = /* ]]; then
        script="$entry"
    else
        script="${SCRIPT_DIR}/${entry}"
    fi
    if [[ ! -x "$script" ]]; then
        chmod +x "$script" 2>/dev/null || true
    fi
    if [[ ! -f "$script" ]]; then
        echo "Skipping missing script: $script"
        continue
    fi
    label="$(basename "$script" .sh)"
    log_file="${LOG_DIR}/${label}.log"
    job_id=$(ts -L "$label" bash -c "exec >'$log_file' 2>&1; '$script'")
    echo "  [$job_id] $label  ->  $log_file"
done

echo
echo "Done. Run 'TS_SOCKET=${TS_SOCKET} ts' to inspect the queue."
