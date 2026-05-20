#!/bin/bash
# Show status of both parallel queues + current progress per running job.
set -euo pipefail
LOG_DIR="/nas2/edwin/lmms-eval/results/scene_variation_ts_logs"

show() {
    local label="$1"; local sock="$2"
    export TS_SOCKET="$sock"
    local q running queued finished
    q="$(ts 2>&1 | tail -n +2)"
    running=$(echo "$q" | awk '$2 == "running"' | wc -l)
    queued=$(echo "$q" | awk '$2 == "queued"' | wc -l)
    finished=$(echo "$q" | awk '$2 == "finished"' | wc -l)
    echo "[$label] running=$running queued=$queued finished=$finished"
    local cur
    cur=$(echo "$q" | awk '$2 == "running" {for (i=6; i<=NF; i++) if ($i ~ /^\[/) {gsub(/[\[\]]/, "", $i); print $i; exit}}')
    if [[ -n "${cur:-}" ]]; then
        local log="$LOG_DIR/${cur}.log"
        if [[ -f "$log" ]]; then
            local pg
            pg=$(tail -c 2000 "$log" | tr '\r' '\n' | grep -oE 'Model Responding:[^]]+\][^,]*' | tail -1)
            echo "  running: $cur  $pg"
        fi
    fi
    if [[ $finished -gt 0 ]]; then
        echo "  finished:"
        echo "$q" | awk '$2 == "finished" {for (i=6; i<=NF; i++) if ($i ~ /^\[/) {gsub(/[\[\]]/, "", $i); printf "    - %s\n", $i; break}}'
    fi
}

echo "[$(date '+%H:%M:%S')]"
show "A (GPUs 2,3)" /tmp/ts_socket_edwin_scene_variation_A
show "B (GPUs 4,7)" /tmp/ts_socket_edwin_scene_variation_B
