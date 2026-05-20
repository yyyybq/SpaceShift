#!/bin/bash
# Print compact queue status: running/queued/finished counts + progress of running job.
# Usage: ./ts_status.sh
set -euo pipefail
export TS_SOCKET="${TS_SOCKET:-/tmp/ts_socket_edwin_scene_variation}"
LOG_DIR="/nas2/edwin/lmms-eval/results/scene_variation_ts_logs"

queue="$(ts 2>&1 | tail -n +2)"

running=$(echo "$queue" | awk '$2 == "running"' | wc -l)
queued=$(echo "$queue" | awk '$2 == "queued"' | wc -l)
finished=$(echo "$queue" | awk '$2 == "finished"' | wc -l)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] running=$running queued=$queued finished=$finished"

if [[ $running -gt 0 ]]; then
    label=$(echo "$queue" | awk '$2 == "running" {for (i=6; i<=NF; i++) if ($i ~ /^\[/) {gsub(/[\[\]]/, "", $i); print $i; exit}}')
    log="$LOG_DIR/${label}.log"
    if [[ -f "$log" ]]; then
        progress=$(tail -c 2000 "$log" | tr '\r' '\n' | grep -oE 'Model Responding:\s*[0-9]+%[^]]*\][^ ]*' | tail -1)
        echo "  running: $label  $progress"
    fi
fi

if [[ $finished -gt 0 ]]; then
    echo "  finished:"
    echo "$queue" | awk '$2 == "finished" {for (i=6; i<=NF; i++) if ($i ~ /^\[/) {gsub(/[\[\]]/, "", $i); printf "    %s (exit=%s)\n", $i, $4; break}}'
fi
