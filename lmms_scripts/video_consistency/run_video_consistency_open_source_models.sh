#!/bin/bash
# Sequential eval: all open-weight models on task ``video_consistency_thor_small``.
# Excludes API wrappers: gpt5p2.sh, gemini.sh (use submit script with --api for those).
#
# Usage:
#   cd /nas2/edwin/lmms-eval
#   bash scripts/video_consistency/run_video_consistency_open_source_models.sh
#
# Optional: stop on first failure is default (set -e). To continue on error:
#   CONTINUE_ON_ERROR=1 bash scripts/run_video_consistency_open_source_models.sh
#
# Queue with task-spooler instead (recommended on shared GPUs):
#   ts -S 2
#   for s in qwen3vl_2b.sh qwen3vl_4b.sh qwen3vl_8b.sh internvl3p5_2b.sh internvl3p5_8b.sh \
#            llava_onevision_0p5b.sh llava_onevision_7b.sh \
#            cambrians_1p5b.sh cambrians_3b.sh cambrians_7b.sh; do
#     ts bash scripts/video_consistency_production/$s
#   done
#
# Same jobs via task-spooler (needs ``ts`` on PATH):
#   TS_SLOTS=2 bash scripts/video_consistency/run_video_consistency_open_source_models.sh --ts
#
# Or: bash scripts/video_consistency/submit_video_consistency_production.sh   # identical queue order
#
# Two-machine split (74 = 4 GPU, 104 = 8 GPU), covering all 12 production scripts:
#   bash scripts/video_consistency/run_video_consistency_machine_74.sh
#   bash scripts/video_consistency/run_video_consistency_machine_104.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D="${ROOT}/scripts/video_consistency_production"

# SCRIPTS=(
#   qwen3vl_4b.sh
#   qwen3vl_8b.sh
#   internvl3p5_2b.sh
#   internvl3p5_8b.sh
#   llava_onevision_0p5b.sh
#   llava_onevision_7b.sh
#   cambrians_1p5b.sh
#   cambrians_3b.sh
#   cambrians_7b.sh
# )

SCRIPTS=(
  qwen3vl_4b.sh
  qwen3vl_8b.sh
  internvl3p5_2b.sh
  internvl3p5_8b.sh
  cambrians_3b.sh
  cambrians_7b.sh
)

use_ts=0
if [[ "${1:-}" == "--ts" ]]; then
  use_ts=1
  shift
fi

if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  set +e
fi

if [[ "$use_ts" == "1" ]]; then
  TS_BIN="${TS_BIN:-ts}"
  command -v "$TS_BIN" >/dev/null 2>&1 || {
    echo "error: ${TS_BIN} not on PATH (install task-spooler or run without --ts)" >&2
    exit 1
  }
  if [[ -n "${TS_SLOTS:-}" ]]; then
    "$TS_BIN" -S "${TS_SLOTS}"
  fi
  for s in "${SCRIPTS[@]}"; do
    echo "queue: ${s}"
    "$TS_BIN" bash "${D}/${s}"
  done
  exit 0
fi

for s in "${SCRIPTS[@]}"; do
  echo "======== ${s} ========"
  bash "${D}/${s}"
done
