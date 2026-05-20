#!/bin/bash
# Enqueue GPU evals (task video_consistency_thor_small) with GNU task-spooler (`ts`).
#
# Usage:
#   ts -S 2                                        # set max concurrent slots
#   bash scripts/video_consistency/submit_video_consistency_production.sh          # submit all local models
#   bash scripts/video_consistency/submit_video_consistency_production.sh --api    # also submit GPT / Gemini
#
# Environment:
#   OPENAI_API_KEY   — required for gpt5p2
#   GOOGLE_API_KEY   — required for gemini

LMMS_EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$LMMS_EVAL_ROOT/scripts/video_consistency_production"
TS_BIN="${TS_BIN:-ts}"

command -v "$TS_BIN" >/dev/null 2>&1 || {
  echo "error: ${TS_BIN} not on PATH" >&2
  exit 1
}

if [[ -n "${TS_SLOTS:-}" ]]; then
  "$TS_BIN" -S "${TS_SLOTS}"
fi

LOCAL_MODELS=(
  qwen3vl_8b.sh
  qwen3vl_4b.sh
  qwen3vl_2b.sh
  internvl3p5_8b.sh
  internvl3p5_2b.sh
  llava_onevision_7b.sh
  llava_onevision_0p5b.sh
  cambrians_7b.sh
  cambrians_3b.sh
  cambrians_1p5b.sh
)

API_MODELS=(
  gpt5p2.sh
  gemini.sh
)

for name in "${LOCAL_MODELS[@]}"; do
  "$TS_BIN" bash "$SCRIPTS/$name"
done

if [[ "${1:-}" == "--api" ]]; then
  for name in "${API_MODELS[@]}"; do
    "$TS_BIN" bash "$SCRIPTS/$name"
  done
fi
