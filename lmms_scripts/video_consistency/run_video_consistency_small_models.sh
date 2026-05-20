#!/bin/bash
# Run sub-7B checkpoints on video_consistency_thor_small (Qwen 2B/4B, InternVL 2B,
# LLaVA 0.5B, Cambrian 1.5B/3B).  Larger scripts stay in video_consistency_production/.
#
# Usage (sequential):
#   cd /nas2/edwin/lmms-eval && bash scripts/video_consistency/run_video_consistency_small_models.sh
#
# Usage (task-spooler, one job per model):
#   cd /nas2/edwin/lmms-eval
#   ts -S 2
#   for s in qwen3vl_2b.sh qwen3vl_4b.sh internvl3p5_2b.sh llava_onevision_0p5b.sh cambrians_1p5b.sh cambrians_3b.sh; do
#     ts bash scripts/video_consistency_production/$s
#   done
#
# Override GPUs per model, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/video_consistency/run_video_consistency_small_models.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D="${ROOT}/scripts/video_consistency_production"

for s in \
  qwen3vl_2b.sh \
  qwen3vl_4b.sh \
  internvl3p5_2b.sh \
  llava_onevision_0p5b.sh \
  cambrians_1p5b.sh \
  cambrians_3b.sh
do
  echo "======== ${s} ========"
  bash "${D}/${s}"
done
