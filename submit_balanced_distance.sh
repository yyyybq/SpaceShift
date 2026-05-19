#!/usr/bin/env bash
# Regenerate object_distance_to_camera entries with a uniform distance
# distribution across 0.5 .. 5.9 m (step 0.2) for both image and video
# consistency eval datasets. Every stage is resumable and scene-parallel.
#
# Usage:
#   bash submit_balanced_distance.sh all   [extra flags...]
#   bash submit_balanced_distance.sh scan  [extra flags...]
#   bash submit_balanced_distance.sh plan  [extra flags...]
#   bash submit_balanced_distance.sh render [extra flags...]
#   bash submit_balanced_distance.sh merge [extra flags...]
#
# Typical full-scale run (multi-GPU, high parallelism):
#   bash submit_balanced_distance.sh all \
#     --workers 8 --gpu 0 \
#     --work_dir balanced_distance_work
#
# Single-GPU (workers share the same GPU):
#   bash submit_balanced_distance.sh all --workers 4 --gpu 0 --share_base_gpu
#
# Full-scale safe defaults are already set in the Python script. Override
# anything via extra flags (see `python regenerate_balanced_distance_camera.py --help`).
set -euo pipefail

WORKSPACE=$(cd "$(dirname "$0")" && pwd)
cd "$WORKSPACE"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash submit_balanced_distance.sh {scan|plan|render|merge|all} [args...]" >&2
  exit 1
fi

STAGE="$1"
shift

# Sensible defaults; can be overridden via CLI flags.
export PYTHONPATH="${PYTHONPATH:-}:$WORKSPACE/src"

python -u regenerate_balanced_distance_camera.py "$STAGE" \
  --work_dir "${WORK_DIR:-balanced_distance_work}" \
  --image_eval_dir "${IMAGE_EVAL_DIR:-image_consistency_thor_eval_v2}" \
  --video_eval_dir "${VIDEO_EVAL_DIR:-video_consistency_thor_eval_v2}" \
  "$@"
