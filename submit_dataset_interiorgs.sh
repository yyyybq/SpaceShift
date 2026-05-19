#!/usr/bin/env bash
# Submit wrapper for the InteriorGS-only video consistency benchmark.
#
# Usage:
#   bash submit_dataset_interiorgs.sh plan --output_dir ./video_consistency_output --interiorgs_root /path/to/InteriorGS
#   bash submit_dataset_interiorgs.sh render --output_dir ./video_consistency_output --interiorgs_root /path/to/InteriorGS
#   bash submit_dataset_interiorgs.sh report --output_dir ./video_consistency_output --interiorgs_root /path/to/InteriorGS
#   bash submit_dataset_interiorgs.sh all --output_dir ./video_consistency_output --interiorgs_root /real/path/to/InteriorGS_scenes
#   Multi-GPU render (8 workers; InteriorGS uses CUDA devices gpu..gpu+7):
#   Multi-GPU: --render_parallel_workers 8 --gpu 0  (worker i -> GPU i).
#   Single GPU: add --render_share_base_gpu so all workers use --gpu (VRAM permitting).
#   bash submit_dataset_interiorgs.sh render --output_dir ./out --interiorgs_root /path/InteriorGS --render_parallel_workers 8 --gpu 0
#
# Input spec:
#   All CLI flags are forwarded to src/build_video_consistency_dataset.py with
#   --engines interiorgs added automatically. --interiorgs_root is required
#   (same scene root as InteriorGS training data: per-scene labels.json, structure.json).
#
# After mining, plan_mined_candidates.pkl is written; retry plan with --reuse_mined_plan_candidates
# to skip mining if group selection / metadata write failed once.
#
# Output spec:
#   <output_dir>/
#     metadata.json
#     plan_mined_candidates.pkl   (checkpoint after mining)
#     qa.json
#     clips.jsonl
#     consistency_groups.json
#     dataset_stats.json
#     video_consistency_dataset.md
#     videos/<clip_id>/frames/  video.mp4

set -euo pipefail

WORKSPACE=$(cd "$(dirname "$0")" && pwd)

if [ "$#" -lt 1 ]; then
  echo "Usage: bash submit_dataset_interiorgs.sh {plan|render|report|all} [--output_dir DIR] --interiorgs_root PATH [args...]" >&2
  exit 1
fi

STAGE="$1"
shift

cd "$WORKSPACE"
python -u src/build_video_consistency_dataset.py "$STAGE" --engines interiorgs "$@"
