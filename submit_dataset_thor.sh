#!/usr/bin/env bash
# Submit wrapper for the THOR-only video consistency benchmark.
#
# Usage:
#   bash submit_dataset_thor.sh plan --output_dir ./video_consistency_output
#   bash submit_dataset_thor.sh render --output_dir ./video_consistency_output
#   bash submit_dataset_thor.sh report --output_dir ./video_consistency_output
#   bash submit_dataset_thor.sh all --output_dir ./video_consistency_output
#   bash submit_dataset_thor.sh all --wipe --output_dir ./video_consistency_output
#
# Full-scale production (all 120 scenes, wide radius, all candidates):
#   Workers are distributed round-robin across GPUs (auto-detected).
#   Each GPU can run ~4 THOR controllers concurrently. With 8 GPUs -> 32 workers is safe.
#   bash submit_dataset_thor.sh all --output_dir ./video_consistency_thor_full \
#     --plan_parallel_workers 32 --render_parallel_workers 8 --gpu 0
#
# Constrained mode (old behaviour, 22 groups per engine):
#   bash submit_dataset_thor.sh all --output_dir ./video_consistency_thor_constrained \
#     --no_exhaust_all_candidates --plan_parallel_workers 32 --render_parallel_workers 8 \
#     --render_share_base_gpu --gpu 0
#
# Custom radii:
#   bash submit_dataset_thor.sh all --output_dir ./video_consistency_thor_custom \
#     --thor_radii 0.3 0.5 1.0 1.5 2.0 3.0
#
# Multi-GPU render: --render_parallel_workers 8 --gpu 0  (worker i uses GPU i).
# Single GPU: add --render_share_base_gpu (all workers use --gpu).
#
# After mining, plan_mined_candidates.pkl is written; retry plan with --reuse_mined_plan_candidates
# if group selection / metadata write failed once.
#
# Input spec:
#   All CLI flags are forwarded to src/build_video_consistency_dataset.py with
#   --engines thor added automatically. Optional: --thor_scenes FloorPlan1 ...
#
# Output spec:
#   <output_dir>/
#     metadata.json          (config, groups, clips, render_records, stats)
#     benchmark_plan.json
#     plan_mined_candidates.pkl   (checkpoint after mining)
#     qa.json                (per-clip QA with scene_id, radius, object name, etc.)
#     clips.jsonl
#     consistency_groups.json
#     dataset_stats.json
#     video_consistency_dataset.md
#     videos/<clip_id>/frames/  video.mp4

set -euo pipefail

WORKSPACE=$(cd "$(dirname "$0")" && pwd)

if [ "$#" -lt 1 ]; then
  echo "Usage: bash submit_dataset_thor.sh {plan|render|report|all} [args...]" >&2
  exit 1
fi

STAGE="$1"
shift

cd "$WORKSPACE"
python -u src/build_video_consistency_dataset.py "$STAGE" --engines thor "$@"
