#!/usr/bin/env bash
# Submit wrapper for the matched video consistency benchmark.
#
# Usage:
#   bash submit_dataset.sh plan --output_dir ./video_consistency_output --engines thor
#   bash submit_dataset.sh render --output_dir ./video_consistency_output --engines thor
#   bash submit_dataset.sh report --output_dir ./video_consistency_output --engines thor
#   bash submit_dataset.sh all --output_dir ./video_consistency_output --engines thor
#   bash submit_dataset.sh all --wipe --output_dir ./video_consistency_output --engines thor
#   bash submit_dataset.sh all --output_dir ./video_consistency_output --interiorgs_root /path/to/InteriorGS
#   Multi-GPU render (8 workers): append --render_parallel_workers 8 --gpu 0
#   Parallel plan mining (cap ~ scene count): --plan_parallel_workers 32
#
# Input spec:
#   All CLI flags are forwarded to src/build_video_consistency_dataset.py.
#
# Output spec:
#   <output_dir>/
#     metadata.json
#     qa.json
#     clips.jsonl
#     consistency_groups.json
#     dataset_stats.json
#     video_consistency_dataset.md
#     videos/<clip_id>/frames/  video.mp4

set -euo pipefail

WORKSPACE=$(cd "$(dirname "$0")" && pwd)

cd "$WORKSPACE"
python -u src/build_video_consistency_dataset.py "$@"
