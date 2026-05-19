"""Build the image (view variation) consistency benchmark from video benchmark output.

Usage:
  # Extract image benchmark from an existing video benchmark output
  python src/build_image_consistency_dataset.py extract \
    --video_output_dir ./video_consistency_thor_eval_v2 \
    --output_dir ./image_consistency_output \
    --views_per_group 10 \
    --target_groups_per_family 100

  # Backfill frame_visibility.json for pair-distance clips (THOR only)
  python src/build_image_consistency_dataset.py backfill-visibility \
    --video_output_dir ./video_consistency_thor_eval_v2 \
    --gpu 0

Stages:
  extract              Read video output, select frames, write image benchmark.
  backfill-visibility  Re-open THOR simulator for pair_distance clips missing
                       frame_visibility.json and write sidecars.

Output:
  <output_dir>/
    qa.json                 Per-image QA entries
    consistency_groups.json Per-group with image_ids
    dataset_stats.json      Statistics
    image_consistency_dataset.md
    images/
      <image_id>/
        image.png           Materialized image copied from source video frame
"""

import argparse
from pathlib import Path

from image_consistency_dataset.extractor import extract_image_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the image consistency benchmark.")
    sub = parser.add_subparsers(dest="stage", required=True)

    extract_p = sub.add_parser("extract", help="Extract image benchmark from video output.")
    extract_p.add_argument("--video_output_dir", required=True, help="Path to video benchmark output directory.")
    extract_p.add_argument("--output_dir", required=True, help="Path to write image benchmark output.")
    extract_p.add_argument("--views_per_group", type=int, default=10, help="Number of images per group.")
    extract_p.add_argument("--target_groups_per_family", type=int, default=100, help="Max groups per question family.")
    extract_p.add_argument("--max_per_label", type=int, default=10, help="Max groups per anchor label within each family.")
    extract_p.add_argument("--max_per_scene", type=int, default=5, help="Max groups per scene within each family.")

    backfill_p = sub.add_parser("backfill-visibility", help="Backfill frame_visibility.json for pair clips.")
    backfill_p.add_argument("--video_output_dir", required=True, help="Path to video benchmark output directory.")
    backfill_p.add_argument("--gpu", type=int, default=0, help="GPU device for THOR.")
    backfill_p.add_argument(
        "--clip_ids_path",
        default=None,
        help="Optional text file containing one clip_id per line to limit backfill scope.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "extract":
        extract_image_benchmark(
            video_dir=args.video_output_dir,
            output_dir=args.output_dir,
            views_per_group=args.views_per_group,
            target_groups_per_family=args.target_groups_per_family,
            max_per_label=args.max_per_label,
            max_per_scene=args.max_per_scene,
        )
    elif args.stage == "backfill-visibility":
        from image_consistency_dataset.pair_visibility import backfill_thor_visibility
        clip_ids = None
        if args.clip_ids_path:
            clip_ids = {
                line.strip()
                for line in Path(args.clip_ids_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        backfill_thor_visibility(
            video_output_dir=args.video_output_dir,
            gpu=args.gpu,
            clip_ids=clip_ids,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
