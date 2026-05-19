"""Build the matched video consistency dataset.

Usage:
  python src/build_video_consistency_dataset.py plan --output_dir ./video_consistency_output --engines thor
  python src/build_video_consistency_dataset.py render --output_dir ./video_consistency_output --engines thor
  python src/build_video_consistency_dataset.py report --output_dir ./video_consistency_output --engines thor
  python src/build_video_consistency_dataset.py all --output_dir ./video_consistency_output --engines thor
  python src/build_video_consistency_dataset.py all --wipe --output_dir ./video_consistency_output --engines thor
  python src/build_video_consistency_dataset.py all --output_dir ./video_consistency_output --engines interiorgs --interiorgs_root /real/path/InteriorGS_scenes

  --wipe removes the entire --output_dir if it exists, then runs the stage. Use with all for a clean rebuild.

Input spec:
  stage                one of: plan, render, report, all
  --output_dir         benchmark artifact root
  --engines            enabled engines from: thor, interiorgs
  --interiorgs_root    InteriorGS scene root with per-scene labels.json and structure.json
  --thor_scenes        optional THOR scene allowlist
  --interiorgs_scenes  optional InteriorGS scene allowlist
  --random_seed        seed for randomized start angles and cycle counts

Output spec:
  <output_dir>/
    metadata.json
    qa.json
    clips.jsonl
    consistency_groups.json
    dataset_stats.json
    video_consistency_dataset.md
    videos/<clip_id>/frames/  video.mp4
"""

import argparse
import shutil
from pathlib import Path

from video_consistency_dataset.benchmark_config import BenchmarkConfig
from video_consistency_dataset.scene_lists import default_thor_scenes, discover_interiorgs_scenes
from video_consistency_dataset.workflow import run_all_stages, run_plan_stage, run_render_stage, run_report_stage


def _wipe_output_dir(output_dir: str) -> None:
    root = Path(output_dir).expanduser().resolve()
    assert root.name, "output_dir must not be empty or filesystem root."
    assert root != root.parent, "Refusing to wipe a path with no directory parent."
    if root.exists():
        shutil.rmtree(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the video consistency benchmark.")
    parser.add_argument("stage", choices=("plan", "render", "report", "all"))
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete output_dir completely if present before running (full regenerate).",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--engines", nargs="+", choices=("thor", "interiorgs"), default=("thor", "interiorgs"))
    parser.add_argument("--interiorgs_root", default=None)
    parser.add_argument("--thor_scenes", nargs="*", default=None)
    parser.add_argument("--interiorgs_scenes", nargs="*", default=None)
    parser.add_argument("--group_size", type=int, default=30)
    parser.add_argument("--min_group_size", type=int, default=25)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--plan_parallel_workers",
        type=int,
        default=1,
        help="Plan-stage scene mining processes (spawn). Capped by number of scenes per engine. 1 = serial.",
    )
    parser.add_argument(
        "--render_parallel_workers",
        type=int,
        default=1,
        help="Render stage processes (spawn). Worker i uses GPU (gpu + i) unless --render_share_base_gpu. 1 = serial.",
    )
    parser.add_argument(
        "--render_share_base_gpu",
        action="store_true",
        help="Parallel render: every worker uses --gpu (single-GPU machine). Default: worker i uses gpu+i.",
    )
    parser.add_argument(
        "--reuse_mined_plan_candidates",
        action="store_true",
        help="Plan: load plan_mined_candidates.pkl from output_dir and skip mining (after a prior run saved mining).",
    )
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--increment", type=float, default=10.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--interiorgs_image_size", type=int, default=512)
    parser.add_argument(
        "--exhaust_all_candidates",
        action="store_true",
        default=True,
        help="Use every mined candidate as a group (maximize output). Disable with --no_exhaust_all_candidates.",
    )
    parser.add_argument("--no_exhaust_all_candidates", dest="exhaust_all_candidates", action="store_false")
    parser.add_argument(
        "--thor_radii",
        nargs="+",
        type=float,
        default=None,
        help="Orbit radii for THOR mining (default: 0.3 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5).",
    )
    parser.add_argument(
        "--max_groups_per_scene",
        type=int,
        default=0,
        help="Cap groups per scene for balanced scene distribution. 0 = no cap.",
    )
    parser.add_argument("--max_interiorgs_scenes", type=int, default=20)
    parser.add_argument(
        "--interiorgs_mining_max_objects",
        type=int,
        default=48,
        help="Cap single-object mining per scene (0 = no cap). Lower is faster.",
    )
    parser.add_argument(
        "--interiorgs_rotation_pair_pose_samples",
        type=int,
        default=24,
        help="Max rotation poses used for pair-distance mining per sequence (subsamples long spins).",
    )
    parser.add_argument(
        "--reuse_video_dirs",
        nargs="*",
        default=None,
        help="Directories with existing rendered videos to prefer during balanced planning and symlink before rendering.",
    )
    parser.add_argument(
        "--plan_gpus",
        nargs="+",
        type=int,
        default=None,
        help="GPU IDs for plan-stage mining (round-robin across workers). Default: use --gpu.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    thor_scenes = tuple(args.thor_scenes) if args.thor_scenes else default_thor_scenes()
    enabled_engines = tuple(dict.fromkeys(args.engines))
    if "interiorgs" in enabled_engines:
        assert args.interiorgs_root is not None, "--interiorgs_root is required when interiorgs is enabled."
        interiorgs_root = Path(args.interiorgs_root)
        assert interiorgs_root.is_dir(), (
            f"--interiorgs_root must be an existing directory (got {args.interiorgs_root!r}). "
            "Use the path that contains per-scene folders (e.g. 0267_840790/), not a doc placeholder."
        )
        interiorgs_scenes = (
            tuple(args.interiorgs_scenes)
            if args.interiorgs_scenes
            else discover_interiorgs_scenes(str(interiorgs_root), args.max_interiorgs_scenes)
        )
    else:
        interiorgs_scenes = ()
    thor_radii = tuple(args.thor_radii) if args.thor_radii else BenchmarkConfig.thor_radii
    return BenchmarkConfig(
        output_dir=args.output_dir,
        interiorgs_root=args.interiorgs_root,
        thor_scenes=thor_scenes,
        interiorgs_scenes=interiorgs_scenes,
        enabled_engines=enabled_engines,
        group_size=args.group_size,
        min_group_size=args.min_group_size,
        gpu=args.gpu,
        plan_parallel_workers=args.plan_parallel_workers,
        render_parallel_workers=args.render_parallel_workers,
        render_share_base_gpu=args.render_share_base_gpu,
        reuse_mined_plan_candidates=args.reuse_mined_plan_candidates,
        fps=args.fps,
        increment=args.increment,
        random_seed=args.random_seed,
        image_size=args.image_size,
        interiorgs_image_size=args.interiorgs_image_size,
        interiorgs_max_scenes=args.max_interiorgs_scenes,
        interiorgs_mining_max_objects=args.interiorgs_mining_max_objects,
        interiorgs_rotation_pair_pose_samples=args.interiorgs_rotation_pair_pose_samples,
        thor_radii=thor_radii,
        exhaust_all_candidates=args.exhaust_all_candidates,
        max_groups_per_scene=args.max_groups_per_scene,
        reuse_video_dirs=tuple(args.reuse_video_dirs) if args.reuse_video_dirs else (),
        plan_gpus=tuple(args.plan_gpus) if args.plan_gpus else (),
    )


def main() -> int:
    args = parse_args()
    if args.wipe:
        _wipe_output_dir(args.output_dir)
    config = build_config(args)
    if args.stage == "plan":
        run_plan_stage(config)
    elif args.stage == "render":
        run_render_stage(config)
    elif args.stage == "report":
        run_report_stage(config)
    else:
        run_all_stages(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
