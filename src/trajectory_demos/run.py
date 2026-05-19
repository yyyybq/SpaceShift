"""CLI helpers for curated AI2-THOR trajectory demos."""

import argparse
import json

from trajectory_demos.demo_config import (
    DEFAULT_CANDIDATE_SCENES,
    TrajectoryDemoConfig,
)
from trajectory_demos.exporter import export_selection
from trajectory_demos.selector import select_scene


def parse_args() -> TrajectoryDemoConfig:
    parser = argparse.ArgumentParser(
        description="Export curated AI2-THOR trajectory demo videos."
    )
    parser.add_argument(
        "--scene",
        default="auto",
        help="AI2-THOR scene name or 'auto' to score candidate scenes.",
    )
    parser.add_argument("--output_dir", default="./trajectory_output")
    parser.add_argument("--trajectory", default="all")
    parser.add_argument("--object_index", type=int, default=None)
    parser.add_argument("--object_id", default=None)
    parser.add_argument("--object_type", default=None)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--arc_degrees", type=float, default=360.0)
    parser.add_argument("--increment", type=float, default=5.0)
    parser.add_argument("--fov", type=int, default=75)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min_frames", type=int, default=30)
    parser.add_argument("--candidate_scenes", nargs="*", default=None)
    args = parser.parse_args()
    candidate_scenes = tuple(args.candidate_scenes) if args.candidate_scenes else DEFAULT_CANDIDATE_SCENES
    return TrajectoryDemoConfig(
        scene=args.scene,
        output_dir=args.output_dir,
        trajectory=args.trajectory,
        object_index=args.object_index,
        object_id=args.object_id,
        object_type=args.object_type,
        radius=args.radius,
        arc_degrees=args.arc_degrees,
        increment=args.increment,
        field_of_view=args.fov,
        image_size=args.image_size,
        fps=args.fps,
        gpu=args.gpu,
        min_frames=args.min_frames,
        candidate_scenes=candidate_scenes,
    )


def run(config: TrajectoryDemoConfig) -> dict:
    selection = select_scene(config)
    return export_selection(selection, config)


def main() -> int:
    manifest = run(parse_args())
    print(json.dumps(manifest, indent=2))
    return 0
