"""Generate large-scale video QA dataset from AI2-THOR trajectory videos.

Usage:
  python src/generate_video_qa_dataset.py
  python src/generate_video_qa_dataset.py --scenes FloorPlan1 FloorPlan201
  python src/generate_video_qa_dataset.py --room_types kitchen living_room
  python src/generate_video_qa_dataset.py --trajectories around_cw spherical_cw --max_objects 3

Input spec:
  --output_dir        root directory for dataset output
  --scenes            explicit scene names (FloorPlan1, FloorPlan201, ...)
  --room_types        room categories (kitchen, living_room, bedroom, bathroom)
  --trajectories      subset of trajectory names, or 'all'
  --radius            orbit radius in meters
  --increment         angular step in degrees
  --fps               video frame rate
  --image_size        square resolution in pixels
  --fov               camera field of view in degrees
  --gpu               CUDA device id
  --min_frames        minimum visible frames per trajectory
  --max_objects        max objects per scene per trajectory variation
  --skip_existing     skip trajectories with existing qa.json

Output spec:
  <output_dir>/
    dataset_manifest.json
    <scene>/
      <trajectory>_<ObjectType>_<variation_tag>/
        frames/          frame_00000.png, ...
        video.mp4
        manifest.json    per-video metadata
        qa.json          size + distance QA pairs
"""

import argparse
import json

from trajectory_demos.demo_config import ALL_TRAJECTORIES
from video_qa_dataset.dataset_config import DatasetConfig, expanded_scenes
from video_qa_dataset.dataset_pipeline import run_pipeline


def parse_args() -> DatasetConfig:
    parser = argparse.ArgumentParser(
        description="Generate large-scale video QA dataset."
    )
    parser.add_argument("--output_dir", default="./video_qa_output")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--room_types", nargs="*", default=None)
    parser.add_argument("--trajectories", nargs="*", default=None)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--increment", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--fov", type=int, default=75)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min_frames", type=int, default=30)
    parser.add_argument("--max_objects", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    scenes = expanded_scenes(args.room_types, args.scenes)
    trajectories = tuple(args.trajectories) if args.trajectories else ALL_TRAJECTORIES

    for t in trajectories:
        assert t in ALL_TRAJECTORIES, f"Unknown trajectory: {t}. Options: {ALL_TRAJECTORIES}"

    return DatasetConfig(
        output_dir=args.output_dir,
        scenes=scenes,
        trajectories=trajectories,
        radius=args.radius,
        arc_degrees=360.0,
        increment=args.increment,
        field_of_view=args.fov,
        image_size=args.image_size,
        fps=args.fps,
        gpu=args.gpu,
        min_frames=args.min_frames,
        max_objects_per_scene=args.max_objects,
        skip_existing=args.skip_existing,
    )


def main() -> int:
    config = parse_args()
    manifest = run_pipeline(config)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
