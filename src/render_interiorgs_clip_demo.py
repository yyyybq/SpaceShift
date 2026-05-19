"""Render one InteriorGS trajectory: output_dir/videos/<clip_id>/{frames,video.mp4}. See --help example."""

import argparse
import asyncio
import shutil
import subprocess
from pathlib import Path

from video_consistency_dataset.benchmark_config import BenchmarkConfig
from video_consistency_dataset.defaults import CIRCULAR_MOTIONS
from video_consistency_dataset.interiorgs_imports import load_interiorgs_symbols
from video_consistency_dataset.interiorgs_sequences import build_object_sequence, build_scene_context


def _encode_video(frame_dir: Path, video_path: Path, fps: int) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    assert ffmpeg_path is not None, "ffmpeg not found on PATH."
    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark-layout InteriorGS clip (continuous trajectory).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Verified: python src/render_interiorgs_clip_demo.py --output_dir ./sanity_check/benchmark_style_clips "
        "--interiorgs_root /nas2/anjali/InteriorGS --scene_id 0001_839920 --object_id 67 --motion_family approach "
        "--direction fw --radius 0.5 --interiorgs_linear_steps 12 --fps 10 --gpu 0",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--interiorgs_root", required=True)
    p.add_argument("--scene_id", required=True)
    p.add_argument("--object_id", required=True, help="Scene object id string (e.g. from labels.json).")
    p.add_argument("--motion_family", default="around", choices=("approach", "passby", "around", "spherical"))
    p.add_argument("--direction", default="cw", help="cw|ccw for orbit; fw|bw for linear patterns.")
    p.add_argument("--radius", type=float, default=None, help="Orbit/linear radius; default linear_radius for linear.")
    p.add_argument("--start_angle_offset", type=float, default=0.0)
    p.add_argument("--bounce_count", type=int, default=1)
    p.add_argument("--increment", type=float, default=10.0)
    p.add_argument("--clip_id", default=None, help="Subfolder name under videos/ ; default auto.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--interiorgs_image_size", type=int, default=512)
    p.add_argument("--interiorgs_fov", type=float, default=60.0)
    p.add_argument("--interiorgs_rotation_interval", type=float, default=10.0)
    p.add_argument("--interiorgs_linear_steps", type=int, default=36)
    p.add_argument("--interiorgs_half_span", type=float, default=1.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    scene_path = Path(args.interiorgs_root).resolve() / args.scene_id
    assert scene_path.is_dir(), f"Scene not found: {scene_path}"

    cfg = BenchmarkConfig(
        output_dir=str(root),
        interiorgs_root=str(Path(args.interiorgs_root).resolve()),
        thor_scenes=("dummy",),
        interiorgs_scenes=(args.scene_id,),
        enabled_engines=("interiorgs",),
        fps=args.fps,
        gpu=args.gpu,
        increment=args.increment,
        interiorgs_image_size=args.interiorgs_image_size,
        interiorgs_fov=args.interiorgs_fov,
        interiorgs_rotation_interval=args.interiorgs_rotation_interval,
        interiorgs_linear_steps=args.interiorgs_linear_steps,
        interiorgs_half_span=args.interiorgs_half_span,
    )

    clip_id = args.clip_id or f"interiorgs_demo_{args.scene_id}_{args.object_id}_{args.motion_family}_{args.direction}"
    clip_root = root / "videos" / clip_id
    frame_dir = clip_root / "frames"
    if clip_root.exists():
        shutil.rmtree(clip_root)
    frame_dir.mkdir(parents=True)

    symbols = load_interiorgs_symbols()
    sel = symbols["ObjectSelector"](symbols["ObjectSelectionConfig"]())
    sampler = symbols["CameraSampler"](
        symbols["CameraSamplingConfig"](
            image_width=cfg.interiorgs_image_size,
            image_height=cfg.interiorgs_image_size,
            fov_deg=cfg.interiorgs_fov,
            rotation_interval=cfg.interiorgs_rotation_interval,
        )
    )
    scene_object_to_aabb = symbols["scene_object_to_aabb"]
    all_scene_objects = sel.get_all_parsed_objects(scene_path, include_walls=False)
    obj = None
    for item in all_scene_objects:
        if str(item.id) == str(args.object_id):
            obj = item
            break
    assert obj is not None, f"object_id {args.object_id!r} not in scene {args.scene_id}"

    context = build_scene_context(sampler, scene_path, all_scene_objects, scene_object_to_aabb)

    def _build_seq(radius: float, start_off: float) -> list:
        return build_object_sequence(
            sampler,
            symbols["CameraPose"],
            scene_path,
            obj,
            context,
            args.motion_family,
            args.direction,
            radius,
            start_off,
            args.bounce_count,
            cfg.increment,
            cfg.interiorgs_linear_steps,
            cfg.interiorgs_half_span,
        )

    sequence: list = []
    if args.motion_family in CIRCULAR_MOTIONS:
        radii_to_try: list[float] = []
        if args.radius is not None:
            radii_to_try.append(args.radius)
        for r in cfg.thor_radii:
            if r not in radii_to_try:
                radii_to_try.append(r)
        angle_candidates = [args.start_angle_offset, 0.0, 90.0, 180.0, 270.0]
        angles_to_try: list[float] = []
        for a in angle_candidates:
            if a not in angles_to_try:
                angles_to_try.append(a)
        for start_off in angles_to_try:
            for r in radii_to_try:
                sequence = _build_seq(r, start_off)
                if sequence:
                    print(f"Orbit ok: r={r} ang={start_off} n={len(sequence)}", flush=True)
                    break
            if sequence:
                break
        assert sequence, (
            f"Empty {args.motion_family} orbit; tried r={radii_to_try} ang={angles_to_try}. "
            "Try approach/passby or another --object_id."
        )
    else:
        linear_radii: list[float] = []
        if args.radius is not None:
            linear_radii.append(args.radius)
        for r in (*cfg.thor_radii, cfg.linear_radius, 0.35, 0.5, 2.0, 3.0):
            if r not in linear_radii:
                linear_radii.append(r)
        for r in linear_radii:
            sequence = _build_seq(r, args.start_angle_offset)
            if sequence:
                print(f"Linear path ok: radius={r}, {len(sequence)} poses", flush=True)
                break
        assert sequence, (
            f"Empty {args.motion_family} sequence; tried radii {linear_radii}. "
            "Lower --interiorgs_linear_steps (e.g. 12), change --object_id, or set --radius (e.g. 0.5)."
        )

    renderer = symbols["SceneRenderer"](
        symbols["RenderConfig"](
            scenes_root=cfg.interiorgs_root,
            render_backend="local",
            image_width=cfg.interiorgs_image_size,
            image_height=cfg.interiorgs_image_size,
            fov_deg=cfg.interiorgs_fov,
            gpu_device=cfg.gpu,
        )
    )
    try:
        asyncio.run(renderer.set_scene(args.scene_id))
        intrinsics = sampler.intrinsics
        look_at = symbols["look_at_matrix"]
        for frame_index, pose in enumerate(sequence):
            image = renderer.render_image_sync(intrinsics, look_at(pose.position, pose.target))
            assert image is not None, f"render failed frame {frame_index}"
            image.save(frame_dir / f"frame_{frame_index:05d}.png")
        _encode_video(frame_dir, clip_root / "video.mp4", cfg.fps)
    finally:
        asyncio.run(renderer.close())

    print(f"Wrote {clip_root / 'video.mp4'} ({len(sequence)} frames)", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
