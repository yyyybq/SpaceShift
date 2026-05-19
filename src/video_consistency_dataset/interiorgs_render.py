"""Render InteriorGS benchmark clips from a planned manifest.

Output per clip:
  videos/<clip_id>/
    frames/
    video.mp4
"""

import asyncio
import hashlib
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

from video_consistency_dataset.benchmark_variations import variation_from_clip
from video_consistency_dataset.defaults import CIRCULAR_MOTIONS
from video_consistency_dataset.interiorgs_imports import load_interiorgs_symbols
from video_consistency_dataset.interiorgs_sequences import build_object_sequence, build_rotation_sequence, build_scene_context


def _encode_video(frame_dir: Path, video_path: Path, fps: int) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    assert ffmpeg_path is not None, "ffmpeg not found on PATH."
    subprocess.run([ffmpeg_path, "-y", "-framerate", str(fps), "-i", str(frame_dir / "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path)], check=True)


def _room_index_from_bucket(room_bucket: str) -> int:
    return int(room_bucket.rsplit("room_", 1)[1])


def render_interiorgs_clips(plan, config, clip_ids: set[str] | None = None) -> dict[str, dict]:
    symbols = load_interiorgs_symbols()
    ObjectSelector = symbols["ObjectSelector"]
    ObjectSelectionConfig = symbols["ObjectSelectionConfig"]
    CameraSampler = symbols["CameraSampler"]
    CameraSamplingConfig = symbols["CameraSamplingConfig"]
    SceneRenderer = symbols["SceneRenderer"]
    RenderConfig = symbols["RenderConfig"]
    scene_object_to_aabb = symbols["scene_object_to_aabb"]
    selector = ObjectSelector(ObjectSelectionConfig())
    sampler = CameraSampler(CameraSamplingConfig(
        image_width=config.interiorgs_image_size,
        image_height=config.interiorgs_image_size,
        fov_deg=config.interiorgs_fov,
        rotation_interval=config.interiorgs_rotation_interval,
    ))
    render_config = RenderConfig(
        scenes_root=config.interiorgs_root,
        render_backend="local",
        image_width=config.interiorgs_image_size,
        image_height=config.interiorgs_image_size,
        fov_deg=config.interiorgs_fov,
        gpu_device=config.gpu,
    )
    renderer = SceneRenderer(render_config)
    clips_by_scene: dict[str, list] = {}
    render_records: dict[str, dict] = {}
    for clip in plan.clips:
        if clip.engine != "interiorgs":
            continue
        if clip_ids is not None and clip.clip_id not in clip_ids:
            continue
        clips_by_scene.setdefault(clip.scene_id, []).append(clip)
    clip_total = sum(len(v) for v in clips_by_scene.values())
    pbar = tqdm(total=clip_total, desc="InteriorGS render", unit="clip", dynamic_ncols=True)
    try:
        for scene_id, scene_clips in clips_by_scene.items():
            asyncio.run(renderer.set_scene(scene_id))
            scene_path = Path(config.interiorgs_root) / scene_id
            all_scene_objects = selector.get_all_parsed_objects(scene_path, include_walls=False)
            context = build_scene_context(sampler, scene_path, all_scene_objects, scene_object_to_aabb)
            for clip in scene_clips:
                variation = variation_from_clip(clip)
                if clip.anchor_kind == "object":
                    obj = [item for item in all_scene_objects if item.id == clip.anchor_ids[0]][0]
                    pose_inc = config.circular_increment if clip.motion_family in CIRCULAR_MOTIONS else config.increment
                    sequence = build_object_sequence(
                        sampler,
                        symbols["CameraPose"],
                        scene_path,
                        obj,
                        context,
                        clip.motion_family,
                        clip.direction,
                        clip.radius or config.linear_radius,
                        variation.start_angle_offset,
                        variation.bounce_count,
                        pose_inc,
                        config.interiorgs_linear_steps,
                        config.interiorgs_half_span,
                    )
                else:
                    sequence = build_rotation_sequence(sampler, scene_path, all_scene_objects, _room_index_from_bucket(clip.room_bucket), clip.direction, variation.start_angle_offset, variation.bounce_count)
                assert sequence, f"No InteriorGS sequence produced for {clip.clip_id}"
                clip_root = Path(clip.output_dir)
                if clip_root.exists():
                    shutil.rmtree(clip_root)
                frame_dir = clip_root / "frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                intrinsics = sampler.intrinsics
                saved_frames = 0
                seen_hashes: set[str] = set()
                for pose in sequence:
                    image = renderer.render_image_sync(intrinsics, symbols["look_at_matrix"](pose.position, pose.target))
                    assert image is not None, f"Rendering failed for {clip.clip_id}"
                    current_hash = hashlib.md5(image.tobytes()).hexdigest()
                    if current_hash in seen_hashes:
                        continue
                    seen_hashes.add(current_hash)
                    image.save(frame_dir / f"frame_{saved_frames:05d}.png")
                    saved_frames += 1
                _encode_video(frame_dir, clip_root / "video.mp4", config.fps)
                render_records[clip.clip_id] = {
                    "saved_frames": saved_frames,
                    "total_poses": len(sequence),
                    "pose_source": f"interiorgs.{clip.motion_family}.{clip.variation_tag}",
                }
                pbar.update(1)
    finally:
        pbar.close()
        asyncio.run(renderer.close())
    return render_records
