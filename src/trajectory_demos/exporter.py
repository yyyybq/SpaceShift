"""Frame export and manifest writing for curated trajectory demos."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from utils import local_storage_utils

from trajectory_demos.controller_utils import build_controller, prepare_scene, teleport_pose


def _save_frame(frame, frame_path: Path, output_root: Path) -> None:
    relative_path = str(frame_path.relative_to(output_root))
    saved_path = local_storage_utils.upload_numpy_array_to_local(
        frame,
        relative_path,
        str(output_root),
    )
    assert saved_path is not None, f"Failed to save frame to {frame_path}."


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


def _trajectory_manifest(candidate, config, video_path: Path) -> dict:
    return {
        "scene": candidate.scene,
        "trajectory": candidate.trajectory,
        "object_id": candidate.object_id,
        "object_type": candidate.object_type,
        "radius": config.radius,
        "arc_degrees": config.arc_degrees,
        "increment": config.increment,
        "fps": config.fps,
        "total_poses": len(candidate.poses),
        "saved_frames": candidate.saved_frames,
        "field_of_view": config.field_of_view,
        "image_size": config.image_size,
        "pose_source": candidate.pose_source,
        "video_path": str(video_path),
    }


def _curated_manifest(selection, config) -> dict:
    return {
        "scene": selection.scene,
        "trajectory_count": len(selection.trajectories),
        "total_saved_frames": selection.total_saved_frames,
        "config": {
            "scene": config.scene,
            "trajectory": config.trajectory,
            "radius": config.radius,
            "arc_degrees": config.arc_degrees,
            "increment": config.increment,
            "fps": config.fps,
            "field_of_view": config.field_of_view,
            "image_size": config.image_size,
            "min_frames": config.min_frames,
        },
        "trajectories": {
            name: {
                "object_id": candidate.object_id,
                "object_type": candidate.object_type,
                "total_poses": len(candidate.poses),
                "saved_frames": candidate.saved_frames,
                "pose_source": candidate.pose_source,
                "video_path": str(
                    Path(config.output_dir).resolve()
                    / selection.scene
                    / f"{name}_{candidate.object_type}"
                    / "video.mp4"
                ),
            }
            for name, candidate in selection.trajectories.items()
        },
    }


def _staging_root(output_root: Path) -> Path:
    return output_root.parent / f".{output_root.name}_staging"


def export_selection(selection, config) -> dict:
    output_root = Path(config.output_dir).resolve()
    staging_root = _staging_root(output_root)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    scene_root = staging_root / selection.scene
    scene_root.mkdir(parents=True, exist_ok=True)

    controller = build_controller(
        selection.scene,
        config.gpu,
        config.image_size,
        config.field_of_view,
    )
    try:
        for trajectory in config.trajectory_names():
            candidate = selection.trajectories[trajectory]
            visible_pose_indices = set(candidate.visible_pose_indices)
            trajectory_root = scene_root / f"{trajectory}_{candidate.object_type}"
            frame_dir = trajectory_root / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            prepare_scene(controller, selection.scene)
            saved_frames = 0
            seen_hashes: set[str] = set()
            for pose_index, pose in enumerate(candidate.poses):
                success = teleport_pose(controller, pose)
                assert success, f"Teleport failed for {trajectory} pose {pose_index}."
                if pose_index not in visible_pose_indices:
                    continue
                frame = controller.last_event.frame
                current_hash = hashlib.md5(frame.tobytes()).hexdigest()
                if current_hash in seen_hashes:
                    continue
                seen_hashes.add(current_hash)
                _save_frame(
                    frame,
                    frame_dir / f"frame_{saved_frames:05d}.png",
                    staging_root,
                )
                saved_frames += 1
            video_path = trajectory_root / "video.mp4"
            _encode_video(frame_dir, video_path, config.fps)
            manifest_path = trajectory_root / "manifest.json"
            final_video_path = (
                output_root
                / selection.scene
                / f"{trajectory}_{candidate.object_type}"
                / "video.mp4"
            )
            manifest = _trajectory_manifest(candidate, config, final_video_path)
            manifest_path.write_text(
                json.dumps(local_storage_utils.sanitize_for_json(manifest), indent=2)
            )
        curated_manifest = _curated_manifest(selection, config)
        curated_manifest_path = staging_root / "curated_manifest.json"
        curated_manifest_path.write_text(
            json.dumps(local_storage_utils.sanitize_for_json(curated_manifest), indent=2)
        )
    finally:
        controller.stop()

    if output_root.exists():
        shutil.rmtree(output_root)
    staging_root.rename(output_root)
    return curated_manifest
