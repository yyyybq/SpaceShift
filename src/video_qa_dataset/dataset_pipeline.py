"""Orchestrates large-scale video QA dataset generation.

Pipeline per scene:
  1. Boot AI2-THOR controller
  2. For each trajectory pattern x direction:
     a. For each variation (start angle, bounce count):
        i.   Build candidates, score, select best per object
        ii.  Render frames + encode video
        iii. Collect visible objects (union for linear/circular,
             per-frame for rotation)
        iv.  Generate QA dispatched by category:
             linear  -> size only
             circular -> size + camera distance
             rotation -> sole-visible size + co-visible distance
  3. Write per-video manifest + QA JSON
  4. Aggregate into dataset manifest

Output tree:
  <output_dir>/
    dataset_manifest.json
    <scene>/
      <trajectory>_<ObjectType>_<variation_tag>/
        frames/          frame_00000.png ...
        video.mp4
        manifest.json
        qa.json
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from utils import local_storage_utils

from trajectory_demos.controller_utils import build_controller, prepare_scene, teleport_pose
from trajectory_demos.demo_config import TrajectoryDemoConfig
from trajectory_demos.frame_quality import (
    bbox_area_ratio,
    bbox_center_ratio,
    longest_contiguous_indices,
    object_bbox,
)
from trajectory_demos.selector import _score_candidate
from trajectory_demos.trajectory_candidate import TrajectoryCandidate

import global_config

from video_qa_dataset.dataset_config import DatasetConfig
from video_qa_dataset.variation_builder import build_varied_candidates
from video_qa_dataset.variation_config import (
    TrajectoryVariation,
    pattern_from_trajectory,
    variations_for_pattern,
)
from video_qa_dataset.video_qa_generator import generate_qa_for_trajectory
from video_qa_dataset.video_qa_templates import PATTERN_QA_CATEGORY


def _to_demo_config(config: DatasetConfig) -> TrajectoryDemoConfig:
    return TrajectoryDemoConfig(
        scene="auto",
        output_dir=config.output_dir,
        trajectory="all",
        object_index=None,
        object_id=None,
        object_type=None,
        radius=config.radius,
        arc_degrees=config.arc_degrees,
        increment=config.increment,
        field_of_view=config.field_of_view,
        image_size=config.image_size,
        fps=config.fps,
        gpu=config.gpu,
        min_frames=config.min_frames,
        candidate_scenes=config.scenes,
    )


def _save_frame(frame, frame_path: Path, output_root: Path) -> None:
    relative_path = str(frame_path.relative_to(output_root))
    saved_path = local_storage_utils.upload_numpy_array_to_local(
        frame, relative_path, str(output_root),
    )
    assert saved_path is not None, f"Failed to save frame to {frame_path}."


def _encode_video(frame_dir: Path, video_path: Path, fps: int) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    assert ffmpeg_path is not None, "ffmpeg not found on PATH."
    subprocess.run(
        [
            ffmpeg_path, "-y", "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path),
        ],
        check=True,
    )


def _visible_objects_from_controller(controller, candidate: TrajectoryCandidate) -> list[dict]:
    """Gather unique visible objects across all visible poses of a candidate."""
    seen_ids: set[str] = set()
    visible_objects: list[dict] = []
    for pose_index in candidate.visible_pose_indices:
        pose = candidate.poses[pose_index]
        if not teleport_pose(controller, pose):
            continue
        for obj in controller.last_event.metadata["objects"]:
            if obj["objectId"] in seen_ids:
                continue
            if not obj.get("visible", False):
                continue
            if obj["objectType"].lower() in global_config.IGNORE_OBJECTS_GLOBAL:
                continue
            bbox = object_bbox(controller, obj["objectId"])
            if bbox is None:
                continue
            area = bbox_area_ratio(controller, bbox)
            if area < 1.0 / 400.0:
                continue
            seen_ids.add(obj["objectId"])
            visible_objects.append(obj)
    return visible_objects


def _per_frame_visible_objects(controller, candidate: TrajectoryCandidate) -> list[list[dict]]:
    """Gather per-frame visible object lists for rotation trajectories."""
    per_frame: list[list[dict]] = []
    for pose_index in candidate.visible_pose_indices:
        pose = candidate.poses[pose_index]
        if not teleport_pose(controller, pose):
            per_frame.append([])
            continue
        frame_objects: list[dict] = []
        for obj in controller.last_event.metadata["objects"]:
            if not obj.get("visible", False):
                continue
            if obj["objectType"].lower() in global_config.IGNORE_OBJECTS_GLOBAL:
                continue
            bbox = object_bbox(controller, obj["objectId"])
            if bbox is None:
                continue
            area = bbox_area_ratio(controller, bbox)
            if area < 1.0 / 400.0:
                continue
            frame_objects.append(obj)
        per_frame.append(frame_objects)
    return per_frame


def _frame_hash(frame: np.ndarray) -> str:
    return hashlib.md5(frame.tobytes()).hexdigest()


def _render_candidate(
    controller,
    candidate: TrajectoryCandidate,
    trajectory_root: Path,
    output_root: Path,
    scene_name: str,
    fps: int,
    *,
    dedupe_identical_frames: bool = True,
) -> int:
    """Render frames and video for a single candidate. Returns saved frame count."""
    frame_dir = trajectory_root / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    prepare_scene(controller, scene_name)
    visible_pose_indices = set(candidate.visible_pose_indices)
    saved_frames = 0
    seen_hashes: set[str] | None = set() if dedupe_identical_frames else None
    for pose_index, pose in enumerate(candidate.poses):
        success = teleport_pose(controller, pose)
        assert success, f"Teleport failed at pose {pose_index}."
        if pose_index not in visible_pose_indices:
            continue
        frame = controller.last_event.frame
        if seen_hashes is not None:
            current_hash = _frame_hash(frame)
            if current_hash in seen_hashes:
                continue
            seen_hashes.add(current_hash)
        _save_frame(
            frame,
            frame_dir / f"frame_{saved_frames:05d}.png",
            output_root,
        )
        saved_frames += 1
    video_path = trajectory_root / "video.mp4"
    _encode_video(frame_dir, video_path, fps)
    return saved_frames


def _select_best_objects(
    controller,
    candidates: list[TrajectoryCandidate],
    min_frames: int,
    max_objects: int,
) -> list[TrajectoryCandidate]:
    """Score candidates and pick top-N unique objects."""
    scored = [_score_candidate(controller, c) for c in candidates]
    valid = [c for c in scored if c.saved_frames >= min_frames]
    valid.sort(key=lambda c: (c.quality_score, c.saved_frames), reverse=True)
    seen_objects: set[str] = set()
    selected: list[TrajectoryCandidate] = []
    for c in valid:
        if c.object_id in seen_objects:
            continue
        seen_objects.add(c.object_id)
        selected.append(c)
        if len(selected) >= max_objects:
            break
    return selected


def _trajectory_dir_name(trajectory: str, object_type: str, variation: TrajectoryVariation) -> str:
    return f"{trajectory}_{object_type}_{variation.tag}"


def process_scene(
    scene_name: str,
    config: DatasetConfig,
) -> dict:
    """Process all trajectories × variations × objects for a single scene."""
    output_root = Path(config.output_dir).resolve()
    scene_root = output_root / scene_name
    scene_root.mkdir(parents=True, exist_ok=True)
    demo_config = _to_demo_config(config)

    controller = build_controller(
        scene_name, config.gpu, config.image_size, config.field_of_view,
    )
    scene_results: list[dict] = []
    try:
        for trajectory in config.trajectory_list():
            pattern = pattern_from_trajectory(trajectory)
            variations = variations_for_pattern(pattern)
            print(f"  [{scene_name}] {trajectory}: {len(variations)} variations")

            for variation in variations:
                dir_suffix = f"{trajectory}_{variation.tag}"
                print(f"    variation: {variation.tag}")

                candidates = build_varied_candidates(
                    scene_name, controller, demo_config, trajectory, variation,
                )
                if not candidates:
                    print(f"      no candidates")
                    continue

                selected = _select_best_objects(
                    controller, candidates, config.min_frames, config.max_objects_per_scene,
                )
                if not selected:
                    print(f"      no valid candidates after scoring")
                    continue

                for candidate in selected:
                    traj_dir_name = _trajectory_dir_name(
                        trajectory, candidate.object_type, variation,
                    )
                    trajectory_root = scene_root / traj_dir_name
                    if config.skip_existing and (trajectory_root / "qa.json").exists():
                        print(f"      skip existing: {traj_dir_name}")
                        continue

                    if trajectory_root.exists():
                        shutil.rmtree(trajectory_root)
                    trajectory_root.mkdir(parents=True)

                    saved_frames = _render_candidate(
                        controller, candidate, trajectory_root, output_root,
                        scene_name, config.fps,
                    )
                    print(f"      rendered {traj_dir_name}: {saved_frames} frames")

                    category = PATTERN_QA_CATEGORY[pattern]
                    primary_type = candidate.object_type if candidate.object_type != "room" else None
                    if category == "rotation":
                        per_frame = _per_frame_visible_objects(controller, candidate)
                        qa_pairs = generate_qa_for_trajectory(
                            trajectory, per_frame_objects=per_frame,
                        )
                        visible_objects = _visible_objects_from_controller(controller, candidate)
                    else:
                        visible_objects = _visible_objects_from_controller(controller, candidate)
                        qa_pairs = generate_qa_for_trajectory(
                            trajectory, visible_objects=visible_objects,
                            radius=config.radius, primary_object_type=primary_type,
                        )

                    video_path = trajectory_root / "video.mp4"
                    manifest = {
                        "scene": scene_name,
                        "trajectory": trajectory,
                        "variation": variation.tag,
                        "object_id": candidate.object_id,
                        "object_type": candidate.object_type,
                        "radius": config.radius,
                        "arc_degrees": variation.arc_degrees,
                        "start_angle_offset": variation.start_angle_offset,
                        "increment": config.increment,
                        "fps": config.fps,
                        "total_poses": len(candidate.poses),
                        "saved_frames": saved_frames,
                        "field_of_view": config.field_of_view,
                        "image_size": config.image_size,
                        "pose_source": candidate.pose_source,
                        "video_path": str(video_path),
                        "visible_object_count": len(visible_objects),
                        "qa_count": len(qa_pairs),
                    }
                    (trajectory_root / "manifest.json").write_text(
                        json.dumps(local_storage_utils.sanitize_for_json(manifest), indent=2)
                    )
                    (trajectory_root / "qa.json").write_text(
                        json.dumps(local_storage_utils.sanitize_for_json(qa_pairs), indent=2)
                    )
                    scene_results.append(manifest)
    finally:
        controller.stop()

    return {
        "scene": scene_name,
        "video_count": len(scene_results),
        "total_qa_pairs": sum(r["qa_count"] for r in scene_results),
        "videos": scene_results,
    }


def run_pipeline(config: DatasetConfig) -> dict:
    """Run the full dataset generation pipeline across all scenes."""
    output_root = Path(config.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_scene_results = []
    for scene_name in config.scene_list():
        print(f"\n{'='*60}")
        print(f"  Processing scene: {scene_name}")
        print(f"{'='*60}")
        result = process_scene(scene_name, config)
        all_scene_results.append(result)

    dataset_manifest = {
        "total_scenes": len(all_scene_results),
        "total_videos": sum(r["video_count"] for r in all_scene_results),
        "total_qa_pairs": sum(r["total_qa_pairs"] for r in all_scene_results),
        "config": {
            "radius": config.radius,
            "arc_degrees": config.arc_degrees,
            "increment": config.increment,
            "fps": config.fps,
            "field_of_view": config.field_of_view,
            "image_size": config.image_size,
            "min_frames": config.min_frames,
            "max_objects_per_scene": config.max_objects_per_scene,
            "trajectories": list(config.trajectories),
            "scenes": list(config.scenes),
        },
        "scenes": all_scene_results,
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(local_storage_utils.sanitize_for_json(dataset_manifest), indent=2)
    )
    print(f"\nDataset manifest written to {manifest_path}")
    print(f"  Total scenes:   {dataset_manifest['total_scenes']}")
    print(f"  Total videos:   {dataset_manifest['total_videos']}")
    print(f"  Total QA pairs: {dataset_manifest['total_qa_pairs']}")
    return dataset_manifest
