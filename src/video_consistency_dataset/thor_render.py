"""Render THOR benchmark clips from a planned manifest.

Output per clip:
  videos/<clip_id>/
    frames/
    video.mp4
    frame_visibility.json   (pair clips only)
"""

import json
import shutil
from pathlib import Path

from tqdm import tqdm

from trajectory_demos.controller_utils import build_controller, prepare_scene, teleport_pose
from trajectory_demos.demo_config import TrajectoryDemoConfig
from trajectory_demos.frame_quality import bbox_area_ratio, object_bbox
from trajectory_demos.selector import _score_candidate
from video_qa_dataset.dataset_pipeline import _render_candidate
from video_qa_dataset.variation_builder import build_varied_candidates
from video_qa_dataset.variation_config import ORBIT_ARC_DEGREES
from video_consistency_dataset.benchmark_variations import variation_from_clip
from video_consistency_dataset.defaults import CIRCULAR_MOTIONS

import global_config


def _write_pair_visibility(controller, candidate, clip, clip_root: Path) -> None:
    """Record per-frame anchor visibility for pair-distance clips."""
    anchor_set = set(clip.anchor_ids)
    frames: dict[str, list[str]] = {}
    frame_index = 0
    for pose_index in sorted(candidate.visible_pose_indices):
        pose = candidate.poses[pose_index]
        if not teleport_pose(controller, pose):
            frames[f"frame_{frame_index:05d}"] = []
            frame_index += 1
            continue
        visible_ids = []
        for obj in controller.last_event.metadata["objects"]:
            if obj["objectId"] not in anchor_set:
                continue
            if not obj.get("visible", False):
                continue
            if obj["objectType"].lower() in global_config.IGNORE_OBJECTS_GLOBAL:
                continue
            bbox = object_bbox(controller, obj["objectId"])
            if bbox is None:
                continue
            if bbox_area_ratio(controller, bbox) < 1.0 / 400.0:
                continue
            visible_ids.append(obj["objectId"])
        frames[f"frame_{frame_index:05d}"] = visible_ids
        frame_index += 1
    sidecar = {"anchor_ids": list(clip.anchor_ids), "frames": frames}
    (clip_root / "frame_visibility.json").write_text(json.dumps(sidecar, indent=2))


def _thor_config(config, scene_id: str, radius: float, motion_family: str) -> TrajectoryDemoConfig:
    increment = config.circular_increment if motion_family in CIRCULAR_MOTIONS else config.increment
    arc = ORBIT_ARC_DEGREES if motion_family in CIRCULAR_MOTIONS else 360.0
    return TrajectoryDemoConfig(
        scene=scene_id,
        output_dir=config.output_dir,
        trajectory="all",
        object_index=None,
        object_id=None,
        object_type=None,
        radius=radius,
        arc_degrees=arc,
        increment=increment,
        field_of_view=config.field_of_view,
        image_size=config.image_size,
        fps=config.fps,
        gpu=config.gpu,
        min_frames=config.min_frames,
        candidate_scenes=config.thor_scenes,
    )

def render_thor_clips(plan, config, clip_ids: set[str] | None = None) -> dict[str, dict]:
    clips_by_scene: dict[str, list] = {}
    render_records: dict[str, dict] = {}
    for clip in plan.clips:
        if clip.engine != "thor":
            continue
        if clip_ids is not None and clip.clip_id not in clip_ids:
            continue
        clips_by_scene.setdefault(clip.scene_id, []).append(clip)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for scene_id, scene_clips in clips_by_scene.items():
        pending = [c for c in scene_clips if not (Path(c.output_dir) / "video.mp4").exists()]
        if not pending:
            for c in scene_clips:
                render_records[c.clip_id] = {"skipped": True}
            continue
        controller = build_controller(scene_id, config.gpu, config.image_size, config.field_of_view)
        try:
            prepare_scene(controller, scene_id)
            for clip in pending:
                variation = variation_from_clip(clip)
                radius = clip.radius if clip.radius is not None else config.linear_radius
                demo_config = _thor_config(config, scene_id, radius, clip.motion_family)
                target_object_ids = list(clip.anchor_ids) if clip.anchor_kind == "object" else None
                candidates = build_varied_candidates(scene_id, controller, demo_config, clip.trajectory, variation, target_object_ids)
                scored = [_score_candidate(controller, candidate) for candidate in candidates]
                valid = [candidate for candidate in scored if candidate.saved_frames >= config.min_frames]
                if clip.anchor_kind == "object":
                    valid = [candidate for candidate in valid if candidate.object_id == clip.anchor_ids[0]]
                if not valid:
                    print(f"[thor] WARN: no valid candidate for {clip.clip_id}, skipping", flush=True)
                    continue
                candidate = max(valid, key=lambda item: (item.quality_score, item.saved_frames))
                clip_root = Path(clip.output_dir)
                if clip_root.exists():
                    shutil.rmtree(clip_root)
                clip_root.mkdir(parents=True, exist_ok=True)
                saved_frames = _render_candidate(
                    controller,
                    candidate,
                    clip_root,
                    output_root,
                    scene_id,
                    config.fps,
                    dedupe_identical_frames=False,
                )
                if clip.anchor_kind == "pair":
                    _write_pair_visibility(controller, candidate, clip, clip_root)
                render_records[clip.clip_id] = {
                    "saved_frames": saved_frames,
                    "total_poses": len(candidate.poses),
                    "pose_source": candidate.pose_source,
                }
        finally:
            controller.stop()
    return render_records
