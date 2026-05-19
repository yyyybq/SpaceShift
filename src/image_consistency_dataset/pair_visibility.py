"""Backfill frame_visibility.json for pair-distance clips that lack it.

Re-opens the THOR simulator, replays poses, and writes the sidecar
without re-rendering images.  Works from qa.json (no metadata.json required).
"""

import json
from collections.abc import Collection
from pathlib import Path

from tqdm import tqdm

import global_config
from trajectory_demos.controller_utils import build_controller, prepare_scene, teleport_pose
from trajectory_demos.demo_config import TrajectoryDemoConfig
from trajectory_demos.frame_quality import bbox_area_ratio, object_bbox
from trajectory_demos.selector import _score_candidate
from video_consistency_dataset.benchmark_variations import variation_from_clip
from video_consistency_dataset.clip_spec import ClipSpec
from video_consistency_dataset.defaults import CIRCULAR_MOTIONS
from video_qa_dataset.variation_builder import build_varied_candidates
from video_qa_dataset.variation_config import ORBIT_ARC_DEGREES


# Default render settings matching BenchmarkConfig defaults
_IMAGE_SIZE = 384
_FOV = 75
_FPS = 1
_MIN_FRAMES = 30
_INCREMENT = 10.0
_CIRCULAR_INCREMENT = 6.0
_LINEAR_RADIUS = 1.0


def _clip_from_qa_entry(entry: dict, video_root: Path) -> ClipSpec:
    """Reconstruct a ClipSpec from a qa.json entry."""
    q = entry["questions"][0]
    clip_dir = video_root / entry["clip_id"]
    return ClipSpec(
        clip_id=entry["clip_id"],
        group_id=entry["group_id"],
        engine=entry["engine"],
        scene_id=entry["scene_id"],
        room_bucket=entry.get("room_bucket", ""),
        motion_family=entry["motion_family"],
        trajectory=entry["trajectory"],
        direction=entry["direction"],
        variation_tag=entry["variation_tag"],
        question=q["question"],
        ground_truth=q["answer"],
        question_type=q["question_type"],
        question_family=q["question_family"],
        dimension=q.get("dimension"),
        radius=entry.get("radius"),
        start_angle_offset=entry["start_angle_offset"],
        cycle_count=entry["cycle_count"],
        anchor_kind=q["anchor_kind"],
        anchor_ids=tuple(q["anchor_ids"]),
        anchor_labels=tuple(q["anchor_labels"]),
        video_path=entry["video_path"],
        output_dir=str(clip_dir),
        quality_score=entry.get("quality_score", 0.0),
    )


def _needs_backfill(clip: ClipSpec) -> bool:
    if clip.anchor_kind != "pair":
        return False
    clip_root = Path(clip.output_dir)
    # Historical source clips may retain only video.mp4 after frame cleanup.
    if not (clip_root / "video.mp4").exists() and not (clip_root / "frames").is_dir():
        return False
    return not (clip_root / "frame_visibility.json").exists()


def _thor_config(scene_id: str, radius: float, motion_family: str, gpu: int) -> TrajectoryDemoConfig:
    increment = _CIRCULAR_INCREMENT if motion_family in CIRCULAR_MOTIONS else _INCREMENT
    arc = ORBIT_ARC_DEGREES if motion_family in CIRCULAR_MOTIONS else 360.0
    return TrajectoryDemoConfig(
        scene=scene_id,
        output_dir="/tmp",
        trajectory="all",
        object_index=None,
        object_id=None,
        object_type=None,
        radius=radius,
        arc_degrees=arc,
        increment=increment,
        field_of_view=_FOV,
        image_size=_IMAGE_SIZE,
        fps=_FPS,
        gpu=gpu,
        min_frames=_MIN_FRAMES,
        candidate_scenes=(),
    )


def _write_visibility_sidecar(controller, candidate, clip: ClipSpec, clip_root: Path) -> None:
    """Check anchor visibility at each rendered frame pose and write sidecar."""
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


def backfill_thor_visibility(
    video_output_dir: str,
    gpu: int = 0,
    clip_ids: Collection[str] | None = None,
) -> int:
    """Backfill frame_visibility.json for all THOR pair clips missing it.

    Reads qa.json from video_output_dir. No metadata.json required.
    Returns number of clips backfilled.
    """
    video_dir = Path(video_output_dir)
    qa_path = video_dir / "qa.json"
    assert qa_path.exists(), f"Missing {qa_path}"
    qa_entries = json.loads(qa_path.read_text())
    video_root = video_dir / "videos"

    clips = [_clip_from_qa_entry(e, video_root) for e in qa_entries if e["engine"] == "thor"]

    pending_by_scene: dict[str, list[ClipSpec]] = {}
    for clip in clips:
        if clip_ids is not None and clip.clip_id not in clip_ids:
            continue
        if not _needs_backfill(clip):
            continue
        pending_by_scene.setdefault(clip.scene_id, []).append(clip)

    total = sum(len(v) for v in pending_by_scene.values())
    if total == 0:
        print("[backfill] no THOR pair clips need visibility backfill", flush=True)
        return 0

    print(f"[backfill] {total} THOR pair clips across {len(pending_by_scene)} scenes", flush=True)
    done = 0
    for scene_id, scene_clips in tqdm(pending_by_scene.items(), desc="backfill scenes"):
        controller = build_controller(scene_id, gpu, _IMAGE_SIZE, _FOV)
        try:
            prepare_scene(controller, scene_id)
            for clip in scene_clips:
                variation = variation_from_clip(clip)
                radius = clip.radius if clip.radius is not None else _LINEAR_RADIUS
                demo_config = _thor_config(scene_id, radius, clip.motion_family, gpu)
                candidates = build_varied_candidates(
                    scene_id, controller, demo_config, clip.trajectory, variation, None
                )
                scored = [_score_candidate(controller, c) for c in candidates]
                valid = [c for c in scored if c.saved_frames >= _MIN_FRAMES]
                if not valid:
                    print(f"  WARN: no valid candidate for {clip.clip_id}, skipping", flush=True)
                    continue
                candidate = max(valid, key=lambda c: (c.quality_score, c.saved_frames))
                _write_visibility_sidecar(controller, candidate, clip, Path(clip.output_dir))
                done += 1
        finally:
            controller.stop()

    print(f"[backfill] wrote {done}/{total} visibility sidecars", flush=True)
    return done
