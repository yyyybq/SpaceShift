"""THOR candidate mining for the video consistency benchmark.

Candidate schema:
  {
    "engine": "thor",
    "motion_family": "around",
    "question_type": "object_dimensions",
    "anchor_ids": ["Chair|..."],
    "available_clips": [{...}]
  }
"""

from tqdm import tqdm

from trajectory_demos.controller_utils import build_controller, prepare_scene
from trajectory_demos.demo_config import TrajectoryDemoConfig
from trajectory_demos.selector import _score_candidate
from video_qa_dataset.dataset_pipeline import _per_frame_visible_objects
from video_qa_dataset.variation_builder import build_varied_candidates
from video_qa_dataset.variation_config import ORBIT_ARC_DEGREES, direction_from_trajectory
from video_consistency_dataset.benchmark_variations import benchmark_variations
from video_consistency_dataset.defaults import CIRCULAR_MOTIONS, DIMENSIONS, LINEAR_DIRECTIONS, OBJECT_MOTIONS, ORBIT_DIRECTIONS
from video_consistency_dataset.question_entries import (
    build_object_dimension_entry,
    build_object_distance_entry,
    build_object_pair_distance_entry,
)
from video_consistency_dataset.thor_common import object_by_id, object_center_distance, object_dimensions, room_bucket_for_thor_scene


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


def _clip_template(trajectory: str, variation, radius: float | None, quality_score: float) -> dict:
    return {
        "trajectory": trajectory,
        "direction": direction_from_trajectory(trajectory),
        "variation_tag": variation.tag,
        "radius": radius,
        "start_angle_offset": variation.start_angle_offset,
        "cycle_count": variation.bounce_count,
        "quality_score": quality_score,
    }


def _object_candidates(scene_id: str, config, controller, objects: list[dict]) -> list[dict]:
    size_pools: dict[tuple[str, str], list[dict]] = {}
    distance_pools: dict[tuple[str, str, float], list[dict]] = {}
    for motion_family in OBJECT_MOTIONS:
        radii = config.thor_radii if motion_family in CIRCULAR_MOTIONS else (config.linear_radius,)
        directions = ORBIT_DIRECTIONS[motion_family] if motion_family in CIRCULAR_MOTIONS else LINEAR_DIRECTIONS[motion_family]
        trajectories = [f"{motion_family}_{direction}" for direction in directions]
        for radius in radii:
            demo_config = _thor_config(config, scene_id, radius, motion_family)
            for trajectory in trajectories:
                for variation in benchmark_variations(config, motion_family):
                    candidates = build_varied_candidates(scene_id, controller, demo_config, trajectory, variation)
                    valid = [_score_candidate(controller, candidate) for candidate in candidates]
                    valid = [candidate for candidate in valid if candidate.saved_frames >= config.min_frames]
                    for candidate in valid:
                        template = _clip_template(trajectory, variation, radius, candidate.quality_score)
                        size_pools.setdefault((motion_family, candidate.object_id), []).append(template)
                        if motion_family in CIRCULAR_MOTIONS:
                            distance_pools.setdefault((motion_family, candidate.object_id, radius), []).append(template)
    room_bucket = room_bucket_for_thor_scene(scene_id)
    mined: list[dict] = []
    for (motion_family, object_id), templates in size_pools.items():
        if len(templates) < config.min_group_size:
            continue
        obj = object_by_id(objects, object_id)
        values = object_dimensions(obj)
        for dimension, value in zip(DIMENSIONS, values):
            entry = build_object_dimension_entry(obj["objectType"], dimension, value)
            mined.append({
                "engine": "thor",
                "scene_id": scene_id,
                "room_bucket": room_bucket,
                "motion_family": motion_family,
                "question": entry["question"],
                "ground_truth": entry["ground_truth"],
                "question_type": entry["question_type"],
                "question_family": entry["question_family"],
                "dimension": dimension,
                "radius": None,
                "anchor_kind": "object",
                "anchor_ids": (object_id,),
                "anchor_labels": (obj["objectType"],),
                "available_clips": templates,
            })
    for (motion_family, object_id, radius), templates in distance_pools.items():
        if len(templates) < config.min_group_size:
            continue
        obj = object_by_id(objects, object_id)
        entry = build_object_distance_entry(obj["objectType"], radius)
        mined.append({
            "engine": "thor",
            "scene_id": scene_id,
            "room_bucket": room_bucket,
            "motion_family": motion_family,
            "question": entry["question"],
            "ground_truth": entry["ground_truth"],
            "question_type": entry["question_type"],
            "question_family": entry["question_family"],
            "dimension": None,
            "radius": radius,
            "anchor_kind": "object",
            "anchor_ids": (object_id,),
            "anchor_labels": (obj["objectType"],),
            "available_clips": templates,
        })
    return mined


def _rotation_candidates(scene_id: str, config, controller, objects: list[dict]) -> list[dict]:
    pools: dict[tuple[str, str], list[dict]] = {}
    pair_labels: dict[tuple[str, str], tuple[str, str]] = {}
    demo_config = _thor_config(config, scene_id, config.linear_radius, "rotation")
    for trajectory in ("rotation_cw", "rotation_ccw"):
        for variation in benchmark_variations(config, "rotation"):
            candidates = build_varied_candidates(scene_id, controller, demo_config, trajectory, variation)
            valid = [_score_candidate(controller, candidate) for candidate in candidates]
            valid = [candidate for candidate in valid if candidate.saved_frames >= config.min_frames]
            assert len(valid) <= 1, "Rotation mining expects one room candidate per variation."
            for candidate in valid:
                template = _clip_template(trajectory, variation, None, candidate.quality_score)
                per_frame = _per_frame_visible_objects(controller, candidate)
                pair_counts: dict[tuple[str, str], int] = {}
                for frame_objects in per_frame:
                    for index, obj_a in enumerate(frame_objects):
                        for obj_b in frame_objects[index + 1:]:
                            if obj_a["objectType"] == obj_b["objectType"]:
                                continue
                            pair_id = tuple(sorted((obj_a["objectId"], obj_b["objectId"])))
                            pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
                            pair_labels[pair_id] = tuple(sorted((obj_a["objectType"], obj_b["objectType"])))
                for pair_id, count in pair_counts.items():
                    if count >= 3:
                        pools.setdefault(pair_id, []).append(template)
    room_bucket = room_bucket_for_thor_scene(scene_id)
    mined: list[dict] = []
    for pair_id, templates in pools.items():
        if len(templates) < config.min_group_size:
            continue
        obj_a = object_by_id(objects, pair_id[0])
        obj_b = object_by_id(objects, pair_id[1])
        entry = build_object_pair_distance_entry(pair_labels[pair_id][0], pair_labels[pair_id][1], object_center_distance(obj_a, obj_b))
        mined.append({
            "engine": "thor",
            "scene_id": scene_id,
            "room_bucket": room_bucket,
            "motion_family": "rotation",
            "question": entry["question"],
            "ground_truth": entry["ground_truth"],
            "question_type": entry["question_type"],
            "question_family": entry["question_family"],
            "dimension": None,
            "radius": None,
            "anchor_kind": "pair",
            "anchor_ids": pair_id,
            "anchor_labels": pair_labels[pair_id],
            "available_clips": templates,
        })
    return mined


def mine_thor_single_scene(config, scene_id: str, quiet: bool = False) -> list[dict]:
    import time
    t0 = time.monotonic()
    controller = build_controller(scene_id, config.gpu, config.image_size, config.field_of_view)
    try:
        prepare_scene(controller, scene_id)
        objects = list(controller.last_event.metadata["objects"])
        if not quiet:
            print(f"[thor] {scene_id}: {len(objects)} objects, mining object candidates...", flush=True)
        object_candidates = _object_candidates(scene_id, config, controller, objects)
        if not quiet:
            print(f"[thor] {scene_id}: object candidates={len(object_candidates)}, mining rotation...", flush=True)
        rotation_candidates = _rotation_candidates(scene_id, config, controller, objects)
        elapsed = time.monotonic() - t0
        if not quiet:
            print(
                f"[thor] finished {scene_id}: objects={len(object_candidates)} "
                f"rotation={len(rotation_candidates)} elapsed={elapsed:.0f}s",
                flush=True,
            )
        return object_candidates + rotation_candidates
    finally:
        controller.stop()


def mine_thor_candidates(config) -> list[dict]:
    if config.plan_parallel_workers > 1:
        from video_consistency_dataset.plan_parallel import parallel_mine_thor

        return parallel_mine_thor(config)
    mined: list[dict] = []
    total_scenes = len(config.thor_scenes)
    for scene_index, scene_id in enumerate(
        tqdm(config.thor_scenes, desc="THOR plan", unit="scene", dynamic_ncols=True),
        start=1,
    ):
        print(f"[thor] mining scene {scene_index}/{total_scenes}: {scene_id}", flush=True)
        mined.extend(mine_thor_single_scene(config, scene_id))
    print(f"[thor] mined total candidates={len(mined)}", flush=True)
    return mined
