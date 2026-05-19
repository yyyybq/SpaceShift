"""Per-scene InteriorGS mining core (shared by sequential and parallel plan stages)."""

import time
from pathlib import Path

import global_config
from video_consistency_dataset.benchmark_variations import benchmark_variations
from video_consistency_dataset.defaults import (
    CIRCULAR_MOTIONS,
    DIMENSIONS,
    LINEAR_DIRECTIONS,
    OBJECT_MOTIONS,
    ORBIT_DIRECTIONS,
)
from video_consistency_dataset.interiorgs_sequences import build_object_sequence, build_rotation_sequence, build_scene_context
from video_consistency_dataset.question_entries import (
    build_object_dimension_entry,
    build_object_distance_entry,
    build_object_pair_distance_entry,
)


def _room_bucket(scene_id: str, room_index: int | None) -> str:
    return f"{scene_id}:room_{room_index if room_index is not None else 'unknown'}"


_SKIP_LABEL_LOWER = frozenset(s.strip().lower() for s in global_config.IGNORE_OBJECTS_GLOBAL)


def _skip_interior_object_label(label: str) -> bool:
    return (label or "").strip().lower() in _SKIP_LABEL_LOWER


def _limit_mining_objects(single_objects: list, max_objects: int) -> list:
    if max_objects <= 0 or len(single_objects) <= max_objects:
        return single_objects
    return single_objects[:max_objects]


def _subsample_rotation_poses_for_pairs(sequence: list, max_poses: int) -> list:
    if len(sequence) <= max_poses:
        return sequence
    step = max(1, len(sequence) // max_poses)
    out = sequence[::step]
    return out[:max_poses] if len(out) > max_poses else out


def _clip_template(trajectory: str, variation, radius: float | None, quality_score: float) -> dict:
    return {
        "trajectory": trajectory,
        "direction": trajectory.rsplit("_", 1)[1],
        "variation_tag": variation.tag,
        "radius": radius,
        "start_angle_offset": variation.start_angle_offset,
        "cycle_count": variation.bounce_count,
        "quality_score": quality_score,
    }


def _mine_interiorgs_scene_core(
    config,
    scene_id: str,
    scene_index: int,
    total_scenes: int,
    symbols,
    selector,
    sampler,
    quiet: bool = False,
) -> list[dict]:
    get_visible_objects = symbols["get_visible_objects"]
    scene_object_to_aabb = symbols["scene_object_to_aabb"]
    scene_t0 = time.perf_counter()
    if not quiet:
        print(f"[plan] InteriorGS: scene {scene_index}/{total_scenes} {scene_id} ...", flush=True)
    scene_path = Path(config.interiorgs_root) / scene_id
    single_objects = selector.select_single_objects(scene_path)
    mining_objects = _limit_mining_objects(single_objects, config.interiorgs_mining_max_objects)
    mining_objects = [o for o in mining_objects if not _skip_interior_object_label(o.label)]
    all_scene_objects = selector.get_all_parsed_objects(scene_path, include_walls=False)
    context = build_scene_context(sampler, scene_path, all_scene_objects, scene_object_to_aabb)
    if not quiet:
        print(
            f"[plan] InteriorGS: scene {scene_id} ... objects {len(mining_objects)}/{len(single_objects)}, "
            f"context ready (+{time.perf_counter() - scene_t0:.1f}s)",
            flush=True,
        )
    size_pools: dict[tuple[str, str], list[dict]] = {}
    distance_pools: dict[tuple[str, str, float], list[dict]] = {}
    for motion_family in OBJECT_MOTIONS:
        motion_t0 = time.perf_counter()
        radii = config.thor_radii if motion_family in CIRCULAR_MOTIONS else (config.linear_radius,)
        directions = (
            ORBIT_DIRECTIONS[motion_family] if motion_family in CIRCULAR_MOTIONS else LINEAR_DIRECTIONS[motion_family]
        )
        for radius in radii:
            for direction in directions:
                trajectory = f"{motion_family}_{direction}"
                pose_inc = config.circular_increment if motion_family in CIRCULAR_MOTIONS else config.increment
                for variation in benchmark_variations(config, motion_family):
                    for obj in mining_objects:
                        sequence = build_object_sequence(
                            sampler,
                            symbols["CameraPose"],
                            scene_path,
                            obj,
                            context,
                            motion_family,
                            direction,
                            radius,
                            variation.start_angle_offset,
                            variation.bounce_count,
                            pose_inc,
                            config.interiorgs_linear_steps,
                            config.interiorgs_half_span,
                        )
                        if not sequence:
                            continue
                        template = _clip_template(trajectory, variation, radius, float(len(sequence)))
                        size_pools.setdefault((motion_family, obj.id), []).append(template)
                        if motion_family in CIRCULAR_MOTIONS:
                            distance_pools.setdefault((motion_family, obj.id, radius), []).append(template)
        if not quiet:
            print(
                f"[plan] InteriorGS: scene {scene_id} ... {motion_family} (+{time.perf_counter() - motion_t0:.1f}s, "
                f"elapsed {time.perf_counter() - scene_t0:.1f}s)",
                flush=True,
            )
    scene_mined: list[dict] = []
    for (motion_family, object_id), templates in size_pools.items():
        if len(templates) < config.min_group_size:
            continue
        obj = [item for item in single_objects if item.id == object_id][0]
        for dimension, value in zip(DIMENSIONS, obj.get_obb_size()):
            entry = build_object_dimension_entry(obj.label, dimension, value)
            scene_mined.append(
                {
                    "engine": "interiorgs",
                    "scene_id": scene_id,
                    "room_bucket": _room_bucket(scene_id, obj.room_index),
                    "motion_family": motion_family,
                    "question": entry["question"],
                    "ground_truth": entry["ground_truth"],
                    "question_type": entry["question_type"],
                    "question_family": entry["question_family"],
                    "dimension": dimension,
                    "radius": None,
                    "anchor_kind": "object",
                    "anchor_ids": (obj.id,),
                    "anchor_labels": (obj.label,),
                    "available_clips": templates,
                }
            )
    for (motion_family, object_id, radius), templates in distance_pools.items():
        if len(templates) < config.min_group_size:
            continue
        obj = [item for item in single_objects if item.id == object_id][0]
        entry = build_object_distance_entry(obj.label, radius)
        scene_mined.append(
            {
                "engine": "interiorgs",
                "scene_id": scene_id,
                "room_bucket": _room_bucket(scene_id, obj.room_index),
                "motion_family": motion_family,
                "question": entry["question"],
                "ground_truth": entry["ground_truth"],
                "question_type": entry["question_type"],
                "question_family": entry["question_family"],
                "dimension": None,
                "radius": radius,
                "anchor_kind": "object",
                "anchor_ids": (obj.id,),
                "anchor_labels": (obj.label,),
                "available_clips": templates,
            }
        )
    rotation_pools: dict[tuple[int, str, str], list[dict]] = {}
    room_indices = sorted({obj.room_index for obj in all_scene_objects if obj.room_index is not None})
    for room_index in room_indices:
        room_t0 = time.perf_counter()
        for direction in ORBIT_DIRECTIONS["rotation"]:
            trajectory = f"rotation_{direction}"
            for variation in benchmark_variations(config, "rotation"):
                sequence = build_rotation_sequence(
                    sampler,
                    scene_path,
                    all_scene_objects,
                    room_index,
                    direction,
                    variation.start_angle_offset,
                    variation.bounce_count,
                )
                if not sequence:
                    continue
                template = _clip_template(trajectory, variation, None, float(len(sequence)))
                pair_counts: dict[tuple[int, str, str], int] = {}
                poses_for_pairs = _subsample_rotation_poses_for_pairs(
                    sequence,
                    config.interiorgs_rotation_pair_pose_samples,
                )
                for pose in poses_for_pairs:
                    visible = get_visible_objects(
                        all_scene_objects,
                        pose,
                        sampler.intrinsics,
                        sampler.config.image_width,
                        sampler.config.image_height,
                        context["all_aabbs"],
                        True,
                        sampler.config.min_visible_corners,
                    )
                    for index, obj_a in enumerate(visible):
                        for obj_b in visible[index + 1:]:
                            is_valid, _ = selector.filter_object_pair(obj_a, obj_b)
                            if is_valid and obj_a.label != obj_b.label:
                                if _skip_interior_object_label(obj_a.label) or _skip_interior_object_label(obj_b.label):
                                    continue
                                pair_key = (room_index, *sorted((obj_a.id, obj_b.id)))
                                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
                for pair_key, count in pair_counts.items():
                    if count >= 3:
                        rotation_pools.setdefault(pair_key, []).append(template)
        if not quiet:
            print(
                f"[plan] InteriorGS: scene {scene_id} ... rotation room {room_index} "
                f"(+{time.perf_counter() - room_t0:.1f}s, elapsed {time.perf_counter() - scene_t0:.1f}s)",
                flush=True,
            )
    for (room_index, object_a_id, object_b_id), templates in rotation_pools.items():
        if len(templates) < config.min_group_size:
            continue
        obj_a = [item for item in all_scene_objects if item.id == object_a_id][0]
        obj_b = [item for item in all_scene_objects if item.id == object_b_id][0]
        entry = build_object_pair_distance_entry(
            obj_a.label,
            obj_b.label,
            float(((obj_a.center - obj_b.center) ** 2).sum() ** 0.5),
        )
        scene_mined.append(
            {
                "engine": "interiorgs",
                "scene_id": scene_id,
                "room_bucket": _room_bucket(scene_id, room_index),
                "motion_family": "rotation",
                "question": entry["question"],
                "ground_truth": entry["ground_truth"],
                "question_type": entry["question_type"],
                "question_family": entry["question_family"],
                "dimension": None,
                "radius": None,
                "anchor_kind": "pair",
                "anchor_ids": tuple(sorted((object_a_id, object_b_id))),
                "anchor_labels": tuple(sorted((obj_a.label, obj_b.label))),
                "available_clips": templates,
            }
        )
    if not quiet:
        print(
            f"[plan] InteriorGS: scene {scene_id} done in {time.perf_counter() - scene_t0:.1f}s "
            f"(mining_objects={len(mining_objects)}/{len(single_objects)})",
            flush=True,
        )
    return scene_mined
