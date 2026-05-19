"""Top-level benchmark planning orchestration.

Output:
  BenchmarkPlan with selected groups and clip specs ready for rendering.
"""

from collections import Counter

from video_consistency_dataset.benchmark_plan import BenchmarkPlan
from video_consistency_dataset.clip_selection import select_group_clips
from video_consistency_dataset.clip_spec import ClipSpec
from video_consistency_dataset.consistency_group import ConsistencyGroup
from video_consistency_dataset.defaults import CIRCULAR_MOTIONS, group_targets_per_engine
from video_consistency_dataset.group_selection import select_candidates_for_targets


def _group_prefix(candidate: dict) -> str:
    parts = [
        candidate["engine"],
        candidate["motion_family"],
        candidate["question_type"],
    ]
    if candidate["dimension"] is not None:
        parts.append(candidate["dimension"])
    if candidate["radius"] is not None:
        parts.append(str(candidate["radius"]).replace(".", "p"))
    return "_".join(parts)


def _target_key_from_candidate(candidate: dict) -> tuple:
    return (
        candidate["motion_family"],
        candidate["question_type"],
        candidate["dimension"],
        candidate["radius"],
    )


def _target_key(target: dict) -> tuple:
    return (
        target["motion_family"],
        target["question_type"],
        target["dimension"],
        target["radius"],
    )


def _target_from_candidate(candidate: dict) -> dict:
    return {
        "motion_family": candidate["motion_family"],
        "question_type": candidate["question_type"],
        "question_family": candidate["question_family"],
        "dimension": candidate["dimension"],
        "radius": candidate["radius"],
    }


def _interleave_targets_by_question_type(targets: list[dict]) -> list[dict]:
    order = (
        "object_dimensions",
        "object_distance_to_camera",
        "object_pair_distance_center",
    )
    buckets = {question_type: [] for question_type in order}
    for target in targets:
        buckets[target["question_type"]].append(target)
    merged: list[dict] = []
    while any(buckets[question_type] for question_type in order):
        for question_type in order:
            if buckets[question_type]:
                merged.append(buckets[question_type].pop(0))
    return merged


def _adaptive_targets(candidates: list[dict], config) -> list[dict]:
    assert candidates, "No candidates available for planning."
    preferred_targets = group_targets_per_engine(config.thor_radii)
    availability = Counter(_target_key_from_candidate(candidate) for candidate in candidates)
    target_lookup = {}
    for candidate in candidates:
        key = _target_key_from_candidate(candidate)
        if key not in target_lookup:
            target_lookup[key] = _target_from_candidate(candidate)
    selected_targets: list[dict] = []
    used_targets = Counter()
    for target in preferred_targets:
        key = _target_key(target)
        if used_targets[key] < availability[key]:
            selected_targets.append(target)
            used_targets[key] += 1
    ordered_keys = {
        motion_family: sorted(
            [key for key in target_lookup if key[0] == motion_family],
            key=lambda key: (-availability[key], str(key)),
        )
        for motion_family in ("approach", "passby", "around", "spherical", "rotation")
    }
    while len(selected_targets) < len(preferred_targets):
        added = False
        for motion_family in ("approach", "passby", "around", "spherical", "rotation"):
            for key in ordered_keys[motion_family]:
                if used_targets[key] >= availability[key]:
                    continue
                selected_targets.append(target_lookup[key])
                used_targets[key] += 1
                added = True
                break
            if len(selected_targets) >= len(preferred_targets):
                break
        if not added:
            break
    return selected_targets


def _make_clip(
    group_id: str,
    clip_index: int,
    clip_template: dict,
    candidate: dict,
    output_dir: str,
) -> ClipSpec:
    clip_id = f"{group_id}_clip{clip_index:02d}"
    clip_root = f"{output_dir}/videos/{clip_id}"
    return ClipSpec(
        clip_id=clip_id,
        group_id=group_id,
        engine=candidate["engine"],
        scene_id=candidate["scene_id"],
        room_bucket=candidate["room_bucket"],
        motion_family=candidate["motion_family"],
        trajectory=clip_template["trajectory"],
        direction=clip_template["direction"],
        variation_tag=clip_template["variation_tag"],
        question=candidate["question"],
        ground_truth=candidate["ground_truth"],
        question_type=candidate["question_type"],
        question_family=candidate["question_family"],
        dimension=candidate["dimension"],
        radius=clip_template["radius"],
        start_angle_offset=clip_template["start_angle_offset"],
        cycle_count=clip_template["cycle_count"],
        anchor_kind=candidate["anchor_kind"],
        anchor_ids=tuple(candidate["anchor_ids"]),
        anchor_labels=tuple(candidate["anchor_labels"]),
        video_path=f"{clip_root}/video.mp4",
        output_dir=clip_root,
        quality_score=clip_template["quality_score"],
    )


def _materialize_group(candidate: dict, group_index: int, config) -> tuple[ConsistencyGroup, list[ClipSpec]]:
    group_id = f"{_group_prefix(candidate)}_{group_index:03d}"
    allow_radius_variation = candidate["motion_family"] in CIRCULAR_MOTIONS and candidate["question_type"] == "object_dimensions"
    chosen_templates = select_group_clips(
        available_clips=candidate["available_clips"],
        target_group_size=config.group_size,
        min_group_size=config.min_group_size,
        allow_radius_variation=allow_radius_variation,
    )
    clips = [
        _make_clip(group_id, clip_index, clip_template, candidate, config.output_dir)
        for clip_index, clip_template in enumerate(chosen_templates)
    ]
    group = ConsistencyGroup(
        group_id=group_id,
        engine=candidate["engine"],
        scene_id=candidate["scene_id"],
        room_bucket=candidate["room_bucket"],
        motion_family=candidate["motion_family"],
        question_type=candidate["question_type"],
        question_family=candidate["question_family"],
        dimension=candidate["dimension"],
        radius=candidate["radius"],
        anchor_kind=candidate["anchor_kind"],
        anchor_ids=tuple(candidate["anchor_ids"]),
        anchor_labels=tuple(candidate["anchor_labels"]),
        shared_ground_truth=candidate["ground_truth"],
        clip_ids=tuple(clip.clip_id for clip in clips),
        target_group_size=config.group_size,
        selected_group_size=len(clips),
    )
    return group, clips


def _exhaust_all(candidates: list[dict], config, reuse_scores: dict | None = None) -> list[dict]:
    """Use every mined candidate that meets the min_group_size threshold."""
    valid = [
        candidate for candidate in candidates
        if len(candidate["available_clips"]) >= config.min_group_size
    ]
    if config.max_groups_per_scene <= 0:
        return valid
    return _balanced_scene_selection(valid, config.max_groups_per_scene, reuse_scores)


def _balanced_scene_selection(
    candidates: list[dict],
    max_per_scene: int,
    reuse_scores: dict | None = None,
) -> list[dict]:
    """Select up to *max_per_scene* candidates per scene, diversifying motion and question type.

    *reuse_scores* maps candidate index (position in *candidates*) to the
    fraction of its clips that already have rendered videos.  When provided the
    selector prefers candidates with higher reuse to minimize new rendering.
    """
    from collections import defaultdict

    by_scene: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, c in enumerate(candidates):
        by_scene[c["scene_id"]].append((idx, c))

    selected: list[dict] = []
    for scene_id in sorted(by_scene):
        pool = by_scene[scene_id]
        # Sort for diversity: prefer rarer motion families first, then question type variety,
        # break ties with reuse score (higher = more existing videos), then pool size.
        motion_order = {"approach": 0, "passby": 1, "spherical": 2, "around": 3, "rotation": 4}
        qtype_order = {"object_dimensions": 0, "object_distance_to_camera": 1, "object_pair_distance_center": 2}
        _reuse = reuse_scores or {}
        pool.sort(key=lambda ic: (
            motion_order.get(ic[1]["motion_family"], 9),
            qtype_order.get(ic[1]["question_type"], 9),
            -_reuse.get(ic[0], 0.0),
            -len(ic[1]["available_clips"]),
        ))
        # Greedy pick: maximize diversity of (motion_family, question_type) tuples
        chosen: list[dict] = []
        used_keys: set[tuple] = set()
        # First pass: pick one per unique (motion_family, question_type)
        for idx, c in pool:
            if len(chosen) >= max_per_scene:
                break
            key = (c["motion_family"], c["question_type"])
            if key not in used_keys:
                chosen.append(c)
                used_keys.add(key)
        # Second pass: fill remaining slots with different anchor objects
        if len(chosen) < max_per_scene:
            used_anchors = {tuple(c["anchor_ids"]) for c in chosen}
            for idx, c in pool:
                if len(chosen) >= max_per_scene:
                    break
                if c in chosen:
                    continue
                anchor_key = tuple(c["anchor_ids"])
                if anchor_key not in used_anchors:
                    chosen.append(c)
                    used_anchors.add(anchor_key)
        # Third pass: just fill the rest (prefer high reuse)
        if len(chosen) < max_per_scene:
            remaining = [(idx, c) for idx, c in pool if c not in chosen]
            remaining.sort(key=lambda ic: -_reuse.get(ic[0], 0.0))
            for idx, c in remaining:
                if len(chosen) >= max_per_scene:
                    break
                chosen.append(c)
        selected.extend(chosen)
    return selected


def _compute_reuse_scores(candidates: list[dict], reuse_video_dirs: tuple[str, ...]) -> dict[int, float]:
    """Score each candidate by how many of its clips already have rendered videos."""
    if not reuse_video_dirs:
        return {}
    import json
    import os

    # Build index: (scene_id, anchor_ids, motion_family, trajectory, direction, start_angle, cycle_count, radius) -> bool
    existing: set[tuple] = set()
    for d in reuse_video_dirs:
        qa_path = os.path.join(d, "qa.json")
        if not os.path.exists(qa_path):
            continue
        with open(qa_path) as f:
            qa = json.load(f)
        for entry in qa:
            questions = entry.get("questions", [])
            anchor_ids = tuple(questions[0]["anchor_ids"]) if questions else ()
            key = (
                entry["scene_id"], anchor_ids, entry["motion_family"], entry["trajectory"],
                entry["direction"], entry.get("start_angle_offset", 0), entry.get("cycle_count", 0),
                entry.get("radius"),
            )
            clip_id = entry["clip_id"]
            video_path = os.path.join(d, "videos", clip_id, "video.mp4")
            if os.path.exists(video_path):
                existing.add(key)
    print(f"[plan] reuse index: {len(existing)} existing rendered clips across {len(reuse_video_dirs)} dirs", flush=True)

    scores: dict[int, float] = {}
    for idx, cand in enumerate(candidates):
        matched = 0
        for clip in cand["available_clips"]:
            key = (
                cand["scene_id"], tuple(cand["anchor_ids"]), cand["motion_family"], clip["trajectory"],
                clip["direction"], clip.get("start_angle_offset", 0), clip.get("cycle_count", 0),
                clip.get("radius"),
            )
            if key in existing:
                matched += 1
        scores[idx] = matched / max(len(cand["available_clips"]), 1)
    return scores


def build_benchmark_plan(config, thor_candidates: list[dict], interiorgs_candidates: list[dict]) -> BenchmarkPlan:
    config.validate()
    reuse_scores_thor = _compute_reuse_scores(thor_candidates, config.reuse_video_dirs) if config.reuse_video_dirs else None
    selected_candidates: list[dict] = []
    if "thor" in config.enabled_engines:
        if config.exhaust_all_candidates:
            selected_candidates.extend(_exhaust_all(thor_candidates, config, reuse_scores_thor))
        else:
            thor_targets = _interleave_targets_by_question_type(_adaptive_targets(thor_candidates, config))
            selected_candidates.extend(select_candidates_for_targets(thor_candidates, thor_targets))
    if "interiorgs" in config.enabled_engines:
        if config.exhaust_all_candidates:
            selected_candidates.extend(_exhaust_all(interiorgs_candidates, config))
        else:
            interior_targets = _interleave_targets_by_question_type(_adaptive_targets(interiorgs_candidates, config))
            selected_candidates.extend(select_candidates_for_targets(interiorgs_candidates, interior_targets))
    groups: list[ConsistencyGroup] = []
    clips: list[ClipSpec] = []
    for group_index, candidate in enumerate(selected_candidates, start=1):
        group, group_clips = _materialize_group(candidate, group_index, config)
        groups.append(group)
        clips.extend(group_clips)
    return BenchmarkPlan(config=config.to_dict(), groups=tuple(groups), clips=tuple(clips))
