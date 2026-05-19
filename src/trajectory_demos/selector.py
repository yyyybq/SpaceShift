"""Scene and object selection for curated trajectory demos."""

from dataclasses import replace

from trajectory_demos.controller_utils import (
    build_controller,
    object_is_visible,
    teleport_pose,
)
from trajectory_demos.frame_quality import (
    bbox_area_ratio,
    bbox_center_ratio,
    longest_contiguous_indices,
    monotonic_fraction,
    normalized_span,
    object_bbox,
)
from trajectory_demos.pattern_builders import build_candidates
from trajectory_demos.scene_selection import SceneSelection
from trajectory_demos.trajectory_candidate import TrajectoryCandidate

MIN_BBOX_AREA_RATIO = 1.0 / 400.0


def _filter_candidates(
    candidates: list[TrajectoryCandidate],
    config,
    trajectory: str,
) -> list[TrajectoryCandidate]:
    if trajectory.startswith("rotation_"):
        return candidates
    filtered = candidates
    if config.object_id is not None:
        filtered = [candidate for candidate in filtered if candidate.object_id == config.object_id]
    elif config.object_type is not None:
        filtered = [
            candidate
            for candidate in filtered
            if candidate.object_type.lower() == config.object_type.lower()
        ]
    if config.object_index is not None and config.object_id is None and config.object_type is None:
        assert 0 <= config.object_index < len(filtered), (
            f"object_index {config.object_index} is out of range ({len(filtered)} candidates)."
        )
        return [filtered[config.object_index]]
    return filtered


def _pattern_name(candidate: TrajectoryCandidate) -> str:
    return candidate.trajectory.rsplit("_", 1)[0]


def _direction_name(candidate: TrajectoryCandidate) -> str:
    return candidate.trajectory.rsplit("_", 1)[1]


def _bbox_is_usable(candidate: TrajectoryCandidate, area_ratio: float, center_x: float, center_y: float) -> bool:
    pattern = _pattern_name(candidate)
    if pattern == "passby":
        if area_ratio < (1.0 / 700.0):
            return False
        return 0.02 <= center_x <= 0.98 and abs(center_y - 0.5) <= 0.4
    if area_ratio < MIN_BBOX_AREA_RATIO:
        return False
    if pattern == "spherical":
        return abs(center_x - 0.5) <= 0.28 and abs(center_y - 0.5) <= 0.28
    if pattern == "approach":
        return abs(center_x - 0.5) <= 0.2 and abs(center_y - 0.5) <= 0.25
    return True


def _quality_score(candidate: TrajectoryCandidate, area_ratios: list[float], center_xs: list[float]) -> float:
    pattern = _pattern_name(candidate)
    if not area_ratios:
        return 0.0
    average_area = sum(area_ratios) / len(area_ratios)
    if pattern == "approach":
        is_forward = _direction_name(candidate) == "fw"
        return average_area * 1000.0 + monotonic_fraction(area_ratios, increasing=is_forward) * 100.0
    if pattern == "passby":
        is_backward = _direction_name(candidate) == "bw"
        return (
            average_area * 1000.0
            + monotonic_fraction(center_xs, increasing=is_backward) * 100.0
            + normalized_span(center_xs) * 100.0
        )
    if pattern == "spherical":
        center_penalty = sum(abs(center_x - 0.5) for center_x in center_xs) / len(center_xs)
        return average_area * 1000.0 + (1.0 - center_penalty) * 100.0
    return average_area * 1000.0


def _requires_full_visibility(candidate: TrajectoryCandidate) -> bool:
    return _pattern_name(candidate) == "around"


def _score_candidate(controller, candidate: TrajectoryCandidate) -> TrajectoryCandidate:
    if candidate.object_id == "room":
        return replace(
            candidate,
            visible_pose_indices=tuple(range(len(candidate.poses))),
            quality_score=float(len(candidate.poses)),
        )
    valid_pose_indices = []
    metrics_by_index: dict[int, tuple[float, float]] = {}
    for pose_index, pose in enumerate(candidate.poses):
        if not teleport_pose(controller, pose):
            continue
        if not object_is_visible(controller, candidate.object_id):
            continue
        bbox = object_bbox(controller, candidate.object_id)
        if bbox is None:
            continue
        area_ratio = bbox_area_ratio(controller, bbox)
        center_x, center_y = bbox_center_ratio(controller, bbox)
        if not _bbox_is_usable(candidate, area_ratio, center_x, center_y):
            continue
        valid_pose_indices.append(pose_index)
        metrics_by_index[pose_index] = (area_ratio, center_x)
    if _requires_full_visibility(candidate):
        visible_pose_indices = tuple(range(len(candidate.poses)))
        if tuple(valid_pose_indices) != visible_pose_indices:
            return replace(candidate, visible_pose_indices=(), quality_score=0.0)
    else:
        visible_pose_indices = longest_contiguous_indices(valid_pose_indices)
        if not visible_pose_indices:
            return replace(candidate, visible_pose_indices=(), quality_score=0.0)
    area_ratios = [metrics_by_index[index][0] for index in visible_pose_indices]
    center_xs = [metrics_by_index[index][1] for index in visible_pose_indices]
    return replace(
        candidate,
        visible_pose_indices=visible_pose_indices,
        quality_score=_quality_score(candidate, area_ratios, center_xs),
    )


def _candidate_key(candidate: TrajectoryCandidate) -> tuple[int, float, int, str, str]:
    return (
        candidate.saved_frames,
        candidate.quality_score,
        len(candidate.poses),
        candidate.object_type,
        candidate.object_id,
    )


def _select_scene(scene_name: str, config) -> SceneSelection | None:
    print(f"Scoring scene {scene_name}...")
    controller = build_controller(
        scene_name,
        config.gpu,
        config.image_size,
        config.field_of_view,
    )
    try:
        selected: dict[str, TrajectoryCandidate] = {}
        for trajectory in config.trajectory_names():
            print(f"  Selecting {trajectory}...")
            candidates = build_candidates(scene_name, controller, config, trajectory)
            candidates = _filter_candidates(candidates, config, trajectory)
            if not candidates:
                print(f"  No candidates for {trajectory}.")
                return None
            scored_candidates = [_score_candidate(controller, candidate) for candidate in candidates]
            scored_candidates = [
                candidate
                for candidate in scored_candidates
                if candidate.saved_frames >= config.min_frames
            ]
            if not scored_candidates:
                print(f"  No valid candidates for {trajectory}.")
                return None
            selected[trajectory] = max(scored_candidates, key=_candidate_key)
            chosen = selected[trajectory]
            print(
                f"  Chosen {trajectory}: {chosen.object_type} "
                f"({chosen.saved_frames}/{len(chosen.poses)} frames)"
            )
        return SceneSelection(scene=scene_name, trajectories=selected)
    finally:
        controller.stop()


def select_scene(config) -> SceneSelection:
    selections = []
    for scene_name in config.scene_names():
        selection = _select_scene(scene_name, config)
        if selection is not None:
            selections.append(selection)
    assert selections, "No candidate scene produced the requested trajectory set."
    return max(selections, key=lambda selection: (selection.total_saved_frames, selection.scene))
