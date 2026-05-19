"""Around-trajectory candidate builder."""

from trajectory_demos.controller_utils import (
    current_eye_y,
    objects_2d,
    prepare_scene,
    reachable_points,
)
from trajectory_demos.orbit_builder import build_orbit_poses
from trajectory_demos.trajectory_candidate import TrajectoryCandidate


def _movearound_object(
    object_info: dict,
    points: list[dict],
    objects_metadata: list[dict],
    eye_y: float,
    config,
):
    poses = build_orbit_poses(
        object_info=object_info,
        reachable_points=points,
        objects_metadata=objects_metadata,
        eye_y=eye_y,
        radius=config.radius,
        arc_degrees=config.arc_degrees,
        increment=config.increment,
    )
    if poses is None or len(poses) < config.min_frames:
        return None
    return {
        "id": object_info["id"],
        "type": object_info.get("type", "unknown"),
        "poses": poses,
    }


def build_around_candidates(scene_name: str, controller, config, trajectory: str) -> list[TrajectoryCandidate]:
    prepare_scene(controller, scene_name)
    eye_y = current_eye_y(controller)
    points = reachable_points(controller)
    objects_metadata = controller.last_event.metadata["objects"]
    reverse = trajectory.endswith("_ccw")
    candidates = []
    for object_info in objects_2d(controller):
        result = _movearound_object(object_info, points, objects_metadata, eye_y, config)
        if result is None:
            continue
        poses = result["poses"]
        if reverse:
            poses = list(reversed(poses))
        candidates.append(
            TrajectoryCandidate(
                scene=scene_name,
                trajectory=trajectory,
                object_id=result["id"],
                object_type=result["type"],
                poses=tuple(poses),
                pose_source="movearound_editor",
            )
        )
    return candidates
