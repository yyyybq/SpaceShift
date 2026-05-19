"""Deterministic linear trajectory candidates for demos."""

import math

import numpy as np

from trajectory_demos.collision_checks import safe_snapped_position
from trajectory_demos.controller_utils import current_eye_y, objects_2d, prepare_scene, reachable_points
from trajectory_demos.pose_record import PoseRecord, deduplicate_consecutive_poses
from trajectory_demos.trajectory_candidate import TrajectoryCandidate


LINEAR_SEARCH_ANGLES = tuple(np.linspace(0.0, 330.0, 12))
LINEAR_STEP_COUNT = 36
LINEAR_MIN_DISTANCE_MULTIPLIER = 0.5
LINEAR_MAX_DISTANCE_MULTIPLIER = 3.0
LINEAR_SNAP_DISTANCE = 0.25
PASSBY_HALF_SPAN = 1.5


def _approach_poses(
    object_info: dict,
    points: list[dict],
    objects_metadata: list[dict],
    eye_y: float,
    config,
    angle_deg: float,
    forward: bool,
):
    centroid_x, centroid_z = object_info["centroid"]
    centroid_height = object_info["centroid_height"]
    direction = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))], dtype=float)
    far_distance = config.radius * LINEAR_MAX_DISTANCE_MULTIPLIER
    near_distance = config.radius * LINEAR_MIN_DISTANCE_MULTIPLIER
    distances = np.linspace(far_distance, near_distance, LINEAR_STEP_COUNT)
    poses = []
    for distance in distances:
        target_x = centroid_x + distance * direction[0]
        target_z = centroid_z + distance * direction[1]
        safe_pose = safe_snapped_position(
            points,
            objects_metadata,
            float(target_x),
            float(target_z),
            max_snap_distance=LINEAR_SNAP_DISTANCE,
        )
        if safe_pose is None:
            return None
        cam_x, cam_y, cam_z = safe_pose
        dx = centroid_x - cam_x
        dz = centroid_z - cam_z
        snapped_distance = max(0.1, math.sqrt(dx**2 + dz**2))
        yaw = math.degrees(math.atan2(dx, dz))
        horizon = math.degrees(math.atan((eye_y - centroid_height) / snapped_distance))
        poses.append(
            PoseRecord(
                x=float(cam_x),
                y=float(cam_y),
                z=float(cam_z),
                yaw=float(yaw),
                horizon=float(np.clip(horizon, -30.0, 60.0)),
            )
        )
    poses = deduplicate_consecutive_poses(poses)
    if not forward:
        poses = list(reversed(poses))
    return tuple(poses)


def _passby_poses(
    object_info: dict,
    points: list[dict],
    objects_metadata: list[dict],
    eye_y: float,
    config,
    angle_deg: float,
    forward: bool,
):
    centroid_x, centroid_z = object_info["centroid"]
    centroid_height = object_info["centroid_height"]
    pass_distance = config.radius * 0.85
    radial = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))], dtype=float)
    trajectory_dir = np.array([radial[1], -radial[0]], dtype=float)
    closest = np.array([centroid_x, centroid_z], dtype=float) + pass_distance * radial
    camera_forward = -radial
    yaw = math.degrees(math.atan2(camera_forward[0], camera_forward[1]))
    horizon = math.degrees(math.atan((eye_y - centroid_height) / max(pass_distance, 0.1)))
    poses = []
    for offset in np.linspace(-PASSBY_HALF_SPAN, PASSBY_HALF_SPAN, LINEAR_STEP_COUNT):
        target_position = closest + offset * trajectory_dir
        safe_pose = safe_snapped_position(
            points,
            objects_metadata,
            float(target_position[0]),
            float(target_position[1]),
            max_snap_distance=LINEAR_SNAP_DISTANCE,
        )
        if safe_pose is None:
            return None
        cam_x, cam_y, cam_z = safe_pose
        poses.append(
            PoseRecord(
                x=float(cam_x),
                y=float(cam_y),
                z=float(cam_z),
                yaw=float(yaw),
                horizon=float(np.clip(horizon, -30.0, 60.0)),
            )
        )
    poses = deduplicate_consecutive_poses(poses)
    if not forward:
        poses = list(reversed(poses))
    return tuple(poses)


def build_linear_candidates(scene_name: str, controller, config, trajectory: str) -> list[TrajectoryCandidate]:
    prepare_scene(controller, scene_name)
    points = reachable_points(controller)
    objects_metadata = controller.last_event.metadata["objects"]
    eye_y = current_eye_y(controller)
    is_approach = trajectory.startswith("approach")
    forward = trajectory.endswith("_fw")
    candidates = []
    for object_info in objects_2d(controller):
        for angle_deg in LINEAR_SEARCH_ANGLES:
            if is_approach:
                poses = _approach_poses(
                    object_info,
                    points,
                    objects_metadata,
                    eye_y,
                    config,
                    angle_deg,
                    forward,
                )
                pose_source = "interiorgs_like.approach"
            else:
                poses = _passby_poses(
                    object_info,
                    points,
                    objects_metadata,
                    eye_y,
                    config,
                    angle_deg,
                    forward,
                )
                pose_source = "interiorgs_like.pass_by"
            if poses is None:
                continue
            candidates.append(
                TrajectoryCandidate(
                    scene=scene_name,
                    trajectory=trajectory,
                    object_id=object_info["id"],
                    object_type=object_info.get("type", "unknown"),
                    poses=poses,
                    pose_source=pose_source,
                )
            )
    return candidates
