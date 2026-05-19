"""Deterministic spherical-style trajectory candidates for demos."""

import math

import numpy as np

from trajectory_demos.around_candidates import _movearound_object
from trajectory_demos.collision_checks import camera_hits_object
from trajectory_demos.controller_utils import current_eye_y, objects_2d, prepare_scene, reachable_points
from trajectory_demos.pose_record import PoseRecord
from trajectory_demos.trajectory_candidate import TrajectoryCandidate


SPHERICAL_ELEVATION_TOP = 55.0
SPHERICAL_ELEVATION_BOTTOM = -20.0


def _spiral_poses(
    around_poses: list[PoseRecord],
    object_info: dict,
    objects_metadata: list[dict],
    config,
    clockwise: bool,
):
    locs = [(p.x, p.y, p.z) for p in around_poses]
    if not clockwise:
        locs = list(reversed(locs))
    elevations = np.linspace(SPHERICAL_ELEVATION_TOP, SPHERICAL_ELEVATION_BOTTOM, len(locs))
    centroid_x, centroid_z = object_info["centroid"]
    centroid_height = object_info["centroid_height"]
    poses = []
    for (cam_x, _, cam_z), elevation_deg in zip(locs, elevations):
        cam_y = np.clip(
            centroid_height + config.radius * math.sin(math.radians(elevation_deg)),
            0.3,
            2.5,
        )
        if camera_hits_object(cam_x, cam_y, cam_z, objects_metadata):
            return None
        dx = centroid_x - cam_x
        dz = centroid_z - cam_z
        yaw = math.degrees(math.atan2(dx, dz))
        horizontal_distance = max(0.01, math.sqrt(dx**2 + dz**2))
        dy = centroid_height - cam_y
        horizon = math.degrees(math.atan2(-dy, horizontal_distance))
        poses.append(
            PoseRecord(
                x=float(cam_x),
                y=float(cam_y),
                z=float(cam_z),
                yaw=float(yaw),
                horizon=float(np.clip(horizon, -60.0, 60.0)),
            )
        )
    return tuple(poses)


def build_spherical_candidates(scene_name: str, controller, config, trajectory: str) -> list[TrajectoryCandidate]:
    prepare_scene(controller, scene_name)
    clockwise = trajectory.endswith("_cw")
    eye_y = current_eye_y(controller)
    points = reachable_points(controller)
    objects_metadata = controller.last_event.metadata["objects"]
    candidates = []
    for object_info in objects_2d(controller):
        result = _movearound_object(object_info, points, objects_metadata, eye_y, config)
        if result is None:
            continue
        poses = _spiral_poses(result["poses"], object_info, objects_metadata, config, clockwise)
        if poses is None:
            continue
        candidates.append(
            TrajectoryCandidate(
                scene=scene_name,
                trajectory=trajectory,
                object_id=result["id"],
                object_type=result["type"],
                poses=poses,
                pose_source="movearound_spiral",
            )
        )
    return candidates
