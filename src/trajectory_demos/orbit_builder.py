"""Exact orbit pose builder for object-centric camera trajectories.

Usage:
    Imported by around and spherical trajectory builders.

Input spec:
    - object_info: object metadata from `objects_2d()`
    - reachable_points: AI2-THOR reachable floor points
    - objects_metadata: full AI2-THOR object metadata
    - eye_y: crouched camera eye height
    - radius, arc_degrees, increment, start_angle_offset: orbit controls

Output spec:
    - list[PoseRecord] for a fully valid exact orbit
    - None when any requested pose is too far from reachable floor or collides
"""

import math

import numpy as np

from trajectory_demos.collision_checks import camera_hits_object, nearest_reachable_point
from trajectory_demos.pose_record import PoseRecord


def orbit_angles(
    arc_degrees: float,
    increment: float,
    start_angle_offset: float = 0.0,
) -> list[float]:
    assert arc_degrees > 0.0, f"arc_degrees must be positive, got {arc_degrees}."
    assert increment > 0.0, f"increment must be positive, got {increment}."
    relative_angles = np.arange(0.0, arc_degrees + 1e-5, increment, dtype=float)
    if relative_angles.size == 0 or abs(float(relative_angles[-1]) - arc_degrees) > 1e-5:
        relative_angles = np.append(relative_angles, arc_degrees)
    return [float(start_angle_offset + angle) for angle in relative_angles]


def build_orbit_poses(
    object_info: dict,
    reachable_points: list[dict],
    objects_metadata: list[dict],
    eye_y: float,
    radius: float,
    arc_degrees: float,
    increment: float,
    start_angle_offset: float = 0.0,
    max_floor_distance: float = 0.185,
    clearance: float = 0.12,
) -> list[PoseRecord] | None:
    assert radius > 0.0, f"radius must be positive, got {radius}."
    centroid_x, centroid_z = object_info["centroid"]
    centroid_height = object_info["centroid_height"]
    horizon = np.round(
        np.degrees(np.arctan((eye_y - centroid_height) / radius)) / 30.0
    ) * 30.0
    poses: list[PoseRecord] = []
    for theta in orbit_angles(
        arc_degrees=arc_degrees,
        increment=increment,
        start_angle_offset=start_angle_offset,
    ):
        cam_x = centroid_x + radius * math.cos(math.radians(theta))
        cam_z = centroid_z + radius * math.sin(math.radians(theta))
        floor_point, floor_distance = nearest_reachable_point(reachable_points, cam_x, cam_z)
        if floor_distance > max_floor_distance:
            return None
        cam_y = float(floor_point["y"])
        if camera_hits_object(
            cam_x,
            cam_y,
            cam_z,
            objects_metadata,
            clearance=clearance,
        ):
            return None
        dx = centroid_x - cam_x
        dz = centroid_z - cam_z
        yaw = math.degrees(math.atan2(dx, dz))
        poses.append(
            PoseRecord(
                x=float(cam_x),
                y=cam_y,
                z=float(cam_z),
                yaw=float(yaw),
                horizon=float(horizon),
            )
        )
    return poses
