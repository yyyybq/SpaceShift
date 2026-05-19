"""Collision-aware reachability helpers for demo camera poses."""

import math

import numpy as np

import global_config


def nearest_reachable_point(reachable_points: list[dict], target_x: float, target_z: float) -> tuple[dict, float]:
    best_point = min(
        reachable_points,
        key=lambda point: (point["x"] - target_x) ** 2 + (point["z"] - target_z) ** 2,
    )
    distance = math.sqrt((best_point["x"] - target_x) ** 2 + (best_point["z"] - target_z) ** 2)
    return best_point, distance


def _object_bounds_3d(obj: dict):
    corners = None
    if obj.get("objectOrientedBoundingBox") and obj["objectOrientedBoundingBox"].get("cornerPoints"):
        corners = obj["objectOrientedBoundingBox"]["cornerPoints"]
    elif obj.get("axisAlignedBoundingBox") and obj["axisAlignedBoundingBox"].get("cornerPoints"):
        corners = obj["axisAlignedBoundingBox"]["cornerPoints"]
    if corners is None:
        return None
    coordinates = np.asarray(corners, dtype=float)
    return coordinates.min(axis=0), coordinates.max(axis=0)


def camera_hits_object(
    x: float,
    y: float,
    z: float,
    objects_metadata: list[dict],
    clearance: float = 0.12,
) -> bool:
    for obj in objects_metadata:
        if obj["objectType"].lower() in global_config.IGNORE_OBJECTS_GLOBAL:
            continue
        bounds = _object_bounds_3d(obj)
        if bounds is None:
            continue
        mins, maxs = bounds
        if (
            mins[0] - clearance <= x <= maxs[0] + clearance
            and mins[1] - clearance <= y <= maxs[1] + clearance
            and mins[2] - clearance <= z <= maxs[2] + clearance
        ):
            return True
    return False


def safe_snapped_position(
    reachable_points: list[dict],
    objects_metadata: list[dict],
    target_x: float,
    target_z: float,
    target_y: float | None = None,
    max_snap_distance: float = 0.25,
    clearance: float = 0.12,
):
    best_point, distance = nearest_reachable_point(reachable_points, target_x, target_z)
    if distance > max_snap_distance:
        return None
    snapped_y = best_point["y"] if target_y is None else target_y
    if camera_hits_object(
        best_point["x"],
        snapped_y,
        best_point["z"],
        objects_metadata,
        clearance=clearance,
    ):
        return None
    return best_point["x"], snapped_y, best_point["z"]
