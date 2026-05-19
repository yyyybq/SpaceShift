"""Pose data for AI2-THOR trajectory demos."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseRecord:
    x: float
    y: float
    z: float
    yaw: float
    horizon: float


def deduplicate_consecutive_poses(poses: list[PoseRecord]) -> list[PoseRecord]:
    """Remove consecutive poses with identical camera position and orientation.

    AI2-THOR snaps requested positions to a discrete reachable grid, so
    multiple consecutive ideal positions often collapse to the same grid
    point, producing identical frames.
    """
    if len(poses) <= 1:
        return list(poses)
    result = [poses[0]]
    for pose in poses[1:]:
        prev = result[-1]
        if (pose.x, pose.y, pose.z, pose.yaw, pose.horizon) != (prev.x, prev.y, prev.z, prev.yaw, prev.horizon):
            result.append(pose)
    return result
