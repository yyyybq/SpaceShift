"""Deterministic InteriorGS sequence builders for matched benchmark clips.

Exports:
  build_scene_context()
  build_object_sequence()
  build_rotation_sequence()
"""

import math

import numpy as np

from video_qa_dataset.variation_config import ORBIT_ARC_DEGREES


def build_scene_context(sampler, scene_path, all_scene_objects, scene_object_to_aabb) -> dict:
    wall_aabbs = sampler.load_wall_aabbs(scene_path)
    all_aabbs = [scene_object_to_aabb(obj) for obj in all_scene_objects] + wall_aabbs
    return {
        "scene_bounds": sampler.load_scene_bounds(scene_path),
        "room_polys": sampler.load_room_polys(scene_path),
        "all_aabbs": all_aabbs,
        "all_scene_objects": all_scene_objects,
    }


def _apply_bounce(poses: list, bounce_count: int) -> list:
    if bounce_count <= 1 or len(poses) < 2:
        return list(poses)
    result = list(poses)
    for bounce_index in range(1, bounce_count):
        segment = list(reversed(poses)) if bounce_index % 2 == 1 else list(poses)
        result.extend(segment[1:])
    return result


def _pose_key(pose) -> tuple:
    pos = tuple(float(v) for v in pose.position)
    return (pos, round(pose.yaw, 4), round(pose.pitch, 4))


def _deduplicate_interiorgs_poses(poses: list) -> list:
    if len(poses) <= 1:
        return list(poses)
    result = [poses[0]]
    for pose in poses[1:]:
        if _pose_key(pose) != _pose_key(result[-1]):
            result.append(pose)
    return result


def _camera_pose(CameraPose, position, target, radius, label):
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    dz = target[2] - position[2]
    horizontal_distance = max(float(np.linalg.norm([dx, dy])), 1e-6)
    return CameraPose(
        position=np.array(position, dtype=float),
        target=np.array(target, dtype=float),
        yaw=float(np.degrees(np.arctan2(dy, dx))),
        pitch=float(np.degrees(np.arctan2(-dz, horizontal_distance))),
        radius=float(radius),
        target_objects=[label],
    )


def _validate_pose(sampler, pose, obj, context, linear: bool) -> bool:
    if not sampler.is_position_valid(pose.position, context["scene_bounds"], context["room_polys"], context["all_scene_objects"]):
        return False
    target_ids = {str(obj.id)}
    if linear:
        return sampler._validate_visibility_linear(pose, [obj], context["all_aabbs"], target_ids)
    return sampler._validate_visibility(pose, [obj], context["all_aabbs"], target_ids)


def build_object_sequence(sampler, CameraPose, scene_path, obj, context, motion_family: str, direction: str, radius: float, start_angle_offset: float, cycle_count: int, increment: float, linear_steps: int, half_span: float) -> list:
    target = obj.center.copy()
    camera_height = sampler.compute_camera_height([obj])
    if motion_family in ("around", "spherical"):
        step_count = max(1, int(round(ORBIT_ARC_DEGREES / increment)))
        signed_increment = increment if direction == "cw" else -increment
        angles = [start_angle_offset + signed_increment * index for index in range(step_count)]
        elevations = np.linspace(55.0, -20.0, step_count) if motion_family == "spherical" else np.zeros(step_count)
        poses = []
        for angle_deg, elevation_deg in zip(angles, elevations):
            horizontal_radius = radius * math.cos(math.radians(elevation_deg))
            position = np.array([
                target[0] + horizontal_radius * math.cos(math.radians(angle_deg)),
                target[1] + horizontal_radius * math.sin(math.radians(angle_deg)),
                target[2] + radius * math.sin(math.radians(elevation_deg)) if motion_family == "spherical" else camera_height,
            ])
            pose = _camera_pose(CameraPose, position, target, radius, obj.label)
            if not _validate_pose(sampler, pose, obj, context, linear=False):
                return []
            poses.append(pose)
        return _deduplicate_interiorgs_poses(_apply_bounce(poses, cycle_count))
    radial = np.array([
        math.cos(math.radians(start_angle_offset)),
        math.sin(math.radians(start_angle_offset)),
        0.0,
    ])
    if motion_family == "approach":
        distances = np.linspace(radius * 3.0, radius * 0.5, linear_steps)
        if direction == "bw":
            distances = list(reversed(distances))
        poses = []
        for distance in distances:
            position = target + distance * radial
            position[2] = camera_height
            pose = _camera_pose(CameraPose, position, target, float(distance), obj.label)
            if not _validate_pose(sampler, pose, obj, context, linear=True):
                return []
            poses.append(pose)
        return _deduplicate_interiorgs_poses(_apply_bounce(poses, cycle_count))
    pass_distance = radius * 0.85
    trajectory_dir = np.array([radial[1], -radial[0], 0.0])
    closest = target + pass_distance * radial
    offsets = np.linspace(-half_span, half_span, linear_steps)
    if direction == "bw":
        offsets = list(reversed(offsets))
    forward_vector = -radial
    poses = []
    for offset in offsets:
        position = closest + offset * trajectory_dir
        position[2] = camera_height
        look_target = position + 10.0 * forward_vector
        look_target[2] = target[2]
        pose = _camera_pose(CameraPose, position, look_target, float(np.linalg.norm(position[:2] - target[:2])), obj.label)
        if not _validate_pose(sampler, pose, obj, context, linear=True):
            return []
        poses.append(pose)
    return _deduplicate_interiorgs_poses(_apply_bounce(poses, cycle_count))


def build_rotation_sequence(sampler, scene_path, all_scene_objects, room_index: int, direction: str, start_angle_offset: float, cycle_count: int) -> list:
    room_pose_pairs = sampler.generate_rotation_poses(scene_path, all_scene_objects)
    room_poses = [pose for pose, pose_room_index in room_pose_pairs if pose_room_index == room_index]
    if not room_poses:
        return []
    if direction == "ccw":
        room_poses = list(reversed(room_poses))
    interval = sampler.config.rotation_interval
    offset_steps = int(round(start_angle_offset / interval)) % len(room_poses)
    shifted = room_poses[offset_steps:] + room_poses[:offset_steps]
    return _deduplicate_interiorgs_poses(_apply_bounce(shifted, cycle_count))
