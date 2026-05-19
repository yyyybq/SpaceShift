"""Build trajectory candidates with systematic variations applied.

Takes the existing candidate builders from trajectory_demos and wraps them
to apply start-angle offsets and multi-rotation arcs from TrajectoryVariation.
"""

import math
from dataclasses import replace

import numpy as np

from trajectory_demos.collision_checks import safe_snapped_position, camera_hits_object
from trajectory_demos.controller_utils import current_eye_y, objects_2d, prepare_scene, reachable_points
from trajectory_demos.pose_record import PoseRecord, deduplicate_consecutive_poses
from trajectory_demos.orbit_builder import build_orbit_poses
from trajectory_demos.trajectory_candidate import TrajectoryCandidate

from video_qa_dataset.variation_config import TrajectoryVariation


def _apply_bounce(poses: list[PoseRecord], bounce_count: int) -> list[PoseRecord]:
    """Repeat a one-way pose sequence with alternating direction.

    bounce_count=1 -> [A B C]
    bounce_count=2 -> [A B C B A]           (there and back)
    bounce_count=3 -> [A B C B A B C]       (there, back, there)
    """
    if bounce_count <= 1 or len(poses) < 2:
        return list(poses)
    result = list(poses)
    for i in range(1, bounce_count):
        segment = list(reversed(poses)) if i % 2 == 1 else list(poses)
        result.extend(segment[1:])
    return result


def build_around_varied(
    scene_name: str,
    controller,
    config,
    trajectory: str,
    variation: TrajectoryVariation,
    target_object_ids: list[str] | None = None,
) -> list[TrajectoryCandidate]:
    """Build around candidates with variation-specific start angle and arc."""
    prepare_scene(controller, scene_name)
    eye_y = current_eye_y(controller)
    points = reachable_points(controller)
    objects_metadata = controller.last_event.metadata["objects"]
    reverse = trajectory.endswith("_ccw")
    varied_config = _apply_variation_to_around(config, variation)

    candidates = []
    for object_info in objects_2d(controller):
        if target_object_ids and object_info["id"] not in target_object_ids:
            continue
        poses = _movearound_object_varied(
            object_info, points, objects_metadata, eye_y, varied_config, variation.start_angle_offset,
        )
        if poses is None:
            continue
        if reverse:
            poses = list(reversed(poses))
        bounced = deduplicate_consecutive_poses(_apply_bounce(poses, variation.bounce_count))
        if len(bounced) < config.min_frames:
            continue
        candidates.append(
            TrajectoryCandidate(
                scene=scene_name,
                trajectory=trajectory,
                object_id=object_info["id"],
                object_type=object_info.get("type", "unknown"),
                poses=tuple(bounced),
                pose_source=f"movearound_editor.{variation.tag}",
            )
        )
    return candidates


def _apply_variation_to_around(config, variation: TrajectoryVariation):
    increment = config.increment if variation.increment is None else variation.increment
    return replace(
        config,
        arc_degrees=variation.arc_degrees,
        radius=config.radius * variation.radius_multiplier,
        increment=increment,
    )


def _movearound_object_varied(
    object_info: dict,
    points: list[dict],
    objects_metadata: list[dict],
    eye_y: float,
    config,
    start_angle_offset: float,
):
    """Build a smooth exact orbit with an explicit start angle offset."""
    poses = build_orbit_poses(
        object_info=object_info,
        reachable_points=points,
        objects_metadata=objects_metadata,
        eye_y=eye_y,
        radius=config.radius,
        arc_degrees=config.arc_degrees,
        increment=config.increment,
        start_angle_offset=start_angle_offset,
    )
    if poses is None:
        return None
    poses = deduplicate_consecutive_poses(poses)
    if len(poses) < config.min_frames:
        return None
    return poses


def build_spherical_varied(
    scene_name: str,
    controller,
    config,
    trajectory: str,
    variation: TrajectoryVariation,
    target_object_ids: list[str] | None = None,
) -> list[TrajectoryCandidate]:
    """Build spherical candidates with varied start angle and arc."""
    from trajectory_demos.spherical_candidates import SPHERICAL_ELEVATION_TOP, SPHERICAL_ELEVATION_BOTTOM

    prepare_scene(controller, scene_name)
    clockwise = trajectory.endswith("_cw")
    eye_y = current_eye_y(controller)
    points = reachable_points(controller)
    objects_metadata = controller.last_event.metadata["objects"]

    varied_config = _apply_variation_to_around(config, variation)
    candidates = []
    for object_info in objects_2d(controller):
        if target_object_ids and object_info["id"] not in target_object_ids:
            continue
        around_poses = _movearound_object_varied(
            object_info, points, objects_metadata, eye_y, varied_config, variation.start_angle_offset,
        )
        if around_poses is None:
            continue
        poses = _spiral_poses_varied(
            around_poses, object_info, objects_metadata, varied_config, clockwise,
            SPHERICAL_ELEVATION_TOP, SPHERICAL_ELEVATION_BOTTOM,
        )
        if poses is None:
            continue
        bounced = deduplicate_consecutive_poses(_apply_bounce(list(poses), variation.bounce_count))
        if len(bounced) < config.min_frames:
            continue
        candidates.append(
            TrajectoryCandidate(
                scene=scene_name,
                trajectory=trajectory,
                object_id=object_info["id"],
                object_type=object_info.get("type", "unknown"),
                poses=tuple(bounced),
                pose_source=f"movearound_spiral.{variation.tag}",
            )
        )
    return candidates


def _spiral_poses_varied(
    around_poses: list[PoseRecord],
    object_info: dict,
    objects_metadata: list[dict],
    config,
    clockwise: bool,
    elevation_top: float,
    elevation_bottom: float,
):
    locs = [(p.x, p.y, p.z) for p in around_poses]
    if not clockwise:
        locs = list(reversed(locs))
    elevations = np.linspace(elevation_top, elevation_bottom, len(locs))
    centroid_x, centroid_z = object_info["centroid"]
    centroid_height = object_info["centroid_height"]
    poses = []
    for (cam_x, _, cam_z), elevation_deg in zip(locs, elevations):
        cam_y = np.clip(
            centroid_height + config.radius * math.sin(math.radians(elevation_deg)),
            0.3, 2.5,
        )
        if camera_hits_object(cam_x, cam_y, cam_z, objects_metadata):
            return None
        dx = centroid_x - cam_x
        dz = centroid_z - cam_z
        yaw = math.degrees(math.atan2(dx, dz))
        horizontal_distance = max(0.01, math.sqrt(dx**2 + dz**2))
        dy = centroid_height - cam_y
        horizon = math.degrees(math.atan2(-dy, horizontal_distance))
        poses.append(PoseRecord(
            x=float(cam_x), y=float(cam_y), z=float(cam_z),
            yaw=float(yaw), horizon=float(np.clip(horizon, -60.0, 60.0)),
        ))
    return tuple(poses)


def build_linear_varied(
    scene_name: str,
    controller,
    config,
    trajectory: str,
    variation: TrajectoryVariation,
    target_object_ids: list[str] | None = None,
) -> list[TrajectoryCandidate]:
    """Build approach/passby candidates at a specific angle from variation."""
    from trajectory_demos.linear_candidates import (
        LINEAR_STEP_COUNT, LINEAR_MIN_DISTANCE_MULTIPLIER,
        LINEAR_MAX_DISTANCE_MULTIPLIER, LINEAR_SNAP_DISTANCE,
        PASSBY_HALF_SPAN,
    )

    prepare_scene(controller, scene_name)
    points = reachable_points(controller)
    objects_metadata = controller.last_event.metadata["objects"]
    eye_y = current_eye_y(controller)
    is_approach = trajectory.startswith("approach")
    forward = trajectory.endswith("_fw")
    angle_deg = variation.start_angle_offset

    candidates = []
    for object_info in objects_2d(controller):
        if target_object_ids and object_info["id"] not in target_object_ids:
            continue
        if is_approach:
            poses = _approach_poses_at_angle(
                object_info, points, objects_metadata, eye_y, config,
                angle_deg, forward,
                LINEAR_STEP_COUNT, LINEAR_MIN_DISTANCE_MULTIPLIER,
                LINEAR_MAX_DISTANCE_MULTIPLIER, LINEAR_SNAP_DISTANCE,
            )
            pose_source = f"interiorgs_like.approach.{variation.tag}"
        else:
            poses = _passby_poses_at_angle(
                object_info, points, objects_metadata, eye_y, config,
                angle_deg, forward,
                LINEAR_STEP_COUNT, PASSBY_HALF_SPAN, LINEAR_SNAP_DISTANCE,
            )
            pose_source = f"interiorgs_like.pass_by.{variation.tag}"
        if poses is None:
            continue
        bounced = deduplicate_consecutive_poses(_apply_bounce(list(poses), variation.bounce_count))
        if len(bounced) < config.min_frames:
            continue
        candidates.append(TrajectoryCandidate(
            scene=scene_name,
            trajectory=trajectory,
            object_id=object_info["id"],
            object_type=object_info.get("type", "unknown"),
            poses=tuple(bounced),
            pose_source=pose_source,
        ))
    return candidates


def _approach_poses_at_angle(
    object_info, points, objects_metadata, eye_y, config,
    angle_deg, forward, step_count, min_dist_mult, max_dist_mult, snap_dist,
):
    centroid_x, centroid_z = object_info["centroid"]
    centroid_height = object_info["centroid_height"]
    direction = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))], dtype=float)
    distances = np.linspace(config.radius * max_dist_mult, config.radius * min_dist_mult, step_count)
    poses = []
    for distance in distances:
        target_x = centroid_x + distance * direction[0]
        target_z = centroid_z + distance * direction[1]
        safe_pose = safe_snapped_position(
            points, objects_metadata, float(target_x), float(target_z),
            max_snap_distance=snap_dist,
        )
        if safe_pose is None:
            return None
        cam_x, cam_y, cam_z = safe_pose
        dx, dz = centroid_x - cam_x, centroid_z - cam_z
        snapped_distance = max(0.1, math.sqrt(dx**2 + dz**2))
        yaw = math.degrees(math.atan2(dx, dz))
        horizon = math.degrees(math.atan((eye_y - centroid_height) / snapped_distance))
        poses.append(PoseRecord(
            x=float(cam_x), y=float(cam_y), z=float(cam_z),
            yaw=float(yaw), horizon=float(np.clip(horizon, -30.0, 60.0)),
        ))
    poses = deduplicate_consecutive_poses(poses)
    if not forward:
        poses = list(reversed(poses))
    return tuple(poses)


def _passby_poses_at_angle(
    object_info, points, objects_metadata, eye_y, config,
    angle_deg, forward, step_count, half_span, snap_dist,
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
    for offset in np.linspace(-half_span, half_span, step_count):
        target_position = closest + offset * trajectory_dir
        safe_pose = safe_snapped_position(
            points, objects_metadata, float(target_position[0]), float(target_position[1]),
            max_snap_distance=snap_dist,
        )
        if safe_pose is None:
            return None
        cam_x, cam_y, cam_z = safe_pose
        poses.append(PoseRecord(
            x=float(cam_x), y=float(cam_y), z=float(cam_z),
            yaw=float(yaw), horizon=float(np.clip(horizon, -30.0, 60.0)),
        ))
    poses = deduplicate_consecutive_poses(poses)
    if not forward:
        poses = list(reversed(poses))
    return tuple(poses)


def build_rotation_varied(
    scene_name: str,
    controller,
    config,
    trajectory: str,
    variation: TrajectoryVariation,
) -> list[TrajectoryCandidate]:
    """Build rotation candidates with varied arc and start angle."""
    from multiview_qa_gen.camera_sampler import CameraSampler
    from trajectory_demos.pattern_helpers import sampler_config, pose_records_from_sampler

    prepare_scene(controller, scene_name)
    varied_config = replace(config, arc_degrees=variation.arc_degrees)
    sampler = CameraSampler(sampler_config(varied_config, "rotation"))
    reachable = sampler.get_reachable_positions(controller)
    points = reachable_points(controller)
    room_center = controller.last_event.metadata["sceneBounds"]["center"]
    reverse = trajectory.endswith("_ccw")
    trajectory_info = sampler.sample_rotation_pattern(
        (room_center["x"], room_center["z"]),
        reachable, points, current_eye_y(controller),
    )
    if trajectory_info is None or not trajectory_info.is_valid:
        return []
    poses = pose_records_from_sampler(trajectory_info.camera_poses, reverse)
    if variation.start_angle_offset > 0:
        offset_count = int(variation.start_angle_offset / max(config.increment, 1))
        offset_count = min(offset_count, len(poses) - 1)
        poses = poses[offset_count:] + poses[:offset_count]
    bounced = deduplicate_consecutive_poses(_apply_bounce(list(poses), variation.bounce_count))
    if len(bounced) < config.min_frames:
        return []
    return [
        TrajectoryCandidate(
            scene=scene_name,
            trajectory=trajectory,
            object_id="room",
            object_type="room",
            poses=tuple(bounced),
            pose_source=f"camera_sampler.rotation.{variation.tag}",
        )
    ]


def build_varied_candidates(
    scene_name: str,
    controller,
    config,
    trajectory: str,
    variation: TrajectoryVariation,
    target_object_ids: list[str] | None = None,
) -> list[TrajectoryCandidate]:
    """Dispatch to pattern-specific varied builder."""
    if trajectory.startswith("around_"):
        return build_around_varied(scene_name, controller, config, trajectory, variation, target_object_ids)
    if trajectory.startswith("spherical_"):
        return build_spherical_varied(scene_name, controller, config, trajectory, variation, target_object_ids)
    if trajectory.startswith("rotation_"):
        return build_rotation_varied(scene_name, controller, config, trajectory, variation)
    return build_linear_varied(scene_name, controller, config, trajectory, variation, target_object_ids)
