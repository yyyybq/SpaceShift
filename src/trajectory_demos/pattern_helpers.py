"""Shared helpers for trajectory candidate builders."""

from multiview_qa_gen.config import CameraSamplingConfig

from trajectory_demos.pose_record import PoseRecord


def pose_records_from_movearound(object_info: dict, reverse: bool) -> tuple[PoseRecord, ...]:
    poses = tuple(
        PoseRecord(
            x=float(loc[0]),
            y=float(loc[1]),
            z=float(loc[2]),
            yaw=float(270.0 - rot),
            horizon=float(horizon),
        )
        for loc, rot, horizon in zip(
            object_info["cam_locs"],
            object_info["cam_rots"],
            object_info["cam_horizons"],
        )
    )
    return tuple(reversed(poses)) if reverse else poses


def pose_records_from_sampler(camera_poses, reverse: bool) -> tuple[PoseRecord, ...]:
    poses = tuple(
        PoseRecord(
            x=float(pose.position[0]),
            y=float(pose.position[1]),
            z=float(pose.position[2]),
            yaw=float(pose.rotation),
            horizon=float(pose.horizon),
        )
        for pose in camera_poses
    )
    return tuple(reversed(poses)) if reverse else poses


def linear_pose_count(radius: float, min_frames: int, sub_pattern: str) -> int:
    base = max(min_frames, int(round(radius * 8.0)))
    if sub_pattern == "approach":
        return base + 10
    return base + 4


def sampler_config(config, move_pattern: str, linear_sub_pattern: str | None = None):
    spherical_samples = max(config.min_frames, int(round(config.arc_degrees / config.increment)))
    return CameraSamplingConfig(
        move_pattern=move_pattern,
        linear_sub_pattern=linear_sub_pattern or "approach",
        total_rotation=config.arc_degrees,
        increment_rotation=config.increment,
        radius=config.radius,
        spherical_samples=spherical_samples,
        num_cameras_per_item=linear_pose_count(
            config.radius,
            config.min_frames,
            linear_sub_pattern or "approach",
        ),
        field_of_view=config.field_of_view,
        image_width=config.image_size,
        image_height=config.image_size,
    )
