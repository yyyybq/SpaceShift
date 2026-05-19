"""CameraSampler-backed candidates that still fit demo output needs."""

from multiview_qa_gen.camera_sampler import CameraSampler

from trajectory_demos.controller_utils import current_eye_y, prepare_scene, reachable_points
from trajectory_demos.pattern_helpers import pose_records_from_sampler, sampler_config
from trajectory_demos.trajectory_candidate import TrajectoryCandidate


def build_rotation_candidates(scene_name: str, controller, config, trajectory: str) -> list[TrajectoryCandidate]:
    prepare_scene(controller, scene_name)
    sampler = CameraSampler(sampler_config(config, "rotation"))
    reachable = sampler.get_reachable_positions(controller)
    points = reachable_points(controller)
    room_center = controller.last_event.metadata["sceneBounds"]["center"]
    reverse = trajectory.endswith("_ccw")
    trajectory_info = sampler.sample_rotation_pattern(
        (room_center["x"], room_center["z"]),
        reachable,
        points,
        current_eye_y(controller),
    )
    assert trajectory_info is not None and trajectory_info.is_valid, "Rotation trajectory is invalid."
    return [
        TrajectoryCandidate(
            scene=scene_name,
            trajectory=trajectory,
            object_id="room",
            object_type="room",
            poses=pose_records_from_sampler(trajectory_info.camera_poses, reverse),
            pose_source="camera_sampler.rotation",
        )
    ]
