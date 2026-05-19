"""Candidate trajectory metadata for demo selection."""

from dataclasses import dataclass

from trajectory_demos.pose_record import PoseRecord


@dataclass(frozen=True)
class TrajectoryCandidate:
    scene: str
    trajectory: str
    object_id: str
    object_type: str
    poses: tuple[PoseRecord, ...]
    pose_source: str
    visible_pose_indices: tuple[int, ...] = ()
    quality_score: float = 0.0

    @property
    def saved_frames(self) -> int:
        return len(self.visible_pose_indices)
