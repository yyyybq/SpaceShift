"""Selected scene metadata for a curated trajectory demo set."""

from dataclasses import dataclass

from trajectory_demos.trajectory_candidate import TrajectoryCandidate


@dataclass(frozen=True)
class SceneSelection:
    scene: str
    trajectories: dict[str, TrajectoryCandidate]

    @property
    def total_saved_frames(self) -> int:
        return sum(candidate.saved_frames for candidate in self.trajectories.values())
