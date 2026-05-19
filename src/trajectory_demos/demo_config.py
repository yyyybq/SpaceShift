"""Configuration for AI2-THOR trajectory demo export."""

from dataclasses import dataclass


ALL_TRAJECTORIES = (
    "around_cw",
    "around_ccw",
    "spherical_cw",
    "spherical_ccw",
    "rotation_cw",
    "rotation_ccw",
    "approach_fw",
    "approach_bw",
    "passby_fw",
    "passby_bw",
)

DEFAULT_CANDIDATE_SCENES = tuple(
    [f"FloorPlan{i}" for i in range(1, 31)]
    + [f"FloorPlan{200 + i}" for i in range(1, 31)]
    + [f"FloorPlan{300 + i}" for i in range(1, 31)]
    + [f"FloorPlan{400 + i}" for i in range(1, 31)]
)


@dataclass(frozen=True)
class TrajectoryDemoConfig:
    scene: str
    output_dir: str
    trajectory: str
    object_index: int | None
    object_id: str | None
    object_type: str | None
    radius: float
    arc_degrees: float
    increment: float
    field_of_view: int
    image_size: int
    fps: int
    gpu: int
    min_frames: int
    candidate_scenes: tuple[str, ...]

    def scene_names(self) -> list[str]:
        if self.scene != "auto":
            return [self.scene]
        return list(self.candidate_scenes)

    def trajectory_names(self) -> list[str]:
        if self.trajectory == "all":
            return list(ALL_TRAJECTORIES)
        assert self.trajectory in ALL_TRAJECTORIES, (
            f"Unknown trajectory: {self.trajectory}. Choose from {ALL_TRAJECTORIES}"
        )
        return [self.trajectory]
