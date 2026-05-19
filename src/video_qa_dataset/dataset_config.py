"""Configuration for the video QA dataset generation pipeline."""

from dataclasses import dataclass

from trajectory_demos.demo_config import ALL_TRAJECTORIES, DEFAULT_CANDIDATE_SCENES


ALL_ROOM_TYPES = ("kitchen", "living_room", "bedroom", "bathroom")

SCENES_BY_ROOM_TYPE: dict[str, tuple[str, ...]] = {
    "kitchen": tuple(f"FloorPlan{i}" for i in range(1, 31)),
    "living_room": tuple(f"FloorPlan{200 + i}" for i in range(1, 31)),
    "bedroom": tuple(f"FloorPlan{300 + i}" for i in range(1, 31)),
    "bathroom": tuple(f"FloorPlan{400 + i}" for i in range(1, 31)),
}


@dataclass(frozen=True)
class DatasetConfig:
    output_dir: str
    scenes: tuple[str, ...]
    trajectories: tuple[str, ...]
    radius: float
    arc_degrees: float
    increment: float
    field_of_view: int
    image_size: int
    fps: int
    gpu: int
    min_frames: int
    max_objects_per_scene: int
    skip_existing: bool

    def scene_list(self) -> list[str]:
        return list(self.scenes)

    def trajectory_list(self) -> list[str]:
        return list(self.trajectories)


def default_config(**overrides) -> DatasetConfig:
    defaults = dict(
        output_dir="./video_qa_output",
        scenes=DEFAULT_CANDIDATE_SCENES,
        trajectories=ALL_TRAJECTORIES,
        radius=1.0,
        arc_degrees=360.0,
        increment=5.0,
        field_of_view=75,
        image_size=384,
        fps=10,
        gpu=0,
        min_frames=30,
        max_objects_per_scene=5,
        skip_existing=True,
    )
    defaults.update(overrides)
    return DatasetConfig(**defaults)


def expanded_scenes(room_types: list[str] | None, scene_names: list[str] | None) -> tuple[str, ...]:
    """Resolve scene list from room types and/or explicit names."""
    scenes: list[str] = []
    if room_types:
        for room_type in room_types:
            assert room_type in SCENES_BY_ROOM_TYPE, f"Unknown room type: {room_type}"
            scenes.extend(SCENES_BY_ROOM_TYPE[room_type])
    if scene_names:
        scenes.extend(scene_names)
    if not scenes:
        return DEFAULT_CANDIDATE_SCENES
    return tuple(dict.fromkeys(scenes))
