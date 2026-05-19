"""Normalized clip entry for the video consistency benchmark.

Schema:
  {
    "clip_id": "thor_around_length_001_clip00",
    "group_id": "thor_around_length_001",
    "video_path": "/abs/path/to/video.mp4",
    "question": "...",
    "ground_truth": "1.2",
    "question_type": "object_dimensions",
    "dimension": "length",
    "engine": "thor",
    "scene_id": "FloorPlan1",
    "motion_family": "around"
  }
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    group_id: str
    engine: str
    scene_id: str
    room_bucket: str
    motion_family: str
    trajectory: str
    direction: str
    variation_tag: str
    question: str
    ground_truth: str
    question_type: str
    question_family: str
    dimension: str | None
    radius: float | None
    start_angle_offset: float
    cycle_count: int
    anchor_kind: str
    anchor_ids: tuple[str, ...]
    anchor_labels: tuple[str, ...]
    video_path: str
    output_dir: str
    quality_score: float

    def to_dict(self) -> dict:
        return asdict(self)
