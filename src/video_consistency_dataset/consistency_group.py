"""Consistency group metadata for matched benchmark slices.

Schema:
  {
    "group_id": "interiorgs_rotation_distance_004",
    "engine": "interiorgs",
    "scene_id": "0267_840790",
    "motion_family": "rotation",
    "question_type": "object_pair_distance_center",
    "anchor_ids": ["chair_1", "table_3"],
    "clip_ids": ["..."],
    "shared_ground_truth": "2.3"
  }
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConsistencyGroup:
    group_id: str
    engine: str
    scene_id: str
    room_bucket: str
    motion_family: str
    question_type: str
    question_family: str
    dimension: str | None
    radius: float | None
    anchor_kind: str
    anchor_ids: tuple[str, ...]
    anchor_labels: tuple[str, ...]
    shared_ground_truth: str
    clip_ids: tuple[str, ...]
    target_group_size: int
    selected_group_size: int

    def to_dict(self) -> dict:
        return asdict(self)
