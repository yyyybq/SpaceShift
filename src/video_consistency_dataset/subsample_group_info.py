"""Group summary row for subsampling plans."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupInfo:
    group_id: str
    question_type: str
    scene_id: str
    room_bucket: str
    anchor_ids: tuple[str, ...]


def leak_key(info: GroupInfo) -> tuple[str, str, tuple[str, ...]]:
    return (info.question_type, info.scene_id, tuple(sorted(info.anchor_ids)))
