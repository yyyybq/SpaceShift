"""Serializable benchmark plan container.

Schema:
  {
    "config": {...},
    "groups": [{...}],
    "clips": [{...}]
  }
"""

from dataclasses import dataclass

from video_consistency_dataset.clip_spec import ClipSpec
from video_consistency_dataset.consistency_group import ConsistencyGroup


@dataclass(frozen=True)
class BenchmarkPlan:
    config: dict
    groups: tuple[ConsistencyGroup, ...]
    clips: tuple[ClipSpec, ...]

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "groups": [group.to_dict() for group in self.groups],
            "clips": [clip.to_dict() for clip in self.clips],
        }
