"""Video consistency benchmark package.

Tree:
  video_consistency_dataset/
    benchmark_config.py
    benchmark_plan.py
    clip_spec.py
    consistency_group.py
    defaults.py
    planner.py
    plan_io.py
    stats_report.py
    thor_mining.py
    thor_render.py
    interiorgs_mining.py
    interiorgs_render.py

Public interface:
  BenchmarkConfig
  BenchmarkPlan
  ClipSpec
  ConsistencyGroup
"""

from video_consistency_dataset.benchmark_config import BenchmarkConfig
from video_consistency_dataset.benchmark_plan import BenchmarkPlan
from video_consistency_dataset.clip_spec import ClipSpec
from video_consistency_dataset.consistency_group import ConsistencyGroup

__all__ = [
    "BenchmarkConfig",
    "BenchmarkPlan",
    "ClipSpec",
    "ConsistencyGroup",
]
