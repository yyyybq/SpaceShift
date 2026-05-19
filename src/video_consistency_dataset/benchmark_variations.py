"""Benchmark-local trajectory variations.

This benchmark reuses the stable pattern-specific variation sets from the
trajectory QA pipeline so candidate coverage remains high enough for planning.
"""

from video_qa_dataset.variation_config import ORBIT_ARC_DEGREES, TrajectoryVariation, variations_for_pattern
from video_consistency_dataset.defaults import CIRCULAR_MOTIONS


def benchmark_variations(config, pattern: str) -> tuple[TrajectoryVariation, ...]:
    del config
    return variations_for_pattern(pattern)


def variation_from_clip(clip) -> TrajectoryVariation:
    arc = ORBIT_ARC_DEGREES if clip.motion_family in CIRCULAR_MOTIONS else 360.0
    return TrajectoryVariation(
        name=clip.variation_tag,
        arc_degrees=arc,
        start_angle_offset=clip.start_angle_offset,
        radius_multiplier=1.0,
        increment=None,
        bounce_count=clip.cycle_count,
    )
