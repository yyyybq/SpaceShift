"""Systematic variation definitions for trajectory-based video QA dataset.

Each trajectory pattern gets a set of named variations that alter generation
parameters such as starting angle and bounce count (forward+reverse passes).

Variation axes:
  - arc_degrees: requested sweep for orbit-like patterns (around/spherical: half orbit)
  - start_angle_offset: initial azimuth in degrees (every 30° from 0 to 330)
  - bounce_count: forward+reverse pass count (1 through 5)
  - direction: cw / ccw or fw / bw (handled by trajectory naming)
"""

from dataclasses import dataclass

ORBIT_ARC_DEGREES = 180.0

START_ANGLES = tuple(range(0, 360, 30))
MAX_BOUNCE = 5


@dataclass(frozen=True)
class TrajectoryVariation:
    name: str
    arc_degrees: float
    start_angle_offset: float
    radius_multiplier: float
    increment: float | None
    bounce_count: int = 1

    @property
    def tag(self) -> str:
        return self.name


def _orbit(name: str, start: float, bounce: int) -> TrajectoryVariation:
    return TrajectoryVariation(
        name,
        arc_degrees=ORBIT_ARC_DEGREES,
        start_angle_offset=start,
        radius_multiplier=1.0,
        increment=None,
        bounce_count=bounce,
    )


def _linear(name: str, start: float, bounce: int) -> TrajectoryVariation:
    return TrajectoryVariation(
        name,
        arc_degrees=360.0,
        start_angle_offset=start,
        radius_multiplier=1.0,
        increment=None,
        bounce_count=bounce,
    )


def _rotation(name: str, start: float, bounce: int) -> TrajectoryVariation:
    return TrajectoryVariation(
        name,
        arc_degrees=360.0,
        start_angle_offset=start,
        radius_multiplier=1.0,
        increment=None,
        bounce_count=bounce,
    )


AROUND_VARIATIONS = tuple(
    _orbit(f"{bounce}x_start{angle}", start=angle, bounce=bounce)
    for bounce in range(1, MAX_BOUNCE + 1)
    for angle in START_ANGLES
)

SPHERICAL_VARIATIONS = tuple(
    _orbit(f"{bounce}x_start{angle}", start=angle, bounce=bounce)
    for bounce in range(1, MAX_BOUNCE + 1)
    for angle in START_ANGLES
)

ROTATION_VARIATIONS = tuple(
    _rotation(f"{bounce}x_start{angle}", start=angle, bounce=bounce)
    for bounce in range(1, MAX_BOUNCE + 1)
    for angle in START_ANGLES
)

APPROACH_VARIATIONS = tuple(
    _linear(f"{bounce}x_angle{angle}", start=angle, bounce=bounce)
    for bounce in range(1, MAX_BOUNCE + 1)
    for angle in START_ANGLES
)

PASSBY_VARIATIONS = tuple(
    _linear(f"{bounce}x_angle{angle}", start=angle, bounce=bounce)
    for bounce in range(1, MAX_BOUNCE + 1)
    for angle in START_ANGLES
)

PATTERN_VARIATIONS: dict[str, tuple[TrajectoryVariation, ...]] = {
    "around": AROUND_VARIATIONS,
    "spherical": SPHERICAL_VARIATIONS,
    "rotation": ROTATION_VARIATIONS,
    "approach": APPROACH_VARIATIONS,
    "passby": PASSBY_VARIATIONS,
}


def variations_for_pattern(pattern: str) -> tuple[TrajectoryVariation, ...]:
    assert pattern in PATTERN_VARIATIONS, f"Unknown pattern: {pattern}"
    return PATTERN_VARIATIONS[pattern]


def pattern_from_trajectory(trajectory: str) -> str:
    return trajectory.rsplit("_", 1)[0]


def direction_from_trajectory(trajectory: str) -> str:
    return trajectory.rsplit("_", 1)[1]
