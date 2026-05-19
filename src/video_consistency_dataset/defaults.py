"""Default benchmark targets and selection constants.

Target schema:
  {
    "motion_family": "around",
    "question_type": "object_dimensions",
    "dimension": "length",
    "radius": null
  }
"""

DIMENSIONS = ("length", "width", "height")
THOR_ENGINES = ("thor", "interiorgs")
OBJECT_MOTIONS = ("approach", "passby", "around", "spherical")
CIRCULAR_MOTIONS = ("around", "spherical")
ROTATION_MOTION = "rotation"
LINEAR_DIRECTIONS = {"approach": ("fw", "bw"), "passby": ("fw", "bw")}
ORBIT_DIRECTIONS = {"around": ("cw", "ccw"), "spherical": ("cw", "ccw"), "rotation": ("cw", "ccw")}


def _spread_radii(radii: tuple[float, ...], count: int) -> tuple[float, ...]:
    ordered = tuple(sorted(dict.fromkeys(radii)))
    assert ordered, "At least one radius is required."
    if len(ordered) == 1:
        return tuple(ordered[0] for _ in range(count))
    if len(ordered) >= count:
        indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
        return tuple(ordered[index] for index in indices)
    values = list(ordered)
    while len(values) < count:
        values.append(ordered[len(values) % len(ordered)])
    return tuple(values)


def group_targets_per_engine(radii: tuple[float, ...]) -> list[dict]:
    targets: list[dict] = []

    # Size targets: 8 per engine; dimensions 3× length, 3× width, 2× height (fewer height questions)
    for motion_family, dimension in (
        ("approach", "length"),
        ("approach", "width"),
        ("passby", "length"),
        ("passby", "width"),
        ("around", "length"),
        ("around", "height"),
        ("spherical", "width"),
        ("spherical", "height"),
    ):
        targets.append({
            "motion_family": motion_family,
            "question_type": "object_dimensions",
            "question_family": "size",
            "dimension": dimension,
            "radius": None,
        })

    # Camera-distance: 7 per engine (4 around + 3 spherical) to balance with size and pair counts
    distance_radii = _spread_radii(radii, 4)
    for motion_family, radius in (
        ("around", distance_radii[0]),
        ("around", distance_radii[1]),
        ("around", distance_radii[2]),
        ("around", distance_radii[3]),
        ("spherical", distance_radii[0]),
        ("spherical", distance_radii[1]),
        ("spherical", distance_radii[2]),
    ):
        targets.append({
            "motion_family": motion_family,
            "question_type": "object_distance_to_camera",
            "question_family": "camera_distance",
            "dimension": None,
            "radius": radius,
        })

    # Pair-distance: 7 per engine (matches size and distance slots for near-even question-type mix)
    for _ in range(7):
        targets.append({
            "motion_family": ROTATION_MOTION,
            "question_type": "object_pair_distance_center",
            "question_family": "pair_distance",
            "dimension": None,
            "radius": None,
        })
    return targets
