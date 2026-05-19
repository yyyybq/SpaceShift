"""Subset selection for matched clips inside a consistency group.

Input clip schema:
  {
    "direction": "cw",
    "variation_tag": "sample_00_2x_start140p00",
    "start_angle_offset": 140.0,
    "cycle_count": 2,
    "radius": 1.0,
    "quality_score": 125.0
  }

With 30 clips per group and 60 variations per motion (12 start angles × 5 bounces),
exhaustive C(n,k) is infeasible. Selection uses a greedy strategy: rank available clips
by diversity contribution (unique start angle, under-represented cycle count, direction
balance, quality), then pick one at a time until the subset is full.
"""

def _greedy_clip_score(clip: dict, selected: list[dict], allow_radius_variation: bool) -> tuple:
    """Score a candidate clip for greedy addition to the current selection."""
    existing_angles = {c["start_angle_offset"] for c in selected}
    existing_cycles = {c["cycle_count"] for c in selected}
    existing_dirs = {c["direction"] for c in selected}
    existing_radii = {c["radius"] for c in selected if c["radius"] is not None}

    new_angle = clip["start_angle_offset"] not in existing_angles
    new_cycle = clip["cycle_count"] not in existing_cycles
    new_direction = clip["direction"] not in existing_dirs

    cycle_balance = sum(1 for c in selected if c["cycle_count"] == clip["cycle_count"])
    dir_balance = sum(1 for c in selected if c["direction"] == clip["direction"])

    radii_ok = True
    if not allow_radius_variation and clip["radius"] is not None:
        radii_ok = len(existing_radii) == 0 or clip["radius"] in existing_radii

    return (
        int(radii_ok),
        int(new_angle),
        int(new_direction),
        int(new_cycle),
        -cycle_balance,
        -dir_balance,
        clip["quality_score"],
    )


def select_group_clips(
    available_clips: list[dict],
    target_group_size: int,
    min_group_size: int,
    allow_radius_variation: bool,
) -> list[dict]:
    assert available_clips, "No clips available for group selection."
    if len(available_clips) >= target_group_size:
        subset_size = target_group_size
    elif len(available_clips) >= min_group_size:
        subset_size = len(available_clips)
    else:
        assert False, f"Candidate pool ({len(available_clips)}) is smaller than minimum group size ({min_group_size})."

    ranked = sorted(
        available_clips,
        key=lambda c: (
            c["quality_score"],
            c.get("variation_tag") or "",
            c.get("direction") or "",
            c.get("start_angle_offset", 0.0),
        ),
        reverse=True,
    )

    selected: list[dict] = [ranked[0]]
    remaining = set(range(1, len(ranked)))

    while len(selected) < subset_size and remaining:
        best_idx = max(
            remaining,
            key=lambda i: _greedy_clip_score(ranked[i], selected, allow_radius_variation),
        )
        selected.append(ranked[best_idx])
        remaining.discard(best_idx)

    return selected
