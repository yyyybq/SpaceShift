"""Sample frames from pre-rendered clip directories for training data.

Usage:
    from training.frame_sampler import sample_frame

    frame_path = sample_frame(clip_dir, question_type, seed=42)

Input:
    clip_dir      -- Path to clip directory containing frames/frame_NNNNN.png
    question_type -- One of: object_dimensions, object_distance_to_camera,
                     object_pair_distance_center

Output:
    Absolute Path to a selected frame PNG.

Strategy:
    - object_dimensions / object_distance_to_camera: random frame (object is
      the camera focus, visible in virtually all frames).
    - object_pair_distance_center: random frame from the middle third of the
      clip (room-rotation trajectory; both anchor objects are most likely
      co-visible in the middle portion).
"""

from __future__ import annotations

import random
from pathlib import Path


PAIR_DISTANCE_TYPE = "object_pair_distance_center"


def clip_has_frames(clip_dir: Path) -> bool:
    """Return True if *clip_dir* contains a non-empty frames/ subdirectory."""
    frames_dir = clip_dir / "frames"
    if not frames_dir.is_dir():
        return False
    return any(frames_dir.glob("frame_*.png"))


def _sorted_frames(clip_dir: Path) -> list[Path]:
    frames_dir = clip_dir / "frames"
    assert frames_dir.is_dir(), f"No frames/ in {clip_dir}"
    frames = sorted(frames_dir.glob("frame_*.png"))
    assert frames, f"No frame PNGs in {frames_dir}"
    return frames


def sample_frame(
    clip_dir: Path,
    question_type: str,
    rng: random.Random | None = None,
) -> Path:
    """Return a single frame path suitable for the given question type."""
    if rng is None:
        rng = random.Random()
    frames = _sorted_frames(clip_dir)
    if question_type == PAIR_DISTANCE_TYPE:
        n = len(frames)
        lo = n // 3
        hi = 2 * n // 3
        candidates = frames[lo:hi]
        assert candidates, f"Middle-third slice empty for {clip_dir}"
    else:
        candidates = frames
    return rng.choice(candidates)


def sample_frames(
    clip_dir: Path,
    question_type: str,
    count: int = 1,
    rng: random.Random | None = None,
) -> list[Path]:
    """Return *count* distinct frame paths from one clip."""
    if rng is None:
        rng = random.Random()
    frames = _sorted_frames(clip_dir)
    if question_type == PAIR_DISTANCE_TYPE:
        n = len(frames)
        candidates = frames[n // 3 : 2 * n // 3]
    else:
        candidates = frames
    k = min(count, len(candidates))
    return rng.sample(candidates, k)
