"""Select diverse frames from rendered video clips for the image benchmark.

For object groups (size, camera_distance): any frame is usable.
For pair groups (pair_distance): only frames where both anchors are co-visible.

Selection strategy: greedy diverse pick across clips, with perceptual dedup.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrameCandidate:
    clip_id: str
    frame_path: Path
    frame_index: int


def _dhash(image_path: Path, hash_size: int = 8) -> str:
    """Compute a difference hash for near-duplicate detection."""
    from PIL import Image
    img = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size))
    pixels = list(img.getdata())
    width = hash_size + 1
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            bits.append(1 if pixels[row * width + col] < pixels[row * width + col + 1] else 0)
    return "".join(str(b) for b in bits)


def _hamming(a: str, b: str) -> int:
    return sum(ca != cb for ca, cb in zip(a, b))


def _list_frames(clip_dir: Path) -> list[Path]:
    frame_dir = clip_dir / "frames"
    if not frame_dir.is_dir():
        return []
    return sorted(frame_dir.glob("frame_*.png"))


def _build_object_pool(clip_entries: list[dict], video_root: Path) -> list[FrameCandidate]:
    """Build frame pool for object-anchored groups (all frames usable)."""
    pool: list[FrameCandidate] = []
    for entry in clip_entries:
        clip_dir = video_root / entry["clip_id"]
        frames = _list_frames(clip_dir)
        for idx, frame_path in enumerate(frames):
            pool.append(FrameCandidate(clip_id=entry["clip_id"], frame_path=frame_path, frame_index=idx))
    return pool


def _build_pair_pool(clip_entries: list[dict], video_root: Path, anchor_ids: set[str]) -> list[FrameCandidate]:
    """Build frame pool for pair groups — only co-visible frames."""
    pool: list[FrameCandidate] = []
    for entry in clip_entries:
        clip_dir = video_root / entry["clip_id"]
        vis_path = clip_dir / "frame_visibility.json"
        if not vis_path.exists():
            continue
        vis = json.loads(vis_path.read_text())
        frames = _list_frames(clip_dir)
        frame_name_to_path = {f.stem: f for f in frames}
        for frame_name, visible_ids in vis["frames"].items():
            if not anchor_ids.issubset(set(visible_ids)):
                continue
            frame_path = frame_name_to_path.get(frame_name)
            if frame_path is None:
                continue
            idx = int(frame_name.split("_")[1])
            pool.append(FrameCandidate(clip_id=entry["clip_id"], frame_path=frame_path, frame_index=idx))
    return pool


def _greedy_diverse_select(pool: list[FrameCandidate], target_count: int, dedup_threshold: int = 6) -> list[FrameCandidate]:
    """Greedy selection maximizing clip diversity and perceptual variety.

    Priority:
      1. Frames from clips not yet represented
      2. Maximise frame index spacing within same clip
      3. Reject near-duplicate frames (dhash hamming distance)
    """
    if not pool:
        return []

    selected: list[FrameCandidate] = []
    selected_hashes: list[str] = []
    selected_clips: dict[str, int] = {}

    def _score(candidate: FrameCandidate) -> tuple:
        clip_count = selected_clips.get(candidate.clip_id, 0)
        min_frame_gap = min(
            (abs(candidate.frame_index - s.frame_index) for s in selected if s.clip_id == candidate.clip_id),
            default=9999,
        )
        return (-clip_count, min_frame_gap)

    remaining = list(pool)
    while len(selected) < target_count and remaining:
        remaining.sort(key=_score, reverse=True)
        picked = None
        for candidate in remaining:
            h = _dhash(candidate.frame_path)
            if any(_hamming(h, sh) < dedup_threshold for sh in selected_hashes):
                continue
            picked = candidate
            picked_hash = h
            break
        if picked is None:
            # All remaining are near-duplicates; relax dedup
            picked = remaining[0]
            picked_hash = _dhash(picked.frame_path)
        selected.append(picked)
        selected_hashes.append(picked_hash)
        selected_clips[picked.clip_id] = selected_clips.get(picked.clip_id, 0) + 1
        remaining.remove(picked)

    return selected


def select_frames(
    clip_entries: list[dict],
    anchor_kind: str,
    anchor_ids: tuple[str, ...],
    video_root: Path,
    target_count: int = 10,
) -> list[FrameCandidate]:
    """Select diverse frames for one consistency group.

    Args:
        clip_entries: qa.json entries for clips in this group.
        anchor_kind: "object" or "pair".
        anchor_ids: anchor object IDs for the group.
        video_root: path to videos/ directory.
        target_count: number of frames to select.

    Returns:
        List of FrameCandidate (up to target_count).
    """
    if anchor_kind == "pair":
        pool = _build_pair_pool(clip_entries, video_root, set(anchor_ids))
        if len(pool) < target_count:
            print(
                f"  WARN: pair group has only {len(pool)} co-visible frames "
                f"(target={target_count}), using all available",
                flush=True,
            )
    else:
        pool = _build_object_pool(clip_entries, video_root)
    return _greedy_diverse_select(pool, target_count)
