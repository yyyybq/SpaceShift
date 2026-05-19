"""Materialize image benchmark files from source video.mp4 clips.

This repairs image benchmarks that currently store broken symlinks to deleted
frame PNGs by extracting the requested frame from the original source video.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


def _discover_source_videos(root: Path) -> dict[str, list[Path]]:
    source_by_clip: dict[str, list[Path]] = defaultdict(list)
    for video_path in sorted(root.glob("old/**/videos/*/video.mp4")):
        clip_id = video_path.parent.name
        source_by_clip[clip_id].append(video_path)
    return dict(source_by_clip)


def _parse_frame_index(name: str) -> int:
    stem = Path(name).stem
    assert stem.startswith("frame_"), f"Unexpected frame name: {name}"
    return int(stem.split("_", 1)[1])


def _materialize_clip_frames(video_path: Path, requests: list[tuple[int, Path]]) -> None:
    capture = cv2.VideoCapture(str(video_path))
    assert capture.isOpened(), f"Failed to open video: {video_path}"
    try:
        for frame_index, target_path in sorted(requests, key=lambda item: item[0]):
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            assert ok and frame is not None, f"Failed to decode frame {frame_index} from {video_path}"
            ok = cv2.imwrite(str(target_path), frame)
            assert ok, f"Failed to write image: {target_path}"
    finally:
        capture.release()


def _frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    assert capture.isOpened(), f"Failed to open video: {video_path}"
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _video_from_symlink(target_path: Path) -> Path | None:
    if not target_path.is_symlink():
        return None
    frame_path = target_path.resolve(strict=False)
    if frame_path.parent.name != "frames":
        return None
    video_path = frame_path.parent.parent / "video.mp4"
    return video_path if video_path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="image_consistency_thor_eval_v2")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    dataset_dir = (root / args.dataset_dir).resolve()
    qa_path = dataset_dir / "qa.json"
    rows = json.loads(qa_path.read_text(encoding="utf-8"))
    source_by_clip = _discover_source_videos(root)
    frame_count_cache: dict[Path, int] = {}

    requests_by_video: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
    for row in rows:
        target_path = dataset_dir / "images" / str(row["image_id"]) / "image.png"
        if target_path.exists() and not target_path.is_symlink():
            continue
        frame_index = _parse_frame_index(str(row["source_frame"]))
        video_path = _video_from_symlink(target_path)
        if video_path is None:
            clip_id = str(row["source_clip_id"])
            candidates = source_by_clip.get(clip_id, [])
            assert candidates, f"Missing source video for clip {clip_id}"
            viable = []
            for candidate in candidates:
                count = frame_count_cache.setdefault(candidate, _frame_count(candidate))
                if count > frame_index:
                    viable.append((count, candidate))
            assert viable, f"No source video with frame {frame_index} for clip {clip_id}"
            _, video_path = max(viable, key=lambda item: (item[0], str(item[1])))
        requests_by_video[video_path].append((frame_index, target_path))

    for video_path, requests in requests_by_video.items():
        _materialize_clip_frames(video_path, requests)

    symlink_count = sum(1 for path in dataset_dir.rglob("*") if path.is_symlink())
    file_count = sum(1 for path in (dataset_dir / "images").rglob("image.png") if path.is_file() and not path.is_symlink())
    print(
        {
            "images_total": len(rows),
            "images_written_this_run": sum(len(v) for v in requests_by_video.values()),
            "source_videos_used": len(requests_by_video),
            "materialized_file_count": file_count,
            "symlink_count": symlink_count,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
