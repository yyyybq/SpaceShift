"""Build single-image spatial tuning JSONL: N frames from 1 clip per group.

Usage:
    PYTHONPATH=src python src/training/single_image_builder.py \\
        --qa_json train/qa.json \\
        --groups_json train/consistency_groups.json \\
        --out_jsonl output.jsonl --frames_per_group 3

Input:
    qa.json              -- One or more JSON arrays of clip dicts with nested
                            questions[].  Multiple files are concatenated.
    consistency_groups.json -- One or more JSON arrays of group dicts.

Output:
    JSONL with frames_per_group rows per group.  Each row:
        {"image": "<frame_path>", "conversations": [...]}

Strategy:
    For each group, pick the highest quality_score clip, sample N distinct
    frames from it, adapt question text "in this video" -> "in this image".
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from training.frame_sampler import clip_has_frames, sample_frames
from training.question_adapter import adapt_question_for_image

FRAMES_PER_GROUP = 3


def _clip_dir(clip: dict) -> Path:
    return Path(clip["video_path"]).parent


def _index_clips_by_group(qa: list[dict]) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = {}
    for clip in qa:
        if not clip_has_frames(_clip_dir(clip)):
            continue
        gid = clip["group_id"]
        by_group.setdefault(gid, []).append(clip)
    return by_group


def _best_clip(clips: list[dict]) -> dict:
    return max(clips, key=lambda c: c.get("quality_score", 0.0))


def _make_row(image_path: str, question: str, answer: str) -> dict:
    return {
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def build_single_image_jsonl(
    qa: list[dict],
    groups: list[dict],
    frames_per_group: int = FRAMES_PER_GROUP,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    by_group = _index_clips_by_group(qa)
    rows: list[dict] = []
    for group in groups:
        gid = group["group_id"]
        clips = by_group.get(gid, [])
        if not clips:
            continue
        clip = _best_clip(clips)
        q = clip["questions"][0]
        question_type = q["question_type"]
        frames = sample_frames(_clip_dir(clip), question_type, count=frames_per_group, rng=rng)
        question = adapt_question_for_image(q["question"])
        answer = q["answer"]
        for frame in frames:
            rows.append(_make_row(str(frame), question, answer))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa_json", type=Path, required=True, nargs="+")
    ap.add_argument("--groups_json", type=Path, required=True, nargs="+")
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--frames_per_group", type=int, default=FRAMES_PER_GROUP)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qa: list[dict] = []
    for p in args.qa_json:
        qa.extend(json.loads(p.read_text(encoding="utf-8")))
    groups: list[dict] = []
    for p in args.groups_json:
        groups.extend(json.loads(p.read_text(encoding="utf-8")))

    rows = build_single_image_jsonl(qa, groups, frames_per_group=args.frames_per_group, seed=args.seed)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
