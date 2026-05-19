"""Build multi-image spatial tuning JSONL: N frames per consistency group.

Usage:
    PYTHONPATH=src python src/training/multi_image_builder.py \\
        --qa_json train/qa.json supplement/qa.json \\
        --groups_json train/consistency_groups.json supplement/consistency_groups.json \\
        --out_jsonl /nas/edwin_n/sceneshift_training/qwen-vl-finetune/data/spatial_multi_image.jsonl

Input:
    qa.json              -- One or more JSON arrays of clip dicts.
    consistency_groups.json -- One or more JSON arrays of group dicts.

Output:
    JSONL with clips_per_group rows per group.  Same format as single-image.

Strategy:
    For each group, pick clips_per_group clips with highest quality_score
    (spread across distinct variation_tag values).  From each clip,
    sample 1 frame.  All rows per group share the same question/answer.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from training.frame_sampler import clip_has_frames, sample_frame
from training.question_adapter import adapt_question_for_image

CLIPS_PER_GROUP = 5


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


def _select_diverse_clips(clips: list[dict], k: int) -> list[dict]:
    """Pick *k* clips maximising variation_tag diversity, breaking ties by quality_score."""
    ranked = sorted(clips, key=lambda c: c.get("quality_score", 0.0), reverse=True)
    selected: list[dict] = []
    seen_tags: set[str] = set()
    for clip in ranked:
        tag = clip.get("variation_tag", "")
        if tag not in seen_tags:
            selected.append(clip)
            seen_tags.add(tag)
        if len(selected) == k:
            break
    if len(selected) < k:
        for clip in ranked:
            if clip not in selected:
                selected.append(clip)
            if len(selected) == k:
                break
    return selected


def _make_row(image_path: str, question: str, answer: str) -> dict:
    return {
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def build_multi_image_jsonl(
    qa: list[dict],
    groups: list[dict],
    clips_per_group: int = CLIPS_PER_GROUP,
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
        chosen = _select_diverse_clips(clips, clips_per_group)
        for clip in chosen:
            q = clip["questions"][0]
            question_type = q["question_type"]
            frame = sample_frame(_clip_dir(clip), question_type, rng=rng)
            question = adapt_question_for_image(q["question"])
            answer = q["answer"]
            rows.append(_make_row(str(frame), question, answer))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa_json", type=Path, required=True, nargs="+")
    ap.add_argument("--groups_json", type=Path, required=True, nargs="+")
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--clips_per_group", type=int, default=CLIPS_PER_GROUP)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qa: list[dict] = []
    for p in args.qa_json:
        qa.extend(json.loads(p.read_text(encoding="utf-8")))
    groups: list[dict] = []
    for p in args.groups_json:
        groups.extend(json.loads(p.read_text(encoding="utf-8")))

    rows = build_multi_image_jsonl(
        qa, groups, clips_per_group=args.clips_per_group, seed=args.seed,
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
