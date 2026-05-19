"""Print distribution statistics for a spatial-training JSONL file.

Usage:
    PYTHONPATH=src python src/training/training_data_stats.py \\
        --jsonl /nas/edwin_n/sceneshift_training/qwen-vl-finetune/data/spatial_single_image.jsonl

Input:
    JSONL file with rows containing "image" or "video" key, plus
    "conversations" with human/gpt turns.

Output (stdout):
    Total rows, modality split (image vs video), question-type distribution
    (inferred from question text keywords), and answer-value histogram.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _infer_question_type(question: str) -> str:
    q = question.lower()
    if "distance between" in q or "distance from" in q and "camera" not in q:
        return "pair_distance"
    if "distance" in q and "camera" in q:
        return "camera_distance"
    if "width" in q or "height" in q or "length" in q:
        return "size"
    return "unknown"


def _extract_answer(row: dict) -> str:
    for turn in row.get("conversations", []):
        if turn.get("from") == "gpt":
            return turn["value"]
    return ""


def _extract_question(row: dict) -> str:
    for turn in row.get("conversations", []):
        if turn.get("from") == "human":
            text = turn["value"]
            for tag in ("<image>", "<video>"):
                text = text.replace(tag, "")
            return text.strip()
    return ""


def report(jsonl_path: Path) -> None:
    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    modality = Counter("video" if "video" in r else "image" for r in rows)
    qtypes = Counter(_infer_question_type(_extract_question(r)) for r in rows)
    answers = Counter(_extract_answer(r) for r in rows)

    print(f"File: {jsonl_path}")
    print(f"Total rows: {n}")
    print(f"Modality:   {dict(modality)}")
    print(f"Question types (inferred): {dict(qtypes)}")
    print(f"Unique answers: {len(answers)}")

    top = answers.most_common(10)
    print("Top-10 answers:")
    for ans, cnt in top:
        print(f"  {ans!r:>8s}  {cnt:>4d}  ({100*cnt/n:.1f}%)")

    try:
        vals = [float(v) for v in answers]
        print(f"Answer range: [{min(vals):.1f}, {max(vals):.1f}]")
    except ValueError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, required=True, nargs="+")
    args = ap.parse_args()
    for p in args.jsonl:
        report(p)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
