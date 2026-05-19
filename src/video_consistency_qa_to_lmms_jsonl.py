"""Flatten video-consistency `qa.json` into lmms-eval JSONL rows for `video_consistency_production`.

Usage:
  PYTHONPATH=src python src/video_consistency_qa_to_lmms_jsonl.py \\
    --qa_json ./video_consistency_thor_eval/qa.json \\
    --out_jsonl /nas2/edwin/lmms-eval/data/video_consistency_thor_eval.jsonl

Input spec:
  JSON array of clip dicts with `questions[]` (see `plan_io._qa_entry`).

Output spec:
  One JSON object per line with top-level fields: `question`, `ground_truth`,
  `clip_id`, `group_id`, `video_path`, `question_type`, `question_family`,
  `engine`, `motion_family`, and the same metadata keys as
  `video_consistency_production_merged.jsonl` so `sceneshift_*` hooks apply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _clip_to_rows(clip: dict) -> list[dict]:
    out: list[dict] = []
    video_path = clip.get("video_path", "")
    vp = Path(video_path) if video_path else Path()
    output_dir = str(vp.parent) if vp.parts else ""
    base = {
        "clip_id": clip.get("clip_id", ""),
        "group_id": clip.get("group_id", ""),
        "engine": clip.get("engine", ""),
        "scene_id": clip.get("scene_id", ""),
        "room_bucket": clip.get("room_bucket", ""),
        "motion_family": clip.get("motion_family", ""),
        "trajectory": clip.get("trajectory", ""),
        "direction": clip.get("direction", ""),
        "variation_tag": clip.get("variation_tag", ""),
        "radius": clip.get("radius"),
        "start_angle_offset": clip.get("start_angle_offset"),
        "cycle_count": clip.get("cycle_count"),
        "quality_score": clip.get("quality_score"),
        "video_path": video_path,
        "output_dir": output_dir,
    }
    for q in clip.get("questions") or []:
        row = {
            **base,
            "question": q.get("question", ""),
            "ground_truth": q.get("answer", ""),
            "question_type": q.get("question_type", ""),
            "question_family": q.get("question_family", ""),
            "dimension": q.get("dimension"),
            "anchor_kind": q.get("anchor_kind", ""),
            "anchor_ids": q.get("anchor_ids"),
            "anchor_labels": q.get("anchor_labels"),
        }
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa_json", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    args = ap.parse_args()

    payload = json.loads(args.qa_json.expanduser().resolve().read_text(encoding="utf-8"))
    assert isinstance(payload, list), args.qa_json

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.expanduser().resolve().open("w", encoding="utf-8") as sink:
        for clip in payload:
            for row in _clip_to_rows(clip):
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
