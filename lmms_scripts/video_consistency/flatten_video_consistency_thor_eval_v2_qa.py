"""Flatten video_consistency_thor_eval_v2 qa.json into lmms-eval JSONL rows.

Usage:
  PYTHONPATH=. python scripts/video_consistency/flatten_video_consistency_thor_eval_v2_qa.py \
    --qa-json /nas2/edwin/spatial-scene-variations/video_consistency_thor_eval_v2/qa.json \
    --repo-root /nas2/edwin/spatial-scene-variations \
    --out-jsonl /nas2/edwin/lmms-eval/data/video_consistency_thor_eval_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _normalize_video_path(repo_root: Path, video_path: str) -> str:
    raw = Path(video_path).expanduser()
    if raw.is_absolute():
        return str(raw)
    return str((repo_root / raw).resolve())


def _clip_to_rows(clip: dict, repo_root: Path) -> list[dict]:
    normalized_video_path = _normalize_video_path(repo_root, clip.get("video_path", ""))
    vp = Path(normalized_video_path)
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
        "video_path": normalized_video_path,
        "output_dir": str(vp.parent) if normalized_video_path else "",
    }
    out: list[dict] = []
    for question in clip.get("questions") or []:
        out.append({
            **base,
            "question": question.get("question", ""),
            "ground_truth": question.get("answer", ""),
            "question_type": question.get("question_type", ""),
            "question_family": question.get("question_family", ""),
            "dimension": question.get("dimension"),
            "anchor_kind": question.get("anchor_kind", ""),
            "anchor_ids": question.get("anchor_ids"),
            "anchor_labels": question.get("anchor_labels"),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    args = parser.parse_args()

    qa_json_path = args.qa_json.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    payload = json.loads(qa_json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list), qa_json_path

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with args.out_jsonl.expanduser().resolve().open("w", encoding="utf-8") as sink:
        for clip in payload:
            for row in _clip_to_rows(clip, repo_root):
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"wrote {n_rows} rows to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
