"""Merge two production `qa.json` files and write merged JSON plus detailed QA stats.

Usage:
  PYTHONPATH=src python src/merge_production_qa_report.py \\
    --thor_qa ./video_consistency_thor_production/qa.json \\
    --interiorgs_qa ./video_consistency_interiorGS_production/qa.json \\
    --out_merged ./video_consistency_production_figures/qa_merged.json \\
    --out_stats ./video_consistency_production_figures/qa_merged_stats.json

Input: two JSON arrays of clip dicts (each with `questions[]`).

Output:
  - `out_merged`: concatenated clips (Thor first), `clip_id` must be unique.
  - `out_stats`: nested counters / lists with counts and shares for all distributions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _load_qa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list), path
    return data


def _share_rows(counter: Counter[str], total: int) -> list[dict]:
    rows = []
    for k, v in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        rows.append({"key": k, "count": v, "share_pct": round(100.0 * v / total, 4) if total else 0.0})
    return rows


def _q_label(q: dict) -> str:
    al = q.get("anchor_labels") or []
    qt = q.get("question_type", "")
    if isinstance(al, list) and al:
        if qt == "object_pair_distance_center" and len(al) >= 2:
            a = str(al[0]).split("|")[0].strip()
            b = str(al[1]).split("|")[0].strip()
            return "–".join(sorted([a, b]))
        return str(al[0]).split("|")[0].strip() or "unknown"
    return "unknown"


def _parse_variation(tag: str) -> tuple[str, str]:
    if not tag:
        return ("", "")
    s = str(tag).strip()
    m = re.match(r"^(\d+x)(?:_(.+))?$", s, re.I)
    if m:
        return (m.group(1).lower(), (m.group(2) or "").lower())
    return (s, "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thor_qa", type=Path, required=True)
    ap.add_argument("--interiorgs_qa", type=Path, required=True)
    ap.add_argument("--out_merged", type=Path, required=True)
    ap.add_argument("--out_stats", type=Path, required=True)
    args = ap.parse_args()
    thor = _load_qa(args.thor_qa.expanduser().resolve())
    intgs = _load_qa(args.interiorgs_qa.expanduser().resolve())
    merged = thor + intgs
    ids = [c["clip_id"] for c in merged]
    assert len(ids) == len(set(ids)), "duplicate clip_id after merge"
    args.out_merged.parent.mkdir(parents=True, exist_ok=True)
    args.out_merged.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    n_clips = len(merged)
    qa_rows: list[dict] = []
    q_per_clip = Counter()
    for c in merged:
        qs = c.get("questions") or []
        q_per_clip[len(qs)] += 1
        for q in qs:
            mf = c.get("motion_family", "")
            eng = c.get("engine", "")
            var = c.get("variation_tag", "")
            cyc, sfx = _parse_variation(var)
            qa_rows.append({
                "engine": eng,
                "question_type": q.get("question_type", ""),
                "question_family": q.get("question_family", ""),
                "dimension": q.get("dimension") or "",
                "anchor_kind": q.get("anchor_kind", ""),
                "motion_family": mf,
                "trajectory": c.get("trajectory", ""),
                "direction": c.get("direction", ""),
                "scene_id": c.get("scene_id", ""),
                "group_id": c.get("group_id", ""),
                "variation_tag": var,
                "variation_cycle": cyc,
                "variation_suffix": sfx,
                "clip_id": c.get("clip_id", ""),
                "object_label": _q_label(q),
            })

    n_qa = len(qa_rows)
    assert n_qa > 0

    def ctr(f: str) -> Counter[str]:
        return Counter(str(r[f]) for r in qa_rows)

    def ctr_nonempty(f: str) -> Counter[str]:
        return Counter(str(r[f]) for r in qa_rows if r[f])

    def pair_ctr(fa: str, fb: str) -> Counter[str]:
        out: Counter[str] = Counter()
        for r in qa_rows:
            out[f"{r[fa]}|{r[fb]}"] += 1
        return out

    def pair_ctr_engine_dimension() -> Counter[str]:
        out: Counter[str] = Counter()
        for r in qa_rows:
            if not r["dimension"]:
                continue
            out[f"{r['engine']}|{r['dimension']}"] += 1
        return out

    n_dim = sum(1 for r in qa_rows if r["dimension"])
    stats = {
        "summary": {
            "thor_clips": len(thor),
            "interiorgs_clips": len(intgs),
            "merged_clips": n_clips,
            "merged_qa_instances": n_qa,
            "questions_per_clip": _share_rows(q_per_clip, n_clips),
        },
        "qa_by": {
            "engine": _share_rows(ctr("engine"), n_qa),
            "question_type": _share_rows(ctr("question_type"), n_qa),
            "question_family": _share_rows(ctr("question_family"), n_qa),
            "dimension": _share_rows(ctr_nonempty("dimension"), sum(1 for r in qa_rows if r["dimension"])),
            "anchor_kind": _share_rows(ctr("anchor_kind"), n_qa),
            "motion_family": _share_rows(ctr("motion_family"), n_qa),
            "trajectory": _share_rows(ctr("trajectory"), n_qa),
            "direction": _share_rows(ctr("direction"), n_qa),
            "variation_tag": _share_rows(ctr("variation_tag"), n_qa),
            "variation_cycle": _share_rows(ctr_nonempty("variation_cycle"), n_qa),
            "variation_suffix": _share_rows(ctr_nonempty("variation_suffix"), n_qa),
            "scene_id_per_qa": _share_rows(ctr("scene_id"), n_qa),
            "object_or_pair_label": _share_rows(ctr("object_label"), n_qa),
        },
        "joint_qa": {
            "engine_question_type": _share_rows(pair_ctr("engine", "question_type"), n_qa),
            "engine_question_family": _share_rows(pair_ctr("engine", "question_family"), n_qa),
            "engine_motion_family": _share_rows(pair_ctr("engine", "motion_family"), n_qa),
            "engine_dimension": _share_rows(pair_ctr_engine_dimension(), n_dim),
            "question_type_motion_family": _share_rows(pair_ctr("question_type", "motion_family"), n_qa),
            "engine_variation_cycle": _share_rows(pair_ctr("engine", "variation_cycle"), n_qa),
            "engine_variation_suffix": _share_rows(pair_ctr("engine", "variation_suffix"), n_qa),
        },
        "clip_level": {
            "engine": _share_rows(Counter(str(c.get("engine", "")) for c in merged), n_clips),
            "motion_family": _share_rows(Counter(str(c.get("motion_family", "")) for c in merged), n_clips),
            "trajectory": _share_rows(Counter(str(c.get("trajectory", "")) for c in merged), n_clips),
            "scene_id": _share_rows(Counter(str(c.get("scene_id", "")) for c in merged), n_clips),
            "group_id": _share_rows(Counter(str(c.get("group_id", "")) for c in merged), n_clips),
            "variation_tag": _share_rows(Counter(str(c.get("variation_tag", "")) for c in merged), n_clips),
        },
    }
    args.out_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats["summary"], indent=2))
    print(f"Wrote {args.out_merged}")
    print(f"Wrote {args.out_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
