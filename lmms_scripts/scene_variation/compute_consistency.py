"""
Compute per-(model, task, question_type) consistency from combined sample
JSONs and print a markdown table.

Usage:
    python scripts/scene_variation/compute_consistency.py
    python scripts/scene_variation/compute_consistency.py --combined-dir /custom/path

Definitions:
    For each group_id (with at least 2 successfully parsed predictions and a
    positive mean), CV = std(predictions, ddof=1) / mean(predictions).
    Per question_type within a task, mean_CV = average CV across that
    question_type's groups. Then:
        consistency_score = 1 - exp(-mean_CV)
    A score of 0 means perfectly consistent (predictions identical across
    variations); higher = more spread.

Input:
    /nas2/edwin/lmms-eval/results/combined/<model>.json
        Bundles produced by combine_results.py. Each sample dict has
        sceneshift_score sub-dict with prediction_parse, group_id,
        question_type, plus a top-level "_task" field.

Output:
    Prints a markdown table to stdout, one row per (model, task,
    question_type), columns: n_samples, n_groups, mean_CV, consistency.
    Optionally writes the same table as TSV with --tsv-out PATH.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_COMBINED_DIR = Path("/nas2/edwin/lmms-eval/results/combined")


def _to_float(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _group_cv(predictions: list[float]) -> float | None:
    arr = np.asarray(predictions, dtype=float)
    if len(arr) < 2:
        return None
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    if mu == 0:
        return 0.0 if sigma == 0 or math.isnan(sigma) else None
    if mu < 0:
        return None
    return sigma / mu


def _extract_metric_dict(sample: dict) -> dict:
    score = sample.get("sceneshift_score")
    assert isinstance(score, dict), f"Sample missing sceneshift_score: {sample.get('doc_id')}"
    return score


def _compute_for_bundle(bundle: dict) -> list[dict]:
    rows: list[dict] = []
    by_task_qtype: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in bundle["samples"]:
        task = sample.get("_task") or "unknown"
        score = _extract_metric_dict(sample)
        qtype = score.get("question_type") or "unknown"
        by_task_qtype[(task, qtype)].append(score)

    for (task, qtype), entries in sorted(by_task_qtype.items()):
        groups: dict[str, list[float]] = defaultdict(list)
        parse_success = 0
        for entry in entries:
            pred = _to_float(entry.get("prediction_parse"))
            gid = entry.get("group_id")
            if pred is None or gid is None:
                continue
            parse_success += 1
            groups[gid].append(pred)

        cvs: list[float] = []
        for preds in groups.values():
            cv = _group_cv(preds)
            if cv is not None:
                cvs.append(cv)

        n_samples = len(entries)
        n_groups_total = len(groups)
        n_groups_used = len(cvs)
        mean_cv = float(np.mean(cvs)) if cvs else None
        consistency = (1 - math.exp(-mean_cv)) if mean_cv is not None else None
        rows.append({
            "model": bundle["model"],
            "task": task,
            "question_type": qtype,
            "n_samples": n_samples,
            "parse_success_n": parse_success,
            "parse_success_rate": (parse_success / n_samples) if n_samples else None,
            "n_groups_total": n_groups_total,
            "n_groups_used": n_groups_used,
            "mean_CV": mean_cv,
            "consistency_score": consistency,
        })
    return rows


_TASK_SHORT = {
    "scene_variation": "sv",
    "image_consistency_thor_eval_v2": "img_v2",
    "video_consistency_thor_eval_v2": "vid_v2",
}
_QTYPE_SHORT = {
    "object_dimensions": "dim",
    "object_distance_to_camera": "dist_cam",
    "object_pair_distance_center": "pair_dist",
    "object_size": "size",
}


def _format_markdown(rows: list[dict]) -> str:
    headers = ["model", "task", "question_type", "n_samples", "n_groups", "parse_rate", "mean_CV", "consistency=1-exp(-CV)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cv = "—" if r["mean_CV"] is None else f"{r['mean_CV']:.4f}"
        cs = "—" if r["consistency_score"] is None else f"{r['consistency_score']:.4f}"
        rate = "—" if r["parse_success_rate"] is None else f"{r['parse_success_rate']:.3f}"
        lines.append(
            f"| {r['model']} | {r['task']} | {r['question_type']} | {r['n_samples']} | "
            f"{r['n_groups_used']}/{r['n_groups_total']} | {rate} | {cv} | {cs} |"
        )
    return "\n".join(lines)


def _format_pivot_markdown(rows: list[dict]) -> str:
    columns: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["task"], r["question_type"])
        if key not in seen:
            seen.add(key)
            columns.append(key)
    columns.sort(key=lambda c: (c[0], c[1]))

    by_model: dict[str, dict[tuple[str, str], dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[(r["task"], r["question_type"])] = r

    short_cols = [f"{_TASK_SHORT.get(t, t)}:{_QTYPE_SHORT.get(q, q)}" for t, q in columns]
    headers = ["model"] + short_cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for model in sorted(by_model):
        cells = [model]
        for col in columns:
            r = by_model[model].get(col)
            cs = r and r.get("consistency_score")
            cells.append("—" if cs is None else f"{cs:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_tsv(rows: list[dict]) -> str:
    headers = ["model", "task", "question_type", "n_samples", "parse_success_n",
               "parse_success_rate", "n_groups_total", "n_groups_used", "mean_CV", "consistency_score"]
    lines = ["\t".join(headers)]
    for r in rows:
        lines.append("\t".join("" if r[h] is None else str(r[h]) for h in headers))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-dir", default=str(DEFAULT_COMBINED_DIR))
    parser.add_argument("--tsv-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    combined_dir = Path(args.combined_dir)
    assert combined_dir.exists(), f"Missing combined dir: {combined_dir}. Run combine_results.py first."

    all_rows: list[dict] = []
    for json_path in sorted(combined_dir.glob("*.json")):
        with json_path.open() as f:
            bundle = json.load(f)
        rows = _compute_for_bundle(bundle)
        all_rows.extend(rows)

    md = _format_markdown(all_rows)
    pivot = _format_pivot_markdown(all_rows)
    print("## Long table")
    print(md)
    print()
    print("## Pivot (consistency = 1-exp(-CV); lower = more consistent)")
    print(pivot)

    if args.markdown_out:
        out = Path(args.markdown_out)
        out.write_text("## Long table\n" + md + "\n\n## Pivot (consistency = 1-exp(-CV))\n" + pivot + "\n")
        print(f"\nWrote markdown to {out}")

    if args.tsv_out:
        Path(args.tsv_out).write_text(_format_tsv(all_rows) + "\n")
        print(f"Wrote TSV to {args.tsv_out}")


if __name__ == "__main__":
    main()
