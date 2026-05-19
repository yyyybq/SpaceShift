"""Scan lmms-eval result dirs and build a per-model × per-question-type MRA/CV table.

Usage:
  python src/summarize_video_consistency_results.py
  python src/summarize_video_consistency_results.py \
      --results_root /nas2/edwin/lmms-eval/results \
      --output figures/vc_summary.md

Input spec:
  --results_root  Directory containing per-run folders (default: /nas2/edwin/lmms-eval/results)
  --task_glob     Only dirs matching this glob (default: *video_consistency_production)
  --exclude_prefix  Skip exp dirs whose name starts with this (repeatable, default: tmp_)
  --output        Write markdown here; omit to print stdout

Output spec:
  Markdown table with rows = models, columns = question types (+ Overall).
  Each question type gets two sub-columns: MRA (%) and CV.

  <output_dir>/
    <output>.md    markdown table

Result JSON structure consumed:
  results.video_consistency_production.sceneshift_score,none.{
    overall_MRA, overall_mean_CV,
    qtype:<name>_MRA, qtype:<name>_mean_CV, ...
  }
"""

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path


QUESTION_TYPES = [
    "object_dimensions",
    "object_distance_to_camera",
    "object_pair_distance_center",
]

QUESTION_TYPE_LABELS = {
    "object_dimensions": "Object Dimensions",
    "object_distance_to_camera": "Camera Distance",
    "object_pair_distance_center": "Pair Distance",
}


def _short_model_name(full_name: str) -> str:
    return full_name.rsplit("/", 1)[-1] if "/" in full_name else full_name


def _pick_latest_json(exp_dir: Path) -> Path | None:
    candidates = list(exp_dir.rglob("*_results.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_result(path: Path) -> dict | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload.get("results", {}).get("video_consistency_production", {})
    block = None
    for v in task.values():
        if isinstance(v, dict) and "overall_MRA" in v:
            block = dict(v)
            break
    if block is None:
        return None
    model = payload.get("model_name", "")
    return {
        "model": _short_model_name(model),
        "model_full": model,
        "block": block,
    }


def _collect(results_root: Path, task_glob: str, exclude_prefixes: list[str]) -> list[dict]:
    rows = []
    for exp_dir in sorted(results_root.glob(task_glob)):
        if not exp_dir.is_dir():
            continue
        if any(exp_dir.name.startswith(p) for p in exclude_prefixes if p):
            continue
        latest = _pick_latest_json(exp_dir)
        if latest is None:
            continue
        entry = _load_result(latest)
        if entry is not None:
            rows.append(entry)
    return rows


def _fmt(v, nd=2):
    if v is None:
        return "—"
    return f"{float(v):.{nd}f}"


def _build_table(models: list[dict]) -> str:
    models = sorted(models, key=lambda m: m["model"].lower())

    col_keys = []
    col_labels = []
    for qtype in QUESTION_TYPES:
        label = QUESTION_TYPE_LABELS.get(qtype, qtype)
        col_keys.append(("qtype", qtype))
        col_labels.append(label)
    col_keys.append(("overall", None))
    col_labels.append("**Overall**")

    header = "| Model |"
    divider = "| --- |"
    for label in col_labels:
        header += f" {label} MRA ↑ | {label} CV ↓ |"
        divider += " ---: | ---: |"

    lines = [header, divider]

    for m in models:
        b = m["block"]
        row = f"| {m['model']} |"
        for kind, qtype in col_keys:
            if kind == "overall":
                mra = b.get("overall_MRA")
                cv = b.get("overall_mean_CV")
            else:
                mra = b.get(f"qtype:{qtype}_MRA")
                cv = b.get(f"qtype:{qtype}_mean_CV")
            bold = kind == "overall"
            m_str = f"**{_fmt(mra)}**" if bold else _fmt(mra)
            c_str = f"**{_fmt(cv, 4)}**" if bold else _fmt(cv, 4)
            row += f" {m_str} | {c_str} |"
        lines.append(row)

    lines.append("")
    lines.append(
        "_MRA = Mean Relative Accuracy (%, higher better). "
        "CV = mean within-group Coefficient of Variation (lower = more consistent)._"
    )
    return "\n".join(lines)


def _build_json(models: list[dict]) -> list[dict]:
    out = []
    for m in models:
        b = m["block"]
        entry = OrderedDict()
        entry["model"] = m["model"]
        entry["model_full"] = m["model_full"]
        entry["overall_MRA"] = b.get("overall_MRA")
        entry["overall_CV"] = b.get("overall_mean_CV")
        entry["overall_n"] = b.get("overall_n")
        entry["overall_n_groups"] = b.get("overall_n_groups")
        for qtype in QUESTION_TYPES:
            entry[f"{qtype}_MRA"] = b.get(f"qtype:{qtype}_MRA")
            entry[f"{qtype}_CV"] = b.get(f"qtype:{qtype}_mean_CV")
            entry[f"{qtype}_n"] = b.get(f"qtype:{qtype}_n")
            entry[f"{qtype}_n_groups"] = b.get(f"qtype:{qtype}_n_groups")
        out.append(entry)
    return out


def main() -> int:
    default_root = Path(os.environ.get("LMMS_EVAL_RESULTS", "/nas2/edwin/lmms-eval/results"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", type=Path, default=default_root)
    p.add_argument("--task_glob", default="*video_consistency_production")
    p.add_argument("--exclude_prefix", action="append", default=["tmp_"])
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    assert args.results_root.is_dir(), f"Missing results root: {args.results_root}"

    models = _collect(args.results_root, args.task_glob, args.exclude_prefix)
    if not models:
        md = (
            "_No `*video_consistency_production` result JSON under "
            f"`{args.results_root}` yet. Run eval first._\n"
        )
    else:
        md = _build_table(models)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        json_path = args.output.with_suffix(".json")
        json_path.write_text(json.dumps(_build_json(models), indent=2), encoding="utf-8")
        print(f"Wrote {args.output} and {json_path}")
    else:
        print(md)
        print()
        print(json.dumps(_build_json(models), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
