"""Build a markdown table of average raw-answer variance by task and model.

Usage:
  PYTHONPATH=src python src/build_lmms_raw_consistency_variance_table.py \
    --results_dir /nas2/edwin/lmms-eval/results \
    --out_path /nas2/edwin/lmms-eval/results/consistency_variance_plots/raw_consistency_variance_table.md

Input spec:
  `--results_dir` must contain model result trees with files named
  `*_samples_video_consistency_production.jsonl`.
  Each JSONL row should include `filtered_resps` or `sceneshift_score.prediction`,
  plus `group_id`, `question`, and `question_type`.

Output spec:
  Writes a markdown file with models as rows, tasks as columns, and each cell
  showing the average per-question variance across video variations. The final
  column reports each model's mean across tasks.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from plot_lmms_raw_consistency_variance import (
    DEFAULT_EXCLUDED_MODELS,
    QUESTION_TYPE_DISPLAY,
    _display_model,
    _latest_sample_paths,
    _load_model_variances,
    _model_label,
)


def _build_table(model_to_variances: dict[str, dict[str, list[float]]], task_order: list[str]) -> str:
    models = sorted(model_to_variances)
    header = ["Model"] + [QUESTION_TYPE_DISPLAY.get(qtype, qtype) for qtype in task_order] + ["Average"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for model in models:
        row = [_display_model(model)]
        averages: list[float] = []
        for qtype in task_order:
            vals = model_to_variances[model].get(qtype, [])
            if vals:
                mean_value = statistics.mean(vals)
                row.append(f"{mean_value:.4f}")
                averages.append(mean_value)
            else:
                row.append("NA")
        row.append(f"{statistics.mean(averages):.4f}" if averages else "NA")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument(
        "--exclude_models",
        default=",".join(sorted(DEFAULT_EXCLUDED_MODELS)),
        help="Comma-separated internal model ids to exclude.",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    excluded = {item.strip() for item in args.exclude_models.split(",") if item.strip()}
    paths = _latest_sample_paths(results_dir)
    assert paths, f"No sample JSONL files found under {results_dir}"
    model_to_variances = {
        _model_label(path): _load_model_variances(path)
        for path in paths
        if _model_label(path) not in excluded
    }
    assert model_to_variances, "No models left after exclusion filter"
    task_order = [
        "object_dimensions",
        "object_distance_to_camera",
        "object_pair_distance_center",
    ]
    title = "# Raw Consistency Average Variance\n\n"
    note = "Average per-question variance across video variations, using numeric answers extracted from raw model output.\n\n"
    if excluded:
        excluded_models = ", ".join(_display_model(model) for model in sorted(excluded))
        note += f"Excluded models: {excluded_models}\n\n"
    out_path.write_text(title + note + _build_table(model_to_variances, task_order), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
