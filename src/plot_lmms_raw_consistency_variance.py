"""Plot raw-answer consistency variance from lmms-eval sample JSONL files.

Usage:
  PYTHONPATH=src python src/plot_lmms_raw_consistency_variance.py \
    --results_dir /nas2/edwin/lmms-eval/results \
    --out_dir /nas2/edwin/lmms-eval/results/consistency_variance_plots

Input spec:
  `--results_dir` must contain model result trees with files named
  `*_samples_video_consistency_production.jsonl`.
  Each JSONL row should include `filtered_resps` or `sceneshift_score.prediction`,
  plus `group_id`, `question`, and `question_type`.

Output spec:
  Writes:
    `raw_consistency_variance_overview.png` / `.pdf`
    `raw_consistency_variance_summary.json`
  The figure contains mean / median / count heatmaps plus three task-wise
  symlog box-and-point plots across models. The JSON summary stores the
  aggregated statistics used.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUESTION_TYPE_DISPLAY = {
    "object_dimensions": "Object size",
    "object_distance_to_camera": "Camera-object distance",
    "object_pair_distance_center": "Object-object distance",
}

MODEL_DISPLAY = {
    "cambrians_1p5b": "Cambrian 1.5B",
    "cambrians_3b": "Cambrian 3B",
    "cambrians_7b": "Cambrian 7B",
    "llava_onevision_0p5b": "LLaVA-OV 0.5B",
    "llava_onevision_7b": "LLaVA-OV 7B",
    "qwen3vl_2b": "Qwen3-VL 2B",
    "qwen3vl_4b": "Qwen3-VL 4B",
    "qwen3vl_8b": "Qwen3-VL 8B",
}

DEFAULT_EXCLUDED_MODELS = {"llava_onevision_0p5b"}

ANSWER_PATTERN = re.compile(r"answer\s*['\"]?\s*:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _paper_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _extract_raw_text(row: dict) -> str:
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return str(filtered[0])
    score = row.get("sceneshift_score", {})
    prediction = score.get("prediction")
    if prediction is not None:
        return str(prediction)
    return ""


def _extract_numeric_answer(text: str) -> float | None:
    if not text:
        return None
    match = ANSWER_PATTERN.search(text)
    if match is not None:
        return round(float(match.group(1)), 1)
    numbers = NUMBER_PATTERN.findall(text)
    if not numbers:
        return None
    return round(float(numbers[-1]), 1)


def _question_key(score: dict) -> tuple[str, str]:
    group_id = str(score.get("group_id", ""))
    question = str(score.get("question", ""))
    return group_id, question


def _model_label(path: Path) -> str:
    stem = path.parts[-3]
    return stem.removesuffix("_video_consistency_production")


def _display_model(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def _latest_sample_paths(results_dir: Path) -> list[Path]:
    latest: dict[str, Path] = {}
    for path in sorted(results_dir.glob("*/*/*_samples_video_consistency_production.jsonl")):
        latest[_model_label(path)] = path
    return [latest[key] for key in sorted(latest)]


def _load_model_variances(path: Path) -> dict[str, list[float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        score = row["sceneshift_score"]
        qtype = str(score.get("question_type", "unknown"))
        group_id, question = _question_key(score)
        raw_text = _extract_raw_text(row)
        value = _extract_numeric_answer(raw_text)
        if value is None:
            continue
        grouped[(qtype, group_id, question)].append(value)
    by_type: dict[str, list[float]] = defaultdict(list)
    for (qtype, _, _), values in grouped.items():
        if len(values) < 2:
            continue
        by_type[qtype].append(statistics.pvariance(values))
    return dict(by_type)


def _summary_payload(model_to_variances: dict[str, dict[str, list[float]]], task_order: list[str]) -> dict:
    payload: dict[str, dict] = {"models": {}, "task_order": task_order}
    for model, by_type in model_to_variances.items():
        model_entry: dict[str, dict] = {}
        for qtype in task_order:
            vals = by_type.get(qtype, [])
            if vals:
                sorted_vals = sorted(vals)
                q90_idx = min(len(sorted_vals) - 1, math.ceil(0.9 * len(sorted_vals)) - 1)
                model_entry[qtype] = {
                    "question_count": len(vals),
                    "mean_variance": statistics.mean(vals),
                    "median_variance": statistics.median(vals),
                    "max_variance": max(vals),
                    "p90_variance": sorted_vals[q90_idx],
                }
            else:
                model_entry[qtype] = {
                    "question_count": 0,
                    "mean_variance": None,
                    "median_variance": None,
                    "max_variance": None,
                    "p90_variance": None,
                }
        payload["models"][model] = model_entry
    return payload


def _plot_heatmap(
    ax,
    models: list[str],
    task_order: list[str],
    summary: dict,
    metric_key: str,
    title: str,
    colorbar_label: str,
    fmt: str,
    cmap: str,
) -> None:
    matrix = []
    for model in models:
        row = []
        for qtype in task_order:
            value = summary["models"][model][qtype][metric_key]
            row.append(0.0 if value is None else value)
        matrix.append(row)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(task_order)), [QUESTION_TYPE_DISPLAY.get(q, q) for q in task_order], rotation=15, ha="right")
    ax.set_yticks(range(len(models)), [_display_model(model) for model in models])
    ax.set_title(title)
    ax.set_xlabel("Task")
    ax.set_ylabel("Model")
    for i, model in enumerate(models):
        for j, qtype in enumerate(task_order):
            value = summary["models"][model][qtype][metric_key]
            text = "NA" if value is None else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=7)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


def _point_offsets(count: int, width: float = 0.22) -> list[float]:
    if count <= 1:
        return [0.0]
    return [(-width / 2.0) + width * idx / (count - 1) for idx in range(count)]


def _plot_task_distribution(ax, models: list[str], model_to_variances: dict[str, dict[str, list[float]]], qtype: str, title: str) -> None:
    values = [model_to_variances[model].get(qtype, []) for model in models]
    positions = list(range(1, len(models) + 1))
    box = ax.boxplot(values, positions=positions, widths=0.6, showfliers=False, patch_artist=True)
    for patch, color in zip(box["boxes"], plt.cm.tab20.colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for pos, vals in zip(positions, values):
        offsets = _point_offsets(len(vals))
        xs = [pos + off for off in offsets]
        ax.scatter(xs, vals, color="#1b1b1b", alpha=0.45, s=12, zorder=3)
    means = [statistics.mean(v) if v else math.nan for v in values]
    medians = [statistics.median(v) if v else math.nan for v in values]
    ax.scatter(positions, means, color="#1b1b1b", marker="D", s=22, label="Mean", zorder=4)
    ax.scatter(positions, medians, color="#b2182b", marker="o", s=24, label="Median", zorder=4)
    ax.set_xticks(positions, [_display_model(model) for model in models], rotation=28, ha="right")
    ax.set_ylabel("Per-question variance")
    ax.set_title(title)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right")
    for pos, vals in zip(positions, values):
        ymax = max(vals) if vals else 0.0
        ax.text(pos, ymax * 1.15 + 1e-4, f"n={len(vals)}", ha="center", va="bottom", fontsize=7)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--exclude_models",
        default=",".join(sorted(DEFAULT_EXCLUDED_MODELS)),
        help="Comma-separated internal model ids to exclude.",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _latest_sample_paths(results_dir)
    assert paths, f"No sample JSONL files found under {results_dir}"
    excluded = {item.strip() for item in args.exclude_models.split(",") if item.strip()}
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
    summary = _summary_payload(model_to_variances, task_order)
    models = sorted(model_to_variances, key=lambda model: (
        -sum(
            summary["models"][model][qtype]["mean_variance"] or 0.0
            for qtype in task_order
        ),
        model,
    ))
    _paper_rc()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
    excluded_text = ", ".join(_display_model(model) for model in sorted(excluded)) or "none"
    fig.suptitle(
        f"Raw-answer consistency variance across video variations\nExcluded models: {excluded_text}",
        fontsize=12,
        y=1.03,
    )
    _plot_heatmap(
        axes[0, 0],
        models,
        task_order,
        summary,
        "mean_variance",
        "(a) Mean variance",
        "Mean variance",
        ".2f",
        "magma_r",
    )
    _plot_heatmap(
        axes[0, 1],
        models,
        task_order,
        summary,
        "median_variance",
        "(b) Median variance",
        "Median variance",
        ".3f",
        "viridis_r",
    )
    _plot_heatmap(
        axes[0, 2],
        models,
        task_order,
        summary,
        "question_count",
        "(c) Usable question count",
        "Question count",
        ".0f",
        "Blues",
    )
    _plot_task_distribution(axes[1, 0], models, model_to_variances, "object_dimensions", "(d) Object size")
    _plot_task_distribution(axes[1, 1], models, model_to_variances, "object_distance_to_camera", "(e) Camera-object distance")
    _plot_task_distribution(axes[1, 2], models, model_to_variances, "object_pair_distance_center", "(f) Object-object distance")
    plt.tight_layout()
    stem = out_dir / "raw_consistency_variance_overview"
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    summary_path = out_dir / "raw_consistency_variance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
