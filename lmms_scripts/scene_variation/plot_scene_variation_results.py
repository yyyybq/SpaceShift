"""
Plot scene_variation eval results across all models.

Usage:
    python scripts/scene_variation/plot_scene_variation_results.py
    python scripts/scene_variation/plot_scene_variation_results.py --metric CAPE

Input:
    Auto-scans /nas2/edwin/lmms-eval/results/<model>_scene_variation/<run>/*_results.json
    For each model takes the latest run by timestamp.

Output:
    /nas2/edwin/lmms-eval/results/scene_variation_plots/
        scene_variation_overall.png
        scene_variation_by_qtype.png
        scene_variation_by_etype.png
        scene_variation_heatmap_etype_x_qtype.png
        scene_variation_summary.csv
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_ROOT = Path("/nas2/edwin/lmms-eval/results")
OUTPUT_DIR = RESULTS_ROOT / "scene_variation_plots"
QUESTION_TYPES = ("object_size", "object_distance_to_camera", "object_pair_distance_center")
EDIT_TYPES = ("translate_single_obj", "rotate_single_obj", "remove_single_obj")
METRIC_DEFAULTS = {"MRA": (0, 60, "MRA (%)"), "CAPE": (0, 1.0, "CAPE (lower=better)")}


def _model_label(results_path: Path) -> str:
    """Use the parent dir of the *_results.json file (the model dir from --output_path)."""
    parent = results_path.parent.name
    grandparent = results_path.parent.parent.name
    label = grandparent.replace("_scene_variation", "").replace("_full_benchmark", "")
    if not label or label.startswith("results"):
        label = parent
    return label


def _load_results(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    sv = payload.get("results", {}).get("scene_variation")
    if not sv:
        return None
    return sv.get("sceneshift_score,none")


def _discover() -> dict[str, Path]:
    """Map model_label -> latest results.json that contains scene_variation."""
    latest_by_label: dict[str, tuple[float, Path]] = {}
    for path in RESULTS_ROOT.rglob("*_results.json"):
        if "scene_variation_plots" in path.parts:
            continue
        score = _load_results(path)
        if not score:
            continue
        label = _model_label(path)
        mtime = path.stat().st_mtime
        if label not in latest_by_label or mtime > latest_by_label[label][0]:
            latest_by_label[label] = (mtime, path)
    return {label: p for label, (_, p) in latest_by_label.items()}


def _collect(metric: str) -> pd.DataFrame:
    rows = []
    for label, path in _discover().items():
        score = _load_results(path)
        n = score.get("overall_n", 0)
        if not n:
            continue
        row = {"model": label, "n": n, "overall": score.get(f"overall_{metric}")}
        for qt in QUESTION_TYPES:
            row[f"q:{qt}"] = score.get(f"qtype:{qt}_{metric}")
        for et in EDIT_TYPES:
            row[f"e:{et}"] = score.get(f"etype:{et}_{metric}")
        for et in EDIT_TYPES:
            for qt in QUESTION_TYPES:
                row[f"ex:{et}|{qt}"] = score.get(f"etype_x_qtype:{et}_x_{qt}_{metric}")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("overall", ascending=False).reset_index(drop=True)
    return df


def _model_palette(models: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(models)}


def _plot_overall(df: pd.DataFrame, metric: str, out: Path) -> None:
    lo, hi, ylabel = METRIC_DEFAULTS[metric]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(df)), 5))
    palette = _model_palette(df["model"].tolist())
    colors = [palette[m] for m in df["model"]]
    bars = ax.bar(df["model"], df["overall"], color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, df["overall"]):
        if val is None or pd.isna(val):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, val + (hi - lo) * 0.01, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"scene_variation: overall {metric}")
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_grouped(df: pd.DataFrame, prefix: str, axes_labels: tuple[str, ...], metric: str, title: str, out: Path) -> None:
    lo, hi, ylabel = METRIC_DEFAULTS[metric]
    cols = [f"{prefix}{name}" for name in axes_labels]
    n_models = len(df)
    n_groups = len(cols)
    palette = _model_palette(df["model"].tolist())
    width = 0.8 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * n_groups + 0.6 * n_models), 5))
    x = np.arange(n_groups)
    for i, (_, row) in enumerate(df.iterrows()):
        offsets = x + (i - (n_models - 1) / 2) * width
        vals = [row[c] if pd.notna(row[c]) else 0 for c in cols]
        ax.bar(offsets, vals, width=width, color=palette[row["model"]], label=row["model"], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_single_obj", "").replace("object_", "") for l in axes_labels], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(lo, hi)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_heatmap(df: pd.DataFrame, metric: str, out: Path) -> None:
    cell_cols = [f"ex:{et}|{qt}" for et in EDIT_TYPES for qt in QUESTION_TYPES]
    matrix = df[cell_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(cell_cols)), 0.55 * len(df) + 2))
    cmap = "viridis" if metric == "MRA" else "viridis_r"
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(cell_cols)))
    ax.set_xticklabels(
        [f"{et.replace('_single_obj','')}\n{qt.replace('object_','')}" for et in EDIT_TYPES for qt in QUESTION_TYPES],
        rotation=0,
        fontsize=9,
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"], fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            text = f"{v:.1f}"
            color = "white" if (metric == "MRA" and v < 30) or (metric == "CAPE" and v > 0.5) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label(metric)
    ax.set_title(f"scene_variation: edit_type × question_type ({metric})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=("MRA", "CAPE"), default="MRA")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _collect(args.metric)
    assert not df.empty, "No scene_variation results found"

    csv_path = args.output_dir / f"scene_variation_summary_{args.metric}.csv"
    df.to_csv(csv_path, index=False, float_format="%.3f")

    _plot_overall(df, args.metric, args.output_dir / f"scene_variation_overall_{args.metric}.png")
    _plot_grouped(df, "q:", QUESTION_TYPES, args.metric, f"scene_variation: {args.metric} by question_type", args.output_dir / f"scene_variation_by_qtype_{args.metric}.png")
    _plot_grouped(df, "e:", EDIT_TYPES, args.metric, f"scene_variation: {args.metric} by edit_type", args.output_dir / f"scene_variation_by_etype_{args.metric}.png")
    _plot_heatmap(df, args.metric, args.output_dir / f"scene_variation_heatmap_etype_x_qtype_{args.metric}.png")

    print(f"Wrote {len(df)} models to {args.output_dir}")
    print(df[["model", "n", "overall"]].to_string(index=False))


if __name__ == "__main__":
    main()
