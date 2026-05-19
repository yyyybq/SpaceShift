"""Aggregate CV/MRA analysis for SceneShift result folders.

This script scans ``results/**/*_results.json`` and builds:

- overall metrics CSVs
- per-run group-level MRA/CV correlation CSV
- quadrant plots for full-model comparisons and the Qwen3-VL-2B finetune trajectory
- question-type tradeoff plot
- a markdown summary with key findings

Usage::

  PYTHONPATH=. python scripts/video_consistency/analyze_results_cv_mra.py \
    --results-root results \
    --output-dir results/cv_mra_analysis_20260414
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mra_cv_sample_stats import load_group_table
from video_consistency_chance_baseline import weighted_chance_mra_percent_cached

TASK_INFO = {
    "video_consistency_thor_eval_v2": {
        "title": "Video benchmark",
        "short": "video",
        "data_jsonl": "video_consistency_thor_eval_v2.jsonl",
    },
    "view_variation_image": {
        "title": "Image benchmark",
        "short": "image",
        "data_jsonl": "view_variation_image.jsonl",
    },
}

FULL_MODEL_LABELS = {
    "cambrians_7b_full": "Cambrian-S-7B",
    "qwen2_5_vl_7b_full": "Qwen2.5-VL-7B",
    "qwen3vl_2b_full": "Qwen3-VL-2B",
    "qwen3vl_4b_full": "Qwen3-VL-4B",
    "qwen3vl_8b_full": "Qwen3-VL-8B",
}

FULL_MODEL_ORDER = [
    "Qwen2.5-VL-7B",
    "Cambrian-S-7B",
    "Qwen3-VL-2B",
    "Qwen3-VL-4B",
    "Qwen3-VL-8B",
]

TRAJECTORY_ORDER = [
    "base",
    "ckpt-60",
    "ckpt-70",
    "ckpt-80",
    "ckpt-100",
    "ckpt-108",
    "final",
]

QTYPE_PRETTY = {
    "object_dimensions": "object size",
    "object_distance_to_camera": "camera distance",
    "object_pair_distance_center": "pair distance",
}

TASK_PRETTY = {task: info["title"] for task, info in TASK_INFO.items()}
TASK_SHORT = {task: info["short"] for task, info in TASK_INFO.items()}


@dataclass(frozen=True)
class RunInfo:
    result_path: Path
    top_dir: str
    full_label: str
    trajectory_label: str | None


def _repo_root() -> Path:
    return _REPO_ROOT


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Helvetica"],
        "font.size": 11,
        "axes.facecolor": "#f7f3ea",
        "figure.facecolor": "#f7f3ea",
        "axes.edgecolor": "#272727",
        "grid.color": "#d7d0c4",
        "grid.alpha": 0.6,
        "grid.linestyle": (0, (1, 2)),
        "figure.dpi": 120,
        "savefig.dpi": 220,
    })


def _discover_results(results_root: Path) -> list[Path]:
    return sorted({p.resolve() for p in results_root.rglob("*_results.json") if p.is_file()})


def _top_dir(results_root: Path, path: Path) -> str:
    return path.relative_to(results_root).parts[0]


def _trajectory_label(path: Path) -> str | None:
    raw = str(path)
    m = re.search(r"/checkpoint-(\d+)/", raw)
    if m:
        return f"ckpt-{m.group(1)}"
    if "/final/" in raw:
        return "final"
    return None


def _full_model_label(results_root: Path, path: Path) -> str:
    top_dir = _top_dir(results_root, path)
    if top_dir in FULL_MODEL_LABELS:
        return FULL_MODEL_LABELS[top_dir]
    if top_dir == "qwen3vl_2b_video_interiorgs_trained_ckpts":
        traj = _trajectory_label(path)
        if traj is not None:
            return f"Qwen3-VL-2B ft {traj}"
    return top_dir


def _run_info(results_root: Path, path: Path) -> RunInfo:
    top_dir = _top_dir(results_root, path)
    traj_label = _trajectory_label(path)
    return RunInfo(
        result_path=path,
        top_dir=top_dir,
        full_label=_full_model_label(results_root, path),
        trajectory_label=traj_label,
    )


def _flat_metrics_for_task(payload: dict, task_name: str) -> dict | None:
    task_block = payload.get("results", {}).get(task_name)
    if not task_block:
        return None
    for value in task_block.values():
        if isinstance(value, dict) and value.get("overall_MRA") is not None:
            return value
    return None


def _sample_jsonl_path(result_path: Path, task_name: str) -> Path:
    stem = result_path.name[: -len("_results.json")]
    return result_path.with_name(f"{stem}_samples_{task_name}.jsonl")


def _load_overall_rows(results_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for result_path in _discover_results(results_root):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        info = _run_info(results_root, result_path)
        for task_name, task_meta in TASK_INFO.items():
            flat = _flat_metrics_for_task(payload, task_name)
            if flat is None:
                continue
            sample_path = _sample_jsonl_path(result_path, task_name)
            rows.append({
                "result_path": str(result_path),
                "top_dir": info.top_dir,
                "model_label": info.full_label,
                "trajectory_label": info.trajectory_label,
                "task": task_name,
                "task_pretty": task_meta["title"],
                "task_short": task_meta["short"],
                "overall_MRA": float(flat["overall_MRA"]),
                "overall_mean_CV": float(flat["overall_mean_CV"]),
                "overall_n": int(flat.get("overall_n", 0)),
                "overall_n_groups": int(flat.get("overall_n_groups", 0)),
                "samples_path": str(sample_path) if sample_path.exists() else "",
            })
    df = pd.DataFrame(rows)
    assert not df.empty, f"no matching result rows under {results_root}"
    return df


def _complete_run_paths(overall_df: pd.DataFrame) -> set[str]:
    counts = overall_df.groupby("result_path")["task"].nunique()
    return set(counts[counts == len(TASK_INFO)].index.tolist())


def _extract_qtype_rows(result_path: Path, payload: dict, model_label: str) -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(r"^qtype:(.+)_(MRA|mean_CV)$")
    for task_name in TASK_INFO:
        flat = _flat_metrics_for_task(payload, task_name)
        if flat is None:
            continue
        by_qtype: dict[str, dict[str, float]] = defaultdict(dict)
        for key, value in flat.items():
            match = pattern.match(key)
            if not match:
                continue
            qtype, field = match.groups()
            by_qtype[qtype][field] = float(value)
        for qtype, metrics in sorted(by_qtype.items()):
            rows.append({
                "result_path": str(result_path),
                "task": task_name,
                "task_pretty": TASK_PRETTY[task_name],
                "model_label": model_label,
                "question_type": qtype,
                "question_type_pretty": QTYPE_PRETTY.get(qtype, qtype.replace("_", " ")),
                "MRA": float(metrics["MRA"]),
                "mean_CV": float(metrics["mean_CV"]),
            })
    return rows


def _load_qtype_rows(results_root: Path, allowed_paths: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for result_path in _discover_results(results_root):
        if str(result_path) not in allowed_paths:
            continue
        top_dir = _top_dir(results_root, result_path)
        if top_dir not in FULL_MODEL_LABELS:
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows.extend(_extract_qtype_rows(result_path, payload, FULL_MODEL_LABELS[top_dir]))
    df = pd.DataFrame(rows)
    assert not df.empty, "no qtype rows found for complete full-model runs"
    return df


def _load_group_correlations(overall_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in overall_df.itertuples(index=False):
        sample_path = Path(row.samples_path) if row.samples_path else None
        if sample_path is None or not sample_path.is_file():
            continue
        groups = load_group_table(sample_path)
        pearson = stats.pearsonr(groups["mean_mra_pct"], groups["cv"])
        spearman = stats.spearmanr(groups["mean_mra_pct"], groups["cv"])
        traj_label = row.trajectory_label
        analysis_label = traj_label if traj_label else row.model_label
        rows.append({
            "result_path": row.result_path,
            "model_label": row.model_label,
            "analysis_label": analysis_label,
            "trajectory_label": traj_label or "",
            "task": row.task,
            "task_pretty": row.task_pretty,
            "task_short": row.task_short,
            "n_groups": int(len(groups)),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        })
    df = pd.DataFrame(rows)
    assert not df.empty, "no group correlations could be computed"
    return df


def _pareto_frontier(df: pd.DataFrame) -> pd.Series:
    flags: list[bool] = []
    for row in df.itertuples(index=False):
        dominates = (
            (df["overall_MRA"] >= row.overall_MRA)
            & (df["overall_mean_CV"] <= row.overall_mean_CV)
            & (
                (df["overall_MRA"] > row.overall_MRA)
                | (df["overall_mean_CV"] < row.overall_mean_CV)
            )
        )
        flags.append(not bool(dominates.any()))
    return pd.Series(flags, index=df.index)


def _ordered_labels(labels: list[str], preferred: list[str]) -> list[str]:
    pref_rank = {label: i for i, label in enumerate(preferred)}
    return sorted(labels, key=lambda label: (pref_rank.get(label, 999), label))


def _palette(labels: list[str]) -> dict[str, str]:
    base = [
        "#c45c3e",
        "#6d8b59",
        "#5b4b7d",
        "#2d6a6a",
        "#b38728",
        "#5f7ea8",
        "#a05f8c",
        "#6f6f6f",
    ]
    return {label: base[i % len(base)] for i, label in enumerate(labels)}


def _annotate_points(ax: plt.Axes, data: pd.DataFrame, label_col: str) -> None:
    seen: dict[tuple[float, float], int] = defaultdict(int)
    offsets = [(6, 6), (6, -10), (-44, 6), (-44, -10), (10, 14), (10, -18)]
    for row in data.itertuples(index=False):
        key = (round(row.overall_MRA, 4), round(row.overall_mean_CV, 4))
        dup_idx = seen[key]
        seen[key] += 1
        dx, dy = offsets[dup_idx % len(offsets)]
        ax.annotate(
            getattr(row, label_col),
            (row.overall_MRA, row.overall_mean_CV),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color="#1f1f1f",
        )


def _quadrant_text(ax: plt.Axes, x_mid: float, y_mid: float, x_lim: tuple[float, float], y_lim: tuple[float, float]) -> None:
    x0, x1 = x_lim
    y_top, y_bottom = y_lim
    boxes = [
        ((x0 + x_mid) / 2.0, y_top + 0.18 * (y_bottom - y_top), "low MRA\nstable"),
        ((x_mid + x1) / 2.0, y_top + 0.18 * (y_bottom - y_top), "high MRA\nstable"),
        ((x0 + x_mid) / 2.0, y_top + 0.80 * (y_bottom - y_top), "low MRA\nvolatile"),
        ((x_mid + x1) / 2.0, y_top + 0.80 * (y_bottom - y_top), "high MRA\nvolatile"),
    ]
    for x, y, label in boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#555",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fffdfa", "edgecolor": "#ddd6c9", "alpha": 0.9},
        )


def _plot_quadrant(
    data: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    label_col: str,
    order: list[str],
    chance_mra: float | None,
    connect_path: bool = False,
) -> None:
    data = data.copy()
    assert not data.empty, "quadrant plot received empty frame"
    labels = _ordered_labels(data[label_col].tolist(), order)
    pal = _palette(labels)
    data["pareto"] = _pareto_frontier(data)

    x_pad = 2.0
    x_min = max(0.0, float(data["overall_MRA"].min()) - x_pad)
    x_max = float(data["overall_MRA"].max()) + x_pad
    if chance_mra is not None:
        x_max = max(x_max, chance_mra + x_pad)
    y_pad = 0.015
    y_min = max(0.0, float(data["overall_mean_CV"].min()) - y_pad)
    y_max = float(data["overall_mean_CV"].max()) + y_pad

    x_mid = float(data["overall_MRA"].median())
    y_mid = float(data["overall_mean_CV"].median())

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.grid(True)

    ax.axvline(x_mid, color="#333", linestyle="--", linewidth=1.4)
    ax.axhline(y_mid, color="#333", linestyle="--", linewidth=1.4)
    if chance_mra is not None:
        ax.axvline(chance_mra, color="#8d8d8d", linestyle=":", linewidth=1.3)

    if connect_path and len(data) > 1:
        ordered = data.assign(_rank=data[label_col].map({label: i for i, label in enumerate(order)})).sort_values("_rank")
        ax.plot(
            ordered["overall_MRA"],
            ordered["overall_mean_CV"],
            color="#767676",
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
        )

    for row in data.itertuples(index=False):
        size = 96 if row.pareto else 70
        ax.scatter(
            [row.overall_MRA],
            [row.overall_mean_CV],
            s=size,
            c=[pal[getattr(row, label_col)]],
            edgecolors="#151515",
            linewidths=1.4 if row.pareto else 0.8,
            alpha=0.95,
            zorder=3,
        )

    _annotate_points(ax, data, label_col)
    _quadrant_text(ax, x_mid, y_mid, (x_min, x_max), (y_min, y_max))

    ax.set_xlabel("Overall MRA (%)")
    ax.set_ylabel("Overall mean CV (lower is better; top is better)")
    ax.set_title(title)
    foot = (
        f"Crosshairs use model-set medians: MRA={x_mid:.2f}, CV={y_mid:.4f}. "
        f"Pareto points use thicker outlines."
    )
    if chance_mra is not None:
        foot += f" Dotted vertical line: best constant-baseline MRA={chance_mra:.2f}%."
    fig.text(0.01, 0.02, foot, fontsize=8, color="#555")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_group_corr_heatmap(corr_df: pd.DataFrame, out_path: Path) -> None:
    matrix = (
        corr_df.pivot(index="analysis_label", columns="task_short", values="spearman_rho")
        .reindex(columns=["video", "image"])
    )
    full_rows = [label for label in FULL_MODEL_ORDER if label in matrix.index]
    traj_rows = [label for label in TRAJECTORY_ORDER if label in matrix.index]
    row_order = full_rows + traj_rows
    matrix = matrix.reindex(row_order)

    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(matrix.index) + 2.0))
    im = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-0.35, vmax=0.35, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Within-run group-level Spearman rho: MRA vs CV")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.isna(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="#111")
    if full_rows and traj_rows:
        ax.axhline(len(full_rows) - 0.5, color="#333", linewidth=1.1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Spearman rho")
    fig.text(
        0.01,
        0.02,
        "Values near 0 mean per-group consistency does not cleanly predict per-group accuracy within a run.",
        fontsize=8,
        color="#555",
    )
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_qtype_tradeoff(qtype_df: pd.DataFrame, out_path: Path) -> None:
    agg = (
        qtype_df.groupby(["task", "question_type_pretty"], as_index=False)
        .agg(avg_MRA=("MRA", "mean"), avg_CV=("mean_CV", "mean"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=False)
    for ax, task_name in zip(axes, TASK_INFO):
        sub = agg[agg["task"] == task_name].copy().sort_values("avg_MRA")
        ax.grid(True)
        ax.set_xlim(max(0.0, sub["avg_MRA"].min() - 2.0), sub["avg_MRA"].max() + 2.0)
        y_min = max(0.0, sub["avg_CV"].min() - 0.02)
        y_max = sub["avg_CV"].max() + 0.02
        ax.set_ylim(y_max, y_min)
        for row in sub.itertuples(index=False):
            ax.scatter([row.avg_MRA], [row.avg_CV], s=95, c="#2d6a6a", edgecolors="#111", linewidths=0.8)
            ax.annotate(row.question_type_pretty, (row.avg_MRA, row.avg_CV), xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.set_title(TASK_PRETTY[task_name])
        ax.set_xlabel("Average MRA across full models (%)")
        ax.set_ylabel("Average mean CV (top is better)")
    fig.suptitle("Question-type tradeoff across full-model runs")
    fig.text(
        0.01,
        0.02,
        "This highlights which question types are structurally easier or harder across architectures.",
        fontsize=8,
        color="#555",
    )
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _incomplete_top_dirs(results_root: Path, discovered_paths: list[Path], exclude: set[str] | None = None) -> tuple[list[str], list[str]]:
    result_dirs = {_top_dir(results_root, path) for path in discovered_paths}
    excluded = exclude or set()
    missing: list[str] = []
    cache_only: list[str] = []
    for child in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if child.name in excluded:
            continue
        if child.name in result_dirs:
            continue
        if any(child.rglob("*_response.json")):
            cache_only.append(child.name)
        else:
            missing.append(child.name)
    return missing, cache_only


def _format_frontier(df: pd.DataFrame, label_col: str) -> str:
    marked = df.loc[_pareto_frontier(df), label_col].tolist()
    return ", ".join(marked)


def _best_row(df: pd.DataFrame, metric: str, ascending: bool) -> pd.Series:
    return df.sort_values(metric, ascending=ascending).iloc[0]


def _write_summary(
    out_path: Path,
    *,
    full_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    qtype_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    chance_by_task: dict[str, float],
    missing_dirs: list[str],
    cache_only_dirs: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# CV/MRA results analysis")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- full-model comparison set: `{full_df['model_label'].nunique()}` runs with both tasks")
    checkpoint_count = int(trajectory_df[trajectory_df["trajectory_label"] != "base"]["trajectory_label"].nunique())
    lines.append(f"- Qwen3-VL-2B trajectory set: `{checkpoint_count}` checkpoints plus base anchor")
    if missing_dirs:
        lines.append(f"- directories with no saved `_results.json`: `{', '.join(missing_dirs)}`")
    if cache_only_dirs:
        lines.append(f"- directories with cache-only artifacts: `{', '.join(cache_only_dirs)}`")
    lines.append("")

    lines.append("## Main findings")
    for task_name in TASK_INFO:
        task_full = full_df[full_df["task"] == task_name].copy()
        task_traj = trajectory_df[trajectory_df["task"] == task_name].copy()
        task_ckpt = task_traj[task_traj["trajectory_label"] != "base"].copy()
        best_mra = _best_row(task_full, "overall_MRA", ascending=False)
        best_cv = _best_row(task_full, "overall_mean_CV", ascending=True)
        best_traj = _best_row(task_ckpt, "overall_MRA", ascending=False)
        frontier = _format_frontier(task_full, "model_label")
        lines.append(f"### {TASK_PRETTY[task_name]}")
        lines.append(
            f"- best full-model MRA: `{best_mra['model_label']}` at `MRA={best_mra['overall_MRA']:.2f}`, "
            f"`CV={best_mra['overall_mean_CV']:.4f}`"
        )
        lines.append(
            f"- lowest full-model CV: `{best_cv['model_label']}` at `CV={best_cv['overall_mean_CV']:.4f}`, "
            f"`MRA={best_cv['overall_MRA']:.2f}`"
        )
        lines.append(f"- full-model Pareto frontier: `{frontier}`")
        lines.append(f"- best checkpoint by MRA: `{best_traj['trajectory_label']}` at `MRA={best_traj['overall_MRA']:.2f}`, `CV={best_traj['overall_mean_CV']:.4f}`")
        lines.append(
            f"- best constant-baseline reference MRA: `{chance_by_task[task_name]:.2f}%`"
        )
        lines.append("")

    base_rows = trajectory_df[trajectory_df["trajectory_label"] == "base"].set_index("task")
    ckpt70_rows = trajectory_df[trajectory_df["trajectory_label"] == "ckpt-70"].set_index("task")
    final_rows = trajectory_df[trajectory_df["trajectory_label"] == "final"].set_index("task")
    lines.append("## Qwen3-VL-2B trajectory")
    for task_name in TASK_INFO:
        base = base_rows.loc[task_name]
        ck70 = ckpt70_rows.loc[task_name]
        final = final_rows.loc[task_name]
        lines.append(
            f"- {TASK_PRETTY[task_name]} vs base: `ckpt-70` changes "
            f"`MRA {ck70['overall_MRA'] - base['overall_MRA']:+.2f}` and "
            f"`CV {ck70['overall_mean_CV'] - base['overall_mean_CV']:+.4f}`; "
            f"`final` changes `MRA {final['overall_MRA'] - base['overall_MRA']:+.2f}` and "
            f"`CV {final['overall_mean_CV'] - base['overall_mean_CV']:+.4f}`."
        )
    lines.append("")

    lines.append("## CV vs MRA relationship")
    corr_means = (
        corr_df.groupby("task_short", as_index=False)
        .agg(mean_spearman=("spearman_rho", "mean"), mean_pearson=("pearson_r", "mean"))
    )
    for row in corr_means.itertuples(index=False):
        lines.append(
            f"- average within-run correlation on `{row.task_short}`: "
            f"`Spearman={row.mean_spearman:+.3f}`, `Pearson={row.mean_pearson:+.3f}`"
        )
    lines.append(
        "- Interpretation: lower CV does not reliably imply higher per-group MRA inside a single run; "
        "consistency and accuracy move together only weakly here."
    )
    lines.append("")

    qtype_avg = (
        qtype_df.groupby(["task", "question_type_pretty"], as_index=False)
        .agg(avg_MRA=("MRA", "mean"), avg_CV=("mean_CV", "mean"))
    )
    lines.append("## Question-type pattern")
    for task_name in TASK_INFO:
        sub = qtype_avg[qtype_avg["task"] == task_name].copy()
        easiest = sub.sort_values(["avg_MRA", "avg_CV"], ascending=[False, True]).iloc[0]
        hardest = sub.sort_values(["avg_MRA", "avg_CV"], ascending=[True, False]).iloc[0]
        lines.append(
            f"- {TASK_PRETTY[task_name]} easiest slice: `{easiest['question_type_pretty']}` "
            f"(`MRA={easiest['avg_MRA']:.2f}`, `CV={easiest['avg_CV']:.4f}`)"
        )
        lines.append(
            f"- {TASK_PRETTY[task_name]} hardest slice: `{hardest['question_type_pretty']}` "
            f"(`MRA={hardest['avg_MRA']:.2f}`, `CV={hardest['avg_CV']:.4f}`)"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CV/MRA analysis across result folders.")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = _repo_root()
    results_root = (args.results_root if args.results_root is not None else root / "results").resolve()
    output_dir = (args.output_dir if args.output_dir is not None else results_root / "cv_mra_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _style()

    overall_df = _load_overall_rows(results_root)
    complete_paths = _complete_run_paths(overall_df)
    overall_df = overall_df[overall_df["result_path"].isin(complete_paths)].copy()

    full_df = overall_df[overall_df["top_dir"].isin(FULL_MODEL_LABELS)].copy()
    assert not full_df.empty, "no complete full-model runs found"

    base_anchor = full_df[full_df["model_label"] == "Qwen3-VL-2B"].copy()
    trajectory_only = overall_df[overall_df["top_dir"] == "qwen3vl_2b_video_interiorgs_trained_ckpts"].copy()
    trajectory_only["trajectory_label"] = trajectory_only["trajectory_label"].fillna("")
    base_anchor = base_anchor.assign(trajectory_label="base")
    trajectory_df = pd.concat([base_anchor, trajectory_only], ignore_index=True)

    qtype_df = _load_qtype_rows(results_root, complete_paths)
    corr_df = _load_group_correlations(overall_df)

    full_wide = (
        full_df.pivot(index="model_label", columns="task_short", values=["overall_MRA", "overall_mean_CV"])
        .sort_index()
    )

    qtype_avg = (
        qtype_df.groupby(["task", "task_pretty", "question_type", "question_type_pretty"], as_index=False)
        .agg(avg_MRA=("MRA", "mean"), avg_CV=("mean_CV", "mean"))
    )

    discovered = _discover_results(results_root)
    missing_dirs, cache_only_dirs = _incomplete_top_dirs(results_root, discovered, exclude={output_dir.name})

    chance_cache = output_dir / "chance_cache"
    chance_cache.mkdir(parents=True, exist_ok=True)
    chance_by_task: dict[str, float] = {}
    for task_name, meta in TASK_INFO.items():
        data_jsonl = root / "data" / meta["data_jsonl"]
        chance_by_task[task_name] = float(
            weighted_chance_mra_percent_cached(
                data_jsonl,
                chance_cache / f"{task_name}_chance_mra_cache.json",
            )
        )

    full_df.to_csv(output_dir / "overall_metrics_long.csv", index=False)
    full_wide.to_csv(output_dir / "overall_metrics_wide.csv")
    qtype_df.to_csv(output_dir / "full_model_qtype_metrics.csv", index=False)
    qtype_avg.to_csv(output_dir / "full_model_qtype_avg.csv", index=False)
    corr_df.to_csv(output_dir / "group_level_correlations.csv", index=False)
    trajectory_df.to_csv(output_dir / "qwen3vl_2b_trajectory_metrics.csv", index=False)

    for task_name in TASK_INFO:
        full_task = full_df[full_df["task"] == task_name].copy()
        _plot_quadrant(
            full_task,
            output_dir / f"01_full_models_{TASK_SHORT[task_name]}_quadrant.png",
            title=f"Full models: {TASK_PRETTY[task_name]}",
            label_col="model_label",
            order=FULL_MODEL_ORDER,
            chance_mra=chance_by_task[task_name],
            connect_path=False,
        )

        traj_task = trajectory_df[trajectory_df["task"] == task_name].copy()
        _plot_quadrant(
            traj_task,
            output_dir / f"02_qwen3vl_2b_trajectory_{TASK_SHORT[task_name]}_quadrant.png",
            title=f"Qwen3-VL-2B trajectory: {TASK_PRETTY[task_name]}",
            label_col="trajectory_label",
            order=TRAJECTORY_ORDER,
            chance_mra=chance_by_task[task_name],
            connect_path=True,
        )

    _plot_group_corr_heatmap(corr_df, output_dir / "03_group_level_spearman_heatmap.png")
    _plot_qtype_tradeoff(qtype_df, output_dir / "04_full_models_qtype_tradeoff.png")
    _write_summary(
        output_dir / "summary.md",
        full_df=full_df,
        trajectory_df=trajectory_df,
        qtype_df=qtype_df,
        corr_df=corr_df,
        chance_by_task=chance_by_task,
        missing_dirs=missing_dirs,
        cache_only_dirs=cache_only_dirs,
    )

    print(f"wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
