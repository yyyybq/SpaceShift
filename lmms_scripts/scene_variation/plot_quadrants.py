"""
4-quadrant MRA% vs mean_CV plots from the combined per-model bundles.

Each panel shows model points as scatter on (MRA%, mean_CV) axes.
Reference lines: chance MRA (vertical, computed from JSONL ground truths) and a
configurable CV reference (horizontal, default 0.15). Y-axis is inverted so
"better consistency" sits toward the top.

Usage:
    python scripts/scene_variation/plot_quadrants.py
    python scripts/scene_variation/plot_quadrants.py --cv-ref 0.1 --pdf

Input:
    /nas2/edwin/lmms-eval/results/combined/<model>.json bundles produced by
    combine_results.py.

Output (under {out_dir}, default = combined/quadrants/):
    quadrants_modality_qtype.png   2x3 grid: rows=image,video; cols=qtypes
    quadrants_variation_pattern.png  multi-panel grid for each variation pattern
    quadrants_overall.png   single panel averaging all available data per model
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COMBINED_DIR = Path("/nas2/edwin/lmms-eval/results/combined")
DEFAULT_OUT_DIR = DEFAULT_COMBINED_DIR / "quadrants"
JSONL_PATHS = {
    "scene_variation": Path("/nas2/edwin/lmms-eval/data/scene_variation_042226_aligned.jsonl"),
    "image_consistency_thor_eval_v2": Path("/nas2/edwin/lmms-eval/data/image_consistency_thor_eval_v2.jsonl"),
    "video_consistency_thor_eval_v2": Path("/nas2/edwin/lmms-eval/data/video_consistency_thor_eval_v2.jsonl"),
}

QTYPE_LABEL = {
    "object_dimensions": "Obj. Size",
    "object_distance_to_camera": "Cam. Dist.",
    "object_pair_distance_center": "Pair Dist.",
}
QTYPE_ORDER = ("object_dimensions", "object_distance_to_camera", "object_pair_distance_center")
PATTERN_ORDER = ("rotation", "passby", "translate", "around", "approach", "spherical", "static", "remove")
PATTERN_LABEL = {
    "rotation": "Rotation", "passby": "Pass-by", "translate": "Translate",
    "around": "Around", "approach": "Approach", "spherical": "Spherical",
    "static": "Static", "remove": "Remove",
}
MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "p")
COLORS = plt.cm.tab10.colors + plt.cm.Set2.colors

MRA_START = 0.5
MRA_END = 0.95
MRA_INTERVAL = 0.05

_IMG_V2_GROUP_RE = re.compile(r"thor_(?P<motion>\w+?)_object_(?:pair_distance_center|distance_to_camera|dimensions)_")


def _to_float(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _modality(task: str) -> str:
    return "video" if task == "video_consistency_thor_eval_v2" else "image"


def _variation_pattern(task: str, score: dict) -> str | None:
    if task == "scene_variation":
        return score.get("edit_type")
    if task == "video_consistency_thor_eval_v2":
        return score.get("motion_family")
    if task == "image_consistency_thor_eval_v2":
        gid = score.get("group_id") or ""
        m = _IMG_V2_GROUP_RE.match(gid + "_")
        return m.group("motion") if m else None
    return None


def _vectorized_mra(pred: float, targets: np.ndarray) -> float:
    num_pts = int((MRA_END - MRA_START) / MRA_INTERVAL + 2)
    thresholds = np.linspace(MRA_START, MRA_END, num_pts)
    rel = np.where(targets != 0, np.abs(pred - targets) / np.abs(targets),
                   np.where(pred == 0, 0.0, np.inf))
    return float((rel[:, None] <= (1 - thresholds[None, :])).mean())


def _chance_mra_pct(targets: list[float]) -> float | None:
    if not targets:
        return None
    arr = np.asarray(targets, dtype=float)
    candidates = list(np.unique(arr))
    extra = set()
    for t in np.unique(arr):
        for ci in np.arange(MRA_START, MRA_END + 0.01, MRA_INTERVAL):
            if t != 0:
                extra.add(t * ci)
                extra.add(t * (2 - ci))
    candidates = list(np.unique(candidates + list(extra)))
    best = -np.inf
    for pred in candidates:
        m = _vectorized_mra(pred, arr)
        if m > best:
            best = m
    return best * 100.0


def _load_targets_by_slice() -> tuple[dict[tuple[str, str], list[float]], dict[str, list[float]]]:
    """Return ({(modality, qtype): [gts]}, {variation_pattern: [gts]})."""
    by_modality: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_variation: dict[str, list[float]] = defaultdict(list)
    for task, jsonl in JSONL_PATHS.items():
        if not jsonl.exists():
            continue
        with jsonl.open() as f:
            for line in f:
                obj = json.loads(line)
                qt = obj.get("question_type")
                gt = _to_float(obj.get("ground_truth"))
                if gt is None or qt not in QTYPE_ORDER:
                    continue
                by_modality[(_modality(task), qt)].append(gt)
                pat = _variation_pattern(task, obj)
                if pat:
                    by_variation[pat].append(gt)
    return by_modality, by_variation


def _group_cv(preds: list[float]) -> float | None:
    arr = np.asarray(preds, dtype=float)
    if len(arr) < 2:
        return None
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    if mu == 0:
        return 0.0 if sigma == 0 or math.isnan(sigma) else None
    if mu < 0:
        return None
    return sigma / mu


def _aggregate(samples: list[dict]) -> tuple[float | None, float | None]:
    """Return (MRA_pct, mean_CV) across all given samples."""
    if not samples:
        return None, None
    mras = [s.get("MRA") for s in samples if s.get("MRA") is not None]
    mra_pct = float(np.mean(mras)) * 100.0 if mras else None
    grouped: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        pred = _to_float(s.get("prediction_parse"))
        gid = s.get("group_id")
        if pred is None or not gid:
            continue
        grouped[gid].append(pred)
    cvs = [cv for cv in (_group_cv(v) for v in grouped.values()) if cv is not None]
    mean_cv = float(np.mean(cvs)) if cvs else None
    return mra_pct, mean_cv


def _model_color_marker(idx: int) -> tuple[tuple[float, float, float], str]:
    return COLORS[idx % len(COLORS)], MARKERS[idx % len(MARKERS)]


def _quadrant_panel(ax, points: list[tuple[float, float, str]], chance_mra: float | None, cv_ref: float, *, title: str) -> None:
    if not points:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return
    mras = [p[0] for p in points]
    cvs = [p[1] for p in points]
    x_lo, x_hi = 0.0, max(100.0, max(mras) + 5.0)
    y_lo, y_hi = 0.0, max(0.4, max(cvs) + 0.05)

    for i, (mra, cv, label) in enumerate(points):
        color, marker = _model_color_marker(i)
        ax.scatter([mra], [cv], s=70, color=color, marker=marker, edgecolors="black",
                   linewidths=0.6, zorder=3, label=label, alpha=0.95)

    if chance_mra is not None:
        ax.axvline(chance_mra, color="tab:orange", linestyle="--", linewidth=1.0, zorder=1,
                   label=f"chance MRA ({chance_mra:.1f}%)")
    ax.axhline(cv_ref, color="tab:blue", linestyle="--", linewidth=1.0, zorder=1,
               label=f"CV ref ({cv_ref:.2f})")

    cx = chance_mra if chance_mra is not None else (x_lo + x_hi) / 2
    quadrant_text = [
        ((x_lo + cx) / 2, (cvs and y_lo + (cv_ref - y_lo) / 2) or cv_ref / 2, "low MRA\nhigh consistency"),
        ((cx + x_hi) / 2, (cvs and y_lo + (cv_ref - y_lo) / 2) or cv_ref / 2, "high MRA\nhigh consistency"),
        ((x_lo + cx) / 2, (cv_ref + y_hi) / 2, "low MRA\nlow consistency"),
        ((cx + x_hi) / 2, (cv_ref + y_hi) / 2, "high MRA\nlow consistency"),
    ]
    for tx, ty, label in quadrant_text:
        ax.text(tx, ty, label, ha="center", va="center", fontsize=6.5,
                color="0.55", style="italic", zorder=1)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.invert_yaxis()
    ax.set_xlabel("MRA% (higher = better →)", fontsize=8)
    ax.set_ylabel("mean CV (lower = better, ↑)", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.tick_params(labelsize=7)


def _add_legend_outside(fig, points: list[tuple[float, float, str]]) -> None:
    handles = []
    seen = set()
    for i, (_, _, label) in enumerate(points):
        if label in seen:
            continue
        seen.add(label)
        color, marker = _model_color_marker(i)
        handles.append(plt.Line2D([], [], color=color, marker=marker,
                                  linestyle="", markersize=8,
                                  markeredgecolor="black", markeredgewidth=0.5,
                                  label=label))
    handles.append(plt.Line2D([], [], color="tab:orange", linestyle="--", label="chance MRA"))
    handles.append(plt.Line2D([], [], color="tab:blue", linestyle="--", label="CV ref"))
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               fontsize=8, bbox_to_anchor=(0.5, -0.01))


def _bundle_samples_by_slice(bundle: dict, slice_fn) -> dict:
    out: dict = defaultdict(list)
    for sample in bundle["samples"]:
        score = sample.get("sceneshift_score") or {}
        task = sample.get("_task") or ""
        key = slice_fn(task, score)
        if key is None:
            continue
        out[key].append(score)
    return out


def _gather_all_points(bundles: list[dict], slice_fn) -> dict:
    """For each slice -> list of (MRA, CV, model)."""
    out: dict = defaultdict(list)
    for bundle in bundles:
        per_slice = _bundle_samples_by_slice(bundle, slice_fn)
        for key, samples in per_slice.items():
            mra, cv = _aggregate(samples)
            if mra is None or cv is None:
                continue
            out[key].append((mra, cv, bundle["model"]))
    return out


def _plot_modality_qtype(bundles: list[dict], chance_by: dict, cv_ref: float, out_path: Path, pdf: bool) -> None:
    def slice_fn(task, score):
        qt = score.get("question_type")
        if qt not in QTYPE_ORDER:
            return None
        return (_modality(task), qt)

    points_by_slice = _gather_all_points(bundles, slice_fn)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for r, mod in enumerate(("image", "video")):
        for c, qt in enumerate(QTYPE_ORDER):
            pts = sorted(points_by_slice.get((mod, qt), []), key=lambda p: p[2])
            chance = chance_by.get((mod, qt))
            _quadrant_panel(axes[r][c], pts, chance, cv_ref,
                            title=f"{mod.title()} · {QTYPE_LABEL[qt]} (n={len(pts)} models)")
    all_points = [p for pts in points_by_slice.values() for p in pts]
    _add_legend_outside(fig, sorted(all_points, key=lambda p: p[2]))
    fig.suptitle("MRA × Consistency by Modality × Question Type", fontsize=14)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_variation_pattern(bundles: list[dict], chance_by: dict, cv_ref: float, out_path: Path, pdf: bool) -> None:
    def slice_fn(task, score):
        return _variation_pattern(task, score)

    points_by_slice = _gather_all_points(bundles, slice_fn)
    patterns_present = [p for p in PATTERN_ORDER if p in points_by_slice]
    n = len(patterns_present)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.5), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        if idx >= n:
            ax.axis("off")
            continue
        pat = patterns_present[idx]
        pts = sorted(points_by_slice.get(pat, []), key=lambda p: p[2])
        chance = chance_by.get(pat)
        _quadrant_panel(ax, pts, chance, cv_ref,
                        title=f"{PATTERN_LABEL.get(pat, pat)} (n={len(pts)} models)")
    all_points = [p for pts in points_by_slice.values() for p in pts]
    _add_legend_outside(fig, sorted(all_points, key=lambda p: p[2]))
    fig.suptitle("MRA × Consistency by Variation Pattern", fontsize=14)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_overall(bundles: list[dict], chance_overall: float | None, cv_ref: float, out_path: Path, pdf: bool) -> None:
    points = []
    for bundle in bundles:
        scores = [s.get("sceneshift_score") or {} for s in bundle["samples"]]
        mra, cv = _aggregate(scores)
        if mra is None or cv is None:
            continue
        points.append((mra, cv, bundle["model"]))
    points = sorted(points, key=lambda p: p[2])
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    _quadrant_panel(ax, points, chance_overall, cv_ref, title="Overall (all tasks combined)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
    fig.suptitle("MRA × Consistency – Overall", fontsize=13)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-dir", default=str(DEFAULT_COMBINED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cv-ref", type=float, default=0.15)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    combined_dir = Path(args.combined_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = []
    for jp in sorted(combined_dir.glob("*.json")):
        with jp.open() as f:
            bundles.append(json.load(f))
    assert bundles, f"No bundles found in {combined_dir}"

    print(f"Loading chance MRA per slice from {len(JSONL_PATHS)} JSONLs ...")
    targets_modality, targets_variation = _load_targets_by_slice()
    chance_modality = {k: _chance_mra_pct(v) for k, v in targets_modality.items()}
    chance_variation = {k: _chance_mra_pct(v) for k, v in targets_variation.items()}
    all_targets = [g for v in targets_modality.values() for g in v]
    chance_overall = _chance_mra_pct(all_targets)
    print(f"Overall chance MRA = {chance_overall:.2f}%")

    out_modality = out_dir / "quadrants_modality_qtype.png"
    out_variation = out_dir / "quadrants_variation_pattern.png"
    out_overall = out_dir / "quadrants_overall.png"

    _plot_modality_qtype(bundles, chance_modality, args.cv_ref, out_modality, args.pdf)
    print(f"wrote {out_modality}")
    _plot_variation_pattern(bundles, chance_variation, args.cv_ref, out_variation, args.pdf)
    print(f"wrote {out_variation}")
    _plot_overall(bundles, chance_overall, args.cv_ref, out_overall, args.pdf)
    print(f"wrote {out_overall}")


if __name__ == "__main__":
    main()
