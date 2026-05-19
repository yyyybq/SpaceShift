"""Plot MRA vs CV quadrant chart with arrows showing movement across training stages.

Usage:
    PYTHONPATH=src python src/plot_training_stage_quadrants.py \\
        --base      "Base,38.5,0.12" \\
        --stage     "Single-Image,41.2,0.11" \\
        --stage     "Multi-Image,40.8,0.08" \\
        --stage     "Video,42.0,0.09" \\
        --mra_chance_pct 35.0 \\
        --out_png   figures/training_stages_quadrant.png

    Or with a JSON file:
    PYTHONPATH=src python src/plot_training_stage_quadrants.py \\
        --summary_json figures/training_stages.json \\
        --mra_chance_pct 35.0 \\
        --out_png figures/training_stages_quadrant.png

Input:
    --base           NAME,MRA,CV for the base (untrained) model.
    --stage          NAME,MRA,CV for each training stage (ordered).
    --summary_json   JSON file: {"base": {"name":..,"mra":..,"cv":..},
                     "stages": [{"name":..,"mra":..,"cv":..}, ...]}.
    --mra_chance_pct Chance-level MRA percentage (vertical center line).
    --cv_threshold   CV midpoint (horizontal center line, default 0.1).

Output:
    PNG (and optional PDF) scatter with:
      - Base model as a square marker.
      - Each training stage as a circle marker.
      - Arrows from base -> stage1 -> stage2 -> stage3 showing trajectory.
      - Quadrant labels (above/below chance, better/worse consistency).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


STAGE_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
BASE_COLOR = "#424242"


def _parse_point(raw: str) -> tuple[str, float, float]:
    parts = raw.split(",")
    assert len(parts) == 3, f"Expected NAME,MRA,CV but got: {raw}"
    return parts[0].strip(), float(parts[1]), float(parts[2])


def _paper_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _apply_centered_axes(
    ax: plt.Axes,
    mras: list[float],
    cvs: list[float],
    chance_pct: float,
    cv_ref: float,
) -> tuple[float, float]:
    """Center the axes around (chance_pct, cv_ref), return (y_lo, y_hi)."""
    margin_x = max(abs(m - chance_pct) for m in mras) * 1.4 + 2.0
    x_lo = chance_pct - margin_x
    x_hi = chance_pct + margin_x
    ax.set_xlim(x_lo, x_hi)

    margin_y = max(abs(c - cv_ref) for c in cvs) * 1.4 + 0.02
    y_lo = cv_ref - margin_y
    y_hi = cv_ref + margin_y
    ax.set_ylim(y_lo, y_hi)
    ax.invert_yaxis()

    ax.axvline(chance_pct, color="0.55", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(cv_ref, color="0.55", linewidth=0.8, linestyle="--", zorder=1)
    return y_lo, y_hi


def _add_quadrant_labels(
    ax: plt.Axes,
    chance_pct: float,
    cv_ref: float,
    y_lo: float,
    y_hi: float,
) -> None:
    x_lo, x_hi = ax.get_xlim()
    labels = [
        ((x_lo + chance_pct) / 2, (y_lo + cv_ref) / 2, "below chance\nbetter consistency"),
        ((chance_pct + x_hi) / 2, (y_lo + cv_ref) / 2, "above chance\nbetter consistency"),
        ((x_lo + chance_pct) / 2, (cv_ref + y_hi) / 2, "below chance\nworse consistency"),
        ((chance_pct + x_hi) / 2, (cv_ref + y_hi) / 2, "above chance\nworse consistency"),
    ]
    for x, y, text in labels:
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color="0.65", style="italic")


def plot_training_stages(
    base: tuple[str, float, float],
    stages: list[tuple[str, float, float]],
    chance_pct: float,
    cv_ref: float,
    out_png: Path,
    out_pdf: Path | None = None,
) -> None:
    _paper_rc()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    all_points = [base] + stages
    mras = [p[1] for p in all_points]
    cvs = [p[2] for p in all_points]

    y_lo, y_hi = _apply_centered_axes(ax, mras, cvs, chance_pct, cv_ref)
    _add_quadrant_labels(ax, chance_pct, cv_ref, y_lo, y_hi)

    ax.scatter(
        [base[1]], [base[2]], s=80, marker="s", color=BASE_COLOR,
        edgecolors="black", linewidths=0.6, zorder=5, label=base[0],
    )
    ax.annotate(
        base[0], (base[1], base[2]),
        textcoords="offset points", xytext=(6, 6), fontsize=8.5, fontweight="bold",
    )

    for i, (name, mra, cv) in enumerate(stages):
        color = STAGE_COLORS[i % len(STAGE_COLORS)]
        ax.scatter(
            [mra], [cv], s=70, marker="o", color=color,
            edgecolors="black", linewidths=0.5, zorder=5, label=name,
        )
        ax.annotate(
            name, (mra, cv),
            textcoords="offset points", xytext=(6, -8), fontsize=8,
        )

    chain = [base] + stages
    for i in range(len(chain) - 1):
        src = chain[i]
        dst = chain[i + 1]
        arrow = FancyArrowPatch(
            (src[1], src[2]), (dst[1], dst[2]),
            arrowstyle="-|>", mutation_scale=12,
            color="0.35", linewidth=1.2, zorder=4,
        )
        ax.add_patch(arrow)

    ax.set_xlabel("Overall MRA (%) — higher is better  →")
    ax.set_ylabel("Overall mean CV — lower CV toward top  (↑ better)")
    ax.set_title("Spatial SFT: accuracy vs consistency across training stages")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.text(
        0.01, 0.02,
        f"Axes centered at (chance MRA={chance_pct:.1f}%, CV midpoint={cv_ref:.2f}). "
        f"Arrows show training-stage progression.",
        fontsize=7.5, color="0.35",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    if out_pdf:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_png}" + (f", {out_pdf}" if out_pdf else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=str, default=None, metavar="NAME,MRA,CV")
    ap.add_argument("--stage", action="append", default=[], metavar="NAME,MRA,CV")
    ap.add_argument("--summary_json", type=Path, default=None)
    ap.add_argument("--mra_chance_pct", type=float, required=True)
    ap.add_argument("--cv_threshold", type=float, default=0.1)
    ap.add_argument("--out_png", type=Path, required=True)
    ap.add_argument("--out_pdf", type=Path, default=None)
    args = ap.parse_args()

    base: tuple[str, float, float] | None = None
    stages: list[tuple[str, float, float]] = []

    if args.summary_json:
        data = json.loads(args.summary_json.read_text(encoding="utf-8"))
        b = data["base"]
        base = (str(b["name"]), float(b["mra"]), float(b["cv"]))
        for s in data["stages"]:
            stages.append((str(s["name"]), float(s["mra"]), float(s["cv"])))

    if args.base:
        base = _parse_point(args.base)
    for raw in args.stage:
        stages.append(_parse_point(raw))

    assert base is not None, "Provide --base NAME,MRA,CV or --summary_json"
    assert stages, "Provide at least one --stage NAME,MRA,CV or --summary_json"

    plot_training_stages(
        base, stages,
        chance_pct=args.mra_chance_pct,
        cv_ref=args.cv_threshold,
        out_png=args.out_png,
        out_pdf=args.out_pdf,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
