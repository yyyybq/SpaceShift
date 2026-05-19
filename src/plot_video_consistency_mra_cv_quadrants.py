"""Multi-model MRA (%) vs mean CV with chance MRA line and CV midpoint (default 0.1).

Intersection (chance MRA, CV midpoint) is centered; CV axis is reversed so lower CV
is toward the top. Single-model figures from lmms-eval use the same layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lmms_eval.tasks.sceneshift.video_consistency_chance_baseline import weighted_chance_mra_percent
from lmms_eval.tasks.sceneshift.video_consistency_quadrant_axes import (
    add_quadrant_labels,
    apply_centered_quadrant_axes,
)


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def _chance_cached(jsonl_path: Path, cache_path: Path | None) -> float:
    if cache_path is None:
        return weighted_chance_mra_percent(jsonl_path.resolve())
    js = jsonl_path.resolve()
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("jsonl") == str(js) and payload.get("jsonl_mtime") == _mtime(js):
            return float(payload["chance_MRA_pct"])
    v = weighted_chance_mra_percent(js)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"jsonl": str(js), "jsonl_mtime": _mtime(js), "chance_MRA_pct": v}, indent=2) + "\n",
        encoding="utf-8",
    )
    return v


def _paper_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary_json", type=Path, default=None)
    ap.add_argument("--jsonl", type=Path, default=None, help="For chance MRA; required unless --mra_chance_pct set")
    ap.add_argument(
        "--chance_cache",
        type=Path,
        default=None,
        help="Read/write JSON with cached chance_MRA_pct when jsonl mtime matches",
    )
    ap.add_argument("--mra_chance_pct", type=float, default=None)
    ap.add_argument("--cv_threshold", type=float, default=0.1, help="CV midpoint (quadrant center on y)")
    ap.add_argument("--point", action="append", default=[], metavar="NAME,MRA,CV")
    ap.add_argument("--out_png", type=Path, required=True)
    ap.add_argument("--out_pdf", type=Path, default=None)
    args = ap.parse_args()

    points: list[tuple[str, float, float]] = []
    for raw in args.point:
        parts = raw.split(",")
        assert len(parts) == 3, raw
        points.append((parts[0].strip(), float(parts[1]), float(parts[2])))

    if args.summary_json:
        data = json.loads(args.summary_json.expanduser().resolve().read_text(encoding="utf-8"))
        assert isinstance(data, list), args.summary_json
        for row in data:
            points.append(
                (
                    str(row["model"]),
                    float(row["overall_MRA"]),
                    float(row["overall_CV"]),
                )
            )

    assert points, "Provide --summary_json and/or --point"

    if args.mra_chance_pct is not None:
        chance_pct = float(args.mra_chance_pct)
    else:
        assert args.jsonl is not None, "Need --jsonl or --mra_chance_pct"
        chance_pct = _chance_cached(args.jsonl.expanduser().resolve(), args.chance_cache)

    cv_ref = float(args.cv_threshold)

    _paper_rc()
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    mras = [p[1] for p in points]
    cvs = [p[2] for p in points]
    ax.scatter(mras, cvs, s=42, alpha=0.85, edgecolors="black", linewidths=0.4, zorder=3)
    for name, mra, cv in points:
        ax.annotate(name, (mra, cv), textcoords="offset points", xytext=(4, 4), fontsize=8)

    y_lo, y_hi = apply_centered_quadrant_axes(ax, mras, cvs, chance_pct, cv_ref)
    add_quadrant_labels(ax, chance_pct, cv_ref, y_lo, y_hi)

    ax.set_xlabel("Overall MRA (%) — higher is better →")
    ax.set_ylabel("Overall mean CV — reversed: lower CV toward top (↑ better)")
    ax.set_title("Video consistency: accuracy vs cross-view consistency")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.text(
        0.01,
        0.02,
        f"Axes centered at (chance MRA, CV midpoint). "
        f"Chance from NA-type baseline ({args.jsonl.name if args.jsonl else 'override'}). "
        f"CV midpoint = {cv_ref:.2f}.",
        fontsize=7.5,
        color="0.35",
    )

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png)
    if args.out_pdf:
        args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_pdf)
    plt.close(fig)
    print(f"Wrote {args.out_png}" + (f", {args.out_pdf}" if args.out_pdf else ""))
    print(f"chance_MRA_pct={chance_pct:.4f} cv_reference={cv_ref:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
