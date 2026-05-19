"""Compare QA distributions for Thor vs InteriorGS production `qa.json` runs.

Usage:
  PYTHONPATH=src python src/plot_video_consistency_production_compare.py \\
    --thor_dir ./video_consistency_thor_production \\
    --interiorgs_dir ./video_consistency_interiorGS_production \\
    --out_dir ./video_consistency_production_figures

Input: each directory must contain `qa.json` (list of clips with `questions[]`).

Output: `production_distribution_compare.png` and `.pdf` with grouped bars:
  Thor-only, InteriorGS-only, and merged (sum) per category for question type,
  question family, motion, and dimension.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_video_consistency_qa import (
    QUESTION_FAMILY_DISPLAY,
    QUESTION_TYPE_DISPLAY,
    _display_counter,
    _flatten_qa,
)


def _paper_rc():
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


def _grouped_bars(
    ax,
    categories: list[str],
    thor_c: Counter,
    int_c: Counter,
    title: str,
) -> None:
    x = np.arange(len(categories))
    w = 0.25
    t_vals = [thor_c.get(c, 0) for c in categories]
    i_vals = [int_c.get(c, 0) for c in categories]
    m_vals = [ta + tb for ta, tb in zip(t_vals, i_vals)]
    ax.bar(x - w, t_vals, width=w, label="Thor", color="#2c7fb8")
    ax.bar(x, i_vals, width=w, label="InteriorGS", color="#dd8d34")
    ax.bar(x + w, m_vals, width=w, label="Merged", color="#7b3294", alpha=0.85)
    ax.set_xticks(x, categories, rotation=22, ha="right")
    ax.set_ylabel("QA instance count")
    ax.set_title(title)
    ax.legend(loc="upper right")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--thor_dir", type=Path, required=True)
    p.add_argument("--interiorgs_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()
    thor_qa = args.thor_dir.expanduser().resolve() / "qa.json"
    int_qa = args.interiorgs_dir.expanduser().resolve() / "qa.json"
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert thor_qa.is_file(), thor_qa
    assert int_qa.is_file(), int_qa
    rows_t = _flatten_qa(thor_qa)
    rows_i = _flatten_qa(int_qa)
    n_t, n_i = len(rows_t), len(rows_i)
    _paper_rc()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Production QA distributions — Thor N={n_t}, InteriorGS N={n_i}, merged N={n_t + n_i}",
        fontsize=12,
        y=1.02,
    )

    qt_t = _display_counter(Counter(r["question_type"] for r in rows_t), QUESTION_TYPE_DISPLAY)
    qt_i = _display_counter(Counter(r["question_type"] for r in rows_i), QUESTION_TYPE_DISPLAY)
    keys_qt = sorted(set(qt_t) | set(qt_i), key=lambda k: -(qt_t.get(k, 0) + qt_i.get(k, 0)))
    _grouped_bars(axes[0, 0], keys_qt, qt_t, qt_i, "(a) Question type")

    qf_t = _display_counter(Counter(r["question_family"] for r in rows_t), QUESTION_FAMILY_DISPLAY)
    qf_i = _display_counter(Counter(r["question_family"] for r in rows_i), QUESTION_FAMILY_DISPLAY)
    keys_qf = sorted(set(qf_t) | set(qf_i), key=lambda k: -(qf_t.get(k, 0) + qf_i.get(k, 0)))
    _grouped_bars(axes[0, 1], keys_qf, qf_t, qf_i, "(b) Question family")

    mf_t = Counter(r["motion_family"] for r in rows_t)
    mf_i = Counter(r["motion_family"] for r in rows_i)
    keys_mf = sorted(set(mf_t) | set(mf_i), key=lambda k: -(mf_t.get(k, 0) + mf_i.get(k, 0)))
    _grouped_bars(axes[1, 0], keys_mf, mf_t, mf_i, "(c) Motion family")

    dim_t = Counter(str(r["dimension"]) for r in rows_t if r["dimension"])
    dim_i = Counter(str(r["dimension"]) for r in rows_i if r["dimension"])
    keys_d = sorted(set(dim_t) | set(dim_i), key=lambda k: -(dim_t.get(k, 0) + dim_i.get(k, 0)))
    if keys_d:
        _grouped_bars(axes[1, 1], keys_d, dim_t, dim_i, "(d) Dimension (size questions)")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "No dimension fields", ha="center", va="center")

    plt.tight_layout()
    stem = out_dir / "production_distribution_compare"
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')}\nWrote {stem.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
