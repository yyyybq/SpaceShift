"""Plot before/after benchmark reorganization summary figures.

Usage:
  python src/plot_video_consistency_reorganization_report.py \
    --before_dir old/benchmarks/video_consistency_thor_eval_v2_before_family_balance \
    --after_dir video_consistency_thor_eval_v2 \
    --out_dir old/reports/video_consistency_reorganization
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


def _load_groups(dataset_dir: Path) -> list[dict]:
    return json.loads((dataset_dir / "consistency_groups.json").read_text(encoding="utf-8"))


def _load_group_motion_counts(dataset_dir: Path, groups: list[dict]) -> Counter:
    if groups and any((group.get("motion_family") not in (None, "")) for group in groups):
        return Counter((group.get("motion_family") or "unknown") for group in groups)
    qa_path = dataset_dir / "qa.json"
    if qa_path.exists():
        qa_rows = json.loads(qa_path.read_text(encoding="utf-8"))
        by_gid: dict[str, str] = {}
        for row in qa_rows:
            by_gid.setdefault(str(row["group_id"]), str(row.get("motion_family") or "unknown"))
        return Counter(by_gid.values())
    return Counter()


def _atomic_label_counts(groups: list[dict]) -> Counter:
    return Counter(
        str(label).split("|")[0].strip()
        for group in groups
        for label in (group.get("anchor_labels") or [])
        if str(label).strip()
    )


def _group_size_counts(groups: list[dict]) -> Counter:
    return Counter(len(group["clip_ids"]) for group in groups)


def _scene_counts(groups: list[dict], family: str | None = None) -> Counter:
    rows = groups if family is None else [group for group in groups if group["question_family"] == family]
    return Counter(group["scene_id"] for group in rows)


def _paired_barh(ax, before: Counter, after: Counter, title: str, topn: int = 15):
    ordered = []
    for key, _ in after.most_common(topn):
        if key not in ordered:
            ordered.append(key)
    for key, _ in before.most_common(topn):
        if key not in ordered:
            ordered.append(key)
    ordered = ordered[:topn]
    y = np.arange(len(ordered))
    before_vals = [before.get(key, 0) for key in ordered]
    after_vals = [after.get(key, 0) for key in ordered]
    ax.barh(y + 0.18, before_vals, height=0.34, color="#9aa5b1", label="Before")
    ax.barh(y - 0.18, after_vals, height=0.34, color="#1f77b4", label="After")
    ax.set_yticks(y, ordered)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Groups")
    ax.legend(frameon=False, loc="lower right")


def _motion_panel(ax, before_counts: Counter, after_counts: Counter):
    ordered = []
    canonical = ["rotation", "spherical", "around", "approach", "passby", "unknown"]
    for key in canonical:
        if before_counts.get(key, 0) or after_counts.get(key, 0):
            ordered.append(key)
    for key in list(before_counts) + list(after_counts):
        if key not in ordered:
            ordered.append(key)
    x = np.arange(len(ordered))
    before_vals = [before_counts.get(key, 0) for key in ordered]
    after_vals = [after_counts.get(key, 0) for key in ordered]
    ax.bar(x - 0.18, before_vals, width=0.34, color="#9aa5b1", label="Before")
    ax.bar(x + 0.18, after_vals, width=0.34, color="#1f77b4", label="After")
    ax.set_xticks(x, ordered, rotation=25, ha="right")
    ax.set_ylabel("Groups")
    ax.set_title("Motion-family distribution")
    ax.legend(frameon=False)


def _summary_text(before_groups: list[dict], after_groups: list[dict]) -> str:
    before_scene = _scene_counts(before_groups)
    after_scene = _scene_counts(after_groups)
    before_pair = _scene_counts(before_groups, family="pair_distance")
    after_pair = _scene_counts(after_groups, family="pair_distance")
    before_atomic = _atomic_label_counts(before_groups)
    after_atomic = _atomic_label_counts(after_groups)
    return "\n".join(
        [
            f"Scenes: {len(before_scene)} → {len(after_scene)}",
            f"Atomic labels: {len(before_atomic)} → {len(after_atomic)}",
            f"CoffeeTable: {before_atomic.get('CoffeeTable', 0)} → {after_atomic.get('CoffeeTable', 0)}",
            f"Pair FloorPlan1 groups: {before_pair.get('FloorPlan1', 0)} → {after_pair.get('FloorPlan1', 0)}",
            f"Max scene count: {max(before_scene.values())} → {max(after_scene.values())}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before_dir", required=True)
    parser.add_argument("--after_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    before_dir = Path(args.before_dir).expanduser().resolve()
    after_dir = Path(args.after_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    before_groups = _load_groups(before_dir)
    after_groups = _load_groups(after_dir)
    before_motion = _load_group_motion_counts(before_dir, before_groups)
    after_motion = _load_group_motion_counts(after_dir, after_groups)

    plt.rcParams.update(
        {
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
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Video consistency benchmark reorganization: before vs after", fontsize=13, y=1.02)

    _motion_panel(axes[0, 0], before_motion, after_motion)
    _paired_barh(
        axes[0, 1],
        _scene_counts(before_groups),
        _scene_counts(after_groups),
        "Overall scene distribution",
        topn=15,
    )
    _paired_barh(
        axes[1, 0],
        _scene_counts(before_groups, family="pair_distance"),
        _scene_counts(after_groups, family="pair_distance"),
        "Pair-distance scene distribution",
        topn=15,
    )
    _paired_barh(
        axes[1, 1],
        _atomic_label_counts(before_groups),
        _atomic_label_counts(after_groups),
        "Atomic object label distribution",
        topn=15,
    )

    fig.text(0.66, 0.05, _summary_text(before_groups, after_groups), fontsize=10, family="monospace")
    plt.tight_layout()

    png_path = out_dir / "video_consistency_reorganization_report.png"
    pdf_path = out_dir / "video_consistency_reorganization_report.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
