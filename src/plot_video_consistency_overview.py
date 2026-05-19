"""Plot an overall benchmark overview for the current video consistency v2 set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_groups(path: Path) -> list[dict]:
    groups = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(groups, list) and groups, f"Expected non-empty list in {path}"
    return groups


def _atomic_label_counts(groups: list[dict]) -> Counter:
    return Counter(
        str(label).split("|")[0].strip()
        for group in groups
        for label in (group.get("anchor_labels") or [])
        if str(label).strip()
    )


def _bar(ax, counts: Counter, title: str, xlabel: str = "Groups"):
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    colors = plt.cm.Set2.colors[: len(labels)]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(xlabel)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center", va="bottom", fontsize=8)


def _barh(ax, counts: Counter, title: str, topn: int):
    items = counts.most_common(topn)
    labels = [label for label, _ in items][::-1]
    values = [value for _, value in items][::-1]
    ax.barh(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Groups")
    for idx, value in enumerate(values):
        ax.text(value + 0.3, idx, str(value), va="center", fontsize=8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups_path", default="video_consistency_thor_eval_v2/consistency_groups.json")
    parser.add_argument("--out_path", default="figures/analysis/video_v2/video_benchmark_overview.png")
    args = parser.parse_args()

    groups_path = Path(args.groups_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = _load_groups(groups_path)
    family_counts = Counter(group["question_family"] for group in groups)
    motion_counts = Counter((group.get("motion_family") or "unknown") for group in groups)
    size_groups = [group for group in groups if group["question_family"] == "size"]
    dimension_counts = Counter((group.get("dimension") or "unknown") for group in size_groups)
    room_counts = Counter((group.get("room_bucket") or "unknown") for group in groups)
    scene_counts = Counter(group["scene_id"] for group in groups)
    label_counts = _atomic_label_counts(groups)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Video Consistency v2 Benchmark Overview", fontsize=13, y=1.02)

    _bar(axes[0, 0], family_counts, "Question family distribution")
    _bar(axes[0, 1], motion_counts, "Motion family distribution")
    _bar(axes[0, 2], dimension_counts, "Size dimension distribution")
    _barh(axes[1, 0], scene_counts, "Top scenes", topn=12)
    _barh(axes[1, 1], label_counts, "Top atomic object labels", topn=12)

    axes[1, 2].axis("off")
    summary = "\n".join(
        [
            f"Groups: {len(groups)}",
            f"Videos: {sum(len(group['clip_ids']) for group in groups)}",
            f"Unique scenes: {len(scene_counts)}",
            f"Unique object labels: {len(label_counts)}",
            f"Size groups: {len(size_groups)}",
            f"Room buckets: {dict(room_counts)}",
            f"Max scene count: {max(scene_counts.values())}",
        ]
    )
    axes[1, 2].text(
        0.03,
        0.95,
        summary,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F7F3E8", "alpha": 0.9},
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].set_title("Dataset summary")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
