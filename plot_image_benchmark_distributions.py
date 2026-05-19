import matplotlib
"""Plot distribution charts for the image consistency benchmark."""

import argparse
import json
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", default="image_consistency_thor_eval_v2/dataset_stats.json")
    parser.add_argument("--out_path", default="figures/analysis/image_v2/image_benchmark_distributions.png")
    args = parser.parse_args()

    stats_path = Path(args.stats_path).expanduser().resolve()
    stats = json.loads(stats_path.read_text())
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Image Consistency Benchmark — {stats['total_groups']} groups, {stats['total_images']} images",
        fontsize=16, fontweight="bold",
    )

    # 1. Question family distribution (pie)
    ax = axes[0, 0]
    families = stats["question_family_distribution"]
    labels = [d["question_family"] for d in families]
    counts = [d["count"] for d in families]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    ax.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Question Family Distribution")

    # 2. Question type distribution (bar)
    ax = axes[0, 1]
    types = stats["question_type_distribution"]
    type_labels = [d["question_type"].replace("object_", "").replace("_", "\n") for d in types]
    type_counts = [d["count"] for d in types]
    bars = ax.bar(type_labels, type_counts, color=colors)
    ax.set_title("Question Type Distribution")
    ax.set_ylabel("Count")
    for bar, c in zip(bars, type_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(c),
                ha="center", va="bottom", fontsize=10)

    # 3. Images per group distribution (bar)
    ax = axes[0, 2]
    ipg = stats["images_per_group_distribution"]
    ipg_labels = [str(d["images_per_group"]) for d in ipg]
    ipg_counts = [d["count"] for d in ipg]
    bars = ax.bar(ipg_labels, ipg_counts, color="#4C72B0")
    ax.set_title("Images Per Group")
    ax.set_xlabel("Images per group")
    ax.set_ylabel("Number of groups")
    for bar, c in zip(bars, ipg_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(c),
                ha="center", va="bottom", fontsize=10)

    # 4. Scene distribution (top 20, horizontal bar)
    ax = axes[1, 0]
    scenes = stats["scene_distribution"][:20]
    scene_labels = [d["scene_id"] for d in scenes][::-1]
    scene_counts = [d["count"] for d in scenes][::-1]
    ax.barh(scene_labels, scene_counts, color="#55A868")
    ax.set_title("Scene Distribution (top 20)")
    ax.set_xlabel("Count")

    # 5. Anchor label distribution (horizontal bar)
    ax = axes[1, 1]
    anchors = stats["anchor_label_distribution"][:15]
    anchor_labels = [d["anchor_label"] for d in anchors][::-1]
    anchor_counts = [d["count"] for d in anchors][::-1]
    ax.barh(anchor_labels, anchor_counts, color="#C44E52")
    ax.set_title("Anchor Label Distribution (top 15)")
    ax.set_xlabel("Count")

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        f"Total groups: {stats['total_groups']}\n"
        f"Total images: {stats['total_images']}\n"
        f"Unique scenes: {len(stats['scene_distribution'])}\n"
        f"Unique anchor labels: {len(stats['anchor_label_distribution'])}\n\n"
        f"Groups with 10 images: {ipg[0]['count']}\n"
        f"Groups with <10 images: {sum(d['count'] for d in ipg if d['images_per_group'] < 10)}\n"
    )
    ax.text(0.1, 0.5, summary, fontsize=13, verticalalignment="center",
            fontfamily="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Summary")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
