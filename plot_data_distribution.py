"""
Visualization of eval_v2 / train_v2 OOD split distributions.
"""
import json, os
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path("/nas2/edwin/spatial-scene-variations")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Load data ──
splits = {}
for split in ["eval_v2", "train_v2"]:
    with open(ROOT / f"video_consistency_thor_{split}" / "qa.json") as f:
        splits[split] = json.load(f)

COLORS = {"eval_v2": "#2196F3", "train_v2": "#FF9800"}
QT_LABELS = {
    "object_dimensions": "Object\nDimensions",
    "object_distance_to_camera": "Camera\nDistance",
    "object_pair_distance_center": "Pair\nDistance",
}
QT_SHORT = {
    "object_dimensions": "dims",
    "object_distance_to_camera": "cam_dist",
    "object_pair_distance_center": "pair_dist",
}
ROOM_ORDER = ["kitchen", "living_room", "bedroom", "bathroom"]

# ── Helper ──
def extract(data):
    qt = Counter()
    rooms = Counter()
    objs = Counter()
    scenes = Counter()
    motion = Counter()
    qt_rooms = defaultdict(Counter)
    qt_objs = defaultdict(Counter)
    for e in data:
        rooms[e.get("room_bucket", "?")] += 1
        scenes[e["scene_id"]] += 1
        motion[e.get("motion_family", "?")] += 1
        for q in e["questions"]:
            qtype = q["question_type"]
            qt[qtype] += 1
            qt_rooms[qtype][e.get("room_bucket", "?")] += 1
            for l in q.get("anchor_labels", []):
                objs[l] += 1
                qt_objs[qtype][l] += 1
    return dict(qt=qt, rooms=rooms, objs=objs, scenes=scenes, motion=motion,
                qt_rooms=qt_rooms, qt_objs=qt_objs)

stats = {s: extract(d) for s, d in splits.items()}

# ================================================================
# Figure 1: Overview (QA types, rooms, motion) side by side
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Eval vs Train — Overview Distribution", fontsize=15, fontweight="bold", y=1.02)

# 1a. QA type
ax = axes[0]
qt_keys = sorted(QT_LABELS.keys())
x = np.arange(len(qt_keys))
w = 0.35
for i, split in enumerate(["eval_v2", "train_v2"]):
    vals = [stats[split]["qt"].get(k, 0) for k in qt_keys]
    bars = ax.bar(x + i*w, vals, w, label=split.replace("_v2",""), color=COLORS[split], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, str(v),
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + w/2)
ax.set_xticklabels([QT_LABELS[k] for k in qt_keys], fontsize=9)
ax.set_ylabel("Clips")
ax.set_title("QA Type Distribution")
ax.legend()

# 1b. Room bucket
ax = axes[1]
x = np.arange(len(ROOM_ORDER))
for i, split in enumerate(["eval_v2", "train_v2"]):
    vals = [stats[split]["rooms"].get(k, 0) for k in ROOM_ORDER]
    bars = ax.bar(x + i*w, vals, w, label=split.replace("_v2",""), color=COLORS[split], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, str(v),
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + w/2)
ax.set_xticklabels(ROOM_ORDER, fontsize=9)
ax.set_ylabel("Clips")
ax.set_title("Room Distribution")
ax.legend()

# 1c. Motion family
ax = axes[2]
motion_keys = sorted(set(list(stats["eval_v2"]["motion"].keys()) + list(stats["train_v2"]["motion"].keys())))
x = np.arange(len(motion_keys))
for i, split in enumerate(["eval_v2", "train_v2"]):
    vals = [stats[split]["motion"].get(k, 0) for k in motion_keys]
    bars = ax.bar(x + i*w, vals, w, label=split.replace("_v2",""), color=COLORS[split], edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, str(v),
                    ha="center", va="bottom", fontsize=7)
ax.set_xticks(x + w/2)
ax.set_xticklabels(motion_keys, fontsize=9)
ax.set_ylabel("Clips")
ax.set_title("Motion Family Distribution")
ax.legend()

plt.tight_layout()
fig.savefig(FIG_DIR / "ood_split_overview.png", dpi=150, bbox_inches="tight")
print("Saved ood_split_overview.png")
plt.close()

# ================================================================
# Figure 2: Object distribution — eval (10 objects)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Eval — Object Distribution per QA Type", fontsize=14, fontweight="bold", y=1.02)

for idx, qtype in enumerate(qt_keys):
    ax = axes[idx]
    obj_counts = stats["eval_v2"]["qt_objs"][qtype]
    if not obj_counts:
        ax.set_title(QT_SHORT[qtype])
        continue
    sorted_objs = sorted(obj_counts.items(), key=lambda x: -x[1])
    labels = [o for o, _ in sorted_objs]
    vals = [c for _, c in sorted_objs]
    bars = ax.barh(range(len(labels)), vals, color=COLORS["eval_v2"], edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(v),
                va="center", fontsize=8)
    ax.set_title(QT_SHORT[qtype], fontsize=11)
    ax.set_xlabel("Clips")

plt.tight_layout()
fig.savefig(FIG_DIR / "ood_split_eval_objects.png", dpi=150, bbox_inches="tight")
print("Saved ood_split_eval_objects.png")
plt.close()

# ================================================================
# Figure 3: Object distribution — train (top 30)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 10))
fig.suptitle("Train (OOD) — Object Distribution per QA Type (top 30)", fontsize=14, fontweight="bold", y=1.01)

for idx, qtype in enumerate(qt_keys):
    ax = axes[idx]
    obj_counts = stats["train_v2"]["qt_objs"][qtype]
    if not obj_counts:
        ax.set_title(QT_SHORT[qtype])
        continue
    sorted_objs = sorted(obj_counts.items(), key=lambda x: -x[1])[:30]
    labels = [o for o, _ in sorted_objs]
    vals = [c for _, c in sorted_objs]
    bars = ax.barh(range(len(labels)), vals, color=COLORS["train_v2"], edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(v),
                va="center", fontsize=7)
    ax.set_title(QT_SHORT[qtype], fontsize=11)
    ax.set_xlabel("Clips")

plt.tight_layout()
fig.savefig(FIG_DIR / "ood_split_train_objects.png", dpi=150, bbox_inches="tight")
print("Saved ood_split_train_objects.png")
plt.close()

# ================================================================
# Figure 4: Room x QA type heatmaps
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
fig.suptitle("Room x QA Type Heatmap", fontsize=14, fontweight="bold", y=1.02)

for idx, split in enumerate(["eval_v2", "train_v2"]):
    ax = axes[idx]
    matrix = []
    for room in ROOM_ORDER:
        row = [stats[split]["qt_rooms"][qt].get(room, 0) for qt in qt_keys]
        matrix.append(row)
    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap="Blues" if split == "eval_v2" else "Oranges", aspect="auto")
    ax.set_xticks(range(len(qt_keys)))
    ax.set_xticklabels([QT_SHORT[k] for k in qt_keys], fontsize=9)
    ax.set_yticks(range(len(ROOM_ORDER)))
    ax.set_yticklabels(ROOM_ORDER, fontsize=9)
    for i in range(len(ROOM_ORDER)):
        for j in range(len(qt_keys)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max()*0.6 else "black")
    ax.set_title(split.replace("_v2", ""), fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
fig.savefig(FIG_DIR / "ood_split_room_qa_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved ood_split_room_qa_heatmap.png")
plt.close()

# ================================================================
# Figure 5: Object OOD Venn-style bar chart
# ================================================================
eval_objs = set()
train_objs = set()
for e in splits["eval_v2"]:
    for q in e["questions"]:
        eval_objs.update(q.get("anchor_labels", []))
for e in splits["train_v2"]:
    for q in e["questions"]:
        train_objs.update(q.get("anchor_labels", []))

fig, ax = plt.subplots(figsize=(8, 4))
categories = ["Eval only", "Overlap", "Train only"]
values = [len(eval_objs - train_objs), len(eval_objs & train_objs), len(train_objs - eval_objs)]
colors_v = [COLORS["eval_v2"], "#9E9E9E", COLORS["train_v2"]]
bars = ax.bar(categories, values, color=colors_v, edgecolor="white", width=0.5)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(v),
            ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Objects")
ax.set_title("Object OOD Split — Zero Overlap", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "ood_split_object_overlap.png", dpi=150, bbox_inches="tight")
print("Saved ood_split_object_overlap.png")
plt.close()

# ================================================================
# Figure 6: Scene distribution
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Scene Distribution (clips per scene)", fontsize=14, fontweight="bold", y=1.02)

for idx, split in enumerate(["eval_v2", "train_v2"]):
    ax = axes[idx]
    scene_counts = stats[split]["scenes"]
    vals = sorted(scene_counts.values(), reverse=True)
    ax.bar(range(len(vals)), vals, color=COLORS[split], edgecolor="none", width=1.0)
    ax.set_xlabel(f"Scenes (n={len(vals)})")
    ax.set_ylabel("Clips")
    ax.set_title(split.replace("_v2", ""), fontsize=12)
    ax.axhline(np.mean(vals), color="red", ls="--", lw=1, label=f"mean={np.mean(vals):.1f}")
    ax.legend()

plt.tight_layout()
fig.savefig(FIG_DIR / "ood_split_scene_distribution.png", dpi=150, bbox_inches="tight")
print("Saved ood_split_scene_distribution.png")
plt.close()

print("\nAll figures saved to:", FIG_DIR)
