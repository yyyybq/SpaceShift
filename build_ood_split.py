"""
Build OOD eval/train split from existing rendered data.

Eval: 19 objects, 1000 clips per QA type = 3000 total
  - Filters out degenerate GT (objects where all answers are identical)
  - Balanced across: object > scene > motion > room
Train: all clips where every object is OOD (not in eval set)
"""
import json, os, random, shutil
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path("/nas2/edwin/spatial-scene-variations")
EVAL_OBJECTS = {
    # Living room / kitchen (dims + cam + pair)
    "ArmChair", "Box", "Chair", "CoffeeTable", "DiningTable",
    # Kitchen pair anchors
    "Cabinet", "CoffeeMachine", "Drawer", "Faucet", "Fridge",
    # Bathroom + bedroom (pair)
    "Toilet", "Sink", "Mirror", "Bathtub",
    # Multi-room diversity
    "HousePlant", "GarbageCan", "Dresser", "Laptop", "Painting",
    # New: all-4-room + bathroom + bedroom
    "LightSwitch", "SideTable",  # all 4 rooms
    "HandTowel", "DeskLamp", "Pillow",  # bathroom/bedroom
}
PER_QA = 1000
MIN_GT_UNIQUE = 2
SOURCE_DATASETS = [
    "old/video_consistency_thor_eval_final",
    "old/video_consistency_thor_train_expanded",
    "old/video_consistency_thor_balanced",
    "old/video_consistency_thor_train",
    "old/video_consistency_thor_small",
    "video_consistency_thor_motion_supplement",
    "video_consistency_thor_diversity_supplement",
]
OUT_EVAL = ROOT / "video_consistency_thor_eval_v2"
OUT_TRAIN = ROOT / "video_consistency_thor_train_v2"

# ── 1. Pool ──
print("Pooling clips...")
all_clips = {}
for rel in SOURCE_DATASETS:
    qa_path = ROOT / rel / "qa.json"
    vid_dir = ROOT / rel / "videos"
    if not qa_path.exists() or not vid_dir.is_dir():
        continue
    with open(qa_path) as f:
        data = json.load(f)
    rendered = set(os.listdir(vid_dir))
    count = 0
    for e in data:
        cid = e["clip_id"]
        if cid in rendered and cid not in all_clips:
            if not (vid_dir / cid / "video.mp4").exists():
                continue
            e["_source"] = rel
            e["_video_dir"] = str(vid_dir)
            all_clips[cid] = e
            count += 1
    print(f"  {rel}: +{count}")
print(f"  Total: {len(all_clips)}")

# ── 2. GT variation index ──
gt_values = defaultdict(set)
for e in all_clips.values():
    for q in e["questions"]:
        labels = q.get("anchor_labels", [])
        primary = labels[0] if labels else "?"
        qt = q["question_type"]
        try:
            val = round(float(q.get("answer", q.get("ground_truth", ""))), 2)
            gt_values[(primary, qt)].add(val)
        except (ValueError, TypeError):
            pass

degenerate = set()
for (obj, qt), vals in gt_values.items():
    if len(vals) < MIN_GT_UNIQUE:
        degenerate.add((obj, qt))
        print(f"  DEGENERATE: {obj} x {qt} ({len(vals)} unique)")

# ── 3. Partition ──
eval_pool = []
train_entries = []

for e in all_clips.values():
    all_labels = set()
    for q in e["questions"]:
        all_labels.update(q.get("anchor_labels", []))
    if not all_labels:
        continue
    if all_labels.issubset(EVAL_OBJECTS):
        qt = e["questions"][0]["question_type"]
        primary = e["questions"][0].get("anchor_labels", ["?"])[0]
        if (primary, qt) in degenerate:
            continue
        eval_pool.append(e)
    elif all_labels.isdisjoint(EVAL_OBJECTS):
        train_entries.append(e)

print(f"\nEval-eligible: {len(eval_pool)}")
print(f"OOD train: {len(train_entries)}")

# ── 4. Balanced sampling: object > scene > motion ──


def get_primary_object(e):
    for q in e["questions"]:
        labels = q.get("anchor_labels", [])
        if labels:
            return labels[0]
    return "?"


def balanced_fill(groups: dict[str, list], total: int) -> list:
    """Draw from groups as evenly as possible. Smallest groups get fully used."""
    keys = sorted(groups, key=lambda k: len(groups[k]))
    remaining = total
    n_groups = len(keys)
    selected = []
    for i, key in enumerate(keys):
        g = list(groups[key])
        random.shuffle(g)
        share = remaining // (n_groups - i)
        take = min(len(g), share)
        selected.extend(g[:take])
        remaining -= take
    random.shuffle(selected)
    return selected[:total]


def sample_qa_type(pool, total, qt_name):
    """
    Sample with hierarchy: object > scene > motion.
    This ensures both object diversity AND scene diversity.
    """
    # Level 1: group by primary object
    by_object = defaultdict(list)
    for e in pool:
        by_object[get_primary_object(e)].append(e)

    # Within each object, balance by scene, then motion
    balanced_objects = {}
    for obj, clips in by_object.items():
        # Level 2: within this object, group by scene
        by_scene = defaultdict(list)
        for e in clips:
            by_scene[e["scene_id"]].append(e)

        # Level 3: within each scene, group by motion
        balanced_scenes = {}
        for scene, s_clips in by_scene.items():
            by_motion = defaultdict(list)
            for e in s_clips:
                by_motion[e.get("motion_family", "?")].append(e)
            balanced_scenes[scene] = balanced_fill(by_motion, len(s_clips))

        balanced_objects[obj] = balanced_fill(balanced_scenes, len(clips))

    result = balanced_fill(balanced_objects, total)

    # Report
    mc = Counter(e.get("motion_family", "?") for e in result)
    oc = Counter(get_primary_object(e) for e in result)
    rc = Counter(e.get("room_bucket", "?") for e in result)
    sc = Counter(e["scene_id"] for e in result)
    print(f"\n  {qt_name}: {len(result)} clips")
    print(f"    objects ({len(oc)}): {dict(sorted(oc.items(), key=lambda x: -x[1]))}")
    print(f"    scenes: {len(sc)} unique, top3: {sc.most_common(3)}, max/clip={max(sc.values())}")
    print(f"    motions: {dict(sorted(mc.items(), key=lambda x: -x[1]))}")
    print(f"    rooms:   {dict(sorted(rc.items(), key=lambda x: -x[1]))}")
    return result


pool_by_qt = defaultdict(list)
for e in eval_pool:
    qt = e["questions"][0]["question_type"]
    pool_by_qt[qt].append(e)

eval_entries = []
print("\nSampling eval:")
for qt in ["object_dimensions", "object_distance_to_camera", "object_pair_distance_center"]:
    sampled = sample_qa_type(pool_by_qt[qt], PER_QA, qt)
    eval_entries.extend(sampled)

print(f"\nFinal eval: {len(eval_entries)}")
print(f"Final train: {len(train_entries)}")

# ── 5. Write ──
def write_dataset(out_dir, entries):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    vid_out = out_dir / "videos"
    vid_out.mkdir(exist_ok=True)
    qa_out = []
    for e in entries:
        cid = e["clip_id"]
        src_vid = Path(e["_video_dir"]) / cid
        dst_vid = vid_out / cid
        if not dst_vid.exists():
            os.symlink(src_vid, dst_vid)
        entry = {k: v for k, v in e.items() if not k.startswith("_")}
        entry["video_path"] = f"./{out_dir.name}/videos/{cid}/video.mp4"
        qa_out.append(entry)
    with open(out_dir / "qa.json", "w") as f:
        json.dump(qa_out, f, indent=2)
    print(f"  Wrote {out_dir.name}: {len(qa_out)} clips")

print("\nWriting datasets...")
write_dataset(OUT_EVAL, eval_entries)
write_dataset(OUT_TRAIN, train_entries)

# ── 6. Summary ──
print("\n" + "=" * 70)
print("EVAL SUMMARY")
print("=" * 70)

scenes = Counter(e["scene_id"] for e in eval_entries)
qt_c = Counter(e["questions"][0]["question_type"] for e in eval_entries)
rooms = Counter(e.get("room_bucket", "?") for e in eval_entries)
obj_c = Counter()
for e in eval_entries:
    for q in e["questions"]:
        for l in q.get("anchor_labels", []):
            obj_c[l] += 1

print(f"Total: {len(eval_entries)} clips")
print(f"QA types: {dict(qt_c)}")
print(f"Scenes: {len(scenes)} unique, max clips/scene={max(scenes.values())}, "
      f"mean={np.mean(list(scenes.values())):.1f}, median={np.median(list(scenes.values())):.0f}")
print(f"Rooms: {dict(rooms)}")
print(f"Objects ({len(obj_c)}): {dict(obj_c.most_common())}")

# Per QA type detail
for qt in sorted(qt_c):
    clips = [e for e in eval_entries if e["questions"][0]["question_type"] == qt]
    sc = Counter(e["scene_id"] for e in clips)
    obj_scene = defaultdict(set)
    for e in clips:
        obj_scene[get_primary_object(e)].add(e["scene_id"])
    print(f"\n  {qt}:")
    print(f"    scenes: {len(sc)} unique, max={max(sc.values())}")
    for obj in sorted(obj_scene, key=lambda o: -len(obj_scene[o])):
        print(f"      {obj:<16} {len(obj_scene[obj]):>3} scenes")

# OOD check
eval_objs = set()
train_objs = set()
for e in eval_entries:
    for q in e["questions"]:
        eval_objs.update(q.get("anchor_labels", []))
for e in train_entries:
    for q in e["questions"]:
        train_objs.update(q.get("anchor_labels", []))
overlap = eval_objs & train_objs
print(f"\nObject overlap: {len(overlap)}")
if overlap:
    print(f"  OVERLAP: {sorted(overlap)}")
else:
    print("  CONFIRMED: 0 overlap — fully OOD")
