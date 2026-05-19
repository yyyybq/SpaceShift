"""Build a maximally GT-diverse training dataset from mined candidates.

Usage:
    # After mining:
    python build_diverse_training.py \
        --mined_pkl video_consistency_thor_train_diverse/plan_mined_candidates.pkl \
        --output_dir video_consistency_thor_train_diverse \
        --clips_per_qt 600 \
        --clips_per_group 3

    # Then render:
    bash submit_dataset.sh render \
        --output_dir ./video_consistency_thor_train_diverse \
        --engines thor --render_parallel_workers 4 --gpu 0

    # Then convert to training JSONL:
    python build_diverse_training.py \
        --convert_jsonl video_consistency_thor_train_diverse/qa.json \
        --jsonl_out /nas2/edwin/sceneshift_training/qwen-vl-finetune/data/qwen3VL_2b_video.jsonl
"""

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path


EVAL_OBJECTS = frozenset({
    "CoffeeTable", "Chair", "Box", "ArmChair", "DiningTable",
    "Cabinet", "CoffeeMachine", "Drawer", "Faucet", "Fridge",
    "Toilet", "Sink", "Bathtub", "HandTowel",
    "Mirror", "LightSwitch", "SideTable", "GarbageCan",
    "HousePlant", "Dresser", "Laptop", "Painting",
    "DeskLamp", "Pillow",
})


def _has_eval_object(labels):
    return any(l in EVAL_OBJECTS for l in labels)


def select_diverse_candidates(candidates, clips_per_qt, clips_per_group, seed=42):
    """Select candidates maximizing GT diversity per QA type, balanced across types."""
    rng = random.Random(seed)

    # Group by QA type
    by_qt = defaultdict(list)
    for c in candidates:
        if _has_eval_object(c["anchor_labels"]):
            continue
        gt_vals = set()
        # Check for degenerate: need the GT
        gt = str(c["ground_truth"])
        qt = c["question_type"]
        by_qt[qt].append(c)

    print(f"\n{'='*70}")
    print(f"Candidate pool (excluding eval objects):")
    for qt, cands in sorted(by_qt.items()):
        unique_gt = len(set(str(c["ground_truth"]) for c in cands))
        total_clips = sum(len(c["available_clips"]) for c in cands)
        print(f"  {qt}: {len(cands)} candidates, {unique_gt} unique GT, {total_clips} clips")

    selected_all = []

    for qt, cands in sorted(by_qt.items()):
        # Group candidates by GT value
        gt_groups = defaultdict(list)
        for c in cands:
            gt_groups[str(c["ground_truth"])].append(c)

        unique_gts = sorted(gt_groups.keys(), key=lambda x: float(x))
        n_gts = len(unique_gts)

        # Target: clips_per_qt clips, distributed across GT values
        # Each GT value gets ~ clips_per_qt / n_gts clips
        target_per_gt = max(1, clips_per_qt // n_gts)
        remainder = clips_per_qt - target_per_gt * n_gts

        print(f"\n--- {qt}: {n_gts} unique GT, target {clips_per_qt} clips ---")

        selected_qt = []
        for gt_val in unique_gts:
            gt_cands = gt_groups[gt_val]
            rng.shuffle(gt_cands)

            # Prioritize scene diversity
            scene_seen = set()
            gt_selected = []

            for c in gt_cands:
                if c["scene_id"] not in scene_seen:
                    # Pick best clips from this candidate
                    clips = sorted(
                        c["available_clips"],
                        key=lambda x: x.get("quality_score", 0),
                        reverse=True,
                    )[:clips_per_group]
                    gt_selected.append((c, clips))
                    scene_seen.add(c["scene_id"])

                total_clips_so_far = sum(len(clips) for _, clips in gt_selected)
                if total_clips_so_far >= target_per_gt:
                    break

            # If not enough, add from same scenes
            if sum(len(clips) for _, clips in gt_selected) < target_per_gt:
                for c in gt_cands:
                    if not any(c is s[0] for s in gt_selected):
                        clips = sorted(
                            c["available_clips"],
                            key=lambda x: x.get("quality_score", 0),
                            reverse=True,
                        )[:clips_per_group]
                        gt_selected.append((c, clips))
                    if sum(len(clips) for _, clips in gt_selected) >= target_per_gt:
                        break

            selected_qt.extend(gt_selected)

        # Trim to target
        total = sum(len(clips) for _, clips in selected_qt)
        print(f"  Selected: {len(selected_qt)} candidates, {total} clips")
        selected_all.extend(selected_qt)

    return selected_all


def build_plan_files(selected, output_dir, config_template=None):
    """Write benchmark_plan.json, metadata.json, qa.json, clips.jsonl, consistency_groups.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_str = str(out.resolve())

    all_clips = []
    all_groups = []
    all_qa = []
    group_counter = 0

    for cand, clips in selected:
        group_counter += 1
        group_id = f"train_diverse_{group_counter:05d}"

        group_clips = []
        for i, clip_spec in enumerate(clips):
            clip_id = f"{group_id}_clip{i:02d}"
            video_path = f"./{out.name}/videos/{clip_id}/video.mp4"

            clip_row = {
                "clip_id": clip_id,
                "group_id": group_id,
                "output_dir": out_str,
                "video_path": video_path,
                "engine": cand["engine"],
                "scene_id": cand["scene_id"],
                "room_bucket": cand.get("room_bucket", ""),
                "motion_family": cand["motion_family"],
                "trajectory": clip_spec["trajectory"],
                "direction": clip_spec["direction"],
                "variation_tag": clip_spec["variation_tag"],
                "radius": clip_spec.get("radius") or cand.get("radius"),
                "start_angle_offset": clip_spec.get("start_angle_offset", 0),
                "cycle_count": clip_spec.get("cycle_count", 1),
                "quality_score": clip_spec.get("quality_score", 0),
                "question": cand["question"],
                "ground_truth": str(cand["ground_truth"]),
                "question_type": cand["question_type"],
                "question_family": cand["question_family"],
                "dimension": cand.get("dimension"),
                "anchor_kind": cand["anchor_kind"],
                "anchor_ids": list(cand["anchor_ids"]),
                "anchor_labels": list(cand["anchor_labels"]),
            }
            all_clips.append(clip_row)
            group_clips.append(clip_row)

            all_qa.append({
                "clip_id": clip_id,
                "group_id": group_id,
                "video_path": video_path,
                "engine": cand["engine"],
                "scene_id": cand["scene_id"],
                "room_bucket": cand.get("room_bucket", ""),
                "motion_family": cand["motion_family"],
                "trajectory": clip_spec["trajectory"],
                "direction": clip_spec["direction"],
                "variation_tag": clip_spec["variation_tag"],
                "radius": clip_spec.get("radius") or cand.get("radius"),
                "start_angle_offset": clip_spec.get("start_angle_offset", 0),
                "cycle_count": clip_spec.get("cycle_count", 1),
                "quality_score": clip_spec.get("quality_score", 0),
                "questions": [{
                    "question": cand["question"],
                    "answer": str(cand["ground_truth"]),
                    "question_type": cand["question_type"],
                    "question_family": cand["question_family"],
                    "dimension": cand.get("dimension"),
                    "anchor_kind": cand["anchor_kind"],
                    "anchor_ids": list(cand["anchor_ids"]),
                    "anchor_labels": list(cand["anchor_labels"]),
                    "primary_object": list(cand["anchor_ids"])[0] if cand["anchor_ids"] else "",
                }],
            })

        all_groups.append({
            "group_id": group_id,
            "engine": cand["engine"],
            "scene_id": cand["scene_id"],
            "room_bucket": cand.get("room_bucket", ""),
            "motion_family": cand["motion_family"],
            "question_type": cand["question_type"],
            "question_family": cand["question_family"],
            "dimension": cand.get("dimension"),
            "radius": cand.get("radius"),
            "anchor_kind": cand["anchor_kind"],
            "anchor_ids": list(cand["anchor_ids"]),
            "anchor_labels": list(cand["anchor_labels"]),
            "shared_ground_truth": str(cand["ground_truth"]),
            "clip_ids": [c["clip_id"] for c in group_clips],
            "target_group_size": len(group_clips),
            "selected_group_size": len(group_clips),
        })

    # Write files
    with (out / "clips.jsonl").open("w") as f:
        for row in all_clips:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out / "consistency_groups.json").write_text(
        json.dumps(all_groups, indent=2, ensure_ascii=False))
    (out / "qa.json").write_text(
        json.dumps(all_qa, indent=2, ensure_ascii=False))

    # Build metadata (needed by render stage)
    config = {
        "output_dir": out_str,
        "enabled_engines": ("thor",),
        "thor_scenes": sorted(set(c["scene_id"] for c in all_clips)),
        "thor_radii": sorted(set(
            c["radius"] for c in all_clips if c.get("radius") is not None
        )),
        "group_size": max(len(clips) for _, clips in selected),
        "min_group_size": 1,
        "gpu": 0,
        "render_parallel_workers": 1,
        "render_share_base_gpu": False,
        "fps": 1,
        "increment": 10.0,
        "random_seed": 42,
        "image_size": 384,
        "exhaust_all_candidates": True,
    }
    metadata = {
        "config": config,
        "groups": all_groups,
        "clips": all_clips,
        "render_records": {},
        "stats": None,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False))

    plan = {"config": config, "groups": all_groups, "clips": all_clips}
    (out / "benchmark_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False))

    # Summary
    qt_counts = Counter(g["question_type"] for g in all_groups)
    gt_diversity = defaultdict(set)
    for g in all_groups:
        gt_diversity[g["question_type"]].add(g["shared_ground_truth"])

    print(f"\n{'='*70}")
    print(f"Plan written to {out}")
    print(f"  Total groups: {len(all_groups)}")
    print(f"  Total clips:  {len(all_clips)}")
    for qt in sorted(qt_counts):
        print(f"  {qt}: {qt_counts[qt]} groups, {len(gt_diversity[qt])} unique GT")
    print(f"\nReady to render:")
    print(f"  bash submit_dataset.sh render --output_dir {out} --engines thor --render_parallel_workers 4 --gpu 0")


def convert_to_jsonl(qa_json_path, jsonl_out_path, video_base=None):
    """Convert qa.json to training JSONL after rendering."""
    with open(qa_json_path) as f:
        data = json.load(f)

    if video_base is None:
        video_base = str(Path(qa_json_path).parent.resolve())

    count = 0
    missing = 0
    with open(jsonl_out_path, "w") as out:
        for clip in data:
            rel = clip["video_path"]
            if rel.startswith("./"):
                rel = rel[2:]
            # Try to resolve video path
            vid_path = os.path.join(os.path.dirname(qa_json_path), "..", rel)
            vid_path = os.path.abspath(vid_path)
            if not os.path.exists(vid_path):
                # Try direct under output dir
                vid_path = os.path.join(video_base, "videos",
                                       clip["clip_id"], "video.mp4")
            if not os.path.exists(vid_path):
                missing += 1
                continue

            for q in clip["questions"]:
                entry = {
                    "video": vid_path,
                    "conversations": [
                        {"from": "human", "value": f"<video>\n{q['question']}"},
                        {"from": "gpt", "value": q["answer"]},
                    ],
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    print(f"Written {count} entries to {jsonl_out_path} ({missing} missing videos)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    # --- select subcommand ---
    sel = sub.add_parser("select", help="Select diverse candidates and write plan files")
    sel.add_argument("--mined_pkl", required=True, help="Path to plan_mined_candidates.pkl")
    sel.add_argument("--output_dir", required=True, help="Output directory for plan files")
    sel.add_argument("--clips_per_qt", type=int, default=600,
                     help="Target clips per QA type")
    sel.add_argument("--clips_per_group", type=int, default=3,
                     help="Clips per candidate group")
    sel.add_argument("--seed", type=int, default=42)

    # --- convert subcommand ---
    conv = sub.add_parser("convert", help="Convert rendered qa.json to training JSONL")
    conv.add_argument("--qa_json", required=True)
    conv.add_argument("--jsonl_out", required=True)

    args = ap.parse_args()

    if args.cmd == "select":
        with open(args.mined_pkl, "rb") as f:
            all_candidates = pickle.load(f)

        thor_cands = all_candidates.get("thor", [])
        print(f"Loaded {len(thor_cands)} THOR candidates")

        selected = select_diverse_candidates(
            thor_cands, args.clips_per_qt, args.clips_per_group, args.seed)
        build_plan_files(selected, args.output_dir)

    elif args.cmd == "convert":
        convert_to_jsonl(args.qa_json, args.jsonl_out)

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
