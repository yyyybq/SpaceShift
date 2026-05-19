"""
Render diversity supplement: cam_dist new scenes + bathroom/bedroom pair_dist.
Adds 5 new eval objects: HandTowel, ScrubBrush, Pillow, DeskLamp, AlarmClock

Resumable, multi-GPU.

Usage:
  python render_diversity_supplement.py --gpus 0 1 2 3
  python render_diversity_supplement.py --dry_run
"""

import argparse
import json
import os
import random
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from multiprocessing import Process, Queue
from pathlib import Path

random.seed(42)

ROOT = Path("/nas2/edwin/spatial-scene-variations")
FULL_CLIPS = ROOT / "old" / "video_consistency_thor_full" / "clips.jsonl"
OUTPUT_DIR = ROOT / "video_consistency_thor_diversity_supplement"

EVAL_OBJECTS = {
    "ArmChair", "Box", "Chair", "CoffeeTable", "DiningTable",
    "Cabinet", "CoffeeMachine", "Drawer", "Faucet", "Fridge",
    "Toilet", "Sink", "Mirror", "Bathtub",
    "HousePlant", "GarbageCan", "Dresser", "Laptop", "Painting",
    # New: all-4-room coverage + bathroom + bedroom
    "LightSwitch", "SideTable",       # all 4 rooms
    "HandTowel", "DeskLamp", "Pillow",  # bathroom/bedroom
}


def find_rendered() -> set[str]:
    done = set()
    for rel in [
        "old/video_consistency_thor_train_expanded",
        "old/video_consistency_thor_train",
        "old/video_consistency_thor_small",
        "video_consistency_thor_motion_supplement",
    ]:
        vid_dir = ROOT / rel / "videos"
        if vid_dir.is_dir():
            for d in vid_dir.iterdir():
                if (d / "video.mp4").exists():
                    done.add(d.name)
    vid_dir = OUTPUT_DIR / "videos"
    if vid_dir.is_dir():
        for d in vid_dir.iterdir():
            if (d / "video.mp4").exists():
                done.add(d.name)
    return done


def select_clips() -> list[dict]:
    with open(FULL_CLIPS) as f:
        full = [json.loads(line) for line in f]
    rendered = find_rendered()

    to_render = []

    # 1. cam_dist from ALL scenes (expand scene diversity)
    cam_clips = [c for c in full
                 if c["question_type"] == "object_distance_to_camera"
                 and set(c["anchor_labels"]).issubset(EVAL_OBJECTS)
                 and c["clip_id"] not in rendered]
    by_scene = defaultdict(list)
    for c in cam_clips:
        by_scene[c["scene_id"]].append(c)
    for scene, clips in by_scene.items():
        random.shuffle(clips)
        to_render.extend(clips[:12])

    # 2. bathroom pair_dist (new + existing objects)
    bath_pair = [c for c in full
                 if c["room_bucket"] == "bathroom"
                 and c["question_type"] == "object_pair_distance_center"
                 and set(c["anchor_labels"]).issubset(EVAL_OBJECTS)
                 and c["clip_id"] not in rendered]
    by_scene = defaultdict(list)
    for c in bath_pair:
        by_scene[c["scene_id"]].append(c)
    for scene, clips in by_scene.items():
        random.shuffle(clips)
        to_render.extend(clips[:5])

    # 3. bedroom pair_dist (new + existing objects)
    bed_pair = [c for c in full
                if c["room_bucket"] == "bedroom"
                and c["question_type"] == "object_pair_distance_center"
                and set(c["anchor_labels"]).issubset(EVAL_OBJECTS)
                and c["clip_id"] not in rendered]
    by_scene = defaultdict(list)
    for c in bed_pair:
        by_scene[c["scene_id"]].append(c)
    for scene, clips in by_scene.items():
        random.shuffle(clips)
        to_render.extend(clips[:5])

    # Deduplicate
    seen = set()
    deduped = []
    for c in to_render:
        if c["clip_id"] not in seen:
            seen.add(c["clip_id"])
            deduped.append(c)
    return deduped


def shard_clips(clips, n):
    by_scene = defaultdict(list)
    for c in clips:
        by_scene[c["scene_id"]].append(c)
    shards = [[] for _ in range(n)]
    for _, sc in sorted(by_scene.items(), key=lambda x: -len(x[1])):
        smallest = min(range(n), key=lambda i: len(shards[i]))
        shards[smallest].extend(sc)
    return shards


def worker_fn(worker_id, gpu, clips, result_queue):
    sys.path.insert(0, str(ROOT / "src"))
    from video_consistency_dataset.benchmark_config import BenchmarkConfig
    from video_consistency_dataset.benchmark_variations import variation_from_clip
    from video_consistency_dataset.clip_spec import ClipSpec
    from video_consistency_dataset.thor_render import _thor_config
    from video_qa_dataset.dataset_pipeline import _render_candidate
    from video_qa_dataset.variation_builder import build_varied_candidates
    from trajectory_demos.controller_utils import build_controller, prepare_scene
    from trajectory_demos.selector import _score_candidate

    output_root = OUTPUT_DIR / "videos"
    config = BenchmarkConfig(
        output_dir=str(OUTPUT_DIR), interiorgs_root=None,
        thor_scenes=(), interiorgs_scenes=(), gpu=gpu,
    )

    by_scene = defaultdict(list)
    for c in clips:
        by_scene[c["scene_id"]].append(c)

    done = failed = 0
    total = len(clips)

    for scene_id, scene_clips in by_scene.items():
        print(f"[W{worker_id}|GPU{gpu}] scene {scene_id}: {len(scene_clips)} clips", flush=True)
        controller = None
        try:
            controller = build_controller(scene_id, gpu, config.image_size, config.field_of_view)
            prepare_scene(controller, scene_id)
            for cd in scene_clips:
                cid = cd["clip_id"]
                clip_root = output_root / cid
                video_path = clip_root / "video.mp4"
                if video_path.exists():
                    done += 1
                    continue
                try:
                    clip = ClipSpec(
                        clip_id=cd["clip_id"], group_id=cd["group_id"], engine=cd["engine"],
                        scene_id=cd["scene_id"], room_bucket=cd["room_bucket"],
                        motion_family=cd["motion_family"], trajectory=cd["trajectory"],
                        direction=cd["direction"], variation_tag=cd["variation_tag"],
                        question=cd["question"], ground_truth=cd["ground_truth"],
                        question_type=cd["question_type"], question_family=cd["question_family"],
                        dimension=cd.get("dimension"), radius=cd.get("radius"),
                        start_angle_offset=cd["start_angle_offset"], cycle_count=cd["cycle_count"],
                        anchor_kind=cd["anchor_kind"], anchor_ids=tuple(cd["anchor_ids"]),
                        anchor_labels=tuple(cd["anchor_labels"]),
                        video_path=str(video_path), output_dir=str(clip_root),
                        quality_score=cd.get("quality_score", 0.0),
                    )
                    variation = variation_from_clip(clip)
                    radius = clip.radius if clip.radius is not None else config.linear_radius
                    demo_config = _thor_config(config, scene_id, radius, clip.motion_family)
                    target_ids = list(clip.anchor_ids) if clip.anchor_kind == "object" else None
                    candidates = build_varied_candidates(scene_id, controller, demo_config, clip.trajectory, variation, target_ids)
                    scored = [_score_candidate(controller, c) for c in candidates]
                    valid = [c for c in scored if c.saved_frames >= config.min_frames]
                    if clip.anchor_kind == "object":
                        valid = [c for c in valid if c.object_id == clip.anchor_ids[0]]
                    if not valid:
                        failed += 1
                        continue
                    candidate = max(valid, key=lambda c: (c.quality_score, c.saved_frames))
                    if clip_root.exists():
                        shutil.rmtree(clip_root)
                    clip_root.mkdir(parents=True, exist_ok=True)
                    saved = _render_candidate(controller, candidate, clip_root, output_root, scene_id, config.fps, dedupe_identical_frames=False)
                    done += 1
                    print(f"  [W{worker_id}] OK {cid} ({saved}f) [{done}/{total}]", flush=True)
                except Exception:
                    failed += 1
                    traceback.print_exc()
                    if clip_root.exists():
                        shutil.rmtree(clip_root, ignore_errors=True)
        except Exception:
            failed += len(scene_clips)
            traceback.print_exc()
        finally:
            if controller:
                try:
                    controller.stop()
                except:
                    pass

    result_queue.put({"worker_id": worker_id, "status": "done", "rendered": done, "failed": failed})


def write_qa_json():
    with open(FULL_CLIPS) as f:
        clip_map = {}
        for line in f:
            c = json.loads(line)
            clip_map[c["clip_id"]] = c

    qa = []
    vid_dir = OUTPUT_DIR / "videos"
    if not vid_dir.is_dir():
        return
    for d in sorted(vid_dir.iterdir()):
        if not (d / "video.mp4").exists():
            continue
        cid = d.name
        if cid not in clip_map:
            continue
        c = clip_map[cid]
        qa.append({
            "clip_id": c["clip_id"], "group_id": c["group_id"],
            "video_path": f"./video_consistency_thor_diversity_supplement/videos/{cid}/video.mp4",
            "engine": c["engine"], "scene_id": c["scene_id"],
            "room_bucket": c["room_bucket"], "motion_family": c["motion_family"],
            "trajectory": c["trajectory"], "direction": c["direction"],
            "variation_tag": c["variation_tag"], "radius": c.get("radius"),
            "start_angle_offset": c["start_angle_offset"], "cycle_count": c["cycle_count"],
            "quality_score": c.get("quality_score", 0.0),
            "questions": [{"question": c["question"], "answer": c["ground_truth"],
                           "question_type": c["question_type"], "question_family": c["question_family"],
                           "dimension": c.get("dimension"), "anchor_kind": c["anchor_kind"],
                           "anchor_ids": c["anchor_ids"], "anchor_labels": c["anchor_labels"],
                           "primary_object": c["anchor_ids"][0] if c["anchor_ids"] else None}],
        })
    with open(OUTPUT_DIR / "qa.json", "w") as f:
        json.dump(qa, f, indent=2)
    mc = Counter(e["motion_family"] for e in qa)
    rc = Counter(e["room_bucket"] for e in qa)
    qt = Counter(e["questions"][0]["question_type"] for e in qa)
    print(f"Wrote qa.json: {len(qa)} entries — QA: {dict(qt)}, rooms: {dict(rc)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    clips = select_clips()
    qt_c = Counter(c["question_type"] for c in clips)
    room_c = Counter(c["room_bucket"] for c in clips)
    scene_c = Counter(c["scene_id"] for c in clips)
    obj_c = Counter(c["anchor_labels"][0] for c in clips)
    print(f"{'='*50}")
    print(f"RENDER PLAN: {len(clips)} clips")
    print(f"  QA types: {dict(qt_c)}")
    print(f"  Rooms: {dict(room_c)}")
    print(f"  Scenes: {len(scene_c)} unique")
    print(f"  Workers: {len(args.gpus)} (GPUs: {args.gpus})")
    print(f"  Objects: {dict(obj_c.most_common())}")
    print(f"{'='*50}")

    if args.dry_run:
        print("[dry_run] Done.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "videos").mkdir(exist_ok=True)

    n = len(args.gpus)
    shards = shard_clips(clips, n)
    print(f"Shard sizes: {[len(s) for s in shards]}")

    q = Queue()
    workers = []
    for i, gpu in enumerate(args.gpus):
        p = Process(target=worker_fn, args=(i, gpu, shards[i], q), daemon=True)
        p.start()
        workers.append(p)

    finished = 0
    while finished < n:
        msg = q.get()
        if msg["status"] == "done":
            finished += 1
            print(f"[main] Worker {msg['worker_id']}: {msg['rendered']} ok, {msg['failed']} fail", flush=True)

    for p in workers:
        p.join(timeout=30)

    write_qa_json()
    print("\nDone. Update build_ood_split.py with new source + objects, then rerun.")


if __name__ == "__main__":
    main()
