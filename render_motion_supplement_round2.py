"""
Round 2: render ALL remaining untried approach/passby clips.
Reuses the same output dir and is fully resumable.

Usage:
  python render_motion_supplement_round2.py --gpus 0 1 2 3
"""

import argparse
import json
import os
import sys
import shutil
import traceback
from collections import Counter
from multiprocessing import Process, Queue
from pathlib import Path

ROOT = Path("/nas2/edwin/spatial-scene-variations")
FULL_CLIPS = ROOT / "old" / "video_consistency_thor_full" / "clips.jsonl"
OUTPUT_DIR = ROOT / "video_consistency_thor_motion_supplement"

EVAL_OBJECTS = {
    "ArmChair", "Box", "Chair", "CoffeeTable", "DiningTable",
    "Cabinet", "CoffeeMachine", "Drawer", "Faucet", "Fridge",
}


def find_done_or_failed() -> set[str]:
    """Clip IDs that already have video.mp4 OR were attempted (have a directory but no video)."""
    done = set()
    vid_dir = OUTPUT_DIR / "videos"
    if vid_dir.is_dir():
        for d in vid_dir.iterdir():
            if (d / "video.mp4").exists():
                done.add(d.name)
    # Also count old sources
    for name in [
        "video_consistency_thor_eval_final", "video_consistency_thor_train_expanded",
        "video_consistency_thor_balanced", "video_consistency_thor_train",
        "video_consistency_thor_small",
    ]:
        vd = ROOT / "old" / name / "videos"
        if vd.is_dir():
            done.update(os.listdir(vd))
    return done


def select_remaining() -> list[dict]:
    with open(FULL_CLIPS) as f:
        all_clips = [json.loads(line) for line in f]

    done = find_done_or_failed()

    remaining = []
    for clip in all_clips:
        motion = clip.get("motion_family")
        if motion not in ("approach", "passby"):
            continue
        labels = set(clip.get("anchor_labels", []))
        if not labels or not labels.issubset(EVAL_OBJECTS):
            continue
        if clip["clip_id"] in done:
            continue
        remaining.append(clip)
    return remaining


def shard_clips(clips, n):
    by_scene = {}
    for c in clips:
        by_scene.setdefault(c["scene_id"], []).append(c)
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

    by_scene = {}
    for c in clips:
        by_scene.setdefault(c["scene_id"], []).append(c)

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
                    done += 1; continue
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
                        failed += 1; continue
                    candidate = max(valid, key=lambda c: (c.quality_score, c.saved_frames))
                    if clip_root.exists(): shutil.rmtree(clip_root)
                    clip_root.mkdir(parents=True, exist_ok=True)
                    saved = _render_candidate(controller, candidate, clip_root, output_root, scene_id, config.fps, dedupe_identical_frames=False)
                    done += 1
                    print(f"  [W{worker_id}] OK {cid} ({saved}f) [{done}/{total}]", flush=True)
                except Exception:
                    failed += 1
                    traceback.print_exc()
                    if clip_root.exists(): shutil.rmtree(clip_root, ignore_errors=True)
        except Exception:
            failed += len(scene_clips)
            traceback.print_exc()
        finally:
            if controller:
                try: controller.stop()
                except: pass

    result_queue.put({"worker_id": worker_id, "status": "done", "rendered": done, "failed": failed})


def write_qa_json():
    """Scan all rendered videos in supplement dir and write qa.json."""
    with open(FULL_CLIPS) as f:
        all_clips = {json.loads(l)["clip_id"]: json.loads(l) for l in open(FULL_CLIPS)}

    qa = []
    vid_dir = OUTPUT_DIR / "videos"
    if not vid_dir.is_dir():
        return
    for d in sorted(vid_dir.iterdir()):
        if not (d / "video.mp4").exists():
            continue
        cid = d.name
        if cid not in all_clips:
            continue
        c = all_clips[cid]
        qa.append({
            "clip_id": c["clip_id"], "group_id": c["group_id"],
            "video_path": f"./video_consistency_thor_motion_supplement/videos/{cid}/video.mp4",
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
    print(f"Wrote qa.json: {len(qa)} entries — {dict(mc)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    clips = select_remaining()
    mc = Counter(c["motion_family"] for c in clips)
    print(f"Remaining untried clips: {len(clips)} — {dict(mc)}")
    for m in sorted(mc):
        oc = Counter(l for c in clips if c["motion_family"] == m for l in c["anchor_labels"])
        print(f"  {m}: {mc[m]} — objects: {dict(oc)}")

    if args.dry_run:
        print("[dry_run] Done."); return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "videos").mkdir(exist_ok=True)

    n = len(args.gpus)
    shards = shard_clips(clips, n)
    print(f"Shards: {[len(s) for s in shards]}")

    q = Queue()
    workers = []
    for i, gpu in enumerate(args.gpus):
        p = Process(target=worker_fn, args=(i, gpu, shards[i], q), daemon=True)
        p.start(); workers.append(p)

    finished = 0
    while finished < n:
        msg = q.get()
        if msg["status"] == "done":
            finished += 1
            print(f"[main] Worker {msg['worker_id']}: {msg['rendered']} ok, {msg['failed']} fail", flush=True)

    for p in workers:
        p.join(timeout=30)

    write_qa_json()
    print("\nDone. Rerun build_ood_split.py to rebuild eval/train.")


if __name__ == "__main__":
    main()
