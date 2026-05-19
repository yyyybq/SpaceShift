"""
Render approach/passby clips from thor_full to supplement the OOD eval split.

Data-parallel across GPUs: each worker gets a shard of clips and its own GPU.
Resumable: skips clips that already have video.mp4 on disk.

Usage:
  cd /nas2/edwin/spatial-scene-variations
  python render_motion_supplement.py --gpus 0 1 2 3          # 4 GPU workers
  python render_motion_supplement.py --gpus 0 1 2 3          # resume: just rerun, rendered clips are skipped
  python render_motion_supplement.py --dry_run               # preview only
"""

import argparse
import json
import os
import random
import shutil
import sys
import traceback
from collections import Counter
from multiprocessing import Process, Queue
from pathlib import Path

random.seed(42)

ROOT = Path("/nas2/edwin/spatial-scene-variations")
FULL_CLIPS = ROOT / "old" / "video_consistency_thor_full" / "clips.jsonl"
OUTPUT_DIR = ROOT / "video_consistency_thor_motion_supplement"

EVAL_OBJECTS = {
    "ArmChair", "Box", "Chair", "CoffeeTable", "DiningTable",
    "Cabinet", "CoffeeMachine", "Drawer", "Faucet", "Fridge",
}
TARGET_PER_MOTION = 250
BUFFER_MULTIPLIER = 1.3


def find_already_rendered() -> set[str]:
    rendered = set()
    for name in [
        "video_consistency_thor_eval_final",
        "video_consistency_thor_train_expanded",
        "video_consistency_thor_balanced",
        "video_consistency_thor_train",
        "video_consistency_thor_small",
    ]:
        vid_dir = ROOT / "old" / name / "videos"
        if vid_dir.is_dir():
            rendered.update(os.listdir(vid_dir))
    vid_dir = OUTPUT_DIR / "videos"
    if vid_dir.is_dir():
        for d in vid_dir.iterdir():
            if (d / "video.mp4").exists():
                rendered.add(d.name)
    return rendered


def select_clips_to_render() -> list[dict]:
    with open(FULL_CLIPS) as f:
        all_clips = [json.loads(line) for line in f]

    already = find_already_rendered()

    by_motion: dict[str, list[dict]] = {"approach": [], "passby": []}
    for clip in all_clips:
        motion = clip.get("motion_family")
        if motion not in by_motion:
            continue
        labels = set(clip.get("anchor_labels", []))
        if not labels or not labels.issubset(EVAL_OBJECTS):
            continue
        if clip["clip_id"] in already:
            continue
        by_motion[motion].append(clip)

    target = int(TARGET_PER_MOTION * BUFFER_MULTIPLIER)
    result = []
    for motion in by_motion:
        random.shuffle(by_motion[motion])
        result.extend(by_motion[motion][:target])
    return result


def shard_clips(clips: list[dict], n_workers: int) -> list[list[dict]]:
    """Split clips into n shards, grouping by scene to avoid controller thrashing."""
    by_scene: dict[str, list[dict]] = {}
    for clip in clips:
        by_scene.setdefault(clip["scene_id"], []).append(clip)

    # Distribute scene groups round-robin by size (largest first)
    scenes_sorted = sorted(by_scene.items(), key=lambda x: -len(x[1]))
    shards: list[list[dict]] = [[] for _ in range(n_workers)]
    for _, scene_clips in scenes_sorted:
        # Put in the smallest shard
        smallest = min(range(n_workers), key=lambda i: len(shards[i]))
        shards[smallest].extend(scene_clips)
    return shards


def worker_fn(worker_id: int, gpu: int, clips: list[dict], result_queue: Queue):
    """Single worker: render its shard of clips on the given GPU."""
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
        output_dir=str(OUTPUT_DIR),
        interiorgs_root=None,
        thor_scenes=(),
        interiorgs_scenes=(),
        gpu=gpu,
    )

    by_scene: dict[str, list[dict]] = {}
    for clip in clips:
        by_scene.setdefault(clip["scene_id"], []).append(clip)

    done = 0
    failed = 0
    total = len(clips)

    for scene_id, scene_clips in by_scene.items():
        print(f"[W{worker_id}|GPU{gpu}] scene {scene_id}: {len(scene_clips)} clips", flush=True)
        controller = None
        try:
            controller = build_controller(scene_id, gpu, config.image_size, config.field_of_view)
            prepare_scene(controller, scene_id)

            for clip_dict in scene_clips:
                clip_id = clip_dict["clip_id"]
                clip_root = output_root / clip_id
                video_path = clip_root / "video.mp4"

                # Resume: skip already rendered
                if video_path.exists():
                    result_queue.put({"clip_id": clip_id, "status": "skipped"})
                    done += 1
                    continue

                try:
                    clip = ClipSpec(
                        clip_id=clip_dict["clip_id"],
                        group_id=clip_dict["group_id"],
                        engine=clip_dict["engine"],
                        scene_id=clip_dict["scene_id"],
                        room_bucket=clip_dict["room_bucket"],
                        motion_family=clip_dict["motion_family"],
                        trajectory=clip_dict["trajectory"],
                        direction=clip_dict["direction"],
                        variation_tag=clip_dict["variation_tag"],
                        question=clip_dict["question"],
                        ground_truth=clip_dict["ground_truth"],
                        question_type=clip_dict["question_type"],
                        question_family=clip_dict["question_family"],
                        dimension=clip_dict.get("dimension"),
                        radius=clip_dict.get("radius"),
                        start_angle_offset=clip_dict["start_angle_offset"],
                        cycle_count=clip_dict["cycle_count"],
                        anchor_kind=clip_dict["anchor_kind"],
                        anchor_ids=tuple(clip_dict["anchor_ids"]),
                        anchor_labels=tuple(clip_dict["anchor_labels"]),
                        video_path=str(video_path),
                        output_dir=str(clip_root),
                        quality_score=clip_dict.get("quality_score", 0.0),
                    )

                    variation = variation_from_clip(clip)
                    radius = clip.radius if clip.radius is not None else config.linear_radius
                    demo_config = _thor_config(config, scene_id, radius, clip.motion_family)
                    target_ids = list(clip.anchor_ids) if clip.anchor_kind == "object" else None

                    candidates = build_varied_candidates(
                        scene_id, controller, demo_config, clip.trajectory, variation, target_ids
                    )
                    scored = [_score_candidate(controller, c) for c in candidates]
                    valid = [c for c in scored if c.saved_frames >= config.min_frames]
                    if clip.anchor_kind == "object":
                        valid = [c for c in valid if c.object_id == clip.anchor_ids[0]]

                    if not valid:
                        print(f"  [W{worker_id}] SKIP {clip_id}: no valid candidate", flush=True)
                        result_queue.put({"clip_id": clip_id, "status": "no_candidate"})
                        failed += 1
                        continue

                    candidate = max(valid, key=lambda c: (c.quality_score, c.saved_frames))

                    if clip_root.exists():
                        shutil.rmtree(clip_root)
                    clip_root.mkdir(parents=True, exist_ok=True)

                    saved_frames = _render_candidate(
                        controller, candidate, clip_root, output_root, scene_id, config.fps,
                        dedupe_identical_frames=False,
                    )
                    done += 1
                    result_queue.put({
                        "clip_id": clip_id, "status": "ok",
                        "saved_frames": saved_frames,
                    })
                    print(f"  [W{worker_id}] OK {clip_id} ({saved_frames}f) [{done}/{total}]", flush=True)

                except Exception:
                    failed += 1
                    result_queue.put({"clip_id": clip_id, "status": "error"})
                    print(f"  [W{worker_id}] FAIL {clip_id}:", flush=True)
                    traceback.print_exc()
                    if clip_root.exists():
                        shutil.rmtree(clip_root, ignore_errors=True)

        except Exception:
            print(f"  [W{worker_id}] FAIL scene {scene_id}:", flush=True)
            traceback.print_exc()
            failed += len(scene_clips)
        finally:
            if controller is not None:
                try:
                    controller.stop()
                except Exception:
                    pass

    result_queue.put({"worker_id": worker_id, "status": "done", "rendered": done, "failed": failed})


def write_qa_json(all_candidate_clips: list[dict]) -> None:
    """Write qa.json for clips that have video.mp4 on disk."""
    qa_entries = []
    for clip in all_candidate_clips:
        vid_path = OUTPUT_DIR / "videos" / clip["clip_id"] / "video.mp4"
        if not vid_path.exists():
            continue
        entry = {
            "clip_id": clip["clip_id"],
            "group_id": clip["group_id"],
            "video_path": f"./video_consistency_thor_motion_supplement/videos/{clip['clip_id']}/video.mp4",
            "engine": clip["engine"],
            "scene_id": clip["scene_id"],
            "room_bucket": clip["room_bucket"],
            "motion_family": clip["motion_family"],
            "trajectory": clip["trajectory"],
            "direction": clip["direction"],
            "variation_tag": clip["variation_tag"],
            "radius": clip.get("radius"),
            "start_angle_offset": clip["start_angle_offset"],
            "cycle_count": clip["cycle_count"],
            "quality_score": clip.get("quality_score", 0.0),
            "questions": [{
                "question": clip["question"],
                "answer": clip["ground_truth"],
                "question_type": clip["question_type"],
                "question_family": clip["question_family"],
                "dimension": clip.get("dimension"),
                "anchor_kind": clip["anchor_kind"],
                "anchor_ids": clip["anchor_ids"],
                "anchor_labels": clip["anchor_labels"],
                "primary_object": clip["anchor_ids"][0] if clip["anchor_ids"] else None,
            }],
        }
        qa_entries.append(entry)

    with open(OUTPUT_DIR / "qa.json", "w") as f:
        json.dump(qa_entries, f, indent=2)

    motion_counts = Counter(e["motion_family"] for e in qa_entries)
    print(f"Wrote qa.json: {len(qa_entries)} entries — {dict(motion_counts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU ids for parallel workers")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    clips = select_clips_to_render()

    motion_counts = Counter(c["motion_family"] for c in clips)
    print("=" * 50)
    print(f"Clips to render: {len(clips)}")
    for m, c in sorted(motion_counts.items()):
        obj_c = Counter(l for cl in clips if cl["motion_family"] == m for l in cl.get("anchor_labels", []))
        print(f"  {m}: {c} clips — objects: {dict(obj_c)}")
    print(f"Workers: {len(args.gpus)} (GPUs: {args.gpus})")
    print("=" * 50)

    if args.dry_run:
        print("[dry_run] Done.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "videos").mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "render_manifest.json", "w") as f:
        json.dump(clips, f, indent=2)

    n_workers = len(args.gpus)
    shards = shard_clips(clips, n_workers)

    print(f"\nShard sizes: {[len(s) for s in shards]}")
    for i, s in enumerate(shards):
        scenes = set(c["scene_id"] for c in s)
        print(f"  Worker {i} (GPU {args.gpus[i]}): {len(s)} clips, {len(scenes)} scenes")

    result_queue = Queue()
    workers = []
    for i, gpu in enumerate(args.gpus):
        p = Process(target=worker_fn, args=(i, gpu, shards[i], result_queue), daemon=True)
        p.start()
        workers.append(p)

    # Collect results
    finished = 0
    while finished < n_workers:
        msg = result_queue.get()
        if msg.get("status") == "done":
            finished += 1
            print(f"[main] Worker {msg['worker_id']} finished: {msg['rendered']} rendered, {msg['failed']} failed", flush=True)

    for p in workers:
        p.join(timeout=30)

    # Write qa.json from all clips that ended up with videos on disk
    # Re-read full manifest to include clips from previous runs too
    with open(FULL_CLIPS) as f:
        all_full = [json.loads(line) for line in f]
    eligible = [c for c in all_full
                if c.get("motion_family") in ("approach", "passby")
                and set(c.get("anchor_labels", [])).issubset(EVAL_OBJECTS)
                and c.get("anchor_labels")]
    write_qa_json(eligible)

    print("\nDone. Now rerun build_ood_split.py to rebuild eval_v2/train_v2.")


if __name__ == "__main__":
    main()
