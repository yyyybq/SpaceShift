#!/usr/bin/env python3
"""Regenerate object_distance_to_camera entries with a uniform distance distribution.

Replaces the concentrated distance-to-camera subsets in both:
  image_consistency_thor_eval_v2/
  video_consistency_thor_eval_v2/

Distance buckets are spaced at 0.2 m increments (one decimal place) across
0.5 .. 5.9 m. For each bucket we mine objects/scenes where a set of
reachable camera positions lands at that exact (rounded) distance from the
object, then render a balanced set of groups.

Pipeline -- every stage is resumable and scene-parallel:
  scan    enumerate per-scene (object, bucket, reachable positions)
  plan    select balanced groups per dataset
  render  render images (static views) and videos (short orbit arcs)
  merge   rewrite qa.json / consistency_groups.json / dataset_stats.json + markdown

Usage:
  PYTHONPATH=src python regenerate_balanced_distance_camera.py all \\
      --work_dir balanced_distance_work \\
      --image_eval_dir image_consistency_thor_eval_v2 \\
      --video_eval_dir video_consistency_thor_eval_v2 \\
      --workers 4 --gpu 0

Stages can also run one at a time (scan / plan / render / merge).

All file writes are idempotent. Re-running a stage resumes from wherever it
left off (scene-level for scan/render, group-level for render outputs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _stable_hash(*parts: str) -> int:
    """Deterministic hash that does not depend on Python's hash randomization."""
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"), digest_size=6)
    return int.from_bytes(h.digest(), "big")

# --- src/ import setup -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Imports from src/ are deferred until worker process start to avoid GPU init
# in the parent process.

# --- constants ---------------------------------------------------------------
DEFAULT_BUCKET_MIN = 0.5
DEFAULT_BUCKET_MAX = 5.9
DEFAULT_BUCKET_STEP = 0.2           # distance-to-camera quantisation step (1 decimal place)

DEFAULT_THOR_SCENES: tuple[str, ...] = tuple(
    [f"FloorPlan{i}" for i in range(1, 31)]
    + [f"FloorPlan{i}" for i in range(201, 231)]
    + [f"FloorPlan{i}" for i in range(301, 331)]
    + [f"FloorPlan{i}" for i in range(401, 431)]
)

DEFAULT_GROUPS_PER_BUCKET = 4       # target; balances ~28 * 4 = 112 groups per dataset
VIEWS_PER_GROUP = 10                # to match existing structure
CLIPS_PER_GROUP = 10                # to match existing structure
MIN_IMAGE_POSITIONS = 3             # minimum reachable positions to seed an image group
MIN_VIDEO_POSITIONS = 5             # minimum reachable positions to seed a video group
MIN_VIEWS_PER_GROUP = 3             # accept groups with at least this many visible views
MIN_CLIPS_PER_GROUP = 3             # accept groups with at least this many visible clips
MIN_FRAMES_PER_CLIP = 5             # short clips are fine for distance QA
MAX_FRAMES_PER_CLIP = 30
VISIBLE_FRAME_RATIO = 0.5           # fraction of a clip's frames that must see the target


def build_buckets(lo: float, hi: float, step: float) -> tuple[float, ...]:
    """Generate distance buckets inclusive of `hi` (with 1 decimal rounding)."""
    n = int(round((hi - lo) / step)) + 1
    return tuple(round(lo + i * step, 1) for i in range(n) if lo + i * step <= hi + 1e-6)

IMAGE_SIZE = 384
FIELD_OF_VIEW = 75
FPS = 1
HORIZON_CLIP = (-45.0, 60.0)
SCAN_GRID_SIZE = 0.1                # dense sampling for scan (AI2-THOR default is 0.25m)

# Anchor-object visibility requirement: target object must project into the frame
MIN_BBOX_AREA_RATIO = 1.0 / 400.0


# --- tiny helpers ------------------------------------------------------------
def _round1(value: float) -> float:
    return round(float(value) + 1e-9, 1)


def _bucket_tag(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _rmtree_nfs_safe(path: Path) -> bool:
    """Best-effort rmtree that tolerates NFS ENOTEMPTY races with brief retries."""
    if not Path(path).exists():
        return True
    for attempt in range(4):
        try:
            shutil.rmtree(path)
            return True
        except OSError:
            time.sleep(0.25 * (attempt + 1))
    # last resort -- swallow error so render can proceed (overwrites files in place)
    shutil.rmtree(path, ignore_errors=True)
    return not Path(path).exists()


def _json_load(path: Path):
    return json.loads(Path(path).read_text())


def _json_dump(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _angle_of(x: float, z: float, obj_x: float, obj_z: float) -> float:
    return math.degrees(math.atan2(z - obj_z, x - obj_x)) % 360.0


def _look_at(cam: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, float]:
    dx = target[0] - cam[0]
    dz = target[2] - cam[2]
    dy = target[1] - cam[1]
    horizontal = max(1e-3, math.hypot(dx, dz))
    yaw = math.degrees(math.atan2(dx, dz))
    horizon = math.degrees(math.atan2(-dy, horizontal))
    return float(yaw), float(np.clip(horizon, *HORIZON_CLIP))


# --- scan stage --------------------------------------------------------------
def _build_dense_controller(scene_id: str, gpu: int):
    """Build an AI2-THOR controller with a denser reachable grid for scanning."""
    import os as _os
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    import global_config
    del gpu
    orig = _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        return Controller(
            scene=scene_id,
            visibilityDistance=global_config.VISIBILITY_DISTANCE,
            renderInstanceSegmentation=True,
            platform=CloudRendering,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            fieldOfView=FIELD_OF_VIEW,
            gridSize=SCAN_GRID_SIZE,
        )
    finally:
        if orig is not None:
            _os.environ["CUDA_VISIBLE_DEVICES"] = orig


def scan_scene(scene_id: str, gpu: int, work_dir: str) -> dict:
    """Mine (object, bucket, reachable positions) candidates in a single scene."""
    scan_path = Path(work_dir) / "scans" / f"{scene_id}.json"
    if scan_path.exists():
        return {"scene": scene_id, "status": "cached"}

    from trajectory_demos.controller_utils import (
        current_eye_y,
        objects_2d,
        prepare_scene,
        reachable_points,
    )
    from video_consistency_dataset.thor_common import room_bucket_for_thor_scene

    controller = _build_dense_controller(scene_id, gpu)
    try:
        prepare_scene(controller, scene_id)
        eye_y = float(current_eye_y(controller))
        reachable = reachable_points(controller)
        objs_meta = list(controller.last_event.metadata["objects"])
        objs = objects_2d(controller)
    finally:
        controller.stop()

    obj_pos = {o["objectId"]: (o["position"]["x"], o["position"]["y"], o["position"]["z"])
               for o in objs_meta if o.get("position")}

    # Capture every reachable position per object, grouped by its round-to-1dp
    # camera-to-object distance. The plan stage decides which buckets to keep.
    result_objects: dict[str, dict] = {}
    for obj in objs:
        obj_id = obj["id"]
        if obj_id not in obj_pos:
            continue
        ox, oy, oz = obj_pos[obj_id]
        buckets: dict[str, list[dict]] = {}
        for p in reachable:
            cx, cz = float(p["x"]), float(p["z"])
            cy = float(p["y"])
            dist = math.sqrt((cx - ox) ** 2 + (eye_y - oy) ** 2 + (cz - oz) ** 2)
            bucket = _round1(dist)
            if bucket < 0.3 or bucket > 8.0:
                continue
            angle = _angle_of(cx, cz, ox, oz)
            buckets.setdefault(f"{bucket:.1f}", []).append({
                "pos": [cx, cy, cz],
                "angle": angle,
                "distance": round(dist, 4),
            })
        buckets = {k: sorted(v, key=lambda r: r["angle"]) for k, v in buckets.items() if v}
        if not buckets:
            continue
        result_objects[obj_id] = {
            "object_type": obj["type"],
            "object_position": [ox, oy, oz],
            "centroid_xz": [float(obj["centroid"][0]), float(obj["centroid"][1])],
            "centroid_height": float(obj["centroid_height"]),
            "obj_radius": float(obj["obj_radius"]),
            "buckets": buckets,
        }

    out = {
        "scene_id": scene_id,
        "room_bucket": room_bucket_for_thor_scene(scene_id),
        "eye_y": eye_y,
        "objects": result_objects,
    }
    _json_dump(scan_path, out)
    n_buckets = sum(len(v["buckets"]) for v in result_objects.values())
    return {"scene": scene_id, "status": "scanned", "objects": len(result_objects), "bucket_entries": n_buckets}


def _scan_worker(args: tuple[str, int, str]) -> dict:
    scene_id, gpu, work_dir = args
    try:
        r = scan_scene(scene_id, gpu, work_dir)
        r["scene"] = scene_id
        return r
    except Exception as exc:  # noqa: BLE001
        return {"scene": scene_id, "status": "error", "error": repr(exc)}


def run_scan_stage(scenes: list[str], work_dir: Path, workers: int, base_gpu: int, share_base_gpu: bool) -> None:
    (work_dir / "scans").mkdir(parents=True, exist_ok=True)
    todo = [s for s in scenes if not (work_dir / "scans" / f"{s}.json").exists()]
    print(f"[scan] scenes total={len(scenes)} pending={len(todo)} workers={workers}", flush=True)
    if not todo:
        return
    task_args = [(s, base_gpu if share_base_gpu else (base_gpu + i % workers), str(work_dir))
                 for i, s in enumerate(todo)]
    if workers == 1:
        for args in task_args:
            t0 = time.monotonic()
            r = _scan_worker(args)
            print(f"[scan] {r['scene']}: {r['status']} ({time.monotonic() - t0:.0f}s)", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=_spawn_ctx()) as pool:
        futures = {pool.submit(_scan_worker, args): args[0] for args in task_args}
        for fut in as_completed(futures):
            r = fut.result()
            print(f"[scan] {r['scene']}: {r['status']} -> {r}", flush=True)


# --- plan stage --------------------------------------------------------------
def _load_all_scans(work_dir: Path) -> list[dict]:
    scans = []
    for path in sorted((work_dir / "scans").glob("*.json")):
        scans.append(_json_load(path))
    return scans


def _candidate_rows(scans: list[dict], min_positions: int, allowed_buckets: set[float]) -> list[dict]:
    rows = []
    for scan in scans:
        scene_id = scan["scene_id"]
        eye_y = scan["eye_y"]
        room_bucket = scan["room_bucket"]
        for obj_id, info in scan["objects"].items():
            for bucket_str, positions in info["buckets"].items():
                bucket_val = _round1(float(bucket_str))
                if bucket_val not in allowed_buckets:
                    continue
                if len(positions) < min_positions:
                    continue
                rows.append({
                    "scene_id": scene_id,
                    "room_bucket": room_bucket,
                    "eye_y": eye_y,
                    "object_id": obj_id,
                    "object_type": info["object_type"],
                    "object_position": info["object_position"],
                    "centroid_xz": info["centroid_xz"],
                    "centroid_height": info["centroid_height"],
                    "bucket": float(bucket_str),
                    "positions": positions,
                })
    return rows


def _balanced_pick(
    rows: list[dict],
    buckets: tuple[float, ...],
    groups_per_bucket: int,
    max_per_scene: int,
    max_per_label: int,
    seed: int,
) -> list[dict]:
    """Greedy round-robin across buckets with scene/label caps."""
    by_bucket: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)
    # deterministic shuffle
    rng = np.random.default_rng(seed)
    for b in by_bucket:
        indices = list(range(len(by_bucket[b])))
        rng.shuffle(indices)
        by_bucket[b] = [by_bucket[b][i] for i in indices]

    scene_counts: Counter = Counter()
    label_counts: Counter = Counter()
    picks: list[dict] = []
    used: set[tuple[str, str, float]] = set()

    def _pass(relax_scene: bool, relax_label: bool) -> None:
        for bucket in buckets:
            have = sum(1 for p in picks if p["bucket"] == bucket)
            remaining = groups_per_bucket - have
            if remaining <= 0:
                continue
            for cand in by_bucket.get(bucket, []):
                if remaining <= 0:
                    break
                key = (cand["scene_id"], cand["object_id"], cand["bucket"])
                if key in used:
                    continue
                if not relax_scene and scene_counts[cand["scene_id"]] >= max_per_scene:
                    continue
                if not relax_label and label_counts[cand["object_type"]] >= max_per_label:
                    continue
                picks.append(cand)
                used.add(key)
                scene_counts[cand["scene_id"]] += 1
                label_counts[cand["object_type"]] += 1
                remaining -= 1

    _pass(False, False)
    _pass(True, False)
    _pass(True, True)
    return picks


def run_plan_stage(
    work_dir: Path,
    buckets: tuple[float, ...],
    groups_per_bucket: int,
    max_per_scene: int,
    max_per_label: int,
    min_image_positions: int,
    min_video_positions: int,
    seed: int,
) -> dict:
    scans = _load_all_scans(work_dir)
    print(f"[plan] loaded {len(scans)} scene scans; buckets={list(buckets)}", flush=True)

    bucket_set = set(buckets)
    image_rows = _candidate_rows(scans, min_image_positions, bucket_set)
    video_rows = _candidate_rows(scans, min_video_positions, bucket_set)
    # supply distribution summary for debuggability
    image_pool = Counter(r["bucket"] for r in image_rows)
    video_pool = Counter(r["bucket"] for r in video_rows)
    print(f"[plan] image candidate rows={len(image_rows)} per-bucket={dict(sorted(image_pool.items()))}", flush=True)
    print(f"[plan] video candidate rows={len(video_rows)} per-bucket={dict(sorted(video_pool.items()))}", flush=True)

    image_plan = _balanced_pick(image_rows, buckets, groups_per_bucket, max_per_scene, max_per_label, seed)
    video_plan = _balanced_pick(video_rows, buckets, groups_per_bucket, max_per_scene, max_per_label, seed + 1)

    # annotate group_ids (unique per dataset, stable across invocations)
    for row in image_plan:
        row["group_id"] = (
            f"thor_static_object_distance_to_camera_{_bucket_tag(row['bucket'])}_"
            f"{_stable_hash(row['scene_id'], row['object_id'], 'img') % 100000}"
        )
        row["kind"] = "image"
    for row in video_plan:
        row["group_id"] = (
            f"thor_around_object_distance_to_camera_{_bucket_tag(row['bucket'])}_"
            f"{_stable_hash(row['scene_id'], row['object_id'], 'vid') % 100000}"
        )
        row["kind"] = "video"

    image_by_bucket = Counter(f"{r['bucket']:.1f}" for r in image_plan)
    video_by_bucket = Counter(f"{r['bucket']:.1f}" for r in video_plan)
    print(f"[plan] image groups by bucket: {dict(sorted(image_by_bucket.items()))}", flush=True)
    print(f"[plan] video groups by bucket: {dict(sorted(video_by_bucket.items()))}", flush=True)

    plan = {
        "config": {
            "buckets": list(buckets),
            "groups_per_bucket": groups_per_bucket,
            "max_per_scene": max_per_scene,
            "max_per_label": max_per_label,
            "min_image_positions": min_image_positions,
            "min_video_positions": min_video_positions,
            "views_per_group": VIEWS_PER_GROUP,
            "clips_per_group": CLIPS_PER_GROUP,
        },
        "image_groups": image_plan,
        "video_groups": video_plan,
    }
    _json_dump(work_dir / "plan.json", plan)
    print(f"[plan] wrote {len(image_plan)} image groups and {len(video_plan)} video groups to plan.json",
          flush=True)
    return plan


# --- render stage ------------------------------------------------------------
def _select_diverse_by_angle(positions: list[dict], count: int) -> list[dict]:
    if count <= 0 or not positions:
        return []
    if len(positions) <= count:
        return list(positions)
    n = len(positions)
    indices = [int(round(i * n / count)) % n for i in range(count)]
    # dedupe while preserving order
    seen: set[int] = set()
    out: list[dict] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(positions[i])
    # top up if we had duplicates
    k = 0
    while len(out) < count and k < n:
        if k not in seen:
            out.append(positions[k])
            seen.add(k)
        k += 1
    return out


def _contiguous_arc_segments(positions: list[dict], gap_deg: float = 25.0) -> list[list[dict]]:
    """Split angle-sorted positions into contiguous arcs (wrap-aware)."""
    if not positions:
        return []
    sorted_pos = sorted(positions, key=lambda r: r["angle"])
    segments: list[list[dict]] = [[sorted_pos[0]]]
    for prev, cur in zip(sorted_pos, sorted_pos[1:]):
        gap = cur["angle"] - prev["angle"]
        if gap > gap_deg:
            segments.append([cur])
        else:
            segments[-1].append(cur)
    # wrap merge
    if len(segments) > 1:
        wrap_gap = (segments[0][0]["angle"] + 360.0) - segments[-1][-1]["angle"]
        if wrap_gap <= gap_deg:
            segments[-1].extend(segments[0])
            segments = segments[1:]
    return segments


def _video_clip_seeds(positions: list[dict], clips: int) -> list[list[dict]]:
    """Carve `clips` contiguous arcs (each MIN_FRAMES_PER_CLIP-MAX_FRAMES_PER_CLIP long)."""
    segments = _contiguous_arc_segments(positions)
    segments = [s for s in segments if len(s) >= MIN_FRAMES_PER_CLIP]
    if not segments:
        return []
    segments.sort(key=len, reverse=True)
    clips_out: list[list[dict]] = []
    # distribute requested clip count proportional to segment length
    total_len = sum(len(s) for s in segments)
    for seg in segments:
        share = max(1, round(clips * len(seg) / total_len))
        seg = list(seg)
        if len(seg) >= MIN_FRAMES_PER_CLIP:
            # pick start indices spaced evenly to produce `share` clips
            max_start = max(0, len(seg) - MIN_FRAMES_PER_CLIP)
            if share == 1:
                starts = [0]
            else:
                starts = [int(round(i * max_start / (share - 1))) for i in range(share)]
                starts = sorted(set(starts))
            for s_idx in starts:
                end = min(len(seg), s_idx + MAX_FRAMES_PER_CLIP)
                sub = seg[s_idx:end]
                if len(sub) >= MIN_FRAMES_PER_CLIP:
                    clips_out.append(sub)
                if len(clips_out) >= clips:
                    break
        if len(clips_out) >= clips:
            break
    return clips_out[:clips]


def _object_visible(controller, object_id: str) -> bool:
    """Check that the target object is currently visible with non-trivial bbox."""
    from trajectory_demos.frame_quality import bbox_area_ratio, object_bbox

    visible = False
    for obj in controller.last_event.metadata["objects"]:
        if obj["objectId"] != object_id:
            continue
        visible = bool(obj.get("visible", False))
        break
    if not visible:
        return False
    bbox = object_bbox(controller, object_id)
    if bbox is None:
        return False
    return bbox_area_ratio(controller, bbox) >= MIN_BBOX_AREA_RATIO


def _render_image_group(
    controller,
    group: dict,
    images_root: Path,
    eval_rel_prefix: str,
) -> tuple[list[dict], dict]:
    """Render VIEWS_PER_GROUP static views; return (qa_entries, group_entry)."""
    from trajectory_demos.controller_utils import teleport_pose
    from trajectory_demos.pose_record import PoseRecord
    from PIL import Image

    positions = group["positions"]
    # over-sample candidates so we can drop occluded ones
    candidates = _select_diverse_by_angle(positions, min(len(positions), VIEWS_PER_GROUP * 3))
    target_pos = tuple(group["object_position"])
    target_id = group["object_id"]
    eye_y = float(group["eye_y"])
    group_id = group["group_id"]
    bucket = group["bucket"]

    qa_entries: list[dict] = []
    image_ids: list[str] = []
    for pos in candidates:
        if len(image_ids) >= VIEWS_PER_GROUP:
            break
        view_idx = len(image_ids)
        image_id = f"{group_id}_view{view_idx:02d}"
        img_dir = images_root / image_id
        img_path = img_dir / "image.png"

        if not img_path.exists():
            cam = (pos["pos"][0], eye_y, pos["pos"][2])
            yaw, horizon = _look_at(cam, target_pos)
            pose = PoseRecord(
                x=pos["pos"][0], y=pos["pos"][1], z=pos["pos"][2],
                yaw=yaw, horizon=horizon,
            )
            if not teleport_pose(controller, pose):
                continue
            if not _object_visible(controller, target_id):
                continue
            img_dir.mkdir(parents=True, exist_ok=True)
            frame = controller.last_event.frame
            Image.fromarray(frame).save(img_path, format="PNG")

        image_ids.append(image_id)
        qa_entries.append(_image_qa_entry(
            image_id=image_id,
            group_id=group_id,
            img_path_rel=f"{eval_rel_prefix}/images/{image_id}/image.png",
            scene_id=group["scene_id"],
            object_id=group["object_id"],
            object_type=group["object_type"],
            distance=bucket,
        ))
    if len(image_ids) < MIN_VIEWS_PER_GROUP:
        # insufficient visible views; drop group
        for image_id in image_ids:
            shutil.rmtree(images_root / image_id, ignore_errors=True)
        return [], _image_group_entry(group, [], bucket)
    group_entry = _image_group_entry(group, image_ids, bucket)
    return qa_entries, group_entry


def _image_qa_entry(
    image_id: str,
    group_id: str,
    img_path_rel: str,
    scene_id: str,
    object_id: str,
    object_type: str,
    distance: float,
) -> dict:
    from video_consistency_dataset.question_entries import build_object_distance_entry
    q = build_object_distance_entry(object_type, distance)
    return {
        "image_id": image_id,
        "group_id": group_id,
        "image_path": img_path_rel,
        "source_clip_id": None,
        "source_frame": None,
        "engine": "thor",
        "scene_id": scene_id,
        "questions": [{
            "question": q["question"],
            "answer": q["ground_truth"],
            "question_type": q["question_type"],
            "question_family": q["question_family"],
            "dimension": None,
            "anchor_kind": "object",
            "anchor_ids": [object_id],
            "anchor_labels": [object_type],
        }],
    }


def _image_group_entry(group: dict, image_ids: list[str], bucket: float) -> dict:
    return {
        "group_id": group["group_id"],
        "engine": "thor",
        "scene_id": group["scene_id"],
        "question_family": "camera_distance",
        "question_type": "object_distance_to_camera",
        "dimension": None,
        "anchor_kind": "object",
        "anchor_ids": [group["object_id"]],
        "anchor_labels": [group["object_type"]],
        "shared_ground_truth": f"{bucket:.1f}",
        "image_ids": image_ids,
        "images_selected": len(image_ids),
    }


def _encode_video(frame_dir: Path, video_path: Path, fps: int) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    assert ffmpeg_path is not None, "ffmpeg not found on PATH."
    subprocess.run(
        [ffmpeg_path, "-y", "-framerate", str(fps),
         "-i", str(frame_dir / "frame_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart",
         str(video_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _render_video_group(
    controller,
    group: dict,
    videos_root: Path,
    eval_rel_prefix: str,
) -> tuple[list[dict], dict]:
    from trajectory_demos.controller_utils import teleport_pose
    from trajectory_demos.pose_record import PoseRecord
    from PIL import Image

    positions = group["positions"]
    clip_seeds = _video_clip_seeds(positions, CLIPS_PER_GROUP)
    target_pos = tuple(group["object_position"])
    target_id = group["object_id"]
    eye_y = float(group["eye_y"])
    group_id = group["group_id"]
    bucket = group["bucket"]

    qa_entries: list[dict] = []
    clip_ids: list[str] = []
    for clip_idx, seed_positions in enumerate(clip_seeds):
        clip_id = f"{group_id}_clip{clip_idx:02d}"
        clip_root = videos_root / clip_id
        frame_dir = clip_root / "frames"
        video_path = clip_root / "video.mp4"

        if not video_path.exists():
            _rmtree_nfs_safe(clip_root)
            frame_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            visible_count = 0
            for pos in seed_positions:
                cam = (pos["pos"][0], eye_y, pos["pos"][2])
                yaw, horizon = _look_at(cam, target_pos)
                pose = PoseRecord(
                    x=pos["pos"][0], y=pos["pos"][1], z=pos["pos"][2],
                    yaw=yaw, horizon=horizon,
                )
                if not teleport_pose(controller, pose):
                    continue
                frame = controller.last_event.frame
                Image.fromarray(frame).save(frame_dir / f"frame_{saved:05d}.png", format="PNG")
                saved += 1
                if _object_visible(controller, target_id):
                    visible_count += 1
            min_visible = max(1, int(VISIBLE_FRAME_RATIO * max(saved, 1)))
            if saved < MIN_FRAMES_PER_CLIP or visible_count < min_visible:
                shutil.rmtree(clip_root, ignore_errors=True)
                continue
            _encode_video(frame_dir, video_path, FPS)

        clip_ids.append(clip_id)
        qa_entries.append(_video_qa_entry(
            clip_id=clip_id,
            group_id=group_id,
            video_path_rel=f"{eval_rel_prefix}/videos/{clip_id}/video.mp4",
            scene_id=group["scene_id"],
            room_bucket=group["room_bucket"],
            object_id=group["object_id"],
            object_type=group["object_type"],
            distance=bucket,
            start_angle=float(seed_positions[0]["angle"]),
        ))

    if len(clip_ids) < MIN_CLIPS_PER_GROUP:
        for clip_id in clip_ids:
            shutil.rmtree(videos_root / clip_id, ignore_errors=True)
        return [], _video_group_entry(group, [], bucket)
    group_entry = _video_group_entry(group, clip_ids, bucket)
    return qa_entries, group_entry


def _video_qa_entry(
    clip_id: str,
    group_id: str,
    video_path_rel: str,
    scene_id: str,
    room_bucket: str,
    object_id: str,
    object_type: str,
    distance: float,
    start_angle: float,
) -> dict:
    from video_consistency_dataset.question_entries import build_object_distance_entry
    q = build_object_distance_entry(object_type, distance)
    return {
        "clip_id": clip_id,
        "group_id": group_id,
        "video_path": video_path_rel,
        "engine": "thor",
        "scene_id": scene_id,
        "room_bucket": room_bucket,
        "motion_family": "around",
        "trajectory": "around_ccw",
        "direction": "ccw",
        "variation_tag": f"dist_{_bucket_tag(distance)}",
        "radius": distance,
        "start_angle_offset": round(start_angle, 2),
        "cycle_count": 1,
        "quality_score": 0.0,
        "questions": [{
            "question": q["question"],
            "answer": q["ground_truth"],
            "question_type": q["question_type"],
            "question_family": q["question_family"],
            "dimension": None,
            "anchor_kind": "object",
            "anchor_ids": [object_id],
            "anchor_labels": [object_type],
            "primary_object": object_id,
        }],
    }


def _video_group_entry(group: dict, clip_ids: list[str], bucket: float) -> dict:
    return {
        "group_id": group["group_id"],
        "engine": "thor",
        "scene_id": group["scene_id"],
        "room_bucket": group["room_bucket"],
        "motion_family": "around",
        "question_family": "camera_distance",
        "question_type": "object_distance_to_camera",
        "dimension": None,
        "anchor_kind": "object",
        "anchor_ids": [group["object_id"]],
        "anchor_labels": [group["object_type"]],
        "shared_ground_truth": f"{bucket:.1f}",
        "clip_ids": clip_ids,
        "selected_group_size": len(clip_ids),
        "available_rendered_clips": len(clip_ids),
        "from_eval_plan": False,
    }


def _group_output_path(work_dir: Path, kind: str, group_id: str) -> Path:
    return work_dir / "groups" / kind / f"{group_id}.json"


def render_scene(
    scene_id: str,
    gpu: int,
    work_dir: str,
    plan_path: str,
    image_eval_dir: str,
    video_eval_dir: str,
) -> dict:
    """Render all planned groups for a single scene."""
    plan = _json_load(plan_path)
    image_scene_groups = [g for g in plan["image_groups"] if g["scene_id"] == scene_id]
    video_scene_groups = [g for g in plan["video_groups"] if g["scene_id"] == scene_id]
    if not image_scene_groups and not video_scene_groups:
        return {"scene": scene_id, "status": "no_groups"}

    image_root = Path(image_eval_dir) / "images"
    video_root = Path(video_eval_dir) / "videos"
    image_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    image_eval_rel_prefix = f"../{Path(image_eval_dir).name}"
    video_eval_rel_prefix = Path(video_eval_dir).name

    # Work dirs to stash per-group QA entries
    img_groups_dir = Path(work_dir) / "groups" / "image"
    vid_groups_dir = Path(work_dir) / "groups" / "video"
    img_groups_dir.mkdir(parents=True, exist_ok=True)
    vid_groups_dir.mkdir(parents=True, exist_ok=True)

    # Skip scene entirely if every group record already exists AND outputs look complete
    def _complete(group: dict, kind: str) -> bool:
        record_path = _group_output_path(Path(work_dir), kind, group["group_id"])
        if not record_path.exists():
            return False
        data = _json_load(record_path)
        if kind == "image":
            for image_id in data["group_entry"]["image_ids"]:
                if not (image_root / image_id / "image.png").exists():
                    return False
        else:
            for clip_id in data["group_entry"]["clip_ids"]:
                if not (video_root / clip_id / "video.mp4").exists():
                    return False
        return True

    image_pending = [g for g in image_scene_groups if not _complete(g, "image")]
    video_pending = [g for g in video_scene_groups if not _complete(g, "video")]
    if not image_pending and not video_pending:
        return {"scene": scene_id, "status": "cached",
                "image_groups": len(image_scene_groups), "video_groups": len(video_scene_groups)}

    from trajectory_demos.controller_utils import build_controller, prepare_scene

    controller = build_controller(scene_id, gpu, IMAGE_SIZE, FIELD_OF_VIEW)
    stats = {"scene": scene_id, "image_rendered": 0, "video_rendered": 0}
    try:
        prepare_scene(controller, scene_id)
        for group in image_pending:
            qa_entries, group_entry = _render_image_group(
                controller, group, image_root, image_eval_rel_prefix,
            )
            if not qa_entries:
                continue
            _json_dump(img_groups_dir / f"{group['group_id']}.json", {
                "qa_entries": qa_entries, "group_entry": group_entry,
            })
            stats["image_rendered"] += 1
        for group in video_pending:
            qa_entries, group_entry = _render_video_group(
                controller, group, video_root, video_eval_rel_prefix,
            )
            if not qa_entries:
                continue
            _json_dump(vid_groups_dir / f"{group['group_id']}.json", {
                "qa_entries": qa_entries, "group_entry": group_entry,
            })
            stats["video_rendered"] += 1
    finally:
        controller.stop()
    return stats


def _render_worker(args: tuple) -> dict:
    scene_id, gpu, work_dir, plan_path, image_eval_dir, video_eval_dir = args
    try:
        return render_scene(scene_id, gpu, work_dir, plan_path, image_eval_dir, video_eval_dir)
    except Exception as exc:  # noqa: BLE001
        return {"scene": scene_id, "status": "error", "error": repr(exc)}


def run_render_stage(
    work_dir: Path,
    image_eval_dir: Path,
    video_eval_dir: Path,
    workers: int,
    base_gpu: int,
    share_base_gpu: bool,
) -> None:
    plan_path = work_dir / "plan.json"
    assert plan_path.exists(), f"Missing plan at {plan_path}. Run `plan` stage first."
    plan = _json_load(plan_path)
    scenes = sorted({g["scene_id"] for g in plan["image_groups"]} | {g["scene_id"] for g in plan["video_groups"]})
    print(f"[render] scenes={len(scenes)} image_groups={len(plan['image_groups'])} video_groups={len(plan['video_groups'])}",
          flush=True)
    task_args = [(scene, base_gpu if share_base_gpu else (base_gpu + i % workers),
                  str(work_dir), str(plan_path), str(image_eval_dir), str(video_eval_dir))
                 for i, scene in enumerate(scenes)]
    if workers == 1:
        for args in task_args:
            t0 = time.monotonic()
            r = _render_worker(args)
            print(f"[render] {r.get('scene')}: {r} ({time.monotonic() - t0:.0f}s)", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=_spawn_ctx()) as pool:
        futures = {pool.submit(_render_worker, args): args[0] for args in task_args}
        for fut in as_completed(futures):
            r = fut.result()
            print(f"[render] {r}", flush=True)


# --- merge stage -------------------------------------------------------------
def _collect_group_records(
    work_dir: Path,
    kind: str,
    plan_group_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load per-group rendered records. If plan_group_ids is given, only keep those."""
    qa_entries: list[dict] = []
    group_entries: list[dict] = []
    root = work_dir / "groups" / kind
    for path in sorted(root.glob("*.json")):
        data = _json_load(path)
        group_id = data["group_entry"]["group_id"]
        if plan_group_ids is not None and group_id not in plan_group_ids:
            continue
        if not data["group_entry"].get("image_ids") and not data["group_entry"].get("clip_ids"):
            continue  # dropped groups (below MIN_VIEWS / MIN_CLIPS threshold)
        qa_entries.extend(data["qa_entries"])
        group_entries.append(data["group_entry"])
    return qa_entries, group_entries


def _water_fill_strict(
    qa_entries: list[dict],
    group_entries: list[dict],
    n_per_group: int,
    target_total: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Strict balanced selection: only groups with exactly n_per_group children.

    Uses water-fill across buckets: each step picks one more full group from
    the least-represented bucket that still has unpicked groups, until
    `target_total` entries are reached (or all buckets exhausted).
    """
    qa_by_group: dict[str, list[dict]] = defaultdict(list)
    for e in qa_entries:
        qa_by_group[e["group_id"]].append(e)

    rng = np.random.default_rng(seed)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for g in group_entries:
        child_key = "image_ids" if "image_ids" in g else "clip_ids"
        if len(g.get(child_key, [])) != n_per_group:
            continue
        by_bucket[g["shared_ground_truth"]].append(g)
    for b in by_bucket:
        idx = list(range(len(by_bucket[b])))
        rng.shuffle(idx)
        by_bucket[b] = [by_bucket[b][i] for i in idx]

    buckets = sorted(by_bucket.keys(), key=lambda x: float(x))
    counts: dict[str, int] = {b: 0 for b in buckets}
    kept_groups: list[dict] = []
    kept_qa: list[dict] = []
    total = 0
    while total + n_per_group <= target_total:
        eligible = [b for b in buckets if counts[b] < len(by_bucket[b])]
        if not eligible:
            break
        next_b = min(eligible, key=lambda b: (counts[b], float(b)))
        g = by_bucket[next_b][counts[next_b]]
        counts[next_b] += 1
        kept_groups.append(g)
        kept_qa.extend(qa_by_group.get(g["group_id"], []))
        total += n_per_group
    return kept_qa, kept_groups


def _trim_per_bucket(
    qa_entries: list[dict],
    group_entries: list[dict],
    answer_field: tuple[str, ...],
    max_per_bucket: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Post-render trimming: cap distance-to-camera entries per answer bucket.

    Accumulates groups in shuffled order. If a group would push the bucket
    past the cap, truncates that group's image_ids / clip_ids to the remaining
    room (group entry stays a valid but smaller group). Subsequent groups for
    that bucket are skipped.
    """
    if max_per_bucket <= 0:
        return qa_entries, group_entries

    qa_by_group: dict[str, list[dict]] = defaultdict(list)
    for e in qa_entries:
        qa_by_group[e["group_id"]].append(e)

    rng = np.random.default_rng(seed)
    bucket_counts: Counter = Counter()
    kept_groups: list[dict] = []
    kept_qa: list[dict] = []

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for g in group_entries:
        by_bucket[g["shared_ground_truth"]].append(g)
    for b in by_bucket:
        order = list(range(len(by_bucket[b])))
        rng.shuffle(order)
        by_bucket[b] = [by_bucket[b][i] for i in order]

    for bucket, groups in by_bucket.items():
        for g in groups:
            room = max_per_bucket - bucket_counts[bucket]
            if room <= 0:
                break
            entries = qa_by_group.get(g["group_id"], [])
            if not entries:
                continue
            if len(entries) <= room:
                kept_groups.append(g)
                kept_qa.extend(entries)
                bucket_counts[bucket] += len(entries)
            else:
                # truncate this group to fit the remaining room
                child_key = "image_ids" if "image_ids" in g else "clip_ids"
                truncated = dict(g)
                # preserve original view/clip ordering by syncing qa entries to group's child_ids
                by_child_id = {
                    (e.get("image_id") or e.get("clip_id")): e for e in entries
                }
                kept_children = list(g.get(child_key, []))[:room]
                truncated[child_key] = kept_children
                if child_key == "image_ids":
                    truncated["images_selected"] = len(kept_children)
                else:
                    truncated["selected_group_size"] = len(kept_children)
                    truncated["available_rendered_clips"] = len(kept_children)
                partial_entries = [by_child_id[c] for c in kept_children if c in by_child_id]
                kept_groups.append(truncated)
                kept_qa.extend(partial_entries)
                bucket_counts[bucket] += len(partial_entries)
                break
    return kept_qa, kept_groups


def _remove_distance_entries(entries: list[dict], kind: str) -> list[dict]:
    """Filter out object_distance_to_camera entries so they can be replaced."""
    out = []
    for entry in entries:
        q = entry["questions"][0] if kind == "qa" else None
        qtype = (q or {}).get("question_type") if kind == "qa" else entry.get("question_type")
        if qtype == "object_distance_to_camera":
            continue
        out.append(entry)
    return out


def _cleanup_old_distance_outputs(
    dataset_dir: Path,
    kind: str,
    existing_groups: list[dict],
    keep_child_ids: set[str] | None = None,
) -> None:
    """Delete image/video folders for old distance_to_camera groups.

    `keep_child_ids` is a whitelist of image_ids/clip_ids that the upcoming
    merge will reference — these are preserved even if they also appear under
    an old distance group. Without this, re-running merge across iterations
    would delete folders that were just rendered and are still in use.
    """
    target_key = "image_ids" if kind == "image" else "clip_ids"
    root = dataset_dir / ("images" if kind == "image" else "videos")
    keep = keep_child_ids or set()
    removed = 0
    for g in existing_groups:
        if g.get("question_type") != "object_distance_to_camera":
            continue
        for child_id in g.get(target_key, []):
            if child_id in keep:
                continue
            child_path = root / child_id
            if child_path.exists():
                shutil.rmtree(child_path, ignore_errors=True)
                removed += 1
    print(f"[merge] removed {removed} stale {kind} dirs under {root}", flush=True)


def _write_image_stats(dataset_dir: Path, qa_entries: list[dict], group_entries: list[dict]) -> None:
    from image_consistency_dataset.stats_report import write_image_dataset_report
    write_image_dataset_report(qa_entries, group_entries, dataset_dir)


def _write_video_stats(dataset_dir: Path, qa_entries: list[dict], group_entries: list[dict]) -> None:
    """Emit a video dataset_stats.json and markdown using the video stats writer.

    The project's `write_dataset_report` expects a BenchmarkPlan, which we don't
    have here (regeneration is partial). Produce a Counter-based stats file that
    mirrors the shape that downstream consumers rely on.
    """
    total_clips = len(qa_entries)
    total_groups = len(group_entries)
    def _ct(counter: Counter, key: str) -> list[dict]:
        return [
            {key: k, "count": v, "share_pct": round(100.0 * v / max(total_clips, 1), 1)}
            for k, v in sorted(counter.items(), key=lambda it: (-it[1], str(it[0])))
        ]
    stats = {
        "total_groups": total_groups,
        "total_videos": total_clips,
        "engine_distribution": _ct(Counter(e["engine"] for e in qa_entries), "engine"),
        "motion_distribution": _ct(Counter(e.get("motion_family", "unknown") for e in qa_entries), "motion_family"),
        "qa_distribution": _ct(Counter(e["questions"][0]["question_type"] for e in qa_entries), "question_type"),
        "question_family_distribution": _ct(Counter(e["questions"][0]["question_family"] for e in qa_entries), "question_family"),
        "radius_distribution": _ct(Counter(e.get("radius") for e in qa_entries if e.get("radius") is not None), "radius"),
        "room_distribution": _ct(Counter(e.get("room_bucket", "unknown") for e in qa_entries), "room_bucket"),
        "object_distribution": _ct(Counter(e["questions"][0]["anchor_labels"][0] for e in qa_entries), "anchor_label"),
        "scene_reuse_distribution": _ct(Counter(e["scene_id"] for e in qa_entries), "scene_id"),
    }
    (dataset_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    # Simple markdown mirroring the original format at a high level
    md = ["# Video Consistency Dataset", "",
          "## Summary", "",
          f"- Total groups: `{total_groups}`",
          f"- Total videos: `{total_clips}`",
          "",
          "## QA Type Distribution", "",
          _md_table(stats["qa_distribution"]),
          "",
          "## Radius Distribution", "",
          _md_table(stats["radius_distribution"]),
          "",
          "## Scene Reuse Distribution", "",
          _md_table(stats["scene_reuse_distribution"]),
          ""]
    (dataset_dir / "video_consistency_dataset.md").write_text("\n".join(md))


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "| value | count |\n|---|---|\n"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(out)


def run_merge_stage(
    work_dir: Path,
    image_eval_dir: Path,
    video_eval_dir: Path,
    cleanup_old_files: bool,
    max_entries_per_bucket: int,
    seed: int,
    strict_target_total: int = 0,
    strict_n_per_group: int = 10,
) -> None:
    plan_path = work_dir / "plan.json"
    plan_image_ids: set[str] | None = None
    plan_video_ids: set[str] | None = None
    if plan_path.exists():
        plan = _json_load(plan_path)
        plan_image_ids = {g["group_id"] for g in plan.get("image_groups", [])}
        plan_video_ids = {g["group_id"] for g in plan.get("video_groups", [])}

    # image: collect the new group records first so we know which child_ids to keep
    new_image_qa, new_image_groups = _collect_group_records(work_dir, "image", plan_image_ids)
    keep_image_children = {cid for g in new_image_groups for cid in g.get("image_ids", [])}
    existing_image_qa = _json_load(image_eval_dir / "qa.json")
    existing_image_groups = _json_load(image_eval_dir / "consistency_groups.json")
    if cleanup_old_files:
        _cleanup_old_distance_outputs(image_eval_dir, "image", existing_image_groups,
                                      keep_child_ids=keep_image_children)
    kept_image_qa = _remove_distance_entries(existing_image_qa, "qa")
    kept_image_groups = _remove_distance_entries(existing_image_groups, "group")
    if strict_target_total > 0:
        before = len(new_image_qa)
        new_image_qa, new_image_groups = _water_fill_strict(
            new_image_qa, new_image_groups, strict_n_per_group, strict_target_total, seed,
        )
        print(f"[merge] image: strict n={strict_n_per_group} water-fill target={strict_target_total}; "
              f"trimmed {before} -> {len(new_image_qa)}", flush=True)
    elif max_entries_per_bucket > 0:
        before = len(new_image_qa)
        new_image_qa, new_image_groups = _trim_per_bucket(
            new_image_qa, new_image_groups, ("answer",), max_entries_per_bucket, seed,
        )
        print(f"[merge] image: per-bucket cap={max_entries_per_bucket}; trimmed {before} -> {len(new_image_qa)}",
              flush=True)
    final_image_qa = kept_image_qa + new_image_qa
    final_image_groups = kept_image_groups + new_image_groups
    _json_dump(image_eval_dir / "qa.json", final_image_qa)
    _json_dump(image_eval_dir / "consistency_groups.json", final_image_groups)
    _write_image_stats(image_eval_dir, final_image_qa, final_image_groups)
    new_img_dists = Counter(
        e["questions"][0]["answer"] for e in new_image_qa
        if e["questions"][0]["question_type"] == "object_distance_to_camera"
    )
    print(f"[merge] image: replaced distance entries. new dist distribution = {dict(sorted(new_img_dists.items()))}",
          flush=True)
    print(f"[merge] image: total_qa={len(final_image_qa)} total_groups={len(final_image_groups)}", flush=True)

    # video
    new_video_qa, new_video_groups = _collect_group_records(work_dir, "video", plan_video_ids)
    keep_video_children = {cid for g in new_video_groups for cid in g.get("clip_ids", [])}
    existing_video_qa = _json_load(video_eval_dir / "qa.json")
    existing_video_groups = _json_load(video_eval_dir / "consistency_groups.json")
    if cleanup_old_files:
        _cleanup_old_distance_outputs(video_eval_dir, "video", existing_video_groups,
                                      keep_child_ids=keep_video_children)
    kept_video_qa = _remove_distance_entries(existing_video_qa, "qa")
    kept_video_groups = _remove_distance_entries(existing_video_groups, "group")
    if strict_target_total > 0:
        before = len(new_video_qa)
        new_video_qa, new_video_groups = _water_fill_strict(
            new_video_qa, new_video_groups, strict_n_per_group, strict_target_total, seed + 1,
        )
        print(f"[merge] video: strict n={strict_n_per_group} water-fill target={strict_target_total}; "
              f"trimmed {before} -> {len(new_video_qa)}", flush=True)
    elif max_entries_per_bucket > 0:
        before = len(new_video_qa)
        new_video_qa, new_video_groups = _trim_per_bucket(
            new_video_qa, new_video_groups, ("answer",), max_entries_per_bucket, seed + 1,
        )
        print(f"[merge] video: per-bucket cap={max_entries_per_bucket}; trimmed {before} -> {len(new_video_qa)}",
              flush=True)
    final_video_qa = kept_video_qa + new_video_qa
    final_video_groups = kept_video_groups + new_video_groups
    _json_dump(video_eval_dir / "qa.json", final_video_qa)
    _json_dump(video_eval_dir / "consistency_groups.json", final_video_groups)
    _write_video_stats(video_eval_dir, final_video_qa, final_video_groups)
    new_vid_dists = Counter(
        e["questions"][0]["answer"] for e in new_video_qa
        if e["questions"][0]["question_type"] == "object_distance_to_camera"
    )
    print(f"[merge] video: replaced distance entries. new dist distribution = {dict(sorted(new_vid_dists.items()))}",
          flush=True)
    print(f"[merge] video: total_qa={len(final_video_qa)} total_groups={len(final_video_groups)}", flush=True)


# --- CLI ---------------------------------------------------------------------
def _spawn_ctx():
    import multiprocessing as mp
    return mp.get_context("spawn")


def _count_distance_totals(image_eval_dir: Path, video_eval_dir: Path) -> tuple[int, int]:
    def _count(dataset_dir: Path) -> int:
        qa = _json_load(dataset_dir / "qa.json")
        return sum(
            1 for e in qa
            if e["questions"][0]["question_type"] == "object_distance_to_camera"
        )
    return _count(image_eval_dir), _count(video_eval_dir)


def iterate_to_target(
    args,
    work_dir: Path,
    image_eval_dir: Path,
    video_eval_dir: Path,
    buckets: tuple[float, ...],
) -> None:
    """Loosen thresholds in steps, re-planning/rendering/merging, until every
    dataset reaches `--target_total` distance entries, or progress stalls.

    Each step reuses the scan cache (no rescan). The plan greedy pick is
    deterministic given --seed, so subsequent iterations are supersets of
    earlier ones (newly loosened buckets add groups on top).
    """
    target = args.target_total
    schedule = [
        # (groups_per_bucket, min_image, min_video, max_per_scene, max_per_label, max_per_bucket)
        (args.groups_per_bucket, args.min_image_positions, args.min_video_positions,
         args.max_per_scene, args.max_per_label, args.max_entries_per_bucket),
        (12, 2, 3, 4, 8, max(args.max_entries_per_bucket, 45)),
        (18, 2, 3, 6, 12, max(args.max_entries_per_bucket, 55)),
        (24, 2, 2, 8, 16, max(args.max_entries_per_bucket, 70)),
        (32, 1, 2, 12, 24, max(args.max_entries_per_bucket, 100)),
    ]
    last_totals: tuple[int, int] | None = None
    for step, (gpb, mi, mv, mps, mpl, mpb) in enumerate(schedule):
        print(
            f"\n[auto-iter {step}] groups_per_bucket={gpb} min_img={mi} min_vid={mv} "
            f"max_per_scene={mps} max_per_label={mpl} max_per_bucket={mpb}",
            flush=True,
        )
        run_plan_stage(
            work_dir, buckets, gpb, mps, mpl, mi, mv, args.seed,
        )
        run_render_stage(
            work_dir, image_eval_dir, video_eval_dir,
            args.workers, args.gpu, args.share_base_gpu,
        )
        run_merge_stage(
            work_dir, image_eval_dir, video_eval_dir,
            cleanup_old_files=not args.no_cleanup_old_files,
            max_entries_per_bucket=mpb, seed=args.seed,
        )
        img_n, vid_n = _count_distance_totals(image_eval_dir, video_eval_dir)
        print(f"[auto-iter {step}] totals: image={img_n} video={vid_n}", flush=True)
        if img_n >= target and vid_n >= target:
            print(f"[auto-iter] reached target={target}", flush=True)
            return
        if last_totals == (img_n, vid_n):
            print(f"[auto-iter] no further progress at step {step}; stopping", flush=True)
            return
        last_totals = (img_n, vid_n)
    print(f"[auto-iter] exhausted schedule; final totals image={img_n} video={vid_n}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("scan", "plan", "render", "merge", "all", "iterate"))
    p.add_argument("--work_dir", required=True, help="Working directory for intermediate artefacts.")
    p.add_argument("--image_eval_dir", default="image_consistency_thor_eval_v2")
    p.add_argument("--video_eval_dir", default="video_consistency_thor_eval_v2")
    p.add_argument("--thor_scenes", nargs="*", default=None, help="Override scene list.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0, help="Base GPU id.")
    p.add_argument("--share_base_gpu", action="store_true",
                   help="All workers use --gpu; default assigns gpu+i to worker i.")
    p.add_argument("--groups_per_bucket", type=int, default=DEFAULT_GROUPS_PER_BUCKET)
    p.add_argument("--max_per_scene", type=int, default=3,
                   help="Cap groups per scene per dataset.")
    p.add_argument("--max_per_label", type=int, default=6,
                   help="Cap groups per object-type label per dataset.")
    p.add_argument("--bucket_min", type=float, default=DEFAULT_BUCKET_MIN)
    p.add_argument("--bucket_max", type=float, default=DEFAULT_BUCKET_MAX)
    p.add_argument("--bucket_step", type=float, default=DEFAULT_BUCKET_STEP,
                   help="Distance step between bucket centers (default 0.2, i.e. 0.5 0.7 0.9 ...).")
    p.add_argument("--min_image_positions", type=int, default=MIN_IMAGE_POSITIONS)
    p.add_argument("--min_video_positions", type=int, default=MIN_VIDEO_POSITIONS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_entries_per_bucket", type=int, default=0,
                   help="Merge: cap distance QA entries per answer bucket (0 = no cap). "
                        "Trims whole groups to keep group integrity.")
    p.add_argument("--target_total", type=int, default=0,
                   help="iterate stage: progressively loosen thresholds until each "
                        "dataset has at least this many distance QA entries (0 disables).")
    p.add_argument("--strict_target_total", type=int, default=0,
                   help="Merge: only keep groups with exactly --strict_n_per_group children, "
                        "then water-fill buckets until total == this many entries. "
                        "Overrides --max_entries_per_bucket when > 0.")
    p.add_argument("--strict_n_per_group", type=int, default=10,
                   help="Required group size for --strict_target_total (default 10).")
    p.add_argument("--no_cleanup_old_files", action="store_true",
                   help="Do not delete image/video folders for replaced distance groups.")
    p.add_argument("--rescan", action="store_true",
                   help="Delete work_dir/scans/* before the scan stage (needed after changing scan logic).")
    p.add_argument("--clean_rendered", action="store_true",
                   help="Delete work_dir/groups/* before rendering (wipes prior group records).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    image_eval_dir = Path(args.image_eval_dir).resolve()
    video_eval_dir = Path(args.video_eval_dir).resolve()
    assert image_eval_dir.is_dir(), f"missing {image_eval_dir}"
    assert video_eval_dir.is_dir(), f"missing {video_eval_dir}"

    scenes = list(args.thor_scenes) if args.thor_scenes else list(DEFAULT_THOR_SCENES)
    buckets = build_buckets(args.bucket_min, args.bucket_max, args.bucket_step)
    print(f"[config] stage={args.stage} work_dir={work_dir} scenes={len(scenes)} workers={args.workers} "
          f"buckets={list(buckets)}",
          flush=True)

    if args.rescan and args.stage in ("scan", "all"):
        scans_dir = work_dir / "scans"
        if scans_dir.exists():
            shutil.rmtree(scans_dir)
            print(f"[config] --rescan: cleared {scans_dir}", flush=True)
    if args.clean_rendered and args.stage in ("render", "all"):
        groups_dir = work_dir / "groups"
        if groups_dir.exists():
            shutil.rmtree(groups_dir)
            print(f"[config] --clean_rendered: cleared {groups_dir}", flush=True)

    if args.stage in ("scan", "all"):
        run_scan_stage(scenes, work_dir, args.workers, args.gpu, args.share_base_gpu)
    if args.stage in ("plan", "all"):
        run_plan_stage(
            work_dir, buckets, args.groups_per_bucket,
            args.max_per_scene, args.max_per_label,
            args.min_image_positions, args.min_video_positions,
            args.seed,
        )
    if args.stage in ("render", "all"):
        run_render_stage(work_dir, image_eval_dir, video_eval_dir,
                         args.workers, args.gpu, args.share_base_gpu)
    if args.stage in ("merge", "all"):
        run_merge_stage(work_dir, image_eval_dir, video_eval_dir,
                        cleanup_old_files=not args.no_cleanup_old_files,
                        max_entries_per_bucket=args.max_entries_per_bucket,
                        seed=args.seed,
                        strict_target_total=args.strict_target_total,
                        strict_n_per_group=args.strict_n_per_group)
    if args.stage == "iterate":
        assert args.target_total > 0, "iterate stage requires --target_total > 0"
        iterate_to_target(args, work_dir, image_eval_dir, video_eval_dir, buckets)
    elif args.stage == "all" and args.target_total > 0:
        # after the first plan/render/merge pass above, keep loosening until target
        iterate_to_target(args, work_dir, image_eval_dir, video_eval_dir, buckets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
