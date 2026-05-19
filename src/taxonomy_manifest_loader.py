"""Load dataset_manifest.json + per-video qa.json with strict QA–video pairing.

Each manifest video row has a target object_id (or \"room\" for rotation). Only
qa.json items whose primary_object matches that id are attached, except rotation
room clips where the whole qa.json is scene-level pair QA for that video.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from taxonomy_demo_data import question_text

_ROTATION_PAIR_TYPES = frozenset(
    {
        "object_pair_distance_center",
        "object_pair_distance_center_w_size",
        "object_distance_comparison_relative",
    }
)


def _repo_root_near_manifest(manifest_path: Path) -> Path:
    cur = manifest_path.resolve().parent
    for _ in range(8):
        if (cur / "src").is_dir():
            return cur
        cur = cur.parent
    return manifest_path.resolve().parents[2]


def resolve_video_path(video_path_str: str, repo_root: Path, manifest_dir: Path) -> Path:
    p = Path(video_path_str).expanduser()
    if p.is_file():
        return p.resolve()
    relative_candidates = [
        manifest_dir / video_path_str,
        repo_root / video_path_str,
    ]
    for candidate in relative_candidates:
        if candidate.is_file():
            return candidate.resolve()
    marker = "spatial-scene-variations/"
    if marker in video_path_str:
        rel = video_path_str.split(marker, 1)[1]
        cand = repo_root / rel
        assert cand.is_file(), f"Video not found after path remap: {cand}"
        return cand.resolve()
    assert p.is_file(), f"Video not found: {video_path_str}"
    return p.resolve()


def _motion_family(trajectory: str) -> str:
    base = trajectory.split("_", 1)[0]
    assert base in (
        "approach",
        "passby",
        "around",
        "spherical",
        "rotation",
    ), f"Unknown trajectory prefix: {trajectory}"
    return base


def _cycle_from_variation(variation: str) -> int | None:
    m = re.search(r"(?:^|_)(\d+)x(?:_|$)", variation or "")
    return int(m.group(1)) if m else None


def _direction_from_trajectory(trajectory: str) -> str:
    parts = trajectory.rsplit("_", 1)
    return parts[1] if len(parts) == 2 else ""


def _engine_from_manifest(scene: str, video: dict, qa: dict) -> str:
    engine = str(video.get("engine", "")).strip().lower()
    if engine:
        return engine
    source = str(qa.get("source", video.get("source", ""))).strip().lower()
    if source == "interiorgs":
        return "interiorgs"
    if scene.startswith("FloorPlan"):
        return "thor"
    if re.match(r"^\d{4}_\d+$", scene):
        return "interiorgs"
    return ""


def _qa_applies_to_video(qa: dict, object_id: str, trajectory: str) -> bool:
    qtype = qa.get("question_type", "")
    primary = qa.get("primary_object", "")
    if object_id == "room" and _motion_family(trajectory) == "rotation":
        return qtype in _ROTATION_PAIR_TYPES or qtype.startswith("object_distance_comparison")
    if not primary:
        return False
    return primary == object_id


def _entry_from_pair(scene: str, video: dict, qa: dict, video_path: Path) -> dict:
    traj = video["trajectory"]
    var = video.get("variation", "")
    oid = video.get("object_id", "")
    anchor_label = video.get("object_type") or oid
    return {
        "qa_key": f"{scene}|{qa.get('question_id', '')}",
        "group_id": qa.get("question_id", ""),
        "clip_id": f"{traj}_{video.get('object_type', 'obj')}_{var}",
        "video_path": str(video_path),
        "question": qa["question"],
        "ground_truth": str(qa.get("answer", "")),
        "question_type": qa.get("question_type", ""),
        "engine": _engine_from_manifest(scene, video, qa),
        "scene_id": scene,
        "motion_family": _motion_family(traj),
        "trajectory": traj,
        "direction": _direction_from_trajectory(traj),
        "variation_tag": var,
        "radius": video.get("radius"),
        "start_angle_offset": video.get("start_angle_offset"),
        "cycle_count": _cycle_from_variation(var),
        "anchor_labels": [anchor_label] if anchor_label else [],
    }


def load_manifest_paired_entries(manifest_path: str | Path) -> list[dict]:
    mp = Path(manifest_path).expanduser().resolve()
    assert mp.is_file(), f"Missing manifest: {mp}"
    repo_root = _repo_root_near_manifest(mp)
    manifest_dir = mp.parent
    payload = json.loads(mp.read_text(encoding="utf-8"))
    entries: list[dict] = []
    for block in payload.get("scenes") or []:
        scene = block["scene"]
        for video in block.get("videos") or []:
            vpath = resolve_video_path(video["video_path"], repo_root, manifest_dir)
            qa_path = vpath.parent / "qa.json"
            assert qa_path.is_file(), f"Missing qa.json for {vpath}"
            qas = json.loads(qa_path.read_text(encoding="utf-8"))
            assert isinstance(qas, list), f"qa.json must be a list: {qa_path}"
            oid = video.get("object_id", "")
            traj = video["trajectory"]
            for qa in qas:
                if not isinstance(qa, dict):
                    continue
                if not _qa_applies_to_video(qa, oid, traj):
                    continue
                entries.append(_entry_from_pair(scene, video, qa, vpath))
    assert entries, f"No paired QA entries under manifest {mp}"
    return entries


def load_multi_video_groups_from_manifest(
    manifest_path: str | Path,
    min_videos: int = 1,
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in load_manifest_paired_entries(manifest_path):
        groups[entry["qa_key"]].append(entry)
    filtered = {
        key: sorted(
            value,
            key=lambda e: (
                e["motion_family"],
                e["trajectory"],
                e["cycle_count"] is None,
                e["cycle_count"] or -1,
                e["start_angle_offset"] if e["start_angle_offset"] is not None else -1,
                e["variation_tag"],
                e["clip_id"],
            ),
        )
        for key, value in groups.items()
        if len(value) >= min_videos
    }
    assert filtered, (
        f"No QA groups with at least {min_videos} video(s) after strict pairing. "
        f"Try --min_videos 1 or check manifest paths / qa.json primary_object."
    )
    return filtered
