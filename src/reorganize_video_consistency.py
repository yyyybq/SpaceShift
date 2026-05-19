"""Reorganize the Thor video-consistency benchmark into balanced 10-clip groups.

This version is intended for metric computation where:
  - each consistency group is one QA pair with 10 videos
  - overall CV is the average of intra-group CVs
  - MRA is computed globally across all 10n responses

Selection policy:
  - family-balanced: 100 camera_distance, 100 pair_distance, 100 size
  - source rendered clips only from `old/video_consistency_thor_*/videos`
  - maximize scene / anchor-label diversity subject to the family quotas
  - for pair_distance, use every non-FloorPlan1 group with >=10 renders before
    filling the remaining quota from FloorPlan1
  - materialize local clip directories in the output rather than symlinking

Usage:
  python src/reorganize_video_consistency.py \
    --eval_plan old/video_consistency_thor_eval/metadata.json \
    --full_plan old/video_consistency_thor_full/metadata.json \
    --output_dir video_consistency_thor_eval_v2 \
    --clips_per_group 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_FAMILY_TARGETS = {
    "camera_distance": 100,
    "pair_distance": 100,
    "size": 100,
}


@dataclass(frozen=True)
class CandidateGroup:
    group: dict
    available_clip_ids: tuple[str, ...]
    label_key: str
    anchor_key: str
    atomic_labels: tuple[str, ...]
    from_eval_plan: bool

    @property
    def family(self) -> str:
        return str(self.group["question_family"])

    @property
    def group_id(self) -> str:
        return str(self.group["group_id"])

    @property
    def scene_id(self) -> str:
        return str(self.group["scene_id"])

    @property
    def motion_family(self) -> str:
        return str(self.group.get("motion_family", "unknown"))

    @property
    def dimension(self) -> str:
        dim = self.group.get("dimension")
        return "none" if dim in (None, "") else str(dim)

    @property
    def room_bucket(self) -> str:
        room = self.group.get("room_bucket")
        return "unknown" if room in (None, "") else str(room)

    @property
    def ground_truth(self) -> str:
        return str(self.group.get("shared_ground_truth", ""))

    @property
    def available_count(self) -> int:
        return len(self.available_clip_ids)


def _parse_targets(raw: str | None) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_FAMILY_TARGETS)
    out = dict(DEFAULT_FAMILY_TARGETS)
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        family, count = piece.split("=", 1)
        out[family.strip()] = int(count)
    return out


def _parse_label_caps(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    out: dict[str, int] = {}
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        label, count = piece.split("=", 1)
        out[label.strip()] = int(count)
    return out


def _parse_scene_caps(raw: str | None) -> dict[str, int]:
    return _parse_label_caps(raw)


def _fair_targets(capacities: Counter, total: int) -> dict[str, int]:
    if total <= 0:
        return {str(key): 0 for key in capacities}
    items = [(str(key), int(value)) for key, value in capacities.items() if int(value) > 0]
    if not items:
        return {}
    items.sort(key=lambda kv: (kv[1], kv[0]))
    targets = {key: 0 for key, _ in items}
    remaining_total = total
    active = list(items)
    while active and remaining_total > 0:
        level = remaining_total / len(active)
        smallest_key, smallest_cap = active[0]
        if level <= smallest_cap:
            base = int(level)
            for key, cap in active:
                targets[key] = min(cap, base)
            remaining_total -= sum(targets[key] for key, _ in active)
            break
        targets[smallest_key] = smallest_cap
        remaining_total -= smallest_cap
        active.pop(0)
    if active and remaining_total > 0:
        candidates = sorted(active, key=lambda kv: (targets[kv[0]], kv[1], kv[0]))
        while remaining_total > 0:
            moved = False
            for key, cap in candidates:
                if remaining_total <= 0:
                    break
                if targets[key] < cap:
                    targets[key] += 1
                    remaining_total -= 1
                    moved = True
            if not moved:
                break
    return targets


def _discover_render_dirs() -> list[Path]:
    root = Path.cwd()
    seen: set[Path] = set()
    render_dirs: list[Path] = []
    for path in sorted(root.glob("old/video_consistency_thor_*/videos")):
        resolved = path.resolve()
        if not path.is_dir() or resolved in seen:
            continue
        seen.add(resolved)
        render_dirs.append(path)
    return render_dirs


def _render_index(render_dirs: Iterable[Path]) -> tuple[set[str], dict[str, Path]]:
    all_rendered: set[str] = set()
    source_by_clip: dict[str, Path] = {}
    for render_dir in render_dirs:
        try:
            children = sorted(os.listdir(render_dir))
        except FileNotFoundError:
            continue
        for clip_id in children:
            clip_path = render_dir / clip_id
            if not clip_path.exists():
                continue
            all_rendered.add(clip_id)
            source_by_clip.setdefault(clip_id, clip_path)
    return all_rendered, source_by_clip


def _label_key(anchor_labels: Iterable[str], question_type: str) -> str:
    labels = [str(x).split("|")[0].strip() for x in (anchor_labels or []) if str(x).strip()]
    if not labels:
        return "unknown"
    if question_type == "object_pair_distance_center" and len(labels) >= 2:
        return "–".join(sorted(labels[:2]))
    return labels[0]


def _atomic_labels(anchor_labels: Iterable[str]) -> tuple[str, ...]:
    labels = [str(x).split("|")[0].strip() for x in (anchor_labels or []) if str(x).strip()]
    return tuple(sorted(dict.fromkeys(labels)))


def _anchor_key(anchor_ids: Iterable[str], question_type: str) -> str:
    ids = [str(x).strip() for x in (anchor_ids or []) if str(x).strip()]
    if not ids:
        return "unknown"
    if question_type == "object_pair_distance_center" and len(ids) >= 2:
        return "||".join(sorted(ids[:2]))
    return ids[0]


def _candidate_groups(
    full_plan: dict,
    rendered_clip_ids: set[str],
    eval_group_ids: set[str],
    clips_per_group: int,
) -> dict[str, list[CandidateGroup]]:
    out: dict[str, list[CandidateGroup]] = defaultdict(list)
    for raw_group in full_plan["groups"]:
        avail = tuple(cid for cid in raw_group["clip_ids"] if cid in rendered_clip_ids)
        if len(avail) < clips_per_group:
            continue
        question_type = str(raw_group["question_type"])
        out[str(raw_group["question_family"])].append(
            CandidateGroup(
                group=raw_group,
                available_clip_ids=avail,
                label_key=_label_key(raw_group.get("anchor_labels", []), question_type),
                anchor_key=_anchor_key(raw_group.get("anchor_ids", []), question_type),
                atomic_labels=_atomic_labels(raw_group.get("anchor_labels", [])),
                from_eval_plan=raw_group["group_id"] in eval_group_ids,
            )
        )
    return out


def _pick_groups(
    candidates: list[CandidateGroup],
    target: int,
    rng: random.Random,
    global_scene_counts: Counter,
    global_label_counts: Counter,
    global_room_counts: Counter,
    global_atomic_label_counts: Counter,
    global_atomic_label_caps: dict[str, int] | None = None,
    global_scene_caps: dict[str, int] | None = None,
    family_scene_counts: Counter | None = None,
    family_label_counts: Counter | None = None,
    family_label_targets: dict[str, int] | None = None,
    family_motion_counts: Counter | None = None,
    family_dim_counts: Counter | None = None,
    family_dim_targets: dict[str, int] | None = None,
    family_gt_counts: Counter | None = None,
    family_room_counts: Counter | None = None,
) -> list[CandidateGroup]:
    if family_scene_counts is None:
        family_scene_counts = Counter()
    if family_label_counts is None:
        family_label_counts = Counter()
    if family_motion_counts is None:
        family_motion_counts = Counter()
    if family_dim_counts is None:
        family_dim_counts = Counter()
    if family_gt_counts is None:
        family_gt_counts = Counter()
    if family_room_counts is None:
        family_room_counts = Counter()

    remaining = list(candidates)
    rng.shuffle(remaining)
    selected: list[CandidateGroup] = []
    local_anchor_counts: Counter = Counter()
    local_scene_label_counts: Counter = Counter()

    while len(selected) < target and remaining:
        eligible = []
        for cand in remaining:
            if global_scene_caps:
                scene_cap = global_scene_caps.get(cand.scene_id)
                if scene_cap is not None and global_scene_counts[cand.scene_id] >= scene_cap:
                    continue
            if global_atomic_label_caps:
                over = False
                for label in cand.atomic_labels:
                    cap = global_atomic_label_caps.get(label)
                    if cap is not None and global_atomic_label_counts[label] >= cap:
                        over = True
                        break
                if over:
                    continue
            eligible.append(cand)
        if not eligible:
            break

        def score(cand: CandidateGroup) -> tuple:
            label_fill = 0.0
            label_over_target = 0
            if family_label_targets:
                label_target = max(1, family_label_targets.get(cand.label_key, 1))
                label_fill = family_label_counts[cand.label_key] / label_target
                other_labels_underfilled = any(
                    family_label_counts[label] < target_value
                    for label, target_value in family_label_targets.items()
                )
                if other_labels_underfilled and family_label_counts[cand.label_key] >= family_label_targets.get(cand.label_key, 0):
                    label_over_target = 1
            dim_fill = 0.0
            dim_over_target = 0
            if family_dim_targets:
                dim_target = max(1, family_dim_targets.get(cand.dimension, 1))
                dim_fill = family_dim_counts[cand.dimension] / dim_target
                other_dims_underfilled = any(
                    family_dim_counts[dim] < dim_target_val
                    for dim, dim_target_val in family_dim_targets.items()
                )
                if other_dims_underfilled and family_dim_counts[cand.dimension] >= family_dim_targets.get(cand.dimension, 0):
                    dim_over_target = 1
            return (
                label_over_target,
                dim_over_target,
                global_scene_counts[cand.scene_id],
                family_scene_counts[cand.scene_id],
                label_fill,
                dim_fill,
                global_label_counts[cand.label_key],
                family_label_counts[cand.label_key],
                family_motion_counts[cand.motion_family],
                family_gt_counts[cand.ground_truth],
                global_room_counts[cand.room_bucket],
                family_room_counts[cand.room_bucket],
                local_scene_label_counts[(cand.scene_id, cand.label_key)],
                local_anchor_counts[cand.anchor_key],
                0 if cand.from_eval_plan else 1,
                -cand.available_count,
                cand.group_id,
            )

        chosen = min(eligible, key=score)
        remaining.remove(chosen)
        selected.append(chosen)

        global_scene_counts[chosen.scene_id] += 1
        global_label_counts[chosen.label_key] += 1
        global_room_counts[chosen.room_bucket] += 1
        for label in chosen.atomic_labels:
            global_atomic_label_counts[label] += 1
        family_scene_counts[chosen.scene_id] += 1
        family_label_counts[chosen.label_key] += 1
        family_motion_counts[chosen.motion_family] += 1
        family_dim_counts[chosen.dimension] += 1
        family_gt_counts[chosen.ground_truth] += 1
        family_room_counts[chosen.room_bucket] += 1
        local_anchor_counts[chosen.anchor_key] += 1
        local_scene_label_counts[(chosen.scene_id, chosen.label_key)] += 1

    return selected


def _select_family_balanced(
    candidates_by_family: dict[str, list[CandidateGroup]],
    family_targets: dict[str, int],
    rng: random.Random,
    global_atomic_label_caps: dict[str, int] | None = None,
    global_scene_caps: dict[str, int] | None = None,
) -> list[CandidateGroup]:
    global_scene_counts: Counter = Counter()
    global_label_counts: Counter = Counter()
    global_room_counts: Counter = Counter()
    global_atomic_label_counts: Counter = Counter()

    selected: list[CandidateGroup] = []

    # pair_distance is the limiting family. Use every non-FloorPlan1 candidate first.
    pair_target = family_targets["pair_distance"]
    pair_candidates = list(candidates_by_family["pair_distance"])
    pair_non_fp1 = [cand for cand in pair_candidates if cand.scene_id != "FloorPlan1"]
    pair_fp1 = [cand for cand in pair_candidates if cand.scene_id == "FloorPlan1"]

    pair_family_scene = Counter()
    pair_family_label = Counter()
    pair_family_motion = Counter()
    pair_family_dim = Counter()
    pair_family_gt = Counter()
    pair_family_room = Counter()

    non_fp1_target = min(pair_target, len(pair_non_fp1))
    selected.extend(
        _pick_groups(
            pair_non_fp1,
            non_fp1_target,
            rng,
            global_scene_counts,
            global_label_counts,
            global_room_counts,
            global_atomic_label_counts,
            global_atomic_label_caps,
            global_scene_caps,
            pair_family_scene,
            pair_family_label,
            None,
            pair_family_motion,
            pair_family_dim,
            None,
            pair_family_gt,
            pair_family_room,
        )
    )
    if len(selected) < pair_target:
        selected.extend(
            _pick_groups(
                pair_fp1,
                pair_target - len(selected),
                rng,
                global_scene_counts,
                global_label_counts,
                global_room_counts,
                global_atomic_label_counts,
                global_atomic_label_caps,
                global_scene_caps,
                pair_family_scene,
                pair_family_label,
                None,
                pair_family_motion,
                pair_family_dim,
                None,
                pair_family_gt,
                pair_family_room,
            )
        )

    if len([cand for cand in selected if cand.family == "pair_distance"]) != pair_target:
        raise ValueError("Unable to satisfy pair_distance target with currently rendered clips.")

    for family in ("size", "camera_distance"):
        target = family_targets[family]
        family_candidates = list(candidates_by_family[family])
        family_scene = Counter()
        family_label = Counter()
        family_motion = Counter()
        family_dim = Counter()
        family_gt = Counter()
        family_room = Counter()
        family_label_targets = _fair_targets(Counter(cand.label_key for cand in family_candidates), target)
        family_dim_targets = None
        if family == "size":
            dims = Counter(cand.dimension for cand in family_candidates)
            active_dims = sorted(dim for dim in dims if dim != "none")
            base = target // len(active_dims)
            remainder = target % len(active_dims)
            family_dim_targets = {
                dim: base + (1 if idx < remainder else 0)
                for idx, dim in enumerate(active_dims)
            }
        motion_caps = Counter(cand.motion_family for cand in family_candidates)
        motion_targets = _fair_targets(motion_caps, target)
        motion_order = sorted(
            motion_targets,
            key=lambda motion: (motion_targets[motion], motion_caps[motion], motion),
        )
        family_selected_before = len(selected)
        for motion in motion_order:
            motion_target = motion_targets[motion]
            if motion_target <= 0:
                continue
            motion_candidates = [cand for cand in family_candidates if cand.motion_family == motion]
            selected.extend(
                _pick_groups(
                    motion_candidates,
                    motion_target,
                    rng,
                    global_scene_counts,
                    global_label_counts,
                    global_room_counts,
                    global_atomic_label_counts,
                    global_atomic_label_caps,
                    global_scene_caps,
                    family_scene,
                    family_label,
                    family_label_targets,
                    family_motion,
                    family_dim,
                    family_dim_targets,
                    family_gt,
                    family_room,
                )
            )
        family_selected_count = sum(cand.family == family for cand in selected[family_selected_before:])
        if family_selected_count != target:
            raise ValueError(f"Unable to satisfy target for {family}: expected {target}, got {family_selected_count}")

    family_counts = Counter(cand.family for cand in selected)
    for family, target in family_targets.items():
        if family_counts[family] != target:
            raise ValueError(
                f"Family target not met for {family}: expected {target}, got {family_counts[family]}"
            )
    return sorted(selected, key=lambda cand: (cand.family, cand.scene_id, cand.group_id))


def _select_diverse_clips(
    clip_ids: Iterable[str],
    clips_by_id: dict[str, dict],
    n: int,
    rng: random.Random,
) -> list[str]:
    available = [clips_by_id[cid] for cid in clip_ids]
    if len(available) <= n:
        return [clip["clip_id"] for clip in available]

    remaining = list(available)
    rng.shuffle(remaining)
    selected: list[str] = []
    direction_counts: Counter = Counter()
    trajectory_counts: Counter = Counter()
    variation_counts: Counter = Counter()
    cycle_counts: Counter = Counter()
    angle_counts: Counter = Counter()

    while len(selected) < n and remaining:
        def score(clip: dict) -> tuple:
            return (
                direction_counts[str(clip.get("direction", ""))],
                trajectory_counts[str(clip.get("trajectory", ""))],
                cycle_counts[int(clip.get("cycle_count", 0) or 0)],
                angle_counts[int(clip.get("start_angle_offset", 0) or 0)],
                variation_counts[str(clip.get("variation_tag", ""))],
                -float(clip.get("quality_score", 0.0) or 0.0),
                str(clip["clip_id"]),
            )

        best_idx = min(range(len(remaining)), key=lambda idx: score(remaining[idx]))
        clip = remaining.pop(best_idx)
        selected.append(str(clip["clip_id"]))
        direction_counts[str(clip.get("direction", ""))] += 1
        trajectory_counts[str(clip.get("trajectory", ""))] += 1
        cycle_counts[int(clip.get("cycle_count", 0) or 0)] += 1
        angle_counts[int(clip.get("start_angle_offset", 0) or 0)] += 1
        variation_counts[str(clip.get("variation_tag", ""))] += 1
    return selected


def _clear_output_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _materialize_clip_dir(source: Path, target: Path, mode: str) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source clip directory: {source}")
    if target.exists() or target.is_symlink():
        _clear_output_path(target)
    target.mkdir(parents=True, exist_ok=True)
    if mode == "copy_dir":
        shutil.copytree(source, target, dirs_exist_ok=True)
        return
    if mode != "copy_mp4":
        raise ValueError(f"Unsupported video mode: {mode}")
    source_video = source / "video.mp4"
    if not source_video.exists():
        raise FileNotFoundError(f"Missing source video: {source_video}")
    shutil.copy2(source_video, target / "video.mp4")


def _build_stats(groups: list[dict], qa_entries: list[dict], render_dirs: list[Path], selection_report: dict) -> dict:
    family_counts = Counter(group["question_family"] for group in groups)
    scene_counts = Counter(group["scene_id"] for group in groups)
    label_counts = Counter(
        _label_key(group.get("anchor_labels", []), str(group["question_type"]))
        for group in groups
    )
    atomic_label_counts = Counter(
        str(label).split("|")[0].strip()
        for group in groups
        for label in (group.get("anchor_labels") or [])
        if str(label).strip()
    )
    room_counts = Counter(group.get("room_bucket") or "unknown" for group in groups)
    motion_counts = Counter(group.get("motion_family") or "unknown" for group in groups)
    dim_counts = Counter((group.get("dimension") or "none") for group in groups)
    clips_per_group = Counter(len(group["clip_ids"]) for group in groups)

    family_stats: dict[str, dict] = {}
    for family in sorted(family_counts):
        fam_groups = [group for group in groups if group["question_family"] == family]
        fam_scene = Counter(group["scene_id"] for group in fam_groups)
        fam_label = Counter(
            _label_key(group.get("anchor_labels", []), str(group["question_type"]))
            for group in fam_groups
        )
        fam_atomic = Counter(
            str(label).split("|")[0].strip()
            for group in fam_groups
            for label in (group.get("anchor_labels") or [])
            if str(label).strip()
        )
        family_stats[family] = {
            "groups": len(fam_groups),
            "unique_scenes": len(fam_scene),
            "unique_anchor_signatures": len(fam_label),
            "unique_atomic_labels": len(fam_atomic),
            "motion_family_distribution": dict(sorted(Counter(group["motion_family"] for group in fam_groups).items())),
            "dimension_distribution": dict(sorted(Counter((group.get("dimension") or "none") for group in fam_groups).items())),
            "room_bucket_distribution": dict(sorted(Counter((group.get("room_bucket") or "unknown") for group in fam_groups).items())),
            "top_scenes": fam_scene.most_common(15),
            "top_anchor_signatures": fam_label.most_common(15),
            "top_atomic_labels": fam_atomic.most_common(15),
        }

    return {
        "total_groups": len(groups),
        "total_clips": len(qa_entries),
        "clips_per_group_distribution": dict(sorted(clips_per_group.items())),
        "question_family_distribution": dict(sorted(family_counts.items())),
        "unique_scenes": len(scene_counts),
        "unique_anchor_signatures": len(label_counts),
        "unique_atomic_labels": len(atomic_label_counts),
        "motion_family_distribution": dict(sorted(motion_counts.items())),
        "dimension_distribution": dict(sorted(dim_counts.items())),
        "room_bucket_distribution": dict(sorted(room_counts.items())),
        "top_scenes": scene_counts.most_common(20),
        "top_anchor_signatures": label_counts.most_common(20),
        "top_atomic_labels": atomic_label_counts.most_common(20),
        "family_stats": family_stats,
        "selection_report": selection_report,
        "render_dirs": [str(path) for path in render_dirs],
    }


def _build_report(stats: dict) -> str:
    lines = [
        "# Video Consistency Dataset (Family-Balanced Reorganization)",
        "",
        "## Metric Contract",
        "",
        "- **Group definition**: 1 QA pair × 10 videos",
        "- **Overall CV**: average of the intra-group CVs",
        "- **MRA**: computed globally across all 10n model responses",
        "",
        "## Summary",
        "",
        f"- **Total groups**: {stats['total_groups']}",
        f"- **Total clips**: {stats['total_clips']}",
        f"- **Clips per group**: {stats['clips_per_group_distribution']}",
        f"- **Unique scenes**: {stats['unique_scenes']}",
        f"- **Unique anchor signatures**: {stats['unique_anchor_signatures']}",
        f"- **Unique atomic object labels**: {stats['unique_atomic_labels']}",
        f"- **Motion families**: {stats['motion_family_distribution']}",
        f"- **Dimensions**: {stats['dimension_distribution']}",
        "",
        "## Family Distribution",
        "",
        "| Family | Groups | Unique scenes | Unique anchor signatures | Unique atomic labels |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, payload in stats["family_stats"].items():
        lines.append(
            f"| {family} | {payload['groups']} | {payload['unique_scenes']} | {payload['unique_anchor_signatures']} | {payload['unique_atomic_labels']} |"
        )

    lines += [
        "",
        "## Top Scenes",
        "",
        "| Scene | Groups |",
        "|---|---:|",
    ]
    for scene, count in stats["top_scenes"]:
        lines.append(f"| {scene} | {count} |")

    lines += [
        "",
        "## Top Atomic Object Labels",
        "",
        "| Label | Groups |",
        "|---|---:|",
    ]
    for label, count in stats["top_atomic_labels"]:
        lines.append(f"| {label} | {count} |")

    lines += [
        "",
        "## Selection Notes",
        "",
        f"- **Rendered clip union**: {stats['selection_report']['rendered_clip_union_count']}",
        f"- **Atomic label caps**: {stats['selection_report'].get('atomic_label_caps', {})}",
        f"- **Pair-distance non-FloorPlan1 groups used**: {stats['selection_report']['pair_distance_non_floorplan1_selected']}",
        f"- **Pair-distance FloorPlan1 groups used**: {stats['selection_report']['pair_distance_floorplan1_selected']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reorganize the Thor video-consistency benchmark.")
    parser.add_argument("--eval_plan", required=True, help="Existing eval metadata.json, used as a tie-break preference.")
    parser.add_argument("--full_plan", required=True, help="Full Thor metadata.json, used as the candidate pool.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--clips_per_group", type=int, default=10)
    parser.add_argument("--family_targets", default=None, help="Comma-separated overrides, e.g. size=100,camera_distance=100,pair_distance=100")
    parser.add_argument("--atomic_label_caps", default="CoffeeTable=13", help="Comma-separated hard caps on atomic labels, e.g. CoffeeTable=13")
    parser.add_argument("--scene_caps", default=None, help="Comma-separated hard caps on scenes, e.g. FloorPlan1=50,FloorPlan203=11")
    parser.add_argument("--video_mode", choices=["copy_mp4", "copy_dir"], default="copy_mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    family_targets = _parse_targets(args.family_targets)
    atomic_label_caps = _parse_label_caps(args.atomic_label_caps)
    scene_caps = _parse_scene_caps(args.scene_caps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    eval_plan = json.loads(Path(args.eval_plan).read_text(encoding="utf-8"))
    full_plan = json.loads(Path(args.full_plan).read_text(encoding="utf-8"))
    eval_group_ids = {group["group_id"] for group in eval_plan["groups"]}

    render_dirs = _discover_render_dirs()
    rendered_clip_ids, source_by_clip = _render_index(render_dirs)
    print(f"[reorg] discovered {len(render_dirs)} render dirs", flush=True)
    print(f"[reorg] rendered clip union: {len(rendered_clip_ids)}", flush=True)

    full_clips_by_id = {clip["clip_id"]: clip for clip in full_plan["clips"]}
    candidates_by_family = _candidate_groups(
        full_plan,
        rendered_clip_ids,
        eval_group_ids,
        args.clips_per_group,
    )
    candidate_summary = {
        family: len(candidates)
        for family, candidates in sorted(candidates_by_family.items())
    }
    print(f"[reorg] candidate groups by family: {candidate_summary}", flush=True)

    selected_candidates = _select_family_balanced(
        candidates_by_family,
        family_targets,
        rng,
        atomic_label_caps,
        scene_caps,
    )
    selected_groups: list[dict] = []
    qa_entries: list[dict] = []
    selected_clip_ids: set[str] = set()

    for candidate in selected_candidates:
        chosen_clip_ids = _select_diverse_clips(
            candidate.available_clip_ids,
            full_clips_by_id,
            args.clips_per_group,
            rng,
        )
        selected_clip_ids.update(chosen_clip_ids)

        for clip_id in chosen_clip_ids:
            source = source_by_clip.get(clip_id)
            if source is None:
                raise FileNotFoundError(f"Missing rendered source for clip {clip_id}")
            target = videos_dir / clip_id
            _materialize_clip_dir(source, target, args.video_mode)

            clip = full_clips_by_id[clip_id]
            qa_entries.append(
                {
                    "clip_id": clip_id,
                    "group_id": candidate.group_id,
                    "video_path": str(target / "video.mp4"),
                    "engine": clip["engine"],
                    "scene_id": clip["scene_id"],
                    "room_bucket": clip.get("room_bucket", ""),
                    "motion_family": clip["motion_family"],
                    "trajectory": clip["trajectory"],
                    "direction": clip["direction"],
                    "variation_tag": clip["variation_tag"],
                    "radius": clip.get("radius"),
                    "start_angle_offset": clip["start_angle_offset"],
                    "cycle_count": clip["cycle_count"],
                    "quality_score": clip.get("quality_score", 0.0),
                    "questions": [
                        {
                            "question": clip["question"],
                            "answer": clip["ground_truth"],
                            "question_type": clip["question_type"],
                            "question_family": clip["question_family"],
                            "dimension": clip.get("dimension"),
                            "anchor_kind": clip["anchor_kind"],
                            "anchor_ids": list(clip["anchor_ids"]),
                            "anchor_labels": list(clip["anchor_labels"]),
                            "primary_object": clip["anchor_ids"][0],
                        }
                    ],
                }
            )

        selected_groups.append(
            {
                "group_id": candidate.group_id,
                "engine": candidate.group["engine"],
                "scene_id": candidate.group["scene_id"],
                "room_bucket": candidate.group.get("room_bucket", ""),
                "motion_family": candidate.group.get("motion_family", ""),
                "question_family": candidate.group["question_family"],
                "question_type": candidate.group["question_type"],
                "dimension": candidate.group.get("dimension"),
                "anchor_kind": candidate.group["anchor_kind"],
                "anchor_ids": list(candidate.group["anchor_ids"]),
                "anchor_labels": list(candidate.group["anchor_labels"]),
                "shared_ground_truth": candidate.group["shared_ground_truth"],
                "clip_ids": chosen_clip_ids,
                "selected_group_size": len(chosen_clip_ids),
                "available_rendered_clips": candidate.available_count,
                "from_eval_plan": candidate.from_eval_plan,
            }
        )

    # Remove stale symlinks from earlier benchmark versions.
    for child in videos_dir.iterdir():
        if child.name in selected_clip_ids:
            continue
        _clear_output_path(child)

    pair_selected = [group for group in selected_groups if group["question_family"] == "pair_distance"]
    selection_report = {
        "family_targets": family_targets,
        "atomic_label_caps": atomic_label_caps,
        "scene_caps": scene_caps,
        "candidate_groups_by_family": candidate_summary,
        "selected_groups_by_family": dict(sorted(Counter(group["question_family"] for group in selected_groups).items())),
        "rendered_clip_union_count": len(rendered_clip_ids),
        "pair_distance_non_floorplan1_selected": sum(group["scene_id"] != "FloorPlan1" for group in pair_selected),
        "pair_distance_floorplan1_selected": sum(group["scene_id"] == "FloorPlan1" for group in pair_selected),
    }

    stats = _build_stats(selected_groups, qa_entries, render_dirs, selection_report)

    (output_dir / "qa.json").write_text(json.dumps(qa_entries, indent=2), encoding="utf-8")
    (output_dir / "consistency_groups.json").write_text(json.dumps(selected_groups, indent=2), encoding="utf-8")
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (output_dir / "selection_report.json").write_text(json.dumps(selection_report, indent=2), encoding="utf-8")
    (output_dir / "video_consistency_dataset.md").write_text(_build_report(stats), encoding="utf-8")

    print(f"[reorg] wrote {len(selected_groups)} groups / {len(qa_entries)} clips to {output_dir}", flush=True)
    print(f"[reorg] family counts: {stats['question_family_distribution']}", flush=True)
    print(
        "[reorg] unique scenes: "
        f"{stats['unique_scenes']} | unique atomic labels: {stats['unique_atomic_labels']} "
        f"| unique anchor signatures: {stats['unique_anchor_signatures']}",
        flush=True,
    )
    print(f"[reorg] top scenes: {stats['top_scenes'][:10]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
