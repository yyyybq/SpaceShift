"""Main orchestration for extracting the image consistency benchmark from video output.

Reads the video benchmark qa.json and consistency_groups, selects diverse frames
per group, and writes the image benchmark artifacts.

Balanced selection: caps per-label and per-scene counts within each family,
and for pair_distance groups pre-filters to groups with enough co-visible frames.
"""

import json
import os
import shutil
from collections import Counter
from pathlib import Path

from image_consistency_dataset.frame_selector import FrameCandidate, select_frames
from image_consistency_dataset.question_entries import rewrite_question_for_image
from image_consistency_dataset.stats_report import write_image_dataset_report


def _load_video_qa(video_dir: Path) -> list[dict]:
    qa_path = video_dir / "qa.json"
    assert qa_path.exists(), f"Missing {qa_path}"
    return json.loads(qa_path.read_text())


def _load_video_groups(video_dir: Path) -> list[dict] | None:
    groups_path = video_dir / "consistency_groups.json"
    if groups_path.exists():
        return json.loads(groups_path.read_text())
    return None


def _groups_from_qa(qa_entries: list[dict]) -> list[dict]:
    """Reconstruct minimal group info from qa.json entries when consistency_groups.json is missing."""
    by_group: dict[str, list[dict]] = {}
    for entry in qa_entries:
        by_group.setdefault(entry["group_id"], []).append(entry)
    groups = []
    for group_id, entries in by_group.items():
        first = entries[0]
        q = first["questions"][0]
        groups.append({
            "group_id": group_id,
            "engine": first["engine"],
            "scene_id": first["scene_id"],
            "room_bucket": first.get("room_bucket", ""),
            "question_type": q["question_type"],
            "question_family": q["question_family"],
            "dimension": q.get("dimension"),
            "anchor_kind": q["anchor_kind"],
            "anchor_ids": tuple(q["anchor_ids"]),
            "anchor_labels": tuple(q["anchor_labels"]),
            "shared_ground_truth": q["answer"],
            "clip_ids": tuple(e["clip_id"] for e in entries),
        })
    return groups


def _partition_by_family(groups: list[dict]) -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = {}
    for g in groups:
        family = g["question_family"]
        by_family.setdefault(family, []).append(g)
    return by_family


def _count_covisible_frames(group: dict, qa_by_group: dict, video_root: Path) -> int:
    """Count co-visible frames for a pair group (both anchors visible)."""
    anchor_set = set(group["anchor_ids"])
    clip_entries = qa_by_group.get(group["group_id"], [])
    total = 0
    for entry in clip_entries:
        vis_path = video_root / entry["clip_id"] / "frame_visibility.json"
        if not vis_path.exists():
            continue
        vis = json.loads(vis_path.read_text())
        for visible_ids in vis["frames"].values():
            if anchor_set.issubset(set(visible_ids)):
                total += 1
    return total


def _balanced_select(
    groups: list[dict],
    target: int,
    max_per_label: int,
    max_per_scene: int,
) -> list[dict]:
    """Greedy balanced selection: pick one group at a time, always the least-represented.

    Three passes with increasing relaxation:
      1. Strict label + scene caps
      2. Relax scene cap, keep label cap
      3. Relax both (fill remaining)
    """
    label_counts: Counter = Counter()
    scene_counts: Counter = Counter()
    selected: list[dict] = []
    selected_ids: set[str] = set()
    remaining = list(groups)

    def _pick_best(candidates: list[dict]) -> dict | None:
        best = None
        best_score = None
        for g in candidates:
            score = (label_counts[g["anchor_labels"][0]], scene_counts[g["scene_id"]])
            if best_score is None or score < best_score:
                best = g
                best_score = score
        return best

    # Pass 1: strict label + scene caps
    while len(selected) < target and remaining:
        eligible = [
            g for g in remaining
            if label_counts[g["anchor_labels"][0]] < max_per_label
            and scene_counts[g["scene_id"]] < max_per_scene
        ]
        best = _pick_best(eligible)
        if best is None:
            break
        selected.append(best)
        selected_ids.add(best["group_id"])
        label_counts[best["anchor_labels"][0]] += 1
        scene_counts[best["scene_id"]] += 1
        remaining.remove(best)

    # Pass 2: relax scene cap, keep label cap
    while len(selected) < target and remaining:
        eligible = [
            g for g in remaining
            if label_counts[g["anchor_labels"][0]] < max_per_label
        ]
        best = _pick_best(eligible)
        if best is None:
            break
        selected.append(best)
        selected_ids.add(best["group_id"])
        label_counts[best["anchor_labels"][0]] += 1
        scene_counts[best["scene_id"]] += 1
        remaining.remove(best)

    # Pass 3: relax both, just fill with least-represented
    while len(selected) < target and remaining:
        best = _pick_best(remaining)
        if best is None:
            break
        selected.append(best)
        selected_ids.add(best["group_id"])
        label_counts[best["anchor_labels"][0]] += 1
        scene_counts[best["scene_id"]] += 1
        remaining.remove(best)

    return selected


def _build_image_entry(
    image_id: str,
    group_id: str,
    image_path: str,
    source_clip_entry: dict,
    frame_candidate: FrameCandidate,
    source_question: dict,
) -> dict:
    """Build a single image QA entry."""
    rewritten_q = rewrite_question_for_image(source_question["question"])
    return {
        "image_id": image_id,
        "group_id": group_id,
        "image_path": image_path,
        "source_clip_id": frame_candidate.clip_id,
        "source_frame": frame_candidate.frame_path.name,
        "engine": source_clip_entry["engine"],
        "scene_id": source_clip_entry["scene_id"],
        "questions": [{
            "question": rewritten_q,
            "answer": source_question["answer"],
            "question_type": source_question["question_type"],
            "question_family": source_question["question_family"],
            "dimension": source_question.get("dimension"),
            "anchor_kind": source_question["anchor_kind"],
            "anchor_ids": source_question["anchor_ids"],
            "anchor_labels": source_question["anchor_labels"],
        }],
    }


def extract_image_benchmark(
    video_dir: str | Path,
    output_dir: str | Path,
    views_per_group: int = 10,
    target_groups_per_family: int = 100,
    max_per_label: int = 10,
    max_per_scene: int = 5,
) -> dict:
    """Extract the image consistency benchmark from a video benchmark output.

    Args:
        video_dir: path to the video benchmark output (contains qa.json and videos/).
        output_dir: path to write image benchmark artifacts.
        views_per_group: number of images to select per consistency group.
        target_groups_per_family: max groups per question family.
        max_per_label: max groups per anchor label within each family.
        max_per_scene: max groups per scene within each family.

    Returns:
        Dataset stats dict.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Determine video root (where clip directories live)
    video_root = video_dir / "videos"
    assert video_root.is_dir(), f"Missing videos/ directory at {video_root}"

    # Load video benchmark data
    qa_entries = _load_video_qa(video_dir)
    video_groups = _load_video_groups(video_dir)
    if video_groups is None:
        video_groups = _groups_from_qa(qa_entries)
        print(f"[extract] reconstructed {len(video_groups)} groups from qa.json", flush=True)
    else:
        print(f"[extract] loaded {len(video_groups)} groups from consistency_groups.json", flush=True)

    # Index qa entries by group_id and clip_id
    qa_by_group: dict[str, list[dict]] = {}
    qa_by_clip: dict[str, dict] = {}
    for entry in qa_entries:
        qa_by_group.setdefault(entry["group_id"], []).append(entry)
        qa_by_clip[entry["clip_id"]] = entry

    # Partition groups by question family
    by_family = _partition_by_family(video_groups)
    print(f"[extract] question families: { {k: len(v) for k, v in by_family.items()} }", flush=True)

    # Pre-filter pair_distance groups: only keep those with >= views_per_group co-visible frames
    if "pair_distance" in by_family:
        pair_groups = by_family["pair_distance"]
        print(f"[extract] pair_distance: checking co-visibility for {len(pair_groups)} groups...", flush=True)
        eligible = []
        for g in pair_groups:
            covis = _count_covisible_frames(g, qa_by_group, video_root)
            if covis >= views_per_group:
                eligible.append(g)
        print(
            f"[extract] pair_distance: {len(eligible)}/{len(pair_groups)} groups "
            f"have >= {views_per_group} co-visible frames",
            flush=True,
        )
        by_family["pair_distance"] = eligible

    all_image_entries: list[dict] = []
    all_image_groups: list[dict] = []

    for family, groups in sorted(by_family.items()):
        selected = _balanced_select(groups, target_groups_per_family, max_per_label, max_per_scene)
        label_dist = Counter(g["anchor_labels"][0] for g in selected).most_common(5)
        scene_dist = Counter(g["scene_id"] for g in selected).most_common(5)
        print(
            f"[extract] family={family}: selected {len(selected)}/{len(groups)} groups "
            f"(top labels: {label_dist}, top scenes: {scene_dist})",
            flush=True,
        )

        for group in selected:
            group_id = group["group_id"]
            clip_entries = qa_by_group.get(group_id, [])
            if not clip_entries:
                continue

            anchor_kind = group["anchor_kind"]
            anchor_ids = tuple(group["anchor_ids"])

            frames = select_frames(
                clip_entries=clip_entries,
                anchor_kind=anchor_kind,
                anchor_ids=anchor_ids,
                video_root=video_root,
                target_count=views_per_group,
            )

            if not frames:
                print(f"  WARN: no frames selected for group {group_id}, skipping", flush=True)
                continue

            image_ids = []
            for view_idx, fc in enumerate(frames):
                image_id = f"{group_id}_view{view_idx:02d}"
                image_ids.append(image_id)

                # Materialize a real image file in the benchmark output.
                img_dir = images_dir / image_id
                img_dir.mkdir(parents=True, exist_ok=True)
                image_path = img_dir / "image.png"
                if image_path.exists() or image_path.is_symlink():
                    image_path.unlink()
                source_abs = fc.frame_path.resolve()
                shutil.copy2(source_abs, image_path)

                # Build QA entry
                source_clip = qa_by_clip.get(fc.clip_id, clip_entries[0])
                source_q = source_clip["questions"][0]
                entry = _build_image_entry(
                    image_id=image_id,
                    group_id=group_id,
                    image_path=str(image_path),
                    source_clip_entry=source_clip,
                    frame_candidate=fc,
                    source_question=source_q,
                )
                all_image_entries.append(entry)

            all_image_groups.append({
                "group_id": group_id,
                "engine": group["engine"],
                "scene_id": group["scene_id"],
                "question_family": group["question_family"],
                "question_type": group["question_type"],
                "dimension": group.get("dimension"),
                "anchor_kind": group["anchor_kind"],
                "anchor_ids": list(group["anchor_ids"]),
                "anchor_labels": list(group["anchor_labels"]),
                "shared_ground_truth": group["shared_ground_truth"],
                "image_ids": image_ids,
                "images_selected": len(image_ids),
            })

    # Write output artifacts
    (output_dir / "qa.json").write_text(json.dumps(all_image_entries, indent=2))
    (output_dir / "consistency_groups.json").write_text(json.dumps(all_image_groups, indent=2))

    print(
        f"[extract] wrote {len(all_image_entries)} image entries, "
        f"{len(all_image_groups)} groups",
        flush=True,
    )

    # Stats and report
    stats = write_image_dataset_report(all_image_entries, all_image_groups, output_dir)
    print(f"[extract] stats: {stats['total_groups']} groups, {stats['total_images']} images", flush=True)
    return stats
