"""Read and write benchmark plan artifacts.

Artifacts:
  metadata.json
  qa.json
  clips.jsonl
  consistency_groups.json
  plan_mined_candidates.pkl   written as soon as mining finishes (before group selection)
"""

import json
import pickle
from pathlib import Path

MINED_CANDIDATES_FILE = "plan_mined_candidates.pkl"

from video_consistency_dataset.benchmark_plan import BenchmarkPlan
from video_consistency_dataset.clip_spec import ClipSpec
from video_consistency_dataset.consistency_group import ConsistencyGroup


def _qa_entry(clip: ClipSpec) -> dict:
    payload = clip.to_dict()
    return {
        "clip_id": payload["clip_id"],
        "group_id": payload["group_id"],
        "video_path": payload["video_path"],
        "engine": payload["engine"],
        "scene_id": payload["scene_id"],
        "room_bucket": payload["room_bucket"],
        "motion_family": payload["motion_family"],
        "trajectory": payload["trajectory"],
        "direction": payload["direction"],
        "variation_tag": payload["variation_tag"],
        "radius": payload["radius"],
        "start_angle_offset": payload["start_angle_offset"],
        "cycle_count": payload["cycle_count"],
        "quality_score": payload["quality_score"],
        "questions": [{
            "question": payload["question"],
            "answer": payload["ground_truth"],
            "question_type": payload["question_type"],
            "question_family": payload["question_family"],
            "dimension": payload["dimension"],
            "anchor_kind": payload["anchor_kind"],
            "anchor_ids": payload["anchor_ids"],
            "anchor_labels": payload["anchor_labels"],
            "primary_object": payload["anchor_ids"][0],
        }],
    }


def _metadata_payload(
    plan: BenchmarkPlan,
    existing_payload: dict | None = None,
    render_records: dict[str, dict] | None = None,
    stats: dict | None = None,
) -> dict:
    payload = existing_payload or {}
    prior_render_records = payload.get("render_records", {})
    merged_render_records = dict(prior_render_records)
    if render_records is not None:
        merged_render_records.update(render_records)
    resolved_stats = stats if stats is not None else payload.get("stats")
    return {
        "config": plan.config,
        "groups": [group.to_dict() for group in plan.groups],
        "clips": [clip.to_dict() for clip in plan.clips],
        "render_records": merged_render_records,
        "stats": resolved_stats,
    }


def mined_candidates_checkpoint_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / MINED_CANDIDATES_FILE


def write_mined_candidates_checkpoint(
    output_dir: str | Path,
    thor_candidates: list[dict],
    interiorgs_candidates: list[dict],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = mined_candidates_checkpoint_path(root)
    payload = {"thor": thor_candidates, "interiorgs": interiorgs_candidates}
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    print(
        f"[plan] checkpoint {path.name}: thor={len(thor_candidates)} interiorgs={len(interiorgs_candidates)} "
        "(safe if later stages fail)",
        flush=True,
    )


def load_mined_candidates_checkpoint(output_dir: str | Path) -> tuple[list[dict], list[dict]]:
    path = mined_candidates_checkpoint_path(output_dir)
    assert path.is_file(), f"Missing {path}; run plan without --reuse_mined_plan_candidates first."
    payload = pickle.loads(path.read_bytes())
    assert isinstance(payload, dict) and "thor" in payload and "interiorgs" in payload
    return list(payload["thor"]), list(payload["interiorgs"])


def load_benchmark_plan(path: str | Path) -> BenchmarkPlan:
    payload = json.loads(Path(path).read_text())
    groups = tuple(ConsistencyGroup(**group) for group in payload["groups"])
    clips = tuple(ClipSpec(**clip) for clip in payload["clips"])
    return BenchmarkPlan(config=payload["config"], groups=groups, clips=clips)


def update_dataset_artifacts(
    plan: BenchmarkPlan,
    output_dir: str | Path,
    render_records: dict[str, dict] | None = None,
    stats: dict | None = None,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.json"
    existing_payload = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    metadata_payload = _metadata_payload(plan, existing_payload, render_records, stats)
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))
    (output_root / "qa.json").write_text(
        json.dumps([_qa_entry(clip) for clip in plan.clips], indent=2)
    )


def write_benchmark_plan(plan: BenchmarkPlan, output_dir: str | Path) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    update_dataset_artifacts(plan, output_root)
    (output_root / "benchmark_plan.json").write_text(json.dumps(plan.to_dict(), indent=2))
    (output_root / "consistency_groups.json").write_text(
        json.dumps([group.to_dict() for group in plan.groups], indent=2)
    )
    with (output_root / "clips.jsonl").open("w", encoding="utf-8") as handle:
        for clip in plan.clips:
            handle.write(json.dumps(clip.to_dict()) + "\n")
