"""Build a supplemental plan slice from a full clips.jsonl (exclude prior eval/train roots)."""

import json
import random
from pathlib import Path

from video_consistency_dataset.benchmark_plan import BenchmarkPlan
from video_consistency_dataset.plan_io import write_benchmark_plan
from video_consistency_dataset.subsample_benchmark_plan_helpers import (
    collect_clip_rows,
    extract_config_from_metadata,
    make_group,
    reroot_row,
    row_to_clip,
    stream_group_aggregate,
)
from video_consistency_dataset.subsample_plan_selection import greedy_pick, infos_for_type


def _load_group_ids_from_plan(plan_dir: Path) -> set[str]:
    path = plan_dir / "clips.jsonl"
    assert path.is_file(), path
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            ids.add(json.loads(line)["group_id"])
    return ids


def run_supplement(
    source_plan_dir: Path,
    exclude_plan_dirs: list[Path],
    out_dir: Path,
    question_types: tuple[str, ...],
    target_groups: int,
    random_seed: int,
    require_clips: int,
) -> dict:
    source_plan_dir = source_plan_dir.resolve()
    clips_path = source_plan_dir / "clips.jsonl"
    metadata_path = source_plan_dir / "metadata.json"
    assert clips_path.is_file(), clips_path
    assert metadata_path.is_file(), metadata_path

    excluded: set[str] = set()
    for d in exclude_plan_dirs:
        excluded |= _load_group_ids_from_plan(d.resolve())

    base_config = extract_config_from_metadata(metadata_path)
    old_out = base_config["output_dir"]
    target_gs = int(base_config["group_size"])

    groups_agg = stream_group_aggregate(clips_path)
    rng = random.Random(random_seed)

    assert question_types, "At least one --question_types value is required."
    per_type = target_groups // len(question_types)
    remainder = target_groups % len(question_types)
    counts = tuple(per_type + (1 if i < remainder else 0) for i in range(len(question_types)))

    selected: list = []
    for qt, k in zip(question_types, counts, strict=True):
        if k <= 0:
            continue
        pool = infos_for_type(groups_agg, qt, require_clips, excluded)
        assert len(pool) >= k, (
            f"Not enough {qt} groups after exclusions: {len(pool)} < {k}"
        )
        selected.extend(greedy_pick(pool, k, rng))

    out_dir.mkdir(parents=True, exist_ok=True)
    new_out_str = str(out_dir.resolve())
    gid_order = [g.group_id for g in selected]
    id_order = {gid: groups_agg[gid]["clip_ids"] for gid in gid_order}
    clip_rows = collect_clip_rows(clips_path, gid_order, id_order)
    config = dict(base_config)
    config["output_dir"] = new_out_str
    built_groups: list = []
    built_clips: list = []
    for gid in gid_order:
        raw_rows = [reroot_row(r, old_out, new_out_str) for r in clip_rows[gid]]
        built_groups.append(make_group(raw_rows, target_gs))
        built_clips.extend(row_to_clip(r) for r in raw_rows)
    plan = BenchmarkPlan(config=config, groups=tuple(built_groups), clips=tuple(built_clips))
    write_benchmark_plan(plan, out_dir)

    report = {
        "question_types": list(question_types),
        "target_groups": target_groups,
        "selected_groups": len(selected),
        "clips": len(built_clips),
        "excluded_group_ids": len(excluded),
        "random_seed": random_seed,
        "require_clips": require_clips,
    }
    (out_dir / "selection_report.json").write_text(json.dumps(report, indent=2))
    return report
