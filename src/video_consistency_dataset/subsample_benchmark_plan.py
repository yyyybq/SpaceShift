"""Stream a full clips.jsonl plan into smaller eval/train plans with disjoint leak keys."""

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
from video_consistency_dataset.subsample_group_info import leak_key
from video_consistency_dataset.subsample_plan_selection import (
    QUESTION_TYPES_ORDER,
    select_eval_train_for_type,
)


def run_subsample(
    source_plan_dir: Path,
    eval_out: Path,
    train_out: Path,
    eval_per_type: int,
    train_per_type: int,
    random_seed: int,
    require_clips: int,
) -> tuple[dict, dict]:
    source_plan_dir = source_plan_dir.resolve()
    clips_path = source_plan_dir / "clips.jsonl"
    metadata_path = source_plan_dir / "metadata.json"
    assert clips_path.is_file(), clips_path
    assert metadata_path.is_file(), metadata_path
    base_config = extract_config_from_metadata(metadata_path)
    old_out = base_config["output_dir"]
    target_gs = int(base_config["group_size"])

    groups_agg = stream_group_aggregate(clips_path)
    rng = random.Random(random_seed)

    eval_groups: list = []
    train_groups: list = []
    for qt in QUESTION_TYPES_ORDER:
        ev, tr = select_eval_train_for_type(
            groups_agg, qt, require_clips, eval_per_type, train_per_type, rng
        )
        eval_groups.extend(ev)
        train_groups.extend(tr)

    eval_keys = {leak_key(g) for g in eval_groups}
    train_keys = {leak_key(g) for g in train_groups}
    assert not (train_keys & eval_keys)

    train_report = {
        "groups_per_type": {
            qt: sum(1 for g in train_groups if g.question_type == qt) for qt in QUESTION_TYPES_ORDER
        },
        "total_groups": len(train_groups),
    }
    eval_report = {
        "groups_per_type": {
            qt: sum(1 for g in eval_groups if g.question_type == qt) for qt in QUESTION_TYPES_ORDER
        },
        "total_groups": len(eval_groups),
    }

    def materialize(selections, out_dir: Path, report: dict) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        new_out_str = str(out_dir.resolve())
        gid_order = [g.group_id for g in selections]
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
        out_report = {
            **report,
            "clips": len(built_clips),
            "random_seed": random_seed,
            "require_clips": require_clips,
        }
        (out_dir / "selection_report.json").write_text(json.dumps(out_report, indent=2))
        return out_report

    eval_summary = materialize(eval_groups, eval_out, eval_report)
    train_summary = materialize(train_groups, train_out, train_report)
    overlap_note = {"eval_train_leak_key_overlap": len(train_keys & eval_keys)}
    (eval_out / "selection_report.json").write_text(json.dumps(eval_summary, indent=2))
    (train_out / "selection_report.json").write_text(
        json.dumps({**train_summary, **overlap_note}, indent=2)
    )
    return eval_summary, train_summary
