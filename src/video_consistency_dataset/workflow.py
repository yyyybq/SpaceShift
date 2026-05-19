"""Stage runners for the video consistency benchmark.

Stages:
  plan
  render
  report
  all
"""

from pathlib import Path

from video_consistency_dataset.interiorgs_mining import mine_interiorgs_candidates
from video_consistency_dataset.interiorgs_render import render_interiorgs_clips
from video_consistency_dataset.render_parallel import run_parallel_clip_render
from video_consistency_dataset.plan_io import (
    load_benchmark_plan,
    load_mined_candidates_checkpoint,
    update_dataset_artifacts,
    write_benchmark_plan,
    write_mined_candidates_checkpoint,
)
from video_consistency_dataset.planner import build_benchmark_plan
from video_consistency_dataset.stats_report import write_dataset_report
from video_consistency_dataset.thor_mining import mine_thor_candidates
from video_consistency_dataset.thor_render import render_thor_clips


def run_plan_stage(config):
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[plan] output_dir={config.output_dir}", flush=True)
    if config.reuse_mined_plan_candidates:
        thor_candidates, interiorgs_candidates = load_mined_candidates_checkpoint(config.output_dir)
        if "thor" not in config.enabled_engines:
            thor_candidates = []
        if "interiorgs" not in config.enabled_engines:
            interiorgs_candidates = []
        print(
            f"[plan] reused checkpoint thor={len(thor_candidates)} interiorgs={len(interiorgs_candidates)}",
            flush=True,
        )
    else:
        if "thor" in config.enabled_engines:
            print("[plan] mining THOR candidates...", flush=True)
        thor_candidates = mine_thor_candidates(config) if "thor" in config.enabled_engines else []
        if "thor" in config.enabled_engines:
            print(f"[plan] THOR candidates={len(thor_candidates)}", flush=True)
        if "interiorgs" in config.enabled_engines:
            print(
                "[plan] mining InteriorGS candidates (slow: many trajectories per scene; progress prints follow)...",
                flush=True,
            )
        interiorgs_candidates = mine_interiorgs_candidates(config) if "interiorgs" in config.enabled_engines else []
        if "interiorgs" in config.enabled_engines:
            print(f"[plan] InteriorGS candidates={len(interiorgs_candidates)}", flush=True)
        write_mined_candidates_checkpoint(config.output_dir, thor_candidates, interiorgs_candidates)
    print("[plan] building benchmark plan (group + subset selection; no GPU until render)...", flush=True)
    plan = build_benchmark_plan(
        config=config,
        thor_candidates=thor_candidates,
        interiorgs_candidates=interiorgs_candidates,
    )
    print(f"[plan] selected_groups={len(plan.groups)} selected_clips={len(plan.clips)}", flush=True)
    write_benchmark_plan(plan, config.output_dir)
    print(f"[plan] wrote artifacts to {config.output_dir}", flush=True)
    return plan


def run_render_stage(config):
    print(f"[render] loading plan from {config.plan_path()}", flush=True)
    plan = load_benchmark_plan(config.plan_path())
    render_records: dict[str, dict] = {}
    if config.render_parallel_workers > 1:
        render_records = run_parallel_clip_render(config)
    else:
        if "thor" in config.enabled_engines:
            render_records.update(render_thor_clips(plan, config))
        if "interiorgs" in config.enabled_engines:
            render_records.update(render_interiorgs_clips(plan, config))
    update_dataset_artifacts(plan, config.output_dir, render_records=render_records)
    print(f"[render] rendered_records={len(render_records)}", flush=True)
    return plan


def run_report_stage(config):
    print(f"[report] loading plan from {config.plan_path()}", flush=True)
    plan = load_benchmark_plan(config.plan_path())
    stats = write_dataset_report(plan, config.output_dir)
    update_dataset_artifacts(plan, config.output_dir, stats=stats)
    print(f"[report] wrote stats for {stats['total_groups']} groups", flush=True)
    return stats


def run_all_stages(config):
    print("[all] starting plan stage", flush=True)
    plan = run_plan_stage(config)
    render_records: dict[str, dict] = {}
    print("[all] starting render stage", flush=True)
    if config.render_parallel_workers > 1:
        render_records = run_parallel_clip_render(config)
    else:
        if "thor" in config.enabled_engines:
            render_records.update(render_thor_clips(plan, config))
        if "interiorgs" in config.enabled_engines:
            render_records.update(render_interiorgs_clips(plan, config))
    print("[all] starting report stage", flush=True)
    stats = write_dataset_report(plan, config.output_dir)
    update_dataset_artifacts(plan, config.output_dir, render_records=render_records, stats=stats)
    print(f"[all] finished groups={stats['total_groups']} videos={stats['total_videos']}", flush=True)
    return stats
