"""Benchmark stats and markdown reporting.

Outputs:
  dataset_stats.json
  video_consistency_dataset.md
"""

import json
from collections import Counter
from pathlib import Path

from video_consistency_dataset.benchmark_plan import BenchmarkPlan


def _counter_table(counter: Counter, key_name: str, total: int) -> list[dict]:
    return [
        {
            key_name: key,
            "count": value,
            "share_pct": round((100.0 * value) / max(total, 1), 1),
        }
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    ]


def _angle_bin(angle: float) -> str:
    start = int((angle % 360.0) // 30.0) * 30
    end = start + 30
    return f"[{start}, {end})"


def _pair_counter_table(counter: Counter, key_names: tuple[str, str], total: int) -> list[dict]:
    return [
        {
            key_names[0]: key[0],
            key_names[1]: key[1],
            "count": value,
            "share_pct": round((100.0 * value) / max(total, 1), 1),
        }
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    ]


def build_dataset_stats(plan: BenchmarkPlan) -> dict:
    clips = plan.clips
    groups = plan.groups
    total_clips = len(clips)
    total_groups = len(groups)
    return {
        "total_groups": total_groups,
        "total_videos": total_clips,
        "engine_distribution": _counter_table(Counter(clip.engine for clip in clips), "engine", total_clips),
        "motion_distribution": _counter_table(Counter(clip.motion_family for clip in clips), "motion_family", total_clips),
        "qa_distribution": _counter_table(Counter(clip.question_type for clip in clips), "question_type", total_clips),
        "question_family_distribution": _counter_table(Counter(clip.question_family for clip in clips), "question_family", total_clips),
        "dimension_distribution": _counter_table(Counter(clip.dimension for clip in clips if clip.dimension is not None), "dimension", total_clips),
        "engine_motion_distribution": _pair_counter_table(Counter((clip.engine, clip.motion_family) for clip in clips), ("engine", "motion_family"), total_clips),
        "engine_qa_distribution": _pair_counter_table(Counter((clip.engine, clip.question_type) for clip in clips), ("engine", "question_type"), total_clips),
        "room_distribution": _counter_table(Counter(clip.room_bucket for clip in clips), "room_bucket", total_clips),
        "object_distribution": _counter_table(Counter(clip.anchor_labels[0] for clip in clips), "anchor_label", total_clips),
        "radius_distribution": _counter_table(Counter(clip.radius for clip in clips if clip.radius is not None), "radius", total_clips),
        "start_position_distribution": _counter_table(Counter(_angle_bin(clip.start_angle_offset) for clip in clips), "start_angle_bin", total_clips),
        "cycle_distribution": _counter_table(Counter(clip.cycle_count for clip in clips), "cycle_count", total_clips),
        "group_size_distribution": _counter_table(Counter(group.selected_group_size for group in groups), "group_size", total_groups),
        "scene_reuse_distribution": _counter_table(Counter(clip.scene_id for clip in clips), "scene_id", total_clips),
    }


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "| value | count |\n|---|---|\n"
    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(str(row[header]) for header in headers) + " |" for row in rows]
    return "\n".join([header_row, divider_row] + body_rows)


def build_markdown_report(plan: BenchmarkPlan, stats: dict) -> str:
    sections = [
        "# Video Consistency Dataset",
        "",
        "## Summary",
        "",
        f"- Total groups: `{stats['total_groups']}`",
        f"- Total videos: `{stats['total_videos']}`",
        "",
        "## Engine Distribution",
        "",
        _markdown_table(stats["engine_distribution"]),
        "",
        "## Motion Distribution",
        "",
        _markdown_table(stats["motion_distribution"]),
        "",
        "## QA Type Distribution",
        "",
        _markdown_table(stats["qa_distribution"]),
        "",
        "## Question Family Distribution",
        "",
        _markdown_table(stats["question_family_distribution"]),
        "",
        "## Dimension Distribution",
        "",
        _markdown_table(stats["dimension_distribution"]),
        "",
        "## Engine x Motion Distribution",
        "",
        _markdown_table(stats["engine_motion_distribution"]),
        "",
        "## Engine x QA Distribution",
        "",
        _markdown_table(stats["engine_qa_distribution"]),
        "",
        "## Room Distribution",
        "",
        _markdown_table(stats["room_distribution"]),
        "",
        "## Object Distribution",
        "",
        _markdown_table(stats["object_distribution"]),
        "",
        "## Radius Distribution",
        "",
        _markdown_table(stats["radius_distribution"]),
        "",
        "## Start Position Distribution",
        "",
        _markdown_table(stats["start_position_distribution"]),
        "",
        "## Cycle Distribution",
        "",
        _markdown_table(stats["cycle_distribution"]),
        "",
        "## Group Size Distribution",
        "",
        _markdown_table(stats["group_size_distribution"]),
        "",
        "## Scene Reuse Distribution",
        "",
        _markdown_table(stats["scene_reuse_distribution"]),
        "",
    ]
    return "\n".join(sections)


def write_dataset_report(plan: BenchmarkPlan, output_dir: str | Path) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stats = build_dataset_stats(plan)
    (output_root / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (output_root / "video_consistency_dataset.md").write_text(build_markdown_report(plan, stats))
    return stats
