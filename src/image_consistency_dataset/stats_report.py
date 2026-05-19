"""Image consistency benchmark statistics and markdown report."""

import json
from collections import Counter
from pathlib import Path


def _counter_table(counter: Counter, key_name: str, total: int) -> list[dict]:
    return [
        {
            key_name: key,
            "count": value,
            "share_pct": round((100.0 * value) / max(total, 1), 1),
        }
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    ]


def build_image_dataset_stats(qa_entries: list[dict], group_entries: list[dict]) -> dict:
    total_images = len(qa_entries)
    total_groups = len(group_entries)
    return {
        "total_groups": total_groups,
        "total_images": total_images,
        "engine_distribution": _counter_table(
            Counter(e["engine"] for e in qa_entries), "engine", total_images
        ),
        "question_family_distribution": _counter_table(
            Counter(e["questions"][0]["question_family"] for e in qa_entries), "question_family", total_images
        ),
        "question_type_distribution": _counter_table(
            Counter(e["questions"][0]["question_type"] for e in qa_entries), "question_type", total_images
        ),
        "scene_distribution": _counter_table(
            Counter(e["scene_id"] for e in qa_entries), "scene_id", total_images
        ),
        "images_per_group_distribution": _counter_table(
            Counter(len(g["image_ids"]) for g in group_entries), "images_per_group", total_groups
        ),
        "anchor_label_distribution": _counter_table(
            Counter(e["questions"][0]["anchor_labels"][0] for e in qa_entries), "anchor_label", total_images
        ),
    }


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "| value | count |\n|---|---|\n"
    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join([header_row, divider_row] + body_rows)


def build_markdown_report(stats: dict) -> str:
    sections = [
        "# Image Consistency Dataset",
        "",
        "## Summary",
        "",
        f"- Total groups: `{stats['total_groups']}`",
        f"- Total images: `{stats['total_images']}`",
        "",
        "## Engine Distribution",
        "",
        _markdown_table(stats["engine_distribution"]),
        "",
        "## Question Family Distribution",
        "",
        _markdown_table(stats["question_family_distribution"]),
        "",
        "## Question Type Distribution",
        "",
        _markdown_table(stats["question_type_distribution"]),
        "",
        "## Images Per Group Distribution",
        "",
        _markdown_table(stats["images_per_group_distribution"]),
        "",
        "## Scene Distribution",
        "",
        _markdown_table(stats["scene_distribution"]),
        "",
        "## Anchor Label Distribution",
        "",
        _markdown_table(stats["anchor_label_distribution"]),
        "",
    ]
    return "\n".join(sections)


def write_image_dataset_report(
    qa_entries: list[dict],
    group_entries: list[dict],
    output_dir: str | Path,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stats = build_image_dataset_stats(qa_entries, group_entries)
    (output_root / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (output_root / "image_consistency_dataset.md").write_text(build_markdown_report(stats))
    return stats
