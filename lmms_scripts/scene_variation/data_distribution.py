"""Compute SceneShift data distribution and emit a donut figure + LaTeX table.

Counts every QA pair across the three released sub-benchmarks
(scene_variation, image_consistency_thor_eval_v2, video_consistency_thor_eval_v2)
and breaks them down by:
  - top-level bucket: View Variation (camera moves) vs Scene Edit (camera fixed)
  - sub-pattern   : object rotation / translate / remove / cam rotation /
                    pass-by / orbit / approach / spherical / static
  - question type : Obj. Size, Cam. Dist., Pair Dist.
  - input modality: image vs video (only for View Variation)

Usage:
    python scripts/scene_variation/data_distribution.py
    python scripts/scene_variation/data_distribution.py \
        --tex-out /nas2/edwin/lmms-eval/data_distribution.tex \
        --pdf-out /nas2/edwin/lmms-eval/data_distribution.pdf \
        --png-out /nas2/edwin/lmms-eval/data_distribution.png

Input JSONL spec (per row, only the relevant keys are listed):
    scene_variation_042226_aligned.jsonl     -> question_type, edit_type
    image_consistency_thor_eval_v2.jsonl     -> question_type, group_id (motion)
    video_consistency_thor_eval_v2.jsonl     -> question_type, motion_family

Outputs:
    {tex_out}  LaTeX table (booktabs) of the distribution
    {pdf_out}  Donut figure (matplotlib) replacing the legacy infographic
    {png_out}  Same figure as PNG (for quick previews)
    Summary printed to stdout as a JSON payload.
"""

import argparse
import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path("/nas2/edwin/lmms-eval/data")
DEFAULT_TEX_OUT = Path("/nas2/edwin/lmms-eval/data_distribution.tex")
DEFAULT_PDF_OUT = Path("/nas2/edwin/lmms-eval/data_distribution.pdf")
DEFAULT_PNG_OUT = Path("/nas2/edwin/lmms-eval/data_distribution.png")

JSONL_FILES = OrderedDict(
    [
        ("scene_variation", "scene_variation_042226_aligned.jsonl"),
        ("image_consistency_thor_eval_v2", "image_consistency_thor_eval_v2.jsonl"),
        ("video_consistency_thor_eval_v2", "video_consistency_thor_eval_v2.jsonl"),
    ]
)

QTYPE_LABEL = OrderedDict(
    [
        ("object_dimensions", "Obj. Size"),
        ("object_distance_to_camera", "Cam. Dist."),
        ("object_pair_distance_center", "Pair Dist."),
    ]
)

SCENE_EDIT_PATTERNS = OrderedDict(
    [
        ("obj_rotation", "Object Rotation"),
        ("translate", "Object Translation"),
        ("remove", "Object Removal"),
    ]
)

CAMERA_VAR_PATTERNS = OrderedDict(
    [
        ("cam_rotation", "Camera Rotation"),
        ("passby", "Linear Pass-by"),
        ("around", "Orbit Around"),
        ("approach", "Approach / Recede"),
        ("spherical", "Spherical Orbit"),
        ("static", "Randomized Static"),
    ]
)

ALL_PATTERNS = OrderedDict(list(SCENE_EDIT_PATTERNS.items()) + list(CAMERA_VAR_PATTERNS.items()))

PALETTE = {
    "View Variation": "#4F8FE0",
    "Scene Edit": "#F4A24A",
    "obj_rotation": "#F9C57A",
    "translate": "#F4A24A",
    "remove": "#E08032",
    "cam_rotation": "#A4C7F0",
    "passby": "#7FB1EA",
    "around": "#5C9BE3",
    "approach": "#3F86DB",
    "spherical": "#2B71C8",
    "static": "#1F5DAB",
    "Obj. Size": "#9CC2F1",
    "Cam. Dist.": "#5392DC",
    "Pair Dist.": "#2E6CB6",
}

_IMG_V2_GROUP_RE = re.compile(
    r"thor_(?P<motion>\w+?)_object_(?:pair_distance_center|distance_to_camera|dimensions)_"
)


@dataclass
class PatternStats:
    pattern: str
    bucket: str
    qtype_counts: Counter = field(default_factory=Counter)
    medium_counts: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.qtype_counts.values())


def _classify_scene_variation(row: dict) -> str | None:
    et = row.get("edit_type")
    if et == "rotate":
        return "obj_rotation"
    if et == "translate":
        return "translate"
    if et == "remove":
        return "remove"
    return None


def _classify_image_consistency(row: dict) -> str | None:
    gid = row.get("group_id") or ""
    m = _IMG_V2_GROUP_RE.match(gid + "_")
    if not m:
        return None
    mf = m.group("motion")
    if mf == "rotation":
        return "cam_rotation"
    return mf if mf in CAMERA_VAR_PATTERNS else None


def _classify_video_consistency(row: dict) -> str | None:
    mf = row.get("motion_family")
    if mf == "rotation":
        return "cam_rotation"
    return mf if mf in CAMERA_VAR_PATTERNS else None


CLASSIFIERS = {
    "scene_variation": _classify_scene_variation,
    "image_consistency_thor_eval_v2": _classify_image_consistency,
    "video_consistency_thor_eval_v2": _classify_video_consistency,
}

MEDIUM_OF = {
    "scene_variation": "image",
    "image_consistency_thor_eval_v2": "image",
    "video_consistency_thor_eval_v2": "video",
}

BUCKET_OF_PATTERN = {p: "Scene Edit" for p in SCENE_EDIT_PATTERNS}
BUCKET_OF_PATTERN.update({p: "View Variation" for p in CAMERA_VAR_PATTERNS})


def _iter_rows(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_stats(data_dir: Path) -> dict[str, PatternStats]:
    stats: dict[str, PatternStats] = {
        p: PatternStats(pattern=p, bucket=BUCKET_OF_PATTERN[p]) for p in ALL_PATTERNS
    }
    skipped: Counter = Counter()
    for task, fname in JSONL_FILES.items():
        path = data_dir / fname
        assert path.exists(), f"Missing JSONL: {path}"
        classify = CLASSIFIERS[task]
        medium = MEDIUM_OF[task]
        for row in _iter_rows(path):
            qt = row.get("question_type")
            if qt not in QTYPE_LABEL:
                skipped[("qtype", qt)] += 1
                continue
            pat = classify(row)
            if pat is None:
                skipped[("pattern", task)] += 1
                continue
            stats[pat].qtype_counts[qt] += 1
            stats[pat].medium_counts[medium] += 1
    if skipped:
        print(f"[warn] skipped rows: {dict(skipped)}")
    return stats


def summarise(stats: dict[str, PatternStats]) -> dict:
    total = sum(s.total for s in stats.values())
    bucket_totals = Counter()
    for s in stats.values():
        bucket_totals[s.bucket] += s.total
    summary = {
        "total": total,
        "buckets": [
            {
                "name": bucket,
                "count": count,
                "pct": round(100.0 * count / total, 2),
            }
            for bucket, count in bucket_totals.most_common()
        ],
        "patterns": [],
    }
    for pat, label in ALL_PATTERNS.items():
        s = stats[pat]
        summary["patterns"].append(
            {
                "pattern": pat,
                "label": label,
                "bucket": s.bucket,
                "count": s.total,
                "pct": round(100.0 * s.total / total, 2) if total else 0.0,
                "by_qtype": {QTYPE_LABEL[q]: s.qtype_counts.get(q, 0) for q in QTYPE_LABEL},
                "by_medium": dict(s.medium_counts),
            }
        )
    return summary


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def _fmt_count(n: int) -> str:
    return f"{n:,}"


def render_tex(summary: dict) -> str:
    total = summary["total"]
    rows: list[str] = []

    rows.append("\\begin{table}[H]")
    rows.append("    \\centering")
    rows.append("    \\resizebox{\\textwidth}{!}{%")
    rows.append("    \\setlength\\tabcolsep{6pt}")
    rows.append("    \\renewcommand{\\arraystretch}{1.15}")
    rows.append("    \\begin{tabular}{@{}l l c c c c c@{}}")
    rows.append("    \\toprule")
    rows.append(
        "    \\textbf{Bucket} & \\textbf{Sub-pattern} & \\textbf{Obj. Size} & "
        "\\textbf{Cam. Dist.} & \\textbf{Pair Dist.} & \\textbf{Total} & \\textbf{Share} \\\\"
    )
    rows.append("    \\midrule")

    for bucket_name in ("View Variation", "Scene Edit"):
        sub_rows = [p for p in summary["patterns"] if p["bucket"] == bucket_name]
        if not sub_rows:
            continue
        bucket_total = sum(p["count"] for p in sub_rows)
        bucket_pct = 100.0 * bucket_total / total if total else 0.0
        first = True
        for p in sub_rows:
            qt = p["by_qtype"]
            bucket_cell = (
                f"\\multirow{{{len(sub_rows)}}}{{*}}{{\\textbf{{{bucket_name}}}}}"
                if first
                else ""
            )
            first = False
            rows.append(
                "    "
                + " & ".join(
                    [
                        bucket_cell,
                        p["label"],
                        _fmt_count(qt["Obj. Size"]),
                        _fmt_count(qt["Cam. Dist."]),
                        _fmt_count(qt["Pair Dist."]),
                        _fmt_count(p["count"]),
                        f"{p['pct']:.1f}\\%",
                    ]
                )
                + " \\\\"
            )
        rows.append("    \\cmidrule(l){2-7}")
        rows.append(
            "    "
            + " & ".join(
                [
                    "",
                    "\\textit{Subtotal}",
                    _fmt_count(sum(p["by_qtype"]["Obj. Size"] for p in sub_rows)),
                    _fmt_count(sum(p["by_qtype"]["Cam. Dist."] for p in sub_rows)),
                    _fmt_count(sum(p["by_qtype"]["Pair Dist."] for p in sub_rows)),
                    f"\\textbf{{{_fmt_count(bucket_total)}}}",
                    f"\\textbf{{{bucket_pct:.1f}\\%}}",
                ]
            )
            + " \\\\"
        )
        rows.append("    \\midrule")

    qt_totals = Counter()
    for p in summary["patterns"]:
        for q, c in p["by_qtype"].items():
            qt_totals[q] += c
    rows.append(
        "    "
        + " & ".join(
            [
                "\\textbf{Total}",
                "",
                f"\\textbf{{{_fmt_count(qt_totals['Obj. Size'])}}}",
                f"\\textbf{{{_fmt_count(qt_totals['Cam. Dist.'])}}}",
                f"\\textbf{{{_fmt_count(qt_totals['Pair Dist.'])}}}",
                f"\\textbf{{{_fmt_count(total)}}}",
                "\\textbf{100.0\\%}",
            ]
        )
        + " \\\\"
    )
    rows.append("    \\bottomrule")
    rows.append("    \\end{tabular}%")
    rows.append("    }")
    rows.append(
        "    \\caption{\\textbf{SceneShift data distribution.} "
        f"The benchmark contains {_fmt_count(total)} numeric QA pairs split into two "
        "top-level buckets. \\textit{View Variation} keeps the scene fixed and varies the "
        "camera (image inputs from \\texttt{image\\_consistency\\_thor\\_eval\\_v2}, video "
        "inputs from \\texttt{video\\_consistency\\_thor\\_eval\\_v2}). \\textit{Scene Edit} "
        "keeps the camera fixed and edits the scene (\\texttt{scene\\_variation}: object "
        "rotation, translation, and removal). Every pattern is queried with three numeric "
        "question types: object size (longest base dimension), object-to-camera distance, "
        "and object-pair center distance.}"
    )
    rows.append("    \\label{tab:sceneshift_data_distribution}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Donut figure
# ---------------------------------------------------------------------------


def _draw_donut(ax, summary: dict, total: int) -> None:
    from math import cos, radians, sin

    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    by_pattern = {p["pattern"]: p for p in summary["patterns"]}

    inner_radius = 0.55
    mid_radius = 0.78
    outer_radius = 1.00

    bucket_order = ["View Variation", "Scene Edit"]
    angle = 90.0
    bucket_arcs: dict[str, tuple[float, float]] = {}
    for bucket in bucket_order:
        share = sum(p["count"] for p in summary["patterns"] if p["bucket"] == bucket) / total
        sweep = -360.0 * share
        end_angle = angle + sweep
        ax.add_patch(
            Wedge(
                (0, 0),
                mid_radius,
                end_angle,
                angle,
                width=mid_radius - inner_radius,
                facecolor=PALETTE[bucket],
                edgecolor="white",
                linewidth=2.0,
            )
        )
        bucket_arcs[bucket] = (angle, end_angle)
        angle = end_angle

    angle = 90.0
    for bucket in bucket_order:
        for pat in (CAMERA_VAR_PATTERNS if bucket == "View Variation" else SCENE_EDIT_PATTERNS):
            entry = by_pattern[pat]
            if entry["count"] == 0:
                continue
            share = entry["count"] / total
            sweep = -360.0 * share
            end_angle = angle + sweep
            ax.add_patch(
                Wedge(
                    (0, 0),
                    outer_radius,
                    end_angle,
                    angle,
                    width=outer_radius - mid_radius,
                    facecolor=PALETTE[pat],
                    edgecolor="white",
                    linewidth=1.2,
                )
            )
            angle = end_angle

    ax.text(0, 0.16, "SceneShift", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0, 0.00, f"{total:,}", ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0, -0.16, "QA pairs", ha="center", va="center", fontsize=10, color="#555555")

    for bucket, (a0, a1) in bucket_arcs.items():
        mid = (a0 + a1) / 2.0
        r = (inner_radius + mid_radius) / 2.0
        x = r * cos(radians(mid))
        y = r * sin(radians(mid))
        rotation = (mid + 90.0) % 360.0
        if rotation > 90.0 and rotation < 270.0:
            rotation -= 180.0
        ax.text(
            x,
            y,
            bucket,
            ha="center",
            va="center",
            fontsize=10.5,
            color="white",
            fontweight="bold",
            rotation=rotation,
            rotation_mode="anchor",
        )

    legend_lines: list[tuple[str, str, bool]] = []
    for bucket in bucket_order:
        bucket_count = sum(p["count"] for p in summary["patterns"] if p["bucket"] == bucket)
        bucket_pct = 100.0 * bucket_count / total
        legend_lines.append((PALETTE[bucket], f"{bucket} ({bucket_pct:.1f}%)", True))
        for pat, label in (
            CAMERA_VAR_PATTERNS.items() if bucket == "View Variation" else SCENE_EDIT_PATTERNS.items()
        ):
            entry = by_pattern[pat]
            if entry["count"] == 0:
                continue
            legend_lines.append((PALETTE[pat], f"   {label} ({entry['pct']:.1f}%)", False))

    legend_x = -1.55
    legend_y_top = 1.10
    line_step = 0.135
    for i, (color, text, bold) in enumerate(legend_lines):
        y = legend_y_top - i * line_step
        ax.add_patch(
            plt.Rectangle(
                (legend_x, y - 0.045), 0.085, 0.085, facecolor=color, edgecolor="none"
            )
        )
        ax.text(
            legend_x + 0.115,
            y,
            text,
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold" if bold else "normal",
            color="black" if bold else "#222222",
        )

    ax.set_xlim(-1.7, 1.15)
    ax.set_ylim(-1.20, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_breakdown_table(ax, summary: dict, total: int) -> None:
    by_pattern = {p["pattern"]: p for p in summary["patterns"]}
    bucket_order = ["View Variation", "Scene Edit"]

    column_groups: list[tuple[str, str, dict]] = []
    for bucket in bucket_order:
        for pat, label in (
            CAMERA_VAR_PATTERNS.items() if bucket == "View Variation" else SCENE_EDIT_PATTERNS.items()
        ):
            entry = by_pattern[pat]
            if entry["count"] == 0:
                continue
            column_groups.append((bucket, label, entry))

    ax.axis("off")

    n_cols = 3
    n_rows = (len(column_groups) + n_cols - 1) // n_cols
    cell_w = 1.0 / n_cols
    cell_h = 0.92 / n_rows
    label_pad = 0.04
    qtype_indent = 0.025

    for idx, (bucket, label, entry) in enumerate(column_groups):
        col = idx % n_cols
        row = idx // n_cols
        x0 = col * cell_w
        y_top = 1.0 - row * cell_h

        ax.text(
            x0 + label_pad,
            y_top,
            label,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color=PALETTE[entry["pattern"]],
            transform=ax.transAxes,
        )
        ax.text(
            x0 + cell_w - label_pad,
            y_top,
            f"({entry['count']:,})",
            ha="right",
            va="top",
            fontsize=9.5,
            color="#333333",
            transform=ax.transAxes,
        )
        ax.plot(
            [x0 + label_pad, x0 + cell_w - label_pad],
            [y_top - 0.038, y_top - 0.038],
            color="#999999",
            linewidth=0.8,
            transform=ax.transAxes,
        )
        y_cursor = y_top - 0.06
        for q_label in QTYPE_LABEL.values():
            count = entry["by_qtype"].get(q_label, 0)
            if count == 0:
                continue
            ax.text(
                x0 + label_pad + qtype_indent,
                y_cursor,
                q_label,
                ha="left",
                va="top",
                fontsize=9,
                color="#222222",
                transform=ax.transAxes,
            )
            ax.text(
                x0 + cell_w - label_pad,
                y_cursor,
                f"{count:,}",
                ha="right",
                va="top",
                fontsize=9,
                color="#222222",
                transform=ax.transAxes,
            )
            y_cursor -= 0.05

    ax.text(
        1.0,
        -0.02,
        f"Total: {total:,} QA pairs",
        ha="right",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        transform=ax.transAxes,
    )


def render_figure(summary: dict, pdf_path: Path, png_path: Path) -> None:
    import matplotlib.pyplot as plt

    total = summary["total"]
    fig, (ax_donut, ax_table) = plt.subplots(
        1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.0, 1.6]}
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.06, wspace=0.05)

    _draw_donut(ax_donut, summary, total)
    _draw_breakdown_table(ax_table, summary, total)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tex-out", type=Path, default=DEFAULT_TEX_OUT)
    parser.add_argument("--pdf-out", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-out", type=Path, default=DEFAULT_PNG_OUT)
    args = parser.parse_args()

    stats = collect_stats(args.data_dir)
    summary = summarise(stats)

    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.write_text(render_tex(summary))
    render_figure(summary, args.pdf_out, args.png_out)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {args.tex_out}\nWrote: {args.pdf_out}\nWrote: {args.png_out}")


if __name__ == "__main__":
    main()
