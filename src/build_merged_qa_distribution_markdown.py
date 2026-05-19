"""Build `video_consistency_merged_qa_distributions.md` with pie + bar PNGs from `qa_merged_stats.json`.

Usage:
  PYTHONPATH=src python src/build_merged_qa_distribution_markdown.py \\
    --stats_json ./video_consistency_production_figures/qa_merged_stats.json \\
    --out_md ./docs/video_consistency_merged_qa_distributions.md \\
    --fig_dir ./docs/assets/merged_qa_distributions

All user-facing text (markdown and chart captions) is English.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MD_TITLES = dict(
    ln.strip().split("|", 1)
    for ln in """
qa_by.engine|Engine (QA-weighted)
qa_by.question_type|Question type (`question_type`)
qa_by.question_family|Question family (`question_family`)
qa_by.dimension|Dimension (size QA only)
qa_by.anchor_kind|Anchor kind (`anchor_kind`)
qa_by.motion_family|Motion family (`motion_family`)
qa_by.trajectory|Trajectory (`trajectory`)
qa_by.direction|Direction (`direction`)
qa_by.variation_tag|Variation tag (`variation_tag`, raw string)
qa_by.variation_cycle|Variation cycle (parsed prefix)
qa_by.variation_suffix|Variation suffix (after `_`)
qa_by.scene_id_per_qa|Scene ID (`scene_id`, one row per QA)
qa_by.object_or_pair_label|Object / pair label (anchors)
joint_qa.engine_question_type|Joint: engine × question type
joint_qa.engine_question_family|Joint: engine × question family
joint_qa.engine_motion_family|Joint: engine × motion family
joint_qa.engine_dimension|Joint: engine × dimension (size QA)
joint_qa.question_type_motion_family|Joint: question type × motion family
joint_qa.engine_variation_cycle|Joint: engine × variation cycle
joint_qa.engine_variation_suffix|Joint: engine × variation suffix
clip_level.group_id|Group ID (`group_id`, clip-level; 6 clips per group)
""".strip().splitlines()
)

_FIG_EN = {
    "object_dimensions": "Object size",
    "object_pair_distance_center": "Pair distance",
    "object_distance_to_camera": "Camera–object dist.",
    "size": "Size",
    "pair_distance": "Pair dist.",
    "camera_distance": "Cam. dist.",
    "object": "Single object",
    "pair": "Object pair",
    "rotation": "Rotation",
    "around": "Around",
    "approach": "Approach",
    "passby": "Passby",
    "spherical": "Spherical",
    "thor": "THOR",
    "interiorgs": "InteriorGS",
    "cw": "cw",
    "ccw": "ccw",
    "fw": "fw",
    "bw": "bw",
}


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def _lab(k: str) -> str:
    return _FIG_EN.get(k, str(k))


def _maps(sec: str) -> tuple:
    if "question_type" in sec and "joint" not in sec:
        return (_FIG_EN,)
    if "question_family" in sec and "joint" not in sec:
        return (_FIG_EN,)
    if sec.endswith("anchor_kind"):
        return (_FIG_EN,)
    if "motion_family" in sec and "joint" not in sec:
        return (_FIG_EN,)
    if sec.endswith("engine") and "joint" not in sec:
        return (_FIG_EN,)
    if sec.endswith("direction"):
        return (_FIG_EN,)
    if "engine_question" in sec or "engine_motion" in sec:
        return (_FIG_EN,)
    if "engine_dimension" in sec or "engine_variation" in sec:
        return (_FIG_EN,)
    if "question_type_motion" in sec:
        return (_FIG_EN,)
    return ({},)


def _norm_rows(raw: object) -> list[dict]:
    assert isinstance(raw, list)
    out: list[dict] = []
    for r in raw:
        assert isinstance(r, dict) and "key" in r and "count" in r
        k = r["key"]
        key = str(k) if k is not None else ""
        out.append({"key": key, "count": int(r["count"]), "share_pct": float(r.get("share_pct", 0))})
    return out


def _lbl_row(r: dict, maps: tuple) -> str:
    for m in maps:
        if r["key"] in m:
            return m[r["key"]]
    return str(r["key"])


def _plot_combo(rows: list[dict], fig_title_en: str, out_path: Path, maps: tuple) -> None:
    rows = sorted(rows, key=lambda x: (-x["count"], x["key"]))
    total = sum(r["count"] for r in rows)
    assert total > 0
    cap = 8
    if len(rows) > cap:
        head = rows[: cap - 1]
        rest_n = sum(r["count"] for r in rows[cap - 1 :])
        pie_rows = head + [{"key": "__other__", "count": rest_n, "share_pct": 0.0}]
    else:
        pie_rows = list(rows)

    def lp(r):
        return "Other" if r["key"] == "__other__" else _lbl_row(r, maps)

    labels_p = [lp(r) for r in pie_rows]
    sizes_p = [r["count"] for r in pie_rows]
    _rc()
    fig, (axp, axb) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = plt.cm.tab20.colors
    axp.pie(
        sizes_p,
        labels=labels_p,
        autopct=lambda pct: f"{pct:.0f}%\n({int(round(pct / 100 * total))})",
        colors=colors[: len(sizes_p)],
        pctdistance=0.72,
        labeldistance=1.04,
        textprops={"fontsize": 8},
    )
    axp.set_title("Pie (merged tail → Other)")
    labels_b = [_lbl_row(r, maps) for r in rows]
    vals = [r["count"] for r in rows]
    y = range(len(labels_b))
    axb.barh(list(y), vals, color="#2c7fb8", height=0.7)
    axb.set_yticks(list(y), labels_b, fontsize=7)
    axb.invert_yaxis()
    axb.set_xlabel("Count")
    axb.set_title("Horizontal bar (all categories)")
    fig.suptitle(f"{fig_title_en} — N={total}", fontsize=12, y=1.03)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _joint_lbl(k: str) -> str:
    a, _, b = k.partition("|")
    if not b:
        return k
    return f"{_lab(a)} × {_lab(b)}"


def _write_md(stats: dict, md_path: Path, fig_rel: str, fig_names: list[tuple[str, str, str]]) -> None:
    lines = [
        "# Merged Thor + InteriorGS QA distributions",
        "",
        "Source: `qa_merged_stats.json` produced by `merge_production_qa_report.py` after merging the two production `qa.json` files.",
        "Each section has a **pie chart** (left; if more than 8 categories, the tail is merged as *Other*) and a **horizontal bar chart** (right; all categories).",
        "",
        "## Summary",
        "",
    ]
    sm = stats["summary"]
    lines += [
        f"- **Thor clips**: {sm['thor_clips']}",
        f"- **InteriorGS clips**: {sm['interiorgs_clips']}",
        f"- **Merged clips**: {sm['merged_clips']}",
        f"- **Merged QA instances**: {sm['merged_qa_instances']}",
    ]
    for row in sm.get("questions_per_clip", []):
        lines.append(f"- **Questions per clip = {row['key']}**: {row['count']} clips ({row['share_pct']:.1f}%)")
    lines += [
        "",
        "## Takeaways",
        "",
        "- **Scale**: 264 QA rows paired 1:1 with 264 clips; Thor and InteriorGS each contribute half.",
        "- **Question types**: about **48%** object size (L/W/H), **32%** inter-object distance under rotation, **20%** camera–object distance.",
        "- **Motion**: rotation-dominated, then *around*; *approach*, *passby*, and *spherical* are each about **14%** of QA-weighted clips.",
        "- **Variations**: raw `variation_tag` mixes bare `1x`/`2x` with `start*` / `angle*` styles across engines; parsed cycles are mostly 1x/2x, with 3x about **14%**.",
        "- **Scenes & groups**: 34 distinct `scene_id` values; 44 `group_id` values with exactly 6 clips each (multi-view / variant sets per consistency group).",
        "",
    ]
    for section_key, fname, blurb in fig_names:
        title = _MD_TITLES.get(section_key, section_key)
        lines.append(f"## {title}")
        lines.append("")
        if blurb:
            lines.append(blurb)
            lines.append("")
        lines.append(f"![{title}]({fig_rel}/{fname}.png)")
        lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_json", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    ap.add_argument("--fig_dir", type=Path, required=True)
    args = ap.parse_args()
    stats = json.loads(args.stats_json.expanduser().resolve().read_text(encoding="utf-8"))
    fig_dir = args.fig_dir.expanduser().resolve()
    md_path = args.out_md.expanduser().resolve()
    fig_names: list[tuple[str, str, str]] = []

    _en_title = {
        "engine": "Engine",
        "question_type": "Question type",
        "question_family": "Question family",
        "dimension": "Dimension (size QA)",
        "anchor_kind": "Anchor kind",
        "motion_family": "Motion family",
        "trajectory": "Trajectory",
        "direction": "Direction",
        "variation_tag": "Variation tag (raw)",
        "variation_cycle": "Variation cycle (parsed)",
        "variation_suffix": "Variation suffix",
        "scene_id_per_qa": "Scene ID (per QA)",
        "object_or_pair_label": "Object / pair label",
    }

    for key, block in stats["qa_by"].items():
        sk = f"qa_by.{key}"
        fn = f"qa_by_{key}"
        _plot_combo(_norm_rows(block), _en_title.get(key, key), fig_dir / f"{fn}.png", _maps(sk))
        fig_names.append((sk, fn, ""))

    _jen = {
        "engine_question_type": "Joint: engine × question type",
        "engine_question_family": "Joint: engine × question family",
        "engine_motion_family": "Joint: engine × motion",
        "engine_dimension": "Joint: engine × dimension",
        "question_type_motion_family": "Joint: Q-type × motion",
        "engine_variation_cycle": "Joint: engine × cycle",
        "engine_variation_suffix": "Joint: engine × suffix",
    }
    for key, block in stats["joint_qa"].items():
        sk = f"joint_qa.{key}"
        fn = f"joint_{key}"
        rows = [{"key": _joint_lbl(r["key"]), "count": r["count"], "share_pct": r["share_pct"]} for r in _norm_rows(block)]
        _plot_combo(rows, _jen.get(key, key), fig_dir / f"{fn}.png", ({}))
        fig_names.append((sk, fn, "Stats keys are `engine|...`; plot labels use the English legend above."))

    sk = "clip_level.group_id"
    fn = "clip_level_group_id"
    rows = _norm_rows(stats["clip_level"]["group_id"])
    short = [{"key": (r["key"][:52] + "...") if len(r["key"]) > 54 else r["key"], "count": r["count"], "share_pct": r["share_pct"]} for r in rows]
    _plot_combo(short, "Group ID (trunc.)", fig_dir / f"{fn}.png", ({}))
    fig_names.append((sk, fn, "44 groups, 6 clips each; bar labels show truncated `group_id` strings."))

    try:
        rel = fig_dir.resolve().relative_to(md_path.parent.resolve()).as_posix()
    except ValueError:
        rel = "assets/merged_qa_distributions"
    _write_md(stats, md_path, rel, fig_names)
    print(f"Wrote figures under {fig_dir}\nWrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
