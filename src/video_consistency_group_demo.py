"""Gradio demo: browse video consistency query groups.

Each group (group_id) asks the same spatial question about the same object(s)
from different viewpoints / trajectories.  All 6 clips in a group should
produce the same answer; consistency is measured by CV = std / mean.

Usage:
  python src/video_consistency_group_demo.py \
      --data /nas2/edwin/lmms-eval/data/video_consistency_production_merged.jsonl

  # With model results overlay:
  python src/video_consistency_group_demo.py \
      --data /nas2/edwin/lmms-eval/data/video_consistency_production_merged.jsonl \
      --results /path/to/samples_video_consistency_production.jsonl

  # Multiple result files (compare models):
  python src/video_consistency_group_demo.py \
      --data /nas2/edwin/lmms-eval/data/video_consistency_production_merged.jsonl \
      --results results_a.jsonl results_b.jsonl

Input spec:
  --data      JSONL with one row per clip (lmms-eval data file)
  --results   Optional lmms-eval results JSONL(s) for prediction overlay
  --port      Server port (default 7865; if busy, next free port is used)
              Use 0 to pick a free port starting at 7865.

Output spec:
  Launches Gradio server at 0.0.0.0:<port>
"""

import argparse
import json
import math
import socket
from collections import defaultdict
from pathlib import Path

import gradio as gr

COLS = 3
MAX_SLOTS = 30


def _find_free_port(start: int, span: int = 64) -> int:
    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    assert False, f"No free TCP port in {start}..{start + span - 1}"


def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    assert rows, f"Empty JSONL: {path}"
    return rows


def _load_data(path):
    """Load data JSONL and group by group_id."""
    rows = _load_jsonl(path)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    for gid in groups:
        groups[gid].sort(key=lambda r: r["clip_id"])
    return dict(groups)


def _load_results(paths):
    """Load lmms-eval results JSONL(s).  Returns {model_label: {clip_id: score_dict}}."""
    if not paths:
        return {}
    models = {}
    for p in paths:
        label = Path(p).stem
        lookup = {}
        for row in _load_jsonl(p):
            score = row.get("sceneshift_score", row)
            cid = score.get("clip_id", "")
            if cid:
                lookup[cid] = score
        if lookup:
            models[label] = lookup
    return models


def _question_text(q):
    return q.split("[Output]")[0].strip()


def _calculate_cv(values):
    """CV = std / mean.  Returns None if mean is zero or fewer than 2 values."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    mu = sum(vals) / len(vals)
    if mu == 0:
        return None
    variance = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance) / mu


def _error_color(pred, gt):
    try:
        p, g = float(pred), float(gt)
    except (TypeError, ValueError):
        return "#888"
    if abs(g) < 1e-9:
        return "#22c55e" if abs(p) < 0.05 else "#ef4444"
    err = abs(p - g) / abs(g)
    if err <= 0.25:
        return "#22c55e"
    if err <= 0.5:
        return "#eab308"
    return "#ef4444"


def _cv_color(cv):
    if cv is None:
        return "#888"
    if cv <= 0.1:
        return "#22c55e"
    if cv <= 0.3:
        return "#eab308"
    return "#ef4444"


def _header_md(entries, results_models):
    """Build header markdown for selected group."""
    s = entries[0]
    lines = [
        f"### {s['group_id']}",
        f"**Question:** {_question_text(s['question'])}",
        f"**Ground truth:** `{s['ground_truth']}`",
        f"**Type:** {s.get('question_type', '')} ({s.get('question_family', '')})",
        f"**Engine:** {s.get('engine', '')} | **Scene:** {s.get('scene_id', '')} | **Motion:** {s.get('motion_family', '')}",
        f"**Clips in group:** {len(entries)}",
    ]
    anchors = s.get("anchor_labels", [])
    if anchors:
        lines.append(f"**Anchors:** {', '.join(anchors)}")

    for model_label, lookup in results_models.items():
        preds = []
        for e in entries:
            sc = lookup.get(e["clip_id"])
            if sc:
                pp = sc.get("prediction_parse")
                try:
                    preds.append(float(pp))
                except (TypeError, ValueError):
                    preds.append(None)
        cv = _calculate_cv(preds)
        cv_str = f"{cv:.4f}" if cv is not None else "N/A"
        color = _cv_color(cv)
        lines.append(
            f"**[{model_label}]** CV = <span style='color:{color};font-weight:700'>{cv_str}</span>"
        )

    return "  \n".join(lines)


def _card_html(entry, results_models):
    """Build HTML card for one clip, optionally with predictions."""
    tag = entry.get("variation_tag", "") or entry.get("trajectory", "")
    direction = entry.get("direction", "")
    cycle = entry.get("cycle_count")
    radius = entry.get("radius")

    parts = [
        '<div style="text-align:center;padding:4px 0">',
        f'<div style="font-size:0.9em;font-weight:600">{tag}</div>',
    ]

    meta_bits = [b for b in [direction, entry.get("scene_id", "")] if b]
    if meta_bits:
        parts.append(f'<div style="font-size:0.78em;color:#666">{" | ".join(meta_bits)}</div>')

    detail_bits = []
    if cycle is not None:
        detail_bits.append(f"{int(cycle)}x")
    if radius is not None:
        detail_bits.append(f"r={radius}m")
    if detail_bits:
        parts.append(f'<div style="font-size:0.78em;color:#666">{" | ".join(detail_bits)}</div>')

    gt = entry["ground_truth"]
    for model_label, lookup in results_models.items():
        sc = lookup.get(entry["clip_id"])
        if sc is None:
            continue
        pred = sc.get("prediction_parse")
        cape = sc.get("CAPE")
        try:
            pred_text = f"{float(pred):.2f}"
        except (TypeError, ValueError):
            pred_text = str(pred) if pred is not None else "parse fail"
        color = _error_color(pred, gt)
        cape_text = f" | CAPE {cape:.2f}" if isinstance(cape, (int, float)) else ""
        parts.append(
            f'<div style="font-size:0.75em;color:#999">{model_label}:</div>'
            f'<div style="font-size:1.15em;font-weight:700;color:{color}">{pred_text}'
            f'<span style="font-size:0.65em;font-weight:400;color:#999">{cape_text}</span></div>'
        )

    parts.append("</div>")
    return "".join(parts)


def build_demo(data_path, results_paths):
    groups = _load_data(data_path)
    results_models = _load_results(results_paths)
    data_root = str(Path(data_path).parent.resolve())

    choice_labels = sorted(
        groups.keys(),
        key=lambda gid: (
            groups[gid][0].get("engine", ""),
            groups[gid][0].get("question_family", ""),
            groups[gid][0].get("question_type", ""),
            gid,
        ),
    )
    dropdown_choices = []
    for gid in choice_labels:
        s = groups[gid][0]
        label = f"{s.get('engine','')} | {s.get('question_type','')} | {gid}"
        dropdown_choices.append(label)
    label_to_gid = dict(zip(dropdown_choices, choice_labels))

    video_dirs = set()
    for clips in groups.values():
        for c in clips:
            vp = c.get("video_path", "")
            if vp:
                video_dirs.add(str(Path(vp).parent.resolve()))

    with gr.Blocks(title="Video Consistency Groups") as demo:
        gr.Markdown(
            "# Video Consistency — Query Group Browser\n"
            "Each **group** asks the same spatial question from different viewpoints.  \n"
            "Consistency = CV (std/mean) of model predictions within the group. **Lower CV = more consistent.**"
        )
        selector = gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0], label="Query Group")
        header = gr.Markdown("")

        video_slots = []
        card_slots = []
        for _ in range(0, MAX_SLOTS, COLS):
            with gr.Row():
                for _ in range(COLS):
                    with gr.Column(min_width=220):
                        video_slots.append(gr.Video(height=220, autoplay=False, loop=True, visible=False, show_label=False))
                        card_slots.append(gr.HTML("", visible=False))

        def update(selected):
            gid = label_to_gid.get(selected, "")
            entries = groups.get(gid, [])
            if not entries:
                outs = [""]
                for _ in range(MAX_SLOTS):
                    outs.extend([gr.update(value=None, visible=False), gr.update(value="", visible=False)])
                return outs

            shown = entries[:MAX_SLOTS]
            md = _header_md(shown, results_models)
            if len(entries) > MAX_SLOTS:
                md += f"\n\n*Showing {MAX_SLOTS} of {len(entries)} clips*"

            outs = [md]
            for i in range(MAX_SLOTS):
                if i < len(shown):
                    e = shown[i]
                    vp = e.get("video_path", "")
                    resolved = str(Path(vp).resolve()) if vp else None
                    outs.append(gr.update(value=resolved, visible=True))
                    outs.append(gr.update(value=_card_html(e, results_models), visible=True))
                else:
                    outs.append(gr.update(value=None, visible=False))
                    outs.append(gr.update(value="", visible=False))
            return outs

        all_outputs = [header]
        for v, c in zip(video_slots, card_slots):
            all_outputs.extend([v, c])
        selector.change(fn=update, inputs=[selector], outputs=all_outputs)
        demo.load(fn=update, inputs=[selector], outputs=all_outputs)

    allowed = list(video_dirs) + [data_root]
    return demo, allowed


def main():
    parser = argparse.ArgumentParser(description="Browse video consistency query groups")
    parser.add_argument("--data", required=True, help="JSONL data file (lmms-eval format)")
    parser.add_argument("--results", nargs="*", default=[], help="lmms-eval results JSONL(s)")
    parser.add_argument("--port", type=int, default=7865, help="0 = auto from 7865; if taken, next free")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    base = 7865 if args.port == 0 else args.port
    port = _find_free_port(base)
    if port != args.port and args.port != 0:
        print(f"Port {args.port} in use; using {port} instead.")

    demo, allowed = build_demo(args.data, args.results)
    demo.launch(server_name="0.0.0.0", server_port=port, share=args.share, allowed_paths=allowed)


if __name__ == "__main__":
    main()
