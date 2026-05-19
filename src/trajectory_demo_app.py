"""Gradio demo: compare Qwen3-VL-8B vs Gemini on identical spatial questions.

Usage:
    python src/trajectory_demo_app.py [--port PORT] [--share]

Input spec:
    Reads JSONL result files from lmms-eval for both models.

Output spec:
    Gradio web app showing orbit videos of a CoffeeTable in FloorPlan215
    from 4 starting angles, with model predictions displayed below.
"""

import argparse
import json
from pathlib import Path

import gradio as gr


SCENE = "FloorPlan215"
OBJECT_TYPE = "CoffeeTable"

QWEN_RESULTS = (
    "/nas2/edwin/lmms-eval/results/qwen3vl_8b_trajectory_demo_v2/"
    "Qwen__Qwen3-VL-8B-Instruct/"
    "20260319_021507_samples_trajectory_demo_v2.jsonl"
)
GEMINI_RESULTS_DIR = (
    "/nas2/edwin/lmms-eval/results/gemini_3p1_pro_trajectory_demo_v2/"
    "gemini-3.1-pro-preview"
)

VIDEO_BASE = (
    "/nas2/edwin/spatial-scene-variations/video_qa_output/"
    f"{SCENE}/around_cw_{OBJECT_TYPE}_{{var}}"
)

ANGLES = ["1x_start0", "1x_start90", "1x_start180", "1x_start270"]
ANGLE_LABELS = ["0\u00b0", "90\u00b0", "180\u00b0", "270\u00b0"]


def _discover_model_results() -> dict[str, str]:
    results = {"Qwen3-VL-8B": QWEN_RESULTS}
    gemini_dir = Path(GEMINI_RESULTS_DIR)
    if gemini_dir.is_dir():
        candidates = sorted(gemini_dir.glob("*trajectory_demo_v2*.jsonl"))
        if candidates:
            results["Gemini 3.1 Pro"] = str(candidates[-1])
    return results


def _load_results(path: str) -> dict:
    lookup: dict[tuple, dict] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            s = row.get("sceneshift_score", row)
            key = (
                s.get("input_type", ""),
                s.get("variation", ""),
                s.get("question_type", ""),
                s.get("question_id", ""),
            )
            lookup[key] = s
    return lookup


def _fmt_pred(score_dict: dict) -> str:
    pred = score_dict.get("prediction_parse")
    if pred is None:
        return "parse fail"
    if isinstance(pred, list):
        return "[" + ", ".join(
            f"{v:.1f}" if isinstance(v, float) else str(v) for v in pred
        ) + "]"
    if isinstance(pred, float):
        return f"{pred:.2f}"
    return str(pred)


def _error_color(pred_val, gt_val) -> str:
    if pred_val is None:
        return "color: #888"
    try:
        if isinstance(pred_val, list) and isinstance(gt_val, list):
            errors = []
            for p, g in zip(pred_val, gt_val):
                if abs(g) < 1e-6:
                    errors.append(1.0 if abs(p) > 0.05 else 0.0)
                else:
                    errors.append(abs(p - g) / abs(g))
            err = max(errors)
        else:
            p, g = float(pred_val), float(gt_val)
            if abs(g) < 1e-6:
                return "color: #ef4444"
            err = abs(p - g) / abs(g)
        if err <= 0.25:
            return "color: #22c55e"
        if err <= 0.5:
            return "color: #eab308"
        return "color: #ef4444"
    except (ValueError, TypeError):
        return "color: #888"


def _answer_md(score_dict: dict | None, gt_str: str) -> str:
    if score_dict is None:
        return '<div style="text-align:center; color:#888">N/A</div>'
    pred = _fmt_pred(score_dict)
    style = _error_color(
        score_dict.get("prediction_parse"),
        score_dict.get("ground_truth_parse"),
    )
    return (
        f'<div style="text-align:center">'
        f'<b style="{style}; font-size:1.2em">{pred}</b><br>'
        f'<span style="color:#888; font-size:0.8em">GT: {gt_str}</span>'
        f'</div>'
    )


def _build_question_choices(
    all_lookups: dict[str, dict],
) -> list[dict]:
    """Return list of {label, qtype, qid, gt, question_text}."""
    seen: set[str] = set()
    choices: list[dict] = []
    for lookup in all_lookups.values():
        for (_, _, qtype, qid), s in lookup.items():
            if qid in seen:
                continue
            if OBJECT_TYPE.lower() not in qid.lower():
                continue
            seen.add(qid)
            gt = s.get("ground_truth", "")
            question_text = s.get("question", "")
            if qtype == "object_dimensions":
                dim = s.get("dimension", "")
                label = f"{dim.capitalize()} of {OBJECT_TYPE} (GT: {gt}m)"
            elif qtype == "object_pair_distance_center":
                parts = qid.replace(
                    "object_pair_distance_center_", ""
                ).split("_AND_")
                if len(parts) == 2:
                    obj_a = parts[0].split("|")[0]
                    obj_b = parts[1].split("|")[0]
                    label = f"{obj_a}\u2013{obj_b} distance (GT: {gt}m)"
                else:
                    label = f"Distance (GT: {gt}m)"
            else:
                label = f"{qtype}: {qid[:40]}... (GT: {gt})"
            choices.append({
                "label": label, "qtype": qtype, "qid": qid,
                "gt": gt, "question_text": question_text,
            })
    choices.sort(key=lambda x: (x["qtype"], x["label"]))
    return choices


def _find_gt(all_lookups, qtype, qid):
    for lookup in all_lookups.values():
        for key, s in lookup.items():
            if key[2] == qtype and key[3] == qid:
                return s.get("ground_truth", "?")
    return "?"


def build_demo():
    model_results = _discover_model_results()
    all_lookups = {
        name: _load_results(path) for name, path in model_results.items()
    }
    model_names = list(model_results.keys())

    question_choices = _build_question_choices(all_lookups)
    choice_labels = [c["label"] for c in question_choices]
    label_to_choice = {c["label"]: c for c in question_choices}

    with gr.Blocks(title="Trajectory Inconsistency Demo") as demo:
        gr.Markdown(
            "# Spatial QA Inconsistency: "
            + " vs ".join(model_names) + "\n"
            f"Orbit videos of a **{OBJECT_TYPE}** in {SCENE} from 4 starting "
            "angles. The **same physical quantity** is asked in each column "
            "\u2014 answers should be identical but models give different "
            "answers depending on viewpoint.\n\n"
            "**Colors:** "
            '<span style="color:#22c55e">green</span> = \u226425% error, '
            '<span style="color:#eab308">yellow</span> = \u226450%, '
            '<span style="color:#ef4444">red</span> = >50%.'
        )

        question_selector = gr.Dropdown(
            choices=choice_labels,
            value=choice_labels[0] if choice_labels else None,
            label="Select Question",
            interactive=True,
        )

        question_display = gr.Markdown("")

        answer_components: list[gr.Markdown] = []

        gr.Markdown("### Videos (full orbit)")
        with gr.Row():
            for angle, alabel in zip(ANGLES, ANGLE_LABELS):
                vid_path = f"{VIDEO_BASE.format(var=angle)}/video.mp4"
                gr.Video(
                    value=vid_path, label=alabel,
                    autoplay=True, loop=True, height=220,
                )

        for model_name in model_names:
            gr.Markdown(f"#### {model_name} (video)")
            with gr.Row():
                for _ in ANGLES:
                    answer_components.append(gr.Markdown(""))

        gr.Markdown("---\n### Images (first frame)")
        with gr.Row():
            for angle, alabel in zip(ANGLES, ANGLE_LABELS):
                img_path = (
                    f"{VIDEO_BASE.format(var=angle)}/frames/frame_00000.png"
                )
                gr.Image(value=img_path, label=alabel, height=220)

        for model_name in model_names:
            gr.Markdown(f"#### {model_name} (image)")
            with gr.Row():
                for _ in ANGLES:
                    answer_components.append(gr.Markdown(""))

        n_models = len(model_names)
        n_angles = len(ANGLES)

        def update_answers(selected_label):
            empty = [""] * (n_models * 2 * n_angles)
            if selected_label not in label_to_choice:
                return [""] + empty
            choice = label_to_choice[selected_label]
            qtype, qid = choice["qtype"], choice["qid"]
            gt_str = _find_gt(all_lookups, qtype, qid)

            q_text = choice["question_text"]
            q_text_short = q_text.split("[Output]")[0].strip()
            question_md = (
                f'> **Question asked to the model:**\n>\n'
                f'> *{q_text_short}*\n>\n'
                f'> **Ground truth:** {gt_str}'
            )

            answers = []
            for modality in ["video", "image"]:
                for model_name in model_names:
                    lookup = all_lookups[model_name]
                    for angle in ANGLES:
                        key = (modality, angle, qtype, qid)
                        answers.append(
                            _answer_md(lookup.get(key), gt_str)
                        )
            return [question_md] + answers

        all_outputs = [question_display] + answer_components

        question_selector.change(
            fn=update_answers,
            inputs=[question_selector],
            outputs=all_outputs,
        )
        demo.load(
            fn=update_answers,
            inputs=[question_selector],
            outputs=all_outputs,
        )

        gr.Markdown("---\n### Summary Table")

        header_cols = ["Question", "GT"]
        for modality_tag in ["V", "I"]:
            for model_name in model_names:
                short = model_name.split("-")[0] if "-" in model_name else model_name[:6]
                for alabel in ANGLE_LABELS:
                    header_cols.append(f"{short} {modality_tag}:{alabel}")
        header = "| " + " | ".join(header_cols) + " |\n"
        sep = "|" + "|".join(["---"] * len(header_cols)) + "|\n"

        rows_md = ""
        for choice in question_choices:
            short_label = choice["label"].split(" (GT")[0]
            gt_str = _find_gt(all_lookups, choice["qtype"], choice["qid"])
            row = f"| {short_label} | {gt_str} |"
            for modality in ["video", "image"]:
                for model_name in model_names:
                    lookup = all_lookups[model_name]
                    for angle in ANGLES:
                        key = (modality, angle, choice["qtype"], choice["qid"])
                        s = lookup.get(key)
                        row += f" {_fmt_pred(s)} |" if s else " - |"
            rows_md += row + "\n"

        gr.Markdown(header + sep + rows_md)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    _, local, share = demo.launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share,
    )
    if share:
        with open("/tmp/gradio_share_url.txt", "w") as f:
            f.write(share)
        print(f"\n  Share URL: {share}\n", flush=True)
