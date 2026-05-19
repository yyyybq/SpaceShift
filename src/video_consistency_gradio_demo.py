"""Gradio demo: browse `video_consistency_output` from `qa.json` and rendered `videos/`.

Usage:
  PYTHONPATH=src python src/video_consistency_gradio_demo.py --output_dir ./video_consistency_output
  PYTHONPATH=src python src/video_consistency_gradio_demo.py --output_dir ./video_consistency_output --port 7864

Input spec:
  --output_dir  Benchmark root containing `qa.json` and `videos/<clip_id>/video.mp4`.

Output spec:
  Prints local URL for the Gradio server.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import gradio as gr

from taxonomy_demo_data import card_html, find_result, load_qa_json_entries, load_results, question_markdown


COLS = 3
MAX_SLOTS = 30


def _resolve_video(entry: dict, output_dir: Path) -> str:
    direct = output_dir / "videos" / entry["clip_id"] / "video.mp4"
    if direct.is_file():
        return str(direct.resolve())
    raw = Path(entry["video_path"]).expanduser()
    if raw.is_file():
        return str(raw.resolve())
    assert False, f"Missing video for {entry['clip_id']}: expected {direct}"


def _group_map(entries: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[e["group_id"]].append(e)
    for group_id in groups:
        groups[group_id].sort(key=lambda x: x["clip_id"])
    return dict(groups)


def _unique_clips(entries: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for e in entries:
        if e["clip_id"] in seen:
            continue
        seen.add(e["clip_id"])
        out.append(e)
    return out


def build_demo(*, output_dir: Path, results_jsonl: str | None):
    qa_path = output_dir / "qa.json"
    entries = load_qa_json_entries(qa_path)
    groups = _group_map(entries)
    choice_labels = sorted(groups.keys(), key=lambda gid: (groups[gid][0]["motion_family"], gid))
    results_lookup = load_results(results_jsonl)
    allowed = [str(output_dir.resolve())]

    with gr.Blocks(title="Video Consistency Output") as demo:
        gr.Markdown("# Video consistency dataset  \nBrowse each **group**: same QA across matched clips. Videos are read from `videos/<clip_id>/video.mp4`.")
        selector = gr.Dropdown(choices=choice_labels, value=choice_labels[0], label="Group")
        question_display = gr.Markdown("")
        video_slots = []
        card_slots = []
        for _ in range(0, MAX_SLOTS, COLS):
            with gr.Row():
                for _ in range(COLS):
                    with gr.Column(min_width=220):
                        video_slots.append(
                            gr.Video(height=200, autoplay=False, loop=True, visible=False, show_label=False)
                        )
                        card_slots.append(gr.HTML("", visible=False))

        def update(selected: str):
            if selected not in groups:
                outputs = [""]
                for _ in range(MAX_SLOTS):
                    outputs.extend([gr.update(value=None, visible=False), gr.update(value="", visible=False)])
                return outputs
            raw_list = groups[selected]
            shown_entries = _unique_clips(raw_list)
            if len(shown_entries) > MAX_SLOTS:
                shown_entries = shown_entries[:MAX_SLOTS]
            question_md = question_markdown(shown_entries, bool(results_lookup))
            if len(_unique_clips(raw_list)) > MAX_SLOTS:
                question_md += f"  \n**Showing:** {MAX_SLOTS} of {len(_unique_clips(raw_list))} clips"
            outputs = [question_md]
            for idx in range(MAX_SLOTS):
                if idx < len(shown_entries):
                    entry = shown_entries[idx]
                    path = _resolve_video(entry, output_dir)
                    outputs.append(gr.update(value=path, visible=True))
                    outputs.append(
                        gr.update(value=card_html(entry, find_result(entry, results_lookup)), visible=True)
                    )
                else:
                    outputs.append(gr.update(value=None, visible=False))
                    outputs.append(gr.update(value="", visible=False))
            return outputs

        outputs_list = [question_display]
        for v, c in zip(video_slots, card_slots):
            outputs_list.extend([v, c])
        selector.change(fn=update, inputs=[selector], outputs=outputs_list)
        demo.load(fn=update, inputs=[selector], outputs=outputs_list)

    return demo, allowed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./video_consistency_output")
    parser.add_argument("--results_jsonl", default=None)
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    demo, allowed_paths = build_demo(output_dir=output_dir, results_jsonl=args.results_jsonl)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, allowed_paths=allowed_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
