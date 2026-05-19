"""Gradio demo: browse one benchmark QA across multiple matched videos.

Usage:
    python src/taxonomy_demo_app.py [--clips_jsonl PATH | --dataset_manifest PATH]
        [--min_videos N] [--results_jsonl PATH] [--port PORT] [--share]

Input spec:
    `clips_jsonl` is the video consistency benchmark output (JSONL).
    `dataset_manifest` is a `dataset_manifest.json` with per-clip `qa.json`;
    QA rows are paired to each video via `primary_object` == manifest `object_id`
    (rotation + `object_id` room uses scene-level pair / distance-comparison types only).
    `results_jsonl` is optional lmms-eval samples JSONL for prediction overlays.
    `min_videos` defaults to 2 for JSONL and 1 for manifest (override with flag).

Output spec:
    Gradio web app with one QA page at a time, showing the same question and
    ground truth across matched videos.
"""

import argparse
from pathlib import Path

import gradio as gr

from taxonomy_demo_data import build_choice_map, card_html, find_result, load_multi_video_groups, load_results, question_markdown
from taxonomy_manifest_loader import load_multi_video_groups_from_manifest

DEFAULT_CLIPS_JSONL = "/nas2/edwin/spatial-scene-variations/video_consistency_output/clips.jsonl"
COLS = 3
MAX_SLOTS = 12


def _repo_root_from_manifest(manifest_path: Path) -> Path:
    cur = manifest_path.resolve().parent
    for _ in range(10):
        if (cur / "src").is_dir():
            return cur
        cur = cur.parent
    return manifest_path.resolve().parent


def build_demo(*, clips_jsonl=None, dataset_manifest=None, results_jsonl=None, min_videos=2):
    assert (clips_jsonl is not None) ^ (dataset_manifest is not None), (
        "Set exactly one of clips_jsonl or dataset_manifest"
    )
    if dataset_manifest:
        groups = load_multi_video_groups_from_manifest(dataset_manifest, min_videos=min_videos)
        choice_labels, label_to_key = build_choice_map(groups, sort_by_video_count_desc=True)
        mp = Path(dataset_manifest).expanduser().resolve()
        repo_root = _repo_root_from_manifest(mp)
        allowed_paths = [str(repo_root), str(mp.parent), "/nas2/edwin/spatial-scene-variations"]
    else:
        groups = load_multi_video_groups(clips_jsonl, min_videos=min_videos)
        choice_labels, label_to_key = build_choice_map(groups)
        allowed_paths = [
            str(Path(clips_jsonl).expanduser().resolve().parent),
            "/nas2/edwin/spatial-scene-variations",
        ]
    results_lookup = load_results(results_jsonl)

    with gr.Blocks(title="Video Consistency QA Browser") as demo:
        gr.Markdown(
            "# Video Consistency QA Browser\n"
            "Each page shows one QA slice: the same question and ground truth across matched videos. "
            "With `--dataset_manifest`, clips use strict QA–video pairing from each folder's `qa.json`. "
            "Pass `--results_jsonl` to show model predictions below each clip."
        )
        question_selector = gr.Dropdown(
            choices=choice_labels,
            value=choice_labels[0],
            label="Select QA Page",
            interactive=True,
        )
        question_display = gr.Markdown("")
        video_slots = []
        card_slots = []
        for _ in range(0, MAX_SLOTS, COLS):
            with gr.Row():
                for _ in range(COLS):
                    with gr.Column(min_width=240):
                        video_slots.append(
                            gr.Video(height=220, autoplay=True, loop=True, visible=False, show_label=False)
                        )
                        card_slots.append(gr.HTML("", visible=False))

        def update(selected_label):
            if selected_label not in label_to_key:
                outputs = [""]
                for _ in range(MAX_SLOTS):
                    outputs.extend([gr.update(value=None, visible=False), gr.update(value="", visible=False)])
                return outputs
            entries = groups[label_to_key[selected_label]]
            shown_entries = entries[:MAX_SLOTS]
            question_md = question_markdown(entries, bool(results_lookup))
            if len(entries) > MAX_SLOTS:
                question_md += f"  \n**Showing:** {MAX_SLOTS} of {len(entries)} videos"
            outputs = [question_md]
            for index in range(MAX_SLOTS):
                if index < len(shown_entries):
                    entry = shown_entries[index]
                    outputs.append(gr.update(value=entry["video_path"], visible=True))
                    outputs.append(gr.update(value=card_html(entry, find_result(entry, results_lookup)), visible=True))
                else:
                    outputs.append(gr.update(value=None, visible=False))
                    outputs.append(gr.update(value="", visible=False))
            return outputs

        outputs = [question_display]
        for video_slot, card_slot in zip(video_slots, card_slots):
            outputs.extend([video_slot, card_slot])
        question_selector.change(fn=update, inputs=[question_selector], outputs=outputs)
        demo.load(fn=update, inputs=[question_selector], outputs=outputs)

    return demo, allowed_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_jsonl", default=None)
    parser.add_argument("--dataset_manifest", default=None)
    parser.add_argument("--min_videos", type=int, default=None)
    parser.add_argument("--results_jsonl", default=None)
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    if args.dataset_manifest:
        min_videos = 1 if args.min_videos is None else args.min_videos
        demo, allowed_paths = build_demo(
            dataset_manifest=args.dataset_manifest,
            results_jsonl=args.results_jsonl,
            min_videos=min_videos,
        )
    else:
        clips = args.clips_jsonl or DEFAULT_CLIPS_JSONL
        min_videos = 2 if args.min_videos is None else args.min_videos
        demo, allowed_paths = build_demo(
            clips_jsonl=clips,
            results_jsonl=args.results_jsonl,
            min_videos=min_videos,
        )
    _, _, share = demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        allowed_paths=allowed_paths,
    )
    if share:
        with open("/tmp/gradio_share_url.txt", "w", encoding="utf-8") as handle:
            handle.write(share)
        print(f"\n  Share URL: {share}\n", flush=True)
