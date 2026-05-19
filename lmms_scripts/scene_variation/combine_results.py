"""
Combine per-model sample JSONLs from scene_variation + full_benchmark into one
big JSON per model, written to /nas2/edwin/lmms-eval/results/combined/.

Usage:
    python scripts/scene_variation/combine_results.py
    python scripts/scene_variation/combine_results.py --output-dir /custom/path

Input:
    /nas2/edwin/lmms-eval/results/<model_dir>/.../*samples_<task>.jsonl
        Per-task per-model lmms_eval sample logs. <task> is one of
        scene_variation, image_consistency_thor_eval_v2,
        video_consistency_thor_eval_v2.

Output:
    /nas2/edwin/lmms-eval/results/combined/<model_name>.json
        {
          "model": "<model_name>",
          "task_files": {<task>: <abs_path_to_sample_jsonl>, ...},
          "samples": [<sample_dict>, ...]   # all samples; each sample has an
                                            # added "_task" field identifying the
                                            # source task.
        }

Model registry below maps a canonical model_name -> list of result dir prefixes
to scan under /nas2/edwin/lmms-eval/results.
"""

import argparse
import json
from pathlib import Path

RESULTS_ROOT = Path("/nas2/edwin/lmms-eval/results")
TASKS = ("scene_variation", "image_consistency_thor_eval_v2", "video_consistency_thor_eval_v2")

MODEL_DIR_REGISTRY: dict[str, list[str]] = {
    "cambrians_1p5b": ["cambrians_1p5b_scene_variation", "cambrians_1p5b_full"],
    "cambrians_3b": ["cambrians_3b_scene_variation", "cambrians_3b_full"],
    "cambrians_7b": ["cambrians_7b_scene_variation", "cambrians_7b_full"],
    "internvl3p5_2b": ["internvl3p5_2b_scene_variation", "internvl3p5_2b_full"],
    "internvl3p5_8b": ["internvl3p5_8b_scene_variation", "internvl3p5_8b_full"],
    "qwen3vl_2b": ["qwen3vl_2b_scene_variation", "qwen3vl_2b_full"],
    "qwen3vl_4b": ["qwen3vl_4b_scene_variation", "qwen3vl_4b_full"],
    "qwen3vl_8b": ["qwen3vl_8b_scene_variation", "qwen3vl_8b_full"],
    "qwen2_5_vl_3b": ["qwen2_5_vl_3b_full"],
    "qwen2_5_vl_7b": ["qwen2_5_vl_7b_full", "qwen2_5_vl_7b_scene_variation"],
    "llava_onevision_0p5b": ["llava_onevision_0p5b_full_benchmark"],
    "llava_onevision_7b": ["llava_onevision_7b_full_benchmark"],
    "gemini_3_1_pro_preview": ["gemini_3_1_pro_preview_scene_variation", "gemini_3_1_pro_preview_full_benchmark"],
    "gemini_3_flash_preview": ["gemini_3_flash_preview_scene_variation", "gemini_3_flash_preview_with_thinking_full_benchmark"],
    "gpt5p2": ["gpt5p2_scene_variation", "gpt5p2_full_benchmark"],
    "gpt5_mini": ["gpt5_mini_scene_variation", "gpt5_mini_full_benchmark"],
}


def _find_latest_samples_jsonl(result_dir: Path, task: str) -> Path | None:
    pattern = f"*samples_{task}.jsonl"
    matches = sorted(result_dir.rglob(pattern))
    if not matches:
        return None
    return matches[-1]


def _load_samples(jsonl_path: Path, task: str) -> list[dict]:
    samples = []
    with jsonl_path.open() as f:
        for line in f:
            obj = json.loads(line)
            obj["_task"] = task
            samples.append(obj)
    return samples


def _combine_for_model(model_name: str, dir_prefixes: list[str]) -> dict | None:
    task_files: dict[str, str] = {}
    samples: list[dict] = []
    for prefix in dir_prefixes:
        result_dir = RESULTS_ROOT / prefix
        if not result_dir.exists():
            continue
        for task in TASKS:
            jsonl_path = _find_latest_samples_jsonl(result_dir, task)
            if jsonl_path is None:
                continue
            existing = task_files.get(task)
            if existing is not None:
                print(f"  WARN {model_name}/{task}: multiple files; keeping latest {jsonl_path}")
            task_files[task] = str(jsonl_path)
            samples.extend(_load_samples(jsonl_path, task))

    if not samples:
        return None

    return {
        "model": model_name,
        "task_files": task_files,
        "n_samples": len(samples),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "combined"))
    parser.add_argument("--models", nargs="*", default=list(MODEL_DIR_REGISTRY.keys()))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for model in args.models:
        prefixes = MODEL_DIR_REGISTRY.get(model)
        assert prefixes, f"Unknown model name: {model!r}. Add it to MODEL_DIR_REGISTRY."
        bundle = _combine_for_model(model, prefixes)
        if bundle is None:
            print(f"[SKIP] {model}: no sample files found")
            continue
        out_path = out_dir / f"{model}.json"
        with out_path.open("w") as f:
            json.dump(bundle, f)
        per_task_counts = {}
        for s in bundle["samples"]:
            per_task_counts[s["_task"]] = per_task_counts.get(s["_task"], 0) + 1
        per_task_str = ", ".join(f"{k}={v}" for k, v in per_task_counts.items())
        summary.append((model, bundle["n_samples"], per_task_str, str(out_path)))
        print(f"[OK] {model}: {bundle['n_samples']:>6} samples ({per_task_str}) -> {out_path}")

    print()
    print(f"Wrote {len(summary)} combined files under {out_dir}")


if __name__ == "__main__":
    main()
