"""Build a focused JSONL eval dataset for the trajectory inconsistency demo.

Usage:
    python src/build_trajectory_demo_jsonl.py

Input spec:
    Reads qa.json from video_qa_output/ for around_cw CoffeeTable variations
    in FloorPlan215 (4 start angles + 2x bounce).

Output spec:
    /nas2/edwin/lmms-eval/data/trajectory_demo_v2.jsonl
    Each line: {"video_path"|"image_path", "question", "ground_truth",
                "question_type", "input_type", "variation", "trajectory",
                "scene", "object_type", "dimension", "question_id"}

    object_size questions are split into separate scalar questions per
    dimension (length, width, height) instead of a single vector question.
"""

import json
from pathlib import Path


SCENE = "FloorPlan215"
VIDEO_QA_ROOT = Path("/nas2/edwin/spatial-scene-variations/video_qa_output")
OUTPUT_PATH = Path("/nas2/edwin/lmms-eval/data/trajectory_demo_v2.jsonl")

DEMO_CONFIGS = [
    {
        "trajectory": "around_cw",
        "object_type": "CoffeeTable",
        "target_object_id": "CoffeeTable|-02.49|+00.02|+04.04",
        "variations": ["1x_start0", "1x_start90", "1x_start180", "1x_start270", "2x_start0"],
        "dir_pattern": "around_cw_CoffeeTable_{var}",
    },
    {
        "trajectory": "approach_fw",
        "object_type": "FloorLamp",
        "target_object_id": "FloorLamp|-04.62|+00.02|+00.31",
        "variations": ["angle0", "angle90"],
        "dir_pattern": "approach_fw_FloorLamp_{var}",
    },
]

KEEP_QUESTION_TYPES = {
    "object_size",
    "object_pair_distance_center",
}

SIZE_QUESTION_TEMPLATE = (
    "What is the estimated {dimension} of the {object} in this video in meters? "
    "Length and width are the two dimensions that define the 'base' of the object. "
    "Of these two base dimensions, let length be the longer and width be the shorter."
)

POST_PROMPT = (
    " [Output]\n"
    "You have to end your response with the answer formatted in a "
    "dictionary: {{'answer': <answer>}}. For example, "
    "{{'answer': 'Z'}} or {{'answer': 0}} or "
    "{{'answer': [0, 0, 0]}}, depending on the question."
)

DIMENSION_INDEX = {"length": 0, "width": 1, "height": 2}


def _parse_size_gt(answer_str: str) -> dict[str, float] | None:
    """Parse '[1.4, 0.9, 0.5]' -> {'length': 1.4, 'width': 0.9, 'height': 0.5}."""
    try:
        vals = json.loads(answer_str)
        if isinstance(vals, list) and len(vals) == 3:
            return {"length": vals[0], "width": vals[1], "height": vals[2]}
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _split_size_qa(qa: dict, object_type: str) -> list[dict]:
    """Split a vector size QA into 3 scalar questions."""
    dims = _parse_size_gt(qa["answer"])
    if dims is None:
        return []
    results = []
    base_qid = qa.get("question_id", "")
    for dim_name, gt_val in dims.items():
        question = SIZE_QUESTION_TEMPLATE.format(
            dimension=dim_name, object=object_type,
        ) + POST_PROMPT
        results.append({
            "question": question,
            "ground_truth": str(gt_val),
            "question_type": "object_dimensions",
            "question_id": f"{base_qid}_{dim_name}",
            "dimension": dim_name,
        })
    return results


def _entries_for_dir(
    traj_dir: Path,
    trajectory: str,
    variation: str,
    target_object_id: str,
    object_type: str,
) -> list[dict]:
    qa_path = traj_dir / "qa.json"
    video_path = traj_dir / "video.mp4"
    first_frame = traj_dir / "frames" / "frame_00000.png"
    if not qa_path.exists():
        return []

    qa_pairs = json.loads(qa_path.read_text())
    entries = []
    seen_qids: set[str] = set()

    for qa in qa_pairs:
        if qa["question_type"] not in KEEP_QUESTION_TYPES:
            continue
        if target_object_id not in qa.get("primary_object", ""):
            continue
        orig_qid = qa.get("question_id", "")
        if orig_qid in seen_qids:
            continue
        seen_qids.add(orig_qid)

        if qa["question_type"] == "object_size":
            scalar_qas = _split_size_qa(qa, object_type)
            for sq in scalar_qas:
                base = {
                    "question": sq["question"],
                    "ground_truth": sq["ground_truth"],
                    "question_type": sq["question_type"],
                    "question_id": sq["question_id"],
                    "trajectory": trajectory,
                    "variation": variation,
                    "scene": SCENE,
                    "object_type": object_type,
                    "dimension": sq["dimension"],
                }
                if video_path.exists():
                    e = dict(base)
                    e["video_path"] = str(video_path)
                    e["input_type"] = "video"
                    entries.append(e)
                if first_frame.exists():
                    e = dict(base)
                    e["image_path"] = str(first_frame)
                    e["input_type"] = "image"
                    entries.append(e)
        else:
            base = {
                "question": qa["question"],
                "ground_truth": qa["answer"],
                "question_type": qa["question_type"],
                "question_id": orig_qid,
                "trajectory": trajectory,
                "variation": variation,
                "scene": SCENE,
                "object_type": object_type,
                "dimension": "distance",
            }
            if video_path.exists():
                e = dict(base)
                e["video_path"] = str(video_path)
                e["input_type"] = "video"
                entries.append(e)
            if first_frame.exists():
                e = dict(base)
                e["image_path"] = str(first_frame)
                e["input_type"] = "image"
                entries.append(e)

    return entries


def main():
    all_entries: list[dict] = []
    for cfg in DEMO_CONFIGS:
        for var in cfg["variations"]:
            traj_dir = VIDEO_QA_ROOT / SCENE / cfg["dir_pattern"].format(var=var)
            if not traj_dir.exists():
                print(f"  skip missing: {traj_dir.name}")
                continue
            entries = _entries_for_dir(
                traj_dir, cfg["trajectory"], var,
                cfg["target_object_id"], cfg["object_type"],
            )
            all_entries.extend(entries)
            print(f"  {cfg['trajectory']} {var}: {len(entries)} entries")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWrote {len(all_entries)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
