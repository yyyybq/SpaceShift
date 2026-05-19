"""
Curate the scene_variation eval JSONL from anjali's combined rotate+translate+remove dataset.

Usage:
    python scripts/scene_variation/curate_data.py

Input:
    /nas2/sceneshift/data_041726_scene_variation/aithor_scene_variation_eval_042226.json
    A list of records where each QA has an explicit `edit_type` in
    {'rotate', 'translate', 'remove'}.
    Image paths are preserved exactly as provided in input records.

Output:
    /nas2/edwin/lmms-eval/data/scene_variation.jsonl
    One JSON per line with unified keys:
        image_path (the image to score on, unchanged from input),
        original_image_path (unchanged from input, or None for remove),
        question, ground_truth (str), question_type, edit_type,
        scene_name, question_id, primary_object, image_id, group_id.
    No `record_id` per line; passes through existing group_id from input.
"""

import json
from pathlib import Path

INPUT_JSON = Path("/nas2/sceneshift/data_041726_scene_variation/aithor_scene_variation_eval_042226.json")
OUTPUT_JSONL = Path("/nas2/edwin/lmms-eval/data/scene_variation_042226.jsonl")

EDIT_TYPE_ROTATE = "rotate"
EDIT_TYPE_TRANSLATE = "translate"
EDIT_TYPE_REMOVE = "remove"

def _classify(record: dict) -> str:
    edit_type = record.get("edit_type")
    valid_edit_types = {EDIT_TYPE_ROTATE, EDIT_TYPE_TRANSLATE, EDIT_TYPE_REMOVE}
    if edit_type not in valid_edit_types:
        raise ValueError(
            f"Invalid or missing edit_type={edit_type!r}; expected one of {sorted(valid_edit_types)}"
        )
    return edit_type

def _extract_scene_name(record: dict, edit_type: str) -> str:
    if record.get("scene_name"):
        return record["scene_name"]
    if edit_type == EDIT_TYPE_REMOVE:
        parts = record["image_path"].lstrip("./").split("/")
        for part in parts:
            if part.startswith("FloorPlan"):
                return part
    raise ValueError(f"Cannot extract scene_name: {record}")

def _make_image_id(record: dict) -> str:
    if record.get("image_id"):
        return str(record["image_id"])
    path = record.get("edited_image_path") or record.get("image_path") or ""
    parts = path.lstrip("./").split("/")
    return "_".join(parts[-3:]).replace(".png", "")

def _build_entry(record: dict) -> dict:
    edit_type = _classify(record)

    if edit_type == EDIT_TYPE_REMOVE:
        scoring_rel = record["image_path"]
        original_rel = None
        gt = record["ground_truth"]
    else:
        scoring_rel = record["edited_image_path"]
        original_rel = record["original_image_path"]
        gt = record["answer"]

    entry = {
        "image_path": scoring_rel,
        "original_image_path": original_rel if original_rel else None,
        "question": record["question"].strip(),
        "ground_truth": str(gt),
        "question_type": record["question_type"],
        "edit_type": edit_type,
        "scene_name": _extract_scene_name(record, edit_type),
        "question_id": record.get("question_id"),
        "primary_object": record.get("primary_object"),
        "image_id": _make_image_id(record),
        "group_id": record.get("group_id"),  # pass-through
        "translation_direction": record.get("translation_direction"),
        "translation_distance": record.get("translation_distance"),
        "scene_edit_id": record.get("scene_edit_id"),
        "camera_id": record.get("camera_id"),
    }
    return entry

def main() -> None:
    with INPUT_JSON.open() as fp:
        records = json.load(fp)
    assert records, f"Empty input: {INPUT_JSON}"

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with OUTPUT_JSONL.open("w") as fp:
        for record in records:
            entry = _build_entry(record)
            fp.write(json.dumps(entry) + "\n")
            counts[entry["edit_type"]] = counts.get(entry["edit_type"], 0) + 1

    total = sum(counts.values())
    print(f"Wrote {total} entries to {OUTPUT_JSONL}")
    for edit_type, count in sorted(counts.items()):
        print(f"  {edit_type}: {count}")

if __name__ == "__main__":
    main()
