"""Convert dataset2_equivariance_flat.json into lmms-eval JSONL.

Description:
    Flattens the equivariance dataset into one JSON object per line and
    rewrites the absolute image paths from anjali's homedir mirror to a
    readable nas2 mirror. Emits one row per question (using the edited
    image as the visual). Adds `group_id` so consistency CV across
    translation distances/directions can be computed downstream.

Usage:
    python build_dataset.py \
        --input /nas2/sceneshift/dataset2_equivariance_flat.json \
        --output /nas2/edwin/lmms-eval/data/equivariance.jsonl

Input JSON (list of records, sample fields):
    record_id, scene_name, edit_type,
    original_image_path, edited_image_path,
    translation_direction, translation_distance,
    question, answer, question_type, question_id, primary_object

Output JSONL (one record per line):
    image_path, original_image_path, question, ground_truth,
    question_type, edit_type, scene_name, primary_object,
    translation_direction, translation_distance,
    record_id, question_id, group_id
"""

import argparse
import json
import os
from pathlib import Path

OLD_BASE = "/home/anjali/spatial-scene-variations/src/041624_translate_4/translation_equivariance/sceneshift/images"
NEW_BASE = "/nas2/sceneshift/translation_equivariance/sceneshift/images"


def remap_path(path: str, old_base: str, new_base: str) -> str:
    assert path.startswith(old_base), f"unexpected image path prefix: {path}"
    return new_base + path[len(old_base):]


def build_group_id(record: dict) -> str:
    return f"{record['scene_name']}|{record['primary_object']}|{record['question_id']}"


def convert_record(record: dict, old_base: str, new_base: str) -> dict:
    edited = remap_path(record["edited_image_path"], old_base, new_base)
    original = remap_path(record["original_image_path"], old_base, new_base)
    assert os.path.exists(edited), f"missing edited image: {edited}"
    return {
        "image_path": edited,
        "original_image_path": original,
        "question": record["question"],
        "ground_truth": record["answer"],
        "question_type": record["question_type"],
        "edit_type": record["edit_type"],
        "scene_name": record["scene_name"],
        "primary_object": record["primary_object"],
        "translation_direction": record["translation_direction"],
        "translation_distance": record["translation_distance"],
        "record_id": record["record_id"],
        "question_id": record["question_id"],
        "group_id": build_group_id(record),
    }


def main(input_path: Path, output_path: Path, old_base: str, new_base: str) -> None:
    assert input_path.exists(), f"input not found: {input_path}"
    with input_path.open() as f:
        data = json.load(f)
    assert isinstance(data, list) and data, "input JSON must be a non-empty list"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in data:
            converted = convert_record(record, old_base, new_base)
            f.write(json.dumps(converted) + "\n")

    print(f"wrote {len(data)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("/nas2/sceneshift/dataset2_equivariance_flat.json"))
    parser.add_argument("--output", type=Path, default=Path("/nas2/edwin/lmms-eval/data/equivariance.jsonl"))
    parser.add_argument("--old-base", default=OLD_BASE)
    parser.add_argument("--new-base", default=NEW_BASE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, args.old_base, args.new_base)
