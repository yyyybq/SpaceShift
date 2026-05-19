"""
Align scene_variation_042226.jsonl with the v2 sceneshift schema so the same
sceneshift_doc_to_text (auto-built short numeric-only prompt) renders for both.

Usage:
    python scripts/scene_variation/align_post_prompt.py

Input:
    /nas2/edwin/lmms-eval/data/scene_variation_042226.jsonl
        Original entries with verbose CoT-style questions and dict-format
        post prompts. question_type in {object_distance_to_camera, object_size,
        object_pair_distance_center}.

Output:
    /nas2/edwin/lmms-eval/data/scene_variation_042226_aligned.jsonl
        Same entries with anchor_labels / anchor_kind / dimension fields added.
        object_size with a single dimension is renamed to object_dimensions.
        object_size with list ground truth [length, width, height] is split
        into 3 separate object_dimensions entries (one per dim) so each entry
        has a single numeric ground truth, mirroring the v2 schema.

Output schema mirrors image_consistency_thor_eval_v2:
    image_path, question, ground_truth, question_type ('object_dimensions' |
    'object_distance_to_camera' | 'object_pair_distance_center'), anchor_labels
    (list[str]), anchor_kind ('single' | 'pair'), dimension ('height'|'length'
    |'width' for object_dimensions, else None), group_id, image_id, plus all
    original metadata.
"""

import json
import re
from pathlib import Path

INPUT = Path("/nas2/edwin/lmms-eval/data/scene_variation_042226.jsonl")
OUTPUT = Path("/nas2/edwin/lmms-eval/data/scene_variation_042226_aligned.jsonl")

_PAIR_QID_RE = re.compile(
    r"^object_pair_distance_center_"
    r"(?P<a>\w+?)\|[+-]?\d+\.\d+\|[+-]?\d+\.\d+\|[+-]?\d+\.\d+_"
    r"(?P<b>\w+?)\|[+-]?\d+\.\d+\|[+-]?\d+\.\d+\|[+-]?\d+\.\d+$"
)
_DIM_RE = re.compile(r"estimated (height|length|width) of the ", re.IGNORECASE)
_DIMS_LIST_RE = re.compile(r"estimated length, width,? and the height of the ", re.IGNORECASE)


def _primary_label(record: dict) -> str:
    primary = record.get("primary_object") or ""
    assert primary, f"Missing primary_object in record: {record}"
    return primary.split("|", 1)[0]


def _pair_labels(record: dict) -> list[str]:
    qid = record.get("question_id") or ""
    match = _PAIR_QID_RE.match(qid)
    assert match, f"Could not extract two labels from question_id={qid!r}"
    return [match.group("a"), match.group("b")]


def _convert_distance(record: dict) -> list[dict]:
    out = dict(record)
    out["anchor_labels"] = [_primary_label(record)]
    out["anchor_kind"] = "single"
    out["dimension"] = None
    return [out]


def _convert_pair(record: dict) -> list[dict]:
    out = dict(record)
    out["anchor_labels"] = _pair_labels(record)
    out["anchor_kind"] = "pair"
    out["dimension"] = None
    return [out]


def _convert_size(record: dict) -> list[dict]:
    question = record.get("question", "")
    label = _primary_label(record)
    base_group = record.get("group_id") or ""
    base_image_id = record.get("image_id") or ""

    list_match = _DIMS_LIST_RE.search(question)
    single_match = _DIM_RE.search(question)

    if list_match:
        gt_text = record["ground_truth"]
        gt_list = json.loads(gt_text)
        assert isinstance(gt_list, list) and len(gt_list) == 3, f"Expected list[3] gt, got {gt_text!r}"
        dims = ["length", "width", "height"]
        outputs = []
        for dim, value in zip(dims, gt_list):
            new = dict(record)
            new["question_type"] = "object_dimensions"
            new["anchor_labels"] = [label]
            new["anchor_kind"] = "single"
            new["dimension"] = dim
            new["ground_truth"] = str(value)
            new["group_id"] = f"{base_group}__dim_{dim}"
            new["image_id"] = f"{base_image_id}__dim_{dim}"
            outputs.append(new)
        return outputs

    assert single_match, f"Could not detect dimension in object_size question: {question[:200]!r}"
    dim = single_match.group(1).lower()
    out = dict(record)
    out["question_type"] = "object_dimensions"
    out["anchor_labels"] = [label]
    out["anchor_kind"] = "single"
    out["dimension"] = dim
    return [out]


_CONVERTERS = {
    "object_distance_to_camera": _convert_distance,
    "object_pair_distance_center": _convert_pair,
    "object_size": _convert_size,
}


def main() -> None:
    assert INPUT.exists(), f"Missing input file: {INPUT}"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    n_in = 0
    with INPUT.open() as fp_in, OUTPUT.open("w") as fp_out:
        for line in fp_in:
            record = json.loads(line)
            n_in += 1
            qtype = record["question_type"]
            convert = _CONVERTERS.get(qtype)
            assert convert is not None, f"Unsupported question_type {qtype!r}"
            for entry in convert(record):
                fp_out.write(json.dumps(entry) + "\n")
                counts[entry["question_type"]] = counts.get(entry["question_type"], 0) + 1

    print(f"Read {n_in} records from {INPUT}")
    print(f"Wrote {sum(counts.values())} records to {OUTPUT}")
    for qt, n in sorted(counts.items()):
        print(f"  {qt}: {n}")


if __name__ == "__main__":
    main()
