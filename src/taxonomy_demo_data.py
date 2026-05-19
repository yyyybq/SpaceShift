"""Helpers for the taxonomy QA browser.

Schema:
  normalized clip entry:
    {"qa_key": "...", "group_id": "...", "clip_id": "...", "video_path": "...",
     "question": "...", "ground_truth": "...", "question_type": "...",
     "engine": "...", "scene_id": "...", "motion_family": "...",
     "trajectory": "...", "direction": "...", "variation_tag": "...",
     "radius": 1.0, "start_angle_offset": 90.0, "cycle_count": 2,
     "anchor_labels": ["chair"]}
"""
import json
from collections import defaultdict
from pathlib import Path

def load_jsonl(path):
    resolved = Path(path).expanduser().resolve()
    assert resolved.exists(), f"Missing input file: {resolved}"
    with resolved.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert rows, f"No rows found in {resolved}"
    return rows

def question_text(text):
    return text.split("[Output]")[0].strip()

def _shorten(text, limit=56):
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

_MULTI_OBJECT_QUESTION_TYPES = {
    "object_pair_distance_center", "object_pair_distance_center_w_size",
    "object_size_comparison_absolute", "object_size_comparison_relative",
    "object_distance_comparison_relative",
}

def _trajectory_base(traj_variation):
    """'around_cw_CoffeeTable_1x_start0' → 'around_cw_CoffeeTable'."""
    import re
    return re.sub(r"_\d+x_(?:start|angle)\d+$", "", traj_variation)

def _orbit_object(traj_base):
    """'around_cw_CoffeeTable' → 'CoffeeTable'."""
    parts = traj_base.split("_", 2)
    return parts[2] if len(parts) >= 3 else ""

def _normalize_lmms_eval_row(score):
    """Normalize one sceneshift_score dict from lmms-eval results JSONL.

    Returns None when the video's orbit object doesn't match the question's
    primary object (e.g. asking Bowl height on a CoffeeTable orbit video).
    Multi-object and room-rotation entries are always kept.
    """
    question_id = score.get("question_id", "")
    question_type = score.get("question_type", "")
    ground_truth = str(score.get("ground_truth", ""))
    traj_variation = score.get("trajectory_variation", "")
    traj_base = _trajectory_base(traj_variation)
    primary_object = score.get("primary_object", "")
    primary_type = primary_object.split("|")[0] if primary_object else ""
    orbit_obj = _orbit_object(traj_base)
    is_room = orbit_obj.startswith("room")
    is_multi = question_type in _MULTI_OBJECT_QUESTION_TYPES
    if not is_room and not is_multi and primary_type != orbit_obj:
        return None
    return {
        "qa_key": "|".join([question_type, question_id, ground_truth, traj_base]),
        "group_id": f"{question_id} @ {traj_base}",
        "clip_id": traj_variation,
        "video_path": score.get("video_path", ""),
        "question": score.get("question", ""),
        "ground_truth": ground_truth,
        "question_type": question_type,
        "engine": "",
        "scene_id": score.get("scene", ""),
        "motion_family": "",
        "trajectory": traj_variation,
        "direction": "",
        "variation_tag": traj_variation,
        "radius": None,
        "start_angle_offset": None,
        "cycle_count": None,
        "anchor_labels": [primary_object] if primary_object else [],
        "_embedded_score": score,
    }

def _normalize_benchmark_row(row):
    """Normalize one benchmark clips.jsonl row (or a row with a questions list)."""
    entries = []
    for question_row in row.get("questions") or [row]:
        question = question_row["question"]
        ground_truth = str(question_row["ground_truth"])
        question_type = question_row.get("question_type", row.get("question_type", ""))
        qid = question_row.get("question_id", "")
        stable = qid if qid else question
        entries.append(
            {
                "qa_key": "|".join(
                    [row.get("group_id", row.get("clip_id", "")), question_type, stable, ground_truth]
                ),
                "group_id": row.get("group_id", row.get("clip_id", "")),
                "clip_id": row.get("clip_id", ""),
                "video_path": row.get("video_path", ""),
                "question": question,
                "ground_truth": ground_truth,
                "question_type": question_type,
                "engine": row.get("engine", ""),
                "scene_id": row.get("scene_id", ""),
                "motion_family": row.get("motion_family", ""),
                "trajectory": row.get("trajectory", ""),
                "direction": row.get("direction", ""),
                "variation_tag": row.get("variation_tag", ""),
                "radius": row.get("radius"),
                "start_angle_offset": row.get("start_angle_offset"),
                "cycle_count": row.get("cycle_count"),
                "dimension": question_row.get("dimension", row.get("dimension")),
                "anchor_labels": question_row.get("anchor_labels") or row.get("anchor_labels", []),
            }
        )
    return entries


def load_qa_json_entries(qa_path: str | Path) -> list[dict]:
    qa_path = Path(qa_path).expanduser().resolve()
    assert qa_path.is_file(), f"Missing qa.json: {qa_path}"
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list), f"qa.json must be a list: {qa_path}"
    entries: list[dict] = []
    for item in payload:
        base = {
            "clip_id": item["clip_id"],
            "group_id": item["group_id"],
            "video_path": item.get("video_path", ""),
            "engine": item.get("engine", ""),
            "scene_id": item.get("scene_id", ""),
            "motion_family": item.get("motion_family", ""),
            "trajectory": item.get("trajectory", ""),
            "direction": item.get("direction", ""),
            "variation_tag": item.get("variation_tag", ""),
            "radius": item.get("radius"),
            "start_angle_offset": item.get("start_angle_offset"),
            "cycle_count": item.get("cycle_count"),
        }
        qrows = [
            {
                "question": q["question"],
                "ground_truth": str(q.get("answer", "")),
                "question_type": q.get("question_type", ""),
                "question_id": q.get("question_id", ""),
                "dimension": q.get("dimension"),
                "anchor_labels": q.get("anchor_labels"),
            }
            for q in item.get("questions", [])
        ]
        base["questions"] = qrows
        base["anchor_labels"] = item.get("anchor_labels") or (qrows[0].get("anchor_labels") if qrows else []) or []
        entries.extend(_normalize_benchmark_row(base))
    assert entries, f"No entries parsed from {qa_path}"
    return entries


def normalize_clips(path):
    raw_rows = load_jsonl(path)
    is_lmms_eval = "sceneshift_score" in raw_rows[0]
    entries = []
    if is_lmms_eval:
        for row in raw_rows:
            entry = _normalize_lmms_eval_row(row["sceneshift_score"])
            if entry is not None:
                entries.append(entry)
    else:
        for row in raw_rows:
            entries.extend(_normalize_benchmark_row(row))
    assert entries, f"No clip entries found in {path}"
    return entries

def load_multi_video_groups(clips_jsonl, min_videos: int = 2):
    groups = defaultdict(list)
    for entry in normalize_clips(clips_jsonl):
        groups[entry["qa_key"]].append(entry)
    multi_video = {
        key: sorted(
            value,
            key=lambda entry: (
                entry["cycle_count"] is None,
                entry["cycle_count"] or -1,
                entry["start_angle_offset"] if entry["start_angle_offset"] is not None else -1,
                entry["variation_tag"],
                entry["clip_id"],
            ),
        )
        for key, value in groups.items()
        if len(value) >= min_videos
    }
    assert multi_video, (
        f"No QA groups with at least {min_videos} video(s) found in {clips_jsonl}"
    )
    return multi_video

def build_choice_map(groups, *, sort_by_video_count_desc: bool = False):
    sorted_keys = sorted(
        groups,
        key=lambda key: (
            groups[key][0]["engine"],
            groups[key][0]["motion_family"],
            groups[key][0]["question_type"],
            groups[key][0]["group_id"],
        ),
    )
    if sort_by_video_count_desc:
        sorted_keys = sorted(sorted_keys, key=lambda key: -len(groups[key]))
    choice_labels = []
    label_to_key = {}
    for key in sorted_keys:
        sample = groups[key][0]
        label = " | ".join([sample["group_id"], sample["motion_family"] or sample["question_type"], f"{len(groups[key])} videos", _shorten(question_text(sample["question"]))])
        choice_labels.append(label)
        label_to_key[label] = key
    return choice_labels, label_to_key

def load_results(path):
    if not path:
        return {}
    lookup = {}
    for row in load_jsonl(path):
        score = row.get("sceneshift_score", row)
        video_path = score.get("video_path")
        question = score.get("question")
        if video_path and question:
            lookup[(str(Path(video_path).expanduser().resolve()), question_text(question))] = score
    return lookup

def find_result(entry, lookup):
    if entry.get("_embedded_score"):
        return entry["_embedded_score"]
    if not lookup:
        return None
    return lookup.get((str(Path(entry["video_path"]).expanduser().resolve()), question_text(entry["question"])))

def _error_color(prediction, ground_truth):
    if prediction is None:
        return "#666"
    try:
        pred = float(prediction)
        gt = float(ground_truth)
    except (TypeError, ValueError):
        return "#666"
    if abs(gt) < 1e-6:
        return "#22c55e" if abs(pred) < 0.05 else "#ef4444"
    error = abs(pred - gt) / abs(gt)
    if error <= 0.25:
        return "#22c55e"
    if error <= 0.5:
        return "#eab308"
    return "#ef4444"

def _format_angle(value):
    if value is None:
        return ""
    angle = float(value)
    return f"{int(angle)} deg" if angle.is_integer() else f"{angle:.1f} deg"

def card_html(entry, score):
    title = entry["variation_tag"] or entry["trajectory"] or entry["clip_id"]
    meta = " | ".join(part for part in [entry["motion_family"], entry["direction"], entry["scene_id"]] if part)
    details = []
    if entry["cycle_count"] is not None:
        details.append(f"{int(entry['cycle_count'])}x")
    if entry["start_angle_offset"] is not None:
        details.append(f"start {_format_angle(entry['start_angle_offset'])}")
    if entry["radius"] is not None:
        details.append(f"radius {entry['radius']}m")
    pieces = ['<div style="text-align:center">', f'<div style="font-size:0.88em; font-weight:600">{title}</div>']
    if meta:
        pieces.append(f'<div style="font-size:0.78em; color:#666">{meta}</div>')
    if details:
        pieces.append(f'<div style="font-size:0.78em; color:#666">{" | ".join(details)}</div>')
    if score is not None:
        prediction = score.get("prediction_parse")
        gt_value = score.get("ground_truth_parse", entry["ground_truth"])
        cape = score.get("CAPE")
        cape_text = f" | CAPE {cape:.2f}" if isinstance(cape, (float, int)) else ""
        prediction_text = f"{prediction:.2f}" if isinstance(prediction, float) else str(prediction)
        if prediction is None:
            prediction_text = "parse fail"
        pieces.append(f'<div style="font-size:1.25em; font-weight:700; color:{_error_color(prediction, gt_value)}">{prediction_text}</div>')
        pieces.append(f'<div style="font-size:0.78em; color:#777">GT: {entry["ground_truth"]}{cape_text}</div>')
    pieces.append("</div>")
    return "".join(pieces)

def question_markdown(entries, has_results):
    sample = entries[0]
    has_results = has_results or bool(sample.get("_embedded_score"))
    anchors = sample["anchor_labels"]
    if isinstance(anchors, str):
        anchors = [anchors]
    lines = [f"**Question:** {question_text(sample['question'])}", f"**Ground truth:** {sample['ground_truth']}", f"**Question type:** {sample['question_type']}", f"**Group:** {sample['group_id']}", f"**Videos:** {len(entries)}"]
    if sample["engine"]:
        lines.append(f"**Engine:** {sample['engine']}")
    if sample["motion_family"]:
        lines.append(f"**Motion family:** {sample['motion_family']}")
    if sample["scene_id"]:
        lines.append(f"**Scene:** {sample['scene_id']}")
    if anchors:
        lines.append(f"**Anchors:** {', '.join(anchors)}")
    if has_results:
        lines.append("**Prediction overlay:** enabled")
    return "  \n".join(lines)
