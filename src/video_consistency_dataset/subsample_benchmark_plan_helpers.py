"""Streaming reads and clip/group row helpers for subsample_benchmark_plan."""

import json
from json import JSONDecoder
from pathlib import Path

from video_consistency_dataset.clip_spec import ClipSpec
from video_consistency_dataset.consistency_group import ConsistencyGroup


def extract_config_from_metadata(metadata_path: Path) -> dict:
    chunk_size = 8_388_608
    blob = ""
    with metadata_path.open("r", encoding="utf-8") as handle:
        while True:
            piece = handle.read(chunk_size)
            if not piece:
                break
            blob += piece
            key = '"config"'
            idx_config = blob.find(key)
            if idx_config < 0:
                continue
            colon = blob.find(":", idx_config + len(key))
            assert colon >= 0, "Malformed metadata: config key without colon."
            i = colon + 1
            while i < len(blob) and blob[i].isspace():
                i += 1
            assert i < len(blob) and blob[i] == "{", "Malformed metadata: config value is not an object."
            dec = JSONDecoder()
            cfg, _ = dec.raw_decode(blob[i:])
            assert isinstance(cfg, dict)
            return cfg
    assert False, f"No config key in {metadata_path}"


def stream_group_aggregate(clips_path: Path) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    with clips_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            gid = row["group_id"]
            if gid not in groups:
                groups[gid] = {"first": row, "n": 0, "clip_ids": []}
            groups[gid]["n"] += 1
            groups[gid]["clip_ids"].append(row["clip_id"])
    return groups


def collect_clip_rows(
    clips_path: Path,
    group_ids: list[str],
    id_order: dict[str, list[str]],
) -> dict[str, list[dict]]:
    need = set(group_ids)
    acc: dict[str, list[dict]] = {gid: [] for gid in group_ids}
    with clips_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            gid = row["group_id"]
            if gid not in need:
                continue
            acc[gid].append(row)
    for gid in group_ids:
        idx = {cid: i for i, cid in enumerate(id_order[gid])}
        acc[gid].sort(key=lambda r: idx[r["clip_id"]])
        assert len(acc[gid]) == len(id_order[gid])
    return acc


def reroot_row(row: dict, old_out: str, new_out: str) -> dict:
    row = dict(row)
    for key in ("output_dir", "video_path"):
        assert old_out in row[key], f"Expected {old_out!r} in {key}={row[key]!r}"
        row[key] = row[key].replace(old_out, new_out, 1)
    return row


def row_to_clip(row: dict) -> ClipSpec:
    return ClipSpec(
        clip_id=row["clip_id"],
        group_id=row["group_id"],
        engine=row["engine"],
        scene_id=row["scene_id"],
        room_bucket=row["room_bucket"],
        motion_family=row["motion_family"],
        trajectory=row["trajectory"],
        direction=row["direction"],
        variation_tag=row["variation_tag"],
        question=row["question"],
        ground_truth=row["ground_truth"],
        question_type=row["question_type"],
        question_family=row["question_family"],
        dimension=row["dimension"],
        radius=row["radius"],
        start_angle_offset=row["start_angle_offset"],
        cycle_count=row["cycle_count"],
        anchor_kind=row["anchor_kind"],
        anchor_ids=tuple(row["anchor_ids"]),
        anchor_labels=tuple(row["anchor_labels"]),
        video_path=row["video_path"],
        output_dir=row["output_dir"],
        quality_score=row["quality_score"],
    )


def make_group(rows: list[dict], target_group_size: int) -> ConsistencyGroup:
    r0 = rows[0]
    return ConsistencyGroup(
        group_id=r0["group_id"],
        engine=r0["engine"],
        scene_id=r0["scene_id"],
        room_bucket=r0["room_bucket"],
        motion_family=r0["motion_family"],
        question_type=r0["question_type"],
        question_family=r0["question_family"],
        dimension=r0["dimension"],
        radius=r0["radius"],
        anchor_kind=r0["anchor_kind"],
        anchor_ids=tuple(r0["anchor_ids"]),
        anchor_labels=tuple(r0["anchor_labels"]),
        shared_ground_truth=r0["ground_truth"],
        clip_ids=tuple(r["clip_id"] for r in rows),
        target_group_size=target_group_size,
        selected_group_size=len(rows),
    )
