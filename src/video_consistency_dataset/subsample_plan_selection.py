"""Greedy selection and leak-key partitioning for subsample_benchmark_plan."""

import random

from video_consistency_dataset.subsample_group_info import GroupInfo, leak_key

QUESTION_TYPES_ORDER = (
    "object_dimensions",
    "object_distance_to_camera",
    "object_pair_distance_center",
)


def infos_for_type(
    groups: dict[str, dict],
    question_type: str,
    require_clips: int,
    excluded_group_ids: set[str] | None = None,
) -> list[GroupInfo]:
    excluded = excluded_group_ids or set()
    out: list[GroupInfo] = []
    for gid, data in groups.items():
        if gid in excluded:
            continue
        if data["n"] != require_clips:
            continue
        row = data["first"]
        if row["question_type"] != question_type:
            continue
        out.append(
            GroupInfo(
                group_id=gid,
                question_type=question_type,
                scene_id=row["scene_id"],
                room_bucket=row["room_bucket"],
                anchor_ids=tuple(row["anchor_ids"]),
            )
        )
    return out


def partition_leak_keys(
    key_map: dict[tuple, list[GroupInfo]],
    eval_need: int,
    train_need: int,
) -> tuple[set[tuple], set[tuple]]:
    total = sum(len(v) for v in key_map.values())
    assert total >= eval_need + train_need, (
        f"Not enough groups for split: {total} < {eval_need + train_need}"
    )
    items = sorted(key_map.items(), key=lambda kv: len(kv[1]))
    eval_keys: set[tuple] = set()
    ge = 0
    for lk, gs in items:
        if ge >= eval_need:
            break
        eval_keys.add(lk)
        ge += len(gs)
    all_keys = set(key_map.keys())
    train_keys = all_keys - eval_keys

    def group_count(ks: set[tuple]) -> int:
        return sum(len(key_map[k]) for k in ks)

    while group_count(train_keys) < train_need:
        movable = [k for k in eval_keys if ge - len(key_map[k]) >= eval_need]
        assert movable, "Leak-key partition infeasible; reduce eval/train per-type counts."
        k_move = min(movable, key=lambda x: len(key_map[x]))
        eval_keys.remove(k_move)
        train_keys.add(k_move)
        ge = group_count(eval_keys)
    assert ge >= eval_need
    assert group_count(train_keys) >= train_need
    return eval_keys, train_keys


def greedy_pick(infos: list[GroupInfo], k: int, rng: random.Random) -> list[GroupInfo]:
    pool = list(infos)
    rng.shuffle(pool)
    picked: list[GroupInfo] = []
    used_scenes: set[str] = set()
    used_anchors: set[str] = set()
    used_rooms: set[str] = set()
    assert len(pool) >= k, f"Pool size {len(pool)} < required {k}"
    while len(picked) < k:
        best: GroupInfo | None = None
        best_key: tuple[int, str] = (-1, "")
        for g in pool:
            if any(p.group_id == g.group_id for p in picked):
                continue
            ns = 1000 if g.scene_id not in used_scenes else 0
            na = 50 * len(set(g.anchor_ids) - used_anchors)
            nr = 10 if g.room_bucket not in used_rooms else 0
            key = (ns + na + nr, g.group_id)
            if key > best_key:
                best_key = key
                best = g
        assert best is not None
        picked.append(best)
        pool = [g for g in pool if g.group_id != best.group_id]
        used_scenes.add(best.scene_id)
        used_rooms.add(best.room_bucket)
        used_anchors.update(best.anchor_ids)
    return picked


def select_eval_train_for_type(
    groups: dict[str, dict],
    question_type: str,
    require_clips: int,
    eval_k: int,
    train_k: int,
    rng: random.Random,
) -> tuple[list[GroupInfo], list[GroupInfo]]:
    pool = infos_for_type(groups, question_type, require_clips)
    key_map: dict[tuple, list[GroupInfo]] = {}
    for g in pool:
        key_map.setdefault(leak_key(g), []).append(g)
    ekeys, tkeys = partition_leak_keys(key_map, eval_k, train_k)
    eval_candidates = [g for lk in ekeys for g in key_map[lk]]
    train_candidates = [g for lk in tkeys for g in key_map[lk]]
    return greedy_pick(eval_candidates, eval_k, rng), greedy_pick(train_candidates, train_k, rng)
