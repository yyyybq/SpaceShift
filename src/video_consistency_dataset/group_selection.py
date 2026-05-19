"""Greedy group selection over mined candidate groups.

Candidate schema:
  {
    "engine": "thor",
    "motion_family": "around",
    "question_type": "object_dimensions",
    "dimension": "length",
    "radius": null,
    "scene_id": "FloorPlan1",
    "room_bucket": "kitchen",
    "anchor_labels": ["Sofa"],
    "available_clips": [...]
  }
"""

from collections import Counter


def _candidate_key(candidate: dict) -> tuple:
    return (
        candidate["engine"],
        candidate["motion_family"],
        candidate["question_type"],
        candidate["dimension"],
        candidate["radius"],
        tuple(candidate["anchor_ids"]),
    )


def _label_penalty(candidate: dict, state: dict) -> int:
    labels = candidate["anchor_labels"]
    if candidate["anchor_kind"] == "pair":
        return sum(state["label_counts"][label] for label in labels)
    return state["label_counts"][labels[0]]


def _score_candidate(candidate: dict, state: dict) -> tuple:
    anchor_key = _candidate_key(candidate)
    return (
        -state["candidate_counts"][anchor_key],
        -_label_penalty(candidate, state),
        -state["scene_counts"][candidate["scene_id"]],
        -state["room_counts"][candidate["room_bucket"]],
        len(candidate["available_clips"]),
        round(sum(clip["quality_score"] for clip in candidate["available_clips"]), 3),
        candidate["scene_id"],
    )


def select_candidates_for_targets(candidates: list[dict], targets: list[dict]) -> list[dict]:
    selected: list[dict] = []
    state = {
        "candidate_counts": Counter(),
        "label_counts": Counter(),
        "scene_counts": Counter(),
        "room_counts": Counter(),
    }
    for target in targets:
        matches = [
            candidate
            for candidate in candidates
            if candidate["motion_family"] == target["motion_family"]
            and candidate["question_type"] == target["question_type"]
            and candidate["dimension"] == target["dimension"]
            and candidate["radius"] == target["radius"]
            and state["candidate_counts"][_candidate_key(candidate)] == 0
        ]
        assert matches, f"No candidate groups available for target {target}"
        chosen = max(matches, key=lambda candidate: _score_candidate(candidate, state))
        selected.append(chosen)
        state["candidate_counts"][_candidate_key(chosen)] += 1
        for label in chosen["anchor_labels"]:
            state["label_counts"][label] += 1
        state["scene_counts"][chosen["scene_id"]] += 1
        state["room_counts"][chosen["room_bucket"]] += 1
    return selected
