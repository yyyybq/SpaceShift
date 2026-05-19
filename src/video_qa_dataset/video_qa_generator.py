"""Generate size and distance QA pairs from AI2-THOR object metadata.

QA generation is dispatched by trajectory category:
  - linear  (approach, passby):  size only (individual dims)
  - circular (around, spherical): size + camera-to-object distance
  - rotation:                     size (sole-visible) + object-to-object distance (co-visible)
"""

import math
from itertools import combinations

from utils.object_utils import get_object_size_from_obb
from video_qa_dataset.video_qa_templates import (
    DISTANCE_POST_PROMPT,
    NA_POST_PROMPT,
    OBJECT_DISTANCE_COMPARISON_RELATIVE_TEMPLATE,
    OBJECT_DISTANCE_TO_CAMERA_TEMPLATE,
    OBJECT_PAIR_DISTANCE_TEMPLATE,
    OBJECT_PAIR_DISTANCE_W_SIZE_TEMPLATE,
    OBJECT_SIZE_COMPARISON_ABSOLUTE_TEMPLATE,
    OBJECT_SIZE_COMPARISON_RELATIVE_TEMPLATE,
    OBJECT_SIZE_TEMPLATE,
    PATTERN_QA_CATEGORY,
    POST_PROMPT,
)
from video_qa_dataset.variation_config import pattern_from_trajectory

DIMENSIONS = ("length", "width", "height")
MAX_COMPARISON_PAIRS = 2


def _obj_size(obj: dict) -> tuple[float, float, float] | None:
    obb = obj.get("objectOrientedBoundingBox")
    if not obb or not obb.get("cornerPoints"):
        return None
    result = get_object_size_from_obb(obb["cornerPoints"])
    if result is None:
        return None
    return tuple(result)


def _dim_value(size: tuple[float, float, float], dimension: str) -> float:
    return {"length": size[0], "width": size[1], "height": size[2]}[dimension]


def _center_distance(obj1: dict, obj2: dict) -> float | None:
    p1, p2 = obj1.get("position"), obj2.get("position")
    if not p1 or not p2:
        return None
    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2
        + (p1["y"] - p2["y"]) ** 2
        + (p1["z"] - p2["z"]) ** 2
    )


def _deduplicate_by_type(objects: list[dict]) -> list[dict]:
    seen_types: set[str] = set()
    unique: list[dict] = []
    for obj in objects:
        obj_type = obj.get("objectType", "")
        if obj_type not in seen_types:
            seen_types.add(obj_type)
            unique.append(obj)
    return unique


# ---------------------------------------------------------------------------
# Size QA (individual dimensions)
# ---------------------------------------------------------------------------

def generate_size_qa(primary_obj: dict) -> list[dict]:
    """One QA pair per dimension (length, width, height)."""
    size = _obj_size(primary_obj)
    if size is None:
        return []
    qa_pairs = []
    for dim in DIMENSIONS:
        value = round(_dim_value(size, dim), 1)
        qa_pairs.append({
            "question": " ".join([
                OBJECT_SIZE_TEMPLATE.format(
                    dimension=dim, object=primary_obj["objectType"],
                ),
                NA_POST_PROMPT,
                POST_PROMPT,
            ]),
            "answer": str(value),
            "question_type": f"object_{dim}",
            "question_id": f"object_{dim}_{primary_obj['objectId']}",
            "primary_object": primary_obj["objectId"],
        })
    return qa_pairs


def generate_size_comparison_qa(obj1: dict, obj2: dict) -> list[dict]:
    """Relative and absolute size comparison across all dimensions."""
    size1 = _obj_size(obj1)
    size2 = _obj_size(obj2)
    if size1 is None or size2 is None:
        return []
    qa_pairs = []
    for dim in DIMENSIONS:
        v1, v2 = _dim_value(size1, dim), _dim_value(size2, dim)
        if v2 == 0:
            continue
        qa_pairs.append({
            "question": " ".join([
                OBJECT_SIZE_COMPARISON_RELATIVE_TEMPLATE.format(
                    dimension=dim, object1=obj1["objectType"], object2=obj2["objectType"],
                ),
                NA_POST_PROMPT,
                POST_PROMPT,
            ]),
            "answer": str(round(v1 / v2, 1)),
            "question_type": "object_size_comparison_relative",
            "question_id": f"object_size_comparison_relative_{obj1['objectId']}_{obj2['objectId']}_{dim}",
            "primary_object": obj1["objectId"],
        })
        qa_pairs.append({
            "question": " ".join([
                OBJECT_SIZE_COMPARISON_ABSOLUTE_TEMPLATE.format(
                    dimension=dim,
                    object1=obj1["objectType"],
                    object2=obj2["objectType"],
                    obj2_dimension=round(v2, 1),
                ),
                NA_POST_PROMPT,
                POST_PROMPT,
            ]),
            "answer": str(round(v1, 1)),
            "question_type": "object_size_comparison_absolute",
            "question_id": f"object_size_comparison_absolute_{obj1['objectId']}_{obj2['objectId']}_{dim}",
            "primary_object": obj1["objectId"],
        })
    return qa_pairs


# ---------------------------------------------------------------------------
# Camera distance QA (circular trajectories only)
# ---------------------------------------------------------------------------

def generate_camera_distance_qa(primary_obj: dict, radius: float) -> list[dict]:
    """Distance from camera to object, using the orbit radius as ground truth."""
    return [{
        "question": " ".join([
            OBJECT_DISTANCE_TO_CAMERA_TEMPLATE.format(object=primary_obj["objectType"]),
            NA_POST_PROMPT,
            POST_PROMPT,
        ]),
        "answer": str(round(radius, 1)),
        "question_type": "object_distance_to_camera",
        "question_id": f"object_distance_to_camera_{primary_obj['objectId']}",
        "primary_object": primary_obj["objectId"],
    }]


# ---------------------------------------------------------------------------
# Object-to-object distance QA (rotation trajectories only)
# ---------------------------------------------------------------------------

def generate_distance_qa(obj1: dict, obj2: dict) -> list[dict]:
    """Center-to-center distance between two objects."""
    dist = _center_distance(obj1, obj2)
    if dist is None:
        return []
    qa_pairs = [{
        "question": " ".join([
            OBJECT_PAIR_DISTANCE_TEMPLATE.format(
                object1=obj1["objectType"], object2=obj2["objectType"],
            ),
            DISTANCE_POST_PROMPT,
            NA_POST_PROMPT,
            POST_PROMPT,
        ]),
        "answer": str(round(dist, 1)),
        "question_type": "object_pair_distance_center",
        "question_id": f"object_pair_distance_center_{obj1['objectId']}_{obj2['objectId']}",
        "primary_object": obj1["objectId"],
    }]
    size1 = _obj_size(obj1)
    if size1 is not None:
        for dim in DIMENSIONS:
            v1 = _dim_value(size1, dim)
            if v1 == 0:
                continue
            qa_pairs.append({
                "question": " ".join([
                    OBJECT_PAIR_DISTANCE_W_SIZE_TEMPLATE.format(
                        object1=obj1["objectType"],
                        object2=obj2["objectType"],
                        dimension=dim,
                        obj1_dimension=round(v1, 1),
                    ),
                    DISTANCE_POST_PROMPT,
                    NA_POST_PROMPT,
                    POST_PROMPT,
                ]),
                "answer": str(round(dist, 1)),
                "question_type": "object_pair_distance_center_w_size",
                "question_id": f"object_pair_distance_w_size_{obj1['objectId']}_{obj2['objectId']}_{dim}",
                "primary_object": obj1["objectId"],
            })
    return qa_pairs


def generate_distance_comparison_qa(
    obj_a: dict, obj_b: dict, obj_x: dict, obj_y: dict,
) -> list[dict]:
    """Relative distance ratio between two object pairs."""
    dist_ab = _center_distance(obj_a, obj_b)
    dist_xy = _center_distance(obj_x, obj_y)
    if dist_ab is None or dist_xy is None or dist_xy == 0:
        return []
    return [{
        "question": " ".join([
            OBJECT_DISTANCE_COMPARISON_RELATIVE_TEMPLATE.format(
                objectA=obj_a["objectType"],
                objectB=obj_b["objectType"],
                objectX=obj_x["objectType"],
                objectY=obj_y["objectType"],
            ),
            DISTANCE_POST_PROMPT,
            NA_POST_PROMPT,
            POST_PROMPT,
        ]),
        "answer": str(round(dist_ab / dist_xy, 1)),
        "question_type": "object_distance_comparison_relative",
        "question_id": (
            f"object_distance_comparison_relative_"
            f"{obj_a['objectId']}_{obj_b['objectId']}_"
            f"{obj_x['objectId']}_{obj_y['objectId']}"
        ),
        "primary_object": obj_a["objectId"],
    }]


# ---------------------------------------------------------------------------
# Per-category dispatch
# ---------------------------------------------------------------------------

def _capped_pairs(objects: list[dict], max_pairs: int = MAX_COMPARISON_PAIRS) -> list[tuple[dict, dict]]:
    pairs = []
    for i, obj1 in enumerate(objects):
        for obj2 in objects[i + 1:]:
            pairs.append((obj1, obj2))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _filter_primary(objects: list[dict], primary_object_type: str | None) -> list[dict]:
    if primary_object_type is None:
        return objects
    return [o for o in objects if o.get("objectType", "") == primary_object_type]


def _generate_linear_qa(
    visible_objects: list[dict], primary_object_type: str | None = None,
) -> list[dict]:
    """Size QA only for the primary target object (approach, passby)."""
    primary = _deduplicate_by_type(_filter_primary(visible_objects, primary_object_type))
    qa_pairs = []
    for obj in primary:
        qa_pairs.extend(generate_size_qa(obj))
    for obj1, obj2 in _capped_pairs(primary):
        qa_pairs.extend(generate_size_comparison_qa(obj1, obj2))
    return qa_pairs


def _generate_circular_qa(
    visible_objects: list[dict], radius: float, primary_object_type: str | None = None,
) -> list[dict]:
    """Size QA + camera distance for the primary target object (around, spherical)."""
    primary = _deduplicate_by_type(_filter_primary(visible_objects, primary_object_type))
    qa_pairs = []
    for obj in primary:
        qa_pairs.extend(generate_size_qa(obj))
        qa_pairs.extend(generate_camera_distance_qa(obj, radius))
    for obj1, obj2 in _capped_pairs(primary):
        qa_pairs.extend(generate_size_comparison_qa(obj1, obj2))
    return qa_pairs


def _generate_rotation_qa(per_frame_objects: list[list[dict]]) -> list[dict]:
    """Size QA for sole-visible objects, distance QA for co-visible pairs.

    per_frame_objects: list of per-frame visible object lists (already filtered).
    """
    sole_visible_ids: set[str] = set()
    covisible_pairs: set[tuple[str, str]] = set()
    all_objects_by_id: dict[str, dict] = {}

    for frame_objects in per_frame_objects:
        deduped = _deduplicate_by_type(frame_objects)
        for obj in deduped:
            all_objects_by_id[obj["objectId"]] = obj
        if len(deduped) == 1:
            sole_visible_ids.add(deduped[0]["objectId"])
        for a, b in combinations(deduped, 2):
            pair = tuple(sorted([a["objectId"], b["objectId"]]))
            covisible_pairs.add(pair)

    sole_objects = _deduplicate_by_type(
        [all_objects_by_id[oid] for oid in sole_visible_ids if oid in all_objects_by_id]
    )
    qa_pairs = []
    for obj in sole_objects:
        qa_pairs.extend(generate_size_qa(obj))

    seen_type_pairs: set[tuple[str, str]] = set()
    distance_pair_count = 0
    for id_a, id_b in covisible_pairs:
        if distance_pair_count >= MAX_COMPARISON_PAIRS:
            break
        obj_a = all_objects_by_id.get(id_a)
        obj_b = all_objects_by_id.get(id_b)
        if obj_a is None or obj_b is None:
            continue
        type_pair = tuple(sorted([obj_a["objectType"], obj_b["objectType"]]))
        if type_pair[0] == type_pair[1]:
            continue
        if type_pair in seen_type_pairs:
            continue
        seen_type_pairs.add(type_pair)
        qa_pairs.extend(generate_distance_qa(obj_a, obj_b))
        distance_pair_count += 1

    covisible_objects = _deduplicate_by_type(list(all_objects_by_id.values()))
    if len(covisible_objects) >= 4:
        a, b, x, y = covisible_objects[:4]
        qa_pairs.extend(generate_distance_comparison_qa(a, b, x, y))

    return qa_pairs


def generate_qa_for_trajectory(
    trajectory: str,
    visible_objects: list[dict] | None = None,
    radius: float = 0.0,
    per_frame_objects: list[list[dict]] | None = None,
    primary_object_type: str | None = None,
) -> list[dict]:
    """Dispatch QA generation by trajectory category."""
    pattern = pattern_from_trajectory(trajectory)
    category = PATTERN_QA_CATEGORY[pattern]

    if category == "linear":
        assert visible_objects is not None
        return _generate_linear_qa(visible_objects, primary_object_type)

    if category == "circular":
        assert visible_objects is not None
        return _generate_circular_qa(visible_objects, radius, primary_object_type)

    assert category == "rotation"
    assert per_frame_objects is not None
    return _generate_rotation_qa(per_frame_objects)
