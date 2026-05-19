"""THOR-specific metadata helpers for benchmark mining and rendering.

Example:
  room_bucket_for_thor_scene("FloorPlan203") -> "living_room"
"""

from utils.object_utils import get_object_size_from_obb


def room_bucket_for_thor_scene(scene_id: str) -> str:
    numeric = int(scene_id.replace("FloorPlan", ""))
    if 1 <= numeric <= 30:
        return "kitchen"
    if 201 <= numeric <= 230:
        return "living_room"
    if 301 <= numeric <= 330:
        return "bedroom"
    if 401 <= numeric <= 430:
        return "bathroom"
    return "unknown"


def object_by_id(objects: list[dict], object_id: str) -> dict:
    matches = [obj for obj in objects if obj["objectId"] == object_id]
    assert matches, f"Object {object_id} not found in scene metadata."
    return matches[0]


def object_dimensions(obj: dict) -> tuple[float, float, float]:
    obb = obj.get("objectOrientedBoundingBox")
    assert obb and obb.get("cornerPoints"), f"Missing OBB for {obj['objectId']}"
    result = get_object_size_from_obb(obb["cornerPoints"])
    assert result is not None, f"Failed to compute size for {obj['objectId']}"
    return tuple(float(value) for value in result)


def object_center_distance(obj_a: dict, obj_b: dict) -> float:
    position_a = obj_a["position"]
    position_b = obj_b["position"]
    dx = position_a["x"] - position_b["x"]
    dy = position_a["y"] - position_b["y"]
    dz = position_a["z"] - position_b["z"]
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)
