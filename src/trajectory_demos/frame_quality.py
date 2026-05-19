"""Frame-quality helpers for trajectory demo selection."""


def object_bbox(controller, object_id: str):
    detections = controller.last_event.instance_detections2D or {}
    return detections.get(object_id)


def bbox_area_ratio(controller, bbox) -> float:
    height, width = controller.last_event.frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bbox_width = max(0.0, float(x2) - float(x1))
    bbox_height = max(0.0, float(y2) - float(y1))
    return (bbox_width * bbox_height) / float(width * height)


def bbox_center_ratio(controller, bbox) -> tuple[float, float]:
    height, width = controller.last_event.frame.shape[:2]
    x1, y1, x2, y2 = bbox
    center_x = (float(x1) + float(x2)) * 0.5 / float(width)
    center_y = (float(y1) + float(y2)) * 0.5 / float(height)
    return center_x, center_y


def longest_contiguous_indices(valid_pose_indices: list[int]) -> tuple[int, ...]:
    if not valid_pose_indices:
        return ()
    best_run = [valid_pose_indices[0]]
    current_run = [valid_pose_indices[0]]
    for current_index in valid_pose_indices[1:]:
        if current_index == current_run[-1] + 1:
            current_run.append(current_index)
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = [current_index]
    if len(current_run) > len(best_run):
        best_run = current_run
    return tuple(best_run)


def monotonic_fraction(values: list[float], increasing: bool) -> float:
    if len(values) < 2:
        return 1.0
    matches = 0
    total = 0
    for previous, current in zip(values[:-1], values[1:]):
        if abs(current - previous) < 1e-6:
            continue
        total += 1
        if increasing and current > previous:
            matches += 1
        if not increasing and current < previous:
            matches += 1
    if total == 0:
        return 1.0
    return matches / total


def normalized_span(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)
