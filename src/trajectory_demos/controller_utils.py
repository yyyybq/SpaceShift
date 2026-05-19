"""Controller helpers for AI2-THOR trajectory demos."""

import os
from contextlib import contextmanager

import global_config
from utils import object_utils

from trajectory_demos.pose_record import PoseRecord


@contextmanager
def _without_cuda_visible_devices():
    original = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        yield
    finally:
        if original is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


def build_controller(scene_name: str, gpu: int, image_size: int, field_of_view: int):
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    del gpu
    with _without_cuda_visible_devices():
        return Controller(
            scene=scene_name,
            visibilityDistance=global_config.VISIBILITY_DISTANCE,
            renderInstanceSegmentation=True,
            platform=CloudRendering,
            width=image_size,
            height=image_size,
            fieldOfView=field_of_view,
        )


def prepare_scene(controller, scene_name: str) -> None:
    controller.reset(scene_name)
    controller.step(action="Crouch")


def current_eye_y(controller) -> float:
    return controller.last_event.metadata["cameraPosition"]["y"]


def objects_2d(controller) -> list[dict]:
    return object_utils.get_objects_2d(controller.last_event.metadata["objects"])


def reachable_points(controller) -> list[dict]:
    points = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    assert points is not None, "GetReachablePositions returned None."
    return points


def teleport_pose(controller, pose: PoseRecord) -> bool:
    controller.step(
        action="Teleport",
        position={"x": float(pose.x), "y": float(pose.y), "z": float(pose.z)},
        rotation={"y": float(pose.yaw)},
        horizon=float(pose.horizon),
        standing=False,
        forceAction=True,
    )
    return controller.last_event.metadata.get("lastActionSuccess", False)


def object_is_visible(controller, object_id: str) -> bool:
    return any(
        obj["objectId"] == object_id and obj.get("visible", False)
        for obj in controller.last_event.metadata["objects"]
    )
