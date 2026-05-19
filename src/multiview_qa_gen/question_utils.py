"""
Question Utilities for Multiview QA Generation

Utility functions for constructing question-answer pairs from AI2THOR scene metadata.
Adapted from spatial-scene-variations/src/question_generation/question_utils.py
"""

import math
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R

try:
    from . import question_templates
    from .config import IMAGE_WIDTH_HEIGHT
except ImportError:
    import question_templates
    from config import IMAGE_WIDTH_HEIGHT


def get_object_size_from_obb(corner_points: List[Dict[str, float]]) -> Tuple[float, float, float]:
    """
    Calculate object size (length, width, height) from oriented bounding box corners.
    
    Length >= width by convention.
    
    Args:
        corner_points: List of 8 corner point dicts with x, y, z keys
        
    Returns:
        Tuple of (length, width, height) in meters
    """
    if not corner_points or len(corner_points) < 8:
        return (0, 0, 0)
    
    # Convert to numpy array
    if isinstance(corner_points[0], dict):
        points = np.array([[p['x'], p['y'], p['z']] for p in corner_points])
    else:
        points = np.array(corner_points)
    
    # Calculate dimensions from axis-aligned bounding box
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    dims = max_coords - min_coords
    
    # x, z are horizontal; y is vertical (height)
    horizontal_dims = sorted([dims[0], dims[2]], reverse=True)
    length = horizontal_dims[0]
    width = horizontal_dims[1]
    height = dims[1]
    
    return (length, width, height)


def construct_object_size_qa(obj: Dict[str, Any], fov: int) -> Dict[str, Any]:
    """
    Construct a question about the dimensions (length, width, height) of an object.
    """
    obb = obj.get('objectOrientedBoundingBox', {})
    corner_points = obb.get('cornerPoints', [])
    
    if not corner_points:
        aabb = obj.get('axisAlignedBoundingBox', {})
        corner_points = aabb.get('cornerPoints', [])
    
    if not corner_points:
        return {}
    
    length, width, height = get_object_size_from_obb(corner_points)
    
    rounded_length = round(length, 1)
    rounded_width = round(width, 1)
    rounded_height = round(height, 1)
    
    answer_string = f"[{rounded_length}, {rounded_width}, {rounded_height}]"
    
    return {
        "question": " ".join([
            question_templates.OBJECT_SIZE_TEMPLATE.format(object=obj['objectType']),
            question_templates.POST_PROMPT.format(fov=fov, image_res=IMAGE_WIDTH_HEIGHT)
        ]),
        "answer": answer_string,
        "question_type": "object_size",
        "question_id": f"object_size_{obj['objectId']}",
        "primary_object": obj['objectId']
    }


def construct_object_size_comparison_relative_qa(obj1: Dict[str, Any], obj2: Dict[str, Any], 
                                                  dimension: str, fov: int) -> Dict[str, Any]:
    """
    Construct a question comparing the relative size of two objects along a specific dimension.
    """
    obb1 = obj1.get('objectOrientedBoundingBox', {})
    obb2 = obj2.get('objectOrientedBoundingBox', {})
    
    corners1 = obb1.get('cornerPoints', [])
    corners2 = obb2.get('cornerPoints', [])
    
    if not corners1 or not corners2:
        return {}
    
    obj1_length, obj1_width, obj1_height = get_object_size_from_obb(corners1)
    obj2_length, obj2_width, obj2_height = get_object_size_from_obb(corners2)
    
    if dimension == "length":
        if obj2_length == 0:
            return {}
        ratio = obj1_length / obj2_length
    elif dimension == "width":
        if obj2_width == 0:
            return {}
        ratio = obj1_width / obj2_width
    else:  # height
        if obj2_height == 0:
            return {}
        ratio = obj1_height / obj2_height
    
    return {
        "question": " ".join([
            question_templates.OBJECT_SIZE_COMPARISON_RELATIVE_TEMPLATE.format(
                dimension=dimension,
                object1=obj1['objectType'],
                object2=obj2['objectType']
            ),
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT.format(fov=fov, image_res=IMAGE_WIDTH_HEIGHT)
        ]),
        "answer": str(round(ratio, 1)),
        "question_type": "object_size_comparison_relative",
        "question_id": f"object_size_comparison_relative_{obj1['objectId']}_{obj2['objectId']}_{dimension}",
        "primary_object": obj1['objectId']
    }


def construct_object_size_comparison_absolute_qa(obj1: Dict[str, Any], obj2: Dict[str, Any],
                                                  dimension: str, fov: int) -> Dict[str, Any]:
    """
    Construct a question comparing the absolute size of two objects along a specific dimension.
    """
    obb1 = obj1.get('objectOrientedBoundingBox', {})
    obb2 = obj2.get('objectOrientedBoundingBox', {})
    
    corners1 = obb1.get('cornerPoints', [])
    corners2 = obb2.get('cornerPoints', [])
    
    if not corners1 or not corners2:
        return {}
    
    obj1_length, obj1_width, obj1_height = get_object_size_from_obb(corners1)
    obj2_length, obj2_width, obj2_height = get_object_size_from_obb(corners2)
    
    if dimension == "length":
        obj1_dim = obj1_length
        obj2_dim = obj2_length
    elif dimension == "width":
        obj1_dim = obj1_width
        obj2_dim = obj2_width
    else:  # height
        obj1_dim = obj1_height
        obj2_dim = obj2_height
    
    return {
        "question": " ".join([
            question_templates.OBJECT_SIZE_COMPARISON_ABSOLUTE_TEMPLATE.format(
                dimension=dimension,
                object1=obj1['objectType'],
                object2=obj2['objectType'],
                obj2_dimension=round(obj2_dim, 1)
            ),
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT.format(fov=fov, image_res=IMAGE_WIDTH_HEIGHT)
        ]),
        "answer": str(round(obj1_dim, 1)),
        "question_type": "object_size_comparison_absolute",
        "question_id": f"object_size_comparison_absolute_{obj1['objectId']}_{obj2['objectId']}_{dimension}",
        "primary_object": obj1['objectId']
    }


def construct_object_pair_distance_center_qa(obj1: Dict[str, Any], obj2: Dict[str, Any],
                                              fov: int) -> Dict[str, Any]:
    """
    Construct a question about the distance between the centers of two objects.
    """
    pos1 = obj1.get('position', {})
    pos2 = obj2.get('position', {})
    
    if not pos1 or not pos2:
        return {}
    
    distance = math.sqrt(
        (pos1['x'] - pos2['x'])**2 +
        (pos1['y'] - pos2['y'])**2 +
        (pos1['z'] - pos2['z'])**2
    )
    
    return {
        "question": " ".join([
            question_templates.OBJECT_PAIR_DISTANCE_CENTER_TEMPLATE.format(
                object1=obj1['objectType'],
                object2=obj2['objectType']
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT.format(fov=fov, image_res=IMAGE_WIDTH_HEIGHT)
        ]),
        "answer": str(round(distance, 1)),
        "question_type": "object_pair_distance_center",
        "question_id": f"object_pair_distance_center_{obj1['objectId']}_{obj2['objectId']}",
        "primary_object": obj1['objectId']
    }


def construct_object_distance_to_camera_qa(obj: Dict[str, Any], fov: int) -> Dict[str, Any]:
    """
    Construct a question about the distance of an object from the camera.
    """
    distance = obj.get('distance')
    
    if distance is None:
        return {}
    
    return {
        "question": " ".join([
            question_templates.OBJECT_DISTANCE_TO_CAMERA_TEMPLATE.format(object1=obj['objectType']),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT.format(fov=fov, image_res=IMAGE_WIDTH_HEIGHT)
        ]),
        "answer": str(round(distance, 1)),
        "question_type": "object_distance_to_camera",
        "question_id": f"object_distance_to_camera_{obj['objectId']}",
        "primary_object": obj['objectId']
    }


def construct_object_comparison_absolute_distance_qa(obj_a: Dict[str, Any], obj_b: Dict[str, Any],
                                                      obj_x: Dict[str, Any], obj_y: Dict[str, Any],
                                                      fov: int) -> Dict[str, Any]:
    """
    Construct a question that provides the distance between two objects (X and Y) 
    and asks for the distance between two other objects (A and B).
    """
    dist_a_b = math.sqrt(
        (obj_a['position']['x'] - obj_b['position']['x'])**2 +
        (obj_a['position']['y'] - obj_b['position']['y'])**2 +
        (obj_a['position']['z'] - obj_b['position']['z'])**2
    )
    
    dist_x_y = math.sqrt(
        (obj_x['position']['x'] - obj_y['position']['x'])**2 +
        (obj_x['position']['y'] - obj_y['position']['y'])**2 +
        (obj_x['position']['z'] - obj_y['position']['z'])**2
    )
    
    return {
        "question": " ".join([
            question_templates.OBJECT_COMPARISON_ABSOLUTE_DISTANCE_TEMPLATE.format(
                objectA=obj_a['objectType'],
                objectB=obj_b['objectType'],
                distance=round(dist_x_y, 1),
                objectX=obj_x['objectType'],
                objectY=obj_y['objectType']
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT.format(fov=fov, image_res=IMAGE_WIDTH_HEIGHT)
        ]),
        "answer": str(round(dist_a_b, 1)),
        "question_type": "object_comparison_absolute_distance",
        "question_id": f"object_comparison_absolute_distance_{obj_a['objectId']}_{obj_b['objectId']}_{obj_x['objectId']}_{obj_y['objectId']}",
        "primary_object": obj_a['objectId']
    }


def construct_object_comparison_relative_distance_qa(obj_a: Dict[str, Any], obj_b: Dict[str, Any],
                                                      obj_x: Dict[str, Any], obj_y: Dict[str, Any],
                                                      fov: int) -> Dict[str, Any]:
    """
    Construct a question asking for the relative distance ratio between two pairs of objects.
    """
    dist_a_b = math.sqrt(
        (obj_a['position']['x'] - obj_b['position']['x'])**2 +
        (obj_a['position']['y'] - obj_b['position']['y'])**2 +
        (obj_a['position']['z'] - obj_b['position']['z'])**2
    )
    
    dist_x_y = math.sqrt(
        (obj_x['position']['x'] - obj_y['position']['x'])**2 +
        (obj_x['position']['y'] - obj_y['position']['y'])**2 +
        (obj_x['position']['z'] - obj_y['position']['z'])**2
    )
    
    if dist_x_y == 0:
        return {}
    
    ratio = dist_a_b / dist_x_y
    
    return {
        "question": " ".join([
            question_templates.OBJECT_COMPARISON_RELATIVE_DISTANCE_TEMPLATE.format(
                objectA=obj_a['objectType'],
                objectB=obj_b['objectType'],
                objectX=obj_x['objectType'],
                objectY=obj_y['objectType']
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT.format(fov=fov, image_res=IMAGE_WIDTH_HEIGHT)
        ]),
        "answer": str(round(ratio, 1)),
        "question_type": "object_comparison_relative_distance",
        "question_id": f"object_comparison_relative_distance_{obj_a['objectId']}_{obj_b['objectId']}_{obj_x['objectId']}_{obj_y['objectId']}",
        "primary_object": obj_a['objectId']
    }


def construct_object_pair_distance_center_w_size_qa(obj1: Dict[str, Any], obj2: Dict[str, Any],
                                                     dimension: str, fov: int) -> Dict[str, Any]:
    """
    Construct a question about the distance between two objects, 
    given a specific dimension of the first object.
    """
    pos1 = obj1.get('position')
    pos2 = obj2.get('position')
    obb1 = obj1.get('objectOrientedBoundingBox', {})
    
    if not pos1 or not pos2 or not obb1.get('cornerPoints'):
        return {}
    
    distance = math.sqrt(
        (pos1['x'] - pos2['x'])**2 +
        (pos1['y'] - pos2['y'])**2 +
        (pos1['z'] - pos2['z'])**2
    )
    
    obj1_length, obj1_width, obj1_height = get_object_size_from_obb(obb1['cornerPoints'])
    
    if dimension == "length":
        ref_dim = obj1_length
    elif dimension == "width":
        ref_dim = obj1_width
    else:
        ref_dim = obj1_height
    
    if ref_dim == 0:
        return {}
    
    return {
        "question": " ".join([
            question_templates.OBJECT_PAIR_DISTANCE_CENTER_W_SIZE_TEMPLATE.format(
                object1=obj1['objectType'],
                object2=obj2['objectType'],
                dimension=dimension,
                obj1_dimension=round(ref_dim, 1)
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT.format(fov=fov, image_res=IMAGE_WIDTH_HEIGHT)
        ]),
        "answer": str(round(distance, 1)),
        "question_type": "object_pair_distance_center_w_size",
        "question_id": f"object_pair_distance_center_w_size_{obj1['objectId']}_{obj2['objectId']}_{dimension}",
        "primary_object": obj1['objectId']
    }


def construct_object_pair_distance_vector_qa(obj_a: Dict[str, Any], obj_b: Dict[str, Any],
                                              agent: Dict[str, Any], fov: int) -> Dict[str, Any]:
    """
    Construct a question asking for the vector from one object to another in agent's local coordinates.
    """
    # Global vector from A to B
    distance_vector = np.array([
        obj_b['position']['x'] - obj_a['position']['x'],
        obj_b['position']['y'] - obj_a['position']['y'],
        obj_b['position']['z'] - obj_a['position']['z']
    ])
    
    # Apply inverse agent rotation to get local vector
    agent_rot = agent.get('rotation', {'x': 0, 'y': 0, 'z': 0})
    r = R.from_euler('yxz', [agent_rot['y'], -agent_rot.get('x', 0), agent_rot.get('z', 0)], degrees=True)
    local_vector = r.inv().apply(distance_vector)
    
    answer_string = f"[{round(local_vector[0], 2)}, {round(local_vector[1], 2)}, {round(local_vector[2], 2)}]"
    
    return {
        "question": " ".join([
            question_templates.OBJECT_PAIR_DISTANCE_VECTOR_TEMPLATE.format(
                objectA=obj_a['objectType'],
                objectB=obj_b['objectType']
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.POST_PROMPT.format(fov=fov, image_res=IMAGE_WIDTH_HEIGHT)
        ]),
        "answer": answer_string,
        "question_type": "object_pair_distance_vector",
        "question_id": f"object_pair_distance_vector_{obj_a['objectId']}_{obj_b['objectId']}",
        "primary_object": obj_a['objectId']
    }


# Mapping of question types to constructor functions
QUESTION_CONSTRUCTORS = {
    'object_size': construct_object_size_qa,
    'object_distance_to_camera': construct_object_distance_to_camera_qa,
    'object_size_comparison_relative': construct_object_size_comparison_relative_qa,
    'object_size_comparison_absolute': construct_object_size_comparison_absolute_qa,
    'object_pair_distance_center': construct_object_pair_distance_center_qa,
    'object_pair_distance_center_w_size': construct_object_pair_distance_center_w_size_qa,
    'object_comparison_absolute_distance': construct_object_comparison_absolute_distance_qa,
    'object_comparison_relative_distance': construct_object_comparison_relative_distance_qa,
    'object_pair_distance_vector': construct_object_pair_distance_vector_qa,
}
