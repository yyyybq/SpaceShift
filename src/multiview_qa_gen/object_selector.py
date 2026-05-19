"""
Object Selector Module for AI2THOR Multiview QA Generation

This module handles filtering and selecting suitable objects from AI2THOR scenes
for spatial reasoning question generation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from itertools import combinations
from collections import Counter
from dataclasses import dataclass

try:
    from .config import ObjectSelectionConfig, IMAGE_WIDTH_HEIGHT, CORNER_POINTS_THRESHOLD
except ImportError:
    from config import ObjectSelectionConfig, IMAGE_WIDTH_HEIGHT, CORNER_POINTS_THRESHOLD


@dataclass
class SceneObject:
    """Represents a scene object with its properties from AI2THOR."""
    object_id: str
    object_type: str
    name: str
    position: Dict[str, float]  # {x, y, z}
    rotation: Dict[str, float]  # {x, y, z}
    bbox_corners: List[List[float]]  # 8 corner points of AABB
    obb_corners: Optional[List[Dict[str, float]]]  # OBB corner points
    dims: np.ndarray  # (length, width, height)
    center: np.ndarray  # (x, y, z) center point
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    visible: bool
    distance: Optional[float] = None  # Distance to camera
    bbox_2d: Optional[Dict] = None  # 2D bounding box in image
    
    @property
    def max_dim(self) -> float:
        """Maximum dimension of the object."""
        return float(np.max(self.dims))
    
    @property
    def min_dim(self) -> float:
        """Minimum dimension of the object."""
        return float(np.min(self.dims))
    
    @property
    def volume(self) -> float:
        """Volume of the object bounding box."""
        return float(np.prod(self.dims))
    
    @property
    def height(self) -> float:
        """Height of the object."""
        return float(self.dims[2])
    
    @property
    def centroid_2d(self) -> Tuple[float, float]:
        """2D centroid (x, z) for top-down view."""
        return (self.center[0], self.center[2])
    
    @property
    def centroid_height(self) -> float:
        """Y coordinate of center."""
        return float(self.center[1])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'objectId': self.object_id,
            'objectType': self.object_type,
            'name': self.name,
            'position': self.position,
            'rotation': self.rotation,
            'center': self.center.tolist(),
            'dims': self.dims.tolist(),
            'aabb_min': self.aabb_min.tolist(),
            'aabb_max': self.aabb_max.tolist(),
            'visible': self.visible,
            'distance': self.distance,
            'volume': self.volume,
        }
    
    def to_ai2thor_format(self) -> Dict[str, Any]:
        """Convert back to AI2THOR metadata format for compatibility."""
        return {
            'objectId': self.object_id,
            'objectType': self.object_type,
            'name': self.name,
            'position': self.position,
            'rotation': self.rotation,
            'visible': self.visible,
            'distance': self.distance,
            'axisAlignedBoundingBox': {
                'cornerPoints': self.bbox_corners,
                'center': {'x': self.center[0], 'y': self.center[1], 'z': self.center[2]},
                'size': {'x': self.dims[0], 'y': self.dims[1], 'z': self.dims[2]},
            },
            'objectOrientedBoundingBox': {
                'cornerPoints': self.obb_corners
            } if self.obb_corners else None,
        }
    
    def get_obb_size(self) -> Tuple[float, float, float]:
        """
        Get object size as (length, width, height).
        Length >= Width by convention.
        """
        length = max(self.dims[0], self.dims[2])
        width = min(self.dims[0], self.dims[2])
        height = self.dims[1]
        return (length, width, height)


class ObjectSelector:
    """
    Selects and filters objects from AI2THOR scenes for question generation.
    
    Filters objects based on:
    - Semantic category (blacklist)
    - Geometric constraints (size, volume, aspect ratio)
    - Visibility in current view
    - Uniqueness (no duplicate object types)
    """
    
    def __init__(self, config: ObjectSelectionConfig):
        self.config = config
    
    def parse_ai2thor_object(self, obj: Dict[str, Any], 
                             camera_position: Optional[Dict[str, float]] = None) -> Optional[SceneObject]:
        """
        Parse AI2THOR object metadata into SceneObject.
        
        Args:
            obj: AI2THOR object metadata dictionary
            camera_position: Optional camera position for distance calculation
            
        Returns:
            SceneObject or None if parsing fails
        """
        try:
            # Get axis-aligned bounding box
            aabb = obj.get('axisAlignedBoundingBox', {})
            corner_points = aabb.get('cornerPoints', [])
            
            if not corner_points or len(corner_points) < 8:
                return None
            
            corners = np.array(corner_points)
            aabb_min = np.min(corners, axis=0)
            aabb_max = np.max(corners, axis=0)
            center = (aabb_min + aabb_max) / 2
            dims = aabb_max - aabb_min
            
            # Get OBB if available
            obb = obj.get('objectOrientedBoundingBox', {})
            obb_corners = obb.get('cornerPoints') if obb else None
            
            # Calculate distance to camera
            distance = None
            if camera_position:
                pos = obj.get('position', {})
                distance = np.sqrt(
                    (pos['x'] - camera_position['x'])**2 +
                    (pos['y'] - camera_position['y'])**2 +
                    (pos['z'] - camera_position['z'])**2
                )
            elif 'distance' in obj:
                distance = obj['distance']
            
            return SceneObject(
                object_id=obj['objectId'],
                object_type=obj['objectType'],
                name=obj.get('name', obj['objectId']),
                position=obj.get('position', {}),
                rotation=obj.get('rotation', {}),
                bbox_corners=corner_points,
                obb_corners=obb_corners,
                dims=dims,
                center=center,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                visible=obj.get('visible', False),
                distance=distance,
                bbox_2d=None,
            )
        except Exception as e:
            print(f"Warning: Failed to parse object {obj.get('objectId', 'unknown')}: {e}")
            return None
    
    def filter_by_semantic(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Filter objects by semantic category (blacklist)."""
        return [
            obj for obj in objects
            if obj.object_type.lower() not in self.config.blacklist
        ]
    
    def filter_by_geometry(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Filter objects by geometric constraints."""
        filtered = []
        
        for obj in objects:
            # Dimension filter
            if self.config.enable_dim_filter:
                if obj.min_dim < self.config.min_dim_component:
                    continue
                if obj.max_dim > self.config.max_dim_component:
                    continue
            
            # Volume filter
            if self.config.enable_volume_filter:
                if obj.volume < self.config.min_volume:
                    continue
            
            # Aspect ratio filter
            if self.config.enable_aspect_ratio_filter:
                aspect = obj.min_dim / obj.max_dim if obj.max_dim > 0 else 0
                if aspect < self.config.min_aspect_ratio:
                    continue
            
            filtered.append(obj)
        
        return filtered
    
    def filter_by_visibility(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Filter to only visible objects."""
        return [obj for obj in objects if obj.visible]
    
    def filter_unique_types(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Filter to keep only objects with unique types (no duplicates)."""
        type_counter = Counter(obj.object_type for obj in objects)
        return [
            obj for obj in objects
            if type_counter[obj.object_type] == 1
        ]
    
    def select_objects(self, ai2thor_objects: List[Dict[str, Any]],
                       camera_position: Optional[Dict[str, float]] = None,
                       require_visible: bool = True,
                       require_unique: bool = True) -> List[SceneObject]:
        """
        Select valid objects from AI2THOR scene metadata.
        
        Args:
            ai2thor_objects: List of AI2THOR object metadata dictionaries
            camera_position: Optional camera position for distance calculation
            require_visible: Only include visible objects
            require_unique: Only include objects with unique types
            
        Returns:
            List of filtered SceneObject instances
        """
        # Parse all objects
        objects = []
        for obj_data in ai2thor_objects:
            scene_obj = self.parse_ai2thor_object(obj_data, camera_position)
            if scene_obj is not None:
                objects.append(scene_obj)
        
        # Apply filters
        objects = self.filter_by_semantic(objects)
        objects = self.filter_by_geometry(objects)
        
        if require_visible:
            objects = self.filter_by_visibility(objects)
        
        if require_unique:
            objects = self.filter_unique_types(objects)
        
        return objects
    
    def select_object_pairs(self, objects: List[SceneObject]) -> List[Tuple[SceneObject, SceneObject]]:
        """
        Select valid object pairs for relational questions.
        
        Args:
            objects: List of SceneObject instances
            
        Returns:
            List of (obj1, obj2) tuples that satisfy pair constraints
        """
        valid_pairs = []
        
        for obj1, obj2 in combinations(objects, 2):
            # Calculate distance between objects
            dist = np.linalg.norm(obj1.center - obj2.center)
            
            # Check distance constraints
            if dist < self.config.min_pair_dist or dist > self.config.max_pair_dist:
                continue
            
            # Check dimension ratio constraint
            dim_ratio = obj1.max_dim / obj2.max_dim if obj2.max_dim > 0 else float('inf')
            if dim_ratio > self.config.max_pair_dim_ratio or dim_ratio < 1/self.config.max_pair_dim_ratio:
                continue
            
            # Check absolute dimension difference
            dim_diff = abs(obj1.max_dim - obj2.max_dim)
            if dim_diff > self.config.max_pair_dim_diff:
                continue
            
            valid_pairs.append((obj1, obj2))
        
        return valid_pairs


def project_3d_to_2d(corner_points: List, metadata: Dict, field_of_view_deg: float,
                     image_size: int = IMAGE_WIDTH_HEIGHT) -> List[Optional[Tuple[float, float]]]:
    """
    Project 3D corner points to 2D image coordinates.
    
    Args:
        corner_points: List of 3D points
        metadata: AI2THOR metadata containing agent info
        field_of_view_deg: Camera field of view in degrees
        image_size: Image width/height in pixels
        
    Returns:
        List of 2D points (x, y) or None if behind camera
    """
    agent = metadata.get('agent', {})
    cam_pos = agent.get('position', {'x': 0, 'y': 0, 'z': 0})
    cam_rot = agent.get('rotation', {'x': 0, 'y': 0, 'z': 0})
    
    # Camera parameters
    fov_rad = np.radians(field_of_view_deg)
    focal_length = (image_size / 2) / np.tan(fov_rad / 2)
    
    # Rotation: yaw from rotation.y, pitch from cameraHorizon
    # AI2THOR convention: yaw=0 is +Z, yaw=90 is +X (clockwise from above)
    # cameraHorizon positive = look down, negative = look up
    yaw = np.radians(cam_rot['y'])
    pitch = np.radians(agent.get('cameraHorizon', 0))
    
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    
    projected = []
    for point in corner_points:
        if isinstance(point, dict):
            p = np.array([point['x'], point['y'], point['z']])
        else:
            p = np.array(point)
        
        # Transform to camera space
        rel = p - np.array([cam_pos['x'], cam_pos['y'], cam_pos['z']])
        
        # Apply yaw rotation (world to camera)
        # In AI2THOR, forward vector at yaw is (sin(yaw), 0, cos(yaw))
        # Right vector is (cos(yaw), 0, -sin(yaw))
        # x_cam (right) = rel · right, z_cam (forward) = rel · forward
        x_cam = cos_y * rel[0] - sin_y * rel[2]
        z_cam_yaw = sin_y * rel[0] + cos_y * rel[2]
        y_cam_yaw = rel[1]
        
        # Apply pitch rotation (around camera's X axis)
        # Positive pitch means looking down
        y_cam = cos_p * y_cam_yaw - sin_p * z_cam_yaw
        z_cam = sin_p * y_cam_yaw + cos_p * z_cam_yaw
        
        if z_cam <= 0:
            projected.append(None)
            continue
        
        # Project to image
        u = focal_length * x_cam / z_cam + image_size / 2
        v = image_size / 2 - focal_length * y_cam / z_cam
        
        projected.append((u, v))
    
    return projected


def check_corner_points(corner_points: List, metadata: Dict, field_of_view_deg: float,
                         corner_points_threshold: int = CORNER_POINTS_THRESHOLD,
                         image_size: int = IMAGE_WIDTH_HEIGHT) -> bool:
    """
    Check if enough corner points of the 3D bounding box are visible in the camera view.
    
    Args:
        corner_points: List of 3D corner points
        metadata: AI2THOR metadata
        field_of_view_deg: Camera FOV in degrees
        corner_points_threshold: Minimum number of visible corners
        image_size: Image size in pixels
        
    Returns:
        True if enough corners are visible
    """
    projected = project_3d_to_2d(corner_points, metadata, field_of_view_deg, image_size)
    
    visible_count = 0
    for point_2d in projected:
        if point_2d is not None:
            x, y = point_2d
            if 0 <= x <= image_size and 0 <= y <= image_size:
                visible_count += 1
    
    return visible_count >= corner_points_threshold
