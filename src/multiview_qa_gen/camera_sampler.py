"""
Camera Sampler Module for AI2THOR Multiview QA Generation

This module handles sampling valid camera poses around objects for question generation.
Supports multiple move patterns: around, spherical, rotation, linear.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

try:
    from .config import CameraSamplingConfig
    from .object_selector import SceneObject
except ImportError:
    from config import CameraSamplingConfig
    from object_selector import SceneObject


@dataclass
class CameraPose:
    """Represents a camera pose in the scene."""
    position: np.ndarray  # (x, y, z) world position
    rotation: float  # Yaw angle in degrees
    horizon: float  # Pitch angle in degrees (positive = look down)
    target: np.ndarray  # Point the camera is looking at
    pattern: str  # Move pattern used to generate this pose
    index: int  # Index within the trajectory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'position': {
                'x': float(self.position[0]),
                'y': float(self.position[1]),
                'z': float(self.position[2])
            },
            'rotation': {'y': float(self.rotation)},
            'horizon': float(self.horizon),
            'target': {
                'x': float(self.target[0]),
                'y': float(self.target[1]),
                'z': float(self.target[2])
            },
            'pattern': self.pattern,
            'index': self.index,
        }
    
    def to_ai2thor_teleport(self) -> Dict[str, Any]:
        """Convert to AI2THOR Teleport action parameters.
        
        Note: self.rotation is stored in AI2THOR convention directly:
        - 0 degrees = facing +Z direction
        - 90 degrees = facing +X direction (clockwise from above)
        """
        return {
            'position': {
                'x': float(self.position[0]),
                'y': float(self.position[1]),
                'z': float(self.position[2])
            },
            'rotation': {'y': float(self.rotation)},  # Already in AI2THOR convention
            'horizon': float(self.horizon),
            'standing': False,  # Use crouch position
            'forceAction': True,
        }


@dataclass
class ObjectTrajectory:
    """Represents a camera trajectory around an object."""
    object_id: str
    object_type: str
    centroid: Tuple[float, float]  # (x, z) 2D position
    centroid_height: float  # y position
    camera_poses: List[CameraPose]
    pattern: str
    is_valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'object_id': self.object_id,
            'object_type': self.object_type,
            'centroid': list(self.centroid),
            'centroid_height': self.centroid_height,
            'camera_poses': [p.to_dict() for p in self.camera_poses],
            'pattern': self.pattern,
            'num_poses': len(self.camera_poses),
            'is_valid': self.is_valid,
        }


class CameraSampler:
    """
    Samples valid camera poses around objects for multiview question generation.
    
    Supports multiple move patterns:
    - 'around': Horizontal circle around object
    - 'spherical': Sample on sphere surface around object
    - 'rotation': Stand at room center, rotate 360°
    - 'linear': Walk toward or past object
    """
    
    def __init__(self, config: CameraSamplingConfig):
        self.config = config
    
    def get_reachable_positions(self, controller) -> np.ndarray:
        """Get reachable positions from AI2THOR controller."""
        reachable_points = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        return np.array([[p['x'], p['z']] for p in reachable_points])
    
    def is_reachable(self, point_2d: np.ndarray, reachable_xz: np.ndarray,
                     threshold: float = None) -> bool:
        """Check if a 2D point (x, z) is near a reachable position."""
        if threshold is None:
            threshold = self.config.reachability_threshold
        
        if len(reachable_xz) == 0:
            return False
        
        distances = np.linalg.norm(reachable_xz - point_2d, axis=1)
        return np.min(distances) <= threshold
    
    def get_snapped_y(self, reachable_points: List[Dict], x: float, z: float) -> float:
        """Get the Y coordinate for a position snapped to nearest reachable point."""
        min_dist = float('inf')
        best_y = 0.0
        
        for p in reachable_points:
            dist = (p['x'] - x)**2 + (p['z'] - z)**2
            if dist < min_dist:
                min_dist = dist
                best_y = p['y']
        
        return best_y
    
    def compute_horizon(self, eye_y: float, target_y: float, distance: float) -> float:
        """Compute horizon (pitch) angle to look at target."""
        if distance <= 0:
            return 0.0
        
        angle = np.degrees(np.arctan((eye_y - target_y) / distance))
        # Snap to increment
        snapped = np.round(angle / self.config.horizon_increment) * self.config.horizon_increment
        # Clamp to valid range
        return np.clip(snapped, self.config.min_horizon, self.config.max_horizon)
    
    def check_object_visibility(self, controller, object_id: str, 
                                 position: np.ndarray, yaw: float, horizon: float = 0.0) -> bool:
        """
        Check if an object is visible from a given camera pose using AI2THOR.
        
        Args:
            controller: AI2THOR controller
            object_id: ID of the object to check visibility for
            position: Camera position (x, y, z)
            yaw: Camera yaw in degrees
            horizon: Camera pitch in degrees
            
        Returns:
            True if object is visible, False otherwise
        """
        try:
            # Teleport to the position
            controller.step(
                action='Teleport',
                position={'x': float(position[0]), 'y': float(position[1]), 'z': float(position[2])},
                rotation={'y': float(yaw)},
                horizon=float(horizon),
                standing=False,
                forceAction=True
            )
            
            if not controller.last_event.metadata.get('lastActionSuccess', False):
                return False
            
            # Check if object is in visible objects
            visible_objects = controller.last_event.metadata.get('objects', [])
            for obj in visible_objects:
                if obj['objectId'] == object_id and obj.get('visible', False):
                    return True
            
            return False
        except Exception as e:
            logging.warning(f"Visibility check failed: {e}")
            return False
    
    def sample_around_pattern(self, object_info: Dict, 
                              reachable_xz: np.ndarray,
                              reachable_points: List[Dict],
                              eye_y: float) -> Optional[ObjectTrajectory]:
        """
        Sample camera poses in a horizontal circle around the object.
        
        Randomly samples camera positions on a circle around the object,
        only requiring that each position is reachable (no continuous arc needed).
        Trajectories with fewer than min_cameras_for_around valid poses are discarded.
        """
        centroid = np.array(object_info['centroid'])
        centroid_height = object_info['centroid_height']
        radius = self.config.radius
        num_cameras = self.config.num_cameras
        
        # Minimum cameras required for a valid trajectory (default 10)
        min_cameras = getattr(self.config, 'min_cameras_for_around', 10)
        
        # Step 1: Densely sample the circle to find all reachable positions
        # Use 360 test points (1 degree resolution) for thorough coverage
        num_test_points = 360
        test_angles = np.linspace(0, 2 * np.pi, num_test_points, endpoint=False)
        
        reachable_angles = []
        for angle in test_angles:
            test_point = centroid + np.array([np.cos(angle), np.sin(angle)]) * radius
            if self.is_reachable(test_point, reachable_xz):
                reachable_angles.append(angle)
        
        # Step 2: Check if we have enough reachable positions
        if len(reachable_angles) < min_cameras:
            return ObjectTrajectory(
                object_id=object_info['id'],
                object_type=object_info.get('type', 'unknown'),
                centroid=tuple(centroid),
                centroid_height=centroid_height,
                camera_poses=[],
                pattern='around',
                is_valid=False
            )
        
        # Step 3: Randomly sample num_cameras positions from reachable angles
        # If we have more reachable positions than needed, randomly select
        if len(reachable_angles) >= num_cameras:
            selected_angles = np.random.choice(reachable_angles, size=num_cameras, replace=False)
        else:
            # Use all available reachable positions
            selected_angles = np.array(reachable_angles)
        
        # Sort angles for consistent ordering (optional, helps with visualization)
        selected_angles = np.sort(selected_angles)
        
        # Step 4: Generate camera poses for selected angles
        camera_poses = []
        for i, theta_rad in enumerate(selected_angles):
            # Camera position on the circle
            cam_x = centroid[0] + radius * np.cos(theta_rad)
            cam_z = centroid[1] + radius * np.sin(theta_rad)
            cam_y = self.get_snapped_y(reachable_points, cam_x, cam_z)
            
            # Horizon angle (pitch to look at object center)
            horizon = self.compute_horizon(eye_y, centroid_height, radius)
            
            # Target is object center
            target = np.array([centroid[0], centroid_height, centroid[1]])
            
            # Calculate yaw to face the object
            dx = centroid[0] - cam_x
            dz = centroid[1] - cam_z
            yaw = np.degrees(np.arctan2(dx, dz))
            
            pose = CameraPose(
                position=np.array([cam_x, cam_y, cam_z]),
                rotation=yaw,
                horizon=horizon,
                target=target,
                pattern='around',
                index=i
            )
            camera_poses.append(pose)
        
        # Step 5: Final check - discard if we don't have enough cameras
        if len(camera_poses) < min_cameras:
            return ObjectTrajectory(
                object_id=object_info['id'],
                object_type=object_info.get('type', 'unknown'),
                centroid=tuple(centroid),
                centroid_height=centroid_height,
                camera_poses=[],
                pattern='around',
                is_valid=False
            )
        
        return ObjectTrajectory(
            object_id=object_info['id'],
            object_type=object_info.get('type', 'unknown'),
            centroid=tuple(centroid),
            centroid_height=centroid_height,
            camera_poses=camera_poses,
            pattern='around',
            is_valid=True
        )
    
    def sample_spherical_pattern(self, object_info: Dict,
                                 reachable_xz: np.ndarray,
                                 reachable_points: List[Dict],
                                 eye_y: float) -> Optional[ObjectTrajectory]:
        """
        Sample camera poses on a sphere surface around the object.
        
        Provides varied viewing angles including different heights.
        The camera is placed on a sphere centered at the object's centroid,
        with the camera looking toward the object center.
        """
        centroid = np.array(object_info['centroid'])  # (x, z) in world coordinates
        centroid_height = object_info['centroid_height']  # y coordinate of object center
        radius = self.config.radius
        n_samples = self.config.spherical_samples
        elev_min, elev_max = self.config.elevation_range
        
        # Sample spherical coordinates
        camera_poses = []
        valid_count = 0
        
        # Use golden ratio for more uniform sphere sampling
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(n_samples * 3):  # Oversample to get enough valid poses
            if valid_count >= n_samples:
                break
            
            # Golden spiral sampling on unit sphere
            theta = 2 * np.pi * i / golden_ratio  # Azimuth angle
            z_sphere = 1 - (2 * i + 1) / (n_samples * 3)  # Vertical position on sphere [-1, 1]
            r_sphere = np.sqrt(1 - z_sphere**2)  # Horizontal radius at this height
            
            # Convert z_sphere to elevation angle (angle from horizontal plane)
            elev_deg = np.degrees(np.arcsin(z_sphere))
            if elev_deg < elev_min or elev_deg > elev_max:
                continue
            
            # Camera horizontal position (xz plane)
            cam_x = centroid[0] + radius * r_sphere * np.cos(theta)
            cam_z = centroid[1] + radius * r_sphere * np.sin(theta)
            
            # Check reachability (horizontal position)
            test_point = np.array([cam_x, cam_z])
            if not self.is_reachable(test_point, reachable_xz):
                continue
            
            # Camera height: on sphere centered at object centroid
            # elev_deg > 0 means camera is above object center
            # elev_deg < 0 means camera is below object center
            cam_y = centroid_height + radius * np.sin(np.radians(elev_deg))
            cam_y = np.clip(cam_y, self.config.min_camera_height, self.config.max_camera_height)
            
            # Rotation (yaw) to face object: atan2(dx, dz) where d = target - camera
            dx = centroid[0] - cam_x
            dz = centroid[1] - cam_z
            yaw = np.degrees(np.arctan2(dx, dz))
            
            # Horizon (pitch): compute based on height difference to object center
            # Positive horizon = look down
            dy = centroid_height - cam_y  # Height of target relative to camera
            horizontal_dist = np.sqrt(dx**2 + dz**2)
            if horizontal_dist > 0.01:  # Avoid division by zero
                horizon = np.degrees(np.arctan2(-dy, horizontal_dist))  # Negative dy because look down is positive
            else:
                horizon = 0.0
            horizon = np.clip(horizon, -self.config.max_horizon, self.config.max_horizon)
            
            target = np.array([centroid[0], centroid_height, centroid[1]])
            
            pose = CameraPose(
                position=np.array([cam_x, cam_y, cam_z]),
                rotation=yaw,
                horizon=horizon,
                target=target,
                pattern='spherical',
                index=valid_count
            )
            camera_poses.append(pose)
            valid_count += 1
        
        return ObjectTrajectory(
            object_id=object_info['id'],
            object_type=object_info.get('type', 'unknown'),
            centroid=tuple(centroid),
            centroid_height=centroid_height,
            camera_poses=camera_poses,
            pattern='spherical',
            is_valid=len(camera_poses) > 0
        )
    
    def sample_rotation_pattern(self, room_center: Tuple[float, float],
                                reachable_xz: np.ndarray,
                                reachable_points: List[Dict],
                                eye_y: float) -> Optional[ObjectTrajectory]:
        """
        Sample camera poses by standing at room center and rotating 360°.
        
        Room-centric approach that captures all visible objects.
        """
        total_rot = 360.0
        increment = self.config.increment_rotation
        
        # Find closest reachable position to room center
        room_center_arr = np.array(room_center)
        if len(reachable_xz) == 0:
            return None
        
        distances = np.linalg.norm(reachable_xz - room_center_arr, axis=1)
        closest_idx = np.argmin(distances)
        cam_xz = reachable_xz[closest_idx]
        cam_y = self.get_snapped_y(reachable_points, cam_xz[0], cam_xz[1])
        
        camera_poses = []
        for i, theta in enumerate(np.arange(0, total_rot, increment)):
            # Target point at fixed distance in front
            # In AI2THOR: theta=0 looks toward +Z, theta=90 looks toward +X
            look_dist = 3.0  # Look 3 meters ahead
            target_x = cam_xz[0] + look_dist * np.sin(np.radians(theta))
            target_z = cam_xz[1] + look_dist * np.cos(np.radians(theta))
            target = np.array([target_x, eye_y, target_z])
            
            pose = CameraPose(
                position=np.array([cam_xz[0], cam_y, cam_xz[1]]),
                rotation=theta,  # Direct angle for rotation pattern
                horizon=0,  # Look straight ahead
                target=target,
                pattern='rotation',
                index=i
            )
            camera_poses.append(pose)
        
        return ObjectTrajectory(
            object_id='room_center',
            object_type='room',
            centroid=tuple(cam_xz),
            centroid_height=eye_y,
            camera_poses=camera_poses,
            pattern='rotation',
            is_valid=True
        )
    
    def sample_linear_pattern(self, object_info: Dict,
                              reachable_xz: np.ndarray,
                              reachable_points: List[Dict],
                              eye_y: float,
                              controller=None) -> Optional[ObjectTrajectory]:
        """
        Sample camera poses along a linear trajectory toward/past the object.
        
        Sub-patterns:
        - 'approach': Walk toward the object along a radial line
        - 'pass_by': Walk past the object along a tangent line
        
        This implementation uses actual reachable points and finds those
        that form an approximate line, rather than sampling theoretical
        positions that may not be reachable.
        
        Args:
            controller: AI2THOR controller for visibility checking (required for pass_by)
        """
        centroid = np.array(object_info['centroid'])
        centroid_height = object_info['centroid_height']
        sub_pattern = self.config.linear_sub_pattern
        num_poses = self.config.num_cameras_per_item
        
        camera_poses = []
        
        # Get distances from all reachable points to object
        dists_to_obj = np.linalg.norm(reachable_xz - centroid, axis=1)
        
        # Filter to points within reasonable range
        min_dist = self.config.radius * 0.5
        max_dist = self.config.radius * 3.0
        valid_mask = (dists_to_obj >= min_dist) & (dists_to_obj <= max_dist)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < num_poses:
            # Not enough points, return empty trajectory
            return ObjectTrajectory(
                object_id=object_info['id'],
                object_type=object_info.get('type', 'unknown'),
                centroid=tuple(centroid),
                centroid_height=centroid_height,
                camera_poses=[],
                pattern=f'linear_{sub_pattern}',
                is_valid=False
            )
        
        if sub_pattern == 'approach':
            # Find reachable points that lie approximately on a radial line from the object
            # Try different angles and find the one with most collinear points
            best_poses = []
            
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                # Direction vector from object
                direction = np.array([np.cos(angle), np.sin(angle)])
                
                # For each valid point, check how close it is to this radial line
                candidates = []
                for idx in valid_indices:
                    point = reachable_xz[idx]
                    to_point = point - centroid
                    
                    # Project onto direction
                    proj_length = np.dot(to_point, direction)
                    if proj_length <= 0:  # Wrong side of object
                        continue
                    
                    # Distance from line (perpendicular distance)
                    perp_dist = np.abs(np.cross(direction, to_point / np.linalg.norm(to_point + 1e-6)))
                    
                    # Accept if close to the line (within 0.3m tolerance)
                    if perp_dist < 0.3:
                        candidates.append((idx, proj_length, point))
                
                if len(candidates) >= num_poses:
                    # Sort by distance from object (far to near for approach)
                    candidates.sort(key=lambda x: -x[1])
                    
                    # Take evenly spaced points
                    step = max(1, len(candidates) // num_poses)
                    selected = candidates[::step][:num_poses]
                    
                    poses = []
                    for i, (idx, proj_len, point) in enumerate(selected):
                        cam_y = self.get_snapped_y(reachable_points, point[0], point[1])
                        
                        # Face the object: yaw = atan2(dx, dz) where d = target - camera
                        dx = centroid[0] - point[0]
                        dz = centroid[1] - point[1]
                        yaw = np.degrees(np.arctan2(dx, dz))
                        dist = np.linalg.norm(point - centroid)
                        horizon = self.compute_horizon(eye_y, centroid_height, dist)
                        
                        pose = CameraPose(
                            position=np.array([point[0], cam_y, point[1]]),
                            rotation=yaw,
                            horizon=horizon,
                            target=np.array([centroid[0], centroid_height, centroid[1]]),
                            pattern='linear_approach',
                            index=i
                        )
                        poses.append(pose)
                    
                    if len(poses) > len(best_poses):
                        best_poses = poses
            
            camera_poses = best_poses
        
        elif sub_pattern == 'pass_by':
            # Linear pass-by: camera walks along a straight line, passing by the object
            # 
            # KEY BEHAVIOR:
            # - Camera moves along a straight trajectory line
            # - Camera yaw is FIXED and PERPENDICULAR to the trajectory (facing sideways)
            # - Object appears: right side of view → center → left side of view
            # - At the middle point, object is directly to the side (perpendicular to camera forward)
            #
            # Geometry:
            # - Object at 'centroid'  
            # - Trajectory passes at distance 'pass_dist' from object
            # - Camera faces perpendicular to trajectory (toward object side)
            # - As camera moves along trajectory, object crosses the view from right to left
            #
            # VISIBILITY CHECK:
            # - Use AI2THOR to verify object is actually visible at each position
            
            best_poses = []
            best_score = -1
            pass_dist = self.config.radius
            half_fov = self.config.field_of_view / 2.0
            t_max = pass_dist * np.tan(np.radians(half_fov)) * 0.7
            
            object_id = object_info['id']
            
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                radial = np.array([np.cos(angle), np.sin(angle)])
                closest_point = centroid + pass_dist * radial
                
                # Trajectory direction: 90° CW so object moves right→left in view
                trajectory_dir = np.array([radial[1], -radial[0]])
                
                # Camera forward: perpendicular to trajectory, toward object
                camera_forward = -radial
                camera_yaw = np.degrees(np.arctan2(camera_forward[0], camera_forward[1]))
                
                # Find reachable points strictly along this trajectory line
                candidates = []
                for idx in valid_indices:
                    point = reachable_xz[idx]
                    to_point = point - closest_point
                    
                    # Position along trajectory
                    t = np.dot(to_point, trajectory_dir)
                    
                    # Distance from trajectory line (must be very small)
                    perp_dist = np.abs(np.dot(to_point, radial))
                    
                    # Strict linearity: only accept points very close to the line
                    if perp_dist > 0.15:
                        continue
                    
                    # Must be within FOV range
                    if abs(t) > t_max:
                        continue
                    
                    candidates.append((idx, t, point, perp_dist))
                
                if len(candidates) < num_poses // 2:
                    continue
                
                # Sort by t value
                candidates.sort(key=lambda x: x[1])
                
                # Check we span both sides of t=0
                t_values = [c[1] for c in candidates]
                if min(t_values) > -0.1 or max(t_values) < 0.1:
                    continue
                
                # Select evenly spaced points along the trajectory
                t_min, t_max_actual = t_values[0], t_values[-1]
                t_span = t_max_actual - t_min
                
                # Target step size ~0.2m
                target_step = 0.2
                target_span = target_step * (num_poses - 1)
                
                if t_span > target_span:
                    # Center the selection around t=0
                    half_span = target_span / 2
                    candidates = [c for c in candidates if abs(c[1]) <= half_span]
                    candidates.sort(key=lambda x: x[1])
                
                if len(candidates) < num_poses // 2:
                    continue
                
                # Select num_poses points evenly distributed
                if len(candidates) >= num_poses:
                    # Pick evenly spaced indices
                    indices = np.linspace(0, len(candidates)-1, num_poses, dtype=int)
                    selected = [candidates[i] for i in indices]
                else:
                    selected = candidates
                
                # Verify visibility for each position
                if controller is not None:
                    visible_poses = []
                    for idx, t, point, perp in selected:
                        cam_y = self.get_snapped_y(reachable_points, point[0], point[1])
                        position = np.array([point[0], cam_y, point[1]])
                        
                        if self.check_object_visibility(controller, object_id, 
                                                        position, camera_yaw, horizon=0.0):
                            visible_poses.append((t, point, cam_y))
                    
                    if len(visible_poses) >= num_poses // 2:
                        # Sort by t to ensure correct order
                        visible_poses.sort(key=lambda x: x[0])
                        
                        poses = []
                        for i, (t, point, cam_y) in enumerate(visible_poses):
                            pose = CameraPose(
                                position=np.array([point[0], cam_y, point[1]]),
                                rotation=camera_yaw,
                                horizon=0.0,
                                target=np.array([centroid[0], centroid_height, centroid[1]]),
                                pattern='linear_pass_by',
                                index=i
                            )
                            poses.append(pose)
                        
                        score = len(visible_poses)
                        if score > best_score:
                            best_poses = poses
                            best_score = score
                else:
                    # No controller - fallback
                    poses = []
                    for i, (idx, t, point, perp) in enumerate(selected):
                        cam_y = self.get_snapped_y(reachable_points, point[0], point[1])
                        pose = CameraPose(
                            position=np.array([point[0], cam_y, point[1]]),
                            rotation=camera_yaw,
                            horizon=0.0,
                            target=np.array([centroid[0], centroid_height, centroid[1]]),
                            pattern='linear_pass_by',
                            index=i
                        )
                        poses.append(pose)
                    
                    if len(poses) >= num_poses // 2:
                        best_poses = poses
                        best_score = len(poses)
            
            camera_poses = best_poses
        
        return ObjectTrajectory(
            object_id=object_info['id'],
            object_type=object_info.get('type', 'unknown'),
            centroid=tuple(centroid),
            centroid_height=centroid_height,
            camera_poses=camera_poses,
            pattern=f'linear_{sub_pattern}',
            is_valid=len(camera_poses) >= num_poses // 2
        )
    
    def sample_trajectories(self, objects: List[Dict],
                            controller,
                            room_center: Optional[Tuple[float, float]] = None) -> List[ObjectTrajectory]:
        """
        Sample camera trajectories for all objects based on configured move pattern.
        
        Args:
            objects: List of object info dicts with 'centroid', 'centroid_height', 'id', 'type'
            controller: AI2THOR controller
            room_center: Optional room center for rotation pattern
            
        Returns:
            List of ObjectTrajectory instances
        """
        # Get reachable positions
        reachable_points = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        reachable_xz = np.array([[p['x'], p['z']] for p in reachable_points])
        
        # Get eye height (crouch if configured)
        if self.config.use_crouch:
            controller.step(action="Crouch")
        eye_y = controller.last_event.metadata['cameraPosition']['y']
        
        trajectories = []
        pattern = self.config.move_pattern
        
        if pattern == 'rotation':
            # Room-centric: single trajectory
            if room_center is None:
                bounds = controller.last_event.metadata['sceneBounds']['center']
                room_center = (bounds['x'], bounds['z'])
            
            traj = self.sample_rotation_pattern(room_center, reachable_xz, reachable_points, eye_y)
            if traj and traj.is_valid:
                trajectories.append(traj)
        else:
            # Object-centric: trajectory per object
            for obj_info in objects:
                if pattern == 'around':
                    traj = self.sample_around_pattern(obj_info, reachable_xz, reachable_points, eye_y)
                elif pattern == 'spherical':
                    traj = self.sample_spherical_pattern(obj_info, reachable_xz, reachable_points, eye_y)
                elif pattern == 'linear':
                    traj = self.sample_linear_pattern(obj_info, reachable_xz, reachable_points, eye_y, controller)
                else:
                    logging.warning(f"Unknown pattern: {pattern}")
                    continue
                
                if traj and traj.is_valid:
                    trajectories.append(traj)
        
        return trajectories
    
    def get_objects_for_movearound(self, controller, objects: List[SceneObject]) -> List[Dict]:
        """
        Convert SceneObjects to object info format for trajectory sampling.
        Equivalent to get_objects_to_movearound from movearound_editor.py.
        """
        object_infos = []
        
        for obj in objects:
            obj_info = {
                'id': obj.object_id,
                'type': obj.object_type,
                'centroid': obj.centroid_2d,
                'centroid_height': obj.centroid_height,
            }
            object_infos.append(obj_info)
        
        return object_infos
