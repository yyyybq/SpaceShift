"""
Configuration module for Multiview QA Generation Pipeline

This module defines configuration dataclasses for object selection,
camera sampling, question generation, and the overall pipeline.
Compatible with AI2THOR simulator.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path


# Global constants (matching spatial-scene-variations/src/global_config.py)
IMAGE_WIDTH_HEIGHT = 384
VISIBILITY_DISTANCE = 20
CORNER_POINTS_THRESHOLD = 5


@dataclass
class ObjectSelectionConfig:
    """Configuration for object selection/filtering from AI2THOR scenes."""
    
    # Semantic blacklist for excluded object categories
    blacklist: Set[str] = field(default_factory=lambda: {
        # Structural elements
        "floor", "window", "curtains", "blinds", "walls", "shelf",
        # Additional exclusions
        "ceiling", "room", "door",
    })
    
    # Geometric filter toggles
    enable_dim_filter: bool = False
    enable_volume_filter: bool = False
    enable_aspect_ratio_filter: bool = False
    
    # Geometric constraints (only applied if corresponding filter is enabled)
    min_dim_component: float = 0.1  # Minimum dimension per axis (m)
    max_dim_component: float = 3.0  # Maximum dimension per axis (m)
    min_volume: float = 0.01  # Minimum volume (m^3)
    min_aspect_ratio: float = 0.05  # Min shortest/longest edge ratio
    
    # Visibility constraints
    min_visible_corners: int = CORNER_POINTS_THRESHOLD  # Minimum visible bbox corners
    occlusion_threshold: float = 0.5  # Maximum allowed occlusion ratio
    
    # Pair constraints for multi-object questions
    min_pair_dist: float = 0.3  # Minimum distance between paired objects
    max_pair_dist: float = 5.0  # Maximum distance between paired objects
    max_pair_dim_ratio: float = 3.0  # Max ratio of longest edges
    max_pair_dim_diff: float = 2.5  # Max absolute difference in longest edge


@dataclass
class CameraSamplingConfig:
    """Configuration for camera pose sampling in AI2THOR scenes.
    
    Move Patterns:
        - 'around': Horizontal circle - sample cameras around object on horizontal plane
        - 'spherical': Spherical sampling - sample on sphere surface (varied heights/angles)
        - 'rotation': Room rotation - stand at room center, rotate 360 degrees
        - 'linear': Linear trajectory - walk toward or past object in straight line
    """
    
    # Number of camera poses to sample per object/pair
    num_cameras_per_item: int = 5
    
    # Number of cameras for around pattern (can be different from num_cameras_per_item)
    num_cameras: int = 12
    
    # Minimum cameras required for a valid 'around' trajectory
    min_cameras_for_around: int = 10
    
    # Move pattern: 'around', 'spherical', 'rotation', or 'linear'
    move_pattern: str = 'around'
    
    # Linear pattern sub-type: 'approach' or 'pass_by'
    linear_sub_pattern: str = 'approach'
    
    # Trajectory parameters
    total_rotation: float = 90.0  # Total rotation angle for trajectory (degrees)
    increment_rotation: float = 15.0  # Angle increment between samples (degrees)
    radius: float = 1.5  # Distance from object center for 'around' pattern (m)
    
    # Spherical sampling parameters
    spherical_samples: int = 30  # Number of samples for spherical pattern
    elevation_range: tuple = (-30, 60)  # Min/max elevation angles (degrees)
    
    # Camera height constraints (m)
    # Note: For spherical pattern, these should allow full range of elevation angles
    # With radius=1.5m and object at ~1m height, max_camera_height=2.5 allows up to ~60° elevation
    max_camera_height: float = 2.5
    min_camera_height: float = 0.3
    use_crouch: bool = True  # Use crouched agent position
    
    # Field of view
    field_of_view: int = 75  # Camera FOV in degrees
    
    # Image resolution
    image_width: int = IMAGE_WIDTH_HEIGHT
    image_height: int = IMAGE_WIDTH_HEIGHT
    
    # Collision detection
    reachability_threshold: float = 0.185  # Distance threshold for reachability check
    
    # Horizon (look up/down) parameters
    horizon_increment: float = 30.0  # Horizon angle snap increment (degrees)
    max_horizon: float = 60.0  # Maximum look down angle
    min_horizon: float = -30.0  # Maximum look up angle


@dataclass
class QuestionConfig:
    """Configuration for question generation."""
    
    # Question types to generate
    enabled_question_types: Set[str] = field(default_factory=lambda: {
        # Single-object questions
        'object_size',
        'object_distance_to_camera',
        # Pair-object questions
        'object_size_comparison_relative',
        'object_size_comparison_absolute',
        'object_pair_distance_center',
        'object_pair_distance_center_w_size',
        # Multi-object questions
        'object_comparison_absolute_distance',
        'object_comparison_relative_distance',
        # Vector questions
        'object_pair_distance_vector',
    })
    
    # Dimensions to use for size comparison questions
    dimensions: List[str] = field(default_factory=lambda: ['length', 'width', 'height'])
    
    # Generate 'all' question types or specific list
    qa_types_to_generate: str = 'all'  # 'all' or list of types
    
    # Random seed for reproducibility
    random_seed: int = 42


@dataclass
class RenderConfig:
    """Configuration for image rendering."""
    
    enable_rendering: bool = True
    render_instance_segmentation: bool = True
    
    # Image settings
    image_width: int = IMAGE_WIDTH_HEIGHT
    image_height: int = IMAGE_WIDTH_HEIGHT
    
    # Output format
    image_format: str = 'png'


@dataclass
class PipelineConfig:
    """Main configuration for the multiview QA generation pipeline."""
    
    # AI2THOR scene configuration
    scenes: List[str] = field(default_factory=lambda: ['FloorPlan201'])
    num_scenes: Optional[int] = None  # Process first N scenes, None for all
    
    # Output configuration
    output_dir: str = './output'
    experiment_name: str = 'default'
    
    # Sub-configurations
    object_selection: ObjectSelectionConfig = field(default_factory=ObjectSelectionConfig)
    camera_sampling: CameraSamplingConfig = field(default_factory=CameraSamplingConfig)
    question_config: QuestionConfig = field(default_factory=QuestionConfig)
    render_config: RenderConfig = field(default_factory=RenderConfig)
    
    # Processing options
    max_objects_of_interest: int = 50  # Maximum objects to process per scene
    use_local_storage: bool = True
    
    # GPU device
    gpu_device_id: int = 0
    
    # Resume support
    resume: bool = True  # Resume from previous run if output exists
    
    def validate(self):
        """Validate configuration parameters."""
        valid_patterns = {'around', 'spherical', 'rotation', 'linear'}
        if self.camera_sampling.move_pattern not in valid_patterns:
            raise ValueError(f"Invalid move_pattern: {self.camera_sampling.move_pattern}. "
                           f"Must be one of {valid_patterns}")
        
        if self.camera_sampling.move_pattern == 'linear':
            valid_sub = {'approach', 'pass_by'}
            if self.camera_sampling.linear_sub_pattern not in valid_sub:
                raise ValueError(f"Invalid linear_sub_pattern: {self.camera_sampling.linear_sub_pattern}. "
                               f"Must be one of {valid_sub}")
        
        if self.camera_sampling.radius <= 0:
            raise ValueError("radius must be positive")
        
        if self.camera_sampling.total_rotation <= 0:
            raise ValueError("total_rotation must be positive")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create PipelineConfig from dictionary."""
        # Extract sub-configs
        obj_config = ObjectSelectionConfig(**config_dict.pop('object_selection', {}))
        cam_config = CameraSamplingConfig(**config_dict.pop('camera_sampling', {}))
        q_config = QuestionConfig(**config_dict.pop('question_config', {}))
        r_config = RenderConfig(**config_dict.pop('render_config', {}))
        
        return cls(
            object_selection=obj_config,
            camera_sampling=cam_config,
            question_config=q_config,
            render_config=r_config,
            **config_dict
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        import json
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
