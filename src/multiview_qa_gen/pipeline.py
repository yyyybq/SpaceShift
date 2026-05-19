"""
Main Pipeline Module for Multiview QA Generation

This module orchestrates the complete question generation process for AI2THOR scenes:
1. Initialize AI2THOR controller
2. Load scene and filter/select objects
3. Sample camera trajectories based on move pattern
4. For each camera pose, render images and generate questions
5. Save the dataset in JSONL format with image paths
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict
from PIL import Image
import numpy as np
from tqdm import tqdm

try:
    from .config import PipelineConfig, IMAGE_WIDTH_HEIGHT
    from .object_selector import ObjectSelector, SceneObject
    from .camera_sampler import CameraSampler, CameraPose, ObjectTrajectory
    from .question_generator import QuestionGenerator
except ImportError:
    from config import PipelineConfig, IMAGE_WIDTH_HEIGHT
    from object_selector import ObjectSelector, SceneObject
    from camera_sampler import CameraSampler, CameraPose, ObjectTrajectory
    from question_generator import QuestionGenerator


class MultiviewQAPipeline:
    """
    Main pipeline for generating spatial reasoning questions from AI2THOR scenes
    with multi-view camera sampling support.
    
    Pipeline Flow:
        1. Initialize AI2THOR controller for the scene
        2. Filter and select objects based on semantic and geometric constraints
        3. Sample camera trajectories around objects based on move_pattern
        4. For each camera pose, render images and generate questions
        5. Save the dataset in JSONL format matching spatial-scene-variations format
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        config.validate()
        
        self.object_selector = ObjectSelector(config.object_selection)
        self.camera_sampler = CameraSampler(config.camera_sampling)
        self.question_generator = QuestionGenerator(config.question_config)
        
        # Controller will be initialized per scene
        self.controller = None
        
        # Output directories
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging setup
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_path = self.output_dir / 'pipeline.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_controller(self, scene_name: str):
        """Initialize AI2THOR controller for a scene."""
        try:
            from ai2thor.controller import Controller
            from ai2thor.platform import CloudRendering
        except ImportError:
            raise ImportError("ai2thor is required. Install with: pip install ai2thor")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_device_id)
        
        self.controller = Controller(
            scene=scene_name,
            visibilityDistance=20,
            renderInstanceSegmentation=self.config.render_config.render_instance_segmentation,
            platform=CloudRendering,
            width=self.config.render_config.image_width,
            height=self.config.render_config.image_height,
            fieldOfView=self.config.camera_sampling.field_of_view,
        )
        
        # Set to crouch if configured
        if self.config.camera_sampling.use_crouch:
            self.controller.step(action="Crouch")
        
        self.logger.info(f"Initialized controller for scene: {scene_name}")
    
    def _cleanup_controller(self):
        """Cleanup AI2THOR controller."""
        if self.controller:
            self.controller.stop()
            self.controller = None
    
    def _get_objects_2d(self, all_objects: List[Dict]) -> List[Dict]:
        """
        Get 2D representation of objects for trajectory sampling.
        Similar to object_utils.get_objects_2d from spatial-scene-variations.
        """
        objects_2d = []
        
        for obj in all_objects:
            if obj.get('objectType', '').lower() in self.config.object_selection.blacklist:
                continue
            
            pos = obj.get('position', {})
            aabb = obj.get('axisAlignedBoundingBox', {})
            
            if not pos or not aabb:
                continue
            
            obj_2d = {
                'id': obj['objectId'],
                'type': obj['objectType'],
                'centroid': np.array([pos['x'], pos['z']]),
                'centroid_height': pos['y'],
                'position': pos,
                'aabb': aabb,
            }
            objects_2d.append(obj_2d)
        
        return objects_2d
    
    def _save_image(self, image_data: np.ndarray, sub_path: str) -> str:
        """Save image to disk and return path."""
        image_path = self.output_dir / 'images' / sub_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        
        pil_image = Image.fromarray(image_data.astype(np.uint8))
        pil_image.save(image_path, format='PNG')
        
        return str(image_path)
    
    def _construct_camera_id(self, scene_name: str, view_metadata: Dict) -> str:
        """Construct unique camera ID."""
        parts = [
            scene_name,
            f"pos{view_metadata.get('agent_position', 'unknown')}",
            f"angle{view_metadata.get('agent_rotation', 0):03d}",
            f"fov{view_metadata.get('field_of_view', 75)}",
            f"pitch{view_metadata.get('camera_pitch_degree', 0)}",
        ]
        return "_".join(parts)
    
    def _construct_image_id(self, camera_id: str, scene_edit_id: str) -> str:
        """Construct unique image ID."""
        return f"{camera_id}__{scene_edit_id}"
    
    def _teleport_to_pose(self, pose: CameraPose) -> bool:
        """Teleport agent to camera pose."""
        try:
            teleport_params = pose.to_ai2thor_teleport()
            self.controller.step(action='Teleport', **teleport_params)
            return self.controller.last_event.metadata.get('lastActionSuccess', False)
        except Exception as e:
            self.logger.error(f"Teleport failed: {e}")
            return False
    
    def _generate_data_for_view(self, scene_name: str, trajectory: ObjectTrajectory,
                                 pose: CameraPose, pose_idx: int,
                                 common_visible_object_ids: Optional[set] = None) -> Optional[Dict[str, Any]]:
        """
        Generate QA data for a single view (camera pose).
        
        Args:
            scene_name: Scene name
            trajectory: Object trajectory
            pose: Camera pose
            pose_idx: Index of pose in trajectory
            common_visible_object_ids: If provided, only use these object IDs for pair/multi questions
                                       (intersection of visible objects across all views in trajectory)
        
        Returns:
            Dictionary containing image_id, image_path, qa_pairs, metadata
        """
        # Build view metadata
        fov = self.config.camera_sampling.field_of_view
        
        # For 'around' and 'linear' patterns, use the specific object_of_interest_id
        # since the camera trajectory is designed around that object
        # For 'spherical' and 'rotation', don't filter to specific object
        # so that we can generate questions about any visible object
        object_focused_patterns = ['around', 'linear_approach', 'linear_pass_by']
        object_of_interest = trajectory.object_id if trajectory.pattern in object_focused_patterns else None
        
        view_metadata = {
            'scene_id': scene_name,
            'field_of_view': fov,
            'camera_pitch_action': 'Teleport',
            'camera_pitch_degree': int(pose.horizon),
            'agent_position': f"{pose.position[0]:.2f}_{pose.position[2]:.2f}",  # Use actual camera position (x, z)
            'agent_rotation': int(pose.rotation),
            'object_of_interest_id': object_of_interest,
            'move_pattern': trajectory.pattern,
            'pose_index': pose_idx,
        }
        
        # Construct simple image filename: ObjectType_XXX.png
        object_type = trajectory.object_type
        image_id = f"{object_type}_{pose_idx:03d}"
        
        # Save image in pattern-specific folder
        sub_folder = f"{scene_name}_{trajectory.pattern}"
        sub_path = f"{sub_folder}/{image_id}.png"
        image_path = self._save_image(image_data=self.controller.last_event.frame, sub_path=sub_path)
        
        # Build metadata for QA generation
        metadata = {
            'sceneName': self.controller.last_event.metadata['sceneName'],
            'sceneBounds': self.controller.last_event.metadata['sceneBounds'],
            'agent': self.controller.last_event.metadata['agent'],
            'objects': self.controller.last_event.metadata['objects'],
            '2dbbox': dict(self.controller.last_event.instance_detections2D or {}),
        }
        
        # Generate QA pairs
        qa_record = {
            'image_id': image_id,
            'image_path': image_path,
            'object_type': object_type,
            'object_id': trajectory.object_id,
            **view_metadata,
            'metadata': metadata,
            'common_visible_object_ids': common_visible_object_ids,  # Pass to QA generator
        }
        
        qa_pairs = self.question_generator.construct_metric_qa(qa_record)
        qa_record['qa_pairs'] = qa_pairs
        
        # Remove full metadata from output (too large)
        del qa_record['metadata']
        if 'common_visible_object_ids' in qa_record:
            del qa_record['common_visible_object_ids']
        qa_record['num_objects_visible'] = len([o for o in metadata['objects'] if o.get('visible', False)])
        
        return qa_record
    
    def process_scene(self, scene_name: str) -> List[Dict[str, Any]]:
        """
        Process a single scene and generate questions with rendered images.
        
        Args:
            scene_name: AI2THOR scene name (e.g., 'FloorPlan201')
            
        Returns:
            List of QA record dictionaries
        """
        self.logger.info(f"Processing scene: {scene_name}")
        
        # Initialize controller
        self._init_controller(scene_name)
        
        try:
            # Get all scene objects
            all_objects = self.controller.last_event.metadata['objects']
            objects_2d = self._get_objects_2d(all_objects)
            
            self.logger.info(f"Found {len(objects_2d)} valid objects for trajectory sampling")
            
            # Limit number of objects
            if len(objects_2d) > self.config.max_objects_of_interest:
                objects_2d = objects_2d[:self.config.max_objects_of_interest]
            
            # Sample trajectories
            trajectories = self.camera_sampler.sample_trajectories(
                objects_2d, self.controller)
            
            self.logger.info(f"Generated {len(trajectories)} valid trajectories")
            
            # Process each trajectory
            all_qa_records = []
            progressive_path = self.output_dir / f"{scene_name}_progressive.jsonl"
            
            for traj_idx, trajectory in enumerate(tqdm(trajectories, desc=f"[{scene_name}] Trajectories")):
                if not trajectory.is_valid:
                    continue
                
                # ============================================================
                # PHASE 1: Collect visible objects from all views in trajectory
                # to find the intersection (objects visible in ALL views)
                # ============================================================
                visible_object_ids_per_view = []
                view_metadata_cache = []  # Cache metadata for phase 2
                
                for pose_idx, pose in enumerate(trajectory.camera_poses):
                    # Teleport to pose
                    if not self._teleport_to_pose(pose):
                        self.logger.warning(f"Failed to teleport to pose {pose_idx} for {trajectory.object_id}")
                        continue
                    
                    # Get visible objects for this view
                    fov = self.config.camera_sampling.field_of_view
                    metadata = {
                        'sceneName': self.controller.last_event.metadata['sceneName'],
                        'sceneBounds': self.controller.last_event.metadata['sceneBounds'],
                        'agent': self.controller.last_event.metadata['agent'],
                        'objects': self.controller.last_event.metadata['objects'],
                        '2dbbox': dict(self.controller.last_event.instance_detections2D or {}),
                    }
                    
                    # Get visible object IDs using the same filtering as QA generator
                    visible_objs = self.question_generator.get_visible_objects_from_metadata(metadata, fov)
                    visible_ids = {obj['objectId'] for obj in visible_objs}
                    visible_object_ids_per_view.append(visible_ids)
                    
                    # Cache for phase 2
                    view_metadata_cache.append({
                        'pose_idx': pose_idx,
                        'pose': pose,
                        'frame': self.controller.last_event.frame.copy(),
                        'metadata': metadata,
                    })
                
                if not visible_object_ids_per_view:
                    continue
                
                # Compute intersection: objects visible in ALL views
                common_visible_object_ids = set.intersection(*visible_object_ids_per_view)
                
                # ============================================================
                # PHASE 2: Generate QA records using only common visible objects
                # ============================================================
                trajectory_records = []
                
                for cache in view_metadata_cache:
                    pose_idx = cache['pose_idx']
                    pose = cache['pose']
                    
                    # Teleport again to get correct state (or we could cache more)
                    if not self._teleport_to_pose(pose):
                        continue
                    
                    # Generate data for this view with common visible objects constraint
                    qa_record = self._generate_data_for_view(
                        scene_name, trajectory, pose, pose_idx,
                        common_visible_object_ids=common_visible_object_ids)
                    
                    if qa_record and qa_record.get('qa_pairs'):
                        qa_record['num_common_visible_objects'] = len(common_visible_object_ids)
                        trajectory_records.append(qa_record)
                        
                        # Progressive save
                        with open(progressive_path, 'a') as f:
                            f.write(json.dumps({qa_record['image_id']: qa_record}) + '\n')
                
                # Add multi-view info to trajectory records
                if trajectory_records:
                    num_views = len(trajectory_records)
                    image_paths = [r['image_path'] for r in trajectory_records]
                    
                    for record in trajectory_records:
                        record['trajectory_id'] = f"{scene_name}_{trajectory.object_id}_{trajectory.pattern}"
                        record['num_views_in_trajectory'] = num_views
                        record['all_view_paths'] = image_paths
                    
                    all_qa_records.extend(trajectory_records)
            
            self.logger.info(f"Generated {len(all_qa_records)} QA records for scene {scene_name}")
            return all_qa_records
        
        finally:
            self._cleanup_controller()
    
    def run(self) -> Tuple[List[Dict], str]:
        """
        Run the full pipeline on all configured scenes.
        
        Returns:
            Tuple of (all_qa_records, output_path)
        """
        self.logger.info("Starting Multiview QA Generation Pipeline")
        self.logger.info(f"Move pattern: {self.config.camera_sampling.move_pattern}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        all_qa_records = []
        scenes = self.config.scenes
        
        if self.config.num_scenes:
            scenes = scenes[:self.config.num_scenes]
        
        for scene_name in scenes:
            try:
                scene_records = self.process_scene(scene_name)
                all_qa_records.extend(scene_records)
            except Exception as e:
                self.logger.error(f"Failed to process scene {scene_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save final output
        output_path = self.output_dir / 'qa_pairs.json'
        with open(output_path, 'w') as f:
            json.dump(all_qa_records, f, indent=2)
        
        # Also save as JSONL for streaming
        jsonl_path = self.output_dir / 'qa_pairs.jsonl'
        with open(jsonl_path, 'w') as f:
            for record in all_qa_records:
                f.write(json.dumps(record) + '\n')
        
        # Save flat QA format (one entry per question)
        flat_qa = self._convert_to_flat_qa(all_qa_records)
        flat_path = self.output_dir / 'qa_pairs_flat.json'
        with open(flat_path, 'w') as f:
            json.dump(flat_qa, f, indent=2)
        
        # Save statistics
        self._save_statistics(all_qa_records)
        
        self.logger.info(f"Pipeline complete. Output saved to {self.output_dir}")
        self.logger.info(f"Total QA records: {len(all_qa_records)}")
        self.logger.info(f"Total questions: {len(flat_qa)}")
        
        return all_qa_records, str(output_path)
    
    def _convert_to_flat_qa(self, qa_records: List[Dict]) -> List[Dict]:
        """
        Convert nested QA records to flat format (one entry per question).
        Matches spatial-scene-variations format.
        """
        flat_qa = []
        
        for record in qa_records:
            base_info = {
                'scene_id': record.get('scene_id'),
                'image_id': record.get('image_id'),
                'image_path': record.get('image_path'),
                'camera_id': record.get('camera_id'),
                'move_pattern': record.get('move_pattern'),
                'field_of_view': record.get('field_of_view'),
                'trajectory_id': record.get('trajectory_id'),
                'num_views_in_trajectory': record.get('num_views_in_trajectory'),
                'all_view_paths': record.get('all_view_paths', []),
            }
            
            for qa in record.get('qa_pairs', []):
                flat_entry = {
                    **base_info,
                    'question': qa.get('question'),
                    'answer': qa.get('answer'),
                    'question_type': qa.get('question_type'),
                    'question_id': qa.get('question_id'),
                    'primary_object': qa.get('primary_object'),
                }
                flat_qa.append(flat_entry)
        
        return flat_qa
    
    def _save_statistics(self, qa_records: List[Dict]):
        """Save dataset statistics."""
        from collections import Counter
        
        stats = {
            'total_records': len(qa_records),
            'total_questions': sum(len(r.get('qa_pairs', [])) for r in qa_records),
            'scenes': list(set(r.get('scene_id') for r in qa_records)),
            'move_pattern': self.config.camera_sampling.move_pattern,
            'question_type_counts': Counter(),
            'questions_per_trajectory': [],
        }
        
        for record in qa_records:
            for qa in record.get('qa_pairs', []):
                stats['question_type_counts'][qa.get('question_type', 'unknown')] += 1
            stats['questions_per_trajectory'].append(len(record.get('qa_pairs', [])))
        
        # Convert Counter to dict for JSON
        stats['question_type_counts'] = dict(stats['question_type_counts'])
        stats['avg_questions_per_view'] = (
            sum(stats['questions_per_trajectory']) / len(stats['questions_per_trajectory'])
            if stats['questions_per_trajectory'] else 0
        )
        
        stats_path = self.output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Statistics saved to {stats_path}")


def create_pipeline_from_args(args) -> MultiviewQAPipeline:
    """Create pipeline from command-line arguments."""
    from .config import (
        PipelineConfig, ObjectSelectionConfig, CameraSamplingConfig,
        QuestionConfig, RenderConfig
    )
    
    # Build config from args
    camera_config = CameraSamplingConfig(
        move_pattern=args.move_pattern,
        linear_sub_pattern=getattr(args, 'linear_sub_pattern', 'approach'),
        total_rotation=args.total_rotation,
        increment_rotation=args.increment_rotation,
        radius=args.radius,
        num_cameras_per_item=args.num_cameras,
        field_of_view=args.fov,
    )
    
    config = PipelineConfig(
        scenes=args.scenes,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        camera_sampling=camera_config,
        max_objects_of_interest=args.max_objects,
        gpu_device_id=args.gpu,
    )
    
    return MultiviewQAPipeline(config)
