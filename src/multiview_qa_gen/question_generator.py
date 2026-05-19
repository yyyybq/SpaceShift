"""
Question Generator Module for Multiview QA Generation

This module generates spatial reasoning questions from AI2THOR scenes
using visible objects and camera poses.
"""

import random
from typing import List, Dict, Any, Optional
from collections import Counter

try:
    from .config import QuestionConfig, IMAGE_WIDTH_HEIGHT, CORNER_POINTS_THRESHOLD
    from .object_selector import SceneObject, check_corner_points
    from .camera_sampler import CameraPose
    from . import question_utils
    from . import question_templates
except ImportError:
    from config import QuestionConfig, IMAGE_WIDTH_HEIGHT, CORNER_POINTS_THRESHOLD
    from object_selector import SceneObject, check_corner_points
    from camera_sampler import CameraPose
    import question_utils
    import question_templates


# Objects to ignore globally
IGNORE_OBJECTS_GLOBAL = {"floor", "window", "curtains", "blinds", "walls", "shelf"}


class QuestionGenerator:
    """
    Generates questions from AI2THOR scene metadata and camera poses.
    
    Supports three categories of questions:
    1. Single-object questions (size, distance to camera)
    2. Pair-object questions (size comparison, distance between objects)
    3. Multi-object questions (distance comparisons across multiple objects)
    """
    
    def __init__(self, config: QuestionConfig):
        self.config = config
        random.seed(config.random_seed)
    
    def get_visible_objects_from_metadata(self, metadata: Dict[str, Any], 
                                           fov: int,
                                           filter_to_object_id: Optional[str] = None) -> List[Dict]:
        """
        Get list of visible objects that pass all visibility checks.
        
        Args:
            metadata: AI2THOR scene metadata
            fov: Field of view in degrees
            filter_to_object_id: If provided, filter to only this object (used internally)
            
        Returns:
            List of visible object metadata dicts
        """
        all_objects = metadata.get('objects', [])
        bbox_2d = metadata.get('2dbbox', {})
        
        visible_objects = []
        
        for obj in all_objects:
            # Basic visibility check
            if not obj.get('visible', False):
                continue
            
            # Check object type not in blacklist
            if obj.get('objectType', '').lower() in IGNORE_OBJECTS_GLOBAL:
                continue
            
            # Check 2D bbox exists
            if obj['objectId'] not in bbox_2d:
                continue
            
            # Check corner points visibility
            aabb = obj.get('axisAlignedBoundingBox', {})
            corner_points = aabb.get('cornerPoints', [])
            if corner_points and not check_corner_points(corner_points, metadata, fov):
                continue
            
            visible_objects.append(obj)
        
        # If filter_to_object_id specified, filter to it
        if filter_to_object_id:
            visible_objects = [obj for obj in visible_objects 
                             if obj['objectId'] == filter_to_object_id]
        
        # Filter to unique object types (no duplicates)
        type_counter = Counter(obj['objectType'] for obj in visible_objects)
        visible_objects = [obj for obj in visible_objects 
                          if type_counter[obj['objectType']] == 1]
        
        return visible_objects
    
    def generate_single_object_questions(self, obj: Dict[str, Any], 
                                          fov: int) -> List[Dict[str, Any]]:
        """
        Generate questions for a single object.
        
        Args:
            obj: Object metadata dict
            fov: Field of view in degrees
            
        Returns:
            List of question dictionaries
        """
        questions = []
        
        for q_type in question_templates.SINGLE_OBJECT_QUESTIONS:
            if self.config.qa_types_to_generate != 'all':
                if q_type not in self.config.enabled_question_types:
                    continue
            
            try:
                if q_type == 'object_size':
                    if not obj.get('objectOrientedBoundingBox'):
                        continue
                    qa = question_utils.construct_object_size_qa(obj, fov)
                elif q_type == 'object_distance_to_camera':
                    qa = question_utils.construct_object_distance_to_camera_qa(obj, fov)
                else:
                    continue
                
                if qa:
                    questions.append(qa)
            except Exception as e:
                print(f"Warning: Failed to generate {q_type} question: {e}")
        
        return questions
    
    def generate_pair_object_questions(self, obj1: Dict[str, Any], obj2: Dict[str, Any],
                                        fov: int) -> List[Dict[str, Any]]:
        """
        Generate questions for a pair of objects.
        
        Args:
            obj1: First object metadata
            obj2: Second object metadata
            fov: Field of view in degrees
            
        Returns:
            List of question dictionaries
        """
        questions = []
        
        for q_type in question_templates.PAIR_OBJECT_QUESTIONS:
            if self.config.qa_types_to_generate != 'all':
                if q_type not in self.config.enabled_question_types:
                    continue
            
            try:
                if 'size' in q_type:
                    # Size questions require OBB
                    if not obj1.get('objectOrientedBoundingBox') or not obj2.get('objectOrientedBoundingBox'):
                        continue
                    
                    for dimension in self.config.dimensions:
                        if q_type == 'object_size_comparison_relative':
                            qa = question_utils.construct_object_size_comparison_relative_qa(
                                obj1, obj2, dimension, fov)
                        elif q_type == 'object_size_comparison_absolute':
                            qa = question_utils.construct_object_size_comparison_absolute_qa(
                                obj1, obj2, dimension, fov)
                        elif q_type == 'object_pair_distance_center_w_size':
                            qa = question_utils.construct_object_pair_distance_center_w_size_qa(
                                obj1, obj2, dimension, fov)
                        else:
                            continue
                        
                        if qa:
                            questions.append(qa)
                else:
                    if q_type == 'object_pair_distance_center':
                        qa = question_utils.construct_object_pair_distance_center_qa(obj1, obj2, fov)
                    else:
                        continue
                    
                    if qa:
                        questions.append(qa)
            
            except Exception as e:
                print(f"Warning: Failed to generate {q_type} question: {e}")
        
        return questions
    
    def generate_multi_object_questions(self, primary_obj: Dict[str, Any],
                                         other_objects: List[Dict[str, Any]],
                                         fov: int) -> List[Dict[str, Any]]:
        """
        Generate multi-object questions (comparing distances between multiple object pairs).
        
        Args:
            primary_obj: The primary object
            other_objects: List of other visible objects
            fov: Field of view in degrees
            
        Returns:
            List of question dictionaries
        """
        questions = []
        
        for q_type in question_templates.MULTI_OBJECT_QUESTIONS:
            if self.config.qa_types_to_generate != 'all':
                if q_type not in self.config.enabled_question_types:
                    continue
            
            try:
                # X-->A and X-->B (same primary object to two different objects)
                if len(other_objects) >= 2:
                    random.seed(self.config.random_seed)
                    obj_a, obj_b = random.sample(other_objects, 2)
                    
                    if q_type == 'object_comparison_absolute_distance':
                        qa = question_utils.construct_object_comparison_absolute_distance_qa(
                            primary_obj, obj_a, primary_obj, obj_b, fov)
                    elif q_type == 'object_comparison_relative_distance':
                        qa = question_utils.construct_object_comparison_relative_distance_qa(
                            primary_obj, obj_a, primary_obj, obj_b, fov)
                    else:
                        continue
                    
                    if qa:
                        questions.append(qa)
                
                # X-->A and Y-->B (different primary objects)
                if len(other_objects) >= 3:
                    random.seed(self.config.random_seed)
                    obj_a, obj_b, obj_y = random.sample(other_objects, 3)
                    
                    if q_type == 'object_comparison_absolute_distance':
                        qa = question_utils.construct_object_comparison_absolute_distance_qa(
                            primary_obj, obj_a, obj_y, obj_b, fov)
                    elif q_type == 'object_comparison_relative_distance':
                        qa = question_utils.construct_object_comparison_relative_distance_qa(
                            primary_obj, obj_a, obj_y, obj_b, fov)
                    else:
                        continue
                    
                    if qa:
                        questions.append(qa)
            
            except Exception as e:
                print(f"Warning: Failed to generate {q_type} question: {e}")
        
        return questions
    
    def generate_vector_questions(self, primary_obj: Dict[str, Any],
                                   other_objects: List[Dict[str, Any]],
                                   agent: Dict[str, Any],
                                   fov: int) -> List[Dict[str, Any]]:
        """
        Generate vector-based questions (object-to-object vectors in agent coordinates).
        
        Args:
            primary_obj: The primary object
            other_objects: List of other visible objects
            agent: Agent metadata (position, rotation)
            fov: Field of view in degrees
            
        Returns:
            List of question dictionaries
        """
        questions = []
        
        if 'object_pair_distance_vector' not in self.config.enabled_question_types:
            if self.config.qa_types_to_generate != 'all':
                return questions
        
        for obj2 in other_objects:
            try:
                qa = question_utils.construct_object_pair_distance_vector_qa(
                    primary_obj, obj2, agent, fov)
                if qa:
                    questions.append(qa)
            except Exception as e:
                print(f"Warning: Failed to generate vector question: {e}")
        
        return questions
    
    def construct_metric_qa(self, qa_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Construct all metric QA pairs for a given view.
        
        This is the main entry point, matching the interface from 
        spatial-scene-variations/src/question_generation/construct_metric_qa.py
        
        Args:
            qa_record: Dictionary containing:
                - metadata: AI2THOR scene metadata
                - field_of_view: Camera FOV
                - object_of_interest_id: Optional primary object to focus on
                - common_visible_object_ids: Optional set of object IDs visible in ALL views
                                             (for multi-view consistency)
                
        Returns:
            List of question-answer pair dictionaries
        """
        qa_pairs = []
        
        metadata = qa_record['metadata']
        fov = qa_record.get('field_of_view', 75)
        object_of_interest_id = qa_record.get('object_of_interest_id')
        common_visible_object_ids = qa_record.get('common_visible_object_ids')
        agent = metadata.get('agent', {})
        
        # Get ALL visible objects in current view (without filtering to object_of_interest)
        all_visible_objects = self.get_visible_objects_from_metadata(metadata, fov)
        
        if not all_visible_objects:
            return qa_pairs
        
        # If common_visible_object_ids is provided, filter to only those objects
        # for pair/multi questions (ensures consistency across all views in trajectory)
        if common_visible_object_ids is not None:
            objects_for_pairing = [obj for obj in all_visible_objects 
                                   if obj['objectId'] in common_visible_object_ids]
        else:
            objects_for_pairing = all_visible_objects
        
        # Determine which objects to use as primary for QA generation
        if object_of_interest_id:
            # If object_of_interest is specified, use it as the primary object
            # for single-object questions, but use common visible objects for pair/multi questions
            primary_objects = [obj for obj in all_visible_objects 
                              if obj['objectId'] == object_of_interest_id]
            if not primary_objects:
                # Object of interest is not visible in this view
                return qa_pairs
        else:
            # No object of interest specified, use all visible objects as primary
            primary_objects = all_visible_objects
        
        # Generate questions
        for primary_obj in primary_objects:
            # Single-object questions (only for primary object)
            qa_pairs.extend(self.generate_single_object_questions(primary_obj, fov))
            
            # Other objects for pair/multi questions (from common visible objects)
            other_objects = [obj for obj in objects_for_pairing 
                           if obj['objectId'] != primary_obj['objectId']]
            
            # Pair-object questions
            for obj2 in other_objects:
                qa_pairs.extend(self.generate_pair_object_questions(primary_obj, obj2, fov))
            
            # Multi-object questions
            qa_pairs.extend(self.generate_multi_object_questions(primary_obj, other_objects, fov))
            
            # Vector questions
            qa_pairs.extend(self.generate_vector_questions(primary_obj, other_objects, agent, fov))
        
        return qa_pairs
    
    def add_multiview_context(self, questions: List[Dict[str, Any]], 
                               num_views: int) -> List[Dict[str, Any]]:
        """
        Add multi-view context information to questions.
        
        Args:
            questions: List of question dicts
            num_views: Number of views available
            
        Returns:
            Questions with multi-view context added
        """
        for qa in questions:
            qa['num_views'] = num_views
            
            if num_views > 1:
                context = question_templates.MULTIVIEW_CONTEXT_PROMPT.format(num_views=num_views)
            else:
                context = question_templates.SINGLE_VIEW_CONTEXT_PROMPT
            
            qa['view_context'] = context
        
        return questions
