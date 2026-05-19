"""
Multiview QA Generation Pipeline for AI2THOR

A modular pipeline for generating spatial reasoning questions from AI2THOR scenes
with multi-view camera sampling support. This combines:
- Multi-view camera patterns from question_gen_InteriorGS
- AI2THOR simulator from spatial-scene-variations
- QA generation format from spatial-scene-variations/question_generation
"""

from .config import (
    PipelineConfig,
    ObjectSelectionConfig,
    CameraSamplingConfig,
    QuestionConfig,
    RenderConfig,
)
from .object_selector import ObjectSelector
from .camera_sampler import CameraSampler
from .question_generator import QuestionGenerator
from .pipeline import MultiviewQAPipeline

__version__ = "1.0.0"
__all__ = [
    "PipelineConfig",
    "ObjectSelectionConfig", 
    "CameraSamplingConfig",
    "QuestionConfig",
    "RenderConfig",
    "ObjectSelector",
    "CameraSampler",
    "QuestionGenerator",
    "MultiviewQAPipeline",
]
