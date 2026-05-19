#!/usr/bin/env python3
"""
Run Multiview QA Generation Pipeline

Entry point for generating spatial reasoning questions from AI2THOR scenes
with multi-view camera sampling support.

Usage:
    # Basic usage with single scene
    python run_pipeline.py --scenes FloorPlan201 --output_dir ./output
    
    # Full example with all parameters
    python run_pipeline.py \
        --scenes FloorPlan201 FloorPlan301 \
        --output_dir ./output \
        --experiment_name my_experiment \
        --move_pattern around \
        --total_rotation 90 \
        --increment_rotation 15 \
        --radius 1.5 \
        --num_cameras 5 \
        --fov 75 \
        --max_objects 50 \
        --gpu 0
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multiview_qa_gen.config import PipelineConfig, CameraSamplingConfig, QuestionConfig, RenderConfig
from multiview_qa_gen.pipeline import MultiviewQAPipeline


# Default AI2THOR scenes
DEFAULT_SCENES = [
    'FloorPlan201',  # Living room
    'FloorPlan301',  # Bedroom
    'FloorPlan401',  # Bathroom
    'FloorPlan501',  # Kitchen
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate multiview spatial reasoning QA from AI2THOR scenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save generated data'
    )
    
    # Scene selection
    parser.add_argument(
        '--scenes', type=str, nargs='+', default=DEFAULT_SCENES,
        help='AI2THOR scene names to process (e.g., FloorPlan201)'
    )
    parser.add_argument(
        '--num_scenes', type=int, default=None,
        help='Process only first N scenes (default: all)'
    )
    
    # Experiment naming
    parser.add_argument(
        '--experiment_name', type=str, default='default',
        help='Name for this experiment (used as subfolder name)'
    )
    
    # Move pattern configuration
    parser.add_argument(
        '--move_pattern', type=str, default='around',
        choices=['around', 'spherical', 'rotation', 'linear'],
        help='Camera trajectory pattern (default: around)'
    )
    parser.add_argument(
        '--linear_sub_pattern', type=str, default='approach',
        choices=['approach', 'pass_by'],
        help='Sub-pattern for linear trajectory (default: approach)'
    )
    
    # Trajectory parameters
    parser.add_argument(
        '--total_rotation', type=float, default=90.0,
        help='Total rotation angle for trajectory in degrees (default: 90)'
    )
    parser.add_argument(
        '--increment_rotation', type=float, default=15.0,
        help='Angle increment between camera poses in degrees (default: 15)'
    )
    parser.add_argument(
        '--radius', type=float, default=1.5,
        help='Distance from object center for around pattern in meters (default: 1.5)'
    )
    parser.add_argument(
        '--num_cameras', type=int, default=5,
        help='Number of camera poses per object (default: 5)'
    )
    
    # Camera parameters
    parser.add_argument(
        '--fov', type=int, default=75,
        help='Camera field of view in degrees (default: 75)'
    )
    parser.add_argument(
        '--image_size', type=int, default=384,
        help='Image width and height in pixels (default: 384)'
    )
    
    # Processing options
    parser.add_argument(
        '--max_objects', type=int, default=50,
        help='Maximum objects to process per scene (default: 50)'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device ID (default: 0)'
    )
    
    # Question types
    parser.add_argument(
        '--question_types', type=str, nargs='+', default=None,
        help='Specific question types to generate (default: all)'
    )
    
    # Config file
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to JSON config file (overrides command-line args)'
    )
    
    # Resume support
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from previous run if output exists'
    )
    
    # Rendering options
    parser.add_argument(
        '--enable_rendering', action='store_true', default=True,
        help='Enable image rendering (default: True)'
    )
    parser.add_argument(
        '--no_rendering', action='store_true',
        help='Disable image rendering (only generate QA metadata)'
    )
    
    return parser.parse_args()


def build_config(args) -> PipelineConfig:
    """Build PipelineConfig from command-line arguments or config file."""
    
    # If config file provided, load it
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        return PipelineConfig.from_dict(config_dict)
    
    # Build from command-line args
    camera_config = CameraSamplingConfig(
        move_pattern=args.move_pattern,
        linear_sub_pattern=args.linear_sub_pattern,
        total_rotation=args.total_rotation,
        increment_rotation=args.increment_rotation,
        radius=args.radius,
        num_cameras_per_item=args.num_cameras,
        field_of_view=args.fov,
        image_width=args.image_size,
        image_height=args.image_size,
    )
    
    question_config = QuestionConfig()
    if args.question_types:
        question_config.enabled_question_types = set(args.question_types)
        question_config.qa_types_to_generate = 'specific'
    
    # Render config
    render_config = RenderConfig(
        enable_rendering=not args.no_rendering,
        image_width=args.image_size,
        image_height=args.image_size,
    )
    
    config = PipelineConfig(
        scenes=args.scenes,
        num_scenes=args.num_scenes,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        camera_sampling=camera_config,
        question_config=question_config,
        render_config=render_config,
        max_objects_of_interest=args.max_objects,
        gpu_device_id=args.gpu,
        resume=args.resume,
    )
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("Multiview QA Generation Pipeline")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Experiment name: {args.experiment_name}")
    print(f"Move pattern: {args.move_pattern}")
    print(f"Scenes: {args.scenes}")
    print("=" * 60)
    
    # Build configuration
    config = build_config(args)
    
    # Create and run pipeline
    pipeline = MultiviewQAPipeline(config)
    
    try:
        qa_records, output_path = pipeline.run()
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"Total QA records: {len(qa_records)}")
        print(f"Output saved to: {output_path}")
        print("=" * 60)
        
        return 0
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
