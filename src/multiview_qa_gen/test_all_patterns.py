#!/usr/bin/env python3
"""
Test script to generate examples for each move pattern.

This script runs the multiview QA generation pipeline for each of the 4 move patterns,
generating a few examples for each to verify the pipeline works correctly.

Usage:
    python test_all_patterns.py
    
    # With custom output directory
    python test_all_patterns.py --output_dir ./my_test_output
    
    # Test specific patterns only
    python test_all_patterns.py --patterns around spherical
    
    # Quick test with minimal objects
    python test_all_patterns.py --quick
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multiview_qa_gen.config import (
    PipelineConfig, CameraSamplingConfig, QuestionConfig, RenderConfig
)
from multiview_qa_gen.pipeline import MultiviewQAPipeline


# Available move patterns and their configurations
PATTERN_CONFIGS = {
    'around': {
        'description': 'Horizontal circle around object',
        'config': {
            'move_pattern': 'around',
            'total_rotation': 60,
            'increment_rotation': 20,
            'radius': 1.5,
        }
    },
    'spherical': {
        'description': 'Sample on sphere surface around object',
        'config': {
            'move_pattern': 'spherical',
            'spherical_samples': 15,
            'radius': 1.5,
        }
    },
    'rotation': {
        'description': 'Stand at room center, rotate 360°',
        'config': {
            'move_pattern': 'rotation',
            'total_rotation': 360,
            'increment_rotation': 45,
        }
    },
    'linear_approach': {
        'description': 'Walk toward object',
        'config': {
            'move_pattern': 'linear',
            'linear_sub_pattern': 'approach',
            'radius': 2.0,
        }
    },
    'linear_passby': {
        'description': 'Walk past object sideways',
        'config': {
            'move_pattern': 'linear',
            'linear_sub_pattern': 'pass_by',
            'radius': 1.5,
        }
    },
}


def run_pattern_test(pattern_name: str, pattern_config: dict,
                     output_dir: str, scene: str, max_objects: int,
                     num_cameras: int, enable_rendering: bool,
                     gpu: int = 0) -> dict:
    """
    Run pipeline test for a single pattern.
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing pattern: {pattern_name}")
    print(f"Description: {pattern_config['description']}")
    print(f"{'='*60}")
    
    # Build camera config
    cam_cfg = pattern_config['config'].copy()
    cam_cfg['num_cameras_per_item'] = num_cameras
    camera_config = CameraSamplingConfig(**cam_cfg)
    
    # Build render config
    render_config = RenderConfig(enable_rendering=enable_rendering)
    
    # Build pipeline config
    config = PipelineConfig(
        scenes=[scene],
        output_dir=output_dir,
        experiment_name=f"test_{pattern_name}",
        camera_sampling=camera_config,
        render_config=render_config,
        max_objects_of_interest=max_objects,
        gpu_device_id=gpu,
    )
    
    # Run pipeline
    pipeline = MultiviewQAPipeline(config)
    
    try:
        qa_records, output_path = pipeline.run()
        
        result = {
            'pattern': pattern_name,
            'success': True,
            'num_records': len(qa_records),
            'num_questions': sum(len(r.get('qa_pairs', [])) for r in qa_records),
            'output_path': output_path,
            'error': None,
        }
        
        print(f"✓ Success: {result['num_records']} records, {result['num_questions']} questions")
        
    except Exception as e:
        result = {
            'pattern': pattern_name,
            'success': False,
            'num_records': 0,
            'num_questions': 0,
            'output_path': None,
            'error': str(e),
        }
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Test multiview QA generation for all move patterns'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./test_output',
        help='Output directory for test results'
    )
    parser.add_argument(
        '--scene', type=str, default='FloorPlan201',
        help='AI2THOR scene to test with'
    )
    parser.add_argument(
        '--patterns', type=str, nargs='+', default=None,
        choices=list(PATTERN_CONFIGS.keys()),
        help='Specific patterns to test (default: all)'
    )
    parser.add_argument(
        '--max_objects', type=int, default=3,
        help='Maximum objects per scene (default: 3 for quick testing)'
    )
    parser.add_argument(
        '--num_cameras', type=int, default=3,
        help='Number of camera poses per object (default: 3)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test mode (1 object, 2 cameras)'
    )
    parser.add_argument(
        '--no_rendering', action='store_true',
        help='Disable image rendering'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '--clean', action='store_true',
        help='Clean output directory before running'
    )
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.max_objects = 1
        args.num_cameras = 2
    
    # Determine patterns to test
    patterns_to_test = args.patterns or list(PATTERN_CONFIGS.keys())
    
    # Clean output if requested
    if args.clean and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Header
    print("\n" + "=" * 60)
    print("Multiview QA Generation - Pattern Tests")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Scene: {args.scene}")
    print(f"Patterns to test: {patterns_to_test}")
    print(f"Max objects: {args.max_objects}")
    print(f"Num cameras: {args.num_cameras}")
    print(f"Rendering: {'disabled' if args.no_rendering else 'enabled'}")
    print("=" * 60)
    
    # Run tests
    results = []
    for pattern_name in patterns_to_test:
        pattern_config = PATTERN_CONFIGS[pattern_name]
        result = run_pattern_test(
            pattern_name=pattern_name,
            pattern_config=pattern_config,
            output_dir=args.output_dir,
            scene=args.scene,
            max_objects=args.max_objects,
            num_cameras=args.num_cameras,
            enable_rendering=not args.no_rendering,
            gpu=args.gpu,
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    total_success = 0
    total_questions = 0
    
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['pattern']:20s} - {r['num_questions']:4d} questions")
        if r['success']:
            total_success += 1
            total_questions += r['num_questions']
    
    print("-" * 60)
    print(f"  Total: {total_success}/{len(results)} passed, {total_questions} questions generated")
    print("=" * 60)
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'test_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'scene': args.scene,
            'patterns_tested': patterns_to_test,
            'total_success': total_success,
            'total_questions': total_questions,
            'results': results,
        }, f, indent=2)
    
    print(f"\nTest summary saved to: {summary_path}")
    
    # Return exit code based on results
    return 0 if total_success == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
