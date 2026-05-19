# Multiview QA Generation Pipeline for AI2THOR

A modular pipeline for generating spatial reasoning questions from AI2THOR scenes with multi-view camera sampling support. This pipeline combines:

- **Multi-view camera patterns** from `question_gen_InteriorGS` (around, spherical, rotation, linear)
- **AI2THOR simulator** from `spatial-scene-variations`
- **QA generation format** from `spatial-scene-variations/question_generation`

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Move Patterns](#move-patterns)
- [Question Types](#question-types)
- [Output Format](#output-format)
- [Configuration](#configuration)
- [API Reference](#api-reference)

## Overview

This pipeline generates spatial reasoning VQA datasets from AI2THOR 3D indoor scenes by:

1. **Object Selection** - Filtering meaningful objects from AI2THOR scene metadata
2. **Trajectory Sampling** - Generating camera trajectories around objects based on move patterns
3. **Image Rendering** - Rendering images from each camera pose using AI2THOR
4. **Question Generation** - Creating diverse question-answer pairs about spatial relationships
5. **Dataset Export** - Saving in JSONL format compatible with spatial-scene-variations

## Features

- **4 move patterns** for camera trajectories: `around`, `spherical`, `rotation`, `linear`
- **9 question types** covering size estimation, distance measurement, and spatial relationships
- **AI2THOR integration** for photorealistic rendering and accurate 3D metadata
- **Multi-view support** with trajectory grouping for multi-image reasoning
- **Progressive saving** with resume support for long-running jobs
- **Flexible configuration** via command-line arguments or JSON config files

## Installation

```bash
# Navigate to the module directory
cd /path/to/spatial-scene-variations/src/multiview_qa_gen

# Install dependencies
pip install ai2thor numpy scipy pillow tqdm

# Optional: Install in development mode
pip install -e .
```

### Requirements

- Python 3.8+
- ai2thor
- numpy
- scipy
- pillow
- tqdm

## Quick Start

### Basic Usage

```bash
# Generate questions for a single scene
python run_pipeline.py \
    --scenes FloorPlan201 \
    --output_dir ./output \
    --move_pattern around
```

### Full Example

```bash
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
```

### Using Config File

```bash
python run_pipeline.py --config config.json --output_dir ./output
```

Example `config.json`:
```json
{
    "scenes": ["FloorPlan201", "FloorPlan301"],
    "experiment_name": "multiview_exp",
    "camera_sampling": {
        "move_pattern": "around",
        "total_rotation": 90,
        "increment_rotation": 15,
        "radius": 1.5
    },
    "max_objects_of_interest": 50
}
```

## Pipeline Architecture

```
AI2THOR Scene
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Object Selection                                      │
│    - Parse AI2THOR object metadata                       │
│    - Filter by semantic category (blacklist)             │
│    - Filter by visibility and geometry                   │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Camera Trajectory Sampling                            │
│    - Sample trajectories based on move_pattern           │
│    - Validate reachability using GetReachablePositions   │
│    - Compute horizon angles to face target objects       │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ 3. For Each Camera Pose:                                 │
│    a. Teleport agent to pose                             │
│    b. Render image                                       │
│    c. Generate QA pairs from visible objects             │
│    d. Save progressively                                 │
└─────────────────────────────────────────────────────────┘
      │
      ▼
Output: JSONL dataset + images
```

## Move Patterns

### Pattern Overview

| Pattern | Description | Use Case |
|---------|-------------|----------|
| `around` | Horizontal circle around object | Multi-view of single object |
| `spherical` | Sample on sphere surface | Varied viewing angles (up/down) |
| `rotation` | Stand at room center, rotate 360° | Room panorama |
| `linear` | Walk toward or past object | Approach/passing motion |

### Around Pattern (Default)

Camera moves in a horizontal circle around the target object.

```
         cam_2
          ○
         /
cam_1 ○   [OBJ]   ○ cam_3
           \
            ○
          cam_4
```

Parameters:
- `radius`: Distance from object center (meters)
- `total_rotation`: Arc coverage (degrees)
- `increment_rotation`: Angle between poses (degrees)

### Spherical Pattern

Camera samples on a sphere surface around the object, providing varied elevation angles.

### Rotation Pattern

Camera stands at room center and rotates 360°, capturing the entire room.

### Linear Pattern

Camera walks in a straight line toward or past the object.

Sub-patterns:
- `approach`: Walk toward the object (object gets larger)
- `pass_by`: Walk sideways past the object (object moves across FOV)

## Question Types

### Single-Object Questions

| Type | Description | Answer Format |
|------|-------------|---------------|
| `object_size` | Length, width, height | `[L, W, H]` |
| `object_distance_to_camera` | Distance from camera | Number (m) |

### Pair-Object Questions

| Type | Description | Answer Format |
|------|-------------|---------------|
| `object_size_comparison_relative` | Size ratio | Number |
| `object_size_comparison_absolute` | Given one, find other | Number (m) |
| `object_pair_distance_center` | Distance between objects | Number (m) |
| `object_pair_distance_center_w_size` | Distance given object size | Number (m) |

### Multi-Object Questions

| Type | Description | Answer Format |
|------|-------------|---------------|
| `object_comparison_absolute_distance` | Given one distance, find another | Number (m) |
| `object_comparison_relative_distance` | Distance ratio | Number |

### Vector Questions

| Type | Description | Answer Format |
|------|-------------|---------------|
| `object_pair_distance_vector` | Vector in agent coordinates | `[x, y, z]` |

## Output Format

### Directory Structure

```
output/
└── experiment_name/
    ├── images/
    │   └── scene_pattern/
    │       ├── image_001.png
    │       ├── image_002.png
    │       └── ...
    ├── qa_pairs.json          # Full nested format
    ├── qa_pairs.jsonl         # Streaming format
    ├── qa_pairs_flat.json     # One entry per question
    ├── statistics.json        # Dataset statistics
    └── pipeline.log           # Execution log
```

### QA Record Format

```json
{
  "scene_id": "FloorPlan201",
  "image_id": "FloorPlan201_pos1.0_2.0_angle045_fov75_pitch0__FloorPlan201_around",
  "image_path": "./output/images/FloorPlan201_around/image_001.png",
  "camera_id": "FloorPlan201_pos1.0_2.0_angle045_fov75_pitch0",
  "move_pattern": "around",
  "field_of_view": 75,
  "trajectory_id": "FloorPlan201_Chair|1_around",
  "num_views_in_trajectory": 7,
  "all_view_paths": ["path1.png", "path2.png", ...],
  "qa_pairs": [
    {
      "question": "What is the estimated length, width, and height of the Chair...",
      "answer": "[0.6, 0.5, 0.9]",
      "question_type": "object_size",
      "question_id": "object_size_Chair|1",
      "primary_object": "Chair|1"
    }
  ]
}
```

### Flat QA Format

```json
{
  "scene_id": "FloorPlan201",
  "image_path": "./output/images/...",
  "question": "What is the estimated distance...",
  "answer": "1.5",
  "question_type": "object_distance_to_camera",
  "question_id": "object_distance_to_camera_Chair|1",
  "move_pattern": "around",
  "num_views_in_trajectory": 7,
  "all_view_paths": [...]
}
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scenes` | FloorPlan201 | AI2THOR scene names |
| `--move_pattern` | around | Camera trajectory pattern |
| `--total_rotation` | 90 | Arc coverage in degrees |
| `--increment_rotation` | 15 | Angle between poses |
| `--radius` | 1.5 | Distance from object (m) |
| `--fov` | 75 | Camera field of view |
| `--max_objects` | 50 | Max objects per scene |

### Object Filtering

Objects are filtered based on:
- **Blacklist**: floor, window, curtains, blinds, walls, shelf
- **Visibility**: Must be visible in current view
- **Uniqueness**: Only one object per type (no duplicates)

## API Reference

### Python API

```python
from multiview_qa_gen import (
    PipelineConfig,
    CameraSamplingConfig,
    MultiviewQAPipeline
)

# Configure
config = PipelineConfig(
    scenes=['FloorPlan201'],
    output_dir='./output',
    camera_sampling=CameraSamplingConfig(
        move_pattern='around',
        radius=1.5
    )
)

# Run pipeline
pipeline = MultiviewQAPipeline(config)
qa_records, output_path = pipeline.run()
```

### Module Structure

```
multiview_qa_gen/
├── __init__.py              # Package exports
├── config.py                # Configuration dataclasses
├── object_selector.py       # Object filtering
├── camera_sampler.py        # Trajectory sampling
├── question_templates.py    # Question templates
├── question_utils.py        # QA construction utilities
├── question_generator.py    # Question generator
├── pipeline.py              # Main pipeline
├── run_pipeline.py          # CLI entry point
└── README.md                # This file
```

## Comparison with Related Projects

| Feature | question_gen_InteriorGS | spatial-scene-variations | multiview_qa_gen |
|---------|-------------------------|--------------------------|------------------|
| Simulator | Gaussian Splatting | AI2THOR | AI2THOR |
| Move Patterns | ✓ (4 patterns) | ✗ (single view) | ✓ (4 patterns) |
| QA Format | Custom | JSONL | JSONL (compatible) |
| Multi-view | ✓ | ✗ | ✓ |

## License

MIT License
