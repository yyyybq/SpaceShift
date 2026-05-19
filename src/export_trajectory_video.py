#!/usr/bin/env python3
"""
Export curated AI2-THOR trajectory demo videos.

Usage:
    python src/export_trajectory_video.py --trajectory all
    python src/export_trajectory_video.py --scene FloorPlan203 --trajectory around_cw
    python src/export_trajectory_video.py --scene auto --trajectory all --candidate_scenes FloorPlan2 FloorPlan203

Input spec:
    - scene: AI2-THOR scene name or `auto` to score candidate scenes
    - output_dir: directory replaced with the curated demo export
    - trajectory: one taxonomy trajectory or `all`
    - object_index, object_id, object_type: optional object filters for object-centric exports
    - radius, arc_degrees, increment: trajectory sampling controls
    - fov, image_size, fps, gpu, min_frames: rendering and acceptance controls
    - candidate_scenes: optional scene shortlist used when `scene=auto`

Output spec:
    - trajectory_output/<scene>/<trajectory>_<ObjectType>/frames/frame_*.png
    - trajectory_output/<scene>/<trajectory>_<ObjectType>/video.mp4
    - trajectory_output/<scene>/<trajectory>_<ObjectType>/manifest.json
    - trajectory_output/curated_manifest.json
"""

from trajectory_demos.run import main


if __name__ == "__main__":
    raise SystemExit(main())
