"""Import helpers for the InteriorGS pipeline package.

Exports:
  load_interiorgs_symbols()
"""

import importlib
import sys
from pathlib import Path


def load_interiorgs_symbols() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "spatial-scene-variations-InteriorGS-QA-pipeline",
        repo_root / "old" / "archive_root" / "spatial-scene-variations-InteriorGS-QA-pipeline",
    ]
    package_root = next((path for path in candidates if path.exists()), None)
    assert package_root is not None, f"InteriorGS package not found in any of: {candidates}"
    package_str = str(package_root)
    if package_str not in sys.path:
        sys.path.insert(0, package_str)
    return {
        "CameraSampler": importlib.import_module("camera_sampler").CameraSampler,
        "CameraPose": importlib.import_module("camera_utils").CameraPose,
        "ObjectSelector": importlib.import_module("object_selector").ObjectSelector,
        "SceneObject": importlib.import_module("object_selector").SceneObject,
        "scene_object_to_aabb": importlib.import_module("camera_sampler").scene_object_to_aabb,
        "get_visible_objects": importlib.import_module("camera_sampler").get_visible_objects,
        "look_at_matrix": importlib.import_module("render_utils").look_at_matrix,
        "compute_intrinsics": importlib.import_module("render_utils").compute_intrinsics,
        "SceneRenderer": importlib.import_module("render_utils").SceneRenderer,
        "RenderConfig": importlib.import_module("render_utils").RenderConfig,
        "ObjectSelectionConfig": importlib.import_module("config").ObjectSelectionConfig,
        "CameraSamplingConfig": importlib.import_module("config").CameraSamplingConfig,
    }
