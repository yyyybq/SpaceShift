"""Scene list helpers for the benchmark.

Example:
  discover_interiorgs_scenes("/data/InteriorGS", max_scenes=20)
"""

from pathlib import Path

from trajectory_demos.demo_config import DEFAULT_CANDIDATE_SCENES


def default_thor_scenes() -> tuple[str, ...]:
    return DEFAULT_CANDIDATE_SCENES


def discover_interiorgs_scenes(root: str, max_scenes: int) -> tuple[str, ...]:
    scene_root = Path(root)
    scene_names = [
        path.name
        for path in sorted(scene_root.iterdir())
        if path.is_dir() and (path / "labels.json").exists() and (path / "structure.json").exists()
    ]
    assert scene_names, f"No InteriorGS scenes found under {root}"
    return tuple(scene_names[:max_scenes])
