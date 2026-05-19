"""Configuration for planning, rendering, and reporting the benchmark.

Example:
  BenchmarkConfig(
    output_dir="/tmp/video_consistency",
    interiorgs_root="/data/InteriorGS",
    thor_scenes=("FloorPlan1",),
    interiorgs_scenes=("0267_840790",),
  )
"""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: str
    interiorgs_root: str | None
    thor_scenes: tuple[str, ...]
    interiorgs_scenes: tuple[str, ...]
    enabled_engines: tuple[str, ...] = ("thor", "interiorgs")
    group_size: int = 30
    min_group_size: int = 25
    fps: int = 1
    gpu: int = 0
    plan_parallel_workers: int = 1
    render_parallel_workers: int = 1
    render_share_base_gpu: bool = False
    reuse_mined_plan_candidates: bool = False
    image_size: int = 384
    field_of_view: int = 75
    min_frames: int = 30
    increment: float = 10.0
    circular_increment: float = 6.0
    random_seed: int = 0
    linear_radius: float = 1.0
    thor_radii: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    interiorgs_image_size: int = 512
    interiorgs_fov: float = 60.0
    interiorgs_rotation_interval: float = 10.0
    interiorgs_linear_steps: int = 36
    interiorgs_half_span: float = 1.5
    interiorgs_max_scenes: int = 20
    interiorgs_mining_max_objects: int = 48
    interiorgs_rotation_pair_pose_samples: int = 24
    target_total_videos: int = 0
    exhaust_all_candidates: bool = True
    max_groups_per_scene: int = 0  # 0 = no cap; >0 = cap groups per scene for balanced distribution
    reuse_video_dirs: tuple[str, ...] = ()  # directories with existing rendered videos for reuse scoring
    plan_gpus: tuple[int, ...] = ()  # GPU IDs for plan-stage mining (round-robin); empty = use self.gpu

    def validate(self) -> None:
        assert self.group_size >= 1, "group_size must be >= 1."
        assert self.min_group_size >= 1, "min_group_size must be >= 1."
        assert self.min_group_size <= self.group_size, "min_group_size must be <= group_size."
        assert Path(self.output_dir).parent.exists(), f"Parent directory missing for {self.output_dir}"
        assert self.enabled_engines, "At least one engine must be enabled."
        assert set(self.enabled_engines).issubset({"thor", "interiorgs"}), f"Unknown engines: {self.enabled_engines}"
        if "thor" in self.enabled_engines:
            assert self.thor_scenes, "At least one THOR scene is required when THOR is enabled."
        if "interiorgs" in self.enabled_engines:
            assert self.interiorgs_root is not None, "InteriorGS root is required when InteriorGS is enabled."
            assert Path(self.interiorgs_root).exists(), f"InteriorGS root not found: {self.interiorgs_root}"
            assert self.interiorgs_scenes, "At least one InteriorGS scene is required when InteriorGS is enabled."
        assert self.plan_parallel_workers >= 1, "plan_parallel_workers must be >= 1."
        assert self.render_parallel_workers >= 1, "render_parallel_workers must be >= 1."

    def metadata_path(self) -> Path:
        return Path(self.output_dir) / "metadata.json"

    def plan_path(self) -> Path:
        return self.metadata_path()

    def qa_path(self) -> Path:
        return Path(self.output_dir) / "qa.json"

    def clips_path(self) -> Path:
        return Path(self.output_dir) / "clips.jsonl"

    def groups_path(self) -> Path:
        return Path(self.output_dir) / "consistency_groups.json"

    def stats_path(self) -> Path:
        return Path(self.output_dir) / "dataset_stats.json"

    def report_path(self) -> Path:
        return Path(self.output_dir) / "video_consistency_dataset.md"

    def videos_root(self) -> Path:
        return Path(self.output_dir) / "videos"

    def to_dict(self) -> dict:
        return asdict(self)
