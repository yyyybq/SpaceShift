"""InteriorGS candidate mining for the video consistency benchmark.

Candidate schema:
  {
    "engine": "interiorgs",
    "motion_family": "spherical",
    "question_type": "object_distance_to_camera",
    "anchor_ids": ["chair_1"],
    "available_clips": [{...}]
  }
"""

import time

from tqdm import tqdm

from video_consistency_dataset.interiorgs_imports import load_interiorgs_symbols
from video_consistency_dataset.interiorgs_mining_scene import _mine_interiorgs_scene_core


def mine_interiorgs_candidates(config) -> list[dict]:
    if config.plan_parallel_workers > 1:
        from video_consistency_dataset.plan_parallel import parallel_mine_interiorgs

        return parallel_mine_interiorgs(config)
    print("[plan] InteriorGS: loading camera/object modules (first run can take ~30–120s)...", flush=True)
    t0 = time.perf_counter()
    symbols = load_interiorgs_symbols()
    print(f"[plan] InteriorGS: modules loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    ObjectSelector = symbols["ObjectSelector"]
    ObjectSelectionConfig = symbols["ObjectSelectionConfig"]
    CameraSampler = symbols["CameraSampler"]
    CameraSamplingConfig = symbols["CameraSamplingConfig"]
    selector = ObjectSelector(ObjectSelectionConfig())
    sampler = CameraSampler(
        CameraSamplingConfig(
            image_width=config.interiorgs_image_size,
            image_height=config.interiorgs_image_size,
            fov_deg=config.interiorgs_fov,
            rotation_interval=config.interiorgs_rotation_interval,
        )
    )
    mined: list[dict] = []
    total_scenes = len(config.interiorgs_scenes)
    for scene_index, scene_id in enumerate(
        tqdm(config.interiorgs_scenes, desc="InteriorGS plan", unit="scene", dynamic_ncols=True),
        start=1,
    ):
        mined.extend(
            _mine_interiorgs_scene_core(config, scene_id, scene_index, total_scenes, symbols, selector, sampler)
        )
    return mined
