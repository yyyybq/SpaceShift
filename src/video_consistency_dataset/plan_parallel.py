"""Parallel plan-stage mining: one process per scene (spawn)."""

import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from video_consistency_dataset.benchmark_config import BenchmarkConfig


def _thor_plan_worker(payload: bytes) -> tuple[str, list[dict]]:
    config, scene_id = pickle.loads(payload)
    from video_consistency_dataset.thor_mining import mine_thor_single_scene

    assert isinstance(config, BenchmarkConfig)
    return scene_id, mine_thor_single_scene(config, scene_id, quiet=True)


def _interiorgs_plan_worker(payload: bytes) -> tuple[str, list[dict]]:
    config, scene_id, scene_index, total_scenes = pickle.loads(payload)
    from video_consistency_dataset.interiorgs_imports import load_interiorgs_symbols
    from video_consistency_dataset.interiorgs_mining_scene import _mine_interiorgs_scene_core

    assert isinstance(config, BenchmarkConfig)
    symbols = load_interiorgs_symbols()
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
    return scene_id, _mine_interiorgs_scene_core(
        config, scene_id, scene_index, total_scenes, symbols, selector, sampler, quiet=True
    )


def parallel_mine_thor(config: BenchmarkConfig) -> list[dict]:
    scenes = list(config.thor_scenes)
    n = max(1, min(config.plan_parallel_workers, len(scenes)))
    print(f"[plan] THOR parallel scenes={len(scenes)} max_workers={n}", flush=True)
    ctx = multiprocessing.get_context("spawn")
    results: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
        futs = {
            pool.submit(_thor_plan_worker, pickle.dumps((config, sid))): sid
            for sid in scenes
        }
        with tqdm(
            total=len(futs),
            desc="THOR plan",
            unit="scene",
            dynamic_ncols=True,
            file=sys.stderr,
        ) as pbar:
            for fut in as_completed(futs):
                scene_id, lst = fut.result()
                results[scene_id] = lst
                pbar.set_postfix_str(f"{scene_id} ({len(lst)} cands)", refresh=False)
                pbar.update(1)
    merged: list[dict] = []
    for sid in scenes:
        merged.extend(results[sid])
    print(f"[plan] THOR mined total candidates={len(merged)}", flush=True)
    return merged


def parallel_mine_interiorgs(config: BenchmarkConfig) -> list[dict]:
    scenes = list(config.interiorgs_scenes)
    n = max(1, min(config.plan_parallel_workers, len(scenes)))
    print(
        f"[plan] InteriorGS parallel scenes={len(scenes)} max_workers={n} "
        "(each process loads InteriorGS modules once; worker logs suppressed — bar advances when a scene finishes)",
        flush=True,
    )
    ctx = multiprocessing.get_context("spawn")
    results: dict[str, list[dict]] = {}
    total = len(scenes)
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
        futs = {}
        for scene_index, scene_id in enumerate(scenes, start=1):
            payload = pickle.dumps((config, scene_id, scene_index, total))
            futs[pool.submit(_interiorgs_plan_worker, payload)] = scene_id
        with tqdm(
            total=len(futs),
            desc="InteriorGS plan",
            unit="scene",
            dynamic_ncols=True,
            file=sys.stderr,
        ) as pbar:
            for fut in as_completed(futs):
                scene_id, lst = fut.result()
                results[scene_id] = lst
                pbar.set_postfix_str(scene_id[:20], refresh=False)
                pbar.update(1)
    merged: list[dict] = []
    for sid in scenes:
        merged.extend(results[sid])
    return merged
