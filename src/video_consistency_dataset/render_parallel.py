"""Multi-process render: shard scenes across workers (spawn).

Each worker handles a slice of scenes so controller reuse is maximized.
"""

import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from dataclasses import replace
from pathlib import Path

from video_consistency_dataset.benchmark_config import BenchmarkConfig
from video_consistency_dataset.plan_io import load_benchmark_plan


def _render_scene_shard(payload: bytes) -> dict[str, dict]:
    worker_id, num_workers, config, metadata_path, engine = pickle.loads(payload)
    from video_consistency_dataset.thor_render import render_thor_clips
    from video_consistency_dataset.interiorgs_render import render_interiorgs_clips

    assert isinstance(config, BenchmarkConfig)
    plan = load_benchmark_plan(Path(metadata_path))

    engine_clips = [c for c in plan.clips if c.engine == engine]
    scenes = sorted(set(c.scene_id for c in engine_clips))
    worker_scenes = set(scenes[i] for i in range(len(scenes)) if i % num_workers == worker_id)
    clip_ids = {c.clip_id for c in engine_clips if c.scene_id in worker_scenes}

    if not clip_ids:
        return {}

    records: dict[str, dict] = {}
    if engine == "thor":
        records.update(render_thor_clips(plan, config, clip_ids=clip_ids))
    elif engine == "interiorgs":
        records.update(render_interiorgs_clips(plan, config, clip_ids=clip_ids))

    print(
        f"[render] worker {worker_id + 1}/{num_workers} engine={engine} "
        f"scenes={len(worker_scenes)} clips_done={len(records)}",
        flush=True,
    )
    return records


def run_parallel_clip_render(config: BenchmarkConfig) -> dict[str, dict]:
    assert config.render_parallel_workers > 1
    meta = config.metadata_path()
    assert meta.is_file(), f"Missing plan at {meta}; run plan stage first."
    n = config.render_parallel_workers
    ctx = multiprocessing.get_context("spawn")
    merged: dict[str, dict] = {}

    engines = [e for e in config.enabled_engines]
    total_shards = n * len(engines)
    print(
        f"[render] parallel workers={n} engines={engines} "
        f"share_base_gpu={config.render_share_base_gpu}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
        futures = []
        for engine in engines:
            for w in range(n):
                payload = pickle.dumps((w, n, config, str(meta.resolve()), engine))
                futures.append(pool.submit(_render_scene_shard, payload))
        with tqdm(
            total=len(futures),
            desc="Render shards",
            unit="shard",
            dynamic_ncols=True,
            file=sys.stderr,
        ) as pbar:
            for fut in as_completed(futures):
                merged.update(fut.result())
                pbar.update(1)
    return merged
