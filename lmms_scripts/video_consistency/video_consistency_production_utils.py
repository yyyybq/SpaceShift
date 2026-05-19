"""Per-sample MRA/CAPE accuracy and per-group CV consistency for video consistency evaluation.

Query group
-----------
Clips sharing the same ``group_id`` ask the same spatial question about the
same object(s) from different viewpoints / trajectories.  An ideal model
produces identical numeric answers across the group, yielding CV = 0.

Metrics
-------
MRA  — Mean Relative Accuracy (%, higher is better)
CAPE — Clipped Absolute Percentage Error (0–1, lower is better)
CV   — Coefficient of Variation of parsed predictions within each query group
       (dimensionless, lower is better; 0 = perfectly consistent)

Breakdowns: question_type, question_family, engine, motion_family.

Usage (JSONL row fields consumed)::

    group_id          — query-group key for consistency
    question_type     — e.g. object_dimensions, object_distance_to_camera
    question_family   — e.g. size, camera_distance, pair_distance
    engine            — e.g. thor, interiorgs
    motion_family     — e.g. approach, around, rotation

Input ``results`` rows come from ``sceneshift_process_results`` and already
contain ``prediction_parse``, ``ground_truth_parse``, ``MRA``, ``CAPE``.

After each eval, ``EvaluationTracker`` calls ``video_consistency_eval_sidecars``
to write ``*_metrics_video_consistency.json`` and ``*_mra_cv_quadrants.{png,pdf}``.
"""

import re
from collections import OrderedDict
from typing import Any

import pandas as pd

from lmms_eval.tasks.sceneshift.utils import to_float

BREAKDOWN_AXES = [
    ("question_type", "qtype"),
    ("question_family", "qfamily"),
    ("engine", "engine"),
    ("motion_family", "motion"),
]

VIDEO_CONSISTENCY_COMPAT_TASKS = [
    "video_consistency_thor_eval",
    "video_consistency_thor_eval_v2",
    "video_consistency_thor_small",
    "video_consistency_production",
    "small_blind",
    "view_variation_image",
    "image_consistency_thor_eval_v2",
    "scene_variation",
]


def calculate_cv(series):
    """Coefficient of Variation: std / mean.  Returns None when mean is zero."""
    mu = series.mean()
    if mu == 0 or pd.isna(mu):
        return None
    return series.std() / mu


def _group_cv(df):
    """Mean CV of float predictions across query groups (``group_id``).

    Groups with fewer than 2 parseable predictions are excluded.
    Returns (mean_cv, n_groups).
    """
    if "group_id" not in df.columns:
        return None, 0
    pf = df["prediction_parse"].apply(to_float)
    valid = df.assign(_pf=pf).dropna(subset=["_pf"])
    if valid.empty:
        return None, 0
    cvs = (
        valid.groupby("group_id")["_pf"]
        .apply(lambda s: calculate_cv(s) if len(s) >= 2 else None)
        .dropna()
    )
    if cvs.empty:
        return None, 0
    return float(cvs.mean()), int(len(cvs))


def _emit(df, out, tag):
    """Emit MRA, CAPE, and CV for a data slice under *tag* prefix."""
    out[f"{tag}_n"] = int(len(df))

    if "MRA" in df.columns:
        vals = df["MRA"].dropna()
        if not vals.empty:
            out[f"{tag}_MRA"] = round(float(vals.mean()) * 100.0, 2)

    if "CAPE" in df.columns:
        vals = df["CAPE"].dropna()
        if not vals.empty:
            out[f"{tag}_CAPE"] = round(float(vals.mean()), 4)

    mcv, ng = _group_cv(df)
    if mcv is not None:
        out[f"{tag}_mean_CV"] = round(mcv, 4)
        out[f"{tag}_n_groups"] = ng


def video_consistency_aggregate_results(results):
    """Aggregate per-sample results into accuracy (MRA/CAPE) and consistency (CV)."""
    out: OrderedDict = OrderedDict()
    df = pd.DataFrame(results)
    if df.empty:
        return out

    _emit(df, out, "overall")

    for col, prefix in BREAKDOWN_AXES:
        if col not in df.columns:
            continue
        for val, sub in sorted(df.groupby(col)):
            _emit(sub, out, f"{prefix}:{val}")

    return out


def extract_video_consistency_flat_metrics(results: dict, task_name: str | None = None) -> dict[str, Any] | None:
    candidates = [task_name] if task_name is not None else VIDEO_CONSISTENCY_COMPAT_TASKS
    for name in candidates:
        task = results.get("results", {}).get(name)
        if task:
            break
    if not task:
        return None
    for v in task.values():
        if isinstance(v, dict) and "overall_MRA" in v:
            return dict(v)
    return None


def _parse_vc_metric_groups(flat: dict) -> dict[str, Any]:
    overall = {
        "MRA": flat.get("overall_MRA"),
        "CAPE": flat.get("overall_CAPE"),
        "mean_CV": flat.get("overall_mean_CV"),
        "n": flat.get("overall_n"),
        "n_groups": flat.get("overall_n_groups"),
    }
    prefix_bucket = {
        "qtype": "by_question_type",
        "qfamily": "by_question_family",
        "engine": "by_engine",
        "motion": "by_motion_family",
        "etype": "by_edit_type",
    }
    pat = re.compile(r"^(qtype|qfamily|engine|motion|etype):(.+)_(n|MRA|CAPE|mean_CV|n_groups)$")
    tmp: dict[str, dict[str, dict]] = {b: {} for b in prefix_bucket.values()}
    for key, val in flat.items():
        m = pat.match(key)
        if not m:
            continue
        pfx, name, field = m.group(1), m.group(2), m.group(3)
        bname = prefix_bucket.get(pfx)
        if bname is None:
            continue
        row = tmp[bname].setdefault(name, {})
        if field == "n":
            row["n"] = val
        elif field == "MRA":
            row["MRA"] = val
        elif field == "CAPE":
            row["CAPE"] = val
        elif field == "mean_CV":
            row["mean_CV"] = val
        elif field == "n_groups":
            row["n_groups"] = val
    out_buckets = {b: {k: v for k, v in sorted(inner.items())} for b, inner in tmp.items()}
    return {"overall": overall, **out_buckets}


def build_organized_video_consistency_metrics_for_task(results: dict, task_name: str) -> dict[str, Any] | None:
    """Nested MRA / CAPE / CV dict for a specific compat task."""
    flat = extract_video_consistency_flat_metrics(results, task_name=task_name)
    if flat is None:
        return None
    cfg = results.get("configs", {}).get(task_name) or {}
    dk = cfg.get("dataset_kwargs") or {}
    meta: dict[str, Any] = {
        "model_name": results.get("model_name") or results.get("config", {}).get("model"),
        "model_dtype": results.get("model_dtype"),
        "task": cfg.get("task", task_name),
        "dataset_files": dk.get("data_files"),
    }
    return {**meta, **_parse_vc_metric_groups(flat)}


def iter_organized_video_consistency_metrics(results: dict) -> list[dict[str, Any]]:
    """Return one organized metrics dict per compat task found in the payload."""
    out: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for task_name in VIDEO_CONSISTENCY_COMPAT_TASKS:
        if task_name in seen_tasks:
            continue
        organized = build_organized_video_consistency_metrics_for_task(results, task_name)
        if organized is None:
            continue
        out.append(organized)
        seen_tasks.add(task_name)
    return out


def build_organized_video_consistency_metrics(results: dict) -> dict[str, Any] | None:
    """Nested MRA / CAPE / CV dict for the first compat task in ``results``."""
    all_metrics = iter_organized_video_consistency_metrics(results)
    if not all_metrics:
        return None
    return all_metrics[0]
