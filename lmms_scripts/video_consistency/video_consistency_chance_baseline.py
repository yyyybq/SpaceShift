"""Weighted best-constant MRA baseline for numeric SceneShift JSONL.

Mirrors ``find_optimal_prediction_na`` in the repo-root ``chance_level.py`` so
installs that only ship ``lmms_eval`` still get the same baseline.

``weighted_chance_mra_percent(path, question_type=...)`` restricts rows before aggregating;
``slice_question_type=None`` means the full JSONL (same as original single-argument call).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lmms_eval.tasks.sceneshift.utils import SUPPORTED_NUMERIC_QUESTION_TYPES, parse_numeric_answer_from_response

MRA_START = 0.5
MRA_END = 0.95
MRA_INTERVAL = 0.05
CHANCE_BASELINE_CACHE_VERSION = 2


def _vectorized_mra(pred, targets, start=MRA_START, end=MRA_END, interval=MRA_INTERVAL):
    targets = np.asarray(targets, dtype=float)
    num_pts = int((end - start) / interval + 2)
    thresholds = np.linspace(start, end, num_pts)
    rel_errors = np.where(targets != 0, np.abs(pred - targets) / np.abs(targets), np.where(pred == 0, 0.0, np.inf))
    accuracy_matrix = rel_errors[:, None] <= (1 - thresholds[None, :])
    return accuracy_matrix.mean()


def find_optimal_prediction_na(sub_docs: pd.DataFrame):
    targets = sub_docs["ground_truth"].apply(lambda x: float(parse_numeric_answer_from_response(x))).values
    candidates = list(np.unique(targets))
    extra = set()
    for t in np.unique(targets):
        for ci in np.arange(MRA_START, MRA_END + 0.01, MRA_INTERVAL):
            if t != 0:
                extra.add(t * ci)
                extra.add(t * (2 - ci))
    candidates = np.unique(candidates + list(extra))
    best_pred, best_mra = None, -np.inf
    for pred in candidates:
        mra = _vectorized_mra(pred, targets)
        if mra > best_mra:
            best_mra = mra
            best_pred = pred
    return best_pred, {"MRA": best_mra}


def weighted_chance_mra_percent(jsonl_path: Path, question_type: str | None = None) -> float:
    rows = [json.loads(line) for line in Path(jsonl_path).open(encoding="utf-8") if line.strip()]
    docs = pd.DataFrame(rows)
    assert not docs.empty
    if question_type is not None:
        docs = docs[docs["question_type"] == question_type]
    assert not docs.empty
    acc = 0.0
    total = 0.0
    for qt, sub in docs.groupby("question_type"):
        if qt not in SUPPORTED_NUMERIC_QUESTION_TYPES:
            continue
        _, metrics = find_optimal_prediction_na(sub)
        n = len(sub)
        acc += float(metrics["MRA"]) * 100.0 * n
        total += n
    assert total > 0
    return acc / total


def weighted_chance_mra_percent_cached(
    jsonl_path: Path,
    cache_path: Path,
    *,
    slice_question_type: str | None = None,
) -> float:
    js = jsonl_path.resolve()
    mt = js.stat().st_mtime
    slice_key = slice_question_type if slice_question_type is not None else "__overall__"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            payload.get("jsonl") == str(js)
            and float(payload.get("jsonl_mtime", -1)) == mt
            and payload.get("slice_question_type") == slice_key
            and int(payload.get("cache_version", -1)) == CHANCE_BASELINE_CACHE_VERSION
            and float(payload.get("mra_start", -1.0)) == MRA_START
            and float(payload.get("mra_end", -1.0)) == MRA_END
            and float(payload.get("mra_interval", -1.0)) == MRA_INTERVAL
        ):
            return float(payload["chance_MRA_pct"])
    v = weighted_chance_mra_percent(js, slice_question_type)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "jsonl": str(js),
                "jsonl_mtime": mt,
                "slice_question_type": slice_key,
                "cache_version": CHANCE_BASELINE_CACHE_VERSION,
                "mra_start": MRA_START,
                "mra_end": MRA_END,
                "mra_interval": MRA_INTERVAL,
                "chance_MRA_pct": v,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return v
