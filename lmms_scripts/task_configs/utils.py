import ast
import os
import re
from collections import OrderedDict
from functools import partial

import datasets
import numpy as np
import pandas as pd
from loguru import logger as eval_logger
from PIL import Image, UnidentifiedImageError

_DEFAULT_IMAGE_BASE = "/nas/baiqiao/spatial-scene-variations/inference_data"
SCENESHIFT_IMAGE_BASE = os.environ.get("SCENESHIFT_IMAGE_BASE", _DEFAULT_IMAGE_BASE)

SUPPORTED_NUMERIC_QUESTION_TYPES = {
    "object_dimensions",
    "object_distance_to_camera",
    "object_pair_distance_center",
    "object_size",
    "object_size_comparison_transitivity",
}
MRA_METRIC_KEY = "MRA:.5:.95:.05"

_SCENESHIFT_DIMENSION_DEFINITION = (
    "Length and width are the two dimensions that define the 'base' of the object. "
    "Of these two base dimensions, let length be the longer and width be the shorter."
)
_SCENESHIFT_SIMPLE_NUMERIC_POST_PROMPT = (
    "Answer with only the numeric value in meters. Preserve useful decimal precision. Do not include any explanation."
)
_SCENESHIFT_ANSWER_DICT_RE = re.compile(r"\{\s*['\"]answer['\"]\s*:\s*(?P<value>[^{}]+?)\s*\}", flags=re.IGNORECASE | re.DOTALL)
_SCENESHIFT_DOUBLE_BRACE_ANSWER_DICT_RE = re.compile(r"\{\{\s*['\"]answer['\"]\s*:\s*(?P<value>[^{}]+?)\s*\}\}", flags=re.IGNORECASE | re.DOTALL)
_SCENESHIFT_LAST_LINE_NUMERIC_RE = re.compile(
    r"^(?:answer\s*[:=]\s*)?(?P<value>[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\.?$",
    flags=re.IGNORECASE,
)


def _load_image(path: str) -> Image.Image:
    resolved = _resolve_path(path)
    assert os.path.exists(resolved), f"Image file not found: '{resolved}' (original: '{path}')"
    return Image.open(resolved).convert("RGB")


def _resolve_path(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.join(SCENESHIFT_IMAGE_BASE, path)
    return path


def sceneshift_doc_to_visual(doc):
    video_path = doc.get("video_path")
    if video_path:
        path = _resolve_path(video_path)
        assert os.path.exists(path), f"Video not found: {path}"
        return [path]

    question_type = doc.get("question_type")
    assert question_type in SUPPORTED_NUMERIC_QUESTION_TYPES, f"Unsupported question type: {question_type!r}"

    image_path = doc.get("image_path", "")
    assert image_path, f"Missing image_path for question type {question_type!r}"
    return [_load_image(image_path)]


def sceneshift_blind_doc_to_visual(doc):
    """Blind ablation: provide the prompt but no image/video evidence."""
    return []


def _sceneshift_medium(doc: dict) -> str:
    return "video" if doc.get("video_path") else "image"


def _build_sceneshift_prompt(doc: dict) -> str:
    qtype = str(doc.get("question_type") or "")
    labels = list(doc.get("anchor_labels") or [])
    medium = _sceneshift_medium(doc)

    if qtype == "object_dimensions" and labels and doc.get("dimension"):
        dimension = str(doc["dimension"])
        return (
            f"What is the estimated {dimension} of the {labels[0]} in this {medium} in meters? "
            f"{_SCENESHIFT_DIMENSION_DEFINITION}"
        )

    if qtype == "object_distance_to_camera" and labels:
        return (
            f"What is the distance from the camera to the approximate center of the {labels[0]} "
            f"in this {medium}, in meters?"
        )

    if qtype == "object_pair_distance_center" and len(labels) >= 2:
        return (
            f"What is the center-to-center distance between {labels[0]} and {labels[1]} "
            f"in this {medium}, in meters?"
        )

    raise ValueError(f"Unsupported question type for prompt construction: {qtype}")


def _compose_sceneshift_prompt(base_prompt: str, lmms_eval_specific_kwargs: dict | None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = str(kwargs.get("pre_prompt", "")).strip()
    post_prompt_raw = kwargs.get("post_prompt", "")
    post_prompt = str(post_prompt_raw).strip() if str(post_prompt_raw).strip() else _SCENESHIFT_SIMPLE_NUMERIC_POST_PROMPT
    parts = [p for p in (pre_prompt, base_prompt.strip(), post_prompt) if p]
    return " ".join(parts)


def sceneshift_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    base_prompt = _build_sceneshift_prompt(doc)
    return _compose_sceneshift_prompt(base_prompt, lmms_eval_specific_kwargs)


def sceneshift_blind_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = dict(lmms_eval_specific_kwargs or {})
    blind_pre_prompt = (
        "No image or video is provided. Estimate from the question text, object categories, "
        "and typical indoor-scene priors only."
    )
    existing_pre_prompt = str(kwargs.get("pre_prompt", "")).strip()
    kwargs["pre_prompt"] = " ".join(p for p in (blind_pre_prompt, existing_pre_prompt) if p)
    return sceneshift_doc_to_text(doc, kwargs)


def scene_variation_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Use the prebuilt question text from the dataset; only attach optional pre/post prompts."""
    question = str(doc.get("question") or "").strip()
    assert question, "scene_variation doc is missing the 'question' field"
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = str(kwargs.get("pre_prompt", "")).strip()
    post_prompt = str(kwargs.get("post_prompt", "")).strip()
    parts = [p for p in (pre_prompt, question, post_prompt) if p]
    return " ".join(parts)


def _doc_media_path(doc: dict) -> str:
    video_path = doc.get("video_path")
    if video_path:
        return _resolve_path(video_path)
    question_type = doc.get("question_type")
    assert question_type in SUPPORTED_NUMERIC_QUESTION_TYPES, f"Unsupported question type: {question_type!r}"
    image_path = doc.get("image_path", "")
    assert image_path, f"Missing image_path for question type {question_type!r}"
    return _resolve_path(image_path)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        dataset = dataset.shuffle(seed=42)

    missing = [(idx, _doc_media_path(doc)) for idx, doc in enumerate(dataset) if not os.path.exists(_doc_media_path(doc))]
    if missing:
        sample = "\n".join(f"  [{idx}] {path}" for idx, path in missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"SceneShift task has {len(missing)}/{len(dataset)} docs whose media file does not exist on disk. "
            f"Re-render the missing assets or prune them from the source qa.json.\n"
            f"First missing entries:\n{sample}{suffix}"
        )
    return dataset


def process_docs_blind(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        dataset = dataset.shuffle(seed=42)
    return dataset


def _parse_answer_value_literal(value_text):
    if value_text is None:
        return None
    if isinstance(value_text, (int, float)):
        return value_text
    if not isinstance(value_text, str):
        return None

    value_text = value_text.strip()
    if not value_text:
        return None

    try:
        parsed = ast.literal_eval(value_text)
        if isinstance(parsed, dict):
            return to_float(parsed.get("answer"))
        if isinstance(parsed, (int, float)):
            return parsed
    except (ValueError, SyntaxError, TypeError):
        pass

    match = _SCENESHIFT_LAST_LINE_NUMERIC_RE.match(value_text)
    if match:
        number_text = match.group("value")
        try:
            if "." in number_text or "e" in number_text.lower():
                return float(number_text)
            return int(number_text)
        except (ValueError, TypeError):
            return None

    return None


def parse_numeric_answer_from_response(response_text):
    if response_text is None:
        return None
    if isinstance(response_text, (int, float)):
        return response_text
    if not isinstance(response_text, str):
        return None
    if not response_text.strip():
        return None

    non_empty_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if non_empty_lines:
        parsed = _parse_answer_value_literal(non_empty_lines[-1])
        if parsed is not None:
            return parsed

    for pattern in (_SCENESHIFT_DOUBLE_BRACE_ANSWER_DICT_RE, _SCENESHIFT_ANSWER_DICT_RE):
        matches = list(pattern.finditer(response_text))
        if matches:
            parsed = _parse_answer_value_literal(matches[-1].group("value"))
            if parsed is not None:
                return parsed

    number_matches = list(re.finditer(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", response_text))
    if number_matches:
        number_text = number_matches[-1].group(0)
        try:
            return float(number_text) if "." in number_text or "e" in number_text.lower() else int(number_text)
        except (ValueError, TypeError, OverflowError):
            return None

    print(f"Warning: Could not parse answer from response: {str(response_text)[:200]}...")
    return None


def abs_dist_norm(pred, target):
    if target == 0:
        return np.inf if pred != 0 else 0
    return abs(pred - target) / abs(target)


def clipped_abs_pct_error(pred, gt):
    """
    Compute the Clipped Absolute Percentage Error (CAPE).

    CAPE = min(1, |pred - gt| / gt)

    Args:
        pred (float or list of floats): Predicted value(s).
        gt (float or list of floats): Ground truth value(s).

    Returns:
        float: Error between 0 and 1 (if scalars),
               or mean error across elements (if lists).
    """

    # Case: both are lists
    if isinstance(pred, (list, tuple)) and isinstance(gt, (list, tuple)):
        if len(pred) != len(gt):
            raise ValueError("Pred and GT lists must have the same length.")
        errors = [min(1.0, abs(p - g) / abs(g)) if g != 0 else 1.0 for p, g in zip(pred, gt)]
        return sum(errors) / len(errors)

    # Case: scalars
    if not isinstance(pred, (int, float)) or not isinstance(gt, (int, float)):
        raise TypeError("Inputs must be floats or lists of floats.")
    if gt == 0:
        raise ValueError("Ground truth (gt) must be non-zero.")
    return min(1.0, abs(pred - gt) / abs(gt))


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


METRICS_FOR_NA = {
    MRA_METRIC_KEY: partial(mean_relative_accuracy, start=0.5, end=0.95, interval=0.05),
}

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    MRA_METRIC_KEY: 0.0,
    "MRA": 0.0,
    "CAPE": 1.0,
}

NUMERIC_METRICS = {
    MRA_METRIC_KEY: METRICS_FOR_NA[MRA_METRIC_KEY],
    "MRA": partial(mean_relative_accuracy, start=0.5, end=0.95, interval=0.05),
    "CAPE": clipped_abs_pct_error,
}


def to_float(pred):
    try:
        return float(pred)
    except (TypeError, ValueError):
        return None


def refresh_sceneshift_score_doc(doc):
    question_type = doc["question_type"]
    pred = parse_numeric_answer_from_response(doc.get("prediction"))
    raw_gt = doc.get("ground_truth_value") or doc.get("ground_truth")
    gt = parse_numeric_answer_from_response(raw_gt)
    doc["prediction_parse"] = pred
    doc["ground_truth_parse"] = gt

    if question_type not in SUPPORTED_NUMERIC_QUESTION_TYPES:
        raise ValueError(f"Unknown question type: {question_type}")

    pred_value = to_float(pred)
    gt_value = to_float(gt)
    doc["prediction_parse_success"] = pred_value is not None
    for key, metric_fn in NUMERIC_METRICS.items():
        try:
            doc[key] = metric_fn(pred_value, gt_value)
        except (TypeError, ValueError):
            doc[key] = WORST_CASE_FOR_METRICS[key]

    return doc


def sceneshift_process_results(doc, results):
    doc["prediction"] = results[0]
    refresh_sceneshift_score_doc(doc)

    return {"sceneshift_score": doc}


def sceneshift_aggregate_results(results):
    results_df = pd.DataFrame(results)

    if "question_type" in results_df.columns:
        results_df = results_df[results_df["question_type"] != "spot_difference_general"]

    final_scores = OrderedDict()

    metrics_to_agg = {"accuracy": "mean", MRA_METRIC_KEY: "mean", "MRA": "mean", "MRA_list": "mean", "cosine_similarity": "mean", "magnitude_mra": "mean", "CAPE": "mean", "CAPE_list": "mean", "magnitude_CAPE": "mean"}

    non_percentage_metrics = ["cosine_similarity", "CAPE", "CAPE_list", "magnitude_CAPE"]

    def get_valid_metrics(df):
        valid_metrics = {col: "mean" for col in metrics_to_agg.keys() if col in df.columns}
        return valid_metrics

    # --- Aggregation by question type ---
    if "question_type" in results_df.columns:
        valid_metrics_q_type = get_valid_metrics(results_df)
        if valid_metrics_q_type:
            question_type_scores = results_df.groupby("question_type").agg(valid_metrics_q_type).dropna(axis=1, how="all")

            for q_type, row in question_type_scores.iterrows():
                for metric, score in row.items():
                    if not pd.isna(score):
                        # Conditionally scale the score
                        if metric not in non_percentage_metrics:
                            score *= 100.0
                        final_scores[f"question:{q_type}_{metric}"] = score

    # --- Aggregation by edit type ---
    if "edit_type" in results_df.columns:
        valid_metrics_e_type = get_valid_metrics(results_df)
        if valid_metrics_e_type:
            edit_type_scores = results_df.groupby("edit_type").agg(valid_metrics_e_type).dropna(axis=1, how="all")

            for e_type, row in edit_type_scores.iterrows():
                for metric, score in row.items():
                    if not pd.isna(score):
                        # Conditionally scale the score
                        if metric not in non_percentage_metrics:
                            score *= 100.0
                        final_scores[f"edit_type:{e_type}_{metric}"] = score

    return final_scores


# ---------------------------------------------------------------------------
# Video-consistency aggregation (MRA / CAPE accuracy + CV consistency)
# ---------------------------------------------------------------------------

BREAKDOWN_AXES = [
    ("question_type", "qtype"),
    ("question_family", "qfamily"),
    ("engine", "engine"),
    ("motion_family", "motion"),
]


def _calculate_cv(series):
    """Coefficient of Variation: std / mean.

    Treat an exactly constant zero series as CV=0. For zero-mean but non-constant
    series, return None because the ratio is undefined.
    """
    arr = series.to_numpy(dtype=float, copy=False)
    if len(arr) == 0:
        return None
    if np.all(arr == 0):
        return 0.0

    mu = arr.mean()
    if pd.isna(mu):
        return None
    sigma = arr.std(ddof=1) if len(arr) >= 2 else 0.0
    if mu == 0:
        return 0.0 if sigma == 0 or pd.isna(sigma) else None
    if mu < 0:
        return None
    return sigma / mu


def _group_cv(df):
    if "group_id" not in df.columns:
        return {
            "mean_CV": None,
            "n_groups": 0,
            "total_groups_n": 0,
            "parse_success_n": 0,
            "parse_success_rate": None,
            "complete_groups_n": 0,
            "complete_group_rate": None,
            "parse_failed_groups_n": 0,
        }

    pf = df["prediction_parse"].apply(to_float)
    total_samples = int(len(df))
    parse_success_n = int(pf.notna().sum())
    parse_success_rate = (parse_success_n / total_samples) if total_samples else None

    total_groups_n = int(df["group_id"].nunique())
    cvs = []
    complete_groups_n = 0
    parse_failed_groups_n = 0

    for _, sub in df.assign(_pf=pf).groupby("group_id"):
        parsed = sub["_pf"].dropna()
        has_parse_failure = len(parsed) != len(sub)
        if has_parse_failure:
            parse_failed_groups_n += 1
            continue

        complete_groups_n += 1
        group_cv = _calculate_cv(parsed) if len(parsed) >= 2 else None

        if group_cv is not None:
            cvs.append(float(group_cv))

    n_groups = int(len(cvs))
    complete_group_rate = (complete_groups_n / total_groups_n) if total_groups_n else None
    mean_cv = float(np.mean(cvs)) if cvs else None
    return {
        "mean_CV": mean_cv,
        "n_groups": n_groups,
        "total_groups_n": total_groups_n,
        "parse_success_n": parse_success_n,
        "parse_success_rate": parse_success_rate,
        "complete_groups_n": complete_groups_n,
        "complete_group_rate": complete_group_rate,
        "parse_failed_groups_n": parse_failed_groups_n,
    }


def _emit(df, out, tag):
    out[f"{tag}_n"] = int(len(df))
    if "MRA" in df.columns:
        vals = df["MRA"].dropna()
        if not vals.empty:
            out[f"{tag}_MRA"] = round(float(vals.mean()) * 100.0, 2)
    if "CAPE" in df.columns:
        vals = df["CAPE"].dropna()
        if not vals.empty:
            out[f"{tag}_CAPE"] = round(float(vals.mean()), 4)
    cv_stats = _group_cv(df)
    if cv_stats["mean_CV"] is not None:
        out[f"{tag}_mean_CV"] = round(cv_stats["mean_CV"], 4)
        out[f"{tag}_n_groups"] = cv_stats["n_groups"]
    if cv_stats["total_groups_n"]:
        out[f"{tag}_total_groups_n"] = cv_stats["total_groups_n"]
    if cv_stats["parse_success_rate"] is not None:
        out[f"{tag}_parse_success_n"] = cv_stats["parse_success_n"]
        out[f"{tag}_parse_success_rate"] = round(cv_stats["parse_success_rate"], 4)
    if cv_stats["complete_group_rate"] is not None:
        out[f"{tag}_complete_groups_n"] = cv_stats["complete_groups_n"]
        out[f"{tag}_complete_group_rate"] = round(cv_stats["complete_group_rate"], 4)
    if cv_stats["parse_failed_groups_n"]:
        out[f"{tag}_parse_failed_groups_n"] = cv_stats["parse_failed_groups_n"]


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


SCENE_VARIATION_BREAKDOWN_AXES = [
    ("question_type", "qtype"),
    ("edit_type", "etype"),
]


def scene_variation_aggregate_results(results):
    """Aggregate scene_variation results by question_type and edit_type."""
    out: OrderedDict = OrderedDict()
    df = pd.DataFrame(results)
    if df.empty:
        return out
    _emit(df, out, "overall")
    for col, prefix in SCENE_VARIATION_BREAKDOWN_AXES:
        if col not in df.columns:
            continue
        for val, sub in sorted(df.groupby(col)):
            _emit(sub, out, f"{prefix}:{val}")
    if {"question_type", "edit_type"}.issubset(df.columns):
        for (etype, qtype), sub in sorted(df.groupby(["edit_type", "question_type"])):
            _emit(sub, out, f"etype_x_qtype:{etype}_x_{qtype}")
    return out


# ---------------------------------------------------------------------------
# Higher-order consistency aggregation
# ---------------------------------------------------------------------------

HIGHER_ORDER_TRIANGLE_EPS_ABS = 0.11
HIGHER_ORDER_TRIANGLE_WEAK_EPS_ABS = 0.22
HIGHER_ORDER_HIGH_MRA_THRESHOLD = 0.70
_SIZE_DIM_TO_INDEX = {"length": 0, "width": 1, "height": 2}


def _coerce_float_triplet(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("answer")
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError, TypeError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def parse_numeric_triplet_answer_from_response(response_text):
    if response_text is None:
        return None
    direct = _coerce_float_triplet(response_text)
    if direct is not None:
        return direct
    if not isinstance(response_text, str):
        return None

    non_empty_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if non_empty_lines:
        parsed = _coerce_float_triplet(non_empty_lines[-1])
        if parsed is not None:
            return parsed

    for pattern in (_SCENESHIFT_DOUBLE_BRACE_ANSWER_DICT_RE, _SCENESHIFT_ANSWER_DICT_RE):
        matches = list(pattern.finditer(response_text))
        for match in reversed(matches):
            parsed = _coerce_float_triplet(match.group("value"))
            if parsed is not None:
                return parsed

    list_matches = list(re.finditer(r"\[[^\[\]]+\]", response_text))
    for match in reversed(list_matches):
        parsed = _coerce_float_triplet(match.group(0))
        if parsed is not None:
            return parsed

    number_matches = list(re.finditer(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", response_text))
    if len(number_matches) >= 3:
        try:
            return [float(match.group(0)) for match in number_matches[-3:]]
        except (TypeError, ValueError):
            return None

    print(f"Warning: Could not parse size triplet from response: {str(response_text)[:200]}...")
    return None


def _coerce_choice_letter(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("answer")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        # Direct single-letter answer.
        if len(text) == 1 and text.upper() in {"A", "B"}:
            return text.upper()

        # Literal dict/list wrappers.
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return _coerce_choice_letter(parsed.get("answer"))
            if isinstance(parsed, str) and len(parsed.strip()) == 1 and parsed.strip().upper() in {"A", "B"}:
                return parsed.strip().upper()
        except (ValueError, SyntaxError, TypeError):
            pass

        # Generic "answer: A" patterns and fallback letter search.
        ans_match = re.search(r"answer\s*[:=]\s*['\"]?([AB])['\"]?", text, flags=re.IGNORECASE)
        if ans_match:
            return ans_match.group(1).upper()
        letters = re.findall(r"\b([AB])\b", text, flags=re.IGNORECASE)
        if letters:
            return letters[-1].upper()
    return None


def parse_choice_answer_from_response(response_text):
    if response_text is None:
        return None

    direct = _coerce_choice_letter(response_text)
    if direct is not None:
        return direct

    if not isinstance(response_text, str):
        return None

    non_empty_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if non_empty_lines:
        parsed = _coerce_choice_letter(non_empty_lines[-1])
        if parsed is not None:
            return parsed

    for pattern in (_SCENESHIFT_DOUBLE_BRACE_ANSWER_DICT_RE, _SCENESHIFT_ANSWER_DICT_RE):
        matches = list(pattern.finditer(response_text))
        for match in reversed(matches):
            parsed = _coerce_choice_letter(match.group("value"))
            if parsed is not None:
                return parsed

    parsed = _coerce_choice_letter(response_text)
    if parsed is not None:
        return parsed

    print(f"Warning: Could not parse choice answer from response: {str(response_text)[:200]}...")
    return None


def higher_order_process_results(doc, results):
    doc["prediction"] = results[0]
    question_type = doc["question_type"]

    if question_type == "object_size":
        pred = parse_numeric_triplet_answer_from_response(doc.get("prediction"))
        gt = _coerce_float_triplet(doc.get("ground_truth_value") or doc.get("ground_truth"))
    elif question_type == "object_size_comparison_transitivity":
        pred = parse_choice_answer_from_response(doc.get("prediction"))
        gt = _coerce_choice_letter(doc.get("ground_truth_value") or doc.get("ground_truth"))
    else:
        pred = parse_numeric_answer_from_response(doc.get("prediction"))
        gt = parse_numeric_answer_from_response(doc.get("ground_truth_value") or doc.get("ground_truth"))

    doc["prediction_parse"] = pred
    doc["ground_truth_parse"] = gt
    doc["prediction_parse_success"] = pred is not None
    return {"sceneshift_score": doc}


def _higher_order_parse_success_rate(df):
    if df.empty or "prediction_parse_success" not in df.columns:
        return None
    return float(df["prediction_parse_success"].fillna(False).mean())


def _higher_order_scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _higher_order_eval_value(row):
    question_type = row.get("question_type")
    if question_type == "object_size":
        dim = str(row.get("constraint_dimension") or "")
        dim_idx = _SIZE_DIM_TO_INDEX.get(dim)
        if dim_idx is None:
            return None
        triplet = _coerce_float_triplet(row.get("prediction_parse"))
        if triplet is None:
            return None
        return triplet[dim_idx]
    return _higher_order_scalar(row.get("prediction_parse"))


def _higher_order_gt_value(row):
    question_type = row.get("question_type")
    if question_type == "object_size":
        dim = str(row.get("constraint_dimension") or "")
        dim_idx = _SIZE_DIM_TO_INDEX.get(dim)
        if dim_idx is None:
            return None
        triplet = _coerce_float_triplet(row.get("ground_truth_parse"))
        if triplet is None:
            return None
        return triplet[dim_idx]
    return _higher_order_scalar(row.get("ground_truth_parse"))


def _higher_order_row_mra(row):
    pred = _higher_order_eval_value(row)
    gt = _higher_order_gt_value(row)
    if pred is None or gt is None:
        return None
    try:
        return float(mean_relative_accuracy(pred, gt, start=0.5, end=0.95, interval=0.05))
    except (TypeError, ValueError):
        return None


def _triangle_violation(values):
    if len(values) != 3 or any(value is None for value in values):
        return None
    d_ab, d_bc, d_ac = values
    return max(
        0.0,
        d_ab - d_bc - d_ac,
        d_bc - d_ab - d_ac,
        d_ac - d_ab - d_bc,
    )


def _camera_triangle_violation(values):
    if len(values) != 3 or any(value is None for value in values):
        return None
    camera_a, camera_b, pair_ab = values
    return max(
        0.0,
        abs(camera_a - camera_b) - pair_ab,
        pair_ab - camera_a - camera_b,
    )


def _higher_order_violation(group):
    analysis = str(group.iloc[0].get("higher_order_analysis") or "")
    ordered = group.sort_values("constraint_row_index")
    values = [_higher_order_eval_value(row) for _, row in ordered.iterrows()]
    if analysis == "distance_triangle":
        return _triangle_violation(values)
    if analysis == "camera_distance_triangle":
        return _camera_triangle_violation(values)
    return None


def _higher_order_scale(group):
    values = [_higher_order_eval_value(row) for _, row in group.sort_values("constraint_row_index").iterrows()]
    if any(value is None for value in values):
        return None
    return max(max(abs(value) for value in values), 1e-9)


def _evaluate_higher_order_group(group, weak=False):
    violation = _higher_order_violation(group)
    if violation is None:
        return None
    eps_abs = HIGHER_ORDER_TRIANGLE_WEAK_EPS_ABS if weak else HIGHER_ORDER_TRIANGLE_EPS_ABS
    if str(group.iloc[0].get("higher_order_analysis") or "") in {"distance_triangle", "camera_distance_triangle"}:
        return bool(violation <= eps_abs)
    return None


def _evaluate_ordered_triplet(group, value_fn, weak=False):
    ordered = group.sort_values("constraint_row_index")
    values = [value_fn(row) for _, row in ordered.iterrows()]
    if any(value is None for value in values):
        return None
    if weak:
        return bool(values[0] >= values[1] >= values[2])
    return bool(values[0] > values[1] > values[2])


def _evaluate_size_group(group, weak=False):
    dim = str(group.iloc[0].get("constraint_dimension") or "")
    dim_idx = _SIZE_DIM_TO_INDEX.get(dim)
    if dim_idx is None:
        return None

    def value_fn(row):
        triplet = _coerce_float_triplet(row.get("prediction_parse"))
        if triplet is None:
            return None
        return triplet[dim_idx]

    return _evaluate_ordered_triplet(group, value_fn, weak=weak)


def _evaluate_size_transitivity_mc_group(group):
    ordered = group.sort_values("constraint_row_index")
    preds = [str(row.get("prediction_parse") or "").strip().upper() for _, row in ordered.iterrows()]
    gts = [str(row.get("ground_truth_parse") or "").strip().upper() for _, row in ordered.iterrows()]
    if len(preds) != 3:
        return None
    if any(pred not in {"A", "B"} for pred in preds):
        return None
    if any(gt not in {"A", "B"} for gt in gts):
        return None
    return bool(all(pred == gt for pred, gt in zip(preds, gts)))


def _evaluate_depth_group(group, weak=False):
    return _evaluate_ordered_triplet(group, lambda row: _higher_order_scalar(row.get("prediction_parse")), weak=weak)


def _evaluate_triangle_group(group, eps_abs=HIGHER_ORDER_TRIANGLE_EPS_ABS):
    ordered = group.sort_values("constraint_row_index")
    values = [_higher_order_scalar(row.get("prediction_parse")) for _, row in ordered.iterrows()]
    if len(values) != 3 or any(value is None for value in values):
        return None
    d_ab, d_bc, d_ac = values
    return bool(
        d_ab <= d_bc + d_ac + eps_abs
        and d_bc <= d_ab + d_ac + eps_abs
        and d_ac <= d_ab + d_bc + eps_abs
    )


def _emit_higher_order_analysis(df, out, analysis, tag):
    sub = df[df["higher_order_analysis"] == analysis]
    out[f"{tag}_n"] = int(len(sub))
    if sub.empty:
        return

    total_groups = int(sub["constraint_id"].nunique())
    out[f"{tag}_total_groups_n"] = total_groups
    parse_success_rate = _higher_order_parse_success_rate(sub)
    if parse_success_rate is not None:
        out[f"{tag}_parse_success_rate"] = round(parse_success_rate, 4)

    complete = 0
    passed = 0
    weak_passed = 0
    high_mra_complete = 0
    high_mra_failed = 0
    high_mra_weak_failed = 0
    all_high_mra_complete = 0
    all_high_mra_failed = 0
    all_high_mra_weak_failed = 0
    group_mras = []
    raw_violations = []
    excess_violations = []
    normalized_excess_violations = []
    for _, group in sub.groupby("constraint_id"):
        if len(group) != 3 or not group["prediction_parse_success"].fillna(False).all():
            continue
        complete += 1
        if analysis in {"distance_triangle", "camera_distance_triangle"}:
            strict_result = _evaluate_higher_order_group(group, weak=False)
            weak_result = _evaluate_higher_order_group(group, weak=True)
        elif analysis == "size_transitivity":
            if str(group.iloc[0].get("question_type") or "") == "object_size_comparison_transitivity":
                strict_result = _evaluate_size_transitivity_mc_group(group)
                weak_result = strict_result
            else:
                strict_result = _evaluate_size_group(group, weak=False)
                weak_result = _evaluate_size_group(group, weak=True)
        elif analysis == "depth_transitivity":
            strict_result = _evaluate_depth_group(group, weak=False)
            weak_result = _evaluate_depth_group(group, weak=True)
        elif analysis == "distance_triangle":
            strict_result = _evaluate_triangle_group(group, eps_abs=0.0)
            weak_result = _evaluate_triangle_group(group, eps_abs=HIGHER_ORDER_TRIANGLE_EPS_ABS)
        else:
            continue
        passed += int(bool(strict_result))
        weak_passed += int(bool(weak_result))
        violation = _higher_order_violation(group)
        scale = _higher_order_scale(group)
        if violation is not None and scale is not None:
            excess_violation = max(0.0, violation - HIGHER_ORDER_TRIANGLE_EPS_ABS)
            raw_violations.append(float(violation))
            excess_violations.append(float(excess_violation))
            normalized_excess_violations.append(float(excess_violation / scale))
        row_mras = [_higher_order_row_mra(row) for _, row in group.iterrows()]
        if not any(value is None for value in row_mras):
            group_mra = float(np.mean(row_mras))
            group_mras.append(group_mra)
            if group_mra >= HIGHER_ORDER_HIGH_MRA_THRESHOLD:
                high_mra_complete += 1
                high_mra_failed += int(not bool(strict_result))
                high_mra_weak_failed += int(not bool(weak_result))
            if min(row_mras) >= HIGHER_ORDER_HIGH_MRA_THRESHOLD:
                all_high_mra_complete += 1
                all_high_mra_failed += int(not bool(strict_result))
                all_high_mra_weak_failed += int(not bool(weak_result))

    out[f"{tag}_complete_groups_n"] = complete
    out[f"{tag}_complete_group_rate"] = round(complete / total_groups, 4) if total_groups else None
    out[f"{tag}_pass_rate"] = round(passed / complete, 4) if complete else None
    out[f"{tag}_weak_pass_rate"] = round(weak_passed / complete, 4) if complete else None
    if raw_violations:
        out[f"{tag}_mean_raw_violation_m"] = round(float(np.mean(raw_violations)), 4)
        out[f"{tag}_p95_raw_violation_m"] = round(float(np.percentile(raw_violations, 95)), 4)
        out[f"{tag}_mean_excess_violation_m"] = round(float(np.mean(excess_violations)), 4)
        out[f"{tag}_p95_excess_violation_m"] = round(float(np.percentile(excess_violations, 95)), 4)
        out[f"{tag}_mean_normalized_excess_violation"] = round(float(np.mean(normalized_excess_violations)), 4)
        failed_excess = [value for value in excess_violations if value > 0]
        out[f"{tag}_failed_groups_n"] = len(failed_excess)
        out[f"{tag}_failure_rate"] = round(len(failed_excess) / complete, 4) if complete else None
        out[f"{tag}_failed_mean_excess_violation_m"] = round(float(np.mean(failed_excess)), 4) if failed_excess else 0.0
    if group_mras:
        out[f"{tag}_mean_group_MRA"] = round(float(np.mean(group_mras)) * 100.0, 2)
        out[f"{tag}_high_mra_threshold"] = HIGHER_ORDER_HIGH_MRA_THRESHOLD
        out[f"{tag}_high_mra_groups_n"] = high_mra_complete
        out[f"{tag}_high_mra_failure_rate"] = round(high_mra_failed / high_mra_complete, 4) if high_mra_complete else None
        out[f"{tag}_high_mra_weak_failure_rate"] = (
            round(high_mra_weak_failed / high_mra_complete, 4) if high_mra_complete else None
        )
        out[f"{tag}_all_high_mra_groups_n"] = all_high_mra_complete
        out[f"{tag}_all_high_mra_failure_rate"] = (
            round(all_high_mra_failed / all_high_mra_complete, 4) if all_high_mra_complete else None
        )
        out[f"{tag}_all_high_mra_weak_failure_rate"] = (
            round(all_high_mra_weak_failed / all_high_mra_complete, 4) if all_high_mra_complete else None
        )


def higher_order_aggregate_results(results):
    """Aggregate three-row higher-order realizability groups."""
    out: OrderedDict = OrderedDict()
    df = pd.DataFrame(results)
    if df.empty:
        return out

    out["overall_n"] = int(len(df))
    out["overall_total_groups_n"] = int(df["constraint_id"].nunique()) if "constraint_id" in df.columns else 0
    parse_success_rate = _higher_order_parse_success_rate(df)
    if parse_success_rate is not None:
        out["overall_parse_success_rate"] = round(parse_success_rate, 4)

    present_analyses = set(df["higher_order_analysis"].dropna().astype(str)) if "higher_order_analysis" in df.columns else set()
    analyses = [
        analysis
        for analysis in ("distance_triangle", "camera_distance_triangle", "size_transitivity", "depth_transitivity")
        if analysis in present_analyses
    ]
    for analysis in analyses:
        _emit_higher_order_analysis(df, out, analysis, analysis)

    out["distance_triangle_eps_abs"] = HIGHER_ORDER_TRIANGLE_EPS_ABS
    out["distance_triangle_weak_eps_abs"] = HIGHER_ORDER_TRIANGLE_WEAK_EPS_ABS
    return out


def _weighted_average(values, weights):
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def aggregate_sceneshift_group_metrics(metrics, sizes, weight_by_size=True):
    """Merge sceneshift summary dicts across subtasks for group-level reporting.

    The generic group aggregator only supports scalar metrics. SceneShift task
    aggregations return structured summaries, so we merge them field-by-field
    using the relevant denominators already embedded in each summary.
    """
    merged: OrderedDict = OrderedDict()
    valid_metrics = [metric for metric in metrics if isinstance(metric, dict)]
    if not valid_metrics:
        return merged

    all_keys = []
    for metric in valid_metrics:
        for key in metric:
            if key not in all_keys:
                all_keys.append(key)

    for key in all_keys:
        if key.endswith(("_n", "_n_groups", "_total_groups_n")):
            merged[key] = int(sum(int(metric.get(key, 0) or 0) for metric in valid_metrics))
            continue

        values = []
        weights = []
        for metric, size in zip(metrics, sizes):
            if not isinstance(metric, dict) or key not in metric:
                continue
            value = metric[key]
            if value is None:
                continue

            if key.endswith("_mean_CV"):
                weight_key = key[: -len("_mean_CV")] + "_n_groups"
                weight = (metric.get(weight_key, 0) or 0) if weight_by_size else 1
            elif key.endswith("_parse_success_rate"):
                weight_key = key[: -len("_parse_success_rate")] + "_n"
                weight = (metric.get(weight_key, 0) or 0) if weight_by_size else 1
            elif key.endswith("_complete_group_rate"):
                weight_key = key[: -len("_complete_group_rate")] + "_total_groups_n"
                weight = (metric.get(weight_key, 0) or 0) if weight_by_size else 1
            elif key.endswith(("_MRA", "_CAPE")):
                suffix = "_MRA" if key.endswith("_MRA") else "_CAPE"
                weight_key = key[: -len(suffix)] + "_n"
                weight = (metric.get(weight_key, 0) or 0) if weight_by_size else 1
            else:
                weight = size if weight_by_size else 1

            if weight <= 0:
                continue

            values.append(float(value))
            weights.append(float(weight))

        averaged = _weighted_average(values, weights)
        if averaged is None:
            continue
        if key.endswith("_MRA"):
            merged[key] = round(float(averaged), 2)
        else:
            merged[key] = round(float(averaged), 4)

    return merged
