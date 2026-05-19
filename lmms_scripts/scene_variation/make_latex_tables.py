"""
Render the four MRA / consistency pivots into a single LaTeX file with
hierarchical column headers, matching the existing booktabs style used in
the SceneShift paper draft (table.tex).

Usage:
    python scripts/scene_variation/make_latex_tables.py
    python scripts/scene_variation/make_latex_tables.py --tex-out /custom/path.tex

Input:
    /nas2/edwin/lmms-eval/results/combined/<model>.json bundles produced by
    combine_results.py.

Output:
    {tex_out} (default: /nas2/edwin/lmms-eval/table.tex)
        Contains 4 \\begin{table}...\\end{table} environments:
            tab:mra_modality_qtype          MRA% by modality x qtype
            tab:consistency_modality_qtype  consistency by modality x qtype
            tab:mra_variation_pattern       MRA% by variation pattern
            tab:consistency_variation_pattern  consistency by variation pattern
        Top row groups columns by Image / Video (Table 1) or Camera Variation /
        Scene Edit (Table 2). Models are split into Proprietary vs Open Source
        sections with bold marking the best per column within each section.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_COMBINED_DIR = Path("/nas2/edwin/lmms-eval/results/combined")
DEFAULT_TEX_OUT = Path("/nas2/edwin/lmms-eval/table.tex")

JSONL_PATHS = {
    "scene_variation": Path("/nas2/edwin/lmms-eval/data/scene_variation_042226_aligned.jsonl"),
    "image_consistency_thor_eval_v2": Path("/nas2/edwin/lmms-eval/data/image_consistency_thor_eval_v2.jsonl"),
    "video_consistency_thor_eval_v2": Path("/nas2/edwin/lmms-eval/data/video_consistency_thor_eval_v2.jsonl"),
}

MRA_START = 0.5
MRA_END = 0.95
MRA_INTERVAL = 0.05

QTYPE_LABEL = {
    "object_dimensions": "Obj. Size",
    "object_distance_to_camera": "Cam. Dist.",
    "object_pair_distance_center": "Pair Dist.",
}
QTYPE_ORDER = ("object_dimensions", "object_distance_to_camera", "object_pair_distance_center")
MODALITY_ORDER = ("scene_edit", "view_var_image", "view_var_video")
MODALITY_LABEL = {
    "scene_edit": "Scene Edit",
    "view_var_image": "View Var. (Image)",
    "view_var_video": "View Var. (Video)",
}

PATTERN_GROUPS = (
    ("Camera Variation", ("cam_rotation", "passby", "around", "approach", "spherical", "static")),
    ("Scene Edit", ("obj_rotation", "translate", "remove")),
)
PATTERN_LABEL = {
    "cam_rotation": "Cam. Rot.",
    "passby": "Pass-by",
    "around": "Around",
    "approach": "Approach",
    "spherical": "Spherical",
    "static": "Static",
    "obj_rotation": "Obj. Rot.",
    "translate": "Translate",
    "remove": "Remove",
}
GROUP_LABEL_SHORT = {
    "Camera Variation": "Camera Var.",
    "Scene Edit": "Scene Edit",
}

MODEL_DISPLAY = {
    "cambrians_3b": "Cambrian-S-3B",
    "cambrians_7b": "Cambrian-S-7B",
    "internvl3p5_2b": "InternVL3.5-2B",
    "internvl3p5_8b": "InternVL3.5-8B",
    "llava_onevision_0p5b": "LLaVA-OV-0.5B",
    "llava_onevision_7b": "LLaVA-OV-7B",
    "qwen2_5_vl_3b": "Qwen2.5-VL-3B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3vl_4b": "Qwen3-VL-4B",
    "qwen3vl_8b": "Qwen3-VL-8B",
    "gemini_3_1_pro_preview": "Gemini 3.1 Pro",
    "gemini_3_flash_preview": "Gemini 3.0 Flash",
    "gpt5p2": "GPT-5.2",
}

_IMG_V2_GROUP_RE = re.compile(
    r"thor_(?P<motion>\w+?)_object_(?:pair_distance_center|distance_to_camera|dimensions)_"
)


def _to_float(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _vectorized_mra(pred: float, targets: np.ndarray) -> float:
    num_pts = int((MRA_END - MRA_START) / MRA_INTERVAL + 2)
    thresholds = np.linspace(MRA_START, MRA_END, num_pts)
    rel = np.where(
        targets != 0,
        np.abs(pred - targets) / np.abs(targets),
        np.where(pred == 0, 0.0, np.inf),
    )
    return float((rel[:, None] <= (1 - thresholds[None, :])).mean())


def _chance_mra_pct(targets: list[float]) -> float | None:
    if not targets:
        return None
    arr = np.asarray(targets, dtype=float)
    candidates = list(np.unique(arr))
    extra: set[float] = set()
    for t in np.unique(arr):
        for ci in np.arange(MRA_START, MRA_END + 0.01, MRA_INTERVAL):
            if t != 0:
                extra.add(t * ci)
                extra.add(t * (2 - ci))
    candidates = list(np.unique(candidates + list(extra)))
    best = -np.inf
    for pred in candidates:
        m = _vectorized_mra(pred, arr)
        if m > best:
            best = m
    return best * 100.0


def _load_chance_baselines() -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Returns (chance per (modality, qtype), chance per variation_group)."""
    pattern_to_group: dict[str, str] = {}
    for group_label, pats in PATTERN_GROUPS:
        for p in pats:
            pattern_to_group[p] = group_label

    by_modality: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_group: dict[str, list[float]] = defaultdict(list)
    for task, jsonl in JSONL_PATHS.items():
        if not jsonl.exists():
            continue
        with jsonl.open() as f:
            for line in f:
                obj = json.loads(line)
                gt = _to_float(obj.get("ground_truth"))
                qt = obj.get("question_type")
                if gt is None or qt not in QTYPE_ORDER:
                    continue
                by_modality[(_modality(task), qt)].append(gt)
                pat = _variation_pattern(task, obj)
                if pat is not None and pat in pattern_to_group:
                    by_group[pattern_to_group[pat]].append(gt)
    return (
        {k: _chance_mra_pct(v) for k, v in by_modality.items()},
        {k: _chance_mra_pct(v) for k, v in by_group.items()},
    )


def _group_cv(preds: list[float]) -> float | None:
    """Standard Coefficient of Variation: std(preds, ddof=1) / mean(preds).

    Returns None if mean is 0 (matches pandas-style guard).
    """
    arr = np.asarray(preds, dtype=float)
    if len(arr) < 2:
        return None
    mu = float(arr.mean())
    if mu == 0 or math.isnan(mu):
        return None
    return float(arr.std(ddof=1)) / mu


def _modality(task: str) -> str:
    """Map each source task to one of three top-level input buckets.

    scene_variation              -> scene_edit       (camera fixed, scene edited)
    image_consistency_thor_v2    -> view_var_image   (scene fixed, view varies, image input)
    video_consistency_thor_v2    -> view_var_video   (scene fixed, view varies, video input)
    """
    if task == "scene_variation":
        return "scene_edit"
    if task == "image_consistency_thor_eval_v2":
        return "view_var_image"
    if task == "video_consistency_thor_eval_v2":
        return "view_var_video"
    return "view_var_image"


def _variation_pattern(task: str, score: dict) -> str | None:
    """Map task + sample to a variation-pattern bucket.

    scene_variation.edit_type='rotate' is an OBJECT rotation in a static scene
    (scene edit) and is mapped to 'obj_rotation'. *_v2.motion_family='rotation'
    is a CAMERA orbit around a fixed scene (camera variation) and is mapped to
    'cam_rotation'. They MUST stay separate columns.
    """
    if task == "scene_variation":
        et = score.get("edit_type")
        if et == "rotate":
            return "obj_rotation"
        if et == "translate":
            return "translate"
        if et == "remove":
            return "remove"
        return None
    if task == "video_consistency_thor_eval_v2":
        mf = score.get("motion_family")
        if mf == "rotation":
            return "cam_rotation"
        return mf
    if task == "image_consistency_thor_eval_v2":
        gid = score.get("group_id") or ""
        m = _IMG_V2_GROUP_RE.match(gid + "_")
        if not m:
            return None
        mf = m.group("motion")
        if mf == "rotation":
            return "cam_rotation"
        return mf
    return None


def _per_model_mra_cv(bundle: dict, key_fn) -> tuple[dict, dict]:
    """Return (mra_pct_by_key, mean_cv_by_key) where CV = std/mean of predictions per group_id.

    Aggregation: MRA = mean of per-sample MRA across all samples in the bucket.
    CV = mean of per-group CV across all groups in the bucket. Both pool all
    samples/groups whose bucket key matches, regardless of source task.
    """
    mra_buckets: dict = defaultdict(list)
    pred_buckets: dict[object, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in bundle["samples"]:
        score = sample.get("sceneshift_score") or {}
        task = sample.get("_task") or ""
        key = key_fn(task, score)
        if key is None:
            continue
        mra = _to_float(score.get("MRA"))
        if mra is not None:
            mra_buckets[key].append(mra)
        gid = score.get("group_id")
        pred = _to_float(score.get("prediction_parse"))
        if gid and pred is not None:
            pred_buckets[key][gid].append(pred)

    mra_out: dict = {}
    cv_out: dict = {}
    for k, mras in mra_buckets.items():
        mra_out[k] = float(np.mean(mras)) * 100.0 if mras else None
    for k, groups in pred_buckets.items():
        cvs = [cv for cv in (_group_cv(p) for p in groups.values()) if cv is not None]
        cv_out[k] = float(np.mean(cvs)) if cvs else None
    return mra_out, cv_out


def _per_model_group_aggregate(bundle: dict) -> tuple[dict, dict]:
    """Aggregate MRA and CV per top-level variation group (Camera Variation / Scene Edit).

    Pools every sample whose pattern belongs to the group, so the group score
    is sample-weighted (groups with more underlying patterns/samples contribute
    proportionally more).
    """
    pattern_to_group: dict[str, str] = {}
    for group_label, pats in PATTERN_GROUPS:
        for p in pats:
            pattern_to_group[p] = group_label

    mra_buckets: dict[str, list[float]] = defaultdict(list)
    pred_buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in bundle["samples"]:
        score = sample.get("sceneshift_score") or {}
        task = sample.get("_task") or ""
        pat = _variation_pattern(task, score)
        if pat is None:
            continue
        group = pattern_to_group.get(pat)
        if group is None:
            continue
        mra = _to_float(score.get("MRA"))
        if mra is not None:
            mra_buckets[group].append(mra)
        gid = score.get("group_id")
        pred = _to_float(score.get("prediction_parse"))
        if gid and pred is not None:
            pred_buckets[group][gid].append(pred)

    mra_out: dict = {}
    cv_out: dict = {}
    for g, mras in mra_buckets.items():
        mra_out[g] = float(np.mean(mras)) * 100.0 if mras else None
    for g, groups in pred_buckets.items():
        cvs = [cv for cv in (_group_cv(p) for p in groups.values()) if cv is not None]
        cv_out[g] = float(np.mean(cvs)) if cvs else None
    return mra_out, cv_out


def _modality_key(task, score):
    qt = score.get("question_type")
    if qt not in QTYPE_ORDER:
        return None
    return (_modality(task), qt)


def _variation_key(task, score):
    return _variation_pattern(task, score)


def _format_value(value: float | None, *, fmt: str, is_best: bool, suffix: str = "") -> str:
    if value is None:
        return "--"
    s = format(value, fmt) + suffix
    return f"\\textbf{{{s}}}" if is_best else s


def _column_best(
    rows: list[tuple[str, dict, float | None]],
    columns: list,
    *,
    higher_is_better: bool,
) -> dict:
    op = max if higher_is_better else min
    best: dict = {}
    for col_key in columns:
        vals = [scores.get(col_key) for _, scores, _ in rows if scores.get(col_key) is not None]
        if vals:
            best[col_key] = op(vals)
    return best


def _avg_best(
    rows: list[tuple[str, dict, float | None]],
    *,
    higher_is_better: bool,
) -> float | None:
    op = max if higher_is_better else min
    avgs = [r[2] for r in rows if r[2] is not None]
    return op(avgs) if avgs else None


def _row_avg(scores: dict, columns: list) -> float | None:
    vals = [scores.get(c) for c in columns]
    present = [v for v in vals if v is not None]
    return float(np.mean(present)) if present else None


def _format_combined_cell(
    mra: float | None,
    cv: float | None,
    *,
    is_best_mra: bool,
    is_best_cv: bool,
) -> str:
    """Stack CV on top (primary, regular size), MRA below in scriptsize (secondary).

    Bold each if best in column. For chance rows where CV is unavailable, the
    top is `--` and the MRA chance baseline appears below in scriptsize.
    """
    mra_s = "--" if mra is None else f"{mra:.2f}"
    cv_s = "--" if cv is None else f"{cv:.2f}"
    if is_best_mra and mra is not None:
        mra_s = f"\\textbf{{{mra_s}}}"
    if is_best_cv and cv is not None:
        cv_s = f"\\textbf{{{cv_s}}}"
    return f"\\shortstack[c]{{{cv_s} \\\\ {{\\scriptsize {mra_s}}}}}"


def _row_avg_pair(scores_mra: dict, scores_cv: dict, columns: list) -> tuple[float | None, float | None]:
    mras = [scores_mra.get(c) for c in columns]
    cvs = [scores_cv.get(c) for c in columns]
    mras_present = [v for v in mras if v is not None]
    cvs_present = [v for v in cvs if v is not None]
    avg_mra = float(np.mean(mras_present)) if mras_present else None
    avg_cv = float(np.mean(cvs_present)) if cvs_present else None
    return avg_mra, avg_cv


def _rank_models(rows: list[tuple[str, dict, float | None]], *, higher_is_better: bool) -> dict[str, int]:
    """Rank models by their Avg value. Returns {display_name: rank} (1 = best).

    Models with Avg=None are not ranked.
    """
    valid = [(d, a) for d, _, a in rows if a is not None]
    sorted_rows = sorted(valid, key=lambda r: -r[1] if higher_is_better else r[1])
    return {d: i + 1 for i, (d, _) in enumerate(sorted_rows)}


def _emit_rows(
    out: list[str],
    rows: list[tuple[str, dict, float | None]],
    columns: list,
    *,
    fmt: str,
    higher_is_better: bool,
    col_best: dict,
    avg_best: float | None,
    n_full_cols: int,
) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: -r[2] if (higher_is_better and r[2] is not None) else (r[2] if r[2] is not None else float("inf")),
    )
    for display, scores, avg in rows_sorted:
        n_present = sum(1 for c in columns if scores.get(c) is not None)
        partial = n_present < n_full_cols
        avg_suffix = "$^\\dagger$" if partial else ""
        cells = [display]
        cells.append(
            _format_value(
                avg,
                fmt=fmt,
                is_best=(not partial and avg_best is not None and avg is not None and abs(avg - avg_best) < 1e-6),
                suffix=avg_suffix,
            )
        )
        for col_key in columns:
            v = scores.get(col_key)
            best = col_best.get(col_key)
            cells.append(
                _format_value(
                    v,
                    fmt=fmt,
                    is_best=(best is not None and v is not None and abs(v - best) < 1e-6),
                )
            )
        out.append("    " + " & ".join(cells) + " \\\\")


def _build_modality_table(
    rows_data: list[tuple[str, dict]],
    *,
    title_prefix: str,
    label: str,
    fmt: str,
    higher_is_better: bool,
    metric_name: str,
    chance_modality: dict[tuple[str, str], float] | None = None,
) -> str:
    """Render Image (3 qtypes) | Video (3 qtypes) with Avg column. If
    chance_modality is provided, prepend a Baseline row with chance MRA.
    """
    cols = [(mod, qt) for mod in MODALITY_ORDER for qt in QTYPE_ORDER]
    n_data_cols = len(cols)
    total_cols = 1 + 1 + n_data_cols

    rows: list[tuple[str, dict, float | None]] = []
    for model_key, scores in rows_data:
        if model_key not in MODEL_DISPLAY:
            continue
        display = MODEL_DISPLAY[model_key]
        avg = _row_avg(scores, cols)
        rows.append((display, scores, avg))

    col_best = _column_best(rows, cols, higher_is_better=higher_is_better)
    avg_best = _avg_best(rows, higher_is_better=higher_is_better)

    col_spec = "@{}l " + " ".join(["c"] * (total_cols - 1)) + "@{}"
    out: list[str] = []
    out.append("\\begin{table}[H]")
    out.append("    \\centering")
    out.append("    \\resizebox{\\textwidth}{!}{%")
    out.append("    \\setlength\\tabcolsep{5pt}")
    out.append("    \\renewcommand{\\arraystretch}{1.1}")
    out.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    out.append("    \\toprule")
    out.append("    & & \\multicolumn{3}{c}{\\textbf{Image}} & \\multicolumn{3}{c}{\\textbf{Video}} \\\\")
    out.append("    \\cmidrule(r){3-5} \\cmidrule(l){6-8}")

    header_cells = ["", "\\rotatebox[origin=l]{45}{\\small \\textbf{Avg.}}"]
    for _mod, qt in cols:
        header_cells.append(f"\\rotatebox[origin=l]{{45}}{{\\small {QTYPE_LABEL[qt]}}}")
    out.append("    " + " & ".join(header_cells) + " \\\\[-2pt]")
    out.append("    \\midrule")

    if chance_modality is not None:
        chance_scores = {col: chance_modality.get(col) for col in cols}
        chance_avg = _row_avg(chance_scores, cols)
        cells = ["Chance Level (Best Constant)"]
        cells.append(_format_value(chance_avg, fmt=fmt, is_best=False))
        for col in cols:
            cells.append(_format_value(chance_scores.get(col), fmt=fmt, is_best=False))
        out.append("    " + " & ".join(cells) + " \\\\")
        out.append("    \\midrule")

    _emit_rows(
        out, rows, cols,
        fmt=fmt, higher_is_better=higher_is_better,
        col_best=col_best, avg_best=avg_best,
        n_full_cols=n_data_cols,
    )

    out.append("    \\bottomrule")
    out.append("    \\end{tabular}%")
    out.append("    }")
    direction = "higher is better" if higher_is_better else "lower is better"
    dagger_note = " $^\\dagger$ marks rows whose Avg is computed over fewer than the full set of columns." if any(
        any(s.get(c) is None for c in cols) for _m, s in rows_data if _m in MODEL_DISPLAY
    ) else ""
    out.append(f"    \\caption{{\\textbf{{{title_prefix} by Modality \\(\\times\\) Question Type.}} ")
    out.append(f"    {metric_name} ({direction}) per model, grouped by input modality (Image vs.\\ Video) and three numeric question types: object size, distance from camera, and pair-wise center distance.")
    out.append("    Image columns aggregate \\texttt{scene\\_variation} (1{,}021/1{,}522/675 entries per qtype) and \\texttt{image\\_consistency\\_thor\\_eval\\_v2} (1{,}000 entries each).")
    out.append("    Video columns are from \\texttt{video\\_consistency\\_thor\\_eval\\_v2} (1{,}000 entries each).")
    out.append(f"    Bold marks the best per column.{dagger_note}}}")
    out.append(f"    \\label{{tab:{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def _build_split_modality_table(
    rows_data: list[tuple[str, dict]],
    *,
    metric: str,  # "cv" or "mra"
    label: str,
    chance_modality: dict[tuple[str, str], float] | None = None,
) -> str:
    """Render a single-metric Image/Video x QType table (CV-only or MRA-only) with a Rank column.

    Models are sorted by Avg of the metric (CV ascending, MRA descending). Bold
    marks the best per column. The Rank column shows 1..N based on Avg of this
    metric. Chance row is shown only for the MRA table (CV is unsupervised).
    """
    assert metric in ("cv", "mra"), metric
    higher_is_better = metric == "mra"
    fmt = ".2f" if metric == "mra" else ".3f"  # CV needs 3 decimals to differentiate close models (e.g., 0.106 vs 0.110)

    cols = [(mod, qt) for mod in MODALITY_ORDER for qt in QTYPE_ORDER]
    n_data_cols = len(cols)
    total_cols = 1 + 1 + n_data_cols + 1  # model + Avg + cols + Rank

    rows: list[tuple[str, dict, float | None]] = []
    for model_key, scores in rows_data:
        if model_key not in MODEL_DISPLAY:
            continue
        display = MODEL_DISPLAY[model_key]
        avg = _row_avg(scores, cols)
        rows.append((display, scores, avg))

    rank_by_model = _rank_models(rows, higher_is_better=higher_is_better)
    col_best = _column_best(rows, cols, higher_is_better=higher_is_better)
    avg_best = _avg_best(rows, higher_is_better=higher_is_better)

    col_spec = "@{}l " + " ".join(["c"] * (n_data_cols + 1)) + " | c@{}"
    out: list[str] = []
    out.append("\\begin{table}[H]")
    out.append("    \\centering")
    out.append("    \\resizebox{\\textwidth}{!}{%")
    out.append("    \\setlength\\tabcolsep{5pt}")
    out.append("    \\renewcommand{\\arraystretch}{1.1}")
    out.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    out.append("    \\toprule")

    top_cells = ["", ""]
    cmidrules = []
    cur = 3
    for i, mod in enumerate(MODALITY_ORDER):
        n_qt = len(QTYPE_ORDER)
        top_cells.append(f"\\multicolumn{{{n_qt}}}{{c}}{{\\textbf{{{MODALITY_LABEL[mod]}}}}}")
        if i == 0:
            edge = "(r)"
        elif i == len(MODALITY_ORDER) - 1:
            edge = "(l)"
        else:
            edge = "(lr)"
        cmidrules.append(f"\\cmidrule{edge}{{{cur}-{cur + n_qt - 1}}}")
        cur += n_qt
    top_cells.append("")  # rank column header in second row
    out.append("    " + " & ".join(top_cells) + " \\\\")
    out.append("    " + " ".join(cmidrules))

    header_cells = ["", "\\rotatebox[origin=l]{45}{\\small \\textbf{Avg.}}"]
    for _mod, qt in cols:
        header_cells.append(f"\\rotatebox[origin=l]{{45}}{{\\small {QTYPE_LABEL[qt]}}}")
    header_cells.append("\\textbf{Rank}")
    out.append("    " + " & ".join(header_cells) + " \\\\[-2pt]")
    out.append("    \\midrule")

    if metric == "mra" and chance_modality is not None:
        chance_avg = float(np.mean([v for v in (chance_modality.get(c) for c in cols) if v is not None]))
        cells = ["Chance Level (Best Constant)"]
        cells.append(format(chance_avg, fmt))
        for col in cols:
            v = chance_modality.get(col)
            cells.append(format(v, fmt) if v is not None else "--")
        cells.append("--")
        out.append("    " + " & ".join(cells) + " \\\\")
        out.append("    \\midrule")

    rows_sorted = sorted(
        rows,
        key=lambda r: (-r[2] if higher_is_better else r[2]) if r[2] is not None else float("inf"),
    )
    for display, scores, avg in rows_sorted:
        cells = [display]
        is_best_avg = avg_best is not None and avg is not None and abs(avg - avg_best) < 1e-6
        cells.append(_format_value(avg, fmt=fmt, is_best=is_best_avg))
        for col in cols:
            v = scores.get(col)
            best = col_best.get(col)
            cells.append(_format_value(
                v, fmt=fmt,
                is_best=(best is not None and v is not None and abs(v - best) < 1e-6),
            ))
        rank = rank_by_model.get(display)
        cells.append(f"\\textbf{{{rank}}}" if rank == 1 else (str(rank) if rank else "--"))
        out.append("    " + " & ".join(cells) + " \\\\")

    out.append("    \\bottomrule")
    out.append("    \\end{tabular}%")
    out.append("    }")

    if metric == "mra":
        caption = (
            "    \\caption{\\textbf{Accuracy (MRA\\%, higher is better) by Input Type \\(\\times\\) Question Type.} "
            "Each cell is mean MRA over all samples in that bucket. \\textit{Scene Edit}: \\texttt{scene\\_variation} (camera fixed, scene edited). "
            "\\textit{View Var.\\ (Image/Video)}: scene fixed, camera moved. "
            "Models sorted by Avg MRA (best first); the rightmost column gives the rank by Avg MRA. "
            "Bold marks the best per column. Chance is the best constant predictor.}"
        )
    else:
        caption = (
            "    \\caption{\\textbf{Consistency (Mean CV, lower is better) by Input Type \\(\\times\\) Question Type.} "
            "Each cell is the mean over per-\\texttt{group\\_id} CVs ($\\sigma/\\mu$ across the variations within a group). "
            "Models sorted by Avg CV (best first); the rightmost column gives the rank by Avg CV. "
            "Bold marks the best per column. CV has no chance baseline (it is an unsupervised, scale-invariant metric).}"
        )
    out.append(caption)
    out.append(f"    \\label{{tab:{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def _build_combined_modality_table(
    mra_rows_data: list[tuple[str, dict]],
    cv_rows_data: list[tuple[str, dict]],
    *,
    label: str,
    chance_modality: dict[tuple[str, str], float] | None = None,
) -> str:
    """Render Image (3 qtypes) | Video (3 qtypes) with stacked MRA / CV per cell.

    Each cell shows MRA (top, larger) and CV (bottom, smaller). Best MRA per
    column is bolded; best CV per column is bolded in scriptsize. Models are
    sorted by Avg MRA (best first).
    """
    cols = [(mod, qt) for mod in MODALITY_ORDER for qt in QTYPE_ORDER]
    n_data_cols = len(cols)
    total_cols = 1 + 1 + n_data_cols

    mra_by_model = {k: v for k, v in mra_rows_data}
    cv_by_model = {k: v for k, v in cv_rows_data}

    enriched: list[tuple[str, dict, dict, float | None, float | None]] = []
    for model_key in mra_by_model:
        if model_key not in MODEL_DISPLAY:
            continue
        s_mra = mra_by_model[model_key]
        s_cv = cv_by_model.get(model_key, {})
        avg_mra, avg_cv = _row_avg_pair(s_mra, s_cv, cols)
        enriched.append((MODEL_DISPLAY[model_key], s_mra, s_cv, avg_mra, avg_cv))

    col_best_mra: dict = {c: max((s.get(c) for _, s, _, _, _ in enriched if s.get(c) is not None), default=None) for c in cols}
    col_best_cv: dict = {c: min((s.get(c) for _, _, s, _, _ in enriched if s.get(c) is not None), default=None) for c in cols}
    avg_best_mra = max((r[3] for r in enriched if r[3] is not None), default=None)
    avg_best_cv = min((r[4] for r in enriched if r[4] is not None), default=None)

    col_spec = "@{}l " + " ".join(["c"] * (total_cols - 1)) + "@{}"
    out: list[str] = []
    out.append("\\begin{table}[H]")
    out.append("    \\centering")
    out.append("    \\resizebox{\\textwidth}{!}{%")
    out.append("    \\setlength\\tabcolsep{5pt}")
    out.append("    \\renewcommand{\\arraystretch}{1.2}")
    out.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    out.append("    \\toprule")

    top_cells = ["", ""]
    cmidrules = []
    cur = 3
    for i, mod in enumerate(MODALITY_ORDER):
        n_qt = len(QTYPE_ORDER)
        top_cells.append(f"\\multicolumn{{{n_qt}}}{{c}}{{\\textbf{{{MODALITY_LABEL[mod]}}}}}")
        if i == 0:
            edge = "(r)"
        elif i == len(MODALITY_ORDER) - 1:
            edge = "(l)"
        else:
            edge = "(lr)"
        cmidrules.append(f"\\cmidrule{edge}{{{cur}-{cur + n_qt - 1}}}")
        cur += n_qt
    out.append("    " + " & ".join(top_cells) + " \\\\")
    out.append("    " + " ".join(cmidrules))

    header_cells = ["", "\\rotatebox[origin=l]{45}{\\small \\textbf{Avg.}}"]
    for _mod, qt in cols:
        header_cells.append(f"\\rotatebox[origin=l]{{45}}{{\\small {QTYPE_LABEL[qt]}}}")
    out.append("    " + " & ".join(header_cells) + " \\\\[-2pt]")
    out.append("    \\midrule")

    if chance_modality is not None:
        cells = ["Chance Level (Best Constant)"]
        chance_avg = float(np.mean([v for v in (chance_modality.get(c) for c in cols) if v is not None]))
        cells.append(_format_combined_cell(chance_avg, None, is_best_mra=False, is_best_cv=False))
        for col in cols:
            cells.append(_format_combined_cell(chance_modality.get(col), None, is_best_mra=False, is_best_cv=False))
        out.append("    " + " & ".join(cells) + " \\\\")
        out.append("    \\midrule")

    enriched_sorted = sorted(enriched, key=lambda r: -r[3] if r[3] is not None else float("inf"))
    for display, s_mra, s_cv, avg_mra, avg_cv in enriched_sorted:
        cells = [display]
        cells.append(_format_combined_cell(
            avg_mra, avg_cv,
            is_best_mra=(avg_best_mra is not None and avg_mra is not None and abs(avg_mra - avg_best_mra) < 1e-6),
            is_best_cv=(avg_best_cv is not None and avg_cv is not None and abs(avg_cv - avg_best_cv) < 1e-6),
        ))
        for col in cols:
            v_mra = s_mra.get(col)
            v_cv = s_cv.get(col)
            cells.append(_format_combined_cell(
                v_mra, v_cv,
                is_best_mra=(col_best_mra.get(col) is not None and v_mra is not None and abs(v_mra - col_best_mra[col]) < 1e-6),
                is_best_cv=(col_best_cv.get(col) is not None and v_cv is not None and abs(v_cv - col_best_cv[col]) < 1e-6),
            ))
        out.append("    " + " & ".join(cells) + " \\\\")

    out.append("    \\bottomrule")
    out.append("    \\end{tabular}%")
    out.append("    }")
    out.append("    \\caption{\\textbf{Spatial reasoning by Input Type \\(\\times\\) Question Type.} ")
    out.append("    Each cell stacks two metrics: top is \\textbf{Mean CV} ($\\overline{\\sigma_\\text{group}/\\mu_\\text{group}}$, lower is better) measured across the variations within each \\texttt{group\\_id}, bottom in {\\scriptsize\\textit{small text}} is \\textbf{MRA (\\%, higher is better)}.")
    out.append("    \\textbf{High MRA does not imply consistency.} For example, Gemini models lead on MRA but their CV is mid-pack, while smaller open-source models can have lower MRA yet markedly tighter CV.")
    out.append("    \\textit{Scene Edit}: \\texttt{scene\\_variation} (camera fixed, object rotated/translated/removed).")
    out.append("    \\textit{View Var.\\ (Image)}: \\texttt{image\\_consistency\\_thor\\_eval\\_v2} (scene fixed, 10 randomized static camera placements or camera-trajectory frames per \\texttt{group\\_id}).")
    out.append("    \\textit{View Var.\\ (Video)}: \\texttt{video\\_consistency\\_thor\\_eval\\_v2} (scene fixed, camera follows a trajectory clip per \\texttt{group\\_id}).")
    out.append("    Bold CV marks the best column-wise; bold MRA (in {\\scriptsize\\textit{small text}}) marks the highest. Models sorted by Avg MRA.}")
    out.append(f"    \\label{{tab:{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def _build_dual_metric_variation_table(
    group_mra_rows: list[tuple[str, dict]],
    group_cv_rows: list[tuple[str, dict]],
    *,
    label: str,
    chance_group: dict[str, float] | None = None,
) -> str:
    """Render a single table where Mean CV and MRA are placed in adjacent column blocks.

    Layout:
        | Model | CV (Avg | Cam Var | Scene Edit) | MRA (Avg | Cam Var | Scene Edit) | Rank-CV | Rank-MRA |

    Models sorted by Avg MRA (descending). The two Rank columns make the
    accuracy-vs-consistency leaderboard discrepancy visually obvious.
    """
    group_cols = [g for g, _ in PATTERN_GROUPS]
    n_data_cols_per_block = len(group_cols) + 1  # Avg + per-group
    total_cols = 1 + 2 * n_data_cols_per_block + 2  # model + (CV block + MRA block) + 2 ranks

    mra_by_model = {k: v for k, v in group_mra_rows}
    cv_by_model = {k: v for k, v in group_cv_rows}

    enriched: list[tuple[str, dict, dict, float | None, float | None]] = []
    for model_key in mra_by_model:
        if model_key not in MODEL_DISPLAY:
            continue
        s_mra = mra_by_model[model_key]
        s_cv = cv_by_model.get(model_key, {})
        avg_mra, avg_cv = _row_avg_pair(s_mra, s_cv, group_cols)
        enriched.append((MODEL_DISPLAY[model_key], s_mra, s_cv, avg_mra, avg_cv))

    rank_mra = _rank_models([(d, s, a) for d, s, _, a, _ in enriched], higher_is_better=True)
    rank_cv = _rank_models([(d, s, a) for d, _, s, _, a in enriched], higher_is_better=False)

    col_best_mra = {c: max((s.get(c) for _, s, _, _, _ in enriched if s.get(c) is not None), default=None) for c in group_cols}
    col_best_cv = {c: min((s.get(c) for _, _, s, _, _ in enriched if s.get(c) is not None), default=None) for c in group_cols}
    avg_best_mra = max((r[3] for r in enriched if r[3] is not None), default=None)
    avg_best_cv = min((r[4] for r in enriched if r[4] is not None), default=None)

    col_spec = (
        "@{}l |"
        + " " + " ".join(["c"] * n_data_cols_per_block) + " |"
        + " " + " ".join(["c"] * n_data_cols_per_block) + " |"
        + " c c@{}"
    )
    out: list[str] = []
    out.append("\\begin{table}[H]")
    out.append("    \\centering")
    out.append("    \\resizebox{\\textwidth}{!}{%")
    out.append("    \\setlength\\tabcolsep{6pt}")
    out.append("    \\renewcommand{\\arraystretch}{1.2}")
    out.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    out.append("    \\toprule")

    top_cells = ["",
                 f"\\multicolumn{{{n_data_cols_per_block}}}{{c|}}{{\\textbf{{Mean CV}} (lower is better)}}",
                 f"\\multicolumn{{{n_data_cols_per_block}}}{{c|}}{{\\textbf{{MRA}}\\,\\% (higher is better)}}",
                 "\\multicolumn{2}{c}{\\textbf{Rank}}"]
    out.append("    " + " & ".join(top_cells) + " \\\\")
    cv_start = 2
    cv_end = cv_start + n_data_cols_per_block - 1
    mra_start = cv_end + 1
    mra_end = mra_start + n_data_cols_per_block - 1
    rank_start = mra_end + 1
    rank_end = rank_start + 1
    out.append(
        f"    \\cmidrule(lr){{{cv_start}-{cv_end}}} "
        f"\\cmidrule(lr){{{mra_start}-{mra_end}}} "
        f"\\cmidrule(lr){{{rank_start}-{rank_end}}}"
    )

    short = lambda g: GROUP_LABEL_SHORT.get(g, g)
    header_cells = [""]
    header_cells.append("\\textbf{Avg.}")
    for g in group_cols:
        header_cells.append(short(g))
    header_cells.append("\\textbf{Avg.}")
    for g in group_cols:
        header_cells.append(short(g))
    header_cells.append("CV")
    header_cells.append("MRA")
    out.append("    " + " & ".join(header_cells) + " \\\\")
    out.append("    \\midrule")

    if chance_group is not None:
        chance_avg = float(np.mean([v for v in (chance_group.get(g) for g in group_cols) if v is not None]))
        cells = ["Chance Level"]
        cells.append("--")
        for _ in group_cols:
            cells.append("--")
        cells.append(format(chance_avg, ".2f"))
        for g in group_cols:
            v = chance_group.get(g)
            cells.append(format(v, ".2f") if v is not None else "--")
        cells.append("--")
        cells.append("--")
        out.append("    " + " & ".join(cells) + " \\\\")
        out.append("    \\midrule")

    enriched_sorted = sorted(enriched, key=lambda r: -r[3] if r[3] is not None else float("inf"))
    for display, s_mra, s_cv, avg_mra, avg_cv in enriched_sorted:
        cells = [display]
        cells.append(_format_value(
            avg_cv, fmt=".3f",
            is_best=(avg_best_cv is not None and avg_cv is not None and abs(avg_cv - avg_best_cv) < 1e-6),
        ))
        for g in group_cols:
            v = s_cv.get(g)
            best = col_best_cv.get(g)
            cells.append(_format_value(
                v, fmt=".3f",
                is_best=(best is not None and v is not None and abs(v - best) < 1e-6),
            ))
        cells.append(_format_value(
            avg_mra, fmt=".2f",
            is_best=(avg_best_mra is not None and avg_mra is not None and abs(avg_mra - avg_best_mra) < 1e-6),
        ))
        for g in group_cols:
            v = s_mra.get(g)
            best = col_best_mra.get(g)
            cells.append(_format_value(
                v, fmt=".2f",
                is_best=(best is not None and v is not None and abs(v - best) < 1e-6),
            ))
        rcv = rank_cv.get(display)
        rmr = rank_mra.get(display)
        cells.append(f"\\textbf{{{rcv}}}" if rcv == 1 else (str(rcv) if rcv else "--"))
        cells.append(f"\\textbf{{{rmr}}}" if rmr == 1 else (str(rmr) if rmr else "--"))
        out.append("    " + " & ".join(cells) + " \\\\")

    out.append("    \\bottomrule")
    out.append("    \\end{tabular}%")
    out.append("    }")
    out.append("    \\caption{\\textbf{Spatial reasoning by Variation Type.} "
               "Left block: \\textbf{Mean CV} (lower=more consistent) measured per \\texttt{group\\_id} as $\\sigma/\\mu$ of model predictions across the variations within the group, then averaged. "
               "Right block: \\textbf{MRA\\%} (higher=more accurate). "
               "\\textit{Camera Var.}: scene fixed, camera moved (rotation, pass-by, around, approach, spherical, static). "
               "\\textit{Scene Edit}: camera fixed, scene edited (object rotation, translation, removal). "
               "Rank-CV and Rank-MRA are computed from the Avg columns of each block; the two ranks differ for almost every model, demonstrating that consistency and accuracy are distinct axes. "
               "Models sorted by Avg MRA. Bold marks best per column / Rank-1.}")
    out.append(f"    \\label{{tab:{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def _build_combined_variation_table(
    group_mra_rows: list[tuple[str, dict]],
    group_cv_rows: list[tuple[str, dict]],
    *,
    label: str,
    chance_group: dict[str, float] | None = None,
) -> str:
    """One column per top-level variation group (Camera Variation, Scene Edit).

    Each group score pools MRA and CV across all the patterns inside that
    group (camera-rotation + pass-by + around + approach + spherical + static
    for Camera Variation; object-rotation + translate + remove for Scene Edit).
    """
    group_cols = [g for g, _ in PATTERN_GROUPS]
    n_data_cols = len(group_cols)
    total_cols = 1 + 1 + n_data_cols

    mra_by_model = {k: v for k, v in group_mra_rows}
    cv_by_model = {k: v for k, v in group_cv_rows}

    enriched: list[tuple[str, dict, dict, float | None, float | None]] = []
    for model_key in mra_by_model:
        if model_key not in MODEL_DISPLAY:
            continue
        s_mra = mra_by_model[model_key]
        s_cv = cv_by_model.get(model_key, {})
        avg_mra, avg_cv = _row_avg_pair(s_mra, s_cv, group_cols)
        enriched.append((MODEL_DISPLAY[model_key], s_mra, s_cv, avg_mra, avg_cv))

    col_best_mra: dict = {c: max((s.get(c) for _, s, _, _, _ in enriched if s.get(c) is not None), default=None) for c in group_cols}
    col_best_cv: dict = {c: min((s.get(c) for _, _, s, _, _ in enriched if s.get(c) is not None), default=None) for c in group_cols}
    avg_best_mra = max((r[3] for r in enriched if r[3] is not None), default=None)
    avg_best_cv = min((r[4] for r in enriched if r[4] is not None), default=None)

    col_spec = "@{}l " + " ".join(["c"] * (total_cols - 1)) + "@{}"
    out: list[str] = []
    out.append("\\begin{table}[H]")
    out.append("    \\centering")
    out.append("    \\resizebox{0.85\\textwidth}{!}{%")
    out.append("    \\setlength\\tabcolsep{8pt}")
    out.append("    \\renewcommand{\\arraystretch}{1.2}")
    out.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    out.append("    \\toprule")

    header_cells = ["", "\\textbf{Avg.}"]
    for g in group_cols:
        header_cells.append(f"\\textbf{{{GROUP_LABEL_SHORT.get(g, g)}}}")
    out.append("    " + " & ".join(header_cells) + " \\\\")
    out.append("    \\midrule")

    if chance_group is not None:
        cells = ["Chance Level (Best Constant)"]
        chance_avg = float(np.mean([v for v in (chance_group.get(g) for g in group_cols) if v is not None]))
        cells.append(_format_combined_cell(chance_avg, None, is_best_mra=False, is_best_cv=False))
        for g in group_cols:
            cells.append(_format_combined_cell(chance_group.get(g), None, is_best_mra=False, is_best_cv=False))
        out.append("    " + " & ".join(cells) + " \\\\")
        out.append("    \\midrule")

    enriched_sorted = sorted(enriched, key=lambda r: -r[3] if r[3] is not None else float("inf"))
    for display, s_mra, s_cv, avg_mra, avg_cv in enriched_sorted:
        cells = [display]
        cells.append(_format_combined_cell(
            avg_mra, avg_cv,
            is_best_mra=(avg_best_mra is not None and avg_mra is not None and abs(avg_mra - avg_best_mra) < 1e-6),
            is_best_cv=(avg_best_cv is not None and avg_cv is not None and abs(avg_cv - avg_best_cv) < 1e-6),
        ))
        for g in group_cols:
            v_mra = s_mra.get(g)
            v_cv = s_cv.get(g)
            cells.append(_format_combined_cell(
                v_mra, v_cv,
                is_best_mra=(col_best_mra.get(g) is not None and v_mra is not None and abs(v_mra - col_best_mra[g]) < 1e-6),
                is_best_cv=(col_best_cv.get(g) is not None and v_cv is not None and abs(v_cv - col_best_cv[g]) < 1e-6),
            ))
        out.append("    " + " & ".join(cells) + " \\\\")

    out.append("    \\bottomrule")
    out.append("    \\end{tabular}%")
    out.append("    }")
    out.append("    \\caption{\\textbf{Spatial reasoning by Variation Type.} ")
    out.append("    Each cell stacks two metrics: top is \\textbf{Mean CV} ($\\overline{\\sigma_\\text{group}/\\mu_\\text{group}}$, lower is better), bottom in {\\scriptsize\\textit{small text}} is \\textbf{MRA (\\%, higher is better)}.")
    out.append("    \\textit{Camera Variation} pools all sub-patterns where the camera moves around or is placed differently while the scene is fixed (camera rotation, linear pass-by, orbit-around, approach/recede, spherical orbit, randomized static placements).")
    out.append("    \\textit{Scene Edit} pools all sub-patterns where the scene itself is edited with the camera fixed (object rotation, object translation, object removal).")
    out.append("    Both metrics are sample-/group-weighted aggregates across all sub-patterns within the group.")
    out.append("    Bold CV marks the best column-wise; bold MRA (in {\\scriptsize\\textit{small text}}) marks the highest. Models sorted by Avg MRA.}")
    out.append(f"    \\label{{tab:{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def _build_variation_table(
    rows_data: list[tuple[str, dict]],
    *,
    title_prefix: str,
    label: str,
    fmt: str,
    higher_is_better: bool,
    metric_name: str,
    available_patterns: set[str],
    chance_variation: dict[str, float] | None = None,
) -> str:
    pattern_cols: list[str] = []
    grouped: list[tuple[str, list[str]]] = []
    for group_label, pats in PATTERN_GROUPS:
        present = [p for p in pats if p in available_patterns]
        if present:
            grouped.append((group_label, present))
            pattern_cols.extend(present)

    n_data_cols = len(pattern_cols)
    total_cols = 1 + 1 + n_data_cols

    rows: list[tuple[str, dict, float | None]] = []
    for model_key, scores in rows_data:
        if model_key not in MODEL_DISPLAY:
            continue
        display = MODEL_DISPLAY[model_key]
        avg = _row_avg(scores, pattern_cols)
        rows.append((display, scores, avg))

    col_best = _column_best(rows, pattern_cols, higher_is_better=higher_is_better)
    avg_best = _avg_best(rows, higher_is_better=higher_is_better)

    col_spec = "@{}l " + " ".join(["c"] * (total_cols - 1)) + "@{}"
    out: list[str] = []
    out.append("\\begin{table}[H]")
    out.append("    \\centering")
    out.append("    \\resizebox{\\textwidth}{!}{%")
    out.append("    \\setlength\\tabcolsep{5pt}")
    out.append("    \\renewcommand{\\arraystretch}{1.1}")
    out.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    out.append("    \\toprule")

    top_cells = [" & "]
    cmidrules: list[str] = []
    cur = 3
    for i, (group_label, present) in enumerate(grouped):
        n = len(present)
        top_cells.append(f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{group_label}}}}}")
        if len(grouped) == 1:
            edge = "(lr)"
        elif i == 0:
            edge = "(r)"
        elif i == len(grouped) - 1:
            edge = "(l)"
        else:
            edge = "(lr)"
        cmidrules.append(f"\\cmidrule{edge}{{{cur}-{cur + n - 1}}}")
        cur += n
    out.append("    " + " & ".join(top_cells) + " \\\\")
    out.append("    " + " ".join(cmidrules))

    header_cells = ["", "\\rotatebox[origin=l]{45}{\\small \\textbf{Avg.}}"]
    for pat in pattern_cols:
        header_cells.append(f"\\rotatebox[origin=l]{{45}}{{\\small {PATTERN_LABEL[pat]}}}")
    out.append("    " + " & ".join(header_cells) + " \\\\[-2pt]")
    out.append("    \\midrule")

    if chance_variation is not None:
        chance_scores = {p: chance_variation.get(p) for p in pattern_cols}
        chance_avg = _row_avg(chance_scores, pattern_cols)
        cells = ["Chance Level (Best Constant)"]
        cells.append(_format_value(chance_avg, fmt=fmt, is_best=False))
        for p in pattern_cols:
            cells.append(_format_value(chance_scores.get(p), fmt=fmt, is_best=False))
        out.append("    " + " & ".join(cells) + " \\\\")
        out.append("    \\midrule")

    _emit_rows(
        out, rows, pattern_cols,
        fmt=fmt, higher_is_better=higher_is_better,
        col_best=col_best, avg_best=avg_best,
        n_full_cols=n_data_cols,
    )

    out.append("    \\bottomrule")
    out.append("    \\end{tabular}%")
    out.append("    }")
    direction = "higher is better" if higher_is_better else "lower is better"
    dagger_note = " $^\\dagger$ marks rows whose Avg is computed over fewer than the full set of columns." if any(
        any(s.get(c) is None for c in pattern_cols) for _m, s in rows_data if _m in MODEL_DISPLAY
    ) else ""
    out.append(f"    \\caption{{\\textbf{{{title_prefix} by Variation Pattern.}} ")
    out.append(f"    {metric_name} ({direction}) per model, grouped by the kind of variation that holds the ground truth constant within each \\texttt{{group\\_id}}.")
    out.append("    \\textit{Camera Variation} columns describe how the camera moves while the scene stays fixed (rotation, linear pass-by, orbit around an object, approach/recede, spherical orbit, static views).")
    out.append("    \\textit{Scene Edit} columns are AI2-Thor scene edits that modify the underlying scene (object translation or removal); they only exist in \\texttt{scene\\_variation}.")
    out.append("    The Rotation column merges \\texttt{scene\\_variation.edit\\_type=rotate} (object rotation) with \\texttt{*\\_v2.motion\\_family=rotation} (camera rotation).")
    out.append(f"    Bold marks the best per column.{dagger_note}}}")
    out.append(f"    \\label{{tab:{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-dir", default=str(DEFAULT_COMBINED_DIR))
    parser.add_argument("--tex-out", default=str(DEFAULT_TEX_OUT))
    args = parser.parse_args()

    combined_dir = Path(args.combined_dir)
    tex_out = Path(args.tex_out)

    bundles = []
    for jp in sorted(combined_dir.glob("*.json")):
        with jp.open() as f:
            bundles.append(json.load(f))
    assert bundles, f"No bundles found in {combined_dir}"

    modality_mra_rows: list = []
    modality_cv_rows: list = []
    group_mra_rows: list = []
    group_cv_rows: list = []

    for bundle in bundles:
        m_mra, m_cv = _per_model_mra_cv(bundle, _modality_key)
        modality_mra_rows.append((bundle["model"], m_mra))
        modality_cv_rows.append((bundle["model"], m_cv))
        g_mra, g_cv = _per_model_group_aggregate(bundle)
        group_mra_rows.append((bundle["model"], g_mra))
        group_cv_rows.append((bundle["model"], g_cv))

    print("Computing chance MRA baselines (best constant predictor) ...")
    chance_modality, chance_group = _load_chance_baselines()

    table_3a_cv = _build_split_modality_table(
        modality_cv_rows,
        metric="cv",
        label="modality_qtype_cv",
        chance_modality=chance_modality,
    )
    table_3b_mra = _build_split_modality_table(
        modality_mra_rows,
        metric="mra",
        label="modality_qtype_mra",
        chance_modality=chance_modality,
    )
    table_4 = _build_dual_metric_variation_table(
        group_mra_rows, group_cv_rows,
        label="variation_type_dual",
        chance_group=chance_group,
    )

    preamble = (
        "% This file is auto-generated by scripts/scene_variation/make_latex_tables.py.\n"
        "% Tables 3a/3b: split-metric Input Type x Question Type tables.\n"
        "%   Tab 3a: Mean CV only (sorted by Avg CV).\n"
        "%   Tab 3b: MRA% only (sorted by Avg MRA).\n"
        "% Table 4: dual-metric Variation Type table (CV block | MRA block | Rank-CV | Rank-MRA).\n"
        "% Required packages in your main document:\n"
        "%   \\usepackage{booktabs}\n"
        "%   \\usepackage{graphicx}  % for \\rotatebox and \\resizebox\n"
        "%   \\usepackage{float}     % for [H] table placement\n"
    )
    blocks = [preamble.rstrip(), "", table_3a_cv, "", table_3b_mra, "", table_4]
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.write_text("\n".join(blocks) + "\n")
    print(f"Wrote {tex_out} ({len(blocks)} blocks)")


if __name__ == "__main__":
    main()
