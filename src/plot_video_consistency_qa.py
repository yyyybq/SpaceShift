"""Publication-style QA distribution plots from `qa.json`.

Usage:
  PYTHONPATH=src python src/plot_video_consistency_qa.py --qa_path ./video_consistency_output/qa.json --out_dir ./video_consistency_output/figures
  PYTHONPATH=src python src/plot_video_consistency_qa.py --qa_path ./qa.json --out_dir ./figures --engine thor

Input spec:
  --qa_path   Benchmark `qa.json` (list of clips, each with `questions[]`).
  --engine    If set (e.g. thor, interiorgs), keep only clips with this `engine` field (case-insensitive).

Output spec:
  `qa_distribution.png` / `.pdf`, or `qa_distribution_{engine}.png` / `.pdf` when `--engine` is set.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUESTION_TYPE_DISPLAY = {
    "object_dimensions": "Object size (L / W / H)",
    "object_distance_to_camera": "Camera ↔ object distance",
    "object_pair_distance_center": "Object ↔ object distance (rotation)",
}

QUESTION_FAMILY_DISPLAY = {
    "size": "Size",
    "camera_distance": "Camera distance",
    "pair_distance": "Pair distance (rotation)",
}


def _flatten_qa(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list), f"Expected list in {path}"
    rows: list[dict] = []
    for clip in payload:
        motion = clip.get("motion_family", "")
        engine = clip.get("engine", "")
        scene = clip.get("scene_id", "")
        for q in clip.get("questions", []):
            qtype = q.get("question_type", "unknown")
            al = q.get("anchor_labels") or []
            label = "unknown"
            if isinstance(al, list) and al:
                if qtype == "object_pair_distance_center" and len(al) >= 2:
                    a = str(al[0]).split("|")[0].strip()
                    b = str(al[1]).split("|")[0].strip()
                    label = "–".join(sorted([a, b]))
                else:
                    label = str(al[0]).split("|")[0].strip() or "unknown"
            rows.append({
                "question_type": qtype,
                "question_family": q.get("question_family", "unknown"),
                "dimension": q.get("dimension"),
                "motion_family": motion,
                "engine": engine or "unknown",
                "scene_id": scene,
                "object_type": label,
            })
    assert rows, f"No QA rows in {path}"
    return rows


def _paper_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _pie(ax, counts: Counter, title: str):
    labels = list(counts.keys())
    sizes = [counts[k] for k in labels]
    colors = plt.cm.tab20.colors[: len(labels)]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.0f}%\n({int(round(pct / 100.0 * sum(sizes)))})",
        colors=colors,
        pctdistance=0.75,
        labeldistance=1.05,
    )
    for t in autotexts:
        t.set_fontsize(7)
    ax.set_title(title)


def _display_counter(counter: Counter, mapping: dict[str, str]) -> Counter:
    out: Counter = Counter()
    for key, value in counter.items():
        out[mapping.get(key, str(key))] += value
    return out


def _barh(ax, counts: Counter, title: str, max_items: int = 12):
    items = counts.most_common(max_items)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    y = range(len(labels))
    ax.barh(y, vals, color="#2c7fb8")
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v + 0.02 * max(vals, default=1), i, str(v), va="center", fontsize=7)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", default="./video_consistency_output/qa.json")
    parser.add_argument("--out_dir", default="./video_consistency_output/figures")
    parser.add_argument("--engine", default=None, help="e.g. thor or interiorgs; filter clips by engine")
    args = parser.parse_args()
    qa_path = Path(args.qa_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _flatten_qa(qa_path)
    if args.engine is not None:
        key = args.engine.strip().lower()
        rows = [r for r in rows if (r["engine"] or "").strip().lower() == key]
        assert rows, f"No QA rows left after --engine {args.engine!r} filter in {qa_path}"
    n_qa = len(rows)
    _paper_rc()
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    engine_tag = f" — {args.engine} only" if args.engine else ""
    fig.suptitle(f"QA distribution{engine_tag} (N = {n_qa} question instances)", fontsize=12, y=1.02)

    qt = _display_counter(Counter(r["question_type"] for r in rows), QUESTION_TYPE_DISPLAY)
    qf = _display_counter(Counter(r["question_family"] for r in rows), QUESTION_FAMILY_DISPLAY)
    dim_rows = [r for r in rows if r["dimension"]]
    _pie(axes[0, 0], qt, "(a) Question type")
    _pie(axes[0, 1], qf, "(b) Question family")
    if dim_rows:
        dim = Counter(str(r["dimension"]) for r in dim_rows)
        _pie(axes[0, 2], dim, f"(c) Dimension (n={len(dim_rows)})")
    else:
        axes[0, 2].axis("off")
        axes[0, 2].text(0.5, 0.5, "No dimension fields", ha="center", va="center", fontsize=11)
        axes[0, 2].set_title("(c) Dimension")

    mf = Counter(r["motion_family"] for r in rows)
    eng = Counter(r["engine"] for r in rows)
    ot = Counter(r["object_type"] for r in rows)

    _barh(axes[1, 0], mf, "(d) Motion family (per QA instance)")
    _barh(axes[1, 1], eng, "(e) Engine")
    _barh(axes[1, 2], ot, "(f) Object or sorted pair (rotation: A–B)", max_items=15)

    plt.tight_layout()
    stem = f"qa_distribution_{args.engine.strip().lower()}" if args.engine else "qa_distribution"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}\nWrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
