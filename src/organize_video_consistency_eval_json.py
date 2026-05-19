"""CLI wrapper: nested metrics JSON from a saved lmms-eval ``*_results.json``.

The implementation lives in lmms-eval
``lmms_eval.tasks.sceneshift.video_consistency_production_utils``.
Eval runs also write this automatically next to each ``*_results.json``
(``*_metrics_video_consistency.json``) when the task is ``video_consistency_production``.

Usage:
  PYTHONPATH=src:/nas2/edwin/lmms-eval python src/organize_video_consistency_eval_json.py \\
    --results /path/to/TIMESTAMP_results.json \\
    --out /path/to/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        from lmms_eval.tasks.sceneshift.video_consistency_production_utils import build_organized_video_consistency_metrics
    except ImportError:
        sys.stderr.write("Install lmms-eval or set PYTHONPATH to its repo root.\n")
        return 1

    payload = json.loads(args.results.expanduser().resolve().read_text(encoding="utf-8"))
    organized = build_organized_video_consistency_metrics(payload)
    if organized is None:
        raise ValueError("No video_consistency_production results in JSON")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(organized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
