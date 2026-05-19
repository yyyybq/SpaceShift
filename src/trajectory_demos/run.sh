#!/usr/bin/env bash
# Thin shell wrapper for `trajectory_demos.run`.
#
# Usage:
#   bash src/trajectory_demos/run.sh
#   bash src/trajectory_demos/run.sh --scene FloorPlan203 --trajectory all
#
# Input spec:
#   Forward all CLI flags directly to `python -m trajectory_demos.run`.
#
# Output spec:
#   Writes the curated AI2-THOR trajectory demo set to the requested output directory.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -u -m trajectory_demos.run "$@"
