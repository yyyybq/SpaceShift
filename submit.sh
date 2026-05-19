#!/usr/bin/env bash
# Submit wrapper for curated AI2-THOR trajectory demos.
#
# Usage:
#   bash submit.sh
#   bash submit.sh --scene FloorPlan203
#
# Input spec:
#   Forward all CLI flags to scripts/generate_trajectory_demos.sh.
#
# Output spec:
#   Rebuilds the requested trajectory demo output directory.

set -euo pipefail

bash scripts/generate_trajectory_demos.sh "$@"
