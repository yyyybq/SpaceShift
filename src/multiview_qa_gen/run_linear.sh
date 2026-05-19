#!/bin/bash
# Run linear patterns only

set -e

OUTPUT_DIR="./final_dataset"
SCENES="FloorPlan1 FloorPlan2 FloorPlan201 FloorPlan301 FloorPlan401"
GPU=0
RADIUS=1.5

source /data/baiqiao/miniconda3/bin/activate ai2thor
cd "$(dirname "$0")"

LOG_FILE="$OUTPUT_DIR/generation_log_linear_$(date +%Y%m%d_%H%M%S).txt"
echo "Log file: $LOG_FILE"

echo "Starting linear patterns at $(date)"
echo ""

# Linear Approach (5 cameras)
echo ""
echo "============================================================"
echo "Generating: linear_approach"
echo "  Pattern: linear"
echo "  Sub-pattern: approach"
echo "  Num cameras: 5"
echo "  Scenes: $SCENES"
echo "============================================================"
python run_pipeline.py \
    --scenes $SCENES \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "linear_approach" \
    --move_pattern "linear" \
    --linear_sub_pattern "approach" \
    --radius $RADIUS \
    --num_cameras 5 \
    --max_objects 9999 \
    2>&1 | tee -a "$LOG_FILE"

# Linear Pass By (5 cameras)
echo ""
echo "============================================================"
echo "Generating: linear_pass_by"
echo "  Pattern: linear"
echo "  Sub-pattern: pass_by"
echo "  Num cameras: 5"
echo "  Scenes: $SCENES"
echo "============================================================"
python run_pipeline.py \
    --scenes $SCENES \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "linear_pass_by" \
    --move_pattern "linear" \
    --linear_sub_pattern "pass_by" \
    --radius $RADIUS \
    --num_cameras 5 \
    --max_objects 9999 \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "============================================================"
echo "Linear patterns complete!"
echo "============================================================"
echo "Output directories:"
ls -la "$OUTPUT_DIR"
echo ""
echo "Completed at $(date)"
