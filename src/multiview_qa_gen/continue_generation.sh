#!/bin/bash
# Continue dataset generation from spherical onwards

set -e

OUTPUT_DIR="./final_dataset"
SCENES="FloorPlan1 FloorPlan2 FloorPlan201 FloorPlan301 FloorPlan401"
GPU=0
RADIUS=1.5

source /data/baiqiao/miniconda3/bin/activate ai2thor
cd "$(dirname "$0")"

LOG_FILE="$OUTPUT_DIR/generation_log_continue_$(date +%Y%m%d_%H%M%S).txt"
echo "Log file: $LOG_FILE"

run_pipeline() {
    local pattern=$1
    local sub_pattern=$2
    local num_cameras=$3
    local experiment_name=$4
    
    echo ""
    echo "============================================================"
    echo "Generating: $experiment_name"
    echo "  Pattern: $pattern"
    echo "  Sub-pattern: $sub_pattern"
    echo "  Num cameras: $num_cameras"
    echo "  Scenes: $SCENES"
    echo "============================================================"
    
    if [ "$sub_pattern" == "none" ]; then
        python run_pipeline.py \
            --scenes $SCENES \
            --output_dir "$OUTPUT_DIR" \
            --experiment_name "$experiment_name" \
            --move_pattern "$pattern" \
            --radius $RADIUS \
            --num_cameras $num_cameras \
            --max_objects 9999 \
            2>&1 | tee -a "$LOG_FILE"
    else
        python run_pipeline.py \
            --scenes $SCENES \
            --output_dir "$OUTPUT_DIR" \
            --experiment_name "$experiment_name" \
            --move_pattern "$pattern" \
            --sub_pattern "$sub_pattern" \
            --radius $RADIUS \
            --num_cameras $num_cameras \
            --max_objects 9999 \
            2>&1 | tee -a "$LOG_FILE"
    fi
}

echo "Starting continued generation at $(date)"
echo ""

# 2. Spherical (12 cameras)
run_pipeline "spherical" "none" 12 "spherical"

# 3. Rotation (12 cameras)
run_pipeline "rotation" "none" 12 "rotation"

# 4. Linear Approach (5 cameras)
run_pipeline "linear" "approach" 5 "linear_approach"

# 5. Linear Pass By (5 cameras)
run_pipeline "linear" "pass_by" 5 "linear_pass_by"

echo ""
echo "============================================================"
echo "All generation complete!"
echo "============================================================"
echo "Output directories:"
ls -la "$OUTPUT_DIR"
echo ""
echo "Completed at $(date)"
