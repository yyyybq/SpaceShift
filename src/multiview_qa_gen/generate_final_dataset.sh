#!/bin/bash
# =============================================================================
# Final Dataset Generation Script
# =============================================================================
# This script generates the complete multiview QA dataset across:
# - 5 scenes: FloorPlan1, FloorPlan101, FloorPlan201, FloorPlan301, FloorPlan401
# - 5 move patterns: around, spherical, rotation, linear_approach, linear_pass_by
# - No object limit (all valid objects in scene)
# - num_cameras: linear=5, others=12
# =============================================================================

set -e  # Exit on error

# Configuration
OUTPUT_DIR="./final_dataset"
# Note: FloorPlan101 doesn't exist in AI2THOR. The valid floor plans are:
# - FloorPlan1-30: Kitchens
# - FloorPlan201-230: Living rooms
# - FloorPlan301-330: Bedrooms  
# - FloorPlan401-430: Bathrooms
# Using FloorPlan2 instead of FloorPlan101
SCENES="FloorPlan1 FloorPlan2 FloorPlan201 FloorPlan301 FloorPlan401"
GPU=0
RADIUS=1.5

# Activate conda environment
echo "============================================================"
echo "Activating conda environment..."
echo "============================================================"
source /data/baiqiao/miniconda3/bin/activate ai2thor

# Change to script directory
cd "$(dirname "$0")"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/generation_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Log file: $LOG_FILE"

# Function to run pipeline
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
            --gpu $GPU \
            2>&1 | tee -a "$LOG_FILE"
    else
        python run_pipeline.py \
            --scenes $SCENES \
            --output_dir "$OUTPUT_DIR" \
            --experiment_name "$experiment_name" \
            --move_pattern "$pattern" \
            --linear_sub_pattern "$sub_pattern" \
            --radius $RADIUS \
            --num_cameras $num_cameras \
            --max_objects 9999 \
            --gpu $GPU \
            2>&1 | tee -a "$LOG_FILE"
    fi
}

# Start time
START_TIME=$(date +%s)
echo "Starting dataset generation at $(date)" | tee "$LOG_FILE"

# =============================================================================
# Run all patterns
# =============================================================================

# 1. Around pattern (12 cameras)
run_pipeline "around" "none" 12 "around"

# 2. Spherical pattern (12 cameras)
run_pipeline "spherical" "none" 12 "spherical"

# 3. Rotation pattern (12 cameras)
run_pipeline "rotation" "none" 12 "rotation"

# 4. Linear approach (5 cameras)
run_pipeline "linear" "approach" 5 "linear_approach"

# 5. Linear pass_by (5 cameras)
run_pipeline "linear" "pass_by" 5 "linear_pass_by"

# =============================================================================
# Generate summary
# =============================================================================
echo ""
echo "============================================================"
echo "Generating summary..."
echo "============================================================"

python3 << 'EOF'
import json
import os
from pathlib import Path

output_dir = Path("./final_dataset")
patterns = ["around", "spherical", "rotation", "linear_approach", "linear_pass_by"]

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

total_records = 0
total_questions = 0

for pattern in patterns:
    stats_path = output_dir / pattern / "statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"\n{pattern}:")
        print(f"  Total records: {stats['total_records']}")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Scenes: {stats.get('scenes', 'N/A')}")
        total_records += stats['total_records']
        total_questions += stats['total_questions']
    else:
        print(f"\n{pattern}: NOT FOUND")

print("\n" + "-" * 70)
print(f"TOTAL RECORDS: {total_records}")
print(f"TOTAL QUESTIONS: {total_questions}")
print("=" * 70)
EOF

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
echo "Dataset generation complete!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "============================================================"
