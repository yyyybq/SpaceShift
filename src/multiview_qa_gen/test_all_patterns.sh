#!/bin/bash
"""
Test script to generate examples for each move pattern.

This script runs the multiview QA generation pipeline for each of the 4 move patterns,
generating a few examples for each to verify the pipeline works correctly.

Usage:
    bash test_all_patterns.sh
    
Or make it executable:
    chmod +x test_all_patterns.sh
    ./test_all_patterns.sh
"""

# Configuration
OUTPUT_BASE_DIR="./test_output"
SCENE="FloorPlan201"
MAX_OBJECTS=3  # Limit objects for quick testing
NUM_CAMERAS=3  # Few cameras per object for testing

echo "========================================"
echo "Multiview QA Generation - Pattern Tests"
echo "========================================"
echo ""

# Clean up previous test output
rm -rf $OUTPUT_BASE_DIR
mkdir -p $OUTPUT_BASE_DIR

# Test 1: Around pattern (default)
echo "[1/5] Testing AROUND pattern..."
python run_pipeline.py \
    --scenes $SCENE \
    --output_dir $OUTPUT_BASE_DIR \
    --experiment_name test_around \
    --move_pattern around \
    --total_rotation 60 \
    --increment_rotation 20 \
    --radius 1.5 \
    --num_cameras $NUM_CAMERAS \
    --max_objects $MAX_OBJECTS

echo ""

# Test 2: Spherical pattern
echo "[2/5] Testing SPHERICAL pattern..."
python run_pipeline.py \
    --scenes $SCENE \
    --output_dir $OUTPUT_BASE_DIR \
    --experiment_name test_spherical \
    --move_pattern spherical \
    --radius 1.5 \
    --num_cameras $NUM_CAMERAS \
    --max_objects $MAX_OBJECTS

echo ""

# Test 3: Rotation pattern (room-centric)
echo "[3/5] Testing ROTATION pattern..."
python run_pipeline.py \
    --scenes $SCENE \
    --output_dir $OUTPUT_BASE_DIR \
    --experiment_name test_rotation \
    --move_pattern rotation \
    --total_rotation 360 \
    --increment_rotation 45 \
    --max_objects $MAX_OBJECTS

echo ""

# Test 4: Linear pattern - approach
echo "[4/5] Testing LINEAR (approach) pattern..."
python run_pipeline.py \
    --scenes $SCENE \
    --output_dir $OUTPUT_BASE_DIR \
    --experiment_name test_linear_approach \
    --move_pattern linear \
    --linear_sub_pattern approach \
    --radius 2.0 \
    --num_cameras $NUM_CAMERAS \
    --max_objects $MAX_OBJECTS

echo ""

# Test 5: Linear pattern - pass_by
echo "[5/5] Testing LINEAR (pass_by) pattern..."
python run_pipeline.py \
    --scenes $SCENE \
    --output_dir $OUTPUT_BASE_DIR \
    --experiment_name test_linear_passby \
    --move_pattern linear \
    --linear_sub_pattern pass_by \
    --radius 1.5 \
    --num_cameras $NUM_CAMERAS \
    --max_objects $MAX_OBJECTS

echo ""
echo "========================================"
echo "All pattern tests completed!"
echo "========================================"
echo ""
echo "Output structure:"
find $OUTPUT_BASE_DIR -type d -name "test_*" | head -20
echo ""
echo "Generated files:"
find $OUTPUT_BASE_DIR -name "*.json" -o -name "*.jsonl" | head -20
echo ""
echo "Generated images (sample):"
find $OUTPUT_BASE_DIR -name "*.png" | head -10
echo ""
echo "Statistics:"
for dir in $OUTPUT_BASE_DIR/test_*/; do
    if [ -f "${dir}statistics.json" ]; then
        echo "--- $(basename $dir) ---"
        cat "${dir}statistics.json" | python -c "import sys,json; d=json.load(sys.stdin); print(f'  Total questions: {d.get(\"total_questions\", 0)}')"
    fi
done
