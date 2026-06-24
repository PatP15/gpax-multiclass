#!/bin/bash
# Submit Binary AnnoMI jobs
# Usage: ./run_binary_cluster.sh [qwen|gemma]

MODEL_TYPE=${1:-qwen}
export ANNOMI_MODEL_TYPE=$MODEL_TYPE

echo "========================================"
echo "Running Binary AnnoMI Pipeline with Model: $MODEL_TYPE"
echo "========================================"

LOG_DIR="/n/home01/ppuma/gpax-multiclass/experiments/annomi/binary/logs/$MODEL_TYPE"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"

echo "Submitting Step 1 Binary..."
JID1=$(sbatch --parsable --export=ALL -o "$LOG_DIR/step1_%%j.out" -e "$LOG_DIR/step1_%%j.err" experiments/annomi/binary/batch_step1_binary.sh)
echo "Submitted Step 1: $JID1"

echo "Submitting Step 2 Binary..."
JID2=$(sbatch --parsable --export=ALL --dependency=afterok:$JID1 -o "$LOG_DIR/step2_%%j.out" -e "$LOG_DIR/step2_%%j.err" experiments/annomi/binary/batch_step2_binary.sh)
echo "Submitted Step 2: $JID2"

echo "Submitting Step 3 Binary..."
JID3=$(sbatch --parsable --export=ALL --dependency=afterok:$JID2 -o "$LOG_DIR/step3_%%j.out" -e "$LOG_DIR/step3_%%j.err" experiments/annomi/binary/batch_step3_binary.sh)
echo "Submitted Step 3: $JID3"

echo "All binary jobs submitted. Check status with 'squeue -u $USER'"
