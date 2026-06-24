#!/bin/bash
# Submit all AnnoMI jobs with dependencies
# Usage: ./run_all_cluster.sh [gemma|qwen]

# Default to gemma if not provided
MODEL_TYPE=${1:-gemma}
export ANNOMI_MODEL_TYPE=$MODEL_TYPE

echo "========================================"
echo "Running AnnoMI Pipeline with Model: $MODEL_TYPE"
echo "========================================"

# Determine Log Subdirectory
if [ "$MODEL_TYPE" == "qwen" ]; then
    LOG_SUBDIR="qwen"
else
    LOG_SUBDIR="gemma"
fi

LOG_DIR="/n/home01/ppuma/gpax-multiclass/annoMI/logs/$LOG_SUBDIR"
# Ensure logs dir
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"

echo "Submitting Step 1 (Embeddings - Unprompted)..."
JID1=$(sbatch --parsable --export=ALL -o "$LOG_DIR/step1_%%j.out" -e "$LOG_DIR/step1_%%j.err" annoMI/batch_step1.sh)
echo "Submitted Step 1: $JID1"

echo "Submitting Step 1 Prompted (Embeddings - Prompted)..."
JID1P=$(sbatch --parsable --export=ALL -o "$LOG_DIR/step1_prompted_%%j.out" -e "$LOG_DIR/step1_prompted_%%j.err" annoMI/batch_step1_prompted.sh)
echo "Submitted Step 1 Prompted: $JID1P"

echo "Submitting Step 2 Unprompted (Context & Context+Quality)..."
JID2=$(sbatch --parsable --export=ALL --dependency=afterok:$JID1 -o "$LOG_DIR/step2_unprompted_%%j.out" -e "$LOG_DIR/step2_unprompted_%%j.err" annoMI/batch_step2_unprompted.sh)
echo "Submitted Step 2 Unprompted: $JID2"

echo "Submitting Step 2 Prompted (Context & Context+Quality)..."
JID2P=$(sbatch --parsable --export=ALL --dependency=afterok:$JID1P -o "$LOG_DIR/step2_prompted_%%j.out" -e "$LOG_DIR/step2_prompted_%%j.err" annoMI/batch_step2_prompted.sh)
echo "Submitted Step 2 Prompted: $JID2P"

echo "Submitting Step 2 LPE (Baseline)..."
JID2LPE=$(sbatch --parsable --export=ALL --dependency=afterok:$JID1P -o "$LOG_DIR/step2_lpe_%%j.out" -e "$LOG_DIR/step2_lpe_%%j.err" annoMI/batch_step2_lpe.sh)
echo "Submitted Step 2 LPE: $JID2LPE"

echo "Submitting Step 3 (Visualize)..."
# Step 3 depends on ALL Step 2 jobs completing
JID3=$(sbatch --parsable --export=ALL --dependency=afterok:$JID2:$JID2P:$JID2LPE -o "$LOG_DIR/step3_%%j.out" -e "$LOG_DIR/step3_%%j.err" annoMI/batch_step3.sh)
echo "Submitted Step 3: $JID3"

echo "All jobs submitted. Check status with 'squeue -u $USER'"
