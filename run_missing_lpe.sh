#!/bin/bash
# Submit missing LPE and Step 3
MODEL=$1
if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model_type>"
    exit 1
fi

export ANNOMI_MODEL_TYPE=$MODEL
LOG_DIR="/n/home01/ppuma/gpax-multiclass/annoMI/logs/$MODEL"
mkdir -p "$LOG_DIR"

echo "Submitting LPE for $MODEL..."
# No dependency needed as Step 1 Prompted is done
JID_LPE=$(sbatch --parsable --export=ALL -o "$LOG_DIR/step2_lpe_resubmit_%%j.out" -e "$LOG_DIR/step2_lpe_resubmit_%%j.err" annoMI/batch_step2_lpe.sh)
echo "Submitted LPE: $JID_LPE"

echo "Submitting Step 3 for $MODEL..."
JID_3=$(sbatch --parsable --export=ALL --dependency=afterok:$JID_LPE -o "$LOG_DIR/step3_resubmit_%%j.out" -e "$LOG_DIR/step3_resubmit_%%j.err" annoMI/batch_step3.sh)
echo "Submitted Step 3: $JID_3"

