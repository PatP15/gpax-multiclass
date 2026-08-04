#!/bin/bash
#SBATCH --job-name=dis_probe
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=dis_probe_%x_%j.out
# CPU-only probing/epistemic for the disagreement study (step2 on committed emb npz).
# Usage: sbatch --export=ALL,DS=goemotions,MODEL=llama,VARIANT=mean256 experiments/disagreement/batch_probe.sbatch
#        DS may be a colon-separated list; set EPISTEMIC=1 to also run step2_epistemic (K>=3 datasets only);
#        set SELECTIVE=1 to also run step2_selective (needs step2_probe's probe_summary.json, so it runs after).
set -e
export JAX_PLATFORMS=cpu
export PYTHONNOUSERSITE=1
module load python/3.10.12-fasrc01
source /n/home01/ppuma/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate gpax-multiclass
python -c "import numpy, jax" || { echo "FATAL: conda env not active"; exit 1; }
export PYTHONPATH=$(pwd)
VARIANT_FLAG=""; [ -n "$VARIANT" ] && VARIANT_FLAG="--variant $VARIANT"
echo "=== $(date) dataset=$DS model=$MODEL variant=${VARIANT:-plain} epistemic=${EPISTEMIC:-0} selective=${SELECTIVE:-0} ==="
for DSx in $(echo "$DS" | tr ":" " "); do
  echo "--- probe $DSx ($(date)) ---"
  python experiments/disagreement/step2_probe.py --dataset "$DSx" --model "$MODEL" $VARIANT_FLAG
  if [ -n "$SELECTIVE" ]; then
    echo "--- selective $DSx ($(date)) ---"
    python experiments/disagreement/step2_selective.py --dataset "$DSx" --model "$MODEL" $VARIANT_FLAG
  fi
  if [ -n "$EPISTEMIC" ]; then
    echo "--- epistemic $DSx ($(date)) ---"
    python experiments/disagreement/step2_epistemic.py --dataset "$DSx" --model "$MODEL" $VARIANT_FLAG
  fi
done
echo "=== done $(date) ==="
