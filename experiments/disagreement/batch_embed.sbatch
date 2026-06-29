#!/bin/bash
#SBATCH --job-name=dis_embed
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=dis_embed_%x_%j.out
# Usage: sbatch --export=ALL,DS=chaosnli_snli,MODEL=gemma experiments/disagreement/batch_embed.sbatch
set -e
export HF_HOME=/n/netscratch/walsh_lab_seas/Everyone/ppuma/.cache/huggingface
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu                      # keep JAX off the GPU; torch uses it
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load python/3.10.12-fasrc01 cuda/12.4.1-fasrc01
# Robustly (re)activate the conda env regardless of inherited shell state:
# source conda.sh, deactivate any inherited env, then activate ours so its python
# (with numpy/torch) is first on PATH ahead of the module python.
source /n/home01/ppuma/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate gpax-multiclass
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -c "import numpy, torch" || { echo "FATAL: conda env not active (no numpy/torch)"; exit 1; }
export PYTHONPATH=$(pwd)
PROMPT_FLAG=""; [ -n "$PROMPT" ] && PROMPT_FLAG="--prompt"
POOL=${POOL:-last}; PCADIM=${PCADIM:-64}
echo "=== $(date) dataset=$DS model=$MODEL prompt=${PROMPT:-0} python=$(which python) ==="
for DSx in $(echo "$DS" | tr "," " "); do
  echo "--- extracting $DSx ($(date)) ---"
  ANNOMI_MODEL_TYPE=$MODEL python experiments/disagreement/step1_embed.py --dataset "$DSx" --model "$MODEL" $PROMPT_FLAG --pool "$POOL" --pca-dim "$PCADIM"
done
echo "=== done $(date) ==="
