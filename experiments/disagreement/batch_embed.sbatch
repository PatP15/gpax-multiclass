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
source ~/.bashrc 2>/dev/null || true
conda activate gpax-multiclass
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)
echo "=== $(date) dataset=$DS model=$MODEL python=$(which python) ==="
ANNOMI_MODEL_TYPE=$MODEL python experiments/disagreement/step1_embed.py --dataset "$DS" --model "$MODEL"
echo "=== done $(date) ==="
