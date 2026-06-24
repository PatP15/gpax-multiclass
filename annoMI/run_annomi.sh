#!/bin/bash
# run_annomi.sh
# Usage: ./annoMI/run_annomi.sh

# Ensure we are in the project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Activate environment (assuming existing gpax-multiclass env)
source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
conda activate gpax-multiclass

# Install additional requirements if needed
echo "Installing AnnoMI dependencies..."
pip install -r annoMI/requirements.txt

# Add project root to PYTHONPATH so we can import GPax
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Run the pipeline
echo "Running AnnoMI Pipeline..."
python annoMI/run_annomi_pipeline.py

