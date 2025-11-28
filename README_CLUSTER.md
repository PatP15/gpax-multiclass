# GPP Reproduction on Cluster (A100)

This directory contains the scripts and instructions to reproduce Figures 4, 5, and 6 from the GPP paper using a high-compute environment (e.g., A100 GPU).

## Prerequisites

- **Hardware**: A100 GPU (or equivalent) with at least 40GB VRAM is recommended for the full dataset.
- **Software**: Python 3.10+, CUDA drivers compatible with JAX.

## Setup

1.  **Create a Conda Environment** (recommended):
    ```bash
    conda create -n gpax-cluster python=3.10
    conda activate gpax-cluster
    ```

2.  **Install JAX with CUDA Support**:
    *Note: JAX installation depends on your CUDA version. See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).*
    
    For CUDA 12:
    ```bash
    pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
    
    For CUDA 11:
    ```bash
    pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

3.  **Install Other Dependencies**:
    ```bash
    pip install -r requirements_cluster.txt
    ```

## Data

Ensure `3dshapes.h5` is present in the root directory.
- You can download it via `gsutil cp gs://3d-shapes/3dshapes.h5 .` if you have access, or move it from your local machine.

## Running the Experiment

The main script is `gpp_extended_verification.py`.

1.  **Verify Dataset Size**:
    The script is configured for a large run (`N_SUBSET = 100000`). You can adjust this variable in `gpp_extended_verification.py` if needed.
    ```python
    # gpp_extended_verification.py
    N_SUBSET = 100000  # Set this to 100000 or None for full dataset
    ```

2.  **Run the Script**:
    ```bash
    # JAX usually auto-detects GPU, but you can force it:
    JAX_PLATFORM_NAME=gpu python gpp_extended_verification.py
    ```

## Output

The script will generate:
- `figure4_auroc_Binary.png` & `figure4_auroc_Multiclass.png` (Learning curves)
- `figure5_uncertainty_Binary.png` & `figure5_uncertainty_Multiclass.png` (Judged Probability vs Episteme)
- `figure6_alea_episteme_Binary.png` & `figure6_alea_episteme_Multiclass.png` (Alea vs Episteme)

