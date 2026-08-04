# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of Google's **GPax** that extends **Gaussian Process Probes (GPP)** — Wang et al., *"Gaussian Process Probes (GPP) for Uncertainty-Aware Probing"*, [arXiv:2305.18213](https://arxiv.org/abs/2305.18213), NeurIPS 2023 — from **binary** to **multiclass** classification, and applies it to probe open-weight LLM representations.

The **headline** contribution is `experiments/disagreement/` (see `docs/DISAGREEMENT_STUDY.md`): does the probe's *aleatoric* component recover genuine **human annotator disagreement** on multi-annotator NLP datasets (LeWiDi K=2 ×4, ChaosNLI K=3, GoEmotions K=28 × 3 LLMs), while its *epistemic* component tracks evidence instead — plus the kernel-capacity diagnosis in `docs/CALIBRATION_STUDY.md`.

`experiments/annomi/` is an applied **case study** (not a headline claim): classifying the **client talk type** (`change` / `sustain` / `neutral`) of utterances in **motivational-interviewing** transcripts, from the hidden representations of **Gemma-3-27b-it / Qwen3-VL-30B / Llama-3.3-70B**. **The dataset is IC-AnnoMI** — the ChatGPT-augmented extension of AnnoMI (Wu et al. 2023) from LREC-COLING 2024, loaded from `IC_AnnoMI.csv` / `IC-AnnoMI (test set).csv` — so part of it is synthetically rewritten dialogue and results are not comparable to published AnnoMI numbers. Lead with **balanced accuracy / macro-F1**, never plain accuracy: training is class-balanced by undersampling while the test set keeps its **65.3% majority** skew. `experiments/shapes3d/` is the controlled synthetic verification that reproduces the paper's Figures 4/5/6.

The cluster (Harvard FASRC Cannon) is where the LLM embedding extraction runs; results are committed back and pulled locally. The local clone has historically lagged `origin` — always `git fetch` first.

## Repository layout

```
GPax/            JAX/Flax implementation of GPP (source of truth for the method)
GPtorch/         pure-PyTorch port of GPax (no JAX deps); parity asserted, not yet used by experiments
experiments/
  shapes3d/      3D-Shapes synthetic verification (CNN embeddings → GPP; reproduces Figs 4/5/6)
  annomi/        IC-AnnoMI motivational-interviewing pipeline (LLM embeddings → GPP); binary/ = 2-way variant
tools/           standalone helper/debug scripts (check_*, debug_jax, download_model)
docs/            reports + extra READMEs, incl. BUGS.md (audit) and PORTING_REPORT.md
tests/           test_parity.py (binary-vs-multiclass episteme parity)
```

**Run convention:** scripts are launched from the repo root (e.g. `python experiments/annomi/step2_exp2_context.py`). The shared modules (`experiments/annomi/annomi_common.py`, `experiments/shapes3d/gpp_common.py`) now discover the repo root from `__file__` and resolve their data/output dirs relative to their own location, so they work regardless of cwd. SLURM batch scripts additionally `export PYTHONPATH=$(pwd)`.

## The three layers

1. **`GPax/` — JAX/Flax (the method).**
   - `GPax/probing/gp.py` — binary **Beta GP**: latent log-normal transform, `gp_predict` (heteroscedastic-noise GP posterior via Cholesky), `beta_gp_predict`, `get_latent_gp` (`f = fα − fβ`), `gp_uncertainty`/`beta_gp_uncertainty`, `classifier_samples_uncertainty`, `mvn_nll`, `cosine_kernel`.
   - `GPax/probing/gp_multiclass.py` — **multiclass extension** (Dirichlet GP): `get_latent_observations_dirichlet`, `dirichlet_gp_predict` (K independent latent GPs), `dirichlet_gp_uncertainty` + `classifier_samples_uncertainty_multiclass` (Monte-Carlo softmax → aleatoric/epistemic).
   - `GPax/probing/probabilistic_probe{,_multiclass}.py` — public API `gpp` / `gpp_multiclass` + baselines `lpe`, `lp_maxprob`, `maha`. `@jax.jit`-compiled.
   - `GPax/models`, `objectives`, `bayesopt` — inherited from upstream gpax/hyperbo; not central. `setup.py` still names the package `hyperbo`.

2. **`GPtorch/` — pure-PyTorch port of `GPax`.** Mirrors `GPax/probing`. `docs/PORTING_REPORT.md` claims parity to `atol=1e-5`, but **no JAX↔torch parity test is committed** (`tests/test_parity.py` only checks binary-vs-multiclass episteme within JAX), and the experiments import the JAX `GPax` probes, not `GPtorch`.

3. **Experiments** (both import JAX `GPax`):
   - **`experiments/shapes3d/`** — `gpp_extended_verification.py` is the original monolith (loads `3dshapes.h5`, trains 3 CNNs via `cnn_model.py`, labels via `ontology.py`, runs probes, plots). Refactored into `step1_train_and_embed.py` → `step2_run_probes.py` → `step3_plot_*.py` with shared `gpp_common.py`. `verify_equivalence.py` checks binary≡multiclass at K=2; `visualize_manifold.py` draws the 3-simplex. `run_*.sh` are the runners.
   - **`experiments/annomi/`** — the LLM/motivation pipeline. `annomi_common.py` holds config: model via env var `ANNOMI_MODEL_TYPE` (`gemma`|`qwen`|`llama`); data/results/figures namespaced per model under `experiments/annomi/{data,results,figures}/<model>/`. `step1_process_data*.py` extract LLM embeddings (HF `transformers`, last-token hidden state, PCA→64-D) → `.npz`. `step2_exp*.py` are the experiments (client-only, +therapist-context, prompted/few-shot, quality) sweeping `N ∈ {50…2400}`. **`step2_exp4_quality.py` is an oracle upper bound** — it embeds the gold therapist-quality label into the test prompt (B2). The "cascading uncertainty" variant exists only in the orphan `run_annomi_pipeline.py`, produces no reported number, and is no longer a claimed contribution (B3). `step3_*.py` visualize/confusion-matrix. `binary/` is the 2-way (`change` vs not) variant. `batch_*.sh` request an A100-80GB on `seas_gpu`; `run_all_cluster.sh` chains them with SLURM dependencies.

## The math (have this down before touching `gp_multiclass.py`)

Per-class, the Dirichlet GP follows Milios et al. 2018, exactly as the paper sketches for multiclass:
- Observed one-hot label → concentration `α_k = α_eps + s·y_k` (`alpha_eps`=ε prior, `strength`=s).
- Log-normal moment-match of `Gamma(α_k,1)`: latent target `μ = log(α_k) − v/2`, heteroscedastic noise `v = log(1/α_k + 1)`. (`get_latent_var_mu_dirichlet`)
- Prior: constant mean `log ε − v/2`, cosine kernel scaled so `k(a,a)=v`. (`set_default_params_dirichlet`)
- K independent GP regressions on those latent targets; class probabilities via **`softmax` of the latent samples** = normalized Gammas = a Dirichlet draw, so MC `E[softmax(f)]` is the correct posterior predictive.
- Uncertainty decomposition: aleatoric `E[H(p)]`, total `H(E[p])`, mutual information `= total − aleatoric`.
- Binary (`gp.py`) is the K=2 special case: `g = σ(fα − fβ)`; cosine kernel = augmented-bias linear kernel (Eq. 3/4).

The latent-transform, kernel, prior, posterior, and the K=2 equivalence have been checked against the paper line-by-line and empirically (`tests/test_parity.py`: judged-probability correlation 1.0000 at K=2). **Subtlety:** the `'Episteme'` key is a **confidence** in *both* paths (binary `−H[g]`; multiclass `−approx_entropy` of the softmax marginals — higher = more certain). Mutual information (the usual "epistemic uncertainty", higher = more uncertain) is returned separately as `'information_gain'`. Don't use `'Episteme'` as an *uncertainty* score without negating it (this was the cause of the inverted misclassification-AUROC — see `docs/BUGS.md`, now fixed).

## Environment & commands

Conda env `gpax-multiclass` (JAX/Flax/Optax + scikit-learn + transformers + h5py). Local: `~/miniconda3/envs/gpax-multiclass`. Cannon: `module load python/3.10.12-fasrc01 cuda/12.4.1-fasrc01 && conda activate gpax-multiclass`. **`pip install` is blocked by a deny rule** — use the env's existing packages.

```bash
mamba env create -f environment.yaml && mamba activate gpax-multiclass   # or: pip install -e .

# 3D-Shapes verification (run from repo root)
python experiments/shapes3d/download_data.py            # fetch 3dshapes.h5 (~480k imgs, to repo root)
python experiments/shapes3d/gpp_extended_verification.py
# or refactored: bash experiments/shapes3d/run_step1.sh && ... run_step2.sh && ... run_step3.sh
python experiments/shapes3d/verify_equivalence.py       # binary ≡ multiclass at K=2
python tests/test_parity.py                             # episteme parity (binary vs multiclass)

# IC-AnnoMI (LLM motivation) — pick a model via env var
ANNOMI_MODEL_TYPE=gemma python experiments/annomi/step1_process_data.py   # extract embeddings (needs GPU+LLM)
ANNOMI_MODEL_TYPE=gemma python experiments/annomi/step2_exp2_context.py   # probe (runs on committed .npz)
# Cluster: sbatch experiments/annomi/batch_step1.sh  (A100-80GB, seas_gpu)
#          or experiments/annomi/run_all_cluster.sh  (full dependency chain)
```

`step2_*` probing runs locally on the **committed** `experiments/annomi/data/<model>/embeddings*.npz` (≈64-D, no GPU/LLM needed); only `step1` embedding-extraction needs the cluster. Read the paper PDF with `pymupdf` (installed): `python -c "import fitz; print(fitz.open(p).get_text())"` (poppler is absent, so the Read tool can't render PDF pages).

## Known issues

A full audited list (with file:line, severity, repro, and fix status) is in **`docs/BUGS.md`**. Fixed on branch `cleanup-and-fixes`: sign-inverted uncertainty-AUROC (B1), NaN binary aleatoric (B4), leftover `jax.debug.print` (B13), missing pandas import (B8), the correct GP NLML (B14), and — as of 2026-08-04 — the **LPE bootstrap** (B11 class coverage + **B20 unseeded RNG**, which made every committed LPE number irreproducible), the **partial-correlation control** (B21) and `mono_alea` (B22).

**Withdrawn rather than fixed** (documented, not repaired): the "Context+Quality" variant leaks the gold therapist-quality label into the *test* prompt, so it is labeled an **oracle upper bound** (B2); the **"cascading uncertainty"** contribution was never in the run path and is no longer claimed (B3). Still open: the missing GPtorch parity test (B17), un-jittered Cholesky (B19), and B5/B7/B10/B12/B15/B16.

**Before quoting any LPE number**, check whether it predates the B20 fix — the disagreement/calibration/K-scaling LPE cells committed before 2026-08-04 are superseded pending a cluster rerun.

## Conventions & gotchas

- `gpp_multiclass` accepts labels as integer indices `(n,)` or one-hot `(n,K)`; pass `num_classes` explicitly (don't let it infer at small `N`).
- The cosine kernel internally augments with a bias dim and L2-normalizes — **do not pre-normalize embeddings**. CNN/ReLU embeddings are non-negative, which compresses cosine similarities into `[0,1]`.
- Large artifacts are git-ignored and **not on GitHub**: 3D-Shapes `*.npy` embeddings (~117 MB each — `rsync` from the cluster). Committed: `experiments/annomi/data/**.npz` (~70 MB), `results/embeddings/data_labels.npz` (58 MB).
- Never commit `huggingfacetoken.txt` (git-ignored). It's read from the repo root at runtime; keep it local to the cluster.
