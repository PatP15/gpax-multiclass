# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of Google's **GPax** that extends **Gaussian Process Probes (GPP)** — Wang et al., *"Gaussian Process Probes (GPP) for Uncertainty-Aware Probing"*, [arXiv:2305.18213](https://arxiv.org/abs/2305.18213), NeurIPS 2023 — from **binary** to **multiclass** classification, and applies it to probe open-weight LLM representations.

The headline research application lives in `annoMI/`: classifying the **client talk type** (`change` / `sustain` / `neutral`) of utterances in **motivational-interviewing** transcripts (the AnnoMI dataset), from the hidden representations of **Gemma-3-27b-it / Qwen3-VL-30B / Llama-3.3-70B**, and decomposing the probe's uncertainty into aleatoric vs. epistemic. The `3D-Shapes` code is the controlled synthetic verification of the method that reproduces the paper's Figures 4/5/6.

The cluster (Harvard FASRC Cannon) is where the LLM embedding extraction and experiments run; results are committed back and pulled locally. Note: the local working copy historically lagged `origin` — always `git fetch` first.

## The three layers (this is the big picture)

1. **`GPax/` — JAX/Flax implementation (the source of truth for the method).**
   - `GPax/probing/gp.py` — binary **Beta GP** (the original GPP). Latent log-normal transform, `gp_predict` (heteroscedastic-noise GP posterior via Cholesky), `beta_gp_predict`, `get_latent_gp` (combines the two latent GPs into `f = fα − fβ`), `gp_uncertainty`/`beta_gp_uncertainty`, `mvn_nll`, `cosine_kernel`.
   - `GPax/probing/gp_multiclass.py` — the **multiclass extension** (Dirichlet GP). `get_latent_observations_dirichlet`, `dirichlet_gp_predict` (K independent latent GPs), `dirichlet_gp_uncertainty` + `classifier_samples_uncertainty_multiclass` (Monte-Carlo softmax → aleatoric/epistemic).
   - `GPax/probing/probabilistic_probe.py` / `probabilistic_probe_multiclass.py` — the **public API**: `gpp` / `gpp_multiclass` plus baselines `lpe`/`lpe_multiclass` (bootstrap logistic ensemble), `lp_maxprob`, `maha`. These are `@jax.jit`-compiled.
   - `GPax/models`, `GPax/objectives`, `GPax/bayesopt` — inherited from upstream gpax/hyperbo (Bayesian-optimization GPs, EKL/NLL objectives). Not central to the probing work. `setup.py` still names the package `hyperbo`.

2. **`GPtorch/` — a pure-PyTorch port of `GPax` (no JAX deps).** Mirrors the `GPax/probing` structure. Purpose: torch-native inference alongside torch LLM embeddings. `docs/PORTING_REPORT.md` claims parity to `atol=1e-5`, checked by `tests/test_parity.py`. **The experiment scripts currently import the JAX `GPax` probes, not `GPtorch`** — so `GPtorch` parity is asserted but not yet wired into the result-producing path.

3. **Experiments** (both import JAX `GPax`):
   - **3D-Shapes reproduction.** `gpp_extended_verification.py` is the original monolith (loads `3dshapes.h5`, trains 3 CNNs `cnn_model.py`/`cnn_model_torch.py`, builds concept labels `ontology.py`, runs probes, plots Figs 4/5/6). It was later refactored into `step1_train_and_embed.py` → `step2_run_probes.py` → `step3_plot_*.py` with shared helpers in `gpp_common.py`. `verify_equivalence.py` checks binary≡multiclass at K=2; `visualize_manifold.py` draws the 3-simplex.
   - **`annoMI/` — the LLM/motivation pipeline (the real contribution).** `annomi_common.py` holds config: model is chosen by env var `ANNOMI_MODEL_TYPE` (`gemma`|`qwen`|`llama`), data/results/figures are namespaced per model under `annoMI/{data,results,figures}/<model>/`. `step1_process_data.py` extracts LLM embeddings (HF `transformers`, `AutoModelForCausalLM`/`AutoModelForVision2Seq`) → `.npz`. `step2_exp*.py` are four experiments (client-only baseline, +therapist-context, therapist-quality binary, cascading-uncertainty) sweeping `N ∈ {50…2400}`. `step3_*.py` visualize / confusion-matrix. `annoMI/binary/` is the 2-way (`change` vs not) variant. SLURM `batch_step*.sh` request an A100-80GB on `seas_gpu`.

## The math (have this down before touching `gp_multiclass.py`)

Per-class, the Dirichlet GP follows Milios et al. 2018, exactly as the paper sketches for the multiclass case:
- Observed one-hot label → concentration `α_k = α_eps + s·y_k` (`alpha_eps`=ε prior, `strength`=s).
- Log-normal moment-match of `Gamma(α_k,1)`: latent target `μ = log(α_k) − v/2`, heteroscedastic noise `v = log(1/α_k + 1)`. (`get_latent_var_mu_dirichlet`)
- Prior: constant mean `log ε − v/2`, cosine kernel scaled so `k(a,a)=v`. (`set_default_params_dirichlet`)
- K independent GP regressions on those latent targets; class probabilities via **`softmax` of the latent samples** — which equals normalized Gammas = a Dirichlet draw, so the Monte-Carlo `E[softmax(f)]` is the correct posterior predictive.
- Uncertainty: aleatoric `E[H(p)]`, total `H(E[p])`, **epistemic = mutual information = total − aleatoric**.
- Binary (`gp.py`) is the K=2 special case: `g = σ(fα − fβ)`, cosine kernel = augmented-bias linear kernel (Eq. 3/4 of the paper).

The latent-transform, kernel, prior, and posterior code all match the paper/Milios; this has been checked line-by-line. **One genuine inconsistency to keep in mind:** the binary path reports `Episteme = −H[g]` (differential entropy, the paper's §3.4 definition) while the multiclass path reports `Episteme = mutual information`. They are different quantities under the same key, so binary and multiclass `Episteme` are **not on the same scale** — don't compare them directly (see Known issues).

## Environment & commands

Conda env is `gpax-multiclass` (JAX/Flax/Optax + scikit-learn + transformers + h5py). Locally: `~/miniconda3/envs/gpax-multiclass`. On Cannon: `module load python/3.10.12-fasrc01 cuda/12.4.1-fasrc01 && conda activate gpax-multiclass`.

```bash
# Create env
mamba env create -f environment.yaml && mamba activate gpax-multiclass
# or pip-editable: pip install -e .  (Python ≥3.10)

# 3D-Shapes: data then the reproduction
python download_data.py                      # fetches 3dshapes.h5 (~480k imgs)
python gpp_extended_verification.py          # monolith → figure4/5/6_*.png + *.csv
# or the refactored pipeline:
bash run_step1.sh && bash run_step2.sh && bash run_step3.sh

# Equivalence / parity checks
python verify_equivalence.py                 # binary ≡ multiclass at K=2
python tests/test_parity.py                  # JAX vs torch / binary-vs-multiclass episteme parity

# AnnoMI (LLM motivation) — pick a model via env var
ANNOMI_MODEL_TYPE=gemma python annoMI/step1_process_data.py     # extract embeddings
ANNOMI_MODEL_TYPE=gemma python annoMI/step2_exp2_context.py     # an experiment
# On the cluster, submit the SLURM stages instead:
sbatch annoMI/batch_step1.sh                 # A100-80GB, seas_gpu
```

Reading the paper PDF locally: poppler is **not** installed (so the Read tool can't render PDF pages), but `pymupdf` is installed in the `gpax-multiclass` env — extract text with `python -c "import fitz; print(fitz.open(p).get_text())"`. `pip install` is blocked by a deny rule; use the env's existing packages.

## Known correctness issues (audited against the paper; fix-worthy before trusting figures)

The core GP math is correct. The issues are concentrated in the experiment harness and the shared uncertainty API. Highest-impact first:

- **Silent `except: pass` around every probe call** (`gpp_extended_verification.py` ~414/444/462/472/490) and `compute_auroc → np.nan` on error: a probe that fails on *every* iteration just vanishes from the plotted curve — indistinguishable from "not run." Log failures and assert ≥1 success per cell.
- **`Episteme` means different things** in binary (`−H[g]`) vs multiclass (mutual information). Pick one convention or rename; don't cross-compare. (`gp.py` vs `gp_multiclass.py`; docs `README_MULTI.md` and `annoMI/README.md` also disagree.)
- **Monte-Carlo memory blowup**: `dirichlet_gp_uncertainty` allocates `(n_query × K × n)` with `n` defaulting to `1e5`. Large query sets (AnnoMI test sets, 3D-Shapes) can OOM; lower `n` or batch the queries.
- **`uncertainty_analysis.py` uses `pd.DataFrame` without importing pandas** → `NameError` before Figs 5/6.
- **Synthetic "ambiguity" ≠ the paper's mechanism**: `simulate_ambiguous_data` interpolates class centroids and calls the interpolation weight the ground-truth probability, instead of flipping labels positive→negative as the paper does. Fig 5/6 calibration rests on this.
- **`JAX_PLATFORM_NAME=cpu` is hard-set** in `gpp_extended_verification.py`'s `__main__` — forces CPU even on an A100. Remove for cluster runs.
- **Teacher/ground-truth leakage (medium)**: the KNN "teacher" that defines `gt_prob` is fit on the same embeddings/split the M1 probe uses; `num_classes` is derived from `gt_probs.shape[1]` (classes the KNN happened to see), not true `K`.
- **Multiclass AUROC column alignment**: `roc_auc_score(multi_class='ovr')` assumes probability columns map to sorted label indices; sklearn baselines (`classes_` order) and small-`N` runs that omit a class can silently corrupt or `nan` the score.
- **Dead/never-imported code**: `dirichlet_gp_nll` calls `mvn_nll` without importing it (would `NameError`); `dirichlet_mnll` reconstructs `alpha` from the posterior variance incorrectly. Both unused by the experiment path — don't wire them in without fixing.
- **Leftover `jax.debug.print`** (~15 calls inside the jitted probe path) — host callbacks fire every call; noisy and slow at experiment scale. Delete.

## Conventions & gotchas

- `gpp_multiclass` accepts labels as either integer indices `(n,)` or one-hot `(n,K)`; pass `num_classes` explicitly (don't let it infer from the data, which can undercount classes at small `N`).
- Cosine kernel internally augments with a bias dim and L2-normalizes — **do not pre-normalize embeddings**. Note CNN/ReLU embeddings are non-negative, which compresses cosine similarities into `[0,1]`.
- Large result artifacts (`*.npy` 3D-Shapes embeddings, ~117 MB each) are git-ignored and **not on GitHub** — `rsync` them from the cluster if needed. `annoMI/data/**.npz` (~70 MB) and `results/embeddings/data_labels.npz` (58 MB) are committed.
- Never commit `huggingfacetoken.txt` (now git-ignored). HF auth on the cluster uses that token; keep it local to the cluster.
- Tests use Google `absl.testing` (upstream gpax) and a plain-script `tests/test_parity.py`; there is no pytest config.
