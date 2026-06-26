# GPax — Multiclass Gaussian Process Probes for LLM representations

A fork of Google's **GPax** that extends **Gaussian Process Probes (GPP)** — Wang et al.,
*[Gaussian Process Probes (GPP) for Uncertainty-Aware Probing](https://arxiv.org/abs/2305.18213)* (NeurIPS 2023) —
from **binary** to **multiclass** classification, and applies it to probe the frozen representations of
open-weight LLMs with **calibrated, decomposable uncertainty** (aleatoric vs. epistemic).

> Disclaimer: builds on code that is not an officially supported Google product.

**What's new in this fork**
- `gpp_multiclass` — Dirichlet-GP multiclass probe (the binary Beta-GP is the K=2 special case).
- Selectable kernels + **`gpp_multiclass_select`**: standardizes inputs and auto-tunes the kernel
  lengthscale by GP marginal likelihood (label-free). This fixes a calibration ceiling of the default
  cosine kernel — see [`docs/CALIBRATION_STUDY.md`](docs/CALIBRATION_STUDY.md).
- A validation suite mirroring the paper's pillars for the multiclass setting
  (see [`docs/MULTICLASS_VALIDATION.md`](docs/MULTICLASS_VALIDATION.md)).
- Datasets for validating the **aleatoric/epistemic decomposition against human label disagreement**
  (ChaosNLI, LeWiDi, GoEmotions) under `experiments/disagreement/`.

---

## 1. Installation

```bash
# Recommended: conda/mamba env (JAX/Flax/Optax + scikit-learn + transformers + h5py)
mamba env create -f environment.yaml
mamba activate gpax-multiclass        # env name is gpax-multiclass

# or editable pip install (Python >= 3.10)
python3 -m venv env && source env/bin/activate && pip install -e .
```

After a fresh `git clone`, also pull the LeWiDi data submodule (see §4):
```bash
git submodule update --init experiments/disagreement/lewidi
```

Optional extra (only to load the GoEmotions parquet): `pip install pyarrow`.

---

## 2. The multiclass GPP API

```python
from GPax.probing.probabilistic_probe_multiclass import gpp_multiclass, gpp_multiclass_select

# x_observed: (n, d) frozen embeddings;  y_observed: (n,) int labels or (n, K) one-hot
# x_query:    (n', d) embeddings to probe
measures = gpp_multiclass(x_query, x_observed, y_observed, num_classes=K)
```

**Output dict** (each is `(n', K)` or `(n', 1)`):

| key | meaning |
|---|---|
| `categorical_mu` | judged class probabilities `E[softmax(f)]` (the Bayesian posterior predictive) |
| `Alea` / `expected_aleatory_entropy` | **aleatoric** uncertainty `E[H(p)]` (concept fuzziness) |
| `information_gain` | **epistemic** uncertainty = mutual information `H(E[p]) − E[H(p)]` |
| `Episteme` | a *confidence* score (higher = more certain); negate to use as an uncertainty |
| `latent_var` | per-class latent posterior variance (OOD proxy; `dirichlet_gp_ood_score`) |

**Kernels & the calibration fix.** The default kernel is the paper's `cosine` (linear); it has no
lengthscale and hits a capacity ceiling as classes/concepts get complex. Use a local kernel with an
auto-selected lengthscale to restore calibration *and* accuracy:

```python
# productized: standardize inputs + pick the lengthscale by marginal likelihood (no held-out set)
m = gpp_multiclass_select(x_query, x_observed, y_observed, num_classes=K,
                          kernel='laplace', lengthscale='auto')   # or kernel='rbf' / 'cosine' / 'rbf_sphere'
# or pass an explicit kernel/lengthscale to the jitted core:
m = gpp_multiclass(x_query, x_observed, y_observed, num_classes=K, kernel='rbf', lengthscale=3.0)
```

**Hyperparameters:** `alpha_eps` (Dirichlet prior ε, default 0.1), `strength` (observation weight s,
default 5.0), `n` (Monte-Carlo samples), `seed`. Do **not** pre-normalize embeddings for the cosine
kernel (it augments + L2-normalizes internally); local kernels standardize inside `gpp_multiclass_select`.

**Binary (original):** `from GPax.probing.probabilistic_probe import gpp` — Beta-GP; `gpp_multiclass`
reduces to it at K=2 (asserted in `tests/test_parity.py`).

**Baselines** (binary / multiclass): `lpe`/`lpe_multiclass` (linear-probe ensemble),
`lp_maxprob`/`lp_maxprob_multiclass` (max-softmax / MSP), `maha`/`maha_multiclass` (Mahalanobis).

<details><summary><b>The math (Dirichlet GP, Milios et al. 2018)</b></summary>

Per class `k`: one-hot label → concentration `α_k = ε + s·y_k`; log-normal moment-match gives a latent
target `μ = log α_k − v/2` with heteroscedastic noise `v = log(1/α_k + 1)`. K independent latent GPs are
fit on those targets (constant mean `log ε − v/2`, kernel scaled so `k(a,a)=v`). Class probabilities are
`softmax` of latent samples (= a Dirichlet draw), so MC `E[softmax(f)]` is the posterior predictive, and
the aleatoric/epistemic split is `E[H(p)]` / mutual information. K=2 recovers the binary Beta-GP `g=σ(fα−fβ)`.
</details>

---

## 3. Repository layout

```
GPax/probing/        the method (JAX/Flax): gp.py (binary Beta-GP), gp_multiclass.py (Dirichlet GP),
                     probabilistic_probe{,_multiclass}.py (public gpp / gpp_multiclass + baselines)
GPtorch/             pure-PyTorch port of GPax/probing (parity-checked; experiments use the JAX version)
experiments/
  shapes3d/          3D-Shapes synthetic verification (CNN embeddings → GPP; reproduces paper Figs 4/5/6)
                     + step4_{decomposition_validation,ood,kscaling}.py (multiclass validation)
  annomi/            AnnoMI motivational-interviewing pipeline (LLM embeddings → GPP, K=3 talk type)
  calibration_study/ GPP-vs-LPE calibration analysis + the kernel rescue (+ annomi_kernel_repeats.py)
  disagreement/      aleatoric-vs-human-disagreement datasets + loaders (ChaosNLI / LeWiDi / GoEmotions)
docs/                MULTICLASS_VALIDATION.md, CALIBRATION_STUDY.md, BUGS.md, PORTING_REPORT.md
tests/               test_parity.py (binary≡multiclass at K=2)
```

Run scripts from the repo root, e.g. `python experiments/annomi/step2_exp2_context.py`.

---

## 4. Datasets & setup

| dataset | task | where / how to get it |
|---|---|---|
| **3D-Shapes** | synthetic, controllable fuzziness (paper's substrate) | `python experiments/shapes3d/download_data.py` (≈480k imgs → repo root `3dshapes.h5`); CNN embedding runs on a GPU/cluster |
| **AnnoMI** | LLM-embedding probe, K=3 client talk-type | committed under `experiments/annomi/data/<model>/*.npz` (no download needed for `step2_*`) |
| **ChaosNLI** | NLI with 100 human annotations/example (human-disagreement ground truth) | **committed** under `experiments/disagreement/data/chaosNLI_v1.0/` (source: [easonnie/ChaosNLI](https://github.com/easonnie/ChaosNLI)) |
| **GoEmotions** | 28-way Reddit emotions, per-rater labels | `python experiments/disagreement/download_data.py` (HF parquet); **`pip install pyarrow`** to load |
| **LeWiDi** | 4 subjective text tasks w/ per-annotator soft labels (2021/2023/2025 editions) | **git submodule**: `git submodule update --init experiments/disagreement/lewidi` |

**Disagreement loaders** (uniform `{text, soft_label, hard_label, n_annot, entropy}` records — the
substrate for "does the probe recover the human label distribution?"):

```bash
python experiments/disagreement/download_data.py     # fetch GoEmotions + check ChaosNLI/LeWiDi
python experiments/disagreement/data_loaders.py      # smoke-test: prints clearest vs most-ambiguous rows
```
```python
from experiments.disagreement.data_loaders import load_chaosnli, load_lewidi, load_goemotions
recs = load_chaosnli('snli')                 # 1,514 NLI items, soft_label over {entail,neutral,contradict}
recs = load_lewidi('MD-Agreement', 'train')  # 6,592 tweets, binary soft_label from ≥5 annotators
recs = load_goemotions(min_annot=3)          # per-comment emotion soft labels (needs pyarrow)
```

---

## 5. Running the experiments

```bash
# 3D-Shapes verification (reproduces paper Figs 4/5/6)
python experiments/shapes3d/gpp_extended_verification.py
python tests/test_parity.py                                   # binary ≡ multiclass at K=2 (asserts)

# multiclass validation suite (3D-Shapes M1 embeddings; see docs/MULTICLASS_VALIDATION.md)
python experiments/shapes3d/step4_decomposition_validation.py # aleatoric/epistemic decomposition
python experiments/shapes3d/step4_ood.py                      # multiclass OOD detection
python experiments/shapes3d/step4_kscaling.py                 # K ∈ {2,4,8,16} scaling

# AnnoMI (LLM probe) — pick a model via env var; step2 runs on committed .npz (no GPU)
ANNOMI_MODEL_TYPE=gemma python experiments/annomi/step2_exp2_context.py

# calibration study + productized-kernel re-validation on real LLM embeddings
python experiments/calibration_study/kernel_compare.py
python experiments/calibration_study/annomi_kernel_repeats.py
```

Cluster: SLURM batch scripts under `experiments/annomi/` request an A100 for embedding extraction
(`step1`); the probing (`step2`) and the synthetic/calibration studies run on CPU.

---

## 6. Key results & docs

- **[`docs/MULTICLASS_VALIDATION.md`](docs/MULTICLASS_VALIDATION.md)** — the multiclass extension validated
  against the paper's three pillars (probing/data-efficiency, fuzziness + rational uncertainty, OOD),
  plus K-scaling to K=16.
- **[`docs/CALIBRATION_STUDY.md`](docs/CALIBRATION_STUDY.md)** — the cosine-kernel calibration ceiling and
  the marginal-likelihood-tuned local-kernel fix (synthetic + real LLM embeddings).
- **[`docs/BUGS.md`](docs/BUGS.md)** — audited issue list with fix status.
- **[`CLAUDE.md`](CLAUDE.md)** — orientation for the codebase internals.

---

## 7. Citation

```bibtex
@article{wang2023gpp,
  title={{Gaussian Process Probes (GPP) for Uncertainty-Aware Probing}},
  author={Zi Wang and Alexander Ku and Jason Baldridge and Thomas L Griffiths and Been Kim},
  journal={arXiv preprint arXiv:2305.18213}, year={2023}
}
```

Datasets, if used: ChaosNLI (Nie et al., EMNLP 2020), LeWiDi (Leonardelli et al., SemEval-2023),
GoEmotions (Demszky et al., ACL 2020), AnnoMI (Wu et al., 2023), 3D-Shapes (Burgess & Kim, 2018).
The Dirichlet-GP construction follows Milios et al. (NeurIPS 2018).
