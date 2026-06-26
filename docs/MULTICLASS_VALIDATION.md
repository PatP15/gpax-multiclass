# Multiclass GPP — validation against the paper's standard

**Context.** The GPP paper (arXiv:2305.18213) contains *no* multiclass experiments — it only states
(p.4) that extending Beta→Dirichlet is "straightforward." This repo's multiclass extension is therefore
novel and is held here to the paper's *binary* validation bar: its three pillars are (Fig 4) probing
accuracy / data-efficiency, (Figs 5–6) fuzziness + rational uncertainty, and (Fig 7) OOD detection. The
core math was independently checked line-by-line against the paper and reduces to the binary method at
K=2 (`tests/test_parity.py`, judged-prob correlation 0.999998).

All experiments below use 3D-Shapes M1 CNN embeddings (64-D); LLM results are in
`docs/CALIBRATION_STUDY.md` §6. Scripts: `experiments/shapes3d/step4_*.py`. Figures:
`experiments/shapes3d/figures/{decomposition,ood,kscaling}/`. All runs use ≥5 seeds.

---

## Pillar 2a — aleatoric/epistemic decomposition is rational (the headline multiclass claim)

`step4_decomposition_validation.py`, K=3, label-flip fuzziness (paper §4.3). Three quantitative tests,
GPP-Dirichlet vs LPE:

| test | GPP | LPE | reading |
|---|---|---|---|
| **Aleatoric tracks fuzziness** — Pearson(H_gt(p), predicted E[H(p)]) at n=128 | **0.31** | 0.22 | GPP's aleatoric better tracks the true injected label entropy |
| **Epistemic tracks scarcity** — Spearman(n_obs, MI) | **−0.86** | +0.38 | GPP's MI falls monotonically with data (rational); LPE's rises (irrational) |
| **Not confidently ignorant** — under low episteme: mean &#124;judged−1/K&#124; / frac extreme | **0.16 / 0%** | 0.29 / 71% | GPP stays near the 1/K prior; LPE makes extreme predictions with no knowledge |

This is the multiclass analog of paper Figs 5–6, and it holds: the decomposition is not just *computed*
but *behaves correctly*. This was the single biggest gap (previously only plotted qualitatively).

## Pillar 3 — multiclass OOD detection

`step4_ood.py`. Score = negative summed latent posterior variance (paper §4.4) vs Maha / MSP / LPE.
Mean AUROC(ID/OOD) over n_obs ∈ {8…128}:

| regime | GPP (neg latent var) | GPP (Episteme) | Maha | MSP | LPE |
|---|---|---|---|---|---|
| **near-OOD** (held-out 4th shape, novel class) | **0.83** (→0.90 at n=128) | 0.55 | 0.78 | 0.36 | 0.54 |
| **far-OOD** (uniform-noise images) | 0.44 | 0.54 | 0.26 | **0.64** | 0.50 |

- **Near-OOD / novel-class: GPP wins** — the paper's claim reproduces for multiclass.
- **Far-OOD / noise:** noise embeddings collapse toward the data centroid (a known ReLU-CNN feature-
  collapse effect), which defeats *all* distance/variance scores (GPP-latentvar 0.44, Maha 0.26); the
  classifier-confidence scores (MSP 0.64, GPP-Episteme 0.54) are more robust. Honest, task-dependent
  finding — the paper itself notes GPP "might not be the best for each task."

## Pillar 1b — generality beyond K=3 (K-scaling)

`step4_kscaling.py`, object-hue task, K ∈ {2,4,5,10}, GPP vs LPE at n_obs=512:

| K | GPP acc | LPE acc | GPP ECE | LPE ECE | MI monotone (Spearman n_obs,MI) |
|---|---|---|---|---|---|
| 2 | 0.566 | 0.574 | **0.016** | 0.040 | −0.945 |
| 4 | 0.579 | 0.572 | **0.037** | 0.053 | −0.945 |
| 5 | 0.367 | 0.380 | **0.086** | 0.105 | −0.945 |
| 10 | 0.206 | 0.213 | **0.098** | 0.117 | −0.945 |

- **No collapse relative to baseline:** GPP tracks LPE accuracy at every K.
- **GPP keeps its calibration edge:** lower ECE than LPE at *every* K.
- **Decomposition stays rational:** MI is monotone-decreasing in #obs at every K (Spearman −0.945).
- *Caveat:* absolute accuracy is low because fine object hue is a hard target for M1's
  color-*binarized* embeddings; the point here is relative GPP-vs-LPE behavior and scaling, which is clean.

## Productized kernel fix on real LLM embeddings

The cosine-kernel capacity ceiling (`docs/CALIBRATION_STUDY.md` §3–5c) is now fixable through the probe
API (`gpp_multiclass(kernel=…)`, `gpp_multiclass_select(lengthscale='auto')`). On AnnoMI (K=3, 4 LLMs,
5 seeds), GPP-Laplace has the best accuracy on every model and slashes ECE on the scale-sensitive ones
(gemma 0.124→0.027, gemma4 0.211→0.067), reversing the earlier cosine-only "LPE better calibrated"
finding — see `docs/CALIBRATION_STUDY.md` §6.

---

## Status of the paper's three pillars for multiclass

| pillar | before | now |
|---|---|---|
| 1. accuracy / data-efficiency | ✅ (shapes + AnnoMI) | ✅ + K-scaling generality + error bars |
| 2. fuzziness + rational uncertainty | ⚠️ judged-prob only, decomposition merely plotted | ✅ decomposition quantitatively validated |
| 3. OOD detection | ❌ absent for multiclass | ✅ near-OOD win; far-OOD characterized honestly |
