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
| **Not confidently ignorant** — under low episteme: mean &#124;judged−1/K&#124; / frac extreme (>0.9 or <0.1) | **0.16 / 8%** | 0.30 / 73% | GPP stays near the 1/K prior; LPE makes extreme predictions with no knowledge |

This is the multiclass analog of paper Figs 5–6, and it holds: the decomposition is not just *computed*
but *behaves correctly*. This was the single biggest gap (previously only plotted qualitatively).

## Pillar 3 — multiclass OOD detection

`step4_ood.py` (fair: baselines ID-standardized, LPE 50 members, far-OOD noise regenerated per repeat).
Score = negative summed latent posterior variance (paper §4.4) vs Maha / MSP / LPE / deep-kNN. Mean
AUROC(ID/OOD) over n_obs ∈ {8…128}, 5 seeds:

| regime | GPP (neg latent var) | GPP (Episteme) | kNN | Maha | LPE | MSP |
|---|---|---|---|---|---|---|
| **near-OOD** (held-out 4th shape, novel class) | **0.83** (→0.90 at n=128) | 0.55 | 0.82 | 0.79 | 0.57 | 0.35 |
| **far-OOD** (uniform-noise images) | 0.48 | 0.56 | 0.35 | 0.27 | 0.29 | **0.61** |

- **Near-OOD / novel-class: GPP wins** — the paper's claim reproduces for multiclass. The added deep-kNN
  (Sun et al. 2022) is the strongest baseline (0.82, beating Maha/LPE/MSP), but GPP still edges it (0.83),
  and the win survives the fairness fixes (standardized baselines, 50-member LPE).
- **Far-OOD / noise:** noise embeddings collapse toward the data centroid (a known ReLU-CNN feature-
  collapse effect), which defeats *all* distance/variance scores (GPP-latentvar 0.48, kNN 0.35, LPE 0.29,
  Maha 0.27); the classifier-confidence scores (MSP 0.61, GPP-Episteme 0.56) are more robust. Honest,
  task-dependent finding — the paper itself notes GPP "might not be the best for each task."

## Pillar 1b — generality beyond K=3 (K-scaling)

`step4_kscaling.py`, on labels M1 actually encodes (K=2 floor, K=4 shape, K=8 shape×scale, K=16
shape×scale×floor), GPP-cosine vs **GPP-rbf** (productized: standardize + ML lengthscale) vs the
probing baselines LP / SVM / LPE, at n_obs=512, 5 seeds. Accuracy (higher better):

| K | GPP-cosine | **GPP-rbf** | SVM | LP | LPE | MI monotone (Spearman n_obs,MI) |
|---|---|---|---|---|---|---|
| 2  | 0.990 | **1.000** | 1.000 | 0.998 | 0.998 | −0.945 |
| 4  | 0.638 | **0.992** | 0.825 | 0.736 | 0.740 | −0.945 |
| 8  | 0.639 | **0.984** | 0.903 | 0.803 | 0.805 | −0.945 |
| 16 | 0.689 | **0.984** | 0.968 | 0.907 | 0.900 | −0.945 |

- **GPP-rbf beats the full probing-baseline suite at every K** — ≥0.98 accuracy with no degradation,
  best ECE at K=16 (0.196), best Brier everywhere. SVM (linear) is the strongest classical baseline
  (up to 0.97 at K=16) but still trails GPP-rbf; LP≈LPE.
- **The cosine kernel is what fails to scale:** GPP-cosine collapses to ~0.64 and its gap to the
  baselines *widens* with K — the same capacity ceiling as §3–5c, now visible as a function of K. (This
  corrects an earlier version that used object hue, which M1's color-binarized embeddings cannot resolve,
  masking the effect.)
- **Decomposition stays rational at every K:** MI monotone-decreasing in #obs (Spearman −0.945,
  kernel-independent). So K-scaling reinforces the unified story: *the kernel, not the Dirichlet
  extension, is the lever — and the productized fix removes the ceiling across K.*

## Productized kernel fix on real LLM embeddings

The cosine-kernel capacity ceiling (`docs/CALIBRATION_STUDY.md` §3–5c) is now fixable through the probe
API (`gpp_multiclass(kernel=…)`, `gpp_multiclass_select(lengthscale='auto')`). On AnnoMI (K=3, 4 LLMs,
5 seeds), GPP-Laplace has the best accuracy on every model and slashes ECE on the scale-sensitive ones
(gemma 0.125→0.027, gemma4 0.211→0.069), taking GPP from clearly-worse-calibrated (cosine) to
calibration-competitive-or-better than LPE while leading on accuracy — see `docs/CALIBRATION_STUDY.md` §6.
The same kernel fix is what lets GPP scale to K=16 (Pillar 1b).

---

## Status of the paper's three pillars for multiclass

| pillar | before | now |
|---|---|---|
| 1. accuracy / data-efficiency | ✅ (shapes + AnnoMI) | ✅ + K-scaling to K=16 (productized kernel) + 5-seed error bars |
| 2. fuzziness + rational uncertainty | ⚠️ judged-prob only, decomposition merely plotted | ✅ decomposition quantitatively validated |
| 3. OOD detection | ❌ absent for multiclass | ✅ near-OOD win; far-OOD characterized honestly |
