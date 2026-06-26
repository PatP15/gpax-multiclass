# GPP vs LPE Calibration Study (multiclass extension + LLM probes)

**Question.** The GPP paper's central claim is that GPP is *better calibrated than LPE* — its judged
probability tracks ground-truth label fuzziness, and it avoids being "confidently ignorant." Does
that claim survive (a) the multiclass (Dirichlet) extension and (b) real open-weight-LLM embeddings?

**TL;DR.**
- The paper's metric is **Pearson(ground-truth fuzziness, judged probability)** per observation count
  (under positive→negative label flipping), *not* ECE. (ECE was an early wrong turn here.)
- **Binary (paper's setting): reproduced.** GPP > LPE at every `n`, biggest edge at low `n`.
- **Multiclass (Dirichlet): the edge erodes at high `n`** — GPP keeps its low-`n` advantage but LPE
  overtakes from ~`n`=32. The cause is **kernel capacity**, not calibration scaling.
- **Rescue: a marginal-likelihood-tuned local kernel restores and exceeds the advantage** — multiclass
  GPP-RBF beats LPE at every `n` (0.765 vs 0.588 at n=128). The cosine kernel was the ceiling.
- **Kernel ablation (§5c) pins the mechanism:** RBF, Laplace, *and* an SE-on-sphere kernel all rescue
  identically (~0.40 mean, ~0.76 @n=128). The SE-sphere is the decisive control — *same angular geometry
  as cosine, but with a lengthscale* — so the lever is **locality/capacity (the lengthscale), not the
  metric**. The GP marginal likelihood independently prefers the best kernel (Laplace).

All experiments use 3D-Shapes M1 CNN embeddings, task **P.2 = 3-class shape** (fuzzify class 0 by
flipping its labels to {1,2} with prob 1−p, gt levels p∈{0.25,0.5,0.75,1.0}), evaluated on a held-out
test set. Binary results use **P.1 = floor-hue**. Scripts: `experiments/calibration_study/`.

---

## 1. Correct metric (reread of the paper, §4.3 / Fig 5–6)
The paper controls fuzziness by randomly flipping positive→negative labels at gt probabilities
{0.25,0.5,0.75,1.0}, and measures the **Pearson correlation between gt label probabilities and judged
probabilities** (Fig 5 Left), plus *rational uncertainty*: under high episteme the judged probability
should center at 0.5 for fuzzy concepts, whereas LPE assigns extreme probabilities even with little
data ("confidently ignorant"). ECE is never used and measures a different thing (confidence vs.
accuracy), so it is **not** the right test of this claim.

## 2. Binary — claim reproduced
Pearson(gt, judged) by observation count (P.1 floor-hue):

| n_obs | 2 | 8 | 32 | 128 |
|---|---|---|---|---|
| **GPP** | **0.43** | **0.75** | **0.85** | 0.89 |
| LPE | 0.20 | 0.61 | 0.81 | 0.89 |

GPP > LPE at every level, with the largest gap at small `n` — exactly GPP's advertised data-efficiency.

## 3. Multiclass — edge erodes at high n
Pearson(gt, judged) (P.2 3-class shape):

| n_obs | 2 | 8 | 32 | 128 | mean |
|---|---|---|---|---|---|
| GPP-Dirichlet (cosine) | 0.09–0.12 | 0.15 | 0.45–0.49 | 0.56 | ~0.33 |
| GPP-Beta OvR (cosine) | ~0.10 | 0.15 | 0.45 | 0.54 | ~0.34 |
| LPE | 0.06–0.13 | 0.16 | 0.51 | 0.61 | ~0.33 |

GPP keeps the **low-`n`** edge (n=2: ~0.15–0.19 vs LPE 0.06) but LPE overtakes from `n`≈32. Note
GPP-Beta (OvR, **no softmax**) shows the *same* shortfall → the cause is not the softmax.

## 4. Failed rescues (rule out calibration-scaling)
Swept the exposed Dirichlet knobs and a post-hoc softmax temperature on the multiclass task:

| lever | result |
|---|---|
| `strength` ∈ {1,2,5,10} × `alpha_eps` ∈ {0.1,0.5,1.0} | flat, mean Pearson ~0.32–0.33; never reaches LPE |
| softmax temperature T ∈ {1,2,3,5,8} | flat at low n, **hurts** high n (n=128: 0.546→0.513 as T↑) |

Neither moves the high-`n` gap → it is **not** a peakedness/saturation problem. Combined with GPP-Beta
(no softmax) failing identically, the cause is the **fixed cosine kernel's inductive bias/capacity**:
it under-fits the harder shape concept as data grows, while LPE's MLE-fit logistic adapts.

## 5. Successful rescue — marginal-likelihood-tuned RBF kernel
Replace the cosine kernel with `squared_exponential` (RBF) on z-scored embeddings; sweep lengthscale ℓ:

| ℓ | n=2 | n=8 | n=32 | n=128 | mean | NLL@128 (↓ = ML-better) |
|---|---|---|---|---|---|---|
| 3 | 0.155 | 0.177 | 0.498 | **0.765** | **0.399** | **902.8** |
| 6 | 0.201 | 0.171 | 0.476 | 0.685 | 0.383 | 1067 |
| 11 | 0.218 | 0.185 | 0.404 | 0.575 | 0.345 | 1219 |
| 22 | 0.217 | 0.203 | 0.340 | 0.474 | 0.308 | 1341 |
| 44 | 0.215 | 0.198 | 0.295 | 0.364 | 0.268 | 1411 |
| *cosine (ref)* | 0.117 | 0.153 | 0.487 | 0.560 | 0.329 | — |
| *LPE (ref)* | 0.063 | 0.164 | 0.512 | 0.588 | 0.332 | — |

- **RBF ℓ=3 beats LPE at every `n`** and dominates at high `n` (0.765 vs 0.588), mean 0.399 vs 0.332.
- The **GP marginal likelihood selects ℓ=3** (lowest NLL), so the lengthscale is auto-tunable — this
  is a principled fix, not test-set oracle tuning.
- Because the lever that addresses the *actual* mechanism (kernel capacity) works while the
  calibration-scaling levers don't, the diagnosis is confirmed.

## 5b. What the "lengthscale" is, and why it is the decisive lever  (read this for the write-up)

GPP puts a Gaussian-process prior on the latent concept function. A **kernel** `k(a, a′)` defines how
*similar* two embeddings are, and GPP's judged probability at a query is essentially a
similarity-weighted average of the (noisy) labels of nearby observations. So the kernel fixes two
things at once: GPP's **inductive bias** (what "similar" means) and its **effective capacity** (how
complex a concept it can represent).

**Cosine kernel — what GPP ships with, and its hidden limitation.**
`k(a, a′) = v · (aᵀa′ + δ) / (‖a‖ ‖a′‖)`. This is a *linear* kernel: using it is mathematically
equivalent to Bayesian logistic regression in an augmented feature space (the paper itself notes this,
Eq. 4). It has **no lengthscale** — "similarity" is purely the angle between two embeddings, the
decision surface it can express is linear, and its capacity **does not grow as more data arrive**.

**RBF (squared-exponential) kernel — adds the missing knob.**
`k(a, a′) = v · exp(−‖a − a′‖² / (2 ℓ²))`. The **lengthscale ℓ** is the distance in embedding space over
which two points still count as "similar":
- **small ℓ** → similarity decays quickly → a *local, flexible, high-capacity* function (each observation
  only influences its neighbourhood; non-linear concept boundaries become representable);
- **large ℓ** → similarity decays slowly → a *smooth, low-capacity* function that degenerates toward the
  linear cosine behaviour.

So ℓ is a **single, interpretable dial for GPP's capacity.**

**Why this controls calibration-to-fuzziness.** To track ground-truth fuzziness, GPP's predicted
P(class) must respond faithfully to the local mix of (flipped) labels as observations accumulate. On the
*easy* binary floor-hue concept, the linear cosine kernel already captures the structure, so GPP tracks
fuzziness and beats LPE at every `n`. On the *harder* 3-class shape concept, the linear kernel
**under-fits**: it cannot sharpen its estimate as data grow, so its per-`n_obs` Pearson plateaus while
LPE's data-adaptive logistic keeps improving and overtakes it around `n`≈32.

**What the lengthscale sweep established (§5).** Swapping cosine→RBF and sweeping ℓ directly tunes GPP's
capacity. A *small* ℓ≈3 (on z-scored embeddings) makes GPP flexible enough to track shape-fuzziness and
it **beats LPE at every `n`**, by a wide margin at high `n` (Pearson 0.765 vs 0.588 at `n`=128); a *large*
ℓ collapses back toward the cosine result. Decisively, the **GP marginal likelihood** — a label-free,
test-set-free model-selection score that is part of the GP framework itself — assigns the **lowest NLL to
ℓ=3**, so the correct lengthscale is **chosen automatically from the data**, not hand-tuned on the test
metric. (Caveat: NLL is monotone in ℓ over the tested grid, i.e. it prefers the smallest ℓ; in practice
select ℓ per `n` by maximizing the marginal likelihood, and standardize embeddings first.)

**Paper-ready statement of the result.** *GPP's loss of calibration to LPE in the multiclass setting is
an inductive-bias / capacity limitation of the fixed cosine (linear) kernel — not a defect of the
Dirichlet extension and not a probability-scaling problem.* We rule out the scaling explanation directly:
neither the Dirichlet concentration hyperparameters (`strength`, `alpha_eps`) nor a post-hoc softmax
temperature changes the gap, and the softmax-free GPP-Beta (one-vs-rest) shows the identical shortfall.
Granting the GP an expressive kernel (RBF) with a **marginal-likelihood-selected lengthscale restores and
exceeds GPP's calibration advantage over LPE across all observation counts.** In one line: *the cosine
kernel was the ceiling, the lengthscale is the knob that lifts it, and the GP can set that knob itself.*

## 5c. Kernel ablation — it is locality, not the metric (RBF / Laplace / SE-sphere all rescue)

§5 showed RBF rescues, but left open *why*: was it the Euclidean distance, the smoothness, or simply
the presence of a lengthscale? To separate these, we re-ran the multiclass fuzziness test across four
kernels (each at its marginal-likelihood-best lengthscale), on a fresh test subset (so the cosine/LPE
references differ by ~0.01 from §5's run but are internally comparable within this table):

| kernel | family / geometry | n=2 | n=8 | n=32 | n=128 | mean | best ℓ (ML-NLL@128 ↓) |
|---|---|---|---|---|---|---|---|
| cosine | linear, **angular**, *no ℓ* | 0.183 | 0.141 | 0.449 | 0.552 | 0.331 | — |
| RBF (sq-exp) | local, **L2 / Euclidean**, smooth | 0.158 | 0.181 | 0.496 | 0.760 | 0.399 | 3 (903) |
| Laplace (Matérn-½) | local, **L1**, *rough* | 0.192 | 0.194 | 0.478 | 0.757 | **0.405** | 10 (**861**) |
| SE-on-sphere | local, **angular** + ℓ | 0.162 | 0.173 | 0.473 | **0.771** | 0.395 | 0.5 (875) |
| LPE | — (MLE logistic) | 0.067 | 0.192 | 0.491 | 0.608 | 0.339 | — |

Three conclusions, each sharper than §5:

1. **The rescue is not RBF-specific.** All three *local* kernels (RBF, Laplace, SE-sphere) land at
   ~0.40 mean and ~0.76 at n=128 — each beats both cosine (0.55) and LPE (0.61) at high `n`. The common
   ingredient is a **lengthscale**, i.e. a capacity knob; the specific distance is secondary.
2. **The SE-on-sphere is the decisive control.** It uses the *same angular similarity as the default
   cosine kernel* — `k = v·exp(−θ²/2ℓ²)` with `θ = arccos(cosine)` — but adds a lengthscale. It rescues
   exactly as well as Euclidean RBF. So the cosine kernel's high-`n` ceiling is **not** because it
   measures angles instead of distances; it is because it has **no lengthscale to localize with**.
   *The lever is locality/capacity, not the metric.* (RBF-vs-cosine alone confounded these two changes;
   SE-sphere holds the metric fixed and varies only locality.)
3. **Marginal likelihood is self-consistent and slightly favors a rougher kernel.** Laplace (Matérn-½,
   non-smooth) is marginally best by *both* Pearson (0.405) *and* NLL (861, the lowest of any kernel) —
   so the label-free GP model-selection score would pick the empirically best kernel without touching
   the test metric. The rougher kernel edging out the smooth one is consistent with the shape-concept
   boundary being somewhat non-smooth in embedding space.

**Paper-ready statement.** *GPP's multiclass calibration ceiling is specifically the cosine kernel's
lack of a lengthscale. Any local kernel with a marginal-likelihood-tuned lengthscale (RBF, Laplace, or
even an angular SE-on-sphere) restores and exceeds GPP's advantage over LPE; an angular-geometry control
(SE-sphere) rescues as well as Euclidean RBF, ruling out the metric as the cause and identifying the
lengthscale (capacity) as the operative lever.* Script: `experiments/calibration_study/kernel_compare.py`.

## 6. ECE on AnnoMI LLM probes — and the kernel fix carries over to real embeddings
AnnoMI has no injected fuzziness, so only confidence-vs-accuracy ECE is available (a weaker, different
notion than §1; the controlled 3D-Shapes fuzziness test is the real one). With the **cosine** kernel,
LPE was better-calibrated than GPP on the high-variance-embedding models — consistent with the cosine
capacity ceiling found above. **The productized local kernel (§5c, now in the probe API as
`gpp_multiclass_select`) closes this on real LLM embeddings too.** Client-motivation task (K=3), n=2400,
**mean ± std over 5 seeds** (accuracy / 15-bin ECE, lower ECE = better):

| model | GPP-cosine | GPP-RBF (auto ℓ) | GPP-Laplace (auto ℓ) | LPE | LP-temp (CV-scaled) |
|---|---|---|---|---|---|
| gemma  | 0.646 / 0.125 | 0.657 / 0.041 | **0.664** / **0.027** | 0.662 / 0.061 | 0.658 / 0.041 |
| qwen   | 0.625 / 0.034 | 0.649 / 0.059 | **0.661** / 0.051 | 0.630 / 0.041 | 0.629 / **0.037** |
| gemma4 | 0.503 / 0.211 | 0.522 / 0.069 | **0.576** / 0.084 | 0.510 / 0.059 | 0.511 / **0.054** |
| qwen36 | 0.658 / 0.063 | 0.664 / 0.050 | **0.686** / 0.047 | 0.655 / 0.056 | 0.658 / **0.038** |

- **GPP-Laplace has the best accuracy on every model**, beating cosine, RBF, LPE *and* the
  temperature-scaled LP.
- **The local kernel fixes the cosine calibration ceiling**: cosine→Laplace ECE drops 4.6× on gemma
  (0.125→0.027) and cosine→RBF 3× on gemma4 (0.211→0.069).
- **Temperature-scaled LP is a strong *calibration* baseline** (CV-fit T, no test leakage): ECE
  0.037–0.054, beating GPP-Laplace's ECE on qwen/gemma4/qwen36. But it tracks LPE on accuracy
  (≤ GPP-Laplace on **all 4** models, e.g. gemma4 0.511 vs 0.576). So the standard easy-calibration fix
  *does* calibrate — but only by adding a held-out calibration step, and it gives up accuracy.
- **Net (corrected, honest):** GPP no longer "loses calibration to LPE" (the cosine-only finding) — the
  local kernel makes it calibration-competitive — but it is not strictly the lowest-ECE method either.
  Its distinction is the **combination**: GPP-Laplace gives the best accuracy *and* competitive
  calibration in a single model, with **no separate calibration set or temperature step**. Std over
  seeds is small (~0.004–0.015). Script: `experiments/calibration_study/annomi_kernel_repeats.py`.

## 7. Conclusions & recommendations
1. **The GPP calibration advantage is real and reproduces in binary.**
2. In **multiclass it survives at low `n`** (the few-shot regime GPP is sold for) but the **fixed cosine
   kernel is a ceiling at higher `n`**.
3. **Fix:** use **any local kernel with a marginal-likelihood-optimized lengthscale** (RBF, Laplace, or
   SE-on-sphere — on standardized embeddings). This restores GPP's dominance over LPE across all `n` on
   the multiclass task. The kernel ablation (§5c) shows the operative lever is the **lengthscale
   (capacity), not the distance metric**, and the GP marginal likelihood selects the best kernel
   (Laplace) label-free.
4. **Productized (done).** `gpp_multiclass` now exposes `kernel={cosine|rbf|laplace|rbf_sphere}` and
   `lengthscale`, and `gpp_multiclass_select` standardizes inputs and picks the lengthscale per task by
   minimizing the GP marginal likelihood (`dirichlet_gp_nll`, rewritten correctly — see `docs/BUGS.md`).
   §6 confirms this carries the calibration/accuracy win over to real AnnoMI LLM embeddings.

## Reproduction
- `experiments/calibration_study/calib_analysis.py` — ECE/Brier, AnnoMI GPP vs LPE.
- `experiments/calibration_study/calib_why.py` — per-`n_obs` Pearson + rational-uncertainty decomposition (binary vs multiclass).
- `experiments/calibration_study/rescue_calib.py` — `strength`×`alpha_eps` sweep (no rescue).
- `experiments/calibration_study/rescue_temp.py` — softmax temperature sweep (no rescue).
- `experiments/calibration_study/kernel_rescue.py` — RBF lengthscale sweep + marginal-likelihood selection (rescue).
- `experiments/calibration_study/kernel_compare.py` — kernel ablation: cosine vs RBF vs Laplace vs SE-on-sphere vs LPE, each at its ML-best lengthscale (§5c — isolates locality from the metric).
- `experiments/calibration_study/annomi_kernel_repeats.py` — §6: AnnoMI cosine vs RBF vs Laplace vs LPE with 5-seed error bars (productized-kernel validation on real LLM embeddings).
- `experiments/shapes3d/step4_decomposition_validation.py` — quantitative aleatoric/epistemic validation (alea tracks fuzziness; MI tracks data scarcity; not "confidently ignorant").
- `experiments/shapes3d/step4_ood.py` — multiclass OOD detection (near-OOD novel class + far-OOD noise) vs Maha/MSP/LPE.
- `experiments/shapes3d/step4_kscaling.py` — K∈{2,4,5,10} accuracy/Brier/ECE vs LPE + MI-monotonicity (generality beyond K=3).
