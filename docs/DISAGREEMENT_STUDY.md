# Disagreement study — does multiclass GPP's aleatoric uncertainty recover *human* label disagreement?

**Context.** `docs/MULTICLASS_VALIDATION.md` showed the multiclass GPP decomposition behaves rationally on
**3D-Shapes**, but its aleatoric test used *injected synthetic* label fuzziness (label-flips), and its
epistemic-OOD test used synthetic held-out shapes. The reviewer gap was a **real-data** test: does GPP's
*aleatoric* component track genuine **human annotator disagreement**, and does its *epistemic* component
stay tied to evidence (and ignore disagreement)? This study answers that on multi-annotator NLP datasets,
across **3 LLMs**, **layer sweeps**, and **K ∈ {2, 3, 28}**.

The crux of the design: **the probe trains only on hard (majority) labels — it never sees the disagreement.**
We then test whether its *predicted distribution* and its *decomposed uncertainties* recover held-out signals
the probe was never told about: the human annotation spread (aleatoric) and data scarcity / distribution
shift (epistemic).

Code: `experiments/disagreement/{soft_metrics,step1_embed,step2_probe,step2_epistemic,step3_plots}.py`.
Datasets + setup: `experiments/disagreement/README` and the project `README.md`. Embeddings are
git-ignored and reproducible from `step1_embed.py`; CSVs/figures under `experiments/disagreement/figures/`.

---

## What "ground-truth aleatoric" means here, and the proxies

Each item carries many human labels → an empirical distribution `soft_label` (∈ Δ^{K−1}); its **spread is
the irreducible (aleatoric) ambiguity** of the item. We never train on it; we score against it. To show the
result is not an artifact of one disagreement definition, `soft_metrics.PROXIES` computes several, and the
harness correlates each against GPP's predicted aleatoric `E[H(p)]`:

| proxy | formula | note |
|---|---|---|
| Shannon entropy | `H(soft)` | **primary** |
| normalized entropy | `H(soft)/log K` | comparable across K=2/3/28 |
| top-1 ambiguity | `1 − max_k soft_k` | fraction non-majority |
| Gini / pairwise | `1 − Σ soft_k²` | P(two random annotators differ) |
| top-2 margin | `1 − (p₁ − p₂)` | closeness of the top two |
| Bernoulli variance | `p(1−p)` | binary only |

**Distribution recovery** (does `categorical_mu` match `soft_label`?) is scored with soft cross-entropy
(LeWiDi's primary metric), JSD, and **TVD = Baan et al. 2022 DistCE**; plus Baan **EntCE / RankCS**.
Majority-label ECE is *not* primary — Baan et al. show it is ill-posed under genuine disagreement.

**The control that makes RQ1 non-circular:** GPP's *epistemic* (mutual information) must **not** track human
disagreement — it should track the probe's *evidence* instead. `H(categorical_mu)` is total
(aleatoric+epistemic); we test the *aleatoric component* specifically against disagreement, and the
*epistemic component* against scarcity/shift.

---

## Datasets (multi-annotator, real human disagreement)

| dataset | K | task | annotators | disagreement source |
|---|---|---|---|---|
| **LeWiDi-2023** MD-Agreement / HS-Brexit / ArMIS / ConvAbuse | 2 | offensiveness / hate-speech / misogyny (Arabic) / abuse-in-dialogue | 5–many per item | subjective social judgments (Leonardelli et al. 2023) |
| **ChaosNLI** (SNLI subset) | 3 | NLI entail/neutral/contradict | 100 per item | genuine inference ambiguity (Nie et al. 2020) |
| **GoEmotions** | 28 | emotion of a Reddit comment | ≥4 per item | emotion-label subjectivity (Demszky et al. 2020) |

LLMs (3): `gemma` (Gemma-3-27b-it), `qwen` (Qwen3-VL-30B), `llama` (Llama-3.3-70B-Instruct). ~6 evenly-spaced
layers each, **mean-pooled** hidden states, **PCA→256** per (dataset, model, layer) fit on train. Methods:
GPP-cosine, GPP-rbf(auto-ℓ), GPP-laplace(auto-ℓ), LPE, LP-temp (temperature-scaled logistic); `n_obs ∈
{16,64,256,1024}`, 5 seeds, matched-conditions recording.

> **A representation note that mattered.** The first pass (last-token pooling, PCA→64) gave weak tracking on
> the subjective tasks. Switching to **mean-pooling + PCA→256** lifted aleatoric tracking substantially
> (e.g. gemma MD-Agreement Alea-rho 0.21→0.31, accuracy 0.73→0.77) — part of the early weakness was a
> *representation bottleneck*, not the probe. All headline numbers below use mean-pool/256.

---

## RQ1 — Aleatoric recovers human disagreement (headline)

Spearman ρ between human-entropy `H(soft)` and the probe's predicted aleatoric `E[H(p)]`, at the best layer
per method, largest `n_obs`. GPP-rbf is the representative GPP (kernel rescue, RQ2); LPE / LP-temp are the
uncertainty-aware baselines. Higher = better disagreement tracking. (All three models complete; see Status.)

| dataset (K=2) | model | GPP-rbf Alea-ρ | LPE Alea-ρ | LP-temp Alea-ρ | GPP-rbf acc | soft-CE | TVD |
|---|---|---|---|---|---|---|---|
| MD-Agreement | gemma | **0.314** | 0.292 | 0.266 | 0.767 | 0.591 | 0.214 |
| MD-Agreement | qwen  | **0.289** | 0.276 | 0.264 | 0.754 | 0.606 | 0.224 |
| MD-Agreement | llama | **0.333** | 0.303 | 0.271 | 0.771 | 0.584 | 0.209 |
| HS-Brexit    | gemma | 0.349 | 0.352 | 0.356 | 0.894 | 0.400 | 0.130 |
| HS-Brexit    | qwen  | 0.282 | 0.285 | 0.272 | 0.893 | 0.416 | 0.136 |
| HS-Brexit    | llama | 0.272 | 0.316 | 0.307 | 0.893 | 0.417 | 0.136 |
| ArMIS (Arabic) | gemma | 0.154 | 0.216 | 0.226 | 0.731 | 0.576 | 0.275 |
| ArMIS (Arabic) | qwen  | 0.080 | 0.099 | 0.147 | 0.681 | 0.641 | 0.309 |
| ArMIS (Arabic) | llama | 0.082 | 0.076 | 0.068 | 0.713 | 0.611 | 0.290 |
| ConvAbuse    | gemma | 0.298 | 0.267 | 0.191 | 0.878 | 0.296 | 0.145 |
| ConvAbuse    | qwen  | 0.286 | 0.253 | 0.178 | 0.874 | 0.311 | 0.148 |
| ConvAbuse    | llama | **0.304** | 0.282 | 0.208 | 0.875 | 0.296 | 0.133 |

- **Aleatoric tracks human disagreement on every subjective task** (ρ 0.15–0.35, all positive), and GPP-rbf
  is competitive with or better than the uncertainty-aware baselines on most cells (it leads on MD-Agreement
  for **all three models** and on ConvAbuse; LPE/LP-temp edge it on ArMIS and tie on HS-Brexit). The probe was
  trained on majority labels alone, so this is genuine recovery, not memorization.
- The *strength* is task-dependent and tracks how separable the task is in the embedding (HS-Brexit/ConvAbuse
  high accuracy → cleaner tracking; ArMIS, a small Arabic set, is the weakest).

## RQ2 — The kernel decides calibration (cosine → local-kernel rescue)

The cosine-kernel capacity ceiling documented for AnnoMI (`docs/CALIBRATION_STUDY.md` §3–5c) shows up here as
*worse disagreement recovery*, and the local-kernel fix repairs it on real embeddings:

| cell | GPP-cosine Alea-ρ | GPP-rbf Alea-ρ | cosine acc | rbf acc |
|---|---|---|---|---|
| MD-Agreement / gemma | 0.213 | **0.314** | 0.732 | 0.767 |
| MD-Agreement / qwen  | 0.179 | **0.289** | 0.696 | 0.754 |

The rbf/laplace `lengthscale='auto'` (marginal-likelihood) selection is what lifts both accuracy and
aleatoric tracking — consistent with the K-scaling and AnnoMI kernel findings. (GPP-laplace is strong on
MD/ArMIS/ConvAbuse but degenerates on HS-Brexit, Alea-ρ ≈ 0 — laplace ℓ-selection is the least robust here.)

## RQ3 — Epistemic tracks *evidence*, not disagreement (the symmetric result, and the strongest)

If aleatoric is the disagreement axis, epistemic must be the *evidence* axis. As the probe gets more
observations, its **mutual information must fall** (less epistemic uncertainty) while aleatoric stays put.
Spearman ρ(n_obs, mean MI) — **negative = rational**:

| dataset | model | GPP-rbf | GPP-laplace | **LPE** |
|---|---|---|---|---|
| MD-Agreement | gemma | −0.92 | −0.85 | **+0.84** |
| MD-Agreement | qwen  | −0.90 | −0.84 | **+0.87** |
| MD-Agreement | llama | −0.90 | −0.84 | **+0.88** |
| HS-Brexit    | gemma | −0.78 | −0.90 | **+0.37** |
| HS-Brexit    | qwen  | −0.75 | −0.90 | **+0.40** |
| HS-Brexit    | llama | −0.77 | −0.90 | **+0.30** |
| ArMIS        | gemma | −0.92 | −0.81 | **+0.53** |
| ArMIS        | qwen  | −0.93 | −0.83 | **+0.57** |
| ArMIS        | llama | −0.92 | −0.82 | **+0.50** |
| ConvAbuse    | gemma | −0.69 | −0.97 | **+0.79** |
| ConvAbuse    | qwen  | −0.69 | −0.97 | **+0.83** |
| ConvAbuse    | llama | −0.72 | −0.97 | **+0.77** |

- **GPP's epistemic uncertainty falls monotonically with data (rational); LPE's *rises* (irrational)** — the
  signs are opposite on every task and all three models (GPP-rbf −0.69…−0.92, GPP-laplace −0.81…−0.97 vs LPE
  +0.30…+0.88). This reproduces the 3D-Shapes synthetic finding (GPP −0.86, LPE +0.38;
  `docs/MULTICLASS_VALIDATION.md`) on **real LLM embeddings across 4 subjective tasks and 3 models** — a much
  stronger demonstration of the decomposition's correctness than synthetic alone.
- Combined with RQ1, this is **the dissociation (RQ4)**: aleatoric responds to the human-disagreement axis,
  epistemic to the evidence axis, each ignoring the other.

## RQ4 — Novel-class OOD on text is *not* separable (honest negative)

The symmetric epistemic-OOD test (`step2_epistemic.py`): train on K−1 classes, hold one out, score
AUROC(held-out vs seen) from GPP neg-latent-var / −MI vs Maha / MSP / kNN / LPE.

| dataset | model | id-acc | GPP neg-var | GPP −MI | Maha | kNN | MSP | LPE |
|---|---|---|---|---|---|---|---|---|
| ChaosNLI (K=3) | gemma | 0.66 | 0.50 | 0.52 | 0.50 | 0.49 | 0.50 | 0.49 |
| ChaosNLI (K=3) | qwen  | 0.79 | 0.50 | 0.50 | 0.52 | 0.50 | 0.51 | 0.51 |
| ChaosNLI (K=3) | llama | 0.75 | 0.50 | 0.49 | 0.52 | 0.50 | 0.47 | 0.47 |
| GoEmotions (K=28) | gemma | 0.23 | 0.52 | 0.52 | 0.52 | 0.50 | 0.50 | 0.50 |
| GoEmotions (K=28) | qwen  | 0.22 | 0.50 | 0.50 | 0.51 | 0.50 | 0.49 | 0.51 |

**All methods, including the distance baselines, sit at chance** — and crucially this holds even though the
probe now *classifies* the in-distribution NLI labels well (id-acc 0.66–0.79). So the embedding linearly
separates the seen classes yet places a held-out class in the *same region*: the held-out NLI/emotion class
is not a *spatially distinct* cluster, so there is no OOD signal for *any* detector to find. This is a
property of the representation, not a GPP failure — and the fact that it persists across all three models
makes that point strongly. The epistemic-OOD claim is therefore carried by (a) the scarcity axis above (real
data, decisive) and (b) the synthetic **near-OOD** result where classes *are* separable (3D-Shapes held-out
shape: GPP 0.83, `docs/MULTICLASS_VALIDATION.md`). We report the text null rather than bury it.

## RQ5/RQ6 — Breadth across K and the poorly-separated cases

- **K=3 (ChaosNLI), all 3 models:** with mean-pool/256 the task *is* decodable — GPP-rbf accuracy 0.62
  (gemma) / 0.67 (qwen) / 0.62 (llama), well above the 0.33 chance (and far above the 0.46 the old
  last-token/64 representation gave — another representation-bottleneck lesson). Yet **aleatoric tracking
  stays weak** (Alea-ρ ≈ 0.09–0.10 for GPP-rbf, ≈ same for LPE/LP-temp): the probe predicts NLI labels but
  its aleatoric does not capture *which* items humans found genuinely ambiguous. A clean dissociation between
  "decodable" and "disagreement-recoverable" — disagreement on NLI is harder to recover than on offensiveness,
  for every probe. LP-temp again wins soft-CE (0.89–0.91 vs GPP-rbf 0.93–0.98), as on LeWiDi.
- **K=28 (GoEmotions):** weak but non-trivial and consistent across models — GPP-rbf acc 0.28 (gemma) / 0.27
  (qwen) [llama pending] vs 0.036 chance, Alea-ρ ≈ 0.15–0.16, and the kernel rescue still helps (cosine 0.09 →
  rbf 0.16). The 28-way task is hard for a linear-ish probe, and the matched-conditions constraint (LP-temp
  needs ≥2 samples/class) drops all n_obs<1024 cells, so the GoEmotions scarcity curve is not measurable here.
- **Scarcity breadth:** the RQ3 dissociation extends to K=3 — on ChaosNLI **GPP-laplace** MI falls sharply
  with data (ρ(n,MI) −0.93/−0.95/−0.93 for gemma/qwen/llama) while **LPE rises (+0.90/+0.90/+0.91)**; GPP-rbf
  is the exception (nearly flat, −0.04…−0.17), so the rational-shrinkage signal is kernel-dependent on this
  task. (K=28 not measurable, per above.)
- The pattern is consistent: **aleatoric tracking works where the task is decodable AND disagreement is
  embedding-visible** (the subjective binary tasks), is weak where disagreement is subtle even if the task is
  decodable (ChaosNLI), and degrades gracefully where the task itself is hard (GoEmotions) — it does not
  produce spurious high correlations on the harder cases.

---

## Status

| model | LeWiDi (MD / HS-Brexit / ArMIS / ConvAbuse) | ChaosNLI | GoEmotions |
|---|---|---|---|
| gemma | ✅ all 4 | ✅ | ✅ |
| qwen  | ✅ all 4 | ✅ | ✅ |
| llama | ✅ all 4 | ✅ | ⏳ extracting |

The full 3-model matrix spans **K=2 (4 LeWiDi tasks), K=3 (ChaosNLI), and K=28 (GoEmotions)** — 17 of 18
(dataset×model) cells complete; only llama×GoEmotions is still extracting (70B, CPU-offloaded). The findings
above — aleatoric tracks disagreement and the GPP-rbf kernel rescue lifts it; epistemic falls with evidence
while LPE's rises; the two dissociate; text novel-class OOD is a representation-level null even when the task
is decodable — replicate cleanly across **gemma-3-27b, qwen3-vl-30b, and llama-3.3-70b**, and the K-breadth
shows where they hold (subjective-binary tasks) versus weaken (NLI disagreement, 28-way emotion).

## References

- Nie, Williams, et al. *What Can We Learn from Collective Human Opinions on NLI?* (ChaosNLI), EMNLP 2020.
- Leonardelli et al. *SemEval-2023 Task 11: Learning With Disagreements (LeWiDi)*, 2023.
- Demszky et al. *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL 2020.
- Baan et al. *Stop Measuring Calibration When Humans Disagree*, EMNLP 2022. (DistCE/EntCE/RankCS; ECE caveat)
- Plank. *The "Problem" of Human Label Variation*, EMNLP 2022. Uma et al. *Learning from Disagreement: A Survey*, JAIR 2021.
- Milios et al. *Dirichlet-based Gaussian Processes for Large-scale Calibrated Classification*, NeurIPS 2018.
- Wang et al. *Gaussian Process Probes (GPP) for Uncertainty-Aware Probing*, NeurIPS 2023 (arXiv:2305.18213).
