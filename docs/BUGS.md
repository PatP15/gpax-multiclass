# GPP Multiclass Fork — Bug & Correctness Audit

Audit of the multiclass GPP fork and its experiments against the original paper
(*Gaussian Process Probes*, [arXiv:2305.18213](https://arxiv.org/abs/2305.18213)) and
Milios et al. 2018 (Dirichlet-based GP classification).

**Method:** (1) read the full paper and hand-checked the implementation against the
equations; (2) two adversarial multi-agent audits (each finding independently
re-verified by a skeptic against the source) over the core library + 3D-Shapes driver
and over the AnnoMI pipeline; (3) reproduced the headline issues locally by running the
real harness on the committed Gemma embeddings.

**Verdict:** the **core GP math is correct** — the Dirichlet extension faithfully
generalizes the paper's Beta GP (latent log-normal transform, `α_k = ε + s·y_k`, cosine
kernel with `k(a,a)=v`, K independent latent GPs, softmax-of-latents = Dirichlet draw).
The K=2 equivalence holds empirically (`tests/test_parity.py`: judged-probability
correlation = **1.0000**). **The bugs are in the experiment harness and the uncertainty
API, not the GP itself.** Several materially affect reported numbers/figures.

Paths below are post-restructure. Fixes marked ✅ are applied on branch `cleanup-and-fixes`
(commit `398f4a5`).

---

## Summary table

| ID | Severity | Status | One-line |
|----|----------|--------|----------|
| B1 | 🔴 Critical | ✅ Fixed | Uncertainty-AUROC sign-inverted (reported ~0.31, true ~0.69) |
| B2 | 🔴 Critical | 🚫 Closed (descoped) | Gold test-set quality label leaked into "Context+Quality" prompt — now labeled an oracle upper bound |
| B3 | 🔴 Critical | 🚫 Closed (descoped) | "Cascading uncertainty" not implemented in the run path — withdrawn as a contribution |
| B4 | 🟠 High | ✅ Fixed | Binary aleatoric/info-gain → `NaN` for confident samples |
| B5 | 🟠 High | ⚠️ Open | Silent `except: pass` hides probe failures in 3D-Shapes driver |
| B6 | 🟠 High | ⚠️ Open (design) | Synthetic "ambiguity" ≠ paper's label-flip; `gt_prob` = interpolation weight |
| B7 | 🟠 High | ⚠️ Open | MC uncertainty allocates `n_query×K×n` (`n` default 1e5) → OOM risk |
| B8 | 🟠 High | ✅ Fixed | `uncertainty_analysis.py` `NameError` (no `import pandas`) |
| B9 | 🟠 High | ✅ Addressed | Repeats added: `annomi_kernel_repeats.py` + step4 scripts run ≥5 seeds with mean±std error bars |
| B10 | 🟡 Med | ⚠️ Open | 3D-Shapes ground-truth teacher leakage + `num_classes` inferred from KNN |
| B11 | 🟡 Med | ✅ Fixed | LPE bootstrap can omit a class → zero-prob column biases entropy/MI |
| B12 | 🟡 Med | ⚠️ Open | Multiclass OvR-AUROC assumes prob columns align to sorted labels |
| B13 | 🔵 Low | ✅ Fixed | ~14 leftover `jax.debug.print` in the jitted probe path |
| B14 | 🔵 Low | ✅ Fixed | `dirichlet_gp_nll` rewritten (correct GP NLML, used for ML lengthscale selection); `dirichlet_mnll` reimplemented on the MC-softmax predictive |
| B15 | 🔵 Low | ⚠️ Open | `set_default_params_dirichlet` mutates the shared `params` dict |
| B16 | 🔵 Low | ⚠️ Open | `verify_equivalence.py` uses the SE kernel, not the cosine kernel probes use |
| B17 | 🔵 Low | ⚠️ Open | GPtorch JAX↔torch parity test claimed in PORTING_REPORT but not committed |
| B18 | 🔵 Low | ⚠️ Watch | Last-token pooling assumes right-padding (OK only at batch_size=1) |
| B19 | 🔵 Low | ⚠️ Partly fixed | Un-jittered Cholesky; transcript-level split now asserted in `load_annomi_data` (number pending a cluster run) |
| B20 | 🔴 Critical | ✅ Fixed | LPE bootstrap used the unseeded global `np.random` → every committed LPE number irreproducible |
| B21 | 🟡 Med | ✅ Fixed | Non-circularity control measured as a raw correlation between two coupled quantities |
| B22 | 🔵 Low | ✅ Fixed | `mono_alea` was `nan` on 5070/9030 rows (quantile edges collapse on discrete human entropy) |

---

## Critical

### B1 — Uncertainty-AUROC is sign-inverted ✅ Fixed
**Where:** `experiments/annomi/annomi_common.py` (`run_training_loop`, the `roc_auc_score(misclassified, epistemic)` line) and `experiments/annomi/binary/annomi_common_binary.py`; root cause in `GPax/probing/gp.py` and `gp_multiclass.py`.

The harness's reported "auroc" is **misclassification detection**: target = `(pred != y_test)`,
score = the per-query `'Episteme'`. But `'Episteme'` is a **confidence** in *both* paths —
binary `gp.py` returns `Episteme = -entropy`; multiclass `gp_multiclass.py` returns
`Episteme = -approx_entropy` (sum of softmax-marginal differential entropies), explicitly
commented "higher means more concentrated p". Higher Episteme ⇒ more *certain* ⇒ *less*
likely misclassified, so scoring misclassification with `+Episteme` is backwards and lands
**below 0.5**. (Mutual information — the usual epistemic uncertainty — is computed but stored
separately as `'information_gain'`, unused here.)

**Evidence (reproduced):** on the committed Gemma embeddings, scoring with `Episteme`
gives AUROC ≈ 0.31; with `-Episteme`, ≈ 0.69. Local re-run before/after the fix:

| | before | after |
|---|---|---|
| Multiclass AUROC (n=50/100) | 0.39 / 0.37 | **0.61 / 0.63** |
| Binary AUROC (n=50/100) | 0.38 / 0.34 | **0.62 / 0.66** |

**Impact:** every uncertainty-AUROC number in the AnnoMI tables/figures was inverted — the
method looked broken when it actually detects errors well.
**Fix:** negate the score (`roc_auc_score(misclassified, -epistemic)`) in both harnesses.

> Note: an earlier draft of this audit (and the pre-`git pull` code) described multiclass
> `Episteme` as *mutual information* with opposite polarity to binary. The current code sets
> both to `−entropy` (confidence), **same** polarity, **both** inverted — confirmed by the
> empirical run above.

### B2 — Gold test-set quality label leaked into the input 🚫 Closed (descoped)

**Resolution (2026-08-04).** Not repaired; the variant is withdrawn as a result. The
"Context+Quality" scenario is labeled an **oracle upper bound** wherever it appears, and
IC-AnnoMI is demoted from headline claim to case study. Implementing the honest version
(train a quality probe on TRAIN only, inject out-of-fold predictions for TRAIN and model
predictions for TEST) remains the fix if the variant is ever promoted back. Original
diagnosis below.
**Where:** `experiments/annomi/step1_process_data.py` (`process_split`, the `qual_map`/
`X_context_qual` construction).

The "Context+Quality" feature embeds the **ground-truth** therapist-quality label into the
prompt text — `INPUT: Therapist: {t} (Quality: {High|Low}) Client: {c}` — for the **test**
split too (`y_qual` is the gold label from `annomi_common.map_quality`, not a prediction).
This is consumed by `step2_exp4_quality.py` and fed straight to the 3-class motivation GPP.

**Impact:** inflates the "Context+Quality" scenario relative to the "Context" baseline —
exactly the comparison the experiment reports. The leak is into the input features (not the
motivation target), but gold therapist quality correlates with client talk type.
**Recommended fix (changes the experiment, your call):** train the binary quality GPP on
TRAIN only and inject the *predicted* quality for TEST (and out-of-fold predictions for
TRAIN), or label the variant explicitly as an oracle upper bound.

### B3 — "Cascading uncertainty" is not implemented in the run path 🚫 Closed (descoped)

**Resolution (2026-08-04).** Withdrawn from the contribution list rather than implemented —
the paper now rests on the disagreement study and the kernel-capacity diagnosis. The orphan
`run_annomi_pipeline.py` path stays in the tree but is documented as not producing any
reported number. Original diagnosis below.
**Where:** `experiments/annomi/step2_exp4_quality.py`, `experiments/annomi/run_annomi_pipeline.py`.

The README's novel contribution — concatenating the upstream quality model's
`[aleatoric, epistemic]` onto the 64-D embedding (→ 66-D) — is **absent from the submitted
pipeline**. `step2_exp4_quality.py` just loads a different text embedding (`X_*_context_qual`,
still 64-D) and runs the standard loop. The actual `hstack` of uncertainties exists only in
`run_annomi_pipeline.py`, which **no SLURM batch script runs**, uses a different/older model,
and there concatenates **all-zeros** because it reads dict keys that don't exist (the GPP
measures use `'Judged probability'`/`'Alea'`/`'Episteme'`, not the keys it looks up).
**Impact:** the stated contribution is not in the reported results. Decide which experiment
is the real one and implement it in the modular pipeline if needed.

---

## High

### B4 — Binary aleatoric / info-gain → `NaN` for confident samples ✅ Fixed
**Where:** `GPax/probing/gp.py` (`classifier_samples_uncertainty`, `gp_uncertainty`).

`bernoulli_entropy`/`info_gain` took `log(p)`/`log(1-p)` with no clip; at large `n` the most
confident queries hit `p≈0`/`p≈1` and produced `NaN` (a guard `+eps=1e-10` was insufficient
in float32). One NaN poisons the whole `np.mean(aleatoric)` column.
**Evidence:** 11/973 NaN at n=100 on Gemma → **0** after the fix.
**Fix:** clip `p` and `mu` into `[1e-7, 1-1e-7]` before the logs; also `latent_var = max(latent_var, 1e-32)` before `sqrt`/`log` in `gp_uncertainty`.

### B5 — Silent `except: pass` hides probe failures ⚠️ Open
**Where:** `experiments/shapes3d/gpp_extended_verification.py` (multiple probe-call sites);
`compute_auroc` also returns `np.nan` on any exception.

A probe that fails on *every* iteration contributes zero rows and silently vanishes from the
plotted curve — indistinguishable from "not run". **Not auto-fixed** because converting these
to `raise` could abort long runs on benign small-`n` failures; the right fix is to *log* the
exception (count + first traceback per method) and assert ≥1 success per `(model, n, method)`
cell.

### B6 — Synthetic "ambiguity" ≠ the paper's mechanism ⚠️ Open (design)
**Where:** `experiments/shapes3d/gpp_extended_verification.py` (`simulate_ambiguous_data`).

Ambiguity is fabricated by linearly interpolating two class centroids and declaring the
interpolation weight `α` the ground-truth class-1 probability (`y_gt = alphas`). The paper
instead injects label noise (flip positive→negative with probability `p`; `gt_prob = 1-p`).
An embedding at the midpoint of two centroids has no defined Bernoulli label probability, so
this conflates feature-space position with label uncertainty. **Scope:** this applies only to the
`simulate_ambiguous_data` *accuracy* path. The calibration study and the new decomposition validation
use the paper's proper label-flip mechanism (`gpp_common.run_fuzziness_experiment`,
`experiments/calibration_study/*`, `experiments/shapes3d/step4_decomposition_validation.py`), so the
Fig 5/6 / §1 calibration and decomposition results are *not* affected by this issue.

### B7 — Monte-Carlo memory blow-up ⚠️ Open
**Where:** `GPax/probing/gp_multiclass.py` (`dirichlet_gp_uncertainty`/`gp_uncertainty_multiclass`).

Allocates a `(n_query × K × n)` array with `n` defaulting to `1e5`. A few-thousand-row query
set × K × 1e5 is tens of GB. Mitigated in AnnoMI (`run_training_loop` passes `n=1000`) but
the default and the 3D-Shapes paths are exposed. Lower `n` or batch the queries.

### B8 — `uncertainty_analysis.py` `NameError` ✅ Fixed
**Where:** `experiments/shapes3d/uncertainty_analysis.py`. Used `pd.DataFrame` with no
`import pandas as pd` → crashes before Figs 5/6. **Fix:** added the import.

### B9 — No repeats / single seed → no error bars ✅ Addressed
**Where:** `experiments/annomi/annomi_common.py` (`run_training_loop`),
`experiments/shapes3d/gpp_extended_verification.py`.

The original `run_training_loop` still fits one seed, but the new validation scripts all run
≥5 seeds and report mean ± std with CI bands: `experiments/calibration_study/annomi_kernel_repeats.py`
(AnnoMI, 5 seeds) and `experiments/shapes3d/step4_{decomposition_validation,ood,kscaling}.py`. The
headline AnnoMI/calibration claims now carry error bars; the legacy single-seed path remains for the
original step2 plots.

---

## Medium

### B10 — 3D-Shapes ground-truth teacher leakage ⚠️ Open
`experiments/shapes3d/gpp_extended_verification.py`: the KNN "teacher" that defines `gt_prob`
is fit on the same embeddings/split the M1 probe uses (circular), and `num_classes` is taken
from `gt_probs.shape[1]` (classes the KNN happened to see), not the true `K` — which can
undercount classes and corrupt one-hot/AUROC at small `N`.

### B11 — LPE bootstrap can omit a class ✅ Fixed
`GPax/probing/probabilistic_probe_multiclass.py` (`lpe_multiclass`): bootstrap resamples may
miss a class; the padding leaves that class column at exactly zero probability, biasing the
entropy / mutual-information estimates downward.

**Why it mattered more than "medium" suggests:** LPE is the comparator in the flagship RQ3
dissociation (GPP's MI falls with evidence, LPE's rises), so a known downward bias in LPE's
entropy/MI sat underneath the headline claim.

**Evidence (reproduced):** with a 2-member rare class at `n_obs=40`, P(a member omits it) =
0.129, and 4 of 40 members omitted it — each contributing an exactly-zero column.

**Fix:** guarantee class coverage per resample (one forced draw per observed class, then
`n_obs − |classes|` free draws), mirroring the trick the binary `lpe` already used
(`probabilistic_probe.py:131-133`). Classes absent from `y_observed` entirely now receive the
Laplace floor `1/(n + K)` and the row is renormalized, instead of exactly zero. Guarded by
`tests/test_lpe_bootstrap.py`.

### B12 — Multiclass OvR-AUROC column alignment ⚠️ Open
`compute_auroc(..., multi_class='ovr')` assumes the probability columns map to sorted label
indices. GPP's `categorical_mu` columns are class-indexed (OK if labels are `0..K-1`), but
sklearn baselines order columns by `classes_`, and small-`N` runs that omit a class produce
fewer columns → silently corrupted or `nan` scores.

---

## Low / cleanup / watch

- **B13 ✅ Fixed** — removed ~14 `jax.debug.print` host-callbacks from `GPax/probing/gp_multiclass.py` and `probabilistic_probe_multiclass.py` (they fire every jitted call → noise + serialized device→host sync).
- **B14 ✅ Fixed** — `gp_multiclass.py`: `dirichlet_gp_nll` rewritten to the correct summed GP log-marginal-likelihood (no more `mvn_nll`/`params['constant']` bug); it is now the basis for label-free lengthscale selection in `gpp_multiclass_select`. `dirichlet_mnll` reimplemented on the canonical MC-softmax predictive. `tests/test_parity.py` now asserts the K=2 judged-probability parity (corr > 0.999) instead of only printing it.
- **B15 ⚠️** — `set_default_params_dirichlet` mutates the shared `params` dict in place (side effects can leak across the K-class loop / across calls).
- **B16 ⚠️** — `experiments/shapes3d/verify_equivalence.py` builds its check with `squared_exponential_kernel`, not the `cosine_kernel` the probes actually use, and its broad `try/except` can make the check pass/return without printing FAILURE.
- **B17 ⚠️** — `docs/PORTING_REPORT.md` claims a GPU-verified JAX↔torch parity suite, but `tests/test_parity.py` only checks binary-vs-multiclass episteme within JAX. No committed test exercises `GPtorch` against `GPax`.
- **B18 ⚠️ watch** — `experiments/annomi/annomi_common.py` last-token pooling uses `attention_mask.sum(1)-1`, which assumes **right** padding; correct only because `step1` extracts at `batch_size=1`. A left-padding tokenizer with batching would grab a pad token.
- **B19 ⚠️ partly fixed** — `GPax/probing/gp.py` `gp_predict` Cholesky still has no jitter (diagnosed *not* to be the cause of B4 — the posterior was finite — but a risk for near-duplicate observations). The transcript-leakage half is now handled: `load_annomi_data` computes the `transcript_id` intersection between the two external CSVs, prints train/test/shared counts, and raises unless `ANNOMI_ALLOW_OVERLAP=1`. The CSVs live only on the cluster, so the actual number lands with the next cluster run.

---

## Added by the 2026-08-04 pre-rerun pass

### B20 — LPE bootstrap used the unseeded global RNG ✅ Fixed
**Where:** `GPax/probing/probabilistic_probe_multiclass.py` (`lpe_multiclass`, the
`np.random.choice` resample) and `GPax/probing/probabilistic_probe.py` (`lpe`, three draws).

Neither took a seed, and no caller ever seeded the global `np.random` (callers use local
`RandomState`/`default_rng` objects, which do not affect it). So the seed loops in
`step2_probe.py`, `annomi_kernel_repeats.py`, `step4_kscaling.py`, `step4_ood.py` and
`annomi_common.run_training_loop` **did not control LPE at all**: its per-seed "repeats" were
independent draws from OS entropy, and no committed LPE number is reproducible.

**Impact:** every LPE cell in `docs/DISAGREEMENT_STUDY.md`, `docs/CALIBRATION_STUDY.md` §6 and
`docs/MULTICLASS_VALIDATION.md` — including the LPE side of the flagship RQ3 dissociation and
the error bars, which were partly over RNG noise rather than seed variation.
**Fix:** `rng=` parameter → `np.random.default_rng(rng)`, threaded from all five callers.
Guarded by `tests/test_lpe_bootstrap.py` (same `rng` → bit-identical; different `rng` → differs).

### B21 — The non-circularity control was measured with the wrong statistic ✅ Fixed
**Where:** `experiments/disagreement/step2_probe.py` (`corrMI_*` rows) and the claim at
`docs/DISAGREEMENT_STUDY.md` §"What ground-truth aleatoric means here".

The doc asserts that epistemic MI "must **not** track human disagreement" and expects ≈0. The
committed `corrMI_entropy` is 0.20–0.35 on 11 of 54 GPP cells, which reads as the control
failing. It is not: aleatoric `E[H(p)]` and MI are both functions of the same posterior and are
strongly coupled here (`corrAleaMI` = 0.62–0.95), so a raw `corr(MI, human-H)` inherits
aleatoric's association by construction and cannot answer the question.

**Evidence:** partial Spearman on the committed `per_item.npz` arrays — MI's association
vanishes or inverts once aleatoric is held fixed (MD/qwen +0.129 → −0.114; HS-Brexit/gemma
+0.242 → −0.181) while aleatoric survives conditioning (+0.309 → +0.304; +0.327 → +0.287).
**Fix:** `soft_metrics.partial_spearman`; `step2_probe` records
`pcorrAleaS_entropy_given_MI`, `pcorrMIS_entropy_given_alea` and `corrAleaMI`. The doc's control
is restated as a partial correlation (RQ1b), which also yields a stronger claim: GPP-rbf's
aleatoric survives conditioning where LPE's collapses (HS-Brexit 0.343 → 0.046).

### B22 — `mono_alea` silently `nan` on 44% of rows ✅ Fixed
**Where:** `experiments/disagreement/step2_probe.py` (`binned_mono`).

Human entropy is discrete (few annotators → few attainable values), so `np.quantile` edges
collapse and `np.digitize` leaves fewer than 3 non-empty bins; the function then returns `nan`.
Empirically `nan` on **5070 of 9030** committed rows, and the `mono_alea` column of
`aggregate_summary.csv` is empty for every cell — a reported metric that never populated.
**Fix:** when there are fewer distinct human-entropy values than requested bins, group by
distinct value instead of by quantile (still `nan` below 3 distinct values, which is honest).

## Notes on things checked and found OK / dismissed

- **Core math** — latent transform, kernel, prior, posterior, softmax-MC predictive, and K=2 equivalence all match the paper/Milios.
- **`JAX_PLATFORM_NAME=cpu`** — an earlier concern about the main driver forcing CPU no longer applies to the current code; it only appears in two small utility scripts (`uncertainty_analysis.py`, `verify_equivalence.py`), where CPU is harmless.
- **3-class masking** in the 3D-Shapes `main()` (flagged with a worried comment by the author) is actually consistent — embeddings and labels are masked from the same subset.
- **Un-jittered Cholesky causing the binary NaN** — investigated and refuted; the NaN was in the entropy log (B4), not the GP posterior.
