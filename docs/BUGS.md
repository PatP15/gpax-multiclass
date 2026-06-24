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
| B2 | 🔴 Critical | ⚠️ Open (design) | Gold test-set quality label leaked into "Context+Quality" prompt |
| B3 | 🔴 Critical | ⚠️ Open (design) | "Cascading uncertainty" not implemented in the run path; orphan version broken |
| B4 | 🟠 High | ✅ Fixed | Binary aleatoric/info-gain → `NaN` for confident samples |
| B5 | 🟠 High | ⚠️ Open | Silent `except: pass` hides probe failures in 3D-Shapes driver |
| B6 | 🟠 High | ⚠️ Open (design) | Synthetic "ambiguity" ≠ paper's label-flip; `gt_prob` = interpolation weight |
| B7 | 🟠 High | ⚠️ Open | MC uncertainty allocates `n_query×K×n` (`n` default 1e5) → OOM risk |
| B8 | 🟠 High | ✅ Fixed | `uncertainty_analysis.py` `NameError` (no `import pandas`) |
| B9 | 🟠 High | ⚠️ Open (design) | No repeats / single seed → no error bars (paper uses repeats) |
| B10 | 🟡 Med | ⚠️ Open | 3D-Shapes ground-truth teacher leakage + `num_classes` inferred from KNN |
| B11 | 🟡 Med | ⚠️ Open | LPE bootstrap can omit a class → zero-prob column biases entropy/MI |
| B12 | 🟡 Med | ⚠️ Open | Multiclass OvR-AUROC assumes prob columns align to sorted labels |
| B13 | 🔵 Low | ✅ Fixed | ~14 leftover `jax.debug.print` in the jitted probe path |
| B14 | 🔵 Low | ⚠️ Open | Dead/broken code: `dirichlet_gp_nll` (un-imported `mvn_nll`), `dirichlet_mnll` |
| B15 | 🔵 Low | ⚠️ Open | `set_default_params_dirichlet` mutates the shared `params` dict |
| B16 | 🔵 Low | ⚠️ Open | `verify_equivalence.py` uses the SE kernel, not the cosine kernel probes use |
| B17 | 🔵 Low | ⚠️ Open | GPtorch JAX↔torch parity test claimed in PORTING_REPORT but not committed |
| B18 | 🔵 Low | ⚠️ Watch | Last-token pooling assumes right-padding (OK only at batch_size=1) |
| B19 | 🔵 Low | ⚠️ Watch | Un-jittered Cholesky; transcript-level train/test split unverifiable in-repo |

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

### B2 — Gold test-set quality label leaked into the input ⚠️ Open (design decision)
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

### B3 — "Cascading uncertainty" is not implemented in the run path ⚠️ Open (design)
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
this conflates feature-space position with label uncertainty — and it's the *preferred* source
for the Fig 5/6 calibration plots.

### B7 — Monte-Carlo memory blow-up ⚠️ Open
**Where:** `GPax/probing/gp_multiclass.py` (`dirichlet_gp_uncertainty`/`gp_uncertainty_multiclass`).

Allocates a `(n_query × K × n)` array with `n` defaulting to `1e5`. A few-thousand-row query
set × K × 1e5 is tens of GB. Mitigated in AnnoMI (`run_training_loop` passes `n=1000`) but
the default and the 3D-Shapes paths are exposed. Lower `n` or batch the queries.

### B8 — `uncertainty_analysis.py` `NameError` ✅ Fixed
**Where:** `experiments/shapes3d/uncertainty_analysis.py`. Used `pd.DataFrame` with no
`import pandas as pd` → crashes before Figs 5/6. **Fix:** added the import.

### B9 — No repeats / single seed → no error bars ⚠️ Open (design)
**Where:** `experiments/annomi/annomi_common.py` (`run_training_loop`),
`experiments/shapes3d/gpp_extended_verification.py`.

AnnoMI fits one stratified subsample per `n` with a fixed seed — no repeats, so the curves
have no variance estimate (the paper averages over repeats). Wrap the per-`n` fit/eval in a
seed loop and report mean ± std.

---

## Medium

### B10 — 3D-Shapes ground-truth teacher leakage ⚠️ Open
`experiments/shapes3d/gpp_extended_verification.py`: the KNN "teacher" that defines `gt_prob`
is fit on the same embeddings/split the M1 probe uses (circular), and `num_classes` is taken
from `gt_probs.shape[1]` (classes the KNN happened to see), not the true `K` — which can
undercount classes and corrupt one-hot/AUROC at small `N`.

### B11 — LPE bootstrap can omit a class ⚠️ Open
`GPax/probing/probabilistic_probe_multiclass.py` (`lpe_multiclass`): bootstrap resamples may
miss a class; the padding leaves that class column at exactly zero probability, biasing the
entropy / mutual-information estimates downward.

### B12 — Multiclass OvR-AUROC column alignment ⚠️ Open
`compute_auroc(..., multi_class='ovr')` assumes the probability columns map to sorted label
indices. GPP's `categorical_mu` columns are class-indexed (OK if labels are `0..K-1`), but
sklearn baselines order columns by `classes_`, and small-`N` runs that omit a class produce
fewer columns → silently corrupted or `nan` scores.

---

## Low / cleanup / watch

- **B13 ✅ Fixed** — removed ~14 `jax.debug.print` host-callbacks from `GPax/probing/gp_multiclass.py` and `probabilistic_probe_multiclass.py` (they fire every jitted call → noise + serialized device→host sync).
- **B14 ⚠️** — `gp_multiclass.py`: `dirichlet_gp_nll` calls `mvn_nll` which it never imports (would `NameError`) and assigns the whole params dict to `params['constant']`; `dirichlet_mnll` reconstructs `alpha` from the *posterior* variance (not the moment-match inverse). All unused by the experiment path — don't wire them in without fixing.
- **B15 ⚠️** — `set_default_params_dirichlet` mutates the shared `params` dict in place (side effects can leak across the K-class loop / across calls).
- **B16 ⚠️** — `experiments/shapes3d/verify_equivalence.py` builds its check with `squared_exponential_kernel`, not the `cosine_kernel` the probes actually use, and its broad `try/except` can make the check pass/return without printing FAILURE.
- **B17 ⚠️** — `docs/PORTING_REPORT.md` claims a GPU-verified JAX↔torch parity suite, but `tests/test_parity.py` only checks binary-vs-multiclass episteme within JAX. No committed test exercises `GPtorch` against `GPax`.
- **B18 ⚠️ watch** — `experiments/annomi/annomi_common.py` last-token pooling uses `attention_mask.sum(1)-1`, which assumes **right** padding; correct only because `step1` extracts at `batch_size=1`. A left-padding tokenizer with batching would grab a pad token.
- **B19 ⚠️ watch** — `GPax/probing/gp.py` `gp_predict` Cholesky has no jitter (diagnosed *not* to be the cause of B4 — the posterior was finite — but a risk for near-duplicate observations). Also, AnnoMI's train/test split comes from two external CSVs; whether they share `transcript_id`s (transcript-level leakage) can't be verified in-repo — add an assertion in `load_annomi_data`.

## Notes on things checked and found OK / dismissed

- **Core math** — latent transform, kernel, prior, posterior, softmax-MC predictive, and K=2 equivalence all match the paper/Milios.
- **`JAX_PLATFORM_NAME=cpu`** — an earlier concern about the main driver forcing CPU no longer applies to the current code; it only appears in two small utility scripts (`uncertainty_analysis.py`, `verify_equivalence.py`), where CPU is harmless.
- **3-class masking** in the 3D-Shapes `main()` (flagged with a worried comment by the author) is actually consistent — embeddings and labels are masked from the same subset.
- **Un-jittered Cholesky causing the binary NaN** — investigated and refuted; the NaN was in the entropy log (B4), not the GP posterior.
