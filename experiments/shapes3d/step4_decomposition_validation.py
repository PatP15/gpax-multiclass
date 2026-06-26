"""WS1: Quantitative validation of the multiclass aleatoric/epistemic decomposition.

The headline novel capability of the multiclass GPP extension is decomposing the
predictive uncertainty into aleatoric (expected entropy, E[H(p)]) and epistemic
(mutual information, MI). The paper validates this for binary GPP (Figs 5-6); here
we validate it quantitatively for K=3 on 3D-Shapes M1 embeddings, against LPE.

Three tests (paper analogs in parentheses):
  1.1 Aleatoric tracks fuzziness (Fig 6): inject class-0 label noise at gt prob
      p in {0.25,0.5,0.75,1.0}; the true label dist of a class-0 stimulus is
      [p,(1-p)/2,(1-p)/2] with known entropy H_gt(p). At high data (n=128) the
      predicted aleatoric should correlate with H_gt. Report Pearson(H_gt, alea).
  1.2 Epistemic tracks data scarcity (Fig 6): on clean labels (p=1.0), MI should
      decrease monotonically as observations grow 2 -> 128 (more data -> less
      epistemic). Report Spearman(n_obs, MI) (expect strongly negative).
  1.3 "Not confidently ignorant" (Fig 5 mid/right): under low episteme on fuzzy
      data (p=0.5), GPP's judged P(class 0) should concentrate near the prior
      1/K = 1/3, whereas LPE makes extreme (near 0/1) predictions.

Self-contained (mirrors experiments/calibration_study/*.py). Run on the cluster
where results/embeddings/embeddings_M1.npy lives. CPU only.
"""
import os, sys
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, jax, jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.model_selection import train_test_split
from GPax.probing import gp_multiclass as gpm
from GPax.probing import probabilistic_probe_multiclass as ppm

# ---- data: M1 embeddings, 3-class shape task (shapes 0,1,2; shape 3 reserved) ----
emb = np.load(f'{REPO}/results/embeddings/embeddings_M1.npy')
d = np.load(f'{REPO}/results/embeddings/data_labels.npz', allow_pickle=True)
y_all = d['P2_shape']; L = min(len(emb), len(y_all)); emb, y_all = emb[:L], y_all[:L]
m = y_all < 3; X, y = emb[m], y_all[m].astype(int)
Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
ridx = np.random.RandomState(0).choice(len(Xte_raw), 1000, replace=False)
Xte_raw, yte = Xte_raw[ridx], yte[ridx]
mu_, sd_ = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-8
Xtr_z, Xte_z = (Xtr_raw - mu_) / sd_, (Xte_raw - mu_) / sd_
Xte_raw_j, Xte_z_j = jnp.array(Xte_raw), jnp.array(Xte_z)

GT = [0.25, 0.5, 0.75, 1.0]; NOBS = [2, 8, 32, 128]; TARGET = 0; K = 3; NMC = 2000; REPEATS = 5
def H_gt(p):  # ground-truth aleatoric entropy of a class-0 stimulus at fuzziness p
    return float(entropy([p, (1 - p) / 2, (1 - p) / 2]))

def noisy_labels(p, seed):
    yn = ytr.copy(); rng = np.random.RandomState(seed)
    pos = np.where(ytr == TARGET)[0]; nf = int(len(pos) * (1 - p))
    if nf:
        fi = rng.choice(pos, nf, replace=False); yn[fi] = rng.choice([1, 2], size=nf)
    return yn, rng

rows = []  # one row per (method, repeat, p, n_obs, test-query)
for rep in range(REPEATS):
    for p in GT:
        yn, rng = noisy_labels(p, seed=1000 * rep + int(p * 100))
        for n in NOBS:
            io = rng.choice(len(Xtr_raw), n, replace=False)
            yo = yn[io]
            # GPP-Dirichlet (cosine kernel, raw embeddings)
            try:
                g = ppm.gpp_multiclass(Xte_raw_j, jnp.array(Xtr_raw[io]),
                                       jax.nn.one_hot(yo, K), num_classes=K, n=NMC)
                gj = np.array(g['categorical_mu'])[:, TARGET]
                ga = np.array(g['Alea']).flatten(); gmi = np.array(g['information_gain']).flatten()
                ge = np.array(g['Episteme']).flatten()
            except Exception as e:
                print('GPP fail', p, n, e); gj = ga = gmi = ge = np.full(len(yte), np.nan)
            # LPE (z-scored embeddings)
            try:
                if len(np.unique(yo)) < 2:
                    raise ValueError('need >=2 classes')
                l = ppm.lpe_multiclass(Xte_z_j, jnp.array(Xtr_z[io]), yo, num_classes=K, repeats=50)
                lj = np.array(l['categorical_mu'])[:, TARGET]
                la = np.array(l['Alea']).flatten(); lmi = np.array(l['information_gain']).flatten()
                le = np.array(l['Episteme']).flatten()
            except Exception as e:
                lj = la = lmi = le = np.full(len(yte), np.nan)
            # Matched conditions: keep this (rep,p,n) cell only if BOTH methods produced
            # output, so GPP vs LPE summaries are never over different subsets of draws.
            if np.all(np.isnan(gj)) or np.all(np.isnan(lj)):
                continue
            hg = H_gt(p)
            for i in range(len(yte)):
                base = dict(rep=rep, p=p, n_obs=n, is_target=int(yte[i] == TARGET), H_gt=hg)
                rows.append(dict(method='GPP', judged=gj[i], alea=ga[i], mi=gmi[i], epi=ge[i], **base))
                rows.append(dict(method='LPE', judged=lj[i], alea=la[i], mi=lmi[i], epi=le[i], **base))
        print(f'  rep{rep} p={p} done', flush=True)

import pandas as pd
df = pd.DataFrame(rows)
outdir = f'{REPO}/experiments/shapes3d/figures/decomposition'
os.makedirs(outdir, exist_ok=True)
df.to_csv(f'{outdir}/decomposition_raw.csv', index=False)

# ================= 1.1 aleatoric tracks fuzziness (n=128, class-0 queries) =================
print("\n===== 1.1 Aleatoric tracks injected fuzziness (n=128, class-0 origin) =====")
res11 = {}
hi = df[(df.n_obs == 128) & (df.is_target == 1)]
for meth in ['GPP', 'LPE']:
    s = hi[hi.method == meth].dropna(subset=['alea'])
    r, _ = pearsonr(s.H_gt, s.alea) if len(s) > 2 and s.alea.std() > 1e-9 else (float('nan'), 0)
    res11[meth] = r
    means = s.groupby('p').alea.mean().to_dict()
    print(f"  {meth}: Pearson(H_gt, alea) = {r:.3f}   mean alea by p {{p: {{:.3f}}}}: "
          + ", ".join(f"{k}:{v:.3f}" for k, v in sorted(means.items(), reverse=True)))
print(f"  -> H_gt by p: " + ", ".join(f"{p}:{H_gt(p):.3f}" for p in sorted(GT, reverse=True)))

# ================= 1.2 epistemic (MI) tracks scarcity (p=1.0, clean) =================
print("\n===== 1.2 Epistemic MI decreases with #observations (clean labels p=1.0) =====")
res12 = {}
clean = df[(df.p == 1.0)]
for meth in ['GPP', 'LPE']:
    s = clean[clean.method == meth].dropna(subset=['mi'])
    curve = s.groupby('n_obs').mi.mean()
    rho, _ = spearmanr(s.n_obs, s.mi) if len(s) > 2 else (float('nan'), 0)
    res12[meth] = rho
    print(f"  {meth}: Spearman(n_obs, MI) = {rho:.3f}   mean MI by n_obs: "
          + ", ".join(f"{int(k)}:{v:.4f}" for k, v in curve.items()))

# ================= 1.3 not confidently ignorant (p=0.5, low data n=2) =================
print("\n===== 1.3 'Not confidently ignorant': judged P(class0) under low episteme (p=0.5, n=2) =====")
res13 = {}
lo = df[(df.p == 0.5) & (df.n_obs == 2)]
prior = 1.0 / K
for meth in ['GPP', 'LPE']:
    s = lo[lo.method == meth].dropna(subset=['judged'])
    # restrict to the lowest-episteme half (least knowledge)
    thr = s.epi.median()
    low_epi = s[s.epi <= thr]
    dist_to_prior = float(np.mean(np.abs(low_epi.judged - prior)))
    # Boundary-symmetric: 'extreme' = within 0.1 of either certainty boundary (0 or 1),
    # so the confident-yes and confident-no sides are treated equally around the 1/K prior.
    frac_extreme = float(np.mean((low_epi.judged > 0.9) | (low_epi.judged < 0.1)))
    res13[meth] = (dist_to_prior, frac_extreme)
    print(f"  {meth}: mean|judged - 1/K| = {dist_to_prior:.3f}   frac extreme(>0.9 or <0.1) = {frac_extreme:.3f}")
print(f"  (prior 1/K = {prior:.3f}; lower mean-dist and lower frac-extreme = more rational)")

# ================= figure =================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
# (a) alea vs H_gt
for meth, c in [('GPP', 'C0'), ('LPE', 'C1')]:
    s = hi[hi.method == meth].dropna(subset=['alea'])
    g = s.groupby('p').alea.agg(['mean', 'std'])
    hgs = [H_gt(p) for p in g.index]
    ax[0].errorbar(hgs, g['mean'], yerr=g['std'], marker='o', capsize=3,
                   label=f"{meth} (r={res11[meth]:.2f})", color=c)
lim = [0, max(H_gt(p) for p in GT) * 1.05]
ax[0].plot(lim, lim, 'k--', alpha=0.4, label='identity')
ax[0].set_xlabel('ground-truth aleatoric  H_gt(p)'); ax[0].set_ylabel('predicted aleatoric  E[H(p)]')
ax[0].set_title('1.1  Aleatoric tracks fuzziness (n=128)'); ax[0].legend()
# (b) MI vs n_obs
for meth, c in [('GPP', 'C0'), ('LPE', 'C1')]:
    s = clean[clean.method == meth].dropna(subset=['mi'])
    g = s.groupby('n_obs').mi.agg(['mean', 'std'])
    ax[1].errorbar(g.index, g['mean'], yerr=g['std'], marker='o', capsize=3,
                   label=f"{meth} (rho={res12[meth]:.2f})", color=c)
ax[1].set_xscale('log'); ax[1].set_xlabel('# observations'); ax[1].set_ylabel('epistemic MI  I(y;p)')
ax[1].set_title('1.2  Epistemic MI decreases with data (clean)'); ax[1].legend()
# (c) judged vs episteme scatter
for meth, c in [('GPP', 'C0'), ('LPE', 'C1')]:
    s = lo[lo.method == meth].dropna(subset=['judged', 'epi'])
    if len(s) > 1500: s = s.sample(1500, random_state=0)
    ax[2].scatter(s.epi, s.judged, s=5, alpha=0.25, color=c, label=meth)
ax[2].axhline(prior, ls='--', color='k', alpha=0.5, label='prior 1/K')
ax[2].set_xlabel('episteme (confidence)'); ax[2].set_ylabel('judged P(class 0)')
ax[2].set_title('1.3  Not confidently ignorant (p=0.5, n=2)'); ax[2].legend()
plt.suptitle('Multiclass aleatoric/epistemic decomposition validation (3D-Shapes M1, K=3)', fontsize=13)
plt.tight_layout()
fig.savefig(f'{outdir}/decomposition_validation.png', dpi=200)
print(f"\nSaved figure -> {outdir}/decomposition_validation.png")
print(f"Saved raw    -> {outdir}/decomposition_raw.csv")

# ================= machine-readable summary =================
summary = {
    '1.1_pearson_Hgt_alea': res11,
    '1.2_spearman_nobs_mi': res12,
    '1.3_lowepi_meandist_to_prior_and_fracextreme': {k: list(v) for k, v in res13.items()},
}
import json
open(f'{outdir}/decomposition_summary.json', 'w').write(json.dumps(summary, indent=2))
print("SUMMARY:", json.dumps(summary))
