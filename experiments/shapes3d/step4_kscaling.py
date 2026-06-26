"""WS3: K-scaling — does multiclass GPP degrade as the number of classes grows?

Every multiclass experiment in the repo fixes K=3. Here we sweep K on a single
clean label family (3D-Shapes *object hue*, a native 10-way primitive that M1's
color-trained embeddings represent well) by grouping the 10 hues into K contiguous
bins: K in {2,4,5,10}. For each K we report, GPP-Dirichlet vs LPE:
  - accuracy (argmax of the judged probabilities)
  - multiclass Brier score (lower = better calibrated probabilities)
  - ECE (expected calibration error, max-prob binning)
  - decomposition sanity: mean epistemic MI must still decrease with #observations.

Deliverable: metric-vs-K curves showing GPP does not collapse as K grows.
Run on the cluster (needs embeddings_M1.npy). CPU only.
"""
import os, sys
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from GPax.probing import probabilistic_probe_multiclass as ppm

K_LIST = [2, 4, 5, 10]; NOBS = [32, 128, 512]; REPEATS = 5; NMC = 2000; N_TEST = 2000

emb = np.load(f'{REPO}/results/embeddings/embeddings_M1.npy')
d = np.load(f'{REPO}/results/embeddings/data_labels.npz', allow_pickle=True)
# labels_raw stores the FLOAT factor values; object hue (col 2) takes 10 distinct
# values. Map them to integer indices 0..9 (astype(int) would collapse to 0).
hue_raw = d['labels_raw'][:, 2]
hue = np.searchsorted(np.unique(hue_raw), hue_raw).astype(int)
L = min(len(emb), len(hue)); emb, hue = emb[:L], hue[:L]


def k_labels(hue, K):
    return np.minimum((hue * K) // 10, K - 1)       # group 10 hues into K contiguous bins


def multiclass_ece(probs, y_true, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier(probs, y_true, K):
    onehot = np.eye(K)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


rows = []
for K in K_LIST:
    y = k_labels(hue, K)
    Xtr_pool, Xte, ytr_pool, yte = train_test_split(emb, y, test_size=0.3, random_state=42, stratify=y)
    ridx = np.random.RandomState(0).choice(len(Xte), N_TEST, replace=False)
    Xte_s, yte_s = Xte[ridx], yte[ridx]; Xte_j = jnp.array(Xte_s)
    mu_, sd_ = Xtr_pool.mean(0), Xtr_pool.std(0) + 1e-8   # standardize LPE inputs (GPP stays raw)
    Xte_z = jnp.array((Xte_s - mu_) / sd_)
    print(f"\n===== K={K}  (class counts: {np.bincount(ytr_pool)}) =====", flush=True)
    for rep in range(REPEATS):
        rng = np.random.RandomState(200 + rep)
        for n in NOBS:
            io = rng.choice(len(Xtr_pool), n, replace=False)
            Xo, yo = Xtr_pool[io], ytr_pool[io]
            if len(np.unique(yo)) < 2:
                continue
            # GPP-Dirichlet
            try:
                g = ppm.gpp_multiclass(Xte_j, jnp.array(Xo), jax.nn.one_hot(yo, K), num_classes=K, n=NMC)
                gp_p = np.array(g['categorical_mu'])
                rows.append(dict(K=K, rep=rep, n_obs=n, method='GPP',
                                 acc=float((gp_p.argmax(1) == yte_s).mean()),
                                 brier=brier(gp_p, yte_s, K), ece=multiclass_ece(gp_p, yte_s),
                                 mi=float(np.mean(np.array(g['information_gain'])))))
            except Exception as e:
                print('GPP fail', K, n, e)
            # LPE
            try:
                l = ppm.lpe_multiclass(Xte_z, jnp.array((Xo - mu_) / sd_), yo, num_classes=K, repeats=10)
                lp_p = np.array(l['categorical_mu'])
                rows.append(dict(K=K, rep=rep, n_obs=n, method='LPE',
                                 acc=float((lp_p.argmax(1) == yte_s).mean()),
                                 brier=brier(lp_p, yte_s, K), ece=multiclass_ece(lp_p, yte_s),
                                 mi=float(np.mean(np.array(l['information_gain'])))))
            except Exception as e:
                pass
        print(f"  rep{rep} done", flush=True)

import pandas as pd
df = pd.DataFrame(rows)
outdir = f'{REPO}/experiments/shapes3d/figures/kscaling'
os.makedirs(outdir, exist_ok=True)
df.to_csv(f'{outdir}/kscaling_raw.csv', index=False)

# decomposition sanity: MI must decrease with n_obs at every K (GPP)
print("\n===== Decomposition sanity: Spearman(n_obs, MI) per K (GPP, expect < 0) =====")
mono = {}
for K in K_LIST:
    s = df[(df.K == K) & (df.method == 'GPP')]
    rho, _ = spearmanr(s.n_obs, s.mi) if len(s) > 2 else (float('nan'), 0)
    mono[K] = float(rho); print(f"  K={K}: Spearman(n_obs, MI) = {rho:.3f}")

# metric-vs-K table + curves at the largest n_obs
big = df[df.n_obs == max(NOBS)]
print(f"\n===== Metrics vs K at n_obs={max(NOBS)} (mean over reps) =====")
for metric in ['acc', 'brier', 'ece']:
    piv = big.groupby(['method', 'K'])[metric].mean().unstack()
    print(f"\n[{metric}]"); print(piv.round(3).to_string())

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for j, (metric, lab, better) in enumerate([('acc', 'accuracy', 'higher'),
                                           ('brier', 'Brier score', 'lower'),
                                           ('ece', 'ECE', 'lower')]):
    for meth, c in [('GPP', 'C0'), ('LPE', 'C1')]:
        s = big[big.method == meth]
        g = s.groupby('K')[metric].agg(['mean', 'std'])
        ax[j].errorbar(g.index, g['mean'], yerr=g['std'], marker='o', capsize=3, label=meth, color=c)
    ax[j].set_xlabel('number of classes K'); ax[j].set_ylabel(f'{lab} ({better} better)')
    ax[j].set_title(f'{lab} vs K  (n_obs={max(NOBS)})'); ax[j].set_xticks(K_LIST); ax[j].legend()
plt.suptitle('Multiclass GPP K-scaling (3D-Shapes M1, object hue)', fontsize=13)
plt.tight_layout()
fig.savefig(f'{outdir}/kscaling.png', dpi=200)
print(f"\nSaved figure -> {outdir}/kscaling.png")

import json
summary = {'mi_monotonicity_spearman_by_K': mono,
           'metrics_vs_K_at_max_nobs': {
               m: big.groupby(['method', 'K'])[m].mean().unstack().round(4).to_dict()
               for m in ['acc', 'brier', 'ece']}}
open(f'{outdir}/kscaling_summary.json', 'w').write(json.dumps(summary, indent=2, default=str))
print("SUMMARY:", json.dumps(summary, default=str))
