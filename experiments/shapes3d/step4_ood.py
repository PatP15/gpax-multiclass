"""WS2: Multiclass OOD detection (paper Fig 7 analog, K=3).

The paper validates GPP as an OOD detector using the negative latent posterior
variance as an episteme proxy (§4.4), competitive with Maha / MSP / LPE. There is
no multiclass OOD experiment in the repo; this adds one on 3D-Shapes M1, K=3.

Two regimes:
  A) near-OOD (novel class): train the probe on shapes {0,1,2}; the held-out 4th
     shape (label 3) is OOD. Pure relabeling of existing embeddings.
  B) far-OOD (paper protocol): ID = shape {0,1,2} queries; OOD = pixel-wise
     uniform-noise images pushed through the trained M1 CNN.

Scores (higher = in-distribution):
  GPP-Dirichlet : negative summed latent posterior variance (paper proxy);
                  also reports Episteme (neg softmax-marginal entropy).
  Maha          : negative min Mahalanobis distance (maha_multiclass).
  MSP           : max softmax prob of a logistic probe (lp_maxprob_multiclass).
  LPE           : negative summed ensemble predictive variance.
Metric: AUROC(ID vs OOD) vs #observations, averaged over repeats.

Run on the cluster (needs embeddings_M1.npy, the M1 checkpoint and 3dshapes.h5
for regime B). CPU only.
"""
import os, sys
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
sys.path.insert(0, f'{REPO}/experiments/shapes3d')
import numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from GPax.probing import probabilistic_probe_multiclass as ppm
from GPax.probing import gp_multiclass as gpm

NOBS = [8, 16, 32, 64, 128]; K = 3; NMC = 2000; REPEATS = 5; N_ID = 800; N_OOD = 800

emb = np.load(f'{REPO}/results/embeddings/embeddings_M1.npy')
d = np.load(f'{REPO}/results/embeddings/data_labels.npz', allow_pickle=True)
shape = d['P2_shape']; L = min(len(emb), len(shape)); emb, shape = emb[:L], shape[:L].astype(int)
in_mask = shape < 3
X_in, y_in = emb[in_mask], shape[in_mask]                 # shapes 0,1,2
X_shape3 = emb[shape == 3]                                 # held-out 4th shape (near-OOD)
Xtr_pool, Xte_in, ytr_pool, _ = train_test_split(X_in, y_in, test_size=0.3, random_state=42, stratify=y_in)


# ID statistics for standardizing the (scale-sensitive) baselines; GPP-cosine uses raw.
mu_pool, sd_pool = Xtr_pool.mean(0), Xtr_pool.std(0) + 1e-8


def restore_m1_state():
    """Restore the trained M1 CNN (its final Dense is 64-way, M1's training label set)."""
    import cnn_model
    from flax.training import checkpoints
    nc = int(d['M1'].max()) + 1
    state = cnn_model.create_train_state(jax.random.PRNGKey(0), num_classes=nc)
    state = checkpoints.restore_checkpoint(
        ckpt_dir=os.path.abspath(f'{REPO}/results/embeddings'), target=state, prefix='checkpoint_M1_')
    return cnn_model, state


def embed_noise(cnn_model, state, n_imgs, seed):
    noise = np.random.RandomState(seed).randint(0, 256, size=(n_imgs, 64, 64, 3), dtype=np.uint8)
    embs = [np.array(cnn_model.get_embeddings(state, jnp.array(noise[i:i + 2048])))
            for i in range(0, n_imgs, 2048)]
    return np.concatenate(embs, 0)


def gpp_scores(Xq, Xo, yo):
    g = ppm.gpp_multiclass(jnp.array(Xq), jnp.array(Xo), jax.nn.one_hot(yo, K), num_classes=K, n=NMC)
    return {
        'GPP (neg latent var)': np.array(gpm.dirichlet_gp_ood_score(g)),  # paper §4.4 proxy
        'GPP (Episteme)': np.array(g['Episteme']).flatten(),
    }


def baseline_scores(Xq, Xo, yo):
    # Logistic/distance baselines run on ID-standardized inputs (fit on the train pool),
    # matching step4_kscaling / step4_decomposition; only GPP-cosine uses raw embeddings.
    Xq_z = jnp.array((Xq - mu_pool) / sd_pool); Xo_z = jnp.array((Xo - mu_pool) / sd_pool)
    out = {}
    try:
        out['Maha'] = np.array(ppm.maha_multiclass(Xq_z, Xo_z, yo, num_classes=K)['episteme'])
    except Exception as e:
        out['Maha'] = np.full(len(Xq), np.nan); print('maha fail', e)
    try:
        out['MSP'] = np.array(ppm.lp_maxprob_multiclass(Xq_z, Xo_z, yo)['episteme'])
    except Exception as e:
        out['MSP'] = np.full(len(Xq), np.nan); print('msp fail', e)
    try:
        l = ppm.lpe_multiclass(Xq_z, Xo_z, yo, num_classes=K, repeats=50)   # match calibration study
        out['LPE'] = -np.sum(np.array(l['epistemic_var']), axis=1)
    except Exception as e:
        out['LPE'] = np.full(len(Xq), np.nan); print('lpe fail', e)
    return out


def run_regime(name, ood_sampler):
    """ood_sampler(rep, rng) -> OOD query embeddings for that rep (fresh/decorrelated)."""
    print(f"\n===== OOD regime: {name} =====", flush=True)
    rows = []
    for rep in range(REPEATS):
        rng = np.random.RandomState(100 + rep)
        id_q = Xte_in[rng.choice(len(Xte_in), N_ID, replace=False)]
        ood_q = ood_sampler(rep, rng)
        Xq = np.vstack([id_q, ood_q])
        is_ood = np.r_[np.zeros(N_ID), np.ones(len(ood_q))]      # 1 = OOD
        for n in NOBS:
            io = rng.choice(len(Xtr_pool), n, replace=False)
            Xo, yo = Xtr_pool[io], ytr_pool[io]
            if len(np.unique(yo)) < 2:
                continue
            scores = {}
            scores.update(gpp_scores(Xq, Xo, yo))
            scores.update(baseline_scores(Xq, Xo, yo))
            for method, s in scores.items():
                auroc = (roc_auc_score(is_ood, -s)               # higher score = ID -> OOD low
                         if np.all(np.isfinite(s)) and np.std(s) > 1e-12 else np.nan)
                rows.append(dict(regime=name, method=method, rep=rep, n_obs=n, auroc=auroc))
        print(f"  rep{rep} done", flush=True)
    return rows


all_rows = []
# near-OOD: held-out 4th shape — large pool, independent per-rep draws.
all_rows += run_regime('near-OOD (held-out shape 3)',
                       lambda rep, rng: X_shape3[rng.choice(len(X_shape3), N_OOD, replace=False)])
# far-OOD: FRESH uniform-noise images per rep (decorrelated repeats).
try:
    _cnn, _m1 = restore_m1_state()
    all_rows += run_regime('far-OOD (uniform noise)',
                           lambda rep, rng: embed_noise(_cnn, _m1, N_OOD, seed=1000 + rep))
except Exception as e:
    import traceback; traceback.print_exc()
    print('Far-OOD skipped (CNN restore failed):', e)

import pandas as pd
df = pd.DataFrame(all_rows)
outdir = f'{REPO}/experiments/shapes3d/figures/ood'
os.makedirs(outdir, exist_ok=True)
df.to_csv(f'{outdir}/ood_raw.csv', index=False)

# summary table (mean AUROC over reps, at each n_obs)
print("\n===== AUROC(ID/OOD) mean over reps =====")
methods = ['GPP (neg latent var)', 'GPP (Episteme)', 'Maha', 'MSP', 'LPE']
regimes = df.regime.unique()
summary = {}
for reg in regimes:
    print(f"\n[{reg}]")
    sub = df[df.regime == reg]
    piv = sub.groupby(['method', 'n_obs']).auroc.mean().unstack()
    print(piv.round(3).to_string())
    summary[reg] = {m: float(sub[sub.method == m].auroc.mean()) for m in methods if m in sub.method.values}

# figure: AUROC vs n_obs, one panel per regime
fig, axes = plt.subplots(1, len(regimes), figsize=(7 * len(regimes), 5), squeeze=False)
for j, reg in enumerate(regimes):
    ax = axes[0][j]; sub = df[df.regime == reg]
    for method in methods:
        s = sub[sub.method == method]
        if s.empty: continue
        g = s.groupby('n_obs').auroc.agg(['mean', 'std'])
        ax.errorbar(g.index, g['mean'], yerr=g['std'], marker='o', capsize=3, label=method)
    ax.axhline(0.5, ls='--', color='k', alpha=0.4)
    ax.set_xscale('log'); ax.set_xlabel('# observations'); ax.set_ylabel('AUROC (ID vs OOD)')
    ax.set_ylim(0.4, 1.02); ax.set_title(reg); ax.legend(fontsize=8)
plt.suptitle('Multiclass OOD detection (3D-Shapes M1, K=3)', fontsize=13)
plt.tight_layout()
fig.savefig(f'{outdir}/ood_detection.png', dpi=200)
print(f"\nSaved figure -> {outdir}/ood_detection.png")
import json
open(f'{outdir}/ood_summary.json', 'w').write(json.dumps(summary, indent=2))
print("SUMMARY (mean AUROC over all n_obs):", json.dumps(summary))
