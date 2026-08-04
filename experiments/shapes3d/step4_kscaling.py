"""WS3: K-scaling — does multiclass GPP degrade as the number of classes grows?

Every multiclass experiment in the repo fixes K=3. Here we sweep K using targets the
M1 embedding actually encodes (M1 was trained on the 64-way product of binarized
scale/floor/wall/object-color and the 4 shapes), via balanced composite labels:
K=2 floor warm/cool, K=4 shape, K=8 shape×scale, K=16 shape×scale×floor. This avoids
the confound of probing features M1 never learned. For each K we report, GPP-Dirichlet vs LPE:
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
import sklearn.linear_model as sklm, sklearn.svm as sksvm, sklearn.neural_network as skmlp
from GPax.probing import probabilistic_probe_multiclass as ppm


def _proba_padded(clf, Xtr_z, ytr, Xte_z, K):
    """Fit a sklearn classifier and return (n, K) probabilities, padding any class
    absent from the observed sample with zero-probability columns."""
    clf.fit(Xtr_z, ytr)
    p = clf.predict_proba(Xte_z)
    if p.shape[1] == K:
        return p
    # Scatter present-class columns to absolute indices; valid because k_labels are
    # contiguous 0..K-1 (guard so a future non-contiguous mapping fails loudly).
    cls = clf.classes_.astype(int)
    assert cls.min() >= 0 and cls.max() < K, f"non-contiguous class labels {cls} for K={K}"
    full = np.zeros((len(Xte_z), K)); full[:, cls] = p
    return full

K_LIST = [2, 4, 8, 16]; NOBS = [32, 128, 512]; REPEATS = 5; NMC = 2000; N_TEST = 2000

emb = np.load(f'{REPO}/results/embeddings/embeddings_M1.npy')
d = np.load(f'{REPO}/results/embeddings/data_labels.npz', allow_pickle=True)
# K-scaling must use targets the M1 embedding actually encodes, else accuracy decay
# with K just reflects missing features, not a GPP property. M1 was trained on the
# 64-way product of {binarized scale, floor, wall, object color} x {4 shapes}, so we
# build balanced composite labels from those M1-supported factors:
#   K=2  floor warm/cool ; K=4  shape ; K=8  shape x scale ; K=16 shape x scale x floor.
floor = d['P1_floor'].astype(int); scale = d['P2_scale'].astype(int); shape = d['P2_shape'].astype(int)
L = min(len(emb), len(floor)); emb = emb[:L]; floor, scale, shape = floor[:L], scale[:L], shape[:L]


def k_labels(K):
    if K == 2:  return floor
    if K == 4:  return shape
    if K == 8:  return shape * 2 + scale
    if K == 16: return (shape * 2 + scale) * 2 + floor
    raise ValueError(K)


def multiclass_ece(probs, y_true, n_bins=15):   # 15 bins to match calibration_study/calib_analysis.py
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
    y = k_labels(K)
    Xtr_pool, Xte, ytr_pool, yte = train_test_split(emb, y, test_size=0.3, random_state=42, stratify=y)
    ridx = np.random.RandomState(0).choice(len(Xte), N_TEST, replace=False)
    Xte_s, yte_s = Xte[ridx], yte[ridx]; Xte_j = jnp.array(Xte_s)
    print(f"\n===== K={K}  (class counts: {np.bincount(ytr_pool)}) =====", flush=True)
    for rep in range(REPEATS):
        rng = np.random.RandomState(200 + rep)
        for n in NOBS:
            io = rng.choice(len(Xtr_pool), n, replace=False)
            Xo, yo = Xtr_pool[io], ytr_pool[io]
            if len(np.unique(yo)) < 2:
                continue
            # Single standardization reference for ALL standardized methods: the
            # observations themselves (the realistic few-shot reference, identical
            # across GPP-rbf/LPE/LP/SVM). GPP-cosine uses raw embeddings by design.
            mu_o, sd_o = Xo.mean(0), Xo.std(0) + 1e-8
            Xo_z = (Xo - mu_o) / sd_o; Xte_zo = (Xte_s - mu_o) / sd_o
            yoh = jax.nn.one_hot(yo, K)
            # Compute every method first; record the (K,rep,n) cell only if ALL
            # succeed, so method averages are never over mismatched cell counts.
            cell = {}
            try:
                g = ppm.gpp_multiclass(Xte_j, jnp.array(Xo), yoh, num_classes=K, n=NMC)
                cell['GPP-cosine'] = (np.array(g['categorical_mu']),
                                      float(np.mean(np.array(g['information_gain']))))
                gr = ppm.gpp_multiclass_select(Xte_zo, Xo_z, yo, num_classes=K, kernel='rbf',
                                               lengthscale='auto', standardize=False, n=NMC)
                cell['GPP-rbf'] = (np.array(gr['categorical_mu']),
                                   float(np.mean(np.array(gr['information_gain']))))
                l = ppm.lpe_multiclass(jnp.array(Xte_zo), jnp.array(Xo_z), yo, num_classes=K,
                                       repeats=50, rng=(K, rep, n))
                cell['LPE'] = (np.array(l['categorical_mu']),
                               float(np.mean(np.array(l['information_gain']))))
                # LP/SVM are linear, so GPP-rbf beating them would only show
                # nonlinear-vs-linear. SVM-rbf and the MLP are nonlinear baselines at
                # comparable capacity, which is the comparison a reviewer will demand.
                for mname, clf in [('LP', sklm.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)),
                                   ('SVM', sksvm.SVC(kernel='linear', probability=True)),
                                   ('SVM-rbf', sksvm.SVC(kernel='rbf', probability=True, random_state=rep)),
                                   ('MLP', skmlp.MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                                                               random_state=rep))]:
                    cell[mname] = (_proba_padded(clf, Xo_z, yo, Xte_zo, K), float('nan'))
            except Exception as e:
                print('kscaling cell dropped (matched conditions)', K, rep, n, repr(e)); continue
            for mname, (p, mi) in cell.items():
                rows.append(dict(K=K, rep=rep, n_obs=n, method=mname,
                                 acc=float((p.argmax(1) == yte_s).mean()),
                                 brier=brier(p, yte_s, K), ece=multiclass_ece(p, yte_s), mi=mi))
        print(f"  rep{rep} done", flush=True)

import pandas as pd
df = pd.DataFrame(rows)
outdir = f'{REPO}/experiments/shapes3d/figures/kscaling'
os.makedirs(outdir, exist_ok=True)
df.to_csv(f'{outdir}/kscaling_raw.csv', index=False)

# decomposition sanity: MI must decrease with n_obs at every K (GPP-cosine)
print("\n===== Decomposition sanity: Spearman(n_obs, MI) per K (GPP-cosine, expect < 0) =====")
mono = {}
for K in K_LIST:
    s = df[(df.K == K) & (df.method == 'GPP-cosine')]
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
    for meth, c in [('GPP-cosine', 'C0'), ('GPP-rbf', 'C2'), ('LPE', 'C1'), ('LP', 'C4'),
                    ('SVM', 'C5'), ('SVM-rbf', 'C6'), ('MLP', 'C3')]:
        s = big[big.method == meth]
        if s.empty: continue
        g = s.groupby('K')[metric].agg(['mean', 'std'])
        ax[j].errorbar(g.index, g['mean'], yerr=g['std'], marker='o', capsize=3, label=meth, color=c)
    ax[j].set_xlabel('number of classes K'); ax[j].set_ylabel(f'{lab} ({better} better)')
    ax[j].set_title(f'{lab} vs K  (n_obs={max(NOBS)})'); ax[j].set_xticks(K_LIST); ax[j].legend()
plt.suptitle('Multiclass GPP K-scaling (3D-Shapes M1; floor / shape / shape×scale / shape×scale×floor)', fontsize=12)
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
