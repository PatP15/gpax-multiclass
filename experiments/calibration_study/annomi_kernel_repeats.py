"""WS4 + WS5.3: AnnoMI probing with repeats (error bars) and the productized kernel.

Two goals in one run, on the committed AnnoMI 'context' (client motivation, K=3)
embeddings for every available model:

  WS4  - replace the single-seed AnnoMI probing with >=5 seeds and report
         mean +/- std (the experiments previously had no error bars).
  WS5.3 - validate the productized kernel fix (gpp_multiclass_select: standardize
         + marginal-likelihood-selected lengthscale) on REAL LLM embeddings, where
         the calibration study so far only had cosine. Compares, per model:
         GPP-cosine vs GPP-rbf(auto) vs GPP-laplace(auto) vs LPE.

Metrics per (model, method, n): accuracy, ECE (max-prob binning), Brier — each
mean+/-std over seeds. Runs locally on the committed .npz (no GPU/cluster).
"""
import os, sys, json
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from GPax.probing import probabilistic_probe_multiclass as ppm

MODELS = [m for m in ['gemma', 'qwen', 'gemma4', 'qwen36']
          if os.path.exists(f'{REPO}/experiments/annomi/data/{m}/embeddings.npz')]
NOBS = [100, 500, 1500, 2400]; SEEDS = list(range(5)); K = 3; NMC = 1500
METHODS = ['GPP-cosine', 'GPP-rbf', 'GPP-laplace', 'LPE']


def balance(X, y, seed):
    rng = np.random.default_rng(seed); cls, cnt = np.unique(y, return_counts=True); mn = cnt.min()
    idx = np.concatenate([rng.choice(np.where(y == c)[0], mn, replace=False) for c in cls])
    rng.shuffle(idx); return X[idx], y[idx]


def ece(probs, y, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1); corr = (pred == y).astype(float)
    b = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def brier(probs, y):
    return float(np.mean(np.sum((probs - np.eye(K)[y]) ** 2, axis=1)))


def predict(method, Xtr, ytr, Xte):
    if method == 'GPP-cosine':
        m = ppm.gpp_multiclass(jnp.array(Xte), jnp.array(Xtr), jax.nn.one_hot(ytr, K),
                               num_classes=K, n=NMC)
        return np.array(m['categorical_mu'])
    if method == 'GPP-rbf':
        m = ppm.gpp_multiclass_select(Xte, Xtr, ytr, num_classes=K, kernel='rbf',
                                      lengthscale='auto', standardize=True, n=NMC)
        return np.array(m['categorical_mu'])
    if method == 'GPP-laplace':
        m = ppm.gpp_multiclass_select(Xte, Xtr, ytr, num_classes=K, kernel='laplace',
                                      lengthscale='auto', standardize=True, n=NMC)
        return np.array(m['categorical_mu'])
    if method == 'LPE':
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        m = ppm.lpe_multiclass(jnp.array((Xte - mu) / sd), jnp.array((Xtr - mu) / sd), ytr,
                               num_classes=K, repeats=50)
        return np.array(m['categorical_mu'])


rows = []
for model in MODELS:
    d = np.load(f'{REPO}/experiments/annomi/data/{model}/embeddings.npz', allow_pickle=True)
    Xtr_full, ytr_full = d['X_train_context'], d['y_train_mot']
    Xte, yte = np.array(d['X_test_context']), np.array(d['y_test_mot'])
    print(f"\n===== {model}: train {Xtr_full.shape} test {Xte.shape} =====", flush=True)
    for seed in SEEDS:
        Xb, yb = balance(Xtr_full, ytr_full, seed)
        for n in NOBS:
            if n > len(Xb): continue
            try:
                Xtr, _, ytr, _ = train_test_split(Xb, yb, train_size=n, random_state=seed, stratify=yb)
            except ValueError:
                Xtr, ytr = Xb[:n], yb[:n]
            for method in METHODS:
                try:
                    p = predict(method, Xtr, ytr, Xte)
                    rows.append(dict(model=model, method=method, n=n, seed=seed,
                                     acc=float((p.argmax(1) == yte).mean()),
                                     ece=ece(p, yte), brier=brier(p, yte)))
                except Exception as ex:
                    print(f"  {model} {method} n={n} seed={seed} FAIL: {ex}")
        print(f"  seed {seed} done", flush=True)

import pandas as pd
df = pd.DataFrame(rows)
outdir = f'{REPO}/experiments/calibration_study/annomi_kernel'
os.makedirs(outdir, exist_ok=True)
df.to_csv(f'{outdir}/annomi_kernel_repeats_raw.csv', index=False)

# summary at n=2400 (mean +/- std over seeds)
print("\n===== n=2400 mean+/-std over seeds (accuracy / ECE) =====")
big = df[df.n == 2400]
summary = {}
for model in MODELS:
    summary[model] = {}
    print(f"\n[{model}]")
    for method in METHODS:
        s = big[(big.model == model) & (big.method == method)]
        if s.empty: continue
        summary[model][method] = dict(acc=[float(s.acc.mean()), float(s.acc.std())],
                                      ece=[float(s.ece.mean()), float(s.ece.std())],
                                      brier=[float(s.brier.mean()), float(s.brier.std())])
        print(f"  {method:12s} acc={s.acc.mean():.3f}+/-{s.acc.std():.3f}  "
              f"ECE={s.ece.mean():.3f}+/-{s.ece.std():.3f}  Brier={s.brier.mean():.3f}")

# figure: accuracy + ECE vs n, per model, methods overlaid with CI bands
nM = len(MODELS)
fig, axes = plt.subplots(2, nM, figsize=(5 * nM, 9), squeeze=False)
colors = {'GPP-cosine': 'C0', 'GPP-rbf': 'C2', 'GPP-laplace': 'C3', 'LPE': 'C1'}
for j, model in enumerate(MODELS):
    for mi, metric in enumerate(['acc', 'ece']):
        ax = axes[mi][j]
        for method in METHODS:
            s = df[(df.model == model) & (df.method == method)]
            if s.empty: continue
            g = s.groupby('n')[metric].agg(['mean', 'std'])
            ax.plot(g.index, g['mean'], marker='o', color=colors[method], label=method)
            ax.fill_between(g.index, g['mean'] - g['std'], g['mean'] + g['std'],
                            color=colors[method], alpha=0.15)
        ax.set_xlabel('# observations'); ax.set_ylabel('accuracy' if metric == 'acc' else 'ECE')
        ax.set_title(f"{model} — {'accuracy' if metric=='acc' else 'ECE (lower=better)'}")
        if j == 0 and mi == 0: ax.legend(fontsize=8)
plt.suptitle('AnnoMI motivation (K=3): kernel comparison with 5-seed error bars', fontsize=14)
plt.tight_layout()
fig.savefig(f'{outdir}/annomi_kernel_repeats.png', dpi=180)
print(f"\nSaved figure -> {outdir}/annomi_kernel_repeats.png")
open(f'{outdir}/annomi_kernel_summary.json', 'w').write(json.dumps(summary, indent=2))
print("SUMMARY:", json.dumps(summary))
