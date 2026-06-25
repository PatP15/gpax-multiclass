import os, sys
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
import jax
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from GPax.probing import probabilistic_probe_multiclass as ppm

# --- load cached M1 embeddings + 3-class shape labels (same as run_fuzziness_experiment) ---
emb = np.load(f'{REPO}/results/embeddings/embeddings_M1.npy')
d = np.load(f'{REPO}/results/embeddings/data_labels.npz', allow_pickle=True)
y = d['P2_shape']; L = min(len(emb), len(y)); emb, y = emb[:L], y[:L]
m = y < 3; X, y = emb[m], y[m].astype(int)
Xtr_pool, Xte, ytr_pool, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# subsample test for sweep speed
if len(Xte) > 1500:
    idx = np.random.RandomState(0).choice(len(Xte), 1500, replace=False); Xte, yte = Xte[idx], yte[idx]
GT = [0.25, 0.5, 0.75, 1.0]; NOBS = [2, 8, 32, 128]; TARGET = 0
print(f"train_pool={len(Xtr_pool)} test={len(Xte)}  3-class shape; fuzzify class {TARGET}")

def make_noisy(p):
    yn = ytr_pool.copy(); rng = np.random.RandomState(42 + int(p*100))
    pos = np.where(ytr_pool == TARGET)[0]; nf = int(len(pos)*(1-p))
    if nf > 0:
        fi = rng.choice(pos, nf, replace=False); yn[fi] = rng.choice([1, 2], size=nf)
    return yn, rng

def per_nobs_pearson(judged_fn):
    """judged_fn(X_obs, y_obs) -> P(class0) over Xte. Returns {n_obs: pearson over pooled gt levels}."""
    out = {}
    for n in NOBS:
        gts, jds = [], []
        for p in GT:
            yn, rng = make_noisy(p)
            io = rng.choice(len(Xtr_pool), n, replace=False)
            Xo, yo = Xtr_pool[io], yn[io]
            gt = np.where(yte == TARGET, p, 0.0)
            try:
                jd = judged_fn(Xo, yo)
            except Exception:
                continue
            gts.append(gt); jds.append(np.asarray(jd))
        gts = np.concatenate(gts); jds = np.concatenate(jds)
        out[n] = pearsonr(gts, jds)[0] if np.std(jds) > 1e-9 else float('nan')
    return out

import jax.numpy as jnp
def gpp_judged(S, eps):
    def f(Xo, yo):
        yoh = jax.nn.one_hot(yo, 3)
        u = ppm.gpp_multiclass(jnp.array(Xte), jnp.array(Xo), yoh, num_classes=3,
                               alpha_eps=eps, strength=S, n=1000)
        return np.array(u['categorical_mu'])[:, TARGET]
    return f
def lpe_judged(Xo, yo):
    u = ppm.lpe_multiclass(jnp.array(Xte), jnp.array(Xo), yo, num_classes=3, repeats=10)
    return np.array(u['categorical_mu'])[:, TARGET]

print("\n=== LPE baseline ===")
rl = per_nobs_pearson(lpe_judged)
print("  LPE per-n_obs:", {k: round(v,3) for k,v in rl.items()}, "mean", round(np.nanmean(list(rl.values())),3))

print("\n=== GPP sweep (strength S, alpha_eps eps) — current default is S=5, eps=0.1 ===")
print(f"{'S':>5} {'eps':>5} | {'n2':>6} {'n8':>6} {'n32':>6} {'n128':>6} | {'mean':>6}")
best=None
for S in [1.0, 2.0, 5.0, 10.0]:
    for eps in [0.1, 0.5, 1.0]:
        r = per_nobs_pearson(gpp_judged(S, eps))
        mean = np.nanmean(list(r.values()))
        flag = " <-default" if (S==5.0 and eps==0.1) else ""
        print(f"{S:5.1f} {eps:5.1f} | {r.get(2,float('nan')):6.3f} {r.get(8,float('nan')):6.3f} {r.get(32,float('nan')):6.3f} {r.get(128,float('nan')):6.3f} | {mean:6.3f}{flag}")
        if best is None or mean > best[0]: best=(mean,S,eps,r)
print(f"\nBEST GPP: mean r={best[0]:.3f} at S={best[1]}, eps={best[2]}  per-n_obs={ {k:round(v,3) for k,v in best[3].items()} }")
print(f"LPE mean r={np.nanmean(list(rl.values())):.3f}")
print("RESCUED" if best[0] >= np.nanmean(list(rl.values())) else "NOT rescued by S/eps alone")
