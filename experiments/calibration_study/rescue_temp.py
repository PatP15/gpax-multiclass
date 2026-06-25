import os, sys
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, jax, jax.numpy as jnp
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from GPax.probing import gp_multiclass as gpm
from GPax.probing import probabilistic_probe_multiclass as ppm

emb = np.load(f'{REPO}/results/embeddings/embeddings_M1.npy')
d = np.load(f'{REPO}/results/embeddings/data_labels.npz', allow_pickle=True)
y = d['P2_shape']; L = min(len(emb), len(y)); emb, y = emb[:L], y[:L]
m = y < 3; X, y = emb[m], y[m].astype(int)
Xtr_pool, Xte, ytr_pool, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
idx = np.random.RandomState(0).choice(len(Xte), 1500, replace=False); Xte, yte = Xte[idx], yte[idx]
Xte_j = jnp.array(Xte)
GT = [0.25, 0.5, 0.75, 1.0]; NOBS = [2, 8, 32, 128]; TARGET = 0; TS = [1.0, 2.0, 3.0, 5.0, 8.0]

def noisy(p):
    yn = ytr_pool.copy(); rng = np.random.RandomState(42 + int(p*100))
    pos = np.where(ytr_pool == TARGET)[0]; nf = int(len(pos)*(1-p))
    if nf: fi = rng.choice(pos, nf, replace=False); yn[fi] = rng.choice([1,2], size=nf)
    return yn, rng

def gp_latents(Xo, yo):
    yoh = jax.nn.one_hot(yo, 3)
    preds = gpm.dirichlet_gp_predict(mean_func=gpm.constant_mean, cov_func=gpm.cosine_kernel,
            x_query=Xte_j, x_observed=jnp.array(Xo), y_observed=yoh,
            params={'alpha_eps':0.1,'strength':5.0}, var_only=True)
    mus, vars = gpm.get_latent_gp_dirichlet(preds)
    mu = jnp.hstack(mus); var = jnp.maximum(jnp.hstack(vars), 1e-32)
    return mu, var

def judged_at_T(mu, var, T, n=1000):
    z = jax.random.normal(jax.random.PRNGKey(0), (mu.shape[0], 3, n))
    f = z*jnp.sqrt(var)[:,:,None] + mu[:,:,None]
    return np.array(jax.nn.softmax(f/T, axis=1).mean(axis=2))[:, TARGET]

# accumulate (gt, judged) per (n_obs, T)
acc = {(n,T):([],[]) for n in NOBS for T in TS}
for n in NOBS:
    for p in GT:
        yn, rng = noisy(p); io = rng.choice(len(Xtr_pool), n, replace=False)
        mu, var = gp_latents(Xtr_pool[io], yn[io])
        gt = np.where(yte==TARGET, p, 0.0)
        for T in TS:
            jd = judged_at_T(mu, var, T)
            acc[(n,T)][0].append(gt); acc[(n,T)][1].append(jd)

# LPE baseline
lpe = {}
for n in NOBS:
    g_, j_ = [], []
    for p in GT:
        yn, rng = noisy(p); io = rng.choice(len(Xtr_pool), n, replace=False)
        if len(np.unique(yn[io])) < 2:  # LPE needs >=2 classes
            continue
        try:
            u = ppm.lpe_multiclass(Xte_j, jnp.array(Xtr_pool[io]), yn[io], num_classes=3, repeats=10)
        except Exception:
            continue
        g_.append(np.where(yte==TARGET,p,0.0)); j_.append(np.array(u['categorical_mu'])[:,TARGET])
    if g_:
        g_=np.concatenate(g_); j_=np.concatenate(j_); lpe[n]=pearsonr(g_,j_)[0]
    else:
        lpe[n]=float('nan')

print("LPE per-n_obs:", {k:round(v,3) for k,v in lpe.items()}, "mean", round(np.mean(list(lpe.values())),3))
print(f"\nGPP-Dirichlet Pearson by softmax temperature T (T=1 is current):")
print(f"{'T':>5} | {'n2':>6} {'n8':>6} {'n32':>6} {'n128':>6} | {'mean':>6}")
for T in TS:
    r = {}
    for n in NOBS:
        g_=np.concatenate(acc[(n,T)][0]); j_=np.concatenate(acc[(n,T)][1])
        r[n]=pearsonr(g_,j_)[0] if np.std(j_)>1e-9 else float('nan')
    print(f"{T:5.1f} | {r[2]:6.3f} {r[8]:6.3f} {r[32]:6.3f} {r[128]:6.3f} | {np.nanmean(list(r.values())):6.3f}")
