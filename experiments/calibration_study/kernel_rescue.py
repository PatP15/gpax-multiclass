import os, sys
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, jax, jax.numpy as jnp
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import jax.scipy.linalg as jspla
from GPax.probing import gp_multiclass as gpm
from GPax.probing import probabilistic_probe_multiclass as ppm

emb = np.load(f'{REPO}/results/embeddings/embeddings_M1.npy')
d = np.load(f'{REPO}/results/embeddings/data_labels.npz', allow_pickle=True)
y = d['P2_shape']; L = min(len(emb), len(y)); emb, y = emb[:L], y[:L]
m = y < 3; X, y = emb[m], y[m].astype(int)
Xtr_pool, Xte, ytr_pool, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# z-score (fit on train pool) so RBF lengthscale is on a sane scale
mu_, sd_ = Xtr_pool.mean(0), Xtr_pool.std(0) + 1e-8
Xtr_pool = (Xtr_pool - mu_)/sd_; Xte = (Xte - mu_)/sd_
ridx = np.random.RandomState(0).choice(len(Xte), 1500, replace=False); Xte, yte = Xte[ridx], yte[ridx]
Xte_j = jnp.array(Xte)
GT=[0.25,0.5,0.75,1.0]; NOBS=[2,8,32,128]; TARGET=0; LS=[3.,6.,11.,22.,44.]

def noisy(p):
    yn=ytr_pool.copy(); rng=np.random.RandomState(42+int(p*100))
    pos=np.where(ytr_pool==TARGET)[0]; nf=int(len(pos)*(1-p))
    if nf: fi=rng.choice(pos,nf,replace=False); yn[fi]=rng.choice([1,2],size=nf)
    return yn,rng

def gpp_judged(Xo, yo, cov, extra):
    yoh=jax.nn.one_hot(yo,3); params={'alpha_eps':0.1,'strength':5.0}; params.update(extra)
    preds=gpm.dirichlet_gp_predict(mean_func=gpm.constant_mean,cov_func=cov,x_query=Xte_j,
            x_observed=jnp.array(Xo),y_observed=yoh,params=params,var_only=True)
    mus,vars=gpm.get_latent_gp_dirichlet(preds)
    mu=jnp.hstack(mus); var=jnp.maximum(jnp.hstack(vars),1e-32)
    z=jax.random.normal(jax.random.PRNGKey(0),(mu.shape[0],3,800))
    f=z*jnp.sqrt(var)[:,:,None]+mu[:,:,None]
    return np.array(jax.nn.softmax(f,axis=1).mean(2))[:,TARGET]

def marglik(Xo, yo, cov, extra):
    """ -log marginal likelihood (summed over classes) of the latent Dirichlet-GP regression. """
    yoh=jax.nn.one_hot(yo,3); params={'alpha_eps':0.1,'strength':5.0}; params.update(extra)
    params=gpm.set_default_params_dirichlet(params,3)
    yl,vl=gpm.get_latent_observations_dirichlet(params,yoh)
    Xo_j=jnp.array(Xo); K=cov(params,Xo_j); mfun=gpm.constant_mean(params,Xo_j)
    tot=0.0
    for k in range(3):
        Kn=K+jnp.diag(vl[:,k]); ch=jspla.cholesky(Kn,lower=True)
        r=yl[:,k:k+1]-mfun; a=jspla.cho_solve((ch,True),r)
        tot+=float(0.5*(r.T@a).squeeze()+jnp.sum(jnp.log(jnp.diag(ch)))+0.5*len(yo)*jnp.log(2*jnp.pi))
    return tot

def per_nobs(cov, extra):
    out={}; nll={}
    for n in NOBS:
        g_,j_,ml=[],[],[]
        for p in GT:
            yn,rng=noisy(p); io=rng.choice(len(Xtr_pool),n,replace=False)
            Xo,yo=Xtr_pool[io],yn[io]
            g_.append(np.where(yte==TARGET,p,0.)); j_.append(gpp_judged(Xo,yo,cov,extra))
            try: ml.append(marglik(Xo,yo,cov,extra))
            except Exception: pass
        g_=np.concatenate(g_); j_=np.concatenate(j_)
        out[n]=pearsonr(g_,j_)[0] if np.std(j_)>1e-9 else float('nan'); nll[n]=np.mean(ml) if ml else float('nan')
    return out,nll

print("=== Cosine kernel (current GPP) ===")
rc,_=per_nobs(gpm.cosine_kernel,{}); print(" ",{k:round(v,3) for k,v in rc.items()},"mean",round(np.nanmean(list(rc.values())),3))
print("\n=== RBF (squared_exponential) by lengthscale ===")
print(f"{'ls':>5} | {'n2':>6} {'n8':>6} {'n32':>6} {'n128':>6} | {'meanR':>6} | {'NLL@128':>8}")
best=None; nll_by_ls={}
for ls in LS:
    r,nll=per_nobs(gpm.squared_exponential_kernel,{'lengthscale':ls})
    nll_by_ls[ls]=nll; mr=np.nanmean(list(r.values()))
    print(f"{ls:5.0f} | {r[2]:6.3f} {r[8]:6.3f} {r[32]:6.3f} {r[128]:6.3f} | {mr:6.3f} | {nll[128]:8.1f}")
    if best is None or mr>best[0]: best=(mr,ls,r,nll)
print(f"\nbest RBF (by Pearson): ls={best[1]} meanR={best[0]:.3f} per-n_obs={ {k:round(v,3) for k,v in best[2].items()} }")
mlsel=min(((best_nll[128], ls2) for ls2,best_nll in nll_by_ls.items() if not np.isnan(best_nll[128])), default=(None,None))
print(f"ML-selected ls (lowest NLL@128): ls={mlsel[1]}")
print("LPE baseline per-n_obs (from prior run): {2:0.063, 8:0.164, 32:0.512, 128:0.588} mean 0.332")
