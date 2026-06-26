import os, sys
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if '__file__' in dir() else '/n/home01/ppuma/gpax-multiclass'
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, jax, jax.numpy as jnp
import jax.scipy.linalg as jspla
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from GPax.probing import gp_multiclass as gpm
from GPax.probing import probabilistic_probe_multiclass as ppm

emb = np.load(f'{REPO}/results/embeddings/embeddings_M1.npy')
d = np.load(f'{REPO}/results/embeddings/data_labels.npz', allow_pickle=True)
y = d['P2_shape']; L = min(len(emb), len(y)); emb, y = emb[:L], y[:L]
m = y < 3; X, y = emb[m], y[m].astype(int)
Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
ridx = np.random.RandomState(0).choice(len(Xte_raw), 1200, replace=False)
Xte_raw, yte = Xte_raw[ridx], yte[ridx]
mu_, sd_ = Xtr_raw.mean(0), Xtr_raw.std(0)+1e-8
Xtr_z, Xte_z = (Xtr_raw-mu_)/sd_, (Xte_raw-mu_)/sd_
GT=[0.25,0.5,0.75,1.0]; NOBS=[2,8,32,128]; TARGET=0; NMC=600

def noisy(p):
    yn=ytr.copy(); rng=np.random.RandomState(42+int(p*100))
    pos=np.where(ytr==TARGET)[0]; nf=int(len(pos)*(1-p))
    if nf: fi=rng.choice(pos,nf,replace=False); yn[fi]=rng.choice([1,2],size=nf)
    return yn,rng

def judged(Xtr, Xte_j, Xo_idx_src, cov, extra, Xo, yo):
    yoh=jax.nn.one_hot(yo,3); params={'alpha_eps':0.1,'strength':5.0}; params.update(extra)
    preds=gpm.dirichlet_gp_predict(mean_func=gpm.constant_mean,cov_func=cov,x_query=Xte_j,
            x_observed=jnp.array(Xo),y_observed=yoh,params=params,var_only=True)
    mus,vars=gpm.get_latent_gp_dirichlet(preds); mu=jnp.hstack(mus); var=jnp.maximum(jnp.hstack(vars),1e-32)
    z=jax.random.normal(jax.random.PRNGKey(0),(mu.shape[0],3,NMC)); f=z*jnp.sqrt(var)[:,:,None]+mu[:,:,None]
    return np.array(jax.nn.softmax(f,axis=1).mean(2))[:,TARGET]

def marglik(Xtr, cov, extra, Xo, yo):
    yoh=jax.nn.one_hot(yo,3); params={'alpha_eps':0.1,'strength':5.0}; params.update(extra)
    params=gpm.set_default_params_dirichlet(params,3); yl,vl=gpm.get_latent_observations_dirichlet(params,yoh)
    K=cov(params,jnp.array(Xo)); mf=gpm.constant_mean(params,jnp.array(Xo)); tot=0.0
    for k in range(3):
        ch=jspla.cholesky(K+jnp.diag(vl[:,k]),lower=True); r=yl[:,k:k+1]-mf; a=jspla.cho_solve((ch,True),r)
        tot+=float(0.5*(r.T@a).squeeze()+jnp.sum(jnp.log(jnp.diag(ch)))+0.5*len(yo)*jnp.log(2*jnp.pi))
    return tot

def run(name, Xtr, Xte_arr, cov, extra):
    Xte_j=jnp.array(Xte_arr); r={}; nll={}
    for n in NOBS:
        g_,j_,ml=[],[],[]
        for p in GT:
            yn,rng=noisy(p); io=rng.choice(len(Xtr),n,replace=False); Xo,yo=Xtr[io],yn[io]
            g_.append(np.where(yte==TARGET,p,0.)); j_.append(judged(Xtr,Xte_j,io,cov,extra,Xo,yo))
            try: ml.append(marglik(Xtr,cov,extra,Xo,yo))
            except Exception: pass
        g_=np.concatenate(g_); j_=np.concatenate(j_)
        r[n]=pearsonr(g_,j_)[0] if np.std(j_)>1e-9 else float('nan'); nll[n]=np.mean(ml) if ml else float('nan')
    return r,nll

def best_over_ls(name, Xtr, Xte_arr, cov, ls_name, grid):
    best=None
    for ls in grid:
        r,nll=run(name,Xtr,Xte_arr,cov,{ls_name:ls})
        if best is None or np.nanmean(list(r.values()))>np.nanmean(list(best[1].values())): best=(ls,r,nll)
    return best

print(f"{'kernel':24s} | {'n2':>6} {'n8':>6} {'n32':>6} {'n128':>6} | {'mean':>6} | best-ls (ML-NLL@128)")
# cosine (linear, angular) -- raw
rc,_=run('cosine',Xtr_raw,Xte_raw,gpm.cosine_kernel,{})
print(f"{'cosine (linear)':24s} | {rc[2]:6.3f} {rc[8]:6.3f} {rc[32]:6.3f} {rc[128]:6.3f} | {np.nanmean(list(rc.values())):6.3f} | (no ls)")
# RBF -- z-scored
ls,r,nll=best_over_ls('RBF',Xtr_z,Xte_z,gpm.squared_exponential_kernel,'lengthscale',[3.,6.,11.,22.])
print(f"{'RBF (sq-exp, L2)':24s} | {r[2]:6.3f} {r[8]:6.3f} {r[32]:6.3f} {r[128]:6.3f} | {np.nanmean(list(r.values())):6.3f} | ls={ls} (NLL {nll[128]:.0f})")
# Laplace (Matern-1/2, L1) -- z-scored
ls,r,nll=best_over_ls('Laplace',Xtr_z,Xte_z,gpm.laplace_kernel,'lengthscale',[10.,30.,60.,120.])
print(f"{'Laplace (Matern-1/2,L1)':24s} | {r[2]:6.3f} {r[8]:6.3f} {r[32]:6.3f} {r[128]:6.3f} | {np.nanmean(list(r.values())):6.3f} | ls={ls} (NLL {nll[128]:.0f})")
# SE-on-sphere (RBF on arccos(cosine)) -- raw (angular, ReLU>=0)
ls,r,nll=best_over_ls('SEsphere',Xtr_raw,Xte_raw,gpm.squared_exponential_sphere_kernel,'lengthscale',[0.2,0.5,1.0,1.5])
print(f"{'SE-sphere (angular+ls)':24s} | {r[2]:6.3f} {r[8]:6.3f} {r[32]:6.3f} {r[128]:6.3f} | {np.nanmean(list(r.values())):6.3f} | ls={ls} (NLL {nll[128]:.0f})")
# LPE baseline -- z-scored
rl={}
for n in NOBS:
    g_,j_=[],[]
    for p in GT:
        yn,rng=noisy(p); io=rng.choice(len(Xtr_z),n,replace=False)
        if len(np.unique(yn[io]))<2: continue
        try: u=ppm.lpe_multiclass(jnp.array(Xte_z),jnp.array(Xtr_z[io]),yn[io],num_classes=3,repeats=10)
        except Exception: continue
        g_.append(np.where(yte==TARGET,p,0.)); j_.append(np.array(u['categorical_mu'])[:,TARGET])
    if g_: g_=np.concatenate(g_); j_=np.concatenate(j_); rl[n]=pearsonr(g_,j_)[0]
    else: rl[n]=float('nan')
print(f"{'LPE (baseline)':24s} | {rl[2]:6.3f} {rl[8]:6.3f} {rl[32]:6.3f} {rl[128]:6.3f} | {np.nanmean(list(rl.values())):6.3f} | --")
