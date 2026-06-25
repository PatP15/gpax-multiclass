import pandas as pd, numpy as np, os
from scipy.stats import pearsonr
REPO='/n/home01/ppuma/gpax-multiclass'

def analyze(fname, label):
    p=f'{REPO}/results/csv/{fname}'
    if not os.path.exists(p): print(fname,"missing"); return
    df=pd.read_csv(p)
    print(f"\n################  {label}  ################")
    for meth in ['GPP-Dirichlet','GPP-Beta','LPE']:
        d=df[df.method==meth].dropna(subset=['gt_prob','judged_prob','episteme'])
        if len(d)<10: continue
        # (1) per-n_obs Pearson (paper: GPP higher at each n, improving with n)
        rs=[]
        for n in sorted(d.n_obs.unique()):
            g=d[d.n_obs==n]
            if g.gt_prob.nunique()>1: rs.append((n,round(pearsonr(g.gt_prob,g.judged_prob)[0],3)))
        # (2) rational uncertainty on genuinely-fuzzy queries (gt=0.5): should center ~0.5, NOT extreme
        fz=d[np.isclose(d.gt_prob,0.5)]
        ext=float(((fz.judged_prob<0.2)|(fz.judged_prob>0.8)).mean())
        # at LOW episteme (bottom tercile) - "confidently ignorant" test
        loE=fz[fz.episteme<=fz.episteme.quantile(0.33)]
        ext_loE=float(((loE.judged_prob<0.2)|(loE.judged_prob>0.8)).mean()) if len(loE)>5 else float('nan')
        # (3) non-target queries (gt=0): judged should be low; GPP prior-floor vs LPE overconfidence
        non=d[d.gt_prob==0]
        tgt=d[d.gt_prob>0]
        r_tgt=round(pearsonr(tgt.gt_prob,tgt.judged_prob)[0],3) if tgt.gt_prob.nunique()>1 else float('nan')
        print(f"  {meth:14s} per-n_obs r={rs}")
        print(f"        fuzzy(gt=0.5): judged mean={fz.judged_prob.mean():.3f} std={fz.judged_prob.std():.3f} frac_extreme={ext:.3f} (lowEpi {ext_loE:.3f})")
        print(f"        non-target(gt=0): judged mean={non.judged_prob.mean():.3f}  | target-only Pearson r={r_tgt}")

analyze('results_jax_fuzziness_p1.csv','BINARY P1 (paper original setting)')
analyze('results_jax_fuzziness_p2.csv','MULTICLASS P2 (Dirichlet extension)')
print("\nDONE")
