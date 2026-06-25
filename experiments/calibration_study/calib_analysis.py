import numpy as np, os
import pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ece(probs, y, n_bins=15):
    probs = np.asarray(probs); y = np.asarray(y).astype(int)
    conf = probs.max(1); pred = probs.argmax(1); corr = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; N = len(y)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.sum() > 0:
            e += m.sum()/N * abs(conf[m].mean() - corr[m].mean())
    return e

def brier(probs, y):
    probs = np.asarray(probs); y = np.asarray(y).astype(int)
    K = probs.shape[1]; oh = np.eye(K)[y]
    return ((probs - oh)**2).sum(1).mean()

def load_raw(path, n='n2400'):
    d = np.load(path, allow_pickle=True)
    return d[f'{n}_probs'], d[f'{n}_y_true']

print("=== AnnoMI: calibration GPP(few-shot) vs LPE(few-shot), n=2400 ===")
print(f"{'model':8s}  {'ECE_GPP':>8s} {'ECE_LPE':>8s}   {'Brier_GPP':>9s} {'Brier_LPE':>9s}   winner")
for M in ['gemma','qwen','gemma4','qwen36']:
    try:
        gp, gy = load_raw(f'{REPO}/experiments/annomi/data/{M}/raw_exp2_fewshot.npz')
        lp, ly = load_raw(f'{REPO}/experiments/annomi/data/{M}/raw_lpe.npz')
        eg, el = ece(gp, gy), ece(lp, ly); bg, bl = brier(gp, gy), brier(lp, ly)
        win = 'GPP' if eg < el else 'LPE'
        print(f"{M:8s}  {eg:8.4f} {el:8.4f}   {bg:9.4f} {bl:9.4f}   ECE:{win}")
    except Exception as e:
        print(f"{M:8s}  ERR {e}")

print("\n=== 3D-Shapes: Pearson(judged_prob, gt_prob) GPP vs LPE  [paper Fig 5 metric] ===")
try:
    from scipy.stats import pearsonr
    for f in ['results_jax_fuzziness_p1.csv','results_jax_fuzziness_p2.csv','results_jax_unc_binary.csv','results_jax_sim_binary.csv']:
        p = f'{REPO}/results/csv/{f}'
        if not os.path.exists(p): continue
        df = pd.read_csv(p)
        cols = list(df.columns)
        gtc = next((c for c in cols if 'gt' in c.lower()), None)
        jpc = next((c for c in cols if 'judg' in c.lower() or 'prob' in c.lower()), None)
        mc = 'method' if 'method' in cols else None
        print(f"-- {f}  cols={cols}")
        if gtc and jpc and mc:
            for meth in sorted(df[mc].dropna().unique()):
                d = df[df[mc]==meth].dropna(subset=[gtc,jpc])
                if len(d) > 5:
                    r = pearsonr(d[gtc], d[jpc])[0]
                    print(f"     {meth:12s} Pearson(gt,judged)={r:.3f}  n={len(d)}")
except Exception as e:
    print("3D-Shapes Pearson ERR:", e)
print("DONE")
