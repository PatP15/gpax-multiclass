"""Disagreement-study probing harness.

For one (dataset, model) emb.npz, sweeps layers x methods x seeds x n_obs. Each
cell: train the probe on n_obs HARD-labeled examples, predict the test distribution,
and score it against the held-out HUMAN SOFT labels. Records, per cell:
  - distribution recovery: soft_ce, jsd, tvd, entce, rankcs, accuracy
  - aleatoric tracking: corr(disagreement-proxy(soft), predicted-aleatoric)  [RQ1]
  - epistemic control:  corr(disagreement-proxy(soft), predicted-MI)         [should be ~0]
  - epistemic scarcity: mean MI (so MI-vs-n_obs can be checked downstream)    [RQ3]
Matched-conditions: a cell is recorded only if every method succeeds.

Usage:
  python experiments/disagreement/step2_probe.py --dataset chaosnli_snli --model gemma [--synth]
"""
import os, sys, argparse, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import sklearn.linear_model as sklm
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'experiments', 'calibration_study'))
import soft_metrics as sm
from GPax.probing import probabilistic_probe_multiclass as ppm
import jax, jax.numpy as jnp

NOBS = [16, 64, 256, 1024]; SEEDS = list(range(5))
METHODS = ['GPP-cosine', 'GPP-rbf', 'GPP-laplace', 'LPE', 'LP-temp']


def _nmc(K):
    return 1000 if K > 10 else 2000


def predict(method, Xtr, ytr, Xte, K, nmc):
    """Return (P:(n,K) predicted dist, alea:(n,) aleatoric, mi:(n,) MI or nan)."""
    if method.startswith('GPP'):
        kern = {'GPP-cosine': 'cosine', 'GPP-rbf': 'rbf', 'GPP-laplace': 'laplace'}[method]
        if kern == 'cosine':
            m = ppm.gpp_multiclass(jnp.array(Xte), jnp.array(Xtr), jax.nn.one_hot(ytr, K),
                                   num_classes=K, n=nmc)
        else:
            m = ppm.gpp_multiclass_select(Xte, Xtr, ytr, num_classes=K, kernel=kern,
                                          lengthscale='auto', standardize=True, n=nmc)
        return (np.array(m['categorical_mu']), np.array(m['Alea']).flatten(),
                np.array(m['information_gain']).flatten())
    if method == 'LPE':
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        m = ppm.lpe_multiclass(jnp.array((Xte - mu) / sd), jnp.array((Xtr - mu) / sd), ytr,
                               num_classes=K, repeats=50)
        return (np.array(m['categorical_mu']), np.array(m['Alea']).flatten(),
                np.array(m['information_gain']).flatten())
    if method == 'LP-temp':
        from sklearn.model_selection import cross_val_predict
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_z, Xte_z = (Xtr - mu) / sd, (Xte - mu) / sd
        mk = lambda: sklm.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        cv = int(min(5, np.min(np.bincount(ytr, minlength=K)[np.bincount(ytr, minlength=K) > 0])))
        def pad(dec, cls):
            if dec.ndim == 1: dec = np.vstack([-dec, dec]).T
            full = np.full((len(dec), K), -1e9); full[:, cls.astype(int)] = dec; return full
        def smx(L, T):
            z = L / T; z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
        if cv < 2:
            raise ValueError('LP-temp cv<2')
        oof = cross_val_predict(mk(), Xtr_z, ytr, cv=cv, method='decision_function')
        Loof = pad(np.asarray(oof), np.unique(ytr))
        T = float(minimize_scalar(lambda T: -np.mean(np.log(smx(Loof, T)[np.arange(len(ytr)), ytr] + 1e-12)),
                                  bounds=(0.05, 100.0), method='bounded').x)
        clf = mk().fit(Xtr_z, ytr)
        P = smx(pad(clf.decision_function(Xte_z), clf.classes_), T)
        return P, sm.entropy(P), np.full(len(P), np.nan)   # point classifier: no MI
    raise ValueError(method)


def _clean(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return None
    return a[ok], b[ok]


def corr(a, b):
    c = _clean(a, b)
    return float('nan') if c is None else float(pearsonr(*c)[0])


def corr_s(a, b):
    """Spearman rank correlation — robust to range restriction / nonlinearity."""
    c = _clean(a, b)
    return float('nan') if c is None else float(spearmanr(*c)[0])


def binned_mono(human, alea, nq=4):
    """Does predicted aleatoric rise monotonically across human-disagreement
    quartiles? Spearman of quartile index vs per-quartile mean alea (-1..1)."""
    c = _clean(human, alea)
    if c is None:
        return float('nan')
    h, a = c
    edges = np.quantile(h, np.linspace(0, 1, nq + 1)[1:-1])
    q = np.digitize(h, edges)
    means = [a[q == i].mean() for i in range(nq) if np.any(q == i)]
    return float(spearmanr(np.arange(len(means)), means)[0]) if len(means) > 2 else float('nan')


def run(dataset, model, synth=False, prompt=False, variant=None):
    if variant:
        fn, suffix = f'emb_{variant}.npz', f'_{variant}'
    elif synth:
        fn, suffix = 'emb_synth.npz', '_synth'
    elif prompt:
        fn, suffix = 'emb_prompted.npz', '_prompted'
    else:
        fn, suffix = 'emb.npz', ''
    d = np.load(os.path.join(HERE, 'data', dataset, model, fn))
    K = int(d['K']); layers = list(d['layers']); soft_te = d['soft_test']; hard_te = d['hard_test']
    nmc = _nmc(K)
    # disagreement proxies on the human soft labels (ground-truth aleatoric)
    proxies = {name: fn(soft_te) for name, fn in sm.PROXIES.items()}
    if K == 2:
        proxies['bern_var'] = sm.bern_var(soft_te)
    rows = []
    for li in layers:
        Xtr_all, Xte = d[f'X_train_L{li}'], d[f'X_test_L{li}']
        ytr_all = d['hard_train']
        for seed in SEEDS:
            rng = np.random.RandomState(100 + seed)
            for n in NOBS:
                if n > len(Xtr_all): continue
                try:
                    io = rng.choice(len(Xtr_all), n, replace=False)
                    Xo, yo = Xtr_all[io], ytr_all[io]
                    if len(np.unique(yo)) < 2: continue
                    cell = {m: predict(m, Xo, yo, Xte, K, nmc) for m in METHODS}
                except Exception as e:
                    print('cell dropped (matched):', dataset, model, li, seed, n, repr(e)[:90]); continue
                for m, (P, alea, mi) in cell.items():
                    row = dict(dataset=dataset, model=model, layer=int(li), method=m, seed=seed, n_obs=n,
                               soft_ce=float(sm.soft_ce(soft_te, P).mean()),
                               jsd=float(sm.jsd(soft_te, P).mean()),
                               tvd=float(sm.tvd(soft_te, P).mean()),
                               entce=float(np.abs(sm.entce(soft_te, P)).mean()),
                               rankcs=float(sm.rankcs(soft_te, P).mean()),
                               acc=float((P.argmax(1) == hard_te).mean()),
                               mean_alea=(float(np.nanmean(alea)) if np.any(np.isfinite(alea)) else float('nan')),
                               mean_mi=(float(np.nanmean(mi)) if np.any(np.isfinite(mi)) else float('nan')))
                    for pname, pv in proxies.items():
                        row[f'corrAlea_{pname}'] = corr(pv, alea)      # RQ1: expect HIGH
                        row[f'corrMI_{pname}'] = corr(pv, mi)          # control: expect ~0
                    he = proxies['entropy']                            # robust views (primary proxy)
                    row['corrAleaS_entropy'] = corr_s(he, alea)        # Spearman
                    row['corrMIS_entropy'] = corr_s(he, mi)
                    row['mono_alea'] = binned_mono(he, alea)           # quartile monotonicity
                    rows.append(row)
        print(f'  layer {li} done', flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    outdir = os.path.join(REPO, 'experiments', 'disagreement', 'figures', dataset, model + suffix)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, 'probe_raw.csv'), index=False)

    # headline summary (at the largest n_obs, best layer per method by soft_ce)
    big = df[df.n_obs == df.n_obs.max()]
    summary = {'dataset': dataset, 'model': model, 'K': K}
    print(f"\n===== {dataset}/{model} (n_obs={df.n_obs.max()}) — aleatoric tracking (corr with human entropy) =====")
    for m in METHODS:
        s = big[big.method == m]
        if s.empty: continue
        # pick the layer with best (lowest) soft_ce for this method
        bl = s.groupby('layer').soft_ce.mean().idxmin()
        sl = s[s.layer == bl]
        ca, cm = sl['corrAlea_entropy'].mean(), sl['corrMI_entropy'].mean()
        cas, mono = sl['corrAleaS_entropy'].mean(), sl['mono_alea'].mean()
        summary[m] = dict(best_layer=int(bl), soft_ce=float(sl.soft_ce.mean()), tvd=float(sl.tvd.mean()),
                          acc=float(sl.acc.mean()), corrAlea_entropy=float(ca), corrMI_entropy=float(cm),
                          corrAleaS_entropy=float(cas), mono_alea=float(mono))
        print(f"  {m:11s} L{bl} soft_ce={sl.soft_ce.mean():.3f} tvd={sl.tvd.mean():.3f} acc={sl.acc.mean():.3f}"
              f"  Alea: r={ca:+.2f} rho={cas:+.2f} mono={mono:+.2f} | MI(ctrl): r={cm:+.2f}")
    json.dump(summary, open(os.path.join(outdir, 'probe_summary.json'), 'w'), indent=2)

    # per-item arrays at the headline config (best layer per method, largest n, seed 0)
    # for the aleatoric-vs-human-disagreement scatter figure (step3).
    he_all = sm.entropy(soft_te); per = {'human_entropy': he_all}
    nmax = df.n_obs.max()
    for m in METHODS:
        if m not in summary: continue
        bl = summary[m]['best_layer']
        rng = np.random.RandomState(100)
        io = rng.choice(len(d[f'X_train_L{bl}']), min(nmax, len(d[f'X_train_L{bl}'])), replace=False)
        try:
            P, alea, mi = predict(m, d[f'X_train_L{bl}'][io], d['hard_train'][io], d[f'X_test_L{bl}'], K, nmc)
            per[f'alea_{m}'] = alea; per[f'mi_{m}'] = mi
        except Exception as e:
            print('per-item headline skipped for', m, repr(e)[:60])
    np.savez(os.path.join(outdir, 'per_item.npz'), **per)
    print(f"saved {outdir}/probe_raw.csv + probe_summary.json + per_item.npz")
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--synth', action='store_true'); ap.add_argument('--prompt', action='store_true'); ap.add_argument('--variant', default=None)
    a = ap.parse_args()
    run(a.dataset, a.model, a.synth, a.prompt, a.variant)
