"""Epistemic axis of the disagreement study (uses existing emb.npz; CPU).

Novel-class OOD: train the probe on K-1 classes (one class held out, never seen),
then score every test item. The held-out class's items are epistemically OOD --
the probe has no evidence about that region -- so GPP's epistemic signals
(negative summed latent variance, mutual information) should rank them as OOD.
Metric: AUROC(is-held-out-class) per held-out class, averaged, at the best layer,
vs Maha / MSP / deep-kNN / LPE baselines. (Symmetric to the aleatoric test:
epistemic responds to *missing evidence*, not to human disagreement.)

Only K>=3 datasets (holding out a class from K=2 leaves one class).

Usage: python experiments/disagreement/step2_epistemic.py --dataset chaosnli_snli --model gemma
"""
import os, sys, argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import sklearn.linear_model as sklm

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
from GPax.probing import probabilistic_probe_multiclass as ppm
import jax, jax.numpy as jnp

N_TRAIN = 512; SEEDS = [0, 1, 2]; LAYER_FRACS = None   # use all layers, pick best by ID accuracy


def scores_for(Xtr, ytr, Xte, K):
    """Return dict name->score (higher = more in-distribution) for each method,
    plus the probe's ID accuracy (to pick the informative layer)."""
    out = {}
    g = ppm.gpp_multiclass(jnp.array(Xte), jnp.array(Xtr), jax.nn.one_hot(ytr, K), num_classes=K, n=1500)
    out['GPP neg-latent-var'] = -np.sum(np.array(g['latent_var']), axis=1)
    out['GPP -MI'] = -np.array(g['information_gain']).flatten()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_z, Xte_z = (Xtr - mu) / sd, (Xte - mu) / sd
    try:
        out['Maha'] = np.array(ppm.maha_multiclass(jnp.array(Xte_z), jnp.array(Xtr_z), ytr, num_classes=K)['episteme'])
    except Exception: out['Maha'] = np.full(len(Xte), np.nan)
    try:
        out['MSP'] = np.array(ppm.lp_maxprob_multiclass(jnp.array(Xte_z), jnp.array(Xtr_z), ytr)['episteme'])
    except Exception: out['MSP'] = np.full(len(Xte), np.nan)
    try:
        k = min(5, len(Xtr_z)); nn = NearestNeighbors(n_neighbors=k).fit(Xtr_z)
        out['kNN'] = -nn.kneighbors(Xte_z)[0].mean(1)
    except Exception: out['kNN'] = np.full(len(Xte), np.nan)
    try:
        l = ppm.lpe_multiclass(jnp.array(Xte_z), jnp.array(Xtr_z), ytr, num_classes=K, repeats=40)
        out['LPE -var'] = -np.sum(np.array(l['epistemic_var']), axis=1)
    except Exception: out['LPE -var'] = np.full(len(Xte), np.nan)
    return out, np.array(g['categorical_mu'])


def run(dataset, model, variant=None):
    d = np.load(os.path.join(HERE, 'data', dataset, model, f'emb_{variant}.npz' if variant else 'emb.npz'))
    K = int(d['K']); layers = list(d['layers'])
    assert K >= 3, 'novel-class holdout needs K>=3'
    hard_tr, hard_te = d['hard_train'], d['hard_test']
    classes, counts = np.unique(hard_tr, return_counts=True)
    # hold out classes that have enough train + test support to measure AUROC
    te_counts = {c: int((hard_te == c).sum()) for c in classes}
    holdout = [int(c) for c in classes if counts[list(classes).index(c)] >= 20 and te_counts[c] >= 15]
    holdout = sorted(holdout, key=lambda c: -te_counts[c])[:8]
    methods = ['GPP neg-latent-var', 'GPP -MI', 'Maha', 'MSP', 'kNN', 'LPE -var']
    rows = []
    for li in layers:
        Xtr_all, Xte = d[f'X_train_L{li}'], d[f'X_test_L{li}']
        for c in holdout:
            tr_keep = hard_tr != c
            Xtr_pool, ytr_pool = Xtr_all[tr_keep], hard_tr[tr_keep]
            is_ood = (hard_te == c).astype(int)
            if is_ood.sum() < 10 or (is_ood == 0).sum() < 10:
                continue
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                if len(np.unique(ytr_pool)) < 2:
                    continue
                io = rng.choice(len(Xtr_pool), min(N_TRAIN, len(Xtr_pool)), replace=False)
                try:
                    sc, mu = scores_for(Xtr_pool[io], ytr_pool[io], Xte, K)
                except Exception as e:
                    print('skip', li, c, seed, repr(e)[:70]); continue
                # ID accuracy on the kept (non-held-out) test items, to rank layers
                idm = is_ood == 0
                id_acc = float((mu[idm].argmax(1) == hard_te[idm]).mean())
                for name in methods:
                    s = sc[name]
                    auroc = (roc_auc_score(is_ood, -s) if np.all(np.isfinite(s)) and np.std(s) > 1e-12
                             else np.nan)
                    rows.append(dict(dataset=dataset, model=model, layer=int(li), heldout=c,
                                     seed=seed, method=name, auroc=auroc, id_acc=id_acc))
        print(f'  layer {li} done', flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    outdir = os.path.join(REPO, 'experiments', 'disagreement', 'figures', dataset,
                          model + (f'_{variant}' if variant else ''))
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, 'epistemic_raw.csv'), index=False)
    # best layer = highest ID accuracy (the layer that actually represents the task)
    bl = int(df.groupby('layer').id_acc.mean().idxmax())
    big = df[df.layer == bl]
    print(f"\n===== {dataset}/{model} novel-class OOD AUROC (best layer L{bl}, id_acc={big.id_acc.mean():.2f}, "
          f"{len(holdout)} held-out classes) =====")
    summary = {'dataset': dataset, 'model': model, 'K': K, 'best_layer': bl, 'holdout_classes': holdout}
    for name in methods:
        a = big[big.method == name].auroc
        summary[name] = float(a.mean())
        print(f"  {name:20s} AUROC = {a.mean():.3f} +/- {a.std():.3f}")
    json.dump(summary, open(os.path.join(outdir, 'epistemic_summary.json'), 'w'), indent=2)
    print(f"saved {outdir}/epistemic_raw.csv + epistemic_summary.json")
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--model', required=True); ap.add_argument('--variant', default=None)
    a=ap.parse_args(); run(a.dataset, a.model, a.variant)
