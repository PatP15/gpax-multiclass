"""Figures for the disagreement study (consumes step2_probe outputs).

Per (dataset, model) under figures/<dataset>/<model>/:
  - aleatoric-vs-human-disagreement scatter (the RQ1 headline), GPP vs LPE
  - layer profile: aleatoric-tracking (Spearman) and recovery (TVD) by depth
  - the dissociation bars: corr(Alea, human-H) vs corr(MI, human-H) per method
  - epistemic scarcity: mean MI vs n_obs per method
Aggregate (no --dataset): cross dataset/model recovery + tracking table.

Usage:
  python experiments/disagreement/step3_plots.py --dataset lewidi_md --model gemma
  python experiments/disagreement/step3_plots.py            # aggregate over all available
"""
import os, sys, glob, json, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.join(HERE, 'figures')
METHODS = ['GPP-cosine', 'GPP-rbf', 'GPP-laplace', 'LPE', 'LP-temp']
COLORS = {'GPP-cosine': 'C0', 'GPP-rbf': 'C2', 'GPP-laplace': 'C3', 'LPE': 'C1', 'LP-temp': 'C5'}


def per_model(dataset, model):
    d = os.path.join(FIGROOT, dataset, model)
    df = pd.read_csv(os.path.join(d, 'probe_raw.csv'))
    big = df[df.n_obs == df.n_obs.max()]
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))

    # (1) headline scatter from per_item.npz
    pi_path = os.path.join(d, 'per_item.npz')
    if os.path.exists(pi_path):
        pi = np.load(pi_path); h = pi['human_entropy']
        for m, c in [('GPP-laplace', 'C3'), ('LPE', 'C1')]:
            key = f'alea_{m}'
            if key in pi.files:
                a = pi[key]
                # bin means for a clean trend line over the scatter
                ax[0].scatter(h, a, s=6, alpha=0.15, color=c, label=m)
                edges = np.quantile(h, np.linspace(0, 1, 6)); cx, cy = [], []
                for i in range(5):
                    msk = (h >= edges[i]) & (h <= edges[i + 1])
                    if msk.sum(): cx.append(h[msk].mean()); cy.append(a[msk].mean())
                ax[0].plot(cx, cy, color=c, lw=2.5)
        ax[0].set_xlabel('human disagreement  H(soft label)'); ax[0].set_ylabel('predicted aleatoric  E[H(p)]')
        ax[0].set_title('RQ1: aleatoric vs human disagreement'); ax[0].legend(fontsize=8)

    # (2) layer profile: aleatoric-tracking Spearman by depth
    for m in METHODS:
        s = big[big.method == m]
        if s.empty: continue
        g = s.groupby('layer')['corrAleaS_entropy'].agg(['mean', 'std'])
        ax[1].errorbar(g.index, g['mean'], yerr=g['std'], marker='o', capsize=3, color=COLORS[m], label=m)
    ax[1].axhline(0, ls='--', color='k', alpha=0.4)
    ax[1].set_xlabel('layer'); ax[1].set_ylabel('Spearman(human-H, Alea)')
    ax[1].set_title('RQ6: aleatoric tracking by depth'); ax[1].legend(fontsize=8)

    # (3) dissociation: Alea-tracking vs MI-tracking (best layer per method)
    labels, alea_c, mi_c = [], [], []
    for m in METHODS:
        s = big[big.method == m]
        if s.empty: continue
        bl = s.groupby('layer').soft_ce.mean().idxmin(); sl = s[s.layer == bl]
        labels.append(m); alea_c.append(sl['corrAleaS_entropy'].mean()); mi_c.append(sl['corrMIS_entropy'].mean())
    x = np.arange(len(labels)); w = 0.38
    ax[2].bar(x - w / 2, alea_c, w, label='Alea (want high)', color='C3')
    ax[2].bar(x + w / 2, mi_c, w, label='MI control (want ~0)', color='C0')
    ax[2].axhline(0, color='k', lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax[2].set_ylabel('Spearman with human-H'); ax[2].set_title('RQ1/RQ4: aleatoric vs epistemic dissociation'); ax[2].legend(fontsize=8)

    # (4) epistemic scarcity: mean MI vs n_obs (GPP/LPE)
    for m in ['GPP-cosine', 'GPP-rbf', 'GPP-laplace', 'LPE']:
        s = df[(df.method == m)]
        if s.empty or s['mean_mi'].isna().all(): continue
        g = s.groupby('n_obs')['mean_mi'].mean()
        ax[3].plot(g.index, g.values, marker='o', color=COLORS[m], label=m)
    ax[3].set_xscale('log'); ax[3].set_xlabel('# observations'); ax[3].set_ylabel('mean epistemic MI')
    ax[3].set_title('RQ3: epistemic MI vs evidence'); ax[3].legend(fontsize=8)

    plt.suptitle(f'Disagreement study — {dataset} / {model}', fontsize=14); plt.tight_layout()
    out = os.path.join(d, 'disagreement_overview.png'); fig.savefig(out, dpi=170)
    print('saved', out)


def aggregate():
    rows = []
    for sj in glob.glob(os.path.join(FIGROOT, '*', '*', 'probe_summary.json')):
        s = json.load(open(sj))
        for m in METHODS:
            if m in s:
                rows.append(dict(dataset=s['dataset'], model=s['model'], K=s['K'], method=m, **s[m]))
    if not rows:
        print('no probe_summary.json found yet'); return
    df = pd.DataFrame(rows)
    print('\n=== cross dataset/model headline (best layer, largest n) ===')
    cols = ['dataset', 'model', 'K', 'method', 'soft_ce', 'tvd', 'acc', 'corrAleaS_entropy', 'mono_alea', 'corrMI_entropy']
    print(df[cols].to_string(index=False))
    df.to_csv(os.path.join(FIGROOT, 'aggregate_summary.csv'), index=False)
    print('\nsaved', os.path.join(FIGROOT, 'aggregate_summary.csv'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset'); ap.add_argument('--model')
    a = ap.parse_args()
    if a.dataset and a.model:
        per_model(a.dataset, a.model)
    else:
        aggregate()
