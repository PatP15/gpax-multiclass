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

    # (3) dissociation: raw tracking vs the *partial* correlations. Alea and MI come
    # from the same posterior, so the raw MI bar inherits Alea's association; only the
    # partials separate the disagreement axis from the evidence axis (RQ1b).
    labels, alea_c, mi_c, pa_c, pm_c = [], [], [], [], []
    for m in METHODS:
        s = big[big.method == m]
        if s.empty: continue
        bl = s.groupby('layer').soft_ce.mean().idxmin(); sl = s[s.layer == bl]
        labels.append(m); alea_c.append(sl['corrAleaS_entropy'].mean()); mi_c.append(sl['corrMIS_entropy'].mean())
        pa_c.append(sl['pcorrAleaS_entropy_given_MI'].mean() if 'pcorrAleaS_entropy_given_MI' in sl else np.nan)
        pm_c.append(sl['pcorrMIS_entropy_given_alea'].mean() if 'pcorrMIS_entropy_given_alea' in sl else np.nan)
    x = np.arange(len(labels)); w = 0.2
    ax[2].bar(x - 1.5 * w, alea_c, w, label='Alea (raw)', color='C3', alpha=0.45)
    ax[2].bar(x - 0.5 * w, mi_c, w, label='MI (raw)', color='C0', alpha=0.45)
    ax[2].bar(x + 0.5 * w, pa_c, w, label='Alea | MI (want high)', color='C3')
    ax[2].bar(x + 1.5 * w, pm_c, w, label='MI | Alea (want ~0)', color='C0')
    ax[2].axhline(0, color='k', lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax[2].set_ylabel('Spearman with human-H'); ax[2].set_title('RQ1/RQ1b: aleatoric vs epistemic dissociation'); ax[2].legend(fontsize=7)

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

    selective_panel(dataset, model)


def selective_panel(dataset, model, method='GPP-rbf'):
    """RQ7: is the decomposition useful? AURC by abstention score, at each n_obs,
    plus the human-entropy of the abstained set. Skipped if step2_selective has not
    run for this cell."""
    d = os.path.join(FIGROOT, dataset, model)
    f = os.path.join(d, 'selective_raw.csv')
    if not os.path.exists(f):
        print('no selective_raw.csv — skipping RQ7 panel for', dataset, model); return
    df = pd.read_csv(f)
    sub = df[df.method == method]
    if sub.empty:
        print(f'no {method} rows in selective_raw.csv'); return
    scols = {'alea': 'C3', 'mi': 'C0', 'total': 'C7', 'msp': 'C5', 'random': 'k'}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for sc, c in scols.items():
        s = sub[sub.score == sc]
        if s.empty: continue
        g = s.groupby('n_obs')['aurc'].agg(['mean', 'std'])
        ax[0].errorbar(g.index, g['mean'], yerr=g['std'], marker='o', capsize=3,
                       color=c, label=sc, ls='--' if sc == 'random' else '-')
    ax[0].set_xscale('log'); ax[0].set_xlabel('# observations'); ax[0].set_ylabel('AURC (lower = better)')
    ax[0].set_title(f'RQ7: abstention quality by score ({method})'); ax[0].legend(fontsize=8)

    # Does aleatoric abstention reject the items humans disagreed about?
    for sc, c in scols.items():
        s = sub[sub.score == sc]
        if s.empty or 'absH@0.7' not in s: continue
        g = s.groupby('n_obs')['absH@0.7'].mean()
        ax[1].plot(g.index, g.values, marker='s', color=c, label=sc,
                   ls='--' if sc == 'random' else '-')
    ax[1].set_xscale('log'); ax[1].set_xlabel('# observations')
    ax[1].set_ylabel('mean human entropy of abstained set')
    ax[1].set_title('who gets rejected at 70% coverage'); ax[1].legend(fontsize=8)

    plt.suptitle(f'Selective prediction — {dataset} / {model}', fontsize=13); plt.tight_layout()
    out = os.path.join(d, 'selective_prediction.png'); fig.savefig(out, dpi=170)
    print('saved', out)


def aggregate(variant=None):
    """Cross dataset/model headline table. `variant` (e.g. 'mean256') filters to
    figures/<ds>/<model>_<variant>/ dirs; None includes everything. The probe_summary
    JSON only stores the bare model name, so the run variant is recovered from the
    directory basename (model + optional _suffix) and reported as its own column."""
    rows = []
    for sj in glob.glob(os.path.join(FIGROOT, '*', '*', 'probe_summary.json')):
        s = json.load(open(sj))
        dirname = os.path.basename(os.path.dirname(sj))          # e.g. gemma_mean256
        var = dirname[len(s['model']):].lstrip('_') or 'last64'  # '' -> plain last-token/64
        if variant is not None and var != variant:
            continue
        for m in METHODS:
            if m in s:
                rows.append(dict(dataset=s['dataset'], model=s['model'], variant=var,
                                 K=s['K'], method=m, **s[m]))
    if not rows:
        print('no probe_summary.json found yet'); return
    df = pd.DataFrame(rows).sort_values(['variant', 'K', 'dataset', 'model', 'method'])
    print(f'\n=== cross dataset/model headline (best layer, largest n)'
          f"{' — variant=' + variant if variant else ''} ===")
    # Raw corrMI_entropy is reported next to the partial correlations on purpose: the
    # two components share a posterior (corrAleaMI), so only the partials separate the
    # disagreement axis from the evidence axis. See docs/DISAGREEMENT_STUDY.md RQ1b.
    cols = ['dataset', 'model', 'variant', 'K', 'method', 'soft_ce', 'tvd', 'acc',
            'corrAleaS_entropy', 'mono_alea', 'corrMIS_entropy',
            'pcorrAleaS_entropy_given_MI', 'pcorrMIS_entropy_given_alea', 'corrAleaMI']
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    out = os.path.join(FIGROOT, 'aggregate_summary.csv')
    df.to_csv(out, index=False)
    print('\nsaved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset'); ap.add_argument('--model'); ap.add_argument('--variant')
    a = ap.parse_args()
    if a.dataset and a.model:
        per_model(a.dataset, a.model)
    else:
        aggregate(a.variant)
