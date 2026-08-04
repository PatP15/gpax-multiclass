"""Selective prediction / abstention for the disagreement study.

RQ1 and RQ3 show that GPP's aleatoric tracks *human disagreement* and its mutual
information tracks *evidence*. Those are correlations. This script asks whether the
decomposition is USEFUL: does routing by the right component actually buy accuracy
on the items you keep?

The design makes the two-axis claim falsifiable as a decision rule:

  * evidence-poor regime (n_obs small) -- the probe's errors are mostly reducible,
    so abstaining by MUTUAL INFORMATION should win: it flags "I have not seen
    enough data near this point".
  * evidence-rich regime (n_obs large) -- what remains is irreducible human
    ambiguity, so abstaining by ALEATORIC should win, and the items it rejects
    should be the ones humans themselves disagreed about (we measure exactly that:
    the mean human entropy of the abstained set).

If the ordering flips between the two regimes, the decomposition is doing real work
that a single scalar confidence cannot do. If neither beats total entropy anywhere,
the decomposition is decoration -- and we would report that.

Runs on the committed emb npz, at the best layer per method as chosen by
step2_probe.py (reads its probe_summary.json), so this adds one fit per
(method, seed, n_obs) rather than a whole layer sweep.

Usage:
  python experiments/disagreement/step2_selective.py --dataset lewidi_md --model gemma --variant mean256
"""
import os, sys, argparse, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
import soft_metrics as sm
from step2_probe import predict, _nmc, NOBS, SEEDS, METHODS

COVERAGES = [0.5, 0.7, 0.9]

# Which abstention scores each method can supply. 'alea' and 'mi' are the two
# decomposed components; 'total' is the single-scalar confidence a non-Bayesian
# probe would use (the baseline the decomposition has to beat); 'msp' is the
# classic max-softmax-probability detector; 'random' is the no-information floor.
SCORES = ['alea', 'mi', 'total', 'msp', 'random']


def risk_coverage(err, score, coverages=COVERAGES):
    """Sort by ascending score (= most confident first) and accumulate error.

    Returns (aurc, {coverage: risk}). `err` is a 0/1 per-item error indicator,
    `score` the abstention score (higher = abstain sooner). AURC is the mean risk
    over every prefix, so lower is better and a perfect ranker approaches 0 while
    an uninformative one sits at the overall error rate.
    """
    err = np.asarray(err, dtype=float)
    order = np.argsort(score, kind='stable')
    cum = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    aurc = float(cum.mean())
    at = {c: float(cum[max(0, int(np.ceil(c * len(err))) - 1)]) for c in coverages}
    return aurc, at


def abstained_human_entropy(human, score, coverages=COVERAGES):
    """Mean human-annotation entropy of the items REJECTED at each coverage.

    This is the test that aleatoric abstention is rejecting genuinely ambiguous
    items rather than merely hard ones: a disagreement-aware score should push the
    high-human-entropy items into the abstained set.
    """
    human = np.asarray(human, dtype=float)
    order = np.argsort(score, kind='stable')          # kept first, abstained last
    out = {}
    for c in coverages:
        k = max(0, int(np.ceil(c * len(human))))
        rej = order[k:]
        out[c] = float(human[rej].mean()) if len(rej) else float('nan')
    return out


def run(dataset, model, variant=None, synth=False, prompt=False):
    if variant:
        fn, suffix = f'emb_{variant}.npz', f'_{variant}'
    elif synth:
        fn, suffix = 'emb_synth.npz', '_synth'
    elif prompt:
        fn, suffix = 'emb_prompted.npz', '_prompted'
    else:
        fn, suffix = 'emb.npz', ''
    d = np.load(os.path.join(HERE, 'data', dataset, model, fn))
    K = int(d['K']); soft_te = d['soft_test']; hard_te = d['hard_test']; nmc = _nmc(K)
    human = sm.entropy(soft_te)

    outdir = os.path.join(REPO, 'experiments', 'disagreement', 'figures', dataset, model + suffix)
    sj = os.path.join(outdir, 'probe_summary.json')
    if not os.path.exists(sj):
        raise SystemExit(f'need {sj} for the best layer per method — run step2_probe.py first')
    summary = json.load(open(sj))
    best_layer = {m: summary[m]['best_layer'] for m in METHODS if m in summary}
    print(f'{dataset}/{model}{suffix}: K={K}, best layers {best_layer}', flush=True)

    rows = []
    for m, li in best_layer.items():
        Xtr_all, Xte = d[f'X_train_L{li}'], d[f'X_test_L{li}']
        ytr_all = d['hard_train']
        for seed in SEEDS:
            rng = np.random.RandomState(100 + seed)
            for n in NOBS:
                if n > len(Xtr_all):
                    continue
                io = rng.choice(len(Xtr_all), n, replace=False)
                Xo, yo = Xtr_all[io], ytr_all[io]
                if len(np.unique(yo)) < 2:
                    continue
                try:
                    P, alea, mi = predict(m, Xo, yo, Xte, K, nmc, seed=seed)
                except Exception as e:
                    print('  cell dropped:', m, seed, n, repr(e)[:80]); continue
                err = (P.argmax(1) != hard_te).astype(float)
                cand = {'alea': alea, 'mi': mi, 'total': np.asarray(sm.entropy(P), dtype=float),
                        'msp': 1.0 - P.max(1),
                        'random': np.random.RandomState(1000 + seed).rand(len(P))}
                for sname in SCORES:
                    s = np.asarray(cand[sname], dtype=float)
                    if not np.all(np.isfinite(s)) or np.std(s) < 1e-12:
                        continue          # e.g. LP-temp has no MI
                    aurc, risk = risk_coverage(err, s)
                    absH = abstained_human_entropy(human, s)
                    row = dict(dataset=dataset, model=model, layer=int(li), method=m, score=sname,
                               seed=seed, n_obs=n, K=K, full_risk=float(err.mean()), aurc=aurc)
                    for c in COVERAGES:
                        row[f'risk@{c}'] = risk[c]
                        row[f'absH@{c}'] = absH[c]
                    rows.append(row)
        print(f'  {m} done (L{li})', flush=True)

    if not rows:
        raise SystemExit('no rows produced')
    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, 'selective_raw.csv'), index=False)

    # ---- the headline: does the winning component flip between the two regimes? ----
    out = {'dataset': dataset, 'model': model, 'K': K}
    n_lo, n_hi = min(df.n_obs), max(df.n_obs)
    print(f'\n===== {dataset}/{model}{suffix} selective prediction '
          f'(AURC, lower=better; abstained human entropy at 70% coverage) =====')
    for m in best_layer:
        sub = df[df.method == m]
        if sub.empty:
            continue
        out[m] = {}
        for n in (n_lo, n_hi):
            s = sub[sub.n_obs == n]
            if s.empty:
                continue
            g = s.groupby('score')[['aurc', 'risk@0.7', 'absH@0.7', 'full_risk']].mean()
            out[m][f'n{n}'] = {sc: {k: float(v) for k, v in g.loc[sc].items()} for sc in g.index}
            best = g.aurc.idxmin()
            parts = ' '.join(f'{sc}={g.aurc[sc]:.3f}' for sc in g.index)
            print(f'  {m:11s} n={n:5d} full_risk={g["full_risk"].iloc[0]:.3f} | AURC {parts}'
                  f' | best={best} | absH@0.7 alea='
                  f'{g["absH@0.7"].get("alea", float("nan")):.3f}'
                  f' mi={g["absH@0.7"].get("mi", float("nan")):.3f}')
        # the dissociation, stated as a single verdict per method
        lo = out[m].get(f'n{n_lo}', {}); hi = out[m].get(f'n{n_hi}', {})
        if 'alea' in lo and 'mi' in lo and 'alea' in hi and 'mi' in hi:
            out[m]['flip'] = dict(
                mi_wins_low_n=bool(lo['mi']['aurc'] < lo['alea']['aurc']),
                alea_wins_high_n=bool(hi['alea']['aurc'] < hi['mi']['aurc']),
                alea_beats_total_high_n=bool(hi['alea']['aurc'] < hi['total']['aurc']))
            print(f'    -> flip: MI wins at n={n_lo}: {out[m]["flip"]["mi_wins_low_n"]}; '
                  f'alea wins at n={n_hi}: {out[m]["flip"]["alea_wins_high_n"]}; '
                  f'alea beats total entropy at n={n_hi}: {out[m]["flip"]["alea_beats_total_high_n"]}')
    json.dump(out, open(os.path.join(outdir, 'selective_summary.json'), 'w'), indent=2)
    print(f'\nsaved {outdir}/selective_raw.csv + selective_summary.json')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--variant', default=None)
    ap.add_argument('--synth', action='store_true'); ap.add_argument('--prompt', action='store_true')
    a = ap.parse_args()
    run(a.dataset, a.model, a.variant, a.synth, a.prompt)
