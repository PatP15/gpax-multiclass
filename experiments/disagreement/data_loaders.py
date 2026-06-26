"""Loaders for the human-disagreement datasets used to validate the GPP probe's
ALEATORIC uncertainty against ground-truth human label distributions.

Each loader returns a uniform record list so the downstream experiment is
dataset-agnostic. A record is:
    {
      'text':       str (or dict for NLI premise/hypothesis),
      'soft_label': np.ndarray (K,)  -- empirical human label distribution,
      'hard_label': int             -- majority label,
      'n_annot':    int             -- number of annotators,
      'entropy':    float           -- entropy of soft_label (the aleatoric target),
    }

These are the substrates for the experiment "does the probe's predicted
distribution recover the human label distribution?" (soft cross-entropy / JSD /
TVD / Baan-et-al. instance-level calibration). See README.md.

Datasets (all set up under experiments/disagreement/):
  - ChaosNLI  : data/chaosNLI_v1.0/*.jsonl              (committed; ~2.3 MB)
  - LeWiDi    : lewidi/LeWiDi_2-2023/<task>_dataset/*.json  (git submodule)
  - GoEmotions: data/goemotions/go_emotions_raw.parquet (download_data.py; needs pyarrow)
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- ChaosNLI ----
# 3-class NLI (entailment / neutral / contradiction), 100 human annotations each.
_CHAOS_LABELS = {'e': 0, 'n': 1, 'c': 2}          # SNLI/MNLI
_CHAOS_DIR = os.path.join(HERE, 'data', 'chaosNLI_v1.0')


def load_chaosnli(subset='snli'):
    """subset in {'snli','mnli','alphanli'}. Returns uniform records.

    For snli/mnli the 3 classes are entailment/neutral/contradiction; for
    alphanli they are the 2 hypotheses (binary).
    """
    fn = {'snli': 'chaosNLI_snli.jsonl', 'mnli': 'chaosNLI_mnli_m.jsonl',
          'alphanli': 'chaosNLI_alphanli.jsonl'}[subset]
    path = os.path.join(_CHAOS_DIR, fn)
    recs = []
    for line in open(path):
        d = json.loads(line)
        lc = d['label_counter']
        if subset == 'alphanli':
            keys = ['1', '2']
        else:
            keys = ['e', 'n', 'c']
        counts = np.array([lc.get(k, 0) for k in keys], dtype=float)
        soft = counts / counts.sum()
        ex = d['example']
        text = ({'premise': ex['premise'], 'hypothesis': ex['hypothesis']}
                if 'premise' in ex else
                {'obs1': ex['obs1'], 'obs2': ex['obs2'], 'hyp1': ex['hyp1'], 'hyp2': ex['hyp2']})
        recs.append({'text': text, 'soft_label': soft, 'hard_label': int(soft.argmax()),
                     'n_annot': int(counts.sum()), 'entropy': float(d['entropy'])})
    return recs


# ------------------------------------------------------------------ LeWiDi ----
# Binary subjective tasks; each record ships a per-annotator 'annotations' string
# and a 'soft_label' dict {'0':p0,'1':p1}. Edition 2 (SemEval-2023) by default.
_LEWIDI_TASKS = {'HS-Brexit', 'ArMIS', 'ConvAbuse', 'MD-Agreement'}


def load_lewidi(task='MD-Agreement', split='train', edition='LeWiDi_2-2023'):
    """task in {HS-Brexit, ArMIS, ConvAbuse, MD-Agreement}; split in {train,dev,test}."""
    assert task in _LEWIDI_TASKS, f'task must be one of {_LEWIDI_TASKS}'
    path = os.path.join(HERE, 'lewidi', edition, f'{task}_dataset', f'{task}_{split}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Init the submodule: '
            f'git submodule update --init experiments/disagreement/lewidi')
    d = json.load(open(path))
    recs = []
    for it in d.values():
        sl = it.get('soft_label', {})
        soft = np.array([float(sl.get('0', 0.0)), float(sl.get('1', 0.0))])
        if soft.sum() > 0:
            soft = soft / soft.sum()
        recs.append({'text': it.get('text'), 'soft_label': soft,
                     'hard_label': int(it.get('hard_label', soft.argmax())),
                     'n_annot': int(it.get('number of annotations', 0)),
                     'entropy': float(-(soft[soft > 0] * np.log(soft[soft > 0])).sum())})
    return recs


# -------------------------------------------------------------- GoEmotions ----
# 28-way (27 emotions + neutral), up to 5 raters/comment. The 'raw' parquet has
# one row per (comment, rater); we group by comment id to get the vote distribution.
# Requires pyarrow/fastparquet to read parquet (pip install pyarrow).
_GOEMO_PARQUET = os.path.join(HERE, 'data', 'goemotions', 'go_emotions_raw.parquet')
_GOEMO_META = {'text', 'id', 'author', 'subreddit', 'link_id', 'parent_id',
               'created_utc', 'rater_id', 'example_very_unclear'}


def load_goemotions(min_annot=3):
    """Group the per-rater 'raw' parquet into per-comment soft labels over 28 emotions.

    Returns records with soft_label of shape (28,). Requires pyarrow:
        pip install pyarrow
    """
    try:
        import pandas as pd
        df = pd.read_parquet(_GOEMO_PARQUET)
    except ImportError as e:
        raise ImportError(
            'Reading the GoEmotions parquet needs pyarrow: `pip install pyarrow`. '
            'Or use the datasets-server JSON API (see download_data.py).') from e
    emo = [c for c in df.columns if c not in _GOEMO_META]
    recs = []
    for cid, sub in df.groupby('id'):
        if len(sub) < min_annot:
            continue
        counts = sub[emo].sum(0).values.astype(float)
        if counts.sum() == 0:
            continue
        soft = counts / counts.sum()
        nz = soft[soft > 0]
        recs.append({'text': sub['text'].iloc[0], 'soft_label': soft,
                     'hard_label': int(soft.argmax()), 'n_annot': int(len(sub)),
                     'entropy': float(-(nz * np.log(nz)).sum()), 'emotions': emo})
    return recs


if __name__ == '__main__':
    # quick smoke: print one clear and one ambiguous record per available dataset
    def peek(name, recs):
        recs = [r for r in recs if r['n_annot'] > 0]
        recs.sort(key=lambda r: r['entropy'])
        print(f'\n=== {name}: {len(recs)} records ===')
        for tag, r in [('clearest', recs[0]), ('most ambiguous', recs[-1])]:
            t = r['text'] if isinstance(r['text'], str) else r['text']
            print(f'  [{tag}] entropy={r["entropy"]:.3f} soft={np.round(r["soft_label"],2)} n={r["n_annot"]}')
            print(f'         text={t}')
    peek('ChaosNLI/snli', load_chaosnli('snli'))
    peek('LeWiDi/MD-Agreement', load_lewidi('MD-Agreement', 'train'))
    try:
        peek('GoEmotions', load_goemotions())
    except ImportError as e:
        print('\nGoEmotions: skipped —', e)
