"""Multi-layer LLM embedding extractor for the disagreement datasets.

For one (dataset, model): builds a probe train/test split, embeds every text at
~6 evenly-spaced transformer layers (last-token pooled), PCA-reduces each layer to
64-D (fit on train), and saves one emb.npz with per-layer X_train/X_test plus the
hard labels (probe trains on these) and the human SOFT labels for the test set
(what we score against). Reuses annomi_common.load_model / its pooling logic.

Usage (cluster GPU):
  python experiments/disagreement/step1_embed.py --dataset chaosnli_snli --model gemma
  python experiments/disagreement/step1_embed.py --dataset lewidi_md     --model qwen
  python experiments/disagreement/step1_embed.py --dataset goemotions    --model llama
Local pipeline check (no LLM, random embeddings that encode the hard label):
  python experiments/disagreement/step1_embed.py --dataset chaosnli_snli --model gemma --synth
"""
import os, sys, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import data_loaders as dl

N_LAYERS_SAMPLED = 6
PCA_DIM = 64


# ---- dataset dispatch: -> (records, K, has_split) with uniform text strings ----
# Two encodings: raw (just the text) and prompted (a task instruction so the
# last-token embedding captures the model's task-relevant judgment, like AnnoMI).
def _nli_text(t, prompt):
    if prompt:
        return (f"Does the premise entail the hypothesis, contradict it, or neither?\n"
                f"Premise: {t['premise']}\nHypothesis: {t['hypothesis']}\nAnswer:")
    return f"Premise: {t['premise']}\nHypothesis: {t['hypothesis']}"


def _wrap(text, name, prompt):
    if not prompt:
        return text
    if name == 'lewidi_md':
        return f"Is the following social media post offensive?\nPost: {text}\nAnswer:"
    if name == 'goemotions':
        return f"What emotion does this comment express?\nComment: {text}\nAnswer:"
    return text


def get_dataset(name, subsample=8000, seed=0, prompt=False):
    """Returns dict with train/test lists of {emb_text, soft_label, hard_label, n_annot, domain}."""
    if name == 'chaosnli_snli':
        recs = dl.load_chaosnli('snli'); K = 3
        for r in recs:
            r['emb_text'] = _nli_text(r['text'], prompt); r['domain'] = 0
        tr, te = _split(recs, seed)
    elif name == 'lewidi_md':
        K = 2
        tr = dl.load_lewidi('MD-Agreement', 'train'); te = dl.load_lewidi('MD-Agreement', 'test')
        for r in tr + te:
            r['emb_text'] = _wrap(r['text'], name, prompt); r['domain'] = 0
    elif name == 'goemotions':
        recs = dl.load_goemotions(min_annot=4); K = 28
        rng = np.random.RandomState(seed)
        if len(recs) > subsample:
            recs = [recs[i] for i in rng.choice(len(recs), subsample, replace=False)]
        for r in recs:
            r['emb_text'] = _wrap(r['text'], name, prompt); r['domain'] = 0
        tr, te = _split(recs, seed)
    else:
        raise ValueError(name)
    return {'train': tr, 'test': te, 'K': K, 'name': name}


def _split(recs, seed, test_frac=0.3):
    y = np.array([r['hard_label'] for r in recs])
    idx = np.arange(len(recs))
    tr_i, te_i = train_test_split(idx, test_size=test_frac, random_state=seed,
                                  stratify=y if len(np.unique(y)) > 1 else None)
    return [recs[i] for i in tr_i], [recs[i] for i in te_i]


# ----------------------------- embedding backends -----------------------------
def embed_synth(texts, hard, K, n_layers=N_LAYERS_SAMPLED, seed=0):
    """Random embeddings that weakly encode the hard label (for pipeline checks)."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_layers, K, PCA_DIM) * 2.0
    out = {}
    hard = np.asarray(hard)
    for li in range(n_layers):
        X = centers[li][hard] + rng.randn(len(hard), PCA_DIM)
        out[li] = X.astype(np.float32)
    return out, list(range(n_layers))


def embed_llm(texts, model_type, n_layers=N_LAYERS_SAMPLED, batch_size=8, max_length=256, pool='last'):
    """Real extraction: last-token or masked-mean pooling at ~n_layers layers."""
    import torch
    sys.path.insert(0, os.path.join(REPO, 'experiments', 'annomi'))
    os.environ.setdefault('ANNOMI_MODEL_TYPE', model_type)
    import annomi_common as common
    model, tok = common.load_model()
    if hasattr(model.config, 'output_hidden_states'):
        model.config.output_hidden_states = True
    # decide layer indices once (hidden_states has num_layers+1 entries)
    probe = tok(['hi'], return_tensors='pt', padding=True).to(model.device)
    with torch.no_grad():
        nh = len(model(**probe, output_hidden_states=True).hidden_states)
    layer_idx = [int(round(f * (nh - 1))) for f in np.linspace(0, 1, n_layers)]
    layer_idx = sorted(set(layer_idx))
    by_layer = {li: [] for li in layer_idx}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inp = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            hs = model(**inp, output_hidden_states=True).hidden_states
        last = inp.attention_mask.sum(1) - 1
        ar = torch.arange(len(batch))
        mask = inp.attention_mask.unsqueeze(-1).float()   # (B,T,1) for mean pooling
        for li in layer_idx:
            if pool == 'mean':
                pooled = ((hs[li] * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
            else:
                pooled = hs[li][ar, last].float().cpu().numpy()
            by_layer[li].append(pooled)
        if (i // batch_size) % 20 == 0:
            print(f'  embedded {i+len(batch)}/{len(texts)}', flush=True)
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {li: np.concatenate(v, 0) for li, v in by_layer.items()}, layer_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['chaosnli_snli', 'lewidi_md', 'goemotions'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--synth', action='store_true')
    ap.add_argument('--prompt', action='store_true', help='wrap texts in a task instruction')
    ap.add_argument('--pool', choices=['last', 'mean'], default='last')
    ap.add_argument('--pca-dim', type=int, default=PCA_DIM, dest='pca_dim')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ds = get_dataset(args.dataset, seed=args.seed, prompt=args.prompt)
    tr, te, K = ds['train'], ds['test'], ds['K']
    print(f"{args.dataset} / {args.model}: train={len(tr)} test={len(te)} K={K}", flush=True)

    tr_txt = [r['emb_text'] for r in tr]; te_txt = [r['emb_text'] for r in te]
    hard_tr = np.array([r['hard_label'] for r in tr]); hard_te = np.array([r['hard_label'] for r in te])
    soft_te = np.stack([r['soft_label'] for r in te]).astype(np.float32)
    nann_te = np.array([r['n_annot'] for r in te])
    dom_tr = np.array([r.get('domain', 0) for r in tr]); dom_te = np.array([r.get('domain', 0) for r in te])

    backend = embed_synth if args.synth else embed_llm
    if args.synth:
        raw_tr, layer_idx = embed_synth(tr_txt, hard_tr, K, seed=args.seed)
        raw_te, _ = embed_synth(te_txt, hard_te, K, seed=args.seed + 1)
    else:
        all_txt = tr_txt + te_txt
        raw_all, layer_idx = embed_llm(all_txt, args.model, pool=args.pool)
        raw_tr = {li: raw_all[li][:len(tr_txt)] for li in layer_idx}
        raw_te = {li: raw_all[li][len(tr_txt):] for li in layer_idx}

    out = {'K': K, 'layers': np.array(layer_idx), 'hard_train': hard_tr, 'hard_test': hard_te,
           'soft_test': soft_te, 'n_annot_test': nann_te, 'domain_train': dom_tr, 'domain_test': dom_te}
    for li in layer_idx:
        pca = PCA(n_components=min(args.pca_dim, raw_tr[li].shape[1], len(raw_tr[li])))
        Xtr = pca.fit_transform(np.nan_to_num(raw_tr[li]))
        Xte = pca.transform(np.nan_to_num(raw_te[li]))
        out[f'X_train_L{li}'] = Xtr.astype(np.float32)
        out[f'X_test_L{li}'] = Xte.astype(np.float32)

    outdir = os.path.join(HERE, 'data', args.dataset, args.model)
    os.makedirs(outdir, exist_ok=True)
    if args.synth:
        tag = 'emb_synth'
    elif args.prompt:
        tag = 'emb_prompted'
    elif args.pool != 'last' or args.pca_dim != PCA_DIM:
        tag = f'emb_{args.pool}{args.pca_dim}'
    else:
        tag = 'emb'
    path = os.path.join(outdir, f'{tag}.npz')
    np.savez(path, **out)
    print(f"saved {path}  layers={layer_idx}", flush=True)


if __name__ == '__main__':
    main()
