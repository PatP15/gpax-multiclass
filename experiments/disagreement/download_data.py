"""Fetch the human-disagreement datasets (no wget / no pip required).

Run from the repo root:  python experiments/disagreement/download_data.py

- ChaosNLI : committed in the repo (data/chaosNLI_v1.0/); nothing to download.
             Original source: https://github.com/easonnie/ChaosNLI
- GoEmotions: downloads the HuggingFace 'raw' parquet (per-rater) via urllib.
             To LOAD it you also need pyarrow:  pip install pyarrow
- LeWiDi   : a git submodule; fetch with
             git submodule update --init experiments/disagreement/lewidi
"""
import os, sys, urllib.request, socket

HERE = os.path.dirname(os.path.abspath(__file__))
socket.setdefaulttimeout(120)


def fetch_goemotions():
    out = os.path.join(HERE, 'data', 'goemotions')
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, 'go_emotions_raw.parquet')
    if os.path.exists(dst):
        print(f'GoEmotions: already present ({os.path.getsize(dst)} bytes)')
        return
    url = ('https://huggingface.co/datasets/google-research-datasets/go_emotions/'
           'resolve/main/raw/train-00000-of-00001.parquet')
    print('GoEmotions: downloading per-rater parquet from HuggingFace ...')
    urllib.request.urlretrieve(url, dst)
    print(f'  saved {dst} ({os.path.getsize(dst)} bytes). To load it: pip install pyarrow')


def check_chaosnli():
    p = os.path.join(HERE, 'data', 'chaosNLI_v1.0', 'chaosNLI_snli.jsonl')
    print('ChaosNLI:', 'present (committed)' if os.path.exists(p) else
          'MISSING — see https://github.com/easonnie/ChaosNLI')


def check_lewidi():
    p = os.path.join(HERE, 'lewidi', 'LeWiDi_2-2023')
    if os.path.isdir(p) and os.listdir(p):
        print('LeWiDi: submodule present (all 3 editions under lewidi/).')
    else:
        print('LeWiDi: submodule not initialised. Run: '
              'git submodule update --init experiments/disagreement/lewidi')


if __name__ == '__main__':
    check_chaosnli()
    check_lewidi()
    fetch_goemotions()
    print('\nDone. Quick check:  python experiments/disagreement/data_loaders.py')
