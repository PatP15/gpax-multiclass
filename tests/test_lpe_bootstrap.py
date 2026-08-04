"""LPE bootstrap correctness tests (docs/BUGS.md B11 + B20).

Two defects this guards against:
  B20 — the bootstrap used the unseeded global `np.random`, so LPE results were
        not reproducible and the callers' seed loops did not control them.
  B11 — a resample could omit a class, whose probability column was then padded
        to exactly zero, biasing the ensemble's entropy / mutual information.

Run with `python tests/test_lpe_bootstrap.py` or `pytest tests/test_lpe_bootstrap.py`.
"""
import numpy as np
from GPax.probing import probabilistic_probe as pp
from GPax.probing import probabilistic_probe_multiclass as ppm


def _data(K=4, n=40, d=6, seed=0):
    """Class-imbalanced multiclass data: the rarest class has 2 members, so an
    unguarded bootstrap omits it often (p = (1-2/n)^n ~ 13% per member at n=40)."""
    rng = np.random.RandomState(seed)
    counts = [2, 4] + [(n - 6) // (K - 2)] * (K - 2)
    X, y = [], []
    for k, c in enumerate(counts):
        X.append(rng.randn(c, d) + 3.0 * np.eye(K, d)[k])
        y.append(np.full(c, k))
    return np.vstack(X), np.concatenate(y).astype(int)


def test_lpe_multiclass_is_reproducible():
    """B20: same rng -> bit-identical measures; different rng -> different draws."""
    X, y = _data()
    Xq = np.random.RandomState(1).randn(15, X.shape[1])

    a = ppm.lpe_multiclass(Xq, X, y, num_classes=4, repeats=20, rng=0)
    b = ppm.lpe_multiclass(Xq, X, y, num_classes=4, repeats=20, rng=0)
    c = ppm.lpe_multiclass(Xq, X, y, num_classes=4, repeats=20, rng=1)

    for key in ('categorical_mu', 'Alea', 'information_gain'):
        np.testing.assert_array_equal(np.array(a[key]), np.array(b[key]),
                                      err_msg=f'rng=0 twice differed on {key}')
    assert not np.allclose(np.array(a['categorical_mu']), np.array(c['categorical_mu'])), \
        'rng=0 and rng=1 produced identical ensembles — the seed is being ignored'
    print('  B20 reproducibility: PASS')


def test_lpe_binary_is_reproducible():
    """B20, binary path."""
    rng = np.random.RandomState(2)
    X = rng.randn(30, 5)
    y = (X[:, 0] > 0).astype(int)
    Xq = rng.randn(12, 5)

    a = pp.lpe(Xq, X, y, repeats=20, rng=7)
    b = pp.lpe(Xq, X, y, repeats=20, rng=7)
    c = pp.lpe(Xq, X, y, repeats=20, rng=8)

    np.testing.assert_array_equal(np.array(a['Judged probability']),
                                  np.array(b['Judged probability']),
                                  err_msg='binary lpe not reproducible at fixed rng')
    assert not np.allclose(np.array(a['Judged probability']), np.array(c['Judged probability'])), \
        'binary lpe ignored the seed'
    print('  B20 reproducibility (binary): PASS')


def test_no_zero_probability_columns():
    """B11: every class present in y_observed must get nonzero probability mass in
    every ensemble member, so the entropy/MI estimates are not biased downward."""
    K = 4
    X, y = _data(K=K)
    assert len(np.unique(y)) == K, 'fixture must contain all K classes'
    Xq = np.random.RandomState(3).randn(20, X.shape[1])

    # 40 members x a rare class of size 2: the pre-fix code hit a zero column with
    # near-certainty over this many draws.
    m = ppm.lpe_multiclass(Xq, X, y, num_classes=K, repeats=40, rng=0)
    mu = np.array(m['categorical_mu'])
    assert mu.shape == (len(Xq), K), f'unexpected shape {mu.shape}'
    assert np.all(mu > 0), f'zero-probability column survived: min={mu.min():.3g}'
    np.testing.assert_allclose(mu.sum(1), 1.0, atol=1e-5, err_msg='rows not normalized')
    assert np.all(np.isfinite(np.array(m['Alea']))), 'non-finite aleatoric'
    assert np.all(np.isfinite(np.array(m['information_gain']))), 'non-finite MI'
    print('  B11 class coverage: PASS')


def test_absent_class_gets_smoothed_not_zero():
    """A class absent from y_observed entirely (K declared larger than observed)
    must receive the Laplace-smoothed floor, not exactly zero."""
    X, y = _data(K=3)
    Xq = np.random.RandomState(4).randn(10, X.shape[1])
    m = ppm.lpe_multiclass(Xq, X, y, num_classes=5, repeats=10, rng=0)   # classes 3,4 unobserved
    mu = np.array(m['categorical_mu'])
    assert mu.shape[1] == 5
    assert np.all(mu[:, 3:] > 0), 'unobserved classes still land at exactly zero'
    assert np.all(mu[:, 3:] < mu[:, :3].max()), 'unobserved classes should carry little mass'
    print('  B11 smoothing of unobserved classes: PASS')


if __name__ == '__main__':
    print('=== LPE bootstrap tests (B11 + B20) ===')
    test_lpe_multiclass_is_reproducible()
    test_lpe_binary_is_reproducible()
    test_no_zero_probability_columns()
    test_absent_class_gets_smoothed_not_zero()
    print('ALL PASS')
