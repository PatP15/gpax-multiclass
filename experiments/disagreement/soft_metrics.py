"""Soft-label evaluation metrics for the disagreement study.

Two families:
  (A) DISAGREEMENT PROXIES phi(p): scalar measures of how spread a label distribution
      is -- our proxies for *aleatoric* (irreducible) uncertainty derived from the
      human annotation distribution. Several definitions so the headline result is
      not an artifact of one choice.
  (B) DISTRIBUTION-RECOVERY METRICS d(p_true, p_pred): how well a predicted class
      distribution matches the empirical human distribution (soft-label evaluation,
      following LeWiDi-2023 / Baan et al. 2022 / Nie et al. 2020).

All functions are pure numpy, accept either a single (K,) vector or a batch (N, K),
and return a scalar-per-row (N,) [or float for a single vector]. Natural log (nats)
throughout; JSD is normalised to [0,1].

References: Plank 2022 (human label variation); Nie et al. 2020 (ChaosNLI, JSD/KL);
Baan et al. 2022 (DistCE=TVD, EntCE, RankCS — calibration under disagreement);
Leonardelli et al. 2023 (LeWiDi, soft cross-entropy primary metric).
"""
import numpy as np

_EPS = 1e-12


def _2d(p):
    p = np.asarray(p, dtype=float)
    return p[None, :] if p.ndim == 1 else p


def _maybe_scalar(arr, was_1d):
    return float(arr[0]) if was_1d else arr


# ============================ (A) disagreement proxies ========================
# phi(p) in [0, .]; larger = more human disagreement = more aleatoric.

def entropy(p):
    """Shannon entropy H(p) = -sum p log p  (nats)."""
    q = _2d(p); was = np.asarray(p).ndim == 1
    h = -np.sum(q * np.log(q + _EPS), axis=1)
    return _maybe_scalar(h, was)


def norm_entropy(p):
    """Entropy normalised by log K -> [0,1]; comparable across different K."""
    q = _2d(p); was = np.asarray(p).ndim == 1
    K = q.shape[1]
    h = -np.sum(q * np.log(q + _EPS), axis=1) / np.log(K)
    return _maybe_scalar(h, was)


def one_minus_max(p):
    """1 - max_k p_k : the fraction of annotators NOT in the majority."""
    q = _2d(p); was = np.asarray(p).ndim == 1
    return _maybe_scalar(1.0 - q.max(1), was)


def gini(p):
    """Gini impurity 1 - sum p_k^2 = P(two random annotators pick different labels)."""
    q = _2d(p); was = np.asarray(p).ndim == 1
    return _maybe_scalar(1.0 - np.sum(q ** 2, axis=1), was)


def top2_margin(p):
    """1 - (p_(1) - p_(2)) : how close the top two classes are (1 = tie)."""
    q = _2d(p); was = np.asarray(p).ndim == 1
    s = np.sort(q, axis=1)[:, ::-1]
    m = 1.0 - (s[:, 0] - s[:, 1])
    return _maybe_scalar(m, was)


def bern_var(p):
    """Bernoulli variance p(1-p) for binary tasks (uses the positive-class prob)."""
    q = _2d(p); was = np.asarray(p).ndim == 1
    assert q.shape[1] == 2, 'bern_var is for K=2'
    p1 = q[:, 1]
    return _maybe_scalar(p1 * (1.0 - p1), was)


PROXIES = {'entropy': entropy, 'norm_entropy': norm_entropy, 'one_minus_max': one_minus_max,
           'gini': gini, 'top2_margin': top2_margin}   # bern_var added only for K=2


# ===================== (B) distribution-recovery metrics ======================
# d(p_true, p_pred): lower = better match to the human distribution.

def soft_ce(p_true, p_pred):
    """Soft cross-entropy H(p_true, p_pred) = -sum p_true log p_pred (nats). Primary
    LeWiDi metric. Minimised (= H(p_true)) iff p_pred == p_true."""
    a, b = _2d(p_true), _2d(p_pred); was = np.asarray(p_true).ndim == 1
    ce = -np.sum(a * np.log(b + _EPS), axis=1)
    return _maybe_scalar(ce, was)


def kl(p_true, p_pred):
    """KL(p_true || p_pred)."""
    a, b = _2d(p_true), _2d(p_pred); was = np.asarray(p_true).ndim == 1
    d = np.sum(a * (np.log(a + _EPS) - np.log(b + _EPS)), axis=1)
    return _maybe_scalar(d, was)


def jsd(p_true, p_pred):
    """Jensen-Shannon divergence, normalised to [0,1] (divide by ln 2). Symmetric."""
    a, b = _2d(p_true), _2d(p_pred); was = np.asarray(p_true).ndim == 1
    m = 0.5 * (a + b)
    def _kl(x, y):
        return np.sum(x * (np.log(x + _EPS) - np.log(y + _EPS)), axis=1)
    d = (0.5 * _kl(a, m) + 0.5 * _kl(b, m)) / np.log(2.0)
    return _maybe_scalar(np.clip(d, 0.0, 1.0), was)


def tvd(p_true, p_pred):
    """Total variation distance = 0.5 * sum |p_true - p_pred| in [0,1]. This is
    Baan et al.'s DistCE (a proper scoring rule, no binning, needs >1 annotation)."""
    a, b = _2d(p_true), _2d(p_pred); was = np.asarray(p_true).ndim == 1
    return _maybe_scalar(0.5 * np.sum(np.abs(a - b), axis=1), was)


distce = tvd  # alias (Baan et al. naming)


def entce(p_true, p_pred, signed=True):
    """Entropy Calibration Error: H(p_pred) - H(p_true) (Baan et al.). signed=False
    returns |.|. Near 0 = the model's overall uncertainty matches the humans'."""
    e = entropy(p_pred) - entropy(p_true)
    return e if signed else np.abs(e)


def rankcs(p_true, p_pred):
    """Simplified Ranking Calibration: top-1 agreement (argmax_pred == argmax_human),
    returned per row as 0/1 (mean it for the dataset-level score)."""
    a, b = _2d(p_true), _2d(p_pred); was = np.asarray(p_true).ndim == 1
    agree = (a.argmax(1) == b.argmax(1)).astype(float)
    return _maybe_scalar(agree, was)


def _self_check():
    rng = np.random.RandomState(0)
    P = rng.dirichlet(np.ones(5), size=200)        # random distributions
    Q = rng.dirichlet(np.ones(5), size=200)
    # bounds
    assert np.all((tvd(P, Q) >= 0) & (tvd(P, Q) <= 1)), 'TVD out of [0,1]'
    assert np.all((jsd(P, Q) >= 0) & (jsd(P, Q) <= 1)), 'JSD out of [0,1]'
    assert np.all((norm_entropy(P) >= 0) & (norm_entropy(P) <= 1)), 'norm_entropy out of [0,1]'
    # identities at p==q
    assert np.allclose(tvd(P, P), 0) and np.allclose(jsd(P, P), 0), 'distance to self != 0'
    assert np.allclose(soft_ce(P, P), entropy(P)), 'soft_ce(p,p) != H(p)'
    assert np.allclose(kl(P, P), 0), 'KL(p,p) != 0'
    assert np.allclose(entce(P, P), 0), 'EntCE(p,p) != 0'
    # one-hot -> zero disagreement on every proxy
    oh = np.eye(5)[rng.randint(0, 5, size=50)]
    for name, fn in PROXIES.items():
        assert np.allclose(fn(oh), 0, atol=1e-6), f'{name} nonzero on one-hot'
    # uniform -> maximal norm_entropy and gini
    u = np.full((1, 5), 0.2)
    assert np.isclose(norm_entropy(u), 1.0), 'uniform norm_entropy != 1'
    assert np.isclose(gini(u), 1 - 0.2), 'uniform gini wrong'
    # single-vector API
    assert isinstance(entropy(np.array([0.5, 0.5])), float)
    assert isinstance(tvd(np.array([0.5, 0.5]), np.array([0.4, 0.6])), float)
    print('soft_metrics self-check: PASS')


if __name__ == '__main__':
    _self_check()
