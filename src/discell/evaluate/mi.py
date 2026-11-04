"""Nearest-neighbor mutual information between a label and a latent."""

from __future__ import annotations

import numpy as np

#: Neighbors for the estimator.
MI_K = 5


def mi_discrete_continuous(x: np.ndarray, y: np.ndarray, k: int = MI_K,
                           rng=None) -> float:
    """``I(y; x)`` for a discrete *y* and a continuous vector *x*, in nats (Ross 2014).

    For each point the distance to its k-th neighbor within its own class sets
    a ball (max norm) and the count of all points in the ball gives the
    correction: ``I = psi(N) - <psi(N_c)> + psi(k) - <psi(m)>``. Ties are broken
    by a tiny jitter drawn from *rng*. Clipped at 0.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x = np.asarray(x, float)
    x = x.reshape(len(x), -1)
    rng = np.random.default_rng(0) if rng is None else rng
    x = x + 1e-10 * rng.standard_normal(x.shape) * (x.std(axis=0) + 1e-12)
    y = np.asarray(y)
    n = len(x)
    full = cKDTree(x)
    radius = np.zeros(n)
    n_class = np.zeros(n)
    k_eff = np.zeros(n)
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        n_class[rows] = len(rows)
        kk = min(k, len(rows) - 1)
        if kk < 1:
            radius[rows], k_eff[rows] = np.inf, np.nan
            continue
        d = cKDTree(x[rows]).query(x[rows], k=kk + 1, p=np.inf)[0][:, kk]
        radius[rows], k_eff[rows] = d, kk
    good = np.isfinite(radius) & np.isfinite(k_eff)
    # one ball per point: scipy takes a per-point radius vector
    m = np.asarray(full.query_ball_point(x[good], r=radius[good] * (1 - 1e-12),
                                         p=np.inf, return_length=True))
    value = (digamma(n) - digamma(n_class[good]).mean()
             + digamma(k_eff[good]).mean() - digamma(np.maximum(m, 1)).mean())
    return float(max(value, 0.0))


def within_type_permutation(y: np.ndarray, t: np.ndarray, rng) -> np.ndarray:
    """*y* shuffled among cells of the same type *t*.

    The MI floor this builds is what a latent gets from cell type alone.
    """
    out = y.copy()
    for g in np.unique(t):
        rows = np.flatnonzero(t == g)
        out[rows] = y[rng.permutation(rows)]
    return out


def ross_terms(x: np.ndarray, y: np.ndarray, k: int = MI_K, rng=None) -> np.ndarray:
    """The per-point terms of :func:`mi_discrete_continuous` (NaN where a point's
    class is too small): their mean is the estimate before its clip at 0. Draws
    the jitter from *rng* exactly as the estimator does, so a tile resampling
    of the estimate is a weighted mean of these terms."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x = np.asarray(x, float).reshape(len(x), -1)
    rng = np.random.default_rng(0) if rng is None else rng
    x = x + 1e-10 * rng.standard_normal(x.shape) * (x.std(axis=0) + 1e-12)
    y = np.asarray(y)
    n = len(x)
    full = cKDTree(x)
    radius, n_class, k_eff = np.zeros(n), np.zeros(n), np.zeros(n)
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        n_class[rows] = len(rows)
        kk = min(k, len(rows) - 1)
        if kk < 1:
            radius[rows], k_eff[rows] = np.inf, np.nan
            continue
        d = cKDTree(x[rows]).query(x[rows], k=kk + 1, p=np.inf)[0][:, kk]
        radius[rows], k_eff[rows] = d, kk
    good = np.isfinite(radius) & np.isfinite(k_eff)
    m = np.asarray(full.query_ball_point(x[good], r=radius[good] * (1 - 1e-12),
                                         p=np.inf, return_length=True))
    terms = np.full(n, np.nan)
    terms[good] = (digamma(n) - digamma(n_class[good]) + digamma(k_eff[good])
                   - digamma(np.maximum(m, 1)))
    return terms
