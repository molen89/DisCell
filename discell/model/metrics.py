#!/usr/bin/env python3
"""The diagnostics of spec 7.10, each as a function of collected arrays.

Everything here consumes plain numpy -- the trainer collects ``mu_z``, ``mu_w``,
``c`` per cell and hands them over -- so each diagnostic can be tested on
synthetic inputs with a known answer, and none can silently read training state.

The probe (spec 4.6) is deliberately a *fresh* model fitted here, never the
training-time covariance tracker or adversary: an invariance check evaluated
with the machinery being checked would be circular.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("discell.model.metrics")

#: Cells used for k-means / probe fits; more adds cost, not information.
MAX_EVAL_CELLS = 30_000


def _subsample(n: int, limit: int, rng) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(rng.choice(n, limit, replace=False))


def _subsample_rows(rows: np.ndarray, rng, limit: int = MAX_EVAL_CELLS) -> np.ndarray:
    return rows if len(rows) <= limit else np.sort(rng.choice(rows, limit, replace=False))


def z_type_nmi(z: np.ndarray, t: np.ndarray, seed: int = 0) -> float:
    """k-means over ``z`` against the labels -- the hard floor of spec 7.10.

    The CSVAE failure mode is this number collapsing while the invariance
    penalty looks great; it is half of the joint early-stopping criterion.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    rng = np.random.default_rng(seed)
    rows = _subsample(len(z), MAX_EVAL_CELLS, rng)
    k = int(t.max()) + 1
    labels = KMeans(k, n_init=4, random_state=seed).fit_predict(z[rows])
    return float(normalized_mutual_info_score(t[rows], labels))


def mirror_r2(z: np.ndarray, c: np.ndarray, t: np.ndarray,
              seed: int = 0) -> dict:
    """Within-type ``R^2`` of ``z`` on ``c``, against a within-type permuted control.

    The mirror attractor (spec 7.2): attention reconstructing the cell through
    similar neighbours shows up as ``c`` predicting ``z``. **Within type**
    (spec 7.10 as patched): the pooled regression partly measures type
    separability, which ``z`` and ``c`` both carry legitimately -- both are
    residualised against their per-type means, and the control permutes the
    residualised ``c`` within each type.
    """
    rng = np.random.default_rng(seed)
    rows = _subsample(len(z), MAX_EVAL_CELLS, rng)
    z_s, c_s, t_s = z[rows].copy(), c[rows].copy(), t[rows]
    permuted = np.empty_like(c_s)
    for k in np.unique(t_s):
        members = np.flatnonzero(t_s == k)
        z_s[members] -= z_s[members].mean(axis=0)
        c_s[members] -= c_s[members].mean(axis=0)
        permuted[members] = c_s[members[rng.permutation(len(members))]]

    def fit_r2(design: np.ndarray) -> float:
        design = np.hstack([design, np.ones((len(design), 1))])
        coef, *_ = np.linalg.lstsq(design, z_s, rcond=None)
        return float(1.0 - (z_s - design @ coef).var() / z_s.var())

    return {"r2": fit_r2(c_s), "r2_permuted": fit_r2(permuted)}


def probe_delta_ce(z: np.ndarray, t: np.ndarray, v: np.ndarray,
                   vbar_t: np.ndarray, train: np.ndarray, test: np.ndarray,
                   seed: int = 0) -> dict:
    """Excess held-out skill of ``(z, t)`` at predicting the niche block ``v``.

    ``dCE = CE(v, vbar(t)) - CE(v, probe(z, t))`` on held-out cells, where the
    probe is a fresh ridge regression and CE is Gaussian (mean squared error):
    the niche block here is continuous ([y minus one column, PCs of Phi]), so
    squared error is its cross-entropy up to constants. Two reference scales
    (spec 4.6): the noise floor (z permuted within type; should be ~0) tells
    you what zero looks like, and the run at ``alpha_a = 0`` is the
    uncontrolled baseline to compare against.
    """
    rng = np.random.default_rng(seed)

    def ridge(design: np.ndarray, target: np.ndarray, rows: np.ndarray):
        d = np.hstack([design[rows], np.ones((len(rows), 1))])
        gram = d.T @ d + 1e-3 * np.eye(d.shape[1])
        return np.linalg.solve(gram, d.T @ target[rows])

    def mse(design: np.ndarray, coef: np.ndarray, rows: np.ndarray) -> float:
        d = np.hstack([design[rows], np.ones((len(rows), 1))])
        return float(((v[rows] - d @ coef) ** 2).mean())

    onehot = np.eye(int(t.max()) + 1, dtype=np.float64)[t]
    design = np.hstack([z, onehot])
    z_permuted = z.copy()
    for k in np.unique(t):                        # permute z within type
        rows = np.flatnonzero(t == k)
        z_permuted[rows] = z[rows[rng.permutation(len(rows))]]
    design_floor = np.hstack([z_permuted, onehot])

    # random subsample: the first rows follow tile order, i.e. space
    train = _subsample_rows(np.flatnonzero(train), rng)
    test_rows = _subsample_rows(np.flatnonzero(test), rng)

    baseline = float(((v[test_rows] - vbar_t[t[test_rows]]) ** 2).mean())
    with_z = mse(design, ridge(design, v, train), test_rows)
    floor = mse(design_floor, ridge(design_floor, v, train), test_rows)
    return {"delta_ce": baseline - with_z,
            "noise_floor": baseline - floor,
            "baseline_ce": baseline}


def held_out_reconstruction(x: np.ndarray, log_p: np.ndarray) -> float:
    """Mean per-count log-likelihood on held-out seeds; the early-stop signal."""
    totals = x.sum(axis=1).clip(min=1.0)
    return float(((x * log_p).sum(axis=1) / totals).mean())


def attention_beta_correlation(alpha: np.ndarray, beta: np.ndarray) -> float:
    """corr(alpha_ij, beta_ij) -- only meaningful if edge features are ever
    added back (spec 7.3): high correlation means the GAT rediscovered the
    leak kernel and ``w`` is becoming collinear with ``rho_bar``."""
    if len(alpha) < 2:
        return float("nan")
    return float(np.corrcoef(alpha, beta)[0, 1])
