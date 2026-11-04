"""Diagnostics of a fit, each a function of collected numpy arrays.

The trainer collects ``mu_z``, ``mu_w`` and ``c`` per cell and hands them over,
so every read can be tested on inputs with a known answer and none can read
training state. The probes are fresh models fitted here, never the training
adversary: an invariance check made with the machinery being checked would be
circular.

Every k-means runs at :data:`discell.model.data.NUM_THREADS` threads (one),
pinned in code, so the reads are deterministic.
"""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from discell.model.data import NUM_THREADS

#: Cells used for k-means and probe fits; more adds cost, not information.
MAX_EVAL_CELLS = 30_000


def _subsample(n: int, limit: int, rng) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(rng.choice(n, limit, replace=False))


def _subsample_rows(rows: np.ndarray, rng, limit: int = MAX_EVAL_CELLS) -> np.ndarray:
    return rows if len(rows) <= limit else np.sort(rng.choice(rows, limit, replace=False))


def z_type_nmi(z: np.ndarray, t: np.ndarray, seed: int = 0,
               n_threads: int = NUM_THREADS) -> float:
    """NMI between k-means clusters of ``z`` (k = number of types) and the labels.

    Half of the joint early-stopping rule: a ``z`` that stops carrying the type
    shows up here while the invariance terms look good. At most
    :data:`MAX_EVAL_CELLS` cells, drawn with the stream ``seed``.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    rng = np.random.default_rng(seed)
    rows = _subsample(len(z), MAX_EVAL_CELLS, rng)
    k = int(t.max()) + 1
    with threadpool_limits(limits=n_threads):
        labels = KMeans(k, n_init=4, random_state=seed).fit_predict(z[rows])
    return float(normalized_mutual_info_score(t[rows], labels))


def mirror_r2(z: np.ndarray, c: np.ndarray, t: np.ndarray, seed: int = 0) -> dict:
    """Within-type ``R^2`` of ``z`` on the context ``c``, and a permuted control.

    Attention that reconstructs the cell through similar neighbors shows up as
    ``c`` predicting ``z``. Both are centered per type (type separability is
    legitimate in both), and the control permutes the centered ``c`` within type.
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


def probe_delta_ce(z: np.ndarray, t: np.ndarray, v: np.ndarray, vbar_t: np.ndarray,
                   train: np.ndarray, test: np.ndarray, seed: int = 0) -> dict:
    """Excess held-out skill of ``(z, t)`` at predicting the niche block ``v``.

    ``dCE = CE(v, vbar(t)) - CE(v, ridge(z, t))`` on held-out cells, CE being
    Gaussian (mean squared error). ``noise_floor`` is the same with ``z``
    permuted within type (about 0). The per-block probe is
    :func:`discell.evaluate.probe.probe_blocks`; this pooled read is the one the
    trainer logs.
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
    for k in np.unique(t):
        rows = np.flatnonzero(t == k)
        z_permuted[rows] = z[rows[rng.permutation(len(rows))]]
    design_floor = np.hstack([z_permuted, onehot])

    # random subsample: the first rows follow tile order, i.e. space
    train = _subsample_rows(np.flatnonzero(train), rng)
    test_rows = _subsample_rows(np.flatnonzero(test), rng)

    baseline = float(((v[test_rows] - vbar_t[t[test_rows]]) ** 2).mean())
    with_z = mse(design, ridge(design, v, train), test_rows)
    floor = mse(design_floor, ridge(design_floor, v, train), test_rows)
    return {"delta_ce": baseline - with_z, "noise_floor": baseline - floor,
            "baseline_ce": baseline}


def held_out_reconstruction(x: np.ndarray, log_p: np.ndarray) -> float:
    """Mean per-count log-likelihood of held-out cells; the early-stopping signal."""
    return float(per_cell_reconstruction(x, log_p).mean())


def per_cell_reconstruction(x: np.ndarray, log_p: np.ndarray) -> np.ndarray:
    """``sum_g x_g log p_g / l`` per cell."""
    totals = x.sum(axis=1).clip(min=1.0)
    return (x * log_p).sum(axis=1) / totals


def type_means(z: np.ndarray, t: np.ndarray, n_types: int) -> np.ndarray:
    """Per-type mean of ``z`` ``(K, d)``; a type without cells gets the global mean."""
    out = np.tile(z.mean(axis=0), (n_types, 1))
    for g in np.unique(t):
        out[g] = z[t == g].mean(axis=0)
    return out.astype(z.dtype)


def type_degeneracy(z: np.ndarray, t: np.ndarray, train: np.ndarray,
                    test: np.ndarray, seed: int = 0) -> dict:
    """Is ``z`` just ``t``? ``I(z; t) / H(t)`` and the within-type variance share.

    ``mi = h_t - ce_heldout``, with ``ce_heldout`` the held-out cross-entropy of
    a multinomial logistic regression ``z -> t`` fitted on training cells and
    ``h_t`` that of the training type frequencies: a held-out lower bound on the
    mutual information. ``within_var_fraction = tr Cov(z|t) / tr Cov(z)`` over
    every cell passed in. Degenerate = ratio near 1 and within fraction near 0;
    a high ratio alone is expected, since the decoder has no ``t``.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    rows_train = _subsample_rows(np.flatnonzero(train), rng)
    rows_test = _subsample_rows(np.flatnonzero(test), rng)
    k = int(t.max()) + 1
    z = z.astype(np.float64)
    scale = z[rows_train].std(axis=0) + 1e-8
    standardized = (z - z[rows_train].mean(axis=0)) / scale
    probe = LogisticRegression(C=1.0, max_iter=500).fit(
        standardized[rows_train], t[rows_train])
    log_prob = np.full((len(rows_test), k), np.log(1e-12))
    log_prob[:, probe.classes_] = np.log(
        probe.predict_proba(standardized[rows_test]).clip(1e-12))
    ce = float(-log_prob[np.arange(len(rows_test)), t[rows_test]].mean())
    freq = (np.bincount(t[rows_train], minlength=k) / len(rows_train)).clip(1e-12)
    h_t = float(-np.log(freq)[t[rows_test]].mean())
    # log_prob is k wide, so its argmax is already a type index
    accuracy = float((log_prob.argmax(axis=1) == t[rows_test]).mean())

    total = z.var(axis=0)
    within = np.zeros(z.shape[1])
    for g in np.unique(t):
        members = t == g
        within += members.mean() * z[members].var(axis=0)
    return {"mi_ratio": (h_t - ce) / h_t, "mi": h_t - ce, "h_t": h_t,
            "ce_heldout": ce, "accuracy": accuracy,
            "within_var_fraction": float(within.sum() / max(total.sum(), 1e-12)),
            "within_var_fraction_per_dim": (within / total.clip(min=1e-12)).tolist()}


def type_profile_reconstruction(x_train, t_train: np.ndarray, x_test: np.ndarray,
                                t_test: np.ndarray) -> float:
    """Held-out per-count log-likelihood of the empirical per-type count profile.

    What a lookup from ``t`` alone achieves (no ``w``, no spillover): the reference
    line for the type-mean-``z`` reconstruction gap. *x_train* and *x_test* may
    be sparse.
    """
    import scipy.sparse as sp

    k = int(max(t_train.max(), t_test.max())) + 1
    onehot = sp.csr_matrix((np.ones(len(t_train)), (t_train, np.arange(len(t_train)))),
                           shape=(k, len(t_train)))
    pooled = np.asarray((onehot @ x_train).todense() if sp.issparse(x_train)
                        else onehot @ x_train, dtype=np.float64)
    log_profile = np.log((pooled / pooled.sum(axis=1, keepdims=True).clip(min=1.0)
                          ).clip(min=1e-8))
    totals = np.asarray(x_test.sum(axis=1)).ravel().clip(min=1.0)
    per_cell = np.empty(len(t_test))
    for g in np.unique(t_test):
        members = t_test == g
        per_cell[members] = np.asarray(x_test[members] @ log_profile[g]).ravel() \
            / totals[members]
    return float(per_cell.mean())
