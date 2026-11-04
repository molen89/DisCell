"""The two checks that mark a fit as failed: a dead context channel.

When the prior network ``m_psi(c, t)`` ignores the context, ``q(w)`` sits on
it, ``w`` is a per-type constant, and the reconstruction, NMI and probe reads
all look healthy. Two detectors:

* during the fit, :func:`kl_w_is_dead`: the summed per-dimension ``KL_w`` of
  the held-out cells is about 0 at an opening evaluation;
* after it, :func:`w_channel_guard`: the niche information in ``mu_w`` above
  its within-type permutation floor, ``I(niche; mu_w)`` by the kNN estimator,
  is at most 0.

Refit such a grid point with another seed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from discell.evaluate.mi import mi_discrete_continuous, within_type_permutation
from discell.model.data import NUM_THREADS

#: Healthy fits carry a summed KL_w of 3e-4 to 8e-3 at the first evaluations.
DEAD_W_KL_SUM = 1e-5
#: The in-fit detector looks at evaluations before this epoch only.
DEAD_W_MAX_EPOCH = 20
#: Niches behind the guard: k-means on the neighbor composition.
W_GUARD_NICHES = 10
W_GUARD_PERMS = 5           #: within-type permutations behind the floor
W_GUARD_MAX_CELLS = 20_000  #: cap on held-out cells for the kNN estimator
DEAD_W_FLAG = "failed fit: dead context channel"


def kl_w_is_dead(kl_w_per_dim, epoch: int) -> bool:
    """True when *epoch* is an opening epoch and the summed ``KL_w`` is about 0."""
    return epoch < DEAD_W_MAX_EPOCH and float(np.sum(kl_w_per_dim)) < DEAD_W_KL_SUM


def dead_w_channel(history: Sequence[dict]) -> bool:
    """:func:`kl_w_is_dead` over a fit's evaluation history."""
    return any(kl_w_is_dead(record["kl_w_per_dim"], int(record["epoch"]))
               for record in history if record.get("kl_w_per_dim") is not None)


def niche_labels(y: np.ndarray, degrees: np.ndarray, k: int, seed: int,
                 n_threads: int = NUM_THREADS) -> np.ndarray:
    """k-means niches of the neighbor composition *y*; -1 for isolated cells.

    Fitted on at most 100,000 connected cells, predicted for all of them. The
    labels come from the data, never from the latents.
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(degrees > 0)
    fit_rows = rows if len(rows) <= 100_000 else np.sort(
        rng.choice(rows, 100_000, replace=False))
    with threadpool_limits(limits=n_threads):
        model = KMeans(k, n_init=4, random_state=seed).fit(y[fit_rows])
        labels = np.full(len(y), -1, dtype=np.int64)
        labels[rows] = model.predict(y[rows])
    return labels


def w_channel_guard(w: np.ndarray, niche: np.ndarray, t: np.ndarray, seed: int = 0,
                    perms: int = W_GUARD_PERMS,
                    max_cells: int = W_GUARD_MAX_CELLS) -> dict:
    """The guard on held-out arrays.

    *w* ``(n, d_w)`` posterior means, *niche* the niche label (negative =
    isolated, dropped), *t* the cell type. Flagged when the excess of
    ``I(niche; w)`` over its within-type permutation floor is at most 0.
    ``w_var_fraction_across_cells`` is ``tr Cov(w|t) / tr Cov(w)``: a dead
    channel reads 0 however large ``w`` is.
    """
    w = np.asarray(w, dtype=np.float64)
    niche, t = np.asarray(niche), np.asarray(t)
    rng = np.random.default_rng(seed)

    total = w.var(axis=0).sum()
    across = np.zeros(w.shape[1])
    for g in np.unique(t):
        members = t == g
        across += members.mean() * w[members].var(axis=0)
    var_fraction = float(across.sum() / max(total, 1e-12))

    rows = np.flatnonzero(niche >= 0)
    if len(rows) > max_cells:
        rows = np.sort(rng.choice(rows, max_cells, replace=False))
    y, t_rows, w_rows = niche[rows], t[rows], w[rows]
    mi = mi_discrete_continuous(w_rows, y, rng=rng)
    floor = [mi_discrete_continuous(w_rows, within_type_permutation(y, t_rows, rng),
                                    rng=rng)
             for _ in range(perms)]
    excess = float(mi - np.mean(floor))
    return {"w_niche_mi": float(mi),
            "w_niche_mi_floor": float(np.mean(floor)),
            "w_niche_mi_floor_sd": float(np.std(floor)),
            "niche_information_check": excess,
            "w_var_fraction_across_cells": var_fraction,
            "w_total_var": float(total),
            "n_cells": int(len(rows)), "n_niches": int(len(np.unique(y))),
            "n_perms": int(perms),
            "dead_context_channel": bool(excess <= 0.0),
            "flag": DEAD_W_FLAG if excess <= 0.0 else None}
