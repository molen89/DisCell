"""The dead-context-channel guard (2026-09-23, FF best_s2).

Planted arrays only: a collapsed w (a per-type constant) must read its
permutation floor and be flagged; a w that tracks the niche must not.
"""

from __future__ import annotations

import numpy as np

from discell.model.degeneracy import DEAD_W_FLAG, w_channel_guard


def planted(n=3000, d_w=6, n_types=5, n_niches=10, collapsed=True, seed=0):
    """Cells with a type and a niche; w is either a per-type constant (the
    failure) or a per-niche response on top of the per-type gauge offset."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, n_types, n)
    niche = rng.integers(0, n_niches, n)
    offset = rng.normal(scale=5.0, size=(n_types, d_w))     # the gauge, huge
    w = offset[t]
    if not collapsed:
        response = rng.normal(scale=1.0, size=(n_niches, d_w))
        w = w + response[niche] + 0.1 * rng.normal(size=(n, d_w))
    return w, niche, t


def test_collapsed_w_is_flagged():
    # seed pinned: with w exactly constant within type every within-niche kNN
    # radius is a tie, and both the estimate and its permutation floor are
    # estimator noise of order 5e-3 nats about zero (a live channel clears its
    # floor by ~0.2 nats, so the rule is not fragile where it is used)
    w, niche, t = planted(collapsed=True, seed=1)
    out = w_channel_guard(w, niche, t, seed=1, perms=8)
    # w is a function of t alone: nothing above the within-type floor, and no
    # variance across cells within a type -- all of it is the gauge offset
    assert out["w_niche_mi"] == 0.0          # the FF best_s2 signature
    assert out["w_niche_mi_excess"] <= 0.0
    assert out["dead_context_channel"] is True
    assert out["flag"] == DEAD_W_FLAG
    assert out["w_var_fraction_across_cells"] < 1e-9


def test_live_w_clears_its_floor():
    w, niche, t = planted(collapsed=False)
    out = w_channel_guard(w, niche, t, seed=0, perms=3)
    assert out["w_niche_mi_excess"] > 0.0
    assert out["dead_context_channel"] is False
    assert out["flag"] is None
    # the response and the noise vary within type; the offset does not
    assert 0.0 < out["w_var_fraction_across_cells"] < 1.0


def test_isolated_cells_are_dropped_from_the_mi_read():
    w, niche, t = planted(collapsed=False)
    niche = niche.copy()
    niche[: len(niche) // 2] = -1
    out = w_channel_guard(w, niche, t, seed=0, perms=3)
    assert out["n_cells"] == int((niche >= 0).sum())
