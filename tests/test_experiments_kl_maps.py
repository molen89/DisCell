"""The kl_maps statistics helpers on a planted enrichment.

Cells of type 1 carry a larger quantity (planted), on a grid of tiles; the
top decile must be enriched for type 1 (OR > 1, CI above 1), the planted
continuous feature must show a positive SMD, and an unrelated feature must
not pass. The tile-aggregated bootstrap's point estimates must equal the
plain helpers.
"""

from __future__ import annotations

import numpy as np

from discell.experiments.kl_maps import (depth_residual, enrichment,
                                         odds_ratio, passes, smd,
                                         tile_multiplicities, top_decile)


def _world(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 3, n)
    tiles = rng.integers(0, 40, n)
    feature = rng.normal(size=n)                  # drives the quantity
    noise = rng.normal(size=n)                    # unrelated
    q = 1.0 + 1.5 * (t == 1) + 0.8 * feature + rng.normal(scale=0.5, size=n)
    return t, tiles, feature, noise, q


def test_top_decile_picks_the_largest():
    score = np.arange(100.0)[::-1]
    top = top_decile(score)
    assert top.sum() == 10 and top[:10].all() and not top[10:].any()


def test_depth_residual_removes_per_type_slope():
    rng = np.random.default_rng(1)
    t = np.repeat([0, 1], 500)
    log_l = rng.normal(size=1000)
    q = np.where(t == 0, 2.0 * log_l, -3.0 * log_l + 5.0)
    assert np.allclose(depth_residual(q, log_l, t), 0.0, atol=1e-9)


def test_planted_enrichment_passes_and_null_does_not():
    t, tiles, feature, noise, q = _world()
    top = top_decile(q)
    assert odds_ratio(top, t == 1) > 3.0
    assert odds_ratio(top, t == 0) < 1.0
    assert smd(top, feature) > 0.5
    assert abs(smd(top, noise)) < 0.2

    mult = tile_multiplicities(40, 200, seed=0)
    out = enrichment(top, {"type": (t, {0: "a", 1: "b", 2: "c"})},
                     {"feature": feature, "noise": noise}, tiles, 40, mult)
    # tile-aggregated point estimates equal the plain helpers
    assert np.isclose(out["type"]["b"]["estimate"], odds_ratio(top, t == 1))
    assert np.isclose(out["feature"]["estimate"], smd(top, feature))
    assert passes(out["type"]["b"]) == 1
    assert passes(out["type"]["a"]) == -1
    assert passes(out["feature"]) == 1
    assert passes(out["noise"]) == 0
    lo, hi = out["type"]["b"]["ci95"]
    assert lo < out["type"]["b"]["estimate"] < hi


def test_smd_ignores_undefined_cells():
    top = np.array([True, True, False, False, False])
    x = np.array([2.0, np.nan, 0.0, 1.0, np.nan])
    assert np.isfinite(smd(top, np.where(np.isnan(x), 0.0, x)))
    # one top cell defined: variance 0 on that side, still finite
    assert np.isfinite(smd(top, x))
