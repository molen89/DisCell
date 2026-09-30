"""The 8.19 breakdown layer (``discell.experiments.breakdown``): pooling over
seeds, the Bonferroni level from m_s, and kappa* under the lean entry's rule
(devlog "8.19 revised to a lean version (author, 2026-09-29)")."""

from __future__ import annotations

import numpy as np
import pytest

from discell.experiments import breakdown as B
from discell.experiments.breakdown_draws import _weighted_median

GRID = B.GRID


def _entry(estimate, sd, n=4000, seed=0, method="percentile", c=1.0):
    rng = np.random.default_rng(seed)
    return {"estimate": float(estimate),
            "draws": estimate + sd * rng.standard_normal(n),
            "method": method, "c": c}


def _collected(means, sd=0.01, per_seed=None):
    """{kappa: {seed: {"m": entry}}} for one member whose seed mean at each
    grid point is *means*; *per_seed* offsets move single seeds."""
    out = {}
    for i, k in enumerate(GRID):
        out[k] = {}
        for s in range(3):
            off = (per_seed or {}).get((k, s), 0.0)
            out[k][s] = {"m": _entry(means[i] + off, sd, seed=100 * i + s)}
    return out


def _kstar(means, m_s=8, **kw):
    table = B.section_table("test", _collected(means, **kw), m_s=m_s)
    return table["members"]["m"]


def test_planted_breakdown_point():
    # a contrast that decays with kappa and loses its interval at 0.3
    m = _kstar([0.30, 0.25, 0.20, 0.12, 0.005, -0.05], sd=0.01)
    assert m["status"] == "breaks" and m["kappa_star"] == 0.3
    assert m["reason"] == "interval contains 0"
    assert m["sign"] == 1


def test_sign_flip_is_detected():
    # every interval excludes 0, but the sign turns at 0.2
    m = _kstar([0.3, 0.2, 0.1, -0.1, -0.2, -0.3], sd=0.005)
    assert m["status"] == "breaks" and m["kappa_star"] == 0.2
    assert m["reason"] == "a seed flips sign"


def test_single_seed_flip_breaks_even_when_the_pool_holds():
    # seed 2 alone crosses 0 at 0.05; the pooled interval stays clear of 0
    m = _kstar([0.3] * 6, sd=0.001,
               per_seed={(0.05, 2): -0.35, (0.05, 0): 0.2, (0.05, 1): 0.2})
    assert m["status"] == "breaks" and m["kappa_star"] == 0.05
    assert m["reason"] == "a seed flips sign"


def test_no_finding_at_kappa_zero():
    m = _kstar([0.001, 0.2, 0.2, 0.2, 0.2, 0.2], sd=0.05)
    assert m["status"] == "no finding" and m["kappa_star"] is None


def test_above_the_grid():
    m = _kstar([-0.5] * 6, sd=0.01)
    assert m["status"] == "above the grid" and m["sign"] == -1


def test_gap_is_not_survival():
    c = _collected([0.5] * 6)
    c[0.2] = {s: {} for s in range(3)}
    m = B.section_table("test", c, m_s=8)["members"]["m"]
    assert m["status"] == "gap at kappa = 0.2"


def test_bonferroni_level_from_m_s():
    assert B.level(8) == pytest.approx(1 - 0.05 / 8)
    assert B.level(9) == pytest.approx(1 - 0.05 / 9)
    assert B.FAMILY_SIZE == {"ovarian": 9, "lung": 8, "ff": 8, "gse": 8,
                             "gse_dual": 1}
    # a planted member whose interval excludes 0 at m = 1 but not at m = 50
    means = [0.02] * 6
    wide = B.section_table("t", _collected(means, sd=0.015), m_s=50)
    narrow = B.section_table("t", _collected(means, sd=0.015), m_s=1)
    assert narrow["members"]["m"]["status"] == "above the grid"
    assert wide["members"]["m"]["status"] == "no finding"
    lo50, hi50 = wide["members"]["m"]["trajectory"]["0"]["ci"]
    lo1, hi1 = narrow["members"]["m"]["trajectory"]["0"]["ci"]
    assert lo50 < lo1 and hi50 > hi1
    with pytest.raises(ValueError):
        B.level(0)


def test_pool_averages_draws_across_seeds():
    # three seeds, independent draws of sd 1: the pooled draws have sd 1/sqrt 3
    seeds = [_entry(v, 1.0, n=200_000, seed=s) for s, v in enumerate((1, 2, 3))]
    p = B.pool(seeds, alpha=0.05)
    assert p["estimate"] == pytest.approx(2.0)
    half = 0.5 * (p["ci"][1] - p["ci"][0])
    assert half == pytest.approx(1.96 / np.sqrt(3), rel=0.02)
    assert p["n_seeds"] == 3


def test_subsample_form_is_placed_on_the_point():
    seeds = [_entry(v, 1.0, n=100_000, seed=s, method="subsample", c=0.5)
             for s, v in enumerate((10.0, 10.0, 10.0))]
    for e in seeds:
        e["draws"] = e["draws"] + 3.0         # a shifted half-sample median
    p = B.pool(seeds, alpha=0.05)
    mid = 0.5 * (p["ci"][0] + p["ci"][1])
    assert mid == pytest.approx(10.0, abs=0.02)
    assert 0.5 * (p["ci"][1] - p["ci"][0]) == pytest.approx(
        0.5 * 1.96 / np.sqrt(3), rel=0.03)


def test_ranking_pairs_on_shared_draws():
    d = np.random.default_rng(0).normal(size=100)
    members = {"a": {"estimate": 2.0, "draws": 2 + d, "method": "percentile",
                     "c": 1.0},
               "b": {"estimate": 1.0, "draws": 1 + d, "method": "percentile",
                     "c": 1.0}}
    pair = B.ranking_pairs(members, ["a", "b"])["a - b"]
    assert pair["estimate"] == 1.0 and np.allclose(pair["draws"], 1.0)


def test_npz_round_trip(tmp_path):
    from discell.experiments.breakdown_draws import write_group

    path = tmp_path / "runs" / "r" / "breakdown" / "cycle.npz"
    write_group(path, {"x": (0.5, np.arange(5.0), "percentile", 1.0,
                             {"reproduces": True})}, {"run": "r"})
    got = B.read_draws(path)["x"]
    assert got["estimate"] == 0.5 and got["draws"].tolist() == list(range(5))


def test_weighted_median_is_the_resampled_median():
    rng = np.random.default_rng(1)
    for m in (5, 6, 31):
        v = rng.normal(size=m)
        w = rng.integers(0, 3, (40, m))
        ref = [np.median(np.repeat(v, wi)) if wi.sum() else np.nan for wi in w]
        assert np.allclose(_weighted_median(v, w), ref, equal_nan=True)


def test_a_point_some_seed_lacks_is_a_gap():
    c = _collected([0.5] * 6)
    del c[0.2][1]["m"]
    m = B.section_table("test", c, m_s=8)["members"]["m"]
    assert m["status"] == "gap at kappa = 0.2"


def test_transport_members_use_all_panels():
    """Author, 2026-09-29: the transport members are scored on ALL panels,
    untrusted and supported-tier ones included."""
    from discell.experiments.breakdown_draws import (all_panel_contrasts,
                                                      select_panels)

    def panel(cf, prog, leak, trusted, overlap):
        return {"counterfactual": {"r2": cf}, "program_only": {"r2": prog},
                "leak_only": {"r2": leak}, "trusted": trusted,
                "overlap_flag": overlap}
    panels = [panel(0.5, 0.3, 0.1, True, True),
              panel(0.2, 0.1, 0.0, False, True),
              panel(0.4, 0.4, 0.2, False, False)]
    assert select_panels(panels) == panels
    got = all_panel_contrasts(panels)
    assert got["transport_cf_minus_program"] == pytest.approx(
        np.mean([0.5, 0.2, 0.4]) - np.mean([0.3, 0.1, 0.4]))
    assert got["transport_cf_minus_leak"] == pytest.approx(
        np.mean([0.5, 0.2, 0.4]) - np.mean([0.1, 0.0, 0.2]))
    assert all_panel_contrasts([]) == {}
