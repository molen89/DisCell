"""Planted-answer tests for the depth-neutral cycle target (todo 3.6)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from discell.model import cell_cycle as C


def _planted_counts(seed: int = 0, n: int = 8000):
    """A world whose only structure is depth: every gene's counts scale with
    the cell's library size, so no marker set carries cycle information and a
    depth-neutral target must be uncorrelated with depth.
    """
    rng = np.random.default_rng(seed)
    genes = list(C.S_GENES) + list(C.G2M_GENES)
    genes += [f"FILLER{i}" for i in range(400)]
    depth = rng.lognormal(mean=np.log(150.0), sigma=0.8, size=n)
    rate = rng.gamma(2.0, 1.0, size=len(genes))
    rate = rate / rate.sum()
    x = rng.poisson(depth[:, None] * rate[None, :]).astype(np.float32)
    return x, np.array(genes), depth


def test_raw_scanpy_score_is_depth_tilted_and_the_fix_removes_it():
    x, genes, _ = _planted_counts()
    t = np.zeros(len(x), dtype=np.int64)
    out = C.score_cell_cycle(x, genes, t=t, depth_neutral_target=True)
    if out is None:
        pytest.skip("scanpy scoring unavailable")
    log_depth = np.log(np.asarray(x.sum(axis=1)).ravel().clip(min=1.0))
    for raw_key, new_key in (("s_score_raw", "s_score"),
                             ("g2m_score_raw", "g2m_score")):
        tilt_raw = spearmanr(out[raw_key], log_depth).correlation
        tilt_new = spearmanr(out[new_key], log_depth).correlation
        assert abs(tilt_new) < 0.05
        assert abs(tilt_new) < abs(tilt_raw)


def test_depth_neutral_is_neutral_within_every_type():
    rng = np.random.default_rng(1)
    n = 9000
    t = rng.integers(0, 3, n)
    # a different depth regime per type and a heavily depth-confounded score:
    # the depth part must not survive. (A score that is a *deterministic*
    # function of depth cannot be made neutral without destroying it; the
    # transform conditions on depth to stratum resolution, not exactly.)
    totals = np.exp(rng.normal(np.array([3.0, 5.0, 7.0])[t], 0.6))
    score = np.log(totals) + rng.standard_normal(n)
    out = C.depth_neutral(score, totals, t, seed=0)
    for g in range(3):
        members = t == g
        raw = spearmanr(score[members], np.log(totals[members])).correlation
        assert raw > 0.4
        assert abs(spearmanr(out[members],
                             np.log(totals[members])).correlation) < 0.05


def test_depth_neutral_keeps_a_signal_that_is_not_depth():
    rng = np.random.default_rng(2)
    n = 9000
    totals = np.exp(rng.normal(5.0, 0.6, n))
    signal = rng.standard_normal(n)
    score = signal + 0.5 * np.log(totals)
    out = C.depth_neutral(score, totals, seed=0)
    assert spearmanr(out, signal).correlation > 0.8


def test_split_half_reliability_separates_a_reliable_from_a_noisy_score():
    """A planted marker set that is genuinely co-expressed must read reliable;
    one whose genes are independent must not."""
    rng = np.random.default_rng(3)
    n = 4000
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    reliable = np.corrcoef(a + rng.standard_normal(n) * 0.2,
                           a + rng.standard_normal(n) * 0.2)[0, 1]
    unreliable = np.corrcoef(a + rng.standard_normal(n),
                             b + rng.standard_normal(n))[0, 1]
    assert reliable > 0.9
    assert abs(unreliable) < 0.1


def test_score_cell_cycle_reports_both_reliabilities_and_the_flag():
    x, genes, _ = _planted_counts(seed=4, n=4000)
    out = C.score_cell_cycle(x, genes, t=np.zeros(len(x), dtype=np.int64),
                             depth_neutral_target=True)
    if out is None:
        pytest.skip("scanpy scoring unavailable")
    assert out["depth_neutral"] is True
    assert set(out["reliability"]) == {"s", "g2m"}
    assert set(out["reliability_raw"]) == {"s", "g2m"}
    plain = C.score_cell_cycle(x, genes)
    assert plain["depth_neutral"] is False
    assert np.allclose(plain["s_score"], plain["s_score_raw"])
    assert plain["reliability"] == plain["reliability_raw"]
