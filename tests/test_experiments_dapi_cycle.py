"""Planted-signal tests for the six DAPI cell-cycle gates.

Every test plants a signal whose answer is known in advance, so a green run
means the instrument reads a real 2N/4N split when one is there and refuses
one when it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from discell.experiments import dapi_cycle as dc


def test_flat_field_recovers_a_planted_tile_gain():
    """A smooth planted illumination field is recovered and removed."""
    rng = np.random.default_rng(0)
    col = rng.integers(0, 20, 40_000)
    row = rng.integers(0, 20, 40_000)
    planted = np.exp(0.03 * col - 0.02 * row + 0.001 * col * row)
    planted /= np.exp(np.mean(np.log(planted)))
    truth = rng.lognormal(mean=7.0, sigma=0.2, size=col.size)
    observed = truth * planted

    gain = dc.flat_field_gain(col, row, observed)

    # the estimated gain tracks the planted one, and dividing it out removes
    # nearly all of the tile-to-tile drift.
    assert np.corrcoef(np.log(gain), np.log(planted))[0, 1] > 0.99
    assert np.allclose(gain, planted, rtol=0.05)
    before = dc.tile_median_spread(col, row, observed)
    after = dc.tile_median_spread(col, row, observed / gain)
    assert before > 1.5
    assert after < 1.1


def test_flat_field_leaves_a_flat_field_alone():
    rng = np.random.default_rng(1)
    col = rng.integers(0, 12, 20_000)
    row = rng.integers(0, 12, 20_000)
    value = rng.lognormal(mean=7.0, sigma=0.2, size=col.size)
    gain = dc.flat_field_gain(col, row, value)
    assert np.allclose(gain, 1.0, atol=0.05)


def test_gate5_passes_a_planted_bimodal_integral_with_ratio_two():
    """Two log-normal modes a factor of two apart: the gate must see them."""
    rng = np.random.default_rng(2)
    two_n = rng.lognormal(mean=np.log(1e6), sigma=0.12, size=8_000)
    four_n = rng.lognormal(mean=np.log(2e6), sigma=0.12, size=4_000)
    values = np.concatenate([two_n, four_n])

    out = dc.bimodality(np.log(values))

    assert out["bic_margin"] > dc.GATE5_MIN_BIC_MARGIN
    assert out["two_components_win"]
    assert dc.GATE5_RATIO_RANGE[0] <= out["ratio"] <= dc.GATE5_RATIO_RANGE[1]
    assert out["ratio_in_range"]


def test_gate5_fails_a_unimodal_lognormal_integral():
    """The ovarian failure mode: one heavy-tailed body, no second mode."""
    rng = np.random.default_rng(3)
    values = rng.lognormal(mean=np.log(1e6), sigma=0.55, size=12_000)

    out = dc.bimodality(np.log(values))

    assert not (out["two_components_win"] and out["ratio_in_range"])
    assert out["dip_p"] > 0.05


def test_gate5_slide_level_verdict_follows_the_types():
    rng = np.random.default_rng(4)
    bimodal = np.concatenate([
        rng.lognormal(np.log(1e6), 0.12, 4_000),
        rng.lognormal(np.log(2e6), 0.12, 2_000)])
    unimodal = rng.lognormal(np.log(1e6), 0.55, 6_000)

    good = dc.gate5_bimodality(
        np.concatenate([bimodal, bimodal]),
        np.array(["A"] * bimodal.size + ["B"] * bimodal.size))
    bad = dc.gate5_bimodality(
        np.concatenate([unimodal, unimodal]),
        np.array(["A"] * unimodal.size + ["B"] * unimodal.size))

    assert good["passed"] and good["n_types_passing"] == 2
    assert not bad["passed"] and bad["n_types_passing"] == 0


def test_auroc_matches_a_known_ordering():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert dc.auroc(scores, np.array([0, 0, 1, 1], bool)) == 1.0
    assert dc.auroc(scores, np.array([1, 1, 0, 0], bool)) == 0.0
    assert dc.auroc(scores, np.array([0, 1, 0, 1], bool)) == 0.75


def test_r_squared_recovers_a_planted_line():
    rng = np.random.default_rng(5)
    x = rng.normal(size=5_000)
    slope, r2 = dc.r_squared(x, 1.5 * x)
    assert slope == pytest.approx(1.5, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)
    _, r2_noise = dc.r_squared(x, rng.normal(size=5_000))
    assert r2_noise < 0.01


def test_tile_median_spread_of_a_uniform_field_is_one():
    rng = np.random.default_rng(6)
    col = rng.integers(0, 10, 10_000)
    row = rng.integers(0, 10, 10_000)
    value = np.full(10_000, 5.0)
    assert dc.tile_median_spread(col, row, value) == pytest.approx(1.0)


def test_gate3_exclusions_count_each_reason():
    slide = {
        "dapi_sum": np.array([1.0, np.nan, 2.0, 3.0, 4.0]),
        "nucleus_count": np.array([1.0, 1.0, 2.0, 1.0, 1.0]),
        "segmentation": np.array(["interior stain", "interior stain",
                                  "interior stain",
                                  "Segmented by nucleus expansion of 5.0um",
                                  "boundary stain"]),
    }
    out = dc.gate3_exclusions(slide)
    assert (out["n_no_nucleus"], out["n_multinucleate"],
            out["n_nucleus_expansion"], out["n_kept"]) == (1, 1, 1, 2)


def test_corrected_integral_subtracts_background_and_gain():
    slide = {"dapi_sum": np.array([1000.0, 2000.0]),
             "area_px": np.array([100.0, 100.0])}
    out = dc.corrected_integral(slide, np.array([1.0, 2.0]), 1.0,
                                np.array([True, True]))
    assert out == pytest.approx(np.array([900.0, 950.0]))
