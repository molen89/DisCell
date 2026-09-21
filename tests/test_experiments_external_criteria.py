"""Planted-truth tests for the three external criteria (6b.1, 6b.3, 6b.4).

Each experiment gets a world where the answer is known before the estimator
runs: a planted LR-enriched leak share (6b.1), planted independent and
planted dependent latents (6b.3), a planted gradient along the true axis
only (6b.4). The instrument must read the plant and must NOT read the
non-plant.
"""

from __future__ import annotations

import numpy as np
import pytest

from discell.experiments.external_criteria import (
    abundance_matched_other, accumulate_shares, axis_counts, cmi_knn,
    equal_count_bins, kendall_tau_rows, mann_whitney, mi_continuous,
    mi_discrete_continuous, one_hot, within_type_permutation)


# --- 6b.1 -----------------------------------------------------------------

def _panels(rng, n_genes, lr_rows, leak_boost, resp_boost, n_panels=40):
    panels = []
    for _ in range(n_panels):
        keep = np.arange(n_genes)
        resp = rng.standard_normal(n_genes)
        leak = rng.standard_normal(n_genes)
        leak[lr_rows] *= leak_boost
        resp[lr_rows] *= resp_boost
        noise = rng.standard_normal(n_genes)
        panels.append({"keep": keep, "response": resp, "leak": leak,
                       "observed": resp + leak + noise})
    return panels


def test_planted_lr_leak_share_is_recovered():
    rng = np.random.default_rng(0)
    n_genes = 600
    is_lr = np.zeros(n_genes, bool)
    is_lr[:120] = True
    panels = _panels(rng, n_genes, np.flatnonzero(is_lr),
                     leak_boost=3.0, resp_boost=1.0)
    shares = accumulate_shares(panels, n_genes)
    ok = shares["scored"]
    assert ok.all()
    leak = mann_whitney(shares["leak"][is_lr], shares["leak"][~is_lr])
    resp = mann_whitney(shares["response"][is_lr], shares["response"][~is_lr])
    # the plant is in the leak channel only
    assert leak["p"] < 1e-10 and leak["effect"] > 0.8
    assert resp["effect"] < 0.0          # response share is *displaced* down
    assert abs(resp["effect"]) < abs(leak["effect"])


def test_planted_lr_response_share_is_recovered_and_leak_is_not():
    rng = np.random.default_rng(1)
    n_genes = 600
    is_lr = np.zeros(n_genes, bool)
    is_lr[:120] = True
    panels = _panels(rng, n_genes, np.flatnonzero(is_lr),
                     leak_boost=1.0, resp_boost=3.0)
    shares = accumulate_shares(panels, n_genes)
    resp = mann_whitney(shares["response"][is_lr], shares["response"][~is_lr])
    leak = mann_whitney(shares["leak"][is_lr], shares["leak"][~is_lr])
    assert resp["p"] < 1e-10 and resp["effect"] > 0.8
    assert leak["effect"] < 0.0


def test_no_plant_gives_no_separation():
    rng = np.random.default_rng(2)
    n_genes = 600
    is_lr = np.zeros(n_genes, bool)
    is_lr[:120] = True
    panels = _panels(rng, n_genes, np.flatnonzero(is_lr), 1.0, 1.0)
    shares = accumulate_shares(panels, n_genes)
    for channel in ("response", "leak", "both"):
        t = mann_whitney(shares[channel][is_lr], shares[channel][~is_lr])
        assert abs(t["effect"]) < 0.15


def test_shares_are_a_partition():
    rng = np.random.default_rng(3)
    panels = _panels(rng, 100, np.arange(10), 2.0, 1.0, n_panels=5)
    s = accumulate_shares(panels, 100)
    total = s["response"] + s["leak"] + s["unexplained"]
    assert np.allclose(total[s["scored"]], 1.0)
    assert np.allclose(s["both"][s["scored"]],
                       (s["response"] + s["leak"])[s["scored"]])


def test_genes_absent_from_every_panel_are_unscored():
    rng = np.random.default_rng(4)
    panels = _panels(rng, 50, np.arange(5), 1.0, 1.0, n_panels=3)
    for p in panels:                       # only the first 30 genes scored
        p["keep"] = p["keep"][:30]
        for k in ("response", "leak", "observed"):
            p[k] = p[k][:30]
    s = accumulate_shares(panels, 50)
    assert s["scored"][:30].all()
    assert not s["scored"][30:].any()


def test_abundance_matching_picks_the_lr_expression_range():
    rng = np.random.default_rng(5)
    n = 1000
    is_lr = np.zeros(n, bool)
    is_lr[:200] = True
    rate = np.concatenate([rng.uniform(1e-3, 3e-3, 200),
                           rng.uniform(1e-5, 3e-3, 800)])
    scored = np.ones(n, bool)
    picked = abundance_matched_other(is_lr, scored, rate, rng)
    assert not is_lr[picked].any()
    # matched set sits at the LR genes' level, the full non-LR set does not
    assert (abs(np.median(rate[picked]) - np.median(rate[is_lr]))
            < abs(np.median(rate[~is_lr]) - np.median(rate[is_lr])))


# --- 6b.3 -----------------------------------------------------------------

def test_mi_reads_planted_dependence_and_not_independence():
    rng = np.random.default_rng(10)
    n, k = 4000, 5
    y = rng.integers(0, k, n)
    centres = rng.standard_normal((k, 3)) * 3.0
    dependent = centres[y] + rng.standard_normal((n, 3))
    independent = rng.standard_normal((n, 3))
    h_y = np.log(k)
    mi_dep = mi_discrete_continuous(dependent, y)
    mi_ind = mi_discrete_continuous(independent, y)
    assert mi_dep > 0.6 * h_y
    assert mi_ind < 0.05 * h_y
    assert mi_dep > 8 * max(mi_ind, 1e-3)


def test_mig_and_mic_on_planted_latents():
    """w carries the niche, z does not: MIC near 1, MIG well above 0."""
    rng = np.random.default_rng(11)
    n, k = 4000, 6
    y = rng.integers(0, k, n)
    centres = rng.standard_normal((k, 4)) * 3.0
    w = centres[y] + rng.standard_normal((n, 4))
    z = rng.standard_normal((n, 4))
    h_y = np.log(k)
    i_yw = mi_discrete_continuous(w, y)
    i_yz = mi_discrete_continuous(z, y)
    mig = (i_yw - i_yz) / h_y
    mic = i_yw / (i_yw + i_yz)
    assert mic > 0.9
    assert mig > 0.5


def test_continuous_mi_matches_the_gaussian_closed_form():
    rng = np.random.default_rng(12)
    n, r = 5000, 0.8
    a = rng.standard_normal(n)
    b = r * a + np.sqrt(1 - r ** 2) * rng.standard_normal(n)
    truth = -0.5 * np.log(1 - r ** 2)
    assert abs(mi_continuous(a[:, None], b[:, None]) - truth) < 0.1


def test_conditional_mi_is_zero_when_w_and_z_are_independent_given_y():
    rng = np.random.default_rng(13)
    n, k = 4000, 5
    y = rng.integers(0, k, n)
    cw = rng.standard_normal((k, 2)) * 3.0
    cz = rng.standard_normal((k, 2)) * 3.0
    w = cw[y] + rng.standard_normal((n, 2))
    z = cz[y] + rng.standard_normal((n, 2))
    # marginally dependent (both carry y), conditionally independent
    assert mi_continuous(w, z) > 0.2
    assert cmi_knn(w, z, y) < 0.08


def test_within_type_permutation_preserves_the_type_niche_table():
    rng = np.random.default_rng(14)
    n = 2000
    t = rng.integers(0, 4, n)
    y = (t + rng.integers(0, 3, n)) % 5
    y_perm = within_type_permutation(y, t, rng)
    for g in range(4):
        a = np.bincount(y[t == g], minlength=5)
        b = np.bincount(y_perm[t == g], minlength=5)
        assert (a == b).all()
    assert not (y_perm == y).all()


def test_permutation_floor_is_the_type_carried_mi():
    """A latent that knows only the TYPE reads at its own floor, not above."""
    rng = np.random.default_rng(15)
    n = 5000
    t = rng.integers(0, 5, n)
    y = np.where(rng.random(n) < 0.7, t, rng.integers(0, 5, n))
    tcent = rng.standard_normal((5, 3)) * 3.0
    type_only = tcent[t] + rng.standard_normal((n, 3))
    mi = mi_discrete_continuous(type_only, y)
    floor = np.mean([mi_discrete_continuous(
        type_only, within_type_permutation(y, t, rng)) for _ in range(3)])
    assert mi - floor < 0.15 * mi        # nothing above the type-carried part


def test_one_hot_shape_and_rows():
    y = np.array([2, 2, 5, 7])
    h = one_hot(y)
    assert h.shape == (4, 3)
    assert (h.sum(axis=1) == 1).all()


# --- 6b.4 -----------------------------------------------------------------

def test_kendall_tau_matches_scipy():
    from scipy.stats import kendalltau

    rng = np.random.default_rng(20)
    values = rng.standard_normal((6, 40))
    order = np.arange(6, dtype=float)
    mine_tau = kendall_tau_rows(values, order)
    for g in range(40):
        assert abs(mine_tau[g] - kendalltau(order, values[:, g])[0]) < 1e-9


def test_planted_gradient_is_seen_on_the_true_axis_only():
    rng = np.random.default_rng(21)
    n_bands, n_genes, n_planted = 6, 500, 100
    order = np.arange(n_bands, dtype=float)
    # the plant: the first n_planted genes rise monotonically with the band
    true_vals = 0.05 * rng.standard_normal((n_bands, n_genes))
    true_vals[:, :n_planted] += order[:, None]
    # the false axis re-bins the SAME cells: the planted structure is
    # scrambled, only noise survives
    false_vals = 0.05 * rng.standard_normal((n_bands, n_genes))
    tau_t = kendall_tau_rows(true_vals, order)
    tau_f = kendall_tau_rows(false_vals, order)
    counts = axis_counts(tau_t, tau_f)["0.9"]
    assert counts["true_axis"] >= n_planted
    assert counts["false_axis"] < 0.25 * n_planted
    assert counts["ratio"] > 4
    assert np.abs(tau_t[:n_planted]).mean() > 0.95


def test_symmetric_structure_gives_a_ratio_near_one():
    """The negative control: if the signal is colocalisation, both axes fire."""
    rng = np.random.default_rng(22)
    order = np.arange(6, dtype=float)
    a = 0.05 * rng.standard_normal((6, 400)) + order[:, None]
    b = 0.05 * rng.standard_normal((6, 400)) + order[:, None]
    counts = axis_counts(kendall_tau_rows(a, order),
                         kendall_tau_rows(b, order))["0.9"]
    assert 0.8 < counts["ratio"] < 1.25


def test_equal_count_bins_are_balanced():
    rng = np.random.default_rng(23)
    v = rng.standard_normal(1200)
    bins = equal_count_bins(v, 6)
    sizes = np.bincount(bins, minlength=6)
    assert len(sizes) == 6
    assert sizes.max() - sizes.min() <= 2


def test_axis_counts_handles_an_empty_false_axis():
    counts = axis_counts(np.array([1.0, 1.0]), np.array([0.0, 0.0]))["0.6"]
    assert counts["false_axis"] == 0
    assert counts["ratio"] == float("inf")
    assert counts["excess"] == 2


def test_kendall_tau_ties_read_zero():
    values = np.zeros((5, 3))
    assert np.allclose(kendall_tau_rows(values, np.arange(5, dtype=float)), 0.0)


@pytest.mark.parametrize("boost", [1.5, 2.0, 4.0])
def test_leak_plant_effect_grows_with_the_plant(boost):
    rng = np.random.default_rng(30)
    n_genes = 400
    is_lr = np.zeros(n_genes, bool)
    is_lr[:80] = True
    panels = _panels(rng, n_genes, np.flatnonzero(is_lr), boost, 1.0)
    s = accumulate_shares(panels, n_genes)
    t = mann_whitney(s["leak"][is_lr], s["leak"][~is_lr])
    assert t["effect"] > 0.3
