"""Planted worlds for R12 Test 1: receiver-only vs donor-only foreign counts."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from discell.experiments import r12_depth_test as r12


def test_indicator_data_is_one_weight_per_cell():
    groups = np.array([2, 0, 1, 2, 0])
    g = r12.indicator(groups, 3)
    assert np.array_equal(g.indptr, np.arange(6))
    g.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # group 2 holds cells 0 and 3, weights 1 and 4
    assert (g @ np.ones(5)).tolist() == [7.0, 3.0, 5.0]


def test_fe_fit_matches_dummy_variable_least_squares():
    rng = np.random.default_rng(0)
    n, k = 3000, 4
    groups = rng.integers(0, k, n)
    x = rng.normal(size=(n, 3)) + groups[:, None] * 0.3
    x[:, 1] += 0.6 * x[:, 0]
    y = 0.8 * x[:, 0] - 0.3 * x[:, 1] + 0.1 * x[:, 2] + groups + rng.normal(size=n)
    block = np.column_stack([y, x])
    fit = r12.fe_fit(r12.indicator(groups, k) @ r12.moment_rows(block), 4)

    dummies = np.eye(k)[groups]
    design = np.column_stack([x, dummies])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    assert np.allclose(fit["coef"], coef[:3], atol=1e-8)
    sse_full = np.sum((y - design @ coef) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    assert fit["r2_total"] == pytest.approx(1 - sse_full / sst, abs=1e-8)
    for j in range(3):
        red = np.delete(design, j, axis=1)
        c, *_ = np.linalg.lstsq(red, y, rcond=None)
        sse_red = np.sum((y - red @ c) ** 2)
        assert fit["partial_r2"][j] == pytest.approx((sse_red - sse_full) / sse_red,
                                                     abs=1e-8)
    # LMG is a Shapley split: it sums to the within-type R^2
    assert sum(fit["lmg"]) == pytest.approx(fit["r2_within"], abs=1e-10)


def test_line_fit_matches_polyfit():
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 50.0, 2000)
    y = 3.0 + 0.12 * x + rng.normal(size=2000)
    stats = r12.moment_rows(np.column_stack([y, x])).sum(axis=0, keepdims=True)
    fit = r12.line_fit(stats)
    slope, intercept = np.polyfit(x, y, 1)
    assert fit["slope"] == pytest.approx(slope, rel=1e-9)
    assert fit["intercept"] == pytest.approx(intercept, rel=1e-7)
    assert fit["slope_through_origin"] == pytest.approx(x @ y / (x @ x), rel=1e-9)
    assert fit["ratio_of_sums"] == pytest.approx(y.sum() / x.sum(), rel=1e-12)


def lattice(side: int, pitch_um: float):
    """Square lattice positions, 4-neighbour adjacency and beta = 1/degree."""
    n = side * side
    ii, jj = np.divmod(np.arange(n), side)
    positions = np.column_stack([ii, jj]).astype(float) * pitch_um
    src, dst = [], []
    for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        a, b = ii + di, jj + dj
        ok = (a >= 0) & (a < side) & (b >= 0) & (b < side)
        src.append(np.arange(n)[ok])
        dst.append((a * side + b)[ok])
    src, dst = np.concatenate(src), np.concatenate(dst)
    adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    beta = sp.diags(1.0 / np.asarray(adj.sum(axis=1)).ravel()) @ adj
    return positions, adj, beta


def lattice_world(side: int = 60, pitch_um: float = 20.0, seed: int = 0):
    """Cells on a square lattice, 4-neighbour beta = 1/degree, smooth depth field.

    Depth is ``exp(field + noise)`` with a smooth field, so the receiver's depth
    and its neighbours' depth are correlated as on a slide and the joint
    regression has to separate them.
    """
    rng = np.random.default_rng(seed)
    n = side * side
    positions, adj, beta = lattice(side, pitch_um)
    field = rng.normal(size=n)
    for _ in range(6):                      # smooth over ~6 lattice steps
        field = 0.5 * field + 0.5 * (beta @ field)
    field = 0.9 * field / field.std()
    total = np.round(np.exp(np.log(200.0) + field + rng.normal(0.0, 0.5, n))) + 1.0
    area = np.exp(np.log(60.0) + rng.normal(0.0, 0.3, n))
    types = rng.integers(0, 3, n)
    cells = {
        "total": total, "area": area,
        "donor_depth": beta @ total, "donor_density": beta @ (total / area),
        "donor_area": beta @ area,
        "nuclear": np.round(0.5 * total), "donor_nuclear": beta @ np.round(0.5 * total),
        "flux_degree": np.asarray(adj.sum(axis=1)).ravel().astype(np.int64),
        "model_degree": np.asarray(adj.sum(axis=1)).ravel().astype(np.int64),
        "has_nucleus": np.ones(n, dtype=bool), "positions_um": positions,
        "type_index": types,
        "segmentation": np.where(rng.random(n) < 0.5, "boundary", "interior"),
    }
    return cells, rng


def planted(world: str, kappa: float = 0.1, seed: int = 0) -> dict:
    cells, rng = lattice_world(seed=seed)
    type_gain = np.array([0.7, 1.0, 1.4])[cells["type_index"]]
    driver = cells["total"] if world == "receiver" else cells["donor_depth"]
    cells["F_hard"] = rng.poisson(kappa * type_gain * driver).astype(float)
    cells["F_ramp"] = cells["F_hard"] * 1.2
    return cells


@pytest.mark.parametrize("world", ["receiver", "donor"])
def test_planted_world_is_called_for_its_driver(world):
    cells = planted(world)
    result = r12.analyse(cells, n_boot=60, seed=0)
    primary = result["joint"]["primary"]
    pt = primary["point"]
    # the two regressors really are collinear, as on a slide
    assert primary["within_corr"][0][1] > 0.4
    if world == "receiver":
        assert primary["verdict"] == "receiver dominates"
        assert pt["coef.receiver_log_l"] == pytest.approx(1.0, abs=0.12)
        assert abs(pt["coef.donor_log_sum_beta_l"]) < 0.1
        assert pt["partial_r2.receiver_log_l"] > 10 * pt["partial_r2.donor_log_sum_beta_l"]
        slope = result["raw_slopes"]["F_on_receiver_l"]["point"]
        # F = kappa * gain * l with mean gain 1.03: slope near kappa, intercept near 0
        assert slope["slope"] == pytest.approx(0.1, rel=0.2)
        assert abs(slope["intercept"]) < 3.0
    else:
        assert primary["verdict"] == "donor dominates"
        assert pt["coef.donor_log_sum_beta_l"] == pytest.approx(1.0, abs=0.15)
        assert abs(pt["coef.receiver_log_l"]) < 0.1
        assert pt["partial_r2.donor_log_sum_beta_l"] > 10 * pt["partial_r2.receiver_log_l"]
    assert set(result["joint_by_segmentation"]) == {"boundary", "interior"}


def test_a_world_driven_by_both_has_no_clear_winner():
    """F = kappa * sqrt(l_i D_i): elasticity 1/2 on each, neither dominates."""
    cells = planted("receiver")
    rng = np.random.default_rng(5)
    cells["F_hard"] = rng.poisson(
        0.1 * np.sqrt(cells["total"] * cells["donor_depth"])).astype(float)
    result = r12.analyse(cells, n_boot=60, seed=0)
    assert result["joint"]["primary"]["verdict"] == "no clear winner"


def test_markdown_and_figure_render(tmp_path):
    cells = planted("receiver")
    result = r12.analyse(cells, n_boot=20, seed=0)
    result["meta"] = {"dataset": "planted", "label_key": "t", "n_cells": 3600,
                      "transcripts_read": 0, "placed_on_an_edge": 0}
    text = r12.markdown(result)
    assert "receiver dominates" in text and "| primary |" in text
    assert "## Test 1b" in text and "| area_conditioned |" in text
    r12.figure(cells, result, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 10_000


def test_cache_round_trip_keeps_string_columns(tmp_path):
    cells = {"total": np.arange(3.0),
             "segmentation": np.array(["boundary", "interior", "boundary"], dtype=object),
             "meta": {"dataset": "planted", "type_names": ["a", "b"]}}
    r12._save_cache(cells, tmp_path / "cells.npz")
    back = r12._load_cache(tmp_path / "cells.npz")
    assert back["meta"] == cells["meta"]
    assert back["segmentation"].tolist() == cells["segmentation"].tolist()
    assert np.array_equal(back["total"], cells["total"])


# --------------------------------------------------------------------------
# test 1b: donor scaling at fixed area
# --------------------------------------------------------------------------


def test_classify_1b_applies_the_preregistered_rule():
    assert r12.classify_1b(-0.10, [-0.20, 0.05]) == "ruled out"
    assert r12.classify_1b(0.0, [-0.05, 0.24]) == "ruled out"
    assert r12.classify_1b(-0.05, [-0.30, 0.30]) == "open"
    assert r12.classify_1b(0.10, [0.05, 0.20]) == "small, not zero"
    assert r12.classify_1b(0.10, [-0.02, 0.20]) == "small, not zero"
    assert r12.classify_1b(0.20, [0.15, 0.26]) == "open"
    assert r12.classify_1b(0.36, [0.34, 0.38]) == "practically positive"
    assert r12.overall_1b({"a": "ruled out", "b": "ruled out"}) == "ruled out"
    assert r12.overall_1b({"a": "ruled out", "b": "small, not zero"}).startswith(
        "not ruled out: small")
    assert r12.overall_1b({"a": "open", "b": "ruled out"}).startswith("open")
    assert r12.overall_1b({"a": "practically positive", "b": "small, not zero"}) \
        == "not ruled out: practically positive on a"


def _standardise(x):
    return (x - x.mean()) / x.std()


def area_world(kind: str, seed: int = 0, depth_area_corr: float = 0.8,
               area_noise: float = 0.0, side: int = 60) -> dict:
    """Lattice cells whose depth tracks their true size at *depth_area_corr*.

    ``donor``: F_i ~ Poisson(kappa * sum_j beta_ij l_j), a donor-scaled leak.
    ``geometry``: no leak at all; F_i ~ Binomial(l_i, g_i), the cell's own
    transcripts beyond the nucleus bisector, with g_i rising with the true size
    ratio S_i / sum_j beta_ij S_j (a large cell beside small ones).  The
    segmented area is ``S * exp(area_noise * noise)``: a proxy of the size that
    sets the geometry, not the size itself.
    """
    rng = np.random.default_rng(seed)
    positions, adj, beta = lattice(side, 20.0)
    n = side * side
    field = rng.normal(size=n)
    for _ in range(4):
        field = 0.5 * field + 0.5 * (beta @ field)
    z_size = _standardise(_standardise(field) + rng.normal(size=n))
    size = np.exp(np.log(60.0) + 0.4 * z_size)
    z_depth = depth_area_corr * z_size + np.sqrt(1 - depth_area_corr ** 2) * rng.normal(size=n)
    total = np.round(np.exp(np.log(200.0) + 0.8 * z_depth)) + 1.0
    area = size * np.exp(area_noise * rng.normal(size=n))
    donor = beta @ total
    if kind == "donor":
        foreign = rng.poisson(0.1 * donor)
    else:
        share = np.clip(0.12 * (size / (beta @ size)) ** 1.5, 0.0, 0.9)
        foreign = rng.binomial(total.astype(np.int64), share)
    degree = np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)
    return {"total": total, "area": area, "donor_depth": donor,
            "donor_density": beta @ (total / area), "donor_area": beta @ area,
            "nuclear": np.round(0.5 * total), "donor_nuclear": beta @ np.round(0.5 * total),
            "flux_degree": degree, "model_degree": degree,
            "has_nucleus": np.ones(n, dtype=bool), "positions_um": positions,
            "type_index": rng.integers(0, 3, n), "segmentation": np.full(n, "boundary"),
            "F_hard": foreign.astype(float), "F_ramp": foreign.astype(float)}


def _run_1b(cells: dict) -> dict:
    tiles = np.floor(cells["positions_um"] / r12.TILE_UM).astype(np.int64)
    return r12.donor_at_fixed_area(cells, cells["type_index"],
                                   tiles[:, 0] * 1_000_003 + tiles[:, 1], n_boot=60)


def test_a_donor_scaled_world_is_not_ruled_out_at_depth_area_corr_08():
    block = _run_1b(area_world("donor"))
    fit = block["specs"]["area_conditioned"]
    assert fit["correlations"]["receiver_depth_area"]["pooled"] == pytest.approx(0.8, abs=0.03)
    assert fit["correlations"]["donor_depth_area"]["pooled"] > 0.8
    donor = fit["elasticities"]["donor_log_sum_beta_l"]
    assert donor["coef"] == pytest.approx(1.0, abs=0.1)
    assert block["slide_verdict"] != "ruled out"
    assert block["slide_verdict"] == "practically positive"
    assert block["specs"]["density_area_conditioned"]["classification"] == "practically positive"


def test_a_geometry_only_world_with_segmented_areas_is_ruled_out():
    """Segmented area only proxies the size that sets the geometry, so depth
    still carries the neighbour's size at fixed area and its elasticity is
    negative: the pre-registered rule reads that as ruled out."""
    block = _run_1b(area_world("geometry", area_noise=0.3))
    donor = block["specs"]["area_conditioned"]["elasticities"]["donor_log_sum_beta_l"]
    assert donor["coef"] < 0 and donor["tile_ci95"][1] < r12.PRACTICAL_ELASTICITY
    assert block["slide_verdict"] == "ruled out"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_geometry_only_world_with_exact_areas_is_never_open(seed):
    """With the true size measured exactly the donor elasticity is 0 by
    construction; the rule's "<= 0" then splits it between "ruled out" and
    "small, not zero" by the sign of the noise, but never leaves it open."""
    block = _run_1b(area_world("geometry", seed=seed))
    donor = block["specs"]["area_conditioned"]["elasticities"]["donor_log_sum_beta_l"]
    assert abs(donor["coef"]) < 0.05
    assert donor["tile_ci95"][1] < r12.PRACTICAL_ELASTICITY
    assert block["slide_verdict"] in {"ruled out", "small, not zero"}
