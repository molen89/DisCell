"""The tile bootstrap (metrics package item 1): coverage on planted spatial
data, Holm, and each adapter reproducing its metric at unit weights."""

from __future__ import annotations

import numpy as np
import pytest

from discell.experiments import bootstrap as B


def _planted(rng, side: int = 15, per_tile: int = 10, tile_um: float = 200.0,
             effect_sd: float = 1.0, noise_sd: float = 1.0, mean: float = 0.0):
    """Cells on a side x side grid of tiles; each tile a shared random effect
    (the spatial dependence), each cell its own noise. True mean = *mean*."""
    tiles = np.repeat(np.arange(side * side), per_tile)
    xy = np.stack([(tiles % side) * tile_um, (tiles // side) * tile_um], axis=1)
    xy = xy + rng.uniform(1, tile_um - 1, size=xy.shape)
    values = (mean + effect_sd * rng.standard_normal(side * side)[tiles]
              + noise_sd * rng.standard_normal(len(tiles)))
    return xy, values


def test_tile_index_groups_cells_by_square():
    xy = np.array([[10, 10], [190, 199], [201, 0], [0, 250], [399, 399]])
    ids, n = B.tile_index(xy, 200.0)
    assert n == 4
    assert ids[0] == ids[1] and len({ids[0], ids[2], ids[3], ids[4]}) == 4


def test_tile_bootstrap_covers_the_planted_mean_about_95_percent():
    """Spatially dependent cells: the tile interval covers the true mean at
    close to its nominal rate; a cell-level interval (every cell its own
    tile) on the same data covers far less -- the reason for tiles."""
    rng = np.random.default_rng(0)
    hits_tile = hits_cell = 0
    sims = 200
    for s in range(sims):
        xy, values = _planted(rng)
        tile = B.tile_bootstrap(xy, values, n=300, seed=s)
        cell = B.tile_bootstrap(xy, values, n=300, tile_um=1e-3, seed=s)
        hits_tile += tile["ci95"][0] <= 0.0 <= tile["ci95"][1]
        hits_cell += cell["ci95"][0] <= 0.0 <= cell["ci95"][1]
    assert 0.88 <= hits_tile / sims <= 0.99
    assert hits_cell / sims < 0.75


def test_tile_bootstrap_estimate_is_the_unit_weight_statistic_and_dicts_work():
    rng = np.random.default_rng(1)
    xy, values = _planted(rng, mean=2.0)
    out = B.tile_bootstrap(xy, values, n=50)
    assert out["estimate"] == pytest.approx(values.mean())
    assert out["n_cells"] == len(values) and out["n_tiles"] == 225
    both = B.tile_bootstrap(
        xy, values, lambda v, w: {"mean": B.weighted_mean(v, w),
                                  "sq": B.weighted_mean(v ** 2, w)}, n=50)
    assert set(both["reads"]) == {"mean", "sq"}
    assert both["reads"]["sq"]["estimate"] == pytest.approx((values ** 2).mean())


def test_holm_matches_the_textbook_step_down():
    out = B.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    # sorted 0.005, 0.01, 0.03, 0.04 -> 0.02, 0.03, 0.06, 0.06 (monotone)
    assert out["p_adjusted"] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert out["reject"] == [True, False, False, True]
    assert B.holm([0.9, 0.8])["p_adjusted"] == pytest.approx([1.0, 1.0])


def test_paired_comparisons_find_the_planted_difference_and_not_the_null():
    """Three grid levels on the same cells: b = a + 0.5, c = a + small noise.
    Shared tile draws cancel the tile effect in the differences."""
    rng = np.random.default_rng(2)
    xy, a = _planted(rng)
    per = {"a": a, "b": a + 0.5, "c": a + 0.01 * rng.standard_normal(len(a))}
    out = B.paired_comparisons(xy, per, n=500)
    rows = {(r["a"], r["b"]): r for r in out["comparisons"]}
    assert rows[("a", "b")]["reject_holm"] and rows[("b", "c")]["reject_holm"]
    assert not rows[("a", "c")]["reject_holm"]
    assert rows[("a", "b")]["difference"] == pytest.approx(-0.5)
    assert out["family"]["m"] == 3


def test_bootstrap_p_is_two_sided():
    assert B.bootstrap_p(np.full(99, 1.0)) == pytest.approx(0.02)
    assert B.bootstrap_p(np.r_[np.full(50, 1.0), np.full(50, -1.0)]) == 1.0


def test_nmi_statistic_is_sklearns_at_unit_weights_and_duplicates_count():
    from sklearn.metrics import normalized_mutual_info_score

    rng = np.random.default_rng(3)
    t = rng.integers(0, 4, 500)
    c = np.where(rng.random(500) < 0.7, t, rng.integers(0, 4, 500))
    cells = {"t": t, "cluster": c}
    assert B.nmi_statistic(cells, np.ones(500)) == pytest.approx(
        normalized_mutual_info_score(t, c))
    w = rng.integers(0, 3, 500).astype(float)
    rep = np.repeat(np.arange(500), w.astype(int))
    assert B.nmi_statistic(cells, w) == pytest.approx(
        normalized_mutual_info_score(t[rep], c[rep]))


def test_nmi_cells_reproduce_z_type_nmi():
    from discell.model import metrics as M

    rng = np.random.default_rng(4)
    t = rng.integers(0, 3, 600)
    z = rng.standard_normal((600, 4)) + 3.0 * np.eye(4)[t]
    cells = B.nmi_cells(z, t, seed=0)
    assert B.nmi_statistic(cells, np.ones(len(cells["rows"]))) == pytest.approx(
        M.z_type_nmi(z, t, seed=0))


def test_cycle_cells_reproduce_the_pooled_cycle_r2():
    from discell.model import metrics as M

    rng = np.random.default_rng(5)
    n = 2000
    t = rng.integers(0, 3, n)
    latent = rng.standard_normal((n, 5))
    scores = latent[:, :2] @ np.array([[1.0, 0.2], [0.3, 1.0]]) \
        + 0.5 * rng.standard_normal((n, 2))
    train = rng.random(n) < 0.7
    ref = M.cycle_r2(latent, t, scores, [0, 1], train, ~train, seed=0)
    cells = B.cycle_cells(latent, t, scores, [0, 1], train, ~train, seed=0)
    assert B.cycle_statistic(cells, np.ones(len(cells["rows_test"]))) == \
        pytest.approx(ref["r2_pooled"])


@pytest.mark.parametrize("family", ["ridge", "mlp"])
def test_probe_cells_reproduce_the_per_block_excess(family):
    from discell.model import metrics as M

    rng = np.random.default_rng(6)
    n, k = 1200, 3
    t = rng.integers(0, k, n)
    z = rng.standard_normal((n, 4))
    v = np.hstack([z[:, :1] + rng.standard_normal((n, 1)),     # comp column
                   rng.standard_normal((n, 1)),                 # comp column
                   rng.standard_normal((n, 2))])                # image block
    vbar = np.stack([v[t == g].mean(axis=0) for g in range(k)])
    train = np.arange(n) < 900
    fn = M.probe_gain_per_block if family == "ridge" else M.probe_gain_per_block_mlp
    ref = fn(z, t, v, vbar, train, ~train, n_comp=2, seed=0, n_perm=3)
    cells = B.probe_cells(z, t, v, vbar, train, ~train, n_comp=2, seed=0,
                          n_perm=3, family=family)
    stat = B.probe_statistic(cells, {"comp": 0.1, "img": 0.05})(
        cells, np.ones(len(cells["test_rows"])))
    assert stat["comp_excess"] == pytest.approx(ref["comp"]["excess"], abs=1e-9)
    assert stat["img_excess"] == pytest.approx(ref["img"]["excess"], abs=1e-9)
    assert stat["comp_frac"] == pytest.approx(ref["comp"]["excess"] / 0.1)


def test_w_mi_cells_reproduce_the_guard_excess():
    from discell.experiments.external_criteria import mi_discrete_continuous
    from discell.model.degeneracy import w_channel_guard

    rng = np.random.default_rng(7)
    n = 1500
    t = rng.integers(0, 3, n)
    niche = rng.integers(0, 4, n)
    w = rng.standard_normal((n, 2)) + 0.8 * niche[:, None]
    ref = w_channel_guard(w, niche, t, seed=0, perms=3)
    cells = B.w_mi_cells(w, niche, t, seed=0, perms=3)
    assert B.w_mi_statistic(cells, np.ones(len(cells["rows"]))) == \
        pytest.approx(ref["w_niche_mi_excess"], abs=1e-10)
    terms = B.ross_terms(w, niche, rng=np.random.default_rng(1))
    assert max(np.nanmean(terms), 0.0) == pytest.approx(
        mi_discrete_continuous(w, niche, rng=np.random.default_rng(1)))


def test_weighted_mmd_is_the_unbiased_estimator_at_unit_weights_and_on_duplicates():
    import torch

    from discell.model import transport as T

    rng = np.random.default_rng(8)
    x = torch.as_tensor(rng.random((30, 5)), dtype=torch.float64)
    y = torch.as_tensor(rng.random((25, 5)), dtype=torch.float64)
    kxx, kyy, kxy = (T._kernel_parts(a, b, 2.0) for a, b in
                     ((x, x), (y, y), (x, y)))
    syy = float((kyy.sum() - kyy.diagonal().sum()) / (25 * 24))
    part = B._weighted_mmd_parts(kxx, kxy.mean(dim=1),
                                 torch.ones(1, 30, dtype=torch.float64))
    assert float(part[0]) + syy == pytest.approx(T._mmd2(kxx, kyy, kxy))
    # integer weights: pairs of DISTINCT cells weighted w_i w_j (no cell
    # paired with its own bootstrap copy), cross term the weighted mean
    w = rng.integers(0, 3, 30).astype(float)
    k = kxx.numpy()
    pairs = sum(w[i] * w[j] * k[i, j] for i in range(30) for j in range(30)
                if i != j)
    expected = (pairs / (w.sum() ** 2 - (w ** 2).sum())
                - 2.0 * float(np.dot(w, kxy.mean(dim=1).numpy())) / w.sum())
    part = B._weighted_mmd_parts(kxx, kxy.mean(dim=1),
                                 torch.as_tensor(w[None, :], dtype=torch.float64))
    assert float(part[0]) == pytest.approx(expected)


def test_split_weights_halve_the_resampled_multiset():
    rng = np.random.default_rng(9)
    mult = np.array([0, 3, 1, 2, 0, 5])
    halves = B._split_weights(mult, rng, 4)
    assert halves.shape == (4, 2, 6)
    assert np.all(halves.sum(axis=1) == mult)
    assert np.all(halves[:, 0].sum(axis=1) == mult.sum() // 2)


def test_gap_closed_matches_distribution_scores_definition():
    assert float(B._gap(1.0, 0.4, 0.2)) == pytest.approx(0.75)
    assert float(B._gap(1.0, -5.0, 0.2)) == 1.0
    assert np.isnan(float(B._gap(0.2, 0.1, 0.2)))
