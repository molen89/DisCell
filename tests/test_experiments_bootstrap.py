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


def test_cycle_cells_reproduce_the_top_decile_cycle_r2():
    """The q90 read (``cycle_r2(..., cells=)``, author's decision 2026-09-28)
    replayed per cell: the unit-weight statistic is the metric, and the
    type path is unchanged by the ``cells`` option."""
    from discell.model import metrics as M
    from discell.model.cell_cycle import cycling_set

    rng = np.random.default_rng(6)
    n = 20000
    t = rng.integers(0, 4, n)
    latent = rng.standard_normal((n, 5))
    scores = latent[:, :2] @ np.array([[1.0, 0.2], [0.3, 1.0]]) \
        + 0.5 * rng.standard_normal((n, 2))
    train = rng.random(n) < 0.7
    cells = cycling_set(scores, ~train, eligible=t != 3)
    ref = M.cycle_r2(latent, t, scores, None, train, ~train, seed=0,
                     cells=cells)
    got = B.cycle_cells(latent, t, scores, None, train, ~train, seed=0,
                        cells=cells)
    assert np.isfinite(ref["r2_pooled"])
    assert B.cycle_statistic(got, np.ones(len(got["rows_test"]))) == \
        pytest.approx(ref["r2_pooled"])
    assert not np.isin(got["rows_test"], np.flatnonzero(t == 3)).any()


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


def test_split_weights_halve_the_distinct_resampled_cells():
    rng = np.random.default_rng(9)
    mult = np.array([0, 3, 1, 2, 0, 5])
    halves = B._split_weights(mult, rng, 4)
    assert halves.shape == (4, 2, 6)
    assert np.all(halves.sum(axis=1) == mult)
    assert not np.any((halves[:, 0] > 0) & (halves[:, 1] > 0))  # no cell in both
    assert np.all((halves[:, 0] > 0).sum(axis=1) == 2)          # 4 cells -> 2 + 2


def test_gap_closed_matches_distribution_scores_definition():
    assert float(B._gap(1.0, 0.4, 0.2)) == pytest.approx(0.75)
    assert float(B._gap(1.0, -5.0, 0.2)) == 1.0
    assert np.isnan(float(B._gap(0.2, 0.1, 0.2)))


def test_probe_cells_replay_survives_the_metrics_float32_baseline_rounding():
    """Planted: float32 v (as ``data.v_block``) with 30k held-out cells whose
    composition column sits far from its type mean -- thousands of identical
    float32 squares. The metric averages them in float32 (a naive axis-0
    sum), which drifts ~3e-4 relative from the float64 mean the bootstrap
    takes: the lineage finalL runs' failure. The replay must still check out
    (it compares like with like) and grade the stored excess exactly."""
    from discell.model import metrics as M

    rng = np.random.default_rng(9)
    n_train, n_test, k = 3000, 30000, 3
    n = n_train + n_test
    t = rng.integers(0, k, n)
    z = rng.standard_normal((n, 4))
    v = np.hstack([(rng.random((n, 1)) < 0.01),                    # rare type
                   z[:, :1] + rng.standard_normal((n, 1)),         # comp column
                   rng.standard_normal((n, 2))]).astype(np.float32)
    vbar = np.stack([v[t == g].mean(axis=0) for g in range(k)])
    vbar[:, 0] = 0.7                                  # the planted offset
    train = np.arange(n) < n_train
    ref_cols = M._probe_columns(z, t, v, vbar, train, ~train, 0, 2,
                                M._ridge_fit_predict)
    exact = ((v[~train].astype(np.float64)
              - vbar.astype(np.float64)[t[~train]]) ** 2).mean(axis=0)
    drift = np.abs(ref_cols["baseline"] - exact) / exact
    assert drift[0] > 1e-4                            # the planted rounding

    cells = B.probe_cells(z, t, v, vbar, train, ~train, n_comp=2, seed=0,
                          n_perm=2, family="ridge")
    assert np.allclose(cells["baseline"].mean(axis=0), exact, rtol=1e-9)
    ref = M.probe_gain_per_block(z, t, v, vbar, train, ~train, n_comp=2,
                                 seed=0, n_perm=2)
    stat = B.probe_statistic(cells)(cells, np.ones(len(cells["test_rows"])))
    assert stat["comp_excess"] == pytest.approx(ref["comp"]["excess"], abs=1e-9)
    assert stat["img_excess"] == pytest.approx(ref["img"]["excess"], abs=1e-9)


def _split_half_ceiling(x_a, x_b, mult_a, mult_b, seed, n_splits=4):
    """The bootstrap's per-draw split-half ceiling, in numpy (same halves,
    MIN_RATE floor and Spearman-Brown step as transport_mean_bootstrap)."""
    from discell.model import transport as T

    rng = np.random.default_rng(seed)
    halves = [B._split_weights(m, rng, n_splits) for m in (mult_a, mult_b)]
    corrs = []
    for s in range(n_splits):
        shifts = []
        for side in (0, 1):
            rate_a = halves[0][s, side] @ x_a / halves[0][s, side].sum()
            rate_b = halves[1][s, side] @ x_b / halves[1][s, side].sum()
            shift = (np.log(np.maximum(rate_b, T.MIN_RATE))
                     - np.log(np.maximum(rate_a, T.MIN_RATE)))
            shifts.append(shift - shift.mean())
        corrs.append(np.corrcoef(*shifts)[0, 1])
    r = float(np.mean(corrs))
    return 2 * r / (1 + r) if r > 0 else 0.0


def _planted_panel(rng, n_cells=600, n_genes=150, per_tile=6, tile_um=200.0):
    """One transport panel on a planted slide: niche B's rates are A's times
    exp(shift), counts Poisson at depth 1000; the prediction is the shift
    plus noise. Cells sit ``per_tile`` to a 200 um tile."""
    import scipy.sparse as sp

    from discell.model import transport as T

    base = np.exp(rng.normal(np.log(1e-3), 0.8, n_genes))
    shift = rng.normal(0.0, 0.05, n_genes)
    rates = np.vstack([np.tile(base, (n_cells, 1)),
                       np.tile(base * np.exp(shift), (n_cells, 1))])
    x_rate = sp.csr_matrix(rng.poisson(rates * 1000.0) / 1000.0)
    tiles = np.arange(2 * n_cells) // per_tile
    side = int(np.ceil(np.sqrt(tiles.max() + 1)))
    positions = np.stack([(tiles % side) * tile_um, (tiles // side) * tile_um], 1)
    positions = positions + rng.uniform(1, tile_um - 1, positions.shape)
    rows_a, rows_b = np.arange(n_cells), np.arange(n_cells, 2 * n_cells)
    obs_a = np.asarray(x_rate[rows_a].mean(axis=0)).ravel()
    obs_b = np.asarray(x_rate[rows_b].mean(axis=0)).ravel()
    keep = (obs_a > T.MIN_RATE) & (obs_b > T.MIN_RATE)
    observed = np.log(obs_b[keep] + T.EPS) - np.log(obs_a[keep] + T.EPS)
    prediction = shift[keep] + rng.normal(0.0, 0.02, int(keep.sum()))
    panel = {"rows_a": rows_a, "rows_b": rows_b, "keep": keep,
             "prediction": prediction, "overlap_flag": True, "trusted": True,
             "r2": T.score_shift(prediction, observed)["r2"],
             "noise_ceiling": T.noise_ceiling(x_rate, rows_a, rows_b, keep, rng)}
    return panel, x_rate, positions


def test_duplicated_resample_keeps_the_unduplicated_split_half_ceiling():
    """Planted: every cell resampled twice is the same sample; its split-half
    ceiling must be the unduplicated one. Splitting copies put a cell in
    both halves and pushed the ceiling toward 1 (the finalL transport CIs
    that excluded their own points)."""
    rng = np.random.default_rng(11)
    panel, x_rate, _ = _planted_panel(rng)
    genes = np.flatnonzero(panel["keep"])
    x_a = x_rate[panel["rows_a"]][:, genes].toarray()
    x_b = x_rate[panel["rows_b"]][:, genes].toarray()
    ones_a, ones_b = np.ones(len(x_a)), np.ones(len(x_b))
    single = _split_half_ceiling(x_a, x_b, ones_a, ones_b, seed=3)
    double = _split_half_ceiling(x_a, x_b, 2 * ones_a, 2 * ones_b, seed=3)
    assert single < 0.6                     # a noisy panel, as the real ones
    assert double == pytest.approx(single, abs=1e-9)


def test_transport_fraction_of_ceiling_ci_contains_its_point():
    rng = np.random.default_rng(12)
    panel, x_rate, positions = _planted_panel(rng)
    boot = B.transport_mean_bootstrap([panel], x_rate, positions, n=200,
                                      seed=0, device="cpu")
    entry = boot["reads"]["transport_of_ceiling"]
    lo, hi = entry["ci95"]
    assert lo <= entry["estimate"] <= hi


def _clustered_slide(rng, base, shift, pred, n_tiles=40, per_tile=20,
                     tile_sd=0.15, tile_um=200.0):
    """One planted panel on a clustered slide: per niche ``n_tiles`` tiles of
    ``per_tile`` cells sharing a per-tile, per-gene lognormal rate effect;
    niche B's rates are A's times exp(shift); counts Poisson at depth 1000;
    the prediction *pred* is fixed (the fit is conditioned on)."""
    import scipy.sparse as sp

    from discell.model import transport as T

    rows = []
    for niche in (0, 1):
        mu = base * (np.exp(shift) if niche else 1.0)
        effect = np.exp(rng.normal(0.0, tile_sd, (n_tiles, len(base)))
                        - tile_sd ** 2 / 2)
        rows.append(rng.poisson(np.repeat(mu * effect, per_tile, axis=0)
                                * 1000.0) / 1000.0)
    x_rate = sp.csr_matrix(np.vstack(rows))
    tiles = np.arange(x_rate.shape[0]) // per_tile
    side = int(np.ceil(np.sqrt(tiles.max() + 1)))
    positions = np.stack([(tiles % side) * tile_um, (tiles // side) * tile_um], 1)
    positions = positions + rng.uniform(1, tile_um - 1, positions.shape)
    n = n_tiles * per_tile
    rows_a, rows_b = np.arange(n), np.arange(n, 2 * n)
    obs_a = np.asarray(x_rate[rows_a].mean(axis=0)).ravel()
    obs_b = np.asarray(x_rate[rows_b].mean(axis=0)).ravel()
    keep = (obs_a > T.MIN_RATE) & (obs_b > T.MIN_RATE)
    observed = np.log(obs_b[keep] + T.EPS) - np.log(obs_a[keep] + T.EPS)
    panel = {"rows_a": rows_a, "rows_b": rows_b, "keep": keep,
             "prediction": pred[keep], "overlap_flag": True, "trusted": True,
             "r2": T.score_shift(pred[keep], observed)["r2"],
             "noise_ceiling": T.noise_ceiling(x_rate, rows_a, rows_b, keep, rng)}
    return panel, x_rate, positions


def test_half_tile_subsampling_covers_the_fraction_of_ceiling():
    """Planted coverage on clustered cells (tile random effects, point
    ceiling ~0.85): over independent slides of one population, the half-tile
    subsampling interval must cover the estimator's mean ~95 % of the time.
    Measured at 400 slides x 200 draws: 0.940 and 0.960 on two populations
    (the reverse-form interval: 0.667). Lower ceilings cover less -- 0.82-0.88
    at ~0.6, 0.78 at ~0.52 -- see the final-repair report."""
    pop = np.random.default_rng(0)
    base = np.exp(pop.normal(np.log(2e-3), 0.6, 150))
    shift = pop.normal(0.0, 0.08, 150)
    pred = shift + pop.normal(0.0, 0.04, 150)
    rng = np.random.default_rng(1)
    points, cis = [], []
    for rep in range(150):
        panel, x_rate, positions = _clustered_slide(rng, base, shift, pred)
        entry = B.transport_mean_bootstrap(
            [panel], x_rate, positions, n=100, seed=rep,
            device="cpu")["reads"]["transport_of_ceiling"]
        points.append(entry["estimate"])
        cis.append(entry["ci95"])
    cis, target = np.array(cis), float(np.mean(points))
    coverage = float(np.mean((cis[:, 0] <= target) & (target <= cis[:, 1])))
    assert 0.88 <= coverage <= 0.99, coverage
