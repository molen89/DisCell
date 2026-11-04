"""Breakdown points (the two-level bootstrap over fits and tiles, the Bonferroni level,
kappa*), the tile draws, and a synthetic end-to-end sweep on CPU."""

from __future__ import annotations

import numpy as np
import pytest

from discell import sensitivity as S

GRID = S.KAPPA_GRID


def _entry(estimate, sd, n=4000, seed=0, method="percentile", c=1.0):
    rng = np.random.default_rng(seed)
    return {"estimate": float(estimate), "draws": estimate + sd * rng.standard_normal(n),
            "method": method, "c": c}


def _draws(means, sd=0.01, per_seed=None):
    """``{"m": {kappa: {seed: entry}}}`` whose seed mean at each grid point is *means*."""
    out = {}
    for i, k in enumerate(GRID):
        out[k] = {s: _entry(means[i] + (per_seed or {}).get((k, s), 0.0), sd,
                            seed=100 * i + s) for s in range(3)}
    return {"m": out}


def _kstar(means, m_s=8, **kw):
    return S.breakdown(_draws(means, **kw), n_findings=m_s)["members"]["m"]


def test_planted_breakdown_point():
    m = _kstar([0.30, 0.25, 0.20, 0.12, 0.005, -0.05], sd=0.01)
    assert m["status"] == "breaks" and m["kappa_star"] == 0.3
    assert m["reason"] == "interval contains 0" and m["sign"] == 1


def test_sign_flip_is_detected():
    # every interval excludes 0, but the sign turns at 0.2: the interval lies on the other side
    m = _kstar([0.3, 0.2, 0.1, -0.1, -0.2, -0.3], sd=0.005)
    assert m["status"] == "breaks" and m["kappa_star"] == 0.2
    assert m["reason"] == "interval on the other side of 0"


def test_a_far_seed_widens_the_interval():
    # seed 2 alone falls to -0.05 at 0.05; resampling the fits carries it into the interval
    m = _kstar([0.3] * 6, sd=0.001,
               per_seed={(0.05, 2): -0.35, (0.05, 0): 0.2, (0.05, 1): 0.2})
    assert m["status"] == "breaks" and m["kappa_star"] == 0.05
    assert m["reason"] == "interval contains 0"


def test_more_seeds_do_not_add_chances_to_fail():
    # ten seeds around 0.1, one of them just below 0: the interval of the mean still lies above 0
    est = [0.10, 0.12, 0.08, 0.11, 0.09, 0.13, 0.10, 0.07, 0.12, -0.005]
    p = S.pool([_entry(v, 0.01, seed=s) for s, v in enumerate(est)], alpha=0.05 / 7)
    assert S.holds(p, 1)


def test_more_seeds_narrow_the_interval():
    three = [_entry(v, 0.01, seed=s) for s, v in enumerate((0.08, 0.10, 0.12))]
    nine = [_entry(v, 0.01, seed=10 + s) for s, v in enumerate((0.08, 0.10, 0.12) * 3)]
    w3 = np.diff(S.pool(three, alpha=0.05 / 7)["ci"])[0]
    w9 = np.diff(S.pool(nine, alpha=0.05 / 7)["ci"])[0]
    assert w9 < w3


def test_no_finding_at_kappa_zero():
    m = _kstar([0.001, 0.2, 0.2, 0.2, 0.2, 0.2], sd=0.05)
    assert m["status"] == "no finding" and m["kappa_star"] is None


def test_above_the_grid():
    m = _kstar([-0.5] * 6, sd=0.01)
    assert m["status"] == "above the grid" and m["sign"] == -1


def test_gap_is_not_survival():
    d = _draws([0.5] * 6)
    d["m"][0.2] = {}
    m = S.breakdown(d, grid=GRID, seeds=(0, 1, 2), n_findings=8)["members"]["m"]
    assert m["status"] == "gap at kappa = 0.2"


def test_a_point_some_seed_lacks_is_a_gap():
    d = _draws([0.5] * 6)
    del d["m"][0.2][1]
    m = S.breakdown(d, n_findings=8)["members"]["m"]
    assert m["status"] == "gap at kappa = 0.2"


def test_bonferroni_level_from_the_number_of_findings():
    assert S.level(8) == pytest.approx(1 - 0.05 / 8)
    means = [0.02] * 6
    wide = S.breakdown(_draws(means, sd=0.015), n_findings=50)
    narrow = S.breakdown(_draws(means, sd=0.015), n_findings=1)
    assert narrow["members"]["m"]["status"] == "above the grid"
    assert wide["members"]["m"]["status"] == "no finding"
    lo50, hi50 = wide["members"]["m"]["trajectory"]["0"]["ci"]
    lo1, hi1 = narrow["members"]["m"]["trajectory"]["0"]["ci"]
    assert lo50 < lo1 and hi50 > hi1
    assert wide["level"] == pytest.approx(1 - 0.05 / 50)
    # the default is the number of findings given
    two = {"a": _draws(means)["m"], "b": _draws(means)["m"]}
    assert S.breakdown(two)["m_s"] == 2
    with pytest.raises(ValueError):
        S.level(0)


def test_pool_averages_draws_across_seeds():
    # three equal seeds with independent draws of sd 1: the mean has sd 1/sqrt 3
    seeds = [_entry(2.0, 1.0, n=200_000, seed=s) for s in range(3)]
    p = S.pool(seeds, alpha=0.05)
    assert p["estimate"] == pytest.approx(2.0)
    assert 0.5 * (p["ci"][1] - p["ci"][0]) == pytest.approx(1.96 / np.sqrt(3), rel=0.02)
    assert p["n_seeds"] == 3


def test_pool_adds_the_spread_between_seeds():
    # seeds at 1, 2, 3 with almost no tile noise: the interval reflects the seed spread
    seeds = [_entry(v, 1e-4, n=200_000, seed=s) for s, v in enumerate((1.0, 2.0, 3.0))]
    p = S.pool(seeds, alpha=0.05)
    assert p["estimate"] == pytest.approx(2.0)
    assert p["ci"][0] < 1.5 and p["ci"][1] > 2.5


def test_subsample_form_is_placed_on_the_point():
    seeds = [_entry(10.0, 1.0, n=100_000, seed=s, method="subsample", c=0.5)
             for s in range(3)]
    for e in seeds:
        e["draws"] = e["draws"] + 3.0
    p = S.pool(seeds, alpha=0.05)
    assert 0.5 * (p["ci"][0] + p["ci"][1]) == pytest.approx(10.0, abs=0.02)
    assert 0.5 * (p["ci"][1] - p["ci"][0]) == pytest.approx(0.5 * 1.96 / np.sqrt(3), rel=0.03)


def test_nan_padded_draws_are_dropped_and_mixed_forms_refused():
    a, b = _entry(1.0, 0.1, n=100), _entry(1.0, 0.1, n=100, seed=1)
    a["draws"][-10:] = np.nan
    p = S.pool([a, b], 0.05)
    assert 0 < p["n_draws"] < 100 and np.isfinite(p["ci"]).all()
    b["method"] = "subsample"
    with pytest.raises(ValueError, match="interval form"):
        S.pool([a, b], 0.05)


def test_ranking_pairs_on_shared_draws():
    d = np.random.default_rng(0).normal(size=100)
    members = {"a": {"estimate": 2.0, "draws": 2 + d, "method": "percentile", "c": 1.0},
               "b": {"estimate": 1.0, "draws": 1 + d, "method": "percentile", "c": 1.0}}
    pair = S.ranking_pairs(members, ["a", "b"])["a - b"]
    assert pair["estimate"] == 1.0 and np.allclose(pair["draws"], 1.0)


def test_draw_file_round_trip(tmp_path):
    members = {"x": {"estimate": 0.5, "draws": np.arange(5.0), "method": "percentile",
                     "c": 1.0},
               "y": {"estimate": -1.0, "draws": np.arange(3.0), "method": "subsample",
                     "c": 0.9}}
    path = S.write_draws(tmp_path / "f.npz", members, {"run": "r"})
    got = S.read_draws(path)
    assert got["x"]["estimate"] == 0.5 and got["x"]["draws"].tolist() == list(range(5))
    assert got["y"]["method"] == "subsample" and got["y"]["c"] == 0.9
    assert np.isnan(got["y"]["draws"][3:]).all()


def test_tile_index_groups_cells_by_square():
    xy = np.array([[10, 10], [190, 199], [201, 0], [0, 250], [399, 399]])
    ids, n = S.tile_index(xy, 200.0)
    assert n == 4
    assert ids[0] == ids[1] and len({ids[0], ids[2], ids[3], ids[4]}) == 4


def test_tile_draws_resample_whole_tiles():
    xy = np.array([[10, 10], [20, 20], [250, 10], [450, 10]], dtype=float)
    tiles, n = S.tile_index(xy)
    assert n == 3
    for w in S.tile_draws(tiles, n, 50, seed=3):
        assert w[0] == w[1]                    # cells of one tile share its weight
        assert w[[0, 2, 3]].sum() == 3         # n_tiles tiles drawn per resample


def test_half_tile_weights_keep_half_the_tiles():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, (500, 2))
    tiles, n_tiles = S.tile_index(xy)
    weights, c = S.half_tile_weights(xy, 20, seed=1)
    assert weights.shape == (20, 500)
    for w in weights:
        kept = np.unique(tiles[w > 0])
        assert len(kept) == n_tiles // 2
        assert set(np.unique(w)) <= {0.0, 1.0}
    assert c == pytest.approx(np.sqrt((n_tiles // 2) / (n_tiles - n_tiles // 2)))


def test_tile_bootstrap_covers_a_planted_mean():
    """Spatially dependent cells: the tile interval covers the true mean near its
    nominal rate; treating every cell as its own tile covers far less."""
    rng = np.random.default_rng(0)
    side, per_tile = 15, 10
    hits_tile = hits_cell = 0
    for s in range(100):
        tiles = np.repeat(np.arange(side * side), per_tile)
        xy = np.stack([(tiles % side) * 200.0, (tiles // side) * 200.0], 1)
        xy = xy + rng.uniform(1, 199, xy.shape)
        values = rng.standard_normal(side * side)[tiles] + rng.standard_normal(len(tiles))

        def mean(w, v=values):
            return float((w * v).sum() / w.sum())

        for tile_um, counter in ((200.0, "tile"), (1e-3, "cell")):
            e = S.finding_draws(xy, mean, n=300, seed=s, tile_um=tile_um)
            lo, hi = np.percentile(e["draws"], [2.5, 97.5])
            hit = lo <= 0.0 <= hi
            if counter == "tile":
                hits_tile += hit
            else:
                hits_cell += hit
    assert 0.85 <= hits_tile / 100 <= 1.0
    assert hits_cell / 100 < 0.75


def test_finding_draws_forms():
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 1000, (300, 2))
    v = rng.normal(size=300)

    def stat(w):
        return float((w * v).sum() / w.sum())

    p = S.finding_draws(xy, stat, n=30, seed=0)
    assert p["estimate"] == pytest.approx(v.mean()) and p["method"] == "percentile"
    assert p["c"] == 1.0 and len(p["draws"]) == 30
    h = S.finding_draws(xy, stat, n=30, seed=0, method="subsample")
    assert h["method"] == "subsample" and 0 < h["c"] <= 1.0
    with pytest.raises(ValueError):
        S.finding_draws(xy, stat, n=3, method="jackknife")


def test_synthetic_sweep_end_to_end(tmp_path):
    """2 kappa x 2 seeds on CPU: fit, a finding of each fit's latents, its tile
    draws, breakdown points; the sweep resumes from disk without refitting."""
    import discell as dc
    from discell.sim import simulate

    adata = simulate(n_cells=1200, n_genes=30, kappa=0.1, seed=0).to_anndata()
    dc.build_graph(adata, "cell_type")
    cfg = dc.Config(label_key="cell_type", image_key="X_image", d_z=4, d_w=2, hidden=32,
                    gat_dim=8, v_pcs=4, tile_cells=300, epochs=4, eval_every=2,
                    w_warmup_epochs=2, patience=100, device="cpu")
    kappas, seeds = (0.0, 0.2), (0, 1)
    models = S.fit_sweep(adata, cfg, kappas=kappas, seeds=seeds, out_dir=tmp_path)
    assert sorted(models) == [(k, s) for k in kappas for s in seeds]
    assert models[(0.2, 1)].config.kappa == 0.2 and models[(0.2, 1)].config.seed == 1
    assert (tmp_path / "k0.2_s1" / "model.pt").exists()

    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    first = labels == sorted(set(labels))[0]
    draws: dict = {"w_norm_type0_minus_rest": {}}
    for (kappa, seed), model in models.items():
        model.encode(adata)
        norm = np.linalg.norm(adata.obsm["X_discell_w"], axis=1)

        def finding(w, n=norm):
            return float((w * n * first).sum() / (w * first).sum()
                         - (w * n * ~first).sum() / (w * ~first).sum())

        draws["w_norm_type0_minus_rest"].setdefault(kappa, {})[seed] = S.finding_draws(
            xy, finding, n=50, seed=8190 + seed)
    table = S.breakdown(draws, grid=kappas)
    member = table["members"]["w_norm_type0_minus_rest"]
    assert member["status"] in ("breaks", "above the grid", "no finding")
    assert table["grid"] == [0.0, 0.2] and table["seeds"] == [0, 1]
    assert set(member["trajectory"]) == {"0", "0.2"}

    again = S.fit_sweep(adata, cfg, kappas=kappas, seeds=seeds, out_dir=tmp_path)
    assert again[(0.0, 0)].metrics["best"] == models[(0.0, 0)].metrics["best"]
