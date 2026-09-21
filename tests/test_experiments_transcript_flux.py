"""Planted worlds for the transcript-flux estimator (devlog 2026-09-17)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from discell.experiments import transcript_flux as tf


def chain_world(n_cells: int = 5, pitch: float = 10.0):
    """Cells in a row, nucleus at the centroid, chain edges, all eligible."""
    nucleus_xy = np.column_stack([np.arange(n_cells) * pitch,
                                  np.zeros(n_cells)]).astype(float)
    ei = np.arange(n_cells - 1)
    ej = ei + 1
    adjacency = tf.adjacency_csr(ei, ej, n_cells, np.ones(n_cells, bool))
    return nucleus_xy, adjacency


def slot_lookup(adjacency):
    host = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    return {(int(h), int(j)): int(s)
            for h, j, s in zip(host, adjacency.indices, adjacency.data)}


def test_weights_are_monotone_shares():
    s = np.linspace(-20, 20, 101)
    for form in tf.FORMS:
        w = tf.weights(s, 4.0, form)
        assert w.min() >= 0 and w.max() <= 1
        assert np.all(np.diff(w) >= -1e-12)
    assert tf.weights(np.array([0.0]), 4.0, "ramp")[0] == pytest.approx(0.5)
    assert tf.weights(np.array([0.0]), 4.0, "logistic")[0] == pytest.approx(0.5)


def test_segment_argmin_picks_the_first_minimum():
    values = np.array([3.0, 1.0, 1.0, 5.0, 2.0])
    starts = np.array([0, 3])
    counts = np.array([3, 2])
    assert tf.segment_argmin(values, starts, counts).tolist() == [1, 4]


def test_planted_flux_recovers_beta_and_kappa():
    """Transcripts planted on a neighbour's nucleus must be fully attributed."""
    nucleus_xy, adjacency = chain_world(pitch=40.0)
    slots = slot_lookup(adjacency)
    planted = {(1, 0): 30, (1, 2): 10, (2, 3): 20, (3, 2): 5}
    cell_of, xy = [], []
    for (i, j), n in planted.items():
        cell_of += [i] * n
        xy += [nucleus_xy[j]] * n
    # decoys deep inside their own cell: nearest other nucleus is a neighbour
    # but they sit on their own nucleus, so every weight form must ignore them.
    for i in range(5):
        cell_of += [i] * 100
        xy += [nucleus_xy[i]] * 100
    cell_of = np.asarray(cell_of)
    xy = np.asarray(xy, dtype=float)

    acc = tf.FluxAccumulator(adjacency.nnz, [(b, f) for b in tf.BANDS
                                             for f in tf.FORMS])
    tf.flux_from_transcripts(cell_of, xy, nucleus_xy, adjacency, acc)

    total = np.full(5, 200.0)
    for key, flux in acc.flux.items():
        for (i, j), n in planted.items():
            assert flux[slots[(i, j)]] == pytest.approx(n, abs=0.02), key
        assert flux.sum() == pytest.approx(sum(planted.values()), abs=0.05), key
        beta, kappa = tf.beta_and_kappa(flux, adjacency.indptr, total)
        assert beta[slots[(1, 0)]] == pytest.approx(30 / 40, abs=1e-3)
        assert beta[slots[(1, 2)]] == pytest.approx(10 / 40, abs=1e-3)
        assert beta[slots[(2, 3)]] == pytest.approx(1.0, abs=1e-3)
        assert kappa[1] == pytest.approx(40 / 200, abs=1e-4)
        assert kappa[2] == pytest.approx(20 / 200, abs=1e-4)
        assert kappa[3] == pytest.approx(5 / 200, abs=1e-4)
        assert kappa[0] < 1e-5
        # beta rows are shares on every cell that received anything
        rows = np.add.reduceat(np.append(beta, 0.0), adjacency.indptr[:-1])
        assert rows[[1, 2, 3]] == pytest.approx(np.ones(3), abs=1e-6)


def test_band_gates_the_far_side_only_through_the_weight():
    """A transcript 1 um past the bisector counts fully only for hard."""
    nucleus_xy, adjacency = chain_world(n_cells=2)
    acc = tf.FluxAccumulator(adjacency.nnz, [(2.0, "hard"), (2.0, "ramp"),
                                             (8.0, "ramp")])
    # midpoint is 5.0; put it at 6.0 -> s = 6 - 4 = 2
    tf.flux_from_transcripts(np.array([0]), np.array([[6.0, 0.0]]),
                             nucleus_xy, adjacency, acc)
    assert acc.flux[(2.0, "hard")].sum() == pytest.approx(1.0)
    assert acc.flux[(2.0, "ramp")].sum() == pytest.approx(1.0)
    assert acc.flux[(8.0, "ramp")].sum() == pytest.approx((2 + 8) / 16)


def test_homotypic_world_shows_no_excess_and_a_leak_world_does():
    """Check (1) on planted gene content."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 40
    _, adjacency = chain_world(n_cells)
    slots = slot_lookup(adjacency)
    rho = rng.dirichlet(np.ones(n_genes) * 0.3, size=2)
    type_index = rng.integers(0, 2, size=n_cells)

    def band_matrix(source_of_edge):
        rows, cols, vals = [], [], []
        for (i, j), slot in slots.items():
            draw = rng.choice(n_genes, size=200, p=rho[source_of_edge(i, j)])
            for g, c in zip(*np.unique(draw, return_counts=True)):
                rows.append(slot); cols.append(g); vals.append(float(c))
        return sp.csr_matrix((vals, (rows, cols)),
                             shape=(adjacency.nnz, n_genes))

    own = tf.gene_content_check(band_matrix(lambda i, j: type_index[i]),
                                adjacency, type_index, rho)
    leak = tf.gene_content_check(band_matrix(lambda i, j: type_index[j]),
                                 adjacency, type_index, rho)
    # homotypic edges can carry no excess by construction (one profile)
    assert own["homotypic"]["excess_neighbour_minus_own"] == pytest.approx(0.0)
    assert leak["homotypic"]["excess_neighbour_minus_own"] == pytest.approx(0.0)
    # heterotypic: content drawn from the host shows no excess, from the
    # neighbour it does, and clearly so
    assert own["heterotypic"]["excess_neighbour_minus_own"] < 0.0
    assert leak["heterotypic"]["excess_neighbour_minus_own"] > 0.3
    assert leak["heterotypic"]["frac_neighbour_gt_own"] == pytest.approx(1.0)

    # the within-cell control: a core drawn from the host must make the leak
    # world's band look more like the neighbour than its own deep transcripts
    core = band_matrix(lambda i, j: type_index[i])
    leak_c = tf.gene_content_check(band_matrix(lambda i, j: type_index[j]),
                                   adjacency, type_index, rho, core_genes=core)
    own_c = tf.gene_content_check(band_matrix(lambda i, j: type_index[i]),
                                  adjacency, type_index, rho, core_genes=core)
    assert leak_c["heterotypic"]["band_minus_core_cos_to_neighbour"] > 0.3
    assert own_c["heterotypic"]["band_minus_core_cos_to_neighbour"] == \
        pytest.approx(0.0, abs=0.05)
    assert own_c["homotypic"]["band_minus_core_cos_to_neighbour"] == \
        pytest.approx(0.0, abs=0.05)


def test_neighbours_without_a_nucleus_are_dropped():
    nucleus_xy, _ = chain_world(3)
    eligible = np.array([True, True, False])
    adjacency = tf.adjacency_csr(np.array([0, 1]), np.array([1, 2]), 3, eligible)
    assert adjacency.nnz == 2                      # only 0<->1 survives
    assert set(zip(*adjacency.nonzero())) <= {(0, 1), (1, 0)}


def test_graded_admixture_is_monotone_and_shows_the_check_s_blind_spot():
    """The planted leak world is a *share*, not a 100% swap (refuter's point).

    The pre-registered excess is scored against pooled type profiles, so at
    small admixtures it can sit at or below zero even though the band really
    does carry the neighbour's transcripts.  The curve must be monotone in p
    and must cross zero somewhere well above the 10-15% the model posits --
    that crossing is the check's blind spot, and it is what turns a negative
    excess on a real slide from a verdict into a power statement.
    """
    rng = np.random.default_rng(1)
    n_genes, n_edges, n_types = 300, 400, 4
    rho = rng.dirichlet(np.ones(n_genes) * 0.2, size=n_types)
    ti = rng.integers(0, n_types, size=n_edges)
    tj = (ti + 1 + rng.integers(0, n_types - 1, size=n_edges)) % n_types
    counts = rng.integers(20, 120, size=n_edges)
    rho_n = rho / np.linalg.norm(rho, axis=1, keepdims=True)

    excess = []
    for p in (0.0, 0.1, 0.2, 0.5, 1.0):
        vec = tf.simulate_band(counts, ti, tj, rho, p, rng)
        assert np.asarray(vec.sum(axis=1)).ravel().tolist() == counts.tolist()
        excess.append(tf._excess_of(vec, ti, tj, rho_n).mean())

    assert np.all(np.diff(excess) > 0), excess     # monotone in the admixture
    assert excess[0] < 0                            # p = 0 is not at zero
    assert excess[-1] > 0.3                         # a full swap is obvious
    # the blind spot: a 10% admixture is still scored negative
    assert excess[1] < 0


def test_admixture_power_recovers_a_planted_share():
    """Feed the power machinery a panel planted at a known p; it must find it."""
    rng = np.random.default_rng(2)
    n_genes, n_edges, n_types = 400, 1500, 5
    rho = rng.dirichlet(np.ones(n_genes) * 0.2, size=n_types)
    ti = rng.integers(0, n_types, size=n_edges)
    tj = (ti + 1 + rng.integers(0, n_types - 1, size=n_edges)) % n_types
    counts = rng.integers(30, 150, size=n_edges)
    truth = 0.25
    vec = tf.simulate_band(counts, ti, tj, rho, truth, rng)
    rho_n = rho / np.linalg.norm(rho, axis=1, keepdims=True)
    ex = tf._excess_of(vec, ti, tj, rho_n)
    panel = {"homo": np.zeros(n_edges, bool), "ti": ti, "tj": tj, "n": counts,
             "rho_n": rho_n, "c_nb": ex, "c_own": np.zeros(n_edges),
             "core_gap": None}
    out = tf.admixture_power(panel, rho, seed=0, n_boot=50)
    assert out["implied_p"] == pytest.approx(truth, abs=0.06)
    lo, hi = out["implied_p_2se"]
    assert lo <= truth <= hi


def test_bootstrap_did_is_centred_on_a_planted_gap():
    rng = np.random.default_rng(3)
    homo = np.zeros(2000, bool)
    homo[1000:] = True
    gap = np.where(homo, rng.normal(0.10, 0.2, 2000), rng.normal(0.16, 0.2, 2000))
    out = tf.bootstrap_did({"core_gap": gap, "homo": homo}, n_boot=400, seed=0)
    assert out["difference_in_differences"] == pytest.approx(0.06, abs=0.02)
    lo, hi = out["ci95"]
    assert lo < out["difference_in_differences"] < hi
    assert out["excludes_zero"]


def test_deflate_profiles_removes_planted_influx():
    """A profile built as (1-p) clean + p neighbour must deflate back to clean."""
    rng = np.random.default_rng(4)
    n_genes, p = 200, 0.2
    clean = rng.dirichlet(np.ones(n_genes) * 0.3, size=2)
    # two cells, one of each type, each bordering only the other
    adjacency = tf.adjacency_csr(np.array([0]), np.array([1]), 2,
                                 np.ones(2, bool))
    type_index = np.array([0, 1])
    observed = (1 - p) * clean + p * clean[::-1]
    observed /= observed.sum(axis=1, keepdims=True)
    back = tf.deflate_profiles(observed, np.ones(adjacency.nnz), adjacency,
                               type_index, p)
    # exact inversion up to the clip at zero: genes the influx over-explains
    # are floored, which costs a little mass on renormalisation
    assert np.abs(back - clean).max() < 5e-3
    cos = float((back * clean).sum()
                / (np.linalg.norm(back) * np.linalg.norm(clean)))
    assert cos > 0.998


# --------------------------------------------------------------------------
# the three decisive tests (devlog 2026-09-17, todo 5.4-5.7)
# --------------------------------------------------------------------------


def block_world(n_cells: int = 240, n_types: int = 3, n_genes: int = 30,
                seed: int = 0):
    """A ring of cells with three block profiles and a directed ring graph."""
    rng = np.random.default_rng(seed)
    ei = np.arange(n_cells - 1)
    ej = ei + 1
    adjacency = tf.adjacency_csr(ei, ej, n_cells, np.ones(n_cells, bool))
    type_index = rng.integers(0, n_types, n_cells)
    rho = np.full((n_types, n_genes), 0.2 / n_genes)
    per = n_genes // n_types
    for t in range(n_types):
        rho[t, t * per:(t + 1) * per] += 0.8 / per
    rho /= rho.sum(axis=1, keepdims=True)
    return adjacency, type_index, rho


def planted_bands(adjacency, type_index, rho, p_of_slot, counts, seed=0):
    """Band matrix (n_slots, G) drawn at ``(1 - p) rho_host + p rho_neighbour``."""
    rng = np.random.default_rng(seed)
    host = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    order = np.argsort(adjacency.data)
    ti, tj = type_index[host[order]], type_index[adjacency.indices[order]]
    rows, cols = [], []
    for slot in range(adjacency.nnz):
        p = float(p_of_slot[slot])
        mix = (1 - p) * rho[ti[slot]] + p * rho[tj[slot]]
        draw = rng.choice(rho.shape[1], size=int(counts[slot]), p=mix / mix.sum())
        rows.append(np.full(len(draw), slot))
        cols.append(draw)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                         shape=(adjacency.nnz, rho.shape[1])).tocsr()


def test_tile_shuffle_permutes_nuclei_and_keeps_the_point_field():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 400, size=(300, 2))
    nuc = pos + rng.normal(0, 2.0, size=(300, 2))
    eligible = np.ones(300, bool)
    shuffled = tf.tile_shuffled(nuc, pos, eligible, seed=1, tile_um=200.0)
    # the multiset of nucleus positions inside each tile is unchanged
    key = np.floor(pos / 200.0).astype(int)
    flat = key[:, 0] * 1000 + key[:, 1]
    for value in np.unique(flat):
        take = flat == value
        assert np.allclose(np.sort(nuc[take], axis=0),
                           np.sort(shuffled[take], axis=0))
    assert not np.allclose(nuc, shuffled)
    # the offset mode keeps every nucleus within the spread of the offsets
    off = tf.tile_shuffled(nuc, pos, eligible, seed=1, mode="offset")
    assert np.abs(off - pos).max() <= np.abs(nuc - pos).max() + 1e-9


def test_a_leak_free_world_has_no_corrected_kappa():
    """If the observed and shuffled geometries agree, the correction is zero."""
    nucleus_xy, adjacency = chain_world(n_cells=40, pitch=20.0)
    n = adjacency.shape[0]
    acc = tf.FluxAccumulator(adjacency.nnz, [tf.PRIMARY])
    acc.flux[tf.PRIMARY][:] = 1.0
    same = tf.FluxAccumulator(adjacency.nnz, [tf.PRIMARY])
    same.flux[tf.PRIMARY][:] = 1.0
    slide = {"adjacency": adjacency, "total_counts": np.full(n, 100.0),
             "type_index": np.arange(n) % 2,
             "type_names": np.array(["a", "b"])}
    out = tf.test_55_offcentring(slide, {"nucleus": acc, "shuffled": same})
    assert out["distributions"]["corrected"]["quantiles"]["50"] == pytest.approx(0.0)
    assert out["bar_corrected_median_above_0.02"] is False


def test_shuffled_world_is_recovered_as_geometry_only():
    """Half the observed flux being geometry leaves exactly half as corrected."""
    nucleus_xy, adjacency = chain_world(n_cells=200, pitch=20.0)
    n = adjacency.shape[0]
    obs = tf.FluxAccumulator(adjacency.nnz, [tf.PRIMARY])
    obs.flux[tf.PRIMARY][:] = 2.0
    shuf = tf.FluxAccumulator(adjacency.nnz, [tf.PRIMARY])
    shuf.flux[tf.PRIMARY][:] = 1.0
    rng = np.random.default_rng(0)
    slide = {"adjacency": adjacency,
             "total_counts": rng.uniform(40.0, 60.0, n),
             "type_index": np.arange(n) % 5,
             "type_names": np.array(list("abcde"))}
    out = tf.test_55_offcentring(slide, {"nucleus": obs, "shuffled": shuf})
    obs_med = out["distributions"]["nucleus"]["quantiles"]["50"]
    cor_med = out["distributions"]["corrected"]["quantiles"]["50"]
    assert cor_med == pytest.approx(obs_med / 2.0, rel=1e-6)
    assert out["bar_corrected_median_above_0.02"] is True
    assert out["spearman_vs_observed"]["corrected"] == pytest.approx(1.0)


def test_specific_gene_set_gives_every_type_genes():
    n_types, n_genes = 4, 40
    nuc = np.full((n_types, n_genes), 10.0)
    per = n_genes // n_types
    for t in range(n_types):
        nuc[t, t * per:(t + 1) * per] = 1000.0
    genes, per_type = tf.specific_gene_set(nuc, nuc * 2.0, ratio=3.0,
                                           min_counts=200.0, top_n=5)
    assert len(genes) > 0
    assert all(v["n_used"] > 0 for v in per_type.values())


def test_54_recovers_a_planted_admixture_and_crosses_early():
    adjacency, type_index, rho = block_world()
    slots = adjacency.nnz
    band = planted_bands(adjacency, type_index, rho,
                         np.full(slots, 0.13), np.full(slots, 60), seed=3)
    panel = tf.restricted_panel(band, adjacency, type_index, rho, None, 10)
    out = tf.test_54_content_power(panel, seed=0, n_boot=100)
    assert out["excess"]["implied_p"] == pytest.approx(0.13, abs=0.05)
    lo, hi = out["excess"]["implied_p_2se"]
    assert lo <= 0.13 <= hi
    # the crossing is pinned at p = 0.5 by the symmetry of the mixture, on any
    # gene subset; what the subset buys is separation, not an earlier crossing
    assert out["zero_crossing_p_mean_excess"] == pytest.approx(0.5, abs=0.05)
    assert out["separable_at_p0.13"] is True


def test_56_recovers_a_planted_per_edge_admixture():
    """p_ij planted proportional to the flux share must come back through deciles."""
    adjacency, type_index, rho = block_world(n_cells=600, seed=1)
    slots = adjacency.nnz
    rng = np.random.default_rng(5)
    share = rng.uniform(0.0, 0.3, size=slots)
    band = planted_bands(adjacency, type_index, rho, share,
                         np.full(slots, 120), seed=4)
    panel = tf.restricted_panel(band, adjacency, type_index, rho, None, 1,
                                require_full=30)
    curve = tf.test_54_content_power(panel, seed=0, n_boot=50)["curve"]
    total = np.full(adjacency.shape[0], 1.0)
    out = tf.test_56_per_edge(panel, share, total, curve)
    assert out["spearman_flux_share_vs_implied_p"] >= 0.5
    assert 0.5 <= out["calibration_slope"] <= 2.0


def test_did_power_separates_a_planted_band_from_p0():
    adjacency, type_index, rho = block_world(n_cells=600, seed=2)
    slots = adjacency.nnz
    band = planted_bands(adjacency, type_index, rho, np.full(slots, 0.0),
                         np.full(slots, 80), seed=6)
    panel = tf.restricted_panel(band, adjacency, type_index, rho, None, 10,
                                core_genes=band)
    out = tf.did_power(panel, seed=0, n_boot=100)
    curve = {r["p"]: r["did"] for r in out["curve"]}
    assert curve[0.0] == pytest.approx(0.0, abs=0.01)
    assert curve[0.3] > curve[0.05] > curve[0.0]
