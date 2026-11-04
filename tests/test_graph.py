"""The graph and tile machinery, on toy graphs where the answer is known by hand."""

from __future__ import annotations

import numpy as np
import pytest

from discell.graph import (
    build_graph,
    graph_from_edges,
    read_graph,
    rings,
    spatial_tiles,
    split_tiles,
    tile_batch,
)
from discell.io import fold_labels


def path_graph(n=6, long_edge=None):
    """0-1-2-...-(n-1), unit faces, 10 um spacing; one edge optionally long."""
    ei = np.arange(n - 1)
    ej = ei + 1
    dist = np.full(n - 1, 10.0)
    if long_edge is not None:
        dist[long_edge] = 100.0
    return graph_from_edges(ei, ej, np.ones(n - 1), dist, n,
                            type_index=np.arange(n) % 2, n_types=2)


def test_prune_drops_the_long_edge():
    g = path_graph(6, long_edge=2)               # cuts 2-3
    assert len(g.edge_i) == 4
    assert g.degrees.tolist() == [1, 2, 1, 1, 2, 1]
    assert not g.isolated.any()


def test_pruning_can_isolate_a_cell_and_the_flag_catches_it():
    g = graph_from_edges(np.array([0, 1]), np.array([1, 2]), np.ones(2),
                         np.array([100.0, 100.0]), 3)
    assert g.isolated.tolist() == [True, True, True]


def test_beta_rows_sum_to_one_per_destination():
    sums = np.asarray(path_graph(6).in_edges.sum(axis=1)).ravel()
    assert np.allclose(sums, 1.0)


def test_beta_follows_face_and_distance():
    g = graph_from_edges(np.array([0, 0]), np.array([1, 2]), np.array([1.0, 2.0]),
                         np.array([10.0, 20.0]), 3, tau_um=20.0)
    raw = np.array([1.0 * np.exp(-0.5), 2.0 * np.exp(-1.0)])
    assert g.in_edges[0, 1] == pytest.approx(raw[0] / raw.sum())
    assert g.in_edges[0, 2] == pytest.approx(raw[1] / raw.sum())
    assert g.in_edges[1, 0] == pytest.approx(1.0)


def test_y_excludes_ego_and_sums_to_one():
    g = path_graph(4)                            # types 0, 1, 0, 1
    assert np.allclose(g.y.sum(axis=1), 1.0)
    assert np.allclose(g.y[0], [0.0, 1.0])
    assert np.allclose(g.y[1], [1.0, 0.0])
    assert g.ybar_t.shape == (2, 2)


def test_rings_on_a_path():
    g = path_graph(6)
    r1, r2 = rings(g.in_edges, np.array([0]))
    assert r1.tolist() == [1] and r2.tolist() == [2]
    r1, r2 = rings(g.in_edges, np.array([2, 3]))
    assert r1.tolist() == [1, 4] and r2.tolist() == [0, 5]


def test_rings_are_disjoint_and_closed_on_a_random_graph():
    rng = np.random.default_rng(0)
    n = 300
    ei = rng.integers(0, n - 1, 800)
    ej = ei + rng.integers(1, 5, 800)
    keep = ej < n
    pairs = np.unique(np.stack([ei[keep], ej[keep]], axis=1), axis=0)
    g = graph_from_edges(pairs[:, 0], pairs[:, 1], np.ones(len(pairs)),
                         np.full(len(pairs), 10.0), n)
    seeds = np.unique(rng.integers(0, n, 40))
    r1, r2 = rings(g.in_edges, seeds)
    assert not np.intersect1d(seeds, r1).size
    assert not np.intersect1d(np.union1d(seeds, r1), r2).size
    hop1 = np.unique(g.in_edges[seeds].indices)
    assert np.isin(hop1, np.union1d(seeds, r1)).all()


def test_tiles_partition_exactly_and_respect_the_cap():
    pos = np.random.default_rng(1).uniform(0, 1000, size=(5000, 2))
    tiles = spatial_tiles(pos, max_cells=400)
    joined = np.concatenate(tiles)
    assert len(joined) == 5000 and len(np.unique(joined)) == 5000
    assert max(len(t) for t in tiles) <= 400
    assert min(len(t) for t in tiles) > 200


def test_split_is_seeded_disjoint_and_complete():
    tiles = spatial_tiles(np.random.default_rng(2).uniform(0, 1, (4000, 2)), 200)
    train, val = split_tiles(tiles, 0.15, seed=0)
    assert len(val) == round(0.15 * len(tiles)) and len(train) + len(val) == len(tiles)
    again = split_tiles(tiles, 0.15, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(val, again[1]))
    other = split_tiles(tiles, 0.15, seed=1)[1]
    assert not all(np.array_equal(a, b) for a, b in zip(val, other))


def test_tile_batch_layout_and_spillover_edges():
    g = path_graph(8)
    b = tile_batch(g, np.array([3, 4]))
    assert b.nodes.tolist() == [3, 4, 2, 5, 1, 6]  # [seeds | ring1 | ring2]
    assert b.n_seeds == 2 and b.n_ring1 == 2 and b.n_context == 4
    for s in range(b.n_seeds):
        assert b.spillover_beta[b.spillover_dst == s].sum() == pytest.approx(1.0)
    assert set(b.gat_dst.tolist()) == {0, 1, 2, 3}
    assert b.gat_src.max() < len(b.nodes)
    n_spillover = len(b.spillover_src)
    assert np.array_equal(b.spillover_src, b.gat_src[:n_spillover])
    assert (b.spillover_dst < b.n_seeds).all()


def test_tile_batch_isolated_seed_contributes_no_edges():
    g = graph_from_edges(np.array([0, 1]), np.array([1, 2]), np.ones(2),
                         np.array([10.0, 100.0]), 4)
    b = tile_batch(g, np.array([2, 3]))
    assert b.n_seeds == 2 and b.n_ring1 == 0
    assert len(b.spillover_src) == 0 and len(b.gat_src) == 0


# -- labels and the AnnData round trip -------------------------------------


def test_fold_labels_sorts_and_reuses_the_panel_spelling():
    idx, names = fold_labels(["b", None, "a", "unassigned", np.nan])
    assert names.tolist() == ["a", "b", "unassigned"]
    assert idx.tolist() == [1, 2, 0, 2, 2]
    idx, names = fold_labels(["b", None])
    assert names.tolist() == ["Unassigned", "b"] and idx.dtype == np.int64


def test_build_graph_writes_the_contract(adata):
    g = build_graph(adata, "lineage")
    n = adata.n_obs
    assert adata.obsp["discell_beta"].shape == (n, n)
    assert (adata.obsp["discell_connectivities"] != adata.obsp["discell_connectivities"].T
            ).nnz == 0
    assert adata.obsm["discell_y"].shape == (n, len(adata.uns["discell"]["graph"]["types"]))
    meta = adata.uns["discell"]["graph"]
    assert meta["n_edges"] == len(g.edge_i) and meta["n_pruned"] >= 2
    assert "Unassigned" in meta["types"]
    connected = ~adata.obs["discell_isolated"].to_numpy()
    sums = np.asarray(adata.obsp["discell_beta"].sum(axis=1)).ravel()
    assert np.allclose(sums[connected], 1.0) and np.all(sums[~connected] == 0)


def test_read_graph_round_trips_through_h5ad(adata, tmp_path):
    import anndata as ad

    built = build_graph(adata, "lineage")
    adata.write_h5ad(tmp_path / "a.h5ad")
    back = read_graph(ad.read_h5ad(tmp_path / "a.h5ad"), "lineage")
    for field in ("edge_i", "edge_j", "beta", "degrees", "y", "ybar_t"):
        assert np.array_equal(getattr(built, field), getattr(back, field)), field
    assert (built.in_edges != back.in_edges).nnz == 0


def test_read_graph_refuses_changed_labels(adata):
    build_graph(adata, "lineage")
    adata.obs["lineage"] = adata.obs["lineage"].astype(object).where(
        adata.obs["lineage"] != "type0", "other")
    with pytest.raises(ValueError):
        read_graph(adata)
    with pytest.raises(ValueError):
        read_graph(adata, "cell_group")


def test_build_graph_needs_the_voronoi_input(adata):
    del adata.obsp["voronoi_shared_wall_um"]
    with pytest.raises(KeyError):
        build_graph(adata, "lineage")
