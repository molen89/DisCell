"""The graph/tile machinery, on toy graphs where the answer is known by hand."""

from __future__ import annotations

import numpy as np
import pytest

from discell.model.prepare import build_graph, rings, spatial_tiles, tile_batch


def path_graph(n=6, long_edge=None):
    """0-1-2-...-(n-1), unit faces, 10 um spacing; one edge optionally long."""
    ei = np.arange(n - 1)
    ej = ei + 1
    dist = np.full(n - 1, 10.0)
    if long_edge is not None:
        dist[long_edge] = 100.0
    return build_graph(ei, ej, np.ones(n - 1), dist, n,
                       type_index=np.arange(n) % 2, n_types=2)


# -- build_graph ----------------------------------------------------------


def test_prune_drops_the_long_edge_and_counts_it_per_cell():
    g = path_graph(6, long_edge=2)               # cuts 2-3
    assert len(g.edge_i) == 4
    assert g.pruned_per_cell.tolist() == [0, 0, 1, 1, 0, 0]
    assert g.degrees.tolist() == [1, 2, 1, 1, 2, 1]
    assert not g.isolated.any()


def test_pruning_can_isolate_a_cell_and_the_flag_catches_it():
    ei, ej = np.array([0, 1]), np.array([1, 2])
    g = build_graph(ei, ej, np.ones(2), np.array([100.0, 100.0]), 3)
    assert g.isolated.tolist() == [True, True, True]


def test_beta_rows_sum_to_one_per_destination():
    g = path_graph(6)
    sums = np.asarray(g.in_edges.sum(axis=1)).ravel()
    assert np.allclose(sums, 1.0)


def test_y_excludes_ego_and_sums_to_one():
    g = path_graph(4)                            # types 0,1,0,1
    assert np.allclose(g.y.sum(axis=1), 1.0)
    # cell 0's only neighbour is cell 1 (type 1): composition [0, 1]
    assert np.allclose(g.y[0], [0.0, 1.0])
    # cell 1 sees cells 0 and 2, both type 0
    assert np.allclose(g.y[1], [1.0, 0.0])
    assert g.ybar_t.shape == (2, 2)


# -- rings ----------------------------------------------------------------


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
    g = build_graph(pairs[:, 0], pairs[:, 1], np.ones(len(pairs)),
                    np.full(len(pairs), 10.0), n)
    seeds = np.unique(rng.integers(0, n, 40))
    r1, r2 = rings(g.in_edges, seeds)
    assert not np.intersect1d(seeds, r1).size
    assert not np.intersect1d(np.union1d(seeds, r1), r2).size
    # closure: every neighbour of a seed is in seeds ∪ ring1
    hop1 = np.unique(g.in_edges[seeds].indices)
    assert np.isin(hop1, np.union1d(seeds, r1)).all()


# -- tiles ----------------------------------------------------------------


def test_tiles_partition_exactly_and_respect_the_cap():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 1000, size=(5000, 2))
    tiles = spatial_tiles(pos, max_cells=400)
    joined = np.concatenate(tiles)
    assert len(joined) == 5000 and len(np.unique(joined)) == 5000
    assert max(len(t) for t in tiles) <= 400
    assert min(len(t) for t in tiles) > 200        # median split: within 2x


# -- tile_batch -----------------------------------------------------------


def test_tile_batch_layout_and_leak_edges():
    g = path_graph(8)
    b = tile_batch(g, np.array([3, 4]))
    assert b.nodes.tolist() == [3, 4, 2, 5, 1, 6]  # [seeds | ring1 | ring2]
    assert b.n_seeds == 2 and b.n_ring1 == 2 and b.n_context == 4
    # beta into each seed sums to 1
    for s in range(b.n_seeds):
        assert b.leak_beta[b.leak_dst == s].sum() == pytest.approx(1.0)
    # every context node has its in-edges; sources never reach outside the halo
    assert set(b.gat_dst.tolist()) == {0, 1, 2, 3}
    assert b.gat_src.max() < len(b.nodes)
    # leak edges are exactly the seed rows of the gat edges
    n_leak = len(b.leak_src)
    assert np.array_equal(b.leak_src, b.gat_src[:n_leak])
    assert (b.leak_dst < b.n_seeds).all()


def test_tile_batch_isolated_seed_contributes_no_edges():
    ei, ej = np.array([0, 1]), np.array([1, 2])
    g = build_graph(ei, ej, np.ones(2), np.array([10.0, 100.0]), 4)   # 3 isolated, 2 too
    b = tile_batch(g, np.array([2, 3]))
    assert b.n_seeds == 2 and b.n_ring1 == 0
    assert len(b.leak_src) == 0 and len(b.gat_src) == 0


# -- on the real bundle ---------------------------------------------------


def test_real_graph_beta_is_row_stochastic_after_pruning(dataset):
    from discell.model.prepare import from_dataset

    g = from_dataset(dataset)
    sums = np.asarray(g.in_edges.sum(axis=1)).ravel()
    connected = g.degrees > 0
    assert np.allclose(sums[connected], 1.0)
    assert np.allclose(sums[~connected], 0.0)
    assert np.allclose(g.y[connected].sum(axis=1), 1.0)


def test_real_tiles_cover_every_cell_once(dataset):
    from discell.model.prepare import from_dataset

    g = from_dataset(dataset)
    pos = dataset.positions_um
    tiles = spatial_tiles(pos, max_cells=256)
    assert sum(len(t) for t in tiles) == g.n_cells
    batch = tile_batch(g, tiles[0])
    assert batch.n_seeds == len(tiles[0])
