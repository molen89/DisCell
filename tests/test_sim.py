"""The simulator produces the structure the model exists to disentangle."""

from __future__ import annotations

import numpy as np
import pytest

from discell import build_graph
from discell.sim import simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(n_cells=3000, seed=0)


def test_shapes_conservation_and_seed(sim):
    n, g = sim.x.shape
    assert n == 3000 and g == 60
    assert np.array_equal(sim.x.sum(axis=1), sim.totals)
    assert np.allclose(sim.rho_true.sum(axis=1), 1.0)
    assert np.allclose(sim.p_true.sum(axis=1), 1.0)
    again = simulate(n_cells=3000, seed=0)
    assert np.array_equal(sim.x, again.x) and np.array_equal(sim.t, again.t)


def test_types_cluster_in_space_and_w_reads_the_niche(sim):
    assert (np.diag(sim.graph.ybar_t) > 1.0 / sim.n_types).all()
    coef, *_ = np.linalg.lstsq(sim.graph.y, sim.w_true, rcond=None)
    assert 1 - (sim.w_true - sim.graph.y @ coef).var() / sim.w_true.var() > 0.7
    coef, *_ = np.linalg.lstsq(sim.graph.y, sim.phi, rcond=None)
    assert 1 - (sim.phi - sim.graph.y @ coef).var() / sim.phi.var() > 0.5


def test_leak_mixture_is_the_planted_convex_combination(sim):
    connected = sim.graph.degrees > 0
    expected = (1 - sim.kappa) * sim.rho_true + sim.kappa * (sim.graph.in_edges @ sim.rho_true)
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(sim.p_true[connected], expected[connected], atol=1e-12)


def test_leakage_makes_cells_resemble_their_neighbors():
    def neighbor_cosine(kappa):
        s = simulate(n_cells=2000, kappa=kappa, seed=1)
        comp = s.x / s.totals[:, None]
        neigh = s.graph.in_edges @ comp
        connected = s.graph.degrees > 0
        a, b = comp[connected], neigh[connected]
        return ((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                                  + 1e-12)).mean()

    assert neighbor_cosine(0.4) > neighbor_cosine(0.0) + 0.02


def test_to_anndata_meets_the_contract_and_rebuilds_the_same_graph(sim):
    adata = sim.to_anndata()
    assert adata.obsm["spatial"].shape == (3000, 2) and adata.obsm["X_image"].shape[1] == 8
    graph = build_graph(adata, "cell_type")
    assert np.array_equal(graph.edge_i, sim.graph.edge_i)
    assert np.array_equal(graph.edge_j, sim.graph.edge_j)
    assert np.array_equal(graph.beta, sim.graph.beta)
    assert np.array_equal(graph.y, sim.graph.y)
    assert adata.uns["discell"]["graph"]["types"] == [f"type{g}" for g in range(4)]
