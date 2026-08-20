"""The simulator must produce the structure the model exists to disentangle."""

from __future__ import annotations

import numpy as np
import pytest

from discell.model.synthetic import simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(n_cells=3000, seed=0)


def test_shapes_and_conservation(sim):
    n, g = sim.x.shape
    assert n == 3000 and g == 60
    assert np.array_equal(sim.x.sum(axis=1), sim.totals)
    assert np.allclose(sim.rho_true.sum(axis=1), 1.0)
    assert np.allclose(sim.p_true.sum(axis=1), 1.0)


def test_types_cluster_in_space(sim):
    """Without spatial type structure, y is uninformative and w has no target."""
    ybar = sim.graph.ybar_t
    assert (np.diag(ybar) > 1.0 / sim.n_types).all()   # own type over-represented


def test_response_is_niche_predictable(sim):
    """w_true = M y + noise by construction: a linear fit must see that."""
    y, w = sim.graph.y, sim.w_true
    coef, *_ = np.linalg.lstsq(y, w, rcond=None)
    residual = w - y @ coef
    r2 = 1 - residual.var() / w.var()
    assert r2 > 0.7


def test_leakage_makes_cells_resemble_their_neighbours():
    """The central confound must actually appear in the counts."""
    def neighbour_cosine(kappa):
        s = simulate(n_cells=2000, kappa=kappa, seed=1)
        comp = s.x / s.totals[:, None]
        neigh = s.graph.in_edges @ comp                 # beta-weighted mean
        connected = s.graph.degrees > 0
        a, b = comp[connected], neigh[connected]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1)
                                * np.linalg.norm(b, axis=1) + 1e-12)
        return cos.mean()

    assert neighbour_cosine(0.4) > neighbour_cosine(0.0) + 0.02


def test_leak_mixture_is_the_planted_convex_combination(sim):
    connected = sim.graph.degrees > 0
    rho_bar = sim.graph.in_edges @ sim.rho_true
    expected = (1 - sim.kappa) * sim.rho_true + sim.kappa * rho_bar
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(sim.p_true[connected], expected[connected], atol=1e-12)
    # isolated cells: p degenerates to rho exactly
    if (~connected).any():
        assert np.allclose(sim.p_true[~connected], sim.rho_true[~connected])


def test_phi_reads_the_niche_not_the_cell(sim):
    """phi = P y + noise: predictable from composition, not from own type
    beyond what composition already carries."""
    y = sim.graph.y
    coef, *_ = np.linalg.lstsq(y, sim.phi, rcond=None)
    r2 = 1 - (sim.phi - y @ coef).var() / sim.phi.var()
    assert r2 > 0.5
