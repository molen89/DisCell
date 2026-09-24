"""The 6b.8 posterior must be right, or the amortisation gap measures nothing.

The quadrature in :func:`posterior_moments` is checked against a long
importance sample from the prior on a tiny world (5 genes, shallow counts,
where the prior proposal has ample overlap with the posterior).
"""

from __future__ import annotations

import numpy as np
import pytest

from discell.experiments import planted_posterior as pp


class _TinySim:
    """The three fields :func:`posterior_moments` reads."""

    def __init__(self, x, t, kappa):
        self.x, self.t, self.kappa = x, t, kappa


def _importance_moments(x, m, rho_bar, kappa, A, sigma, rng, n=4_000_000):
    """E[z|x] and sd by importance sampling with the prior as proposal."""
    z = m + sigma * rng.standard_normal((n, len(m)))
    logits = z @ A
    rho = np.exp(logits - logits.max(axis=1, keepdims=True))
    rho /= rho.sum(axis=1, keepdims=True)
    p = (1.0 - kappa) * rho + kappa * rho_bar
    p /= p.sum(axis=1, keepdims=True)
    ll = (x * np.log(np.clip(p, 1e-30, None))).sum(axis=1)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    mean = w @ z
    var = w @ (z - mean) ** 2
    return mean, np.sqrt(var), 1.0 / np.sum(w ** 2)


@pytest.mark.parametrize("depth,kappa", [(20, 0.2), (60, 0.35)])
def test_posterior_matches_importance_sampling(depth, kappa):
    rng = np.random.default_rng(7)
    n_genes, d_z = 5, pp.D_Z
    A = rng.normal(0.0, 1.0, size=(d_z, n_genes))
    m_t = rng.normal(0.0, 1.0, size=(1, d_z))
    sigma = 0.5

    n_cells = 6
    z = m_t[0] + sigma * rng.standard_normal((n_cells, d_z))
    logits = z @ A
    rho = np.exp(logits - logits.max(axis=1, keepdims=True))
    rho /= rho.sum(axis=1, keepdims=True)
    rho_bar = rng.dirichlet(np.ones(n_genes), size=n_cells)
    p = (1.0 - kappa) * rho + kappa * rho_bar
    p /= p.sum(axis=1, keepdims=True)
    x = np.stack([rng.multinomial(depth, p[i]) for i in range(n_cells)])

    sim = _TinySim(x.astype(np.float32), np.zeros(n_cells, dtype=int), kappa)
    truth = {"A": A, "m_t": m_t, "rho_bar": rho_bar, "sigma_z": sigma}
    mean, sd, diag = pp.posterior_moments(sim, truth, device="cpu", chunk=3)

    assert diag["max_boundary_mass"] < 1e-6
    for i in range(n_cells):
        ref_mean, ref_sd, ess = _importance_moments(
            x[i], m_t[0], rho_bar[i], kappa, A, sigma, rng)
        assert ess > 20_000, f"importance sample degenerate (ESS {ess:.0f})"
        # 3 s.e. of the importance estimate is ~ sd/sqrt(ESS); 0.02 sd is
        # far wider and still a sharp test of the quadrature.
        assert np.allclose(mean[i], ref_mean, atol=0.02 * ref_sd.max()), (
            i, mean[i], ref_mean)
        assert np.allclose(sd[i], ref_sd, rtol=0.02), (i, sd[i], ref_sd)


def test_posterior_collapses_to_prior_without_counts():
    """No counts -> the posterior is the prior, exactly."""
    rng = np.random.default_rng(3)
    n_genes = 5
    truth = {"A": rng.normal(size=(pp.D_Z, n_genes)),
             "m_t": np.array([[0.4, -1.1]]),
             "rho_bar": np.full((2, n_genes), 1.0 / n_genes),
             "sigma_z": 0.5}
    sim = _TinySim(np.zeros((2, n_genes), dtype=np.float32),
                   np.zeros(2, dtype=int), 0.2)
    mean, sd, _ = pp.posterior_moments(sim, truth, device="cpu", chunk=2)
    assert np.allclose(mean, truth["m_t"], atol=1e-8)
    assert np.allclose(sd, 0.5, rtol=1e-8)


def test_world_is_the_leak_mixture_it_claims():
    """The planted world's counts really come from (1-k)rho + k rho_bar."""
    sim, truth = pp.build_world(seed=0, depth=200)
    rho_bar = np.asarray(sim.graph.in_edges @ sim.rho_true)
    assert np.allclose(rho_bar, truth["rho_bar"])
    p = (1.0 - sim.kappa) * sim.rho_true + sim.kappa * rho_bar
    p /= p.sum(axis=1, keepdims=True)
    assert np.allclose(p, sim.p_true)
    logits = sim.z_true @ truth["A"]
    rho = np.exp(logits - logits.max(axis=1, keepdims=True))
    assert np.allclose(rho / rho.sum(axis=1, keepdims=True), sim.rho_true)
