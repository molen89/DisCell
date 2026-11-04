"""Each equation against an independent implementation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from discell.model.equations import (
    foreign_influx,
    gaussian_kl,
    multinomial_loglik,
    spillover_mix,
)

torch.manual_seed(0)


def test_multinomial_matches_scipy_up_to_the_constant():
    from scipy.special import gammaln
    from scipy.stats import multinomial

    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(30))
    x = rng.multinomial(200, p)
    ours = multinomial_loglik(torch.tensor(x, dtype=torch.float64),
                              torch.tensor(np.log(p)), scale_by_total=False)
    coefficient = gammaln(x.sum() + 1) - gammaln(x + 1).sum()
    theirs = multinomial.logpmf(x, n=200, p=p) - coefficient
    assert float(ours) == pytest.approx(theirs, rel=1e-10)


def test_multinomial_scaling_divides_by_the_total():
    x = torch.tensor([[2.0, 8.0]])
    log_p = torch.log(torch.tensor([[0.3, 0.7]]))
    raw = multinomial_loglik(x, log_p, scale_by_total=False)
    assert float(multinomial_loglik(x, log_p)) == pytest.approx(float(raw) / 10.0)


def test_kl_matches_torch_distributions():
    from torch.distributions import Normal, kl_divergence

    mu_q, lv_q = torch.randn(5, 4), torch.randn(5, 4)
    mu_p, lv_p = torch.randn(5, 4), torch.randn(5, 4)
    ours = gaussian_kl(mu_q, lv_q, mu_p, lv_p)
    theirs = kl_divergence(Normal(mu_q, (0.5 * lv_q).exp()),
                           Normal(mu_p, (0.5 * lv_p).exp())).sum(-1)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_kl_default_prior_is_standard_normal():
    mu, lv = torch.zeros(3, 2), torch.zeros(3, 2)
    assert torch.allclose(gaussian_kl(mu, lv), torch.zeros(3))


def test_foreign_influx_is_the_beta_weighted_neighbor_average():
    rho = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    src = torch.tensor([1, 2])          # both feed cell 0
    dst = torch.tensor([0, 0])
    beta = torch.tensor([0.25, 0.75])
    out = foreign_influx(rho, src, dst, beta, n_dst=2)
    assert torch.allclose(out[0], torch.tensor([0.375, 0.625]))
    assert torch.allclose(out[1], torch.zeros(2))   # no in-edges: zero row


def test_spillover_mix_stays_on_the_simplex_and_nests_the_null():
    rho = torch.softmax(torch.randn(4, 10), dim=-1)
    rho_bar = torch.softmax(torch.randn(4, 10), dim=-1)
    for kappa in (0.0, 0.2, 0.4):
        p = spillover_mix(rho, rho_bar, kappa).exp()
        assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(spillover_mix(rho, rho_bar, 0.0), rho.log(), atol=1e-6)


def test_spillover_mix_gives_isolated_cells_kappa_zero():
    """A zero foreign row degenerates to rho, not to a deficient mixture."""
    rho = torch.softmax(torch.randn(2, 6), dim=-1)
    rho_bar = torch.stack([torch.softmax(torch.randn(6), -1), torch.zeros(6)])
    log_p = spillover_mix(rho, rho_bar, 0.3)
    assert torch.allclose(log_p.exp().sum(-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(log_p[1], rho[1].log(), atol=1e-5)
    assert not torch.allclose(log_p[0], rho[0].log(), atol=1e-3)


def test_spillover_mix_rejects_kappa_outside_the_unit_interval():
    rho = torch.softmax(torch.randn(2, 3), dim=-1)
    with pytest.raises(ValueError):
        spillover_mix(rho, rho, 1.0)
