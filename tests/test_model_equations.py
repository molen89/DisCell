"""Each equation against an independent implementation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from discell.model.equations import (
    TypeCovariances,
    foreign_influx,
    gaussian_kl,
    leakage_mix,
    multinomial_loglik,
    shrink_to_diagonal,
)

torch.manual_seed(0)


# -- multinomial ----------------------------------------------------------


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


# -- gaussian KL ----------------------------------------------------------


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


# -- leakage --------------------------------------------------------------


def test_foreign_influx_is_the_beta_weighted_neighbour_average():
    rho = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    src = torch.tensor([1, 2])          # both feed cell 0
    dst = torch.tensor([0, 0])
    beta = torch.tensor([0.25, 0.75])
    out = foreign_influx(rho, src, dst, beta, n_dst=2)
    assert torch.allclose(out[0], torch.tensor([0.375, 0.625]))
    assert torch.allclose(out[1], torch.zeros(2))   # no in-edges -> zero row


def test_leakage_mix_stays_on_the_simplex_and_nests_the_null():
    rho = torch.softmax(torch.randn(4, 10), dim=-1)
    rho_bar = torch.softmax(torch.randn(4, 10), dim=-1)
    for kappa in (0.0, 0.2, 0.4):
        p = leakage_mix(rho, rho_bar, kappa).exp()
        assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(leakage_mix(rho, rho_bar, 0.0), rho.log(), atol=1e-6)


# -- the invariance penalty ----------------------------------------------


def _analytic_gaussian_mi(cov: np.ndarray, dz: int) -> float:
    sign_z, ld_z = np.linalg.slogdet(cov[:dz, :dz])
    sign_v, ld_v = np.linalg.slogdet(cov[dz:, dz:])
    sign_j, ld_j = np.linalg.slogdet(cov)
    return 0.5 * (ld_z + ld_v - ld_j)


@pytest.mark.parametrize("coupling,expect_positive", [(0.0, False), (0.8, True)])
def test_penalty_matches_analytic_gaussian_mi(coupling, expect_positive):
    """One full-population update (ema=1, shrink=0) must reproduce the truth."""
    rng = np.random.default_rng(1)
    dz, dv, n = 4, 3, 200_000
    z = rng.standard_normal((n, dz))
    mix = rng.standard_normal((dz, dv))
    v = coupling * z @ mix + rng.standard_normal((n, dv))
    truth = _analytic_gaussian_mi(np.cov(np.hstack([z, v]).T), dz)

    tracker = TypeCovariances(1, dz, dv, ema=1.0, shrink=0.0, min_count=1)
    penalty, info = tracker.penalty(torch.tensor(z), torch.tensor(v),
                                    torch.zeros(n, dtype=torch.long),
                                    p_t=torch.ones(1))
    assert float(penalty) == pytest.approx(truth, abs=2e-3)
    assert (truth > 0.05) == expect_positive
    assert info["excluded_fraction"] == 0.0


def test_penalty_is_conditional_not_marginal():
    """v predictable from t alone must not be penalised (spec section 7.8)."""
    rng = np.random.default_rng(2)
    n = 100_000
    t = rng.integers(0, 2, n)
    z = rng.standard_normal((n, 2)) + 3.0 * t[:, None]   # z carries the type
    v = rng.standard_normal((n, 2)) + 3.0 * t[:, None]   # so does v
    tracker = TypeCovariances(2, 2, 2, ema=1.0, shrink=0.0, min_count=1)
    penalty, _ = tracker.penalty(torch.tensor(z, dtype=torch.float64),
                                 torch.tensor(v, dtype=torch.float64),
                                 torch.tensor(t), p_t=torch.full((2,), 0.5))
    # marginally z and v are strongly dependent; conditionally they are not
    marginal = _analytic_gaussian_mi(np.cov(np.hstack([z, v]).T), 2)
    assert marginal > 0.3
    assert abs(float(penalty)) < 5e-3


def test_shrinkage_keeps_variances_and_damps_correlations():
    cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    out = shrink_to_diagonal(cov, 0.4)
    assert torch.allclose(torch.diagonal(out), torch.diagonal(cov))
    assert float(out[0, 1]) == pytest.approx(0.6)


def test_penalty_survives_a_singular_batch_via_the_ema():
    """Fewer cells than dimensions in one batch must not blow up the estimate."""
    rng = np.random.default_rng(3)
    dz, dv = 6, 6
    tracker = TypeCovariances(1, dz, dv, ema=0.05, shrink=0.05, min_count=50)
    t = torch.zeros(8, dtype=torch.long)
    value = None
    for _ in range(400):                     # 8 cells/step, 12+12 dims
        z = torch.tensor(rng.standard_normal((8, dz)))
        v = torch.tensor(rng.standard_normal((8, dv)))
        value, info = tracker.penalty(z, v, t, p_t=torch.ones(1))
    assert torch.isfinite(value)
    assert info["excluded_fraction"] == 0.0   # ema has accumulated support
    assert abs(float(value)) < 0.5            # independent data -> near zero


def test_penalty_gradient_reaches_the_current_batch():
    z = torch.randn(64, 3, requires_grad=True)
    v = torch.randn(64, 2)
    tracker = TypeCovariances(1, 3, 2, ema=0.1, shrink=0.05, min_count=1)
    penalty, _ = tracker.penalty(z, v, torch.zeros(64, dtype=torch.long),
                                 p_t=torch.ones(1))
    penalty.backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0


def test_support_floor_excludes_and_reports():
    tracker = TypeCovariances(1, 2, 2, ema=0.01, min_count=1e6)
    _, info = tracker.penalty(torch.randn(32, 2), torch.randn(32, 2),
                              torch.zeros(32, dtype=torch.long),
                              p_t=torch.ones(1))
    assert info["excluded_fraction"] == 1.0


def test_leakage_mix_gives_isolated_cells_kappa_zero():
    """A zero foreign row must degenerate to rho, not to a deficient mixture."""
    rho = torch.softmax(torch.randn(2, 6), dim=-1)
    rho_bar = torch.stack([torch.softmax(torch.randn(6), -1), torch.zeros(6)])
    log_p = leakage_mix(rho, rho_bar, 0.3)
    assert torch.allclose(log_p.exp().sum(-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(log_p[1], rho[1].log(), atol=1e-5)   # isolated row
    assert not torch.allclose(log_p[0], rho[0].log(), atol=1e-3)  # mixed row
