"""The objective assembly, checked against torch.distributions end to end."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from discell.model.elbo import Weights, discell_loss
from discell.model.networks import DisCell

torch.manual_seed(0)


def tiny_tile(n=10, genes=25, types=3, phi_dim=4, isolate_last=True):
    from discell.model.prepare import build_graph, tile_batch

    ei, ej = np.arange(n - 1), np.arange(1, n)
    dist = np.full(n - 1, 10.0)
    if isolate_last:
        dist[-1] = 100.0
    graph = build_graph(ei, ej, np.ones(n - 1), dist, n)
    batch = tile_batch(graph, np.arange(3, n))          # seeds 3..n-1
    model = DisCell(genes, types, phi_dim, d_z=4, d_w=2, hidden=16,
                    t_dim=4, gat_dim=6)
    rng = np.random.default_rng(1)
    tensors = dict(
        x=torch.tensor(rng.poisson(3.0, (len(batch.nodes), genes)), dtype=torch.float32),
        t=torch.tensor(rng.integers(0, types, len(batch.nodes))),
        phi=torch.randn(len(batch.nodes), phi_dim),
        isolated=torch.tensor(graph.isolated[batch.nodes]),
        gat_src=torch.tensor(batch.gat_src), gat_dst=torch.tensor(batch.gat_dst),
        leak_src=torch.tensor(batch.leak_src), leak_dst=torch.tensor(batch.leak_dst),
        leak_beta=torch.tensor(batch.leak_beta, dtype=torch.float32),
        n_seeds=batch.n_seeds, n_context=batch.n_context,
    )
    return model, tensors, batch


def independent_loss(fwd, x, omega):
    """The same J computed only through torch.distributions and lgamma."""
    from torch.distributions import Multinomial, Normal, kl_divergence

    n = x.shape[0]
    totals = x.sum(-1)

    def recon(log_p):
        rows = []
        for k in range(n):     # Multinomial wants one total per instance
            full = Multinomial(int(totals[k]), probs=log_p[k].exp()).log_prob(x[k])
            coefficient = (torch.lgamma(totals[k] + 1)
                           - torch.lgamma(x[k] + 1).sum())
            rows.append((full - coefficient) / totals[k])
        return torch.stack(rows).mean()

    kl_z = kl_divergence(
        Normal(fwd.mu_z[:n], (0.5 * fwd.logvar_z[:n]).exp()),
        Normal(torch.zeros_like(fwd.mu_z[:n]), torch.ones_like(fwd.mu_z[:n])),
    ).sum(-1).mean()
    kl_w = kl_divergence(
        Normal(fwd.mu_w[:n], (0.5 * fwd.logvar_w[:n]).exp()),
        Normal(fwd.prior_mean_w[:n], torch.ones_like(fwd.mu_w[:n])),
    ).sum(-1).mean()
    return -(recon(fwd.log_p) + omega * recon(fwd.log_p_breve)
             - (1 + omega) * kl_z - kl_w)


@pytest.mark.parametrize("kappa,omega", [(0.0, 0.0), (0.0, 1.0), (0.3, 1.0)])
def test_loss_matches_torch_distributions(kappa, omega):
    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=kappa)
    x_seeds = tensors["x"][:batch.n_seeds]
    ours = discell_loss(fwd, x_seeds, tensors["t"][:batch.n_seeds],
                        weights=Weights(omega=omega))
    theirs = independent_loss(fwd, x_seeds, omega)
    assert float(ours.loss) == pytest.approx(float(theirs), rel=1e-5)


def test_one_plus_omega_multiplies_the_z_kl():
    """Spec 6.2 calls omitting the second KL copy 'the natural mistake'."""
    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=0.2)
    x_seeds = tensors["x"][:batch.n_seeds]
    t_seeds = tensors["t"][:batch.n_seeds]
    base = discell_loss(fwd, x_seeds, t_seeds, weights=Weights(omega=0.0))
    two = discell_loss(fwd, x_seeds, t_seeds, weights=Weights(omega=2.0))
    expected = -(base.recon_a + 2.0 * base.recon_b
                 - 3.0 * base.kl_z - base.kl_w)
    assert float(two.loss) == pytest.approx(expected, rel=1e-6)


def test_gradients_reach_every_network():
    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=0.2)
    terms = discell_loss(fwd, tensors["x"][:batch.n_seeds],
                         tensors["t"][:batch.n_seeds])
    terms.loss.backward()
    for name in ("enc_z", "enc_w", "prior_w", "dec_a", "B", "gat", "embed_t"):
        module = getattr(model, name)
        got = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in module.parameters())
        assert got, f"no gradient reached {name}"


def test_term_b_does_not_train_the_w_posterior():
    """(b) draws w from the prior; enc_w must be invisible to it."""
    from discell.model.equations import multinomial_loglik

    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=0.2)
    multinomial_loglik(tensors["x"][:batch.n_seeds], fwd.log_p_breve).sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in model.enc_w.parameters())


def test_penalty_plumbs_through_and_reaches_enc_z():
    from discell.model.equations import TypeCovariances

    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=0.2)
    t_seeds = tensors["t"][:batch.n_seeds]
    cov = TypeCovariances(3, model.d_z, 2, ema=0.5, min_count=1)
    terms = discell_loss(
        fwd, tensors["x"][:batch.n_seeds], t_seeds,
        weights=Weights(alpha_a=1.0), v=torch.randn(batch.n_seeds, 2),
        covariances=cov, p_t=torch.full((3,), 1 / 3))
    assert terms.penalty != 0.0
    terms.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.enc_z.parameters())


def test_alpha_a_without_v_is_an_error():
    from discell.model.equations import TypeCovariances

    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=0.2)
    with pytest.raises(ValueError, match="alpha_a"):
        discell_loss(fwd, tensors["x"][:batch.n_seeds],
                     tensors["t"][:batch.n_seeds],
                     weights=Weights(alpha_a=1.0),
                     covariances=TypeCovariances(3, 4, 2))


def test_loss_is_finite_across_the_kappa_grid():
    model, tensors, batch = tiny_tile()
    for kappa in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
        fwd = model(**tensors, kappa=kappa)
        terms = discell_loss(fwd, tensors["x"][:batch.n_seeds],
                             tensors["t"][:batch.n_seeds])
        assert torch.isfinite(terms.loss), f"kappa={kappa}"
