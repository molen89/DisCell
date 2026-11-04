"""The objective, checked against torch.distributions end to end, and the adversary."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from discell.graph import graph_from_edges, tile_batch
from discell.model.equations import multinomial_loglik
from discell.model.networks import Adversary, DisCell, soft_cross_entropy
from discell.model.objective import Weights, adversary_terms, discell_loss

torch.manual_seed(0)


def tiny_tile(n=10, genes=25, types=3, phi_dim=4):
    ei, ej = np.arange(n - 1), np.arange(1, n)
    dist = np.full(n - 1, 10.0)
    dist[-1] = 100.0
    graph = graph_from_edges(ei, ej, np.ones(n - 1), dist, n)
    batch = tile_batch(graph, np.arange(3, n))
    model = DisCell(genes, types, phi_dim, median_counts=100.0, d_z=4, d_w=2,
                    hidden=16, gat_dim=6)
    rng = np.random.default_rng(1)
    tensors = dict(
        x=torch.tensor(rng.poisson(3.0, (len(batch.nodes), genes)), dtype=torch.float32),
        t=torch.tensor(rng.integers(0, types, len(batch.nodes))),
        phi=torch.randn(len(batch.nodes), phi_dim),
        isolated=torch.tensor(graph.isolated[batch.nodes]),
        gat_src=torch.tensor(batch.gat_src), gat_dst=torch.tensor(batch.gat_dst),
        spillover_src=torch.tensor(batch.spillover_src),
        spillover_dst=torch.tensor(batch.spillover_dst),
        spillover_beta=torch.tensor(batch.spillover_beta, dtype=torch.float32),
        n_seeds=batch.n_seeds, n_context=batch.n_context,
    )
    return model, tensors, batch


def independent_loss(fwd, x, omega):
    """The same J through torch.distributions and lgamma only."""
    from torch.distributions import Multinomial, Normal, kl_divergence

    n = x.shape[0]
    totals = x.sum(-1)

    def recon(log_p):
        rows = []
        for k in range(n):
            full = Multinomial(int(totals[k]), probs=log_p[k].exp()).log_prob(x[k])
            coefficient = torch.lgamma(totals[k] + 1) - torch.lgamma(x[k] + 1).sum()
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
    ours = discell_loss(fwd, x_seeds, Weights(omega=omega))
    theirs = independent_loss(fwd, x_seeds, omega).detach()
    assert float(ours.loss.detach()) == pytest.approx(float(theirs), rel=1e-5)


def test_one_plus_omega_multiplies_the_z_kl():
    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=0.2)
    x_seeds = tensors["x"][:batch.n_seeds]
    base = discell_loss(fwd, x_seeds, Weights(omega=0.0))
    two = discell_loss(fwd, x_seeds, Weights(omega=2.0))
    expected = -(base.recon_a + 2.0 * base.recon_b - 3.0 * base.kl_z - base.kl_w)
    assert float(two.loss.detach()) == pytest.approx(expected, rel=1e-6)


def test_gradients_reach_every_network():
    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=0.2)
    discell_loss(fwd, tensors["x"][:batch.n_seeds]).loss.backward()
    for name in ("enc_z", "enc_w", "prior_w", "dec_a", "B", "gat", "embed_t"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in getattr(model, name).parameters()), name


def test_term_b_does_not_train_the_w_posterior():
    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=0.2)
    multinomial_loglik(tensors["x"][:batch.n_seeds], fwd.log_p_breve).sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in model.enc_w.parameters())


def test_loss_is_finite_across_the_kappa_grid():
    model, tensors, batch = tiny_tile()
    for kappa in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
        terms = discell_loss(model(**tensors, kappa=kappa), tensors["x"][:batch.n_seeds])
        assert torch.isfinite(terms.loss), kappa


# -- the adversary ---------------------------------------------------------


def test_soft_cross_entropy_matches_manual():
    target = torch.tensor([[0.2, 0.8]])
    log_pred = torch.log(torch.tensor([[0.5, 0.5]]))
    assert float(soft_cross_entropy(target, log_pred)) == pytest.approx(-np.log(0.5))


def _toy(n=512, d_z=4, k=3, predictable=True, seed=0):
    rng = np.random.default_rng(seed)
    t = torch.tensor(rng.integers(0, k, n))
    z = torch.randn(n, d_z)
    if predictable:                       # y depends on z within type
        y = torch.softmax(z @ torch.randn(d_z, k) * 2.0, dim=-1)
    else:                                 # y depends on t alone
        y = torch.softmax(torch.randn(k, k) * 2.0, dim=-1)[t]
    ybar = torch.stack([y[t == g].mean(0) for g in range(k)])
    return z, t, y, ybar


def test_head_loss_touches_only_head_parameters():
    z, t, y, ybar = _toy()
    z = z.requires_grad_(True)
    heads = Adversary(4, 3, 3, hidden=16)
    adversary_terms(heads, z, t, y, y, ybar, ybar).head_loss.backward()
    assert z.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in heads.parameters())


def test_encoder_term_reaches_z():
    z, t, y, ybar = _toy()
    z = z.requires_grad_(True)
    adversary_terms(Adversary(4, 3, 3, hidden=16), z, t, y, y, ybar, ybar
                    ).encoder_term.backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0


def _train_heads(heads, z, t, y, ybar, steps=400):
    optimizer = torch.optim.Adam(heads.parameters(), lr=5e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        adversary_terms(heads, z, t, y, y, ybar, ybar).head_loss.backward()
        optimizer.step()
    return adversary_terms(heads, z, t, y, y, ybar, ybar)


def test_zero_excess_when_z_carries_nothing_beyond_type():
    z, t, y, ybar = _toy(predictable=False, n=2048)
    assert abs(_train_heads(Adversary(4, 3, 3, hidden=16), z, t, y, ybar).excess_y) < 0.03


def test_positive_excess_when_z_predicts_the_niche():
    z, t, y, ybar = _toy(predictable=True, n=2048)
    assert _train_heads(Adversary(4, 3, 3, hidden=16), z, t, y, ybar).excess_y > 0.15


def test_adversary_without_an_image_head():
    torch.manual_seed(0)
    full = Adversary(6, 4, 4)
    torch.manual_seed(0)
    free = Adversary(6, 4, 0)
    assert free.head_phi is None
    a, b = full.head_y.state_dict(), free.head_y.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)     # head_y is built first
    z, t = torch.randn(10, 6), torch.randint(0, 4, (10,))
    log_y, log_phi = free(z, t)
    assert log_phi is None and torch.equal(log_y, full(z, t)[0])
    y = torch.softmax(torch.randn(10, 4), -1)
    ybar = torch.softmax(torch.randn(4, 4), -1)
    terms = adversary_terms(free, z, t, y, None, ybar, None, comp_weight=3.0)
    ref = adversary_terms(full, z, t, y, torch.full((10, 4), 0.25), ybar,
                          torch.full((4, 4), 0.25), comp_weight=3.0)
    assert terms.excess_phi == 0.0
    assert terms.excess_y == pytest.approx(ref.excess_y)
    assert torch.isclose(terms.encoder_term, 3.0 * torch.tensor(ref.excess_y))
