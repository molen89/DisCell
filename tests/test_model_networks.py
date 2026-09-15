"""The three load-bearing architectural properties, asserted, plus the GAT."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from discell.model.networks import DisCell, GATv2

torch.manual_seed(0)


# -- the GAT layer --------------------------------------------------------


def test_attention_sums_to_one_per_destination():
    gat = GATv2(src_dim=5, dst_dim=3, out_dim=8, heads=4)
    src = torch.randn(6, 5)
    dst = torch.randn(3, 3)
    e_src = torch.tensor([0, 1, 2, 3, 4, 5])
    e_dst = torch.tensor([0, 0, 0, 1, 1, 2])
    _, alpha = gat(src, dst, e_src, e_dst)
    for k in range(3):
        sums = alpha[e_dst == k].sum(dim=0)
        assert torch.allclose(sums, torch.ones(4), atol=1e-6)


def test_destination_shapes_attention_but_never_the_value():
    """With one in-edge alpha is 1 regardless of the query, so the output is
    purely the projected source -- the property the mirror argument rests on."""
    gat = GATv2(src_dim=4, dst_dim=4, out_dim=6, heads=2)
    src = torch.randn(1, 4)
    edge = (torch.tensor([0]), torch.tensor([0]))
    out_a, _ = gat(src, torch.randn(1, 4), *edge)
    out_b, _ = gat(src, torch.randn(1, 4) * 10, *edge)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_empty_destination_is_exactly_zero_not_nan():
    gat = GATv2(src_dim=4, dst_dim=4, out_dim=6, heads=2)
    out, _ = gat(torch.randn(2, 4), torch.randn(3, 4),
                 torch.tensor([0, 1]), torch.tensor([0, 0]))   # dst 1, 2 empty
    assert torch.isfinite(out).all()
    assert torch.allclose(out[1], torch.zeros(6))
    assert torch.allclose(out[2], torch.zeros(6))


def test_neighbour_order_does_not_matter():
    gat = GATv2(src_dim=4, dst_dim=4, out_dim=6, heads=2)
    src, dst = torch.randn(3, 4), torch.randn(1, 4)
    fwd = lambda order: gat(src, dst, torch.tensor(order), torch.tensor([0, 0, 0]))[0]
    assert torch.allclose(fwd([0, 1, 2]), fwd([2, 0, 1]), atol=1e-6)


# -- a tiny tile through the full forward ---------------------------------


@pytest.fixture()
def tiny():
    from discell.model.prepare import build_graph, tile_batch

    n, genes, types, phi_dim = 8, 30, 3, 5
    ei, ej = np.arange(n - 1), np.arange(1, n)
    dist = np.full(n - 1, 10.0)
    dist[-1] = 100.0                       # isolate cell 7
    graph = build_graph(ei, ej, np.ones(n - 1), dist, n)
    batch = tile_batch(graph, np.array([3, 4, 7]))
    model = DisCell(genes, types, phi_dim, median_counts=100.0, d_z=4, d_w=2,
                    hidden=16, gat_dim=6)
    rng = np.random.default_rng(0)
    tensors = dict(
        x=torch.tensor(rng.poisson(2.0, (len(batch.nodes), genes)), dtype=torch.float32),
        t=torch.tensor(rng.integers(0, types, len(batch.nodes))),
        phi=torch.randn(len(batch.nodes), phi_dim),
        isolated=torch.tensor(graph.isolated[batch.nodes]),
        gat_src=torch.tensor(batch.gat_src), gat_dst=torch.tensor(batch.gat_dst),
        leak_src=torch.tensor(batch.leak_src), leak_dst=torch.tensor(batch.leak_dst),
        leak_beta=torch.tensor(batch.leak_beta, dtype=torch.float32),
        n_seeds=batch.n_seeds, n_context=batch.n_context,
    )
    return model, tensors, batch


def test_forward_shapes_and_finiteness(tiny):
    model, tensors, batch = tiny
    out = model(**tensors, kappa=0.2)
    n_nodes, n_ctx, n_seeds = len(batch.nodes), batch.n_context, batch.n_seeds
    assert out.z.shape == (n_nodes, 4)
    assert out.c.shape[0] == n_ctx
    assert out.w.shape == (n_ctx, 2)
    assert out.log_p.shape == (n_seeds, 30)
    assert out.log_p_breve.shape == (n_seeds, 30)
    for field in ("log_p", "log_p_breve", "log_rho", "c"):
        assert torch.isfinite(getattr(out, field)).all(), field


def test_isolated_seed_gets_zero_context_a_flag_and_zero_influx(tiny):
    model, tensors, batch = tiny
    out = model(**tensors, kappa=0.2)
    row = 2                                          # seed 7, isolated
    gat_dim = model.gat.out_dim
    assert torch.allclose(out.c[row, :gat_dim], torch.zeros(gat_dim))
    assert out.c[row, -1] == 1.0                     # the flag
    assert torch.allclose(out.rho_bar[row], torch.zeros(30))
    # its mixture degenerates to (1-kappa) rho, still a finite log
    assert torch.isfinite(out.log_p[row]).all()


# -- gradient isolation: the properties the spec is built on --------------


def test_context_cannot_train_the_ego_encoder(tiny):
    """sg mu_z: the GAT source path must leak no gradient into enc_z."""
    model, tensors, _ = tiny
    out = model(**tensors, kappa=0.2)
    out.c.sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in model.enc_z.parameters())


def test_foreign_influx_is_fully_frozen(tiny):
    """rho_bar is data, not a differentiable path (spec 4.5).

    Not merely zero gradient -- it carries no autograd graph at all, so nothing
    downstream can re-attach it by accident.
    """
    model, tensors, _ = tiny
    out = model(**tensors, kappa=0.2)
    assert not out.rho_bar.requires_grad
    assert out.rho_bar.grad_fn is None


def test_reconstruction_still_trains_the_cell_s_own_path(tiny):
    model, tensors, _ = tiny
    out = model(**tensors, kappa=0.5)
    out.log_p.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.dec_a.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.enc_z.parameters())


def test_term_b_trains_the_prior_network(tiny):
    """(b) is what rewards m_psi for predicting the response (spec 6.3)."""
    model, tensors, _ = tiny
    out = model(**tensors, kappa=0.2)
    out.log_p_breve.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.prior_w.parameters())


def test_decoder_has_no_route_from_t():
    """The only path from identity to expression is z (spec 7.1)."""
    model = DisCell(20, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16)
    z = torch.randn(5, 4, requires_grad=True)
    w = torch.randn(5, 2, requires_grad=True)
    model.log_rho(z, w).sum().backward()
    assert model.embed_t.weight.grad is None


def test_enc_z_signature_admits_no_context():
    import inspect

    params = inspect.signature(DisCell.posterior_z).parameters
    assert set(params) == {"self", "x", "t"}


def _with_leak_subtraction(model):
    model.subtract_leak = True
    return model


def test_leak_subtraction_rewrites_seeds_only_and_is_a_no_op_at_kappa_zero(tiny):
    """Spec 7.13: pass 2 replaces the seeds' rows from x~ = x - kappa*l*rho_bar;
    ring rows keep pass 1; at kappa = 0 (x~ == x) both paths agree exactly,
    and the isolated seed (rho_bar = 0) is unchanged at any kappa."""
    model, tensors, batch = tiny
    torch.manual_seed(0)
    raw = model(**tensors, kappa=0.2, sample=False)
    _with_leak_subtraction(model)
    two = model(**tensors, kappa=0.2, sample=False)
    n_seeds = batch.n_seeds
    assert two.mu_z.shape == raw.mu_z.shape and two.log_rho.shape == raw.log_rho.shape
    assert torch.allclose(two.mu_z[n_seeds:], raw.mu_z[n_seeds:])
    assert torch.allclose(two.mu_w[n_seeds:], raw.mu_w[n_seeds:])
    assert torch.equal(two.rho_bar, raw.rho_bar)             # pass-1 rho_bar is data
    connected = ~tensors["isolated"][:n_seeds]
    assert not torch.allclose(two.mu_z[:n_seeds][connected], raw.mu_z[:n_seeds][connected])
    iso = tensors["isolated"][:n_seeds]
    assert torch.allclose(two.mu_z[:n_seeds][iso], raw.mu_z[:n_seeds][iso])
    assert torch.allclose(model(**tensors, kappa=0.0, sample=False).mu_z,
                          _no_subtraction(model, tensors).mu_z)
    for field in ("log_p", "log_p_breve", "log_rho", "mu_z", "mu_w"):
        assert torch.isfinite(getattr(two, field)).all(), field


def _no_subtraction(model, tensors):
    model.subtract_leak = False
    out = model(**tensors, kappa=0.0, sample=False)
    model.subtract_leak = True
    return out


def test_leak_subtraction_trains_enc_z_through_pass_two_only(tiny):
    """Condition 4: the likelihood gradient reaches enc_z through the x~ view;
    nothing flows back into pass 1 through rho_bar (still data)."""
    model, tensors, batch = tiny
    _with_leak_subtraction(model)
    out = model(**tensors, kappa=0.2)
    out.log_p.sum().backward()
    assert out.rho_bar.grad_fn is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.enc_z.parameters())
