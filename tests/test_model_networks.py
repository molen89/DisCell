"""The architectural properties of the networks, asserted, plus the GAT layer."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from discell.graph import graph_from_edges, tile_batch
from discell.model.networks import DisCell, GATv2

torch.manual_seed(0)


def test_attention_sums_to_one_per_destination():
    gat = GATv2(src_dim=5, dst_dim=3, out_dim=8, heads=4)
    e_src = torch.tensor([0, 1, 2, 3, 4, 5])
    e_dst = torch.tensor([0, 0, 0, 1, 1, 2])
    _, alpha = gat(torch.randn(6, 5), torch.randn(3, 3), e_src, e_dst)
    for k in range(3):
        assert torch.allclose(alpha[e_dst == k].sum(dim=0), torch.ones(4), atol=1e-6)


def test_destination_shapes_attention_but_never_the_value():
    """With one in-edge alpha is 1 whatever the query, so the output is the source."""
    gat = GATv2(src_dim=4, dst_dim=4, out_dim=6, heads=2)
    src = torch.randn(1, 4)
    edge = (torch.tensor([0]), torch.tensor([0]))
    out_a, _ = gat(src, torch.randn(1, 4), *edge)
    out_b, _ = gat(src, torch.randn(1, 4) * 10, *edge)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_empty_destination_is_exactly_zero_not_nan():
    gat = GATv2(src_dim=4, dst_dim=4, out_dim=6, heads=2)
    out, _ = gat(torch.randn(2, 4), torch.randn(3, 4),
                 torch.tensor([0, 1]), torch.tensor([0, 0]))
    assert torch.isfinite(out).all()
    assert torch.equal(out[1], torch.zeros(6))
    assert torch.equal(out[2], torch.zeros(6))


def test_neighbor_order_does_not_matter():
    gat = GATv2(src_dim=4, dst_dim=4, out_dim=6, heads=2)
    src, dst = torch.randn(3, 4), torch.randn(1, 4)

    def fwd(order):
        return gat(src, dst, torch.tensor(order), torch.tensor([0, 0, 0]))[0]

    assert torch.allclose(fwd([0, 1, 2]), fwd([2, 0, 1]), atol=1e-6)


def make_tile(phi_dim: int = 5, genes: int = 30, types: int = 3):
    n = 8
    ei, ej = np.arange(n - 1), np.arange(1, n)
    dist = np.full(n - 1, 10.0)
    dist[-1] = 100.0                       # isolates cell 7
    graph = graph_from_edges(ei, ej, np.ones(n - 1), dist, n)
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
        spillover_src=torch.tensor(batch.spillover_src),
        spillover_dst=torch.tensor(batch.spillover_dst),
        spillover_beta=torch.tensor(batch.spillover_beta, dtype=torch.float32),
        n_seeds=batch.n_seeds, n_context=batch.n_context,
    )
    return model, tensors, batch


@pytest.fixture()
def tiny():
    return make_tile()


def test_forward_shapes_and_finiteness(tiny):
    model, tensors, batch = tiny
    out = model(**tensors, kappa=0.2)
    n_nodes, n_ctx, n_seeds = len(batch.nodes), batch.n_context, batch.n_seeds
    assert n_nodes > n_ctx                     # the tile has a ring 2
    assert out.z.shape == (n_ctx, 4)           # ring 2 is not encoded
    assert out.c.shape == (n_ctx, 6 + 5 + 1)
    assert out.w.shape == (n_ctx, 2)
    assert out.log_p.shape == out.log_p_breve.shape == (n_seeds, 30)
    for field in ("log_p", "log_p_breve", "log_rho", "c"):
        assert torch.isfinite(getattr(out, field)).all(), field


def test_ring_two_counts_are_never_read(tiny):
    model, tensors, batch = tiny
    out = model(**tensors, kappa=0.2, sample=False)
    poked = dict(tensors, x=tensors["x"].clone())
    poked["x"][batch.n_context:] += 50.0
    again = model(**poked, kappa=0.2, sample=False)
    assert torch.equal(out.c, again.c) and torch.equal(out.log_p, again.log_p)


def test_isolated_seed_gets_zero_context_a_flag_and_zero_influx(tiny):
    model, tensors, _ = tiny
    out = model(**tensors, kappa=0.2)
    row, gat_dim = 2, model.gat.out_dim             # seed 7 is isolated
    assert torch.equal(out.c[row, :gat_dim], torch.zeros(gat_dim))
    assert out.c[row, -1] == 1.0
    assert torch.equal(out.rho_bar[row], torch.zeros(30))
    assert torch.isfinite(out.log_p[row]).all()


def test_context_cannot_train_the_ego_encoder(tiny):
    model, tensors, _ = tiny
    model(**tensors, kappa=0.2).c.sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in model.enc_z.parameters())


def test_foreign_influx_is_fully_frozen(tiny):
    model, tensors, _ = tiny
    out = model(**tensors, kappa=0.2)
    assert not out.rho_bar.requires_grad and out.rho_bar.grad_fn is None


def test_reconstruction_still_trains_the_cell_s_own_path(tiny):
    model, tensors, _ = tiny
    model(**tensors, kappa=0.5).log_p.sum().backward()
    for module in (model.dec_a, model.enc_z):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in module.parameters())


def test_term_b_trains_the_prior_network(tiny):
    model, tensors, _ = tiny
    model(**tensors, kappa=0.2).log_p_breve.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.prior_w.parameters())


def test_decoder_has_no_route_from_t():
    model = DisCell(20, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16)
    z = torch.randn(5, 4, requires_grad=True)
    w = torch.randn(5, 2, requires_grad=True)
    model.log_rho(z, w).sum().backward()
    assert model.embed_t.weight.grad is None


def test_enc_z_signature_admits_no_context():
    assert set(inspect.signature(DisCell.posterior_z).parameters) == {"self", "x", "t"}


def test_state_dict_names_match_the_research_checkpoints():
    model = DisCell(30, 3, 5, median_counts=100.0)
    prefixes = {k.split(".")[0] for k in model.state_dict()}
    assert prefixes == {"embed_t", "enc_z", "gat", "prior_w", "enc_w", "dec_a", "B"}
    assert {k for k in model.state_dict() if k.startswith("gat.")} == {
        "gat.attn", "gat.w_src.weight", "gat.w_dst.weight"}


def test_image_free_model_has_a_context_of_gat_plus_flag():
    model, tensors, batch = make_tile(phi_dim=0)
    assert model.image_free
    out = model(**tensors, kappa=0.1)
    assert out.c.shape == (batch.n_context, 6 + 1)
    assert model.prior_w[0].in_features == 6 + 1 + 3

