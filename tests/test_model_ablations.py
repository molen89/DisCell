"""The four objective ablations of 6b.5 (2026-09-21).

Two properties per flag: **off** reproduces the pinned loss bit for bit, and
**on** changes exactly the one term it names and nothing else. The "exactly"
half is planted -- the difference is predicted in closed form from the tile's
own components, not merely asserted to be nonzero.
"""

from __future__ import annotations

import dataclasses as dc

import numpy as np
import torch

from discell.model.elbo import Weights, adversary_terms, discell_loss
from discell.model.networks import Adversary, ClassMeanPrior, DisCell
from tests.test_model_elbo import tiny_tile


def _forward(seed=0, **model_kw):
    torch.manual_seed(seed)
    model, tensors, batch = tiny_tile()
    if model_kw:
        torch.manual_seed(seed)
        model = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2,
                        hidden=16, gat_dim=6, **model_kw)
    torch.manual_seed(seed + 100)
    fwd = model(**tensors, kappa=0.1)
    return model, tensors, batch, fwd


# -- (i) the second copy of the z-KL ---------------------------------------

def test_second_kl_default_is_the_pinned_loss_bit_for_bit():
    _, tensors, batch, fwd = _forward()
    x, t = tensors["x"][: batch.n_seeds], tensors["t"][: batch.n_seeds]
    w = Weights(omega=1.0, alpha_z=0.007, alpha_w=0.1)
    assert w.second_kl is True
    a = discell_loss(fwd, x, t, weights=w).loss
    b = discell_loss(fwd, x, t, weights=dc.replace(w, second_kl=True)).loss
    assert a.item() == b.item()


def test_second_kl_off_removes_exactly_one_copy_of_alpha_z_kl_z():
    _, tensors, batch, fwd = _forward()
    x, t = tensors["x"][: batch.n_seeds], tensors["t"][: batch.n_seeds]
    for omega in (0.5, 1.0, 2.0):
        w = Weights(omega=omega, alpha_z=0.007, alpha_w=0.1)
        on = discell_loss(fwd, x, t, weights=w)
        off = discell_loss(fwd, x, t, weights=dc.replace(w, second_kl=False))
        # loss = -J, so dropping a -omega*alpha_z*KL term lowers the loss by it
        planted = omega * w.alpha_z * on.kl_z
        # float32 loss ~5.3 against a difference ~1e-3: 1e-3 relative is the
        # tightest the dtype allows, and still pins the term to three digits
        assert np.isclose(on.loss.item() - off.loss.item(), planted, rtol=1e-3)
        # every logged component is untouched
        for key in ("recon_a", "recon_b", "kl_z", "kl_w", "w_penalty"):
            assert on.scalars()[key] == off.scalars()[key]


def test_dropping_the_second_copy_is_exactly_a_halved_alpha_z():
    """At omega = 1 the arm is an identity, not a new objective: 1*alpha_z ==
    (1+1)*(alpha_z/2). Anything the arm measures is an alpha_z move."""
    _, tensors, batch, fwd = _forward()
    x, t = tensors["x"][: batch.n_seeds], tensors["t"][: batch.n_seeds]
    dropped = discell_loss(fwd, x, t, weights=Weights(
        omega=1.0, alpha_z=0.007, alpha_w=0.1, second_kl=False)).loss
    halved = discell_loss(fwd, x, t, weights=Weights(
        omega=1.0, alpha_z=0.0035, alpha_w=0.1)).loss
    assert dropped.item() == halved.item()


def test_omega_zero_makes_the_two_settings_coincide():
    """Arm (ii): with omega = 0 the (1+omega) factor already reads 1."""
    _, tensors, batch, fwd = _forward()
    x, t = tensors["x"][: batch.n_seeds], tensors["t"][: batch.n_seeds]
    w = Weights(omega=0.0, alpha_z=0.007, alpha_w=0.1)
    on = discell_loss(fwd, x, t, weights=w).loss
    off = discell_loss(fwd, x, t, weights=dc.replace(w, second_kl=False)).loss
    assert on.item() == off.item()


# -- (iii) the class-mean prior ---------------------------------------------

def test_class_mean_prior_off_is_the_msi_mlp_bit_for_bit():
    torch.manual_seed(7)
    base = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                   gat_dim=6)
    torch.manual_seed(7)
    explicit = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                       gat_dim=6, class_mean_prior=False)
    assert not isinstance(base.prior_w, ClassMeanPrior)
    for a, b in zip(base.state_dict().values(), explicit.state_dict().values()):
        assert torch.equal(a, b)

    _, tensors, batch, fwd_a = _forward(seed=3)
    _, _, _, fwd_b = _forward(seed=3, class_mean_prior=False)
    assert torch.equal(fwd_a.prior_mean_w, fwd_b.prior_mean_w)
    assert torch.equal(fwd_a.log_p, fwd_b.log_p)


def test_class_mean_prior_on_reads_the_type_and_nothing_else():
    _, tensors, batch, fwd = _forward(seed=3, class_mean_prior=True)
    t = tensors["t"][: batch.n_context]
    prior = fwd.prior_mean_w
    # one vector per type, exactly -- the context c is ignored
    for g in torch.unique(t):
        rows = prior[t == g]
        assert torch.equal(rows, rows[0].expand_as(rows))
    # and it is the embedding table itself
    model, _, _, fwd2 = _forward(seed=3, class_mean_prior=True)
    assert torch.equal(fwd2.prior_mean_w, model.prior_w.mu_t(t))


def test_class_mean_prior_leaves_the_w_posterior_conditioning_intact():
    """q(w | c, t, z, x) is unchanged: enc_w still reads the context block."""
    model, tensors, batch, fwd = _forward(seed=3, class_mean_prior=True)
    c_dim = model.enc_w[0].in_features - (model.n_types + model.d_z + 25 + 1)
    assert c_dim == fwd.c.shape[1]
    loss = discell_loss(fwd, tensors["x"][: batch.n_seeds],
                        tensors["t"][: batch.n_seeds],
                        weights=Weights(alpha_w=0.1)).loss
    loss.backward()
    assert model.prior_w.mu_t.weight.grad.abs().sum() > 0


def test_class_mean_prior_is_a_drop_in_for_the_prior_w_call_sites():
    """transport/degeneracy/neighbour_dose call model.prior_w([c, onehot(t)])."""
    import torch.nn.functional as F

    head = ClassMeanPrior(3, 2)
    c = torch.randn(5, 7)
    t = torch.tensor([0, 1, 2, 1, 0])
    out = head(torch.cat([c, F.one_hot(t, 3).float()], dim=-1))
    assert torch.equal(out, head.mu_t(t))
    other = head(torch.cat([torch.randn(5, 7), F.one_hot(t, 3).float()], -1))
    assert torch.equal(out, other)


# -- (iv) the adversary on the decoded composition --------------------------

def test_adversary_terms_are_unchanged_on_mu_z():
    model, tensors, batch, fwd = _forward(seed=5)
    n, k = batch.n_seeds, 3
    torch.manual_seed(11)
    heads = Adversary(model.d_z, k, 4)
    t = tensors["t"][:n]
    y = torch.softmax(torch.randn(n, k), -1)
    e_phi = torch.softmax(torch.randn(n, 4), -1)
    ybar = torch.softmax(torch.randn(k, k), -1)
    phibar = torch.softmax(torch.randn(k, 4), -1)
    a = adversary_terms(heads, fwd.mu_z[:n], t, y, e_phi, ybar, phibar)
    b = adversary_terms(heads, fwd.mu_z[:n], t, y, e_phi, ybar, phibar)
    assert a.encoder_term.item() == b.encoder_term.item()


def test_adversary_on_xhat_reads_rho_and_reaches_the_decoder():
    """The head input is log rho = log_softmax(a(z) + Bw), so the encoder term
    now pushes on B and dec_a as well as on the z encoder."""
    model, tensors, batch, fwd = _forward(seed=5)
    n, k, genes = batch.n_seeds, 3, 25
    torch.manual_seed(11)
    heads = Adversary(genes, k, 4)       # G columns, not d_z
    assert heads.head_y[0].in_features == genes + k
    t = tensors["t"][:n]
    y = torch.softmax(torch.randn(n, k), -1)
    e_phi = torch.softmax(torch.randn(n, 4), -1)
    ybar = torch.softmax(torch.randn(k, k), -1)
    phibar = torch.softmax(torch.randn(k, 4), -1)
    adv = adversary_terms(heads, fwd.log_rho[:n], t, y, e_phi, ybar, phibar)
    adv.encoder_term.backward()
    assert model.B.weight.grad.abs().sum() > 0
    assert model.dec_a[0].weight.grad.abs().sum() > 0
    # the heads do pick up gradient here, exactly as they do on mu_z: the
    # trainer never steps them from the model loss and zeroes them before the
    # head step, which is unchanged by this arm.
    assert any(p.grad is not None for p in heads.parameters())


def test_trainer_wiring_matches_each_flag():
    """The three TrainConfig knobs default to the pinned objective and each
    reaches exactly its own consumer."""
    from discell.model.train import TrainConfig

    base = TrainConfig(dataset="x")
    assert (base.second_kl, base.class_mean_prior, base.adv_input) == (
        True, False, "mu_z")
    assert base.weights().second_kl is True
    assert TrainConfig(dataset="x", second_kl=False).weights().second_kl is False
    # the other two never touch Weights: the loss assembly is untouched by them
    for kw in ({"class_mean_prior": True}, {"adv_input": "xhat"}):
        assert TrainConfig(dataset="x", **kw).weights() == base.weights()


# -- the three query/prior ablations (2026-09-23) ---------------------------

def _pinned_parameter_names(model):
    """Everything that existed before the query arms -- the parameters a fit
    with a new query must still initialise identically."""
    return [n for n, _ in model.named_parameters()
            if not n.startswith(("query_vec", "query_phi"))]


def test_query_default_is_the_pinned_embed_t_bit_for_bit():
    torch.manual_seed(7)
    base = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                   gat_dim=6)
    torch.manual_seed(7)
    explicit = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                       gat_dim=6, query="type")
    for a, b in zip(base.state_dict().values(), explicit.state_dict().values()):
        assert torch.equal(a, b)
    assert not hasattr(base, "query_vec") and not hasattr(base, "query_phi")

    _, _, _, fwd_a = _forward(seed=3)
    _, _, _, fwd_b = _forward(seed=3, query="type")
    assert torch.equal(fwd_a.c, fwd_b.c)
    assert torch.equal(fwd_a.alpha, fwd_b.alpha)
    assert torch.equal(fwd_a.log_p, fwd_b.log_p)


def test_the_query_arms_leave_every_pinned_parameter_s_draw_untouched():
    """The new query module is built last, so turning an arm on cannot move
    any other layer's initialisation -- an arm difference is the arm's."""
    torch.manual_seed(7)
    base = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                   gat_dim=6)
    for arm in ("type_free", "image"):
        torch.manual_seed(7)
        other = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                        gat_dim=6, query=arm)
        assert _pinned_parameter_names(other) == _pinned_parameter_names(base)
        named = dict(other.named_parameters())
        for name, p in base.named_parameters():
            assert torch.equal(p, named[name])


def test_query_type_free_is_one_shared_vector_and_ignores_embed_t():
    import torch.nn.functional as F

    model, tensors, batch, fwd = _forward(seed=3, query="type_free")
    n_context, gat_dim = batch.n_context, 6
    assert model.query_vec.shape == (model.n_types,)     # type_only src_dim
    # planted: the context block is the GAT queried with the shared vector
    h_src = F.one_hot(tensors["t"], model.n_types).float()
    gat, alpha = model.gat(h_src, model.query_vec.expand(n_context, -1),
                           tensors["gat_src"], tensors["gat_dst"])
    assert torch.equal(fwd.c[:, :gat_dim], gat)
    assert torch.equal(fwd.alpha, alpha)
    # and the type embedding is no longer read at all
    with torch.no_grad():
        model.embed_t.weight.zero_()
    torch.manual_seed(103)
    again = model(**tensors, kappa=0.1)
    assert torch.equal(again.alpha, fwd.alpha)


def test_query_image_reads_phi_where_the_pinned_query_does_not():
    import torch.nn.functional as F

    model, tensors, batch, fwd = _forward(seed=3, query="image")
    n_context, gat_dim = batch.n_context, 6
    h_src = F.one_hot(tensors["t"], model.n_types).float()
    gat, alpha = model.gat(h_src, model.query_phi(tensors["phi"][:n_context]),
                           tensors["gat_src"], tensors["gat_dst"])
    assert torch.equal(fwd.c[:, :gat_dim], gat)
    assert torch.equal(fwd.alpha, alpha)

    # the plant: Phi now moves the attention, which under "type" it cannot
    moved = dict(tensors, phi=tensors["phi"] + 3.0)
    torch.manual_seed(103)
    assert not torch.allclose(model(**moved, kappa=0.1).alpha, fwd.alpha)
    pinned, pinned_tensors, _, pinned_fwd = _forward(seed=3)
    torch.manual_seed(103)
    moved = dict(pinned_tensors, phi=pinned_tensors["phi"] + 3.0)
    assert torch.equal(pinned(**moved, kappa=0.1).alpha, pinned_fwd.alpha)


def test_prior_type_free_off_is_the_m_psi_mlp_bit_for_bit():
    from discell.model.networks import TypeFreePrior

    torch.manual_seed(7)
    base = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                   gat_dim=6)
    torch.manual_seed(7)
    explicit = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                       gat_dim=6, prior_type_free=False)
    assert not isinstance(base.prior_w, TypeFreePrior)
    for a, b in zip(base.state_dict().values(), explicit.state_dict().values()):
        assert torch.equal(a, b)

    _, _, _, fwd_a = _forward(seed=3)
    _, _, _, fwd_b = _forward(seed=3, prior_type_free=False)
    assert torch.equal(fwd_a.prior_mean_w, fwd_b.prior_mean_w)
    assert torch.equal(fwd_a.log_p, fwd_b.log_p)


def test_prior_type_free_on_reads_the_context_and_nothing_else():
    model, tensors, batch, fwd = _forward(seed=3, prior_type_free=True)
    # planted: the prior is the MLP on c alone
    assert torch.equal(fwd.prior_mean_w, model.prior_w.net(fwd.c))
    # the type input is dropped: any one-hot tail gives the same answer
    import torch.nn.functional as F
    t = tensors["t"][: batch.n_context]
    rolled = (t + 1) % model.n_types
    a = model.prior_w(torch.cat([fwd.c, F.one_hot(t, model.n_types).float()], -1))
    b = model.prior_w(torch.cat([fwd.c,
                                 F.one_hot(rolled, model.n_types).float()], -1))
    assert torch.equal(a, b) and torch.equal(a, fwd.prior_mean_w)


def test_prior_type_free_leaves_the_w_posterior_conditioning_intact():
    """q(w | c, t, z, x) still reads t; only the target moved."""
    model, tensors, batch, fwd = _forward(seed=3, prior_type_free=True)
    c_dim = model.enc_w[0].in_features - (model.n_types + model.d_z + 25 + 1)
    assert c_dim == fwd.c.shape[1]
    assert model.prior_w.net[0].in_features == c_dim      # no one-hot block
    loss = discell_loss(fwd, tensors["x"][: batch.n_seeds],
                        tensors["t"][: batch.n_seeds],
                        weights=Weights(alpha_w=0.1)).loss
    loss.backward()
    assert model.prior_w.net[0].weight.grad.abs().sum() > 0


def test_type_free_prior_is_a_drop_in_for_the_prior_w_call_sites():
    """transport/degeneracy call model.prior_w([c, onehot(t)]) unchanged."""
    import torch.nn.functional as F

    from discell.model.networks import TypeFreePrior, mlp

    head = TypeFreePrior(3, mlp([7, 4, 2]))
    c = torch.randn(5, 7)
    t = torch.tensor([0, 1, 2, 1, 0])
    out = head(torch.cat([c, F.one_hot(t, 3).float()], dim=-1))
    assert torch.equal(out, head.net(c))
    other = head(torch.cat([c, F.one_hot((t + 1) % 3, 3).float()], dim=-1))
    assert torch.equal(out, other)


def test_trainer_wiring_matches_the_query_and_prior_flags():
    from discell.model.train import TrainConfig

    base = TrainConfig(dataset="x")
    assert (base.query, base.prior_type_free) == ("type", False)
    # neither flag touches the objective: the loss assembly is untouched
    for kw in ({"query": "image"}, {"query": "type_free"},
               {"prior_type_free": True}):
        assert TrainConfig(dataset="x", **kw).weights() == base.weights()


def test_an_unknown_query_is_refused_at_construction():
    import pytest

    with pytest.raises(ValueError, match="type_free"):
        DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                gat_dim=6, query="phi")


# -- review R12 Test 2: the leak coefficient's form (2026-09-24) -------------

def _leak_forward(seed=3, x=None, area=None, **model_kw):
    """The tiny tile under one leak form; same init and noise draws as the
    pinned forward, so any difference is the form's."""
    torch.manual_seed(seed)
    model, tensors, batch = tiny_tile()
    torch.manual_seed(seed)
    model = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                    gat_dim=6, **model_kw)
    if x is not None:
        tensors = dict(tensors, x=x)
    if area is not None:
        tensors = dict(tensors, area=area)
    torch.manual_seed(seed + 100)
    return model, tensors, batch, model(**tensors, kappa=0.1)


def _has_leak_edge(tensors, n_seeds):
    return torch.zeros(n_seeds).index_add_(
        0, tensors["leak_dst"], tensors["leak_beta"]) > 0


def test_kappa_mode_default_is_the_pinned_mixture_bit_for_bit():
    from discell.model.equations import leakage_mix

    torch.manual_seed(7)
    base = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                   gat_dim=6)
    torch.manual_seed(7)
    explicit = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                       gat_dim=6, kappa_mode="global")
    assert list(base.state_dict()) == list(explicit.state_dict())
    assert "kappa_gene_scale" not in base.state_dict()
    for a, b in zip(base.state_dict().values(), explicit.state_dict().values()):
        assert torch.equal(a, b)

    _, _, batch, fwd_a = _leak_forward()
    _, _, _, fwd_b = _leak_forward(kappa_mode="global")
    assert torch.equal(fwd_a.log_p, fwd_b.log_p)
    assert torch.equal(fwd_a.log_p_breve, fwd_b.log_p_breve)
    # the float itself goes into the mixture: the pinned code path
    assert isinstance(fwd_a.kappa_eff, float) and fwd_a.kappa_eff == 0.1
    n = batch.n_seeds
    assert torch.equal(fwd_a.log_p, leakage_mix(fwd_a.log_rho[:n].exp(),
                                                fwd_a.rho_bar, 0.1))


def test_depth_mode_reduces_to_global_when_every_depth_is_equal():
    _, tensors, batch, _ = _leak_forward()
    flat = torch.full_like(tensors["x"], 2.0)          # l = 50 on every node
    _, _, _, pinned = _leak_forward(x=flat)
    # every ratio is 1, so is its training mean
    _, _, _, depth = _leak_forward(x=flat, kappa_mode="depth",
                                   kappa_ratio_mean=1.0)
    n = batch.n_seeds
    assert depth.kappa_eff.shape == (n, 1)
    assert torch.allclose(depth.kappa_eff, torch.full((n, 1), 0.1), atol=1e-6)
    assert torch.allclose(depth.log_p, pinned.log_p, atol=1e-6)
    assert torch.allclose(depth.log_p_breve, pinned.log_p_breve, atol=1e-6)


def test_depth_mode_is_donor_over_receiver_depth_clipped_at_one_half():
    """Planted: kappa_i = min(kappa * (sum_j beta_ij l_j / l_i) / m, 0.5),
    m the training-mean ratio, computed by hand from the tile's leak edges; a
    shallow receiver among deep donors hits the cap, and the cell without a
    leak edge keeps kappa."""
    _, tensors, batch, _ = _leak_forward()
    n, n_ctx = batch.n_seeds, batch.n_context
    x = tensors["x"].clone()
    x[1] = 0.0
    x[1, 0] = 2.0                  # seed 1: l = 2 among l ~ 75 neighbours
    _, tensors, _, fwd = _leak_forward(x=x, kappa_mode="depth",
                                       kappa_ratio_mean=1.3)
    depth = x[:n_ctx].sum(-1).numpy().astype(np.float64)
    src, dst = tensors["leak_src"].numpy(), tensors["leak_dst"].numpy()
    beta = tensors["leak_beta"].numpy().astype(np.float64)
    donor = np.zeros(n)
    np.add.at(donor, dst, beta * depth[src])
    edge = _has_leak_edge(tensors, n).numpy()
    planted = np.where(edge, np.minimum(0.1 * donor / depth[:n] / 1.3, 0.5),
                       0.1)
    got = fwd.kappa_eff[:, 0].numpy()
    assert np.allclose(got, planted, atol=1e-6)
    assert got[1] == 0.5                                   # the cap binds
    assert (got <= 0.5).all() and (got >= 0.0).all()
    assert (~edge).any() and np.all(got[~edge] == np.float32(0.1))
    # c_max = 0.5 / kappa: the uncapped ratio of seed 1 is far above it
    assert donor[1] / depth[1] / 1.3 > 0.5 / 0.1


def test_gene_mode_reduces_to_global_when_s_g_is_constant():
    _, _, _, pinned = _leak_forward()
    _, _, batch, gene = _leak_forward(kappa_mode="gene",
                                      kappa_gene_share=np.full(25, 0.37))
    assert gene.kappa_eff.shape == (batch.n_seeds, 25)
    assert torch.allclose(gene.log_p, pinned.log_p, atol=1e-6)
    assert torch.allclose(gene.log_p_breve, pinned.log_p_breve, atol=1e-6)


def test_gene_mode_is_the_renormalised_per_gene_mixture_with_the_clip():
    """Planted: kappa_g = min(kappa * s_g / mean(s_g), 0.9), p_ig proportional
    to (1-kappa_g) rho_ig + kappa_g rho_bar_ig for a cell with a leak edge, rho
    for one without."""
    share = np.random.default_rng(4).uniform(0.05, 0.7, 25)
    share[3] = 60.0                        # kappa * s / mean(s) > 0.9: clipped
    _, tensors, batch, fwd = _leak_forward(kappa_mode="gene",
                                           kappa_gene_share=share)
    n = batch.n_seeds
    k_g = np.minimum(0.1 * share / share.mean(), 0.9)
    assert k_g[3] == 0.9
    edge = _has_leak_edge(tensors, n)
    got = fwd.kappa_eff.numpy()
    assert np.allclose(got[edge.numpy()], k_g[None, :], atol=1e-6)
    assert np.all(got[~edge.numpy()] == 0.0)
    rho = fwd.log_rho[:n].exp().double()
    bar = fwd.rho_bar.double()
    k = torch.as_tensor(k_g)[None, :]
    mix = (1 - k) * rho + k * bar
    planted = torch.where(edge[:, None], mix / mix.sum(-1, keepdim=True), rho)
    assert torch.allclose(fwd.log_p.exp().double(), planted, atol=1e-6)


def test_every_leak_form_decodes_rows_that_sum_to_one():
    share = np.random.default_rng(5).uniform(0.05, 0.7, 25)
    for kw in ({}, {"kappa_mode": "depth", "kappa_ratio_mean": 1.2},
               {"kappa_mode": "gene", "kappa_gene_share": share},
               {"kappa_mode": "density", "kappa_ratio_mean": 1.2,
                "area": _areas(10)}):
        _, tensors, batch, fwd = _leak_forward(**kw)
        for log_p in (fwd.log_p, fwd.log_p_breve):
            assert torch.allclose(log_p.exp().sum(-1),
                                  torch.ones(batch.n_seeds), atol=1e-5), kw
        # a cell without a leak edge decodes to its clean composition
        lone = ~_has_leak_edge(tensors, batch.n_seeds)
        assert lone.any()
        assert torch.allclose(fwd.log_p[lone].exp(),
                              fwd.log_rho[:batch.n_seeds][lone].exp(),
                              atol=1e-6), kw


def test_gene_mode_s_g_travels_in_the_checkpoint():
    share = np.random.default_rng(6).uniform(0.05, 0.7, 25)
    model, tensors, _, fwd = _leak_forward(kappa_mode="gene",
                                           kappa_gene_share=share)
    state = model.state_dict()
    assert torch.allclose(state["kappa_gene_scale"],
                          torch.as_tensor(share / share.mean(),
                                          dtype=torch.float32))
    # a gene-form model rebuilt without s_g cannot decode until it is loaded
    bare = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                   gat_dim=6, kappa_mode="gene")
    assert torch.isnan(bare.kappa_gene_scale).all()
    bare.load_state_dict(state)
    torch.manual_seed(103)
    again = bare(**tensors, kappa=0.1)
    assert torch.equal(again.log_p, fwd.log_p)


def test_the_leak_forms_leave_every_parameter_draw_untouched():
    torch.manual_seed(7)
    base = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                   gat_dim=6)
    for kw in ({"kappa_mode": "depth"},
               {"kappa_mode": "gene", "kappa_gene_share": np.ones(25)}):
        torch.manual_seed(7)
        other = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2,
                        hidden=16, gat_dim=6, **kw)
        named = dict(other.named_parameters())
        assert list(named) == [n for n, _ in base.named_parameters()]
        for name, p in base.named_parameters():
            assert torch.equal(p, named[name])


def test_unknown_or_incompatible_leak_forms_are_refused():
    import pytest

    with pytest.raises(ValueError, match="kappa_mode"):
        DisCell(25, 3, 4, median_counts=100.0, kappa_mode="type")
    with pytest.raises(ValueError, match="subtract_leak"):
        DisCell(25, 3, 4, median_counts=100.0, kappa_mode="depth",
                subtract_leak=True)
    with pytest.raises(ValueError, match="shape"):
        DisCell(25, 3, 4, median_counts=100.0, kappa_mode="gene",
                kappa_gene_share=np.ones(24))


def test_trainer_wiring_matches_the_kappa_mode_flags(tmp_path, monkeypatch):
    import json

    import pytest

    from discell import paths
    from discell.model.networks import read_gene_share
    from discell.model.train import TrainConfig, build_parser

    base = TrainConfig(dataset="x")
    assert (base.kappa_mode, base.kappa_gene_source) == ("global", None)
    monkeypatch.setattr(paths, "dataset",
                        lambda _: type("D", (), {"root": tmp_path})())
    gene = TrainConfig(dataset="x", kappa_mode="gene")
    assert gene.kappa_gene_source == str(
        tmp_path / "experiments" / "gene_extranuclear_share.npy")
    # the form is not an objective weight: the loss assembly is untouched
    for kw in ({"kappa_mode": "depth"}, {"kappa_mode": "gene"}):
        assert TrainConfig(dataset="x", **kw).weights() == base.weights()
    with pytest.raises(ValueError, match="kappa_mode"):
        TrainConfig(dataset="x", kappa_mode="cell")
    args = vars(build_parser().parse_args(
        ["--dataset", "x", "--kappa-mode", "depth"]))
    args.pop("quiet")
    assert TrainConfig(**args).kappa_mode == "depth"

    # the share file: aligned by its sidecar's gene order, refused otherwise
    path = tmp_path / "s.npy"
    np.save(path, np.array([0.2, 0.4, 0.6]))
    path.with_suffix(".json").write_text(json.dumps({"gene_names": ["a", "b", "c"]}))
    assert np.allclose(read_gene_share(path, ["a", "b", "c"]), [0.2, 0.4, 0.6])
    with pytest.raises(ValueError, match="order"):
        read_gene_share(path, ["b", "a", "c"])
    with pytest.raises(ValueError, match="genes"):
        read_gene_share(path, ["a", "b"])


def test_trainer_fits_under_each_leak_form_and_records_the_source(tmp_path,
                                                                 monkeypatch):
    import json

    from discell import paths
    from discell.model.train import TrainConfig, Trainer
    from tests.test_model_train import _small_data

    monkeypatch.setattr(paths, "dataset",
                        lambda _: type("D", (), {"root": tmp_path})())
    data = _small_data()
    share = np.random.default_rng(8).uniform(0.05, 0.7, data.x.shape[1])
    np.save(tmp_path / "share.npy", share)
    for mode in ("depth", "gene"):
        config = TrainConfig(
            dataset="synthetic-smoke", run_name=f"r12_{mode}", kappa=0.1,
            d_z=6, d_w=2, hidden=32, gat_dim=8, epochs=2, eval_every=1,
            figures_every=1000, patience=100, device="cpu", v_pcs=4,
            invariance="closed_form", kappa_mode=mode,
            kappa_gene_source=(str(tmp_path / "share.npy")
                               if mode == "gene" else None))
        trainer = Trainer(config, data)
        summary = trainer.fit()
        assert np.isfinite(summary["final"]["recon_val"])
        assert np.isfinite(summary["final"]["recon_gap"]["gap"])
        stored = json.loads((tmp_path / "runs" / f"r12_{mode}"
                             / "config.json").read_text())
        assert stored["kappa_mode"] == mode
        assert stored["kappa_gene_source"] == config.kappa_gene_source
        # the depth normaliser is computed at setup and recorded
        assert (stored["kappa_ratio_mean"] is None) == (mode == "gene")
        rebuilt = TrainConfig(**{k: v for k, v in stored.items() if k != "git"})
        assert rebuilt == trainer.config


def test_transport_reads_the_group_kappa_and_keeps_the_float_when_global():
    from types import SimpleNamespace

    from discell.model.equations import leakage_mix
    from discell.model.transport import _decode_cells, _stack_kappa, group_kappa

    kappa = 0.1
    assert group_kappa({"rho": np.zeros((3, 5))}, 1, kappa) is kappa
    assert _stack_kappa([kappa, kappa, kappa]) is kappa
    per_cell = {"kappa": np.array([[0.1], [0.3], [0.2]])}
    assert group_kappa(per_cell, 1, kappa) == 0.3
    per_gene = {"kappa": np.arange(15.0).reshape(3, 5) / 100}
    assert np.array_equal(group_kappa(per_gene, 2, kappa), per_gene["kappa"][2])
    rows = _stack_kappa([0.1, 0.3, 0.3])
    assert rows.shape == (3, 1) and rows[1, 0] == 0.3

    # the decode takes each form the channels hand it; chunked per-row kappa
    model, tensors, batch, fwd = _leak_forward()
    trainer = SimpleNamespace(model=model.eval())
    n = batch.n_seeds
    mu_z, w = fwd.mu_z[:n].detach().numpy(), fwd.mu_w[:n].detach().numpy()
    bar = fwd.rho_bar.numpy()
    with torch.no_grad():
        rho = model.log_rho(fwd.mu_z[:n], fwd.mu_w[:n]).exp()
    ref = leakage_mix(rho, fwd.rho_bar, kappa).exp().numpy()
    # one chunk: the float takes the pinned path exactly; several chunks
    # only reorder the matmuls, so they agree to float rounding
    assert np.array_equal(_decode_cells(trainer, mu_z, w, bar, kappa), ref)
    assert np.allclose(_decode_cells(trainer, mu_z, w, bar, kappa, chunk=3),
                       ref, atol=1e-7)
    k_rows = np.linspace(0.05, 0.4, n)[:, None]
    got = _decode_cells(trainer, mu_z, w, bar, k_rows, chunk=3)
    want = leakage_mix(rho, fwd.rho_bar,
                       torch.as_tensor(k_rows, dtype=torch.float32)).exp().numpy()
    assert np.allclose(got, want, atol=1e-7)
    k_genes = np.linspace(0.02, 0.3, 25)
    got = _decode_cells(trainer, mu_z, w, bar, k_genes, chunk=3)
    want = leakage_mix(rho, fwd.rho_bar,
                       torch.as_tensor(k_genes, dtype=torch.float32)).exp().numpy()
    assert np.allclose(got, want, atol=1e-7)


def test_collect_channels_carries_the_group_kappa_only_off_the_global_form():
    from types import SimpleNamespace

    from discell.model.train import Trainer
    from discell.model.transport import collect_channels

    share = np.random.default_rng(9).uniform(0.05, 0.7, 25)
    for kw in ({}, {"kappa_mode": "depth", "kappa_ratio_mean": 1.0},
               {"kappa_mode": "gene", "kappa_gene_share": share},
               {"kappa_mode": "density", "kappa_ratio_mean": 1.0,
                "area": _areas(10)}):
        model, tensors, batch, _ = _leak_forward(**kw)
        n = batch.n_seeds
        tile = dict(tensors, nodes=np.asarray(batch.nodes))
        trainer = SimpleNamespace(model=model.eval(), train_batches=[tile],
                                  val_batches=[], config=SimpleNamespace(kappa=0.1),
                                  _forward_kwargs=Trainer._forward_kwargs)
        group = np.full(int(np.max(batch.nodes)) + 1, -1)
        group[np.asarray(batch.nodes)[:n]] = np.arange(n) % 2      # two groups
        channels = collect_channels(trainer, group, 2)
        if not kw:
            assert "kappa" not in channels
            continue
        with torch.no_grad():
            fwd = model(**Trainer._forward_kwargs(tile), kappa=0.1, sample=False)
        k = fwd.kappa_eff.double().numpy()
        for gid in (0, 1):
            assert np.allclose(channels["kappa"][gid],
                               k[np.arange(n) % 2 == gid].mean(axis=0))


def test_depth_mode_moves_the_leaked_amount_where_row_renormalising_cannot():
    """The entry's note, executable: reweighting beta_ij by l_j / l_i and
    renormalising the row cancels l_i, so the leaked count stays kappa * l_i
    whoever the donors are. The depth form leaks kappa_i * l_i = kappa *
    sum_j beta_ij l_j (below the cap), which moves with the donors."""
    _, tensors, batch, _ = _leak_forward()
    n, n_ctx = batch.n_seeds, batch.n_context
    x = tensors["x"].clone()
    x[:n_ctx] *= torch.linspace(0.5, 3.0, n_ctx)[:, None]      # varied depths
    _, tensors, _, fwd = _leak_forward(x=x, kappa_mode="depth",
                                       kappa_ratio_mean=1.0)
    depth = x[:n_ctx].sum(-1).double()
    src, dst = tensors["leak_src"], tensors["leak_dst"]
    beta = tensors["leak_beta"].double()
    edge = _has_leak_edge(tensors, n)

    reweighted = beta * depth[src] / depth[dst]
    row = torch.zeros(n, dtype=torch.float64).index_add_(0, dst, reweighted)
    renormalised = reweighted / row[dst]
    leaked_renorm = 0.1 * depth[:n] * torch.zeros(
        n, dtype=torch.float64).index_add_(0, dst, renormalised)
    assert torch.allclose(leaked_renorm[edge], 0.1 * depth[:n][edge])

    donor = torch.zeros(n, dtype=torch.float64).index_add_(
        0, dst, beta * depth[src])
    leaked_depth = fwd.kappa_eff[:, 0].double() * depth[:n]
    below = edge & (0.1 * donor / depth[:n] < 0.5)
    assert below.sum() >= 3
    assert torch.allclose(leaked_depth[below], 0.1 * donor[below], rtol=1e-5)
    assert not torch.allclose(leaked_depth[below], 0.1 * depth[:n][below],
                              rtol=1e-3)



# -- the density form (author, 2026-09-24): depth on l / A ------------------

def _areas(n, seed=11):
    return torch.as_tensor(np.random.default_rng(seed).uniform(20.0, 200.0, n),
                           dtype=torch.float32)


def test_density_mode_reduces_to_depth_when_every_area_is_equal():
    _, tensors, batch, _ = _leak_forward()
    same = torch.full((tensors["x"].shape[0],), 73.0)
    _, _, _, depth = _leak_forward(kappa_mode="depth", kappa_ratio_mean=1.1)
    _, _, _, dens = _leak_forward(kappa_mode="density", area=same,
                                  kappa_ratio_mean=1.1)
    assert torch.allclose(dens.kappa_eff, depth.kappa_eff, atol=1e-6)
    assert torch.allclose(dens.log_p, depth.log_p, atol=1e-6)
    assert torch.allclose(dens.log_p_breve, depth.log_p_breve, atol=1e-6)


def test_density_mode_is_donor_over_receiver_density_with_the_depth_clip():
    """Planted: kappa_i = min(kappa * sum_j beta_ij (l_j/A_j) / (l_i/A_i),
    0.5); the cell without a leak edge keeps kappa; rows sum to one and that
    cell decodes to rho."""
    _, tensors, batch, _ = _leak_forward()
    n, n_ctx = batch.n_seeds, batch.n_context
    area = _areas(tensors["x"].shape[0])
    area[1] = 2000.0               # seed 1: a huge, sparse receiver -> the cap
    _, tensors, _, fwd = _leak_forward(kappa_mode="density", area=area,
                                       kappa_ratio_mean=0.8)
    depth = tensors["x"][:n_ctx].sum(-1).double().numpy()
    a = area[:n_ctx].double().numpy()
    src, dst = tensors["leak_src"].numpy(), tensors["leak_dst"].numpy()
    beta = tensors["leak_beta"].numpy().astype(np.float64)
    donor = np.zeros(n)
    np.add.at(donor, dst, beta * depth[src] / a[src])
    edge = _has_leak_edge(tensors, n).numpy()
    receiver = np.maximum(depth[:n], 1.0) / a[:n]
    planted = np.where(edge, np.minimum(0.1 * donor / receiver / 0.8, 0.5),
                       0.1)
    got = fwd.kappa_eff[:, 0].numpy()
    assert np.allclose(got, planted, atol=1e-6)
    assert got[1] == 0.5
    assert torch.allclose(fwd.log_p.exp().sum(-1), torch.ones(n), atol=1e-5)
    lone = ~_has_leak_edge(tensors, n)
    assert torch.allclose(fwd.log_p[lone].exp(), fwd.log_rho[:n][lone].exp(),
                          atol=1e-6)


def test_density_mode_refuses_to_run_without_areas():
    import pytest

    with pytest.raises(ValueError, match="areas"):
        _leak_forward(kappa_mode="density")


def test_density_areas_replace_expansion_cells_by_nucleus_times_type_ratio():
    from discell.model.networks import density_areas

    cell = np.array([100.0, 80.0, 60.0, 50.0, 40.0, 30.0, 90.0, 70.0])
    nuc = np.array([50.0, 20.0, 30.0, 25.0, 10.0, 15.0, np.nan, 35.0])
    method = np.array(["Segmented by boundary stain", "Segmented by interior stain (18S)",
                       "Segmented by nucleus expansion of 5.0um",
                       "Segmented by interior stain (18S)",
                       "Segmented by nucleus expansion of 5.0um",
                       "Segmented by nucleus expansion of 5.0um",
                       "Segmented by nucleus expansion of 5.0um",
                       "Segmented by interior stain (18S)"])
    t = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    area, report = density_areas(cell, nuc, method, t, 4)
    # type 0: non-expansion ratios 2.0 and 4.0 -> median 3.0
    assert report["by_type"][0]["median_cell_over_nucleus"] == 3.0
    assert area[2] == 30.0 * 3.0
    assert area[0] == 100.0 and area[1] == 80.0          # untouched
    # type 1: one non-expansion cell, ratio 2.0
    assert area[4] == 10.0 * 2.0
    # type 2 has no non-expansion cell: the pooled median (2, 4, 2, 2) = 2.0
    assert report["by_type"][2]["ratio_from"] == "pooled"
    assert area[5] == 15.0 * report["pooled_median_cell_over_nucleus"] == 30.0
    # an expansion cell without a nuclear area keeps its cell area, counted
    assert area[6] == 90.0
    assert report["by_type"][2]["expansion_without_nucleus"] == 1
    assert report["by_type"][0]["expansion_fraction"] == 1 / 3
    assert report["by_type"][3]["expansion_fraction"] == 0.0
    assert report["expansion_fraction"] == 4 / 8


def test_trainer_carries_areas_under_the_density_form_only(tmp_path, monkeypatch):
    import json

    import discell.model.train as train_module
    from discell import paths
    from discell.model.train import TrainConfig, Trainer
    from tests.test_model_train import _small_data

    monkeypatch.setattr(paths, "dataset",
                        lambda _: type("D", (), {"root": tmp_path})())
    data = _small_data()
    areas = np.random.default_rng(12).uniform(20.0, 200.0, data.x.shape[0])
    calls = []

    def fake_read(dataset, variant, t, n_types, type_names=None):
        calls.append(dataset)
        return areas, {"rule": "planted", "expansion_fraction": 0.0, "by_type": {}}

    monkeypatch.setattr(train_module, "read_density_areas", fake_read)
    base = dict(dataset="synthetic-smoke", kappa=0.1, d_z=6, d_w=2, hidden=32,
                gat_dim=8, epochs=2, eval_every=1, figures_every=1000,
                patience=100, device="cpu", v_pcs=4, invariance="closed_form")
    pinned = Trainer(TrainConfig(run_name="r12_global", **base), data)
    assert "area" not in pinned.train_batches[0]
    assert "area" not in Trainer._forward_kwargs(pinned.train_batches[0])
    assert not calls

    trainer = Trainer(TrainConfig(run_name="r12_density", kappa_mode="density",
                                  **base), data)
    batch = trainer.train_batches[0]
    n_ctx = batch["n_context"]
    assert np.allclose(batch["area"].numpy(), areas[batch["nodes"][:n_ctx]])
    assert torch.equal(Trainer._forward_kwargs(batch)["area"], batch["area"])
    summary = trainer.fit()
    assert np.isfinite(summary["final"]["recon_val"])
    stored = json.loads((tmp_path / "runs" / "r12_density"
                         / "kappa_density.json").read_text())
    assert stored["area"]["rule"] == "planted"
    assert stored["ratio_mean_used"] == trainer.config.kappa_ratio_mean



# -- the amendment: depth/density normalised to a mean leak fraction of kappa

def _mean_kappa_over_training_seeds(trainer):
    """kappa_i of every connected training seed, by the model's own forward."""
    from discell.model.networks import leak_ratio

    got, nodes = [], []
    with torch.no_grad():
        for batch in trainer.train_batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch),
                                kappa=trainer.config.kappa, sample=False)
            n = batch["n_seeds"]
            got.append(fwd.kappa_eff[:, 0].numpy())
            nodes.append(np.asarray(batch["nodes"])[:n])
    got, nodes = np.concatenate(got), np.concatenate(nodes)
    _, has_edge = leak_ratio(trainer.data.graph.in_edges, trainer.data.totals,
                             trainer.cell_area)
    return got, nodes, has_edge[nodes]


def test_depth_and_density_have_mean_kappa_i_kappa_over_the_training_cells(
        tmp_path, monkeypatch):
    """Planted: the normaliser is solved on the training cells' r, so the
    post-clip mean kappa_i over the connected training cells is kappa; the
    full-slide ratio computed once at setup is the one each tile's forward
    pass sees, cell for cell."""
    import discell.model.train as train_module
    from discell import paths
    from discell.model.networks import leak_ratio
    from discell.model.train import TrainConfig, Trainer
    from tests.test_model_train import _small_data

    monkeypatch.setattr(paths, "dataset",
                        lambda _: type("D", (), {"root": tmp_path})())
    data = _small_data()
    areas = np.random.default_rng(13).uniform(20.0, 200.0, data.x.shape[0])
    monkeypatch.setattr(train_module, "read_density_areas",
                        lambda *a, **k: (areas, {"rule": "planted"}))
    for mode, kappa in (("depth", 0.1), ("density", 0.1), ("depth", 0.3)):
        trainer = Trainer(TrainConfig(
            dataset="synthetic-smoke", run_name=f"norm_{mode}", kappa=kappa,
            d_z=6, d_w=2, hidden=32, gat_dim=8, device="cpu", v_pcs=4,
            invariance="closed_form", kappa_mode=mode), data)
        report = trainer.kappa_ratio_report
        assert trainer.config.kappa_ratio_mean == report["normaliser"]
        assert float(trainer.model.kappa_ratio_mean) == np.float32(
            report["normaliser"])
        # the clip-aware m sits at or below the plain mean of r
        assert report["normaliser"] <= report["ratio_mean_train"]
        got, nodes, connected = _mean_kappa_over_training_seeds(trainer)
        # the tiles' seeds are exactly the training cells
        assert np.array_equal(np.sort(nodes),
                              np.sort(np.concatenate(data.train_tiles)))
        mean = got[connected].mean()
        assert abs(report["kappa_i_mean_train"] - kappa) < 1e-9
        assert abs(mean - kappa) < 1e-3              # float32 forward pass
        # cell for cell: the tile ratio is the slide ratio over its mean
        r, _ = leak_ratio(data.graph.in_edges, data.totals, trainer.cell_area)
        planted = np.minimum(kappa * r[nodes] / report["normaliser"], 0.5)
        assert np.allclose(got[connected], planted[connected], atol=1e-5)
        assert np.all(got[~connected] == np.float32(kappa))
        # without the normaliser the arm would leak more on average
        assert report["ratio_mean_train"] != 1.0


def test_the_normaliser_travels_in_the_checkpoint_and_the_config():
    model, tensors, _, fwd = _leak_forward(kappa_mode="depth",
                                           kappa_ratio_mean=1.37)
    state = model.state_dict()
    assert float(state["kappa_ratio_mean"]) == np.float32(1.37)
    bare = DisCell(25, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                   gat_dim=6, kappa_mode="depth")
    assert torch.isnan(bare.kappa_ratio_mean)            # cannot decode unloaded
    bare.load_state_dict(state)
    torch.manual_seed(103)
    assert torch.equal(bare(**tensors, kappa=0.1).log_p, fwd.log_p)
    # the global and gene forms carry no normaliser
    for kw in ({}, {"kappa_mode": "gene", "kappa_gene_share": np.ones(25)}):
        other = DisCell(25, 3, 4, median_counts=100.0, **kw)
        assert "kappa_ratio_mean" not in other.state_dict()



def test_the_clip_aware_normaliser_gives_mean_kappa_on_a_heavy_tailed_ratio():
    """Amendment 2, planted: on a heavy-tailed r (lognormal, sigma 1.5, where
    the plain-mean normaliser leaves the post-clip mean well below kappa) the
    solved m restores it to within 1e-3; c_max stays 0.5 / kappa."""
    import pytest

    from discell.model.networks import solve_ratio_normaliser

    r = np.random.default_rng(14).lognormal(0.0, 1.5, 200_000)
    for kappa in (0.1, 0.3):
        plain = np.minimum(kappa * r / r.mean(), 0.5).mean()
        assert kappa - plain > 0.005                 # the clip bites
        m = solve_ratio_normaliser(r, kappa)
        k_i = np.minimum(kappa * r / m, 0.5)
        assert abs(k_i.mean() - kappa) < 1e-3
        assert (k_i == 0.5).mean() > (np.minimum(kappa * r / r.mean(), 0.5)
                                      == 0.5).mean()
        assert m < r.mean()
    # a ratio the cap never reaches keeps the plain mean
    flat = np.random.default_rng(15).uniform(0.9, 1.1, 1000)
    assert solve_ratio_normaliser(flat, 0.1) == flat.mean()
    assert solve_ratio_normaliser(r, 0.0) == r.mean()
    # too few cells with r > 0 for the cap to carry a mean of kappa
    sparse = np.zeros(1000)
    sparse[:100] = 1.0
    with pytest.raises(ValueError, match="normaliser"):
        solve_ratio_normaliser(sparse, 0.1)
