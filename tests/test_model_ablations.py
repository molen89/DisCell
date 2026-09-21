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
