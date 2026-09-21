"""The optional L2 on the w channel (devlog 2026-09-21, issues V12).

Three planted checks: the default is inert bit-for-bit, each variant equals
its hand computation, and the ``type_mean`` variant is blind to a planted
within-type perturbation while ``w`` is not -- the property the whole
pre-registration turns on.
"""

from __future__ import annotations

import dataclasses as dc

import pytest
import torch

from discell.model.elbo import Weights, discell_loss
from tests.test_model_elbo import tiny_tile


def t_context(fwd, t_seeds):
    """Type ids padded to the context length: only the seed rows matter, the
    ring-1 rows exist so the planted tensors keep the forward pass' shapes."""
    pad = fwd.prior_mean_w.shape[0] - len(t_seeds)
    return torch.cat([t_seeds, t_seeds[:1].repeat(pad)])


def fitted_tile():
    model, tensors, batch = tiny_tile()
    fwd = model(**tensors, kappa=0.2)
    return fwd, tensors["x"][:batch.n_seeds], tensors["t"][:batch.n_seeds]


@pytest.mark.parametrize("kind", ["none", "w", "type_mean"])
def test_lambda_zero_reproduces_the_loss_bit_for_bit(kind):
    fwd, x, t = fitted_tile()
    base = discell_loss(fwd, x, t, weights=Weights(alpha_w=0.1))
    off = discell_loss(fwd, x, t,
                       weights=Weights(alpha_w=0.1, lambda_w=0.0, w_penalty=kind))
    assert float(off.loss) == float(base.loss)
    assert off.w_penalty == 0.0


def test_penalty_w_is_the_mean_squared_norm_of_the_sampled_w():
    fwd, x, t = fitted_tile()
    n = x.shape[0]
    expected = float((fwd.w[:n] ** 2).sum(dim=-1).mean())
    lam = 0.37
    base = discell_loss(fwd, x, t, weights=Weights(alpha_w=0.1))
    got = discell_loss(fwd, x, t,
                       weights=Weights(alpha_w=0.1, lambda_w=lam, w_penalty="w"))
    assert got.w_penalty == pytest.approx(expected, rel=1e-6)
    assert float(got.loss) == pytest.approx(float(base.loss) + lam * expected,
                                            rel=1e-6)


def test_penalty_type_mean_is_the_cell_weighted_per_type_mean_of_m_psi():
    fwd, x, t = fitted_tile()
    n = x.shape[0]
    m = fwd.prior_mean_w[:n]
    total = 0.0
    for type_id in t.unique():
        rows = m[t == type_id]
        total += float(len(rows)) * float((rows.mean(dim=0) ** 2).sum())
    expected = total / n
    got = discell_loss(fwd, x, t,
                       weights=Weights(lambda_w=1.0, w_penalty="type_mean"))
    assert got.w_penalty == pytest.approx(expected, rel=1e-6)


def test_type_mean_equals_mean_square_minus_the_within_type_part():
    """The decomposition the variant is chosen for: it is the ``w`` penalty
    with the context-varying channel exempted."""
    fwd, x, t = fitted_tile()
    n = x.shape[0]
    m = fwd.prior_mean_w[:n]
    centred = m.clone()
    for type_id in t.unique():
        mask = t == type_id
        centred[mask] = m[mask] - m[mask].mean(dim=0)
    expected = float((m ** 2).sum(-1).mean() - (centred ** 2).sum(-1).mean())
    got = discell_loss(fwd, x, t, weights=Weights(lambda_w=1.0,
                                                  w_penalty="type_mean"))
    assert got.w_penalty == pytest.approx(expected, rel=1e-5)


def test_a_planted_per_type_offset_is_charged_by_both_variants():
    fwd, x, t = fitted_tile()
    n = x.shape[0]
    d_w = fwd.mu_w.shape[1]
    n_context = fwd.prior_mean_w.shape[0]
    offset = torch.zeros(int(t.max()) + 1, d_w)
    offset[:, 0] = 5.0
    rows = offset[t_context(fwd, t)]
    shifted = dc.replace(fwd, prior_mean_w=fwd.prior_mean_w + rows,
                         w=fwd.w + rows, mu_w=fwd.mu_w + rows)
    assert n_context >= n
    for kind in ("w", "type_mean"):
        before = discell_loss(fwd, x, t,
                              weights=Weights(lambda_w=1.0, w_penalty=kind))
        after = discell_loss(shifted, x, t,
                             weights=Weights(lambda_w=1.0, w_penalty=kind))
        assert after.w_penalty > before.w_penalty + 10.0


def test_a_planted_within_type_perturbation_is_charged_only_by_w():
    """Zero-mean-within-type noise on w must leave ``type_mean`` untouched."""
    fwd, x, t = fitted_tile()
    n = x.shape[0]
    # the penalty reads the seed rows only; centre there, leave the ring at 0
    noise = torch.zeros_like(fwd.prior_mean_w)
    noise[:n] = torch.randn_like(fwd.prior_mean_w[:n])
    for type_id in t.unique():
        mask = torch.zeros(len(noise), dtype=torch.bool)
        mask[:n] = t == type_id
        noise[mask] = noise[mask] - noise[mask].mean(dim=0)
    shifted = dc.replace(fwd, prior_mean_w=fwd.prior_mean_w + noise,
                         w=fwd.w + noise)
    for kind, changes in (("w", True), ("type_mean", False)):
        before = discell_loss(fwd, x, t,
                              weights=Weights(lambda_w=1.0, w_penalty=kind))
        after = discell_loss(shifted, x, t,
                             weights=Weights(lambda_w=1.0, w_penalty=kind))
        moved = abs(after.w_penalty - before.w_penalty) > 1e-3
        assert moved is changes


def test_unknown_kind_raises():
    fwd, x, t = fitted_tile()
    with pytest.raises(ValueError, match="unknown w_penalty"):
        discell_loss(fwd, x, t, weights=Weights(lambda_w=1.0, w_penalty="nope"))
