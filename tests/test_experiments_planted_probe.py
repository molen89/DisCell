"""The planted-world probe's local fit loop must be fit_synthetic's loop.

:func:`planted_probe.fit_planted` replaces fit_synthetic's invariance term
with the trainer's adversary; in fit_synthetic's own mode it must reproduce
fit_synthetic bit for bit, and the x3 composition weight must be exactly the
pinned encoder term plus twice its composition excess.
"""

from __future__ import annotations

import torch

from discell.experiments import planted_probe as pp


def test_loop_reproduces_fit_synthetic():
    out = pp.check_identity(epochs=1)
    assert out["identical"], out


def test_weighted_encoder_term_matches_pinned():
    from discell.model.elbo import adversary_terms
    from discell.model.networks import Adversary

    torch.manual_seed(0)
    k, n = 5, 64
    heads = Adversary(3, k, k)
    z, t = torch.randn(n, 3), torch.randint(0, k, (n,))
    y, e = (torch.softmax(torch.randn(n, k), -1) for _ in range(2))
    yb, pb = (torch.softmax(torch.randn(k, k), -1) for _ in range(2))
    pinned = adversary_terms(heads, z, t, y, e, yb, pb)
    assert torch.equal(pp.weighted_encoder_term(heads, z, t, y, e, yb, pb, 1.0),
                       pinned.encoder_term)
    tripled = pp.weighted_encoder_term(heads, z, t, y, e, yb, pb, 3.0)
    assert abs(float(tripled - pinned.encoder_term) - 2 * pinned.excess_y) < 1e-5
