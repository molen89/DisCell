#!/usr/bin/env python3
"""Assembling the DisCell objective from one tile's forward pass.

The spec's J (section 5), per seed, averaged over the batch::

    J =   recon(a)                       x under the posterior-drawn w
        + omega * recon(b)               x under the prior-drawn   w-breve
        - (1 + omega) * alpha_z * KL( q(z) || N(0, I) )
        -               alpha_w * KL( q(w) || N(m_psi(c, t), I) )
        -               alpha_a * Pen                                (optional)

    loss = -J

The ``(1 + omega)`` is not a style choice: (a) and (b) are two bounds on the
same data and each carries its own ``-KL(q(z)||p(z))``; dropping the second
copy -- the natural mistake -- costs the bound property even at unit weights
(spec 6.2). Once any ``alpha != 1`` this is a weighted surrogate, not a bound,
and uncertainty comes from the kappa sweep, not from ``q`` (spec 5).

Reconstruction rows arrive scaled by ``1/l_i`` (inside the multinomial), so the
alphas transfer across sections with different depth. The penalty acts on
``mu_z``, not the sample: encoder noise would dilute the measured dependence,
and hiding dependence behind noise is exactly what the penalty must not allow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from discell.model.equations import TypeCovariances, gaussian_kl, multinomial_loglik
from discell.model.networks import Adversary, Forward, soft_cross_entropy


@dataclass(frozen=True)
class Weights:
    """The loss hyperparameters. ``kappa`` lives in the forward pass, not here."""

    omega: float = 1.0      # weight of the intrinsic path (b)
    alpha_z: float = 1.0    # KL(q(z) || N(0,I)) -- applied (1+omega)-fold
    alpha_w: float = 1.0    # KL(q(w) || p(w|c,t))
    alpha_a: float = 0.0    # invariance penalty; 0 = off


@dataclass
class Terms:
    """The scalar to minimise plus every component, detached, for logging."""

    loss: torch.Tensor
    recon_a: float
    recon_b: float
    kl_z: float
    kl_w: float
    penalty: float = 0.0
    penalty_info: dict = field(default_factory=dict)

    def scalars(self) -> dict[str, float]:
        return {"loss": float(self.loss), "recon_a": self.recon_a,
                "recon_b": self.recon_b, "kl_z": self.kl_z, "kl_w": self.kl_w,
                "penalty": self.penalty}


def discell_loss(fwd: Forward, x: torch.Tensor, t: torch.Tensor,
                 weights: Weights | None = None,
                 v: torch.Tensor | None = None,
                 covariances: TypeCovariances | None = None,
                 p_t: torch.Tensor | None = None) -> Terms:
    """The loss for one tile. *x*, *t*, *v* cover the seeds only.

    Every term is a per-seed mean, so the magnitude is invariant to tile size
    and the alphas mean the same thing at every batch shape.
    """
    n_seeds = x.shape[0]
    w = weights or Weights()

    recon_a = multinomial_loglik(x, fwd.log_p).mean()
    recon_b = multinomial_loglik(x, fwd.log_p_breve).mean()
    kl_z = gaussian_kl(fwd.mu_z[:n_seeds], fwd.logvar_z[:n_seeds]).mean()
    # prior variance fixed at sigma_w = 1 (logvar 0): a learned constant is
    # non-identifiable against a rescaling of B (spec 7.12).
    kl_w = gaussian_kl(fwd.mu_w[:n_seeds], fwd.logvar_w[:n_seeds],
                       fwd.prior_mean_w[:n_seeds], 0.0).mean()

    objective = (recon_a + w.omega * recon_b
                 - (1.0 + w.omega) * w.alpha_z * kl_z
                 - w.alpha_w * kl_w)

    penalty, info = torch.zeros((), device=x.device), {}
    if w.alpha_a and covariances is not None:
        if v is None or p_t is None:
            raise ValueError("alpha_a > 0 needs v (the [y, PCs(Phi)] block) and p_t")
        penalty, info = covariances.penalty(fwd.mu_z[:n_seeds], v, t, p_t)
        objective = objective - w.alpha_a * penalty

    return Terms(loss=-objective,
                 recon_a=float(recon_a.detach()), recon_b=float(recon_b.detach()),
                 kl_z=float(kl_z.detach()), kl_w=float(kl_w.detach()),
                 penalty=float(penalty.detach()), penalty_info=info)


@dataclass
class AdversaryTerms:
    """Both directions of the minimax, plus the halves the spec wants logged."""

    encoder_term: torch.Tensor   # Adv_i mean: add alpha_a * this to the loss
    head_loss: torch.Tensor      # what the adversary optimiser minimises
    excess_y: float              # CE(y, ybar(t)) - CE(y, y-hat)   -- half 1
    excess_phi: float            # CE(ephi, phibar(t)) - CE(ephi, ephi-hat)


def adversary_terms(heads: Adversary, mu_z: torch.Tensor, t: torch.Tensor,
                    y: torch.Tensor, e_phi: torch.Tensor,
                    ybar_t: torch.Tensor, phibar_t: torch.Tensor
                    ) -> AdversaryTerms:
    """Spec 4.6's ``Adv_i``, in both of its roles.

    The heads are trained on ``sg mu_z`` (their optimiser owns them); the
    encoder term evaluates the same heads *with gradient flowing to mu_z only*
    -- the trainer never steps head parameters from the model loss, and zeroes
    any gradient that reached them before the head step. Baselines are the
    type-only lookups, so each half reads as excess predictive skill over
    knowing the type alone: zero at the optimum.
    """
    log_y, log_phi = heads(mu_z, t)
    base_y = soft_cross_entropy(y, ybar_t.clamp(min=1e-8).log()[t])
    base_phi = soft_cross_entropy(e_phi, phibar_t.clamp(min=1e-8).log()[t])
    excess_y = base_y - soft_cross_entropy(y, log_y)
    excess_phi = base_phi - soft_cross_entropy(e_phi, log_phi)
    encoder_term = (excess_y + excess_phi).mean()

    log_y_sg, log_phi_sg = heads(mu_z.detach(), t)
    head_loss = (soft_cross_entropy(y, log_y_sg)
                 + soft_cross_entropy(e_phi, log_phi_sg)).mean()
    return AdversaryTerms(encoder_term=encoder_term, head_loss=head_loss,
                          excess_y=float(excess_y.mean().detach()),
                          excess_phi=float(excess_phi.mean().detach()))
