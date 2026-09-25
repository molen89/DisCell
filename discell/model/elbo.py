#!/usr/bin/env python3
"""Assembling the DisCell objective from one tile's forward pass.

The spec's J (section 5), per seed, averaged over the batch::

    J =   recon(a)                       x under the posterior-drawn w
        + omega * recon(b)               x under the prior-drawn   w-breve
        - (1 + omega) * alpha_z * KL( q(z) || N(0, I) )
        -               alpha_w * KL( q(w) || N(m_psi(c, t), I) )
        -               alpha_a * Pen                                (optional)
        -               lambda_w * W_pen                             (optional)

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

from discell.model.equations import (TypeCovariances, gaussian_kl,
                                     gaussian_kl_per_dim, multinomial_loglik)
from discell.model.networks import Adversary, Forward, soft_cross_entropy


@dataclass(frozen=True)
class Weights:
    """The loss hyperparameters. ``kappa`` lives in the forward pass, not here."""

    omega: float = 1.0      # weight of the intrinsic path (b)
    alpha_z: float = 1.0    # KL(q(z) || N(0,I)) -- applied (1+omega)-fold
    alpha_w: float = 1.0    # KL(q(w) || p(w|c,t))
    alpha_a: float = 0.0    # invariance penalty; 0 = off
    #: optional L2 on the w channel (the translation-gauge penalty, 2026-09-21).
    #: 0 = off = every pinned run. "w" penalises E||w||^2 per seed (the sampled
    #: w, so the posterior variance is inside it); "type_mean" penalises the
    #: cell-weighted per-type batch mean of the prior field, sum_t (n_t/n)
    #: ||mean_{i in t} m_psi(c_i, t)||^2 -- which is E||m_psi||^2 minus its
    #: within-type part, i.e. "w" with the context-varying channel exempted.
    lambda_w: float = 0.0
    w_penalty: str = "none"  # "none" | "w" | "type_mean"
    #: objective ablation (i), 6b.5 2026-09-21. True = every pinned run: the
    #: z-KL is charged ``(1 + omega)``-fold, because (a) and (b) are two bounds
    #: on the same data and each carries its own ``-KL(q(z)||p(z))`` (spec 6.2).
    #: False drops the second copy -- weight ``1 * alpha_z`` -- the natural
    #: mistake the spec names, which costs the bound property.
    second_kl: bool = True
    #: free bits on the context channel (2026-09-23 pre-registration): the
    #: per-dimension KL_w is charged only above ``w_free_bits`` nats,
    #: ``sum_k max(KL_k - lambda, 0)``, with ``KL_k`` the tile mean of
    #: dimension k -- the same per-dimension quantity the dead-channel guard
    #: reads. 0 = off = every run before it, and then the term is the plain
    #: summed KL bit for bit.
    w_free_bits: float = 0.0


@dataclass
class Terms:
    """The scalar to minimise plus every component, detached, for logging."""

    loss: torch.Tensor
    recon_a: float
    recon_b: float
    kl_z: float
    kl_w: float
    #: the KL_w actually charged: equal to ``kl_w`` unless free bits are on
    kl_w_charged: float = 0.0
    penalty: float = 0.0
    w_penalty: float = 0.0
    penalty_info: dict = field(default_factory=dict)

    def scalars(self) -> dict[str, float]:
        return {"loss": float(self.loss), "recon_a": self.recon_a,
                "recon_b": self.recon_b, "kl_z": self.kl_z, "kl_w": self.kl_w,
                "kl_w_charged": self.kl_w_charged,
                "penalty": self.penalty, "w_penalty": self.w_penalty}


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
    kl_w_charged = kl_w
    if w.w_free_bits:
        per_dim = gaussian_kl_per_dim(fwd.mu_w[:n_seeds],
                                      fwd.logvar_w[:n_seeds],
                                      fwd.prior_mean_w[:n_seeds], 0.0).mean(0)
        kl_w_charged = (per_dim - w.w_free_bits).clamp(min=0.0).sum()

    kl_z_factor = (1.0 + w.omega) if w.second_kl else 1.0
    objective = (recon_a + w.omega * recon_b
                 - kl_z_factor * w.alpha_z * kl_z
                 - w.alpha_w * kl_w_charged)

    w_pen = torch.zeros((), device=x.device)
    if w.lambda_w and w.w_penalty != "none":
        w_pen = _w_penalty(fwd, t, n_seeds, w.w_penalty)
        objective = objective - w.lambda_w * w_pen

    penalty, info = torch.zeros((), device=x.device), {}
    if w.alpha_a and covariances is not None:
        if v is None or p_t is None:
            raise ValueError("alpha_a > 0 needs v (the [y, PCs(Phi)] block) and p_t")
        penalty, info = covariances.penalty(fwd.mu_z[:n_seeds], v, t, p_t)
        objective = objective - w.alpha_a * penalty

    return Terms(loss=-objective,
                 recon_a=float(recon_a.detach()), recon_b=float(recon_b.detach()),
                 kl_z=float(kl_z.detach()), kl_w=float(kl_w.detach()),
                 kl_w_charged=float(kl_w_charged.detach()),
                 penalty=float(penalty.detach()),
                 w_penalty=float(w_pen.detach()), penalty_info=info)


def _w_penalty(fwd: Forward, t: torch.Tensor, n_seeds: int,
               kind: str) -> torch.Tensor:
    """The optional L2 on the w channel (spec 7.12's translation gauge).

    ``"w"``: ``E_q ||w_i||^2`` on the sampled w, meaned over seeds. Given the
    KL to ``N(m_psi, I)`` this is ``||m_psi||^2 + 2<m_psi, mu_w - m_psi> +
    ||mu_w - m_psi||^2 + sum_k sigma_k^2`` -- only the first two terms are new,
    and at the operating point (KL_w ~ 0.002/dim) it is ``||m_psi||^2 + d_w``
    to within a fraction of a percent, so it is a penalty on the prior field.

    ``"type_mean"``: the V12 proposal, ``sum_t (n_t/n) ||mean_{i in t} m_psi||^2``.
    By the within/between decomposition this is ``E||m_psi||^2`` minus the
    within-type variance of ``m_psi``, so it charges the per-type offset (the
    unidentified translation gauge) and leaves the context-varying channel --
    the one todo 2.3 measured as the only part of w that buys likelihood --
    free of any pull toward zero.
    """
    if kind == "w":
        return (fwd.w[:n_seeds] ** 2).sum(dim=-1).mean()
    if kind != "type_mean":
        raise ValueError(f"unknown w_penalty {kind!r}")
    m = fwd.prior_mean_w[:n_seeds]
    _, inverse = torch.unique(t[:n_seeds], return_inverse=True)
    n_present = int(inverse.max()) + 1
    counts = torch.zeros(n_present, device=m.device).index_add_(
        0, inverse, torch.ones(n_seeds, device=m.device))
    sums = torch.zeros(n_present, m.shape[1], device=m.device).index_add_(
        0, inverse, m)
    means = sums / counts[:, None]
    return ((means ** 2).sum(dim=-1) * counts).sum() / n_seeds


@dataclass
class AdversaryTerms:
    """Both directions of the minimax, plus the halves the spec wants logged."""

    encoder_term: torch.Tensor   # Adv_i mean: add alpha_a * this to the loss
    head_loss: torch.Tensor      # what the adversary optimiser minimises
    excess_y: float              # CE(y, ybar(t)) - CE(y, y-hat)   -- half 1
    excess_phi: float            # CE(ephi, phibar(t)) - CE(ephi, ephi-hat)


def adversary_terms(heads: Adversary, mu_z: torch.Tensor, t: torch.Tensor,
                    y: torch.Tensor, e_phi: torch.Tensor,
                    ybar_t: torch.Tensor, phibar_t: torch.Tensor,
                    comp_weight: float = 1.0) -> AdversaryTerms:
    """Spec 4.6's ``Adv_i``, in both of its roles.

    The heads are trained on ``sg mu_z`` (their optimiser owns them); the
    encoder term evaluates the same heads *with gradient flowing to mu_z only*
    -- the trainer never steps head parameters from the model loss, and zeroes
    any gradient that reached them before the head step. Baselines are the
    type-only lookups, so each half reads as excess predictive skill over
    knowing the type alone: zero at the optimum.

    *comp_weight* (8.17, 2026-09-24) multiplies the composition half in both
    roles -- the excess in the encoder term and the CE in the head loss; 1 is
    every run before it, bit for bit. ``excess_y`` is logged unweighted.
    """
    log_y, log_phi = heads(mu_z, t)
    base_y = soft_cross_entropy(y, ybar_t.clamp(min=1e-8).log()[t])
    base_phi = soft_cross_entropy(e_phi, phibar_t.clamp(min=1e-8).log()[t])
    excess_y = base_y - soft_cross_entropy(y, log_y)
    excess_phi = base_phi - soft_cross_entropy(e_phi, log_phi)
    encoder_term = (comp_weight * excess_y + excess_phi).mean()

    log_y_sg, log_phi_sg = heads(mu_z.detach(), t)
    head_loss = (comp_weight * soft_cross_entropy(y, log_y_sg)
                 + soft_cross_entropy(e_phi, log_phi_sg)).mean()
    return AdversaryTerms(encoder_term=encoder_term, head_loss=head_loss,
                          excess_y=float(excess_y.mean().detach()),
                          excess_phi=float(excess_phi.mean().detach()))


def ensemble_adversary_terms(heads: list[Adversary], mu_z: torch.Tensor,
                             t: torch.Tensor, y: torch.Tensor,
                             e_phi: torch.Tensor, ybar_t: torch.Tensor,
                             phibar_t: torch.Tensor,
                             comp_weight: float = 1.0) -> AdversaryTerms:
    """:func:`adversary_terms` over K independent head pairs (8.17).

    The encoder is penalised on the members' mean excess, the head loss is
    their sum (one optimiser over all members steps each on its own CE), and
    the logged excesses are member means. K = 1 returns the single pair's
    terms unchanged.
    """
    members = [adversary_terms(h, mu_z, t, y, e_phi, ybar_t, phibar_t,
                               comp_weight) for h in heads]
    if len(members) == 1:
        return members[0]
    return AdversaryTerms(
        encoder_term=torch.stack([m.encoder_term for m in members]).mean(),
        head_loss=torch.stack([m.head_loss for m in members]).sum(),
        excess_y=sum(m.excess_y for m in members) / len(members),
        excess_phi=sum(m.excess_phi for m in members) / len(members))
