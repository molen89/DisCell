"""The DisCell objective for one tile, and the adversary's two roles.

Per seed, averaged over the tile::

    J =   recon(a)                          x under the posterior w
        + omega * recon(b)                  x under a prior-drawn w
        - (1 + omega) * alpha_z * KL(q(z) || N(0, I))
        -               alpha_w * KL(q(w) || N(m_psi(c, t), I))

    loss = -J + alpha_a * Adv

Terms (a) and (b) are two bounds on the same data and each carries its own
``KL(q(z) || p(z))``, hence the ``(1 + omega)``. Reconstruction rows are scaled
by ``1 / l_i`` (inside :func:`multinomial_loglik`). The adversary term ``Adv``
is computed by :func:`adversary_terms` and added by the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from discell.model.equations import gaussian_kl, multinomial_loglik
from discell.model.networks import Adversary, Forward, soft_cross_entropy


@dataclass(frozen=True)
class Weights:
    """The weights of :func:`discell_loss`. ``kappa`` lives in the forward pass."""

    omega: float = 1.0      # weight of term (b)
    alpha_z: float = 1.0    # KL(q(z) || N(0, I)), applied (1 + omega)-fold
    alpha_w: float = 1.0    # KL(q(w) || N(m_psi(c, t), I))


@dataclass
class Terms:
    """The scalar to minimize plus its components (detached floats)."""

    loss: torch.Tensor
    recon_a: float
    recon_b: float
    kl_z: float
    kl_w: float

    def scalars(self) -> dict[str, float]:
        return {"loss": float(self.loss.detach()), "recon_a": self.recon_a,
                "recon_b": self.recon_b, "kl_z": self.kl_z, "kl_w": self.kl_w}


def discell_loss(fwd: Forward, x: torch.Tensor,
                 weights: Weights | None = None) -> Terms:
    """``-J`` for one tile. *x* holds the seeds' counts only.

    Every term is a per-seed mean, so its size does not depend on the tile.
    The prior variance of ``w`` is fixed at 1.
    """
    n_seeds = x.shape[0]
    w = weights or Weights()
    recon_a = multinomial_loglik(x, fwd.log_p).mean()
    recon_b = multinomial_loglik(x, fwd.log_p_breve).mean()
    kl_z = gaussian_kl(fwd.mu_z[:n_seeds], fwd.logvar_z[:n_seeds]).mean()
    kl_w = gaussian_kl(fwd.mu_w[:n_seeds], fwd.logvar_w[:n_seeds],
                       fwd.prior_mean_w[:n_seeds], 0.0).mean()
    objective = (recon_a + w.omega * recon_b
                 - (1.0 + w.omega) * w.alpha_z * kl_z
                 - w.alpha_w * kl_w)
    return Terms(loss=-objective,
                 recon_a=float(recon_a.detach()), recon_b=float(recon_b.detach()),
                 kl_z=float(kl_z.detach()), kl_w=float(kl_w.detach()))


@dataclass
class AdversaryTerms:
    """Both directions of the minimax, plus the excesses that are logged."""

    encoder_term: torch.Tensor   # Adv: add alpha_a * this to the model loss
    head_loss: torch.Tensor      # what the adversary optimizer minimizes
    excess_y: float              # CE(y, ybar_t) - CE(y, y_hat)
    excess_phi: float            # CE(e_Phi, phibar_t) - CE(e_Phi, e_Phi_hat)


def adversary_terms(heads: Adversary, mu_z: torch.Tensor, t: torch.Tensor,
                    y: torch.Tensor, e_phi: torch.Tensor | None,
                    ybar_t: torch.Tensor, phibar_t: torch.Tensor | None,
                    comp_weight: float = 1.0) -> AdversaryTerms:
    """The adversary's excess predictive skill over the type-only baselines.

    The encoder term evaluates the heads with the gradient flowing to *mu_z*
    (the trainer never steps the heads from the model loss); the head loss
    evaluates them on ``sg mu_z``. Both are zero-excess at the optimum.
    *comp_weight* multiplies the composition half in both roles; ``excess_y``
    is logged unweighted. With *e_phi* None (image-free) only the composition
    half is used and ``excess_phi`` is 0.
    """
    log_y, log_phi = heads(mu_z, t)
    excess_y = (soft_cross_entropy(y, ybar_t.clamp(min=1e-8).log()[t])
                - soft_cross_entropy(y, log_y))
    if e_phi is None:
        log_y_sg, _ = heads(mu_z.detach(), t)
        return AdversaryTerms(
            encoder_term=(comp_weight * excess_y).mean(),
            head_loss=(comp_weight * soft_cross_entropy(y, log_y_sg)).mean(),
            excess_y=float(excess_y.mean().detach()), excess_phi=0.0)
    excess_phi = (soft_cross_entropy(e_phi, phibar_t.clamp(min=1e-8).log()[t])
                  - soft_cross_entropy(e_phi, log_phi))
    encoder_term = (comp_weight * excess_y + excess_phi).mean()
    log_y_sg, log_phi_sg = heads(mu_z.detach(), t)
    head_loss = (comp_weight * soft_cross_entropy(y, log_y_sg)
                 + soft_cross_entropy(e_phi, log_phi_sg)).mean()
    return AdversaryTerms(encoder_term=encoder_term, head_loss=head_loss,
                          excess_y=float(excess_y.mean().detach()),
                          excess_phi=float(excess_phi.mean().detach()))
