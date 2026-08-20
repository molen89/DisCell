#!/usr/bin/env python3
"""The closed-form pieces of the DisCell objective, free of any network.

Everything here is a pure function of tensors (plus one stateful covariance
tracker), so each equation can be tested against an independent implementation
-- scipy for the multinomial, ``torch.distributions`` for the KL, analytic
Gaussian mutual information for the invariance penalty.
"""

from __future__ import annotations

import torch

#: Numerical floor inside logs. Probabilities here are softmax outputs mixed
#: convexly, so zero is only reachable by underflow.
EPS = 1e-8


def multinomial_loglik(x: torch.Tensor, log_p: torch.Tensor,
                       scale_by_total: bool = True) -> torch.Tensor:
    """``sum_g x_g log p_g`` per cell, dropping the multinomial coefficient.

    The coefficient ``log(l! / prod x_g!)`` is constant in the parameters, so it
    is omitted -- tests account for it. With *scale_by_total* the row is divided
    by ``l_i``: raw, reconstruction is O(-200..-400) per cell against KLs of
    O(1-10) and varies severalfold between sections, so unnormalised loss
    weights would not transfer across slides (spec section 5).
    """
    row = (x * log_p).sum(dim=-1)
    if scale_by_total:
        row = row / x.sum(dim=-1).clamp(min=1.0)
    return row


def gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                mu_p: torch.Tensor | float = 0.0,
                logvar_p: torch.Tensor | float = 0.0) -> torch.Tensor:
    """``KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) )``, diagonal, summed over dims.

    The spec's form: ``sum_k [ log(s/sigma) + (sigma^2 + (mu-m)^2)/(2 s^2) - 1/2 ]``.
    Defaults give the standard-normal prior used for ``z``.
    """
    mu_p = torch.as_tensor(mu_p, dtype=mu_q.dtype, device=mu_q.device)
    logvar_p = torch.as_tensor(logvar_p, dtype=mu_q.dtype, device=mu_q.device)
    per_dim = 0.5 * (logvar_p - logvar_q
                     + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp()
                     - 1.0)
    return per_dim.sum(dim=-1)


def foreign_influx(rho_src: torch.Tensor, edge_src: torch.Tensor,
                   edge_dst: torch.Tensor, beta: torch.Tensor,
                   n_dst: int) -> torch.Tensor:
    """``rho_bar_i = sum_j beta_ij rho_j`` by scatter over the leak edges.

    *rho_src* must arrive **detached** -- the caller owns the stop-gradient,
    because whether a row is a frozen neighbour is batch structure, not an
    equation. Rows of cells with no in-edges come back all-zero, matching the
    zero beta row they carry.
    """
    out = rho_src.new_zeros((n_dst, rho_src.shape[-1]))
    out.index_add_(0, edge_dst, rho_src[edge_src] * beta[:, None])
    return out


def leakage_mix(rho: torch.Tensor, rho_bar: torch.Tensor,
                kappa: float) -> torch.Tensor:
    """``log p_i`` for ``p_i = (1-kappa) rho_i + kappa rho_bar_i``.

    Mixed in probability space -- the mixture of two simplex points is on the
    simplex -- then logged once, with a floor for underflow. At ``kappa = 0``
    this is exactly ``log rho``, the nested no-correction null.

    The row-normalisation is for **isolated cells**: their foreign row is zero,
    so the raw mixture would sum to ``1 - kappa`` and charge them a constant
    ``-log(1-kappa)`` for leakage they cannot receive. Renormalised, their
    mixture degenerates to ``rho`` -- effectively ``kappa_i = 0``, the honest
    model for a cell with no neighbours. Connected rows already sum to one, so
    for them it is a no-op.
    """
    if not 0.0 <= kappa < 1.0:
        raise ValueError(f"kappa must be in [0, 1), got {kappa}")
    mix = (1.0 - kappa) * rho + kappa * rho_bar
    mix = mix / mix.sum(dim=-1, keepdim=True).clamp(min=EPS)
    return mix.clamp(min=EPS).log()


def shrink_to_diagonal(cov: torch.Tensor, shrink: float) -> torch.Tensor:
    """``(1-s) Sigma + s diag(Sigma)``: damp correlations, keep every variance.

    Deliberately not ``+ lambda I``: an additive ridge inflates all variances
    and biases the log-det upward by most exactly where the estimate is worst
    -- the rare types -- making the penalty systematically weakest where there
    is least data. Shrinking toward the diagonal leaves the marginals intact
    and only pulls the estimated correlations toward zero.
    """
    diag = torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
    return (1.0 - shrink) * cov + shrink * diag


class TypeCovariances:
    """EMA joint covariance of ``u = [z, v]`` per cell type, with gradients.

    The penalty needs ``Sigma_{[z,v]|t}`` for every type, but a batch holds too
    few cells of the rare ones for a stable estimate. So the covariance used is
    ``(1-ema) * stored.detach() + ema * batch``: the history conditions the
    estimate, the batch term carries the gradient into the encoder. Types whose
    effective sample (``batch count / ema``) is below *min_count* are excluded
    from the penalty -- and must be logged, because those cells go
    unregularised.
    """

    def __init__(self, n_types: int, dim_z: int, dim_v: int,
                 ema: float = 0.01, shrink: float = 0.05,
                 min_count: float | None = None) -> None:
        self.dim_z, self.dim_v = dim_z, dim_v
        dim = dim_z + dim_v
        self.ema, self.shrink = float(ema), float(shrink)
        #: floor on the effective per-type sample; default 10x the joint dim.
        self.min_count = float(min_count if min_count is not None else 10 * dim)
        self.mean = torch.zeros(n_types, dim)
        self.second = torch.eye(dim).expand(n_types, dim, dim).clone()
        self.batch_count = torch.zeros(n_types)
        self.seen = torch.zeros(n_types, dtype=torch.bool)

    def to(self, device) -> "TypeCovariances":
        for name in ("mean", "second", "batch_count", "seen"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def effective_count(self) -> torch.Tensor:
        """Roughly how many cells each type's EMA covariance rests on."""
        return self.batch_count / self.ema

    def penalty(self, z: torch.Tensor, v: torch.Tensor, t: torch.Tensor,
                p_t: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """``sum_t p(t) * 1/2 [logdet S_v + logdet S_z - logdet S_joint]``.

        The Gaussian estimate of ``I(z; v | t)``: zero when the two blocks are
        uncorrelated within every type, positive otherwise. Also advances the
        EMA. Returns ``(penalty, info)`` where *info* carries the per-type
        values and the fraction of cells excluded by the support floor.
        """
        u = torch.cat([z, v], dim=-1)
        dim = u.shape[-1]
        total = u.new_zeros(())
        per_type, excluded = {}, 0
        for k in t.unique().tolist():
            rows = u[t == k]
            batch_mean = rows.mean(dim=0)
            batch_second = rows.T @ rows / rows.shape[0]

            mixed_mean = (1 - self.ema) * self.mean[k].detach() + self.ema * batch_mean
            mixed_second = ((1 - self.ema) * self.second[k].detach()
                            + self.ema * batch_second)
            if not self.seen[k]:      # first sight: adopt the batch outright
                mixed_mean, mixed_second = batch_mean, batch_second
            self.mean[k] = mixed_mean.detach()
            self.second[k] = mixed_second.detach()
            self.batch_count[k] = ((1 - self.ema) * self.batch_count[k]
                                   + self.ema * rows.shape[0])
            self.seen[k] = True

            if self.effective_count()[k] < self.min_count:
                excluded += rows.shape[0]
                continue

            cov = mixed_second - mixed_mean[:, None] * mixed_mean[None, :]
            # Work on the correlation form: MI is invariant to per-dimension
            # scaling (the log-std terms cancel exactly between the blocks and
            # the joint), and correlation matrices keep the log-det gradient --
            # an inverse covariance -- bounded even when a dimension's variance
            # is tiny, e.g. the near-constant y of cells deep inside one niche.
            std = torch.diagonal(cov).clamp(min=1e-12).sqrt()
            corr = cov / (std[:, None] * std[None, :])
            corr = shrink_to_diagonal(corr, self.shrink)
            corr = corr + 1e-5 * torch.eye(dim, device=corr.device)
            ld = lambda m: torch.linalg.slogdet(m)[1]  # noqa: E731
            value = 0.5 * (ld(corr[self.dim_z:, self.dim_z:])
                           + ld(corr[:self.dim_z, :self.dim_z]) - ld(corr))
            total = total + p_t[k] * value
            per_type[int(k)] = float(value)
        info = {"per_type": per_type,
                "excluded_fraction": excluded / max(len(t), 1),
                "dim": dim}
        return total, info
