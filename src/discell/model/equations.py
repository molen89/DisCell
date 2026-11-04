"""The closed-form pieces of the DisCell objective, free of any network.

Each function is a pure function of tensors, so every equation can be tested
against an independent implementation (scipy for the multinomial,
``torch.distributions`` for the KL).
"""

from __future__ import annotations

import torch

#: Numerical floor inside logs. The probabilities are convex mixtures of
#: softmax outputs, so zero is reachable only by underflow.
EPS = 1e-8


def multinomial_loglik(x: torch.Tensor, log_p: torch.Tensor,
                       scale_by_total: bool = True) -> torch.Tensor:
    """``sum_g x_g log p_g`` per cell, without the multinomial coefficient.

    The coefficient ``log(l! / prod_g x_g!)`` does not depend on the
    parameters and is omitted. With *scale_by_total* each row is divided by its
    total count ``l_i``, so the loss weights transfer between sections of
    different depth.
    """
    row = (x * log_p).sum(dim=-1)
    if scale_by_total:
        row = row / x.sum(dim=-1).clamp(min=1.0)
    return row


def gaussian_kl_per_dim(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                        mu_p: torch.Tensor | float = 0.0,
                        logvar_p: torch.Tensor | float = 0.0) -> torch.Tensor:
    """``KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2))`` per dimension (diagonal)."""
    mu_p = torch.as_tensor(mu_p, dtype=mu_q.dtype, device=mu_q.device)
    logvar_p = torch.as_tensor(logvar_p, dtype=mu_q.dtype, device=mu_q.device)
    return 0.5 * (logvar_p - logvar_q
                  + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp()
                  - 1.0)


def gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                mu_p: torch.Tensor | float = 0.0,
                logvar_p: torch.Tensor | float = 0.0) -> torch.Tensor:
    """The diagonal Gaussian KL summed over the last dimension.

    The defaults give the standard-normal prior used for ``z``.
    """
    return gaussian_kl_per_dim(mu_q, logvar_q, mu_p, logvar_p).sum(dim=-1)


def foreign_influx(rho_src: torch.Tensor, edge_src: torch.Tensor,
                   edge_dst: torch.Tensor, beta: torch.Tensor,
                   n_dst: int) -> torch.Tensor:
    """``rho_bar_i = sum_j beta_ij rho_j``, scattered over the spillover edges.

    *rho_src* must arrive detached: the stop-gradient on neighbors is the
    caller's responsibility. Destinations with no in-edge get a zero row.
    """
    out = rho_src.new_zeros((n_dst, rho_src.shape[-1]))
    out.index_add_(0, edge_dst, rho_src[edge_src] * beta[:, None])
    return out


def spillover_mix(rho: torch.Tensor, rho_bar: torch.Tensor,
                kappa: float) -> torch.Tensor:
    """``log p_i`` for ``p_i = (1 - kappa) rho_i + kappa rho_bar_i``.

    The mixture is formed in probability space, renormalized per row and
    logged once with a floor. Connected rows already sum to one, so the
    renormalization only affects isolated cells: their ``rho_bar`` row is
    zero and their mixture becomes ``rho``. At ``kappa = 0`` the result is
    ``log rho``.
    """
    if not 0.0 <= kappa < 1.0:
        raise ValueError(f"kappa must be in [0, 1), got {kappa}")
    mix = (1.0 - kappa) * rho + kappa * rho_bar
    mix = mix / mix.sum(dim=-1, keepdim=True).clamp(min=EPS)
    return mix.clamp(min=EPS).log()
