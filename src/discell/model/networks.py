"""DisCell's networks and the tile forward pass that wires them.

Three properties of the architecture carry the identification argument and are
asserted in ``tests/test_model_networks.py``:

* the decoder never sees the type ``t``: the only route from identity to
  expression is ``z``;
* the identity encoder ``enc_z`` never sees the context ``c``;
* neighbor quantities enter under stop-gradient: ``rho_j`` into the spillover
  mixture (and ``mu_z`` is never a GAT source; the sources are one-hot types).

Parameter names match the ``state_dict`` of the paper's reference fits, so their
checkpoints load unchanged. ``phi_dim = 0`` is the image-free model: the context is then the
GAT output plus the isolated-cell flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from discell.model.equations import foreign_influx, spillover_mix


def mlp(dims: list[int]) -> nn.Sequential:
    """``Linear -> ReLU -> ... -> Linear`` through *dims*."""
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def encode_counts(x: torch.Tensor, median_counts: float) -> torch.Tensor:
    """Counts as the encoders see them: ``[log1p(x / l * median(l)), log l]``."""
    totals = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return torch.cat([torch.log1p(x / totals * median_counts), totals.log()], dim=-1)


def soft_cross_entropy(target: torch.Tensor, log_pred: torch.Tensor) -> torch.Tensor:
    """``-sum_k target_k log pred_k`` per row, for distribution targets."""
    return -(target * log_pred).sum(dim=-1)


class GATv2(nn.Module):
    """One bipartite GATv2 layer (heads averaged, no self-loops, no edge features).

    ``e = a^T LeakyReLU(W_dst h_dst + W_src h_src)``, softmax over each
    destination's in-edges, aggregation ``sum alpha W_src h_src``. The
    destination shapes the attention but never the value. A destination with
    no in-edge aggregates nothing and is exactly zero.
    """

    def __init__(self, src_dim: int, dst_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.w_src = nn.Linear(src_dim, heads * out_dim, bias=False)
        self.w_dst = nn.Linear(dst_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.attn)

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(context (n_dst, out_dim), alpha (n_edges, heads))``."""
        n_dst = h_dst.shape[0]
        source = self.w_src(h_src).view(-1, self.heads, self.out_dim)
        dest = self.w_dst(h_dst).view(-1, self.heads, self.out_dim)

        gate = F.leaky_relu(source[edge_src] + dest[edge_dst], 0.2)
        logits = (gate * self.attn).sum(-1)                       # (E, H)

        # segment softmax; empty segments are never gathered, so the -inf
        # initial maximum cannot produce a NaN
        peak = logits.new_full((n_dst, self.heads), -torch.inf)
        peak = peak.scatter_reduce(0, edge_dst[:, None].expand_as(logits),
                                   logits, "amax", include_self=True)
        weight = (logits - peak[edge_dst]).exp()
        norm = logits.new_zeros(n_dst, self.heads).index_add_(0, edge_dst, weight)
        alpha = weight / norm[edge_dst].clamp(min=1e-30)

        out = source.new_zeros(n_dst, self.heads, self.out_dim)
        out.index_add_(0, edge_dst, alpha[..., None] * source[edge_src])
        return out.mean(dim=1), alpha


@dataclass
class Forward:
    """One tile's forward pass. Rows are in tile order, seeds first.

    ``z``-, ``c``- and ``w``-shaped tensors cover the context nodes (seeds and
    ring 1); ``rho_bar``, ``log_p`` and ``log_p_breve`` cover the seeds.
    """

    mu_z: torch.Tensor           # (n_context, d_z)
    logvar_z: torch.Tensor
    z: torch.Tensor
    c: torch.Tensor              # (n_context, c_dim)
    prior_mean_w: torch.Tensor   # (n_context, d_w), m_psi(c, t)
    mu_w: torch.Tensor
    logvar_w: torch.Tensor
    w: torch.Tensor
    log_rho: torch.Tensor        # (n_context, G), clean profile
    rho_bar: torch.Tensor        # (n_seeds, G), foreign influx, detached
    log_p: torch.Tensor          # (n_seeds, G), mixture under the posterior w
    log_p_breve: torch.Tensor    # (n_seeds, G), mixture under a prior-drawn w
    alpha: torch.Tensor          # (n_edges, heads), GAT attention


class Adversary(nn.Module):
    """The adversary heads ``(z, t) -> y`` and ``(z, t) -> e_Phi``.

    Trained by their own optimizer on ``sg mu_z`` to predict the niche
    composition ``y`` and the image cluster memberships ``e_Phi``; the encoder
    is trained against the frozen heads. ``e_phi = 0`` (image-free) builds no
    image head.
    """

    def __init__(self, d_z: int, n_types: int, e_phi: int, hidden: int = 64):
        super().__init__()
        self.n_types = n_types
        self.head_y = mlp([d_z + n_types, hidden, hidden, n_types])
        self.head_phi = mlp([d_z + n_types, hidden, hidden, e_phi]) if e_phi else None

    def forward(self, z: torch.Tensor, t: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(log y_hat, log e_Phi_hat or None)``."""
        joined = torch.cat([z, F.one_hot(t, self.n_types).float()], dim=-1)
        log_y = F.log_softmax(self.head_y(joined), dim=-1)
        if self.head_phi is None:
            return log_y, None
        return log_y, F.log_softmax(self.head_phi(joined), dim=-1)


class DisCell(nn.Module):
    """The encoders, the GAT context, the prior, the decoder and the tile pass.

    Parameters
    ----------
    n_genes, n_types
        Panel width ``G`` and number of cell types ``K``.
    phi_dim
        Width of the per-cell image descriptor; 0 for the image-free model.
    median_counts
        Median total count of the section, the encoders' library scale.
    d_z, d_w, hidden, gat_dim, heads
        Latent sizes and network widths.
    """

    def __init__(self, n_genes: int, n_types: int, phi_dim: int,
                 median_counts: float, d_z: int = 20, d_w: int = 6,
                 hidden: int = 256, gat_dim: int = 32, heads: int = 4):
        super().__init__()
        self.n_types, self.d_z, self.d_w = n_types, d_z, d_w
        self.phi_dim = phi_dim
        self.median_counts = float(median_counts)
        # Construction order fixes the initialization draws; it matches the
        # reference fits' code so a seed gives the same initial weights.
        self.embed_t = nn.Embedding(n_types, n_types)      # the GAT query
        self.enc_z = mlp([n_genes + 1 + n_types, hidden, hidden, 2 * d_z])
        self.gat = GATv2(src_dim=n_types, dst_dim=n_types, out_dim=gat_dim, heads=heads)
        c_dim = gat_dim + phi_dim + 1                       # +1: isolated flag
        self.prior_w = mlp([c_dim + n_types, hidden // 4, d_w])
        self.enc_w = mlp([c_dim + n_types + d_z + n_genes + 1, hidden, 2 * d_w])
        # the decoder a_g(z) + <w, B_g>; no t below this line
        self.dec_a = mlp([d_z, hidden, n_genes])
        self.B = nn.Linear(d_w, n_genes, bias=False)

    @property
    def image_free(self) -> bool:
        """True when the context carries no image descriptor."""
        return self.phi_dim == 0

    def posterior_z(self, x: torch.Tensor, t: torch.Tensor):
        """``(mu_z, logvar_z)`` from the counts and the type alone."""
        out = self.enc_z(torch.cat([encode_counts(x, self.median_counts),
                                    F.one_hot(t, self.n_types).float()], dim=-1))
        return out.chunk(2, dim=-1)

    def context(self, t: torch.Tensor, phi: torch.Tensor, isolated: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor, n_context: int):
        """``(c, alpha)`` with ``c = GAT(onehot(t_j); query embed(t_i)) + Phi_i + flag_i``
        for the first *n_context* nodes."""
        h_src = F.one_hot(t, self.n_types).float()
        h_dst = self.embed_t(t[:n_context])
        gat, alpha = self.gat(h_src, h_dst, edge_src, edge_dst)
        flag = isolated[:n_context].float()[:, None]
        return torch.cat([gat, phi[:n_context], flag], dim=-1), alpha

    def log_rho(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Clean profile ``log softmax(a(z) + B w)``."""
        return F.log_softmax(self.dec_a(z) + self.B(w), dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, phi: torch.Tensor,
                isolated: torch.Tensor, gat_src: torch.Tensor,
                gat_dst: torch.Tensor, spillover_src: torch.Tensor,
                spillover_dst: torch.Tensor, spillover_beta: torch.Tensor,
                n_seeds: int, n_context: int, kappa: float,
                sample: bool = True) -> Forward:
        """One tile in ``[seeds | ring1 | ring2]`` layout.

        ``z``, ``c``, ``w`` and ``rho`` are computed for seeds and ring 1 (ring
        1's ``rho`` feeds the seeds' influx); ring 2 contributes only its types
        as GAT sources. *x* needs rows for the context nodes only. With
        ``sample=False`` the posterior and prior means are used throughout.
        """
        n_nodes = t.shape[0]
        mu_z, logvar_z = self.posterior_z(x[:n_context], t[:n_context])
        # the draw is sized by the node count so the RNG stream matches the
        # reference fits' code
        noise = (torch.randn(n_nodes, self.d_z, dtype=mu_z.dtype,
                             device=mu_z.device)[:n_context] if sample
                 else torch.zeros_like(mu_z))
        z = mu_z + noise * (0.5 * logvar_z).exp()

        c, alpha = self.context(t, phi, isolated, gat_src, gat_dst, n_context)

        t_c = F.one_hot(t[:n_context], self.n_types).float()
        prior_mean_w = self.prior_w(torch.cat([c, t_c], dim=-1))
        mu_w, logvar_w = self.enc_w(torch.cat(
            [c, t_c, z[:n_context], encode_counts(x[:n_context], self.median_counts)],
            dim=-1)).chunk(2, dim=-1)
        w_noise = torch.randn_like(mu_w) if sample else torch.zeros_like(mu_w)
        w = mu_w + w_noise * (0.5 * logvar_w).exp()

        log_rho = self.log_rho(z[:n_context], w)
        rho_bar = foreign_influx(log_rho.detach().exp(), spillover_src, spillover_dst,
                                 spillover_beta, n_dst=n_seeds)
        log_p = spillover_mix(log_rho[:n_seeds].exp(), rho_bar, kappa)

        # term (b): the same seeds with w drawn from the prior N(m_psi, I)
        w_breve = prior_mean_w[:n_seeds] + (
            torch.randn_like(mu_w[:n_seeds]) if sample
            else torch.zeros_like(mu_w[:n_seeds]))
        log_p_breve = spillover_mix(self.log_rho(z[:n_seeds], w_breve).exp(),
                                  rho_bar, kappa)

        return Forward(mu_z=mu_z, logvar_z=logvar_z, z=z, c=c,
                       prior_mean_w=prior_mean_w, mu_w=mu_w, logvar_w=logvar_w,
                       w=w, log_rho=log_rho, rho_bar=rho_bar, log_p=log_p,
                       log_p_breve=log_p_breve, alpha=alpha)
