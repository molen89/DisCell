#!/usr/bin/env python3
"""DisCell's networks and the one forward pass that wires them.

The architecture carries the identification argument, so its three load-bearing
properties are stated here and asserted in ``tests/test_model_networks.py``:

* the **decoder never sees** ``t`` -- the only route from identity to expression
  is ``z``, which is what keeps ``z`` interpretable as identity (spec 7.1);
* ``enc_z`` **never sees** ``c`` -- invariance by architecture, the penalty only
  mops up what ``x`` itself carries (spec 4.2);
* neighbour quantities enter under **stop-gradient**: ``mu_z(x_j)`` into the
  GAT, ``rho_j`` into the leak mixture (spec 4.5, the resolVI collapse).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def encode_counts(x: torch.Tensor) -> torch.Tensor:
    """The one transform applied to raw counts on the way into an encoder."""
    return torch.log1p(x)


class GATv2(nn.Module):
    """One bipartite GATv2 layer, hand-rolled: ~50 lines beats a PyG dependency.

    ``e = a^T LeakyReLU(W_dst h_dst + W_src h_src)``, softmax over each
    destination's in-edges, and the aggregation is ``sum alpha * W_src h_src``
    -- the destination enters the *attention* but never the *value*, which is
    the property the mirror-attractor argument rests on (spec 7.2). Heads are
    averaged (``concat=False``). No self-loops -- the edge list has none -- and
    no edge features (spec 7.3). A destination with no in-edges aggregates
    nothing and comes out exactly zero, which is the isolated-cell contract.
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

        # segment softmax over each destination's in-edges; empty segments are
        # never gathered, so the -inf initial max cannot produce a NaN.
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
    """Everything one tile's forward pass produces, seeds first in every tensor.

    ``log_p`` and ``log_p_breve`` are the leakage-mixed probabilities for loss
    terms (a) and (b); the ``mu/logvar`` pairs feed the KLs; ``c``, ``mu_z`` and
    ``alpha`` exist for the diagnostics (mirror check, attention entropy).
    """

    mu_z: torch.Tensor          # (n_nodes, d_z)
    logvar_z: torch.Tensor
    z: torch.Tensor
    c: torch.Tensor             # (n_context, c_dim)
    prior_mean_w: torch.Tensor  # (n_context, d_w)
    mu_w: torch.Tensor
    logvar_w: torch.Tensor
    w: torch.Tensor
    log_rho: torch.Tensor       # (n_context, G) clean composition
    rho_bar: torch.Tensor       # (n_seeds, G) foreign influx, detached
    log_p: torch.Tensor         # (n_seeds, G) mixed, posterior w   -- term (a)
    log_p_breve: torch.Tensor   # (n_seeds, G) mixed, prior-drawn w -- term (b)
    alpha: torch.Tensor


class DisCell(nn.Module):
    """The five networks of the spec, plus the tile forward pass."""

    def __init__(self, n_genes: int, n_types: int, phi_dim: int,
                 d_z: int = 20, d_w: int = 6, hidden: int = 256,
                 t_dim: int = 16, gat_dim: int = 32, heads: int = 4):
        super().__init__()
        self.n_types, self.d_z, self.d_w = n_types, d_z, d_w
        self.embed_t = nn.Embedding(n_types, t_dim)
        self.enc_z = mlp([n_genes + t_dim, hidden, hidden, 2 * d_z])
        self.gat = GATv2(src_dim=n_types + d_z, dst_dim=t_dim,
                         out_dim=gat_dim, heads=heads)
        c_dim = gat_dim + phi_dim + 1                     # +1: isolated flag
        self.prior_w = mlp([c_dim + t_dim, hidden // 4, d_w])
        self.enc_w = mlp([c_dim + t_dim + d_z + n_genes, hidden, 2 * d_w])
        # the decoder: a_g(z) + <w, B_g>. No t anywhere below this line.
        self.dec_a = mlp([d_z, hidden, n_genes])
        self.B = nn.Linear(d_w, n_genes, bias=False)

    # -- pieces ------------------------------------------------------------

    def posterior_z(self, x: torch.Tensor, t: torch.Tensor):
        out = self.enc_z(torch.cat([encode_counts(x), self.embed_t(t)], dim=-1))
        return out.chunk(2, dim=-1)

    def context(self, mu_z: torch.Tensor, t: torch.Tensor, phi: torch.Tensor,
                isolated: torch.Tensor, edge_src: torch.Tensor,
                edge_dst: torch.Tensor, n_context: int):
        """``c = GAT ⊕ Phi ⊕ flag`` for the first *n_context* nodes.

        The GAT query is ``embed(t)``, never ``z`` -- the mirror attractor.
        Sources are ``[onehot(t_j), sg mu_z(x_j)]``: the stop-gradient keeps the
        niche encoder from training the ego encoder through the neighbours.
        """
        h_src = torch.cat([F.one_hot(t, self.n_types).float(),
                           mu_z.detach()], dim=-1)
        gat, alpha = self.gat(h_src, self.embed_t(t[:n_context]),
                              edge_src, edge_dst)
        flag = isolated[:n_context].float()[:, None]
        c = torch.cat([gat, phi[:n_context], flag], dim=-1)
        return c, alpha

    def log_rho(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Clean composition: baseline programme plus environmental shift."""
        return F.log_softmax(self.dec_a(z) + self.B(w), dim=-1)

    # -- the tile forward pass ---------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor, phi: torch.Tensor,
                isolated: torch.Tensor, gat_src: torch.Tensor,
                gat_dst: torch.Tensor, leak_src: torch.Tensor,
                leak_dst: torch.Tensor, leak_beta: torch.Tensor,
                n_seeds: int, n_context: int, kappa: float) -> Forward:
        """One tile: all nodes in [seeds | ring1 | ring2] layout.

        ``z`` is encoded for every node (ring2 feeds the GAT), ``c``/``w``/
        ``rho`` for seeds and ring1 (ring1's rho feeds the leak), the losses for
        seeds only. Exact two hops, no cache.
        """
        from discell.model.equations import foreign_influx, leakage_mix

        mu_z, logvar_z = self.posterior_z(x, t)
        z = mu_z + torch.randn_like(mu_z) * (0.5 * logvar_z).exp()

        c, alpha = self.context(mu_z, t, phi, isolated, gat_src, gat_dst, n_context)

        t_c = self.embed_t(t[:n_context])
        prior_mean_w = self.prior_w(torch.cat([c, t_c], dim=-1))
        mu_w, logvar_w = self.enc_w(torch.cat(
            [c, t_c, z[:n_context], encode_counts(x[:n_context])], dim=-1)
        ).chunk(2, dim=-1)
        w = mu_w + torch.randn_like(mu_w) * (0.5 * logvar_w).exp()

        log_rho = self.log_rho(z[:n_context], w)
        # the leak mixture reads every neighbour -- ring1 or fellow seed --
        # through the stop-gradient (spec 4.5).
        rho_bar = foreign_influx(log_rho.detach().exp(), leak_src, leak_dst,
                                 leak_beta, n_dst=n_seeds)
        log_p = leakage_mix(log_rho[:n_seeds].exp(), rho_bar, kappa)

        # term (b): same seeds, w drawn from the prior instead of the posterior
        w_breve = prior_mean_w[:n_seeds] + torch.randn_like(mu_w[:n_seeds])
        log_rho_breve = self.log_rho(z[:n_seeds], w_breve)
        log_p_breve = leakage_mix(log_rho_breve.exp(), rho_bar, kappa)

        return Forward(mu_z=mu_z, logvar_z=logvar_z, z=z, c=c,
                       prior_mean_w=prior_mean_w, mu_w=mu_w, logvar_w=logvar_w,
                       w=w, log_rho=log_rho, rho_bar=rho_bar, log_p=log_p,
                       log_p_breve=log_p_breve, alpha=alpha)
