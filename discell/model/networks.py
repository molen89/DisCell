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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: The form of the leak coefficient (review R12 Test 2, devlog 2026-09-24).
#: "global": one kappa for every cell and gene -- every run before it, bit for
#: bit. "depth": kappa_i = kappa * clip(sum_j beta_ij l_j / l_i, 0, c_max), the
#: leaked amount set by the donors' depth, not only the receiver's; c_max =
#: KAPPA_DEPTH_MAX / kappa (5 at kappa 0.1), so kappa_i <= 0.5. "gene":
#: kappa_g = min(kappa * s_g / mean_g(s_g), KAPPA_GENE_MAX), s_g the gene's
#: measured extranuclear transcript share; the mixture is renormalised over
#: genes. "density" (author, 2026-09-24): "depth" with each count total over
#: its cell's area, r_i = sum_j beta_ij (l_j/A_j) / (l_i/A_i) -- donor density,
#: not donor size. Amendment (coordinator, 2026-09-24, before any fit): in
#: both "depth" and "density" the ratio is divided by its mean over the
#: connected training cells, kappa_i = kappa * clip(r_i / mean_train(r), 0,
#: c_max), so the mean leak fraction is kappa (up to the clip) and the arms
#: move where leakage lands, not how much there is in total -- as "gene",
#: whose kappa_g has mean kappa by construction. Amendment 2: the normaliser
#: is the clip-aware m (solve_ratio_normaliser), so the mean is kappa after
#: the clip too. In every form a cell with no
#: leak edge decodes to rho exactly.
KAPPA_MODES = ("global", "depth", "gene", "density")
KAPPA_DEPTH_MAX = 0.5
KAPPA_GENE_MAX = 0.9


def read_gene_share(path, gene_names=None) -> np.ndarray:
    """``s_g`` from *path* (``gene_extranuclear_share.npy``), in decoder order.

    The file is aligned to the bundle's ``var_names``; when its JSON sidecar
    carries the gene order, that order must equal *gene_names*. Refuses a
    length mismatch and a non-finite or non-positive mean.
    """
    import json
    from pathlib import Path

    path = Path(path)
    share = np.load(path).astype(np.float64)
    sidecar = path.with_suffix(".json")
    if gene_names is not None:
        names = np.asarray([str(g) for g in gene_names])
        if len(share) != len(names):
            raise ValueError(f"{path}: {len(share)} genes, model has {len(names)}")
        if sidecar.exists():
            stored = json.loads(sidecar.read_text()).get("gene_names")
            if stored is not None and list(stored) != names.tolist():
                raise ValueError(f"{path}: gene order differs from the model's")
    if not np.isfinite(share).all() or share.mean() <= 0:
        raise ValueError(f"{path}: s_g must be finite with a positive mean")
    return share


def density_areas(cell_area, nucleus_area, method, t, n_types: int
                  ) -> tuple[np.ndarray, dict]:
    """``A_i`` for the "density" form, and the per-type record of how.

    ``A_i`` is the segmented cell area, except for cells segmented by nucleus
    expansion: their cell area is a fixed dilation and says nothing about
    the cell, so there ``A_i`` = nuclear area x the median cell/nucleus
    area ratio of the non-expansion cells of the same type (all
    non-expansion cells when a type has none with a nucleus). An expansion
    cell without a nuclear area keeps its cell area (counted).
    """
    cell_area = np.asarray(cell_area, dtype=np.float64)
    nucleus_area = np.asarray(nucleus_area, dtype=np.float64)
    expansion = np.array(["expansion" in str(m).lower() for m in method])
    t = np.asarray(t)
    usable = (~expansion & np.isfinite(nucleus_area) & (nucleus_area > 0)
              & np.isfinite(cell_area) & (cell_area > 0))
    ratio = np.where(usable, cell_area / np.where(usable, nucleus_area, 1.0),
                     np.nan)
    pooled = float(np.nanmedian(ratio[usable]))
    area = cell_area.copy()
    by_type = {}
    for g in range(n_types):
        members = t == g
        own = usable & members
        r = float(np.median(ratio[own])) if own.any() else pooled
        fix = expansion & members & np.isfinite(nucleus_area) & (nucleus_area > 0)
        area[fix] = nucleus_area[fix] * r
        by_type[int(g)] = {
            "n_cells": int(members.sum()),
            "expansion_fraction": float(expansion[members].mean())
            if members.any() else 0.0,
            "median_cell_over_nucleus": r,
            "ratio_from": "type" if own.any() else "pooled",
            "expansion_without_nucleus": int((expansion & members & ~fix).sum())}
    if not (np.isfinite(area) & (area > 0)).all():
        raise ValueError("cell areas must be finite and positive")
    report = {"rule": "A = xenium_cell_area; nucleus-expansion cells: "
                      "xenium_nucleus_area x type-median cell/nucleus ratio "
                      "of non-expansion cells",
              "expansion_fraction": float(expansion.mean()),
              "pooled_median_cell_over_nucleus": pooled, "by_type": by_type}
    return area, report


def read_density_areas(dataset: str, variant: str, t, n_types: int,
                       type_names=None) -> tuple[np.ndarray, dict]:
    """:func:`density_areas` from the bundle's obs, in the model's cell order
    (the bundle's row order, which ``assemble`` keeps)."""
    import anndata as ad

    from discell import paths

    obs = ad.read_h5ad(paths.dataset(dataset).bundle(variant), backed="r").obs
    if len(obs) != len(t):
        raise ValueError(f"bundle has {len(obs)} cells, the model {len(t)}")
    method = (obs["xenium_segmentation_method"].astype(str).to_numpy()
              if "xenium_segmentation_method" in obs else np.full(len(obs), ""))
    area, report = density_areas(obs["xenium_cell_area"].to_numpy(),
                                 obs["xenium_nucleus_area"].to_numpy(),
                                 method, t, n_types)
    if type_names is not None:
        for g, entry in report["by_type"].items():
            entry["type"] = str(type_names[g])
    return area, report


def leak_ratio(in_edges, totals, area=None) -> tuple[np.ndarray, np.ndarray]:
    """``(r, has_edge)`` over the whole slide, the per-batch ratio of
    :meth:`DisCell.leak_kappa` computed once: ``r_i = sum_j beta_ij l_j /
    max(l_i, 1)``, each total over its cell's area when *area* is given.

    *in_edges* is ``ModelGraph.in_edges`` (rows = receiver, data = beta), the
    same weights a tile carries as ``leak_beta``; a seed's leak edges in a
    tile are all of its in-edges, so this is the ratio the forward pass sees.
    """
    totals = np.asarray(totals, dtype=np.float64)
    own = np.maximum(totals, 1.0)
    if area is not None:
        area = np.asarray(area, dtype=np.float64)
        totals, own = totals / area, own / area
    donor = np.asarray(in_edges @ totals).ravel()
    has_edge = np.asarray(in_edges.sum(axis=1)).ravel() > 0
    return donor / own, has_edge


def solve_ratio_normaliser(r: np.ndarray, kappa: float,
                           tol: float = 1e-10) -> float:
    """The scalar ``m`` with ``mean(kappa * clip(r / m, 0, c_max)) = kappa``,
    c_max = KAPPA_DEPTH_MAX / kappa (amendment 2, 2026-09-24): the clip-aware
    normaliser, by bisection.

    The post-clip mean ``min(kappa r / m, KAPPA_DEPTH_MAX)`` falls
    monotonically in m. At ``m = mean(r)`` it is at most kappa (the clip only
    removes), so the root lies at or below ``mean(r)``; as m -> 0 it tends to
    ``KAPPA_DEPTH_MAX`` x the share of r > 0, which must exceed kappa for a
    root to exist. At kappa 0 every kappa_i is 0 and ``mean(r)`` is returned.
    """
    r = np.asarray(r, dtype=np.float64)
    hi = float(r.mean())
    if kappa <= 0 or hi <= 0:
        return hi

    def excess(m):
        return float(np.minimum(kappa * r / m, KAPPA_DEPTH_MAX).mean()) - kappa

    if excess(hi) >= 0:
        return hi                                   # the clip never binds
    lo = hi
    for _ in range(200):
        lo /= 2.0
        if excess(lo) > 0:
            break
    else:
        raise ValueError("no normaliser reaches a mean kappa_i of kappa: too "
                         f"few cells with r > 0 for the {KAPPA_DEPTH_MAX} cap")
    for _ in range(200):                           # excess(lo) > 0 > excess(hi)
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * hi:
            break
    return 0.5 * (lo + hi)


def kappa_ratio_stats(in_edges, totals, train_rows, kappa: float,
                      area=None) -> dict:
    """The normaliser over the connected training cells and what it does.

    ``normaliser`` is the clip-aware m of :func:`solve_ratio_normaliser`
    (amendment 2), so the post-clip mean kappa_i over those cells is kappa;
    ``ratio_mean_train`` is the plain mean of r (amendment 1's normaliser),
    kept for the record.
    """
    r, has_edge = leak_ratio(in_edges, totals, area)
    rows = np.asarray(train_rows)
    rows = rows[has_edge[rows]]
    m = solve_ratio_normaliser(r[rows], kappa)
    k_i = np.minimum(kappa * r[rows] / m, KAPPA_DEPTH_MAX)
    return {"normaliser": m,
            "ratio_mean_train": float(r[rows].mean()),
            "n_train_connected": int(len(rows)),
            "ratio_quantiles_train": {str(q): float(np.percentile(r[rows], q))
                                      for q in (5, 25, 50, 75, 95)},
            "kappa": float(kappa), "c_max": (KAPPA_DEPTH_MAX / kappa
                                             if kappa > 0 else None),
            "kappa_i_mean_train": float(k_i.mean()),
            "kappa_i_cap_share_train": float((k_i >= KAPPA_DEPTH_MAX).mean())}


def mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def encode_counts(x: torch.Tensor, median_counts: float,
                  log_depth: torch.Tensor | None = None) -> torch.Tensor:
    """Counts as an encoder sees them: ``[log1p(x/l * median(l)), log l]``.

    *log_depth* overrides the depth scalar: the leak-subtracted view ``x~``
    is normalised by its own sum (~(1-kappa) l) but reports the cell's raw
    ``l`` (architect condition 3, 2026-09-14).

    Library-normalised before ``log1p`` -- raw ``log1p(x)`` entangles depth with
    composition, and since ``l`` is conditioned on rather than modelled, ``z``
    should not spend capacity on it. Depth is still weakly state-informative,
    so ``log l`` rides along as one scalar instead of being discarded
    (spec 4.2).
    """
    totals = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
    depth = totals.log() if log_depth is None else log_depth
    return torch.cat([torch.log1p(x / totals * median_counts), depth], dim=-1)


class GATv2(nn.Module):
    """One bipartite GATv2 layer, hand-rolled: ~50 lines beats a PyG dependency.

    ``e = a^T LeakyReLU(W_dst h_dst + W_src h_src)``, softmax over each
    destination's in-edges, and the aggregation is ``sum alpha * W_src h_src``
    -- the destination enters the *attention* but never the *value*, which is
    the property the mirror-attractor argument rests on (spec 7.2). Heads are
    averaged (``concat=False``). No self-loops -- the edge list has none -- and
    no edge features (spec 7.3). A destination with no in-edges aggregates
    nothing and comes out exactly zero, which is the isolated-cell contract.

    ``sink=True`` adds a fixed null logit 0 to every destination's softmax
    (an attention sink): the weights then sum to < 1 and the aggregate grows
    with the number of neighbours, ``n e^e / (1 + n e^e)`` for n identical
    ones -- a saturating dose whose half-point ``e^-e`` is learned per type
    pair. Plain softmax sees fractions only (devlog 2026-09-16, neighbour
    dose). The isolated contract is unchanged: all mass on the sink, zero out.
    """

    def __init__(self, src_dim: int, dst_dim: int, out_dim: int, heads: int = 4,
                 sink: bool = False):
        super().__init__()
        self.heads, self.out_dim, self.sink = heads, out_dim, sink
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
        peak = logits.new_full((n_dst, self.heads), 0.0 if self.sink else -torch.inf)
        peak = peak.scatter_reduce(0, edge_dst[:, None].expand_as(logits),
                                   logits, "amax", include_self=True)
        weight = (logits - peak[edge_dst]).exp()
        norm = logits.new_zeros(n_dst, self.heads).index_add_(0, edge_dst, weight)
        if self.sink:
            norm = norm + (-peak).exp()          # the null edge, logit 0
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

    mu_z: torch.Tensor          # (n_encoded, d_z); n_encoded is
    #: n_nodes under type_z, n_context (ring 2 skipped) under type_only
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
    #: the leak coefficient the mixture used: the float kappa under the
    #: "global" form, else (n_seeds, 1) per cell ("depth") or (n_seeds, G)
    #: per gene, zero rows for cells without a leak edge ("gene")
    kappa_eff: float | torch.Tensor | None = None
    #: the false-positive floor's eta_i per seed, (n_seeds, 1); None = off
    #: (todo 8.15b, discell.model.fp_floor)
    eta: torch.Tensor | None = None


class Adversary(nn.Module):
    """The escalation heads of spec 4.6: ``(z, t) -> y-hat`` and ``-> e_Phi-hat``.

    Deliberately nonlinear -- the closed-form penalty already removes the
    linear (Gaussian) dependence, and the escalation trigger is precisely the
    *nonlinear* excess an MLP probe still finds. Trained by a separate
    optimiser to predict the niche from ``sg mu_z``; the encoder is trained
    against frozen heads to defeat them. Zero excess skill over the type-only
    baselines at the optimum.
    """

    def __init__(self, d_z: int, n_types: int, e_phi: int, hidden: int = 64):
        super().__init__()
        self.n_types = n_types
        self.head_y = mlp([d_z + n_types, hidden, hidden, n_types])
        self.head_phi = mlp([d_z + n_types, hidden, hidden, e_phi])

    def forward(self, z: torch.Tensor, t: torch.Tensor):
        joined = torch.cat([z, F.one_hot(t, self.n_types).float()], dim=-1)
        return (F.log_softmax(self.head_y(joined), dim=-1),
                F.log_softmax(self.head_phi(joined), dim=-1))


def soft_cross_entropy(target: torch.Tensor, log_pred: torch.Tensor) -> torch.Tensor:
    """``-sum_k target_k log pred_k`` per row, for distribution targets."""
    return -(target * log_pred).sum(dim=-1)


class ClassMeanPrior(nn.Module):
    """DisCoVR's prior, standing in for ``m_psi(c, t)`` -- ablation (iii), 6b.5.

    ``p(w | t) = N(mu_t, I)`` with ``mu_t`` a learned per-type vector, so the
    prior no longer reads the context ``c`` at all: DisCoVR's class-mean prior
    ``mu_k = E[z | y = k]``, here on the w channel and learned rather than
    accumulated. The posterior ``q(w | c, t, z, x)`` is untouched, so the KL
    still pulls a context-dependent posterior toward a context-free target.

    It is a drop-in for the ``prior_w`` MLP -- same call signature, the
    ``[c, onehot(t)]`` concatenation -- so ``transport.py``, ``degeneracy.py``
    and ``neighbour_dose.py`` keep working unchanged; the ``c`` block is simply
    ignored and the one-hot tail is read back as the type index.
    """

    def __init__(self, n_types: int, d_w: int):
        super().__init__()
        self.n_types = n_types
        self.mu_t = nn.Embedding(n_types, d_w)

    def forward(self, joined: torch.Tensor) -> torch.Tensor:
        t = joined[:, -self.n_types:].argmax(dim=-1)
        return self.mu_t(t)


class TypeFreePrior(nn.Module):
    """``m_psi(c)`` -- the type-free prior, arm (iii) of 2026-09-23.

    The prior reads the environment only: ``p(w | c) = N(m_psi(c), I)``, so a
    neighbourhood proposes the same ``w`` whatever the centre cell is. The
    posterior ``q(w | c, t, z, x)`` is untouched -- only the target moves.

    Like ``ClassMeanPrior`` it is a drop-in for the ``prior_w`` MLP: the call
    sites in ``transport.py`` and ``degeneracy.py`` pass ``[c, onehot(t)]``
    and the one-hot tail is simply dropped here.
    """

    def __init__(self, n_types: int, net: nn.Module):
        super().__init__()
        self.n_types = n_types
        self.net = net

    def forward(self, joined: torch.Tensor) -> torch.Tensor:
        return self.net(joined[:, : -self.n_types])


class DisCell(nn.Module):
    """The five networks of the spec, plus the tile forward pass."""

    def __init__(self, n_genes: int, n_types: int, phi_dim: int,
                 median_counts: float, d_z: int = 20, d_w: int = 6,
                 hidden: int = 256, gat_dim: int = 32, heads: int = 4,
                 gat_sources: str = "type_only", subtract_leak: bool = False,
                 gat_sink: bool = False, class_mean_prior: bool = False,
                 query: str = "type", prior_type_free: bool = False,
                 kappa_mode: str = "global",
                 kappa_gene_share: np.ndarray | None = None,
                 kappa_ratio_mean: float | None = None,
                 phi_proj: int = 0):
        super().__init__()
        self.n_types, self.d_z, self.d_w = n_types, d_z, d_w
        self.median_counts = float(median_counts)
        self.gat_sources = gat_sources
        # spec 7.13 (2026-09-14): encoders read the leak-subtracted view
        # x~ = x - kappa*l*rho_bar in a second pass over the seeds
        self.subtract_leak = subtract_leak
        # The type embedding exists ONLY as the GAT query: it stands in for the
        # z_i the centre cell deliberately does not contribute to its own
        # context, so it is sized to match the source features it is queried
        # against -- onehot(t_j) alone under the default "type_only" (ratified
        # 2026-09-12), or [onehot(t_j), sg z_j] (K + d_z) under the original
        # spec-4.1 "type_z". Everywhere t is a plain input (enc_z, enc_w,
        # m_psi) it enters as a one-hot.
        src_dim = n_types + (d_z if gat_sources == "type_z" else 0)
        self.embed_t = nn.Embedding(n_types, src_dim)
        # spec 4.2: normalised counts + log-depth scalar + one-hot type
        self.enc_z = mlp([n_genes + 1 + n_types, hidden, hidden, 2 * d_z])
        self.gat = GATv2(src_dim=src_dim, dst_dim=src_dim,
                         out_dim=gat_dim, heads=heads, sink=gat_sink)
        # the projection test (S53, 2026-09-25): phi_proj > 0 puts a learned
        # linear map Phi -> phi_proj columns in c; 0 = the full Phi (pinned)
        c_dim = gat_dim + (phi_proj or phi_dim) + 1       # +1: isolated flag
        # ablation (iii): p(w|t) = N(mu_t, I), the context-free DisCoVR prior.
        # Default False = the spec's m_psi(c, t) = every pinned run.
        # arm (iii), 2026-09-23: prior_type_free=True drops t from m_psi.
        if class_mean_prior:
            self.prior_w = ClassMeanPrior(n_types, d_w)
        elif prior_type_free:
            self.prior_w = TypeFreePrior(n_types,
                                         mlp([c_dim, hidden // 4, d_w]))
        else:
            self.prior_w = mlp([c_dim + n_types, hidden // 4, d_w])
        self.enc_w = mlp([c_dim + n_types + d_z + n_genes + 1, hidden, 2 * d_w])
        # the decoder: a_g(z) + <w, B_g>. No t anywhere below this line.
        self.dec_a = mlp([d_z, hidden, n_genes])
        self.B = nn.Linear(d_w, n_genes, bias=False)
        # arms (i)/(ii), 2026-09-23: what the GAT queries with. Built last so
        # every pinned parameter above keeps its initialisation draw.
        self.query = query
        if query == "type_free":
            self.query_vec = nn.Parameter(torch.randn(src_dim))
        elif query == "image":
            self.query_phi = nn.Linear(phi_dim, src_dim)
        elif query != "type":
            raise ValueError(f"query must be type|type_free|image, got {query!r}")
        # review R12: the leak coefficient's form. A buffer, not a parameter,
        # and built after every layer, so no initialisation draw moves; under
        # "global" nothing is registered and the state dict is the pinned one.
        if kappa_mode not in KAPPA_MODES:
            raise ValueError(f"kappa_mode must be one of {KAPPA_MODES}, "
                             f"got {kappa_mode!r}")
        if kappa_mode != "global" and subtract_leak:
            raise ValueError("subtract_leak is defined for the global kappa "
                             "only")
        self.kappa_mode = kappa_mode
        if kappa_mode == "gene":
            # s_g / mean(s_g); NaN until filled, so a gene-form model that was
            # neither given s_g nor loaded from a checkpoint cannot decode
            scale = (torch.full((n_genes,), float("nan"))
                     if kappa_gene_share is None else
                     torch.as_tensor(np.asarray(kappa_gene_share, np.float64)
                                     / np.mean(kappa_gene_share),
                                     dtype=torch.float32))
            if scale.shape != (n_genes,):
                raise ValueError(f"kappa_gene_share has shape {tuple(scale.shape)}, "
                                 f"expected ({n_genes},)")
            self.register_buffer("kappa_gene_scale", scale)
        if kappa_mode in ("depth", "density"):
            # mean_train(r) (the amendment); NaN until given or loaded
            self.register_buffer("kappa_ratio_mean", torch.tensor(
                float("nan") if kappa_ratio_mean is None
                else float(kappa_ratio_mean)))
        # S53: built last, so every pinned layer keeps its initialisation
        # draw; None (the default) registers nothing and draws nothing
        self.phi_proj = nn.Linear(phi_dim, phi_proj) if phi_proj else None

    # -- pieces ------------------------------------------------------------

    def posterior_z(self, x: torch.Tensor, t: torch.Tensor):
        out = self.enc_z(torch.cat([encode_counts(x, self.median_counts),
                                    F.one_hot(t, self.n_types).float()], dim=-1))
        return out.chunk(2, dim=-1)

    def context(self, mu_z: torch.Tensor, t: torch.Tensor, phi: torch.Tensor,
                isolated: torch.Tensor, edge_src: torch.Tensor,
                edge_dst: torch.Tensor, n_context: int):
        """``c = GAT ⊕ Phi ⊕ flag`` for the first *n_context* nodes.

        The GAT query is ``embed(t)``, never ``z`` -- the mirror attractor.
        Under ``query="type_free"`` it is one learned vector shared by every
        cell, under ``query="image"`` a linear map of the ego-masked ``Phi_i``
        (arms (i)/(ii), 2026-09-23); neither reads the centre cell's label.
        Sources are ``[onehot(t_j), sg mu_z(x_j)]``: the stop-gradient keeps the
        niche encoder from training the ego encoder through the neighbours.
        """
        h_src = F.one_hot(t, self.n_types).float()
        if self.gat_sources == "type_z":
            h_src = torch.cat([h_src, mu_z.detach()], dim=-1)
        if self.query == "type_free":
            h_dst = self.query_vec.expand(n_context, -1)
        elif self.query == "image":
            h_dst = self.query_phi(phi[:n_context])
        else:
            h_dst = self.embed_t(t[:n_context])
        gat, alpha = self.gat(h_src, h_dst, edge_src, edge_dst)
        flag = isolated[:n_context].float()[:, None]
        phi_c = (phi[:n_context] if self.phi_proj is None
                 else self.phi_proj(phi[:n_context]))
        c = torch.cat([gat, phi_c, flag], dim=-1)
        return c, alpha

    def log_rho(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Clean composition: baseline programme plus environmental shift."""
        return F.log_softmax(self.dec_a(z) + self.B(w), dim=-1)

    def leak_kappa(self, kappa: float, x_context: torch.Tensor,
                   leak_src: torch.Tensor, leak_dst: torch.Tensor,
                   leak_beta: torch.Tensor, n_seeds: int,
                   area: torch.Tensor | None = None):
        """The leak coefficient per seed under ``self.kappa_mode``.

        "global" returns *kappa* itself, the same float, so the mixture takes
        its pinned path bit for bit. "depth": ``l`` is the raw count total of
        each context node (the batch's own counts), the donor depth
        ``sum_j beta_ij l_j`` is scattered over the leak edges, and
        ``kappa_i = kappa * clip(donor / l_i, 0, KAPPA_DEPTH_MAX / kappa)``,
        written as ``min(kappa * donor / l_i, KAPPA_DEPTH_MAX)`` (the same
        number, and it holds at kappa 0); a seed without a leak edge keeps
        ``kappa``. "gene": ``min(kappa * s_g / mean(s_g), KAPPA_GENE_MAX)``
        for every seed with a leak edge, 0 without one -- its rho_bar row is
        zero, so both keep an isolated cell's mixture at rho, as the global
        form's renormalisation does. "density" is "depth" with every count
        total divided by its cell's area (*area*, the context nodes' ``A``):
        ``kappa_i = kappa * clip(sum_j beta_ij (l_j/A_j) / (l_i/A_i), 0,
        c_max)``, the same clip; with every area equal it is "depth". In both,
        the ratio is divided by ``kappa_ratio_mean`` (its mean over the
        connected training cells, :func:`kappa_ratio_stats`), so the mean
        kappa_i over those cells is kappa up to the clip.

        Why "depth" is not the reweighting ``beta_ij * l_j / l_i`` with rows
        renormalised (the writer's proposal, review R12): renormalising
        cancels ``l_i``, so a cell still leaks ``kappa * l_i`` counts whoever
        its donors are -- only the split across neighbours moves. Here the
        row is not renormalised, so the leaked count is ``kappa_i * l_i =
        kappa * sum_j beta_ij l_j / mean_train(r)`` below the cap: the
        amount moves with the donors, cell by cell.
        """
        if not 0.0 <= kappa < 1.0:
            raise ValueError(f"kappa must be in [0, 1), got {kappa}")
        if self.kappa_mode == "global":
            return kappa
        has_edge = leak_beta.new_zeros(n_seeds).index_add_(
            0, leak_dst, leak_beta) > 0
        if self.kappa_mode in ("depth", "density"):
            depth = x_context.sum(dim=-1)
            own = depth[:n_seeds].clamp(min=1.0)
            if self.kappa_mode == "density":
                if area is None:
                    raise ValueError("the density form needs the cell areas")
                area = area[:depth.shape[0]]
                depth, own = depth / area, own / area[:n_seeds]
            donor = depth.new_zeros(n_seeds).index_add_(
                0, leak_dst, leak_beta * depth[leak_src])
            ratio = donor / own / self.kappa_ratio_mean
            k_i = (kappa * ratio).clamp(min=0.0, max=KAPPA_DEPTH_MAX)
            return torch.where(has_edge, k_i, torch.full_like(k_i, kappa))[:, None]
        k_g = (kappa * self.kappa_gene_scale).clamp(max=KAPPA_GENE_MAX)
        return k_g[None, :] * has_edge[:, None].to(k_g.dtype)

    # -- the tile forward pass ---------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor, phi: torch.Tensor,
                isolated: torch.Tensor, gat_src: torch.Tensor,
                gat_dst: torch.Tensor, leak_src: torch.Tensor,
                leak_dst: torch.Tensor, leak_beta: torch.Tensor,
                n_seeds: int, n_context: int, kappa: float,
                sample: bool = True,
                area: torch.Tensor | None = None,
                eta: torch.Tensor | None = None) -> Forward:
        """One tile: all nodes in [seeds | ring1 | ring2] layout.

        *eta* ``(n_seeds, 1)``: the fixed false-positive floor (todo 8.15b),
        ``p = (1 - kappa_i - eta_i) rho + kappa_i rho_bar + eta_i u`` in both
        mixtures (:func:`leakage_mix`); None, the default, is the pinned pass.

        ``c``/``w``/``rho`` are built for seeds and ring1 (ring1's rho feeds
        the leak), the losses for seeds only. Exact two hops, no cache.

        ``z`` is encoded for every node under ``type_z``, where ring 2's
        ``sg mu_z`` is a GAT source for ring 1. Under the default
        ``type_only`` the GAT sources are ``onehot(t_j)`` alone, so ring-2
        ``mu_z`` is never read and the encoder runs on seeds u ring1 only --
        spec 4.5 ("ring2: type/Phi lookup only -- no encoder pass").
        """
        from discell.model.equations import foreign_influx, leakage_mix

        n_nodes = t.shape[0]
        n_enc = n_nodes if self.gat_sources == "type_z" else n_context
        mu_z, logvar_z = self.posterior_z(x[:n_enc], t[:n_enc])
        # sample=False: posterior means throughout, so an evaluation sweep is
        # deterministic -- early stopping should not ride reparameterisation
        # noise. Training always samples.
        # The draw keeps the *node* count even when only n_enc rows are
        # encoded, so skipping ring 2 leaves the RNG stream -- and therefore
        # the whole loss trajectory -- bit-identical to the pre-skip pass.
        noise = (torch.randn(n_nodes, self.d_z, dtype=mu_z.dtype,
                             device=mu_z.device)[:n_enc] if sample
                 else torch.zeros_like(mu_z))
        z = mu_z + noise * (0.5 * logvar_z).exp()

        c, alpha = self.context(mu_z, t, phi, isolated, gat_src, gat_dst, n_context)

        t_c = F.one_hot(t[:n_context], self.n_types).float()
        prior_mean_w = self.prior_w(torch.cat([c, t_c], dim=-1))
        mu_w, logvar_w = self.enc_w(torch.cat(
            [c, t_c, z[:n_context],
             encode_counts(x[:n_context], self.median_counts)], dim=-1)
        ).chunk(2, dim=-1)
        w_noise = torch.randn_like(mu_w) if sample else torch.zeros_like(mu_w)
        w = mu_w + w_noise * (0.5 * logvar_w).exp()

        log_rho = self.log_rho(z[:n_context], w)
        # the leak mixture reads every neighbour -- ring1 or fellow seed --
        # through the stop-gradient (spec 4.5).
        rho_bar = foreign_influx(log_rho.detach().exp(), leak_src, leak_dst,
                                 leak_beta, n_dst=n_seeds)
        kappa_eff = self.leak_kappa(kappa, x[:n_context], leak_src, leak_dst,
                                    leak_beta, n_seeds, area)
        if self.subtract_leak:
            # Pass 2 (spec 7.13): the true posterior is p(z | x, rho_bar) and
            # the bias kappa*l*rho_bar cannot be removed by any function of x
            # alone, so the seeds are re-encoded from x~ = x - kappa*l*rho_bar
            # (rho_bar is data). Features only -- the likelihood below stays
            # on raw x. Ring-1 rho_j keep their pass-1 (raw-input) encoding:
            # an O(kappa^2) inconsistency accepted by the spec. enc_w reads
            # x~ as well; x~ is normalised by its own sum, log-depth stays l.
            x_s = x[:n_seeds]
            depth = x_s.sum(dim=-1, keepdim=True)
            x_tilde = (x_s - kappa * depth * rho_bar).clamp(min=0.0)
            feat = encode_counts(x_tilde, self.median_counts,
                                 log_depth=depth.clamp(min=1.0).log())
            t_s = t_c[:n_seeds]
            mu_z2, logvar_z2 = self.enc_z(
                torch.cat([feat, t_s], dim=-1)).chunk(2, dim=-1)
            z2 = mu_z2 + noise[:n_seeds] * (0.5 * logvar_z2).exp()
            mu_w2, logvar_w2 = self.enc_w(torch.cat(
                [c[:n_seeds], t_s, z2, feat], dim=-1)).chunk(2, dim=-1)
            w2 = mu_w2 + w_noise[:n_seeds] * (0.5 * logvar_w2).exp()
            mu_z = torch.cat([mu_z2, mu_z[n_seeds:]])
            logvar_z = torch.cat([logvar_z2, logvar_z[n_seeds:]])
            z = torch.cat([z2, z[n_seeds:]])
            mu_w = torch.cat([mu_w2, mu_w[n_seeds:]])
            logvar_w = torch.cat([logvar_w2, logvar_w[n_seeds:]])
            w = torch.cat([w2, w[n_seeds:]])
            log_rho = torch.cat([self.log_rho(z2, w2), log_rho[n_seeds:]])
        log_p = leakage_mix(log_rho[:n_seeds].exp(), rho_bar, kappa_eff, eta)

        # term (b): same seeds, w drawn from the prior instead of the posterior
        w_breve = prior_mean_w[:n_seeds] + (
            torch.randn_like(mu_w[:n_seeds]) if sample
            else torch.zeros_like(mu_w[:n_seeds]))
        log_rho_breve = self.log_rho(z[:n_seeds], w_breve)
        log_p_breve = leakage_mix(log_rho_breve.exp(), rho_bar, kappa_eff, eta)

        return Forward(mu_z=mu_z, logvar_z=logvar_z, z=z, c=c,
                       prior_mean_w=prior_mean_w, mu_w=mu_w, logvar_w=logvar_w,
                       w=w, log_rho=log_rho, rho_bar=rho_bar, log_p=log_p,
                       log_p_breve=log_p_breve, alpha=alpha,
                       kappa_eff=kappa_eff, eta=eta)
