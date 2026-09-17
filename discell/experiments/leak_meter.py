#!/usr/bin/env python3
"""Leak meter: per-cell leak-in estimated from the slide alone (concept doc 16).

How. A gene's own transcripts sit over the nucleus at that gene's retention
``eta_g``; leaked transcripts land in the segmentation rim at one nuclear
share ``zeta``. For genes that only one type ``t`` makes, pooled over the
receivers of type ``r``, the nuclear shortfall ``D_g = N_nuc_g - eta_g(t)
X_tot_g`` equals ``(zeta - eta_g(t))`` times the leaked copies, which are
modelled as ``c_rt Phi_g`` with ``Phi_g = S_rt rho_t(g)``: ``S_rt`` the pooled
face-weighted, distance-decayed transcript density of type-t neighbours
around those receivers, ``rho_t`` the sender's mean profile. Weighted least
squares over the specific genes gives ``c_rt`` (a rim depth in um) and
``zeta`` per pair. The off-diagonal table is factorised as ``a_r b_t``, which
also predicts the same-type entries no gene can measure, and per cell
``kappa_i = a_r(i) sum_j f_ij exp(-d_ij/tau) dens_j b_t(j) / l_i``.

Evaluated by. The document's own checks: specific genes per type (>= 30
wished), the retention of those genes against the fitted ``zeta``, the WLS
standard errors, the factorisation misfit, the ``kappa_i`` distribution
against the global kappa = 0.1 in use, and the tendencies against the
per-type nuclear fractions already measured. Tiles are split in two halves:
genes and profiles are learned on one, the table is fitted on the other.

What is wished for. A leak input DisCell could take as fixed
(``p_i = (1 - m kappa_i) rho_i + m kappa_i rhobar_i``, ``m`` swept) in place
of one swept global kappa. A failed check is a finding, not a tuning target.

Usage::

    OMP_NUM_THREADS=8 python -m discell.experiments.leak_meter --dataset <id> \\
        [--variant full] [--merge "Tumour=Tumor Cells|VEGFA+ Tumor Cells;..."] [--tag lineage]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from discell import paths
from discell.model.prepare import DEFAULT_MAX_EDGE_UM, DEFAULT_TAU_UM, spatial_tiles

log = logging.getLogger("discell.experiments.leak_meter")

UNASSIGNED = "Unassigned"
#: The global kappa every pinned run uses (TrainConfig.kappa), for comparison.
GLOBAL_KAPPA = 0.1


@dataclass
class Slide:
    """Everything the meter reads, all from the slide."""

    counts: sp.csr_matrix          # (N, G) cell-assigned q20 counts (bundle)
    nuclear: sp.csr_matrix         # (N, G) the nucleus-flagged subset of them
    type_index: np.ndarray         # (N,)
    type_names: np.ndarray         # (K,)
    edge_i: np.ndarray             # undirected, already pruned by length
    edge_j: np.ndarray
    face_um: np.ndarray            # shared Voronoi face per edge
    dist_um: np.ndarray            # centroid distance per edge
    area_um2: np.ndarray           # (N,) segmented cell area
    positions_um: np.ndarray       # (N, 2)
    has_nucleus: np.ndarray        # (N,) receivers need a nucleus polygon

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    @property
    def total(self) -> np.ndarray:
        return np.asarray(self.counts.sum(axis=1)).ravel()

    @property
    def degrees(self) -> np.ndarray:
        return np.bincount(np.concatenate([self.edge_i, self.edge_j]),
                           minlength=self.counts.shape[0])


@dataclass
class Profiles:
    """Type-pooled counts on one half of the slide."""

    nuc: np.ndarray        # (K, G) pooled nuclear counts
    tot: np.ndarray        # (K, G) pooled total counts
    n_cells: np.ndarray    # (K,)

    @property
    def nuc_share(self) -> np.ndarray:
        """Nuclear-only mean profile, the cleaner view used to pick genes."""
        return self.nuc / np.maximum(self.nuc.sum(axis=1, keepdims=True), 1.0)

    @property
    def rho(self) -> np.ndarray:
        """Total mean profile: what the document says a type-t neighbour leaks."""
        return self.tot / np.maximum(self.tot.sum(axis=1, keepdims=True), 1.0)

    @property
    def rho_extranuclear(self) -> np.ndarray:
        """Extranuclear mean profile: the rim's own composition, as a check."""
        extra = np.maximum(self.tot - self.nuc, 0.0)
        return extra / np.maximum(extra.sum(axis=1, keepdims=True), 1.0)

    @property
    def eta(self) -> np.ndarray:
        """Retention ``eta_g(t)``: each gene's nuclear share within type t."""
        return self.nuc / np.maximum(self.tot, 1.0)


# --------------------------------------------------------------------------
# the estimator, on arrays
# --------------------------------------------------------------------------


def split_halves(positions: np.ndarray, seed: int, tile_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Learn / fit masks: a random half of the contiguous spatial tiles each."""
    tiles = spatial_tiles(positions, tile_cells)
    order = np.random.default_rng([seed, 16]).permutation(len(tiles))
    learn = np.zeros(len(positions), dtype=bool)
    for k in order[: len(tiles) // 2]:
        learn[tiles[k]] = True
    return learn, ~learn


def pooled(matrix, type_index: np.ndarray, n_types: int, mask: np.ndarray) -> np.ndarray:
    """(K, G) column sums of *matrix* over the cells of each type inside *mask*."""
    rows = np.flatnonzero(mask)
    indicator = sp.csr_matrix((np.ones(len(rows)), (type_index[rows], rows)),
                              shape=(n_types, matrix.shape[0]))
    return np.asarray((indicator @ matrix).todense(), dtype=np.float64)


def profiles(slide: Slide, mask: np.ndarray) -> Profiles:
    return Profiles(pooled(slide.nuclear, slide.type_index, slide.n_types, mask),
                    pooled(slide.counts, slide.type_index, slide.n_types, mask),
                    np.bincount(slide.type_index[mask], minlength=slide.n_types))


def specific_genes(learn: Profiles, senders: Sequence[int], ratio: float,
                   min_counts: float) -> dict[int, np.ndarray]:
    """Genes whose nuclear share in t is >= *ratio* x that in every other sender
    type, with at least *min_counts* nuclear copies in t (document step 3)."""
    share = learn.nuc_share
    out = {}
    for t in senders:
        rivals = [u for u in senders if u != t]
        rival = share[rivals].max(axis=0) if rivals else np.zeros(share.shape[1])
        keep = (share[t] > 0) & (share[t] >= ratio * rival) & (learn.nuc[t] >= min_counts)
        out[int(t)] = np.flatnonzero(keep)
    return out


def exposure(slide: Slide, tau_um: float) -> np.ndarray:
    """(N, K): ``sum_j f_ij exp(-d_ij/tau) dens_j`` over each cell's neighbours
    of each type, ``dens_j`` = counts / area (the document's density proxy)."""
    total = slide.total
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(slide.area_um2 > 0, total / slide.area_um2, 0.0)
    dens = np.nan_to_num(dens)
    weight = slide.face_um * np.exp(-slide.dist_um / float(tau_um))
    out = np.zeros((slide.counts.shape[0], slide.n_types))
    np.add.at(out, (slide.edge_i, slide.type_index[slide.edge_j]), weight * dens[slide.edge_j])
    np.add.at(out, (slide.edge_j, slide.type_index[slide.edge_i]), weight * dens[slide.edge_i])
    return out


def leak_pair(n_nuc: np.ndarray, x_tot: np.ndarray, phi: np.ndarray, eta_t: np.ndarray,
              genes: np.ndarray) -> tuple[float, float, float, float]:
    """The document's fit for one receiver-sender pair, with standard errors.

    ``D_g = c (zeta - eta_g) Phi_g`` as WLS on ``[Phi, eta Phi]`` with weights
    ``1 / max(X_tot, 1)``; returns ``(c, zeta, se_c, se_zeta)``, the residual
    scale estimated from the fit and ``zeta``'s error by the delta method.
    """
    d = n_nuc[genes] - eta_t[genes] * x_tot[genes]
    design = np.stack([phi[genes], eta_t[genes] * phi[genes]], axis=1)
    sw = np.sqrt(1.0 / np.maximum(x_tot[genes], 1.0))
    aw, dw = design * sw[:, None], d * sw
    coef, *_ = np.linalg.lstsq(aw, dw, rcond=None)
    resid = dw - aw @ coef
    dof = max(len(genes) - 2, 1)
    cov = (resid @ resid / dof) * np.linalg.pinv(aw.T @ aw)
    c = -coef[1]
    if c == 0:
        return 0.0, float("nan"), float(np.sqrt(cov[1, 1])), float("nan")
    zeta = coef[0] / c
    grad = np.array([1.0 / c, coef[0] / c ** 2])
    return float(c), float(zeta), float(np.sqrt(cov[1, 1])), float(np.sqrt(grad @ cov @ grad))


def fit_table(learn: Profiles, specific: dict[int, np.ndarray], fit_nuc: np.ndarray,
              fit_tot: np.ndarray, s_rt: np.ndarray, rho: np.ndarray,
              min_genes: int = 3) -> dict[str, np.ndarray]:
    """Document step 4 for every receiver r and sender t != r with genes.

    Also two one-parameter references over the same genes: ``c_counts`` =
    total counts / Phi (spec 7.7's marker ceiling, an upper bound if the
    genes are truly specific) and ``c_zeta0`` (the fit with ``zeta`` pinned
    at 0, immune to the extrapolation that ``zeta`` needs).
    """
    k = s_rt.shape[0]
    out = {key: np.full((k, k), np.nan) for key in
           ("c", "zeta", "se_c", "se_zeta", "c_counts", "c_zeta0")}
    out["n_genes"] = np.zeros((k, k), dtype=int)
    eta = learn.eta
    for r in range(k):
        for t, genes in specific.items():
            if t == r or len(genes) < min_genes or s_rt[r, t] <= 0:
                continue
            phi = s_rt[r, t] * rho[t]
            c, zeta, se_c, se_zeta = leak_pair(fit_nuc[r], fit_tot[r], phi, eta[t], genes)
            out["c"][r, t], out["zeta"][r, t] = c, zeta
            out["se_c"][r, t], out["se_zeta"][r, t] = se_c, se_zeta
            out["n_genes"][r, t] = len(genes)
            out["c_counts"][r, t] = fit_tot[r, genes].sum() / max(phi[genes].sum(), 1e-300)
            w = 1.0 / np.maximum(fit_tot[r, genes], 1.0)
            d = fit_nuc[r, genes] - eta[t, genes] * fit_tot[r, genes]
            x = eta[t, genes] * phi[genes]
            out["c_zeta0"][r, t] = -(w * d * x).sum() / max((w * x * x).sum(), 1e-300)
    return out


def factorise(c: np.ndarray, se: np.ndarray, iters: int = 500) -> dict:
    """Weighted rank-1 fit ``c_rt ~ a_r b_t`` over the measured entries.

    Alternating least squares with weights ``1/se^2``; gauge: the mean sender
    tendency over the senders that have measurements is 1, so ``a_r`` is a
    rim depth in um against an average sender. Rows or columns with no
    measurement come back NaN.
    """
    mask = np.isfinite(c) & np.isfinite(se) & (se > 0)
    w = np.where(mask, 1.0 / np.maximum(np.where(mask, se, 1.0), 1e-12) ** 2, 0.0)
    cc = np.where(mask, c, 0.0)
    rows, cols = mask.any(axis=1), mask.any(axis=0)
    a, b = np.ones(c.shape[0]), np.ones(c.shape[1])
    for _ in range(iters):
        den = (w * b[None, :] ** 2).sum(axis=1)
        a = np.where(rows, (w * cc * b[None, :]).sum(axis=1) / np.maximum(den, 1e-300), 0.0)
        den = (w * a[:, None] ** 2).sum(axis=0)
        b = np.where(cols, (w * cc * a[:, None]).sum(axis=0) / np.maximum(den, 1e-300), 0.0)
        scale = b[cols].mean()
        if scale > 0:
            b, a = b / scale, a * scale
    pred = a[:, None] * b[None, :]
    resid = (cc - pred)[mask]
    n_par = int(rows.sum() + cols.sum() - 1)
    dof = max(int(mask.sum()) - n_par, 1)
    chi2 = float((resid ** 2 * w[mask]).sum() / dof)
    rel_rms = float(np.sqrt((resid ** 2).mean()) / max(np.sqrt((cc[mask] ** 2).mean()), 1e-300))
    r2 = float(1.0 - (resid ** 2).sum() / max(((cc[mask] - cc[mask].mean()) ** 2).sum(), 1e-300))
    beyond = float((np.abs(resid) > 2.0 * se[mask]).mean())
    return {"a": np.where(rows, a, np.nan), "b": np.where(cols, b, np.nan), "predicted": pred,
            "chi2_per_dof": chi2, "rel_rms": rel_rms, "r2": r2, "frac_beyond_2se": beyond,
            "n_entries": int(mask.sum())}


def per_cell_kappa(expo: np.ndarray, a: np.ndarray, b: np.ndarray, type_index: np.ndarray,
                   total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``kappa_i`` (all neighbours) and its part from other-type neighbours only.

    A sender type without a tendency (e.g. Unassigned) contributes the mean
    ``b``; a receiver type without one gets NaN.
    """
    b_eff = np.where(np.isfinite(b), b, np.nanmean(b))
    own = np.eye(len(b_eff))[type_index]
    leaked = a[type_index] * (expo @ b_eff)
    leaked_off = a[type_index] * ((expo * (1.0 - own)) @ b_eff)
    denom = np.maximum(total, 1.0)
    return leaked / denom, leaked_off / denom


def measure(slide: Slide, seed: int = 0, ratio: float = 10.0, min_counts: float = 200.0,
            tau_um: float = DEFAULT_TAU_UM, tile_cells: int = 4096,
            sender_profile: str = "total",
            specific: dict[int, np.ndarray] | None = None) -> dict:
    """Document steps 1-6 on one slide; arrays only, no writing."""
    names = [str(n) for n in slide.type_names]
    senders = [k for k, n in enumerate(names) if n.casefold() != UNASSIGNED.casefold()]
    learn_mask, fit_mask = split_halves(slide.positions_um, seed, tile_cells)
    total = slide.total
    receiver_ok = slide.has_nucleus & (total > 0)
    learn_cells = learn_mask & receiver_ok
    fit_cells = fit_mask & receiver_ok & (slide.degrees > 0)

    learn = profiles(slide, learn_cells)
    if specific is None:
        specific = specific_genes(learn, senders, ratio, min_counts)
    expo = exposure(slide, tau_um)
    s_rt = pooled(sp.csr_matrix(expo), slide.type_index, slide.n_types, fit_cells)
    fit_nuc = pooled(slide.nuclear, slide.type_index, slide.n_types, fit_cells)
    fit_tot = pooled(slide.counts, slide.type_index, slide.n_types, fit_cells)
    rho = learn.rho if sender_profile == "total" else learn.rho_extranuclear
    table = fit_table(learn, specific, fit_nuc, fit_tot, s_rt, rho)
    fac = factorise(table["c"], table["se_c"])
    kappa, kappa_off = per_cell_kappa(expo, fac["a"], fac["b"], slide.type_index, total)
    exposed = np.zeros_like(s_rt, dtype=int)
    for r in range(slide.n_types):
        rows = fit_cells & (slide.type_index == r)
        exposed[r] = (expo[rows] > 0).sum(axis=0)
    return {"names": names, "senders": senders, "learn": learn, "specific": specific,
            "learn_mask": learn_cells, "fit_mask": fit_cells, "exposure": expo, "s_rt": s_rt,
            "n_exposed": exposed, "table": table, "factorisation": fac,
            "kappa": kappa, "kappa_off": kappa_off, "total": total}


# --------------------------------------------------------------------------
# the slide
# --------------------------------------------------------------------------


def parse_merge(text: str | None) -> dict[str, list[str]]:
    """``"New=Old1|Old2;New2=Old3"`` -> {"New": ["Old1", "Old2"], ...}."""
    out: dict[str, list[str]] = {}
    for group in filter(None, (text or "").split(";")):
        new, olds = group.split("=", 1)
        out[new.strip()] = [o.strip() for o in olds.split("|") if o.strip()]
    return out


def merge_types(type_index: np.ndarray, type_names: np.ndarray,
                merge: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Relabel types onto lineage groups; names not mentioned are kept."""
    mapping = {old: new for new, olds in merge.items() for old in olds}
    unknown = set(mapping) - set(map(str, type_names))
    if unknown:
        raise ValueError(f"merge names not in the labels: {sorted(unknown)}")
    renamed = np.array([mapping.get(str(n), str(n)) for n in type_names])
    names = np.asarray(sorted(set(renamed)))
    lookup = {n: k for k, n in enumerate(names)}
    remap = np.array([lookup[n] for n in renamed])
    return remap[type_index], names


def _full_cell_ids(h5ad_path) -> np.ndarray:
    """obs_names of a bundle without opening it: the qc matrix is full-slide."""
    import h5py

    with h5py.File(h5ad_path, "r") as f:
        key = f["obs"].attrs["_index"]
        return f["obs"][key][:].astype(str)


def load_slide(dataset: str, variant: str, label_key: str | None,
               merge: dict[str, list[str]], max_edge_um: float) -> Slide:
    import pandas as pd

    from discell.data.loader import CellGraphDataset

    ds = paths.dataset(dataset)
    opened = CellGraphDataset.from_dataset(ds, variant, label_key=label_key)
    nuclear = sp.load_npz(ds.root / "qc" / "nuclear_counts.npz").tocsr()
    if nuclear.shape[0] != opened.adata.n_obs:
        rows = pd.Index(_full_cell_ids(ds.bundle_dir / "full.h5ad")).get_indexer(opened.cell_ids)
        if (rows < 0).any():
            raise ValueError(f"{int((rows < 0).sum())} cells of {variant!r} missing from full.h5ad")
        nuclear = nuclear[rows]
    if nuclear.shape != opened.counts.shape:
        raise ValueError(f"nuclear {nuclear.shape} vs counts {opened.counts.shape}")

    face = opened._pull_edges("shared_wall_um")
    dist = opened._pull_edges("centroid_dist_um")
    keep = dist <= max_edge_um
    obs = opened.adata.obs
    if "xenium_nucleus_count" in obs:
        has_nucleus = obs["xenium_nucleus_count"].to_numpy() >= 1
    elif "xenium_nucleus_area" in obs:
        has_nucleus = obs["xenium_nucleus_area"].notna().to_numpy()
    else:
        log.warning("no nucleus column in obs -- every cell admitted as receiver")
        has_nucleus = np.ones(opened.adata.n_obs, dtype=bool)
    type_index, type_names = opened.type_index, opened.type_names
    if merge:
        type_index, type_names = merge_types(type_index, type_names, merge)
    return Slide(counts=opened.counts.tocsr(), nuclear=nuclear, type_index=type_index,
                 type_names=type_names, edge_i=opened.edge_i[keep], edge_j=opened.edge_j[keep],
                 face_um=face[keep].astype(np.float64), dist_um=dist[keep].astype(np.float64),
                 area_um2=obs["area_um2"].to_numpy(dtype=np.float64),
                 positions_um=opened.positions_um, has_nucleus=has_nucleus)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


def _quantiles(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"n": int(len(v)), "mean": float(v.mean()), "p5": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "p99": float(q[5]),
            "frac_gt_0.9": float((v > 0.9).mean()), "frac_gt_1": float((v > 1.0).mean())}


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return None
    return float(spearmanr(x[ok], y[ok]).statistic)


def summarise(slide: Slide, res: dict, gene_names: np.ndarray | None) -> dict:
    names, table, fac = res["names"], res["table"], res["factorisation"]
    learn, specific = res["learn"], res["specific"]
    k = len(names)
    eta = learn.eta
    genes_out = {}
    for t, genes in specific.items():
        e = eta[t, genes]
        genes_out[names[t]] = {
            "n": int(len(genes)),
            "eta_min": float(e.min()) if len(e) else None,
            "eta_median": float(np.median(e)) if len(e) else None,
            "eta_max": float(e.max()) if len(e) else None,
            "eta_sd": float(e.std()) if len(e) else None,
            "nuclear_counts_learn": float(learn.nuc[t, genes].sum()),
            "genes": ([str(gene_names[g]) for g in genes[:200]] if gene_names is not None
                      else genes[:200].tolist()),
        }
    pairs = {}
    for r in range(k):
        row = {}
        for t in range(k):
            if not np.isfinite(table["c"][r, t]):
                continue
            row[names[t]] = {
                "c_um": float(table["c"][r, t]), "se_c": float(table["se_c"][r, t]),
                "zeta": float(table["zeta"][r, t]), "se_zeta": float(table["se_zeta"][r, t]),
                "c_counts_only": float(table["c_counts"][r, t]),
                "c_zeta0": float(table["c_zeta0"][r, t]),
                "n_genes": int(table["n_genes"][r, t]),
                "exposure_S": float(res["s_rt"][r, t]),
                "n_receivers_exposed": int(res["n_exposed"][r, t]),
                "predicted_ab": float(fac["predicted"][r, t]),
            }
        if row:
            pairs[names[r]] = row

    kappa, kappa_off, total = res["kappa"], res["kappa_off"], res["total"]
    finite = np.isfinite(kappa)
    by_type = {}
    for r in range(k):
        rows = (slide.type_index == r) & finite
        if rows.any():
            by_type[names[r]] = {"median": float(np.median(kappa[rows])),
                                 "p95": float(np.quantile(kappa[rows], 0.95)),
                                 "pooled": float((kappa[rows] * total[rows]).sum()
                                                 / max(total[rows].sum(), 1.0))}
    with np.errstate(invalid="ignore", divide="ignore"):
        same_share = np.where(kappa > 0, 1.0 - kappa_off / kappa, np.nan)
    connected = slide.degrees > 0

    all_cells = profiles(slide, slide.has_nucleus & (total > 0))
    nuc_frac = all_cells.nuc.sum(axis=1) / np.maximum(all_cells.tot.sum(axis=1), 1.0)
    a, b = fac["a"], fac["b"]
    zeta_flat, se_zeta_flat = table["zeta"], table["se_zeta"]
    ok = np.isfinite(zeta_flat) & np.isfinite(se_zeta_flat) & (se_zeta_flat > 0)
    eta_median_by_sender = np.array([np.median(eta[t, specific[t]]) if t in specific
                                     and len(specific[t]) else np.nan for t in range(k)])
    gap = (eta_median_by_sender[None, :] - zeta_flat) / np.where(ok, se_zeta_flat, np.nan)
    ok_c = np.isfinite(table["c"]) & (table["se_c"] > 0)
    return {
        "n_types": k, "types": names,
        "n_cells": int(slide.counts.shape[0]),
        "n_learn_cells": int(res["learn_mask"].sum()), "n_fit_cells": int(res["fit_mask"].sum()),
        "n_without_nucleus": int((~slide.has_nucleus).sum()),
        "n_isolated": int((~connected).sum()),
        "specific_genes": genes_out,
        "n_types_with_30_genes": int(sum(len(g) >= 30 for g in specific.values())),
        "pairs": pairs,
        "pairs_measured": int(ok_c.sum()),
        "pairs_c_positive_2se": int((ok_c & (table["c"] > 2 * table["se_c"])).sum()),
        "pairs_c_negative": int((ok_c & (table["c"] < 0)).sum()),
        "zeta_check": {
            "zeta_median_over_pairs": float(np.nanmedian(zeta_flat[ok])) if ok.any() else None,
            "zeta_iqr": ([float(q) for q in np.nanquantile(zeta_flat[ok], [0.25, 0.75])]
                         if ok.any() else None),
            "frac_pairs_eta_minus_zeta_beyond_2se": (float((np.abs(gap[ok]) > 2).mean())
                                                     if ok.any() else None),
            "frac_pairs_zeta_in_unit_interval": (float(((zeta_flat[ok] >= 0)
                                                        & (zeta_flat[ok] <= 1)).mean())
                                                 if ok.any() else None),
        },
        "factorisation": {
            "a_receiver_um": {names[r]: (float(a[r]) if np.isfinite(a[r]) else None) for r in range(k)},
            "b_sender": {names[t]: (float(b[t]) if np.isfinite(b[t]) else None) for t in range(k)},
            "same_type_predicted_um": {names[r]: (float(a[r] * b[r]) if np.isfinite(a[r] * b[r])
                                                  else None) for r in range(k)},
            "chi2_per_dof": fac["chi2_per_dof"], "rel_rms": fac["rel_rms"], "r2": fac["r2"],
            "frac_entries_beyond_2se": fac["frac_beyond_2se"], "n_entries": fac["n_entries"],
        },
        "kappa": {
            "global_reference": GLOBAL_KAPPA,
            "all_connected": _quantiles(kappa[connected]),
            "other_type_only_connected": _quantiles(kappa_off[connected]),
            "pooled_leaked_over_total": float(np.nansum(kappa[connected] * total[connected])
                                              / max(total[connected].sum(), 1.0)),
            "same_type_share_median": float(np.nanmedian(same_share[connected])),
            "by_type": by_type,
        },
        "nuclear_fraction_by_type": {names[r]: float(nuc_frac[r]) for r in range(k)},
        "spearman_a_vs_nuclear_fraction": _spearman(a, nuc_frac),
        "spearman_b_vs_nuclear_fraction": _spearman(b, nuc_frac),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    merge = parse_merge(args.merge)
    slide = load_slide(args.dataset, args.variant, args.label_key, merge, args.max_edge_um)
    log.info("slide: %d cells, %d types, %d edges kept (<= %g um), %d cells without nucleus",
             slide.counts.shape[0], slide.n_types, len(slide.edge_i), args.max_edge_um,
             int((~slide.has_nucleus).sum()))
    res = measure(slide, seed=args.seed, ratio=args.ratio, min_counts=args.min_counts,
                  tau_um=args.tau_um, tile_cells=args.tile_cells,
                  sender_profile=args.sender_profile)
    from discell.data.loader import CellGraphDataset  # noqa: F401  (gene names via the bundle)
    import anndata as ad

    var = ad.read_h5ad(paths.dataset(args.dataset).bundle_dir / f"{args.variant}.h5ad",
                       backed="r").var
    gene_names = np.asarray(var["gene_name"].astype(str)) if "gene_name" in var else None
    summary = summarise(slide, res, gene_names)
    summary = {"dataset": args.dataset, "variant": args.variant, "label_key": args.label_key,
               "merge": merge, "seed": args.seed, "tau_um": args.tau_um,
               "max_edge_um": args.max_edge_um, "ratio": args.ratio,
               "min_counts": args.min_counts, "tile_cells": args.tile_cells,
               "sender_profile": args.sender_profile, **summary}

    out_dir = paths.dataset(args.dataset).root / "experiments"
    out_dir.mkdir(exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = out_dir / f"leak_meter_{args.variant}{tag}.json"
    out.write_text(json.dumps(summary, indent=1))
    log.info("wrote %s", out)

    names = res["names"]
    print(f"specific genes per type ({summary['n_types_with_30_genes']} of "
          f"{len(res['senders'])} senders reach 30):")
    for t in res["senders"]:
        g = summary["specific_genes"][names[t]]
        print(f"  {names[t]:40s} {g['n']:4d}  eta median {g['eta_median'] if g['eta_median'] is not None else float('nan'):.2f}")
    fac = summary["factorisation"]
    print(f"table: {summary['pairs_measured']} pairs measured, "
          f"{summary['pairs_c_positive_2se']} positive beyond 2 SE, "
          f"{summary['pairs_c_negative']} negative; factorisation chi2/dof "
          f"{fac['chi2_per_dof']:.2f}, rel RMS {fac['rel_rms']:.2f}, R2 {fac['r2']:.2f}")
    print("receiver a_r (um) / sender b_t / same-type a_r b_r (um):")
    for n in names:
        a, b, ab = fac["a_receiver_um"][n], fac["b_sender"][n], fac["same_type_predicted_um"][n]
        fmt = lambda v: "   nan" if v is None else f"{v:6.3f}"  # noqa: E731
        print(f"  {n:40s} {fmt(a)}  {fmt(b)}  {fmt(ab)}")
    kq = summary["kappa"]["all_connected"]
    print(f"kappa_i (connected): median {kq.get('median', float('nan')):.3f}, "
          f"IQR [{kq.get('p25', float('nan')):.3f}, {kq.get('p75', float('nan')):.3f}], "
          f"p95 {kq.get('p95', float('nan')):.3f}, >0.9: {kq.get('frac_gt_0.9', float('nan')):.4f}; "
          f"pooled {summary['kappa']['pooled_leaked_over_total']:.3f} vs global {GLOBAL_KAPPA}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--merge", default=None,
                        help='lineage groups, "New=Old1|Old2;New2=Old3"')
    parser.add_argument("--tag", default=None, help="suffix for the output file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=10.0,
                        help="nuclear-share specificity ratio against every other type")
    parser.add_argument("--min-counts", type=float, default=200.0,
                        help="minimum nuclear copies of a specific gene in its type (learn half)")
    parser.add_argument("--tau-um", type=float, default=DEFAULT_TAU_UM)
    parser.add_argument("--max-edge-um", type=float, default=DEFAULT_MAX_EDGE_UM)
    parser.add_argument("--tile-cells", type=int, default=4096)
    parser.add_argument("--sender-profile", choices=("total", "extranuclear"), default="total",
                        help="what a neighbour leaks: its total profile (document) or its rim's")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
