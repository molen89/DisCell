#!/usr/bin/env python3
"""Transcript-flux beta and per-cell kappa from extranuclear transcript geometry.

Pre-registered in devlog 2026-09-17, "Transcript-flux beta and per-cell kappa
from extranuclear transcript geometry (motivation)".

For every extranuclear q>=20 transcript assigned to cell ``i`` we measure the
distance to ``i``'s own nucleus and to the nearest *other* nucleus among ``i``'s
pruned Voronoi neighbours ``j``.  With ``s = d_own - d_j`` (positive = the
transcript sits closer to the neighbour's nucleus than to its own), the flux

    F_ij = sum over i's extranuclear transcripts whose nearest other nucleus is
           j, of a weight w(s; band) that is 0 well inside i and 1 well inside j

    beta_T_ij = F_ij / sum_j F_ij      kappa_i = sum_j F_ij / l_i

with ``l_i`` the cell's total assigned q20 counts.  ``kappa_i`` is an upper
bound on the leaked share: every transcript nearer a neighbour's nucleus than
its own is counted as possibly foreign.

Nothing here touches the model; the module writes a JSON report under
``data/datasets/<id>/experiments/``.

Usage::

    OMP_NUM_THREADS=8 python -m discell.experiments.transcript_flux \\
        --dataset xenium_prime_ovarian_cancer_ffpe [--variant full]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from discell import paths
from discell.model.prepare import DEFAULT_MAX_EDGE_UM, DEFAULT_TAU_UM

log = logging.getLogger("discell.experiments.transcript_flux")

#: Minimum Phred-style quality, matching ``applications.shared``.
MIN_QV = 20.0
#: Band half-widths (um) and weight forms swept for the sensitivity check (5).
BANDS = (2.0, 4.0, 8.0)
FORMS = ("hard", "ramp", "logistic")
PRIMARY = (4.0, "ramp")
#: Directed edges sampled for the per-edge gene-content check (1).
GENE_EDGE_SAMPLE = 200_000
#: Minimum banded transcripts on a sampled edge before its cosines are used.
MIN_BAND_COUNT = 10


def weights(s: np.ndarray, band: float, form: str) -> np.ndarray:
    """Weight of a transcript at signed bisector offset ``s`` (um).

    ``s = d_own - d_neighbour``: negative deep inside the host cell, 0 on the
    nucleus bisector, positive on the neighbour's side.  All three forms are
    monotone in ``s`` and live in [0, 1].
    """
    if form == "hard":
        return (s > 0).astype(np.float64)
    if form == "ramp":
        return np.clip((s + band) / (2.0 * band), 0.0, 1.0)
    if form == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(4.0 * s / band, -60.0, 60.0)))
    raise ValueError(f"unknown weight form {form!r}")


def segment_argmin(values: np.ndarray, starts: np.ndarray,
                   counts: np.ndarray) -> np.ndarray:
    """Index of the smallest ``values`` entry in each contiguous segment.

    ``starts``/``counts`` describe segments of a flat array; empty segments are
    not allowed.  ``np.minimum.reduceat`` gives the per-segment minimum in one
    pass and the first position attaining it is then taken, so ties resolve to
    the lowest neighbour index -- deterministic, and ties are measure-zero.
    """
    seg = np.repeat(np.arange(len(starts)), counts)
    best = np.minimum.reduceat(values, starts)
    hit = values <= best[seg]
    # first hit per segment
    order = np.flatnonzero(hit)
    first = np.ones(len(order), dtype=bool)
    first[1:] = seg[order[1:]] != seg[order[:-1]]
    return order[first]


@dataclass
class FluxAccumulator:
    """Directed-edge flux for the whole (band, form) grid, filled by streaming."""

    n_slots: int
    grid: Sequence[tuple[float, str]]
    flux: dict = field(init=False)
    n_transcripts: int = 0
    n_extranuclear: int = 0
    n_placed: int = 0

    def __post_init__(self) -> None:
        self.flux = {k: np.zeros(self.n_slots) for k in self.grid}

    def add(self, slot: np.ndarray, s: np.ndarray) -> None:
        self.n_placed += len(slot)
        for band, form in self.grid:
            self.flux[(band, form)] += np.bincount(
                slot, weights=weights(s, band, form), minlength=self.n_slots)


def adjacency_csr(edge_i: np.ndarray, edge_j: np.ndarray, n_cells: int,
                  eligible: np.ndarray) -> sp.csr_matrix:
    """Directed CSR over the pruned graph, rows = host i, cols = neighbour j.

    Only edges whose *both* endpoints carry a nucleus polygon are kept: a
    neighbour without a nucleus cannot be a distance reference.  Data holds the
    directed slot id so the streamer can address ``flux`` positionally.
    """
    keep = eligible[edge_i] & eligible[edge_j]
    src = np.concatenate([edge_i[keep], edge_j[keep]])
    dst = np.concatenate([edge_j[keep], edge_i[keep]])
    mat = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n_cells, n_cells))
    mat.data = np.arange(mat.nnz, dtype=np.int64)
    return mat


def beta_and_kappa(flux: np.ndarray, indptr: np.ndarray,
                   total_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(beta_T over the directed slots, kappa_i per cell)``.

    Cells whose slots carry no flux keep an all-zero beta row and kappa 0.
    """
    per_cell = np.add.reduceat(np.append(flux, 0.0), indptr[:-1])
    per_cell[np.diff(indptr) == 0] = 0.0
    denom = np.repeat(np.maximum(per_cell, 1e-12), np.diff(indptr))
    beta = flux / denom
    kappa = per_cell / np.maximum(total_counts, 1.0)
    return beta, kappa


def flux_from_transcripts(cell_of: np.ndarray, xy: np.ndarray,
                          nucleus_xy: np.ndarray, adjacency: sp.csr_matrix,
                          acc: FluxAccumulator,
                          gene_of: np.ndarray | None = None,
                          gene_sink: dict | None = None,
                          sampled: np.ndarray | None = None) -> None:
    """Assign one chunk of extranuclear transcripts to directed edge slots.

    *cell_of* are bundle row indices (already filtered to cells with a nucleus
    and at least one eligible neighbour); *xy* the transcript coordinates in um.
    """
    acc.n_extranuclear += len(cell_of)
    if len(cell_of) == 0:
        return
    indptr, indices, slots = adjacency.indptr, adjacency.indices, adjacency.data
    counts = np.diff(indptr)[cell_of]
    live = counts > 0
    if not live.all():
        cell_of, xy, counts = cell_of[live], xy[live], counts[live]
        if gene_of is not None:
            gene_of = gene_of[live]
    if len(cell_of) == 0:
        return
    d_own = np.linalg.norm(xy - nucleus_xy[cell_of], axis=1)
    starts = indptr[cell_of]
    total = int(counts.sum())
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    flat = np.repeat(starts, counts) + offsets
    nb = indices[flat]
    d_nb = np.linalg.norm(np.repeat(xy, counts, axis=0) - nucleus_xy[nb], axis=1)
    pick = segment_argmin(d_nb, np.cumsum(counts) - counts, counts)
    slot = slots[flat[pick]]
    s = d_own - d_nb[pick]
    acc.add(slot, s)
    if gene_sink is not None and gene_of is not None and sampled is not None:
        band, form = PRIMARY
        w = weights(s, band, form)
        on = sampled[slot]
        for key, take in (("band", (w > 0) & on), ("core", (w == 0) & on)):
            if take.any():
                gene_sink.setdefault(key, {}).setdefault("rows", []).append(slot[take])
                gene_sink[key].setdefault("cols", []).append(gene_of[take])
                gene_sink[key].setdefault(
                    "vals", []).append(w[take] if key == "band"
                                       else np.ones(int(take.sum())))


# --------------------------------------------------------------------------
# slide-side plumbing
# --------------------------------------------------------------------------


def _full_cell_ids(h5ad_path) -> np.ndarray:
    import h5py

    with h5py.File(h5ad_path, "r") as f:
        node = f["obs"][f["obs"].attrs["_index"]]
        if isinstance(node, h5py.Group):
            node = node["values"] if "values" in node else node["categories"]
        return node[:].astype(str)


def nucleus_centroids(xenium_dir, cell_ids: np.ndarray) -> np.ndarray:
    """Polygon-vertex mean per cell, in um, aligned to *cell_ids* (NaN if none).

    The vertex mean, not the area centroid: nuclei are near-convex blobs of ~25
    vertices, the two differ by well under a micron, and the mean is one pass.
    """
    import pandas as pd

    frame = pd.read_parquet(xenium_dir / "nucleus_boundaries.parquet",
                            columns=["cell_id", "vertex_x", "vertex_y"])
    frame["cell_id"] = frame["cell_id"].astype(str)
    means = frame.groupby("cell_id", sort=False)[["vertex_x", "vertex_y"]].mean()
    out = means.reindex(cell_ids).to_numpy(dtype=np.float64)
    log.info("nucleus centroids: %d of %d cells", int(np.isfinite(out[:, 0]).sum()),
             len(cell_ids))
    return out


def stream_slide(xenium_dir, cell_ids: np.ndarray, nucleus_xy: np.ndarray,
                 adjacency: sp.csr_matrix, gene_names: np.ndarray,
                 sampled: np.ndarray, batch_size: int = 4_000_000,
                 max_rows: int | None = None) -> tuple[FluxAccumulator, sp.csr_matrix]:
    import pandas as pd
    import pyarrow.parquet as pq

    cell_index = pd.Series(np.arange(len(cell_ids)), index=pd.Index(cell_ids))
    gene_index = pd.Series(np.arange(len(gene_names)), index=pd.Index(gene_names))
    grid = [(b, f) for b in BANDS for f in FORMS]
    acc = FluxAccumulator(adjacency.nnz, grid)
    sink: dict = {}
    has_nuc = np.isfinite(nucleus_xy[:, 0])
    parquet = pq.ParquetFile(xenium_dir / "transcripts.parquet")
    started = time.time()
    for batch in parquet.iter_batches(
            columns=["cell_id", "feature_name", "overlaps_nucleus", "qv",
                     "x_location", "y_location"],
            batch_size=batch_size):
        chunk = batch.to_pandas()
        acc.n_transcripts += len(chunk)
        chunk = chunk[(chunk["overlaps_nucleus"] == 0) & (chunk["qv"] >= MIN_QV)]
        if chunk.empty:
            continue
        rows = cell_index.reindex(chunk["cell_id"].astype(str)).to_numpy()
        keep = np.isfinite(rows)
        rows = rows[keep].astype(np.int64)
        keep2 = has_nuc[rows]
        rows = rows[keep2]
        sub = chunk[keep].iloc[keep2]
        genes = gene_index.reindex(sub["feature_name"].astype(str)).to_numpy()
        genes = np.where(np.isfinite(genes), genes, -1).astype(np.int64)
        xy = np.column_stack([sub["x_location"].to_numpy(dtype=np.float64),
                              sub["y_location"].to_numpy(dtype=np.float64)])
        ok = genes >= 0
        flux_from_transcripts(rows[ok], xy[ok], nucleus_xy, adjacency, acc,
                              gene_of=genes[ok], gene_sink=sink, sampled=sampled)
        if acc.n_transcripts % (10 * batch_size) < batch_size:
            log.info("  %.0fM transcripts, %.0fM placed, %.0f s",
                     acc.n_transcripts / 1e6, acc.n_placed / 1e6,
                     time.time() - started)
        if max_rows is not None and acc.n_transcripts >= max_rows:
            log.warning("stopping early at %d rows (--max-transcripts)",
                        acc.n_transcripts)
            break
    def assemble(key):
        part = sink.get(key)
        if not part:
            return sp.csr_matrix((adjacency.nnz, len(gene_names)))
        return sp.coo_matrix(
            (np.concatenate(part["vals"]),
             (np.concatenate(part["rows"]), np.concatenate(part["cols"]))),
            shape=(adjacency.nnz, len(gene_names))).tocsr()

    log.info("stream done in %.0f s: %d transcripts, %d extranuclear q20 placed",
             time.time() - started, acc.n_transcripts, acc.n_placed)
    return acc, assemble("band"), assemble("core")


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return None
    value = spearmanr(x[ok], y[ok]).statistic
    return None if not np.isfinite(value) else float(value)


def _quantiles(values: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.percentile(v, q)) for q in qs}


def edge_panel(band_genes: sp.csr_matrix, adjacency: sp.csr_matrix,
               type_index: np.ndarray, rho: np.ndarray,
               core_genes: sp.csr_matrix | None = None) -> dict:
    """Per-edge arrays for the gene-content check and its power analysis.

    One place decides which sampled directed edges are live, what their host
    and neighbour types are, and what their banded gene vectors look like, so
    the check and the planted-admixture curve are scored on exactly the same
    edges with exactly the same counts.
    """
    n = np.asarray(band_genes.sum(axis=1)).ravel()
    live = np.flatnonzero(n >= MIN_BAND_COUNT)
    host = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    order = np.argsort(adjacency.data)          # slot id -> position
    rho_n = rho / np.maximum(np.linalg.norm(rho, axis=1, keepdims=True), 1e-12)
    panel = {"live": live, "rho_n": rho_n,
             "ti": type_index[host[order][live]],
             "tj": type_index[adjacency.indices[order][live]],
             "n": n[live]}
    if len(live) == 0:
        return panel
    vec = band_genes[live]
    norms = sp.linalg.norm(vec, axis=1)
    panel["vec"], panel["norms"] = vec, norms

    def cos_to(types):
        return np.asarray(vec.multiply(rho_n[types]).sum(axis=1)).ravel() \
            / np.maximum(norms, 1e-12)

    panel["cos_to"] = cos_to
    panel["c_own"], panel["c_nb"] = cos_to(panel["ti"]), cos_to(panel["tj"])
    panel["homo"] = panel["ti"] == panel["tj"]
    panel["core_gap"] = None
    if core_genes is not None:
        cvec = core_genes[live]
        cnorm = sp.linalg.norm(cvec, axis=1)
        enough = np.asarray(cvec.sum(axis=1)).ravel() >= MIN_BAND_COUNT
        core_nb = np.asarray(cvec.multiply(rho_n[panel["tj"]]).sum(axis=1)).ravel() \
            / np.maximum(cnorm, 1e-12)
        panel["core_gap"] = np.where(enough, panel["c_nb"] - core_nb, np.nan)
    return panel


def gene_content_check(band_genes: sp.csr_matrix, adjacency: sp.csr_matrix,
                       type_index: np.ndarray, rho: np.ndarray,
                       seed: int = 0,
                       core_genes: sp.csr_matrix | None = None,
                       panel: dict | None = None) -> dict:
    """Check (1): does the banded content look like the neighbour's type?

    Per sampled directed edge (host i, neighbour j) with at least
    ``MIN_BAND_COUNT`` banded transcripts, cosine of the banded gene vector to
    the host's type mean, to the neighbour's type mean, and to a random other
    type's mean (the null).  Split by homotypic / heterotypic.

    The excess is a *biased-low* estimator of the admixture: see
    :func:`admixture_power` for the curve that turns it into a share, and
    :func:`deflate_profiles` for the circularity that flattens it.
    """
    rng = np.random.default_rng(seed)
    if panel is None:
        panel = edge_panel(band_genes, adjacency, type_index, rho, core_genes)
    if len(panel["live"]) == 0:
        return {"n_edges": 0}
    ti, tj, homo = panel["ti"], panel["tj"], panel["homo"]
    c_own, c_nb = panel["c_own"], panel["c_nb"]
    k = rho.shape[0]
    c_rand = panel["cos_to"]((tj + rng.integers(1, k, size=len(tj))) % k)
    core_gap = panel["core_gap"]
    out = {"n_edges": int(len(panel["live"])), "n_homotypic": int(homo.sum()),
           "min_band_count": MIN_BAND_COUNT}
    for name, mask in (("heterotypic", ~homo), ("homotypic", homo)):
        if mask.sum() < 10:
            out[name] = None
            continue
        out[name] = {
            "n": int(mask.sum()),
            "cos_own": float(c_own[mask].mean()),
            "cos_neighbour": float(c_nb[mask].mean()),
            "cos_random_type": float(c_rand[mask].mean()),
            "excess_neighbour_minus_own": float((c_nb - c_own)[mask].mean()),
            "excess_random_minus_own": float((c_rand - c_own)[mask].mean()),
            "frac_neighbour_gt_own": float((c_nb > c_own)[mask].mean()),
            "median_band_count": float(np.median(panel["n"][mask])),
        }
        if core_gap is not None:
            sel = core_gap[mask]
            sel = sel[np.isfinite(sel)]
            out[name]["n_with_core"] = int(len(sel))
            # the within-cell control: band vs the same cell's deep (w = 0)
            # extranuclear transcripts, both scored against the neighbour type
            out[name]["band_minus_core_cos_to_neighbour"] = (
                float(sel.mean()) if len(sel) else None)
            out[name]["frac_band_gt_core"] = (
                float((sel > 0).mean()) if len(sel) else None)
    return out


# --------------------------------------------------------------------------
# power: what admixture could check (1) actually see?
# --------------------------------------------------------------------------


def simulate_band(n_per_edge: np.ndarray, ti: np.ndarray, tj: np.ndarray,
                  rho: np.ndarray, p: float,
                  rng: np.random.Generator) -> sp.csr_matrix:
    """Planted band content: ``(1 - p) rho_host + p rho_neighbour`` per edge.

    Multinomial draws at the *observed* per-edge transcript counts, so the
    simulated curve carries exactly the sampling noise the real edges have.
    Edges are grouped by (host type, neighbour type) so one cumulative
    distribution serves every edge of a pair.
    """
    n_genes = rho.shape[1]
    key = ti.astype(np.int64) * rho.shape[0] + tj
    rows = np.repeat(np.arange(len(n_per_edge)), n_per_edge)
    cols = np.empty(int(n_per_edge.sum()), dtype=np.int64)
    edge_key = np.repeat(key, n_per_edge)
    for value in np.unique(key):
        take = np.flatnonzero(edge_key == value)
        if len(take) == 0:
            continue
        a, b = divmod(int(value), rho.shape[0])
        mix = (1.0 - p) * rho[a] + p * rho[b]
        cdf = np.cumsum(mix)
        cols[take] = np.searchsorted(cdf, rng.random(len(take)) * cdf[-1])
    return sp.coo_matrix((np.ones(len(rows)), (rows, np.minimum(cols, n_genes - 1))),
                         shape=(len(n_per_edge), n_genes)).tocsr()


def _excess_of(vec: sp.csr_matrix, ti: np.ndarray, tj: np.ndarray,
               rho_n: np.ndarray) -> np.ndarray:
    norms = np.maximum(sp.linalg.norm(vec, axis=1), 1e-12)
    own = np.asarray(vec.multiply(rho_n[ti]).sum(axis=1)).ravel() / norms
    nb = np.asarray(vec.multiply(rho_n[tj]).sum(axis=1)).ravel() / norms
    return nb - own


def _boot_se(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return float(values[idx].mean(axis=1).std(ddof=1))


def admixture_power(panel: dict, rho: np.ndarray,
                    fractions: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20,
                                                  0.30, 0.50, 1.0),
                    seed: int = 0, n_boot: int = 200,
                    max_edges: int | None = None) -> dict:
    """Expected check-(1) excess as a function of a planted admixture ``p``.

    Heterotypic edges only, at the observed per-edge band counts and the
    observed pooled type profiles.  The returned curve says what excess the
    check *would* report if a share ``p`` of every band were genuinely the
    neighbour's -- i.e. whether the pre-registered bar is reachable at all.
    """
    rng = np.random.default_rng(seed)
    het = np.flatnonzero(~panel["homo"])
    if max_edges is not None and len(het) > max_edges:
        het = rng.choice(het, size=max_edges, replace=False)
    ti, tj = panel["ti"][het], panel["tj"][het]
    n = panel["n"][het].astype(np.int64)
    rho_n = panel["rho_n"]
    curve = []
    for p in fractions:
        vec = simulate_band(n, ti, tj, rho, float(p), rng)
        ex = _excess_of(vec, ti, tj, rho_n)
        nb_gt = float((ex > 0).mean())
        curve.append({"p": float(p), "excess_mean": float(ex.mean()),
                      "excess_se": _boot_se(ex, n_boot, rng),
                      "frac_neighbour_gt_own": nb_gt})
    observed = float((panel["c_nb"] - panel["c_own"])[~panel["homo"]].mean())
    obs_se = _boot_se((panel["c_nb"] - panel["c_own"])[~panel["homo"]], n_boot, rng)
    ps = np.array([c["p"] for c in curve])
    ex = np.array([c["excess_mean"] for c in curve])
    se = np.array([c["excess_se"] for c in curve])
    order = np.argsort(ex)
    implied = float(np.interp(observed, ex[order], ps[order]))
    lo = float(np.interp(observed - 2 * obs_se, ex[order], ps[order]))
    hi = float(np.interp(observed + 2 * obs_se, ex[order], ps[order]))
    # can the check tell p = 0.13 from p = 0 at these counts?
    at0 = float(np.interp(0.0, ps, ex))
    at13 = float(np.interp(0.13, ps, ex))
    sep = abs(at13 - at0) / max(np.sqrt(np.interp(0.0, ps, se) ** 2
                                        + np.interp(0.13, ps, se) ** 2), 1e-12)
    return {
        "n_heterotypic_edges": int(len(het)),
        "median_band_count": float(np.median(n)),
        "curve": curve,
        "observed_excess": observed, "observed_excess_se": obs_se,
        "observed_frac_neighbour_gt_own": float(
            (panel["c_nb"] > panel["c_own"])[~panel["homo"]].mean()),
        "implied_p": implied, "implied_p_2se": [lo, hi],
        "excess_at_p0": at0, "excess_at_p0.13": at13,
        "separation_sigma_p0.13_vs_p0": float(sep),
        "separable_at_p0.13": bool(sep > 2.0),
    }


def bootstrap_did(panel: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap CI over edges for the band-minus-core difference-in-differences.

    ``(heterotypic mean) - (homotypic mean)`` of the within-cell control.  The
    homotypic arm is the null: whatever generic sparsity effect makes a band
    look more profile-like than the same cell's core is present there too.
    """
    gap, homo = panel["core_gap"], panel["homo"]
    if gap is None:
        return {"n_heterotypic": 0, "n_homotypic": 0}
    rng = np.random.default_rng(seed)
    het_v = gap[(~homo) & np.isfinite(gap)]
    hom_v = gap[homo & np.isfinite(gap)]
    if len(het_v) < 10 or len(hom_v) < 10:
        return {"n_heterotypic": int(len(het_v)), "n_homotypic": int(len(hom_v))}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = (het_v[rng.integers(0, len(het_v), len(het_v))].mean()
                    - hom_v[rng.integers(0, len(hom_v), len(hom_v))].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n_heterotypic": int(len(het_v)), "n_homotypic": int(len(hom_v)),
            "heterotypic_mean": float(het_v.mean()),
            "homotypic_mean": float(hom_v.mean()),
            "difference_in_differences": float(het_v.mean() - hom_v.mean()),
            "ci95": [float(lo), float(hi)], "n_boot": n_boot,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def deflate_profiles(rho: np.ndarray, flux: np.ndarray, adjacency: sp.csr_matrix,
                     type_index: np.ndarray, p: float) -> np.ndarray:
    """Type profiles with an assumed admixture ``p`` of their neighbours removed.

    The pooled profile of type ``t`` is itself contaminated:
    ``rho_obs = (1 - p) rho_clean + p rho_influx``, with ``rho_influx`` the
    flux-weighted mean profile of what type ``t`` cells actually border.
    Inverting that (clipped at zero, renormalised) gives the profiles the
    check *should* have used, and re-scoring on them measures how much the
    circularity flattens the excess.
    """
    host = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    order = np.argsort(adjacency.data)
    ti, tj = type_index[host[order]], type_index[adjacency.indices[order]]
    k = rho.shape[0]
    table = np.zeros((k, k))
    np.add.at(table, (ti, tj), flux)
    table /= np.maximum(table.sum(axis=1, keepdims=True), 1e-12)
    influx = table @ rho
    clean = np.clip(rho - p * influx, 0.0, None) / max(1.0 - p, 1e-6)
    return clean / np.maximum(clean.sum(axis=1, keepdims=True), 1e-12)


def measure(slide: dict, acc: FluxAccumulator, band_genes: sp.csr_matrix,
            core_genes: sp.csr_matrix | None = None, seed: int = 0) -> dict:
    adjacency = slide["adjacency"]
    indptr = adjacency.indptr
    total = slide["total_counts"]
    type_index, type_names = slide["type_index"], slide["type_names"]
    host = np.repeat(np.arange(adjacency.shape[0]), np.diff(indptr))
    order = np.argsort(adjacency.data)
    slot_host, slot_nb = host[order], adjacency.indices[order]

    grid = {}
    for key, flux in acc.flux.items():
        beta, kappa = beta_and_kappa(flux, indptr, total)
        band, form = key
        connected = np.diff(indptr) > 0
        entry = {
            "band_um": band, "weight": form,
            "kappa_quantiles": _quantiles(kappa[connected]),
            "kappa_mean": float(kappa[connected].mean()),
            "kappa_pooled": float(flux.sum() / max(total[connected].sum(), 1.0)),
            "frac_kappa_above_0.9": float((kappa[connected] > 0.9).mean()),
            "frac_kappa_above_1": float((kappa[connected] > 1.0).mean()),
        }
        # check (3): beta_T against the face-length beta, on live slots
        live = flux > 0
        entry["spearman_beta_T_vs_beta_face"] = _spearman(
            beta[live], slide["beta_face"][live]) if live.sum() > 10 else None
        entry["n_live_slots"] = int(live.sum())
        grid[f"band{band:g}_{form}"] = entry
        if key == PRIMARY:
            primary_beta, primary_kappa = beta, kappa

    band, form = PRIMARY
    connected = np.diff(indptr) > 0
    per_type = {}
    for k, name in enumerate(type_names):
        mask = (type_index == k) & connected
        if mask.sum() < 20:
            continue
        per_type[str(name)] = {
            "n_cells": int(mask.sum()),
            "kappa_median": float(np.median(primary_kappa[mask])),
            "kappa_pooled": float(
                np.add.reduceat(np.append(acc.flux[PRIMARY], 0.0), indptr[:-1])[mask].sum()
                / max(total[mask].sum(), 1.0)),
        }
    anchor = slide.get("nuclear_fraction_by_type", {})
    shared = [t for t in per_type if t in anchor]
    check4 = {
        "kappa_quantiles": _quantiles(primary_kappa[connected]),
        "frac_above_0.9": float((primary_kappa[connected] > 0.9).mean()),
        "by_type": per_type,
        "n_types_with_anchor": len(shared),
        "spearman_kappa_vs_nuclear_fraction": _spearman(
            np.array([per_type[t]["kappa_median"] for t in shared]),
            np.array([anchor[t] for t in shared])) if len(shared) >= 4 else None,
    }
    return {
        "primary": {"band_um": band, "weight": form},
        "counts": {"transcripts_read": acc.n_transcripts,
                   "extranuclear_q20_considered": acc.n_extranuclear,
                   "placed_on_an_edge": acc.n_placed,
                   "cells_connected": int(connected.sum()),
                   "directed_slots": int(adjacency.nnz)},
        "check1_gene_content": gene_content_check(band_genes, adjacency,
                                                  type_index, slide["rho"], seed,
                                                  core_genes=core_genes),
        "check3_beta": {"spearman_beta_T_vs_beta_face":
                        grid[f"band{band:g}_{form}"]["spearman_beta_T_vs_beta_face"],
                        "n_live_slots": grid[f"band{band:g}_{form}"]["n_live_slots"]},
        "check4_kappa": check4,
        "check5_sensitivity": grid,
        "per_type_kappa_median": {t: v["kappa_median"] for t, v in per_type.items()},
    }


def power_block(slide: dict, acc: FluxAccumulator, band_genes: sp.csr_matrix,
                core_genes: sp.csr_matrix, seed: int = 0, n_boot: int = 1000,
                max_power_edges: int | None = None) -> dict:
    """The three follow-up measurements on check (1), on this slide's own edges."""
    adjacency, rho = slide["adjacency"], slide["rho"]
    type_index = slide["type_index"]
    panel = edge_panel(band_genes, adjacency, type_index, rho, core_genes)
    power = admixture_power(panel, rho, seed=seed,
                            n_boot=min(n_boot, 200), max_edges=max_power_edges)
    did = bootstrap_did(panel, n_boot=n_boot, seed=seed)

    # circularity: re-score the observed bands on profiles with the implied
    # admixture removed, and re-plant at p = 0.13 on the same deflated basis
    deflation = {}
    for p_assumed in (0.13,):
        clean = deflate_profiles(rho, acc.flux[PRIMARY], adjacency, type_index,
                                 p_assumed)
        clean_n = clean / np.maximum(np.linalg.norm(clean, axis=1, keepdims=True), 1e-12)
        het = ~panel["homo"]
        obs = _excess_of(panel["vec"], panel["ti"], panel["tj"], clean_n)[het]
        rng = np.random.default_rng(seed + 1)
        planted = _excess_of(
            simulate_band(panel["n"][het].astype(np.int64), panel["ti"][het],
                          panel["tj"][het], rho, p_assumed, rng),
            panel["ti"][het], panel["tj"][het], clean_n)
        deflation[f"p{p_assumed:g}"] = {
            "observed_excess_on_deflated_profiles": float(obs.mean()),
            "observed_excess_on_pooled_profiles": power["observed_excess"],
            "attenuation": float(obs.mean() - power["observed_excess"]),
            "planted_excess_on_deflated_profiles": float(planted.mean()),
            "planted_excess_on_pooled_profiles": float(
                np.interp(p_assumed, [c["p"] for c in power["curve"]],
                          [c["excess_mean"] for c in power["curve"]])),
        }
    return {"admixture_power": power, "band_minus_core_did": did,
            "profile_deflation": deflation}


def load_slide(dataset: str, variant: str, label_key: str | None,
               max_edge_um: float, tau_um: float) -> dict:
    from pathlib import Path

    from discell.data.loader import CellGraphDataset
    from discell.data.priors import smoothing_weights

    ds = paths.dataset(dataset)
    opened = CellGraphDataset.from_dataset(ds, variant, label_key=label_key)
    xenium_dir = Path(str(opened.adata.uns["xenium_dir"]))
    face = opened._pull_edges("shared_wall_um").astype(np.float64)
    dist = opened._pull_edges("centroid_dist_um").astype(np.float64)
    keep = dist <= max_edge_um
    ei, ej = opened.edge_i[keep], opened.edge_j[keep]
    n = opened.adata.n_obs
    nucleus_xy = nucleus_centroids(xenium_dir, opened.cell_ids)
    eligible = np.isfinite(nucleus_xy[:, 0])
    adjacency = adjacency_csr(ei, ej, n, eligible)

    # the face-length beta on exactly the same directed slots, for check (3)
    bw = smoothing_weights(face[keep], dist[keep], ei, ej, n, tau_um)
    face_mat = sp.csr_matrix(
        (np.concatenate([bw[:, 0], bw[:, 1]]),
         (np.concatenate([ei, ej]), np.concatenate([ej, ei]))), shape=(n, n))
    rows = np.repeat(np.arange(n), np.diff(adjacency.indptr))
    beta_face = np.zeros(adjacency.nnz)
    beta_face[adjacency.data] = np.asarray(
        face_mat[rows, adjacency.indices]).ravel()

    counts = opened.counts.tocsr()
    total = np.asarray(counts.sum(axis=1)).ravel()
    k = opened.n_types
    rho = np.zeros((k, counts.shape[1]))
    for g in range(k):
        rows = np.flatnonzero(opened.type_index == g)
        if len(rows):
            rho[g] = np.asarray(counts[rows].sum(axis=0)).ravel()
    rho = rho / np.maximum(rho.sum(axis=1, keepdims=True), 1e-12)

    nuc_pooled = None
    nuc_path = ds.root / "qc" / "nuclear_counts.npz"
    if nuc_path.exists():
        import pandas as pd

        nuclear = sp.load_npz(nuc_path).tocsr()
        if nuclear.shape[0] != opened.adata.n_obs:
            rows_n = pd.Index(_full_cell_ids(ds.bundle_dir / "full.h5ad")
                              ).get_indexer(opened.cell_ids)
            if (rows_n < 0).any():
                raise ValueError("cells missing from full.h5ad")
            nuclear = nuclear[rows_n]
        ind = sp.csr_matrix(
            (np.ones(n), (opened.type_index, np.arange(n))), shape=(k, n))
        nuc_pooled = np.asarray((ind @ nuclear).todense(), dtype=np.float64)
        tot_pooled = np.asarray((ind @ counts).todense(), dtype=np.float64)

    anchor = {}
    summary_path = ds.root / "qc" / "nuclear_summary.json"
    if summary_path.exists():
        by_type = json.loads(summary_path.read_text()).get("by_type", {})
        anchor = {t: v["nuclear_fraction_pooled"] for t, v in by_type.items()}
    return {"adjacency": adjacency, "beta_face": beta_face,
            "total_counts": total, "type_index": opened.type_index,
            "type_names": np.asarray([str(t) for t in opened.type_names]),
            "rho": rho, "cell_ids": opened.cell_ids, "nucleus_xy": nucleus_xy,
            "gene_names": np.asarray([str(g) for g in opened.adata.var_names]),
            "xenium_dir": xenium_dir, "n_cells": n,
            "nuclear_fraction_by_type": anchor,
            "n_edges_pruned": int(keep.sum()),
            "n_cells_with_nucleus": int(eligible.sum()),
            "positions_um": np.asarray(opened.positions_um, dtype=np.float64),
            "nuc_pooled": nuc_pooled,
            "tot_pooled": tot_pooled if nuc_pooled is not None else None}


def run(args: argparse.Namespace) -> dict:
    slide = load_slide(args.dataset, args.variant, args.label_key,
                       args.max_edge_um, args.tau_um)
    rng = np.random.default_rng(args.seed)
    sampled = np.zeros(slide["adjacency"].nnz, dtype=bool)
    pick = rng.choice(slide["adjacency"].nnz,
                      size=min(GENE_EDGE_SAMPLE, slide["adjacency"].nnz),
                      replace=False)
    sampled[pick] = True
    acc, band_genes, core_genes = stream_slide(
        slide["xenium_dir"], slide["cell_ids"], slide["nucleus_xy"],
        slide["adjacency"], slide["gene_names"], sampled,
        max_rows=args.max_transcripts)
    report = measure(slide, acc, band_genes, core_genes, seed=args.seed)
    if args.power:
        report["power"] = power_block(slide, acc, band_genes, core_genes,
                                      seed=args.seed, n_boot=args.n_boot,
                                      max_power_edges=args.max_power_edges)
    report["dataset"] = args.dataset
    report["variant"] = args.variant
    report["parameters"] = {
        "min_qv": MIN_QV, "max_edge_um": args.max_edge_um, "tau_um": args.tau_um,
        "bands_um": list(BANDS), "weight_forms": list(FORMS),
        "nucleus_reference": "polygon vertex mean (centroid)",
        "gene_edge_sample": int(sampled.sum()),
        "seed": args.seed,
        "n_cells": slide["n_cells"],
        "n_cells_with_nucleus": slide["n_cells_with_nucleus"],
        "n_edges_pruned": slide["n_edges_pruned"],
    }
    out_dir = paths.dataset(args.dataset).root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.variant
    out = out_dir / f"transcript_flux_{tag}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    log.info("wrote %s", out)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-edge-um", type=float, default=DEFAULT_MAX_EDGE_UM)
    parser.add_argument("--tau-um", type=float, default=DEFAULT_TAU_UM)
    parser.add_argument("--power", action="store_true",
                        help="add the check-(1) admixture power curve, the "
                             "difference-in-differences bootstrap and the "
                             "profile-deflation estimate")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--max-power-edges", type=int, default=None)
    parser.add_argument("--decisive", action="store_true",
                        help="run the three decisive tests (todo 5.4-5.7) and "
                             "write transcript_flux_decisive_<tag>.json instead")
    parser.add_argument("--gene-edge-sample", type=int, default=GENE_EDGE_SAMPLE)
    parser.add_argument("--max-did-edges", type=int, default=60_000)
    parser.add_argument("--max-transcripts", type=int, default=None,
                        help="debug: stop after roughly this many rows")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    (run_decisive if args.decisive else run)(args)
    return 0




# --------------------------------------------------------------------------
# the three decisive tests (devlog 2026-09-17, todo 5.4-5.7)
# --------------------------------------------------------------------------

#: 5.4 -- specificity ratio and floor handed to ``leak_meter.specific_genes``.
SPECIFIC_RATIO = 3.0
SPECIFIC_MIN_COUNTS = 200.0
#: 5.4 -- top-N genes per type by nuclear-share fold change, so no type is empty.
SPECIFIC_TOP_N = 25
#: 5.5 -- side of the square tile the nucleus shuffle permutes inside (um).
SHUFFLE_TILE_UM = 200.0
#: 5.6 -- banded transcripts a heterotypic edge needs before it is scored.
DECISIVE_MIN_BAND = 30
#: the p grid the planted curves are evaluated on.
P_GRID = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
DID_P_GRID = (0.0, 0.05, 0.13, 0.30)


def polygon_centroids(xenium_dir, cell_ids: np.ndarray,
                      filename: str = "cell_boundaries.parquet") -> np.ndarray:
    """Vertex mean of each cell's own segmentation polygon, aligned to *cell_ids*."""
    import pandas as pd

    frame = pd.read_parquet(xenium_dir / filename,
                            columns=["cell_id", "vertex_x", "vertex_y"])
    frame["cell_id"] = frame["cell_id"].astype(str)
    means = frame.groupby("cell_id", sort=False)[["vertex_x", "vertex_y"]].mean()
    return means.reindex(cell_ids).to_numpy(dtype=np.float64)


def tile_shuffled(nucleus_xy: np.ndarray, positions_um: np.ndarray,
                  eligible: np.ndarray, seed: int,
                  tile_um: float = SHUFFLE_TILE_UM,
                  mode: str = "absolute") -> np.ndarray:
    """Nucleus references with the cell-to-nucleus pairing destroyed in tiles.

    ``mode="absolute"`` is the pre-registered control: within each ``tile_um``
    square the nucleus *positions* are permuted among the eligible cells, so the
    nucleus point field of the tile is preserved exactly and only which cell owns
    which nucleus changes.  ``mode="offset"`` permutes the nucleus-minus-polygon
    offset instead, which keeps each cell's nucleus inside its own neighbourhood
    and isolates the off-centring direction alone; it is a diagnostic, not the
    pre-registered null.
    """
    rng = np.random.default_rng(seed)
    out = nucleus_xy.copy()
    if mode == "offset":
        source = nucleus_xy - positions_um
    elif mode == "absolute":
        source = nucleus_xy
    else:
        raise ValueError(f"unknown shuffle mode {mode!r}")
    key = np.floor(positions_um / float(tile_um)).astype(np.int64)
    flat = key[:, 0] * 1_000_003 + key[:, 1]
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    bounds = np.flatnonzero(np.diff(flat_sorted)) + 1
    for part in np.split(order, bounds):
        rows = part[eligible[part]]
        if len(rows) < 2:
            continue
        perm = rows[rng.permutation(len(rows))]
        out[rows] = (source[perm] + positions_um[rows] if mode == "offset"
                     else source[perm])
    return out


def stream_references(xenium_dir, cell_ids: np.ndarray,
                      references: dict[str, np.ndarray],
                      adjacency: sp.csr_matrix, gene_names: np.ndarray,
                      sampled: np.ndarray, type_index: np.ndarray, n_types: int,
                      primary_reference: str = "nucleus",
                      batch_size: int = 4_000_000,
                      max_rows: int | None = None) -> dict:
    """One pass over ``transcripts.parquet`` filling every reference at once.

    Returns ``{"acc": {name: FluxAccumulator}, "band": csr, "core": csr,
    "extranuclear_pooled": (K, G)}``.  The gene sink and the pooled extranuclear
    profile are taken from *primary_reference* only; every reference sees exactly
    the same admitted transcripts, so the kappa distributions are comparable
    cell by cell.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    cell_index = pd.Series(np.arange(len(cell_ids)), index=pd.Index(cell_ids))
    gene_index = pd.Series(np.arange(len(gene_names)), index=pd.Index(gene_names))
    accs = {name: FluxAccumulator(adjacency.nnz, [PRIMARY]) for name in references}
    sink: dict = {}
    extra = np.zeros((n_types, len(gene_names)))
    has_nuc = np.isfinite(references[primary_reference][:, 0])
    for name, ref in references.items():
        has_nuc &= np.isfinite(ref[:, 0])
    parquet = pq.ParquetFile(xenium_dir / "transcripts.parquet")
    started = time.time()
    for batch in parquet.iter_batches(
            columns=["cell_id", "feature_name", "overlaps_nucleus", "qv",
                     "x_location", "y_location"],
            batch_size=batch_size):
        chunk = batch.to_pandas()
        for acc in accs.values():
            acc.n_transcripts += len(chunk)
        chunk = chunk[(chunk["overlaps_nucleus"] == 0) & (chunk["qv"] >= MIN_QV)]
        if chunk.empty:
            continue
        rows = cell_index.reindex(chunk["cell_id"].astype(str)).to_numpy()
        keep = np.isfinite(rows)
        rows = rows[keep].astype(np.int64)
        keep2 = has_nuc[rows]
        rows = rows[keep2]
        sub = chunk[keep].iloc[keep2]
        genes = gene_index.reindex(sub["feature_name"].astype(str)).to_numpy()
        genes = np.where(np.isfinite(genes), genes, -1).astype(np.int64)
        xy = np.column_stack([sub["x_location"].to_numpy(dtype=np.float64),
                              sub["y_location"].to_numpy(dtype=np.float64)])
        ok = genes >= 0
        rows, xy, genes = rows[ok], xy[ok], genes[ok]
        np.add.at(extra, (type_index[rows], genes), 1.0)
        for name, ref in references.items():
            is_primary = name == primary_reference
            flux_from_transcripts(
                rows, xy, ref, adjacency, accs[name], gene_of=genes,
                gene_sink=sink if is_primary else None,
                sampled=sampled if is_primary else None)
        first = accs[primary_reference]
        if first.n_transcripts % (10 * batch_size) < batch_size:
            log.info("  %.0fM transcripts, %.0fM placed, %.0f s",
                     first.n_transcripts / 1e6, first.n_placed / 1e6,
                     time.time() - started)
        if max_rows is not None and first.n_transcripts >= max_rows:
            log.warning("stopping early at %d rows", first.n_transcripts)
            break

    def assemble(key):
        part = sink.get(key)
        if not part:
            return sp.csr_matrix((adjacency.nnz, len(gene_names)))
        return sp.coo_matrix(
            (np.concatenate(part["vals"]),
             (np.concatenate(part["rows"]), np.concatenate(part["cols"]))),
            shape=(adjacency.nnz, len(gene_names))).tocsr()

    log.info("decisive stream done in %.0f s", time.time() - started)
    return {"acc": accs, "band": assemble("band"), "core": assemble("core"),
            "extranuclear_pooled": extra}


# -- 5.4 -------------------------------------------------------------------


def specific_gene_set(nuc_pooled: np.ndarray, tot_pooled: np.ndarray,
                      ratio: float = SPECIFIC_RATIO,
                      min_counts: float = SPECIFIC_MIN_COUNTS,
                      top_n: int = SPECIFIC_TOP_N) -> tuple[np.ndarray, dict]:
    """Union of type-specific genes, with a top-N floor so no type is empty.

    ``leak_meter.specific_genes`` at *ratio* gives the strict set; every type is
    then topped up to at least *top_n* genes by nuclear-share fold change over
    the best rival type, so a type the strict rule leaves empty still contributes.
    """
    from discell.experiments.leak_meter import Profiles, specific_genes

    k = nuc_pooled.shape[0]
    prof = Profiles(nuc=nuc_pooled, tot=tot_pooled, n_cells=np.zeros(k))
    strict = specific_genes(prof, list(range(k)), ratio, min_counts)
    share = prof.nuc_share
    per_type, chosen = {}, set()
    for t in range(k):
        rivals = [u for u in range(k) if u != t]
        best = share[rivals].max(axis=0) if rivals else np.zeros(share.shape[1])
        fold = share[t] / np.maximum(best, 1e-12)
        fold = np.where(share[t] > 0, fold, 0.0)
        top = np.argsort(-fold)[:top_n]
        genes = np.union1d(strict[t], top[fold[top] > 1.0])
        per_type[t] = {"n_strict": int(len(strict[t])), "n_used": int(len(genes))}
        chosen.update(int(g) for g in genes)
    return np.array(sorted(chosen), dtype=np.int64), per_type


def restricted_panel(band_genes: sp.csr_matrix, adjacency: sp.csr_matrix,
                     type_index: np.ndarray, rho: np.ndarray,
                     genes: np.ndarray | None, min_count: int,
                     require_full: int | None = None,
                     rho_host: np.ndarray | None = None,
                     core_genes: sp.csr_matrix | None = None) -> dict:
    """Edge panel on a gene subset, with the host arm optionally a second profile.

    Same contract as :func:`edge_panel` but (a) restricted to *genes* columns,
    with the type profiles renormalised over that subset, (b) admitting an edge
    on the subset count ``min_count`` and, if given, the *full*-panel count
    ``require_full``, and (c) allowing ``rho_host != rho`` for 5.7.
    """
    full_n = np.asarray(band_genes.sum(axis=1)).ravel()
    sub = band_genes if genes is None else band_genes[:, genes]
    n = np.asarray(sub.sum(axis=1)).ravel()
    ok = n >= min_count
    if require_full is not None:
        ok &= full_n >= require_full
    live = np.flatnonzero(ok)
    host = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    order = np.argsort(adjacency.data)

    def _prof(mat):
        cut = mat if genes is None else mat[:, genes]
        return cut / np.maximum(cut.sum(axis=1, keepdims=True), 1e-12)

    rho_s = _prof(rho)
    rho_h = rho_s if rho_host is None else _prof(rho_host)
    unit = lambda m: m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    panel = {"live": live, "rho_sub": rho_s, "rho_host_sub": rho_h,
             "rho_n": unit(rho_s), "rho_host_n": unit(rho_h),
             "ti": type_index[host[order][live]],
             "tj": type_index[adjacency.indices[order][live]],
             "host_cell": host[order][live],
             "n": n[live], "n_full": full_n[live], "genes": genes}
    panel["homo"] = panel["ti"] == panel["tj"]
    if len(live) == 0:
        panel["vec"] = sub[live]
        panel["c_own"] = panel["c_nb"] = np.zeros(0)
        panel["core"] = None
        return panel
    vec = sub[live]
    norms = np.maximum(sp.linalg.norm(vec, axis=1), 1e-12)
    panel["vec"], panel["norms"] = vec, norms
    panel["c_own"] = np.asarray(
        vec.multiply(panel["rho_host_n"][panel["ti"]]).sum(axis=1)).ravel() / norms
    panel["c_nb"] = np.asarray(
        vec.multiply(panel["rho_n"][panel["tj"]]).sum(axis=1)).ravel() / norms
    panel["core"] = None
    if core_genes is not None:
        panel["core"] = (core_genes if genes is None else core_genes[:, genes])[live]
    return panel


def planted_curve(panel: dict, p_grid: Sequence[float] = P_GRID,
                  seed: int = 0, n_boot: int = 200,
                  max_edges: int | None = None) -> dict:
    """Excess and ``frac(neighbour > own)`` of a planted mixture, on *panel*'s edges.

    The mixture is ``(1 - p) rho_host + p rho_neighbour`` on the panel's gene
    subset, drawn multinomially at the observed per-edge subset counts, and
    scored with the panel's own two reference profiles -- so the curve inherits
    every asymmetry the observed statistic has, including the uncentred cosine.
    """
    rng = np.random.default_rng(seed)
    het = np.flatnonzero(~panel["homo"])
    if max_edges is not None and len(het) > max_edges:
        het = rng.choice(het, size=max_edges, replace=False)
    ti, tj = panel["ti"][het], panel["tj"][het]
    n = np.maximum(panel["n"][het].astype(np.int64), 1)
    rows = []
    for p in p_grid:
        vec = _mix_draw(n, ti, tj, panel["rho_host_sub"], panel["rho_sub"],
                        float(p), rng)
        norms = np.maximum(sp.linalg.norm(vec, axis=1), 1e-12)
        own = np.asarray(vec.multiply(panel["rho_host_n"][ti]).sum(axis=1)).ravel() / norms
        nb = np.asarray(vec.multiply(panel["rho_n"][tj]).sum(axis=1)).ravel() / norms
        ex = nb - own
        rows.append({"p": float(p), "excess_mean": float(ex.mean()),
                     "excess_se": _boot_se(ex, n_boot, rng),
                     "frac_neighbour_gt_own": float((ex > 0).mean())})
    return {"curve": rows, "n_edges": int(len(het)),
            "median_band_count": float(np.median(n))}


def _mix_draw(n_per_edge: np.ndarray, ti: np.ndarray, tj: np.ndarray,
              rho_host: np.ndarray, rho_nb: np.ndarray, p: float,
              rng: np.random.Generator) -> sp.csr_matrix:
    """Multinomial draws of ``(1 - p) rho_host[ti] + p rho_nb[tj]`` per edge."""
    k, n_genes = rho_nb.shape
    key = ti.astype(np.int64) * k + tj
    rows = np.repeat(np.arange(len(n_per_edge)), n_per_edge)
    cols = np.empty(int(n_per_edge.sum()), dtype=np.int64)
    edge_key = np.repeat(key, n_per_edge)
    for value in np.unique(key):
        take = np.flatnonzero(edge_key == value)
        if len(take) == 0:
            continue
        a, b = divmod(int(value), k)
        cdf = np.cumsum((1.0 - p) * rho_host[a] + p * rho_nb[b])
        cols[take] = np.searchsorted(cdf, rng.random(len(take)) * cdf[-1])
    return sp.coo_matrix((np.ones(len(rows)), (rows, np.minimum(cols, n_genes - 1))),
                         shape=(len(n_per_edge), n_genes)).tocsr()


def _cross(curve: Sequence[dict], field_name: str, target: float) -> float | None:
    """First ``p`` at which a monotone curve field reaches *target* (linear interp)."""
    ps = np.array([c["p"] for c in curve])
    vs = np.array([c[field_name] for c in curve])
    order = np.argsort(vs)
    if target < vs[order][0] or target > vs[order][-1]:
        return None
    return float(np.interp(target, vs[order], ps[order]))


def _implied(curve: Sequence[dict], field_name: str, observed: float,
             se: float) -> dict:
    ps = np.array([c["p"] for c in curve])
    vs = np.array([c[field_name] for c in curve])
    order = np.argsort(vs)
    f = lambda v: float(np.interp(v, vs[order], ps[order]))
    return {"observed": float(observed), "observed_se": float(se),
            "implied_p": f(observed), "implied_p_2se": [f(observed - 2 * se),
                                                        f(observed + 2 * se)]}


def test_54_content_power(panel: dict, seed: int = 0, n_boot: int = 200,
                          max_edges: int | None = None) -> dict:
    """5.4: is the content check decisive at p ~ 0.13 on type-specific genes?

    Bar (pre-registered): the zero-crossing of "neighbour > own" falls at
    p <= 0.15.  Reported both as the p where the mean excess crosses zero and as
    the p where the majority of edges flip, plus the implied p with 2-SE.

    Note, and it is not a tuning remark: for a two-profile mixture scored
    against those same two profiles the crossing sits at p = 0.5 *identically*.
    At p = 0.5 the mixture is symmetric under swapping host and neighbour, which
    maps the excess to its own negative, so its mean is exactly zero whatever
    the gene subset.  The gene subset can sharpen the curve -- and so the
    implied p and the p = 0.13 separation -- but it cannot move the crossing.
    """
    het = ~panel["homo"]
    if het.sum() < 50:
        return {"n_heterotypic_edges": int(het.sum()), "verdict": "too few edges"}
    power = planted_curve(panel, seed=seed, n_boot=n_boot, max_edges=max_edges)
    curve = power["curve"]
    ex = (panel["c_nb"] - panel["c_own"])[het]
    rng = np.random.default_rng(seed + 7)
    se = _boot_se(ex, n_boot, rng)
    frac = float((ex > 0).mean())
    frac_se = _boot_se((ex > 0).astype(float), n_boot, rng)
    ps = np.array([c["p"] for c in curve])
    ex_c = np.array([c["excess_mean"] for c in curve])
    se_c = np.array([c["excess_se"] for c in curve])
    sep = abs(np.interp(0.13, ps, ex_c) - np.interp(0.0, ps, ex_c)) / max(
        np.hypot(np.interp(0.13, ps, se_c), np.interp(0.0, ps, se_c)), 1e-12)
    cross_mean = _cross(curve, "excess_mean", 0.0)
    cross_frac = _cross(curve, "frac_neighbour_gt_own", 0.5)
    bar = min([c for c in (cross_mean, cross_frac) if c is not None], default=None)
    return {
        "n_genes": int(0 if panel["genes"] is None else len(panel["genes"])),
        "n_heterotypic_edges": int(het.sum()),
        "median_band_count_on_genes": float(np.median(panel["n"][het])),
        "curve": curve,
        "zero_crossing_p_mean_excess": cross_mean,
        "zero_crossing_p_frac_half": cross_frac,
        "bar_zero_crossing_le_0.15": (None if bar is None else bool(bar <= 0.15)),
        "separation_sigma_p0.13_vs_p0": float(sep),
        "separable_at_p0.13": bool(sep > 2.0),
        "excess": _implied(curve, "excess_mean", float(ex.mean()), se),
        "frac": _implied(curve, "frac_neighbour_gt_own", frac, frac_se),
    }


# -- 5.5 -------------------------------------------------------------------


def test_55_offcentring(slide: dict, accs: dict[str, FluxAccumulator]) -> dict:
    """5.5: how much of kappa is nucleus off-centring rather than influx?

    Bars: the corrected kappa (observed minus tile-shuffled) keeps a median
    above 0.02 and its per-type ordering agrees with the uncorrected one at
    Spearman >= 0.8.
    """
    indptr = slide["adjacency"].indptr
    total, type_index = slide["total_counts"], slide["type_index"]
    connected = np.diff(indptr) > 0
    kappa = {name: beta_and_kappa(acc.flux[PRIMARY], indptr, total)[1]
             for name, acc in accs.items()}
    kappa["corrected"] = kappa["nucleus"] - kappa["shuffled"]
    if "polygon" in kappa:
        kappa["polygon_corrected"] = kappa["nucleus"] - kappa["polygon"]
    out = {"distributions": {}, "by_type": {}, "spearman_vs_observed": {}}
    per_type = {}
    for name, v in kappa.items():
        out["distributions"][name] = {
            "quantiles": _quantiles(v[connected]),
            "mean": float(v[connected].mean()),
            "frac_above_0.9": float((v[connected] > 0.9).mean()),
            "frac_negative": float((v[connected] < 0).mean()),
        }
        table = {}
        for t, label in enumerate(slide["type_names"]):
            mask = (type_index == t) & connected
            if mask.sum() >= 20:
                table[str(label)] = float(np.median(v[mask]))
        per_type[name] = table
    out["by_type"] = per_type
    base = per_type["nucleus"]
    shared = sorted(base)
    for name, table in per_type.items():
        if name == "nucleus":
            continue
        common = [t for t in shared if t in table]
        out["spearman_vs_observed"][name] = _spearman(
            np.array([base[t] for t in common]),
            np.array([table[t] for t in common])) if len(common) >= 4 else None
    med = out["distributions"]["corrected"]["quantiles"]["50"]
    sp_corr = out["spearman_vs_observed"].get("corrected")
    out["bar_corrected_median_above_0.02"] = (None if med is None
                                              else bool(med > 0.02))
    out["bar_type_ordering_spearman_ge_0.8"] = (None if sp_corr is None
                                                else bool(sp_corr >= 0.8))
    return out


# -- 5.6 -------------------------------------------------------------------


def _decile_panel(panel: dict, rows: np.ndarray) -> dict:
    """The slice of *panel* a planted curve needs, over the given edge rows."""
    return {"homo": np.zeros(len(rows), dtype=bool), "ti": panel["ti"][rows],
            "tj": panel["tj"][rows], "n": panel["n"][rows],
            "rho_sub": panel["rho_sub"], "rho_host_sub": panel["rho_host_sub"],
            "rho_n": panel["rho_n"], "rho_host_n": panel["rho_host_n"]}


def test_56_per_edge(panel: dict, flux: np.ndarray, total_counts: np.ndarray,
                     curve: Sequence[dict], n_deciles: int = 10,
                     seed: int = 0, n_boot: int = 50) -> dict:
    """5.6: does the geometric flux share track the content-implied admixture?

    Heterotypic edges only.  The flux share ``F_ij / l_i`` is binned into
    deciles; within each decile the mean content excess is turned into an
    implied ``p`` through the 5.4 planted curve, which averages the multinomial
    noise that makes the per-edge excess useless on its own.

    Bars: Spearman over the deciles >= 0.5 and a calibration slope in [0.5, 2].

    The excess is strongly count-dependent, and the banded count itself rises
    with the flux share, so a single curve fitted at one median count maps the
    deciles through ten different biases.  Each decile therefore also gets its
    *own* planted curve, simulated at that decile's per-edge counts and its own
    type pairs; that count-matched inversion is the primary read and the
    single-curve one is reported beside it.
    """
    het = np.flatnonzero(~panel["homo"])
    if len(het) < 10 * n_deciles:
        return {"n_edges": int(len(het)), "verdict": "too few edges"}
    slots = panel["live"][het]
    share = flux[slots] / np.maximum(total_counts[panel["host_cell"][het]], 1.0)
    ex = (panel["c_nb"] - panel["c_own"])[het]
    edges = np.quantile(share, np.linspace(0, 1, n_deciles + 1))
    edges[-1] += 1e-12
    which = np.clip(np.searchsorted(edges, share, side="right") - 1, 0, n_deciles - 1)
    rows = []
    for d in range(n_deciles):
        take = which == d
        if take.sum() < 5:
            continue
        own = planted_curve(_decile_panel(panel, het[take]), seed=seed + d,
                            n_boot=n_boot)["curve"]
        rows.append({
            "decile": d, "n": int(take.sum()),
            "flux_share_mean": float(share[take].mean()),
            "excess_mean": float(ex[take].mean()),
            "implied_p": _cross(own, "excess_mean", float(ex[take].mean())),
            "implied_p_single_curve": _cross(curve, "excess_mean",
                                             float(ex[take].mean())),
            "median_band_count": float(np.median(panel["n_full"][het][take])),
            "median_band_count_on_genes": float(np.median(panel["n"][het][take])),
        })

    def _fit(field_name):
        usable = [r for r in rows if r[field_name] is not None]
        if len(usable) < 4:
            return None, None, 0
        x = np.array([r["flux_share_mean"] for r in usable])
        y = np.array([r[field_name] for r in usable])
        sl = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else None
        return _spearman(x, y), sl, len(usable)

    rho_s, slope, n_use = _fit("implied_p")
    rho_1, slope_1, _ = _fit("implied_p_single_curve")
    return {
        "n_edges": int(len(het)), "n_deciles_usable": n_use,
        "min_band_count": DECISIVE_MIN_BAND, "deciles": rows,
        "spearman_flux_share_vs_implied_p": rho_s,
        "calibration_slope": slope,
        "spearman_single_curve": rho_1, "calibration_slope_single_curve": slope_1,
        "bar_spearman_ge_0.5": None if rho_s is None else bool(rho_s >= 0.5),
        "bar_slope_in_0.5_2": (None if slope is None
                               else bool(0.5 <= slope <= 2.0)),
    }


# -- 5.7 -------------------------------------------------------------------


def test_57_moments(panel: dict, seed: int = 0, n_boot: int = 200,
                    max_edges: int | None = None) -> dict:
    """5.7 part one: do the two moments agree once the host arm is the rim?

    *panel* must carry ``rho_host`` = the pooled *extranuclear* profile of the
    host type -- the arm the band should be compared against, since a band is
    rim content either way.  Bar: the excess-implied and fraction-implied p
    overlap within their 2-SE intervals.
    """
    block = test_54_content_power(panel, seed=seed, n_boot=n_boot,
                                  max_edges=max_edges)
    if "excess" not in block:
        return block
    a, b = block["excess"]["implied_p_2se"], block["frac"]["implied_p_2se"]
    block["moments_overlap"] = bool(a[0] <= b[1] and b[0] <= a[1])
    block["bar_moments_agree_within_2se"] = block["moments_overlap"]
    return block


def did_power(panel: dict, p_grid: Sequence[float] = DID_P_GRID,
              seed: int = 0, n_boot: int = 200,
              max_edges: int | None = None) -> dict:
    """5.7 part two: the power curve the band-minus-core DiD never had.

    At each planted ``p`` the band is drawn from ``(1 - p) rho_host + p rho_nb``
    and the core from ``rho_host`` alone, both at the observed per-edge counts;
    the statistic is ``mean(gap | heterotypic) - mean(gap | homotypic)`` with
    ``gap = cos(band, rho_nb) - cos(core, rho_nb)``.  On homotypic edges the
    plant is inert by construction, which is exactly what makes that arm a null.
    """
    if panel.get("core") is None:
        return {"verdict": "no core matrix"}
    rng = np.random.default_rng(seed)
    core_n = np.asarray(panel["core"].sum(axis=1)).ravel().astype(np.int64)
    ok = (core_n >= MIN_BAND_COUNT) & (panel["n"] >= MIN_BAND_COUNT)
    idx = np.flatnonzero(ok)
    if max_edges is not None and len(idx) > max_edges:
        idx = rng.choice(idx, size=max_edges, replace=False)
    het = panel["homo"][idx] == False  # noqa: E712
    if het.sum() < 20 or (~het).sum() < 20:
        return {"n_heterotypic": int(het.sum()), "n_homotypic": int((~het).sum()),
                "verdict": "too few edges"}
    ti, tj = panel["ti"][idx], panel["tj"][idx]
    nb_n = panel["rho_n"][tj]

    def cos_to_nb(vec):
        norms = np.maximum(sp.linalg.norm(vec, axis=1), 1e-12)
        return np.asarray(vec.multiply(nb_n).sum(axis=1)).ravel() / norms

    rows = []
    for p in p_grid:
        band = _mix_draw(np.maximum(panel["n"][idx].astype(np.int64), 1), ti, tj,
                         panel["rho_host_sub"], panel["rho_sub"], float(p), rng)
        core = _mix_draw(np.maximum(core_n[idx], 1), ti, tj,
                         panel["rho_host_sub"], panel["rho_sub"], 0.0, rng)
        gap = cos_to_nb(band) - cos_to_nb(core)
        draws = np.empty(n_boot)
        h, m = gap[het], gap[~het]
        for b in range(n_boot):
            draws[b] = (h[rng.integers(0, len(h), len(h))].mean()
                        - m[rng.integers(0, len(m), len(m))].mean())
        rows.append({"p": float(p), "did": float(h.mean() - m.mean()),
                     "did_se": float(draws.std(ddof=1))})
    at0 = rows[0]
    sep = {r["p"]: abs(r["did"] - at0["did"])
           / max(np.hypot(r["did_se"], at0["did_se"]), 1e-12) for r in rows[1:]}
    return {"n_heterotypic": int(het.sum()), "n_homotypic": int((~het).sum()),
            "curve": rows, "separation_sigma_vs_p0": {str(k): float(v)
                                                      for k, v in sep.items()},
            "separable_at_p0.13": bool(sep.get(0.13, 0.0) > 2.0)}


# -- driver ----------------------------------------------------------------


def run_decisive(args: argparse.Namespace) -> dict:
    slide = load_slide(args.dataset, args.variant, args.label_key,
                       args.max_edge_um, args.tau_um)
    if slide["nuc_pooled"] is None:
        raise SystemExit("qc/nuclear_counts.npz is required for 5.4")
    rng = np.random.default_rng(args.seed)
    nnz = slide["adjacency"].nnz
    sampled = np.zeros(nnz, dtype=bool)
    sampled[rng.choice(nnz, size=min(args.gene_edge_sample, nnz),
                       replace=False)] = True
    eligible = np.isfinite(slide["nucleus_xy"][:, 0])
    poly = polygon_centroids(slide["xenium_dir"], slide["cell_ids"])
    references = {
        "nucleus": slide["nucleus_xy"],
        "shuffled": tile_shuffled(slide["nucleus_xy"], slide["positions_um"],
                                  eligible, args.seed, mode="absolute"),
        "shuffled_offset": tile_shuffled(slide["nucleus_xy"], poly, eligible,
                                         args.seed, mode="offset"),
        "polygon": poly,
    }
    stream = stream_references(
        slide["xenium_dir"], slide["cell_ids"], references, slide["adjacency"],
        slide["gene_names"], sampled, slide["type_index"],
        len(slide["type_names"]), max_rows=args.max_transcripts)
    band, core = stream["band"], stream["core"]
    rho = slide["rho"]
    extra = stream["extranuclear_pooled"]
    rho_extra = extra / np.maximum(extra.sum(axis=1, keepdims=True), 1e-12)

    genes, per_type_genes = specific_gene_set(slide["nuc_pooled"],
                                              slide["tot_pooled"])
    panel54 = restricted_panel(band, slide["adjacency"], slide["type_index"],
                               rho, genes, MIN_BAND_COUNT)
    t54 = test_54_content_power(panel54, seed=args.seed, n_boot=args.n_boot,
                                max_edges=args.max_power_edges)
    t54["genes_per_type"] = {str(slide["type_names"][t]): v
                             for t, v in per_type_genes.items()}
    t54["specificity"] = {"ratio": SPECIFIC_RATIO,
                          "min_nuclear_counts": SPECIFIC_MIN_COUNTS,
                          "top_n_per_type": SPECIFIC_TOP_N,
                          "n_genes_union": int(len(genes)),
                          "n_genes_panel": int(rho.shape[1])}

    t55 = test_55_offcentring(slide, stream["acc"])

    panel56 = restricted_panel(band, slide["adjacency"], slide["type_index"],
                               rho, genes, 1, require_full=DECISIVE_MIN_BAND)
    t56 = test_56_per_edge(panel56, stream["acc"]["nucleus"].flux[PRIMARY],
                           slide["total_counts"], t54.get("curve", []),
                           seed=args.seed)

    panel57 = restricted_panel(band, slide["adjacency"], slide["type_index"],
                               rho, None, MIN_BAND_COUNT, rho_host=rho_extra,
                               core_genes=core)
    t57 = test_57_moments(panel57, seed=args.seed, n_boot=args.n_boot,
                          max_edges=args.max_power_edges)
    t57["host_arm"] = "pooled extranuclear profile of the host type"
    t57["did_power"] = did_power(panel57, seed=args.seed, n_boot=args.n_boot,
                                 max_edges=args.max_did_edges)
    t57["observed_did"] = bootstrap_did(
        {"core_gap": (None if panel57.get("core") is None
                      else _band_minus_core(panel57)), "homo": panel57["homo"]},
        n_boot=args.n_boot, seed=args.seed)

    report = {
        "dataset": args.dataset, "variant": args.variant,
        "test_5_4_content_power": t54,
        "test_5_5_offcentring": t55,
        "test_5_6_per_edge_agreement": t56,
        "test_5_7_moments_and_did": t57,
        "counts": {"transcripts_read": stream["acc"]["nucleus"].n_transcripts,
                   "placed_on_an_edge": stream["acc"]["nucleus"].n_placed,
                   "directed_slots": int(nnz),
                   "gene_edge_sample": int(sampled.sum())},
        "parameters": {
            "min_qv": MIN_QV, "max_edge_um": args.max_edge_um,
            "tau_um": args.tau_um, "primary": {"band_um": PRIMARY[0],
                                               "weight": PRIMARY[1]},
            "shuffle_tile_um": SHUFFLE_TILE_UM,
            "decisive_min_band": DECISIVE_MIN_BAND,
            "seed": args.seed, "n_boot": args.n_boot,
            "n_cells": slide["n_cells"],
            "n_cells_with_nucleus": slide["n_cells_with_nucleus"],
            "n_cells_with_polygon": int(np.isfinite(poly[:, 0]).sum()),
            "n_edges_pruned": slide["n_edges_pruned"],
        },
    }
    out_dir = paths.dataset(args.dataset).root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"transcript_flux_decisive_{args.tag or args.variant}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))
    log.info("wrote %s", out)
    return report


def _band_minus_core(panel: dict) -> np.ndarray:
    """``cos(band, rho_nb) - cos(core, rho_nb)`` per edge, NaN where the core is thin."""
    core = panel["core"]
    nb_n = panel["rho_n"][panel["tj"]]
    cn = np.asarray(core.sum(axis=1)).ravel()
    norms = np.maximum(sp.linalg.norm(core, axis=1), 1e-12)
    cos_core = np.asarray(core.multiply(nb_n).sum(axis=1)).ravel() / norms
    return np.where(cn >= MIN_BAND_COUNT, panel["c_nb"] - cos_core, np.nan)


if __name__ == "__main__":
    raise SystemExit(main())
