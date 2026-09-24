#!/usr/bin/env python3
"""Review R12, Test 1: does the foreign transcript count track the receiver or the donor?

Pre-registered in devlog 2026-09-24, "Review R12: what the leakage term
assumes, tested (motivation)", Test 1.  Read-only; no model is fitted.

The leak mixture ``(1 - kappa) rho_i + kappa rho_bar_i`` puts ``kappa * l_i``
leaked counts into cell ``i``: the leaked amount scales with the *receiver's*
depth.  If what a cell gains from its neighbours were set by the neighbours,
its foreign count would instead track the face-weighted donor depth
``D_i = sum_j beta_ij l_j`` (``beta`` the model's shared-wall kernel from
``discell.model.prepare``), or the donor density ``sum_j beta_ij l_j / A_j``.

Per cell, ``F_i`` is the number of its extranuclear q20 transcripts that sit
nearer a neighbour's nucleus than its own -- the ``hard`` weight of
:mod:`discell.experiments.transcript_flux`, whose adjacency, nucleus
references and streaming are used unchanged.  Then, within cell type,

    log(1 + F_i) = a_type + b_r log l_i + b_d log D_i + e_i

reported as standardised coefficients (within-type SDs), partial R^2 of each
term, and the LMG (Shapley) split of the within-type R^2, with bootstrap CIs
over cells and over 200 um tiles.  ``F_i = kappa l_i`` predicts the
elasticities ``b_r = 1``, ``b_d = 0``.

"Dominates", fixed before the runs: the term's |standardised coefficient| is at
least ``DOMINANCE_RATIO`` times the other's, its partial R^2 at least
``DOMINANCE_PARTIAL`` times the other's, and the tile-bootstrap 95 % CI of
``|b_r^std| - |b_d^std|`` excludes zero on its side.  Anything else is "no clear
winner".  The LMG split is reported but not in the rule: it hands shared
variance out equally, so with the two depths correlated at ~0.8 even a planted
receiver-only world gives the receiver only ~0.69 of it.

Test 1b (devlog 2026-09-24, "R12 test 1b -- donor scaling at fixed area"):
the same outcome on log l_i, log D_i, log A_i and log sum_j beta_ij A_j
(``area_conditioned``), and on the density variant with l/A for receiver and
donor (``density_area_conditioned``).  The pre-registered quantity is the
unstandardised donor elasticity at fixed areas: donor scaling is ruled out on a
slide only if it is <= 0 with a tile-bootstrap 95 % CI upper bound below
``PRACTICAL_ELASTICITY``; see :func:`classify_1b`.

Usage::

    OMP_NUM_THREADS=8 uv run python -m discell.experiments.r12_depth_test \\
        --dataset xenium_prime_ovarian_cancer_ffpe
    uv run python -m discell.experiments.r12_depth_test --combine-1b \\
        xenium_prime_ovarian_cancer_ffpe xenium_prime_human_ovary_ff
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp

from discell import paths
from discell.experiments import transcript_flux as tf
from discell.model.prepare import DEFAULT_MAX_EDGE_UM, DEFAULT_TAU_UM

log = logging.getLogger("discell.experiments.r12_depth_test")

#: The model's kappa at the final configuration (the kappa*l_i line).
KAPPA_MODEL = 0.1
#: The literal foreign count; ``hard`` is band-invariant, the band is a key only.
FOREIGN = (4.0, "hard")
#: transcript_flux's primary weighted flux, the kappa_i of the 2026-09-17 entries.
WEIGHTED = tf.PRIMARY
#: Side of the square tiles the spatial bootstrap resamples (um).
TILE_UM = 200.0
N_BOOT = 1000
DOMINANCE_RATIO = 2.0
DOMINANCE_PARTIAL = 4.0
#: Equal-count bins for the figure's binned medians.
N_BINS = 40
#: Test 1b: a donor elasticity at or above this is "practically positive"
#: (a quarter of the receiver elasticity F = kappa l_i implies).
PRACTICAL_ELASTICITY = 0.25


# --------------------------------------------------------------------------
# within-group regression from sufficient statistics
# --------------------------------------------------------------------------


def moment_rows(columns: np.ndarray) -> np.ndarray:
    """Per-row ``[1, x, vec(x x^T)]`` of an ``(n, p)`` block.

    Summed over the cells of a group (optionally weighted) these are the
    sufficient statistics of every OLS fit on the block, so a bootstrap only
    re-sums them.
    """
    n, p = columns.shape
    outer = (columns[:, :, None] * columns[:, None, :]).reshape(n, p * p)
    return np.concatenate([np.ones((n, 1)), columns, outer], axis=1)


def indicator(groups: np.ndarray, n_groups: int) -> sp.csc_matrix:
    """``(n_groups, n)`` one-hot, CSC with one entry per column.

    One entry per column means ``data[k]`` belongs to cell ``k``, so a bootstrap
    reweights cells by overwriting ``data``.
    """
    n = len(groups)
    return sp.csc_matrix((np.ones(n), (groups, np.arange(n))), shape=(n_groups, n))


def _cross_products(stats: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """``(within-group, total)`` centred cross-product matrices, ``p x p``."""
    w = stats[:, 0]
    s = stats[:, 1:1 + p]
    ss = stats[:, 1 + p:].reshape(-1, p, p)
    live = w > 0
    within = (ss[live] - s[live, :, None] * s[live, None, :]
              / w[live, None, None]).sum(axis=0)
    grand = s.sum(axis=0)
    total = ss.sum(axis=0) - np.outer(grand, grand) / w.sum()
    return within, total


def fe_fit(stats: np.ndarray, p: int) -> dict:
    """Within-group OLS of column 0 on columns ``1 .. p-1``.

    Returns the coefficients, the standardised coefficients (within-group SDs),
    partial R^2 of each regressor given the others and the groups, the LMG
    (Shapley) split of the within-group R^2, and the R^2 of the groups alone and
    of the full model.
    """
    within, total = _cross_products(stats, p)
    syy = within[0, 0]
    k = p - 1
    sse_cache: dict[tuple, float] = {}

    def sse(subset: tuple) -> float:
        if subset not in sse_cache:
            if not subset:
                sse_cache[subset] = syy
            else:
                idx = np.asarray(subset) + 1
                c = within[idx, 0]
                sse_cache[subset] = float(
                    syy - c @ np.linalg.solve(within[np.ix_(idx, idx)], c))
        return sse_cache[subset]

    full = tuple(range(k))
    coef = np.linalg.solve(within[1:, 1:], within[1:, 0])
    r2w = lambda subset: 1.0 - sse(subset) / syy
    partial = [(sse(tuple(m for m in full if m != j)) - sse(full))
               / sse(tuple(m for m in full if m != j)) for j in full]
    lmg = []
    for j in full:
        others = [m for m in full if m != j]
        value = 0.0
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            for subset in itertools.combinations(others, size):
                value += weight * (r2w(tuple(sorted(subset + (j,)))) - r2w(subset))
        lmg.append(value)
    std = coef * np.sqrt(np.diag(within)[1:] / syy)
    return {"coef": coef.tolist(), "std_coef": std.tolist(),
            "partial_r2": partial, "lmg": lmg,
            "r2_within": r2w(full),
            "r2_groups_only": float(1.0 - syy / total[0, 0]),
            "r2_total": float(1.0 - sse(full) / total[0, 0]),
            "within_corr": (within[1:, 1:] / np.sqrt(np.outer(
                np.diag(within)[1:], np.diag(within)[1:]))).tolist()}


def line_fit(stats: np.ndarray) -> dict:
    """Pooled ``y = a + b x`` (one group, raw scale) plus the through-origin slope."""
    w, sy, sx = stats[0, 0], stats[0, 1], stats[0, 2]
    syy, syx, sxx = stats[0, 3], stats[0, 4], stats[0, 6]
    cxx = sxx - sx * sx / w
    cxy = syx - sx * sy / w
    cyy = syy - sy * sy / w
    slope = cxy / cxx
    return {"slope": float(slope), "intercept": float(sy / w - slope * sx / w),
            "r2": float(cxy * cxy / (cxx * cyy)),
            "slope_through_origin": float(syx / sxx),
            "ratio_of_sums": float(sy / sx)}


def _flatten(fit: dict, names: Sequence[str]) -> dict[str, float]:
    out = {}
    for key in ("coef", "std_coef", "partial_r2", "lmg"):
        for name, value in zip(names, fit[key]):
            out[f"{key}.{name}"] = float(value)
    for key in ("r2_within", "r2_total", "r2_groups_only"):
        out[key] = float(fit[key])
    if len(names) >= 2:
        out["margin_abs_std.r_minus_d"] = (abs(fit["std_coef"][0])
                                           - abs(fit["std_coef"][1]))
        pair = fit["lmg"][0] + fit["lmg"][1]
        out["lmg_share.first_of_two"] = fit["lmg"][0] / pair if pair > 0 else float("nan")
    return out


def bootstrap(rows: np.ndarray, groups: np.ndarray, n_groups: int,
              tiles: np.ndarray, summary: Callable[[np.ndarray], dict],
              n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Percentile 95 % CIs of every entry of ``summary(stats)``.

    ``cell``: cells resampled with replacement (multinomial counts).  ``tile``:
    whole ``TILE_UM`` tiles resampled, the scheme that respects the spatial
    dependence of neighbouring cells, so the one the verdict reads.
    """
    rng = np.random.default_rng(seed)
    n = len(groups)
    out = {}
    g = indicator(groups, n_groups)
    n_tiles = int(tiles.max()) + 1
    per_tile = (indicator(tiles * n_groups + groups, n_tiles * n_groups) @ rows
                ).reshape(n_tiles, n_groups, rows.shape[1])
    for scheme in ("cell", "tile"):
        draws: dict[str, list] = {}
        for _ in range(n_boot):
            if scheme == "cell":
                g.data = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
                stats = g @ rows
            else:
                mult = np.bincount(rng.integers(0, n_tiles, n_tiles),
                                   minlength=n_tiles).astype(np.float64)
                stats = np.tensordot(mult, per_tile, axes=1)
            for key, value in summary(stats).items():
                draws.setdefault(key, []).append(value)
        out[scheme] = {key: {"ci95": [float(np.nanpercentile(v, 2.5)),
                                      float(np.nanpercentile(v, 97.5))],
                             "se": float(np.nanstd(v, ddof=1))}
                       for key, v in draws.items()}
    out["n_boot"] = n_boot
    out["n_tiles"] = n_tiles
    return out


def verdict(point: dict[str, float], ci: dict, names: Sequence[str]) -> str:
    """The pre-set rule, on the first two regressors (receiver, donor)."""
    r, d = (abs(point[f"std_coef.{n}"]) for n in names[:2])
    pr, pd = (point[f"partial_r2.{n}"] for n in names[:2])
    lo, hi = ci["tile"]["margin_abs_std.r_minus_d"]["ci95"]
    if r >= DOMINANCE_RATIO * d and pr >= DOMINANCE_PARTIAL * pd and lo > 0:
        return "receiver dominates"
    if d >= DOMINANCE_RATIO * r and pd >= DOMINANCE_PARTIAL * pr and hi < 0:
        return "donor dominates"
    return "no clear winner"


def regress(y: np.ndarray, regressors: dict[str, np.ndarray], groups: np.ndarray,
            tiles: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Joint within-type regression of *y*, point fit, both bootstraps, verdict."""
    names = list(regressors)
    block = np.column_stack([y] + [regressors[n] for n in names])
    rows = moment_rows(block)
    _, groups = np.unique(groups, return_inverse=True)
    n_groups = int(groups.max()) + 1
    _, tiles = np.unique(tiles, return_inverse=True)
    p = block.shape[1]
    summary = lambda stats: _flatten(fe_fit(stats, p), names)
    fit = fe_fit(indicator(groups, n_groups) @ rows, p)
    point = _flatten(fit, names)
    ci = bootstrap(rows, groups, n_groups, tiles, summary, n_boot, seed)
    return {"regressors": names, "n_cells": int(len(y)), "n_groups": n_groups,
            "point": point, "within_corr": fit["within_corr"], "bootstrap": ci,
            "verdict": verdict(point, ci, names) if len(names) >= 2 else None}


def raw_line(y: np.ndarray, x: np.ndarray, tiles: np.ndarray,
             n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Pooled raw-scale ``y = a + b x`` with a tile-bootstrap CI on each number."""
    rows = moment_rows(np.column_stack([y, x]))
    groups = np.zeros(len(y), dtype=np.int64)
    _, tiles = np.unique(tiles, return_inverse=True)
    point = line_fit(indicator(groups, 1) @ rows)
    ci = bootstrap(rows, groups, 1, tiles, line_fit, n_boot, seed)
    return {"point": point, "tile_ci95": {k: v["ci95"] for k, v in ci["tile"].items()},
            "cell_ci95": {k: v["ci95"] for k, v in ci["cell"].items()}}


# --------------------------------------------------------------------------
# slide-side plumbing (transcript_flux's assignment, the model's beta)
# --------------------------------------------------------------------------


def load_cells(dataset: str, variant: str, label_key: str | None,
               max_edge_um: float, tau_um: float,
               max_transcripts: int | None = None) -> dict:
    """Per-cell F, depths, donor depths and covariates over the whole slide."""
    import pandas as pd

    from discell.data.loader import CellGraphDataset
    from discell.model.prepare import from_dataset as model_graph

    ds = paths.dataset(dataset)
    opened = CellGraphDataset.from_dataset(ds, variant, label_key=label_key)
    n = opened.adata.n_obs
    xenium_dir = Path(str(opened.adata.uns["xenium_dir"]))
    graph = model_graph(opened, max_edge_um=max_edge_um, tau_um=tau_um)
    beta = graph.in_edges.astype(np.float64)       # rows i, cols j: beta_ij
    nucleus_xy = tf.nucleus_centroids(xenium_dir, opened.cell_ids)
    eligible = np.isfinite(nucleus_xy[:, 0])
    # the flux slots exactly as transcript_flux.load_slide builds them
    adjacency = tf.adjacency_csr(graph.edge_i, graph.edge_j, n, eligible)
    total = np.asarray(opened.counts.sum(axis=1)).ravel().astype(np.float64)
    obs = opened.adata.obs
    area = obs["xenium_cell_area"].to_numpy(dtype=np.float64)
    seg = np.asarray(obs["xenium_segmentation_method"].astype(str).to_numpy(), dtype=str)
    gene_names = np.asarray([str(g) for g in opened.adata.var_names])

    nuclear = None
    nuc_path = ds.root / "qc" / "nuclear_counts.npz"
    if nuc_path.exists():
        matrix = sp.load_npz(nuc_path).tocsr()
        nuclear = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
        if len(nuclear) != n:
            rows = pd.Index(tf._full_cell_ids(ds.bundle_dir / "full.h5ad")
                            ).get_indexer(opened.cell_ids)
            if (rows < 0).any():
                raise ValueError("cells missing from full.h5ad")
            nuclear = nuclear[rows]
        del matrix

    positions = np.asarray(opened.positions_um, dtype=np.float64)
    cell_ids = opened.cell_ids
    type_index = opened.type_index
    type_names = np.asarray([str(t) for t in opened.type_names])
    label = opened.label_key
    del opened, obs

    sampled = np.zeros(adjacency.nnz, dtype=bool)   # no per-edge gene content
    acc, _, _ = tf.stream_slide(xenium_dir, cell_ids, nucleus_xy, adjacency,
                                gene_names, sampled, max_rows=max_transcripts)
    indptr = adjacency.indptr
    per_cell = {}
    for name, key in (("F_hard", FOREIGN), ("F_ramp", WEIGHTED)):
        _, kappa = tf.beta_and_kappa(acc.flux[key], indptr, total)
        per_cell[name] = kappa * np.maximum(total, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(area > 0, total / area, np.nan)
    per_cell.update({
        "total": total,
        "donor_depth": beta @ total,
        "donor_density": beta @ np.nan_to_num(density, nan=0.0),
        "donor_area": beta @ np.nan_to_num(area, nan=0.0),
        "area": area,
        "nuclear": nuclear if nuclear is not None else np.full(n, np.nan),
        "donor_nuclear": (beta @ nuclear) if nuclear is not None else np.full(n, np.nan),
        "flux_degree": np.diff(indptr).astype(np.int64),
        "model_degree": np.diff(beta.indptr).astype(np.int64),
        "has_nucleus": eligible,
        "positions_um": positions,
        "type_index": type_index,
        "segmentation": seg,
    })
    per_cell["meta"] = {
        "dataset": dataset, "variant": variant, "label_key": label,
        "type_names": type_names.tolist(), "n_cells": int(n),
        "transcripts_read": int(acc.n_transcripts),
        "extranuclear_q20_considered": int(acc.n_extranuclear),
        "placed_on_an_edge": int(acc.n_placed),
        "n_edges_pruned": int(len(graph.edge_i)),
        "directed_flux_slots": int(adjacency.nnz),
        "has_nuclear_counts": nuclear is not None,
    }
    return per_cell


def _save_cache(cells: dict, path: Path) -> None:
    arrays = {k: (v.astype(str) if v.dtype == object else v)
              for k, v in cells.items() if k != "meta"}
    np.savez(path, meta=json.dumps(cells["meta"]), **arrays)


def _load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        cells = {k: f[k] for k in f.files if k != "meta"}
        cells["meta"] = json.loads(str(f["meta"]))
    return cells


# --------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------


def analysis_set(cells: dict) -> tuple[np.ndarray, dict]:
    """Cells the regression can use, with the count dropped at each filter."""
    steps = {"all": np.ones(len(cells["total"]), dtype=bool)}
    steps["has_nucleus"] = steps["all"] & cells["has_nucleus"].astype(bool)
    steps["has_flux_neighbour"] = steps["has_nucleus"] & (cells["flux_degree"] > 0)
    steps["depth_positive"] = steps["has_flux_neighbour"] & (cells["total"] > 0)
    steps["donor_positive"] = (steps["depth_positive"] & (cells["donor_depth"] > 0)
                               & (cells["donor_density"] > 0))
    steps["area_positive"] = (steps["donor_positive"] & np.isfinite(cells["area"])
                              & (cells["area"] > 0) & (cells["donor_area"] > 0))
    return steps["area_positive"], {k: int(v.sum()) for k, v in steps.items()}


def analyse(cells: dict, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    keep, funnel = analysis_set(cells)
    c = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in cells.items()}
    tiles_xy = np.floor(c["positions_um"] / TILE_UM).astype(np.int64)
    tiles = tiles_xy[:, 0] * 1_000_003 + tiles_xy[:, 1]
    types = c["type_index"]
    y = np.log1p(c["F_hard"])
    log_l = np.log(c["total"])
    log_d = np.log(c["donor_depth"])
    specs = {
        "primary": (y, {"receiver_log_l": log_l, "donor_log_sum_beta_l": log_d}),
        "density": (y, {"receiver_log_l": log_l,
                        "donor_log_sum_beta_l_over_A": np.log(c["donor_density"])}),
        # robustness, not pre-registered ----------------------------------
        "weighted_F_ramp4": (np.log1p(c["F_ramp"]),
                             {"receiver_log_l": log_l, "donor_log_sum_beta_l": log_d}),
        "size_controls": (y, {"receiver_log_l": log_l, "donor_log_sum_beta_l": log_d,
                              "log_A_i": np.log(c["area"]),
                              "log_sum_beta_A_j": np.log(c["donor_area"])}),
    }
    if np.isfinite(c["nuclear"]).all():
        specs["nuclear_depths"] = (y, {
            "receiver_log1p_nuclear": np.log1p(c["nuclear"]),
            "donor_log1p_sum_beta_nuclear": np.log1p(c["donor_nuclear"])})
    fits = {}
    for name, (target, regs) in specs.items():
        started = time.time()
        fits[name] = regress(target, regs, types, tiles, n_boot, seed)
        log.info("%s: %s  std %s  (%.0f s)", name, fits[name]["verdict"],
                 {k.split(".", 1)[1]: round(v, 3)
                  for k, v in fits[name]["point"].items() if k.startswith("std_coef")},
                 time.time() - started)
    by_seg = {}
    for method in sorted(set(c["segmentation"].tolist())):
        m = c["segmentation"] == method
        if m.sum() < 1000:
            continue
        by_seg[method] = regress(y[m], {"receiver_log_l": log_l[m],
                                        "donor_log_sum_beta_l": log_d[m]},
                                 types[m], tiles[m], n_boot, seed)
    raw = {
        "F_on_receiver_l": raw_line(c["F_hard"], c["total"], tiles, n_boot, seed),
        "F_on_donor_sum_beta_l": raw_line(c["F_hard"], c["donor_depth"], tiles,
                                          n_boot, seed),
    }
    kappa_i = c["F_hard"] / c["total"]
    return {
        "funnel": funnel, "n_cells_used": int(keep.sum()),
        "n_types_used": int(len(np.unique(types))),
        "descriptives": {
            "F_hard_quantiles": tf._quantiles(c["F_hard"]),
            "l_quantiles": tf._quantiles(c["total"]),
            "donor_depth_quantiles": tf._quantiles(c["donor_depth"]),
            "kappa_hard_quantiles": tf._quantiles(kappa_i),
            "kappa_hard_pooled": float(c["F_hard"].sum() / c["total"].sum()),
            "kappa_ramp4_pooled": float(c["F_ramp"].sum() / c["total"].sum()),
            "kappa_ramp4_median": float(np.median(c["F_ramp"] / c["total"])),
            "segmentation_counts": {m: int((c["segmentation"] == m).sum())
                                    for m in sorted(set(c["segmentation"].tolist()))},
        },
        "joint": fits, "joint_by_segmentation": by_seg, "raw_slopes": raw,
        "test_1b": donor_at_fixed_area(c, types, tiles, n_boot, seed),
    }


# --------------------------------------------------------------------------
# test 1b: donor scaling at fixed area
# --------------------------------------------------------------------------


def classify_1b(coef: float, ci95: Sequence[float]) -> str:
    """The pre-registered reading of one slide's donor elasticity at fixed areas.

    ``ruled out``: coef <= 0 and CI upper bound < 0.25.  ``small, not zero``:
    coef in (0, 0.25) and the CI excludes 0.25.  ``open``: the CI covers 0.25.
    ``practically positive``: the whole CI lies above 0.25 (the entry's own
    definition of a practically positive elasticity; the rule names no fourth
    class).
    """
    lo, hi = ci95
    t = PRACTICAL_ELASTICITY
    if coef <= 0 and hi < t:
        return "ruled out"
    if 0 < coef < t and hi < t:
        return "small, not zero"
    if lo > t:
        return "practically positive"
    return "open"


def overall_1b(per_slide: dict[str, str]) -> str:
    """Donor scaling is ruled out only if it is ruled out on every slide."""
    classes = list(per_slide.values())
    if classes and all(c == "ruled out" for c in classes):
        return "ruled out"
    hot = [s for s, c in per_slide.items() if c == "practically positive"]
    if hot:
        return "not ruled out: practically positive on " + ", ".join(hot)
    if any(c == "open" for c in classes):
        return "open: the depth-scaled sensitivity arm decides"
    return "not ruled out: small, not zero"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def donor_at_fixed_area(c: dict, types: np.ndarray, tiles: np.ndarray,
            n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Both pre-registered specifications, their elasticities and classifications."""
    y = np.log1p(c["F_hard"])
    log_a, log_sa = np.log(c["area"]), np.log(c["donor_area"])
    areas = {"log_A_i": log_a, "log_sum_beta_A_j": log_sa}
    specs = {
        "area_conditioned": {"receiver_log_l": np.log(c["total"]),
                             "donor_log_sum_beta_l": np.log(c["donor_depth"]),
                             **areas},
        "density_area_conditioned": {
            "receiver_log_l_over_A": np.log(c["total"] / c["area"]),
            "donor_log_sum_beta_l_over_A": np.log(c["donor_density"]), **areas},
    }
    out = {"threshold_elasticity": PRACTICAL_ELASTICITY, "specs": {}}
    for name, regs in specs.items():
        started = time.time()
        fit = regress(y, regs, types, tiles, n_boot, seed)
        fit.pop("verdict")                  # test 1's rule does not apply here
        names = fit["regressors"]
        fit["elasticities"] = {
            n: {"coef": fit["point"][f"coef.{n}"],
                "tile_ci95": fit["bootstrap"]["tile"][f"coef.{n}"]["ci95"],
                "cell_ci95": fit["bootstrap"]["cell"][f"coef.{n}"]["ci95"]}
            for n in names}
        donor = fit["elasticities"][names[1]]
        fit["donor_term"] = names[1]
        fit["classification"] = classify_1b(donor["coef"], donor["tile_ci95"])
        fit["correlations"] = {
            "receiver_depth_area": {
                "pooled": _pearson(regs[names[0]], log_a),
                "within_type": float(fit["within_corr"][0][2])},
            "donor_depth_area": {
                "pooled": _pearson(regs[names[1]], log_sa),
                "within_type": float(fit["within_corr"][1][3])},
        }
        out["specs"][name] = fit
        log.info("1b %s: donor elasticity %+.3f tile CI [%+.3f, %+.3f] -> %s (%.0f s)",
                 name, donor["coef"], *donor["tile_ci95"], fit["classification"],
                 time.time() - started)
    out["slide_verdict"] = out["specs"]["area_conditioned"]["classification"]
    return out


def added_variable(y: np.ndarray, x: np.ndarray, controls: np.ndarray,
                   groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residuals of *y* and *x* on type means and *controls* (Frisch-Waugh).

    The OLS slope of the first on the second is *x*'s coefficient in the joint
    within-type fit, so their binned means are the picture of that coefficient.
    """
    _, g = np.unique(groups, return_inverse=True)
    block = np.column_stack([y, x, controls])
    counts = np.bincount(g).astype(np.float64)
    means = np.stack([np.bincount(g, weights=col) for col in block.T], axis=1)
    block = block - (means / counts[:, None])[g]
    ctrl = block[:, 2:]
    coef, *_ = np.linalg.lstsq(ctrl, block[:, :2], rcond=None)
    resid = block[:, :2] - ctrl @ coef
    return resid[:, 0], resid[:, 1]


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------


def binned(x: np.ndarray, ys: dict[str, np.ndarray], n_bins: int = N_BINS) -> dict:
    """Equal-count bins in *x*: bin medians of *x* and quartiles of each *y*."""
    order = np.argsort(x, kind="stable")
    parts = np.array_split(order, n_bins)
    out = {"x": np.array([np.median(x[p]) for p in parts])}
    for name, y in ys.items():
        out[name] = np.array([np.percentile(y[p], [25, 50, 75]) for p in parts])
    return out


def figure(cells: dict, result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    keep, _ = analysis_set(cells)
    f, ell, d = (cells[k][keep] for k in ("F_hard", "total", "donor_depth"))
    blue, orange, grey, ink = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b"
    pooled = result["descriptives"]["kappa_hard_pooled"]
    has_1b = "test_1b" in result
    fig, axes = plt.subplots(1, 3 if has_1b else 2, figsize=(16.5 if has_1b else 11, 4.6))
    for ax, x, xlabel in ((axes[0], ell, r"receiver depth $\ell_i$ (total counts)"),
                          (axes[1], d, r"donor depth $\sum_j \beta_{ij}\ell_j$")):
        b = binned(x, {"F": 1.0 + f, "model": 1.0 + KAPPA_MODEL * ell})
        ax.fill_between(b["x"], b["F"][:, 0], b["F"][:, 2], color=blue, alpha=0.18,
                        lw=0, label=r"$1 + F_i$, IQR")
        ax.plot(b["x"], b["F"][:, 1], color=blue, lw=2, marker="o", ms=4,
                label=r"$1 + F_i$, bin median")
        if ax is axes[0]:
            grid = np.geomspace(b["x"][0], b["x"][-1], 100)
            ax.plot(grid, 1.0 + KAPPA_MODEL * grid, color=orange, lw=2,
                    label=rf"$1 + \kappa\ell_i$, $\kappa$ = {KAPPA_MODEL:g} (model)")
            ax.plot(grid, 1.0 + pooled * grid, color=grey, lw=1.2, ls=":",
                    label=rf"$1 + \hat\kappa\ell_i$, pooled $\hat\kappa$ = {pooled:.3f}")
        else:
            ax.plot(b["x"], b["model"][:, 1], color=orange, lw=2, ls="--",
                    label=r"$1 + \kappa\ell_i$, bin median (model rule)")
            grid = np.geomspace(b["x"][0], b["x"][-1], 100)
            ax.plot(grid, 1.0 + KAPPA_MODEL * grid, color=grey, lw=1.2, ls=":",
                    label=rf"$1 + \kappa\sum_j\beta_{{ij}}\ell_j$ (donor rule)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(xlabel, color=ink)
        ax.set_ylabel(r"$1 + F_i$ (foreign-side transcripts)", color=ink)
        ax.grid(True, which="major", color="0.88", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    if has_1b:
        ax = axes[2]
        fit = result["test_1b"]["specs"]["area_conditioned"]
        donor = fit["elasticities"]["donor_log_sum_beta_l"]
        controls = np.column_stack([np.log(ell), np.log(cells["area"][keep]),
                                    np.log(cells["donor_area"][keep])])
        ry, rx = added_variable(np.log1p(f), np.log(d), controls,
                                cells["type_index"][keep])
        order = np.array_split(np.argsort(rx, kind="stable"), N_BINS)
        bx = np.array([rx[p].mean() for p in order])
        by = np.array([ry[p].mean() for p in order])
        ax.plot(bx, by, color=blue, lw=2, marker="o", ms=4,
                label=r"bin mean (added-variable residuals)")
        grid = np.array([bx[0], bx[-1]])
        lo, hi = donor["tile_ci95"]
        ax.plot(grid, donor["coef"] * grid, color=blue, lw=1, ls="-",
                label=f"fitted elasticity {donor['coef']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        ax.plot(grid, PRACTICAL_ELASTICITY * grid, color=grey, lw=1.2, ls=":",
                label=f"threshold {PRACTICAL_ELASTICITY:g} (practically positive)")
        ax.plot(grid, 1.0 * grid, color=orange, lw=2, ls="--",
                label="elasticity 1 (donor-scaled leak)")
        ax.axhline(0, color="0.75", lw=0.6)
        ax.set_xlabel(r"log $\sum_j\beta_{ij}\ell_j$ | log $\ell_i$, log $A_i$, "
                      r"log $\sum_j\beta_{ij}A_j$, type  (residual)", color=ink)
        ax.set_ylabel(r"log$(1 + F_i)$ residual", color=ink)
        ax.set_title(f"Test 1b, donor depth at fixed areas: "
                     f"{result['test_1b']['slide_verdict']}", fontsize=9, color=ink)
        ax.grid(True, color="0.88", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    prim = result["joint"]["primary"]["point"]
    fig.suptitle(
        f"{result['meta']['dataset']}: {result['n_cells_used']:,} cells, "
        f"{N_BINS} equal-count bins.  Joint within-type std coef: receiver "
        f"{prim['std_coef.receiver_log_l']:+.3f}, donor "
        f"{prim['std_coef.donor_log_sum_beta_l']:+.3f}",
        fontsize=9, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _ci(fit: dict, key: str, scheme: str = "tile") -> str:
    lo, hi = fit["bootstrap"][scheme][key]["ci95"]
    return f"[{lo:+.3f}, {hi:+.3f}]"


def markdown(result: dict) -> str:
    meta = result["meta"]
    lines = [f"# R12 Test 1: foreign count vs receiver and donor depth, {meta['dataset']}",
             "",
             f"Label `{meta['label_key']}`, {result['n_types_used']} types as fixed effects; "
             f"{result['n_cells_used']:,} of {meta['n_cells']:,} cells "
             f"(funnel {result['funnel']}). F_i = extranuclear q20 transcripts nearer a "
             f"neighbour's nucleus than the cell's own (transcript_flux `hard`); "
             f"{meta['transcripts_read']:,} transcripts read, "
             f"{meta['placed_on_an_edge']:,} extranuclear placed. "
             f"Bootstrap {result['joint']['primary']['bootstrap']['n_boot']} reps; "
             f"CIs below are over {TILE_UM:g} um tiles "
             f"({result['joint']['primary']['bootstrap']['n_tiles']} tiles), cell-level CIs in the JSON.",
             "",
             "## Joint within-type regressions of log(1+F_i)",
             "",
             "| spec | term | coef (elasticity) | std coef [tile 95% CI] | partial R2 | LMG | verdict |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, fit in result["joint"].items():
        pt = fit["point"]
        for k, term in enumerate(fit["regressors"]):
            lines.append(
                f"| {name if k == 0 else ''} | {term} | {pt['coef.' + term]:+.3f} | "
                f"{pt['std_coef.' + term]:+.3f} {_ci(fit, 'std_coef.' + term)} | "
                f"{pt['partial_r2.' + term]:.3f} | {pt['lmg.' + term]:.3f} | "
                f"{fit['verdict'] if k == 0 else ''} |")
        lines.append(
            f"| | *within R2 {pt['r2_within']:.3f}, types-only R2 {pt['r2_groups_only']:.3f}, "
            f"total R2 {pt['r2_total']:.3f}; LMG share of first term "
            f"{pt['lmg_share.first_of_two']:.3f}; within-type corr(term1, term2) "
            f"{fit['within_corr'][0][1]:.3f}* | | | | | |")
    lines += ["", "## Primary spec by segmentation method", "",
              "| segmentation | n | std receiver [CI] | std donor [CI] | partial R2 r / d | verdict |",
              "| --- | --- | --- | --- | --- | --- |"]
    for method, fit in result["joint_by_segmentation"].items():
        pt = fit["point"]
        lines.append(
            f"| {method} | {fit['n_cells']:,} | {pt['std_coef.receiver_log_l']:+.3f} "
            f"{_ci(fit, 'std_coef.receiver_log_l')} | {pt['std_coef.donor_log_sum_beta_l']:+.3f} "
            f"{_ci(fit, 'std_coef.donor_log_sum_beta_l')} | "
            f"{pt['partial_r2.receiver_log_l']:.3f} / {pt['partial_r2.donor_log_sum_beta_l']:.3f} | "
            f"{fit['verdict']} |")
    lines += ["", "## Raw-scale slopes, pooled, no type effects", "",
              f"Model rule F_i = kappa l_i: slope kappa = {KAPPA_MODEL:g}, intercept 0.", "",
              "| fit | slope [tile CI] | intercept [tile CI] | R2 | through-origin slope | sum F / sum x |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name, fit in result["raw_slopes"].items():
        pt, ci = fit["point"], fit["tile_ci95"]
        lines.append(
            f"| {name} | {pt['slope']:.4f} [{ci['slope'][0]:.4f}, {ci['slope'][1]:.4f}] | "
            f"{pt['intercept']:.2f} [{ci['intercept'][0]:.2f}, {ci['intercept'][1]:.2f}] | "
            f"{pt['r2']:.3f} | {pt['slope_through_origin']:.4f} | {pt['ratio_of_sums']:.4f} |")
    d = result["descriptives"]
    lines += ["", "## Descriptives", "",
              f"- F_i quantiles {d['F_hard_quantiles']}",
              f"- l_i quantiles {d['l_quantiles']}",
              f"- donor depth quantiles {d['donor_depth_quantiles']}",
              f"- kappa_i (hard) quantiles {d['kappa_hard_quantiles']}; pooled "
              f"{d['kappa_hard_pooled']:.4f}; ramp-4 pooled {d['kappa_ramp4_pooled']:.4f}, "
              f"median {d['kappa_ramp4_median']:.4f}",
              f"- segmentation {d['segmentation_counts']}",
              "",
              f"Rule: a term dominates when |std coef| >= {DOMINANCE_RATIO:g}x the other's, "
              f"its partial R2 >= {DOMINANCE_PARTIAL:g}x the other's, and the tile CI of "
              "|std_r| - |std_d| excludes 0 on its side. LMG is reported, not ruled on "
              "(it splits shared variance equally). Specs other than `primary` and "
              "`density` are robustness checks, not pre-registered.", ""]
    if "test_1b" in result:
        lines += markdown_1b(result["test_1b"])
    return "\n".join(lines)


def markdown_1b(block: dict) -> list[str]:
    t = block["threshold_elasticity"]
    lines = ["## Test 1b: donor scaling at fixed area (pre-registered 2026-09-24)", "",
             f"**Slide verdict (area_conditioned donor elasticity): {block['slide_verdict']}.**",
             "", "Outcome log(1+F_i), type fixed effects. Elasticities are the "
             "unstandardised log-log coefficients; CIs are 95 % bootstrap over tiles "
             "(and over cells).", "",
             "| spec | term | elasticity | tile 95% CI | cell 95% CI | std coef | partial R2 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, fit in block["specs"].items():
        for k, term in enumerate(fit["regressors"]):
            e = fit["elasticities"][term]
            lines.append(
                f"| {name if k == 0 else ''} | {term} | {e['coef']:+.3f} | "
                f"[{e['tile_ci95'][0]:+.3f}, {e['tile_ci95'][1]:+.3f}] | "
                f"[{e['cell_ci95'][0]:+.3f}, {e['cell_ci95'][1]:+.3f}] | "
                f"{fit['point']['std_coef.' + term]:+.3f} | "
                f"{fit['point']['partial_r2.' + term]:.3f} |")
        cr = fit["correlations"]
        lines.append(
            f"| | *donor term `{fit['donor_term']}`: **{fit['classification']}**; "
            f"depth-area corr receiver {cr['receiver_depth_area']['pooled']:.3f} pooled / "
            f"{cr['receiver_depth_area']['within_type']:.3f} within type, donor "
            f"{cr['donor_depth_area']['pooled']:.3f} / "
            f"{cr['donor_depth_area']['within_type']:.3f}; within R2 "
            f"{fit['point']['r2_within']:.3f}* | | | | | |")
    lines += ["", f"Rule: donor scaling is ruled out on this slide only if the donor-depth "
              f"elasticity at fixed areas is <= 0 and its tile CI upper bound < {t:g}; "
              f"in (0, {t:g}) with the CI excluding {t:g} is 'small, not zero'; a CI "
              f"covering {t:g} is 'open'; a CI wholly above {t:g} is reported as "
              "'practically positive'. The cross-slide verdict needs both slides "
              "(`--combine-1b`). The density variant is reported with the same "
              "classification; the slide verdict reads `area_conditioned`.", ""]
    return lines


def combine_1b(datasets: Sequence[str]) -> dict:
    """The cross-slide 1b verdict from each slide's written ``r12_depth_test.json``."""
    per_slide, rows = {}, []
    for ds in datasets:
        path = paths.dataset(ds).root / "experiments" / "r12_depth_test.json"
        block = json.loads(path.read_text())["test_1b"]
        per_slide[ds] = block["slide_verdict"]
        for name, fit in block["specs"].items():
            e = fit["elasticities"][fit["donor_term"]]
            rows.append({"dataset": ds, "spec": name, "donor_term": fit["donor_term"],
                         "elasticity": e["coef"], "tile_ci95": e["tile_ci95"],
                         "classification": fit["classification"]})
    return {"per_slide": per_slide, "rows": rows, "overall": overall_1b(per_slide)}


def run(args: argparse.Namespace) -> dict:
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        log.info("per-cell arrays from cache %s", cache)
        cells = _load_cache(cache)
    else:
        cells = load_cells(args.dataset, args.variant, args.label_key,
                           args.max_edge_um, args.tau_um, args.max_transcripts)
        if cache is not None:
            _save_cache(cells, cache)
    result = analyse(cells, n_boot=args.n_boot, seed=args.seed)
    result["meta"] = cells["meta"]
    result["parameters"] = {
        "kappa_model": KAPPA_MODEL, "foreign_weight": list(FOREIGN),
        "weighted_variant": list(WEIGHTED), "min_qv": tf.MIN_QV,
        "max_edge_um": args.max_edge_um, "tau_um": args.tau_um,
        "tile_um": TILE_UM, "n_boot": args.n_boot, "seed": args.seed,
        "dominance_ratio": DOMINANCE_RATIO, "dominance_partial": DOMINANCE_PARTIAL,
        "area_column": "xenium_cell_area",
        "beta": "model in_edges from discell.model.prepare.from_dataset",
    }
    out_dir = paths.dataset(args.dataset).root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r12_depth_test.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float))
    (out_dir / "r12_depth_test.md").write_text(markdown(result))
    figure(cells, result, out_dir / "r12_depth_test.png")
    log.info("wrote %s", out_dir / "r12_depth_test.{json,md,png}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--combine-1b", nargs="+", default=None, metavar="DATASET",
                        help="print the cross-slide test-1b verdict from the "
                             "slides' written JSONs and exit")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--max-edge-um", type=float, default=DEFAULT_MAX_EDGE_UM)
    parser.add_argument("--tau-um", type=float, default=DEFAULT_TAU_UM)
    parser.add_argument("--cache", default=None,
                        help="npz of the per-cell arrays: read if present, else "
                             "written after the transcript stream")
    parser.add_argument("--max-transcripts", type=int, default=None,
                        help="debug: stop the stream after roughly this many rows")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.combine_1b:
        print(json.dumps(combine_1b(args.combine_1b), indent=2))
        return 0
    if args.dataset is None:
        parser.error("--dataset is required unless --combine-1b is given")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
