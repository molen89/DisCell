#!/usr/bin/env python3
"""Six pre-registered gates for a DNA-content cell-cycle label from nuclear DAPI.

Pre-registered in devlog 2026-09-17, "DNA-content cell-cycle label on the
fresh-frozen slide (motivation)".  On the ovarian FFPE slide the integrated
nuclear DAPI turned out to be nuclear footprint area with a 2.65x tile drift
and no second mode; the fresh-frozen slide is the only remaining candidate and
it has to clear six gates, in order, before a 2N/4N label is shipped:

1. flat-field / tile correction of ``dapi_mean``      residual spread  < 1.2x
2. per-tile background from a pixel crop              share           < 3 %
3. exclusions (no nucleus, multinucleate, expansion)  counts reported
4. truncation: log dapi_sum on log nuclear area       R^2             < 0.1
5. bimodality per type on the log integral            2 comps win, ratio 1.8-2.2
6. the bar: G2M AUROC 4N vs 2N in top-4 MKI67 types   >= 0.70, MKI67 ratio >= 2

The label decision stops at the first failed gate; the later gates are still
evaluated as diagnostics where that is cheap, so the report is complete.
Nothing here touches the model or ``scripts/test_cell_cycle.py``; the module
writes JSON + PNG under ``data/datasets/<id>/experiments/``.

Usage::

    OMP_NUM_THREADS=8 python -m discell.experiments.dapi_cycle \\
        --dataset xenium_prime_human_ovary_ff
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from discell import paths

log = logging.getLogger("discell.experiments.dapi_cycle")

#: Tile edge (um) for the flat-field / background / drift statistics.
TILE_UM = 500.0
#: Thresholds, all pre-registered -- never tuned here.
GATE1_MAX_SPREAD = 1.2
GATE2_MAX_BACKGROUND_SHARE = 0.03
GATE4_MAX_R2 = 0.10
GATE5_RATIO_RANGE = (1.8, 2.2)
GATE5_MIN_BIC_MARGIN = 10.0
GATE6_MIN_AUROC = 0.70
GATE6_MIN_MKI67_RATIO = 2.0
#: A type needs this many kept cells before its bimodality is read.
MIN_CELLS_PER_TYPE = 500
#: Pixel crops for the background estimate.
N_BACKGROUND_CROPS = 120
BACKGROUND_CROP_PX = 512
BACKGROUND_PERCENTILE = 10.0
#: Cells scored for the marker panel in gate 6 (scoring 1.16M rows is heavy).
GATE6_SUBSAMPLE = 200_000


# --------------------------------------------------------------- primitives

def tile_ids(x_um: np.ndarray, y_um: np.ndarray,
             tile_um: float = TILE_UM) -> tuple[np.ndarray, np.ndarray]:
    """Integer tile column/row of each cell centroid."""
    return (np.floor(x_um / tile_um).astype(np.int64),
            np.floor(y_um / tile_um).astype(np.int64))



def _group_medians(inverse: np.ndarray, value: np.ndarray,
                   n_groups: int) -> np.ndarray:
    """Median of ``value`` within each group id, one sort rather than a scan."""
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(n_groups + 1))
    return np.array([np.median(value[order[a:b]]) if b > a else np.nan
                     for a, b in zip(bounds[:-1], bounds[1:])])


def flat_field_gain(col: np.ndarray, row: np.ndarray, value: np.ndarray,
                    min_cells: int = 20, degree: int = 2) -> np.ndarray:
    """Smooth multiplicative gain per cell from per-tile medians of ``value``.

    The per-tile median of the nuclear-pixel mean is taken as the local
    illumination level; a degree-``degree`` polynomial is fitted to its log
    over the tile centres (tiles with fewer than ``min_cells`` cells are
    dropped from the fit) and the fitted surface, normalised to a geometric
    mean of one, is returned per cell.  Only the smooth component is removed:
    tile-to-tile scatter that no smooth field explains stays in the data,
    which is what gate 1 measures.
    """
    value = np.asarray(value, dtype=float)
    keys, inverse = np.unique(np.stack([col, row], 1), axis=0,
                              return_inverse=True)
    counts = np.bincount(inverse, minlength=len(keys))
    medians = _group_medians(inverse, value, len(keys))
    use = (counts >= min_cells) & np.isfinite(medians) & (medians > 0)
    if use.sum() < 6:
        return np.ones(len(value))
    design = _poly_design(keys[use, 0], keys[use, 1], degree)
    coef, *_ = np.linalg.lstsq(design, np.log(medians[use]), rcond=None)
    fitted = _poly_design(col, row, degree) @ coef
    return np.exp(fitted - fitted.mean())


def _poly_design(cx: np.ndarray, cy: np.ndarray, degree: int) -> np.ndarray:
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    sx = (cx - cx.mean()) / max(cx.std(), 1e-9)
    sy = (cy - cy.mean()) / max(cy.std(), 1e-9)
    cols = [np.ones_like(sx)]
    for d in range(1, degree + 1):
        for k in range(d + 1):
            cols.append(sx ** (d - k) * sy ** k)
    return np.stack(cols, axis=1)


def tile_median_spread(col: np.ndarray, row: np.ndarray, value: np.ndarray,
                       min_cells: int = 20,
                       lo: float = 5.0, hi: float = 95.0) -> float:
    """Ratio of the ``hi``- to the ``lo``-percentile tile median of ``value``.

    Percentiles rather than min/max so a single sparse tile cannot set the
    number; ovarian's 2.65x was measured the same way.
    """
    keys, inverse = np.unique(np.stack([col, row], 1), axis=0,
                              return_inverse=True)
    counts = np.bincount(inverse, minlength=len(keys))
    medians = _group_medians(inverse, value, len(keys))
    medians = medians[(counts >= min_cells) & np.isfinite(medians)
                      & (medians > 0)]
    if medians.size < 2:
        return 1.0
    return float(np.percentile(medians, hi) / np.percentile(medians, lo))


def dip_test(values: np.ndarray, max_n: int = 50_000,
             seed: int = 0) -> tuple[float, float]:
    """Hartigan's dip statistic and its p-value (``diptest``), subsampled."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size > max_n:
        values = np.random.default_rng(seed).choice(values, max_n,
                                                    replace=False)
    try:
        import diptest
    except ImportError:                                    # pragma: no cover
        log.warning("diptest unavailable; dip reported as NaN")
        return float("nan"), float("nan")
    stat, pval = diptest.diptest(values)
    return float(stat), float(pval)


def bimodality(log_values: np.ndarray, seed: int = 0) -> dict:
    """1- vs 2-component BIC, component ratio and dip on a log-scale sample.

    ``ratio`` is the linear-scale ratio of the two component means, i.e. what a
    2N/4N split would put at 2.0.
    """
    from sklearn.mixture import GaussianMixture

    values = np.asarray(log_values, dtype=float)
    values = values[np.isfinite(values)].reshape(-1, 1)
    out = {"n": int(values.size), "bic_1": float("nan"), "bic_2": float("nan"),
           "bic_margin": float("nan"), "ratio": float("nan"),
           "weight_high": float("nan"), "dip": float("nan"),
           "dip_p": float("nan"), "two_components_win": False,
           "ratio_in_range": False}
    if values.size < 50:
        return out
    fits = {}
    for k in (1, 2):
        gmm = GaussianMixture(k, random_state=seed, n_init=3).fit(values)
        fits[k] = (gmm, gmm.bic(values))
    mu = np.sort(fits[2][0].means_.ravel())
    order = np.argsort(fits[2][0].means_.ravel())
    out["bic_1"] = float(fits[1][1])
    out["bic_2"] = float(fits[2][1])
    out["bic_margin"] = float(fits[1][1] - fits[2][1])   # >0: 2 components win
    out["ratio"] = float(np.exp(mu[1] - mu[0]))
    out["weight_high"] = float(fits[2][0].weights_.ravel()[order][1])
    out["dip"], out["dip_p"] = dip_test(values.ravel(), seed=seed)
    out["two_components_win"] = bool(out["bic_margin"] > GATE5_MIN_BIC_MARGIN
                                     and (out["dip_p"] < 0.05
                                          or not np.isfinite(out["dip_p"])))
    out["ratio_in_range"] = bool(GATE5_RATIO_RANGE[0] <= out["ratio"]
                                 <= GATE5_RATIO_RANGE[1])
    return out


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank AUROC of ``scores`` for the boolean ``positive`` class."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def r_squared(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(slope, R^2) of an ordinary least-squares fit of ``y`` on ``x``."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if x.size < 10 or x.std() == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return float(slope), float(1.0 - resid.var() / y.var())


# ------------------------------------------------------------------- inputs

def load_slide(dataset: str) -> dict:
    """Per-cell DAPI table joined to the bundle obs, bundle order."""
    import anndata as ad
    import pandas as pd

    ds = paths.dataset(dataset)
    frame = pd.read_parquet(ds.root / "qc" / "nuclear_dapi.parquet")
    adata = ad.read_h5ad(ds.root / "bundle" / "full.h5ad", backed="r")
    obs = adata.obs
    frame = frame.reindex(adata.obs_names)
    xenium_dir = Path(str(adata.uns["xenium_dir"]))
    mpp = float(adata.uns.get("microns_per_pixel", 0.2125))
    return {
        "dataset": dataset, "xenium_dir": xenium_dir, "mpp": mpp,
        "cell_ids": adata.obs_names.to_numpy(),
        "dapi_sum": frame["dapi_sum"].to_numpy(float),
        "dapi_mean": frame["dapi_mean"].to_numpy(float),
        "area_px": frame["nucleus_area_px"].to_numpy(float),
        "area_um2": frame["nucleus_area_um2"].to_numpy(float),
        "x_um": obs["centroid_x_um"].to_numpy(float),
        "y_um": obs["centroid_y_um"].to_numpy(float),
        "nucleus_count": obs["xenium_nucleus_count"].to_numpy(float),
        "segmentation": obs["xenium_segmentation_method"].astype(str)
                            .to_numpy(),
        "graphclust": obs["graphclust"].astype("object").fillna("NA")
                          .astype(str).to_numpy(),
        "total_counts": obs["xenium_total_counts"].to_numpy(float),
        "h5ad": ds.root / "bundle" / "full.h5ad",
    }


def _dapi_plane(xenium_dir: Path):
    """Lazy zarr handle on channel 0 of the morphology image (never read whole)."""
    import tifffile
    import zarr

    path = xenium_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
    node = zarr.open(tifffile.imread(path, aszarr=True), mode="r")
    if isinstance(node, zarr.Group):
        node = node["0"] if "0" in node else node[sorted(node.array_keys())[0]]
    return node


def _crop(node, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """A window of channel 0, read through zarr without materialising the plane."""
    if node.ndim == 3:
        return np.asarray(node[0, y0:y1, x0:x1])
    return np.asarray(node[y0:y1, x0:x1])


# -------------------------------------------------------------------- gates

def gate1_flat_field(slide: dict) -> dict:
    """Tile-median spread of ``dapi_mean`` before and after the smooth gain."""
    ok = np.isfinite(slide["dapi_mean"]) & (slide["dapi_mean"] > 0)
    col, row = tile_ids(slide["x_um"], slide["y_um"])
    raw = tile_median_spread(col[ok], row[ok], slide["dapi_mean"][ok])
    gain = np.ones(len(ok))
    gain[ok] = flat_field_gain(col[ok], row[ok], slide["dapi_mean"][ok])
    corrected = np.where(ok, slide["dapi_mean"] / gain, np.nan)
    residual = tile_median_spread(col[ok], row[ok], corrected[ok])
    return {"gate": 1, "name": "flat-field / tile correction",
            "threshold": f"residual tile-median spread < {GATE1_MAX_SPREAD}x",
            "raw_tile_spread": raw, "residual_tile_spread": residual,
            "n_tiles": int(np.unique(np.stack([col[ok], row[ok]], 1),
                                     axis=0).shape[0]),
            "passed": bool(residual < GATE1_MAX_SPREAD),
            "_gain": gain}


def gate2_background(slide: dict, n_crops: int = N_BACKGROUND_CROPS,
                     seed: int = 0) -> dict:
    """Per-tile background level from pixel crops, as a share of the integral.

    The background of a crop is its ``BACKGROUND_PERCENTILE``-th pixel
    percentile: nuclei cover well under half of a 512 px window, so that
    percentile sits in extracellular space.  The share is
    ``background x median nuclear area / median dapi_sum``.
    """
    rng = np.random.default_rng(seed)
    node = _dapi_plane(slide["xenium_dir"])
    shape = node.shape[-2:]
    ok = np.isfinite(slide["dapi_sum"]) & (slide["dapi_sum"] > 0)
    px_x = slide["x_um"][ok] / slide["mpp"]
    px_y = slide["y_um"][ok] / slide["mpp"]
    col, row = tile_ids(slide["x_um"][ok], slide["y_um"][ok])
    keys, inverse = np.unique(np.stack([col, row], 1), axis=0,
                              return_inverse=True)
    counts = np.bincount(inverse, minlength=len(keys))
    populated = np.flatnonzero(counts >= 50)
    pick = rng.choice(populated, min(n_crops, populated.size), replace=False)
    half = BACKGROUND_CROP_PX // 2
    levels = []
    for k in pick:
        members = np.flatnonzero(inverse == k)
        cx, cy = int(np.median(px_x[members])), int(np.median(px_y[members]))
        y0, x0 = max(cy - half, 0), max(cx - half, 0)
        crop = _crop(node, y0, min(y0 + 2 * half, shape[0]),
                     x0, min(x0 + 2 * half, shape[1]))
        if crop.size:
            levels.append(float(np.percentile(crop, BACKGROUND_PERCENTILE)))
    levels = np.array(levels, dtype=float)
    median_area = float(np.median(slide["area_px"][ok]))
    median_sum = float(np.median(slide["dapi_sum"][ok]))
    share = float(np.median(levels) * median_area / median_sum)
    return {"gate": 2, "name": "per-tile background",
            "threshold": f"background < {GATE2_MAX_BACKGROUND_SHARE:.0%} of "
                         "the mean nuclear integral",
            "n_crops": int(levels.size),
            "background_level_median": float(np.median(levels)),
            "background_level_p5_p95": [float(np.percentile(levels, 5)),
                                        float(np.percentile(levels, 95))],
            "median_nucleus_area_px": median_area,
            "median_dapi_sum": median_sum,
            "background_share_of_integral": share,
            "passed": bool(share < GATE2_MAX_BACKGROUND_SHARE),
            "_level": float(np.median(levels))}


def gate3_exclusions(slide: dict) -> dict:
    """Drop cells with no nucleus polygon, multinucleate cells, expansion route."""
    n = len(slide["dapi_sum"])
    no_nucleus = ~(np.isfinite(slide["dapi_sum"]) & (slide["dapi_sum"] > 0))
    multinucleate = np.nan_to_num(slide["nucleus_count"], nan=0.0) != 1
    expansion = np.char.find(slide["segmentation"].astype(str),
                             "nucleus expansion") >= 0
    keep = ~(no_nucleus | multinucleate | expansion)
    return {"gate": 3, "name": "exclusions",
            "threshold": "counts reported (no pass/fail)",
            "n_cells": n,
            "n_no_nucleus": int(no_nucleus.sum()),
            "n_multinucleate": int(multinucleate.sum()),
            "n_nucleus_expansion": int(expansion.sum()),
            "n_kept": int(keep.sum()), "frac_kept": float(keep.mean()),
            "passed": True, "_keep": keep}


def corrected_integral(slide: dict, gain: np.ndarray, background: float,
                       keep: np.ndarray) -> np.ndarray:
    """Flat-field- and background-corrected integrated nuclear DAPI."""
    value = (slide["dapi_sum"] - background * slide["area_px"]) / gain
    out = np.where(keep & np.isfinite(value) & (value > 0), value, np.nan)
    return out


def gate4_truncation(slide: dict, integral: np.ndarray) -> dict:
    """Is the corrected integral still nuclear footprint area?"""
    ok = np.isfinite(integral) & (slide["area_px"] > 0)
    lx = np.log(slide["area_px"][ok])
    ly = np.log(integral[ok])
    slope, r2 = r_squared(lx, ly)
    resid = ly - np.polyval(np.polyfit(lx, ly, 1), lx)
    _, r2_resid = r_squared(lx, resid)
    density = integral[ok] / slide["area_px"][ok]
    return {"gate": 4, "name": "truncation / area confound",
            "threshold": f"R^2 of the integral on area < {GATE4_MAX_R2}",
            "n": int(ok.sum()), "loglog_slope": slope, "r2_on_area": r2,
            "r2_of_residual_on_area": r2_resid,
            "corr_density_area": float(np.corrcoef(
                density, slide["area_px"][ok])[0, 1]),
            "passed": bool(r2 < GATE4_MAX_R2)}


def gate5_bimodality(integral: np.ndarray, types: np.ndarray,
                     seed: int = 0) -> dict:
    """Per-type 1-vs-2 component BIC + dip on the log corrected integral.

    Pre-registration fixes the per-type quantities; the slide-level verdict is
    operationalised here as: a majority of the types with at least
    ``MIN_CELLS_PER_TYPE`` kept cells must have two components winning
    (BIC margin > 10 and dip p < 0.05) *and* a ratio inside [1.8, 2.2].
    """
    ok = np.isfinite(integral)
    per_type = {}
    for label in np.unique(types[ok]):
        sel = ok & (types == label)
        if sel.sum() < MIN_CELLS_PER_TYPE:
            continue
        per_type[str(label)] = bimodality(np.log(integral[sel]), seed=seed)
    both = [v["two_components_win"] and v["ratio_in_range"]
            for v in per_type.values()]
    ratios = np.array([v["ratio"] for v in per_type.values()])
    return {"gate": 5, "name": "bimodality per type",
            "threshold": "two components win clearly AND ratio in "
                         f"[{GATE5_RATIO_RANGE[0]}, {GATE5_RATIO_RANGE[1]}] "
                         "for a majority of types",
            "n_types": len(per_type),
            "n_types_passing": int(np.sum(both)),
            "frac_types_passing": float(np.mean(both)) if both else 0.0,
            "ratio_median": float(np.median(ratios)) if ratios.size else
                            float("nan"),
            "ratio_range": [float(ratios.min()), float(ratios.max())]
                           if ratios.size else [float("nan")] * 2,
            "per_type": per_type,
            "passed": bool(both and np.mean(both) > 0.5)}


def gate6_bar(slide: dict, integral: np.ndarray, seed: int = 0) -> dict:
    """G2M AUROC of the 4N-vs-2N gate and the MKI67 ratio in top-4 MKI67 types."""
    import anndata as ad
    from sklearn.mixture import GaussianMixture

    from discell.model.cell_cycle import score_cell_cycle

    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(np.isfinite(integral))
    if ok.size > GATE6_SUBSAMPLE:
        ok = np.sort(rng.choice(ok, GATE6_SUBSAMPLE, replace=False))
    adata = ad.read_h5ad(slide["h5ad"], backed="r")
    sub = adata[ok].to_memory()
    genes = np.array([str(g) for g in sub.var_names])
    counts = sub.X
    mki67 = (np.asarray(counts[:, int(np.flatnonzero(genes == "MKI67")[0])]
                        .todense()).ravel() > 0
             if "MKI67" in set(genes) else np.zeros(len(ok), bool))
    scores = score_cell_cycle(counts, genes)
    types = slide["graphclust"][ok]
    values = integral[ok]
    total = slide["total_counts"][ok]

    order = sorted(np.unique(types),
                   key=lambda t: -float(mki67[types == t].mean()))
    top4 = order[:4]
    per_type, states = {}, np.full(len(ok), "", dtype=object)
    for label in top4:
        sel = types == label
        logv = np.log(values[sel]).reshape(-1, 1)
        gmm = GaussianMixture(2, random_state=seed, n_init=3).fit(logv)
        hi = int(np.argmax(gmm.means_.ravel()))
        post = gmm.predict_proba(logv)[:, hi]
        four_n, two_n = post > 0.5, post <= 0.5
        idx = np.flatnonzero(sel)
        states[idx[four_n]] = "4N"
        states[idx[two_n]] = "2N"
        g2m = scores["g2m_score"][sel] if scores else np.full(sel.sum(),
                                                              np.nan)
        ratio = (float(mki67[idx[four_n]].mean() / mki67[idx[two_n]].mean())
                 if mki67[idx[two_n]].mean() > 0 else float("nan"))
        per_type[str(label)] = {
            "n": int(sel.sum()),
            "mki67_positive_fraction": float(mki67[sel].mean()),
            "frac_4N": float(four_n.mean()),
            "component_ratio": float(np.exp(np.ptp(gmm.means_.ravel()))),
            "auroc_g2m_4N_vs_2N": auroc(g2m, four_n),
            "mki67_ratio_4N_over_2N": ratio,
        }
    aurocs = [v["auroc_g2m_4N_vs_2N"] for v in per_type.values()]
    ratios = [v["mki67_ratio_4N_over_2N"] for v in per_type.values()]
    from scipy.stats import spearmanr
    gated = np.isin(states, ["2N", "4N"])
    corr_state_counts = (float(spearmanr((states[gated] == "4N").astype(float),
                                         total[gated]).statistic)
                         if gated.any() else float("nan"))
    return {"gate": 6, "name": "the bar",
            "threshold": f"AUROC >= {GATE6_MIN_AUROC} and MKI67 ratio >= "
                         f"{GATE6_MIN_MKI67_RATIO} in the top-4 MKI67 clusters",
            "n_scored": int(len(ok)),
            "top4_mki67_clusters": [str(t) for t in top4],
            "per_type": per_type,
            "auroc_median": float(np.nanmedian(aurocs)) if aurocs else
                            float("nan"),
            "mki67_ratio_median": float(np.nanmedian(ratios)) if ratios else
                                  float("nan"),
            "split_half_reliability": (scores["reliability"] if scores
                                       else None),
            "n_marker_hits": (list(scores["n_genes"]) if scores else None),
            "corr_integral_total_counts": float(spearmanr(
                values, total).statistic),
            "corr_4N_state_total_counts": corr_state_counts,
            "passed": bool(aurocs and np.nanmin(aurocs) >= GATE6_MIN_AUROC
                           and np.nanmin(ratios) >= GATE6_MIN_MKI67_RATIO)}


# -------------------------------------------------------------------- driver

def _figure(slide: dict, integral: np.ndarray, report: dict, out: Path,
            seed: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ok = np.flatnonzero(np.isfinite(integral))
    pick = rng.choice(ok, min(50_000, ok.size), replace=False)

    col, row = tile_ids(slide["x_um"], slide["y_um"])
    good = np.isfinite(slide["dapi_mean"]) & (slide["dapi_mean"] > 0)
    keys, inverse = np.unique(np.stack([col[good], row[good]], 1), axis=0,
                              return_inverse=True)
    med = _group_medians(inverse, slide["dapi_mean"][good], len(keys))
    sc = axes[0, 0].scatter(keys[:, 0], keys[:, 1], c=med, s=14, marker="s",
                            cmap="viridis")
    fig.colorbar(sc, ax=axes[0, 0], label="tile median dapi_mean")
    axes[0, 0].set_title("gate 1: raw tile field (%.2fx raw, %.2fx residual)"
                         % (report["gate1"]["raw_tile_spread"],
                            report["gate1"]["residual_tile_spread"]))
    axes[0, 0].set_xlabel("tile col (500 um)")
    axes[0, 0].set_ylabel("tile row")
    axes[0, 0].invert_yaxis()

    axes[0, 1].hexbin(np.log(slide["area_px"][pick]), np.log(integral[pick]),
                      gridsize=60, bins="log", cmap="magma")
    axes[0, 1].set_title("gate 4: log integral vs log nuclear area "
                         "(R2 %.3f, slope %.3f)"
                         % (report["gate4"]["r2_on_area"],
                            report["gate4"]["loglog_slope"]))
    axes[0, 1].set_xlabel("log nucleus area (px)")
    axes[0, 1].set_ylabel("log corrected dapi_sum")

    axes[1, 0].hist(np.log(integral[pick]), bins=160, color="0.35")
    axes[1, 0].set_title("gate 5: log corrected integral, all kept cells "
                         "(%d/%d types pass)"
                         % (report["gate5"]["n_types_passing"],
                            report["gate5"]["n_types"]))
    axes[1, 0].set_xlabel("log corrected dapi_sum")

    labels = report["gate6"]["top4_mki67_clusters"]
    for label in labels:
        sel = np.isfinite(integral) & (slide["graphclust"] == label)
        if sel.sum() > 200:
            axes[1, 1].hist(np.log(integral[sel]), bins=80, density=True,
                            histtype="step", label=label)
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].set_title("gate 6: top-4 MKI67 clusters (AUROC median %.3f)"
                         % report["gate6"]["auroc_median"])
    axes[1, 1].set_xlabel("log corrected dapi_sum")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def run(dataset: str, seed: int = 0, tag: str = "") -> dict:
    """All six gates in order; the label decision stops at the first failure."""
    slide = load_slide(dataset)
    g1 = gate1_flat_field(slide)
    log.info("gate 1: raw %.2fx -> residual %.2fx (%s)", g1["raw_tile_spread"],
             g1["residual_tile_spread"], "pass" if g1["passed"] else "FAIL")
    g2 = gate2_background(slide, seed=seed)
    log.info("gate 2: background share %.3f (%s)",
             g2["background_share_of_integral"],
             "pass" if g2["passed"] else "FAIL")
    g3 = gate3_exclusions(slide)
    log.info("gate 3: kept %d / %d cells", g3["n_kept"], g3["n_cells"])

    integral = corrected_integral(slide, g1["_gain"], g2["_level"],
                                  g3["_keep"])
    g4 = gate4_truncation(slide, integral)
    log.info("gate 4: R^2 on area %.3f, slope %.3f (%s)", g4["r2_on_area"],
             g4["loglog_slope"], "pass" if g4["passed"] else "FAIL")
    g5 = gate5_bimodality(integral, slide["graphclust"], seed=seed)
    log.info("gate 5: %d/%d types pass, ratio median %.2f (%s)",
             g5["n_types_passing"], g5["n_types"], g5["ratio_median"],
             "pass" if g5["passed"] else "FAIL")
    g6 = gate6_bar(slide, integral, seed=seed)
    log.info("gate 6: AUROC median %.3f, MKI67 ratio median %.2f (%s)",
             g6["auroc_median"], g6["mki67_ratio_median"],
             "pass" if g6["passed"] else "FAIL")

    gates = [g1, g2, g3, g4, g5, g6]
    failed = [g["gate"] for g in gates if not g["passed"]]
    report = {
        "dataset": dataset, "seed": seed,
        "parameters": {"tile_um": TILE_UM,
                       "background_percentile": BACKGROUND_PERCENTILE,
                       "background_crop_px": BACKGROUND_CROP_PX,
                       "n_background_crops": N_BACKGROUND_CROPS,
                       "min_cells_per_type": MIN_CELLS_PER_TYPE,
                       "gate6_subsample": GATE6_SUBSAMPLE},
        **{f"gate{g['gate']}": {k: v for k, v in g.items()
                                if not k.startswith("_")} for g in gates},
        "first_failed_gate": failed[0] if failed else None,
        "all_gates_passed": not failed,
        "label_shipped": not failed,
    }
    out_dir = paths.dataset(dataset).root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dapi_cycle{('_' + tag) if tag else ''}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float))
    _figure(slide, integral, report, out_dir / f"{stem}.png", seed=seed)
    log.info("wrote %s", out_dir / f"{stem}.json")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="xenium_prime_human_ovary_ff")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    run(args.dataset, seed=args.seed, tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
