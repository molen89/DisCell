#!/usr/bin/env python3
"""Per-cell KL and posterior-uncertainty maps of the final fits.

Pre-registered in the devlog, "Per-cell KL and posterior-uncertainty maps;
within-type Moran's I table (motivation, 2026-09-29; author, run by the
writer)". On average KL_w is tiny at the operating point (alpha_w = 0.1); the
question is the tail: do some cells deviate from the context prior, and which
ones. KL (the posterior's distance from its prior) and the posterior's own
spread are different quantities; both are mapped, for both latents.

Per cell, at ``best.pt`` with posterior means (``sample=False``), every cell
once as a seed (training tiles, then held-out tiles, re-ordered to node
order as ``validate.collect_latents``):

``kl_z``     KL(q(z) || N(0, I)), per dimension and summed;
``kl_w``     KL(q(w) || N(m_psi(c, t), I)) (sigma_w = 1), per dim and summed;
``sigma_z``  the mean posterior standard deviation over z's dimensions;
``nu_w``     the same for w;
``log_l``    log total count; ``held_out``: the cell sits on a val tile.

The per-cell KLs are checked against ``Trainer._sweep`` (the trainer's own
formula) and the held-out mean KL_w against ``runs/<run>/w_deviation.json``.

Statistics (rules fixed in the devlog before the first read), per quantity,
over the metric targets (``eval_mask``: Unassigned is not a target but stays
a neighbour in every graph feature):

* depth: Spearman(quantity, log l). |rho| >= ``DEPTH_RHO`` -> the primary
  ranking is the within-type OLS residual of the quantity on log l; else the
  raw value. Both rankings are read; the pass rule uses the primary.
* the top decile (the top 10 % of target cells; for sigma / nu the most
  uncertain) against the other 90 %: odds ratios for every target type, every
  K = 10 niche (``validate.niche_labels``; k-means seed fixed at
  ``NICHE_SEED`` for every run so the niche ids are shared across seeds),
  isolated (degree 0) and held-out tile; standardised mean differences for
  the heterotypic-neighbour share (pruned graph, isolated cells excluded),
  degree and log l. 95 % percentile intervals from the tile bootstrap of
  ``bootstrap`` (``TILE_UM`` squares, resampled with replacement, ``N_BOOT``
  draws; conditional on the top-decile call and the niche labels).
* concentration: the share of the quantity's total over target cells held by
  the top decile.
* Moran's I of each map: ``validate.morans_i`` on
  ``validate.row_normalised_graph`` over the target cells, values centred per
  type, as ``validate.analysis_morans``; ``N_PERM`` permutations. The same on
  log10 of the quantity is recorded beside it (secondary: a heavy tail can
  swamp the raw I).
* log10-spaced histogram of each quantity (target cells).

Summary over finalL_s0-s2 (``--summary``): per-cell Spearman between seeds,
top-decile overlap, and the pass rule -- OR >= 1.5 (or <= 1/1.5) or
|SMD| >= 0.3 with the CI excluding the null, same direction, in all three
seeds (each seed's primary ranking).

Usage::

    python -m discell.experiments.kl_maps --dataset <id> --run finalL_s0
    python -m discell.experiments.kl_maps --dataset <id> --run finalL_s0 --stats-only
    python -m discell.experiments.kl_maps --dataset <id> --summary

Writes ``experiments/kl_maps_<run>.{npz,json}`` and
``experiments/kl_maps_summary.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from discell.model import eval_mask as EM

log = logging.getLogger("discell.experiments.kl_maps")

QUANTITIES = ("kl_z", "kl_w", "sigma_z", "nu_w")
TOP_FRACTION = 0.10
DEPTH_RHO = 0.3
OR_PASS = 1.5
SMD_PASS = 0.3
N_BOOT = 500
N_PERM = 200
K_NICHE = 10
NICHE_SEED = 0
N_BINS = 60
RUNS = ("finalL_s0", "finalL_s1", "finalL_s2")


# -- the per-cell read --------------------------------------------------------

def collect(trainer) -> dict:
    """Posterior parameters of every cell once, node-ordered.

    Mirrors ``Trainer._sweep`` over training then validation tiles, seeds
    only, eval mode, ``sample=False``; keeps the log-variances it drops.
    """
    import torch

    model = trainer.model
    model.eval()
    keys = ("nodes", "mu_z", "logvar_z", "mu_w", "logvar_w", "prior_w")
    out: dict = {k: [] for k in keys}
    with torch.no_grad():
        for batch in trainer.train_batches + trainer.val_batches:
            fwd = model(**trainer._forward_kwargs(batch),
                        kappa=trainer.config.kappa, sample=False)
            n = batch["n_seeds"]
            out["nodes"].append(np.asarray(batch["nodes"][:n]))
            out["mu_z"].append(fwd.mu_z[:n].cpu().numpy())
            out["logvar_z"].append(fwd.logvar_z[:n].cpu().numpy())
            out["mu_w"].append(fwd.mu_w[:n].cpu().numpy())
            out["logvar_w"].append(fwd.logvar_w[:n].cpu().numpy())
            out["prior_w"].append(fwd.prior_mean_w[:n].cpu().numpy())
    cat = {k: np.concatenate(v) for k, v in out.items()}
    order = np.argsort(cat["nodes"])
    assert np.array_equal(cat["nodes"][order], np.arange(len(order))), \
        "tiles must partition the cells exactly once"
    return {k: v[order] for k, v in cat.items() if k != "nodes"}


def per_cell(lat: dict) -> dict:
    """KLs (per dim and summed) and mean posterior SDs from :func:`collect`."""
    lz, lw = lat["logvar_z"].astype(np.float64), lat["logvar_w"].astype(np.float64)
    kl_z_dim = 0.5 * (-lz + np.exp(lz) + lat["mu_z"].astype(np.float64) ** 2 - 1.0)
    kl_w_dim = 0.5 * (-lw + np.exp(lw) + (lat["mu_w"].astype(np.float64)
                                          - lat["prior_w"]) ** 2 - 1.0)
    return {"kl_z_dim": kl_z_dim, "kl_z": kl_z_dim.sum(axis=1),
            "kl_w_dim": kl_w_dim, "kl_w": kl_w_dim.sum(axis=1),
            "sigma_z": np.exp(0.5 * lz).mean(axis=1),
            "nu_w": np.exp(0.5 * lw).mean(axis=1)}


def sanity_check(trainer, cells: dict, held_out: np.ndarray,
                 run_dir: Path) -> dict:
    """This read against the trainer's ``_sweep`` and ``w_deviation.json``."""
    swept = trainer._sweep(trainer.train_batches + trainer.val_batches)
    trainer.model.eval()                      # _sweep leaves it in train mode
    order = np.argsort(swept["nodes"])
    kl_z, kl_w = swept["kl_z"][order], swept["kl_w"][order]
    diff_z = float(np.abs(cells["kl_z"] - kl_z).max())
    diff_w = float(np.abs(cells["kl_w_dim"] - kl_w).max())
    ok = (np.allclose(cells["kl_z"], kl_z, rtol=1e-4, atol=1e-5)
          and np.allclose(cells["kl_w_dim"], kl_w, rtol=1e-4, atol=1e-5))
    out = {"sweep_max_abs_diff_kl_z": diff_z,
           "sweep_max_abs_diff_kl_w_dim": diff_w, "sweep_match": bool(ok),
           "mean_kl_w_val_all": float(cells["kl_w"][held_out].mean()),
           "n_val_cells": int(held_out.sum())}
    path = run_dir / "w_deviation.json"
    if path.exists():
        stored = json.loads(path.read_text())
        out["w_deviation_kl_w_sum"] = stored.get("kl_w_sum")
        if stored.get("kl_w_per_dim_recomputed") is not None:
            out["w_deviation_kl_w_recomputed_sum"] = float(
                np.sum(stored["kl_w_per_dim_recomputed"]))
        out["w_deviation_n_val_cells"] = stored.get("n_val_cells")
    assert ok, f"per-cell KL disagrees with Trainer._sweep: {out}"
    return out


def graph_features(n: int, edge_i: np.ndarray, edge_j: np.ndarray,
                   t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Degree and heterotypic-neighbour share on the pruned graph (every
    neighbour counts, Unassigned included); share is NaN for isolated cells."""
    degree = (np.bincount(edge_i, minlength=n)
              + np.bincount(edge_j, minlength=n)).astype(np.float64)
    other = (t[edge_i] != t[edge_j]).astype(np.float64)
    hetero = (np.bincount(edge_i, other, minlength=n)
              + np.bincount(edge_j, other, minlength=n))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(degree > 0, hetero / degree, np.nan)
    return degree, share


def pixel_xy(dataset: str, variant: str, xy_um: np.ndarray) -> tuple:
    """``obsm['spatial']`` (pixels) of the bundle, rows in the data's order,
    checked against the microns the model data carries."""
    import h5py

    from discell import paths

    path = paths.dataset(dataset).bundle_dir / f"{variant}.h5ad"
    with h5py.File(path, "r") as f:
        xy_px = np.asarray(f["obsm"]["spatial"][:], dtype=np.float32)
        mpp = float(f["uns"]["microns_per_pixel"][()])
    assert xy_px.shape == xy_um.shape
    err = float(np.abs(xy_px.astype(np.float64) * mpp - xy_um).max())
    assert err < 1.0, f"pixel and micron coordinates disagree by {err} um"
    return xy_px, mpp, err


def read_run(dataset: str, run: str, device: str) -> tuple[dict, dict]:
    """Every per-cell array of one run (the npz payload) and its checks."""
    from discell.model import validate as V

    t0 = time.time()
    config, data, trainer, run_dir, _ = V.load_run(dataset, run, device)
    t_load = time.time() - t0
    t0 = time.time()
    lat = collect(trainer)
    t_fwd = time.time() - t0
    cells = per_cell(lat)
    n = data.graph.n_cells
    held_out = np.zeros(n, dtype=bool)
    held_out[np.concatenate(data.val_tiles)] = True
    model_tile = np.zeros(n, dtype=np.int32)
    for k, tile in enumerate(data.train_tiles + data.val_tiles):
        model_tile[tile] = k
    t0 = time.time()
    sanity = sanity_check(trainer, cells, held_out, run_dir)
    sanity["mean_kl_w_val_targets"] = float(cells["kl_w"][
        held_out & EM.metric_target_mask(data.t, data.type_names)].mean())
    t_check = time.time() - t0
    degree, share = graph_features(n, data.graph.edge_i, data.graph.edge_j,
                                   data.t)
    assert np.array_equal(degree, data.graph.degrees), "degree mismatch"
    t0 = time.time()
    niche = V.niche_labels(data, K_NICHE, NICHE_SEED)
    t_niche = time.time() - t0
    xy_px, mpp, xy_err = pixel_xy(dataset, config.variant, data.positions)

    f32 = lambda a: np.asarray(a, dtype=np.float32)
    payload = {
        **{k: f32(v) for k, v in cells.items()},
        **{k: f32(lat[k]) for k in ("mu_z", "logvar_z", "mu_w", "logvar_w",
                                    "prior_w")},
        "log_l": np.log(np.clip(data.totals, 1.0, None)).astype(np.float32),
        "held_out": held_out, "model_tile": model_tile,
        "t": data.t.astype(np.int32),
        "type_names": np.asarray([str(s) for s in data.type_names]),
        "xy_px": xy_px, "xy_um": f32(data.positions),
        "degree": degree.astype(np.int32), "hetero_share": f32(share),
        "niche": niche.astype(np.int32),
        "edge_i": data.graph.edge_i.astype(np.int32),
        "edge_j": data.graph.edge_j.astype(np.int32),
    }
    meta = {"dataset": dataset, "run": run, "seed": int(config.seed),
            "variant": config.variant, "alpha_w": config.alpha_w,
            "n_cells": int(n), "device": str(trainer.device),
            "microns_per_pixel": mpp, "xy_px_vs_um_max_err_um": xy_err,
            "sanity": sanity,
            "timing_s": {"load": t_load, "forward": t_fwd,
                         "sanity_sweep": t_check, "niche_kmeans": t_niche}}
    return payload, meta


# -- statistics helpers -------------------------------------------------------

def depth_residual(q: np.ndarray, log_l: np.ndarray,
                   t: np.ndarray) -> np.ndarray:
    """Within-type OLS residual of *q* on *log_l* (intercept + slope per type)."""
    out = np.empty(len(q), dtype=np.float64)
    for g in np.unique(t):
        m = t == g
        design = np.stack([np.ones(m.sum()), log_l[m]], axis=1)
        coef, *_ = np.linalg.lstsq(design, q[m], rcond=None)
        out[m] = q[m] - design @ coef
    return out


def top_decile(score: np.ndarray, fraction: float = TOP_FRACTION) -> np.ndarray:
    """Bool mask of the ``round(fraction * n)`` largest scores."""
    k = int(round(fraction * len(score)))
    top = np.zeros(len(score), dtype=bool)
    top[np.argsort(-score, kind="stable")[:k]] = True
    return top


def odds_ratio(top: np.ndarray, member: np.ndarray,
               weights: np.ndarray | None = None) -> float:
    """OR of membership in the top vs the rest (+0.5 in every cell)."""
    w = np.ones(len(top)) if weights is None else weights
    a, b = w[top & member].sum(), w[top & ~member].sum()
    c, d = w[~top & member].sum(), w[~top & ~member].sum()
    return float(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))


def smd(top: np.ndarray, x: np.ndarray,
        weights: np.ndarray | None = None) -> float:
    """(mean_top - mean_rest) / sqrt((var_top + var_rest) / 2); NaN x dropped."""
    w = np.ones(len(top)) if weights is None else weights
    ok = np.isfinite(x)
    stats = []
    for g in (top & ok, ~top & ok):
        m = np.average(x[g], weights=w[g])
        stats.append((m, np.average((x[g] - m) ** 2, weights=w[g])))
    (m1, v1), (m0, v0) = stats
    return float((m1 - m0) / np.sqrt(max((v1 + v0) / 2.0, 1e-300)))


def tile_multiplicities(n_tiles: int, n: int, seed: int) -> np.ndarray:
    """(n, n_tiles) with-replacement tile counts, the RNG stream of
    ``bootstrap.tile_draws`` (which expands the same draws to cells)."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_tiles, n_tiles),
                                 minlength=n_tiles) for _ in range(n)]
                    ).astype(np.float64)


def _ci(values: np.ndarray) -> list:
    finite = values[np.isfinite(values)]
    return ([float(np.percentile(finite, 2.5)),
             float(np.percentile(finite, 97.5))] if len(finite) >= 2
            else [float("nan")] * 2)


def enrichment(top: np.ndarray, categorical: dict, continuous: dict,
               tiles: np.ndarray, n_tiles: int, mult: np.ndarray) -> dict:
    """ORs and SMDs of *top* vs the rest with tile-bootstrap intervals.

    *categorical*: name -> (labels int array, {code: label name}); an OR per
    listed code (membership = labels == code). *continuous*: name -> float
    array (NaN = not defined for that cell). *mult*: (B, n_tiles) draws.
    Each draw re-weights per-tile sufficient statistics, so the point
    estimate equals :func:`odds_ratio` / :func:`smd` at unit weights.
    """
    ones = np.ones((1, n_tiles))
    per_tile = lambda mask, values=None: np.bincount(
        tiles[mask], None if values is None else values[mask],
        minlength=n_tiles)
    tot = np.stack([per_tile(top), per_tile(~top)], axis=1)       # (T, 2)
    out: dict = {}
    for name, (labels, codes) in categorical.items():
        entry = {}
        for code, label in codes.items():
            member = labels == code
            cnt = np.stack([per_tile(top & member), per_tile(~top & member)],
                           axis=1)
            vals = []
            for m in (ones, mult):
                a, c = m @ cnt[:, 0], m @ cnt[:, 1]
                b, d = m @ tot[:, 0] - a, m @ tot[:, 1] - c
                vals.append(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
            entry[str(label)] = {
                "kind": "or", "estimate": float(vals[0][0]),
                "ci95": _ci(vals[1]),
                "n_top": int((top & member).sum()),
                "n_rest": int((~top & member).sum())}
        out[name] = entry
    for name, x in continuous.items():
        ok = np.isfinite(x)
        xx = np.where(ok, x, 0.0).astype(np.float64)
        sums = [np.stack([per_tile(g & ok), per_tile(g & ok, xx),
                          per_tile(g & ok, xx ** 2)], axis=1)
                for g in (top, ~top)]                             # (T, 3)
        vals = []
        for m in (ones, mult):
            moments = []
            for s in sums:
                n_, s1, s2 = (m @ s).T
                mean = s1 / n_
                moments.append((mean, s2 / n_ - mean ** 2))
            (m1, v1), (m0, v0) = moments
            vals.append((m1 - m0) / np.sqrt(np.clip((v1 + v0) / 2, 1e-300,
                                                    None)))
        out[name] = {"kind": "smd", "estimate": float(vals[0][0]),
                     "ci95": _ci(vals[1]),
                     "mean_top": float(xx[top & ok].mean()),
                     "mean_rest": float(xx[~top & ok].mean())}
    return out


def passes(entry: dict) -> int:
    """+1 / -1 when an enrichment passes the fixed rule upwards / downwards;
    0 otherwise."""
    est, (lo, hi) = entry["estimate"], entry["ci95"]
    if entry["kind"] == "or":
        if est >= OR_PASS and lo > 1.0:
            return 1
        if est <= 1.0 / OR_PASS and hi < 1.0:
            return -1
        return 0
    if est >= SMD_PASS and lo > 0.0:
        return 1
    if est <= -SMD_PASS and hi < 0.0:
        return -1
    return 0


def histogram(q: np.ndarray, n_bins: int = N_BINS) -> dict:
    positive = q[q > 0]
    edges = np.logspace(np.log10(positive.min()), np.log10(positive.max()),
                        n_bins + 1)
    counts, _ = np.histogram(positive, bins=edges)
    return {"log10_edges": np.log10(edges).tolist(), "counts": counts.tolist(),
            "n_nonpositive": int((q <= 0).sum()), "n": int(len(q))}


def run_stats(p: dict, n_boot: int = N_BOOT, n_perm: int = N_PERM,
              seed: int = 0) -> dict:
    """Every statistic of one run from its per-cell payload (npz keys)."""
    from scipy.stats import spearmanr

    from discell.experiments.bootstrap import TILE_UM, tile_index
    from discell.model import validate as V

    names = [str(s) for s in p["type_names"]]
    t = p["t"].astype(np.int64)
    targets = EM.metric_target_mask(t, names)
    rows = np.flatnonzero(targets)
    log_l = p["log_l"].astype(np.float64)
    tiles, n_tiles = tile_index(p["xy_um"][rows], TILE_UM)
    mult = tile_multiplicities(n_tiles, n_boot, seed)

    type_codes = {g: names[g] for g in np.unique(t[rows])}
    niche_codes = {k: f"niche{k}" for k in range(K_NICHE)}
    categorical = {
        "type": (t[rows], type_codes),
        "niche": (p["niche"][rows], niche_codes),
        "isolated": ((p["degree"][rows] == 0).astype(np.int64), {1: "isolated"}),
        "held_out": (p["held_out"][rows].astype(np.int64), {1: "held_out"}),
    }
    continuous = {"hetero_share": p["hetero_share"][rows].astype(np.float64),
                  "degree": p["degree"][rows].astype(np.float64),
                  "log_l": log_l[rows]}

    out: dict = {"n_targets": int(len(rows)), "n_cells": int(len(t)),
                 "eval_mask": EM.record(names, t),
                 "rules": {"top_fraction": TOP_FRACTION, "depth_rho": DEPTH_RHO,
                           "or_pass": OR_PASS, "smd_pass": SMD_PASS,
                           "n_boot": n_boot, "tile_um": TILE_UM,
                           "n_tiles": n_tiles, "n_perm": n_perm,
                           "k_niche": K_NICHE, "niche_seed": NICHE_SEED,
                           "or_smoothing": 0.5},
                 "n_isolated_targets": int((p["degree"][rows] == 0).sum()),
                 "n_held_out_targets": int(p["held_out"][rows].sum()),
                 "quantities": {}}

    # Moran's I, as validate.analysis_morans: graph induced on the targets,
    # values centred per type over its connected cells
    graph = SimpleNamespace(graph=SimpleNamespace(
        n_cells=len(t), edge_i=p["edge_i"].astype(np.int64),
        edge_j=p["edge_j"].astype(np.int64)))
    weights, connected = V.row_normalised_graph(graph, targets)
    weights = weights[connected][:, connected]
    raw = np.stack([p[q].astype(np.float64) for q in QUANTITIES], axis=1)
    logged = np.log10(np.clip(raw, 1e-12, None))
    moran = {}
    for label, values in (("raw", raw), ("log10", logged)):
        centred = V.center_per_type(values, t, connected)[connected]
        moran[label] = V.morans_i(centred, weights, n_perms=n_perm, seed=seed)

    for j, q in enumerate(QUANTITIES):
        value = p[q].astype(np.float64)[rows]
        rho = float(spearmanr(value, log_l[rows]).statistic)
        resid = depth_residual(value, log_l[rows], t[rows])
        primary = "residual" if abs(rho) >= DEPTH_RHO else "raw"
        entry: dict = {
            "depth_spearman": rho, "primary": primary,
            "mean": float(value.mean()), "median": float(np.median(value)),
            "quantiles": {str(k): float(np.quantile(value, k))
                          for k in (0.5, 0.9, 0.99, 0.999)},
            "histogram": histogram(value),
            "morans_i": {lab: {"I": moran[lab]["I"][j],
                               "null_lo": moran[lab]["null_lo"][j],
                               "null_hi": moran[lab]["null_hi"][j]}
                         for lab in moran},
            "rankings": {}}
        if q in ("kl_z", "kl_w"):
            dim = p[f"{q}_dim"].astype(np.float64)[rows]
            entry["mean_per_dim"] = dim.mean(axis=0).tolist()
        total = value.sum()
        for ranking, score in (("raw", value), ("residual", resid)):
            top = top_decile(score)
            entry["rankings"][ranking] = {
                "n_top": int(top.sum()),
                "threshold": float(score[top].min()),
                "concentration": float(value[top].sum() / total),
                "enrichment": enrichment(top, categorical, continuous, tiles,
                                         n_tiles, mult)}
        out["quantities"][q] = entry
        log.info("%s: depth rho %+.3f -> %s; top-decile share %.3f; Moran I "
                 "%.4f [null %.4f, %.4f]", q, rho, primary,
                 entry["rankings"][primary]["concentration"],
                 entry["morans_i"]["raw"]["I"], entry["morans_i"]["raw"]["null_lo"],
                 entry["morans_i"]["raw"]["null_hi"])
    out["n_morans_cells"] = int(connected.sum())
    return out


# -- the dataset summary --------------------------------------------------------

def _flat(enrich: dict):
    for feature, entry in enrich.items():
        if "kind" in entry:
            yield feature, entry
        else:
            for label, sub in entry.items():
                yield f"{feature}:{label}", sub


def summarise(dataset: str, runs: Sequence[str] = RUNS) -> dict:
    from scipy.stats import spearmanr

    from discell import paths

    root = paths.dataset(dataset).root / "experiments"
    stats = {r: json.loads((root / f"kl_maps_{r}.json").read_text())["stats"]
             for r in runs}
    npz = {r: np.load(root / f"kl_maps_{r}.npz") for r in runs}
    names = [str(s) for s in npz[runs[0]]["type_names"]]
    t = npz[runs[0]]["t"].astype(np.int64)
    for r in runs[1:]:
        assert np.array_equal(npz[r]["t"], t), "seeds must share the cells"
    targets = EM.metric_target_mask(t, names)
    out: dict = {"dataset": dataset, "runs": list(runs),
                 "n_targets": int(targets.sum()),
                 "niche_labels_identical": {
                     f"{a}-{b}": bool(np.array_equal(npz[a]["niche"],
                                                     npz[b]["niche"]))
                     for a, b in itertools.combinations(runs, 2)},
                 "quantities": {}}
    for q in QUANTITIES:
        values = {r: npz[r][q].astype(np.float64)[targets] for r in runs}
        tops = {}
        for r in runs:
            st = stats[r]["quantities"][q]
            if st["primary"] == "residual":
                score = depth_residual(values[r],
                                       npz[r]["log_l"][targets].astype(np.float64),
                                       t[targets])
            else:
                score = values[r]
            tops[r] = top_decile(score)
        pairs = {}
        for a, b in itertools.combinations(runs, 2):
            inter = (tops[a] & tops[b]).sum()
            pairs[f"{a}-{b}"] = {
                "spearman": float(spearmanr(values[a], values[b]).statistic),
                "top_decile_jaccard": float(inter / (tops[a] | tops[b]).sum())}
        per_seed = {r: {"primary": stats[r]["quantities"][q]["primary"],
                        "depth_spearman": stats[r]["quantities"][q]["depth_spearman"],
                        "concentration": stats[r]["quantities"][q]["rankings"][
                            stats[r]["quantities"][q]["primary"]]["concentration"],
                        "morans_i": stats[r]["quantities"][q]["morans_i"]}
                    for r in runs}
        flat = {r: dict(_flat(stats[r]["quantities"][q]["rankings"][
            stats[r]["quantities"][q]["primary"]]["enrichment"])) for r in runs}
        passing = []
        for key in flat[runs[0]]:
            calls = [passes(flat[r][key]) if key in flat[r] else 0
                     for r in runs]
            if calls[0] != 0 and all(c == calls[0] for c in calls):
                passing.append({
                    "feature": key, "direction": "up" if calls[0] > 0 else "down",
                    # counts carried so a pass resting on a handful of cells
                    # is visible (the rule sets no minimum)
                    "per_seed": {r: {k: flat[r][key][k] for k in
                                     ("estimate", "ci95", "n_top", "n_rest",
                                      "mean_top", "mean_rest")
                                     if k in flat[r][key]}
                                 for r in runs}})
        out["quantities"][q] = {"per_seed": per_seed, "cross_seed": pairs,
                                "passing": passing}
    return out


# -- CLI ------------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main(argv: Sequence[str] | None = None) -> int:
    from discell import paths

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stats-only", action="store_true",
                        help="recompute the statistics from the saved npz")
    parser.add_argument("--summary", action="store_true",
                        help="write kl_maps_summary.json over finalL_s0-s2")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    out_dir = paths.dataset(args.dataset).root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary:
        summary = _jsonable(summarise(args.dataset))
        path = out_dir / "kl_maps_summary.json"
        path.write_text(json.dumps(summary, indent=1))
        log.info("wrote %s", path)
        return 0
    if not args.run:
        parser.error("--run is required unless --summary")

    npz_path = out_dir / f"kl_maps_{args.run}.npz"
    json_path = out_dir / f"kl_maps_{args.run}.json"
    if args.stats_only:
        payload = dict(np.load(npz_path))
        meta = (json.loads(json_path.read_text()) if json_path.exists()
                else {"dataset": args.dataset, "run": args.run})
        meta.pop("stats", None)
    else:
        t0 = time.time()
        payload, meta = read_run(args.dataset, args.run, args.device)
        np.savez(npz_path, **payload)
        meta["timing_s"]["read_total"] = time.time() - t0
        s = meta["sanity"]
        log.info("sanity: max |dKL_z| %.3g, max |dKL_w| %.3g vs _sweep; mean "
                 "KL_w over %d val cells %.5f (w_deviation kl_w_sum %s)",
                 s["sweep_max_abs_diff_kl_z"], s["sweep_max_abs_diff_kl_w_dim"],
                 s["n_val_cells"], s["mean_kl_w_val_all"],
                 s.get("w_deviation_kl_w_sum"))
        log.info("wrote %s", npz_path)
    t0 = time.time()
    meta["stats"] = run_stats(payload, n_boot=args.n_boot, n_perm=args.n_perm)
    meta.setdefault("timing_s", {})["stats"] = time.time() - t0
    json_path.write_text(json.dumps(_jsonable(meta), indent=1))
    log.info("wrote %s", json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
