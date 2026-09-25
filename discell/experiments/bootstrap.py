#!/usr/bin/env python3
"""Tile-bootstrap intervals for the headline reads (metrics package, item 1).

Devlog "Metrics package for the final tables (motivation, 2026-09-25 07:30)",
review R33. Every headline read gets a 95 % percentile interval from
resampling whole ``TILE_UM`` square tiles of the cells it is computed on,
with replacement, ``N_BOOT`` times -- the scheme ``r12_depth_test`` already
uses, which respects the spatial dependence of neighbouring cells. A draw
turns into one multiplicity per cell (its tile's), and every read here is
written as a function of those per-cell weights, so a draw costs a weighted
re-sum and never a refit.

That makes every interval **conditional on the fitted objects**: the ridge
and MLP probes, the k-means clustering behind NMI, the cycle ridge, the kNN
balls of the MI estimator and the model's predictions stay as fitted on the
full sample; what is resampled is the held-out cells they are graded on.
It is the sampling uncertainty of the grade, not of the fit.

Reads and their per-cell artefacts (all recomputed from ``best.pt`` via
``validate.load_run``, and each point estimate checked against the number
already on disk -- ``reproduces`` in the output):

``probe_{ridge,mlp}_{comp,img}_{excess,frac}``
    per held-out cell and column: squared errors of the type-mean baseline,
    the probe and each within-type permutation floor; ``frac`` divides by
    the uncontrolled reference excess recorded in ``probe_blocks.json``
    (held fixed).
``cycle_z``, ``cycle_w``
    pooled within-type cycle R^2: per held-out cell residual and target.
``nmi``
    per clustered cell: cluster label and type (weighted contingency).
``w_niche_mi_excess``
    I(niche; w) minus its within-type permutation floor, Ross estimator:
    per-cell terms of the estimator.
``transport_of_ceiling``, ``transport_of_ceiling_trusted``
    mean read, extrapolation tier: per panel the held-out cells of both
    niches (observed shift and its split-half noise ceiling are recomputed
    per draw); tier membership is held at the point estimate's.
``readA_gap_group``, ``readA_gap_own``, ``readA_type_mean_own``
    distribution read, pairwise panels: the source cells each panel scored
    (weighted unbiased MMD^2); target cells and the floor are held fixed.

Multiplicity across the kappa grid is one stated treatment: Holm over the
grid's pairwise comparisons (:func:`holm`, :func:`paired_comparisons`).

Usage::

    python -m discell.experiments.bootstrap --dataset <id> --run <run>
    python -m discell.experiments.bootstrap --dataset <id> --run <run> \\
        --reads probe,cycle,nmi,w_mi --n 1000

Writes ``runs/<run>/bootstrap_ci.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from typing import Callable, Mapping, Sequence

import numpy as np

log = logging.getLogger("discell.experiments.bootstrap")

#: Side of the square tiles the bootstrap resamples (um), as r12_depth_test.
TILE_UM = 200.0
N_BOOT = 1000
#: Family-wise level of the Holm step-down.
ALPHA = 0.05
READS = ("probe", "cycle", "nmi", "w_mi", "transport_mean", "transport_dist")


# --------------------------------------------------------------------------
# the core
# --------------------------------------------------------------------------


def tile_index(cells_xy: np.ndarray, tile_um: float = TILE_UM
               ) -> tuple[np.ndarray, int]:
    """Contiguous tile id per cell (``tile_um`` squares) and the tile count."""
    xy = np.floor(np.asarray(cells_xy, dtype=np.float64)[:, :2] / tile_um
                  ).astype(np.int64)
    xy -= xy.min(axis=0)
    _, ids = np.unique(xy[:, 0] * (int(xy[:, 1].max()) + 1) + xy[:, 1],
                       return_inverse=True)
    return ids.astype(np.int64), int(ids.max()) + 1


def tile_draws(tiles: np.ndarray, n_tiles: int, n: int, seed: int):
    """Yield *n* per-cell weight vectors: tiles drawn with replacement,
    each cell weighted by its tile's multiplicity."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        mult = np.bincount(rng.integers(0, n_tiles, n_tiles),
                           minlength=n_tiles).astype(np.float64)
        yield mult[tiles]


def weighted_mean(values: np.ndarray, weights: np.ndarray):
    """Column means of *values* (cells on axis 0) under per-cell *weights*."""
    total = weights.sum()
    if total <= 0:
        return np.full(np.shape(values)[1:], np.nan) if np.ndim(values) > 1 \
            else float("nan")
    out = np.tensordot(weights, values, axes=1) / total
    return float(out) if np.ndim(out) == 0 else out


def _as_dict(value) -> dict:
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    return {"value": float(value)}


def _summarise(point: dict, draws: dict, alpha: float) -> dict:
    out = {}
    for key, estimate in point.items():
        v = np.asarray(draws[key], dtype=np.float64)
        # an undefined point estimate (e.g. a ceiling below 0.05) gets no CI
        finite = np.isfinite(v) & bool(np.isfinite(estimate))
        out[key] = {
            "estimate": estimate,
            "ci95": ([float(np.percentile(v[finite], 100 * alpha / 2)),
                      float(np.percentile(v[finite], 100 * (1 - alpha / 2)))]
                     if finite.sum() >= 2 else [float("nan")] * 2),
            "se": float(np.std(v[finite], ddof=1)) if finite.sum() >= 2
            else float("nan"),
            "n_finite": int(finite.sum())}
    return out


def tile_bootstrap(cells_xy: np.ndarray, values,
                   statistic: Callable | None = None, n: int = N_BOOT,
                   tile_um: float = TILE_UM, seed: int = 0,
                   alpha: float = ALPHA, keep_draws: bool = False) -> dict:
    """Estimate and percentile ``1 - alpha`` interval of ``statistic``.

    *values* is whatever *statistic* reads -- per-cell arrays (cells on axis
    0, in the order of *cells_xy*), or per-panel structures that index into
    the cells -- and ``statistic(values, weights)`` returns a float or a
    dict of floats, *weights* being one non-negative multiplicity per cell.
    The default statistic is the weighted mean of a 1-D *values*.

    The point estimate is the statistic at unit weights; each of the *n*
    draws resamples the ``tile_um`` tiles the cells occupy, with
    replacement. Returns, per key (``value`` for a scalar statistic),
    ``estimate``, ``ci95``, ``se`` and ``n_finite``, plus ``n_boot``,
    ``n_tiles``, ``n_cells``, ``tile_um``; with *keep_draws* also the raw
    draws (for paired comparisons).
    """
    statistic = statistic or (lambda v, w: weighted_mean(v, w))
    tiles, n_tiles = tile_index(cells_xy, tile_um)
    point = _as_dict(statistic(values, np.ones(len(tiles))))
    draws: dict[str, list] = {k: [] for k in point}
    for weights in tile_draws(tiles, n_tiles, n, seed):
        for key, value in _as_dict(statistic(values, weights)).items():
            draws[key].append(value)
    out = {"reads": _summarise(point, draws, alpha), "n_boot": int(n),
           "n_tiles": n_tiles, "n_cells": int(len(tiles)),
           "tile_um": float(tile_um), "seed": int(seed)}
    if list(point) == ["value"]:
        out.update(out.pop("reads")["value"])
    if keep_draws:
        out["draws"] = {k: np.asarray(v) for k, v in draws.items()}
    return out


def holm(pvalues: Sequence[float], alpha: float = ALPHA) -> dict:
    """Holm's step-down over one family: adjusted p-values (monotone, capped
    at 1) and which hypotheses are rejected at family-wise level *alpha*."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return {"p_adjusted": adjusted.tolist(),
            "reject": (adjusted <= alpha).tolist(), "alpha": alpha,
            "m": m, "method": "Holm step-down"}


def bootstrap_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value of ``difference = 0`` from its draws
    (the share of draws on the far side of zero, doubled, +1 corrected)."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    tail = min((d <= 0).sum(), (d >= 0).sum())
    return float(min(1.0, 2.0 * (tail + 1) / (len(d) + 1)))


def paired_comparisons(cells_xy: np.ndarray, per_label: Mapping[str, object],
                       statistic: Callable | None = None, n: int = N_BOOT,
                       tile_um: float = TILE_UM, seed: int = 0,
                       alpha: float = ALPHA,
                       pairs: Sequence[tuple[str, str]] | None = None) -> dict:
    """Every pairwise difference of one read across a grid (the kappa grid),
    on the SAME cells and the SAME tile draws, with Holm over the family.

    *per_label* maps a grid label to the values its read is computed from
    (per-cell arrays in the order of *cells_xy*, e.g. each kappa run's
    per-cell held-out reconstruction on a shared validation split);
    *statistic* is as in :func:`tile_bootstrap`. Pairs default to every
    unordered pair in *per_label*'s order.
    """
    statistic = statistic or (lambda v, w: weighted_mean(v, w))
    labels = list(per_label)
    pairs = list(pairs or itertools.combinations(labels, 2))

    def joint(values, weights):
        return {lab: float(statistic(values[lab], weights)) for lab in labels}

    boot = tile_bootstrap(cells_xy, dict(per_label), joint, n, tile_um, seed,
                          alpha, keep_draws=True)
    rows = []
    for a, b in pairs:
        d = boot["draws"][a] - boot["draws"][b]
        finite = d[np.isfinite(d)]
        rows.append({"a": a, "b": b,
                     "difference": boot["reads"][a]["estimate"]
                     - boot["reads"][b]["estimate"],
                     "ci95": [float(np.percentile(finite, 100 * alpha / 2)),
                              float(np.percentile(finite, 100 * (1 - alpha / 2)))],
                     "p": bootstrap_p(d)})
    adjusted = holm([r["p"] for r in rows], alpha)
    for row, p_adj, rej in zip(rows, adjusted["p_adjusted"], adjusted["reject"]):
        row["p_holm"], row["reject_holm"] = p_adj, bool(rej)
    return {"levels": {lab: boot["reads"][lab] for lab in labels},
            "comparisons": rows, "family": {k: v for k, v in adjusted.items()
                                            if k not in ("p_adjusted", "reject")},
            "n_boot": boot["n_boot"], "n_tiles": boot["n_tiles"],
            "n_cells": boot["n_cells"], "tile_um": boot["tile_um"]}


# --------------------------------------------------------------------------
# adapters: per-cell artefacts of each headline read, and the statistic on
# them. Each ``*_cells`` reproduces its metric's random stream exactly, so
# the statistic at unit weights IS the number on disk.
# --------------------------------------------------------------------------

_EPS = 1e-12                          # metrics._MSE_EPS


def probe_cells(z, t, v, vbar_t, train, test, n_comp: int, seed: int = 0,
                n_perm: int = 5, family: str = "ridge") -> dict:
    """Per held-out cell squared errors behind ``metrics.probe_gain_per_block``
    (*family* ``ridge``) or ``probe_gain_per_block_mlp`` (``mlp``).

    Runs the metric's own ``_probe_columns`` with a recording grader, so the
    split, subsample, permutations and fits are the metric's; the test rows
    are recovered by replaying the head of its random stream.
    """
    from discell.model import metrics as M

    z = np.asarray(z)
    if family == "ridge":
        z_in, v_in, vbar_in = z, v, vbar_t
        fit = M._ridge_fit_predict
    else:
        from discell.model.calibrate import mlp_fit_predict

        train_rows = np.flatnonzero(train)
        z_mean, z_sd = M._standardise(np.asarray(z, dtype=np.float64), train_rows)
        v_mean, v_sd = M._standardise(np.asarray(v, dtype=np.float64), train_rows)
        z_in, v_in = (z - z_mean) / z_sd, (v - v_mean) / v_sd
        vbar_in = (vbar_t - v_mean) / v_sd

        def fit(d_train, v_train, d_test):
            design = np.vstack([d_train, d_test])
            rows = np.arange(len(d_train))
            test_rows = np.arange(len(d_train), len(design))
            return np.hstack([mlp_fit_predict(design, v_train[:, block], rows,
                                              test_rows, seed)
                              for block in (slice(0, n_comp),
                                            slice(n_comp, None))])
    predictions: list = []

    def recording(a, b, c):
        out = fit(a, b, c)
        predictions.append(np.asarray(out, dtype=np.float64))
        return out

    cols = M._probe_columns(z_in, t, v_in, vbar_in, train, test, seed, n_perm,
                            recording)
    rng = np.random.default_rng(seed)             # the stream's head
    M._permute_within_type(z_in, t, rng)
    M._subsample_rows(np.flatnonzero(train), rng)
    test_rows = M._subsample_rows(np.flatnonzero(test), rng)
    v_test = np.asarray(v_in)[test_rows]          # the metric's dtype
    out = {"test_rows": test_rows, "n_comp": int(n_comp),
           "graded": cols["graded"],
           "baseline": ((v_test - np.asarray(vbar_in)[t[test_rows]]) ** 2
                        ).astype(np.float64),
           "probe": (v_test - predictions[0]) ** 2,
           "floors": np.stack([(v_test - p) ** 2 for p in predictions[1:]])}
    # the recovered cells are the metric's: their means are its MSEs (on the
    # graded columns; a constant column's MSE is rounding noise)
    g = cols["graded"]
    assert np.allclose(out["probe"].mean(axis=0)[g], cols["probe"][g],
                       rtol=1e-5), "probe replay drifted from _probe_columns"
    # (the metric averages the baseline's float32 squares in float32)
    assert np.allclose(out["baseline"].mean(axis=0)[g], cols["baseline"][g],
                       rtol=1e-4), "baseline replay drifted from _probe_columns"
    return out


def probe_statistic(cells: dict, reference: Mapping[str, float] | None = None):
    """``(cells, weights) -> {comp_excess, img_excess, [comp_frac, img_frac]}``
    for one grader; *reference* maps a block to the uncontrolled excess."""
    n_cols = cells["baseline"].shape[1]
    blocks = {"comp": np.arange(cells["n_comp"]),
              "img": np.arange(cells["n_comp"], n_cols)}
    blocks = {k: sel[cells["graded"][sel]] for k, sel in blocks.items()}

    def stat(c, w):
        total = w.sum()
        if total <= 0:
            return {f"{b}_{k}": float("nan") for b in blocks
                    for k in ("excess", "frac")}
        mse = lambda sq: np.tensordot(w, sq, axes=(0, 0)) / total
        base, probe = mse(c["baseline"]), mse(c["probe"])
        floors = [mse(f) for f in c["floors"]]
        out = {}
        for name, sel in blocks.items():
            gain = lambda m: float(np.mean(0.5 * np.log(
                (base[sel] + _EPS) / (m[sel] + _EPS))))
            excess = gain(probe) - float(np.mean([gain(f) for f in floors]))
            out[f"{name}_excess"] = excess
            ref = (reference or {}).get(name)
            if ref is not None and ref > 0:
                out[f"{name}_frac"] = excess / ref
        return out
    return stat


def cycle_cells(latent, t, scores, types, train, test, seed: int = 0) -> dict:
    """Held-out residuals and targets of ``metrics.cycle_r2`` (same stream)."""
    from discell.model import metrics as M

    rng = np.random.default_rng(seed)
    keep = np.isin(t, types)
    rows_train = np.flatnonzero(train & keep)
    rows_test = np.flatnonzero(test & keep)
    if len(rows_train) < 200 or len(rows_test) < 200:
        return {"rows_test": rows_test[:0], "residual": np.zeros((0, 2)),
                "target": np.zeros((0, 2))}
    rows_train = M._subsample_rows(rows_train, rng)
    rows_test = M._subsample_rows(rows_test, rng)
    latent = latent.copy().astype(np.float64)
    target = scores.copy().astype(np.float64)
    for g in types:
        members = np.flatnonzero(t == g)
        latent[members] -= latent[members].mean(axis=0)
        target[members] -= target[members].mean(axis=0)
    design = np.hstack([latent[rows_train], np.ones((len(rows_train), 1))])
    gram = design.T @ design + 1e-3 * np.eye(design.shape[1])
    coef = np.linalg.solve(gram, design.T @ target[rows_train])
    held = np.hstack([latent[rows_test], np.ones((len(rows_test), 1))])
    return {"rows_test": rows_test, "residual": target[rows_test] - held @ coef,
            "target": target[rows_test]}


def cycle_statistic(cells: dict, weights: np.ndarray) -> float:
    """Pooled R^2 = 1 - var(residual) / var(target), numpy's ``.var()`` over
    all entries of the two columns, weighted per cell."""
    if weights.sum() <= 0 or not len(cells["target"]):
        return float("nan")
    w = np.repeat(weights, cells["target"].shape[1])

    def var(a):
        a = a.ravel()
        mean = np.dot(w, a) / w.sum()
        return np.dot(w, (a - mean) ** 2) / w.sum()
    return float(1.0 - var(cells["residual"]) / max(var(cells["target"]), 1e-12))


def nmi_cells(z, t, seed: int = 0) -> dict:
    """The cells and k-means labels behind ``metrics.z_type_nmi``."""
    from sklearn.cluster import KMeans

    from discell.model import metrics as M

    rng = np.random.default_rng(seed)
    rows = M._subsample(len(z), M.MAX_EVAL_CELLS, rng)
    k = int(t.max()) + 1
    labels = KMeans(k, n_init=4, random_state=seed).fit_predict(z[rows])
    return {"rows": rows, "t": np.asarray(t)[rows], "cluster": labels}


def nmi_statistic(cells: dict, weights: np.ndarray) -> float:
    """Arithmetic-mean-normalised MI (sklearn's default) of a weighted
    contingency table of type against cluster."""
    t, c = cells["t"], cells["cluster"]
    table = np.zeros((int(t.max()) + 1, int(c.max()) + 1))
    np.add.at(table, (t, c), weights)
    total = table.sum()
    if total <= 0:
        return float("nan")
    p = table / total
    pt, pc = p.sum(axis=1), p.sum(axis=0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / np.outer(pt, pc)[nz])).sum())
    h = lambda q: float(-(q[q > 0] * np.log(q[q > 0])).sum())
    ht, hc = h(pt), h(pc)
    if ht == 0 and hc == 0:
        return 1.0
    denom = 0.5 * (ht + hc)
    return float(mi / denom) if denom > 0 else 0.0


def ross_terms(x: np.ndarray, y: np.ndarray, k: int | None = None,
               rng=None) -> np.ndarray:
    """Per-point terms of ``external_criteria.mi_discrete_continuous``
    (NaN where a point's class is too small): their mean is the estimator
    before its clip at zero. Consumes *rng* exactly as the estimator does."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    from discell.experiments.external_criteria import MI_K

    k = MI_K if k is None else k
    x = np.asarray(x, float).reshape(len(x), -1)
    rng = np.random.default_rng(0) if rng is None else rng
    x = x + 1e-10 * rng.standard_normal(x.shape) * (x.std(axis=0) + 1e-12)
    y = np.asarray(y)
    n = len(x)
    full = cKDTree(x)
    radius, n_class, k_eff = np.zeros(n), np.zeros(n), np.zeros(n)
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        n_class[rows] = len(rows)
        kk = min(k, len(rows) - 1)
        if kk < 1:
            radius[rows], k_eff[rows] = np.inf, np.nan
            continue
        d = cKDTree(x[rows]).query(x[rows], k=kk + 1, p=np.inf)[0][:, kk]
        radius[rows], k_eff[rows] = d, kk
    good = np.isfinite(radius) & np.isfinite(k_eff)
    m = np.asarray(full.query_ball_point(x[good], r=radius[good] * (1 - 1e-12),
                                         p=np.inf, return_length=True))
    terms = np.full(n, np.nan)
    terms[good] = (digamma(n) - digamma(n_class[good]) + digamma(k_eff[good])
                   - digamma(np.maximum(m, 1)))
    return terms


def w_mi_cells(w, niche, t, seed: int = 0, perms: int | None = None,
               max_cells: int | None = None) -> dict:
    """Per-cell Ross terms behind ``degeneracy.w_channel_guard``'s MI excess
    (same stream: subsample, real MI, then each permutation's floor)."""
    from discell.experiments.external_criteria import within_type_permutation
    from discell.model.degeneracy import W_GUARD_MAX_CELLS, W_GUARD_PERMS

    perms = W_GUARD_PERMS if perms is None else perms
    max_cells = W_GUARD_MAX_CELLS if max_cells is None else max_cells
    w = np.asarray(w, dtype=np.float64)
    niche, t = np.asarray(niche), np.asarray(t)
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(niche >= 0)
    if len(rows) > max_cells:
        rows = np.sort(rng.choice(rows, max_cells, replace=False))
    y, t_rows, w_rows = niche[rows], t[rows], w[rows]
    real = ross_terms(w_rows, y, rng=rng)
    floors = np.stack([ross_terms(w_rows, within_type_permutation(y, t_rows, rng),
                                  rng=rng) for _ in range(perms)])
    return {"rows": rows, "real": real, "floors": floors}


def w_mi_statistic(cells: dict, weights: np.ndarray) -> float:
    """``max(MI, 0) - mean_p max(MI_p, 0)``, each MI a weighted mean of its
    per-cell terms over the cells where the term is defined."""
    def mi(terms):
        ok = np.isfinite(terms)
        total = weights[ok].sum()
        return max(float(np.dot(weights[ok], terms[ok]) / total), 0.0) \
            if total > 0 else float("nan")
    return mi(cells["real"]) - float(np.mean([mi(f) for f in cells["floors"]]))


# --------------------------------------------------------------------------
# transport: the per-panel artefacts need the model, so these take a trainer
# --------------------------------------------------------------------------


def transport_mean_panels(trainer, data, config, b_matrix, labels, fold,
                          device: str) -> list[dict]:
    """The mean read's panels, replayed (``transport.transport_check``'s
    loop and random stream, minus figures and the Phi-fixed channels): per
    panel the counterfactual prediction on the kept genes, the held-out rows
    of both niches, and the point R^2 and noise ceiling."""
    from discell.model import transport as T

    connected = data.graph.degrees > 0
    rng = np.random.default_rng(config.seed)
    held_out = fold == 0
    n_types = len(data.p_t)
    n_niches = int(labels.max()) + 1
    n_groups = n_niches * n_types
    model_rows = connected & ~held_out & (labels >= 0)
    group = np.full(data.graph.n_cells, -1, dtype=np.int64)
    group[model_rows] = labels[model_rows] * n_types + data.t[model_rows]
    channels = T.collect_channels(trainer, group, n_groups)
    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    names = [str(n) for n in data.type_names]
    panels = []
    for niche_a, niche_b in T.pick_pairs(labels, data.graph.y, connected):
        for g in range(len(names)):
            members, ok = {}, True
            for niche, side in ((niche_a, "A"), (niche_b, "B")):
                train_rows = np.flatnonzero(connected & ~held_out
                                            & (data.t == g) & (labels == niche))
                test_rows = np.flatnonzero(connected & held_out
                                           & (data.t == g) & (labels == niche))
                if (len(train_rows) < T.MIN_CELLS
                        or len(test_rows) < T.MIN_CELLS // 5):
                    ok = False
                    break
                members[side] = (train_rows, test_rows)
            if not ok:
                continue
            y_a = data.graph.y[members["A"][0]]
            y_b = data.graph.y[members["B"][0]]
            gap_vec = y_b.mean(0) - y_a.mean(0)
            gap = np.linalg.norm(gap_vec)
            unit = gap_vec / max(gap, 1e-12)
            spread = 0.5 * (float((y_a @ unit).std()) + float((y_b @ unit).std()))
            gid_a, gid_b = niche_a * n_types + g, niche_b * n_types + g
            program = b_matrix @ (channels["prior_w"][gid_b]
                                  - channels["prior_w"][gid_a])
            k_a = T.group_kappa(channels, gid_a, config.kappa)
            k_b = T.group_kappa(channels, gid_b, config.kappa)
            e_a = T.group_eta(channels, gid_a)
            leak_only = (np.log(T.leak_rate(channels["rho"][gid_a],
                                            channels["rho_bar"][gid_b], k_b, e_a)
                                + T.EPS)
                         - np.log(T.leak_rate(channels["rho"][gid_a],
                                              channels["rho_bar"][gid_a], k_a, e_a)
                                  + T.EPS))
            obs_a = np.asarray(x_rate[members["A"][1]].mean(axis=0)).ravel()
            obs_b = np.asarray(x_rate[members["B"][1]].mean(axis=0)).ravel()
            keep = (obs_a > T.MIN_RATE) & (obs_b > T.MIN_RATE)
            observed = np.log(obs_b[keep] + T.EPS) - np.log(obs_a[keep] + T.EPS)
            prediction = (program + leak_only)[keep]
            ceiling = T.noise_ceiling(x_rate, members["A"][1], members["B"][1],
                                      keep, rng)
            panels.append({
                "pair": (int(niche_a), int(niche_b)), "type": names[g],
                "rows_a": members["A"][1], "rows_b": members["B"][1],
                "keep": keep, "prediction": prediction,
                "overlap_flag": bool(gap > 3 * spread),
                "r2": T.score_shift(prediction, observed)["r2"],
                "noise_ceiling": ceiling,
                "trusted": bool(ceiling >= T.TRUST_CEILING
                                and int(keep.sum()) >= T.TRUST_GENES)})
    return panels


def _split_weights(mult: np.ndarray, rng, n_splits: int) -> np.ndarray:
    """``(n_splits, 2, n)`` half weights: the resampled multiset of cells
    (cell i repeated mult[i] times) permuted and cut in half, as
    ``noise_ceiling`` cuts the real sample."""
    idx = np.repeat(np.arange(len(mult)), mult.astype(np.int64))
    out = np.zeros((n_splits, 2, len(mult)))
    for s in range(n_splits):
        order = rng.permutation(len(idx))
        half = len(idx) // 2
        out[s, 0] = np.bincount(idx[order[:half]], minlength=len(mult))
        out[s, 1] = np.bincount(idx[order[half:]], minlength=len(mult))
    return out


def transport_mean_bootstrap(panels: list[dict], x_rate, positions: np.ndarray,
                             n: int = N_BOOT, tile_um: float = TILE_UM,
                             seed: int = 0, device: str = "cpu",
                             alpha: float = ALPHA, chunk: int = 100) -> dict:
    """Tile CIs of the extrapolation tiers' fraction of ceiling (mean R^2 over
    the tier's panels / their mean noise ceiling, as ``tier_summary``).

    One tile draw per replicate over every held-out cell any panel uses; per
    panel the observed shift and its split-half ceiling (``N_SPLITS`` halves
    of the resampled cells, ``MIN_RATE`` floor) are recomputed on the GPU.
    """
    import torch

    from discell.model import transport as T

    cells = np.unique(np.concatenate([np.concatenate([p["rows_a"], p["rows_b"]])
                                      for p in panels]))
    pos = np.full(len(positions), -1, dtype=np.int64)
    pos[cells] = np.arange(len(cells))
    tiles, n_tiles = tile_index(positions[cells], tile_um)
    rng = np.random.default_rng(seed)
    mults = np.stack([np.bincount(rng.integers(0, n_tiles, n_tiles),
                                  minlength=n_tiles)[tiles] for _ in range(n)])
    split_rng = np.random.default_rng(seed + 1)
    r2 = np.full((n, len(panels)), np.nan)
    ceil = np.full((n, len(panels)), np.nan)
    dev = torch.device(device)
    ns = T.N_SPLITS
    for j, p in enumerate(panels):
        pred = torch.as_tensor(p["prediction"], dtype=torch.float32, device=dev)
        pc = pred - pred.mean()
        genes = np.flatnonzero(p["keep"])
        xs = [torch.as_tensor(np.asarray(x_rate[rows][:, genes].todense()),
                              dtype=torch.float32, device=dev)
              for rows in (p["rows_a"], p["rows_b"])]
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            sides = []
            for x, rows in zip(xs, (p["rows_a"], p["rows_b"])):
                m = mults[lo:hi, pos[rows]].astype(np.float64)   # (b, cells)
                halves = np.stack([_split_weights(m[b], split_rng, ns)
                                   for b in range(hi - lo)])     # (b, s, 2, c)
                wts = torch.as_tensor(np.concatenate(
                    [m[:, None, :], halves.reshape(hi - lo, 2 * ns, -1)], axis=1),
                    dtype=torch.float32, device=dev)             # (b, 1+2s, c)
                sums = torch.einsum("nkc,cg->nkg", wts, x)
                sides.append(sums / wts.sum(dim=2, keepdim=True).clamp(min=1e-12))
            a, b = sides
            observed = torch.log(b[:, 0] + T.EPS) - torch.log(a[:, 0] + T.EPS)
            oc = observed - observed.mean(dim=1, keepdim=True)
            r2[lo:hi, j] = (1.0 - ((oc - pc) ** 2).sum(dim=1)
                            / (oc ** 2).sum(dim=1).clamp(min=1e-12)).cpu().numpy()
            # column 1 + 2s + side is half `side` of split s
            shift = (torch.log(b[:, 1:].clamp(min=T.MIN_RATE))
                     - torch.log(a[:, 1:].clamp(min=T.MIN_RATE)))
            h0, h1 = shift[:, 0::2], shift[:, 1::2]
            h0 = h0 - h0.mean(dim=2, keepdim=True)
            h1 = h1 - h1.mean(dim=2, keepdim=True)
            corr = ((h0 * h1).sum(dim=2) / (h0.norm(dim=2) * h1.norm(dim=2))
                    .clamp(min=1e-12)).mean(dim=1)
            r = corr.cpu().numpy()
            ceil[lo:hi, j] = np.where(r > 0, 2 * r / (1 + r), 0.0)
        empty = (mults[:, pos[p["rows_a"]]].sum(axis=1) == 0) | \
                (mults[:, pos[p["rows_b"]]].sum(axis=1) == 0)
        r2[empty, j] = ceil[empty, j] = np.nan
        del xs
    point_r2 = np.array([p["r2"] for p in panels])
    point_ceil = np.array([p["noise_ceiling"] for p in panels])
    tiers = {"transport_of_ceiling": [p["overlap_flag"] for p in panels],
             "transport_of_ceiling_trusted": [p["overlap_flag"] and p["trusted"]
                                              for p in panels]}
    out = {}
    for key, sel in tiers.items():
        sel = np.asarray(sel, dtype=bool)
        if not sel.any():
            continue

        def frac(r, c):
            mc = np.nanmean(c)
            return float(np.nanmean(r) / mc) if mc >= 0.05 else float("nan")
        point = frac(point_r2[sel], point_ceil[sel])
        draws = [frac(r2[b, sel], ceil[b, sel]) for b in range(n)]
        out[key] = _summarise({"v": point}, {"v": draws}, alpha)["v"]
        out[key]["n_panels"] = int(sel.sum())
    return {"reads": out, "n_boot": int(n), "n_tiles": n_tiles,
            "n_cells": int(len(cells)), "tile_um": float(tile_um)}


def _weighted_mmd_parts(kxx, row_kxy, weights):
    """``sxx - 2 sxy`` of the unbiased MMD^2 with the source side weighted
    (the target term is fixed). ``sxx`` is the U-statistic over pairs of
    DISTINCT cells, each pair weighted ``w_i w_j``:
    ``(w'Kw - sum w_i^2 K_ii) / (M^2 - sum w_i^2)`` -- a cell drawn twice by
    the bootstrap is not paired with its own copy, which would add ``K_ii =
    1`` terms and bias every resampled MMD^2 upwards. ``sxy = w' r / M`` with
    *row_kxy* the per-source-cell mean kernel to the target. *weights* is
    ``(n_draws, m)``; at unit weights this is ``_mmd2``'s."""
    import torch

    wm = weights.sum(dim=1)
    w2 = (weights ** 2).sum(dim=1)
    quad = ((weights @ kxx) * weights).sum(dim=1)
    diag = (weights ** 2) @ kxx.diagonal()
    sxx = (quad - diag) / (wm ** 2 - w2).clamp(min=1e-12)
    sxy = (weights @ row_kxy) / wm.clamp(min=1e-12)
    part = sxx - 2.0 * sxy
    ok = (wm ** 2 - w2) > 0
    return torch.where(ok, part, torch.full_like(part, float("nan")))


def _gap(u, t, floor):
    """``gap closed`` = clip((U - T) / (U - floor), -1, 1), NaN on a null
    denominator (``distribution_scores``)."""
    u, t = np.asarray(u, dtype=np.float64), np.asarray(t, dtype=np.float64)
    denom = u - floor
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((u - t) / np.where(np.abs(denom) > 1e-12, denom, np.nan),
                       -1.0, 1.0)


def _mmd_draws(p_trans, p_unt, p_target, rows, rng, device: str,
               weights_of: Callable, chunk: int = 250) -> dict:
    """``distribution_scores``' subsample, bandwidth, floor and MMD^2 values
    (drawn in its order from its fresh generator *rng*), plus the transported
    and untransported MMD^2 under every tile draw of the scored source cells
    (``weights_of(cells) -> (n_draws, m)``). Kernels never leave this call."""
    import torch

    from discell.model import transport as T

    n_s, n_t = len(p_trans), len(p_target)
    m = int(min(n_s, n_t, T.SIZE_CAP))
    if m < 20:
        return {"insufficient": True}
    idx_s = rng.choice(n_s, m, replace=False)
    idx_t = rng.choice(n_t, m, replace=False)
    to_t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
    target = np.asarray(p_target)[idx_t]
    y = to_t(T._hellinger(target))
    xt = to_t(T._hellinger(np.asarray(p_trans)[idx_s]))
    xu = to_t(T._hellinger(np.asarray(p_unt)[idx_s]))
    xm = to_t(T._hellinger(np.tile(target.mean(axis=0), (m, 1))))
    cells = np.asarray(rows)[idx_s]
    weights = torch.as_tensor(weights_of(cells), dtype=torch.float64,
                              device=y.device)
    out = {"insufficient": False, "m": m, "cells": cells, "mmd2": {},
           "draws": {}}
    with torch.no_grad():
        d_yy = torch.cdist(y, y)
        off = d_yy[~torch.eye(m, dtype=torch.bool, device=d_yy.device)]
        sigma = float(off.median())
        gamma = 1.0 / (2.0 * max(sigma, 1e-8) ** 2)
        kyy = torch.exp(-gamma * d_yy.pow(2))
        syy = float((kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)))
        for name, x in (("transported", xt), ("untransported", xu),
                        ("type_mean", xm)):
            kxx = T._kernel_parts(x, x, gamma)
            kxy = T._kernel_parts(x, y, gamma)
            out["mmd2"][name] = T._mmd2(kxx, kyy, kxy)
            if name != "type_mean":
                kd, rd = kxx.double(), kxy.double().mean(dim=1)
                out["draws"][name] = torch.cat(
                    [_weighted_mmd_parts(kd, rd, weights[i:i + chunk])
                     for i in range(0, len(weights), chunk)]).cpu().numpy() + syy
                del kd, rd
            del kxx, kxy
        half = m // 2
        perm = torch.as_tensor(rng.permutation(m), device=y.device)
        y1, y2 = y[perm[:half]], y[perm[half:2 * half]]
        out["mmd2"]["floor"] = T._mmd2(T._kernel_parts(y1, y1, gamma),
                                       T._kernel_parts(y2, y2, gamma),
                                       T._kernel_parts(y1, y2, gamma))
    return out


def transport_dist_panels(trainer, data, config, labels, fold, device: str,
                          boot_replayed: int, n: int = N_BOOT,
                          tile_um: float = TILE_UM, seed: int = 0) -> dict:
    """Read A's pairwise panels, replayed, with their tile draws.

    ``distribution_check``'s shared random stream is advanced exactly as the
    run did (its ``scores`` read with *boot_replayed* paired draws), so each
    panel's source and target rows are the run's; the model-vs-model scores
    (``group``: target cells at the niche-group w, ``scores_model``; ``own``:
    at their own posterior, ``scores_model_own``) are recomputed from their
    own fresh generators, which pins subsample, bandwidth and floor. The tile
    universe is every source row any panel draws from; one draw per replicate
    is shared by all panels.
    """
    from discell.model import transport as T
    from discell.model.validate import collect_latents

    mu_z = collect_latents(trainer, data)["mu_z"]
    connected = data.graph.degrees > 0
    rng = np.random.default_rng(config.seed)
    held_out = fold == 0
    n_types = len(data.p_t)
    n_niches = int(labels.max()) + 1
    n_groups = n_niches * n_types
    model_rows = connected & ~held_out & (labels >= 0)
    group = np.full(data.graph.n_cells, -1, dtype=np.int64)
    group[model_rows] = labels[model_rows] * n_types + data.t[model_rows]
    channels = T.collect_channels(trainer, group, n_groups)
    train_rows, test_rows = {}, {}
    for k in range(n_niches):
        for g in range(n_types):
            tr = np.flatnonzero(connected & ~held_out & (data.t == g) & (labels == k))
            te = np.flatnonzero(connected & held_out & (data.t == g) & (labels == k))
            if len(tr) >= T.MIN_CELLS and len(te) >= T.MIN_CELLS // 5:
                train_rows[(k, g)], test_rows[(k, g)] = tr, te
    w_of = {(k, g): (T._niche_w(trainer, channels["c"][k * n_types + g], g, n_types),
                     channels["rho_bar"][k * n_types + g],
                     T.group_kappa(channels, k * n_types + g, config.kappa))
            for (k, g) in train_rows}
    names = [str(x) for x in data.type_names]

    def advance_scores(n_s, n_t):
        """The shared-stream draws of the ``scores`` read (no count depths):
        two subsamples, the floor permutation, the paired draws."""
        m = int(min(n_s, n_t, T.SIZE_CAP))
        if m < 20:
            return
        rng.choice(n_s, m, replace=False)
        rng.choice(n_t, m, replace=False)
        rng.permutation(m)
        for _ in range(boot_replayed):
            rng.integers(0, m, m)
            rng.integers(0, m, m)

    raw = []
    for niche_a, niche_b in T.pick_pairs(labels, data.graph.y, connected):
        for g in range(n_types):
            if (niche_a, g) not in train_rows or (niche_b, g) not in train_rows:
                continue
            src = test_rows[(niche_a, g)]
            rows = src[rng.permutation(len(src))[:T.SIZE_CAP]]
            tgt = test_rows[(niche_b, g)]
            tgt = tgt[rng.permutation(len(tgt))[:T.SIZE_CAP]]
            advance_scores(len(rows), len(tgt))
            raw.append({"pair": [int(niche_a), int(niche_b)], "type": names[g],
                        "g": g, "target": (niche_b, g), "rows": rows,
                        "own": np.full(len(rows), niche_a), "tgt": tgt})
    if not raw:
        return {"panels": [], "n_tiles": 0, "n_cells": 0}
    universe = np.unique(np.concatenate([r["rows"] for r in raw]))
    pos = np.full(data.graph.n_cells, -1, dtype=np.int64)
    pos[universe] = np.arange(len(universe))
    tiles, n_tiles = tile_index(np.asarray(data.positions)[universe], tile_um)
    draw_rng = np.random.default_rng(seed)
    mults = np.stack([np.bincount(draw_rng.integers(0, n_tiles, n_tiles),
                                  minlength=n_tiles)[tiles] for _ in range(n)])
    weights_of = lambda cells: mults[:, pos[cells]]

    wanted = np.unique(np.concatenate([r["tgt"] for r in raw]))
    own_p = T.collect_own_p(trainer, wanted, data.graph.n_cells)
    where = np.full(data.graph.n_cells, -1, dtype=np.int64)
    where[wanted] = np.arange(len(wanted))
    panels = []
    for r in raw:
        rows, tgt, g = r["rows"], r["tgt"], r["g"]
        w_a, bar_a, k_a = w_of[r["target"]]
        eta_rows = T._cell_eta(trainer, rows)
        p_trans = T._decode_cells(trainer, mu_z[rows], np.tile(w_a, (len(rows), 1)),
                                  np.tile(bar_a, (len(rows), 1)), k_a, eta=eta_rows)
        p_unt = T._decode_cells(
            trainer, mu_z[rows], np.stack([w_of[(int(k), g)][0] for k in r["own"]]),
            np.stack([w_of[(int(k), g)][1] for k in r["own"]]),
            T._stack_kappa([w_of[(int(k), g)][2] for k in r["own"]]), eta=eta_rows)
        p_tgt_group = T._decode_cells(
            trainer, mu_z[tgt], np.tile(w_a, (len(tgt), 1)),
            np.tile(bar_a, (len(tgt), 1)), k_a, eta=T._cell_eta(trainer, tgt))
        p_tgt_own = own_p[where[tgt]].astype(np.float64)
        entry = {"pair": r["pair"], "type": r["type"]}
        for name, p_tgt, s in (("group", p_tgt_group, config.seed + 1),
                               ("own", p_tgt_own, config.seed + 101)):
            entry[name] = _mmd_draws(p_trans, p_unt, p_tgt, rows,
                                     np.random.default_rng(s), device, weights_of)
        panels.append(entry)
    return {"panels": panels, "n_tiles": n_tiles, "n_cells": int(len(universe))}


def transport_dist_summary(panels: list[dict], alpha: float = ALPHA) -> dict:
    """Read A's pairwise medians over panels, point and per draw: gap closed
    with the group-w and own-w targets, and the type-mean predictor's gap
    with the own-w target (its MMD^2 depends on the target only, so only the
    untransported side moves)."""
    keys = {"readA_gap_group": ("group", "transported"),
            "readA_gap_own": ("own", "transported"),
            "readA_type_mean_own": ("own", "type_mean")}
    out = {}
    for key, (target, pred) in keys.items():
        point, draws = [], []
        for e in panels:
            s = e[target]
            if s["insufficient"]:
                continue
            u = s["draws"]["untransported"]
            t = (s["draws"]["transported"] if pred == "transported"
                 else np.full(len(u), s["mmd2"]["type_mean"]))
            point.append(float(_gap(s["mmd2"]["untransported"], s["mmd2"][pred],
                                    s["mmd2"]["floor"])))
            draws.append(_gap(u, t, s["mmd2"]["floor"]))
        if not point:
            continue
        per_draw = np.nanmedian(np.stack(draws, axis=1), axis=1)
        out[key] = _summarise({"v": float(np.nanmedian(point))},
                              {"v": per_draw}, alpha)["v"]
        out[key]["n_panels"] = len(point)
    return out


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def _check(stored, estimate, tol: float = 1e-4) -> dict:
    """Is the unit-weight statistic the number on disk?"""
    if stored is None or estimate is None:
        return {"stored": stored, "reproduces": None}
    if not np.isfinite(float(stored)) or not np.isfinite(float(estimate)):
        return {"stored": float(stored), "reproduces": bool(
            np.isfinite(float(stored)) == np.isfinite(float(estimate)))}
    diff = abs(float(stored) - float(estimate))
    return {"stored": float(stored), "abs_diff": diff,
            "reproduces": bool(diff <= tol * max(1.0, abs(float(stored))))}


def run_bootstrap(dataset: str, run: str, reads: Sequence[str] = READS,
                  n: int = N_BOOT, tile_um: float = TILE_UM, seed: int = 0,
                  device: str = "cuda", boot_replayed: int = 200) -> dict:
    """Every requested read of one run, with its tile CI and its check
    against the stored number; writes ``runs/<run>/bootstrap_ci.json``
    (merging into an existing file, so reads can be added one at a time)."""
    import torch

    from discell.model import metrics as M
    from discell.model.validate import load_run, niche_labels

    config, data, trainer, run_dir, b_matrix = load_run(dataset, run, device)
    dev = str(next(trainer.model.parameters()).device)
    metrics = _load(run_dir / "metrics.json") or {}
    final = metrics.get("final") or {}
    target = run_dir / "bootstrap_ci.json"
    record = _load(target) or {"run": run, "dataset": dataset, "reads": {}}
    record.update({"n_boot": n, "tile_um": tile_um, "seed": seed,
                   "conditional_on": "the fitted probes, clustering, cycle "
                   "ridge, kNN balls and model predictions (held-out cells "
                   "resampled by tile)"})
    positions = np.asarray(data.positions, dtype=np.float64)

    def put(key, boot, stored, extra=None, tol=1e-4):
        entry = {k: boot[k] for k in ("estimate", "ci95", "se", "n_finite")}
        entry.update({k: boot[k] for k in ("n_boot", "n_tiles", "n_cells")
                      if k in boot})
        entry.update(_check(stored, entry["estimate"], tol))
        entry["check_tol"] = tol
        entry.update(extra or {})
        record["reads"][key] = entry
        log.info("%s %s: %.4f [%.4f, %.4f]%s", run, key, entry["estimate"],
                 *entry["ci95"], "" if entry.get("reproduces") in (True, None)
                 else f"  (stored {stored} -- NOT reproduced)")

    train = trainer._sweep(trainer.train_batches)
    val = trainer._sweep(trainer.val_batches)
    rows_all = np.concatenate([train["nodes"], val["nodes"]])
    z_all = np.vstack([train["mu_z"], val["mu_z"]])
    w_all = np.vstack([train["mu_w"], val["mu_w"]])
    t_all = data.t[rows_all]
    train_mask = np.arange(len(rows_all)) < len(train["nodes"])

    if "nmi" in reads:
        t0 = time.time()
        cells = nmi_cells(z_all, t_all, config.seed)
        boot = tile_bootstrap(positions[rows_all[cells["rows"]]], cells,
                              nmi_statistic, n, tile_um, seed)
        put("nmi", boot, metrics.get("best", {}).get("nmi"))
        log.info("nmi done (%.0f s)", time.time() - t0)

    if "cycle" in reads and data.cycle is not None:
        cyc = data.cycle
        type_names = [str(x) for x in data.type_names]
        cycling = np.array([g for g in cyc["cycling_types"]
                            if "nassigned" not in type_names[g]][:4])
        scores = np.stack([cyc["s_score"], cyc["g2m_score"]], axis=1)[rows_all]
        for name, latent in (("z", z_all), ("w", w_all)):
            cells = cycle_cells(latent, t_all, scores, cycling, train_mask,
                                ~train_mask, config.seed)
            boot = tile_bootstrap(positions[rows_all[cells["rows_test"]]], cells,
                                  cycle_statistic, n, tile_um, seed)
            stored = ((final.get("cycle") or {}).get(name) or {}).get("r2_pooled")
            put(f"cycle_{name}", boot, stored)

    if "w_mi" in reads:
        from discell.model.degeneracy import W_GUARD_NICHES

        niche = niche_labels(data, W_GUARD_NICHES, config.seed)
        vrows = val["nodes"]
        cells = w_mi_cells(val["mu_w"], niche[vrows], data.t[vrows], config.seed)
        boot = tile_bootstrap(positions[vrows[cells["rows"]]], cells,
                              w_mi_statistic, n, tile_um, seed)
        stored = ((_load(run_dir / "degeneracy.json") or {}).get("w_channel")
                  or {}).get("w_niche_mi_excess")
        put("w_niche_mi_excess", boot, stored)

    if "probe" in reads:
        blocks = _load(run_dir / "validation" / "probe_blocks.json")
        if blocks is None:
            log.warning("%s: no validation/probe_blocks.json -- probe skipped", run)
        else:
            mu_node = np.empty((data.graph.n_cells, z_all.shape[1]), z_all.dtype)
            mu_node[rows_all] = z_all
            rows = np.concatenate(data.train_tiles + data.val_tiles)
            n_train = sum(len(tile) for tile in data.train_tiles)
            tr = np.arange(len(rows)) < n_train
            for family in ("ridge", "mlp"):
                fam = blocks[family]
                cells = probe_cells(mu_node[rows], data.t[rows], data.v_block[rows],
                                    data.vbar_t, tr, ~tr, n_comp=data.n_comp,
                                    seed=int(fam["seed"]), n_perm=int(fam["n_perm"]),
                                    family=family)
                reference = {b: fam[b].get("uncontrolled_excess")
                             for b in ("comp", "img")}
                boot = tile_bootstrap(positions[rows[cells["test_rows"]]], cells,
                                      probe_statistic(cells, reference), n,
                                      tile_um, seed)
                for key, entry in boot["reads"].items():
                    block, what = key.split("_")
                    stored = (fam[block]["excess"] if what == "excess"
                              else fam[block].get("fraction_of_uncontrolled"))
                    # the MLP is refitted: 300 Adam steps on CPU threads
                    # reproduce the stored fit to ~1e-4 nats, not bit for bit
                    put(f"probe_{family}_{block}_{what}",
                        {**entry, "n_boot": n, "n_tiles": boot["n_tiles"],
                         "n_cells": boot["n_cells"]}, stored,
                        {"reference_excess": reference[block]} if what == "frac"
                        else None, tol=1e-4 if family == "ridge" else 5e-3)

    labels = fold = None
    if "transport_mean" in reads or "transport_dist" in reads:
        from discell.model.validate import collect_latents

        labels = niche_labels(data, 10, config.seed)
        fold = collect_latents(trainer, data)["fold"]

    if "transport_mean" in reads:
        stored = _load(run_dir / "transport" / "transport.json")
        t0 = time.time()
        panels = transport_mean_panels(trainer, data, config, b_matrix, labels,
                                       fold, dev)
        replay = {"n_panels": len(panels)}
        if stored:
            by = {(tuple(p["pair"]), p["type"]): p for p in stored["panels"]}
            diffs = [max(abs(p["r2"] - by[(p["pair"], p["type"])]
                             ["counterfactual"]["r2"]),
                         abs(p["noise_ceiling"] - by[(p["pair"], p["type"])]
                             ["noise_ceiling"]))
                     for p in panels if (p["pair"], p["type"]) in by]
            replay.update(n_stored=len(stored["panels"]),
                          max_abs_diff=float(max(diffs)) if diffs else None)
        x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
        boot = transport_mean_bootstrap(panels, x_rate, positions, n, tile_um,
                                        seed, dev)
        for key, entry in boot["reads"].items():
            tier = ("extrapolation_trusted" if key.endswith("trusted")
                    else "extrapolation")
            value = (((stored or {}).get("summary") or {}).get(tier)
                     or {}).get("counterfactual_of_ceiling")
            put(key, {**entry, "n_boot": n, "n_tiles": boot["n_tiles"],
                      "n_cells": boot["n_cells"]}, value,
                {"n_panels": entry["n_panels"], "panel_replay": replay})
        log.info("transport mean read done (%.0f s)", time.time() - t0)
        torch.cuda.empty_cache()

    if "transport_dist" in reads:
        stored = _load(run_dir / "transport" / "transport_distribution.json")
        t0 = time.time()
        result = transport_dist_panels(trainer, data, config, labels, fold, dev,
                                       boot_replayed, n, tile_um, seed)
        panels = result["panels"]
        replay = {"n_panels": len(panels)}
        if stored:
            by = {(tuple(p["pair"]), p["type"]): p for p in stored["pairwise"]}
            diffs = []
            for e in panels:
                s = by.get((tuple(e["pair"]), e["type"]))
                if s is None:
                    continue
                for name, key in (("group", "scores_model"),
                                  ("own", "scores_model_own")):
                    if e[name]["insufficient"] or key not in s:
                        continue
                    diffs += [abs(e[name]["mmd2"][k] - s[key]["mmd2"][k])
                              / max(abs(s[key]["mmd2"][k]), 1e-8)
                              for k in ("transported", "untransported", "floor",
                                        "type_mean")]
            replay.update(n_stored=len(stored["pairwise"]),
                          max_rel_diff_mmd2=float(max(diffs)) if diffs else None)
        boot = {"reads": transport_dist_summary(panels),
                "n_tiles": result["n_tiles"], "n_cells": result["n_cells"]}
        for key, entry in boot["reads"].items():
            block = ("summary_model" if key == "readA_gap_group"
                     else "summary_model_own")
            field = ("median_gap_closed_type_mean" if key == "readA_type_mean_own"
                     else "median_gap_closed")
            value = (((stored or {}).get(block) or {}).get("pairwise")
                     or {}).get(field)
            put(key, {**entry, "n_boot": n, "n_tiles": boot["n_tiles"],
                      "n_cells": boot["n_cells"]}, value,
                {"n_panels": entry["n_panels"], "panel_replay": replay})
        log.info("transport distribution read done (%.0f s)", time.time() - t0)

    target.write_text(json.dumps(record, indent=2, default=float))
    log.info("wrote %s", target)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reads", default=",".join(READS),
                        help=f"comma list of {', '.join(READS)}")
    parser.add_argument("--n", type=int, default=N_BOOT)
    parser.add_argument("--tile-um", type=float, default=TILE_UM)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boot-replayed", type=int, default=200,
                        help="--boot of the transport run being replayed "
                             "(distribution read's shared stream)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    reads = [r for r in args.reads.split(",") if r]
    unknown = set(reads) - set(READS)
    if unknown:
        parser.error(f"unknown reads {sorted(unknown)}")
    run_bootstrap(args.dataset, args.run, reads, args.n, args.tile_um,
                  args.seed, args.device, args.boot_replayed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
