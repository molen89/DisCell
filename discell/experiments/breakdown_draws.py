#!/usr/bin/env python3
"""Per-draw statistics of the 8.19 breakdown family, one fit at a time.

The family (devlog "8.19 revised to a lean version (author, 2026-09-29)";
``scripts/logs/breakdown_2026-09-29/FAMILY.md``) is the headline signed
contrasts, null 0. For every member this module writes the point estimate
(the statistic at unit weights, checked against the number already on disk:
``reproduces``) and ``n`` draws, each draw resampling the 200 um tiles
(``bootstrap.tile_index``) of the cells the read is computed on. As in
``bootstrap.py`` every draw is **conditional on the fitted objects** (probes,
ridge fits, kNN balls, model predictions, panels and tiers held at the
point estimate's). :mod:`discell.experiments.breakdown` pools the three
seeds of a grid point and reads kappa*.

Groups (``--groups``) and their members:

``cycle``           ``cycle_asym_q90`` = R^2(mu_z) - R^2(mu_w) on the held-out
                    top-decile cycling set (``bootstrap.cycle_cells``; both
                    ridges share the subsample, so one weight vector).
                    With ``--eval-dataset`` (the GSE dual) the fit is applied
                    to that section as ``crossslide.apply_fit`` does.
``w_mi``            ``w_niche_mi_excess`` (``bootstrap.w_mi_cells``).
``marker``          ``marker_excl_minus_ctrl_dc`` = [DP_excl(decode_corrected)
                    - DP_excl(raw)] - [DP_ctrl(decode_corrected) -
                    DP_ctrl(raw)] (``marker_pairs``' cells and calls).
``signalling``      ``signalling_response_lr_vs_other``: rank-biserial effect
                    of the per-gene response share, LR vs all other scored
                    genes (``external_criteria.signalling_share``'s panels);
                    half-tile subsampling of the held-out cells behind the
                    observed shifts (as the transport fraction-of-ceiling
                    rows: a with-replacement draw adds duplication noise to
                    an observed shift).
``axis``            ``axis_tau_true_minus_false`` (ovarian): pooled
                    w-predicted mean|tau(true)| - mean|tau(false)|
                    (``external_criteria.axis_test``). The w-predicted band
                    shift depends on the MODEL-side cells only (the band
                    means of m_psi over training folds), so the draws
                    resample those cells' tiles; bands and genes stay the
                    point estimate's.
``transport_mean``  ``transport_cf_minus_program``, ``transport_cf_minus_leak``:
                    ALL panels of the mean read (author, devlog 2026-09-29: no
                    tier selection; the trusted tier only labels the fraction-
                    of-ceiling magnitude), mean R^2(program + leak_only) minus
                    mean R^2(program_only) / R^2(leak_only); half-tile
                    subsampling of the scored cells (``subsample`` form).
``transport_dist``  ``readA_minus_typemean`` (median gap closed, own target,
                    minus the type-mean predictor's, pairwise panels) and
                    ``readB_twin_margin`` (median over panels of
                    (d_random - d_twin) / d_random, own target).

Every draw file is ``runs/<run>/breakdown/<group>.npz`` (the dual:
``dual__<eval>__cycle.npz``): ``members``, ``estimate``, ``draws`` (members x
n), ``method`` (``percentile`` | ``subsample``), ``c`` (the subsample scale)
and ``meta`` (JSON: run, kappa, seed, draw seed, reproduction checks, mask).
The draw generator of a fit is ``DRAW_SEED + 1000 * group index + run seed``,
so the three seeds of a grid point draw independently.

Usage::

    python -m discell.experiments.breakdown_draws --dataset <id> --run <run> \\
        --groups cycle,w_mi,marker,signalling [--n 10000]
    python -m discell.experiments.breakdown_draws --dataset <GSE> --run <run> \\
        --groups cycle --eval-dataset gse315411_pdltma06_10_prime_dual
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from discell.experiments import bootstrap as BS
from discell.model import eval_mask as EM

log = logging.getLogger("discell.experiments.breakdown_draws")

GROUPS = ("cycle", "w_mi", "marker", "signalling", "axis", "transport_mean",
          "transport_dist")
DRAW_SEED = 8190
N_DRAWS = 10_000
#: GPU-heavy groups default to fewer draws (still >= 11 per tail at m = 9)
N_DRAWS_HEAVY = 4_000
HEAVY = ("signalling", "axis", "transport_mean", "transport_dist")
TOL = 1e-4


def draw_seed(group: str, run_seed: int) -> int:
    return DRAW_SEED + 1000 * GROUPS.index(group) + int(run_seed)


def percentile_draws(positions: np.ndarray, statistic: Callable, n: int,
                     seed: int) -> tuple[float, np.ndarray]:
    """``statistic(weights)`` at unit weights and under *n* tile draws with
    replacement (``bootstrap.tile_draws``)."""
    tiles, n_tiles = BS.tile_index(positions)
    point = float(statistic(np.ones(len(tiles))))
    draws = np.array([statistic(w) for w in BS.tile_draws(tiles, n_tiles, n,
                                                           seed)])
    return point, draws


def half_tile_weights(positions: np.ndarray, n: int, seed: int):
    """``(n, cells)`` 0/1 weights of half-tile subsamples without
    replacement (``bootstrap.transport_mean_bootstrap``'s) and c."""
    tiles, n_tiles = BS.tile_index(positions)
    rng = np.random.default_rng(seed)
    m = n_tiles // 2
    out = np.zeros((n, len(tiles)), dtype=np.float32)
    for b in range(n):
        chosen = np.zeros(n_tiles, dtype=np.float32)
        chosen[rng.choice(n_tiles, m, replace=False)] = 1.0
        out[b] = chosen[tiles]
    return out, float(np.sqrt(m / (n_tiles - m)))


def _check(stored, estimate, tol=TOL) -> dict:
    return BS._check(stored, estimate, tol)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------
# the groups
# --------------------------------------------------------------------------


def _sweeps(trainer):
    train = trainer._sweep(trainer.train_batches)
    val = trainer._sweep(trainer.val_batches)
    return train, val


def group_cycle(config, data, trainer, stored_q90: dict, n: int, seed: int):
    from discell.model.cell_cycle import slide_cycling_set

    train, val = _sweeps(trainer)
    rows_all = np.concatenate([train["nodes"], val["nodes"]])
    z_all = np.vstack([train["mu_z"], val["mu_z"]])
    w_all = np.vstack([train["mu_w"], val["mu_w"]])
    t_all = data.t[rows_all]
    tr = np.arange(len(rows_all)) < len(train["nodes"])
    cyc = data.cycle
    scores = np.stack([cyc["s_score"], cyc["g2m_score"]], axis=1)[rows_all]
    q90 = slide_cycling_set(data)[rows_all]
    cz = BS.cycle_cells(z_all, t_all, scores, None, tr, ~tr, config.seed,
                        cells=q90)
    cw = BS.cycle_cells(w_all, t_all, scores, None, tr, ~tr, config.seed,
                        cells=q90)
    if not len(cz["rows_test"]):
        return {}
    assert np.array_equal(cz["rows_test"], cw["rows_test"])
    point, draws = percentile_draws(
        np.asarray(data.positions, float)[rows_all[cz["rows_test"]]],
        lambda w: BS.cycle_statistic(cz, w) - BS.cycle_statistic(cw, w),
        n, seed)
    stored = None
    if stored_q90.get("z") is not None and stored_q90.get("w") is not None:
        stored = stored_q90["z"] - stored_q90["w"]
    return {"cycle_asym_q90": (point, draws, "percentile", 1.0,
                               _check(stored, point))}


def group_w_mi(config, data, trainer, run_dir, n, seed):
    from discell.model.degeneracy import W_GUARD_NICHES
    from discell.model.validate import niche_labels

    _, val = _sweeps(trainer)
    niche = niche_labels(data, W_GUARD_NICHES, config.seed)
    on = EM.metric_target_mask(data.t[val["nodes"]], data.type_names)
    vrows = val["nodes"][on]
    cells = BS.w_mi_cells(val["mu_w"][on], niche[vrows], data.t[vrows],
                          config.seed)
    point, draws = percentile_draws(
        np.asarray(data.positions, float)[vrows[cells["rows"]]],
        lambda w: BS.w_mi_statistic(cells, w), n, seed)
    stored = ((_load(run_dir / "degeneracy.json") or {}).get("w_channel")
              or {}).get("w_niche_mi_excess")
    return {"w_niche_mi_excess": (point, draws, "percentile", 1.0,
                                  _check(stored, point))}


def group_marker(dataset, run, device, n):
    import dataclasses

    import torch

    from discell import paths
    from discell.experiments import marker_pairs as MP
    from discell.model.prepare import assemble
    from discell.model.train import config_from_record
    from discell.model.validate import load_run

    run_dir = paths.dataset(dataset).root / "runs" / run
    payload = torch.load(run_dir / "best.pt", map_location="cpu",
                         weights_only=False)
    config = config_from_record(payload["config"])
    del payload
    seed = draw_seed("marker", config.seed)
    full = assemble(dataset, config.variant, config.embeddings,
                    tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                    v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                    seed=config.seed, label_key=config.label_key)
    held = dataclasses.replace(full, train_tiles=[])
    config, data, trainer, run_dir, _ = load_run(dataset, run, device,
                                                 data=held)
    pairs = MP.load_pairs(dataset)
    genes = sorted({g for p in pairs for g in (p["gene_a"], p["gene_b"])})
    names = [str(g) for g in data.gene_names]
    cells = MP.collect_cells(trainer, trainer.val_batches,
                             np.array([names.index(g) for g in genes]))
    del trainer
    torch.cuda.empty_cache()
    targets = EM.metric_target_mask(data.t[cells["nodes"]], data.type_names)
    cells = {k: v[targets] for k, v in cells.items()}
    col = {g: i for i, g in enumerate(genes)}
    kinds = np.array([p["kind"] for p in pairs])
    on = MP.on_calls(cells["x"], cells["depth"], cells["rho_bar"], cells["p"],
                     float(config.kappa))
    dp = MP.double_positive(on, np.array([col[p["gene_a"]] for p in pairs]),
                            np.array([col[p["gene_b"]] for p in pairs]))

    def stat(w):
        r = MP.panel_reads(BS.weighted_mean(dp, w), kinds)
        return (r["decode_corrected-raw|exclusive"]
                - r["decode_corrected-raw|control"])
    point, draws = percentile_draws(
        np.asarray(data.positions, float)[cells["nodes"]], stat, n, seed)
    rec = _load(run_dir / "marker_pairs.json") or {}
    d = (rec.get("differences") or {}).get("decode_corrected-raw") or {}
    stored = (d["exclusive"]["estimate"] - d["control"]["estimate"]
              if d else None)
    return config, data, {"marker_excl_minus_ctrl_dc": (
        point, draws, "percentile", 1.0, _check(stored, point, 1e-6))}


def _rank_biserial_rows(v: "torch.Tensor", is_lr: "torch.Tensor"):
    """Rank-biserial 2U/(n1 n2) - 1 of every row of *v* (draws x genes), LR
    genes against the rest (ordinal ranks; ties do not occur in shares)."""
    import torch

    ranks = torch.argsort(torch.argsort(v, dim=1), dim=1).double() + 1.0
    n1 = float(is_lr.sum())
    n2 = float((~is_lr).sum())
    u = ranks[:, is_lr].sum(dim=1) - n1 * (n1 + 1) / 2.0
    return 2.0 * u / (n1 * n2) - 1.0


def group_signalling(dataset, run, config, data, trainer, b_matrix, device,
                     n, seed):
    """``external_criteria.signalling_share``'s panels (k-means niches,
    fold 0 scored), the response share's LR-vs-rest effect per draw."""
    import torch

    from discell.experiments import external_criteria as E
    from discell.model import transport as T
    from discell.model.validate import collect_latents, niche_labels

    latents = collect_latents(trainer, data)
    connected = data.graph.degrees > 0
    labels = niche_labels(data, 10, config.seed)
    held_out = latents["fold"] == 0
    n_types = len(data.p_t)
    n_niches = int(labels.max()) + 1
    model_rows = connected & ~held_out & (labels >= 0)
    group = np.full(data.graph.n_cells, -1, dtype=np.int64)
    group[model_rows] = labels[model_rows] * n_types + data.t[model_rows]
    ch = T.collect_channels(trainer, group, n_niches * n_types)
    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    built = []
    for a, b in T.pick_pairs(labels, data.graph.y, connected):
        for g in EM.exclude_types(range(n_types), data.type_names):
            rows, ok = {}, True
            for niche, side in ((a, "A"), (b, "B")):
                tr_ = np.flatnonzero(connected & ~held_out & (data.t == g)
                                     & (labels == niche))
                te = np.flatnonzero(connected & held_out & (data.t == g)
                                    & (labels == niche))
                if len(tr_) < E.MIN_CELLS or len(te) < E.MIN_CELLS // 5:
                    ok = False
                    break
                rows[side] = te
            if not ok:
                continue
            ga, gb = a * n_types + g, b * n_types + g
            response = b_matrix @ (ch["prior_w"][gb] - ch["prior_w"][ga])
            k_a = T.group_kappa(ch, ga, config.kappa)
            k_b = T.group_kappa(ch, gb, config.kappa)
            e_a = T.group_eta(ch, ga)
            leak = (np.log(T.leak_rate(ch["rho"][ga], ch["rho_bar"][gb], k_b,
                                       e_a) + T.EPS)
                    - np.log(T.leak_rate(ch["rho"][ga], ch["rho_bar"][ga], k_a,
                                         e_a) + T.EPS))
            obs_a = np.asarray(x_rate[rows["A"]].mean(axis=0)).ravel()
            obs_b = np.asarray(x_rate[rows["B"]].mean(axis=0)).ravel()
            keep = np.flatnonzero((obs_a > T.MIN_RATE) & (obs_b > T.MIN_RATE))
            if len(keep) < 100:
                continue
            built.append({"keep": keep, "response": response[keep],
                          "leak": leak[keep], "rows_a": rows["A"],
                          "rows_b": rows["B"],
                          "observed": (np.log(obs_b[keep] + T.EPS)
                                       - np.log(obs_a[keep] + T.EPS))})
    gene_names = np.asarray([str(x) for x in data.gene_names])
    shares = E.accumulate_shares(built, len(gene_names))
    is_lr = np.isin(gene_names, sorted(E.lr_gene_union(set(gene_names.tolist()))))
    scored = shares["scored"]
    v = shares["response"]
    point = E.mann_whitney(v[scored & is_lr], v[scored & ~is_lr])["effect"]
    # draws: half-tile subsamples of every held-out cell a panel uses
    cells = np.unique(np.concatenate([np.concatenate([p["rows_a"], p["rows_b"]])
                                      for p in built]))
    pos = np.full(data.graph.n_cells, -1, dtype=np.int64)
    pos[cells] = np.arange(len(cells))
    weights, c = half_tile_weights(np.asarray(data.positions, float)[cells], n,
                                   seed)
    dev = torch.device(device)
    n_genes = len(gene_names)
    acc = torch.zeros((3, n, n_genes), dtype=torch.float64, device=dev)
    chunk = 250
    for p in built:
        keep = torch.as_tensor(p["keep"], device=dev)
        resp = torch.as_tensor(p["response"] - p["response"].mean(), device=dev)
        leak = torch.as_tensor(p["leak"] - p["leak"].mean(), device=dev)
        xs = [torch.as_tensor(np.asarray(x_rate[r][:, p["keep"]].todense()),
                              dtype=torch.float32, device=dev)
              for r in (p["rows_a"], p["rows_b"])]
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            means = []
            for x, r in zip(xs, (p["rows_a"], p["rows_b"])):
                w = torch.as_tensor(weights[lo:hi, pos[r]], device=dev)
                tot = w.sum(dim=1, keepdim=True)
                means.append(((w @ x) / tot.clamp(min=1e-12)).double())
            obs = torch.log(means[1] + T.EPS) - torch.log(means[0] + T.EPS)
            obs = obs - obs.mean(dim=1, keepdim=True)
            valid = torch.isfinite(obs).all(dim=1)
            obs = torch.where(valid[:, None], obs, torch.zeros_like(obs))
            vmask = valid[:, None].double()
            acc[0, lo:hi][:, keep] += vmask * resp.abs()[None, :]
            acc[1, lo:hi][:, keep] += vmask * leak.abs()[None, :]
            acc[2, lo:hi][:, keep] += vmask * (obs - resp - leak).abs()
        del xs
    total = acc.sum(dim=0)
    sc = torch.as_tensor(scored, device=dev)
    share = acc[0][:, sc] / total[:, sc].clamp(min=1e-12)
    lr = torch.as_tensor(is_lr[scored], device=dev)
    draws = _rank_biserial_rows(share, lr).cpu().numpy()
    stored = ((((_load(paths_dataset(dataset) / "experiments" /
                       f"external_signalling_share_{run}.json") or {})
                .get("tests") or {}).get("response") or {})
              .get("all_other") or {}).get("effect")
    return {"signalling_response_lr_vs_other": (
        float(point), draws, "subsample", c, _check(stored, point, 1e-6))}


def paths_dataset(dataset):
    from discell import paths
    return paths.dataset(dataset).root


def group_axis(dataset, run, config, data, trainer, b_matrix, n, seed):
    """``external_criteria.axis_test``'s pooled w-predicted row; the draws
    resample the model-side cells behind each band's mean m_psi."""
    from scipy.stats import spearmanr

    from discell.experiments import external_criteria as E
    from discell.model import transport as T
    from discell.model.validate import collect_latents

    latents = collect_latents(trainer, data)
    train, val = _sweeps(trainer)
    prior = np.zeros((data.graph.n_cells, train["prior_w"].shape[1]))
    prior[train["nodes"]] = train["prior_w"]
    prior[val["nodes"]] = val["prior_w"]
    connected = data.graph.degrees > 0
    bands = T.tumour_band_labels(data)
    held_out = latents["fold"] == 0
    n_types = len(data.p_t)
    n_bands = int(bands.max()) + 1
    have = connected & (bands >= 0)
    rho_x = abs(float(spearmanr(data.positions[have, 0], bands[have])[0]))
    rho_y = abs(float(spearmanr(data.positions[have, 1], bands[have])[0]))
    false_field = data.positions[:, 0 if rho_x <= rho_y else 1]
    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    specs = []          # per (type, axis): member rows per kept band, genes
    for g in EM.exclude_types(range(n_types), data.type_names):
        members = connected & (data.t == g)
        if members.sum() < E.MIN_BAND_CELLS * n_bands:
            continue
        fb = np.full(len(bands), -1, dtype=np.int64)
        fb[members] = E.equal_count_bins(false_field[members], E.N_FALSE_BINS)
        per = {}
        for axis, lab in (("true", bands), ("false", fb)):
            n_groups = int(lab[members].max()) + 1
            keep_band, band_rows, raw = [], [], []
            for b in range(n_groups):
                model = np.flatnonzero(members & ~held_out & (lab == b))
                te = np.flatnonzero(members & held_out & (lab == b))
                if len(model) < E.MIN_CELLS or len(te) < E.MIN_BAND_CELLS:
                    continue
                keep_band.append(b)
                band_rows.append(model)
                raw.append(np.log(np.asarray(x_rate[te].mean(axis=0)).ravel()
                                  + T.EPS))
            if len(keep_band) < 4:
                per = {}
                break
            per[axis] = {"order": np.array(keep_band, float),
                         "rows": band_rows,
                         "genes": np.all(np.exp(np.array(raw)) > T.MIN_RATE,
                                         axis=0)}
        if "true" in per and "false" in per:
            common = per["true"]["genes"] & per["false"]["genes"]  # same genes
            per["true"]["genes"] = per["false"]["genes"] = common  # both axes
            specs.append(per)
    universe = np.unique(np.concatenate([r for s in specs for a in s.values()
                                         for r in a["rows"]]))
    pos = np.full(data.graph.n_cells, -1, dtype=np.int64)
    pos[universe] = np.arange(len(universe))

    def stat(weights):
        tt_all, tf_all = [], []
        for s in specs:
            taus = {}
            for axis in ("true", "false"):
                a = s[axis]
                prog = []
                for r in a["rows"]:
                    w = weights[pos[r]]
                    if w.sum() <= 0:
                        return float("nan")
                    prog.append(b_matrix @ (w @ prior[r] / w.sum()))
                taus[axis] = E.kendall_tau_rows(np.array(prog)[:, a["genes"]],
                                                a["order"])
            tt, tf = taus["true"], taus["false"]
            m = min(len(tt), len(tf))
            tt_all.append(tt[:m])
            tf_all.append(tf[:m])
        tt, tf = np.concatenate(tt_all), np.concatenate(tf_all)
        return float(np.abs(tt).mean() - np.abs(tf).mean())
    point, draws = percentile_draws(np.asarray(data.positions, float)[universe],
                                    stat, n, seed)
    rec = _load(paths_dataset(dataset) / "experiments"
                / f"external_axis_test_{run}.json") or {}
    row = (rec.get("pooled") or {}).get("w_predicted")
    stored = (row["mean_abs_tau_true"] - row["mean_abs_tau_false"]
              if row else None)
    return {"axis_tau_true_minus_false": (point, draws, "percentile", 1.0,
                                          _check(stored, point, 1e-3))}


def select_panels(panels: list[dict]) -> list[dict]:
    """The transport members' panels: every panel the mean read kept (all
    tiers; excluded types already left in the replay)."""
    return list(panels)


def all_panel_contrasts(stored_panels: list[dict]) -> dict:
    """The two transport members' point estimates from ``transport.json``'s
    per-panel R^2 over ALL panels (its summary holds per-tier means only)."""
    if not stored_panels:
        return {}
    r2 = {k: np.array([p[k]["r2"] for p in stored_panels], float)
          for k in ("counterfactual", "program_only", "leak_only")}
    return {"transport_cf_minus_program": float(
                r2["counterfactual"].mean() - r2["program_only"].mean()),
            "transport_cf_minus_leak": float(
                r2["counterfactual"].mean() - r2["leak_only"].mean())}


def group_transport_mean(config, data, trainer, run_dir, b_matrix, device,
                         n, seed, scored_cells="fold0"):
    """The mean read's trusted extrapolation panels (``bootstrap.
    transport_mean_panels``) with their program-only and leak-only
    predictions re-formed from the same group channels."""
    import torch

    from discell.model import transport as T
    from discell.model.validate import collect_latents, niche_labels

    labels = niche_labels(data, 10, config.seed)
    fold = collect_latents(trainer, data)["fold"]
    panels = BS.transport_mean_panels(trainer, data, config, b_matrix, labels,
                                      fold, device, scored_cells)
    panels = select_panels(panels)
    out = {}
    stored = all_panel_contrasts((_load(T.out_root(run_dir, scored_cells)
                                        / "transport" / "transport.json")
                                  or {}).get("panels") or [])
    if not panels:
        return out
    connected = data.graph.degrees > 0
    _, model_side = T.split_cells(data, fold, scored_cells)
    n_types = len(data.p_t)
    n_groups = (int(labels.max()) + 1) * n_types
    model_rows = connected & model_side & (labels >= 0)
    group = np.full(data.graph.n_cells, -1, dtype=np.int64)
    group[model_rows] = labels[model_rows] * n_types + data.t[model_rows]
    ch = T.collect_channels(trainer, group, n_groups)
    names = [str(x) for x in data.type_names]
    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    for p in panels:
        g = names.index(p["type"])
        ga, gb = p["pair"][0] * n_types + g, p["pair"][1] * n_types + g
        program = b_matrix @ (ch["prior_w"][gb] - ch["prior_w"][ga])
        k_a = T.group_kappa(ch, ga, config.kappa)
        k_b = T.group_kappa(ch, gb, config.kappa)
        e_a = T.group_eta(ch, ga)
        leak = (np.log(T.leak_rate(ch["rho"][ga], ch["rho_bar"][gb], k_b, e_a)
                       + T.EPS)
                - np.log(T.leak_rate(ch["rho"][ga], ch["rho_bar"][ga], k_a, e_a)
                         + T.EPS))
        p["program"], p["leak"] = program[p["keep"]], leak[p["keep"]]
        assert np.allclose(p["program"] + p["leak"], p["prediction"],
                           atol=1e-5), "channels drifted from the panel replay"
    cells = np.unique(np.concatenate([np.concatenate([p["rows_a"], p["rows_b"]])
                                      for p in panels]))
    pos = np.full(data.graph.n_cells, -1, dtype=np.int64)
    pos[cells] = np.arange(len(cells))
    weights, c = half_tile_weights(np.asarray(data.positions, float)[cells], n,
                                   seed)
    dev = torch.device(device)
    r2 = {k: np.full((n + 1, len(panels)), np.nan)
          for k in ("cf", "program", "leak")}
    chunk = 250
    for j, p in enumerate(panels):
        preds = {k: torch.as_tensor(v - v.mean(), device=dev)
                 for k, v in (("cf", p["prediction"]), ("program", p["program"]),
                              ("leak", p["leak"]))}
        xs = [torch.as_tensor(np.asarray(x_rate[r][:, p["keep"]].todense()),
                              dtype=torch.float32, device=dev)
              for r in (p["rows_a"], p["rows_b"])]
        for lo in range(0, n + 1, chunk):
            hi = min(lo + chunk, n + 1)
            means = []
            for x, r in zip(xs, (p["rows_a"], p["rows_b"])):
                # row 0 = unit weights (the point), rows 1.. = the subsamples
                w = np.vstack([np.ones((1, len(r)), np.float32),
                               weights[:, pos[r]]])[lo:hi]
                w = torch.as_tensor(w, device=dev)
                means.append(((w @ x) / w.sum(dim=1, keepdim=True)
                              .clamp(min=1e-12)).double())
            obs = torch.log(means[1] + T.EPS) - torch.log(means[0] + T.EPS)
            oc = obs - obs.mean(dim=1, keepdim=True)
            den = (oc ** 2).sum(dim=1).clamp(min=1e-12)
            for k, pc in preds.items():
                r2[k][lo:hi, j] = (1.0 - ((oc - pc[None, :]) ** 2).sum(dim=1)
                                   / den).cpu().numpy()
        empty = np.concatenate([[False], (weights[:, pos[p["rows_a"]]].sum(1) == 0)
                                | (weights[:, pos[p["rows_b"]]].sum(1) == 0)])
        for k in r2:
            r2[k][empty, j] = np.nan
    for key, other, name in (("cf", "program", "transport_cf_minus_program"),
                             ("cf", "leak", "transport_cf_minus_leak")):
        diff = np.nanmean(r2[key], axis=1) - np.nanmean(r2[other], axis=1)
        s = stored.get(name)
        out[name] = (float(diff[0]), diff[1:], "subsample", c,
                     {**_check(s, float(diff[0]), 1e-4),
                      "n_panels": len(panels)})
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Median of *values* (m,) repeated by each row of integer *weights*
    (n, m) -- ``np.median`` of the resampled multiset (the two middle
    values averaged when its size is even); NaN for an empty draw."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    cum = np.cumsum(np.rint(weights[:, order]).astype(np.int64), axis=1)
    total = cum[:, -1]
    out = np.full(len(weights), np.nan)
    for i in np.flatnonzero(total > 0):
        lo = np.searchsorted(cum[i], (total[i] - 1) // 2, side="right")
        hi = np.searchsorted(cum[i], total[i] // 2, side="right")
        out[i] = 0.5 * (v[lo] + v[hi])
    return out


def group_transport_dist(config, data, trainer, run_dir, device, n, seed,
                         boot_replayed=200, scored_cells="fold0"):
    """Read A (``bootstrap.transport_dist_panels``' tile draws, the paired
    difference per draw) and Read B (per-target-cell twin distances,
    weighted medians under tile draws of the target cells)."""
    from discell.model import transport as T
    from discell.model.validate import collect_latents, niche_labels

    labels = niche_labels(data, 10, config.seed)
    fold = collect_latents(trainer, data)["fold"]
    res = BS.transport_dist_panels(trainer, data, config, labels, fold, device,
                                   boot_replayed, n, BS.TILE_UM, seed,
                                   scored_cells, twin_cells=True)
    panels = res["panels"]
    root = T.out_root(run_dir, scored_cells) / "transport"
    out = {}
    # Read A: median gap (own, transported) - median gap (own, type-mean)
    pt, pm, dt, dm = [], [], [], []
    for e in panels:
        s = e["own"]
        if s["insufficient"]:
            continue
        u = s["draws"]["untransported"]
        pt.append(float(BS._gap(s["mmd2"]["untransported"],
                                s["mmd2"]["transported"], s["mmd2"]["floor"])))
        pm.append(float(BS._gap(s["mmd2"]["untransported"],
                                s["mmd2"]["type_mean"], s["mmd2"]["floor"])))
        dt.append(BS._gap(u, s["draws"]["transported"], s["mmd2"]["floor"]))
        dm.append(BS._gap(u, np.full(len(u), s["mmd2"]["type_mean"]),
                          s["mmd2"]["floor"]))
    dist = _load(root / "transport_distribution.json") or {}
    summ = ((dist.get("summary_model_own") or {}).get("pairwise") or {})
    if pt:
        point = float(np.nanmedian(pt) - np.nanmedian(pm))
        draws = (np.nanmedian(np.stack(dt, 1), 1)
                 - np.nanmedian(np.stack(dm, 1), 1))
        stored = (summ["median_gap_closed"] - summ["median_gap_closed_type_mean"]
                  if "median_gap_closed_type_mean" in summ else None)
        out["readA_minus_typemean"] = (point, draws, "percentile", 1.0,
                                       {**_check(stored, point, 1e-3),
                                        "n_panels": len(pt)})
    # Read B: twin margin, own target
    tw = [e["twins_own"] for e in panels
          if not e["twins_own"].get("insufficient")]
    if tw:
        tgt = np.unique(np.concatenate([t["tgt"] for t in tw]))
        tiles, n_tiles = BS.tile_index(np.asarray(data.positions, float)[tgt])
        where = np.full(data.graph.n_cells, -1, dtype=np.int64)
        where[tgt] = np.arange(len(tgt))
        rng = np.random.default_rng(seed + 7)
        mults = np.stack([np.bincount(rng.integers(0, n_tiles, n_tiles),
                                      minlength=n_tiles)[tiles]
                          for _ in range(n)])
        pts, drs = [], []
        for t in tw:
            med_tr = np.median(t["d_tr"])
            med_rd = np.median(t["d_rd"])
            pts.append((med_rd - med_tr) / med_rd if med_rd > 1e-12 else np.nan)
            w = mults[:, where[t["tgt"]]]
            wt = _weighted_median(t["d_tr"], w)
            wr = _weighted_median(t["d_rd"], w)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.where(w.sum(1) > 0, (wr - wt) / np.where(wr > 1e-12, wr,
                                                                  np.nan), np.nan)
            drs.append(m)
        point = float(np.nanmedian(pts))
        draws = np.nanmedian(np.stack(drs, 1), 1)
        twins = _load(root / "transport_twins.json") or {}
        stored = None
        blk = ((twins.get("summary_own") or {}).get("pairwise") or {})
        stored = blk.get("median_twin_margin")
        out["readB_twin_margin"] = (point, draws, "percentile", 1.0,
                                    {**_check(stored, point, 1e-6),
                                     "n_panels": len(tw)})
    return out


# --------------------------------------------------------------------------
# one fit
# --------------------------------------------------------------------------


def write_group(path: Path, members: dict, meta: dict) -> None:
    names = list(members)
    n = max(len(members[k][1]) for k in names)
    draws = np.full((len(names), n), np.nan)
    for i, k in enumerate(names):
        draws[i, :len(members[k][1])] = members[k][1]
    meta = {**meta, "checks": {k: members[k][4] for k in names}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, members=np.array(names),
             estimate=np.array([members[k][0] for k in names], float),
             draws=draws, method=np.array([members[k][2] for k in names]),
             c=np.array([members[k][3] for k in names], float),
             meta=np.array(json.dumps(meta, default=float)))
    tmp.replace(path)


def run(dataset: str, run_name: str, groups: Sequence[str], n: int | None,
        device: str = "cuda", eval_dataset: str | None = None,
        scored_cells: str = "fold0", boot_replayed: int = 200) -> dict:
    from discell.experiments.breakdown import draws_path
    from discell.model.validate import load_run

    written = {}
    loaded = None
    for group in groups:
        t0 = time.time()
        n_g = n or (N_DRAWS_HEAVY if group in HEAVY else N_DRAWS)
        if group == "marker":
            config, data, members = group_marker(dataset, run_name, device,
                                                 n_g)
            run_dir = paths_dataset(dataset) / "runs" / run_name
        else:
            if loaded is None:
                loaded = load_run(dataset, run_name, device)
            config, data, trainer, run_dir, b_matrix = loaded
            seed = draw_seed(group, config.seed)
            dev = str(next(trainer.model.parameters()).device)
            if group == "cycle" and eval_dataset:
                data_b, trainer_b = _dual(config, trainer, eval_dataset)
                rec = _load(run_dir / "crossslide" / f"{eval_dataset}.json") or {}
                q = ((rec.get("held_out_section") or {}).get("cycle_q90") or {})
                stored = {k: (q.get(k) or {}).get("r2_pooled") for k in "zw"}
                members = group_cycle(config, data_b, trainer_b, stored, n_g,
                                      seed)
            elif group == "cycle":
                q = (((_load(run_dir / "degeneracy.json") or {}).get("battery")
                      or {}).get("cycle_q90") or {})
                stored = {k: (q.get(k) or {}).get("r2_pooled") for k in "zw"}
                members = group_cycle(config, data, trainer, stored, n_g, seed)
            elif group == "w_mi":
                members = group_w_mi(config, data, trainer, run_dir, n_g, seed)
            elif group == "signalling":
                members = group_signalling(dataset, run_name, config, data,
                                           trainer, b_matrix, dev, n_g, seed)
            elif group == "axis":
                members = group_axis(dataset, run_name, config, data, trainer,
                                     b_matrix, n_g, seed)
            elif group == "transport_mean":
                members = group_transport_mean(config, data, trainer, run_dir,
                                               b_matrix, dev, n_g, seed,
                                               scored_cells)
            elif group == "transport_dist":
                members = group_transport_dist(config, data, trainer, run_dir,
                                               dev, n_g, seed, boot_replayed,
                                               scored_cells)
            else:
                raise ValueError(f"unknown group {group!r}")
        path = draws_path(run_dir, group, eval_dataset if group == "cycle"
                          else None)
        meta = {"dataset": dataset, "run": run_name,
                "kappa": float(config.kappa), "seed": int(config.seed),
                "group": group, "n_draws": n_g,
                "draw_seed": draw_seed(group, config.seed),
                "eval_dataset": eval_dataset, "scored_cells": scored_cells,
                "tile_um": BS.TILE_UM,
                "eval_mask": EM.record(),
                "conditional_on": "the fitted objects (bootstrap.py convention)"}
        if members:
            write_group(path, members, meta)
        for k, v in members.items():
            log.info("%s %s %s: %.4f [%.4f, %.4f] (%s) reproduces=%s  %.0fs",
                     run_name, group, k, v[0], *np.nanpercentile(v[1], [2.5, 97.5]),
                     v[2], v[4].get("reproduces"), time.time() - t0)
        if not members:
            log.warning("%s %s: no member defined on this fit", run_name, group)
        written[group] = {k: {"estimate": v[0], **v[4]} for k, v in
                          members.items()}
    return written


def _dual(config, trainer, eval_dataset):
    """The held-out section under the fit's assembly settings, with the
    fit's model (``crossslide.evaluate_on`` / ``apply_fit``)."""
    from discell.model.prepare import assemble
    from discell.model.train import Trainer

    data_b = assemble(eval_dataset, config.variant, config.embeddings,
                      tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                      v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                      seed=config.seed, label_key=config.label_key)
    if [str(x) for x in data_b.type_names] != [str(x) for x in
                                               trainer.data.type_names]:
        raise ValueError("type vocabularies differ")
    trainer_b = Trainer(config, data_b)
    trainer_b.model = trainer.model.eval()
    return data_b, trainer_b


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--groups", required=True,
                        help=f"comma list of {', '.join(GROUPS)}")
    parser.add_argument("--n", type=int, default=None,
                        help=f"draws (default {N_DRAWS}; {N_DRAWS_HEAVY} for "
                             f"{', '.join(HEAVY)})")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-dataset", default=None,
                        help="cycle only: apply the fit to this section")
    parser.add_argument("--scored-cells", default="fold0",
                        choices=("fold0", "heldout-tiles"))
    parser.add_argument("--boot-replayed", type=int, default=200)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    groups = [g for g in args.groups.split(",") if g]
    bad = set(groups) - set(GROUPS)
    if bad:
        parser.error(f"unknown groups {sorted(bad)}")
    if args.eval_dataset and groups != ["cycle"]:
        parser.error("--eval-dataset reads the cycle group only")
    out = run(args.dataset, args.run, groups, args.n, args.device,
              args.eval_dataset, args.scored_cells, args.boot_replayed)
    failed = [f"{g}:{k}" for g, m in out.items() for k, v in m.items()
              if v.get("reproduces") is False]
    if failed:
        log.error("point estimates NOT reproduced: %s", failed)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
