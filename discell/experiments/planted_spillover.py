#!/usr/bin/env python3
"""Planted spill-over positive control (devlog "RECOMB story map approved;
three additions (motivation, 2026-10-01; author)", item 1).

Does the breakdown point kappa* of a contrast created ONLY by spill-over
track the spill-over that created it? **The world** is
:func:`discell.model.synthetic.simulate` at the recovery settings (K 8,
G 60, planted d_z 4, d_w 2, d_Phi 8, mean counts 150) on a larger tissue,
N = 30,000 cells at the recovery density (box sqrt(N/6000) x 1 mm), so
that (niche, type) panels have enough held-out cells. Twenty of the 60
genes, drawn with the ``[world_seed, 7]`` stream, get **no response**:
their columns of B are zero (``simulate(b_zero=...)``), so whatever niche
dependence they show is the leak mixture at kappa_true alone. The other 40
carry the planted response ``w B`` (and leak too). kappa_true in {0, 0.1,
0.2}; world seed 0 for every kappa_true (only p differs between them).

**The fit** is ``synthetic_recovery.fit_config`` (the final configuration
at the recovery widths) with one change: d_w = 6, the final configuration's
response width (:data:`D_W_FIT`). At the recovery width d_w = 2 (= the
planted d_w) the response channel has no spare capacity, and a smoke fit
(kappa_true 0.2, assumed 0) put none of the unmodelled spill into the
programme (spill contrast 5e-5), so the control would have nothing to break.
Every assumed kappa of the breakdown grid, model seeds 0-2.

**The contrast** is the transport member ``transport_cf_minus_leak``
(``breakdown_draws.group_transport_mean``): over all panels, mean R^2 of the
counterfactual (programme + leak-only) minus mean R^2 of leak-only, read
on one gene set. It is the claimed member that asks whether the response
channel adds to the predicted niche shift beyond spill-over; it is signed
with a non-degenerate kappa = 0 reference (leak-only predicts no shift at
kappa = 0, so R^2(leak) = 0 there), and, unlike I(niche; w), it can be read
per gene set, so the spill-only and the genuine contrast sit in one world:

* ``spill_cf_minus_leak``: on the 20 no-response genes. Predicted to hold
  while kappa < kappa_true (the programme must absorb the unmodelled leak)
  and to break at the first grid kappa >= kappa_true;
* ``response_cf_minus_leak``: on the 40 response genes. Predicted to hold
  across the grid.

Panels are built as ``transport.transport_check`` builds them, scaled to
the simulation: k-means niches on y (``validate.niche_labels``, 8 niches,
the model seed), every niche pair x type with >= 150 model-side and >= 30
scored cells per side, fold 0 scored (tiles k mod 5, as
``collect_latents``), group channels from ``transport.collect_channels``,
genes kept where both scored niche means exceed ``transport.MIN_RATE``.
Draws: half-tile subsampling of the scored cells (200 um tiles,
``breakdown_draws.half_tile_weights``), the ``subsample`` interval form,
draw seed ``breakdown_draws.draw_seed("transport_mean", seed)``.
kappa*: :func:`discell.experiments.breakdown.section_table` (3 seeds pooled,
two-sided 1 - 0.05/m with m = 2, the two members of a world).

Usage::

    python -m discell.experiments.planted_spillover fit --kappa-true 0.1 \\
        --assumed 0.05 --model-seed 0
    python -m discell.experiments.planted_spillover aggregate
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from discell.experiments import synthetic_recovery as SR

log = logging.getLogger("discell.experiments.planted_spillover")

N_CELLS = 30_000
BOX_UM = 1000.0 * np.sqrt(N_CELLS / SR.N_CELLS)    # the recovery density
WORLD_SEED = 0
N_SPILL = 20
KAPPA_TRUE = (0.0, 0.1, 0.2)
SEEDS = (0, 1, 2)
N_NICHES = 8
MIN_MODEL, MIN_SCORED = 150, 30
N_DRAWS = 2000
MEMBERS = {"spill_cf_minus_leak": "spill", "response_cf_minus_leak": "response"}
MIN_SET_GENES = 5
#: the fitted response width: the final configuration's d_w (6), not the
#: recovery width (2 = the planted d_w), so that w has spare capacity an
#: unmodelled leak could occupy, as on the real sections
D_W_FIT = 6
OUT = Path("scripts/logs/recomb_additions_2026-10-01/spillover")
RESULT = Path(f"data/datasets/{SR.DATASET}/experiments/planted_spillover")


# -- the world ---------------------------------------------------------------

def spill_genes(world_seed: int = WORLD_SEED, n_genes: int = 60) -> np.ndarray:
    rng = np.random.default_rng([world_seed, 7])
    mask = np.zeros(n_genes, dtype=bool)
    mask[rng.choice(n_genes, N_SPILL, replace=False)] = True
    return mask


def world(kappa_true: float, world_seed: int = WORLD_SEED):
    from discell.model.synthetic import simulate

    return simulate(n_cells=N_CELLS, n_types=SR.N_TYPES, kappa=kappa_true,
                    box_um=BOX_UM, seed=world_seed,
                    b_zero=spill_genes(world_seed))


def run_name(kappa_true: float, assumed: float, seed: int) -> str:
    return f"spill_p{kappa_true:g}_a{assumed:g}_m{seed}"


# -- the read ----------------------------------------------------------------

def folds(data) -> np.ndarray:
    """``collect_latents``' fold: prepare tile k -> fold k mod N_FOLDS."""
    from discell.model.validate import N_FOLDS

    fold = np.zeros(data.graph.n_cells, dtype=np.int64)
    for k, tile in enumerate(data.train_tiles + data.val_tiles):
        fold[tile] = k % N_FOLDS
    return fold


def build_panels(data, labels: np.ndarray, scored: np.ndarray,
                 model_side: np.ndarray) -> list[dict]:
    """(pair, type) panels with enough cells on both sides, in
    ``transport_check``'s order (``pick_pairs``, then types)."""
    from discell.model import transport as T

    connected = data.graph.degrees > 0
    out = []
    for a, b in T.pick_pairs(labels, data.graph.y, connected):
        for g in range(len(data.p_t)):
            rows = {}
            for niche, side in ((a, "A"), (b, "B")):
                base = connected & (data.t == g) & (labels == niche)
                n_model = int((base & model_side).sum())
                te = np.flatnonzero(base & scored)
                if n_model < MIN_MODEL or len(te) < MIN_SCORED:
                    break
                rows[side] = te
            else:
                out.append({"pair": (int(a), int(b)), "type": g,
                            "rows_a": rows["A"], "rows_b": rows["B"]})
    return out


def channel_predictions(trainer, data, labels, model_side, panels, kappa,
                        b_matrix) -> None:
    """Programme and leak-only shifts per panel (``group_transport_mean``'s
    formulas, global kappa), added to each panel in place."""
    from discell.model import transport as T

    connected = data.graph.degrees > 0
    n_types = len(data.p_t)
    n_groups = (int(labels.max()) + 1) * n_types
    model_rows = connected & model_side & (labels >= 0)
    group = np.full(data.graph.n_cells, -1, dtype=np.int64)
    group[model_rows] = labels[model_rows] * n_types + data.t[model_rows]
    ch = T.collect_channels(trainer, group, n_groups)
    for p in panels:
        ga = p["pair"][0] * n_types + p["type"]
        gb = p["pair"][1] * n_types + p["type"]
        p["program"] = b_matrix @ (ch["prior_w"][gb] - ch["prior_w"][ga])
        p["leak"] = (np.log(T.leak_rate(ch["rho"][ga], ch["rho_bar"][gb], kappa)
                            + T.EPS)
                     - np.log(T.leak_rate(ch["rho"][ga], ch["rho_bar"][ga],
                                          kappa) + T.EPS))


def r2_rows(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """``transport.score_shift``'s R^2 (both sides centred, zero-prediction
    null) for every row of *obs* (draws x genes) against one *pred*."""
    oc = obs - obs.mean(axis=1, keepdims=True)
    pc = pred - pred.mean()
    den = np.maximum((oc ** 2).sum(axis=1), 1e-12)
    return 1.0 - ((oc - pc[None, :]) ** 2).sum(axis=1) / den


def contrast_draws(data, panels, gene_sets: dict, n: int, seed: int,
                   positions: np.ndarray | None = None) -> tuple[dict, float]:
    """``{member: (n + 1,)}``: row 0 the point (unit weights), rows 1.. the
    half-tile subsamples; and the subsample scale c."""
    from discell.experiments.breakdown_draws import half_tile_weights
    from discell.model import transport as T

    positions = (np.asarray(data.positions, float) if positions is None
                 else positions)
    cells = np.unique(np.concatenate([np.concatenate([p["rows_a"], p["rows_b"]])
                                      for p in panels]))
    pos = np.full(data.graph.n_cells, -1, dtype=np.int64)
    pos[cells] = np.arange(len(cells))
    weights, c = half_tile_weights(positions[cells], n, seed)
    weights = np.vstack([np.ones((1, len(cells)), np.float32), weights])
    x = data.x.toarray() if hasattr(data.x, "toarray") else np.asarray(data.x)
    rate = x / np.clip(np.asarray(data.totals, float), 1.0, None)[:, None]
    r2 = {m: {k: np.full((n + 1, len(panels)), np.nan) for k in ("cf", "leak")}
          for m in gene_sets}
    for j, p in enumerate(panels):
        means = []
        for r in (p["rows_a"], p["rows_b"]):
            w = weights[:, pos[r]].astype(np.float64)
            tot = w.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                means.append((w @ rate[r]) / tot)
        point_a, point_b = means[0][0], means[1][0]
        keep = (point_a > T.MIN_RATE) & (point_b > T.MIN_RATE)
        with np.errstate(invalid="ignore", divide="ignore"):
            obs = np.log(means[1] + T.EPS) - np.log(means[0] + T.EPS)
        empty = ~np.isfinite(obs).all(axis=1)
        for member, mask in gene_sets.items():
            genes = keep & mask
            if genes.sum() < MIN_SET_GENES:
                continue
            o = obs[:, genes]
            for k, pred in (("cf", p["program"] + p["leak"]),
                            ("leak", p["leak"])):
                col = r2_rows(np.where(np.isfinite(o), o, 0.0), pred[genes])
                col[empty] = np.nan
                r2[member][k][:, j] = col
    out = {}
    for member in gene_sets:
        with np.errstate(invalid="ignore"):
            out[member] = (np.nanmean(r2[member]["cf"], axis=1)
                           - np.nanmean(r2[member]["leak"], axis=1))
    return out, c


def oracle(sim, panels, gene_sets: dict) -> dict:
    """The planted structure on the same panels: mean R^2 against the
    observed shift of the TRUE shift (the scored cells' mean p_true), and
    the true shift's spread, per gene set."""
    from discell.model import transport as T

    rate = sim.x / sim.totals[:, None]
    out = {m: {"r2": [], "sd": []} for m in gene_sets}
    for p in panels:
        ra, rb = p["rows_a"], p["rows_b"]
        oa, ob = rate[ra].mean(0), rate[rb].mean(0)
        keep = (oa > T.MIN_RATE) & (ob > T.MIN_RATE)
        obs = np.log(ob + T.EPS) - np.log(oa + T.EPS)
        true = np.log(sim.p_true[rb].mean(0)) - np.log(sim.p_true[ra].mean(0))
        for m, mask in gene_sets.items():
            g = keep & mask
            if g.sum() < MIN_SET_GENES:
                continue
            out[m]["r2"].append(float(r2_rows(obs[g][None], true[g])[0]))
            out[m]["sd"].append(float((true[g] - true[g].mean()).std()))
    return {m: {"mean_r2_true_shift": float(np.mean(v["r2"])) if v["r2"] else None,
                "mean_sd_true_shift": float(np.mean(v["sd"])) if v["sd"] else None,
                "n_panels": len(v["r2"])} for m, v in out.items()}


def read(trainer, data, sim, config, n: int, seed: int) -> dict:
    from discell.experiments.breakdown_draws import draw_seed
    from discell.model.validate import niche_labels

    spill = spill_genes(WORLD_SEED, sim.x.shape[1])
    gene_sets = {"spill_cf_minus_leak": spill, "response_cf_minus_leak": ~spill}
    labels = niche_labels(data, N_NICHES, config.seed)
    scored = folds(data) == 0
    panels = build_panels(data, labels, scored, ~scored)
    if not panels:
        return {"n_panels": 0, "members": {}, "oracle": None}
    b_matrix = trainer.model.B.weight.detach().cpu().numpy()
    channel_predictions(trainer, data, labels, ~scored, panels,
                        float(config.kappa), b_matrix)
    values, c = contrast_draws(data, panels, gene_sets, n,
                               draw_seed("transport_mean", seed))
    members = {m: {"estimate": float(v[0]), "draws": v[1:], "c": c,
                   "method": "subsample"} for m, v in values.items()}
    return {"n_panels": len(panels), "members": members,
            "oracle": oracle(sim, panels, gene_sets),
            "panels": [{"pair": p["pair"], "type": int(p["type"]),
                        "n_scored": [len(p["rows_a"]), len(p["rows_b"])]}
                       for p in panels]}


# -- one fit -----------------------------------------------------------------

def fit_one(kappa_true: float, assumed: float, seed: int, n: int = N_DRAWS,
            epochs: int | None = None, patience: int | None = None,
            device: str = "cuda", out_dir: Path = OUT / "fits") -> dict:
    import torch

    from discell.experiments.breakdown_draws import write_group
    from discell.model.train import Trainer

    started = time.time()
    sim = world(kappa_true)
    data = SR.model_data(sim, seed)
    name = run_name(kappa_true, assumed, seed)
    config = dataclasses.replace(
        SR.fit_config(sim, assumed, seed, name, False, epochs, patience,
                      device), d_w=D_W_FIT)
    trainer = Trainer(config, data)
    summary = trainer.fit()                       # best checkpoint restored
    trainer.model.eval()
    res = read(trainer, data, sim, config, n, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run": name, "kappa_true": kappa_true, "kappa": assumed,
            "seed": seed, "n_panels": res["n_panels"]}
    if res["members"]:
        write_group(out_dir / f"{name}.npz",
                    {m: (e["estimate"], e["draws"], e["method"], e["c"], {})
                     for m, e in res["members"].items()}, meta)
    record = {**meta,
              "estimates": {m: e["estimate"] for m, e in res["members"].items()},
              "oracle": res["oracle"], "panels": res["panels"],
              "fit": {"best_epoch": summary["best"]["epoch"],
                      "last_epoch": summary["last_epoch"],
                      "dead_w_channel": summary["dead_w_channel"],
                      "train_minutes": summary["minutes"],
                      "run_dir": str(trainer.run_dir),
                      "peak_gpu_mib": (torch.cuda.max_memory_allocated() / 2 ** 20
                                       if torch.cuda.is_available() else None)},
              "config": json.loads((trainer.run_dir / "config.json").read_text()),
              "wall_minutes": (time.time() - started) / 60}
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1,
                                                     default=float))
    log.info("%s: %d panels, %s (%.1f min)", name, res["n_panels"],
             record["estimates"], record["wall_minutes"])
    return record


# -- aggregation ---------------------------------------------------------------

def predicted_kappa_star(kappa_true: float, grid=None) -> float | None:
    """The first grid kappa >= kappa_true (the devlog's prediction)."""
    from discell.experiments.breakdown import GRID

    grid = GRID if grid is None else grid
    return next((float(k) for k in grid if k >= kappa_true - 1e-12), None)


def collect(fits_dir: Path, kappa_true: float, grid, seeds) -> dict:
    from discell.experiments.breakdown import read_draws

    out = {}
    for kappa in grid:
        out[kappa] = {}
        for seed in seeds:
            p = fits_dir / f"{run_name(kappa_true, kappa, seed)}.npz"
            out[kappa][seed] = read_draws(p) if p.exists() else {}
    return out


def verdict(table: dict, kappa_true: float) -> dict:
    spill = table["members"].get("spill_cf_minus_leak")
    resp = table["members"].get("response_cf_minus_leak")
    want = predicted_kappa_star(kappa_true)
    out = {"predicted_spill_kappa_star": want}
    if spill is not None:
        out["spill_status"] = spill["status"]
        out["spill_kappa_star"] = spill["kappa_star"]
        if kappa_true > 0:
            out["spill_tracks_kappa_true"] = bool(
                spill["status"] == "breaks"
                and np.isclose(spill["kappa_star"], want))
        else:     # the null world: nothing planted, nothing to find
            out["spill_null_no_finding"] = spill["status"] == "no finding"
    if resp is not None:
        out["response_status"] = resp["status"]
        out["response_kappa_star"] = resp["kappa_star"]
        out["response_survives_grid"] = resp["status"] == "above the grid"
    return out


def aggregate(fits_dir: Path = OUT / "fits", out_stem: Path = RESULT) -> dict:
    from discell.experiments.breakdown import GRID, section_table

    m = len(MEMBERS)
    worlds = {}
    for kt in KAPPA_TRUE:
        collected = collect(fits_dir, kt, GRID, SEEDS)
        n_fits = sum(bool(v) for k in collected.values() for v in k.values())
        if not n_fits:
            continue
        table = section_table(f"planted_k{kt:g}", collected, m_s=m, grid=GRID,
                              seeds=SEEDS)
        records = [json.loads(p.read_text()) for p in
                   sorted(fits_dir.glob(f"spill_p{kt:g}_a*_m*.json"))]
        worlds[f"{kt:g}"] = {
            "kappa_true": kt, "n_fits": n_fits, "table": table,
            "verdict": verdict(table, kt),
            "oracle": records[0]["oracle"] if records else None,
            "fits": [{k: r[k] for k in ("run", "kappa", "seed", "n_panels",
                                        "estimates")}
                     | {"best_epoch": r["fit"]["best_epoch"],
                        "dead_w_channel": r["fit"]["dead_w_channel"]}
                     for r in records]}
    result = {"spec": "devlog 'RECOMB story map approved; three additions "
                      "(motivation, 2026-10-01; author)', item 1",
              "world": {"n_cells": N_CELLS, "box_um": BOX_UM,
                        "n_types": SR.N_TYPES, "n_genes": 60,
                        "n_spill_genes": N_SPILL, "world_seed": WORLD_SEED,
                        "spill_genes": np.flatnonzero(spill_genes()).tolist(),
                        "niches": N_NICHES, "min_model": MIN_MODEL,
                        "min_scored": MIN_SCORED, "n_draws": N_DRAWS},
              "family_size": m, "level": 1 - 0.05 / m, "grid": list(GRID),
              "seeds": list(SEEDS), "worlds": worlds}
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".json").write_text(json.dumps(result, indent=1,
                                                        default=float))
    out_stem.with_suffix(".md").write_text(render(result))
    return result


def _cell(e) -> str:
    if e is None:
        return "–"
    lo, hi = e["ci"]
    return (f"{e['estimate']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
            f"(seeds {', '.join(f'{v:+.2f}' for v in e['per_seed'])})")


def render(result: dict) -> str:
    from discell.experiments.breakdown import _kstar

    w = result["world"]
    lines = [
        "# Planted spill-over positive control", "",
        f"World: simulate(N = {w['n_cells']:,}, K = {w['n_types']}, G = "
        f"{w['n_genes']}, box {w['box_um']:.0f} um, world seed "
        f"{w['world_seed']}); {w['n_spill_genes']} genes with B = 0 (spill "
        "only), the rest with the planted response. Fit: the final "
        "configuration at every assumed κ, model seeds "
        f"{', '.join(map(str, result['seeds']))}. Contrast: transport "
        "counterfactual − leak-only (mean R² over all panels) on each gene "
        f"set. Interval: 3 seeds pooled, half-tile subsampling, two-sided "
        f"{result['level']:.3f} (Bonferroni, m = {result['family_size']}). "
        "κ* = first grid κ where the contrast stops holding (breakdown rule).",
        "", "Prediction (devlog, fixed in advance): the spill-only contrast's "
        "κ* is the first grid point ≥ κ_true; the genuine response contrast "
        "survives the grid. κ_true = 0 is a null world (no spill to find).",
        "", "| κ_true | contrast | κ* | predicted | as predicted |",
        "|---|---|---|---|---|"]
    for kt, wd in result["worlds"].items():
        v = wd["verdict"]
        for member in MEMBERS:
            entry = wd["table"]["members"].get(member)
            if entry is None:
                continue
            if member.startswith("spill"):
                pred = ("no finding" if float(kt) == 0 else
                        f"breaks at {v['predicted_spill_kappa_star']:g}")
                ok = (v.get("spill_null_no_finding") if float(kt) == 0
                      else v.get("spill_tracks_kappa_true"))
            else:
                pred, ok = "above the grid", v.get("response_survives_grid")
            lines.append(f"| {kt} | {member} | {_kstar(entry)} | {pred} | "
                         f"{'yes' if ok else 'no'} |")
    for kt, wd in result["worlds"].items():
        lines += ["", f"## κ_true = {kt} ({wd['n_fits']} fits)", "",
                  "| contrast | " + " | ".join(f"κ = {k:g}" for k in
                                                 result["grid"]) + " |",
                  "|---" * (len(result["grid"]) + 1) + "|"]
        for member in MEMBERS:
            m = wd["table"]["members"].get(member)
            if m is None:
                continue
            lines.append(f"| {member} | " + " | ".join(
                _cell(m["trajectory"].get(f"{k:g}")) for k in result["grid"])
                + " |")
        o = wd.get("oracle") or {}
        if o:
            lines += ["", "World check (the planted structure on the same "
                      "panels, model seed 0's niches): mean R² of the TRUE "
                      "shift against the observed one, and its spread:"]
            for member, v in o.items():
                if v.get("mean_r2_true_shift") is not None:
                    lines.append(f"- {MEMBERS[member]} genes: R² "
                                 f"{v['mean_r2_true_shift']:.3f}, sd "
                                 f"{v['mean_sd_true_shift']:.3f} "
                                 f"({v['n_panels']} panels)")
    return "\n".join(lines) + "\n"


# -- CLI -----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--kappa-true", type=float, required=True)
    f.add_argument("--assumed", type=float, required=True)
    f.add_argument("--model-seed", type=int, required=True)
    f.add_argument("--n", type=int, default=N_DRAWS)
    f.add_argument("--epochs", type=int, default=None, help="smoke only")
    f.add_argument("--patience", type=int, default=None, help="smoke only")
    f.add_argument("--device", default="cuda")
    f.add_argument("--out", type=Path, default=OUT / "fits")
    a = sub.add_parser("aggregate")
    a.add_argument("--fits", type=Path, default=OUT / "fits")
    a.add_argument("--out", type=Path, default=RESULT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.cmd == "fit":
        fit_one(args.kappa_true, args.assumed, args.model_seed, args.n,
                args.epochs, args.patience, args.device, args.out)
    else:
        aggregate(args.fits, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
