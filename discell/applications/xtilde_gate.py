#!/usr/bin/env python3
"""The x~ gate: a powered planted world for the amortisation gap (doc 11 A4).

**How.** Gate scaffold (6 000 cells, 8 types, kappa = 0.2). A programme is
planted in 30% of the cells of two "cycling" types: those cells devote a
share ``s`` of their transcripts to 12 programme genes drawn from the low-
but-expressed range. Re-mixed through the true leak operator, counts
resampled. *Victims* are non-cycling cells whose beta-weighted exposure to
planted neighbours is >= 0.2; *controls* are non-cycling cells with no
planted neighbour. Five scores are read in the victims and in the cycling
types: (a) the raw programme score; (b) a z-probe from a model trained on
raw counts; (c) the same probe from a model whose encoders read
``x~ = x - kappa l rho_bar`` (``subtract_leak``, same torch seed);
(d) the programme score on ``x~`` built from (b)'s own ``rho_bar`` -- the
counts-level correction the model licenses without touching the encoder;
(e) the same with the true ``rho_bar`` (ceiling of the counts-level route).

**Evaluated by** (issues V9 fixes). Thresholds are set *within type* at the
(1 - 0.3) quantile of that type's controls, so the synthetic type offsets
that swamped the old global threshold cancel; the leak-attributable
statistic is the *excess* FPR, victims minus controls, pooled cell-weighted
over types with >= 30 cells in both groups, with a stratified 500-draw
bootstrap shared across arms (paired differences). A *power gate* on the
raw arm -- AUROC(planted | raw score) >= 0.9 inside the cycling types and
raw excess >= 0.1 -- says whether the world can adjudicate at all; it is
model-free and is checked before any fit. *Beating* another arm is
pre-registered as: excess lower by >= 0.05, the paired bootstrap CI of the
difference above 0, and AUROC within 0.02 of the arm it is compared with.
Per seed: H1 (x~ in the encoder delivers per-cell decontamination) = (c)
beats (b); H2 (counts-level correction) = (d) beats (a); each caller is
also read against raw. A hypothesis is supported over seeds when
>= 2 of 3 pass and none reverses. The amortisation-gap reading itself is
excess(b) / excess(a): "z inherits the leak" when (b)'s CI is above 0 and
the ratio >= 0.5.

**Wished for.** A powered world (gate passes), and a clear answer to
whether the per-cell benefit x~ was built for exists: H1 pass -> x~ earns
its place for per-cell applications; H1 fail and H2 pass -> the
counts-level route is the per-cell fix and the encoder change can be
dropped; both fail -> neither route removes leak-induced per-cell false
positives and the doc-11 per-cell claims must be re-scoped.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m discell.applications.xtilde_gate --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from discell import paths

log = logging.getLogger("discell.applications.xtilde_gate")

CYCLING_TYPES = (0, 1)
PLANT_RATE = 0.3          # fraction of cycling cells planted
N_GENES = 12
SHARE = 0.25              # transcript share of the programme in planted cells
GENE_POOL = (0.2, 0.5)    # programme genes: quantile band of global mean rate
KAPPA = 0.2
EXPOSURE_CUT = 0.2        # victims: beta mass from planted neighbours >= this
NOMINAL_RATE = 0.3        # within-type threshold: this FPR in the controls
MIN_CELLS = 30
POWER_AUROC = 0.9
POWER_EXCESS = 0.1
AUROC_MARGIN = 0.02
EXCESS_MARGIN = 0.05       # a caller beats another only if excess is this much lower
N_BOOT = 500
ARMS = ("raw", "z_raw", "z_xtilde", "xtilde_counts", "oracle_counts")
# -- the 2026-09-21 follow-up -------------------------------------------------
TRUE_VICTIM_TYPES = (2, 3)  # arm (c): non-cycling types that get real plants
TRUE_VICTIM_RATE = 0.1      # fraction of their cells, all at exposure 0
RECALL_MARGIN = 0.03        # arm (c): recall must stay within this of plain z
SEARCH_AUROC = 0.92         # arm (b): weaken the plant until raw AUROC is below
MIN_PASSES = 8              # arm (a)+(b): 8 of 12 seeds on the margin


# -- read-out --------------------------------------------------------------

def within_type_threshold(score: np.ndarray, t: np.ndarray, controls: np.ndarray,
                          rate: float = NOMINAL_RATE) -> dict[int, float]:
    """Per type, the score quantile that calls *rate* of that type's controls."""
    return {int(g): float(np.quantile(score[controls & (t == g)], 1.0 - rate))
            for g in np.unique(t[controls])}


def excess_fpr(score: np.ndarray, t: np.ndarray, victims: np.ndarray,
               controls: np.ndarray, rate: float = NOMINAL_RATE,
               min_cells: int = MIN_CELLS) -> dict:
    """Victim FPR minus control FPR at within-type thresholds, pooled over
    types with >= *min_cells* victims and controls, weighted by victims."""
    thresholds = within_type_threshold(score, t, controls, rate)
    per_type, num, n_v, n_c = {}, 0.0, 0, 0
    for g, thr in thresholds.items():
        v, c = victims & (t == g), controls & (t == g)
        if v.sum() < min_cells or c.sum() < min_cells:
            continue
        fpr_v, fpr_c = float((score[v] > thr).mean()), float((score[c] > thr).mean())
        per_type[g] = {"n_victims": int(v.sum()), "n_controls": int(c.sum()),
                       "victim_fpr": fpr_v, "control_fpr": fpr_c,
                       "excess": fpr_v - fpr_c}
        num += v.sum() * (fpr_v - fpr_c)
        n_v += int(v.sum())
        n_c += int(c.sum())
    if not n_v:
        return {"excess": float("nan"), "victim_fpr": float("nan"),
                "control_fpr": float("nan"), "n_victims": 0, "n_controls": 0,
                "per_type": {}}
    weights = np.array([r["n_victims"] for r in per_type.values()], dtype=float)
    return {"excess": num / n_v,
            "victim_fpr": float(np.average([r["victim_fpr"] for r in per_type.values()], weights=weights)),
            "control_fpr": float(np.average([r["control_fpr"] for r in per_type.values()], weights=weights)),
            "n_victims": n_v, "n_controls": n_c, "per_type": per_type}


def recall_at_threshold(score: np.ndarray, t: np.ndarray, true_victims: np.ndarray,
                        controls: np.ndarray, rate: float = NOMINAL_RATE,
                        min_cells: int = MIN_CELLS) -> dict:
    """Arm (c): the fraction of *genuinely* planted cells called, at the same
    within-type thresholds the FPRs use. Pooled over types with >= *min_cells*
    planted cells and controls, weighted by planted cells. This is the
    sensitivity half the saturated world could not test: a caller that removes
    leak by blunting the programme signal loses here."""
    thresholds = within_type_threshold(score, t, controls, rate)
    per_type, num, n = {}, 0.0, 0
    for g, thr in thresholds.items():
        v, c = true_victims & (t == g), controls & (t == g)
        if v.sum() < min_cells or c.sum() < min_cells:
            continue
        r = float((score[v] > thr).mean())
        per_type[g] = {"n_true_victims": int(v.sum()), "recall": r}
        num += v.sum() * r
        n += int(v.sum())
    if not n:
        return {"recall": float("nan"), "n_true_victims": 0, "per_type": {}}
    return {"recall": num / n, "n_true_victims": n, "per_type": per_type}


def bootstrap_excess(scores: dict[str, np.ndarray], t: np.ndarray,
                     victims: np.ndarray, controls: np.ndarray,
                     rng: np.random.Generator, n_boot: int = N_BOOT,
                     rate: float = NOMINAL_RATE, min_cells: int = MIN_CELLS
                     ) -> dict[str, np.ndarray]:
    """Pooled excess per arm over *n_boot* stratified resamples: cells drawn
    with replacement within each type's victims and controls, the same draw
    for every arm, so differences between arms are paired."""
    groups = [np.flatnonzero(m & (t == g)) for g in np.unique(t) for m in (victims, controls)]
    groups = [g for g in groups if len(g)]
    draws = {name: np.zeros(n_boot) for name in scores}
    for b in range(n_boot):
        rows = np.concatenate([rng.choice(g, len(g)) for g in groups])
        t_b, v_b, c_b = t[rows], victims[rows], controls[rows]
        for name, score in scores.items():
            draws[name][b] = excess_fpr(score[rows], t_b, v_b, c_b, rate, min_cells)["excess"]
    return draws


def power_gate(auroc_raw: float, excess_raw: float, min_auroc: float = POWER_AUROC,
               min_excess: float = POWER_EXCESS) -> dict:
    """The world adjudicates only if the raw score sees the plant in the
    planted cells (AUROC) and the leak in the victims (excess)."""
    return {"auroc_raw": float(auroc_raw), "excess_raw": float(excess_raw),
            "min_auroc": min_auroc, "min_excess": min_excess,
            "passed": bool(auroc_raw >= min_auroc and excess_raw >= min_excess)}


def ci(draws: np.ndarray) -> list[float]:
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


# -- scores ----------------------------------------------------------------

def programme_score(x: np.ndarray, genes: np.ndarray, median: float | None = None) -> np.ndarray:
    """Mean log1p library-normalised expression over the programme genes."""
    totals = x.sum(axis=1)
    median = float(np.median(totals)) if median is None else median
    xn = np.log1p(x[:, genes] / np.clip(totals, 1, None)[:, None] * median)
    return xn.mean(axis=1)


def leak_subtracted_counts(x: np.ndarray, rho_bar: np.ndarray, kappa: float) -> np.ndarray:
    """``x~ = clip(x - kappa l rho_bar, 0)`` -- the counts-level correction."""
    return np.clip(x - kappa * x.sum(axis=1, keepdims=True) * rho_bar, 0.0, None)


def z_probe(mu_z: np.ndarray, t: np.ndarray, target: np.ndarray,
            cycling: np.ndarray, fold: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Cross-fitted ridge from type-centred ``mu_z`` to *target* inside the
    cycling types (predictions out of fold); every other cell is projected
    with the pooled fit. The probe never sees the planted truth."""
    z = mu_z.astype(float).copy()
    y = target.astype(float).copy()
    for g in np.unique(t):
        z[t == g] -= z[t == g].mean(axis=0)
        if cycling[t == g].any():
            y[t == g] -= y[t == g].mean()

    def solve(rows):
        gram = z[rows].T @ z[rows] + lam * np.eye(z.shape[1])
        return np.linalg.solve(gram, z[rows].T @ y[rows])

    score = np.zeros(len(t))
    cyc = np.flatnonzero(cycling)
    for f in np.unique(fold[cyc]):
        train, test = cyc[fold[cyc] != f], cyc[fold[cyc] == f]
        score[test] = z[test] @ solve(train)
    score[~cycling] = z[~cycling] @ solve(cyc)
    return score


# -- the world -------------------------------------------------------------

def gene_band(mean_rate: np.ndarray, n_genes: int,
              gene_pool: tuple[float, float] = GENE_POOL) -> np.ndarray:
    """Gene indices in the *gene_pool* quantile band of ``mean_rate``, by rank.

    Widened symmetrically until it holds ``n_genes`` -- the pre-registered
    "more genes" strengthening must not be blocked by a narrow band.
    """
    order = np.argsort(mean_rate, kind="stable")
    lo, hi = int(gene_pool[0] * len(order)), int(gene_pool[1] * len(order))
    while hi - lo < n_genes:
        lo, hi = max(0, lo - 1), min(len(order), hi + 1)
    return order[lo:hi]



@dataclass
class World:
    sim: object
    planted: np.ndarray
    genes: np.ndarray
    cycling: np.ndarray
    exposure: np.ndarray
    victims: np.ndarray
    controls: np.ndarray
    true_victims: np.ndarray


def build_world(seed: int, share: float = SHARE, n_genes: int = N_GENES,
                gene_pool: tuple[float, float] = GENE_POOL, kappa: float = KAPPA,
                true_victim_rate: float = 0.0,
                true_victim_types: tuple[int, ...] = TRUE_VICTIM_TYPES) -> World:
    """The gate scaffold, optionally with *genuinely* cycling victims (arm c).

    With ``true_victim_rate > 0`` that fraction of the cells of
    *true_victim_types* -- non-cycling types, and only cells with **no planted
    cycling neighbour** (exposure 0) -- carry the same programme. They are a
    clean sensitivity set: a caller should call them, and it is a real loss if
    the leak correction stops calling them. They leak onward like any planted
    cell, so the controls are cells with zero exposure to *any* planted cell.
    The extra draw is taken only when the rate is positive, so the arm-(a)
    worlds are bit-identical to the three-seed run of 2026-09-21.
    """
    from discell.applications.planted import plant_programme
    from discell.model.synthetic import simulate

    rng = np.random.default_rng(seed)
    sim = simulate(n_cells=6000, n_types=8, kappa=kappa, seed=seed)
    cycling = np.isin(sim.t, CYCLING_TYPES)
    planted_cycling = cycling & (rng.random(len(sim.t)) < PLANT_RATE)
    exposure = np.asarray(sim.graph.in_edges @ planted_cycling.astype(float)).ravel()
    true_victims = np.zeros(len(sim.t), dtype=bool)
    if true_victim_rate > 0:
        eligible = ~cycling & np.isin(sim.t, true_victim_types) & (exposure == 0)
        true_victims = eligible & (rng.random(len(sim.t)) < true_victim_rate)
    planted = planted_cycling | true_victims
    mean_rate = sim.rho_true.mean(axis=0)
    genes = rng.choice(gene_band(mean_rate, n_genes, gene_pool), n_genes, replace=False)
    plant_programme(sim, planted, genes, share, rng)
    total = np.asarray(sim.graph.in_edges @ planted.astype(float)).ravel()
    return World(sim=sim, planted=planted, genes=genes, cycling=cycling, exposure=exposure,
                 victims=~cycling & ~planted & (exposure >= EXPOSURE_CUT),
                 controls=~cycling & ~planted & (total == 0),
                 true_victims=true_victims)


def search_share(seed: int = 0, start: float = SHARE, target_auroc: float = SEARCH_AUROC,
                 n_genes: int = N_GENES, floor: float = 1e-3, refine: int = 6) -> dict:
    """Arm (b): weaken the plant until the raw caller stops saturating.

    Pre-registered as "halve the share until raw AUROC < *target_auroc*, then
    verify the power gate". Halving is coarse, so when the halved share also
    drops the gate (raw AUROC < 0.9 or raw excess < 0.1) the interval between
    the last saturated share and the first unsaturated one is bisected for the
    **largest** share still below *target_auroc* -- the point of the arm is an
    AUROC the rule can bind on, not a broken world. Model-free: no fit is
    spent on the search."""
    from sklearn.metrics import roc_auc_score

    def probe(share: float) -> dict:
        world = build_world(seed, share=share, n_genes=n_genes)
        raw = programme_score(world.sim.x, world.genes, float(np.median(world.sim.totals)))
        auroc = float(roc_auc_score(world.planted[world.cycling], raw[world.cycling]))
        excess = excess_fpr(raw, world.sim.t, world.victims, world.controls)["excess"]
        row = power_gate(auroc, excess)
        row["share"] = float(share)
        log.info("share search: %s", json.dumps(row))
        return row

    trace, share, hi = [], float(start), None
    while share > floor:
        row = probe(share)
        trace.append(row)
        if row["auroc_raw"] < target_auroc:
            break
        hi, share = share, share / 2.0
    chosen = trace[-1]
    if hi is not None and not chosen["passed"]:
        lo = chosen["share"]
        best = None
        for _ in range(refine):
            mid = 0.5 * (lo + hi)
            row = probe(mid)
            trace.append(row)
            if row["auroc_raw"] < target_auroc:
                best, lo = row, mid          # still unsaturated: push the share up
            else:
                hi = mid
        chosen = best if best is not None else chosen
    return {"share": chosen["share"], "target_auroc": target_auroc,
            "chosen": chosen, "trace": trace, "seed": seed}


def read_out(world: World, scores: dict[str, np.ndarray], rng: np.random.Generator,
             n_boot: int = N_BOOT) -> dict:
    """Every arm against the pre-registered rules; *scores* must hold 'raw'."""
    from sklearn.metrics import roc_auc_score

    sim, cyc = world.sim, world.cycling
    arms, draws = {}, bootstrap_excess(scores, sim.t, world.victims, world.controls, rng, n_boot)
    for name, score in scores.items():
        arm = excess_fpr(score, sim.t, world.victims, world.controls)
        arm["excess_ci"] = ci(draws[name])
        arm["auroc_cycling"] = float(roc_auc_score(world.planted[cyc], score[cyc]))
        arms[name] = arm
    if world.true_victims.any():
        for name, score in scores.items():
            arms[name]["recall"] = recall_at_threshold(score, sim.t, world.true_victims,
                                                       world.controls)
    out = {"n_planted": int(world.planted.sum()), "n_victims": int(world.victims.sum()),
           "n_controls": int(world.controls.sum()),
           "n_true_victims": int(world.true_victims.sum()), "genes": world.genes.tolist(),
           "power": power_gate(arms["raw"]["auroc_cycling"], arms["raw"]["excess"]),
           "arms": arms, "tests": {}}

    def beats(better, worse, margin=EXCESS_MARGIN):
        """*better* beats *worse*: excess lower by >= *margin*, the paired CI
        above 0, and AUROC inside AUROC_MARGIN of *worse* (pre-registered)."""
        diff = draws[worse] - draws[better]
        delta = float(arms[worse]["excess"] - arms[better]["excess"])
        auroc_kept = bool(arms[better]["auroc_cycling"] >= arms[worse]["auroc_cycling"] - AUROC_MARGIN)
        return {"excess_difference": delta,
                "difference_ci": ci(diff),
                "margin": margin,
                "auroc_kept": auroc_kept,
                "passed": bool(delta >= margin and np.percentile(diff, 2.5) > 0 and auroc_kept),
                "reversed": bool(np.percentile(diff, 97.5) < 0)}

    if "z_raw" in arms:
        out["tests"]["z_raw_below_raw"] = beats("z_raw", "raw")
        ratio = arms["z_raw"]["excess"] / arms["raw"]["excess"] if arms["raw"]["excess"] > 0 else float("nan")
        out["tests"]["amortisation_gap"] = {
            "excess_ratio_z_over_raw": float(ratio),
            "z_inherits_leak": bool(arms["z_raw"]["excess_ci"][0] > 0 and ratio >= 0.5)}
    if "z_xtilde" in arms:
        out["tests"]["H1_xtilde_encoder"] = beats("z_xtilde", "z_raw")
        out["tests"]["z_xtilde_below_raw"] = beats("z_xtilde", "raw")
    if "xtilde_counts" in arms:
        out["tests"]["H2_counts_correction"] = beats("xtilde_counts", "raw")
    if "oracle_counts" in arms:
        out["tests"]["ceiling_oracle_counts"] = beats("oracle_counts", "raw")
    if world.true_victims.any() and "z_xtilde" in arms and "z_raw" in arms:
        delta = float(arms["z_xtilde"]["recall"]["recall"] - arms["z_raw"]["recall"]["recall"])
        out["tests"]["recall_z_xtilde_vs_z_raw"] = {
            "recall_z_raw": arms["z_raw"]["recall"]["recall"],
            "recall_z_xtilde": arms["z_xtilde"]["recall"]["recall"],
            "delta": delta, "margin": RECALL_MARGIN,
            "within_margin": bool(delta >= -RECALL_MARGIN)}
    return out


def run_seed(seed: int, device: str = "cuda", epochs: int = 400, share: float = SHARE,
             n_genes: int = N_GENES, fit_xtilde: bool = True, n_boot: int = N_BOOT,
             fit_unpowered: bool = False, true_victim_rate: float = 0.0) -> dict:
    """Build the world, fit the model(s), score every arm, read out."""
    from discell.applications.planted import fit_synthetic

    from sklearn.metrics import roc_auc_score

    world = build_world(seed, share=share, n_genes=n_genes,
                        true_victim_rate=true_victim_rate)
    sim = world.sim
    median = float(np.median(sim.totals))
    raw = programme_score(sim.x, world.genes, median)
    # The power gate is model-free, so it is computed and logged *before* the
    # fits: an unpowered world must not be spent on GPU, let alone adjudicate.
    gate = power_gate(roc_auc_score(world.planted[world.cycling], raw[world.cycling]),
                      excess_fpr(raw, sim.t, world.victims, world.controls)["excess"])
    log.info("seed %d: %d planted, %d victims, %d controls; power gate (pre-fit) %s",
             seed, world.planted.sum(), world.victims.sum(), world.controls.sum(),
             json.dumps(gate))
    if not gate["passed"] and not fit_unpowered:
        return {"seed": seed, "share": share, "n_genes": n_genes, "kappa": sim.kappa,
                "power": gate, "arms": {}, "tests": {},
                "true_victim_rate": true_victim_rate,
                "skipped": "power gate failed before the fits"}
    fit = fit_synthetic(sim, epochs=epochs, device=device, seed=seed, subtract_leak=False)
    scores = {"raw": raw,
              "z_raw": z_probe(fit["z"], sim.t, raw, world.cycling, fit["fold"]),
              "xtilde_counts": programme_score(leak_subtracted_counts(sim.x, fit["rho_bar"], sim.kappa), world.genes, median),
              "oracle_counts": programme_score(leak_subtracted_counts(
                  sim.x, np.asarray(sim.graph.in_edges @ sim.rho_true), sim.kappa), world.genes, median)}
    if fit_xtilde:
        fit_x = fit_synthetic(sim, epochs=epochs, device=device, seed=seed, subtract_leak=True)
        scores["z_xtilde"] = z_probe(fit_x["z"], sim.t, raw, world.cycling, fit_x["fold"])
    out = read_out(world, scores, np.random.default_rng(seed + 1000), n_boot=n_boot)
    out.update(seed=seed, share=share, n_genes=n_genes, kappa=sim.kappa, epochs=epochs,
               true_victim_rate=true_victim_rate)
    for name, arm in out["arms"].items():
        log.info("seed %d %-14s excess %.3f [%.3f, %.3f]  victim/control FPR %.3f/%.3f  AUROC(cycling) %.3f",
                 seed, name, arm["excess"], *arm["excess_ci"], arm["victim_fpr"], arm["control_fpr"], arm["auroc_cycling"])
    if world.true_victims.any():
        for name, arm in out["arms"].items():
            log.info("seed %d %-14s recall on %d genuinely planted cells %.3f",
                     seed, name, arm["recall"]["n_true_victims"], arm["recall"]["recall"])
    for name, test in out["tests"].items():
        log.info("seed %d %s: %s", seed, name, json.dumps(test))
    return out


def verdict(seeds: list[dict]) -> dict:
    """Over seeds: supported if >= 2/3 pass and no seed reverses; power counted separately."""
    out = {"n_seeds": len(seeds), "power_passed": sum(s["power"]["passed"] for s in seeds)}
    for key in ("H1_xtilde_encoder", "H2_counts_correction", "z_raw_below_raw",
                "z_xtilde_below_raw", "ceiling_oracle_counts"):
        tests = [s["tests"][key] for s in seeds if key in s["tests"]]
        if not tests:
            continue
        passed, reversed_ = sum(t["passed"] for t in tests), sum(t["reversed"] for t in tests)
        out[key] = {"passed": passed, "reversed": reversed_,
                    "supported": bool(passed >= 2 and reversed_ == 0 and len(tests) >= 2)}
    gaps = [s["tests"]["amortisation_gap"] for s in seeds if "amortisation_gap" in s["tests"]]
    if gaps:
        out["amortisation_gap"] = {"z_inherits_leak_seeds": sum(g["z_inherits_leak"] for g in gaps),
                                   "excess_ratio_mean": float(np.mean([g["excess_ratio_z_over_raw"] for g in gaps]))}
    return out


def follow_up_verdict(seeds_ab: list[dict], seeds_c: list[dict],
                      min_passes: int = MIN_PASSES,
                      recall_margin: float = RECALL_MARGIN) -> dict:
    """The 2026-09-21 rule, verbatim: default-on if, over the 12 seeds of
    (a)+(b), x~-z lowers excess FPR vs plain z by >= 0.05 with the paired CI
    above zero on >= *min_passes* of them with no reversal, AUROC within 0.02
    of plain z on every seed, and in (c) recall on genuinely cycling victims
    within *recall_margin* of plain z. Otherwise option-only."""
    tests = [s["tests"]["H1_xtilde_encoder"] for s in seeds_ab
             if "H1_xtilde_encoder" in s.get("tests", {})]
    recalls = [s["tests"]["recall_z_xtilde_vs_z_raw"] for s in seeds_c
               if "recall_z_xtilde_vs_z_raw" in s.get("tests", {})]
    passed = sum(t["passed"] for t in tests)
    reversed_ = sum(t["reversed"] for t in tests)
    auroc_kept = bool(tests) and all(t["auroc_kept"] for t in tests)
    recall_kept = bool(recalls) and all(r["within_margin"] for r in recalls)
    default_on = bool(len(tests) >= min_passes and passed >= min_passes
                      and reversed_ == 0 and auroc_kept and recall_kept)
    return {"n_seeds_ab": len(tests), "min_passes": min_passes,
            "h1_passed": passed, "h1_reversed": reversed_,
            "auroc_within_margin_every_seed": auroc_kept,
            "n_seeds_c": len(recalls), "recall_within_margin_every_seed": recall_kept,
            "recall_deltas": [r["delta"] for r in recalls],
            "default_on": default_on,
            "verdict": "default-on" if default_on else "option-only"}


def summary_figure(arms: dict[str, dict], path) -> None:
    """One figure for the follow-up: excess per caller per seed in (a) and (b),
    and recall on the genuinely planted cells in (c)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ("a", "b", "c") if k in arms]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        seeds = [s for s in arms[key]["seeds"] if s.get("arms")]
        names = [a for a in ARMS if seeds and all(a in s["arms"] for s in seeds)]
        width = 0.8 / max(len(seeds), 1)
        recall = key == "c"
        for k, s in enumerate(seeds):
            xs = np.arange(len(names)) + (k - (len(seeds) - 1) / 2) * width
            if recall:
                ys = [s["arms"][a]["recall"]["recall"] for a in names]
                ax.bar(xs, ys, width, label=f"seed {s['seed']}")
            else:
                ys = [s["arms"][a]["excess"] for a in names]
                lo = [y - s["arms"][a]["excess_ci"][0] for y, a in zip(ys, names)]
                hi = [s["arms"][a]["excess_ci"][1] - y for y, a in zip(ys, names)]
                ax.bar(xs, ys, width, yerr=[lo, hi], capsize=2, label=f"seed {s['seed']}")
        share = seeds[0]["share"] if seeds else float("nan")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=7)
        ax.set_title({"a": f"(a) 25 % plant: excess FPR (share {share:.3f})",
                      "b": f"(b) weakened plant: excess FPR (share {share:.3f})",
                      "c": "(c) recall on genuinely planted cells"}[key], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure(seeds: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [s for s in seeds if s["arms"]]
    arms = [a for a in ARMS if seeds and all(a in s["arms"] for s in seeds)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / max(len(seeds), 1)
    for k, s in enumerate(seeds):
        xs = np.arange(len(arms)) + (k - (len(seeds) - 1) / 2) * width
        ex = [s["arms"][a]["excess"] for a in arms]
        lo = [s["arms"][a]["excess"] - s["arms"][a]["excess_ci"][0] for a in arms]
        hi = [s["arms"][a]["excess_ci"][1] - s["arms"][a]["excess"] for a in arms]
        axes[0].bar(xs, ex, width, yerr=[lo, hi], capsize=2, label=f"seed {s['seed']}")
        axes[1].plot(np.arange(len(arms)), [s["arms"][a]["auroc_cycling"] for a in arms], "o-", label=f"seed {s['seed']}")
    for ax, title in ((axes[0], "excess FPR in victims (victim - control, within-type thresholds)"),
                      (axes[1], "AUROC(planted | score) inside the cycling types")):
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=20, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axhline(POWER_EXCESS, color="gray", ls="--", lw=0.8)
    axes[1].axhline(POWER_AUROC, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--share", type=float, default=SHARE)
    parser.add_argument("--n-genes", type=int, default=N_GENES)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--true-victim-rate", type=float, default=0.0,
                        help="arm (c): fraction of the non-cycling victim types genuinely planted")
    parser.add_argument("--search-share", action="store_true",
                        help="arm (b): halve the plant share on the first seed until raw AUROC < 0.92")
    parser.add_argument("--summary", nargs="*", default=None,
                        help="arm JSONs (a b c) -> one summary figure and the follow-up verdict")
    parser.add_argument("--fit-unpowered", action="store_true",
                        help="fit even when the pre-fit power gate fails (reported as such)")
    parser.add_argument("--out", default=None, help="JSON path; default data/experiments_synthetic/xtilde_gate.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out_path = paths.DATA_ROOT / "experiments_synthetic" / "xtilde_gate.json" if args.out is None else paths.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.summary is not None:
        arms = {key: json.loads(paths.Path(p).read_text())
                for key, p in zip(("a", "b", "c"), args.summary)}
        combined = follow_up_verdict(arms.get("a", {}).get("seeds", []) + arms.get("b", {}).get("seeds", []),
                                     arms.get("c", {}).get("seeds", []))
        summary_figure(arms, out_path.with_suffix(".png"))
        out_path.write_text(json.dumps({"follow_up_verdict": combined,
                                        "arms": {k: v.get("verdict") for k, v in arms.items()}}, indent=2))
        log.info("FOLLOW-UP VERDICT %s", json.dumps(combined))
        log.info("written %s", out_path)
        return 0
    search = search_share(args.seeds[0], start=args.share, n_genes=args.n_genes) if args.search_share else None
    share = search["share"] if search else args.share
    seeds = [run_seed(s, device=args.device, epochs=args.epochs, share=share,
                      n_genes=args.n_genes, n_boot=args.n_boot,
                      fit_unpowered=args.fit_unpowered,
                      true_victim_rate=args.true_victim_rate) for s in args.seeds]
    result = {"design": {"share": share, "share_search": search,
                         "true_victim_rate": args.true_victim_rate,
                         "true_victim_types": list(TRUE_VICTIM_TYPES),
                         "n_genes": args.n_genes, "kappa": KAPPA,
                         "plant_rate": PLANT_RATE, "exposure_cut": EXPOSURE_CUT,
                         "nominal_rate": NOMINAL_RATE, "min_cells": MIN_CELLS,
                         "power_gate": {"auroc": POWER_AUROC, "excess": POWER_EXCESS},
                         "auroc_margin": AUROC_MARGIN, "n_boot": args.n_boot, "epochs": args.epochs,
                         "how_evaluated_wished_for": __doc__},
              "seeds": seeds, "verdict": verdict(seeds)}
    out_path.write_text(json.dumps(result, indent=2))
    figure(seeds, out_path.with_suffix(".png"))
    log.info("VERDICT %s", json.dumps(result["verdict"]))
    log.info("written %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
