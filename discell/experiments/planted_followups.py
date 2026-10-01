#!/usr/bin/env python3
"""Two follow-ups to the planted spill-over control (devlog "RECOMB
additions: results and follow-ups (2026-10-01; author decisions)").

Both re-read the EXISTING planted fits (``planted_spillover``'s 54 runs,
best checkpoints, CPU); nothing is refitted.

**(a) Null-world diagnostic.** The spill-only contrast of the control is
``R^2(cf) - R^2(leak)`` on the 20 no-response genes, with
``transport.score_shift``'s centred, zero-null R^2. Per panel, with every
vector centred over the panel's kept genes, it is exactly

    [2<o - l, p> - ||p||^2] / ||o||^2

(o observed shift, p programme, l leak-only). Splitting the observed shift
with the simulator's truth, ``o = c_pop + (c - c_pop) + s + e``: c the CLEAN
shift of the scored cells (their mean rho_true; no spill, no counts), c_pop
the same over every cell of the type in the two niches (the population's
niche effect), s the spill (mean p_true minus c), e the count noise.
:func:`decompose` returns the six terms ``2<c_pop,p>, 2<c - c_pop,p>,
2<s,p>, 2<e,p>, -2<l,p>, -||p||^2`` (each / ||o||^2); they sum to the
contrast. Under kappa_true = 0, s = 0, but the other terms are not zero:
the -||p||^2 term is never positive, c - c_pop (which cells were scored:
z varies within a type) is real expression variation, and c_pop can be
non-zero through the softmax closure (a B = 0 gene shares the cell's
normaliser). None of them is spill-over.

The **corrected** spill-only contrast keeps only the spill-attributable
cross term,

    spill_alignment = mean over panels of 2<p, o - c> / ||t||^2,

i.e. how much the programme lines up with the part of the observed shift
the clean truth does not explain, in units of the count-free true shift t
(mean p_true; t = c when nothing leaks). Under kappa_true = 0, o - c = e
is count noise on the scored cells, independent of the programme and of
the scale, so its expectation is zero for every assumed kappa (to first
order: log of a mean has a small Jensen bias). It needs the clean truth, so
it is a simulation-only control (not a real-data read). Draws: the same
half-tile subsamples as the original (the clean shift is recomputed on
every subsample's cells); kappa* by the same breakdown rule, family m = 2
(this member with the original response member).

**(b) Relocation against the clean truth.** On the control's panels
((niche pair, type), k-means niches, fold 0 scored), DISCELL's relocation
(counterfactual = programme + leak-only, and the programme alone, its
spill-free part) and the plain regression of
:mod:`discell.model.transport_regression` (``log`` and ``rate`` forms,
ridge on the simulated observed counts of the type's model-side cells) are
scored with ``transport.score_shift`` against (i) the observed held-out
shift, as on real data, and (ii) the true clean shift (scored cells' mean
rho_true). Ceilings: ``transport.noise_ceiling`` of each target (counts for
(i), rho_true for (ii)); fraction of ceiling = mean R^2 / mean ceiling.

Usage::

    python -m discell.experiments.planted_followups null
    python -m discell.experiments.planted_followups relocation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from discell.experiments import planted_spillover as PS
from discell.experiments import synthetic_recovery as SR

log = logging.getLogger("discell.experiments.planted_followups")

TERMS = ("niche", "sampling", "spill", "noise", "leak", "penalty")
MEMBER = "spill_alignment"
RESULT_DIR = PS.RESULT.parent
FOLLOWUP = PS.OUT.parent / "FOLLOWUP.md"


# -- pure pieces ---------------------------------------------------------------

def _c(v: np.ndarray) -> np.ndarray:
    return v - v.mean(axis=-1, keepdims=True)


def decompose(obs: np.ndarray, clean: np.ndarray, true: np.ndarray,
              prog: np.ndarray, leak: np.ndarray,
              clean_pop: np.ndarray | None = None) -> dict:
    """The original contrast's six terms on one panel's genes (module
    docstring); they sum to it. *clean* is the scored cells' clean shift,
    *clean_pop* the population's (every cell of the type in the two
    niches; None = *clean*), *true* the scored cells' contaminated shift
    (mean p_true). niche = clean_pop, sampling = clean - clean_pop (which
    scored cells were drawn), spill = true - clean, noise = obs - true."""
    clean_pop = clean if clean_pop is None else clean_pop
    o, c, t, cp = _c(obs), _c(clean), _c(true), _c(clean_pop)
    p, l = _c(prog), _c(leak)
    den = max(float(o @ o), 1e-12)
    return {"niche": 2 * float(cp @ p) / den,
            "sampling": 2 * float((c - cp) @ p) / den,
            "spill": 2 * float((t - c) @ p) / den,
            "noise": 2 * float((o - t) @ p) / den,
            "leak": -2 * float(l @ p) / den,
            "penalty": -float(p @ p) / den}


def alignment_rows(obs: np.ndarray, clean: np.ndarray, true: np.ndarray,
                   prog: np.ndarray) -> np.ndarray:
    """``2<p, o - c> / ||t||^2`` (all centred) for every row of *obs*,
    *clean* and *true* (draws x genes) against one programme. The scale is
    the count-free true shift t, not o: ||o||^2 carries the count noise, and
    a programme correlated with c would then bias the ratio through the
    2<c, e> part of the denominator (unit test)."""
    o, c, t, p = _c(obs), _c(clean), _c(true), _c(prog)
    den = np.maximum((t ** 2).sum(axis=1), 1e-12)
    return 2.0 * ((o - c) @ p) / den


# -- loading -----------------------------------------------------------------

_WORLDS: dict = {}


def world(kappa_true: float):
    if kappa_true not in _WORLDS:
        _WORLDS[kappa_true] = PS.world(kappa_true)
    return _WORLDS[kappa_true]


def load(kappa_true: float, assumed: float, seed: int):
    """``(sim, data, config, trainer)`` of one existing fit, on CPU."""
    from discell.model.validate import load_run

    sim = world(kappa_true)
    data = SR.model_data(sim, seed)
    config, data, trainer, _, _ = load_run(
        SR.DATASET, PS.run_name(kappa_true, assumed, seed), "cpu", data=data)
    return sim, data, config, trainer


def setup(trainer, data, config):
    """The control's panels with their programme and leak shifts, the niche
    labels and the scored mask (``planted_spillover.read``'s)."""
    from discell.model.validate import niche_labels

    labels = niche_labels(data, PS.N_NICHES, config.seed)
    scored = PS.folds(data) == 0
    panels = PS.build_panels(data, labels, scored, ~scored)
    b_matrix = trainer.model.B.weight.detach().cpu().numpy()
    PS.channel_predictions(trainer, data, labels, ~scored, panels,
                           float(config.kappa), b_matrix)
    connected = data.graph.degrees > 0
    for p in panels:
        p["pop_a"], p["pop_b"] = (
            np.flatnonzero(connected & (data.t == p["type"]) & (labels == k))
            for k in p["pair"])
    return panels, labels, scored


def _mean_log_shift(m: np.ndarray, rows_a, rows_b) -> np.ndarray:
    from discell.model import transport as T

    return (np.log(m[rows_b].mean(0) + T.EPS)
            - np.log(m[rows_a].mean(0) + T.EPS))


# -- (a) the null diagnostic ---------------------------------------------------

def rescore(sim, data, panels, n: int, seed: int) -> dict:
    """Original spill-only contrast (point; reproduces the stored one),
    its five-term decomposition (point), and the corrected member (point +
    half-tile draws, the original's draw seed and subsamples)."""
    from discell.experiments.breakdown_draws import draw_seed, half_tile_weights
    from discell.model import transport as T

    spill = PS.spill_genes(PS.WORLD_SEED, sim.x.shape[1])
    rate = sim.x / np.clip(sim.totals, 1.0, None)[:, None]
    cells = np.unique(np.concatenate(
        [np.concatenate([p["rows_a"], p["rows_b"]]) for p in panels]))
    pos = np.full(data.graph.n_cells, -1, dtype=np.int64)
    pos[cells] = np.arange(len(cells))
    weights, c = half_tile_weights(np.asarray(data.positions, float)[cells],
                                   n, draw_seed("transport_mean", seed))
    weights = np.vstack([np.ones((1, len(cells)), np.float32), weights])
    align = np.full((n + 1, len(panels)), np.nan)
    orig, terms, clean_sd, corr_pc, pop_sd = [], [], [], [], []
    for j, p in enumerate(panels):
        ra, rb = p["rows_a"], p["rows_b"]
        oa, ob = rate[ra].mean(0), rate[rb].mean(0)
        keep = (oa > T.MIN_RATE) & (ob > T.MIN_RATE)
        genes = keep & spill
        if genes.sum() < PS.MIN_SET_GENES:
            continue
        obs = np.log(ob + T.EPS) - np.log(oa + T.EPS)
        clean = _mean_log_shift(sim.rho_true, ra, rb)
        true = _mean_log_shift(sim.p_true, ra, rb)
        prog, leak = p["program"], p["leak"]
        g = genes
        orig.append(PS.r2_rows(obs[g][None], (prog + leak)[g])[0]
                    - PS.r2_rows(obs[g][None], leak[g])[0])
        pop = _mean_log_shift(sim.rho_true, p["pop_a"], p["pop_b"])
        terms.append(decompose(obs[g], clean[g], true[g], prog[g], leak[g],
                               pop[g]))
        pop_sd.append(float(_c(pop[g]).std()))
        clean_sd.append(float(_c(clean[g]).std()))
        corr_pc.append(float(np.corrcoef(_c(prog[g]), _c(clean[g]))[0, 1]))
        means = {}
        for key, mat in (("obs", rate), ("clean", sim.rho_true),
                         ("true", sim.p_true)):
            side = []
            for r in (ra, rb):
                w = weights[:, pos[r]].astype(np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    side.append((w @ mat[r][:, g]) / w.sum(axis=1,
                                                           keepdims=True))
            with np.errstate(invalid="ignore", divide="ignore"):
                means[key] = (np.log(side[1] + T.EPS)
                              - np.log(side[0] + T.EPS))
        ok = np.all([np.isfinite(v).all(1) for v in means.values()], axis=0)
        col = np.full(n + 1, np.nan)
        col[ok] = alignment_rows(means["obs"][ok], means["clean"][ok],
                                 means["true"][ok], prog[g])
        align[:, j] = col
    with np.errstate(invalid="ignore"):
        values = np.nanmean(align, axis=1)
    return {"n_panels": len(orig),
            "original_point": float(np.mean(orig)) if orig else float("nan"),
            "terms": {k: float(np.mean([t[k] for t in terms]))
                      for k in TERMS} if terms else {},
            "clean_shift_sd": float(np.mean(clean_sd)) if clean_sd else None,
            "clean_pop_shift_sd": float(np.mean(pop_sd)) if pop_sd else None,
            "corr_programme_clean": float(np.nanmean(corr_pc))
            if corr_pc else None,
            "estimate": float(values[0]), "draws": values[1:], "c": c,
            "method": "subsample"}


def closure_check(sim_null, panels) -> dict:
    """The no-response genes' clean shift in the null world, against the
    same world with B = 0 for EVERY gene (no draw moves, so same cells, z
    and types): if the spread vanishes there, it is the response genes
    acting on the shared normaliser (closure), not anything spill-like."""
    from discell.model import transport as T
    from discell.model.synthetic import simulate

    flat = simulate(n_cells=PS.N_CELLS, n_types=SR.N_TYPES, kappa=0.0,
                    box_um=PS.BOX_UM, seed=PS.WORLD_SEED,
                    b_zero=np.ones(60, dtype=bool))
    spill = PS.spill_genes()
    rate = sim_null.x / sim_null.totals[:, None]
    out = {"with_response": [], "no_response_anywhere": [],
           "pop_with_response": [], "pop_no_response_anywhere": [],
           "observed_spill_sd": []}
    for p in panels:
        ra, rb = p["rows_a"], p["rows_b"]
        pa, pb = p["pop_a"], p["pop_b"]
        keep = ((rate[ra].mean(0) > T.MIN_RATE) & (rate[rb].mean(0) > T.MIN_RATE)
                & spill)
        for key, m, a, b in (("with_response", sim_null.rho_true, ra, rb),
                             ("no_response_anywhere", flat.rho_true, ra, rb),
                             ("pop_with_response", sim_null.rho_true, pa, pb),
                             ("pop_no_response_anywhere", flat.rho_true, pa,
                              pb)):
            out[key].append(float(_c(_mean_log_shift(m, a, b)[keep]).std()))
        out["observed_spill_sd"].append(
            float(_c(_mean_log_shift(rate, ra, rb)[keep]).std()))
    return {k: float(np.mean(v)) for k, v in out.items()} | {
        "n_panels": len(panels)}


def noise_by_tiles(sim, data, panels, min_cells: int = 8) -> dict:
    """Per panel, corr(programme, count noise e) on the no-response genes,
    with e (observed minus mean p_true shift) formed separately on the
    scored cells inside the model's training tiles and inside its held-out
    tiles (``data.val_tiles``). A programme that fitted the scored cells'
    own count noise correlates with e on the first only."""
    from discell.model import transport as T

    spill = PS.spill_genes(PS.WORLD_SEED, sim.x.shape[1])
    val = np.zeros(data.graph.n_cells, dtype=bool)
    val[np.concatenate(data.val_tiles)] = True
    rate = sim.x / np.clip(sim.totals, 1.0, None)[:, None]
    out = {"train_tiles": [], "val_tiles": []}
    for name, sub in (("train_tiles", ~val), ("val_tiles", val)):
        for p in panels:
            ra = p["rows_a"][sub[p["rows_a"]]]
            rb = p["rows_b"][sub[p["rows_b"]]]
            if min(len(ra), len(rb)) < min_cells:
                continue
            with np.errstate(divide="ignore"):
                o = np.log(rate[rb].mean(0) + T.EPS) - np.log(rate[ra].mean(0)
                                                              + T.EPS)
            e = o - _mean_log_shift(sim.p_true, ra, rb)
            g = spill & np.isfinite(o)
            out[name].append(float(np.corrcoef(_c(e[g]),
                                               _c(p["program"][g]))[0, 1]))
    return out


def null_diagnostic(n: int = PS.N_DRAWS, fits_dir: Path = PS.OUT / "fits",
                    out_stem: Path = RESULT_DIR / "null_diagnostic") -> dict:
    from discell.experiments.breakdown import GRID, read_draws, section_table

    worlds, closure = {}, None
    by_tiles = {"train_tiles": [], "val_tiles": []}
    for kt in PS.KAPPA_TRUE:
        collected, fits = {}, []
        for kappa in GRID:
            collected[kappa] = {}
            for seed in PS.SEEDS:
                name = PS.run_name(kt, kappa, seed)
                sim, data, config, trainer = load(kt, kappa, seed)
                panels, _, _ = setup(trainer, data, config)
                if closure is None and kt == 0.0:
                    closure = closure_check(sim, panels)
                r = rescore(sim, data, panels, n, seed)
                if kt == 0.0:
                    for k, v in noise_by_tiles(sim, data, panels).items():
                        by_tiles[k] += v
                stored = read_draws(fits_dir / f"{name}.npz")
                drift = abs(r["original_point"]
                            - stored["spill_cf_minus_leak"]["estimate"])
                if drift > 1e-4:
                    raise RuntimeError(f"{name}: original contrast does not "
                                       f"reproduce ({drift:.2e})")
                collected[kappa][seed] = {
                    "response_cf_minus_leak": stored["response_cf_minus_leak"],
                    MEMBER: {k: r[k] for k in ("estimate", "draws", "c",
                                               "method")}}
                fits.append({"run": name, "kappa": kappa, "seed": seed,
                             "n_panels": r["n_panels"],
                             "original": r["original_point"],
                             "original_drift": drift,
                             "corrected": r["estimate"],
                             "terms": r["terms"],
                             "clean_shift_sd": r["clean_shift_sd"],
                             "clean_pop_shift_sd": r["clean_pop_shift_sd"],
                             "corr_programme_clean": r["corr_programme_clean"]})
                log.info("%s: original %+.3f corrected %+.3f terms %s", name,
                         r["original_point"], r["estimate"],
                         {k: round(v, 3) for k, v in r["terms"].items()})
        table = section_table(f"planted_k{kt:g}_corrected", collected, m_s=2,
                              grid=GRID, seeds=PS.SEEDS)
        original = json.loads(PS.RESULT.with_suffix(".json").read_text())[
            "worlds"][f"{kt:g}"]["table"]["members"]["spill_cf_minus_leak"]
        worlds[f"{kt:g}"] = {"kappa_true": kt, "fits": fits,
                             "corrected": table["members"][MEMBER],
                             "original": {k: original[k] for k in
                                          ("status", "kappa_star", "reason",
                                           "trajectory") if k in original}}
    result = {"spec": "devlog 'RECOMB additions: results and follow-ups "
                      "(2026-10-01; author decisions)', follow-up (a)",
              "definition": {
                  "original": "mean over panels [R2(prog+leak) - R2(leak)] "
                              "= [2<o-l,p> - ||p||^2]/||o||^2 (centred)",
                  "corrected": "mean over panels 2<p, o - c>/||t||^2, c = "
                               "clean shift (scored cells' mean rho_true), "
                               "t = true contaminated shift (mean p_true)"},
              "grid": list(GRID), "level": 1 - 0.05 / 2, "n_draws": n,
              "closure_check": closure,
              "noise_alignment_null": {
                  k: {"mean_corr": float(np.mean(v)),
                      "se": float(np.std(v) / np.sqrt(len(v))),
                      "n_panel_fits": len(v)} for k, v in by_tiles.items()},
              "worlds": worlds}
    out_stem.with_suffix(".json").write_text(json.dumps(result, indent=1,
                                                        default=float))
    out_stem.with_suffix(".md").write_text(render_null(result))
    return result


def _fmt_entry(e) -> str:
    if e is None:
        return "–"
    lo, hi = e["ci"]
    return (f"{e['estimate']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
            f"({', '.join(f'{v:+.2f}' for v in e['per_seed'])})")


def _kstar(e) -> str:
    if e["status"] == "breaks":
        return f"{e['kappa_star']:g} ({e['reason']})"
    return e["status"] + (f" ({e['reason']})" if e.get("reason") else "")


def render_null(result: dict) -> str:
    grid = result["grid"]
    cl = result["closure_check"]
    na = result["noise_alignment_null"]
    null_fits = result["worlds"]["0"]["fits"]

    def at(k, key):
        fs = [f for f in null_fits if np.isclose(f["kappa"], k)]
        return float(np.mean([f["original"] if key == "original"
                              else f["terms"][key] for f in fs]))
    nul0 = {key: at(0.0, key) for key in ("original", *TERMS)}
    nul_leak = {k: at(k, "leak") for k in grid}
    nul_corr = float(np.mean([f["corr_programme_clean"] for f in null_fits]))
    lines = [
        "# Null-world diagnostic: the spill-only contrast", "",
        "Follow-up (a), devlog 'RECOMB additions: results and follow-ups'. "
        "Existing fits re-read on CPU; no refits. Original contrast "
        f"reproduced on every fit (max drift {max(f['original_drift'] for w in result['worlds'].values() for f in w['fits']):.1e}).", "",
        "## Summary", "",
        "- **Definition.** The spill-only contrast is R²(programme + leak) − "
        "R²(leak) on the 20 no-response genes. Per panel this is exactly "
        "[2⟨o − l, p⟩ − ‖p‖²] / ‖o‖² (o observed shift, p programme, "
        "l leak-only; centred over genes). Split o with the simulator's "
        "truth into niche (population clean shift), sampling (which cells "
        "were scored), spill, and count noise, and the contrast becomes six "
        "additive terms (tables below).",
        f"- **Why it is a 'finding' at κ = 0 in the null world.** The leak "
        f"term is 0 there, so the contrast is R²(programme) alone: "
        f"{nul0['original']:+.3f}. It is set mainly by the penalty "
        f"−‖p‖²/‖o‖² = {nul0['penalty']:+.3f}, which can never be positive. "
        f"Any non-zero programme on these genes scores below 0, however "
        f"small the programme is. Small positive alignments of the programme "
        f"with the clean niche shift ({nul0['niche']:+.3f}) and with count "
        f"noise ({nul0['noise']:+.3f}) only partly offset it.",
        f"- **Why it breaks at 0.1.** As the assumed κ grows past κ_true = 0, "
        f"the leak channel predicts a spill that is not there. The programme "
        f"turns against it, and the leak term −2⟨l, p⟩/‖o‖² goes from "
        f"{nul_leak[0.05]:+.3f} (κ = 0.05) to {nul_leak[0.2]:+.3f} (κ = 0.2) "
        f"and {nul_leak[0.4]:+.3f} (κ = 0.4). The sign change in between is "
        f"the 'break'. Both regimes are properties of the definition (an R² "
        f"difference against a zero-prediction null), not spill-over: a read "
        f"artefact.",
        f"- **Not the simulator's gene set.** The no-response genes do have "
        f"a clean between-niche shift in the null world (sd "
        f"{cl['with_response']:.3f} on the scored cells). It is almost the "
        f"same with B = 0 on every gene ({cl['no_response_anywhere']:.3f}), "
        f"so it is within-type z variation between finite cell sets, not "
        f"closure through the response genes (population level: "
        f"{cl['pop_with_response']:.3f} vs {cl['pop_no_response_anywhere']:.3f}). "
        f"The programme barely aligns with it (corr(p, c) ≈ "
        f"{nul_corr:+.2f}).",
        f"- **A small training-overlap effect.** The programme correlates "
        f"with the count noise of scored cells that lie in the model's "
        f"training tiles (mean corr {na['train_tiles']['mean_corr']:+.3f} ± "
        f"{na['train_tiles']['se']:.3f}, {na['train_tiles']['n_panel_fits']} "
        f"panel × fit) but not with those in its held-out tiles "
        f"({na['val_tiles']['mean_corr']:+.3f} ± {na['val_tiles']['se']:.3f}, "
        f"{na['val_tiles']['n_panel_fits']}). Fold 0 lies mostly in training "
        f"tiles, so the model has seen the counts it is scored on. (Panels "
        f"repeat across the κ grid, so the ± understates the uncertainty.)",
        "- **Corrected contrast.** spill alignment = mean over panels of "
        "2⟨p, o − c⟩/‖t‖². Here c is the simulator's clean shift on the "
        "scored cells and t is their count-free contaminated shift (t = c when "
        "nothing leaks). It keeps only the programme's alignment with the "
        "part of the observed shift that the clean truth does not explain "
        "(spill + count noise). It drops the penalty and the leak term. "
        "Under κ_true = 0 it is zero in expectation for every assumed κ, "
        "except for the training-overlap effect above. It needs the truth, "
        "so it is a simulation-only control. Same draws, same breakdown "
        "rule, family m = 2 with the original response member.", "",
        "## Closure check (null world, model seed 0's panels)", "",
        f"Spread (sd over genes, centred) of the no-response genes' shift, "
        f"mean over {cl['n_panels']} panels:", "",
        "| quantity | scored cells | population (all cells of the type in "
        "the two niches) |", "|---|---|---|",
        f"| clean shift, planted world (B = 0 on these genes only) | "
        f"{cl['with_response']:.3f} | {cl['pop_with_response']:.3f} |",
        f"| clean shift, same world with B = 0 on every gene | "
        f"{cl['no_response_anywhere']:.3f} | "
        f"{cl['pop_no_response_anywhere']:.3f} |",
        f"| observed shift (counts) | {cl['observed_spill_sd']:.3f} | – |", "",
        "## κ* — original and corrected, side by side", "",
        "| κ_true | original κ* | corrected κ* | prediction for a "
        "spill-only contrast |", "|---|---|---|---|"]
    for kt, w in result["worlds"].items():
        pred = ("no finding" if float(kt) == 0 else
                f"breaks at {PS.predicted_kappa_star(float(kt)):g}")
        lines.append(f"| {kt} | {_kstar(w['original'])} | "
                     f"{_kstar(w['corrected'])} | {pred} |")
    for kt, w in result["worlds"].items():
        lines += ["", f"## κ_true = {kt}", "",
                  "| contrast | " + " | ".join(f"κ = {k:g}" for k in grid)
                  + " |", "|---" * (len(grid) + 1) + "|",
                  "| original | " + " | ".join(
                      _fmt_entry(w["original"]["trajectory"].get(f"{k:g}"))
                      for k in grid) + " |",
                  "| corrected | " + " | ".join(
                      _fmt_entry(w["corrected"]["trajectory"].get(f"{k:g}"))
                      for k in grid) + " |", "",
                  "Decomposition of the original (point, mean over 3 seeds; "
                  "the six terms, each / ‖o‖², sum to the original):", "",
                  "| κ | original | 2⟨c_pop,p⟩ niche | 2⟨c−c_pop,p⟩ "
                  "sampling | 2⟨s,p⟩ spill | 2⟨e,p⟩ noise | −2⟨l,p⟩ leak | "
                  "−‖p‖² penalty | corr(p, c) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k in grid:
            fs = [f for f in w["fits"] if np.isclose(f["kappa"], k)]
            t = {key: np.mean([f["terms"][key] for f in fs]) for key in TERMS}
            lines.append(
                f"| {k:g} | {np.mean([f['original'] for f in fs]):+.3f} | "
                + " | ".join(f"{t[key]:+.3f}" for key in TERMS)
                + f" | {np.mean([f['corr_programme_clean'] for f in fs]):+.2f} |")
    return "\n".join(lines) + "\n"


# -- (b) relocation against the clean truth ------------------------------------

METHODS = ("discell_cf", "discell_programme", "regression_log",
           "regression_rate")
TARGETS = ("observed", "clean", "true_contaminated")
RELOC_FITS = [(0.0, 0.0), (0.0, 0.1), (0.1, 0.1), (0.2, 0.2), (0.2, 0.1)]


def regression_predictions(data, labels, scored, panels) -> dict:
    """``{(panel index, form): shift}``: transport_regression's ridge of the
    type's rate (or log1p) on y, model-side connected cells in a niche."""
    from discell.model import transport_regression as TR

    connected = data.graph.degrees > 0
    model_side = ~scored
    y = np.asarray(data.graph.y, dtype=np.float64)
    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    scale = float(data.median_counts)
    fits = {}
    for g in range(len(data.p_t)):
        rows = np.flatnonzero(connected & model_side & (labels >= 0)
                              & (data.t == g))
        for form in TR.FORMS:
            fits[(g, form)] = TR.fit_type(y[rows], x_rate[rows], form, scale)
    out = {}
    for j, p in enumerate(panels):
        g = p["type"]
        side = [np.flatnonzero(connected & model_side & (data.t == g)
                               & (labels == niche)) for niche in p["pair"]]
        ybar_a, ybar_b = y[side[0]].mean(0), y[side[1]].mean(0)
        p["overlap_flag"] = _overlap(y[side[0]], y[side[1]])
        for form in TR.FORMS:
            out[(j, form)] = TR.predicted_shift(fits[(g, form)], form,
                                                ybar_a, ybar_b)
    return out


def _overlap(y_a: np.ndarray, y_b: np.ndarray) -> bool:
    """transport_check's composition-overlap guard (True = extrapolation)."""
    gap_vec = y_b.mean(0) - y_a.mean(0)
    gap = np.linalg.norm(gap_vec)
    unit = gap_vec / max(gap, 1e-12)
    spread = 0.5 * (float((y_a @ unit).std()) + float((y_b @ unit).std()))
    return bool(gap > 3 * spread)


def relocation_one(sim, data, config, trainer) -> dict:
    from discell.model import transport as T

    panels, labels, scored = setup(trainer, data, config)
    reg = regression_predictions(data, labels, scored, panels)
    rate = sim.x / np.clip(sim.totals, 1.0, None)[:, None]
    spill = PS.spill_genes()
    rng = np.random.default_rng(config.seed)
    rng_clean = np.random.default_rng([config.seed, 1])
    rows = []
    for j, p in enumerate(panels):
        ra, rb = p["rows_a"], p["rows_b"]
        oa, ob = rate[ra].mean(0), rate[rb].mean(0)
        keep = (oa > T.MIN_RATE) & (ob > T.MIN_RATE)
        targets = {"observed": np.log(ob + T.EPS) - np.log(oa + T.EPS),
                   "clean": _mean_log_shift(sim.rho_true, ra, rb),
                   "true_contaminated": _mean_log_shift(sim.p_true, ra, rb)}
        preds = {"discell_cf": p["program"] + p["leak"],
                 "discell_programme": p["program"],
                 "regression_log": reg[(j, "log")],
                 "regression_rate": reg[(j, "rate")]}
        row = {"pair": list(p["pair"]), "type": int(p["type"]),
               "n_scored": [len(ra), len(rb)], "n_genes": int(keep.sum()),
               "overlap_flag": p["overlap_flag"],
               "ceiling": {"observed": T.noise_ceiling(rate, ra, rb, keep, rng),
                           "clean": T.noise_ceiling(sim.rho_true, ra, rb, keep,
                                                    rng_clean)},
               "r2": {}}
        for gset, mask in (("all", keep), ("spill", keep & spill),
                           ("response", keep & ~spill)):
            row["r2"][gset] = {
                tgt: {m: T.score_shift(pred[mask], val[mask])["r2"]
                      for m, pred in preds.items()}
                for tgt, val in targets.items()}
        rows.append(row)
    return {"n_panels": len(rows), "panels": rows}


def summarise(panels: list[dict]) -> dict:
    out = {}
    for gset in ("all", "spill", "response"):
        out[gset] = {}
        for tgt in TARGETS:
            out[gset][tgt] = {}
            ceil = (float(np.mean([p["ceiling"][tgt] for p in panels]))
                    if tgt in ("observed", "clean") else None)
            for m in METHODS:
                r2 = float(np.mean([p["r2"][gset][tgt][m] for p in panels]))
                out[gset][tgt][m] = {
                    "mean_r2": r2,
                    "of_ceiling": (r2 / ceil if ceil is not None
                                   and ceil >= 0.05 else None)}
            out[gset][tgt]["ceiling"] = ceil
    return out


def relocation(out_stem: Path = RESULT_DIR / "relocation_clean_truth") -> dict:
    fits = []
    for kt, assumed in RELOC_FITS:
        for seed in PS.SEEDS:
            sim, data, config, trainer = load(kt, assumed, seed)
            r = relocation_one(sim, data, config, trainer)
            s = summarise(r["panels"])
            fits.append({"run": PS.run_name(kt, assumed, seed),
                         "kappa_true": kt, "assumed": assumed, "seed": seed,
                         "n_panels": r["n_panels"],
                         "n_extrapolation": int(sum(p["overlap_flag"]
                                                    for p in r["panels"])),
                         "summary": s, "panels": r["panels"]})
            log.info("%s: clean R2 %s", fits[-1]["run"],
                     {m: round(s["all"]["clean"][m]["mean_r2"], 3)
                      for m in METHODS})
    result = {"spec": "devlog 'RECOMB additions: results and follow-ups "
                      "(2026-10-01; author decisions)', follow-up (b)",
              "prediction": "against the clean truth the regression degrades "
                            "as kappa_true grows, and DISCELL degrades less",
              "methods": list(METHODS), "targets": list(TARGETS),
              "fits": fits}
    out_stem.with_suffix(".json").write_text(json.dumps(result, indent=1,
                                                        default=float))
    out_stem.with_suffix(".md").write_text(render_relocation(result))
    return result


NAMES = {"discell_cf": "DISCELL counterfactual",
         "discell_programme": "DISCELL programme (spill-free)",
         "regression_log": "regression (log)",
         "regression_rate": "regression (rate)"}


def world_means(result: dict, gset: str = "all") -> dict:
    """``{(kappa_true, assumed): {target: {method: (mean, min, max)}}}``
    over seeds, of mean R^2 and of the fraction of ceiling."""
    out = {}
    for kt, assumed in RELOC_FITS:
        fs = [f for f in result["fits"] if f["kappa_true"] == kt
              and f["assumed"] == assumed]
        cell = {}
        for tgt in TARGETS:
            cell[tgt] = {}
            for m in METHODS:
                v = [f["summary"][gset][tgt][m]["mean_r2"] for f in fs]
                q = [f["summary"][gset][tgt][m]["of_ceiling"] for f in fs]
                cell[tgt][m] = {"r2": (np.mean(v), min(v), max(v)),
                                "of_ceiling": (np.mean(q), min(q), max(q))
                                if None not in q else None}
            cell[tgt]["ceiling"] = (np.mean([f["summary"][gset][tgt]["ceiling"]
                                             for f in fs])
                                    if tgt != "true_contaminated" else None)
        out[(kt, assumed)] = cell
    return out


def _r(t) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def render_relocation(result: dict) -> str:
    wm = world_means(result)
    lines = [
        "# Relocation against the clean truth (planted worlds)", "",
        "Follow-up (b), devlog 'RECOMB additions: results and follow-ups'. "
        "Existing planted fits, re-read on CPU. Panels: the control's (niche "
        "pair, type) panels (k-means niches on y, fold 0 scored, ≥150 "
        "model-side / ≥30 scored cells per side). R²: "
        "`transport.score_shift` (centred, zero-prediction null) on the "
        "panel's kept genes (all 60 gene types, spill and response). "
        "Ceilings: `transport.noise_ceiling` of each target (split-half "
        "reliability); of ceiling = mean R² / mean ceiling. Cells: 3-seed "
        "mean [min, max].", "",
        "Targets: **observed** = held-out counts' log mean-rate shift (as on "
        "real data); **clean** = the scored cells' mean rho_true shift (no "
        "spill, no counts). Methods: DISCELL counterfactual = programme + "
        "leak-only (the published relocation); DISCELL programme = its "
        "spill-free part; regression = transport_regression's ridge on the "
        "simulated observed counts.", ""]

    def clean(kt, m):
        return wm[(kt, kt)]["clean"][m]["r2"][0]
    trail = {m: " → ".join(f"{clean(k, m):.2f}" for k in (0.0, 0.1, 0.2))
             for m in METHODS}
    lines += [
        "## Summary", "",
        "Against the clean truth, at assumed κ = κ_true (κ_true 0 → 0.1 → "
        "0.2): DISCELL programme " + trail["discell_programme"]
        + "; regression (log) " + trail["regression_log"]
        + "; regression (rate) " + trail["regression_rate"]
        + "; DISCELL counterfactual (programme + leak) "
        + trail["discell_cf"] + ". The prediction (the regression degrades "
        "with κ_true, DISCELL less) holds for DISCELL's spill-free programme. "
        "It does not hold for the published counterfactual, which adds the "
        "predicted spill and so targets the contaminated shift, as the "
        "regression does. Against the observed shift (as on real data) the "
        "rate regression and the counterfactual are close whenever spill is "
        "present.", ""]
    for tgt in ("observed", "clean"):
        lines += [f"## Against the {tgt} shift: mean R² (fraction of ceiling)",
                  "", "| κ_true | assumed κ | ceiling | "
                  + " | ".join(NAMES[m] for m in METHODS) + " |",
                  "|---" * (3 + len(METHODS)) + "|"]
        for (kt, a), cell in wm.items():
            c = cell[tgt]
            lines.append(f"| {kt:g} | {a:g} | {c['ceiling']:.3f} | " + " | ".join(
                f"{_r(c[m]['r2'])} ({c[m]['of_ceiling'][0]:.3f})"
                for m in METHODS) + " |")
        lines.append("")
    lines += ["## By gene set, against the clean shift (3-seed mean R²)", "",
              "| κ_true | assumed κ | gene set | "
              + " | ".join(NAMES[m] for m in METHODS) + " |",
              "|---" * (3 + len(METHODS)) + "|"]
    for gset in ("spill", "response"):
        for (kt, a), cell in world_means(result, gset).items():
            lines.append(f"| {kt:g} | {a:g} | {gset} | " + " | ".join(
                f"{cell['clean'][m]['r2'][0]:.3f}" for m in METHODS) + " |")
    lines += ["", "## Per fit (all genes)", "",
              "Cells: R² observed (fraction of ceiling) / R² clean (fraction "
              "of ceiling).", "",
              "| run | panels | ceiling obs / clean | "
              + " | ".join(NAMES[m] for m in METHODS)
              + " |", "|---" * (3 + len(METHODS)) + "|"]

    def frac(v):
        return "–" if v is None else f"{v:.2f}"
    for f in result["fits"]:
        s = f["summary"]["all"]
        lines.append(
            f"| {f['run']} | {f['n_panels']} | {s['observed']['ceiling']:.2f}"
            f" / {s['clean']['ceiling']:.2f} | " + " | ".join(
                f"{s['observed'][m]['mean_r2']:.3f} "
                f"({frac(s['observed'][m]['of_ceiling'])}) / "
                f"{s['clean'][m]['mean_r2']:.3f} "
                f"({frac(s['clean'][m]['of_ceiling'])})"
                for m in METHODS) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    nl = sub.add_parser("null")
    nl.add_argument("--n", type=int, default=PS.N_DRAWS)
    sub.add_parser("relocation")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.cmd == "null":
        null_diagnostic(args.n)
    else:
        relocation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
