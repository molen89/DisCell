#!/usr/bin/env python3
"""resolVI's double-positive metric on raw / x~ / counts-corrected decodes (6b.2).

Todo 6b.2; devlog "Baseline articles reviewed (2026-09-21)" (resolVI Methods
p. 21) and "6b.2 curation"; pairs approved 2026-09-25 (the core sets of
``scripts/logs/marker_pairs_2026-09-23/AGENT_REPORT.md``).

**How.** One eval-mode forward pass (posterior means) over a run's held-out
(validation) tiles gives every held-out cell its raw counts ``x``, depth
``l = sum_g x_g``, the foreign influx ``rho_bar`` (the neighbours' decodes
through the leak edges, ``Forward.rho_bar``) and the decoded mixture
``p = (1 - kappa) rho + kappa rho_bar`` (``Forward.log_p``). A gene is *on*
in a cell under each arm (:data:`ARMS`):

``raw``               ``x >= 1``
``xtilde``            ``rint(x~) >= 1``, ``x~ = clip(x - kappa l rho_bar, 0)``
                      -- the counts-level correction, formed as in
                      ``networks.DisCell.forward`` (primary x~ arm)
``xtilde_ge1``        ``x~ >= 1`` unrounded -- sensitivity only: rho_bar > 0
                      for every gene of a connected cell, so this is ``x >= 2``
                      there at any kappa > 0, a threshold change
``decode``            ``rint(l p) >= 1`` -- the decode with the leak left in
``decode_corrected``  ``rint(l (p - kappa rho_bar)) >= 1`` -- the kappa-l-
                      corrected decode, ``l (1 - kappa) rho`` on connected cells

The thresholds are fixed a priori, so nothing is fitted on training tiles.

**Evaluated by.** Per pair the double-positive (DP) rate -- the share of
held-out cells with both genes on; per arm the unweighted mean over the
exclusive pairs, over the control pairs, and the ratio exclusive / control
(the headline). 95 % intervals from :func:`bootstrap.tile_bootstrap` (200 um
tiles), differences arm - raw (and ``decode_corrected - decode``) paired on
the same draws. Per seed: *exclusive falls* = CI of the exclusive difference
below 0; *control holds* = CI of the control difference not below 0;
*specific* = CI of the ratio difference below 0. Across the kappa grid
(:func:`kappa_summary`) the three-seed mean, the envelope of the per-seed
intervals, and per arm and kappa the verdict: *as wished* (exclusive falls
and control holds on every seed), *specific, control also falls*, or *no
specific removal*.

**Wished for.** Exclusive DP falls with the correction, control DP does not.

Usage::

    python -m discell.experiments.marker_pairs --dataset <id> --run finalL_s0 [more runs]
    python -m discell.experiments.marker_pairs --dataset <id> --run <run> --pairs my.csv
    python -m discell.experiments.marker_pairs --dataset <id> --kappa

Writes ``runs/<run>/marker_pairs.json``; ``--kappa`` writes
``experiments/marker_pairs_kappa.{json,md,png}`` from those files.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("discell.experiments.marker_pairs")

ARMS = ("raw", "xtilde", "xtilde_ge1", "decode", "decode_corrected")
#: (arm, reference) pairs whose paired differences are graded
COMPARISONS = (("xtilde", "raw"), ("xtilde_ge1", "raw"), ("decode", "raw"),
               ("decode_corrected", "raw"), ("decode_corrected", "decode"))
PANELS = ("exclusive", "control", "ratio")
KAPPAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
SEEDS = (0, 1, 2)
SWEEP_PREFIX = "sweepL"

#: The approved core sets (2026-09-23 curation, approved 2026-09-25). FF ovary
#: is the list as written: EPCAM/{CD3E,CD68,PECAM1,PDGFRA} and WT1/CD68 out,
#: RGS5/EPCAM kept (flagged in the scoring report).
CORE = {
    "xenium_prime_ovarian_cancer_ffpe": {
        "exclusive": ["EPCAM/CD3E", "EPCAM/CD68", "EPCAM/PECAM1", "EPCAM/DCN",
                      "CD3E/MS4A1", "CD68/MS4A1", "CD68/PECAM1", "PECAM1/DCN",
                      "DCN/MYH11", "RGS5/CD3E", "FOXJ1/CD3E", "CD8A/MS4A1"],
        "control": ["CD3E/CD3G", "CD79A/MS4A1", "EPCAM/CDH1", "PECAM1/CDH5",
                    "CD68/CSF1R", "DCN/PDGFRA"]},
    "xenium_prime_human_ovary_ff": {
        "exclusive": ["PAX8/CD3E", "CD3E/MS4A1", "CD68/MS4A1", "CD68/PECAM1",
                      "PECAM1/PDGFRA", "PDGFRA/CD3E", "RGS5/CD3E", "RGS5/EPCAM",
                      "CD8A/MS4A1", "FOXP3/CD68", "MZB1/CD3E", "KLRD1/MS4A1"],
        "control": ["CD3E/CD3G", "CD79A/MS4A1", "EPCAM/CDH1", "PECAM1/CDH5",
                    "CD68/CSF1R", "CD8A/GZMB"]},
    "xenium_prime_human_lung_cancer_ffpe": {
        "exclusive": ["EPCAM/CD3E", "EPCAM/CD68", "EPCAM/PECAM1", "NKX2-1/CD3E",
                      "ABCA3/CD3E", "FOXJ1/CD3E", "TP63/CD3E", "MUC5AC/CD3E",
                      "CD3E/MS4A1", "CD68/MS4A1", "PECAM1/PDGFRA", "CD8A/MS4A1"],
        "control": ["CD3E/CD3G", "CD79A/MS4A1", "EPCAM/CDH1", "PECAM1/CDH5",
                    "CD68/CSF1R", "AGER/HOPX", "ABCA3/LAMP3"]},
    "gse315411_pdltma06_11_prime_solo": {
        "exclusive": ["EPCAM/CD3E", "EPCAM/CD68", "EPCAM/PECAM1", "ABCA3/CD3E",
                      "AGER/CD3E", "FOXJ1/CD3E", "TP63/CD3E", "MUC5AC/CD3E",
                      "CD3E/MS4A1", "CD68/MS4A1", "PECAM1/PDGFRA", "CD8A/MS4A1"],
        "control": ["CD3E/CD3G", "CD79A/MS4A1", "EPCAM/CDH1", "ABCA3/LAMP3",
                    "AGER/HOPX", "PECAM1/CDH5", "CD68/CSF1R"]},
}


# --------------------------------------------------------------------------
# pairs
# --------------------------------------------------------------------------


def load_pairs(dataset: str, path: str | Path | None = None) -> list[dict]:
    """``[{gene_a, gene_b, kind}]``. Default: the dataset's proposed CSV
    restricted to the approved core set (every core pair must be in it);
    with *path*, every row of that CSV."""
    from discell import paths

    default = path is None
    path = Path(path) if path else (paths.dataset(dataset).root / "experiments"
                                    / "marker_pairs_proposed.csv")
    with open(path, newline="") as fh:
        rows = [{"gene_a": r["gene_a"], "gene_b": r["gene_b"],
                 "kind": r["kind"]} for r in csv.DictReader(fh)]
    bad = {r["kind"] for r in rows} - {"exclusive", "control"}
    if bad:
        raise ValueError(f"{path}: unknown kind(s) {sorted(bad)}")
    if not default:
        return rows
    core = CORE[dataset]
    by_name = {(f"{r['gene_a']}/{r['gene_b']}", r["kind"]): r for r in rows}
    missing = [(n, k) for k, names in core.items() for n in names
               if (n, k) not in by_name]
    if missing:
        raise ValueError(f"core pairs absent from {path}: {missing}")
    return [by_name[(n, k)] for k in ("exclusive", "control") for n in core[k]]


# --------------------------------------------------------------------------
# the calls (pure numpy; the planted test runs these)
# --------------------------------------------------------------------------


def on_calls(x: np.ndarray, depth: np.ndarray, rho_bar: np.ndarray,
             p: np.ndarray, kappa: float) -> dict[str, np.ndarray]:
    """Boolean ``(n, g)`` "on" matrix per arm (see the module docstring).

    *x*, *rho_bar*, *p*: ``(n, g)`` for the scored genes; *depth* ``(n,)``
    is the raw total over the whole panel.
    """
    x = np.asarray(x, dtype=np.float64)
    l = np.asarray(depth, dtype=np.float64)[:, None]
    leak = kappa * l * np.asarray(rho_bar, dtype=np.float64)
    x_tilde = np.clip(x - leak, 0.0, None)
    rate = l * np.asarray(p, dtype=np.float64)
    return {"raw": x >= 1,
            "xtilde": np.rint(x_tilde) >= 1,
            "xtilde_ge1": x_tilde >= 1,
            "decode": np.rint(rate) >= 1,
            "decode_corrected": np.rint(np.clip(rate - leak, 0.0, None)) >= 1}


def double_positive(on: dict[str, np.ndarray], idx_a: np.ndarray,
                    idx_b: np.ndarray) -> np.ndarray:
    """Per-cell DP indicators ``(n, len(ARMS) * P)``, arm-major, float32."""
    return np.hstack([(on[a][:, idx_a] & on[a][:, idx_b]) for a in ARMS]
                     ).astype(np.float32)


def panel_reads(rates: np.ndarray, kinds: np.ndarray) -> dict:
    """``{arm|panel: value}`` and ``{arm|pair<i>: rate}`` from the flat
    ``len(ARMS) * P`` vector of DP rates, plus the graded differences."""
    n_pairs = len(kinds)
    excl, ctrl = kinds == "exclusive", kinds == "control"
    out: dict = {}
    for k, arm in enumerate(ARMS):
        r = rates[k * n_pairs:(k + 1) * n_pairs]
        e, c = float(r[excl].mean()), float(r[ctrl].mean())
        out[f"{arm}|exclusive"] = e
        out[f"{arm}|control"] = c
        out[f"{arm}|ratio"] = e / c if c > 0 else float("nan")
        for i in range(n_pairs):
            out[f"{arm}|pair{i}"] = float(r[i])
    for arm, ref in COMPARISONS:
        for panel in PANELS:
            out[f"{arm}-{ref}|{panel}"] = out[f"{arm}|{panel}"] - out[f"{ref}|{panel}"]
    return out


def grade(reads: dict) -> dict:
    """Per comparison: exclusive falls / control holds / specific, from the
    paired CIs in *reads* (``tile_bootstrap``'s ``reads`` block)."""
    out = {}
    for arm, ref in COMPARISONS:
        key = f"{arm}-{ref}"
        hi = {p: reads[f"{key}|{p}"]["ci95"][1] for p in PANELS}
        out[key] = {"exclusive_falls": bool(hi["exclusive"] < 0),
                    "control_holds": bool(not hi["control"] < 0),
                    "specific": bool(hi["ratio"] < 0)}
    return out


def score(cells: dict, kappa: float, pairs: list[dict], genes: list[str],
          positions: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """DP rates per arm with tile CIs and the per-seed grading."""
    from discell.experiments.bootstrap import tile_bootstrap, weighted_mean

    col = {g: i for i, g in enumerate(genes)}
    idx_a = np.array([col[p["gene_a"]] for p in pairs])
    idx_b = np.array([col[p["gene_b"]] for p in pairs])
    kinds = np.array([p["kind"] for p in pairs])
    on = on_calls(cells["x"], cells["depth"], cells["rho_bar"], cells["p"], kappa)
    dp = double_positive(on, idx_a, idx_b)

    boot = tile_bootstrap(positions, dp,
                          lambda v, w: panel_reads(weighted_mean(v, w), kinds),
                          n_boot, seed=seed)
    reads = boot["reads"]
    names = [f"{p['gene_a']}/{p['gene_b']}" for p in pairs]
    frac_on = {arm: {g: float(on[arm][:, i].mean()) for g, i in col.items()}
               for arm in ARMS}
    leak = kappa * cells["depth"][:, None] * cells["rho_bar"]
    return {
        "arms": {arm: {panel: reads[f"{arm}|{panel}"] for panel in PANELS}
                 for arm in ARMS},
        "differences": {f"{a}-{r}": {panel: reads[f"{a}-{r}|{panel}"]
                                     for panel in PANELS}
                        for a, r in COMPARISONS},
        "grading": grade(reads),
        "pairs": [{"pair": names[i], "kind": str(kinds[i]),
                   **{arm: reads[f"{arm}|pair{i}"] for arm in ARMS}}
                  for i in range(len(pairs))],
        "frac_on": frac_on,
        "expected_leak_counts": {g: {"mean": float(leak[:, i].mean()),
                                     "p90": float(np.percentile(leak[:, i], 90)),
                                     "share_ge_0.5": float((leak[:, i] >= 0.5).mean())}
                                 for g, i in col.items()},
        "bootstrap": {k: boot[k] for k in ("n_boot", "n_tiles", "n_cells",
                                           "tile_um", "seed")},
    }


# --------------------------------------------------------------------------
# the model side
# --------------------------------------------------------------------------


def collect_cells(trainer, batches: list[dict], gene_index: np.ndarray) -> dict:
    """Per held-out seed: ``x``, ``rho_bar``, ``p`` on *gene_index*, the
    panel depth, whether the cell is isolated, and its node id."""
    import torch

    gi = torch.as_tensor(gene_index, dtype=torch.long)
    out: dict = {k: [] for k in ("nodes", "x", "depth", "rho_bar", "p",
                                 "isolated")}
    trainer.model.eval()
    with torch.no_grad():
        for batch in batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch),
                                kappa=trainer.config.kappa, sample=False)
            n = batch["n_seeds"]
            g = gi.to(fwd.rho_bar.device)
            x = batch["x"][:n].float()
            out["nodes"].append(np.asarray(batch["nodes"][:n]))
            out["x"].append(x[:, g].cpu().numpy())
            out["depth"].append(x.sum(dim=1).cpu().numpy())
            out["rho_bar"].append(fwd.rho_bar[:, g].cpu().numpy())
            out["p"].append(fwd.log_p[:n].exp()[:, g].cpu().numpy())
            out["isolated"].append(batch["isolated"][:n].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _assembly_key(config) -> tuple:
    return (config.dataset, config.variant, config.embeddings,
            config.tile_cells, config.phi_pca, config.v_pcs,
            config.val_fraction, config.seed, config.label_key)


def score_run(dataset: str, run: str, pairs: list[dict], pairs_source: str,
              device: str = "cuda", n_boot: int = 1000, seed: int = 0,
              cache: dict | None = None) -> dict:
    """Score one run; *cache* shares assemblies between runs of one split."""
    import torch

    from discell import paths
    from discell.model.prepare import assemble
    from discell.model.train import config_from_record
    from discell.model.validate import load_run

    run_dir = paths.dataset(dataset).root / "runs" / run
    payload = torch.load(run_dir / "best.pt", map_location="cpu",
                         weights_only=False)
    config = config_from_record(payload["config"])
    del payload
    if config.kappa_mode != "global" or config.fp_floor or config.subtract_leak:
        raise ValueError(f"{run}: scored for the global-kappa, no-floor, "
                         "raw-input model only")
    cache = {} if cache is None else cache
    key = _assembly_key(config)
    if key not in cache:
        cache.clear()                     # one assembly resident at a time
        cache[key] = assemble(dataset, config.variant, config.embeddings,
                              tile_cells=config.tile_cells,
                              phi_pca=config.phi_pca, v_pcs=config.v_pcs,
                              val_fraction=config.val_fraction,
                              seed=config.seed, label_key=config.label_key)
    # only the held-out tiles go to the device
    held = dataclasses.replace(cache[key], train_tiles=[])
    config, data, trainer, run_dir, _ = load_run(dataset, run, device, data=held)
    genes = sorted({g for p in pairs for g in (p["gene_a"], p["gene_b"])})
    names = [str(g) for g in data.gene_names]
    absent = [g for g in genes if g not in names]
    if absent:
        raise ValueError(f"{dataset}: genes not on the panel: {absent}")
    cells = collect_cells(trainer, trainer.val_batches,
                          np.array([names.index(g) for g in genes]))
    del trainer
    torch.cuda.empty_cache()
    positions = np.asarray(data.positions, dtype=np.float64)[cells["nodes"]]
    result = {"run": run, "dataset": dataset, "kappa": float(config.kappa),
              "seed": int(config.seed), "pairs_source": pairs_source,
              "section": "validation tiles (held out)",
              "n_cells": int(len(cells["nodes"])),
              "isolated_share": float(cells["isolated"].mean()),
              "genes": genes,
              **score(cells, float(config.kappa), pairs, genes, positions,
                      n_boot, seed),
              "definitions": {
                  "raw": "x >= 1",
                  "xtilde": "rint(clip(x - kappa l rho_bar, 0)) >= 1",
                  "xtilde_ge1": "clip(x - kappa l rho_bar, 0) >= 1 (sensitivity)",
                  "decode": "rint(l p) >= 1, p the model's decoded mixture",
                  "decode_corrected": "rint(l (p - kappa rho_bar)) >= 1",
                  "dp": "share of held-out cells with both genes on",
                  "panels": "unweighted mean over the kind's pairs; ratio = "
                            "exclusive / control",
                  "grading": "exclusive_falls: CI(diff) < 0; control_holds: "
                             "CI(diff) not < 0; specific: CI(ratio diff) < 0",
                  "l": "raw count total over the whole panel"}}
    (run_dir / "marker_pairs.json").write_text(json.dumps(result, indent=1))
    a = result["arms"]
    log.info("%s/%s k=%g: exclusive DP raw %.4f xtilde %.4f dec %.4f dec_corr "
             "%.4f | control raw %.4f xtilde %.4f dec %.4f dec_corr %.4f",
             dataset, run, config.kappa,
             *[a[arm]["exclusive"]["estimate"] for arm in
               ("raw", "xtilde", "decode", "decode_corrected")],
             *[a[arm]["control"]["estimate"] for arm in
               ("raw", "xtilde", "decode", "decode_corrected")])
    return result


# --------------------------------------------------------------------------
# the kappa grid
# --------------------------------------------------------------------------


def _envelope(entries: list[dict]) -> list[float]:
    lo = [e["ci95"][0] for e in entries if np.isfinite(e["ci95"][0])]
    hi = [e["ci95"][1] for e in entries if np.isfinite(e["ci95"][1])]
    return [min(lo) if lo else float("nan"), max(hi) if hi else float("nan")]


def _combine(entries: list[dict]) -> dict:
    est = [e["estimate"] for e in entries]
    return {"mean": float(np.mean(est)), "per_seed": [float(v) for v in est],
            "ci95_envelope": _envelope(entries)}


def verdict(gradings: list[dict]) -> str:
    n = len(gradings)
    falls = sum(g["exclusive_falls"] for g in gradings)
    holds = sum(g["control_holds"] for g in gradings)
    spec = sum(g["specific"] for g in gradings)
    if falls == n and holds == n:
        return "as wished"
    if falls == n and spec == n:
        return "specific, control also falls"
    return "no specific removal"


def kappa_summary(dataset: str, prefix: str = SWEEP_PREFIX,
                  kappas: Sequence[float] = KAPPAS,
                  seeds: Sequence[int] = SEEDS) -> dict:
    """Three-seed reads per kappa from the runs' ``marker_pairs.json``."""
    from discell import paths

    root = paths.dataset(dataset).root
    grid: dict = {}
    pairs = None
    for k in kappas:
        runs = [f"{prefix}_k{k:g}_s{s}" for s in seeds]
        recs = [json.loads((root / "runs" / r / "marker_pairs.json").read_text())
                for r in runs]
        pairs = pairs or [(p["pair"], p["kind"]) for p in recs[0]["pairs"]]
        for rec in recs:
            if [(p["pair"], p["kind"]) for p in rec["pairs"]] != pairs:
                raise ValueError(f"{dataset}: pair sets differ across runs")
        grid[f"{k:g}"] = {
            "runs": runs,
            "n_cells": [r["n_cells"] for r in recs],
            "arms": {arm: {panel: _combine([r["arms"][arm][panel] for r in recs])
                           for panel in PANELS} for arm in ARMS},
            "differences": {f"{a}-{ref}": {
                panel: _combine([r["differences"][f"{a}-{ref}"][panel]
                                 for r in recs]) for panel in PANELS}
                for a, ref in COMPARISONS},
            "grading": {f"{a}-{ref}": {
                "per_seed": [r["grading"][f"{a}-{ref}"] for r in recs],
                "verdict": verdict([r["grading"][f"{a}-{ref}"] for r in recs])}
                for a, ref in COMPARISONS},
            "pairs": [{"pair": name, "kind": kind,
                       **{arm: float(np.mean([r["pairs"][i][arm]["estimate"]
                                              for r in recs])) for arm in ARMS}}
                      for i, (name, kind) in enumerate(pairs)],
        }
    out = {"dataset": dataset, "prefix": prefix, "kappas": list(kappas),
           "seeds": list(seeds), "pairs_source": recs[0]["pairs_source"],
           "grid": grid,
           "combination": "mean of the seed estimates; interval = envelope of "
                          "the per-seed 200 um tile-bootstrap 95 % CIs"}
    exp = root / "experiments"
    (exp / "marker_pairs_kappa.json").write_text(json.dumps(out, indent=1))
    (exp / "marker_pairs_kappa.md").write_text(kappa_markdown(out))
    kappa_figure(out, exp / "marker_pairs_kappa.png")
    log.info("%s: wrote %s", dataset, exp / "marker_pairs_kappa.{json,md,png}")
    return out


def _pct(v: float) -> str:
    return f"{100 * v:.3f}"


def kappa_markdown(out: dict) -> str:
    lines = [f"# Double-positive rates across κ — {out['dataset']}", "",
             f"Runs `{out['prefix']}_k<κ>_s{{{','.join(map(str, out['seeds']))}}}` "
             "(κ = 0.1 = finalL). Held-out cells. DP in %; three-seed mean "
             "[envelope of per-seed 200 µm tile-bootstrap 95 % CIs]. Ratio = "
             "exclusive / control. Pairs: " + out["pairs_source"], ""]
    for k, g in out["grid"].items():
        lines += [f"## κ = {k}", "",
                  "| arm | exclusive DP % | control DP % | ratio excl/ctrl |",
                  "|---|---|---|---|"]
        for arm in ARMS:
            a = g["arms"][arm]
            lines.append(
                f"| {arm} | {_pct(a['exclusive']['mean'])} "
                f"[{_pct(a['exclusive']['ci95_envelope'][0])}, "
                f"{_pct(a['exclusive']['ci95_envelope'][1])}] | "
                f"{_pct(a['control']['mean'])} "
                f"[{_pct(a['control']['ci95_envelope'][0])}, "
                f"{_pct(a['control']['ci95_envelope'][1])}] | "
                f"{a['ratio']['mean']:.4f} [{a['ratio']['ci95_envelope'][0]:.4f}, "
                f"{a['ratio']['ci95_envelope'][1]:.4f}] |")
        lines += ["", "| comparison | Δ exclusive (pp) | Δ control (pp) | Δ ratio | "
                  "excl falls / ctrl holds / specific (seeds) | verdict |",
                  "|---|---|---|---|---|---|"]
        for cmp_, d in g["differences"].items():
            per = g["grading"][cmp_]["per_seed"]
            counts = "/".join(str(sum(s[f] for s in per)) for f in
                              ("exclusive_falls", "control_holds", "specific"))
            lines.append(
                f"| {cmp_} | {_pct(d['exclusive']['mean'])} | "
                f"{_pct(d['control']['mean'])} | {d['ratio']['mean']:+.4f} | "
                f"{counts} of {len(per)} | {g['grading'][cmp_]['verdict']} |")
        lines.append("")
    pin = out["grid"].get("0.1")
    if pin:
        lines += ["## Per pair at κ = 0.1 (three-seed mean DP %)", "",
                  "| pair | kind | " + " | ".join(ARMS) + " |",
                  "|---|---|" + "---|" * len(ARMS)]
        for p in pin["pairs"]:
            lines.append(f"| {p['pair']} | {p['kind']} | "
                         + " | ".join(_pct(p[a]) for a in ARMS) + " |")
        lines.append("")
    return "\n".join(lines)


#: validated categorical order (dataviz reference palette, light mode)
COLOURS = {"raw": "#2a78d6", "xtilde": "#eb6834", "xtilde_ge1": "#1baf7a",
           "decode": "#eda100", "decode_corrected": "#e87ba4"}
MARKERS = {"raw": "o", "xtilde": "s", "xtilde_ge1": "D", "decode": "^",
           "decode_corrected": "v"}


def kappa_figure(out: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [float(k) for k in out["grid"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    titles = {"exclusive": "exclusive pairs: DP %", "control":
              "control pairs: DP %", "ratio": "ratio exclusive / control"}
    for ax, panel in zip(axes, PANELS):
        scale = 1.0 if panel == "ratio" else 100.0
        for arm in ARMS:
            reads = [g["arms"][arm][panel] for g in out["grid"].values()]
            mean = np.array([r["mean"] for r in reads]) * scale
            lo = np.array([r["ci95_envelope"][0] for r in reads]) * scale
            hi = np.array([r["ci95_envelope"][1] for r in reads]) * scale
            ax.fill_between(ks, lo, hi, color=COLOURS[arm], alpha=0.15, lw=0)
            ax.plot(ks, mean, color=COLOURS[arm], lw=2, marker=MARKERS[arm],
                    ms=6, label=arm)
        ax.set_title(titles[panel], fontsize=10)
        ax.set_xlabel("κ of the run")
        ax.axvline(0.1, color="0.6", lw=0.8, ls=":")
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{out['dataset']}: double-positive rate on held-out cells "
                 "(3-seed mean, band = envelope of per-seed tile CIs; dotted = pinned κ)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", nargs="+", default=[],
                        help="run(s) to score; runs sharing a split share one "
                             "assembly")
    parser.add_argument("--pairs", default=None,
                        help="CSV (gene_a, gene_b, kind); default: the approved "
                             "core set of experiments/marker_pairs_proposed.csv")
    parser.add_argument("--kappa", action="store_true",
                        help="write experiments/marker_pairs_kappa.* from the "
                             "sweep runs' marker_pairs.json")
    parser.add_argument("--prefix", default=SWEEP_PREFIX)
    parser.add_argument("--overwrite", action="store_true",
                        help="rescore runs that already have marker_pairs.json")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    from discell import paths

    pairs = load_pairs(args.dataset, args.pairs)
    source = args.pairs or ("experiments/marker_pairs_proposed.csv, approved "
                            "core set (marker_pairs.CORE)")
    cache: dict = {}
    for run in args.run:
        done = paths.dataset(args.dataset).root / "runs" / run / "marker_pairs.json"
        if done.exists() and not args.overwrite:
            log.info("%s: %s exists, skipped", run, done)
            continue
        score_run(args.dataset, run, pairs, source, args.device, args.n_boot,
                  args.seed, cache)
    if args.kappa:
        kappa_summary(args.dataset, args.prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
