#!/usr/bin/env python3
"""Cycle read 2x2 after the lineage relabel: a read change or a model change?

Devlog "Final queue at lineage labels + composition weight 3: results
(2026-09-27 read ...)": the pooled cycle R^2 of z fell on FF (0.78 -> 0.63)
and lung (0.45 -> 0.22) while GSE and ovarian held, and FF cycle_w rose from
0.003 to 0.026 (over the 0.02 guard). Two candidate causes:

(a) READ: ``metrics.cycle_r2`` pools over the top-4 MKI67-ranked types of
    the label column the run was trained on and centres per type; a lineage
    set covers other cells, and its per-lineage centring is coarser;
(b) MODEL: composition weight 3 or the relabel moved cycle signal out of z
    (into w on FF: the over-merging risk pre-registered in "Lineage-level
    labels (R9, motivation ...)").

Rows = the old model (``final_s{k}``, old labels) and the new model
(``finalL_s{k}``, ``lineage``); columns = the read over the OLD cycling set
(top-4 old-label types, centred per old type) and over the NEW set (top-4
lineages, centred per lineage). A seed fixes the tiles, so both models of a
seed are graded on the same held-out cells and each column is one read of two
models. Every entry is the in-trainer read: ``metrics.cycle_r2`` on the
posterior means of the accepted checkpoint (``best.pt``), training-tile seeds
fit and validation-tile seeds graded, with its within-type permuted control,
and beside it the 50-PC linear reference on counts and the log-depth
baseline -- both model-free, so they move with the read alone. The diagonal
(old model / old set, new model / new set) must reproduce the stored battery
at the best epoch (asserted).

Centrings per set (the per-type centring decides how much between-type
variance is left to predict; ``cycle_r2`` removes the per-type mean of the
score and of the latent, hence of the linear prediction):

``old``      per old-label type: the old read's centring;
``lineage``  per lineage: the new read's centring, the within-lineage read;
``none``     one group: the uncentred pooled R^2, between-type variance kept.

Also per set: held-out cells covered (and MKI67+ held-out cells covered), the
share of the set's held-out score variance between lineages / between old
types; the exact split of the new-set protocol R^2 over its lineages
(variance share x within-lineage R^2, summing to the pooled value); and a
per-lineage own fit for every lineage (z and w, both models, centred per
lineage and per old type) -- which lineages carry cycle, and where FF's
cycle_w comes from.

Usage::

    python -m discell.experiments.cycle_2x2 --dataset <id>        # seeds 0-2
    python -m discell.experiments.cycle_2x2 --combine <out.md>

Writes ``experiments/cycle_2x2.{json,md}``; ``--combine`` gathers every
dataset's JSON into one markdown.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from discell.model import eval_mask as EM
from discell.model import metrics as M

log = logging.getLogger("discell.experiments.cycle_2x2")

OLD_RUN, NEW_RUN = "final_s{}", "finalL_s{}"
MODELS = ("old", "new")
SETS = ("old", "new")
CENTRINGS = ("old", "lineage", "none")
#: the set's own centring -- the read as the trainer ran it
PROTOCOL = {"old": "old", "new": "lineage"}
#: |recomputed - stored| on the diagonal: latents are re-encoded on the GPU
#: (scatter order), the model-free designs are CPU-deterministic
TOL_LATENT, TOL_COUNTS = 2e-3, 1e-4
SHORT = {"xenium_prime_human_ovary_ff": "FF",
         "xenium_prime_human_lung_cancer_ffpe": "lung",
         "gse315411_pdltma06_11_prime_solo": "GSE",
         "xenium_prime_ovarian_cancer_ffpe": "ovarian"}


def cycling_set(cycling_types, type_names) -> list[int]:
    """The trainer's cycling set: top-4 MKI67-ranked, Unassigned skipped."""
    names = [str(n) for n in type_names]
    return [int(g) for g in cycling_types if "nassigned" not in names[g]][:4]


def grouped_r2(latent, t_set, types_set, groups, scores, train,
               seed: int = 0) -> dict:
    """``metrics.cycle_r2`` on one cycling set's cells, centred per *groups*.

    *groups* None is the protocol read, bit for bit (the set's own labels,
    types in ranking order). Otherwise each kept cell is relabelled by its
    group (a constant = uncentred) and every other cell dropped.
    """
    if groups is None:
        return M.cycle_r2(latent, t_set, scores, np.asarray(types_set), train,
                          ~train, seed=seed)
    keep = np.isin(t_set, types_set)
    t_comp = np.where(keep, groups, -1)
    return M.cycle_r2(latent, t_comp, scores, np.unique(t_comp[keep]), train,
                      ~train, seed=seed)


def decompose(latent, t_set, types_set, scores, train, seed: int = 0) -> dict:
    """The protocol's pooled R^2 split over its types, exactly.

    With ``bootstrap.cycle_cells`` (the same held-out residuals and targets),
    R^2 = 1 - SSE/SST = sum_g (SST_g - SSE_g)/SST, sums over all entries of
    the two score columns around their pooled means. Per type: ``share`` =
    SST_g/SST, ``contribution`` = (SST_g - SSE_g)/SST, ``r2`` = their ratio.
    """
    from discell.experiments.bootstrap import cycle_cells

    cells = cycle_cells(latent, t_set, scores, np.asarray(types_set), train,
                        ~train, seed)
    if not len(cells["target"]):
        return {"r2_pooled": float("nan"), "by_type": {}}
    res = cells["residual"] - cells["residual"].mean()
    tgt = cells["target"] - cells["target"].mean()
    sst = float((tgt ** 2).sum())
    groups = t_set[cells["rows_test"]]
    by_type = {}
    for g in types_set:
        sel = groups == g
        sst_g, sse_g = float((tgt[sel] ** 2).sum()), float((res[sel] ** 2).sum())
        by_type[int(g)] = {"n": int(sel.sum()), "share": sst_g / sst,
                           "contribution": (sst_g - sse_g) / sst,
                           "r2": 1.0 - sse_g / sst_g if sst_g > 0 else float("nan")}
    return {"r2_pooled": 1.0 - float((res ** 2).sum()) / sst, "by_type": by_type}


def between_share(scores, groups) -> float:
    """Share of the score variance (both columns) between the *groups*' means."""
    total = ((scores - scores.mean(axis=0)) ** 2).sum()
    within = sum(((scores[groups == g] - scores[groups == g].mean(axis=0)) ** 2).sum()
                 for g in np.unique(groups))
    return float(1.0 - within / max(total, 1e-12))


def coverage(t_set, types_set, test, mki67, scores, t_old, t_lin) -> dict:
    """Held-out cells (and MKI67+ held-out cells) the set covers, and how much
    of its held-out score variance sits between lineages / old types."""
    keep = np.isin(t_set, types_set)
    held, sel = test & keep, np.flatnonzero(test & keep)
    return {"n_heldout": int(held.sum()),
            "share_heldout": float(held.sum() / max(test.sum(), 1)),
            "share_heldout_mki67pos": float((held & mki67).sum()
                                            / max((test & mki67).sum(), 1)),
            "mki67pos_fraction": float(mki67[sel].mean()) if len(sel) else float("nan"),
            "between_lineage_share": between_share(scores[sel], t_lin[sel]),
            "between_old_type_share": between_share(scores[sel], t_old[sel])}


def mki67_positive(x, gene_names) -> np.ndarray:
    names = [str(g) for g in gene_names]
    if "MKI67" not in names:
        return np.zeros(x.shape[0], dtype=bool)
    col = x[:, names.index("MKI67")]
    col = col.toarray() if hasattr(col, "toarray") else np.asarray(col)
    return np.asarray(col).ravel() > 0


def encode(dataset: str, run: str, device: str) -> dict:
    """Posterior means of the accepted checkpoint over the trainer's rows
    (training-tile seeds, then validation-tile seeds), plus what the reads
    need from the run's own assembly."""
    import torch

    from discell.experiments.at_best import battery_at_best
    from discell.model.validate import load_run

    config, data, trainer, run_dir, _ = load_run(dataset, run, device)
    train = trainer._sweep(trainer.train_batches)
    val = trainer._sweep(trainer.val_batches)
    cyc = data.cycle
    names = [str(n) for n in data.type_names]
    out = {"run": run, "label_key": config.label_key, "seed": config.seed,
           "rows": np.concatenate([train["nodes"], val["nodes"]]),
           "n_train": len(train["nodes"]),
           "mu_z": np.vstack([train["mu_z"], val["mu_z"]]),
           "mu_w": np.vstack([train["mu_w"], val["mu_w"]]),
           "t": data.t.copy(), "type_names": names,
           "types": cycling_set(cyc["cycling_types"], names),
           "scores": np.stack([cyc["s_score"], cyc["g2m_score"]], axis=1),
           "x_pcs": cyc["x_pcs"], "totals": data.totals.copy(),
           "mki67": mki67_positive(data.x, data.gene_names),
           "battery": battery_at_best(run_dir)}
    trainer.train_batches = trainer.val_batches = []
    del trainer, data
    gc.collect()
    torch.cuda.empty_cache()
    log.info("%s: %d rows encoded (label %s, cycling set %s)", run,
             len(out["rows"]), out["label_key"],
             [names[g] for g in out["types"]])
    return out


def _entry(result: dict) -> dict:
    return {"r2": result["r2_pooled"], "r2_permuted": result["r2_permuted"]}


def one_seed(dataset: str, seed: int, device: str) -> dict:
    enc = {"old": encode(dataset, OLD_RUN.format(seed), device),
           "new": encode(dataset, NEW_RUN.format(seed), device)}
    old, new = enc["old"], enc["new"]
    # one seed -> one tiling, one split, one cycle scoring, one PC basis
    assert np.array_equal(old["rows"], new["rows"]), "row orders differ"
    assert old["n_train"] == new["n_train"]
    assert np.array_equal(old["totals"], new["totals"])
    assert np.allclose(old["scores"], new["scores"])
    assert np.allclose(old["x_pcs"], new["x_pcs"], atol=1e-4)
    rows = new["rows"]
    train = np.arange(len(rows)) < new["n_train"]
    scores = new["scores"][rows]
    t = {"old": old["t"][rows], "new": new["t"][rows]}
    t_lin = t["new"]
    types = {"old": old["types"], "new": new["types"]}
    names = {"old": old["type_names"], "new": new["type_names"]}
    groups = {"old": t["old"], "lineage": t_lin,
              "none": np.zeros(len(rows), dtype=np.int64)}
    mki67 = new["mki67"][rows]
    designs = {("old", "z"): old["mu_z"], ("old", "w"): old["mu_w"],
               ("new", "z"): new["mu_z"], ("new", "w"): new["mu_w"],
               ("counts", "linear_ref"): new["x_pcs"][rows],
               ("counts", "lbaseline"): np.log(new["totals"][rows].clip(min=1.0)
                                               )[:, None].astype(np.float64)}

    reads, protocol = {}, {}
    for s in SETS:
        for c in CENTRINGS:
            for (m, latent), design in designs.items():
                result = grouped_r2(design, t[s], types[s],
                                    None if c == PROTOCOL[s] else groups[c],
                                    scores, train, seed)
                reads[f"{m}|{s}|{c}|{latent}"] = _entry(result)
                if c == PROTOCOL[s]:
                    protocol[f"{m}|{s}|{latent}"] = result

    # the diagonal reproduces the stored in-trainer battery at the best epoch
    checks = {}
    for m in MODELS:
        stored = enc[m]["battery"]
        assert stored.get("at_best"), f"{enc[m]['run']}: no at-best battery row"
        assert stored["cycle"]["types"] == types[m], (
            f"{enc[m]['run']}: cycling set {types[m]} != stored "
            f"{stored['cycle']['types']}")
        for latent, key, tol in (("z", m, TOL_LATENT), ("w", m, TOL_LATENT),
                                 ("linear_ref", "counts", TOL_COUNTS),
                                 ("lbaseline", "counts", TOL_COUNTS)):
            value = reads[f"{key}|{m}|{PROTOCOL[m]}|{latent}"]["r2"]
            want = stored["cycle"][latent]["r2_pooled"]
            diff = abs(value - want)
            assert diff <= tol, (f"{enc[m]['run']} cycle {latent}: {value:.5f} "
                                 f"!= stored {want:.5f} (|diff| {diff:.1e})")
            checks[f"{enc[m]['run']}|{latent}"] = {
                "stored": want, "recomputed": value, "abs_diff": diff,
                "epoch": stored["epoch"]}

    cover = {s: coverage(t[s], types[s], ~train, mki67, scores, t["old"], t_lin)
             for s in SETS}

    # the new-set protocol read split over its lineages (both models)
    split = {f"{m}|{latent}": decompose(designs[(m, latent)], t_lin,
                                        types["new"], scores, train, seed)
             for m in MODELS for latent in ("z", "w")}
    for key, dec in split.items():
        m, latent = key.split("|")
        assert abs(dec["r2_pooled"] - protocol[f"{m}|new|{latent}"]["r2_pooled"]) < 1e-8

    # every lineage on its own: own fit, centred per lineage / per old type
    held = ~train
    lineages = {}
    for g, name in enumerate(names["new"]):
        if EM.is_excluded(name):          # eval_mask: never a read's target
            continue
        in_g = t_lin == g
        entry = {"name": name, "n_heldout": int((in_g & held).sum()),
                 "mki67pos_fraction": float(new["mki67"][new["t"] == g].mean()),
                 "in_new_set": g in types["new"]}
        for (m, latent), design in designs.items():
            if latent == "lbaseline":
                continue
            for c, grp in (("lineage", None), ("old", t["old"])):
                result = grouped_r2(design, t_lin, [g], grp, scores, train, seed)
                entry[f"{m}|{c}|{latent}"] = _entry(result)
        lineages[name] = entry

    return {"seed": seed, "runs": {m: enc[m]["run"] for m in MODELS},
            "label_keys": {m: enc[m]["label_key"] for m in MODELS},
            "n_rows": int(len(rows)), "n_heldout": int(held.sum()),
            "sets": {s: {"types": [names[s][g] for g in types[s]],
                         "label": "old label" if s == "old" else "lineage",
                         **cover[s]} for s in SETS},
            "reads": reads,
            "new_set_split": {key: {"r2_pooled": dec["r2_pooled"],
                                    "by_lineage": {names["new"][g]: v for g, v
                                                   in dec["by_type"].items()}}
                              for key, dec in split.items()},
            "lineages": lineages, "checks": checks}


# -- summaries ---------------------------------------------------------------

def _stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "min": float(v.min()),
            "max": float(v.max()), "n": int(len(v))}


def effects(reads: dict, latent: str) -> dict:
    """The 2x2's read and model effects on one latent (per seed)."""
    r = lambda m, s: reads[f"{m}|{s}|{PROTOCOL[s]}|{latent}"]["r2"]
    return {"total (new/new - old/old)": r("new", "new") - r("old", "old"),
            "read, old model (old/new - old/old)": r("old", "new") - r("old", "old"),
            "read, new model (new/new - new/old)": r("new", "new") - r("new", "old"),
            "model, old set (new/old - old/old)": r("new", "old") - r("old", "old"),
            "model, new set (new/new - old/new)": r("new", "new") - r("old", "new")}


def summarise(per_seed: list[dict]) -> dict:
    keys = per_seed[0]["reads"]
    out = {"reads": {k: {f: _stats([s["reads"][k][f] for s in per_seed])
                         for f in ("r2", "r2_permuted")} for k in keys},
           "effects": {latent: {name: _stats([effects(s["reads"], latent)[name]
                                              for s in per_seed])
                                for name in effects(per_seed[0]["reads"], latent)}
                       for latent in ("z", "w")},
           "sets": {s: {f: _stats([p["sets"][s][f] for p in per_seed])
                        for f in ("n_heldout", "share_heldout",
                                  "share_heldout_mki67pos", "mki67pos_fraction",
                                  "between_lineage_share", "between_old_type_share")}
                    for s in SETS}}
    split = {}
    for key in per_seed[0]["new_set_split"]:
        names = per_seed[0]["new_set_split"][key]["by_lineage"]
        split[key] = {"r2_pooled": _stats([p["new_set_split"][key]["r2_pooled"]
                                           for p in per_seed]),
                      "by_lineage": {n: {f: _stats([p["new_set_split"][key]
                                                    ["by_lineage"][n][f]
                                                    for p in per_seed])
                                         for f in ("share", "contribution", "r2")}
                                     for n in names}}
    out["new_set_split"] = split
    lin = {}
    for name, entry in per_seed[0]["lineages"].items():
        lin[name] = {k: ({f: _stats([p["lineages"][name][k][f] for p in per_seed])
                          for f in ("r2", "r2_permuted")} if isinstance(v, dict)
                         else _stats([p["lineages"][name][k] for p in per_seed])
                         if k in ("n_heldout", "mki67pos_fraction") else v)
                     for k, v in entry.items()}
    out["lineages"] = lin
    return out


# -- markdown ----------------------------------------------------------------

def _fmt(stat: dict, digits: int = 3) -> str:
    if not stat or not stat.get("n"):
        return "—"
    f = f"{{:.{digits}f}}"
    if stat["n"] == 1 or stat["min"] == stat["max"]:
        return f.format(stat["mean"])
    return f"{f.format(stat['mean'])} [{f.format(stat['min'])}, {f.format(stat['max'])}]"


def two_by_two(result: dict, latent: str, digits: int = 3) -> list[str]:
    reads = result["summary"]["reads"]
    lines = [f"| cycle R² of {latent} | OLD set (old types, centred per old type) "
             "| NEW set (lineages, centred per lineage) |", "|---|---|---|"]
    for m in (MODELS if latent in ("z", "w") else ("counts",)):
        label = {"old": "old model `final_s*`", "new": "new model `finalL_s*`",
                 "counts": "(model-free)"}[m]
        cells = [_fmt(reads[f"{m}|{s}|{PROTOCOL[s]}|{latent}"]["r2"], digits)
                 for s in SETS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def dataset_markdown(result: dict) -> str:
    s = result["summary"]
    ds = result["dataset"]
    seeds = [p["seed"] for p in result["per_seed"]]
    first = result["per_seed"][0]
    L = [f"# Cycle read 2×2 — {SHORT.get(ds, ds)} (`{ds}`)", "",
         "Reads at the accepted checkpoint (R26): `best.pt` re-encoded, "
         "`metrics.cycle_r2` exactly as the trainer runs it (training-tile "
         "seeds fit, validation-tile seeds graded). Seeds "
         f"{seeds}; every entry mean [min, max] over seeds. Old model "
         f"`final_s*` (label `{first['label_keys']['old'] or 'bundle default'}`), "
         "new model `finalL_s*` (label `lineage`, composition weight 3). "
         "The diagonal reproduces the stored in-trainer battery (checks below).",
         "", "## Cycling sets and coverage", "",
         "| set | types (seed 0) | held-out cells | share of held-out | share of "
         "held-out MKI67⁺ | MKI67⁺ fraction in set | score variance between "
         "lineages | between old types |", "|---|---|---|---|---|---|---|---|"]
    for name in SETS:
        c = s["sets"][name]
        L.append(f"| {name} | {', '.join(first['sets'][name]['types'])} | "
                 f"{_fmt(c['n_heldout'], 0)} | {_fmt(c['share_heldout'])} | "
                 f"{_fmt(c['share_heldout_mki67pos'])} | "
                 f"{_fmt(c['mki67pos_fraction'])} | "
                 f"{_fmt(c['between_lineage_share'])} | "
                 f"{_fmt(c['between_old_type_share'])} |")
    L += ["", "## The 2×2 (protocol read: each set centred per its own label)", ""]
    for latent, digits in (("z", 3), ("w", 4), ("linear_ref", 3), ("lbaseline", 4)):
        L += two_by_two(result, latent, digits) + [""]
    L += ["| permuted control (latent permuted within type), z / w | OLD set "
          "| NEW set |", "|---|---|---|"]
    for m in MODELS:
        perm = lambda st, lat, d: _fmt(
            s["reads"][f"{m}|{st}|{PROTOCOL[st]}|{lat}"]["r2_permuted"], d)
        L.append(f"| {m} model | " + " | ".join(
            f"{perm(st, 'z', 4)} / {perm(st, 'w', 4)}" for st in SETS) + " |")
    L.append("")
    L += ["## Effects (per seed, then mean [min, max])", "",
          "| effect | z | w |", "|---|---|---|"]
    for name in s["effects"]["z"]:
        L.append(f"| {name} | {_fmt(s['effects']['z'][name])} | "
                 f"{_fmt(s['effects']['w'][name], 4)} |")
    L += ["", "## Centring (between-type variance kept or removed)", "",
          "`old` = per old type (the old read), `lineage` = per lineage (the "
          "within-lineage read), `none` = uncentred.", "",
          "| set | centring | z old | z new | w old | w new | linear ref |",
          "|---|---|---|---|---|---|---|"]
    for st in SETS:
        for c in CENTRINGS:
            g = lambda m, lat, d=3: _fmt(s["reads"][f"{m}|{st}|{c}|{lat}"]["r2"], d)
            mark = " (protocol)" if c == PROTOCOL[st] else ""
            L.append(f"| {st} | {c}{mark} | {g('old', 'z')} | {g('new', 'z')} | "
                     f"{g('old', 'w', 4)} | {g('new', 'w', 4)} | "
                     f"{g('counts', 'linear_ref')} |")
    L += ["", "## New-set protocol R² split over its lineages", "",
          "Exact: pooled R² = Σ contribution; contribution = variance share × "
          "within-lineage R² of the shared fit.", "",
          "| lineage | variance share | z new: R² / contribution | z old: R² / "
          "contribution | w new: R² / contribution | w old: R² / contribution |",
          "|---|---|---|---|---|---|"]
    sp = s["new_set_split"]
    for n in sp["new|z"]["by_lineage"]:
        row = [n, _fmt(sp["new|z"]["by_lineage"][n]["share"])]
        for key, d in (("new|z", 3), ("old|z", 3), ("new|w", 4), ("old|w", 4)):
            b = sp[key]["by_lineage"][n]
            row.append(f"{_fmt(b['r2'], d)} / {_fmt(b['contribution'], d)}")
        L.append("| " + " | ".join(row) + " |")
    L.append("| **pooled** | 1 | " + " | ".join(
        _fmt(sp[k]["r2_pooled"], d) for k, d in
        (("new|z", 3), ("old|z", 3), ("new|w", 4), ("old|w", 4))) + " |")
    L += ["", "## Every lineage on its own (own fit)", "",
          "Centred per lineage (`lin`) or per old type within the lineage "
          "(`old`); — = fewer than 200 training or held-out cells.", "",
          "| lineage | in new set | held-out | MKI67⁺ | z new lin | z new old | "
          "z old lin | w new lin | w new old | w old lin | linear ref lin |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, e in s["lineages"].items():
        g = lambda k, d=3: _fmt(e[k]["r2"], d)
        L.append(f"| {n} | {'yes' if e['in_new_set'] else ''} | "
                 f"{_fmt(e['n_heldout'], 0)} | {_fmt(e['mki67pos_fraction'])} | "
                 f"{g('new|lineage|z')} | {g('new|old|z')} | {g('old|lineage|z')} | "
                 f"{g('new|lineage|w', 4)} | {g('new|old|w', 4)} | "
                 f"{g('old|lineage|w', 4)} | {g('counts|lineage|linear_ref')} |")
    L += ["", "## Checks (diagonal vs stored in-trainer battery at the best epoch)",
          "", "| run · read | stored | recomputed | abs diff |", "|---|---|---|---|"]
    for p in result["per_seed"]:
        for k, c in p["checks"].items():
            L.append(f"| {k.replace('|', ' · ')} | {c['stored']:.5f} | "
                     f"{c['recomputed']:.5f} | "
                     f"{c['abs_diff']:.1e} |")
    return "\n".join(L) + "\n"


def combined_markdown(results: list[dict]) -> str:
    L = ["# Cycle read 2×2 — all datasets", "",
         "Protocol read per column (OLD set centred per old type, NEW set per "
         "lineage); mean [min, max] over seeds 0–2, accepted checkpoints. "
         "Per-dataset detail in `data/datasets/<id>/experiments/cycle_2x2.md`.", ""]
    for latent, digits in (("z", 3), ("w", 4), ("linear_ref", 3)):
        L += [f"## cycle R² of {latent}", "",
              "| dataset | model | OLD set | NEW set |", "|---|---|---|---|"]
        for r in results:
            reads = r["summary"]["reads"]
            for m in (MODELS if latent != "linear_ref" else ("counts",)):
                L.append(f"| {SHORT.get(r['dataset'], r['dataset'])} | {m} | " +
                         " | ".join(_fmt(reads[f"{m}|{s}|{PROTOCOL[s]}|{latent}"]
                                         ["r2"], digits) for s in SETS) + " |")
        L.append("")
    L += ["## Effects on z and w", "",
          "| dataset | effect | z | w |", "|---|---|---|---|"]
    for r in results:
        e = r["summary"]["effects"]
        for name in e["z"]:
            L.append(f"| {SHORT.get(r['dataset'], r['dataset'])} | {name} | "
                     f"{_fmt(e['z'][name])} | {_fmt(e['w'][name], 4)} |")
    L += ["", "## Within-lineage read on the OLD set, and uncentred", "",
          "| dataset | model | z: old-type centred (protocol) | z: lineage centred "
          "| z: uncentred | w: old-type centred | w: lineage centred |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        reads = r["summary"]["reads"]
        for m in MODELS:
            g = lambda c, lat, d=3: _fmt(reads[f"{m}|old|{c}|{lat}"]["r2"], d)
            L.append(f"| {SHORT.get(r['dataset'], r['dataset'])} | {m} | "
                     f"{g('old', 'z')} | {g('lineage', 'z')} | {g('none', 'z')} | "
                     f"{g('old', 'w', 4)} | {g('lineage', 'w', 4)} |")
    L += ["", "## Coverage", "",
          "| dataset | set | types (seed 0) | share of held-out | share of "
          "held-out MKI67⁺ | MKI67⁺ in set |", "|---|---|---|---|---|---|"]
    for r in results:
        for s in SETS:
            c = r["summary"]["sets"][s]
            L.append(f"| {SHORT.get(r['dataset'], r['dataset'])} | {s} | "
                     f"{', '.join(r['per_seed'][0]['sets'][s]['types'])} | "
                     f"{_fmt(c['share_heldout'])} | "
                     f"{_fmt(c['share_heldout_mki67pos'])} | "
                     f"{_fmt(c['mki67pos_fraction'])} |")
    return "\n".join(L) + "\n"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def run(dataset: str, seeds: Sequence[int], device: str) -> dict:
    from discell import paths

    out = paths.dataset(dataset).root / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in seeds:
        # a finished seed is kept, so a failure later (GPU memory taken by
        # another job) does not cost the seeds already read
        cache = out / f"cycle_2x2_seed{seed}.json"
        if cache.exists():
            per_seed.append(json.loads(cache.read_text()))
            continue
        record = _jsonable(one_seed(dataset, seed, device))
        cache.write_text(json.dumps(record))
        per_seed.append(record)
    result = _jsonable({"dataset": dataset, "per_seed": per_seed,
                        "summary": summarise(per_seed)})
    (out / "cycle_2x2.json").write_text(json.dumps(result, indent=1))
    (out / "cycle_2x2.md").write_text(dataset_markdown(result))
    log.info("%s -> %s", dataset, out / "cycle_2x2.{json,md}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--combine", type=Path, default=None,
                        help="write the all-dataset markdown here from the "
                             "existing per-dataset JSONs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    for dataset in args.dataset:
        run(dataset, args.seeds, args.device)
    if args.combine is not None:
        from discell import paths

        results = [json.loads(p.read_text()) for p in
                   (paths.dataset(d).root / "experiments" / "cycle_2x2.json"
                    for d in SHORT) if p.exists()]
        args.combine.parent.mkdir(parents=True, exist_ok=True)
        args.combine.write_text(combined_markdown(results))
        log.info("combined %d datasets -> %s", len(results), args.combine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
