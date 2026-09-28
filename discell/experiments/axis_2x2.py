#!/usr/bin/env python3
"""SIMVI axis test (6b.4) 2x2 after the lineage relabel: read or model?

Devlog "External criteria at finalL (lineage labels, weight 3)" (2026-09-28):
the ovarian axis test regressed -- pooled true-axis |tau| of the w-predicted
shift 0.78-0.82, false-axis genes at |tau| >= 0.9 198-1,141 -- against 0.95-
1.00 and 15-231. Those "was" numbers are the alpha_z/2 triple (``best_s*``,
2026-09-23); the warm-up re-pin ``final_s*`` (old labels) sits between the
two and was never quoted. Two things changed together at finalL:

(a) READ: ``external_criteria.axis_test`` reads one panel per class of the
    label column the run was trained on and builds the tumour band from that
    column's tumour classes. The old band counted every name containing
    "Tumor Cells" or "Malignant" (so SOX2-OT+ Tumor Cells and the cyst
    lining were tumour); the lineage band counts ``Tumour`` only
    (``labels.is_tumour``; SOX2-OT+ went to Unassigned, the lining to its
    own lineage). The false axis is the in-plane coordinate least correlated
    with the band, so it follows the band;
(b) MODEL: lineage labels + adversary composition weight 3.

Rows (models; a seed fixes the tiles, so every model of a seed is graded on
the same held-out cells): ``old`` = ``final_s{k}`` (old labels), ``new`` =
``finalL_s{k}`` (lineage), plus ``az2`` = ``best_s{k}`` (alpha_z/2, old
labels, no warm-up) as context, because the quoted baseline is that triple.

Columns (reads): a TYPE SET (which label column's classes are read, one
panel per class) x a BAND (which label column + tumour matcher builds the
band, and hence the false axis). ``old/old`` and ``new/new`` are the two
protocol reads (the 2x2); ``old/new`` and ``new/old`` are the hybrids that
split the read change into its type-set part and its band part.

Every cell is ``external_criteria.axis_test`` itself, run verbatim: the run
is loaded once, and the function is handed a shallow view of the run's own
assembly whose ``t`` / ``type_names`` are the column's and whose band is the
column's (both patched in for the call only). The model's inputs are
untouched -- it is conditioned on the labels it was trained on; only the
grouping of the read changes. Checks: the diagonal (old/old for ``old`` and
``az2``, new/new for ``new``) reproduces the stored
``external_axis_test_{final,best,finalL}_s{k}.json``; within a column the
model-free rows (raw held-out shift, l-row) are identical across models.

Usage::

    python -m discell.experiments.axis_2x2 [--dataset <id>] [--seeds 0 1 2]

Writes ``experiments/axis_2x2.{json,md}`` (per-seed caches beside them).
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import sys
from typing import Callable, Sequence
from unittest import mock

import numpy as np

from discell.experiments import external_criteria as E
from discell.experiments.cycle_2x2 import _fmt, _jsonable, _stats
from discell.model import transport as T
from discell.model.labels import is_tumour
from discell.model.validate import N_FOLDS

log = logging.getLogger("discell.experiments.axis_2x2")

DATASET = "xenium_prime_ovarian_cancer_ffpe"
#: model rows: run pattern, label column it was trained on, stored 6b.4 tag
MODELS = {"az2": ("best_s{}", "old", "best_s{}"),
          "old": ("final_s{}", "old", "final_s{}"),
          "new": ("finalL_s{}", "new", "finalL_s{}")}
LABEL_KEY = {"old": None, "new": "lineage"}
#: (type set, band); the first two are the protocol reads
COLUMNS = (("old", "old"), ("new", "new"), ("old", "new"), ("new", "old"))
PROTOCOL = {"old": "old/old", "new": "new/new"}
ROWS = ("w_predicted", "raw_observed", "z_row", "l_row")
MODEL_FREE = ("raw_observed", "l_row")
THR = "0.9"
#: |recomputed - stored| on a model row's pooled mean |tau| (the forward
#: pass is re-run on the GPU: scatter order); model-free rows must be exact
TOL_TAU = 5e-3
ASSEMBLY = ("variant", "embeddings", "tile_cells", "phi_pca", "v_pcs",
            "val_fraction", "seed", "label_key")


def old_is_tumour(name: str) -> bool:
    """The band matcher that wrote the stored old-label reads.

    ``transport.tumour_band_labels`` before ``labels.is_tumour`` (commit
    ea36e69, used for ``external_axis_test_{best,final}_s*.json``):
    case-sensitive "Tumor Cells" or "Malignant" in the name -- on the old
    ovarian vocabulary that includes SOX2-OT+ Tumor Cells and Malignant
    Cells Lining Cyst, which ``is_tumour`` on the lineage column does not."""
    return "Tumor Cells" in name or "Malignant" in name


MATCHER: dict[str, Callable[[str], bool]] = {"old": old_is_tumour,
                                             "new": is_tumour}


# -- the read ------------------------------------------------------------------

def band_labels(data, t: np.ndarray, names, matcher) -> np.ndarray:
    """``transport.tumour_band_labels`` on *data* with the labels *t* /
    *names* and the tumour matcher *matcher* (patched in for the call)."""
    view = copy.copy(data)
    view.t, view.type_names = t, np.asarray(names)
    with mock.patch.object(T, "is_tumour", matcher):
        return T.tumour_band_labels(view)


def make_column(types: dict, band: dict) -> dict:
    """One read: the classes of *types* ({t, names, p_t}), the band of
    *band* ({t, names, matcher})."""
    names = [str(n) for n in band["names"]]
    return {"t": types["t"], "names": np.asarray(types["names"]),
            "p_t": types["p_t"],
            "band": band_labels(band["data"], band["t"], names,
                                band["matcher"]),
            "tumour_classes": [n for n in names if band["matcher"](n)]}


def read_cell(model: tuple, column: dict, dataset: str, run: str) -> dict:
    """``external_criteria.axis_test`` verbatim for one loaded run under one
    column: the function sees the run's own assembly with the column's
    labels and band, and the run's own trainer and loadings."""
    config, data, trainer, run_dir, b_matrix = model
    view = copy.copy(data)
    view.t, view.type_names, view.p_t = (column["t"], column["names"],
                                         column["p_t"])
    loaded = (config, view, trainer, run_dir, b_matrix)
    with mock.patch.object(E, "load_run", lambda *a, **k: loaded), \
            mock.patch.object(E, "tumour_band_labels",
                              lambda _data: column["band"].copy()):
        return E.axis_test(argparse.Namespace(dataset=dataset, run=run,
                                              device="(loaded)"))


# -- checks ----------------------------------------------------------------

def same_rows(a: dict, b: dict, rows: Sequence[str] = MODEL_FREE) -> bool:
    """Exact equality of *rows* (pooled and per type) between two reads."""
    if set(a["types"]) != set(b["types"]):
        return False
    pairs = [(a["pooled"], b["pooled"])] + [
        (a["types"][g]["rows"], b["types"][g]["rows"]) for g in a["types"]]
    return all(x[r] == y[r] for x, y in pairs for r in rows)


def against_stored(result: dict, stored: dict) -> dict:
    """The recomputed diagonal against the stored 6b.4 JSON."""
    out = {"types_equal": sorted(result["types"]) == sorted(stored["types"]),
           "false_axis_equal": (result["false_axis"]["coordinate"]
                                == stored["false_axis"]["coordinate"]),
           "model_free_exact": same_rows(result, stored), "rows": {}}
    for row in ROWS:
        r, s = result["pooled"][row], stored["pooled"][row]
        out["rows"][row] = {
            "abs_diff_tau_true": abs(r["mean_abs_tau_true"] - s["mean_abs_tau_true"]),
            "abs_diff_tau_false": abs(r["mean_abs_tau_false"] - s["mean_abs_tau_false"]),
            "tp_fp_recomputed": [r["counts"][THR]["true_axis"],
                                 r["counts"][THR]["false_axis"]],
            "tp_fp_stored": [s["counts"][THR]["true_axis"],
                             s["counts"][THR]["false_axis"]]}
    worst = max(max(v["abs_diff_tau_true"], v["abs_diff_tau_false"])
                for v in out["rows"].values())
    out["max_abs_diff_tau"] = worst
    out["ok"] = (out["types_equal"] and out["false_axis_equal"]
                 and out["model_free_exact"] and worst <= TOL_TAU)
    return out


def leave_one_out(result: dict, row: str = "w_predicted") -> dict:
    """The pooled read with each class removed (and each class's share).

    Exact: pooled |tau| is the gene-count-weighted mean of the per-class
    means, pooled counts are the per-class sums (``axis_test`` concatenates
    the classes' per-gene taus)."""
    types = result["types"]
    n = {g: e["n_genes"] for g, e in types.items()}
    m_t = {g: e["rows"][row]["mean_abs_tau_true"] for g, e in types.items()}
    m_f = {g: e["rows"][row]["mean_abs_tau_false"] for g, e in types.items()}
    tp = {g: e["rows"][row]["counts"][THR]["true_axis"] for g, e in types.items()}
    fp = {g: e["rows"][row]["counts"][THR]["false_axis"] for g, e in types.items()}
    pooled = result["pooled"][row]
    total = sum(n.values())
    assert total == pooled["n_gene_panels"]
    assert abs(sum(n[g] * m_t[g] for g in n) / total
               - pooled["mean_abs_tau_true"]) < 1e-9
    assert sum(fp.values()) == pooled["counts"][THR]["false_axis"]
    out = {}
    for c in types:
        rest = [g for g in types if g != c]
        n_rest = sum(n[g] for g in rest)
        out[c] = {"gene_share": n[c] / total,
                  "fp_share": fp[c] / max(sum(fp.values()), 1),
                  "without": {
                      "mean_abs_tau_true": sum(n[g] * m_t[g] for g in rest) / n_rest,
                      "mean_abs_tau_false": sum(n[g] * m_f[g] for g in rest) / n_rest,
                      "tp": sum(tp[g] for g in rest),
                      "fp": sum(fp[g] for g in rest)}}
    return out


def false_axis_alignment(data, column: dict, coordinate: str,
                         classes: Sequence[str]) -> dict:
    """Model-free: how far each read class's FALSE axis already follows its band.

    ``axis_test`` cuts each class into equal-count bins of one in-plane
    coordinate, chosen as the one least correlated with the band over ALL
    cells; within a class it can still track the band. Per class (all its
    connected cells -- the w row's group means are over its training cells):
    ``tau_bins`` = Kendall tau of the class's mean band index over its false
    bins against the bin order (the same bins and tau as ``axis_test``), and
    ``spearman`` = the class's coordinate vs band. |tau_bins| near 1 means a
    predictor that is any monotone function of the band fires on the false
    axis too: that class's false-positive count is then a property of the
    read, not of w."""
    from scipy.stats import spearmanr

    dim = "xy".index(coordinate)
    names = [str(n) for n in column["names"]]
    connected = data.graph.degrees > 0
    out = {}
    for c in classes:
        members = connected & (column["t"] == names.index(c)) & (column["band"] >= 0)
        coord, band = data.positions[members, dim], column["band"][members]
        bins = E.equal_count_bins(coord, E.N_FALSE_BINS)
        means = np.array([[band[bins == b].mean()] for b in range(E.N_FALSE_BINS)])
        out[c] = {"tau_bins": float(E.kendall_tau_rows(
                      means, np.arange(E.N_FALSE_BINS, dtype=float))[0]),
                  "spearman": float(spearmanr(coord, band)[0]),
                  # in band units: ~0 when the axis is orthogonal (then
                  # tau_bins is the order of sampling noise and means nothing)
                  "band_range": float(means.max() - means.min()),
                  "mean_band_per_bin": means.ravel().tolist()}
    return out


def alignment(dataset: str, first: dict) -> dict:
    """:func:`false_axis_alignment` for every read of *first* (a seed record;
    positions, labels and bands do not depend on the seed), from CPU
    assemblies of the ``old`` and ``new`` runs' own configurations."""
    import torch

    from discell import paths
    from discell.model.prepare import assemble
    from discell.model.train import config_from_record

    root = paths.dataset(dataset).root / "runs"
    data = {}
    for key, m in (("old", "old"), ("new", "new")):
        payload = torch.load(root / first["runs"][m] / "best.pt",
                             map_location="cpu", weights_only=False)
        cfg = config_from_record(payload["config"])
        data[key] = assemble(dataset, cfg.variant, cfg.embeddings,
                             tile_cells=cfg.tile_cells, phi_pca=cfg.phi_pca,
                             v_pcs=cfg.v_pcs, val_fraction=cfg.val_fraction,
                             seed=cfg.seed, label_key=cfg.label_key)
    _aligned(data["old"], data["new"])
    labels = {k: {"t": d.t, "names": [str(n) for n in d.type_names],
                  "p_t": d.p_t, "data": d, "matcher": MATCHER[k]}
              for k, d in data.items()}
    out = {}
    for ty, bd in COLUMNS:
        name = f"{ty}/{bd}"
        col = make_column(labels[ty], labels[bd])
        read = first["reads"][f"old|{name}"]
        out[name] = {"coordinate": read["false_axis"]["coordinate"],
                     "classes": false_axis_alignment(
                         data["old"], col, read["false_axis"]["coordinate"],
                         sorted(read["types"]))}
    return out


def composition(t_from: np.ndarray, names_from, t_to: np.ndarray, names_to,
                rows: np.ndarray, classes: Sequence[str],
                floor: float = 0.01) -> dict:
    """For each read class of one label column, its make-up in the other
    column (share of its *rows*; entries below *floor* dropped)."""
    names_from = [str(n) for n in names_from]
    names_to = [str(n) for n in names_to]
    out = {}
    for c in classes:
        sel = rows & (t_from == names_from.index(c))
        counts = np.bincount(t_to[sel], minlength=len(names_to))
        share = counts / max(sel.sum(), 1)
        out[c] = {names_to[k]: float(share[k]) for k in np.argsort(-share)
                  if share[k] >= floor}
    return out


# -- one seed ----------------------------------------------------------------

def resolve_run(dataset: str, name: str) -> str:
    """The run directory under ``runs/`` (archived runs under ``_archive``)."""
    from discell import paths

    root = paths.dataset(dataset).root / "runs"
    for cand in (name, f"_archive/{name}"):
        if (root / cand / "best.pt").exists():
            return cand
    raise FileNotFoundError(f"{dataset}: no run {name} (nor _archive/{name})")


def _release(model) -> None:
    import torch

    model[2].train_batches = model[2].val_batches = []
    gc.collect()
    torch.cuda.empty_cache()


def _aligned(a, b) -> None:
    """Both label assemblies of a seed hold the same cells, counts and tiles."""
    assert a.graph.n_cells == b.graph.n_cells
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.totals, b.totals)
    assert np.array_equal(a.graph.degrees, b.graph.degrees)
    for x, y in ((a.x.indptr, b.x.indptr), (a.x.indices, b.x.indices),
                 (a.x.data, b.x.data)):
        assert np.array_equal(x, y)
    for ta, tb in ((a.train_tiles, b.train_tiles), (a.val_tiles, b.val_tiles)):
        assert len(ta) == len(tb) and all(np.array_equal(p, q)
                                          for p, q in zip(ta, tb))


def one_seed(dataset: str, seed: int, device: str) -> dict:
    from discell import paths
    from discell.model.validate import load_run

    exp = paths.dataset(dataset).root / "experiments"
    runs = {m: resolve_run(dataset, pat.format(seed))
            for m, (pat, _, _) in MODELS.items()}
    data: dict = {}
    first_config: dict = {}

    def load(m: str):
        key = MODELS[m][1]
        model = load_run(dataset, runs[m], device, data=data.get(key))
        config = model[0]
        assert config.label_key == LABEL_KEY[key], (runs[m], config.label_key)
        if key in data:            # a shared assembly: vouch that it matches
            want = first_config[key]
            got = {f: getattr(config, f) for f in ASSEMBLY}
            assert got == want, f"{runs[m]}: assembly {got} != {want}"
        else:
            data[key] = model[1]
            first_config[key] = {f: getattr(config, f) for f in ASSEMBLY}
        return model

    # one assembly per label column; the first load of "new" only provides it
    _release(load("new"))
    order = ["old", "az2", "new"]
    loaded = {"old": load("old")}
    _aligned(data["old"], data["new"])
    labels = {k: {"t": d.t, "names": [str(n) for n in d.type_names],
                  "p_t": d.p_t, "data": d, "matcher": MATCHER[k]}
              for k, d in data.items()}
    columns = {f"{ty}/{bd}": make_column(labels[ty], labels[bd])
               for ty, bd in COLUMNS}

    reads: dict = {}
    for m in order:
        model = loaded.pop(m) if m in loaded else load(m)
        for name, col in columns.items():
            reads[f"{m}|{name}"] = read_cell(model, col, dataset, runs[m])
            log.info("seed %d %s | %s: w true %.3f false %.3f, TP/FP@0.9 %d/%d",
                     seed, m, name,
                     reads[f"{m}|{name}"]["pooled"]["w_predicted"]["mean_abs_tau_true"],
                     reads[f"{m}|{name}"]["pooled"]["w_predicted"]["mean_abs_tau_false"],
                     reads[f"{m}|{name}"]["pooled"]["w_predicted"]["counts"][THR]["true_axis"],
                     reads[f"{m}|{name}"]["pooled"]["w_predicted"]["counts"][THR]["false_axis"])
        _release(model)
        del model

    # model-free rows are one read of the cells: identical across models
    for name in columns:
        base = reads[f"old|{name}"]
        for m in MODELS:
            assert same_rows(reads[f"{m}|{name}"], base), (
                f"seed {seed} column {name}: model-free rows differ ({m})")

    checks = {}
    for m, (_, key, tag) in MODELS.items():
        stored = json.loads((exp / f"external_axis_test_{tag.format(seed)}.json")
                            .read_text())
        chk = against_stored(reads[f"{m}|{PROTOCOL[key]}"], stored)
        checks[m] = chk
        assert chk["ok"], f"seed {seed} {m}: diagonal != stored ({chk})"

    # the read's held-out cells: fold 0 of validate.collect_latents' folds
    fold = np.zeros(data["old"].graph.n_cells, dtype=np.int64)
    for k, tile in enumerate(data["old"].train_tiles + data["old"].val_tiles):
        fold[tile] = k % N_FOLDS
    held = (fold == 0) & (data["old"].graph.degrees > 0)
    make_up = {
        "old": composition(labels["old"]["t"], labels["old"]["names"],
                           labels["new"]["t"], labels["new"]["names"], held,
                           sorted({g for k, r in reads.items()
                                   if k.split("|")[1].startswith("old/")
                                   for g in r["types"]})),
        "new": composition(labels["new"]["t"], labels["new"]["names"],
                           labels["old"]["t"], labels["old"]["names"], held,
                           sorted({g for k, r in reads.items()
                                   if k.split("|")[1].startswith("new/")
                                   for g in r["types"]}))}
    bands = {}
    for bd in ("old", "new"):
        b = columns[f"{bd}/{bd}"]["band"]
        bands[bd] = {"tumour_classes": columns[f"{bd}/{bd}"]["tumour_classes"],
                     "cells_per_band": np.bincount(b[b >= 0]).tolist(),
                     "heldout_cells_per_band": np.bincount(b[held & (b >= 0)]).tolist()}
    agree = columns["old/old"]["band"] == columns["new/new"]["band"]
    bands["same_band_share"] = float(agree[data["old"].graph.degrees > 0].mean())

    return {"seed": seed, "runs": runs, "reads": reads, "checks": checks,
            "composition": make_up, "bands": bands,
            "loo": {k: leave_one_out(r) for k, r in reads.items()}}


# -- summaries ---------------------------------------------------------------

def _w(read: dict, row: str = "w_predicted") -> dict:
    p = read["pooled"][row]
    return {"tau_true": p["mean_abs_tau_true"], "tau_false": p["mean_abs_tau_false"],
            "tp": p["counts"][THR]["true_axis"], "fp": p["counts"][THR]["false_axis"],
            "n": p["n_gene_panels"]}


def effects(reads: dict) -> dict:
    """Differences of the w row's pooled read between cells of the grid."""
    def d(a, b):
        va, vb = _w(reads[a]), _w(reads[b])
        return {k: va[k] - vb[k] for k in ("tau_true", "tau_false", "tp", "fp")}
    return {
        "total: new model, new read - old model, old read": d("new|new/new", "old|old/old"),
        "read, old model (new/new - old/old)": d("old|new/new", "old|old/old"),
        "read, new model (new/new - old/old)": d("new|new/new", "new|old/old"),
        "model, old read (new - old)": d("new|old/old", "old|old/old"),
        "model, new read (new - old)": d("new|new/new", "old|new/new"),
        "warm-up step, old read (old - az2)": d("old|old/old", "az2|old/old"),
        "warm-up step, new read (old - az2)": d("old|new/new", "az2|new/new"),
        "band only, old types, new model (old/new - old/old)": d("new|old/new", "new|old/old"),
        "band only, old types, old model (old/new - old/old)": d("old|old/new", "old|old/old"),
        "types only, old band, new model (new/old - old/old)": d("new|new/old", "new|old/old"),
        "types only, old band, old model (new/old - old/old)": d("old|new/old", "old|old/old"),
    }


def summarise(per_seed: list[dict]) -> dict:
    keys = list(per_seed[0]["reads"])
    pooled = {k: {row: {f: _stats([_w(s["reads"][k], row)[f] for s in per_seed])
                        for f in ("tau_true", "tau_false", "tp", "fp", "n")}
                  for row in ROWS} for k in keys}
    eff0 = effects(per_seed[0]["reads"])
    eff = {name: {f: _stats([effects(s["reads"])[name][f] for s in per_seed])
                  for f in eff0[name]} for name in eff0}
    by_type: dict = {}
    for k in keys:
        classes = sorted({g for s in per_seed for g in s["reads"][k]["types"]})
        by_type[k] = {}
        for g in classes:
            got = [s["reads"][k]["types"][g] for s in per_seed
                   if g in s["reads"][k]["types"]]
            by_type[k][g] = {"n_seeds": len(got), "rows": {}}
            by_type[k][g]["n_genes"] = _stats([e["n_genes"] for e in got])
            for row in ROWS:
                r = [e["rows"][row] for e in got]
                by_type[k][g]["rows"][row] = {
                    "tau_true": _stats([x["mean_abs_tau_true"] for x in r]),
                    "tau_false": _stats([x["mean_abs_tau_false"] for x in r]),
                    "tp": _stats([x["counts"][THR]["true_axis"] for x in r]),
                    "fp": _stats([x["counts"][THR]["false_axis"] for x in r])}
    loo: dict = {}
    for k in keys:
        loo[k] = {}
        for g in by_type[k]:
            got = [s["loo"][k][g] for s in per_seed if g in s["loo"][k]]
            loo[k][g] = {"gene_share": _stats([x["gene_share"] for x in got]),
                         "fp_share": _stats([x["fp_share"] for x in got]),
                         **{f"without_{f}": _stats([x["without"][f] for x in got])
                            for f in ("mean_abs_tau_true", "mean_abs_tau_false",
                                      "tp", "fp")}}
    return {"pooled": pooled, "effects": eff, "by_type": by_type, "loo": loo}


# -- markdown ----------------------------------------------------------------

LABEL = {"az2": "α_z/2 `best_s*` (old labels, no warm-up) — context",
         "old": "old model `final_s*` (old labels)",
         "new": "new model `finalL_s*` (lineage, weight 3)"}
COLNAME = {"old/old": "old types / old band", "new/new": "lineages / lineage band",
           "old/new": "old types / lineage band", "new/old": "lineages / old band"}


def _pair(a: dict, b: dict, digits: int = 3) -> str:
    return f"{_fmt(a, digits)} / {_fmt(b, digits)}"


def markdown(result: dict) -> str:
    s = result["summary"]
    per = result["per_seed"]
    first = per[0]
    seeds = [p["seed"] for p in per]
    cols = [f"{a}/{b}" for a, b in COLUMNS]
    L = [f"# Axis test (6b.4) 2×2 — ovarian (`{result['dataset']}`)", "",
         "`external_criteria.axis_test` run verbatim on each (model, read) cell: "
         "per class, Kendall τ of the w-predicted per-band shift against the ordered "
         "tumour-band index (true axis) and against equal-count bins of the in-plane "
         "coordinate least correlated with the band (false axis), held-out tiles "
         "(fold 0 of the seed's tiling, shared by every model of a seed). "
         f"Seeds {seeds}; every entry mean [min, max] over seeds. "
         f"TP / FP = gene-panels with |τ| ≥ {THR} on the true / false axis.", "",
         "Reads (columns): a type set (whose classes get one panel each) × a band "
         "(whose tumour classes build the band; the false axis follows the band).", ""]
    for bd in ("old", "new"):
        b = first["bands"][bd]
        L.append(f"- **{bd} band** tumour classes: {', '.join(b['tumour_classes'])}; "
                 f"cells per band (seed {first['seed']}, all connected): "
                 f"{b['cells_per_band']}; held-out {b['heldout_cells_per_band']}.")
    L.append(f"- The two bands agree on {first['bands']['same_band_share']:.3f} of "
             "connected cells (seed 0).")
    false_axis = {c: first["reads"][f"old|{c}"]["false_axis"] for c in cols}
    L.append("- False axis per read (seed 0): " + "; ".join(
        f"{COLNAME[c]} → {v['coordinate']} (|Spearman| with band x {v['spearman_with_band']['x']:.3f}, "
        f"y {v['spearman_with_band']['y']:.3f})" for c, v in false_axis.items()) + ".")
    L += ["", "## The 2×2 on the w row (protocol reads)", "",
          "| model | old types / old band: τ true / τ false | TP / FP | "
          "lineages / lineage band: τ true / τ false | TP / FP |",
          "|---|---|---|---|---|"]
    for m in MODELS:
        cells = []
        for c in ("old/old", "new/new"):
            p = s["pooled"][f"{m}|{c}"]["w_predicted"]
            cells += [_pair(p["tau_true"], p["tau_false"]),
                      _pair(p["tp"], p["fp"], 0)]
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    L += ["", "## Hybrid reads on the w row (split the read change)", "",
          "| model | " + " | ".join(f"{COLNAME[c]}: τ true / false; TP / FP" for c in cols)
          + " |", "|---|" + "---|" * len(cols)]
    for m in MODELS:
        cells = []
        for c in cols:
            p = s["pooled"][f"{m}|{c}"]["w_predicted"]
            cells.append(f"{_pair(p['tau_true'], p['tau_false'])}; "
                         f"{_pair(p['tp'], p['fp'], 0)}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Effects on the w row (per seed, then mean [min, max])", "",
          "| effect | Δ τ true | Δ τ false | Δ TP | Δ FP |", "|---|---|---|---|---|"]
    for name, e in s["effects"].items():
        L.append(f"| {name} | {_fmt(e['tau_true'])} | {_fmt(e['tau_false'])} | "
                 f"{_fmt(e['tp'], 0)} | {_fmt(e['fp'], 0)} |")
    L += ["", "## Reference rows", "",
          "raw held-out shift and ℓ-row are model-free (asserted identical across "
          "models within a read); the z row is the model's decontaminated rate "
          "minus its w path.", "",
          "| read | row | model | τ true / τ false | TP / FP |", "|---|---|---|---|---|"]
    for c in cols:
        for row in ("raw_observed", "l_row"):
            p = s["pooled"][f"old|{c}"][row]
            L.append(f"| {COLNAME[c]} | {row} | (model-free) | "
                     f"{_pair(p['tau_true'], p['tau_false'])} | "
                     f"{_pair(p['tp'], p['fp'], 0)} |")
        for m in MODELS:
            p = s["pooled"][f"{m}|{c}"]["z_row"]
            L.append(f"| {COLNAME[c]} | z_row | {m} | "
                     f"{_pair(p['tau_true'], p['tau_false'])} | "
                     f"{_pair(p['tp'], p['fp'], 0)} |")
    L += ["", "## Per class (w row)", "",
          "Each class's own panel; pooled = gene-weighted mean of these. "
          "`n` = genes scored in the class. **false-axis alignment** (model-free, "
          "all connected cells of the class): Kendall τ of the class's mean band "
          "index over its false-axis bins vs bin order; the spread of those means "
          "(band units); Spearman(coordinate, band) within the class. A large "
          "spread with |τ| near 1 means the false axis is not orthogonal for that "
          "class: any band-following predictor fires on it.", ""]
    for c in cols:
        align = result.get("alignment", {}).get(c, {}).get("classes", {})
        L += [f"### {COLNAME[c]}", "",
              "| class | n | false-axis alignment τ; spread; ρ | "
              + " | ".join(f"{m}: τ true / false; TP / FP" for m in MODELS)
              + " |", "|---|---|---|" + "---|" * len(MODELS)]
        for g in s["by_type"][f"old|{c}"]:
            a = align.get(g)
            row = [g, _fmt(s["by_type"][f"old|{c}"][g]["n_genes"], 0),
                   f"{a['tau_bins']:+.2f}; {a['band_range']:.2f}; {a['spearman']:+.2f}"
                   if a else "—"]
            for m in MODELS:
                e = s["by_type"][f"{m}|{c}"].get(g)
                if e is None:
                    row.append("—")
                    continue
                w = e["rows"]["w_predicted"]
                seeds_note = "" if e["n_seeds"] == len(per) else f" ({e['n_seeds']} seeds)"
                row.append(f"{_pair(w['tau_true'], w['tau_false'])}; "
                           f"{_pair(w['tp'], w['fp'], 0)}{seeds_note}")
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    L += ["## Pooled w read with one class left out", "",
          "Exact (pooled = gene-weighted mean / sum of the classes).", "",
          "| read | model | class left out | its gene share | its FP share | "
          "τ true / τ false without it | TP / FP without it |",
          "|---|---|---|---|---|---|---|"]
    for c in ("old/old", "new/new"):
        for m in ("old", "new"):
            for g, e in s["loo"][f"{m}|{c}"].items():
                L.append(f"| {COLNAME[c]} | {m} | {g} | {_fmt(e['gene_share'])} | "
                         f"{_fmt(e['fp_share'])} | "
                         f"{_pair(e['without_mean_abs_tau_true'], e['without_mean_abs_tau_false'])} | "
                         f"{_pair(e['without_tp'], e['without_fp'], 0)} |")
    L += ["", "## What each read class is in the other vocabulary", "",
          f"Share of the class's connected held-out cells (seed {first['seed']}; ≥ 1 %).", ""]
    for key, other in (("old", "lineage"), ("new", "old label")):
        for g, comp in first["composition"][key].items():
            L.append(f"- {g} → " + ", ".join(f"{k} {v:.2f}" for k, v in comp.items())
                     + f" ({other})")
    L += ["", "## Checks", "",
          "Diagonal vs the stored `external_axis_test_<run>.json` (model rows "
          f"within {TOL_TAU:g} on pooled mean |τ|; model-free rows exact); model-free "
          "rows identical across models within every read (asserted).", "",
          "| seed | model | types equal | false axis equal | model-free exact | "
          "max |Δ τ| | w TP/FP recomputed | stored |", "|---|---|---|---|---|---|---|---|"]
    for p in per:
        for m, c in p["checks"].items():
            w = c["rows"]["w_predicted"]
            L.append(f"| {p['seed']} | {p['runs'][m]} | {c['types_equal']} | "
                     f"{c['false_axis_equal']} | {c['model_free_exact']} | "
                     f"{c['max_abs_diff_tau']:.1e} | {w['tp_fp_recomputed']} | "
                     f"{w['tp_fp_stored']} |")
    return "\n".join(L) + "\n"


def run(dataset: str, seeds: Sequence[int], device: str) -> dict:
    from discell import paths

    out = paths.dataset(dataset).root / "experiments"
    per_seed = []
    for seed in seeds:
        cache = out / f"axis_2x2_seed{seed}.json"
        if cache.exists():
            per_seed.append(json.loads(cache.read_text()))
            continue
        record = _jsonable(one_seed(dataset, seed, device))
        cache.write_text(json.dumps(record))
        per_seed.append(record)
    cache = out / "axis_2x2_alignment.json"
    if not cache.exists():
        cache.write_text(json.dumps(_jsonable(alignment(dataset, per_seed[0]))))
    result = _jsonable({"dataset": dataset, "per_seed": per_seed,
                        "summary": summarise(per_seed),
                        "alignment": json.loads(cache.read_text())})
    (out / "axis_2x2.json").write_text(json.dumps(result, indent=1))
    (out / "axis_2x2.md").write_text(markdown(result))
    log.info("%s -> %s", dataset, out / "axis_2x2.{json,md}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    run(args.dataset, args.seeds, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
