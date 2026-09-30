#!/usr/bin/env python3
"""What each method's *context* latent carries: the positive control.

Devlog "Context-latent grading for every method (motivation, 2026-09-30;
author)". The battery grades every method on its intrinsic latent only; this
grades the other latent -- DISCELL's mu_w, SIMVI's spatial latent, MintFlow's
microenvironment latent (``X_mintflow_micro_in``), Cellina's s -- with the
same machinery, on the same cells and targets as the probe regrade
(``probe_regrade.py``). resolVI has a single cell latent and no context
latent: it is recorded as absent, not as zero. Three reads per latent:

``probe``    :func:`metrics.probe_blocks` run in the positive direction: the
             held-out ridge and MLP V-information gain for the composition
             and image blocks, each against its within-type permutation
             floor; ``var_fraction`` = exp(2 excess) - 1 is the share of the
             block's within-type variance recovered (as tab:probe).
``niche_mi`` :func:`degeneracy.w_channel_guard`: I(niche; latent) by the
             Ross kNN estimator over the within-type permutation floor,
             K = 10 composition niches (``validate.niche_labels``, the
             split's seed, as the w guard). The estimator depends on the
             latent's width, so this read is not compared across widths.
``nmi``      k-means over the context latent against the lineage type, as
             the battery's NMI: how much of the latent is type, not context.

Two steps per section, so the k-means niches are computed at the thread
count the w guard was read at (``OMP_NUM_THREADS=8``) and the rest at 2::

    OMP_NUM_THREADS=8 python -m discell.experiments.context_grade niches \\
        --dataset D [--config-from D_solo]
    OMP_NUM_THREADS=2 python -m discell.experiments.context_grade grade \\
        --dataset D [--config-from D_solo]

CPU only (the job hides the GPUs). Artefacts under
``data/datasets/<D>/experiments/``: ``context_grade/`` (the niche labels and
DISCELL's node-ordered mu_w, cached) and ``context_grade.{json,md}``. Nothing
existing is read differently or rewritten.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from discell import paths
from discell.model import eval_mask as EM
from discell.model import metrics as M

log = logging.getLogger("discell.context_grade")

RUNS = ("finalL_s0", "finalL_s1", "finalL_s2")
#: the probe regrade's baseline records: which latents, graded on which split
BASELINE_DIR = "probe_regrade_lineage"
#: each tool's context latent (devlog mapping, 2026-09-30); resolVI has none
CONTEXT_KEYS = {"resolvi": None, "simvi": "X_simvi_spatial",
                "mintflow": "X_mintflow_micro_in",
                "cellina": "X_cellina_spatial"}
#: the w guard's tolerance against degeneracy.json (paper_tables.MI_TOL)
MI_TOL = 5e-4
N_PERM = 5


def label(method: str) -> str:
    """The comparison method's name in the tables, from its record name."""
    m = method.lower()
    for key, name in (("cellina_nicheadv", "Cellina, niche domain"),
                      ("cellina_owngraph", "Cellina, own graph"),
                      ("cellina", "Cellina"), ("simvi", "SIMVI"),
                      ("mintflow", "MintFlow"), ("resolvi", "resolVI")):
        if m.startswith(key):
            return name
    return method


# -- the reads, on plain arrays ---------------------------------------------

def grade(latent: np.ndarray, t: np.ndarray, train: np.ndarray,
          test: np.ndarray, v_block: np.ndarray, vbar_t: np.ndarray,
          n_comp: int, niche: np.ndarray, type_names, seed: int = 0,
          n_perm: int = N_PERM) -> dict:
    """The three reads of one context latent; no IO, no model state.

    Rows are the cells the method has a latent for, in the order the probe
    regrade grades them (for DISCELL: training-tile seeds, then held-out
    tiles, so the held-out rows are in the w guard's order). *niche* is the
    composition niche per row (negative = isolated).
    """
    from discell.model.degeneracy import w_channel_guard

    latent = np.asarray(latent, dtype=np.float64)
    on = EM.metric_target_mask(t, type_names)
    probe = M.probe_blocks(latent[on], t[on], v_block[on], vbar_t, train[on],
                           test[on], n_comp=n_comp, seed=seed, n_perm=n_perm)
    for family in ("ridge", "mlp"):          # per-column detail: not needed
        for block in ("comp", "img", "pooled"):
            probe[family][block].pop("gain_per_col", None)
    held = on & test
    guard = w_channel_guard(latent[held], niche[held], t[held], seed=seed)
    nmi = M.z_type_nmi(*EM.nmi_inputs(latent, t, type_names), seed=seed)
    return {"width": int(latent.shape[1]), "n_cells": int(on.sum()),
            "n_heldout": int(held.sum()), "probe": probe,
            "niche_mi": guard, "nmi": float(nmi),
            "eval_mask": EM.record(type_names, t)}


# -- inputs -----------------------------------------------------------------

def _root(dataset: str) -> Path:
    return paths.dataset(dataset).root / "experiments"


def niche_path(dataset: str, seed: int) -> Path:
    return _root(dataset) / "context_grade" / f"niches_s{seed}.npz"


def mu_w_path(dataset: str, run: str) -> Path:
    return _root(dataset) / "context_grade" / f"mu_w_{run}.npz"


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_niches(dataset: str, seed: int, n_cells: int) -> np.ndarray:
    path = niche_path(dataset, seed)
    if not path.exists():
        raise FileNotFoundError(f"{path}: run the 'niches' step first "
                                "(OMP_NUM_THREADS=8)")
    labels = np.load(path)["labels"]
    if len(labels) != n_cells:
        raise ValueError(f"{path} has {len(labels)} cells, not {n_cells}")
    return labels


def encode_w(dataset: str, run: str, config_from: str | None, assemblies,
             device: str = "cpu") -> np.ndarray:
    """Node-ordered mu_w of *run*'s accepted checkpoint (``best.pt``) on
    *dataset*'s cells: ``probe_regrade.encode`` with mu_w kept."""
    import torch

    from discell.experiments.probe_regrade import run_config
    from discell.model.train import Trainer
    from discell.model.validate import collect_latents, load_run

    source = config_from or dataset
    cfg = run_config(source, run)
    data_a = assemblies.get(source, cfg)
    config, data_a, trainer, _, _ = load_run(source, run, device, data=data_a)
    if source != dataset:            # the crossslide protocol
        data_b = assemblies.get(dataset, cfg)
        if ([str(x) for x in data_a.type_names]
                != [str(x) for x in data_b.type_names]):
            raise ValueError(f"type vocabularies differ: {source} / {dataset}")
        model = trainer.model
        trainer.train_batches, trainer.val_batches = [], []
        del trainer
        trainer = Trainer(config, data_b)
        trainer.model = model.to(trainer.device).eval()
        data_a = data_b
    with torch.no_grad():
        mu_w = collect_latents(trainer, data_a)["mu_w"].astype(np.float32)
    return mu_w


def load_context(path: str | Path):
    """(tool, obsm key or None, rows, context latent or None) of a tool's
    ``latents.h5ad``; the tool is recognised by its intrinsic key."""
    import anndata as ad

    from discell.experiments.baseline_battery import LATENT_KEYS

    adata = ad.read_h5ad(path)
    tool = next(k for k, (intrinsic, _) in LATENT_KEYS.items()
                if intrinsic in adata.obsm)
    rows = np.array([int(str(s).split("_")[-1]) for s in adata.obs_names],
                    dtype=np.int64)
    key = CONTEXT_KEYS[tool]
    if key is None:
        return tool, None, rows, None
    return tool, key, rows, np.asarray(adata.obsm[key])


def baseline_records(dataset: str) -> list[dict]:
    """The probe regrade's baseline records of *dataset*, one per method label
    (the serial section holds Cellina twice, under two names, same file)."""
    root = _root(dataset) / BASELINE_DIR
    out: dict[str, dict] = {}
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        rec = json.loads(path.read_text())
        name = label(rec["method"])
        if name in out:
            if out[name]["latents"] != rec["latents"]:
                raise ValueError(f"{dataset}: two {name} records with "
                                 "different latents")
            out[name]["entries"].append(rec["method"])
            continue
        out[name] = {"label": name, "entries": [rec["method"]],
                     "latents": rec["latents"], "split_run": rec["split_run"],
                     "split_from": rec["split_from"], "record": str(path),
                     "intrinsic_probe": {f"{f}_{b}": rec[f][b]["var_fraction"]
                                         for f, b in M.GUARD_BLOCKS}}
    return list(out.values())


def intrinsic_record(dataset: str, run: str, config_from: str | None) -> Path:
    from discell.experiments.probe_regrade import record_path

    return record_path(dataset, run, config_from)


# -- the two steps ----------------------------------------------------------

def step_niches(dataset: str, config_from: str | None, k: int = 10) -> None:
    """K-means composition niches for each split seed used on *dataset*."""
    from discell.experiments.probe_regrade import Assemblies, run_config
    from discell.model.validate import niche_labels

    source = config_from or dataset
    assemblies = Assemblies()
    for run in RUNS:
        cfg = run_config(source, run)
        path = niche_path(dataset, cfg["seed"])
        if path.exists():
            log.info("%s: %s exists -- skipped", dataset, path.name)
            continue
        data = assemblies.get(dataset, cfg)
        started = time.time()
        labels = niche_labels(data, k, cfg["seed"])
        _save(path, labels=labels, k=k, seed=cfg["seed"],
              omp_num_threads=str(os.environ.get("OMP_NUM_THREADS")))
        log.info("%s: niches seed %d (K=%d, OMP %s) in %.0fs -> %s", dataset,
                 cfg["seed"], k, os.environ.get("OMP_NUM_THREADS"),
                 time.time() - started, path)


def _trainer_order(data) -> tuple[np.ndarray, np.ndarray]:
    """``probe_blocks_for_run``'s rows: training tiles, then held-out tiles."""
    rows = np.concatenate(data.train_tiles + data.val_tiles)
    n_train = sum(len(tile) for tile in data.train_tiles)
    return rows, np.arange(len(rows)) < n_train


def discell_entry(dataset: str, run: str, config_from: str | None,
                  assemblies, n_perm: int) -> dict:
    from discell.experiments.probe_regrade import run_config

    source = config_from or dataset
    cfg = run_config(source, run)
    data = assemblies.get(dataset, cfg)
    cache = mu_w_path(dataset, run)
    if cache.exists() and len(np.load(cache)["mu_w"]) == data.n_cells:
        mu_w = np.load(cache)["mu_w"]
    else:
        started = time.time()
        mu_w = encode_w(dataset, run, config_from, assemblies)
        _save(cache, mu_w=mu_w)
        log.info("%s/%s: mu_w encoded on CPU in %.0fs", dataset, run,
                 time.time() - started)
        data = assemblies.get(dataset, cfg)
    niche = load_niches(dataset, cfg["seed"], data.n_cells)
    rows, train = _trainer_order(data)
    out = grade(mu_w[rows], data.t[rows], train, ~train, data.v_block[rows],
                data.vbar_t, data.n_comp, niche[rows], data.type_names,
                seed=cfg["seed"], n_perm=n_perm)
    out.update(run=run, trained_on=source, checkpoint="best.pt",
               latent="mu_w", split_run=run, split_seed=int(cfg["seed"]))
    # the intrinsic latent's composition read of the same fit (tab:probe)
    rec = intrinsic_record(dataset, run, config_from)
    if rec.exists():
        r = json.loads(rec.read_text())
        out["intrinsic_probe"] = {f"{f}_{b}": r[f][b]["var_fraction"]
                                  for f, b in M.GUARD_BLOCKS}
        out["intrinsic_record"] = str(rec)
    # the check: the w guard of degeneracy.json, same run, same cells
    deg = paths.dataset(source).root / "runs" / run / "degeneracy.json"
    if source == dataset and deg.exists():
        stored = json.loads(deg.read_text())["w_channel"]
        diff = out["niche_mi"]["w_niche_mi_excess"] - stored["w_niche_mi_excess"]
        out["check_degeneracy"] = {
            "stored_excess": stored["w_niche_mi_excess"],
            "stored_n_cells": stored["n_cells"], "diff": float(diff),
            "tolerance": MI_TOL, "ok": bool(abs(diff) <= MI_TOL)}
        log.info("%s/%s: I(niche; w) excess %.4f vs degeneracy.json %.4f "
                 "(diff %+.1e, %s)", dataset, run,
                 out["niche_mi"]["w_niche_mi_excess"],
                 stored["w_niche_mi_excess"], diff,
                 "ok" if abs(diff) <= MI_TOL else "OUTSIDE TOLERANCE")
    return out


def baseline_entry(dataset: str, base: dict, assemblies, n_perm: int) -> dict:
    """``probe_regrade.grade_baseline``'s cells and targets, context latent."""
    from discell.experiments.probe_regrade import run_config

    tool, key, rows, ctx = load_context(base["latents"])
    info = {k: base[k] for k in ("label", "entries", "latents", "split_run",
                                 "split_from", "record", "intrinsic_probe")}
    info.update(tool=tool, key=key)
    if ctx is None:
        info.update(absent=True, why="single cell latent: no context latent")
        return info
    cfg = run_config(base["split_from"], base["split_run"])
    data = assemblies.get(dataset, cfg)
    if rows.max() >= data.n_cells:
        raise ValueError("latents carry cell ids outside the dataset")
    test = np.zeros(data.n_cells, dtype=bool)
    for tile in data.val_tiles:
        test[tile] = True
    niche = load_niches(dataset, cfg["seed"], data.n_cells)
    info.update(grade(ctx, data.t[rows], ~test[rows], test[rows],
                      data.v_block[rows], data.vbar_t, data.n_comp,
                      niche[rows], data.type_names, seed=cfg["seed"],
                      n_perm=n_perm), split_seed=int(cfg["seed"]))
    return info


def step_grade(dataset: str, config_from: str | None, n_perm: int = N_PERM,
               only: list[str] | None = None) -> dict:
    from discell.experiments.probe_regrade import Assemblies

    source = config_from or dataset
    js = _root(dataset) / "context_grade.json"
    result = json.loads(js.read_text()) if js.exists() else {
        "dataset": dataset, "config_from": source, "methods": {}}
    assemblies = Assemblies(capacity=1 if source == dataset else 2)
    methods = result["methods"]
    want = lambda name: not only or name in only      # noqa: E731
    if want("DISCELL"):
        runs = {}
        for run in RUNS:
            started = time.time()
            runs[run] = discell_entry(dataset, run, config_from, assemblies,
                                      n_perm)
            runs[run]["minutes"] = (time.time() - started) / 60
            log.info("%s DISCELL/%s: %s", dataset, run, summary(runs[run]))
        methods["DISCELL"] = {"label": "DISCELL", "latent": "mu_w",
                              "width": runs[RUNS[0]]["width"], "runs": runs}
        _write(dataset, result)
    for base in baseline_records(dataset):
        if not want(base["label"]):
            continue
        started = time.time()
        entry = baseline_entry(dataset, base, assemblies, n_perm)
        entry["minutes"] = (time.time() - started) / 60
        methods[base["label"]] = entry
        log.info("%s %s: %s", dataset, base["label"], summary(entry))
        _write(dataset, result)
    return result


# -- artefacts --------------------------------------------------------------

def summary(e: dict) -> str:
    if e.get("absent"):
        return "no context latent"
    p, g = e["probe"], e["niche_mi"]
    return (f"width {e['width']}; comp mlp {100 * p['mlp']['comp']['var_fraction']:.1f} % "
            f"ridge {100 * p['ridge']['comp']['var_fraction']:.1f} %; img mlp "
            f"{100 * p['mlp']['img']['var_fraction']:.1f} % ridge "
            f"{100 * p['ridge']['img']['var_fraction']:.1f} %; I(niche) excess "
            f"{g['w_niche_mi_excess']:.3f}; NMI {e['nmi']:.3f}")


def _entries(methods: dict):
    for name, m in methods.items():
        if "runs" in m:
            for run, e in m["runs"].items():
                yield f"{name}/{run}", e
        else:
            yield name, m


def markdown(result: dict) -> str:
    head = ["method", "width", "comp MLP %", "comp ridge %", "img MLP %",
            "img ridge %", "I(niche; ·) excess (floor)", "NMI(·, type)",
            "comp MLP % of the intrinsic latent"]
    lines = [f"# Context-latent grading -- {result['dataset']}", "",
             "Generated by `discell/experiments/context_grade.py` (devlog "
             "2026-09-30, \"Context-latent grading for every method\"). Each "
             "method's context latent on the probe regrade's cells and "
             "targets. Percentages: the within-type variance of the block "
             "recovered beyond the permutation floor, exp(2 excess) - 1. "
             "I(niche; ·): Ross kNN estimate over the within-type floor, "
             "K = 10 composition niches; it depends on the width and is not "
             "compared across widths.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    pc = lambda e, f, b: f"{100 * e['probe'][f][b]['var_fraction']:.2f}"  # noqa: E731
    for name, e in _entries(result["methods"]):
        if e.get("absent"):
            lines.append(f"| {name} | -- | " + " | ".join(
                ["absent (no context latent)"] + ["--"] * 7) + " |")
            continue
        g = e["niche_mi"]
        ip = e.get("intrinsic_probe", {}).get("mlp_comp")
        lines.append("| " + " | ".join(
            [name, str(e["width"]), pc(e, "mlp", "comp"),
             pc(e, "ridge", "comp"), pc(e, "mlp", "img"),
             pc(e, "ridge", "img"),
             f"{g['w_niche_mi_excess']:.3f} ({g['w_niche_mi_floor']:.3f})",
             f"{e['nmi']:.3f}",
             "--" if ip is None else f"{100 * ip:.2f}"]) + " |")
    checks = [(name, e["check_degeneracy"]) for name, e
              in _entries(result["methods"]) if "check_degeneracy" in e]
    if checks:
        lines += ["", "Check against each run's degeneracy.json w guard "
                  f"(tolerance {MI_TOL}): " + "; ".join(
                      f"{n} {c['diff']:+.1e} {'ok' if c['ok'] else 'FAIL'}"
                      for n, c in checks) + "."]
    return "\n".join(lines) + "\n"


def _write(dataset: str, result: dict) -> None:
    from discell.experiments.probe_regrade import write_atomic

    root = _root(dataset)
    write_atomic(root / "context_grade.json",
                 json.dumps(result, indent=1, default=float))
    write_atomic(root / "context_grade.md", markdown(result))


# -- CLI --------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("niches", "grade"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--config-from", default=None,
                   help="dataset the DISCELL runs were trained on "
                        "(the held-out-section protocol)")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--only", nargs="*", default=None,
                   help="grade these method labels only (e.g. DISCELL)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.step == "niches":
        step_niches(args.dataset, args.config_from)
        return 0
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or 2))
    step_grade(args.dataset, args.config_from, args.n_perm, args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
