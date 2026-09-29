#!/usr/bin/env python3
"""Re-read the cycle rows on the top-decile cycling set, no refits (2026-09-28).

Author's decision 2026-09-28: the cycle reads (pooled R^2 of z and of w,
centred within lineage, the linear count reference beside) pool over the top
decile of the S + G2M score among a slide's held-out non-Unassigned cells --
``cell_cycle.cycling_set`` -- for DisCell and every baseline alike; the
label-derived "top-4 MKI67 types" set is retired. Runs fitted before the code
carried the read are re-read here from their accepted checkpoint (``best.pt``)
and the new keys are written into the records that hold the old ones:

* ``runs/<R>/metrics.json["final"]``: the ``cycle_q90`` block of
  ``cell_cycle.trainer_cycle_q90`` and the flat keys ``cycle_r2_z_q90``,
  ``cycle_r2_w_q90``, ``cycle_linear_q90`` -- what ``Trainer.evaluate`` now
  writes, on its rows in its order. Only for a run whose ``final`` describes
  ``best.pt`` (trained after the R26 fix);
* ``runs/<R>/crossslide/<E>.json["held_out_section"]`` (``--eval-dataset E``):
  the same on the held-out section, as ``crossslide.apply_fit`` evaluates it;
* the ``DisCell/<R>`` column of every battery JSON of the dataset (and of E)
  that has one: ``cycle_q90`` exactly as ``baseline_battery.battery`` computes it;
* ``--baselines``: every tool column of the dataset's lineage battery JSONs,
  from the tool's latents on disk (CPU, no model).

Before a record is written its stored label-derived read is recomputed from
the same latents; a record that does not reproduce is not written (its
latents are not the ones that were scored). A record that already has
``cycle_q90`` is skipped unless ``--force``. Battery JSONs are rewritten
under the baselines queue's table lock (they share them), their markdown
re-rendered.

    python -m discell.experiments.cycle_reread --dataset D --runs R1 R2 ... \
        [--eval-dataset E]
    python -m discell.experiments.cycle_reread --dataset D --baselines \
        [--config-from S]
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import gc
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np

from discell import paths
from discell.model import metrics as M
from discell.model.cell_cycle import q90_keys, trainer_cycle_q90

log = logging.getLogger("discell.experiments.cycle_reread")

GS, GD = "gse315411_pdltma06_11_prime_solo", "gse315411_pdltma06_10_prime_dual"
OV, LU = "xenium_prime_ovarian_cancer_ffpe", "xenium_prime_human_lung_cancer_ffpe"
FF = "xenium_prime_human_ovary_ff"
REPO = Path(__file__).resolve().parents[2]
BRES = Path("/home/rmolen/github/DisCell-baselines/results")
#: battery column -> latents dir under BRES: stage (f) of
#: scripts/queue_2026-09-25_final_lineage.sh, and the lineage refits of
#: scripts/queue_2026-09-28_baselines_lineage.sh
LATENTS = {
    GS: {"resolVI": f"resolvi_lineage/{GS}", "SIMVI": f"simvi/{GS}",
         "MintFlow": f"mintflow/{GS}",
         "MintFlow (w=600, 5/50 epochs)": f"mintflow/{GS}_w600",
         "MintFlow (50 epochs, w=600)": f"mintflow/{GS}_full",
         "SIMVI (lineage)": f"simvi/{GS}_lineage",
         "MintFlow (lineage)": f"mintflow/{GS}_lineage"},
    GD: {"resolVI": f"resolvi_lineage/{GD}",
         "SIMVI (fit on this section)": f"simvi/{GD}_fit_on_target",
         "MintFlow (transfer, 5/50 epochs)": f"mintflow/{GD}_transfer",
         "MintFlow (transfer, 50 epochs)": f"mintflow/{GD}_transfer_full",
         "SIMVI (lineage, fit on this section)": f"simvi/{GD}_lineage",
         "MintFlow (lineage, transfer)": f"mintflow/{GD}_lineage"},
    OV: {"resolVI": f"resolvi_lineage/{OV}"},
    LU: {"resolVI": f"resolvi_lineage/{LU}"},
    FF: {"resolVI": f"resolvi_lineage/{FF}",
         "SIMVI (100k-cell window)": f"simvi/{FF}",
         "SIMVI (lineage, 100k-cell window)": f"simvi/{FF}_lineage"},
}
#: the lineage battery tables (baselines queue tag _lineage, and the
#: old-label columns it moves aside) and the run that scores their columns
BATTERY_TAGS = ("_lineage", "_lineage_oldlabels")
BATTERY_CONFIG_RUN = "finalL_s0"
TABLES_LOCK = REPO / "scripts/logs/baselines_lineage_2026-09-28/tables.lock"
#: |recomputed - stored| of the label-derived read: latents re-encoded on the
#: GPU (scatter order) vs read from disk
TOL_GPU, TOL_CPU = 2e-3, 1e-6


# -- records -------------------------------------------------------------------

@contextmanager
def tables_lock():
    TABLES_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLES_LOCK, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_json(path: Path, obj, indent: int = 2) -> None:
    """Atomic: a reader never sees half a record."""
    tmp = path.with_name(f".{path.name}.cycle_reread.tmp")
    tmp.write_text(json.dumps(obj, indent=indent, default=str))
    os.replace(tmp, path)


def provenance(diffs: dict) -> dict:
    return {"by": "discell/experiments/cycle_reread.py", "checkpoint": "best.pt",
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "stored_label_set_abs_diff": diffs}


def check(stored: dict, recomputed: dict, tol: float, what: str) -> dict:
    """|recomputed - stored| of the label-derived read's pooled R^2 per latent;
    raises when one exceeds *tol*."""
    diffs = {}
    for name, value in recomputed.items():
        want = ((stored or {}).get(name) or {}).get("r2_pooled")
        if want is None or not np.isfinite(want):
            continue
        diffs[name] = abs(value["r2_pooled"] - want)
        if not diffs[name] <= tol:
            raise ValueError(f"{what}: stored label-derived cycle {name} "
                             f"{want:.5f} not reproduced ({value['r2_pooled']:.5f})")
    if not diffs:
        raise ValueError(f"{what}: no stored label-derived read to check against")
    return diffs


def _scores(data) -> np.ndarray:
    return np.stack([data.cycle["s_score"], data.cycle["g2m_score"]], axis=1)


# -- one trainer's rows, as Trainer.evaluate orders them -------------------------------

def evaluate_order(trainer) -> tuple[np.ndarray, np.ndarray]:
    """(rows, train mask) of ``Trainer.evaluate``: training-tile seeds, then
    validation-tile seeds, batch by batch."""
    seeds = lambda batches: [b["nodes"][:b["n_seeds"]] for b in batches]
    train = np.concatenate(seeds(trainer.train_batches))
    rows = np.concatenate([train] + seeds(trainer.val_batches))
    return rows, np.arange(len(rows)) < len(train)


def trainer_block(data, trainer, latents: dict, stored_cycle: dict,
                  seed: int, what: str) -> dict:
    """``Trainer.evaluate``'s top-decile block from node-ordered *latents*,
    after the stored label-derived block is reproduced from the same ones."""
    rows, train = evaluate_order(trainer)
    z, w = latents["mu_z"][rows], latents["mu_w"][rows]
    types = np.asarray(stored_cycle["types"])
    old = {name: M.cycle_r2(design, data.t[rows], _scores(data)[rows], types,
                            train, ~train, seed=seed)
           for name, design in (("z", z), ("w", w))}
    diffs = check(stored_cycle, old, TOL_GPU, what)
    block = trainer_cycle_q90(data, rows, z, w, train, seed=seed)
    block["reread"] = provenance(diffs)
    return block


def battery_block(data, z: np.ndarray, spatial, rows: np.ndarray | None,
                  stored: dict, seed: int, tol: float, what: str) -> dict:
    """``baseline_battery.battery``'s top-decile block for a column scored on
    *rows* (None = every cell, node order), after its stored label-derived
    block is reproduced."""
    from discell.experiments.baseline_battery import cycle_inputs, cycle_q90_column

    cycle = cycle_inputs(data)
    test = np.zeros(data.n_cells, dtype=bool)
    for tile in data.val_tiles:
        test[tile] = True
    rows = np.arange(data.n_cells) if rows is None else rows
    sub = {"types": cycle["types"], "scores": cycle["scores"][rows],
           "cells": cycle["cells"][rows], "x_pcs": cycle["x_pcs"][rows]}
    t, train, held = data.t[rows], ~test[rows], test[rows]
    types = np.asarray(stored["cycle"]["types"])
    old = {"z": M.cycle_r2(np.asarray(z, np.float64), t, sub["scores"], types,
                           train, held, seed=seed)}
    if spatial is not None and spatial.shape[1] > 0 and "spatial" in stored["cycle"]:
        old["spatial"] = M.cycle_r2(np.asarray(spatial, np.float64), t,
                                    sub["scores"], types, train, held, seed=seed)
    diffs = check(stored["cycle"], old, tol, what)
    block = cycle_q90_column(z, spatial, t, train, held, sub, seed=seed)
    block["reread"] = provenance(diffs)
    return block


def battery_files(dataset: str) -> list[tuple[Path, Path, str]]:
    from discell.experiments.baseline_battery import artefacts

    out = []
    for tag in BATTERY_TAGS:
        js, md = artefacts(dataset, tag)
        if js.exists():
            out.append((js, md, tag))
    return out


def write_battery(dataset: str, js: Path, md: Path, tag: str, column: str,
                  block: dict) -> bool:
    """``entries[column]["cycle_q90"] = block``, under the tables lock, the
    markdown re-rendered (the old-label table keeps its note)."""
    from discell.experiments.baseline_battery import markdown

    with tables_lock():
        entries = json.loads(js.read_text())
        if column not in entries:
            log.warning("%s: column %r left %s meanwhile -- not written",
                        dataset, column, js.name)
            return False
        entries[column]["cycle_q90"] = block
        write_json(js, entries)
        text = markdown(entries, dataset, tag)
        if tag.endswith("_oldlabels") and md.exists():
            old = md.read_text()
            note = old[old.find("\nThese columns"):] if "\nThese columns" in old else ""
            text += note
        md.write_text(text)
    log.info("%s %s: %r cycle_q90 written (z %.4f)", dataset, js.name, column,
             block["z"]["r2_pooled"])
    return True


# -- DisCell runs ---------------------------------------------------------------------

def _free(trainer) -> None:
    import torch

    trainer.train_batches, trainer.val_batches = [], []
    gc.collect()
    torch.cuda.empty_cache()


ASSEMBLY_FIELDS = ("variant", "embeddings", "tile_cells", "phi_pca", "v_pcs",
                   "val_fraction", "seed", "label_key")


def assembly_key(config) -> tuple:
    """What ``assemble`` depends on: runs that share it share their data."""
    get = config.get if isinstance(config, dict) else lambda k: getattr(config, k)
    return tuple(get(k) for k in ASSEMBLY_FIELDS)


def reread_run(dataset: str, run: str, eval_dataset: str | None, device: str,
               force: bool, cache: dict) -> dict:
    """Every record of one run; *cache* keeps one assembly per configuration
    (runs of one seed share their tiles) across calls."""
    import torch

    from discell.model.prepare import assemble
    from discell.model.train import Trainer
    from discell.model.validate import collect_latents, load_run

    run_dir = paths.dataset(dataset).root / "runs" / run
    recorded = json.loads((run_dir / "config.json").read_text())
    key = assembly_key(recorded)
    done = {}
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    todo_metrics = force or "cycle_q90" not in (metrics.get("final") or {})
    column = f"DisCell/{run}"
    todo_battery = [(js, md, tag) for js, md, tag in battery_files(dataset)
                    if column in (e := json.loads(js.read_text()))
                    and e[column].get("config", {}).get("trained_on") == dataset
                    and (force or "cycle_q90" not in e[column])]
    cross_path = (run_dir / "crossslide" / f"{eval_dataset}.json"
                  if eval_dataset else None)
    todo_cross = (cross_path is not None and cross_path.exists()
                  and (force or "cycle_q90" not in json.loads(
                      cross_path.read_text())["held_out_section"]))
    todo_eval_battery = ([(js, md, tag) for js, md, tag in battery_files(eval_dataset)
                          if column in (e := json.loads(js.read_text()))
                          and e[column].get("config", {}).get("trained_on") == dataset
                          and (force or "cycle_q90" not in e[column])]
                         if eval_dataset else [])
    if not (todo_metrics or todo_battery or todo_cross or todo_eval_battery):
        log.info("%s/%s: every record has cycle_q90 -- skipped", dataset, run)
        return {"run": run, "skipped": True}

    if (dataset, key) not in cache:
        cache.clear()                               # one assembly resident
        gc.collect()
        cache[(dataset, key)] = assemble(
            dataset, **{k: recorded[k] for k in ASSEMBLY_FIELDS})
    data = cache[(dataset, key)]
    config, data, trainer, _, _ = load_run(dataset, run, device, data=data)
    if assembly_key(config) != key:
        raise ValueError(f"{dataset}/{run}: best.pt's configuration assembles "
                         f"other data than config.json ({assembly_key(config)} "
                         f"vs {key})")
    latents = collect_latents(trainer, data)

    if todo_metrics:
        if metrics.get("final_epoch") != metrics["best"]["epoch"]:
            log.warning("%s/%s: final is not best.pt (pre-R26) -- metrics.json "
                        "not written", dataset, run)
        else:
            block = trainer_block(data, trainer, latents,
                                  metrics["final"]["cycle"], config.seed,
                                  f"{dataset}/{run} metrics.json")
            fresh = json.loads(metrics_path.read_text())
            fresh["final"]["cycle_q90"] = block
            fresh["final"].update(q90_keys(block))
            write_json(metrics_path, fresh)
            done["metrics"] = q90_keys(block)
            log.info("%s/%s metrics.json: %s", dataset, run, done["metrics"])
    for js, md, tag in todo_battery:
        stored = json.loads(js.read_text())[column]
        block = battery_block(data, latents["mu_z"], latents["mu_w"], None,
                              stored, config.seed, TOL_GPU,
                              f"{dataset} {js.name} {column}")
        if write_battery(dataset, js, md, tag, column, block):
            done[f"battery{tag}"] = q90_keys(block | {"w": block.get("spatial")})

    if todo_cross or todo_eval_battery:
        model = trainer.model
        _free(trainer)
        del trainer
        data_b = assemble(eval_dataset, config.variant, config.embeddings,
                          tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                          v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                          seed=config.seed, label_key=config.label_key)
        if [str(n) for n in data_b.type_names] != [str(n) for n in data.type_names]:
            raise ValueError(f"type vocabularies differ: {dataset} vs {eval_dataset}")
        trainer_b = Trainer(config, data_b)
        trainer_b.model = model.to(trainer_b.device).eval()
        latents_b = collect_latents(trainer_b, data_b)
        if todo_cross:
            record = json.loads(cross_path.read_text())
            block = trainer_block(data_b, trainer_b, latents_b,
                                  record["held_out_section"]["cycle"], config.seed,
                                  f"{dataset}/{run} crossslide {eval_dataset}")
            fresh = json.loads(cross_path.read_text())
            fresh["held_out_section"]["cycle_q90"] = block
            fresh["held_out_section"].update(q90_keys(block))
            write_json(cross_path, fresh)
            done["crossslide"] = q90_keys(block)
            log.info("%s/%s crossslide %s: %s", dataset, run, eval_dataset,
                     done["crossslide"])
        for js, md, tag in todo_eval_battery:
            stored = json.loads(js.read_text())[column]
            block = battery_block(data_b, latents_b["mu_z"], latents_b["mu_w"],
                                  None, stored, config.seed, TOL_GPU,
                                  f"{eval_dataset} {js.name} {column}")
            if write_battery(eval_dataset, js, md, tag, column, block):
                done[f"eval_battery{tag}"] = q90_keys(
                    block | {"w": block.get("spatial")})
        _free(trainer_b)
        del trainer_b, data_b
    else:
        _free(trainer)
        del trainer
    torch.cuda.empty_cache()
    return {"run": run, **done}


# -- baseline columns -------------------------------------------------------------------

def reread_baselines(dataset: str, config_from: str | None, force: bool) -> dict:
    """Every tool column of the dataset's lineage battery JSONs (CPU)."""
    from discell.experiments.baseline_battery import load_latents, run_config, shared_side

    source = config_from or dataset
    todo = []
    for js, md, tag in battery_files(dataset):
        for column, entry in json.loads(js.read_text()).items():
            if column.startswith("DisCell/") or column.startswith("smoke"):
                continue
            if not force and "cycle_q90" in entry:
                continue
            todo.append((js, md, tag, column, entry))
    if not todo:
        log.info("%s: every baseline column has cycle_q90 -- nothing to do", dataset)
        return {}
    config = run_config(source, BATTERY_CONFIG_RUN)
    data, _, _, _ = shared_side(dataset, config)
    out = {}
    for js, md, tag, column, entry in todo:
        rel = LATENTS.get(dataset, {}).get(column)
        path = BRES / rel / "latents.h5ad" if rel else None
        if path is None or not path.exists():
            log.warning("%s %s: no latents known for %r (%s) -- not re-read",
                        dataset, js.name, column, path)
            out[f"{tag}|{column}"] = "no latents"
            continue
        tool = column.split("/")[0].split(" ")[0].lower()
        rows, z, spatial, _ = load_latents(path, tool)
        if len(rows) != entry["n_cells"]:
            log.warning("%s %s %r: %d latent rows vs %d scored -- not re-read",
                        dataset, js.name, column, len(rows), entry["n_cells"])
            out[f"{tag}|{column}"] = "row count differs"
            continue
        try:
            block = battery_block(data, z, spatial, rows, entry, config["seed"],
                                  TOL_CPU, f"{dataset} {js.name} {column}")
        except ValueError as exc:
            log.warning("%s -- not written", exc)
            out[f"{tag}|{column}"] = f"not reproduced: {exc}"
            continue
        if write_battery(dataset, js, md, tag, column, block):
            out[f"{tag}|{column}"] = q90_keys(block | {"w": block.get("spatial")})
    return out


# -- CLI --------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--eval-dataset", default=None,
                        help="the held-out section: its crossslide record and "
                             "battery DisCell columns")
    parser.add_argument("--baselines", action="store_true",
                        help="re-read every tool column of the lineage batteries")
    parser.add_argument("--config-from", default=None,
                        help="with --baselines on a held-out section: the "
                             "dataset whose finalL_s0 configuration scored it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    summary, failed = {}, []
    if args.baselines:
        summary["baselines"] = reread_baselines(args.dataset, args.config_from,
                                                args.force)
    cache: dict = {}
    for run in args.runs:
        try:
            summary[run] = reread_run(args.dataset, run, args.eval_dataset,
                                      args.device, args.force, cache)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            log.error("%s/%s: %r", args.dataset, run, exc)
            failed.append(run)
            summary[run] = {"error": repr(exc)}
    print(json.dumps(summary, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
