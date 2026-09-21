#!/usr/bin/env python3
"""Post-hoc cycle read under the old and the depth-neutral target (todo 3.6).

Reloads a finished fit the way :mod:`discell.model.degeneracy` does, collects
``mu_z`` / ``mu_w`` over the same train/val tiles ``Trainer.evaluate`` uses,
and runs :func:`discell.model.metrics.cycle_r2` twice: against the raw Scanpy
S/G2M scores and against their depth-conditional normal scores. Alongside each
read: the within-type permuted floor (inside ``cycle_r2``), the l-baseline
(log total counts alone), the 50-PC linear expression reference, the split-half
reliability of the target, and the within-type Spearman of each score against
log depth on every type with at least ``MIN_TYPE_CELLS`` cells.

Usage::

    python -m discell.model.cycle_target --dataset <id> --run <run> --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

import numpy as np

from discell import paths
from discell.model import metrics as M
from discell.model.cell_cycle import depth_neutral, score_cell_cycle
from discell.model.degeneracy import load_trainer

log = logging.getLogger("discell.model.cycle_target")

#: types this small say nothing about a depth tilt (pre-registration, todo 3.6)
MIN_TYPE_CELLS = 2000


def depth_tilt(scores: dict[str, np.ndarray], t: np.ndarray,
               totals: np.ndarray, type_names: Sequence[str]) -> list[dict]:
    """Within-type Spearman of every score against log total counts."""
    from scipy.stats import spearmanr

    log_depth = np.log(totals.clip(min=1.0))
    rows = []
    for g in np.unique(t):
        members = np.flatnonzero(t == g)
        if len(members) < MIN_TYPE_CELLS:
            continue
        row = {"type": str(type_names[g]), "n": int(len(members))}
        for key, v in scores.items():
            row[key] = float(spearmanr(v[members], log_depth[members]).correlation)
        rows.append(row)
    return rows


def cycle_reads(trainer, scores: np.ndarray, cycling: np.ndarray,
                sweep: dict) -> dict:
    """The four ``cycle_r2`` reads of ``Trainer.evaluate`` for one target."""
    rows_all, t_all = sweep["rows_all"], sweep["t_all"]
    kwargs = dict(t=t_all, types=cycling, train=sweep["train_mask"],
                  test=~sweep["train_mask"], seed=trainer.config.seed)
    log_depth = np.log(trainer.data.totals[rows_all].clip(min=1.0))
    designs = {"z": sweep["z_all"], "w": sweep["w_all"],
               "linear_ref": trainer.data.cycle["x_pcs"][rows_all],
               "lbaseline": log_depth[:, None].astype(np.float64)}
    return {name: M.cycle_r2(design, scores=scores, **kwargs)
            for name, design in designs.items()}


def collect(trainer) -> dict:
    """``mu_z``/``mu_w`` over the train and val tiles, with the split mask."""
    train = trainer._sweep(trainer.train_batches)
    val = trainer._sweep(trainer.val_batches)
    rows_all = np.concatenate([train["nodes"], val["nodes"]])
    train_mask = np.zeros(len(rows_all), dtype=bool)
    train_mask[:len(train["nodes"])] = True
    return {"rows_all": rows_all, "t_all": trainer.data.t[rows_all],
            "z_all": np.vstack([train["mu_z"], val["mu_z"]]),
            "w_all": np.vstack([train["mu_w"], val["mu_w"]]),
            "train_mask": train_mask}


def compare(trainer, seed: int = 0) -> dict:
    """The old-vs-new table for one fit."""
    data = trainer.data
    type_names = [str(n) for n in data.type_names]
    fresh = score_cell_cycle(data.x, data.gene_names, t=data.t,
                             depth_neutral_target=True, seed=seed)
    if fresh is None:
        raise SystemExit("panel carries too few cycle markers")

    old = np.stack([fresh["s_score_raw"], fresh["g2m_score_raw"]], axis=1)
    new = np.stack([fresh["s_score"], fresh["g2m_score"]], axis=1)
    # attribution control: the same within-type normal-score transform and the
    # same random tie-breaking, but ONE depth stratum -- so it keeps whatever
    # depth the raw score carried. What separates it from "new" is the depth
    # conditioning alone; what separates it from "old" is the transform alone.
    control = np.stack([depth_neutral(fresh[key], data.totals, data.t,
                                      seed=seed, cells_per_bin=len(data.t))
                        for key in ("s_score_raw", "g2m_score_raw")], axis=1)
    cycling = np.array([g for g in data.cycle["cycling_types"]
                        if "nassigned" not in type_names[g]][:4])
    sweep = collect(trainer)
    reads = {name: cycle_reads(trainer, target[sweep["rows_all"]], cycling, sweep)
             for name, target in (("old", old), ("new", new),
                                  ("rank_control", control))}
    tilt = depth_tilt({"s_old": fresh["s_score_raw"], "s_new": fresh["s_score"],
                       "s_rank_control": control[:, 0],
                       "g2m_old": fresh["g2m_score_raw"],
                       "g2m_new": fresh["g2m_score"],
                       "g2m_rank_control": control[:, 1]},
                      data.t, data.totals, type_names)
    return {"reads": reads, "depth_tilt": tilt,
            "reliability": {"old": fresh["reliability_raw"],
                            "new": fresh["reliability"]},
            "cycling_types": [type_names[g] for g in cycling],
            "n_cells": int(len(sweep["rows_all"]))}


def _table(entry: dict) -> str:
    lines = ["read            old      new   rank_control",
             "-------------------------------------------"]
    for latent in ("z", "w", "linear_ref", "lbaseline"):
        o, n, c = (entry["reads"][k][latent]
                   for k in ("old", "new", "rank_control"))
        lines.append("%-12s %7.4f  %7.4f  %7.4f"
                     % (latent, o["r2_pooled"], n["r2_pooled"], c["r2_pooled"]))
        lines.append("%-12s %7.4f  %7.4f  %7.4f"
                     % ("  permuted", o["r2_permuted"], n["r2_permuted"],
                        c["r2_permuted"]))
    r = entry["reliability"]
    lines.append("reliability  S %.3f -> %.3f, G2M %.3f -> %.3f"
                 % (r["old"]["s"], r["new"]["s"],
                    r["old"]["g2m"], r["new"]["g2m"]))
    worst_old = max(abs(row["g2m_old"]) for row in entry["depth_tilt"])
    worst_new = max(abs(row["g2m_new"]) for row in entry["depth_tilt"])
    lines.append("|Spearman(G2M, log depth)| max over types: %.3f -> %.3f"
                 % (worst_old, worst_new))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    trainer, _, epoch = load_trainer(args.dataset, args.run, args.device)
    entry = compare(trainer, seed=args.seed)
    entry["run"], entry["epoch"] = args.run, epoch
    out_path = (paths.dataset(args.dataset).root / "experiments"
                / f"cycle_target_{args.run}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entry, indent=2))
    log.info("%s (epoch %d), cycling types %s\n%s", args.run, epoch,
             entry["cycling_types"], _table(entry))
    log.info("written %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
