#!/usr/bin/env python3
"""Apply a finished fit to another dataset -- the held-out-section protocol.

    python -m discell.model.crossslide --dataset A --run R --eval-dataset B
                                       [--eval-variant V]

Loads R's best weights, assembles B under R's label key / variant /
embeddings name / tile size / seed, refuses unless B's sorted type vocabulary
equals A's (``t`` is one-hot over that order; every per-type table is index
aligned -- a mismatch is an error, never a remap), then reports the same reads
``metrics.json`` carries -- computed by ``Trainer.evaluate`` on B's tiles,
exactly as during training -- plus held-out reconstruction over **every**
tile of B, since the whole section is held out and B's 15 % split only
serves the probe's block-CV. R's own numbers sit beside them for the
same-section comparison. Writes ``runs/R/crossslide/<B>[.<V>].json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Sequence

import numpy as np

from discell import paths
from discell.model import metrics as M
from discell.model.prepare import assemble
from discell.model.train import Trainer
from discell.model.validate import load_run

log = logging.getLogger("discell.crossslide")


def apply_fit(config, model, data) -> dict:
    """``Trainer.evaluate`` reads of a trained *model* on *data*'s tiles, plus
    held-out reconstruction over every tile (the section is held out whole)."""
    trainer = Trainer(config, data)
    trainer.model = model.to(trainer.device).eval()
    reads = trainer.evaluate()
    reads.pop("collected")
    every = trainer.train_batches + trainer.val_batches
    swept = trainer._sweep(every, want_log_p=True)
    x_all = np.vstack([data.x[b["nodes"][:b["n_seeds"]]].toarray() for b in every])
    reads["recon_all_tiles"] = M.held_out_reconstruction(x_all, swept["log_p"])
    reads["n_cells"] = int(data.n_cells)
    reads["n_tiles"] = len(every)
    return reads


def evaluate_on(dataset: str, run: str, eval_dataset: str,
                eval_variant: str | None = None, device: str = "cuda") -> dict:
    config, data_a, trainer_a, run_dir, _ = load_run(dataset, run, device=device)
    variant = eval_variant or config.variant
    data_b = assemble(eval_dataset, variant, config.embeddings,
                      tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                      v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                      seed=config.seed, label_key=config.label_key)
    names_a = [str(n) for n in data_a.type_names]
    names_b = [str(n) for n in data_b.type_names]
    if names_a != names_b:
        only_a, only_b = sorted(set(names_a) - set(names_b)), sorted(set(names_b) - set(names_a))
        raise ValueError(f"type vocabularies differ: only in {dataset}: {only_a}; "
                         f"only in {eval_dataset}: {only_b}")
    if data_b.x.shape[1] != data_a.x.shape[1]:
        raise ValueError(f"gene panels differ: {data_a.x.shape[1]} vs {data_b.x.shape[1]}")

    reads = apply_fit(config, trainer_a.model, data_b)

    own = json.loads((run_dir / "metrics.json").read_text())
    out = {"run": run, "trained_on": dataset, "evaluated_on": eval_dataset,
           "variant": variant, "held_out_section": reads,
           "same_section": {"best": own["best"], "final": own["final"]}}
    tag = eval_dataset if variant == config.variant else f"{eval_dataset}.{variant}"
    target = run_dir / "crossslide" / f"{tag}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, indent=2, default=str))
    log.info("%s on %s: recon all tiles %.4f (own val %.4f), NMI %.3f (own %.3f), "
             "probe dCE %+.3f, mirror %.3f -> %s",
             run, eval_dataset, reads["recon_all_tiles"], own["best"]["recon_val"],
             reads["nmi"], own["final"]["nmi"], reads["probe"]["delta_ce"],
             reads["mirror"]["r2"] if isinstance(reads["mirror"], dict) else reads["mirror"],
             target)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="dataset the run was trained on")
    parser.add_argument("--run", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--eval-variant", default=None,
                        help="bundle variant on the evaluation dataset (default: the run's)")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    evaluate_on(args.dataset, args.run, args.eval_dataset, args.eval_variant, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
