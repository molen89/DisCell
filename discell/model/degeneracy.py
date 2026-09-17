#!/usr/bin/env python3
"""Post-hoc degeneracy diagnostics (spec 7.10) for a finished run.

Two reads of "is z just t?": ``I(mu_z; t)/H(t)`` from a held-out logistic
probe with the within-type share of z's variance, and the held-out
reconstruction lost when every cell's z is replaced by its type's mean z
(w, the leak mixture and kappa unchanged). Fits from 2026-09-17 on record
both in ``metrics.json`` through ``Trainer.evaluate``; this module runs that
same evaluation from ``best.pt`` for earlier runs and writes
``runs/<run>/degeneracy.json``.

Usage::

    python -m discell.model.degeneracy --dataset <id> --run <run> [--device cpu]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from typing import Sequence

from discell import paths
from discell.model.prepare import assemble
from discell.model.train import TrainConfig, Trainer

log = logging.getLogger("discell.model.degeneracy")


def load_trainer(dataset: str, run: str, device: str):
    """``validate.load_run`` with *device* honoured for the resident tiles too.

    ``load_run`` builds the Trainer from the checkpoint's own ``device``
    ("cuda" for every pinned run), which would park the whole slide on a GPU
    even when the caller asked for the CPU.
    """
    import torch

    run_dir = paths.dataset(dataset).root / "runs" / run
    payload = torch.load(run_dir / "best.pt", map_location="cpu",
                         weights_only=False)
    payload["config"].setdefault("gat_sources", "type_z")   # pre-field era
    payload["config"].setdefault("subtract_leak", False)
    payload["config"].setdefault("gat_sink", False)
    config = dataclasses.replace(TrainConfig(**payload["config"]), device=device)
    data = assemble(dataset, config.variant, config.embeddings,
                    tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                    v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                    seed=config.seed, label_key=config.label_key)
    trainer = Trainer(config, data)
    trainer.model.load_state_dict(payload["model"])
    trainer.model.eval()
    return trainer, run_dir, int(payload["epoch"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    trainer, run_dir, epoch = load_trainer(args.dataset, args.run, args.device)
    report = trainer.evaluate()
    out = {"run": args.run, "epoch": epoch, "recon_val": report["recon_val"],
           "nmi": report["nmi"], "degeneracy": report["degeneracy"],
           "recon_gap": report["recon_gap"]}
    (run_dir / "degeneracy.json").write_text(json.dumps(out, indent=2))
    d, g = out["degeneracy"], out["recon_gap"]
    log.info("%s (epoch %d): I/H %.3f  within-var %.3f  recon %.4f  "
             "type-mean z %.4f  gap %.4f  type profile %.4f",
             args.run, epoch, d["mi_ratio"], d["within_var_fraction"],
             g["recon"], g["recon_typemean_z"], g["gap"], g["recon_type_profile"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
