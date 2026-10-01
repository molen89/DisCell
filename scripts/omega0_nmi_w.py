#!/usr/bin/env python3
"""NMI of mu_w with the lineage type, read exactly as the battery reads z's.

For the omega = 0 ablation (devlog "omega = 0 ablation (motivation,
2026-09-30)"): no existing read gives the k-means NMI of mu_w with type on
the battery's cells (context_grade reads finalL only). This loads a run's
accepted checkpoint (best.pt), sweeps the training then the held-out seeds
as ``Trainer.evaluate`` does, and applies the battery's own NMI
(``metrics.z_type_nmi`` on ``eval_mask.nmi_inputs``: target cells, labels
compacted, k = number of target types, the fit's seed) to mu_z and to mu_w.
The mu_z value must reproduce degeneracy.json's ``nmi_targets``; it is
written beside the mu_w value as the check.

Run at OMP_NUM_THREADS=8 (k-means; issue T-omp).

    python scripts/omega0_nmi_w.py --dataset D --run R --out path.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from discell.model import eval_mask as EM
from discell.model import metrics as M
from discell.model.degeneracy import load_trainer

TOL = 1e-6


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    trainer, run_dir, epoch = load_trainer(args.dataset, args.run, args.device)
    train = trainer._sweep(trainer.train_batches)
    val = trainer._sweep(trainer.val_batches)
    rows = np.concatenate([train["nodes"], val["nodes"]])
    t_all = trainer.data.t[rows]
    names = trainer.data.type_names
    seed = trainer.config.seed
    out = {"dataset": args.dataset, "run": args.run, "epoch": epoch,
           "checkpoint": "best.pt", "seed": seed,
           "omega": trainer.config.omega,
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "eval_mask": EM.record(names, t_all)}
    for key, latent in (("z", "mu_z"), ("w", "mu_w")):
        x = np.vstack([train[latent], val[latent]])
        out[f"nmi_{key}_targets"] = M.z_type_nmi(*EM.nmi_inputs(x, t_all, names),
                                                 seed=seed)
    deg = run_dir / "degeneracy.json"
    if deg.exists():
        stored = json.loads(deg.read_text())
        diff = out["nmi_z_targets"] - stored["nmi_targets"]
        out["check"] = {"degeneracy_nmi_targets": stored["nmi_targets"],
                        "diff": diff, "ok": bool(abs(diff) <= TOL)}
    else:
        out["check"] = None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
