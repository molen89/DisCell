#!/usr/bin/env python3
"""The spec 4.6 operating point, plus the two architect cross-checks.

Three questions, answered with short real-slide fits (stability-length, not
convergence-length -- the outputs are curves and deltas, never biology):

1. **Where is alpha_a's operating point?** Sweep it, plot held-out probe dCE
   and z-type NMI against it; the spec wants dCE near its noise floor while NMI
   is still above its own floor. The alpha_a = 0 run is the uncontrolled
   baseline that scales the escalation rule (escalate to the adversary only if
   dCE stays above ~20% of it).
2. **Does a nonlinear probe agree with the ridge?** The ridge can under-report
   nonlinear leakage. One small-MLP probe per run; if it tracks the ridge, the
   escalation decision may rest on the ridge alone.
3. **What does Phi buy?** Delta held-out reconstruction with Phi present vs
   zeroed -- composition of `c` cannot answer this, type-AUC could not either
   (the mask experiment measured type information only). This ablation also
   settles the deferred PCA decision.

Usage::

    python -m discell.model.calibrate --dataset <id> --embeddings egomask_ego_v1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from typing import Sequence

import numpy as np

from discell import paths
from discell.model.prepare import ModelData, assemble
from discell.model.train import TrainConfig, Trainer

log = logging.getLogger("discell.model.calibrate")

DEFAULT_ALPHA_GRID = (0.0, 0.01, 0.02, 0.05, 0.1)


def mlp_probe_delta_ce(z: np.ndarray, t: np.ndarray, v: np.ndarray,
                       vbar_t: np.ndarray, train: np.ndarray,
                       test: np.ndarray, seed: int = 0) -> float:
    """The ridge probe's nonlinear cross-check: a small MLP, same dCE."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(train)
    rows = rows if len(rows) <= 30_000 else np.sort(rng.choice(rows, 30_000, False))
    test_rows = np.flatnonzero(test)

    onehot = np.eye(int(t.max()) + 1, dtype=np.float32)[t]
    design = np.hstack([z, onehot]).astype(np.float32)
    net = nn.Sequential(nn.Linear(design.shape[1], 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, v.shape[1]))
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
    d_train = torch.tensor(design[rows])
    v_train = torch.tensor(v[rows], dtype=torch.float32)
    for _ in range(300):
        pick = torch.randint(0, len(rows), (2048,))
        loss = ((net(d_train[pick]) - v_train[pick]) ** 2).mean()
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    with torch.no_grad():
        predicted = net(torch.tensor(design[test_rows])).numpy()
    baseline = float(((v[test_rows] - vbar_t[t[test_rows]]) ** 2).mean())
    return baseline - float(((v[test_rows] - predicted) ** 2).mean())


def _short_fit(data: ModelData, base: TrainConfig, **overrides) -> dict:
    """One stability-length fit; returns final metrics plus collected arrays."""
    config = dataclasses.replace(base, **overrides)
    trainer = Trainer(config, data)
    trainer.fit()
    report = trainer.evaluate()
    collected = report.pop("collected")
    rows = collected["rows"]
    val_rows = np.concatenate([b["nodes"][:b["n_seeds"]]
                               for b in trainer.val_batches])
    in_val = np.isin(rows, val_rows)
    report["mlp_delta_ce"] = mlp_probe_delta_ce(
        collected["z"], data.t[rows], data.v_block[rows], data.vbar_t,
        ~in_val, in_val, seed=config.seed)
    return report


def run(args: argparse.Namespace) -> int:
    data = assemble(args.dataset, args.variant, args.embeddings,
                    tile_cells=args.tile_cells, seed=args.seed)
    base = TrainConfig(
        dataset=args.dataset, variant=args.variant, embeddings=args.embeddings,
        kappa=args.kappa, epochs=args.epochs, eval_every=args.eval_every,
        figures_every=args.epochs * 2 // args.eval_every * args.eval_every
        or args.eval_every,                     # effectively off
        patience=10 * args.epochs, device=args.device, seed=args.seed,
        tile_cells=args.tile_cells,
    )

    results: dict = {"alpha_a": {}, "phi_ablation": {}}
    started = time.time()

    for alpha in args.alpha_grid:
        report = _short_fit(data, base, alpha_a=float(alpha),
                            run_name=f"cal_alpha_a_{alpha:g}")
        results["alpha_a"][f"{alpha:g}"] = {
            "delta_ce": report["probe"]["delta_ce"],
            "noise_floor": report["probe"]["noise_floor"],
            "mlp_delta_ce": report["mlp_delta_ce"],
            "nmi": report["nmi"],
            "recon_val": report["recon_val"],
            "mirror_r2": report["mirror"]["r2"],
        }
        log.info("alpha_a=%g  dCE %.4f (mlp %.4f, floor %.4f)  nmi %.3f",
                 alpha, report["probe"]["delta_ce"], report["mlp_delta_ce"],
                 report["probe"]["noise_floor"], report["nmi"])

    # -- Phi ablation: same config, image zeroed --------------------------
    chosen = args.alpha_grid[len(args.alpha_grid) // 2]
    with_phi = results["alpha_a"].get(f"{chosen:g}") or _short_fit(
        data, base, alpha_a=float(chosen), run_name=f"cal_alpha_a_{chosen:g}")
    ablated_data = dataclasses.replace(data, phi=np.zeros_like(data.phi))
    ablated = _short_fit(ablated_data, base, alpha_a=float(chosen),
                         run_name="cal_no_phi")
    results["phi_ablation"] = {
        "alpha_a": float(chosen),
        "recon_with_phi": with_phi["recon_val"],
        "recon_without_phi": ablated["recon_val"],
        "delta_recon": with_phi["recon_val"] - ablated["recon_val"],
        "nmi_without_phi": ablated["nmi"],
        "note": "delta > 0 means Phi improves held-out reconstruction through "
                "m_psi/c -- the test the mask experiment could not run",
    }

    results["minutes"] = (time.time() - started) / 60
    out = paths.dataset(args.dataset).root / "experiments" / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"results": results,
         "args": {k: str(v) for k, v in vars(args).items()}}, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--embeddings", default="egomask_ego_v1")
    parser.add_argument("--alpha-grid", type=float, nargs="*",
                        default=list(DEFAULT_ALPHA_GRID))
    parser.add_argument("--kappa", type=float, default=0.1,
                        help="mid-grid: calibration should not sit on either "
                             "end of the sweep")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=40,
                        help="one evaluation at the end is what calibration needs")
    parser.add_argument("--tile-cells", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
