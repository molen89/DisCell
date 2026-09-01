#!/usr/bin/env python3
"""The kappa sweep -- the deliverable -- and its cross-run report.

Fits one model per (kappa, seed) and then reads every run back into one
comparison. Per the spec (7.7) and the architect review: **at least three seeds
per kappa**, because B is seed-bistable under the confound; the reportable
object is the effect envelope over kappa x seed jointly. A gene programme
stable across kappa but flapping across seeds is not a finding.

The report also **stratifies held-out reconstruction by the edges-lost QC
column**: partial isolation (degree 1-2 after pruning) still receives full
kappa against a thin rho_bar, and any effect concentrated in those cells is
suspect.

Usage::

    python -m discell.model.sweep --dataset <id> --embeddings egomask_ego_v1 \\
        --alpha-a 0.02
    python -m discell.model.sweep --dataset <id> --report-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

import numpy as np

from discell import paths
from discell.model.prepare import assemble
from discell.model.train import TrainConfig, Trainer

log = logging.getLogger("discell.model.sweep")

DEFAULT_KAPPAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
DEFAULT_SEEDS = (0, 1, 2)


def matched_correlation(b_a: np.ndarray, b_b: np.ndarray) -> float:
    """Mean |correlation| of B columns after optimal matching.

    B is identified only up to an invertible mix of ``w``, so columns are
    matched by the assignment maximising |corr| over the gene axis before
    averaging -- the number reads "how much of run A's programme space does
    run B reproduce".
    """
    from scipy.optimize import linear_sum_assignment

    a = (b_a - b_a.mean(0)) / (b_a.std(0) + 1e-12)
    b = (b_b - b_b.mean(0)) / (b_b.std(0) + 1e-12)
    corr = np.abs(a.T @ b / len(a))
    rows, cols = linear_sum_assignment(-corr)
    return float(corr[rows, cols].mean())


def run_name(kappa: float, seed: int) -> str:
    return f"sweep_k{kappa:g}_s{seed}"


def fit_grid(args: argparse.Namespace) -> None:
    data = assemble(args.dataset, args.variant, args.embeddings,
                    tile_cells=args.tile_cells, seed=0)
    for seed in args.seeds:
        for kappa in args.kappas:
            name = run_name(kappa, seed)
            run_dir = paths.dataset(args.dataset).root / "runs" / name
            if (run_dir / "metrics.json").exists() and not args.force:
                log.info("skip %s (metrics.json exists)", name)
                continue
            config = TrainConfig(
                dataset=args.dataset, variant=args.variant,
                embeddings=args.embeddings, run_name=name,
                kappa=float(kappa), seed=int(seed),
                alpha_z=args.alpha_z, alpha_w=args.alpha_w,
                alpha_a=args.alpha_a, omega=args.omega,
                invariance=args.invariance, adv_steps=args.adv_steps,
                adv_lr=args.adv_lr,
                epochs=args.epochs, tile_cells=args.tile_cells,
                device=args.device,
            )
            # one split for every run: the data seed is fixed, only the model
            # seed varies (prepare.assemble draws the split from its own stream)
            Trainer(config, data).fit()


def report(args: argparse.Namespace) -> dict:
    """Read every finished run back into the cross-(kappa x seed) comparison."""
    import torch

    from discell.model.networks import DisCell

    ds = paths.dataset(args.dataset)
    data = assemble(args.dataset, args.variant, args.embeddings,
                    tile_cells=args.tile_cells, seed=0)

    rows: list[dict] = []
    loaded: dict[tuple, dict] = {}
    for seed in args.seeds:
        for kappa in args.kappas:
            run_dir = ds.root / "runs" / run_name(kappa, seed)
            if not (run_dir / "best.pt").exists():
                log.warning("missing %s -- skipped", run_dir.name)
                continue
            payload = torch.load(run_dir / "best.pt", map_location="cpu",
                                 weights_only=False)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            b_matrix = payload["model"]["B.weight"].numpy()      # (G, d_w)
            loaded[(kappa, seed)] = {"B": b_matrix, "payload": payload}
            rows.append({"kappa": kappa, "seed": seed,
                         "recon": metrics["best"]["recon_val"],
                         "nmi": metrics["best"]["nmi"],
                         "epoch": metrics["best"]["epoch"]})

    # -- B stability: within kappa across seeds, and along kappa ----------
    stability = {"across_seeds": {}, "along_kappa": {}}
    for kappa in args.kappas:
        seeds_here = [s for s in args.seeds if (kappa, s) in loaded]
        pairs = [matched_correlation(loaded[(kappa, a)]["B"],
                                     loaded[(kappa, b)]["B"])
                 for i, a in enumerate(seeds_here) for b in seeds_here[i + 1:]]
        if pairs:
            stability["across_seeds"][f"{kappa:g}"] = {
                "mean": float(np.mean(pairs)), "min": float(np.min(pairs))}
    anchor = next(((k, s) for k in args.kappas for s in args.seeds
                   if (k, s) in loaded), None)
    if anchor:
        for (kappa, seed), entry in loaded.items():
            stability["along_kappa"][f"k{kappa:g}_s{seed}"] = matched_correlation(
                loaded[anchor]["B"], entry["B"])

    # -- per-cell effects: |w| per type, and recon stratified by QC --------
    per_type_w: dict = {}
    strata: dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lost = data.graph.pruned_per_cell
    for (kappa, seed), entry in loaded.items():
        state = entry["payload"]
        model = DisCell(n_genes=data.x.shape[1], n_types=len(data.p_t),
                        phi_dim=data.phi.shape[1],
                        median_counts=data.median_counts,
                        d_z=state["config"]["d_z"], d_w=state["config"]["d_w"],
                        hidden=state["config"]["hidden"],
                        gat_dim=state["config"]["gat_dim"],
                        heads=state["config"]["heads"]).to(device)
        model.load_state_dict(state["model"])
        trainer = Trainer(TrainConfig(**state["config"]), data)
        trainer.model = model.eval()
        sweep_out = trainer._sweep(trainer.val_batches, want_log_p=True)
        val_rows = sweep_out["nodes"]
        x_val = np.vstack([data.x[b["nodes"][:b["n_seeds"]]].toarray()
                           for b in trainer.val_batches])

        key = f"k{kappa:g}_s{seed}"
        w_norm = np.linalg.norm(sweep_out["mu_w"], axis=1)
        per_type_w[key] = {
            str(data.type_names[g]): float(w_norm[data.t[val_rows] == g].mean())
            for g in range(len(data.p_t)) if (data.t[val_rows] == g).any()}
        per_cell = (x_val * sweep_out["log_p"]).sum(1) / x_val.sum(1).clip(min=1)
        strata[key] = {
            "recon_lost0": float(per_cell[lost[val_rows] == 0].mean()),
            "recon_lost1plus": float(per_cell[lost[val_rows] >= 1].mean())
            if (lost[val_rows] >= 1).any() else None,
            "recon_degree_le2": float(
                per_cell[data.graph.degrees[val_rows] <= 2].mean()),
        }
        del model, trainer
        if device == "cuda":
            torch.cuda.empty_cache()

    summary = {"runs": rows, "B_stability": stability,
               "per_type_w_norm": per_type_w, "recon_strata": strata,
               "config": {"kappas": list(args.kappas), "seeds": list(args.seeds),
                          "alpha_a": args.alpha_a, "alpha_w": args.alpha_w,
                          "alpha_z": args.alpha_z, "omega": args.omega}}
    out = ds.root / "experiments" / "kappa_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--embeddings", default="egomask_ego_v1")
    parser.add_argument("--kappas", type=float, nargs="*", default=list(DEFAULT_KAPPAS))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--alpha-z", type=float, default=0.007)
    parser.add_argument("--alpha-w", type=float, default=0.1)
    parser.add_argument("--alpha-a", type=float, default=0.02)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--invariance", default="closed_form",
                        choices=("closed_form", "adversary"))
    parser.add_argument("--adv-steps", type=int, default=6)
    parser.add_argument("--adv-lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--tile-cells", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    if not args.report_only:
        fit_grid(args)
    print(json.dumps(report(args), indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
