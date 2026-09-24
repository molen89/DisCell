#!/usr/bin/env python3
"""The ablation sweeps -- kappa, d_w, alpha_w -- and their cross-run report.

Fits one model per (value, seed) of one swept knob and then reads every run
back into one comparison. Per the spec (7.7) and the architect review: **at
least three seeds per value**, because B is seed-bistable under the confound;
the reportable object is the effect envelope over value x seed jointly. A gene
programme stable along the knob but flapping across seeds is not a finding.

The report also **stratifies held-out reconstruction by the edges-lost QC
column**: partial isolation (degree 1-2 after pruning) still receives full
kappa against a thin rho_bar, and any effect concentrated in those cells is
suspect.

Every per-dataset knob the four slides need (label column, alpha_z, tile size,
variant, budget) is passed through, so one CLI runs the same three ablations on
any dataset; ``docs/sweep_programme.md`` holds the schedule.

Usage::

    python -m discell.model.sweep --dataset <id> --embeddings egomask_ego_v1
    python -m discell.model.sweep --dataset <id> --param alpha_w --tag aw
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
from discell.model.degeneracy import W_GUARD_NICHES, w_channel_guard
from discell.model.prepare import assemble
from discell.model.train import TrainConfig, Trainer

log = logging.getLogger("discell.model.sweep")

#: swept knob -> (run-name abbreviation, value type). The kappa abbreviation
#: is "k" so that sweep3's run names keep working unchanged.
PARAMS = {"kappa": ("k", float), "d_w": ("dw", int), "alpha_w": ("aw", float),
          "alpha_z": ("az", float)}
DEFAULT_VALUES = {"kappa": (0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
                  "d_w": (2, 3, 6, 8),
                  "alpha_w": (0.03, 0.05, 0.07, 0.1, 0.2, 0.3),
                  # the alpha_z ladder is per-dataset (multiples of
                  # 1/mean-count), so --values is required in practice
                  "alpha_z": (0.00175, 0.0035, 0.007, 0.014)}
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


def run_name(value: float, seed: int, tag: str = "sweep",
             param: str = "kappa") -> str:
    """``<tag>_<abbrev><value>_s<seed>``; an empty *tag* drops its segment.

    kappa keeps the historical spelling (``sweep3_k0.1_s0``); ``--tag ""``
    with ``--param d_w`` reproduces the hand-launched ``dw2_s0`` names.
    """
    parts = [tag, f"{PARAMS[param][0]}{value:g}", f"s{seed}"]
    return "_".join(part for part in parts if part)


def base_config(args: argparse.Namespace, name: str, seed: int,
                value) -> TrainConfig:
    """The fit's config: the sweep's fixed knobs, then the swept one on top."""
    fields = dict(
        dataset=args.dataset, variant=args.variant,
        embeddings=args.embeddings, label_key=args.label_key, run_name=name,
        kappa=args.kappa, seed=int(seed), d_w=args.d_w,
        alpha_z=args.alpha_z, alpha_w=args.alpha_w,
        alpha_a=args.alpha_a, omega=args.omega,
        invariance=args.invariance, adv_steps=args.adv_steps,
        adv_lr=args.adv_lr, gat_sources=args.gat_sources,
        epochs=args.epochs, patience=args.patience,
        tile_cells=args.tile_cells, figures_every=args.figures_every,
        device=args.device,
    )
    fields[args.param] = PARAMS[args.param][1](value)
    return TrainConfig(**fields)


def planned_fits(args: argparse.Namespace, runs_dir) -> list[tuple]:
    """The ``(name, value, seed)`` still to fit -- idempotence lives here.

    A run with a ``metrics.json`` is finished and is skipped, so an
    interrupted grid is resumed by relaunching the same command.
    """
    plan = []
    for seed in args.seeds:
        for value in args.values:
            name = run_name(value, seed, args.tag, args.param)
            if (runs_dir / name / "metrics.json").exists() and not args.force:
                log.info("skip %s (metrics.json exists)", name)
                continue
            plan.append((name, value, seed))
    return plan


def fit_grid(args: argparse.Namespace) -> None:
    runs_dir = paths.dataset(args.dataset).root / "runs"
    plan = planned_fits(args, runs_dir)
    if not plan:
        return
    data = assemble(args.dataset, args.variant, args.embeddings,
                    tile_cells=args.tile_cells, seed=0,
                    label_key=args.label_key)
    for name, value, seed in plan:
        # one split for every run: the data seed is fixed, only the model
        # seed varies (prepare.assemble draws the split from its own stream)
        Trainer(base_config(args, name, seed, value), data).fit()


def metric_row(param: str, value, seed: int, metrics: dict) -> dict:
    """One row of the comparison table from a run's parsed ``metrics.json``.

    Everything past ``best`` is read with ``.get``: the battery grew over
    time and older runs wrote fewer keys, so a missing instrument is None,
    never a crash.
    """
    final = metrics.get("final") or {}
    cycle = final.get("cycle") or {}

    def _mt(latent):
        # the headline cycle read is the cell-weighted pooled R^2 (issues M6:
        # r2_mean_types is dominated by non-cycling types' noise); the old
        # key is kept beside it
        return (cycle.get(latent) or {}).get("r2_pooled")

    def _mean_types(latent):
        return (cycle.get(latent) or {}).get("r2_mean_types")

    return {param: value, "value": value, "seed": seed,
            "recon": metrics["best"]["recon_val"],
            "nmi": metrics["best"]["nmi"],
            "epoch": metrics["best"]["epoch"],
            # the quality battery, from the final evaluation
            "cycle_r2_z": _mt("z"), "cycle_r2_w": _mt("w"),
            "cycle_r2_z_mean_types": _mean_types("z"),
            "cycle_r2_linear_ref": (_mt("linear_ref")
                                    if _mt("linear_ref") is not None
                                    else _mt("ceiling")           # pre-rename key
                                    if _mt("ceiling") is not None
                                    else _mean_types("ceiling")), # oldest runs: mean-of-types only
            "mirror_r2": (final.get("mirror") or {}).get("r2"),
            "probe_delta_ce": (final.get("probe") or {}).get("delta_ce"),
            # spec 7.10 degeneracy pair and the type-mean recon gap: carried
            # through whole, from wherever the producing run wrote them, and
            # only when it did (older runs wrote neither)
            "degeneracy": final.get("degeneracy",
                                    metrics.get("degeneracy")),
            "recon_gap": final.get("recon_gap", metrics.get("recon_gap"))}


def report_filename(param: str, tag: str) -> str:
    """``<param>_sweep[_<tag>].json``; the default tag keeps the old name.

    ``kappa_sweep_sweep3.json`` is what the handover and the paper cite, so
    the untagged spelling stays reserved for ``--tag sweep``.
    """
    return (f"{param}_sweep.json" if tag == "sweep"
            else f"{param}_sweep_{tag}.json" if tag
            else f"{param}_sweep_untagged.json")


def b_stability(loaded: dict, values, seeds, param: str) -> dict:
    """B agreement within a value across seeds, and along the value axis.

    Both legs are matched-column |corr|. Along the axis every run is compared
    to the first loaded one; in a d_w sweep the shapes differ, and those pairs
    are dropped rather than reported as a number that does not mean anything.
    """
    stability = {"across_seeds": {}, f"along_{param}": {}}
    for value in values:
        seeds_here = [s for s in seeds if (value, s) in loaded]
        pairs = [matched_correlation(loaded[(value, a)]["B"],
                                     loaded[(value, b)]["B"])
                 for i, a in enumerate(seeds_here) for b in seeds_here[i + 1:]]
        if pairs:
            stability["across_seeds"][f"{value:g}"] = {
                "mean": float(np.mean(pairs)), "min": float(np.min(pairs))}
    anchor = next(((v, s) for v in values for s in seeds if (v, s) in loaded),
                  None)
    if anchor is None:
        return stability
    for (value, seed), entry in loaded.items():
        if entry["B"].shape != loaded[anchor]["B"].shape:
            continue            # d_w sweep: columns are not comparable
        stability[f"along_{param}"][run_name(value, seed, "", param)] = \
            matched_correlation(loaded[anchor]["B"], entry["B"])
    return stability


def report(args: argparse.Namespace) -> dict:
    """Read every finished run back into the cross-(value x seed) comparison."""
    import torch

    from discell.model.networks import DisCell
    from discell.model.validate import niche_labels

    ds = paths.dataset(args.dataset)
    data = assemble(args.dataset, args.variant, args.embeddings,
                    tile_cells=args.tile_cells, seed=0,
                    label_key=args.label_key)

    rows: list[dict] = []
    loaded: dict[tuple, dict] = {}
    for seed in args.seeds:
        for value in args.values:
            run_dir = ds.root / "runs" / run_name(value, seed, args.tag,
                                                  args.param)
            if not (run_dir / "best.pt").exists():
                log.warning("missing %s -- skipped", run_dir.name)
                continue
            payload = torch.load(run_dir / "best.pt", map_location="cpu",
                                 weights_only=False)
            payload["config"].setdefault("gat_sources", "type_z")
            payload["config"].setdefault("subtract_leak", False)
            payload["config"].setdefault("gat_sink", False)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            b_matrix = payload["model"]["B.weight"].numpy()      # (G, d_w)
            loaded[(value, seed)] = {"B": b_matrix, "payload": payload}
            rows.append(metric_row(args.param, value, seed, metrics))

    stability = b_stability(loaded, args.values, args.seeds, args.param)

    # -- per-cell effects: |w| per type, and recon stratified by QC --------
    per_type_w: dict = {}
    strata: dict = {}
    # the dead-context-channel guard (2026-09-23): one entry per run, and the
    # same two numbers merged into that run's row of the table
    w_guard: dict = {}
    niche_cache: dict = {}
    row_by_key = {(r["value"], r["seed"]): r for r in rows}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lost = data.graph.pruned_per_cell
    for (value, seed), entry in loaded.items():
        state = entry["payload"]
        model = DisCell(n_genes=data.x.shape[1], n_types=len(data.p_t),
                        phi_dim=data.phi.shape[1],
                        median_counts=data.median_counts,
                        d_z=state["config"]["d_z"], d_w=state["config"]["d_w"],
                        hidden=state["config"]["hidden"],
                        gat_dim=state["config"]["gat_dim"],
                        heads=state["config"]["heads"],
                        gat_sources=state["config"]["gat_sources"],
                        subtract_leak=state["config"]["subtract_leak"],
                        gat_sink=state["config"]["gat_sink"]).to(device)
        model.load_state_dict(state["model"])
        trainer = Trainer(TrainConfig(**state["config"]), data)
        trainer.model = model.eval()
        sweep_out = trainer._sweep(trainer.val_batches, want_log_p=True)
        val_rows = sweep_out["nodes"]
        x_val = np.vstack([data.x[b["nodes"][:b["n_seeds"]]].toarray()
                           for b in trainer.val_batches])

        key = run_name(value, seed, "", args.param)
        if seed not in niche_cache:
            niche_cache[seed] = niche_labels(data, W_GUARD_NICHES, seed)
        guard = w_channel_guard(sweep_out["mu_w"], niche_cache[seed][val_rows],
                                data.t[val_rows], seed=seed)
        w_guard[key] = guard
        row_by_key[(value, seed)].update(
            w_niche_mi=guard["w_niche_mi"],
            w_niche_mi_floor=guard["w_niche_mi_floor"],
            w_niche_mi_excess=guard["w_niche_mi_excess"],
            w_var_fraction_across_cells=guard["w_var_fraction_across_cells"],
            failed_fit=guard["flag"])
        if guard["flag"]:
            log.warning("%s: %s (I(niche;w) %.4f, floor %.4f)", key,
                        guard["flag"], guard["w_niche_mi"],
                        guard["w_niche_mi_floor"])
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
               "w_channel_guard": w_guard,
               "config": {"param": args.param, "values": list(args.values),
                          "seeds": list(args.seeds), "dataset": args.dataset,
                          "variant": args.variant, "label_key": args.label_key,
                          "tile_cells": args.tile_cells, "epochs": args.epochs,
                          "patience": args.patience, "kappa": args.kappa,
                          "d_w": args.d_w, "alpha_a": args.alpha_a,
                          "alpha_w": args.alpha_w, "alpha_z": args.alpha_z,
                          "omega": args.omega}}
    out = ds.root / "experiments" / report_filename(args.param, args.tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--embeddings", default="egomask_ego_v1")
    parser.add_argument("--param", default="kappa", choices=tuple(PARAMS),
                        help="the swept knob; everything else stays fixed")
    parser.add_argument("--values", type=float, nargs="*", default=None,
                        help="grid for --param (default: the pre-registered "
                             "grid of that ablation)")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--label-key", default=None,
                        help="obs column for t (None = the bundle default; "
                             "graphclust on the annotation-free slides)")
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--d-w", type=int, default=6)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--alpha-z", type=float, default=0.007,
                        help="per-dataset: 1/mean-count (ovarian 0.007, lung "
                             "0.004, FF 0.0007, GSE core 0.0036)")
    parser.add_argument("--alpha-w", type=float, default=0.1)
    parser.add_argument("--alpha-a", type=float, default=0.3)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--invariance", default="adversary",
                        choices=("closed_form", "adversary"))
    parser.add_argument("--adv-steps", type=int, default=6)
    parser.add_argument("--gat-sources", default="type_only",
                        choices=("type_z", "type_only"))
    parser.add_argument("--adv-lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--tile-cells", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="sweep",
                        help="run-name prefix; a new tag never touches an old "
                             "sweep's runs")
    parser.add_argument("--figures-every", type=int, default=50,
                        help="sparser than a reference run: metrics log every "
                             "eval regardless; figures are heavy x18")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse, then fill the grid of whichever knob is being swept."""
    args = build_parser().parse_args(argv)
    cast = PARAMS[args.param][1]
    args.values = [cast(v) for v in (args.values
                                     if args.values is not None
                                     else DEFAULT_VALUES[args.param])]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
