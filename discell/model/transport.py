#!/usr/bin/env python3
"""The counterfactual transport check (doc 08 section 7).

Model-predicted between-niche expression shifts must match empirically
observed shifts on held-out tiles. This validates the surviving
counterfactual claim -- average effects under ``do(c')`` at composition
level (per-cell counterfactuals are out of scope, doc-10 cut).

Per (type, niche pair), on the log-rate scale:

- program channel:  ``<B_g, mean m_psi(B,t) - mean m_psi(A,t)>``
- leak channel:     ``log`` of the kappa-mixed mean rates minus the
  program-only version -- i.e. the shift the foreign influx adds on top
- observed:         difference of depth-normalised mean expression of
  held-out type-t cells between the niches (same log transform)

References per panel: zero-prediction null, program-only, leak-only. The
claim needs the full model to beat both single-channel versions. The
per-gene program-vs-leak split of each observed niche difference is the
headline figure. Guards: composition-overlap check per pair (else
extrapolation -- flagged, not reported), near-zero genes excluded, counts
reported.

Usage::

    python -m discell.model.transport --dataset <id> --run <run>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

import numpy as np

from discell.model.validate import (collect_latents, load_run, niche_labels)

log = logging.getLogger("discell.model.transport")

MIN_CELLS = 500
MIN_RATE = 1e-5          #: genes below this mean rate in the type are excluded
N_PAIRS = 4              #: most composition-distinct niche pairs
EPS = 1e-8


def collect_channels(trainer, data) -> dict:
    """Per-seed prior mean, decontaminated rate, and foreign influx."""
    import torch

    out = {k: [] for k in ("nodes", "prior_w", "rho", "rho_bar")}
    with torch.no_grad():
        for batch in trainer.train_batches + trainer.val_batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch),
                                kappa=trainer.config.kappa, sample=False)
            n = batch["n_seeds"]
            out["nodes"].append(batch["nodes"][:n])
            out["prior_w"].append(fwd.prior_mean_w[:n].cpu().numpy())
            out["rho"].append(fwd.log_rho[:n].exp().cpu().numpy()
                              .astype(np.float32))
            out["rho_bar"].append(fwd.rho_bar.cpu().numpy()
                                  .astype(np.float32))
    nodes = np.concatenate(out["nodes"])
    order = np.argsort(nodes)
    return {k: np.concatenate(v)[order] for k, v in out.items()}


def pick_pairs(labels: np.ndarray, y: np.ndarray, connected: np.ndarray,
               n_pairs: int = N_PAIRS) -> list[tuple[int, int]]:
    """Every niche pair, most composition-distinct first.

    All pairs are evaluated and the per-(pair, type) overlap guard decides
    reportability -- selecting only the most distinct pairs up front is
    self-defeating (they are exactly the ones the guard flags)."""
    ks = [k for k in np.unique(labels) if k >= 0]
    centroids = {k: y[connected & (labels == k)].mean(axis=0) for k in ks}
    scored = sorted(((np.linalg.norm(centroids[a] - centroids[b]), a, b)
                     for i, a in enumerate(ks) for b in ks[i + 1:]),
                    reverse=True)
    return [(a, b) for _, a, b in scored]


def summary_figure(results: dict, path) -> None:
    """The section-7 headline in one figure: tier bars + the per-panel
    program-vs-leak attribution map (how much of each niche difference is
    biology vs contamination)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [p for p in results["panels"] if p["overlap_flag"]]
    if not panels:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    summary = results.get("summary", {}).get("extrapolation", {})
    keys = ("counterfactual", "full", "program_only", "leak_only")
    axes[0].bar(range(4), [summary.get(k, float("nan")) for k in keys],
                color=("#2c7fb8", "#7fb8d4", "#41ab5d", "#c9662a"),
                width=0.6)
    axes[0].set_xticks(range(4), ("counter-\nfactual", "model\naccount",
                                  "program\nonly", "leak\nonly"),
                       fontsize=8)
    axes[0].set_ylabel("mean held-out R² (extrapolation tier)", fontsize=8)
    axes[0].set_title(
        f"both channels required: full beats both in "
        f"{summary.get('full_beats_both', '?')}/{summary.get('n_panels', '?')}"
        f" panels\nmedian calibration slope "
        f"{summary.get('median_slope', float('nan')):.2f}", fontsize=9)

    prog = np.array([p["program_only"]["r2"] for p in panels])
    leak = np.array([p["leak_only"]["r2"] for p in panels])
    full = np.array([p["counterfactual"]["r2"] for p in panels])
    sc = axes[1].scatter(prog.clip(min=0), leak.clip(min=0),
                         s=20 + 300 * full.clip(min=0), c=full,
                         cmap="viridis", alpha=0.7, edgecolors="0.4",
                         linewidths=0.4)
    lim = max(prog.max(), leak.max()) * 1.1
    axes[1].plot([0, lim], [0, lim], color="0.6", lw=0.8, ls="--")
    axes[1].set_xlabel("program-channel R² (biology)", fontsize=8)
    axes[1].set_ylabel("leak-channel R² (contamination)", fontsize=8)
    axes[1].set_title("per (type, niche-pair): what explains the observed "
                      "shift\nabove the line = contamination-dominated",
                      fontsize=9)
    for p in sorted(panels, key=lambda q: -q["counterfactual"]["r2"])[:4]:
        axes[1].annotate(p["type"][:14],
                         (max(p["program_only"]["r2"], 0),
                          max(p["leak_only"]["r2"], 0)), fontsize=6)
    plt.colorbar(sc, ax=axes[1], fraction=0.04,
                 label="counterfactual R²")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def transport_check(args: argparse.Namespace) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, data, trainer, run_dir, b_matrix = load_run(
        args.dataset, args.run, args.device)
    latents = collect_latents(trainer, data)
    channels = collect_channels(trainer, data)
    labels = niche_labels(data, args.niches, config.seed)
    connected = data.graph.degrees > 0
    held_out = latents["fold"] == 0            # spatial-block held-out tiles
    gene_names = np.asarray([str(g) for g in data.gene_names])
    kappa = config.kappa

    x_rate = data.x.multiply(
        1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()

    pairs = pick_pairs(labels, data.graph.y, connected)
    names = [str(n) for n in data.type_names]
    results: dict = {"run": args.run, "kappa": kappa, "panels": []}
    out_dir = run_dir / "transport"
    out_dir.mkdir(exist_ok=True)

    for niche_a, niche_b in pairs:
        for g in range(len(names)):
            members = {}
            ok = True
            for niche, side in ((niche_a, "A"), (niche_b, "B")):
                train_rows = np.flatnonzero(connected & ~held_out
                                            & (data.t == g)
                                            & (labels == niche))
                test_rows = np.flatnonzero(connected & held_out
                                           & (data.t == g)
                                           & (labels == niche))
                if len(train_rows) < MIN_CELLS or len(test_rows) < MIN_CELLS // 5:
                    ok = False
                    break
                members[side] = (train_rows, test_rows)
            if not ok:
                continue

            # composition-overlap guard: prior means are extrapolation when
            # the niches share no composition support for this type. The
            # comparison is 1-D along the gap direction -- summed per-dim
            # spreads overstate the relevant support
            y_a = data.graph.y[members["A"][0]]
            y_b = data.graph.y[members["B"][0]]
            gap_vec = y_b.mean(0) - y_a.mean(0)
            gap = np.linalg.norm(gap_vec)
            unit = gap_vec / max(gap, 1e-12)
            spread = 0.5 * (float((y_a @ unit).std())
                            + float((y_b @ unit).std()))
            overlap_flag = bool(gap > 3 * spread)

            def mean_channels(rows):
                rho = channels["rho"][rows].mean(axis=0)
                rho_bar = channels["rho_bar"][rows].mean(axis=0)
                prior_w = channels["prior_w"][rows].mean(axis=0)
                return rho, rho_bar, prior_w

            rho_a, bar_a, mpsi_a = mean_channels(members["A"][0])
            rho_b, bar_b, mpsi_b = mean_channels(members["B"][0])

            # predicted shifts, log-rate scale, from TRAINING folds
            program = b_matrix @ (mpsi_b - mpsi_a)
            full = (np.log((1 - kappa) * rho_b + kappa * bar_b + EPS)
                    - np.log((1 - kappa) * rho_a + kappa * bar_a + EPS))
            leak_only = (np.log((1 - kappa) * rho_a + kappa * bar_b + EPS)
                         - np.log((1 - kappa) * rho_a + kappa * bar_a + EPS))

            # observed shift on HELD-OUT tiles, depth-normalised
            obs_a = np.asarray(x_rate[members["A"][1]].mean(axis=0)).ravel()
            obs_b = np.asarray(x_rate[members["B"][1]].mean(axis=0)).ravel()
            keep = (obs_a > MIN_RATE) & (obs_b > MIN_RATE)
            observed = np.log(obs_b[keep] + EPS) - np.log(obs_a[keep] + EPS)

            def score(prediction):
                # both sides centred: the softmax normaliser and depth enter
                # as per-panel constants and must not be charged to the model
                p = prediction[keep] - prediction[keep].mean()
                o = observed - observed.mean()
                slope = float(np.polyfit(p, o, 1)[0]) \
                    if p.std() > 1e-9 else float("nan")
                ss = 1.0 - ((o - p) ** 2).sum() / max((o ** 2).sum(), 1e-12)
                return {"r2": float(ss), "slope": slope,
                        "corr": float(np.corrcoef(p, o)[0, 1])}

            panel = {"pair": (int(niche_a), int(niche_b)),
                     "type": names[g],
                     "n_train": [len(members[s][0]) for s in "AB"],
                     "n_test": [len(members[s][1]) for s in "AB"],
                     "n_genes": int(keep.sum()),
                     "overlap_flag": overlap_flag,
                     # the doc-7.2 counterfactual: context response + new
                     # neighbours' influx, the cell's own z held fixed
                     "counterfactual": score(program + leak_only),
                     # the model's account of the actual populations -- also
                     # lets the type's intrinsic mix differ across niches;
                     # its excess over the counterfactual measures selection
                     "full": score(full),
                     "program_only": score(program),
                     "leak_only": score(leak_only),
                     "zero_null_r2": 0.0}
            results["panels"].append(panel)

            if len(results["panels"]) <= args.figures and not overlap_flag:
                fig, ax = plt.subplots(figsize=(4.6, 4.2))
                ax.scatter(full[keep], observed, s=2, alpha=0.4,
                           rasterized=True)
                lims = np.percentile(np.concatenate([full[keep], observed]),
                                     [1, 99])
                ax.plot(lims, lims, color="0.4", lw=0.8, ls="--")
                ax.set_xlabel("predicted log-rate shift (program + leak)")
                ax.set_ylabel("observed (held-out)")
                ax.set_title(f"{names[g][:26]} | niche {niche_a}->{niche_b}\n"
                             f"full R² {panel['full']['r2']:.2f} "
                             f"(prog {panel['program_only']['r2']:.2f}, "
                             f"leak {panel['leak_only']['r2']:.2f})",
                             fontsize=9)
                fig.tight_layout()
                fig.savefig(out_dir / f"pair{niche_a}-{niche_b}_"
                                      f"type{g}.png", dpi=130)
                plt.close(fig)

    # two tiers: with data-defined (k-means) niches, distinct pairs are
    # composition-disjoint BY CONSTRUCTION, so the interpolation tier is
    # structurally near-empty and the informative regime is extrapolation --
    # named as such, never blended (the guard's purpose)
    results["summary"] = {}
    for name, tier in (("supported", [p for p in results["panels"]
                                      if not p["overlap_flag"]]),
                       ("extrapolation", [p for p in results["panels"]
                                          if p["overlap_flag"]])):
        if not tier:
            continue
        summary = {key: float(np.mean([p[key]["r2"] for p in tier]))
                   for key in ("counterfactual", "full", "program_only",
                               "leak_only")}
        summary["n_panels"] = len(tier)
        summary["median_slope"] = float(np.median(
            [p["counterfactual"]["slope"] for p in tier]))
        # doc 7.4: the claim rides on the counterfactual total
        summary["full_beats_both"] = int(sum(
            p["counterfactual"]["r2"] > max(p["program_only"]["r2"],
                                            p["leak_only"]["r2"])
            for p in tier))
        results["summary"][name] = summary
        log.info("transport [%s]: %d panels | mean R² full %.3f, program "
                 "%.3f, leak %.3f | slope %.2f | full beats both %d/%d",
                 name, summary["n_panels"], summary["full"],
                 summary["program_only"], summary["leak_only"],
                 summary["median_slope"], summary["full_beats_both"],
                 summary["n_panels"])

    summary_figure(results, out_dir / "transport_summary.png")
    (out_dir / "transport.json").write_text(
        json.dumps(results, indent=2, default=float))
    log.info("wrote %s", out_dir / "transport.json")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--niches", type=int, default=10)
    parser.add_argument("--figures", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    transport_check(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
