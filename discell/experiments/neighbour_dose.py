#!/usr/bin/env python3
"""Neighbour dose: what the context ``c`` can and cannot see (devlog 2026-09-16).

Under ``type_only`` the GAT part of ``c_i`` is a softmax-weighted composition of
neighbour types, so the *number* of neighbours of a type cannot change it --
only their *fraction* can. This probes a pinned model with synthetic one-cell
tiles (a receiver of type ``t`` plus ``n`` neighbours of chosen types, Phi
fixed) and reads the realised prior shift ``B . m_psi(c, t)`` in gene space,
always as a difference (the w gauge is not identified).

Setups: A count at fixed composition (expected exactly zero), B fraction at
degree 6, C composition swap at degree 6, D the image channel alone (Phi from
real cells, composition fixed), E the natural spread on real cells split by
channel (real neighbours / real Phi / both). Scale reference per receiver: its measured
within-type realised shift ``mean ||B (w - mean_t w)||``.

Usage::

    python -m discell.experiments.neighbour_dose --dataset <id> --run <run>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

import numpy as np

from discell import paths
from discell.model.validate import collect_latents, load_run

log = logging.getLogger("discell.experiments.neighbour_dose")

RECEIVERS = ("Macrophages", "T and NK Cells", "Tumor Associated Fibroblasts",
             "Tumor Cells")
SOURCES = ("Tumor Cells", "Tumor Associated Fibroblasts", "Macrophages",
           "T and NK Cells", "Smooth Muscle Cells")
DEGREE = 6
N_PHI_CELLS = 500


def prior_mean(model, receiver: int, neighbours: Sequence[int],
               phi_rows: np.ndarray, device) -> np.ndarray:
    """``m_psi(c, t)`` for one receiver under one synthetic neighbourhood,
    for each Phi row given: ``(len(phi_rows), d_w)``."""
    import torch
    import torch.nn.functional as F

    n = len(neighbours)
    t = torch.tensor([receiver] + list(neighbours), device=device)
    edge_src = torch.arange(1, n + 1, device=device)
    edge_dst = torch.zeros(n, dtype=torch.long, device=device)
    isolated = torch.tensor([n == 0] + [False] * n, device=device)
    mu_z = torch.zeros(n + 1, model.d_z, device=device)     # unread: type_only
    out = []
    with torch.no_grad():
        for row in phi_rows:
            phi = torch.zeros(n + 1, len(row), device=device)
            phi[0] = torch.as_tensor(row, device=device)
            c, _ = model.context(mu_z, t, phi, isolated, edge_src, edge_dst, 1)
            t_c = F.one_hot(t[:1], model.n_types).float()
            out.append(model.prior_w(torch.cat([c, t_c], dim=-1))[0].cpu().numpy())
    return np.stack(out)


def run(args: argparse.Namespace) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, data, trainer, run_dir, B = load_run(args.dataset, args.run,
                                                 args.device)
    if config.gat_sources != "type_only":
        raise SystemExit("the synthetic tile assumes type_only GAT sources")
    model = trainer.model
    device = next(model.parameters()).device
    names = [str(n) for n in data.type_names]
    idx = {n: names.index(n) for n in RECEIVERS + SOURCES}
    connected = data.graph.degrees > 0
    rng = np.random.default_rng(args.seed)

    # scale reference: the measured within-type realised shift
    w = collect_latents(trainer, data)["mu_w"]
    reference = {}
    for name in RECEIVERS:
        m = (data.t == idx[name]) & connected
        reference[name] = float(np.linalg.norm(
            (w[m] - w[m].mean(0)) @ B.T, axis=1).mean())

    def shift(receiver, neighbours, phi_rows):
        return prior_mean(model, receiver, neighbours, phi_rows, device) @ B.T

    results = {"run": args.run, "reference": reference, "A": {}, "B": {},
               "C": {}, "D": {}, "E": {}}
    for name in RECEIVERS:
        r = idx[name]
        m = (data.t == r) & connected
        phi_mean = data.phi[m].mean(0, keepdims=True)
        tumour, own = idx["Tumor Cells"], r
        # A: count at fixed composition
        base = shift(r, [tumour], phi_mean)[0]
        base_half = shift(r, [tumour, own], phi_mean)[0]
        results["A"][name] = {
            "isolated": float(np.linalg.norm(shift(r, [], phi_mean)[0] - base)),
            "tumour_only": [float(np.linalg.norm(shift(r, [tumour] * n, phi_mean)[0] - base))
                            for n in range(1, 9)],
            "half_half": [float(np.linalg.norm(shift(r, [tumour, own] * k, phi_mean)[0] - base_half))
                          for k in range(1, 5)]}
        # B: fraction at fixed degree, own-type background
        zero = shift(r, [own] * DEGREE, phi_mean)[0]
        results["B"][name] = {
            source: [float(np.linalg.norm(
                shift(r, [idx[source]] * k + [own] * (DEGREE - k), phi_mean)[0] - zero))
                for k in range(DEGREE + 1)]
            for source in ("Tumor Cells", "Tumor Associated Fibroblasts")}
        # C: composition swap at degree 6
        results["C"][name] = {
            source: float(np.linalg.norm(shift(r, [idx[source]] * DEGREE, phi_mean)[0] - zero))
            for source in SOURCES if idx[source] != r}
        # D: the image channel alone
        rows = rng.choice(np.flatnonzero(m), min(N_PHI_CELLS, m.sum()), replace=False)
        shifts = shift(r, [own] * DEGREE, data.phi[rows])
        results["D"][name] = float(np.linalg.norm(shifts - shifts.mean(0), axis=1).mean())
        # E: natural variation on the same real cells -- real neighbours with
        # the mean Phi, mean-composition (6 own) with real Phi, and both real --
        # the like-for-like split of w's context-dependence between channels
        real_nb = [data.t[data.graph.in_edges[i].indices] for i in rows]
        comp_only = np.stack([shift(r, nb, phi_mean)[0] for nb in real_nb])
        both = np.stack([shift(r, nb, data.phi[i:i + 1])[0]
                         for nb, i in zip(real_nb, rows)])
        spread = lambda a: float(np.linalg.norm(a - a.mean(0), axis=1).mean())
        results["E"][name] = {"composition_only": spread(comp_only),
                              "phi_only": results["D"][name], "both": spread(both)}
        log.info("%s: natural spread -- composition only %.1f | Phi only %.1f | both %.1f",
                 name, *results["E"][name].values())
        log.info("%s: reference %.1f | A max %.2e | B tumour k=6 %.1f | C max %.1f | D %.1f",
                 name, reference[name], max(results["A"][name]["tumour_only"]),
                 results["B"][name]["Tumor Cells"][-1],
                 max(results["C"][name].values()), results["D"][name])

    out_dir = paths.dataset(args.dataset).root / "experiments"
    out_dir.mkdir(exist_ok=True)
    stem = f"neighbour_dose_{args.run}"
    (out_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colours = dict(zip(RECEIVERS, plt.get_cmap("tab10").colors))
    ax = axes[0, 0]
    for name in RECEIVERS:
        a = results["A"][name]
        ax.plot(range(1, 9), a["tumour_only"], "o-", color=colours[name], label=name)
        ax.plot(range(2, 9, 2), a["half_half"], "s--", color=colours[name], alpha=0.6)
        ax.plot([0], [a["isolated"]], "x", color=colours[name], ms=9)
    ax.set_xlabel("number of neighbours")
    ax.set_ylabel("‖B·(m_ψ(n) − m_ψ(n=1))‖")
    ax.set_title("A. count at fixed composition (expected 0)\n"
                 "solid: all tumour · dashed: 50/50 tumour+own · x: isolated", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    for name in RECEIVERS:
        b = results["B"][name]
        ax.plot(np.arange(DEGREE + 1) / DEGREE, b["Tumor Cells"], "o-", color=colours[name], label=f"{name} ← tumour")
        ax.plot(np.arange(DEGREE + 1) / DEGREE, b["Tumor Associated Fibroblasts"], "s--", color=colours[name], alpha=0.6)
        ax.axhline(reference[name], color=colours[name], lw=0.8, ls=":")
    ax.set_xlabel("source fraction among 6 neighbours")
    ax.set_ylabel("‖B·(m_ψ(k) − m_ψ(0))‖")
    ax.set_title("B. fraction at degree 6, own-type background\n"
                 "solid: tumour source · dashed: TAF source · dotted: measured within-type shift", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    width = 0.8 / (len(SOURCES) + 1)
    for i, source in enumerate(SOURCES):
        vals = [results["C"][name].get(source, np.nan) for name in RECEIVERS]
        ax.bar(np.arange(len(RECEIVERS)) + i * width, vals, width, label=f"all {source}")
    ax.bar(np.arange(len(RECEIVERS)) + len(SOURCES) * width,
           [results["D"][name] for name in RECEIVERS], width, color="k", hatch="//",
           label="D. Φ alone (real cells, 6 own)")
    ax.set_xticks(np.arange(len(RECEIVERS)) + 0.4, [n[:14] for n in RECEIVERS], fontsize=8)
    ax.set_ylabel("‖B·Δm_ψ‖ vs 6 own-type neighbours")
    ax.set_title("C. composition swap at degree 6, and D. the image channel", fontsize=10)
    ax.legend(fontsize=6)
    ax = axes[1, 1]
    ratio = [results["D"][name] / max(results["C"][name].values()) for name in RECEIVERS]
    ax.bar(range(len(RECEIVERS)), ratio, color=[colours[n] for n in RECEIVERS])
    ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(range(len(RECEIVERS)), [n[:14] for n in RECEIVERS], fontsize=8)
    ax.set_ylabel("D / max C")
    ax.set_title("image-channel spread over the largest composition swap", fontsize=10)
    fig.suptitle(f"neighbour dose through m_ψ — {args.run} (gene-space norms of B·Δm_ψ)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"{stem}.png", dpi=130)
    log.info("wrote %s", out_dir / f"{stem}.png")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
