"""Figure fig:synthetic-misspec: recovery on simulated sections fitted at a wrong leak rate.

The worlds planted at kappa = 0.2, fitted with the final configuration at each assumed
kappa in {0, 0.1, 0.2, 0.3, 0.4}. One panel per recovery read: NMI of mu_z with the
planted type; first canonical correlation of mu_w with the planted response; cosine of
the fitted against the planted loadings (with the 95th percentile of random subspaces,
the reference of tab:synthetic); within-type R^2 of neighbour composition from the
intrinsic latent. Line: mean over the nine fits (three worlds x three model seeds);
band: their range. The planted value kappa = 0.2 is marked.

Source: data/datasets/synthetic_smoke/experiments/synthetic_recovery.json (groups with
arm 'final', planted_kappa 0.2; world_references), the file tab:synthetic reads (path
from scripts/paper_tables.py).

Outputs ../fig_synthetic_misspec.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()
OUT = S.out_path(Path(__file__).resolve().parent.parent / "fig_synthetic_misspec")
d = json.loads((PT.SYNTH / "synthetic_recovery.json").read_text())
if not d["simulator_checks"]["all_pass"]:
    raise SystemExit("synthetic simulator checks do not all pass")
PLANTED = 0.2
# short titles; y labels name the quantity, the upright marker above each axis its direction
READS = [("nmi_z", "type", r"NMI of $\boldsymbol{\mu}_z$ with type", True),
         ("w_cca", "response", r"CCA of $\boldsymbol{\mu}_w$, response", True),
         ("b_cosine", "loadings", "largest cosine", True),
         ("comp_r2", "composition leakage", r"composition $R^2$ from $\boldsymbol{\mu}_z$", False)]
groups = sorted((g for g in d["groups"] if g["arm"] == "final" and g["planted_kappa"] == PLANTED),
                key=lambda g: g["assumed_kappa"])
ks = [g["assumed_kappa"] for g in groups]

S.apply()
fig, axes = plt.subplots(1, 4, figsize=(S.TEXTWIDTH, 1.9))
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.86, wspace=0.50)
for c, (ax, (key, title, ylab, up)) in enumerate(zip(axes, READS)):
    S.light_grid(ax, "y")
    lo, mid, hi = [], [], []
    for g in groups:
        v = [f[key] for f in g["fits"]]
        assert len(v) == g[key]["n"] and abs(min(v) - g[key]["min"]) < 1e-12
        lo.append(min(v)); hi.append(max(v)); mid.append(float(np.mean(v)))
    ax.axvline(PLANTED, color=S.CELL_EDGE, lw=0.6, zorder=2.5)
    ax.fill_between(ks, lo, hi, color=S.BLUE_BAND, lw=0, zorder=2)
    ax.plot(ks, mid, color=S.BLUE, lw=1.1, marker="o", ms=2.5, zorder=3)
    if key == "b_cosine":
        wr = d["world_references"][f"{PLANTED:g}"]["b_cosine_random_q95"]
        ax.axhline(wr["max"], color=S.INK2, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
        ax.text(0.41, wr["max"] + 0.02, "random, 95th pct.", fontsize=5.8, color=S.INK2, ha="right", va="bottom")
        ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4], ["0", "0.1", r"\textbf{0.2}", "0.3", "0.4"])
    ax.set_xlim(-0.02, 0.42)
    ax.set_xlabel(r"assumed $\kappa$", labelpad=1)
    ax.set_title(title, fontsize=7, color=S.INK, pad=3)
    ax.set_ylabel(ylab, fontsize=6.5, labelpad=2)
    S.ybetter(ax, up=up, fontsize=6)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
axes[-1].set_ylim(bottom=0)
axes[0].text(PLANTED + 0.01, 0.03, "planted", transform=axes[0].get_xaxis_transform(),
             fontsize=5.8, color=S.INK2, ha="left", va="bottom")
S.save(fig, OUT)
