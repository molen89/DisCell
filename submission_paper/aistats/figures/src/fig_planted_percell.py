"""Figure fig:planted-percell: the planted per-cell response against its strength s.

(a) KL AUC of the response posterior, planted cells against the other cells of their
type, with the rule's 0.8 line; (b) absorption of the plant by the intrinsic latent; (c)
the cross-seed Spearman correlation of the per-cell divergence, with the rule's 0.5
line. World 0 (the one the rule, fixed in advance, decides on) in bold; worlds 1 and 2
(replications) lighter. Lines join the mean over model seeds (seed pairs in c); bars are
the range.

Source: data/datasets/synthetic_smoke/experiments/planted_percell.json (groups[*].per_seed,
groups[*].cross_seed, rule), the file tab:planted-percell reads (path from
scripts/paper_tables.py).

Outputs ../fig_planted_percell.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()
OUT = Path(__file__).resolve().parent.parent / "fig_planted_percell"
d = json.loads((PT.SYNTH / "planted_percell.json").read_text())
rule = d["rule"]
# KL AUC: higher is better (detection); absorption and the cross-seed agreement have no
# direction of their own, only the rule's line. (key, title, y label, line)
PANELS = [("kl_auc", r"planted vs.\ other cells", "KL AUC", rule["auc_min"]),
          ("z_absorption", r"absorption by $\mathbf{z}$", r"absorption by $\mathbf{z}$", None),
          ("spearman", r"agreement between seeds", r"Spearman $\rho$ of the KL", rule["spearman_min"])]


def vals(g, key):
    if key == "spearman":
        return [c[key] for c in g["cross_seed"]]
    return [x[key] for x in g["per_seed"]]


S.apply()
fig, axes = plt.subplots(1, 3, figsize=(S.TEXTWIDTH, 1.85))
fig.subplots_adjust(left=0.09, right=0.985, bottom=0.21, top=0.88, wspace=0.40)
worlds = sorted({g["world_seed"] for g in d["groups"]})
for c, (ax, (key, title, ylab, line)) in enumerate(zip(axes, PANELS)):
    S.light_grid(ax, "y")
    for w in worlds:
        gs = sorted((g for g in d["groups"] if g["world_seed"] == w), key=lambda g: g["s"])
        s = np.array([g["s"] for g in gs])
        v = [vals(g, key) for g in gs]
        for g, vv in zip(gs, v):            # the stored summary is the per-seed range
            assert abs(min(vv) - g[key]["min"]) < 1e-12 and abs(max(vv) - g[key]["max"]) < 1e-12
        m = np.array([np.mean(x) for x in v])
        lo, hi = m - [min(x) for x in v], [max(x) for x in v] - m
        main = w == rule["world_seed"]
        dx = {0: 0.0, 1: -0.04, 2: 0.04}.get(w, 0.0)      # dodge so the bars do not overlap
        col = S.BLUE if main else S.BLUE_RAMP[2]
        ax.errorbar(s + dx, m, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=0.9 if main else 0.6,
                    capsize=1.5, zorder=3 if main else 2)
        ax.plot(s + dx, m, color=col, lw=1.6 if main else 0.8, marker="o", ms=3.2 if main else 2.2,
                zorder=4 if main else 2)
        if c == 0:
            ax.annotate(f"world {w}", (s[-1] + dx, m[-1]),
                        xytext=(4, {0: 0, 1: -6, 2: 6}.get(w, 0)), textcoords="offset points",
                        fontsize=6, color=S.INK if main else S.INK2, va="center", ha="left")
    if line is not None:
        ax.axhline(line, color=S.INK2, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
        # two lines, so it ends before the bars at s = 1
        ax.text(0.01, line + 0.005, "detection\n" rf"threshold (${line:g}$)",
                transform=ax.get_yaxis_transform(), fontsize=5.5, color=S.INK2, ha="left", va="bottom",
                linespacing=1.0)
    ax.set_xticks([0, 0.5, 1, 2])
    ax.set_xlim(-0.12, 2.12 if c else 2.55)
    ax.set_xlabel(r"strength $s$ of the plant", labelpad=1)
    ax.set_title(title, fontsize=7.5, color=S.INK, pad=3)
    ax.set_ylabel(ylab, fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6.5)
    S.panel_label(ax, "abc"[c], x=-0.17, y=1.11)   # above the direction marker
    if key == "kl_auc":
        S.ybetter(ax, up=True, fontsize=6)
axes[0].set_ylim(top=1.0)
axes[2].set_ylim(0, 0.8)
axes[1].set_ylim(bottom=-0.03)
S.save(fig, OUT)
print("rule: smallest detected s on world", rule["world_seed"], "=", rule["smallest_detected_s"])
