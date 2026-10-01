"""Figure fig:probe-data: what the adversary removes from the intrinsic latent, per section.

One panel per block and probe (composition and image; the nonlinear MLP probe, then the
ridge probe as the lighter pair). Per section, a dumbbell from DISCELL without the
adversary (alpha_a = 0, two seeds; open circle) to DISCELL with it (the final fits,
three seeds; filled circle), each with its range over seeds, on the share of the block's
within-type variance the held-out probe recovers beyond the permutation floor
(1 - exp(-2 excess), in %; scripts/paper_tables.py probe_share). Below each dumbbell, the comparison methods on the same
section, one marker each (as in fig:tradeoff).

Source: as tab:probe -- data/datasets/<section>/experiments/probe_regrade_lineage_final.json
and, for a method not yet in it, its own probe record, read through
scripts/paper_tables.py (section_methods, _probe_with_methods). Whole-section fits not
scored yet are absent (listed on stdout).

Outputs ../fig_probe_data.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()
OUT = Path(__file__).resolve().parent.parent / "fig_probe_data"
PANELS = [("mlp_comp", "Composition, MLP probe"), ("mlp_img", "Image, MLP probe"),
          ("ridge_comp", "Composition, ridge probe"), ("ridge_img", "Image, ridge probe")]
ROWS = [(PT.OVARIAN, "Ovarian FFPE"), (PT.LUNG, "Lung FFPE"), (PT.FF, "Ovarian FF"),
        (PT.GSE, "Lung TMA core"), (PT.DUAL, "TMA serial")]
FINALS = [f"DisCell/finalL_s{s}" for s in PT.SEEDS]
UNCS = ["DisCell/uncontrolledL_s0", "DisCell/uncontrolledL_s1"]
BASE_ORDER = ["resolVI", "SIMVI", "MintFlow", "Cellina", "Cellina, niche domain", "Cellina, own graph"]

tr = PT.Trace("fig:probe-data")
data = {}
for ds, _ in ROWS:
    bat = json.loads((PT.DATA / ds / "experiments" / "baseline_battery_lineage.json").read_text())
    methods = PT.section_methods(tr, ds, bat)
    _, prb, _ = PT._probe_with_methods(tr, ds, [m["probe"] for m in methods if m.get("probe")])
    missing = [f"{m['label']} ({m.get('state')})" for m in methods
               if not m.get("probe") or m["probe"] not in prb]
    if missing:
        print(f"{ds}: not drawn (not scored yet): {', '.join(missing)}")
    data[ds] = (prb, [m for m in methods if m.get("probe") in prb])

S.apply()
fig = plt.figure(figsize=(S.TEXTWIDTH, 2.35))
gs = fig.add_gridspec(1, 5, left=0.125, right=0.995, bottom=0.30, top=0.90,
                      width_ratios=[1, 1, 0.12, 1, 1], wspace=0.10)
axes = [fig.add_subplot(gs[0, c]) for c in (0, 1, 3, 4)]
YD, YB = 0.16, -0.18                    # DISCELL above, comparison methods below the row
for c, (ax, (block, title)) in enumerate(zip(axes, PANELS)):
    ridge = block.startswith("ridge")
    for i, (ds, _) in enumerate(ROWS):
        y = -i
        prb, methods = data[ds]
        vu = [PT.probe_share(prb[e][block]["excess"]) for e in UNCS]
        vf = [PT.probe_share(prb[e][block]["excess"]) for e in FINALS]
        mu, mf = float(np.mean(vu)), float(np.mean(vf))
        # without -> with the adversary: the arrow points from the open to the filled circle
        ax.annotate("", xy=(mf, y + YD), xytext=(mu, y + YD), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=S.BLUE_RAMP[1], lw=1.6, mutation_scale=7,
                                    shrinkA=3.2, shrinkB=3.2))
        if c == 0 and i == 2:                   # say it once, above the longest arrow
            ax.text(mu, y + YD + 0.13, "adding the adversary", ha="right",
                    va="bottom", fontsize=5.6, color=S.INK2, style="italic")
        for v, m in ((vu, mu), (vf, mf)):
            ax.plot([min(v), max(v)], [y + YD] * 2, color=S.BLUE, lw=0.8, zorder=3)
            ax.plot([min(v)] * 2 + [np.nan] + [max(v)] * 2,
                    [y + YD - 0.07, y + YD + 0.07, np.nan, y + YD - 0.07, y + YD + 0.07],
                    color=S.BLUE, lw=0.6, zorder=3)
        S.method_point(ax, mu, y + YD, "DISCELL, no adversary", scale=0.5)
        S.method_point(ax, mf, y + YD, "DISCELL", scale=0.5)
        for m in methods:
            S.method_point(ax, PT.probe_share(prb[m["probe"]][block]["excess"]), y + YB, m["label"], scale=0.45)
        if i < len(ROWS) - 1:
            ax.axhline(y - 0.5, color=S.GHOST, lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 60)
    ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.3", "1", "3", "10", "30"])
    ax.grid(axis="x", color=S.GHOST, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-len(ROWS) + 0.5, 0.5)
    ax.set_yticks([-i for i in range(len(ROWS))])
    ax.set_yticklabels([n for _, n in ROWS] if c == 0 else [])
    ax.tick_params(axis="y", length=0, labelsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.spines["left"].set_visible(False)
    ax.set_title(title + S.LEFT, fontsize=7.5, color=S.INK2 if ridge else S.INK, pad=3)
    if ridge:                                   # the secondary probe, visually lighter
        ax.set_facecolor("#fafaf8")
        for s in ("bottom",):
            ax.spines[s].set_color(S.CELL_EDGE)
fig.text(0.56, 0.185, r"share of the block's within-type variance recovered by the probe (\%, log scale)",
         ha="center", fontsize=7.5, color=S.INK)
names = ["DISCELL, no adversary", "DISCELL"] + [n for n in BASE_ORDER
                                                if any(m["label"] == n for _, (_, ms) in data.items() for m in ms)]
handles = S.method_handles(names, scale=0.8)
handles[0].set_label(r"DISCELL, no adversary ($\alpha_a = 0$)")
handles[1].set_label(r"DISCELL (with the adversary; arrow: without $\rightarrow$ with)")
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5,
           handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.56, -0.015))
S.save(fig, OUT)
