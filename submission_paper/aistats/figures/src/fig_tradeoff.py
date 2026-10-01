"""Figure fig:tradeoff: composition leakage against retained cycling state, per section.

One panel per section (the four fitted sections and the TMA core's held-out serial section).
x: composition leakage of the intrinsic latent -- the held-out MLP probe's share of the
within-type variance of neighbour composition it explains beyond the permutation floor, in %
(1 - exp(-2 excess), as in tab:probe). y: cycle R^2 of the intrinsic latent on the label-free
top-decile cycling set (as in tab:battery). A method can reach low leakage by keeping little
within-type state; the y axis shows whether it did.

Points: DISCELL (mean over finalL_s0-s2, bars = range over seeds on both axes); DISCELL
without the adversary (uncontrolledL_s0-s1, mean and range; its cycle read is the trainer's
masked battery read in runs/uncontrolledL_s*/degeneracy.json, which equals the baseline
battery's read to 1e-4 on every finalL fit; not read on the serial section); resolVI;
Cellina; Cellina with our niche label as its domain; Cellina on its own graph (TMA core and
serial); SIMVI and MintFlow where scored. Every number is read from the files that
scripts/paper_tables.py reads for tab:probe and tab:battery, through that module, so the
figure and the tables cannot disagree. Whole-section fits still queued or running are
absent (listed on stdout).

Outputs ../fig_tradeoff.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = HERE.parent / "fig_tradeoff"
spec = importlib.util.spec_from_file_location("paper_tables", REPO / "scripts" / "paper_tables.py")
PT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PT)

PANELS = [(PT.OVARIAN, "Ovarian cancer (FFPE)"), (PT.LUNG, "Lung cancer (FFPE)"),
          (PT.FF, "Ovarian cancer (FF)"), (PT.GSE, "Lung TMA core"),
          (PT.DUAL, "Lung TMA, serial section")]
# identity by marker shape (and a legend); colour groups: DISCELL blue, the Cellina
# variants orange (validated slots 1-2 of style.py), the other methods grey
GREY = S.INK2
STYLE = {
    "DISCELL": dict(marker="o", color=S.BLUE, fill=True, size=34),
    "DISCELL, no adversary": dict(marker="o", color=S.BLUE, fill=False, size=30),
    "resolVI": dict(marker="s", color=GREY, fill=True, size=24),
    "SIMVI": dict(marker="v", color=GREY, fill=True, size=28),
    "MintFlow": dict(marker="D", color=GREY, fill=False, size=22),
    "Cellina": dict(marker="^", color=S.ORANGE, fill=True, size=30),
    "Cellina, niche domain": dict(marker="^", color=S.ORANGE, fill=False, size=30),
    "Cellina, own graph": dict(marker="P", color=S.ORANGE, fill=True, size=30),
}
FINALS = [f"DisCell/finalL_s{s}" for s in PT.SEEDS]
LABEL_AT = {PT.GSE: ((-6, -9), "right")}      # direct-label offset where the default collides
UNCS = ["DisCell/uncontrolledL_s0", "DisCell/uncontrolledL_s1"]


def section_points(ds):
    tr = PT.Trace("fig:tradeoff")
    bat = json.loads((PT.DATA / ds / "experiments" / "baseline_battery_lineage.json").read_text())
    methods = PT.section_methods(tr, ds, bat)
    _, prb, _ = PT._probe_with_methods(tr, ds, [m["probe"] for m in methods if m.get("probe")])
    leak = lambda e: PT.probe_share(prb[e]["mlp_comp"]["excess"])       # noqa: E731
    cyc = lambda e: bat[e]["cycle_q90"]["z"]["r2_pooled"]                # noqa: E731
    pts = {"DISCELL": ([leak(e) for e in FINALS], [cyc(e) for e in FINALS])}
    if ds != PT.DUAL:
        yu = []
        for s in (0, 1):
            deg = json.loads((PT.DATA / ds / "runs" / f"uncontrolledL_s{s}" / "degeneracy.json").read_text())
            assert PT.MASK in deg["battery"]["eval_mask"]["excluded_types"]
            yu.append(deg["battery"]["cycle_r2_z_q90"])
        pts["DISCELL, no adversary"] = ([leak(e) for e in UNCS], yu)
    missing = []
    for m in methods:
        if "battery" not in m:
            missing.append(f"{m['label']} ({m['state']})")
            continue
        assert PT.MASK in bat[m["battery"]]["eval_mask"]["excluded_types"]
        pts[m["label"]] = ([leak(m["probe"])], [cyc(m["battery"])])
    return pts, missing


S.apply()
fig, axes = plt.subplots(2, 3, figsize=(S.TEXTWIDTH, 4.3))
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.1, top=0.95, wspace=0.28, hspace=0.42)
for ax, (ds, title) in zip(axes.flat, PANELS):
    pts, missing = section_points(ds)
    if missing:
        print(f"{title}: not drawn (not scored yet): {', '.join(missing)}")
    xmax = max(max(x) for x, _ in pts.values())
    for name, (xs, ys) in pts.items():
        st = STYLE[name]
        x, y = float(np.mean(xs)), float(np.mean(ys))
        if len(xs) > 1:
            ax.errorbar(x, y, xerr=[[x - min(xs)], [max(xs) - x]], yerr=[[y - min(ys)], [max(ys) - y]],
                        fmt="none", ecolor=st["color"], elinewidth=0.8, capsize=1.5, zorder=3)
        ax.scatter([x], [y], s=st["size"], marker=st["marker"], linewidths=1.0, zorder=4,
                   facecolor=st["color"] if st["fill"] else "white", edgecolor=st["color"])
    dx, dy = pts["DISCELL"]
    off, ha = LABEL_AT.get(ds, ((6, -9), "left"))
    ax.annotate(r"DISCELL", (np.mean(dx), np.mean(dy)), xytext=off,
                textcoords="offset points", fontsize=6.5, color=S.BLUE, ha=ha, va="top")
    ax.set_xlim(0, 1.12 * xmax)
    ax.set_ylim(-0.03, 0.85)          # a read near 0 stays visible
    ax.set_title(title, fontsize=7.5, color=S.INK, pad=3)
    ax.grid(axis="both", color=S.GHOST, lw=0.5, zorder=0)
    ax.tick_params(labelsize=6.5)
# the preferred direction on each axis, next to it: upright above the y axis, under the
# right end of the x axis (the headers name sections); the labels name the quantities
YLAB = r"cycle $R^2$ of $\mathbf{z}$"
XLAB = r"composition leakage of $\mathbf{z}$ (\%)"
for ax in axes[:, 0]:
    ax.set_ylabel(YLAB, labelpad=2)
    S.ybetter(ax, up=True)
for ax in list(axes[1, :2]) + [axes[0, 2]]:
    ax.set_xlabel(XLAB, labelpad=9)
    S.xbetter(ax, right=False)
# the desirable corner, once, where the first panel leaves it empty
axes[0, 0].annotate(r"low leakage, keeps cycle state", xy=(0.012, 0.985), xycoords="axes fraction",
                    xytext=(0.13, 0.88), textcoords="axes fraction", fontsize=6.2, color=S.INK2,
                    ha="left", va="center", style="italic",
                    arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=0.6, shrinkA=1, shrinkB=0,
                                    mutation_scale=6))

leg = axes[1, 2]
leg.axis("off")
handles = [Line2D([], [], ls="none", marker=st["marker"], markersize=np.sqrt(st["size"]) * 0.95,
                  markerfacecolor=st["color"] if st["fill"] else "white", markeredgecolor=st["color"],
                  markeredgewidth=1.0, label=name)
           for name, st in STYLE.items()]
leg.legend(handles=handles, loc="center left", frameon=False, fontsize=7, handletextpad=0.5,
           labelspacing=0.75, borderaxespad=0.0)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=250)
print("wrote", OUT.with_suffix(".pdf"))
