"""Figure fig:separation: composition in the context latent against composition left in
the intrinsic latent, per method and section.

One panel per section (the four fitted sections and the TMA core's serial section).
x: neighbour composition recovered from the method's context latent by the held-out MLP
probe beyond its permutation floor, as in tab:context (1 - exp(-2 excess), in %): higher
means the context channel carries the niche. y: composition left in the intrinsic latent,
the same probe on the intrinsic latent, as in tab:probe (1 - exp(-2 excess), in %): lower
means the intrinsic latent is cleaner. A method that separates the two sits bottom right.
resolVI has no context latent and is not drawn. DISCELL: mean over finalL_s0-s2 with the
range over seeds on both axes.

Source: data/datasets/<section>/experiments/context_grade.json (x) and the probe files of
tab:probe (y), read through scripts/paper_tables.py (table_context runs first, so its checks
apply; section_methods and _probe_with_methods pick the entries, as in fig:tradeoff).
Whole-section fits not scored yet are absent (listed on stdout).

Outputs ../fig_separation.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()
OUT = Path(__file__).resolve().parent.parent / "fig_separation"
PANELS = [(PT.OVARIAN, "Ovarian cancer (FFPE)"), (PT.LUNG, "Lung cancer (FFPE)"),
          (PT.FF, "Ovarian cancer (FF)"), (PT.GSE, "Lung TMA core"),
          (PT.DUAL, "Lung TMA, serial section")]
FINALS = [f"DisCell/finalL_s{s}" for s in PT.SEEDS]
SHORT = {"DISCELL": "DISCELL", "SIMVI": "SIMVI", "MintFlow": "MintFlow", "Cellina": "Cellina",
         "Cellina, niche domain": "Cellina, niche", "Cellina, own graph": "Cellina, own graph"}
#: direct labels where the default (right of the point) collides: ("off", (dx, dy) points,
#: ha) or, in the crowded TMA panels, ("at", (x, y) data, ha) with a thin leader line
LABEL_AT = {
    (PT.OVARIAN, "Cellina"): ("off", (-5, 3), "right"),
    (PT.LUNG, "Cellina"): ("off", (-5, 3), "right"),
    (PT.LUNG, "MintFlow"): ("off", (-3, 6), "right"),
    (PT.OVARIAN, "MintFlow"): ("off", (-3, 6), "right"),
    (PT.FF, "Cellina"): ("off", (-5, 3), "right"),
    (PT.GSE, "Cellina, niche domain"): ("at", (32, 6.4), "left"),
    (PT.GSE, "SIMVI"): ("at", (32, 5.5), "left"),
    (PT.GSE, "Cellina, own graph"): ("at", (32, 4.6), "left"),
    (PT.GSE, "DISCELL"): ("at", (32, 3.7), "left"),
    (PT.GSE, "Cellina"): ("at", (32, 2.8), "left"),
    (PT.DUAL, "Cellina, own graph"): ("at", (32, 9.4), "left"),
    (PT.DUAL, "Cellina"): ("at", (32, 8.6), "left"),
    (PT.DUAL, "Cellina, niche domain"): ("at", (32, 7.8), "left"),
    (PT.DUAL, "SIMVI"): ("at", (32, 7.0), "left"),
}
DEFAULT_AT = ("off", (5, 2), "left")

PT.table_context("figures/src/fig_separation.py")       # its checks


def section_points(ds):
    tr = PT.Trace("fig:separation")
    bat = json.loads((PT.DATA / ds / "experiments" / "baseline_battery_lineage.json").read_text())
    methods = PT.section_methods(tr, ds, bat)
    _, prb, _ = PT._probe_with_methods(tr, ds, [m["probe"] for m in methods if m.get("probe")])
    ctx = json.loads((PT.DATA / ds / "experiments" / "context_grade.json").read_text())["methods"]
    leak = lambda e: PT.probe_share(prb[e]["mlp_comp"]["excess"])              # noqa: E731
    share = lambda rec: PT.probe_share(rec["probe"]["mlp"]["comp"]["excess"])  # noqa: E731
    runs = ctx["DISCELL"]["runs"]
    pts = {"DISCELL": ([share(runs[f"finalL_s{s}"]) for s in PT.SEEDS], [leak(e) for e in FINALS])}
    missing = []
    for m in methods:
        e = ctx.get(m["label"])
        if m["label"] == "resolVI":
            continue
        if e is None or e.get("absent") or m.get("probe") not in prb:
            missing.append(f"{m['label']} ({m.get('state', 'not graded')})")
            continue
        pts[m["label"]] = ([share(e)], [leak(m["probe"])])
    return pts, missing


S.apply()
fig, axes = plt.subplots(2, 3, figsize=(S.TEXTWIDTH, 4.3))
fig.subplots_adjust(left=0.095, right=0.975, bottom=0.1, top=0.95, wspace=0.22, hspace=0.36)
for ax, (ds, title) in zip(axes.flat, PANELS):
    pts, missing = section_points(ds)
    if missing:
        print(f"{title}: not drawn: {', '.join(missing)}")
    for name, (xs, ys) in pts.items():
        st = S.METHOD_STYLE[name]
        x, y = float(np.mean(xs)), float(np.mean(ys))
        if len(xs) > 1:
            ax.errorbar(x, y, xerr=[[x - min(xs)], [max(xs) - x]], yerr=[[y - min(ys)], [max(ys) - y]],
                        fmt="none", ecolor=st["color"], elinewidth=0.8, capsize=1.5, zorder=3)
        S.method_point(ax, x, y, name)
        kind, where, ha = LABEL_AT.get((ds, name), DEFAULT_AT)
        colour = st["color"] if name == "DISCELL" else S.INK2
        if kind == "off":
            ax.annotate(SHORT[name], (x, y), xytext=where, textcoords="offset points", fontsize=6,
                        color=colour, ha=ha, va="bottom")
        else:
            ax.annotate(SHORT[name], (x, y), xytext=where, textcoords="data", fontsize=6,
                        color=colour, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.4, shrinkA=1, shrinkB=3))
    ymax = max(max(y) for _, y in pts.values())
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.2 * ymax)
    ax.set_title(title, fontsize=7.5, color=S.INK, pad=3)
    S.light_grid(ax)
    ax.tick_params(labelsize=6.5)
fig.supxlabel(r"neighbour composition recovered from the context latent (\%)",
              fontsize=8, y=0.01)
fig.supylabel(r"neighbour composition left in the intrinsic latent $\mathbf{z}$ (\%)",
              fontsize=8, x=0.005)
for ax in axes[:, 0]:
    S.ybetter(ax, up=False)       # lower is better on the vertical axis
for ax in list(axes[1, :2]) + [axes[0, 2]]:
    S.xbetter(ax, right=True)     # higher is better on the horizontal axis
axes[1, 2].axis("off")
axes[1, 2].text(0.05, 0.5, "resolVI has no context latent.\nSIMVI and MintFlow are shown\nwhere their whole-section\nfits exist.",
                transform=axes[1, 2].transAxes, fontsize=7, color=S.INK2, va="center", ha="left")
S.save(fig, OUT)
