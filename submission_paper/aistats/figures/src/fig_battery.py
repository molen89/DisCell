"""Figure fig:battery: every method on the battery's read-outs, per section.

One panel per read-out: NMI of the intrinsic latent with the type; the MLP probe's
composition excess as a fraction of DISCELL's without the adversary (the "fraction
left"); mirror R^2; cycle R^2 of the intrinsic latent on the top-decile cycling set;
held-out reconstruction. Sections are rows; within a row each method has its own line,
with the markers of fig:tradeoff. DISCELL: mean over three seeds and the range.
MintFlow's reconstruction is not shown (under inspection), SIMVI has no count decoder.

Source: as tab:battery -- data/datasets/<section>/experiments/baseline_battery_lineage.json
and the probe files, read through scripts/paper_tables.py (section_methods,
_probe_with_methods); the table's builder runs first, so its checks apply. Whole-section
fits not scored yet are absent (listed on stdout).

Outputs ../fig_battery.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()
OUT = Path(__file__).resolve().parent.parent / "fig_battery"
# titles carry the preferred direction: higher (up arrow) or lower (down arrow) is better
READS = [("nmi", "battery", r"NMI of $\mathbf{z}$ $\uparrow$"),
         ("mlp_comp", "probe", r"composition, fraction left $\downarrow$"),
         ("mirror.r2", "battery", r"mirror $R^2$ $\downarrow$"),
         ("cycle_q90.z.r2_pooled", "battery", r"cycle $R^2$ of $\mathbf{z}$ $\uparrow$"),
         ("reconstruction.recon", "battery", r"reconstruction $\uparrow$")]
ROWS = [(PT.OVARIAN, "Ovarian FFPE"), (PT.LUNG, "Lung FFPE"), (PT.FF, "Ovarian FF"),
        (PT.GSE, "Lung TMA core"), (PT.DUAL, "TMA serial")]
ORDER = ["DISCELL", "resolVI", "SIMVI", "MintFlow", "Cellina", "Cellina, niche domain", "Cellina, own graph"]
FINALS = [f"DisCell/finalL_s{s}" for s in PT.SEEDS]

tex, _ = PT.table_battery("figures/src/fig_battery.py")      # its checks; pending cells below
if "\\pending{check}" in tex:
    print("WARNING: tab:battery has cells that failed a check; they are drawn here from the same entries")

tr = PT.Trace("fig:battery")
sec = {}
for ds, name in ROWS:
    bat = json.loads((PT.DATA / ds / "experiments" / "baseline_battery_lineage.json").read_text())
    methods = PT.section_methods(tr, ds, bat)
    _, prb, _ = PT._probe_with_methods(tr, ds, [m["probe"] for m in methods if m.get("probe")])
    miss = [f"{m['label']} ({m.get('state')})" for m in methods if "battery" not in m]
    if miss:
        print(f"{name}: not drawn (not scored yet): {', '.join(miss)}")
    sec[ds] = (bat, prb, [m for m in methods if "battery" in m])


def value(ds, entry_b, entry_p, key, src):
    bat, prb, _ = sec[ds]
    if src == "probe":
        return prb[entry_p][key]["fraction_of_uncontrolled"]
    return PT._dig(bat[entry_b], key)


S.apply()
n = len(ORDER)
fig, axes = plt.subplots(1, len(READS), figsize=(S.TEXTWIDTH, 3.0), sharey=True)
fig.subplots_adjust(left=0.125, right=0.99, bottom=0.2, top=0.93, wspace=0.14)
for ax, (key, src, title) in zip(axes, READS):
    for i, (ds, _) in enumerate(ROWS):
        base = -i * (n + 1.2)
        if i:
            ax.axhline(base + 1.1, color="#e4e3dd", lw=0.5, zorder=0)
        # DISCELL: mean and range over seeds
        v = [value(ds, e, e, key, src) for e in FINALS]
        y = base
        ax.plot([min(v), max(v)], [y, y], color=S.BLUE, lw=1.0, zorder=3)
        S.method_point(ax, float(np.mean(v)), y, "DISCELL", scale=0.45)
        for m in sec[ds][2]:
            y = base - ORDER.index(m["label"])
            if m["label"] == "MintFlow" and key == "reconstruction.recon":
                continue                                    # not reported (under inspection)
            x = value(ds, m["battery"], m.get("probe"), key, src)
            if x is None:
                continue                                    # no decoder
            S.method_point(ax, x, y, m["label"], scale=0.4)
    ax.set_title(title, fontsize=7, color=S.INK, pad=3)
    ax.grid(axis="x", color=S.GHOST, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
axes[-1].xaxis.set_major_locator(plt.MaxNLocator(3))
axes[1].axvline(0, color=S.MUTED, lw=0.5, zorder=1)
axes[1].axvline(1, color=S.MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
ax0 = axes[0]
ax0.set_yticks([-i * (n + 1.2) - (n - 1) / 2 for i in range(len(ROWS))])
ax0.set_yticklabels([nm for _, nm in ROWS], fontsize=7)
ax0.set_ylim(-(len(ROWS) - 1) * (n + 1.2) - n + 0.3, 0.8)
present = [nm for nm in ORDER if nm == "DISCELL" or any(m["label"] == nm for s in sec.values() for m in s[2])]
h = S.method_handles(present, scale=0.75)
h[0].set_label("DISCELL (bar: range over seeds)")
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=6.5, handletextpad=0.3,
           columnspacing=1.2, bbox_to_anchor=(0.56, -0.015))
S.save(fig, OUT)
