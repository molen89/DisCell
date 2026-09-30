"""Figure fig:kappa-sweep: the diagnostics across the leakage sweep, per section.

One row per fitted section, one column per readout: held-out reconstruction, NMI of z
with the type, mirror R^2, cycle R^2 of z and of w on the top-decile cycling set, and
I(niche; w) as its excess over the within-type permutation floor. Each panel: the
mean over the three seeds (line) and the range over seeds (band) at every kappa of the
grid; the operating point kappa = 0.1 is marked.

Source: data/datasets/<section>/experiments/kappa_sweep_sweepL.json, runs[*] and
w_channel_guard['k<kappa>_s<seed>'] (the re-read that grades every fit on its own
split), the file tab:kappa-sweep reads. The table's own checks are run first, through
scripts/paper_tables.py (the kappa = 0.1 reads equal the final fits' own reads; the
I(niche; w) re-read agrees with every fit's guard read); a readout that fails them is
not drawn and is named on stdout.

Outputs ../fig_kappa_sweep.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()
OUT = Path(__file__).resolve().parent.parent / "fig_kappa_sweep"
# titles carry the preferred direction (up / down arrow: higher / lower is better);
# the cycle R^2 of w has none, its target is zero (drawn as a line)
READS = [("recon", r"reconstruction $\uparrow$"), ("nmi", r"NMI of $\mathbf{z}$ $\uparrow$"),
         ("mirror_r2", r"mirror $R^2$ $\downarrow$"),
         ("cycle_r2_z_q90", r"cycle $R^2$, $\mathbf{z}$ $\uparrow$"),
         ("cycle_r2_w_q90", r"cycle $R^2$, $\mathbf{w}$ ($\approx 0$)"),
         ("mi", r"$I(\mathrm{niche};\mathbf{w})$ $\uparrow$")]
ROWS = [(PT.OVARIAN, "Ovarian FFPE"), (PT.LUNG, "Lung FFPE"), (PT.FF, "Ovarian FF"),
        (PT.GSE, "Lung TMA core")]

# the table's checks, verbatim: pending cells there are withheld here
tex, tr = PT.table_kappa_sweep("figures/src/fig_kappa_sweep.py")
withheld = {"mi"} if "I(niche; w) excess across" in tex else set()
for n in tr.notes:
    if "mismatch" in n or "the sweep's" in n:
        print("check:", n)
        for k, _ in READS:
            if k in n:
                withheld.add(k)

S.apply()
fig, axes = plt.subplots(len(ROWS), len(READS), figsize=(S.TEXTWIDTH, 3.55), sharex=True)
fig.subplots_adjust(left=0.085, right=0.995, bottom=0.085, top=0.94, wspace=0.52, hspace=0.30)
for i, (ds, name) in enumerate(ROWS):
    d = json.loads((PT.DATA / ds / "experiments" / "kappa_sweep_sweepL.json").read_text())
    grid = d["config"]["values"]
    k_op = d["config"]["kappa"]
    for j, (key, title) in enumerate(READS):
        ax = axes[i, j]
        S.light_grid(ax, "y")
        ax.axvline(k_op, color=S.CELL_EDGE, lw=0.6, zorder=2.5)
        if key in withheld:
            ax.text(0.5, 0.5, "withheld\n(check failed)", transform=ax.transAxes, ha="center",
                    va="center", fontsize=6, color=S.INK2)
            continue
        lo, mid, hi = [], [], []
        for k in grid:
            rows = [r for r in d["runs"] if r["kappa"] == k]
            if key == "mi":
                v = [d["w_channel_guard"][f"k{k:g}_s{r['seed']}"]["w_niche_mi_excess"] for r in rows]
            else:
                v = [r[key] for r in rows]
            assert len(v) == len(PT.SEEDS), (ds, k, key)
            lo.append(min(v)); hi.append(max(v)); mid.append(float(np.mean(v)))
        if key == "cycle_r2_w_q90":
            ax.axhline(0, color=S.MUTED, lw=0.6, ls=(0, (4, 2.5)), zorder=2.4)
        ax.fill_between(grid, lo, hi, color=S.BLUE_BAND, lw=0, zorder=2)
        ax.plot(grid, mid, color=S.BLUE, lw=1.1, zorder=3)
        ax.scatter(grid, mid, s=5, color=S.BLUE, zorder=4, lw=0)
        ax.set_xlim(-0.02, 0.42)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4], ["0", r"\textbf{0.1}", "", "", "0.4"])
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(labelsize=6, pad=1.5)
        if i == 0:
            ax.set_title(title, fontsize=7, color=S.INK, pad=3)
        if i == len(ROWS) - 1:
            ax.set_xlabel(r"$\kappa$", labelpad=0.5)
    axes[i, 0].annotate(name, xy=(-0.58, 0.5), xycoords="axes fraction", rotation=90,
                        ha="center", va="center", fontsize=7, color=S.INK)
S.save(fig, OUT)
print("withheld readouts:", sorted(withheld) or "none")
