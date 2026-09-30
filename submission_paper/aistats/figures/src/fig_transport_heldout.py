"""Figure fig:transport-heldout: the transport read, published against held-out tiles.

One column per fitted section; top row the fraction of the noise ceiling over all
panels, bottom row over the trusted tier. In each panel, every fit is a pair: its
published read (left) joined to its read scored on held-out tiles only (right), each
with its 95% tile-bootstrap interval. DISCELL: one pair per seed (blue); Cellina's
neighbour-rewiring counterfactual (orange), fitted on the training tiles and read on the
panels of the first seed. A read with no value (no trusted panel, or NaN in the file) is
not drawn, and the panel says so.

Source: scripts/logs/transport_heldout_2026-09-29/transport_heldout_comparison.json, the
file tab:transport-heldout reads, through scripts/paper_tables.py (_heldout_records,
which checks the held-out side is scored on held-out tiles).

Outputs ../fig_transport_heldout.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import sys

from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()
OUT = Path(__file__).resolve().parent.parent / "fig_transport_heldout"
COLS = [(PT.OVARIAN, "Ovarian FFPE"), (PT.LUNG, "Lung FFPE"), (PT.FF, "Ovarian FF"),
        (PT.GSE, "Lung TMA core")]
ROWS = [("transport_of_ceiling", "all panels"), ("transport_of_ceiling_trusted", "trusted tier")]

_, recs = PT._heldout_records(PT.Trace("fig:transport-heldout"))


def read(side, key):
    v = side[key]
    ok = PT.finite(v.get("value"))
    return (v["value"], v["ci95"]) if ok else (None, None)


S.apply()
fig, axes = plt.subplots(2, 4, figsize=(S.TEXTWIDTH, 3.35))
fig.subplots_adjust(left=0.085, right=0.995, bottom=0.13, top=0.935, wspace=0.34, hspace=0.32)
notes = []
for r, (key, rlab) in enumerate(ROWS):
    for c, (ds, title) in enumerate(COLS):
        ax = axes[r, c]
        S.light_grid(ax, "y")
        mine = [x for x in recs if x["dataset"] == ds]
        fits = ([x for x in mine if x["method"] == "DisCell"]
                + [x for x in mine if x["method"] != "DisCell"])
        for k, x in enumerate(fits):
            cell = x["method"] != "DisCell"
            col = S.ORANGE if cell else S.BLUE
            mk = "^" if cell else "o"
            off = -0.15 + 0.1 * k                      # dodge the fits within a side
            pts = []
            for j, side in enumerate(("published", "heldout")):
                v, ci = read(x[side], key)
                if v is None:
                    notes.append(f"{title} / {rlab} / {x['method']} {x['run']} / {side}: no value")
                    continue
                xx = j + off
                ax.plot([xx, xx], ci, color=col, lw=0.7, zorder=2)
                ax.scatter([xx], [v], s=14, marker=mk, color=col, lw=0, zorder=3)
                pts.append((xx, v))
            if len(pts) == 2:
                ax.plot(*zip(*pts), color=col, lw=0.6, alpha=0.8, zorder=1)
        if not any(read(x["heldout"], key)[0] is not None for x in fits):
            ax.text(1, 0.5, "no value on\nheld-out tiles", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", fontsize=6, color=S.INK2)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1], ["published", "held-out"] if r == 1 else ["", ""])
        ax.tick_params(labelsize=6.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if r == 0:
            ax.set_title(title, fontsize=7.5, color=S.INK, pad=3)
    # higher is better; the label is turned, so its right arrow reads as an up arrow
    axes[r, 0].set_ylabel(f"fraction of ceiling,\n{rlab}, better " r"$\rightarrow$", labelpad=2, fontsize=7)
h = [plt.Line2D([], [], marker="o", ls="-", lw=0.6, ms=3.5, color=S.BLUE, label="DISCELL, one line per seed"),
     plt.Line2D([], [], marker="^", ls="-", lw=0.6, ms=3.5, color=S.ORANGE, label="Cellina counterfactual")]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7, handlelength=1.8,
           columnspacing=2.0, bbox_to_anchor=(0.54, -0.01))
S.save(fig, OUT)
print("not drawn:", *notes, sep="\n  ")
