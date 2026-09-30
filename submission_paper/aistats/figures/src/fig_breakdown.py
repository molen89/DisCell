"""Figure: the breakdown point of Definition 1, on schematic readouts.

Four illustrative readouts over the kappa grid, each with an interval at every
grid point and a null value r0: (a) a signed contrast that survives the grid;
(b) a contrast that shrinks with kappa, as readouts do by construction, but
stays clear of zero; (c) a contrast that breaks down inside the grid; (d) one that
breaks down at the first grid point above zero. The curves are fixed synthetic
functions, not fitted values; only the grid is the model's. Filled markers:
interval clear of the null with the sign it has at kappa = 0; hollow: from the
breakdown point on. Outputs ../fig_breakdown.pdf (+ .png). Run from the
repository root.
"""
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4])          # the model's kappa grid
OUT = Path(__file__).resolve().parent.parent / "fig_breakdown"

# (title, mean(kappa), interval half-width, null value)
READOUTS = [
    (r"real at every leak fraction", lambda k: 0.62 - 0.35 * k, 0.12, 0.0),
    (r"shrinks, but stays real", lambda k: 0.72 - 1.35 * k, 0.08, 0.0),
    (r"explained away at $\kappa = 0.2$", lambda k: 0.50 - 1.9 * k, 0.20, 0.0),
    (r"explained away at $\kappa = 0.05$", lambda k: 0.30 - 1.5 * k, 0.24, 0.0),
]


def breakdown(m, h, r0):
    """Index of the first grid point whose interval contains r0 or whose sign
    (relative to r0) differs from that at kappa = 0; None if there is none."""
    s0 = np.sign(m[0] - r0)
    for i, (mi, hi) in enumerate(zip(m, h)):
        if (mi - hi <= r0 <= mi + hi) or np.sign(mi - r0) != s0:
            return i
    return None


S.apply()
fig = plt.figure(figsize=(S.TEXTWIDTH, 2.05))
gs = fig.add_gridspec(1, 4, left=0.03, right=0.99, bottom=0.30, top=0.86, wspace=0.16)
for c, (title, f, hw, r0) in enumerate(READOUTS):
    ax = fig.add_subplot(gs[0, c])
    m, h = f(GRID), np.full(GRID.size, hw)
    b = breakdown(m, h, r0)
    ok = np.arange(GRID.size) < (GRID.size if b is None else b)
    ax.axhline(r0, color=S.MUTED, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    ax.vlines(GRID, m - h, m + h, color=S.INK2, lw=1.0, zorder=2)
    ax.scatter(GRID[ok], m[ok], s=16, color=S.INK, zorder=4, lw=0)
    ax.scatter(GRID[~ok], m[~ok], s=16, facecolor="white", edgecolor=S.INK2, lw=0.9, zorder=4)
    ax.set_xlim(-0.025, 0.43); ax.set_ylim(-0.55, 1.2)
    ax.set_xticks(GRID, ["0", "", "0.1", "0.2", "0.3", "0.4"])
    ax.set_yticks([])
    ax.set_xlabel(r"$\kappa$", labelpad=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if b is None:                              # right-censored: survives the whole grid
        ax.annotate("", xy=(0.425, 1.06), xytext=(0.30, 1.06),
                    arrowprops=dict(arrowstyle="-|>", color=S.BLUE, lw=1.0, mutation_scale=7))
        ax.text(0.29, 1.06, r"$\kappa^\star > 0.4$", fontsize=7, color=S.INK, ha="right", va="center")
    else:
        ax.axvline(GRID[b], color=S.BLUE, lw=1.0, ls=(0, (1.2, 1.4)), zorder=3)
        ax.text(GRID[b] + 0.012, 1.06, rf"$\kappa^\star = {GRID[b]:g}$", fontsize=7, color=S.INK,
                ha="left", va="center")
    if r0 != 0:
        ax.text(0.425, r0 - 0.06, r"permutation null", fontsize=6.5, color=S.INK2, ha="right", va="top")
    ax.set_title(title, fontsize=7.5, color=S.INK, loc="left", pad=4)
    S.panel_label(ax, "abcd"[c], x=-0.03, y=1.035)
fig.axes[0].set_ylabel(r"readout", labelpad=2)
# how to read one point, on panel (a)
ax0 = fig.axes[0]
ax0.annotate(r"one refit at $\kappa = 0.1$;" + "\n" + r"bar: its interval", xy=(0.1, 0.62 - 0.035 - 0.125), xytext=(0.13, 0.24),
             fontsize=6.5, color=S.INK2, ha="left", va="center", linespacing=1.1,
             arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.45, shrinkA=1, shrinkB=1.5))
ax0.text(0.425, -0.06, r"no effect", fontsize=6.5, color=S.INK2, ha="right", va="top")

# key, in ink beside its samples
keys = [Line2D([], [], marker="o", ls="", ms=4, color=S.INK, label=r"interval clear of the null, sign kept"),
        Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=S.INK2, label=r"from the breakdown point on"),
        Line2D([], [], color=S.MUTED, lw=0.8, ls=(0, (4, 2.5)), label=r"null value $r_0$"),
        Line2D([], [], color=S.BLUE, lw=1.0, ls=(0, (1.2, 1.4)), label=r"breakdown point $\kappa^\star$")]
fig.legend(handles=keys, loc="lower center", ncol=4, frameon=False, fontsize=7, handlelength=1.8,
           columnspacing=1.6, bbox_to_anchor=(0.5, -0.01))

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=220)
print("wrote", OUT.with_suffix(".pdf"), "| breakdown indices",
      [breakdown(f(GRID), np.full(GRID.size, hw), r0) for _, f, hw, r0 in READOUTS])
