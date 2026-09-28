"""Figure: what the data say about kappa, on the simplex of three genes.

For a cell with composition p_i and neighbour influx rho_bar_i (held fixed, as in
the derivations), a leak fraction kappa implies the clean composition
rho_i(kappa) = (p_i - kappa rho_bar_i) / (1 - kappa): a line that starts at p_i
(kappa = 0) and runs away from rho_bar_i. It is a valid composition up to
kappa_bar_i = min_g p_ig / rho_bar_ig, where some gene's share reaches zero, and
the true kappa lies in [0, kappa_bar_i]. (a) A cell that expresses a little of
the neighbours' marker (gene 3) itself: kappa is bounded above, not below.
(b) A cell that cannot express gene 3: the line leaves the simplex exactly at the
true composition, so kappa_bar_i equals the true kappa.

Schematic: the compositions are chosen for legibility, not measured; the true
kappa is 0.3 in both panels. Outputs ../fig_kappa_bound.pdf (+ .png). Run from
the repository root.
"""
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "fig_kappa_bound"
KAPPA = 0.3                                                # schematic, chosen for legibility
RHOBAR = np.array([0.20, 0.15, 0.65])                     # neighbours: rich in gene 3
CELLS = [(np.array([0.55, 0.35, 0.10]), r"bounded above, not below"),
         (np.array([0.60, 0.40, 0.00]), r"a gene the cell cannot express")]
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def xy(bary):
    return np.asarray(bary) @ CORNERS


def clean(p, k):
    return (p - k * RHOBAR) / (1 - k)


def callout(ax, pt, text, y):
    """Label in a column right of the simplex, with a leader line to the point."""
    ax.annotate(text, xy=pt, xytext=(1.08, y), fontsize=7, color=S.INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.45, shrinkA=1.5, shrinkB=3))


S.apply()
fig = plt.figure(figsize=(S.TEXTWIDTH, 2.6))
for c, (rho, title) in enumerate(CELLS):
    ax = fig.add_axes([0.01 + 0.5 * c, 0.02, 0.48, 0.90])
    p = (1 - KAPPA) * rho + KAPPA * RHOBAR
    kbar = float(np.min(p / RHOBAR))
    ax.add_patch(Polygon(CORNERS, closed=True, fc=S.GHOST, ec=S.INK2, lw=0.8, zorder=1))
    ax.text(-0.02, -0.05, r"gene 1", fontsize=7, color=S.INK, ha="center", va="top")
    ax.text(1.035, 0.0, r"gene 2", fontsize=7, color=S.INK, ha="left", va="center")
    ax.text(0.54, CORNERS[2, 1] - 0.005, r"gene 3: the neighbours' marker", fontsize=7, color=S.INK,
            ha="left", va="center")
    ax.plot(*np.stack([xy(rho), xy(RHOBAR)]).T, color=S.MUTED, lw=0.7, ls=(0, (1.2, 1.6)), zorder=2)
    path = np.array([xy(clean(p, k)) for k in np.linspace(0, kbar, 60)])
    ax.plot(*path.T, color=S.INK, lw=1.6, zorder=3, solid_capstyle="round")
    beyond = np.array([xy(clean(p, k)) for k in np.linspace(kbar, kbar + 0.12, 10)])
    ax.plot(*beyond.T, color=S.INK2, lw=1.0, ls=(0, (2.5, 1.5)), zorder=3)
    k_arrow = 0.5 * KAPPA if c == 0 else 0.45 * kbar
    ax.annotate("", xy=xy(clean(p, k_arrow + 0.04)), xytext=xy(clean(p, k_arrow)),
                arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=0, mutation_scale=10), zorder=4)
    end = xy(clean(p, kbar))
    ax.scatter(*xy(RHOBAR), s=26, color=S.BLUE, zorder=5, lw=0)
    ax.scatter(*xy(p), s=22, color=S.INK, zorder=5, lw=0)
    ax.scatter(*end, s=26, marker="D", facecolor="white", edgecolor=S.BLUE, lw=1.0, zorder=7)
    callout(ax, xy(RHOBAR), r"$\bar{\bm\rho}_i$: the neighbours' influx", xy(RHOBAR)[1])
    callout(ax, xy(p), r"$\mathbf{p}_i$: the cell as observed ($\kappa = 0$)", xy(p)[1] + 0.03)
    if c == 0:
        ax.scatter(*xy(rho), s=26, facecolor="white", edgecolor=S.INK, lw=1.0, zorder=6)
        callout(ax, xy(rho), rf"$\bm\rho_i$ at the true $\kappa$ ($={KAPPA}$)", 0.11)
        callout(ax, end, rf"$\bar\kappa_i = {kbar:.2f}$: gene 3 reaches zero", -0.17)
        note = r"every $\kappa \in [0, \bar\kappa_i]$ fits;" + "\n" + r"the true $\kappa$ lies inside"
    else:
        callout(ax, end, rf"$\bar\kappa_i = {kbar:.2f}$, the true $\kappa$: $\bm\rho_i$" + "\n" + r"lies on the edge", -0.19)
        note = r"only $\kappa = \bar\kappa_i$ keeps gene 3" + "\n" + r"at zero: the data pin $\kappa$"
    ax.text(1.08, -0.37, note, fontsize=7, color=S.INK2, ha="left", va="center")
    ax.set_xlim(-0.08, 2.02); ax.set_ylim(-0.47, 0.98); ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.03, 1.0, title, transform=ax.transAxes, fontsize=7.5, color=S.INK, ha="left", va="bottom")
    S.panel_label(ax, "ab"[c], x=0.02, y=1.0)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=220)
print("wrote", OUT.with_suffix(".pdf"))
