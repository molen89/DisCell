"""Figure: three neighbour graphs on one field of the primary section.

(a) Exact contact: model edges whose segmented polygons touch. (b) Contact
within 1 um: model edges along which the two polygons run within 1 um of each
other, the contact kernel of tab:contact. (c) The model's graph: Delaunay
adjacency pruned at 40 um. Cells with no edge in a panel's graph are filled dark.

All three are subsets of the same edge set, so they differ only in which edges
are kept; the definitions are those of tab:contact and fig_contact_kernel. The
printed shares are over the whole section, among cells with at least one model
edge (the table's denominator). The field is the 160 um window, on a 20 um grid,
that is most representative of the section: among windows with at least 150
connected cells and at least 25 cells from each of the section's sparsest and
densest fifths (density = centroids within 20 um), the one whose no-neighbour
shares under the two contact graphs are closest to the section's (sum of absolute
differences). Outputs ../fig_graph_compare.pdf (+ .png) and
the plotted numbers in data/fig_graph_compare.json. Run from the repository root.
"""
from pathlib import Path
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import shapely
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

DS, NAME = "xenium_prime_ovarian_cancer_ffpe", "full"
BUNDLE = Path(f"data/datasets/{DS}/bundle")
PRUNE_UM, R_DENS, WIN_UM, STEP_UM, MIN_CELLS, MIN_FIFTH = 40.0, 20.0, 160.0, 20.0, 150, 25
ISOLATED = "#7a7873"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fig_graph_compare"

S.apply()
a = ad.read_h5ad(BUNDLE / f"{NAME}.h5ad", backed="r")
n = a.n_obs
xy = np.asarray(a.obsm["spatial_um"], dtype=np.float64)
mpp = float(a.uns["microns_per_pixel"])
cent = np.asarray(a.obsm["spatial"], dtype=np.float64)
polys = np.asarray(shapely.from_wkb(pd.read_parquet(BUNDLE / f"{NAME}_polygons.parquet")["wkb"].to_numpy()), dtype=object)
e = pd.read_parquet(BUNDLE / f"{NAME}_edges_voronoi.parquet",
                    columns=["i", "j", "centroid_dist_um", "wall_dist_um", "apposed_wall_um"])
e = e[e.centroid_dist_um <= PRUNE_UM]

GRAPHS = [("exact", r"exact contact", S.ORANGE, (e.wall_dist_um <= 0).to_numpy()),
          ("tol", r"contact within 1\,$\mu$m", S.ORANGE, (e.apposed_wall_um > 0).to_numpy()),
          ("voronoi", r"Delaunay, pruned at 40\,$\mu$m (model)", S.BLUE, np.ones(len(e), bool))]


def degree(keep):
    return np.bincount(np.r_[e.i.values[keep], e.j.values[keep]], minlength=n)


connected = degree(GRAPHS[-1][3]) > 0
deg = {key: degree(keep) for key, _, _, keep in GRAPHS}
share = {key: float(100 * (deg[key][connected] == 0).mean()) for key in deg}

# ---- the field: representative of the section, with sparse and dense tissue in it
dens = cKDTree(xy).query_ball_point(xy, r=R_DENS, return_length=True) - 1
idx = np.flatnonzero(connected)
order = np.random.default_rng(0).permutation(len(idx))                 # as fig_contact_kernel
ranks = np.empty(len(idx), dtype=np.int64)
ranks[order[np.argsort(dens[idx][order], kind="stable")]] = np.arange(len(idx))
fifth = np.full(n, -1)
fifth[idx] = (ranks * 5) // len(idx)
lo = xy.min(0)
gx = np.floor((xy - lo) / STEP_UM).astype(int)
k = int(WIN_UM / STEP_UM)
shape = gx.max(0) + 1


def window_count(mask):
    h = np.zeros(shape)
    np.add.at(h, tuple(gx[mask].T), 1)
    c = np.pad(h, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


N = window_count(connected)
eligible = (N >= MIN_CELLS) & (np.minimum(window_count(fifth == 0), window_count(fifth == 4)) >= MIN_FIFTH)
dev = sum(np.abs(window_count(connected & (deg[key] == 0)) / np.maximum(N, 1) - share[key] / 100)
          for key in ("exact", "tol"))
dev[~eligible] = np.inf
bx, by = np.unravel_index(np.argmin(dev), dev.shape)
x0, y0 = lo + np.array([bx, by]) * STEP_UM
field = (x0, x0 + WIN_UM, y0, y0 + WIN_UM)                                # um
inside = np.flatnonzero((xy[:, 0] > x0 - 30) & (xy[:, 0] < x0 + WIN_UM + 30)
                        & (xy[:, 1] > y0 - 30) & (xy[:, 1] < y0 + WIN_UM + 30))
core = (xy[inside, 0] >= x0) & (xy[inside, 0] < x0 + WIN_UM) & (xy[inside, 1] >= y0) & (xy[inside, 1] < y0 + WIN_UM)
field_share = {key: float(100 * (deg[key][inside[core & connected[inside]]] == 0).mean()) for key in deg}

# ---- draw, in um (polygons are stored in pixels)
fig = plt.figure(figsize=(S.TEXTWIDTH, 2.62))
gs = fig.add_gridspec(1, 3, left=0.012, right=0.988, bottom=0.075, top=0.845, wspace=0.045)
sel = set(inside.tolist())
e_in = e[e.i.isin(sel) | e.j.isin(sel)]
for c, (key, title, col, keep) in enumerate(GRAPHS):
    ax = fig.add_subplot(gs[0, c])
    iso = [np.asarray(polys[j].exterior.coords) * mpp for j in inside if deg[key][j] == 0]
    con = [np.asarray(polys[j].exterior.coords) * mpp for j in inside if deg[key][j] > 0]
    ax.add_collection(PolyCollection(con, facecolors=S.CELL_FILL, edgecolors=S.CELL_EDGE, linewidths=0.3, zorder=1))
    ax.add_collection(PolyCollection(iso, facecolors=ISOLATED, edgecolors="#5f5d58", linewidths=0.3, zorder=2))
    m = keep[np.searchsorted(e.index.values, e_in.index.values)]
    ee = e_in[m]
    segs = np.stack([xy[ee.i.values], xy[ee.j.values]], axis=1)
    ax.add_collection(LineCollection(segs, colors=col, linewidths=0.6, zorder=3, capstyle="round"))
    ax.set_xlim(field[0], field[1]); ax.set_ylim(field[2], field[3])
    S.tissue_axes(ax)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=S.CELL_EDGE, lw=0.5, zorder=5))
    ax.text(0.045, 1.075, title, transform=ax.transAxes, fontsize=8, color=S.INK, ha="left", va="bottom")
    ax.text(0.045, 1.02, rf"no neighbour: {share[key]:.1f}\,\% of the section's cells", transform=ax.transAxes,
            fontsize=7, color=S.INK2, ha="left", va="bottom")
    S.panel_label(ax, "abc"[c], x=0.03, y=1.075)
    if c == 2:                                 # one scale for all panels: bar under the last, right-aligned
        ax.plot([1 - 25 / WIN_UM, 1], [-0.045, -0.045], transform=ax.transAxes, color=S.INK, lw=1.2,
                solid_capstyle="butt", clip_on=False)
        ax.text(1 - 25 / WIN_UM - 0.02, -0.045, r"25\,$\mu$m", transform=ax.transAxes, fontsize=7,
                color=S.INK, ha="right", va="center")
# key: the dark fill, in ink beside its sample
fig.patches.append(Rectangle((0.012, 0.022), 0.012, 0.03, transform=fig.transFigure, fc=ISOLATED, ec="#5f5d58", lw=0.3))
fig.text(0.029, 0.037, r"cell with no neighbour in that graph", fontsize=7, color=S.INK, va="center")

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=220)
(HERE / "data").mkdir(exist_ok=True)
json.dump({"section": DS, "no_neighbour_pct_of_connected": share, "no_neighbour_pct_in_field": field_share,
           "mean_degree": {k_: float(v.mean()) for k_, v in deg.items()},
           "field_um": [float(v) for v in field], "field_cells": int(core.sum()),
           "eligible_windows": int(eligible.sum()), "field_abs_dev": float(dev[bx, by])},
          open(HERE / "data" / "fig_graph_compare.json", "w"), indent=1)
print("wrote", OUT.with_suffix(".pdf"), "| section", {k_: round(v, 1) for k_, v in share.items()},
      "| field", {k_: round(v, 1) for k_, v in field_share.items()}, "| cells", int(core.sum()))
