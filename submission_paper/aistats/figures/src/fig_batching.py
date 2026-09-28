"""Figure: tiles, the held-out split, and the two-hop halo of one training tile.

(a) The primary section cut into contiguous tiles by recursive median split, as
the model does it (at most 4096 cells per tile), with the held-out tiles of the
seed-0 split hatched. (b) A window where one training tile meets a held-out tile
and another training tile: the tile's cells (seeds; loss and gradients), ring one
(its graph neighbours; context, response and clean composition computed, used in
rho_bar under a stop-gradient), ring two (neighbours of ring one; type only, as
attention sources for ring one). Cells of held-out tiles are hatched: ring cells
drawn from them are the transductive use stated in method section 2.6.

The tiles and split are recomputed with the model's own functions and seed; the
tile outlines come from a bounds-tracking copy of the same recursive split,
checked against it. The window is the one, among ring-one cells of training
tiles, that holds the most ring-one cells of a held-out tile and of another
training tile at once (the larger of the two smaller counts). Outputs
../fig_batching.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

from discell.model.prepare import rings, spatial_tiles  # noqa: E402

DS, NAME = "xenium_prime_ovarian_cancer_ffpe", "full"
BUNDLE = Path(f"data/datasets/{DS}/bundle")
PRUNE_UM, TILE_CELLS, VAL_FRACTION, SEED, WIN_UM = 40.0, 4096, 0.15, 0, 110.0
OUT = Path(__file__).resolve().parent.parent / "fig_batching"
ROLE = {"seed": (S.FOCAL_FILL, "#9ec5f4", 0.45), "ring1": ("#b9b7b0", "#8f8d86", 0.4),
        "ring2": ("#dcdad4", "#b5b3ab", 0.35), "other": (S.GHOST, "#e2e0da", 0.3)}
HATCH = "//////"


def tiles_with_bounds(pos, max_cells, box):
    """spatial_tiles with the rectangle of each tile tracked (same split order)."""
    out, stack = [], [(np.arange(len(pos)), box)]
    while stack:
        idx, (x0, y0, x1, y1) = stack.pop()
        if len(idx) <= max_cells:
            out.append((np.sort(idx), (x0, y0, x1, y1)))
            continue
        span = pos[idx].max(axis=0) - pos[idx].min(axis=0)
        axis = int(span[1] > span[0])
        order = idx[np.argsort(pos[idx, axis], kind="stable")]
        mid = len(order) // 2
        cut = 0.5 * (pos[order[mid - 1], axis] + pos[order[mid], axis])
        lo = (x0, y0, cut, y1) if axis == 0 else (x0, y0, x1, cut)
        hi = (cut, y0, x1, y1) if axis == 0 else (x0, cut, x1, y1)
        stack += [(order[:mid], lo), (order[mid:], hi)]
    return out


S.apply()
a = ad.read_h5ad(BUNDLE / f"{NAME}.h5ad", backed="r")
n = a.n_obs
xy = np.asarray(a.obsm["spatial_um"], dtype=np.float64)
mpp = float(a.uns["microns_per_pixel"])
polys = np.asarray(shapely.from_wkb(pd.read_parquet(BUNDLE / f"{NAME}_polygons.parquet")["wkb"].to_numpy()), dtype=object)
e = pd.read_parquet(BUNDLE / f"{NAME}_edges_voronoi.parquet", columns=["i", "j", "centroid_dist_um"])
e = e[e.centroid_dist_um <= PRUNE_UM]
A = sp.csr_matrix((np.ones(2 * len(e)), (np.r_[e.i.values, e.j.values], np.r_[e.j.values, e.i.values])), shape=(n, n))

# ---- tiles and split, exactly as the model builds them
box = (*(xy.min(0) - 1.0), *(xy.max(0) + 1.0))
tb = tiles_with_bounds(xy, TILE_CELLS, box)
ref = spatial_tiles(xy, TILE_CELLS)
assert len(tb) == len(ref) and all(np.array_equal(t, r) for (t, _), r in zip(tb, ref)), "split copy diverged"
order = np.random.default_rng([SEED, 2]).permutation(len(ref))
n_val = max(1, int(round(VAL_FRACTION * len(ref))))
held = np.zeros(len(ref), bool); held[order[:n_val]] = True
tile_of = np.empty(n, dtype=np.int64)
for k, (idx, _) in enumerate(tb):
    tile_of[idx] = k

# ---- the window: most ring-one cells from a held-out tile AND from another training tile
tree = cKDTree(xy)
best = (-1, None, None)
for k in np.flatnonzero(~held):
    seeds = tb[k][0]
    r1, _ = rings(A, seeds)
    if not held[tile_of[r1]].any():
        continue
    in_held = held[tile_of[r1]]
    for c in r1[in_held][:: max(1, in_held.sum() // 60)]:
        near = np.asarray(tree.query_ball_point(xy[c], WIN_UM / 2, p=np.inf))
        near_r1 = np.intersect1d(near, r1)
        h = held[tile_of[near_r1]]
        score = min(h.sum(), (~h).sum())
        if score > best[0]:
            best = (score, k, xy[c].copy())
_, K, centre = best
seeds = tb[K][0]
r1, r2 = rings(A, seeds)
role = np.full(n, "other", dtype=object)
role[seeds], role[r1], role[r2] = "seed", "ring1", "ring2"
half = WIN_UM / 2
win = np.flatnonzero(np.all(np.abs(xy - centre) <= half + 25, axis=1))

# ---- figure
fig = plt.figure(figsize=(S.TEXTWIDTH, 2.55))
axa = fig.add_axes([0.025, 0.07, 0.405, 0.84])
axb = fig.add_axes([0.445, 0.07, 0.30, 0.84])

# (a) the section and its tiles
axa.scatter(xy[::2, 0], xy[::2, 1], s=0.03, color="#a3a19a", lw=0, rasterized=True, zorder=1)
for k, (_, (x0, y0, x1, y1)) in enumerate(tb):
    axa.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=S.INK2, lw=0.35, zorder=2,
                            hatch=HATCH if held[k] else None))
x0, y0, x1, y1 = tb[K][1]
axa.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=S.BLUE, lw=1.3, zorder=4))
axa.add_patch(Rectangle(tuple(centre - half), WIN_UM, WIN_UM, fill=False, ec=S.INK, lw=0.9, zorder=5))
axa.text(centre[0] + half + 90, centre[1] - half, r"\textbf{b}", fontsize=7, color=S.INK, va="top", zorder=6)
axa.set_xlim(box[0], box[2]); axa.set_ylim(box[1], box[3])
S.tissue_axes(axa)
axa.plot([box[0] + 250, box[0] + 1250], [box[3] - 250] * 2, color=S.INK, lw=1.2, solid_capstyle="butt")
axa.text(box[0] + 750, box[3] - 330, r"1\,mm", fontsize=7, color=S.INK, ha="center", va="bottom")
S.panel_label(axa, "a", x=-0.005, y=1.0)

# (b) one tile's halo
plt.rcParams["hatch.linewidth"] = 0.4
for r, (fc, ec, lw) in ROLE.items():
    for is_held in (False, True):
        sel = [j for j in win if role[j] == r and held[tile_of[j]] == is_held]
        if not sel:
            continue
        axb.add_collection(PolyCollection([np.asarray(polys[j].exterior.coords) * mpp for j in sel],
                                          facecolors=fc, edgecolors="#6f6d67" if is_held else ec,
                                          linewidths=lw, hatch=HATCH if is_held else None,
                                          zorder=2 if r != "seed" else 3))
ew = e[e.i.isin(win) & e.j.isin(win)]
ri, rj = role[ew.i.values], role[ew.j.values]
into_seed = ((ri == "seed") & (rj == "ring1")) | ((ri == "ring1") & (rj == "seed"))
into_r1 = ((ri == "ring1") & (rj == "ring2")) | ((ri == "ring2") & (rj == "ring1"))
for m, col, ls in ((into_r1, S.INK2, (0, (2, 1.2))), (into_seed, S.BLUE, "solid")):
    segs = np.stack([xy[ew.i.values[m]], xy[ew.j.values[m]]], axis=1)
    axb.add_collection(LineCollection(segs, colors=col, linewidths=0.8, linestyles=[ls], zorder=4))
# tile outlines inside the window
for k, (_, (tx0, ty0, tx1, ty1)) in enumerate(tb):
    if tx1 < centre[0] - half or tx0 > centre[0] + half or ty1 < centre[1] - half or ty0 > centre[1] + half:
        continue
    axb.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, fill=False, ec=S.INK, lw=0.8,
                            ls=(0, (4, 2)), zorder=5))
axb.set_xlim(centre[0] - half, centre[0] + half); axb.set_ylim(centre[1] - half, centre[1] + half)
S.tissue_axes(axb)
axb.add_patch(Rectangle((0, 0), 1, 1, transform=axb.transAxes, fill=False, ec=S.INK, lw=0.6, zorder=6))
bx, by = centre[0] - half + 6, centre[1] + half - 7
axb.plot([bx, bx + 20], [by, by], color=S.INK, lw=1.2, solid_capstyle="butt", zorder=7)
axb.text(bx + 10, by - 2, r"20\,$\mu$m", fontsize=7, color=S.INK, ha="center", va="bottom", zorder=7,
         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
S.panel_label(axb, "b", x=-0.01, y=1.0)

# key, in ink beside its samples
keys = [Patch(fc=ROLE["seed"][0], ec=ROLE["seed"][1], lw=0.6, label="the tile's cells (seeds):\nloss and gradients"),
        Patch(fc=ROLE["ring1"][0], ec=ROLE["ring1"][1], lw=0.5,
              label="ring one: context, response\nand clean composition\ncomputed; used under\na stop-gradient"),
        Patch(fc=ROLE["ring2"][0], ec=ROLE["ring2"][1], lw=0.5, label="ring two: type only,\nas attention sources"),
        Patch(fc="white", ec="#6f6d67", lw=0.5, hatch=HATCH, label="cell of a held-out tile"),
        Line2D([], [], color=S.BLUE, lw=0.9, label="edges into the seeds\n(leakage and attention)"),
        Line2D([], [], color=S.INK2, lw=0.9, ls=(0, (2, 1.2)), label="edges from ring two\n(attention)"),
        Line2D([], [], color=S.INK, lw=0.8, ls=(0, (4, 2)), label="tile boundary")]
fig.legend(handles=keys, loc="center left", bbox_to_anchor=(0.755, 0.49), frameon=False, fontsize=6.5,
           handlelength=1.6, handleheight=1.1, labelspacing=0.75, borderaxespad=0)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=220)
print("wrote", OUT.with_suffix(".pdf"), f"| tiles {len(tb)}, held out {held.sum()} | tile {K}: seeds {len(seeds)}, "
      f"ring1 {len(r1)} ({100 * len(r1) / len(seeds):.1f}%), ring2 {len(r2)} ({100 * len(r2) / len(seeds):.1f}%) "
      f"| window score {best[0]}")
