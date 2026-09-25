"""Figure: how the Voronoi face is built, and what bounds it.

Row 1 follows one typical cell from segmentation to its faces: (a) segmented
polygons and centroids; (b) the polygons are dropped, each neighbour contributes
a perpendicular bisector; (c) the bisectors cut one another and the Voronoi
region is clipped to a disc, and what survives between two cells is the face.
Row 2 shows what density does to it: (d) a dense neighbourhood; (e) a cell at a
tissue border, where the unclipped region runs into the gap and the disc bounds
it; (f) every edge of the section against the chord ceiling the clip imposes.

Data: the primary section's stored bundle. Exemplar cells are chosen by the
rules in `pick_*`, never by hand. Output: ../fig_voronoi_face.pdf (+ .png
preview). Run from the repository root:

    python submission_paper/aistats/figures/src/fig_voronoi_face.py
"""
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import shapely
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

from discell.preprocess.geometry import build_voronoi_graph  # noqa: E402

BUNDLE = Path("data/datasets/xenium_prime_ovarian_cancer_ffpe/bundle")
NAME = "full"
CLIP_UM, PRUNE_UM = 30.0, 40.0
OUT = Path(__file__).resolve().parent.parent / "fig_voronoi_face"

S.apply()
a = ad.read_h5ad(BUNDLE / f"{NAME}.h5ad", backed="r")
mpp = float(a.uns["microns_per_pixel"])
cent_all = np.asarray(a.obsm["spatial"], dtype=np.float64)
polys_all = np.asarray(shapely.from_wkb(
    pd.read_parquet(BUNDLE / f"{NAME}_polygons.parquet")["wkb"].to_numpy()), dtype=object)
nn = cKDTree(cent_all).query(cent_all, k=7)[0] * mpp            # um; column 6 = 6th NN
deg = np.asarray(a.obs["voronoi_degree"])


def local(center, halo_um=160.0, clip=CLIP_UM):
    d = np.linalg.norm(cent_all - cent_all[center], axis=1) * mpp
    sel = np.flatnonzero(d <= halo_um)
    lf = int(np.flatnonzero(sel == center)[0])
    g, regs = build_voronoi_graph(polys_all[sel], cent_all[sel], mpp, clip, 100.0)
    return sel, lf, g, regs


def pick_typical():
    """Degree 6 and the 6th-NN distance closest to the section median."""
    ok = np.flatnonzero(deg == 6)
    return int(ok[np.argmin(np.abs(nn[ok, 6] - np.median(nn[:, 6])))])


def pick_dense():
    """Degree >= 6 and the 6th-NN distance closest to the section's 10th percentile."""
    ok = np.flatnonzero(deg >= 6)
    return int(ok[np.argmin(np.abs(nn[ok, 6] - np.percentile(nn[:, 6], 10)))])


def pick_border():
    """A tissue-edge cell (close first neighbours, far sixth) whose region the disc
    cuts by 20-60 % of its area and which keeps >= 3 model edges (d <= 40 um).
    Candidates in a fixed seeded order; the first that qualifies is used."""
    cand = np.flatnonzero((nn[:, 6] > np.percentile(nn[:, 6], 90))
                          & (nn[:, 2] < np.percentile(nn[:, 2], 60)) & (deg >= 3))
    for c in np.random.default_rng(0).permutation(cand)[:80]:
        sel, lf, g, regs = local(int(c))
        _, regs_nc = build_voronoi_graph(polys_all[sel], cent_all[sel], mpp, None, 100.0)
        if regs[lf] is None or regs_nc[lf] is None or regs[lf].is_empty:
            continue
        cut = 1 - regs[lf].area / regs_nc[lf].area
        mine = (g.i == lf) | (g.j == lf)
        kept = int((mine & (g.centroid_dist_um <= PRUNE_UM)).sum())
        if 0.2 <= cut <= 0.6 and kept >= 3:
            return int(c), (sel, lf, g, regs, regs_nc, cut)
    raise RuntimeError("no border exemplar qualified")


# ---------------------------------------------------------------- drawing
def geoms(g):
    if g is None or g.is_empty:
        return []
    return list(g.geoms) if hasattr(g, "geoms") else [g]


def draw_polys(ax, polys, face=S.CELL_FILL, edge=S.CELL_EDGE, lw=0.4, z=1, alpha=1.0):
    verts = [np.asarray(p.exterior.coords) for p in polys if p is not None and not p.is_empty]
    ax.add_collection(PolyCollection(verts, facecolors=face, edgecolors=edge,
                                     linewidths=lw, zorder=z, alpha=alpha))


def draw_regions(ax, regions, color=S.REGION_EDGE, lw=0.5, z=3, ls="-"):
    segs = []
    for r in regions:
        for gg in geoms(r):
            segs.append(np.asarray(gg.exterior.coords))
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=z, linestyles=ls))


def fill_region(ax, region, face=S.FOCAL_FILL, z=2.5):
    """Tint the focal region above the cell polygons and outline it in ink, so
    any stretch of its boundary that is not a face (the clip arc) stays visible."""
    for gg in geoms(region):
        ax.fill(*gg.exterior.xy, facecolor=face, edgecolor="none", zorder=z, alpha=0.9)
        ax.plot(*gg.exterior.xy, color=S.INK2, lw=0.9, zorder=z + 2)


def faces_of(g, regions, k):
    out = []
    for i, j, f, d in zip(g.i, g.j, g.shared_wall_um, g.centroid_dist_um):
        if k not in (int(i), int(j)):
            continue
        shared = regions[int(i)].boundary.intersection(regions[int(j)].boundary)
        out.append((shared, float(f), float(d)))
    return out


def draw_face(ax, shared, color=S.BLUE, lw=2.0, z=6, ls="-"):
    for seg in geoms(shared):
        if hasattr(seg, "xy"):
            ax.plot(*seg.xy, color=color, lw=lw, solid_capstyle="round", zorder=z, ls=ls)


def view(ax, c, half_um):
    h = half_um / mpp
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h)
    S.tissue_axes(ax)


def label_face(ax, shared, text):
    m = shared.centroid
    ax.text(m.x, m.y, text, fontsize=6, ha="center", va="center", color=S.INK, zorder=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


# ---------------------------------------------------------------- panels
typ = pick_typical()
sel, lf, g, regs = local(typ)
cent, polys = cent_all[sel], polys_all[sel]
c0 = cent[lf]
H1 = 17.0                                   # half-width of row-1 panels, um

fig = plt.figure(figsize=(S.TEXTWIDTH, 4.62))
gs = fig.add_gridspec(2, 3, left=0.02, right=0.985, bottom=0.075, top=0.965,
                      wspace=0.12, hspace=0.16)
ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

# (a) segmentation
draw_polys(ax[0], polys)
draw_polys(ax[0], [polys[lf]], face=S.FOCAL_FILL, edge=S.BLUE, lw=0.8, z=2)
ax[0].scatter(cent[:, 0], cent[:, 1], s=3, color=S.INK, zorder=4, lw=0)
view(ax[0], c0, H1); S.scalebar(ax[0], 10, mpp); S.panel_label(ax[0], "a")

# (b) centroids and bisectors
draw_polys(ax[1], polys, face=S.GHOST, edge="#dedcd6", lw=0.3)
nbrs = sorted({int(j) if int(i) == lf else int(i) for i, j in zip(g.i, g.j) if lf in (int(i), int(j))})
for k in nbrs:
    m = 0.5 * (c0 + cent[k]); v = cent[k] - c0
    perp = np.array([-v[1], v[0]]) / np.linalg.norm(v); L = 60 / mpp
    ax[1].plot(*np.array([m - perp * L, m + perp * L]).T, color=S.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=3)
    ax[1].plot(*np.array([c0, cent[k]]).T, color=S.INK2, lw=0.5, zorder=3)
ax[1].scatter(cent[:, 0], cent[:, 1], s=3, color=S.INK, zorder=4, lw=0)
ax[1].scatter(*c0, s=16, color=S.BLUE, zorder=5, lw=0)
view(ax[1], c0, H1); S.panel_label(ax[1], "b")

# (c) clipped regions and faces
draw_polys(ax[2], polys, face=S.GHOST, edge="#dedcd6", lw=0.3)
fill_region(ax[2], regs[lf])
draw_regions(ax[2], regs)
for shared, f, d in faces_of(g, regs, lf):
    draw_face(ax[2], shared)
    label_face(ax[2], shared, rf"{f:.1f}")
ax[2].scatter(*c0, s=12, color=S.BLUE, zorder=7, lw=0)
view(ax[2], c0, H1); S.panel_label(ax[2], "c")

# (d) dense neighbourhood and (e) tissue border, one scale
H2 = 42.0
for axi, (center, pack, letter) in zip(
        [ax[3], ax[4]],
        [(pick_dense(), None, "d"), (*pick_border(), "e")]):
    if pack is None:
        s2, l2, g2, r2 = local(center); r2_nc, cut = None, None
    else:
        s2, l2, g2, r2, r2_nc, cut = pack
    c2, p2 = cent_all[s2], polys_all[s2]
    draw_polys(axi, p2, lw=0.3)
    draw_regions(axi, r2, lw=0.45)
    if r2_nc is not None:
        for gg in geoms(r2_nc[l2]):
            axi.plot(*gg.exterior.xy, color=S.INK2, lw=0.8, ls=(0, (2, 1.5)), zorder=5)
    fill_region(axi, r2[l2])
    axi.add_patch(Circle(c2[l2], CLIP_UM / mpp, fill=False, ec=S.INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=5))
    for shared, f, d in faces_of(g2, r2, l2):
        if d <= PRUNE_UM:
            draw_face(axi, shared, lw=1.8)
        else:
            draw_face(axi, shared, color=S.MUTED, lw=1.4, ls=(0, (2, 1.2)))
    axi.scatter(*c2[l2], s=12, color=S.BLUE, zorder=7, lw=0)
    view(axi, c2[l2], H2); S.scalebar(axi, 20, mpp); S.panel_label(axi, letter)
    if letter == "e":
        border_id, border_cut = a.obs_names[center], cut

# (f) every edge under the chord ceiling
e = pd.read_parquet(BUNDLE / f"{NAME}_edges_voronoi.parquet", columns=["shared_wall_um", "centroid_dist_um"])
x, y = e.centroid_dist_um.to_numpy(), e.shared_wall_um.to_numpy()
cmap = LinearSegmentedColormap.from_list("blue", S.BLUE_RAMP)
axf = ax[5]
YMAX = 71
axf.axvspan(PRUNE_UM, 63, color=S.GHOST, zorder=0, lw=0)
hb = axf.hexbin(x, y, gridsize=(70, 45), extent=(0, 63, 0, 62), cmap=cmap, norm=LogNorm(),
                mincnt=1, linewidths=0.0, zorder=1)
dd = np.linspace(0, 2 * CLIP_UM, 400)
axf.plot(dd, 2 * np.sqrt(np.maximum(CLIP_UM**2 - (dd / 2) ** 2, 0)), color=S.INK, lw=1.0, zorder=3)
axf.axvline(PRUNE_UM, color=S.INK2, lw=0.7, ls=(0, (3, 2)), zorder=3)
axf.set_xlim(0, 63); axf.set_ylim(0, YMAX)
axf.set_xlabel(r"centroid distance $d_{ij}$ ($\mu$m)", labelpad=1.5)
axf.set_ylabel(r"face length $f_{ij}$ ($\mu$m)", labelpad=1.5)
axf.set_xticks([0, 20, 40, 60]); axf.set_yticks([0, 20, 40, 60])
# headroom above the ceiling carries the labels; no edge can sit there
axf.text(1.5, 66.5, r"$f_{ij}\le 2\sqrt{R^2-(d_{ij}/2)^2}$", fontsize=7, color=S.INK, va="center")
axf.text(PRUNE_UM + 1.2, 66.5, r"pruned", fontsize=7, color=S.INK2, va="center")
cax = axf.inset_axes([0.775, 0.765, 0.2, 0.03])
cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, width=0.4, length=2, pad=1)
cb.ax.xaxis.set_label_position("top")
cb.set_label(r"edges", fontsize=6, labelpad=1.5)
axf.set_box_aspect(1)
S.panel_label(axf, "f", x=-0.14)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=220)
print("wrote", OUT.with_suffix(".pdf"))
print("typical", a.obs_names[typ], "| dense", a.obs_names[pick_dense()],
      "| border", border_id, f"(area cut {100 * border_cut:.0f}%)", f"| {len(e):,} edges")
