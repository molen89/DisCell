"""How you get from segmentation polygons (A) to the tessellation (B),
and what happens as the neighbourhood empties out."""
import numpy as np, pandas as pd, anndata as ad, shapely
from shapely.geometry import Point
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from discell.preprocess.geometry import build_voronoi_graph

BUNDLE = "data/datasets/gse315411_pdltma06_10_prime_dual/bundle"
NAME, CLIP_UM = "pdl018d", 30.0
OUT = "scratchpad"

a = ad.read_h5ad(f"{BUNDLE}/{NAME}.h5ad", backed="r")
mpp = float(a.uns["microns_per_pixel"])
cent_all = np.asarray(a.obsm["spatial"], dtype=np.float64)
polys_all = np.array(list(shapely.from_wkb(
    pd.read_parquet(f"{BUNDLE}/{NAME}_polygons.parquet")["wkb"].to_numpy())), dtype=object)

nn = cKDTree(cent_all).query(cent_all, k=7)[0] * mpp


def patch(centre_idx, halo_um=170.0):
    d = np.linalg.norm(cent_all - cent_all[centre_idx], axis=1) * mpp
    s = np.flatnonzero(d <= halo_um)
    lf = int(np.flatnonzero(s == centre_idx)[0])
    g, regs = build_voronoi_graph(polys_all[s], cent_all[s], mpp, CLIP_UM, 100.0)
    return s, lf, cent_all[s], polys_all[s], g, regs


def view(ax, c, half_um):
    h = half_um / mpp
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def draw_region(ax, g, fill=None, **kw):
    if g is None or g.is_empty:
        return
    for gg in (g.geoms if hasattr(g, "geoms") else [g]):
        xs, ys = gg.exterior.xy
        if fill:
            ax.fill(xs, ys, facecolor=fill, zorder=0)
        ax.plot(xs, ys, **kw)


def scalebar(ax, c, half_um, length_um=20):
    h = half_um / mpp
    x0, y0 = c[0] - h * 0.9, c[1] - h * 0.85
    ax.plot([x0, x0 + length_um / mpp], [y0, y0], color="#111", lw=2.5)
    ax.annotate(f"{length_um} um", (x0 + length_um / (2 * mpp), y0),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)


# a typical dense interior cell
dense_i = int(np.argsort(np.abs(nn[:, 6] - np.median(nn[:, 6])))[0])
sel, lf, cent, polys, graph, regions = patch(dense_i)
HALF = 26.0

fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))

# ---- step 1: what we start with
ax = axes[0, 0]
for p in polys:
    xs, ys = p.exterior.xy
    ax.fill(xs, ys, facecolor="#cfd8e3", edgecolor="#5a6b82", lw=0.7)
ax.plot(cent[:, 0], cent[:, 1], ".", color="#b4341f", ms=6)
view(ax, cent[lf], HALF); scalebar(ax, cent[lf], HALF)
ax.set_title("1  what we start with\nsegmentation polygons + their centroids", fontsize=10)

# ---- step 2: throw the polygons away, keep the points, draw bisectors
ax = axes[0, 1]
for p in polys:
    xs, ys = p.exterior.xy
    ax.fill(xs, ys, facecolor="#f2f4f7", edgecolor="#dde3ea", lw=0.5)
nbrs = sorted({int(j) if int(i) == lf else int(i)
               for i, j in zip(graph.i, graph.j) if lf in (int(i), int(j))})
c0 = cent[lf]
for k in nbrs:
    m = 0.5 * (c0 + cent[k])
    v = cent[k] - c0
    perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    L = 40 / mpp
    ax.plot(*np.array([m - perp * L, m + perp * L]).T, color="#8d9db0", lw=1.0, ls="--")
    ax.plot(*np.array([c0, cent[k]]).T, color="#b4341f", lw=0.8, alpha=0.6)
ax.plot(cent[:, 0], cent[:, 1], ".", color="#b4341f", ms=6)
ax.plot(*c0, "o", color="#b4341f", ms=9)
view(ax, cent[lf], HALF); scalebar(ax, cent[lf], HALF)
ax.set_title("2  polygons discarded, points kept\nperpendicular bisector to each neighbour "
             "(dashed)", fontsize=10)

# ---- step 3: the bisectors cut each other -> the region
ax = axes[0, 2]
for p in polys:
    xs, ys = p.exterior.xy
    ax.fill(xs, ys, facecolor="#f2f4f7", edgecolor="#dde3ea", lw=0.5)
for g in regions:
    draw_region(ax, g, color="#8d9db0", lw=0.9)
draw_region(ax, regions[lf], fill="#fdf0d2", color="#1f3d5c", lw=2.4)
for i, j, w in zip(graph.i, graph.j, graph.shared_wall_um):
    if lf not in (int(i), int(j)):
        continue
    sh = regions[int(i)].boundary.intersection(regions[int(j)].boundary)
    for seg in (sh.geoms if hasattr(sh, "geoms") else [sh]):
        if not seg.is_empty and hasattr(seg, "xy"):
            ax.plot(*seg.xy, color="#b4341f", lw=4.5, solid_capstyle="round", zorder=5)
    if not sh.is_empty:
        m = sh.centroid
        ax.annotate(f"{w:.1f}", (m.x, m.y), fontsize=8, ha="center", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6))
ax.plot(*c0, "o", color="#b4341f", ms=7, zorder=7)
view(ax, cent[lf], HALF); scalebar(ax, cent[lf], HALF)
ax.set_title("3  each bisector is cut by the others\nwhat survives is the face f_ij, in um",
             fontsize=10)

# ---- density series: same scale, emptier and emptier
targets = [("dense", 5), ("sparse", 92), ("very sparse", 99.5)]
HALF2 = 46.0
for col, (lab, q) in enumerate(targets[:2]):
    ax = axes[1, col]
    tgt = np.percentile(nn[:, 6], q)
    idx = int(np.argsort(np.abs(nn[:, 6] - tgt))[0])
    s2, lf2, c2, p2, g2, r2 = patch(idx, halo_um=200.0)
    for p in p2:
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, facecolor="#cfd8e3", edgecolor="#9fb0c4", lw=0.5)
    for g in r2:
        draw_region(ax, g, color="#8d9db0", lw=0.8)
    draw_region(ax, r2[lf2], fill="#fdf0d2", color="#1f3d5c", lw=2.4)
    ax.add_patch(Circle(c2[lf2], CLIP_UM / mpp, fill=False, ec="#b4341f", lw=1.3, ls=":"))
    fl = []
    for i, j, w in zip(g2.i, g2.j, g2.shared_wall_um):
        if lf2 not in (int(i), int(j)):
            continue
        fl.append(w)
        sh = r2[int(i)].boundary.intersection(r2[int(j)].boundary)
        for seg in (sh.geoms if hasattr(sh, "geoms") else [sh]):
            if not seg.is_empty and hasattr(seg, "xy"):
                ax.plot(*seg.xy, color="#b4341f", lw=4.0, solid_capstyle="round", zorder=5)
    ax.plot(*c2[lf2], "o", color="#b4341f", ms=6, zorder=7)
    view(ax, c2[lf2], HALF2); scalebar(ax, c2[lf2], HALF2)
    ax.set_title(f"{'4' if col==0 else '5'}  {lab} neighbourhood "
                 f"(6th-NN {nn[idx,6]:.0f} um)\n{len(fl)} faces, "
                 f"median {np.median(fl):.1f} um, max {max(fl):.1f} um", fontsize=10)

# ---- the ceiling the clip imposes
ax = axes[1, 2]
e = pd.read_parquet(f"{BUNDLE}/{NAME}_edges_voronoi.parquet")
f, d = e.shared_wall_um.to_numpy(), e.centroid_dist_um.to_numpy()
ax.hexbin(d, f, gridsize=90, bins="log", cmap="Blues", mincnt=1)
dd = np.linspace(0.01, 60, 400)
ax.plot(dd, 2 * np.sqrt(np.maximum(CLIP_UM**2 - (dd / 2) ** 2, 0)), color="#b4341f",
        lw=2.2, label=r"chord ceiling $2\sqrt{R^2-(d/2)^2}$")
ax.axvline(2 * CLIP_UM, color="#1b7f3b", lw=1.6, ls="--",
           label=f"hard cutoff d = 2R = {2*CLIP_UM:.0f} um")
ax.set_xlabel("centroid distance $d_{ij}$ (um)"); ax.set_ylabel("face $f_{ij}$ (um)")
ax.set_xlim(0, 66); ax.set_ylim(0, 62)
ax.legend(fontsize=8, loc="upper right")
ax.set_title(f"6  all {len(e):,} edges of this slide\nthe clip caps the face and kills the "
             "edge past 60 um", fontsize=10)

fig.suptitle("From polygons to faces, and what sparsity does to them "
             f"- {NAME}, R = {CLIP_UM:g} um", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/face_construction.png", dpi=145)
print("wrote", f"{OUT}/face_construction.png")
