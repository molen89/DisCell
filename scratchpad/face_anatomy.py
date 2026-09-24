"""What the Voronoi face actually measures, drawn on real cells.

Diagnostic only: illustrates the preprocessing claim in the AISTATS draft.
Uses discell.preprocess.geometry directly so the picture cannot drift from
the code that builds the bundles.
"""
import numpy as np, pandas as pd, anndata as ad, shapely
from shapely import wkb as shp_wkb
from shapely.geometry import Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from discell.preprocess.geometry import build_voronoi_graph, apposed_wall_um

BUNDLE = "data/datasets/gse315411_pdltma06_10_prime_dual/bundle"
NAME = "pdl018d"
CLIP_UM, TOL_UM = 30.0, 1.0
OUT = "scratchpad"

a = ad.read_h5ad(f"{BUNDLE}/{NAME}.h5ad", backed="r")
mpp = float(a.uns["microns_per_pixel"])
cent_all = np.asarray(a.obsm["spatial"], dtype=np.float64)
polys_df = pd.read_parquet(f"{BUNDLE}/{NAME}_polygons.parquet")
assert list(polys_df["cell"]) == list(a.obs_names), "polygon order must match obs"
polys_all = np.array([shp_wkb.loads(b) for b in polys_df["wkb"]], dtype=object)

deg = np.asarray(a.obs["voronoi_degree"])
focal = np.flatnonzero(deg == int(np.median(deg)))
focal = focal[len(focal) // 2]

# Voronoi over a generous halo so displayed cells get their true regions.
HALO_UM = 260.0
d_um = np.linalg.norm(cent_all - cent_all[focal], axis=1) * mpp
sel = np.flatnonzero(d_um <= HALO_UM)
lf = int(np.flatnonzero(sel == focal)[0])
cent, polys = cent_all[sel], polys_all[sel]

graph, regions = build_voronoi_graph(polys, cent, mpp, CLIP_UM, 100.0)
graph_nc, regions_nc = build_voronoi_graph(polys, cent, mpp, None, 100.0)
app = apposed_wall_um(polys, graph.i, graph.j, mpp, TOL_UM)

# --- does the clip ever bind? only count cells well inside the halo, so the
# unbounded outer ring of the local tessellation does not fake a border.
inner = np.flatnonzero(np.linalg.norm(cent - cent[lf], axis=1) * mpp <= HALO_UM - 60)
bound = np.array([
    regions_nc[k] is not None and regions[k] is not None
    and regions[k].area < regions_nc[k].area * (1 - 1e-6) for k in inner])
print(f"clip binds on {100*bound.mean():.1f}% of {len(inner)} interior cells "
      f"(region area reduced by the {CLIP_UM:g} um disc)")

key_nc = {(int(i), int(j)): w for i, j, w in zip(graph_nc.i, graph_nc.j, graph_nc.shared_wall_um)}
key_c = {(int(i), int(j)): w for i, j, w in zip(graph.i, graph.j, graph.shared_wall_um)}
inset = set(inner.tolist())
shared_keys = [k for k in key_nc if k[0] in inset and k[1] in inset]
dropped = [k for k in shared_keys if k not in key_c]
shrunk = [k for k in shared_keys if k in key_c and key_c[k] < key_nc[k] * (1 - 1e-6)]
print(f"of {len(shared_keys)} interior edges: {len(dropped)} removed by the clip, "
      f"{len(shrunk)} shortened")

nb = [(int(i), int(j), w, g, ap) for i, j, w, g, ap
      in zip(graph.i, graph.j, graph.shared_wall_um, graph.wall_dist_um, app)
      if lf in (i, j)]
print(f"\nfocal cell {a.obs_names[focal]}, {len(nb)} voronoi neighbours")
for i, j, w, g, ap in nb:
    print(f"  face {w:6.2f} um | polygon gap {g:6.2f} um | apposed(1um) {ap:6.2f} um")

gaps = graph.wall_dist_um
print(f"\nover {len(gaps)} local voronoi edges: {100*(gaps <= 0).mean():.1f}% of "
      f"polygon pairs touch, median gap {np.median(gaps):.2f} um, "
      f"{100*(app > 0).mean():.1f}% have nonzero apposed wall at {TOL_UM:g} um")

# A border cell where the disc really does bite, chosen slide-wide. Want a
# tissue EDGE, not an isolated cell: dense on one side, open on the other, so
# the region still has real faces as well as clipped arcs.
from scipy.spatial import cKDTree
tree_all = cKDTree(cent_all)
nn = tree_all.query(cent_all, k=7)[0] * mpp
edgey = np.flatnonzero((nn[:, 6] > 22) & (nn[:, 6] < 60) & (nn[:, 2] < 12))
print(f"{len(edgey)} tissue-edge candidates")

best = None
rng = np.random.default_rng(0)
for c in rng.permutation(edgey)[:60]:
    c = int(c)
    d = np.linalg.norm(cent_all - cent_all[c], axis=1) * mpp
    sidx = np.flatnonzero(d <= 200.0)
    if len(sidx) < 40:
        continue
    lfc = int(np.flatnonzero(sidx == c)[0])
    g, regs = build_voronoi_graph(polys_all[sidx], cent_all[sidx], mpp, CLIP_UM, 100.0)
    _, regs_nc = build_voronoi_graph(polys_all[sidx], cent_all[sidx], mpp, None, 100.0)
    if regs[lfc] is None or regs_nc[lfc] is None or regs[lfc].is_empty:
        continue
    cut = 1 - regs[lfc].area / regs_nc[lfc].area
    ndeg = int(((g.i == lfc) | (g.j == lfc)).sum())
    if ndeg >= 3 and (best is None or cut > best[0]):
        best = (cut, c, sidx, lfc, g, regs, regs_nc)
assert best is not None, "no tissue-edge cell found"
bcut, pick, bsel, blf, bgraph, bregions, bregions_nc = best
bcent, bpolys = cent_all[bsel], polys_all[bsel]
print(f"border exemplar: {a.obs_names[pick]}, 6th-NN {nn[pick,6]:.1f} um, "
      f"area cut {100*bcut:.0f}%, degree "
      f"{int(((bgraph.i==blf)|(bgraph.j==blf)).sum())}")


def draw_region(ax, g, fill=None, **kw):
    if g is None or g.is_empty:
        return
    for gg in (g.geoms if hasattr(g, "geoms") else [g]):
        xs, ys = gg.exterior.xy
        if fill:
            ax.fill(xs, ys, facecolor=fill, zorder=0)
        ax.plot(xs, ys, **kw)


def view(ax, c, half_um):
    h = half_um / mpp
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def faces_of(ax, k, label=True):
    out = []
    for i, j, w, *_ in [(int(i), int(j), w) for i, j, w in
                        zip(graph.i, graph.j, graph.shared_wall_um) if k in (i, j)]:
        sh = regions[i].boundary.intersection(regions[j].boundary)
        for seg in (sh.geoms if hasattr(sh, "geoms") else [sh]):
            if not seg.is_empty and hasattr(seg, "xy"):
                sx, sy = seg.xy
                ax.plot(sx, sy, color="#b4341f", lw=4.5, solid_capstyle="round", zorder=5)
        if label and not sh.is_empty:
            m = sh.centroid
            ax.annotate(f"{w:.1f}", (m.x, m.y), fontsize=8, ha="center", zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        out.append(w)
    return out


fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# A: what segmentation gives us, wide enough to show a tissue gap
ax = axes[0, 0]
for p in polys:
    xs, ys = p.exterior.xy
    ax.fill(xs, ys, facecolor="#cfd8e3", edgecolor="#5a6b82", lw=0.5)
ax.plot(cent[:, 0], cent[:, 1], ".", color="#b4341f", ms=2.5)
xs, ys = polys[lf].exterior.xy
ax.fill(xs, ys, facecolor="#f2b441", edgecolor="#8a5c00", lw=1.5, zorder=3)
view(ax, cent[lf], 70)
ax.set_title("A  segmentation polygons + centroids\n"
             "nuclear-expansion outlines; they mostly do not touch", fontsize=9)

# B: interior -- clip is a no-op, faces are pure bisectors
ax = axes[0, 1]
for p in polys:
    xs, ys = p.exterior.xy
    ax.fill(xs, ys, facecolor="#e2e8f0", edgecolor="#b0bccb", lw=0.4)
for g in regions:
    draw_region(ax, g, color="#8d9db0", lw=0.8)
draw_region(ax, regions[lf], fill="#fdf0d2", color="#1f3d5c", lw=2.2)
ax.add_patch(Circle(cent[lf], CLIP_UM / mpp, fill=False, ec="#b4341f", lw=1.2, ls=":"))
faces_of(ax, lf)
ax.plot(cent[lf, 0], cent[lf, 1], "o", color="#b4341f", ms=5, zorder=7)
view(ax, cent[lf], 26)
ax.set_title(f"B  interior cell: clipped region (blue) = Voronoi cell\n"
             f"the {CLIP_UM:g} um disc (dotted) never reaches it; red = f_ij in um",
             fontsize=9)

# C: border cell -- the disc actually bites, arcs appear
ax = axes[1, 0]
for p in bpolys:
    xs, ys = p.exterior.xy
    ax.fill(xs, ys, facecolor="#e2e8f0", edgecolor="#b0bccb", lw=0.4)
draw_region(ax, bregions_nc[blf], color="#8d9db0", lw=1.4, ls="--")
draw_region(ax, bregions[blf], fill="#fdf0d2", color="#1f3d5c", lw=2.4)
ax.add_patch(Circle(bcent[blf], CLIP_UM / mpp, fill=False, ec="#b4341f", lw=1.2, ls=":"))
for i, j, w in zip(bgraph.i, bgraph.j, bgraph.shared_wall_um):
    if blf not in (int(i), int(j)):
        continue
    sh = bregions[int(i)].boundary.intersection(bregions[int(j)].boundary)
    for seg in (sh.geoms if hasattr(sh, "geoms") else [sh]):
        if not seg.is_empty and hasattr(seg, "xy"):
            sx, sy = seg.xy
            ax.plot(sx, sy, color="#b4341f", lw=4.5, solid_capstyle="round", zorder=5)
    if not sh.is_empty:
        m = sh.centroid
        ax.annotate(f"{w:.1f}", (m.x, m.y), fontsize=8, ha="center", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
ax.plot(bcent[blf, 0], bcent[blf, 1], "o", color="#b4341f", ms=5, zorder=7)
view(ax, bcent[blf], 42)
ax.set_title(f"C  border cell ({a.obs_names[pick]}): dashed = unclipped Voronoi,\n"
             f"running off into the gap. solid = after clipping, area cut "
             f"{100*bcut:.0f}%; curves are the disc", fontsize=9)

# D: one edge, both metrics
ax = axes[1, 1]
i, j, w, gap, ap = max(nb, key=lambda t: t[4])
for k, col in ((i, "#f2b441"), (j, "#7ec8a9")):
    xs, ys = polys[k].exterior.xy
    ax.fill(xs, ys, facecolor=col, edgecolor="#333", lw=1.0, alpha=0.85)
draw_region(ax, regions[i], color="#1f3d5c", lw=1.2)
draw_region(ax, regions[j], color="#1f3d5c", lw=1.2)
sh = regions[i].boundary.intersection(regions[j].boundary)
for seg in (sh.geoms if hasattr(sh, "geoms") else [sh]):
    if not seg.is_empty and hasattr(seg, "xy"):
        sx, sy = seg.xy
        ax.plot(sx, sy, color="#b4341f", lw=5.5, solid_capstyle="round", zorder=5)
for k, other in ((i, j), (j, i)):
    near = shapely.intersection(polys[k].boundary,
                                shapely.buffer(polys[other], TOL_UM / mpp))
    for seg in (near.geoms if hasattr(near, "geoms") else [near]):
        if not seg.is_empty and hasattr(seg, "xy"):
            sx, sy = seg.xy
            ax.plot(sx, sy, color="#1b7f3b", lw=3.5, solid_capstyle="round", zorder=4)
ax.plot([], [], color="#b4341f", lw=5.5, label=f"Voronoi face  f_ij = {w:.1f} um  (USED)")
ax.plot([], [], color="#1b7f3b", lw=3.5,
        label=f"apposed wall, tol {TOL_UM:g} um = {ap:.1f} um  (not used)")
ax.plot(cent[[i, j], 0], cent[[i, j], 1], "o-", color="#111", ms=4, lw=1.0,
        label=f"centroid distance  d_ij = "
              f"{np.linalg.norm(cent[i]-cent[j])*mpp:.1f} um  (USED)")
mx = 0.5 * (cent[i] + cent[j])
view(ax, mx, 13)
ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
ax.set_title(f"D  one edge, zoomed. polygon gap here = {gap:.2f} um\n"
             "red sits in the empty space between cells; green rides the membranes",
             fontsize=9)

fig.suptitle(f"What the Voronoi face measures - {NAME}, focal cell {a.obs_names[focal]}",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/face_anatomy.png", dpi=150)
print("wrote", f"{OUT}/face_anatomy.png")
