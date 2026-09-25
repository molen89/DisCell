"""Figure: why the leakage kernel is built on Voronoi faces, not polygon contact.

(a) An edge whose segmented polygons touch: the Voronoi face (blue), the
membrane each cell runs within 1 um of the other (orange), and the centroid
distance. (b) An edge at the same scale whose polygons do not touch: contact is
zero, the face is still defined. (c) On every section, the share of cells that a
contact-weighted kernel would leave with no leakage at all, by local-density
fifth; the Voronoi kernel leaves none in any fifth.

Both kernels in (c) use the model's edge set (Voronoi adjacency, d <= 40 um),
the same decay exp(-d/tau) and the same row normalisation, exactly as in
tab:contact. Local density = centroids within 20 um. Exemplar edges are chosen
by the rules in `pick_edge`. Outputs ../fig_contact_kernel.pdf (+ .png) and the
plotted numbers in data/fig_contact_kernel.json. Run from the repository root.
"""
from pathlib import Path
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import shapely
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

from discell.preprocess.geometry import build_voronoi_graph  # noqa: E402

PRUNE_UM, TOL_UM, R_DENS, TAU = 40.0, 1.0, 20.0, 20.0
PRIMARY = ("xenium_prime_ovarian_cancer_ffpe", "full")
SECTIONS = [("xenium_prime_ovarian_cancer_ffpe", "full"), ("xenium_prime_human_lung_cancer_ffpe", "full"),
            ("xenium_prime_human_ovary_ff", "full"), ("gse315411_pdltma06_11_prime_solo", "full"),
            ("gse315411_pdltma06_10_prime_dual", "full")]
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fig_contact_kernel"


def bundle(ds, var):
    return Path(f"data/datasets/{ds}/bundle"), var


# ------------------------------------------------ (c): no-leak share by density fifth
def noleak_by_fifth(ds, var):
    b, v = bundle(ds, var)
    a = ad.read_h5ad(b / f"{v}.h5ad", backed="r")
    xy = np.asarray(a.obsm["spatial_um"], dtype=np.float64)
    e = pd.read_parquet(b / f"{v}_edges_voronoi.parquet",
                        columns=["i", "j", "centroid_dist_um", "apposed_wall_um", "shared_wall_um"])
    e = e[e.centroid_dist_um <= PRUNE_UM]
    n = a.n_obs
    recv = np.concatenate([e.i.values, e.j.values])
    decay = np.exp(-np.tile(e.centroid_dist_um.values, 2) / TAU)
    contact_row = np.bincount(recv, np.tile(e.apposed_wall_um.values, 2) * decay, minlength=n)
    face_row = np.bincount(recv, np.tile(e.shared_wall_um.values, 2) * decay, minlength=n)
    connected = np.bincount(recv, minlength=n) > 0
    dens = cKDTree(xy).query_ball_point(xy, r=R_DENS, return_length=True) - 1
    idx = np.flatnonzero(connected)
    # equal-sized fifths by rank, ties broken by a seeded shuffle
    order = np.random.default_rng(0).permutation(len(idx))
    ranks = np.empty(len(idx), dtype=np.int64)
    ranks[order[np.argsort(dens[idx][order], kind="stable")]] = np.arange(len(idx))
    fifth = (ranks * 5) // len(idx)
    contact = [float(100 * (contact_row[idx][fifth == q] <= 0).mean()) for q in range(5)]
    face = [float(100 * (face_row[idx][fifth == q] <= 0).mean()) for q in range(5)]
    return {"contact_noleak_pct": contact, "face_noleak_pct": face, "cells": int(len(idx))}


# ------------------------------------------------ (a), (b): exemplar edges
def pick_edge(e, touching):
    """Among model edges, touching (gap 0, apposed wall > 0) or not (gap in
    1.5-3 um, apposed wall 0), the one whose centroid distance and face length
    are jointly closest to the section medians."""
    e = e[e.centroid_dist_um <= PRUNE_UM]
    md, mf = e.centroid_dist_um.median(), e.shared_wall_um.median()
    if touching:
        c = e[(e.wall_dist_um <= 0) & (e.apposed_wall_um > 0)]
    else:
        c = e[(e.wall_dist_um.between(1.5, 3.0)) & (e.apposed_wall_um <= 0)]
    score = ((c.centroid_dist_um - md) / md) ** 2 + ((c.shared_wall_um - mf) / mf) ** 2
    return c.loc[score.idxmin()]


S.apply()
b, v = bundle(*PRIMARY)
a = ad.read_h5ad(b / f"{v}.h5ad", backed="r")
mpp = float(a.uns["microns_per_pixel"])
cent_all = np.asarray(a.obsm["spatial"], dtype=np.float64)
polys_all = np.asarray(shapely.from_wkb(pd.read_parquet(b / f"{v}_polygons.parquet")["wkb"].to_numpy()), dtype=object)
edges = pd.read_parquet(b / f"{v}_edges_voronoi.parquet")


def geoms(g):
    if g is None or g.is_empty:
        return []
    return list(g.geoms) if hasattr(g, "geoms") else [g]


def draw_edge(ax, row, half_um, annotate):
    i, j = int(row.i), int(row.j)
    mid = 0.5 * (cent_all[i] + cent_all[j])
    d = np.linalg.norm(cent_all - mid, axis=1) * mpp
    sel = np.flatnonzero(d <= 130)
    li, lj = int(np.flatnonzero(sel == i)[0]), int(np.flatnonzero(sel == j)[0])
    g, regs = build_voronoi_graph(polys_all[sel], cent_all[sel], mpp, 30.0, 100.0)
    polys = polys_all[sel]
    others = [np.asarray(p.exterior.coords) for k, p in enumerate(polys) if k not in (li, lj)]
    ax.add_collection(PolyCollection(others, facecolors=S.GHOST, edgecolors="#dedcd6", linewidths=0.3, zorder=1))
    pair = [np.asarray(polys[k].exterior.coords) for k in (li, lj)]
    ax.add_collection(PolyCollection(pair, facecolors=S.CELL_FILL, edgecolors=S.CELL_EDGE, linewidths=0.6, zorder=2))
    for k in (li, lj):
        for gg in geoms(regs[k]):
            ax.plot(*gg.exterior.xy, color=S.REGION_EDGE, lw=0.6, zorder=3, ls=(0, (2.5, 1.5)))
    face = regs[li].boundary.intersection(regs[lj].boundary)
    for seg in geoms(face):
        ax.plot(*seg.xy, color=S.BLUE, lw=2.4, solid_capstyle="round", zorder=6)
    contact_any, contact_segs = False, []
    for k, other in ((li, lj), (lj, li)):
        near = shapely.intersection(polys[k].boundary, shapely.buffer(polys[other], TOL_UM / mpp))
        for seg in geoms(near):
            if hasattr(seg, "xy") and seg.length > 0:
                ax.plot(*seg.xy, color=S.ORANGE, lw=1.8, solid_capstyle="round", zorder=5)
                contact_any = True
                contact_segs.append(seg)
    ci, cj = cent_all[i], cent_all[j]
    ax.plot([ci[0], cj[0]], [ci[1], cj[1]], color=S.INK, lw=0.7, zorder=7)
    ax.scatter([ci[0], cj[0]], [ci[1], cj[1]], s=9, color=S.INK, zorder=8, lw=0)
    h = half_um / mpp
    ax.set_xlim(mid[0] - h, mid[0] + h); ax.set_ylim(mid[1] - h, mid[1] + h)
    S.tissue_axes(ax)
    return face, contact_segs, (ci, cj)


def lab(ax, xy, text, color, dx, dy, ha="left"):
    ax.annotate(text, xy=xy, xytext=(xy[0] + dx, xy[1] + dy), fontsize=7, color=color, ha=ha, va="center",
                zorder=10, arrowprops=dict(arrowstyle="-", color=color, lw=0.5, shrinkA=0.5, shrinkB=1.5))


fig = plt.figure(figsize=(S.TEXTWIDTH, 2.35))
gs = fig.add_gridspec(1, 3, left=0.01, right=0.985, bottom=0.17, top=0.93, wspace=0.10,
                      width_ratios=[1, 1, 1.25])
axa, axb, axc = (fig.add_subplot(gs[0, k]) for k in range(3))
H = 11.0
rows = {"touching": pick_edge(edges, True), "apart": pick_edge(edges, False)}


def unit(v):
    return v / np.linalg.norm(v)


for axi, key, letter in ((axa, "touching", "a"), (axb, "apart", "b")):
    row = rows[key]
    face, contact_segs, (ci, cj) = draw_edge(axi, row, H, True)
    mid = 0.5 * (ci + cj)
    n = unit(np.array([-(cj - ci)[1], (cj - ci)[0]]))          # normal to the centroid line
    fc = np.asarray(face.centroid.coords[0])
    side = np.sign(np.dot(fc - mid, n)) or 1.0
    # d_ij: a third of the way along the centroid line, on the side away from the face
    dp = ci + 0.16 * (cj - ci)
    lab(axi, tuple(dp), r"$d_{ij}$", S.INK, *(-side * n * 2.6 / mpp), ha="center")
    # face: just beyond the face end that lies away from the centroid line
    e0, e1 = (np.asarray(c) for c in (face.coords[0], face.coords[-1])) if face.geom_type == "LineString" else \
        (np.asarray(c) for c in (geoms(face)[0].coords[0], geoms(face)[-1].coords[-1]))
    end_pt = e0 if np.dot(e0 - mid, unit(e0 - e1)) > np.dot(e1 - mid, unit(e1 - e0)) else e1
    u = unit(end_pt - (e1 if end_pt is e0 else e0))
    lab(axi, tuple(end_pt), r"face $f_{ij}$", S.INK, *((u * 1.4 + side * n * 1.2) / mpp))
    if contact_segs:
        longest = max(contact_segs, key=lambda g: g.length)
        cp = np.asarray(longest.interpolate(0.8, normalized=True).coords[0])
        lab(axi, tuple(cp), r"contact", S.INK, *((-side * n * 2.4 + unit(cj - ci) * 0.8) / mpp))
    else:
        axi.text(0.97, 0.93, r"no contact", transform=axi.transAxes, fontsize=7, color=S.INK2,
                 va="center", ha="right", zorder=11)
    S.scalebar(axi, 5, mpp, pad=0.06)
    S.panel_label(axi, letter)

# (c)
res = {ds: noleak_by_fifth(ds, var) for ds, var in SECTIONS}
C = np.array([res[ds]["contact_noleak_pct"] for ds, _ in SECTIONS])
F = np.array([res[ds]["face_noleak_pct"] for ds, _ in SECTIONS])
q = np.arange(1, 6)
for row in C:
    axc.plot(q, row, color=S.ORANGE, lw=0.6, alpha=0.55, zorder=2)
axc.fill_between(q, C.min(0), C.max(0), color=S.ORANGE, alpha=0.12, lw=0, zorder=1)
axc.plot(q, np.median(C, 0), color=S.ORANGE, lw=1.8, marker="o", ms=3.2, zorder=3)
axc.plot(q, F.max(0), color=S.BLUE, lw=1.8, marker="o", ms=3.2, zorder=4)
axc.set_xticks(q, [r"1" + "\n" + r"sparsest", "2", "3", "4", r"5" + "\n" + r"densest"])
axc.set_xlabel(r"local density fifth", labelpad=1.5)
axc.set_ylabel(r"cells with no leakage (\%)", labelpad=1.5)
axc.set_xlim(0.8, 5.2); axc.set_ylim(-1.5, 40); axc.set_yticks([0, 10, 20, 30, 40])
# keys in the empty upper right, each beside its own line sample
for yk, col, text in ((35.0, S.ORANGE, r"contact kernel"), (30.5, S.BLUE, r"Voronoi kernel $\beta$")):
    axc.plot([2.75, 3.15], [yk, yk], color=col, lw=1.8, zorder=5)
    axc.plot([2.95], [yk], color=col, marker="o", ms=3.2, zorder=6)
    axc.text(3.25, yk, text, fontsize=7, color=S.INK, va="center")
S.panel_label(axc, "c", x=-0.16)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=220)
(HERE / "data").mkdir(exist_ok=True)
json.dump({"sections": res, "edges": {k: {c: float(r[c]) for c in ("centroid_dist_um", "shared_wall_um", "wall_dist_um", "apposed_wall_um")}
                                      | {"cell_i": str(r.cell_i), "cell_j": str(r.cell_j)} for k, r in rows.items()}},
          open(HERE / "data" / "fig_contact_kernel.json", "w"), indent=1)
print("wrote", OUT.with_suffix(".pdf"))
for k, r in rows.items():
    print(k, r.cell_i, r.cell_j, f"d {r.centroid_dist_um:.1f} f {r.shared_wall_um:.1f} gap {r.wall_dist_um:.2f} apposed {r.apposed_wall_um:.2f}")
print("contact no-leak % by fifth, per section:\n", np.round(C, 1), "\nface:", np.round(F, 2))
