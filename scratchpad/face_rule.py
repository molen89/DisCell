"""Check the claim: f_ij = distance between the circumcentres of the two
Delaunay triangles sharing edge ij. That is what sets the face width."""
import numpy as np, pandas as pd, anndata as ad
from collections import defaultdict
from scipy.spatial import Delaunay
from discell.preprocess.geometry import build_voronoi_graph

BUNDLE = "data/datasets/gse315411_pdltma06_10_prime_dual/bundle"
NAME, CLIP_UM = "pdl018d", 30.0

a = ad.read_h5ad(f"{BUNDLE}/{NAME}.h5ad", backed="r")
mpp = float(a.uns["microns_per_pixel"])
cent_all = np.asarray(a.obsm["spatial"], dtype=np.float64)
polys_all = np.array(
    list(__import__("shapely").from_wkb(
        pd.read_parquet(f"{BUNDLE}/{NAME}_polygons.parquet")["wkb"].to_numpy())),
    dtype=object)

# interior patch, so the outer ring of unbounded regions cannot contaminate
seed = int(np.argsort(np.linalg.norm(cent_all - cent_all.mean(0), axis=1))[0])
d = np.linalg.norm(cent_all - cent_all[seed], axis=1) * mpp
sel = np.flatnonzero(d <= 220.0)
cent = cent_all[sel]
graph, _ = build_voronoi_graph(polys_all[sel], cent, mpp, None, 100.0)  # unclipped


def circumcentre(p, q, r):
    ax, ay = p; bx, by = q; cx, cy = r
    dd = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
          + (cx**2 + cy**2) * (ay - by)) / dd
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
          + (cx**2 + cy**2) * (bx - ax)) / dd
    return np.array([ux, uy])


seed_local = int(np.flatnonzero(sel == seed)[0])
tri = Delaunay(cent)
edge_tris = defaultdict(list)
for s in tri.simplices:
    cc = circumcentre(cent[s[0]], cent[s[1]], cent[s[2]])
    for u, v in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
        edge_tris[(min(u, v), max(u, v))].append(cc)

pred, obs = [], []
for i, j, w in zip(graph.i, graph.j, graph.shared_wall_um):
    ccs = edge_tris.get((int(i), int(j)))
    if ccs is None or len(ccs) != 2:       # hull edges have one triangle
        continue
    pred.append(np.linalg.norm(ccs[0] - ccs[1]) * mpp)
    obs.append(w)
pred, obs = np.array(pred), np.array(obs)
err = np.abs(pred - obs)
print(f"{len(pred)} interior edges with two adjacent Delaunay triangles")
print(f"predicted vs stored face: max abs err {err.max():.6f} um, "
      f"median {np.median(err):.2e} um, corr {np.corrcoef(pred, obs)[0,1]:.6f}")

# what actually drives the width: how close the flanking cells sit to the ij line
flank, faces, dij = [], [], []
for i, j, w in zip(graph.i, graph.j, graph.shared_wall_um):
    ccs = edge_tris.get((int(i), int(j)))
    if ccs is None or len(ccs) != 2:
        continue
    mid = 0.5 * (cent[int(i)] + cent[int(j)])
    # mean distance from the pair's midpoint to the two flanking circumcentres
    flank.append(np.mean([np.linalg.norm(c - mid) for c in ccs]) * mpp)
    faces.append(w)
    dij.append(np.linalg.norm(cent[int(i)] - cent[int(j)]) * mpp)
flank, faces, dij = np.array(flank), np.array(faces), np.array(dij)
print(f"\ncorr(face, centroid distance d_ij)        = {np.corrcoef(faces, dij)[0,1]:+.3f}")
print(f"corr(face, flank circumcentre offset)    = {np.corrcoef(faces, flank)[0,1]:+.3f}")

# how many edges violate the rule, and where do they sit?
bad = err > 1e-6
print(f"\nedges failing the rule: {bad.sum()} / {len(err)} = {100*bad.mean():.2f}%")
if bad.any():
    print(f"  their err quantiles (um): "
          f"{np.round(np.quantile(err[bad], [0.5, 0.9, 1.0]), 2)}")
    # distance of the failing edges from the patch centre, vs all edges
    mids = []
    for i, j, w in zip(graph.i, graph.j, graph.shared_wall_um):
        ccs = edge_tris.get((int(i), int(j)))
        if ccs is None or len(ccs) != 2:
            continue
        mids.append(np.linalg.norm(0.5*(cent[int(i)]+cent[int(j)]) - cent[seed_local]) * mpp)
    mids = np.array(mids)
    print(f"  radius from patch centre, um: failing median {np.median(mids[bad]):.0f}, "
          f"passing median {np.median(mids[~bad]):.0f}, patch radius 220")
