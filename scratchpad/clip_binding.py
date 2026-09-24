"""Slide-wide: how often does the 30 um clip actually change anything?"""
import numpy as np, pandas as pd, anndata as ad
from shapely.geometry import MultiPoint, Point
from shapely.ops import voronoi_diagram
from shapely.strtree import STRtree

BUNDLE = "data/datasets/gse315411_pdltma06_10_prime_dual/bundle"
NAME = "pdl018d"
CLIP_UM = 30.0

a = ad.read_h5ad(f"{BUNDLE}/{NAME}.h5ad", backed="r")
mpp = float(a.uns["microns_per_pixel"])
cent = np.asarray(a.obsm["spatial"], dtype=np.float64)
print(f"{len(cent)} cells, mpp {mpp}")

pts = MultiPoint([Point(x, y) for x, y in cent])
regions = list(voronoi_diagram(pts, tolerance=0.0).geoms)
tree = STRtree(regions)
owner = [None] * len(cent)
for k, (x, y) in enumerate(cent):
    probe = Point(x, y)
    for hit in tree.query(probe):
        if regions[hit].contains(probe):
            owner[k] = regions[hit]
            break
print("unmatched:", sum(o is None for o in owner))

r_px = CLIP_UM / mpp
# a region is affected iff any of its vertices lies beyond r from the seed
far = np.zeros(len(cent), dtype=bool)
maxr = np.full(len(cent), np.nan)
for k, g in enumerate(owner):
    if g is None:
        continue
    xy = np.asarray(g.exterior.coords)
    d = np.linalg.norm(xy - cent[k], axis=1)
    maxr[k] = d.max() * mpp
    far[k] = d.max() > r_px

ok = ~np.isnan(maxr)
print(f"\nmax vertex distance from seed, um: median {np.nanmedian(maxr):.1f}, "
      f"p90 {np.nanpercentile(maxr[ok],90):.1f}, "
      f"p99 {np.nanpercentile(maxr[ok],99):.1f}, max {np.nanmax(maxr):.0f}")
print(f"cells whose Voronoi region the {CLIP_UM:g} um clip touches at all: "
      f"{far.sum()} / {ok.sum()} = {100*far[ok].mean():.2f}%")

# area actually removed, for those it touches
import shapely
cut = []
for k in np.flatnonzero(far):
    g = owner[k]
    c = g.intersection(Point(*cent[k]).buffer(r_px))
    cut.append(1 - c.area / g.area)
cut = np.asarray(cut)
if len(cut):
    print(f"for those, area removed: median {100*np.median(cut):.1f}%, "
          f"p90 {100*np.percentile(cut,90):.1f}%, max {100*cut.max():.1f}%")
