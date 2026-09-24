"""What the 30 um clip implies for sparse neighbourhoods, checked on stored edges.

If the face is the bisector chord at distance d/2 from each seed, and each
region is clipped to a disc of radius R, then an edge can only survive while
d/2 < R, and its face cannot exceed the chord of that disc:

    d_ij < 2R = 60 um        and        f_ij <= 2*sqrt(R^2 - (d_ij/2)^2)
"""
import numpy as np, pandas as pd
from discell import paths

R = 30.0
for ds_id, name in [("gse315411_pdltma06_10_prime_dual", "pdl018d"),
                    ("xenium_prime_ovarian_cancer_ffpe", "full"),
                    ("xenium_prime_human_lung_cancer_ffpe", "full")]:
    e = pd.read_parquet(paths.dataset(ds_id).bundle_dir / f"{name}_edges_voronoi.parquet")
    f, d = e.shared_wall_um.to_numpy(), e.centroid_dist_um.to_numpy()
    ceiling = 2 * np.sqrt(np.maximum(R**2 - (d / 2) ** 2, 0))
    print(f"\n=== {ds_id} ({len(e)} edges) ===")
    print(f"  max d_ij = {d.max():.2f} um   (predicted hard limit 2R = {2*R:.0f})")
    print(f"  edges with d_ij >= 60 um: {(d >= 60).sum()}")
    over = f > ceiling + 1e-6
    print(f"  faces above the chord ceiling: {over.sum()} "
          f"({100*over.mean():.3f}%), worst excess {np.max(f - ceiling):.3f} um")
    # where does the ceiling actually bite?
    tight = f > 0.9 * ceiling
    print(f"  edges within 10% of the ceiling: {100*tight.mean():.2f}%  "
          f"(these are the sparse/border ones)")
    print(f"  d_ij > 30 um: {100*(d > 30).mean():.2f}% of edges")
