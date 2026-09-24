"""Slide-wide numbers straight from the stored bundle edge tables."""
import numpy as np, pandas as pd
from discell import paths

for ds_id in ["gse315411_pdltma06_10_prime_dual", "xenium_prime_ovarian_cancer_ffpe"]:
    ds = paths.dataset(ds_id)
    name = "pdl018d" if "gse" in ds_id else "full"
    e = pd.read_parquet(ds.bundle_dir / f"{name}_edges_voronoi.parquet")
    f, d = e.shared_wall_um.to_numpy(), e.centroid_dist_um.to_numpy()
    gap, ap = e.wall_dist_um.to_numpy(), e.apposed_wall_um.to_numpy()
    print(f"\n=== {ds_id} / {name}: {len(e)} voronoi edges ===")
    print(f"face f_ij      : median {np.median(f):6.2f} um, "
          f"zero on {100*(f<=0).mean():.3f}% of edges, max {f.max():.1f}")
    print(f"centroid d_ij  : median {np.median(d):6.2f} um")
    print(f"polygon gap    : {100*(gap<=0).mean():5.1f}% of pairs actually touch, "
          f"median gap {np.median(gap):.2f} um")
    print(f"apposed wall   : zero on {100*(ap<=0).mean():5.1f}% of edges "
          f"(so a contact-based f would vanish on that many)")
    print(f"corr(face, gap)     = {np.corrcoef(f, gap)[0,1]:+.3f}")
    print(f"corr(apposed, gap)  = {np.corrcoef(ap, gap)[0,1]:+.3f}")
    print(f"corr(face, d_ij)    = {np.corrcoef(f, d)[0,1]:+.3f}")
    # does the 30um clip show up as a ceiling on the face?
    print(f"faces > 30 um: {100*(f>30).mean():.2f}%   faces > 60 um: {100*(f>60).mean():.3f}%")
