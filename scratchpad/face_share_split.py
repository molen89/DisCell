"""Split rho(beta share, gap) into its face part and its distance-decay part."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
TAU, PRUNE = 20.0, 40.0
rng = np.random.default_rng(0); res = {}
for ds in ["xenium_prime_ovarian_cancer_ffpe", "xenium_prime_human_lung_cancer_ffpe", "xenium_prime_human_ovary_ff",
           "gse315411_pdltma06_11_prime_solo", "gse315411_pdltma06_10_prime_dual"]:
    e = pd.read_parquet(f"data/datasets/{ds}/bundle/full_edges_voronoi.parquet",
                        columns=["i", "j", "shared_wall_um", "centroid_dist_um", "wall_dist_um"])
    e = e[e.centroid_dist_um <= PRUNE]
    n = int(max(e.i.max(), e.j.max())) + 1
    recv = np.concatenate([e.i.values, e.j.values])
    f = np.tile(e.shared_wall_um.values, 2); d = np.tile(e.centroid_dist_um.values, 2)
    g = np.tile(e.wall_dist_um.values, 2); dec = np.exp(-d / TAU)
    def share(w): return w / np.bincount(recv, w, minlength=n)[recv]
    s = rng.choice(len(recv), min(len(recv), 1_000_000), replace=False)
    res[ds] = {k: float(spearmanr(v[s], g[s])[0]) for k, v in
               {"beta": share(f * dec), "face_only": share(f), "decay_only": share(dec)}.items()}
    print(ds[:34], {k: round(v, 3) for k, v in res[ds].items()}, flush=True)
json.dump(res, open("scratchpad/face_share_split.json", "w"), indent=1)
