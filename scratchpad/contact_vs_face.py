"""Review R11: a contact-based leakage kernel vs the Voronoi-face kernel beta.

Both kernels are built on the model's edge set (Voronoi adjacency, pruned at
40 um), with the same distance decay exp(-d/tau) and the same row
normalisation; only the edge weight differs (apposed wall at 1 um tolerance vs
Voronoi face). Reads are per section. Analysis only; nothing is fitted.
"""
import json, sys
import anndata as ad, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

TAU, PRUNE, R_DENS = 20.0, 40.0, 20.0
SECTIONS = [
    ("Ovarian cancer (FFPE)", "xenium_prime_ovarian_cancer_ffpe", "full"),
    ("Lung cancer (FFPE)", "xenium_prime_human_lung_cancer_ffpe", "full"),
    ("Ovary (fresh frozen)", "xenium_prime_human_ovary_ff", "full"),
    ("Lung TMA, section 1", "gse315411_pdltma06_11_prime_solo", "full"),
    ("Lung TMA, section 2", "gse315411_pdltma06_10_prime_dual", "full"),
]
rng = np.random.default_rng(0)
out = []
for label, ds, var in SECTIONS:
    b = f"data/datasets/{ds}/bundle"
    a = ad.read_h5ad(f"{b}/{var}.h5ad", backed="r")
    xy = np.asarray(a.obsm["spatial_um"], dtype=np.float64)
    seg = a.obs["xenium_segmentation_method"].astype(str).value_counts(normalize=True)
    e = pd.read_parquet(f"{b}/{var}_edges_voronoi.parquet",
                        columns=["i", "j", "shared_wall_um", "centroid_dist_um",
                                 "wall_dist_um", "apposed_wall_um"])
    e = e[e.centroid_dist_um <= PRUNE]
    n = a.n_obs
    dens = cKDTree(xy).query_ball_point(xy, r=R_DENS, return_length=True) - 1

    # undirected edge facts
    touch = float((e.wall_dist_um <= 0).mean())
    gap_med = float(e.wall_dist_um.median())
    zero_contact_e = float((e.apposed_wall_um <= 0).mean())
    zero_face_e = float((e.shared_wall_um <= 0).mean())
    s = rng.choice(len(e), min(len(e), 1_000_000), replace=False)
    rho_contact_gap = spearmanr(e.apposed_wall_um.values[s], e.wall_dist_um.values[s])[0]

    # directed rows: cell i receives from neighbour j, both directions
    recv = np.concatenate([e.i.values, e.j.values])
    face = np.tile(e.shared_wall_um.values, 2)
    cont = np.tile(e.apposed_wall_um.values, 2)
    dist = np.tile(e.centroid_dist_um.values, 2)
    gap = np.tile(e.wall_dist_um.values, 2)
    decay = np.exp(-dist / TAU)
    beta_raw, c_raw = face * decay, cont * decay
    beta_row = np.bincount(recv, beta_raw, minlength=n)
    c_row = np.bincount(recv, c_raw, minlength=n)
    connected = np.bincount(recv, minlength=n) > 0
    beta = beta_raw / beta_row[recv]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(c_row[recv] > 0, c_raw / c_row[recv], np.nan)

    zl_c = (c_row <= 0) & connected
    zl_b = (beta_row <= 0) & connected
    dc = dens[connected]
    q_lo, q_hi = np.quantile(dc, [0.2, 0.8])
    sparse = connected & (dens <= q_lo)
    dense_ = connected & (dens >= q_hi)
    ok = ~np.isnan(c)
    s2 = rng.choice(len(recv), min(len(recv), 1_000_000), replace=False)
    s2c = s2[ok[s2]]
    row = {
        "section": label, "cells": n,
        "seg": {k: round(float(v), 4) for k, v in seg.items()},
        "touch_frac": touch, "gap_median_um": gap_med,
        "zero_contact_edges": zero_contact_e, "zero_face_edges": zero_face_e,
        "rho_contact_gap": float(rho_contact_gap),
        "noleak_contact": float(zl_c.sum() / connected.sum()),
        "noleak_face": float(zl_b.sum() / connected.sum()),
        "noleak_contact_sparse": float(zl_c[sparse].mean()),
        "noleak_contact_dense": float(zl_c[dense_].mean()),
        "noleak_face_sparse": float(zl_b[sparse].mean()),
        "noleak_face_dense": float(zl_b[dense_].mean()),
        "rho_share_gap_contact": float(spearmanr(c[s2c], gap[s2c])[0]),
        "rho_share_gap_face": float(spearmanr(beta[s2], gap[s2])[0]),
        "density_q20_q80": [float(q_lo), float(q_hi)],
    }
    out.append(row)
    print(json.dumps(row), flush=True)
json.dump(out, open("scratchpad/contact_vs_face.json", "w"), indent=1)
