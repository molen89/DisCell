#!/usr/bin/env python
"""
Diagnostic — subcluster mixed Leiden clusters on the scVI latent and print
their markers, the way an annotator splits a cluster that mixes lineages.
Writes subclusters.csv (cell, cluster, subcluster) for the clusters given.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--clusters", nargs="+", required=True)
    p.add_argument("--resolution", type=float, default=0.3)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()
    pin_gpu(-1)
    set_seed(args.seed)
    root, work = resolve_work(args)

    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy import sparse

    adata = sc.read_h5ad(work / "query_emb.h5ad")
    x = sparse.csr_matrix(adata.layers["counts"]); genes = np.asarray(adata.var_names)
    depth = np.asarray(x.sum(axis=1)).ravel().astype(float); depth[depth == 0] = 1
    xn = sparse.diags(100.0 / depth) @ x
    gmean = np.asarray(xn.mean(axis=0)).ravel()
    cl = adata.obs["leiden"].astype(str).to_numpy()
    donor = adata.obs["donor"].astype(str).to_numpy()
    canon = [g for g in "PECAM1 CDH5 CLDN5 HPGD APLN CA4 EDNRB SOX17 GJA5 TCF21 PDGFRA PDGFRB WNT2 AGER HOPX EPCAM ABCA3 CD3E CD8A CD4 KLRD1 GZMB MS4A1 CD79A MZB1 CD68 HSPA6 SERPINE1".split() if g in genes]
    gi = {g: np.where(genes == g)[0][0] for g in canon}
    out = []
    for k in args.clusters:
        m = np.flatnonzero(cl == k)
        sub = adata[m].copy()
        sc.pp.neighbors(sub, use_rep="X_scVI", n_neighbors=15)
        sc.tl.leiden(sub, resolution=args.resolution, key_added="sub", flavor="igraph",
                     n_iterations=2, directed=False)
        labels = sub.obs["sub"].astype(str).to_numpy()
        print(f"\n===== cluster {k}: {len(m)} cells -> {len(set(labels))} subclusters at resolution {args.resolution}")
        for s in sorted(set(labels), key=int):
            idx = m[labels == s]
            mu = np.asarray(xn[idx].mean(axis=0)).ravel()
            lfc = np.log2((mu + 1e-3) / (gmean + 1e-3)); lfc[mu < 0.02] = -np.inf
            top = np.argsort(-lfc)[:args.top]
            dd = pd.Series(donor[idx]).value_counts(normalize=True).head(2)
            print(f"  {k}.{s} n={len(idx):6d} depth={np.median(depth[idx]):4.0f} donors: "
                  + ", ".join(f"{a} {b:.0%}" for a, b in dd.items()))
            print("      fc: " + " ".join(f"{genes[g]}({lfc[g]:.1f})" for g in top))
            print("      canon: " + " ".join(f"{g}={mu[gi[g]]:.2f}" for g in canon if mu[gi[g]] >= 0.05))
            out.append(pd.DataFrame({"cell": adata.obs_names[idx], "cluster": k, "subcluster": f"{k}.{s}"}))
    pd.concat(out).to_csv(work / "subclusters.csv", index=False)
    print(f"\nwrote {work / 'subclusters.csv'}")


if __name__ == "__main__":
    main()
