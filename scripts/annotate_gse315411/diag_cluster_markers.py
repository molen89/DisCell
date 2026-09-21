#!/usr/bin/env python
"""
Diagnostic — top marker genes per Leiden cluster, from the query counts alone.

Independent of any reference: per-cluster mean of depth-normalised counts,
log2 fold change against the mean of all other clusters, restricted to genes
with a minimum in-cluster mean so a 1-count gene cannot top the list. Prints
one line per cluster and writes cluster_markers.csv; use it to sanity-check the
names stage 3 assigns (a "fibroblast" cluster whose markers are PTPRC/CD68 is
not a fibroblast cluster).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-mean", type=float, default=0.02,
                   help="min in-cluster mean (counts per 100 transcripts)")
    p.add_argument("--labels", nargs="*", default=[],
                   help="obs columns from query_labelled.h5ad to print next to each cluster")
    args = p.parse_args()
    pin_gpu(-1)
    set_seed(args.seed)
    root, work = resolve_work(args)

    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy import sparse

    src = work / ("query_labelled.h5ad" if args.labels and (work / "query_labelled.h5ad").exists()
                  else "query_emb.h5ad")
    adata = sc.read_h5ad(src)
    x = adata.layers["counts"] if "counts" in adata.layers else adata.X
    x = sparse.csr_matrix(x)
    depth = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    depth[depth == 0] = 1.0
    xn = sparse.diags(100.0 / depth) @ x                       # counts per 100 transcripts

    clusters = adata.obs["leiden"].astype(str).to_numpy()
    names = sorted(set(clusters), key=lambda s: int(s) if s.isdigit() else s)
    onehot = sparse.csr_matrix((np.ones(len(clusters)), (pd.Categorical(clusters, names).codes,
                                                          np.arange(len(clusters)))),
                               shape=(len(names), len(clusters)))
    sizes = np.asarray(onehot.sum(axis=1)).ravel()
    means = np.asarray((onehot @ xn).todense()) / sizes[:, None]   # cluster x gene
    total = (means * sizes[:, None]).sum(axis=0)
    genes = np.asarray(adata.var_names)

    rows = []
    for i, name in enumerate(names):
        rest = (total - means[i] * sizes[i]) / (sizes.sum() - sizes[i])
        lfc = np.log2((means[i] + 1e-3) / (rest + 1e-3))
        lfc[means[i] < args.min_mean] = -np.inf
        top = np.argsort(-lfc)[:args.top]
        extra = {c: adata.obs.loc[clusters == name, c].astype(str).value_counts().index[0]
                 for c in args.labels if c in adata.obs}
        rows.append(dict(cluster=name, n_cells=int(sizes[i]),
                         median_depth=float(np.median(depth[clusters == name])),
                         markers=" ".join(f"{genes[g]}({lfc[g]:.1f})" for g in top), **extra))
        print(f"{name:>3} n={int(sizes[i]):7d} depth={rows[-1]['median_depth']:5.0f}  "
              + "  ".join(f"{k}={v}" for k, v in extra.items()) + "\n     " + rows[-1]["markers"])
    pd.DataFrame(rows).to_csv(work / "cluster_markers.csv", index=False)
    print(f"\nwrote {work / 'cluster_markers.csv'}")


if __name__ == "__main__":
    main()
