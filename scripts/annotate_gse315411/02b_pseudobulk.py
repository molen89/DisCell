#!/usr/bin/env python
"""
Stage 2b — cluster naming by pseudobulk correlation (replacement for scANVI).

Why: at ~100 transcripts per cell the scArches classifier collapses onto one
default class (HLCA leg: 60 % "Alveolar fibroblasts", 26 of 44 clusters incl.
macrophages, T cells, neutrophils, cartilage) with max-probability ~1.0 -- a
confident wrong label the stage-3 gates cannot see. Summing a Leiden cluster's
counts gives a profile with 10^5-10^7 transcripts, where the depth problem is
gone, and a correlation to reference type centroids is transparent.

Per reference: per-type mean of log1p(CP10k) over the shared genes (read in
row chunks, no full load); per query cluster: log1p(CP10k) of the summed
counts; Spearman r over the top-N reference-variable shared genes. Writes
pseudobulk_<tag>.csv with best/second type, r, margin, and the reference's
lineage of the best type; stage 3b combines references and applies the gate.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed


def reference_profiles(path, label_key, gene_col, genes, lineage_keys, chunk=50_000):
    """Per-type summed raw counts over *genes*, reading .raw/X in row chunks."""
    import h5py
    import numpy as np
    import pandas as pd
    import anndata as ad
    from scipy import sparse

    rd = ad.io.read_elem if hasattr(ad, "io") else ad.experimental.read_elem
    with h5py.File(path) as h:
        grp = h["raw"] if "raw" in h else h
        obs = rd(h["obs"])
        var = rd(grp["var"])
        symbols = var[gene_col].astype(str).to_numpy() if gene_col else var.index.to_numpy()
        col = pd.Series(np.arange(len(symbols)), index=symbols)
        col = col[~col.index.duplicated()]
        keep = col.reindex(genes).dropna().astype(int)
        labels = obs[label_key].astype(str).to_numpy()
        types = np.array(sorted(set(labels)))
        t_index = pd.Series(np.arange(len(types)), index=types)[labels].to_numpy()
        sums = np.zeros((len(types), len(keep)), dtype=np.float64)
        n = np.bincount(t_index, minlength=len(types))
        X = grp["X"]
        indptr = X["indptr"][:]
        for start in range(0, len(labels), chunk):
            stop = min(start + chunk, len(labels))
            a, b = indptr[start], indptr[stop]
            m = sparse.csr_matrix((X["data"][a:b], X["indices"][a:b], indptr[start:stop + 1] - a),
                                  shape=(stop - start, len(symbols)))[:, keep.to_numpy()]
            onehot = sparse.csr_matrix((np.ones(stop - start), (t_index[start:stop],
                                                                np.arange(stop - start))),
                                       shape=(len(types), stop - start))
            sums += np.asarray((onehot @ m).todense())
            print(f"    {stop:>8,d}/{len(labels):,d} reference cells", end="\r")
        print()
        lineage = {}
        for k in lineage_keys:
            if k in obs:
                lineage[k] = obs.groupby(labels, observed=True)[k].agg(
                    lambda s: s.astype(str).value_counts().index[0])
    return types, n, keep.index.to_numpy(), sums, lineage


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--ref", required=True)
    p.add_argument("--ref-label-key", required=True)
    p.add_argument("--ref-gene-col", default="feature_name")
    p.add_argument("--ref-lineage-keys", nargs="*", default=[],
                   help="obs columns giving the coarse lineage of each reference type")
    p.add_argument("--tag", required=True)
    p.add_argument("--min-cells-per-type", type=int, default=50)
    p.add_argument("--n-genes", type=int, default=1500,
                   help="top reference-variable shared genes used for the correlation")
    args = p.parse_args()
    pin_gpu(-1)
    set_seed(args.seed)
    root, work = resolve_work(args)

    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy import sparse
    from scipy.stats import spearmanr

    query = sc.read_h5ad(work / "query_emb.h5ad")
    x = sparse.csr_matrix(query.layers["counts"])
    genes = np.asarray(query.var_names)
    clusters = query.obs["leiden"].astype(str).to_numpy()
    names = sorted(set(clusters), key=lambda s: int(s) if s.isdigit() else s)
    codes = pd.Categorical(clusters, names).codes
    onehot = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                               shape=(len(names), len(codes)))
    q_sums = np.asarray((onehot @ x).todense())
    q_n = np.asarray(onehot.sum(axis=1)).ravel().astype(int)
    depth = np.asarray(x.sum(axis=1)).ravel()
    q_depth = np.array([np.median(depth[codes == i]) for i in range(len(names))])
    print(f"query: {len(names)} clusters, pseudobulk depth "
          f"{q_sums.sum(axis=1).min():,.0f}-{q_sums.sum(axis=1).max():,.0f} transcripts")

    print(f"reference {args.ref}")
    types, n, shared, r_sums, lineage = reference_profiles(
        args.ref, args.ref_label_key, args.ref_gene_col, genes, args.ref_lineage_keys)
    keep = n >= args.min_cells_per_type
    types, n, r_sums = types[keep], n[keep], r_sums[keep]
    print(f"  {len(types)} types (>= {args.min_cells_per_type} cells), {len(shared)} shared genes")

    def logcp10k(s):
        return np.log1p(1e4 * s / s.sum(axis=1, keepdims=True))

    r_prof = logcp10k(r_sums)
    q_prof = logcp10k(q_sums[:, pd.Series(np.arange(len(genes)), index=genes)[shared].to_numpy()])
    top = np.argsort(-r_prof.var(axis=0))[:args.n_genes]
    print(f"  correlating on the {len(top)} most type-variable reference genes")

    rows = []
    for i, name in enumerate(names):
        r = np.array([spearmanr(q_prof[i, top], r_prof[j, top]).correlation
                      for j in range(len(types))])
        order = np.argsort(-r)
        b, s = order[0], order[1]
        row = dict(cluster=name, n_cells=q_n[i], median_depth=q_depth[i],
                   best=types[b], r_best=r[b], second=types[s], r_second=r[s],
                   margin=r[b] - r[s], third=types[order[2]], r_third=r[order[2]])
        for k, m in lineage.items():
            row[f"{k}_best"] = m.get(types[b], "")
            row[f"{k}_second"] = m.get(types[s], "")
        rows.append(row)
        print(f"{name:>3} n={q_n[i]:7d} depth={q_depth[i]:4.0f}  {types[b]:35s} r={r[b]:.3f}  "
              f"| {types[s]:30s} r={r[s]:.3f}  margin={r[b]-r[s]:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(work / f"pseudobulk_{args.tag}.csv", index=False)
    print(f"\nwrote {work / f'pseudobulk_{args.tag}.csv'}")


if __name__ == "__main__":
    main()
