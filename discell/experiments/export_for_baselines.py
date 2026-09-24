"""Export one dataset variant as an .h5ad for the external baselines.

The baselines (resolVI, SIMVI, MintFlow) live in their own environments under
``/home/rmolen/github/DisCell-baselines``; the fairness protocol asks that they
see *our* tile split and *our* pruned Delaunay graph rather than rebuilding
their own. This module writes exactly that: counts, coordinates, the label
column that becomes ``t``, the tile ids and train/val flag, log depth, and the
pruned graph as two ``obsp`` matrices.

Everything comes from :func:`discell.model.prepare.assemble`, so the split and
the graph are bit-identical to a DisCell fit with the same seed and config.

    python -m discell.experiments.export_for_baselines \
        --dataset gse315411_pdltma06_11_prime_solo --variant pdl018d \
        --window 5000
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)


def _adjacency(graph) -> sp.csr_matrix:
    """Symmetric binary adjacency of the pruned undirected graph."""
    n = graph.n_cells
    i = np.concatenate([graph.edge_i, graph.edge_j])
    j = np.concatenate([graph.edge_j, graph.edge_i])
    return sp.csr_matrix((np.ones(len(i), dtype=np.float32), (i, j)),
                         shape=(n, n))


def build(data, dataset: str, variant: str, label_key: str | None = None):
    """An :class:`anndata.AnnData` carrying the DisCell split and graph."""
    import anndata as ad
    import pandas as pd

    n = data.graph.n_cells
    tile = np.full(n, -1, dtype=np.int64)
    split = np.empty(n, dtype=object)
    for k, cells in enumerate(data.train_tiles):
        tile[cells] = k
        split[cells] = "train"
    for k, cells in enumerate(data.val_tiles, start=len(data.train_tiles)):
        tile[cells] = k
        split[cells] = "val"
    assert (tile >= 0).all(), "tiles must partition the cells exactly once"

    obs = pd.DataFrame({
        "cell_type": pd.Categorical(np.asarray(data.type_names)[data.t]),
        "cell_type_index": data.t,
        "tile": tile,
        "split": pd.Categorical(split.astype(str)),
        "total_counts": data.totals.astype(np.float32),
        "log_depth": np.log(np.maximum(data.totals, 1.0)).astype(np.float32),
        "degree": data.graph.degrees.astype(np.int32),
        # MintFlow wants a slice / biological-batch key; one slide = one value
        "slice_id": pd.Categorical([f"{dataset}_{variant}"] * n),
        "x_um": data.positions[:, 0].astype(np.float32),
        "y_um": data.positions[:, 1].astype(np.float32),
    }, index=pd.Index([f"cell_{i}" for i in range(n)], name="cell"))

    # the cycle target the DisCell battery reads, carried along so a baseline
    # environment can see it without rescoring (the battery itself rescores
    # from ``assemble``; these columns are for inspection and for tools that
    # want a continuous covariate)
    if data.cycle is not None:
        obs["s_score"] = np.asarray(data.cycle["s_score"], dtype=np.float32)
        obs["g2m_score"] = np.asarray(data.cycle["g2m_score"], dtype=np.float32)

    counts = sp.csr_matrix(data.x)
    adata = ad.AnnData(X=counts, obs=obs)
    adata.var_names = pd.Index(np.asarray(data.gene_names).astype(str))
    adata.layers["counts"] = counts.copy()
    adata.obsm["spatial"] = data.positions.astype(np.float32)
    # the pruned graph, in the two forms a baseline may want: a binary
    # connectivity matrix, and our directed smoothing weights (rows = dst)
    adata.obsp["discell_connectivities"] = _adjacency(data.graph)
    adata.obsp["discell_beta"] = sp.csr_matrix(data.graph.in_edges,
                                               dtype=np.float32)
    adata.uns["discell"] = {
        "dataset": dataset, "variant": variant,
        "label_key": label_key or "bundle_default",
        "n_train_tiles": len(data.train_tiles),
        "n_val_tiles": len(data.val_tiles),
        "type_names": np.asarray(data.type_names).astype(str),
        "median_counts": data.median_counts,
        "cycling_types": np.asarray(
            (data.cycle or {}).get("cycling_types", []), dtype=np.int64),
    }
    return adata


def window(adata, n_cells: int):
    """The *n_cells* cells nearest the densest point: one contiguous disc."""
    from scipy.spatial import cKDTree

    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    tree = cKDTree(xy)
    # densest point = the cell whose 50th neighbour is closest
    k = min(50, len(xy))
    d, _ = tree.query(xy, k=k)
    centre = xy[int(np.argmin(d[:, -1]))]
    _, rows = tree.query(centre, k=min(n_cells, len(xy)))
    rows = np.sort(np.atleast_1d(rows))
    sub = adata[rows].copy()
    log.info("Window: %d cells, %.0f x %.0f um, %d train / %d val",
             sub.n_obs, np.ptp(sub.obsm["spatial"][:, 0]),
             np.ptp(sub.obsm["spatial"][:, 1]),
             int((sub.obs["split"] == "train").sum()),
             int((sub.obs["split"] == "val").sum()))
    return sub


def main(argv=None) -> None:
    from discell import paths
    from discell.model.prepare import assemble

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--variant", default="full")
    p.add_argument("--embeddings", default="egomask_ego_v1")
    p.add_argument("--label-key", default=None)
    p.add_argument("--tile-cells", type=int, default=4096)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=None,
                   help="also write a contiguous window of this many cells")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ds = paths.dataset(args.dataset)
    out_dir = (ds.root / "baselines") if args.out_dir is None else \
        __import__("pathlib").Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = assemble(args.dataset, args.variant, args.embeddings,
                    tile_cells=args.tile_cells,
                    val_fraction=args.val_fraction, seed=args.seed,
                    label_key=args.label_key)
    adata = build(data, args.dataset, args.variant, args.label_key)
    path = out_dir / f"{args.variant}_baselines.h5ad"
    adata.write_h5ad(path)
    log.info("Wrote %s (%d cells, %d genes)", path, adata.n_obs, adata.n_vars)

    if args.window:
        sub = window(adata, args.window)
        wpath = out_dir / f"{args.variant}_window{args.window}.h5ad"
        sub.write_h5ad(wpath)
        log.info("Wrote %s (%d cells)", wpath, sub.n_obs)


if __name__ == "__main__":
    main()
