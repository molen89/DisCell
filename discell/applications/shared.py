#!/usr/bin/env python3
"""Shared data dependencies for the doc-11 applications, built once.

Two extraction stages, both writing under ``data/datasets/<id>/qc/``:

- ``dapi``: integrated nuclear DAPI intensity + nuclear area per cell,
  from the morphology image x the nucleus boundary polygons — the
  orthogonal-physics ground truth for A4 (S/G2M cells carry ~2x DNA).
- ``nuclear-counts``: the counts matrix recomputed from nucleus-flagged
  transcripts only (Prime per-transcript ``overlaps_nucleus``) — the
  segmentation-perturbation arm of A4/A1: signal that vanishes when the
  cytoplasmic expansion is dropped travels with the boundary, i.e. is a
  segmentation/leak artifact.
- ``nuclear-summary``: the nuclear share of each cell's counts from the
  two matrices above — overall, per segmentation method, per type — as
  ``qc/nuclear_summary.json``.

Usage::

    python -m discell.applications.shared --dataset <id> --stage dapi
    python -m discell.applications.shared --dataset <id> --stage nuclear-counts
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from discell import paths

log = logging.getLogger("discell.applications.shared")


def _bundle_meta(dataset: str) -> tuple[Path, float, np.ndarray, np.ndarray]:
    """(xenium raw dir, microns/px, cell_ids, gene names) from the bundle."""
    import anndata as ad

    ds = paths.dataset(dataset)
    adata = ad.read_h5ad(ds.root / "bundle" / "full.h5ad", backed="r")
    xenium_dir = Path(str(adata.uns["xenium_dir"]))
    mpp = float(adata.uns.get("microns_per_pixel", 0.2125))
    return (xenium_dir, mpp, adata.obs_names.to_numpy(),
            adata.var_names.to_numpy())


def extract_nuclear_dapi(dataset: str) -> Path:
    """Per-cell integrated DAPI + nuclear area from image x nuclear mask."""
    import pandas as pd
    import tifffile
    import zarr
    from matplotlib.path import Path as MplPath

    xenium_dir, mpp, cell_ids, _ = _bundle_meta(dataset)
    dapi_path = xenium_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
    store = tifffile.imread(dapi_path, aszarr=True)
    node = zarr.open(store, mode="r")
    if isinstance(node, zarr.Group):                 # resolution pyramid
        node = node[sorted(node.array_keys())[0]]    # level 0 = full res
    plane = np.asarray(node)
    while plane.ndim > 2:
        plane = plane[0]
    height, width = plane.shape
    log.info("DAPI plane %s px at %.4f um/px", (height, width), mpp)

    nuclei = pd.read_parquet(xenium_dir / "nucleus_boundaries.parquet")
    rows = []
    for cell_id, group in nuclei.groupby("cell_id", sort=False):
        px = group["vertex_x"].to_numpy() / mpp
        py = group["vertex_y"].to_numpy() / mpp
        x0, x1 = int(px.min()), int(np.ceil(px.max())) + 1
        y0, y1 = int(py.min()), int(np.ceil(py.max())) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            rows.append((cell_id, 0.0, 0.0, 0))
            continue
        crop = np.asarray(plane[y0:y1, x0:x1])
        yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
        inside = MplPath(np.column_stack([px - x0, py - y0])).contains_points(
            np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5]))
        area_px = int(inside.sum())
        total = float(crop.ravel()[inside].sum()) if area_px else 0.0
        rows.append((cell_id, total,
                     total / area_px if area_px else 0.0, area_px))
    frame = pd.DataFrame(rows, columns=["cell_id", "dapi_sum", "dapi_mean",
                                        "nucleus_area_px"])
    frame["nucleus_area_um2"] = frame["nucleus_area_px"] * mpp * mpp
    frame = frame.set_index("cell_id").reindex(cell_ids)
    missing = int(frame["dapi_sum"].isna().sum())
    if missing:
        log.warning("%d cells lack a nucleus polygon (NaN rows kept)", missing)
    out_dir = paths.dataset(dataset).root / "qc"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "nuclear_dapi.parquet"
    frame.to_parquet(out)
    log.info("wrote %s (%d cells, %d without nucleus)", out, len(frame),
             missing)
    return out


def extract_nuclear_counts(dataset: str) -> Path:
    """Counts from nucleus-overlapping transcripts only, bundle-aligned."""
    import pandas as pd
    import scipy.sparse as sp

    xenium_dir, _, cell_ids, gene_names = _bundle_meta(dataset)
    cell_index = pd.Series(np.arange(len(cell_ids)), index=cell_ids)
    gene_index = pd.Series(np.arange(len(gene_names)), index=gene_names)

    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(xenium_dir / "transcripts.parquet")
    matrix = sp.csr_matrix((len(cell_ids), len(gene_names)), dtype=np.float32)
    kept = total = 0
    for batch in parquet.iter_batches(
            columns=["cell_id", "feature_name", "overlaps_nucleus", "qv"],
            batch_size=5_000_000):
        chunk = batch.to_pandas()
        total += len(chunk)
        chunk = chunk[(chunk["overlaps_nucleus"] == 1) & (chunk["qv"] >= 20)]
        rows = cell_index.reindex(chunk["cell_id"].astype(str)).to_numpy()
        cols = gene_index.reindex(chunk["feature_name"].astype(str)).to_numpy()
        keep = ~(np.isnan(rows) | np.isnan(cols))
        kept += int(keep.sum())
        matrix = matrix + sp.coo_matrix(
            (np.ones(int(keep.sum()), dtype=np.float32),
             (rows[keep].astype(int), cols[keep].astype(int))),
            shape=matrix.shape).tocsr()
    out_dir = paths.dataset(dataset).root / "qc"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "nuclear_counts.npz"
    sp.save_npz(out, matrix)
    log.info("wrote %s: %d nuclear q20 transcripts of %d read (%.1f%%)",
             out, kept, total, 100 * kept / max(total, 1))
    return out


def summarise_nuclear_fraction(dataset: str) -> Path:
    """Nuclear share of each cell's counts, overall / by segmentation method /
    by type, from ``qc/nuclear_counts.npz`` against the bundle's counts."""
    import json

    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    ds = paths.dataset(dataset)
    adata = ad.read_h5ad(ds.root / "bundle" / "full.h5ad", backed="r")
    nuclear = sp.load_npz(ds.root / "qc" / "nuclear_counts.npz")
    total = np.asarray(adata[:].X.sum(axis=1)).ravel()
    nuc = np.asarray(nuclear.sum(axis=1)).ravel()
    if nuclear.shape != adata.shape:
        raise ValueError(f"nuclear matrix {nuclear.shape} vs bundle {adata.shape}")
    excess = int((nuc > total).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, nuc / total, np.nan)
    obs = adata.obs
    label = adata.uns.get("default_label", "cell_group")

    def by(column) -> dict:
        # a categorical's NaN survives astype(str); fold it as the loader does
        groups = obs[column].astype(object).fillna("Unassigned").astype(str).to_numpy()
        out = {}
        for g in sorted(set(groups)):
            m = groups == g
            out[g] = {"n_cells": int(m.sum()),
                      "nuclear_fraction_pooled": float(nuc[m].sum() / max(total[m].sum(), 1)),
                      "nuclear_fraction_median_cell": float(np.nanmedian(frac[m]))}
        return out

    summary = {
        "dataset": dataset, "n_cells": int(len(total)),
        "transcripts_total": int(total.sum()), "transcripts_nuclear": int(nuc.sum()),
        "nuclear_fraction_pooled": float(nuc.sum() / total.sum()),
        "nuclear_fraction_cell_quantiles": {
            str(q): float(np.nanpercentile(frac, q)) for q in (5, 25, 50, 75, 95)},
        "cells_with_zero_counts": int((total == 0).sum()),
        "cells_nuclear_exceeds_total": excess,
        "by_segmentation_method": by("xenium_segmentation_method")
        if "xenium_segmentation_method" in obs else {},
        "by_type": by(label) if label in obs else {},
    }
    out = ds.root / "qc" / "nuclear_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("%s: nuclear fraction %.3f pooled, median cell %.3f; %d cells with "
             "nuclear > total; wrote %s", dataset, summary["nuclear_fraction_pooled"],
             summary["nuclear_fraction_cell_quantiles"]["50"], excess, out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", required=True,
                        choices=("dapi", "nuclear-counts", "nuclear-summary"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.stage == "dapi":
        extract_nuclear_dapi(args.dataset)
    elif args.stage == "nuclear-counts":
        extract_nuclear_counts(args.dataset)
    else:
        summarise_nuclear_fraction(args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())

