#!/usr/bin/env python3
"""Save a complete, reloadable bundle for one analysis run.

An AnnData alone cannot hold everything: ``uns['polygons']`` contains shapely
geometries, which are not h5ad-serialisable, and the graph edge weights are
easier to work with as a table than as four sparse matrices. So a bundle is a
directory:

===========================  =============================================
``<name>.h5ad``              counts, obs, var, obsm, and every graph in obsp
``<name>_polygons.parquet``  cell_id plus the polygon as WKB
``<name>_edges_<graph>.parquet``  one row per edge, all metrics, in microns
``<name>_params.json``       provenance: sample, tolerances, versions, counts
===========================  =============================================

:func:`load_bundle` reverses it, rebuilding ``uns['polygons']`` from the WKB so
the result is identical to a fresh load.

Usage::

    python -m discell.data.export --sample <dir>              # -> data/datasets/<id>/bundle/full.*
    python -m discell.data.export --sample <dir> --variant trimmed \\
        --graph contact --contact-tolerance-um 2 --wall-tolerance-um 1 --min-apposed-um 0.001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from discell import paths

log = logging.getLogger("discell.data.export")

#: uns entries holding shapely geometry, which h5ad cannot store.
GEOMETRY_UNS_KEYS = ("polygons", "voronoi_regions")


def trim_graph_by_apposed_wall(adata, prefix: str, min_apposed_um: float) -> int:
    """Drop edges of graph *prefix* whose apposed wall is below the threshold.

    Rewrites every ``{prefix}_*`` matrix in ``obsp`` so the stored graph -- the
    one a model would consume -- matches what a filtered plot shows. Returns the
    number of edges removed.

    This is the trim that matters when the graph was built with a looser
    tolerance than the wall was measured with: those edges are adjacent by the
    contact test but share no membrane at all.
    """
    import scipy.sparse as sp

    key = f"{prefix}_apposed_wall_um"
    if key not in adata.obsp:
        raise KeyError(f"{key} not in obsp -- load with wall_tolerance_um set")

    conn = sp.triu(adata.obsp[f"{prefix}_connectivities"], k=1).tocoo()
    i, j = conn.row, conn.col
    apposed = np.asarray(adata.obsp[key].tocsr()[i, j]).ravel()
    keep = apposed >= min_apposed_um
    n_removed = int((~keep).sum())

    metrics = [m for m in ("connectivities", "shared_wall_um", "centroid_dist_um",
                           "wall_dist_um", "apposed_wall_um")
               if f"{prefix}_{m}" in adata.obsp]
    ki, kj = i[keep], j[keep]
    for metric in metrics:
        matrix = adata.obsp[f"{prefix}_{metric}"].tocsr()
        values = np.asarray(matrix[ki, kj]).ravel()
        rows = np.concatenate([ki, kj])
        cols = np.concatenate([kj, ki])
        vals = np.concatenate([values, values])
        adata.obsp[f"{prefix}_{metric}"] = sp.csr_matrix(
            (vals, (rows, cols)), shape=(adata.n_obs, adata.n_obs)
        )

    degrees = np.asarray(adata.obsp[f"{prefix}_connectivities"].sum(axis=1)).ravel()
    adata.uns[f"{prefix}_graph"] = {
        **adata.uns.get(f"{prefix}_graph", {}),
        "n_edges": int(keep.sum()),
        "mean_degree": float(degrees.mean()),
        "isolated_fraction": float((degrees == 0).mean()),
        "min_apposed_wall_um": float(min_apposed_um),
    }
    if f"{prefix}_degree" in adata.obs:
        adata.obs[f"{prefix}_degree"] = degrees.astype(np.int32)
    log.info(
        "  trimmed %s: removed %d of %d edges, mean degree now %.2f, isolated %.1f%%",
        prefix, n_removed, len(i), degrees.mean(), 100 * (degrees == 0).mean(),
    )
    return n_removed


def _graph_prefixes(adata) -> list[str]:
    return sorted({
        key.rsplit("_connectivities", 1)[0]
        for key in adata.obsp
        if key.endswith("_connectivities")
    })


def save_bundle(
    adata,
    out_dir: str | Path,
    name: str,
    params: dict | None = None,
    graphs: Sequence[str] | None = None,
    dataset_id: str | None = None,
) -> dict[str, Path]:
    """Write the full run to *out_dir* as ``<name>.*``. Returns the paths.

    *dataset_id* is stamped into the h5ad so a bundle carries its own identity:
    the loader compares it against any embeddings handed in, and a bundle moved
    or renamed still knows which slide it came from.
    """
    import shapely

    if dataset_id:
        adata.uns["dataset_id"] = str(dataset_id)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    graphs = list(graphs) if graphs is not None else _graph_prefixes(adata)

    polys = adata.uns.get("polygons")
    if polys is not None:
        path = out_dir / f"{name}_polygons.parquet"
        frame = pd.DataFrame({
            "cell": np.asarray(adata.obs_names, dtype=object),
            "wkb": list(shapely.to_wkb(np.asarray(polys, dtype=object))),
        })
        frame.to_parquet(path, index=False)
        written["polygons"] = path
        log.info("Wrote %s (%d polygons, %.1f MB)", path.name, len(frame),
                 path.stat().st_size / 1e6)

    from discell.data.geometry import graph_edge_frame

    for prefix in graphs:
        edges = graph_edge_frame(adata, prefix)
        path = out_dir / f"{name}_edges_{prefix}.parquet"
        edges.to_parquet(path, index=False)
        written[f"edges_{prefix}"] = path
        log.info("Wrote %s (%d edges, %.1f MB)", path.name, len(edges),
                 path.stat().st_size / 1e6)

    # h5ad last: strip the shapely objects, keep everything else.
    stripped = adata.copy()
    for key in GEOMETRY_UNS_KEYS:
        stripped.uns.pop(key, None)
    path = out_dir / f"{name}.h5ad"
    stripped.write_h5ad(path)
    written["h5ad"] = path
    log.info("Wrote %s (%.1f MB)", path.name, path.stat().st_size / 1e6)

    meta = {
        "name": name,
        "dataset_id": dataset_id or adata.uns.get("dataset_id", ""),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "microns_per_pixel": float(adata.uns.get("microns_per_pixel", float("nan"))),
        "platform": adata.uns.get("platform", "xenium"),
        "sample_id": adata.uns.get("sample", {}).get("sample_id", ""),
        "label_columns": list(adata.uns.get("label_columns", [])),
        "default_label": adata.uns.get("default_label", ""),
        "graphs": {p: dict(adata.uns.get(f"{p}_graph", {})) for p in graphs},
        "obs_columns": list(map(str, adata.obs.columns)),
        "obsp_keys": list(map(str, adata.obsp.keys())),
        "params": params or {},
    }
    path = out_dir / f"{name}_params.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    written["params"] = path
    log.info("Wrote %s", path.name)
    return written


def load_bundle(out_dir: str | Path, name: str):
    """Reload a bundle, restoring ``uns['polygons']`` from WKB."""
    import anndata as ad
    import shapely

    out_dir = Path(out_dir)
    adata = ad.read_h5ad(out_dir / f"{name}.h5ad")
    poly_path = out_dir / f"{name}_polygons.parquet"
    if poly_path.exists():
        frame = pd.read_parquet(poly_path)
        polys = list(shapely.from_wkb(frame["wkb"].to_numpy()))
        if list(map(str, frame["cell"])) != list(map(str, adata.obs_names)):
            log.warning("polygon order does not match obs_names -- reindexing")
            lookup = dict(zip(map(str, frame["cell"]), polys))
            polys = [lookup.get(str(c)) for c in adata.obs_names]
        adata.uns["polygons"] = polys
    return adata


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", required=True,
                        help="raw sample directory or archive; determines the dataset")
    parser.add_argument("--variant", default=paths.DEFAULT_VARIANT,
                        help="bundle name within the dataset, for holding several "
                             f"graph settings side by side (default {paths.DEFAULT_VARIANT!r})")
    parser.add_argument("--graph", default=None,
                        help="trim only this graph; default trims all present")
    parser.add_argument("--contact-tolerance-um", type=float, default=None)
    parser.add_argument("--wall-tolerance-um", type=float, default=None)
    parser.add_argument("--clip-radius-um", type=float, default=30.0)
    parser.add_argument("--min-apposed-um", type=float, default=None,
                        help="bake an apposed-wall trim into the stored graph; "
                             "omit to keep the whole graph")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from discell.data.xenium import load_sample

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )

    adata, sample_dir = load_sample(
        args.sample, args.clip_radius_um, args.max_cells,
        args.contact_tolerance_um, args.wall_tolerance_um,
    )

    trimmed = {}
    if args.min_apposed_um is not None:
        targets = [args.graph] if args.graph else _graph_prefixes(adata)
        for prefix in targets:
            trimmed[prefix] = trim_graph_by_apposed_wall(adata, prefix, args.min_apposed_um)

    ds = paths.dataset_for(sample_dir).ensure()
    name = args.variant
    params = {
        "sample": str(sample_dir),
        "contact_tolerance_um": args.contact_tolerance_um,
        "wall_tolerance_um": args.wall_tolerance_um,
        "clip_radius_um": args.clip_radius_um,
        "min_apposed_um": args.min_apposed_um,
        "max_cells": args.max_cells,
        "edges_trimmed": trimmed,
    }
    written = save_bundle(adata, ds.bundle_dir, name, params, dataset_id=ds.dataset_id)
    ds.write_manifest(
        bundle=name,
        platform=adata.uns.get("platform", "xenium"),
        source=str(sample_dir),
        sample_id=adata.uns.get("sample", {}).get("sample_id", ""),
        n_cells=int(adata.n_obs),
        n_features=int(adata.n_vars),
        microns_per_pixel=float(adata.uns.get("microns_per_pixel", float("nan"))),
        default_label=adata.uns.get("default_label", ""),
    )

    print(f"\nDataset '{ds.dataset_id}', bundle '{name}' in {ds.bundle_dir}:")
    total = 0
    for kind, path in written.items():
        size = path.stat().st_size
        total += size
        print(f"  {kind:16s} {path.name:52s} {size / 1e6:8.1f} MB")
    print(f"  {'':16s} {'total':52s} {total / 1e6:8.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
