#!/usr/bin/env python3
"""Writing a complete, reloadable bundle for one analysis run.

An AnnData alone cannot hold everything: ``uns['polygons']`` contains shapely
geometries, which are not h5ad-serialisable, and the graph edge weights are
easier to work with as a table than as four sparse matrices. So a bundle is a
directory of four files -- see :mod:`discell.data.bundle`, which reads them back.

Every edge the graph builder found is written. Nothing is trimmed on the way in:
the metrics needed to filter a graph all travel with it, so a consumer can take
whatever subset it wants without the bundle having thrown the rest away.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("discell.preprocess.bundle")

#: uns entries holding shapely geometry, which h5ad cannot store.
GEOMETRY_UNS_KEYS = ("polygons", "voronoi_regions")


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

    from discell.preprocess.geometry import graph_edge_frame

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
