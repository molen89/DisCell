#!/usr/bin/env python3
"""Reading a saved bundle.

A bundle is a directory written by :mod:`discell.preprocess.bundle`:

===========================  =============================================
``<name>.h5ad``              counts, obs, var, obsm, and every graph in obsp
``<name>_polygons.parquet``  cell_id plus the polygon as WKB
``<name>_edges_<graph>.parquet``  one row per edge, all metrics, in microns
``<name>_params.json``       provenance: sample, tolerances, versions, counts
===========================  =============================================

Only the reading half lives here, so the consumer side never has to import the
preprocessing package to open what it produced.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("discell.data.bundle")


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
