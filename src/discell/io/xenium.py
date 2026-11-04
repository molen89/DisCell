"""Read a 10x Xenium output directory into the AnnData contract.

:func:`read_xenium` returns raw counts in ``.X`` (gene-expression features
only, control probes dropped), cell ids as ``.obs_names``, centroids in
micrometres in ``.obsm['spatial']`` and, when present, the label tables beside
the outputs. Centroids are the centers of mass of the segmented cell
boundaries (``cell_boundaries.parquet``), after the same repair and snapping
to a 0.01 um grid the paper's bundles used, so the graph built from them
reproduces the paper's. The same centroids in image pixels are kept in
``.obsm['spatial_px']``: the Voronoi graph is computed in that frame (see
:mod:`discell.graph.voronoi`). Polygons are not kept.

Nothing is written next to the raw data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("discell.io.xenium")

#: Polygon coordinates are snapped to this grid (micrometres) before centroids
#: are taken. Space Ranger polygons meet in floating-point slivers.
DEFAULT_PRECISION_UM = 0.01


class XeniumError(RuntimeError):
    """A directory cannot be read as Xenium output."""


def read_10x_h5(path: str | Path, gex_only: bool = True):
    """Read a 10x ``cell_feature_matrix.h5`` as AnnData (cells x features, float32 CSR).

    ``.var`` has ``gene_id`` and ``feature_type``; ``.var_names`` are the
    feature names, made unique.
    """
    import anndata as ad
    import h5py
    from anndata.utils import make_index_unique

    with h5py.File(path, "r") as handle:
        m = handle["matrix"]
        n_features, n_cells = (int(v) for v in m["shape"][()])
        matrix = sp.csr_matrix((m["data"][()].astype(np.float32), m["indices"][()],
                                m["indptr"][()]), shape=(n_cells, n_features))
        barcodes = [b.decode() for b in m["barcodes"][()]]
        feats = m["features"]
        names = [b.decode() for b in feats["name"][()]]
        ids = [b.decode() for b in feats["id"][()]]
        types = [b.decode() for b in feats["feature_type"][()]]
    var = pd.DataFrame({"gene_id": ids, "feature_type": types},
                       index=make_index_unique(pd.Index(names)))
    obs = pd.DataFrame(index=pd.Index(barcodes, dtype=str))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    if gex_only:
        adata = adata[:, np.asarray(types) == "Gene Expression"].copy()
    adata.var_names = make_index_unique(pd.Index(adata.var_names.astype(str)))
    return adata


def boundary_centroids(path: str | Path, pixel_size: float, cells: Sequence[str] | None = None,
                       precision_um: float | None = DEFAULT_PRECISION_UM) -> pd.DataFrame:
    """Centroids (um) of the cell polygons in a Xenium ``cell_boundaries.parquet``.

    The long-form vertex table becomes one polygon per cell, in image pixels.
    Cells with fewer than 4 vertices are dropped; invalid polygons are made
    valid (keeping the largest polygonal part); vertices are snapped to a
    *precision_um* grid when that leaves a valid polygon; polygons that cannot
    be rescued are dropped. *cells* restricts the read to those ids (every step
    is per cell, so the result is the same as for the whole slide). Returns a
    frame indexed by cell id with ``x_px``, ``y_px``, ``x_um``, ``y_um``
    (pixels times *pixel_size*) and ``covering_radius_um``, the radius of the
    smallest disc at the centroid that contains the polygon (the image extra
    leaves cells larger than its ego disc out).
    """
    import shapely
    from shapely.validation import make_valid

    frame = pd.read_parquet(path, columns=["cell_id", "vertex_x", "vertex_y"])
    if cells is not None:
        frame = frame[frame["cell_id"].isin(pd.Index(cells))]
    codes, ids = pd.factorize(frame["cell_id"], sort=False)
    small = np.flatnonzero(np.bincount(codes, minlength=len(ids)) < 4)
    if len(small):
        log.warning("%d cells have fewer than 4 vertices and are dropped", len(small))
        frame = frame[~np.isin(codes, small)]
        codes, ids = pd.factorize(frame["cell_id"], sort=False)
    coords = frame[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float64) / pixel_size
    polys = shapely.polygons(shapely.linearrings(coords, indices=codes))

    polygon = shapely.GeometryType.POLYGON
    for k in np.flatnonzero(~shapely.is_valid(polys)):
        fixed = make_valid(polys[k])
        if fixed.geom_type != "Polygon":
            parts = [g for g in getattr(fixed, "geoms", []) if g.geom_type == "Polygon"]
            fixed = max(parts, key=lambda g: g.area) if parts else None
        polys[k] = fixed
    alive = np.array([p is not None for p in polys])
    if precision_um and alive.any():
        snapped = shapely.set_precision(polys[alive], precision_um / pixel_size)
        ok = (shapely.is_valid(snapped) & ~shapely.is_empty(snapped)
              & (shapely.get_type_id(snapped) == polygon))
        polys[np.flatnonzero(alive)[ok]] = snapped[ok]
    alive[alive] = ~shapely.is_empty(polys[alive]) & (
        shapely.get_type_id(polys[alive]) == polygon)
    if (~alive).any():
        log.info("dropped %d degenerate polygons", int((~alive).sum()))
    polys = polys[alive]
    cent = shapely.centroid(polys)
    x_px, y_px = shapely.get_x(cent), shapely.get_y(cent)
    # radius of the smallest disc at the centroid that contains the cell
    coords, owner = shapely.get_coordinates(shapely.get_exterior_ring(polys),
                                            return_index=True)
    reach = np.sqrt((coords[:, 0] - x_px[owner]) ** 2 + (coords[:, 1] - y_px[owner]) ** 2)
    radius = np.zeros(len(polys))
    np.maximum.at(radius, owner, reach)
    return pd.DataFrame({"x_px": x_px, "y_px": y_px,
                         "x_um": x_px * pixel_size, "y_um": y_px * pixel_size,
                         "covering_radius_um": radius * pixel_size},
                        index=pd.Index(np.asarray(ids)[alive].astype(str), name="cell_id"))


def read_cell_groups(path: str | Path) -> pd.DataFrame:
    """A cell annotation table: ``cell_id`` plus label columns (e.g. ``group``, ``donor``)."""
    frame = pd.read_csv(path)
    if "cell_id" not in frame:
        raise XeniumError(f"{path} has no cell_id column")
    frame["cell_id"] = frame["cell_id"].astype(str)
    return frame.set_index("cell_id")


def read_clusters(outs: Path) -> pd.DataFrame:
    """Every clustering under ``analysis/clustering`` (or in ``analysis.tar.gz``).

    One column per clustering (``graphclust``, ``kmeans_<k>_clusters``), values
    ``Cluster-<n>``, indexed by cell id; empty when there is none.
    """
    import io
    import tarfile

    tables: dict[str, pd.Series] = {}
    tar_path, folder = outs / "analysis.tar.gz", outs / "analysis"
    if tar_path.exists():
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if member.name.endswith("clusters.csv"):
                    handle = tar.extractfile(member)
                    if handle is not None:
                        name = Path(member.name).parent.name
                        tables[name] = pd.read_csv(io.BytesIO(handle.read()))
    elif folder.is_dir():
        for csv in sorted(folder.glob("clustering/*/clusters.csv")):
            tables[csv.parent.name] = pd.read_csv(csv)
    out = {name.replace("gene_expression_", ""):
           table.set_index(table["Barcode"].astype(str))["Cluster"]
           for name, table in tables.items()}
    frame = pd.DataFrame(out)
    return frame.apply(lambda col: col.map(lambda v: None if pd.isna(v) else f"Cluster-{int(v)}"))


def read_xenium(outs: str | Path, *, labels: str | Path | pd.DataFrame | None = "auto",
                clusters: bool = True, donors: Sequence[str] | None = None,
                donor_key: str = "donor",
                region_um: tuple[float, float, float, float] | None = None,
                precision_um: float | None = DEFAULT_PRECISION_UM):
    """Read a Xenium output directory into AnnData.

    Parameters
    ----------
    outs
        The Xenium output directory (with ``cell_feature_matrix.h5``,
        ``cell_boundaries.parquet`` and ``experiment.xenium``).
    labels
        A cell annotation table: a CSV path or a frame with a ``cell_id``
        column (or index) and one column per label, attached to ``.obs`` as
        categoricals (a ``group`` column is renamed ``cell_group``). ``"auto"``
        uses a ``*cell_groups.csv`` in *outs* when there is exactly one; None
        attaches none.
    clusters
        Attach the 10x clusterings from ``analysis/`` (``graphclust``, ...).
    donors
        Keep only cells whose ``.obs[donor_key]`` is one of these (TMA cores).
    region_um
        Keep only cells with centroid in ``(x0, y0, x1, y1)``, micrometres.
    precision_um
        Snapping grid for the boundary polygons; None disables snapping.

    Returns
    -------
    AnnData with raw counts in ``.X`` (float32 CSR), ``.var['gene_id']``,
    centroids in um in ``.obsm['spatial']`` and in image pixels in
    ``.obsm['spatial_px']`` (both float64), the cell's covering radius in
    ``.obs['discell_covering_radius_um']``, and the provenance in
    ``.uns['discell']['xenium']``. Cells without a boundary are dropped. The
    cell order is the count matrix's.
    """
    outs = Path(outs).expanduser()
    for name in ("cell_feature_matrix.h5", "cell_boundaries.parquet", "experiment.xenium"):
        if not (outs / name).exists():
            raise XeniumError(f"{outs} is not a Xenium output directory: no {name}")
    experiment = json.loads((outs / "experiment.xenium").read_text())
    pixel_size = float(experiment["pixel_size"])
    adata = read_10x_h5(outs / "cell_feature_matrix.h5")

    table = None
    if isinstance(labels, pd.DataFrame):
        table = labels.set_index("cell_id") if "cell_id" in labels else labels
        table.index = table.index.astype(str)
    elif isinstance(labels, str) and labels == "auto":
        hits = sorted(outs.glob("*cell_groups.csv"))
        if len(hits) > 1:
            raise XeniumError(f"several cell_groups tables in {outs}: pass labels= explicitly")
        table = read_cell_groups(hits[0]) if hits else None
    elif labels is not None:
        table = read_cell_groups(labels)
    if table is not None:
        table = table.rename(columns={"group": "cell_group"})
        matched = table[~table.index.duplicated()].reindex(adata.obs_names)
        for column in matched.columns:
            values = matched[column].map(lambda v: None if pd.isna(v) else str(v))
            adata.obs[column] = pd.Categorical(values.astype(object))
    if clusters:
        found = read_clusters(outs)
        if len(found.columns):
            found = found.reindex(adata.obs_names)
            for column in found.columns:
                adata.obs[column] = pd.Categorical(found[column].astype(object))

    if donors is not None:
        if donor_key not in adata.obs:
            raise XeniumError(f"donors were given but no label table has a {donor_key!r} column")
        keep = adata.obs[donor_key].astype(object).isin(list(donors)).to_numpy()
        if not keep.any():
            raise XeniumError(f"no cells from donors {list(donors)}")
        adata = adata[keep].copy()

    cents = boundary_centroids(outs / "cell_boundaries.parquet", pixel_size,
                               cells=adata.obs_names if donors is not None else None,
                               precision_um=precision_um)
    cents = cents[~cents.index.duplicated()]
    has = adata.obs_names.isin(cents.index)
    if (~has).any():
        log.info("%d cells have no boundary and are dropped", int((~has).sum()))
        adata = adata[has].copy()
    cents = cents.loc[adata.obs_names]
    pos = cents[["x_um", "y_um"]].to_numpy(dtype=np.float64)
    pos_px = cents[["x_px", "y_px"]].to_numpy(dtype=np.float64)
    if region_um is not None:
        x0, y0, x1, y1 = region_um
        inside = (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
        adata, pos, pos_px = adata[inside].copy(), pos[inside], pos_px[inside]
    adata.obsm["spatial"] = pos
    adata.obsm["spatial_px"] = pos_px
    radius = cents["covering_radius_um"].to_numpy()
    adata.obs["discell_covering_radius_um"] = radius if region_um is None else radius[inside]
    adata.uns["discell"] = {"xenium": {
        "path": str(outs), "pixel_size_um": pixel_size,
        "region_name": str(experiment.get("region_name", "")),
        "donors": None if donors is None else [str(d) for d in donors],
        "region_um": None if region_um is None else [float(v) for v in region_um],
        "precision_um": precision_um,
    }}
    return adata
