#!/usr/bin/env python3
"""Read 10x Xenium outputs into the AnnData shape the rest of the pipeline uses.

Boundaries arrive as **parquet** in long form (one row per vertex) with
coordinates in **microns**.

Segmentation is *measured*, not inferred -- on the Prime 5K lung sample 97.8% of
cells came from interior or boundary stains and only 2.2% from nucleus expansion
-- which is why the contact graph is worth taking seriously here.

Geometry is converted to image-pixel coordinates on load (dividing by
``pixel_size``) so graphs, metrics and image overlay share one coordinate
system; ``uns['microns_per_pixel']`` converts back.

Usage::

    from discell.preprocess.xenium import load_sample
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from discell.preprocess.labels import read_cell_groups, read_clusters, read_gene_groups
from discell.preprocess.geometry import (
    DEFAULT_CLIP_RADIUS_UM,
    DEFAULT_MAX_EDGE_UM,
    DEFAULT_PRECISION_UM,
    DEFAULT_WALL_TOLERANCE_UM,
    _store_graph,
    add_apposed_wall,
    build_contact_graph,
    build_voronoi_graph,
    polygon_metrics,
)

log = logging.getLogger("discell.preprocess.xenium")


BOUNDARY_FILES = {
    "cell": "cell_boundaries.parquet",
    "nucleus": "nucleus_boundaries.parquet",
}
DEFAULT_POLYGON_KIND = "cell"

#: Xenium boundaries approach each other but almost never coincide exactly: on
#: the lung sample a quarter of neighbouring pairs sit within 0.12 um -- below
#: the 0.2125 um pixel -- yet only 2.5% touch at zero distance. Exact-contact
#: adjacency therefore collapses (mean degree 0.14, 88% isolated), while a 1 um
#: tolerance recovers a mean degree of 2.60. So contact
#: here defaults to a tolerance rather than to zero.
XENIUM_CONTACT_TOLERANCE_UM = 1.0

#: Files pulled out of a ``*_outs.zip`` when given one directly. The morphology
#: images and transcript tables are tens of GB and are not needed for graphs.
ZIP_MEMBERS = (
    "cell_boundaries.parquet",
    "nucleus_boundaries.parquet",
    "cells.parquet",
    "cell_feature_matrix.h5",
    "experiment.xenium",
    "metrics_summary.csv",
    "analysis.tar.gz",
)


class XeniumError(RuntimeError):
    """Raised when a directory cannot be read as Xenium output."""


def extract_xenium_outputs(zip_path: Path, dest: Path, force: bool = False) -> Path:
    """Extract the tabular members of a Xenium ``*_outs.zip`` into *dest*."""
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "cell_boundaries.parquet").exists() and not force:
        log.info("Reusing extracted Xenium outputs: %s", dest)
        return dest

    log.info("Extracting from %s", zip_path.name)
    started = time.time()
    with zipfile.ZipFile(zip_path) as archive:
        available = set(archive.namelist())
        for member in ZIP_MEMBERS:
            if member in available:
                archive.extract(member, dest)
    log.info("Extracted in %.1fs -> %s", time.time() - started, dest)
    return dest


def locate_xenium_dir(sample_path: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Resolve *sample_path* to a directory holding the Xenium outs."""
    path = Path(sample_path).expanduser().resolve()
    if path.is_dir():
        if (path / "cell_boundaries.parquet").exists():
            return path
        nested = sorted(path.glob("*/cell_boundaries.parquet"))
        if nested:
            return nested[0].parent
        zips = sorted(path.glob("*_outs.zip"))
        if zips:
            dest = Path(cache_dir) if cache_dir else path / zips[0].stem
            return extract_xenium_outputs(zips[0], dest)
        raise XeniumError(f"No cell_boundaries.parquet or *_outs.zip under {path}")
    if path.suffix == ".zip":
        dest = Path(cache_dir) if cache_dir else path.parent / path.stem
        return extract_xenium_outputs(path, dest)
    raise XeniumError(f"Cannot read {path} as Xenium output")


def read_experiment(xenium_dir: Path) -> dict:
    path = xenium_dir / "experiment.xenium"
    if not path.exists():
        raise XeniumError(f"Missing {path} -- needed for pixel_size")
    return json.loads(path.read_text())


def read_boundaries(
    xenium_dir: Path,
    kind: str = DEFAULT_POLYGON_KIND,
    pixel_size: float = 1.0,
    precision_um: float | None = DEFAULT_PRECISION_UM,
) -> tuple[np.ndarray, list]:
    """Read boundary vertices into shapely polygons, in image-pixel coordinates.

    The parquet is long-form -- ``cell_id, vertex_x, vertex_y, label_id``, one
    row per vertex, contiguous per cell -- so the rings are built with shapely's
    vectorised constructors rather than a Python loop over ~280k cells.
    """
    import shapely

    path = xenium_dir / BOUNDARY_FILES[kind]
    if not path.exists():
        raise XeniumError(f"Missing {path}")

    log.info("Reading boundaries: %s", path.name)
    started = time.time()
    frame = pd.read_parquet(path, columns=["cell_id", "vertex_x", "vertex_y"])

    codes, cell_ids = pd.factorize(frame["cell_id"], sort=False)
    counts = np.bincount(codes)
    # A ring needs 3 distinct points; anything smaller cannot bound an area.
    too_small = np.flatnonzero(counts < 4)
    if len(too_small):
        log.warning("%d cells have fewer than 4 vertices and are dropped", len(too_small))
        keep_rows = ~np.isin(codes, too_small)
        frame = frame[keep_rows]
        codes, cell_ids = pd.factorize(frame["cell_id"], sort=False)

    coords = frame[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float64) / pixel_size
    rings = shapely.linearrings(coords, indices=codes)
    polys = list(shapely.polygons(rings))
    log.info("  %d polygons from %d vertices in %.1fs", len(polys), len(coords),
             time.time() - started)

    polys = repair_polygons(polys, pixel_size, precision_um)
    cell_ids = np.asarray(cell_ids)
    good = np.array([p is not None for p in polys])
    if not good.all():
        cell_ids = cell_ids[good]
        polys = [p for p in polys if p is not None]
    return cell_ids, polys


def repair_polygons(
    polys: Sequence, pixel_size: float, precision_um: float | None = DEFAULT_PRECISION_UM
) -> list:
    """Make polygons valid and optionally snap them to a precision grid.

    Entries that cannot be rescued as a Polygon become ``None`` for the caller to
    drop. Keeping them would poison downstream geometry: ``make_valid`` can
    return a GeometryCollection of lines and points, whose ``.boundary`` is
    ``None``, which blows up in the shared-boundary computation.
    """
    import shapely
    from shapely.validation import make_valid

    grid = (precision_um / pixel_size) if (precision_um and pixel_size) else None
    out, n_repaired, n_degenerate = [], 0, 0
    for poly in polys:
        if not poly.is_valid:
            poly = make_valid(poly)
            n_repaired += 1
            if poly.geom_type != "Polygon":
                parts = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
                if not parts:
                    n_degenerate += 1
                    out.append(None)
                    continue
                poly = max(parts, key=lambda g: g.area)
        if grid:
            snapped = shapely.set_precision(poly, grid)
            if snapped.is_valid and not snapped.is_empty and snapped.geom_type == "Polygon":
                poly = snapped
        if poly.is_empty or poly.geom_type != "Polygon":
            n_degenerate += 1
            out.append(None)
            continue
        out.append(poly)
    if n_repaired or n_degenerate:
        log.info("  repaired %d invalid polygons, dropped %d degenerate",
                 n_repaired, n_degenerate)
    return out


def read_counts(xenium_dir: Path):
    """Read ``cell_feature_matrix.h5`` as raw per-cell counts."""
    import scanpy as sc

    path = xenium_dir / "cell_feature_matrix.h5"
    if not path.exists():
        raise XeniumError(f"Missing {path}")
    log.info("Reading %s", path.name)
    adata = sc.read_10x_h5(path)
    ids = adata.var["gene_ids"] if "gene_ids" in adata.var else pd.Series(adata.var_names)
    adata.var["gene_id"] = np.asarray(ids, dtype=object)
    adata.var["gene_name"] = np.asarray(adata.var_names, dtype=object)
    adata.var.drop(columns=["gene_ids"], errors="ignore", inplace=True)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    log.info("  %d cells x %d features", adata.n_obs, adata.n_vars)
    return adata








def load_xenium_sample(
    sample_path: str | Path, build_graphs: bool = True,
    contact_tolerance_um: float = XENIUM_CONTACT_TOLERANCE_UM,
    clip_radius_um: float | None = DEFAULT_CLIP_RADIUS_UM,
    precision_um: float | None = DEFAULT_PRECISION_UM,
    wall_tolerance_um: float | None = DEFAULT_WALL_TOLERANCE_UM,
    max_cells: int | None = None,
):
    """Load a Xenium sample: raw counts, polygons, metadata and neighbour graphs.

    Builds the same AnnData structure the rest of the pipeline expects, so
    the plotting and graph tooling works unchanged. *max_cells* takes a spatially
    contiguous subset, which is useful for iterating on a 280k-cell slide.
    """
    xenium_dir = locate_xenium_dir(sample_path)
    experiment = read_experiment(xenium_dir)
    pixel_size = float(experiment.get("pixel_size", 1.0))
    sample_id = str(experiment.get("region_name") or xenium_dir.name)
    log.info("Sample %s (pixel_size=%.4f um)", sample_id, pixel_size)

    adata = read_counts(xenium_dir)
    cell_ids, polys = read_boundaries(xenium_dir, DEFAULT_POLYGON_KIND,
                                     pixel_size, precision_um)

    # Match expression to geometry, in one shared order.
    position = pd.Series(np.arange(len(cell_ids)), index=pd.Index(cell_ids))
    position = position[~position.index.duplicated()].reindex(adata.obs_names)
    keep = position.notna().to_numpy()
    if (~keep).sum():
        log.info("%d cells in the matrix have no boundary -- dropped", int((~keep).sum()))
    order = position[keep].astype(int).to_numpy()
    adata = adata[keep].copy()
    polys = [polys[k] for k in order]
    log.info("Matched %d cells with both expression and geometry", adata.n_obs)

    if max_cells is not None and max_cells < adata.n_obs:
        # Contiguous window, not a random sample: a scattered subset would make
        # every neighbour graph meaningless.
        cents = np.array([[p.centroid.x, p.centroid.y] for p in polys])
        centre = np.median(cents, axis=0)
        nearest = np.argsort(np.linalg.norm(cents - centre, axis=1))[:max_cells]
        nearest = np.sort(nearest)
        adata = adata[nearest].copy()
        polys = [polys[k] for k in nearest]
        log.info("Subset to the %d cells nearest the tissue centre", adata.n_obs)

    metrics = polygon_metrics(polys, pixel_size)
    for column in metrics.columns:
        adata.obs[column] = metrics[column].to_numpy()

    cells_path = xenium_dir / "cells.parquet"
    if cells_path.exists():
        cells = pd.read_parquet(cells_path).set_index("cell_id").reindex(adata.obs_names)
        for column in ("transcript_counts", "total_counts", "cell_area",
                       "nucleus_area", "nucleus_count", "segmentation_method"):
            if column in cells:
                values = cells[column]
                adata.obs[f"xenium_{column}"] = (
                    pd.Categorical(values.astype(object)) if values.dtype == object
                    else values.to_numpy()
                )

    clusters = read_clusters(xenium_dir)
    label_columns: list[str] = []
    if len(clusters):
        clusters = clusters.reindex(adata.obs_names)
        for column in clusters.columns:
            labels = clusters[column].map(lambda v: None if pd.isna(v) else f"Cluster-{int(v)}")
            adata.obs[column] = pd.Categorical(labels.astype(object))
            label_columns.append(column)
    # Curated cell types, when the supplemental CSV has been downloaded. These
    # are named types with a published palette, so they outrank graphclust.
    groups = read_cell_groups(xenium_dir)
    if len(groups):
        matched = groups.reindex(adata.obs_names)
        labels = matched["group"].map(lambda v: None if pd.isna(v) else str(v))
        adata.obs["cell_group"] = pd.Categorical(labels.astype(object))
        if "color" in matched:
            colours = matched["color"].map(lambda v: None if pd.isna(v) else str(v))
            adata.obs["cell_group_color"] = pd.Categorical(colours.astype(object))
        n_missing = int(labels.isna().sum())
        log.info("  attached cell_group to %d cells (%d unlabelled)",
                 adata.n_obs - n_missing, n_missing)
        label_columns.insert(0, "cell_group")

    adata.uns["label_columns"] = label_columns
    if "cell_group" in label_columns:
        adata.uns["default_label"] = "cell_group"
    elif "graphclust" in label_columns:
        adata.uns["default_label"] = "graphclust"
    elif label_columns:
        adata.uns["default_label"] = label_columns[0]

    gene_groups = read_gene_groups(xenium_dir)
    if len(gene_groups):
        # A marker gene can belong to more than one group, so the index is not
        # unique and cannot be reindexed directly. var gets the first group;
        # uns keeps the full many-to-many mapping.
        series = gene_groups["group"]
        deduped = series[~series.index.duplicated(keep="first")]
        n_shared = len(series) - len(deduped)
        if n_shared:
            log.info("  %d marker gene(s) belong to several groups; var keeps the first",
                     n_shared)
        by_name = deduped.reindex(adata.var["gene_name"].astype(str))
        adata.var["gene_group"] = np.asarray(by_name.to_numpy(), dtype=object)
        adata.uns["gene_groups"] = {
            str(k): sorted(map(str, v.index)) for k, v in gene_groups.groupby("group")
        }
        n_hit = int(adata.var["gene_group"].notna().sum())
        log.info("  matched %d panel genes to a marker group", n_hit)

    # metrics was computed from the already-subset polygons, so this is aligned.
    centroids = metrics[["centroid_x_px", "centroid_y_px"]].to_numpy()
    adata.obsm["spatial"] = centroids.astype(np.float32)
    adata.obsm["spatial_um"] = (centroids * pixel_size).astype(np.float32)
    adata.uns["polygons"] = polys
    adata.uns["microns_per_pixel"] = pixel_size
    adata.uns["counts_are_raw"] = True
    adata.uns["platform"] = "xenium"
    adata.uns["xenium_dir"] = str(xenium_dir)
    adata.uns["sample"] = {
        "sample_id": sample_id,
        **{k: str(v) for k, v in experiment.items() if not isinstance(v, (dict, list))},
    }
    adata.obs["sample_id"] = pd.Categorical([sample_id] * adata.n_obs)

    if build_graphs:
        contact = build_contact_graph(
            polys, centroids, pixel_size, contact_tolerance_um, DEFAULT_MAX_EDGE_UM
        )
        voronoi, regions = build_voronoi_graph(
            polys, centroids, pixel_size, clip_radius_um, DEFAULT_MAX_EDGE_UM
        )
        _store_graph(adata, contact, "contact")
        _store_graph(adata, voronoi, "voronoi")
        adata.uns["voronoi_regions"] = regions
        adata.obs["contact_degree"] = contact.degrees()
        adata.obs["voronoi_degree"] = voronoi.degrees()
        if wall_tolerance_um is not None:
            for prefix in ("contact", "voronoi"):
                add_apposed_wall(adata, prefix, wall_tolerance_um, polys)

    return adata


def load_sample(
    sample_path: str, clip_radius_um: float = DEFAULT_CLIP_RADIUS_UM,
    max_cells: int | None = None, contact_tolerance_um: float | None = None,
    wall_tolerance_um: float | None = None, build_graphs: bool = True,
):
    """Load a sample and its graphs. Returns ``(adata, sample_dir)``.

    Passing ``None`` for either tolerance keeps the loader's own default.
    """
    extra = {"build_graphs": build_graphs}
    if contact_tolerance_um is not None:
        extra["contact_tolerance_um"] = contact_tolerance_um
    if wall_tolerance_um is not None:
        extra["wall_tolerance_um"] = wall_tolerance_um

    sample_dir = locate_xenium_dir(Path(sample_path).expanduser().resolve())
    adata = load_xenium_sample(sample_dir, clip_radius_um=clip_radius_um,
                               max_cells=max_cells, **extra)
    return adata, sample_dir
