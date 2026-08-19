#!/usr/bin/env python3
"""Read 10x Visium HD segmented (per-cell) outputs with polygons and cell graphs.

Companion to :mod:`discell.data.visium_hd`, which works on bins. This module
works on the ``*_segmented_outputs.tar.gz`` products, where Space Ranger has
segmented nuclei and assigned counts to individual cells.

Expression and geometry are matched by construction: :func:`load_segmented_sample`
returns an AnnData whose ``obs`` is exactly the set of cells that have *both* a
count vector and a polygon, in one shared order. ``adata.uns['polygons']`` indexes
positionally into ``adata.obs``.

Two neighbour graphs are built, because they answer different questions.

**Contact graph** -- cells whose Space Ranger polygons actually touch. Faithful to
the segmentation, but Space Ranger cell boundaries are nucleus masks dilated
independently, so they do not tile the plane: measured mean degree is ~2.6 where
planar geometry demands ~6, and 5-11% of cells touch nothing at all.

**Voronoi graph** -- an exact planar partition seeded from the cell centroids.
Every adjacent pair shares exactly one edge of positive length, mean degree ~6,
no orphans. ``clip_radius_um`` caps how far a cell may claim territory, trading
coverage against realism.

Both carry the same edge attributes, in microns:

===================  =======================================================
``shared_wall_um``   length of the shared boundary between the two cells
``centroid_dist_um`` distance between cell centroids
``wall_dist_um``     minimum distance between the two *original* polygon
                     boundaries (0 when they touch or overlap)
===================  =======================================================

Usage::

    python -m discell.data.segmented --self-test
    python -m discell.data.segmented --sample <dir-or-tar> --out cells.h5ad
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from discell.data.visium_hd import (
    DEFAULT_DATA_ROOT,
    DEFAULT_TEST_SAMPLE,
    VisiumHDError,
    attach_sample_metadata,
    resolve_sample,
)

log = logging.getLogger("discell.data.segmented")

DEFAULT_CACHE_DIRNAME = "segmented_outputs_extracted"
DEFAULT_POLYGON_KIND = "cell"  # 'cell' (nucleus expanded) or 'nucleus'
DEFAULT_MATRIX = "filtered"

#: Contact counts as adjacency when polygons are within this many microns.
DEFAULT_CONTACT_TOLERANCE_UM = 0.0

#: Voronoi cells are clipped to a disc of this radius about their centroid, so a
#: cell at the tissue edge cannot claim unbounded territory. ``None`` disables.
DEFAULT_CLIP_RADIUS_UM = 30.0

#: Delaunay edges longer than this are dropped before the Voronoi pass -- they
#: bridge tissue holes. Mouse brain p99 is 76um against a 3006um maximum.
DEFAULT_MAX_EDGE_UM = 100.0

#: Tolerance for the apposed-wall metric: how close two membranes must run to
#: count as sharing wall. Deliberately independent of the graph's adjacency
#: tolerance -- adjacency decides who is a neighbour, this decides how much they
#: share. Set to None to skip the computation.
DEFAULT_WALL_TOLERANCE_UM = 1.0

#: Snap polygon coordinates to this grid (microns) before geometry ops. Space
#: Ranger polygons meet in floating-point slivers -- 44% of contacts nominally
#: overlap with a median area of 0.0 um^2 -- which snapping collapses to shared
#: edges.
DEFAULT_PRECISION_UM = 0.01

#: ``cellid_000000123-1`` -> 123
CELL_BARCODE_RE = re.compile(r"^cellid_(?P<cid>\d+)-(?P<gem>\d+)$")

POLYGON_FILES = {
    "cell": "cell_segmentations.geojson",
    "nucleus": "nucleus_segmentations.geojson",
}
ANNOTATED_POLYGON_FILES = {
    "cell": "graphclust_annotated_cell_segmentations.geojson",
    "nucleus": "graphclust_annotated_nucleus_segmentations.geojson",
}


@dataclass
class CellGraph:
    """An undirected neighbour graph over cells, with geometric edge weights.

    ``i``/``j`` are positional indices into the cell axis. All lengths are in
    microns.
    """

    n_cells: int
    i: np.ndarray
    j: np.ndarray
    shared_wall_um: np.ndarray
    centroid_dist_um: np.ndarray
    wall_dist_um: np.ndarray
    kind: str = ""

    def __len__(self) -> int:
        return len(self.i)

    def degrees(self) -> np.ndarray:
        deg = np.zeros(self.n_cells, dtype=np.int32)
        np.add.at(deg, self.i, 1)
        np.add.at(deg, self.j, 1)
        return deg

    def to_sparse(self, weights: np.ndarray):
        """Symmetric CSR matrix carrying *weights* on both triangles."""
        import scipy.sparse as sp

        rows = np.concatenate([self.i, self.j])
        cols = np.concatenate([self.j, self.i])
        vals = np.concatenate([weights, weights])
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.n_cells, self.n_cells))

    def summary(self) -> str:
        deg = self.degrees()
        wall = self.shared_wall_um
        return (
            f"{self.kind}: {len(self)} edges, mean degree {deg.mean():.2f}, "
            f"isolated {100 * (deg == 0).mean():.1f}%, "
            f"median shared wall {np.median(wall) if len(wall) else 0:.2f} um"
        )


# --------------------------------------------------------------------------
# Locating and unpacking
# --------------------------------------------------------------------------


def _wanted_segmented_member(rel_parts: Sequence[str], matrix: str) -> bool:
    if not rel_parts:
        return False
    head = rel_parts[0]
    if head == f"{matrix}_feature_cell_matrix.h5":
        return True
    if head.endswith(".geojson"):
        return True
    if head == "spatial":
        return rel_parts[-1] == "scalefactors_json.json"
    if head == "analysis":
        return rel_parts[-1] == "clusters.csv"
    return False


def extract_segmented_outputs(
    tar_path: Path, dest: Path, matrix: str = DEFAULT_MATRIX, force: bool = False
) -> Path:
    """Extract polygons, per-cell matrix and scalefactors from *tar_path*."""
    import tarfile

    matrix_h5 = dest / f"{matrix}_feature_cell_matrix.h5"
    if matrix_h5.exists() and not force:
        log.info("Reusing extracted segmented outputs: %s", dest)
        return dest

    log.info("Extracting segmented outputs from %s", tar_path.name)
    started = time.time()
    dest.mkdir(parents=True, exist_ok=True)

    n = 0
    with tarfile.open(tar_path, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if "segmented_outputs" not in parts:
                continue
            rel = list(parts[parts.index("segmented_outputs") + 1 :])
            if not _wanted_segmented_member(rel, matrix):
                continue
            target = dest.joinpath(*rel)
            if not str(target.resolve()).startswith(str(dest.resolve())):
                raise VisiumHDError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with open(target, "wb") as handle:
                while chunk := source.read(1 << 20):
                    handle.write(chunk)
            n += 1

    if not matrix_h5.exists():
        raise VisiumHDError(f"{matrix}_feature_cell_matrix.h5 not found in {tar_path}")
    log.info("Extracted %d files in %.1fs -> %s", n, time.time() - started, dest)
    return dest


def locate_segmented_dir(
    sample_path: str | Path,
    matrix: str = DEFAULT_MATRIX,
    cache_dir: str | Path | None = None,
) -> tuple[Path, str, Path]:
    """Return ``(segmented_dir, sample_id, sample_dir)``, extracting if needed."""
    path = Path(sample_path).expanduser().resolve()

    # Pointed straight at an extracted segmented_outputs directory.
    if path.is_dir() and (path / f"{matrix}_feature_cell_matrix.h5").exists():
        sample_dir = path.parent
        while sample_dir.name in ("segmented_outputs", "seg_tmp", DEFAULT_CACHE_DIRNAME):
            sample_dir = sample_dir.parent
        return path, sample_dir.name, sample_dir

    sample_dir, _, sample_id = resolve_sample(path)
    for candidate in (
        sample_dir / "segmented_outputs",
        sample_dir / "seg_tmp" / "segmented_outputs",
        sample_dir / DEFAULT_CACHE_DIRNAME,
    ):
        if (candidate / f"{matrix}_feature_cell_matrix.h5").exists():
            log.info("Using pre-extracted %s", candidate)
            return candidate, sample_id, sample_dir

    tars = sorted(sample_dir.glob("*_segmented_outputs.tar.gz"))
    if not tars:
        raise VisiumHDError(f"No *_segmented_outputs.tar.gz under {sample_dir}")
    dest = Path(cache_dir) if cache_dir else sample_dir / DEFAULT_CACHE_DIRNAME
    return extract_segmented_outputs(tars[0], dest, matrix), sample_id, sample_dir


def read_scalefactors(seg_dir: Path) -> dict:
    path = seg_dir / "spatial" / "scalefactors_json.json"
    if not path.exists():
        raise VisiumHDError(f"Missing {path} -- needed for pixel->micron conversion")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Polygons
# --------------------------------------------------------------------------


def read_polygons(
    seg_dir: Path,
    kind: str = DEFAULT_POLYGON_KIND,
    annotated: bool = True,
    precision_um: float | None = DEFAULT_PRECISION_UM,
    microns_per_pixel: float | None = None,
) -> tuple[np.ndarray, list, pd.DataFrame]:
    """Read segmentation polygons.

    Returns ``(cell_ids, polygons, properties)``. Polygons are shapely geometries
    in full-resolution pixel coordinates -- the same space as
    ``obsm['spatial']`` from the binned loader.

    Self-intersecting rings are repaired with ``make_valid``; when that yields a
    multi-part geometry the largest part is kept. When *precision_um* is set,
    coordinates are snapped to that grid so neighbours that meet in
    floating-point slivers resolve to shared edges.
    """
    from shapely import set_precision
    from shapely.geometry import shape
    from shapely.validation import make_valid

    names = ANNOTATED_POLYGON_FILES if annotated else POLYGON_FILES
    path = seg_dir / names[kind]
    if not path.exists():
        path = seg_dir / POLYGON_FILES[kind]
        if not path.exists():
            raise VisiumHDError(f"No {kind} segmentation geojson in {seg_dir}")

    log.info("Reading polygons: %s", path.name)
    payload = json.loads(path.read_text())
    features = payload["features"]

    grid = None
    if precision_um and microns_per_pixel:
        grid = precision_um / microns_per_pixel

    cell_ids, polys, props = [], [], []
    n_repaired = n_dropped = 0
    for feature in features:
        poly = shape(feature["geometry"])
        if not poly.is_valid:
            poly = make_valid(poly)
            n_repaired += 1
            if poly.geom_type != "Polygon":
                parts = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
                if not parts:
                    n_dropped += 1
                    continue
                poly = max(parts, key=lambda g: g.area)
        if grid:
            snapped = set_precision(poly, grid)
            if snapped.is_valid and not snapped.is_empty and snapped.geom_type == "Polygon":
                poly = snapped
        properties = dict(feature.get("properties", {}))
        classification = properties.pop("classification", None)
        if isinstance(classification, dict):
            properties["cluster"] = classification.get("name")
            colour = classification.get("color")
            if colour:
                properties["cluster_color"] = "#%02x%02x%02x" % tuple(int(c) for c in colour[:3])
        cell_ids.append(properties.get("cell_id"))
        polys.append(poly)
        props.append(properties)

    log.info(
        "  %d polygons (%d repaired, %d dropped)", len(polys), n_repaired, n_dropped
    )
    frame = pd.DataFrame(props)
    return np.asarray(cell_ids), polys, frame


def polygon_metrics(polys: Sequence, microns_per_pixel: float) -> pd.DataFrame:
    """Per-cell geometry: centroid, area, perimeter and shape descriptors.

    Both a centroid and a representative point are returned. The centroid is the
    centre of mass and is the right position coordinate, but for a sufficiently
    concave cell it can lie *outside* the outline -- 2 of 40,222 in the mouse
    brain, at solidity ~0.73. ``rep_*`` is guaranteed to be inside the polygon
    and is the safer anchor when cutting an image patch "around the cell".
    """
    mpp = microns_per_pixel
    cx = np.array([p.centroid.x for p in polys], dtype=np.float64)
    cy = np.array([p.centroid.y for p in polys], dtype=np.float64)
    reps = [p.representative_point() for p in polys]
    rx = np.array([p.x for p in reps], dtype=np.float64)
    ry = np.array([p.y for p in reps], dtype=np.float64)
    area_px = np.array([p.area for p in polys], dtype=np.float64)
    perim_px = np.array([p.length for p in polys], dtype=np.float64)
    hull_area = np.array([p.convex_hull.area for p in polys], dtype=np.float64)

    area_um2 = area_px * mpp**2
    perim_um = perim_px * mpp
    with np.errstate(divide="ignore", invalid="ignore"):
        circularity = np.where(perim_um > 0, 4 * np.pi * area_um2 / perim_um**2, 0.0)
        solidity = np.where(hull_area > 0, area_px / hull_area, 0.0)

    return pd.DataFrame(
        {
            "centroid_x_px": cx,
            "centroid_y_px": cy,
            "centroid_x_um": cx * mpp,
            "centroid_y_um": cy * mpp,
            "rep_x_px": rx,
            "rep_y_px": ry,
            "area_um2": area_um2,
            "perimeter_um": perim_um,
            "equiv_diameter_um": 2 * np.sqrt(np.maximum(area_um2, 0) / np.pi),
            "circularity": circularity,
            "solidity": solidity,
        }
    )


# --------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------


def _candidate_pairs(centroids: np.ndarray, max_edge_px: float) -> np.ndarray:
    """Delaunay edges shorter than *max_edge_px*, as an ``(n, 2)`` array.

    Delaunay is O(n log n) and gives every plausible neighbour, which keeps the
    expensive polygon operations off the O(n^2) all-pairs path. Its long edges
    bridge tissue holes, so they are pruned by length.
    """
    from scipy.spatial import Delaunay

    if len(centroids) < 4:
        pairs = [(i, j) for i in range(len(centroids)) for j in range(i + 1, len(centroids))]
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    tri = Delaunay(centroids)
    simplices = tri.simplices
    edges = np.vstack(
        [simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]]
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(centroids[edges[:, 0]] - centroids[edges[:, 1]], axis=1)
    keep = lengths <= max_edge_px
    log.info(
        "  Delaunay: %d edges, %d within %.0f px", len(edges), int(keep.sum()), max_edge_px
    )
    return edges[keep]


def _shared_boundary_um(a, b, mpp: float) -> float:
    """Length of the boundary a and b have in common, in microns.

    For polygons that meet along an edge this is the edge length. For polygons
    that overlap in area -- which after snapping should be rare -- the shared
    boundary is taken as the intersection of their *boundaries*, so an overlap
    lens does not masquerade as a long wall.
    """
    if a.disjoint(b):
        return 0.0
    inter = a.intersection(b)
    if inter.is_empty:
        return 0.0
    if inter.area > 0:
        # .boundary is None for non-polygonal geometries; treat as no shared wall
        # rather than crashing on a degenerate input.
        ba, bb = a.boundary, b.boundary
        if ba is None or bb is None:
            return 0.0
        shared = ba.intersection(bb)
        return float(shared.length) * mpp if not shared.is_empty else 0.0
    return float(inter.length) * mpp


def apposed_wall_um(
    polys: Sequence,
    i: np.ndarray,
    j: np.ndarray,
    microns_per_pixel: float,
    tolerance_um: float = 1.0,
) -> np.ndarray:
    """Length of membrane along which two cells run close to one another.

    For each pair, the portion of cell *i*'s boundary lying within
    *tolerance_um* of cell *j* is measured, and vice versa; the two are averaged.

    This is the physically meaningful notion of "how much wall do they share"
    when cells do not actually touch, and it is what the Voronoi graph's
    ``shared_wall_um`` is *not*: that measures the boundary between two Voronoi
    territories, which sits in the empty space between cells and is essentially
    uncorrelated with whether the cells are adjacent at all (r = -0.11 on Xenium
    ovarian; pairs 15-100 um apart still score ~5.9 um). Apposed wall correlates
    with the gap as it should (r = -0.51 at 1 um).

    Separating this from adjacency is deliberate: the graph decides *who* is a
    neighbour, this decides *how much* they share, and the two want different
    tolerances.
    """
    import shapely

    if len(i) == 0:
        return np.empty(0, dtype=np.float64)

    eps = tolerance_um / microns_per_pixel
    a = np.asarray(polys, dtype=object)[i]
    b = np.asarray(polys, dtype=object)[j]
    # Vectorised at the C level; a Python loop over ~10^6 edges is not viable.
    a_near_b = shapely.length(
        shapely.intersection(shapely.boundary(a), shapely.buffer(b, eps))
    )
    b_near_a = shapely.length(
        shapely.intersection(shapely.boundary(b), shapely.buffer(a, eps))
    )
    return 0.5 * (a_near_b + b_near_a) * microns_per_pixel


def add_apposed_wall(
    adata, prefix: str, tolerance_um: float = 1.0, polys: Sequence | None = None
) -> np.ndarray:
    """Compute and store ``{prefix}_apposed_wall_um`` for an existing graph."""
    import scipy.sparse as sp

    polys = polys if polys is not None else adata.uns["polygons"]
    mpp = float(adata.uns["microns_per_pixel"])
    conn = sp.triu(adata.obsp[f"{prefix}_connectivities"], k=1).tocoo()
    i, j = conn.row, conn.col

    started = time.time()
    values = apposed_wall_um(polys, i, j, mpp, tolerance_um)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    vals = np.concatenate([values, values])
    adata.obsp[f"{prefix}_apposed_wall_um"] = sp.csr_matrix(
        (vals, (rows, cols)), shape=(adata.n_obs, adata.n_obs)
    )
    nonzero = values > 0
    log.info(
        "  %s apposed wall (tol %.2f um): %.1f%% nonzero, median %.2f um [%.1fs]",
        prefix, tolerance_um, 100 * nonzero.mean() if len(values) else 0.0,
        float(np.median(values[nonzero])) if nonzero.any() else 0.0,
        time.time() - started,
    )
    adata.uns[f"{prefix}_graph"] = {
        **adata.uns[f"{prefix}_graph"],
        "apposed_wall_tolerance_um": tolerance_um,
    }
    return values


def build_contact_graph(
    polys: Sequence,
    centroids: np.ndarray,
    microns_per_pixel: float,
    tolerance_um: float = DEFAULT_CONTACT_TOLERANCE_UM,
    max_edge_um: float = DEFAULT_MAX_EDGE_UM,
) -> CellGraph:
    """Adjacency where the segmentation polygons actually touch.

    Two cells are adjacent when their polygons are within *tolerance_um* of each
    other. Faithful to Space Ranger, but sparse: these polygons are independently
    dilated nuclei and do not tile the plane.
    """
    mpp = microns_per_pixel
    pairs = _candidate_pairs(centroids, max_edge_um / mpp)
    tol_px = tolerance_um / mpp

    i_out, j_out, wall, gap = [], [], [], []
    for a, b in pairs:
        pa, pb = polys[a], polys[b]
        d_px = pa.distance(pb)
        if d_px > tol_px:
            continue
        i_out.append(a)
        j_out.append(b)
        wall.append(_shared_boundary_um(pa, pb, mpp))
        gap.append(d_px * mpp)

    i_arr = np.asarray(i_out, dtype=np.int64)
    j_arr = np.asarray(j_out, dtype=np.int64)
    cdist = np.linalg.norm(centroids[i_arr] - centroids[j_arr], axis=1) * mpp if len(i_arr) else np.empty(0)
    graph = CellGraph(
        n_cells=len(polys),
        i=i_arr,
        j=j_arr,
        shared_wall_um=np.asarray(wall, dtype=np.float64),
        centroid_dist_um=cdist,
        wall_dist_um=np.asarray(gap, dtype=np.float64),
        kind=f"contact(tol={tolerance_um}um)",
    )
    log.info("  %s", graph.summary())
    return graph


def build_voronoi_graph(
    polys: Sequence,
    centroids: np.ndarray,
    microns_per_pixel: float,
    clip_radius_um: float | None = DEFAULT_CLIP_RADIUS_UM,
    max_edge_um: float = DEFAULT_MAX_EDGE_UM,
) -> tuple[CellGraph, list]:
    """Adjacency from a Voronoi partition of the cell centroids.

    An exact planar partition: adjacent cells share exactly one edge of positive
    length, so ``shared_wall_um`` is always meaningful. Returns the graph and the
    clipped Voronoi geometries, which are useful for plotting.

    ``wall_dist_um`` is still measured between the *original* polygons, so it
    keeps its physical meaning even though adjacency comes from the partition.
    """
    from shapely.geometry import MultiPoint, Point
    from shapely.ops import voronoi_diagram

    mpp = microns_per_pixel
    points = MultiPoint([Point(x, y) for x, y in centroids])
    log.info("Building Voronoi partition over %d centroids", len(centroids))
    regions = list(voronoi_diagram(points, tolerance=0.0).geoms)

    # voronoi_diagram does not preserve input order; match each region to the
    # centroid it contains.
    from shapely.strtree import STRtree

    tree = STRtree(regions)
    owner: list = [None] * len(centroids)
    for k, (x, y) in enumerate(centroids):
        probe = Point(x, y)
        for hit in tree.query(probe):
            if regions[hit].contains(probe):
                owner[k] = regions[hit]
                break
    unmatched = sum(o is None for o in owner)
    if unmatched:
        log.warning("%d centroids fell outside every Voronoi region", unmatched)

    if clip_radius_um:
        radius_px = clip_radius_um / mpp
        owner = [
            None if g is None else g.intersection(Point(*c).buffer(radius_px))
            for g, c in zip(owner, centroids)
        ]

    pairs = _candidate_pairs(centroids, max_edge_um / mpp)
    i_out, j_out, wall, gap = [], [], [], []
    for a, b in pairs:
        ga, gb = owner[a], owner[b]
        if ga is None or gb is None or ga.is_empty or gb.is_empty:
            continue
        length = _shared_boundary_um(ga, gb, mpp)
        if length <= 0:
            continue
        i_out.append(a)
        j_out.append(b)
        wall.append(length)
        gap.append(polys[a].distance(polys[b]) * mpp)

    i_arr = np.asarray(i_out, dtype=np.int64)
    j_arr = np.asarray(j_out, dtype=np.int64)
    cdist = np.linalg.norm(centroids[i_arr] - centroids[j_arr], axis=1) * mpp if len(i_arr) else np.empty(0)
    graph = CellGraph(
        n_cells=len(polys),
        i=i_arr,
        j=j_arr,
        shared_wall_um=np.asarray(wall, dtype=np.float64),
        centroid_dist_um=cdist,
        wall_dist_um=np.asarray(gap, dtype=np.float64),
        kind=f"voronoi(clip={clip_radius_um}um)",
    )
    log.info("  %s", graph.summary())
    return graph, owner


# --------------------------------------------------------------------------
# Expression, matched to geometry
# --------------------------------------------------------------------------


def read_cell_counts(seg_dir: Path, matrix: str = DEFAULT_MATRIX):
    """Read the per-cell raw count matrix."""
    import scanpy as sc

    h5 = seg_dir / f"{matrix}_feature_cell_matrix.h5"
    if not h5.exists():
        raise VisiumHDError(f"Missing per-cell matrix: {h5}")
    log.info("Reading %s", h5.name)
    adata = sc.read_10x_h5(h5)
    ids = adata.var["gene_ids"] if "gene_ids" in adata.var else pd.Series(adata.var_names)
    adata.var["gene_id"] = np.asarray(ids, dtype=object)
    adata.var["gene_name"] = np.asarray(adata.var_names, dtype=object)
    adata.var.drop(columns=["gene_ids"], errors="ignore", inplace=True)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    log.info("  %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def parse_cell_ids(barcodes: Iterable[str]) -> np.ndarray:
    """``cellid_000000123-1`` -> ``123``; ``-1`` when the barcode does not match."""
    out = []
    for barcode in barcodes:
        match = CELL_BARCODE_RE.match(str(barcode))
        out.append(int(match.group("cid")) if match else -1)
    return np.asarray(out, dtype=np.int64)


def match_expression_to_polygons(adata, cell_ids: np.ndarray, polys: Sequence, props: pd.DataFrame):
    """Restrict expression and geometry to their intersection, in one order.

    Space Ranger emits a polygon for every segmented object but drops
    low-count cells from the filtered matrix, so the two sets differ. Returns
    ``(adata_subset, polys_subset, props_subset)`` aligned positionally.
    """
    matrix_ids = parse_cell_ids(adata.obs_names)
    unparsed = int((matrix_ids < 0).sum())
    if unparsed:
        log.warning("%d matrix barcodes did not match cellid_<n>-<gem>", unparsed)

    poly_index = pd.Series(np.arange(len(cell_ids)), index=pd.Index(cell_ids))
    poly_index = poly_index[~poly_index.index.duplicated()]
    position = poly_index.reindex(matrix_ids)
    keep = position.notna().to_numpy()

    n_drop_expr = int((~keep).sum())
    n_drop_poly = len(cell_ids) - int(keep.sum())
    if n_drop_expr:
        log.info("%d cells in the matrix have no polygon -- dropped", n_drop_expr)
    if n_drop_poly:
        log.info("%d polygons have no expression -- dropped", n_drop_poly)

    order = position[keep].astype(int).to_numpy()
    subset = adata[keep].copy()
    subset.obs["cell_id"] = matrix_ids[keep]
    polys_out = [polys[k] for k in order]
    props_out = props.iloc[order].reset_index(drop=True) if len(props) else props
    log.info("Matched %d cells with both expression and geometry", subset.n_obs)
    return subset, polys_out, props_out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def load_segmented_sample(
    sample_path: str | Path,
    matrix: str = DEFAULT_MATRIX,
    polygon_kind: str = DEFAULT_POLYGON_KIND,
    build_graphs: bool = True,
    contact_tolerance_um: float = DEFAULT_CONTACT_TOLERANCE_UM,
    clip_radius_um: float | None = DEFAULT_CLIP_RADIUS_UM,
    max_edge_um: float = DEFAULT_MAX_EDGE_UM,
    precision_um: float | None = DEFAULT_PRECISION_UM,
    wall_tolerance_um: float | None = DEFAULT_WALL_TOLERANCE_UM,
    hvg_file: str | Path | None = None,
    n_top_genes: int | None = None,
    cache_dir: str | Path | None = None,
    metadata_csv: str | Path | None = None,
):
    """Load one sample's per-cell expression, polygons and neighbour graphs.

    The returned AnnData has, for every cell and in one shared order: raw counts
    in ``X``, centroid and morphology in ``obs``, centroid coordinates in
    ``obsm['spatial']`` (full-res pixels) and ``obsm['spatial_um']``, shapely
    polygons in ``uns['polygons']``, and both graphs in ``obsp``.
    """
    seg_dir, sample_id, sample_dir = locate_segmented_dir(sample_path, matrix, cache_dir)
    scalefactors = read_scalefactors(seg_dir)
    mpp = float(scalefactors["microns_per_pixel"])

    adata = read_cell_counts(seg_dir, matrix)
    cell_ids, polys, props = read_polygons(
        seg_dir, polygon_kind, annotated=True, precision_um=precision_um, microns_per_pixel=mpp
    )
    adata, polys, props = match_expression_to_polygons(adata, cell_ids, polys, props)

    metrics = polygon_metrics(polys, mpp)
    for column in metrics.columns:
        adata.obs[column] = metrics[column].to_numpy()
    for column in props.columns:
        if column in ("cell_id",):
            continue
        values = props[column]
        adata.obs[column] = (
            pd.Categorical(values.astype(object))
            if values.dtype == object
            else values.to_numpy()
        )

    centroids = metrics[["centroid_x_px", "centroid_y_px"]].to_numpy()
    adata.obsm["spatial"] = centroids.astype(np.float32)
    adata.obsm["spatial_um"] = (centroids * mpp).astype(np.float32)
    adata.uns["polygons"] = polys
    adata.uns["polygon_kind"] = polygon_kind
    adata.uns["microns_per_pixel"] = mpp
    adata.uns["spatial_scalefactors"] = scalefactors
    adata.uns["counts_are_raw"] = True
    adata.uns["segmented_dir"] = str(seg_dir)
    attach_sample_metadata(adata, sample_id, sample_dir, metadata_csv)

    if "cluster" in adata.obs:
        adata.uns["default_label"] = "cluster"
        adata.uns["label_columns"] = ["cluster"]

    if build_graphs:
        contact = build_contact_graph(
            polys, centroids, mpp, contact_tolerance_um, max_edge_um
        )
        voronoi, regions = build_voronoi_graph(
            polys, centroids, mpp, clip_radius_um, max_edge_um
        )
        _store_graph(adata, contact, "contact")
        _store_graph(adata, voronoi, "voronoi")
        adata.uns["voronoi_regions"] = regions
        adata.obs["contact_degree"] = contact.degrees()
        adata.obs["voronoi_degree"] = voronoi.degrees()
        if wall_tolerance_um is not None:
            for prefix in ("contact", "voronoi"):
                add_apposed_wall(adata, prefix, wall_tolerance_um, polys)

    if hvg_file is not None or n_top_genes is not None:
        from discell.data.visium_hd import load_hvg_panel, select_hvgs, subset_to_panel

        if hvg_file is not None:
            panel = load_hvg_panel(hvg_file)
        else:
            panel = select_hvgs(
                adata, n_top_genes=n_top_genes, bin_size="segmented", sample_id=sample_id
            )
        polygons = adata.uns["polygons"]
        regions = adata.uns.get("voronoi_regions")
        adata = subset_to_panel(adata, panel)
        adata.uns["polygons"] = polygons  # geometry is per-cell, unaffected by gene subsetting
        if regions is not None:
            adata.uns["voronoi_regions"] = regions

    return adata


def _store_graph(adata, graph: CellGraph, prefix: str) -> None:
    """Write a graph into ``obsp`` as connectivity plus one matrix per metric."""
    adata.obsp[f"{prefix}_connectivities"] = graph.to_sparse(np.ones(len(graph)))
    adata.obsp[f"{prefix}_shared_wall_um"] = graph.to_sparse(graph.shared_wall_um)
    adata.obsp[f"{prefix}_centroid_dist_um"] = graph.to_sparse(graph.centroid_dist_um)
    adata.obsp[f"{prefix}_wall_dist_um"] = graph.to_sparse(graph.wall_dist_um)
    adata.uns[f"{prefix}_graph"] = {
        "kind": graph.kind,
        "n_edges": int(len(graph)),
        "mean_degree": float(graph.degrees().mean()),
        "isolated_fraction": float((graph.degrees() == 0).mean()),
        "median_shared_wall_um": float(np.median(graph.shared_wall_um)) if len(graph) else 0.0,
    }


def graph_edge_frame(adata, prefix: str = "voronoi") -> pd.DataFrame:
    """Edge list with all geometric weights, as a tidy frame."""
    import scipy.sparse as sp

    conn = sp.triu(adata.obsp[f"{prefix}_connectivities"], k=1).tocoo()
    i, j = conn.row, conn.col
    out = pd.DataFrame({"i": i, "j": j})
    out["cell_i"] = adata.obs_names[i]
    out["cell_j"] = adata.obs_names[j]
    metrics = ["shared_wall_um", "centroid_dist_um", "wall_dist_um"]
    if f"{prefix}_apposed_wall_um" in adata.obsp:
        metrics.append("apposed_wall_um")
    for metric in metrics:
        matrix = adata.obsp[f"{prefix}_{metric}"].tocsr()
        out[metric] = np.asarray(matrix[i, j]).ravel()
    return out


# --------------------------------------------------------------------------
# CLI and self-test
# --------------------------------------------------------------------------


def self_test(sample_path: str | Path) -> int:
    import scipy.sparse as sp

    print(f"\n=== segmented self-test: {Path(sample_path).name} ===\n")

    print("[1/7] load with graphs")
    adata = load_segmented_sample(sample_path)
    assert adata.n_obs > 0, "no cells loaded"
    print(f"      {adata.n_obs} cells x {adata.n_vars} genes")

    print("[2/7] expression and geometry match")
    polys = adata.uns["polygons"]
    assert len(polys) == adata.n_obs, f"{len(polys)} polygons vs {adata.n_obs} cells"
    ids = parse_cell_ids(adata.obs_names)
    assert (ids == adata.obs["cell_id"].to_numpy()).all(), "cell_id/barcode mismatch"
    # Tested with distance rather than contains(): read_polygons applies
    # set_precision, and GEOS carries that precision model through buffer(), so
    # a buffer-then-contains check spuriously rejects boundary-coincident
    # centroids. obsm['spatial'] is float32, hence the 1e-3 px slack.
    from shapely.geometry import Point

    cents = adata.obsm["spatial"]
    outside = [k for k in range(adata.n_obs) if polys[k].distance(Point(*cents[k])) > 1e-3]
    # A centroid outside its own outline is correct behaviour for a concave
    # cell, so this is a rate check, not an absolute one.
    assert len(outside) / adata.n_obs < 0.01, (
        f"{len(outside)}/{adata.n_obs} centroids outside their polygon -- too many to be concavity"
    )
    reps = adata.obs[["rep_x_px", "rep_y_px"]].to_numpy()
    rep_outside = [k for k in range(adata.n_obs) if polys[k].distance(Point(*reps[k])) > 1e-6]
    assert not rep_outside, f"{len(rep_outside)} representative points outside their polygon"
    print(
        f"      {len(polys)} polygons aligned; all {adata.n_obs} representative points inside"
        f" ({len(outside)} concave cells have centroid outside)"
    )

    print("[3/7] counts are raw")
    data = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X).ravel()
    assert data.min() >= 0 and np.allclose(data[:100_000], np.round(data[:100_000]))
    print(f"      total UMIs = {int(adata.X.sum()):,}")

    print("[4/7] graphs present and symmetric")
    for prefix in ("contact", "voronoi"):
        conn = adata.obsp[f"{prefix}_connectivities"]
        assert (conn != conn.T).nnz == 0, f"{prefix} graph is not symmetric"
        info = adata.uns[f"{prefix}_graph"]
        print(
            f"      {prefix:8s} edges={info['n_edges']:7d} "
            f"mean_deg={info['mean_degree']:.2f} isolated={100*info['isolated_fraction']:.1f}% "
            f"wall_med={info['median_shared_wall_um']:.2f}um"
        )

    print("[5/7] Voronoi is a real tessellation, contact is not")
    v, c = adata.uns["voronoi_graph"], adata.uns["contact_graph"]
    assert v["mean_degree"] > c["mean_degree"], "Voronoi should be denser than contact"
    assert (adata.obsp["voronoi_shared_wall_um"].data > 0).all(), "zero-length Voronoi wall"
    print(f"      voronoi {v['mean_degree']:.2f} vs contact {c['mean_degree']:.2f} mean degree")

    print("[6/7] edge metrics are sane")
    edges = graph_edge_frame(adata, "voronoi")
    assert (edges["shared_wall_um"] > 0).all(), "non-positive shared wall"
    assert (edges["centroid_dist_um"] > 0).all(), "non-positive centroid distance"
    assert (edges["wall_dist_um"] >= 0).all(), "negative wall distance"
    assert (edges["wall_dist_um"] <= edges["centroid_dist_um"] + 1e-6).all(), \
        "wall distance exceeds centroid distance"
    print(
        f"      {len(edges)} edges | wall {edges['shared_wall_um'].median():.2f}um "
        f"| centroid {edges['centroid_dist_um'].median():.2f}um "
        f"| gap {edges['wall_dist_um'].median():.2f}um"
    )

    print("[7/7] cluster labels")
    if "cluster" in adata.obs:
        print(f"      {adata.obs['cluster'].nunique()} clusters from graphclust annotation")
    else:
        print("      (no annotated geojson for this sample)")

    print("\nAll checks passed.\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", default=str(DEFAULT_DATA_ROOT / DEFAULT_TEST_SAMPLE))
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, choices=("filtered", "raw"))
    parser.add_argument("--polygon-kind", default=DEFAULT_POLYGON_KIND, choices=("cell", "nucleus"))
    parser.add_argument("--contact-tolerance-um", type=float, default=DEFAULT_CONTACT_TOLERANCE_UM)
    parser.add_argument("--clip-radius-um", type=float, default=DEFAULT_CLIP_RADIUS_UM)
    parser.add_argument("--max-edge-um", type=float, default=DEFAULT_MAX_EDGE_UM)
    parser.add_argument("--wall-tolerance-um", type=float, default=DEFAULT_WALL_TOLERANCE_UM,
                        help="how close two membranes must run to count as shared wall")
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--hvg-file", default=None)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default=None, help="write to .h5ad (polygons are dropped)")
    parser.add_argument("--edges-out", default=None, help="write the Voronoi edge list to .csv")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.self_test:
        return self_test(args.sample)

    adata = load_segmented_sample(
        args.sample,
        matrix=args.matrix,
        polygon_kind=args.polygon_kind,
        build_graphs=not args.no_graphs,
        contact_tolerance_um=args.contact_tolerance_um,
        clip_radius_um=args.clip_radius_um,
        max_edge_um=args.max_edge_um,
        wall_tolerance_um=args.wall_tolerance_um,
        n_top_genes=args.n_top_genes,
        hvg_file=args.hvg_file,
        cache_dir=args.cache_dir,
    )
    print(adata)

    if args.edges_out:
        graph_edge_frame(adata, "voronoi").to_csv(args.edges_out, index=False)
        log.info("Wrote %s", args.edges_out)
    if args.out:
        out = adata.copy()
        for key in ("polygons", "voronoi_regions"):
            out.uns.pop(key, None)  # shapely objects are not h5ad-serialisable
        out.write_h5ad(args.out)
        log.info("Wrote %s (polygons dropped; re-derive with read_polygons)", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
