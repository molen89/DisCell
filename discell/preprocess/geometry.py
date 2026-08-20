#!/usr/bin/env python3
"""Cell polygons in, neighbour graphs out.

Platform-agnostic geometry: everything here takes shapely polygons and cell
centroids, and knows nothing about how they were read off disk.

Two graphs, because they answer different questions:

* **contact** -- cells whose polygons touch, within a tolerance. Sparse and
  literal; it says who is physically adjacent.
* **voronoi** -- the tessellation of the centroids, clipped to a radius. Dense
  and complete; it says who *would* be adjacent if the cells filled the space.

Both carry the same edge metrics in microns, so a model can train on either.

The metric that matters is :func:`apposed_wall_um`: how much boundary two cells
actually share, measured on the real polygons rather than on the tessellation.
On the ovarian slide Voronoi ``shared_wall_um`` correlates with true contact at
only r = -0.107, while ``apposed_wall_um`` reaches r = -0.51.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("discell.preprocess.geometry")

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
    other. Faithful, but sparse: these polygons are independently
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
