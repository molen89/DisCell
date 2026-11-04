"""The clipped-Voronoi face graph of the cell centroids.

Two cells are neighbors when their Voronoi regions, each clipped to a disc of
``clip_radius_um`` about its centroid, share a boundary of positive length. Each
edge carries the face length (the shared boundary, in micrometres) and the
centroid distance. Candidate pairs come from the Delaunay triangulation, with
edges longer than ``max_candidate_um`` dropped first (they bridge tissue holes);
the model's own 40 um prune is applied later, by :func:`discell.build_graph`.

Only centroids are needed: segmentation polygons do not enter the graph.

The face test is an overlay of two clipped regions, and GEOS's overlay is not
invariant to the coordinate frame: on the TMA core (69,422 cells), moving the
centroids by one ulp (micrometres divided by the pixel size instead of the
original pixel values) changes about 60 of 205,929 edges, and computing in
micrometres instead of pixels about 330 (faces up to ~4 um appear or vanish).
The graph is therefore computed in the frame the centroids were measured in:
:func:`discell.read_xenium` keeps the pixel centroids in ``.obsm['spatial_px']``
and :func:`add_voronoi_graph` uses them when present, which reproduces the
paper's graphs exactly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

#: Voronoi regions are clipped to a disc of this radius about their centroid, so
#: a cell at the tissue edge cannot claim unbounded territory.
DEFAULT_CLIP_RADIUS_UM = 30.0
#: Delaunay edges longer than this are not considered as candidate neighbors.
DEFAULT_MAX_CANDIDATE_UM = 100.0


def candidate_pairs(centroids: np.ndarray, max_edge: float) -> np.ndarray:
    """Delaunay edges no longer than *max_edge*, ``(n, 2)`` int64 with ``i < j``."""
    from scipy.spatial import Delaunay

    n = len(centroids)
    if n < 4:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    simplices = Delaunay(centroids).simplices
    edges = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0).astype(np.int64)
    lengths = np.linalg.norm(centroids[edges[:, 0]] - centroids[edges[:, 1]], axis=1)
    return edges[lengths <= max_edge]


def shared_boundary(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Length of the boundary each pair of geometries has in common.

    Regions that meet along an edge share that edge's length. Regions that
    overlap in area (which a partition should not produce) are scored by the
    intersection of their boundaries, so an overlap does not read as a long face.
    """
    import shapely

    out = np.zeros(len(a), dtype=np.float64)
    touching = ~shapely.disjoint(a, b)
    if not touching.any():
        return out
    inter = shapely.intersection(a[touching], b[touching])
    length = np.where(shapely.is_empty(inter), 0.0, shapely.length(inter))
    overlap = shapely.area(inter) > 0
    if overlap.any():
        rows = np.flatnonzero(overlap)
        walls = shapely.intersection(shapely.boundary(a[touching][rows]),
                                     shapely.boundary(b[touching][rows]))
        length[rows] = np.where(shapely.is_empty(walls), 0.0, shapely.length(walls))
    out[touching] = length
    return out


def voronoi_edges(centroids: np.ndarray, clip_radius_um: float | None = DEFAULT_CLIP_RADIUS_UM,
                  max_candidate_um: float = DEFAULT_MAX_CANDIDATE_UM, unit_um: float = 1.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(edge_i, edge_j, face_um, dist_um)`` of the clipped-Voronoi graph.

    *centroids* is ``(N, 2)`` in units of *unit_um* micrometres (1 for
    micrometres, the pixel size for pixels); the geometry is computed in that
    frame and face lengths and distances are returned in micrometres. Each
    undirected edge is listed once with ``edge_i < edge_j``.
    ``clip_radius_um=None`` leaves the regions unclipped.
    """
    import shapely

    unit = float(unit_um)
    pts = np.asarray(centroids, dtype=np.float64)
    points = shapely.points(pts)
    regions = shapely.get_parts(shapely.voronoi_polygons(shapely.multipoints(points),
                                                         tolerance=0.0))
    # The diagram does not keep the input order: give each centroid the region
    # that contains it.
    owner = np.full(len(pts), None, dtype=object)
    point_idx, region_idx = shapely.STRtree(regions).query(points, predicate="within")
    first = np.unique(point_idx, return_index=True)[1]
    owner[point_idx[first]] = regions[region_idx[first]]
    if clip_radius_um:
        has = np.flatnonzero(owner != None)  # noqa: E711 (object array)
        discs = shapely.buffer(points[has], float(clip_radius_um) / unit, quad_segs=16)
        owner[has] = shapely.intersection(owner[has], discs)

    pairs = candidate_pairs(pts, float(max_candidate_um) / unit)
    ga, gb = owner[pairs[:, 0]], owner[pairs[:, 1]]
    usable = np.array([g is not None for g in ga]) & np.array([g is not None for g in gb])
    usable[usable] = ~(shapely.is_empty(ga[usable]) | shapely.is_empty(gb[usable]))
    face = np.zeros(len(pairs), dtype=np.float64)
    face[usable] = shared_boundary(ga[usable], gb[usable])
    keep = face > 0
    ei, ej = pairs[keep, 0], pairs[keep, 1]
    dist = np.linalg.norm(pts[ei] - pts[ej], axis=1) * unit
    return ei, ej, face[keep] * unit, dist


def add_voronoi_graph(adata, *, clip_radius_um: float | None = DEFAULT_CLIP_RADIUS_UM,
                      max_candidate_um: float = DEFAULT_MAX_CANDIDATE_UM,
                      spatial_key: str = "spatial", prefix: str = "voronoi") -> int:
    """Compute the clipped-Voronoi graph of the centroids and store it.

    Writes the symmetric ``.obsp`` matrices ``<prefix>_connectivities``,
    ``<prefix>_shared_wall_um`` (face length) and ``<prefix>_centroid_dist_um``,
    and ``.uns['discell']['voronoi']``. The centroids are
    ``.obsm['spatial_px']`` with ``.uns['discell']['xenium']['pixel_size_um']``
    when both are present (as :func:`discell.read_xenium` writes them),
    otherwise ``.obsm[spatial_key]`` in micrometres, or in pixels when
    ``.uns['discell']['microns_per_pixel']`` is set. Returns the number of edges.
    """
    meta = adata.uns.get("discell", {})
    pixel_size = meta.get("xenium", {}).get("pixel_size_um")
    if "spatial_px" in adata.obsm and pixel_size is not None:
        source, unit = "spatial_px", float(pixel_size)
    elif meta.get("microns_per_pixel") is not None:
        source, unit = spatial_key, float(meta["microns_per_pixel"])
    else:
        source, unit = spatial_key, 1.0
    pos = np.asarray(adata.obsm[source], dtype=np.float64)
    ei, ej, face, dist = voronoi_edges(pos, clip_radius_um, max_candidate_um, unit)
    n = adata.n_obs
    rows, cols = np.concatenate([ei, ej]), np.concatenate([ej, ei])

    def symmetric(values: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix((np.concatenate([values, values]), (rows, cols)), shape=(n, n))

    adata.obsp[f"{prefix}_connectivities"] = symmetric(np.ones(len(ei)))
    adata.obsp[f"{prefix}_shared_wall_um"] = symmetric(face)
    adata.obsp[f"{prefix}_centroid_dist_um"] = symmetric(dist)
    adata.uns.setdefault("discell", {})["voronoi"] = {
        "prefix": prefix, "spatial_key": source, "unit_um": unit,
        "clip_radius_um": None if clip_radius_um is None else float(clip_radius_um),
        "max_candidate_um": float(max_candidate_um), "n_edges": int(len(ei)),
    }
    return int(len(ei))
