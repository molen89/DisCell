"""The pruned cell graph, the spillover kernel ``beta`` and the composition ``y``.

One edge list serves the GAT neighborhoods, the spillover kernel and the neighbor
composition, so the batch halo and the kernel agree edge for edge.

Input is the unpruned Voronoi face graph of the cell centroids, stored in
``.obsp`` under ``<prefix>_connectivities``, ``<prefix>_shared_wall_um`` (face
length) and ``<prefix>_centroid_dist_um``; when it is absent, :func:`build_graph`
first computes it from ``.obsm['spatial']`` (:mod:`discell.graph.voronoi`).
:func:`build_graph` writes::

    .obsp['discell_connectivities']   pruned, undirected, 0/1
    .obsp['discell_face_um']          face length on the kept edges
    .obsp['discell_dist_um']          centroid distance on the kept edges
    .obsp['discell_beta']             directed, rows = receiver, row-normalized
    .obsm['discell_y']                neighbor type fractions, ego excluded
    .obs['discell_isolated']          no edge after pruning
    .uns['discell']['graph']          parameters and summary
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from discell.graph.voronoi import DEFAULT_CLIP_RADIUS_UM, add_voronoi_graph
from discell.io.anndata import UNASSIGNED, fold_labels

#: Edges longer than this are dropped (they cross lumens, vessels and tears).
DEFAULT_MAX_EDGE_UM = 40.0
#: Decay length of the spillover kernel.
DEFAULT_TAU_UM = 20.0
GRAPH_VERSION = 1


@dataclass
class ModelGraph:
    """The pruned graph and every derived quantity the model consumes."""

    n_cells: int
    edge_i: np.ndarray           # undirected, i < j
    edge_j: np.ndarray
    beta: np.ndarray             # (E, 2) float32: col 0 into edge_j, col 1 into edge_i
    in_edges: sp.csr_matrix      # rows = receiver, cols = sender, data = beta
    degrees: np.ndarray          # (N,) int32
    y: np.ndarray | None         # (N, K) float32
    ybar_t: np.ndarray | None    # (K, K) float32, mean y per type (connected cells)

    @property
    def isolated(self) -> np.ndarray:
        """Cells with no edge after pruning."""
        return self.degrees == 0


def smoothing_weights(face_um: np.ndarray, dist_um: np.ndarray, edge_i: np.ndarray,
                      edge_j: np.ndarray, n_cells: int,
                      tau_um: float = DEFAULT_TAU_UM) -> np.ndarray:
    """``beta_ij = face_ij exp(-d_ij / tau)``, normalized over each receiver's in-edges.

    Returns ``(n_edges, 2)`` float32: column 0 is the weight into ``edge_j``,
    column 1 into ``edge_i``. A cell without edges keeps an all-zero row.
    """
    raw = face_um * np.exp(-dist_um / float(tau_um))
    total = np.zeros(n_cells, dtype=np.float64)
    np.add.at(total, edge_i, raw)
    np.add.at(total, edge_j, raw)
    safe = np.maximum(total, 1e-12)
    return np.stack([raw / safe[edge_j], raw / safe[edge_i]], axis=1).astype(np.float32)


def composition(in_edges: sp.csr_matrix, degrees: np.ndarray, type_index: np.ndarray,
                n_types: int) -> tuple[np.ndarray, np.ndarray]:
    """``(y, ybar_t)``: neighbor type fractions per cell and their mean per type.

    ``y`` excludes the cell itself and is zero for isolated cells; ``ybar_t``
    averages connected cells only.
    """
    n_cells = len(degrees)
    onehot = np.zeros((n_cells, n_types), dtype=np.float64)
    onehot[np.arange(n_cells), type_index] = 1.0
    adjacency = in_edges.copy()
    adjacency.data = np.ones_like(adjacency.data)
    counts = adjacency @ onehot
    y = np.zeros_like(counts, dtype=np.float32)
    connected = degrees > 0
    y[connected] = (counts[connected] / degrees[connected, None]).astype(np.float32)
    ybar = np.zeros((n_types, n_types), dtype=np.float32)
    for group in range(n_types):
        rows = (type_index == group) & connected
        if rows.any():
            ybar[group] = y[rows].mean(axis=0)
    return y, ybar


def graph_from_edges(edge_i: np.ndarray, edge_j: np.ndarray, face_um: np.ndarray,
                     dist_um: np.ndarray, n_cells: int,
                     type_index: np.ndarray | None = None, n_types: int | None = None,
                     max_edge_um: float = DEFAULT_MAX_EDGE_UM,
                     tau_um: float = DEFAULT_TAU_UM) -> ModelGraph:
    """Prune by centroid distance, then derive ``beta``, degrees, ``y`` and ``ybar_t``.

    *edge_i*, *edge_j* list each undirected edge once. ``y`` and ``ybar_t``
    are None when no *type_index* is given.
    """
    keep = dist_um <= max_edge_um
    ei, ej = edge_i[keep], edge_j[keep]
    beta = smoothing_weights(face_um[keep], dist_um[keep], ei, ej, n_cells, tau_um)
    dst = np.concatenate([ej, ei])
    src = np.concatenate([ei, ej])
    directed = np.concatenate([beta[:, 0], beta[:, 1]])
    in_edges = sp.csr_matrix((directed, (dst, src)), shape=(n_cells, n_cells))
    degrees = np.diff(in_edges.indptr).astype(np.int32)
    y = ybar = None
    if type_index is not None:
        y, ybar = composition(in_edges, degrees, type_index,
                              n_types or int(type_index.max()) + 1)
    return ModelGraph(n_cells, ei, ej, beta, in_edges, degrees, y, ybar)


def voronoi_edges(adata, prefix: str = "voronoi"
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(edge_i, edge_j, face_um, dist_um)`` of the unpruned graph in ``.obsp``.

    Each undirected edge appears once (upper triangle, row-major order).
    """
    keys = [f"{prefix}_{name}" for name in
            ("connectivities", "shared_wall_um", "centroid_dist_um")]
    missing = [k for k in keys if k not in adata.obsp]
    if missing:
        raise KeyError(f"adata.obsp lacks {missing}: the Voronoi face graph of the "
                       "centroids must be computed first")
    upper = sp.triu(adata.obsp[keys[0]], k=1).tocoo()
    edge_i, edge_j = upper.row, upper.col
    face = np.asarray(adata.obsp[keys[1]].tocsr()[edge_i, edge_j]).ravel()
    dist = np.asarray(adata.obsp[keys[2]].tocsr()[edge_i, edge_j]).ravel()
    return edge_i, edge_j, face, dist


def build_graph(adata, label_key: str, *, max_edge_um: float = DEFAULT_MAX_EDGE_UM,
                tau_um: float = DEFAULT_TAU_UM, prefix: str = "voronoi",
                unassigned_label: str = UNASSIGNED,
                clip_radius_um: float | None = DEFAULT_CLIP_RADIUS_UM) -> ModelGraph:
    """Build the model graph of *adata* and write it into *adata* (see module doc).

    *label_key* is the ``.obs`` column of cell types; missing labels count as
    *unassigned_label*. The columns of ``discell_y`` follow the sorted type
    names, recorded in ``.uns['discell']['graph']['types']``. When
    ``.obsp['<prefix>_connectivities']`` is absent, the Voronoi face graph of
    ``.obsm['spatial']`` is computed first with regions clipped at
    *clip_radius_um* and stored under *prefix*.
    """
    if f"{prefix}_connectivities" not in adata.obsp:
        add_voronoi_graph(adata, clip_radius_um=clip_radius_um, prefix=prefix)
    edge_i, edge_j, face, dist = voronoi_edges(adata, prefix)
    type_index, type_names = fold_labels(adata.obs[label_key], unassigned_label)
    n = adata.n_obs
    graph = graph_from_edges(edge_i, edge_j, face, dist, n, type_index,
                             len(type_names), max_edge_um, tau_um)
    keep = dist <= max_edge_um

    def symmetric(values: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix(
            (np.concatenate([values, values]),
             (np.concatenate([graph.edge_i, graph.edge_j]),
              np.concatenate([graph.edge_j, graph.edge_i]))), shape=(n, n))

    adata.obsp["discell_connectivities"] = symmetric(
        np.ones(len(graph.edge_i), dtype=np.float32))
    adata.obsp["discell_face_um"] = symmetric(face[keep])
    adata.obsp["discell_dist_um"] = symmetric(dist[keep])
    adata.obsp["discell_beta"] = graph.in_edges
    adata.obsm["discell_y"] = graph.y
    adata.obs["discell_isolated"] = graph.isolated
    uns = adata.uns.setdefault("discell", {})
    uns["graph"] = {
        "source": prefix, "label_key": label_key, "unassigned_label": unassigned_label,
        "types": [str(v) for v in type_names], "max_edge_um": float(max_edge_um),
        "tau_um": float(tau_um), "n_edges": int(len(graph.edge_i)),
        "n_pruned": int((~keep).sum()),
        "isolated_fraction": float(graph.isolated.mean()), "version": GRAPH_VERSION,
    }
    return graph


def read_graph(adata, label_key: str | None = None) -> ModelGraph:
    """The :class:`ModelGraph` that :func:`build_graph` stored in *adata*.

    Checks that the labels still fold to the stored type vocabulary.
    """
    meta = adata.uns.get("discell", {}).get("graph")
    if meta is None or "discell_beta" not in adata.obsp:
        raise KeyError("no DisCell graph in adata: run discell.build_graph first")
    if label_key is not None and label_key != meta["label_key"]:
        raise ValueError(f"graph was built with label_key={meta['label_key']!r}, "
                         f"not {label_key!r}")
    type_index, type_names = fold_labels(adata.obs[meta["label_key"]],
                                         meta["unassigned_label"])
    if [str(v) for v in type_names] != list(meta["types"]):
        raise ValueError("the labels no longer match the types the graph was built with")
    in_edges = sp.csr_matrix(adata.obsp["discell_beta"])
    upper = sp.triu(adata.obsp["discell_connectivities"], k=1).tocoo()
    edge_i, edge_j = upper.row, upper.col
    beta = np.stack([np.asarray(in_edges[edge_j, edge_i]).ravel(),
                     np.asarray(in_edges[edge_i, edge_j]).ravel()],
                    axis=1).astype(np.float32)
    degrees = np.diff(in_edges.indptr).astype(np.int32)
    y = np.asarray(adata.obsm["discell_y"], dtype=np.float32)
    _, ybar = composition(in_edges, degrees, type_index, len(type_names))
    return ModelGraph(adata.n_obs, edge_i, edge_j, beta, in_edges, degrees, y, ybar)
