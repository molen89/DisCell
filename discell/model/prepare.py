#!/usr/bin/env python3
"""The graph, tiles and rings DisCell trains on.

One graph serves every role -- the GAT's neighbourhoods, the leak kernel ``beta``
and the composition ``y`` are all built from the same pruned edge list, because
the spec's halo and leak kernel must agree edge-for-edge or the two-hop batch
assembly silently diverges from the generative model.

Batching is by **contiguous spatial tile**, not scattered seeds. A tile's seeds
plus two graph hops give everything one training step needs exactly:

    ring1 = neighbours of seeds      -- need c_j, w_j, rho_j (for rho_bar at seeds)
    ring2 = neighbours of ring1      -- provide sg mu_z as GAT sources for ring1

Because tiles are contiguous, the rings only grow at the perimeter, so the
exact two-hop computation costs ~10% extra nodes instead of the ~100x a
scattered seed set would -- which is what removes the need for a cached-rho
buffer and its staleness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from discell.data.priors import smoothing_weights

log = logging.getLogger("discell.model.prepare")

#: Delaunay connects cells across lumens, vessels and tears; the spec prunes at
#: 30-50 um. On the ovarian slide p95 of edge length is 27.4 um and 40 um drops
#: 1.11% of edges.
DEFAULT_MAX_EDGE_UM = 40.0

#: Contamination decay length (spec section 2), shared with the loader's beta.
DEFAULT_TAU_UM = 20.0


@dataclass
class ModelGraph:
    """The pruned graph with every derived quantity the model consumes."""

    n_cells: int
    edge_i: np.ndarray               # undirected, i < j, pruned
    edge_j: np.ndarray
    beta: np.ndarray                 # (E, 2): col 0 into edge_j, col 1 into edge_i
    in_edges: sp.csr_matrix          # rows = dst, cols = src, data = directed beta
    degrees: np.ndarray              # post-prune
    pruned_per_cell: np.ndarray      # QC: edges each cell lost to the length prune
    y: np.ndarray | None             # (N, K) neighbour composition, ego excluded
    ybar_t: np.ndarray | None        # (K, K) mean composition per type

    @property
    def isolated(self) -> np.ndarray:
        """Cells with no edges after pruning: ``c = 0`` with a flag, zero beta row."""
        return self.degrees == 0


def build_graph(edge_i: np.ndarray, edge_j: np.ndarray, face_um: np.ndarray,
                dist_um: np.ndarray, n_cells: int,
                type_index: np.ndarray | None = None, n_types: int | None = None,
                max_edge_um: float = DEFAULT_MAX_EDGE_UM,
                tau_um: float = DEFAULT_TAU_UM) -> ModelGraph:
    """Prune by length, then derive ``beta``, degrees, ``y`` and the QC column."""
    keep = dist_um <= max_edge_um
    pruned_per_cell = np.bincount(
        np.concatenate([edge_i[~keep], edge_j[~keep]]), minlength=n_cells)
    ei, ej = edge_i[keep], edge_j[keep]

    beta = smoothing_weights(face_um[keep], dist_um[keep], ei, ej, n_cells, tau_um)
    dst = np.concatenate([ej, ei])
    src = np.concatenate([ei, ej])
    directed = np.concatenate([beta[:, 0], beta[:, 1]])
    in_edges = sp.csr_matrix((directed, (dst, src)), shape=(n_cells, n_cells))
    degrees = np.diff(in_edges.indptr).astype(np.int32)

    n_isolated = int((degrees == 0).sum())
    n_weak = int((degrees == 1).sum())
    log.info("Graph: kept %d of %d edges (pruned %.2f%% over %g um); "
             "%d isolated, %d single-neighbour, max %d edges lost by one cell",
             len(ei), len(edge_i), 100 * (~keep).mean(), max_edge_um,
             n_isolated, n_weak, int(pruned_per_cell.max(initial=0)))

    y = ybar = None
    if type_index is not None:
        k = n_types or int(type_index.max()) + 1
        onehot = np.zeros((n_cells, k), dtype=np.float64)
        onehot[np.arange(n_cells), type_index] = 1.0
        adjacency = in_edges.copy()
        adjacency.data = np.ones_like(adjacency.data)
        counts = adjacency @ onehot
        y = np.zeros_like(counts, dtype=np.float32)
        connected = degrees > 0
        y[connected] = (counts[connected] / degrees[connected, None]).astype(np.float32)
        ybar = np.zeros((k, k), dtype=np.float32)
        for group in range(k):
            rows = (type_index == group) & connected
            if rows.any():
                ybar[group] = y[rows].mean(axis=0)

    return ModelGraph(n_cells, ei, ej, beta, in_edges, degrees,
                      pruned_per_cell.astype(np.int32), y, ybar)


def from_dataset(dataset, max_edge_um: float = DEFAULT_MAX_EDGE_UM,
                 tau_um: float = DEFAULT_TAU_UM) -> ModelGraph:
    """A :class:`ModelGraph` from an opened voronoi-graph ``CellGraphDataset``."""
    metrics = list(dataset.edge_metrics)
    return build_graph(
        edge_i=dataset.edge_i, edge_j=dataset.edge_j,
        face_um=dataset._pull_edges("shared_wall_um"),
        dist_um=dataset.edge_values[:, metrics.index("centroid_dist_um")],
        n_cells=dataset.adata.n_obs,
        type_index=dataset.type_index, n_types=dataset.n_types,
        max_edge_um=max_edge_um, tau_um=tau_um,
    )


# --------------------------------------------------------------------------
# tiles and rings
# --------------------------------------------------------------------------


def spatial_tiles(positions: np.ndarray, max_cells: int) -> list[np.ndarray]:
    """Partition cells into contiguous tiles of at most *max_cells*.

    Recursive median split along the longer axis: deterministic, exactly
    partitioning, and every tile is a rectangle -- which is what keeps the
    rings small relative to the seeds.
    """
    tiles: list[np.ndarray] = []
    stack = [np.arange(len(positions))]
    while stack:
        idx = stack.pop()
        if len(idx) <= max_cells:
            tiles.append(np.sort(idx))
            continue
        span = positions[idx].max(axis=0) - positions[idx].min(axis=0)
        axis = int(span[1] > span[0])
        order = idx[np.argsort(positions[idx, axis], kind="stable")]
        mid = len(order) // 2
        stack += [order[:mid], order[mid:]]
    return tiles


def rings(in_edges: sp.csr_matrix, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(ring1, ring2)``: one and two graph hops out from *seeds*, disjoint."""
    hop1 = np.unique(in_edges[seeds].indices)
    ring1 = np.setdiff1d(hop1, seeds)
    if not len(ring1):
        return ring1, ring1
    hop2 = np.unique(in_edges[ring1].indices)
    ring2 = np.setdiff1d(hop2, np.union1d(seeds, ring1))
    return ring1, ring2


@dataclass
class TileBatch:
    """One training step's worth of indices, in the layout [seeds | ring1 | ring2].

    ``gat_*`` are the directed in-edges of every node that needs a context
    vector (seeds and ring1); ``leak_*`` are the prefix of those belonging to
    seeds, carrying the beta weights for ``rho_bar``. All local indices.
    """

    nodes: np.ndarray                # global cell indices
    n_seeds: int
    n_ring1: int
    gat_src: np.ndarray
    gat_dst: np.ndarray
    leak_src: np.ndarray
    leak_dst: np.ndarray
    leak_beta: np.ndarray

    @property
    def seeds(self) -> np.ndarray:
        return self.nodes[:self.n_seeds]

    @property
    def n_context(self) -> int:
        """Nodes needing a context vector: seeds plus ring1."""
        return self.n_seeds + self.n_ring1


def tile_batch(graph: ModelGraph, seeds: np.ndarray) -> TileBatch:
    """Assemble one tile: rings, then both edge lists in local coordinates.

    The GAT edges and the leak edges come from one CSR slice: rows are taken in
    [seeds | ring1] order, so the leak edges are exactly the first
    ``n_seeds`` rows' entries -- same edges, same order, beta attached.
    """
    seeds = np.asarray(seeds)
    ring1, ring2 = rings(graph.in_edges, seeds)
    nodes = np.concatenate([seeds, ring1, ring2])

    local = np.full(graph.n_cells, -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))

    context = nodes[:len(seeds) + len(ring1)]
    sub = graph.in_edges[context]
    gat_dst = np.repeat(np.arange(len(context)), np.diff(sub.indptr))
    gat_src = local[sub.indices]
    if (gat_src < 0).any():          # a source outside seeds+rings: rings are wrong
        raise AssertionError("edge source outside the two-hop halo")

    n_leak = int(sub.indptr[len(seeds)])
    return TileBatch(
        nodes=nodes, n_seeds=len(seeds), n_ring1=len(ring1),
        gat_src=gat_src, gat_dst=gat_dst,
        leak_src=gat_src[:n_leak], leak_dst=gat_dst[:n_leak],
        leak_beta=sub.data[:n_leak],
    )


# --------------------------------------------------------------------------
# the assembled training data
# --------------------------------------------------------------------------


def soft_clusters(points: np.ndarray, k: int, seed: int = 0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Soft k-means memberships: ``softmax(-d^2 / T)``, T the median nearest d^2.

    The spec's ``e_Phi`` (4.6): fit once, freeze, and keep assignments *soft* --
    a smoother CE target with no boundary artefacts. ``E_Phi = K`` so the two
    adversary heads face comparable difficulty and one alpha serves both.
    """
    from sklearn.cluster import KMeans

    centres = KMeans(k, n_init=4, random_state=seed).fit(points).cluster_centers_
    d2 = ((points[:, None, :] - centres[None]) ** 2).sum(-1)
    temperature = max(float(np.median(d2.min(axis=1))), 1e-8)
    logits = -d2 / temperature
    logits -= logits.max(axis=1, keepdims=True)
    soft = np.exp(logits)
    return (soft / soft.sum(axis=1, keepdims=True)).astype(np.float32), centres


@dataclass
class ModelData:
    """Everything one DisCell fit consumes, resident and split.

    ``v_block`` is the invariance target ``[y minus one column, PCs(Phi)]`` --
    one y column dropped because the simplex constraint makes the full block
    singular, Phi as a few PCs because the penalty conditions on their joint
    covariance (spec 4.6). ``phi`` itself stays at full dimension unless
    *phi_pca* asked otherwise; the compression decision belongs to the caller.
    """

    graph: ModelGraph
    x: "object"                   # scipy CSR, rows gathered per tile
    t: np.ndarray
    phi: np.ndarray
    positions: np.ndarray
    totals: np.ndarray
    median_counts: float
    p_t: np.ndarray
    type_names: np.ndarray
    v_block: np.ndarray
    vbar_t: np.ndarray            # per-type mean of v_block, probe baseline
    train_tiles: list[np.ndarray]
    val_tiles: list[np.ndarray]
    # the adversary's targets (spec 4.6 escalation); None under closed_form
    e_phi: np.ndarray | None = None       # (N, E_Phi) soft memberships
    phibar_t: np.ndarray | None = None    # (K, E_Phi) mean membership per type
    gene_names: np.ndarray | None = None  # panel gene symbols, decoder order
    #: Tirosh cell-cycle scores + MKI67-ranked cycling types; None when the
    #: panel lacks the markers (synthetic data)
    cycle: dict | None = None

    @property
    def n_cells(self) -> int:
        return self.graph.n_cells


def assemble(dataset, variant: str, embeddings: str,
             tile_cells: int = 4096, val_fraction: float = 0.15,
             phi_pca: int | None = None, v_pcs: int = 12,
             max_edge_um: float = DEFAULT_MAX_EDGE_UM,
             tau_um: float = DEFAULT_TAU_UM, seed: int = 0,
             label_key: str | None = None) -> ModelData:
    """Open a bundle and build the tensors, tiles and split for one fit.

    *label_key* selects the obs column that becomes ``t`` (None: the bundle's
    default label, i.e. curated ``cell_group`` when present). Everything
    conditioned on type -- y, rho_bar, the adversary, K itself -- follows it.
    """
    from discell import paths
    from discell.data.embeddings import load_embeddings
    from discell.data.loader import CellGraphDataset

    ds = paths.dataset(dataset)
    opened = CellGraphDataset.from_dataset(ds, variant, graph="voronoi",
                                           label_key=label_key)
    graph = from_dataset(opened, max_edge_um=max_edge_um, tau_um=tau_um)

    phi, found = load_embeddings(ds.embeddings_file(embeddings),
                                 opened.cell_ids, dataset=ds.dataset_id)
    if not found.all():
        log.warning("%d cells lack an image embedding (zeros)", int((~found).sum()))

    # Independent streams per purpose: with one shared generator, toggling
    # phi_pca would shift the draws downstream and silently change the
    # train/val split -- runs of a sweep must share one split.
    rng_pca = np.random.default_rng([seed, 1])
    rng_split = np.random.default_rng([seed, 2])
    if phi_pca is not None:
        from sklearn.decomposition import PCA

        rows = rng_pca.choice(len(phi), min(len(phi), 50_000), replace=False)
        phi = PCA(phi_pca, random_state=seed).fit(phi[rows]).transform(phi)
        log.info("Phi compressed to %d PCs", phi_pca)

    from sklearn.decomposition import PCA

    rows = rng_pca.choice(len(phi), min(len(phi), 50_000), replace=False)
    phi_pcs = PCA(v_pcs, random_state=seed).fit(phi[rows]).transform(phi)
    v_block = np.hstack([graph.y[:, :-1], phi_pcs]).astype(np.float32)

    t = opened.type_index
    k = opened.n_types
    # connected cells only, matching ybar_t: an isolated cell's zero y row is
    # an artefact of having no neighbours, not a niche observation
    connected = graph.degrees > 0
    vbar_t = np.stack([
        v_block[(t == g) & connected].mean(axis=0)
        if ((t == g) & connected).any()
        else np.zeros(v_block.shape[1], dtype=np.float32)
        for g in range(k)])

    e_phi, _ = soft_clusters(phi_pcs, k, seed=seed)      # E_Phi = K (spec 4.6)
    phibar_t = np.stack([
        e_phi[(t == g) & connected].mean(axis=0) if ((t == g) & connected).any()
        else np.full(k, 1.0 / k, dtype=np.float32) for g in range(k)])

    tiles = spatial_tiles(opened.positions_um, tile_cells)
    order = rng_split.permutation(len(tiles))
    n_val = max(1, int(round(val_fraction * len(tiles))))
    val = [tiles[i] for i in order[:n_val]]
    train = [tiles[i] for i in order[n_val:]]
    log.info("Tiles: %d train, %d val (of %d cells)", len(train), len(val),
             graph.n_cells)

    x = opened.counts.astype(np.float32)
    totals = np.asarray(x.sum(axis=1)).ravel()

    from discell.model.cell_cycle import cycling_type_ranking, score_cell_cycle

    cycle = score_cell_cycle(x, opened.gene_names)
    if cycle is not None:
        from discell.model.cell_cycle import expression_pcs

        cycle["cycling_types"] = cycling_type_ranking(
            x, opened.gene_names, t, k).tolist()
        # the probe ceiling: what the counts themselves can say about cycle
        cycle["x_pcs"] = expression_pcs(x, seed=seed)
    return ModelData(
        graph=graph, x=x, t=t.astype(np.int64),
        phi=phi.astype(np.float32), positions=opened.positions_um,
        totals=totals, median_counts=float(np.median(totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=opened.type_names, v_block=v_block, vbar_t=vbar_t,
        train_tiles=train, val_tiles=val, e_phi=e_phi, phibar_t=phibar_t,
        cycle=cycle, gene_names=np.asarray(opened.gene_names),
    )
