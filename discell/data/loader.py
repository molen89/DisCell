#!/usr/bin/env python3
"""Batch loader over a saved cell-graph bundle.

Serves subgraph batches for a model that predicts a cell from its neighbourhood:
raw counts, the neighbour graph with geometric edge features, one-hot cell-type
labels, and a per-cell image embedding.

Construction loads the bundle, builds the one-hot labels, and computes the fixed
neighbourhood composition table: for each cell type, the mean distribution of
its neighbours' types, which does not change across batches. Every edge of the
chosen graph is served -- the metrics needed to filter one are in ``edge_attr``,
so a consumer that wants a subset can take it without the bundle losing it.

See :mod:`discell.data.batch` for what a batch carries, and
:attr:`CellGraphDataset.constant` for the arrays that do not.

Usage::

    from discell.data.loader import CellGraphDataset

    data = CellGraphDataset.from_dataset("<dataset id>", embeddings="full_v1")
    for batch in data.batches():
        ...
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from discell.data.batch import DEFAULT_EDGE_METRICS, TorchBatch
from discell.data.embeddings import DEFAULT_EMBEDDING_DIM, load_embeddings
from discell.data.priors import (
    DEFAULT_BETA_TAU_UM,
    Constants,
    composition,
    smoothing_weights,
)

log = logging.getLogger("discell.data.loader")

#: Voronoi rather than contact: it is an exact planar partition, so every edge
#: has a positive shared face and ``beta`` is well defined. Xenium's segmented
#: polygons meet at a point, which leaves ``contact_shared_wall_um`` zero on 98%
#: of edges and ``beta`` all-zero for most cells.
DEFAULT_GRAPH = "voronoi"
DEFAULT_BATCH_CELLS = 64


#: Label for cells with no annotation. An explicit class rather than dropped, so
#: the one-hot rows always sum to 1 and node counts line up. Folded onto the
#: panel's own spelling when it has one -- 10x ships ``Unassigned`` in
#: ``cell_groups.csv``, and a second lowercase class meaning the same thing
#: would ask a model to separate them.
UNASSIGNED = "Unassigned"


class CellGraphDataset:
    """Subgraph batches over a saved bundle."""

    @classmethod
    def from_dataset(cls, dataset, variant: str = "full", **kwargs) -> "CellGraphDataset":
        """Open one dataset's bundle, resolving embeddings within that dataset.

        ``dataset`` may be a :class:`discell.paths.Dataset`, a dataset id, or a
        source sample path -- all spellings of one slide resolve alike, so a
        caller cannot accidentally pair this bundle with another's embeddings.
        """
        from discell import paths

        ds = paths.dataset(dataset)
        if "embeddings" in kwargs:
            kwargs["embeddings"] = ds.embeddings_file(kwargs["embeddings"])
        return cls(ds.bundle_dir, variant, dataset_id=ds.dataset_id, **kwargs)

    def __init__(
        self, bundle_dir: str | Path, name: str, graph: str = DEFAULT_GRAPH,
        label_key: str | None = None, batch_cells: int = DEFAULT_BATCH_CELLS,
        edge_metrics: Sequence[str] = DEFAULT_EDGE_METRICS,
        beta_tau_um: float = DEFAULT_BETA_TAU_UM,
        beta_distance_metric: str = "centroid_dist_um",
        embeddings: str | Path | None = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM, image_scope: str = "seeds",
        device: str | None = None, resident: str | None = None,
        counts_dtype: str = "float32", seed: int = 0, dataset_id: str | None = None,
    ) -> None:
        from discell.data.bundle import load_bundle

        self.adata = load_bundle(bundle_dir, name)
        self.graph = graph
        self.batch_cells = batch_cells
        # 'seeds': (n_seeds, dim) aligned to seed_index, images only for the
        # predicted cells. 'nodes': (n_nodes, dim) aligned to x, so neighbour
        # morphology can be message-passed like any other feature.
        self.image_scope = image_scope
        self.device = device
        self.rng = np.random.default_rng(seed)

        if f"{graph}_connectivities" not in self.adata.obsp:
            raise KeyError(f"no graph {graph!r} in bundle; have "
                           f"{sorted(k for k in self.adata.obsp if k.endswith('_connectivities'))}")

        self.label_key = label_key or self.adata.uns.get("default_label") or "cell_group"
        if self.label_key not in self.adata.obs:
            raise KeyError(f"label {self.label_key!r} not in obs")

        # Drop any metric this graph lacks rather than failing: an older bundle
        # may predate one of them.
        self.edge_metrics = tuple(m for m in edge_metrics
                                  if f"{graph}_{m}" in self.adata.obsp)
        missing = [m for m in edge_metrics if m not in self.edge_metrics]
        if missing:
            log.warning("%s not stored for %r -- edge_attr is %s",
                        ", ".join(missing), graph, list(self.edge_metrics))
        if not self.edge_metrics:
            raise KeyError(f"no metric of {list(edge_metrics)} stored for {graph!r}")

        # 0. the edge list and adjacency everything else is derived from.
        self._build_edges()

        # 1. labels, the composition table, and the per-cell soft niche label
        self._build_labels()
        self.neighbour_composition, self.niche, self.composition_support = composition(
            self.adjacency, self.labels, self.degrees, self.type_index, self.type_names)

        # 2. beta: constant for the run, so computed once rather than per batch
        face = self._pull_edges("shared_wall_um")
        self.beta_tau_um = float(beta_tau_um)
        beta = None if face is None else smoothing_weights(
            face, self._pull_edges(beta_distance_metric),
            self.edge_i, self.edge_j, self.adata.n_obs, beta_tau_um)
        if face is None:
            log.warning("no shared_wall_um for %r -- beta unavailable", graph)
        self._beta = beta

        self.gene_names = np.asarray(self.adata.var["gene_name"].astype(str))

        # counts as CSR once, so batch slicing is cheap
        import scipy.sparse as sp

        matrix = self.adata.X
        self.counts = sp.csr_matrix(matrix) if not sp.issparse(matrix) else matrix.tocsr()

        # Centroids in microns, carried for plotting and inspection rather than
        # as features: the distances a model needs are precomputed into the edge
        # attributes. Microns rather than pixels so the two platforms agree.
        mpp = float(self.adata.uns.get("microns_per_pixel", 1.0))
        if "spatial_um" in self.adata.obsm:
            self.positions_um = np.asarray(self.adata.obsm["spatial_um"], dtype=np.float32)
        else:
            self.positions_um = (np.asarray(self.adata.obsm["spatial"], dtype=np.float32)
                                 * mpp).astype(np.float32)

        self.cell_ids = np.asarray(self.adata.obs_names, dtype=str)
        # Prefer the id recorded in the bundle over one passed in, so a bundle
        # opened by raw path still checks its embeddings against the right slide.
        self.dataset_id = str(self.adata.uns.get("dataset_id") or dataset_id or "") or None
        if embeddings is not None:
            self.embeddings, self.embedding_mask = load_embeddings(
                embeddings, self.cell_ids, dataset=self.dataset_id)
        else:
            log.warning("no image embeddings given -- serving zeros of width %d", embedding_dim)
            self.embeddings = np.zeros((self.adata.n_obs, embedding_dim), dtype=np.float32)
            self.embedding_mask = np.zeros(self.adata.n_obs, dtype=bool)

        import torch

        self.constant = Constants(
            neighbour_composition=torch.as_tensor(
                np.ascontiguousarray(self.neighbour_composition.to_numpy(),
                                     dtype=np.float32)),
            niche=torch.from_numpy(self.niche),
            beta=None if self._beta is None else torch.from_numpy(self._beta),
            type_names=tuple(map(str, self.type_names)),
        )

        self.resident = None
        self._counts_t = self._labels_t = self._embeddings_t = None
        self._composition_t = self._pos_t = None
        if resident:
            self._make_resident(resident, counts_dtype)
        elif device:
            self.constant = self.constant.to(device)

        log.info(
            "Dataset: %d cells, %d genes, %d types, %d edges (%s graph), image dim %d",
            self.adata.n_obs, self.adata.n_vars, self.n_types,
            len(self.edge_i), graph, self.embeddings.shape[1],
        )

    def _make_resident(self, resident: str, counts_dtype: str) -> None:
        """Hold counts, labels and image embeddings as tensors on one device.

        Batches then cost an index_select instead of a sparse slice plus a
        host-to-device copy of every count row. Counts are densified: the
        gene axis is already subset, and a dense matrix is what makes the gather
        a single kernel.
        """
        import torch

        device = torch.device("cuda" if resident in ("gpu", "cuda") else resident)
        dtype = getattr(torch, counts_dtype)
        n, g = self.adata.n_obs, self.adata.n_vars
        needed = (n * g * torch.empty(0, dtype=dtype).element_size()
                  + self.embeddings.nbytes + self.labels.nbytes)
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            log.info("Resident on %s: need %.2f GB, %.2f GB free of %.2f GB",
                     device, needed / 1e9, free / 1e9, total / 1e9)
            if needed > free * 0.9:
                raise MemoryError(
                    f"resident counts need {needed/1e9:.1f} GB but only "
                    f"{free/1e9:.1f} GB is free on {device}; use counts_dtype='float16', "
                    f"pass a gene panel, or resident='cpu'")
        started = time.time()
        # Densify in slices: one 407k x 5101 dense host array would be 8.3 GB.
        counts = torch.empty((n, g), dtype=dtype, device=device)
        step = max(1, int(2e8 // max(g, 1)))
        for start in range(0, n, step):
            block = self.counts[start : start + step].toarray()
            counts[start : start + step] = torch.from_numpy(block).to(device=device, dtype=dtype)
        self._counts_t = counts
        self._labels_t = torch.from_numpy(self.labels).to(device)
        self._embeddings_t = torch.from_numpy(self.embeddings).to(device)
        self.constant = self.constant.to(device)
        self._composition_t = self.constant.neighbour_composition
        self._pos_t = torch.from_numpy(self.positions_um).to(device)
        self.resident = device
        log.info("  resident in %.1fs: counts %s %s, labels %s, embeddings %s",
                 time.time() - started, tuple(counts.shape), counts.dtype,
                 tuple(self._labels_t.shape), tuple(self._embeddings_t.shape))

    def batch_from_seeds(self, seeds: np.ndarray) -> "TorchBatch":
        """One batch from explicit seed cells.

        When the dataset is resident on a device this is an ``index_select``
        with no host-to-device copy, which is where the throughput comes from;
        otherwise the same fields are gathered from numpy. ``edge_index`` is
        ``(2, n_edges)``, the convention message-passing code expects.
        """
        import torch

        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        nodes = np.union1d(seeds, self.adjacency[seeds].indices)
        local, attr, edge_ids, into_j = self._edges_for(seeds, nodes)
        position = pd.Series(np.arange(len(nodes)), index=pd.Index(nodes))
        seed_local = position.reindex(seeds).to_numpy().astype(np.int64)
        image_rows = nodes if self.image_scope == "nodes" else seeds

        device = self.resident
        if device is not None:
            take = lambda t, r: t.index_select(0, torch.as_tensor(r, device=device))  # noqa: E731
            node_index = torch.as_tensor(nodes, device=device)
            fields = dict(
                x=take(self._counts_t, nodes).float(),
                y=take(self._labels_t, nodes),
                image_embedding=take(self._embeddings_t, image_rows),
                seed_composition=take(self._composition_t, self.type_index[seeds]),
                pos=take(self._pos_t, nodes),
            )
        else:
            f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
            comp = self.neighbour_composition.to_numpy().astype(np.float32)
            node_index = torch.as_tensor(nodes, dtype=torch.long)
            fields = dict(
                x=f(self.counts[nodes].toarray()),
                y=f(self.labels[nodes]),
                image_embedding=f(self.embeddings[image_rows]),
                seed_composition=f(comp[self.type_index[seeds]]),
                pos=f(self.positions_um[nodes]),
            )

        batch = TorchBatch(
            edge_index=torch.as_tensor(np.ascontiguousarray(local.T),
                                       dtype=torch.long, device=device),
            edge_attr=torch.as_tensor(attr, dtype=torch.float32, device=device),
            seed_index=torch.as_tensor(seed_local, dtype=torch.long, device=device),
            node_ids=[str(c) for c in self.cell_ids[nodes]],
            edge_attr_names=self.edge_metrics,
            node_index=node_index,
            edge_id=torch.as_tensor(edge_ids, dtype=torch.long, device=device),
            into_j=torch.as_tensor(into_j, dtype=torch.bool, device=device),
            **fields,
        )
        return batch.to(self.device) if device is None and self.device else batch


    # -- construction helpers -------------------------------------------------

    def _build_edges(self) -> None:
        import scipy.sparse as sp

        prefix = self.graph
        conn = sp.triu(self.adata.obsp[f"{prefix}_connectivities"], k=1).tocoo()
        self.edge_i, self.edge_j = conn.row, conn.col
        # edge_metrics is already filtered to keys this graph stores, so every
        # lookup here hits.
        self.edge_values = np.stack(
            [self._pull_edges(m) for m in self.edge_metrics], axis=1)

        n = self.adata.n_obs
        self.adjacency = sp.csr_matrix(
            (np.ones(2 * len(self.edge_i), dtype=np.int8),
             (np.concatenate([self.edge_i, self.edge_j]),
              np.concatenate([self.edge_j, self.edge_i]))),
            shape=(n, n),
        )
        # Positions into the undirected edge list, for edge-feature lookup.
        order = np.arange(len(self.edge_i))
        self.edge_lookup = sp.csr_matrix(
            (np.concatenate([order, order]) + 1,
             (np.concatenate([self.edge_i, self.edge_j]),
              np.concatenate([self.edge_j, self.edge_i]))),
            shape=(n, n),
        )
        self.degrees = np.asarray(self.adjacency.sum(axis=1)).ravel().astype(np.int32)

    def _build_labels(self) -> None:
        series = pd.Series(self.adata.obs[self.label_key]).astype(object)
        # Reuse the panel's own spelling if it already has an unassigned class,
        # so the fill does not add a second class with the same meaning.
        spellings = {str(v).casefold(): str(v) for v in series.dropna().unique()}
        series = series.where(
            series.notna(), spellings.get(UNASSIGNED.casefold(), UNASSIGNED)
        ).astype(str)
        self.type_names = np.asarray(sorted(series.unique()))
        index = pd.Series(np.arange(len(self.type_names)), index=pd.Index(self.type_names))
        self.type_index = index.reindex(series.to_numpy()).to_numpy().astype(np.int64)
        self.labels = np.zeros((len(series), len(self.type_names)), dtype=np.float32)
        self.labels[np.arange(len(series)), self.type_index] = 1.0

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    def _pull_edges(self, metric: str) -> np.ndarray | None:
        """One edge metric for the retained edges, or None if not stored."""
        key = f"{self.graph}_{metric}"
        if key not in self.adata.obsp:
            return None
        matrix = self.adata.obsp[key].tocsr()
        return np.asarray(matrix[self.edge_i, self.edge_j]).ravel()


    # -- batching -------------------------------------------------------------

    def _edges_to_seeds(self, seeds: np.ndarray, nodes: np.ndarray):
        """Directed edges **neighbour -> seed**, the only ones a seed consumes.

        Every seed collects one edge per neighbour, oriented inward. Edges
        between two neighbours are dropped -- they never reach a seed in a single
        message-passing step. Two adjacent seeds yield one edge each way, since
        each is a neighbour of the other.
        """
        indptr, indices = self.adjacency.indptr, self.adjacency.indices
        degrees = indptr[seeds + 1] - indptr[seeds]
        dst_global = np.repeat(seeds, degrees)
        src_global = np.concatenate(
            [indices[indptr[s] : indptr[s + 1]] for s in seeds]
        ) if len(seeds) else np.empty(0, dtype=np.int64)
        if not len(src_global):
            empty = np.empty(0, dtype=np.int64)
            return empty, np.empty((0, 2), dtype=np.int64), empty

        edge_ids = np.asarray(
            self.edge_lookup[src_global, dst_global]
        ).ravel().astype(np.int64) - 1
        position = pd.Series(np.arange(len(nodes)), index=pd.Index(nodes))
        local = np.stack([
            position.reindex(src_global).to_numpy(),
            position.reindex(dst_global).to_numpy(),
        ], axis=1).astype(np.int64)
        return edge_ids, local, dst_global

    def _edges_for(self, seeds: np.ndarray, nodes: np.ndarray):
        """``(local edge pairs, edge_attr, edge_ids, into_j)`` for this batch.

        ``into_j`` records which way each edge points, since the whole-slide
        per-edge arrays on ``constant`` are stored once per undirected edge.
        """
        edge_ids, local, dst = self._edges_to_seeds(seeds, nodes)
        into_j = dst == self.edge_j[edge_ids] if len(edge_ids) else np.empty(0, dtype=bool)
        return local, self.edge_values[edge_ids], edge_ids, into_j


    def batches(self, shuffle: bool = True, drop_last: bool = False) -> Iterator["TorchBatch"]:
        """One epoch: every cell is a seed exactly once."""
        order = np.arange(self.adata.n_obs)
        if shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), self.batch_cells):
            chunk = order[start : start + self.batch_cells]
            if drop_last and len(chunk) < self.batch_cells:
                continue
            yield self.batch_from_seeds(chunk)

    def __len__(self) -> int:
        n = self.adata.n_obs
        return (n + self.batch_cells - 1) // self.batch_cells

    def __iter__(self) -> Iterator["TorchBatch"]:
        return self.batches()
