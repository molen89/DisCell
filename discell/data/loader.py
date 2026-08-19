#!/usr/bin/env python3
"""Batch loader over a saved cell-graph bundle.

Serves subgraph batches for a model that predicts a cell from its neighbourhood:
raw counts, the neighbour graph with geometric edge features, one-hot cell-type
labels, and a per-cell image embedding.

Batches are served as :class:`TorchBatch` tensors by default. Pass
``as_torch=False`` for the numpy :class:`CellBatch`, which is what the geometry
is built in and what the self-test checks.

Construction does three things:

1. loads the bundle and optionally **reduces the graph** by a wall/gap filter
2. builds the one-hot label matrix
3. computes the fixed **neighbourhood composition** table -- for each cell type,
   the mean distribution of its neighbours' types -- which is returned up front
   because it does not change across batches

A batch picks ``batch_cells`` seed cells, pulls their 1-hop neighbours, and by
default serves only the **directed neighbour -> seed** edges.

What a batch carries, by the axis it is aligned to:

============  ==========================================  ===============
seeds         ``image_embedding``, ``seed_index``,        fixed size
              ``seed_composition``
nodes         ``x``, ``y``, ``pos``, ``node_ids``         varies
edges         ``edge_index``, ``edge_attr``               varies
constant      ``neighbour_composition``                   (K, K)
============  ==========================================  ===============

Model inputs are the counts, the edge attributes and the image embedding.
``pos`` and ``node_ids`` are auxiliary -- for plotting and debugging.

Usage::

    python -m discell.data.loader --self-test --dataset xenium_prime_ovarian_cancer_ffpe
    python -m discell.data.loader --dataset xenium_prime_ovarian_cancer_ffpe \\
        --graph contact --min-apposed-um 0.001 --batch-cells 64
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from discell import paths

log = logging.getLogger("discell.data.loader")

DEFAULT_GRAPH = "contact"
DEFAULT_BATCH_CELLS = 64
DEFAULT_DIST_METRIC = "centroid_dist_um"
DEFAULT_WALL_METRIC = "apposed_wall_um"

#: Label given to cells with no annotation. Kept as an explicit class rather
#: than dropped, so the one-hot rows always sum to 1 and node counts line up.
UNASSIGNED = "unassigned"


@dataclass
class CellBatch:
    """One subgraph batch.

    ``x``, ``y`` and ``node_ids`` cover every node -- seeds *and* their
    neighbours -- while ``image_embedding`` covers only the seeds, indexed by
    ``seed_index``. Neighbours are context; the model is asked about the seeds.
    """

    x: np.ndarray                # (n_nodes, n_genes)   raw integer counts
    edge_index: np.ndarray       # (n_edges, 2)         positions within this batch
    edge_attr: np.ndarray        # (n_edges, 2)         [distance_um, wall_um]
    y: np.ndarray                # (n_nodes, n_types)   one-hot
    image_embedding: np.ndarray  # (n_seeds, dim)
    seed_index: np.ndarray       # (n_seeds,)           positions of seeds in the node axis
    node_ids: np.ndarray         # (n_nodes,)           cell ids
    edge_attr_names: tuple[str, str] = (DEFAULT_DIST_METRIC, DEFAULT_WALL_METRIC)
    neighbour_composition: np.ndarray | None = None   # (K, K)
    seed_composition: np.ndarray | None = None        # (n_seeds, K)
    pos: np.ndarray | None = None   # (n_nodes, 2) um -- auxiliary, for plotting
    node_index: np.ndarray | None = None   # (n_nodes,) row index into the dataset

    @property
    def n_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def n_seeds(self) -> int:
        return len(self.seed_index)

    def to_torch(self, device: str | None = None, dtype=None) -> "TorchBatch":
        """Convert to a :class:`TorchBatch` of tensors.

        ``edge_index`` is transposed to ``(2, n_edges)`` on the way, which is the
        convention PyTorch Geometric and most message-passing code expects.
        """
        import torch

        float_dtype = dtype or torch.float32
        batch = TorchBatch(
            x=torch.as_tensor(self.x, dtype=float_dtype),
            edge_index=torch.as_tensor(np.ascontiguousarray(self.edge_index.T),
                                       dtype=torch.long),
            edge_attr=torch.as_tensor(self.edge_attr, dtype=float_dtype),
            y=torch.as_tensor(self.y, dtype=float_dtype),
            image_embedding=torch.as_tensor(self.image_embedding, dtype=float_dtype),
            seed_index=torch.as_tensor(self.seed_index, dtype=torch.long),
            node_ids=list(map(str, self.node_ids)),
            edge_attr_names=self.edge_attr_names,
            neighbour_composition=None if self.neighbour_composition is None
            else torch.as_tensor(self.neighbour_composition, dtype=float_dtype),
            seed_composition=None if self.seed_composition is None
            else torch.as_tensor(self.seed_composition, dtype=float_dtype),
            pos=None if self.pos is None else torch.as_tensor(self.pos, dtype=float_dtype),
            node_index=None if self.node_index is None
            else torch.as_tensor(self.node_index, dtype=torch.long),
        )
        return batch.to(device) if device else batch

    def __repr__(self) -> str:
        return (
            f"CellBatch(nodes={self.n_nodes}, seeds={self.n_seeds}, "
            f"edges={len(self.edge_index)}, genes={self.x.shape[1]}, "
            f"types={self.y.shape[1]}, image_dim={self.image_embedding.shape[1]})"
        )


@dataclass
class TorchBatch:
    """One subgraph batch as tensors, ready for a model.

    Mirrors :class:`CellBatch` field for field, except ``edge_index`` is
    ``(2, n_edges)`` rather than ``(n_edges, 2)``.
    """

    x: "torch.Tensor"                # (n_nodes, n_genes)   raw counts
    edge_index: "torch.Tensor"       # (2, n_edges)
    edge_attr: "torch.Tensor"        # (n_edges, 2)         [distance_um, wall_um]
    y: "torch.Tensor"                # (n_nodes, n_types)   one-hot
    image_embedding: "torch.Tensor"  # (n_seeds, dim)
    seed_index: "torch.Tensor"       # (n_seeds,)
    node_ids: list[str]
    edge_attr_names: tuple[str, str] = (DEFAULT_DIST_METRIC, DEFAULT_WALL_METRIC)
    #: (K, K) mean neighbour-type distribution per cell type -- fixed for the run
    neighbour_composition: "torch.Tensor | None" = None
    #: (n_seeds, K) the row of that table belonging to each seed's own type
    seed_composition: "torch.Tensor | None" = None
    #: (n_nodes, 2) centroid coordinates in microns, absolute slide coordinates.
    #:
    #: **Auxiliary, not a model input.** It is here for plotting a batch back
    #: onto the slide, sanity-checking a neighbourhood, or debugging. The spatial
    #: information a model should consume is already reduced into ``edge_attr``
    #: as ``centroid_dist_um`` and ``apposed_wall_um``; feeding absolute slide
    #: coordinates as features would additionally let a model memorise position.
    pos: "torch.Tensor | None" = None
    #: (n_nodes,) row index of each node in the full dataset, for scattering
    #: predictions back into ``adata`` -- the integer twin of ``node_ids``.
    node_index: "torch.Tensor | None" = None

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_seeds(self) -> int:
        return int(self.seed_index.shape[0])

    def to(self, device) -> "TorchBatch":
        return TorchBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            y=self.y.to(device),
            image_embedding=self.image_embedding.to(device),
            seed_index=self.seed_index.to(device),
            node_ids=self.node_ids,
            edge_attr_names=self.edge_attr_names,
            neighbour_composition=None if self.neighbour_composition is None
            else self.neighbour_composition.to(device),
            seed_composition=None if self.seed_composition is None
            else self.seed_composition.to(device),
            pos=None if self.pos is None else self.pos.to(device),
            node_index=None if self.node_index is None else self.node_index.to(device),
        )

    def seed_labels(self) -> "torch.Tensor":
        """Class index per seed cell, for cross-entropy."""
        return self.y[self.seed_index].argmax(dim=1)

    def __repr__(self) -> str:
        return (
            f"TorchBatch(nodes={self.n_nodes}, seeds={self.n_seeds}, "
            f"edges={int(self.edge_index.shape[1])}, genes={int(self.x.shape[1])}, "
            f"types={int(self.y.shape[1])}, image_dim={int(self.image_embedding.shape[1])}, "
            f"device={self.x.device})"
        )


def load_embeddings(path: str | Path, cell_ids: Sequence[str],
                    dataset: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load per-cell image embeddings aligned to *cell_ids*.

    Accepts ``.npz`` with ``cell_ids`` and ``embeddings`` arrays, or a parquet
    with a cell-id column plus either an ``embedding`` list column or numeric
    feature columns. Cells with no embedding get zeros; the returned mask marks
    which rows are real, so a model can avoid training on fabricated vectors.

    Two guards stop embeddings from one slide being served against another's
    bundle, which used to fail only as a warning about zero-filled rows:

    * if the file records a ``dataset`` and *dataset* is given, they must match;
    * if **no** cell id matches at all, that is a mismatched file rather than an
      incomplete one, whatever the metadata says.
    """
    path = Path(path)
    recorded: str | None = None
    if path.suffix in (".pt", ".pth"):
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        recorded = payload.get("dataset")
        ids = np.asarray(payload["cell_ids"]).astype(str)
        vectors = np.asarray(payload["embeddings"].float().numpy(), dtype=np.float32)
    elif path.suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        ids = np.asarray(payload["cell_ids"]).astype(str)
        vectors = np.asarray(payload["embeddings"], dtype=np.float32)
    elif path.suffix in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
        id_col = next((c for c in ("cell", "cell_id", "barcode") if c in frame), None)
        if id_col is None:
            raise ValueError(f"{path} has no cell id column")
        ids = frame[id_col].astype(str).to_numpy()
        if "embedding" in frame:
            vectors = np.stack(frame["embedding"].to_numpy()).astype(np.float32)
        else:
            numeric = frame.drop(columns=[id_col]).select_dtypes("number")
            vectors = numeric.to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported embedding format: {path.suffix}")

    if dataset and recorded and recorded != dataset:
        raise ValueError(
            f"{path.name} was computed on dataset {recorded!r}, but this bundle is "
            f"{dataset!r}. Use that dataset's own embeddings:\n"
            f"  data/datasets/{dataset}/embeddings/"
        )

    lookup = pd.Series(np.arange(len(ids)), index=pd.Index(ids))
    lookup = lookup[~lookup.index.duplicated()]
    position = lookup.reindex(pd.Index(np.asarray(cell_ids, dtype=str)))
    found = position.notna().to_numpy()

    if not found.any():
        raise ValueError(
            f"{path.name} shares no cell ids with this bundle -- it belongs to a "
            f"different dataset"
            + (f" ({recorded})" if recorded else "")
            + f". Its ids look like {[str(v) for v in ids[:2]]}, the bundle's like "
              f"{[str(v) for v in np.asarray(cell_ids, dtype=str)[:2]]}."
        )

    out = np.zeros((len(cell_ids), vectors.shape[1]), dtype=np.float32)
    out[found] = vectors[position[found].astype(int).to_numpy()]
    if not found.all():
        log.warning("%d of %d cells have no image embedding (zero-filled)",
                    int((~found).sum()), len(cell_ids))
    return out, found


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
        self,
        bundle_dir: str | Path,
        name: str,
        graph: str = DEFAULT_GRAPH,
        label_key: str | None = None,
        batch_cells: int = DEFAULT_BATCH_CELLS,
        min_apposed_um: float | None = None,
        min_wall_um: float | None = None,
        max_gap_um: float | None = None,
        dist_metric: str = DEFAULT_DIST_METRIC,
        wall_metric: str = DEFAULT_WALL_METRIC,
        embeddings: str | Path | None = None,
        embedding_dim: int = 256,
        genes: Sequence[str] | None = None,
        edge_mode: str = "to_seed",
        image_scope: str = "seeds",
        induced: bool = True,
        both_directions: bool = True,
        drop_unassigned: bool = False,
        as_torch: bool = True,
        device: str | None = None,
        resident: str | None = None,
        counts_dtype: str = "float32",
        seed: int = 0,
        dataset_id: str | None = None,
    ) -> None:
        from discell.data.export import load_bundle

        self.adata = load_bundle(bundle_dir, name)
        self.graph = graph
        self.batch_cells = batch_cells
        if edge_mode not in ("to_seed", "induced", "star"):
            raise ValueError("edge_mode must be 'to_seed', 'induced' or 'star'")
        self.edge_mode = edge_mode
        if image_scope not in ("seeds", "nodes"):
            raise ValueError("image_scope must be 'seeds' or 'nodes'")
        # 'seeds': (n_seeds, dim), aligned to seed_index -- images only for the
        # cells being predicted. 'nodes': (n_nodes, dim), aligned to x, so
        # neighbour morphology can be message-passed like any other feature.
        self.image_scope = image_scope
        self.induced = induced
        self.both_directions = both_directions
        self.as_torch = as_torch
        self.device = device
        self.rng = np.random.default_rng(seed)

        if f"{graph}_connectivities" not in self.adata.obsp:
            raise KeyError(f"no graph {graph!r} in bundle; have "
                           f"{sorted(k for k in self.adata.obsp if k.endswith('_connectivities'))}")

        self.label_key = label_key or self.adata.uns.get("default_label") or "cell_group"
        if self.label_key not in self.adata.obs:
            raise KeyError(f"label {self.label_key!r} not in obs")

        self.dist_metric = dist_metric
        self.wall_metric = wall_metric if f"{graph}_{wall_metric}" in self.adata.obsp else "shared_wall_um"
        if self.wall_metric != wall_metric:
            log.warning("%s not stored for %s; using %s", wall_metric, graph, self.wall_metric)

        # 0. optionally reduce the graph before anything is derived from it.
        # Kept so a rebuild after dropping cells reapplies the same filters.
        self._filters = (min_apposed_um, min_wall_um, max_gap_um)
        self._build_edges(*self._filters)

        if drop_unassigned:
            self._drop_unassigned()

        # 1. labels and the fixed neighbourhood composition
        self._build_labels()
        self.neighbour_composition = self._composition()

        # genes
        self.gene_names = np.asarray(self.adata.var["gene_name"].astype(str))
        if genes is not None:
            wanted = pd.Index(np.asarray(genes, dtype=str))
            position = pd.Series(np.arange(self.adata.n_vars),
                                 index=pd.Index(self.gene_names))
            position = position[~position.index.duplicated()].reindex(wanted).dropna()
            self.gene_index = position.astype(int).to_numpy()
            self.gene_names = self.gene_names[self.gene_index]
        else:
            self.gene_index = np.arange(self.adata.n_vars)

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

        self.resident = None
        self._counts_t = self._labels_t = self._embeddings_t = None
        self._composition_t = self._pos_t = None
        if resident:
            self._make_resident(resident, counts_dtype)

        log.info(
            "Dataset: %d cells, %d genes, %d types, %d edges (%s graph), image dim %d",
            self.adata.n_obs, len(self.gene_index), self.n_types,
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
        n, g = self.adata.n_obs, len(self.gene_index)
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
            block = self.counts[start : start + step][:, self.gene_index].toarray()
            counts[start : start + step] = torch.from_numpy(block).to(device=device, dtype=dtype)
        self._counts_t = counts
        self._labels_t = torch.from_numpy(self.labels).to(device)
        self._embeddings_t = torch.from_numpy(self.embeddings).to(device)
        self._composition_t = torch.as_tensor(
            self.neighbour_composition.to_numpy(), dtype=torch.float32, device=device)
        self._pos_t = torch.from_numpy(self.positions_um).to(device)
        self.resident = device
        log.info("  resident in %.1fs: counts %s %s, labels %s, embeddings %s",
                 time.time() - started, tuple(counts.shape), counts.dtype,
                 tuple(self._labels_t.shape), tuple(self._embeddings_t.shape))

    def resident_batch(self, seeds: np.ndarray) -> "TorchBatch":
        """Build a batch by indexing the resident tensors directly."""
        import torch

        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        neighbours = self.adjacency[seeds].indices
        nodes = np.union1d(seeds, neighbours)
        edge_ids, local, attr = self._edges_for(seeds, nodes)

        position = pd.Series(np.arange(len(nodes)), index=pd.Index(nodes))
        seed_local = position.reindex(seeds).to_numpy().astype(np.int64)

        device = self.resident
        node_t = torch.as_tensor(nodes, device=device)
        seed_t = torch.as_tensor(seeds, device=device)
        return TorchBatch(
            x=self._counts_t.index_select(0, node_t).float(),
            edge_index=torch.as_tensor(np.ascontiguousarray(local.T), dtype=torch.long,
                                       device=device),
            edge_attr=torch.as_tensor(attr, dtype=torch.float32, device=device),
            y=self._labels_t.index_select(0, node_t),
            image_embedding=self._embeddings_t.index_select(
                0, node_t if self.image_scope == "nodes" else seed_t),
            seed_index=torch.as_tensor(seed_local, dtype=torch.long, device=device),
            node_ids=[str(c) for c in self.cell_ids[nodes]],
            edge_attr_names=(self.dist_metric, self.wall_metric),
            neighbour_composition=self._composition_t,
            seed_composition=self._composition_t.index_select(
                0, torch.as_tensor(self.type_index[seeds], device=device)),
            pos=self._pos_t.index_select(0, node_t),
            node_index=node_t,
        )

    # -- construction helpers -------------------------------------------------

    def _build_edges(self, min_apposed_um, min_wall_um, max_gap_um) -> None:
        import scipy.sparse as sp

        prefix = self.graph
        conn = sp.triu(self.adata.obsp[f"{prefix}_connectivities"], k=1).tocoo()
        i, j = conn.row, conn.col
        n_before = len(i)

        def pull(metric: str) -> np.ndarray:
            key = f"{prefix}_{metric}"
            if key not in self.adata.obsp:
                return np.zeros(len(i))
            return np.asarray(self.adata.obsp[key].tocsr()[i, j]).ravel()

        dist = pull(self.dist_metric)
        wall = pull(self.wall_metric)
        gap = pull("wall_dist_um")

        keep = np.ones(len(i), dtype=bool)
        if min_apposed_um is not None:
            keep &= pull("apposed_wall_um") >= min_apposed_um
        if min_wall_um is not None:
            keep &= pull("shared_wall_um") >= min_wall_um
        if max_gap_um is not None:
            keep &= gap <= max_gap_um
        if not keep.all():
            log.info("Graph filter kept %d of %d edges (%.1f%%)",
                     int(keep.sum()), n_before, 100 * keep.mean())

        self.edge_i, self.edge_j = i[keep], j[keep]
        self.edge_dist, self.edge_wall = dist[keep], wall[keep]

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

    def _drop_unassigned(self) -> None:
        labels = pd.Series(self.adata.obs[self.label_key]).astype(object)
        keep = labels.notna().to_numpy()
        if keep.all():
            return
        log.info("Dropping %d cells with no %s", int((~keep).sum()), self.label_key)
        self.adata = self.adata[keep].copy()
        # Indices shifted, so the graph must be rebuilt -- with the same filters.
        self._build_edges(*self._filters)

    def _build_labels(self) -> None:
        series = pd.Series(self.adata.obs[self.label_key]).astype(object)
        series = series.where(series.notna(), UNASSIGNED).astype(str)
        self.type_names = np.asarray(sorted(series.unique()))
        index = pd.Series(np.arange(len(self.type_names)), index=pd.Index(self.type_names))
        self.type_index = index.reindex(series.to_numpy()).to_numpy().astype(np.int64)
        self.labels = np.zeros((len(series), len(self.type_names)), dtype=np.float32)
        self.labels[np.arange(len(series)), self.type_index] = 1.0

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    def composition_tensor(self, device: str | None = None):
        """The K x K neighbourhood composition as a float32 tensor."""
        import torch

        out = torch.as_tensor(self.neighbour_composition.to_numpy(), dtype=torch.float32)
        return out.to(device or self.device) if (device or self.device) else out

    def torch_dataloader(self, num_workers: int = 0, shuffle: bool = True, **kwargs):
        """A ``torch.utils.data.DataLoader`` over subgraph batches.

        Each item is already a full batch, so the loader runs with
        ``batch_size=1`` and an identity collate; ``num_workers`` then builds
        several subgraphs in parallel, which is where the time actually goes.
        """
        import torch
        from torch.utils.data import DataLoader, Dataset

        outer = self

        class _Batches(Dataset):
            def __len__(self) -> int:
                return len(outer)

            def __getitem__(self, index: int):
                order = outer._epoch_order(index)
                batch = outer.batch_from_seeds(order)
                return batch.to_torch() if outer.as_torch else batch

        return DataLoader(_Batches(), batch_size=1, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=lambda items: items[0],
                          **kwargs)

    def _epoch_order(self, index: int) -> np.ndarray:
        """Seed cells for batch *index* of a fixed partition of all cells."""
        if getattr(self, "_partition", None) is None:
            order = np.arange(self.adata.n_obs)
            self.rng.shuffle(order)
            self._partition = order
        start = index * self.batch_cells
        return self._partition[start : start + self.batch_cells]

    def _composition(self) -> pd.DataFrame:
        """Mean neighbour-type distribution per cell type.

        Per cell: the one-hot labels of its neighbours, averaged -- so each row
        sums to 1 for any cell with at least one neighbour. Then averaged again
        over the cells of each type, unweighted, as specified.

        Isolated cells have no neighbourhood to describe (0/0) and are excluded
        from their type's mean rather than counted as zeros, which would
        otherwise drag every row toward the origin.
        """
        # float64 throughout: the labels are float32 for the model, but averaging
        # hundreds of thousands of rows in float32 accumulates ~1e-5 of error --
        # enough that the per-type distributions visibly stop summing to 1.
        counts = self.adjacency @ self.labels.astype(np.float64)   # (n_cells, n_types)
        degree = self.degrees.astype(np.float64)
        connected = degree > 0
        per_cell = np.zeros_like(counts)
        per_cell[connected] = counts[connected] / degree[connected, None]

        rows = np.zeros((self.n_types, self.n_types), dtype=np.float64)
        n_used = np.zeros(self.n_types, dtype=np.int64)
        for k in range(self.n_types):
            mask = (self.type_index == k) & connected
            n_used[k] = int(mask.sum())
            if n_used[k]:
                rows[k] = per_cell[mask].mean(axis=0)

        self.per_cell_composition = per_cell
        frame = pd.DataFrame(rows, index=self.type_names, columns=self.type_names)
        frame.index.name = f"{self.label_key} (self)"
        frame.columns.name = "neighbour type"
        self.composition_support = pd.Series(n_used, index=self.type_names, name="n_cells_used")

        n_isolated = int((~connected).sum())
        if n_isolated:
            log.info("Composition excludes %d isolated cells (%.1f%%)",
                     n_isolated, 100 * n_isolated / len(connected))
        return frame

    # -- batching -------------------------------------------------------------

    def _subgraph(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Edges of the induced subgraph over *nodes*, as (edge ids, pairs)."""
        sub = self.edge_lookup[nodes][:, nodes].tocoo()
        # Upper triangle only: the lookup is symmetric, so each edge appears twice.
        upper = sub.row < sub.col
        local = np.stack([sub.row[upper], sub.col[upper]], axis=1)
        edge_ids = sub.data[upper].astype(np.int64) - 1
        return edge_ids, local

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
            return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64)

        edge_ids = np.asarray(
            self.edge_lookup[src_global, dst_global]
        ).ravel().astype(np.int64) - 1
        position = pd.Series(np.arange(len(nodes)), index=pd.Index(nodes))
        local = np.stack([
            position.reindex(src_global).to_numpy(),
            position.reindex(dst_global).to_numpy(),
        ], axis=1).astype(np.int64)
        return edge_ids, local

    def _star_edges(self, seeds: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Only seed-to-neighbour edges, dropping neighbour-neighbour ones."""
        position = pd.Series(np.arange(len(nodes)), index=pd.Index(nodes))
        edge_ids, local = self._subgraph(nodes)
        seed_local = set(position.reindex(seeds).to_numpy().tolist())
        keep = np.array([(a in seed_local) or (b in seed_local) for a, b in local], dtype=bool)
        return edge_ids[keep], local[keep]

    def _edges_for(self, seeds: np.ndarray, nodes: np.ndarray):
        """Edges for this batch under the configured ``edge_mode``."""
        if self.edge_mode == "to_seed":
            edge_ids, local = self._edges_to_seeds(seeds, nodes)
            attr = np.stack([self.edge_dist[edge_ids], self.edge_wall[edge_ids]], axis=1)
            return edge_ids, local, attr

        edge_ids, local = (
            self._subgraph(nodes) if self.edge_mode == "induced"
            else self._star_edges(seeds, nodes)
        )
        attr = np.stack([self.edge_dist[edge_ids], self.edge_wall[edge_ids]], axis=1)
        if self.both_directions and len(local):
            local = np.concatenate([local, local[:, ::-1]], axis=0)
            attr = np.concatenate([attr, attr], axis=0)
        return edge_ids, local, attr

    def batch_from_seeds(self, seeds: np.ndarray) -> CellBatch:
        """Build one batch from explicit seed cells."""
        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        neighbours = self.adjacency[seeds].indices
        nodes = np.union1d(seeds, neighbours)

        edge_ids, local, attr = self._edges_for(seeds, nodes)

        position = pd.Series(np.arange(len(nodes)), index=pd.Index(nodes))
        seed_index = position.reindex(seeds).to_numpy().astype(np.int64)

        x = self.counts[nodes][:, self.gene_index].toarray().astype(np.float32)
        return CellBatch(
            x=x,
            edge_index=local.astype(np.int64),
            edge_attr=attr.astype(np.float32),
            y=self.labels[nodes],
            image_embedding=self.embeddings[nodes if self.image_scope == "nodes" else seeds],
            seed_index=seed_index,
            node_ids=self.cell_ids[nodes],
            edge_attr_names=(self.dist_metric, self.wall_metric),
            neighbour_composition=self.neighbour_composition.to_numpy().astype(np.float32),
            seed_composition=self.neighbour_composition.to_numpy().astype(np.float32)[
                self.type_index[seeds]],
            pos=self.positions_um[nodes],
            node_index=np.asarray(nodes, dtype=np.int64),
        )

    def batches(self, shuffle: bool = True, drop_last: bool = False) -> Iterator[CellBatch]:
        """One epoch: every cell is a seed exactly once."""
        order = np.arange(self.adata.n_obs)
        if shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), self.batch_cells):
            chunk = order[start : start + self.batch_cells]
            if drop_last and len(chunk) < self.batch_cells:
                continue
            if self.resident is not None:
                yield self.resident_batch(chunk)
                continue
            batch = self.batch_from_seeds(chunk)
            yield batch.to_torch(self.device) if self.as_torch else batch

    def __len__(self) -> int:
        n = self.adata.n_obs
        return (n + self.batch_cells - 1) // self.batch_cells

    def __iter__(self) -> Iterator[CellBatch]:
        return self.batches()


# --------------------------------------------------------------------------
# CLI and self-test
# --------------------------------------------------------------------------


def self_test(bundle_dir: str, name: str, graph: str, batch_cells: int) -> int:
    print(f"\n=== loader self-test: {name} / {graph} ===\n")

    print("[1/6] construct")
    # numpy batches here: the checks below assert the geometry contract
    # (edge_index as (n_edges, 2), node_ids indexable by array).
    data = CellGraphDataset(bundle_dir, name, graph=graph,
                            batch_cells=batch_cells, as_torch=False)
    print(f"      {data.adata.n_obs:,} cells, {len(data.gene_index):,} genes, "
          f"{data.n_types} types, {len(data.edge_i):,} edges, {len(data)} batches")

    print("[2/6] neighbourhood composition is a valid distribution")
    comp = data.neighbour_composition
    sums = comp.sum(axis=1).to_numpy()
    used = data.composition_support.to_numpy()
    for k, total in enumerate(sums):
        if used[k]:
            assert abs(total - 1.0) < 1e-6, f"row {comp.index[k]!r} sums to {total}"
    assert (comp.to_numpy() >= 0).all(), "negative composition"
    print(f"      {comp.shape[0]}x{comp.shape[1]}, every supported row sums to 1")

    print("[3/6] batch shapes agree")
    batch = next(iter(data.batches(shuffle=False)))
    print(f"      {batch!r}")
    assert batch.x.shape[0] == batch.y.shape[0] == len(batch.node_ids)
    assert batch.x.shape[1] == len(data.gene_index)
    assert batch.y.shape[1] == data.n_types
    assert batch.edge_attr.shape == (len(batch.edge_index), 2)
    assert batch.image_embedding.shape[0] == batch.n_seeds
    assert batch.n_seeds <= batch.n_nodes

    print("[4/6] edges are valid and features line up")
    if len(batch.edge_index):
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.n_nodes
        assert (batch.edge_index[:, 0] != batch.edge_index[:, 1]).all(), "self loop"
        assert (batch.edge_attr[:, 0] > 0).all(), "non-positive distance"
        assert (batch.edge_attr[:, 1] >= 0).all(), "negative wall"
        # every seed's neighbours must appear as nodes
        seeds = batch.node_ids[batch.seed_index]
        assert len(set(seeds)) == len(seeds), "duplicate seeds"
    print(f"      {len(batch.edge_index)} edges, attrs = {batch.edge_attr_names}")

    print("[5/6] counts are raw")
    assert batch.x.min() >= 0 and np.allclose(batch.x, np.round(batch.x))
    assert (batch.y.sum(axis=1) == 1).all(), "labels are not one-hot"
    print(f"      total counts in batch = {int(batch.x.sum()):,}")

    print("[6/6] one epoch covers every cell exactly once")
    seen = []
    for b in data.batches(shuffle=True):
        seen.append(b.node_ids[b.seed_index])
    seen = np.concatenate(seen)
    assert len(seen) == data.adata.n_obs, f"{len(seen)} seeds vs {data.adata.n_obs} cells"
    assert len(set(seen)) == data.adata.n_obs, "a cell was a seed twice"
    print(f"      {len(seen):,} seeds across {len(data)} batches, no repeats")

    print("\n--- neighbourhood composition (top types) ---")
    top = data.composition_support.sort_values(ascending=False).head(6).index
    print(comp.loc[top, top].round(3).to_string())

    print("\nAll checks passed.\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    paths.add_dataset_args(parser)
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--batch-cells", type=int, default=DEFAULT_BATCH_CELLS)
    parser.add_argument("--resident", default=None, choices=("gpu", "cuda", "cpu"),
                        help="hold counts, labels and embeddings as tensors on this device")
    parser.add_argument("--counts-dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--min-apposed-um", type=float, default=None)
    parser.add_argument("--min-wall-um", type=float, default=None)
    parser.add_argument("--max-gap-um", type=float, default=None)
    parser.add_argument("--embeddings", default=None, help=".npz or .parquet of image embeddings")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--drop-unassigned", action="store_true",
                        help="drop cells with no label instead of keeping an 'unassigned' class")
    parser.add_argument("--image-scope", default="seeds", choices=("seeds", "nodes"),
                        help="seeds: image_embedding is (n_seeds, dim) aligned to "
                             "seed_index; nodes: (n_nodes, dim) aligned to x")
    parser.add_argument("--edge-mode", default="to_seed",
                        choices=("to_seed", "induced", "star"),
                        help="to_seed: directed neighbour->seed edges only (default); "
                             "induced: every edge among the batch nodes; "
                             "star: seed-neighbour edges, both directions")
    parser.add_argument("--star", action="store_true",
                        help="seed-to-neighbour edges only, dropping neighbour-neighbour")
    parser.add_argument("--composition-out", default=None, help="write the K x K table to .csv")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    ds, variant = paths.resolve_bundle(args)
    if args.self_test:
        return self_test(ds.bundle_dir, variant, args.graph, args.batch_cells)

    data = CellGraphDataset.from_dataset(
        ds, variant, graph=args.graph, label_key=args.label_key,
        batch_cells=args.batch_cells, min_apposed_um=args.min_apposed_um,
        min_wall_um=args.min_wall_um, max_gap_um=args.max_gap_um,
        embeddings=args.embeddings, embedding_dim=args.embedding_dim,
        edge_mode=args.edge_mode, image_scope=args.image_scope,
        induced=not args.star, drop_unassigned=args.drop_unassigned,
        resident=args.resident, counts_dtype=args.counts_dtype,
    )
    print(f"\n{len(data)} batches of {args.batch_cells} seed cells")
    print(f"types: {', '.join(data.type_names)}\n")
    print(data.neighbour_composition.round(3).to_string())
    if args.composition_out:
        data.neighbour_composition.to_csv(args.composition_out)
        log.info("Wrote %s", args.composition_out)
    batch = next(iter(data))
    print(f"\nexample batch: {batch!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
