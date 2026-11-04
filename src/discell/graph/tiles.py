"""Spatial tiles, their two-hop rings, and the held-out tile split.

Training batches are contiguous spatial tiles. A tile's seeds plus two graph
hops hold everything one step needs exactly::

    ring1 = neighbors of the seeds   (their c, w and rho feed the seeds' influx)
    ring2 = neighbors of ring1       (their types are GAT sources for ring1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from discell.graph.kernel import ModelGraph


def spatial_tiles(positions: np.ndarray, max_cells: int) -> list[np.ndarray]:
    """Partition cells into contiguous tiles of at most *max_cells*.

    Recursive median split along the longer axis; deterministic, and each tile
    is returned as sorted cell indices.
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


def split_tiles(tiles: list[np.ndarray], val_fraction: float, seed: int
                ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``(train, heldout)`` tiles: a random ``val_fraction`` (at least one) is held out.

    Uses the RNG stream ``[seed, 2]``, independent of every other draw.
    """
    order = np.random.default_rng([seed, 2]).permutation(len(tiles))
    n_val = max(1, int(round(val_fraction * len(tiles))))
    return [tiles[i] for i in order[n_val:]], [tiles[i] for i in order[:n_val]]


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
    """One step's indices in the layout ``[seeds | ring1 | ring2]``.

    ``gat_*`` are the directed in-edges of the context nodes (seeds and ring1)
    in local indices; ``leak_*`` are their prefix that ends at a seed, with the
    kernel weights ``beta``.
    """

    nodes: np.ndarray            # global cell indices
    n_seeds: int
    n_ring1: int
    gat_src: np.ndarray
    gat_dst: np.ndarray
    spillover_src: np.ndarray
    spillover_dst: np.ndarray
    spillover_beta: np.ndarray

    @property
    def seeds(self) -> np.ndarray:
        return self.nodes[:self.n_seeds]

    @property
    def n_context(self) -> int:
        """Nodes that need a context vector: seeds plus ring1."""
        return self.n_seeds + self.n_ring1


def tile_batch(graph: ModelGraph, seeds: np.ndarray) -> TileBatch:
    """Rings of *seeds* and both edge lists, from one CSR slice of ``in_edges``."""
    seeds = np.asarray(seeds)
    ring1, ring2 = rings(graph.in_edges, seeds)
    nodes = np.concatenate([seeds, ring1, ring2])

    local = np.full(graph.n_cells, -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))

    context = nodes[:len(seeds) + len(ring1)]
    sub = graph.in_edges[context]
    gat_dst = np.repeat(np.arange(len(context)), np.diff(sub.indptr))
    gat_src = local[sub.indices]
    if (gat_src < 0).any():
        raise AssertionError("edge source outside the two-hop halo")

    n_spillover = int(sub.indptr[len(seeds)])
    return TileBatch(
        nodes=nodes, n_seeds=len(seeds), n_ring1=len(ring1),
        gat_src=gat_src, gat_dst=gat_dst,
        spillover_src=gat_src[:n_spillover], spillover_dst=gat_dst[:n_spillover],
        spillover_beta=sub.data[:n_spillover],
    )
