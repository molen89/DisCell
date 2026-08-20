#!/usr/bin/env python3
"""What one training batch carries.

A batch picks ``batch_cells`` **seed** cells, pulls their 1-hop neighbours, and
serves the directed **neighbour -> seed** edges. Fields by the axis they are
aligned to:

============  ================================================  ==========
seeds         ``image_embedding, seed_index, seed_composition``  fixed
nodes         ``x, y, pos, node_ids, node_index``                varies
edges         ``edge_index, edge_attr, edge_id, into_j``         varies
============  ================================================  ==========

Model inputs are ``x`` (raw counts), ``edge_attr`` and ``image_embedding``.
``pos`` and ``node_ids`` are auxiliary -- for plotting and debugging.

A batch carries only what varies with it. The fixed per-dataset arrays --
the composition table, the per-cell niche, the smoothing weights ``beta`` --
live on :attr:`~discell.data.loader.CellGraphDataset.constant` and are read
from there::

    beta = data.constant.edge_beta(batch)       # (n_edges,) aligned to edge_index
    table = data.constant.neighbour_composition # (K, K)

``edge_id`` is what makes that possible: the row each batch edge occupies in the
whole-slide edge list, with ``into_j`` recording which direction it points, so
any per-edge quantity held on the dataset can be gathered for a batch.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Edge features served in ``edge_attr``, in column order. All in microns:
#:
#: * ``centroid_dist_um``  -- centre to centre
#: * ``apposed_wall_um``   -- membrane the two cells actually share
#: * ``wall_dist_um``      -- closest approach between the two boundaries
#:
#: The three are complementary: two cells can be close by centroid yet share no
#: membrane (a small cell beside a large one), and apposed wall is zero for
#: every pair that never comes within the wall tolerance, where wall distance
#: still says *how far* apart they are.
DEFAULT_EDGE_METRICS = ("centroid_dist_um", "apposed_wall_um", "wall_dist_um")


@dataclass
class TorchBatch:
    """One subgraph batch as tensors, ready for a model.

    ``edge_index`` is ``(2, n_edges)``.
    """

    x: "torch.Tensor"                # (n_nodes, n_genes)   raw counts
    edge_index: "torch.Tensor"       # (2, n_edges)
    edge_attr: "torch.Tensor"        # (n_edges, len(edge_attr_names))
    y: "torch.Tensor"                # (n_nodes, n_types)   one-hot
    image_embedding: "torch.Tensor"  # (n_seeds, dim)
    seed_index: "torch.Tensor"       # (n_seeds,)
    node_ids: list[str]
    edge_attr_names: tuple[str, ...] = DEFAULT_EDGE_METRICS
    #: (n_edges,) row of each edge in the whole-slide edge list, for gathering
    #: any per-edge array held on ``dataset.constant``.
    edge_id: "torch.Tensor | None" = None
    #: (n_edges,) True where the edge points into ``edge_j``, which selects the
    #: column of ``constant.beta`` that applies.
    into_j: "torch.Tensor | None" = None
    #: (n_seeds, K) the row of the composition table belonging to each seed's
    #: own type -- a gather from ``constant.neighbour_composition``, kept here
    #: because the seeds vary per batch.
    seed_composition: "torch.Tensor | None" = None
    #: (n_nodes, 2) centroid microns. NOT a model input: the spatial information
    #: is already in ``edge_attr``, and absolute coordinates would let a model
    #: memorise position.
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
            edge_id=None if self.edge_id is None else self.edge_id.to(device),
            into_j=None if self.into_j is None else self.into_j.to(device),
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
