#!/usr/bin/env python3
"""The fixed per-dataset quantities a model treats as constants.

None of these depend on the batch, so they are computed once when the dataset
is opened and then only indexed:

* **composition table** (K, K) -- mean neighbour-type mix per cell type
* **soft niche labels** (n_cells, K) -- each cell's own neighbour-type mix
* **smoothing weights** beta (n_edges, 2) -- row-stochastic neighbour weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("discell.data.priors")

#: Decay length for the smoothing weights, shared with the model's leakage
#: kernel so the two never disagree. 20 um is the contamination decay length
#: from the DisCell model spec (discell_specs.md) -- roughly two cell diameters at Xenium's ~11 um
#: nearest-neighbour spacing. Whether beta at this tau still concentrates on
#: the immediate ring is a standing validation item (see docs/devlog.md).
DEFAULT_BETA_TAU_UM = 20.0


def smoothing_weights(face_um: np.ndarray, dist_um: np.ndarray, edge_i: np.ndarray,
                      edge_j: np.ndarray, n_cells: int,
                      tau_um: float = DEFAULT_BETA_TAU_UM) -> np.ndarray:
    """``beta_ij = face_ij * exp(-d_ij / tau)``, row-normalised per destination.

    Normalising over the in-edges of each destination makes ``sum_j beta_ij``
    exactly 1, so a distribution smoothed over the neighbourhood stays on its
    simplex. *face_um* should be the shared **Voronoi face** length, which is
    strictly positive on every edge of the partition -- unlike the segmented
    shared wall, which is zero for 98% of contact pairs on Xenium.

    Returns ``(n_edges, 2)``: column 0 is the weight *into* ``edge_j``, column 1
    *into* ``edge_i``. They differ because the normaliser does. A cell with no
    neighbours keeps an all-zero row.
    """
    raw = face_um * np.exp(-dist_um / float(tau_um))
    total = np.zeros(n_cells, dtype=np.float64)
    np.add.at(total, edge_i, raw)
    np.add.at(total, edge_j, raw)
    safe = np.maximum(total, 1e-12)
    log.info("beta: tau=%.1f um over %d edges (%d cells keep a zero row)",
             tau_um, len(raw), int((total == 0).sum()))
    return np.stack([raw / safe[edge_j], raw / safe[edge_i]], axis=1).astype(np.float32)


def composition(adjacency, labels: np.ndarray, degrees: np.ndarray,
                type_index: np.ndarray, type_names: np.ndarray):
    """``(table, niche, support)`` -- the neighbour-type statistics.

    *niche* is per cell, the mean of its neighbours' one-hot labels; *table* is
    that averaged again over the cells of each type. Isolated cells have no
    neighbourhood (0/0) and are excluded from the table rather than counted as
    zeros, which would drag every row toward the origin.

    float64 throughout: averaging hundreds of thousands of float32 rows
    accumulates ~1e-5 of error, enough that the rows visibly stop summing to 1.
    """
    counts = adjacency @ labels.astype(np.float64)
    degree = degrees.astype(np.float64)
    connected = degree > 0
    niche = np.zeros_like(counts)
    niche[connected] = counts[connected] / degree[connected, None]

    n_types = len(type_names)
    rows = np.zeros((n_types, n_types), dtype=np.float64)
    used = np.zeros(n_types, dtype=np.int64)
    for k in range(n_types):
        mask = (type_index == k) & connected
        used[k] = int(mask.sum())
        if used[k]:
            rows[k] = niche[mask].mean(axis=0)

    table = pd.DataFrame(rows, index=type_names, columns=type_names)
    table.columns.name = "neighbour type"
    if (~connected).any():
        log.info("Composition excludes %d isolated cells (%.1f%%)",
                 int((~connected).sum()), 100 * (~connected).mean())
    return table, niche.astype(np.float32), pd.Series(used, index=type_names,
                                                      name="n_cells_used")


@dataclass(frozen=True)
class Constants:
    """The whole-slide arrays a model treats as fixed, as tensors.

    Reachable as ``dataset.constant``. They are deliberately *not* attached to
    batches: a (K, K) table riding along on every batch of an epoch is pure
    duplication, and ``beta`` is only ever wanted indexed by the edges a batch
    actually carries -- which :meth:`edge_beta` does.

    ``dataset.neighbour_composition`` keeps the labelled DataFrame for
    inspection and plotting; this is the same table as a float32 tensor.
    """

    #: (K, K) mean neighbour-type distribution per cell type.
    neighbour_composition: "torch.Tensor"
    #: (n_cells, K) each cell's own observed neighbour-type mix.
    niche: "torch.Tensor"
    #: (n_edges, 2) row-stochastic smoothing weights over the *whole* graph;
    #: column 0 is the weight into ``edge_j``, column 1 into ``edge_i``.
    #: ``None`` when the graph stores no shared face to weight by.
    beta: "torch.Tensor | None"
    #: The label space, in the order the columns of the two tables follow.
    type_names: tuple[str, ...]

    def edge_beta(self, batch) -> "torch.Tensor | None":
        """``beta`` for *batch*'s edges, aligned to its ``edge_index``.

        The whole-slide array is direction-agnostic, so which column applies
        depends on which way each batch edge points -- that is what
        ``batch.into_j`` records.
        """
        import torch

        if self.beta is None:
            return None
        return torch.where(batch.into_j,
                           self.beta[batch.edge_id, 0],
                           self.beta[batch.edge_id, 1])

    def to(self, device) -> "Constants":
        return Constants(
            neighbour_composition=self.neighbour_composition.to(device),
            niche=self.niche.to(device),
            beta=None if self.beta is None else self.beta.to(device),
            type_names=self.type_names,
        )
