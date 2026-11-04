"""The AnnData input contract: counts, labels and centroids.

Input:

* ``.X`` or ``.layers[counts_layer]``: raw counts, cells x genes;
* ``.obs[label_key]``: cell type labels; missing values count as ``Unassigned``;
* ``.obsm['spatial']``: centroids in micrometres, or in pixels when
  ``.uns['discell']['microns_per_pixel']`` is set;
* optionally ``.obsm[image_key]``: a per-cell image descriptor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

#: Label for cells without an annotation. They stay in the graph and in
#: training as a type of their own.
UNASSIGNED = "Unassigned"


def fold_labels(labels, unassigned: str = UNASSIGNED) -> tuple[np.ndarray, np.ndarray]:
    """``(type_index, type_names)`` from a label column.

    Missing values become *unassigned*, spelled as the column already spells
    it when a case-insensitive match exists. Type names are sorted, which fixes
    the one-hot order; ``type_index`` is int64.
    """
    series = pd.Series(labels).astype(object)
    spellings = {str(v).casefold(): str(v) for v in series.dropna().unique()}
    series = series.where(series.notna(),
                          spellings.get(unassigned.casefold(), unassigned)).astype(str)
    type_names = np.asarray(sorted(series.unique()))
    index = pd.Series(np.arange(len(type_names)), index=pd.Index(type_names))
    type_index = index.reindex(series.to_numpy()).to_numpy().astype(np.int64)
    return type_index, type_names


def counts_matrix(adata, layer: str | None = None) -> sp.csr_matrix:
    """The raw counts as float32 CSR."""
    matrix = adata.X if layer is None else adata.layers[layer]
    matrix = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
    return matrix.astype(np.float32)


def positions_um(adata, key: str = "spatial") -> np.ndarray:
    """Centroids in micrometres, float32 ``(N, 2)``."""
    pos = np.asarray(adata.obsm[key], dtype=np.float32)
    mpp = adata.uns.get("discell", {}).get("microns_per_pixel")
    if mpp is not None:
        pos = (pos * float(mpp)).astype(np.float32)
    return pos
