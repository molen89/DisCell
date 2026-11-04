"""Which cells a metric may grade.

Cells of an excluded type (by default the unassigned label) stay in training,
in the graph, in the neighbor composition ``y`` and in the spillover influx. They
are never a metric *target*: every graded read selects its cells with
:func:`metric_target_mask`, so within-type permutation floors permute within
the target types only. The exclusion is an explicit argument and is recorded
in each read's output (:func:`mask_record`). With nothing excluded every helper
returns its input unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from discell.io.anndata import UNASSIGNED
from discell.model.data import NUM_THREADS

#: The default exclusion: the unassigned label.
EXCLUDED_TYPES = (UNASSIGNED,)


def _folded(names) -> np.ndarray:
    return np.asarray([str(n).casefold() for n in np.asarray(names).ravel()])


def target_types(type_names: Sequence, exclude: Iterable[str] = EXCLUDED_TYPES
                 ) -> np.ndarray:
    """``(K,)`` bool over the type vocabulary: True = a metric target.

    Names are compared case-insensitively, as labels are folded.
    """
    excluded = [str(e).casefold() for e in exclude]
    return ~np.isin(_folded(type_names), excluded)


def metric_target_mask(t: np.ndarray, type_names: Sequence,
                       exclude: Iterable[str] = EXCLUDED_TYPES) -> np.ndarray:
    """``(n,)`` bool: which cells, given their type indices *t*, may be graded."""
    return target_types(type_names, exclude)[np.asarray(t, dtype=np.int64)]


def compact_types(t: np.ndarray, type_names: Sequence,
                  exclude: Iterable[str] = EXCLUDED_TYPES) -> np.ndarray:
    """Type indices of target cells re-indexed without the excluded types.

    ``t.max() + 1`` then counts target types only (the k of the NMI's
    k-means). Returns *t* itself when nothing is excluded.
    """
    keep = target_types(type_names, exclude)
    if keep.all():
        return t
    return (np.cumsum(keep) - 1)[np.asarray(t, dtype=np.int64)]


def nmi_inputs(z: np.ndarray, t: np.ndarray, type_names: Sequence,
               exclude: Iterable[str] = EXCLUDED_TYPES) -> tuple[np.ndarray, np.ndarray]:
    """``(z, t)`` for :func:`discell.evaluate.metrics.z_type_nmi` on target cells.

    Labels are compacted so the k-means asks for one cluster per target type.
    The inputs themselves when nothing is excluded.
    """
    mask = metric_target_mask(t, type_names, exclude)
    if mask.all() and target_types(type_names, exclude).all():
        return z, t
    return z[mask], compact_types(t[mask], type_names, exclude)


def mask_record(type_names: Sequence, t: np.ndarray | None = None,
                exclude: Iterable[str] = EXCLUDED_TYPES) -> dict:
    """What a read records about the mask it was made under."""
    names = [str(n) for n in type_names]
    out = {"excluded_types": [str(e) for e in exclude],
           "excluded_present": [names[g] for g in
                                np.flatnonzero(~target_types(names, exclude))],
           "kmeans_threads": NUM_THREADS}
    if t is not None:
        out["n_cells_excluded"] = int((~metric_target_mask(t, names, exclude)).sum())
    return out
