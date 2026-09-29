"""The held-out evaluation mask: which cells may be a metric's *target*.

Devlog "Unassigned is a training class and a neighbour, never a metric
target (author's decision, 2026-09-28)". Cells labelled ``Unassigned`` (the
annotation's residue, spelled per :data:`discell.data.loader.UNASSIGNED`)
stay in training, in the graph, in the composition ``y`` and in the leak
influx exactly as before. What changes is only which cells an evaluation read
*grades*: every read applies :func:`metric_target_mask` (or, for a set of type
indices, :func:`exclude_types`) at the point where it selects its cells, so

* NMI, the probe, the mirror, degeneracy, cycle, recon modes, the I(niche; w)
  guard, external criteria, subtype recovery and marker pairs grade target
  cells only, and every within-type permutation floor permutes within the
  target types only (the rows it sees are target rows);
* a transport panel never moves an excluded type; the atlas has no excluded
  centre;
* neighbours, context, composition and influx are the model's and are never
  masked.

A slide without an excluded type reads bit-identically with the mask on and
off (every helper returns its input unchanged when nothing is excluded).

The switch: ``DISCELL_EVAL_INCLUDE_UNASSIGNED=1`` in the environment (or
:func:`set_include_unassigned` in-process) empties the exclusion list and so
restores the reads as they were before the rule -- for the one-time "both
ways" report only.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np

from discell.data.loader import UNASSIGNED

#: the types that are never a metric target
EXCLUDED_TYPES = (UNASSIGNED,)
#: environment switch restoring the pre-rule reads ("1" = include Unassigned)
INCLUDE_ENV = "DISCELL_EVAL_INCLUDE_UNASSIGNED"

_override: bool | None = None


def set_include_unassigned(flag: bool | None) -> None:
    """In-process switch; ``None`` defers to the environment again."""
    global _override
    _override = flag


def include_unassigned() -> bool:
    """True when the switch restores the pre-rule reads."""
    if _override is not None:
        return bool(_override)
    return os.environ.get(INCLUDE_ENV, "").strip().lower() in ("1", "true",
                                                               "yes")


def exclusions(exclude: Iterable[str] = EXCLUDED_TYPES) -> tuple[str, ...]:
    """The type names excluded right now: *exclude*, or none under the switch."""
    return () if include_unassigned() else tuple(str(e) for e in exclude)


def _folded(names) -> np.ndarray:
    return np.asarray([str(n).casefold() for n in np.asarray(names).ravel()])


def is_excluded(name, exclude: Iterable[str] = EXCLUDED_TYPES) -> bool:
    """Is the type *name* excluded (case-folded, as the loader folds it)?"""
    return str(name).casefold() in {e.casefold() for e in exclusions(exclude)}


def target_types(type_names: Sequence,
                 exclude: Iterable[str] = EXCLUDED_TYPES) -> np.ndarray:
    """(K,) bool over the type vocabulary: True = a metric target."""
    excluded = [e.casefold() for e in exclusions(exclude)]
    return ~np.isin(_folded(type_names), excluded)


def excluded_type_ids(type_names: Sequence,
                      exclude: Iterable[str] = EXCLUDED_TYPES) -> np.ndarray:
    """Indices into *type_names* of the excluded types."""
    return np.flatnonzero(~target_types(type_names, exclude))


def exclude_types(type_ids, type_names: Sequence,
                  exclude: Iterable[str] = EXCLUDED_TYPES) -> np.ndarray:
    """*type_ids* with the excluded types removed, order kept (int64)."""
    ids = np.asarray(type_ids, dtype=np.int64)
    return ids[target_types(type_names, exclude)[ids]] if len(ids) else ids


def metric_target_mask(t_names_or_ids, type_names: Sequence | None = None,
                       exclude: Iterable[str] = EXCLUDED_TYPES) -> np.ndarray:
    """(n,) bool: which cells may be a metric's target.

    *t_names_or_ids* is either per-cell type indices into *type_names* or
    per-cell type names (then *type_names* is not needed).
    """
    t = np.asarray(t_names_or_ids)
    if t.dtype.kind in "iu":
        if type_names is None:
            raise ValueError("type indices need the type_names they index")
        return target_types(type_names, exclude)[t.astype(np.int64)]
    excluded = [e.casefold() for e in exclusions(exclude)]
    return ~np.isin(_folded(t), excluded)


def compact_types(t: np.ndarray, type_names: Sequence,
                  exclude: Iterable[str] = EXCLUDED_TYPES) -> np.ndarray:
    """Type indices of *target* cells re-indexed with the excluded types
    removed from the index space, so ``t.max() + 1`` counts target types
    only (the k of the NMI's k-means). *t* itself when nothing is excluded."""
    keep = target_types(type_names, exclude)
    if keep.all():
        return t
    return (np.cumsum(keep) - 1)[np.asarray(t, dtype=np.int64)]


def nmi_inputs(z: np.ndarray, t: np.ndarray, type_names: Sequence,
               exclude: Iterable[str] = EXCLUDED_TYPES):
    """``(z, t)`` for ``metrics.z_type_nmi`` under the mask: target rows, and
    labels compacted so the k-means asks for one cluster per target type.
    The inputs themselves when nothing is excluded."""
    mask = metric_target_mask(t, type_names, exclude)
    if mask.all() and target_types(type_names, exclude).all():
        return z, t
    return z[mask], compact_types(t[mask], type_names, exclude)


def record(type_names: Sequence | None = None, t=None,
           exclude: Iterable[str] = EXCLUDED_TYPES) -> dict:
    """What a read's output says about the mask it was made under."""
    out = {"excluded_types": list(exclusions(exclude)),
           "include_unassigned_switch": include_unassigned(),
           # the read's thread count: k-means niche labels depend on it
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "rule": "devlog 2026-09-28: Unassigned is a training class and a "
                   "neighbour, never a metric target"}
    if type_names is not None:
        names = [str(n) for n in type_names]
        out["excluded_present"] = [names[g] for g in
                                   excluded_type_ids(names, exclude)]
        if t is not None:
            out["n_cells_excluded"] = int(
                (~metric_target_mask(t, names, exclude)).sum())
    return out


def same_mask(stored: dict | None, exclude: Iterable[str] = EXCLUDED_TYPES
              ) -> bool:
    """Was a record written under the exclusions in force now? A record
    without an ``eval_mask`` entry predates the rule: it excluded nothing."""
    excluded = (stored or {}).get("excluded_types", [])
    return ({str(e).casefold() for e in excluded}
            == {e.casefold() for e in exclusions(exclude)})
