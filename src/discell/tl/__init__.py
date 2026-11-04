"""Readouts of a fit, scverse style: compute from the AnnData, write results into it.

Each function reads what a fit stores in the AnnData (``.obsm``, ``.varm``, ``.obsp``,
``.uns['discell']``, counts and labels) and writes its record into
``.uns['discell'][...]``, read back with :func:`record`. The relocation read and the
cross-section read also take the model (they need the decoder).

Typical use::

    dc.tl.relocation(model, adata)                # a cell type moved between niches
    dc.tl.gene_split(model, adata)                # per gene: intrinsic / response / spillover

The per-gene split (:func:`gene_split`, :func:`gene_assignments`, :func:`gene_parts`)
returns its arrays rather than storing them, since they are large (types x genes x
bootstrap); ``gene_split(..., path=...)`` keeps one fit's split on disk.
"""

from discell.tl._utils import record, spatial_folds
from discell.tl.crosssection import cross_section
from discell.tl.genes import gene_assignments, gene_parts, gene_split
from discell.tl.relocate import alignment, noise_ceiling, relocation, score_shift

__all__ = ["alignment", "cross_section", "gene_assignments", "gene_parts", "gene_split",
           "noise_ceiling", "record", "relocation", "score_shift", "spatial_folds"]
