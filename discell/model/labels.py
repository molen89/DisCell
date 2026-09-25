"""Robust, case-insensitive matchers for cell-type name substrings.

The lineage relabel (``data/datasets/xenium_prime_ovarian_cancer_ffpe/labels/
lineage_map_proposed.csv``, ``scripts/logs/lineage_2026-09-25/AGENT_REPORT.md``)
renames ovarian classes to a new vocabulary (e.g. ``Tumour``, ``Smooth muscle
cells``, ``Endothelial cells``, ``Mesothelial-like cyst lining``,
``Unassigned``) while the GSE lung vocabulary is unchanged. Readers across
``discell/`` matched cell-type names by ad hoc substrings (``"Tumor Cells" in
name``) that do not survive the relabel or its case changes. These helpers
centralise that matching.

Note: ``Mesothelial-like cyst lining`` is mesothelial, not tumour, under the
new map -- despite descending from the old ``Malignant Cells Lining Cyst``
class -- so ``is_tumour`` explicitly excludes anything mentioning
"mesothelial"/"unassigned"/"lining".
"""

import re

_TUMOUR_RE = re.compile(r"tumou?r|malignant|carcinoma", re.IGNORECASE)
_TUMOUR_EXCLUDE_RE = re.compile(r"mesothelial|unassigned|lining|associated|fibroblast|endothel|macrophage|stroma|immune|\bt[\s/-]?cells?\b", re.IGNORECASE)
_SMOOTH_MUSCLE_RE = re.compile(r"smooth muscle", re.IGNORECASE)
_ENDOTHELIAL_RE = re.compile(r"endothelial", re.IGNORECASE)


def is_tumour(name: str) -> bool:
    """True if ``name`` denotes a tumour/malignant cell type.

    Case-insensitive match on tumou?r|malignant|carcinoma, excluding names
    that mention mesothelial/unassigned/lining (the cyst-lining class is
    mesothelial, not tumour, under the new lineage map)."""
    return bool(_TUMOUR_RE.search(name)) and not bool(_TUMOUR_EXCLUDE_RE.search(name))


def is_smooth_muscle(name: str) -> bool:
    """True if ``name`` denotes a smooth-muscle cell type (case-insensitive)."""
    return bool(_SMOOTH_MUSCLE_RE.search(name))


def is_endothelial(name: str) -> bool:
    """True if ``name`` denotes an endothelial cell type (case-insensitive)."""
    return bool(_ENDOTHELIAL_RE.search(name))
