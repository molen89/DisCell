"""The name matcher that picks a section's tumor types."""

from __future__ import annotations

import re

_TUMOR_RE = re.compile(r"tumou?r|malignant|carcinoma", re.IGNORECASE)
_TUMOR_EXCLUDE_RE = re.compile(
    r"mesothelial|unassigned|lining|associated|fibroblast|endothel|macrophage|stroma|immune"
    r"|\bt[\s/-]?cells?\b", re.IGNORECASE)


def is_tumor(name: str) -> bool:
    """A tumor type: "tumor", "tumour", "malignant" or "carcinoma", unless the name also
    says it is a stromal, immune, lining or unassigned class ("Cancer-associated
    fibroblasts", "Mesothelial-like cyst lining")."""
    return bool(_TUMOR_RE.search(name)) and not bool(_TUMOR_EXCLUDE_RE.search(name))
