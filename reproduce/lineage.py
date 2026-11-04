"""The paper's ``lineage`` labels: a reviewed source-label -> lineage map.

A map (``labels/<section>/lineage_map.csv``: ``source_label, lineage, ...``)
sends every source label (10x's ``cell_group`` or ``graphclust``) to a lineage.
Missing source labels count as ``Unassigned`` first, as the model folds them. A
row whose lineage starts ``per-cell:`` is resolved cell by cell from
reassignment tables (``cell_id, assigned_lineage[, source_label]``; the TMA core
only). Two ovarian classes are author decisions, passed as *choices*.
A source label absent from the map is an error, never a silent default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

UNASSIGNED = "Unassigned"
PER_CELL = "per-cell:"
TUMOR = "Tumour"
#: the author's two decisions on ovarian FFPE: source label -> lineage per choice
CHOICES = {"sox2ot": ("SOX2-OT+ Tumor Cells", {"unassigned": UNASSIGNED, "tumor": TUMOR}),
           "cyst": ("Malignant Cells Lining Cyst",
                    {"mesothelial": "Mesothelial-like cyst lining", "tumor": TUMOR})}


def resolve_map(table: pd.DataFrame, choices: Mapping[str, str]) -> tuple[dict, set]:
    """``({source: lineage}, {per-cell sources})`` with the choices filled in."""
    if table["source_label"].duplicated().any():
        raise ValueError("duplicated source labels in the map")
    choice_of = {source: name for name, (source, _) in CHOICES.items()}
    fixed, per_cell = {}, set()
    for source, lineage in zip(table["source_label"].astype(str), table["lineage"].astype(str)):
        if source in choice_of:
            name = choice_of[source]
            if name not in choices:
                raise ValueError(f"{source!r} is in the map: choose {name} "
                                 f"({'|'.join(CHOICES[name][1])})")
            fixed[source] = CHOICES[name][1][choices[name]]
        elif lineage.startswith(PER_CELL):
            per_cell.add(source)
        else:
            fixed[source] = lineage
    return fixed, per_cell


def fold_unassigned(series: pd.Series) -> pd.Series:
    series = series.astype(object)
    spellings = {str(v).casefold(): str(v) for v in series.dropna().unique()}
    return series.where(series.notna(),
                        spellings.get(UNASSIGNED.casefold(), UNASSIGNED)).astype(str)


def lineage(source: pd.Series, map_csv: str | Path, *, choices: Mapping[str, str] = {},
            reassignments: Sequence[str | Path] = ()) -> pd.Series:
    """The lineage of every cell of *source* (index: cell ids)."""
    fixed, per_cell = resolve_map(pd.read_csv(map_csv), choices)
    source = fold_unassigned(source)
    unknown = sorted(set(source) - set(fixed) - per_cell)
    if unknown:
        raise ValueError(f"source labels missing from the map: {unknown}")
    out = source.map(fixed)
    hit = source.isin(per_cell)
    if hit.any():
        frames = [pd.read_csv(p, dtype={"cell_id": str}) for p in reassignments]
        if not frames:
            raise ValueError("the map has per-cell classes: pass the reassignment tables")
        table = pd.concat(frames, ignore_index=True)
        if table["cell_id"].duplicated().any():
            raise ValueError("a cell is reassigned twice")
        table = table.set_index("cell_id")
        ids = source.index[hit]
        missing = ids.difference(table.index)
        if len(missing):
            raise ValueError(f"{len(missing)} cells of per-cell classes have no reassignment")
        rows = table.loc[ids]
        if "source_label" in rows:
            recorded = rows["source_label"]
            bad = recorded.notna() & (recorded.astype(str).to_numpy()
                                      != source[hit].to_numpy())
            if bad.any():
                raise ValueError(f"{int(bad.sum())} reassignments disagree with the source label")
        out[hit] = rows["assigned_lineage"].astype(str).to_numpy()
    if out.isna().any():
        raise AssertionError("unresolved cells")
    return out.astype(str)
