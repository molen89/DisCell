#!/usr/bin/env python3
"""Apply the reviewed lineage mapping as obs column ``lineage`` (R9, 2026-09-25).

Devlog "Lineage-level labels (R9, motivation ...)" and "Lineage mapping tables
drafted (R9, results ...)". Reads ``labels/lineage_map_proposed.csv`` (one row
per source label: ``source_label, lineage, ...``) and, where a row's lineage is
``per-cell: ...``, the per-cell reassignment files beside it
(``proliferating_reassignment_proposed.csv``, ``state_reassignment_proposed.csv``:
``cell_id, assigned_lineage, ...``). Writes the result into every bundle
variant of the dataset as a categorical obs column ``lineage``, in place --
only that column; ``uns`` (``default_label`` included) is left alone, so a fit
reads it only with ``--label-key lineage``.

The source column is the bundle's ``uns['default_label']`` (``cell_group`` on
ovarian and GSE, ``graphclust`` on lung and FF) with NaN folded onto
``Unassigned`` exactly as the loader folds it -- the mapping tables were built
on that view.

The two author decisions on ovarian are arguments, never defaults:
``--sox2ot unassigned|tumour`` (the "SOX2-OT+ Tumor Cells" class) and
``--cyst mesothelial|tumour`` ("Malignant Cells Lining Cyst"). The GSE solo
and dual sections are always applied together: one vocabulary, asserted equal
between the two, and the observed class set of every shared variant asserted
equal too (the held-out-section protocol requires identical type indices).

Writes ``labels/lineage_map_applied.csv`` (source label -> lineage -> cells per
variant) and ``labels/lineage_map_applied.json`` (sha256 of that CSV and of
every input table, the choices, the vocabulary, the counts).

    python -m discell.experiments.apply_lineage \\
        --dataset xenium_prime_ovarian_cancer_ffpe \\
        --sox2ot unassigned --cyst mesothelial --dry-run
    python -m discell.experiments.apply_lineage \\
        --dataset gse315411_pdltma06_11_prime_solo --dry-run   # + dual

``--dry-run`` prints the class counts before / after and every check, and
writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from discell.data.loader import UNASSIGNED

log = logging.getLogger("discell.experiments.apply_lineage")

LINEAGE_KEY = "lineage"
MAP_FILE = "lineage_map_proposed.csv"
REASSIGN_FILES = ("proliferating_reassignment_proposed.csv",
                  "state_reassignment_proposed.csv")
APPLIED_CSV = "lineage_map_applied.csv"
APPLIED_JSON = "lineage_map_applied.json"
PER_CELL = "per-cell:"
TUMOUR = "Tumour"
#: the two grey classes the author decides (ovarian): source label -> the
#: lineage each choice maps it to
CHOICES = {
    "sox2ot": ("SOX2-OT+ Tumor Cells",
               {"unassigned": UNASSIGNED, "tumour": TUMOUR}),
    "cyst": ("Malignant Cells Lining Cyst",
             {"mesothelial": "Mesothelial-like cyst lining", "tumour": TUMOUR}),
}
#: sections that share one vocabulary (the held-out-section protocol)
PAIRS = {"gse315411_pdltma06_11_prime_solo": "gse315411_pdltma06_10_prime_dual",
         "gse315411_pdltma06_10_prime_dual": "gse315411_pdltma06_11_prime_solo"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- the mapping ------------------------------------------------------------

def resolve_map(proposed: pd.DataFrame, choices: dict[str, str | None]) -> pd.DataFrame:
    """``source_label, lineage, rule`` with the author's choices filled in.

    ``rule`` is ``map`` (the table's lineage), ``per-cell`` (reassigned cell
    by cell) or ``choice:<name>=<value>``. A choice class present in the
    table without its choice given is an error: the author decides it.
    """
    if proposed["source_label"].duplicated().any():
        dup = proposed.loc[proposed["source_label"].duplicated(), "source_label"]
        raise ValueError(f"duplicated source labels in the map: {sorted(dup)}")
    rows = []
    choice_of = {source: name for name, (source, _) in CHOICES.items()}
    for source, lineage in zip(proposed["source_label"].astype(str),
                               proposed["lineage"].astype(str)):
        if source in choice_of:
            name = choice_of[source]
            options, value = CHOICES[name][1], choices.get(name)
            if value is None:
                raise ValueError(f"{source!r} is in the map: pass --{name} "
                                 f"({'|'.join(options)})")
            rows.append((source, options[value], f"choice:{name}={value}"))
        elif lineage.startswith(PER_CELL):
            rows.append((source, None, "per-cell"))
        else:
            rows.append((source, lineage, "map"))
    return pd.DataFrame(rows, columns=["source_label", "lineage", "rule"])


def load_reassignments(labels_dir: Path) -> pd.DataFrame:
    """Per-cell targets from every reassignment file present, indexed by
    cell id (``assigned_lineage``, ``source_label`` where the file has it)."""
    frames = []
    for name in REASSIGN_FILES:
        path = labels_dir / name
        if path.exists():
            frame = pd.read_csv(path, dtype={"cell_id": str})
            frame["file"] = name
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["assigned_lineage", "source_label", "file"],
                            index=pd.Index([], name="cell_id"))
    out = pd.concat(frames, ignore_index=True)
    if out["cell_id"].duplicated().any():
        raise ValueError(f"{int(out['cell_id'].duplicated().sum())} cell ids "
                         "reassigned twice across the reassignment files")
    return out.set_index("cell_id")


def vocabulary(resolved: pd.DataFrame, reassigned: pd.DataFrame) -> list[str]:
    """Every lineage the applied table can produce, sorted (the categories)."""
    names = set(resolved["lineage"].dropna()) | set(reassigned["assigned_lineage"])
    return sorted(map(str, names))


def fold_unassigned(series: pd.Series) -> pd.Series:
    """NaN -> the panel's own spelling of Unassigned, as the loader does
    (``CellGraphDataset._build_labels``)."""
    series = series.astype(object)
    spellings = {str(v).casefold(): str(v) for v in series.dropna().unique()}
    return series.where(series.notna(),
                        spellings.get(UNASSIGNED.casefold(), UNASSIGNED)).astype(str)


def lineage_of(source: pd.Series, resolved: pd.DataFrame,
               reassigned: pd.DataFrame) -> pd.Series:
    """The lineage of every cell (index = cell id, values = folded source
    label). Unmapped source labels, per-cell classes with a cell missing
    from the reassignment files, and a reassignment whose recorded source
    label disagrees with the bundle's are errors."""
    unknown = sorted(set(source) - set(resolved["source_label"]))
    if unknown:
        raise ValueError(f"source labels missing from the map: {unknown}")
    fixed = resolved.dropna(subset=["lineage"]).set_index("source_label")["lineage"]
    out = source.map(fixed)
    per_cell = source.isin(resolved.loc[resolved["rule"] == "per-cell", "source_label"])
    if per_cell.any():
        ids = source.index[per_cell]
        missing = ids.difference(reassigned.index)
        if len(missing):
            raise ValueError(f"{len(missing)} cells of per-cell classes have no "
                             f"reassignment (e.g. {list(missing[:3])})")
        hit = reassigned.loc[ids]
        if "source_label" in hit and hit["source_label"].notna().any():
            recorded = hit["source_label"]
            bad = recorded.notna() & (recorded.astype(str)
                                      != source[per_cell].astype(str).to_numpy())
            if bad.any():
                raise ValueError(f"{int(bad.sum())} reassignment rows disagree "
                                 "with the bundle's source label")
        out[per_cell] = hit["assigned_lineage"].astype(str).to_numpy()
    assert out.notna().all()
    return out.astype(str)


# -- bundles ----------------------------------------------------------------

def read_obs(h5ad: Path) -> tuple[pd.DataFrame, str]:
    """(obs, the bundle's default label column) without loading X."""
    import h5py
    from anndata.io import read_elem

    with h5py.File(h5ad, "r") as f:
        obs = read_elem(f["obs"])
        key = f["uns"]["default_label"][()] if "default_label" in f["uns"] else None
    key = key.decode() if isinstance(key, bytes) else key
    return obs, (str(key) if key else "cell_group")


def write_column(h5ad: Path, values: pd.Series, categories: list[str],
                 key: str = LINEAGE_KEY) -> None:
    """Write (or replace) one categorical obs column in place; nothing else
    in the file is touched."""
    import h5py
    from anndata.io import read_elem, write_elem

    with h5py.File(h5ad, "r+") as f:
        obs = f["obs"]
        index = np.asarray(read_elem(obs[obs.attrs["_index"]])).astype(str)
        if list(index) != list(values.index):
            raise ValueError(f"{h5ad}: obs order changed since it was read")
        if key in obs:
            del obs[key]
        write_elem(obs, key, pd.Categorical(values.to_numpy(), categories=categories))
        order = [str(c) for c in obs.attrs["column-order"]]
        if key not in order:
            obs.attrs["column-order"] = np.array(order + [key], dtype=object)


# -- one dataset ------------------------------------------------------------

@dataclass
class Plan:
    """Everything decided for one dataset before anything is written."""

    dataset: str
    root: Path
    source_key: str
    resolved: pd.DataFrame
    reassigned: pd.DataFrame
    vocab: list[str]
    before: dict            # variant -> source-label counts
    after: dict             # variant -> lineage counts
    columns: dict           # variant -> lineage Series (index = cell id)
    applied: pd.DataFrame   # source_label, lineage, rule, n_<variant>...
    changed: dict           # variant -> cells whose existing `lineage` differs


def plan(dataset: str, root: Path, choices: dict[str, str | None],
         variants: Sequence[str] | None = None) -> Plan:
    labels_dir, bundle_dir = root / "labels", root / "bundle"
    proposed = pd.read_csv(labels_dir / MAP_FILE)
    resolved = resolve_map(proposed, choices)
    reassigned = load_reassignments(labels_dir)
    vocab = vocabulary(resolved, reassigned)
    variants = list(variants or sorted(p.stem for p in bundle_dir.glob("*.h5ad")))
    if not variants:
        raise FileNotFoundError(f"no bundle under {bundle_dir}")
    before, after, columns, changed, transitions, keys = {}, {}, {}, {}, {}, set()
    for variant in variants:
        obs, key = read_obs(bundle_dir / f"{variant}.h5ad")
        keys.add(key)
        source = fold_unassigned(obs[key])
        source.index = obs.index.astype(str)
        lineage = lineage_of(source, resolved, reassigned)
        columns[variant] = lineage
        before[variant] = source.value_counts()
        after[variant] = lineage.value_counts()
        changed[variant] = (int((obs[LINEAGE_KEY].astype(str).to_numpy()
                                 != lineage.to_numpy()).sum())
                            if LINEAGE_KEY in obs else None)
        transitions[variant] = pd.DataFrame(
            {"source_label": source.to_numpy(), "lineage": lineage.to_numpy()}
        ).value_counts()
    if len(keys) != 1:
        raise ValueError(f"variants disagree on the default label: {sorted(keys)}")
    rule = resolved.set_index("source_label")["rule"]
    pairs = sorted(set().union(*[set(t.index) for t in transitions.values()]))
    applied = pd.DataFrame(pairs, columns=["source_label", "lineage"])
    applied.insert(2, "rule", applied["source_label"].map(rule))
    for variant, t in transitions.items():
        applied[f"n_{variant}"] = [int(t.get(p, 0)) for p in pairs]
    return Plan(dataset, root, keys.pop(), resolved, reassigned, vocab, before,
                after, columns, applied, changed)


def check_pair(a: Plan, b: Plan) -> list[str]:
    """The GSE sections: one vocabulary, and the same observed classes on
    every variant both carry. Returns the problems (empty = pass)."""
    problems = []
    if a.vocab != b.vocab:
        problems.append(f"vocabularies differ: only {a.dataset}: "
                        f"{sorted(set(a.vocab) - set(b.vocab))}; only {b.dataset}: "
                        f"{sorted(set(b.vocab) - set(a.vocab))}")
    for variant in sorted(set(a.columns) & set(b.columns)):
        seen_a, seen_b = set(a.after[variant].index), set(b.after[variant].index)
        if seen_a != seen_b:
            problems.append(f"variant {variant}: observed classes differ: only "
                            f"{a.dataset}: {sorted(seen_a - seen_b)}; only "
                            f"{b.dataset}: {sorted(seen_b - seen_a)}")
    return problems


def report(p: Plan) -> str:
    lines = [f"== {p.dataset}  (source column {p.source_key!r} -> {LINEAGE_KEY!r})",
             f"   vocabulary ({len(p.vocab)}): " + ", ".join(p.vocab)]
    for variant in p.columns:
        b, a = p.before[variant], p.after[variant]
        lines.append(f"   [{variant}] {int(b.sum()):,} cells: {len(b)} source "
                     f"classes -> {len(a)} lineages"
                     + ("" if p.changed[variant] is None else
                        f"; existing `{LINEAGE_KEY}` column: "
                        f"{p.changed[variant]:,} cells would change"))
        lines.append("     before: " + "; ".join(f"{k} {v:,}" for k, v in b.items()))
        lines.append("     after:  " + "; ".join(f"{k} {v:,}" for k, v in a.items()))
    per_cell = p.applied[p.applied["rule"] == "per-cell"]
    if len(per_cell):
        lines.append("   per-cell reassignments (source -> lineage: cells per variant):")
        for _, row in per_cell.iterrows():
            counts = ", ".join(f"{c[2:]} {row[c]:,}" for c in p.applied.columns
                               if c.startswith("n_"))
            lines.append(f"     {row['source_label']} -> {row['lineage']}: {counts}")
    choices = p.applied[p.applied["rule"].str.startswith("choice")]
    for _, row in choices.drop_duplicates("source_label").iterrows():
        lines.append(f"   {row['rule']}: {row['source_label']} -> {row['lineage']}")
    return "\n".join(lines)


def write(p: Plan, choices: dict, partner: Plan | None) -> dict:
    """Write the column into every variant, then the applied table and its
    provenance record."""
    labels_dir = p.root / "labels"
    categories = sorted(set(p.vocab) | set(partner.vocab if partner else []))
    for variant, values in p.columns.items():
        write_column(p.root / "bundle" / f"{variant}.h5ad", values, categories)
        log.info("%s: wrote obs[%r] into %s.h5ad", p.dataset, LINEAGE_KEY, variant)
    csv = labels_dir / APPLIED_CSV
    p.applied.to_csv(csv, index=False)
    inputs = [labels_dir / MAP_FILE] + [labels_dir / n for n in REASSIGN_FILES
                                        if (labels_dir / n).exists()]
    record = {
        "dataset": p.dataset, "written_at": datetime.now().isoformat(timespec="seconds"),
        "obs_key": LINEAGE_KEY, "source_key": p.source_key,
        "choices": {k: v for k, v in choices.items() if v is not None},
        "applied_csv": APPLIED_CSV, "applied_csv_sha256": sha256(csv),
        "inputs_sha256": {path.name: sha256(path) for path in inputs},
        "categories": categories, "variants": list(p.columns),
        "counts_before": {v: {k: int(n) for k, n in c.items()} for v, c in p.before.items()},
        "counts_after": {v: {k: int(n) for k, n in c.items()} for v, c in p.after.items()},
        "vocabulary_checked_against": partner.dataset if partner else None,
    }
    (labels_dir / APPLIED_JSON).write_text(json.dumps(record, indent=2) + "\n")
    log.info("%s: %s sha256 %s", p.dataset, csv, record["applied_csv_sha256"])
    return record


def main(argv: Sequence[str] | None = None) -> int:
    from discell import paths

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sox2ot", choices=tuple(CHOICES["sox2ot"][1]), default=None,
                        help="ovarian 'SOX2-OT+ Tumor Cells' -> Unassigned or Tumour")
    parser.add_argument("--cyst", choices=tuple(CHOICES["cyst"][1]), default=None,
                        help="ovarian 'Malignant Cells Lining Cyst' -> its own "
                             "mesothelial lineage or Tumour")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="bundle variants to write (default: every *.h5ad)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the counts and checks; write nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    choices = {"sox2ot": args.sox2ot, "cyst": args.cyst}

    names = [args.dataset] + ([PAIRS[args.dataset]] if args.dataset in PAIRS else [])
    plans = [plan(n, paths.dataset(n).root, choices, args.variants) for n in names]
    for p in plans:
        print(report(p))
    problems = check_pair(*plans) if len(plans) == 2 else []
    if len(plans) == 2:
        print(f"== pair check {plans[0].dataset} / {plans[1].dataset}: "
              + ("PASS (one vocabulary; same observed classes on "
                 + ", ".join(sorted(set(plans[0].columns) & set(plans[1].columns)))
                 + ")" if not problems else "FAIL"))
        for problem in problems:
            print(f"   {problem}")
    if problems:
        return 1
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    for i, p in enumerate(plans):
        partner = plans[1 - i] if len(plans) == 2 else None
        record = write(p, choices, partner)
        print(f"{p.dataset}: wrote {LINEAGE_KEY!r} into {', '.join(record['variants'])}; "
              f"{APPLIED_CSV} sha256 {record['applied_csv_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
