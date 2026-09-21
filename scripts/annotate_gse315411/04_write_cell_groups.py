#!/usr/bin/env python
"""
Stage 4 — hand the shared labels to DisCell's standard ingest.

discell.preprocess reads a ``*cell_groups.csv`` (``cell_id, group[, color, donor]``)
beside each Xenium outs directory and stores it as the bundle's ``cell_group``
label (the curated-label path). This writes one such file per slide from
labels_<tag>.csv, stripping the ``<slide>:`` prefix the annotation stages put on
cell ids, and checks the one thing the held-out-slide design needs: the label
vocabulary is identical on both slides.

Every cell of the bundle gets a row: cells the annotation dropped at QC
(< 10 transcripts) are written as ``Unassigned`` with their donor taken from
the cell_stats files directly, so a per-donor bundle variant keeps them as
neighbours exactly as the full slide does.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed
from importlib import import_module
_stage0 = import_module("00_build_query")
SLIDES, find_bundle = _stage0.SLIDES, _stage0.find_bundle
read_selection_export = _stage0.read_selection_export


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--tag", default="hlca", help="which labels_<tag>.csv to hand over")
    p.add_argument("--name", default="GSE315411",
                   help="file stem prefix: <name>_<slide>_cell_groups.csv")
    p.add_argument("--accept", action="store_true",
                   help="required: the labels have been checked against report_<tag>.txt "
                        "and cluster_markers.csv (a cell_groups.csv beside an outs dir is "
                        "picked up by every later discell.preprocess run)")
    args = p.parse_args()
    if not args.accept:
        sys.exit("refusing to hand over labels without --accept: check the report and "
                 "cluster_markers.csv first (see README)")

    pin_gpu(-1)
    set_seed(args.seed)
    root, work = resolve_work(args)

    import pandas as pd

    labels = pd.read_csv(work / f"labels_{args.tag}.csv", keep_default_na=False)
    labels["cell_id"] = labels["cell"].str.split(":", n=1).str[1]
    vocab = {}
    for slide, meta in SLIDES.items():
        bundle = find_bundle(root, meta["sub"])
        every = pd.read_parquet(bundle / "cells.parquet", columns=["cell_id"])["cell_id"].astype(str)
        donor = pd.Series("NA", index=every.to_numpy())
        for f in sorted((root / "series" / "cell_stats").glob(f"PDLTMA006_*_{slide}_cells_stats.csv.gz")):
            ids = read_selection_export(f)["Cell ID"].astype(str)
            donor.loc[donor.index.intersection(ids)] = f.name.split("_")[1]
        sub = labels[labels["slide"] == slide].set_index("cell_id")
        out = pd.DataFrame({"cell_id": every.to_numpy(),
                            "group": sub["celltype"].reindex(every).fillna("Unassigned").to_numpy(),
                            "donor": donor.to_numpy()})
        n_qc = int(out["group"].isna().sum() + (~every.isin(sub.index)).sum())
        target = bundle / f"{args.name}_{slide}_cell_groups.csv"
        out.to_csv(target, index=False)
        vocab[slide] = set(out["group"])
        print(f"[{slide}] {len(out):,} cells ({n_qc:,} QC-dropped written as Unassigned), "
              f"{len(vocab[slide])} groups, {out['donor'].nunique() - 1} donors -> {target}")

    a, b = vocab.values()
    if a == b:
        print(f"\nlabel vocabulary identical on both slides ({len(a)} classes)")
    else:
        print(f"\nWARNING: vocabularies differ — only on {list(SLIDES)[0]}: {sorted(a - b)}; "
              f"only on {list(SLIDES)[1]}: {sorted(b - a)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
