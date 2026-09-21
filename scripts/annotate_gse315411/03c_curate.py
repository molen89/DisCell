#!/usr/bin/env python
"""
Stage 3c — apply the curated (annotator-style) names.

curated_names.csv maps each unit -- a Leiden cluster, or a subcluster
"<k>.<s>" from diag_subcluster.py for the clusters that mixed lineages -- to a
name, with the marker evidence beside it. Names were decided from the
cluster's own marker genes first (diag_cluster_markers / diag_canonical_markers
/ diag_subcluster), with the two references' pseudobulk calls as a guide.
Writes labels_curated.csv (stage-4 schema, --tag curated),
labels_curated_units.csv and report_curated.txt (class sizes per slide,
cross-slide and per-donor composition JSD, vocabulary check).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed
from importlib import import_module
jsd = import_module("03_name_and_report").jsd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()
    pin_gpu(-1)
    set_seed(args.seed)
    root, work = resolve_work(args)

    import numpy as np
    import pandas as pd
    import anndata as ad
    import h5py

    names = pd.read_csv(Path(__file__).parent / "curated_names.csv", dtype=str).set_index("unit")
    rd = ad.io.read_elem if hasattr(ad, "io") else ad.experimental.read_elem
    with h5py.File(work / "query_emb.h5ad") as h:
        obs = rd(h["obs"])
    unit = obs["leiden"].astype(str)
    sub = pd.read_csv(work / "subclusters.csv", dtype=str).set_index("cell")["subcluster"]
    hit = unit.index.intersection(sub.index)
    unit.loc[hit] = sub.loc[hit]
    missing = sorted(set(unit) - set(names.index))
    if missing:
        sys.exit(f"units without a curated name: {missing}")
    obs["celltype"] = unit.map(names["name"])

    lines = []
    def say(s=""):
        print(s); lines.append(str(s))
    units = obs.groupby(unit.values).size().rename("n_cells").to_frame()
    units["name"] = names["name"].reindex(units.index)
    units["evidence"] = names["evidence"].reindex(units.index)
    units.to_csv(work / "labels_curated_units.csv")
    say(units[["n_cells", "name"]].to_string())
    say(f"\nunits: {len(units)}   classes: {obs['celltype'].nunique()}")
    say("\n--- class sizes per slide ---")
    ct = pd.crosstab(obs["celltype"], obs["slide"])
    ct["total"] = ct.sum(axis=1)
    say(ct.sort_values("total", ascending=False).to_string())

    out = pd.DataFrame({"cell": obs.index, "slide": obs["slide"].astype(str),
                        "donor": obs["donor"].astype(str), "leiden": unit.values,
                        "celltype": obs["celltype"].astype(str), "prob": np.nan})
    out.to_csv(work / "labels_curated.csv", index=False)

    comp = pd.crosstab(obs["celltype"], obs["slide"], normalize="columns")
    say(f"\nJSD(slide A composition || slide B composition) = {jsd(comp.iloc[:, 0], comp.iloc[:, 1]):.4f}")
    rows = []
    for d, g in obs.groupby("donor", observed=True):
        if d == "NA" or g["slide"].nunique() < 2:
            continue
        c = pd.crosstab(g["celltype"], g["slide"], normalize="columns")
        rows.append(dict(donor=d, n_cells=len(g), jsd=jsd(c.iloc[:, 0], c.iloc[:, 1])))
    dd = pd.DataFrame(rows).sort_values("jsd", ascending=False)
    say(f"per-donor JSD: max {dd['jsd'].max():.4f} ({dd.iloc[0]['donor']}), mean {dd['jsd'].mean():.4f}")
    va, vb = (set(out.loc[out.slide == s, "celltype"]) for s in sorted(out.slide.unique()))
    say(f"label vocabulary identical on both slides: {va == vb} ({len(va)} / {len(vb)} classes)")
    (work / "report_curated.txt").write_text("\n".join(lines))
    say(f"\nwrote labels_curated.csv, labels_curated_units.csv, report_curated.txt")


if __name__ == "__main__":
    main()
