#!/usr/bin/env python
"""
Stage 3b — final names from the pseudobulk correlations of both references.

Gate, per Leiden cluster (pre-registered in the devlog before this ran):
  1. median depth >= --min-depth transcripts, else Unknown_<k> (a cluster of
     20-transcript cells has no identity to transfer);
  2. the two references agree on the coarse lineage of their best type
     (HLCA ann_level_1 vs CellRef lineage_level1, Stroma == Mesenchymal),
     else Unknown_<k>;
  3. the name is HLCA's ann_finest_level of the best type; the CellRef call
     is reported beside it, not gated on (the two vocabularies differ).
  4. --overrides: marker-based names for clusters whose identity neither
     reference carries (cartilage, skeletal muscle, erythroid, ...), listed in
     OVERRIDES with their evidence; applied only when the flag is given.

Writes labels_pb.csv (same schema as labels_<tag>.csv, so stage 4 takes
--tag pb), labels_pb_clusters.csv and report_pb.txt (cross-slide and
per-donor composition JSD, vocabulary check).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed
from importlib import import_module
jsd = import_module("03_name_and_report").jsd

#: cluster -> (name, evidence). Identities absent from both references, read
#: off cluster_markers.csv; the references' nearest types are lineage-right
#: but wrong (e.g. erythroid -> "EC general capillary").
OVERRIDES = {
    "10": ("Skeletal muscle", "MYH2 MYH1 MYOT ACTN2 RYR1 (refs: smooth muscle)"),
    "14": ("Neutrophils", "CD177 DEFA1 MMP8 MPO FCGR3B (CellRef: Neutrophil r=0.72; HLCA core has none)"),
    "31": ("Schwann cells", "MPZ L1CAM SOX10 NGFR (refs: fibroblasts)"),
    "34": ("Erythroid", "SLC4A1 HBG1 HBZ SPTA1 GYPA (refs: EC capillary)"),
    "40": ("Megakaryocytes", "ITGA2B ITGB3 GP1BA MPL (refs: EC capillary)"),
    "41": ("Chondrocytes", "COL2A1 COL11A1 CNMD CHAD (refs: fibroblasts)"),
}
LINEAGE = {"Stroma": "Mesenchymal"}      # HLCA level-1 spelling -> CellRef spelling


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--min-depth", type=float, default=40.0)
    p.add_argument("--overrides", action="store_true")
    args = p.parse_args()
    pin_gpu(-1)
    set_seed(args.seed)
    root, work = resolve_work(args)

    import numpy as np
    import pandas as pd
    import anndata as ad
    import h5py

    hl = pd.read_csv(work / "pseudobulk_hlca.csv", dtype={"cluster": str}).set_index("cluster")
    cr = pd.read_csv(work / "pseudobulk_cellref.csv", dtype={"cluster": str}).set_index("cluster")
    mk = pd.read_csv(work / "cluster_markers.csv", dtype={"cluster": str}).set_index("cluster")
    cl = pd.DataFrame({
        "n_cells": hl["n_cells"], "median_depth": hl["median_depth"],
        "hlca": hl["best"], "r_hlca": hl["r_best"], "hlca_second": hl["second"],
        "hlca_lineage": hl["ann_level_1_best"].replace(LINEAGE),
        "cellref": cr["best"], "r_cellref": cr["r_best"], "cellref_second": cr["second"],
        "cellref_lineage": cr["lineage_level1_best"],
        "markers": mk["markers"].str.split().str[:5].str.join(" "),
    })
    cl["deep_enough"] = cl["median_depth"] >= args.min_depth
    cl["lineages_agree"] = cl["hlca_lineage"] == cl["cellref_lineage"]
    ok = cl["deep_enough"] & cl["lineages_agree"]
    cl["final"] = np.where(ok, cl["hlca"], "Unknown_" + cl.index.astype(str))
    cl["reason"] = np.select([~cl["deep_enough"], ~cl["lineages_agree"]],
                             ["depth < %g" % args.min_depth, "lineage disagreement"], "")
    if args.overrides:
        for k, (name, why) in OVERRIDES.items():
            cl.loc[k, ["final", "reason"]] = [name, "override: " + why]
    cl.to_csv(work / "labels_pb_clusters.csv")

    lines = []
    def say(s=""):
        print(s); lines.append(str(s))
    pd.set_option("display.width", 250)
    say(cl[["n_cells", "median_depth", "hlca", "r_hlca", "cellref", "r_cellref",
            "final", "reason", "markers"]].to_string(float_format=lambda v: f"{v:.2f}"))
    say(f"\nclusters: {len(cl)}   named: {int((~cl['final'].str.startswith('Unknown_')).sum())}   "
        f"unknown: {int(cl['final'].str.startswith('Unknown_').sum())}   "
        f"classes: {cl['final'].nunique()}")
    unk = cl[cl["final"].str.startswith("Unknown_")]
    say(f"cells in Unknown_*: {int(unk['n_cells'].sum()):,} ({unk['n_cells'].sum() / cl['n_cells'].sum():.1%})")

    # per-cell labels + the cross-slide reads, from the query obs
    rd = ad.io.read_elem if hasattr(ad, "io") else ad.experimental.read_elem
    with h5py.File(work / "query_emb.h5ad") as h:
        obs = rd(h["obs"])
    obs["celltype"] = obs["leiden"].astype(str).map(cl["final"])
    out = pd.DataFrame({"cell": obs.index, "slide": obs["slide"].astype(str),
                        "donor": obs["donor"].astype(str), "leiden": obs["leiden"].astype(str),
                        "celltype": obs["celltype"].astype(str), "prob": np.nan})
    out.to_csv(work / "labels_pb.csv", index=False)

    say("\n--- cross-slide composition ---")
    comp = pd.crosstab(obs["celltype"], obs["slide"], normalize="columns")
    comp["abs_diff"] = (comp.iloc[:, 0] - comp.iloc[:, 1]).abs()
    say(comp.sort_values("abs_diff", ascending=False).head(12).to_string(float_format=lambda v: f"{v:.4f}"))
    say(f"\nJSD(slide A composition || slide B composition) = {jsd(comp.iloc[:, 0], comp.iloc[:, 1]):.4f}")
    rows = []
    for d, g in obs.groupby("donor", observed=True):
        if d == "NA" or g["slide"].nunique() < 2:
            continue
        c = pd.crosstab(g["celltype"], g["slide"], normalize="columns")
        rows.append(dict(donor=d, n_cells=len(g), jsd=jsd(c.iloc[:, 0], c.iloc[:, 1])))
    dd = pd.DataFrame(rows).sort_values("jsd", ascending=False)
    say("\n--- per-donor JSD between slides ---")
    say(dd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    say(f"mean per-donor JSD {dd['jsd'].mean():.4f}")
    va, vb = (set(out.loc[out.slide == s, "celltype"]) for s in sorted(out.slide.unique()))
    say(f"\nlabel vocabulary identical on both slides: {va == vb} ({len(va)} / {len(vb)} classes)")
    (work / "report_pb.txt").write_text("\n".join(lines))
    say(f"\nwrote labels_pb.csv, labels_pb_clusters.csv, report_pb.txt")


if __name__ == "__main__":
    main()
