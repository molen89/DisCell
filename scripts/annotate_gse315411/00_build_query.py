#!/usr/bin/env python
"""
Stage 0 — build a single AnnData from the two GSE315411 Prime 5K bundles.

  slideA = GSM9427181, prime solo   (TMA section 11)  -> "prime_solo"
  slideB = GSM9427182, prime dual   (TMA section 10)  -> "prime_V1"

Intersects the two gene panels, drops control probes/codewords, attaches cell
metadata (centroids, areas, counts), assigns donor cores from series/cell_stats,
applies QC, and writes query_raw.h5ad.

Run --inspect first: it prints the schema of the cell_stats / coords files so the
donor-assignment step can be adapted if the column names differ from the guess.
"""
import argparse
import sys
import gzip
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed

SLIDES = {
    "prime_solo": dict(gsm="GSM9427181", sub="GSM9427181_prime_only"),
    "prime_V1":   dict(gsm="GSM9427182", sub="GSM9427182_prime_dual"),
}


def read_selection_export(path: Path):
    """A Xenium Explorer selection export: two '#' header lines, then a CSV.

    When the selection name carries a trailing space Explorer quotes the first
    line -- ``"#Selection name: ... "`` -- and pandas' ``comment='#'`` no longer
    recognises it (two of the 34 prime files: PDL026A and PDL085A on the solo
    slide), so the header lines are dropped by hand.
    """
    import io
    import pandas as pd
    with gzip.open(path, "rt") as fh:
        body = [line for line in fh if not line.lstrip('"').startswith("#")]
    return pd.read_csv(io.StringIO("".join(body)))


def find_bundle(root: Path, sub: str) -> Path:
    hits = sorted((root / sub).glob("output-*"))
    hits = [h for h in hits if h.is_dir()]
    if not hits:
        raise FileNotFoundError(f"no output-* directory under {root/sub}")
    if len(hits) > 1:
        print(f"  warning: {len(hits)} bundles under {sub}, using {hits[0].name}")
    return hits[0]


def inspect(root: Path):
    import pandas as pd
    cs = sorted((root / "series" / "cell_stats").glob("*cells_stats.csv.gz"))
    co = sorted((root / "series" / "cell_stats").glob("*coords.csv.gz"))
    print(f"\ncell_stats files: {len(cs)}   coords files: {len(co)}")
    for f in (cs[:1] + co[:1]):
        print(f"\n--- {f.name}")
        with gzip.open(f, "rt") as fh:
            for i, line in enumerate(fh):
                print("   ", line.rstrip()[:300])
                if i >= 3:
                    break
        df = read_selection_export(f)
        print("    columns:", list(df.columns), "rows:", len(df))
    for name, meta in SLIDES.items():
        b = find_bundle(root, meta["sub"])
        print(f"\n--- {name}: {b.name}")
        cells = pd.read_parquet(b / "cells.parquet")
        print("    cells.parquet columns:", list(cells.columns))
        print("    n cells:", len(cells))


def load_slide(bundle: Path, slide: str):
    import scanpy as sc
    import pandas as pd
    import numpy as np

    ad = sc.read_10x_h5(bundle / "cell_feature_matrix.h5", gex_only=False)
    ad.var_names_make_unique()
    # keep real genes only; Xenium h5 also carries Negative Control Probe /
    # Negative Control Codeword / Unassigned Codeword / Genomic Control
    if "feature_types" in ad.var:
        keep = ad.var["feature_types"].astype(str).str.contains("Gene Expression")
        ctrl = ad[:, ~keep.values].X.sum(axis=1)
        ad.obs["control_counts_h5"] = np.asarray(ctrl).ravel()
        ad = ad[:, keep.values].copy()

    cells = pd.read_parquet(bundle / "cells.parquet")
    idcol = "cell_id" if "cell_id" in cells.columns else cells.columns[0]
    cells[idcol] = cells[idcol].astype(str)
    cells = cells.set_index(idcol)
    ad.obs_names = [str(x) for x in ad.obs_names]
    common = ad.obs_names.intersection(cells.index)
    if len(common) < 0.99 * ad.n_obs:
        print(f"    warning: only {len(common)}/{ad.n_obs} cell ids matched cells.parquet")
    ad = ad[common].copy()
    ad.obs = ad.obs.join(cells.loc[common], how="left")

    for xk, yk in (("x_centroid", "y_centroid"), ("x", "y")):
        if xk in ad.obs and yk in ad.obs:
            ad.obsm["spatial"] = ad.obs[[xk, yk]].to_numpy().astype("float32")
            break
    ad.obs["slide"] = slide
    ad.obs_names = [f"{slide}:{c}" for c in ad.obs_names]
    return ad


def assign_donors(ad, root: Path, slide: str):
    """Map cells to donor cores using series/cell_stats/*_<slide>_cells_stats.csv.gz.

    The files are Xenium Explorer selection exports (see read_selection_export)
    with ``Cell ID,Cluster,Transcripts,Area (µm^2)``. The solo bundle's ids
    match the ``*_prime_solo_*`` files, the dual bundle's the ``*_prime_V1_*``
    files (verified 100 % on PDL061; the other segmentation variants 0 %).
    A file without a cell-id column aborts: a silently skipped core would
    surface only as an oversized NA class.
    """
    import pandas as pd
    import numpy as np

    d = root / "series" / "cell_stats"
    pat = f"PDLTMA006_*_{slide}_cells_stats.csv.gz"
    files = sorted(d.glob(pat))
    if not files:
        print(f"    no cell_stats matched {pat}; donor left as NA")
        ad.obs["donor"] = pd.Categorical(["NA"] * ad.n_obs)
        return ad

    mapping = {}
    for f in files:
        donor = f.name.split("_")[1]
        df = read_selection_export(f)
        idcol = next((c for c in df.columns if c.lower().replace(" ", "_") in
                      ("cell_id", "cellid", "cell", "barcode")), None)
        if idcol is None:
            sys.exit(f"ABORT: {f.name}: no cell-id column in {list(df.columns)[:8]}")
        for cid in df[idcol].astype(str):
            mapping[f"{slide}:{cid}"] = donor
    if not mapping:
        print("    donor mapping empty — check --inspect output and adapt assign_donors()")
        ad.obs["donor"] = pd.Categorical(["NA"] * ad.n_obs)
        return ad

    donors = pd.Series(ad.obs_names, index=ad.obs_names).map(mapping).fillna("NA")
    ad.obs["donor"] = pd.Categorical(donors.values)
    n_na = int((donors == "NA").sum())
    print(f"    donors assigned: {ad.obs['donor'].nunique() - 1} cores, {n_na} cells unassigned")
    print("    " + ", ".join(f"{d}={n}" for d, n in donors.value_counts().sort_index().items()))
    return ad


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--inspect", action="store_true",
                   help="print file schemas and exit (run this first)")
    p.add_argument("--min-counts", type=int, default=10,
                   help="drop cells with fewer total gene counts (default 10)")
    p.add_argument("--min-cells-per-gene", type=int, default=5)
    p.add_argument("--drop-unassigned-donor", action="store_true",
                   help="drop cells that fall outside any TMA core")
    args = p.parse_args()

    pin_gpu(-1)          # stage 0 is CPU-only
    set_seed(args.seed)
    root, work = resolve_work(args)

    if args.inspect:
        inspect(root)
        return

    import anndata as ad_mod
    import scanpy as sc
    import numpy as np

    slides = {}
    for slide, meta in SLIDES.items():
        b = find_bundle(root, meta["sub"])
        print(f"[{slide}] {b.name}")
        a = load_slide(b, slide)
        a = assign_donors(a, root, slide)
        print(f"    {a.n_obs} cells x {a.n_vars} genes")
        slides[slide] = a

    shared = sorted(set.intersection(*[set(a.var_names) for a in slides.values()]))
    print(f"\nshared genes: {len(shared)}")
    for s, a in slides.items():
        print(f"  {s}: {a.n_vars} -> {len(shared)}  (dropped {a.n_vars - len(shared)})")
    slides = {s: a[:, shared].copy() for s, a in slides.items()}

    adata = ad_mod.concat(list(slides.values()), label=None, merge="same",
                          index_unique=None)
    adata.obs["slide"] = adata.obs["slide"].astype("category")
    print(f"\ncombined: {adata.n_obs} cells x {adata.n_vars} genes")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_cells(adata, min_counts=args.min_counts)
    sc.pp.filter_genes(adata, min_cells=args.min_cells_per_gene)
    if args.drop_unassigned_donor and "donor" in adata.obs:
        adata = adata[adata.obs["donor"].astype(str) != "NA"].copy()
    print(f"after QC: {adata.n_obs} cells x {adata.n_vars} genes")
    print(adata.obs.groupby("slide", observed=True).size())

    out = work / "query_raw.h5ad"
    adata.write_h5ad(out, compression="gzip")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
