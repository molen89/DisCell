#!/usr/bin/env python
"""
Stage 2 — reference label transfer via scANVI + scArches surgery.

NOTE on the pretrained HLCA model: the published HLCA reference model
(zenodo 7599104) takes 2,000 HVGs as input. A Xenium 5K panel typically covers
only a few hundred of those, and prepare_query_anndata() pads the rest with
zeros — which silently wrecks the mapping. So this script trains scANVI from
scratch on the reference *restricted to the panel genes*, which is correct for
any panel. It costs one extra GPU-hour and is the right trade.

Reference is passed in, so run this once per reference. CELLxGENE exports index
genes by Ensembl id (symbols in var['feature_name']) and keep raw counts in
.raw.X — hence --ref-gene-col / --ref-use-raw:

  --ref hlca_core.h5ad       --ref-label-key ann_finest_level --ref-batch-key dataset \
      --ref-gene-col feature_name --ref-use-raw
  --ref lungmap_cellref.h5ad --ref-label-key <celltype col>   --ref-batch-key donor_id \
      --ref-gene-col feature_name --ref-use-raw

Writes transfer_<tag>.csv  (cell_id, predicted label, max prob) and the soft
probability matrix transfer_<tag>_probs.npz.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--ref", required=True, help="reference .h5ad (raw counts in .X or --ref-layer)")
    p.add_argument("--ref-layer", default=None, help="layer holding raw counts in the reference")
    p.add_argument("--ref-use-raw", action="store_true",
                   help="take raw counts from ref.raw.X (CELLxGENE exports)")
    p.add_argument("--ref-gene-col", default=None,
                   help="var column holding gene symbols when var_names are Ensembl ids "
                        "(CELLxGENE: feature_name)")
    p.add_argument("--ref-label-key", required=True, help="obs column with cell type names")
    p.add_argument("--ref-batch-key", default=None, help="obs column with reference batch")
    p.add_argument("--tag", required=True, help="short name for outputs, e.g. hlca or cellref")
    p.add_argument("--min-overlap", type=int, default=800,
                   help="abort if fewer than this many genes are shared with the panel")
    p.add_argument("--min-cells-per-label", type=int, default=50,
                   help="drop reference labels with fewer cells than this")
    p.add_argument("--ref-subsample", type=int, default=None,
                   help="subsample the reference to N cells (stratified by label) to save time")
    p.add_argument("--ref-epochs", type=int, default=None)
    p.add_argument("--scanvi-epochs", type=int, default=20)
    p.add_argument("--query-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args()

    accel = pin_gpu(args.gpu)
    set_seed(args.seed)
    root, work = resolve_work(args)

    import scanpy as sc
    import scvi
    import numpy as np
    import pandas as pd
    import torch
    from scipy import sparse

    print(f"accelerator={accel}")
    if accel == "gpu":
        print(f"using physical GPU {args.gpu} -> {torch.cuda.get_device_name(0)}")
    scvi.settings.seed = args.seed

    query = sc.read_h5ad(work / "query_emb.h5ad")
    ref = sc.read_h5ad(args.ref)
    print(f"query {query.shape}   reference {ref.shape}")

    if args.ref_use_raw:
        if ref.raw is None:
            sys.exit("ABORT: --ref-use-raw given but the reference has no .raw")
        ref = ref.raw.to_adata()
        print(f"reference raw counts: {ref.shape}")
    if args.ref_gene_col:
        ref.var_names = ref.var[args.ref_gene_col].astype(str).to_numpy()
    ref.var_names_make_unique()
    if args.ref_layer:
        ref.X = ref.layers[args.ref_layer].copy()
    # guard against a log-normalised reference being passed in by accident
    xmax = ref.X[:50].max() if not sparse.issparse(ref.X) else ref.X[:50].max()
    if xmax < 50:
        print(f"WARNING: reference max value in first 50 cells is {xmax:.2f} — this looks "
              f"log-normalised, not raw counts. scANVI needs counts. Check --ref-layer.")

    shared = sorted(set(query.var_names) & set(ref.var_names))
    print(f"shared genes: {len(shared)}  "
          f"({len(shared)/query.n_vars:.1%} of panel, {len(shared)/ref.n_vars:.1%} of reference)")
    if len(shared) < args.min_overlap:
        sys.exit(f"ABORT: only {len(shared)} shared genes (< --min-overlap {args.min_overlap}). "
                 f"Check gene ID conventions — reference may use Ensembl IDs while the panel "
                 f"uses symbols, or vice versa.")

    ref = ref[:, shared].copy()
    q = query[:, shared].copy()
    q.X = q.layers["counts"].copy()   # scANVI needs raw counts

    lab = ref.obs[args.ref_label_key].astype(str)
    counts = lab.value_counts()
    keep = counts[counts >= args.min_cells_per_label].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"dropping {len(dropped)} rare reference labels: {dropped[:10]}"
              f"{' ...' if len(dropped) > 10 else ''}")
    ref = ref[lab.isin(keep).values].copy()
    ref.obs["_label"] = ref.obs[args.ref_label_key].astype(str).astype("category")
    print(f"reference after filtering: {ref.shape}, {ref.obs['_label'].nunique()} labels")

    if args.ref_subsample and ref.n_obs > args.ref_subsample:
        rng = np.random.default_rng(args.seed)
        per = max(1, args.ref_subsample // ref.obs["_label"].nunique())
        idx = np.concatenate([
            rng.choice(np.where(ref.obs["_label"] == l)[0],
                       min(per, int((ref.obs["_label"] == l).sum())), replace=False)
            for l in ref.obs["_label"].cat.categories])
        ref = ref[idx].copy()
        print(f"reference subsampled to {ref.n_obs} cells")

    if args.ref_batch_key and args.ref_batch_key in ref.obs:
        ref.obs["_batch"] = ref.obs[args.ref_batch_key].astype(str)
    else:
        ref.obs["_batch"] = "ref"

    # --- 1. SCVI on the reference ------------------------------------------------
    scvi.model.SCVI.setup_anndata(ref, batch_key="_batch", labels_key="_label")
    ref_scvi = scvi.model.SCVI(ref, n_latent=30, n_layers=2, gene_likelihood="nb",
                               encode_covariates=True, dropout_rate=0.2)
    print("\ntraining reference SCVI ...")
    ref_scvi.train(max_epochs=args.ref_epochs, batch_size=args.batch_size,
                   accelerator=accel, devices=1, early_stopping=True,
                   early_stopping_patience=10)

    # --- 2. SCANVI on top --------------------------------------------------------
    print("\ntraining reference SCANVI ...")
    ref_scanvi = scvi.model.SCANVI.from_scvi_model(ref_scvi, unlabeled_category="Unknown",
                                                   labels_key="_label")
    ref_scanvi.train(max_epochs=args.scanvi_epochs, batch_size=args.batch_size,
                     accelerator=accel, devices=1)
    ref_scanvi.save(str(work / f"scanvi_ref_{args.tag}"), overwrite=True, save_anndata=False)

    # --- 3. surgery onto the query ----------------------------------------------
    q.obs["_batch"] = q.obs["slide"].astype(str)
    q.obs["_label"] = "Unknown"
    scvi.model.SCANVI.prepare_query_anndata(q, ref_scanvi)
    q_model = scvi.model.SCANVI.load_query_data(q, ref_scanvi)
    print("\ntraining query (scArches surgery) ...")
    q_model.train(max_epochs=args.query_epochs, batch_size=args.batch_size,
                  accelerator=accel, devices=1,
                  plan_kwargs=dict(weight_decay=0.0),
                  early_stopping=True, early_stopping_patience=10)
    q_model.save(str(work / f"scanvi_query_{args.tag}"), overwrite=True, save_anndata=False)

    pred = q_model.predict()
    soft = q_model.predict(soft=True)
    soft = pd.DataFrame(soft, index=q.obs_names) if not isinstance(soft, pd.DataFrame) else soft
    maxp = soft.max(axis=1).to_numpy()

    out = pd.DataFrame({
        "cell": q.obs_names,
        "slide": q.obs["slide"].astype(str).to_numpy(),
        "donor": q.obs["donor"].astype(str).to_numpy() if "donor" in q.obs else "NA",
        "leiden": q.obs["leiden"].astype(str).to_numpy(),
        f"pred_{args.tag}": np.asarray(pred),
        f"prob_{args.tag}": maxp,
    })
    out.to_csv(work / f"transfer_{args.tag}.csv", index=False)
    np.savez_compressed(work / f"transfer_{args.tag}_probs.npz",
                        probs=soft.to_numpy().astype("float32"),
                        labels=np.array(soft.columns, dtype=object),
                        cells=np.array(q.obs_names, dtype=object))

    emb = q_model.get_latent_representation()
    np.save(work / f"latent_{args.tag}.npy", emb.astype("float32"))

    print(f"\nwrote transfer_{args.tag}.csv, transfer_{args.tag}_probs.npz, latent_{args.tag}.npy")
    print("\npredicted label distribution (top 25):")
    print(out[f"pred_{args.tag}"].value_counts().head(25))
    print(f"\nmedian max-probability: {np.median(maxp):.3f}  "
          f"fraction < 0.5: {(maxp < 0.5).mean():.1%}")


if __name__ == "__main__":
    main()
