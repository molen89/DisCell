#!/usr/bin/env python
"""
Stage 1 — joint scVI embedding of both slides + Leiden clustering.

batch_key = "slide" is what makes the two runs comparable: slide B is the
dual-chemistry run, so without correction you get clusters that split by slide
and the labels become per-tissue, which is exactly what we are avoiding.

Writes query_emb.h5ad (adds obsm["X_scVI"], obs["leiden"]) and scvi_model/.
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
    p.add_argument("--n-latent", type=int, default=30)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="default: scvi-tools heuristic (fewer epochs for more cells)")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--resolution", type=float, default=2.0,
                   help="Leiden resolution. Go fine — clusters get merged by name in stage 3.")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--subsample-train", type=int, default=None,
                   help="train scVI on this many cells, then encode all (for >2M cells)")
    args = p.parse_args()

    accel = pin_gpu(args.gpu)
    set_seed(args.seed)
    root, work = resolve_work(args)

    import scanpy as sc
    import scvi
    import numpy as np
    import torch

    print(f"accelerator={accel}  visible cuda devices={torch.cuda.device_count()}")
    if accel == "gpu":
        print(f"using physical GPU {args.gpu} -> {torch.cuda.get_device_name(0)}")
    scvi.settings.seed = args.seed

    adata = sc.read_h5ad(work / "query_raw.h5ad")
    print(adata)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    train_ad = adata
    if args.subsample_train and adata.n_obs > args.subsample_train:
        idx = np.random.default_rng(args.seed).choice(adata.n_obs, args.subsample_train, replace=False)
        train_ad = adata[idx].copy()
        print(f"training on a {train_ad.n_obs}-cell subsample")

    scvi.model.SCVI.setup_anndata(train_ad, layer="counts", batch_key="slide")
    model = scvi.model.SCVI(train_ad, n_latent=args.n_latent, n_layers=args.n_layers,
                            gene_likelihood="nb")
    model.train(max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                accelerator=accel, devices=1,
                early_stopping=True,
                early_stopping_patience=10)

    model.save(str(work / "scvi_model"), overwrite=True, save_anndata=False)
    print(f"saved {work/'scvi_model'}")

    adata.obsm["X_scVI"] = model.get_latent_representation(adata)
    print("latent:", adata.obsm["X_scVI"].shape)

    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=args.n_neighbors)
    sc.tl.leiden(adata, resolution=args.resolution, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False)
    n = adata.obs["leiden"].nunique()
    print(f"leiden: {n} clusters at resolution {args.resolution}")

    # immediate sanity check: does every cluster draw from both slides?
    import pandas as pd
    ct = pd.crosstab(adata.obs["leiden"], adata.obs["slide"], normalize="index")
    lopsided = ct[(ct.min(axis=1) < 0.05)]
    if len(lopsided):
        print(f"\nWARNING: {len(lopsided)} clusters are >95% one slide — batch correction "
              f"may be insufficient, or these are real slide-specific artefacts:")
        print(lopsided)
    else:
        print("\nall clusters draw from both slides (min share >= 5%)")

    out = work / "query_emb.h5ad"
    adata.write_h5ad(out, compression="gzip")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
