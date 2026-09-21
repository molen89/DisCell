#!/usr/bin/env python
"""
Stage 3 — cluster-majority naming + cross-slide consistency report.

Per-cell transferred labels are noisy at Xenium count depth, so the final label
is assigned per Leiden cluster: mean soft probability across the cluster's cells,
argmax, with confidence gates. Clusters that fail the gates become
Unknown_<cluster> rather than being forced into the nearest reference type.

Outputs:
  labels_<tag>.csv          cell -> final label  (this is what feeds the model)
  labels_<tag>_clusters.csv per-cluster assignment, confidence, slide balance
  report_<tag>.txt          consistency report
  query_labelled.h5ad       query_emb.h5ad + label columns for every tag
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu import add_common_args, pin_gpu, resolve_work, set_seed


def jsd(p, q, eps=1e-12):
    import numpy as np
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a * np.log2(a / b)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--tags", nargs="+", required=True,
                   help="tags from stage 2, e.g. hlca cellref")
    p.add_argument("--primary", default=None,
                   help="tag to use for the final label column (default: first tag)")
    p.add_argument("--min-mean-prob", type=float, default=0.5,
                   help="cluster's mean max-probability must exceed this")
    p.add_argument("--min-majority", type=float, default=0.5,
                   help="fraction of cells in the cluster carrying the winning label")
    p.add_argument("--min-slide-share", type=float, default=0.05,
                   help="flag clusters where one slide contributes less than this")
    args = p.parse_args()

    pin_gpu(-1)
    set_seed(args.seed)
    root, work = resolve_work(args)
    primary = args.primary or args.tags[0]

    import scanpy as sc
    import numpy as np
    import pandas as pd

    adata = sc.read_h5ad(work / "query_emb.h5ad")
    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    for tag in args.tags:
        say(f"\n{'='*70}\nreference: {tag}\n{'='*70}")
        tr = pd.read_csv(work / f"transfer_{tag}.csv").set_index("cell")
        npz = np.load(work / f"transfer_{tag}_probs.npz", allow_pickle=True)
        probs = pd.DataFrame(npz["probs"], index=npz["cells"], columns=npz["labels"])

        common = adata.obs_names.intersection(tr.index)
        tr, probs = tr.loc[common], probs.loc[common]
        sub = adata[common]
        leiden = sub.obs["leiden"].astype(str)
        slide = sub.obs["slide"].astype(str)

        # cluster-level assignment from mean soft probabilities
        mean_probs = probs.groupby(leiden.values).mean()
        winner = mean_probs.idxmax(axis=1)
        win_prob = mean_probs.max(axis=1)

        hard = tr[f"pred_{tag}"].astype(str)
        majority = hard.groupby(leiden.values).apply(
            lambda s: s.value_counts(normalize=True).iloc[0])
        majority_label = hard.groupby(leiden.values).apply(
            lambda s: s.value_counts().index[0])

        share = pd.crosstab(leiden.values, slide.values, normalize="index")
        size = leiden.value_counts()

        cl = pd.DataFrame({
            "n_cells": size.reindex(mean_probs.index),
            "label_softmean": winner,
            "mean_prob": win_prob,
            "label_majority": majority_label,
            "majority_frac": majority,
        })
        for s in share.columns:
            cl[f"share_{s}"] = share[s].reindex(cl.index)
        cl["slide_min_share"] = share.min(axis=1).reindex(cl.index)

        ok = (cl["mean_prob"] >= args.min_mean_prob) & (cl["majority_frac"] >= args.min_majority)
        cl["final"] = np.where(ok, cl["label_softmean"],
                               ["Unknown_" + str(i) for i in cl.index])
        cl["agrees_softmean_majority"] = cl["label_softmean"] == cl["label_majority"]
        cl["slide_imbalanced"] = cl["slide_min_share"] < args.min_slide_share

        say(f"clusters: {len(cl)}   named: {int(ok.sum())}   unknown: {int((~ok).sum())}")
        say(f"soft-mean and hard-majority disagree in "
            f"{int((~cl['agrees_softmean_majority']).sum())} clusters")
        n_imb = int(cl["slide_imbalanced"].sum())
        if n_imb:
            say(f"\nWARNING: {n_imb} clusters are dominated by one slide "
                f"(min share < {args.min_slide_share:.0%}). These are almost certainly "
                f"chemistry artefacts, not biology — consider excluding them:")
            say(cl.loc[cl["slide_imbalanced"],
                       ["n_cells", "final", "slide_min_share"]].to_string())
        else:
            say(f"\nall clusters draw from both slides (min share >= {args.min_slide_share:.0%})")

        cl.to_csv(work / f"labels_{tag}_clusters.csv")

        final = leiden.map(cl["final"]).astype(str)
        adata.obs[f"celltype_{tag}"] = pd.Categorical(
            final.reindex(adata.obs_names).fillna("Unknown").values)
        adata.obs[f"prob_{tag}"] = tr[f"prob_{tag}"].reindex(adata.obs_names).to_numpy()

        out = pd.DataFrame({
            "cell": adata.obs_names,
            "slide": adata.obs["slide"].astype(str).to_numpy(),
            "donor": adata.obs["donor"].astype(str).to_numpy() if "donor" in adata.obs else "NA",
            "leiden": adata.obs["leiden"].astype(str).to_numpy(),
            "celltype": adata.obs[f"celltype_{tag}"].astype(str).to_numpy(),
            "prob": adata.obs[f"prob_{tag}"].to_numpy(),
        })
        out.to_csv(work / f"labels_{tag}.csv", index=False)

        # ---- cross-slide consistency -------------------------------------------
        say("\n--- cross-slide composition ---")
        comp = pd.crosstab(adata.obs[f"celltype_{tag}"], adata.obs["slide"], normalize="columns")
        comp["abs_diff"] = (comp.iloc[:, 0] - comp.iloc[:, 1]).abs()
        say(comp.sort_values("abs_diff", ascending=False).head(20).to_string(
            float_format=lambda v: f"{v:.4f}"))
        say(f"\nJSD(slide A composition || slide B composition) = "
            f"{jsd(comp.iloc[:, 0], comp.iloc[:, 1]):.4f}   (0 = identical)")

        # ---- per-donor consistency ---------------------------------------------
        if "donor" in adata.obs and adata.obs["donor"].nunique() > 1:
            say("\n--- per-donor composition agreement between slides ---")
            rows = []
            for d, g in adata.obs.groupby("donor", observed=True):
                if g["slide"].nunique() < 2:
                    continue
                c = pd.crosstab(g[f"celltype_{tag}"], g["slide"], normalize="columns")
                if c.shape[1] < 2:
                    continue
                rows.append(dict(donor=d, n_cells=len(g),
                                 jsd=jsd(c.iloc[:, 0], c.iloc[:, 1]),
                                 pearson=float(np.corrcoef(c.iloc[:, 0], c.iloc[:, 1])[0, 1])))
            if rows:
                dd = pd.DataFrame(rows).sort_values("jsd", ascending=False)
                say(dd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
                say(f"\nmean per-donor JSD {dd['jsd'].mean():.4f}, "
                    f"mean Pearson {dd['pearson'].mean():.4f}")
                say("High JSD for a donor means its two sections disagree — check whether that "
                    "core is torn/folded on one slide before blaming the model.")

    # ---- agreement between references -----------------------------------------
    if len(args.tags) > 1:
        say(f"\n{'='*70}\nagreement between references\n{'='*70}")
        a, b = args.tags[0], args.tags[1]
        ct = pd.crosstab(adata.obs[f"celltype_{a}"], adata.obs[f"celltype_{b}"])
        say(f"contingency {ct.shape[0]} x {ct.shape[1]} written to agreement_{a}_vs_{b}.csv")
        ct.to_csv(work / f"agreement_{a}_vs_{b}.csv")
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            x = adata.obs[f"celltype_{a}"].astype(str)
            y = adata.obs[f"celltype_{b}"].astype(str)
            say(f"ARI  = {adjusted_rand_score(x, y):.4f}")
            say(f"NMI  = {normalized_mutual_info_score(x, y):.4f}")
        except ImportError:
            say("(install scikit-learn for ARI/NMI)")

    adata.obs["celltype"] = adata.obs[f"celltype_{primary}"]
    adata.write_h5ad(work / "query_labelled.h5ad", compression="gzip")
    (work / f"report_{primary}.txt").write_text("\n".join(lines))
    say(f"\nwrote query_labelled.h5ad and report_{primary}.txt")
    print(f"\nprimary label column: celltype  (from {primary})")


if __name__ == "__main__":
    main()
