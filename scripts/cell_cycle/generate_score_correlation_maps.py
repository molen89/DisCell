#!/usr/bin/env python3
"""Correlation maps on the continuous cell-cycle *scores*, not the phase labels.

Replaces the ordinal-label encoding of ``generate_correlation_maps.py``
(G0=0 < G1=1 < S=2 < G2/M=3), which rank-correlates a gate with counts and
mostly measures how the gate was cut. Here the variables are the measured
quantities: integrated DAPI (DNA-content proxy), the Scanpy S and G2M scores,
and per-gene counts; Spearman within cell type. Two panels per variable: raw,
and with log total counts partialled out (residual Spearman), because
integrated DAPI tracks nuclear area and area tracks library size -- the
partialled panel is the one that speaks to cell cycle.

Usage:
    OMP_NUM_THREADS=8 uv run python scripts/cell_cycle/generate_score_correlation_maps.py \
        --dataset xenium_prime_ovarian_cancer_ffpe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[2]
GENES = ["MKI67", "TOP2A", "PCNA", "CDK1"]
MIN_CELLS = 2000


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman of x and y after regressing the ranks of both on ranks of z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design = np.c_[np.ones_like(rz), rz]
    rx -= design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ry -= design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xenium_prime_ovarian_cancer_ffpe")
    ap.add_argument("--label-key", default=None, help="obs column for type (default: bundle default)")
    ap.add_argument("--max-cells-per-type", type=int, default=30000)
    args = ap.parse_args()

    ds = REPO / "data" / "datasets" / args.dataset
    adata = ad.read_h5ad(ds / "bundle" / "full.h5ad")
    cc = pd.read_parquet(ds / "qc" / "cell_cycle.parquet").reindex(adata.obs_names)
    label = args.label_key or adata.uns.get("default_label", "cell_group")
    types = adata.obs[label].astype(str)

    X = adata.X.tocsc() if sp.issparse(adata.X) else adata.X
    total = np.asarray(X.sum(axis=1)).ravel()
    gene_counts = {g: np.asarray(X[:, adata.var_names.get_loc(g)].todense()).ravel()
                   for g in GENES if g in adata.var_names}

    variables = {"integrated DAPI": cc["integrated_dapi"].to_numpy(float),
                 "S score": cc["s_score"].to_numpy(float),
                 "G2M score": cc["g2m_score"].to_numpy(float)}
    targets = {"total counts": total, **{f"{g} count": v for g, v in gene_counts.items()}}
    # the scores against each other and against DAPI are the interesting cells
    cross = {"DAPI ~ G2M score": ("integrated DAPI", "G2M score"),
             "DAPI ~ S score": ("integrated DAPI", "S score")}

    vc = types.value_counts()
    keep_types = [t for t in vc.index if vc[t] >= MIN_CELLS]
    rng = np.random.default_rng(0)
    rows = {"raw": {}, "depth-partialled": {}}
    for t in keep_types:
        idx = np.flatnonzero((types == t).to_numpy())
        if len(idx) > args.max_cells_per_type:
            idx = rng.choice(idx, args.max_cells_per_type, replace=False)
        ok = np.isfinite(variables["integrated DAPI"][idx])
        idx = idx[ok]
        lt = np.log1p(total[idx])
        for mode in rows:
            rec = {}
            for vn, v in variables.items():
                for tn, tv in targets.items():
                    if mode == "raw":
                        rec[f"{vn} ~ {tn}"] = spearmanr(v[idx], tv[idx]).statistic
                    elif tn != "total counts":
                        rec[f"{vn} ~ {tn}"] = partial_spearman(v[idx], tv[idx], lt)
            for cn, (a, b) in cross.items():
                rec[cn] = (spearmanr(variables[a][idx], variables[b][idx]).statistic if mode == "raw"
                           else partial_spearman(variables[a][idx], variables[b][idx], lt))
            rows[mode][f"{t} (n={len(idx):,})"] = rec

    fig, axes = plt.subplots(1, 2, figsize=(22, 0.45 * len(keep_types) + 3))
    for ax, (mode, rec) in zip(axes, rows.items()):
        df = pd.DataFrame(rec).T
        im = ax.imshow(df.values.astype(float), cmap="coolwarm", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index, fontsize=8)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                ax.text(j, i, f"{df.values[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Spearman within type — {mode}" + ("" if mode == "raw" else " (log total counts removed)"))
    fig.colorbar(im, ax=axes, shrink=0.6, label="Spearman")
    fig.suptitle(f"{args.dataset}: cell-cycle SCORES vs counts, per cell type", y=1.0)
    fig.tight_layout()
    out_png = ds / "figures" / "cell_cycle_score_correlation_per_celltype.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    out_csv = ds / "figures" / "cell_cycle_score_correlation_per_celltype.csv"
    pd.concat({m: pd.DataFrame(r).T for m, r in rows.items()}, axis=1).to_csv(out_csv)
    print("saved", out_png, "and", out_csv)


if __name__ == "__main__":
    main()
