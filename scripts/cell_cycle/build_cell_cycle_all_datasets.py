import os
import sys
import time
from pathlib import Path
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.mixture import GaussianMixture

from scripts.test_cell_cycle import (
    S_GENES,
    G2M_GENES,
    compute_integrated_dapi,
    call_cell_cycle_dna,
    plot_cell_cycle_diagnostics,
)

DATASETS = [
    ("xenium_prime_ovarian_cancer_ffpe", "full"),
    ("xenium_prime_human_lung_cancer_ffpe", "full"),
    ("gse315411_pdltma06_10_prime_dual", "full"),
    ("gse315411_pdltma06_11_prime_solo", "full"),
    ("xenium_prime_human_ovary_ff", "full"),
]

def process_dataset(ds_name: str, bundle_name: str = "full"):
    print(f"\n=======================================================", flush=True)
    print(f"PROCESSING DATASET: {ds_name} (bundle: {bundle_name})", flush=True)
    print(f"=======================================================", flush=True)
    
    ds_dir = Path("data/datasets") / ds_name
    qc_dapi_path = ds_dir / "qc" / "nuclear_dapi.parquet"
    if not qc_dapi_path.exists():
        raise FileNotFoundError(f"Missing {qc_dapi_path}!")
        
    h5ad_path = ds_dir / "bundle" / f"{bundle_name}.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing {h5ad_path}!")

    print(f"Loading {qc_dapi_path}...", flush=True)
    t0 = time.time()
    df_dapi = pd.read_parquet(qc_dapi_path)
    print(f"Loaded DAPI table: {df_dapi.shape} in {time.time() - t0:.2f}s", flush=True)

    print(f"Loading AnnData {h5ad_path}...", flush=True)
    t0 = time.time()
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded AnnData: {adata.shape} in {time.time() - t0:.2f}s", flush=True)

    # Reindex DAPI to AnnData obs_names
    dapi_sub = df_dapi.reindex(adata.obs_names)
    int_dapi = compute_integrated_dapi(dapi_sub).to_numpy()
    adata.obs["integrated_dapi"] = int_dapi

    # 1. DAPI DNA Gating
    print("Fitting GMM and gating DNA phases...", flush=True)
    t0 = time.time()
    phases_dna, gmm, gmm_info = call_cell_cycle_dna(int_dapi, method="adaptive")
    adata.obs["cell_cycle_phase_dapi"] = phases_dna

    # Disambiguate G0 vs G1 via cell area
    cell_area_col = None
    for col in ["xenium_cell_area", "cell_area", "area_um2"]:
        if col in adata.obs.columns:
            cell_area_col = col
            break
    if cell_area_col:
        ca = adata.obs[cell_area_col].astype(float)
        g0_g1_sel = adata.obs["cell_cycle_phase_dapi"] == "G0/G1"
        med_ca = ca[g0_g1_sel].median()
        adata.obs.loc[g0_g1_sel & (ca < med_ca), "cell_cycle_phase_dapi"] = "G0"
        adata.obs.loc[g0_g1_sel & (ca >= med_ca), "cell_cycle_phase_dapi"] = "G1"
    print(f"DNA gating complete in {time.time() - t0:.2f}s", flush=True)

    # 2. Transcriptomic Marker Scoring
    print("Scoring transcriptomic markers...", flush=True)
    t0 = time.time()
    s_hits = [g for g in S_GENES if g in adata.var_names]
    g2m_hits = [g for g in G2M_GENES if g in adata.var_names]
    print(f"  Panel matches: {len(s_hits)} S genes, {len(g2m_hits)} G2/M genes", flush=True)

    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm)
    sc.pp.log1p(adata_norm)
    sc.tl.score_genes_cell_cycle(adata_norm, s_genes=s_hits, g2m_genes=g2m_hits)
    
    adata.obs["cell_cycle_phase_scanpy"] = adata_norm.obs["phase"].to_numpy()
    adata.obs["s_score"] = adata_norm.obs["S_score"].to_numpy()
    adata.obs["g2m_score"] = adata_norm.obs["G2M_score"].to_numpy()
    del adata_norm
    print(f"Transcript scoring complete in {time.time() - t0:.2f}s", flush=True)

    # 3. Joint Consensus State
    print("Computing joint consensus states...", flush=True)
    dp = adata.obs["cell_cycle_phase_dapi"].values
    gp = adata.obs["cell_cycle_phase_scanpy"].values

    conds = [
        (dp == "G0") & (gp == "G1"),
        (dp == "G1") & (gp == "G1"),
        (dp == "G2/M") & (gp == "G2M"),
        (dp == "S") & (gp == "S"),
        (dp == "G2/M") & (gp == "G1"),
        (np.isin(dp, ["G0", "G1"])) & (gp == "G2M"),
    ]
    choices = [
        "True Quiescent (G0)",
        "True Growth (G1)",
        "High-Confidence G2/M",
        "High-Confidence S",
        "Mitotic Dropout (DNA 4N, Tx G1)",
        "RNA Leakage Candidate (DNA 2N, Tx G2M)",
    ]
    adata.obs["joint_cell_cycle_state"] = np.select(conds, choices, default="Discordant / Transition")
    adata.obs["is_mitotic_dropout"] = (dp == "G2/M") & (gp == "G1")
    adata.obs["is_rna_leak_candidate"] = (np.isin(dp, ["G0", "G1"])) & (gp == "G2M")

    # 4. Save to Disk
    # Save qc/cell_cycle.parquet
    out_table = pd.DataFrame(index=adata.obs_names)
    out_table["integrated_dapi"] = adata.obs["integrated_dapi"].astype(float)
    out_table["dapi_mean"] = dapi_sub["dapi_mean"].astype(float)
    out_table["nucleus_area_px"] = dapi_sub["nucleus_area_px"].astype(float)
    out_table["nucleus_area_um2"] = dapi_sub["nucleus_area_um2"].astype(float)
    out_table["cell_cycle_phase_dapi"] = adata.obs["cell_cycle_phase_dapi"].astype(str)
    out_table["cell_cycle_phase_scanpy"] = adata.obs["cell_cycle_phase_scanpy"].astype(str)
    out_table["s_score"] = adata.obs["s_score"].astype(np.float32)
    out_table["g2m_score"] = adata.obs["g2m_score"].astype(np.float32)
    out_table["joint_cell_cycle_state"] = adata.obs["joint_cell_cycle_state"].astype(str)
    out_table["is_mitotic_dropout"] = adata.obs["is_mitotic_dropout"].astype(bool)
    out_table["is_rna_leak_candidate"] = adata.obs["is_rna_leak_candidate"].astype(bool)

    out_qc_parquet = ds_dir / "qc" / "cell_cycle.parquet"
    out_table.to_parquet(out_qc_parquet)
    print(f"Saved cell cycle table to {out_qc_parquet} ({len(out_table):,} rows)", flush=True)

    # Diagnostic Figure
    fig_path = ds_dir / "figures" / f"cell_cycle_{bundle_name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    # Temporary rename for plotting function compatibility
    adata.obs["cell_cycle_phase_image"] = adata.obs["cell_cycle_phase_dapi"]
    adata.obs["gene_phase"] = adata.obs["cell_cycle_phase_scanpy"]
    adata.obs["S_score"] = adata.obs["s_score"]
    adata.obs["G2M_score"] = adata.obs["g2m_score"]

    plot_cell_cycle_diagnostics(
        integrated_dapi=int_dapi,
        phases_dna=adata.obs["cell_cycle_phase_dapi"].to_numpy(),
        adata=adata,
        info=gmm_info,
        out_path=fig_path,
    )
    print(f"Saved diagnostic figure to {fig_path}", flush=True)

    # Print summary
    print("Summary of Joint States:")
    print(adata.obs["joint_cell_cycle_state"].value_counts(normalize=True).mul(100).round(2).to_string(), flush=True)

if __name__ == "__main__":
    for ds, bundle in DATASETS:
        process_dataset(ds, bundle)
    print("\nALL DATASETS COMPLETED SUCCESSFULLY!", flush=True)
