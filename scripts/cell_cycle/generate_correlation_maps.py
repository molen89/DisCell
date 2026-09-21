import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import spearmanr

print("Starting correlation analysis...", flush=True)

# 1. Load dataset
dataset_dir = "data/datasets/xenium_prime_ovarian_cancer_ffpe"
h5ad_path = f"{dataset_dir}/bundle/test.h5ad"
print(f"Loading {h5ad_path}...", flush=True)
adata = sc.read_h5ad(h5ad_path)

# Load DAPI
dapi_path = f"{dataset_dir}/qc/nuclear_dapi.parquet"
df_dapi = pd.read_parquet(dapi_path).reindex(adata.obs_names)

from scripts.test_cell_cycle import compute_integrated_dapi, call_cell_cycle_dna, score_cell_cycle_markers
adata.obs["integrated_dapi"] = compute_integrated_dapi(df_dapi).to_numpy()

# Case 1: DAPI DNA State
print("Gating DAPI DNA states...", flush=True)
phases_dna, _, _ = call_cell_cycle_dna(adata.obs["integrated_dapi"].to_numpy(), method="adaptive")
adata.obs["cell_cycle_phase_image"] = phases_dna

# Disambiguate G0 vs G1 via cell area
cell_area = adata.obs["xenium_cell_area"].astype(float)
g0_g1_selector = adata.obs["cell_cycle_phase_image"] == "G0/G1"
med_area = cell_area[g0_g1_selector].median()
adata.obs.loc[g0_g1_selector & (cell_area < med_area), "cell_cycle_phase_image"] = "G0"
adata.obs.loc[g0_g1_selector & (cell_area >= med_area), "cell_cycle_phase_image"] = "G1"

# Case 2: Scanpy Count-based State
print("Scoring Scanpy cell cycle markers...", flush=True)
adata, _, _ = score_cell_cycle_markers(adata)

# Case 3: Joint Consensus State (Vectorized with np.select)
print("Assigning joint consensus states...", flush=True)
gp = adata.obs["gene_phase"].values
dp = adata.obs["cell_cycle_phase_image"].values

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
    "High-Conf G2/M",
    "High-Conf S",
    "Mitotic Dropout (DNA 4N, Tx G1)",
    "RNA Leakage Candidate (DNA 2N, Tx G2M)",
]
adata.obs["joint_state"] = np.select(conds, choices, default="Intermediate / Discordant")

# Total transcripts per cell
adata.obs["total_transcripts"] = adata.obs["xenium_total_counts"].astype(float)

# Cycling marker counts
markers = ["MKI67", "TOP2A", "PCNA", "CDK1"]
for m in markers:
    if m in adata.var_names:
        vals = adata[:, m].X
        adata.obs[f"count_{m}"] = np.asarray(vals.todense()).ravel() if hasattr(vals, "todense") else np.asarray(vals).ravel()

# Numeric progression encodings for correlation
dapi_order = {"G0": 0, "G1": 1, "S": 2, "G2/M": 3}
adata.obs["dapi_progression"] = adata.obs["cell_cycle_phase_image"].map(dapi_order)

scanpy_order = {"G1": 0, "S": 1, "G2M": 2}
adata.obs["scanpy_progression"] = adata.obs["gene_phase"].map(scanpy_order)

joint_order = {
    "True Quiescent (G0)": 0,
    "True Growth (G1)": 1,
    "High-Conf S": 2,
    "High-Conf G2/M": 3,
    "Mitotic Dropout (DNA 4N, Tx G1)": 2.5,
    "RNA Leakage Candidate (DNA 2N, Tx G2M)": 1.5,
    "Intermediate / Discordant": np.nan,
}
adata.obs["joint_progression"] = adata.obs["joint_state"].map(joint_order)

# Cell types
cell_types = adata.obs["cell_group"].value_counts()[lambda x: x >= 25].index.tolist()
print(f"Cell types evaluated (N={len(cell_types)}): {cell_types}", flush=True)

feature_cols = ["total_transcripts"] + [f"count_{m}" for m in markers if f"count_{m}" in adata.obs.columns]
feature_labels = ["Total Transcripts"] + [f"{m}" for m in markers if f"count_{m}" in adata.obs.columns]

# Compute correlation matrices for each case
results = {}
for case_name, prog_col in [
    ("Case 1: DAPI DNA State", "dapi_progression"),
    ("Case 2: Scanpy Count State", "scanpy_progression"),
    ("Case 3: Joint Consensus State", "joint_progression"),
]:
    corr_matrix = pd.DataFrame(index=cell_types, columns=feature_labels, dtype=float)
    p_matrix = pd.DataFrame(index=cell_types, columns=feature_labels, dtype=float)
    
    for ct in cell_types:
        sub = adata.obs[adata.obs["cell_group"] == ct]
        valid_rows = sub[prog_col].notna()
        x = sub.loc[valid_rows, prog_col].values
        
        for fcol, flab in zip(feature_cols, feature_labels):
            y = sub.loc[valid_rows, fcol].values
            if len(x) > 10 and np.std(x) > 0 and np.std(y) > 0:
                r, p = spearmanr(x, y)
                corr_matrix.loc[ct, flab] = r
                p_matrix.loc[ct, flab] = p
            else:
                corr_matrix.loc[ct, flab] = 0.0
                p_matrix.loc[ct, flab] = 1.0
                
    results[case_name] = (corr_matrix, p_matrix)

# Print tables
for case_name, (c_mat, _) in results.items():
    print(f"\n--- {case_name} ---", flush=True)
    print(c_mat.round(3).to_string(), flush=True)

# Plot 3-Panel Heatmap Correlation Map
print("Generating heatmap...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(20, max(5.5, len(cell_types) * 0.45 + 1.5)), sharey=True)

for ax, (case_name, (c_mat, _)) in zip(axes, results.items()):
    sns.heatmap(
        c_mat.astype(float),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.15,
        vmax=0.65,
        cbar=(ax == axes[-1]),
        ax=ax,
        linewidths=0.8,
        annot_kws={"size": 10},
    )
    ax.set_title(case_name, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticklabels(c_mat.columns, rotation=35, ha="right", fontsize=10)
    if ax == axes[0]:
        ax.set_ylabel("Cell Type (cell_group)", fontsize=11, fontweight="bold")
    else:
        ax.set_ylabel("")

fig.suptitle(
    "Correlation Maps: Cell Cycle State Label vs. Transcript Counts per Cell Type",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()

out_art = "/home/rmolen/.gemini/antigravity-ide/brain/4ede7ebc-30e6-4624-9f48-d7dbcb61b382/cell_cycle_correlation_map_per_celltype.png"
out_fig = f"{dataset_dir}/figures/cell_cycle_correlation_map_per_celltype.png"
os.makedirs(f"{dataset_dir}/figures", exist_ok=True)

fig.savefig(out_art, dpi=180, bbox_inches="tight")
fig.savefig(out_fig, dpi=180, bbox_inches="tight")
plt.close(fig)

print("Saved correlation maps successfully!", flush=True)
