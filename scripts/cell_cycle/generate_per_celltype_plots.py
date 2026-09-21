import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import norm

print("Loading dataset for per-celltype diagnostics...", flush=True)
ds_dir = "data/datasets/xenium_prime_ovarian_cancer_ffpe"
h5ad_path = f"{ds_dir}/bundle/full.h5ad"
adata = sc.read_h5ad(h5ad_path)

# Load DAPI & annotations
qc_path = f"{ds_dir}/qc/cell_cycle.parquet"
df_qc = pd.read_parquet(qc_path).reindex(adata.obs_names)

for col in df_qc.columns:
    adata.obs[col] = df_qc[col]

print("Cell groups available:", flush=True)
print(adata.obs["cell_group"].value_counts())

# Select 8 representative cell types spanning tumor, stroma, immune, and endothelial compartments
representative_cts = [
    "Proliferative Tumor Cells",
    "Tumor Cells",
    "Tumor Associated Fibroblasts",
    "Macrophages",
    "Smooth Muscle Cells",
    "Stromal Associated Endothelial Cells",
    "T and NK Cells",
    "Pericytes",
]
top_cts = [ct for ct in representative_cts if ct in adata.obs["cell_group"].unique()]
print(f"Plotting for {len(top_cts)} major cell types: {top_cts}", flush=True)

# Generate Figure 1: DAPI DNA Histogram & Gating separated per cell type (2 rows x 4 cols)
fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True, sharey=False)
axes = axes.flatten()

# Overall slide cutoffs from full GMM:
valid_all = np.isfinite(adata.obs["integrated_dapi"]) & (adata.obs["integrated_dapi"] > 0)
p995_all = np.percentile(adata.obs.loc[valid_all, "integrated_dapi"], 99.5)

for i, ct in enumerate(top_cts):
    ax = axes[i]
    sub = adata.obs[(adata.obs["cell_group"] == ct) & valid_all]
    dapi_vals = sub.loc[sub["integrated_dapi"] < p995_all, "integrated_dapi"].values
    
    # Histogram
    ax.hist(dapi_vals, bins=60, density=True, alpha=0.55, color="royalblue", label="DAPI Density")
    
    # Fit cell-type specific 2-Gaussian or show distribution
    from sklearn.mixture import GaussianMixture
    if len(dapi_vals) > 500:
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        fit_data = np.random.choice(dapi_vals, size=min(10000, len(dapi_vals)), replace=False).reshape(-1, 1)
        gmm.fit(fit_data)
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        order = np.argsort(means)
        mu1, mu2 = means[order]
        sd1, sd2 = stds[order]
        
        x_grid = np.linspace(dapi_vals.min(), dapi_vals.max(), 300)
        ax.plot(x_grid, norm.pdf(x_grid, mu1, sd1) * gmm.weights_[order[0]], "g--", lw=1.5, label=f"2N μ={mu1/1e6:.2f}M")
        ax.plot(x_grid, norm.pdf(x_grid, mu2, sd2) * gmm.weights_[order[1]], "r--", lw=1.5, label=f"4N μ={mu2/1e6:.2f}M")
        
        # Adaptive gating cutoff lines for this cell type
        k = (mu2 - mu1) / (sd1 + sd2) * 0.6
        k = min(1.96, max(0.2, k))
        g0_cut = mu1 + k * sd1
        g2_cut = mu2 - k * sd2
        ax.axvline(g0_cut, color="forestgreen", linestyle=":", lw=1.8, label=f"G0/G1 Cut ({g0_cut/1e6:.2f}M)")
        ax.axvline(g2_cut, color="darkred", linestyle=":", lw=1.8, label=f"G2/M Cut ({g2_cut/1e6:.2f}M)")

    # Phase percentage breakdown
    phase_counts = sub["cell_cycle_phase_dapi"].value_counts(normalize=True) * 100
    g0_pct = phase_counts.get("G0", 0)
    g1_pct = phase_counts.get("G1", 0)
    s_pct = phase_counts.get("S", 0)
    g2_pct = phase_counts.get("G2/M", 0)
    
    title_str = f"{ct}\n(N={len(sub):,} | G0:{g0_pct:.0f}% G1:{g1_pct:.0f}% S:{s_pct:.0f}% G2/M:{g2_pct:.0f}%)"
    ax.set_title(title_str, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Integrated DAPI Intensity", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))

fig.suptitle(
    "DNA Content Proxy (Integrated DAPI) GMM Gating & Distribution Separated by Cell Type",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)
fig.tight_layout()

out_fig = f"{ds_dir}/figures/cell_cycle_dapi_per_celltype.png"
out_art = "/home/rmolen/.gemini/antigravity-ide/brain/4ede7ebc-30e6-4624-9f48-d7dbcb61b382/cell_cycle_dapi_per_celltype.png"
os.makedirs(f"{ds_dir}/figures", exist_ok=True)

fig.savefig(out_fig, dpi=160, bbox_inches="tight")
fig.savefig(out_art, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved DAPI per-celltype figure to {out_fig} and {out_art}", flush=True)

# Generate Figure 2: The Full 3-Panel Diagnostics Setup Separated by Cell Type (Comparative Grid)
# 4 columns: [DAPI Hist, S vs G2M Scores, DAPI by Gene Phase Boxplot, Joint State Distribution]
fig, axes = plt.subplots(len(top_cts), 3, figsize=(18, 3.2 * len(top_cts)))

for row_idx, ct in enumerate(top_cts):
    sub = adata.obs[(adata.obs["cell_group"] == ct) & valid_all]
    
    # Col 1: DAPI Hist
    ax1 = axes[row_idx, 0]
    dapi_vals = sub.loc[sub["integrated_dapi"] < p995_all, "integrated_dapi"].values
    ax1.hist(dapi_vals, bins=50, density=True, alpha=0.6, color="royalblue")
    ax1.set_title(f"{ct}: DAPI DNA Content (N={len(sub):,})", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Integrated DAPI", fontsize=8.5)
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
    
    # Col 2: S vs G2M scores colored by DNA Phase
    ax2 = axes[row_idx, 1]
    phase_colors = {"G0": "navy", "G1": "dodgerblue", "S": "goldenrod", "G2/M": "crimson"}
    sample_sub = sub.sample(n=min(2500, len(sub)), random_state=42)
    for p, color in phase_colors.items():
        m = sample_sub["cell_cycle_phase_dapi"] == p
        if np.any(m):
            ax2.scatter(
                sample_sub.loc[m, "s_score"], sample_sub.loc[m, "g2m_score"],
                c=color, s=8, alpha=0.6, label=p
            )
    ax2.set_title(f"{ct}: S vs G2M Scores (colored by DNA Phase)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("S-Score", fontsize=8.5)
    ax2.set_ylabel("G2M-Score", fontsize=8.5)
    ax2.axhline(0, color="gray", linestyle="--", lw=0.6)
    ax2.axvline(0, color="gray", linestyle="--", lw=0.6)
    if row_idx == 0:
        ax2.legend(fontsize=7, loc="upper left")

    # Col 3: DAPI Distribution by Gene Marker Phase (Boxplot)
    ax3 = axes[row_idx, 2]
    gene_phases = [p for p in ["G1", "S", "G2M"] if p in sub["cell_cycle_phase_scanpy"].unique()]
    if gene_phases:
        dapi_groups = [
            sub.loc[(sub["cell_cycle_phase_scanpy"] == gp) & (sub["integrated_dapi"] < p995_all), "integrated_dapi"]
            for gp in gene_phases
        ]
        try:
            ax3.boxplot(dapi_groups, tick_labels=gene_phases, showfliers=False)
        except TypeError:
            ax3.boxplot(dapi_groups, labels=gene_phases, showfliers=False)
        ax3.set_title(f"{ct}: DAPI by Scanpy Gene Phase", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Scanpy Gene Phase", fontsize=8.5)
        ax3.set_ylabel("Integrated DAPI", fontsize=8.5)
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(6, 6))

fig.tight_layout()
out_grid_fig = f"{ds_dir}/figures/cell_cycle_diagnostics_per_celltype.png"
out_grid_art = "/home/rmolen/.gemini/antigravity-ide/brain/4ede7ebc-30e6-4624-9f48-d7dbcb61b382/cell_cycle_diagnostics_per_celltype.png"
fig.savefig(out_grid_fig, dpi=150, bbox_inches="tight")
fig.savefig(out_grid_art, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved full diagnostics grid to {out_grid_fig} and {out_grid_art}", flush=True)
