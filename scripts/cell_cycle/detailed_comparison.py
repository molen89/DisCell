import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.metrics import cohen_kappa_score, adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture

# Run comparison on test and full
from scripts.test_cell_cycle import run_cell_cycle_extraction

print("=== RUNNING ON TEST BUNDLE (1,500 cells) ===")
obs_test = run_cell_cycle_extraction(
    dataset_name="xenium_prime_ovarian_cancer_ffpe",
    bundle_name="test",
    gating_method="adaptive",
    save_plot=False,
)

# Also run on full bundle
print("\n=== RUNNING ON FULL BUNDLE (407,120 cells) ===")
obs_full = run_cell_cycle_extraction(
    dataset_name="xenium_prime_ovarian_cancer_ffpe",
    bundle_name="full",
    gating_method="adaptive",
    save_plot=False,
)

def analyze_concordance(obs: pd.DataFrame, name: str):
    print(f"\n=======================================================")
    print(f"DETAILED COMPARISON FOR: {name} (N={len(obs):,} cells)")
    print(f"=======================================================")
    
    # Map G0 and G1 into G0/G1 for direct 3-way comparison with Scanpy (G1, S, G2M)
    # Scanpy calls G1, S, G2M
    obs = obs.copy()
    obs["dna_3phase"] = obs["cell_cycle_phase_image"].replace({"G0": "G1", "G0/G1": "G1"})
    
    # Filter out Undetermined cells for clean comparison
    valid = obs["dna_3phase"].isin(["G1", "S", "G2/M"]) & obs["gene_phase"].isin(["G1", "S", "G2M"])
    df = obs[valid].copy()
    df["dna_3phase"] = df["dna_3phase"].replace({"G2/M": "G2M"})
    
    phases = ["G1", "S", "G2M"]
    y_true_gene = df["gene_phase"].values
    y_pred_dna = df["dna_3phase"].values
    
    # Overall metrics
    exact_match = (y_true_gene == y_pred_dna).mean()
    kappa = cohen_kappa_score(y_true_gene, y_pred_dna, labels=phases)
    ari = adjusted_rand_score(y_true_gene, y_pred_dna)
    
    print(f"\n1. Overall Agreement (3-phase: G1, S, G2M):")
    print(f"   - Exact 3-phase concordance: {exact_match*100:.2f}% ({exact_match:.4f})")
    print(f"   - Cohen's Kappa:             {kappa:.4f}")
    print(f"   - Adjusted Rand Index (ARI): {ari:.4f}")
    
    # Confusion Matrix: Rows = Gene Phase (Scanpy counts), Columns = DNA Image Phase (DAPI)
    cm = pd.crosstab(
        df["gene_phase"],
        df["dna_3phase"],
        rownames=["Scanpy (Counts)"],
        colnames=["DNA/DAPI (Image)"],
        margins=True
    )
    print(f"\n2. Absolute Contingency Matrix (Cell counts):")
    print(cm.to_string())
    
    # Row-normalized (% of Scanpy phase assigned to each DNA phase)
    cm_row = pd.crosstab(
        df["gene_phase"],
        df["dna_3phase"],
        rownames=["Scanpy (Counts)"],
        colnames=["DNA/DAPI (Image)"],
        normalize="index"
    ) * 100
    print(f"\n3. Scanpy Recall by DNA Phase (Row-normalized %):")
    print(cm_row.round(2).to_string())
    
    # Column-normalized (% of DNA phase assigned to each Scanpy phase)
    cm_col = pd.crosstab(
        df["gene_phase"],
        df["dna_3phase"],
        rownames=["Scanpy (Counts)"],
        colnames=["DNA/DAPI (Image)"],
        normalize="columns"
    ) * 100
    print(f"\n4. DNA Precision by Scanpy Phase (Column-normalized %):")
    print(cm_col.round(2).to_string())
    
    # Per-phase Jaccard index (Intersection over Union)
    print(f"\n5. Per-Phase Agreement (Jaccard Index / IoU):")
    for p in phases:
        inter = ((y_true_gene == p) & (y_pred_dna == p)).sum()
        union = ((y_true_gene == p) | (y_pred_dna == p)).sum()
        jaccard = inter / union if union > 0 else 0
        sens = inter / (y_true_gene == p).sum()
        spec = ((y_true_gene != p) & (y_pred_dna != p)).sum() / (y_true_gene != p).sum()
        print(f"   - {p:4s}: IoU = {jaccard*100:5.2f}% | Sensitivity = {sens*100:5.2f}% | Specificity = {spec*100:5.2f}%")
        
    # Discordance Breakdown
    print(f"\n6. Major Discordance Classes:")
    # Class A: Scanpy G1 but DNA G2M (High DNA mass, low transcript signal)
    disc_g1_g2m = ((df["gene_phase"] == "G1") & (df["dna_3phase"] == "G2M")).sum()
    pct_g1_g2m = disc_g1_g2m / len(df) * 100
    # Class B: Scanpy G2M but DNA G1 (Low DNA mass, high transcript signal / possible RNA leak)
    disc_g2m_g1 = ((df["gene_phase"] == "G2M") & (df["dna_3phase"] == "G1")).sum()
    pct_g2m_g1 = disc_g2m_g1 / len(df) * 100
    # Class C: S-phase mismatches
    disc_s = ((df["gene_phase"] == "S") ^ (df["dna_3phase"] == "S")).sum()
    pct_s = disc_s / len(df) * 100
    
    print(f"   - High DNA (4N/G2M) but Scanpy G1 (dropouts/sparse tx): {disc_g1_g2m:,} cells ({pct_g1_g2m:.2f}%)")
    print(f"   - Low DNA (2N/G1) but Scanpy G2M (potential leak/ambient): {disc_g2m_g1:,} cells ({pct_g2m_g1:.2f}%)")
    print(f"   - S-phase discordance (Scanpy vs DNA S disagreement):     {disc_s:,} cells ({pct_s:.2f}%)")

analyze_concordance(obs_test, "Ovarian Cancer TEST (1,500 cells)")
analyze_concordance(obs_full, "Ovarian Cancer FULL (407,120 cells)")
