import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

h5ad_path = "data/datasets/xenium_prime_ovarian_cancer_ffpe/bundle/test.h5ad"
adata = sc.read_h5ad(h5ad_path)
dapi_path = "data/datasets/xenium_prime_ovarian_cancer_ffpe/qc/nuclear_dapi.parquet"
df_dapi = pd.read_parquet(dapi_path).reindex(adata.obs_names)
adata.obs["dapi_sum"] = df_dapi["dapi_sum"]

from scripts.test_cell_cycle import compute_integrated_dapi, call_cell_cycle_dna, score_cell_cycle_markers
adata.obs["integrated_dapi"] = compute_integrated_dapi(df_dapi).to_numpy()
phases_dna, _, _ = call_cell_cycle_dna(adata.obs["integrated_dapi"].to_numpy(), method="adaptive")
adata.obs["dna_phase"] = phases_dna
adata.obs["dna_3phase"] = adata.obs["dna_phase"].replace({"G0/G1": "G1", "G0": "G1", "G2/M": "G2M"})

adata, _, _ = score_cell_cycle_markers(adata)

# Compute correlations between continuous scores
valid = np.isfinite(adata.obs["integrated_dapi"]) & (adata.obs["integrated_dapi"] > 0)
dapi_vals = adata.obs.loc[valid, "integrated_dapi"]
s_scores = adata.obs.loc[valid, "S_score"]
g2m_scores = adata.obs.loc[valid, "G2M_score"]

corr_s = np.corrcoef(dapi_vals, s_scores)[0, 1]
corr_g2m = np.corrcoef(dapi_vals, g2m_scores)[0, 1]
print(f"Continuous Pearson Correlation with Integrated DAPI:")
print(f"  Corr(DAPI, S_score):   {corr_s:.3f}")
print(f"  Corr(DAPI, G2M_score): {corr_g2m:.3f}")

# Marker expression across the 4 key concordant/discordant quadrants
# Quadrant 1: Concordant G1 (Gene G1 & DNA G1)
# Quadrant 2: Concordant G2M (Gene G2M & DNA G2M)
# Quadrant 3: Gene G1 but DNA G2M (High DNA mass, low transcripts)
# Quadrant 4: Gene G2M but DNA G1 (Low DNA mass, high transcripts - leak candidate)

df = adata.obs[valid].copy()
markers = ["MKI67", "TOP2A", "PCNA", "CDK1"]
for m in markers:
    df[m] = np.asarray((adata[valid, m].X > 0).todense()).ravel() if hasattr(adata.X, "todense") else np.asarray(adata[valid, m].X > 0).ravel()

groups = {
    "Concordant G1 (Gene=G1, DNA=G1)": (df["gene_phase"] == "G1") & (df["dna_3phase"] == "G1"),
    "Concordant G2M (Gene=G2M, DNA=G2M)": (df["gene_phase"] == "G2M") & (df["dna_3phase"] == "G2M"),
    "Gene=G1, DNA=G2M (High DNA, Low Tx)": (df["gene_phase"] == "G1") & (df["dna_3phase"] == "G2M"),
    "Gene=G2M, DNA=G1 (Low DNA, High Tx - Leak candidate)": (df["gene_phase"] == "G2M") & (df["dna_3phase"] == "G1"),
}

print("\nMarker Positivity Across Concordant and Discordant Subsets:")
for grp_name, mask in groups.items():
    print(f"\n--- {grp_name} (n={mask.sum()}) ---")
    dapi_median = df.loc[mask, "dapi_sum"].median()
    area_median = df.loc[mask, "xenium_cell_area"].median()
    print(f"  Median DAPI: {dapi_median:,.0f} | Median Cell Area: {area_median:.1f} um2")
    for m in markers:
        print(f"  {m} positive: {df.loc[mask, m].mean()*100:.1f}%")
