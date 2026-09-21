import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.mixture import GaussianMixture

# Test script to evaluate gating strategies
dapi_path = "data/datasets/xenium_prime_ovarian_cancer_ffpe/qc/nuclear_dapi.parquet"
df_dapi = pd.read_parquet(dapi_path)

h5ad_path = "data/datasets/xenium_prime_ovarian_cancer_ffpe/bundle/test.h5ad"
adata = sc.read_h5ad(h5ad_path)
dapi_subset = df_dapi.reindex(adata.obs_names)
adata.obs["dapi_sum"] = dapi_subset["dapi_sum"]

dapi = adata.obs["dapi_sum"].to_numpy(dtype=float)
valid = np.isfinite(dapi) & (dapi > 0)

# Fit GMM on the dataset
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
gmm.fit(dapi[valid].reshape(-1, 1))

means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_.flatten())
order = np.argsort(means)
mu_2n, mu_4n = means[order]
sd_2n, sd_4n = stds[order]

print(f"mu_2n: {mu_2n:.1f}, sd_2n: {sd_2n:.1f}")
print(f"mu_4n: {mu_4n:.1f}, sd_4n: {sd_4n:.1f}")

# Strategy A: User's formula with 1.96 sigma
g0_196 = mu_2n + 1.96 * sd_2n
g2_196 = mu_4n - 1.96 * sd_4n
print(f"Formula (1.96 sigma): g0={g0_196:.1f}, g2={g2_196:.1f} (inversion: {g0_196 > g2_196})")

# Strategy B: Adaptive sigma k where k = min(1.96, 0.5 * (mu_4n - mu_2n)/(sd_2n + sd_4n))
k_max = (mu_4n - mu_2n) / (sd_2n + sd_4n)
k = min(1.96, 0.6 * k_max)
g0_adapt = mu_2n + k * sd_2n
g2_adapt = mu_4n - k * sd_4n
print(f"Adaptive sigma (k={k:.2f}): g0={g0_adapt:.1f}, g2={g2_adapt:.1f}")

phases_adapt = np.full(len(dapi), "Undetermined", dtype=object)
phases_adapt[valid & (dapi <= g0_adapt)] = "G0/G1"
phases_adapt[valid & (dapi > g0_adapt) & (dapi < g2_adapt)] = "S"
phases_adapt[valid & (dapi >= g2_adapt)] = "G2/M"

print("Adaptive sigma phase counts:")
print(pd.Series(phases_adapt).value_counts())

# Strategy C: GMM posterior probabilities (responsibilities)
probs = gmm.predict_proba(dapi[valid].reshape(-1, 1))
prob_2n = probs[:, order[0]]
prob_4n = probs[:, order[1]]

phases_prob = np.full(len(dapi), "Undetermined", dtype=object)
valid_idx = np.flatnonzero(valid)
phases_prob[valid_idx[prob_2n >= 0.65]] = "G0/G1"
phases_prob[valid_idx[prob_4n >= 0.65]] = "G2/M"
phases_prob[valid_idx[(prob_2n < 0.65) & (prob_4n < 0.65)]] = "S"

print("\nPosterior probability phase counts:")
print(pd.Series(phases_prob).value_counts())

# Cross-tabulate with gene_phase
from discell.model.cell_cycle import S_GENES, G2M_GENES
s_hits = [g for g in S_GENES if g in adata.var_names]
g2m_hits = [g for g in G2M_GENES if g in adata.var_names]
adata_norm = adata.copy()
sc.pp.normalize_total(adata_norm)
sc.pp.log1p(adata_norm)
sc.tl.score_genes_cell_cycle(adata_norm, s_genes=s_hits, g2m_genes=g2m_hits)
adata.obs["gene_phase"] = adata_norm.obs["phase"]
adata.obs["phase_adapt"] = phases_adapt
adata.obs["phase_prob"] = phases_prob

print("\nCross-tabulation (Gene phase vs DAPI Adaptive phase):")
print(pd.crosstab(adata.obs["gene_phase"], adata.obs["phase_adapt"], margins=True))

print("\nCross-tabulation (Gene phase vs DAPI Posterior phase):")
print(pd.crosstab(adata.obs["gene_phase"], adata.obs["phase_prob"], margins=True))
