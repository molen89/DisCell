#!/usr/bin/env python3
"""Temporary test script: Extract cell cycle state using DNA content proxy (integrated DAPI)
and transcriptomic markers in 10x Xenium datasets.

Pipeline:
1. Metric: Integrated DAPI Intensity (DNA Content Proxy = Mean Nuclear DAPI * Nuclear Area or dapi_sum)
2. Classifier: 2N (G0/G1) vs 4N (G2/M) Gating via Gaussian Mixture Model (GMM)
   - Handles imaging variance and potential cutoff inversion with robust adaptive gating
3. Transcriptomic Marker Scoring: S and G2/M phase gene scoring via Scanpy (Tirosh et al. / Seurat markers)
4. Disambiguation: G0 vs G1 via cell size / area and/or marker expression
5. Concordance Validation: Cross-tabulation of DNA-based calls against transcriptomic marker states
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.mixture import GaussianMixture

# Canonical cycle gene sets (Tirosh et al. 2016 / Seurat updated 2019)
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM7", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "CENPU", "HELLS", "RFC2",
    "POLR1B", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7",
    "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1",
    "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B",
    "MRPL36", "E2F8",
]

G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "PIMREG",
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "JPT1", "CDC20", "TTK",
    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
    "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
    "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_cell_cycle")


# 1. METRIC: Integrated DAPI Intensity (DNA Content Proxy)
def compute_integrated_dapi(cells_df: pd.DataFrame) -> pd.Series:
    """Compute integrated nuclear DAPI intensity per cell.
    Supports standard Xenium cells.parquet, nuclear_dapi.parquet, and custom tables.
    """
    if "dapi_sum" in cells_df.columns:
        return cells_df["dapi_sum"].astype(float)
    elif "total_dapi" in cells_df.columns:
        return cells_df["total_dapi"].astype(float)
    elif "nucleus_area" in cells_df.columns and "dapi_mean" in cells_df.columns:
        return (cells_df["nucleus_area"] * cells_df["dapi_mean"]).astype(float)
    elif "nucleus_area_px" in cells_df.columns and "dapi_mean" in cells_df.columns:
        return (cells_df["nucleus_area_px"] * cells_df["dapi_mean"]).astype(float)
    else:
        raise KeyError(
            f"Required nuclear DAPI columns not found in table. Available columns: {list(cells_df.columns)}"
        )


# 2. CLASSIFIER: 2N (G0/G1) vs 4N (G2/M) Gating via GMM
def call_cell_cycle_dna(
    integrated_dapi: np.ndarray,
    method: str = "adaptive",
    random_state: int = 42,
) -> Tuple[np.ndarray, GaussianMixture, dict]:
    """Fit 2 Gaussian components to resolve 2N (G0/G1) and 4N (G2/M) peaks.
    Classify intermediate values as S phase.

    Parameters
    ----------
    integrated_dapi : np.ndarray
        Array of integrated DAPI intensities per cell.
    method : str
        'fixed_196': Strict mu +/- 1.96*sd gating intervals.
        'adaptive': If mu_2n + 1.96*sd > mu_4n - 1.96*sd due to imaging variance,
                    adaptively adjust gating boundary to prevent cutoff inversion.
        'posterior': Assign G0/G1 or G2/M when posterior probability exceeds threshold (e.g. 0.65),
                     intermediate as S phase.
    random_state : int
        Random state for GMM fitting.

    Returns
    -------
    phases : np.ndarray
        Categorical array of phases ('G0/G1', 'S', 'G2/M', 'Undetermined').
    gmm : GaussianMixture
        Fitted GMM object.
    info : dict
        Dictionary of gating parameters, means, standard deviations, and cutoffs.
    """
    integrated_dapi = np.asarray(integrated_dapi, dtype=float)
    n_total = len(integrated_dapi)
    phases = np.full(n_total, "Undetermined", dtype=object)

    # Filter non-positive, non-finite, and extreme imaging artifacts (> 99.5th percentile)
    finite_mask = np.isfinite(integrated_dapi) & (integrated_dapi > 0)
    if not np.any(finite_mask):
        log.warning("No positive finite DAPI values found for GMM fitting.")
        return phases, None, {}

    p995 = np.percentile(integrated_dapi[finite_mask], 99.5)
    valid_mask = finite_mask & (integrated_dapi < p995)
    valid_idx = np.flatnonzero(valid_mask)

    data = integrated_dapi[valid_idx].reshape(-1, 1)

    # Subsample if dataset is enormous for fast fitting (> 100,000 cells)
    if len(data) > 100_000:
        fit_idx = np.random.RandomState(random_state).choice(len(data), size=100_000, replace=False)
        fit_data = data[fit_idx]
    else:
        fit_data = data

    # GMM: Fit G0/G1 (2N) and G2/M (4N)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    gmm.fit(fit_data)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    # Sort so index 0 = G0/G1 (2N), index 1 = G2/M (4N)
    order = np.argsort(means)
    mu_2n, mu_4n = means[order]
    sd_2n, sd_4n = stds[order]
    w_2n, w_4n = weights[order]

    # Flow cytometry standard intervals:
    g0_cutoff_196 = mu_2n + 1.96 * sd_2n
    g2_cutoff_196 = mu_4n - 1.96 * sd_4n
    ratio = mu_4n / max(mu_2n, 1e-6)

    log.info(
        "GMM 2N (G0/G1): mean=%.1f, std=%.1f, weight=%.3f | 4N (G2/M): mean=%.1f, std=%.1f, weight=%.3f",
        mu_2n, sd_2n, w_2n, mu_4n, sd_4n, w_4n,
    )
    log.info("Ratio 4N / 2N mean: %.2f (expected ~2.0)", ratio)

    # Check for cutoff inversion
    inverted = g0_cutoff_196 >= g2_cutoff_196
    if inverted:
        log.warning(
            "Gating intervals with 1.96 sigma overlap (g0=%.1f >= g2=%.1f). "
            "Spatial imaging variance creates wide tails.",
            g0_cutoff_196, g2_cutoff_196,
        )

    if method == "fixed_196" or (not inverted and method == "adaptive"):
        g0_cutoff = g0_cutoff_196
        g2_cutoff = g2_cutoff_196
        phases[valid_mask & (integrated_dapi <= g0_cutoff)] = "G0/G1"
        phases[valid_mask & (integrated_dapi > g0_cutoff) & (integrated_dapi < g2_cutoff)] = "S"
        phases[valid_mask & (integrated_dapi >= g2_cutoff)] = "G2/M"

    elif method == "adaptive":
        # Solve for boundary where tails meet: mu_2n + k*sd_2n = mu_4n - k*sd_4n
        k_boundary = (mu_4n - mu_2n) / (sd_2n + sd_4n)
        # Choose k to leave an intermediate S window (e.g. 60% of boundary distance)
        k_used = min(1.96, max(0.2, 0.6 * k_boundary))
        g0_cutoff = mu_2n + k_used * sd_2n
        g2_cutoff = mu_4n - k_used * sd_4n
        log.info("Using adaptive sigma gating (k=%.2f): G0/G1 <= %.1f, G2/M >= %.1f", k_used, g0_cutoff, g2_cutoff)
        phases[valid_mask & (integrated_dapi <= g0_cutoff)] = "G0/G1"
        phases[valid_mask & (integrated_dapi > g0_cutoff) & (integrated_dapi < g2_cutoff)] = "S"
        phases[valid_mask & (integrated_dapi >= g2_cutoff)] = "G2/M"

    elif method == "posterior":
        probs = gmm.predict_proba(data)
        p2 = probs[:, order[0]]
        p4 = probs[:, order[1]]
        p_thresh = 0.65
        phases[valid_idx[p2 >= p_thresh]] = "G0/G1"
        phases[valid_idx[p4 >= p_thresh]] = "G2/M"
        phases[valid_idx[(p2 < p_thresh) & (p4 < p_thresh)]] = "S"
        g0_cutoff = float("nan")
        g2_cutoff = float("nan")

    else:
        raise ValueError(f"Unknown gating method: {method}")

    info = {
        "mu_2n": float(mu_2n),
        "sd_2n": float(sd_2n),
        "mu_4n": float(mu_4n),
        "sd_4n": float(sd_4n),
        "ratio_4n_2n": float(ratio),
        "g0_cutoff": float(g0_cutoff) if np.isfinite(g0_cutoff) else None,
        "g2_cutoff": float(g2_cutoff) if np.isfinite(g2_cutoff) else None,
        "inversion_detected": bool(inverted),
        "method_used": method,
    }
    return phases, gmm, info


# 3. TRANSCRIPTOMIC MARKER SCORING (Scanpy Cell Cycle Scoring)
def score_cell_cycle_markers(adata: ad.AnnData) -> Tuple[ad.AnnData, list[str], list[str]]:
    """Score cell cycle genes (S and G2/M) using Scanpy's score_genes_cell_cycle."""
    s_hits = [g for g in S_GENES if g in adata.var_names]
    g2m_hits = [g for g in G2M_GENES if g in adata.var_names]
    log.info("Matching cell cycle markers: %d S genes, %d G2/M genes", len(s_hits), len(g2m_hits))

    if len(s_hits) < 3 or len(g2m_hits) < 3:
        log.warning("Too few cell cycle markers in panel to score phases reliably.")
        adata.obs["gene_phase"] = "Undetermined"
        adata.obs["S_score"] = 0.0
        adata.obs["G2M_score"] = 0.0
        return adata, s_hits, g2m_hits

    # Work on normalized log1p expression for scoring
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm)
    sc.pp.log1p(adata_norm)
    sc.tl.score_genes_cell_cycle(adata_norm, s_genes=s_hits, g2m_genes=g2m_hits)

    adata.obs["gene_phase"] = adata_norm.obs["phase"].to_numpy()
    adata.obs["S_score"] = adata_norm.obs["S_score"].to_numpy()
    adata.obs["G2M_score"] = adata_norm.obs["G2M_score"].to_numpy()
    return adata, s_hits, g2m_hits


# 4. PLOTTING DIAGNOSTIC FIGURE
def plot_cell_cycle_diagnostics(
    integrated_dapi: np.ndarray,
    phases_dna: np.ndarray,
    adata: ad.AnnData,
    info: dict,
    out_path: Path,
):
    """Generate diagnostic figure showing DAPI distribution, GMM fits, and marker concordance."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: DAPI Histogram & Gating Cutoffs
    ax = axes[0]
    valid = np.isfinite(integrated_dapi) & (integrated_dapi > 0)
    p995 = np.percentile(integrated_dapi[valid], 99.5)
    dapi_plot = integrated_dapi[valid & (integrated_dapi < p995)]
    ax.hist(dapi_plot, bins=80, density=True, alpha=0.6, color="royalblue", label="DAPI Intensity")

    if info.get("mu_2n") and info.get("mu_4n"):
        mu_2n, sd_2n = info["mu_2n"], info["sd_2n"]
        mu_4n, sd_4n = info["mu_4n"], info["sd_4n"]
        x_grid = np.linspace(dapi_plot.min(), dapi_plot.max(), 300)
        from scipy.stats import norm
        g1_pdf = norm.pdf(x_grid, mu_2n, sd_2n) * 0.5
        g2_pdf = norm.pdf(x_grid, mu_4n, sd_4n) * 0.5
        ax.plot(x_grid, g1_pdf, "g--", label=f"2N (G0/G1) mu={mu_2n:.0f}")
        ax.plot(x_grid, g2_pdf, "r--", label=f"4N (G2/M) mu={mu_4n:.0f}")

    if info.get("g0_cutoff"):
        ax.axvline(info["g0_cutoff"], color="forestgreen", linestyle=":", lw=2, label="G0/G1 Cutoff")
    if info.get("g2_cutoff"):
        ax.axvline(info["g2_cutoff"], color="darkred", linestyle=":", lw=2, label="G2/M Cutoff")

    ax.set_title("DNA Content Proxy (Integrated DAPI) Gating")
    ax.set_xlabel("Integrated DAPI Intensity (a.u.)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: Transcriptomic S vs G2/M Scores colored by DAPI phase
    ax = axes[1]
    phase_colors = {
        "G0": "navy",
        "G1": "dodgerblue",
        "G0/G1": "blue",
        "S": "goldenrod",
        "G2/M": "crimson",
        "Undetermined": "lightgray",
    }
    for p, color in phase_colors.items():
        mask = adata.obs["cell_cycle_phase_image"] == p
        if np.any(mask):
            ax.scatter(
                adata.obs.loc[mask, "S_score"],
                adata.obs.loc[mask, "G2M_score"],
                c=color,
                s=12,
                alpha=0.6,
                label=f"{p} (n={mask.sum()})",
            )
    ax.set_title("Transcriptomic Scores Colored by DNA Phase")
    ax.set_xlabel("S-Phase Marker Score")
    ax.set_ylabel("G2/M-Phase Marker Score")
    ax.legend(fontsize=8, loc="upper left")

    # Panel 3: DAPI Distribution by Transcriptomic Phase
    ax = axes[2]
    gene_phases = [p for p in ["G1", "S", "G2M"] if p in adata.obs["gene_phase"].unique()]
    if gene_phases:
        dapi_groups = [
            adata.obs.loc[(adata.obs["gene_phase"] == gp) & np.isfinite(adata.obs["integrated_dapi"]), "integrated_dapi"]
            for gp in gene_phases
        ]
        try:
            ax.boxplot(dapi_groups, tick_labels=gene_phases, showfliers=False)
        except TypeError:
            ax.boxplot(dapi_groups, labels=gene_phases, showfliers=False)
        ax.set_title("DAPI Intensity by Gene-Expression Phase")
        ax.set_xlabel("Gene Marker Phase")
        ax.set_ylabel("Integrated DAPI Intensity")
    else:
        ax.text(0.5, 0.5, "Gene marker phases unavailable", ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved diagnostic figure to %s", out_path)


# 5. MAIN EXECUTION PIPELINE
def run_cell_cycle_extraction(
    dataset_name: str = "xenium_prime_ovarian_cancer_ffpe",
    bundle_name: str = "test",
    gating_method: str = "adaptive",
    save_plot: bool = True,
) -> pd.DataFrame:
    """Extract cell cycle states using DAPI proxy and transcriptomic markers."""
    repo_root = Path(__file__).resolve().parent.parent
    ds_dir = repo_root / "data" / "datasets" / dataset_name

    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

    # Step 1: Load AnnData
    h5ad_path = ds_dir / "bundle" / f"{bundle_name}.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {h5ad_path}")

    log.info("Loading AnnData: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    log.info("AnnData shape: %s", adata.shape)

    # Step 2: Load or locate DAPI morphology metrics
    # Priority order:
    # 1. qc/nuclear_dapi.parquet (pre-extracted from DAPI image)
    # 2. cells.parquet (standard 10x Xenium output if present)
    # 3. adata.obs itself if it already contains morphology
    dapi_parquet = ds_dir / "qc" / "nuclear_dapi.parquet"
    cells_parquet = ds_dir / "bundle" / "cells.parquet"

    if dapi_parquet.exists():
        log.info("Loading precomputed nuclear DAPI from %s", dapi_parquet)
        cells_df = pd.read_parquet(dapi_parquet).reindex(adata.obs_names)
    elif cells_parquet.exists():
        log.info("Loading cells metrics from %s", cells_parquet)
        cells_df = pd.read_parquet(cells_parquet).reindex(adata.obs_names)
    else:
        log.info("Using adata.obs for cell morphology metrics.")
        cells_df = adata.obs

    # Compute metric: Integrated DAPI
    log.info("Computing integrated DAPI intensity...")
    integrated_dapi = compute_integrated_dapi(cells_df).to_numpy()
    adata.obs["integrated_dapi"] = integrated_dapi

    # Step 3: DNA-based gating via GMM
    log.info("Fitting 2-component GMM for DNA content gating (method='%s')...", gating_method)
    phases_dna, gmm, gmm_info = call_cell_cycle_dna(
        integrated_dapi, method=gating_method
    )
    adata.obs["cell_cycle_phase_image"] = phases_dna

    # Step 4: Disambiguate G0 vs G1 via Cell Size / Area
    # G0 = 2N DNA + low cell area (dormancy/quiescence)
    # G1 = 2N DNA + cell growth / higher cell area
    cell_area_col = None
    for col in ["xenium_cell_area", "cell_area", "area_um2"]:
        if col in adata.obs.columns:
            cell_area_col = col
            break

    if cell_area_col is not None:
        cell_area = adata.obs[cell_area_col].astype(float)
        g0_g1_selector = adata.obs["cell_cycle_phase_image"] == "G0/G1"
        median_area_g0_g1 = cell_area[g0_g1_selector].median()

        log.info("Disambiguating G0 vs G1 using '%s' (median in G0/G1: %.2f)...", cell_area_col, median_area_g0_g1)
        g0_mask = g0_g1_selector & (cell_area < median_area_g0_g1)
        g1_mask = g0_g1_selector & (cell_area >= median_area_g0_g1)

        adata.obs.loc[g0_mask, "cell_cycle_phase_image"] = "G0"
        adata.obs.loc[g1_mask, "cell_cycle_phase_image"] = "G1"
    else:
        log.warning("No cell area column found in adata.obs for G0/G1 disambiguation.")

    # Step 5: Score transcriptomic cell cycle markers
    log.info("Scoring transcriptomic cell cycle markers (S and G2/M)...")
    adata, s_markers, g2m_markers = score_cell_cycle_markers(adata)

    # Summary Reports
    log.info("=== SUMMARY RESULTS ===")
    log.info("DNA Phase Distribution (Image/GMM):")
    phase_counts = adata.obs["cell_cycle_phase_image"].value_counts(dropna=False)
    for phase, cnt in phase_counts.items():
        log.info("  %s: %d cells (%.1f%%)", phase, cnt, 100.0 * cnt / len(adata))

    log.info("Transcriptomic Phase Distribution (Gene Markers):")
    gene_counts = adata.obs["gene_phase"].value_counts(dropna=False)
    for phase, cnt in gene_counts.items():
        log.info("  %s: %d cells (%.1f%%)", phase, cnt, 100.0 * cnt / len(adata))

    # Cross-tabulation concordance
    log.info("Cross-tabulation (Gene Markers vs DNA Image Call):")
    ct = pd.crosstab(
        adata.obs["gene_phase"],
        adata.obs["cell_cycle_phase_image"],
        margins=True,
    )
    print("\n" + ct.to_string() + "\n")

    # Marker positive rates across DNA phases
    log.info("Key Proliferation Marker Positivity per DNA Phase:")
    for marker in ["MKI67", "TOP2A", "PCNA", "CDK1"]:
        if marker in adata.var_names:
            pos_mask = np.asarray((adata[:, marker].X > 0).todense()).ravel() if hasattr(adata[:, marker].X, "todense") else np.asarray(adata[:, marker].X > 0).ravel()
            rates = []
            for p in ["G0", "G1", "S", "G2/M"]:
                sel = (adata.obs["cell_cycle_phase_image"] == p).to_numpy()
                if sel.sum() > 0:
                    r = pos_mask[sel].mean() * 100.0
                    rates.append(f"{p}: {r:.1f}%")
            log.info("  %s positivity: %s", marker, ", ".join(rates))

    # Comprehensive Comparison against Scanpy Count-based Calling
    from sklearn.metrics import cohen_kappa_score, adjusted_rand_score
    obs_comp = adata.obs.copy()
    obs_comp["dna_3phase"] = obs_comp["cell_cycle_phase_image"].replace({"G0": "G1", "G0/G1": "G1", "G2/M": "G2M"})
    valid_comp = obs_comp["dna_3phase"].isin(["G1", "S", "G2M"]) & obs_comp["gene_phase"].isin(["G1", "S", "G2M"])
    sub = obs_comp[valid_comp]
    y_gene = sub["gene_phase"].values
    y_dna = sub["dna_3phase"].values

    exact_acc = (y_gene == y_dna).mean()
    kappa = cohen_kappa_score(y_gene, y_dna, labels=["G1", "S", "G2M"])
    ari = adjusted_rand_score(y_gene, y_dna)

    log.info("=== SCANPY vs DNA GMM COMPARISON METRICS ===")
    log.info("  Overall 3-Phase Concordance (Exact Match): %.2f%%", exact_acc * 100)
    log.info("  Cohen's Kappa:                             %.4f", kappa)
    log.info("  Adjusted Rand Index (ARI):                 %.4f", ari)

    # Per-phase Jaccard (IoU)
    for p in ["G1", "S", "G2M"]:
        inter = ((y_gene == p) & (y_dna == p)).sum()
        union = ((y_gene == p) | (y_dna == p)).sum()
        iou = inter / union if union > 0 else 0
        sens = inter / (y_gene == p).sum()
        log.info("  Phase %-3s: IoU = %5.2f%% | Scanpy Recall = %5.2f%%", p, iou * 100, sens * 100)

    # Discordance summary
    disc_high_dna_low_tx = ((y_gene == "G1") & (y_dna == "G2M")).sum()
    disc_low_dna_high_tx = ((y_gene == "G2M") & (y_dna == "G1")).sum()
    log.info("  Discordance: 4N DNA but Scanpy G1 (sparse tx/dropout): %d (%.2f%%)",
             disc_high_dna_low_tx, disc_high_dna_low_tx / len(sub) * 100)
    log.info("  Discordance: 2N DNA but Scanpy G2M (potential leak/ambient): %d (%.2f%%)",
             disc_low_dna_high_tx, disc_low_dna_high_tx / len(sub) * 100)

    # Save diagnostic figure if requested
    if save_plot:
        fig_path = ds_dir / "figures" / f"cell_cycle_{bundle_name}.png"
        plot_cell_cycle_diagnostics(
            integrated_dapi=integrated_dapi,
            phases_dna=adata.obs["cell_cycle_phase_image"].to_numpy(),
            adata=adata,
            info=gmm_info,
            out_path=fig_path,
        )

    return adata.obs


def main():
    parser = argparse.ArgumentParser(description="Test extracting cell cycle states from Xenium datasets.")
    parser.add_argument(
        "--dataset",
        default="xenium_prime_ovarian_cancer_ffpe",
        help="Dataset name in data/datasets (default: xenium_prime_ovarian_cancer_ffpe)",
    )
    parser.add_argument(
        "--bundle",
        default="test",
        choices=["test", "full", "pdl018d"],
        help="Bundle name (default: test for fast testing)",
    )
    parser.add_argument(
        "--gating-method",
        default="adaptive",
        choices=["adaptive", "posterior", "fixed_196"],
        help="DNA GMM gating method (default: adaptive)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving diagnostic figure.",
    )
    args = parser.parse_args()

    run_cell_cycle_extraction(
        dataset_name=args.dataset,
        bundle_name=args.bundle,
        gating_method=args.gating_method,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
