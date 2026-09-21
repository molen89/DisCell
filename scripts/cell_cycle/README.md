# Cell Cycle Extraction & Comparison Suite

This subfolder contains the scripts for multimodal cell cycle state classification, comparing biophysical DNA mass (integrated nuclear DAPI) against transcript count scoring (Scanpy Tirosh / Seurat), resolving joint consensus states, and evaluating correlation maps across cell types.

---

## Script Inventory

### 1. `test_cell_cycle.py`
Main testing and benchmarking script.
- Computes integrated nuclear DAPI intensity (`dapi_sum` or `dapi_mean * nucleus_area`).
- Fits 2-component Gaussian Mixture Models (GMM) with adaptive boundary gating to prevent spatial variance overlap.
- Disambiguates $G_0$ quiescence from $G_1$ growth using cell area (`xenium_cell_area`).
- Runs Scanpy's `score_genes_cell_cycle` on normalized counts using 18 $S$-phase and 34 $G_2/M$-phase markers.
- Computes agreement metrics: Exact Concordance, Cohen's Kappa ($\kappa$), Adjusted Rand Index (ARI), and per-phase Jaccard IoU.
- Outputs flow-style diagnostic figures (`figures/cell_cycle_<bundle>.png`).

**Usage**:
```bash
# Fast test subset (1,500 cells):
.venv/bin/python scripts/cell_cycle/test_cell_cycle.py --dataset xenium_prime_ovarian_cancer_ffpe --bundle test

# Full slide (407,120 cells):
.venv/bin/python scripts/cell_cycle/test_cell_cycle.py --dataset xenium_prime_ovarian_cancer_ffpe --bundle full
```

---

### 2. `build_cell_cycle_all_datasets.py`
Automated batch runner that executes the pipeline across all 5 spatial Xenium datasets in `data/datasets/`:
1. `xenium_prime_ovarian_cancer_ffpe` (407,120 cells)
2. `xenium_prime_human_lung_cancer_ffpe` (278,324 cells)
3. `gse315411_pdltma06_10_prime_dual` (1,115,054 cells)
4. `gse315411_pdltma06_11_prime_solo` (1,100,349 cells)
5. `xenium_prime_human_ovary_ff` (1,157,637 cells)

For each dataset, it computes and saves:
- `qc/cell_cycle.parquet`: Parquet table of all phase calls, module scores, and flags (`is_mitotic_dropout`, `is_rna_leak_candidate`).
- `figures/cell_cycle_full.png`: Diagnostic flow figure.

**Usage**:
```bash
env PYTHONUNBUFFERED=1 .venv/bin/python scripts/cell_cycle/build_cell_cycle_all_datasets.py
```

---

### 3. `generate_per_celltype_plots.py`
Generates cell-type separated diagnostic figures:
- `figures/cell_cycle_dapi_per_celltype.png`: 2x4 grid showing DAPI intensity histograms, 2N/4N Gaussian fits, and gating cutoffs for each major cell type.
- `figures/cell_cycle_diagnostics_per_celltype.png`: Comprehensive grid showing DAPI histograms, S vs G2M scatter plots, and DAPI boxplots per phase separated by cell type.

**Usage**:
```bash
.venv/bin/python scripts/cell_cycle/generate_per_celltype_plots.py
```

---

### 4. `generate_correlation_maps.py`
Computes Spearman rank correlation matrices between cell cycle progression and transcript counts (Total Transcripts, *MKI67*, *TOP2A*, *PCNA*, *CDK1*) across each cell type for:
- Case 1: DAPI DNA State
- Case 2: Scanpy Count State
- Case 3: Joint Consensus State

Saves the 3-panel correlation heatmap to `figures/cell_cycle_correlation_map_per_celltype.png`.

**Usage**:
```bash
.venv/bin/python scripts/cell_cycle/generate_correlation_maps.py
```

---

### 5. Analysis & Comparison Tools
- `detailed_comparison.py`: Computes 3x3 confusion matrices, row-normalized recall, column-normalized precision, and major discordance classes.
- `compare_gating_strategies.py`: Compares fixed $1.96\sigma$, adaptive $\sigma$, and posterior probability gating.
- `analyze_discordance.py`: Inspects marker positivity rates across the 4 concordant/discordant quadrants.
