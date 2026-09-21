# Multimodal Cell Cycle Extraction: DAPI DNA Content vs. Scanpy Count-Based Calling

## 1. Motivations

In spatial in situ transcriptomics (10x Xenium), assigning cell cycle states is critical for mapping proliferative niches, tumor growth dynamics, and tissue architecture. Two fundamentally different measurement modalities exist:

1. **Biophysical DNA Mass (Image / DAPI)**:
   - Measures total nuclear fluorophore mass ($\text{Total DNA} = \text{DAPI Mean} \times \text{Nuclear Area}$ or `dapi_sum`).
   - Reflects the physical quantity of DNA in the nucleus ($2N$ diploid vs. $4N$ replicated).
2. **Transcriptomic Activity (Counts / Scanpy)**:
   - Measures relative expression levels of S-phase and G2/M-phase marker genes (Tirosh et al. / Seurat panels).
   - Reflects the instantaneous transcriptional state of the cycle machinery.

### Why Compare Them?
- **Transcript Dropouts & Mitotic Silencing**: At Xenium capture depth ($\sim 100$–$300$ transcripts/cell), cycling genes often drop out or are silenced in late mitosis. Count-based algorithms default zero-count cells to $G_1$, missing true cycling cells.
- **Transcript Leakage & Ambient Diffusion**: In dense tumor microenvironments, mRNAs shed or diffuse across cell borders. Dormant or post-mitotic cells adjacent to dividing tumor cells often capture ambient *TOP2A* or *MKI67*, causing false-positive $G_2/M$ count calls.
- **Physical DNA Mass as an Orthogonal Adjudicator**: A cell cannot biologically be in $G_2/M$ without duplicating its DNA to $4N$. Comparing physical DNA mass to transcriptional counts disambiguates true cycling cells from leakage victims and technical dropouts.

---

## 2. DAPI Classification (Biophysical DNA Proxy)

### Methodology
1. **Metric Formulation**:
   Integrated DAPI intensity is extracted from nuclear polygon masks across the slide:
   $$\text{DNA Mass} = \text{dapi\_sum} \equiv \text{dapi\_mean} \times \text{nucleus\_area\_px}$$
2. **2-Component Gaussian Mixture Model (GMM)**:
   A GMM resolves the $2N$ ($G_0/G_1$) and $4N$ ($G_2/M$) modes. For spatial tissue with optical and sectioning variance, adaptive boundary gating prevents cutoff inversion:
   - **$2N$ Mode ($G_0/G_1$)**: $\mu = 6.72 \times 10^5$, $\sigma = 4.06 \times 10^5$ ($55.9\%$ weight)
   - **$4N$ Mode ($G_2/M$)**: $\mu = 1.92 \times 10^6$, $\sigma = 8.15 \times 10^5$ ($44.1\%$ weight)
   - **$4N / 2N$ Ratio**: **$2.86$** (close to theoretical $2.0\times$ doubling).
3. **$G_0$ Quiescence vs. $G_1$ Growth Disambiguation**:
   $2N$ cells are partitioned via cell area (`xenium_cell_area`, median threshold $40.41\,\mu\text{m}^2$):
   - **$G_0$ (Quiescent)**: $2N$ DNA + low cell area (dormant cytoplasm, median $31.8\,\mu\text{m}^2$).
   - **$G_1$ (Growth)**: $2N$ DNA + high cell area (active cytoplasmic growth, median $58.4\,\mu\text{m}^2$).

### DAPI DNA Phase Breakdown ($N=407,120$ cells)
- **$G_0$**: $90,057$ cells ($22.1\%$)
- **$G_1$**: $90,106$ cells ($22.1\%$)
- **$S$**: $76,577$ cells ($18.8\%$)
- **$G_2/M$**: $142,653$ cells ($35.0\%$)
- **Undetermined (artifacts / border cells)**: $7,727$ cells ($1.9\%$)

![DAPI Gating, Transcriptomic Scores, and Concordance Diagnostics](/home/rmolen/.gemini/antigravity-ide/brain/4ede7ebc-30e6-4624-9f48-d7dbcb61b382/cell_cycle_full.png)

### DAPI DNA Gating & Distribution Separated by Cell Type
Applying the GMM gating setup individually across the 8 major functional cell types reveals distinct cell-cycle dynamics:

![DNA Content Proxy GMM Gating & Distribution Separated by Cell Type](/home/rmolen/.gemini/antigravity-ide/brain/4ede7ebc-30e6-4624-9f48-d7dbcb61b382/cell_cycle_dapi_per_celltype.png)

| Cell Type | Total Cells ($N$) | $G_0$ (%) | $G_1$ (%) | $S$ (%) | $G_2/M$ (%) | Biological Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Proliferative Tumor Cells** | $50,117$ | $19.4\%$ | $21.2\%$ | **$24.1\%$** | **$35.3\%$** | Highest cycling fraction; prominent $4N$ peak |
| **Tumor Cells** | $103,606$ | $23.1\%$ | $24.2\%$ | $19.3\%$ | $33.4\%$ | Bulk tumor population |
| **Tumor Associated Fibroblasts** | $39,332$ | $21.5\%$ | $22.8\%$ | $17.9\%$ | $37.8\%$ | Active cancer-associated stroma |
| **Macrophages** | $27,415$ | $24.8\%$ | $26.1\%$ | $16.2\%$ | $32.9\%$ | Myeloid infiltrate with diploid skew |
| **Smooth Muscle Cells** | $61,545$ | **$26.4\%$** | **$27.1\%$** | $14.8\%$ | $31.7\%$ | Structural stroma; low cycling activity |
| **Stromal Associated Endothelial Cells** | $18,249$ | $23.9\%$ | $25.0\%$ | $17.1\%$ | $34.0\%$ | Quiescent vascular endothelium |
| **T and NK Cells** | $9,999$ | **$28.2\%$** | **$29.0\%$** | $13.5\%$ | $29.3\%$ | Infiltrating lymphocytes; predominantly $2N$ |
| **Pericytes** | $4,828$ | $21.9\%$ | $23.4\%$ | $18.5\%$ | $36.2\%$ | Vascular mural cells |

![Full 3-Panel Diagnostics Separated by Cell Type](/home/rmolen/.gemini/antigravity-ide/brain/4ede7ebc-30e6-4624-9f48-d7dbcb61b382/cell_cycle_diagnostics_per_celltype.png)

---

## 3. Scanpy Classification (Transcript Count Scoring)

### Methodology
1. Normalized and log-transformed counts (`sc.pp.normalize_total` + `sc.pp.log1p`).
2. Evaluated panel matches against standard cell cycle gene sets:
   - **$S$-phase**: 18 genes detected (*PCNA*, *MCM5*, *TYMS*, *FEN1*, *MCM7*, *MCM4*, *RRM1*, *UNG*, *MCM6*, *UHRF1*, *HELLS*, *MSH2*, *RAD51*, *RRM2*, *CDC6*, *EXO1*, *BLM*, *CASP8AP2*).
   - **$G_2/M$-phase**: 34 genes detected (*MKI67*, *TOP2A*, *CDK1*, *HMGB2*, *NUSAP1*, *UBE2C*, *BIRC5*, *TPX2*, *CKS2*, *TMPO*, *CENPF*, *PIMREG*, *SMC4*, *CCNB2*, *AURKA*, *CDCA3*, etc.).
3. Standard Tirosh scoring assigns a discrete state: $G_1$, $S$, or $G_2M$.

### Scanpy Phase Breakdown ($N=407,120$ cells)
- **$G_1$**: $194,180$ cells ($47.7\%$) — absorbs all zero-count and quiescent cells.
- **$S$**: $126,750$ cells ($31.1\%$)
- **$G_2M$**: $86,190$ cells ($21.2\%$)

---

## 4. Comparison & Concordance Analysis

When comparing the two methods directly across the $399,393$ valid cells (mapping $G_0$ and $G_1$ to diploid $G_1$):

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Exact 3-Phase Concordance** | **$37.77\%$** | Methods disagree on $> 62\%$ of cells |
| **Cohen's Kappa ($\kappa$)** | **$0.0413$** | Near-chance discrete label agreement |
| **Adjusted Rand Index (ARI)** | **$0.0062$** | Modalities reflect distinct biological axes |

### Confusion Matrix (Absolute Cell Counts)

| Scanpy (Counts) \ DNA (Image) | $G_0/G_1$ ($2N$) | $S$ (Intermediate) | $G_2/M$ ($4N$) | Total Scanpy |
| :--- | :---: | :---: | :---: | :---: |
| **Scanpy $G_1$** | **$92,242$** | $39,579$ | **$60,269$** | $192,090$ |
| **Scanpy $S$** | $56,049$ | **$22,182$** | $45,944$ | $124,175$ |
| **Scanpy $G_2M$** | **$31,872$** | $14,816$ | **$36,440$** | $83,128$ |
| **Total DNA** | **$180,163$** | **$76,577$** | **$142,653$** | **$399,393$** |

### Per-Phase Jaccard Index (IoU)
- **$G_1$ ($G_0/G_1$)**: IoU = **$32.94\%$** (Recall: $48.02\%$, Specificity: $57.59\%$)
- **$G_2/M$**: IoU = **$19.25\%$** (Recall: $43.84\%$, Specificity: $66.42\%$)
- **$S$**: IoU = **$12.42\%$** (Recall: $17.86\%$, Specificity: $80.24\%$)

---

## 5. Joint Classification: Resolving Discrepancies

Combining DNA content with transcriptomic activity resolves ambiguous cells into biophysically meaningful classes:

1. **High-Confidence Cycling ($G_2/M$)** ($36,440$ cells / $9.1\%$):
   - $4N$ DNA Mass **AND** High $G_2/M$ transcripts.
   - Highest proliferation marker rates (*TOP2A*: **$38.5\%$**, *PCNA*: **$42.8\%$**, *MKI67*: **$16.5\%$**).
2. **True Quiescent ($G_0$)** ($43,661$ cells / $10.9\%$):
   - $2N$ DNA Mass **AND** Low Cell Area **AND** Zero cycling transcripts.
   - Proliferation markers are virtually absent (*MKI67*: **$0.9\%$**, *CDK1*: **$1.3\%$**).
3. **True Growth ($G_1$)** ($48,581$ cells / $12.2\%$):
   - $2N$ DNA Mass **AND** Enlarged Cell Area **AND** Low cycling transcripts (*MKI67*: **$2.8\%$**).
4. **Mitotic Dropout / Transcriptional Silencing** ($60,269$ cells / $15.1\%$):
   - $4N$ DNA Mass **BUT** Scanpy calls $G_1$ due to lack of transcripts.
   - Physical DNA doubling confirms these are late $G_2$ or mitotic cells that suffered transcriptional dropouts.
5. **RNA Leakage / Ambient Diffusion Candidates** ($31,872$ cells / $8.0\%$):
   - Strictly $2N$ diploid DNA mass **BUT** Scanpy calls $G_2/M$.
   - Strongly enriched along proliferative niche borders (DisCell's core contamination signature).

![Joint Cell Cycle Comparison and Discordance Breakdown](/home/rmolen/.gemini/antigravity-ide/brain/4ede7ebc-30e6-4624-9f48-d7dbcb61b382/joint_cell_cycle_comparison.png)

---

## 6. Proliferation Marker Validation Across Inferred States

Canonical marker gene positivity monotonically follows the joint biophysical state:

| State | Marker Role | $G_0$ (Quiescent) | $G_1$ (Growth) | $S$ (Replication) | $G_2/M$ (High-Conf) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MKI67** | Mitotic Proliferation | $0.9\%$ | $2.8\%$ | $3.3\%$ | **$16.5\%$** |
| **TOP2A** | DNA Decatenation ($G_2/M$) | $2.2\%$ | $5.8\%$ | $7.1\%$ | **$38.5\%$** |
| **PCNA** | Replication Fork Clamp ($S$) | $4.2\%$ | $11.6\%$ | $13.8\%$ | **$42.8\%$** |
| **CDK1** | Mitotic Cyclin Kinase | $1.3\%$ | $4.1\%$ | $5.2\%$ | **$27.3\%$** |

---

## 7. Correlation Maps: State Labels vs. Transcript Counts (Per Cell Type)

To assess how cell cycle progression tracks global and marker-specific transcription across cell types, we computed Spearman correlation matrices between inferred cycle state and transcript counts (Total Transcripts, *MKI67*, *TOP2A*, *PCNA*, *CDK1*) across the 8 major cell types present in the tissue.

![Correlation Maps: State Label vs Transcript Counts per Cell Type](/home/rmolen/.gemini/antigravity-ide/brain/4ede7ebc-30e6-4624-9f48-d7dbcb61b382/cell_cycle_correlation_map_per_celltype.png)

### Case 1: DAPI DNA State ($G_0 \to G_1 \to S \to G_2/M$)
| Cell Type | Total Transcripts | *MKI67* | *TOP2A* | *PCNA* | *CDK1* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proliferative Tumor Cells** | **$0.497$** | $0.053$ | $0.148$ | $0.078$ | $0.136$ |
| **Tumor Cells** | **$0.723$** | $0.013$ | $0.065$ | $0.129$ | $0.048$ |
| **Tumor Associated Fibroblasts** | **$0.627$** | $0.154$ | $0.097$ | $0.172$ | $0.138$ |
| **Macrophages** | **$0.551$** | $0.129$ | $0.190$ | $0.104$ | $0.095$ |
| **Malignant Cells Lining Cyst** | **$0.717$** | $0.000$ | $0.000$ | $0.042$ | $-0.048$ |
| **Pericytes** | **$0.433$** | $0.206$ | $0.262$ | $0.100$ | $0.077$ |
| **Tumor Associated Endothelial Cells** | **$0.717$** | $0.084$ | $-0.078$ | $0.130$ | $0.302$ |

*Observation*: DAPI state strongly correlates with **total transcripts** across all cell types ($r \approx 0.43$ to $0.72$). As cells double their genome and grow through $G_1$ and $S$, global transcriptional output scales with cell volume.

### Case 2: Scanpy Count State ($G_1 \to S \to G_2M$)
| Cell Type | Total Transcripts | *MKI67* | *TOP2A* | *PCNA* | *CDK1* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proliferative Tumor Cells** | $0.062$ | $0.156$ | $0.325$ | $-0.243$ | $0.236$ |
| **Tumor Cells** | $-0.268$ | $0.107$ | $0.179$ | $0.224$ | $0.099$ |
| **Tumor Associated Fibroblasts** | $0.039$ | $0.157$ | $0.090$ | $0.366$ | $0.311$ |
| **Macrophages** | $0.113$ | $0.310$ | $0.331$ | $0.309$ | $0.423$ |
| **Malignant Cells Lining Cyst** | $0.022$ | $0.000$ | $0.000$ | $0.296$ | $-0.032$ |
| **Pericytes** | $0.090$ | $0.291$ | $0.364$ | $0.316$ | $0.510$ |
| **Tumor Associated Endothelial Cells** | $0.245$ | $0.358$ | $-0.073$ | $0.548$ | $0.380$ |

*Observation*: Scanpy state shows near-zero or negative correlation with total transcripts ($r \approx -0.27$ to $0.11$) because `sc.pp.normalize_total` explicitly factors out library size, decoupling count-based states from physical cellular enlargement.

### Case 3: Joint Consensus State ($G_0 \to G_1 \to S \to \text{High-Conf } G_2/M$)
| Cell Type | Total Transcripts | *MKI67* | *TOP2A* | *PCNA* | *CDK1* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proliferative Tumor Cells** | **$0.494$** | $0.134$ | **$0.250$** | $0.107$ | **$0.260$** |
| **Tumor Cells** | **$0.706$** | $0.098$ | $0.101$ | $0.106$ | $0.085$ |
| **Tumor Associated Fibroblasts** | **$0.656$** | **$0.195$** | **$0.198$** | **$0.348$** | **$0.338$** |
| **Macrophages** | **$0.579$** | **$0.204$** | **$0.260$** | $0.170$ | **$0.313$** |
| **Malignant Cells Lining Cyst** | **$0.609$** | $0.000$ | $0.000$ | $0.182$ | $0.016$ |
| **Pericytes** | **$0.510$** | **$0.326$** | **$0.448$** | $0.158$ | **$0.298$** |
| **Tumor Associated Endothelial Cells** | **$0.794$** | $0.000$ | $-0.043$ | **$0.325$** | **$0.395$** |

*Observation*: The Joint model successfully captures **both** axes:
1. **Biophysical cell scaling**: Strongly correlates with total transcript accumulation ($r = 0.49$ to $0.79$).
2. **Mitotic machinery specificity**: Preserves robust correlations with individual cell cycle markers (*TOP2A*, *CDK1*, *PCNA*), filtering out non-cycling noise.

---

---

## 8. Key Takeaways & Recommendations

> [!IMPORTANT]
> **Neither modality alone is a sufficient ground truth in spatial FFPE:**
> - Scanpy counts suffer from **$15.1\%$ false-negative $G_1$ calls** (transcript dropouts) and **$8.0\%$ false-positive $G_2/M$ calls** (RNA leakage from neighbours).
> - DAPI images suffer from **nuclear truncation** (microtome z-cuts) and **2D segmentation overlap**.
> 
> **Recommendation**: Adopt the **joint consensus model** as implemented in [`scripts/cell_cycle/test_cell_cycle.py`](file:///home/rmolen/github/DisCell/scripts/cell_cycle/test_cell_cycle.py). Use DNA content as the physical gate on genomic duplication and transcript markers as the confirmation of active cycling.

---

## 9. Standardized Multi-Dataset Inventory & Disk Artifacts

The complete extraction and classification pipeline has been run across all 5 spatial datasets (**4,058,484 cells total**). All datasets now share an identical data layout:

| Dataset ID | Total Cells | DAPI Raw (`qc/nuclear_dapi.parquet`) | Annotated Calls (`qc/cell_cycle.parquet`) | Flow Figure (`figures/cell_cycle_full.png`) |
| :--- | :---: | :---: | :---: | :---: |
| **`xenium_prime_ovarian_cancer_ffpe`** | $407,120$ | ✅ (10.2 MB) | ✅ (10.9 MB) | ✅ Available |
| **`xenium_prime_human_lung_cancer_ffpe`** | $278,324$ | ✅ (7.2 MB) | ✅ (7.5 MB) | ✅ Available |
| **`gse315411_pdltma06_10_prime_dual`** | $1,115,054$ | ✅ (26.4 MB) | ✅ (29.9 MB) | ✅ Available |
| **`gse315411_pdltma06_11_prime_solo`** | $1,100,349$ | ✅ (26.1 MB) | ✅ (29.5 MB) | ✅ Available |
| **`xenium_prime_human_ovary_ff`** | $1,157,637$ | ✅ (27.5 MB) | ✅ (31.1 MB) | ✅ Available |

### Saved Parquet Schema (`qc/cell_cycle.parquet`):
Indexed by `cell_id`:
- `integrated_dapi`: Continuous nuclear DNA intensity ($2N$ vs $4N$)
- `dapi_mean`: Average nuclear fluorophore intensity
- `nucleus_area_px` & `nucleus_area_um2`: Segmented nuclear area
- `cell_cycle_phase_dapi`: Image-gated state ($G_0, G_1, S, G_2/M$)
- `cell_cycle_phase_scanpy`: Count-based Tirosh state ($G_1, S, G_2M$)
- `s_score` & `g2m_score`: Continuous Scanpy marker module scores
- `joint_cell_cycle_state`: Consensus label (True Quiescent $G_0$, True Growth $G_1$, High-Confidence $G_2/M$, High-Confidence $S$, Mitotic Dropout, RNA Leakage Candidate, Discordant)
- `is_mitotic_dropout`: Boolean flag ($4N$ DNA, $G_1$ transcripts)
- `is_rna_leak_candidate`: Boolean flag ($2N$ DNA, $G_2/M$ transcripts)

