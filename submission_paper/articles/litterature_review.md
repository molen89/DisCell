# Disentangling intrinsic, spatial and leakage effects in spatial omics
### A literature scan of models that separate cell-intrinsic variation, spatial/microenvironmental effects, and technical "leakage"

*Compiled 14 Feb 2026. Method: iterative web search + source verification (no offline library access; see §11 for access notes and §12 for leads that could not be verified).*

---

## Executive summary

The literature separates into **three layers that are often conflated**:

1. **Formal variance decomposition** — probabilistic models (Gaussian processes, mixed models, factor models) that write observed expression as a sum of components: cell-intrinsic/non-spatial, spatial/autocorrelated, environment/co-variates, and cell–cell interaction. *SVCA, SpatialDE/2, NSF/NSFH, MEfisto, STANCE.*
2. **Latent disentanglement** — deep generative models that split the latent space itself into an "intrinsic" and a "spatial/neighbourhood-induced" block, increasingly with **identifiability guarantees**. *SIMVI, Celcomen, Cellina, DRVI/FADVI (generic machinery).*
3. **Leakage & artifact correction** — models of the *measurement process*: transcripts physically or computationally assigned to the wrong cell/spot (spot swapping, spatial bleeding, segmentation error, misassignment), which induce *fake* spatial correlation and contaminate intrinsic profiles. *SpotClean, spatial-bleeding correction (Mitchel et al.), MisTIC, FastReseg, Baysor/v.PageSize, BYM2 DE models.*

A fourth cluster — **niche-driven expression models** (NCEM, MISTy, C-SIDE, Niche-DE, SpaCET/features) — attributes variation to spatial context via regression/attribution but typically does *not* guarantee that "intrinsic" and "spatial" parts are separately identifiable; they are the empirical half of disentanglement.

Fastest orientation: **SVCA** (the first explicit intrinsic/environment/interaction decomposition), **SIMVI** (the current reference for intrinsic-vs-spatial disentanglement with identifiability), **MISTy** (the simplest explainable attribution framework), **SpotClean +Mitchel et al.** (leakage correction), and the **ICLR 2025 Celcomen** paper (causal identifiability of intra- vs intercellular regulation).

---

## 1. The problem, precisely

Observed expression of cell *i* (or spot *i*) is modelled combinatorially across the literature:

```
Y_i  ≈  intrinsic(i)                      cell type / state / programs
      + spatial(i)                        autocorrelated field (position, domain)
      + niche(i)                          composition/identity of neighbours (interaction)
      + technical(i)                       batch, ambient RNA, spot swapping/bleeding, segmentation error
      + noise_i
```

Different communities slice this differently, which is a big source of confusion when comparing papers:

| Community | "Leakage" usually means | Main tool |
|---|---|---|
| Spatial statistics / GP | — | variance components |
| Deep learning (VSAs) | entangled latent factors | constraints on latent space |
| Measurement modelling | **spot swapping / spatial bleeding** (transcripts from cell A read at spot/cell B), misassigned transcripts, ambient RNA | probabilistic correction of counts |
| ML evaluation methodology | **spatial data leakage** in train/test splits (autocorrelated samples shared between folds) | buffered spatial CV |

Two *different* phenomena share the word "leakage": one corrupts the data itself, the other corrupts the evaluation. Works that target both without saying so are rare; a critical reader should state which one is meant.

---

## 2. Layer 1 — Formal variance decomposition (probabilistic, per-gene or per-factor)

| Method | Venue | Decomposition | Notes |
|---|---|---|---|
| **SVCA** (Arnol, Schapiro … Stegle) | Cell Systems 2019 | intrinsic + environment (GP, identical kernel) + cell–cell interaction (GP over *scaled* expression of neighbours) + noise | First explicit Gaussian-process decomposition for single-cell spatial data; kernels learned per gene; explicitly built to distinguish intrinsic from interaction variation |
| **SpatialDE** (Svensson, Teichmann, Stegle) | Nat Methods 2018 | spatial random effect + non-spatial noise | GP on coordinates, SPICE companion; canonical "spatially variable gene" test |
| **SpatialDE2** (Kats et al.) | bioRxiv 2021 — *not journal-published as of this scan* | spatial + non-spatial + tissue-zone hierarchy | combines SVG detection and zone mapping; Poisson likelihood |
| **NSF / NSFH** (Townes, Engelhardt) | Nat Methods 2023 | nonnegative spatial factors + nonspatial factors, Poisson/NB likelihood | GP prior on spatial factors, hybrid model partitions variability into spatial vs non-spatial |
| **MEfisto** (Velten, Huber, Stegle) | Genome Biology 2022 | spatial + non-spatial latent factors with informative priors | multi-modal spatial factor model; allows covariates |
| **STAN** (Chen, Zhou) | Genome Biology 2022 | gene expression = cell-type composition + cell-type-specific spatial variance component | "gene deconvolution + spatial random effect" |
| **STANCE** | Nat Commun 2025 | overall spatial component + per-cell-type spatial components | mixed model testing for cell-type-specific spatially variable genes |
| **Spanve** | preprint/benchmarking | spatial vs non-spatial via distribution-distance testing | non-parametric; see **openproblems.bio** SVG benchmark page |
| **SPARK / SPARK-X / MERINGUE / Trendsceek** | 2018–2020 | test *if* spatial signal exists (usually not *which* component) | null-model tests, not full decomposition |

**Read as a family:** all of them assume the *intrinsic* and *non-spatial* parts are identifiable because they have a distinct dependence structure on the coordinates (intrinsic variables look spatially white; spatial ones are smooth; niche effects correlate with neighbour composition rather than with absolute position). The specification of these dependence structures is exactly what distinguishes them.

---

## 3. Layer 2 — Deep generative / latent disentanglement

| Method | Venue | Mechanism | Notes |
|---|---|---|---|
| **SIMVI** | Nat Commun 2025 | VAE whose latent space is split into intrinsic and spatial-induced blocks; spatial branch sees a distance-aware graph of neighbours; **theoretical identifiability guarantees**; single-cell spatial-effect (SE) score per gene | Currently the most explicit "disentanglement with proof" reference; applies to Visium, Xenium/MERFISH/multiome, CosMx cohorts; downstream: niche finding, DE, batch integration |
| **Celcomen** (Megas et al.) | ICLR 2025 (also Nat Commun 2026 per publisher records) | energy-based / generative GNN with **mathematical-causality identifiability**; separates intra-cellular (H') vs inter-cellular (H) gene–gene interaction matrices | aimed at disentangling *regulation*, not just variance; supports counterfactual "virtual tissue" perturbations |
| **Cellina** | arXiv 2026 | graph-VAE, dual encoder; supervised disentanglement with cell-type / domain labels as inductive bias; intrinsic vs microenvironmental latents | tested by *counterfactual tissue-graph perturbation*: if intrinsic and extrinsic are conflated, counterfactuals fail — this is a good model-selection test |
| **DRVI** (Moinfar, Theis et al.) | ICML/ workshops 2024–25, bioRxiv | additive-decoder VAE with nonlinear pooling → unsupervised disentangled latent spaces for single-cell data | generic disentanglement machinery; spatial branch not built in, but architectural basis for intrinsic-vs-covariate splits |
| **FADVI** | bioRxiv 2025 | VAE partitioning latent space into batch-specific / label-related / residual subspaces | the "batch" analogue of the intrinsic/spatial split — shows the same disentanglement recipe works for technical covariate removal |
| **SpaVCNN / spatially-informed VAEs** (SpaGCN, STAGATE, SEDR, BayesSpace, Bayes-VAE family) | 2021–2023 | add spatial graph/regularisation to autoencoders | tend to **encourage** spatial smoothness (i.e., mix intrinsic and spatial) rather than separate them; useful as negatives / caveats in a disentanglement review |

**Key conceptual contrast:** layer-1 methods *decompose the variance*, layer-2 methods *split the representation* — the former is a statement about the generative process, the latter about the latent code; identifiability proofs bridge these (SIMVI, Celcomen, Cellina).

---

## 4. Layer 3 — Leakage as a measurement artifact (what most biologists mean by "leakage")

| Work | Venue | What it models / corrects |
|---|---|---|
| **SpotClean** (Ni, Prasad, Chen et al., Rafa lab) | Nat Commun 2022; Bioconductor | Probabilistic model of **spot swapping** on 10x Visium: mRNA detected at a spot that was expressed in a *neighbouring* spot; estimates per-spot contamination rates using background spots and decontaminates the count matrix |
| **"Correcting the Record: Spatial Bleeding Correction with Bayesian Inference"** (Mitchel et al., 2024.08.06.597016) | bioRxiv 2024; peer-reviewed version reported in *Bioinformatics* (2026) as "Correcting spatial transcriptomics data affected by a prevalent transcript leakage problem across platforms, species, and tissues" | formalises **spatial bleeding** as a *platform-wide* imaging-ST artifact; diffusion-like contamination model (closer to cell centroid = more likely endogenous); correction applies across Xenium, MERSCOPE, CosMx and similar imaging platforms, not just spot arrays; shows effects on DE and on intrinsic-profile inference |
| **MisTIC** | bioRxiv 2025.12.11.693759 (preprint) | Probabilistic re-assignment of transcripts mis-assigned **by segmentation error** on imaging SRT; motivated by the observation that contamination in imaging SRT is frequent but usually a modest fraction of a cell's transcriptome, and is strongest near heterotypic neighbours |
| **FastReseg** | Sci Rep 2025 | Refines existing image-based cell segmentations using transcript/local-type evidence; explicitly notes that spatial doublets are **directional** contamination (unlike scRNA-seq doublets) |
| **Baysor** (Petukhov et al.) | Nat Biotechnol 2022 | Cell segmentation that *clusters transcripts* rather than relying strictly on image masks; reduces mask-boundary misassignment at its root |
| **`pageSize` / v.PageSize (Zhu et al.)** | bioRxiv 2025.07.11 (preprint) | Transcript-count-based segmentation refinement; the same "leakage" signal has been reproduced by independent groups, strengthening the case that it is real and pervasive |
| **SoupX (Young & Behjati, 2020); DecontX (Yang et al., NAR 2020)** | 2020 | scoped to scRNA-seq but routinely used on imaging- and array-derived ST after cell segmentation |

**Why this layer matters for disentanglement:** leakage **creates** fake spatial autocorrelation (a strongly expressing cell types its neighbour's profile) and **destroys** intrinsic profile purity. Several papers report that after bleeding/spot-swap correction, the fraction of genes called "spatially variable" changes substantially — i.e., a large part of what naive methods call "spatial signal" is a measurement artifact interacting with cell density.

---

## 5. Layer 4 — Niche-driven / spatially-attributed expression (empirical attribution rather than decomposition)

| Method | Venue | What it attributes |
|---|---|---|
| **NCEM** (Fischer, Schaar, Theis) | Nat Biotechnol 2022 | Graph neural network predicting a cell's expression from the *composition* of its spatial neighbourhood (niche); "node-centric" attribution of expression variance to niche composition; variants for different spatial graph types. Jointly a communication-inference and variance-attribution framework |
| **MISTy** (Tanevski, … Saez-Rodriguez) | Genome Biology 2022 | Explainable multi-view framework: `intraview` (own markers) + `juxtaview` (immediate neighbourhood) + `paraview` (broader tissue) + user-defined views; provides **variance-explained importance** per view per marker; backend-agnostic (any learner); "leakage"-adjacent views (e.g. *secreted/matrix*, *pathway-activity* views) are arbitrary, which is both flexibility and reproducibility risk |
| **C-SIDE** (Cable, … Raphael) | Nat Methods 2022 | Log-linear cell-type-specific DE model whose **covariates can be spatial** (microenvironment, cell-type localization); attributes spatially-varying expression *within* each cell type while properly modelling the pixel/spot as a mixture of cell types — this removes the composition confound that most naive spatial-DE methods share |
| **Niche-DE / nicheDE** (Mason, Zhang et al.) | bioRxiv 2023.01.03.522646 | identifies **(index cell type, niche cell type)** pairs whose gene expression is up/down-regulated under a specific niche; fits expression to an "effective niche" vector via regression. Explicitly notes robustness to spot swapping |
| **SpaCET, STdGCN, CellsFromSpace** | 2022–2024 | go *from spot-level mixtures to cell-type-aware niche/landscape interpretation* — related in spirit, different in unit of analysis (tumour–stroma interface, spatial niches, ICA-based) |
| **SpaOTsc** (Cang & Nie) | Nat Commun 2020 | Not a decomposition method per se — uses **partial information decomposition + optimal transport** to quantify inter-cellular information flow; included as an information-theoretic sibling of the interaction term |

These are "attribution" methods: they return `importance(niche) / importance(intrinsic)` but generally do **not** claim the two latent blocks are separately identifiable — a useful honesty marker when writing a review or picking a method.

---

## 6. Evaluation pitfalls — the *other* "leakage"

A cluster of methodological work warns that models claiming to "disentangle" intrinsic from spatial effects are frequently over-trusted because the validation itself leaks:

| Reference | Core warning |
|---|---|
| Ploton et al., *Nat Commun* 2020 (spatial autocorrelation in ecological model validation, and the follow-up filtered-CV literature: Meyer & Pebesma et al., 2022–2024) | random CV on spatially autocorrelated observations inflates apparent accuracy; buffered / leave-location-and-time-out CV needed |
| **Spatial+** (Wang et al., 2023, geospatial ML) | proposes a CV split that explicitly accounts for the distance-decay of spatial dependence |
| **Differential Expression Analysis for Spatially Correlated Data** (bioRxiv 2024.08.02.606405, preprint) | Count-level DE on imaging SRT is *systematically* mis-signed when neighbouring cells are correlated; BYM2/CAR random-effect models restore correct error control; segmentation bias shifts fold changes — a concrete instance of leakage interacting with intrinsic-signature inference |
| **openproblems.bio SVG benchmark** (Chitre/…/Lun et al., Genome Biology 2023 "Systematic benchmarking of SVG detection") | compares SpatialDE/2, SPARK(-X), MERINGUE, BayesSpace, … on 60 simulated and real datasets — a starting point for anyone validating their own decomposer; note the benchmark simulates spatial signal, not the interaction term |
| Cawley & Talbot (2010) / Varma & Simon (2006) / Whalen et al. (2022) | generic overfitting & selection-bias warnings — cite as foundation if you are building a *benchmark* rather than a *method* |

**Practical rule of thumb** emerging from the literature: if your intrinsic/spatial split is only validated by *reconstruction* (how well you predict held-out genes/spots in the same tissue), a spatially leaking split can make a *conflating* model look better than a *disentangling* one. Use (i) simulated ground truth with known component shares, (ii) held-out *tissue blocks* or *sections* rather than held-out cells, (iii) counterfactual perturbation tests as in Cellina, and (iv) evaluation after artefact correction (SpotClean/bleeding-corrected).

---

## 7. Design patterns (synthesis)

1. **Structural identifiability comes from dependence structure.** Intrinsic ≈ spatially *white*; spatial field ≈ *smooth* / autocorrelated with pairs; niche effect ≈ tied to *neighbour composition*, not coordinates; leakage ≈ *local directional transfer*, stronger at boundaries and heterotypic interfaces. All four are, in principle, separable signatures — most published models exploit at most two of them. Nobody, as of this scan, has published a single model with all four explicit terms, identifiability proofs, and benchmarking on both spot- and imaging-based data. **This is the clearest open gap.**
2. **Correct the measurement layer before the decomposition layer.** A decomposition fitted to bleeding-contaminated counts will absorb artifact into "spatial" signal (this is worth citing: several bleeding papers report changes in SVG calls post-correction).
3. **Two "% variance explained" conventions compete** — per-gene variance decomposition (SVCA, SpatialDE) vs per-cell/basis attribution (MISTy, NCEM). Make the unit of analysis explicit when comparing.
4. **Supervision as an inductive bias, not a crutch.** Cell-type or domain labels can *encourage* cleaner separation (Cellina, STANCE) but reintroduce the annotation-dependence that unsupervised methods (SIMVI, DRVI) try to avoid. Decide which trade-off your study needs.
5. **Composition confounding is the most common hidden failure.** A "niche effect" in naive analysis is frequently just cell-type mixture (i.e., a purely intrinsic signal at the spot level); C-SIDE, STAN, STANCE handle it explicitly.

---

## 8. Adjacent omics

- **Spatial proteomics / multiplexed imaging (CODEX, IMC, MIBI, MACSima):** segmentation-assignment problems are the *same* leakage problem here (cell boundary errors operating on protein panels), and most disentanglement is still by MISTy-style or OPLS-style regression; strictly probabilistic disentanglement analogues of SIMVI/Celcomen are **absent** as of this scan.
- **Spatial epigenomics / multiome:** SIMVI's melanoma multiome application and FADVI's batch/label disentangled VAE are the two clearest entry points.
- **Spatial ATAC-seq** (e.g. spatial ATAC–RNA-seq,Fan et al.): same decomposition structure (intrinsic chromatin accessibility vs spatially induced accessibility) is generally stated qualitatively rather than modelled with identifiability guarantees — a present, but early-stage, bridge for your topic.

---

## 9. Suggested reading order (10 papers)

1. **SVCA** — Arnol et al., *Cell Systems* 2019 — the original intrinsic/environment/interaction decomposition.
2. **SpatialDE** — Svensson et al., *Nat Methods* 2018 — spatial vs non-spatial variance components, GP, per-gene.
3. **NSF/NSFH** — Townes & Engelhardt, *Nat Methods* 2023 — count-aware spatial factor decomposition.
4. **SIMVI** — *Nat Commun* 2025 — intrinsically/spatially disentangled latent blocks with identifiability; current reference point.
5. **NCEM** — *Nat Biotechnol* 2022 — niche-composition attribution by GNN.
6. **MISTy** — *Genome Biology* 2022 — the simplest, most general explainable attribution framework.
7. **C-SIDE** — *Nat Methods* 2022 — the strongest "don't confuse spatial signal with composition" template.
8. **SpotClean** — *Nat Commun* 2022 — spot-swapping/leakage correction.
9. **Mitchel et al. spatial bleeding** — bioRxiv 2024.08.06.597016 (+ peer-reviewed 2026 version) — spatial bleeding across imaging platforms.
10. **Celcomen** — ICLR 2025 / *Nat Commun* 2026 — causal identifiability of intra- vs inter-cellular regulation.

---

## 10. Glossary / terminology mapping

| Term used here | Terms used in the literature |
|---|---|
| intrinsic variation | cell-type/state signal, "cell-intrinsic", non-spatial factor, "self-view" (MISTy intraview) |
| spatial effect | autocorrelated signal, "spatially variable", tissue domain, spatial random effect, "spatially-induced" (SIMVI), paraview |
| niche / interaction effect | neighbourhood effect, juxtaview, intercellular / cell–cell interaction term, "sender–receiver" |
| leakage (measurement) | spot swapping, spatial bleeding, RNA bleed, ambient/stray RNA, misassigned / missegmented transcripts, contamination, admixture |
| leakage (evaluation) | spatial data leakage, spatially-buffered CV, "leave-location-out" |

---

## 11. Sources, access notes

- Retrieved and read through publisher, PMC, bioRxiv, arXiv or project pages; 18 live web resources were fetched this session and used in the synthesis above.
- Where only the **bioRxiv** version exists, the DOI/ID is given so you can check for journal updates later (e.g., SpatialDE2, Niche-DE, MisTIC, Cellina, FADVI, the DE-for-spatially-correlated-data paper, `pageSize`).
- Two documents blocked bot-header-based access (one PMC page, one bioRxiv full text). References citing them come with enough retrieved PID (PMC ID / DOI) that you can obtain the text from another route if you need the numbers in them rather than the claims.

## 12. Leads that could not be verified in this pass

| Lead | Status |
|---|---|
| **DENDER** (mentioned in one web result as a deconvolution method supporting niche-aware expression) | NOT VERIFIED — no preprint, paper, GitHub, or publisher record surfaced; treat as unverified and do not cite without a primary source |
| *pageSize* / v.PageSize (transcript-count segmentation refinement, Zhu et al. 2025) | bioRxiv ID retrieved (2025.07.11.648441) but a journal record was not confirmed in this pass |
| "Dissection of tumoral niches using spatial transcriptomics and deep learning" (TG-ME, PMC11994907) | real preprint/journal paper, but its exact publication venue could not be pinned down in this pass — PMC page ID given above |
| GC-MoE (genomics-guided mixture-of-experts for histology→single-cell ST) | arXiv/alphaXiv record exists but the ID/DOI matching this exact title could not be retrieved; listed only as a lead |
| spatial proteomics disentanglement | no dedicated probabilistic disentanglement model surfaced; frame in the review as a gap, not as a citation |
