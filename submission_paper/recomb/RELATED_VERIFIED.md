# Related work: entry-by-entry verification (tab:related + introduction.tex related-work paragraphs)

Checked 2026-10-01. Read only: no .tex or .bib file was edited.

**Sources read**
- **resolVI**: bioRxiv v1 PDF, `submission_paper/articles/` (Methods: Model, eq. 1–2; Implementation; Supervised scenario), and the scvi-tools code in `.venv/.../scvi/external/resolvi/_module.py`. The journal version came out in *Nature Methods* on 24 Sep 2026 (doi 10.1038/s41592-026-03212-9). It is paywalled and I did **not** read it.
- **SIMVI**: *Nat. Commun.* 16:2990 (2025) PDF (Results; Methods: generative process eq. 1, regularisation, "Spatial effect identification", eqs. 15–17).
- **MintFlow**: bioRxiv PDF on disk. The footer says "posted July 24, 2025", which is v2. A v3 exists and I could not read it in full. I read Methods: model overview, "Notes on identifiability and objective terms" (Objectives 7–9), and Discussion.
- **Celcomen**: *Nat. Commun.* 17:4126 (2026) PDF (Methods, Propositions 1–2; Discussion).
- **Cellina**: arXiv 2606.08493v2 HTML (§3.2, App. D.1, Definitions 1–2, Discussion).
- **NicheCompass**: PMC11985353 (*Nat. Genet.* 57(4):897–909, 2025).
- **NCEM**: bioRxiv 10.1101/2021.07.11.451750 full text and the *Nat. Biotechnol.* abstract.

Confidence levels:
- **H**: verified in the primary text.
- **M**: verified in a secondary rendering, or a judgement call about how a reviewer will read the cell.
- **L**: from memory only.

---

## 1. tab:related, cell by cell

| Row / column | Current | Verified | Evidence | Conf. |
|---|---|---|---|---|
| resolVI / Intrinsic latent | ✓ | ✓ | z_n ~ MoG; "latent embeddings of the cell itself" (Methods, Model, eq. 1–2) | H |
| resolVI / Niche term | × | × (keep) | No niche-dependent term in the cell's own expression. The neighbour term is diffusion, i.e. misassignment (eq. 1). The "spatial niche embedding" in the Discussion is a post-hoc analysis, not a model term. | H |
| resolVI / Leakage | ✓ | ✓ | h_ng = α0 f(z) + α1 Σ β_nN f(z_N) + α2 bg (eq. 1). "gene expression of neighboring cells N(n) that were incorrectly segmented" | H |
| resolVI / Leak share | estimated per cell | ✓, but more precisely "estimated per cell (amortised α), plus per-neighbour weights" | α_n is amortised from the cell's own counts by the "diffusion encoder" (MAP, Dirichlet prior with concentrations 3/2). The per-neighbour β is MAP, not amortised, under an RBF-distance Dirichlet prior (Methods, Model). It is a three-way proportion that includes background. | H |
| resolVI / Invariance | × | × | No regulariser between z and the neighbourhood | H |
| resolVI / Counterfactuals | × | × **only if the caption says "intervention on the neighbourhood"** | resolVI's `model_corrected` sets the diffusion and background proportions to 0 and generates expression from that (Methods, Implementation). Under the current caption ("expression predicted under an intervention"), its authors could call this a ✓. **Fix the caption, not the cell.** | M |
| SIMVI / Intrinsic latent | ✓ | ✓ | z_i ~ N(0, I), encoded from x_i (Methods, eq. 1; Results) | H |
| SIMVI / Niche term | spatially induced latent | ✓ | s_i \| z ~ N(A z_π(N(i)), Σ_s); q(s \| x_N(i)) via GAT; ρ_i = f(z_π(i), s_i, b_i) (eq. 1) | H |
| SIMVI / Leakage | × | × | No mention of segmentation, contamination, spillover or misassignment anywhere in the paper (full-text grep) | H |
| SIMVI / Invariance | independence, marginal | ✓. Optionally "independence (MI or MMD), marginal, applied to z only" | "we utilize an independence regularization term between s and z, and only regularize z by the term"; closed-form Gaussian MI or kernel MMD (Methods) | H |
| SIMVI / Counterfactuals | × | **Change**: "spatial effect (treatment-effect estimate)", or keep × only with a caption restricted to *generated* expression under a changed neighbourhood | SIMVI's headline contribution is the "spatial effect (SE)", estimated "via continuous treatment effect estimation framework in causal inference" (DML / partial regression, eqs. 15–17), with positivity indices. Dong and Kluger will object to a bare ×. | M–H |
| MintFlow / Intrinsic latent | ✓ | ✓ | "z_n, a priori conditioned on cell type, captures the intrinsic variation" (Methods, Model overview) | H |
| MintFlow / Niche term | microenvironment latent | ✓ (more precisely s_n^in, the neighbour average of s_m^out, plus a separate count component x^mic) | s_n^in = mean over neighbours of s_m^out; x_n = x_n^int + x_n^mic (Methods) | H |
| MintFlow / Leakage | × | × | Not modelled. Discussion: "MintFlow is susceptible to segmentation errors produced by current technologies" | H |
| MintFlow / Invariance | adversarial, given type | ✓. Precise wording: "Wasserstein critics, per cell type" | Objectives 7–8: the MCC is not predictable from z or x^int. "for each cell type population, there is a cell type-specific discriminator … predicts if it resides in the neighborhood of any cell of that cell type"; WGAN-GP style (Methods, Notes on identifiability) | H |
| MintFlow / Counterfactuals | in silico perturbation | ✓ (cell deletion or replacement in the microenvironment) | "simulating deletion or replacement of any cell within a tissue microenvironment" (Results; Methods) | H |
| Celcomen / Intrinsic latent | × | × | No per-cell latent. It learns intra- (g′) and inter-cellular (g) gene–gene interaction matrices in an energy/max-entropy model (Methods, eq. 1, Prop. 1) | H |
| Celcomen / Niche term | inter-cellular gene interactions | ✓ | Σ_ij s_i J_ij g s_j term over neighbouring cells (Prop. 1) | H |
| Celcomen / Leakage | × | × | No mention of segmentation or contamination (full-text grep) | H |
| Celcomen / Invariance | -- | -- | n/a | H |
| Celcomen / Counterfactuals | gene knockouts | ✓ | Simcomen generates p({s_{i≠j}} \| s_j^β = 0, …); "prediction of spatial counterfactuals … had a gene been knocked-out" (Methods) | H |
| Cellina / Intrinsic latent | ✓ | ✓ | q(z \| x), MLP on the focal cell's counts (§3.2) | H |
| Cellina / Niche term | extrinsic latent | ✓ | s from the degree-normalised average of the neighbours' log-normalised expression (MLP), or GATv2 in Cellina-GAT (§3.2); decoder on [z; s], NB (App. D.1) | H |
| Cellina / Leakage | × | × | Not modelled. Discussion: segmentation "remains error-prone … transcripts are frequently misassigned across cell boundaries" | H |
| Cellina / Invariance | adversarial against domain labels, marginal | ✓ | A discriminator predicts the spatial-domain label from detached z and the encoder is trained to fool it (alternating; not type-conditional). A cell-type classifier sits on z (§3.2) | H |
| Cellina / Counterfactuals | swapped neighbourhoods | ✓. Optionally add "or perturbed neighbour expression" | Def. 1, edge perturbation (N(v) := N′); Def. 2, node perturbation (neighbour genes scaled by δ_g); evaluated by context transfer against held-out cells | H |

Caption, `—` vs `--`: no issue.

---

## 2. Related-work text claims (introduction.tex)

| # | Claim (current) | Verdict / corrected entry | Evidence | Conf. |
|---|---|---|---|---|
| T1 | "Node-centric expression models … yield a niche descriptor **without a per-cell latent** \citep{fischer2023}" | **Wrong as stated.** The NCEM paper also has a latent-variable variant: a CVAE "in which the condition represents the neighborhood and the cell-type of the cell itself", i.e. a per-cell intrinsic latent conditioned on the niche. Qualify to "the linear and graph-neural NCEMs …", or mention the CVAE-NCEM. Also note that NCEM takes **cell-type labels** as input. | NCEM bioRxiv full text, model section; the *Nat. Biotechnol.* abstract says the same. | H |
| T2 | SIMVI "splits a cell's state into an intrinsic and a spatially induced part with an independence regulariser" | ✓ | Methods (see table) | H |
| T3 | (implicit) only NCEM is credited for "response w has a prior centred on a regression on [the niche]" | **Missing credit.** SIMVI's prior is s_i \| z ~ N(A z_π(N(i)), Σ_s): the spatial latent's prior is a linear regression on the neighbours' intrinsic latents. That is the closest analogue of our w-prior. Cite SIMVI here too. | SIMVI Methods eq. 1 | H |
| T4 | MintFlow "conditions an intrinsic latent on cell type and a microenvironment latent on neighbour-type composition, and discourages the former from predicting the latter with type-specific discriminators" | ✓. Two nuances: the conditioning on MCC is indirect (s^out is conditioned on each neighbour's type, then averaged), and the critics act on both z and x^int. | Methods | H |
| T5 | Cellina "encodes an intrinsic latent from the cell's counts and an extrinsic one from its neighbours' expression, ties the former to cell type with a classifier, removes spatial-domain labels from it adversarially, and evaluates the split by predicting expression under a swapped neighbourhood" | ✓ | §3.2, Defs. 1–2 | H |
| T6 | Celcomen "separates intra- from inter-cellular gene-interaction programmes with a generative graph neural network" | ✓ ("Generative Graph Neural Network", Methods) | Methods | H |
| T7 | NicheCompass "characterises niches with a graph neural network" | ✓ but thin. Better: "learns one embedding per cell whose dimensions are prior-knowledge communication gene programmes, through a GATv2 graph VAE decoding both the cell's own and its aggregated neighbourhood expression". It has no intrinsic/niche split, no leakage term and no counterfactuals. The `\todo` can close. | PMC11985353, Methods | H/M |
| T8 | "None of them models transcript leakage" | ✓ for SIMVI, MintFlow, Cellina, Celcomen, NicheCompass and NCEM. NCEM did correct one multiplexed-imaging dataset "for spatial spillover prior to quantification", which is preprocessing, not modelling. | full-text greps; NCEM text | H |
| T9 | "MintFlow and Cellina name segmentation errors as a limitation" | ✓ for both (quotes in the table above) | Discussions | H |
| T10 | "unlike an independence penalty between two learned latents \citep{dong2025simvi}" (the model "cannot shape [the target] to make invariance trivial") | **Soften.** SIMVI's penalty is asymmetric: "only regularize z by the term", so the penalty gradient does not reshape s. s is still learned, and shaped by the ELBO and its prior, so the point stands only in weakened form. Also, MintFlow (neighbour-type composition) and Cellina (domain labels) already use **fixed data targets**, so do not let the sentence imply that a fixed target is new. | SIMVI Methods; MintFlow Obj. 7–8; Cellina §3.2 | H |
| T11 | Contribution (iii) "A type-conditional adversarial invariance …" | **Must credit MintFlow.** Its cell-type-specific Wasserstein critics against neighbour-type composition are a type-conditional adversarial invariance on the intrinsic code. What is new here is the image-derived target and the held-out probe with a permutation floor. | MintFlow Methods, Obj. 7–8 | H |
| T12 | resolVI "write a cell's counts as a mixture of its own decoded expression, its neighbours' expression decoded by the same network and weighted under a distance-kernel prior, and a slide-wide background, and estimate the mixing and neighbour weights per cell" | ✓. f_θ is shared; RBF Dirichlet prior on β; per-batch background bg_s; α amortised and β MAP per cell. | Methods, Model | H |
| T13 | "including a floor of 0.01 that the software adds" | ✓ | `diffusion_eps = 0.01`, added to the diffusion proportion in the guide ("Set minimum diffusion to 0.01. This helps with stability"), scvi-tools `_module.py` ~l.1013 | H |
| T14 | "we also adopt resolVI's stop-gradient on the neighbour path, which it uses for training stability" | ✓ ("we do not compute the gradient for x_N(n)g … This improves stability and training speed", Implementation). The scvi-tools source comment reads "sample from prior for neighboring cells (**mode collapse when gradient used**)" (`_module.py` ~l.460), so the software, though not the paper, supports the collapse reading that audit A1 removed. | PDF + code | H |
| T15 | "resolVI separates the components by expecting true expression to be low-dimensional" | ✓ ("its implicit expectation that the data can be described with a low-dimensional encoding", Overview) | Overview | H |
| T16 | SpotClean: one bleeding rate per slide and a Gaussian kernel, both estimated with the help of out-of-tissue spots | ✓ per the earlier PMC check (citation_audit, 2026-09-28); not re-read today | — | M |
| T17 | Classical models (SpatialDE, SVCA, NSF, MEFISTO, C-SIDE, MISTy) and decontamination (SoupX, DecontX, CellBender, Baysor, CellAdmix), as characterised in the first related-work paragraph | Consistent with what I know of each method; not re-read today | — | L–M |

### Bib facts found while checking (reading only; no edits made)
- `ergen2025`: now *Nature Methods* (published 24 Sep 2026), doi 10.1038/s41592-026-03212-9. Re-check the stop-gradient and α/β wording against the published Methods, which I could not access.
- `dong2025simvi`: *Nat. Commun.* 16:2990 (2025), doi 10.1038/s41467-025-58089-7. Authors: Mingze Dong, David G. Su, Harriet Kluger, Rong Fan, Yuval Kluger.
- `birk2025nichecompass`: *Nat. Genet.* 57(4):897–909 (2025), doi 10.1038/s41588-025-02120-6. Authors: Birk, Bonafonte-Pardàs, Miraki Feriz, Boxall, Agirre, Memi, Maguza, Yadav, Armingol, Fan, Castelo-Branco, Theis, Bayraktar, Talavera-López, Lotfollahi.
- `fischer2023`: add doi 10.1038/s41587-022-01467-z.
- `akbarnejad2025mintflow`: still a preprint, now at **v3**. The title on disk ("Mapping and reprogramming human tissue microenvironments with MintFlow") matches v3. Re-read v3's Methods before submission, since the critics and limitation quotes above are from v2.
- `moeed2026cellina`: arXiv v2 (10 Jun 2026); not a conference paper.

---

## 3. Optional extra columns reviewers may ask about

| Model | Labels required | Identifiability claim | Images |
|---|---|---|---|
| resolVI | optional (semi-supervised MoG + classifier) | none | no |
| SIMVI | none ("annotation-free"; batch only) | yes (Suppl. Note 1; z up to a nonlinear map, s up to a linear map, needs minimal-information z) | no |
| MintFlow | cell types (required) | yes (iVAE-style, Suppl. Note 2) | no (H&E only for interpretation) |
| Celcomen | none | yes (undirected graph, Prop. 2) | no |
| Cellina | cell types + spatial domains | explicitly **not** claimed | no |
| NicheCompass | none in model (prior gene programmes) | — | no |

None of the compared models uses images. If the DISCELL row emphasises its image-derived niche descriptor, it is the only one that does.

---

## 4. Related methods the table and text miss

**Spatial decontamination and segmentation correction**
- **SPLIT** (Bilous et al. 2026, `bilous2026split`). Cited only as evidence, but it is a correction *method*: RCTD-based two-type purification of Xenium cells against spillover. A direct competitor framing.
- **DeSpotX** (`wang2026despotx`). Identifies contamination via anchor genes; this is our marker-set ceiling turned into an identification assumption (audit A3).
- **ProSeg** (Jones et al. 2025), **BIDCell**, **Segger**, **FastReseg**. Transcript-aware (re)segmentation that reduces misassignment upstream; resolVI itself benchmarks Baysor and ProSeg.
- **ovrlpy** (Tiesmeyer et al. 2026). Detects vertical (3-D) signal overlap, a leakage source that a 2-D β kernel does not cover.
- **MisTIC** (`yang2025mistic`). Cited for prevalence, but it is a missegmentation-correction method.
- **Segmentation-free analyses** (SSAM, FICTURE, Sainsc). They sidestep cell assignment entirely; a reviewer may ask why not.
- **REDSEA** (Bai et al. 2021). Neighbour-boundary spillover compensation for multiplexed protein imaging, the closest non-transcriptomic precedent for a fixed neighbour-mixing correction.

**Intrinsic/extrinsic and niche-aware generative models**
- **scVIVA** (Levy, …, Ergen, Yosef 2025, bioRxiv). A niche-aware VAE from resolVI's lab that jointly models a cell's own and its neighbours' expression. Already in the `\todo`.
- **NCEM CVAE variant**. A per-cell latent conditioned on niche and type (see T1).
- **SpaCeNet** (Schrod et al., *Genome Res.* 2024). Separates intra- from inter-cellular gene associations; a Celcomen-like precedent, and Schrod is a Cellina author.
- **SpiceMix** (Chidester et al., *Nat. Genet.* 2023). NMF plus HMRF splitting intrinsic factors from spatial dependence; benchmarked by SIMVI.
- **ENVI / COVET** (Haviv et al., *Nat. Biotechnol.* 2024). A niche covariance descriptor, analogous to our fixed context descriptor c_i.
- **CellCharter, BANKSY, GraphST, STAGATE**. Neighbour-augmented embeddings for niche and domain detection; SIMVI's benchmark set.
- **SpatialProp** (Cellina's baseline) and **CPA / scGen**. Counterfactual or perturbation predictors that Cellina compares against.
- **DestVI** (Lopez et al. 2022). Continuous within-type variation in spots; already listed in audit C18.

---

## Summary of corrections needed
1. **T1, NCEM**: "without a per-cell latent" is false for the CVAE-NCEM. Qualify it. (H)
2. **T11, Contribution (iii)**: type-conditional adversarial invariance already exists in MintFlow. Credit it and narrow the novelty claim. (H)
3. **T3, SIMVI prior**: credit SIMVI's regression prior s \| z_neigh ~ N(A z, Σ) next to NCEM. (H)
4. **SIMVI / Counterfactuals**: × will be contested because SIMVI estimates causal "spatial effects". Change the cell, or restrict the caption. (M–H)
5. **Caption, Counterfactuals**: define it as an intervention *on the neighbourhood*, so that resolVI's `model_corrected` and SIMVI's SE do not read as counterexamples. (M)
6. **T10, SIMVI penalty**: the "two learned latents" contrast must acknowledge that the penalty is applied to z only, and must not imply that a fixed target is new (MintFlow, Cellina). (H)
7. **T7, NicheCompass**: expand the description; the `\todo` can close. (H)
8. **resolVI / Leak share**: optionally say "amortised per cell, plus per-neighbour weights, plus background". (H, cosmetic)
9. **Bib**: resolVI is now *Nat. Methods* 2026; re-check against the published Methods. MintFlow v3; SIMVI and NicheCompass metadata as above.
