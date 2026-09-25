# Citation and claim audit (opened 2026-09-23)

Every citation in the **built** manuscript, and every claim that relies on outside
work without a citation. Work through it top to bottom; tick a box when you have
read the passage and agree, or when the fix is made.

**Scope.** The files `main.tex` actually inputs: abstract, introduction, method,
conclusion, checklist, and the appendices derivations, rationale, implementation.
The experiments section and the data / calibration / synthetic / validation /
additional-results appendices are commented out of the build and are **not**
audited here. Do them when they come back in.

**Where each pointer comes from.** Trust these unequally:

- **[PDF]**: I read the passage in the PDF in `submission_paper/articles/` this
  session. Section and line pointers are exact.
- **[WEB]**: checked on the web this session. Page contents were summarised by a
  fetch tool, so read the passage yourself before relying on it.
- **[MEM]**: pointer from my own knowledge of the paper, not opened this
  session. The section is probably right, but confirm the number.
- **[DATA]**: checked against our own bundles, configs or code.

**Bib files.** `references.bib` (45 entries) is what compiles.
`articles/discell-literature.bib` (199 entries) is the wider library. Several
papers sit in both under different keys (`brody2022`/`brody2022gatv2`,
`ergen2025`/`ergen2025resolvi`, `janesick2023`/`janesick2023xenium`,
`kronos2025`/`shaban2025kronos`). When adding a reference, copy the
literature-bib entry into `references.bib` and keep one key.

---

## A. Fix before anything else

These are cases where the text currently says something its source doesn't
support, or where an obvious attribution is missing.

- [x] *(2026-09-24: fixed with R7. The intro, method §2.6 and derivations now say resolVI closes the path "for training stability". The degeneracy is our own argument, and the derivations say we have not fitted the open-path model.)* **A1. resolVI never reports mode collapse.** Three places attribute an
  observed collapse to resolVI:
  intro.tex:13 ("because the coupling otherwise collapses the fit"),
  method.tex:204 ("after observing mode collapse with it enabled"),
  derivations.tex:47 ("the empirical fact that matters is the collapse reported
  by \citet{ergen2025}").
  resolVI v1, **Methods → Implementation**, says only: *"During training, we do not
  compute the gradient for x_N(n)g in f_Θ … This improves stability and training
  speed."* "Collapse" appears nowhere in the paper and no open-path experiment is
  reported. **[PDF]**
  *Fix:* cite resolVI for *doing* the stop-gradient "for stability". Present the
  degeneracy as our own argument (the heuristic in app:amplification), or as our
  own observation if we have one in the devlog. derivations.tex:47 then has no
  empirical anchor, so the heuristic has to stand on its own or be dropped (its
  open `\todo` already asks this).

- [ ] **A2. SIMVI already has our closed-form penalty.** SIMVI **Methods, eq. (11)**:
  I(z,s) = ½ log(det Σ̂_z det Σ̂_s / det Σ̂_[z,s]), from sample covariances under a
  joint-Gaussian assumption, "only regularize z by the term" (the Methods text
  around the "Closed-form mutual information" item). It also offers an MMD
  variant. **[PDF]** This is `eq:penalty` (method.tex:172–178) without the type
  conditioning. We currently cite only Cover & Thomas for it.
  *Fix:* cite `dong2025simvi` at method.tex:172 as the source of the Gaussian MI
  regulariser, and state what we change: conditional on type, taken against fixed
  data descriptors rather than a second latent, correlation form, EMA moments,
  shrinkage, straight-through gradient.

- [ ] **A3. DeSpotX makes contamination identifiable, and we say it can't be.**
  `wang2026despotx` (bioRxiv, May 2026; PMC13192984) identifies contamination by
  assuming **anchor genes**, genes a cluster doesn't natively express. That is the
  same device as our marker-set *ceiling* in rationale.tex:49 (app:kappa). They
  use it to identify κ; we use it only as an upper bound. **[WEB]**
  *Fix:* cite it in related work (intro.tex:13) and in app:kappa. Say plainly that
  an anchor-gene assumption buys identification and that we don't make it. The
  sentence "κ is not identified … under the parametric forms we are willing to
  assume" (method.tex:225) already has the right wording, but it needs this
  citation next to it.
  *Read:* the abstract and the Methods section defining anchor genes. Check
  whether their "cluster-masked, distance-weighted average over neighbouring
  cells" is effectively our β.

- [ ] **A4. Checklist 4(a) answers [Yes] but no asset is cited.** checklist.tex:27
  says we cite the creators of existing assets. Nothing in the built text cites:
  - the image model: `shaban2025kronos` (lit bib). method.tex:17 says "a frozen
    pretrained vision model" and never names it. references.md notes that two
    generations were used, so cite both if both appear.
  - the datasets: the three 10x public Xenium Prime datasets and GSE315411. The
    10x dataset pages need entries. `10xgenomics2025prime5ktn` covers the panel.
  - software: `paszke2019`, `pedregosa2011`, `virtanen2020`, `wolf2018` are all
    in references.bib and **uncited**. `barber1996` (Qhull, for the Delaunay
    step) is too. shapely is used for the Voronoi faces and has no entry.

  *Fix:* cite them (the data ones can wait until the data appendix is back in),
  or change 4(a) to `\todo`.

- [ ] **A5. "Thousands of transcripts per cell" is false for our data.**
  intro.tex:4: *"Xenium … measure thousands of transcripts per cell"*. Mean counts
  per cell in our bundles: GSE core **140**, ovarian FFPE **262**, lung **441**,
  ovary FF **1,663** (random 30k-cell samples). **[DATA]** The paper itself says
  "tens to a few hundred counts" at method.tex:55.
  *Fix:* "thousands of **genes**" (panel size). Cite `10xgenomics2025prime5ktn`
  for the 5K panel. `janesick2023` used a ~300-gene panel, so it supports the
  platform but not "thousands".

- [ ] **A6. The Louizos et al. VFAE uses MMD, not an adversary.** method.tex:183
  cites `louizos2016` for "replaced by an adversary". The Variational Fair
  Autoencoder penalises dependence with **Maximum Mean Discrepancy** (its §3 on
  MMD). **[MEM]**
  *Fix:* keep `xie2017` and `ganin2016` for the adversary. Move `louizos2016` to
  the invariance-target sentence (method.tex:164–169) as a non-adversarial
  precedent, which also sits well next to SIMVI's MMD variant.

- [ ] **A7. Ledoit–Wolf shrinks toward scaled identity, not the diagonal.**
  method.tex:180: "Regularisation shrinks toward the diagonal
  \citep{ledoit2004,schafer2005}". Ledoit & Wolf (2004) shrink toward **μI**.
  Schäfer & Strimmer (2005) tabulate several targets, and the "diagonal, unequal
  variance" one (their target D) is ours. **[MEM]**
  *Fix:* cite `schafer2005` alone for the diagonal target. Keep `ledoit2004` only
  if the sentence says "shrinkage estimators, here toward the diagonal".
  *Read:* Ledoit & Wolf §2–3 (the target); Schäfer & Strimmer, the table of
  shrinkage targets.

- [ ] **A8. The cited NMI isn't the NMI we compute.** method.tex:199 cites
  `strehl2002`, who normalise by the **geometric** mean √(H(t)H(c)). The code
  calls scikit-learn's `normalized_mutual_info_score` with its default
  `average_method='arithmetic'` (sklearn 1.9.0). **[DATA]** The values differ.
  *Fix the text, not the code* (changing the code would move every NMI we have).
  Say "normalised by the arithmetic mean of the entropies" and cite Vinh, Epps &
  Bailey (2010) for the variants (see C16).

- [x] *(2026-09-24: resolved by review R8. The mechanism is now cited as the demographic-parity trade-off (Zhao & Gordon 2019), and the CSVAE collapse as "also documented".)* **A9. CSVAE: the symptom is documented, but the mechanism is ours.**
  method.tex:169 and rationale.tex:44. DisCoVR **Table 19** (Appendix H,
  scRNA-seq on PBMC), k-means NMI: CSVAE without the adversary has z–cell-type
  **0.702**, z–stimulation **0.187**; CSVAE with the adversary has z–cell-type
  **0.406**, z–stimulation **0.002** (DisCoVR itself: 0.716 / 0.002). **[PDF]**
  That settles both open `\todo`s ("verify the numbers", "quote the exact
  numbers").
  **But** DisCoVR's condition (IFN-β stimulation) is balanced across cell types,
  so its CSVAE collapse is **not** caused by type–condition dependence. DisCoVR
  attributes it to CSVAE reconstructing x from the condition-specific code alone,
  which lets z go uninformative (**§4, "Summary and comparison"**). **[PDF]** Our
  sentence puts our own mechanism ("carrying 'I am a T cell' predicts 'my
  neighbours are T cells'") right next to their citation, and a reader will take
  the mechanism as theirs.
  *Fix:* cite DisCoVR for the symptom only. Mark the spatial-clustering mechanism
  as our argument.

- [ ] **A10. The "one tenth to one half" misassignment range has no source.**
  intro.tex:4, method.tex:58, rationale.tex:49 (all `\needsource`). The figure
  goes back to `discell_specs.md:496` ("published Xenium estimates (0.1–0.5)"),
  which cites nothing. **[DATA]** I found no paper stating this range for Xenium.
  Closest candidates, all in the lit bib:
  - `bilous2026split` (*Nat. Methods* 23:1152–1162): documents substantial
    spillover that rises with local density of the donor type and falls with
    distance, but gives **no overall fraction**. **[WEB]**
  - `jones2025proseg` (*Nat. Methods*): 37% of DE calls potentially spurious under
    image-based segmentation vs 12% with Proseg. **[WEB]** That measures a
    different quantity.
  - `marcosalas2025xenium` (*Nat. Methods* 22:813–823), `hartman2024comparative`
    (eLife; a mutually-exclusive co-expression rate): read these for any
    per-transcript fraction.

  *Fix:* either find a source that states the range, or rephrase qualitatively
  ("a substantial fraction", citing bilous2026split and jones2025proseg) and
  justify the grid top of 0.4 by the marker-set ceiling. That ceiling is our own
  measurement (app:kappa, still `\pending`).

- [ ] **A11. "Diffuse background negligible": resolVI models background.**
  method.tex:49 (`\needsource`). resolVI's per-cell weight is a **three-way**
  proportion (own / neighbour diffusion / unspecific background), and it states
  that "both phenomena contribute to the observed expression" (Fig. 1 caption).
  Its background prior comes from **negative-control spike-ins** (Methods, "Model",
  the background-proportion prior). **[PDF]** So resolVI cuts against this claim.
  *Fix:* support it from our own data. The Xenium `metrics_summary.csv` gives an
  adjusted negative-control **probe** rate of 0.06–0.13% and a **codeword** rate of
  0.005–0.05% on the three 10x slides, against a κ grid of 5–40%. **[DATA]**
  Caveat: the **genomic-control** probe rate on ovarian FFPE is **2.1%**, which is
  not negligible next to κ = 0.05. Cite the 10x metrics documentation for the
  definitions and `hallinan2025offtarget` (eLife) for off-target binding. The
  numbers can go in once the paper takes numbers again. Note that our pipeline
  drops the control features (`gex_only=True` in `read_10x_h5`), so this needs
  `metrics_summary.csv` or a re-read with the controls kept.

---

## B. Every existing citation

One box per citation instance. "Read" is where in the cited work to look.

### slavutsky2025, DisCoVR (ICML 2026; 9 instances)

Venue confirmed: the PDF (arXiv v3, 6 Jul 2026) carries "Proceedings of the 43rd
International Conference on Machine Learning", and ICML 2026 is the 43rd. **[PDF]**

- [ ] intro.tex:10, intro.tex:13: CSVAE failure mode → see A9.
- [ ] intro.tex:13: "supplies the two-bound objective and the posterior-regularisation
  view". *Read:* **§2.3** eqs. (3)–(4): L_z and L_w as surrogates for the two KL
  targets of eq. (2). **§2.4**: the auxiliary classifier g(y|z) "as a form of
  posterior regularization (Ganchev et al., 2010)". ✔ **[PDF]**
- [ ] method.tex:114: they "condition their condition-specific posterior on the
  data rather than on the shared code". *Read:* **Algorithm 1** / **§3**:
  θ_w = f_ρ^w(x, y), i.e. q(w | x, y), not q(w | z). ✔ **[PDF]**
- [ ] method.tex:169 (×2), derivations.tex:37: the adversarial term has
  posterior-regularisation status. *Read:* **§2.4** (as above). ✔ **[PDF]**
  See also C1: whether an MI penalty fits Ganchev's formal definition.
- [ ] derivations.tex:30: the sum of the two bounds corresponds to minimising
  KL(q(z)‖p(z|x)) + KL(q(z)q(w)‖p(z,w|x,u)). *Read:* **§2.3**, "Since direct
  evaluation of the KL divergences in Equation 2 is intractable…". ✔ **[PDF]**
- [x] *(2026-09-24: closed via review R16.)* derivations.tex:32: "a structured prior couples the two codes through the
  condition" + `\todo{state the exact form of that prior}`. *Answer:* p(w | y) is
  aligned with **the class-wise mean of z**, μ_k = mean of z over class k.
  *Read:* **§2.4**, and **§6 Conclusion**: "a prior over w conditioned on the
  class-wise mean of z". **[PDF]** The `\todo` can be closed with that sentence.
- [ ] rationale.tex:44: see A9.
- [ ] *Bib:* drop `note = {arXiv:2506.17182}` (you prefer no arXiv in the bib) and
  add the PMLR volume and pages once they are out.

### ergen2025, resolVI (bioRxiv; 3 instances)

- [ ] intro.tex:13: "estimate a per-cell mixing weight". Right, but loose: it is a
  **three-way** amortised proportion α_n over own / diffusion / background, MAP-fitted
  with Dirichlet-style concentrations 3 (true) and 2 (diffusion). *Read:*
  **Methods → Model**, the paragraph on α_n and the priors. **[PDF]** Consider "a
  per-cell three-way mixture over own expression, neighbour diffusion and
  background".
- [x] intro.tex:13, method.tex:204, derivations.tex:47: collapse → **A1**. *(2026-09-24: done.)*
- [x] *(2026-09-24: corrected: two authors, title, doi, bioRxiv preprint.)* *Bib:* the references.bib entry is wrong. It has exactly **two** authors, Can
  Ergen and Nir Yosef, so drop "and others". Title: "ResolVI - addressing noise and
  bias in spatial transcriptomics"; doi 10.1101/2025.01.20.634005. The lit-bib
  entry `ergen2025resolvi` is correct, so use it.
- [ ] *Venue:* scvi-tools' own reference list still gives bioRxiv. **[WEB]** Search
  snippets claimed Nature Methods 2026 acceptance, but I couldn't confirm it.
  Check yoseflab.github.io/publications before submission. If it is published,
  re-check A1 against the published Methods, since a revision may add the
  open-path experiment.

### dong2025simvi, SIMVI (*Nat. Commun.* 16:2990, 2025; 1 instance)

- [ ] intro.tex:13: "split a cell's state into an intrinsic and a spatially induced
  part with an independence regulariser". ✔ *Read:* title and **Methods**, the
  "independence regularization term between s and z" (closed-form MI or kernel
  MMD). Also note the GAT variational posterior (Results, Fig. 1b). **[PDF]**
- [ ] intro.tex:13: "none of them models transcript leakage". ✔ for SIMVI: the
  text never mentions contamination, spillover, misassignment or leakage.
  **[PDF]**
- [ ] **Missing instance** → A2 (eq. 11 is our penalty).
- [ ] While there: SIMVI claims identifiability when z "encodes minimal
  information" (**Supplementary Note 1**). That contrasts directly with our
  §2.8 and is worth a clause.

### birk2025nichecompass, NicheCompass (*Nat. Genet.* 2025; 1 instance)

- [ ] intro.tex:13: "niche-characterisation models built on graph neural networks
  … none of them models transcript leakage". *Read:* Methods, model definition
  (graph VAE with gene-program decoders). Confirm there is no segmentation or
  contamination term. **[MEM]**, not on disk.

### fischer2023, NCEM (*Nat. Biotechnol.* 2023; 2 instances)

- [ ] intro.tex:13, method.tex:231: "node-centric expression models … regress a
  cell's expression on its neighbourhood directly … without a per-cell latent".
  *Read:* the abstract and the model section defining NCEMs (linear and
  graph-neural variants). Check that "without a per-cell latent" holds for
  **all** their variants: they also describe a latent-variable (VAE-style)
  NCEM. **[MEM]** If so, qualify this as "the non-latent variants".

### elazar2018 (EMNLP 2018; 2 instances)

- [ ] intro.tex:13, method.tex:191: "adversarial removal is not sufficient without
  an independent check". *Read:* the experiments where a post-hoc attacker
  recovers the protected attribute from adversarially trained representations,
  and their conclusion that an adversary at chance doesn't imply removal.
  **[MEM]** ✔ matches our usage.

### ganchev2010, posterior regularisation (*JMLR* 11; 4 instances)

- [ ] intro.tex:13, method.tex:52, method.tex:169, derivations.tex:37.
  *Read:* **§2**, the PR framework. Constraints are written as expectations,
  Q = {q : E_q[φ(x,z)] ≤ b}, and enforced through a KL projection. **[MEM]**
  *Check:* a mutual-information constraint is **not** a linear expectation
  constraint on q. "Posterior regularisation in the sense of Ganchev" is
  therefore an analogy, the same one DisCoVR makes. Consider "in the spirit
  of", or state that the penalty form generalises PR's expectation constraints.

### klys2018, CSVAE (NeurIPS 2018; 4 instances)

- [ ] intro.tex:10, intro.tex:13, method.tex:169, rationale.tex:44. *Read:* **§3**,
  the CSVAE model and its mutual-information / adversarial term. **[MEM]** The
  failure-mode evidence comes from DisCoVR, not from Klys (A9), so check that no
  sentence attributes the collapse to Klys themselves.

### brody2022, GATv2 (ICLR 2022; 2 instances)

- [ ] method.tex:84: "a single graph-attention layer". ✔
- [ ] rationale.tex:9: "the destination feature enters the attention but never the
  value". *Read:* the section defining GATv2's scoring function
  e(h_i,h_j) = aᵀ LeakyReLU(W[h_i ‖ h_j]) and its update h_i' = σ(Σ_j α_ij W h_j).
  **[MEM]** Two checks. (1) With **no self-loops** the value is built from
  neighbours only (our setting, method.tex:90). With self-loops h_i would enter
  the value, so the claim depends on that choice; say so. (2) Our
  W_dst/W_src split is the bipartite, `share_weights=False` form of the library
  implementation. Make sure the equation in rationale.tex:9 matches what the
  paper defines, or describe it as our parameterisation.
- [ ] *Add:* `velickovic2018` (the original GAT; in references.bib, uncited)
  alongside `brody2022` at method.tex:84.

### kingma2014, rezende2014 (1 instance, together)

- [ ] method.tex:117: learned diagonal variances "make the reparameterised sample
  meaningful". *Read:* Kingma & Welling **§2.4** (the reparameterisation trick)
  and **Appendix B** (Gaussian KL); Rezende et al. **§3** (stochastic
  backpropagation). **[MEM]** ✔

### kingma2016, free bits (NeurIPS 2016; 2 instances)

- [ ] method.tex:219, rationale.tex:11. The IAF paper introduces "free bits" in its
  training-details appendix, not the main text. Search the PDF for "free bits".
  **[MEM]** ✔
- [ ] Naming trap: the code's `w_free_bits` allowance is in **nats**
  (elbo.py:61). If a value is ever quoted, give the unit.

### tomczak2018, VampPrior (AISTATS 2018; 1 instance)

- [ ] rationale.tex:58: "a VampPrior with archetypal pseudo-inputs". *Read:* **§3**:
  the prior as a mixture of variational posteriors at learned pseudo-inputs.
  **[MEM]** ✔

### louizos2016, xie2017, ganin2016 (1 instance, together)

- [ ] method.tex:183: "replaced by an adversary". `xie2017` ✔ (the adversarial
  invariance framework, §3). `ganin2016` ✔ (DANN: a domain classifier with
  gradient reversal, the method section). `louizos2016` ✗ → **A6**. **[MEM]**

### ledoit2004, schafer2005 (1 instance, together)

- [ ] method.tex:180 → **A7**.

### bengio2013, straight-through estimator (arXiv; 1 instance)

- [ ] method.tex:180. *Read:* the section on the "straight-through estimator": the
  forward pass uses the hard function, the backward pass the identity. **[MEM]**
  *Check:* our use is an **analogy**: the value comes from the EMA moments and the
  gradient from the batch. It is not STE through a nondifferentiable unit. Say
  "in the manner of a straight-through estimator".
- [ ] *Bib:* arXiv-only, and no peer-reviewed version exists. It is the canonical
  citation, so keeping it is acceptable.

### cover2006 (1 instance)

- [ ] derivations.tex:42: Gaussian MI as ½[log|Σ_z| + log|Σ_v| − log|Σ_[z,v]|].
  *Read:* **Ch. 8**, the theorem on the differential entropy of a multivariate
  normal; the MI follows as h(z)+h(v)−h(z,v). **[MEM]**
- [ ] The scale-invariance step in the same paragraph (h(AX) = h(X) + log|det A|)
  is also Ch. 8. Cite it there too, or leave the one-line derivation as is.
  **[MEM]**

### strehl2002, NMI (*JMLR* 3; 1 instance)

- [ ] method.tex:199 → **A8**. *Read:* the definition of NMI with the √(H·H)
  normalisation. **[MEM]**

### roberts2017, spatial block CV (*Ecography*; 1 instance)

- [ ] method.tex:214: held-out spatial tiles, because autocorrelation leaks across
  a random split. *Read:* the sections on block cross-validation and the
  explanation of why random splits are optimistic under dependence. **[MEM]** ✔

### khemakhem2020, iVAE (AISTATS 2020; 1 instance)

- [ ] method.tex:225: "recovery up to an affine map, under an injective
  additive-noise decoder and sufficient variability of the conditional prior".
  *Read:* **§2**, the model x = f(z) + ε; **Theorem 1**: conditional
  exponential-family prior, injective f, the "sufficient variability" condition
  (nk+1 distinct values of u), identifiability up to an affine
  (A-)equivalence. **[MEM]** ✔ Our hedge ("which we do not establish") is right.

### locatello2019 (ICML 2019; 1 instance)

- [ ] method.tex:225: "no comparable guarantee … without the constraint". *Read:*
  **§3**, Theorem 1 (the impossibility of unsupervised disentanglement without
  inductive biases). **[MEM]** ✔

### manski2003, rosenbaum2002 (1 instance, together)

- [ ] method.tex:228: the sweep as a sensitivity analysis "in the spirit of partial
  identification". *Read:* Manski, Introduction and Ch. 1–2 (identification
  regions). Rosenbaum, **Ch. 4**, "Sensitivity to hidden bias" (Γ as the
  sensitivity parameter). **[MEM]** Our κ plays Γ's role. Consider saying that
  explicitly: it is the clearest analogy.

### janesick2023, Xenium (*Nat. Commun.* 14:8353; 1 instance)

- [ ] intro.tex:4 → **A5** (the platform ✔; "thousands of transcripts" ✗).
- [ ] *Bib:* the author list is truncated (CHECK note). Complete it or keep
  `others` deliberately.

### kingma2015adam, loshchilov2017 (1 instance)

- [ ] implementation.tex:46: Adam; cosine annealing. Loshchilov & Hutter (SGDR):
  the cosine schedule equation in **§3**. **[MEM]** ✔ Also cite
  `loshchilov2017` at method.tex:217, where "cosine annealing" first appears.

---

## C. Claims that need a reference and have none

Candidates marked **lit** are already in `discell-literature.bib`. **new** means
not in either bib. The full citation is given for those.

### Platform and data

- [ ] **C1.** method.tex:12, rationale.tex:18: "the nuclear-expansion segmentation
  that in situ platforms fall back on". **lit** `10xgenomics2024segmentation`
  (Xenium Onboard Analysis segmentation docs).
- [ ] **C2.** method.tex:12, rationale.tex:20: Voronoi-face ⇔ Delaunay adjacency;
  "average degree near six". **new** de Berg, Cheong, van Kreveld & Overmars,
  *Computational Geometry: Algorithms and Applications*, 3rd ed., Springer 2008
  (Ch. 7 Voronoi diagrams; Ch. 9 Delaunay triangulations, with the ≤ 3n−6 edge
  bound behind average degree < 6). **[MEM]** Also `barber1996` for the
  implementation. Our data agrees: mean Voronoi degree 5.93 on pdl018d **[DATA]**.
- [ ] **C3.** method.tex:25: the contamination decay length τ = 20 µm and the
  distance-decaying kernel form. **lit** `bilous2026split` computes contamination
  against heterotypic neighbours **within a 20 µm radius** and finds it falls with
  distance. **[WEB]**, from a search snippet. If confirmed, this is direct support
  for both τ and the shape of β. *Read:* the Results subsection "Xenium data exhibit
  substantial transcript spillover".
- [x] *(2026-09-24: "dominant" removed by review R13, so the claim no longer needs a source. Erickson et al. 2022 remains optional support for contiguous subclones.)* **C4.** method.tex:52: "on a carcinoma section … subclonal structure is the
  dominant within-type spatial variation". **new** Erickson et al., "Spatially
  resolved clonal copy number alterations in benign and malignant tissue",
  *Nature* 608:360–367 (2022). **[MEM]** It shows spatial subclonal structure but
  not that it is *dominant*. Soften to "a major source" or support it from our own
  data.

### Likelihood and preprocessing

- [ ] **C5.** method.tex:55: the multinomial as Poisson conditioned on the total,
  and why it suits UMI counts. **new** Townes, Hicks, Aryee & Irizarry, "Feature
  selection and dimension reduction for single-cell RNA-Seq based on a multinomial
  model", *Genome Biology* 20:295 (2019). **[MEM]** This is the domain-standard
  source for exactly this argument.
- [ ] **C6.** method.tex:55: "negative-binomial sources with gene-specific means are
  not closed under addition". It's a standard fact, so either no citation or the
  same source as C5.
- [ ] **C7.** method.tex:104: "a raw log(1+x) entangles depth with composition".
  **new** Hafemeister & Satija, "Normalization and variance stabilization of
  single-cell RNA-seq data using regularized negative binomial regression",
  *Genome Biology* 20:296 (2019). **[MEM]** **lit** `lopez2018scvi` for the encoder
  convention.
- [ ] **C8.** method.tex:55: Dirichlet-multinomial "with one concentration scalar,
  not G dispersions". Optional, since this is a design remark.

### Variational inference and VAE claims

- [ ] **C9.** method.tex:114: explaining-away, "w ⊥ x | z, c is false". **new**
  Pearl, *Probabilistic Reasoning in Intelligent Systems*, Morgan Kaufmann 1988;
  or Wellman & Henrion, "Explaining 'explaining away'", *IEEE TPAMI* 15(3):287–292
  (1993). **[MEM]**
- [ ] **C10.** method.tex:117 (rescaling gauge) and method.tex:225 (rotation
  invariance under an isotropic prior with a linear decoder). **new** Tipping &
  Bishop, "Probabilistic principal component analysis", *JRSS-B* 61(3):611–622
  (1999). The ML loading W is determined only up to an arbitrary rotation R.
  **[MEM]** This is the classic statement of our rotation freedom.
- [ ] **C11.** method.tex:153 (eq:objective): per-term weights on the
  divergences. **new** Higgins et al., "β-VAE: Learning basic visual concepts
  with a constrained variational framework", ICLR 2017. **[MEM]** DisCoVR cites it
  for the same move (§2.3, eq. 7). **[PDF]**
- [ ] **C12.** method.tex:219, rationale.tex:34: "posterior collapse under a large
  α_w". **new** Bowman et al., "Generating sentences from a continuous space",
  CoNLL 2016. Lucas, Tucker, Grosse & Norouzi, "Don't blame the ELBO! A linear VAE
  perspective on posterior collapse", NeurIPS 2019. **[MEM]**
- [ ] **C13.** implementation.tex:99: "the amortisation gap". **new** Cremer, Li &
  Duvenaud, "Inference suboptimality in variational autoencoders", ICML 2018.
  **[MEM]**
- [ ] **C14.** method.tex:180 (eq:penalty) → **A2**, SIMVI eq. (11).

### Estimation details

- [ ] **C15.** method.tex:191: k-means "never on the raw high-dimensional
  embedding, where distances are noise-dominated". **new** Beyer, Goldstein,
  Ramakrishnan & Shaft, "When is 'nearest neighbor' meaningful?", ICDT 1999.
  **[MEM]**
- [ ] **C16.** method.tex:199: NMI normalisation variants → A8. **new** Vinh, Epps
  & Bailey, "Information theoretic measures for clusterings comparison",
  *JMLR* 11:2837–2854 (2010). **[MEM]**
- [ ] **C17.** derivations.tex:47: "a row-stochastic P_β … whose spectral radius is
  at most one". **new** Horn & Johnson, *Matrix Analysis*, 2nd ed., CUP 2013
  (Gershgorin, Ch. 6; nonnegative matrices, Ch. 8). **[MEM]** Standard, so this one
  is optional.

### Related work (intro.tex:13 and its two `\todo`s)

- [ ] **C18.** *(2026-09-24: MintFlow and Celcomen now cited in related work, see review R6; scVIVA, DestVI and GASTON still open.)* Further spatial VAEs: **lit** `levy2025scviva` (scVIVA, bioRxiv;
  its own `\todo` asks for it); `akbarnejad2025mintflow` (MintFlow, preprint, on
  disk, and one of our baselines); `megas2026celcomen` (Celcomen, *Nat. Commun.*
  2026, on disk; GO localisation 6b.10 borrows its takeaway);
  `lopez2022destvi`; `chitra2025gaston`.
- [ ] **C19.** *(2026-09-24: `bilous2026split` and CellAdmix (`mitchel2026celladmix`) now cited in the introduction, see review R4.)* Contamination and segmentation correction: **lit**
  `wang2026despotx` (→ A3), `bilous2026split`, `tiesmeyer2026ovrlpy` (3D signal
  overlap, *Nat. Biotechnol.* 2026; spillover along z), `jones2025proseg`,
  `petukhov2022baysor`, `fu2024bidcell`, `heidari2025segger`, `yang2025mistic`.
  Ambient-RNA analogues from droplet data: `young2020soupx`, `yang2020decontx`.
- [ ] **C20.** Baselines we compare against should all be cited somewhere once
  experiments return: resolVI, SIMVI, MintFlow.

### Optional (common knowledge; cite only if a reviewer pushes)

- [ ] intro.tex:10, method.tex:169: "cell types cluster in space". Easiest to show
  on our own data.
- [ ] method.tex:225: the softmax's additive-constant non-identifiability.

---

## D. Bib hygiene: preprints and published alternatives

| key (in text) | status | action |
|---|---|---|
| `slavutsky2025` | ICML 2026, confirmed | drop the arXiv note; add PMLR details when out |
| `ergen2025` | bioRxiv; entry wrong | fix the authors (two) and title; watch for Nature Methods |
| `bengio2013` | arXiv only; never published | keep, it's canonical |
| `shaban2025kronos` (to add) | arXiv 2506.03373; an AACR 2026 abstract exists, but that's not the paper | cite arXiv. The existing `kronos2025` entry has the wrong title ("KRONOS: …"; the real one is "A Foundation Model for Spatial Proteomics") |
| `bilous2026split` (to add) | *Nat. Methods* 23:1152–1162 | use the journal version |
| `wang2026despotx` (to add) | bioRxiv / PMC only | acceptable as the only version |
| `levy2025scviva` (to add) | bioRxiv | check for a published version first |
| `akbarnejad2025mintflow` (to add) | preprint | check for a published version first |
| `janesick2023` | published | complete the author list |

---

## Also found while auditing (not citation issues)

- The model prunes edges at **40 µm** (`model/prepare.py:36`), as the paper says.
  My devlog entry from earlier today said 60 µm. A correction entry has been
  appended to the devlog. **[DATA]**
- implementation.tex:78 `\todo{quantify}` (the clip's effect on β for kept edges)
  can partly draw on the devlog entry "Preprocessing: what the Voronoi face
  actually measures": the clip binds on 2.21% of cells slide-wide, and zero bundle
  edges survive at d ≥ 60 µm.
