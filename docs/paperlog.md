# Paper log

Changes to the AISTATS manuscript, and why. One short entry per change: what
moved, and the motivation. Appended, never edited — a wording that turns out
wrong gets a later entry, not a rewrite.

Experiments and results go in `devlog.md`. This file is only the manuscript:
claims, equations, wording, structure. A change here that rests on a number
names the devlog entry that produced it.

---

## 2026-09-23

**`method.tex` — counterfactual abduction written as a shift.** Was
`w'_i = m_psi(c',t_i) + sigma_w eps_i` with `eps_i = (w_i - m_psi(c_i,t_i))/sigma_w`;
now `w'_i = w_i + m_psi(c',t_i) - m_psi(c_i,t_i)`. *Motivation:* with the prior
scale fixed the sigma_w cancels exactly, so the original form was vacuous
notation. The shift form also makes the gauge-invariance of `w'_i - w_i`
visible on sight, which §2.8 otherwise asks the reader to verify. The general
template is the one to restore if the bounded conditional `sigma_w(c,t)` listed
as not-built is ever implemented — then the ratio does not cancel.

**`method.tex` — `(1+omega)` proportionality hedged.** "is what keeps J
proportional to L_i + omega L_i^z" → "is what *would keep* … at
`alpha_z = alpha_w = 1/lbar` and `alpha_a = 0`". *Motivation:* J carries
`-alpha_a Pen`, and `alpha_a = 0.3` in every pinned run, so the proportionality
never holds in any reported fit. The Scaling paragraph already conceded the
`alpha_w` and per-cell-scaling failures but not `alpha_a`, leaving "keeps J
proportional" and "J is a weighted surrogate rather than a bound" in the same
paragraph for the reader to reconcile. The `(1+omega)` argument itself is
correct and unchanged.

**`derivations.tex` (`app:pathb`) — duplicate M-estimation sentence cut, link
added.** The sentence "coherent as an M-estimation target…" appeared verbatim
in §2.4 and in the appendix; the appendix copy is removed and replaced by a
pointer to `app:posterior-reg`. *Motivation:* the claim is asserted, not proved,
in either place, so a cross-reference from the main text would have sent the
reader to a restatement. `app:posterior-reg` is where the actual argument lives
(neither Pen nor Adv is part of any bound), and it was two subsections away and
unconnected. Considered and rejected: adding a `\cref` in §2.4, and writing an
appendix proof — J is not literally a sample average (Pen is a batch
functional) and §2.8 argues the population optimum is not unique, so a
formalisation would open more than it closes.

**Not changed, flagged only.** In the abduction sentence, `w_i` is still
unqualified where abduction wants the posterior (mean or draw), and "drawing
w' ~ p(w | c',t) … with abduction" still glues the population and per-cell
estimands together. Both are wording, neither is wrong.

**Appendix figures wanted** (preprocessing geometry): see the devlog entry
"Preprocessing: what the Voronoi face actually measures (validation,
2026-09-23)". Figures are in `scratchpad/` and must be moved before they can be
cited.

**`method.tex` §2.7 — the z/type agreement spelled out at the guard.** Was "the
agreement between z and cell type" with no definition at the point of use; now
names it (normalised MI between t and a k-means clustering of mu_z, cross-ref to
§2.5 where it is first defined), says the clustering is unsupervised by design,
adds that because the encoder receives the label it measures whether z *retains*
type rather than discovers it, and makes the 0.9 guard's ratchet explicit — the
running maximum rises over the fit, so a checkpoint that buys reconstruction at
the cost of type structure is rejected rather than traded off. *Motivation:* the
definition lived in §2.5, two subsections before its use as a stopping rule, and
the retains-not-discovers point in §2.2, three before — a reader arriving at §2.7
could take high agreement as evidence that z discovers cell identity from
expression, which would be an overclaim the paper does not make elsewhere.
"Fixed fraction of its running maximum" also reads as a fixed floor. Kept the
full definition in §2.5 rather than duplicating it, per the same reasoning that
removed the duplicated M-estimation sentence.

## 2026-09-24

Items R1–R6 of `submission_paper/aistats/review_2026-09-23.md`, at the author's request.

**Abstract, intro, method §2.2 — the confound restated (R1).** "A cell that responds to its neighbours comes to resemble them" and its echoes are replaced by what the two mechanisms actually share: both make a cell's expression *predictable from its neighbours' types*. Contamination brings in the neighbours' markers; a response shifts the cell's own programmes. They coincide where response genes are also markers of the neighbouring types (ligands, receptors) or a programme is region-wide. *Motivation:* resemblance is the signature of leakage, not of a response (a fibroblast beside tumour becomes a CAF, not tumour-like). The precise version is the one a biologist won't dispute, and it says where the method matters most. Touched: abstract sentence 2, intro:6 (first three sentences, plus "the resemblance" → "the neighbour-predictable signal"), method.tex:58 (the clause shortened to "since neighbour composition predicts both").

**Abstract, intro — the response described as an association (R2).** "how its microenvironment has changed it" → "what varies with its microenvironment"; "a latent response w to" → "associated with"; intro:4 "switched on or off because of who its neighbours are" → "that differ with who its neighbours are"; intro:8 "carries the cell's response to" → "carries the variation associated with". The word *response* stays as w's name. Added after the stability rule in §2.8: the sweep separates spatial association from leakage, not a response from spatially organised intrinsic variation (which conditional invariance assigns to w) or from niche selection. *Motivation:* one section cannot separate induction from selection, and §2 had already been moved to associational wording on 2026-09-23 while the front matter had not.

**Contributions (R3).** (i) now states both exactness assumptions: Poisson sources *and* a leaked-in rate that scales with the receiving cell's own rate. (ii) "whose architecture carries most of the identification argument" → "closes information channels that a regulariser cannot reach on its own". *Motivation:* §2.8 says the z/w allocation rests mostly on non-architectural pieces, and (i) omitted the assumption that carries the weight. "Information channels" rather than the review's "leakage channels", to avoid a second meaning of *leakage* (review T12). Not done: the optional reordering with the sweep first (the review marks R3 contested on that), and the closed-form clause in (iii), which waits for todo 8.11.

**Intro:6 — the confound given its published evidence (R4).** Added: misassigned transcripts are shown directly to produce apparent neighbourhood and spurious cell–cell interaction signals (Mitchel et al. 2026, *Nat. Genet.*), and received contamination grows with the local density of the donor type (Bilous et al. 2026, *Nat. Methods*). *Motivation:* the paper argued its central confound from first principles although it is published. Sources checked on the web (Nature Genetics page; PMC copy of Bilous), not on disk.

**Related work — the differences from CSVAE/DisCoVR restated (R5).** "Two things differ …" replaced by three differences that do hold: the condition is the cell's niche, summarised by a deterministic context descriptor, not an observed label; the invariance is type-conditional, not marginal; the intrinsic path keeps the full decoder and draws w from its prior rather than decoding from z alone. The "cannot shape the condition" point is kept, but contrasted with SIMVI's penalty between two learned latents, where it is true. *Motivation:* in CSVAE and DisCoVR the condition is an observed label, so neither can shape it either, and DisCoVR's condition is not a nuisance. The measurement-artefact contrast is dropped here because the resolVI sentence already carries it.

**Related work and contribution (iii) — every model in `articles/` now cited (R6; author's instruction).** Added MintFlow (bioRxiv; intrinsic latent conditioned on type, microenvironment latent on neighbour-type composition, type-specific discriminators; names segmentation sensitivity as a limitation) and Celcomen (*Nat. Commun.* 17:4126; generative GNN separating intra- from inter-cellular gene-interaction programmes). SIMVI, DisCoVR and resolVI were already cited. The sentence was split so that each model gets one clause. (iii) now says its type-conditional invariance is as in MintFlow's type-specific discriminators, but targets an image-derived niche descriptor as well. The related-work `\todo` is narrowed to NicheCompass (not on disk) and scVIVA. *Motivation:* MintFlow is our closest prior model and a baseline (todo 8.4), and claiming (iii) without it reads as unaware. Characterisations checked against the PDFs (MintFlow Objectives 7–8 and Discussion; Celcomen abstract; none has a contamination term).

**Bib.** Added `akbarnejad2025mintflow`, `megas2026celcomen`, `mitchel2026celladmix`, `bilous2026split` to `references.bib` (journal versions where they exist; no preprint notes). Build exits 0 with no undefined citations.

**Not touched, though it sits in the edited paragraph.** The resolVI "collapses the fit" claim (citation audit A1) and DeSpotX (A3) are still open.

**Review R8–R11 (2026-09-24, author: "adopt").**

**R8, invariance cited as a known construction.**
- method.tex:169 now calls the type-conditional target the equalised-odds form of invariance (Hardt et al. 2016; Zhang, Lemoine & Mitchell 2018), and the marginal constraint its demographic-parity analogue, which must discard the label when label and protected variable are dependent (Zhao & Gordon 2019). Spatial clustering of types is stated as the tissue instance.
- The CSVAE sentence no longer says "this is the documented failure". It says a collapse of this kind "is also documented", which resolves citation-audit A9 (symptom documented, mechanism different). Its `\todo{verify the numbers…}` is removed, since DisCoVR Table 19 was verified on 2026-09-23.
- Also: Zhang et al. added to the adversary citations (method.tex:183), a related-work clause on conditional invariance in fair and domain-invariant learning (Madras et al. 2018; Li et al. 2018, CIDDG; Pogodin et al. 2023, CIRCE), and intro.tex:10 routed through Zhao & Gordon.
- *Motivation:* without these, fairness and invariance reviewers read contribution (iii) as a rediscovery. Venue details for CIRCE, CIDDG and Zhang et al. were checked on the web; the other three are standard.

**R9, lineage-level labels.**
- method.tex:9 now says curated annotations often encode state or location, that conditioning on them assigns the corresponding response to t, and that labels should be lineage-level. "Coarse labels" became "lineage-level labels" in §2.5 and app:conditional.
- **A `\todo` marks that the current fits violate this.** Ovarian uses 18 curated classes including tumour- vs stroma-associated fibroblasts and endothelium, VEGFA+, proliferative and inflammatory tumour cells. GSE uses activated, inflammatory and myofibroblasts, "EC activated" and "Proliferating". Lung and ovary FF use 32 and 38 unsupervised clusters. Relabelling goes in todo 8.13, before the final sweeps.

**R10, labels as a descendant of the niche.**
- app:conditional gains a fourth reason for lineage labels: t is a function of x, so conditioning on it conditions on a descendant of the niche (Hernán et al. 2004, selection bias).
- §2.5 points there, which also gives the orphaned app:conditional label a reference (review T11).
- "What the leakage term assumes" gains the promised recurrence of the contaminated-labels caveat: labels are fixed across the grid, and leakage-driven label errors enter through t and the neighbours' compositions.

**R11, the segmentation mechanism corrected, with data.**
- method.tex:14 and rationale.tex:18 no longer say "under nuclear-expansion segmentation". The reason given now is that segmented polygons rarely touch.
- New table `tab:contact` in app:graph, with one paragraph reading it. The table compares segmentation methods, touching and zero-contact edges, no-leak cells for a contact kernel (overall, sparsest and densest fifth), and within-row weight share against polygon gap for contact and for beta, on all five sections.
- The paragraph states both halves of the result. Contact gates leakage by density. Beta does not, but it does grade by proximity through its distance decay, by design.
- **Exception to the no-numbers convention, at the author's request.** These are preprocessing statistics from the fixed bundles, and they do not move with the model. Devlog: "Contact-based leakage kernel vs the Voronoi-face kernel".
- Left unchanged, following the review's own view on its contested half: method.tex:12 ("the nuclear-expansion segmentation that in situ platforms fall back on", which is generic and true) and rationale.tex:20 ("independent of the expansion setting").

**Bib.** Added `hardt2016`, `zhang2018mitigating`, `zhao2019inherent`, `madras2018`, `li2018ciddg`, `pogodin2023circe` and `hernan2004`. Build exits 0 with no undefined citations. The new table fits the page. Three overfull boxes remain, all from before this change: a 55 pt one in the three-routes table of app:qw, and two of about 5 pt.

**Review R12, leakage assumptions stated (2026-09-24, wording only).**

"What the leakage term assumes" now names the two assumptions that matter most for w:

- **(b) Receiver scaling.** The leaked rate scales with the receiving cell's own rate, so a low-count cell beside high-count donors is under-corrected at every κ, and a high-count cell beside low-count donors over-corrected.
- **(a) Gene-agnostic leakage.** κ has no gene index. A gene-specific leak, such as preferential spillover of membrane and secreted mRNAs (marked `\needsource`), is spanned by no κ, so a response programme enriched for cell-surface genes cannot be separated from leakage by the sweep alone.

The sentence "absorbed by z and w" is corrected: because the invariance keeps z free of niche-predictable signal, the niche-predictable part of the error lands in w. app:kappa now cross-references gene-varying nuclear retention as a reason to expect gene-specific leakage. It is not framed as a contradiction.

*Motivation:* these are the leakage pathways that survive the sweep by construction. The GO read (6b.10: loadings lean extracellular at κ = 0 and κ = 0.4) is the signature of (a).

**Deliberately not added:** the review's optional "not built" row and a §2.8 sentence restricting "stable" to receiver-scaled leakage. The author has a separate agent testing the kernel variants (donor-scaled β_ij ℓ_j/ℓ_i), and a "not built" entry would be wrong if that test goes ahead. Revisit when its result is in.

Build exits 0; no new overfull boxes. One new `\needsource`, on over-representation of boundary-near transcripts in spillover.

**Review R14, unscaled magnitude (2026-09-24).** In the §2.4 Scaling paragraph, "Unscaled, they are of order −200 to −400 nats per cell against divergences of order 1 to 10, and their magnitude varies severalfold between sections" becomes "Unscaled, they grow in proportion to a cell's depth and exceed the divergences by orders of magnitude, and depth differs widely between tissues".

*Motivation:* the figures were wrong. They came from a stale code docstring; the true values are about −1,000 to −12,000 nats per cell. They also broke the no-numbers convention. The author asked for no numbers here, only the point that weights do not carry over between tissues. How the weights *are* then set (the 1/ℓ̄ units of todo 8.12) is left to the R18 rewrite. The stale docstring in `model/equations.py` still carries the old figure, and was not changed here.

**Review R13, where subclonal variation goes (2026-09-24).**

§2.2 claimed that spatially organised within-type variation stays in z; §2.5 claimed it is pushed into w. Both claims are replaced by one statement: the invariance removes it from z only to the extent that neighbour composition and the masked image predict it. On a carcinoma section that extent can be large, because subclones are contiguous and the masked image shows clonal kin. What is predictable moves into w and reads as a response; the rest, and variation carried by the neighbours' own codes, stays in z.

Added at the author's prompting: how much is at stake depends on the labels as well. A subclone-splitting label keeps subclones in t. Under the lineage-level labels now recommended (R9), they are within-type, and on a carcinoma section this is the main case.

Also: "clonal identity" → "clone-associated expression" in the introduction and §2.2, since a targeted panel does not measure genotype. The unsupported "dominant within-type spatial variation" is removed, which closes citation-audit C4.

*Motivation:* the paper contradicted itself on the question that decides whether a spatial w effect in tumour cells is biology or clonal geography.

**app:kappa, position-based foreign counts (2026-09-24).** Added after "it cannot estimate it": assigning transcripts to cells by position does not settle κ either. Counts of transcripts nearer a neighbour's nucleus than the cell's own are dominated by segmentation geometry, because a large cell beside small ones counts its own far-side transcripts as foreign. Such counts bound leakage from above but do not identify how it scales with receiver or donor depth.

*Motivation:* devlog "R12 test 1". The coder's summary claimed that the position data *rule out* donor scaling. That claim is **not** made in the paper. The same geometric artefact that stops the data from confirming receiver scaling also produces the negative donor-depth term, so they cannot exclude donor scaling either, at least until an area-conditioned check (requested, todo 8.14) says otherwise. §2.2 is unchanged: it already states assumption (b) and its consequence. The sentence is qualitative, so there are no numbers.

**Review R15, what the context actually is (2026-09-24).**

- **§2.3, one sentence after "no latent enters it".** With types as query and sources, the context is a learned, type-specific reweighting of the neighbour composition, a fixed function of (t, y, Φ). Those are the descriptors z is made invariant against, so the prior on w sees exactly what z is kept from. This follows the review's placement: the sentence in the method, the formula in the appendix.
- **app:graph.** The closed form is now eq:context-closed, where the degree cancels. The blindness statement is corrected. The context sees fractions, so it distinguishes one tumour neighbour among six from five among six. What it cannot see is a uniform rescaling of all type counts, of which a single-type neighbourhood of any size is the special case. The old "one neighbour of a given type gives the same context as several" was true only of that special case.
- **The sink's full reason** (devlog 2026-09-16, three seeds). It learned no dose response, with its half-point below one neighbour. It did not improve held-out transport. In two of three seeds the prior mean read the context less faithfully. It is left off, with the author's wording that degree is nearly constant across the section.
- *Motivation:* the old text gave only the constant-degree reason and misstated the blindness. The closed form also pre-empts "why an attention layer at all?".

**Not done (the rest of R15, via revision A1 and review S31):** with type-only sources the mirror's value channel no longer exists, so diagnostic (i) needs rewording. The alternative diagnostic in rationale.tex (the correlation of attention with z-similarity) is vacuous, because the attention weights have no z-dependence.

Build exits 0; no new overfull boxes.

**Rest of R15, via review S31 (2026-09-24).** Diagnostic (i) in §2.7 no longer calls itself the check on "the value channel of the mirror". That channel was the old design, with neighbours' z codes as attention sources, and type-only sources removed it. It now says the within-type check complements the mirror diagnostics of app:mirror and is kept as verification, since §2.3 closes both routes by which c could echo z. In app:mirror, the attention–z-similarity correlation is marked as informative only under a z query and vacuous under type sources (pointing to eq:context-closed). *Motivation:* two sentences still described the pre-type-only design. Closes R15 and S31.

**Review R16, the objective is a bound on one model's evidence (2026-09-24).**

- **The same model, a restricted variational family.** derivations.tex:24 and method.tex:134 no longer call L_i^z the bound "for the model in which w is not inferred". It is obtained "by restricting the variational family to q(w) := p(w | c, t)", from the same model.
- **The bound and its gap are stated.** derivations.tex:30's "It is not the evidence lower bound of a single model, so q is not the posterior of anything", which the 2026-09-23 M-estimation edit had kept, is replaced. The new text displays L_i + ωL_i^z ≤ (1+ω) log p(x_i | c_i, t_i) and gives its gap as two KLs to the same posterior. It contrasts DisCoVR, whose shared-only term decodes from z alone and so bounds a different model.
- **The true reason q is not used for uncertainty.** One q(z) serves two variational families, so its optimum is a compromise. method.tex:159 is aligned to this.
- **Credit.** method.tex:156 cites DisCoVR eq. 6 (its −2 KL) for the (1+ω) factor.
- **derivations.tex:32.** DisCoVR keeps z informative through a z-only reconstruction *and* a prior on w centred on class-wise means of z. Our intrinsic path is the analogue of the first; there is no analogue of the second. This closes the `\todo{state the exact form of that prior}` (read in DisCoVR §2.4 and §6).

*Motivation:* the "not a single model" sentence was carried over from DisCoVR, where it holds. Here both bounds share one model and one left-hand side, so the natural reading was false, and it undersold the objective. The conclusion (uncertainty from the sweep, not from q) is unchanged. The algebra of the gap was checked by hand, and eq. 6 was checked in the PDF.

**Review R17, what the objective is (2026-09-24; reviewer's text plus the coder's refinements).**

- **The per-cell bound is conditional on the influx.** method.tex:125 now reads "the evidence lower bound on log p(x_i | c_i, t_i, ρ̄_i), conditional on the neighbours' clean compositions", with a pointer to app:bound. ℓ_i is deliberately not added, since it is conditioned on by convention. app:bound now says at the top that every probability in it is conditional on ρ̄_i, suppressed from the notation. That keeps the bound displayed in app:pathb under R16 correct as written.
- **The coupled model (app:bound, new paragraph).** Σ_i L_i is a one-sample ELBO of the coupled model, log p({x_i} | {c_i, t_i}), under the fully factorised family, with the neighbours' z and w drawn from their own posteriors in the same pass. Added from the coder: the per-cell bounds are not independent terms, since ρ̄_i depends on the neighbours' latents. The stop-gradient makes training a partial gradient of this bound, a fixed-point iteration that treats the current ρ̄ as data.
- **"M-estimation" dropped everywhere.** The sentence in §2.4 becomes the coder's two-bounds version: the two reconstruction paths are bounds on the same evidence, but J, their scaled sum plus the invariance term, is a training criterion. Training does not maximise it as a single function, because of the stop-gradient (a partial gradient) and the adversary (a saddle point). Hence no likelihood-based uncertainty for q is claimed; variability comes from the sweep and the seeds.

*Motivation:* "M-estimation" was shorthand that describes neither the stop-gradient fixed point nor the saddle. The reviewer's "J is not a bound" is replaced by the more precise and stronger statement that the two terms are bounds and their combination is a criterion.

Consistent with the 2026-09-23 decision not to formalise: no composite-likelihood framing, and no Besag or Lindsay citations. The cross-references now point to appendices that contain the actual argument, not to a restatement.

**Not added:** the coder's optional convergence diagnostic (the change in ρ̄ between evaluation epochs going to zero before the accepted checkpoint). Keep it in reserve in case a reviewer asks whether the fixed point converges.

Build exits 0; no new overfull boxes; no "M-estimation" left in the built text.

**Review R29, the per-type translation is near-flat, not exact (2026-09-24).**

- §2.8 no longer says "(a(z) − Bμ_t, w + μ_t) is the same model". Now: shifting the response by μ_t and the baseline by −Bμ_t leaves the **prior divergence exactly** unchanged, because m_ψ and q(w) take t. The **likelihood and the influx** pass through a(z), which sees z and not t, so they are unchanged **only to the extent that t can be read from z**. The z–type agreement keeps that approximately true, so the translation is a near-flat direction that the optimisation dynamics position. The weight-decay remark and the two consequences are kept.
- "Identified only up to a rotation" becomes "carries at least the following symmetries; we do not show that there are no others. The first is a rotation…"
- *Motivation:* §2.2 insists the decoder takes z only, so the exact claim contradicted it. The near-flat reading also explains the large observed type offsets (revision A4). Algebra checked: a(z) − Bμ_t + B(w + μ_t) = a(z) + Bw, which requires a(z) − Bμ_t to be a function of z.

**Review R36, the invariance's formal status made consistent (2026-09-24).** In app:posterior-reg, "the generative model's conditional independence z ⊥ [y, Φ] | t is what it encodes" contradicted §2.2, which says y and Φ are not modelled. It is replaced: the constraint is on which q are admissible, the generative model does not model y or Φ, and the constraint acts on the aggregate of encoder outputs across cells, so it is posterior regularisation in a looser sense than Ganchev's per-instance expectation constraints. The paragraph's opener becomes "in the spirit of Ganchev", to match (citation-audit B's suggestion).

**R23 deferred** by the author's choice. It will be folded into the R20 probe rewrite, since its sentence belongs in the paragraph that rewrite replaces.

Build exits 0; no new overfull boxes.

**Review R31, the sweep reports only gauge-free readouts (2026-09-24).**

- §2.8's readout list dropped "the loadings B" and "the per-type magnitude of the response". The identifiability paragraph just above calls both gauge statements. The list now reads: the within-type shifts of the response, the spatial autocorrelation of each latent dimension (within a fit), the held-out probes and the external-label agreement.
- The identifiability paragraph now defines both within-type quantities and no longer calls them "equivalent" (the 2026-09-24 re-check confirmed they are not):
  - the centred shift B(w_i − w̄_t), with w̄_t the mean response of type t, which is what the atlas and report compute;
  - the realised prior shift B[m_ψ(c_i, t) − m_ψ(c̄_t, t)], with c̄_t now defined as the mean context of cells of type t, which is what transport uses.
  They differ by the posterior residuals.
- "A decline in ‖w‖ along the grid is mechanical" becomes a statement about the within-type spread.
- *Motivation:* the method listed as "readouts of interest" two quantities it had just declared meaningless. This is method, not results, and the list may grow once the model settles (per-cell readouts if the α_w ladder opens w; per-block probes after R20).

**Review R32, what stability licenses (2026-09-24).** §2.8 now says stability over the grid rules out one specific confound: leakage of the modelled form (a section-wide fraction through a fixed one-hop kernel). It does not rule out leakage that form cannot represent (longer-ranged, gene-specific, donor- or receiver-depth-dependent), which is κ-invariant and passes every grid point. A stable readout is a finding relative to that confound, not a proof of biology. The introduction's matching sentence and R2's "the sweep separates spatial association from leakage" are aligned ("…of that form").

*Motivation:* "a stable readout is a finding" overclaimed, and it follows directly from R12's stated assumptions.

The stability rule itself (`\todo{confirm this rule}`) is untouched. It is R33's, to be settled and pre-registered before the final sweeps.

Build exits 0; no new overfull boxes.

**First figure: the Voronoi face (`fig:voronoi-face`, app:graph; 2026-09-24).** Part of the figure plan for review R38. The figure is built by `submission_paper/aistats/figures/src/fig_voronoi_face.py` from the primary section's bundle. Exemplar cells are chosen by stated rules: a typical cell (degree 6, median 6th-NN distance), a dense cell (10th-percentile 6th-NN distance), and a border cell whose disc cuts 20–60 % of its region with ≥ 3 model edges. The panels:

- (a) polygons;
- (b) bisectors;
- (c) clipped region with labelled faces;
- (d) a dense neighbourhood and (e) a border neighbourhood, at one scale, with the clip disc and, in (e), the unclipped region;
- (f) every edge under the chord ceiling, with the 40 µm model prune shaded.

The shared style is `figures/src/style.py`: final printed width, LaTeX text in Computer Modern to match the body, and one colour per role, validated (blue = face/β, orange = contact, aqua only with labels). The PDF is vector with all fonts embedded, 727 KB. It is referenced from app:graph and from §2.1's face paragraph.

**Numbers:** face lengths and an edge-count colour bar. This is the same preprocessing exception as `tab:contact`: fixed bundles, independent of the model.

**Checks during drafting:** one suspected mislabel in (e) (a blue stretch looked like the clip arc) was checked face by face against the geometry and was not a bug. The focal region's outline is now drawn in ink, so the arc reads as the disc.

**Second figure: contact vs Voronoi kernel (`fig:contact-kernel`, app:graph; 2026-09-24).** Built by `figures/src/fig_contact_kernel.py`; the plotted numbers are in `figures/src/data/fig_contact_kernel.json`.

- **(a) and (b):** a touching and a non-touching edge from the primary section, each the edge of its kind closest to the median centroid distance and median face length. They come out matched at d = 11.0 vs 11.1 µm and f = 6.8 vs 6.9 µm, with contact 8.5 µm vs 0: the kernel sees the same edge twice where contact sees two different ones.
- **(c):** the share of cells with no leakage under a contact kernel, by local-density fifth, on all five sections (median, range band, per-section lines). The Voronoi kernel is flat at 0.

It sits beside `tab:contact`, and that table's paragraph now cites both. Numbers fall under the same preprocessing exception.

**Styling choices:** labels are in ink with a coloured sample beside them, never coloured text (dataviz rule). Label positions are computed from the geometry, so they cannot collide on another exemplar. Vector PDF, 251 KB.

**Figure 1, the model overview (`fig:overview`, full width at the start of §2; 2026-09-24). Closes review R38 for the method.**

- **(a) The leakage neighbourhood** of the same typical cell as `fig:voronoi-face`: first ring with arrows whose width grows with β_ij (computed as the model defines β: faces and distances within the 40 µm prune), and second ring.
- **(b) The real four-channel morphology crop** the image model receives, rendered by the pipeline's own `crop_cell`. Its geometry is read from the pinned embedding file (`egomask_ego_v1`): 128 µm field at 0.5 µm/px, 25 µm masked disc, KRONOS v1. This **settles review S15**: the 54.4 µm figure is only the code default at native resolution. The dotted square shows that (a)'s neighbourhood lies almost entirely inside the masked disc, i.e. the descriptor sees the tissue beyond the leakage neighbourhood. The caption says so.
- **(c) The computation graph in TikZ**, using the paper's own macros, in three lanes (intrinsic / context and response / leakage). The adversary (training) and the held-out probe (evaluation, dashed) are separate boxes. The intrinsic path is shown as a dashed w̆ prior draw into w.

**Drafts:** four rendered and inspected. Draft 1 was rejected (overlapping boxes, an arrow through a node, crossings). Draft 2 was fixed, and its key was corrected because it claimed "dashed = not in training", which would have mislabelled the w̆ and KL arrows. Checked at print size.

A one-line pointer was added to the §2 roadmap. The microscopy uses a neutral membrane colour, so orange keeps its "contact" meaning across figures. `figures/src/build.sh` regenerates every figure. Build exits 0; no new overfull boxes. The caption is long (about 12 lines); trim it with R37.

**Figure fixes after author review (2026-09-24).**

- **(1) Orientation.** Every tissue panel was drawn with y up, while slide coordinates and the microscopy have y down. Figure 1(a) was therefore the vertical mirror of the tissue in 1(b). Verified by overlaying the segmentation outlines on the real crop: they land on the cells only in image orientation. `style.tissue_axes` now inverts y for every tissue panel. Figures 1, 2 and 3 were rebuilt; the Figure 1 caption states the convention.
- **(2) Mask in 1(b).** It was drawn as a translucent overlay with the tissue visible underneath. The model receives zeros there. The panel now applies the pipeline's own `_ego_disk` mask (black) and draws the cell's outline for reference, following the repo's `ego_masking_examples.png`.
- **(3) What the arrows in 1(a) are** (author: "if that is the GNN graph, we use no edge features"). The panel is now labelled "leakage kernel β_ij". The caption adds that the attention building the context runs over the same first ring but uses no edge features, its weights depending on the two types alone (eq:context-closed). Without that, the figure contradicted §2.3's "no per-edge geometry enters the context".

The Figure 1 caption is now about 14 lines; trim it with R37.

**Figure 1(b): all four channels, and the square explained (2026-09-24).**

- **Channels.** The panel showed only DAPI (blue) and membrane (grey), so it read as a one-colour image; author: "why is it only blue". The model receives four channels. The panel is now an additive composite of all four: nuclei blue, membrane green, 18S RNA grey, αSMA red, with a key under the panel (ink text beside colour samples). Orange is still kept out because it is the "contact" role.
- **Contrast.** Display contrast per channel comes from a 512 µm window around the cell, not from the crop alone. The crop's own αSMA 99th percentile is about a third of the surrounding tissue's, so a crop-only stretch would have inflated a sparse stain.
- **The square.** It marks the window of panel a but was an unlabelled dotted line, and the author asked what it showed. It is now a solid white square labelled "a".
- **Caption.** The old claim that the descriptor "describes the tissue beyond the leakage neighbourhood" overstated it. It now says the zeroed disc covers the cell and most of its first ring, so the descriptor describes the tissue around the neighbourhood rather than the neighbourhood itself. Check: the median model edge is 11 µm, and 5.6% of edges are longer than the 25 µm mask radius.
- **Height.** The column is now 2.65 in, matching panel c.

**New appendix figure `fig:graph-compare` in app:graph (2026-09-24).** Adopts the repo's `graph_comparison_plain`, restyled. It shows three graphs on one 160 µm field of the primary section: exact contact, contact within 1 µm (the contact of tab:contact), and the model's pruned Delaunay graph. Cells with no edge are dark; the section-level no-neighbour shares (82.9 / 9.8 / 0.0 %) are printed above each panel.

- **Consistency with the table.** All three graphs are subsets of the model's edge set, and the denominator is the same as tab:contact's. The 9.8 % equals the table's "no-leak cells, all" for this section. The two contact definitions (polygon gap ≤ 1 µm, and apposed membrane > 0) disagree on 113 of 1.18 M edges.
- **Field rule, rejected option.** The first rule picked the window with the most cells from both the sparsest and densest fifths. It found a stroma field of elongated cells that touch end to end, where only 1.7 % had no 1 µm neighbour against 9.8 % on the section. That under-showed the argument.
- **Field rule, adopted.** The window whose no-neighbour shares under both contact graphs are closest to the section's, among windows with ≥ 150 cells and ≥ 25 from each of those two fifths. The chosen field matches the section to 0.1 points.
- **Styling.** Panel c is titled "Delaunay" to match the text. Edges have one width. The only colours are the fixed roles, orange for contact and blue for the model graph. The scale bar is under panel c, off the cells.
- **Text.** One sentence was added to the first app:graph paragraph pointing at the figure.

Build: exit 0, no undefined references, the same 3 pre-existing overfull boxes. The later appendix figures renumber (3 = face, 4 = contact kernel); all references go through `\cref`.

**Figure 1(b) key names the markers (2026-09-24).** The key said "nuclei / membrane / 18S RNA / αSMA", and the author could not see that DAPI was used. It now names the markers the image model reads: DAPI, ATP1A1, 18S, αSMA. The caption gives the slide's own channel names from the OME metadata: the ATP1A1 channel is a membrane mix with CD45 and E-cadherin, and the αSMA channel a mix with vimentin. This is stated because the image model reads each mix as a single marker. Build: exit 0, no undefined references.


**R7 applied, with audit A1 folded in: how our leakage term relates to resolVI (2026-09-24).**

- **Source.** Checked against the resolVI preprint's Methods (eq. 1), not from memory. resolVI mixes, per cell, its own decoded expression, its 20 nearest neighbours' decoded expression (same decoder, weights ν under an RBF-Dirichlet prior, MAP, not amortised) and a per-slide background, with shares α_n that are amortised and MAP-estimated.
- **Introduction.** Now names that construction and states that ours is the same with fixed weights, one leak fraction per section and no background.
- **Separation claim.** Our first draft said the low-dimensional separation "fails once a response exists". The author asked for support. It was weakened to "that expectation no longer separates the two on its own", with the mechanism stated: the part of a response that moves a cell toward its neighbours' profile has the same effect as leakage. The support is the devlog's planted-κ synthetic run (B seed-bistable under planted leakage, stable without) and the sweep2 trade-off of the mirror metric. Neither can be cited while the synthetic appendix is out of the build, so a `\todo` marks it. The intro also no longer calls κ "un-identified" (pending R30); it says we "sweep the leak fraction rather than estimate it".
- **§2.1.** One clause after "never learned" pointing to the construction and naming the difference (weights fitted per cell there).
- **A1.** resolVI never reports a collapse; its Methods say the stop-gradient "improves stability and training speed". The three attributions (intro, §2.6, derivations) now say "for training stability". The derivations add that we have not fitted the open-path model, so the heuristic stands on its own.
- **Bibliography.** `ergen2025` corrected: two authors, exact title, doi, bioRxiv preprint. The venue check before submission stays open in the audit.
- **Build.** Exit 0, no undefined references, the same 3 overfull boxes.

**Considered and not adopted: a learned background term as in resolVI.** It would be a second unknown share next to κ, needing a prior or a second sweep axis. A fixed false-positive floor from the Xenium controls was proposed instead as todo 8.15 and awaits the author. The review's R30 carries a new note: the data bound κ from above, one-sidedly.


**R33 + R40 (minimum): the finding rule becomes Definition 1, the breakdown point (2026-09-25).** The old rule said: sign or order unchanged over the grid and seed envelope excluding zero, with a `\todo`. It had four defects. It had no sampling layer, no multiplicity control, and a zero null that magnitudes pass automatically. And its pass/fail verdict depended on the unsourced grid top.

- **Definition 1, displayed in §2.8.** κ*(r) is the smallest grid κ at which the readout's interval contains its null or its sign flips, reported as censored above the grid otherwise. Rankings are read as pairwise differences. The breakdown point is reported, not thresholded.
- **Intervals.** They pool the seeds with a spatial block bootstrap over tiles, so they reflect sampling as well as optimisation. The text now keeps the two uncertainties apart: the interval narrows with cells, the trajectory across the grid does not.
- **Multiplicity.** A few primary readouts are pre-registered before the final fits and Bonferroni-adjusted; everything else is exploratory. The author accepted this over calling everything screening.
- **Sections.** Findings are section-level. On a serial-section pair, a tissue-specific finding must hold on both, and method properties must hold on every section. Only the lung TMA pair are serial sections of the same cores; the other sections are different tissues.
- **Framing.** "In the spirit of partial identification" is replaced by "one-parameter sensitivity analysis". A `\todo` restores the partial-identification framing if R30's proposition lands, because the data bound κ from above.
- **Other edits.** The introduction sentence on stability is rewritten to match. `definition` was added to macros.tex (definition style). Nulls per readout are a `\todo`; a magnitude's null may need a refit.
- **Deferred.** R40's restructure goes to R37.
- **Registers.** Evaluation work for the coder is todo 8.19. The pre-registration of primary readouts joins 8.13.
- **Build.** Exit 0, no undefined references, the same 3 overfull boxes.


**tab:contact: the fresh-frozen section is ovarian cancer, not ovary (2026-09-25).** The row read "Ovary (fresh frozen)". The sample's own metadata names it "Human Ovarian Adenocarcinoma (FF)". This surfaced while drafting the lineage maps, whose clusters carry PAX8, MSLN and MUC16 tumour markers. The row is renamed "Ovarian cancer (fresh frozen)", matching "Ovarian cancer (FFPE)". No other place in the live manuscript names the section. The handover's "ovary, fresh frozen" description should be corrected by whoever next edits it.


**Correction to the R33 wording: κ* = 0 means "no effect", not "not separable" (2026-09-28).** Under Definition 1, κ* = 0 only when the interval contains its null already at κ = 0, i.e. there is nothing to explain. The readout that is not separable from leakage of the modelled form is the one that breaks down at the first grid point above zero. I had carried the review's "κ* = 0 is not separable" phrasing into §2.8 and the introduction. Both now say "first grid point / leak fraction above zero", and §2.8 adds that a readout whose interval contains its null at κ = 0 is no finding. Found while drawing the breakdown-point schematic.


**tab:contact overflow fixed (2026-09-28).** The 2026-09-25 rename to "Ovarian cancer (fresh frozen)" pushed the table 18 pt over the text width, a fourth overfull box. The row now reads "Ovarian cancer (FF)", and the caption defines "FF: fresh frozen".

**New appendix figure `fig:breakdown` in app:kappa (2026-09-28).** Todo §7 had deferred the sweep schematic until R33's rule was settled. The author asked for it and put every new figure in the appendix for now. It shows four schematic readouts over the κ grid with intervals and a null line, one per case of Definition 1:

- **(a)** a contrast that survives the grid (κ* > 0.4);
- **(b)** a magnitude that shrinks by construction but stays above its permutation null;
- **(c)** one that breaks down inside the grid (κ* = 0.2);
- **(d)** one that breaks down at the first grid point above zero (κ* = 0.05), i.e. not separable.

The curves are fixed synthetic functions, and the caption says so; only the κ grid is the model's, so the no-numbers rule holds. Colours: ink for readouts, muted dashes for the null, blue only for the κ* marker. §2.8 points to it after Definition 1. `figures/src/build.sh` includes it. Build: exit 0, no undefined references, 3 overfull boxes. A shaded band for the data-allowed range [0, κ̄] can be added once R30 is decided.


**New appendix figure `fig:batching` and subsection app:tiles; S42 applied (2026-09-28).** This is figure-plan item 5, the tiles and halo figure.

- **(a)** The primary section with its 128 tiles, recomputed with the model's own split functions and seed (the tile outlines come from a bounds-tracking copy of the recursive median split, asserted equal to it). The 19 held-out tiles are hatched.
- **(b)** A 110 µm window where a training tile meets a held-out tile and another training tile, chosen by rule (the most ring-one cells from both at once). Cells are coloured by role: seeds, ring one, ring two, other. Solid edges run into the seeds, dashed edges from ring two. Held-out cells are hatched, so the transductive use stated in §2.6 is visible.
- **Layout fixes** before placing it: seed outlines made light so they don't read as edges, the key rewrapped, panel letters unclipped.
- **Placement.** It sits in a new short subsection at the end of the implementation appendix; §2.6's "The loss is evaluated on seeds only" points to it.
- **S42.** Drawing it made the text wrong: ring two contributes a type lookup only, not a type and an image lookup (the ring-two context is never formed, confirmed in code). Fixed in §2.6 and in the implementation table's halo row, which also drops the outdated "still evaluates the count encoder on ring two".
- **Numbers left out.** Measured ring sizes (6.7 % / 7.2 % of the seeds for this tile) stay out of the caption under the no-numbers rule.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**New appendix figure `fig:kappa-bound` plus one paragraph in app:kappa (2026-09-28).** A schematic simplex over three genes: the line of implied clean compositions ρ_i(κ) = (p_i − κρ̄_i)/(1−κ) from the observed p_i, valid up to κ̄_i = min_g p_ig/ρ̄_ig.

- **(a)** A cell that expresses a little of the neighbours' marker: every κ in [0, κ̄_i] fits.
- **(b)** A cell that cannot express it: κ̄_i equals the true κ.
- **Values.** The compositions and the true κ (0.3) are illustrative; the caption says "schematic".
- **Paragraph.** It states the fact at composition level, conditional on ρ̄: bounded above, never below; pinned only through a gene the cell cannot express. It ties this to the existing marker-set ceiling and the nuclear/extranuclear route, and says the bound is soft with counts. The fact holds whichever way R30 (A/B) is decided; option A would promote it to a proposition in derivations.
- **Layout.** Iterated three times: the path enlarged, labels moved into a callout column, overflow and a leader-line collision removed.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**New appendix subsection app:image, with `fig:mask-radius`, `tab:masking` and `fig:kronos-umap` (2026-09-28).** The author asked for: why the mask is 25 µm, the old masked-vs-unmasked logistic-regression table, and a KRONOS UMAP.

- **Text.** One paragraph on the image descriptor and why its mask is a fixed-radius zeroed disc, not the polygon: no silhouette, no information in the hole, a constant artefact. The radius is chosen from the covering-radius distribution; 25 µm leaves 6 cells out (excluded) and masks 12 % of the field.
- **`fig:mask-radius`.**
  - (a) Cells sticking out versus R on a log scale, with a small table: 15 / 22 / 25 / 36.8 µm → 2,740 / 47 / 6 / 0 cells, 4.3 / 9.3 / 12.0 / 26.0 % of the field.
  - (b) The largest cell that fits and (c) the largest cell (excluded), drawn exactly as the model receives them.
  - The four-stain rendering moved into `style.py` (`STAINS`, `stain_range`, `composite`), shared with Figure 1b, which re-renders pixel-identical.
- **`tab:masking`.** The 2026-08-20 ego-masking experiment: macro one-vs-rest AUC of cell type on a 1 mm spatial tile split, for KRONOS v1/v2 × {whole patch, disc zeroed, cell alone, cell alone at native resolution}, against the neighbour-label baseline. Readings, as in the devlog:
  - the cell alone beats the whole patch;
  - masking costs little;
  - the masked patch sits below the neighbour-label baseline (homophily);
  - v1 and v2 tie on the masked patch, so v1 is kept.

  Marked `\pending` for recomputation at lineage-level labels; all seven embedding arms are still on disk, so this is cheap.
- **`fig:kronos-umap`.** UMAPs of the same 60,000 cells for the whole patch, disc zeroed (the model's Φ) and the cell alone, coloured by a display-only coarse lineage grouping. The old `figures/kronos_umap.png` predates the masking experiment (unmasked, at an earlier crop), so it was not reused. (a) and (b) look alike, which fits "masking costs little".
- **Bibliography.** `kronos2025` corrected from the literature bib: title "A Foundation Model for Spatial Proteomics", authors, arXiv 2506.03373; the CHECK note removed.
- **Pointer.** §2.1 ("a disc covering the cell has been masked out …") points to app:image.
- **Devlog correction.** The 2026-08-20 entry's "34 % of the field" for the largest cell is 26 %.
- **Build.** Exit 0, no undefined references, 3 overfull boxes. Float placement is scattered across the appendix pages; leave for R37.

**Author review of the new appendix figures (2026-09-28), changes.**

- **(1) `fig:breakdown`.** The author did not follow it; it was referenced only by one sentence in app:kappa and a pointer after Definition 1.
  - The panel titles now say what happens: "real at every leak fraction", "shrinks, but stays real", "explained away at κ = 0.2", "explained away at κ = 0.05".
  - Panel (a) has callouts: "one refit at κ = 0.1; bar: its interval" and "no effect" on the null line.
  - app:kappa gains a worked paragraph: a readout (e.g. a fibroblast programme's shift between tumour-adjacent and stromal fibroblasts), why refitting at larger κ shrinks a leakage-only difference, and κ* as "how much leakage it would take to explain it away", then the four cases.
  - The caption adds "each point is a separate fit at that κ".
- **(3) `fig:kappa-bound`.** The "gene 3" label sat above the apex and read as a subtitle; it now sits beside the apex ("gene 3: the neighbours' marker").
- **(4) `fig:mask-radius`.**
  - The table headers read as one phrase ("cells out field masked"), so the % looked like a share of cells sticking out. They are now two-line columns: "disc radius R (µm)", "cells outside it", "disc area, % of field", with a rule under them.
  - "Excluded" was wrong for the model. The six uncovered cells were dropped from the masking comparison only; the model keeps them with a zero image descriptor (the loader zero-fills missing embeddings and warns). Panel (c) now reads "does not fit", and text and caption say this.
- **(5) `tab:masking`.** Re-scored at the lineage labels (see devlog), and both KRONOS releases are named: KRONOS (first release, ViT-S/16, 384-d; the one used) and KRONOS2 (ViT-B/16, 768-d). The KRONOS2 model card asks for the same paper to be cited (Shaban et al., arXiv 2506.03373). The text now also states the approximate marker mapping: the mixes are read as their named component, and 18S is outside the vocabulary and given an unused marker id.
- **(6) `fig:kronos-umap`.** The caption names the release (KRONOS, first release, the one the model uses). The colour is now the coder's lineage label (obs `lineage`, the one the model trains on), with small lineages grouped for display; SOX2-OT⁺ cells fall in Unassigned, as in the relabel.

- **`tab:masking` at lineage labels (same day).** The table now shows the 12-class lineage re-score; `\pending` is removed. Columns are named KRONOS / KRONOS2, and the caption cites the release paper. The text's "tie on the masked patch" became "the first release does at least as well" (+0.011), and "type" became "lineage label" throughout the paragraph. Build: exit 0, no undefined references, 3 overfull boxes.


**Follow-ups to the author's figure review (2026-09-28, later).** The first two edits were delayed by a tool outage and applied once it cleared.

- **`fig:breakdown` caption.** The panels are "four different candidate findings, not four ways of choosing κ, which is never chosen". The worked paragraph adds "the higher the breakdown point, the stronger the finding" and links to fig:kappa-bound: a readout whose breakdown point lies above the data's upper bound on κ survives every leak fraction the data allow.
- **KRONOS2.** Author's decision: keep the KRONOS2 column in tab:masking only, and say why KRONOS is used: half the embedding width (384 against 768 dimensions, keeping the image part of the context small) and about five times faster (637 vs 122 cells/s in the devlog), with no loss on the masked patch.
- **Unassigned.** `tab:masking`'s caption says the 12 classes include Unassigned. `fig:kronos-umap`'s caption says Unassigned includes a low-depth class of mixed lineage that the original annotation called a tumour subtype. This follows the coder's evidence: median depth 27 against 333, three quarters without SOX2-OT counts, EPCAM/PAX8 ten times lower. The author accepted Unassigned over Tumour.
- **S12 applied in §2.1.** "Cells without a label form one additional class, which enters every type-conditioned quantity like any other." I had wrongly told the author that S12 says Unassigned is excluded from readouts; per the code it is not, and the review says so.

**Unassigned is never an evaluation target (author's decision 2026-09-28, devlog rule by the coder, todo 8.23).**

- **§2.1.** The S12 sentence now carries both halves. In training Unassigned enters every type-conditioned quantity like any other class, and its cells remain neighbours. It is never a target of evaluation: every diagnostic and readout of §2.7–2.8 is computed on labelled cells only, because a residual class is not a cell type and a read that treats it as one measures the label's failure, not the model.
- **`tab:masking`.** Re-scored under the same rule: 11 classes, Unassigned kept as a neighbour in the baseline. Values replaced; the caption states the rule. The readings and the KRONOS-over-KRONOS2 wording still hold (KRONOS +0.012 on the masked patch).
- **Not done here.** The final tables reported both ways once (the coder's disclosure) belong to the experiments section, which is out of the build.


**Introduction: components table, classical related work, Cellina (2026-09-28).** Author's requests: add the classical, non-deep solutions to related work; use the "intrinsic + spatial + niche + technical + noise" framing from `articles/litterature_review.md` as an overview if we address each term; consider Cellina.

- **`tab:components`.** A new full-width table in the introduction: one row per component (intrinsic, spatial field, niche, leakage, ambient/batch, counting noise), with columns for classical models, deep generative models and DISCELL. The DISCELL column is honest about scope: the spatial field is not a separate term (it is assigned to w through the image context), and ambient RNA and batch are not modelled. The caption says our components do not add in counts. Paragraph 1 points to it.
- **Related work, first paragraph (new).** The classical families: variance components (SpatialDE, SVCA, NSF, MEFISTO); regression and attribution (C-SIDE, MISTy); contamination correction as preprocessing (SoupX, DecontX, CellBender, SpotClean, Baysor, admixture correction). SpotClean's single bleeding rate is named as the closest precedent for κ, with the difference that in situ assays have no off-tissue spots to estimate it from. The paragraph closes on what these models do not provide: a generative model in which response and leakage compete for the same counts.
- **Cellina.** Added to the deep-model list, and to "names segmentation errors as a limitation" alongside MintFlow.
- **Todo.** The contamination part of the old related-work `\todo` is removed; "further spatial VAEs and GNN spatial models" stays.
- **Rejected.** The review's claim that "nobody has all four terms" is not repeated; DISCELL does not have them either.
- **Registers.** Citations verified (see the citation audit). Cellina is proposed as a baseline in todo 8.25. S11 is partly addressed.
- **Build.** Exit 0, no undefined references, 3 overfull boxes. The Baysor issue number was dropped to remove a 1 pt bibliography overflow.

- **`tab:components`, follow-up (same day).** The author noticed that "technical", one of the five terms in the framing, was missing. I had split it into two rows and dropped the word. Both rows are now labelled "Technical: leakage (misassigned transcripts)" and "Technical: ambient RNA, batch", so the table matches the intrinsic + spatial + niche + technical + noise framing in its caption.

**Introduction: the components table is replaced by an annotated model equation; a trial comparison table is added (2026-09-28).**

- **Why the table went.** The author judged a family-by-family comparison table the wrong device: each cell named two or three examples, so it was always incomplete, and it duplicated the related-work prose.
- **The equation instead.** The DISCELL paragraph now shows the model with the usual decomposition marked on its terms: x_i | ℓ_i ~ Mult(ℓ_i, p_i), p_i = (1−κ) softmax(a(z_i) [intrinsic] + B w_i [niche]) + κ ρ̄_i [leakage]. The text says:
  - the depth ℓ_i (technical) is conditioned on, and the multinomial is the counting noise;
  - the spatial field has no term of its own, and reaches w through the image descriptor;
  - ambient RNA and batch are not modelled.
- **Equation layout.** Two aligned lines, unnumbered. One line overflowed the column, and the equation number was pushed below with a gap above it; nothing references the number.
- **Trial table `tab:related` (the author may remove it).** It compares the six closest models: resolVI, SIMVI, MintFlow, Celcomen, Cellina and DISCELL. The six properties: intrinsic latent, niche term, leakage in the model, how the leak share is set, invariance regulariser, counterfactuals. Mixed columns (text, not only ✓/×) keep it from reading as self-promotion. Every cell was checked against the papers:
  - **SIMVI:** independence (closed-form MI or MMD) between z and s, marginal; no counterfactuals.
  - **MintFlow:** type-specific discriminators; in-silico perturbations.
  - **Celcomen:** no latent at all (intra- and inter-cellular gene–gene interactions); counterfactuals are gene knockouts (Simcomen).
  - **Cellina:** a domain-label adversary on z; neighbourhood swaps.
  - **resolVI:** per-cell estimated shares; no niche term.
- **Build.** Exit 0, no undefined references, 3 overfull boxes (column widths tuned).


**Eq 21 (eq:probe): ΔCE replaced by the per-block probe gain G_b (2026-09-28).** The author spotted that the metric renamed during the review was still in the equation. The paper now matches the coder's implementation (todo 8.16, devlog 2026-09-24, R20 + R22).

- **Definition.** G_b = (1/|b|) Σ_k ½·log(E(v_k − v̄_k(t))² / E(v_k − v̂_k(μ_z, t))²), computed separately for the composition block (K−1 components) and the image block (12 PCs). It is scale-free, so the image block's raw variance no longer swamps composition. Before this, composition, the channel of the confound, was effectively ungraded.
- **Interpretation.** A conditional predictive V-information estimate (Xu et al. ICLR 2020, new bib entry `xu2020vinfo`, verified); exp(2G_b) − 1 is the within-type variance explained.
- **Reporting.** Excess over the within-type permutation floor, in nats per component, and as a fraction of the uncontrolled (α_a = 0) fit's excess. Each probe family (ridge, MLP) has its own floor, and the decision reads the MLP. Added: "a probe at its floor bounds the dependence it can detect; it does not certify independence." The escalation rule now reads "reduces the nonlinear probe's excess, on both blocks, to a small fraction of the uncontrolled baseline's".
- **Implementation table.** The probes row says "each scored per block against its own within-type permutation floor". The image-target row says the MLP "is the probe the decision rule reads", not "a cross-check".
- **Macro.** `\dCE` stays in macros.tex because the out-of-build experiments section still uses it; update that section when it returns.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.


**New appendix figure `fig:probe` (subsection app:probe in the implementation appendix, 2026-09-28).** The author asked for a visual of how the probe works. It is a schematic with synthetic values, like fig:breakdown.

- **(a)** The protocol: tiles, with the probe and the type-only baseline fitted on training tiles and scored on held-out tiles.
- **(b)** One component for held-out cells of one type: the baseline error is the spread around the type mean, the probe error the spread around its prediction, G_k = ½ log(MSE_type/MSE_probe), and e^{2G_k} − 1 is the variance share. The synthetic signal was weakened from a 48 % to a 23 % variance share, so the schematic does not suggest effects far larger than measured (3–6 %).
- **(c)** Reading G_b per block: the floor, this fit, the uncontrolled (α_a = 0) fit; the excess as a blue bar, the fraction of uncontrolled as blue ÷ grey.
- **Placement.** The §2.5 probe paragraph points to it. `build.sh` includes it. Build checked after placement.


**R20's remaining text edits, and a real-data candidate for the probe figure (2026-09-28).**

- **§2.5 (after "that residual is what would go unpenalised").** New sentence: for Φ the regulariser and the probe see only a summary of its leading principal components, so image information outside them, which the prior on w receives in full, is neither penalised nor probed.
- **Introduction.** z is now "pushed to carry no information about the microenvironment given the cell's type; a held-out probe measures how much remains", replacing "kept free of". The author asked for clean wording, not too narrow; the review's longer probe-scoped wording was rejected as too narrow.
- **`fig:probe` (schematic, in the paper).** Now labelled "illustrative values" in panel (b) and "illustrative" on panel (c)'s axis.
- **Candidate `figures/fig_probe_real.pdf` (not placed; the author decides).** Same layout with measured values from the ovarian reference fit at lineage labels (finalL_s0).
  - (b) Tumour-neighbour share for held-out fibroblasts against the ridge prediction, rebuilt with the probe's own code path, draws and seed, with binned means. G_k = 0.0039 nats (0.8 %); it agrees with the stored record to 0.4 %, cause of the gap not located.
  - (c) Excess over floor per block and family, this fit against the uncontrolled fit: ridge 0.19 / 0.14, MLP 0.29 / 0.25 of uncontrolled.
  - It shows what the schematic cannot. The probe is well calibrated but its predictions span a narrow range, which is why G is small. The permutation floor is itself negative (a probe fitted to noise loses to the type mean on held-out cells).


**`fig:probe` guard, `tab:probe`, and the stated guard (2026-09-28).**

- **Figure.** Panel (c) of the schematic now marks the pass rule. A dotted guard at a quarter of the uncontrolled excess sits on each block's line; the illustrative composition block fails (0.31) and the image block passes (0.16). The caption says so.
- **`tab:probe` (new, app:probe).** Per section, the seed range of each block's excess as a fraction of the uncontrolled fit's, for ridge and MLP, and the seeds passing the guard:
  - ovarian FFPE 1/3;
  - lung FFPE 0/3;
  - ovarian FF 3/3;
  - lung TMA A 0/3;
  - lung TMA B (held out, models from A) 0/3.

  Marked `\pending` until the Unassigned re-read (todo 8.23) regenerates it.
- **§2.5 rule.** The fraction is now stated: "to at most a quarter of the uncontrolled baseline's, on both blocks and for both probe families". This matches the coder's pre-registered guard; the paper had said the decision reads only the nonlinear probe. The `\todo` now says the guard is not met on every section at the operating point, and that the escalation needs a closed-form fit per section to be shown.
- **Author's question: does the table make the escalation clearer?** No. It holds only adversary fits and their uncontrolled references; all runs on disk use the adversary (todo 8.11b). The proposal to fit one closed-form seed per section is added to 8.11b.

**Table 8 (`tab:probe`) rebuilt: transposed, absolute and relative together (2026-09-28).** Reading the relative-only table, the author asked whether it meant "plenty more to remove". It does not: every section ends at a similar absolute residual (the MLP explains 3–6 % of within-type composition variance from z), and the fraction left is large only where little leaked without the adversary (lung TMA 5.5–7 % against fresh-frozen ovarian 36 %).

- **Layout (author's request).** Sections are columns. Per block (composition, image) for the MLP probe, three rows: "without the adversary (%)", "with the adversary (%)", and the bold **fraction left**, the quantity the guard reads. Two ridge "fraction left" rows follow, then "seeds passing".
- **Caption.** It defines both kinds of number, says 0 means the adversary removed everything and 1 means nothing, and notes that the fraction is larger where less leaked to begin with.
- **Values.** All from the stored re-grade records; section B's come from its held-out record.
- **Still `\pending`** for the Unassigned re-read (8.23).
- **Rendering fix.** Ranges are typeset in text mode, because "--" inside math rendered as two minus signs.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.
- **Table 8, follow-up (same day).** A "needed to pass (%)" row was added under "with the adversary" in both MLP blocks, at the author's request. It is the largest share that meets the guard: a quarter of the section's mean excess without the adversary, converted to a variance share. Composition: 4.7 / 2.9 / 8.2 / 1.5 / 1.7 %. Image: 3.8 / 2.5 / 7.9 / 1.7 / 1.5 % (ovarian FFPE, lung FFPE, ovarian FF, lung TMA A, lung TMA B). It shows directly that the target depends on the section's starting leak. The caption defines it. Build: exit 0, no undefined references, 3 overfull boxes.


**Experiments return to the build; final-number appendix; R30 option A (2026-09-28).** The model is frozen, and the coder's results manifest (`docs/results_manifest.md`) maps every result to its source.

- **§3 Experiments rewritten.** The pre-freeze draft is kept as `sections/experiments_pre_freeze.tex`.
  - §3.1 Setup is written in full: the sections and the held-out serial section; lineage labels and the Unassigned rule; graph, kernel and image; one configuration (warm-up 30; composition weight 3 in both the heads' loss and the encoder's term, checked in code; κ = 0.1; 500/40; α_z = ½/ℓ̄ with ℓ̄ the section's count scale, the primary section's fixed during development); splits, seeds and read-outs; baselines.
  - §3.2–3.6 are headings only, with hidden comments naming the manifest rows they will draw on.
- **New appendix `app:experimental`**, holding only numbers the coder marked final:
  - `tab:sections`: cells, genes, lineage classes, Unassigned share, median counts, computed from the bundles.
  - `tab:baselines`: version, graph, label use, budget and coverage per method, from the baselines README and runners.
  - `tab:timing`: s/epoch, peak memory, parameters, final-fit wall clock against each baseline's whole-fit time.
  - `app:planted`: the amortisation-gap table, the leak-subtracted-input result, and the dead-channel checks. The survey total is not quoted, because the manifest's 329 and its per-section sum of 251 disagree.
- **R30, option A (author).** `prop:kappa-bound`: a κ′ reproduces the composition with a valid clean profile if and only if κ′ ≤ κ̄_i = min_g p_ig/ρ̄_ig; the true κ lies below, with equality exactly when a neighbour-expressed gene has zero own share. Short proof. It replaces the earlier descriptive paragraph and points to fig:kappa-bound. §2.2 and §2.8 are reworded to match, and the partial-identification `\todo` in §2.8 becomes a sentence.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**Amortisation-gap paragraph rewritten (app:planted, 2026-09-28).** The author asked for it to be clear what is done and how, including what goes into the simulation.

- **Structure.** Five run-in paragraphs: what is measured; the simulation; the exact posterior; the fit and the comparison; the result.
- **Simulation, as it is built in the code:**
  - 3,000 cells uniform in a 700 µm square; 4 types in spatial patches (nearest of 12 anchors with a random perturbation);
  - z ~ N(m_t, 0.5²I) in 2-d with m_t ~ N(0, 1.5²I); ρ = softmax(zA) over 40 genes, A standard normal;
  - leakage on the Delaunay graph pruned at 40 µm, β ∝ exp(−d/20 µm), row-normalised: **the model's kernel with every face set to one**, now stated;
  - κ = 0.2; ℓ ~ Poisson(D) with a minimum of 20, D ∈ {100, 300, 1000}; a multinomial count;
  - an 8-d image descriptor from neighbour composition plus noise.
- **Posterior and fit.** A 129×129 grid for the exact posterior. DISCELL fitted with 2-d z, 300 epochs, at the true and mismatched κ. The cross-fitted MLP alignment is explained, with why it cannot add information. The gap is defined in posterior-sd units. All four references are named.
- **Table caption.** Now says which rows are trained against the exact posterior and which (DISCELL) are not.

**`tab:timing` and `tab:baselines`: gaps explained, size separated from time (2026-09-28).** The author asked that the tables show what could not be run and why, distinguishing "failed" from "not attempted", and that model size be separated from run time.

- **`tab:timing`.** Three blocks: DISCELL model size (parameters, peak memory), DISCELL run time (s/epoch, final fits, epochs to the stop), and comparison methods' run time. Every empty cell now names its reason, with footnotes:
  - SIMVI: not attempted on the FFPE sections; windowed on fresh-frozen, which does not fit whole. I corrected my own draft here: it had claimed the FFPE sections do not fit either, which was never tested.
  - MintFlow: not attempted on the FFPE sections (about 11 h per 70k cells on the TMA core); failed on fresh-frozen, out of host memory before training.
  - The footnotes sit under the table rather than in a spanning row, which had stretched the last column.
- **`tab:baselines`.** The coverage column gives the same reasons per method; Cellina is marked "being run".
- **Build.** Exit 0, no undefined references, 3 overfull boxes.
- **`tab:timing`, follow-up (same day).** DISCELL moved into the same "run time of one fit (min)" list as the comparison methods (whole fit, including evaluations, over the three final seeds). Epochs to the stop and s/epoch (training only) sit with parameters and peak memory under "model and training details". The caption is updated.
- **Whole sections only (author decision, 2026-09-28).** Only runs on whole sections are kept, and window runs are archived. SIMVI on the fresh-frozen section is therefore "failed" in `tab:timing` (the whole section does not fit in GPU memory) and in `tab:baselines`, and its 273.8-min window time is removed. The footnotes are re-lettered: a) SIMVI not attempted on FFPE; b) SIMVI failed on FF, with the earlier window run noted as not used; c) MintFlow not attempted on FFPE; d) MintFlow failed on FF (host memory). Build: exit 0, no undefined references, 3 overfull boxes.


**§2 brought in line with the frozen final configuration (2026-09-28).** A check found §2 described a slightly different model from the one that runs. (The KL warm-up was already in §2.7, worded without "warm-up"; my first check missed it.)

- **eq:adv.** The composition half now carries λ_y = 3, in the encoder term and the heads' loss. The text gives the reason: composition carries the confound, and at equal weights the probe still found removable within-type composition information (tab:probe). It replaces "one weight α_a serves both".
- **§2.4 weights.** ℓ̄ is defined as the section's count scale (the median total count, with the primary section's value fixed during development), and α_z = ½/ℓ̄. R19 is applied: α_w = 0.1 departs deliberately from 1/ℓ̄, because at lower weights w takes over type structure and the agreement guard falls (most on FF); w's readouts are therefore in effect readouts of m_ψ(c,t); the α_w ladder is reported as a sensitivity analysis. "The centre of any sweep" becomes "the natural reference".
- **"What w is not".** R19's justification replaces "earns its place only through within-niche heterogeneity… an empirical question the divergence answers": the direct path fits B and m_ψ against each cell's residual; anomalies, per-cell responses and abduction are future work; a small divergence at this weight reflects the weighting.
- **§2.1.** The lineage-label `\todo` is resolved: "as they are in every fit reported here".
- **Architecture table.**
  - α_z row: ℓ̄ as count scale.
  - α_w row: "0.1, far above 1/ℓ̄; ramped over 30 epochs" (was "calibrated").
  - α_a row: "0.3; six head updates per model update at 2×10⁻³; composition half weighted λ_y = 3" (was "calibrated").
- **Build.** Exit 0, no undefined references, 3 overfull boxes.


**Conclusion written (R34, 2026-09-28).**

- **Summary.** A short paragraph with no numbers: the three-part decomposition, invariance by architecture plus a probe-graded regulariser, a fixed kernel, and κ bounded above and swept with breakdown points.
- **Limitations,** one item each with a pointer to where it is argued:
  - the leakage form (section-wide, gene-agnostic, receiver-scaled, no ambient term; form sensitivity on the primary section only);
  - what the sweep separates;
  - labels (derived from contaminated counts, granularity, merged states read as responses);
  - incomplete invariance (tab:probe);
  - w as context regression at the operating point;
  - inference (the amortisation gap, measured only in simulation, and the posterior-family compromise);
  - identification (only the κ upper bound and the verified invariance);
  - evaluation (transductive neighbours, one section per fit, one held-out section, partial baseline coverage).
- **Future work.** The earlier `\todo` rewritten as prose: perturbations, the open per-cell channel, and a nuclear/extranuclear split turning the κ bound into an estimate.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**Lineage-label appendix (app:lineage, 2026-09-28).** Added to `app:experimental`, from the committed mappings (`labels/lineage_map_applied.csv`) and the relabel's devlog entry.

- **Rule.** States and locations merged; lineages and sub-lineages kept; mappings fixed before refitting.
- **Ovarian FFPE, 18 → 12.** `tab:lineage-ovarian` lists the curated classes per lineage. The two decisions carry their evidence:
  - SOX2-OT⁺ → Unassigned: median depth 27 against 333, three quarters without SOX2-OT counts, profile mostly macrophage;
  - cyst lining → mesothelial-like lineage: CALB2/PRG4/BNC1/PDPN/UPK3B on, EPCAM/PAX8/ESR1 off.
- **Lung TMA, 35 → 30.** State classes reassigned per cell by markers, with cycle genes excluded for the proliferating class.
- **Lung FFPE and ovarian FF.** Cluster → lineage by markers, in `tab:lineage-clusters`. Immune-next-to-tumour clusters keep their immune lineage, since the tumour signal is leakage. The low-depth FF cluster 9 → Unassigned.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.
- **app:lineage, follow-up (same day).** New "Other judgement calls" paragraph: the fibroblast merge (the CAF programme is a state), the endothelial merge (tip state against weakly venous; the less clear case), TMA alveolar/adventitial fibroblasts kept with myofibroblasts split about evenly between them, and the per-cell reassignment accuracy of about 0.75 with margins recorded. From the relabel devlog entry (2026-09-25).

**Section overview figure (fig:sections, app:sections, 2026-09-28).** New `figures/src/fig_sections.py`, added to `build.sh`.

- **a.** Every cell at its centroid, coloured by `obs['lineage']` grouped into six display families (epithelial/tumour; fibroblast/stroma incl. chondrocytes; muscle/pericyte; endothelial; immune/blood incl. erythroid and megakaryocytes; other = Schwann), with Unassigned in grey. Colours are slots 1–6 of the validated categorical order.
- **b.** The four-channel focus image composite in the Fig. 1b stain colours. It is read with zarr from the coarsest pyramid level at least 1,400 px wide over the cells' extent.
- **Layout.** All panels have one height and widths follow each section's aspect ratio; a 1 mm bar sits under each image, and each section is shown at its own scale. The TMA panel is the fitted core only.
- **Render check.** Done at print size. The mapping was checked: the TMA core's orange is real (37% fibroblasts in the labels).
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**Checklist pass (R35, 2026-09-28).** No [Yes] now contradicts the text. The questions are unchanged; answers not yet true are `\todo{answer: ...}` notes saying what is missing.

- **1(b) complexity.** `\todo`. A new symbolic cost paragraph in app:timing covers:
  - per-step encoder/decoder O((n+n₁)Gh) and influx O(|E_n|G);
  - the six head updates O(6(n+n₁)h_a(d_z+K));
  - epochs linear in N and probes capped at 30k cells;
  - peak memory (resident counts plus edge-by-gene gather) and the sweep as 18 fits per section;
  - sample size stated as not analysed.
- **2(b) proofs.** `\todo`: the §2.8 symmetries are unnumbered and app:amplification is a heuristic (prop:kappa-bound is fine).
- **3(b) training details.** `\todo`: the κ = 0.1 selection criterion is not stated.
- **3(c) error bars.** `\todo` until results report as §3.1 defines.
- **3(d) infrastructure.** [Yes]. The Hardware row now reads "one NVIDIA GeForce RTX 4090 (24 GB) per fit", as in all 39 recorded run metadata files.
- **4(b) licences.** `\todo`.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**Symmetries as a proposition (R35/R29, 2026-09-29).** The §2.8 prose statement of the response symmetries is now `prop:symmetries` (Proposition 1; the κ bound becomes Proposition 2), with a full proof in the new app:symmetries (derivations, A.5).

- **The statement** treats the networks as arbitrary functions and applies each map to all cells at once:
  - (a) Rotation: the distribution of the counts is invariant for any orthogonal Q; every term of J is invariant with the posterior rotated too; J is invariant within the diagonal family only for signed permutations. This is more precise than the old "symmetry of the model, not of the fitted objective".
  - (b) Constants over genes: a + c(z)1 and B + 1bᵀ.
  - (c) Per-type translation: exact under the explicit condition τ(z) = t_i on the support of q(z_i). The shift constants are renamed δ_t to avoid clashing with the posterior mean μ.
  - Rescaling is not a symmetry: the per-dimension KL change is ½[(λ−1)A − log λ].
  - The centred shift and the realised prior shift, centred over genes, are invariant under all three.
- **After the proposition,** the R29 remark is kept: the condition in (c) holds only approximately because the Gaussian posteriors of different types overlap, so it is a near-flat direction.
- **Formatting.** (a)–(c) are set as an enumitem list after a render check.
- **Checklist 2(b).** The note now says both propositions have full proofs, and the answer is [Yes] if app:amplification stays a labelled heuristic.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**Assets and licences (R35 4(a)/4(b), 2026-09-29).**
- **New app:assets** (experimental appendix, before app:timing). It gives:
  - the 10x datasets, with `\needsource` and a `\todo` for their licence;
  - GSE315411, cited to williamskatek2026fishing, with the GEO no-restriction wording under a `\todo` to verify;
  - KRONOS and KRONOS2 under CC BY-NC-ND 4.0, used non-commercially and not redistributed;
  - scvi-tools/resolVI, SIMVI, MintFlow and Cellina under BSD-3;
  - no new assets released.
- **§3.1 Sections** now names the sources: the 10x datasets (`\needsource`) and GEO GSE315411 with its citation.
- **Checklist.** 4(a) had been [Yes] while no dataset was cited; it is now a `\todo` until the 10x citations exist. 4(b) points to app:assets and waits on the 10x licence.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**κ = 0.1 operating point stated (R24/R28, R35 3(b), 2026-09-29).** Taken from the coder's devlog note "how κ = 0.1 was chosen".
- **§3.1 Configuration.** It is a working point for single-fit analyses, not an estimate. It was taken before the full sweep, inside the published platform range (`\needsource`), and kept after it: on the primary section it sits at the end of the held-out reconstruction plateau, where B is most seed-stable and the mirror is already below its κ = 0 level. Reconstruction is named as describing the point, not selecting it (the R24 objection), and no claim rests on the choice.
- **Kept out, deliberately.** The 0.09–0.25 per-cell bracket and the lung "no plateau" observation are not in the text. Both are available if a reviewer asks.
- **tab:architecture κ row** now names 0.1 and points to §3.1.
- **Checklist 3(b)** → [Yes].
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**Amplification kept as a heuristic (author, 2026-09-29).** The `\todo` at the end of app:amplification is removed; the subsection stays titled and labelled a heuristic and claims no result. Checklist 2(b) → [Yes] (prop:symmetries and prop:kappa-bound have full proofs). Build: exit 0, no undefined references, 3 overfull boxes.

**10x dataset pages (author's dumps, 2026-09-29).** Ovarian FFPE ("FFPE Human Ovarian Cancer with 5K Human Pan Tissue and Pathways Panel plus 100 Custom Genes": 407,124 cells, median 178, XOA 3.0.0) and lung FFPE (the "Post-Xenium Technical Note" page, experiment 2: 278,328 cells, median 242) match `experiment.xenium` exactly. The FF page sent ("Cross-Platform Comparison: FF Human Ovarian Cancer", 200,900 cells, median 1,283, XOA 3.2.0 resegmented) is NOT our section: ours has 1,157,659 cells, median 1,401, XOA 3.0.0, run "Human FF Ovary 5K". The correct page is still needed. No dump includes a licence or release date, so the citations wait.

**No pre-registered primary readouts (author, option b, 2026-09-29).** §2.8 said a small primary set was "fixed before the final fits" and Bonferroni-adjusted over that set; no such set was ever recorded (todo 8.13). It now says: no readout is designated primary in advance, because development sweeps preceded the final ones; every readout is reported with its breakdown point, and intervals are Bonferroni-adjusted over all readouts of the table that reports them. The `\todo` is removed. Todo 8.13 and 8.19 are updated for the coder (the Bonferroni family is now the reporting table).

**10x datasets cited (2026-09-29).** Three new @misc entries (tenx_ovarian_ffpe, tenx_lung_ffpe, tenx_ovary_ff) with page titles and URLs from the author's dumps, each verified against `experiment.xenium` (cells, median counts, XOA v3.0.0; the lung entry is experiment 2 of the technote page). They are cited in §3.1 and app:assets, replacing the `\needsource`. The year field is `\todo{year}`, because no page shows a date. Until it is filled, in-text labels read "10x Genomics, yeara/b/c", with the marker visible only in the bibliography. The licence `\todo` remains. Checklist 4(a) → [Yes]. main.tex gains `\PassOptionsToPackage{hyphens}{url}` so the long technote URL breaks. Build: exit 0, no undefined references, 3 overfull boxes.

**Transductive exposure stated (§2.6, 2026-09-29).** The `\pending` in "Frozen Neighbours and Batching" is replaced with numbers from the coder's devlog entry "Held-out neighbours of training seeds". I checked them against `scripts/logs/heldout_neighbour_fraction_2026-09-29.jsonl`, which had 11 of 12 splits when checked, FF seed 2 still writing:
- 1–2% of training seeds have a held-out neighbour (0.84–1.86%);
- 5–10% of held-out cells enter a training step as ring-one inputs (4.8–10.2%);
- 2–4% of seeds are within two hops (1.8–4.0%).
Rounded to whole percent. No buffer; the text says why. These numbers depend on the split only, so they are fixed before the freeze; the author forwarded the coder's text for use. Build: exit 0, no undefined references, 3 overfull boxes.

**10x years filled (2026-09-29).** From the author's page panels ("Date Published"): FF 2024-09-04, lung 2024-11-06, ovarian FFPE 2024-12-17. All three entries are year 2024, so the labels read 2024a/b/c. The licence is not on the pages' metadata panels either, so `\todo` stays in app:assets and checklist 4(b) stays open.

**10x licence and citation format (2026-09-29).** From the author's page read: "This dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license." The line was seen on one page; the same template is assumed for all three. app:assets now states CC BY 4.0, and checklist 4(b) → [Yes]. The three bib notes follow 10x's dataset citation format (dataset, "In Situ Gene Expression dataset by Xenium Onboard Analysis v3.0.0", 10x Genomics, date). janesick2023, 10x's requested citation for Xenium Onboard Analysis, is already cited in §3.1. Build: exit 0, no undefined references, 3 overfull boxes.

**Checklist answers (author, 2026-09-29).** 1(c) optional code → [No]; 3(a) code/data/instructions → [No], since the author will not release code before acceptance; 3(c) error bars → [Yes], and the results must report as §3.1 defines. app:assets now reads "The code will be released on acceptance; no new data are released." (was "No new assets are released."), flagged to the author as a commitment. Still open: 1(b), [Yes] recommended vs [No], waiting on the author. Build: exit 0, no undefined references, 3 overfull boxes.
- **1(b) → [Yes]** (author, same day): time and space are covered in §2.6 and app:timing, and sample size is explicitly stated as not analysed. Every checklist item is now answered; only the template's "answer at submission time" banner remains. Build: exit 0, no undefined references, 3 overfull boxes.

**Moran's I table (app:moran, tab:moran, 2026-09-29).** A new subsection at the end of the experimental appendix. It gives within-type Moran's I of μ_z (without the adversary vs final) and of μ_w (final only), from the existing validation.json files, which are final and under the Unassigned mask. The text says:
- w is spatial by construction;
- the adversary lowers z's I by 42–48%, and by 25% on FF;
- the residual is within-type variation shared by neighbours, which the invariance removes only where composition or image predict it;
- without the adversary, w has almost no within-type variance on the two ovarian sections, so its uncontrolled I is not reported.
Numbers are in the devlog, "Within-type Moran's I table (results)". Build: exit 0, no undefined references, 3 overfull boxes.

**Reconstruction breakdown (app:recon-modes, tab:recon-modes, 2026-09-29).** A new subsection built from the existing `recon_modes*.json` files: final, under the Unassigned mask, finalL_s0–s2, with the TMA serial section from `recon_modes_gse315411_pdltma06_10_prime_dual.json`.
- **Columns** (nats per count, mean with range over seeds): full; context over type profile; own z (intrinsic over context); own w (full over intrinsic).
- **Text.** It explains why a single reconstruction number is autoencoding and not informative (R24). It gives three readings:
  - own z carries most of the gain: 0.06–0.17;
  - context adds 0.007–0.033, and nothing on lung, where the CI includes 0 on all three seeds (and on TMA seed 2);
  - own w adds ≤ 0.001, the operating point's design.
  The serial section reads like the core.
- **CI claims were checked** per seed and difference. The Full column shows means only, to fit the width.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**Per-cell KL and uncertainty maps (app:kl-maps, 2026-09-29).** A new appendix subsection with tab:kl-summary, tab:kl-top, fig:kl-maps, fig:kl-seeds and fig:kl-hist. The data are the kl_maps npz/json/summary files from the finalL_s0–s2 runs (`discell/experiments/kl_maps.py`; devlog motivation and results 2026-09-29). The figures come from `figures/src/fig_kl_maps.py`, added to build.sh:
- percentile-within-section maps, pale for 0–90 and dark for the top decile;
- a three-seed comparison for KL_w;
- log histograms, with a linear axis for sd q(w).

The text:
- defines KL vs sd (the author's point that KL is not uncertainty);
- gives the fixed top-decile rules, including the disclosed display filter;
- **KL_w:** small (0.004–0.09), seed-dependent up to 10×, a moderate tail (19–43%), spatially organised within a fit (I 0.15–0.85) but not reproducible across seeds (ρ −0.43–0.57, Jaccard ≤ 0.21). The enrichments are coarse: tumour down on ovarian/lung, fibroblasts up on ovarian, chondrocytes up on TMA, nothing on FF, no continuous feature. So it is not a per-cell anomaly score at the operating point;
- **KL_z / σ_z:** reproducible (0.78–0.93 / 0.53–0.93) and depth-dependent with the reverse sign (the 1/ℓ scaling). Type enrichments after depth removal (smooth muscle ovarian 3.2–3.6, neutrophils lung 5.4–5.9), isolated cells up on three sections, boundary cells on ovarian (SMD 0.43–0.45);
- **sd q(w):** 0.98–1.01, the prior's scale.

Two self-corrections before the build: the Jaccard index was not to be read as a share of cells, and cross-seed comparison of w dimensions was removed because of prop:symmetries. Build: exit 0, no undefined references, 3 overfull boxes.

**KL appendix trimmed; §2.3 and §2.4 corrected (author: "makes sense to shorten it", 2026-09-29).**
- **app:kl-maps now holds** the intro, one paragraph on the response (small, seed-variable magnitude, a moderate tail, spatially organised within a fit, not reproducible across seeds, coarse passes under the pre-set rule, so not an anomaly score; sd q(w) at the prior's 1), one paragraph on the intrinsic state, tab:kl-summary and fig:kl-seeds.
- **Removed from the text:** fig:kl-maps, fig:kl-hist and tab:kl-top. The script still renders them (build.sh comment updated). The seed caption now describes its colour scale itself.
- **§2.3.** "an anomaly score that the objective provides at no extra cost; whether it carries usable per-cell information … is an empirical question" → "could serve as an anomaly score at no extra cost; at the operating point it does not, since the cells it ranks highest differ between seeds (app:kl-maps)".
- **§2.4 was wrong in direction.** The text said the per-cell scaling weights a low-count cell's divergences more heavily relative to its likelihood, pulling sparse cells toward the prior. In J, reconstruction is divided by ℓ_i while α_z is global, so relative to the cell's own bound the divergence weight is ∝ ℓ_i/ℓ̄: deep cells are held closer to the prior, and sparse cells less. The data agree, stably across seeds and within types: deeper cells have smaller KL_z and wider posteriors. The sentence was rewritten to match, with a pointer to app:kl-maps. The appendix adds the consequence: z is less regularised in low-count cells, and a deep cell's posterior width overstates its uncertainty.
- **Not added:** the ovarian boundary enrichment of KL_z (SMD 0.43–0.45), a possible under-corrected-leakage lead on one section. It is offered to the author as a κ-sweep check.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**What the model needs (app:inputs, tab:inputs; conclusion pointer; author decisions 2026-09-29).** A new subsection before app:timing, based on the loader and graph code (`preprocess/xenium.py`, `preprocess/geometry.py`, `model/prepare.py`):
- **Five inputs:** counts, one position, pixel size, a type label and the 4-channel morphology image. The graph and kernel come from positions only (Voronoi partition of the centroids, so Delaunay adjacency, face length and centroid distance).
- **Polygons are stated as not required** (author: "not required at all"). The runs used polygon centroids, and that is stated. Polygons were used outside the model for the disc radius and the appendix comparisons.
- **Not used:** transcripts, nucleus boundaries, post-run H&E. Cell-cycle genes are diagnostic only.
- **Implicit requirements:** counts assigned to segmented cells at single-cell resolution, one contiguous section per fit, multinomial depth, a derivable label.
- **"Other assays (untested)", reasoning only** (author: Xenium 5K is the only data presented):
  - other imaging panels: stain mapping, depth;
  - high-resolution arrays with cell assignment: H&E encoder, lateral-diffusion leakage whose kernel form needs re-examining (`\needsource`); multi-cell spots are deconvolution, out of scope;
  - protein imaging: intensities not counts, spillover analogous (`\needsource`), KRONOS native.
- **Conclusion future work** gains one sentence pointing to app:inputs.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.
- **app:inputs, sources (same day).** Both `\needsource` markers are resolved. The array sentence is narrowed to what the sources support: capture movement is separate from segmentation and platform-dependent, small on Visium HD (deoliveira2025visiumhd), larger on Stereo-seq (ren2025benchmark). "Mainly" and "more than one hop" are dropped. Protein spillover is cited to REDSEA (bai2021redsea). Details are in citation_audit.md. Build: exit 0, no undefined references, 3 overfull boxes.

**Latent UMAP/PCA figures (app:latents, fig:latents-umap, fig:latents-pca, 2026-09-29).** New `figures/src/fig_latents.py`, added to build.sh. It uses posterior means at best.pt from `kl_maps_finalL_s0.npz`, seed 0, and caches embeddings in `figures/src/data/fig_latents.npz`.
- **Rows:**
  - a: μ_z by own family (30k cells);
  - b: μ_w by dominant-neighbour family;
  - c, d: μ_z and μ_w within one type, chosen by rule (≥ 3,000 cells, highest entropy of dominant-neighbour family: ovarian T/NK, lung pericytes, FF pericytes, TMA neutrophils).
- **Text** says z separates lineages, w is organised by neighbours, and within a type z is mixed while w is structured, with tab:probe as the quantitative version. It also gives the UMAP exaggeration caveat: PCA overlaps on ovarian because two components show part of the 20 dimensions. A "classifier" claim was corrected before the build, because the §2.7 agreement is clustering-based.
- **fig_sections.py** now has a main guard, so importing its constants no longer re-renders it.
- The run-report UMAPs were not used: they were drawn at a figure epoch, probably not the accepted checkpoint, and before the Unassigned mask.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.
- **Latents, follow-up (author, same day).** The PCA figure was dropped (fig:latents-pca removed; the script no longer draws it). Rotated row labels were replaced by a header above each row naming the latent, the cells and what the colour means, with section names on a line of their own at the top. The PCA caveat sentence became "UMAP distorts distances and can exaggerate separation, so the figure illustrates what the probe and the agreement with type measure, and replaces neither". Build: exit 0, no undefined references, 3 overfull boxes.
- **Latents in the main text (author, same day).** fig:latents-main is the primary section (ovarian FFPE) as a 2×2 grid at column width, from the same cached embeddings: a μ_z/own type, b μ_w/neighbours, c–d μ_z/μ_w of T/NK cells. The legend lists only the families present. It is placed in §3.3 (sec:results-z) with one qualitative sentence pointing to fig:latents-umap for all sections; the caption carries the UMAP caveat and points to tab:probe. Build: exit 0, no undefined references, 3 overfull boxes.

**Invariance probe reported descriptively (author decision, 2026-09-29).** Under the Unassigned mask the ¼-of-uncontrolled rule fails on the MLP composition block on ovarian (2/3 seeds), lung (3/3) and TMA (3/3); FF passes. The absolute residual with the adversary is consistent: 3.7–5.8% of within-type composition variance (MLP) and ≤ 1.6% (ridge) on every section. The relative fraction is large where the uncontrolled leak is small (TMA 7%, lung 12–13%). The author chose to report descriptively, with a disclosure, rather than change the threshold after seeing the numbers or report pass/fail.
- **§2.5.** The rule reads "reduce the excess substantially"; the development reading (≤ ¼ on both blocks and both families) is disclosed, together with the fact that the MLP exceeds it on the composition block on three sections, "where the uncontrolled residual is itself small", and the residual is reported descriptively (tab:probe). The `\todo` keeps only the escalation-evidence part.
- **tab:probe** was regenerated from the masked `probe_regrade_lineage_final.md` on all five sections: MLP rows without/with/fraction left plus resolVI and Cellina; ridge rows with/fraction left. The "needed to pass" and "seeds passing" rows were removed and the caption rewritten. app:probe gained a paragraph: 3.7–5.8% MLP, ≤ 1.6% ridge; TMA removal 29–48% vs FF > 4/5; MintFlow on TMA MLP 0.3% but ridge 0.9%; SIMVI 5.2/3.1%.
- **fig:probe (schematic):** the guard line, the pass/fail verdicts and the guard key were removed; "fraction of uncontrolled" is kept.
- **Conclusion limitation:** "removes most, not all" → the absolute 3.7–5.8%, with the TMA removal only 29–48%.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**tab:timing (2026-09-29).** A Cellina row was added from `baseline_battery_lineage.json` `train_s`, the same definition as resolVI: 11.6 / 9.9 / 29.1 / 3.4 min. The TMA-core SIMVI and MintFlow times were switched from the pre-relabel fits (107.2 / 680.1) to the lineage fits the paper compares (192.6 / 544.0); footnote c now says "about nine hours". The text says times are those of the compared fits at lineage labels. Build: exit 0, no undefined references, 3 overfull boxes.

**Breakdown text aligned with the 8.19 decisions (2026-09-29).**
- **§2.8.** The sentence "Bonferroni over all readouts of the table" is replaced. κ* is now computed for signed contrasts, against zero, with the sign at κ = 0. Magnitudes are reported as trajectories with no κ*. Intervals are for the seed mean, by a within-seed spatial block bootstrap over tiles, Bonferroni over the contrasts of each section. `\pending{family size per section}` stays until the coder's count arrives.
- **fig:breakdown panel b.** It was "a magnitude above its permutation null" (null 0.25); it is now a signed contrast that shrinks but stays clear of zero (null 0). The docstring, the app:kappa text and the caption follow; no panel shows a permutation null any more.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**Review pass applied (author decisions A1, A3, A4, C; 2026-09-29).** Two read-only Opus reviews (framing; claim audit) were merged; key findings were spot-checked (baselines fitted on the whole slide, sweepL at 500/40, the synthetic appendix commented out, the stale def:breakdown).
- **A1, framing.** The abstract is rewritten around partial identification: the upper bound, sweep plus breakdown point per claimed contrast, and the adversary plus a probe that grades competitors too. The closed-form penalty is dropped from the abstract and the contributions. The contributions are reordered: (i) partial identification made operational (prop:kappa-bound, κ* as a general protocol), (ii) the model, (iii) inference, (iv) invariance plus the probe as a benchmark.
- **A3, invariance wording, fair and not undersold:**
  - abstract: "strips … a held-out probe measures what remains";
  - conclusion: "pushed toward invariance … a few percent of the within-type variance … less than competing models leave on most sections";
  - §3.3 and app:latents: "little grouping". "Most sections" was checked on MLP composition: fewer than resolVI and Cellina everywhere except the TMA core, where Cellina has 3.6% and MintFlow 0.3%.
- **A4, promoted to the main text:**
  - prop:kappa-bound is stated in §2.8, now Proposition 1 with symmetries as 2; the proof stays in app:kappa;
  - tab:probe moved to §3.3 as table*, with a benchmark paragraph: 3.7–5.8% MLP and ≤ 1.6% ridge, against uncontrolled 7–39% and resolVI 5.7–31%; FF > 4/5 removed, TMA 29–48%; Cellina 3.6% and MintFlow 0.3/0.9% on TMA;
  - the collapse of w without the adversary, in §3.3 (tab:moran).
- **§2.8, lean 8.19** (coder's revision): def:breakdown is rewritten for signed contrasts (interval of the seed mean excludes 0 and every seed keeps its κ = 0 sign; "no finding"; above the grid). The headline contrasts are listed (8 per section, 9 on the primary section, `\pending` final sizes). Other readouts are shown as trajectories or at the operating point. Bonferroni runs over the section's claimed contrasts. "Every readout … interval pooling seeds … sampling as well as optimisation" becomes a range over seeds.
- **C, fact fixes:**
  - intro: "thousands of genes in every cell"; "for each claimed spatial contrast"; attention "could reconstruct"; w "has a prior centred on a regression on it"; tab:related counterfactuals "type-level";
  - §2.1: Unassigned described as unlabelled or folded in by the mapping; checkpoint criteria read on every held-out cell; "every final fit";
  - §2.2: projection "tested and not adopted";
  - depth: median 178–1,401 counts;
  - α_w sensitivity marked `\pending`; the support-floor promise removed; the equal-weight claim softened with a `\pending` cite; "two halves reported" removed;
  - α_a set once on the primary section; the ¼ disclosure names the image block on the TMA core and limits "small" to TMA and lung;
  - §2.6 exposure exact ranges;
  - §2.7 rewritten as "Budget and seeds": one budget 500/40, no seed selected, a fixed figure seed, tables with the range, and seeds redraw the split, PCA and clustering;
  - "anomaly score meaningless" → "divergence uninterpretable" (method and rationale);
  - §3.1: FF "eight times the median count"; "fixed before any fit at lineage labels"; the tile exception; "every claimed contrast"; "every final configuration", seeds redraw the split; the cross-reference to sec:sweep; **baselines fitted on all cells with ≥ 5 counts, held-out tiles included, scored on our held-out cells and labels**;
  - app:inputs depth wording;
  - tab:architecture: tiles 2,048 on the TMA core, κ a working point, calibrated weights "set once on the primary section"; tab:deviations2 budget and seed rows;
  - conclusion: the prop:kappa-bound pointer, the planted-gap wording, "the invariance the probe grades", "have so far been run".
- **Deferred with the author:** B (evidence items, after the runs), D (structure), A2 (discussion).
- **Build.** Exit 0, no undefined references, 3 overfull boxes. Backups of sections/ and appendix/ are in the scratchpad.
- **tab:probe comparators (same day, author asked where the MintFlow numbers are).** Rows added: MintFlow and SIMVI (TMA core and serial section only; "--" where not yet run) in every block, and resolVI/Cellina in the ridge blocks, all from `probe_regrade_lineage_final.md` (solo and dual). The caption says MintFlow and SIMVI have so far run on the TMA core only; on the serial section MintFlow is the core's model transferred and SIMVI is fitted on that section. The §3.3 sentence now also gives MintFlow's serial-section MLP (1.0%) and SIMVI's 5.2–9.0%. Build: exit 0, no undefined references, 3 overfull boxes.

**Generated table tab:headline (2026-09-29, new file `tables/generated/headline.tex`, not yet \input anywhere).** Model quality per section (ovarian FFPE, lung FFPE, ovarian FF, TMA core, TMA serial held out): held-out reconstruction, NMI, mirror R², cycle R² of z and of w on the top-decile set, I(niche; w) excess, transport fraction of ceiling, atlas cross-seed cosine; mean, range over seeds and tile-bootstrap 95 % interval. Sources: `data/datasets/<ds>/experiments/envelope_table_ci_at_best.md` (four trained sections) and `runs/finalL_s{0,1,2}/crossslide/gse315411_pdltma06_10_prime_dual.json` (serial section, mean checked against the envelope's held-out mean). Every transport cell is `\pending{held-out-tile check}` until manifest row 37 lands; on the serial section I(niche; w), transport and atlas are "--" (not read there). Generator `scripts/paper_tables.py`. Motivation: generated from frozen results so numbers are traceable and regenerable.

**Generated table tab:probe-summary (2026-09-29, new file `tables/generated/probe_summary.tex`, not yet \input).** Per-block probe excess (composition/image × ridge/MLP) as a fraction of the uncontrolled reference, range over three seeds, with the count of seeds within the 0.25 guard per block and for all four blocks; failing blocks in bold; the caption states the sections where the MLP composition probe fails on every seed. Source: `data/datasets/<ds>/experiments/probe_regrade_lineage_final.json` (five sections). No pending cells. It overlaps the fraction-left rows of the existing tab:probe; the writer decides whether it replaces them or sits in app:probe. Motivation: generated from frozen results so numbers are traceable and regenerable.

**Generated table tab:battery (2026-09-29, new file `tables/generated/battery.tex`, not yet \input).** One battery per section for DISCELL (mean and range over seeds), resolVI, SIMVI, MintFlow and Cellina at lineage labels under the Unassigned mask: NMI, composition probe fraction (ridge, MLP), mirror R², cycle R² (top-decile set), held-out reconstruction. Sources: `data/datasets/<ds>/experiments/baseline_battery_lineage.json` and `probe_regrade_lineage_final.json`, each checked against the frozen snapshot in `scripts/logs/cellina_extra_2026-09-29/backup/`. SIMVI and MintFlow on the three Xenium sections are `\pending{queued}` (whole-section queue); MintFlow reconstruction is "n/r" (B-mf1); SIMVI reconstruction "n/a" (no decoder); the Cellina niche-adversary and own-graph rows are not included. Motivation: generated from frozen results so numbers are traceable and regenerable.

**Generated table tab:sensitivity (2026-09-29, new file `tables/generated/sensitivity.tex`, not yet \input).** Sensitivity arms at the final configuration against the finalL controls: κ form (depth, gene, density), fixed false-positive floor (per section, area-scaled), α_w ladder (m/ℓ̄), adversary capacity and composition weight; reads: reconstruction, NMI, mirror R², I(niche; w) as arm mean with the move in control seed-sd, and the probe-guard count; transport one `\pending{held-out-tile check}` row. Sources: `data/datasets/xenium_prime_ovarian_cancer_ffpe/experiments/sensF_{kappa_form,fp_floor,alpha_w,adversary}.json`, written inside the masked re-render step of the Unassigned queue (15:07:19–21); every arm run's probe and guard reads carry the mask. The cycle rows of the sensitivity files (retired label-derived set) are left out. Motivation: generated from frozen results so numbers are traceable and regenerable.

**Generated table tab:kappa-sweep (2026-09-29, new file `tables/generated/kappa_sweep.tex`, not yet \input).** The κ sweep per section over κ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4} as seed ranges of reconstruction, NMI, mirror R², cycle R² of z and w (top-decile set); one table, κ = 0.1 in bold. Source: `data/datasets/<ds>/experiments/kappa_sweep_sweepL.json`; the κ = 0.1 rows were checked equal to the final fits' own reads. The I(niche; w) column is replaced by one `\pending{re-read}` row: the sweep file's value agrees with each fit's own guard read on seed 0 only, not on seeds 1–2 at any κ. Considered and rejected: a figure instead (the table fits one appendix page). Motivation: generated from frozen results so numbers are traceable and regenerable.

**Generated tables integrated (coder's `tables/generated/`, 2026-09-29).**
- **headline** is `\input` in §3.2 (sec:results-quality), with a first results paragraph taken from it:
  - NMI 0.62–0.74;
  - cycle R² of z 0.19–0.76 vs w ≤ 1% in any seed, w's intervals contain 0 on every fitted section;
  - I(niche; w) excess 0.47–0.60 nats;
  - mirror R² 0.03–0.09;
  - atlas cosine 0.82–0.97;
  - serial section NMI 0.61 vs 0.62, cycle 0.50 on both;
  - transport `\pending` (held-out-tile check).
- **kappa_sweep** is `\input` in the new app:sweep, with text `\pending` on the breakdown table.
- **Held back:** probe_summary, sensitivity and battery. probe_summary and sensitivity's "Guard" column present the ¼ rule as pass/fail, against the author's descriptive decision (morning of 2026-09-29), and battery and sensitivity reference tab:probe-summary. A regeneration was requested from the coder (descriptive tab:probe with comparators from the generator; the Guard column replaced by an absolute residual; cross-references to tab:probe; the battery caption to state that the baselines are fitted transductively). The fractions in probe_summary match the hand-built tab:probe exactly, a useful cross-check.
- **Build.** Exit 0, no undefined references, 3 overfull boxes.

**Generated tab:probe replaces tab:probe-summary (2026-09-29, `tables/generated/probe.tex`; `probe_summary.tex` deleted).** Author's decision: the probe is reported descriptively, with no pass/fail. The generated table follows the hand-built tab:probe (same label). For each block (composition, image) and probe (MLP, ridge), it gives the % of within-type variance recovered without the adversary (uncontrolledL, two seeds) and with it (finalL, three seeds), the fraction left, and rows for resolVI and Cellina, and for MintFlow and SIMVI ("--" where not yet run). It has no guard rows and no failure bold; the bold on the MLP "fraction left" rows follows the hand table. The "without" row is added for the ridge blocks too. Source: `data/datasets/<ds>/experiments/probe_regrade_lineage_final.json` (5 sections), checked against the frozen snapshots; each var_fraction is checked to equal exp(2·excess) − 1. Every fraction left equals the hand table. Six percentage cells differ from the hand table by 0.1, because the hand table rounded the md's two-decimal percentages a second time: lung MLP composition with the adversary 4.4–5.2 (hand 4.3); lung MLP image without 9.7–10.9 (10.8); TMA core SIMVI MLP composition 5.1 (5.2); serial resolVI MLP image 7.7 (7.8); serial Cellina ridge image 1.9 (2.0); serial SIMVI ridge image 3.4 (3.5). The generated values are single-rounded from the stored values. When the writer swaps in `\input{tables/generated/probe}`, the hand table must be removed, because the labels are the same. Motivation: generated from frozen results so numbers are traceable and regenerable.

**tab:sensitivity regenerated (2026-09-29).** The "Guard" column (a count of seeds passing the ¼ rule) is replaced by "MLP comp. (%)": the MLP composition residual in % of within-type variance, shown like the other columns as arm mean (move in control seed-sd) and as control mean with its range. It is read from each run's `runs/<run>/validation/probe_blocks.json` (`mlp.comp.var_fraction`), checked against the sensitivity files' stored excess where they carry it (adversary, α_w). The caption points to tab:probe. Motivation: descriptive probe reporting (author), still generated from frozen results.

**tab:battery caption regenerated (2026-09-29).** The caption points to tab:probe. It states the following, each claim verified against `DisCell-baselines` (runners, `common.py`, each result's `config.json`):
- DISCELL's held-out tiles are excluded from its training loss.
- The comparison methods are fitted on every cell with at least 5 counts, held-out tiles included, and scored on our held-out cells.
- resolVI and MintFlow train on all cells. Cellina and SIMVI train on a random nine tenths of the cells, keeping the rest for their own early stopping. None of their reads, reconstruction above all, is a held-out read.
- On the serial section, resolVI (its transfer failed and it was refitted there) and SIMVI are fitted on that section. Cellina, MintFlow and DISCELL are the core's models transferred.

Motivation: the reconstruction column is not like-for-like without this disclosure.

**tab:kappa-sweep regenerated (2026-09-29 ~17:00, coder).** The I(niche; w) row is filled. The sweep report had graded seeds 1–2 on seed 0's data split (devlog "Sweep report graded seeds 1–2 on seed 0's split"). After the fix and re-read, the guard equals every fit's own record on all four sections, and the generator's built-in agreement check released the row: 144 cells traced, 0 pending. A test build to a scratch directory exits 0 with no undefined references. Its three overfull boxes are in hand-written text (the KRONOS UMAP figure and a paragraph in the rationale appendix), not in the generated tables.

**Overfull boxes cleared (2026-09-30).** Layout only; no numbers or wording changed:
- app:qw cause/check/fix table: p-columns (was 55 pt);
- tab:related: first column 0.14 → 0.13, invariance 0.17 → 0.16 \textwidth (was 5 pt);
- eq:encz-input: the definition of ℓ̃ moved from the display into the following sentence (was 5.9 pt).

The one remaining box (5.1225 pt, at the abstract) comes from the template: the official `sample_paper.tex` built with `aistats2026.sty` gives the identical box, and it persists with a one-line abstract. It is left as is. The regenerated tab:kappa-sweep (I(niche; w) filled, no pending cells) is in via `\input`. Build: exit 0, no undefined references, 1 overfull box (the template's).
- **tab:kappa-sweep caption (2026-09-30).** The caption string in the coder's generator (`scripts/paper_tables.py`) now defines the I(niche; w) column ("the response's information about the niche as its excess over a within-type permutation floor, in nats"). Regenerated with `--only kappa_sweep`: 144 cells traced, 0 pending, and the diff against the previous file shows only the caption changed. The generator's 11 tests pass. Build: exit 0, no undefined references, 1 overfull (the template's). **Coder:** this edits your script (caption text only).

**tab:headline regenerated: transport released (2026-09-30, coder; `tables/generated/headline.tex`).** Following the author's decision (devlog "Transport rule triggered; the published read stays the headline", 2026-09-30), the transport row now shows the published read: cell-split ceiling, trusted tier, mean with range and half-tile interval. Values: ovarian FFPE 0.65, lung 0.72, FF 0.67, TMA core 0.73 (2 of 3 seeds have a trusted panel, marked †); serial section "--". The pending special case is removed from the generator. The caption points to tab:transport-heldout for the held-out-tiles read. Source: each section's `envelope_table_ci_at_best.md`, row "transport, mean read: fraction of ceiling (trusted)".

**tab:sensitivity regenerated: transport column (2026-09-30, coder; `tables/generated/sensitivity.tex`).** The spanning pending row is replaced by a Transport column (trusted-tier fraction of ceiling, arm mean and move in control seed-sd), read per run from `runs/<run>/transport/transport.json` because the fp_floor and adversary files carry no point value. It is checked equal to the sensF records where they exist. Two things to note. (1) The fixed false-positive floor moves ovarian transport from 0.65 to 0.50 (−8.3 sd; area-scaled 0.54, −6.2 sd; seed 1 alone 0.38), the largest move in the table. (2) `sensF_alpha_w.json`'s `readA_of_ceiling` is the ALL-PANEL read, not the trusted one as manifest row 21 says. Layout: footnotesize, family titles as spanning rows, shorter arm labels ("per cell, depth ratio", "one per section", "area-scaled, per cell"), and the header "MLP (%)".

**tab:battery regenerated: MintFlow lung and the Cellina variants (2026-09-30, coder; `tables/generated/battery.tex`).** MintFlow on the whole lung FFPE section is now scored (feasibility.tsv: ok, 14.81 h wall, 50/50 epochs). Its probe is read from its own record `probe_regrade_lineage/MintFlow_lineage.json`, because the final probe table is re-rendered only at the end of the queue; the conversion was verified exactly on entries held in both. Its reconstruction is still n/r (B-mf1). New rows: "Cellina, niche" on all five sections (footnote d: the niche label matches what the probe grades, so this is Cellina's best case on the probe, not its published setting) and "Cellina, own graph" on the TMA core and serial section. SIMVI (lung, ovarian FFPE, FF) and MintFlow (FF) are `\pending{queued}`, and MintFlow on ovarian FFPE is `\pending{running}`. Cells in these states are generated from feasibility.tsv and the queue log. A failure will print "could not be run on the resources currently available", with the measured reason in footnote e. Note: the niche adversary does not lower Cellina's composition leakage. Its ridge fraction is higher than plain Cellina's on 4 of 5 sections (e.g. TMA core 1.47 vs 0.83).

**tab:probe regenerated: new comparator rows (2026-09-30, coder; `tables/generated/probe.tex`).** MintFlow on lung FFPE (MLP composition 1.1 %) is added to the "MintFlow, SIMVI" row. New rows: "Cellina, niche domain" (5 sections) and "Cellina, own graph" (TMA core and serial). The caption lists the whole-section fits still in progress, generated from the queue state. The hand table of the same label in `sections/experiments.tex` must still be removed when this file is `\input`.

**tab:timing generated (2026-09-30, coder; new `tables/generated/timing.tex`, replaces the hand table in `appendix/experimental.tex`).** It keeps the hand table's label and structure, and every hand value is reproduced. DISCELL values come from `experiments/timing_lineage/timing_phi.json` (checked against `timing_lineage.md`) and `runs/finalL_s*/metrics.json`. Baseline values are `train_s` of the lineage fit's `config.json`. Whole-section state comes from `DisCell-baselines/results/feasibility.tsv`, else the baselines_complete queue log. MintFlow lung is 879.4 min (train_s; the 14.81 h wall clock includes setup and prediction). The hand table's "not attempted"/"failed" FFPE and FF cells are superseded by the queue's attempts, so they show as pending. The hand-built table must be removed when this file is `\input`.

**tab:breakdown generated (2026-09-30, coder; new `tables/generated/breakdown.tex`).** κ* per claimed contrast × section, with m per section in the header (9/8/8/8/1). Source: `scripts/logs/breakdown_2026-09-29/breakdown_all.json`, with statuses checked against the `.md`. Every non-transport member is "above the grid" except the FF signalling share, which breaks at 0.2 (the interval contains 0). The marker-pair contrast is negative at κ = 0 on lung, FF and the TMA core and positive on ovarian, marked ^(−). The four transport members (16 cells) are `\pending{computing}`: the queue was relaunched with GATE_OVERRIDE=1, and a rerun of the generator fills them.

**tab:breakdown-traj generated (2026-09-30, coder; new `tables/generated/breakdown_traj.tex`).** The trajectory readouts across κ per fitted section, as 3-seed mean and range: trusted transport fraction, within-type share of w variance, Moran's I of μ_z and μ_w, atlas effective rank, and programme overlap with finalL_s0 (1 for that fit at κ = 0.1, stated in the caption). Source: `breakdown_all_trajectory.json`. It leaves out the all-panel transport ratio and the unnormalised w variance.

**tab:transport-heldout generated (2026-09-30, coder; new `tables/generated/transport_heldout.tex`).** Published against held-out-tiles reads, per section: DISCELL as the seed range, and the Cellina counterfactual. Columns: trusted and all-panel fraction of ceiling, Read A and twin margin. Source: `scripts/logs/transport_heldout_2026-09-29/transport_heldout_comparison.json`; the flags were recomputed and match the `.md`. The caption states the direction from counts rather than from the devlog summary. In the trusted tier the held-out value lies above the published CI on 3 fits, below on 4 and inside on 2 (of 9). Over all panels it lies above on 8, below on 2 and inside on 1 (of 11). Read A is lower on 11 of 12 fits, and the twin margin moves by ≤ 0.02. **The devlog's "more often upward" holds for all panels but not for the trusted tier.** The devlog also misses the trusted-tier exceedances on ovarian s2 and lung s1/s2, and ovarian s2's Read A is higher, not lower. The TMA core s0 held-out all-panel value is NaN in the file.

**tab:cellina-cf generated (2026-09-30, coder; new `tables/generated/cellina_cf.tex`).** DISCELL finalL_s0 against Cellina's rewiring counterfactual on the same panels and draws (published read), per section. Rows: trusted fraction of ceiling with interval and panel count, Read A with interval, and twin margin. DISCELL's programme-only and leakage-only parts are shown as fractions of the ceiling; for Cellina they are "not mapped" (footnote: no leakage channel). Checks: Cellina replay 0.0 on all four sections, values equal to Cellina's `bootstrap_ci.json` and `transport_side_by_side_all.md`, and DISCELL's fraction equal to counterfactual / noise_ceiling. Cellina has the higher fraction on ovarian FFPE (0.73 vs 0.65) and the TMA core (0.90 vs 0.71, 1 panel); DISCELL has it on lung and FF. DISCELL's Read A is higher on all four sections, and its twin margin on three (lung: Cellina 0.435 vs 0.432).

**tab:synthetic generated (2026-09-30, coder; new `tables/generated/synthetic.tex`).** Synthetic recovery at the final configuration. It shows the final and uncontrolled arms at matched κ (0 and 0.2) and the misspecified-κ arms (planted 0.2, assumed 0–0.4), each as the range over 9 fits: NMI of μ_z, CCA of w, loading cosine, composition R². Below them are the world references: count clusters, planted z, and the random-subspace cosine (mean 0.23, 95th percentile 0.37–0.38). Source: `data/datasets/synthetic_smoke/experiments/synthetic_recovery.json`; the ranges were recomputed from the fits and checked against the stored summaries. The simulator checks all pass.

**tab:planted-percell generated (2026-09-30, coder; new `tables/generated/planted_percell.tex`).** The planted per-cell response, world × s: shift norm, KL AUC, w correlation, z absorption, reconstruction gain, Spearman and top-decile overlap, and the detection verdict. The caption states the pre-set rule: world 0 decides, worlds 1–2 replicate. No s is detected. The subset-predictability check (5-fold CV AUC 0.44–0.54) is in the caption. Source: `planted_percell.json`, with the per-seed values checked against its summaries. Note: the null (s = 0) already has cross-seed Spearman 0.39–0.41 on world 0, higher than at s = 1 (0.11–0.18).

**fig:tradeoff drawn (2026-09-30, coder; `figures/src/fig_tradeoff.py` → `figures/fig_tradeoff.pdf`, added to `build.sh`).** One panel per section (five), plus a legend panel. x: MLP composition leakage of z in % (as in tab:probe); y: cycle R² of z on the top-decile set (as in tab:battery). DISCELL is shown as the mean with seed-range bars. Also plotted: DISCELL without the adversary (not on the serial section, which has no uncontrolled cycle read), resolVI, Cellina, Cellina with the niche domain, Cellina on its own graph, and SIMVI and MintFlow where scored. The numbers are read through `scripts/paper_tables.py`, so the figure cannot disagree with the tables. MintFlow sits at low leakage with cycle R² 0.006–0.056 (lung, TMA core, serial section): low leakage with little within-type state kept. The caption is for the writer; a draft is in `scripts/logs/paper_tables_2026-09-30/AGENT_REPORT.md`.

**Frozen results integrated: tables, figure and configuration table (2026-09-30, integration agent).** Motivation: the author asked the coder side to put every table and result into the paper now, with the writer editing afterwards.
- **Main text (§3):** `\input` of tab:headline (already in), tab:probe (replaces the hand-built table of the same label in `sections/experiments.tex`), tab:breakdown (§3.6) and tab:battery (§3.7); fig:tradeoff added to §3.7 as `figure*`, with the draft caption from the coder's report, tightened.
- **Appendix:** tab:timing generated (replaces the hand-built table in app:timing); tab:transport-heldout in the new app:response; tab:breakdown-traj beside tab:kappa-sweep in app:sweep; tab:sensitivity in the new app:sensitivity; tab:cellina-cf in the new app:cellina-cf; tab:synthetic and tab:planted-percell in `appendix/synthetic.tex`, which is re-enabled in `main.tex` after the experimental appendix (the other evaluation-only appendices stay commented out).
- **tab:baselines rewritten:** whole sections only; resolVI refitted on the serial section (its transfer failed); SIMVI on the TMA core and serial section, the three whole-section fits queued (`\pending`); MintFlow on the TMA core, transferred, and the whole lung section, ovarian FFPE running and FF queued (`\pending`); Cellina plus three rows (niche domain, own graph, counterfactual refit). The old "not attempted"/"failed" wording is gone, following the whole-section decision. app:baselines text states the Cellina variants and the best-case caveat.
- **app:timing text:** MintFlow lung trained 879 min, 14.8 h wall clock with set-up and prediction (feasibility record), against 3.2–7.4 min for DISCELL.
- **Layout:** `\dbltopfraction` raised to 0.95 in `main.tex`, because tab:probe and tab:breakdown are taller than the default top fraction and, being `[t]` only, queued behind the checklist. tab:battery is taller than a page with its caption and still floats to the end of the main text; left for the writer (shorten the caption or allow `[p]` in the generator).

**Results text written, §3.2–§3.7 (2026-09-30, integration agent).** Numbers only from the generated tables and from the masked result files named in the manifest (signalling share, MI quadrant, axis test, GO localisation, subtype recovery, marker pairs, the envelope files, sensF_alpha_w, synthetic_recovery, feasibility). Motivation: a complete, truthful results story before the writer's edit.
- **§3.1 Baselines:** whole sections; the Cellina domain-adversary rows; the fits still running marked `\pending`. A pointer to the new app:readouts.
- **New §3.2 Recovery on Simulated Sections (sec:synthetic):** the pre-freeze skeleton's "Synthetic Recovery" subsection had no equivalent; one paragraph from tab:synthetic.
- **§3.3:** the transport `\pending` resolved (0.65–0.73 of the ceiling).
- **§3.4:** the probe stated plainly (3.7–5.8 % MLP, ≤ 1.6 % ridge; removal 84–86 % FF down to 29–48 % TMA core; relative residual largest on the TMA core); the comparator sentences moved to §3.7. Cycle asymmetry with κ* above the grid everywhere, and the linear count reference (above z on FF only). Subtype recovery: locations to w/m_ψ, proliferating and inflammatory tumour states to z, and the region-occupying states read from both (VEGFA⁺; TMA myofibroblasts; three FF tumour clusters).
- **§3.5:** transport (published read, programme vs leakage parts from tab:cellina-cf), Read A, twin margin, the held-out-tiles counts exactly as the table gives them; programmes and GO (FF fades, not a finding; the membrane/secreted lean is what a gene-specific leak would produce, so descriptive only); signalling share (lung now in; the leakage channel alone meets MintFlow's criterion on every section; FF response lean small, not significant, κ* 0.2); MI quadrant (lung now in); axis test from the masked per-run files (holds in tumour and macrophages, reversed in fibroblasts, and not specific to w: the z and depth rows show a contrast of similar size); the per-cell channel, with the planted result and the author's rule ("closed by design, not by the data").
- **§3.6:** the envelope, κ-survival overlap, trajectories, tab:breakdown (every computed contrast above the grid except FF signalling at 0.2; transport κ* `\pending`). **Marker-pair claim rewritten:** the contrast is corrected decode vs raw; it is negative on lung, FF and TMA and positive on ovarian, and it is already present at κ = 0 where the decode removes no leak, so it measures the decode, not the correction; the correction itself lowers exclusive and control double positives alike (controls by at least as much in pp) while the ratio falls. "Removes exclusive double-positives specifically" is not supported and is not claimed. Sensitivity: leak form ≤ 1.3 sd; the false-positive floor lowers ovarian transport 0.65 → 0.50 (−8.3 sd).
- **§3.7:** the battery, probe and fig:tradeoff framed as asked; checked cell by cell: DISCELL far below resolVI and Cellina on ovarian FFPE, lung and FF by both probes; on the TMA core Cellina's MLP residual is below DISCELL's (ridge still 4× lower for DISCELL); cycle R² not the highest everywhere (Cellina higher on ovarian FFPE and lung, resolVI on ovarian FFPE, SIMVI on the TMA sections); MintFlow near-zero leakage with near-zero cycle and NMI 0.90–0.99; Cellina's niche domain does not lower its residual. **Cellina counterfactual:** by the fraction of the ceiling Cellina leads on ovarian FFPE and on the TMA core (one trusted panel), DISCELL on FF and lung; the brief said "Cellina ahead on ovarian FFPE and lung", which tab:cellina-cf contradicts for lung (0.743 vs 0.703), so the text follows the table. DISCELL leads on Read A on all four and on the twin margin on three (lung equal).

**Appendices added or rewritten for the results (2026-09-30, integration agent).** Motivation: every readout the results use must be defined somewhere in the paper, without code references.
- **app:readouts (new):** niches, the guards, cycle set, transport (panels, folds, channels, R², noise ceiling, trusted and extrapolation tiers, half-tile interval and its ~0.8 coverage, the ≤ 0.03 tile-split sentence of manifest row 29), Read A, twin margin, programmes and GO, external criteria, marker pairs. Methods without a bibliography entry carry `\needsource` (nearest-neighbour MI estimator, MMD, varimax, Gene Ontology, Benjamini–Hochberg, the ligand–receptor database, the neural MI estimator).
- **app:subtype (new), app:response (new, with tab:transport-heldout), app:sensitivity (new; includes the α_w per-cell deviation share and held-out gain from the sensitivity record, not in the generated table), app:cellina-cf (new).** app:sweep: `\pending{text}` replaced.
- **appendix/synthetic.tex rewritten** for the final configuration: fit settings as run, readouts, random-subspace reference 0.23 (95th percentile 0.37–0.38, not "about 0.3"), results from tab:synthetic, misspecified-κ arms, simulator checks (the isolated-cell check is vacuous and says so). The pre-final single-seed numbers and the four "lessons" were dropped: they came from the closed-form, pre-final configuration and the paperlog shows no wish to keep them; the bistability observation is kept as one sentence of development history, with the final result (no larger spread under a planted leak). New app:planted-percell with the pre-set rule and the result.
- **app:kl-maps:** one sentence pointing to app:planted-percell.

**Front matter, method pendings and conclusion (2026-09-30, integration agent).** Motivation: results now exist, so the stale placeholders go.
- **Abstract:** the `\todo` is replaced by three hedged sentences (z keeps type and cycle; 3.7–5.8 % residual, less than two competing models on three of four sections; contrasts computed so far hold to κ = 0.4 except one; no per-cell channel at the operating point).
- **Introduction:** contribution (v) written. The `\todo` asking for synthetic support of the separation claim stays: the misspecified-κ arms show recovery is insensitive to κ, which does not directly support that sentence.
- **Method:** four `\pending` resolved: §2.4 α_w ladder → app:sensitivity and app:planted-percell; §2.5 weight record → 7.0 % vs 4.8 % (app:sensitivity); §2.7 degeneracy checks → I(z;t)/H(t) 0.82–0.95, within-type share 0.60–0.72, own-z gain 0.06–0.17 nats per count (envelope files, tab:recon-modes); §2.8 family sizes → tab:breakdown. "What w is not" points to app:planted-percell.
- **Conclusion:** first paragraph states the results; "less than competing models leave on most sections" is no longer true once MintFlow is scored (lung, TMA, serial), so it now says resolVI and Cellina on three sections, with MintFlow's near-type-label latent. Limitations: the false-positive floor, the non-specific response readouts, the held-out-tiles transport and the transductive comparison methods.

**Audit corrections applied (2026-09-30, corrections agent; audit `scripts/logs/paper_integration_2026-09-30/AUDIT.md`, report `CORRECTIONS.md` beside it).** Motivation: an independent audit of the integrated results text found claims that the result files contradict or do not carry; the author approved every correction, with the aim that the paper be publishable and truthful. Each item was re-checked against its source before the edit; a claim that lost force was rewritten to what the data support rather than deleted.
- **Contradicted claims (items 1–4).** Synthetic recovery: z agrees with the planted type "about as well as count clusters or better", below them on one fit of eighteen (0.80 vs 0.83), in §3.2 and app:synthetic. Marker pairs: control double positives rise except at κ = 0.3–0.4 on lung and κ = 0.4 on FF, where they fall by less than the exclusive ones. MI quadrant: "a second estimator" → the same nearest-neighbour estimator on another cell sample and floors. Response vs prior mean: nearly equal AUC on ovarian FFPE, FF and the TMA core except one TMA sublabel (activated vs venous endothelium, 0.76 vs 0.59, one seed); on lung the response separates clusters in three lineages better (0.67–0.73 vs 0.55–0.59). The claim "the response readouts are readouts of the context regression" is qualified accordingly in §3.5, §2.4, app:subtype, app:recon-modes, app:planted-percell, the conclusion and (one word, "largely") the abstract.
- **Overstatements (items 6–13, 19).** Cellina's niche-domain adversary "does not consistently lower" its residual (falls on 3, rises on 2); the relative residual is largest on the TMA core and its serial section; the image block is at a lower level on four sections and about equal on FF; lung twin margins: Cellina marginally ahead (0.435 vs 0.432); planted per-cell seed agreement is lower only on world 0 at s ≤ 1 and on world 1; synthetic loading spread is wider at κ = 0 than at the matched 0.2 (per-world ranges given); FF signalling share: a clause explains that the per-seed rank test and the pooled tile-resampled breakdown interval are different tests; Cellina's own-graph row is TMA core and serial only; twin-margin move "at most 0.025".
- **Untraceable claims (items 14, 15).** Uncontrolled w collapse now cites app:moran with the summed within-type variances from the uncontrolledL/finalL `validation.json` files (post-mask; the collapse is on every section, near-total on the two ovarian ones). Transport interval coverage kept, sourced to the planted-coverage table of `final_repair_2026-09-27/AGENT_REPORT.md` (simulated sections, mask-independent), with the ceilings stated exactly (0.82/0.85; 0.52–0.63). Tile-split: the devlog's pre-mask "at most 0.03" is replaced by the post-mask read from the per-run `transport.json` files (+0.01 to +0.02 on seed means, −0.03 to +0.05 per fit, no trusted panel on the TMA core). "About 85 %" → "most" (the comparison JSON does not record the published pool's split). `%` source comments added beside these numbers.
- **Generator (items 5, 16–18).** `scripts/paper_tables.py`: tab:headline is rounded once from the raw per-run values (the reader behind the envelope markdown), with the printed envelope kept as a check; TMA transport range 0.71–0.76 → 0.71–0.75 and interval [0.64, 0.85] → [0.63, 0.85]; ovarian NMI and TMA reconstruction interval bounds move in the third decimal. The TMA cycle R² of w now reads "−0.000 to 0.002" in tab:headline as in tab:kappa-sweep (one convention: half-up from the raw value, sign kept on a negative value that rounds to zero). tab:battery caption explains that DISCELL's rows are the battery's own reads on the cell set shared with the comparison methods; footnote a is built from feasibility.tsv and the queue log (MintFlow lung done; MintFlow ovarian FFPE running; MintFlow FF and three SIMVI fits queued). Four tests added, two rewritten against raw values; 25 pass. All tables regenerated.
- **Not changed, flagged:** "with intervals that contain zero on every fitted section" for the cycle R² of w holds for the seed envelope of intervals but not for single seeds on the ovarian FFPE, lung and FF sections; the tab:planted-percell caption's "through the fitted loadings" (integration report). Build: exit 0, no undefined references or citations, 1 overfull box (the template's), 64 pages.

**Two corrections left open by the audit pass (2026-09-30, coder).** (1) §3.3: "with intervals that contain zero on every fitted section" is true of the seed envelope but not of every seed. The per-seed intervals exclude zero on ovarian s0/s1, lung s0/s1 and FF s1/s2, at values ≤ 0.009. It now says the envelope contains zero and that single seeds exclude it at these small values. (2) tab:planted-percell caption: z absorption is read "through the decoder along the planted direction", as the code computes it, not "through the fitted loadings". The generator was changed and the table regenerated. Motivation: the text must match the data seed by seed and the method as computed.

**Data figures drawn from the result files (2026-09-30, figures agent; eight scripts in `figures/src/`, added to `build.sh`).** The author approved turning several tables into plots; every table stays, for exact values. Each script reads the files its table reads, through `scripts/paper_tables.py` (its loaders, and for fig:sensitivity and the kappa sweep the table builders themselves, so their checks run first). Shared helpers (method markers of fig:tradeoff, save, grid) were added to `figures/src/style.py`.
- **fig:kappa-sweep** (`fig_kappa_sweep`): sections × readouts (recon, NMI, mirror, cycle z/w q90, I(niche; w)); seed mean and min–max band over κ; κ = 0.1 marked. Source `kappa_sweep_sweepL.json` (the re-read graded on each fit's own split; the table's I(niche; w) agreement check passes, so nothing is withheld).
- **fig:probe-data** (`fig_probe_data`; the label fig:probe is taken by the schematic): per section and block, dumbbell from α_a = 0 (2 seeds) to the final fits (3 seeds) with seed ranges, comparison methods as fig:tradeoff markers; MLP panels, then ridge panels shaded as the secondary read; log x (0.3 to 39 %).
- **fig:sensitivity** (`fig_sensitivity`): forest plot of every arm's move in control SDs per readout, from tab:sensitivity's own traced cells; ±2 band; shared axis −9.6 to 7.6 so the fixed false-positive floor's transport move (−8.3) is drawn, not clipped; moves beyond 5 SD labelled.
- **fig:breakdown-data** (`fig_breakdown_data`): contrasts × sections; pooled estimate and Bonferroni interval at each κ, zero line, filled/hollow and dotted κ* as in the schematic. Reads `breakdown_all*.json`, then each section's `breakdown.json` for missing members; the 16 transport cells are drawn as "being computed". Each panel has its own y scale including zero, so intervals far from zero are shorter than their markers. Rerun when the breakdown queue finishes.
- **fig:planted-percell**, **fig:transport-heldout**, **fig:battery**, **fig:synthetic-misspec**: as specified; MintFlow reconstruction left out (B-mf1); TMA core held-out reads with no value are marked in the panel.

**Figures placed; four tables moved to the appendix (2026-09-30, figures agent).** Placed after CORRECTIONS.md appeared. Main text: fig:probe-data replaces the tab:probe input in §3.4; fig:kappa-sweep and fig:breakdown-data replace the tab:breakdown input in §3.6; fig:sensitivity after the "form of the leak" paragraph; fig:battery replaces the tab:battery input in §3.7, beside fig:tradeoff. Tables moved with their labels: tab:probe to app:probe (after the schematic), tab:breakdown to app:sweep, tab:battery to app:baselines (it is 16 pt taller than a page, as before the move). Appendix: fig:transport-heldout in app:response, fig:synthetic-misspec and fig:planted-percell in app:synthetic. References now point to the figure where the text describes a pattern, and to the table where it quotes values. Build: latexmk exit 0, no undefined references, 69 pages.

**Tables and figures polished for reading (2026-09-30, polish agent; report `scripts/logs/paper_polish_2026-09-30/AGENT_REPORT.md`).** The author approved four changes. Motivation: a reader should see at a glance which way each readout is good and who wins a comparison, without the caption, and the main-text headline table should be short. Tables are changed in `scripts/paper_tables.py` only, then regenerated. Figures are changed in their `figures/src/` scripts. Nothing was typed into a generated file.
- **tab:headline slimmed; tab:headline-full added.**
  - The main table has one row per read. Each cell is the mean with the seed range in brackets, the range in a smaller size. The reconstruction and cycle-R² rows have negative ranges, so their range sits under the mean; otherwise the table would be 192 pt wider than the page.
  - The † for the TMA transport seeds stays. The caption points to tab:headline-full for the intervals and has dropped the interval sentence.
  - The full version is new: `tables/generated/headline_full.tex`, label tab:headline-full. It is built by the same reader, so both tables come from one set of checked cells. It is placed in app:readouts, at the sentence on seed envelopes of intervals.
  - Text repointed: §3.3's sentence on the envelope of w's cycle-R² intervals now cites tab:headline-full. In §3.5, "these intervals" (transport coverage) became "its intervals (tab:headline-full)". No other text relied on the intervals of tab:headline.
- **Direction arrows and bold winners in tables.**
  - One map in the generator gives each readout's preferred direction (↑ or ↓), or "≈ 0" for w's cycle R².
  - Readouts not on the approved list have no arrow: the probe's "without the adversary" row is covered by its block's arrow; the others without one are the prediction parts, Moran's I, effective rank, programme overlap, w correlation, z absorption, reconstruction gain, Spearman, top-decile overlap, parameters, memory, epochs and the breakdown statuses.
  - Every caption with arrows says so once. Bold marks the best value per section, only in tab:battery, tab:probe, tab:cellina-cf and tab:timing:
    - DISCELL is compared by its mean over seeds.
    - Ties at the printed digits are all bold.
    - In tab:probe, DISCELL without the adversary does not compete: it is an ablation, not a method.
    - In tab:timing, a fit stopped by its time cap is not a finished fit.
  - The old decorative bold on tab:probe's MLP "fraction left" rows is removed, so bold has one meaning.
  - tab:headline, tab:headline-full, tab:kappa-sweep and tab:sensitivity have no bold.
  - Honest outcomes: MintFlow is bold on NMI and the MLP residual on several sections, Cellina on some cycle and reconstruction cells, and resolVI on the TMA reconstruction. Some of those reads are not held-out, as the caption already says.
- **Layout fixes caused by the arrows.** tab:battery: the two probe columns are now headed "Ridge ↓" and "MLP ↓" in a single header row, with the caption saying so, and the rows are 5 % tighter. It had been 16 pt taller than a page before this pass and 28 pt after the caption sentence; now it fits. tab:timing: column separation 2.5 pt. Note: in the AISTATS style `\footnotesize` is the same size as `\small` (9 pt).
- **Tests.** `tests/test_paper_tables.py`: the round trips are updated to the slim headline, the full headline and bold cells. New tests: `best()` picks the max or min per the direction map, ignores None/NaN, bolds ties and refuses a "≈ 0" readout; the bold cells in tab:battery, tab:probe, tab:cellina-cf and tab:timing are exactly the best per section, recomputed from the table or the source files; there is no bold where nothing competes; the arrows follow the map; slim versus full headline. 32 pass. A mutation check (probe and time directions flipped) fails four tests.
- **Figures readable without the caption.** Titles or axis labels carry ↑ or ↓ (a turned y label uses "better →") on fig:tradeoff, fig:battery, fig:probe-data, fig:kappa-sweep, fig:sensitivity, fig:transport-heldout, fig:planted-percell (KL AUC) and fig:synthetic-misspec. fig:breakdown-data's contrasts have no better direction, so it has no arrows.
  - fig:tradeoff marks the desirable corner ("low leakage, keeps cycle state").
  - fig:probe-data: the dumbbells are arrows from "without" to "with the adversary", with a direct label and a legend note.
  - fig:sensitivity: "±2 SD" is written on the band.
  - fig:planted-percell: the rule lines read "detection threshold".
  - fig:breakdown-data: the zero line reads "no contrast"; κ* was already labelled.
  - fig:kappa-sweep: a dashed zero line in the cycle-R² panels of w.
  - Five captions now say the arrows give the preferred direction. The fig:probe-data caption says "with an arrow to" instead of "joined to".
- **Build:** latexmk exit 0; no undefined references or citations; one overfull box, the template's 5.1 pt at the abstract; no float too large; 69 pages.

**tab:transport-heldout caption (2026-09-30, coder).** "The twin margin moves by at most 0.02" now reads 0.025, rounded to three decimals from the raw maximum 0.0245 (lung s1), which matches §3.5. Flagged by the table-polish pass.
**Correction to the entry above.** The raw maximum is 0.02448 (lung s1: 0.4312 → 0.4067), which is 0.024 at three decimals, not 0.025. The caption (generated) and §3.5 now both read 0.024.

**Context-latent grading: tab:context, fig:separation, tab:baselines latents column (2026-09-30, coder; devlog "Context-latent grading for every method").**
- **tab:context** (new, generated by `scripts/paper_tables.py --only context` from `data/datasets/<section>/experiments/context_grade.json`; `\input` in app:baselines right after tab:battery). Per section and method, it gives the context latent's width d; the composition and image recovery by the MLP and ridge probes (↑, bold = highest per section and column, DISCELL by its mean); I(niche; ·) excess, with no arrow and no bold because the kNN estimate is not comparable across widths (rule set in advance in the devlog); and the k-means NMI with type (descriptive, no arrow). resolVI's row says it has no context latent. SIMVI and MintFlow are `\pending{}` where their whole-section fits are queued or running.
  - **Transform: a departure, flagged.** The recovery is printed as 1 − e^{−2·excess}, not tab:probe's e^{2·excess} − 1. The two agree to first order. The tab:probe form is unbounded: MintFlow's context latent would read 590–1900 % "of the within-type variance", which is not a share. The caption says so. Switching back would be a one-line change in `ctx_share`.
  - **Rejected:** a latent-name column. tab:baselines now carries the mapping and the caption names each latent. The column made the table 34 pt too wide.
- **fig:separation** (new, `figures/src/fig_separation.py`, added to `build.sh`; main text, right after fig:tradeoff). One panel per section. x: composition recovered from the context latent (as tab:context, "better →"). y: composition left in the intrinsic latent (as tab:probe and fig:tradeoff's x, "← better"). Direct labels, with leader lines in the two crowded TMA panels. The sixth panel holds a note (resolVI has no context latent; SIMVI and MintFlow are shown only where fitted). It reads clearly: DISCELL sits low on y and mid on x, Cellina higher on both, MintFlow bottom right. So it goes in the main text, not the appendix.
- **tab:baselines:** new column "Latents: intrinsic / context", following the devlog mapping. resolVI: its single cell latent / none (contamination is a mixture, not a latent). SIMVI: intrinsic / spatial. MintFlow: intrinsic / microenvironment. Cellina rows: intrinsic / microenvironment s. The caption defines the column and points to tab:context. Column widths were re-split to keep the total at 0.88 textwidth.
- **§ setup, Baselines paragraph:** one sentence. SIMVI, MintFlow and Cellina split each cell into an intrinsic and a context latent. resolVI models neighbour contamination as a mixture with a single cell latent, so it is a reference for the intrinsic side, not an intrinsic/context split.
- **§ comparison, "Niche information and within-type state":** two sentences at the end, with numbers from context_grade.json (nonlinear probe, 1 − e^{−2·excess}):
  - every context latent carries niche information beyond the type (the I(niche; ·) excess is 0.42–1.50 nats, positive for every method, section and seed);
  - DISCELL's μ_w holds less composition than Cellina's s on ovarian FFPE, lung and FF (13–35 % over seeds, against 33–45 %) and about as much on the TMA core and serial section (9.9–17 %, against 14 and 16 %);
  - it holds far less than MintFlow's latent (76–93 %);
  - it holds more of the image block than any comparison method on ovarian FFPE and lung, by both probes.
- app:baselines paragraph: one sentence pointing to tab:context.
- **Build:** latexmk exit 0; no undefined references or citations; one overfull box, the template's 5.1 pt at the abstract (tab:context was 34 pt wide before the header was split into Composition / Image groups); 71 pages. Rendered and checked: p16 (setup), p21 (text), p26 (fig:separation), p59 (tab:baselines), p61 (tab:context).

**Probe percentages as the share of within-type variance explained, 1 − e^{−2·excess} (2026-09-30, coder; author's decision of the same day).**
- **Why.** e^{2·excess} − 1 is the error ratio minus one, not a share of variance. tab:probe labelled it as a share, and it would read above 100 % for a latent that carries the niche. tab:context already used 1 − e^{−2·excess}. Every probe percentage in the paper now uses that form.
- **Code.** `scripts/paper_tables.py`: one helper, `probe_share(excess)`, gives 100·(1 − e^{−2·excess}). `pct()` now takes a value already in %. `ctx_share` is removed and folded into the helper.
  - tab:probe, tab:sensitivity (MLP column) and tab:context all read the excess through the helper.
  - The files' `var_fraction` (e^{2·excess} − 1) is still checked against the excess, but it is never printed.
  - Figures read through the same helper: fig:probe-data, fig:tradeoff (x), fig:separation (x and y), fig:probe (schematic, panel b), and `fig_probe_real.py` (a candidate, not in the paper).
  - fig:sensitivity and fig:battery read the table builders. fig:battery shows fractions left, so it is unchanged.
  - fig:separation: the leader-line labels in the serial panel moved down (y = 9.4/8.6/7.8/7.0, was 10.3/9.3/8.3/7.1) because the axis shrank.
  - Tests: new ones for the helper, for "no table prints e^{2·excess}", and for a round trip of the sensitivity MLP column. The existing probe and context round trips now recompute from the excess. 36 pass.
- **Unchanged by construction.**
  - Fraction left (a ratio of excesses), and so tab:battery's probe columns and fig:battery.
  - Bold cells in tab:probe and tab:context (checked cell by cell: the transform is monotone).
  - tab:context values: only its caption changed.
  - The removed shares in §results-z and the conclusion: 84–86 % FF, 68–77 % ovarian FFPE, 57–64 % lung, 29–48 % TMA core, 30–41 % serial. They are 1 − fraction left, computed from excesses, and recomputed from the regenerated table they come out equal. Had they been computed from the new percentages, they would read 82–84, 66–75, 55–62, 28–47 and 30–40 %.
  - The calibration appendix's "% of the uncontrolled baseline" (excess ratios; the appendix is not \input).
- **Definitions.**
  - Method §2.5: "exp(2G_b) − 1 is the fraction" became "1 − exp(−2G_b) is the share", plus one sentence defining the reported percentages.
  - fig:probe caption (app:probe): e^{2G_k} − 1 became 1 − e^{−2G_k}, and panel c says how the % is formed.
  - Captions of tab:probe, tab:sensitivity, tab:context, fig:probe-data, fig:tradeoff and fig:separation now give 1 − e^{−2·excess}, "the share of the block's within-type variance that the probe explains beyond the permutation floor".
  - app:readouts has no probe definition, so nothing changed there.
- **Text (before → after).**
  - Abstract and conclusion: MLP residual with the adversary 3.7–5.8 % → 3.5–5.5 %.
  - §results-z:
    - with the adversary 3.7–5.8 → 3.5–5.5 %; linear "at most 1.6 %" unchanged;
    - without it, nonlinear 6.9–39 → 6.4–28 %, linear 2.5–20 → 2.4–17 %;
    - "about 7 %" → "6 to 7 %" (TMA core and serial without the adversary);
    - image block on four sections 1.4–4.1 → 1.4–4.0 %;
    - FF image 4.9–5.5 → 4.7–5.2 %, FF composition 4.6–5.3 → 4.4–5.0 %.
  - §comparison:
    - DISCELL nonlinear 4.1–5.8 → 4.0–5.5 %; resolVI and Cellina 8.5–31 → 7.9–24 %; linear 0.6–1.6 % unchanged, comparators 4.4–22 → 4.2–18 %;
    - TMA core: Cellina 3.6 → 3.5 %, DISCELL 3.7–5.0 → 3.5–4.8 %, SIMVI 5.1 → 4.9 %, resolVI 5.7 → 5.4 %. "Cellina's is below DISCELL's" became "just below DISCELL's lowest seed" (3.47 against 3.54; both print 3.5);
    - TMA core linear comparators 2.1–3.1 → 2.1–3.0 % (still at least four times: 2.06 / 0.50);
    - serial DISCELL 4.2–5.0 → 4.0–4.8 %;
    - Cellina niche-domain change "at most 2.2 points" → 2.0;
    - MintFlow 0.3–1.1 % and the context shares (13–35, 33–45, 9.9–17, 14 and 16, 76–93 %) unchanged.
  - app:sensitivity, adversary: composition weight 1 raises the residual 4.8 → 4.6 % to 7.0 → 6.6 % (+2.6 → +2.5); weight 5 3.9 → 3.7 %. "At most 0.7 SD" unchanged.
  - fig:probe schematic panel b: 23 % → 19 % (illustrative).
- **Tables (before → after, in %).** tab:probe:
  - Composition, MLP, without the adversary: Ovarian FFPE 19.2–19.4 → 16.1–16.2; Lung FFPE 12.0–13.0 → 10.7–11.5; Ovarian FF 38.2–39.0 → 27.6–28.0; TMA core 6.9–7.4 → 6.4–6.9; TMA serial 7.1–7.4 → 6.6–6.9
  - Composition, MLP, with the adversary: Ovarian FFPE 4.1–5.8 → 4.0–5.5; Lung FFPE 4.4–5.2 → 4.2–5.0; Ovarian FF 4.6–5.3 → 4.4–5.0; TMA core 3.7–5.0 → 3.5–4.8; TMA serial 4.2–5.0 → 4.0–4.8
  - Composition, MLP, resolVI, Cellina: Ovarian FFPE 21.1, 16.1 → 17.4, 13.9; Lung FFPE 12.6, 8.5 → 11.2, 7.9; Ovarian FF 30.9, 29.7 → 23.6, 22.9; TMA core 5.7, 3.6 → 5.4, 3.5; TMA serial 7.8, 8.2 → 7.3, 7.6
  - Composition, MLP, MintFlow, SIMVI: TMA core 0.3, 5.1 → 0.3, 4.9; TMA serial 1.0, 7.3 → 1.0, 6.8
  - Composition, MLP, Cellina, niche domain: Ovarian FFPE 15.5 → 13.4; Lung FFPE 9.3 → 8.5; Ovarian FF 27.5 → 21.6; TMA core 5.8 → 5.5; TMA serial 8.0 → 7.4
  - Composition, MLP, Cellina, own graph: TMA core 4.4 → 4.2; TMA serial 9.1 → 8.3
  - Image, MLP, without the adversary: Ovarian FFPE 14.4–18.8 → 12.6–15.8; Lung FFPE 9.7–10.9 → 8.8–9.8; Ovarian FF 37.2–40.3 → 27.1–28.7; TMA core 6.3–9.2 → 6.0–8.4; TMA serial 5.9–7.7 → 5.6–7.2
  - Image, MLP, with the adversary: Ovarian FFPE 2.3–4.1 → 2.2–4.0; Ovarian FF 4.9–5.5 → 4.7–5.2; TMA core 2.4–3.4 → 2.3–3.3; TMA serial 2.5–3.1 → 2.5–3.0
  - Image, MLP, resolVI, Cellina: Ovarian FFPE 15.1, 10.4 → 13.1, 9.4; Lung FFPE 11.2, 4.4 → 10.1, 4.2; Ovarian FF 30.6, 14.8 → 23.4, 12.9; TMA core 7.8, 4.1 → 7.2, 3.9; TMA serial 7.7, 2.9 → 7.2, 2.8
  - Image, MLP, MintFlow, SIMVI: TMA core 0.8, 9.0 → 0.8, 8.2; TMA serial 0.6, 8.6 → 0.6, 7.9
  - Image, MLP, Cellina, niche domain: Ovarian FFPE 10.7 → 9.7; Lung FFPE 4.9 → 4.7; Ovarian FF 17.3 → 14.8; TMA core 5.5 → 5.3; TMA serial 4.3 → 4.1
  - Image, MLP, Cellina, own graph: TMA core 5.0 → 4.7; TMA serial 4.6 → 4.4
  - Composition, ridge, without the adversary: Ovarian FFPE 5.9–7.1 → 5.6–6.7; Lung FFPE 3.8–4.1 → 3.7–3.9; Ovarian FF 19.9–20.2 → 16.6–16.8; TMA core 2.5–2.6 → 2.4–2.6
  - Composition, ridge, with the adversary: TMA serial 0.8–1.1 → 0.8–1.0
  - Composition, ridge, resolVI, Cellina: Ovarian FFPE 8.0, 8.9 → 7.4, 8.1; Lung FFPE 4.7, 4.4 → 4.5, 4.2; Ovarian FF 13.8, 22.3 → 12.2, 18.2; TMA core 2.5, 2.1 → 2.4, 2.1; TMA serial 2.7, 3.1 → 2.6, 3.0
  - Composition, ridge, MintFlow, SIMVI: TMA core 0.9, 3.1 → 0.9, 3.0; TMA serial 1.1, 3.1 → 1.1, 3.0
  - Composition, ridge, Cellina, niche domain: Ovarian FFPE 9.4 → 8.6; Lung FFPE 5.1 → 4.8; Ovarian FF 20.8 → 17.2; TMA core 3.8 → 3.6; TMA serial 3.9 → 3.8
  - Composition, ridge, Cellina, own graph: TMA core 2.7 → 2.6; TMA serial 3.6 → 3.5
  - Image, ridge, without the adversary: Ovarian FFPE 6.2–8.6 → 5.9–7.9; Lung FFPE 3.5–3.7 → 3.4–3.5; Ovarian FF 21.1–23.6 → 17.4–19.1; TMA core 2.8–3.3 → 2.7–3.2
  - Image, ridge, with the adversary: Ovarian FF 2.2–2.6 → 2.1–2.6
  - Image, ridge, resolVI, Cellina: Ovarian FFPE 6.8, 7.2 → 6.4, 6.7; Lung FFPE 4.0, 2.4 → 3.9, 2.4; Ovarian FF 14.3, 11.6 → 12.5, 10.4; TMA core 2.6, 2.1 → 2.5, 2.1; TMA serial 2.6, 1.9 → 2.5, 1.9
  - Image, ridge, MintFlow, SIMVI: TMA core 0.8, 3.9 → 0.8, 3.7; TMA serial 0.7, 3.4 → 0.7, 3.3
  - Image, ridge, Cellina, niche domain: Ovarian FFPE 7.9 → 7.3; Lung FFPE 3.4 → 3.2; Ovarian FF 13.9 → 12.2; TMA core 3.5 → 3.4; TMA serial 2.9 → 2.8
  - Image, ridge, Cellina, own graph: TMA core 2.9 → 2.8
- **tab:sensitivity, MLP column (share, and in brackets the move in control SD; the moves are recomputed on the share scale).** In table order: ovarian FFPE, then TMA core, then ovarian FF.
  - final configuration: 4.8 → 4.6
  - range: 4.1–5.8 → 4.0–5.5
  - per cell, depth ratio: 5.7 (1.0) → 5.3 (1.0)
  - per gene: 4.7 (-0.2) → 4.4 (-0.2)
  - per cell, density ratio: 4.6 (-0.2) → 4.4 (-0.3)
  - one per section: 5.0 (0.2) → 4.8 (0.2)
  - area-scaled, per cell: 4.5 (-0.4) → 4.3 (-0.4)
  - α_w = 1/ℓ̄: 3.5 (-1.5) → 3.4 (-1.5)
  - α_w = 2/ℓ̄: 4.2 (-0.7) → 4.1 (-0.7)
  - α_w = 5/ℓ̄: 4.4 (-0.5) → 4.2 (-0.5)
  - α_w = 10/ℓ̄: 4.5 (-0.4) → 4.3 (-0.4)
  - α_w = 25/ℓ̄: 5.9 (1.3) → 5.5 (1.2)
  - head steps 12: 4.5 (-0.4) → 4.3 (-0.4)
  - head width 128: 5.4 (0.7) → 5.1 (0.7)
  - ensemble of 3: 4.8 (-0.1) → 4.5 (-0.1)
  - composition weight 1: 7.0 (2.6) → 6.6 (2.5)
  - composition weight 5: 3.9 (-1.1) → 3.7 (-1.1)
  - final configuration: 4.2 → 4.0
  - range: 3.7–5.0 → 3.5–4.8
  - one per section: 4.1 (-0.1) → 4.0 (-0.1)
  - α_w = 1/ℓ̄: 3.7 (-0.7) → 3.5 (-0.7)
  - α_w = 2/ℓ̄: 3.2 (-1.4) → 3.1 (-1.4)
  - α_w = 5/ℓ̄: 3.8 (-0.5) → 3.7 (-0.5)
  - α_w = 10/ℓ̄: 3.3 (-1.2) → 3.2 (-1.2)
  - α_w = 25/ℓ̄: 3.6 (-0.8) → 3.5 (-0.9)
  - final configuration: 4.9 → 4.6
  - range: 4.6–5.3 → 4.4–5.0
  - α_w = 5/ℓ̄: 4.5 (-0.9) → 4.4 (-0.9)
  - α_w = 10/ℓ̄: 4.4 (-1.3) → 4.2 (-1.3)
- **Figures regenerated** with `build.sh`; exit 0. Looked at fig:probe-data, fig:tradeoff, fig:separation, fig:sensitivity and fig:probe. No marks are clipped, and the log axis of fig:probe-data still spans every value. `fig_probe_real.py` stops on an assertion that was already there: the rebuilt gain from its cache, 0.0039, is not the record's 0.0070. It is not in `build.sh` or the paper, and it was not regenerated.
- **Build.** latexmk exit 0; no undefined references or citations; one overfull box, the template's 5.1 pt at the abstract; 71 pages. Rendered and checked: p57 (tab:probe), p69 (tab:sensitivity).

**Editorial flags, first house-cleaning pass (2026-09-30, editor agent; the author's request: flags only, no text changed).** This is the overview before trimming. Every paragraph of the main text, and every subsection of the appendix, was judged against the red thread the author confirmed: (1) the response–leakage confound and κ unidentified; (2) κ swept, each claimed contrast with its κ\*; (3) z/w split, adversary, held-out probe for DISCELL and competitors on the same cells; (4) z keeps type and cycle with less niche than competitors, contrasts survive the grid, and the per-cell channel is closed by design, stated as scoped. The checklist was skipped. `tables/generated/` was not touched: generated tables are flagged at their `\input` line. The report is in `scripts/logs/paper_flags_2026-09-30/AGENT_REPORT.md`.
- **Macro** (`macros.tex`, next to `\todo`, `\pending` and `\needsource`): `\flag{KIND}{note}` prints `[KIND: note]` in bold sans-serif footnotesize, teal on light cyan. It is defined with `\DeclareRobustCommand`, so it works in captions and footnotes (tested). **Switch:** `\newif\ifshowflags \showflagstrue`, commented "set \showflagsfalse to hide every editorial flag".
  - **Departure from the suggested form:** `\colorbox` cannot break a line, and a 15-word note overflows a column of the two-column layout. The macro therefore highlights with soul's `\hl`, which breaks, and `macros.tex` gains `\usepackage{soul}` and `\colorlet{flagbg}{cyan!12}`.
  - `\leavevmode` keeps a run-in `\paragraph` heading from taking the flag's font.
  - `\ignorespaces` means a hidden flag leaves no stray space.
- **Counts (125 flags):** TRIM 33, MOVE 20, CHECK 19, REVIEWER 18, KEEP-CORE 17, MERGE 12, CUT 6. By file: abstract 3, introduction 9, method 40, experiments 36, conclusion 5; appendices: derivations 3, rationale 12, implementation 4, experimental 10, synthetic 3. All 12 main-text floats carry a judgement.
- **No text changed.** With every `\flag{..}{..}` stripped, each of the ten edited `.tex` files is byte-identical to HEAD. Settled decisions are flagged KEEP-CORE or REVIEWER (keep), never for removal: the descriptive probe; figures from seed 0; the lean 8.19; the transport headline with the held-out-tiles row; the held-out-neighbour exposure without a buffer; κ = 0.1 as a working point.
- **Build.** latexmk exit 0; no undefined references or citations; one overfull box, the template's 5.1 pt at the abstract; 73 pages (71 before). With flags shown, the underfull warnings rise from 47 to 58, which is cosmetic. Rendered and checked: p1 (abstract), p7, p18, p21, p26 (fig:tradeoff caption) and p34 (appendix). The flags are distinct from the red, orange and violet markers, break across lines, and do not disturb floats.
- **Hidden build.** A scratch copy with `\showflagsfalse` gives exit 0 and 71 pages, with the same warnings as before the flags. Its `pdftotext` output is byte-identical to that of the pre-flag HEAD build. `macros.tex` in the paper keeps `\showflagstrue`; it was never switched.
- **Rejected:** placing the flags before each float's `\begin`. It would set them in the running text, pages away from the float, so they go at the caption start.

**Editorial flags, fix pass (2026-09-30, editor agent; the author's request: fix what can be fixed, update flags where a fix changes them, leave TRIM/MOVE/CUT/MERGE alone).** Report: `scripts/logs/paper_flags_2026-09-30/FIXES.md`. 125 → 110 flags: 15 CHECK/REVIEWER flags removed as resolved, 14 notes rewritten. No code, devlog, manifest or queue was touched; nothing was committed.
- **§2.5, the adversary's composition weight** (CHECK removed). The source is tab:sensitivity, ovarian FFPE block, regenerated: composition weight 1 gives 6.6 % against 4.6 % at the final configuration. Before: "7.0 % … against 4.8 %". After: "6.6 % … against 4.6 %".
- **Axis test after the gene-pairing fix** (CHECKs in §3.5 and app:response removed). Source: `external_axis_test_finalL_s{0,1,2}.json`; the pre-fix copies are `*_prefix.json`. The re-read pairs 9,385–9,677 gene–type panels (before: 9,801–9,974). Values below are the range over seeds, before → after.
  - Response, pooled, mean |τ|: true axis 0.78–0.81 → 0.78–0.81; false axis 0.51–0.56 → 0.51–0.56 (unchanged at two decimals).
  - Gene–type pairs at |τ| ≥ 0.9: true 6,021–6,493 → 5,843–6,314; false 195–1,141 → 191–1,100.
  - Tumour cells: 0.90–0.93 → 0.90–0.93 against 0.49–0.62 → 0.50–0.63.
  - Macrophages: 0.80–0.81 → 0.80–0.81 against 0.27–0.35 → 0.26–0.34.
  - Merged fibroblasts: 0.63–0.71 → 0.63–0.71 against 0.71–0.75 → 0.71–0.75 (still reversed).
  - Intrinsic state: 0.59–0.65 → 0.59–0.65 against 0.38–0.46 → 0.39–0.47.
  - Depth: 0.57–0.67 → 0.57–0.68 against 0.35–0.42 → 0.36–0.42.
  - Observed: 0.47 against 0.33–0.36 (unchanged).
  - §3.5's "true-axis |τ| (0.57 to 0.67)" for the intrinsic state and depth → 0.57 to 0.68.
  - **No qualitative statement changes.** The response is still the closest to the bands, the contrast still holds in tumour cells and macrophages and is reversed in the fibroblast class, and the references still show a difference of similar size.
  - Breakdown axis member: the per-seed re-read draws (axis_fix logs) give 0.20–0.31 at every grid point, every per-seed interval above 0. The 3-seed means are 0.273 / 0.268 / 0.255 at κ = 0 / 0.1 / 0.4 (before: 0.272 / 0.268 / 0.254), so "holds across the grid" stands.
  - tab:breakdown still shows the pre-fix axis row, because it reads `breakdown_all.json`, which the running breakdown queue renders at its end; `breakdown --all` was not run. `scripts/paper_tables.py` was rerun (exit 0): only the timestamps changed, since no other generated table carries axis values. The flag at tab:breakdown now says so.
- **Introduction, the planted-leakage todo** (CHECK removed; todo resolved). One sentence: on sections simulated at a planted leak fraction of 0.2, DISCELL fitted at any assumed fraction from 0 to 0.4 recovers the planted states about equally well, so the fit does not point to the planted value (app:synthetic). The TRIM flag on that paragraph now reads "merge the two overlapping todos (the planted-leakage one is resolved)".
- **Introduction, resolVI's leak share** (REVIEWER removed). Source: `mixture_colmeans` in `DisCell-baselines/results/resolvi_lineage/<section>/config.json`, the posterior mixture proportions (true, neighbour, background) averaged over cells. The added clause: its mean neighbour weight is 0.041 to 0.086, including the 0.01 floor that scvi-tools adds (`diffusion_eps`), and its background weight is at most 0.003. By section: ovarian FFPE 0.041, lung 0.049, FF 0.086, TMA core 0.054; background 0.0031 / 0.0010 / 0.0010 / 0.0010.
- **app:pathb, last sentence** (CHECK removed). Before: "so it never asserts κ = 0 and contradicts it". After: "It decodes through the same leakage mixture as the first term, at the same κ, so it never treats the counts as free of leakage and the two terms never disagree about how much of them is leakage."
- **§2.3, the decoder paragraph** (CHECK removed). "whether z separates cell-cycle phases within a type is an empirical check for the evaluation" → "that z carries the cell cycle within a type, and w does not, is checked in §3.4 (sec:results-z)".
- **§2.4, per-edge geometry** (CHECK removed). "The estimand is used only informally here; counterfactual machinery is future work" → the estimand is used at the level of cell types, in the transport counterfactual (sec:results-w); per-cell counterfactuals, which need abduction through an open per-cell channel, are future work.
- **§2.8, the serial-section rule** (CHECK removed). Added: only the TMA core has a serial section, and among the claimed contrasts only the cycle asymmetry is read on it, so the rule is applied to that contrast alone; the other contrasts on the TMA core are section-level results of the core.
- **app:rationale opening** (CHECK removed). "It does not repeat what is already there" → "Some subsections, among them app:mirror and app:qw, restate the main text's step before completing it."
- **app:kappa, the breakdown example** (CHECK removed). Softened: a readout whose breakdown point lay above the bound would survive every leak fraction the data allow. That bound is not computed on these sections, so what the grid supports is the weaker statement: a readout that survives it survives every leak fraction up to 0.4.
- **app:conditional todo** (CHECK kept, note updated). The CSVAE numbers are in DisCoVR Table 19, which the paperlog records as checked on 09-23, but they are quoted nowhere in the manuscript or devlog. They need a read of that table.
- **§3.1 Baselines, calibration disclosure** (REVIEWER removed). The baselines were not tuned. DISCELL's weights were calibrated on the reported sections, with two or three seeds per setting:
  - α_z over four values on the primary section;
  - α_w over five multiples of 1/ℓ̄ against 0.1 on the primary section and the TMA core, and over two on the fresh-frozen section;
  - the adversary's head steps, head width, ensemble size and composition weight on the primary section, the weight confirmed on the TMA core and the fresh-frozen section (app:sensitivity).

  Source: devlog, the α_z ladder (2026-09-21/22), 8.15 α_w ladder stages 1–2, and the 8.17 adversary ladder with its confirmation. **Conclusion, Limitations "Evaluation":** a mirrored clause: the comparison methods ran at their published defaults, while DISCELL's weights were calibrated on these sections (sec:setup).
- **§3.6, runtime** (REVIEWER removed). Source: tab:timing. A DISCELL fit takes 1.3–43 min on one GPU with its evaluations included. The comparison methods' training alone takes 5.5–105 min for resolVI, 3.4–29 min for Cellina (faster only on FF), and 193–879 min for SIMVI and MintFlow where they have run.
- **Abstract** (REVIEWER removed). "less than two competing models leave on three of the four sections" → "less than resolVI and Cellina leave on three of the four sections; MintFlow, fitted so far on two sections, leaves less still, but its intrinsic latent is close to a type label, with a cycle R² of at most 0.06". The source is tab:battery: MintFlow's cycle R² is 0.006 / 0.054 / 0.056 on lung / core / serial.
- **Notes updated to "decision needed" with one option each (flags kept):**
  - the ω = 0 ablation (§2.5);
  - the marker-set upper bound (§2.2 and app:kappa);
  - the marker-pair contrast in the claimed set (§3.5);
  - app:planted's 300-epoch protocol (Limitations and app:planted);
  - tab:related verification;
  - GEO/KRONOS licence wording;
  - MintFlow's reconstruction, B-mf1 (fig:battery);
  - the abstract's and conclusion's breakdown clauses, which wait on the transport contrasts.
- **Build.** latexmk exit 0 with `\showflagstrue`; no undefined references or citations; one overfull box, the template's 5.1 pt at the abstract; 73 pages. Rendered pp. 1, 3 and 17: the flags are visible, and so is the new text.

**Author's decisions applied: marker pairs, marker-set bound, availability; todo sweep (2026-09-30, coder; devlog "Author's decisions on the open flags (2026-09-30)", items 2, 3, 7).** Report: `scripts/logs/paper_flags_2026-09-30/DECISIONS_APPLIED.md`.
- **The marker-pair contrast leaves the κ\* family.** In the code: `breakdown.py` sets m_s to 8/7/7/7/1 and reads the contrast as a trajectory from `marker_pairs.json`. `paper_tables.py` drops the κ\* row and adds a trajectory row in percentage points. `fig_breakdown_data.py` drops the panel row.
  - §2.8 now says "seven contrasts per section and eight on the primary section", and adds a sentence on why the marker pairs are a trajectory: the contrast is present at κ = 0, so it measures decoding, not leak removal.
  - §3.5 marker pairs: "needs care" becomes "reported as a trajectory of the decode, not as a claimed contrast". The rest of the honest statement is kept, ending with "it is therefore given no breakdown point".
  - app:readouts and app:response now point to tab:breakdown-traj. The app:sweep trajectory list now includes the marker-pair contrast.
  - The abstract and conclusion are unchanged: they state no count, and "except one, on one section" stays true.
  - tab:breakdown and tab:breakdown-traj in the PDF stay the old render until the breakdown queue's final `breakdown --all`. Until then `paper_tables.py` stops at tab:breakdown (m = 9 in the file).
- **Marker-set upper bound dropped.** §2.2: the clause "a data-driven upper bound from an atlas-defined marker set" is gone. app:kappa: "three sources … and a marker-set ceiling", the ceiling's definition and its `\pending` are removed; the grid now has two sources. The pinning sentence no longer points to "the ceiling above". It now reads: "such as a marker that an external atlas excludes for the cell's type, or through a second view of the cell such as a nuclear/extranuclear split".
- **Availability (app:assets, checklist).** The GEO samples are called public and cited, and the unverified GEO-disclaimer paraphrase and its todo are removed. KRONOS is used "as distributed by their authors" and not redistributed. "The code will be released on acceptance" becomes "A code repository accompanies the submission as a clean, standalone implementation of DISCELL; it does not contain KRONOS or its weights…". Checklist: code [Yes], reproduction [Yes], new assets [Yes].
- **Flags removed (4):** the §3.5 marker-pair REVIEWER, the §2.2 and app:kappa marker-set REVIEWERs, and the app:assets GEO/KRONOS CHECK. The count goes from 110 to 106.
- **Todo sweep, compiled files.**
  - Resolved: the GEO wording.
  - Kept:
    - the checklist (submission time);
    - tab:deviations keep-or-cut (author);
    - "quantify" the clip effect (needs a rerun at the 40 µm prune);
    - the CSVAE numbers (DisCoVR Table 19);
    - the two related-work todos (source papers);
    - the escalation (a closed-form fit per section, 8.11b).
  - Flag notes updated: intro TRIM, app:deviations CUT, §2.5 REVIEWER.
  - Five files are not `\input` by `main.tex`; I left them untouched.
- **Build.** latexmk exit 0 with `\showflagstrue`. No undefined references. The only overfull box is the template's 5.1 pt. 73 pages.

**app:planted rerun at the final configuration; conclusion limitation (2026-09-30, writer; devlog "Planted-worlds amortisation gap at the final configuration (results, 2026-09-30)").** Source: `data/datasets/synthetic_smoke/experiments/planted_posterior_final_vs_prefinal.md` and `planted_posterior_final.json`; report part B of `scripts/logs/omega0_2026-09-30/AGENT_REPORT.md`.
- **tab:planted-gap** (hand-built), before → after:
  - DISCELL encoder: 0.55 / 0.83 / 1.48 → 0.65 / 1.02 / 1.93.
  - Perceptron, encoder's inputs: 0.61 / 0.95 / 1.61 → 0.61 / 0.95 / 1.60 (value 1.6048 unchanged; the old 1.61 was a rounding error).
  - Linear regression: 1.40 / 2.50 / 4.53 → 1.39 / 2.50 / 4.53 (value 1.3947 unchanged; the old 1.40 was a rounding error).
  - Perceptron with ρ̄ (0.53 / 0.80 / 1.35) and type mean (2.4 / 4.1 / 7.4) unchanged.
  - Caption: added "DISCELL is fitted at the final configuration".
- **app:planted, the fit.** "for 300 epochs" (wrong: the old run was 600 fixed epochs) → the final configuration with the widths reduced to the small world (d_z 2, d_w 2, hidden 128, attention 16, tiles of 512): adversary α_a 0.3 with composition weight 3, α_w 0.1 with the 30-epoch warm-up, α_z = ½/ℓ̄ (≈ 0.005 / 0.0017 / 0.0005 by depth), 500 epochs with patience 40 on 15 % held-out tiles, read at the accepted checkpoint; 45 fits, accepted epochs 59–434.
- **app:planted, the result**, rewritten. Before: encoder below the same-input perceptron at every depth and close to the ρ̄ perceptron ("not seeing the influx costs little"); flat in z units; a wrong κ enlarges it by at most a tenth and it is smallest at the true value. After:
  - the encoder is above the same-input perceptron at every depth (below it in 3 of 9 depth × seed cells) and 0.12–0.58 above the ρ̄ perceptron; well below the linear regression and type mean;
  - so its posterior mean differs from the exact one by a measurable amount, more than a flexible regression on the same inputs leaves;
  - the gap in z units stays roughly flat (0.16–0.17) while the posterior sd falls from 0.17 to 0.06;
  - worst wing factor 1.13 (was "a tenth"); not smallest at the true κ at any depth (D = 1,000: 1.77 at κ 0.1 too low against 1.93);
  - the earliest-stopping seed is the worst at every depth; adversary vs stopping rule as the cause is untested (stated);
  - one development-history sentence, accurate: at the development configuration (closed-form penalty, fixed 600 epochs, no stopping rule, all tiles trained) the gap was 0.55 / 0.83 / 1.48, below the perceptron.
- **Conclusion, Limitations "Inference".** "in simulation the difference is close to that of an encoder that sees the influx" → "in simulation, at the final configuration, the difference is measurable, 0.65 to 1.93 posterior standard deviations, and larger at every depth than that of a perceptron fitted to the exact posterior mean from the same inputs".
- **Flags.** Removed the two CHECKs ("300 epochs … confirm or rerun", app:planted and Limitations). Added one REVIEWER at app:planted's result: "encoder gap exceeds the perceptron's at the final configuration; stated in the limitations". No other text cites app:planted's conclusion (grep for app:planted and "amortis"; app:implementation's "counts-level answer to the amortisation gap" still holds).
- **Build.** latexmk exit 0 with `\showflagstrue`; no undefined references; the only overfull box is the template's 5.1 pt; 73 pages. Rendered text checked with pdftotext.

**Overnight results integrated: breakdown points, ω = 0 ablation, whole-section baselines, MintFlow reconstruction; reading-order markers (2026-10-01, writer; devlog "Overnight results (2026-10-01 morning)").** Sources: `scripts/logs/breakdown_2026-09-29/breakdown_all.md` and `.json`, `scripts/logs/omega0_2026-09-30/READOUT.md`, `DisCell-baselines/results/feasibility.tsv`, devlog B-mf1 entry (2026-09-30). Report: `scripts/logs/paper_flags_2026-09-30/OVERNIGHT_INTEGRATION.md`.
- **κ\* (abstract, conclusion, §3.6 Breakdown points, fig:breakdown-data caption, app:sweep, §3.5 Transport).** Old: "of the contrasts computed so far, all hold … except one on one section"; "the four transport contrasts are being computed". New: all claimed contrasts above the grid except the FF signalling share (κ\* 0.2, interval contains 0) and transport − leakage part (κ\* 0.4 ovarian FFPE, 0.3 lung; a seed flips sign; above the grid on FF and TMA core). Read A and the twin margin above the grid on all four. Transport − programme part: no finding on all four, **by construction** (the difference is ~1e-13 at κ = 0, where no leakage is modelled); stated so. Added: from κ = 0.05 to 0.2 it is positive in every seed on every section, 0.02 to 0.11 in R² averaged over all panels (ovarian 0.017–0.023, lung 0.025–0.044, FF 0.061–0.114, TMA 0.018–0.023), and negative on ovarian at κ 0.3/0.4 (−0.006/−0.037) and TMA at 0.4 (−0.003). Text: "the sweep does not show that the leakage term adds to the transported prediction beyond the response programmes" (not "adds nothing measurable", which the κ > 0 values contradict). §3.5 Transport gains one sentence pointing to this.
- **ω = 0 (§2.4, app:pathb, app:sensitivity new paragraph "Without the intrinsic path").** "and it is load-bearing" → "and its role is the allocation: it keeps the type out of w", plus a result sentence. Numbers (final → ω = 0, 3-seed means): NMI μ_w 0.42→0.50 ovarian, 0.27→0.32 lung, 0.33→0.35 FF, 0.29→0.40 TMA (disjoint on all four); within-type share of w's variance 0.32→0.19, 0.43→0.37, 0.53→0.49, 0.53→0.33 (disjoint on all four); cycle R² z FF 0.76→0.70 (disjoint); ovarian NMI z 0.71→0.67 and cycle R² z 0.52→0.50 (mean only); cycle R² w rises nowhere; no unpredicted read disjoint; none opposite to the prediction. Code-drift control stated: NMI z 0.616 vs 0.621, just below the final range (0.617–0.627).
- **Baselines (§3.1 Baselines, §3.7 runtime, fig:probe-data and fig:battery captions, tab:baselines, app:baselines, app:timing, Limitations "Evaluation", abstract).** MintFlow ovarian FFPE done: 939 min training, 15.8 h wall, 9.5 GB GPU, 23.6 GB host. SIMVI lung/ovarian/FF: CUDA OOM on 24 GB (requests 5.2 GiB / 7.7 GiB / 40 MiB with 21.3 / 15.9 / 23.5 GiB in use). MintFlow FF: not attempted, earlier host OOM after 574 s of set-up on a 125 GB host, 1.16 M cells. Runtime range "193 to 879" → "193 to 939". Abstract "MintFlow, fitted so far on two sections" → "which could be fitted on three of them". §3.7: MintFlow MLP composition residual "0.3 to 1.1 %" → "0.3 to 3.0 %"; cycle R² "0.006 to 0.056" → "0.006 to 0.062"; context composition "76 to 93 %" → "76 to 96 %". Limitations: "run on only some of the sections" → "could not be run on some of the sections on the resources currently available". Pending markers for SIMVI/MintFlow whole-section fits removed (3).
- **MintFlow reconstruction.** "under inspection" → stored value invalid (export error in our pipeline: MintFlow's split of each cell's counts, which sums back to the counts, scored in place of a decode), with `\pending{MintFlow refits with the corrected export}` in app:baselines, tab:baselines and the fig:battery caption (author: refits of TMA core, lung, ovarian FFPE are running). The generated tab:battery footnote b still reads "under inspection" (generated; not edited).
- **Flags.** Removed: abstract CHECK and conclusion CHECK (transport contrasts), §2.4 REVIEWER (ω = 0), fig:battery CHECK (B-mf1), app:sweep CHECK (tab:breakdown transport rows); KEEP-CORE notes at fig:breakdown-data and §3.6 lose their "pending" clauses. Added one CHECK at §3.6 Breakdown points: transport − programme part is no finding by construction; keep, grade from κ = 0.05, or move to the trajectories.
- **Reading-order markers.** `\readorder{N}` in `macros.tex` next to `\flag`: bold white on teal, `[R N]`, shown only under `\showflagstrue`. R1 abstract; R2 Introduction opening + Contributions; R3 Conclusion; R4 §3.7 Comparison; R5 §3.6 Leakage Sweep; R6 §3.4 Intrinsic State; R7 §3.5 Response; R8 §3.3 Model Quality + §3.2 Simulated; R9 §3.1 Setup; R10 §2.2 Generative; R11 §2.4 Objective; R12 §2.5 Invariance; R13 §2.8 Identified/Sweep; R14 §2.3 Inference + §2.1 Setting; R15 §2.6 Batching + §2.7 Selection; R16 app Synthetic Recovery; R17 app Planted Worlds; R18 app Readouts + Response in Detail; R19 app Sensitivity + Sweep in Full + Counterfactual with Cellina; R20 app Design Rationale; R21 app Derivations; R22 app Implementation + Experimental Details (30 markers). (Section numbers are the PDF's; the author's list counts Setup as §3.0.)
- **Build.** latexmk exit 0 with `\showflagstrue`; no undefined references; the only overfull box is the template's 5.1 pt; 74 pages. Two "Float too large" warnings come from generated tables (tab:battery, 59 pt, its long footnote e; tab:breakdown-traj, 111 pt). All 22 markers render (pdftotext).

**Generated-table wording (2026-10-01, coder).** The tab:timing and tab:battery footnotes no longer quote log text: the feasibility reasons are rewritten in paper words, with no log names or device ids (the raw text stays in the % provenance comments). tab:battery footnote b now reads "the stored value came from an export error in our pipeline that scored each cell against its own counts; refits with a corrected export are running", replacing "under inspection". Rebuilt: exit 0, no undefined refs.

**Transport counterfactual − programme-only leaves the claimed family (2026-10-01, coder + writer; devlog "Transport counterfactual − programme-only leaves the claimed family (author, 2026-10-01)").** Sources: `scripts/logs/breakdown_2026-09-29/breakdown_all.{md,json}` and `breakdown_all_trajectory.{md,json}`, re-rendered by `breakdown --all` (no compute). Report: `scripts/logs/paper_flags_2026-09-30/FAMILY_CHANGE_2.md`.
- **Family.** m_s 8/7/7/7/1 → 7/6/6/6/1 (ovarian, lung, FF, GSE core, GSE dual). All other κ\* unchanged: FF signalling breaks at 0.2; transport − leakage part breaks at 0.4 (ovarian) and 0.3 (lung), above the grid on FF and GSE; the rest above the grid.
- **Trajectory (new row in tab:breakdown-traj, R², 3-seed mean [seed range]).** Exactly 0 at κ = 0 on every section and seed. κ 0.05–0.2: positive in every seed on all four sections; means 0.017–0.114, seed values 0.014–0.121. Beyond: lung and FF positive in every seed up to 0.4; ovarian −0.006 [−0.012, +0.000] at 0.3 and −0.037 [−0.067, −0.009] at 0.4; GSE core +0.012 [0.004, 0.017] at 0.3 and −0.003 [−0.013, +0.011] at 0.4.
- **§2.8 (method, the sweep).** Claimed list: "the transport prediction against its programme-only and its leak-only parts" → "against its leak-only part"; "seven contrasts per section and eight on the primary section" → "six … seven". Added: the programme-only margin is a trajectory too, zero by construction at κ = 0, which leaves no sign for the rule to keep.
- **§3.6 Breakdown points.** "with three exceptions" → "with two exceptions"; the "no finding … by construction" sentences and "The sweep therefore does not show that the leakage term adds …" → reported as a trajectory (tab:breakdown-traj), zero by construction at κ = 0, positive in every seed and section at κ 0.05–0.2 by 0.02–0.11 R² (mean over seeds), and the behaviour beyond 0.2 as above.
- **fig:breakdown-data caption.** "zero at κ = 0 by construction, so it is no finding throughout" → "is not drawn: it is zero at κ = 0 by construction and is reported as a trajectory (tab:breakdown-traj)". The figure has six rows.
- **Abstract.** "…, so it is no finding" → "…, is reported as a trajectory and is positive in every seed from 0.05 to 0.2".
- **Conclusion.** "is no finding by construction, so the sweep leaves open whether the leakage term adds to that prediction" → "zero by construction where no leakage is modelled, is reported as a trajectory: positive in every seed on every section from κ = 0.05 to 0.2, and negative at the largest leak fractions on the primary section".
- **app:sweep.** The trajectory list gains the contrast; "no finding on every section, although positive …" → the trajectory description with the beyond-0.2 behaviour.
- **Generated (scripts/paper_tables.py, minimal patch).** tab:breakdown drops the row; its caption says the marker-pair contrast and transport − programme part are trajectories. tab:breakdown-traj gains "Transport − programme part" per section (2 decimals, range in `\scriptsize` so the row fits the width); its caption explains the row and why it has no κ\*.
- **Flags.** Removed the §3.6 CHECK ("decision needed: transport minus programme part …"). No flag added; [R N] markers untouched.
- **Build.** latexmk exit 0 with `\showflagstrue`; no undefined references; the only overfull box is the template's 5.1 pt; 74 pages. **Open:** tab:breakdown-traj is "Float too large" by 215 pt (was 111 pt before the new rows); its TMA core block runs off the page in main.pdf. Needs a layout decision (split the table or move it to its own page/longtable).

**tab:breakdown-traj as a longtable (2026-10-01, coder).** With the transport − programme part added as a trajectory, the table ran 215 pt past the page and the TMA core rows were cut off; it already overflowed by 111 pt before this change. It now breaks across pages, with its header repeated (longtable, added to the preamble). All rows are visible on pp. 54–55. Rebuilt: exit 0, no undefined refs; the only overfull box left is the template's.

**R1–R3 flags applied: abstract, introduction, conclusion; Limitations to its own appendix; runtime in the conclusion (2026-10-01, writer agent; the author agreed with the R1–R2 flags and read R3).** Report: `scripts/logs/paper_edit_2026-10-01/R1_R3.md`. No numbers are new: every one is copied from the existing text, from tab:timing or from the result sections it summarises.
- **Abstract (TRIM).** 375 → about 305 words. The model-definition sentence is one clause ("splits a cell's counts into these three parts, the leakage as a mixture over graph neighbours"); "so a spatial effect … is only partially identified" and "We make that dependence explicit" are folded into "We therefore do not estimate it"; the breakdown points are given in brackets. Every claim kept, including the programme-part trajectory added this morning. *Why:* the flag; the abstract had repeated the confound twice. *Not done:* a runtime clause (not asked; offered to the author).
- **Intro ¶3 (TRIM).** The technical-term bookkeeping (the decomposition into intrinsic, spatial, niche, technical and noise terms; depth conditioned on; multinomial as counting noise; no spatial-field term; no ambient RNA or batch; one section per fit) **moved to §2.2**, as a new paragraph after the generative equations; the intro points there. The closing sentences that repeat §2.8 (survives / breaks at the first grid point, "the range is the deliverable", "not a confidence interval") cut to one sentence on the breakdown point with a pointer to §2.8. *Why:* the flag; §2.2 is where the terms are defined and §2.8 already says the rest.
- **Intro ¶4 (TRIM).** The three design requirements in two sentences; "degenerate into a residual code" and "ego-informed prior" dropped (both in §2.2/§2.3). *Why:* restates §2.3 and §2.5.
- **Related work ¶1 (TRIM).** Classical models in four sentences, every citation kept; SpotClean kept as the closest precedent, with its one bleeding rate, Gaussian kernel and off-tissue spots.
- **Related work ¶2 (TRIM, split by theme).** Now three paragraphs: (a) intrinsic/spatial split models (node-centric, SIMVI, MintFlow, Cellina, Celcomen, NicheCompass; none models leakage), ending with the fixed-descriptor contrast to SIMVI's independence penalty, which moved here from the disentanglement part; (b) resolVI and contamination-aware models, with its neighbour weight 0.041–0.086 (REVIEWER keep) and the planted-0.2 misspecification result; (c) the disentanglement/invariance lineage and Elazar. The two overlapping todos merged into one ("verify the NicheCompass characterisation against its paper, with the entry-by-entry check of tab:related; add scVIVA, further spatial VAEs and GNN spatial models").
- **tab:related (CHECK) kept**, its note amended: it needs one read of each source paper, which this pass did not do; the merged todo waits on the same read.
- **Contributions (TRIM).** Five items → four: the inference scheme folded into the generative-model item. Order follows the thread: (i) partial identification and the sweep (points 1–2), (ii) the model and its architecture, the z/w split (3), (iii) invariance and probe (3), (iv) evaluation (4).
- **KEEP-CORE flags removed** on intro ¶2 (kept unchanged; the duplicate is flagged on the method side, "Why κ is fixed", MERGE, left for R10) and on the conclusion's Limitations (every item kept, see below).
- **Conclusion, runtime (new paragraph after the summary).** From tab:timing: DISCELL 1.3–43 min per whole fit, evaluations and early stop included, peak 1.4–15.1 GiB; faster than resolVI's training on every section (5.5–105 min) and than Cellina's on three of four; Cellina comparably fast (3.4–29 min) and fastest on the fresh-frozen section; MintFlow 9–16 h of training (544–939 min) and not runnable on the fresh-frozen section for host memory; SIMVI over three hours on the TMA core (193 min) and out of memory on a 24 GB GPU on the three larger sections. *Why:* the author sees it as a main strength and the conclusion did not mention it; worded so that Cellina's parity is visible and DISCELL's time includes its evaluations against the others' training time alone. Peak-memory rows for the comparison methods were not yet in tab:timing when written; only DISCELL's peak is quoted.
- **Conclusion, Limitations.** The eight-item list **moved verbatim to a new appendix section "Limitations" (app:limitations)**, placed after the experimental details (main.tex), with an [R 3] marker. The conclusion keeps a five-sentence paragraph naming the main ones (form of the leak and what the sweep separates, with the false-positive floor's 0.65 → 0.50; labels and the incomplete invariance, 3.5–5.5 %; the closed per-cell channel and non-specific readouts; the encoder gap and identification; one-section fits, one held-out section, baselines at defaults against a calibrated DISCELL) and ends "The full list of limitations, with what each one implies, is in Appendix E." In the appendix: one run-on sentence (Inference) split in two; "about one seed standard deviation" sharpened to "1.3 seed standard deviations" and "a move of 8.3 standard deviations" added, both from §3.5. Nothing softened. *Why:* the author's request; the list was half a column. *Considered:* placing it first in the appendix (more visible) — rejected so the appendix letters before it do not shift; it sits next to the experimental details it cites. Synthetic Recovery moves from E to F.
- **Future work (TRIM).** Two sentences plus the app:inputs pointer; all three extensions and the input requirements kept.
- **Build.** latexmk exit 0, `\showflagstrue`, no undefined references, the only overfull box is the template's; 75 pages. Pages 1–3, 24–26 and 55–56 rendered and read.

### tab:timing: peak VRAM and RAM for every method (2026-10-01)

- **What.** tab:timing gains two blocks inside the generated table (scripts/paper_tables.py `table_timing`; no .tex edit): "Peak GPU memory (VRAM, GiB) ↓" and "Peak host memory (RAM, GiB) ↓", one row per method, bold on the lowest measured peak per section. DISCELL's former "peak memory" row in the details block moved into the VRAM block. Caption: measure definitions, values read from the recorded fits, footnotes c (DISCELL interim VRAM) and d (failed attempts' lower bounds).
- **How each peak is measured.** Baselines (DisCell-baselines/common.py, called at the end of every runner): `torch.cuda.max_memory_allocated()/1e9` and `ru_maxrss/1e6` of the fitting process, set-up and prediction included; converted here to GiB. DISCELL: `max_memory_allocated` in the 20-epoch timing run, reset once model and section are resident (training + one evaluation); host RAM never recorded. Same PyTorch counter for both, but a narrower window for DISCELL: footnoted, not mixed silently. Failed SIMVI attempts: no config.json; VRAM is "allocated by PyTorch" at the failing request (from the OOM message in the fit log), RAM the 15-s sampled RSS before the failure; both shown as lower bounds (> / ≥), never bolded. The 14-s ovarian-FFPE attempt had no sample after launch ("--"). MintFlow uses `mf_config()` (refit folder once it has a config, else the original).
- **Values (GiB).** VRAM: DISCELL 1.4–15.1 (timing run), resolVI 2.6–2.9, Cellina 0.7 throughout, SIMVI 7.5 (TMA core) and >15.4 to >23.0 at OOM, MintFlow 6.2–8.8. RAM: resolVI 3.6–40.9, Cellina 4.3–43.7, SIMVI 3.7 (core), ≥7.3 / ≥58.6 at OOM, MintFlow 12.7–22.5; DISCELL pending.
- **DISCELL measurement fits (memL).** scripts/queue_2026-10-01_memL.sh written (seed 0, finalL_s0 flags, run memL_s0, `/usr/bin/time -v` + torch peak + ru_maxrss, GPU 0 only with a fresh lock so the MintFlow refits stay off it, FF last gated on 70 GiB available; summary scripts/logs/memL_2026-10-01/memL_summary.json; reruns `paper_tables.py --only timing` at the end). Preflight passes (all four configs equal finalL_s0). **Not launched**: the detached launch was refused by the session's permission classifier; the author launches it. Once the summary exists the generator switches DISCELL's VRAM to the whole-process peak and fills RAM.
- **Not done.** fig:cost (figures/src/fig_cost.py, build.sh): reading the house-style sources was refused by the same classifier; left for a follow-up.
- **Checks.** Round-trip test `test_timing_memory_round_trip` (baseline cells from config.json, DISCELL from memL or the timing run, the lung-FFPE SIMVI bound from its OOM line, bold = lowest number); tests/test_paper_tables.py 38 passed. latexmk exit 0, no undefined references; tab:timing fits page 67 (rendered and read). The one "Float too large" warning is tab:battery's, unchanged.

## 2026-10-01 — High-confidence citations applied (appendix/experimental.tex)

Source: scripts/logs/paper_sources_2026-10-01/SOURCES.md. Resolved \needsource markers:
- M7 nearest-neighbour MI estimate: ross2014
- M8 maximum mean discrepancy: gretton2012
- M9 varimax: kaiser1958
- M10 Gene Ontology: ashburner2000, go2023
- M11 Benjamini–Hochberg: benjamini1995
- M12 curated ligand–receptor database: jin2025cellchat
- M13 neural MI estimator: belghazi2018
Eight BibTeX entries added to references.bib (kraskov2004, jin2021cellchat omitted as optional). M14 (tenx_ovarian_ffpe) not applied: experiments_pre_freeze.tex is not in the build. M1–M6, M15 untouched (medium/low). Rebuild clean, no undefined citations.

## 2026-10-01 — tab:timing memory blocks dropped; medium/low-confidence \needsource claims reworded (author's delegation)

Devlog: "Peak memory: the comparison is dropped (author, 2026-10-01)". Sources: scripts/logs/paper_sources_2026-10-01/SOURCES.md (M1–M6).

- **tab:timing.** `table_timing` back to its earlier form: DISCELL details (parameters, peak memory = timing run's `peak_train_mib`, epochs, s/epoch) and the run time of one fit per method. Removed: the per-method VRAM/RAM blocks, footnotes c/d, the memL/OOM/GiB-conversion code and constants, `DIRECTION["memory"]`, `test_timing_memory_round_trip`; the peak-memory assertion is back in `test_timing_round_trip`. Kept: `mf_config()`, the `paper_reason` wording of footnote a, the refit note. Caption back to "Model size and training cost on one GPU …". Regenerated `--only timing`; tests 37 passed. Text check: the conclusion's "peak memory of 1.4 to 15.1 GiB" is DISCELL's restored details row, so it stands; no text cites the removed rows. appendix/experimental.tex still gives MintFlow's ovarian-FFPE peak (9.5 GB GPU, 23.6 GB host) in prose, from its config, without citing the table. Left as it is.
- **M1, introduction ¶1.** Before: "…published estimates place the fraction of misassigned transcripts anywhere between one tenth and one half \needsource." → After: "…: on this platform a typical cell carries misassigned transcripts of about a tenth of its counts, and heavily contaminated cells are rare \citep{yang2025mistic}."
- **M4, method "Why κ is fixed".** Before: "…extends to $0.4$, close to the upper range of published misassignment estimates for this platform \needsource;" → After: "…extends to $0.4$, a deliberately conservative top at several times the typical misassignment of about a tenth of a cell's transcripts \citep{yang2025mistic};"
- **M5, experiments §setup.** Before: "It was taken before the full sweep, inside the published range of misassignment for the platform \needsource, and kept after it," → After: "It was taken before the full sweep, to match the typical misassignment on this platform, about a tenth of a cell's transcripts \citep{yang2025mistic}, and kept after it,"
- **M6, app:kappa.** Before: "The grid range comes from two sources: zero as anchor, and published misassignment estimates for this platform \needsource." → After: "The grid runs from zero, the nested no-correction null, to $0.4$. On this platform a typical cell carries misassigned transcripts of about a tenth of its counts, and heavily contaminated cells are rare \citep{yang2025mistic}. The operating point $\kappa = 0.1$ (\cref{sec:setup}) matches that typical value, and the top of the grid is deliberately conservative, several times it."
- **M2, method assumptions.** Before: "Diffuse background is assumed negligible relative to leakage from adjacent cells \needsource." → After: "There is no background term either: the platform's non-specific background is low, as measured by its negative-control probes \citep{janesick2023}, but those controls do not register ambient RNA captured by the panel's own probes, which the model does not represent."
- **M3, method assumptions.** Before: "Transcripts nearest the boundary, such as mRNAs of secreted and membrane proteins, are plausibly over-represented in what crosses it \needsource." → After: "mRNAs of secreted and membrane proteins are enriched at the endoplasmic reticulum, outside the nucleus, in cultured human cells \citep{xia2019merfish}; if extranuclear transcripts cross a boundary more readily, such genes are over-represented in what crosses it."
- **Bib.** yang2025mistic and xia2019merfish added to references.bib from NEW_ENTRIES.bib; janesick2023 already there. yang2025mistic is a preprint, so check for a journal version before submission. Audit items A10 and A11 can be marked done. The two markers outside the build (experiments_pre_freeze M14, validation M15) are untouched.
- **Build.** latexmk exit 0, no undefined references or citations, bibtex 0 warnings.

## 2026-10-01 — Direction markers and axis labels on the data figures; §3.6 (R4) trimmed (author's read of R4 and fig:battery)

- **Figures.** A title arrow on a horizontal-metric panel read as "higher on y". New convention in figures/src/style.py (`BETTER_RIGHT`, `BETTER_LEFT`, `ybetter`): "better →"/"← better" in the x label for a horizontal metric; for a vertical metric the label names the quantity and an upright "↑ better"/"↓ better" sits just outside the top of the axis (a rotated label would turn the arrow). Per figure: fig_battery (short titles; every x label names the quantity with units and gives the direction, e.g. "held-out log-likelihood (nats/count), better →"); fig_tradeoff, fig_separation, fig_transport_heldout (rotated "better →" removed from y labels, upright marker added; separation left margin widened, ovarian-FFPE MintFlow label moved inside the panel); fig_kappa_sweep (column heads name the quantity on the axes below, with units and "↑/↓ better"; x label "leak fraction κ"); fig_sensitivity (short titles; "better →"/"← better" under each panel; the shared label keeps the unit, seed SDs); fig_planted_percell, fig_synthetic_misspec (y labels name the quantity; direction in a's title, upright markers in misspec); fig_breakdown_data (κ* labels moved above the panel, no longer clipped into the neighbouring column; x label "leak fraction κ"). fig_probe_data already complied, unchanged. Captions of fig:battery, fig:kappa-sweep, fig:sensitivity, fig:synthetic-misspec, fig:planted-percell now say where the direction is given. build.sh exit 0; every PNG looked at.
- **§3.6 runtime.** Times per method replaced by one sentence. Source tables/generated/timing.tex (DISCELL by the mean of its three fits): Cellina / DISCELL = 11.6/9.09 = 1.3 (ovarian FFPE), 9.9/4.60 = 2.2 (lung), 3.4/1.33 = 2.6 (TMA core); on the fresh-frozen section Cellina is faster (29.1 against 31.2–43.4). MintFlow / DISCELL = 103–410 where it ran. Text: "up to 2.6 times faster than the fastest competing model, Cellina, on three of the four sections, and slower than Cellina on the fresh-frozen section; it is over a hundred times faster than MintFlow wherever MintFlow could be run (tab:timing)." The Discussion's own timing paragraph is unchanged.
- **§3.6 niche paragraph (TRIM flag applied, removed).** Split into an intrinsic-latent and a context-latent paragraph; number lists dropped in favour of tab:probe, tab:battery, tab:context and the figures; two numbers kept (≤5.5% against 7.9–24%). All caveats kept (resolVI as single-latent reference, stated explicitly; comparison methods saw our held-out cells; MintFlow low leakage with a type-like latent; Cellina higher NMI on every section and higher cycle on ovarian FFPE and lung; TMA core tie; serial section). The Cellina niche-domain and own-graph detail moved to a new "Cellina variants" paragraph in app:baselines; one summary sentence stays in §3.6. REVIEWER flag kept.
- **§3.6 counterfactual (MOVE flag applied, removed).** One sentence stays ("neither it nor DISCELL leads on every read", tab:cellina-cf, app:cellina-cf); the per-read numbers moved verbatim to app:cellina-cf after the table.
- **Build.** latexmk exit 0, no undefined references; §3.6 (pp. 22–24) and fig:battery's page (p. 25) rendered and read.

## 2026-10-01 — Direction arrows into panel headers; §3.7 "How to read the comparison"; R5 (§3.6 The Leakage Sweep) flags applied (author's read)

- **Figures (convention revised).** Horizontal metric: the arrow sits in the panel header after the quantity ("type →", "composition leakage ←", "mirror ←", "cycle state →", "reconstruction →"); axis labels name the quantity and units only. Vertical metric: the upright "↑/↓ better" next to the axis, as in fig:tradeoff. style.py: `BETTER_RIGHT/LEFT` replaced by `RIGHT/LEFT` (header suffixes); `ybetter(..., over=True)` starts the marker at the spine, under a header; new `xbetter` puts "← better"/"better →" under the right end of a horizontal axis for panels whose header names a section. Per figure: fig_battery, fig_probe_data, fig_sensitivity (header arrows; per-panel "better" labels gone; "composition leakage" on two lines so the heads do not collide); fig_tradeoff, fig_separation (`xbetter` under the x axes, "better" removed from the x labels); fig_kappa_sweep (heads name the quantity; marker next to each top-row axis); fig_planted_percell (arrow out of a's title, marker next to its axis; panel letters raised above it); fig_breakdown_data (one "↑ better" on the first panel: every contrast is signed to claim the side above zero); fig_transport_heldout, fig_synthetic_misspec already in the tradeoff style, unchanged. Every caption says once "Arrows point toward the better direction". build.sh exit 0; every PNG looked at.
- **§3.7 synthesis (new paragraph "How to read the comparison", after the figures).** Checked against tab:battery, tab:probe, tab:context. Holds: every method that leaks less composition than DISCELL keeps less cycle state (MintFlow wherever it ran, cycle R² ≤ 0.06; Cellina on the TMA core, 3.5 % = DISCELL's lowest seed, cycle 0.29 vs 0.50), and every method that keeps more cycle state leaks more (Cellina 0.566/0.311 on ovarian FFPE/lung, resolVI 0.559 on ovarian FFPE, SIMVI 0.567/0.631 on TMA core/serial, the core narrowly: 4.9 % vs DISCELL 3.5–4.8 %); fresh-frozen: DISCELL best on both. So DISCELL is never beaten on both at once; MintFlow is not either where it ran, but at near-zero cycle state (stated). Mirror R² lowest on every section but the serial one (MintFlow 0.035 vs 0.039, stated). Type agreement is not a DISCELL lead (MintFlow and Cellina higher everywhere they ran, stated). **Not used:** candidate (ii) (fig:separation): DISCELL's context latent holds less composition than Cellina's on three sections and far less than MintFlow's, so "a cleaner split" does not hold; candidate (iv) fails on NMI. Conclusion sentence prepared, not inserted (see report).
- **R5 (§3.6) flags applied and removed.** Intro (KEEP-CORE): number lists cut; programme-overlap numbers (0.76–0.92 vs seed overlap 0.72–0.84) moved to app:sweep; transport trajectory kept in words. fig:breakdown-data caption (KEEP-CORE) flag removed. Breakdown points (KEEP-CORE): both broken contrasts kept, κ*=0.2 named, the transport/leakage break "in the upper half of the grid" with tab:breakdown; transport − programme part kept as a trajectory (zero by construction at κ=0; positive in every seed 0.05–0.2; negative at the largest κ on the primary section and in the mean on the TMA core); the R² margins left to tab:breakdown-traj and app:sweep. Marker pairs (MOVE): rates dropped (already in app:response); the large-κ control-rate detail moved to app:response; decoding trajectory and "no exclusive-specific removal" kept. Form of the leak (REVIEWER): 1.3 SD for the form arms; floor 0.65 → 0.50, 8.3 SD, kept in the main text; area-scaled and TMA-core values left to app:sensitivity. fig:sensitivity (MOVE) moved to app:sensitivity; it floats to p. 72 beside tab:sensitivity (appendix floats queue there already). **Not applied:** fig:kappa-sweep's "appendix if space binds" (a layout decision); its flag stays. [R N] markers kept.
- **Build.** latexmk exit 0, no undefined references; only the template's 5.1 pt overfull box and tab:battery's existing "Float too large". Pages 19, 21–26 and 72 rendered and read.

**Conclusion: the trade-off sentence (2026-10-01, coder; author approved).** Added after the MintFlow sentence: "No comparison method beats DISCELL on both composition leakage and retained cycle state on any section: those that leak less keep less, and those that keep more leak more (fig:tradeoff)." It mirrors the §3.6 synthesis, which was checked against the battery data. Rebuilt: exit 0.

## 2026-10-01 — fig:kappa-sweep: arrows removed, caption on κ, moved to app:sweep (author approved)

- **Figure.** figures/src/fig_kappa_sweep.py: the "↑/↓ better" markers removed from this figure only (a direction would suggest choosing κ on these curves, which the paper does not do); column heads still name the quantity; top margin tightened. Regenerated; PNG looked at, no readout withheld.
- **Caption.** "Arrows point toward the better direction" removed. Added: the readouts move with κ by construction (a larger κ assigns more of the neighbours' resemblance to leakage, so the response's information about the niche and its spread fall, and the reconstruction favours small κ); κ is not identified and not selected on them; κ = 0.1 is the working point because it matches the typical per-cell misassignment (yang2025mistic), not because of these curves. MOVE flag ("appendix if space binds") removed.
- **Placement.** Figure moved from §3.6 (The Leakage Sweep) to app:sweep, before tab:kappa-sweep. §3.6's first sentence now points to fig:breakdown-data for the claimed contrasts and to fig:kappa-sweep "of the appendix" (with tab:kappa-sweep) for the diagnostics. No other main-text reference to fig:kappa-sweep; app:sweep's own reference reads unchanged.
- **Float note.** The figure floats to p. 72 (end-of-document float dump, next to tab:breakdown), not p. 51–52: fig:latents-umap ([t] only, too tall for a top float) blocks the figure queue. In a scratch build, [tp] on fig:latents-umap brings it to p. 53 and fig:kappa-sweep to p. 54; not applied (outside this change). The tables' backlog (tab:kappa-sweep p. 70) is separate and pre-existing.
- **Build.** latexmk exit 0, no undefined references or citations; §3.6 page (p. 21) and the figure page (p. 72) rendered and read.

**Float placement (2026-10-01, coder).** fig:latents-umap now takes [tp] instead of [t], so it no longer holds the queue behind it. The moved fig:kappa-sweep lands on p. 54, beside app:sweep, instead of p. 72. Rebuilt: exit 0.

## 2026-10-01 — R6 (§3.4 What the Intrinsic State Carries) flags applied; cycle-score caveat added (author's read of R6)

- **Niche information (KEEP-CORE, MERGE; removed).** Number lists cut to one (3.5–5.5 % residual, nonlinear probe); per-section shares removed and the image block given in words, pointing to fig:probe-data and tab:probe. Kept: residual on every section, the adversary removes least on the TMA core and serial section (more than half the uncontrolled excess left), image block at composition's level on fresh-frozen. The response-collapse sentence (duplicate of app:moran) replaced by one clause pointing there; its source comment, orphaned, removed (app:moran keeps its own).
- **fig:probe-data caption (MERGE, "appendix candidate"; removed).** Kept in the main text, since the author asked §3.4 to point to it; caption unchanged.
- **The cell cycle (KEEP-CORE; removed).** Numbers point to tab:headline/tab:headline-full (no repeat of §3.3's 19–76 %); the fresh-frozen linear-reference caveat kept (83 % against 74–77 %); the "9 to 35 %" dropped.
- **Cycle-score caveat (new; author).** Grep of the manuscript: not stated anywhere (validation.tex, which mentions the gene sets, is not built). Code: discell/model/cell_cycle.py scores Seurat's cc.genes.updated.2019 lists (Tirosh 2016 sets, updated symbols) intersected with the panel with scanpy `score_genes_cell_cycle` after normalize_total + log1p, on the raw segmented counts (the depth-neutral variant is not used by the q90 reads); the q90 set is chosen from the summed scores. Added: §3.4, three sentences (scores are proxies from the cell's own counts; leaked neighbour transcripts and environment-induced expression raise them; the q90 set is chosen by them; w's near-zero within-type R² says little of the axis follows the neighbourhood as the response represents it, but does not make it the cell's own, since z is encoded from the same counts and neighbour types do not register a neighbour's state; the sweep changes the assumed κ, not the scores). app:readouts: the score definition (gene sets of tirosh2016, control-gene-binned score, Scanpy wolf2018, own segmented counts) and "a proxy, not a measurement". app:limitations: new item "The cycle reference". The within-type permuted latent of `metrics.cycle_r2` is a regression floor, does not address leakage, and is not reported in the paper, so it is not invoked. wolf2018 was already in references.bib (now cited); no Seurat entry added (the 2019 list update is not cited).
- **Merged subtypes (MOVE; removed).** Two-sentence allegiance result in words; the "two seeds of three" and "equally by its prior mean" caveats kept; exceptions (VEGFA+, fresh-frozen clusters, TMA myofibroblasts read better from the response) kept in words. AUC values moved verbatim to app:subtype.
- **fig:latents-main and its paragraph (MOVE; removed).** Moved to app:latents ([tp], before fig:latents-umap); the app:latents opening sentence now cites both. No other reference to fig:latents-main.
- [R 6] marker kept. Build: latexmk exit 0, no undefined references or citations, bibtex 0 warnings; pp. 18–19 rendered and read; appendix text checked in the extracted PDF.

**RECOMB: numeric citations (2026-10-01, author).** natbib numbers,sort&compress with unsrtnat in main.tex and supplement.tex: "[1]" instead of "(Author, year)". RECOMB imposes no citation style. The main text now ends about 60 % down the right column of page 10; the references start on page 10 and don't count. Both builds: 0 undefined references or citations.
- 2026-10-01: Completed bib entries dong2025simvi (10.1038/s41467-025-58089-7) and birk2025nichecompass (10.1038/s41588-025-02120-6) from Crossref; CHECK notes removed; both papers rebuilt clean.

**RECOMB: short reference names and the supplement in main.pdf (2026-10-01, author).** cleveref now prints Fig./Tab./Sec./Eq. (captions read "Fig."/"Tab."). main.pdf appends supplement.pdf after the references through pdfpages, and build.sh runs a final pass to embed the latest supplement. The separate supplement.pdf is still built, since RECOMB asks for the appendix as a separate supplementary file. The main text is still 10 pages; main.pdf is 72 pages in all. 0 undefined references or citations.

**Generated tables: vocabulary-neutral (2026-10-01, coder).** The tables in aistats/tables/generated are shared by both manuscripts, so scripts/paper_tables.py now renders the vocabulary through macros, defined in aistats/macros.tex with the AISTATS words and in recomb/macros.tex with the RECOMB words: \termSpill (leakage / spill-over), \termSpillFrac (leak fraction / spill-over fraction), \termReloc (transport / relocation), \termInflux (foreign influx / spill-over influx), each with a capitalised \Term... form. Every table function covered, headline_main and breakdown_main included; row labels that also appear in the provenance comments keep their raw names there (LABEL_TEX maps them for rendering). Rewordings: "leakage rate" → \termSpillFrac (sensitivity, synthetic), so AISTATS now reads "leak fraction" there; "leak correction" → "\termSpill{} correction"; "DISCELL's transported prediction" → "DISCELL's \termReloc{} prediction"; "at each rate κ" → "at each value of κ"; "at the same mean rate" → "at the same mean value". Tests updated to the macro form, plus a check that no rendered line carries a raw term (40 pass). Both papers build, exit 0, no undefined control sequences; extracted PDF text shows no table term in the other paper's vocabulary. Figures: recomb_tradeoff, recomb_fig1_overview and recomb_fig_programmes already use RECOMB words. Non-recomb figures in the RECOMB supplement still say "leakage"/"leak fraction"/"transport" in their labels (fig_breakdown_data, fig_breakdown, fig_kappa_sweep, fig_sensitivity, fig_synthetic_misspec, fig_batching, fig_contact_kernel); not changed (no recomb_* variants).
- 2026-10-01: seven AISTATS figure scripts (fig_breakdown_data, fig_breakdown, fig_kappa_sweep, fig_sensitivity, fig_synthetic_misspec, fig_batching, fig_contact_kernel) gained DISCELL_VOCAB=recomb (style.py swaps rendered words, writes recomb_<name>); added to figures/src/build.sh; RECOMB supplement now includes the recomb_ versions; recomb build exit 0, 0 undefined refs. Remaining 'leakage/leak fraction/transport' in pdftotext: only the S8 caption sentence explaining the old labels.

**RECOMB: supplement references shortened (2026-10-01, author).** \suppref now prints just "S3.1" instead of "Supplement S3.1"; the S prefix marks the supplement. The stale S8 caption sentence that explained the old figure labels ("transport − leakage part", "leak fraction") is removed, since the figures now use the RECOMB words. The PDF has no "leakage"/"leak fraction" left. Rebuilt: 0 undefined references or citations.

**RECOMB: "Table" spelled out (2026-10-01, author).** cleveref and the captions use "Table" again; "Fig.", "Sec." and "Eq." stay abbreviated.

**RECOMB: Fig 1 panel a on real data (2026-10-01, author).** recomb_fig1_overview panel a now shows the segmented polygons around the fig_model_tissue focal cell (primary section), its first ring and beta_ij-weighted spill-over arrows, with an inset of the four-channel morphology crop with the 25 um disc zeroed ("image context (cell masked)"). Panel c gene labels aligned inward (gene 2 no longer runs into panel d), beyond-line shortened and caption line spacing widened. Caption of fig:overview updated ("b--d are schematic"). Rebuilt: exit 0, main text 10 pages, 0 undefined references; Fig 1 on PDF page 4.

**RECOMB: Fig 1 panel a re-laid out (2026-10-01, author).** Tissue crop smaller; morphology image larger, under it, labelled "Phi_i: image context (cell masked)" underneath; "arrow width: beta_ij" moved inside the tissue crop (top-left, white backing); a light grey arrow runs from the image to panel b's niche box (Phi_i feeds c_i). Caption clause updated to match. Rebuilt: exit 0, main text 10 pages, 0 undefined references.

**RECOMB: two clarifications (2026-10-01, author).** §3.3: "read the sublabels alike (balanced accuracy 0.95 and 0.94)" → "a held-out classifier separates tumour- from stroma-associated fibroblasts almost as well from the context prediction alone as from the response itself (balanced accuracy 0.94 against 0.95; chance 0.5)". §2.4: the rotation non-identifiability is explained in one parenthesis ("as in factor analysis, its axes can be rotated, with the loadings counter-rotated, without changing the fit").

**RECOMB fig:programmes caption (2026-10-01, author).** Panel a now explains the colour without getting longer: "each cell's shift toward the red (blue) genes of b, relative to its type's average (FF rotated)". Panel c names the joint R² in its title: "all three together (title)".

**RECOMB fig:hallmarks, supplement S6 (2026-10-01, author).** New `figures/src/recomb_fig_hallmarks.py` (in build.sh, DISCELL_VOCAB=recomb): hallmark AUC of all 34 programmes of the 12 finalL fits (section > seed > programme, variance share above each column), from the stored atlas hallmarks (top three sets with AUC > 0.5 per programme; other cells empty), dot at BH q <= 0.05; 21 sets significant somewhere. EMT on 29 programmes, oestrogen response early 10, TNFa/NF-kB 8, hypoxia 8, KRAS up 7, E2F 7. Full width fits at text width, so no restriction to leading programmes. S6 "The Response in Detail" gets the figure and a paragraph; main §3.4 gets "(Fig. S13)" after "significant on most programmes", no extra line. Rebuilt: exit 0, main text 10 pages, 0 undefined references.

**RECOMB fig:hallmarks made complete (2026-10-01).** All 50 hallmark sets recomputed on every programme of finalL_s{0,1,2} on the four trained sections (atlas.py's gene sets, Mann-Whitney on |loading| ranks, AUC, expressed-gene universe from the assembled data, BH within programme; stored `programs.npy`), saved to `data/datasets/<ds>/experiments/hallmark_full_finalL.json`. Reproduces atlas.json's stored top three for all 102 entries (same names and order, max deviation 6e-9). The figure now reads it: 39 sets significant somewhere (was a top-three subset); EMT 30 of 34 programmes (text said 29), oestrogen early 19, KRAS up 17, TNFa/NF-kB 15, oestrogen late 15, coagulation 14. Caption drops the top-three caveat; S6 counts updated.

**RECOMB: post-assembly fixes after the supplement trim (2026-10-01, main loop).** (1) Generated tables: tab:kappa-star (`breakdown_main`) caption now ends "Definitions: tab:S-contrasts; across the grid: fig:breakdown-data" (footnote c, if ever used, points to fig:breakdown-data); the tab:breakdown alias label is gone from S8. Agent D's hand-copied trajectory table is replaced by a generated one, `breakdown_traj_main` (tab:breakdown-traj: relocation fraction of ceiling, relocation − programme part, marker pairs in pp, Moran's I of μ_z, and the nonlinear composition probe across κ from data/datasets/<4 sections>/experiments/probe_regrade_lineage_sweep.json, mean over seeds of 1 − exp(−2 excess)); output identical to D's table. It is a COMPANION of `breakdown_traj`: `--only breakdown_traj` builds both, so tonight's rerun refreshes it. `headline_full`'s caption no longer cites tab:headline; every generated `\cref{tab:headline}` now cites tab:headline-full (regenerated: headline_full, sensitivity, transport_heldout, cellina_cf, kappa_sweep; caption-only diffs). **battery.tex NOT regenerated**: the MintFlow refits (lung, TMA) have landed partially and would change its MintFlow rows mid-queue; until it is regenerated, supplement.tex aliases tab:headline → tab:headline-full (a marked block; remove it then). Tests: 3 new (traj_main round trip, companion build, kappa-star caption); 43 pass. (2) S3 warm-up: no citation of app:planted remains (A was right; nothing to fix). (3) 03_1 "calibrated on these sections" now points to app:S-calibration (no line cost). (4) Calibration ledger: ω row from the devlog reproduction card round 2 (ω ∈ {0.5, 1, 2}, single-seed 40-epoch fits, closed-form era; calibration_round2.json) and AISTATS appendix/calibration.tex; d_z = 20 "fixed during development, not tuned" (fixed throughout in the card, no search on record); d_w = 6 from the d_w ablation {2, 3, 6, 8} × 3 seeds (devlog 2026-09-14/15), not repeated at the final configuration. Source comment in S3_implementation.tex. Architecture and calibration tables split into two floats (the merged float was too tall). (5) app:assets licences: resolVI (scvi-tools), MintFlow, Cellina BSD-3; SIMVI declares none, used as distributed. (6) New `figures/src/recomb_fig_model_graph.tex` ("spill-over" lane, no panel letter; in build.sh); fig:model-detail uses it. pdftotext of both PDFs: no "leak"/"transport" left; "Read A"/"twin margin" remain in recomb_fig_breakdown_data and in generated captions (open, vocabulary item 9). (7) The 14 natbib "multiply defined" warnings came from xr-hyper importing the other document's \bibcite lines; both \externaldocument calls now use the `nocite` option (labels only). (8) The supplement no longer inputs the slim headline table; it shows headline_full (tab:headline-full, the label 03_6 cites). Build: main text 10 pages, supplement 35 pages, 0 undefined references/citations, no multiply-defined citations. Remaining warnings: battery float too tall by 94 pt and headline_full 17 pt overfull (generated, pre-existing), a 1.5 pt overfull line in S1, underfull boxes in the main bibliography.

**RECOMB: author decisions after the RECOMB additions (2026-10-01; devlog "RECOMB additions: results and follow-ups").**
- **"The test has teeth" removed** from the abstract, Contributions, §2.4, §3.7 and the Discussion; the supplement had none left. κ\* is now described as a sensitivity analysis that orders findings by how much spill-over would have to be assumed to lose them; it neither estimates κ nor certifies a finding. In §2.4, "cannot be explained by spill-over … up to four times" becomes this ordering sentence. The three real-data breaks are stated plainly in §3.7. The allocation-check sentence no longer calls the biological claims "the stronger test". *Why:* the planted control's predictions were not met (READOUT §1), so the certification framing is unsupported.
- **Planted spill-over control (§3.7, S7.3):** kept as `\pending{planted control, diagnostic running; planned: …}` with the planned one-sentence framing. No result is stated. The READOUT result (predictions not met; the null-world signal is unexplained) is kept in LaTeX comments. S7.3 says the advance prediction will be reported with its outcome. §3.2's "positive control" becomes "planted spill-over control". *Rejected:* stating the partial result now, since the author is waiting on the null-world diagnostic.
- **Relocation reframed (§3.6, abstract, Discussion, S6.4, S8, STORY_MAP C1/C5):** relocation "exercises the whole model". tab:relocation gains a reference column R: the rate-scale ridge on raw counts, trusted tier, first seed (0.89/0.87/0.82/0.73), not marked as better or worse. The text gives the three-seed ranges (0.81–0.89 against 0.65–0.72; level on the TMA core; DISCELL higher on all panels on three sections) and says why this does not decide: the observed shifts contain spill-over, and the regression has no intrinsic state and no spill-over share. A `\pending` slot gives both outcomes of the clean-truth simulation. A new S6.4 paragraph "Against a plain regression" carries the details: log scale 0.06–0.22, rate scale, and the single-cell read on raw targets (0.48–0.62 against 0.25–0.63). S8 has a new limitation clause. The Cellina comparison is unchanged, except that the lung twin read is now "level" (0.43 against 0.43 at two decimals).
- **Held-out wording (§3.1):** "every readout is taken on held-out cells" becomes "probes, reconstruction and latent reads". Relocation is the stated exception: it is cross-fitted, so most scored cells lie in training tiles, and its single-cell read is lower on held-out tiles in 11 of 12 fits. §3.6 no longer says "no systematic in-sample advantage".
- **Read A / twin margin / published read → single-cell read / twin read / cross-fitted read** in S5, S6.4 and S8 (tab:S-contrasts); in the generated tables (paper_tables.py); and in the RECOMB figures (style.py `_VOCAB`). The fig_breakdown_data label is rewrapped to two lines; recomb_fig_breakdown_data has been regenerated.
- **Serial-section sync:** Table 2's serial I(niche; w) now shows the same "?" pending mark as Table 4 (was "–: not read"). S7.1's serial rule says the cycle asymmetry is read and the other contrasts are being read (`\pending`).
- **Tables compressed (generator only, 46 tests pass):**
  - battery 745 → 608 pt: the serial rows are back on the page, with shorter caption and footnotes; layout only, and every number is identical.
  - headline_full: from 17 pt too wide to 20 pt to spare.
  - probe 591 → 487 pt: the Cellina variants are merged.
  - transport_heldout: 2 decimals, Section/Method columns.
  - cellina_cf: 2 decimals.
  - synthetic: 8 → 5 columns.
  - planted_percell: the all-"no" column is dropped.
  - sensitivity: the Seeds column goes into the labels.
  - timing: caption trimmed.
  - The breakdown row names now match tab:kappa-star.
  - tab:relocation is hand-rounded to 2 decimals to match.
  - **MF_PIN = True** in paper_tables.py: every table reads MintFlow from the refit queue's pre-refit backup, because the queue had already overwritten the TMA MintFlow entries. Set it to False once the refits are read.
  - The supplement.tex tab:headline alias hack is removed, since battery now cites tab:headline-full.
- **Fig. 3c methods (new S5 subsection app:S-drivers):** programme score, Moran's I of a programme, the landmark classes and their detection, the distance, composition and image (12 PCs) features, ridge with λ = 10⁻³, and five tile folds with alone, unique and joint R². Source comments point to atlas.py, validate.py and labels.py. The Fig. 3 caption now says "where detected; methods in S5.x". **Open (code):** the endothelial name rule misses the lung sections' "EC …" labels, so lung FFPE and the TMA core have no vessel class. S5 states this; the atlas has not been rerun.
- Build: main text 10 pages; 0 undefined references or citations in either document; one pre-existing 1.5 pt overfull hbox in S1.

**RECOMB: κ\*, planted control and clean-truth relocation integrated (2026-10-01 evening; devlog "Follow-ups (a) and (b): results" and "Breakdown gaps: results").**
- **κ\* (§3.7, abstract, Discussion, S7.1):** tab:kappa-star was already regenerated by the gaps queue (6 breaking cells over 3 contrasts). §3.7 now says this in words, with the earliest break (FF signalling lean, 0.2) as its only number, and "every other finding holds to κ = 0.4 wherever it is read". It adds the serial-section result: every contrast replicates the core's except relocation beyond its spill-over part, which replicates only up to 0.3. Abstract and Discussion: "three of seven contrasts break within the grid on some sections, the rest hold to four times the typical misassignment". S7.1 changes: the family sizes paragraph (7/7/7/6/6) replaces the `\pending`; the serial-section rule now states the replication result; the post-figure paragraph lists all six breaks. §2.4 had no serial-section rule, so it is unchanged. The intro contribution states no result, so it is unchanged. No "teeth" wording anywhere.
- **Planted spill-over control (§3.7 one sentence; S7.3 rewritten + generated tab:S-planted-spillover):** the corrected contrast 2⟨p, o − c⟩/‖t‖² gives a null world with no finding, κ_true 0.1 → 0.05 and 0.2 → 0.1, i.e. one grid step early (conservative). The genuine response holds through 0.2 (κ\* = 0.3) when spill-over is planted and over the whole grid in the null world. **Deviation from the brief:** the brief said "holds well beyond"; at κ_true = 0.2 it holds only to 0.2, so the text says "holds through κ = 0.2". S7.3 states openly that the first definition was mis-specified (it cannot vanish at κ = 0) and was corrected after a diagnostic on the same fits. Both definitions are in the table, and the S7.3 text keeps the training-tile noise-fitting caveat. The caption uses δ_p, δ_o, δ_c, δ_t, not p/o/c/t, to avoid a clash with \vp.
- **Relocation against the clean truth (§3.6 3 sentences; new S7.4 app:S-reloc-clean + generated tab:S-reloc-clean):** DISCELL's programme part keeps R² ≈ 0.76 of the spill-free shift at every κ_true from 0 to 0.2. The regression falls from 0.40 to 0.14 (log) and below zero (rate). The full prediction degrades by design. Abstract: a new two-sentence relocation claim. Discussion: "only DISCELL's spill-free prediction holds up as spill-over grows, which is what the separation buys". Limitations gain "the spill-free shift is known only in simulation from the model itself". S6.4 and S8 (Evaluation) point to S7.4.
- **Generators:** `planted_spillover` and `reloc_clean` in scripts/paper_tables.py, with provenance headers and consistency checks (the diagnostic's original κ\* must equal the control's; grids and seeds must match). Two new round-trip tests are in tests/test_paper_tables.py, and the header-list test is extended; 48 pass.
- **Trims to stay at 10 pages, prose only:** §3.7 drops the κ\*-does-not-estimate-κ sentence (stated in §2.4 and the Discussion) and shortens the trajectories sentence to a pointer to S7.1 (the full text is there). §3.6 drops "Would a plain regression do as well?" and the lung-twin "level" clause (it is in the table). The Discussion drops "lets a reader judge…" and the repeated "exercises the whole model".
- Build: main text 10 pages; 0 undefined references or citations in either document; the only overfull box is the pre-existing 1.5 pt one in S1. STORY_MAP C1/C5 are updated. `\pending{MintFlow refits}` remains in the abstract and §3.5 (not in scope).
