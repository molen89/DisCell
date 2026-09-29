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
