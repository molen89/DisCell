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
