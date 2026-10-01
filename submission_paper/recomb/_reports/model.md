# Report: model (sections/02_model.tex), 2026-10-01

## What I wrote
§2 Model. It has a one-sentence lead and Fig 1, followed by four subsections.
- **2.1 Generative model** (`sec:generative`, `sec:setting`). The data and the Delaunay graph. The model is eq:gen-z … eq:gen-x, with underbraces labelling the intrinsic, response and spill-over parts. Then come the meaning of a, B, m_ψ and σ_w, and why the decoder takes no type input. **Niche descriptor** paragraph: eq:context-gat, the type-queried GATv2 over neighbour types ⊕ masked KRONOS embedding ⊕ isolation flag, and why no distances enter. **Spill-over** paragraph: the fixed kernel β, the contrast with resolVI's per-cell weights, and the modelled form (one hop, section-wide, gene-agnostic).
- **2.2 Inference and objective** (`sec:inference`, `sec:objective`, `sec:batching`). The posteriors. The objective is a single equation (eq:objective), with the per-count reconstruction written as r_i to keep it short. Then one sentence on the intrinsic path, with the ω = 0 ablation result pointing to the supplement. Then the stop-gradient and tile batching in one sentence, and the operating weights.
- **2.3 Conditional invariance** (`sec:invariance`, `sec:selection` on the probe paragraph). The target (eq:invariance-target) and why it is conditional on type. The adversary (eq:adv), **crediting MintFlow's per-type critics** and naming what is new: the image target and the independent test. Then the held-out probe (eq:probe), a **bold definition of residual niche signal** (the excess over the within-type permutation floor, reported as 1 − e^{−2·excess}), and the checkpoint rule.
- **2.4 What is identified** (`sec:sweep`). The plain-language intuition comes first. Then Proposition `prop:kappa-bound` (proof in the supplement), worded per the publishability critique: "even with the neighbours' clean profiles known". It says the bound is not computed on our sections. One sentence ties in the simulation (sec:simulation) and one gives the symmetries (app:symmetries). **The sweep** is framed as a sensitivity analysis in the tipping-point/E-value tradition (Rosenbaum; VanderWeele & Ding, new bib entry; Manski). Then Definition `def:breakdown`, and what κ\* > 0.4 rules out. The rule for contrasts without a sign at κ = 0 (trajectories) is stated generically; the specific reclassifications belong in §3.7. Last comes the transferability sentence.

**Fig 1** (`fig:overview`) is a new script, `../aistats/figures/src/recomb_fig1_overview.py`, registered in build.sh. It produces a vector PDF at 6.5 × 2.05 in with four panels:
- a: a cartoon of spill-over in tissue (jittered Voronoi cells, β arrows, image-mask disc);
- b: a block model (z, w, prior, ρ, influx, mixture at κ, adversary);
- c: the κ-bound simplex, as fig_kappa_bound panel a;
- d: a schematic breakdown point at κ\* = 0.3.

Everything in it is schematic. It needs a visual polish pass: some labels are tight in panel b, and the β_ij label in panel a overlaps a cell.

## Pages
Private build (`_build_model`): §2 runs from about 70 % down the right column of main page 1, through pages 2 and 3, to about 15 % of the left column of page 4. That is **≈ 2.25 p** against a budget of 2.25 p, Fig 1 included. The text is about 1,980 words including the LaTeX source. No overfull boxes; all of my references resolve. Note on building in an outdir: bibtex cannot find `../aistats/references` from `_build_<name>/`. I ran bibtex on a copy of the .aux with the path rewritten. build.sh, which runs in place, is unaffected.

## Numbers and their sources
All are configuration values or results already stated in the AISTATS method or the supplement:
- 40 µm pruning; τ = 20 µm; 128 µm crop; mask radius 25 µm: aistats method.tex §2.1–2.2 and S2 app:image.
- d_z = 20, d_w = 6, ω = 1, α_z = ½/ℓ̄, α_w = 0.1, λ_y = 3, 90 % NMI guard: aistats method.tex (tab:learned, Scaling, adversary, selection); S3 tab:architecture (ω = 1).
- κ grid {0, 0.05, 0.1, 0.2, 0.3, 0.4} with three seeds: aistats §2.8; S8 (three seeds).
- About a tenth of transcripts misassigned (MisTIC): S2 app:kappa; devlog 2026-10-01.
- ω = 0 result ("agreement of the response with type rises on all four sections"): S8 app:sensitivity "Without the intrinsic path"; devlog 2026-10-01.
- The probe share 1 − e^{−2·excess}: devlog 2026-09-30 (author decision).

## Open \pending
None in §2.

## Requests
**Supplement (material moved out of the AISTATS method):**
1. **S1 app:bound** (or S3): add the isolated-cell renormalisation (AISTATS eq:renorm and its paragraph), the Poisson-exactness argument and "why a multinomial". My text points to `app:bound` for "an isolated cell receives none".
2. **S1 app:symmetries**: host Proposition "Symmetries of the response" (AISTATS prop:symmetries) and the gauge discussion that follows it, under a new label, for example `prop:S-symmetries`. Retarget the supplement's `\cref{prop:symmetries}`. §2.4 cites `app:symmetries` for "identified at most up to rotation and per-type offset".
3. **S1 app:gaussian-mi**: host the closed-form Gaussian penalty (AISTATS eq:penalty and its estimation details) and the escalation rule. The supplement's two `\cref{eq:penalty}` must be retargeted, since the main text no longer has the penalty.
4. **S2 / S3**: the AISTATS paragraphs not carried over:
   - "What the leakage term assumes" → S9 limitations;
   - "What the decomposition means" (subclones) → S2 app:conditional;
   - "Why κ is fixed" → S2 app:kappa;
   - the type-not-z query and the dropped neighbour-mean input → S2 app:mirror;
   - the direct path into q(w) (explaining away) → S2 app:qw;
   - learned and fixed variances → S2 app:variances;
   - the Scaling paragraph (α ladder, surrogate status, (1+ω) factor) → S1 app:pathb or S3;
   - the image-niche k-means target and the λ_y = 3 justification → S3;
   - two hops in one pass, and the held-out neighbour exposure → S3 app:tiles;
   - the KL ramp, the five diagnostics and budget/seeds → S3 app:architecture;
   - the full contrast list, Bonferroni and serial-section rules, and "what w is not" → S8 app:sweep / S5.
5. **Supplement references to retarget:**
   - S2 app:image cites "\cref{fig:overview}b" for the image crop. The new Fig 1 has no crop panel, so retarget it to fig:mask-radius, or move fig_model_tissue into S2 as a new figure.
   - Every `\cref{sec:batching}` now lands on §2.2. If a dedicated target is wanted, point it to app:tiles.
   - The AISTATS-era `tab:learned` is gone.
6. **S2 app:kappa**: its proof environment stays where it is, and prop:kappa-bound is defined in §2.4.

**Other writers:**
- **03_7 (robustness):** state there, in one clause, that the marker-pair and relocation-minus-programme contrasts were reclassified as trajectories, and why (publishability critique). §2.4 gives only the general rule.
- **03_2 (simulation):** §2.4 says that recovery barely changes with the assumed κ (sec:simulation). Please keep that result in 3.2.

**Bib:** appended `vanderweele2017` (E-value; Ann Intern Med 167(4):268–274, doi 10.7326/M16-2607) to ../aistats/references.bib.

**Main build:** natbib warns that some citations are "multiply defined", from xr importing the supplement's bibcites. This is harmless, but worth a check in build.sh.
