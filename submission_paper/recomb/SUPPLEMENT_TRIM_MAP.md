# DISCELL RECOMB supplement: trim map (audit, 2026-10-01; no file edited)

Sources read: STORY_MAP.md, STYLE.md, CRITIQUE_publishability.md, `sections/*.tex`, `supplement/S1–S9`, the generated tables the supplement inputs, `supplement.aux/.log`, `_reports/*.md`. Page numbers are from the current `supplement.pdf` (59 pages).

## One-page summary

**Current: 59 pages. Target: about 27 (range 25–29), −54 %.**

**The biggest structural problem is not text, it is float placement.** Pages 1–39 hold text and figures, 40–42 the bibliography, and **43–59 (17 pages) hold 26 tables flushed *after* the bibliography**, mostly one per page, 20–40 pages from the text that cites them. A reviewer who follows "Table S6" lands at the back of the document. Fixing placement (per-section `\FloatBarrier` via `placeins`, `[!htbp]`, looser `\floatpagefraction`/`\topfraction`) saves pages and is the single largest gain for navigation.

| Current section | Pages now (incl. its flushed tables) | → New home | Target |
|---|---|---|---|
| (new) Guide: claims → where to check | 0 | front matter | 0.4 |
| S1 Derivations | 4.5 | **S1 Model: notation, derivations, proofs** | 3.0 |
| S2 Design rationale | 9.0 | merged into **S2 Implementation, design choices and calibration** | 1.5 |
| S3 Implementation | 6.5 | (same S2) | 2.0 |
| S4 Data, labels, comparison methods | 7.5 | **S3 Data, labels and comparison methods** (gains tab:probe) | 4.0 |
| S5 Readout definitions | 4.0 | **S4 Readout definitions** | 2.75 |
| S7 Simulation and planted tests | 5.5 | **S5 Simulation** (moved before tissue results, as in the main text) | 2.5 |
| S6 Full results | 8.5 | **S6 Results on tissue in full** | 4.0 |
| S8 Sweep, sensitivity, ablations | 8.0 | **S7 Spill-over sweep and sensitivity** | 4.0 |
| S9 Limitations | 1.5 | **S8 Limitations** | 1.0 |
| Bibliography | 3.0 | (fewer citations after cuts) | 2.0 |
| **Total** | **59** | | **≈27** |

**Biggest cuts (pages saved, approximate):**
1. **S2 design rationale, 9 → ~1.5 (−7.5):** cut 7 of its 9 floats (fig:graph-compare, fig:voronoi-face, fig:contact-kernel, fig:kappa-bound and fig:breakdown, which duplicate main Fig. 1c/1d, fig:mask-radius, fig:kronos-umap), the three-routes table, app:variances and app:amplification; keep the design arguments as one short "design choices" subsection with tab:contact and tab:masking.
2. **S6 full results, 8.5 → 4 (−4.5):** cut fig:transport-heldout (same data as its table), tab:kl-summary + fig:kl-seeds (one paragraph keeps the negative result), tab:moran (duplicates main Fig. programmes d), the Cellina-cf prose that repeats §3.6, and the long marker-pair/localisation/tumour-axis paragraphs (condensed, negatives kept).
3. **S8 sweep and sensitivity, 8 → 4 (−4):** cut tab:breakdown (identical rows to main tab:kappa-star), fig:sensitivity (same values as tab:sensitivity), one of fig/tab kappa-sweep; condense the 1-page "sweep in full" paragraph.
4. **S3 implementation + S1 closed-form penalty (−4):** cut tab:deviations (the author's own `\todo` says it duplicates the method), tab:deviations2, fig:probe (schematic of the main-text probe definition), fig:batching, and the Gaussian-MI penalty with its estimation details (not the operating configuration: history).
5. **S4 data/labels + S7 simulation (−6.5 together):** cut tab:lineage-clusters (cluster IDs mean nothing without the clustering), tab:context (no claim uses it), tab:inputs and "Other assays (untested)", fig:planted-percell (same as its table), the amortisation-gap simulation bullet list (one paragraph), the spill-over-subtracted encoder history.

Also cut **tab:headline** (same reads as tab:headline-full, which the main text points to) once the generator retargets the six generated captions that cite it.

**Proposed new order and titles** (follows the main text: §2 → S1–S2, §3.1 → S3–S4, §3.2 → S5, §3.3–3.6 → S6, §3.7 → S7, §4 → S8):

- **Guide** (untitled, before S1): one paragraph plus a 7-row table "Claim C1–C7 → main-text evidence → supplement item to check".
- **S1 Model: Notation, Derivations and Proofs:** notation table (new); the bound, isolated cells, why a multinomial; the intrinsic path, weight scaling and status of the objective; proof of Proposition 1; symmetries of the response.
- **S2 Implementation, Design Choices and Calibration:** architecture table; calibration ledger (new: what was tuned on which section by which read); design choices in brief (type query and mirror, no edge geometry, Voronoi kernel, image mask, direct path into q(w), fixed σ_w); training (warm-up, diagnostics, seeds, dead channel); tiles and halo, including held-out exposure.
- **S3 Data, Labels and Comparison Methods:** sections (table, figure); lineage labels (ovarian mapping table); comparison methods (configuration, label use, reconstruction comparability (new), Cellina variants, MintFlow export disclosure); tab:probe and tab:battery; assets and licences; what the model needs (one paragraph).
- **S4 Readout Definitions:** the full headline reads (tab:headline-full); diagnostics (mirror R², I(niche; w), cycle scores); niches; relocation (panels, ceiling, interval coverage; single-cell read, twin); programmes and hallmark/GO tests; external criteria (signalling share, tumour axis, marker pairs); merged-subtype protocol.
- **S5 Simulation:** recovery and misspecified κ; the planted per-cell response; the amortisation gap.
- **S6 Results on Tissue in Full:** what the two latents carry (niche information of both, without the adversary); merged subtypes (results); the type-level response (recon-modes, per-cell divergence); programmes (hallmarks, localisation, signalling genes, tumour axis); relocation (Cellina head-to-head, held-out tiles); latents on every section; run time.
- **S7 Spill-over Sweep and Sensitivity:** the claimed family and the rules; contrasts across the sweep (fig:breakdown-data); trajectories; diagnostics across κ; the planted spill-over control (new, pending); sensitivity to assumptions (form, false-positive floor, response weight, adversary, ω = 0).
- **S8 Limitations.**

---

## Pointer inventory (main text → supplement)

Everything below must survive, possibly condensed. `file:line` is the line in `sections/` (or the generated main-text table).

| Supplement target | Main-text sentence (gist) |
|---|---|
| app:bound | 02_model:41 "…an isolated cell receives none" |
| app:graph | 02_model:38 "No distances enter, so that the attention cannot rediscover the spill-over kernel" |
| app:tiles | 02_model:57 "batches are contiguous spatial tiles with a two-hop halo" |
| app:sensitivity | 02_model:57 ω = 0 ablation; 03_1:12 "weights were calibrated on these sections"; 03_7:18 spill-over forms and false-positive floor |
| app:architecture | 02_model:57 "remaining settings"; 02_model:84 checkpoint selection |
| app:kappa | 02_model:95 "The proof is in…" |
| app:symmetries | 02_model:95 "identified at most up to a rotation … and a per-type offset" |
| app:sweep | 02_model:104 "shown as a trajectory instead" |
| app:sections | 03_1:6 "69,000 to 1.16 million cells" |
| app:synthetic, tab:synthetic | 03_2:5 simulation set-up |
| fig:synthetic-misspec, app:planted | 03_2:11 wrong κ costs little; encoder posterior mean off the exact one |
| tab:probe | 03_3:22 nonlinear probe 3.5–5.5 %; 03_5:10 trade-off caption; headline_main caption |
| app:response | 03_3:25 "85 to 90 % of what the two latents carry together" |
| app:moran | 03_3:25 "Without the adversary the response nearly vanishes within types" |
| app:subtype | 03_3:28 double dissociation (balanced accuracies) |
| app:recon-modes, app:planted-percell | 03_3:31 type-level response |
| fig:hallmarks | 03_4:19 "EMT … significant on most programmes" |
| tab:battery | 03_5:10, :14 (NMI of MintFlow/Cellina above DISCELL), :20 (reconstruction) |
| app:baselines | 03_5:18 Cellina variants (niche domain, own graph) |
| tab:timing | 03_5:20 |
| tab:cellina-cf | 03_6:9 (tab:relocation caption, intervals), 03_6:29 (parts not additive) |
| app:readouts | 03_6:27 panels, noise ceiling |
| tab:headline-full | 03_6:29 "0.65 to 0.73"; headline_main caption |
| tab:transport-heldout | 03_6:31 held-out tiles |
| tab:breakdown | breakdown_main caption "Full table" |
| app:limitations | 04:11 "The full list is in…" |

**Not pointed to from the main text:** app:pathb, app:posterior-reg, app:gaussian-mi, app:amplification, app:mirror, app:image, app:qw, app:conditional, app:variances, app:deviations, app:probe, app:lineage, app:assets, app:inputs, app:cellina-cf, app:latents, app:kl-maps, app:timing, and every float not listed above.

---

## Per-item table

Roles: **SC** supports-claim (C1–C7), **RD** reviewer-defence, **DUP** duplicate, **HIST** history/design path, **ORPH** obsolete/orphan. Pages are approximate and include flushed floats.

### S1 Derivations (4.5 p → S1, 3.0 p)

| Item | p | Main pointer | Role | Action | Reason |
|---|---|---|---|---|---|
| (new) Notation table | – | – | RD | **add** (~0.4 p) | STORY_MAP promises "a notation table in the supplement"; adapt the STYLE.md table |
| S1.1 app:bound: Jensen bound, eq:jensen, eq:gauss-kl | 0.6 | 02_model:41 | RD | condense to ~0.4 p | standard ELBO steps; keep the bound and the stop-gradient fixed-point sentence |
| S1.1 Isolated cells, eq:S-renorm | 0.3 | 02_model:41 (lands here) | RD | keep | the pointer's target |
| S1.1 Why a multinomial | 0.35 | – | RD | condense to ~6 lines | reviewers ask "why not NB"; also the basis for the reconstruction-comparability sentence (missing, see below) |
| S1.2 app:pathb: second bound, (1+ω) factor, gap | 0.5 | – (main §2.2 cites Slavutsky) | RD | condense to ~0.3 p | keep the two-bound statement and gap; cut the Slavutsky comparison detail |
| S1.2 "Why the intrinsic path is load-bearing" | 0.25 | – | DUP of main §2.2 + S8 ω = 0 | condense to 2 sentences | main text already states it, with the ablation |
| S1.2 Scaling of the weights | 0.6 | – | RD (calibration, surrogate status) | **condense and move its α ladder facts to the new calibration ledger (S2)**; keep "J is a surrogate; no likelihood-based uncertainty" | reviewer-critical: honest status of the objective |
| S1.3 app:posterior-reg | 0.2 | – | RD | merge into S1.2 as 2 sentences | |
| S1.4 app:gaussian-mi: penalty, MI identity, estimation (EMA, shrinkage, straight-through, jitter) | 1.0 | – | **HIST** (not the operating configuration) | **cut**, except the next row | the adversary is used in every reported fit |
| S1.4 Escalation rule and its development threshold | 0.3 | – | RD + **honest negative** | move to the calibration ledger (S2), ~5 lines | must keep: "at the operating point the nonlinear probe exceeds the quarter-of-excess threshold on composition on three sections and on the image block on the TMA core; the residual is therefore reported descriptively" |
| S1.5 app:symmetries: prop:S-symmetries + proof | 1.0 | 02_model:95 | RD/SC (C4b rotation caveat) | keep the proposition and proof; condense the long commentary paragraph to ~6 lines | keep the gauge consequences (‖w‖ is gauge; decode at reference niche; relocation is gauge-free) and the Khemakhem/Locatello sentence |
| S1.6 app:amplification | 0.3 | – | HIST (heuristic, "not a theorem") | **cut**; one clause in S2 tiles: "the stop-gradient, as in resolVI, closes the feedback path" | |

### S2 Design rationale (9.0 p → ~1.5 p inside the new S2)

| Item | p | Main pointer | Role | Action | Reason |
|---|---|---|---|---|---|
| Intro paragraph | 0.1 | – | – | cut | |
| S2.1 app:mirror (type query, selection channel, zeroing z) | 0.6 | – (mirror R² is used in main §3.4) | RD + HIST | condense to ~5 lines in "design choices" | keep why the query is the type and what the mirror diagnostic measures; cut the dropped-neighbour-means history and the self-loop aside |
| S2.2 app:graph: contact vs Delaunay text | 0.6 | 02_model:38 | RD | condense to ~6 lines; **keep the label** | keep: no per-edge geometry so attention cannot rediscover β; why Voronoi faces |
| fig:graph-compare | 0.7 | – | illustration of tab:contact's point | **cut** | |
| fig:voronoi-face | 0.7 | – | illustration | **cut** | |
| tab:contact | 0.4 | – | RD (the evidence for the kernel choice) | keep (consider dropping the "segmented by" columns) | one float carries the argument |
| fig:contact-kernel | 0.7 | – | **DUP** of tab:contact (panel c = the sparse/dense columns) | **cut** | keep the table |
| eq:context-closed + attention-sink paragraph | 0.4 | – | RD + HIST | keep the equation and 2 sentences (context depends on fractions; degree nearly constant); cut the sink ablation detail | |
| "Fallback" paragraph with `\pending{edge-feature comparison}` | 0.2 | – | HIST/future | cut; keep one clause "edge features were not tested" in the limitations | |
| S2.3 app:image text (KRONOS, channel mapping, disc) | 0.6 | – | RD | condense to ~8 lines | keep: approximate channel mapping, disc not polygon, 25 µm leaves 6 cells out, disc also removes first ring |
| fig:model-detail | 0.8 | – | partly **DUP** of main Fig. 1a (tissue panels); panel c computation graph is unique | keep, ideally reduced to panel c; else keep as is | the only computation graph with every input |
| fig:mask-radius | 0.6 | – | illustration | **cut** (one sentence keeps the number) | |
| tab:masking | 0.3 | – | RD (does the image leak the cell's identity?) | keep | key evidence: masked patch < neighbours' labels |
| KRONOS vs KRONOS2 choice | 0.15 | – | HIST | condense to 1 sentence | |
| fig:kronos-umap | 0.7 | – | illustration | **cut** | |
| S2.4 app:qw + three-routes table | 0.5 | – | RD (direct path) + HIST (table) | keep 2 sentences on explaining-away; **cut the table** | |
| S2.5 app:conditional (label granularity, subclones) | 0.5 | – (S9 points here) | **honest limitation** | condense to ~8 lines; **remove the `\todo{quote the exact numbers}`** sentence or resolve it | must keep: spatially organised intrinsic variation (subclones) moves to w; decomposition relative to t; labels are descendants of the niche |
| S2.6 app:kappa: grid range, upgrade path | 0.5 | 02_model:95 (label) | RD + future | keep grid rationale (2 sentences) and 1 sentence on the nuclear/extranuclear upgrade | |
| S2.6 Proof of Proposition 1 + what the bound means | 0.4 | 02_model:95 | SC (C1) | **keep; move into S1** (keep label app:kappa on it) | proofs belong with the derivations |
| fig:kappa-bound | 0.6 | – | **DUP** of main Fig. 1c | **cut** | |
| Breakdown example paragraph | 0.5 | – | DUP of main §2.4 / Def. 1 | condense to 3 sentences | keep "the bound is not computed on the sections, so the grid supports only κ ≤ 0.4" |
| fig:breakdown (four schematic cases) | 0.6 | – | **DUP** of main Fig. 1d | **cut** | |
| S2.7 app:variances | 0.5 | – | RD (σ_w fixed) + speculation | 2 sentences into S1 symmetries; cut conditional σ_w and VampPrior | |

### S3 Implementation (6.5 p → ~2.0 p inside the new S2)

| Item | p | Main pointer | Role | Action | Reason |
|---|---|---|---|---|---|
| tab:architecture | 1.0 | (app:architecture) 02_model:57, :84 | RD | keep; drop rows for the closed form ("α_a closed form", "moving-average rate, shrinkage") | every hyperparameter in one place |
| (new) Calibration ledger | – | 03_1:12 (currently lands on app:sensitivity) | **RD, reviewer-critical** | **add** ~0.4 p: α_z ladder (primary), α_w (primary + TMA core; FF at 10/ℓ̄), α_a and λ_y (primary, confirmed on two more), escalation threshold and that it is not met; point to app:sensitivity | calibration is now scattered over S1.2, S1.4, S3.1 and S8.2 |
| "The adversary's targets" | 0.3 | – | RD | condense to 4 lines | keep λ_y = 3 with its 6.6 vs 4.6 % justification |
| "Warm-up, diagnostics and seeds" | 0.5 | – | RD + **honesty** | condense to ~8 lines | keep: dead-channel fits are discarded and re-seeded (none in final fits), objective not unimodal in seed, seed redraws split/PCs/clusters, the five diagnostics in one line each, I(z;t)/H(t) 0.82–0.95 |
| S3.2 app:deviations + tab:deviations | 1.0 | – | **DUP** (its own `\todo` says so) | **cut** | |
| tab:deviations2 | 1.0 | – | HIST / future work | **cut**; keep 3 sentences: one seed fixed in advance for figures; one budget for final fits and sweep; built-but-off features listed in one line | |
| S3.3 app:tiles text | 0.5 | 02_model:57 | RD + **honest limitation** | condense to ~8 lines | must keep: held-out neighbours enter as stop-gradient inputs (0.8–1.9 % of seeds; 4.8–10 % of held-out cells) |
| fig:batching | 0.7 | – | illustration | **cut** | |
| S3.4 app:probe + fig:probe | 0.8 | – | **DUP** of main eq:probe | **cut** the figure and subsection; keep the label app:probe on a one-line pointer if anything cites it | |
| tab:probe (`\input`) | 1.0 | 03_3:22, 03_5:10, headline_main | SC (C2, C3) | keep; **move** to the new S3 (comparison methods) next to tab:battery | it is a comparison table |

### S4 Data, labels and comparison methods (7.5 p → S3, 4.0 p)

| Item | p | Main pointer | Role | Action | Reason |
|---|---|---|---|---|---|
| tab:sections | 0.3 | (app:sections) 03_1:6 | RD | keep | |
| fig:sections | 0.8 | – | RD (only picture of the data) | keep | reviewers want to see the tissue |
| app:lineage: rules, curated annotations, judgement calls | 0.6 | – | RD (labels define the sublabels of the C4 test) | condense to ~0.35 p | keep the SOX2-OT and cyst-lining decisions and the merged fibroblast/endothelial classes |
| tab:lineage-ovarian | 0.4 | – | SC (defines the sublabels of C4's dissociation) | keep | |
| tab:lineage-clusters | 0.6 | – | ORPH for a reader (cluster IDs) | **cut**; 2 sentences keep the rule | the mapping ships with the code |
| app:baselines text | 0.4 | 03_5:18 | RD (fairness) | keep, tighten | keep the MintFlow export-error disclosure and whole-section rule |
| Cellina variants paragraph | 0.2 | 03_5:18 | SC (C2 "architectural") | keep; **fix**: the main text says Cellina on its own graph "keeps less cycle state than DISCELL at a similar or higher residual", but this paragraph compares it only with Cellina on our graph | |
| tab:baselines | 0.8 | – | RD (label use per method, versions, coverage, OOM reasons) | keep | STORY_MAP: label use goes in this table |
| (new) Reconstruction comparability | – | 03_5:20 | **RD, missing** | **add** 2–3 sentences: multinomial vs NB/ZINB per-count held-out log-likelihood, how the battery puts them on one basis | critique C6 |
| tab:battery | 1.0 | 03_5:10, :14, :20 | SC (C2, C6) | keep | caption cites tab:headline (generator retarget) |
| tab:context | 0.8 | – | none (positive-direction grading; no claim) | **cut** | critique lists it as optional |
| app:assets | 0.3 | – | RD (licences; KRONOS not redistributed) | keep | |
| app:inputs + tab:inputs | 0.5 | – | RD for tool users | condense to 1 paragraph, **cut the table** | |
| "Other assays (untested)" | 0.4 | – | speculation | **cut** (or 1 sentence) | |

### S5 Readout definitions (4.0 p → S4, 2.75 p)

| Item | p | Main pointer | Role | Action | Reason |
|---|---|---|---|---|---|
| Intro paragraph | 0.15 | – | RD | keep | |
| tab:headline | 0.7 | – | **DUP** of tab:headline-full | **cut** after the generator retargets 6 captions (battery, cellina_cf, headline_full, kappa_sweep, transport_heldout, sensitivity) | headline_full's caption must become self-contained; drop its atlas-cosine row (conflicts with the rotation caveat) |
| tab:headline-full | 1.0 | 03_6:29, headline_main | SC | keep | |
| "What the response is and is not" | 0.25 | – | DUP of main "type-level" paragraph | cut (one clause survives in S6) | |
| Niches | 0.05 | – | RD | keep | |
| Diagnostics (mirror R², I(niche; w), cycle scores, cycling set) | 0.4 | – | **RD, reviewer-critical** | keep | keep the "proxy, not a measurement" sentence |
| Relocation (panels, ceiling, trusted tier, interval) | 0.6 | 03_6:27 | SC (C5) | condense ~25 % | must keep: interval coverage closer to 80 % than 95 %; tile-split ceiling sentence; TMA core trusted panel on 2 of 3 seeds |
| Read A and twin margin | 0.3 | (main tab:relocation) | SC (C5) | keep; **rename** to match main ("single-cell read", "twin") | keep "part of this margin follows from the construction" |
| Response programmes + GO localisation method | 0.25 | – | SC (C4b) | condense; GO method to 1 sentence | |
| External criteria: signalling share, niche info of both latents, true/false axis | 0.4 | (κ* rows) | SC (C1 rows) | keep, condensed | keep "the false axis is not orthogonal (≈0.41)" |
| Marker pairs | 0.25 | (03_7 reclassification) | trajectory only | condense to 2 sentences | |
| app:subtype protocol | 0.3 | 03_3:28 | SC (C4) | keep | |
| app:subtype results | 0.6 | 03_3:28 | SC (C4) + honest exceptions | condense to ~0.35 p; **fix the metric mismatch** (see "missing" #1) | keep exceptions: VEGFA⁺, myofibroblasts, lung clusters (response not only the context regression there) |

### S6 Full results (8.5 p → S6, 4.0 p)

| Item | p | Main pointer | Role | Action | Reason |
|---|---|---|---|---|---|
| Relocation on held-out tiles (text) | 0.3 | 03_6:31 | SC (C5) + **honest** | keep, ~6 lines | keep "Read A falls on 11 of 12 fits", "no trusted panel on TMA core" |
| fig:transport-heldout | 0.8 | – | **DUP** of tab:transport-heldout | **cut** | main points to the table |
| tab:transport-heldout | 0.5 | 03_6:31 | SC (C5) | keep | |
| Localisation of programmes (GO) | 0.2 | – | descriptive + **honest negative** | condense to 2 sentences | keep: fades to null on FF; indistinguishable from gene-specific spill-over |
| Hallmark labels paragraph | 0.15 | – | SC (C4b) | keep, 2 sentences | |
| fig:hallmarks | 1.0 | 03_4:19 | SC (C4b) | keep | |
| Signalling genes | 0.2 | (κ* row) | SC (C1) + **honest negative** | keep | must keep: MintFlow's criterion is met by the spill-over channel alone on every section; breaks at κ = 0.2 on FF |
| Niche information of both latents | 0.1 | 03_3:25 | SC (C4) | keep | the 85–90 % pointer lands here |
| True and false axis | 0.3 | (κ* row) | SC (C1) + **honest negative** | condense to ~5 lines | must keep: reversed in fibroblasts; z and depth show contrasts of similar size; not specific to w |
| Marker pairs results | 0.5 | (03_7 reclassification) | trajectory + **honest negative** | condense to ~4 lines | keep: no removal specific to exclusive pairs; no plateau, so it does not identify κ |
| app:moran text | 0.4 | 03_3:25 | SC (C3, C4) | condense to ~5 lines | keep the near-zero within-type variance of w without the adversary |
| tab:moran | 0.3 | – | **DUP** of main Fig. programmes d | **cut** | the main figure carries the values |
| app:cellina-cf setup text | 0.25 | (tab:cellina-cf) | RD | keep, 3 sentences | |
| tab:cellina-cf | 0.6 | 03_6:9, :29 | SC (C5) | keep | |
| Cellina comparison paragraph | 0.2 | – | **DUP** of main §3.6 and tab:relocation | **cut** | |
| app:latents text | 0.2 | – | DUP of the main fig:latents-main caption | cut to 1 sentence | |
| fig:latents-umap | 1.0 | – | SC (C3/C4 visually on every section) | keep (one float; consider a half-page layout) | main shows ovarian only |
| app:kl-maps text | 0.6 | – (S7 and S9 point here) | **honest negative** (type-level response) | condense to 1 paragraph | must keep: per-cell w-divergence does not reproduce between seeds; q(w) sd ≈ prior; z regularised less in low-count cells |
| tab:kl-summary | 0.6 | – | detail | **cut** | |
| fig:kl-seeds | 0.5 | – | illustration | **cut** | |
| app:timing text | 0.4 | (tab:timing) | SC (C6) + honest | condense to ~5 lines | keep the fairness note (DISCELL times include evaluations; baselines training only), "sample size not analysed", sweep = 18 fits |
| tab:timing | 0.5 | 03_5:20 | SC (C6) | keep | |
| (moved in) app:recon-modes + tab:recon-modes from S8 | – | 03_3:31 | SC (C4 limit) | place here next to kl-maps under "the type-level response" | optional move |

### S7 Simulation and planted tests (5.5 p → S5, 2.5 p)

| Item | p | Main pointer | Role | Action | Reason |
|---|---|---|---|---|---|
| app:synthetic Goal / Generative process / Fit / Readouts | 0.9 | 03_2:5 | RD | condense to ~0.4 p | |
| app:synthetic Result | 0.5 | 03_2:5 | SC (C7) | condense; cut the "earlier development run … bistable" history and the per-world cosine list | keep: loadings 0.63–0.99, one fit only 0.63; "recovery does not point to the planted κ" |
| Simulator checks | 0.1 | – | RD | keep 1 sentence | |
| tab:synthetic | 0.5 | 03_2:5 | SC (C7) | keep | |
| fig:synthetic-misspec | 0.6 | 03_2:11 | SC (C1/C7) | keep | partly the same values as tab:synthetic's misspecified rows; both are pointed to |
| app:planted-percell text | 0.8 | 03_3:31 | SC (C4 limit), **honest negative** | condense to ~0.35 p | keep the pre-set rule and "closed by design, not by the data; much of the plant goes to z" |
| fig:planted-percell | 0.6 | – | **DUP** of tab:planted-percell | **cut** | keep the table, which the text cites |
| tab:planted-percell | 0.4 | – | SC | keep | |
| app:planted simulation description (bullet list, exact posterior, fit) | 0.8 | 03_2:11 | RD | condense to one paragraph (~0.3 p) | |
| app:planted result | 0.4 | 03_2:11 | **honest negative** | keep, tightened | must keep: worse than a perceptron on the same inputs at every depth; larger than at the development configuration (0.55/0.83/1.48); not smallest at the true κ |
| tab:planted-gap | 0.3 | – | SC | keep | |
| Spill-over-subtracted encoder input | 0.15 | – | HIST | **cut** (or 1 clause) | |
| A dead response channel | 0.2 | – | RD/honesty | merge into S2 "warm-up" (2 sentences) | duplicated there already |

### S8 Spill-over sweep, sensitivity and ablations (8.0 p → S7, 4.0 p)

| Item | p | Main pointer | Role | Action | Reason |
|---|---|---|---|---|---|
| app:sweep first paragraph | 0.9 | 02_model:104 | SC (C1) | condense to ~0.4 p | keep: relocation − programme part sign changes; relocation falls to 0.35 at κ = 0.4 on the primary section |
| fig:kappa-sweep | 0.8 | – | **DUP** of tab:kappa-sweep | keep **one**: the figure (a trend read); cut the table | neither shows the probe residual (missing #2) |
| tab:kappa-sweep | 0.8 | – | DUP | **cut** (or keep it and cut the figure) | |
| "The claimed contrasts and the rules" | 0.5 | – | **RD, reviewer-critical** | keep; update after the breakdown-gaps queue (`\pending`) | must keep: no readout designated primary in advance (development sweeps preceded); Bonferroni per section; the serial-section rule |
| fig:breakdown-data | 0.8 | – | SC (C1): the only float with the interval trajectories | keep; becomes the "full" version of main tab:kappa-star | |
| tab:breakdown | 0.4 | breakdown_main caption | **DUP** of main tab:kappa-star (same rows; only m differs) | **cut** once the generator retargets "Full table: tab:breakdown" to fig:breakdown-data; else keep | |
| tab:breakdown-traj | 1.0 | – | SC + **honesty** (reclassified contrasts) | keep | the reader checks the post-hoc reclassification here |
| (new) Planted spill-over positive control | – | 03_7 (`\pending`) | **SC (C1), missing** | **add** a subsection: design, the prediction fixed in advance, result when it lands | the main text has no supplement home for its protocol |
| app:sensitivity intro + form of spill-over | 0.3 | 03_7:18 | SC (C1 limit) | keep | |
| False-positive floor | 0.2 | 03_7:18 | **honest limitation** | keep verbatim in substance | 0.65 → 0.50, −8.3 SD |
| The response weight | 0.35 | (calibration) | RD + **honest trade-off** | keep, condensed | lower α_w opens the per-cell channel and raises relocation but costs NMI |
| The adversary | 0.15 | – | RD (calibration) | keep | |
| Without the intrinsic path (ω = 0) | 0.4 | 02_model:57 | SC (model claim) | condense to ~8 lines | keep the code-drift caveat (0.616 vs 0.621) |
| fig:sensitivity | 0.8 | – | **DUP** of tab:sensitivity | **cut** | the main-text numbers trace to the table |
| tab:sensitivity | 1.0 | (03_7:18 numbers) | SC | keep | |
| app:recon-modes + tab:recon-modes | 0.6 | 03_3:31 | SC (C4 limit) | keep; condense text to ~5 lines (optional move to S6) | |

### S9 Limitations (1.5 p → S8, 1.0 p)

Keep every bullet and condense each to 2–4 lines with its pointer. Merge in: the subclone/label-granularity point (from app:conditional), "edge features not tested", "J is a surrogate; no likelihood uncertainty" (already under Inference), and the development threshold for the adversary not being met. Remove the restatements that the condensed sections now carry in full (e.g. the full FP-floor sentence can be one clause with a pointer).

### Not built (orphans)

`supplement/deferred/{additional_results,calibration,data,validation}.tex`: stale single-section drafts, not `\input`, labels would clash. No page effect; delete from the RECOMB tree or leave. (The AISTATS archive has them.)

---

## Honest limitations and negative results that must survive trimming

Every agent must keep each of these, condensed but with its number where given.

1. Proposition 1 holds with the neighbours' clean profiles fixed; the bound is not computed on any section; the grid supports only "survives up to κ = 0.4".
2. Only spill-over of the modelled form is ruled out; a fixed false-positive floor moves the primary relocation read 0.65 → 0.50 (−8.3 SD); receiving-rate scaling under/over-corrects; ambient RNA is not modelled; gene-specific spill-over lands in w.
3. Invariance is incomplete: nonlinear residual 3.5–5.5 %; large fraction left on the TMA sections; the development threshold (a quarter of the uncontrolled excess) is not met on three sections (composition) and on the TMA image block.
4. The response is type-level at α_w = 0.1: the planted per-cell response is not detected and goes partly to z; per-cell w-divergence does not reproduce between seeds; q(w) ≈ its prior; the lung section is a partial exception. Lowering α_w opens the channel at a cost in NMI.
5. The cycle score is a proxy from the same segmented counts.
6. Labels come from the same counts; the decomposition is relative to label granularity; spatially clustered subclones move into w.
7. The amortisation gap is measurable, worse than a perceptron on the same inputs, larger than at the development configuration, and not smallest at the true κ.
8. The objective is a weighted surrogate (stop-gradient, saddle point); no likelihood-based uncertainty; the objective is not unimodal in the seed; dead-channel fits were discarded and re-seeded (none among final fits).
9. Held-out cells enter training as stop-gradient neighbours (0.8–1.9 % of seeds; 4.8–10 % of held-out cells).
10. Relocation: most scored cells lie in training tiles; on held-out tiles Read A falls on 11 of 12 fits; interval coverage nearer 80 % than 95 %; TMA core trusted panel on 2 of 3 seeds and none on held-out tiles; the twin margin partly follows from construction; Cellina leads on ceiling on ovarian FFPE and on one TMA panel.
11. Descriptive reads that are not specific to the response: the LR-gene criterion is met by the spill-over channel alone; the GO lean fades on FF and matches gene-specific spill-over; the tumour axis is followed by z and depth too, reverses in fibroblasts, and its false axis is correlated with the bands (≈0.41).
12. Marker pairs show no exclusive-specific removal and do not identify κ; marker pairs and relocation − programme part were reclassified as trajectories after reading.
13. No readout was designated primary in advance (development sweeps preceded the final ones).
14. Fairness: DISCELL calibrated on these sections; baselines at defaults, trained on our held-out cells; Cellina's niche-domain row is not its published setting; SIMVI OOM on three sections, MintFlow failed on FF; the MintFlow export error and refits.
15. FF leading programme is not interpreted as a niche response (main text; the supplement must not contradict it).
16. One platform; one section held out whole; sample size not analysed; edge features in attention not tested.

## Reviewer-critical items that are missing or hard to find

1. **Subtype metric mismatch (fix before submission).** Main §3.3 (03_3:28, :31) reports *balanced accuracy* (w 0.95 vs z 0.66; 0.90 vs 0.67; tumour states z 0.79 vs w 0.41, chance 0.25; m_ψ 0.94 vs w 0.95). app:subtype, where the pointer lands, reports only *one-vs-rest AUC* (0.96–0.99 vs 0.71–0.72; 0.96–0.97 vs 0.61) and never the balanced accuracies. Add a small generated table of the main text's numbers (sublabel × channel × seed), or align the main text to AUC. Do not invent numbers: they must come from the subtype result files.
2. **The probe across the sweep.** Main §2.4 says the held-out probe is checked "at each value [of κ], so that a change in a readout is not a change in the invariance". No supplement float reports the probe residual by κ (tab/fig kappa-sweep have reconstruction, NMI, mirror, cycle and I(niche; w) only). Either add a column (generator) or soften the main-text sentence.
3. **A calibration ledger** (what was tuned, on which section, by which read). The main text's fairness sentence (03_1:12) points to app:sensitivity, which holds only part of it.
4. **Notation table**, promised by STORY_MAP.
5. **Reconstruction comparability** across likelihood families (critique C6).
6. **Supplement homes for the two pending analyses:** the planted spill-over positive control (03_7) and the plain-regression relocation reference (03_6). Each needs its protocol, its prediction fixed in advance, and the result.
7. **A guide** mapping each claim (C1–C7) to the supplement float that backs it.
8. **Float placement** (tables after the bibliography; see summary).
9. **Vocabulary:** main tab:relocation says "Single cell" and "Twin"; the supplement and the generated cellina_cf/transport_heldout say "Read A" and "twin margin". Use the main-text names, with the internal name in parentheses once.
10. **Cellina on its own graph:** main 03_5:18 compares it with DISCELL; app:baselines compares it with Cellina on our graph. Make the supplement state the comparison the main text cites.
11. **Open `\todo`/`\pending` in the supplement:** S2 (Slavutsky numbers; edge-feature run), S3 (deviations table; graph-clip quantification), S4/S8 (MintFlow refits; breakdown-gaps family sizes). Most disappear with the cuts; the last two must be updated when the queues land.
12. **Headline vs battery NMI** (0.713 vs 0.708): explained in the battery caption by reference to tab:headline; keep the explanation when tab:headline is cut (retarget to tab:headline-full).

---

## Split among four parallel agents (non-overlapping files)

Rules for every agent: edit only your files; **keep every label that the pointer inventory lists**, and keep the label of any subsection you merge (put it on the merged paragraph); when you cut a float, remove every `\cref` to it in your files and report any in other files; never edit `../aistats/tables/generated/*` (generator changes go to the main loop); use the STYLE.md vocabulary (spill-over, residual niche signal, relocation); no new numbers, only numbers already in the files or generated tables; build with `./build.sh` and report undefined references.

| Agent | Files (sole owner) | Produces | Target pages |
|---|---|---|---|
| **A: model** (Opus) | `supplement/S1_derivations.tex`, `S2_design_rationale.tex`, `S3_implementation.tex` | New S1 (notation table, derivations, proof of Prop. 1 moved from S2, symmetries) and new S2 (architecture, **calibration ledger**, design choices in brief, training, tiles). S2 file may become nearly empty or hold the design-choices subsection; `\input` order is agent D's. | 5.0 |
| **B: data and readouts** (Sonnet) | `S4_data_and_labels.tex`, `S5_readouts.tex` | New S3 (adds `\input{\tablesdir/probe}` next to battery; adds the reconstruction-comparability sentences; fixes the Cellina own-graph statement) and new S4 (readout definitions; app:subtype condensed; subtype metric issue reported, not invented) | 6.75 |
| **C: results and simulation** (Opus) | `S6_full_results.tex`, `S7_simulation.tex` | New S6 and S5 as above; optionally receives app:recon-modes from D | 6.5 |
| **D: sweep, limitations, assembly** (Opus) | `S8_sensitivity_and_ablations.tex`, `S9_limitations.tex`, `supplement.tex` | New S7 (with the planted spill-over control subsection as `\pending`), S8 limitations, the Guide, the new `\input` order (S1, S2, S3, S4, S5 = S7 file, S6, S7 = S8 file, S8 = S9 file), float placement (`placeins` with `[section]`, `\FloatBarrier`, float fractions) | 5.0 + guide |

**The only cross-file moves, each done by two owners:**
1. `\input{\tablesdir/probe}`: A deletes it from S3; B adds it to S4 (comparison methods).
2. Proof of Prop. 1: stays within A's files (S2 → S1).
3. app:recon-modes (optional): D deletes the subsection and table from S8; C pastes it into S6 verbatim (then condenses). Agree before starting; if not agreed, it stays in S8.
4. Dead-channel paragraph (S7, C) and the warm-up paragraph (S3, A) say the same thing: C cuts its copy, A keeps 2 sentences.

**Main loop, after the agents (serial; generator `scripts/paper_tables.py`):**
- Retarget `tab:headline` → `tab:headline-full` in the six generated captions; make headline_full's caption self-contained and drop its atlas-cosine row; then remove `\input{\tablesdir/headline}` from S5 (agent B leaves it in place until then).
- Retarget breakdown_main's "Full table: tab:breakdown" to fig:breakdown-data, then drop `\input{\tablesdir/breakdown}` (agent D leaves it until then).
- Decide fig vs tab for kappa-sweep; consider adding the probe residual by κ (missing #2).
- Resolve the subtype balanced-accuracy numbers (missing #1) from the result files.
- Rebuild; check that main-text `\suppref` targets render with the new S-numbers and that no label is defined twice.
