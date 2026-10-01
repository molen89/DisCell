# Trim report: agent A (model): S1_derivations, S2_design_rationale, S3_implementation (+ one main-text edit)

## Pages
- Before: about 20 pages (S1 4.5 + S2 9.0 + S3 6.5, flushed tables included; full supplement 59).
- After: about 7.0 pages (in the private build `_build_A`, from mid page 1 after the Guide to mid page 8, where S3 Data starts). That is over the agent-row target of 5.0 but under the map's per-section budget (3.0 + 1.5 + 2.0 = 6.5, plus the notation table and the calibration table that were added). The rest is in the proof of the symmetries (~0.8 p, kept as the map asks), the notation table (~0.4 p, new) and the merged architecture and calibration float (~1 p).
- Main text: still 10 pages (`\mainpages{10}`). The §2.4 paragraph after Proposition 1 is still 12 lines.

## Structure
- `S1_derivations.tex` → **S1 Model: Notation, Derivations and Proofs**: notation table (new, `tab:S-notation`); bound (shorter; `eq:gauss-kl` display dropped); isolated cells (`eq:S-renorm`); why a multinomial (6 lines); intrinsic path + scaling of the weights + status of the objective (`app:pathb` and `app:posterior-reg` on one subsection); **proof of Prop. 1 moved here from S2** (`app:kappa`), with the grid rationale, the sentence "not computed on our sections, so the grid supports only up to 0.4", and one sentence on the nuclear/extranuclear upgrade path; symmetries (proposition and proof kept, commentary down to ~8 lines, with the fixed-σ_w point from app:variances).
- `S2_design_rationale.tex` **opens** the section **S2 Implementation, Design Choices and Calibration** (labels `app:rationale`, `app:implementation`) with fig:model-detail (now panel c only) and "Design Choices" (paragraphs labelled app:mirror, app:graph, app:image, app:qw, app:conditional).
- `S3_implementation.tex` **continues the same section** (no `\section`): architecture, calibration (new, `app:S-calibration`, `tab:S-calibration`), training/diagnostics/seeds, tiles (`app:tiles`). **Agent D: input S2 then S3 one after the other.**

## Floats
Cut: fig:graph-compare, fig:voronoi-face, fig:contact-kernel, fig:mask-radius, fig:kronos-umap, fig:kappa-bound, fig:breakdown, the three-routes table, tab:deviations, tab:deviations2, fig:batching, fig:probe.
Kept: tab:contact (the "segmented by" columns dropped; one sentence in the text keeps "69 to 87 % interior stain"), tab:masking (shorter caption), tab:architecture (closed-form rows, invariance-block row and history removed; budget 500/40 moved in from tab:deviations2).
Changed: fig:model-detail now shows only `fig_model_graph` (the computation graph); the tissue panels duplicated main Fig. 1a.
Added: tab:S-notation; tab:S-calibration (same float as tab:architecture, so that no page holds only one float).
Moved out: `\input{\tablesdir/probe}` deleted from S3 (agent B adds it to the new S3).

## Text cut or condensed
- app:gaussian-mi cut, apart from the escalation rule and the honest negative (threshold not met: composition on three sections, image on the TMA core), which are now in the calibration subsection. **eq:S-penalty is kept** as one displayed equation there, because S7 (agent C) cites it for the development configuration.
- app:amplification cut; one clause in tiles ("the stop-gradient, as in resolVI, closes the feedback path").
- app:variances cut; two sentences in the symmetries commentary.
- app:deviations and app:probe labels are gone (nothing cited them). The "built but off" features are in one sentence in Training; "one seed fixed in advance" and "no seed selected" are kept.
- The `\todo` items (Slavutsky numbers, graph-clip quantification) and `\pending{edge-feature run}` are gone; "edge features were not tested" points to app:limitations (agent D: please keep that clause in S8 limitations).
- Development history is down to single sentences (attention sources, closed-form penalty → adversary, KRONOS vs KRONOS2).

## Labels
Kept from the pointer inventory: app:bound, app:graph, app:tiles, app:architecture, app:kappa, app:symmetries (and app:derivations, app:rationale, app:implementation, app:pathb, app:posterior-reg, app:mirror, app:image, app:qw, app:conditional, prop:S-symmetries, eq:jensen, eq:S-renorm, eq:context-closed, eq:S-penalty, fig:model-detail, tab:contact, tab:masking, tab:architecture).
Removed (no references left anywhere): app:gaussian-mi, app:amplification, app:variances, app:deviations, app:probe, eq:gauss-kl, and the cut floats above.
New: tab:S-notation, app:S-calibration, tab:S-calibration.
Private build: no undefined references.

## Cross-file requests
1. **Agent D (S8):** the caption of fig:breakdown-data says "the measured counterpart of \cref{fig:breakdown}". fig:breakdown is cut; please retarget to the main-text `fig:overview` (panel d).
2. **Agent D (guide):** if the Guide names the old S2/S3 items, point it to S1.3 (proof of Prop. 1), S2.3 (calibration) and tab:S-calibration. The main text's 03_1:12 "calibrated on these sections" points to app:sensitivity; app:S-calibration is the better target (main loop decides).
3. **Agent C:** S3 no longer cites app:planted (the dead-channel sentence now stands without a pointer). eq:S-penalty still exists for S7:74.
4. **Main loop / figure generator:** `fig_model_graph.pdf` still says "leakage" in its labels and carries the panel letter "c". Regenerate it with "spill-over" and no letter.

## Main-text edit (sections/02_model.tex, §2.4)
After Proposition 1: "$\bar\kappa_i$ is the largest spill-over fraction that cell $i$'s counts could still allow; it limits $\kappa$ from above, but nothing limits it from below. It equals $\kappa$ only through a gene the cell cannot express (proof in Supplement)." The repeated "Even with … from above" sentence is gone and "With finite counts the bound is soft; we do not compute it…" is tightened to "The bound is soft with finite counts and not computed on our sections". Paragraph length is unchanged (12 lines) and the main text is still 10 pages.

## Unresolved
- The ω = 1 and d_z/d_w calibration history is not documented in the current files, so the ledger gives no "set on" section for ω and no row for the widths. I did not invent one.
- The claim that α_a was confirmed on two more sections (from the map) has no source in the current files, so it was left out.
- The section is ~2 pages above the 5-page agent target; cutting more would mean dropping the symmetry proof or the notation table, which the map keeps.
