# DISCELL paper: running list of questions and potential issues

Kept alongside the manuscript. Each entry: what, why it matters, status.
Decisions already taken by the author are recorded under "Decided".

## Decided (author, 2026-09-08)

- Canonical results: sweep2 (6 kappa x 3 seeds) + reference_best are the *starting point*; every
  claim will be re-run with more seeds for error bars. Old sweep1 / reference v2-v6 not cited.
- w framing: present w as the context-conditional response; at the operating point it is
  dominated by its prior. Anomaly scores / counterfactuals are future work. Add *perturbations*
  as a third variable (intrinsic / environment response / perturbation response) to future work.
- B: no strong claims. Loadings are run-specific and identified only up to invertible mixing;
  all gene-level statements are run-relative. Report matched-correlation stability only.
- Invariance mechanism reported: the adversary (design's closed-form default failed the
  escalation rule). "Four hyperparameters, one optimiser" framing dropped.
- DisCoVR = Slavutsky, Beker, Blei, Dumitrascu, arXiv:2506.17182 (ICML 2026).
- Doc 05 is an old version of the model design; not needed. Doc 08 (validation handover) received.
- Unreferenced claims carry a visible "[NEEDS A SOURCE]" marker in the PDF.
- Title: "DISCELL" only for now; author block later; anonymous build.
- Appendix inside the main file after the checklist. Never reference code (files, functions,
  flags, variable names) in the article.
- Overflow of the 8-page limit is fine for now; trimming later.
- Second dataset: the lung section, with graphclust labels (no curated annotation). Also an
  ovarian graphclust run to check the decomposition does not depend on the curated labels.
- Priority now: the method section; experiments are a skeleton until the reruns exist.

## Open questions for the author

1. Held-out semantics: the 15% held-out tiles drive early stopping, the NMI guard and every
   probe. Is an untouched test split of tiles required for the final claims? (Recommended: yes,
   carve a third split before the reruns so that nothing has to be re-done.)
2. Which checkpoint's diagnostics are quoted: best.pt (joint criterion) vs last-epoch. Recommend
   best.pt everywhere and re-evaluating mirror/probe/cycle from it in the sweep table.
3. MLP-probe delta-CE was never computed on sweep2 / reference_best (only at calibration). It
   should be part of the standard evaluation battery for the reruns.
4. The "cycle ceiling" (50-PC ridge) is beaten ~2x by z. Renamed "linear reference" in the
   text; the reason (nonlinear encoder vs linear ridge) must be stated once.
5. Noise floor of the probe is negative (-0.05..-0.06), the design said ~0. Explanation needed
   (probe overfit on held-out cells vs the type-mean baseline). One sentence in the appendix.
6. Landmark analysis: interface signal ~60% definitional (y-baseline 0.042 vs w 0.069);
   vasculature null. Currently appendix-only. Keep, or drop entirely?
7. The z niche-AUC residual (0.65 vs depth baseline 0.59, kappa-reducible) — present as a
   limitation ("invariance is partial") with the per-dimension triage, not as "benign".
8. alpha_z = 0.007 was set on synthetic data (~150 counts); the section's 1/l-bar is 0.004-0.006.
   Scan alpha_z on the section (listed in extra_runs.md).
9. Image embedding at full 384-d contradicts the design's 32-64-d projection; evidence is one
   40-epoch ablation. Run the projected arm at the operating point (listed in extra_runs.md).
10. The 25 um ego mask removes a median of 18 neighbouring cells; Phi describes tissue beyond
    ~25 um. State this scale wherever Phi is interpreted. Is this the intended reading?
11. Channel-to-marker mapping for the image model is approximate (two cocktail channels; the
    ribosomal stain is out of vocabulary). Limitation paragraph in the main text?
12. "Unassigned" class (2,497 cells) is a type in the model and excluded from readouts. OK?
13. Degree <= 2 cells excluded from effect readouts; their extra kappa cost reported. OK?
14. Design table (appendix C.2, decisions vs original design): keep in the paper, or absorb
    into the method text and delete? A reviewer reproducing the method from the text must not
    find mismatches with the described model.
15. DisCoVR "Table 19" numbers quoted in the design notes (CSVAE NMI 0.716 -> 0.406, nuisance
    0.002) must be verified against the paper before they are quoted.
16. The feedback-amplification argument (1/(1-kappa)) is a linearisation heuristic; either
    tighten it or present it as heuristic + the resolVI empirical precedent.
17. Related work is a stub. Needs: spatial VAEs / niche models, GNN-based spatial models,
    segmentation/contamination correction methods, disentanglement in single-cell.
18. Pseudotime neighbour coherence: report.md quoted w 0.64 / z 0.15, validation.json gives
    w 0.53 / z 0.03 (same instrument, different subsample). One protocol, stated.
19. ||w|| ranking: "all 18 runs" is false (16/18 VEGFA+ on top; k0.2_s0 has all norms halved).
    Report counts; investigate k0.2_s0.
20. Sweep validation companion used old landmark definitions; reference_best the rewritten ones.
    Rerun the companion under the rewritten definitions (listed in extra_runs.md).
21. The label-set control (graphclust, K=26) gives lower cycle R^2 from z (0.32 vs 0.43-0.47).
    Consistent with "finer t leaves less for z" — say so, or investigate?
22. Anonymised code release for the checklist (1c, 3a)? The image model weights are gated.

## Open items from the internal critic pass (2026-09-09)

Four critics reviewed the method (against the design document, against the implementation,
as an adversarial AISTATS reviewer, and for style). Fixed in the text: overflowing equations,
objective notation (penalty outside the cell mean), probe baseline and probe input (posterior
means), adversary on posterior means, jitter and support-floor wording, isolated-cell charge
and renormalisation formula, kernel row-stochastic wording, B identified up to rotation (not
"invertible mixing"), identifiability paragraph, sweep framed as a sensitivity analysis,
stop-gradient framed as preventing a degeneracy (amplification as heuristic), degeneracy
diagnostics (v), learned-vs-fixed table, related work (SIMVI, NicheCompass, Elazar & Goldberg,
resolVI delta), leakage-model assumptions paragraph, label caveats, two niche scales, the
held-out ring crossing, and many wording/citation items. Still open, for the author:

23. **Stability rule for the sweep.** The text now proposes: sign (and rank order for
    rankings) unchanged over the grid, seed envelope excludes zero at every grid point.
    Confirm or replace. Also stated: ||w|| declines with kappa by construction.
24. **alpha_a selection rule.** The text now describes what was done ("smallest alpha_a that
    brings the nonlinear probe within a small fraction of the uncontrolled baseline without a
    loss in NMI or held-out reconstruction"); the fractions are left as a TODO until the
    calibration scans are repeated. The old "one fifth" threshold was tuned to the outcome.
25. **Geometry-in-context argument reframed.** The reviewer noted the causal argument was
    inconsistent (Phi also encodes architecture, and no estimand is written down). The text now
    leads with collinearity (attention rediscovering beta), then density-is-in-Phi, then the
    informal exposure/covariate reading. The design document had the causal argument as
    decisive. Author to confirm the new ordering.
26. **Held-out tiles inside training rings.** Ring-one/two of a training tile can contain
    held-out cells used transductively under a stop-gradient. Quantify the fraction, or add a
    one-hop buffer around held-out tiles before the reruns (extra_runs.md item 22).
27. **DisCoVR structured prior.** The exact form is needed in two places (method 2.4,
    appendix A.2); currently a TODO. Also the CSVAE numbers.
28. **Appendix duplication.** The rationale appendix was trimmed to non-duplicated content;
    the "choices made in implementation" table now duplicates the method text and the author
    should decide whether it stays.
29. **Length and tone.** The method is ~6 pages two-column. Overflow accepted for now; the
    reviewer's target is ~4 pages with one reason per decision and appendix pointers. Some
    assertive phrasing was softened; more will go at trimming time.
30. **Per-cell 1/l scaling vs global alphas.** Low-count cells' divergences are weighted more
    heavily relative to their likelihood (now acknowledged in the text). Alternative: scale
    the weights by l_i / l-bar. Decide whether to test it.
31. **Labels derived from contaminated counts** are now flagged as a caveat in 2.1; whether to
    add an experiment (e.g. label robustness under the graphclust control) is open.
32. **Two hyperparameters not discussed**: d_w (=6) has no sensitivity check; E_Phi = K is a
    convention. Add d_w to the scans (extra_runs.md item 23).
33. **Bib entries with CHECK notes** print in the reference list (resolVI, KRONOS, SIMVI,
    NicheCompass, Janesick author list). Resolve before any external circulation.

## Reviewer risks (from the internal review, to be addressed or acknowledged)

- Single section, single tissue, no test split, no external baseline (resolVI, NCEM, plain
  conditional VAE) on the same readouts. Second section and seeds are planned; baselines are not.
- Two knife-edges (alpha_w, alpha_a) and a second optimiser; robustness of the operating point.
- w is effectively the prior at the operating point; "response latent" must not be oversold.
- Residual spatial structure in z (niche AUC above depth baseline; Moran's I above null).
- Nonlinear leak measured only at calibration, not on the sweep.
- Loadings reproducibility 0.48-0.64 across seeds; B bistable on synthetic data with leak.
- kappa grid justified by an uncited range; marker-set ceiling never computed.
- Implementation departs from the original design in many small ways; the text must describe
  what was run.

## Writing conventions

- "[NEEDS A SOURCE]" (red) for claims lacking a citation; "[TODO: ...]" (orange) for editorial
  work; "[PENDING RUNS: ...]" (violet) for numbers waiting on the reruns.
- "held-out", never "test", until a third split exists.
- "linear reference", never "ceiling", for the 50-PC probe.
- Gene-programme statements are always run-relative.
