# Review of DISCELL (RECOMB draft, after the trim), 2026-10-01

Reviewer stance: a demanding but fair RECOMB reviewer. I read `main.pdf` (49 pp.: main text pp. 1–10, references pp. 10–13, supplement S-pp. 1–35), the `sections/` and `supplement/` sources, and the generated tables. Nothing was edited.

---

## 1. Summary, and whether it matches STORY_MAP.md

DISCELL is a generative model for imaging spatial transcriptomics. It writes each cell's counts as a multinomial over a mixture of the cell's own composition, decoded from an intrinsic state z and a response w whose prior depends on a niche descriptor, and a fixed fraction κ of its neighbours' clean compositions (spill-over). Because the counts bound κ only from above (Proposition 1), the authors do not estimate κ. They refit across a κ grid and attach a breakdown point κ⋆ to each finding. A type-conditional adversary keeps the niche out of z, and a held-out probe grades how well it does. On four Xenium 5K sections plus a held-out serial section, they report that z keeps type and cycle state while w carries the niche and maps tissue programmes. They also report that no baseline dominates DISCELL on residual niche signal against cycle state, that relocation recovers 0.65–0.73 of the reproducible between-niche shift (mixed against Cellina), and that most findings survive κ = 0.4.

**Match with STORY_MAP:** yes, for the pitch, vocabulary, claims C2–C4b, C6 and C7, the fairness paragraph in §3.1 and the float plan. The deviations are:
- **C1 is incomplete.** The planted-spill-over positive control, which the story calls "the strongest answer to 'your contrasts survive by construction'", is still a `[PENDING RUNS]` stub in both §3.7 and S7.3. Table 4 still has eight `?` cells.
- **C5 is incomplete.** The plain-regression reference (§3.6) is a stub.
- **C2 and C6** depend on the MintFlow refits. MintFlow's reconstruction is "n/r", and every MintFlow point in Fig. 4 and Table S9 will change.
- The story's limit on C5, "On held-out tiles DISCELL holds up", is told selectively in the text (see §6).
- The story allowed "Read A" and "twin margin" only if defined in one line. The main text avoids them, but the supplement does not (see §4).

---

## 2. Main-text → supplement pointers

All 23 `\suppref`s resolve, and each lands in the intended section. Checked one by one:

| Main text (where) | Target | Verdict |
|---|---|---|
| §2.1 "no distances … (S2.1)" | S2.1 "No per-edge geometry" | OK |
| §2.1 isolated cell "(S1.1)" | S1.1 "Isolated cells" | OK |
| §2.2 ω = 0 refit "(S7.4)" | S7.4 "Without the intrinsic path" | OK |
| §2.2 halo "(S2.5)" | S2.5 | OK |
| §2.2 "remaining settings in S2.2" | S2.2 / Table S4 | OK |
| §2.3 checkpoint rule "(S2.2)" | Table S4 "Evaluation" row | OK, but the rationale is in S2.4; point there too |
| §2.4 "proof in S1.3" | S1.3 | OK, but S1.4 also contains "Proposition 1" and "Proof of Proposition 1" (numbering clash; see §4) |
| §2.4 symmetries "(S1.4)" | S1.4 | OK |
| §2.4 "shown as a trajectory instead (S7.1)" | S7.1 only names the reclassified contrasts; the trajectories are in **S7.2 / Table S20** | Weak: point to S7.2 / Table S20 |
| §3.1 sections "(S3.1)" | S3.1 / Table S6 | OK |
| §3.1 "calibrated on these sections (S2.3)" | S2.3 says every weight was set **on the primary section only** and checked elsewhere | Target is fine; the main wording is looser than the supplement |
| §3.2 "(S5.1, Table S13)", "Fig. S3" | OK | OK |
| §3.2 amortisation "(S5.3)" | S5.3 | OK, but the main sentence "more so at the final configuration than in development" is development history |
| §3.2 "positive control … in Sec. 3.7" | §3.7 → `[PENDING]`; S7.3 → `[PENDING]` | **Lands on nothing** |
| §3.3 "85 to 90% … (S6.1)" | S6.1 | OK, but S6.1 gives 0.41–0.63 nats where Table 2 gives 0.47–0.60 (a different sample and floor, stated in S6.1 only) |
| §3.3 no adversary "(S6.2)" | S6.2 | OK |
| §3.3 subtypes "(S4.2)" | S4.2 / Table S12 | OK, all numbers match |
| §3.3 "one part in a thousand (S7.5)" | S7.5 / Table S22 | OK |
| §3.3 planted per-cell "(S5.2)" | S5.2 / Table S14 | OK |
| §3.3, Table 2 "Tables S9 and S11" | OK | OK |
| §3.4 "Fig. S4" (EMT on most programmes) | Fig. S4 | OK |
| §3.4 / Fig. 3c **drivers** (composition, image, *landmarks*) | **No supplement section.** "Landmarks (vessels, smooth muscle, the tumour–stroma interface, where annotated)" are not defined anywhere. The cross-validated R² regression, the unique-share decomposition and the territory score have no methods text | **Missing target** |
| §3.4 / Fig. 3d Moran's I of w (0.48–0.73) | S6.2 defines within-type Moran's I and tabulates z only. **w's values are tabulated nowhere** | Partial |
| §3.4 FF cosine "as low as 0.67, against 0.72 to 0.98" | Table S11 "atlas cross-seed cosine" ranges are 0.76–0.98 | **Contradicts** (see §4) |
| §3.5 "(S3.3)", "Tables S9, S10", "Table S18" | OK | OK, but Table S10 is clipped at the page foot (see §4) |
| §3.6 "(S4.1)", "Table S11", "Table S16", "Table S17" | OK | OK. Table S17 also shows the single-cell read falling on 11 of 12 fits, which the main text does not say |
| §3.6 plain regression | `[PENDING]` | **Lands on nothing** |
| §3.7 "(S7.4)", Table 4 → "Table S19, Fig. S6" | OK | OK |
| §4 "The full list is in S8" | S8 | OK |

The Guide (S-p. 1) has its own pointer problems:
- C1 points to S7.3, which is empty.
- C4b points to Fig. S4 and S6.1. It should also point to S6.2 (Moran's I) and to the missing drivers methods.
- C5 omits S6.4, which is the actual relocation-results section.
- Its prose says "Secs. S3 and S4 [back] Sec. 3.1", but S4 is the readout definitions that back §§3.3–3.7.

---

## 3. Reviewer-critical gaps

**Metric definitions (the main text must stand alone)**
- **Table 4:** the main text defines none of its five biological contrasts.
  - "Signalling-gene lean of w" and "Tumour axis in w" are **never introduced in §§3.3–3.6**, although the Table 4 lead-in says it "gives each finding of Secs. 3.3 to 3.6 its breakdown point". Table S19 even maps both to "Sec. 3.7".
  - "Relocation > type mean" (the single-cell read minus a type-mean reference) and "Relocation > spill-over part" (computed over *all* panels, not the trusted tier used everywhere else) are also undefined in the main text.
  - Fix: add a one-line gloss per row in the caption, or drop the two rows the main text never discusses to the supplement.
- **Fig. 3:**
  - The territory score is defined only in the caption.
  - "Within-type Moran's I" is defined only in S6.2.
  - The drivers regression and the landmarks are defined nowhere.
  - The hallmark test (Mann–Whitney on absolute loadings, BH within programme) is defined only in the Fig. S4 caption.
  - The "I" in Fig. 3a (a per-programme score: 0.62, 0.45, 0.97, 0.48) is not the "I" in Fig. 3d or the text (whole w: 0.48–0.73). FF's 0.97 and lung's 0.45 fall outside the range the text quotes.
  - "(FF rotated)" in the caption is ambiguous next to rotation non-identifiability.
- **"Niche"** in I(niche; w), in Cellina's "niche label" and in the relocation panels is a k-means(10) clustering of neighbour composition. The main text never says so; it is defined only in S4.1.

**Baselines (fairness)**
- Fairness is stated well in §3.1. Unaddressed:
  - resolVI and MintFlow ran **50 epochs** against DISCELL's ≤500 with early stopping; nothing shows the baselines converged.
  - Cellina's main row runs with **no adversary at all**.
  - The timing comparison mixes DISCELL's wall-clock time (including evaluation) with the baselines' training-only time. That is conservative, but say it in the main text.
- MintFlow reconstruction is excluded pending refits. The §3.5 claim "within 0.015 nats of the best method on every section" silently excludes MintFlow and SIMVI.

**Data and preprocessing**
- There is no cell or gene QC statement for DISCELL: minimum counts, removal of negative-control and blank probes, whether all 5,001 genes enter. The "≥5 counts" filter appears only for the baselines (Table S10 caption).
- The segmentation used is stated only obliquely ("interior RNA stain", S2.1).
- The clustering behind the lung FFPE and FF labels (algorithm, resolution, markers) is not described. "Mappings accompany the code" is not enough for review.
- **Held-out tiles do triple duty**: checkpoint selection (held-out reconstruction plus the NMI ratchet), early stopping (Table S13 caption) and evaluation. The held-out reconstruction and NMI in Tables 2, S10 and S11 are therefore selected on the test cells. There is no separate validation split, and this is not acknowledged.

**κ grid and rule**
- Sec. 2.4 and S7.1 state the grid, the seeds, Definition 1 and Bonferroni clearly. Missing from the main text:
  - The relocation intervals have **~80% rather than 95% coverage** at real ceilings (S4.1). This makes the relocation κ⋆ values anti-conservative.
  - The rule for moving a contrast to a trajectory is muddled. §2.4 says "a contrast without a sign at κ = 0 … or measures decoding", but the marker-pair contrast *does* have a sign at κ = 0 (S7.1: "already present at κ = 0").
  - The reclassified "relocation − programme part" turns **negative from κ = 0.3 on the primary section** (S7.2). Had it stayed in the set, it would be a fourth broken finding. Say so in §3.7.

**Planted controls**
- Planted per-cell response (S5.2): complete, well designed, honestly reported.
- Planted spill-over positive control: **absent**. This is the single most important missing experiment, because without it "the test has teeth" rests on post-hoc characterisation of which findings broke.

**Limitations**
- The §4 paragraph and S8 are good and honest.
- S8 opens "The conclusion names…" but the main section is "Discussion".

**Code availability**
- "[URL to be added]" appears twice (Contributions and §4). RECOMB requires a link.

---

## 4. Consistency

**Numbers**
1. **Cross-seed cosine:** §3.4 says "as low as 0.67 [FF], against 0.72 to 0.98 elsewhere". Table S11 gives FF 0.76–0.90, lung 0.80–0.90, ovarian FFPE 0.97–0.98 and TMA 0.89–0.92. Either the reads differ (pairwise against reference), which must be said, or one is stale.
2. **Moran's I:** Fig. 3a (per programme: 0.45 lung, 0.97 FF) against the text and Fig. 3d (w: 0.48–0.73). Both are labelled "within-type Moran's I".
3. **"Every readout is taken on held-out cells"** (§3.1) against §3.6 "Most scored cells lie in the model's training tiles" (and S6.4, Table S17). This is a direct contradiction.
4. **"Only MintFlow carries less [nonlinear residual] wherever it ran"** (§3.5). On the TMA core, Cellina's composition residual is 3.5% against DISCELL's mean of 4.0% (Table S9), and Table S10's MLP column shows Cellina 0.51 against DISCELL 0.60. The sentence's own hedge ("at DISCELL's lowest seed") concedes that Cellina is below the mean. Reword it as "only MintFlow, and Cellina on the TMA core, …".
5. **"Least image information of all methods on every section" (linear probe):** on the TMA serial section DISCELL's range is 0.5–0.8% against MintFlow's 0.7% (overlapping). It holds by the mean only; say so, or soften.
6. **MintFlow run time:** S6.6 gives 14.8 h (lung) and 15.8 h (ovarian), "wall-clock". Table S18 gives 879.4 min (14.66 h) and 938.6 min (15.64 h), "training time". §3.5 gives "9 to 16 hours". Pick one basis.
7. **Serial section coverage:**
   - Table 2 marks I(niche; w) on the serial section "–: not read".
   - Table 4 gives "?" (pending) for that row and four others.
   - S7.1 says only the cycle asymmetry is read there.
   - The S7.1 pending note says the serial set becomes six contrasts.
   All four must agree after the queue lands.
8. **§3.7 "Every other finding holds across the whole grid on every section"** while Table 4 still has `?` cells (tumour axis on lung and FF; the whole serial column).
9. **Calibration scope:** main "calibrated on these sections" against S2.3 "set once, on the primary section".
10. **Serial cycle R²:** Table 2 and §3.3 give 0.49, but Table S11's 0.495 would round to 0.50. Check the rounding.
11. **Abstract cost claim** "the whole sweep costs less than one fit of the slowest comparison method" is false on FF. No slow method ran there, and §3.5 itself says the FF sweep "takes longer than one resolVI fit". Add "where it ran".
12. **Fig. 3b against the §3.4 text:** the text names INHBA, IL6 and PTGS2, which are not among the figure's genes, and omits ESM1, which is the figure's top positive gene.

**Vocabulary drift**
- **"Read A" (known issue, confirmed):**
  - Fig. S6 row label "Read A − type-mean reference" (the figure itself).
  - Table S16 (`cellina_cf.tex` l.69 row label, and caption).
  - Table S17 (`transport_heldout.tex` column header and caption).
  - Table S19 definition (`S8_sensitivity_and_ablations.tex` l.28).
  - S4.1 "(internally Read A)".
  - S6.4 "(Read A)".
  - `breakdown.tex` l.63.
  - The main text calls it "Single cell" (Table 3) and "single-cell read". Rename everywhere to "single-cell read", and drop "internally Read A / the twin margin" from S4.1.
- **"Twin margin"** (supplement) against "Twin" (Table 3) against "Nearest-twin advantage" (Table 4).
- **"Leakage" and "transport":** not in the rendered text (the file name `transport_heldout.tex` and the label `tab:transport-heldout` are invisible, which is fine).
- **Remaining internal jargon in the supplement:**
  - "battery" (Table S10, S3.3)
  - "published read" (S6.4, Table S17)
  - "gauge" (S1.4, undefined)
  - "development configuration / development sweeps"
  - "the honest unit" (main text §3.5)
- **TMA naming:** Table S2 uses "Lung TMA, section A / section B"; everywhere else it is "TMA core / serial section".
- **Residual signal naming:** "residual niche signal", "composition residual", "nonlinear residual" and "fraction left" name related quantities with different bases. Table S10's Ridge/MLP columns are fractions of DISCELL-without-adversary, while Table S9's are percentages, and both are called "residual".
- **Duplicate Proposition 1:** the supplement compiles separately, so "Proposition 1 (Symmetries of the response)" in S1.4 collides with the main text's Proposition 1 (κ bound). S1.3 and S1.4 each contain a "Proof of Proposition 1". Use `\renewcommand{\theproposition}{S\arabic{proposition}}` in `supplement.tex`.

**Claims stronger in one place than another**
- **Relocation on held-out tiles:**
  - §3.6 says "no systematic in-sample advantage".
  - S6.4 and Table S17 add "The single-cell read does [fall], on 11 of 12 fits".
  - The main text keeps the favourable half.
- **Sublabels:**
  - The abstract says "split as designed" without qualification.
  - §3.3 says locations are called to w on only 2 of 3 seeds, with exceptions.
  - The §3.3 version is fine; the abstract is stronger.
- **"Out-of-sample replication on a second slide"** (§3.3) for a serial section of the same core. This is a technical, not biological, replicate.

**Layout (known issue, confirmed)**
- `battery.tex` (Table S10): `Float too large for page by 94.343pt` (supplement.log l.699). The TMA-serial block runs past the page foot; the page number "14" sits in the middle of the table, and the serial "Cellina, niched / own graph" rows are cut off or hidden.
- `headline_full.tex` (Table S11): `Overfull \hbox 17.46pt` (supplement.log l.703).
- Minor: S1.2, an overfull of 1.5 pt.

---

## 5. Clarity and navigation of the supplement

- **The Guide is useful.** The claim → main → supplement table is exactly what a reviewer wants. Fix its pointers (§2 above) and its prose mapping.
- **Results tables are scattered:**
  - Tables S9 (probe) and S10 (battery) sit in S3 "Data, labels and comparison methods".
  - Table S11 (headline with intervals) sits in S4 "Readout definitions".
  - S6 "Results on tissue in full" holds neither.
  - Moving S9, S10 and S11 into S6 would make S3 and S4 purely methods.
- **Redundancy still present:**
  - The SIMVI and MintFlow out-of-memory reasons are spelled out four times: Table S8, the Table S9 caption (an auto-generated, four-fold "could not be run on the resources currently available"), Table S10 footnote e and Table S18 footnote a. Keep them once (Table S8) and cross-reference.
  - The Cellina "niche domain = best case, not published setting" explanation appears five times (§3.1, S3.3, Table S8, the Table S9 caption, Table S10 footnote d).
  - "The response readouts are readouts of the context regression / the per-cell channel is closed by design" appears in S1.2, S4.2, S5.2, S6.3, S7.4, S7.5 and S8.
- **Development history that can go (or be cut to a clause):**
  - S2.3 paragraph on the closed-form Gaussian penalty (Eq. S4). Keep at most one sentence for why the "quarter" threshold exists.
  - S5.3 last paragraph comparing against the development configuration. The main §3.2 echo ("more so … than in development") should go too.
  - S2.4 "A spill-over-subtracted encoder input, an attention sink and coupled weight decay are implemented but off".
  - S7.4 last sentence ("One final fit on the TMA core, refitted with the software used for the ablation…").
  - Table S5's "primary, development" / "short fits with the closed-form penalty" entries. These are honest provenance, but trim the wording.
  - S3.3 MintFlow export-error paragraph and Table S10 footnote b: remove once the refits land.
- **Keep:** S7.1's "No readout was designated primary in advance…". That is necessary disclosure, not history.
- **Notation:** S1.4's "gauge" needs a definition or a plainer word ("an unidentified direction").
- **Fig. S6:** "no contrast" floats in the ovarian FFPE cycle panel without explanation, and the y-scale of the single-cell contrast (up to 1.5) exceeds the clipped range of a single read without comment.

---

## 6. Overclaims and undersells

**Overclaims**
1. **"The sweep has teeth: it breaks three findings, which are those spill-over could plausibly produce"** (abstract, Contributions, §3.7, §4):
   - The "plausibly produce" characterisation is post hoc.
   - Two contrasts were moved out of the graded set after reading, and one of them (relocation − programme part) turns negative from κ = 0.3 on the primary section.
   - The relocation intervals are ~80% intervals.
   - Without the planted positive control, "teeth" is asserted rather than shown. State the reclassified contrast's behaviour in §3.7, and do not submit without the control.
2. **"Every other finding holds across the whole grid on every section"** while Table 4 has `?` cells.
3. **Abstract cost claim:** false on FF (see §4).
4. **"The separation comes from the architecture, not from the adversary alone"** generalises from one model (Cellina) given one adversary target. Say "is not reproduced by giving Cellina our niche label".
5. **"No systematic in-sample advantage"** for relocation omits the single-cell read falling on 11 of 12 fits.
6. **"Out-of-sample replication on a second slide":** call it a serial-section (technical) replication.
7. **Reconstruction "within 0.015 nats of the best method on every section"**, with MintFlow excluded and DISCELL's checkpoint selected on the same held-out cells.

**Undersells (fair to the authors)**
- The baselines trained *on the test cells*, yet DISCELL still has the best held-out reconstruction on both whole-slide FFPE sections. That is a stronger fairness point than the text makes.
- DISCELL's timing includes evaluations while the baselines' times are training-only, so the speed claims are conservative. Say so in §3.5, not only in S6.6.
- The planted per-cell experiment (S5.2), with a rule fixed in advance and a negative result reported, is exemplary practice. One clause in §3.3 crediting the pre-registration would help.
- Proposition 1 is simple but is the paper's conceptual hinge, and it transfers to resolVI-type models. The Discussion could say once that resolVI's per-cell weights estimate a quantity the counts cannot pin down from below.

---

## 7. Top 10 fixes (prioritised)

1. **Run and report the planted-spill-over positive control** (`sections/03_7_robustness.tex` stub; `supplement/S8_sensitivity_and_ablations.tex` S7.3). Without it, "the test has teeth" is unsupported.
2. **Clear every `[PENDING RUNS]`** (abstract, Contributions, Fig. 4 caption, §3.5 ×2, §3.6, Table 4 `?` cells, S3.3, S7.1, S7.3). Then re-verify C2 and C6 against the MintFlow refits and re-sync the serial-section coverage across Table 2, Table 4 and S7.1.
3. **Fix the two oversized tables:** `../aistats/tables/generated/battery.tex` (Table S10, 94 pt too tall, serial rows clipped) and `headline_full.tex` (Table S11, 17 pt too wide).
4. **Replace "Read A" with "single-cell read"** everywhere: the Fig. S6 source, `cellina_cf.tex` l.69, `transport_heldout.tex` l.52 and caption, `breakdown.tex` l.63, `S8_sensitivity_and_ablations.tex` l.28, `S5_readouts.tex` l.22, `S6_full_results.tex` S6.4. Unify "twin margin" with "twin read".
5. **Resolve the held-out contradiction:** §3.1 (`03_1_setup.tex` l.9) says every readout is on held-out cells; §3.6 says most relocation cells are in training tiles. Also add to §3.6 that the single-cell read falls on held-out tiles on 11 of 12 fits.
6. **Define the Table 4 contrasts in its caption** (`03_7_robustness.tex`): one line each for the signalling-gene lean, tumour axis, type-mean reference and spill-over part. Introduce the lean and the tumour axis somewhere in §§3.3–3.6, or move those rows to the supplement.
7. **Add a methods paragraph for the Fig. 3 drivers, landmarks and territory score** (new paragraph in `S6_full_results.tex` S6.1). Tabulate Moran's I of w, and distinguish programme I (Fig. 3a) from w I (Fig. 3d) in the `03_4_programmes.tex` caption.
8. **Reconcile the cross-seed cosine** ("0.67 / 0.72–0.98", `03_4_programmes.tex`) with Table S11 (0.76–0.98), and the MintFlow hours across §3.5, S6.6 and Table S18.
9. **In §3.7, state the reclassification facts:**
   - The relocation − programme contrast turns negative from κ = 0.3 on the primary section.
   - The relocation intervals have ~80% coverage.
   - Fix the §2.4 trajectory rule's wording so it covers the marker-pair case (`02_model.tex` l.104).
10. **Renumber the supplement's Proposition to S1** (`supplement.tex` theorem counter), add the code URL (`01_introduction.tex`, `04_discussion.tex`), and add a DISCELL QC/preprocessing sentence and the cluster-labelling method to S3.1–S3.2.

Smaller items:
- Qualify "only MintFlow carries less" (TMA core Cellina) and the abstract cost claim.
- Cut the development history (S2.3 Eq. S4 paragraph, S2.4 "implemented but off", S5.3 dev comparison, S7.4 last sentence, main §3.2 "than in development").
- De-duplicate the out-of-memory reasons and the Cellina-niche-domain text.
- Rename Table S2's "section A/B".
- S8 "conclusion" → "Discussion".
- Fix the Guide pointers (C1, C4b, C5, prose mapping).
- Give the paper a descriptive title (currently just "DISCELL").
