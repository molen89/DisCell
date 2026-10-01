# Critique of STORY_MAP.md (draft v1): publishability and truthfulness

Reviewer lens: a demanding RECOMB PC member. Checked against `docs/results_manifest.md`, devlog 2026-09-28 to 2026-10-01 (including the "Corrections to the Overnight results" entry and the transport−programme removal), the generated tables in `submission_paper/aistats/tables/generated/` (as rendered 2026-10-01 08:10), and the current `sections/*.tex`.

**Two facts to keep in mind while reading.** (a) Two queues are still running and can change wording: the MintFlow refits (every MintFlow number, not only reconstruction, is re-scored) and the breakdown-gaps queue (tumour axis on lung and FF; five more contrasts on the serial section; family sizes become 7/7/7/6/6). C1, C2 and C6 must not be frozen until both land. (b) The headline NMI (ovarian 0.713, `headline.tex`) and the battery NMI (0.708, `battery.tex`) differ for the same fits. A reviewer who sees both will ask why. Reconcile them, or say in one caption why the two differ.

---

## 1. Claim by claim

### C1: κ sweep and breakdown points
**As worded:** "the data bound κ only from above … every claimed contrast survives the whole grid … except two partial breaks."

- **Overreach, small.** Proposition 1 holds the neighbours' clean compositions fixed and works on the expected composition p_i, not on counts. "The data bound κ only from above" should be "even with the neighbours' clean profiles known, the counts bound κ only from above". No numeric bound is computed on any section. The plan should say so, or a reviewer will ask "what is the bound on your data?"
- **"Two partial breaks" is ambiguous.** There are two contrasts and three section-level breaks: FF signalling at 0.2; transport−leakage at 0.4 on ovarian and 0.3 on lung. The abstract gets this right; the plan should copy it.
- **Hidden-negative risk.** Two members left the claimed family after the results were seen: the marker pairs (2026-09-30) and transport−programme (2026-10-01). Both reasons are sound: one measures the decode, the other is zero by construction at κ = 0. But a reviewer reading the devlog-derived supplement will see a family pruned after the fact. State in the main text, in one clause, that both were reclassified as trajectories and why.
- **Missed selling opportunity.** The breaks are the best evidence that the test has power, and the plan treats them as a limit. Sell them as proof that the rule is not vacuous: "the sweep does break claims, and the ones it breaks are those a leak could plausibly produce (the signalling-gene lean, the transport margin over leakage)".
- **Missed opportunity, positioning.** The method text already cites Rosenbaum and Manski, but the story map never names the tradition. Comp-bio reviewers know "tipping-point" or "E-value" sensitivity analysis. One sentence placing κ\* there makes the idea read as principled rather than ad hoc. "Breakdown point" clashes with the robust-statistics term. Keep it, since that term is also about tolerated contamination and so fits, but define it on first use.

**Proposed wording:** "Even if the neighbours' clean profiles were known, the counts bound the leak fraction only from above. So we do not estimate it: we refit across κ ∈ [0, 0.4] (four times the typical misassignment) and report, for each claimed contrast, the smallest κ that explains it away. The test has teeth: it breaks the response's lean toward signalling genes on one section (κ\* = 0.2) and the transported prediction's margin over its leakage part on two (κ\* = 0.3, 0.4). Every other claimed contrast holds across the grid on every section."

### C2: the trade-off no competitor beats
**As worded:** "No comparison method beats DISCELL on both niche leakage and retained cell state on any section."

- **True only for one choice of axes.** It holds with the y-axis = cycle R² of z (q90) and the x-axis = the MLP composition probe. With NMI as the "state" axis, MintFlow dominates DISCELL on every section where it ran (NMI 0.90–0.99, MLP probe 0.3–3.0 % against 3.5–5.5 %). Cellina also dominates on the TMA core (NMI 0.84 against 0.62, MLP 3.5 % against 3.5–4.8 %). A reviewer will redraw your figure with NMI and call the claim axis-shopping. The plan's limit ("NMI is not the highest") does not answer this. It needs the argument: NMI with type is what MintFlow (which reads the labels) and Cellina (a classifier on z) are trained toward, so it measures label fidelity, not retained state. Cycle is the only label-free, within-type state read, and it was fixed for every method alike (devlog 2026-09-28). One sentence in §3.4, plus the honest note that the q90 set was chosen before the masked re-scoring but after earlier baseline reads.
- **The limit is too soft.** "On the TMA core Cellina's residual is level with DISCELL's" is wrong in direction. Cellina's MLP residual (3.5 %) is at or below DISCELL's best seed (3.5–4.8 %; fraction-left 0.51 against DISCELL's 0.52–0.71). Say "Cellina's nonlinear residual is at DISCELL's lowest seed".
- **Weaker than the data allow (missed opportunities):**
  - **Linear probe:** DISCELL's intrinsic latent has the lowest composition *and* image residual of all methods on all five sections, MintFlow included (`probe.tex`, ridge rows; e.g. ovarian 1.0–1.1 % against MintFlow 4.0 %, resolVI 7.4 %, Cellina 8.1 %). This is the cleanest dominance claim in the paper and it is absent from the plan.
  - **Mirror R²:** DISCELL is lowest on four of five sections (MintFlow is lower on the serial section, 0.035 against 0.039).
  - **FF:** DISCELL is best on both axes outright.
  - **Cellina's best case on the probe** (its adversary given our niche label) does not lower its composition residual consistently, and costs it NMI and cycle on every section. So "add an adversary to a competitor" does not reproduce the result. That supports the architecture as a contribution.
- **Coverage caveat.** SIMVI exists on the TMA sections only, and MintFlow on three. "On any section" is true, but the denominator differs per method. Say so in the caption, which the plan already half does.

**Proposed wording:** "Under a linear probe, DISCELL's intrinsic state carries the least niche composition and image information of the five methods on every section. Under a nonlinear probe, only MintFlow, whose intrinsic latent keeps almost no within-type state (cycle R² ≤ 0.06), carries less everywhere, and Cellina matches it on the TMA core while keeping less cycle state. No method both leaks less and keeps more cycle state on any section."

### C3: what z and w carry
**As worded:** "… the adversary removes most of the niche information a held-out probe can find."

- **Overreach.** Under the MLP probe, the fraction left is 0.52–0.71 on the TMA core and 0.59–0.70 on the serial section, so "most" is false there. It is true for the ridge probe everywhere (fraction left 0.07–0.20 on the fitted sections) and for the MLP on ovarian, lung and FF (0.14–0.43). The absolute residual is small everywhere (3.5–5.5 % of within-type variance), and that is the honest, sellable number. The fraction left is large on the TMA only because the uncontrolled model leaks little there (6.4–6.9 %).
- **Missing evidence that would make C3 robust to "you graded with your own probe":** within-type Moran's I of μ_z falls by 25–48 % with the adversary (`tab:moran`). This is a probe-free read.
- **Missing evidence for the held-out section:** the core's model reads the unseen serial section as it reads the core (NMI 0.61 against 0.62; cycle R² of z 0.50 on both). The plan mentions the serial section only in Setup.
- "The response does not carry the cycle" is supported (cycle R² of w ≤ 0.006; the asymmetry is above the grid on all five sections).

**Proposed wording:** "The intrinsic state keeps the cell type and the cell-cycle state, which the response does not carry (cycle R² ≤ 0.01), on every section and on a serial section never seen in training. The adversary cuts the niche composition a held-out linear probe finds by 80–93 % and leaves 3.5–5.5 % of within-type variance to a nonlinear probe. A probe-free read agrees: within-type spatial autocorrelation of z falls by 25–48 %."

### C4: what the response captures
**As worded:** "… and that shift is biologically structured", with evidence "tumour-axis test true ≫ false; GO localisation toward secreted/membrane".

- **Overreach on both pieces of biology evidence.**
  - *Tumour axis:* |τ| 0.78–0.81 against 0.51–0.56 is not "≫". Your own text says the contrast "is not specific to the response": predictions from z and from depth alone show a difference of similar size. It also fails in fibroblasts, and the "false" axis is correlated with the band (|ρ| ≈ 0.41).
  - *GO lean:* the manuscript calls it "a description, not evidence of a response", because a gene-specific leak would produce it. It also fades to null on FF.
  - Leading the biology claim with these two hands the reviewer the rebuttal you already wrote against yourself.
- **The strongest biology evidence is not in the plan:** subtype recovery, a clean double dissociation on held-out labels. On ovarian, tumour- versus stroma-associated location is read from w at balanced accuracy 0.95 (fibroblasts) and 0.90 (endothelium), against z 0.66 and 0.67. Tumour states (proliferative, inflammatory, …) are read from z at 0.79, against w 0.41 (floor 0.25). The exception (states that are also places: VEGFA⁺, myofibroblasts) is itself interpretable biology. This is what a RECOMB reviewer wants to see, and it fits in one small panel.
- **The limit belongs in the main text, not the supplement.** "At the operating weight w ≈ the context regression; no per-cell channel (planted test)" is the first thing an expert will find, and it reframes C4. "How each cell type shifts with its niche" is type-level; say so in the claim.

**Proposed wording:** "The response carries the niche: 0.47–0.60 nats beyond a within-type floor, against 0.06–0.11 for z. Held-out labels the model never saw split as designed: locations (tumour- against stroma-associated fibroblasts and endothelium) are read from the response, and intrinsic tumour states from z. At the operating point this is a type-level response: the per-cell channel is closed by design, as a planted per-cell response confirms."

### C5: the counterfactual and Cellina
**As worded:** "predicted well; … DISCELL leads on twin margin and on FF, Cellina on fraction of ceiling on two sections."

- **Factually off.** On lung FFPE Cellina's twin margin is 0.435 against DISCELL's 0.432 (`cellina_cf.tex` bolds Cellina). So "DISCELL leads on twin margin" holds on three of four sections.
- **Weaker than the data allow.** DISCELL leads on Read A on all four sections, with disjoint intervals on three (ovarian 0.79 against 0.68; lung 0.83 against 0.76; FF 0.82 against 0.60). Read A is the per-cell distributional read, the one closest to "moving a cell".
- **Cellina's TMA-core win rests on one trusted panel (0.899).** Say "on a single trusted panel" or drop the cell.
- **The held-out-tile read helps DISCELL against Cellina.** On held-out tiles Cellina's trusted fraction falls more (lung 0.70 → 0.49; ovarian 0.73 → 0.65) than DISCELL's. On lung, DISCELL (0.60–0.74) overtakes Cellina. Not in the plan.
- **"Predicted well" is unanchored.** 0.65–0.73 of the noise ceiling, with ~0.8 interval coverage and a partly in-sample published read, needs a reference: what a plain regression of type-mean expression on niche composition achieves on the same panels. The leakage-only part (0.21–0.39) is a reference the paper already has; use it.

**Proposed wording:** "Moved to another niche, a cell type's predicted shift recovers 0.65–0.73 of the reproducible between-niche difference, and the response programmes alone carry most of it. Against Cellina's neighbour-rewiring counterfactual on the same panels, DISCELL is closer to the target cells' own distribution on every section (Read A) and leads on all three reads on the fresh-frozen section. Cellina recovers more of the ceiling on the ovarian FFPE section, and on the TMA core on one panel."

### C6: reconstruction
**As worded:** "on par or better, even though the other methods trained on our held-out cells."

- **Accurate only as "on par".** DISCELL is best on ovarian FFPE and lung. Cellina (−7.309) and resolVI (−7.319) are above DISCELL (−7.320) on FF; resolVI is above it on the TMA core and serial section. The gaps are ≤ 0.03 nats per count. MintFlow's value is pending its refit.
- **Comparability risk.** DISCELL scores a multinomial, the others an NB or ZINB likelihood. A reviewer can question whether per-count held-out log-likelihoods are comparable across likelihood families. One sentence in the supplement on how the battery puts them on a common basis.

**Proposed wording:** "Held-out reconstruction is within 0.03 nats per count of the best method on every section, and best on two, although the comparison methods were trained on these held-out cells." Keep it to one sentence and give it no figure.

### C7: speed
**As worded:** "up to 2.6× faster than the fastest competing model … over 100× faster than MintFlow; SIMVI cannot fit the larger sections on a 24 GB GPU."

- **Accurate per fit.** The 2.6× is the TMA core against Cellina. On ovarian FFPE it is about 1.3×; on lung it ranges 1.3–3.1× over seeds; MintFlow is 100–420× slower. SIMVI's OOM is measured. Wording is fine.
- **Unanswered attack:** the method *is* the sweep, so the honest unit cost is 6 κ × 3 seeds = 18 fits. Pre-empt it, because it helps:
  - the full 18-fit sweep (~2.7 h ovarian, ~1.6 h lung, ~23 min TMA core) costs less than one MintFlow fit (9–16 h) and, on the TMA core, less than one SIMVI fit (3.2 h);
  - on FF the sweep (~11 h) costs more than resolVI's single fit (1.75 h). Say so.
- **Fairness note.** DISCELL trains on training tiles only and includes its evaluations; the baselines train on all cells and report training time only. The two biases run opposite ways. Say so once in the caption.

### C8: simulation
**As worded:** "recovers planted structure in simulation, also when κ is misspecified."

- **Supported.** At assumed κ 0–0.4 on worlds planted at 0.2, NMI is 0.79–0.97, CCA 0.78–0.94 and loadings 0.71–1.00, all close to the matched fit.
- **Missed opportunity.** Put this result to work for C1: because recovery barely changes with the assumed κ, the fit cannot point to the planted κ. That is the non-identifiability, shown in simulation. One sentence in §2.4 or §3.2.
- **Missed ablation in simulation.** Without the adversary, recovery of the response collapses (CCA 0.29–0.91 against 0.83–0.95; loadings 0.14–0.85 against 0.63–0.99) and composition leaks into z (R² 0.11–0.19 against ≤ 0.04). That is a ground-truth ablation of the adversary, and it is in `tab:synthetic` already.
- **Limit is understated.** At the final configuration the amortisation gap *grew* (0.65 / 1.02 / 1.93 against 0.55 / 0.83 / 1.48), and the encoder is now worse than a perceptron on the same inputs at every depth (devlog 2026-09-30). "Amortisation gap measurable" should read "the encoder's posterior mean is measurably off the exact one, more so at the final configuration than in development". Supplement is fine, but the conclusion's "small in simulation" must go.
- **Loadings range 0.63–0.99** reflects the bistability flagged in advance. State it as found.

---

## 2. Novelty and positioning

**What reviewers will see as the contribution:**
1. **The κ sweep with breakdown points**, a sensitivity-analysis treatment of segmentation leakage. This is the most novel and most portable idea; no competitor states what the data cannot identify.
2. **The first generative model in which a niche response and transcript leakage compete for the same counts.** resolVI has leakage but no niche term; SIMVI, MintFlow and Cellina have niche terms but no leakage.
3. **An evaluation protocol** (held-out probe against a permutation floor; context-side grading; transport against a noise ceiling) applied to competitors on the same cells. Reviewers value this, but will treat it as infrastructure.

**What they will see as incremental:**
- The type-conditional adversary: MintFlow already has type-specific discriminators, and Cellina has an adversary.
- The leakage term: it is resolVI's neighbour mixture with the weights fixed, which your own related work says.
- The two-bound objective (Klys, Slavutsky).

Expect the summary judgement "resolVI's leak term + a MintFlow-style split + a sensitivity sweep". The defence is that the combination is what creates the confound, and the sweep is what handles it.

**Is the sweep sold strongly enough? No.** In the plan it is C1 but argued only as "things survive". Three changes:
- (i) name the tradition (tipping-point or sensitivity analysis, partial identification) in the pitch;
- (ii) say the protocol transfers: any spatial claim from any model with a leak term can carry a κ\*;
- (iii) sell the breaks as evidence of power (see C1).

The abstract's sentence on κ\* is good. The intro's contribution (i) already says "applies to any spatial readout"; the story map should promote it to the one-sentence pitch.

**Is the comparison fair as framed? Mostly, but two asymmetries must be in the main text, not only the supplement:**
- **Calibration asymmetry.** DISCELL's α_z, α_w and adversary settings were calibrated on these sections (α_z on the primary section; α_w on the primary and TMA core; adversary on the primary, confirmed on two more). The baselines ran at their defaults. The manuscript states this in §4.1, but the story map's §3.1 lists only "baselines at their defaults (the comparison methods train on all cells)". That framing reads as if the only asymmetry favours the baselines. Add the calibration clause.
- **Cellina off-label.** Cellina runs without its domain adversary because there are no region labels; that sits between its full and ablated published models. The niche-domain row is the mitigation. Say "Cellina's best case on our probe" in the main text.
- **resolVI as a single-latent reference** is fair only if it is never presented as "beaten". It is not designed for invariance, and the probe grades it on something it does not attempt. The manuscript frames it correctly; the story map's C2 lists resolVI among "methods that keep more leak more". Keep it as a reference point, not a competitor in that sentence.
- **Label use** differs across methods (MintFlow and Cellina read lineage labels; SIMVI does not; DISCELL uses them in the adversary and the context encoder). Put this in the tab:baselines configuration table.

---

## 3. Reviewer attack surface: the 8 most likely objections

| # | Objection | Where the plan answers it | One sentence or panel that pre-empts it |
|---|---|---|---|
| 1 | "The surviving contrasts are model-internal (z vs w, I(niche; w) > 0) and survive nearly by construction; the ones that bear on biology are exactly those that break." | Not at all | Split the κ\* table into *allocation diagnostics* and *biological claims*. Say the breaks show the rule has power. Ideally add a planted-leak positive control in the supplement: a spatial contrast created purely by leakage in simulation, whose κ\* lands near the planted κ. This exists for nothing yet; it is the single most valuable addition to C1. |
| 2 | "DISCELL was tuned on the test sections; the baselines ran at defaults. And Cellina ran without its own adversary." | Supplement only (story map §3.1 omits the calibration) | Main text, §3.1: "DISCELL's weights were calibrated on these sections; the comparison methods ran at their published defaults and were trained on our held-out cells; Cellina, lacking region labels, is also shown with our niche label as its domain, its best case on our probe." |
| 3 | "Your dominance claim depends on choosing cycle R² as the state axis; on NMI, MintFlow and Cellina dominate." | Partly (limit: NMI not highest) | One sentence: NMI rewards label fidelity, which MintFlow and Cellina are trained toward; cycle is the only label-free within-type state read, fixed for all methods alike. Plus the ridge-probe result, where DISCELL is lowest everywhere, so the claim no longer hangs on one probe. |
| 4 | "If the per-cell channel is closed, w is just a regression on niche composition. Why not regress directly? Where is that baseline in transport?" | Supplement only (planted test) | Main text, §3.3 or §3.5: state "type-level response; per-cell channel closed by design" as a scoped result. Add the composition-regression reference on the transport panels, or argue why m_ψ(c, t) inside a model that removes leakage and keeps z is not the same as a plain regression on contaminated counts. The leakage-only part (0.21–0.39) against the programmes (0.52–0.64) is half of that argument already. |
| 5 | "The adversary and the probe come from the same function class; the nonlinear probe still finds over half the uncontrolled leakage on the TMA." | Partly (residual stated) | Reword C3 to absolute residuals (3.5–5.5 %) plus the probe-free Moran's I drop (25–48 %). |
| 6 | "Only one leak form is tested. A background floor moves your headline transport by 8.3 seed-SD, and a gene-specific leak would produce your LR and GO results." | Partly ("modelled form only") | Main-text limit: "A fixed false-positive floor lowers the primary section's transport read from 0.65 to 0.50; per-cell and per-gene leak forms move no read by more than 1.3 SD." Then cut GO from the main text (see C4). |
| 7 | "One section per tissue, all Xenium 5k; does anything generalise?" | Partly (serial section in Setup) | Sell the serial section as an out-of-sample replication (C3), and say "one platform" in the Discussion. |
| 8 | "The speed claim is per fit, but your method requires an 18-fit sweep." | Not at all | One sentence: the full sweep costs less than a single MintFlow fit on every section where MintFlow ran, and less than one SIMVI fit on the TMA core; on FF it costs more than one resolVI fit. |

**Runners-up, worth a clause each:**
- the transport read is partly in-sample: answered in the supplement by the held-out-tiles row; one main-text clause is enough;
- the cycle score is derived from the same contaminated counts: the main-text caveat exists; keep it to one sentence;
- MintFlow reconstruction is "n/r": resolved once the refits land;
- related-work table entries unverified: the CHECK flag is still open. RECOMB reviewers include the authors of MintFlow, SIMVI and Cellina, and a wrong ✓ or ✗ about their model is the fastest way to a reject.

---

## 4. Missing strengths (in the evidence, absent or buried in the plan)

| Strength | Evidence | Where it should go |
|---|---|---|
| **Subtype double dissociation**: locations → w (BA 0.90–0.95 against z 0.66–0.67), tumour states → z (0.79 against w 0.41, floor 0.25) | `subtype_recovery.md` (ovarian, 3 seeds, masked) | **Main text**, as the C4 biology panel instead of the tumour axis and GO |
| **Ridge probe: DISCELL lowest residual of all methods on all 5 sections**, composition and image | `probe.tex` | Main text, one sentence in C2 |
| **ω = 0 ablation**: without the intrinsic path, μ_w's NMI with type rises outside the seed range on 4/4 sections, and w's within-type share falls on 4/4 | devlog 2026-10-01; READOUT under `omega0_2026-09-30` | One main-text sentence in §2.2 ("load-bearing, confirmed by ablation"), with the code-drift caveat (z-side effects mostly within noise) in the supplement |
| **Without the adversary, w collapses** (zero within-type variance on 3 of 4 uncontrolled fits), and z's Moran's I is 25–48 % higher | devlog 2026-09-29 Moran table | Main text, one clause: the adversary is what makes the split exist, not only what cleans z |
| **Simulation ablation of the adversary**: CCA of w 0.29–0.91 without it, against 0.83–0.95 with it | `synthetic.tex` | Main text, §3.2, one clause |
| **Misspecified κ in simulation = non-identifiability shown** | `synthetic.tex` | Main text, §2.4 or §3.2, tied to C1 |
| **Held-out serial section** read without refit (NMI 0.61 against 0.62; cycle z 0.50 against 0.50; cycle asymmetry above the grid) | `headline.tex`, `breakdown.tex` | Main text, C3 |
| **Held-out-tiles transport**: no systematic in-sample inflation, and Cellina degrades more than DISCELL (lung 0.70 → 0.49) | `transport_heldout.tex` | One main-text sentence in §3.5; table in the supplement |
| **Cellina best-case variants**: the niche-domain adversary does not reliably cut its residual and costs NMI and cycle; on its own graph it does not catch up | `battery.tex`, `probe.tex` | One sentence in §3.4: it shows the result is architectural, not "any model plus an adversary" |
| **Read A beats Cellina on all 4 sections** | `cellina_cf.tex` | C5 headline |
| **Context-side grading**: DISCELL's 6-d w holds the most image-block information of any context latent on ovarian FFPE and lung | `context.tex` | Optional; one clause if fig:separation stays, else supplement |
| **Mirror R² lowest on 4/5 sections** | `battery.tex` | One clause in C2 |

---

## 5. What to cut (pages that do not buy acceptance)

1. **GO localisation in the main text.** It is descriptive, it is confounded with gene-specific leak by your own admission, and it fades to null on FF. Supplement.
2. **The tumour-axis test as the "biology panel".** It is not specific to w, fails in fibroblasts, and uses a non-orthogonal false axis. Keep it only as a κ\* row; replace the panel with subtype recovery.
3. **Model section budget, from 2.75 p to about 2.25 p.** Keep the generative model, the objective in one equation, the conditional adversary plus probe definition, and the Proposition plus Definition. Spend the freed 0.5 p on related work, which the plan does not budget at all. For RECOMB, positioning against resolVI, MintFlow, SIMVI and Cellina is non-negotiable. Keep a compressed tab:related only once every entry is verified.
4. **Separate claims C6 and C7.** Merge them into one sentence ("on-par reconstruction; one fit in 1–43 min; a full sweep cheaper than one MintFlow fit") with no float. The timing table goes to the supplement.
5. **Atlas cross-seed cosine and the transport rows in the main headline table.** Transport lives in §3.5; the atlas row does not serve a claim in C1–C8. Shrink the headline table to NMI, cycle z/w, probe residual, mirror and I(niche; w), which also fits a single column.
6. **The probe dumbbell as its own float.** Merge it into the trade-off figure as a panel, or replace it with one row of the headline table (the absolute residual). Six floats in 10 two-column pages is tight. The κ\* figure (7 contrasts × 5 sections of small multiples) will be unreadable at column width. Prefer the compact κ\* *table* (`tab:breakdown`, all text) in the main text and the figure in the supplement, unless the figure is redrawn as one summary strip.
7. **Marker pairs, KL maps, the symmetries proposition, the ego-masking table and the tile-split ceiling.** None appears in the plan's main text; keep them out, including any forward references from the main text.
8. **The Cellina counterfactual table at full size.** Three rows (fraction, Read A, twin) × 4 sections is enough. The "parts of the prediction" rows belong with DISCELL's own transport sentence.

---

## Bottom line

The plan is honest and well ordered, but it undersells its two strongest assets: the sweep as a principled sensitivity analysis that demonstrably breaks claims, and the clean held-out double dissociation (states → z, locations → w). It also leaves three reviewer attacks open that one sentence each would close: the contrasts survive by construction; the tuning asymmetry; the sweep's cost. C3 ("removes most") and C5 ("leads on twin margin") overstate the data and need rewording. C2 must stop depending on a single axis and a single probe. Freeze nothing in C1, C2 or C6 until the MintFlow refits and the breakdown-gaps queue have landed.
