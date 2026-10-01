# Critique of STORY_MAP.md (draft v1): ease of following for a RECOMB reader

Reviewer lens: a computational biologist who knows Xenium, segmentation and scVI-style models, but not posterior regularisation, partial identification or gauge symmetries. The paper has to stand alone in 10 pages, and this reader will probably skip the supplement.

Material checked: STORY_MAP.md; the AISTATS abstract, introduction, conclusion, method §2.4 (Proposition, Definition, sweep) and experiments §3 (section headers, transport paragraph, headline table); figures fig_tradeoff, fig_breakdown_data, fig_model_tissue, fig_model_graph, fig_latents_main and fig_kappa_bound.

---

## 0. The biggest single problem: "leakage" means two different things

The plan uses one word for two unrelated quantities:
- **transcript leakage**: misassigned transcripts, the leak fraction κ, the "leakage mixture";
- **niche information leaking into z**: C2's "niche leakage", the trade-off figure's x-axis "composition leakage of **z** (%)", "methods that leak less keep less".

C1 and C2 sit next to each other in the lead order, so a reader meets "leakage" in both senses within one paragraph. C2 then reads as if resolVI were *worse at handling contamination*, which is not the claim. resolVI is the only competitor that models contamination at all.

**Fix:** keep "leakage", or better "spill-over", only for transcripts (κ). Call the z quantity **"residual niche signal"** (or "niche information left in z") everywhere: the plan, the figure axis, the captions, the text. This is the change that does most for readability.

---

## 1. The one-sentence pitch

The current pitch is three sentences of about 70 words. A reader cannot repeat it. It also has two problems:
- "keeps a cell's intrinsic state free of its niche" overclaims. The probe still finds 3.5–5.5% (C3's own limit). Say "largely free" or "measurably freer".
- It never says what the reader *gets*: a robustness number they can trust.

**Proposed pitch (one sentence):**
> In imaging spatial transcriptomics, transcripts spilled from neighbouring cells look exactly like a cell's response to its neighbours; DISCELL models both, and because the data cannot tell how much spill-over there is, it reports for every spatial finding how much spill-over it would take to explain it away.

**Shorter, for talks and the abstract's first line:**
> Neighbour spill-over and neighbour response look the same in the counts, so DISCELL separates them and tells you, per finding, how much spill-over would be needed to explain it away.

Leave out speed and z-invariance. They are supporting claims, not the hook.

---

## 2. Claim order and narrative arc

**What works.** The lead order (C1 → C2 → C3/C4 → C5–C7 → C8) and the experiment order (simulation → z/w → comparison → counterfactual → κ sweep) are different, and that is defensible. A breakdown point only makes sense once the reader has seen the contrasts it grades, so the sweep belongs last in §3 as the climax.

**What breaks the arc:**
1. **The intro raises the question "is a spatial effect real or spill-over?" and §3 answers it only in its last subsection, 3.6.** In between, 3.3–3.5 report spatial findings with no stated status. Fix: in 3.1 add one sentence: "every contrast below is re-examined under assumed spill-over in §3.6". Also put a small "survives to κ = …" marker next to each claimed contrast as it appears, so the reader is never left wondering.
2. **C4 and C5 both rest on the same transport experiment.** C4 says the response "captures how each cell type shifts with its niche", and C5 says the counterfactual "is predicted well". These are one claim told twice, and §3.3 and §3.5 split it. Fix: merge them. §3.3 should cover z only (what the intrinsic state keeps). The w material (tumour axis, transport, Cellina head-to-head) should all go in one section, "What the response predicts".
3. **C2 before C3 is the wrong order for this reader.** C2's trade-off ("less niche signal at equal retained state") only means something once the reader knows what z is supposed to keep (type, cycle) and how niche signal is measured (the probe). Put C3 (what z carries, measured by the probe) before C2 in the lead order as well. The experiment layout already does this. Make the lead order agree.
4. **Simulation (3.2) as the first result is a speed bump.** It is C8, the lowest claim, and it costs 0.4 p before any real-data result. Two options. (a) Fold it into the setup as two sentences plus a supplement pointer. (b) Keep it only for the misspecified-κ result, which directly supports the sweep logic, and move it next to §3.6. I prefer (b): "planted κ = 0.2, fit at 0–0.4, recovery flat". That is the empirical counterpart of the Proposition and belongs with it.
5. **The biology payoff is thin and the plan hides this.** C4's honest limit is that at the operating weight w is essentially a regression on the niche descriptor. A RECOMB reader will ask "what did I learn about tissue?" Answering that with "tumour-axis test true ≫ false" and "GO localisation toward secreted/membrane" will not land. Either (a) give one concrete, named biological example in the main text (a cell type, a niche, the genes that shift, ideally ligand/receptor), or (b) frame the paper openly as a *method for trustworthy spatial claims* and not a discovery paper, and say so in the intro. The plan currently wants it both ways.

**Revised arc (each section answers the question the previous one raised):**
- Intro: spill-over and response look alike. How do we keep them apart without pretending to know κ?
- Model: three parts plus a κ sweep. Does the split actually do what it says?
- §3.3 z keeps type and cycle, drops most niche signal (C3). Is that better than existing models?
- §3.4 Yes: no method is better on both axes (C2), plus reconstruction and speed (C6, C7). And what does w give you?
- §3.5 w predicts how a cell type changes when moved to another niche, vs Cellina (C4+C5). But is that just spill-over?
- §3.6 Breakdown points: almost every finding survives κ up to 4× the typical rate. Simulation confirms the fit does not depend on assuming the right κ (C1, C8).

---

## 3. The model section

**The equation that must be in the main text** is the generative display already in the AISTATS intro:

x_i ~ Mult(ℓ_i, p_i),  p_i = (1−κ)·softmax(a(z_i) + B w_i) + κ·ρ̄_i,

with the three underbraces "intrinsic", "niche response", "spill-over". On its own this equation *is* the model for this reader. Make it the first display of §2 and refer back to it often.

**What is missing from the plan's §2:**
- **What c_i concretely is**, in one sentence: neighbour-type composition (Delaunay graph, 40 µm) plus a KRONOS image embedding of the surroundings with the cell itself masked out. "Deterministic niche descriptor (neighbour types + image)" is too abstract, and the masking is an important design choice that biologists will ask about.
- **The one intuition behind the Proposition.** A clean expression profile cannot have negative counts. Subtracting more and more of the neighbours' profile eventually drives some gene below zero, so that gives an upper bound. Any smaller κ fits equally well, so there is no lower bound. fig_kappa_bound panel (a) shows exactly this and is the best explanatory figure in the folder. Put it in the main text (as a Fig 1 panel or a small figure), and state the Proposition in words with the formal statement in the supplement.
- **Why the decoder gets no type label, and why invariance is conditional on type** (cell types cluster in space; making z blind to the niche unconditionally would erase cell type). This is the key design insight for biologists. One sentence each.
- **What m_ψ(c, t) means in words**: "the response expected for a cell of type t in niche c".

**What is superfluous for this audience:**
- The two-bound objective as an equation, and ω. Say "a variational objective with a type-conditional adversary on z" and move the equation to the supplement. The ω ablation itself is in the supplement, so introducing ω in the main text costs a symbol for no visible payoff.
- The "Definition (breakdown point)" as a formal environment. One sentence and the figure are enough (see §4).
- 2.75 pages is too much for "minimal math". Aim for 2.0–2.25 p including Fig 1, and give the freed 0.5 p to the biology example or to §3.5.

**Fig 1** has to carry the model. The existing material does not yet work as a Fig 1:
- fig_model_graph is a dense graphical model with GATv2 / μ_ψ / stop-gradient labels. At column width it is unreadable, and its labels are written for ML readers.
- fig_model_tissue (the leakage kernel on segmented cells, and the masked image field) is a good *panel*, not a figure.

**Proposed Fig 1, full width, about 0.4 p:**
- (a) tissue cartoon with a cell, its neighbours, and spill-over arrows (from fig_model_tissue a);
- (b) a simple block diagram, counts → z (intrinsic) and niche c → w (response) → expected profile, mixed with κ × neighbours' profile → counts;
- (c) the κ-bound simplex (fig_kappa_bound a);
- (d) a schematic breakdown point: one readout falling with κ and crossing zero at κ\*.

A reader who looks only at Fig 1 should then understand the paper.

---

## 4. Jargon: plain names and one-line definitions

| Term in plan | Will a RECOMB reader get it unaided? | Proposed name / one-line definition |
|---|---|---|
| leak fraction κ | Partly. "Leak" is ambiguous (see §0). | **spill-over fraction κ**: "the share of a cell's assigned transcripts that actually came from its neighbours (segmentation misassignment)". Reference point: ~0.1 typical on Xenium. |
| breakdown point κ\* | No. It is a robust-statistics term with a different meaning there. | Keep the symbol, and define it every time it first appears in a section: "**the smallest assumed spill-over at which a finding stops being significant**". Optional plain alias: "spill-over tolerance". |
| fraction left (probe) | No. "Fraction of what, left where?" | **residual niche signal in z**: "the % of within-type variation in neighbour composition that a held-out classifier can still predict from z, above a permutation baseline". Use the same name on the figure axis. |
| composition leakage of z (figure axis) | Misleading (§0). | Same as above: "residual niche signal in z (%)". |
| transport / counterfactual | "Transport" suggests optimal transport. "Counterfactual" is fine for some readers but vague. | **in-silico relocation**: "keep a cell type's intrinsic state, swap its niche, and predict the change in its mean expression". Score: **% of the reproducible shift recovered** (in place of "fraction of the noise ceiling"). |
| fraction of the noise ceiling | Borderline. | "% of the reproducible niche-to-niche shift", with the ceiling defined as split-half reproducibility in one clause. |
| Read A | No. An internal label. | Drop from the main text. If needed: "**single-cell relocation score**: how much of the distribution gap to the target niche the relocated cells close". |
| twin margin | No. | "**nearest-twin advantage**: a relocated cell lies X% closer to its intrinsic twin in the target niche than to a random one". Drop from the main text unless C5 needs it (see §5). |
| mirror R² (headline table) | No. It is not even in the plan's glossary. | Define or drop: "how well z can be predicted from the niche descriptor alone". |
| I(niche; w) excess | Readable with a gloss. | "**niche information in w** (above a shuffled-niche floor)". |
| cycle R² | Yes, if the proxy is named. | "cell-cycle score R² (S/G2M score from counts)". |
| operating weight / per-cell channel | No. | The main-text limitation should read: "at the settings used, w adds little beyond what the niche alone predicts". Drop "channel". |

Rule to add to the map's Writing rules: **no internal read-out names (Read A/B, mirror, twin, operating weight, finalL) in the main text.** Every metric has a plain-English name, and that same name is used on figure axes.

---

## 5. Figures and tables

**What carries the story (keep):**
- **Fig trade-off** (fig_tradeoff). Strong and immediately readable: DISCELL top-left, MintFlow bottom-left, the rest right. The "DISCELL, no adversary" open circle also shows the adversary's effect, which makes the **probe dumbbell redundant**. Drop the dumbbell from the main text, and do not merge it in as a panel.
  - Fixes: rename the x-axis (§0).
  - Give "Cellina, niche domain" and "Cellina, own graph" one line in the caption, or show only the default Cellina in the main text. Three Cellina markers look like a variant hunt.
  - Consider dropping the TMA serial panel, which repeats the TMA core.
- **Fig κ\*.** Essential, but fig_breakdown_data in its current form does not work for this reader:
  - It is 7 rows × 5 columns of small multiples.
  - Its row labels are internal ("Read A − type-mean reference", "I(niche; w) − its floor").
  - For the main text, use a **compact summary**: a grid of claims × sections, each cell showing κ\* (">0.4" / "0.4" / "0.3" / "0.2"), coloured, about 0.25 p. Full trajectories go to the supplement.
  - The plan says "two partial breaks". The figure and the abstract show **three** breakdowns: signalling share on Ovarian FF at 0.2, and transport − leakage part on Ovarian FFPE at 0.4 and on Lung FFPE at 0.3. Correct the plan's C1 wording.
- **Table headline.** Keep it, but cut it to the rows the text uses: NMI, cycle R²(z), cycle R²(w), niche information in w, relocation score. Drop reconstruction (that goes in the comparison sentence), mirror R² and atlas cosine, or define them. Show only means in the main text. The ranges make every cell two lines high and the table about 0.4 p tall.

**Essential but missing from the main text:**
- **fig_latents_main** (UMAPs: z grouped by own type, w by neighbours; within T/NK cells, z mixed over neighbours while w is sorted by them). This is the **most intuitive figure in the folder** for a computational biologist. It shows the whole z/w split in one glance, with no metrics needed. It should be the first results figure, in §3.3, and it can replace the dumbbell's slot. It is the best "does the split do what it says?" evidence for this audience.
- **fig_kappa_bound (a)**, as a Fig 1 panel (see §3).
- **One biology panel for w** (if the paper keeps a biology claim): a named niche contrast with its top shifted genes. The plan lists "one biology panel" under C4 but gives it no float. Either budget a float or downgrade C4 to a sentence.
- The **related-work comparison table** (tab:related, compact). For RECOMB it replaces about half a page of related-work prose, and readers scan it. The plan drops it silently. Recommend keeping it single-column, with 5 rows × 4 columns (latent split / models spill-over / how spill-over set / invariance), if the float count allows. Otherwise make it the first supplement table and add a one-line pointer in the intro.

**What a reader would skip (cut from main):**
- The probe dumbbell (redundant with trade-off).
- The Cellina head-to-head as a separate table. It is a mixed result on metrics the reader has just learned. Make it two sentences in §3.5 with a supplement pointer, unless C5 stays a headline claim. Given that it is "mixed, stated as such", it should not be a headline claim (see change 7).
- The timing table. One sentence is enough (C7).

**Resulting float set (5):** Fig 1 overview; Fig latents (UMAPs); Table headline (trimmed); Fig trade-off; Fig κ\* summary grid. Optional sixth: a biology panel or tab:related.

---

## 6. Page budget

Assumptions: two-column, 10 pt, 1-inch margins, about 900–1000 words per full page of text.

| Item | Plan | Realistic estimate |
|---|---|---|
| Floats | "max 6" | Fig 1 full width 0.4 p; latents half-width 0.35 p; headline table 0.3 p (trimmed); trade-off full width 0.45 p; κ\* grid 0.25 p (or 0.55 p as the current small-multiples figure); optional 6th 0.3 p. Total **≈ 1.8–2.4 p** |
| Text left | — | 10 − 2.1 ≈ 7.9 p ≈ **7,100–7,900 words**, minus headings and display equations (~0.3 p) ≈ **6,800–7,500 words** |
| Plan's section sum | 1.25 + 2.75 + 4.75 + 0.75 = **9.5 p** | Leaves 0.5 p of slack. Fine, *if* floats are inside the section budgets. The plan does not say whether they are. Make that explicit. |
| Model §2 | 2.75 p | Current method.tex is about 10,600 words including comments and flags, so the plan implies cutting about 80%. Feasible only if objective, gauge and batching all go. Target ≤ 2,000 words plus Fig 1. |
| Experiments §3 | 4.75 p | With 3–4 floats (≈1.5 p), about 3.2 p of text ≈ 3,000 words. Current experiments.tex is about 5,200 words, so this is a 40% cut. Realistic if Read A / twin / fold counts / external criteria / marker pairs leave. |
| Intro 1.25 p | 1.25 p | Too tight if related work stays as prose. The current intro is about 2,000 words (≈2.1 p). Needs tab:related or a ruthless one-paragraph related work. |

Verdict: the budget is **achievable but tight**. The risk is in §2: the plan sets "minimal math" and 2.75 p side by side, and the extra page will fill up with ML detail. Move 0.5 p from §2 to §3.5 or the biology example.

---

## 7. Things the plan gets wrong or leaves inconsistent

1. **C1's "two partial breaks"** is really three breakdowns across two contrast types (signalling share on FF at 0.2, and transport − leakage part on Ovarian FFPE at 0.4 and on Lung FFPE at 0.3). Also, the transport-vs-programme-part margin is *positive only from 0.05 to 0.2* and negative at the top on the primary section. The abstract reports this, and the plan's C1 omits it. Under "no hidden negatives", say it in the same place.
2. **The pitch overclaims invariance** ("free of its niche") against C3's own limit.
3. **C4's claim** ("captures how each cell type shifts with its niche") and **its limit** (w is essentially the context regression) undercut each other. Rephrase C4 to: "the niche-conditional response predicts how a cell type's mean expression shifts between niches". That is what the evidence supports, and it is type-level, not per-cell.
4. **C5 mixes a headline-sounding claim with a mixed result** and uses three metrics, two of them jargon, to say it. Pick one metric or downgrade.
5. **C7's wording** ("up to 2.6× faster than the fastest competing model (slower than Cellina on FF)") contradicts itself on first reading. Better: "fits a million-cell section in under 45 min on one GPU; as fast as Cellina, 2–20× faster than resolVI, >100× faster than MintFlow; SIMVI did not fit in 24 GB on three sections". Numbers to be checked against tab:timing.
6. **The C2 limit** "DISCELL's type agreement (NMI) is not the highest" is fine. But on the TMA core, DISCELL's error bar on residual niche signal overlaps "Cellina, own graph", and Cellina (default) has *less* residual niche signal (3.5 vs 4.0). The C2 sentence "no method beats DISCELL on both" is still true. The C2 limit should name the TMA core explicitly, as it already half-does.
7. **The comparison methods "train on all cells"** (setup, and C6 "even though the other methods trained on our held-out cells"). Good, but say it once in 3.1, not again under C6.
8. **The code link** is required by RECOMB. The plan places it in the Discussion. Also put it in the abstract or as a footnote on page 1, which reviewers look for. Per the code-release plan, this will be the new clean repo.
9. **"Main text must stand alone"** conflicts with pointing limitations to "the supplement in full". That is fine, but every limitation that changes a number in the main text (for example the false-positive floor lowering transport 0.65 → 0.50) must appear in the main text.

---

## 8. Top 10 changes to the map (prioritised, one line each)

1. Use "leakage/spill-over" only for misassigned transcripts (κ); rename z's metric "residual niche signal" everywhere, figure axes included.
2. Replace the 3-sentence pitch with one sentence: spill-over and response look alike → DISCELL models both → reports, per finding, how much spill-over would explain it away.
3. Add fig_latents_main (z by own type, w by neighbours) as the first results figure; drop the probe dumbbell.
4. Redesign Fig 1 for biologists: tissue spill-over cartoon + simple block model + κ-bound simplex + schematic breakdown point; keep fig_model_graph for the supplement.
5. Main-text κ\* figure becomes a claims × sections grid of breakdown values; trajectories go to the supplement; fix "two partial breaks" to the three actual breakdowns.
6. Make the generative equation with "intrinsic / response / spill-over" underbraces the one display of §2; move the two-bound objective and ω to the supplement; cut §2 to ≤ 2.25 p.
7. Merge C4 and C5 into one "relocation" claim at type level, scored by one plain metric (% of reproducible shift); Cellina head-to-head becomes two sentences; Read A and twin margin go to the supplement.
8. Ban internal read-out names (Read A, mirror, twin, operating weight, channel) from the main text; add a one-line plain definition at first use for κ, κ\*, residual niche signal and relocation score.
9. Decide the biology stance: one named niche/gene example as a float, or frame the paper explicitly as a method for robust spatial claims; do not leave "tumour-axis true ≫ false" as the biology.
10. Move the simulation next to the κ sweep (misspecified-κ recovery as the empirical twin of the Proposition), put C3 before C2 in the lead order, and state in 3.1 that every contrast is re-checked under spill-over in 3.6.
