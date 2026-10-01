# Integration report, 2026-10-01

## Build
- `./build.sh`: **main text 10 pages (limit 10)**, 0 undefined references, 0 undefined citations, 0 overfull boxes in main.
- Supplement: see "Supplement pass" below for its undefined references and overfull boxes.
- Margin: page 10 is full (the κ\* table sits at its top). Any growth will push past 10. Two things will shrink it: the `\pending` markers turning into numbers, and the 3.6 regression slot taking one sentence. The planted-control result in 3.7, by contrast, will add about 2 lines.

## 1. Generator (scripts/paper_tables.py, tests/test_paper_tables.py)
- New `headline_main` writes `tab:headline-main`. It is one column, means only, with the rows of STORY_MAP:
  - NMI of z;
  - cycle R² of z and of w;
  - residual niche signal (MLP composition, 1 − e^{−2·excess}, mean of the three final fits, as tab:probe);
  - mirror R²;
  - I(niche; w).
  
  The rows share the read-and-check code of tab:headline (`_headline_data`). It uses fewer digits (NMI and cycle z 2, cycle w and mirror 3, residual 1, MI 2), and gives arrows by the direction map plus provenance comments.
- New `breakdown_main` writes `tab:kappa-star`. It is one column, with row groups "Biological claims" and "Allocation checks". Cell values:
  - `>0.4`;
  - a bold κ\* with an a/b footnote;
  - `n/a`, with the reason in the caption;
  - a violet `?` for a family member the record does not hold yet; the caption then carries `\pending{breakdown gaps queue}`.
  
  The expected family comes from `discell.experiments.breakdown.FAMILY` / `NOT_APPLICABLE`, so the generator works on both the old (09-29) and the new (gaps) record format. **Rerun after the gaps queue (~21:00): `python scripts/paper_tables.py --only breakdown_main`.** The queue's own render step regenerates only `breakdown` and `breakdown_traj`.
- Existing outputs are unchanged. I regenerated all 14 old tables with the old and the new script into scratch: they are identical except the timestamp and command lines. Only the two new files were written to `tables/generated/`.
- Tests: two new round-trip tests, and both labels were added to the header/label test. 39 pass.
- 03_3 now inputs `headline_main` and 03_7 inputs `breakdown_main`; the hand tables are gone.
- `tab:headline` (the old full-width table) is no longer in the main text, but the generated tables refer to it. The supplement now inputs `headline.tex` in S5, before `headline_full`, so those references resolve.
- `supplement.tex`: added `\newcommand{\suppref}` (in the supplement it reads "Section S…"). Main-text captions imported through xr contain `\suppref`, and without the definition the supplement build failed.

## 2. Cross-file consistency (main text)
- **Definitions once:**
  - mirror R² is defined in the tab:headline-main caption (its first use); 3.5 refers back;
  - residual niche signal is defined in §2.3; the 3.5 figure axis and the table refer back;
  - NMI is defined in §2.3;
  - relocation is defined in 3.6 (italic, first use).
- **Planted spill-over control:** only in 3.7. 3.2 points to it in one sentence.
- **Repeats removed:**
  - The 4× margin is stated once, in §2.4. It is gone from the intro bullet and 3.7 (the abstract keeps it as a summary).
  - The holding rule is stated only in Definition 1. 3.7 now refers to §2.4 for the trajectory rule.
  - The fairness asymmetry is stated only in 3.1. Removed: "although the comparison methods were trained on these held-out cells" (3.5) and "same read-outs on our held-out cells" (3.5). The Discussion limitation refers back to 3.1.
  - The cycle-proxy caveat is stated only in 3.3. The Discussion repeats it as a limitation without detail.
  - The tipping-point/partial-identification framing is now only in §2.4 (cut from the intro).
  - "resolVI's mixture with fixed weights" is said once (intro).
  - The non-identifiability-in-simulation point is made fully in 3.2; §2.4 keeps a pointer clause only.
- **Vocabulary:**
  - "fresh-frozen section" became "FF section" after its definition in 3.1;
  - resolVI is cited everywhere in the main text as `ergen2026resolvi` (Nature Methods 2026), replacing `ergen2025`;
  - the kappa-table rows were reworded to plain names.
- `fig:latents-main` was dropped from S6 by the supplement pass; see below.

## 3. Supplement pass (supplement/ only; done by a supplement agent, checked in the joint build)
- **Result:** 0 undefined references, both ways, and no label defined in both documents. Before this pass there were 22 undefined references, and `fig:latents-main` was defined in both.
- **Moved content, ported from the AISTATS method; only what was missing was added:**
  - **S1 app:bound:** isolated cells (`eq:S-renorm`) and "why a multinomial".
  - **S1 app:pathb:** scaling of the weights.
  - **S1 app:gaussian-mi:** the closed-form penalty (`eq:S-penalty`), its estimation, and the escalation rule.
  - **S1 app:symmetries:** `prop:S-symmetries` and the gauge discussion.
  - **S2:** the mirror selection channel, the dropped neighbour means, explaining away (app:qw), the two requirements and the subclone caveat (app:conditional), why κ is fixed, and the variances.
  - **S3:** the adversary's targets and λ_y; warm-up, diagnostics and seeds; two hops in one pass; held-out neighbour exposure.
  - **S5:** "what the response is and is not".
  - **S8:** the claimed contrasts and the rules (Bonferroni, the serial-section rule, trajectories).
  - **S9:** what the spill-over term assumes.
- **Retargets:**
  - `eq:penalty` → `eq:S-penalty` (S1, S7);
  - `prop:symmetries` → `prop:S-symmetries`;
  - `fig:battery` → `tab:battery`;
  - `fig:probe-data` → `tab:probe`;
  - `fig:breakdown-data` is now a real figure in S8 (fig_breakdown_data). Its internal axis labels still say "leakage", and the caption maps them;
  - the stale `fig:overview b` references → `fig:mask-radius` / app:image;
  - the β arrow-width reference to Fig 1a is kept (still valid);
  - several `sec:objective`/`sec:inference`/`sec:batching`/`sec:generative` references whose content left §2 → app:pathb, app:tiles, app:conditional, app:limitations.
- **`fig:latents-main`:** the float was removed from S6, and S6's sentence points to the main-text figure.
- **`tab:headline`:** now input in S5, before `headline_full`, so the generated tables' references to it resolve.
- **Vocabulary in the supplement text:** leakage → spill-over, leak fraction → spill-over fraction, foreign influx → spill-over influx, transport → relocation for the counterfactual read. Generated tables are untouched; their captions still say "leakage"/"transport" (see "Needs the author").
- **Plain names added once:** "single-cell relocation score (Read A)" and "nearest-twin advantage (twin margin)".
- **Overfull boxes:** 8 remain in the supplement.
  - tab:contact in S2 is 14 pt over (it was before this pass);
  - the others are inside generated tables or ≤ 1 pt.
- **Flagged by the agent:**
  - S3 tab:deviations says the escalation rule "decides on each section". The AISTATS todo says it was triggered once, on the primary section only. Check that wording.
  - S8's contrast-family paragraph describes the pre-gaps family (6/7, cycle only on serial). I marked it `\pending{breakdown gaps queue}`.

## 4. Abstract and Discussion
- Abstract (~230 words). Order: problem → idea (model both; sweep; breakdown point) → C2 → C1's teeth → biology (territories, the stromal-remodelling programme, the subtype dissociation) → speed. C2 carries `\pending{MintFlow refits}` and C1 carries `\pending{breakdown-gaps queue}` (freeze rule).
- Discussion (~0.45 p):
  - one paragraph on C1, with why it matters (transferable; lets a reader judge a claim by its tolerance);
  - one paragraph on C3/C2/C4/C4b/C5, each with its section reference;
  - one limitations paragraph: modelled form only with the false-positive floor (refers to 3.7, no number); one platform; type-level response; cycle proxy; calibration on these sections; S9 pointer;
  - the code link.

## 5. \needsource in 03_4 (all four resolved; sources opened and passages quoted by a verification agent)
1. COL11A1/MMP11/COL10A1/INHBA: **verified**. Cited as "genes of the cancer-associated fibroblasts of invasive tumours, ovarian cancer included" (`kim2010multicancer`, `zhu2021adipose`).
2. C7, DPT: **verified** as non-activated stromal fibroblast genes (`zhu2021adipose`, `buechler2021crosstissue`). **TNXB is not sourced**: it appears only in a table of Zhu 2021. It is now listed outside the sourced label (the stricter option).
3. Lung: only **CCL19** is sourced (immune-interacting fibroblasts in inflamed tissue, `korsunsky2022stromal`). The collective label "inflamed stroma" was dropped, and PLVAP/ADAMDEC1/MMP9 are listed without a role. ADAMDEC1 is a healthy-colon stromal marker in Kinchen 2018, so the old label was wrong for it.
4. TMA: MFAP5 and PI16 (adventitial) and NPNT (alveolar) are **verified** (`tsukui2020collagen`, `hanley2023lungcaf`). PDGFD, CCN5 and FAT3 are listed without a sourced role.

Six bib entries were appended to ../aistats/references.bib, with DOIs checked against Crossref. `korsunsky2022stromal` lists the first 10 of 35 authors plus "and others".

## 6. Figures at print size (rendered at 130 dpi and inspected)
- **Fig 1:** polished by a figure agent.
  - Panel b: padding enlarged, labels cleared, all text ≥ 6.5 pt.
  - Panel a: the β_ij label moved to a legend line under the cartoon.
  - Panel d: tick labels only at 0/0.2/0.4.
  - Height is now 2.15 in.
  - Clean at print size.
- **Programmes figure:** legible. The smallest text (the axis labels in c and the gene names) is about 5–6 pt and readable.
  - Note for the author: panel a prints the Moran's I *of the leading programme's score* (0.45–0.97, FF 0.97). The text quotes Moran's I *of w* (0.48–0.73, panel d). Both are correct, but a reader may conflate them. Consider labelling panel a "score I".
- **Trade-off figure:** legible. The corner annotation "little niche signal, keeps cycle state" crosses a grid line, which is cosmetic.

## 7. What was cut to reach 10 pages (prose only; no result and no limit removed)
- **Intro:**
  - the "read together with cells around it" clause;
  - the tradition sentence (now only in §2.4);
  - a duplicate resolVI sentence;
  - the "ligands and receptors" clause;
  - the NCEM CVAE clause;
  - "the penalty on z" for SIMVI;
  - the contribution-bullet details (probe description).
- **Model:**
  - the Fig 1 caption, shortened;
  - the posterior-input explanation;
  - the "I am a T cell" example;
  - the simulation sentence in §2.4 (a pointer is kept).
- **3.2:** the recovery and adversary paragraphs merged; the positive control moved to 3.7.
- **3.3:**
  - the figure narration compressed;
  - the subtype intro;
  - the type-level paragraph merged into fewer sentences (the "separates some clusters better than its prior mean" detail on lung was cut; "partial exception" kept).
- **3.4:** "what matters is which"; the FF paragraph condensed ("the model cannot tell the two apart" folded into "explains it as well as a response does").
- **3.5:**
  - the axis-choice sentence condensed;
  - the per-method list for "more cycle state, more residual" cut;
  - the caption, shortened;
  - the pointer to supplement tables.
- **3.6:** the two planned regression sentences moved to a LaTeX comment, with a single `\pending`.
- **3.7:** the pending notes shortened; the "if κ\* does not track" contingency moved to a comment.
- **Discussion:** one sentence on speed, and one on Cellina being mixed (3.6 says it).

## Open \pending (main text)
- Breakdown-gaps queue (~21:00):
  - the intro bullet and abstract C1 wording;
  - the 3.7 "recount" note;
  - the `?` cells and caption of tab:kappa-star (rerun `--only breakdown_main`).
- MintFlow refits:
  - the abstract C2;
  - the fig:tradeoff caption;
  - the 3.5 C2 paragraph;
  - the 3.5 reconstruction/speed paragraph.
- Planted spill-over control: 3.7.
- Plain-regression reference: 3.6, plus the one follow-up sentence in the comment there.

## Needs the author
1. Rerun `python scripts/paper_tables.py --only breakdown_main` after the gaps queue, then recount the breaks in 3.7, the intro bullet, the abstract and the Discussion ("breaks the findings that spill-over could most plausibly produce" must still be true).
2. TNXB, PLVAP, ADAMDEC1, MMP9, PDGFD, CCN5 and FAT3 are named without a sourced role (see §5). Accept this, or drop the genes.
3. Page 10 has no slack. When the pending results land, keep each to one sentence.
4. The abstract runs ~230 words against ~200.
5. Programmes figure, panel a label (see §6).
6. The title is still "DISCELL", and the author block is a placeholder.
7. The generated tables still use the AISTATS vocabulary ("leakage", "transport", "leak fraction") in their captions and row labels. This is visible in the supplement: tab:breakdown, tab:headline, tab:battery and others. Changing it is a generator change (the existing outputs would change), so I left it for your decision.
8. The S3 escalation-rule wording and the S8 family paragraph after the gaps queue (§3).
9. No commits were made. scripts/paper_tables.py, tests/test_paper_tables.py, references.bib (6 entries), recomb_fig1_overview.py and the recomb/ files are modified or new.
