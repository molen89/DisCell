# DISCELL for RECOMB: story map (v2, 2026-10-01, after two critiques)

Format: Letter paper, 10 pt, 1-inch margins, two columns. A title page carries the title, authors (placeholders for now) and abstract; the bibliography is not counted. **10 pages** of main text, figures and tables included. The supplement is a separate PDF and may not be read, so **the main text must stand alone**. A code link is required (the clean repository comes later).
Critiques: `CRITIQUE_clarity.md`, `CRITIQUE_publishability.md`.

## Vocabulary (used consistently everywhere)
- **Spill-over:** transcripts assigned to a cell that belong to its neighbours. Its share is the **spill-over fraction κ**. ("Leakage" is never used for anything else.)
- **Residual niche signal:** what a held-out probe can still read about the neighbourhood from z, beyond chance.
- **Breakdown point κ\*:** the smallest spill-over fraction that explains a finding away.
- **Relocation (counterfactual):** predicting how a cell type's expression shifts when it is moved to another niche.
- Internal names (Read A, twin margin, mirror, transport) stay out of the main text or are defined in one line.

## Pitch (one sentence)
Spill-over between neighbouring cells and a cell's response to its niche look alike in spatial transcriptomics counts; DISCELL models both, keeps the cell's intrinsic state apart from its niche, and reports for every spatial finding how much spill-over would be needed to explain it away.

## Claims (each with its main-text evidence and honest limit)

| # | Claim (paper wording, final numbers after the running queues) | Main-text evidence | Limit, stated once |
|---|---|---|---|
| C1 | **Spill-over cannot be estimated from counts, only bounded from above; so we sweep it and report breakdown points.** This is a sensitivity analysis in the tipping-point tradition, and it transfers to any model with a spill-over term. κ\* orders findings by how much spill-over would have to be assumed to lose them; it neither estimates κ nor certifies a finding (author, 2026-10-01: no "teeth" framing). Six cells over three contrasts break within the grid (tumour axis on lung 0.4 and FF 0.3; FF signalling lean 0.2; relocation beyond its spill-over part on ovarian 0.4, lung 0.3, serial 0.4), everything else holds to κ = 0.4; on the serial section every contrast replicates the core except relocation beyond spill-over (up to 0.3). Planted control (corrected contrast): null world no finding, κ_true 0.1 → κ\* 0.05, 0.2 → 0.1 (one step early, conservative); genuine response holds through 0.2. | Proposition + Definition; compact **κ\* table** (claims × sections; findings split into "biological claims" and "allocation checks"); **planted-spill-over positive control** (NEW, see below) | Only spill-over of the modelled form (one-hop, section-wide) is ruled out |
| C2 | **No method both carries less niche signal in its intrinsic state and keeps more of the cell's own state, on any section.** Under a linear probe DISCELL has the least residual niche signal of all five methods on every section; under a nonlinear probe only MintFlow has less, with almost no within-type state left (cycle R² ≤ 0.06). Mirror R² is the lowest on 4 of 5 sections. | **Fig trade-off** (residual niche signal vs cycle state, per section) | NMI with type is not the highest; why cycle state is the right axis (label-free, within-type) is said in one sentence; resolVI is shown as a single-latent reference, not a "beaten" competitor |
| C3 | **The intrinsic state keeps type and cell cycle; the response does not carry the cycle.** The adversary cuts the residual niche signal by 80–93 % under a linear probe, leaving 3.5–5.5 % of within-type variance to a nonlinear one, and spatial autocorrelation of z falls by 25–48 %. All of this holds on a serial section never seen in training. | **Table headline** (slim: NMI, cycle z/w, residual signal, mirror, I(niche; w); 4 sections + serial) | The cycle label is a count-derived proxy; on the TMA sections more residual signal remains under the nonlinear probe |
| C4 | **The response carries the niche, and held-out sublabels split as designed:** locations (tumour- vs stroma-associated fibroblasts and endothelium) are read from the response, and tumour states from the intrinsic state. Without the adversary, the response collapses. | I(niche; w) row; **subtype double-dissociation panel** (ovarian; BA w vs z) | At the operating weight the response is type-level: no per-cell channel (planted test, supplement) |
| C4b | **The response's programmes are interpretable, tissue-scale biology.** Each section's leading response programme is spatially organised (within-type Moran's I of w ≈ 0.5–0.7, against ≤ 0.11 for z) and carries a recognisable signature (e.g. on ovarian FFPE: a stromal-remodelling/EMT programme, COL11A1, MMP11, INHBA against C7, DPT, TNXB, mapping a territory around the tumour), with hallmark labels recurring across seeds. What drives each programme (neighbour composition, image context, distance to landmarks) is decomposed. | **Programme-atlas figure** (NEW to the main text, from the existing run atlases): per section a territory map + top genes + hallmark label + drivers; the Moran's I contrast z vs w in its caption/text | Programmes are identified only up to rotation, so only run-internal summaries are shown; FF's labels are less stable across seeds (stated); verify every label and number at the final configuration before writing |
| C5 | **Relocation exercises the whole model: a cell type's predicted shift recovers 0.65–0.73 of the reproducible between-niche difference**, with the response programmes carrying most of it. A plain regression on raw counts is as accurate or more on the observed shifts (reference column R), which contain the spill-over, so this does not decide between them; DISCELL's advantage is the separation (author, 2026-10-01). Against the clean (spill-free) truth in simulation, DISCELL's programme part keeps R² ≈ 0.76 at every κ_true (0–0.2) while the regression degrades (log 0.40 → 0.14; rate 0.25 → below 0); the full counterfactual degrades by design (it targets the observed shift). Against Cellina's neighbour-rewiring counterfactual on the same panels, DISCELL is closer to the target cells' own distribution on every section, and leads on all reads on FF; Cellina recovers more of the noise ceiling on ovarian FFPE (and on the TMA core, on one panel). On held-out tiles DISCELL holds up; Cellina drops more. | Short relocation table (3 reads × 4 sections, DISCELL vs Cellina) | Mixed against Cellina, stated as such |
| C6 | **Competitive and fast:** held-out reconstruction within 0.03 nats/count of the best method on every section and best on two, although the comparison methods trained on our held-out cells. One fit takes minutes: up to 2.6× faster than the fastest competitor (slower than Cellina on FF) and over 100× faster than MintFlow. The full 18-fit sweep costs less than one MintFlow fit. SIMVI cannot fit the larger sections on a 24 GB GPU. | Two sentences; timing table in the supplement | — |
| C7 | **In simulation it recovers planted states, responses and programmes, also when κ is misspecified, and the adversary is what makes the response recoverable** (CCA 0.83–0.95 with it, against 0.29–0.91 without). | 1 compact table or panel | The amortisation gap is measurable (supplement) |

**Freeze rule:** C1, C2 and C6 are worded finally only after the breakdown-gaps queue and the MintFlow refits land (both running).

## Fairness, stated in the main text (§3.1), not only in the supplement
DISCELL's weights were calibrated on these sections. The comparison methods ran at their published defaults and were trained on all cells, including our held-out cells. Cellina lacks region labels, so it is also shown with our niche label as its domain, its best case on our probe. Label use per method goes in the configuration table (supplement).

## Layout (10 pages)

**Title page:** title, placeholder authors, abstract (~200 words: confound → idea (model both; sweep spill-over; breakdown points) → C2 → C1's teeth → speed).

**1 Introduction [1.5 p]:** the three sources, and why spill-over and response are confounded (one paragraph). **Related work [0.5 p, budgeted]:** resolVI (spill-over, no niche term); SIMVI, MintFlow, Cellina (niche terms, no spill-over); none states what the data cannot identify. A compact comparison table, kept only once every entry is verified against the source papers. Contributions: four bullets (model; invariance + held-out probe; spill-over sweep with breakdown points, transferable; evaluation on 4 sections + serial vs 4 methods with code).

**2 Model [2.25 p], clean math:**
- 2.1 Generative model: the multinomial of the mixture (1−κ)ρ_i + κ ρ̄_i; log ρ_i = a(z_i) + B w_i; the w prior m_ψ(c_i, t_i), with c_i a deterministic niche descriptor. **Fig 1 redesigned:** tissue cartoon → block model → the κ upper bound → a schematic breakdown point.
- 2.2 Inference and objective: one equation; one sentence on the intrinsic path, whose role an ablation confirms.
- 2.3 Conditional invariance: adversary + held-out probe; definition of residual niche signal.
- 2.4 What is identified: Proposition (κ bounded from above only) + Definition (breakdown point) + the sweep.

**3 Experiments [5.6 p; the budget is taken from the Introduction (1.5 → 1.3) and the Discussion (0.75 → 0.6), and is to be confirmed at the first full draft]:**
- 3.1 Setup [0.4 p]: sections (+ serial), lineage labels, splits and seeds, baselines and the fairness paragraph, operating point κ = 0.1 (typical misassignment; MisTIC).
- 3.2 Simulation [0.5 p]: C7, plus the planted-spill-over positive control for C1.
- 3.3 What the intrinsic state and the response carry [1.2 p]: **latent figure first** (z sorts by the cell's own type, w by its neighbours), then the slim headline table and the subtype panel. (C3, C4)
- 3.3b What the response encodes: the programme atlas [0.8 p]: one figure across the sections (territory, signature, hallmark, drivers), combined with the Moran's I contrast (w spatially organised, z not). This is the RECOMB-facing biology result. (C4b)
- 3.4 Comparison [1.1 p]: fig trade-off; one sentence each on the linear probe, mirror, Cellina's best-case variants (the result is architectural, not "any model plus an adversary"), reconstruction and speed. (C2, C6)
- 3.5 Relocation [0.8 p]: the short table and the held-out-tiles sentence. (C5)
- 3.6 Robustness to spill-over [1.0 p]: the κ\* table and the positive control. (C1)

**4 Discussion [0.75 p]:** each claim once, with its reference and why it matters. Limitations in one paragraph: modelled spill-over form only; one platform; the false-positive floor moves the primary relocation read (0.65 → 0.50); the type-level response; the cycle proxy; calibration on these sections. Then the code link.

**Main-text floats (6):** Fig 1; the latent figure; Table headline (slim, one column); **Fig programme atlas**; Fig trade-off; the κ\* table. Optionally the subtype panel merged into the latent figure, and the relocation table as a small one-column table (6th).

## To the supplement
GO localisation; the tumour-axis test (it stays as a κ\* row only); the probe dumbbell; the battery figure and full tables; the context grading; the sensitivity rows; the full κ-sweep diagnostics and the κ\* figure; the trajectories; the held-out-tiles table; the timing table; the planted per-cell test; planted worlds; the ω = 0 ablation details; KL/uncertainty maps; Moran's I; marker pairs; derivations; rationale; implementation; data and labels; the full limitations.

## New work this plan needs (proposed; needs the author's approval)
1. **Planted-spill-over positive control** (simulation, cheap): plant a spatial contrast created *only* by spill-over at a known κ, run the sweep, and show that its κ\* lands near the planted κ, while a true response contrast survives. This is the strongest answer to "your contrasts survive by construction".
2. **A plain-regression reference for relocation:** regress each type's expression shift directly on niche composition from contaminated counts, on the same panels, to answer "why not just regress". Cheap, CPU.
3. **Verify the related-work comparison table** against each source paper (reading only).

## Writing rules
Experiments describe; the discussion claims and references. Zero to two numbers per claim. Nothing is said twice. Each strength is stated where it holds; each limit once. Same vocabulary throughout. Math minimal, with a notation table in the supplement.
