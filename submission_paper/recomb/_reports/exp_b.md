# exp_b report: sections 3.5 Comparison, 3.6 Relocation, 3.7 Robustness (2026-10-01)

## What I wrote
- `sections/03_5_comparison.tex`: fig:tradeoff (full width) plus four paragraphs. In order: why cycle state is the second axis (label-free, within-type; NMI rewards label fidelity); the C2 claim as CRITIQUE_publishability words it (linear probe lowest everywhere; under the nonlinear probe only MintFlow is lower, with cycle R² ≤ 0.06; Cellina at DISCELL's lowest seed on the TMA core); mirror R² lowest on 4 of 5 sections; resolVI as a single-latent reference; Cellina's best-case variants ("architectural"); reconstruction; speed (the substance of the approved AISTATS sentence); the cost of the 18-fit sweep against one MintFlow fit; a pointer to the supplement tables.
- `sections/03_6_relocation.tex`: a plain-language definition; 0.65–0.73; the programmes against the spill-over part; a hand table `tab:relocation` (one column, 3 reads × 4 sections, D/C); the comparison with Cellina worded as mixed; the held-out-tiles sentence; the plain-regression slot with both planned outcomes.
- `sections/03_7_robustness.tex`: a placeholder hand table `tab:kappa-star` split into biological claims and allocation checks; teeth (three breaks); the allocation-check caveat; one clause on the two contrasts moved to trajectories after reading; the planted spill-over control slot; the sensitivity to the spill-over form, with the false-positive floor.
  - 02_model §2.4 already gives the holding rule, the tipping-point framing, the 4× margin, the modelled-form limit and transferability, so 3.7 refers back to it and does not repeat them.
- New figure script `../aistats/figures/src/recomb_tradeoff.py` (added to build.sh). It runs fig_tradeoff.py with the x-axis worded "residual niche signal in z (%)" in place of "composition leakage", as STYLE requires. Output: `recomb_tradeoff.pdf`.

## Pages (private build `_build_exp_b`)
- 3.5: about 1.0 p against 1.1 (figure* about 0.55 p).
- 3.6: about 0.8 p against 0.8.
- 3.7: about 0.75 p against 1.0 (the pending text will shrink once the results land).

## Numbers and sources (tables in ../aistats/tables/generated/)
| Number | Source |
|---|---|
| ridge composition 1.0–1.1 % (ovarian) vs MintFlow 4.0 %; DISCELL ridge composition and image lowest on all 5 sections | probe.tex |
| MintFlow MLP lower everywhere, cycle R² ≤ 0.06 (0.062/0.006/0.054/0.056) | probe.tex, battery.tex |
| Cellina TMA core MLP 3.5 vs DISCELL 3.5–4.8; cycle 0.285 vs 0.497 | probe.tex, battery.tex |
| higher-cycle methods carry more residual (Cellina FFPE 0.566/0.311 at 13.9/7.9 %; SIMVI 0.567/0.631 at 4.9/6.8 %) | battery.tex, probe.tex |
| mirror lowest on 4/5 sections (serial: MintFlow 0.035 vs 0.039) | battery.tex |
| resolVI residual 2–5× DISCELL's (17.4/11.2/23.6 vs means about 4.6) | probe.tex, sensitivity.tex (MLP mean) |
| Cellina niche-domain: residual not consistently lower; NMI and cycle lower on every section; own graph 4.2/8.3 %, cycle 0.388/0.311 | probe.tex, battery.tex |
| reconstruction within 0.015 nats/count (gaps 0.011 FF, 0.005 core, 0.014 serial); best on ovarian and lung FFPE | battery.tex |
| 2.6× vs Cellina; >100× vs MintFlow | timing.tex (3.4/1.3; 938.6/9.3, 879.4/7.4, 544/1.3) |
| sweep ≤ 2.8 h (18 × 9.3 min); MintFlow 9–16 h (544–938.6 min); core sweep 23 min < SIMVI 192.6 min; FF sweep 9.4–13 h > resolVI 105 min | timing.tex (18 × the per-fit run time at the operating point; the fits at other κ were not timed, so this is an estimate) |
| relocation 0.65–0.73 | headline.tex |
| programmes 0.52–0.64, spill-over 0.21–0.39 | cellina_cf.tex |
| tab:relocation (24 values, three decimals, as in the source) | cellina_cf.tex |
| Read A intervals disjoint on 3 of 4 | cellina_cf.tex |
| held-out tiles: Cellina lung 0.703→0.486 (DISCELL 0.595–0.736); DISCELL moves both ways | transport_heldout.tex |
| κ\* entries 0.4ᵇ, 0.3ᵇ, 0.2ᵃ, >0.4 | breakdown.tex (08:10 render) |
| per-cell/per-gene rates ≤ 1.3 SD; floor 0.65→0.50 | sensitivity.tex |

## Open \pending
1. MintFlow refits:
   - fig:tradeoff caption;
   - the C2 paragraph (the MintFlow comparisons, including the ridge "next lowest 4.0 %" and the serial-section mirror);
   - reconstruction, speed and the sweep-vs-MintFlow cost.
2. Breakdown gaps (~21:00):
   - the table cells marked "?" (tumour axis on lung and FF; five serial-section findings);
   - recount the breaks in "three findings".
3. Plain-regression reference (3.6): keep one of the two planned sentences.
4. Planted spill-over control (3.7): result sentence.

## Requests
- **Generator change (to the main loop):** add a compact main-text variant of breakdown.tex, e.g. `breakdown_main.tex` with label `tab:kappa-star`, one column (`table`, not `table*`). Specification:
  - row groups "Biological claims" (relocation > spill-over part; relocation > type mean [Read A]; nearest-twin advantage; LR-gene lean of w; tumour axis in w) and "Allocation checks" (cycle in z, not in w; niche information in w);
  - cells ">0.4", or the bolded κ\* with the a/b footnote, or "n/a";
  - short plain-language row names as in my hand table; columns Ov. FFPE / Lung FFPE / Ov. FF / TMA core / TMA serial.
  - Then I replace the hand table with `\input{\tablesdir/breakdown_main}`. The supplement keeps breakdown.tex.
  - The biological/allocation assignment is my proposal; please confirm it. Read A and the twin margin are counted as biological.
- 03_3 writer: mirror R² is defined in one clause in 3.5. If 03_3 defines it first, I will turn mine into a back-reference.
- 03_2 writer: the planted spill-over control is placed in 3.7. 3.2 should only point to it.
- 04 Discussion: the false-positive floor (0.65 → 0.50) is stated in 3.7. The Discussion's limitation should refer back to it, not repeat the number.
- Supplement: `\suppref{app:baselines}`, `app:readouts`, `app:sweep` and `app:sensitivity` resolve. No change is needed.
