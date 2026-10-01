# Report: intro (sections/01_introduction.tex)

## What I wrote
- Opening: the three sources of a cell's counts (intrinsic state, response, spill-over), then one paragraph on why response and spill-over are confounded at first order.
- Related work (`\label{sec:related}`), positioned per RELATED_VERIFIED.md:
  - classical models and decontamination in one sentence;
  - resolVI: neighbour mixture plus background, weights estimated per cell, no niche term; why its low-dimensionality assumption does not separate a response from spill-over; our term is its mixture with fixed weights;
  - NCEM, including its CVAE variant (T1); SIMVI's regression prior on the neighbours' latents, credited as the closest analogue of the w prior (T3), with the penalty on z only (T10);
  - MintFlow's type-conditional critics, credited (T11); Cellina's marginal adversary and swapped-neighbourhood prediction; Celcomen and NicheCompass in one clause;
  - none models spill-over; SIMVI and MintFlow do state identifiability results, so the gap is stated narrowly: no method states what the counts cannot identify when a response and spill-over compete.
- Comparison table `tab:related` (single column, 5 rows). Columns: niche term, spill-over, invariance, intervention. Every cell is H-verified in RELATED_VERIFIED.md. The counterfactual column is narrowed to "expression generated under an explicit change of the neighbourhood". SIMVI gets × with a footnote that it estimates a spatial treatment effect. The intrinsic-latent column and the Celcomen and NCEM rows are dropped to save space.
- "Our approach" paragraph: model both; κ bounded only from above, so swept; breakdown point; tipping-point and partial-identification framing (rosenbaum2002, manski2003). An explicit honesty sentence: neither ingredient is new alone; the combination creates the confound, and the sweep handles it.
- Four contribution bullets as briefed.

## Pages
Private build (_build_intro): about 1.33 pages (page 1 in full, plus about two-thirds of the first column of page 2), against a budget of 1.3.

## Numbers
- "about a tenth of its counts from neighbours": cited from yang2025mistic, as in the AISTATS setup.
- No other numbers appear outside \pending.

## Open \pending
- Contribution 3: "it breaks three findings, while the others hold to κ = 0.4, four times the typical misassignment". Waiting on the breakdown-gaps queue (C1 freeze rule).

## Bib
- Appended `ergen2026resolvi` (Nature Methods 2026, doi 10.1038/s41592-026-03212-9). Title, authors, journal and date were verified via Crossref; no volume or pages yet. The intro cites it.
- `ergen2025` (bioRxiv) is unchanged. Other writers may want to switch to the new key.
- My resolVI description follows the bioRxiv Methods, because the published Methods are paywalled and were not read.

## Requests / cross-file
- References that other files must define: sec:generative, sec:invariance, sec:sweep, prop:kappa-bound, def:breakdown, sec:experiments. These currently resolve with 02_model.
- `tab:related` and `sec:related` are now defined in main. Neither is in the supplement's reserved list.
