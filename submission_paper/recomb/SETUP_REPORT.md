# RECOMB build setup report (2026-10-01)

## Result
`./build.sh` builds `main.pdf` and `supplement.pdf` with exit code 0, and so do `latexmk -pdf main.tex` and `latexmk -pdf supplement.tex` run on their own. The current output:
```
main.pdf: 2 pages in all (1 title page + main text + bibliography)
main text: 1 pages (limit 10)
supplement.pdf: 51 pages
main: 0 undefined reference warnings, 0 undefined citation warnings
supplement: 89 undefined reference warnings, 0 undefined citation warnings
```
I tested the page counter with 40 paragraphs of filler text. It reported 6 pages, and References started on page 6 of the numbered text, so the count is right.

## Page geometry (main.pdf)
| | |
|---|---|
| Paper | US Letter, 612 × 792 pt |
| Margins | 1 in on all sides (`geometry`) |
| Text width | **469.76 pt = 6.5 in** |
| Column width | **225.84 pt ≈ 3.125 in** (two columns, `\columnsep` = 0.25 in = 18.07 pt) |
| Text height | 9 in |
| Font | 10 pt, Computer Modern (the default of the `article` class, as in the AISTATS build) |

The figures were made for the AISTATS layout: a text width of 6.75 in and columns of about 3.25 in. Figures sized with `width=\columnwidth`/`\textwidth` will scale down by about 4%. The supplement is one column, 6.5 in wide.

## Files
- `main.tex`: preamble, title page, the `\input` of each section, then the bibliography.
  - Title page: title "DISCELL" (placeholder), authors "Author One, Author Two" with placeholder affiliations, and the abstract. The page is numbered `a`, so it doesn't clash with page 1 and isn't counted.
  - After the title page, `\clearpage`, two columns, and `\pagenumbering{arabic}` start the main text at page 1.
  - `\section{Experiments}` lives in `main.tex`, so each `03_*.tex` file starts with a `\subsection`.
  - `\markmainend` writes `\mainpages` to `main.aux` at the start of the bibliography, and `build.sh` reads it from there.
- `sections/00_abstract.tex` … `04_discussion.tex` (11 files). Each holds its heading and label, its STORY_MAP outline and page budget as a comment, and a `\todo{...}`.
- `supplement.tex` with `supplement/S1…S9`. Sections, figures, tables and equations are numbered S1, S2, … The supplement ends with its own bibliography.
- `macros.tex`: a copy of the AISTATS macros.
  - `\flag` and `\readorder` are no-ops, and the `soul` package was removed.
  - `\pending`, `\todo` and `\needsource` are kept.
  - `\codelink` was added; it prints "[URL to be added]".
- `STYLE.md`: the vocabulary, a notation table copied from the AISTATS method, the writing rules, and the label conventions. The conventions include the full list of labels the supplement reserves and of labels it expects from the main text.
- `build.sh` runs two rounds of latexmk; the second round uses `-g` so that each document picks up the other's `.aux`. It then prints:
  - the page counts;
  - the undefined references;
  - any label defined in both documents.

  The latexmk output goes to `build_main.out` and `build_supplement.out`.

## Shared resources
- **Bibliography:** a relative path, `\bibliography{../aistats/references}`, not a copy. The style is natbib with apalike (author-year), as in the AISTATS version.
- **Figures:** `\graphicspath{{../aistats/figures/}}`. In the copied supplement, the `figures/` prefix was stripped from every `\includegraphics`.
- **Tables:** `\newcommand{\tablesdir}{../aistats/tables/generated}`. The supplement uses `\input{\tablesdir/...}`.
- `../aistats` was not modified.

## Cross-references (xr-hyper)
Each document imports the other's labels **without a prefix**, with links into the other PDF. A prefix isn't possible because the shared generated tables refer across documents: `headline.tex` (main) refers to `tab:headline-full` (supplement), and `battery.tex` (supplement) refers to `tab:headline` (main).

So the two label namespaces must stay disjoint. `build.sh` checks this, and `STYLE.md` lists the reserved labels. New supplement labels take an `S-` marker (`fig:S-…`).

In the main text, write `\suppref{app:x}`, which prints "Supplement S3.2", or `\cref{fig:x}`, which prints "Figure S8". I tested both in a scratch copy, together with `\cref{tab:probe}`, which printed "Table S6".

## Supplement provenance (copied; flags and reading-order markers removed from the copies)
| File | Source |
|---|---|
| S1_derivations | appendix/derivations.tex |
| S2_design_rationale | appendix/rationale.tex |
| S3_implementation | appendix/implementation.tex |
| S4_data_and_labels | experimental.tex: Sections, Lineage Labels, Baselines (with the battery and context tables), Assets and Licences, What the Model Needs |
| S5_readouts | experimental.tex: Readouts, Merged Subtypes |
| S6_full_results | experimental.tex: Response in Detail, Spatial Autocorrelation, Counterfactual vs Cellina, Latents, Per-Cell Divergence, Training Cost |
| S7_simulation | synthetic.tex (whole) + experimental.tex: Planted Worlds |
| S8_sensitivity_and_ablations | experimental.tex: Leakage Sweep in Full, Sensitivity to the Assumptions, What Each Part Buys in Reconstruction |
| S9_limitations | appendix/limitations.tex |

The section headings were rewritten in title case. The original labels were kept, and the new section labels are `app:readouts-section`, `app:full-results`, `app:simulation` and `app:sensitivity-section`.

`data.tex`, `calibration.tex`, `validation.tex` and `additional_results.tex` were also excluded from the AISTATS build: they are a stale single-section draft with open `\todo` items. They are copied to `supplement/deferred/` but not `\input`. Their labels `app:sections` and `app:image` would clash with S4 and S2.

The copied text still uses the AISTATS vocabulary ("leakage", "leak fraction", "transport"). This needs one rename pass when the supplement is edited.

## Undefined references (expected at this stage; all in the supplement, pointing at main-text labels not yet written)
`def:breakdown eq:context-gat eq:gen-p eq:gen-rho eq:gen-w eq:penalty eq:probe fig:battery fig:breakdown-data fig:overview fig:probe-data prop:kappa-bound prop:symmetries sec:batching sec:generative sec:inference sec:invariance sec:objective sec:results-sweep sec:results-w sec:results-z sec:selection sec:setting sec:sweep tab:headline`

`STYLE.md` proposes a home for each label, or says to retarget the reference. Some will not come back in the RECOMB main text and must be retargeted in the supplement:
- `fig:battery`, `fig:probe-data` and `fig:breakdown-data`;
- probably `prop:symmetries`, `eq:penalty`, `sec:batching` and `sec:selection`.

`sec:method`, `sec:intro` and `sec:setup` already resolve; `02_model` keeps the label `sec:method` for that reason.

## Notes for the next step
- **Page budgets:** the per-file budgets add up to 9.95 pages.
  - Intro 1.3, model 2.25 and discussion 0.6.
  - The experiments come to **5.8**: 0.4 + 0.5 + 1.2 + 0.8 + 1.1 + 0.8 + 1.0. STORY_MAP states 5.6, which is inconsistent with its own sub-budgets.
- The supplement has 9 small overfull hboxes, at most 17 pt, in tables copied from the 6.75 in AISTATS layout.
- `fig:latents-main` is still in S6. If the latent figure moves into §3.3, take it out of S6.
- Similarly, any generated table that the main text `\input`s, for example `breakdown` for the κ\* table, must leave the supplement.
- Nothing was committed. The build products (`*.aux`, `*.log`, `*.pdf`, `build_*.out`) are untracked in `recomb/`.
