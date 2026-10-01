# Trim report, agent B (data and readouts), 2026-10-01

Files edited: `supplement/S4_data_and_labels.tex` (now S3), `supplement/S5_readouts.tex` (now S4). Nothing else was edited. Build: `latexmk -outdir=_build_data supplement.tex` gives 36 pages, with no undefined references and no duplicate labels. The multiply-defined natbib citations were already there; they come from xr.

## Pages
- Before: about 12.7 p for the two files, measured in an isolated harness that used D's float settings and included tab:probe and tab:headline. Without tab:headline it is about 11.7 p.
- After: about 9.0 p in the assembled build (p. 8 l. 36 to p. 17 l. 34). That includes tab:headline, which takes a full page (p. 15) and is still waiting on the main loop's retarget. **About 8.0 p once tab:headline is removed.**
- Target 6.75 p: **not met, by about 1.25 p.** The S3 floats alone fill about 4.3 p: tab:probe and tab:battery take a full page each, plus tab:sections, fig:sections, tab:lineage-ovarian and tab:baselines. That already exceeds the map's 4.0 p target for S3. tab:headline-full also takes a full page. These are generated, so they are not mine to shrink. The text is now about 3.3 p in total. Cutting further would remove the definitions that the contrasts depend on.

## Floats
- **Cut:** tab:lineage-clusters (two sentences keep the rule; the mappings ship with the code). tab:context (`\input` removed, together with its reference in the tab:baselines caption). tab:inputs (folded into one paragraph). The "Other assays (untested)" paragraphs, now one clause. The "What the response is and is not" paragraph.
- **Added:** `\input{\tablesdir/probe}`, placed next to battery in S3.5. Agent A has already removed it from S3_implementation, so it is now defined exactly once. New tab:S-subtype-ba (balanced accuracy, primary section; see below).
- **Shortened:** the SIMVI and MintFlow coverage cells in tab:baselines. These now point to the measured reasons in the tab:battery caption, which already gives them in full.
- **Kept unchanged:** tab:sections, fig:sections, tab:lineage-ovarian, tab:baselines, battery, headline (until the main loop removes it), headline_full.

## Labels kept
app:experimental, app:sections, app:lineage, app:baselines, app:assets, app:inputs, app:readouts-section, app:readouts, app:subtype, tab:sections, fig:sections, tab:lineage-ovarian, tab:baselines, and the generated tab:probe, tab:battery, tab:headline, tab:headline-full. One new label: tab:S-subtype-ba.

## Map items done
- **Reconstruction comparability:** new paragraph in S3.4. Every method's decoded mean is renormalised per cell and scored by the same multinomial per-count held-out log-likelihood, so dispersion and zero-inflation do not enter. The source is a % comment pointing at the battery and metric code.
- **Cellina on its own graph:** S3.4 now compares it with DISCELL, as the main text 03_5:18 does. Cycle R² is 0.39 and 0.31, against DISCELL's 0.48–0.51 and 0.49–0.50. The nonlinear residual is 4.2 % against 3.5–4.8 % on the core and 8.3 % against 4.0–4.8 % on the serial section. These numbers come from the generated battery and probe tables, cited in a % comment.
- **Vocabulary:** "single-cell read (internally Read A)" and "twin read (internally the twin margin)"; "transport" is removed from S4.
- **Readouts condensed:** about 25 % shorter. The required items survive: interval coverage nearer 80 % than 95 %, the tile-split ceiling sentence, the TMA trusted panel on 2 of 3 seeds, the twin margin partly following from construction, the false axis not orthogonal (≈0.41), and the cycle score as a proxy.
- **Estimators the contrasts are built from:** each is now defined explicitly in S4 so that D's contrast table can cite them.
  - Cycle R² of μ_z and of μ_w (cycle asymmetry).
  - I(niche; w).
  - Relocation over all panels: the full prediction minus the spill-over part, and minus the programme part as a trajectory, which is zero at κ = 0.
  - The single-cell read minus its **type-mean reference**.
  - The twin margin, defined precisely: (d_rand − d_twin)/d_rand per panel, median over panels.
  - The rank-biserial effect of the signalling share, with its half-tile interval.
  - The axis contrast: mean |τ| on the true axis minus mean |τ| on the false axis, over gene–type pairs.
  - The marker-pair contrast.
- **The axis test** is now stated as applying to the sections with tumour cells. FAMILY.md says it covers lung and FF from 2026-10-01 and that the TMA sections have no tumour lineage.

## Addition 2: the subtype metric gap, resolved
Balanced accuracy is available in `data/datasets/xenium_prime_ovarian_cancer_ffpe/experiments/subtype_recovery.md` (runs finalL_s0–s2, written 2026-09-28). The new tab:S-subtype-ba reports the ridge classifier's balanced accuracy by lineage for μ_z, μ_w, m_ψ and both latents, plus chance:

| Lineage | μ_w | μ_z | m_ψ | Chance |
|---|---|---|---|---|
| Fibroblasts | 0.95 | 0.66 | 0.94 | — |
| Endothelial | 0.90 | 0.67 | — | — |
| Tumour | 0.41 | 0.79 | — | 0.25 |

**Every main-text number in 03_3:28 and :31 matches this table, so the main text needs no change.** The text explains that balanced accuracy is the per-lineage score and AUC is the per-sublabel score on which the pre-set call is made. It also adds the perceptron's balanced accuracy on μ_z (0.71, 0.78 and 0.85), from the same file. All exceptions are kept: VEGFA⁺, the TMA myofibroblasts, the TMA endothelium, and the lung clusters.

## Requests to other owners
1. **Main loop:** after the generator retargets tab:headline, delete `\input{\tablesdir/headline}` and the sentence marked by the % comment above it in S5_readouts.tex.
2. **Agent D:** the contrasts paragraph in S8 still says the axis test runs "on the primary section". S5 now says it runs on the sections with tumour cells. Use the names "type-mean reference", "programme part" and "spill-over part" in the contrast table. S4 now cites `app:sweep` instead of tab:breakdown-traj, so the merge is safe.
3. **Generator:** the captions of cellina_cf, transport_heldout and battery still say "Read A" and "twin margin" (vocabulary item 9).

## Unresolved or flagged
- The app:assets licence sentence reads "BSD 3-Clause: scvi-tools, which contains resolVI, SIMVI, MintFlow and Cellina". tab:baselines lists SIMVI, MintFlow and Cellina as separate packages. I left the sentence unchanged because I could not verify each package's licence. The author should check it.
- `\pending{MintFlow refits with the corrected export}` remains in S3.4.
