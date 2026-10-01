# Trim report, agent D (sweep, limitations, assembly), 2026-10-01

## Pages
- S8 file (now S7 "Spill-over Sweep and Sensitivity"): ~8 p before (with flushed floats) -> ~5.3 p (private build, pp. 28-33 of 37).
- S9 file (now S8 "Limitations"): ~1.5 p -> ~0.8 p.
- Final totals: see end of this file (after ./build.sh).

## Floats cut or merged
- tab:breakdown (generated breakdown.tex): \input removed (duplicate of main tab:kappa-star). The LABEL tab:breakdown is kept
  as a second label on the new contrast-definitions table tab:S-contrasts, so the generated caption of tab:kappa-star
  ("Full table: tab:breakdown") still resolves. Main loop: retarget that caption (to tab:S-contrasts or fig:breakdown-data),
  then drop the alias label.
- tab:kappa-sweep (generated kappa_sweep.tex): \input removed; same values as fig:kappa-sweep (figure kept, caption shortened).
  Its caption cited tab:headline; nothing in the supplement references tab:kappa-sweep any more.
- tab:breakdown-traj: generated longtable replaced by a compact hand table (means only, 5 readouts x 4 sections) with the
  label kept; a probe-residual row was added. Within-type share, effective rank, Moran's I of mu_w and programme overlap
  are summarised in text with their ranges (all from breakdown_traj.tex). Generator candidate: main loop may prefer to
  emit this table from scripts/paper_tables.py.
- fig:sensitivity: cut (same values as tab:sensitivity).
- fig:breakdown-data: kept, caption shortened, points to tab:S-contrasts; reference to S2's fig:breakdown removed.
- app:recon-modes + tab:recon-modes: stay in S7, text condensed to one paragraph.

## Added
- tab:S-contrasts: contrast definitions (name as in tab:kappa-star, definition, what positive means, main-text section),
  grouped biological/allocation. Sources: S5 readouts, scripts/logs/breakdown_2026-09-29/FAMILY.md.
- Probe across kappa: sentence + table row. Source: data/datasets/<4 sections>/experiments/probe_regrade_lineage_sweep.json,
  mlp_comp.excess as 1-exp(-2 excess); kappa=0.1 reproduces tab:probe. Seed means 4.0-4.7 % (kappa<=0.1), 2.9-4.5 % beyond,
  single seeds 2.0-6.1 %. The main-text sentence ("checked at each value") is therefore backed.
- app:S-planted-spillover: subsection with design and prediction from main 03_7, result \pending.
- Guide (unnumbered, before S1): claims C1-C7 (+C4b, model, data) -> main section -> supplement items.
- Assembly: \input order S1,S2,S3,S4,S5,S7(sim),S6,S8,S9; placeins[section]; float fractions; \FloatBarrier before the bibliography.

## Labels kept
app:sensitivity-section, app:sweep, app:sensitivity, app:recon-modes, app:limitations, fig:kappa-sweep, fig:breakdown-data,
tab:breakdown (alias), tab:breakdown-traj, tab:sensitivity (generated), tab:recon-modes. New: tab:S-contrasts, app:S-planted-spillover.

## Limitations (S8)
Every bullet kept, condensed; merged in: Prop. 1 caveat (bound not computed; grid supports kappa<=0.4), subclones,
development threshold not met (3 sections composition, TMA image), surrogate objective/non-unimodal seeds/dead-channel
re-seeding, one platform and sample size not analysed, edge features not tested.

## Unresolved / for the main loop
- breakdown_main caption retarget (above); \pending breakdown-gaps paragraph and planted spill-over control remain.

## Final assembly (./build.sh after A, B, C reported)
- supplement.pdf: 35 pages (59 before); main text: 10 pages (limit 10). 0 undefined references in either document; no duplicate labels.
- No float after the bibliography (References start p. 33). S7 = pp. 27-32, S8 = pp. 32-33.
- tab:sensitivity and tab:recon-modes moved to the head of their subsections so they no longer leave half-empty pages.
- Requests handled: fig:breakdown-data no longer cites fig:breakdown; contrast table uses "type-mean reference",
  "spill-over part"; axis test described as "not read on the TMA sections"; Guide names app:S-calibration.
- No other S-file needed a \ref retarget (S6 no longer cites tab:breakdown).
- Still open: 14 natbib "multiply defined" citation warnings (xr imports main's bibcites; pre-existing);
  agent A's S3 warm-up paragraph cites app:planted for the dead-channel discard (C removed that text; ref resolves, content stale);
  main 03_1:12 could point to app:S-calibration instead of app:sensitivity; plain-regression relocation \pending has no supplement home;
  over target: 35 p vs ~27 (S3 data ~6 p, S4 ~4 p incl. tab:headline awaiting retarget).
