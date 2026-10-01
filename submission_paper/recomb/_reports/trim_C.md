# Trim report, agent C (results and simulation), 2026-10-01

Files edited: supplement/S6_full_results.tex, supplement/S7_simulation.tex only. Private build: _build_C/.

## Pages
- Before (map, current supplement.pdf): S6 8.5 p + S7 5.5 p = 14 p. Words 3275 + 3273 = 6548.
- After (private build, tables still flushed because float placement is agent D's): text and figures ~5.7 p
  (S6 from p13 ~0.75 down to S8/S7-sweep start at p19 ~0.5) plus ~2.5 p of my six generated tables. About 8 p; ~7 p
  expected once tables are placed inline. Words 1671 + 1573 = 3244 (-50 %).
- Target 6.5 p **not fully reached**. The floor is ~2.4 p of generated tables (cellina_cf, transport_heldout, timing,
  synthetic, planted_percell; I may not edit them) plus three kept figures (~1.6 p: hallmarks, latents-umap,
  synthetic-misspec). Further text cuts would drop negative results listed in the map's must-keep list.

## Floats cut
- fig:transport-heldout (same data as tab:transport-heldout)
- tab:moran (final values in main fig:programmes d; the without-adversary values and reductions kept in text)
- tab:kl-summary, fig:kl-seeds (one paragraph keeps the negative result)
- fig:planted-percell (same values as tab:planted-percell)
## Floats kept
fig:hallmarks, tab:cellina-cf, tab:transport-heldout, fig:latents-umap (now 0.8 textwidth), tab:timing,
tab:synthetic, fig:synthetic-misspec, tab:planted-percell, tab:planted-gap.

## Text cut or condensed
- S6: Cellina comparison paragraph (repeats main 3.6; held-out-tile Cellina sentence kept); app:latents prose to 2
  sentences; app:kl-maps to one paragraph; timing big-O paragraph to one sentence; MintFlow train minutes (in tab:timing).
  Localisation, signalling, true/false axis, marker pairs condensed; all negatives kept.
- S7: synthetic set-up/readouts condensed (readout definitions defer to tab:synthetic caption); development
  "bistable" history and per-world cosine list cut; amortisation simulation bullet list to one paragraph;
  spill-over-subtracted encoder input cut; dead-channel paragraph cut (A keeps it in S3 warm-up).
- Section titles: S6 file "Results on Tissue in Full"; S7 file "Simulation" (subsection "The Amortisation Gap" keeps app:planted).

## Labels kept
app:full-results, app:response (niche-info + programme readouts, so both main 03_3:25 and S9/S8 pointers land),
app:moran, app:kl-maps, app:cellina-cf (now "Relocation in Full", holds the held-out-tile text too), app:latents,
app:timing, app:simulation, app:synthetic, app:planted-percell, app:planted. Removed labels: only the four cut floats.

## Retargets inside my files
tab:breakdown -> tab:kappa-star (main); tab:headline -> tab:headline-full; tab:breakdown-traj -> app:sweep (D may
merge that table); eq:S-penalty reference removed (development config described in words). Vocabulary: "Read A"
-> "single-cell read (Read A)" once.

## Cross-file requests
- Agent A (S3_implementation.tex, warm-up paragraph): it cites \cref{app:planted} for the dead-channel discard; that
  paragraph is gone from S7 (per map move 4). Drop the citation or point to the warm-up text itself.
- Nobody else cites the cut floats (checked sections/, supplement/, generated tables).
- app:recon-modes: not taken (no confirmation from D); stays in S8. My text cites tab:recon-modes and app:sensitivity.

## Unresolved
- Page target (see above).
- The planted spill-over control \pending lives in main 03_7 and S8 (D); the plain-regression \pending lives in main
  03_6 only. Neither was in my files, so nothing to keep; the plain-regression reference still has no supplement home
  (not assigned to any agent).
- Undefined references in the shared build come from other agents' in-progress files (app:amplification,
  app:gaussian-mi, app:S-calibration, eq:S-penalty, fig:breakdown); none from S6/S7.
