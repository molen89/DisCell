# TODO — improvements to hammer before the sweeps (opened 2026-09-17)

Rule of the list: one item, one owner turn, one verifiable "done" line. Pick from
the top of a block; blocks are ordered by what gates what. Status: `open`,
`in progress`, `done <date>`, `parked`. Decisions already taken are in §0 and are
not re-opened here.

## 0. Decisions taken (2026-09-17, author)

- Four datasets in scope; GSE315411 sweeps on the `pdl018d` core, the two full
  slides are the held-out-section test.
- Compute is not a constraint; every long job runs detached.
- Fresh-frozen slide: its own budget, 500 epochs / patience 40, declared as a
  per-dataset protocol note; `--figures-every 100`.
- Shared centre point (κ 0.1 = d_w 6 = α_w 0.1) fitted once per dataset, reused
  in all three tables.
- α_w grid {0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3}; d_w grid {2, 3, 6, 8} (no 12).
- Cross-slide leg on GSE: all swept core checkpoints evaluated on the dual section.
- LR co-occurrence ladder keeps all three rungs.
- Leak meter (doc 16) rejected as a model input; module kept as a diagnostic.
- Article carries no empirical numbers until evaluation starts.

## 1. Battery and evaluation hygiene (gates every run that follows)

| # | item | why | done when | status |
|---|---|---|---|---|
| 1.1 | §7.10 degeneracy pair + type-mean-z recon gap in `Trainer.evaluate` | z-is-not-just-t must be read on every fit | in metrics.json best/final, history, TB; post-hoc CLI; tests | done 2026-09-17 |
| 1.2 | Run the post-hoc degeneracy CLI on the 18 sweep3 runs | grid comparable before new grids | `runs/sweep3_*/degeneracy.json` present; one line in the devlog | done 2026-09-18 (queue) |
| 1.3 | Diagnose the NMI drift (`reference_best` 0.630 post hoc vs 0.658 stored) | old and new NMI must be comparable before they share a table | reload is bit-for-bit reproducible on the same weights → the drift was a code change in the old run's era; no pre-settle run will be quoted | done 2026-09-17 |
| 1.4 | Per-type read-time centring of w (issue V12) in report/atlas/A2/A5 readers | per-type ‖w‖ is the gauge offset; withdrawn as a read | `Δ_i = B·[m_ψ(c_i,t) − m_ψ(c̄_t,t)]` used everywhere a raw w norm was; report section reworded | done 2026-09-21 in the atlas and the report (gauge = within-type mean of μ_w, registered); A2/A5 application readers untouched (parked) |
| 1.5 | Atlas activity by effective rank (V10), stability by shift-space overlap (V11) | "6 programmes" is a metric artefact | atlas.py + tests; report section rewritten How / Evaluated by / Wished for; rerun on the seed triple | done 2026-09-21 (r = 2 ovarian, EMT/HYPOXIA recur 3/3; cluster slides run; rank-based enrichment; V10–V12 closed). κ-survival moved to shift space 2026-09-21 (EMT/HYPOXIA/G2M recur at every κ ≤ 0.3) |
| 1.6 | Skip the ring-2 encoder pass under `type_only` and re-measure the halo overhead | spec §4.5 says ring 2 is a lookup; the code still encodes it | `posterior_z` runs on seeds ∪ ring1 only when sources are types; overhead re-measured at 4,096-cell tiles | done 2026-09-21: nodes +14.3 % → +6.7 %, memory −6.6 %, wall clock unchanged; loss identical to 1e-7 |
| 1.7 | NMI guard anchored to the first evaluation (FF run blocked two recon-improving checkpoints) | the guard becomes a fixed floor when NMI drifts down | architect ruling recorded; either keep, or anchor to a running window | open (architect) |
| 1.8 | Landmark inventory on graphclust slides | §2 landmarks and the §5 matrix cannot run on `Cluster-N` labels | decision: name clusters (GSE pseudobulk path) or name-free landmarks | open (decision) |

## 2. Model questions to settle (before they are swept)

| # | item | why | done when | status |
|---|---|---|---|---|
| 2.1 | x̃ (leak-subtracted encoder input): powered planted world (fix V9) comparing raw / z-probe / x̃-z-probe / leak-subtracted counts | per-cell decontamination claims hang on it; 3-seed slide result was near-neutral | pre-registered pass/fail per seed; verdict use / option-only / drop | **closed 2026-09-21: option-only** — powered world 3/3; z beats raw 3/3 (0.27/0.37/0.38 → 0.11/0.18/0.12); x̃-z beats z 1/3; counts-correction at its oracle ceiling; `subtract_leak` stays default off. **Follow-up 2026-09-21 (12 seeds + recall arm): every clause fails, x̃ reverses on 3/6 unsaturated seeds and costs up to 22 pts recall — option-only is final; no κ rerun** |
| 2.2 | α_w = 0.05 under `type_only`, 3 seeds at 500/40, battery selection | second context axis becomes seed-stable at 0.05; operating point may move | acceptance as pre-registered in devlog 2026-09-14 (axis-2 cosine ≥ 0.75, cycle_z ≥ 0.44, recon in envelope, w rows intact) | **rejected 2026-09-21**: w channel opens as predicted (KL_w ×5–10, axis-2 share 0.25–0.39, niche AUC w +0.01–0.05) but NMI 0.627 / 0.611 on two seeds vs the 0.63 floor; α_w stays 0.1 |
| 2.3 | The w-side adds nothing over a type lookup once z is type-averaged (battery finding) | first reconstruction-side number for the α_w pinning | a short experiment: recon of type-mean-z + w vs type profile across α_w runs on disk; one devlog entry | done 2026-09-17: context-varying w doubles 0.1→0.05 (0.012→0.017 nats/count, type_z era; type_only at 0.1 reads 0.009) but total recon is flat; (d)−(e) = 0; read 2.2 on the w side |
| 2.4 | Leak modelling beyond a global κ (see §5) | type- or cell-specific leak scale from nuclear/extranuclear geometry | decisive tests 5.4–5.7 all failed (per-edge slope 0.2–0.3 vs [0.5, 2]; κ_i level independent of the nucleus); no model change; global swept κ stands with an order-of-magnitude bracket 0.09–0.25 behind it | **closed, negative** 2026-09-17 |
| 2.5 | Transport counterfactual: κ-sensitivity rerun on the corrected object; Φ-fixed counterfactual row; annotation niches for the supported tier; report clarity | current κ-sensitivity is on the pre-correction object; Φ carries half of w's context dependence | `transport_kappa_sensitivity_v2.json`; Φ-fixed row; plain-language report section | **done 2026-09-21**: all bars met; noise ceiling + trusted tier (ovarian 0.099 all / 0.205 trusted, slope 1.08); annotation niches 71 supported panels; interventionable share ≈ 1; κ-range [0.05, 0.2]; FF runs (0.280, 90 % beats-both) |

## 3. Validation instruments

| # | item | why | done when | status |
|---|---|---|---|---|
| 3.1 | Cycle asymmetry | good as is | — | done |
| 3.2 | LR ladder, three rungs | good as is; per-run artefact | — | done |
| 3.3 | Depth qualifier on "z beats the linear reference 2×" (M6 addendum) | ratio is 0.90 on the deep slide | wording in report/handover; issue closed | done 2026-09-21 |
| 3.4 | FF slide battery (report, validate §§3–4, atlas, transport) | the fit exists, the battery does not | four artefacts under `ovary_ff/runs/reference_graphclust/`; devlog results | validate + report done 2026-09-17; atlas fixed 2026-09-21 (runs on cluster labels); transport OOM fixed 2026-09-21 (group-mean accumulation) — FF battery complete |

## 3b. Cell-cycle state labels (added 2026-09-17)

| # | item | why | done when | status |
|---|---|---|---|---|
| 3.5 | Better cell-cycle state labels from DNA content (`scripts/test_cell_cycle.py`) | independent cycle target | analysed 2026-09-17: nucleus overlap is 0.18 % of nuclei (not the mechanism); integrated DAPI ≈ nuclear area (slope 1.04, R² 0.73) with 2.65× tile drift and a segmentation-route swing 0.17→0.86; no variant clears AUROC ≥ 0.7 / MKI67 ratio ≥ 2. Fix list in `experiments/cell_cycle_dapi_analysis.json`. Only remaining candidate: the FF slide through six gates (flat-field, background, exclusions, truncation model, bimodality test, bar) | **closed negative on both slides** (FF six gates all fail 2026-09-17: R² on area 0.85, dip 0/38, AUROC 0.54). Cycle claim rests on marker scores with **per-depth-stratum** reliability stated (FF 2026-09-21: reliability is one curve in depth across slides; a faint per-cell DNA signal exists at > 2,000 counts in 0.5 % of cells, too small for a label). Note: `diptest` added to the venv, not to pyproject |

| 3.6 | Depth-neutral continuous cycle target (decided 2026-09-21: stay with the Scanpy S/G2M scores, continuous, no hard phase call, no DAPI) | the score is mildly anti-correlated with total counts within type (−0.1 to −0.36 on ovarian), an artefact of the scoring normalisation; the reliability ceiling is computed but not reported next to the R² | scoring with expression-matched control genes on library-normalised counts; within-type Spearman(score, log counts) near 0 on all types; split-half reliability reported beside every cycle R²; before/after of cycle_z / cycle_w / floor / depth baseline / linear ref on the pinned reference; old label-based correlation map marked superseded | built 2026-09-21: depth neutrality passes (worst tilt 0.016 / 0.011); reliability falls to 0.08–0.16 (most of the old agreement was depth); cycle_z −0.08 / −0.16, mostly from the rank transform's tie-breaking. **Decided 2026-09-21: keep the plain Scanpy score as the target** (no transform; it is a stated imperfect reference, defended by floor / depth baseline / reliability); the depth-stratified rank version stays as a robustness column, and the honest ceiling (~0.16, not 0.52) is quoted as a caveat |

## 4. Baselines

| # | item | why | done when | status |
|---|---|---|---|---|
| 4.1 | Survey SIMVI, resolVI, MintFlow (+ scVIVA, NicheCompass briefly): inputs, outputs, scale, install, which of our metrics apply on which dataset | needed before any baseline run | comparison matrix method × dataset × metric; install plan; GPU-hour estimates | done 2026-09-21 (devlog): resolVI in-env, SIMVI py3.10 venv + tutorial gate, MintFlow own venv; order resolVI → SIMVI → MintFlow → scVIVA/NicheCompass, ≈ 50–65 GPU-h |
| 4.2 | Run the feasible baselines on the ovarian tile split with our instruments | the missing external comparison | per-baseline rows in the allegiance battery | stage 0 done 2026-09-21 (all three installed, exported, smoke-fitted; resolVI 0.18 contamination on the 5k window); stage 1 = resolVI full fits, then SIMVI at core scale, then MintFlow — awaiting go |

## 5. Leak scale beyond a global κ (design track)

Failed: doc-16 leak meter (type-pooled, sender-specific genes; see devlog
2026-09-17). Candidates that use the nuclear/extranuclear split without needing
sender-specific genes, in order of simplicity:

| # | idea | what it yields | first check |
|---|---|---|---|
| 5.1 | Rim-share slope per receiver type: regress a cell's extranuclear share on its foreign-neighbour exposure (face-weighted density of other-type neighbours), controlling for area | a type-specific receiver tendency a_r, gene-free | sign and size per type on ovarian; replication across the two GSE sections |
| 5.2 | Transcript-geometry κ_i: share of a cell's assigned transcripts that lie closer to a neighbour's nucleus (or within a boundary band on the shared Voronoi face) than to its own nucleus | a cell-specific upper bound κ_i and a per-edge β_ij from the same transcripts | correlates with 5.1's type slopes; its m-sweep {0, 0.5, 1, 1.5, 2} replaces the κ grid |
| 5.3 | Nuclear-share drop of foreign markers within type: for receiver type r, nuclear share of gene g vs foreign exposure; the drop measures the leaked share of g without needing g to be specific against all types | validation of 5.1/5.2, per gene | sign agreement with 5.2 on the top foreign markers |

Built 2026-09-17 as item 5.2 (`discell/experiments/transcript_flux.py`): β^T and κ_i from the signed nucleus-bisector offset of extranuclear transcripts. Status per check: replication ✓, β^T≠β_face ✓, share ✓ (median 0.13, no tail), sensitivity ✓, gene content ✗ against the pre-registered bar but the bar was a 50 % bar; against the simulated null p ≈ 0.11 [0.09, 0.13] on all three slides. Next, in order:

| # | decisive test | done when |
|---|---|---|
| 5.4 ✗ | W-tf1: check (1) with power at 13 % by design — type-specific genes (3× ratio), scored against the simulated p = 0 null | zero-crossing of the bar ≤ 0.15; observed p reported with 2-SE on all three slides |
| 5.5 ✗ | W-tf2: off-centring control — recompute `s` with nuclei shuffled within a tile, or against the own polygon; subtract | κ_i level after the control; ordering preserved or not |
| 5.6 ✗ | W-tf7: per-edge agreement of geometric κ_i and content-implied p | Spearman over edges and a calibration slope on all three slides |
| 5.7 ✗ | W-tf9: host arm on the extranuclear profile (fixes the ×2.3 moment disagreement); W-tf8: power curve for the DiD or drop it | both moments agree within 2-SE |
| 5.8 — | NOT built (5.4–5.7 failed 2026-09-17). Would reopen only on a per-edge relation with slope in [0.5, 2] on two slides from a likelihood-based content statistic, plus an explanation of W-tf10. If so: model change — vector κ_i capped at 0.5 in `leakage_mix`, β^T + κ_i on `ModelGraph`, `TrainConfig.m` swept {0, 0.5, 1, 1.5, 2}; spec §2/§4.7/§7.7 patch via the architect | tests + one ovarian fit at m = 1 inside the current κ envelope |

## 6. Sweep programme (launch checklist; do not start before §1 is done)

- [ ] 1.1–1.5 done; degeneracy metrics land in every new fit.
- [x] Ovarian α_w leg — done 2026-09-18.
- [x] GSE core: κ, d_w, α_w — done 2026-09-21 (self-contained grids).
- [x] Lung — done 2026-09-18.
- [x] Fresh-frozen — done 2026-09-18 (500/40).
- [x] GSE cross-slide — done 2026-09-21: constant cost (0.015 nats/count, 0.02–0.03 NMI) at every grid point; cycle_z held-out ≥ same-section.
- [x] Reports regenerated 2026-09-21 with the pooled cycle key; per-value B stability still unread.

## 6a. Transport, distribution level (added 2026-09-21)

| # | item | why | done when | status |
|---|---|---|---|---|
| 6a.1 | Distribution-level transport read (MMD on the Hellinger map; pairwise A→B and leave-one-niche-out; four references; count-matched companion) | the author's question: compare the transported population with the one that was there | built and run on 3 ovarian seeds × 2 niche sources + FF | done 2026-09-21: median gap closed 0.45 pairwise / 0.30 pooled count-matched; pooling tighter CI in a majority; **type-mean predictor closes 0.9–1.0** → at Xenium depth the read sees location, not spread |
| 6a.4 | Model-vs-model MMD (target = decoded target cells, not raw counts) + HVG-1000 companion (author, 2026-09-21) | remove reconstruction error and shot noise from the distance | run on pinned reference (both niche sources) + FF; type-mean bar re-read | done 2026-09-21: gap closed 0.76–0.87, type-mean collapses to ≤0; **but target decoded at group-mean w, not own posterior → rerun with own μ_w (6a.6)** |
| 6a.5 | Matched-twin read: nearest-z source cell transported, compared cell to cell; references untransported twin / random same-type / z-NN floor (author, 2026-09-21) | does z carry per-cell information across niches, or was the population read all there was | gap closed and twin margin per panel on the same runs | done 2026-09-21: matched ≫ random 501/501, transported > untransported everywhere; same group-w caveat → 6a.6 |
| 6a.6 | Reads A and B with the target side at each cell's own posterior μ_w / real context | remove the shared-w circularity | numbers beside 6a.4/6a.5 on both `best` runs | open |
| 6a.2 | Fraction-of-ceiling headline + top-50 gene overlap in the mean read | readability without R² | in table and report | done 2026-09-21 (trusted tier 0.34 of ceiling; 15.3 / 50 genes vs 0.7 chance) |

| 6a.3 | L2 on w so B carries the programme (author, 2026-09-21) | push magnitude into B; make B columns comparable across seeds | offset ratio down, atlas axis-2 cosine up, guards inside envelope | **closed 2026-09-21, drop as default**: gauge fixed (offset 2.40 → 1.72, B grows) but atlas reads are gauge-invariant by construction so cosines cannot move; working channel shrinks 1.5× (a) / 6.7× (b, rejected). `--w-penalty w --lambda-w 6.5e-4` kept as option |

## 6b. From the article review (2026-09-21), instruments we already have

| # | item | why | done when | status |
|---|---|---|---|---|
| 6b.1 | MintFlow's signalling-gene criterion on our leak channel: LR-gene vs non-LR attributed share for w, for κℓρ̄, and raw, over the κ grid, composition-residualised | if the leak channel alone reproduces their validation signal, their real-data evidence is confounded — the strongest related-work sentence available | table over κ with doc-09's exposure null as control; ovarian + FF | **done 2026-09-21: leak channel separates LR genes at effect +0.23–0.27 at every κ > 0, response +0.01–0.06 (ns at κ ≥ 0.2); FF reproduces** |
| 6b.2 | resolVI's double-positive metric: mutually exclusive marker pairs (from a matched scRNA reference) with true-positive control pairs, scored on raw / x̃ / counts-corrected across κ | external criterion for the x̃ ceiling and the "leak-induced false positives are rare" finding | pairs curated; thresholds on training tiles; scores on held-out; both panels side by side | open (0 GPU-h + curation) |
| 6b.3 | DisCoVR's MIG / MIC with y = niche (K = 10 and tumour bands), I(y;z), I(y;w), I(w;z\|y) by kNN and MINE, with the within-type permutation floor | the quadrant table in the metric an ML venue expects; a number for the flagged z-niche cell | on the seed triple, lung, FF | done 2026-09-21: raw MIC 0.44–0.60, floor-corrected 0.79–0.87; z's niche excess over floor 0.07–0.13 vs w's 0.38–0.51 — the † cell is type identity |
| 6b.4 | SIMVI's true-axis / false-axis test: per-gene Kendall τ of the w-predicted shift vs the ordered band index and vs the orthogonal in-plane coordinate | answers "your niche reads are type colocalisation through contiguity" | TP/FP gene counts with z, ℓ and linear-reference rows | done 2026-09-21: w mean |τ| 0.85–0.93 true vs 0.52–0.60 false; 7–67× at |τ| ≥ 0.9 on three seeds |
| 6b.5 | DisCoVR-style objective ablations: drop the second KL copy; drop the intrinsic path (b); class-mean prior μ_t in place of m_ψ(c,t); adversary on x̂ instead of z | the ablation table a DisCoVR reviewer asks for; spec §6.3 anticipates the first | quadrant + guards per ablation, 3 seeds | **done 2026-09-21**: objective survives; (i) ≡ α_z/2 (cycle_z up → α_z ladder follow-up); (ii) path (b) load-bearing for cycle_z; (iii) class-mean prior kills transport (beats-both 0/155), cycle_w 0.10–0.15; (iv) adversary on x̂ breaks probe/mirror guards. Flags kept default-off |
| 6b.7 | Three-way per-cell contamination figure: our κ grid, resolVI's per-cell mixture proportion, MintFlow's microenvironment score on the same cells vs the transcript-flux share (review A1) | the central claim in one panel | after the resolVI and MintFlow fits | open (after 4.2) |
| 6b.8 | Analytic-posterior planted world for the amortisation gap (review B2): additive two-latent count world with a known leak operator where E[z\|x] is closed-form | measures the gap against truth rather than by inference | synthetic, ~4 GPU-h | open |
| 6b.9 | MintFlow's in-silico perturbation scored with our noise ceiling and gap-closed statistic (review B3) | the transport row where we have a reliability floor and they do not | after the MintFlow fit | open (after 4.2) |
| 6b.10 | GO-localisation enrichment of B (secreted / membrane / cytoplasmic, from Celcomen) and its κ-sensitivity (review B4) | prior-free label for "is this programme about the outside of the cell"; the confound visible inside the atlas | ovarian seed triple, 0 GPU-h | open |
| 6b.6 | Bib: fix `akbarnejad2025mintflow` (five shared first authors) and `tiesmeyer2026ovrlpy`; check `ergen2025resolvi` author list | flagged CHECK notes | entries complete | open (text) |

## 7. Paper (later)

- Apply the model-text changes of `submission_paper/aistats/revision_2026-09-17.md` §A: type_only sources **done 2026-09-17**; budget/selection protocol, gauge, "not built" rows still pending.
- Numbers return only after the sweeps.
