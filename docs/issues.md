# Bugs and issues found, with their resolutions

Running register, newest last within each section. Each entry: what failed, how
it was caught, what fixed it (or why it stands). The devlog carries the
narrative; this is the lookup table.

## Pipeline

| # | issue | caught by | resolution |
|---|---|---|---|
| P1 | `.gitignore` pattern `data/` matched `discell/data/` at any depth: five package modules invisible to git, two never committed — HEAD could not import from a fresh clone since the initial commit | repo audit (`git check-ignore -v`) | pattern anchored to `/data/`; modules committed |
| P2 | `beta` all-zero for 94.5% of connected cells: the loader fed `smoothing_weights` the selected graph's `shared_wall_um`, and the default graph was `contact`, where that metric is 0 on 98.3% of edges | repo audit; measured on the full slide | `DEFAULT_GRAPH = "voronoi"` (positive face on every edge by construction); zero rows now exactly the isolated cells, asserted by `test_beta_is_row_stochastic_over_every_in_edge` |
| P3 | 19 label classes where 18 exist: loader filled NaN with `"unassigned"` while 10x ships `"Unassigned"` — two classes, one meaning, at opposite ends of the sorted label space | repo audit | fill folds onto the panel's own spelling case-insensitively; `test_no_two_labels_differ_only_by_case` |
| P4 | `--min-apposed-um` on a bundle lacking the metric compared against a silent zero-fill and dropped **every** edge (INFO line only), then crashed in `smoothing_weights` with a misleading sparse-matrix `TypeError` | live reproduction | the load-time filter removed entirely (edges are never trimmed; consumers filter), the zero-fill fallback with it |
| P5 | `embedding_dim` default 256 in the loader vs 384 in both CLIs: the zeros fallback changed width by entry point | repo audit | one `DEFAULT_EMBEDDING_DIM = 384` in `data/embeddings.py` |
| P6 | `uv run --group dev pytest` re-syncs the venv and **uninstalls the extras** (removed `huggingface_hub` mid-session) | live failure during the mask experiment | README commands use `--all-extras --group dev`; warning added |
| P7 | KRONOS2's vendor remote code calls `snapshot_download` without `cache_dir`: the 440 MB checkpoint lands in `~/.cache/huggingface` regardless of `--cache-dir` | dependency audit | upstream; documented in README — not fixable here |
| P8 | `find_tissue_image` crashed on the `str` that `uns["xenium_dir"]` returns | live failure | `Path` coercion at entry |
| P9 | `neighbour_composition` was the **same tensor object** in every batch under residency (in-place mutation would corrupt the dataset) but a fresh copy per batch otherwise | repo audit | constants moved off the batch entirely (`dataset.constant`); batches carry `edge_id`/`into_j` instead |

## Ego-masking experiment

| # | issue | caught by | resolution |
|---|---|---|---|
| E1 | 15 µm mask radius (spec'd from typical cell sizes) leaves 2,740 cells poking out of the hole — a partially masked cell leaks exactly the silhouette the mask exists to remove | measured covering radius of all 407k cells (p100 = 36.8 µm; the tail is real smooth-muscle spindles) | 25 µm disk (99.999% coverage) and the 6 non-fitting cells excluded from **every** arm; `test_default_radius_covers_all_but_a_handful` |
| E2 | "resample to 0.5 µm/px = the FM's native scale" — wrong provenance: 0.5 is H&E-FM lore; KRONOS's own card runs at 0.37 µm/px, reference patch 256 px | model-card check | ran at 0.5/256 px (conservative: bigger field, smaller mask fraction); `--target-mpp` makes a 0.37 arm one flag |

## Model

| # | issue | caught by | resolution |
|---|---|---|---|
| M1 | Isolated cells were charged a constant `−log(1−κ)` for leakage they cannot receive: their `ρ̄` row is zero so the mixture summed to `1−κ` | ELBO assembly review | `leakage_mix` renormalises per row — isolated cells degenerate to `ρ` (κᵢ = 0); `test_leakage_mix_gives_isolated_cells_kappa_zero` |
| M2 | z posterior collapse at the "obvious" weights: 1/ℓ recon scaling makes α=1 price a latent's information ~ℓ-fold above the unscaled ELBO (KL_z → 0, NMI 0.21) | synthetic recovery gate | ELBO-equivalent α ≈ 1/ℓ̄; **now in the spec** (§5 as patched, 2026-09-01) |
| M3 | α_w knife-edge: at ~1/ℓ̄ the direct x-path lets w steal identity (NMI 0.12); at ≥0.1 w collapses onto its prior (KL_w = 0.000) — spec §7.4's three-route table, observed live | synthetic recovery gate | run at the pinned end for the gate; §4.6 probe-based calibration scheduled before the κ sweep |
| M4 | Invariance-penalty NaN, two causes: `y` sums to 1 so the v-block covariance is **singular by construction**, and within-niche y-variance ≈ 0 makes the logdet gradient (Σ⁻¹) explode | synthetic recovery gate (fit went NaN) | one y column dropped from `v`; penalty computed on the **correlation form** — MI is scale-invariant so the value is provably identical (analytic-MI test unchanged) and conditioning is bounded |
| M5 | `rho_bar` needed to be data, not a gradient path (resolVI's mode-collapse note, amplification 1/(1−κ)) | designed in, then asserted | carries no autograd graph at all — `backward()` on it raises; `test_foreign_influx_is_fully_frozen` |
| M6 | `cycle_r2`'s `r2_mean_types` is noisy (−0.08…+0.13 over identical configs): the MKI67⁺ ranking admits essentially non-cycling types (Pericytes, Ciliated Epithelial) whose within-type score variance ≈ 0, so their held-out R² is noise (to −1.2) and dominates the equal-weighted mean; the truly cycling types are stable (0.44–0.51 every run). Also the "ceiling" is not one — z beats the 50-PC ridge ~2×, it is a linear reference, not a bound | sweep2 cross-run comparison (18 runs, 2026-09-08) | **headline read = `r2_pooled`** (cell-weighted, per-type-centred: tight at 0.42–0.46); `by_type` kept for the split. Open option if mean-of-types must be quoted: gate types on within-type score variance |

## Training infrastructure (adversarial verification hunt, 2026-08-20)

Found by a six-agent hunt over `discell/model/` plus a derivation check; the
independent-refutation stage was cut short by a session limit, so each fix
below was confirmed by direct inspection instead.

| # | issue | resolution |
|---|---|---|
| T1 | the invariance-penalty gradient was silently scaled by `cov_ema` — the only live autograd path was the `ema * batch` term, so `alpha_a`'s effective strength depended on an unrelated smoothing constant (0.3 acted like ~0.015) | straight-through estimator: value from the EMA-stabilised moments, gradient from the batch at full scale; `alpha_a` default rescaled 0.3 → 0.02 to preserve behaviour; `test_penalty_gradient_scale_is_independent_of_ema` |
| T2 | `assemble()` drew the PCA subsample and the train/val split from one RNG stream — toggling `phi_pca` would silently change the split, making sweep runs incomparable | independent per-purpose streams (`default_rng([seed, k])`); the split is now invariant to every other knob |
| T3 | `figures()` drew its scatter subsample from the training-shuffle RNG — a logging knob (`figures_every`) changed the fit | figures own `default_rng([seed, 3])` |
| T4 | early-stop patience `patience // eval_every` floors to 0 when patience < eval_every → instant stop | `max(1, ...)` |
| T5 | `figures_every` not a multiple of `eval_every` silently never fired | explicit `ValueError` at construction |
| T6 | `best.pt` omitted the covariance-tracker state its docstring promised | state saved alongside the weights |
| T7 | the probe subsampled the FIRST 30k rows — tile order, i.e. spatially biased | random subsample |
| T8 | evaluation sweeps sampled z and w — held-out reconstruction (the early-stop signal) rode reparameterisation noise | `forward(sample=False)`: posterior means throughout evaluation, deterministic sweeps |
| T9 | `vbar_t` averaged isolated cells' all-zero `y` rows while `ybar_t` excluded them | connected cells only, matching |

## Adversary (escalation, 2026-09-01)

| # | issue | caught by | resolution |
|---|---|---|---|
| A1 | the closed-form penalty cannot remove nonlinear dependence by construction: converged, ridge-ΔCE 0.026 (linear leak gone) but MLP-ΔCE 0.153 (46% of uncontrolled) | convergence check with the MLP probe | §4.6 escalation: adversary built (soft `eΦ`, two nonlinear heads, separate optimiser, halves logged) |
| A2 | a **weak adversary is a placebo**: at `adv_steps=2`, lr 1e-3, the encoder defeats the stale heads while a fresh probe still finds 58–99% of the leak | adversarial calibration round 3 | `adv_steps=6`, lr 2e-3, α_a=0.3 → MLP-ΔCE 14% of uncontrolled at **zero NMI cost** (0.641 vs 0.623 uncontrolled); α_a=1.0 reaches the floor. Head strength must be verified by the *independent* probe, never by the training-time head loss |

## Doc-08 / 09 / 11 instruments (2026-09-08 → 2026-09-11)

| # | issue | caught by | resolution |
|---|---|---|---|
| V1 | landmark banded R² all negative under the pooled ≤ 500 µm fit (more signal = more negative) | first §2 run | fit **within each band** (block-CV inside the band); registered in spec_deviations |
| V2 | landmark gene signatures dominated by control probes | map inspection | prevalence ≥ 1% mask on signature genes |
| V3 | tumour/stroma interface covered 58% of the slide under the graph mode-filter | landmark map (user asked to see the landmarks) | kNN-smoothed tumour fraction (k = 50) → 8.9% |
| V4 | raw z-pseudotime niche R² 0.55 read as "z is spatial" | homophily reasoning | **type-partialled** niche R² (z 0.01–0.03, w 0.55) is the reported read; raw kept alongside |
| V5 | allegiance matrix per-row normalised: a −0.01-vs-0.02 row rendered fully hot, the §4.3 niche-z flag invisible | user-caught | absolute strength scale, "n.s." grey below max(floor, ℓ)+margin, dagger for unexpected-column signal |
| V6 | transport "full" prediction let the type's intrinsic mix vary across niches (selection) — not doc-08 §7.2's counterfactual | user-caught ("old or new neighbours?") | counterfactual = program + leak with z fixed is the headline (0.099 R², 100/155); model account kept; the gap (≈ 0.05) reported as the selection share |
| V7 | hallmark labels shipped ungated (SPERMATOGENESIS p = 0.08 on a tumour program) | architect flag (suspected genome background — verified absent) | BH across sets per program, label only at q ≤ 0.05 |
| V8 | A4 v1 exposure-slope leg used a within-type exposure shuffle as the null — a spatial-structure null that cannot separate contamination from real niche clustering | A4 v1 read | v2: gene-split fingerprint (contamination-specific by physics: Δ = corr(own_A, nbr_A) − corr(own_A, nbr_B), ring 1 vs ring 2) |
| V10 | atlas activity judged on varimax-rotated coordinate variance: a rank-2 w rotated onto 6 axes gives six collinear coordinates that all pass the 1% threshold → "6/6 active, 0 spare" on the pinned reference | eigen-spectrum of cov(μ_w) = [0.83, 0.17, 0, 0, 0, 0] (2026-09-14) | **OPEN** — activity by effective rank r of cov(μ_w), varimax within the r-dim subspace; honest read = 2 programs (s1), ~1 (s0, s2) |
| V11 | `matched_correlation` B-stability reads the four null w-directions (no variance → arbitrary B columns) as instability (0.43–0.60) | shift-space overlap across seeds 0.97–0.99, per-cell shift corr 0.90–0.93 | **OPEN** — replace/augment with shift-space overlap (span of μ_w·B); the κ = 0 → κ > 0 drop still unexplained |
| V9 | A4 v2 planted-world read-out: one global threshold across synthetic types (control FPR 0.02–0.45 by seed) and an under-powered plant (raw AUROC 0.67, leak invisible to raw in 2/3 seeds) | JSON inspection after the run | **OPEN** — A4 parked; fix = within-type thresholds + excess FPR (victim − control) + a power gate (raw AUROC ≥ 0.9, raw excess ≥ 0.1) before any z verdict |
| V12 | w carries a per-type constant offset (‖mean_t w‖ 4–10× the within-type spread on the pinned run; ‖B·mean_t w‖ 18–28) — an unidentified gauge: `(a(z) − B·μ_t, w + μ_t)` is the same model, Adam has no decay, `m_ψ` takes `t`. The report's per-type ‖w‖ ranking is the offset (Spearman −0.41 vs the centred ranking) and doc-11 A2/A5's `softmax(a(z))` drops it along with the niche effect | scratch offset read on `ablation_gat_type_only_s1`; weight-decay pair `gat_type_only_wd{0,1e-4}_s1` (2026-09-16): L2 on parameters moves the gauge *into* w (global offset 23, ‖B·mean_t w‖ 81–92) | **OPEN** — read-time: centre w per type at a reference context, `Δ_i = B·[m_ψ(c_i,t) − m_ψ(c̄_t,t)]`; clean profile `softmax(a(z) + B·m_ψ(c̄_t,t))`; ‖w‖ ranking withdrawn; training-time fix would need a penalty on `E[m_ψ]` per type, not weight decay |

## Open / watch list (updated after architect review, 2026-09-01)

- ~~α_w calibration~~ **done 2026-09-01**: scanned {0.03, 0.06, 0.1, 0.2} on
  the slide; knife-edge reproduced (0.03 → NMI 0.33 identity theft; 0.2 → w
  dead). α_w = 0.1 chosen; consequence to keep in mind: w near-pinned
  (KL ≈ 0.002/dim), anomaly scores conservative. The MLP probe cross-check
  found the ridge under-reports ~2.5–3× — **the escalation rule is judged on
  MLP numbers** from here on.
- **Partial isolation is unhandled by design** — renormalisation fixes degree 0
  only; a degree-1–2 cell gets full κ against a 1–2-neighbour ρ̄. Ruling:
  acceptable, but **the sweep report must stratify effects by the edges-lost
  QC column** already stored on the graph.
- **Φ's value is untested where it matters** — the mask experiment measured
  *type* information only; Φ sitting below the homophily baseline is the
  intended exogeneity, not uselessness. The decisive test is Δ held-out
  reconstruction (and m_ψ fit) **with vs without Φ**; that one ablation also
  settles the deferred PCA decision.
- **Sweep must run ≥3 seeds per κ** — B bistability (§7.7 observed) makes the
  effect envelope over κ × seed jointly the deliverable; stable-across-κ but
  flapping-across-seeds is not a finding.
- **τ is not a sweep axis** — measured inert (face-dominated β); recorded so
  nobody resurrects it. κ is the axis.
- ~~Mirror R² conflates type separability~~ **resolved**: within-type form
  implemented per patched §7.10 (`test_mirror_r2_is_blind_to_pure_type_separation`).
- **β at τ=20 is face-dominated** (effective neighbours 4.68 vs 4.90 with no
  decay): fine for now, but it means τ is nearly inert — κ-sweep conclusions
  should not be attributed to the decay length.
- **Ring overhead** measured 21% (1.6k-seed tiles) / 9.7% (6.4k) vs the 13%/7%
  back-of-envelope: real but higher; use ≥4k-cell tiles.
- **B recovery is seed-bistable at planted κ > 0** (principal cosine 0.15–0.83
  across seeds at one config) and stable in a no-leak world (0.65/0.83): the
  spec's §7.7 confound made visible. The gate asserts B only at κ=0; under κ
  the sweep judges it. Real-data B claims need multi-seed checks.
- **`leakage_mix` EPS clamp** zeroes the corrective gradient for a gene with
  observed counts but p < 1e-8; a log-sum-exp mixture would keep it. Revisit if
  starved-gene pathologies appear.
- **`effective_count`** is a heuristic (mean batch count / ema), not a strict
  effective sample; the support floor inherits that softness.
- ~~**Mirror R²** should get a within-type variant~~ (duplicate of the
  resolved item above — within-type form implemented per patched §7.10).
- **Niche-invariance residual in z (doc-08, 2026-09-08)**: block-CV logistic
  `mu_z -> niche` reads macro-AUC 0.65 vs ℓ-baseline 0.59 and w 0.76 — z is
  not fully at max(floor, ℓ) as doc-08 §4.3 expects. The Moran §3.2 triage
  says the hot z dims are NOT y-explainable (≤3.6% R²), so the current
  reading is benign intrinsic-spatial structure (clone patches/programme
  territories) leaking into *discrete* niche labels through spatial
  contiguity, not context leakage the §4.6 probe missed. Watch: if a future
  operating point pushes it toward w's level, run the triage per dim and
  regress the niche logits on y before concluding either way.
- **Low-α_w bistability (α_w study, 2026-09-10; REVISED 2026-09-11)**: for
  α_w < 0.1 training is seed-bistable; the high-recon basin (−7.17..−7.23 vs
  −7.25..−7.26) was first attributed to m_ψ reading neighbour μ_z ("w-mirror")
  — **that mechanism is falsified** (two certified detectors found no
  separation; the basin's likelihood survives replacing w with a linear
  function of composition, and its w-channel value is composition-
  recoverable). What stands: the bistability itself, and the basin =
  a stronger composition-level w-pathway (co-adapted m_ψ/B) unreachable at
  0.1. Open: better solution or confound absorption (0.03_s2 drained
  cycle_z to 0.354; 0.03_s1 battery unexamined). α_w stays 0.1 until the
  basin is characterised; any run beating the 0.1 recon envelope by ≫ seed
  spread still warrants the channel decomposition (w=0/z=0 + ŵ(y)
  substitution, `experiments/w_mirror_certification.json` methodology).
  **SETTLED 2026-09-11**: `ablation_gat_type_only_s1` reached −7.1924 with
  neighbour z structurally absent and the best battery on record (cycle_z
  0.499, Moran z 0.061 / w 0.628, niche w 0.767 / z 0.654); `alphaw_0.03_s1`
  passes clean too. The high-recon family is a **better optimum, not a
  pathology**; the one bad instance (0.03_s2, cycle_z 0.354) says
  membership does not guarantee quality → **select by the battery, never
  by recon alone**. α_w stays 0.1 (the family exists there); the residual
  open item is reachability (1–2 of 3 seeds at 500 epochs; never at 200).
- **Amortised per-cell z cannot see the cell's own neighbours (2026-09-11,
  UNVERIFIED hypothesis, raised by A4 v2)**: `enc_z` takes
  `[counts_i, log ℓ_i, onehot t_i]` only, by design. The true posterior
  p(z_i | x_i, ρ̄_i) depends on the neighbours' shed profiles; the amortised
  q(z_i | x_i, t_i) cannot, so the leak is removed on the decoder side
  (B, the decoder, the population law of z) while the **per-cell** μ_z
  inherits whatever leaked into x_i. If confirmed, per-cell decontamination
  claims through μ_z (doc-11 A1/A2/A4/A5) need either the counts-level
  correction x̃_i = x_i − κ ℓ_i ρ̄_i (the model's own ρ̄, no encoder
  involved) or semi-amortised refinement of z_i. Adjudicate with a
  *powered* planted world comparing raw / z-probe / leak-subtracted counts
  (issues V9 fixes first).
- **Doc-09 reattribution figure (61% leak-attributed) is uncalibrated**
  until the §5a planted worlds run; do not quote it.
- **A4 real-data outcome is the pre-registered honest one**: gene-split
  Δ ring 1 = 0.006 [−0.006, 0.020] in 342k post-mitotic cells — no
  transcript-transfer fingerprint; "leak-induced cycle false positives are
  rare at κ = 0.1 on this slide". The flatter z exposure gradient (0.07 vs
  raw 0.16) is therefore closer to doc-11's *sensitivity-loss* fail state
  than to a decontamination win; DAPI supports neither caller
  (both-positive median −0.17 vs neither).
- **KL_w oscillates 20–100× between evaluations 5 epochs apart** (all
  seeds; s0's `best.pt` sits on a 0.12-nat spike, s1/s2 on the 0.005
  floor). A checkpoint's KL_w is a snapshot of the prior chasing the
  posterior, not a solution property; any α_w / channel claim must be read
  over the trajectory, not at one checkpoint (2026-09-14).
- **w occupies 1–2 of its 6 dims at α_w = 0.1** (cov eigen-fractions
  0.83/0.17 at s1, 0.98/0.02 at s0/s2; rank 1 even at d_w = 2). **It is an
  α_w effect in the prior field itself**: r = 3 at 0.05, 4 at ≤ 0.03,
  carried by m_ψ and the realised shift, not by the deviation channel
  (2026-09-14). d_w = 6 stays; the operating point is the open question.
  **Extended 2026-09-15**: the effective rank is a (d_w, α_w, budget)
  property of the optimiser, not of the tissue — at the 200-epoch budget
  d_w = 6 is rank ~1 (second axis 0.1–4%), d_w = 3 rank 2 (second axis
  3–27%, seed-variable), d_w = 2 rank 1, d_w = 8 rank 2 (9–14%, seed-
  stable, but cycle_w rises to 0.008–0.020 — watch); at α_w = 0.05 a second axis reproduces across
  seeds (cosine 0.77–0.90), at 0.10 it does not (0.50–0.77). Any "N
  programs" statement must name d_w, α_w and the budget. Candidate
  re-calibration α_w = 0.05 under type_only awaits the architect.
- **Encoder input x̃ = x − κℓρ̄** (two-pass, seeds only) proposed as the
  structural fix for the amortisation gap — design note in the devlog
  2026-09-14; spec change pending the architect.

- M6 addendum (2026-09-17): the remark "z beats the 50-PC ridge ~2×, so it is a linear reference, not a bound" is depth-dependent. On `xenium_prime_human_ovary_ff` (1,401 transcripts/cell median, Tirosh split-half S 0.508 / G2M 0.696) the 50-PC frame reaches 0.852 and z 0.767 → ratio 0.90; pdl018d 0.96; ovarian FFPE 2.1; lung 2.1. The 2× is the linear frame failing on shallow, noise-dominated scores, not a model property. Quote the absolute read (z ≫ permuted / ℓ-baseline, w ≈ 0) across slides; never the ratio. Per-cluster z on FF is uniform (0.66–0.78), so the noisy `r2_mean_types` problem of M6 does not appear at this depth.
- Watch (T-table candidate, 2026-09-17): TensorBoard figure events dominate wall time on a 1.16M-cell slide — 16 events took 136 of 183 min on `ovary_ff/runs/reference_graphclust` (235–335 s each to epoch 254, 540–1,111 s after; +9 UMAP fits per event since 2026-09-14, per-type panels on 8k members × 8 types), while training + evaluation was ~50 min at 7.6 s/epoch. Not a bug; a cost that scales with the slide. For fits of this size pass `--figures-every 100` (must stay a multiple of `eval_every`, T5) — a 500-epoch FF fit would then take ~1.2 h instead of ~3.
- Watch (2026-09-17): the spec §7.10 joint early-stop criterion (`nmi_guard` = 0.9 × running-max NMI) engaged on `ovary_ff/runs/reference_graphclust`: NMI drifted 0.643 (epoch 4) → 0.58–0.60 while recon kept improving, and the guard (threshold 0.579) blocked recon-improving evaluations at epochs 219 (−7.3150, NMI 0.579) and 234 (−7.3144, 0.579); `best.pt` (369, NMI 0.595) is the best *guarded* checkpoint. Two things to keep in view: the guard is anchored to the first evaluation, when z is closest to t, so on any slide where NMI drifts down during the fit it becomes a fixed floor at 0.9 × the epoch-4 value; and a blocked improvement increments `stale`, so the guard also shortens the effective patience. Whether it engaged on any other run was not checked.
- Watch (2026-09-17): `validate.landmark_inventory` selects classes by type-name substrings ("Endothelial", "Pericyte", "Smooth Muscle", "Tumor Cells"/"Malignant"), so on graphclust-labelled slides (`Cluster-N`) the §2 landmark bands and hence the §5 matrix cannot run — lung and ovarian `reference_graphclust` validations stopped at §§3–4 for this reason (their `validation.json` carry only `morans`, `niche`), and FF will too. Unlocking §2/§5 on annotation-free slides needs cluster naming (the GSE315411 pseudobulk path) or a name-free landmark definition; a decision, not a fix.
- Unverified (2026-09-17): the FF training footprint (predicted 18–20 GB with int16 resident counts) is not logged anywhere; the int16 assert held. Sample `nvidia-smi` during the first battery job on `reference_graphclust` (which rebuilds the same resident Trainer via `load_run`) to close it.
- **V-item (watch), 2026-09-17** — `reference_best` (trained 8 September, type_z era) reloaded post-hoc through `discell.model.degeneracy` reproduces its stored `metrics.json` best `recon_val` exactly (−7.2586) but returns z-type NMI 0.630 against the 0.658 on record. The other three runs reproduce both numbers exactly. Since recon is bit-identical the checkpoint reload is right; the likely cause is drift in the NMI evaluation path (subsample ordering or k-means call) between 8 and 11 September. Not chased down. Consequence if real: NMI values quoted from runs older than ~10 September are not comparable with ones computed today, which matters for the NMI floor when old and new runs are tabled together.
- **Closed, 2026-09-17** — the two §7.10 diagnostics marked pending (`I(z;t)/H(t), var(z|t)` and `Δ held-out recon, z vs one-hot t`) are built. They are recorded by `Trainer.evaluate` in `metrics.json` (best and final), `history.jsonl` and three TensorBoard scalars (`val/degeneracy_mi_ratio`, `val/degeneracy_within_var_fraction`, `val/recon_gap_typemean_z`), with a post-hoc CLI for existing runs. The second is built in a substituted form; see the spec-deviation register. Planted-answer coverage in `tests/test_model_metrics.py`; wiring coverage in `tests/test_model_train.py`.
- **Reading note, 2026-09-17** — `within_var_fraction` is easy to read backwards. 0 = `z` is a function of `t` (degenerate); 1 = the type means coincide, i.e. `z` is blind to type (the failure the NMI floor guards). A high `mi_ratio` alone is *not* evidence of degeneracy — the decoder receives no `t`, so `z` must carry type. Degenerate is the conjunction: ratio near 1 **and** within-fraction near 0 **and** a null recon gap. The dev-slide values are 0.80–0.82 / 0.68–0.70 / 0.12–0.15.
- W-item (watch, run hygiene): the ovarian `alphaw_0.02`, `alphaw_0.03{,_s1,_s2}`, `alphaw_0.05{,_s1,_s2}`, `alphaw_0.07` runs are type_z-era (their `config.json` has no `gat_sources` key at all, i.e. they predate the 2026-09-12 type_only ratification) AND were fitted at 500 epochs / patience 40 rather than the 200/20 sweep budget. They are a different architecture at a different budget and must never be pooled into a type_only alpha_w table; `cal2_aw_*` are closer still (closed-form invariance, 40 epochs, alpha_a 0.03). The alpha_w grid in docs/sweep_programme.md re-fits the sub-0.1 seeds for this reason.
- W-item (watch, cross-dataset comparability): the fresh-frozen slide is the only dataset whose reference fit did not early-stop inside 200 epochs - `xenium_prime_human_ovary_ff/runs/reference_graphclust` stopped at epoch 409 (best 369, 182.7 min) where ovarian, lung and GSE stop at 94-104. Any FF sweep run at the 200-epoch budget is a truncated fit, and FF sweep numbers are then comparable to each other but not to FF's own reference nor to the other slides' converged optima. Whichever budget is chosen, the choice belongs in the methods section.
- W-item (watch, instrument): `discell/model/sweep.py` reads the spec-7.10 degeneracy pair and the type-mean recon gap from `metrics.json` at `final.degeneracy` / `final.recon_gap`, falling back to the top-level keys of the same name, and reports `null` when neither is present (which is every run fitted to date). If the metrics land under a third spelling the sweep report will silently show nulls rather than fail; check the first new run's report before trusting a column of nulls.
- V-item: the leak meter (`discell/experiments/leak_meter.py`) is not a usable instrument for per-cell leak. Its estimate is unstable to a nuisance choice -- re-randomising the learn/fit tile split over seeds 0-4 moves pooled kappa on the lineage-merged ovarian slide over 0.143, 0.490, 0.092, 2.901, 0.060 -- and it does not replicate across two sections of one TMA (receiver tendencies correlate at Spearman -0.085 over 29 shared types; pooled kappa -1.98 vs +4.00). Roughly 30% of the fitted `c_rt` entries are negative, which no leak model can produce. Treat any number it prints as a diagnostic about the slide, never as a model input. Artefacts: data/datasets/xenium_prime_ovarian_cancer_ffpe/experiments/leak_meter_full*.json, data/datasets/gse315411_pdltma06_1{0,1}_prime_*/experiments/leak_meter_pdl018d_curated35.json.
- Watch: the concept document's leaked-profile term is mis-specified. Leaked copies come from the rim, so a sender's leaked profile should go as `(1-eta_g) rho_t(g)`, not its total profile `rho_t(g)`; the error is aligned with the `eta_g` direction that identifies `zeta`. The module exposes `--sender-profile extranuclear` as the corrected form. Anyone reusing the document's algebra elsewhere should use the corrected form.
- Watch: `eta_g(t)`, the sender's retention, is measured on type-`t` cells that themselves receive leak at nuclear share `zeta`, so it is biased toward `zeta`. That shrinks the `(zeta - eta_g)` contrast the fit lives on and inflates `c_rt`, more so in dense regions. Any future retention-based leak estimate needs a leak-corrected `eta`, which is circular unless solved jointly.
- Watch: the bundle's `full.h5ad` stores the obs index as an anndata nullable-string-array group (`values` + `mask`), not a plain string dataset. `_full_cell_ids` in `discell/experiments/leak_meter.py` now handles both. Other code that reads obs_names through raw h5py rather than anndata may hit the same shape; not surveyed, since it is outside this work package's files.

- **Transcript-flux instrument (2026-09-17)** — `discell/experiments/transcript_flux.py`: per-edge β^T and per-cell κ_i from extranuclear transcript geometry. Passes replication (GSE sections Spearman 0.843), share checks and sensitivity; the gene-content check failed its pre-registered bar, which the power analysis showed was a 50 %-admixture bar (zero-crossing p ≈ 0.50); against the simulated p = 0 null the content implies p ≈ 0.11 (bracket 0.11–0.26 across two moments). **Open before any model adoption**: (W-tf1) rebuild check (1) with power at 13 % by design — type-specific genes (`leak_meter.specific_genes` at 3×), scored against the simulated null, never against zero; (W-tf2) nucleus off-centring control (shuffled nuclei within a tile, or own polygon) to price the level of κ_i; (W-tf7) per-edge agreement between geometric κ_i and content-implied p — the cleanest decisive test; (W-tf8) the band-minus-core DiD has no power curve, so its null is uninterpretable — build one or drop it; (W-tf9) the two moments disagree ×2.3 → score the host arm on the cell's extranuclear profile, not whole-cell. Minor: band width sets κ_i's level ×2.3 (W-tf3); `hard` weight is band-invariant so the grid has 7 distinct rows (W-tf4); 5,720 ovarian cells lack a nucleus polygon, 899 keep a zero β^T row (W-tf5); the `pdl018d` cores read and discard ~95 % of each section's transcripts (W-tf6). If adopted: `leakage_mix` takes a per-cell κ vector capped at 0.5, `prepare.build_graph` carries β^T and κ_i from the artefact, `TrainConfig.kappa` → `m` swept over {0, 0.5, 1, 1.5, 2}.

- **Transcript-flux decisive tests (2026-09-17), closing W-tf1/2/7/8/9** — W-tf1: the content check cannot carry a zero-crossing bar (pinned at p = 0.5 by the mixture symmetry, any gene subset); report implied p and σ-separation instead (specific genes: 9–13 σ, p 0.10–0.16). W-tf2: the tile shuffle is a destroy-everything null (κ 0.39–0.48); the polygon-centroid arm shows off-centring contributes only 0.006–0.011 to κ_i, and an offset permutation reproduces κ_i within 3 % — κ_i's level does not come from the nucleus. W-tf7: **failed** — per-edge slope 0.20–0.29 against [0.5, 2], Spearman sign unstable; this blocks adoption. W-tf8: DiD power curve built (2–3.8 σ at p = 0.13), keep the statistic. W-tf9: extranuclear host arm does not reconcile the moments (ratio ~1.8–2.5). **New W-tf10**: observed band-minus-core DiD significantly negative on all three slides, 3–6 σ below the planted leak-free world — unexplained; blocks any DiD-based influx claim. **New W-tf11**: per-edge content excess is count-dependent and the banded count correlates with flux share; any per-edge inversion must be count-matched (single-curve inversion gives Spearman −0.89). Artefacts `experiments/transcript_flux_decisive_*.json`.

- **T-item (2026-09-21)** — `metrics.type_degeneracy` indexed `probe.classes_` with the argmax over the k-wide log-probability matrix; when a type is absent from the training rows (GSE core: Plasma cells n = 1) the index overflows. Killed all six GSE sweep legs of the 2026-09-17 queue at their first fit. Fixed: argmax is the type index; regression test `test_type_degeneracy_survives_a_type_absent_from_training`.
- **T-item (2026-09-21)** — `sweep.metric_row` read `cycle_r2_z/w` from `r2_mean_types` (the M6-retired statistic), so every aggregate sweep JSON showed cycle_z ≈ 0. Fixed to `r2_pooled`; `cycle_r2_z_mean_types` kept as an extra column; reports regenerated by `scripts/queue_2026-09-21_gse.sh`.
- **Watch (2026-09-21)** — FF `reference_graphclust`: `atlas` fails (landmark inventory empty on `Cluster-N` labels → DBSCAN on zero points; the atlas's context-driver step should skip landmarks when the inventory is empty, todo 1.8) and `transport` was killed with exit 137 on the 1.16M-cell slide (memory; not diagnosed).
- **V10 CLOSED 2026-09-21** — activity is the effective rank r of cov(μ_w) on within-type-centred w (≥ 1 % of trace), varimax within the r-subspace; certified by a planted rank-2-in-6 world. Reads: ovarian r = 2 (all seeds), lung 1, FF 3, α_w = 0.05 r = 3–4.
- **V11 CLOSED 2026-09-21** — matched-column B correlation dropped; stability = per-axis cross-seed |cos| on gauge-centred loadings + shift-space overlap of the r-dim span, certified on a permuted/flipped basis (1.0) and an orthogonal complement (0.0). Ovarian triple: axis-1 0.93–0.98, axis-2 0.40–0.83, overlap 0.67–0.93 (lower than the 0.97–0.99 of 2026-09-14, which was on raw w and dominated by the per-type offset). The κ = 0 → κ > 0 drop is still unexplained. κ-survival (`validate --sweep-tag`) still uses matched signatures — to be moved to shift space.
- **V12 CLOSED at read time 2026-09-21** — every atlas read is on Δ_i = B·[w_i − w̄_t]; a planted 10× per-type offset changes neither rank, shares, loadings nor labels; the report's per-type ‖w‖ ranking is withdrawn in place. Training-time identification unchanged and still open.
- **V9 CLOSED 2026-09-21** — fixed and re-run as the x̃ gate (`discell/applications/xtilde_gate.py`): within-type thresholds at each type's control quantile (control FPR 0.298–0.301 everywhere, was 0.02–0.45), excess FPR = victim − control, paired stratified bootstrap, model-free power gate before any fit. The under-powered plant (+1.0 log-fold on random genes) is replaced by a 25 % transcript-share plant on 12 mid-expressed genes (raw AUROC 0.9997–1.0, raw excess 0.27–0.38). A4's planted leg now delegates to the gate; regression `test_within_type_thresholds_undo_the_v9_defect`.
- **Amortisation gap, adjudicated 2026-09-21** (replaces the watch item): in the powered world the z-probe carries only 31–48 % of the raw score's excess FPR (below the 0.5 line on 3/3 seeds) and beats raw by 0.16–0.26 at unchanged AUROC — the per-cell posterior mean does not simply inherit the leak. The counts-level correction x̃ = x − κℓρ̄ removes ~a quarter and sits on its oracle ceiling (model ρ̄ = true ρ̄ to three decimals). Encoder x̃ (`subtract_leak`) is **option-only** (right sign 3/3, clears the margin 1/3); default off. Still open: the gate's AUROC saturates and it has no genuinely-cycling victim, so the sensitivity-loss fail state is untestable by it.
- **Closed 2026-09-21** — transport's exit-137 on the 1.16M-cell FF slide: the instrument held per-cell rate matrices (n_cells × n_genes float32 ≈ 23 GB); it now accumulates per-(niche, type) group means in the forward pass. FF transport runs in ~3 min (248 panels, R² 0.280).
- **Watch (2026-09-21)** — transport per-panel noise ceiling is 0.12 on ovarian: 141/155 composition panels are unmeasurable; all-panel means must always be quoted beside the trusted-tier means. GSE core (ceiling 0.049, 0 trusted) does not reach the floor for this instrument.
- **Watch (2026-09-21, from the x̃ follow-up)** — at an unsaturated planted programme (raw AUROC ≈ 0.92) the linear z-probe caller loses 0.25–0.38 of raw's recall on genuinely planted cycling cells and its AUROC within the cycling types falls to 0.61–0.92: per-cell z blunts a real programme while removing leak-induced false positives. Any per-cell decontamination claim through μ_z (A1/A2/A5) must report recall on true positives beside FPR on victims. Independent of x̃, which is closed option-only (12-seed rule failed on every clause).
- **V11 companion closed 2026-09-21** — `validate --sweep-tag` no longer reports matched signatures; κ-survival is shift-space overlap + per-axis |cos| at the effective rank + hallmark-label recurrence via the atlas' own functions. Old numbers kept under `legacy` for one release; they do not reproduce from `best.pt` and their producer is not in the repo.
- **V12 training-time clause closed as not-worth-closing, 2026-09-21** — L2 on w (λ 6.5e-4) fixes the gauge and passes every guard on three seeds but does not improve cross-seed axis-2 cosine and shrinks the realised within-type shift 1.5×; the per-type-mean penalty collapses the model (rank 1, dominant programme lost, cycle_z 0.392, within-type shift 6.17 → 0.92). Structural reason: every atlas read is on Δ_i = B(w_i − w̄_t), invariant to both gauges. Read-time centring is the fix; `--w-penalty w` is an option, default off. Open: pin which runs back the record's α_w = 0.1 axis-cosine triple (0.64 / 0.83 / 0.40 does not reproduce exactly).
