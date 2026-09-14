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
- **w occupies 1–2 of its 6 dims** (cov eigen-fractions 0.83/0.17 at s1,
  0.98/0.02 at s0/s2). d_w = 6 is over-provisioned; a d_w ∈ {2, 3, 6}
  ablation is the cheap decisive test (proposed, not run).
- **Encoder input x̃ = x − κℓρ̄** (two-pass, seeds only) proposed as the
  structural fix for the amortisation gap — design note in the devlog
  2026-09-14; spec change pending the architect.
