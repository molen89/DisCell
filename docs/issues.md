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

## Open / watch list (updated after architect review, 2026-09-01)

- **α_w calibration on real data** — the knife-edge (M3) must be set on the
  slide, not carried over from the synthetic gate. **NOT covered by the
  running calibration** (which grids α_a and ablates Φ at fixed α_w = 0.1):
  a short α_w scan {0.03, 0.06, 0.1} judged on KL_w-alive + NMI + probe is the
  one remaining pre-sweep run. The architect's MLP probe cross-check IS in the
  running calibration.
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
- **Mirror R²** should get a within-type variant: the current z~c regression
  partly measures type separability, which both carry legitimately.
