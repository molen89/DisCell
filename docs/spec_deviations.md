# Deviations from the spec (`07-simple-spec_7.md`)

**2026-09-01, architect review**: every row in "chosen where the spec is
silent" and "deliberate departures" was ratified; the two spec-text findings
were accepted and the spec patched (§6.1 q-side derivation, §4.1 GAT-part-only
zeroing, plus §5 α ≈ 1/ℓ̄ and §7.10 within-type mirror). The straight-through
penalty gradient (issues T1) was judged *better than specified* and is now the
reference behaviour. Rows below stay for the record.

Everywhere the implementation differs from the spec, chose where the spec is
silent, or has not yet built something the spec describes. Each row is a
decision to surface, not a bug; bugs live in `docs/issues.md`.

## Chosen where the spec is silent

| where | spec says | code does | rationale |
|---|---|---|---|
| `enc_w`'s x input | `enc_w(c, t, z, x)` — transform unspecified | same `encode_counts` as `enc_z` (library-normalised + log ℓ) | **RESOLVED 2026-08-21**: author confirms x means the log1p-transformed counts, i.e. the §4.2 representation |
| `t` into `enc_w` / `m_ψ` | one-hot (author clarification) | **corrected**: one-hot everywhere t is an input; `embed(t)` exists only as the GAT query, sized to match the GAT source features it stands in for — **K** under the default `type_only` (sources = onehot(t_j) only), K + d_z under the spec-4.1 `type_z` sources | **RESOLVED 2026-08-21**: was `embed(t)` at a free width; fixed per author; sizing follows `gat_sources` since 2026-09-11 |
| `w_j` for ring-1 `ρ_j` | "`ρ_j` needs `w_j`" — which w unspecified | a sample from `q(w_j)`, same batched call as seeds, **detached** | **RATIFIED 2026-09-01** (architect: "agree without reservation") |
| isolated cells' leak | ~~silent~~ **now spec behaviour**: §4.1 as patched prescribes the per-row renormalisation, `κ_i = 0` | matches | **closed** — no longer a deviation |
| `v`-block parameterisation | `v = [y, PCs(Φ)]` | `[y minus one column, PCs(Φ)]` | `y` on the simplex makes Σ_v singular by construction; MI unchanged (issues M4). **RATIFIED 2026-09-01** |
| penalty computation | log-dets of covariances | log-dets of **correlations** | provably the same value (MI scale-invariance), bounded Σ⁻¹ gradient (issues M4) |
| penalty numerics | shrink-to-diag only; `+λI` prohibited | after shrink, `+1e-5·I` **on the correlation matrix** for slogdet | on correlations (diag = 1) this is a uniform relative jitter, not the variance-inflating ridge the prohibition targets — but it is literally an added identity, so it is registered |
| penalty input | `Pen` over `z` (which draw unspecified) | `mu_z`, not the sample | encoder noise would dilute the measured dependence; hiding dependence behind noise is what the penalty must not allow |
| penalty gradient | (not addressed) | straight-through: value from the EMA moments, gradient from the batch | without it, `alpha_a`'s strength silently scaled with `cov_ema` (issues T1) |
| objective granularity | `J = Σ_i [...]` (a sum) | per-seed **mean** | invariant to tile size, so the alphas mean the same at every batch shape |
| evaluation draws | (not addressed) | evaluation sweeps use posterior means (`sample=False`) | the early-stop signal should not ride reparameterisation noise (issues T8) |

## Spec-text findings (surfaced, spec left untouched)

| where | finding |
|---|---|
| §6.1 derivation *(patched into the spec 2026-09-01)* | the text justifies pulling the w-KL out of `E_q(z)` by the **p-side** ("`p(w|c,t)` depends on neighbours' codes rather than `z_i`") — but the p-side was never the obstacle; the **q-side** is: §4.3's own `q(w|c,t,z,x)` conditions on the sampled `z`, so the exact bound keeps the w-KL inside `E_q(z)`. The implementation is unaffected — evaluating the closed-form Gaussian KL at the one reparameterised `z` is an unbiased one-sample estimate of `E_q(z)[KL]` — verified numerically (assembled loss reproduces the formula to fp32). The spec's *justification* needs the q-side argument. |
| §4.1 isolated cells | ~~is the literal reading intended?~~ **RESOLVED 2026-08-21**: author confirms `c_i = 0` applies to the GATv2 part only; the image stays — which is what the code does |

## Deliberate departures (user-directed or measured)

| where | spec says | code does | status |
|---|---|---|---|
| `Φ_i` dimensionality | "Project to 32–64 dims" (§2) | **full-dimension by default** (384), `--phi-pca` to compress | mask arms landed (they answer type-AUC only); the deciding test is the **Φ ablation in the running calibration** — resolve on its Δ-recon |
| probe target for Φ | CE against `eΦ` clusters (`E_Φ ≈ 15–20`, soft k-means) | Gaussian CE (ridge/MSE) on the continuous `[y', PCs(Φ)]` block | **RATIFIED 2026-09-01**, with the architect's condition attached: one small-MLP probe cross-check at the decision point (running in `calibrate.py`) |
| graph construction | Delaunay; Voronoi faces | the bundle's stored Voronoi-face graph (Delaunay candidates, faces from a partition **clipped to 30 µm discs**) | clipping only removes faces between cells >~60 µm apart, all of which the 40 µm prune removes anyway — no observable difference, but the mechanism differs |
| encoder input (§4.2 / proposed §7.13) | `enc_z(x_i, t_i)`, `enc_w(c, t, z, x)` on raw counts | **`subtract_leak`** (default **off**): a second pass re-encodes the seeds from x̃ = clip(x − κℓρ̄, 0) (normalised by its own sum, log-depth = raw ℓ), enc_w reads x̃ too, likelihood untouched on raw x, ring-1 ρ_j keep their pass-1 encoding (accepted O(κ²) inconsistency), penalty/probes read pass-2 μ_z | **architect-approved spec change gated on results (2026-09-14)**: true posterior is p(z | x, ρ̄) and the bias κℓρ̄ cannot be removed by any function of x alone; result on three seeds **near-neutral** (niche-z residual −0.006 mean, 2/3 seeds in the predicted direction; recon ±0.001; NMI −0.017; Moran-w down in 2/3) — not green enough to change the default; the 07 patch waits; the per-cell benefit is A4's (parked) test |
| GAT sources (§4.1) | sources `[onehot(t_j), sg μ_z(x_j)]` | **`gat_sources = "type_only"`** by default: sources = onehot(t_j) only; `"type_z"` selectable for era reproduction, and pre-field checkpoints reload as type_z (`load_run` era guard) | **measured, then ratified by the user 2026-09-11** (doc-08 v6 step 0): 3-seed ablation better on nearly every axis — recon −7.2586 (type_z reference) → −7.2527 / −7.1924 / −7.2310, mirror 0.056 → 0.044–0.046, cycle_w 0.034 → 0.007–0.010, NMI held, w's spatial character intact; doc-09 showed neighbour-state detail in c bought nothing (exposure null in every arm). The case is **empirical only** — the mechanistic motivation (w-mirror) was falsified (devlog 2026-09-11) |

## Spec'd but not yet built (deliberately deferred)

| what | spec section | trigger for building it |
|---|---|---|
| adversary heads `ŷ_ξ, êΦ_ξ` + separate optimiser | §4.6 escalation | ~~trigger pending~~ **TRIGGERED AND BUILT 2026-09-01**: converged MLP-ΔCE was 46% of uncontrolled with the linear leak already gone — the nonlinear case, exactly |
| `eΦ` image-niche clusters and `Φ̄(t)` lookup | §2, §4.6 | **built** with the adversary (`soft_clusters`, E_Φ = K) |
| `α_a` operating-point sweep (ΔCE vs z–type NMI crossing) | §4.6 | **done** — four rounds; adversarial α_a = 0.3 (14% of uncontrolled, zero NMI cost); 1.0 reaches the floor |
| Dirichlet-Multinomial likelihood | §7.6 | posterior-predictive under-dispersion at higher depth |
| counterfactual machinery (`do(c = c′)` with abduction) | §7.9 | **type-level version built** as the doc-08 §7 transport check (`discell/model/transport.py`: Δ̂prog + Δ̂leak with z fixed, scored on held-out tiles); per-cell abduction stays out of scope by design |
| bounded learned `σ_w(c,t) ∈ [0.5, 2]` | §7.12 | explicitly optional; fixed `σ_w = 1` until a reason appears |
| κ estimation via nuclear/extranuclear split | §7.11 | out of scope for DisCell-simple |

## Hyperparameters the spec leaves open (current defaults)

**Calibrated on the slide, 2026-09-01** (two §4.6 rounds, 12 short fits +
one convergence check; `experiments/calibration*.json`): **`α_a = 0.03`**
(MLP-probe ΔCE 28% of uncontrolled at NMI −9%; 0.04 collapses NMI for 22
points of leak), **`α_w = 0.1`** (the M3 knife-edge reproduced on the slide:
0.03 → identity theft, 0.2 → w dead with no NMI gain; caveat — at 0.1 w is
near-pinned, KL ≈ 0.002/dim, so anomaly scores read conservative),
**`ω = 1`** (0.5 loses NMI everywhere, 2.0 trades leak for recon), `α_z =
0.007` (§5's 1/ℓ̄ rule), `d_z = 20`, `d_w = 6`, `v_pcs = 12`, lr `1e-3`, tiles
4096, Φ full-dimension (ablation: +0.009 per-count nats of held-out recon).
**Escalation resolved 2026-09-01**: the convergence check showed the residual
leak is nonlinear (ridge at floor, MLP at 46%), the adversary was built, and
the sweep runs with `invariance = adversary`, `α_a = 0.3`, `adv_steps = 6`,
`adv_lr = 2e-3` — MLP-ΔCE 14% of uncontrolled at zero NMI cost. Lesson kept in
issues A2: a weak adversary is a placebo; only the independent probe grades it.

**Current defaults (`TrainConfig`, 2026-09-11)**: `gat_sources = type_only`,
`invariance = adversary` (α_a = 0.3, 6 steps, lr 2e-3), κ = 0.1, α_z = 0.007
(lung: 0.004, the 1/ℓ̄ rule at its depth), α_w = 0.1 (the α_w study of
2026-09-10/11 found no safe re-calibration below it — seed-bistable), ω = 1,
d_z = 20, d_w = 6, v_pcs = 12, tiles 4096, lr 1e-3. `epochs = 200` /
`patience = 20` are the *sweep* budget; reference runs use `--epochs 500
--patience 40` — the strong-family optimum (recon −7.19) is a long-budget
phenomenon that 200-epoch sweep seeds do not reach (−7.256 ± 0.001).

# Deviations from doc 08 (`08-validation-analyses_1.md`, validation handover)

Same discipline as above: each row is a surfaced decision, evidence in the
devlog. The handover file itself is left as received; this register + the
devlog carry the deltas (ready for architect ratification like the 07 rounds).

| where | doc says | code does | rationale |
|---|---|---|---|
| §2.4 banding | "fit on cells with d ≤ 500 µm; report held-out R² per band" | **fit within each band** (block-CV inside the band) | the pooled fit scored band-locally punishes the global slope, not the band question — every probe went negative with the floor at zero, more signal = more negative (w −0.76 < z −0.25 < ℓ −0.16 < floor 0) |
| §2.1 inventory | one class per structural type; interface via graph mode-filter | `vasculature` = endothelial ∪ pericytes (co-location **measured** at 90%/30 µm); smooth muscle restricted to compact instances (20–500 cells); interface from kNN-smoothed tumour fraction (k = 50) | map diagnosis: sheets not structures; starved coverage (68–75% beyond cap); mode filter *converged* at 57.8% boundary on this interleaved slide (near-share 84%, target without variance) |
| §1 references | floor / ℓ-baseline / ceiling | **+ y-baseline** (probe from raw one-hop composition) in every landmark table | the landmark sets are type-defined and types are expression-derived (user-raised); y-baseline measures the definitional share instead of arguing about it |
| §2.6 θ | from the w-probe (band unspecified) | from the **mid-band** fit | the evidence band defines the direction; a 0–500 fit mixes the echo into θ |
| §2 claim | "ground truth from *geometry* … zero circularity with the counts" | claim scoped: true of the ruler, not the set | measured: interface mid-band y-baseline 0.042 vs w 0.069 → ~60% definitional, residual +0.027 reframed as compositional allegiance; **vasculature is the only geometry-grade test and is a certified null** (mid y −0.004, w 0.007). Literal zero-circularity needs image-derived landmarks (second-slide upgrade) |
| §3.1 weights | "binary or β_ij" | binary, row-normalised | β adds no discriminative value for the I asymmetry; one fewer knob |
| §4.3 expectation | "z within noise of max(floor, ℓ-baseline)" | measured **above** it: AUC z 0.65 vs ℓ 0.59 (w 0.76) | not silently accepted: registered as a watch item (issues), triaged benign (hot z dims y-R² ≤ 3.6%), and shown κ-reducible (+0.062 → +0.039 over the grid) |
| §5 matrix | rows incl. "mid-band landmark distances" | row present but averages three classes, diluting the interface read (0.069 shown as 0.029) | split into interface row + vessel-null-in-text proposed, **decision pending** |
| §5 matrix rendering | (unspecified) | one **absolute** strength scale (AUC 0.5→0, 0.85→1; R² 0→0, 0.5→1); cells that fail max(floor, ℓ)+margin grey out as "n.s."; significant signal in the *unexpected* column gets a dagger + footnote | per-row normalisation painted a −0.01-vs-0.02 row fully hot and hid the §4.3 niche-z flag (user-caught, 2026-09-11) |
| §6.1 basis | canonical basis unspecified beyond rotation-invariance | varimax on B's columns weighted by each dimension's realised variance | the *procedure* reproduces, raw columns do not; certified on planted sparsity (`tests/test_model_atlas.py`) |
| §6.3 hallmark labels | hypergeometric enrichment | on the **expressed-panel** background (verified — the suspected genome-background bug is absent), **BH across the sets tested per program, label ships only at q ≤ 0.05** | ungated labels shipped SPERMATOGENESIS at p = 0.08 (architect flag, doc-11) |
| §6.5 κ-survival | signature correlation vs the reference across the sweep | reported **sweep-internal** (reference `sweep3_k0.1_s0`) next to the vs-pinned-reference read | vs-s1 confounds κ with the long-budget optimum gap (0.29–0.43, flat); sweep-internal separates κ (0.62–0.64 within seed) from seed (0.25–0.34) |
| §7.2 counterfactual | Δ̂prog + Δ̂leak with z fixed | **as specified** — after a user-caught correction (the first build let the type's intrinsic mix vary across niches); the all-vary "model account" is kept as an extra row and its excess over the counterfactual (≈ 0.05 R²) is reported as the **selection share** | selection became a measurement in its own right |
| §7.3 pairs | the most composition-distinct pairs | **all** type × niche pairs evaluated; the overlap guard (1-D **along the gap direction**) assigns the tier | pre-selecting distinct pairs is self-defeating — they are exactly what the guard flags |
| §7.4 tiers | supported / extrapolation | with k-means niches the supported tier is near-empty *by construction* (3/158 panels); extrapolation (155 panels) is the informative regime and is named as such in the report | annotation niches (rim/core) would populate the supported tier; data-defined ones cannot |

# Deviations from doc 09 (`09-communication-experiment_1.md`)

| where | doc says | code does | rationale |
|---|---|---|---|
| §3.6 low-α_w arm | ablation at a lower α_w | run at α_w = 0.05 seed 0 (the clean sub-0.1 instance); 0.03 is seed-bistable | verdict unchanged: exposure visibility null in every arm (w-R² ≈ −0.01 in four arms), programs *degrade* in the low arm |
| §5a planted worlds A/B | calibration of the reattribution threshold | **not run** — the "61% leak-attributed" figure is uncalibrated and must not be quoted | doc-10/doc-11 took precedence; the gate scaffold (`synthetic.py`) + `applications/planted.py` now exist for it |
| §5f stability | (a winning pair) | the cross-model tally is the result: POSTN 2/3 arms, PDGFB 2/3, nothing 3/3; CD99 decoy-rejected | pre-registered stability doctrine; at most 1–2 fragile candidates is the cellAdmix expectation |
| exposure visibility | measured per arm | under `type_only` the null is **architectural**: c = f(type attention, composition, Φ) carries no channel for neighbour expression detail | measured in four arms before the architecture made it provable |

# Doc 10 (`10-zw-guard-test`): built, parked, removed

Guard built per §2 including the guard-view routing (naive build trained
`enc_z` through `enc_w`'s z input; caught by the routing test), arm 0 run
per §3 (grid α_w × α_zw on the planted world-A gate), verdict **PARK** per
the pre-registered §4 clause (guard-on never held what guard-off lost; KL_w
closed monotonically with α_zw — the backdoor-α_w failure). **Removed from
the code entirely** (user decision, 2026-09-11); what remains is
`data/experiments_synthetic/guard_gate.{json,png}` and the devlog record.
Any second attempt must first reproduce the historical M3 cliff in the
current loss configuration.

# Deviations from doc 11 (`11-z-applications_1.md`) — A4 only so far

| where | doc says | code does | rationale |
|---|---|---|---|
| A4 z call | per-cell calls from `z·β_S`, `z·β_G2M` | **rate-matched** per type: top-k by max(z·β_S, z·β_G2M), k = that type's raw-positive count | the two callers then differ only in *which* cells, never how many |
| A4 leg 3 (nuclear-only recomputation) | in the original A4 list | **dropped in v2** — the redesign names DAPI / gene-split / planted only; v1 showed the leg reads "weak calls are weak" | `qc/nuclear_counts.npz` stays built for A1's segmentation-perturbation leg |
| A4 planted-world read-out | phase-label recovery z vs raw at planted κ | victim FPR at a global threshold set inside the cycling types — **defective**: synthetic type offsets dominate (`control_fpr` 0.02–0.45 across seeds); the leak-attributable excess is victim − control, and no power gate exists | recorded, not fixed — A4 parked 2026-09-11 (devlog "A4 v2 results") |
| A4 DAPI | group-level violin | group-level medians + KS on `dapi_sum` standardised within type, boxes | FFPE sectioning truncates nuclei; supporting evidence only |

# Deviations from doc 09 §8 (`09-communication-experiment_5.md`, the LR co-occurrence ladder)

| where | doc says | code does | rationale |
|---|---|---|---|
| §8.1 significance | within-type permutation of the LR score (~200), BH per map | the 200 permutations calibrate the null's **scale**; the tail is read from a Gaussian at that scale (`p = 2Φ(−|z|/sd_null)`); the empirical p is kept in the JSON | a purely empirical p (min 1/201) can never clear BH over a 60–1000-cell map for a single true survivor — the planted unit test caught it |
| §8.1 null | permutation only | **plus** a Moran-preserving spatial shift (each cell takes the LR score of the same-type cell at a random 500–1000 µm displacement); survivors boxed — the printed verdict line | ratified as the "keeper" in §8.2b (4): a permutation null on ~60k spatially smooth cells is anti-conservative (issues V8) |
| §8.1 columns | Var(exposure) × receiver prevalence, top ~30 | Var(Ẽ) (composition-residualised, §2) × prevalence; **one column per ligand**; ≥ 2 receptor-eligible receiver types | §8.2b (3b), ratified |
| §8.1 rows | top ±10 loadings per active program | the effective-rank **program coordinates** (V10), labelled by top ±4 loadings | §8.2b (1) |
| §8.1 receptor side | depth-normalised log1p receptor counts | log1p of the model's clean rate ρ_g × median depth | §8.2b (3): raw 0–2 counts starve the Spearman; circularity stated in JSON + caption |
| §8.2 panels | A and B | **A′** (uncontrolled, pooled over all validation cells) added as the top rung | §8.2b (2): the SIMVI-comparable view; shows that published-map structure is type composition |
