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
| `t` into `enc_w` / `m_ψ` | one-hot (author clarification) | **corrected**: one-hot everywhere t is an input; `embed(t)` exists only as the GAT query, sized **K + d_z** to match the source features `[onehot(t_j), z_j]` it stands in for | **RESOLVED 2026-08-21**: was `embed(t)` at a free width; fixed per author |
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

## Spec'd but not yet built (deliberately deferred)

| what | spec section | trigger for building it |
|---|---|---|
| adversary heads `ŷ_ξ, êΦ_ξ` + separate optimiser | §4.6 escalation | held-out ΔCE > ~20% of the uncontrolled baseline and clearly above the noise floor — the probe decides, per spec |
| `eΦ` image-niche clusters and `Φ̄(t)` lookup | §2, §4.6 | only the adversary consumes them |
| `α_a` operating-point sweep (ΔCE vs z–type NMI crossing) | §4.6 | **running** (`calibrate.py`, 2026-09-01) |
| Dirichlet-Multinomial likelihood | §7.6 | posterior-predictive under-dispersion at higher depth |
| counterfactual machinery (`do(c = c′)` with abduction) | §7.9 | after a κ sweep produces stable effects worth interrogating |
| bounded learned `σ_w(c,t) ∈ [0.5, 2]` | §7.12 | explicitly optional; fixed `σ_w = 1` until a reason appears |
| κ estimation via nuclear/extranuclear split | §7.11 | out of scope for DisCell-simple |

## Hyperparameters the spec leaves open (current defaults)

`α_z = 0.007`, `α_w = 0.1`, `α_a = 0.02` (post-T1: the straight-through fix
made the old 0.3 twenty-fold stronger, so the default was rescaled to its old
*effective* size), `ω = 1`, `d_w = 6`, `d_z = 20`, `v_pcs = 12`, lr `1e-3`,
tiles 4096 cells. The αs come from the synthetic recovery gate (issues M2/M3)
and are **calibration starting points, not findings**. `α_a` and Φ are being
decided by the running calibration; **`α_w` and `ω` are not in that grid** —
`α_w` gets its own short scan before the sweep (issues M3), `ω = 1` stands
unexamined on real data.
