# TRISECT-simple

Single tissue. Total counts. Cell-type labels. Spatial graph. Registered image.
One training stage, one optimiser, four hyperparameters.

---

## 1. Goal

Split a cell's expression into what it **is** and what its **environment did to it**, while
accounting for the fact that some of its transcripts belong to other cells. `z_i` carries
intrinsic state, `w_i` carries the cell's response to its microenvironment, and a fixed leakage
mixture absorbs misassigned transcripts. Because leakage and spatial signalling produce the same
first-order signature — both make a cell resemble its neighbours — the leak fraction `κ` is not
estimated but **swept**, and the deliverable is which spatial effects survive the sweep.

---

## 2. Notation

### Observed

| | |
|---|---|
| `x_i ∈ ℕ^G` | counts, `G ≈ 5000` genes |
| `ℓ_i` | observed total `Σ_g x_ig`. Fixed, never modelled |
| `t_i ∈ {1..K}` | cell-type label or frozen pseudolabel |
| `Φ_i ∈ ℝ^{d_Φ}` | image patch embedding, centre masked. Project to 32–64 dims |
| `y_i ∈ Δ^{K−1}` | neighbour type composition, `y_ik = Σ_{j∈N(i)} 𝟙[t_j=k]/\|N(i)\|`. Excludes ego |
| `eΦ_i ∈ {1..E_Φ}` | image-niche cluster, k-means on `Φ`, `E_Φ ≈ 15–20` |

### Graph

`N(i)` = **Delaunay** neighbours of `i`, excluding `i`. Two cells are neighbours iff their Voronoi
cells share a face — i.e. no third cell lies between them. No `k`, no radius, average degree ≈ 6,
adapts to local packing, and uses only centroids so it is immune to segmentation boundary noise
and to the nuclear-expansion setting. **Prune edges longer than ~30–50 µm** (or the 95th
percentile) — Delaunay otherwise connects cells across lumens, vessels and tears. Same graph for
the GAT and for `β`.

| | |
|---|---|
| `face_ij` | shared **Voronoi** edge length, µm. Contact-extent proxy, well defined whether or not the segmented polygons touch. Used in `β` only |
| `d_ij` | centroid distance, µm. All Delaunay neighbours are first-order, so cell-size correction is unnecessary here |
| `τ` | 20 µm, the contamination decay length |
| `β_ij` | `∝ face_ij · exp(−d_ij/τ)`, normalised so `Σ_j β_ij = 1`. Sparse, precomputed once |

Segmented shared-wall length is **not** used: under nuclear expansion, whether two cells touch is
largely a measurement of local density, which is a niche property we are trying to keep out of the
technical layer.

### Latent — the only sampled variables

| | |
|---|---|
| `z_i ∈ ℝ^{d_z}` | intrinsic state, `d_z ≈ 20` |
| `w_i ∈ ℝ^{d_w}` | spatial response, `d_w ≈ 4–8` |
| `w̆_i` | a draw from the **prior** `p(w\|c_i,t_i)`, used in loss term (b) |

### Deterministic

| | |
|---|---|
| `c_i` | context descriptor, `GATv2((x_src, x_dst), edges) ⊕ Φ_i`. **No edge features** — §7.3 |
| `x_dst` | GAT query, `embed(t_i)`. **Not `z_i`** — §7.2 |
| `x_src` | GAT key/value, `[onehot(t_j), sg μ_z(x_j)]` |
| `α_ij` | attention weight, `Σ_j α_ij = 1`. Distinct from loss weights `α_z, α_w, α_a` |
| `ρ_i ∈ Δ^{G−1}` | cell's **own clean** composition |
| `ρ̄_i ∈ Δ^{G−1}` | **foreign influx**, `Σ_j β_ij ρ_j` |
| `p_i ∈ Δ^{G−1}` | fitted multinomial probability, `(1−κ)ρ_i + κρ̄_i` |
| `sg[·]` | stop-gradient |

### Fixed before training

`κ` (leak fraction, **swept on a grid**), `β`, `Φ`, `t`, `ℓ`, and two lookup tables: `ȳ(t)` = mean
neighbour composition per type, `Φ̄(t)` = mean image-niche distribution per type.
`σ_w = 1` — fixed deliberately, not for simplicity (§7.12).

### Learned

| | |
|---|---|
| `enc_z` | MLP `(x_i, t_i) → (μ_z, log σ_z²)` |
| `enc_w` | MLP `(c_i, t_i, z_i, x_i) → (μ_w, log σ_w'²)`. Note the prime: `σ_w'` posterior, `σ_w` fixed prior |
| `m_ψ` | prior mean net `(c_i, t_i) → ℝ^{d_w}` |
| `a_g` | decoder MLP `z_i → ℝ^G`. **Takes no `t`** — §7.1 |
| `B ∈ ℝ^{G×d_w}` | response loadings. Column `k` is a gene programme; `w_ik` is how far cell `i` moved along it |
| `ŷ_ξ, êΦ_ξ` | adversary heads `(z_i,t_i) → Δ^{K−1}, Δ^{E_Φ−1}`. Separate optimiser |

### Hyperparameters

`ω` (path-b weight), `α_z`, `α_w`, `d_w`. Plus `α_a` if the adversary is on.

---

## 3. Generative model

```
z_i               ~  𝒩(0, I)
c_i               =  GATv2( query=embed(t_i), src={[onehot t_j, sg z_j]} ) ⊕ Φ_i
w_i | c_i, t_i    ~  𝒩( m_ψ(c_i,t_i) , σ_w² I )

log ρ̃_ig          =  a_g(z_i) + ⟨w_i, B_g⟩          ρ_i = softmax_g(ρ̃_i)
ρ̄_i               =  Σ_{j∈N(i)} β_ij ρ_j
p_i               =  (1−κ) ρ_i + κ ρ̄_i

x_i | ℓ_i         ~  Multinomial( ℓ_i , p_i )
```

**Conditional independencies.** `z_i ⊥ [y_i,Φ_i] | t_i` — within a type, intrinsic state says
nothing further about location. `x_i ⊥ c_i | z_i, w_i, {ρ_j}` — the environment reaches the cell
only through `w` and through leakage.

---

## 4. Approximation

### 4.1 `c_i` — describes the environment

*One vector per cell summarising who and what is nearby. Deterministic, shared by co-located cells.*

```
x_dst = embed(t_i)                    x_src = [ onehot(t_j) , sg μ_z(x_j) ]
GATv2: add_self_loops = False, 1 layer, 4 heads, concat = False, NO edge features
c_i   = GATv2((x_src, x_dst), edges) ⊕ Φ_i
```

- **Query is the type, not `z_i`.** Using `z_i` lets attention mirror the cell through similar
  neighbours, making the prior ego-informed and killing the anomaly score (§7.2).
- **No geometry edge features.** The Delaunay graph already encodes *who* is adjacent; `Φ_i`
  encodes *how packed* the region is. Continuous geometry in `c_i` would put un-interventionable
  variation into the estimand (§7.3).
- **One layer ⇒ exactly one hop.** Neighbour sampling needs `num_neighbors=[k]`, nothing deeper.
- Isolated cells (all edges pruned) softmax over an empty set → NaN. Set `c_i = 0` with a flag.

### 4.2 `q(z_i | x_i, t_i)` — infers what the cell is

*Diagonal Gaussian from an MLP.* **Never sees `c_i`** — invariance by architecture.

### 4.3 `p(w)` and `q(w)` — expected response vs inferred response

*The prior is what a typical cell of this type would do here; the posterior is what this one did.*

```
p(w_i | c_i, t_i)              =  𝒩( m_ψ(c_i,t_i) , σ_w² I )        context only
q(w_i | c_i, t_i, z_i, x_i)    =  𝒩( μ_w(·) , diag σ_w'² )          context + ego
```

`KL(q ‖ p)` therefore reads directly as **how far this cell deviates from its expected response**
— the anomaly score, free from the objective. The prior is also what makes counterfactuals
possible: `do(c = c′)` means drawing `w′ ∼ p(w|c′,t)` and decoding.

`q` needs the **direct `x_i` path**, not just `z_i`: `w ⊥ x | z,c` is false by explaining-away,
and `z` has been actively purged of the niche-predictable signal `w` exists to detect (§7.4).

Both posteriors carry learned diagonal variances. The **prior** variance stays fixed at
`σ_w = 1`: a learned constant is non-identifiable, and a learned `σ_w(c,t)` opens a degenerate
direction (§7.12).

### 4.4 Decoder — `t` deliberately absent

*`a_g(z)` is the baseline programme; `⟨w, B_g⟩` is the environmental shift.*

Withholding `t` from the decoder is what forces `z` to carry cell identity. Give the decoder `t`
and `z` degenerates into a residual code (§7.1).

### 4.5 Frozen neighbours

*Neighbour quantities are treated as data, not as things to backprop through.*

`ρ_j` and `z_j` enter under **stop-gradient**. Not optional: resolVI disables gradients on this
exact path with the comment *"mode collapse when gradient used"*, and the linearised feedback
amplification is `1/(1−κ)`.

### 4.6 Invariance — keeps the environment out of `z`

*Architecture alone is not enough: `x_i` carries environmental signal, and `d_z ≫ d_w` means `z`
will absorb it given the chance.*

Target `I(z_i ; [y_i, Φ_i] | t_i) = 0`. **Both targets, not just `y`** — penalising only
composition leaves `z` free to absorb the morphology-defined niche, and `w` then under-reports
morphology-driven effects (§7.5).

**Default — closed form, no second optimiser:**

```
Pen_i-block = Σ_t p(t) · ½[ log det Σ̂_{v|t} + log det Σ̂_{z|t} − log det Σ̂_{[z,v]|t} ],   v = [y, PCs(Φ)]
```

**Escalation — adversary**, only if the probe below says the penalty is too weak:

```
Adv_i = [ CE(y_i,  ȳ(t_i))  − CE(y_i,  ŷ_ξ(z_i,t_i))  ]
      + [ CE(eΦ_i, Φ̄(t_i))  − CE(eΦ_i, êΦ_ξ(z_i,t_i)) ]
```

Excess predictive skill over knowing the type alone; adversary maximises, encoder minimises, zero
at the optimum. Log the two halves separately.

**Discretising `Φ`.** Only the adversary needs it — the closed-form penalty takes continuous PCs
directly, which is one more reason to start there. k-means on the **projected** embeddings (32–64
dims) or top PCs, never raw 1024-d, where distances are noise-dominated. Set `E_Φ = K` so the two
heads face comparable difficulty and one `α_a` serves both. Fit once, freeze, and use **soft**
assignments (distance softmax) rather than hard — smoother CE target, no boundary artefacts.

**The probe — how to decide.** Train a *fresh* predictor of `[y_i, Φ_i]` from `(z_i, t_i)` on a
train split and evaluate on held-out cells. Never the training adversary; never the training
cells. Report excess nats against the type-only baseline:

```
ΔCE = CE(y, ȳ(t)) − CE(y, ŷ_probe(z,t))          on held-out cells, same for Φ
```

Two reference scales make `ΔCE` interpretable: the **noise floor**, `ΔCE` when `z` is permuted
across cells within type (should be ≈0), and the **uncontrolled baseline**, `ΔCE` from a model fit
at `α_a = 0`.

Decision rule: escalate to the adversary if held-out `ΔCE` stays above ~20% of the uncontrolled
baseline and clearly above the noise floor.

Operating point: sweep `α_a` and plot `ΔCE` and `z`–type NMI against it on the same axis. You want
`ΔCE` near the floor while NMI is still above its own floor. Where those two curves cross is the
answer; if there is no such point, the invariance is costing you cell type and `t` is too fine.

### 4.7 Leakage — fixed, swept

*A precomputed sparse mixing operator, not a learned component.*

`ρ̄_i` needs the neighbours' `ρ_j`, which nominally needs `w_j`, hence `c_j`, hence two hops.
Resolve by **caching `ρ_j`** in a buffer refreshed each step: one hop suffices, cost is slight
staleness. Alternatives are two-hop sampling (exact, ~100× the nodes) or `ρ_j ≈ softmax(a_g(z_j))`
(one hop, biases the leak term).

`κ` is a hyperparameter. Fit independently at each `κ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4}` — `κ = 0` is
the nested no-correction null. Report each spatial effect **as a function of `κ`**: stable across
the grid → a finding; changes sign or vanishes → not separable from leakage. That range is the
deliverable. It is not a confidence interval and does not shrink with more cells.

---

## 5. ELBO and loss

```
J = Σ_i [   Σ_g x_ig log p_ig( z_i , w_i )                                  (a) reconstruction
        + ω Σ_g x_ig log p_ig( z_i , w̆_i ) ,  w̆_i ∼ p(w|c_i,t_i)            (b) intrinsic path
        − (1+ω) · α_z · KL( q(z_i) ‖ 𝒩(0,I) )
        −         α_w · KL( q(w_i) ‖ 𝒩(m_ψ(c_i,t_i), σ_w² I) )
        −         α_a · Pen_i          (or Adv_i)                            ]

KL( 𝒩(μ,σ²) ‖ 𝒩(m,s²) ) = Σ_k [ log(s_k/σ_k) + (σ_k² + (μ_k−m_k)²)/(2s_k²) − ½ ]
```

Maximise over model parameters; minimise the adversarial term over `ξ` if used. One decoder,
evaluated twice — `w` from the posterior in (a), from the prior in (b).

**Scale the reconstruction by `1/ℓ_i`** before weighting. It is `O(−200…−400)` per cell against
KLs of `O(1–10)`, and its magnitude varies several-fold between sections; unnormalised, `α_z, α_w`
will not transfer across your own data.

**This is a weighted surrogate, not a bound**, once any `α ≠ 1`, and `x_i` appears in both (a) and
(b). Uncertainty comes from the `κ` sweep, not from `q`.

---

## 6. Derivations

### 6.1 The reconstruction bound

With `c_i` and `ρ̄_i` held constant by §4.5:

```
log p(x_i | c_i,t_i)
  = log ∫∫ p(x_i|z,w) p(w|c_i,t_i) p(z) dz dw
  ≥ E_{q(z)q(w)}[ log p(x_i|z,w) + log p(w|c_i,t_i) + log p(z) − log q(z) − log q(w) ]
  = E_{q(z)q(w)}[ log p(x_i|z,w) ] − KL(q(z)‖p(z)) − KL(q(w)‖p(w|c_i,t_i))
```

Jensen at line 3. The `w`-KL would normally sit inside `E_{q(z)}`, but `p(w|c_i,t_i)` depends on
neighbours' codes rather than `z_i`, and those are frozen — so it is an ordinary Gaussian KL.

### 6.2 The intrinsic path, and where `(1+ω)` comes from

A second bound on the same data, for the model in which `w` is not inferred — set
`q(w) := p(w|c_i,t_i)`, zeroing that KL:

```
log p(x_i|c_i,t_i) ≥ E_{q(z)} E_{p(w|c_i,t_i)}[ log p(x_i|z,w) ] − KL(q(z)‖p(z))
```

Call the two bounds `L_i` and `L_i^z`. Each carries its own `−KL(q(z)‖p(z))`, so `L_i + ω L_i^z`
contains `−(1+ω)·KL(q(z_i)‖p(z))`. Omitting the second copy — the natural mistake — costs the
bound property even at unit weights.

This mirrors DisCoVR, where `L_z + L_w` corresponds to minimising
`KL(q_z‖p_{z|x}) + KL(q_z q_w‖p_{z,w|x,y})` — a sum of two non-negative divergences. Coherent as
an M-estimation target, but **not** an ELBO for a single model, so `q` is not the posterior of
anything.

### 6.3 Why (b) is load-bearing

Nothing else rewards `z` for carrying signal: unlike DisCoVR's `μ_k = E[z|y=k]`, our prior on `w`
never references `z`. Term (b) supplies that pressure directly — `z` must explain the cell given
only the *expected* response. It runs through the same leakage mixture, so it never asserts
`κ = 0` and contradicts (a).

### 6.4 Adversary as posterior regularisation

`Adv_i` / `Pen_i` is not part of any bound. It is posterior regularisation (Ganchev et al.) — a
constraint on the variational family written as a penalty — the same status DisCoVR gives its
adversarial term.

---

## 7. Appendix

### 7.1 Why the decoder gets no `t`, and what stays in `z`

The decoder is `a_g(z_i) + ⟨w_i,B_g⟩`, so the only route from identity to expression runs through
`z`. If `z` collapsed to a copy of `embed(t)`, both reconstruction terms would lose all within-type
variance — a large fitting penalty. That is what holds `z` open. Give the decoder `t` and the
pressure vanishes: `z` becomes a residual-only code and stops being interpretable as identity. If
reconstruction is poor, widen `z` or the MLP; do not add `t`.

What lives in `z` beyond the label: cell cycle phase; clonal identity (CNV, clonotype); position on
a continuum the label discretises (naive→effector→memory→exhausted); stress and metabolic state;
historical states acquired in a niche since left; and **systemic signals** — a circulating cytokine
or drug affects all cells of a type regardless of position, so it is not niche-predictable and
lands in `z` despite being environmental colloquially.

Coarse `t` (10–20 types) leaves more for `z`. Check by scoring cell cycle with standard gene sets
and confirming `z` separates phases within a type.

### 7.2 The mirror attractor

GATv2 computes `e_ij = a^T LeakyReLU(W_l h_i + W_r h_j + W_e edge)` and aggregates
`c_i = Σ_j α_ij W_r h_j` — the destination feature enters the **attention** but never the
**value**. So `z_i` as query injects no ego content directly, but it opens an attractor: attention
learns `α_ij ∝ exp(sim(z_i,z_j))`, reconstructing the cell through similar neighbours so
`c_i ≈ W_r z_i`. The incentive is real — an ego-informed prior is free decoder capacity — and it is
strongest in homotypic regions, where `κ` is also least identified.

Consequence: `m_ψ` already knows the cell, `KL_w` collapses, `w` measures deviation from *self*.
`embed(t_i)` closes it at no cost, and adds nothing to the prior since `m_ψ(c_i,t_i)` already takes
`t_i`. What you give up is state-specific attention, which belongs in the decoder (as `B_g(t)` or a
low-rank `z`-interaction), not smuggled through the niche encoder.

**Diagnose:** `R²` of `z_i` regressed on `c_i` against a permuted-neighbourhood control, or
`corr(α_ij, sim(z_i,z_j))`. **Warning:** downstream this is indistinguishable from ordinary
posterior collapse — free-bits will mask it and leave the anomaly score meaningless.

Zeroing `z_i` while keeping self-loops is **not** an alternative: softmax still spends attention
mass on the self-edge, down-weighting neighbours, and under `z ∼ 𝒩(0,I)` the origin is the prior
mean, so a zero vector asserts "this cell is average" rather than "ignore this cell".

### 7.3 Why the GAT gets no geometry, and why the graph is Delaunay

**The decisive argument is the counterfactual, not collinearity.** `c_i` is the conditioning
variable, and the headline product is `do(c = c′)`. If `c_i` depends on continuous geometry, a
counterfactual asks "what if my neighbours had been at different distances with different contact
areas" — and that is not interventionable. Packing, cell size and contact extent are determined by
tissue architecture and cell identity; "the same neighbours, but further away" is not a well-posed
biological question. Putting un-interventionable variation into the conditioning variable
contaminates the estimand. This is why the causal framing picks *type composition* as the exposure.

Two supporting points. `Φ_i` already encodes local density, packing and architecture, and does it
better than ten scalar edge weights. And the dynamic range is small — within a first-order
neighbourhood distances span perhaps 5–30 µm, most of which *is* local density, which is again in
`Φ`.

**Why not a contact-based shell indicator either.** A discretised `touching / near / far` encoding
looks like it dodges the objection, but in Xenium under nuclear expansion whether two segmented
polygons touch is largely a measurement of local density — dense regions touch, sparse regions do
not. The indicator would smuggle density back in, in the confounded direction, and a tolerance
parameter only moves the threshold.

**Delaunay resolves it.** Two cells are neighbours iff their Voronoi cells share a face, i.e. no
third cell lies between them — the correct notion of adjacency for a packing. No `k`, no radius,
average degree ≈ 6, adapts to local geometry, and built from centroids alone so it is immune to
both segmentation boundary noise and the expansion setting. The one parameter that returns is
pruning long edges (~30–50 µm, or the 95th percentile), because Delaunay will connect cells across
lumens, vessels and tears. Much less sensitive than choosing `k`.

The resulting division of labour is clean: **the graph says who is adjacent, the image says how
packed the region is.** Two components, two jobs, no duplication.

**Continuous geometry survives in `β`, where it belongs** — `β` is not part of the conditioning
variable and does not touch the estimand. Use **Voronoi face length**, not segmented shared wall:
the Voronoi face is where the boundary between two cells effectively lies, its length is a
contact-extent proxy defined whether or not the polygons touch, and it is independent of the
expansion setting. Strictly better for this purpose.

**The fallback is empirical.** If `m_ψ` predicts responses poorly, or held-out reconstruction is
worse than an A/B run with `edge_dim=3` carrying `[d_ij, face_ij, centroid distance]`, that is
direct evidence the continuous signal mattered. Add it back and monitor `corr(α_ij, β_ij)` — high
correlation means the GAT has rediscovered the leak kernel, and `w` is becoming collinear with
`ρ̄`. That risk is tolerable while `κ` is fixed (there is no free contamination parameter to feed)
and is not tolerable once `κ` is estimated in `05`.

### 7.4 Why `q(w)` needs `x_i`

`w ⊥ x | z,c` is false: `x` is generated from **both** `z` and `w`, so knowing `x` and `z` informs
`w` directly. This is the `z ⊥̸ w | x` explaining-away that DisCoVR builds its objective around, and
why their posterior is `q(w|x,y)`, not `q(w|z,y)`.

`z` is also lossy in exactly the wrong direction: 20 dims from 5000; the invariance penalty
explicitly removes niche-predictable variation, which *is* the spatial response; and term (b)
shapes `z` toward identity. Without the direct path, `w_i` becomes `E[response | type, niche]` —
identical for co-located similar cells — and `KL_w → 0`.

**Three routes to `KL_w → 0`**, indistinguishable from the training log:

| Cause | Check | Fix |
|---|---|---|
| Posterior collapse | `α_w` large vs reconstruction | lower `α_w`, free-bits |
| Mirror attractor | `R²` of `z_i` on `c_i` vs permuted control | `embed(t_i)` as query |
| `w` starved of ego evidence | is `x_i` wired into `enc_w`? | add the direct path |

### 7.5 Why the invariance target includes `Φ`

Penalise only `y` and `z` is purged of composition-predictable variation but not of
image-predictable variation — while `c_i` contains `Φ_i`. `z` then absorbs the morphology-defined
niche (vessel proximity, fibrosis, density, necrosis) and `w` correspondingly under-reports
morphology-driven effects. `y` and `Φ` are correlated, so the part that matters is the residual:
image-predictable-but-not-composition-predictable. That residual is what goes unpenalised.

### 7.6 Why multinomial, not NB

If `x_c ∼ Poisson(λ_c)` independently then `(x | Σx = ℓ) ∼ Multinomial(ℓ, λ/Σλ)`. Since `ℓ_i` is the
observed total, the multinomial *is* the coherent likelihood — no library latent, no double-counting.

NB absorbs overdispersion, which at 50–200 counts across 5000 genes (most entries 0 or 1) you have
no power to estimate; a 20-dim amortised `z` captures much of what NB's dispersion mops up in
coarser models. NB is also actively wrong for a mixture: `NB(μ₁+μ₂,θ)` forces both sources to share
one Gamma draw, inflating variance to `(μ₁+μ₂)+(μ₁+μ₂)²/θ` rather than the correct
`(μ₁+μ₂)+(μ₁²+μ₂²)/θ`, so `κ` could absorb overdispersion that is not leakage. Under Poisson the
mixture is exact. If posterior predictive checks show under-dispersion at higher depth, use
**Dirichlet-Multinomial** — one concentration scalar, not 5000 dispersions.

### 7.7 Why `κ` is swept

Leakage and spatial signalling are confounded — both make a cell resemble its neighbours, and
neighbour composition predicts both. A model free to fit `κ` places it wherever the likelihood is
flattest, and nothing here identifies it. Sweeping is honest and it *is* the contribution.

Grid range from three sources: 0 as anchor; published Xenium estimates (0.1–0.5); and the
marker-set ceiling `Σ_{g∈M} x_ig / (ℓ_i Σ_{g∈M} ρ̄_ig)` for an **atlas-defined** `M`. Never rank
genes by `ρ̄_g/ρ̂_g` from the same counts — that selects the zero-count genes and drives the bound
to 0 with probability 1.

### 7.8 Why conditional, not marginal, invariance

Types cluster in space, so `t_i` and `y_i` are strongly dependent. A marginal adversary enforcing
`I(z;y)=0` deletes cell type from `z`, because carrying "I'm a T cell" predicts "my neighbours are
T cells". DisCoVR's Table 19 shows exactly this: CSVAE drives `z`–stimulation NMI to 0.002 while
collapsing `z`–cell-type NMI from 0.716 to 0.406.

The cost: intrinsic variation that *is* spatially organised within a type gets pushed into `w` —
tumour subclones being the sharp case. The decomposition is relative to `t`, not absolute. Prefer
coarse `t`: fine `t` leaves `z` with nothing to carry, blocks more of the position→fate→expression
path, and destroys overlap.

### 7.9 Why `w` is not the environment

`c_i` is the environment — deterministic, shared by co-located cells, a conditioning variable.
`w_i` is the **response**: latent, per-cell, differing between two cells in the same niche. If you
only want a niche descriptor, drop the latent and feed `c_i` straight to the decoder — that is
roughly NCEM and it is simpler. `w` earns its place only for within-niche heterogeneity:
anomalies, per-cell responses, and counterfactuals with abduction
(`w′_i = m_ψ(c′,t_i) + σ_w ε̂_i`, carrying the cell's residual `ε̂_i = (w_i − m_ψ(c_i,t_i))/σ_w`).

### 7.10 Diagnostics

```
R²(z_i ~ c_i) vs permuted control     mirror check
corr(α_ij, β_ij)                      only if the edge-feature fallback is on (§7.3)
KL_w per dimension                    → 0: rule out the mirror BEFORE free-bits
z–type NMI                            hard floor; the CSVAE failure mode
held-out I(z;[y,Φ]|t)                 from an INDEPENDENTLY trained probe, never the training one
I(z;t)/H(t), var(z|t)                 degeneracy: is z just t?
Δ held-out recon, z vs one-hot t      same, more direct
```

Early stop on held-out reconstruction **and** `z`–type NMI jointly. A model explaining everything
with leakage will have excellent reconstruction.

### 7.11 Upgrade path

This version **sweeps** `κ`; it cannot estimate it. Estimation needs the nuclear/extranuclear split
(`05-model-spec.md`): two views of one cell sharing a foreign profile at different contamination
levels. Its identifying condition is that gene-wise nuclear retention `η_g` **varies across
genes** — with `η` constant both views lie on the same line between `ρ_i` and `ρ̄_i` and nothing is
pinned. Adding it does not restructure anything: the likelihood becomes a multinomial over `2G`
bins and `κ` moves from hyperparameter to latent.

**Note:** `05` writes the foreign influx as `φ_i`; here it is `ρ̄_i`, to keep `Φ` free for the image.

### 7.12 Learned variances — what to learn and what not to

**Posteriors: learn diagonal variances.** `q(z|·)` and `q(w|·)` both do. This is what makes the
reparameterised sample meaningful and lets per-cell uncertainty vary — a cell with 30 counts
should have a wider `q(z)` than one with 300.

**Prior on `w`: keep `σ_w = 1` fixed.** Two separate reasons, and the first is decisive.

A learned *constant* diagonal `σ_w` is **non-identifiable**. `w` and `B` enter the decoder only as
`⟨w_i, B_g⟩`, so rescaling dimension `k` of `w` by `s` and column `k` of `B` by `1/s` leaves the
model unchanged. A learned constant `σ_w` is exactly that rescaling. It adds parameters and
changes nothing.

A learned *conditional* `σ_w(c,t)` is meaningful — it claims some niches produce more variable
responses than others — and it is not absorbed by rescaling, since it varies per cell. It even
improves the anomaly score in principle, by standardising deviation against local variability: a
deviation of 2 in a niche where everyone deviates by 2 should not read as anomalous. But it opens
a degenerate direction: **the model can shrink `KL(q‖p)` for free by inflating `σ_w(c,t)` wherever
the penalty bites**, making the prior vague rather than making `w` predictable. It also breaks the
concavity argument inherited from DisCoVR.

If you want it, bound it — `σ_w ∈ [0.5, 2]` through a scaled sigmoid — so it cannot inflate away,
and check that `E[σ_w(c,t)]` has not drifted upward over training.

**Prior on `z`: `𝒩(0,I)` is fine here.** A richer prior (mixture-of-Gaussians with one component
per type, or a VampPrior with archetypal pseudo-inputs) is a genuine upgrade for cell-type
structure and would make `t` a latent rather than an input — but that is a different change, not a
variance question.
