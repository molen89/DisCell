# 08 — Latent-validation analyses (handover)

Post-hoc analyses testing the allegiance claim — `z` intrinsic, `w` spatial — beyond the
§4.6 training probe. Companion to `07-simple-spec.md`; notation follows its registry.
Already running: cell-cycle probe (z-retention; R² 0.30 from `mu_z` vs 0.02 from `mu_w`
on S/G2M scores) and pseudotime on w. This doc adds three analyses: distance-to-landmark
regression (§2), per-dimension Moran's I (§3), niche invariance (§4). §5 defines the
headline figure. Design principle throughout: every analysis is a target variable with a
*known allegiance*, probed from **both** latents against reference rows — the claim is
always the asymmetry, never one number.

## 1. Conventions (apply to every analysis)

- **Latents**: posterior means `mu_z`, `mu_w` (`sample=False`), converged model. Run only
  after the §4.6 probe has passed and a per-dim variance check on `mu_w`: `Var_i(mu_w)`
  clearly above zero (near-constant dims are collapsed — hatch them, §3.2). Small `KL_w`
  alone does NOT block: KL_w is the ego-specific increment over the context prior, and
  `mu_w ≈ m_ψ(c, t)` still varies richly with context — the analyses then test the
  context channel as a whole, which is still the allegiance claim. Do record that
  `KL_w ≈ 0` deactivates the anomaly score at that operating point (an α_w calibration
  fact for §4.6, not a blocker here).
- **Within type, always.** Type is legitimately in both channels (z carries identity;
  homophily makes the type mosaic spatial). Fit and report per type; pool only as
  cell-weighted means.
- **Probe family**: ridge for continuous targets, multinomial logistic for discrete —
  reuse the §4.6 probe machinery. Always fit the identical probe from `mu_z` AND `mu_w`.
- **Reference rows, per target** (all three analyses, every table):
  1. *floor* — within-type target permutation;
  2. *ℓ-baseline* — the probe from `log ℓ` alone: depth varies across tissue, so a latent
     that encodes depth buys spatial R² technically. One addition to the shared probe
     helper, reused everywhere; retrofit the existing cycle probe with it — G2/M cells
     carry ~2× mRNA, so ℓ is not trivially cold even for intrinsic targets;
  3. *ceiling* (expression-derived targets only) — the probe from the top-50 PCs of
     log-normalised x.
- **CV**: spatial-block folds — reuse training tile ids as fold groups (~5 folds of
  contiguous tiles). Random CV is invalid for every spatial target here: autocorrelation
  leaks the answer through neighbouring cells landing in train and test.
- **κ and seeds**: all three are cheap post-hoc → run per κ-sweep point with multi-seed
  envelopes (like B). Headline numbers at the operating κ.
- **Notation**: `θ ∈ R^{d_w}` = probe coefficient vector (`β` is taken by the graph
  weights `β_ij`); `d(i, L)` = distance from cell i to landmark class L.

## 2. Distance-to-landmark regression (w-allegiance)

Ground truth from *geometry*, not expression — zero circularity with the counts.

### 2.1 Landmark inventory
One distance map per landmark **class**, using everything available:
- annotation-derived: tumor/stroma boundary, region borders, any pathologist polygon;
- structure-derived from structural cell types: e.g. vessels from endothelial cells —
  DBSCAN on that type's centroids (start eps = 40 µm, min_samples = 8; tune per class),
  each cluster = one instance; drop instances < 20 cells and classes < 5 instances;
- derived compartment interfaces (no pathologist needed): mode-filter the type map over
  the graph into dominant compartments (tumor/stroma), take the boundary as a landmark
  class — built from `t` and positions only, same data status as y.

### 2.2 Target
`d(i, L)` = Euclidean centroid distance to the nearest constituent cell (cKDTree) or
annotation polygon edge of class L. Regression target `u = log1p(min(d, 500 µm))` — raw
distance is unbounded and biologically meaningless in the far field.

### 2.3 Exclusions
The landmark's own constituent type is excluded from its regression (endothelial →
vessel distance ≡ 0). Isolated cells stay in (their w rests on the Φ-only context).

### 2.4 Distance bands — where the claim lives
- **near, < 30 µm** (≈ one hop): composition echo — the GAT sees the landmark's cells
  directly; success is trivially spatial. Sanity check only.
- **mid, 50–300 µm: the evidence.** One-hop context cannot see the landmark; prediction
  must run through induced expression response (or Φ). This is the spatial-effect thesis.
- **far, > 300 µm**: expected failure for both latents; do not count it against the model.

Fit on cells with d ≤ 500 µm; report held-out R² per band.

### 2.5 Collinearity
Landmark classes are correlated (vessel density itself varies rim → core), so marginal
R²s overlap. Report marginal R² and say so; optionally add one joint regression on all
distance maps and report partial R² per landmark.

### 2.6 Direction analysis ("w captures different things")
- `θ_L` per landmark from the w-probe, unit-normalised: pairwise cosine matrix, per type
  (check consistency across types). Near-orthogonal ⇒ multi-axis spatial code;
  all-aligned ⇒ one generic "edge-ness" dimension.
- Gene signature per landmark: `B θ̂_L` (G-vector of log-fold loadings) → top ±15 genes.
  Expect interpretable programs (hypoxia/angiogenesis along vessel distance, …). Valid
  within one trained model only — never compare raw w dims or θ across runs/seeds
  (rotation non-identifiability); compare gene signatures instead.
- With ≤ 2 usable classes the cosine *matrix* degenerates: report the single cosine(s)
  and situate each `θ_L` against the pseudotime direction and the top B columns instead.

### 2.7 Deliverables
Heatmap landmarks × {w, z, floor, ℓ-baseline} of mid-band R² (per type + pooled);
R²-vs-band curves; θ cosine matrix; gene-signature tables.

## 3. Per-dimension Moran's I (both latents; needs no annotation)

Cheapest of the three, and the best per-κ sweep diagnostic.

### 3.1 Procedure
1. Center each latent dim **per type** (subtract the type mean). This removes the
   legitimate type mosaic from both latents while keeping the full graph — no induced
   per-type subgraphs, which fragment for scattered types.
2. Graph: the model's own pruned graph; weights binary or `β_ij`, row-normalised → W.
3. For every dim v of `mu_w` and `mu_z`: `I = (n/ΣW) · (vᵀWv)/(vᵀv)`; permutation null
   by shuffling the centered values over cells (≥1,000 shuffles; squidpy
   `spatial_autocorr` with `n_perms` does all of this).

### 3.2 Reading
- w dims: I ≫ null expected — except prior-collapsed dims. Cross-reference per-dim
  `KL_w` and annotate those: for a collapsed dim, I ≈ 0 is vacuous, not a failure.
- z dims: expected ≈ null band. **A high-I z dim is a flag, not automatically a
  failure**: intrinsically-spatial biology (clone patches, proliferative niches) is
  allowed — z claims independence of *context-induced* variance, not spatial uniformity
  of intrinsic state. Triage a hot z dim: correlates with cycle score / CNV clone →
  benign; regresses on y → leakage (and the §4.6 probe should have caught it too).

### 3.3 Deliverables
One bar chart: all dims of both latents, I with null band, coloured by latent,
collapsed w dims hatched. Line plot of mean |I| per latent across the κ sweep.

## 4. Niche invariance of z (z-allegiance, exclusion direction)

Cycle shows z *retains* intrinsic signal; this shows z *excludes* spatial signal. Both
directions are needed — chance-level z is only meaningful next to the retention result.

### 4.1 Niche labels — from data, never from latents
Priority: annotation regions if available. Else k-means on neighbour composition y
(optionally ⊕ top PCs(Φ)) over connected cells, K = 10 (robustness at K = 6 and 15).
Defining niches from w or z is circular; y and Φ are data.

### 4.2 Eligibility and fit
Types with ≥ 2 niches holding ≥ 500 cells each. Per eligible type: multinomial logistic
`mu_z → niche` and `mu_w → niche`, class-weighted (niche prevalence is imbalanced),
spatial-block CV mandatory — niches are spatially contiguous, so random CV leaks the
label through any residual spatial signal.

### 4.3 Metric and references
Macro one-vs-rest AUC (+ balanced accuracy). floor = within-type label permutation
(≈0.5); the ℓ-baseline is mandatory here (density/depth co-varies with niche). Expect:
w ≥ ~0.8; z within noise of max(floor, ℓ-baseline).

### 4.4 Presentation
Where annotation exists, add per-pair AUC on named pairs (tumor rim vs core) for the
paper text. Caption note: this is the presentation-grade cousin of the §4.6 training
probe, run post-hoc; it does not replace the probe.

## 5. Headline figure: the allegiance matrix

Rows = targets with known allegiance: S/G2M score (intrinsic), niche label (spatial),
mid-band landmark distances (spatial), pseudotime (existing). Columns = `mu_z`, `mu_w`,
with floor / ℓ-baseline / ceiling marks per cell. Expected pattern: block-diagonal —
each target hot in exactly one column, and every cold cell certified by its references.
Render at the operating κ; companion line plot of the four key numbers vs κ with
multi-seed envelopes. Order of operations: §4.6 probe pass → `KL_w` check → matrix.
