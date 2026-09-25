# Development log

What was decided, and why. Entries are appended, not edited — a decision that
turned out wrong gets a later entry saying so rather than a rewrite, because the
reasoning that produced it is the part worth keeping.

Measurements are from `Xenium_Prime_Ovarian_Cancer_FFPE` (407,120 cells, 5,101
genes, 0.2125 µm/px) unless another slide is named.

---

## The question the repo exists to answer

Can a cell's identity be predicted from its neighbourhood — and if so, from
what? The pipeline exists to hand a model three things a standard k-NN spatial
pipeline throws away:

1. **real geometric edge weights**, measured on the segmentation polygons rather
   than inferred from centroid distance
2. **raw counts** for the cell and every neighbour, untransformed on disk
3. **a morphology-image embedding** per cell

Everything below is in service of making those three trustworthy.

---

## Graph structure

### Xenium cells barely touch, so exact contact is unusable

The first assumption to die. Xenium segmentation is *measured*, not inferred —
97.8% of cells on the lung slide come from interior or boundary stains and only
2.2% from nucleus expansion — so the polygons might be expected to tile the
tissue. They do not. Boundaries come very close without coinciding: a quarter of
neighbouring pairs sit within **0.12 µm**, below the 0.2125 µm pixel, yet only
2.5% touch at exactly zero distance.

Exact-contact adjacency therefore collapses. On 20k lung cells:

```
tol 0.0 um ->  1,448 edges  mean_deg 0.14  isolated 87.7%
tol 0.5 um -> 23,937 edges  mean_deg 2.39  isolated 12.7%
tol 1.0 um -> 25,960 edges  mean_deg 2.60  isolated 10.8%
tol 2.0 um -> 30,203 edges  mean_deg 3.02  isolated  7.6%
tol 5.0 um -> 39,139 edges  mean_deg 3.91  isolated  3.4%
```

**Decision:** `XENIUM_CONTACT_TOLERANCE_UM = 1.0`. It is the point where the
graph stops being degenerate without the tolerance dominating the geometry.

### Two graphs, kept side by side

`contact` answers *who is physically adjacent*; `voronoi` answers *who would be
adjacent if the cells filled the space*. They disagree enough that picking one
up front would have been a guess:

| graph | edges | mean degree | isolated | median shared wall |
|---|---|---|---|---|
| `contact` (1 µm) | 634,557 | 3.12 | 9.6% | 0.00 µm |
| `voronoi` (clip 30 µm) | 1,197,730 | 5.88 | 0.1% | 6.85 µm |

**Decision:** build both, store both, choose at load time. The cost is one extra
pass at preprocessing and some disk; the alternative is rebuilding a 407k-cell
graph every time the question changes.

### `shared_wall_um` is useless on the contact graph — hence `apposed_wall_um`

The median shared wall on the contact graph is **0.00 µm**, and only 1.65% of
contact edges have any at all. Xenium's touching cells meet at a *point*, not
along a wall. So the one metric that ought to say "how much membrane do these
two share" carries no information on the graph where it matters most.

Voronoi's `shared_wall_um` is always positive, but it measures the boundary
between two *territories* sitting in the empty space between cells, and is
essentially uncorrelated with whether the cells are adjacent at all: **r =
−0.107** against the gap, with pairs 15–100 µm apart still scoring ~5.9 µm.

**Decision:** add `apposed_wall_um` — for each pair, the length of cell *i*'s
boundary lying within a tolerance of cell *j*, and vice versa, averaged. Measured
on the real polygons, not the tessellation. It correlates with the gap as it
should (**r = −0.51** at 1 µm) and is nonzero on 99.99% of contact edges.

The adjacency tolerance and the wall tolerance are deliberately **separate
parameters**: the graph decides *who* is a neighbour, the wall decides *how much*
they share, and the two want different values.

### The default graph became `voronoi` — a bug forced the issue

`beta` is a fixed smoothing weight, `beta_ij = face_ij · exp(−d_ij/τ)`,
row-normalised per destination so a distribution smoothed over the neighbourhood
stays on the simplex. The loader fed it `shared_wall_um` **from whichever graph
was selected**, and the default was `contact` — where that metric is zero on 98%
of edges.

Measured consequence: **94.5% of connected cells got an all-zero `beta` row**.
The field advertised in the batch contract as "row-stochastic" was not, for
almost every cell. `priors.smoothing_weights`' own docstring warned this would
happen; nothing checked it, because the only guard fired on a *missing* key, not
an all-zero one.

**Decision:** `DEFAULT_GRAPH = "voronoi"`. It is an exact planar partition, so
every edge has a positive shared face by construction and `beta` is well defined.
After the change, zero rows fell from 386,866 to **527 — exactly the isolated
cells**, which is what the docstring always claimed. That equality is now a test
(`test_beta_is_row_stochastic_over_every_in_edge`).

The alternative — keep `contact` and feed `beta` from `apposed_wall_um` — would
also have worked (it drops zero rows to 9.6%). It was rejected because voronoi
makes the invariant true *structurally* rather than by choosing a better proxy.

### Nothing is trimmed on the way in

Two edge filters existed: one baked into the stored bundle at export, one applied
at load. Both are gone.

**Decision:** every edge the graph builder finds is written and served. The
metrics needed to filter a graph all travel with it in `edge_attr`, so a consumer
that wants a subset can take one — but a bundle that threw edges away could never
get them back. The load-time filter also had a failure mode worth recording: on a
bundle lacking `apposed_wall_um` it silently compared against a zero-filled
column, dropped **every** edge, and reported it as an INFO line before crashing
elsewhere with a `TypeError` about sparse matrices.

---

## Labels

### 19 classes where there should have been 18

The loader mapped un-annotated cells to a literal `"unassigned"`. 10x's own
`cell_groups.csv` already ships an `"Unassigned"` category. `sorted()` put them
at opposite ends of the label space, and every one-hot, every K×K composition
table and every accuracy number was computed over **two classes meaning the same
thing** — 1,984 cells in one, 513 in the other.

**Decision:** fold the fill onto the panel's own spelling, matched
case-insensitively, so a slide that ships `UNASSIGNED` or `unassigned` cannot
reintroduce the split. Asserted by `test_no_two_labels_differ_only_by_case`.

---

## Image embeddings

### KRONOS v1 vs v2

Both are wired up and selected with `--model`. They are different contracts, not
different sizes of the same thing:

| | v1 | v2 |
|---|---|---|
| architecture | marker-aware ViT-S/16 | marker-aware DINOv2 ViT-B/16 |
| embedding | 384-d | 768-d |
| marker identity | integer `marker_id` | marker **name** |
| vocabulary | 177 markers | 288 markers |
| normalisation | the caller applies mean/std | `model.preprocess` does it |
| out-of-vocabulary | borrow an unused id | register a novel marker |
| throughput (native 256 px) | ~637 cells/s | ~122 cells/s |
| separation / 1-NN | +0.4184 / 36.2% | +0.4166 / 36.2% |

**On the evidence so far the two are indistinguishable on this slide** — 0.0018
of separation apart, identical 1-NN — at 5× the cost for v2. That is a
suspiciously tight tie for models of different size and vocabulary, which is part
of why the current experiment scores both rather than settling on v1.

**Decision:** keep both behind one extra and one flag. Do not pick until a
measurement distinguishes them.

### The marker mapping is approximate, and says so

Xenium's four morphology channels map onto KRONOS's vocabulary imperfectly:

| ch | marker | v1 id | note |
|---|---|---|---|
| 0 | DAPI | 4 | exact |
| 1 | NAKATP | 442 | antibody **cocktail** (ATP1A1+CD45+E-Cadherin); named for the pan-membrane component |
| 2 | 18S | **505** | ribosomal RNA — the vocabulary contains no RNA markers at all |
| 3 | A-SMA | 130 | αSMA/Vimentin cocktail |

**Decision for channel 2:** an out-of-vocabulary id (505, outside the trained
range 4–498) rather than borrowing a real marker's. Marker embeddings are
deterministic sincos of the id, so an untrained id yields a vector the model
never associated with any protein — which is honest. Borrowing CD45's id would
actively assert the channel is a protein it is not. v2 instead registers 18S as a
*novel marker* through the model's own hook, using statistics measured on this
slide's crops.

### Intensity scaling mattered more than the model

KRONOS expects intensities in roughly [0, 1] — its shipped per-marker means run
0.005–0.08 — and its own loader divides by the dtype maximum. Xenium morphology
is uint16. Feeding raw values put the marker z-score four orders of magnitude out
and every patch far outside the pretraining distribution.

Fixing it moved v1 from **+0.2724 / 24.1%** to **+0.4184 / 36.2%** separation /
1-NN. **A larger gain than anything else tried, including the entire v1 → v2
change.** Files written since record `intensity_scaled: True` so a stale artefact
stays identifiable.

The lesson worth keeping: with a foundation model, the preprocessing contract is
a bigger lever than the checkpoint.

---

## 2026-08-20 — what is the image embedding actually reading?

**Status: running.** Seven arms over 407,114 cells; results land in
`data/datasets/<id>/experiments/ego_masking.{csv,json}`.

### The question

The crop KRONOS is fed is far wider than the cell. At the previous setting — 256
px at the slide's native 0.2125 µm/px — the field is 54.4 µm and contains a
**median of 21 other cells**. So the "per-cell image embedding" may be describing
the neighbourhood, not the cell.

That matters because the model is *also* given the neighbourhood as a graph. If
the image vector is mostly reading surrounding tissue, then its apparent
contribution may be homophily the graph already carries, counted twice.

### Design

Three arms over identical patches, plus a resolution control:

| arm | patch |
|---|---|
| `unmasked` | the whole patch |
| `ego` | a fixed-radius disk at the anchor zeroed |
| `ego_only` | everything outside the cell's polygon zeroed |
| `ego_only_hi` | as above, at the slide's native resolution |

**Not the polygon, for arm 2.** Masking with the cell's own outline would leave a
hole that is an exact silhouette of the cell — deleting the pixels while keeping
the single most type-informative morphological feature. A **disk** so there is no
orientation artefact, at a **fixed** radius so the hole is identical for every
cell and therefore carries no information about the one removed. Filled with
**zeros**: the hole is an artefact whatever it contains, and a constant artefact
is one the model encodes identically for every cell.

The polygon mask is arm 3 instead, where the silhouette is the point.

### The radius came from the data, not a guess

The quantity that matters is the radius of the smallest disk centred on the *crop
anchor* that fully contains the cell — not `equiv_diameter_um`, which is derived
from area and badly underestimates anything elongated. Over all 407,120 cells:

```
p50 6.29   p90 9.69   p99 14.25   p99.9 18.27   p99.99 22.56   p100 36.83  (um)
```

The tail is **real**: elongated smooth-muscle spindles, circularity 0.18–0.36,
area only ~350 µm². Not segmentation failures.

| radius | covers | % of a 128 µm field |
|---|---|---|
| 15 µm | 99.327% — 2,740 stick out | 4.3% |
| 22 µm | 99.988% — 47 out | 9.3% |
| **25 µm** | **99.999% — 6 out** | **12.0%** |

**Decision: 25 µm, and the six cells that do not fit are dropped from *every*
arm.** A cell poking out of the hole would leak exactly the identity the hole
exists to remove, so partial coverage is not an option; excluding 6 of 407,120
is. Covering the literal maximum (36.8 µm) would mask 34% of the field and leave
almost no context, which would answer a different question.

`data/datasets/<id>/experiments/ego_masking_examples.png` draws this: the outline of the
largest cell that fits sits tangent to the disk, and the 36.8 µm spindle visibly
crosses it.

### Patch geometry: 256 px at 0.5 µm/px

Two problems with the previous 54.4 µm field, one fix. A 50 µm-diameter mask in a
54.4 µm patch removes 24% of the area and leaves barely one cell layer of
context; and 0.2125 µm/px is finer than anything KRONOS saw.

**Correction worth recording:** the "FMs are trained at 20× / 0.5 µm/px" figure
comes from H&E pathology models (UNI, Virchow). KRONOS is a multiplex-IF model —
its own model card gives a reference patch of 256 × 256 and its worked example
runs at **mpp = 0.37**. So resampling coarser is right, but 0.5 is *past* the
reference rather than at it.

**Decision:** 256 px (KRONOS's documented reference size, and larger than the 224
first proposed) at 0.5 µm/px → a **128 µm field**, mask 12.0% of area, 25→64 µm
annulus of context. A 0.37 µm/px arm is one flag away if the result turns on it.

### Scoring

Multinomial logistic regression predicting cell type from the embedding; macro
one-vs-rest AUC.

**The split is spatial, in contiguous tiles with a margin**, and this is not
optional. Neighbouring cells' 128 µm patches overlap almost completely, so a
random split would put the same pixels on both sides. Worse, the baseline
predicts a cell's type from its neighbours' *labels* — a random split hands it
the answer directly. Cells within half a patch of a tile edge are dropped rather
than assigned.

Every arm is measured against **`AUC(type | neighbour composition)`**: cell type
from the neighbours' labels alone, no image at all.

| comparison | reads as |
|---|---|
| `AUC(unmasked) − AUC(ego)` | what masking actually buys |
| `AUC(ego_only)` | what the cell carries on its own |
| `AUC(ego)` vs the baseline | whether the masked patch knows anything beyond homophily |

The last row is the one that decides. If `AUC(ego)` sits at the baseline, the
masked patch is reading type-homogeneous neighbourhoods and nothing more —
expected, legitimate, and already handled by conditioning on `t`. If it sits
clearly above, the cell leaves an imprint on its surroundings that survives its
own removal. And if `AUC(ego) ≈ AUC(unmasked)`, masking is not earning its keep
and the cheaper grid-tile version should be used instead.

### Known caveats, recorded before the numbers arrive

- **The 25 µm disk removes a median of 18 cells**, not one. At this cell density
  it necessarily takes the ego cell *and its first ring or two*. So `ego`
  measures "type predictable from tissue beyond ~25 µm" — arguably a cleaner
  question than partial-ego-removal, but not the same as "the ego cell removed".
- **`ego_only` is resolution-starved at 0.5 µm/px**: a median cell spans ~17 px,
  barely one 16-px ViT token. `ego_only_hi` at native resolution (~40 px, 2.5
  tokens) exists so a low score can be told apart from not having supplied enough
  pixels.
- **The smallest "cell" on the slide is 2 µm²** (covering radius 1.2 µm), which
  is not a cell. The size floor deserves its own look.

### Follow-up already identified

Crops read pyramid level 0 and downsample. For a 0.5 µm/px target, **level 1 is
0.425 µm/px** — reading it instead would cut bytes and interpolation ~4× and take
the run from ~8.5 h to ~2.5 h. Not changed mid-experiment: all seven arms must
share one resampling path to stay comparable.

---

## Infrastructure decisions

### Bundles

An AnnData alone cannot hold a run: `uns['polygons']` is shapely geometry, which
h5ad cannot serialise. A bundle is a directory — `.h5ad`, polygons as WKB
parquet, one edge parquet per graph, a params JSON. The edge parquets are
redundant with the `obsp` matrices and exist because a table is easier to reason
about than four sparse matrices.

### Everything keyed by dataset

The two cohorts hold 46 candidate samples. In a flat tree, an artefact's slide is
guessable only from a filename prefix, so the second sample processed either
collides or silently pairs one slide's bundle with another's vectors — **neither
of which raises**. Putting the dataset in the *path* makes both impossible, and
every artefact is self-describing besides: bundles carry `uns["dataset_id"]`,
every `.pt` a `dataset` key, and the loader refuses a pair that disagrees.

### One preprocessing entry point

Eight separate CLIs became `python -m discell.preprocess`, with stages that skip
when their output exists. The package now splits along the artefacts —
`preprocess` produces, `data` consumes, and nothing in `data` imports from
`preprocess`. Two seams had to be cut for that to be true: the bundle reader was
split from the writer, and the TIFF layer (previously duplicated across
`crops.py` and `cell_graph.py`) was collected into `discell/tiff.py`.

### A note on `.gitignore`

`data/` as a bare pattern matches **any** directory named `data` at any depth,
including `discell/data/`. Five modules the package cannot import without were
invisible to git — `git ls-tree` confirms `geometry.py` and `labels.py` were
never committed at all, so `HEAD` did not import from a fresh clone from the
first commit until 2026-08-20. Fixed to `/data/`. Worth recording because
`git status` showed nothing wrong the entire time.

---

## 2026-08-20 — the DisCell model: implementation begins

The model of `discell_specs.md`, built in validated stages under
`discell/model/` — equations, networks, ELBO, metrics and the training loop as
separate modules so each is testable against an independent reference.

### The dataloader was already most of the data contract

Nothing about that was luck-free: the spec's `N(i)` is Delaunay/Voronoi-face
adjacency (the stored `voronoi` graph), its `face_ij` is `shared_wall_um`, its
`β` is *literally* `priors.smoothing_weights`, its `y_i` is `constant.niche`,
its `ȳ(t)` is `constant.neighbour_composition`, and its centre-masked `Φ` is the
ego-masking experiment's arm 2, being computed as this is written.

### Batching: contiguous tiles + exact two hops, no ρ cache

The spec resolves the two-hop dependency of `ρ̄` (neighbour ρ needs neighbour w
needs neighbour c) with a cached-ρ buffer — 8.3 GB at 407k × 5101 and one epoch
of staleness. **Replaced with contiguous spatial tiles**: a tile's seeds plus
ring1 (neighbours) plus ring2 (their neighbours) give the exact computation,
because rings only grow at the tile perimeter. Measured on the full slide:
ring overhead **21% of seeds at ~1.6k-seed tiles, 9.7% at ~6.4k**, assembly
0.9–2.7 ms/tile. Four batched calls — enc_z on all three sets, one GAT with
edges into seeds∪ring1, enc_w+decoder on seeds∪ring1, scatter for ρ̄ — no
per-neighbour loops, no cache, no staleness to reason about. Loss on seeds only.

### τ unified at 20 µm — and it barely matters, measurably

The loader's β default was 10 µm, the spec's contamination decay is 20. One
value now (20), because the halo and the leak kernel must agree edge-for-edge.
The standing worry — does β at 20 µm still concentrate? — dissolves on
measurement: within first-ring Delaunay neighbourhoods, distances span so
little that β is **face-dominated**. Effective neighbours (perplexity of β
rows): 4.39 at τ=10, 4.68 at τ=20, 4.90 with no decay at all, against median
degree 6. Corollary: κ-sweep results are unlikely to be sensitive to τ in this
range.

### Pruning at 40 µm, with QC

p95 of voronoi edge length is 27.4 µm; 40 µm prunes 1.11% of edges — the
Delaunay-across-lumens tail. Per-cell edges-lost is kept as a QC column
(16,455 cells lose ≥1 edge, max 10). Pruning newly isolates **372 cells**
(527 → 899): all carry the isolated flag, get `c = GAT-part zero + flag`, a
zero β row, and a zero `y` row. `y` and `ȳ(t)` are recomputed on the *pruned*
graph rather than reused from the loader, for the same edge-for-edge reason.

### Covariance penalty: EMA + shrink-to-diagonal

The closed-form `I(z;v|t)` penalty needs per-type covariances of `[z, v]`; rare
types give singular batch estimates. Design: the covariance used is
`(1−η)·EMA.detach() + η·batch` — history conditions, the batch term carries the
gradient — with shrinkage **toward diag(Σ)**, not `+λI`: an additive ridge
inflates every variance and biases the log-det most where data is scarcest,
weakening the penalty exactly for rare types. Shrink-to-diag keeps the
marginals and damps only the correlations. `v` will use 8–16 PCs of Φ (~20–30
joint dims); types under an effective-sample floor of 10× the dimension are
excluded and the excluded fraction is logged — those cells are unregularised.

`Φ` itself enters `c` at full dimension by default, with PCA compression as an
argument — the projection decision waits on the ego-masking results.

### Gates passed so far (36 model tests)

- **equations**: multinomial ≡ scipy; KL ≡ `torch.distributions`; penalty ≡
  analytic Gaussian MI on constructed covariances; the §7.8 property holds
  (marginal dependence through `t` scores ~0 conditionally); EMA survives
  8-cell batches of 12-dim data; shrinkage preserves variances exactly.
- **prepare**: β row-stochastic per destination on the pruned slide graph;
  rings disjoint with the closure property (every neighbour of a seed ∈
  seeds∪ring1); tiles partition exactly; leak edges are the seed-prefix of the
  GAT edges with β attached.
- **networks**: attention sums to 1; a single-edge destination's output is
  independent of the query (the value never sees the destination — the
  mirror-attractor property, §7.2); empty destination → exact zero, no NaN;
  `rho_bar` carries **no autograd graph at all**; the GAT source path leaks no
  gradient into `enc_z`; term (b) trains `m_ψ`; the decoder has no route from
  `t` (embed_t gets no gradient); `enc_z`'s signature admits no `c`.

### ELBO and the recovery gate (stages 3–4)

`elbo.py` assembles the spec's J and is tested **end-to-end against
`torch.distributions`** at three (κ, ω) settings — the whole loss, not just its
parts — plus the `(1+ω)` factor (§6.2's "natural mistake") as its own test. One
model correction surfaced on the way: the leak mixture now **renormalises per
row**, so an isolated cell degenerates to `κ_i = 0` instead of paying a constant
`−log(1−κ)` for leakage it cannot receive.

The synthetic-recovery gate (`test_model_recovery.py`, 5 assertions on one
fitted simulation) passes, and finding its operating point taught four things
worth more than the pass:

1. **`α = 1` is not the ELBO once reconstruction is 1/ℓ-scaled.** Scaling makes
   recon O(1) while KLs stay absolute, so unit α prices a latent's information
   ~ℓ-fold too high — z collapses (KL→0, NMI 0.21). The ELBO-equivalent point
   is α ≈ 1/ℓ̄.
2. **α_w is a knife-edge, measurably.** Low (~1/ℓ̄): `q(w)`'s direct x-path lets
   w steal identity (NMI 0.12). High (≥0.1): w collapses onto its prior
   (KL_w → 0.000) — §7.4's exact failure, now with numbers. The κ-sweep runs
   must calibrate this via the spec's §4.6 probe procedure, not by feel.
3. **The invariance penalty had a gradient bomb.** `y` sums to 1, so the
   v-block covariance is singular *by construction*; and deep inside a niche
   the within-type y-variance is ~0 — both make the logdet gradient (Σ⁻¹)
   explode → NaN. Fix is exact, not a patch: compute the penalty on the
   **correlation form** (MI is invariant to per-dimension scaling, the log-std
   terms cancel), plus one dropped y column. The analytic-MI test confirms the
   value is unchanged.
4. **The penalty does its designed job**: at α_a = 0.3, B's leading principal
   cosine against the planted programme space rose 0.556 → **0.825** while the
   z→niche leak (within-type R²) fell — deny z the niche and the response
   pathway sharpens.

Gate numbers (K=8, d_w=2, planted κ=0.2): z–type NMI 0.62 against an honest
ceiling of 0.77 (k-means on the *counts* themselves — `z_true` at 0.97 is not
reachable from 150-count multinomial draws), w first canonical correlation
0.80, B principal cosine 0.83. Thresholds frozen well under those.

### Ego-masking, first two arms (v1)

Baseline `AUC(type | neighbour composition)` = **0.9076** on the spatial split.
`unmasked_v1` = **0.8502**, `ego_v1` = **0.8285**. Two readings, pending the
remaining arms:

- **Masking costs 0.0216 AUC** — the patch's type signal barely depends on the
  central 25 µm. Masking is nearly free, so as ego-leak insurance for `c_i` it
  stays.
- **Both patch arms sit *below* the homophily baseline** — the image embedding
  carries no type information beyond what neighbour labels already give (it
  does not even match them). By the spec's decision rule this is the
  "expected, legitimate" case: nothing about the masked patch's skill demands
  a mechanism beyond type-homogeneous neighbourhoods.

`ego_only` (what the cell alone carries) and the v2 arms land overnight.

---

## 2026-08-20 (late) — training infrastructure, verification, and the full mask experiment

### Ego-masking: all seven arms

Spatial split, macro one-vs-rest AUC for cell type, baseline
`AUC(type | neighbours' labels)` = **0.9076**:

| arm | v1 | v2 |
|---|---|---|
| unmasked (whole patch) | 0.8502 | 0.8485 |
| ego (25 µm disk zeroed) | 0.8285 | 0.8183 |
| ego_only (cell alone) | 0.8728 | **0.8960** |
| ego_only at native resolution | **0.8927** | — |

Three readings:

1. **The cell alone beats the whole patch** — by +0.023 (v1) and +0.048 (v2),
   and at native resolution by +0.043. The surrounding tissue does not add type
   information to the embedding; it *dilutes* it. KRONOS pools the patch, and
   95% context pixels drown the one cell that differs from them.
2. **Everything the context knows is homophily.** Both masked arms sit ~0.08
   *below* the neighbour-label baseline; by the spec's decision rule that is
   the "expected, legitimate" case — no mechanism beyond type-homogeneous
   neighbourhoods is needed to explain the context signal. For `Φ_i`'s role in
   `c_i` (describe the niche, not the cell) this is exactly what we want, and
   the ego mask holds the ego out for 0.02 AUC — cheap insurance, kept.
3. **v1 vs v2 finally separate — on the cell, not the context.** ego_only:
   v2 0.896 vs v1 0.873. On masked context they tie (v1 even slightly ahead).
   Since the model consumes the *context* embedding, **v1 remains the choice**
   at 5× the speed; v2's edge lives where the model deliberately does not look.
   The resolution effect (0.873 → 0.893 for v1) confirms the earlier caveat
   that 0.5 µm/px starves single-cell readout.

### The trainer, and what an adversarial hunt found in it

`model/train.py` + `model/metrics.py`: TensorBoard scalars (every loss term,
per-dim KL_w, mirror R² with its permuted control, probe ΔCE with its noise
floor), spatial-w / z-PCA / B-loading figures on a schedule, joint early
stopping, one run directory per κ point.

A six-agent verification workflow over `discell/model/` produced nine
infrastructure fixes (issues T1–T9). The two worth remembering:

- **T1**: the invariance penalty's only live gradient path ran through the
  `ema × batch` term, so `alpha_a`'s real strength was `alpha_a × cov_ema` —
  a hyperparameter silently coupled to a smoothing constant. Fixed with a
  straight-through estimator (value from the EMA, gradient from the batch),
  and `alpha_a` rescaled to keep the effective size.
- **T2/T3**: two RNG couplings — the train/val split moved when `phi_pca`
  toggled, and a *figure* schedule consumed the training-shuffle stream.
  Sweep runs must share one split and logging must never move the fit; both
  now hold by construction.

The derivation check confirmed §5, §6.2 (the `(1+ω)` factor) and the penalty ≡
Gaussian conditional MI, and surfaced one **spec-text gap in §6.1**: the
factored w-KL is justified by p-side independence, but the actual dependence is
q-side (`q(w)` conditions on the sampled `z`); the code is a correct one-sample
estimator of the proper bound either way. Registered in
`docs/spec_deviations.md`, spec left untouched.

### B under κ is seed-bistable — the confound, made visible

At planted κ = 0.2, B's recovery flips between seeds (principal cosine
0.15–0.83, same config); in a no-leak world it is stable (0.65/0.83). Within a
region `w` is nearly constant, so `B·w` produces regional expression shifts —
which is also exactly what the leak term produces. One fit at one κ cannot
apportion them; **this is §7.7 observed in vitro**, and it is why the sweep is
the deliverable. The recovery gate now asserts sharp recovery only where it is
identifiable (κ = 0) and stability-plus-type where it is not (κ = 0.2).

### Stability probes on the slide (407k cells, 4 configs × 12 epochs)

All finite, no NaN, ~7 s/epoch on one 4090 (~25 s with eval). lr 1e-3 ≥ 3e-4;
α_z 0.007 and 0.02 both stable; the soft-α_w probe (0.03) showed the knife-edge
live on real data — reconstruction improving while NMI slid 0.33 → 0.27 — and
the **joint early stop refused to bless those checkpoints**, which is the
guard doing precisely what §7.10 wants. Mirror R² ~0.5 against a permuted
control of ~0.03 needs the within-type variant before it is read as an alarm
(open list). These are stability results only; no biology is claimed from
12-epoch runs.

---

## 2026-09-01 — architect review folded in; calibration under way

The architect ratified every registered choice (the straight-through penalty
gradient judged *better than specified* — its attenuation advice **was** the T1
coupling), owned E2 and the 15 µm radius, accepted both spec-text findings, and
patched the spec in four places. The authority is now **`07-simple-spec_7.md`**:
§6.1 carries the q-side derivation, §4.1 zeroes the GAT part only (Φ stays,
κ_i = 0), §5 records α ≈ 1/ℓ̄ as the sweep centre, §7.10's mirror is
within-type.

Code moved to match the author clarifications and the patched spec:

- **`t` is one-hot wherever it is an input** (`enc_z`, `enc_w`, `m_ψ`);
  `embed(t)` survives only as the GAT query, sized **K + d_z** to live in the
  same space as the source features `[onehot(t_j), z_j]` it stands in for. The
  free `t_dim` knob is gone. Recovery gate re-passed 6/6 under the change.
- **Mirror R² is within-type** with a within-type permuted control;
  `test_mirror_r2_is_blind_to_pure_type_separation` proves the property the
  patch exists for (pure type separation scores < 0.01, genuine within-type
  coupling scores far above its control).

Architect additions now encoded as requirements rather than intentions:

- `sweep.py` (built): ≥3 seeds per κ, B-stability via optimally matched
  column correlations both across seeds and along κ, per-type ‖w‖ envelopes,
  and held-out reconstruction **stratified by the edges-lost QC column** —
  partial isolation (degree 1–2 gets full κ against a thin ρ̄) is accepted but
  must be visible in the report.
- `calibrate.py` (built, running): the §4.6 α_a operating point on the slide
  (ΔCE vs NMI, α_a = 0 as the uncontrolled baseline for the escalation rule),
  a **small-MLP probe cross-check** against the ridge, and the **Φ ablation**
  (Δ held-out recon with vs without Φ) — the test the mask experiment could
  not run, and the one that settles the deferred PCA decision.
- **τ is recorded as a non-axis**: measured face-dominated; κ is the sweep.

### Calibration (§4.6), two rounds — the operating point

Twelve 40-epoch fits at κ = 0.1 (`experiments/calibration.json`,
`calibration_round2.json`). Uncontrolled baseline: MLP-probe ΔCE 0.335,
NMI 0.612, noise floor ≈ −0.055.

| decided | value | the evidence |
|---|---|---|
| α_a | **0.03** | ΔCE 61% → **28%** → 6% of uncontrolled at 0.02/0.03/0.04, with the NMI cliff (0.556 → 0.347) between 0.03 and 0.04 — the spec's crossing, found |
| α_w | **0.1** | the M3 knife-edge on real tissue: 0.03 gives w capacity (KL 0.044/dim) and NMI 0.33 — identity theft; 0.2 kills w with no NMI gain. 0.1 is best NMI and lowest viable leak; w near-pinned there (KL ≈ 0.002/dim) — anomaly scores will read conservative |
| ω | **1** | 0.5 loses NMI everywhere; 2.0 buys recon (−7.32) at more leak (ΔCE 0.161) and lower NMI |
| Φ | **keep, full-dim** | ablation: +0.0092 per-count nats of held-out reconstruction (~1.6 nats/cell) — the value the mask experiment could not measure |

Two findings beyond the numbers: the **MLP probe reads ~2.5–3× the ridge**
consistently — the architect's cross-check was necessary, and the escalation
rule is judged on MLP numbers from now on; and the within-type mirror sits at
0.10–0.17 across every configuration — **no mirror attractor**, and the old
pooled readings (~0.5–0.8) were type-confounding exactly as §7.10's patch
anticipated.

**Open judgment**: at the operating point the converged... rather, the
40-epoch ΔCE is 28% of uncontrolled against the ~20% escalation rule —
marginal, and from short fits. A single convergence-length run at the
operating point is deciding it before the sweep commits 18 runs either way.

### The escalation fired — the adversary is in

The convergence-length check at the operating point split the leak cleanly:
**ridge ΔCE 0.026** (linear dependence essentially gone — the log-det penalty
did the job it can do) against **MLP ΔCE 0.153** (NMI intact at 0.573). What
remains is *nonlinear* dependence, which a covariance penalty cannot see even
in principle — the precise case §4.6's adversary exists for, and the precise
thing the architect's MLP cross-check was inserted to catch. Trigger
documented in the registers; trigger met; component built:

- `soft_clusters` (`e_Φ`, E_Φ = K, soft memberships, fit-once-freeze) and
  `Φ̄(t)` in `ModelData`;
- `Adversary` heads `(z, t) → ŷ, êΦ` — deliberately nonlinear MLPs — on a
  separate optimiser; `adversary_terms` carries both directions of the minimax
  and logs the two excess halves separately, as the spec asks;
- trainer mode `--invariance adversary`: model step against frozen heads,
  then `adv_steps` head updates on detached z;
- six tests: gradient isolation both ways, and zero-at-optimum shown on
  synthetic (excess < 0.03 with nothing to find, > 0.15 with a planted leak).

The probe stack stays independent of the adversary (fresh ridge + fresh MLP at
evaluation), so the invariance check is never graded by its own enforcer.
Adversarial α_a calibration at convergence length is running; the sweep waits
on its operating point.

### Adversarial calibration, rounds 3–4 — and the sweep launch

The first adversary (`adv_steps=2`) was a placebo: the encoder defeated the
stale heads while the fresh probe still saw 58–99% of the leak (issues A2).
With `adv_steps=6`, lr 2e-3, the picture inverted completely:

| adversary α_a | MLP ΔCE | % of uncontrolled | NMI |
|---|---|---|---|
| 0.3 | 0.046 | **14%** | 0.641 |
| 1.0 | −0.018 | at the floor | 0.650 |

Converged uncontrolled baseline: MLP ΔCE 0.333, NMI 0.623. The invariance is
achieved at **zero NMI cost** — both adversarial runs sit *above* the
uncontrolled NMI — and ~0.02 nats of reconstruction. Operating point α_a = 0.3
(clears the rule with ΔCE still a hair positive; 1.0 over-scrubs below zero
for no NMI gain). The general lesson is A2: adversary strength is graded only
by the independent probe, never by its own training loss.

**The κ sweep is live**: 6 κ × 3 seeds, adversarial invariance, both 4090s
(κ ≤ 0.1 on GPU 0, κ ≥ 0.2 on GPU 1), early stopping active, per-eval
`history.jsonl` + TensorBoard (now with UMAP figures: spatial z and w grids,
type-coloured UMAPs for both, and within-type UMAPs coloured by dominant
neighbour type — z should mix, w should organise). Report lands in
`experiments/kappa_sweep.json` via `sweep.py --report-only`.

### Reproduction card — every tuning run, for the article's methods section

Fixed throughout: bundle `full` (407,120 cells × 5,101 genes), `Φ` =
`egomask_ego_v1` (KRONOS v1, 256 px @ 0.5 µm/px, 25 µm ego disk zeroed,
full 384-d), graph = voronoi pruned at 40 µm, τ = 20 µm, tiles 4,096 cells,
split seed 0 (identical split in every run, guaranteed by per-purpose RNG
streams), `d_z = 20`, `d_w = 6`, hidden 256, GAT 32×4 heads, lr 1e-3 cosine,
grad-clip 10, `α_z = 0.007`. All artefacts under
`data/datasets/xenium_prime_ovarian_cancer_ffpe/experiments/`.

| round | what varied | fixed at | length | artefact |
|---|---|---|---|---|
| 1 | `α_a` ∈ {0, .01, .02, .05, .1} closed-form; + Φ ablation (Φ vs zeros at α_a=.02) | κ=.1, α_w=.1, ω=1 | 40 ep, eval@40 | `calibration.json` |
| 2 | `α_a` ∈ {.03, .04}; `α_w` ∈ {.03, .06, .2} at α_a=.03; `ω` ∈ {.5, 2} | κ=.1 | 40 ep | `calibration_round2.json` |
| conv | single fits at α_a=.03 closed-form and α_a=0 | κ=.1, α_w=.1, ω=1 | 200 ep cap, eval@5, patience 20 (early-stopped) | `convergence_check.json`, `convergence_uncontrolled.json` |
| 3 | adversary `α_a` ∈ {.02, .05, .1}, `adv_steps=2`, adv-lr 1e-3 | κ=.1, α_w=.1 | conv-length | `calibration_round3_adversary.json` |
| 4 | adversary `α_a` ∈ {.3, 1.0}, `adv_steps=6`, adv-lr 2e-3 | κ=.1, α_w=.1 | conv-length | `calibration_round4_adversary.json` |

Round 1 is `python -m discell.model.calibrate` verbatim; rounds 2–4 are the
same `_short_fit` machinery with the parameter deltas above (drivers were
throwaway; the JSONs + this table are the record). Probes throughout: fresh
ridge + fresh 64×64-MLP on `(μ_z, onehot t) → [y′, PCs₁₂(Φ)]`, spatial
train/val at tile level, ΔCE against the per-type mean baseline; noise floor
from within-type permutation of z.

**The sweep** (18 runs, 2026-09-01):

```
python -m discell.model.sweep --dataset xenium_prime_ovarian_cancer_ffpe \
  --embeddings egomask_ego_v1 --invariance adversary --alpha-a 0.3 \
  --adv-steps 6 --adv-lr 2e-3 --alpha-w 0.1 --alpha-z 0.007 --omega 1.0 \
  --epochs 200 --kappas <0 0.05 0.1 | 0.2 0.3 0.4>   # split over two GPUs
```

Seeds {0, 1, 2} per κ; per-run record = `runs/sweep_k*_s*/`
(`config.json` incl. git hash, `history.jsonl` per evaluation, `best.pt`
weights + adversary-era covariance state, TensorBoard events); cross-run
report = `experiments/kappa_sweep.json`.

---

## 2026-09-01 — the κ sweep: first read of the deliverable

18 runs complete (6 κ × 3 seeds, adversarial invariance, early-stopped at
~40–90 epochs each; `experiments/kappa_sweep.json`, per-run `history.jsonl`).
Read per §4.7: stable across the grid → finding; vanishing or sign-flipping →
not separable from leakage.

**1. Type-level spatial responsiveness survives the whole sweep.** Mean ‖w‖
per type keeps one ordering at every κ, with seed-envelopes attached:
VEGFA+ Tumor (10.5 ± 0.6) > Proliferative Tumor (9.4) > Inflammatory Tumor
(8.7) > Tumor (7.6) > … > Tumor Associated Fibroblasts (4.8). The tumour
subpopulations are the most niche-responsive cells on the slide and the
ranking is not explainable by leakage at any swept level. Magnitudes shrink
mildly at κ = 0.4 (VEGFA+ 10.5 → 9.0) as the leak term absorbs variance —
expected, ranking intact. *A finding at type granularity.*

**2. z is κ-invariant.** NMI 0.62–0.68 across every (κ, seed) — identity
separation owes nothing to the leakage setting.

**3. B: stable along κ, moderate across seeds — the architect's joint
envelope was the right call.** Within a seed lineage the programme space
tracks the anchor at 0.84–0.88 matched correlation across the entire grid;
across seeds it reproduces at 0.65–0.73 for most κ, with two outlier cells
(κ=0.2 s2 at 0.43, κ=0.4 s2 at 0.44) dragging those κ's minima. Seed
variation > κ variation. Gene-level programme claims therefore need consensus
over seeds (e.g. matched-average B), not a single run; the two outlier runs
are flagged, not averaged away.

**4. The predicted partial-isolation artefact exists and is small.** Cells
with post-prune degree ≤ 2 pay ~0.12 nats of held-out reconstruction from
κ = 0 to 0.4, against ~0.03 for well-connected cells — full κ against a thin
ρ̄, as accepted at the design stage. Effects must not be read off those cells;
the strata are in the report for exactly that.

**5. No likelihood cliff anywhere on the grid** (−7.25 → −7.29 monotone in
κ): the data does not identify κ, which is the premise of sweeping it rather
than fitting it.

Caveats that stay attached to all of the above: one slide, one label set,
early-stopped runs, and ‖w‖ is a magnitude summary — gene-level effects are
the next analysis, from the consensus-B route in (3).

*(Infra: all UMAP figures gained PCA twins on shared subsamples; sweep-run
TensorBoards predate the twins by one commit.)*

### Post-sweep decisions and the evaluation upgrades

**The working point, given the sweep**: reference model at **κ = 0.1**
(inside the published Xenium range, effects stable there, clear of the two
seed-outlier cells), always quoted with the sweep envelope; gene-level claims
via consensus-B over seeds; degree ≤ 2 cells excluded from effect readouts;
calibrated αs unchanged — the sweep confirmed the design, it did not amend it.

**Evaluation figures extended** (all in TensorBoard per run):
per-type ‖w‖ **box plots ordered by mean** — the sweep's headline ranking,
now visible live per run; a **signed-log w** UMAP (`sign(w)·log1p|w|`; a raw
log is undefined on a signed latent); and **w pseudotime**: a principal curve
(Hastie–Stuetzle by moving average, `principal_curve` in `metrics`, recovery-
tested at ρ > 0.95 on a synthetic arc) fitted to the w-UMAP "snake", drawn on
the UMAP and rendered as a colour gradient in tissue coordinates — globally
and within each canonical type. Curve direction is arbitrary and the figures
say so. A shared UMAP cache keeps a figure event at roughly the previous cost
despite tripling its content.

**Reference run launched**: `runs/reference_k0.1/`, calibrated operating
point, full new figure suite, seed 0.

### Pseudotime PCA twin, and the cell-cycle disentanglement read

`figures/w_trajectories_{umap,pca}`: the principal-curve pseudotime now runs
on both projections (same members, same curve fitter) — the PCA snake may
parameterise more cleanly than UMAP's.

**Cell cycle, per the architect's recipe**: Tirosh scoring
(`cc.genes.updated.2019`, scanpy `score_genes_cell_cycle`) computed once at
assembly from the raw counts on a scratch AnnData; types ranked by MKI67⁺
fraction; both cautions encoded rather than remembered — hard `phase`
defaults to G1 at Xenium depth so quantitative reads use the **continuous
S/G2M scores**, and analyses restrict to cycling types. Two consumers:

- `figures/z_cell_cycle`: z UMAP+PCA within the top cycling types, coloured
  by phase (the picture);
- `val/cycle_r2_{z,w}` every evaluation (the test): within-type ridge R² of
  the continuous scores from each latent against a within-type-permuted
  control, per-type means removed so identity itself carries nothing.
  Expected signature — z well above control, w at it; w predicting cycle
  would mean identity leaking into the context channel. The circularity
  (scores come from the same x that z encodes) is the point, not a flaw: the
  question is whether z *retains* that axis. Unit-tested (carrying latent
  R² > 0.5, blind latent ≈ 0).

Reference run relaunched with the full suite; the superseded first run (old
figure set, best recon −7.2622 / NMI 0.656 at epoch 59) was deleted.

### Figure-suite reorganisation (and a silent-patch lesson)

Three changes, user-directed: **(1)** the PCA trajectory was genuinely broken —
an earlier multi-line string patch had failed to match, and `.replace` fails
*silently*, so `w_trajectories_pca` was UMAP coordinates under a PCA title.
The figure code was rewritten wholesale and every claim grep-verified
(process note, twice earned this session: verify that a text patch landed,
and never `pkill`/`kill` by a pattern your own command line contains).
**(2)** Figures are now organised **per variable with projections as
columns** — `figures/{z,w,logw}_by_type` are single figures with UMAP | PCA
panels; `figures/per_type_{z,w}` put both projections side by side within
each canonical type. **(3)** The cell-cycle panels cover the union of cycling
and canonical types (≤6), phase-coloured, both projections; the cycle *probe*
now also excludes `Unassigned`, aligning it with the figure.

Reference relaunched as **`reference_k0.1_v2`** with the corrected suite. The
superseded launch had already delivered the first live disentanglement read:
cycle-R² **z +0.32 / w +0.02** against ~0 controls — the designed z/w split
on real tissue.

### Evaluation round: richer w-spatial, more panel types, symmetric controls

- **`figures/w_spatial` enriched** to match z's information density: per-dim
  posterior mean, per-dim **deviation from the prior** `μ_w − m_ψ(c,t)` (what
  this cell did beyond its niche's expectation), `‖w‖`, and the per-cell
  **KL(q(w)‖p) anomaly** — the spec's §4.3 anomaly score, now rendered in
  tissue coordinates every figure event.
- **Per-type panels widened** from 4 to 8 types (`--panel-types`, canonical
  first then abundance fill) — applies to `per_type_{z,w}`, the cycle panels
  and the trajectories.
- **`val/cycle_r2_w_permuted`** added; both latents now carry their own
  permuted control in TB/history (z-side scalars existed already).

Reference relaunched as **`reference_k0.1_v3`** (v2 had early-stopped and is
kept for comparison).

### Mid-turn additions: per-type cycle R², merged panel figures

- **Cycle R² restructured**: one shared within-type ridge fit, evaluated
  per type — TB carries the **mean over types** as the headline
  (`val/cycle_r2_{z,w}`), the pooled value, both permuted controls, and a
  per-type breakdown (`val/cycle_r2_{z,w}_types/<name>`). History/metrics
  JSON carry the full dicts.
- **`figures/{z,w}_panels`**: the per-variable figures now have rows =
  *all cells* (coloured by type) followed by each of the 8 panel types
  (coloured by dominant neighbour), columns = UMAP | PCA — and the w figure
  carries the **log-w UMAP as a third column**, replacing the separate logw
  figure. The old split global/per-type figures are gone.

Run **`reference_k0.1_v3`** was superseded pre-figures by these arrivals;
**`reference_k0.1_v4`** carries the complete set: enriched w-spatial
(deviation, ‖w‖, KL anomaly), 8 panel types, dual-projection trajectories,
cycle panels + per-type R², symmetric permuted controls.

### v5: the cell-cycle ceiling, reliability, and the right instrument

Architect's diagnosis, encoded: UMAP/PCA display *dominant variance*, not
*retained information* — z's top axes must be type geometry (the decoder has
no `t`), cycle is a low-variance within-type axis, and the invariance penalty
makes z blobby-by-construction within type. A 0.3-R² axis will not paint a
global UMAP; that is the instrument's failure, not the model's. Three
additions for v5:

1. **The ceiling** (`val/cycle_r2_ceiling` + per-type): the identical ridge
   probe run from 50 PCs of log-normalised counts. Decision rule attached:
   z ≈ ceiling → z retains what is measurable, no model change; z ≪ ceiling →
   rate–distortion is pruning the axis, lever is `α_z` down a notch (and check
   held-out recon on the cycle genes specifically).
2. **Split-half score reliability** at assembly (each marker list halved,
   scored twice, correlated) — the ceiling on the ceiling; printed on the
   projection figure.
3. **`figures/z_cycle_projection`**: per proliferative type — z projected
   onto the probe's own directions (`z·β_S` vs `z·β_G2M`; expected G1 blob at
   the origin, arc through S into G2M, phase-coloured) beside the within-type
   z-UMAP coloured by **kNN-smoothed** S and G2M scores (per-cell scores are
   mostly noise at this depth).

Run `reference_k0.1_v5` carries everything from v4 (which was superseded
pre-figures) plus the above.

### v6: full-space pseudotime, z trajectories — and the w-UMAP explained

**Pseudotime was being fitted in the 2-D embedding** (the user asked; the
honest answer was no). Now the principal curve is fitted in the **full latent
space** (a hardcoded 2-D assumption in the smoother surfaced and was fixed,
with a 6-D recovery test added) and rendered per projection by mapping curve
points through their nearest cells — so the UMAP and PCA figures share one
pseudotime and differ only in layout. `z` gets the same treatment
(`figures/{z,w}_trajectories_{umap,pca}`).

**Why the w-UMAP looks like a snake** — inspection of `reference_k0.1_v5`'s
posterior (`experiments`-grade numbers, scratch script):

| quantity | value |
|---|---|
| R²(μ_w ~ m_ψ) | **0.9998** (per-dim corr ≥ 0.9997) |
| per-cell deviation / total std | 1–2.4% per dim |
| w variance in PC1 / PC2 / PC3 | **85.4% / 14.6% / 0.01%** (participation ratio 1.33) |
| R²(μ_w ~ type alone) | 0.77 |
| R²(μ_w ~ type + y + Φ-PCs) | 0.97 |

So at the calibrated α_w = 0.1, **w *is* the prior**: an effectively
~1.3-dimensional deterministic function of (type, niche). The UMAP of a
near-1-D continuum is a filament — the "snake" is the niche manifold seen
through m_ψ, not an artefact. Consequences, all consistent with the standing
register entries: w-pseudotime reads as *ordering along the dominant niche
axis*; the per-cell anomaly is small by construction; and only ~2 of 6 w
dimensions do real work. The lever, if per-cell response is ever needed, is
α_w down toward the knife-edge at a known NMI price — not taken without cause.

### sweep2: the κ sweep rerun under the full quality battery

The original 18 runs predate the adversarial-calibration-era metrics; sweep2
re-runs the identical grid (6 κ × 3 seeds, same operating point, same shared
split) with everything the evaluation now measures, so quality is quantified
per run rather than inferred: **cycle R² for z / w / the x-PC ceiling**
(mean-over-types + per-type), within-type mirror R² ± control, probe ΔCE ±
noise floor, per-eval `history.jsonl`, full-space pseudotime and the rest of
the figure suite (at a sparser cadence — metrics log every eval; figures ×18
runs are the expensive part). Mechanics: `--tag sweep2` so **no original
sweep run is touched or overwritten**; the cross-run report gains the
quality columns and lands in `experiments/kappa_sweep_sweep2.json`. Fleets:
κ ≥ 0.2 on GPU 1 immediately, κ ≤ 0.1 queued on GPU 0 behind the finishing
v6 reference; combined report fires automatically when both exit.

What the battery adds to the sweep's interpretive power: the cycle triplet
tells us at every κ whether z keeps its intrinsic axis (z vs ceiling) and
whether κ pushes identity into the context channel (w vs its control) —
a per-κ disentanglement audit the first sweep could not express.

### What the z-pseudotime captures — the axes, named by their genes

Method: within each type, Spearman-correlate the full-space z-pseudotime with
every gene expressed in ≥5% of that type's cells (log-normalised), plus
decompositions against depth, cell geometry, niche composition, Φ-PCs, cycle
scores, and neighbour-smoothness. Run on `reference_k0.1_v6`; scratch script,
numbers in the log. Two global facts first: **depth R² ≤ 0.04 everywhere**
(the §4.2 normalisation did its job) and **niche-composition R² ≤ 0.11
everywhere** — the dominant intrinsic axis is not a niche readout, i.e. the
invariance holds along the pseudotime too.

| type | the axis, by its genes | spatial ρ |
|---|---|---|
| Proliferative + plain Tumor Cells | one shared **tumour-state continuum**: PVT1 / SNHG15 / SOX2-OT / PABPC1L / PLXNB1 (+) vs H19 / LDHA (−) — a lncRNA-stemness programme against a hypoxic-glycolytic pole. The slide's own label set contains "SOX2-OT+ Tumor Cells" as a *discrete type*; z's leading axis is the **continuum that label discretises** — spec §7.1's prediction, verbatim | 0.20–0.27 |
| Inflammatory Tumor Cells | an **interferon-response gradient**: GBP1, CXCL10, IFIT1/2/3, MX1, RSAD2, TNFSF10 moving together — and strongly spatially organised (ρ 0.65, Φ-R² 0.19). Cycle co-varies along it (ρ ≈ 0.3), which is why this type was the pseudotime-cycle exception | 0.65 |
| Smooth Muscle | the classic **contractile ↔ synthetic** phenotype axis: CNN1 (+) vs TIMP3, NOTCH3, HEYL, EPAS1 (−) | 0.34 |
| Macrophages | a **matrix-gene axis** (DCN, COL5A1, BGN, LUM, POSTN, CTHRC1) — either fibrosis-associated polarisation or **residual spillover from fibroblast neighbours**; flagged, and checkable: if it is leakage, its strength should fall along the κ grid in sweep2 | 0.16 |

So the pseudotime is not noise and not the cycle: it finds a nameable
intrinsic state continuum per type — several straight from §7.1's list of
what z should hold beyond the label. Caveats attached: correlational, one
run, one slide; the macrophage axis carries the spillover hypothesis until
sweep2 says otherwise.

### sweep2 results — the full battery over 6 κ × 3 seeds (2026-09-08)

All 18 runs finished 2026-09-01; the auto-fired report had aggregated only
the GPU-0 half (κ ≤ 0.1), so the report was regenerated over the full grid
(`--report-only`, same tag) — `experiments/kappa_sweep_sweep2.json` now holds
all 18. Reads, in order of what is new:

1. **Reconstruction now argues for small κ.** Held-out per-count recon is
   flat through κ ≤ 0.1 (−7.2531 / −7.2534 / −7.2540) then falls monotonely:
   −7.2599 (0.2), −7.2708 (0.3), −7.2829 (0.4). The 0.1 → 0.4 drop (~0.030
   nats) is ~6× the within-κ seed spread (~0.005). sweep1 saw "no cliff";
   sweep2 (adversarial operating point) resolves a gentle plateau-then-slope.
   Still no likelihood *identification* of κ inside the plateau, as designed.
2. **The mirror metric is the first with a clean monotone κ response**:
   within-type mirror R² 0.067 → 0.062 → 0.060 → 0.052 → 0.046 → 0.041
   along the grid, control flat at ~0.016. Excess over control halves from
   κ = 0 to κ = 0.4 — the explicit leak channel absorbs exactly the
   neighbour-explaining duty that otherwise feeds the mirror attractor.
3. **Cycle triplet, the robust (pooled) read**: z 0.42–0.46 at *every* κ,
   w ≤ 0.024 ≈ 0, within-type-permuted ≈ 0, 50-PC expression reference
   0.216. Cycle is in z and absent from w at every κ and seed. Two lessons
   about the metric itself:
   - the "ceiling" is not a ceiling — z beats the 50-PC ridge ~2×, i.e. a
     linear read from 50 linear PCs underfits; treat it as an
     *expression-PC reference line*, not a bound;
   - `r2_mean_types` is noisy (−0.08…+0.13 across runs) because the MKI67⁺
     ranking pulls two essentially non-cycling types into the top-4:
     Pericytes (held-out R² −0.5…−1.2) and Ciliated Epithelial (~−0.10) —
     while the real cycling types are rock-stable (Proliferative 0.45–0.51,
     Inflammatory 0.44–0.48 in every run). MKI67⁺ pericytes in a tumour
     slide are plausibly *themselves spillover*. Headline read = pooled;
     mean-of-types kept for the per-type split (issues M6).
4. **Invariance holds at every κ**: ridge probe ΔCE −0.03…+0.02 across all
   18 runs (noise floor −0.05…−0.06) — no linear leak anywhere; NMI flat
   0.62–0.67 with no κ trend. The adversary neither weakens at high κ nor
   costs identity.
5. **B seed-stability falls with κ**: across-seed matched |corr| mean
   0.62–0.64 for κ ≤ 0.1 vs 0.48–0.55 for κ ≥ 0.2 (min 0.41). Along-κ
   within seed stays high (s0: 0.83–0.93). Higher κ costs not just
   likelihood but programme-space reproducibility. Note the adversarial
   operating point is overall less B-stable across seeds than sweep1's
   closed-form runs — consensus-B over seeds remains the reporting rule.
6. **The headline biology replicates**: per-type ‖w‖ ranking is unchanged
   from sweep1 and stable over all 18 runs — VEGFA+ 10.1 > Proliferative
   9.2 > Inflammatory 8.6 > Tumor 7.5 > … > Malignant-Cyst 2.3 >
   Fallopian-Tube 2.0 (means; orderings agree run by run at the top/bottom).
7. **The κ cost concentrates in low-degree cells**, as the watch-list item
   predicted: from κ = 0 to 0.4 recon falls 0.030 globally but 0.065 in
   lost-1+ cells and 0.098 in degree ≤ 2 cells (~2–3× the global cost);
   the lost0-vs-lost1+ gap shrinks 0.217 → 0.180. Direction consistent with
   full-κ-against-thin-ρ̄; magnitude small; keep excluding degree ≤ 2 cells
   from effect readouts.

**Verdict**: κ = 0.1 stays the operating point, now *supported* rather than
default — it sits at the end of the recon plateau, in the most B-stable
region, with mirror already 10% below the κ = 0 level. Pushing κ ≥ 0.2 buys
further mirror reduction at a monotone price in likelihood, seed stability,
and low-degree cells. Open: the macrophage matrix-axis spillover prediction
("strength should fall along κ") is not answerable from the battery — it
needs the pseudotime–gene-correlation analysis re-run per κ; deferred.

### The operating point becomes the default; `reference_best` + per-run reports

**Defaults changed** (`TrainConfig`, and the sweep CLI to match): κ = 0.1,
invariance = adversary, α_a = 0.3, adv_steps = 6, adv_lr = 2e-3 — exactly the
calibrated point every reference run and sweep2 used explicitly; a bare
`python -m discell.model.train --dataset <id>` now reproduces it. The
closed-form path stays selectable (`--invariance closed_form`, where the
straight-through-equivalent α_a is ~0.02, still noted in the config comment).
The synthetic train-smoke fixture pins `closed_form` (synthetic data carries
no eΦ). Suite: 116 passed, 1 skipped.

**`reference_best`**: one run at the defaults with the budget widened —
`--epochs 500 --patience 40` (stop after 40 improvement-free epochs instead
of 20), seed 0, eval every 5, figures every 25. Purpose: convergence
certainty for the reportable run; the config is otherwise byte-identical to
`reference_k0.1_v6`.

**`discell/model/report.py`** — one-command per-run report
(`python -m discell.model.report --dataset <id> --run <name>` →
`runs/<name>/report/report.md` + figures). Contents: TB figures extracted at
their final step; a six-panel training-progress figure with a reading guide
(recon, NMI, probe ΔCE vs floor, mirror vs control, cycle pooled z/w/reference,
KLs) and the best-checkpoint epoch marked; the **disentanglement quadrant**
computed fresh from `best.pt` on the validation split — z passes cycle / w
fails it (same in-probe construction on both latents, side by side), w passes
the tissue-gradient read / z fails it; the per-type ‖w‖ boxplot with the
biological reading and the w≈prior caveat; B loadings; provenance block.

**Methodological catch while validating on v6**: the *raw* niche-R² of the
z-pseudotime is 0.55 — not leak, but type read through homophily (z carries
type by design; type is spatially predictable, ego-masking AUC 0.89). After
partialling per-type means from both sides (the mirror_r2 pattern) the
contrast is clean and stark: **z-pseudotime niche R² 0.01 / neighbour
coherence 0.10 vs w-pseudotime 0.53 / 0.63** (v6 numbers). Any spatialness
claim about a type-carrying latent must be type-partialled or it measures
homophily. The report tables the partialled numbers and shows raw in context.

**`reference_best` outcome**: best epoch 59, trained to 99 (the full 40-epoch
grace exhausted with no further improvement — convergence confirmed, not
truncated), 17.8 min. Best recon −7.2586 / NMI 0.658, inside the sweep2
κ = 0.1 envelope. The quadrant replicates v6 on this independent fit:
z cycle 0.47 / w 0.03 pooled; type-partialled pseudotime niche R² z 0.01 vs
w 0.53, coherence z 0.15 vs w 0.64. Report:
`runs/reference_best/report/report.md`.

### Doc-08 latent-validation analyses — `discell/model/validate.py`

Implements the handover doc (08-validation-analyses_1.md): every analysis
probes a known-allegiance target from BOTH latents against floor
(within-type permutation), ℓ-baseline (log-depth-only probe — retrofitted
into the training-time cycle probe too, logged as `cycle/lbaseline`), and
the 50-PC ceiling where expression-derived; within type; **spatial-block CV**
with whole tiles as fold blocks (tile k → fold k mod 5). One CLI:
`python -m discell.model.validate --dataset <id> --run <name>` (headline) or
`--sweep-tag sweep2` (κ companion). Outputs in `runs/<name>/validation/`.
7 known-answer tests (planted allegiances) in `tests/test_model_validate.py`;
they caught two instrument bugs before any real data was touched (binary-AUC
form; probability rows not summing to 1 when a contiguous niche hides from a
fold's training half).

**Two instrument corrections after the first real-data pass, registered:**

1. *Landmark banding*: the doc's letter ("fit on d ≤ 500; report held-out R²
   per band") scored a pooled global fit against band-local variance — every
   probe went negative with the floor at zero, and more signal meant more
   negative (w −0.76 < z −0.25 < ℓ −0.16 < floor 0 mid-band). Deviation: **fit
   within each band** (block-CV inside the band), which is the question the
   mid band asks — "does the latent order cells by distance where one hop
   cannot see the landmark". θ_L now comes from the mid-band fit.
2. *Gene signatures*: B rows of never-expressed control/viral/mutant probes
   are gradient-unconstrained noise and dominated `B·θ̂` (SRY/HPV16 in an
   ovarian slide as "top genes"). Masked to genes expressed in ≥1% of cells.

Also: interface mode-filter raised 3 → 10 iterations (3 left 58% of the slide
"boundary" — the slide is deeply infiltrative), and the §3.2 hot-z-dim triage
(y-R² = leakage read, cycle-corr = benign read) added to the Moran output.

**Headline results (`reference_best`)**:

- **Moran's I**: mean |I| **w 0.545 vs z 0.082** (6.6×), every w dim hot
  (0.40–0.64), no collapsed dims (the α_w concern resolved by the variance
  check: μ_w varies through the context prior even at KL_w ≈ 0). All 20 z
  dims clear the razor-thin 400k-cell null (0.02–0.17), but the triage says
  benign: hottest dims regress on y at ≤ 3.6% R² — intrinsic-spatial
  biology, not leakage.
- **Niche invariance** (K = 10; robust at 6/15): macro-AUC **w 0.76**,
  **z 0.65**, ℓ-baseline 0.59, floor 0.50 (14 eligible types). The
  asymmetry is there (w−ℓ = 0.18 vs z−ℓ = 0.06) but z is *not* fully at
  the ℓ-baseline — a genuine residual, consistent with the Moran triage's
  benign intrinsic-spatial structure, worth watching (issues list).
- **Landmarks** (mid band = the evidence): weak but correctly-ordered —
  w 0.03–0.06 vs z ≈ 0 (ordering w > ℓ ≈ floor in 3 of 4 classes);
  interferon programme (MX1, IFIT1/2/3) recurs along the distance axes,
  matching the known interferon niche gradient. θ cosines mixed (−0.75 to
  0.46), cross-type consistency low (0.05–0.38) — a multi-axis spatial code,
  though θ from R² ~ 0.05 fits is high-variance (caveat attached).
  Inventory: smooth muscle (244 instances), stromal/tumor endothelial
  (92/19), tumor-stroma interface (10-iter mode filter).
- **Allegiance matrix**: near-block-diagonal — S/G2M z 0.45 vs w 0.01
  (ceiling 0.21); niche w 0.76 vs z 0.65 (the one soft cell); mid-band
  distance w 0.03 vs z −0.04; pseudotime tissue-gradient w 0.53 vs z 0.01.

κ companion over the 18 sweep2 runs: launched (capped: 10k cells/type,
200 perms, K=10 only) → `experiments/validation_sweep2.{json,png}`.

**κ companion results** (`experiments/validation_sweep2.{json,png}`, 18 runs,
capped: 10k cells/type, 200 perms, K=10): **Moran's mean |I| on z falls
monotonically with κ — 0.078 → 0.043 from κ=0 to 0.4 — while w stays flat at
~0.52–0.54.** Same shape as the mirror-R² trend: the explicit leak channel
absorbs spatial autocorrelation that otherwise lodges in z; Moran-I_z is the
second metric with a clean monotone κ response, and the cheapest. Niche AUC_z
drifts the same way (0.642 → 0.619 over the grid, ℓ-baseline flat at 0.580 —
z's residual above depth shrinks from +0.062 to +0.039) with AUC_w flat at
0.73–0.75. Landmark mid-band R² is flat/noisy in both latents (z ≈ −0.03,
w 0.02–0.04, no κ trend) — consistent with the inventory diagnosis (the map:
smooth-muscle sheets, starved endothelial coverage, degenerate interface at
84% near-share; landmark redefinition pending a decision). The headline
`reference_best` numbers sit inside all three seed envelopes.

### Motivations register for the doc-08 experiments, and the landmark rewrite

Written down before the rewrite, so the reasoning is on record independent of
how the rewrite turns out.

**Why each validation experiment exists.** The §4.6 training probe is the
model grading its own homework: it asks whether z predicts the very block
(v = [y', Φ-PCs]) the objective was told to scrub, with the machinery chosen
at training time. Doc-08's premise is different — take targets whose
allegiance is known *a priori*, probe them from **both** latents, and let the
asymmetry carry the claim, certified by references (floor, ℓ-baseline,
ceiling):

- **Moran's I** — no annotation, no probe family, no target choice at all:
  just spatial autocorrelation of each latent dimension on the model's own
  graph. w is the context channel, so its dims must be spatially organised;
  z should be near-null except for honest intrinsic-spatial biology, which
  the §3.2 triage (y-R² = leakage, cycle-corr = benign) separates. Cheapest
  per-κ diagnostic — and it delivered: I_z falls 0.078 → 0.043 along κ.
- **Niche invariance** — the *exclusion* direction. Cycle retention alone
  could be satisfied by a z that also memorises the niche; z must also FAIL
  to predict data-defined niche labels (k-means on y — never on latents,
  that would be circular). The ℓ-baseline is mandatory because depth
  co-varies with niche.
- **Distance-to-landmark** — the only **geometry-derived** target: zero
  circularity with the counts, which neither Moran's (latents themselves)
  nor niche (labels from composition) can claim. The mid band (50–300 µm) is
  the thesis itself: one graph hop cannot see the landmark there, so
  predicting distance must travel through induced expression response.
- **Allegiance matrix** — the compression: every claim in one block-diagonal
  figure, each cold cell certified by its references.

**Why the landmarks are being rewritten** (evidence: `landmark_map.png`,
first-pass results, κ companion): the mid-band signal was weak (w 0.03–0.06)
and κ-flat while every other instrument finds strong w-effects — pointing at
the target's *definition*, not at absent spatial structure. The map confirms
three specific failures:

1. *Smooth muscle is sheets, not structures*: one giant wall instance
   dominates; near-share 23%, beyond-cap 53% — distance-to-a-sheet is a thin
   shell, not a gradient.
2. *The good landmarks are starved*: the endothelial classes are
   morphologically ideal (compact bullseyes) but cover a quarter of the
   slide (92 + 19 instances; 68–75% beyond cap; the tumour mass unsampled).
3. *The interface is degenerate and it is not a smoothing bug*: the mode
   filter CONVERGED at 57.8% boundary — the slide is interleaved at
   single-cell scale, so any 1-hop boundary definition makes near-share 84%
   and the target has no variance.

**The redefinitions, each with its success check** (all data-derived, same
status as y; no pathologist input):

1. **`vasculature`** = both endothelial types merged (+ Pericytes iff the
   map-level co-location test passes: ≥50% of pericytes within 30 µm of an
   endothelial cell — decided by the number, logged). Success: instances up,
   beyond-cap share down, tumour mass sampled; mid-band w > z with floor ≈ 0.
2. **Smooth muscle → compact instances only** (20–500 cells): arteriole
   rings stay, anatomical walls drop. If < 5 instances survive, the class
   drops out honestly.
3. **Interface at tissue scale**: kNN-smoothed tumour fraction (k = 50,
   ≈ 75 µm at this density), compartments = field > 0.5, boundary = graph
   edges crossing compartments *of the smoothed field*. Success: boundary
   share ≪ 57.8% and near-share ≪ 84% in the map.

If the mid band stays flat after honest landmarks, that is a *finding* (the
spatial response does not carry metric distance information beyond one hop),
and the demotion question reopens with evidence instead of suspicion.

**Landmark rewrite outcome** (reference_best, landmarks+matrix rerun; new
`landmark_map.png`): every structural success check passed —

- pericyte co-location measured at **90%** within 30 µm of endothelium →
  included; `vasculature` = 8,387 cells in **206 instances** spread over the
  whole slide incl. the tumour mass (mid-band coverage 7%→**55%**,
  beyond-cap 68–75%→16%);
- smooth muscle (compact, 20–500 cells): 238 instances, the 53k-cell sheets
  gone (map shows rings, not walls); coverage still northern (cap 56%);
- interface via kNN-smoothed tumour fraction (k=50): boundary share
  57.8%→**8.9%**, near-share 84%→28% — the map shows a proper filigree
  tracing the tumour lobules.

The science, with honest targets: **the tumor–stroma interface is the one
landmark w carries — mid-band R² 0.069 vs z −0.005, ℓ −0.005, floor −0.001**
(and it strengthened vs the degenerate first pass). **Metric distance to
vasculature is a clean null**: mid-band w 0.007 with 55% of the slide in
band — either one-hop context genuinely cannot propagate perfusion distance,
or labelled vasculature (endothelium+pericytes) is not the perfusion field
(capillaries are unlabelled on this panel). The old smooth-muscle 0.045 is
exposed as sheet-proximity (regional composition), falling to 0.010 with
compact instances. Caveats: matrix row "mid-band landmark distance" averages
all three classes, diluting the interface signal (0.069→0.029 shown);
vasculature near-band shows z 0.06 > w 0.03 (spillover-adjacent intrinsic
states? watch, near band is sanity-only). θ: vasculature ~ interface cosine
0.52 (vessels live in stroma — expected collinearity, doc §2.5).

**The circularity critique and the y-baseline control** (user-raised: the
landmark sets are type-defined, types are expression-derived — so "ground
truth from geometry, zero circularity" holds for the *ruler*, not the *set*).
Where the risk actually sits: the z-column is structurally protected (fits
are within-type so cell i's own label is constant; enc_z sees no neighbour
labels; interface label-noise would bias z toward looking leaky → its null is
conservative). The exposed cell is the w-positive on the interface, whose
generator IS composition — w's own favourite food. Control added: a
**y-baseline** (probe from raw one-hop composition) in every landmark table.
Measured on reference_best:

- *vasculature*: near-band y-baseline 0.45 (pure composition echo, as §2.4
  predicted) but **mid-band y-baseline −0.004** — the invisibility premise
  holds for compact structures; w's mid 0.007 is therefore a certified
  genuine null of the strong beyond-one-hop claim.
- *interface*: mid-band y-baseline **0.042** vs w **0.069** — roughly 60% of
  the interface signal is definitional (composition reads its own boundary);
  the defensible residual is **w − y ≈ +0.027**, and even that flows through
  channels that include Φ. Reframed accordingly: the interface row shows w's
  compositional nature (legitimate allegiance, weakly beyond raw one-hop
  composition), NOT geometry-pure spatial-effect propagation.
- z: ≈ 0 in every mid band under every reference — unchanged.

Standing conclusion for the article: only annotation-free, image-derived
landmarks (vessel lumens / DAPI voids from morphology) would honour the
"zero circularity" pitch literally — noted as the second-slide upgrade; on
this slide the landmark analysis contributes the vessel null + the
quantified-circularity interface read, with Moran's/niche/cycle carrying the
headline allegiance claims.

### The annotation-free control: `reference_graphclust` (lung rehearsal)

Motivation: the lung Prime dataset ships no curated `cell_groups.csv` (CDN
verified — ovarian's file 206s, lung's 403s/absent), so before annotating
lung we measure what training on unsupervised clusters costs, on the slide
where we can compare. Mechanics: `--label-key` now threads through
`assemble` → `TrainConfig` → run record and both post-hoc reloaders
(report/validate rebuild a run under its own label set). Run: identical
config/budget to `reference_best`, only t changes — 26 graphclust classes
(25 clusters + Unassigned for 513 NaN cells via the P3 fold). Same
trajectory: best epoch 59, stop 99, 18.6 min. Numbers
(`experiments/graphclust_comparison.json`; cross-label reads score BOTH runs
against the CURATED labels):

| read | reference_best | reference_graphclust |
|---|---|---|
| held-out recon | −7.2586 | **−7.2556** (marginally better) |
| NMI(kmeans-18(z), cell_group) | 0.657 | 0.646 (−0.011) |
| probe ΔCE vs curated v-block (floor −0.056) | 0.004 | **0.061** |
| cycle z pooled / its 50-PC frame | 0.475 / 0.216 (×2.19) | 0.323 / 0.116 (×2.79) |
| cycle w pooled | 0.034 | 0.004 |
| Moran mean I (z / w), own centring | 0.082 / 0.545 | 0.050 / 0.339 (6.7×; finer 26-way centring removes more from both) |
| niche AUC (w / z / ℓ / floor), own niches | 0.76/0.65/0.59/0.50 | 0.75/0.64/0.57/0.50 |
| per-curated-type ‖w‖ | — | Spearman 0.58; **top-3 identical** (VEGFA+ > Inflammatory > Proliferative) |

**Verdict**: annotation-free training preserves everything except one thing —
recon equal, z organises the curated biology essentially unchanged, cycle
retention relative to its expression frame preserved (absolute numbers are
frame-dependent, the ratio is the honest cross-conditioning read), the
allegiance battery keeps its shape, the headline ‖w‖ biology survives at the
top. The one principled degradation: **the invariance guarantee is
label-set-relative** — z scrubbed against cluster-niche leaks ΔCE 0.061
against the curated-niche block (vs 0.004 when trained on it); the adversary
covers the composition it was shown, not every composition. Lung
implication: graphclust-first is viable for model development and the
allegiance battery; any invariance claim stated w.r.t. *named* types needs
either a re-run after annotation (~19 min) or the reduced guarantee stated.

### Lung: the second slide (`xenium_prime_human_lung_cancer_ffpe`)

Motivation: generalisation beyond the ovarian slide, and the live rehearsal
of the annotation-free path the graphclust control just validated. 10x ships
no curated `cell_groups.csv` for this dataset (CDN-verified), so t =
graphclust per the control's verdict.

Slide facts and per-dataset decisions (each traced to its rule):

- 278,324 cells × 5,001 genes (base Prime 5K panel — ovarian's extra 100
  custom genes absent). Bundle built with the standard pipeline: voronoi
  816,949 edges, mean degree 5.87, isolated 0.1%.
- **Mask radius stays 25 µm** (E1 rule re-run on lung polygons): covering
  radius p100 = 26.5 µm, exactly 2 cells exceed the disk (ovarian: 6) —
  excluded/zero-filled, Φ stays cross-slide comparable.
- **α_z = 0.004** (§5 rule): lung ℓ̄ = 242 vs ovarian's ~143 → 1/242 ≈
  0.0041. Everything else at the calibrated defaults (portability of α_a,
  α_w, ω across slides is an assumption to check against this run's probe
  and KL_w readouts, not a given).
- Labels: graphclust = 32 clusters + Unassigned (182 NaN cells via the P3
  fold). Cycle instruments fully portable: MKI67 present, 18 S + 34 G2M
  panel hits — same coverage as ovarian.
- Embeddings: `egomask_ego_v1`, identical arm (KRONOS v1, 256 px @
  0.5 µm/px, ego disk 25 µm), running at ~100 cells/s.

Then: train `reference_graphclust` with `--label-key graphclust
--alpha-z 0.004 --epochs 500 --patience 40`, defaults otherwise.

### The α_w study: is w about the cell, or only about its context?

Motivation (user-raised, written before the runs). At the operating point
α_w = 0.1, w is measurably prior-pinned: KL_w ≈ 0.002/dim and
R²(μ_w ~ m_ψ(c,t)) = 0.9998 (v5 inspection). The spec intends w as *the
cell's* spatial response, and w has two information routes: the context
prior m_ψ(c,t) — the population-level response every cell in that context
shares — and the posterior deviation μ_w − m_ψ, the per-cell part read from
the cell's own counts. At 0.1 the second route is essentially closed, so
today w is a context-conditional field *evaluated at* the cell, not a
per-cell measurement; the per-cell anomaly channel is deactivated (already
recorded as an α_w calibration fact). The open question: does lowering α_w
re-open a *usable* per-cell channel, or is everything below the knife-edge
just identity theft (M3: closed-form-era 0.03 → NMI 0.33)? The adversary
(α_a = 0.3) now guards the niche route, which may have moved the edge.

Design: full runs (not short fits — the M3 numbers came from short fits in
the closed-form era) at α_w ∈ {0.02, 0.03, 0.05, 0.07}, seed 0, everything
else the calibrated defaults; `reference_best` is the 0.1 point. Per run,
beyond the standard battery (NMI, probe, mirror, cycle z/w, recon):

- **channel opening**: KL_w per dim; prior-R² = R²(μ_w ~ m_ψ); deviation
  magnitude ‖μ_w − m_ψ‖ distribution;
- **what the freed channel contains** (the deciding reads, all on the
  deviation, not on w): kmeans-NMI(deviation, t) = identity-theft;
  cycle-R²(deviation) = intrinsic-state theft; Δ held-out recon vs 0.1 =
  legitimate per-cell response.

Success criteria, stated in advance: (i) a middle α_w where KL_w opens,
recon improves, NMI holds, and the deviation carries signal that is neither
type nor cycle → candidate re-calibration; or (ii) theft all the way down →
"on this slide at this depth, the per-cell spatial response is not
identifiable; w is a context field" — and 0.1 stands with that caveat made
formal. Either outcome answers the question; only one changes the model.

(Parallel: lung `reference_graphclust` launches on the other GPU —
embeddings landed, α_z = 0.004.)

**Lung `reference_graphclust` results** (first cross-tissue fit; only α_z
recalibrated): best epoch 59 — the third run in a row — recon −7.2577,
NMI 0.653 over 33 clusters, 72 min (GPU contention). **The full allegiance
structure transfers**: cycle pooled z 0.347 vs 50-PC frame 0.166 (×2.1) with
w −0.000 and ℓ-baseline 0.000; probe ΔCE −0.010 (floor −0.021); mirror 0.036
vs control 0.015 (lower than ovarian); Moran mean |I| w 0.428 vs z 0.059
(7.3×, z triage benign: max y-R² 1.9%); niche AUC w 0.731 / z 0.617 /
ℓ 0.544 / floor 0.50 — the same shape as ovarian at every read. KL_w
prior-pinned on lung too (≈0.000–0.002/dim): the α_w question the ovarian
study is probing is tissue-general, not an ovarian artefact. Report +
validation under `runs/reference_graphclust/` on the lung dataset.

**α_w study results** (`experiments/alphaw_study.json`; deviation reads on
the validation split, deviation = μ_w − m_ψ):

| α_w | recon | NMI | KL_w (nats, 6 dims) | prior-R² | dev→type NMI | dev→cycle R² | probe ΔCE | mirror |
|---|---|---|---|---|---|---|---|---|
| 0.10 | −7.2586 | 0.658 | 0.004 | 0.9989 | 0.22 | 0.02 | 0.004 | 0.056 |
| 0.07 | −7.2586 | 0.635 | 0.148 | 0.9980 | 0.23 | 0.01 | 0.006 | 0.053 |
| 0.05 | −7.2534 | 0.638 | 0.059 | 0.9978 | 0.24 | 0.03 | −0.018 | 0.052 |
| 0.03 | −7.2510 | 0.631 | 0.180 | 0.9977 | 0.17 | 0.04 | −0.023 | 0.050 |
| 0.02 | −7.2518 | 0.612 | 0.758 | 0.9960 | 0.16 | 0.06 | −0.025 | 0.048 |

Readings, in order of importance:

1. **The closed-form knife-edge is gone.** M3's "0.03 → NMI 0.33" does not
   reproduce under the adversary: NMI at 0.03 is 0.631. The adversary guards
   the niche route that the old identity theft ran through; the edge has
   moved to ~0.02 (NMI 0.612 and sliding at the guard floor mid-training).
2. **A real per-cell channel opens and pays**: recon improves monotonically
   to 0.03 (−7.2586 → −7.2510, +0.0076 per-count nats — the size of the
   entire Φ ablation), KL_w opens to 0.18 nats at 0.03 and 0.76 at 0.02.
3. **What flows through it is NOT identity**: the deviation's type-NMI
   *falls* as the channel opens (0.22 → 0.16). The ~0.2 floor at closed
   channel is enc_w's t-conditioning patterning the near-zero residual, not
   theft — the trend is the read, and it points away from theft. A small
   intrinsic seepage appears instead (dev→cycle 0.02 → 0.06), the cost side.
4. **Amplitude stays context-dominated everywhere**: prior-R² ≥ 0.996 at
   every α_w — w's bulk is the context field at any setting in this range;
   what opens is a small per-cell *correction*, which is precisely the
   spec's deviation/anomaly reading of w. Prior-R² is therefore the wrong
   sensitivity lens; KL_w is the right one.
5. Invariance and mirror are fine or better everywhere (probe ≤ 0, mirror
   falls slightly); cycle_z stable 0.45–0.49; cycle_w ≤ 0.014 below 0.1.

**Verdict on the question "is w about the cell or the context?"**: at
α_w = 0.1 — the context only, measured. The study lands in pre-registered
outcome (i): α_w ≈ 0.03 re-opens a genuine per-cell channel (0.18 nats,
likelihood-positive, non-type, marginally cycle-tinged) while w's bulk
remains the context field. **Candidate re-calibration α_w = 0.03**, pending
the pre-registered seed check (seeds 1, 2 at 0.03 launched); the KL_w
non-monotonicity at 0.05/0.07 is single-seed noise the check will also
bound. Default unchanged until then.

### Doc-09 execution: gate zero, the module, and the α_w plot twist

**Gate zero passed decisively**: 618 CellChat pairs survive panel ∩ (L+R
present) ∩ (≥20 in-panel NicheNet targets) — 463 secreted, 155
contact-dependent, 282 unique ligands. No fallback needed. Provenance:
`CellChatDB.human.rda` (jinworks/CellChat main, downloaded 2026-09-10) and
NicheNet v2 `ligand_target_matrix_nsga2r_final.rds` (Zenodo
10.5281/zenodo.7074291), both in `data/external/`; new deps `rdata` (R
lists; pyreadr cannot read them) + `pyreadr` (the rds matrix).

**`discell/model/communication.py`** implements §1–§4 + §5c: inventory
(sender types = top-quartile per-type mean ligand rate; receivers ≥2k cells,
all receptor subunits ≥5% positive; ranked by Var(Ẽ)×prevalence, top 20),
exposure (one-hop through the model's own β; mid-band grid-KDE Gaussian
shell 50–150 µm), the load-bearing composition residualisation with the
Var(Ẽ)/Var(E) ≥ 0.1 gate, allegiance rows {w, z, floor, ℓ, y} on block-CV,
program B·θ̂ scored by NicheNet-target AUROC against 50 matched-null ligands,
reattribution on decontaminated log ρ (leak-attributed = keeps <30% of the
naive coefficient; BH-FDR 0.05 naive hits), and the decoy-exposure control.
5 planted-truth instrument tests pass (exposure vs hand computation;
mid-band shell sees 100 µm not 30/250; residualisation strips composition;
depth-controlled association; AUROC ordering).

**α_w seed check, interim**: seeds 1/2 at 0.03 are still training past
epoch 99 with recon −7.19/−7.24 (≫ seed 0's −7.2510 — more than any
legitimate deviation could buy) at NMI ~0.613. Reading: **0.03 is
seed-bistable** — an honest-channel basin (seed 0) and a theft basin
(seeds 1/2). Deviation reads on completion decide; 0.05 becomes the
candidate safe point if confirmed. Doc-09 runs will carry the 0.1-vs-0.03
ablation either way (§3.6), with the bistability reported.

**α_w seed verdict — 0.03 revoked, 0.05 on probation.** The full-table
deviation reads over seeds (`experiments/alphaw_study.json`, rows
0.031/0.032 = seeds 1/2 at 0.03): the anomalous recon (−7.173/−7.233 vs
seed 0's −7.251) is **not deviation-carried** — KL_w stays ≈ 0.17–0.20
nats in every 0.03 run, and a 0.2-nat channel cannot buy ~11 nats/cell of
likelihood. The gain comes from a competing solution family in the
*context* machinery (prior/GAT/B over-explaining expression), and it drains
z: cycle_z falls to 0.445 (s1) and 0.354 (s2) vs 0.47–0.49 in every honest
run, invisible to the NMI guard (type separability holds). Reading: low
α_w doesn't just open the deviation channel, it lets the posterior *lead*
an exploratory drift early in training that the prior then chases —
KL-free once converged, so only the seed ensemble exposes it. Conclusion:
**0.03 is seed-bistable and rejected per the pre-registered criteria; the
candidate moves to 0.05** (recon −7.2534, clean deviation reads, cycle_z
0.454) pending its own seed check (s1/s2 launched). A methods lesson for
the paper: prior-R² cannot certify context-determination when the prior is
learned — the prior chases the posterior, so the certificate is KL_w plus
cross-seed stability, never the R².

**Doc-09 first pass, the 0.1 arm (`reference_best`)** — all 20 top-ranked
pairs end-to-end (`runs/reference_best/communication/communication.json`):

1. **The 0.1 side of the §3.6 ablation is now measured: w cannot see the
   residualised exposure at all** — every allegiance R² (w, z, y) ≈ 0 on
   block-CV. Consistent with everything known about the 0.1 point (w ≈
   1.3-D composition manifold): even "exposure visibility" is absent once
   composition is partialled out. The low-α_w arm is where the channel
   question gets answered.
2. **Programs still separate a candidate set** despite the flat R² (θ is an
   in-sample direction in 6-d program space): 5 pairs clear the matched
   null at ≥ 0.94 percentile — POSTN→ITGAV/B5 (AUROC 0.640, decoy 0.545),
   PDGFB→PDGFRB (0.606/0.479), TNFSF10→TNFRSF10B (0.602/0.544),
   IL6→IL6R/IL6ST (0.575/0.499), CD99 (1.00 percentile but decoy 0.602 —
   **decoy-rejected**, sender-proximity artifact). The decoy control does
   its job, and the survivors are all *secreted* pairs — matching the
   cellAdmix expectation that little survives cleaning, and the range
   logic (contact pairs are the leak-mimic hard case).
3. **Reattribution headline (provisional)**: 2,788 naive exposure-
   associated genes across pairs; **61% leak-attributed**, 30 w-retained,
   38% unexplained (expected at 0.1 — w cannot retain what it cannot see).
   **Caveat, logged before anyone quotes the 61%**: the built-in
   falsification is weak — only 9/28 sender-exclusive artifact candidates
   get leak-attributed, and leak examples skew to low-expression genes, so
   the keeps-<30%-on-log-ρ threshold is uncalibrated for rare genes. The
   §5a planted worlds (world B = leakage only, correct answer nothing) are
   the calibration instrument and gate any quotable number from §4.
Next: 0.05 seed check → operating point → the low-α_w communication arm;
then planted worlds.

**α_w study, final verdict: no safe re-calibration below 0.1 — default
stands.** The 0.05 seed check reproduces the bistability (s1 −7.1857,
s2 −7.2285 vs seed 0 −7.2534), and the full table
(`experiments/alphaw_study.json`, 9 runs) sharpens the mechanism: at
0.05/s1 the implausible likelihood arrives with every existing guard clean
— NMI 0.623, cycle_z 0.478, KL_w 0.027, deviation reads normal. So the
basin is not deviation theft and not z-drain (that was 0.03/s2's variant):
the route is the **w-mirror** — m_ψ(c,t) learning to reconstruct the cell
from its neighbours' μ_z through the GAT, KL-free once the prior converges,
unguarded by NMI (type intact), by the invariance probe (targets [y,Φ]),
and by the z-mirror metric (which watches z, not w). Below α_w = 0.1 the
posterior can afford the early exploratory deviation that discovers this
route; 3 of 6 sub-0.1 seed runs fell in, 0 of 21+ runs at 0.1 ever have.
**Pre-registered outcome (ii), amended**: the per-cell channel cannot be
safely opened by lowering α_w alone under the current architecture; the
honest answer to "is w about the cell?" stays "no — context only", now
with the mechanism that enforces it mapped. Registered in issues as the
w-mirror watch item with the proposed guard (a w-side mirror metric:
held-out R² of μ_w's within-type residual from neighbour z's, to be added
to the evaluation battery before any future α_w attempt; architectural
options — capacity-limiting or input-restricting m_ψ — are spec-change
territory for the architect). Doc-09's low-α_w arm runs on `alphaw_0.05`
seed 0 (clean basin, labelled fragile) purely as the §3.6 ablation.

**Doc-09 §3.6 ablation (0.1 vs 0.05), verdict**: opening the deviation
channel does NOT rescue exposure visibility — mean w-R² on residualised
exposure gains +0.006 (still ≈ 0 on all 20 pairs). The blind spot is
architectural, not α_w-gated: a 6-d, one-hop, composition-dominated context
channel retains no within-composition ligand detail at any tested operating
point. Worse for the low arm: the program signal *degrades* there — mean
target-AUROC 0.541 → 0.514, and all five null-clearing pairs collapse
(POSTN pct 1.00→0.72, PDGFB 0.98→0.16, TNFSF10 0.94→0.46, IL6 0.94→0.40).
The communication readout is best at the honest 0.1 point. Standing doc-09
results: 4 decoy-validated secreted candidates at 0.1 (POSTN→ITGAV/B5,
PDGFB→PDGFRB, TNFSF10→TNFRSF10B, IL6→IL6R/ST), CD99 decoy-rejected, 61%
leak-reattribution (uncalibrated until §5a), and the architectural null.
Consequence for §5a: world A now doubles as the sensitivity calibration —
if a planted response of realistic amplitude is also invisible to the
w-probe, the exposure null is a sensitivity statement, not biology.

**GAT-sources ablation result (`ablation_gat_type_only`, seed 0, defaults
otherwise)**: removing neighbour μ_z from the GAT sources is not merely
cost-free — it is a small improvement on nearly every axis. recon −7.2586 →
**−7.2527** (+0.006, ~⅔ of the Φ ablation's worth), NMI 0.658 → 0.667,
mirror 0.056 → **0.044** (the attractor's raw material is gone), cycle_w
0.034 → 0.007 (cleaner w), probe fine, w's spatial character intact
(Moran-I_w 0.531 vs 0.545, niche-AUC_w 0.743 vs 0.763, ℓ-baseline equal).
cycle_z 0.475 → 0.440 — inside the cross-run envelope, the one number a
confirmation seed should watch. KL_w/prior-R² unchanged (channel closed at
0.1 either way). Interpretation: at the operating point neighbour-state
detail was buying nothing (doc-09 exposure null) while its presence (a)
seeds the w-mirror basin below α_w = 0.1 and (b) apparently costs a little
likelihood even at 0.1. Pre-registered criterion (Δrecon ≈ 0 → adopt) is
exceeded in the favourable direction. **Awaiting architect ratification**
before changing the default; recommendation: adopt `type_only`, add the
w-mirror metric to the battery regardless, and reopen the α_w question
afterwards (the basin's raw material no longer exists).

### The w-mirror detector: definition and certification protocol (pre-registered)

**Metric** (`w_mirror_delta_r2`): with N_i = Σ_j β_ij μ_z_j (the model's own
graph weights — the same operator as doc-09 exposure), within-type centred,
held-out on the standard split: **ΔR² = R²(μ_w ~ [y, N]) − R²(μ_w ~ y)** —
the share of w's variance that neighbour *state* explains beyond neighbour
*composition*. Honest w (a composition-level field) predicts ΔR² ≈ 0; the
mirror basin (m_ψ reading per-neighbour z detail to reconstruct the cell)
predicts ΔR² ≫ 0.

**Certification before trust**, on labelled runs already on disk:
positives = {alphaw_0.03_s1, alphaw_0.03_s2, alphaw_0.05_s1} (basin: recon
−7.17..−7.23); negatives = {reference_best, alphaw_0.07, alphaw_0.05,
alphaw_0.03, alphaw_0.02, ablation_gat_type_only} — the last is a
*structural* negative (no z in c at all). alphaw_0.05_s2 (recon −7.2285,
between envelope and basin) is deliberately unlabelled — reported, not used.
**Success criterion, stated in advance**: every positive's ΔR² above every
negative's with a visible gap. If certified → wired into the evaluation
battery; if not → the discriminator hypothesis is wrong and the fallback is
a recon-decomposition detector (does the w-channel's likelihood contribution
exceed composition-level capacity). In parallel: `ablation_gat_type_only`
seed-1 confirmation run (cycle_z 0.475→0.440 is the one number to settle).

**w-mirror certification: FAILED twice — the mechanism claim is retracted.**
(`experiments/w_mirror_certification.json`.) Detector 1 (ΔR² of μ_w from
neighbour-z beyond composition): no separation — negatives at low α_w
(0.068–0.080) exceed every labelled positive (0.036–0.050). Detector 2
(recon drop when w is replaced by a composition-only linear surrogate):
no separation either — and it exposed the decisive fact: **the basin runs'
likelihood advantage survives w-substitution** (0.03_s1 still reconstructs
at −7.1835 with ŵ(y)). The channel decomposition (w=0 / z=0 passes)
completes the picture: the basin has BOTH a strong z channel (d_z ≈ 0.30,
level of the best runs) AND a w channel twice the reference's (d_w 0.074–
0.086 vs 0.039) — and that w value is almost entirely recoverable by a
*linear function of composition*. Three independent tests agree: the extra
likelihood does NOT run through per-neighbour z detail. **The "m_ψ reads
neighbour μ_z" mechanism is falsified.**

What stands, measured: (1) α_w < 0.1 is strongly seed-bistable (recon
spread 0.08 vs ±0.005 at 0.1) — the *instability* is real even though my
mechanism for it was wrong; (2) the basin = the optimizer finding a much
stronger **composition-level** w-pathway (co-adapted m_ψ/B) that 0.1 never
reaches — which is w working *as designed*, only harder; whether it is a
better solution or homophily-confound absorption is now an **open
question** (s2's cycle_z crash to 0.354 says at least one basin variant
damages z; s1's battery is unexamined); (3) the `type_only` ablation's
empirical result is unaffected (all numbers stand) but its *motivation
narrative* ("removes the mirror's raw material") is unsupported —
**the architect summary must be corrected before presentation**: the
honest case for type_only is purely empirical (better on nearly every
axis, simpler, and removes an input that measurably buys nothing), not
mechanistic. Next probe if wanted: full validation battery on 0.03_s1 —
if the strong-w basin passes allegiance clean, low-α_w becomes a candidate
*better* operating point rather than a hazard, inverting the earlier
verdict. Issues entry updated accordingly.

**The strong family vindicated (2026-09-11).** `ablation_gat_type_only_s1`
reached recon −7.1924 — strong-family likelihood — in the architecture
where neighbour z is structurally absent: final, independent falsification
of the mirror story. Its battery is the best recorded: **cycle_z 0.499**
(never seen above 0.49), NMI 0.654, mirror 0.046, cycle_w 0.008, and the
allegiance asymmetry *improves* — Moran-I z 0.061 / w 0.628 (ref: 0.082 /
0.545), niche w 0.767 / z 0.654 / ℓ 0.585, triage benign (2.8%).
`alphaw_0.03_s1` likewise passes clean (Moran 0.058/0.589, niche w 0.793 —
highest w-AUC yet). Conclusion, three reversals deep and now
evidence-settled: **the high-recon solution family is a better optimum,
not a pathology** — stronger z AND stronger composition-level w, with
better disentanglement on the very metrics built to catch cheating. The
one bad instance remains alphaw_0.03_s2 (cycle_z 0.354): membership in
the family does not guarantee quality — **selection must read the battery,
never recon alone**. Open: reachability (type_only found it in 1 of 2
seeds at α_w = 0.1; s2 running). Emerging protocol for the architect
package: adopt type_only, train a small seed ensemble, select by the
battery (recon + cycle_z + allegiance + probes) — standard model
selection, now with certified instruments. The α_w question dissolves:
the strong family exists at the default 0.1.

### Doc-10 received: the z–w guard, build + arm-0 launch

Sequencing per instruction: doc-09 on `type_only` first (both seeds queued:
`ablation_gat_type_only_s1` — the battery-selected strong instance — then
s0 as the same-architecture weak-optimum contrast), guard testing after.

**Guard built** exactly per doc-10 §2: `alpha_zw` weight; the §4.6
closed-form machinery verbatim via `TypeCovariances(K, d_z, d_w)` on
(sg μ_z, μ_w-guard-view); straight-through/EMA/shrink/correlation-form
inherited. **One implementation subtlety the routing test caught before it
shipped**: detaching the penalty's z-block is NOT enough — enc_w *consumes*
z (q(w|c,t,z,x)), so the naive term trains enc_z through enc_w's input,
precisely the direction §1 forbids. Fix: a **guard view** — enc_w re-run on
detached inputs (numerically identical μ_w, gradient reaching enc_w's
parameters only; `forward(guard_view=True)`).
`test_zw_guard_gradient_reaches_enc_w_and_never_enc_z` asserts the routing;
it failed on the naive build and passes on the view. Guard is default-off
(`alpha_zw = 0`), NOT in spec 07, adoption gated on doc-10 §4.

**Arm 0 launched** (`discell/model/guard_gate.py`): synthetic gate with a
planted world-A dose (sender-type lognormal ligand variability → exposure
varies within composition; ~30 mid-expression non-marker genes respond at
log-fold 0.3–0.7 ∝ standardised exposure; ρ edited, κ re-mixed, counts
re-sampled). Grid α_w ∈ {0.003, 0.007, 0.02, 0.05, 0.1} × α_zw ∈ {0, 0.01,
0.03, 0.1}, 600-epoch fits; metrics: z-NMI, matched-B recovery, dose-R²,
plant AUROC, KL_w (the guard must not be a backdoor α_w increase).
Deliverable: the cliff-vs-plateau plot (`data/experiments_synthetic/
guard_gate.{json,png}`). Decision per doc-10 §4, unchanged.

**type_only seed triple complete**: recon −7.2527 / −7.1924 / −7.2310
(s0/s1/s2), every battery clean (s2: cycle_z 0.465, cycle_w 0.010, mirror
0.045, NMI 0.666, probe −0.011). The "strong family" is a **continuum of
optima with large seed variance**, not a binary basin — under type_only,
3/3 seeds are honest by every instrument, and the spread (0.06) is pure
optimisation variance. Architect package amendment: seed-ensemble +
battery-selection is ordinary model selection over a rugged landscape;
selected model = s1.

**Doc-10 v2 received**: the guard-view routing is now normative in §2 (with
the naive-build failure recorded), and §1 gains the coherence resolution
(posterior regularisation, §6.4 status; the anticipated cheap solution —
enc_w ignoring z, collapsing the deviation — is what arm 0's dose gate and
arm 1's KL_w gate exist to catch). Implementation already conforms; the
routing regression test extended to the full normative set: the penalty
reaches **no parameter of enc_z, the GAT, m_ψ, or embed_t** (10/10 pass).

**Doc-09 on `type_only` s1 (the selected model)**: exposure visibility is
identically null (mean w-R² −0.010 vs reference −0.010) — and under
type_only this is now **provable from the architecture, not just
measured**: c = f(type-attention, composition, Φ) carries no channel for
neighbour expression detail, so within-composition ligand variation cannot
reach w by construction. The Ẽ-ridge row is settled structurally. Programs:
the candidate set **tightens to one architecture-robust pair** —
PDGFB→PDGFRB survives everything in both arms (ref 0.606/pct 0.98/decoy
0.479; s1 0.583/pct 1.00/decoy 0.497); POSTN→ITGAV/B5 stays strong on
paper in s1 (0.663, pct 1.00) but loses its decoy margin there (decoy
0.645) → demoted to reference-arm-only; TNFSF10 and IL6 do not replicate
across models. One-to-two robust pairs is precisely the cellAdmix
expectation quoted in doc-09's own motivation. s0 arm still running as the
weak-optimum contrast.

**Doc-09 s0 contrast arm**: w-R² null again (−0.009; fourth arm, same
answer). Null-clearing pairs at pct ≥ 0.9: POSTN (1.00, decoy 0.44 — clean
here) and SEMA4C_PLXNB2 (0.96, weak AUROC 0.533); PDGFB does NOT clear in
s0. Cross-model tally over three arms (ref, s1, s0): POSTN 2/3 (decoy
fails in s1), PDGFB 2/3 (does not clear in s0), nothing 3/3. Per §5f's
stability doctrine the honest statement is: **the program route yields at
most 1–2 candidates and none is stable across model instances** — the
route is fragile, and the paper should report the tally, not a winner.

**Doc-10 arm 0 verdict: PARK** (pre-registered §4 clause; grid in
`data/experiments_synthetic/guard_gate.{json,png}`). The cliff did not
become a plateau — guard-on never holds what guard-off loses:

1. **Little cliff to rescue in this world**: guard-off z-NMI degrades only
   mildly at low α_w (0.68 at 0.003 vs 0.74 at 0.1; the M3-era 0.12
   collapse does not reproduce with the invariance penalty active and the
   world-A plant in place). B recovery does degrade (0.57 vs 0.76) — the
   real low-α_w cost.
2. **The guard charges cost without delivering rescue**: at α_zw ∈ {0.01,
   0.03} B recovery is *worse at every α_w* (0.42–0.57 vs 0.65–0.76
   guard-off) and NMI never improves. The §1-anticipated cheap solution is
   measurably active: KL_w shrinks monotonically with α_zw at matched α_w
   (1.10 → 0.47 at α_w = 0.003) — the guard partially closes the very
   channel it exists to protect, and at α_zw = 0.1, α_w ≥ 0.05 it closes
   it fully (KL_w = 0.000, the backdoor-α_w failure).
3. **One curious side-effect for the record**: at α_zw = 0.1 with the
   channel open (α_w ≤ 0.02), w abandons the composition response
   (B_corr 0.06–0.18) and specialises toward the planted dose (dose-R²
   0.18–0.40, plant AUROC 0.58–0.65) — the guard purges z-correlated
   content so hard that the exposure-driven plant becomes w's main
   remaining food. Double-edged: interesting as a hint for a *designed*
   communication channel, useless as a guardrail.

Park report per §4: the low-α_w instability (where it exists at all in the
current loss configuration) is not visible to a Gaussian MI in the μ's as
an I(z;w|t) excess — consistent with the real-slide finding that the
strong-optimum family passes every dependence-based instrument. Caveat
attached: this arm-0 world (8 types, α_a = 0.02 closed-form, plant
present) is not the original M3 configuration; reproducing the historical
0.12 cliff first would be the prerequisite for any second attempt.

**Guard removed from the code** (user decision after the arm-0 PARK): the
`alpha_zw` weight, the guard covariance tracker, the `guard_view` forward
path / `mu_w_guard` field, the routing regression test, and
`discell/model/guard_gate.py` are all reverted — the working tree carries
no trace, by design ("less confusion later"). The full record stays here:
mechanism (§4.6 machinery on (sg μ_z, μ_w-guard-view)), the guard-view
routing subtlety and its test (which caught the naive build), the arm-0
grid and PARK verdict with the three observations, and the results in
`data/experiments_synthetic/guard_gate.{json,png}`. Doc-10 remains the
proposal document; any future attempt starts from this devlog entry plus
the pre-registered §4 criteria, and must first reproduce the historical
M3 cliff in the current loss configuration. Suite after removal: 26/26 on
the affected files.

### Doc-08 v6 execution: steps 0–4 (post-park programme)

**Step 0 — ratified and pinned.** `gat_sources = "type_only"` is now the
default everywhere (TrainConfig, DisCell, sweep CLI), per the user's
ratification; "type_z" stays selectable for era reproduction. Era-safety
added at every checkpoint reload (`setdefault("gat_sources", "type_z")` —
pre-field checkpoints are type_z and the new default must never reshape
them; a shape mismatch would be loud, but the guard makes old runs load
correctly). **Pinned post-park reference: `ablation_gat_type_only_s1`**
(battery-selected best of the seed triple: recon −7.1924, cycle_z 0.499,
clean allegiance) — doc-08 v6's status line quotes s0's numbers; the
selection protocol says s1, and multi-seed envelopes use all three. Full
suite: 129 passed, 1 skipped. Doc-08 v6 changes registered: §8 is now
SIMVI + resolVI as bracketing ablation baselines (each missing one of our
channels, opposite pre-registered failure directions), DisCoVR demoted to
citation — kept for last per instruction (steps 0–4 first).

**Step 4 launched** (gates §6.5/§7.5 only): sweep3 — the post-park κ sweep,
6 κ × 3 seeds at the operating point with type_only sources, GPU 1,
`--tag sweep3` (sweep2/sweep remain untouched). **Step 1 launched**: §2
landmarks + §5 matrix + full report regenerated on the pinned s1, GPU 0.
**Step 2 prerequisite**: MSigDB hallmarks downloaded —
`data/external/msigdb_hallmarks_h.all.v2023.2.Hs.symbols.gmt` (50 sets,
release 2023.2.Hs, data.broadinstitute.org, 2026-09-12).

**Steps 1–3 delivered on the pinned s1** (sweep3 still running):

- **§2/§5 rerun (step 1)**: the reference-era pattern reproduces on the
  post-park reference — interface mid-band w 0.054 vs y-baseline 0.036,
  vessels null, matrix block-diagonal (cycle z 0.440/w 0.006; niche w
  0.767/z 0.654 with the §4.3 flag standing; pseudotime w 0.565/z 0.005).
- **§6 atlas (step 2)** — `discell/model/atlas.py`, instruments certified
  (varimax orthogonality/decomposition-invariance/planted-sparsity;
  hallmark ranking). On s1: **all 6 programs active** (no spare capacity),
  Moran 0.36–0.70, and the doc's claim lands hard: **joint context-driver
  R² 0.92–0.98 per program** — the programs are almost entirely
  context-explained. Hallmarks: EMT (×2, the top-variance programs),
  adipogenesis, hedgehog, spermatogenesis(-labelled), and a small
  G2M-checkpoint program reading as proliferative-niche territory
  (consistent with cycle_w ≈ 0.008: territory, not per-cell cycle).
  §6.5 stubbed pending sweep3. MSigDB provenance in `data/external/`.
- **§7 transport (step 3)** — `discell/model/transport.py`. Two instrument
  lessons registered while running it: (i) selecting the most
  composition-distinct pairs up front is self-defeating (they are exactly
  what the overlap guard flags) — all pairs are evaluated and the guard
  decides; (ii) the overlap comparison must be 1-D **along the gap
  direction**. Structural finding: with k-means niches the supported
  (interpolation) tier is near-empty *by construction* — 3 of 158 panels,
  adjacent niches, nothing to predict (mean full R² 0.026). **The
  informative regime is extrapolation, named as such**: 155 panels, mean
  full R² 0.146, **median calibration slope 0.96**, and the §7.4
  requirement met overwhelmingly — **full beats both single channels in
  135/155 (87%)** (program-only 0.068, leak-only 0.063). Top panels are
  Macrophages (full 0.44–0.51) with the leak channel carrying up to half
  the predicted shift — the macrophage-spillover hypothesis from the
  pseudotime era, now quantified per gene: the program-vs-leak split works.
  Writeup caveat, pre-stated: annotation niches (rim/core) would populate
  the supported tier; data-defined niches cannot.

**Steps 6.5/7.5 closed + matrix rendering fixed (user-caught).**

- **§6.5 κ-survival** (`experiments/atlas_kappa_survival{,_internal}.json`):
  vs the s1 reference, signature correlation is ~0.34–0.41 and flat in κ —
  but that read confounds κ with the budget/optimum gap (s1 is a
  long-budget optimum). The sweep-internal read (reference sweep3_k0.1_s0)
  separates it: **along κ within seed 0.62–0.64, across seeds 0.25–0.32** —
  programs are κ-stable, seed-variable, the same pattern B has always
  shown. Consensus-programs over seeds remains the reporting rule; per-
  program identity should travel by hallmark label, not raw correlation.
- **§7.5 transport sensitivity** (`experiments/transport_kappa_sensitivity
  .json`, seed 0 across κ): the attribution split moves with κ exactly as
  designed — leak-channel R² 0.000 (κ=0, sanity) → 0.084 (κ=0.4) while
  program falls 0.056 → 0.042; the **total** transported prediction is
  κ-robust on the plateau (full 0.10–0.11 for κ ≤ 0.2) and degrades past
  it (0.084 at 0.4) with "full beats both" declining 128 → 75/155. Quote
  the split as a κ-range, never a point.
- **Allegiance matrix rendering** (user-caught defect): rows were
  auto-included and per-row normalised — a −0.01-vs-0.02 row painted the
  0.02 side fully hot, and the flagged niche-z (0.65 > ℓ 0.59) rendered
  warm with no warning. Fixed: one absolute strength scale (AUC 0.5→0,
  0.85→1; R² 0→0, 0.5→1), cells that fail to clear max(floor, ℓ)+margin
  grey out as "n.s.", and significant signal in the UNEXPECTED column gets
  a dagger + title footnote (niche-z now carries the §4.3 flag visibly;
  the mid-band row reads as the near-null it is).
- **Report integration**: `report.py` now appends the §6 atlas (program
  table + figures) and §7 transport (two-tier summary + the new
  `transport_summary.png`: tier bars + the per-panel program-vs-leak
  attribution map) whenever those artefacts exist for the run; sweep3's
  200-epoch κ=0.1 seeds are tight (±0.0014) — the strong-family optimum
  is a long-budget phenomenon, so the pinned s1 is a selected optimum,
  not a sweep-reachable one.

**Transport correction (user-caught, the second matrix-grade catch).** The
question "are we leaking from the old neighbours or the new ones?" exposed
that the implemented "full" prediction deviated from doc-08 §7.2: it used
the actual niche-B populations' complete rates, silently letting a THIRD
thing vary — the type's intrinsic mix across niches (selection) — where
the doc's counterfactual is Δ̂prog + Δ̂leak with z held fixed. (The direct
answer: leakage always comes from the NEW neighbours; κ never changes.)
Fixed: the **counterfactual total** (program + leak, z fixed) is now the
headline — extrapolation tier mean R² **0.099**, slope **0.92**, beats
both single channels in **100/155 (65%)** — the §7.4 claim passes on the
correct object; the old "full" stays as the *model account*, and its
excess over the counterfactual (0.146 − 0.099 ≈ **0.05**) is now a
measurement in its own right: the selection share of observed niche
differences. Observed niche difference = context response + contamination
+ selection, all three quantified. Report §7 updated (z-fixed statement,
new-neighbours statement, four-bar summary figure).

### Doc-11 received: z-applications A1–A6 — flags baked, dependencies building

**The two architect flags, resolved first (they gate everything):**

1. **"Ceiling" renamed to the 50-PC linear expression reference** across the
   codebase: producer key `cycle.linear_ref` (train evaluate + TB scalar
   `val/cycle_r2_linear_ref*`), readers with pre-rename fallbacks (report,
   sweep, validate matrix — old runs and old TB events keep loading), all
   user-facing labels now say "linear expression reference — not a ceiling:
   z may legitimately exceed it" (it does, ~2x; M6 documented why).
2. **Hallmark background VERIFIED correct** — `read_hallmarks` intersects
   every set with the expressed panel and the hypergeometric population is
   the expressed-panel size; the genome-background bug the architect
   suspected is absent. The actual defect: labels shipped ungated
   (SPERMATOGENESIS at p = 0.08, ADIPOGENESIS at p = 0.3). Fixed:
   **BH correction across the sets tested per program, label ships only at
   q ≤ 0.05**, otherwise "(none significant)"; q shown in the report table.
   Atlas/matrix/report regenerating with both changes.

**Doc-11 structure decision** (user: "well separated for less confusion"):
new package `discell/applications/` — one module per application with its
own CLI and pre-registered pass/fail, nothing imported by model/training
code; shared data dependencies in `applications/shared.py`, built once.
Sequencing per the doc: A4 → A1 → A3 → A6 → A5 → A2.

**Shared dependencies building now** (the doc's two new data extractions):
`transcripts.parquet` pulled from the ovarian archive; `qc/nuclear_dapi.
parquet` (integrated nuclear DAPI + area per cell, morphology image x
nucleus polygons — A4's orthogonal-physics ground truth) and
`qc/nuclear_counts.npz` (nucleus-flagged q20 transcripts only — the
segmentation-perturbation arm for A4/A1) both extracting in background.
One dependency swap: polygon rasterisation via matplotlib.path instead of
adding scikit-image.

### A4 (decontaminated cycle call) — INCONCLUSIVE on this slide, reported honestly

A4 built and run on `ablation_gat_type_only_s1` (three implementable legs;
planted-world leg 4 stubbed). The verdict is *inconclusive*, and per doc-11
("a failed leg is reported as a finding, not tuned away") that is the
result, not something to tune:

- **Leg 1 (DAPI ground truth) is at chance for BOTH callers** — AUROC(call
  -> nuclear DNA content) raw 0.498, z 0.480; group median DAPI-z all within
  ±0.07. The orthogonal-physics adjudicator cannot separate cycling from
  non-cycling here at all, so it cannot arbitrate raw-vs-z. Two compounding
  causes: (i) integrated nuclear DAPI from the 0.21 µm/px morphology image
  is a weak ploidy proxy at this resolution; (ii) within the four MKI67-high
  cycling types the raw Tirosh phase call fires on **92%** of cells, leaving
  the contrast groups tiny (raw-only 1889, both-neg 2867). The cycling-type
  restriction that A4 needs for power is exactly what collapses the DAPI
  contrast.
- **Leg 2 (exposure fingerprint) is confounded as specified** — the z
  cycle-projection's slope on neighbour exposure (2.32) sits far outside a
  permutation band [-0.35, 0.33] built by shuffling exposure within type.
  But that band destroys the *real* spatial clustering of cycling cells
  (proliferative niches are genuinely contiguous, §3.2's benign case), so
  the test conflates real niche clustering with contamination. The band is
  the wrong null; the leg needs a contamination-specific control (e.g. the
  §7-style program-vs-leak split), not a spatial-structure null.
- **Leg 3 (nuclear recomputation) technically "passes" but is
  uninformative** — raw-only positives lose nuclear signal (nuclear -0.004
  vs both-pos 0.045), *but* raw-only cells have raw cycle level ~0.001 to
  begin with (they are marginal phase calls, barely above both-neg -0.041).
  The drop is "weak calls are weak", not "contamination stripped".

**Design conclusion for the architect**: A4 as specified needs a slide where
the cycle signal is strong enough that (a) DAPI separates phases and (b) the
raw call is not near-saturated within cycling types. This HGSOC slide fails
both (cycle split-half reliability S 0.22 / G2M 0.52 already warned of it).
Options: run A4 on a higher-cycling-fraction reference, or replace the DAPI
leg with the planted world (leg 4) as the primary adjudicator here and
demote the real-data legs to supporting. Not tuned away; carried as the A4
finding. Proceeding to A1 (mirage states), which does not depend on the
cycle signal.

### A4 v2 — the doc-11 v1 redesign (motivation written before the run)

Doc-11 v1 returns A4 with a redesign that accepts the diagnosis above: the
flaw was the *instrument*, not the tissue, so no new slide. What changes and
why, pre-registered before the rerun:

- **Contrast population moves to post-mitotic types.** Within the MKI67-high
  types the global Tirosh phase threshold fires on 92% of cells, so the
  "raw-positive / z-negative" contrast group is tiny and least separable.
  The predicted leak victims are stromal/immune cells at the edges of
  proliferative niches: unsaturated base rates, maximal DAPI contrast.
  Population = every labelled, connected type that is not one of the top-4
  MKI67 types and not Unassigned, with ≥ 2 000 cells. Calls: raw = Tirosh
  phase ≠ G1; z = rate-matched per type (top-k by the probe projection
  `max(z·β_S, z·β_G2M)`, β fitted in the cycling types on training folds, k
  = the raw-positive count of that type) so the two callers can only differ
  in *which* cells they pick, not how many.
- **Stratified test replaces the slope-on-exposure.** Exposure = β-weighted
  neighbour cycle score. Statistic = positive rate in the top exposure
  quartile minus the bottom, within type, averaged; band = the same
  statistic under within-type exposure shuffles (200). This is still a
  spatial-structure null (the objection stands for interpretation), so it
  is demoted to "does raw track exposure, and does z track it less" — the
  mechanism claim is not carried by it.
- **Leg 2's null replaced by the gene-split fingerprint.** Contamination
  transfers *transcripts*; niche co-clustering transfers *state*. Split the
  S+G2M set into disjoint random halves A/B: Δ = corr(own_A, nbr_A) −
  corr(own_A, nbr_B), averaged over 40 splits, within type. Under homophily
  the neighbours' A and B halves are equally informative about a cell's A
  score (state is set-independent) → Δ ≈ 0; leakage inflates only the
  same-half correlation → Δ > 0. Physics check: leak is one-hop, so Δ on
  ring 1 must exceed Δ on exact ring 2. Run on the post-mitotic population
  (the claim) and on the cycling types (contrast). Detection rule: ring-1
  CI excludes 0 *and* ring-1 mean > ring-2 mean.
- **Planted world promoted to primary adjudicator** (`planted.py` +
  `planted_world`): gate scaffold (6 000 cells, 8 types, κ = 0.2), a
  cycle-like program (+1.0 log-fold on 12 genes) planted in 30% of the
  cells of two "cycling" types, re-mixed through the true leak operator,
  counts resampled. Short DisCell fit (400 epochs, `fit_synthetic`, nothing
  tuned per world). Raw score = mean log-normalised planted-gene expression;
  z score = the same probe projection as on real data, fitted on training
  folds. Read-outs in the *victims* (non-cycling cells in the top exposure
  quartile to planted cells): FPR at the threshold that reproduces the 30%
  planted rate inside the cycling types; and AUROC(planted | score) within
  the cycling types (sensitivity kept?). Pass per seed: z victim FPR < raw
  victim FPR **and** z AUROC ≥ raw AUROC − 0.02. Three seeds.
- **DAPI retained, group level only** (FFPE sectioning truncates nuclei;
  integrated intensity is a weak ploidy proxy). Groups both-positive /
  raw-only / neither in the post-mitotic population, `dapi_sum`
  standardised within type; medians and KS vs neither. Supporting evidence
  only — no AUROC claim.
- **Pre-registered honest outcome**: if the raw gap sits inside its shuffle
  band *and* the gene-split fingerprint is null in the post-mitotic
  population, the report line is "leak-induced cycle false-positives are
  rare at κ = 0.1 on this slide" — consistent with doc-09's aggregate-small
  picture, and a finding rather than a failure. The planted world then
  still carries the mechanism claim (can z reject leak-induced positives
  when they *do* occur).

Module: `discell/applications/a4_cycle.py` (rewritten; v1 is gone),
`discell/applications/planted.py` (shared synthetic fit for all doc-11
planted worlds). Output `runs/<run>/applications/a4_cycle.{json,png}`.

### A4 v2 results — honest outcome triggered; the planted adjudicator needs rework; A4 PARKED (2026-09-11)

Run: `ablation_gat_type_only_s1`, `applications/a4_cycle.{json,png,log}`;
the run report now carries an "A4" section (`report.a4_section`). Numbers:

- **Population**: 12 post-mitotic types, 342 743 connected cells; raw
  Tirosh phase ≠ G1 fires on 45.5% of them (unsaturated, as the redesign
  wanted; contrast v1's 92% inside the cycling types).
- **Stratified gap** (Q4 − Q1 call rate by neighbour-cycle exposure,
  within type): raw **0.159** (0.35 → 0.51), z rate-matched **0.068**
  (0.38 → 0.45); both far outside the within-type shuffle band
  [−0.009, 0.008]. Per type the raw/z contrast is largest in macrophages
  0.37/0.05, T-NK 0.17/0.01, tumour-associated fibroblasts 0.18/0.04;
  tumour-associated endothelium 0.20 vs stroma-associated 0.06 (raw) is
  the pattern real angiogenic proliferation would also produce.
- **Gene-split fingerprint** (the leak-specific leg): post-mitotic Δ ring 1
  = **0.006, CI [−0.006, 0.020]**, ring 2 0.003 — null; cycling types
  0.008 [−0.009, 0.037] — null. No transcript-transfer signature in the
  predicted victims, on either ring.
- **DAPI** (group level): both-positive median **−0.17** (KS 0.14 vs
  neither, n 82k), raw-only +0.04 (KS 0.04), neither +0.13. The DNA-content
  proxy does not place either caller's post-mitotic positives above the
  negatives — if anything below.
- **Planted world**: **0/3 seeds pass**; victim FPR raw 0.21 vs z 0.32,
  AUROC raw 0.68 vs z 0.64 (seed means).

**Reading, per the pre-registration.** The raw call tracks exposure, but
the fingerprint says that tracking is *state*, not transcript transfer —
so the pre-registered honest outcome stands: **leak-induced cycle false
positives are rare at κ = 0.1 on this slide**, consistent with doc-09's
aggregate-small picture. The flatter z line is therefore not a
decontamination win; it is closer to doc-11's named fail state (z
*under-calls* a real spatial gradient of post-mitotic cycling —
sensitivity, not specificity). DAPI supports neither caller.

**The planted adjudicator did not adjudicate — two defects, recorded as
found, not fixed (A4 parked by the user before either was verified):**

1. *Read-out inconsistency + under-power.* The planted leg thresholds
   globally (70th percentile inside the cycling types) while the real-data
   leg rate-matches within type; the JSON's `control_fpr` (unexposed,
   non-planted cells) ranges 0.02–0.45 for raw and 0.32–0.37 for z across
   seeds, so "victim FPR" is dominated by synthetic type offsets. The
   leak-attributable excess (victim − control) is raw 0.11/0.00/0.02, z
   0.08/−0.08/−0.09: in two of three seeds the leak is invisible even to
   the raw score, and raw detects the planted cells themselves at only
   AUROC ≈ 0.67 (+1.0 log-fold on 12/60 genes is far weaker than real
   MKI67/TOP2A). A power gate (raw AUROC ≥ 0.9 and raw excess ≥ 0.1, else
   the world is uninformative) must precede any z verdict. This is my
   implementation flaw, not a finding about z.
2. *Amortisation-gap hypothesis (UNVERIFIED).* `enc_z` conditions on
   `[counts_i, log ℓ_i, onehot t_i]` only (`networks.py`, by design —
   "enc_z never sees c"). The true posterior p(z_i | x_i, ρ̄_i) depends
   on the cell's own neighbours; the amortised q(z_i | x_i, t_i) cannot.
   If so, the leak is removed on the decoder side (B, the decoder, the
   population-level z law) but the **per-cell** posterior mean inherits
   whatever leaked into x_i, and a per-cell "decontaminated call through
   z" is not something the architecture can deliver as stated. The route
   the model does license is counts-level: x̃_i = x_i − κ ℓ_i ρ̄_i with
   the model's own ρ̄_i, then rescore. This bears on A1/A2/A5 as well
   (per-cell z, `softmax(a(z))`). To be adjudicated by a planted world
   with the power gate above, comparing raw / z-probe / leak-subtracted
   counts — not started.

Status: **A4 and all doc-11 applications parked** (user, 2026-09-11) in
favour of consolidating the project state; the two items above are the
first things to pick up when A4 resumes.

### Crystallisation: registers brought current, `docs/handover.md` written (2026-09-14)

User request after parking A4: consolidate. Done in this order —
(1) the A4 v2 results entry above; (2) `docs/issues.md`: new V1–V9 table
(doc-08/09/11 instrument defects incl. the open A4 planted-adjudicator
defect V9), the low-α_w bistability item marked SETTLED (strong family =
better optimum; select by battery), three new watch items (amortised
per-cell z cannot see its own neighbours — UNVERIFIED; doc-09 61% is
uncalibrated; A4's honest outcome), duplicate mirror item struck;
(3) `docs/spec_deviations.md`: `type_only` registered as a §4.1 departure
with its empirical-only case, GAT-query sizing text corrected, current
defaults + budget note, doc-08 §5 rendering / §6.1 / §6.3 / §6.5 /
§7.2–7.4 rows, new doc-09 table, doc-10 park record, doc-11 A4 table;
(4) `docs/handover.md` — the state document: operating point and seeds,
how to run everything, results by claim with artefact pointers, κ and
seed envelopes, retractions table, ranked limitations (type-label
circularity first, the amortisation gap second), a quality verdict
(publishable / suggestive / unsupported), open architect questions, next
steps (doc-08 §8 baselines first). One number surfaced while
cross-checking that had not been written down: the ovarian graphclust
control's invariance probe re-fitted against the *curated* labels reads
ΔCE 0.061 (reference 0.004) — coarser training labels leave finer-label
niche information in z; recorded in handover §4.8. The report for
`ablation_gat_type_only_s1` was regenerated with the A4 section. Suite:
132 passed, 1 skipped (2026-09-11). Nothing committed yet — the working
tree carries all of doc-08 §6 onward.

### Three user questions on the cons — measured, not argued (2026-09-14)

Ad-hoc analysis on the type_only seed triple (best checkpoints, all
407k cells via `Trainer._sweep`; script in the session scratchpad, numbers
below are the record).

**1. KL_w: is there per-cell variance, and what does it mean in % terms?**
KL_w is against the *learned* conditional prior m_ψ(c,t), so it counts
only what w knows beyond context (KL_z, against N(0,I), counts everything
z knows — not comparable). 93–97% of it is mean shift, so KL ≈ ‖μ_w −
m_ψ‖²/2: at s1 the median cell deviates by ‖dev‖ 0.08 against a prior
field with per-dim SD 0.4–5.1 — the per-cell part is **0.01–0.1% of w's
variance per dim**. Distribution at s1: mean 0.0047 nats, median 0.0028,
p99 0.028, max 0.14; the top 1% of cells carry 8% of the total (no heavy
tail → no anomaly channel to read); per type macrophages / TAFs highest
(0.007), proliferative tumour lowest (0.002). s2 identical in shape.
**s0's best checkpoint is different: KL_w 0.108 mean, 62% of cells above
0.05, uniform (top 1% carry 3%), ‖dev‖ 0.40, 0.2–2.3% of variance per
dim, highest in proliferative tumour (0.20).** Reconciled with
`history.jsonl`: KL_w **oscillates 20–100× between evaluations five epochs
apart** in every seed (s0: 0.008 → 0.124 at epoch 59 → 0.006; s1 spikes
0.37 at 29, 0.28 at 59; s2 0.78 at 54) — the posterior deviates, the
prior catches up within ≤ 5 epochs. s0's `best.pt` (epoch 59) sits on a
spike. Consequence: a checkpoint's KL_w is a snapshot of a chase, not a
property of the solution; the α_w-study's "channel opens" reads were
single snapshots too. Recorded as a watch item.

**2. B seed variance — is it because d_w is small?** The opposite. The
eigen-spectrum of cov(μ_w) is **[0.83, 0.17, 0, 0, 0, 0]** at s1 and
[0.98, 0.02, 0, …] at s0/s2: w occupies a 2-D (s1) or ~1-D (s0, s2)
subspace of its 6 dims; the realised shift μ_w·B has the same spectrum
(0.85/0.15; 0.99/0.01). The four null directions carry no variance, so
their B columns are unconstrained by the data and arbitrary across seeds
— that is what `matched_correlation` (0.47–0.60 here, 0.43–0.52 in
sweep3) measures. The invariant object agrees across seeds: per-cell
shift vectors correlate 0.90–0.93 (median; p10 0.72), A's shift variance
lies 97–99% inside B's shift span (82–96% inside the top-1 direction),
dominant gene direction cosine 0.89–0.95. So: d_w = 6 is 3–6× more than
w uses; "B is seed-bistable" was a metric artefact of the null columns
(the κ = 0 → κ > 0 drop 0.70 → 0.45 may still carry §7.7's confound —
unresolved without those latents). Two fixes to propose: (i) B stability
= shift-space overlap, not matched columns; (ii) a d_w ablation {2, 3, 6}
× 3 seeds — prediction: at d_w = 2 matched|corr| ≈ 0.9 with recon /
cycle / niche unchanged.

**Correction to the §6 atlas (same root).** `atlas.py` judges activity on
the variance of the *varimax-rotated* coordinates (≥ 1% of the max); a
rank-2 w rotated onto 6 axes gives six correlated coordinates that all
pass (rotated variances 2.35 / 1.09 / 1.03 / 0.93 / 0.51 / 0.11). The
honest read of the pinned reference is **two programs (s1) — one
dominant (83%), one secondary (17%) — not six**; the two EMT-labelled
programs 0 and 1 are the top-variance ones and the four "(none
significant)" programs are the null directions. Drivers R² 0.92–0.98 and
Moran 0.36–0.70 on the null coordinates are inherited from the two real
ones through collinearity. Fix: activity by the eigen-spectrum of
cov(μ_w) (effective rank r), varimax *within* the r-dim subspace, report
r. Handover §4.4 and the report table are wrong until then; issues V10.

**3. The amortisation gap — feed the encoder x − κℓρ̄?** Design note for
the architect, not run. The true posterior is p(z_i | x_i, ρ̄_i);
subtracting the expected influx keeps §4.2's "enc_z never sees c" (the
correction *removes* neighbour signal; conditioning q(z | x, t, ρ̄) would
re-open the channel). ρ̄ is why the batch has two rings, and §4.5 forbids
ρ_j ≈ softmax(a(z_j)) (w_j must stay in it). Ordering is the cost: ρ̄_i is
computed after the seeds are encoded, and the exact form regresses (ring
1 corrected needs ring 2's ρ needs ring 3's types …). Practical scheme:
pass 1 as now (raw inputs, all nodes) → ρ_j for ring 1 → ρ̄_i; pass 2
re-encodes **seeds only** from x̃_i = clip(x_i − κℓ_iρ̄_i, 0), decodes,
likelihood unchanged on raw x with p = (1−κ)ρ_i + κρ̄_i (ρ̄ stop-gradient).
Ring-1 ρ_j from uncorrected encoders is an O(κ²) inconsistency; cost ≈
+35% (seeds encoded/decoded twice), no cache. Pre-registered tests: the
flagged niche-z residual (AUC 0.654 vs ℓ 0.585) should drop if part of it
is leaked transcripts; cycle_z must not fall; A4's planted world becomes
answerable. Spec change → architect.

### x̃ = x − κℓρ̄ as the encoder input — architect-approved spec change, gated on results (2026-09-14)

Architect ruling on the design note above (quoted in full in the session;
the substance): the true posterior is p(z | x, ρ̄) and a per-cell bias
κℓρ̄ cannot be removed by any function of x alone — today's enc_z is the
input that carries context; E[x̃ | z, ρ̄] = ℓ(1−κ)ρ, so the subtraction
removes the systematic context component and leaves mean-zero leak
noise. Features only, likelihood untouched on raw x. Approved with five
conditions, all built in:

1. one encoder, two input distributions (raw x on ring 1 / pass 1, x̃ on
   seeds) is an accepted O(κ²) inconsistency — stated in the code, and
   the planted-κ gate asserts it is harmless;
2. enc_w's x input becomes x̃ too (same posterior argument);
3. normalisation: x̃ is normalised by **its own sum** (≈ (1−κ)ℓ), the log-
   depth scalar stays **raw ℓ** — no κ-dependent depth shift across sweep
   points;
4. penalty and probes read **pass-2** μ_z; pass-1 z's exist only to feed
   ρ̄ (they still must be computed: fellow seeds are leak sources);
5. acceptance, pre-registered: niche-z residual drops toward the ℓ-
   baseline (0.654 → toward 0.585; not necessarily to it — leak noise
   remains), cycle_z within the seed envelope (0.44–0.50), held-out recon
   within envelope (−7.19…−7.25), planted-κ gate recovery unchanged, and
   the w-side battery (Moran, niche AUC, transport) unmoved — the change
   touches z's evidence, not w's.

Expectation tempered per the architect: x̃ removes the *mean* leak from
the encoder's view; the invariance penalty keeps its job for the rest
(niche-induced real expression, leak noise). A bias fix, not a penalty
replacement. Green → the architect patches 07 (§2 notation, §4.2 input,
§4.5 two-pass, §7.13 why-x̃).

Implementation: `DisCell(subtract_leak=True)` / `TrainConfig.subtract_leak`
/ `--subtract-leak` (default **off** until green). Pass 1 as before; after
ρ̄ (data) the seeds are re-encoded from x̃ = clip(x − κℓρ̄, 0), enc_w reads
x̃, the seeds' rows of μ_z/z/μ_w/w/log ρ are replaced, term (a)/(b)
likelihoods computed from pass 2 on raw x. κ = 0 ⇒ x̃ ≡ x (exact no-op),
isolated seeds ⇒ ρ̄ = 0 ⇒ x̃ ≡ x. Era guards at every checkpoint reload.
Plan: unit tests → planted gate κ ∈ {0, 0.2} with/without → three 500-
epoch seeds `xtilde_s{0,1,2}` on the ovarian slide → validate/report/
transport → verdict against the five criteria.

### d_w ablation — is B's seed variance the empty dimensions? (motivation, 2026-09-14)

Measured above: cov(μ_w) has effective rank 2 (s1) / ~1 (s0, s2) inside
d_w = 6, and the matched-column B cosine across seeds (0.43–0.52 in
sweep3) is dominated by the four null directions while the realised shift
μ_w·B agrees at 0.97–0.99. Prediction if that reading is right: at
d_w = 2 (and 3) the matched-column cosine across seeds rises to the level
of the shift-space overlap (≳ 0.9), with recon, NMI, cycle_z/w, probe,
mirror and the w-side battery unchanged — d_w = 6 is capacity that is
never used. If instead recon or the w reads *drop* at d_w = 2, w's rank-2
appearance was a property of the optimum, not of the data, and d_w must
stay ≥ 6. Design: `dw2_s{0,1,2}`, `dw3_s{0,1,2}` at the sweep budget (200
epochs, patience 20, defaults otherwise) so the controls are the three
sweep3 κ = 0.1 seeds (d_w = 6, same budget; recon −7.2562 ± 0.0011,
cycle_z 0.450 ± 0.027, B cosine 0.49). Read-outs: matched-column cosine
and shift-space overlap across seeds, recon/NMI/cycle/probe/mirror from
`metrics.json`, niche AUC and Moran for w from the §3–4 battery. Not a
new operating point yet — a diagnosis of the metric and of d_w.

**x̃ planted gate (condition 1/5), result.** κ = 0 is an exact no-op
(differences at the third decimal = CUDA nondeterminism). κ = 0.2, two
seeds, without → with x̃: NMI 0.735/0.763 → **0.703/0.708**, w-CCA
0.906/0.755 → 0.893/0.736, B cosine 0.787/0.803 → 0.763/0.789, niche-R²
in z 0.125/0.090 → 0.122/0.093. Every gate threshold still passes (NMI >
0.45, B > 0.45, niche-R² < 0.25), so "recovery unchanged" holds at the
gate's own resolution — but the z-type NMI moves down by 0.03–0.055 in
both seeds, a consistent direction, not noise. Reading offered, to be
judged on the slide: at κ = 0.2, 20% of a cell's counts are its
neighbours', and in a homophilous tissue those carry the *same type*;
x̃ strips a leak-borne type cue the raw encoder was silently using, and
the clipped view has ~20% fewer effective counts. If the slide (κ = 0.1,
seed spread ±0.015) shows the same, it is the price of the bias fix,
and the honest number; the acceptance criterion for the slide is
cycle_z and recon within envelope, not NMI parity.

### Third slide intake: fresh-frozen ovarian adenocarcinoma (`xenium_prime_human_ovary_ff`) (2026-09-14)

Motivation: a second ovarian slide under a different preservation method
(fresh frozen vs FFPE), for the transfer story and for the image-derived
landmark upgrade the interface analysis asked for. The request named
`downloads/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip`; that archive
**is the development slide** — same `analysis_uuid` (39f5fc8e…), same
Xenium Ranger relabel run (`Xenium_Prime_Ovarian_Cancer_FFPE_XRrun`),
byte-identical `metrics_summary.csv` to `extracted/Xenium_Prime_Ovarian_
Cancer_FFPE/`, and `paths.slug` maps it to `xenium_prime_ovarian_cancer_
ffpe` by design. The fresh-frozen slide is the third zip in that folder,
`Xenium_Prime_Human_Ovary_FF_outs.zip` (10x "Fresh Frozen Human Ovarian
Adenocarcinoma with 5K Human Pan Tissue and Pathways Panel", published
2024-09-04, bundle 5.0.0, analysis xenium-3.0.0.15, chemistry v2). Its
tabular members and the four `morphology_focus` channels are extracted to
`extracted/Xenium_Prime_Human_Ovary_FF/` (15.5 GB; the 37 GB
`morphology.ome.tif` and the transcript tables stay in the zip).

Slide facts, from `experiment.xenium`, `cells.parquet`, `analysis.tar.gz`,
the OME headers, and an intake check with the standard loader and the
`egomask_ego_v1` crop recipe (`figures/intake_image_check.{json,png}` under
the new dataset; the spike script itself is not kept):

- 1,157,659 cells (1,157,637 with expression + geometry; 22 degenerate
  polygons and 22 matrix-only cells dropped) × 5,001 genes — the base
  Prime 5K panel, identical to lung's; ovarian FFPE's 100 custom add-on
  genes absent. Region 198 mm² (FFPE 87); image 54,013 × 101,928 px.
- **Depth 8× the FFPE slides**: transcripts/cell median 1,401, mean 1,668
  (ovarian FFPE 178 / 262; lung 242 / 440), 76 zero-count cells. By the §5
  rule α_z = 1/ℓ̄ ≈ 0.0007 — to be set at training, with the caveat that
  the rule was calibrated over a 143–242 range and is extrapolated here.
- Segmentation: 69.5% interior (18S), 28.0% boundary, 2.5% nucleus
  expansion — same stain kit (`xenium_cell_segmentation_stains_v1`) and
  fractions as FFPE (69.4 / 28.6 / 2.0). Cells larger: median area 88 µm²
  vs 60 (ovarian FFPE) / 52 (lung); high-quality transcript thickness
  6.3 µm vs 4.3.
- **Labels: no curated `cell_groups.csv`** on the 10x CDN (403 at
  3.0.0/3.0.1/3.1.0; the catalogue's file list carries none) → graphclust
  default, **38 clusters** (538–134,246 cells), the 76 zero-count cells
  unclustered → Unassigned via the P3 fold. Cycle instruments portable:
  MKI67 present, 18 S + 34 G2M panel hits, identical to both FFPE slides.
- Images: `morphology_focus` with the same four channels (DAPI;
  ATP1A1/CD45/E-Cadherin; 18S; αSMA/Vimentin), 0.2125 µm/px, uint16,
  1024-px JPEG 2000 tiles, 9 pyramid levels — `tiff.py` and `crops.py`
  read it unchanged. Registration (DAPI in a 9 × 9 px window at 400
  centroids vs 400 random points in the cell bounding box): median 2,434 vs
  169 (FFPE 1,671 vs 81); no centroid outside the image.
- Mask radius (E1 rule re-run): covering radius p50 7.3 µm, p99 14.5,
  p99.9 18.4, p100 32.7; **31 of 1.16 M cells exceed the 25 µm disk**
  (ovarian FFPE 6, lung 2) → 25 µm stands.
- Crop intensities (256 px @ 0.5 µm/px, 192 random cells, /65535):
  per-channel means 0.021 / 0.029 / 0.014 / 0.008 vs FFPE 0.009 / 0.011 /
  0.007 / 0.006 — **FF is 1.4–2.7× brighter in every channel** and visibly
  softer (larger, less condensed nuclei; thicker section). KRONOS v1
  z-scores with fixed marker statistics (DAPI 0.083, NAKATP 0.044, A-SMA
  0.030), so both slides sit below its reference and FF is the nearer of
  the two. Φ is consumed within a slide, so this is a note for any
  cross-slide Φ comparison, not a blocker.

Decisions: pipeline unchanged — `python -m discell.preprocess --sample
data/raw/xenium/extracted/Xenium_Prime_Human_Ovary_FF`, dataset id
`xenium_prime_human_ovary_ff`, bundle first (CPU; the FF load without
graphs took 137 s), embeddings with the standard arm (`--model v1
--target-mpp 0.5 --mask ego --embeddings-name egomask_ego_v1`, ~3 h at
~110 cells/s) once a GPU is free of the x̃ runs. Open, for the user: full
slide vs a `--max-cells` window — resident training needs 1.16 M × 5,001 ×
fp16 ≈ 11.6 GB, the whole of a 24 GB card; and whether this slide replaces
lung or joins it as a third transfer target.

**x̃ seeds, training stage** (`runs/xtilde_s{0,1}/metrics.json`, 500-epoch
budget, best epochs 64 / 84): recon **−7.2525 / −7.1932** vs the same
seeds without x̃ −7.2527 / −7.1924 — identical families, within ±0.001;
cycle_z 0.399 / **0.491** vs 0.440 / 0.499; cycle_w 0.005 / 0.009; mirror
0.042 / 0.045; probe ΔCE −0.009 / −0.005 at the floor; NMI **0.640 /
0.644 vs 0.667 / 0.654** (−0.027 / −0.010, the gate's direction on the
slide too). Battery (§3–5 + transport + report) launched on both; the
decision waits for niche-z AUC vs the ℓ-baseline and the w-side rows.

**Bundle result** (`data/datasets/xenium_prime_human_ovary_ff/bundle/`,
`logs/bundle.log`, 1,494 s; 9.05 GB, of which `full.h5ad` 8.6 GB):
750 invalid polygons repaired, 22 degenerate dropped; Delaunay 3,472,886
edges; **contact (1 µm) 2,130,754 edges, mean degree 3.68, isolated 3.2%**
(ovarian FFPE 3.12 / 9.6%, lung 2.48 / 10.8% — larger, more tightly packed
cells); **voronoi (clip 30 µm) 3,458,443 edges, mean degree 5.98, isolated
0.0%, median shared wall 7.46 µm** (FFPE 5.88 / 6.85 µm, lung 5.87 /
7.41 µm); apposed wall contact 100% nonzero median 7.81 µm, voronoi 61.6%
nonzero median 7.82 µm (lung 42.0%). The loader opens it in 9 s
(`CellGraphDataset.from_dataset("xenium_prime_human_ovary_ff")`): 39 types
(38 clusters + Unassigned), β defined on every non-isolated cell (277
isolated), batches draw with the standard edge attributes. No code
changed. Embeddings not yet run (GPU held by the x̃ seeds).
Embedding queued (user: all cells, first free GPU): `logs/embed_launcher.sh`
polls both GPUs every 30 s and, on the first with no compute process, runs
the standard arm (`--only embed --model v1 --target-mpp 0.5 --mask ego
--embeddings-name egomask_ego_v1`) → `embeddings/egomask_ego_v1.pt`,
`logs/embed.log`. As on lung, every cell is embedded — the 31 cells wider
than the 25 µm disk are not dropped by the standard path (the ovarian
`egomask_ego_v1` with 407,114 cells came from the ego-masking experiment,
which did drop its 6). Expected ~3 h at the lung rate (~110 cells/s).

**x̃ battery, seed 1** (`runs/xtilde_s1/{validation,transport}`; reference
= `ablation_gat_type_only_s1`, same seed, same budget):

| read | reference | x̃ | criterion |
|---|---|---|---|
| niche-z AUC (ℓ-baseline 0.585) | 0.654 (residual +0.069) | **0.641 (+0.056)** | drops toward ℓ — yes, by ~20% of the gap |
| cycle_z (battery / training) | 0.440 / 0.499 | 0.456 / 0.491 | within envelope — yes |
| held-out recon | −7.1924 | −7.1932 | within envelope — yes |
| NMI | 0.654 | 0.644 | not a criterion; −0.010 (gate's direction) |
| Moran mean |I| z / w | 0.061 / 0.628 | 0.064 / 0.642 | w unmoved — yes |
| niche-w AUC | 0.768 | 0.773 | unmoved — yes |
| pseudotime w (niche R² / coherence) | 0.565 / 0.489 | 0.509 / 0.571 | unmoved within noise |
| transport counterfactual / slope / beats-both | 0.099 / 0.92 / 100 | 0.104 / 0.95 / 103 | unmoved (slightly up) |

The flagged cell stays flagged (z on niche label still > ℓ + margin); the
triage is unchanged (max y-R² 0.028). Reading: the bias fix moves the
niche-z residual in the predicted direction and costs nothing on any
other row, but on one seed the move is modest — most of the residual is
not leaked transcripts (the triage's "benign intrinsic spatial structure"
reading stands for the remainder). Seeds 0 and 2 pending (s0's first
battery OOM'd next to the d_w chain; re-queued on GPU 0).

**Two TensorBoard figures added (user request, 2026-09-14)** in
`Trainer.figures`: `figures/kl_spatial` — per-cell KL(q(z)‖N(0,I)) and
KL(q(w)‖m_ψ(c,t)) painted on the tissue, one row for all cells and one
per panel type on its own cells (per-panel 2–98% colour scale, median in
the title; `_sweep` now also returns per-cell `kl_z`); and
`figures/zw_std_trajectories_{umap,pca}` — the existing full-space
principal-curve figure on the joint state [z, w] standardised per
dimension (w's top dim carries ~25× a z dim's variance, so raw
concatenation would be w alone). Both fire at every `figures_every`
event of any run trained from now on; +9 UMAP fits per event. Suite:
`test_model_train` 5/5 (the smoke fit renders them).

### Doc-09 §8 — LR co-occurrence map (SIMVI fig-6h style, controlled): motivation (2026-09-14)

Purpose per the doc: a descriptive figure in the field's standard visual
language showing that w carries organised, nameable context signal — with
the control panel that keeps it honest. No communication claim. Panel A:
rows = top ±10 B-loadings per **effective-rank** program (V10: programs
from the top-r principal directions of cov(μ_w), r = components ≥ 1% of
the variance, varimax within that subspace — not the six rotated raw
dims), row value per cell ⟨w_i, B_g⟩; columns = gate-zero LR pairs ranked
by Var(exposure) × receiver prevalence, top 30, labelled by CellChatDB
pathway; column value per cell LR_i = R_i × E_i (depth-normalised log1p
receptor × one-hop exposure); entry = Spearman within receiver type (≥ 500
receptor-positive validation cells), Fisher-z pooled, validation tiles;
within-type permutation of the LR score (200), BH per map, n.s. greyed.
Panel B: the same after rank-transforming and ridge-residualising both
sides on neighbour composition y within type. Pre-registered expectation:
A dense (composition-mediated co-occurrence), B sparse to empty; B's
survivors cross-referenced against the doc-09 §4.6 tally (PDGFB→PDGFRB if
it holds). Caption fixed in advance (§8.3). Build order: pure functions +
a planted unit test (composition-mediated pair → A hot / B cold; genuine
within-composition pair → both hot), then `lr_map.py` on the pinned
reference with timing; if fast, a reduced version (fewer perms) as a
TensorBoard figure at validation time.

**d_w = 2 result (3 seeds, sweep budget; `runs/dw2_s{0,1,2}`)** — the
prediction was only half right, and the half that failed is the finding.
Training reads at d_w = 2 vs the d_w = 6 controls (sweep3 κ = 0.1): recon
−7.2564 / **−7.1829** / −7.23 vs −7.2546 / −7.2566 / −7.2573 (nothing
lost; seed 1 lands in the strong family inside the 200-epoch budget), NMI
0.672 / 0.667 / – vs 0.667 / 0.636 / 0.674, cycle_w ≤ 0.003, probe and
mirror unchanged, KL_w 0.003–0.008. **cov(μ_w) eigen-fractions at d_w = 2:
[1.00, 0.00], [1.00, 0.00], [0.92, 0.08]** — w collapses to rank **one**
(rank 1.1 in one seed), exactly as it was rank 1–2 inside d_w = 6. The
realised shift agrees across seeds as before (shift-space overlap
0.98–0.99, dominant gene direction cosine 0.83–0.95); the matched-column
cosine rises only to 0.56–0.77, because one of the two columns is still a
null direction. So the seed variance of B was never about d_w: **at this
operating point the model uses one context program** (the tumour ↔ stroma
composition axis; a weak second appears in some seeds), and any B metric
that compares columns beyond the effective rank measures noise. Reporting
rule from here: programs = effective rank (V10), stability = shift-space
overlap (V11); the atlas' "few programs" becomes "one, sometimes two".
Open, and cheap to check on disk (CPU latents of the α_w-study runs):
whether the rank is a property of α_w = 0.1 (the KL_w pressure keeping w
on a 1-D prior manifold) or of the data — launched on `alphaw_0.02 /
0.03 / 0.05 / 0.07` (seed 0, type_z era) and `reference_best`.

**w's rank is an α_w effect, in the prior itself (2026-09-14).** CPU
latents of the α_w-study runs (type_z era, seed 0) and the type_only
seeds; eigen-fractions of cov(μ_w), of cov(m_ψ) (the prior field) and the
singular fractions of the realised shift μ_w·B, with r = components ≥ 1%:

| α_w | run | recon | cov(μ_w) | r | cov(m_ψ) r | shift r | deviation share/dim |
|---|---|---|---|---|---|---|---|
| 0.02 | alphaw_0.02 | −7.2518 | .71/.17/.07/.04/.01 | 4–5 | 4 | 4 | ≤ 1.7% |
| 0.03 | alphaw_0.03 | −7.2510 | .76/.15/.05/.04 | 4 | 4 | 4 | ≤ 1.2% |
| 0.05 | alphaw_0.05 | −7.2534 | .76/.21/.04 | 3 | 3 | 3 | ≤ 0.8% |
| 0.07 | alphaw_0.07 | −7.2586 | .79/.20 | 2 | 2 | 2 | ≤ 5.6% |
| 0.10 | reference_best (type_z) | −7.2586 | .90/.10 | 2 | 2 | 2 | ≤ 3.7% |
| 0.10 | type_only s0 / s1 / s2 | −7.25/−7.19/−7.23 | .98/.02, .83/.17, .98/.02 | 2 | 2 | 2 | ≤ 0.1% |
| 0.10 | dw2 s0 / s1 | −7.26/−7.18 | 1.0 | 1 | 1 | 1 | 0 |

Three readings. (1) **The rank rises monotonically as α_w falls** — 1–2
axes at 0.1, 3 at 0.05, 4 at ≤ 0.03 — and it rises **in the prior field
m_ψ and in the realised shift**, not through the deviation channel (the
per-cell deviation contributes ≤ 2% of variance per dim everywhere). So
the extra axes are context-field structure, not posterior noise; they are
consistent with the α_w-study's recon gain (+0.0076 at 0.03). (2) The
mechanism is the scout dynamic seen in the KL_w spikes: at high α_w the
posterior is pinned and the prior/B must discover context axes on their
own; at lower α_w the posterior explores (KL_w 0.09–0.2 during training)
and the prior chases it into more directions. (3) The "one programme"
reading is therefore a property of the α_w = 0.1 operating point, not of
the tissue — the earlier α_w verdict ("stays 0.1") was made when the
bistability was read as a hazard; with the strong family now understood
as the better optimum, α_w ≈ 0.03–0.05 with seed-ensemble + battery
selection is a live candidate operating point again. Pending before any
recommendation: do the extra axes reproduce across seeds (shift-space
overlap on alphaw_0.03_s1/s2, 0.05_s1/s2 — collecting), do they carry
likelihood when ablated, and are they context-driven (atlas drivers)?
A rank regulariser is not the answer (it would manufacture axes without
likelihood); the KL-floor (free bits per w dim) remains the one
principled candidate if the seed check passes — architect question.

**x̃ verdict — three seeds, every acceptance row (2026-09-14).** Reference
batteries re-run for s0 (full, post-era) and s2 so every seed has its
own control. Mean x̃ − reference over seeds, and the per-seed sign:

| row | Δ mean | per seed | criterion |
|---|---|---|---|
| niche-z residual over ℓ | **−0.006** | −0.011 / −0.013 / **+0.005** | "drops toward ℓ": 2 of 3, ~9% of the residual |
| cycle_z (battery / training) | −0.006 / −0.028 | 0.381→0.360, 0.440→0.456, 0.416→0.402 (battery) | inside the envelope, low side |
| held-out recon | 0.000 | ±0.001 | yes |
| NMI | −0.017 | −0.027 / −0.010 / −0.016 | not a criterion; consistent, the gate's direction |
| niche-w AUC / transport cf / slope | 0.000 / +0.001 / +0.02 | unmoved | yes |
| Moran-w | **−0.061** | 0.531→0.454, 0.628→0.642, 0.575→0.453 | down in 2 of 3 (the most seed-variable row) |
| planted gate | thresholds pass | NMI −0.03…−0.05 at κ = 0.2 | pass with caveat |

Reading, as pre-registered: the change is structurally right and costs
nothing in likelihood, but **on this slide at κ = 0.1 it is empirically
near-neutral** — the leak bias in the encoder's view is a minority of the
niche-z residual (the triage's benign-intrinsic reading holds for the
rest), the headline drop is small and not seed-consistent (2/3), and the
small consistent costs (NMI −0.017, training-time cycle_z −0.03, Moran-w
down in 2/3) are of the same size as the benefit. **Not green enough to
change the default.** `subtract_leak` stays available (default off) for
the per-cell applications — A4's powered planted world is where a
per-cell decontamination effect would show, and that is the test to run
before the flag is judged again. Sent to the architect as such; the 07
patch (§7.13) waits.

**Do the extra context axes reproduce across seeds? (2026-09-14)** Shift-
space overlap and dominant-axis cosines across the three seeds at each
α_w (type_z era for 0.03/0.05; type_only for 0.10):

| α_w | ranks (seeds) | axis-1 cosine | axis-2 cosine | axis-3 cosine | shift inside other seed's top-2 |
|---|---|---|---|---|---|
| 0.03 | 4 / 4 / 4 | 0.86–0.89 | 0.76–0.89 | 0.15–0.67 | 0.82–0.89 |
| 0.05 | 3 / 2 / 3 | 0.91–0.95 | 0.77–0.90 | 0.32–0.85 | 0.90–0.96 |
| 0.10 | 2 / 2 / 2 (≈1) | 0.89–0.95 | 0.50–0.77 | ≈ 0 | 0.98 |

Reading: lowering α_w reliably adds **one** robust context axis — at
0.03/0.05 a second axis is present at ≥ 15% of w's variance in every seed
and reproduces across seeds at cosine 0.76–0.90, whereas at 0.10 it is
seed-dependent (2–17%) and only half-reproducible (0.50–0.77). The third
and fourth axes at ≤ 0.03 do *not* reproduce (cosine 0.15–0.67) — seed-
specific optima, not tissue structure. Likelihood/z cost: at 0.05 recon
−7.2534 / −7.1857 / −7.2285 with cycle_z 0.454 / 0.478 / 0.470 (all
inside the 0.1 envelope); at 0.03 one seed drains cycle_z to 0.354 (the
known bad instance) — so 0.05 is the point where the second axis is
bought without a z cost. **Candidate re-calibration, for the architect:
α_w = 0.05 under type_only**, 3 seeds at the 500-epoch budget, battery
selection; acceptance pre-registered: two robust axes by shift-space
overlap (axis-2 cosine ≥ 0.75 across seeds), cycle_z ≥ 0.44, recon
inside the envelope, w-side rows intact, and atlas context drivers for
axis 2. Not launched — an operating-point change is the user's call. The
0.05 seed runs on disk are type_z era, so this is not yet a measured
number for the current architecture.

**Doc-09 §8 result on the pinned reference (`communication/lr_map.{json,
png}`; 79 s incl. the model load, the map itself ~10 s).** Effective rank
2 → 38 row genes (P0: F13A1/MRC1/TNXB/DPT… vs ESM1/MMP11/COL11A1/SLC2A1…;
P1: VEGFA/ADM/LCN2/IL6… vs COMP/SFRP4/COL10A1/ELN…); 30 columns after
dropping pairs with no receptor-eligible type (BMP4, C4A, IL6, POSTN,
TGFB3 fell out — the pair the §4.6 tally named, PDGFB→PDGFRB, did not
rank into the top 30 by Var(E) × prevalence). Under the doc's within-type
permutation null: **panel A 650 / 1140 cells significant, panel B 438 /
1140** — B does not collapse in *count*; it collapses in *effect size*
(max |ρ| 0.086 → 0.059, nothing ≥ 0.1 in B; |ρ| median 0.018 → 0.012),
and with ~60k validation cells and two spatially smooth fields the plain
permutation is anti-conservative (the A4/V8 lesson). Under a Moran-
preserving null (each cell takes the LR score of the same-type cell at a
random 500–1000 µm displacement; the shuffled field keeps its smoothness
and loses its alignment): **0 / 0 cells in either panel**. The structure
in A is the dominant w axis (stroma/macrophage vs tumour genes) against
pair scores that track tumour/stroma proximity (SEMA4C/D→PLXNB2,
CLDN1→CLDN1, CXCL12→CXCR4, DSC3→DSG2). Reading, honest: the map cannot
distinguish w-program/LR co-occurrence from domain-scale co-location —
which is the caption's message made stronger: the fig-6h genre reports
where things are, not signalling; DisCell's own w sits in the same genre
here. Reported with both nulls (permutation = the doc's significance;
shift = the stringent control, boxed cells = survivors, none). Seeds s0
and s2 to be re-run with the two-null script for the sign-agreement
note. Verdict on TensorBoard: fast enough (~10 s + one DB load), but as
a *training monitor* it adds nothing niche-AUC does not already track and
would show the same dense panel every event — kept as a per-run artefact
in the report instead, pending the user's call.

**§8 map, three seeds (two-null script; `communication/lr_map.json` per
run; s1 report regenerated with the section).** Permutation null: A 650 /
760 / 829 of 1140 cells, B 438 / 627 / 781 (max |ρ| A 0.09 / 0.08 / 0.16,
B 0.06 / 0.07 / 0.11) — the count-level "collapse" is absent in every
seed; the effect-size collapse holds (B's max |ρ| 0.6–0.7× A's). Spatial-
shift null: **0 / 0 in all three seeds, both panels.** Cross-seed sign
agreement among cells significant in both seeds (rows are re-selected
per seed, so the shared set is the same gene × same pair): A 91 / 85 /
78 %, B 89 / 96 / 86 % — where two seeds both call a cell, they agree on
its sign; which cells get called varies with the seed's row set and
effect scale (s2's map is ~2× hotter). Standing reading unchanged: the
map is domain-scale co-location under both w and LR; nothing survives a
Moran-preserving null. Kept as a per-run artefact + report section; not
wired into TensorBoard (user's call pending).

**§8 column rule tightened (user-caught: redundant and empty columns).**
The doc's rule — Var(exposure) × receiver prevalence, top 30 — produced
five FGF, three JAG1→NOTCH and two SEMA4→PLXNB2 columns (pairs sharing a
ligand share E and differ only in the receptor factor) and several all-
grey columns (raw Var(E) is sender proximity, which the within-type
Spearman on the LR score cannot see). Still no hand-picking, three rule
changes: rank by the variance of the **composition-residualised**
exposure (doc-09 §2's own step, within receiver types) × prevalence; **one
column per ligand** (its best-ranked receptor pair); a column needs
**≥ 2 receptor-eligible receiver types** to pool. Re-running the three
seeds and the s1 report.

**§8.2b received (architect, 2026-09-15) — the ladder.** Four items, all
built: (1) rows = the effective-rank **program coordinates** (per-gene
rows at rank 2 were duplicates within program-sign blocks), labelled by
each program's top ± genes; (2) the missing rung — three panels: **A′**
uncontrolled pooled Spearman over all validation cells (the SIMVI-
comparable view), **A** within-type, **B** composition-partialled — so
the figure shows that published-map structure is type composition; (3)
receptor side from the model's **clean rates ρ_g** (softmax(a(z) + Bw),
one sweep over the tiles for the receptor genes only; raw 0–2 counts per
cell starve the Spearman; mild circularity stated in the JSON and the
caption); (3b) the column rule v2 already in place (Var(Ẽ), one column
per ligand, ≥ 2 eligible types) — ratified; (4) the spatial-shift null
stays as the printed verdict line (boxed survivors). Re-running the three
seeds and the s1 report with this version; the earlier per-gene maps are
superseded.

**§8.2b ladder, three seeds (`communication/lr_map.{json,png}`, s1 report
regenerated).** Rows = the two effective-rank programs (P0 stroma/
macrophage F13A1/MRC1/TNXB ↔ tumour ESM1/MMP11/CLEC5A; P1 hypoxia/growth
TFRC/VEGFA/CYP24A1 ↔ matrix COMP/SFRP4/COL10A1); 30 columns, one per
ligand, now including PDGFB→PDGFRB and the PDGF family, GAS6/PROS1→AXL,
VEGFB→FLT1, WNT5A/B→FZD4. Cells (of 60) significant under the permutation
null / max |ρ| / surviving the spatial-shift null, per seed s1 / s0 / s2:

| rung | s1 | s0 | s2 |
|---|---|---|---|
| A′ uncontrolled, all cells | 57, 0.71, **29** | 55, 0.57, 20 | 56, 0.62, 22 |
| A within receiver type | 48, 0.28, 5 | 46, 0.34, 10 | 47, 0.26, 1 |
| B composition-partialled | 35, 0.16, **0** | 44, 0.14, 0 | 48, 0.14, 0 |

The ladder does what §8.2b predicted: uncontrolled, w's programs co-occur
with LR axes at |ρ| up to 0.7 and a third to a half of the cells survive
even the Moran-preserving null (that is the SIMVI-comparable view — real,
strong, and type composition: VEGFB→FLT1 / NECTIN3→NECTIN2 / APP→TNFRSF21
vs P0 is tumour-vs-stroma); within type the effect sizes drop 2–3× and
1–10 cells survive the shift null; after composition control nothing
survives it in any seed and max |ρ| is 0.14–0.16. The two rungs the doc
asked for now show the collapse in *effect size* clearly (0.7 → 0.3 →
0.15), while the permutation-null *counts* stay high at every rung — the
figure's own demonstration that a permutation null on ~60k spatially
smooth cells is not a null. Caption text as fixed in §8.3, extended with
the shift-null sentence. Verdict line for the report: "zero survivors at
autocorrelation-aware significance after composition control, in three
seeds". Descriptive figure; no communication claim; kept per-run + report
(not a TensorBoard monitor, per the user's pending call).

**d_w = 3 (two of three seeds) and the budget-matched d_w = 6 control
(2026-09-15).** All at the sweep budget (200 epochs, patience 20):

| d_w | seed | recon | NMI | cycle_z | cov(μ_w) eigen-fractions | r |
|---|---|---|---|---|---|---|
| 2 | 0 / 1 / 2 | −7.2564 / −7.1829 / −7.2255 | .672 / .667 / .667 | .405 / .386 / .485 | 1.00 / 1.00 / .92-.08 | 1 / 1 / 1 |
| 3 | 0 / 1 / (2 running) | −7.2613 / −7.1835 | .637 / .644 | .337 / .490 | .73-.27 / .82-.19 | 2 / 2 |
| 6 (sweep3) | 0 / 1 / 2 | −7.2546 / −7.2566 / −7.2573 | .667 / .636 / .674 | .413 / .469 / .470 | 1.00 / .96-.04 / .96-.04 | 1 / 2 / 2 |

Reading. (1) Likelihood and the z-side reads are indifferent to d_w in
{2, 3, 6} at this budget (the strong family appears once in each triple;
cycle_z spreads 0.34–0.49 across seeds regardless of d_w — the d_w = 3
s0 low value is inside the seed envelope of the d_w = 6 controls). (2) At
matched budget, **d_w = 6 is effectively rank 1** (second axis 0.1–4% of
w's variance — the 2–17% quoted earlier was the 500-epoch triple), **d_w
= 3 is rank 2 with a substantial second axis (19–27%)**, and d_w = 2 is
rank 1. So the fill is 1 of 2, 2 of 3, ~1 of 6: the model leaves one
direction empty at small d_w and, given six, concentrates on one axis.
That is not a tissue property — it is how the KL_w pressure and the
prior-chasing dynamic distribute a fixed budget of context variance over
the available dims; more empty dims, more concentration. (3) The two
d_w = 3 seeds share the 2-D shift space (overlap 0.97) but split it
differently (dominant-axis cosine 0.35), so "which two axes" is not
seed-stable at this budget either. Consequence for the atlas and for
the α_w question: the effective rank is a **(d_w, α_w, budget)** property
of the optimiser, and any "N programs" statement must name all three.
Cheapest next probe if the second axis matters to the paper: the 500-
epoch budget at d_w = 3 (does the second axis reproduce across seeds by
axis cosine, as it does at α_w = 0.05?). d_w stays 6 for now — nothing
is lost, and the spec's value needs no change to fix a reporting issue.

*Wording fix (user-caught):* in the §8 entries above, "cells of 60" means
heatmap **entries** (2 programs × 30 pairs), each a pooled Spearman over
all eligible validation cells (~60k) — not a cell count. Figure titles,
caption, log line and report section now say "entries".

**d_w = 3, third seed — the triple closes (2026-09-15).** `dw3_s2`: recon
−7.2339, NMI 0.642, cycle_z 0.437, cycle_w 0.002, probe −0.003; cov(μ_w)
= [.966, .034] — rank 2 by the 1% rule but with the second axis at 3%,
like the d_w = 6 controls, not the 19–27% of s0/s1. So the "d_w = 3 gives
a substantial second axis" reading is **two seeds of three**; the honest
statement is that the second axis's share is seed-variable at every d_w
(3–27% at d_w = 3, 0.1–4% at d_w = 6/200 epochs, 2–17% at d_w = 6/500
epochs) and only becomes seed-stable at α_w = 0.05 (15–21%). Cross-seed
axis cosines at d_w = 3: axis 1 0.35 / 0.89 / 0.64, axis 2 0.81 / 0.72 /
0.58 — the 2-D shift plane is shared (overlap 0.96–0.98) and split
differently per seed. Conclusion unchanged: d_w stays 6; the effective
rank is a (d_w, α_w, budget) property; report programs at the effective
rank with the second axis's share and its cross-seed cosine, never a
count. d_w ablation closed; the six runs stay on disk
(`runs/dw{2,3}_s{0,1,2}`), no metrics file beyond `metrics.json`.

### d_w = 8 — closing the ablation from above (motivation, 2026-09-15)

User request: bracket the d_w question on the large side too. Prediction,
pre-registered: at α_w = 0.1 and the sweep budget the fill is set by the
operating point, not the box — d_w = 8 will use 1–2 of 8 (second axis
0–5%, as d_w = 6 at this budget), recon / NMI / cycle_z / cycle_w / probe
/ mirror inside the d_w ∈ {2, 3, 6} envelope, shift-space overlap across
seeds ≥ 0.95. If the second axis's share instead grows with d_w, the
"(d_w, α_w, budget) optimiser property" reading is wrong and d_w matters
in its own right. Design: `dw8_s{0,1,2}`, `--d-w 8`, 200 epochs / patience
20, defaults otherwise; read-outs as for d_w = 2/3 (metrics.json + CPU
latents: cov(μ_w) spectrum, shift-space overlap, axis cosines). Not an
operating-point change; ~40 min per seed.
**Embedding result** (`embeddings/egomask_ego_v1.pt`, 1.8 GB; `logs/embed.log`):
launched 12:19 on GPU 0 once `xtilde_s2` finished, 1,157,637 × 384 in
16,382 s (4 h 33 min, 71 cells/s cumulative — below lung's 113 because the
decode is single-threaded and the machine sat at load 35–53 under the
other session's d_w fits). Every cell embedded, all finite, no zero rows,
row-norm median 53.9; metadata identical to the ovarian and lung files
(kronos1, 256 px @ 0.5 µm/px, ego 25 µm, markers 4/442/505/130). The
loader pairs it with the bundle for 100% of cells in 16 s. The slide is
now at parity with lung: bundle + embeddings, graphclust labels, no runs.

**d_w = 8 result (three seeds, `runs/dw8_s{0,1,2}`) — prediction half
wrong, in an informative way.** recon −7.2645 / **−7.1858** / −7.2401
(strong family once again), NMI 0.662 / 0.669 / 0.658, probe and mirror
at reference; cov(μ_w) = [.86, .14], [.91, .09], [.89, .11] — **rank 2 of
8 with a 9–14% second axis in every seed**, and the two axes reproduce
across seeds (axis-1 cosine 0.83–0.94, axis-2 0.70–0.92, shift-space
overlap 0.99). Predicted was 0–5% as at d_w = 6 / 200 epochs; the fill
is still 2, as predicted, but the second axis's share is 3× larger and
seed-stable at d_w = 8. Full bracket at the sweep budget, second-axis
share by seed: d_w 2 → 0 / 0 / 8%; 3 → 27 / 19 / 3%; 6 → 0.1 / 4 / 4%;
8 → 14 / 9 / 11%. Not monotone in d_w and, with three seeds per cell,
consistent with "the share is seed-noise on top of a weak dependence on
the box" — the optimiser-property reading (fill ≪ d_w, set by α_w and
budget) stands; the box is not *irrelevant* to the share. Costs at
d_w = 8, small but in the wrong direction: cycle_z 0.428 / 0.356 / 0.367
(d_w = 6 controls 0.41–0.47) and **cycle_w 0.020 / 0.014 / 0.008** (every
other type_only run ≤ 0.010; s0's 0.020 is the highest on record) — a
hint of intrinsic state seeping into the extra context capacity. **Verdict
for the finalized analysis: keep d_w = 6.** Nothing in likelihood or the
w-side favours 8, the z/w separation reads are slightly worse, and the
lever that makes a second context axis seed-stable *without* that cost is
α_w (0.05: 15–21%, cosine 0.77–0.90). Ablation table for the paper:
d_w ∈ {2, 3, 6, 8} × 3 seeds, all on disk. Watch item: cycle_w at d_w ≥ 8.

### FF slide, first fit: `reference_graphclust` on `xenium_prime_human_ovary_ff` (motivation, 2026-09-15)

Motivation: the transfer test on the third slide — fresh frozen, 8× deeper,
annotation-free (graphclust, 38 clusters + Unassigned) — run with the lung
protocol and two per-dataset decisions, both derived rather than tuned:

- **α_z = 0.0007** by the §5 rule, 1/median ℓ = 1/1,401. The rule is the
  ELBO, not an empirical fit: `multinomial_loglik` divides each cell's
  reconstruction by ℓ_i (nats per count, O(1)) while the KLs are per cell,
  so 1/ℓ̄ is the weight that restores the ELBO at the slide's depth — the
  same operating point as ovarian (0.007) and lung (0.004) in ELBO units.
  An "in-between" value would be a β-VAE at β ≈ 4, the direction the
  recovery gate showed collapses z. α_w 0.1, α_a 0.3, κ 0.1, ω 1 unchanged;
  α_w now sits 140× above 1/ℓ̄ (ovarian 14×), i.e. further into the
  prior-pinned regime every result lives in — read KL_w, change nothing.
- **Resident counts as int16.** Measured footprint on this slide: 512
  tiles, 1,370,917 nodes incl. rings (×1.18 seeds), float32 x 27.4 GB +
  Φ 2.1 GB — over a 24 GB card (ovarian: 128 tiles, ×1.14, 9.5 + 0.7 GB,
  matching the observed 13 GB). Tile size does not change this: every tile
  is resident. Counts are integer-valued with max entry 856, so int16 is
  exact: `Trainer._to_device` stores int16 (with a < 32768 assert),
  `_forward_kwargs` casts the tile to float32, `_step` reads the cast tile
  for the loss; every post-hoc module already goes through
  `_forward_kwargs`. Test: `test_resident_counts_are_int16_and_cast_per_tile`.
  Expected on device ≈ 13.7 + 2.1 + working ≈ 18–20 GB.
- Budget 500 / patience 40, seed 0, tiles 4096 → 435 train / 77 val tiles;
  per epoch 3–4× the dev slide (512 vs 128 tiles) → ~1.5–2 h.

Pre-registered reads, as on lung: recon, NMI over the 39 classes, probe
ΔCE vs its floor, mirror vs control, KL_z / KL_w per dim from
`metrics.json`; then the §§3–4 battery (Moran |I| w vs z, niche AUC
w / z / ℓ / floor, cycle pooled z vs the 50-PC frame with w ≈ 0). Pass =
the ovarian/lung allegiance shape reproduces. A collapsed z (NMI ≪ 0.6,
KL_z → 0) or an opened w channel (KL_w ≫ 0.002/dim) triggers the
×2 / ×½ α_z bracket at the 200-epoch budget; nothing else is tuned.

### Fourth intake: GSE315411 — a two-section pediatric-lung TMA, for a held-out-slide protocol (motivation, 2026-09-15)

Motivation: every validation so far is a random 15 % of spatial tiles
*within* one slide. GSE315411 gives the missing design — the same tissue
measured twice: TMA `PDLTMA006` (17 donor cores, pediatric lung disease)
sectioned at 10 and 11 and run as two Xenium Prime 5K slides. Plan (a) of
the three discussed: **two datasets, one label vocabulary; train on one
slide, evaluate the checkpoint on the other**. The obstacle is the label
space — the model's `t` is `sorted(unique(label))` per bundle, so a
cross-slide evaluation needs identical class sets with identical
meanings on both slides, which per-slide graphclust cannot give. Hence
the shared-label pipeline below, run *before* any bundle.

Slide facts (`experiment.xenium`, `metrics_summary.csv`, `cells.parquet`,
the series Seurat object `GSE315411_prime_slides_2025_11_20.rds`):

- GSM9427181 "prime solo" = section 11, 1,100,358 cells, median 115
  transcripts / 100 genes per cell; GSM9427182 "prime dual" = section 10,
  run with both the Prime 5K and the V1 lung panel, 1,115,068 cells,
  median 100 / 88. Panel `hAtlas_v1.1`, 5,001 genes — the base Prime 5K,
  identical to lung and FF. FFPE, chemistry v2, xenium-3.3.0.1, bundle 5.2,
  `Xenium Multi-Tissue Stain` (interior 86.7 %, boundary 12.2 %, nucleus
  expansion 1.0 % — far fewer boundary-stain cells than the 10x slides'
  ~28 %). Four `morphology_focus` channels, 0.2125 µm/px; `analysis/` as a
  directory (graphclust present). **Shallowest slides yet** (ovarian FFPE
  178, lung 242, FF 1,401): by the §5 rule α_z = 1/ℓ̄ ≈ 0.0087 (solo) /
  0.010 (dual) — to be set at training, noted now.
- Donors: 17 cores, every one on both slides, per-core cell counts within
  2 % (e.g. PDL034D 153,822 / 156,158; PDL033T 7,679 / 7,627); ~3.4k /
  2.0k cells per slide lie outside every core (Seurat `sample = dropped`).
  Assignment is a per-cell lookup in `series/cell_stats/` — Xenium
  Explorer selection exports, `Cell ID,Cluster,Transcripts,Area (µm^2)`
  under two `#` comment lines. Verified: the solo bundle's ids match the
  `*_prime_solo_*` files and the dual bundle's the `*_prime_V1_*` files
  (100 % overlap on PDL061; the `prime_with_V1_segmentation` / `V1_*`
  variants are other segmentations of the same sections, 0 % overlap).
- **No author cell types anywhere**: the Seurat object carries only
  `slide_type` and `sample` (donor). Labels have to be made.

**Shared-label pipeline** — scripts received from the user, adapted, kept
at `scripts/annotate_gse315411/` (outside the package; its own venv there,
an overlay on the `gaston-mix` conda env for scvi-tools 1.4.0 + torch
2.5.1/cu124, plus igraph, leidenalg, pyarrow), working directory
`<GSE315411>/annotate/`. Four stages:

0. `00_build_query.py` — one AnnData from both bundles: real genes only
   (control probes/codewords dropped), gene panels intersected, cell
   metadata joined, donor from cell_stats, QC `min_counts 10` /
   `min_cells_per_gene 5`, out-of-core cells **kept** with donor `NA`.
   Adapted: `assign_donors` reads the `Cell ID` column through a
   hand-rolled header skip — two of the 34 files (PDL026A, PDL085A on the
   solo slide) have their first `#` line *quoted* by Explorer, which
   pandas' `comment='#'` does not recognise; the first attempt (pandas
   comment handling, skip-on-failure) left those two cores as `NA`
   (143,774 solo cells) and was discarded before any downstream stage;
   a file without a cell-id column now aborts. The received version would
   have produced all-`NA`.
1. `01_embed_cluster.py` — joint scVI (`batch_key = slide`, n_latent 30,
   2 layers, NB) + Leiden at resolution 2.0 (over-cluster on purpose;
   stage 3 merges by name). scvi's epoch heuristic gives 4 epochs at 2.2 M
   cells; set **30 epochs with early stopping (patience 10)** instead —
   the one budget decision, made before the run.
2. `02_transfer.py` — scANVI trained *from scratch on the reference
   restricted to the panel genes* (the published HLCA scArches model
   takes 2,000 HVGs and `prepare_query_anndata` zero-pads the ~75 % a
   Xenium panel lacks — confident labels from a broken mapping), then
   scArches surgery onto the query and soft predictions. Two references,
   both from CELLxGENE as h5ad (no Seurat conversion): **HLCA core**
   (584,944 cells, 5.9 GB, `ann_finest_level`, batch `dataset`,
   stratified subsample 300k) as primary and **LungMAP CellRef 1.0**
   (347,970 cells, 4.2 GB) as the second opinion. Adapted: CELLxGENE
   h5ads index genes by Ensembl id with symbols in `var['feature_name']`
   and raw counts in `.raw.X` — added `--ref-gene-col` / `--ref-use-raw`;
   the ≥ 800 shared-gene abort stays. Query epochs 30 (early stopping),
   not the received 100, for the same 2.2 M-cell reason.
3. `03_name_and_report.py` — the final label is **per Leiden cluster**
   (mean soft probability, argmax, gates `mean_prob ≥ 0.5` and majority
   ≥ 0.5; failing clusters become `Unknown_<k>` rather than the nearest
   adult type — HLCA/CellRef are adult/healthy, this TMA is pediatric
   disease), plus the cross-slide report.
4. (new, small) write `<sample>_cell_groups.csv` (`cell_id, group, donor`)
   beside each outs directory so the standard bundle picks the labels up
   as `cell_group` (the curated-label path, default label); `donor` rides
   along through a one-column extension of `read_cell_groups`.

Pre-registered reads on the labels, before any DisCell run: (i) every
Leiden cluster draws ≥ 5 % of its cells from each slide — a cluster below
that is a dual-chemistry artefact and is reported, not used; (ii) global
composition JSD between the slides (serial sections of the same cores)
< 0.05, per-donor JSD listed, a high donor is checked on the H&E before
the pipeline is blamed; (iii) the label vocabulary is **identical on both
slides** — required by design (a), checked explicitly; (iv) HLCA-vs-CellRef
ARI/NMI reported, `Unknown_*` count reported; nothing is tuned to move
these. Cells failing QC (< 10 transcripts) get no label and become
`Unassigned` in the loader (the P3 fold), count reported.

Then, in order on **one GPU (GPU 1; GPU 0 stays with the FF fit)**:
references download (network only) → stage 0 (CPU) → stages 1–2 (GPU 1)
→ stage 3 (CPU) → cell_groups + bundles for both slides
(`data/raw/gse315411/GSE315411_PDLTMA06_11_prime_solo` → id
`gse315411_pdltma06_11_prime_solo`, and `…_10_prime_dual`; symlinks onto
the outs directories, `--sample` on the symlink so the id is the readable
name while `xenium_dir` resolves to the real bundle) → `egomask_ego_v1`
embeddings for both (~3 h each at the lung rate). Training and the
cross-slide evaluation path (`load_run` + `assemble` on the other
dataset under the same vocabulary) come after and get their own entry.

### GSE315411 shared labels — the scANVI leg fails, pseudobulk naming replaces it (2026-09-15, same day)

Stages 0–1 as planned: 2,102,535 cells after QC (min 10 transcripts:
52,406 / 60,485 dropped = 4.8 % / 5.4 % per slide — they will be
`Unassigned`), all 17 cores on both slides, 3,405 / 2,104 cells outside every
core (`NA`). Joint scVI (30 epochs, early-stopped) + Leiden 2.0 → **44
clusters, every one drawing ≥ 5 % from each slide** (read i ✓). A
reference-free check (`diag_cluster_markers.py`, top log-fold genes per
cluster from the query counts alone → `annotate/cluster_markers.csv`) finds
the clusters textbook-clean: CD163/MRC1 and MARCO macrophages, CD3E/MS4A1
lymphocytes, MPO/DEFA1 neutrophils, FOXJ1/DNAH ciliated, AGER/HOPX AT1,
ABCA3/LAMP3 AT2, PDGFRB/NOTCH3 pericytes, PROX1/FLT4 lymphatics, mast,
plasma, GRP/ASCL1 neuroendocrine, MKI67 proliferating, plus tissue the TMA
carries beyond alveolar lung — COL2A1 cartilage, MYH1/MYH2 skeletal muscle,
HBG1 erythroid, ITGA2B megakaryocytes, MPZ/SOX10 Schwann — and three
low-depth clusters (17: 154k cells at median 20 transcripts; 11: 113k at 33;
8: 32k at 27).

**The scANVI transfer (stage 2, HLCA) is a failed instrument at this
depth.** Per-cell calls: 60 % of all cells "Alveolar fibroblasts" (1.26 M),
immune classes an order of magnitude too small, median max-probability
1.000, 1.1 % below 0.5. Per cluster: the majority HLCA label is "Alveolar
fibroblasts" in **26 of 44 clusters** — including the macrophage (0), T
cell (6), neutrophil (14), pericyte (19), Schwann (31), lymphatic (32),
erythroid (34), megakaryocyte (40), cartilage (41), skeletal muscle (10)
and one AT2 (12) cluster — at mean max-probability 0.92–1.00, i.e. every
one would pass the stage-3 gates (`mean_prob ≥ 0.5`, majority ≥ 0.5). The
frozen scArches classifier, trained on ~10³-UMI scRNA-seq, collapses
~100-transcript query cells onto one default class and reports it with
full confidence; the pre-registered gate cannot see this. The CellRef
leg collapses harder still: 97 % of cells → CAP1 (capillary EC), 42 of 44
clusters; the two scANVI label sets agree with each other at ARI 0.026 /
NMI 0.068. Stage 3's scANVI-based report is kept as the record of the leg
(`report_hlca.txt`, `labels_{hlca,cellref}*.csv`); stage 4 now refuses to
hand labels over without `--accept`, so nothing from it reached an outs
directory.

**Replacement — pseudobulk correlation** (`02b_pseudobulk.py`,
`03b_name_pseudobulk.py`): each Leiden cluster's counts summed
(0.39–27 M transcripts per profile, the depth problem gone), log1p(CP10k)
on the shared genes, Spearman r against each reference type's mean profile
over the 1,500 most type-variable reference genes; HLCA core
(`ann_finest_level`, 60 types ≥ 50 cells, 4,864 shared genes) and CellRef
1.0 (`celltype_level3`, 45 types, 4,718 genes). Gate, fixed in the script
before its first run: (1) cluster median depth ≥ 40 transcripts, (2) the
two references' best types agree at lineage level 1
(Immune/Epithelial/Endothelial/Stroma≡Mesenchymal), (3) name = HLCA finest
of the best type, CellRef reported beside it; (4) marker overrides, behind
a flag, for identities neither reference carries.

Result (`labels_pb_clusters.csv`, `report_pb.txt`): 39 of 44 clusters
named, **5 Unknown_** — 8, 11, 17 by depth (299k cells) and 3 (HSPA6/
SERPINE1 stress signature; HLCA says fibroblast, CellRef capillary) and 23
(SLC6A4/GJA5; pericyte vs capillary) by lineage disagreement — 20.1 % of
cells in Unknown_* in total, the three depth clusters being 14 %. The two
references agree at lineage level on 42/44 and on the fine name (up to
vocabulary) on most; both are lineage-right but wrong on the
out-of-reference clusters (erythroid → "EC general capillary", cartilage →
"Peribronchial fibroblasts"), which is what the overrides are for: 10
Skeletal muscle, 14 Neutrophils (CellRef's own call, r 0.72, margin 0.15;
HLCA core has none), 31 Schwann cells, 34 Erythroid, 40 Megakaryocytes,
41 Chondrocytes. **Reads: 32 classes, vocabulary identical on both slides
(iii ✓); composition JSD between slides 0.0005 (ii ✓, threshold 0.05);
per-donor JSD 0.0002–0.0029, mean 0.0011 — no torn core.**

Fine-level calls the markers contradict, left for the user's decision
(the two references disagree and a wrong one has downstream teeth — the
doc-08 vasculature landmark is endothelial ∪ pericytes): cluster 2
(103k cells, WNT2/RSPO2/FABP7 = alveolar fibroblast; HLCA "Pericytes"
r 0.699 over "Alveolar fibroblasts" 0.660, CellRef AF1); cluster 9
(PLVAP/SELP/SELE = venous EC; HLCA "EC arterial", CellRef SVEC); cluster 6
(mixed CD3E/MS4A1/GZMB lymphocytes, "NK cells"); cluster 1 (MFAP5/PI16 =
adventitial; "Peribronchial fibroblasts", HLCA second Adventitial).
Bundles wait for that decision; embeddings (label-independent) start on
GPU 1 the moment the label pipeline exits (`logs/embed_after_labels.sh`).

### GSE315411 — curated (annotator-style) names, per the user's steer "as close to hand-labelled as possible" (2026-09-15, pending acceptance)

Protocol, written as it was done: names come from each cluster's own
markers first — log-fold genes against the rest (`cluster_markers.csv`),
a canonical-marker table (`diag_canonical_markers.py` →
`canonical_markers.csv`; note the Prime 5K panel lacks PTPRC, COL1A1,
ACTA2, SFTPC and all keratins, so fold-change lists carry the weight) and
the donor distribution — with the two references' pseudobulk calls as a
guide, not a verdict. Clusters that mixed lineages were subclustered on
the scVI latent (`diag_subcluster.py`, Leiden 0.3 within the cluster):
6 → cytotoxic T/NK (21k), CD4 T (9k), proliferating T (1.2k), **B cells**
(6.1k: MS4A1 CD19 PAX5), stressed T (0.6k); 23 → gCap endothelium (33k:
SLC6A4 HPGD PECAM1), alveolar fibroblasts (27k: TCF21 PHEX), adventitial
PRG4⁺ fibroblasts (13k); 3 → capillary EC (19k), fibroblasts (6k),
CXCL13⁺ fibroblasts (15k), activated EC (10k) — the HSPA6 heat-shock
state that defined cluster 3 (PDL029A, PDL042D) is thereby spread over
identities, which is where a state belongs for a model whose z is meant
to carry it; 27 → capillary EC (17k), PTX3⁺ activated fibroblasts (43k),
adventitial fibroblasts (5k); 36 → two ductal-like distal-epithelium
subunits kept under one name. Several clusters are single-donor disease
states (16: 84 % PDL032D; 22: 83 % PDL026A; 27: 84 % PDL040A; 36: 89 %
PDL034D — 60 % of that core; 42: 99 % PDL055T; 24: 77 % PDL072): scVI
corrected slide, not donor, and a pediatric lung-disease TMA has
donor-specific biology; they are named by identity with the state as a
qualifier in the evidence column, never by donor. Where markers and HLCA
disagree the markers win: 2 (WNT2 RSPO2 FABP7 TCF21) → Alveolar
fibroblasts, not Pericytes; 9 (PLVAP SELP SELE) → EC venous, not
arterial; 26 (APLN CA4 F2RL3) → EC aerocyte capillary; 1 (MFAP5 PI16) →
Adventitial fibroblasts. Identities absent from both references get
their marker name (Skeletal muscle, Chondrocytes, Schwann cells,
Erythroid, Megakaryocytes, Neutrophils). The three low-depth clusters
(8, 11, 17; median 20–33 transcripts, 299k cells = 14.2 %) are
`Unassigned`, one class, which the loader also gives the 5.1 % QC-dropped
cells — ~19 % Unassigned in total, the price of the shallowest slides in
the project. The full unit → name → evidence table is
`scripts/annotate_gse315411/curated_names.csv` (56 units), applied by
`03c_curate.py` → `annotate/labels_curated.csv`, `labels_curated_units.csv`,
`report_curated.txt`.

Result: **35 classes, vocabulary identical on both slides; cross-slide
composition JSD 0.0005; per-donor JSD 0.0002–0.0029 (max PDL047T), mean
0.0011; smallest class 1,191 cells on a slide (Mast cells).** Largest:
Unassigned 299k, Alveolar fibroblasts 291k, EC general capillary 231k,
AT2 122k, Distal epithelium 94k, Multiciliated 92k. Awaiting the user's
acceptance before stage 4 hands the labels to the bundles.

**Accepted by the user (2026-09-15 14:20), Unassigned kept as one class.**
Before handing over, the low-depth cells were characterised: real
segmented objects (median area 47 vs 55 µm², nuclei present) at 4.5× lower
transcript density (0.6 vs 2.7 per µm², median 24 vs 136 transcripts),
and **core-level**: PDL097 100 % low-depth (the whole core, 30k cells),
PDL085A 60 %, PDL031A 32 %, PDL026A 26 %, most other cores 2–10 % — tissue
/ RNA quality of particular cores, not scattered cells. Cells at 20–40
transcripts split ~half/half between named and low-depth clusters, so the
low clusters are depth *and* no lineage signal. Other slides had 0–0.5 %
Unassigned; here ≈ 19 % (299k + 113k QC-dropped). Kept in the graph as
neighbours rather than dropped (holes would distort the leak model); a
kNN rescue in the scVI latent was offered with a thinning calibration and
declined. `04_write_cell_groups.py --tag curated --accept` wrote
`GSE315411_prime_solo_cell_groups.csv` / `GSE315411_prime_V1_cell_groups.csv`
(`cell_id, group, donor`) beside the outs directories; bundles launched
(`logs/bundle_both.sh`), the dual-slide embedding still running on GPU 1
(solo done 10:50–13:34, 2 h 44 min).

**Solo bundle** (`gse315411_pdltma06_11_prime_solo/bundle/`, 703 s, 2.4 GB):
1,100,349 cells (9 degenerate polygons dropped) × 5,001 genes; contact
(1 µm) 1,705,421 edges, mean degree 3.10, isolated 6.2 %; Voronoi (clip
30 µm) 3,229,397 edges, mean degree 5.87, isolated 0.1 %, median shared
wall 6.29 µm, apposed wall 52.6 % nonzero (lung 42 %, FF 62 %). Loader
check: `label_key = cell_group`, **K = 35**, Unassigned 193,477 (141,073
low-depth + 52,406 QC-dropped, folded onto the existing spelling), `donor`
in obs (17 cores; NaN for the 3,405 out-of-core and the QC-dropped cells),
`egomask_ego_v1.pt` (1,100,349 × 384) found for 100 % of cells, 951
isolated on the Voronoi graph. Median transcripts 115 → α_z = 1/ℓ̄ ≈
0.0087 for the first fit (dual: 100 → 0.010). Dual bundle building;
dual embedding at ~120 cells/s.

### GSE315411 — the `strong3` variant: three strong-disease-state parenchyma cores (motivation, 2026-09-15)

The TMA is heterogeneous (per-core composition, this day's entry above):
eight alveolar-parenchyma cores, three airway/bronchus cores (no AT1/AT2;
cartilage, skeletal muscle, goblet metaplasia), two malformation-like
cores (PKHD1/HNF1B/CFTR epithelium), one failed core, and three
parenchyma cores dominated by a single activated population — **PDL026A**
(EC activated 34 %: ESM1/ANGPT2/SELE/IL6 inflamed endothelium, plus
inflammatory fibroblasts), **PDL040A** (PTX3⁺/THBS1⁺ activated
fibroblasts 36 %), **PDL018D** (POSTN⁺/TNC⁺ myofibroblasts 27 %, fibrotic).
User's choice: model these three. They are the cores where the question
DisCell asks is sharpest — an activated state that is either intrinsic to
the cell (z) or imposed by an inflamed / fibrotic neighbourhood (w), with
the leak channel in between — and they are alveolar tissue, so the
cell-cycle and niche instruments transfer from ovarian/lung unchanged.
Not chosen: adding PDL029A/PDL031A (inflamed parenchyma) — it would raise
the set to 335k cells per slide but with 37 % Unassigned (PDL031A is a
third low-depth) and dilute the states.

Facts of the set (from `labels_curated.csv`): 214,294 (solo) / 214,917
(dual) cells; all 35 classes present on both slides, so K is identical by
construction (three classes are near-empty: Basal 17/38, Goblet 8/13,
Plasma 11/7 — they stay as types, nothing is relabelled); Unassigned 15 %;
top classes EC activated 47.6k, Activated fibroblasts 39.6k,
Myofibroblasts 36.8k, Multiciliated 31.2k, AT2 29.6k, EC general
capillary 20.6k, Interstitial macrophages 19.1k. ~52 tiles of 4096 per
slide → ~8 validation tiles at 15 %.

Implementation, minimal: a `--donors` option on `discell.preprocess`
(bundle stage) that keeps the cells whose `donor` (from the cell_groups
file) is in the list, applied inside `load_xenium_sample` after the
labels attach and before the graphs — the cores are 3 mm punches with
gaps far beyond the 30 µm Voronoi clip, so a per-core subset has the same
graphs the full slide would give those cells. Bundle variant `strong3`
in both datasets; the embeddings file is per dataset and covers every
cell, so it serves the variant unchanged. Fit protocol as for lung/FF:
`--variant strong3`, α_z = 1/median ℓ of the subset (set when computed),
α_w 0.1, α_a 0.3, κ 0.1, 500 epochs / patience 40, seed 0, `type_only`;
pre-registered reads as in the FF entry (recon, NMI, probe ΔCE vs floor,
mirror, KL_z/KL_w per dim; then Moran / niche / cycle rows). The
cross-slide evaluation path gets its own entry once the fit exists.

**Narrowed to one donor (user, 2026-09-15 15:05): `pdl018d`.** Single-core
candidates among the three: PDL026A 89k cells per section but **26 %
Unassigned** (a quarter of the core is low-depth), K = 34; PDL040A 56k,
13 % Unassigned, seven classes under 20 cells; **PDL018D 69k per section
(70,003 / 68,840), 3.7 % Unassigned, K = 35 on both sections**, the
fibrotic core — Myofibroblasts 36k (POSTN TNC ASPN ELN COMP), EC aerocyte
capillary 15k, Multiciliated 15k, AT2 11k, Adventitial fibroblasts 11k,
Neutrophils 7k, Interstitial macrophages 7k, Pericytes 6k. Chosen for RNA
quality and lineage diversity around one strong state. Three classes are
near-empty there (Basal 11/21, Goblet 6/8, Plasma cells 1/1); they stay
in the vocabulary — `TypeCovariances` drops types below 10·dim effective
count and every read has a ≥ 200 / ≥ 1,000-cell guard, so they carry
nothing and break nothing. The `strong3` set is not built. Image
embeddings: the standard `egomask_ego_v1` arm (KRONOS v1, 256 px at
0.5 µm/px, the 25 µm disk over the centre cell zeroed) is per dataset
and covers every cell of a slide, so the variant reuses it. `--donors`
implemented as planned (`discell.preprocess --donors PDL018D --variant
pdl018d`, recorded in `pdl018d_params.json`; guard test
`test_donor_subset_needs_a_donor_column`, suite 7/7).

### Cross-slide evaluation: apply a fit to the other section (motivation, 2026-09-15)

What design (a) needs and the code base lacks: every evaluation so far
runs on the fit's own dataset (`load_run` rebuilds the run's data;
`Trainer.evaluate` reads its own val tiles). Add one module,
`discell/model/crossslide.py`: `--dataset A --run R --eval-dataset B
[--eval-variant]` loads R's best weights (`validate.load_run`), assembles
B under R's `label_key`, `variant` (or `--eval-variant`), embeddings name,
tile size and seed, **asserts B's sorted type vocabulary equals A's**
(the one-hot `t` and every per-type table are index-aligned; a mismatch
is refused, not remapped), builds a `Trainer` on B and swaps in R's
model, then reports (i) the same reads as `metrics.json` — recon on B's
val tiles, NMI, probe ΔCE vs floor, mirror, cycle rows, KL_w per dim —
computed exactly as during training but on B, and (ii) **held-out
reconstruction over every tile of B** (the whole section is held out, so
the 15 % split is only the probe's block-CV), beside R's own
`metrics.json` numbers for the same-section comparison. Output
`runs/R/crossslide/<B>.json`. Nothing about the model or the readers
changes; `median_counts`, `p_t` and the covariance/adversary state stay
R's (they are evaluation-irrelevant or part of the trained model).
Pre-registered read: the held-out-section recon and the z/w allegiance
rows on B should sit within the seed envelope of the same-section values
(recon ±0.06, the seed spread seen on ovarian); a larger gap is the
section-to-section (and, for dual vs solo, chemistry) generalisation cost,
reported as such. Test: a run evaluated on its own dataset reproduces its
own `evaluate()` numbers (synthetic smoke fit).

**Dual bundle** (`gse315411_pdltma06_10_prime_dual/bundle/full`, 729 s,
2.3 GB): 1,115,054 cells (14 degenerate polygons dropped) × 5,001;
contact 1,739,659 edges, mean degree 3.12, isolated 6.0 %; Voronoi
3,271,738 edges, mean degree 5.87, isolated 0.1 %, median wall 6.28 µm,
apposed 53.0 %. Loader: K = 35, Unassigned 218,681 (158,203 low-depth +
60,483 QC-dropped), 1,095 isolated; **type vocabularies of the two full
bundles identical**. The two full bundles were built from cell_groups
files that carried rows only for annotated cells, so their QC-dropped
cells have `donor = NaN` (label unaffected: the loader folds them to
`Unassigned`). Stage 4 was then changed to write a row for **every** cell
of `cells.parquet` — `Unassigned` for the 52,406 / 60,485 QC-dropped
cells, donor from the cell_stats files directly — so that a per-donor
variant keeps those cells as neighbours the way the full slide does; the
`pdl018d` variants are built from the rewritten files (a first build
from the old files, 68,840 solo cells = annotated only, was discarded).

Two fixes on the way: `--donors` first raised inside `load_xenium_sample`
("truth value of a Series") because the cell_groups block assigned a
local `donors` Series that shadowed the new parameter — invisible to the
guard test (the ovarian file has no donor column); renamed, and a
positive-path test on a synthetic 27-cell Xenium directory with three
donors (`_write_fake_xenium`, `test_donor_subset_keeps_only_those_cores`)
now covers the subset, the label/donor pass-through and graph alignment.
`discell/model/crossslide.py` implemented as motivated (`apply_fit` +
`evaluate_on`, CLI `--dataset --run --eval-dataset [--eval-variant]`);
test `test_apply_fit_on_own_data_matches_own_evaluate` (a smoke fit
applied to its own tiles reproduces its `evaluate()` reads).

**`pdl018d` bundles** (104 s each, ~210 MB): solo 69,422 cells (68,840
annotated + 582 QC-dropped as Unassigned), Voronoi 205,929 edges, mean
degree 5.93, 0 isolated at build (20 after the loader's edge rules),
median wall 7.24 µm; dual 70,757 cells, 209,862 edges, 5.93, 7.21 µm.
Loader: **K = 35 on both, vocabularies identical**, Unassigned 2,899 /
3,370 (4.2 % / 4.8 %), embeddings found for 100 % of solo cells. The
core is deep: median **279 (solo) / 253 (dual) transcripts per cell**
against slide medians of 115 / 100 — PDL018D is high-quality tissue, and
α_z = 1/ℓ̄ = **0.0036** (solo), the lung operating point (0.004) within
rounding.

### First fit on the core: `reference` on `gse315411_pdltma06_11_prime_solo` / `pdl018d` (motivation, 2026-09-15)

Protocol as for lung/FF, two size adaptations stated: **α_z = 0.0036**
(rule above); **tile 2048 cells** instead of 4096 — 69k cells give 17
tiles of 4096, i.e. 2–3 validation tiles and five block-CV folds of 3–4
tiles; 2048 gives 34 tiles, 5 validation, ~7 per fold. Nothing in the
model depends on the tile (two-hop rings are exact), only the split and
the fold granularity. Everything else at the defaults: `type_only`, α_w
0.1, α_a 0.3, κ 0.1, ω 1, d_z 20, d_w 6, 500 epochs / patience 40, seed 0,
`egomask_ego_v1`. Command: `uv run python -m discell.model.train --dataset
gse315411_pdltma06_11_prime_solo --variant pdl018d --run-name reference
--alpha-z 0.0036 --tile-cells 2048 --epochs 500 --patience 40 --seed 0`.

Pre-registered reads, same-section (its `metrics.json`): recon, NMI over
35 classes, probe ΔCE vs floor, mirror vs permuted, cycle_z vs the 50-PC
reference with cycle_w ≈ 0, KL_z / KL_w per dim. Pass = the ovarian/lung
allegiance shape. Then the **held-out section**:
`python -m discell.model.crossslide --dataset gse315411_pdltma06_11_prime_solo
--run reference --eval-dataset gse315411_pdltma06_10_prime_dual` (variant
`pdl018d` inherited) once the dual embeddings exist — recon over all 35
dual tiles and the same rows; pre-registered expectation: within the seed
envelope of the same-section numbers (recon ±0.06); a larger gap is the
section/chemistry generalisation cost, reported as such. A collapsed z
or an opened w channel triggers the ×2 / ×½ α_z bracket at 200 epochs;
nothing else is tuned.

**Results — `reference` on `pdl018d` solo, and held out on the dual
section (2026-09-15 16:24).** Fit: 54 train / 10 val tiles of 2048,
1 s/epoch, early stop at 94 (best 54), 11.7 min; cycle-score reliability
S 0.35 / G2M 0.57 (ovarian 0.22 / 0.52), cycling types by MKI67:
Proliferating, Erythroid, EC aerocyte capillary, Lymphatic EC.

| read | same section (solo, `metrics.json`) | held-out section (dual, `crossslide/…dual.json`) |
|---|---|---|
| recon, val tiles (nats/count) | −7.2201 best / −7.2272 final | −7.2354 |
| **recon, all 64 tiles** | — | **−7.2348** (gap 0.015 to the same-section best) |
| NMI (35 classes) | 0.625 / 0.620 | 0.596 |
| probe ΔCE (floor) | −0.003 (−0.0095) | +0.0015 (−0.0085) |
| mirror R² (permuted) | 0.027 (0.014) | 0.043 (0.014) |
| cycle_z pooled / cycle_w | **0.478 / 0.002** | **0.508 / 0.005** |
| 50-PC linear reference / ℓ-baseline | 0.497 / 0.029 | 0.473 / — |
| KL_w per dim | 0.0000–0.0006 | 0.0003–0.0094 |

Reading: every pre-registered read passes on the held-out section — the
reconstruction gap (0.015) is a quarter of the seed envelope (±0.06); z
carries the cycle (0.51, above the 50-PC linear reference 0.47) and w
does not (0.005); the probe sits at its floor and the mirror stays at
the ovarian level (0.043–0.046); NMI drops 0.02–0.03. Two things to note
rather than claim: on this core z does not exceed the linear reference
~2× as on ovarian (0.478 vs 0.497 same-section; 0.508 vs 0.473 held-out)
— a 279-transcript core with a curated 35-class vocabulary gives the
linear probe more to work with; and KL_w rises 5–15× on the other section
while staying ≤ 0.01/dim — the context prior `m_ψ(c,t)` is section-
specific to that small extent. Dual `pdl018d` graph after the loader's
rules: 207,899 of 209,862 edges kept, 38 isolated. Single seed; the
seed-triple and the §§3–4 battery on both sections are the next step.

### The w gauge: per-type offsets dominate ‖w‖ — weight-decay pair (motivation, 2026-09-15)

Motivation: the decoder is `a(z) + B·w` and z carries the type, so for any
per-type constant `μ_t` the model `(a(z) − B·μ_t, w + μ_t)` is identical:
`m_ψ` takes `t`, KL_w is against `m_ψ`, term (b) uses `m_ψ`. The optimiser
is plain Adam with no decay, so nothing pins that direction — where the
type offset lands is decided by dynamics, not by the objective. Measured
on `ablation_gat_type_only_s1` (scratch, `mu_w` from `collect_latents`,
connected cells, types ≥ 500):

| type | ‖mean_t w‖ (offset) | mean ‖w − mean_t w‖ (context-varying) | ‖B·mean_t w‖ |
|---|---|---|---|
| VEGFA⁺ Tumor | 7.39 | 0.77 | 28.4 |
| Smooth Muscle | 6.73 | 1.09 | 24.2 |
| Stromal Fibroblasts | 6.44 | 1.37 | 23.1 |
| Proliferative Tumor | 5.93 | 1.18 | 22.6 |
| Tumor Cells | 4.90 | 2.14 | 18.2 |
| T and NK | 2.61 | 3.78 | 9.7 |
| SOX2-OT⁺ Tumor | 0.71 | 4.75 | 2.6 |
| Macrophages | 0.55 | 5.08 | 1.8 |

For most types the offset is 4–10× the context-varying part; isolated
cells (no neighbours) still carry ‖w‖ = 4.8 (connected 5.6). Two
consequences already in the record: (i) the per-type ‖w‖ ranking in the
report is the offset, not context-dependence — Spearman between the
raw-‖w‖ ranking and the within-type-centred ranking is **−0.41** (the
actually context-modulated types are macrophages, SOX2-OT⁺ tumour, T/NK;
VEGFA⁺ tumour is among the least); (ii) `softmax(a(z))` as "the clean
profile" (doc-11 A2/A5) drops the type offset along with the niche
effect. Transport is unaffected (differences, both sides centred).

Experiment: a pair of fits differing only in Adam's **coupled** L2
(`--weight-decay`, new `TrainConfig` knob, default 0 = every pinned run).
Coupled rather than AdamW deliberately: in Adam's normalised update a flat
direction whose only gradient is λθ drifts toward zero at ~lr per step
whatever λ is, whereas decoupled decay at 1e-4 would move the gauge by
~0.15 % over the fit. λ = 1e-4, one value; seed 1 so the λ = 0 arm is
also a replicate of the pinned `ablation_gat_type_only_s1` under the
current code (int16 resident counts). Everything else as the reference:
`type_only`, d_w 6, d_z 20, α_z 0.007, α_w 0.1, α_a 0.3, κ 0.1, ω 1,
500 epochs / patience 40, `egomask_ego_v1`.

Runs: `gat_type_only_wd0_s1` (GPU 0), `gat_type_only_wd1e-4_s1` (GPU 1);
logs `logs/gat_type_only_wd{0,1e-4}_s1.log`.

Pre-registered reads (both arms, same scratch measurement):
1. ‖mean_t w‖ per type and ‖B·mean_t w‖ — wished for: offsets collapse
   toward the within-type spread under decay, unchanged in the λ = 0 arm.
2. Spearman(raw ‖w‖ ranking, centred ranking) — wished for: → ≈ +1 under
   decay (the ranking then means what the report says it means).
3. The battery guards must stay inside the seed envelope: val recon
   (±0.06 nats), NMI, cycle_z / cycle_w, KL_w per dim, probe ΔCE vs floor,
   mirror R². Recon worse than the envelope = the decay is too strong for
   the rest of the network, and the read is "gauge fixed at a cost", not
   a pass.
4. The λ = 0 arm vs `ablation_gat_type_only_s1`: same offsets and same
   guards = the int16 change is inert and the offsets are reproducible.
If the offsets do not move under decay, the honest reading is that the
gauge is set by initialisation/early dynamics faster than L2 drifts it;
the fix then stays at read time (centre w per type at a reference
context) rather than in training.

**Results — weight-decay pair (2026-09-16).** `wd0` early-stopped at
104 (best 64, 56 min); `wd1e-4` at 419 (best 379, 105 min). Both
`metrics.json`, plus the scratch offset read (`mu_w` via
`collect_latents`, connected cells, types ≥ 500):

| read | pinned `ablation_gat_type_only_s1` | `wd0_s1` | `wd1e-4_s1` |
|---|---|---|---|
| best recon (epoch) | −7.192 (64) | −7.189 (64) | −7.212 (379) |
| NMI best / final | 0.654 / 0.648 | 0.670 / 0.646 | 0.648 / 0.649 |
| cycle_z pooled (final) / cycle_w | 0.499 / 0.008 | 0.415 / 0.006 | 0.390 / 0.006 |
| probe ΔCE (floor) | 0.002 (−0.046) | 0.011 (−0.045) | −0.008 (−0.045) |
| mirror R² (permuted) | 0.046 (0.016) | 0.047 (0.016) | 0.045 (0.014) |
| KL_w per dim, max | 0.009 | 0.002 | 0.0006 |
| ‖global mean w‖ | 0.77 | 1.76 | **23.1** (dim 5 = −22.3) |
| ‖mean_t w‖: VEGFA⁺ / SM / Prolif / SOX2-OT⁺ / Mac | 7.4 / 6.7 / 5.9 / 0.7 / 0.6 | 8.0 / 6.5 / 6.8 / 1.2 / 1.4 | 23.9 / 24.3 / 24.3 / 22.4 / 23.4 |
| mean ‖w − mean_t w‖: same types | 0.8 / 1.1 / 1.2 / 4.8 / 5.1 | 0.7 / 0.8 / 0.8 / 4.0 / 4.5 | 2.4 / 3.7 / 2.8 / 8.3 / 7.6 |
| ‖B·mean_t w‖ range | 1.8–28 | 3.4–29 | **81–92** |
| Spearman(raw ‖w‖ rank, centred rank) | −0.41 | −0.57 | 0.40 (all raw norms ≈ 24; not meaningful) |
| isolated cells ‖w‖ (connected) | 4.8 (5.6) | 4.5 (5.4) | 23.9 (24.6) |
| Σ‖params‖² / ‖a(z) output bias‖ | 34.0k / 6.2 | 33.8k / 6.3 | **1.6k / 1.1** |
| B column norms | 2.5 3.6 1.4 1.8 2.6 2.1 | 2.3 3.7 1.5 1.9 1.3 2.4 | 1.6 2.3 1.8 1.7 0.8 **3.7** |

Reads against the pre-registration:
1. **Offsets did not collapse — they grew ~4× and became global.** Every
   type now sits at ‖mean_t w‖ ≈ 22–25, almost all of it one shared
   vector on dim 5 (−22.3 for every cell, isolated ones included); the
   per-type component is small relative to it. Mechanism, confirmed from
   the parameters: L2 decays *parameters* (B, the decoder's per-gene
   output bias) but not the *latent* w, which is a network output. A
   fixed gene-bias vector `b` costs ‖b‖² sitting in `a(z)`'s output bias
   and ‖b/s‖² sitting in `B_5·w_5` with `w_5 = s` — so decay pushes the
   global gene bias out of `a(z)` (bias norm 6.3 → 1.1) into a small B
   column times a huge constant w (both gauges — offset and scale — move
   *toward* w). Coupled Adam-L2 does exactly the wrong thing here.
2. The Spearman moved to +0.40 only because every raw norm is ≈ 24; the
   ranking is noise around one global offset. Not a pass.
3. Guards: inside the envelope (recon −0.02 vs pinned; NMI, probe at
   floor, mirror, KL_w all in range). cycle_z 0.39 vs 0.42 (`wd0`) vs
   0.50 (pinned) — the `wd0`↔pinned gap (same config, same seed,
   non-deterministic scatter kernels) already spans 0.08, so the decay
   arm's cycle_z is not distinguishable from run-to-run spread.
4. **`wd0` reproduces the pinned run**: same best epoch, recon within
   0.003, same offset pattern per type (VEGFA⁺ 8.0 vs 7.4, SM 6.5 vs 6.7,
   Mac 1.4 vs 0.6), same inverted ranking (−0.57 vs −0.41). The int16
   change is inert; the type offsets are a reproducible product of the
   dynamics, not a one-seed accident.

Conclusion: parameter weight decay cannot fix a latent's gauge, and makes
it worse. The offsets are real, reproducible, and meaningless (§7.12's
rescaling argument extended to translation). Standing decisions: (a) the
per-type ‖w‖ ranking in the report and handover is withdrawn as a
context-dependence read — the within-type-centred spread is the read
(macrophages, SOX2-OT⁺ tumour, T/NK most context-modulated; VEGFA⁺
tumour least); (b) doc-11 A2/A5's `softmax(a(z))` must decode at a
reference context, `softmax(a(z) + B·m_ψ(c̄_t, t))`, never at w = 0;
(c) if the gauge is to be fixed in training it needs a penalty on the
latent's mean itself (e.g. ‖E_batch[m_ψ(c,t)]‖² per type), not on
parameters — not attempted; read-time centring is sufficient for every
current use. `--weight-decay` stays in `TrainConfig` at default 0.
Issue V12.

### Neighbour dose: what `c` (and `B·m_ψ`) can and cannot see (motivation, 2026-09-16)

Motivation: under `type_only` the GAT part of `c_i` is `Σ_j α_ij W·onehot(t_j)`
with `Σ α = 1` — a type-pair-reweighted *composition*. One tumour neighbour
and nothing else gives the same `c` as eight; the only count information is
the isolated flag. On this graph the pruned Delaunay degree is tight (median
6, 96 % in 4–8), so count ≈ 6 × fraction almost everywhere, and Φ does not
recover count (ridge Φ → log degree R² 0.23; Φ → tumour count | fraction
R² 0.02). The user's question: how much does `c` actually move when the
neighbourhood is manipulated, and through which channel?

Instrument (`discell/experiments/neighbour_dose.py`, no retraining): the
pinned `ablation_gat_type_only_s1`; a one-cell tile — receiver of type `t`,
`n` synthetic neighbours of chosen types, Φ fixed at the receiver type's
mean — through `model.context` → `model.prior_w`; the read is the realised
shift `B·m_ψ` in gene space, always as a difference (the w gauge).
Receivers: Macrophages, T and NK, Tumor Associated Fibroblasts, Tumor Cells.

Setups and pre-registered expectations:
- A. **count at fixed composition** — 1…8 tumour neighbours only, and
  k tumour + k own (50/50): `‖B·(m_ψ(n) − m_ψ(1))‖` = 0 exactly (the
  invariance, shown rather than asserted); n = 0 (isolated) differs.
- B. **fraction at degree 6** — k tumour + (6−k) own, k = 0…6:
  `‖B·(m_ψ(k) − m_ψ(0))‖`; the curve the model can express. Expected
  monotone; its size vs the receiver's measured within-type realised shift
  (mean `‖B·(w − mean_t w)‖`: Mac 18.5, T/NK 13.5, TAF 9.2, Tumor 8.0)
  says how much of w's context-dependence one composition axis explains.
- C. **composition swap at degree 6** — all-of-one source type vs all own,
  per source: the effect-size scale of the composition channel.
- D. **image channel** — composition fixed at 6 own, Φ from 500 real cells
  of the receiver type: mean `‖B·(m_ψ(Φ_i) − mean)‖`. Read: if D ≳ B/C,
  w's context-dependence is mostly image-driven; that bears on the
  interventionability claim, because Φ is not an exposure `do(c′)` can set.

Outputs: `experiments/neighbour_dose_<run>.{json,png}`. Characterisation, not a
pass/fail; the one pre-registered comparison is D vs B/C.

**Results — neighbour dose (2026-09-16, `experiments/neighbour_dose_ablation_gat_type_only_s1.{json,png}`,
pinned `ablation_gat_type_only_s1`).** Gene-space norms of `B·Δm_ψ`.

- **A. Count invariance is exact**: 1…8 tumour neighbours (and 1…4 pairs of
  tumour+own) change the shift by ≤ 1.3e-6. The isolated state (n = 0:
  GAT zeroed + flag) is a different context altogether — 10–17 from the
  one-neighbour state, as large as the largest composition swap. With 899
  isolated cells in training, their `m_ψ` is an extrapolation, not a fit.
- **B. The fraction response is receiver- and source-specific and
  non-linear.** Macrophages ← tumour saturate: one tumour neighbour in six
  gives 12 of the 17.5 at full replacement (concave — "any tumour
  contact"). T/NK ← tumour is convex: < 5 until fraction 0.5, 10 at 0.83,
  24 at 1.0 — close to a threshold at "fully surrounded" (intratumoural vs
  marginal). TAF ← tumour is small and near-linear (4.1). Macrophages ← TAF
  also saturating (10.7 → 16.5); T/NK ← TAF near-linear to 22. Full
  replacement reaches the measured within-type shift for immune receivers
  (Mac 17.5 vs 18.5; T/NK 24 vs 13.5) but not for TAF (4 vs 9.2) or tumour
  (0/4 vs 8.0) — their context-dependence is not about tumour/TAF fraction.
- **C. Composition-swap scale**: 4–25. Largest: smooth muscle for TAF (25)
  and tumour cells (24.5); tumour for T/NK (24); tumour/TAF for macrophages
  (17.5/16.5).
- **D/E. The image channel.** Against the *maximal* swap, Φ's natural
  spread is 24–62 % (D / max C). But like-for-like on the same 500 real
  cells (E: real neighbours with mean Φ vs mean composition with real Φ):

  | receiver | composition only | Φ only | both | measured within-type shift |
  |---|---|---|---|---|
  | Macrophages | 10.8 | 10.9 | 18.0 | 18.5 |
  | T and NK | 7.0 | 8.0 | 13.5 | 13.5 |
  | TAF | 2.9 | **7.2** | 8.9 | 9.2 |
  | Tumor Cells | 4.1 | **6.0** | 7.8 | 8.0 |

  "both" reproduces the measured spread within 3 % — the synthetic-tile
  instrument is faithful and `w ≈ m_ψ` end to end. Under natural
  variation the image channel carries **as much as composition for immune
  receivers and 1.5–2.5× more for TAF and tumour cells**. The
  pre-registered read therefore lands on the uncomfortable side: a large
  part of w's context-dependence is Φ-driven, and Φ is not an exposure
  `do(c′)` can set. Composition and Φ are also not additive (Mac 10.8 +
  10.9 vs 18.0): `m_ψ` mixes them.

Reading for the design: (i) the softmax makes `c` blind to dose at one hop,
exactly, and on this graph that costs little (degree ≈ 6); the real range
limitation is multi-hop (paracrine) count, untested here; (ii) the isolated
flag is an unfitted regime; (iii) the interventionable share of w's
context-dependence is roughly half for immune receivers and a third for
stromal/tumour receivers — a transport counterfactual that changes
composition but carries each niche's real Φ is mixing an intervention
with a description. Standing decision: none yet; candidates are a
Φ-partialled transport read (hold Φ at the receiver type's mean in both
niches) and a count-within-radius test before any architecture change.

### Attention sink as an option: `gat_sink` (design note, 2026-09-16)

Following the neighbour-dose result (count invariance is exact under the
softmax), the user asked for dose to have a meaning inside the GNN. Of the
candidates — sum/ReLU aggregation (unbounded; conflates sparse regions with
few sources), precomputed multi-radius counts appended to `c` (paracrine
range, but density back in the estimand), and an attention sink — the sink
is the minimal change: a fixed null logit 0 joins every destination's
softmax, `α_ij = e^{e_ij} / (1 + Σ_j' e^{e_ij'})`, so for n identical
neighbours the aggregate is `n e^e / (1 + n e^e) · W h` — Michaelis–Menten
in n with the half-point `e^{−e}` learned per (t_i, t_j). Bounded, receptor-
saturation-shaped, and the isolated contract holds with no flag needed (all
mass on the sink, zero out; the flag stays for interface stability).

Built as `GATv2(sink=…)` → `DisCell(gat_sink=…)` → `TrainConfig.gat_sink`
(default **False**, every pinned run unchanged) → `--gat-sink`; the three
checkpoint loaders default old payloads to False. Test:
`test_attention_sink_makes_dose_count_and_keeps_isolated_zero` (4 identical
neighbours ≠ 1 with the sink, = without; empty destinations zero in both).
Registered in `spec_deviations.md` (§4.1). Not yet trained: a first fit
would be a seed-1 `type_only` run with `--gat-sink`, read first through
`experiments/neighbour_dose.py` (panel A should stop being flat) and then
the battery + transport, as for any change to `c`.

### Nuclear vs extranuclear counts on every dataset (motivation, 2026-09-16)

User's request: the nuclear / extranuclear split of each cell's counts
on all five datasets, as a shared check. The instrument exists —
`discell.applications.shared --stage nuclear-counts` (doc-11's
segmentation-perturbation dependency) rebuilds a cells × genes matrix
from the transcripts flagged `overlaps_nucleus` at q ≥ 20, aligned to the
bundle; extranuclear = bundle counts − nuclear, per cell and gene. What
it is for here: the extranuclear share is the part of a cell's profile
that lives in the segmentation's cytoplasmic expansion — the transcripts
the leak channel is about — so the per-dataset, per-segmentation-method
and per-type nuclear fraction is the first-order read of how much
spill-over each slide can carry, and the base for any per-cell
decontamination test (A4's nuclear arm) beyond the ovarian slide.

State: ovarian FFPE has it (`qc/nuclear_counts.npz`, 55.8 M nuclear q20
transcripts = 37.8 %). GSE315411 solo / dual carry `transcripts.parquet`
on disk (3.6 / 3.2 GB). Lung (2.3 GB) and FF ovary (33.7 GB) keep it in
their download zips — extracted into the existing `extracted/` dirs
(other members untouched), then the stage runs on the four datasets in
sequence on CPU (`logs/nuclear_counts_all.sh`), full bundles (the
`pdl018d` variants are row subsets of the same cell ids). Pre-registered
reads, descriptive: per dataset the overall nuclear fraction and the
distribution over cells; by segmentation method (boundary stain / interior
18S / nucleus expansion — the last has by construction a 5 µm cytoplasm
ring, so its fraction is the reference for "expansion-only" cells); by
type from the bundle's default label; and the sanity check nuclear ≤
bundle counts for every cell (both are q20-assigned). No claim attached
to a number before it exists.

**Results (2026-09-16 12:23).** Lung and FF `transcripts.parquet`
extracted (2.3 GB / 33.7 GB, 8 s / 2 min); the stage took 27 s (lung),
48 / 45 s (GSE solo / dual), 12 min (FF, 2.70 G transcripts read).
Summary stage added to the same module (`--stage nuclear-summary` →
`qc/nuclear_summary.json`). Nuclear share of **cell-assigned q20
counts** (the earlier "37.8 %" for ovarian was over all transcripts,
unassigned included; the sanity check nuclear ≤ total holds for every
cell on every slide):

| dataset | cells | tx/cell | nuclear, pooled | per-cell p25 / p50 / p75 | boundary stain | interior 18S | nucleus expansion |
|---|---|---|---|---|---|---|---|
| ovarian FFPE | 407,120 | 262 | **0.524** | 0.43 / 0.56 / 0.67 | 0.561 | 0.489 | 0.620 |
| lung FFPE | 278,324 | 440 | **0.446** | 0.34 / 0.48 / 0.62 | 0.537 | 0.424 | 0.484 |
| ovary FF | 1,157,637 | 1,668 | **0.580** | 0.46 / 0.60 / 0.70 | 0.608 | 0.564 | 0.668 |
| GSE315411 solo (11) | 1,100,349 | 161 | **0.496** | 0.36 / 0.52 / 0.66 | 0.498 | 0.495 | 0.477 |
| GSE315411 dual (10) | 1,115,054 | 142 | **0.503** | 0.37 / 0.53 / 0.67 | 0.501 | 0.503 | 0.468 |

Reading, descriptive: roughly half of every cell's assigned counts lie
outside its nucleus on every slide, with a wide per-cell spread (IQR
≈ 0.3) — the extranuclear half is the material the leak channel and
any decontamination claim act on. Slides differ modestly (lung lowest at
0.45, FF highest at 0.58); the two GSE sections agree to 0.007 pooled and
to the same per-type ordering — B cells / inflammatory fibroblasts /
distal epithelium ≈ 0.57, goblet 0.40–0.43, secretory 0.38, alveolar
macrophages 0.33–0.34 — cells with large cytoplasm and cytoplasm-resident
mRNAs (mucins, macrophage programmes) sit low, small lymphocytes high, as
expected. On ovarian the extremes are stromal fibroblasts / smooth muscle
0.59 vs tumour-associated fibroblasts **0.415** and cyst-lining malignant
cells 0.43 — a candidate leak signature (TAFs sit against tumour) worth a
targeted look, not a claim. Segmentation method is not a fixed offset:
nucleus-expansion cells are the most nuclear on the 10x slides (0.62 /
0.67; lung 0.48) but the *least* on GSE315411 (0.47), where the
expansion cells are the 1 % without a usable membrane stain. Outputs:
`qc/nuclear_counts.npz` and `qc/nuclear_summary.json` per dataset;
per-type tables in the JSON.

### First sink fit: `gat_sink_s1` on the ovarian slide (motivation, 2026-09-16)

Motivation: the sink makes `c` count neighbours (saturating, per type pair)
instead of seeing fractions only. One fit, the reference protocol so the
comparison is against the pinned `ablation_gat_type_only_s1` and nothing
else moves: `--gat-sink --seed 1 --epochs 500 --patience 40`, all other
`TrainConfig` defaults (`type_only`, d_w 6, d_z 20, α_z 0.007, α_w 0.1,
α_a 0.3, κ 0.1, ω 1, `egomask_ego_v1`, no weight decay). Then the full
artefact chain the pinned run carries, in order, in one detached script
(`logs/gat_sink_s1_pipeline.log`): `validate` → `transport` →
`communication` → `lr_map` → `atlas` → `applications.a4_cycle` → `report` →
`experiments.neighbour_dose`.

Pre-registered reads, `gat_sink_s1` vs `ablation_gat_type_only_s1` (and its
`wd0` replicate for the run-to-run band):
1. **Guards inside the envelope**: val recon (±0.06), NMI, probe ΔCE at
   floor, mirror, KL_w, cycle_z / cycle_w. Recon outside the band = the
   sink costs fit; cycle_z below the band = it leaks into z.
2. **Neighbour dose, panel A no longer flat**: `‖B·(m_ψ(n) − m_ψ(1))‖ > 0`
   for 1…8 tumour neighbours, saturating; the isolated point should now
   sit at the n → 0 end of that curve rather than off it. Panel B/E as a
   before/after of the composition-vs-Φ split — wished for: composition's
   share up, since dose is composition information the softmax discarded.
3. **w-side battery** (niche AUC, Moran, pseudotime niche R²) and
   **transport** (counterfactual R², beats-both, slope): ≥ the pinned run.
   Transport is the one held-out test of the response channel; a sink that
   raises it is evidence dose matters, one that lowers it is evidence the
   softmax's fraction view was the better inductive bias on this graph.
4. z-side reads unchanged (cycle_z, probe) — the sink touches `c` only.
No pass/fail on 2 beyond "not flat"; 1 and 3 decide whether the option is
worth a seed pair.

**Addendum (2026-09-16, while `gat_sink_s1` trains) — the invariance target
under the sink.** Every invariance instrument targets fractions and image:
the adversary's y-head sees `y_i = counts/degree` (soft CE vs `ȳ(t)`), its
Φ-head the soft k-means membership of Φ's PCs, the held-out probe and the
closed-form penalty the block `[y[:, :−1], 12 PCs(Φ)]`. Under the plain
softmax that is exactly what `c` contains. With the sink `c` also carries
count, and none of the three can see z predicting count-beyond-fraction —
a hole the user pointed at. Size on the softmax runs (within-type ridge
from z, held-out folds, permuted floor ≈ 0): log degree R² 0.005 / 0.003,
tumour count | fraction 0.001 / 0.001, tumour fraction 0.023 / 0.048
(pinned / wd0) — z carries no count today, and the known niche-z residual
lives in the fraction. **Pre-registered read 5 for `gat_sink_s1`**: the
same probe; if z → log degree or count | fraction rises above the floor,
the target must grow with `c` — the consistent minimal fix is a third
adversary head on a degree bin (1–2, 3, …, 8, ≥ 9; CE against the type's
bin distribution) and `log(1+degree)` appended to `v` for the probe and the
closed form. Also noted: isolated cells have a zero `y` row, so their soft
CE is 0 — silently absent from the y-head; and the hole grows on any graph
with variable degree (radius graphs), so the sink and the target should be
changed together if the option is adopted.

**Results — `gat_sink_s1` (2026-09-16).** Train 17.7 min, early stop 99
(best 59); chain clean, every reference artefact present. Band = the
pinned `ablation_gat_type_only_s1` and its `wd0` replicate (battery and
transport run on `wd0` today for this purpose).

| read | pinned s1 | wd0 (replicate) | **gat_sink_s1** |
|---|---|---|---|
| best recon (epoch) | −7.192 (64) | −7.189 (64) | −7.194 (59) |
| NMI best | 0.654 | 0.670 | 0.635 |
| probe ΔCE at best epoch (history) / final | 0.013 / 0.002 | 0.031 / 0.011 | 0.010 / 0.034 |
| independent z-probe, held-out: composition / Φ block R² | 0.017 / 0.014 | — | 0.017 / 0.008 |
| **read 5**: z → log degree / tumour count │ fraction | 0.005 / 0.001 | 0.003 / 0.001 | 0.003 / 0.002 (floor ≈ 0) |
| mirror R² / KL_w max | 0.046 / 0.009 | 0.047 / 0.002 | 0.048 / 0.027 |
| cycle_z pooled (metrics / battery) | 0.499 / 0.440 | 0.415 / 0.388 | 0.435 / 0.386 |
| cycle_w | 0.008 / 0.006 | 0.006 / 0.002 | 0.009 / 0.005 |
| niche AUC z / w | 0.654 / 0.768 | 0.655 / 0.750 | 0.646 / 0.768 |
| pseudotime niche R² z / w | 0.005 / 0.565 | 0.025 / 0.648 | 0.031 / 0.545 |
| Moran mean│I│ z / w | 0.061 / 0.628 | — | 0.079 / 0.614 |
| transport counterfactual / full / program / leak | 0.099 / 0.146 / 0.068 / 0.063 | 0.092 / 0.143 / 0.056 / 0.063 | 0.103 / 0.149 / 0.068 / 0.062 |
| transport beats-both / slope | 100/155 / 0.92 | 99/155 / 0.92 | **104/155 / 0.97** |

Neighbour dose (`experiments/neighbour_dose_gat_sink_s1.png`):

| receiver | A: max over n = 1…8 tumour (pinned → sink) | isolated | E comp / Φ / both (sink) | ref |
|---|---|---|---|---|
| Macrophages | 0.00 → **0.17** | 15.1 → 15.3 | 10.8 / 9.7 / 17.3 | 16.4 |
| T and NK | 0.00 → **0.63** | 16.4 → 7.7 | 5.5 / 7.3 / 12.1 | 11.4 |
| TAF | 0.00 → **0.38** | 12.2 → 12.3 | 2.9 / 6.9 / 8.5 | 8.4 |
| Tumor Cells | 0.00 → **0.12** | 10.2 → 9.4 | 4.2 / 5.4 / 7.0 | 6.7 |

Reads against the pre-registration:
1. **Guards inside the envelope** — recon, mirror, cycle_z/w, niche AUC all
   in band; NMI 0.635 is 0.02 below the two-run band; KL_w max 0.027 is
   above (still ≈ 0). The final-epoch probe ΔCE +0.034 looked like a leak
   but is epoch-to-epoch noise of that probe (±0.03 in every run's
   history; 0.010 at the best epoch), and the independent held-out probe
   on the best checkpoint shows z **no less invariant** than the pinned
   run (composition 0.017 = 0.017, Φ 0.008 < 0.014).
2. **Panel A is no longer flat — but nearly**: the dose response over
   1…8 tumour neighbours is 0.1–0.6 against composition effects of 5–17,
   i.e. the learned half-point is below one neighbour: given the freedom
   to count, the model saturates at n = 1 and reproduces the softmax's
   fraction view. On a graph with degree ≈ 6 everywhere there is no
   gradient signal from which to learn a dose curve. The isolated state
   stays a separate regime (7.7–15.3 away), except T/NK where it halved.
   The composition-vs-Φ split (E) did not move: Φ still ≥ composition for
   TAF and tumour cells; w's total spread is 5–15 % smaller than pinned.
3. **w-side / transport**: niche AUC w equal, pseudotime R² w and Moran w
   slightly lower (in or at the band); transport counterfactual R² 0.103
   (band 0.092–0.099), beats-both 104 (band 99–100), slope 0.97 (band
   0.92). Marginally the best of the three on every transport read, by
   amounts comparable to the replicate spread. One seed: suggestive, not
   evidence.
4. z-side unchanged; **read 5 at the floor** — z picked up no degree or
   count, so the invariance-target mismatch is empty on this graph as it
   was for the softmax.

Standing decision: `gat_sink` stays an **option, off by default**. On the
Voronoi graph the sink is a no-op in substance (dose saturates below one
neighbour), which is itself the finding: the count blindness costs
nothing here because the data carry no count variation to learn from.
The option becomes meaningful only with a graph where degree varies
(radius / contact graphs, or multi-hop counts), and there the invariance
target must grow with it (addendum above). Transport +0.004 / +4 panels /
+0.05 slope is the one lead; a seed pair would settle whether it is real.

### Sink seed pair: `gat_sink_s0`, `gat_sink_s2` (motivation, 2026-09-16)

Motivation: the user asked what stands between `gat_sink` and default.
The one empirical lead is transport (+0.004 counterfactual R², +4
beats-both panels, +0.05 slope on s1), inside the replicate spread. Seeds
0 and 2 against the existing references `ablation_gat_type_only` (s0) and
`ablation_gat_type_only_s2`, same protocol (`--gat-sink --epochs 500
--patience 40`, defaults otherwise), decision chain only: `validate` →
`transport` → `experiments.neighbour_dose`. Logs
`logs/gat_sink_s{0,2}_pipeline.log`, GPU 0 / GPU 1 in parallel.

Pre-registered decision rule (three paired seeds, sink vs reference):
- **Adopt as default** only if transport counterfactual R² and beats-both
  are ≥ the reference in 3/3 seeds *and* the guards (recon ±0.06, NMI
  within 0.02, cycle_z inside the seed band, probe at floor) hold in 3/3.
- **Keep as option** otherwise — including "wins 2/3": on this graph the
  sink is a near no-op (half-point < 1 neighbour), so a 2/3 result is
  seed noise, not a mechanism.
Independent of the numbers, three structural conditions for a default
change are recorded in the results entry below.

**Addendum — KL_w under the sink is a worse prior, not a richer w
(2026-09-16).** The user flagged KL_w max/dim 0.027 vs 0.009 / 0.002.
Decomposed on validation tiles (scratch `klw_anatomy`): the sink's KL_w
(0.025/dim total) is almost entirely the mean-term (0.106 on one dim);
posterior sd unchanged (≈ 0.99); ‖μ_w − m_ψ‖ in gene space 1.77 vs
0.26 / 0.25; prior-R² per dim 0.74–0.996 vs ≥ 0.998; the posterior's
reconstruction gain over the prior draw, (a) − (b), is 0.00046 vs
0.00051 / 0.00037 — **no gain**; and the displacement is **64 %
predictable from `[y, PCs(Φ)]`** on held-out folds vs 8 % on both softmax
runs. Signature of `m_ψ` under-reading context, with `enc_w` (which also
sees `x`) recovering it per cell. Mechanism (plausible, untested): under
the sink the GAT output's magnitude varies with the learned type-pair
logits (`Σα < 1`), so composition reaches `m_ψ` entangled with scale and
the MLP decodes it less faithfully than from the softmax's convex
combination. Remedy if the option is pursued: give `m_ψ` both the softmax
composition and the sink mass, not one in place of the other. Also
reframes the s1 transport uptick: transport runs on `m_ψ`, and a prior
that reads context worse should not forecast held-out niche shifts better
— noise is the likelier reading, pending the seed pair.

**Results — sink seed pair, decision (2026-09-16).** `gat_sink_s0` (best
epoch 74, 65 min — slow: two runs in parallel oversubscribe the CPU in the
figure step, noted in the background-jobs memory) and `gat_sink_s2` (best
69, 68 min); chains clean. Paired against `ablation_gat_type_only{,_s1,_s2}`:

| seed | recon ref → sink | NMI | cycle_z (metrics) | probe / KL_w max | niche AUC w | Moran w | transport cf R² | beats-both | slope |
|---|---|---|---|---|---|---|---|---|---|
| s0 | −7.253 → −7.255 | 0.667 → 0.652 | 0.440 → 0.440 | 0.017 / 0.002 | 0.743 → 0.740 | 0.531 → 0.678 | 0.085 → **0.081** | 103 → **95** | 0.97 → 0.95 |
| s1 | −7.192 → −7.194 | 0.654 → 0.635 | 0.499 → 0.435 | 0.034* / 0.027 | 0.768 → 0.768 | 0.628 → 0.614 | 0.099 → 0.103 | 100 → 104 | 0.92 → 0.97 |
| s2 | −7.231 → −7.233 | 0.666 → 0.663 | 0.465 → 0.472 | −0.012 / 0.059 | 0.754 → 0.753 | 0.575 → 0.400 | 0.092 → **0.090** | 92 → 95 | 0.91 → 0.92 |

(* final-epoch value; 0.010 at the best epoch, independent probe at
floor.) Neighbour-dose panel A on every seed: max over 1…8 tumour
neighbours 0.0–1.2 against composition effects of 5–17 — the sink never
learns a dose response on this graph; isolated-state distance varies by
seed (1.9–15). E-split unchanged (Φ ≥ composition for TAF and tumour).

Against the pre-registered rule — transport counterfactual R² and
beats-both ≥ reference in 3/3, guards in 3/3: **1/3** on counterfactual
R² (s1 only), 2/3 on beats-both, recon identical in all three (±0.002),
NMI −0.003 to −0.019, cycle_z inside the seed band. The s1 transport
uptick was seed noise, as the KL_w decomposition predicted. The
worse-prior signature (KL_w 0.027 / 0.059) appears in two of three seeds
(s0 stays at 0.002); Moran w moves both ways (+0.15, −0.01, −0.18) — a
large seed spread with no direction.

**Decision: `gat_sink` stays an option, off by default.** Recorded
reasons, in order of weight: (1) on a Voronoi graph the mechanism cannot
be exercised — degree ≈ 6 gives no dose variation to learn from, and the
fitted half-point sits below one neighbour; (2) in 2/3 seeds the prior
`m_ψ` reads context worse (posterior displacement 6× larger, 64 %
context-predictable, zero reconstruction gain), and `m_ψ` is the object
every counterfactual and transport read uses; (3) the invariance target
covers fractions only — empty here, not on a variable-degree graph, so
adopting the sink means adopting a degree-aware target with it; (4) no
transport benefit in 3 seeds. The option earns a second look only paired
with a graph whose degree varies (radius / contact) plus the degree head,
and with `m_ψ` fed both the softmax composition and the sink mass.
`spec_deviations.md` §4.1 row unchanged (option only).

*Session note (2026-09-17).* The entries below were produced by delegated agents (Opus), each checked by an independent reviewer or refuter (Sonnet) before integration; the refuter's two count corrections on the leak-meter entry are applied. Five further packages (transport, atlas, x̃ gate, baselines, FF battery) were paused to save tokens; their partial edits are held as patches outside the tree.

### FF slide, first fit: results (`reference_graphclust`, written up 2026-09-17)

**Fit** (`data/datasets/xenium_prime_human_ovary_ff/runs/reference_graphclust/`,
`logs/reference_graphclust.log`; trained 2026-09-15 08:37–11:40 from git
`88b1c06+dirty`): 1,157,637 cells, 39 types (38 graphclust + Unassigned
for 76 zero-count cells), voronoi graph 3,442,013 edges after the 40 µm
prune (368 isolated, 606 single-neighbour); 512 tiles by bisection to
≤ 4096 (~2,260 cells each), **435 train / 77 val** (~174k val cells);
α_z 0.0007, α_w 0.1, α_a 0.3, κ 0.1, ω 1, `type_only`, adversary, seed 0,
500 / patience 40, `egomask_ego_v1`. Φ *was* attached: the loader's
`no image embeddings given` warning precedes `assemble`'s own load (the
lung log carries the identical line) and the `cells lack an image
embedding` warning never fired. Counts resident as int16 (the < 32768
assert held); the predicted 18–20 GB footprint is **not logged —
unverified**. Early stop at epoch **409, best 369**, 82 evaluations,
**182.7 min**. Cycle instruments: 18 S + 34 G2M markers, split-half
reliability **S 0.508 / G2M 0.696** (ovarian FFPE 0.217 / 0.524, lung
0.213 / 0.279); phases G1 61.4 / S 22.7 / G2M 15.9 % (FFPE slides
~48 / 32 / 20 — noisier scores call more cells cycling); cycling types by
MKI67 fraction: Cluster-36 (2,073 cells), -34 (4,661), -21 (18,850), -4
(70,602).

**Pre-registered reads** (`metrics.json`; "best" = epoch-369 row of
`history.jsonl`, "final" = epoch 409; comparison runs quoted from their
`metrics.json` final blocks unless marked ⁺ = their best-epoch history row):

| read (rule) | FF `reference_graphclust` | ovarian `ablation_gat_type_only_s1` / lung `reference_graphclust` | verdict |
|---|---|---|---|
| held-out recon (early-stop signal; per-count nats are **not comparable across slides**) | **−7.3138** best / −7.3144 final | −7.1924 / −7.2577 | — |
| NMI, k-means-39(z) vs 39 classes (collapse = NMI ≪ 0.6 *with* KL_z → 0) | **0.595** at best (0.643 at epoch 4 → 0.58–0.60 from epoch 110; min 0.577) | 0.654 (18 classes) / 0.653 (33) | not collapsed: at the 0.6 line, and KL_z is the highest of any slide |
| KL_z, TB `train/kl_z` ±2 epochs of best (training batches, sampled) | **24.5 nats/cell = 1.22/dim** (63.6 in epoch 0, 30.9 at epochs 5–9, 24.4 at 45–49, 27.0 last 5) | 10.3 (0.52/dim)⁺ / 16.7 (0.83/dim)⁺ | open, as α_z 10× smaller predicts; z is the opposite of collapsed |
| probe ΔCE vs floor (baseline CE) | **0.0054** best / 0.0025 final; floor −0.0133; baseline 1.394 → excess 0.019 = 1.3 % of baseline | 0.0017 (floor −0.046; 2.29 → 2.1 %) / −0.010 (−0.021; 1.31 → 1.4 %) | at the lung level; passes |
| mirror R² vs within-type permuted | **0.055 / 0.016** (3.3×) | 0.046 / 0.016 (2.9×) / 0.036 / 0.015 (2.4×) | ovarian level; passes |
| cycle_z pooled / cycle_w / permuted / ℓ-baseline | **0.766 / 0.0027** best, 0.767 / 0.0027 final / −0.0004 / 0.0006; by cluster z 0.66 (-36), 0.73 (-34), 0.78 (-21), 0.77 (-4), w ≤ 0.016 | 0.499 / 0.008 / −0.001 / 0.001 ; lung 0.347 / −0.000 / — / 0.000 | z carries the cycle, w does not; passes |
| 50-PC linear expression reference | **0.852** (0.83–0.86 per cluster) | 0.235 / 0.166 | z / reference = **0.90** vs 2.1 / 2.1 — see below |
| KL_w per dim (opened = ≫ 0.002/dim sustained) | at best **sum 0.00016, max dim 0.00006**; final sum 0.00034 (dims 2–4 at 10⁻⁸–10⁻⁹); run median 0.0004/dim; single-evaluation spikes to 0.032/dim (epochs 19, 29, 69, 114, 124, 139, 144, 164–174, 189, 214, 224, 339, 374); 32 of 82 evaluations have one dim above 0.002 | 0.0047⁺ / 0.0020⁺ (sums) | pinned, ~30× harder than ovarian at the checkpoint; the spikes are the chase already registered (issues, 2026-09-14) |

**Verdict: pass; the α_z bracket does not fire.** Neither trigger is
met — z is open (KL_z 1.2/dim, cycle 0.77, NMI 0.595) and w is closed
(KL_w ≤ 0.0001/dim at the checkpoint). The motivation's prediction that
α_w at 140× 1/ℓ̄ pushes w further into the prior-pinned regime is what the
numbers show: KL_w at best is 12–30× below the two FFPE slides. Nothing is
tuned.

**Why cycle_z is 0.77 here and what it does not mean.** The 50-PC linear
reference is 0.852 on this slide against 0.235 (ovarian) and 0.166
(lung). The Tirosh score is a function of log-normalised counts over 52
marker genes; at 178–242 transcripts/cell those genes are mostly zeros,
the score is noise-dominated (split-half G2M 0.28–0.52) and 50 PCs of the
sparse matrix — dominated by type and depth — recover a quarter of its
within-type variance, while the encoder (nonlinear, all 5,001 genes,
trained on the likelihood) recovers half: the "2×". At 1,401
transcripts/cell the score is reliable (0.51 / 0.70), the PC frame
carries it linearly (0.85), and z — 20 dims that must also carry every
other intrinsic axis at 1.2 nats/dim — retains 0.77, i.e. **0.90× the
frame, uniformly across the four clusters** (0.66–0.78; the noisy
per-type spread of issues M6 is absent). Four slides now: z/reference
2.1 (ovarian, 178/cell), 2.1 (lung, 242), 0.96 (pdl018d, 279, curated
35 classes), 0.90 (FF, 1,401). The ratio is a property of the target's
reliability and the frame's strength, not of the model; the cross-slide
claim is the absolute one — z far above the permuted control and the
ℓ-baseline, w at zero — and M6's "z beats the 50-PC ridge ~2×" must carry
the depth qualifier. The transfer read ("scores more reliable at 8×
depth") holds for the *target*: the permuted control and ℓ-baseline sit
at 0 within ±0.001 on every slide, so the 0.77 is not inflated by depth,
it is measured against a better ruler.

**Trajectory** (`history.jsonl`): recon −7.3567 (epoch 4) → −7.3308 (19)
→ −7.3202 (59) → −7.3165 (104) → −7.3158 (199) → −7.3146 (299) → −7.3138
(369) → −7.3144 (409): 0.043 gained in total, 0.0027 of it after epoch
104 — a long, slow tail unlike the FFPE slides (best 59–64 of ~100).
cycle_z 0.39 (4) → 0.71 (19) → 0.75–0.77 from epoch 24 on; cycle_w 0.045
(4) → 0.007 (9) → 0.001–0.003; mirror 0.091 (4) → 0.05–0.06 from epoch
14; probe 0.053 (4) → 0.024 (9) → 0.002 (19) → 0.004–0.015 thereafter
(floor −0.013 throughout). **NMI drifts down** 0.643 (4) → 0.61–0.64
(to 104) → 0.58–0.61 (after), min 0.577 at 374, while recon improves:
the tension spec §7.10's joint criterion exists for, and **on this run it
engaged** — the guard (0.9 × running-max 0.643 = 0.579) blocked the
recon-improving evaluations at epochs 219 (−7.3150, NMI 0.579) and 234
(−7.3144, 0.579); `best.pt` (369, NMI 0.595) is the best *guarded*
checkpoint. Note the guard is anchored to the epoch-4 NMI, when z is
closest to t. Reading of the NMI level itself: k-means-39 over a z that
carries 1.2 nats/dim of within-type state, scored against 38 clusters
that are expression-derived from the same counts at a finer granularity —
z splitting on state rather than cluster, not collapse (collapse in M2 was
NMI 0.21 with KL_z → 0).

**Wall time is 69 % figures.** From the log's `[s]` stamps: a plain
5-epoch interval (5 epochs + one evaluation) is 38 s from epoch 30 to
275 (7.6 s/epoch; 56–59 s before epoch 25 and 40–55 s after 279 —
contention with the other jobs on the box that day, unverified). The
**16 figure events took 136 of the 183 min** (235–335 s each until epoch
254, then 540–1,111 s), ~126 min net after subtracting the plain
interval; training + evaluation alone was ~50 min. The motivation's
"~1.5–2 h" was right for training; the 15 TensorBoard figures (+9 UMAP
fits per event since 2026-09-14, per-type panels on up to 8k members ×
8 types) scale with the slide. `--figures-every 100` (a multiple of
`eval_every` 5, already a CLI knob) would make a 500-epoch FF fit
≈ 1.2 h. No code change.

**Caveats, stated.** Single seed (the seed envelope on FFPE was 0.06 in
recon and 0.44–0.50 in cycle_z; nothing here bounds FF's). No battery
yet: Moran / niche / landmarks / pseudotime / atlas / transport are all
CPU-side reads of a model that has not been reloaded since training;
the checkpoint does reload under the current tree (`TrainConfig(**config)`
checked on CPU: `gat_sink`, `weight_decay` take their defaults). Recon
is a within-slide signal only. The GPU footprint is unmeasured. The
graphclust label set makes the invariance guarantee label-set-relative
(the 2026-09-09 control's lesson) and leaves **§2 landmarks and the §5
matrix unrunnable**: `landmark_inventory` matches type *names*
("Endothelial", "Pericyte", "Smooth Muscle", "Tumor Cells"/"Malignant"),
which `Cluster-N` cannot satisfy — the reason the lung and ovarian
graphclust validations stopped at §§3–4 (`validation.json` keys: run,
kappa, seed, morans, niche). Naming the clusters (the GSE315411
pseudobulk path) would unlock them; separate decision.

**Next (proposed, not run; timings scaled from the 407k logs:
`validate_graphclust.log` morans+niche 2.4 min, full battery 9–22 min,
report 23–33 s after load, transport 1–2 min, atlas 1 min; every reload
builds the resident Trainer, so one battery process per card):**
`report` (~0.25 h), `validate --analyses morans,niche --n-perms 1000`
(~0.5 h; 39 types × up to 30k cells × 1000 perms), `atlas` (~0.15 h),
`transport --niches 10` (~0.25 h) — ≈ 1.2 GPU-h sequentially on one
card. Pass for the battery = lung's shape: Moran |I| w ≫ z, niche AUC
w > z > ℓ > floor, cycle row z ≫ w. A seed pair (s1, s2 at 500 / 40 with
`--figures-every 100`, ~1.3 h each) would bound the envelope but is
outside this pre-registration and needs its own motivation entry.

### Leak meter, slide-only per-cell kappa (motivation, 2026-09-17)

**How.** Concept document 16 proposes estimating, from the slide alone and before DisCell runs, how much each cell receives from its neighbours, using the nuclear/extranuclear split already in `qc/nuclear_counts.npz`. A gene's own transcripts sit over the nucleus at retention `eta_g`; leaked transcripts land in the rim at one nuclear share `zeta` common to all genes. For genes only one type `t` makes, pooled over receivers of type `r`, `D_g = N_nuc_g - eta_g(t) X_tot_g = (zeta - eta_g(t)) x leaked`, with leaked modelled as `c_rt Phi_g`, `Phi_g` the face-weighted, 20 um-decayed density of type-`t` neighbours times `t`'s mean profile. Weighted least squares gives `c_rt` and `zeta` per pair; the off-diagonal table is factorised `a_r b_t`, which also predicts the same-type entries no gene can measure; per cell `kappa_i = a_r(i) sum_j f_ij exp(-d_ij/20) dens_j b_t(j) / l_i`. Genes and profiles are learned on half the spatial tiles, the table fitted on the other. Prototype in `discell/experiments/leak_meter.py`; ovarian slide with the 18 curated types and lineage-merged, then the two GSE315411 TMA sections as replication.

**Evaluated by.** The document's pre-registered checks: (1) at least ~30 sender-specific genes per type; (2) the retention of those genes must differ from the fitted `zeta`, else `D_g` carries no signal; (3) `zeta` must be a share in [0, 1] and below the measured nuclear fractions (0.41-0.59 by type, devlog 2026-09-16), since leak is rim-heavy; (4) small `a_r b_t` misfit, else keep the full table; (5) no negative `c_rt`, which no leak model can produce; (6) a credible `kappa_i` envelope around the global kappa = 0.1, with a small tail above 0.9 since `m kappa_i` must be capped below 1. Two this project adds: (7) stability to the learn/fit tile split over five seeds; (8) replication of the tendencies across the two sections of one TMA. A planted unit test gates the estimator: a world where the assumptions hold must return the planted table, `zeta` and per-cell kappa, and a receiver expressing the sender's "specific" genes must push `c_rt` up -- the weak-sender caveat made testable.

**What is wished for.** A fixed leak input, `p_i = (1 - m kappa_i) rho_i + m kappa_i rhobar_i`, turning the sweep axis into `m` and giving a per-cell envelope instead of a global one: spec 7.11's two-view identification at type-pooled level without restructuring the likelihood. A failed check is a finding and is recorded, not tuned away; if the checks fail we keep spec 4.7's swept global kappa and say why.

### Leak meter: fails its own checks on three slides (results, 2026-09-17)

Rejected. The estimator is coded correctly -- planted worlds recover the table to 5%, `zeta` to 0.02, per-cell kappa to RMSE < 0.02 with correlation > 0.99, and reproduce the predicted upward bias when the receiver expresses the sender's genes -- but on real slides the assumptions do not hold and every substantive check fails.

**Specific genes (1).** Ovarian, 18 curated types: 0 of 17 senders reach 30 genes at 10x nuclear share against every other type; maximum 20 (T and NK Cells); 8 senders have none, including Tumor Cells, Tumor Associated Fibroblasts, Proliferative Tumor Cells, VEGFA+ Tumor Cells. The curated labels are lineage-nested -- five tumour subtypes, two fibroblast, two endothelial -- so "specific against all others" is empty by construction for the dominant compartment. Lineage merging to 9 groups gets 1 of 8 past 30 (Endothelium 37), but the merged Tumour class, the largest neighbour on this slide, still has zero specific genes and hence no `b_t`: the dominant leak source is unmeasurable. GSE315411, 35 curated classes: 0 of 34 senders reach 30 on either section (max 8 solo, 5 dual). Ratio 5x gives 2 of 8, ratio 3x gives 5 of 8, and the document warns that specificity against all other types is what made its simulation work.

**Retention vs zeta (2, 3).** In 80-85% of pairs the specific genes' retention is not separable from the fitted `zeta` by 2 SE, so `D_g` is consistent with zero. Fitted `zeta` median 0.746 (IQR 0.454-0.896) at 18 types, 0.699 lineage-merged -- above the measured nuclear fractions of 0.41-0.59, the wrong side. About 11% of pairs put `zeta` outside [0, 1], with standard errors up to 8. Identification rests wholly on the spread of `eta_g` across a pair's specific genes; with 1-37 genes per pair that spread cannot separate the two regression columns.

**The table (5).** Ovarian 18 types: 66 pairs, 12 positive beyond 2 SE, 20 negative, 77% with |c| < 2 SE. Lineage: 56 pairs, 14 positive, 15 negative, 68% within 2 SE. GSE solo 28 of 80 negative; dual 16 of 49.

**Factorisation (4).** R2 = -4.12, relative RMS 2.26 at 18 types; R2 = -0.10, relative RMS 1.04 lineage-merged; -0.23 on the GSE solo section. `a_r b_t` explains less than the table's own mean, and the fallback of keeping the full table does not help because the table is noise.

**Per-cell kappa (6).** At 18 types `kappa_i` median 2.32, IQR [0.40, 5.39], p95 21.2, 66% of connected cells above 0.9, pooled leaked-over-total 7.77 against the global 0.1: cells are claimed to receive several times their own content. Lineage merging gives median 0.129, IQR [0.068, 0.305], pooled 0.143 -- the right order, but see below. Four receiver tendencies are negative at 18 types (Unassigned -32.5) and one is 182 (VEGFA+ Tumor Cells).

**Split stability (7).** Re-randomising only the learn half, seeds 0-4, lineage-merged ovarian: pooled kappa 0.143, 0.490, 0.092, 2.901, 0.060; median `kappa_i` 0.129, 0.422, 0.056, 0.076, 0.043. A 50x spread from a nuisance choice; the seed-0 value near 0.1 is a coincidence of the split, not a measurement.

**Replication (8).** The two GSE315411 sections of one TMA, same core, same 35 classes: receiver tendencies over the 29 types both estimate correlate at Spearman -0.085 (Pearson -0.213). Alveolar macrophages -0.54 vs 130.6, B cells 1.64 vs -60.4, Multiciliated 0.63 vs 24.4. Pooled kappa -1.98 solo vs +4.00 dual -- opposite signs. Only two sender tendencies are estimable on both (Multiciliated 1.70 vs 1.95, Neutrophils 0.64 vs 0.049).

**Two mis-specifications behind the failure, beyond gene scarcity.** Leaked copies come from the rim, so a sender's leaked profile should go as `(1-eta_g) rho_t(g)`, not `rho_t(g)`; the error lies along the same `eta` direction the fit uses. Fitting with the sender's extranuclear profile (ratio 3) gives R2 = 0.10 with 18 of 64 entries still negative -- right in principle, no rescue. And `eta_g(t)` is measured on type-`t` cells that themselves receive leak at share `zeta`, so it is biased toward `zeta`, shrinking `(zeta - eta_g)` and inflating `c_rt` more where density is higher. The density proxy `counts/area` compounds it: its numerator includes the counts that leaked in, so the regressor is contaminated by the response.

**Tendencies vs nuclear fractions.** No relation: Spearman of receiver tendency against per-type nuclear fraction -0.05 (lineage) and -0.06 (18 types); sender tendency 0.32 and 0.40 over 5-6 estimable senders. The 2026-09-16 ordering (TAFs 0.415 lowest, stromal fibroblasts 0.592 highest) is not reproduced, and that was the one external anchor available.

**Kept.** The module stays an instrument, not a model input: about 6 s per slide on CPU, the cheapest way to ask a slide whether a type-pooled leak table is measurable at all. Spec 4.7's swept global kappa and the kappa envelope of handover 4.2 stand unchanged. The honest upgrade path remains spec 7.11 at cell level, a multinomial over 2G bins with kappa latent, where the effective sample is 407k cells rather than the 1-37 genes a pooled pair supplies.

### The two §7.10 degeneracy diagnostics — motivation and pre-registration (2026-09-17)

*Written before anything was run.*

**What is missing.** Spec §7.10 lists seven diagnostics; five are in the battery. Two never were: `I(z;t)/H(t), var(z|t)` and `Δ held-out recon, z vs one-hot t`, both labelled "degeneracy: is z just t?". Every claim the paper makes about `z` carrying within-type state currently rests on the NMI floor and the cycle read, neither of which answers the converse question — whether `z` is *nothing but* a re-encoding of the label it is handed. The large sweeps are about to start, so the diagnostics go in now, before the runs that will be read with them.

**How (1) is measured.** On posterior means `mu_z` (never a sample — the §7.12/T8 convention), a multinomial logistic regression `mu_z -> t` is fitted on the cells of the *training* tiles and scored on the held-out tiles. `I = H(t) - CE_heldout`, where `H(t)` is the held-out cross-entropy of the training-set type frequencies (the intercept-only probe), so `mi_ratio = I/H(t)` is a held-out, probe-based **lower bound** on the normalised mutual information, in [0,1] up to estimation noise. Alongside it the within-type variance fraction `tr Cov(z|t) / tr Cov(z)` (population-weighted pooled within-type covariance over the total, law of total variance), reported overall and per dimension. Both use the `metrics.py` conventions: seeded rng, `MAX_EVAL_CELLS = 30_000` subsample per split.

**Direction, stated plainly.** `within_var_fraction = 0` means `z` is a deterministic function of `t` — degenerate. `= 1` means the type means coincide, i.e. `z` is blind to type, which is the failure the NMI floor already guards. Neither end is the target. A *high* `mi_ratio` on its own is expected and is not evidence of degeneracy: the decoder is given no `t`, so `z` **must** carry type or reconstruction dies.

**How (2) is measured, and the deviation.** The spec's literal reading — a decoder fed `onehot(t)` in place of `z` — needs a retrained model. The cheap post-hoc substitute: held-out per-count reconstruction of the fitted model, minus the same quantity when every cell's `z` is replaced by the mean of `mu_z` over the *training* cells of its type. `w`, the foreign influx `rho_bar`, the leak mixture and `kappa` are exactly as the forward pass left them; only `z` moves. Reported in nats per count. A third line, the held-out score of the empirical per-type count profile, gives the "type lookup and nothing else" reference. Registered as a spec deviation.

**Pre-registered read.** Degenerate would be: `mi_ratio` near 1 **and** `within_var_fraction` near 0 **and** a recon gap near 0 — the three together, never one alone. Healthy would be a high `mi_ratio` with a clearly non-zero within-type share and a recon gap comfortably above zero. **What is wished for** is the healthy pattern on the pinned reference and agreement across the seed triple; a null (a flat gap) is a finding about the model, not a number to tune away, and would say that everything `z` does is carried by the label already.

**Where it lands.** `Trainer.evaluate` so every future fit records both in `metrics.json` (best and final blocks) and `history.jsonl`, plus three TensorBoard scalars beside probe/mirror/cycle. A post-hoc CLI (`python -m discell.model.degeneracy`) reads existing runs from `best.pt`. To be read first on `ablation_gat_type_only_s1` (pinned), its two sibling seeds, and `reference_best` (type_z era).

### §7.10 degeneracy diagnostics — results: z is not just t (2026-09-17)

Both diagnostics built and read on the development slide. All four runs evaluated on CPU (GPUs reserved), ~1 min wall each on 407k cells, no subsampling of tiles needed. The post-hoc `recon_val` reproduces the stored `metrics.json` best value to four decimals for all four runs, which is the reload cross-check.

| run | epoch | I/H(t) | probe acc | within-var frac | recon | type-mean z | **gap** | type profile |
|---|---|---|---|---|---|---|---|---|
| `ablation_gat_type_only` | 59 | 0.818 | 0.871 | 0.696 | −7.2527 | −7.3782 | **0.1255** | −7.3850 |
| `ablation_gat_type_only_s1` (pinned) | 64 | 0.816 | 0.872 | 0.684 | −7.1924 | −7.3404 | **0.1480** | −7.3531 |
| `ablation_gat_type_only_s2` | 59 | 0.799 | 0.855 | 0.675 | −7.2310 | −7.3780 | **0.1470** | −7.3887 |
| `reference_best` (type_z era) | 59 | 0.814 | 0.869 | 0.695 | −7.2586 | −7.3765 | **0.1179** | −7.3850 |

Recon columns are held-out nats per count; artefacts at `runs/<run>/degeneracy.json`.

**The read.** The pre-registered degenerate pattern does not occur. `I/H(t) ≈ 0.80–0.82` is high, as expected — the decoder gets no `t`, so `z` has to carry identity — but the within-type variance fraction is `0.68–0.70`: only about 30% of `z`'s variance lies between type means, 70% is within-type state. And replacing each cell's `z` by its type's mean `z` costs 0.118–0.148 nats per count of held-out reconstruction, with `w`, `rho_bar` and `kappa` untouched. Per-cell `z` decodes information the label does not contain. The seed triple is tight on all three numbers (ratio spread 0.019, within-fraction spread 0.021, gap spread 0.022), so these are run-level properties, not seed noise. The type_z-era `reference_best` is indistinguishable from the type_only seeds, so the GAT-source change did not move degeneracy either way.

**Per-dimension detail.** The within-type fraction runs from ~0.24–0.29 to ~0.96–0.98 across the 20 `z` dimensions in every run: a handful of dimensions are near-pure type axes and the rest are near-pure within-type state. The scalar `0.69` is a mixture of two populations, not a uniform property of the latent — worth remembering when it is quoted as a single number.

**Unplanned finding, worth keeping.** The type-mean-z decode (−7.34 to −7.38) sits within 0.007–0.013 nats per count of a bare empirical per-type profile lookup (−7.35 to −7.39). Once `z` is type-averaged, `w` plus the leak mixture buy almost nothing over a lookup table on `t`. The entire per-cell reconstruction gain of this model flows through `z`. That is consistent with the known `alpha_w = 0.1` near-pinning of `w` (KL ≈ 0.002/dim) and gives it a reconstruction-side number for the first time.

**Caveats on the numbers.** `I/H(t)` is a *linear*-probe lower bound; an MLP probe would raise it and the gap between the two is itself informative (the A2 lesson). `H(t)` is the held-out CE of training frequencies, not the true entropy. The val split follows `config.seed`, so the three seeds are scored on different held-out tiles — `h_t` differs slightly between them (2.258 / 2.265 / 2.353) and the ratios are not scored on a common set. The recon gap is a *substitution*, not a retrain: it answers "what does per-cell `z` add beyond type-level `z`, holding everything else fixed", which is weaker than the spec's one-hot-`t` decoder. The retrain is proposed as a GPU job.

**Incidental.** `reference_best`'s post-hoc NMI came back 0.630 against the 0.658 stored in its `metrics.json`, while recon matched exactly. That points at evaluation-code drift since 8 September rather than a reload fault, but it was not chased down; logged as a watch item.

### Sweep programme across four datasets (motivation and plan, 2026-09-17)

**Motivation, written before anything runs.** Every ablation we have is on one slide. The claim we want to make is not "kappa does not matter" but "the reads are flat along kappa, d_w and alpha_w *where they are flat*, on four slides of three tissues and two preservation methods" — and a grid that disagrees across slides is the finding, not a failure. Two things were missing: a schedule with honest hours, and a tool that can run any of the three ablations on any dataset without hand-edited configs. Both are now in place; no fit has been launched.

**The tool.** `discell/model/sweep.py` takes `--param {kappa,d_w,alpha_w}` with `--values`, and passes every per-dataset knob through (`--label-key`, `--alpha-z`, `--tile-cells`, `--variant`, `--epochs`, `--patience`). Idempotence is unchanged in meaning — a run with a `metrics.json` is finished and is skipped — but the check now happens *before* `assemble()`, so re-running a completed grid costs seconds instead of a graph build. Run names stay `<tag>_<abbrev><value>_s<seed>` with abbreviations `k`/`dw`/`aw`, so `--tag sweep3 --param kappa` reproduces `sweep3_k0.1_s0` exactly and `--tag '' --param d_w` reproduces the hand-launched `dw2_s0`; the report still lands in `experiments/kappa_sweep_sweep3.json`. The report's metrics table and both B-stability legs (across seeds within a value, along the value axis) now work for any param — along-axis pairs with mismatched B shapes are dropped in a d_w sweep rather than reported as a number that does not mean anything — and the §7.10 degeneracy pair and recon gap are read with `.get` from either `final` or the top level of `metrics.json`, yielding `null` on every run fitted so far.

**The inventory.** kappa {0, 0.05, 0.1, 0.2, 0.3, 0.4} x 3 seeds, d_w {2, 3, 6, 8} x 3, alpha_w {0.03, 0.05, 0.07, 0.1, 0.2, 0.3} x 3 = 48 fits per dataset, 42 once the shared centre (kappa 0.1 = d_w 6 = alpha_w 0.1 = the operating point) is fitted once and reused in all three tables. The alpha_w values above 0.1 are kept deliberately: every measurement so far walks *down* from 0.1 and only ever sees the channel open; what over-tightening costs has never been measured under `type_only`.

**What is reusable.** Ovarian only, and only because sweep3 and `dw{2,3,8}_s{0,1,2}` are already `type_only` at 200/20: kappa 18/18 on disk, d_w 12/12 (d_w = 6 is `sweep3_k0.1_s*`), alpha_w 3/18. The `alphaw_*` runs are **not** reusable — their `config.json` has no `gat_sources` key at all (type_z era, pre-2026-09-12) and they were fitted at 500/40, so they are a different architecture at a different budget. `cal2_aw_*` are closed-form-era 40-epoch calibration fits. 15 new ovarian fits; lung, FF and GSE have one reference fit each and nothing else, so 42 apiece. 141 fits in total.

**Hours, from measured runs, not from guesses** (`metrics.json: minutes` / stop epoch): ovarian `sweep3_k0.1_s0` 6.3 min at ep. 79; GSE core `reference` 11.7 min at ep. 94 (7.5 s/epoch, tile 2048); lung `reference_graphclust` 71.9 min at ep. 99 (43 s/epoch, but that run shared the machine with the alpha_w study); FF `reference_graphclust` **182.7 min, early stop at epoch 409** (26.8 s/epoch). Planning figures per fit: ovarian 10 min, GSE 12, lung 25, FF 90 (at a 200-epoch cap). Legs: ovarian alpha_w 2.5 GPU-h, GSE 8.4, lung 17.5, FF 63 — **91.4 GPU-h, about 46 h of wall clock on 2 GPUs**, one fit per GPU. Order, cheapest and most informative first: ovarian alpha_w, then GSE core, then lung, then FF.

**The FF wrinkle, stated rather than hidden.** FF is the only slide whose reference needed 409 epochs. At the sweep budget every FF fit would hit the 200 cap, so FF sweep numbers would be internally comparable but not comparable to FF's own reference; at FF's own 500/40 budget the leg costs 128 GPU-h instead of 63. Grids are not being shrunk for cost — the author has said compute is not a constraint — and memory does not force a window either: with int16 resident counts FF sits at 18–20 GB of a 24 GB card, so one FF fit per GPU and no `--max-cells` (a flag that does not exist).

**GSE315411, decided by the author.** The sweeps run on the `pdl018d` core of the solo section (69k cells, curated 35 classes, alpha_z 0.0036, tile 2048); the two full slides are the held-out-section test via `crossslide`, an evaluation of a checkpoint rather than a fit, and are not swept.

**Launcher pattern.** Detached (`setsid nohup … > log 2>&1 < /dev/null & disown`), `OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8`, `--figures-every 200` so the x18 figure panel runs at most once per fit, and the two GPUs given disjoint `--seeds` staggered by two minutes so the two `assemble()` passes do not peak together. Idempotence is per run name, so disjoint seed sets never collide. Measured reason for all of it: `dw8_s0` took 45 min for the same 59-epoch best that `sweep3_k0.1_s0` reached in 6.3 min, under contention.

**Pre-registered pass/fail for the programme as a whole.** Pass = on each further slide the kappa envelope is flat on cycle_z, cycle_w, probe and mirror with likelihood falling monotonically past 0.1 (the ovarian §4.2 shape), d_w shows fill ≪ d_w with no gain above 6, and alpha_w shows the channel opening downward with the bulk of w staying context-dominated. Fail = any slide where a read moves along kappa outside its own seed envelope, or where d_w > 6 buys something, or where alpha_w's shape inverts. Either way the number goes in the table; nothing is tuned away, and the four-slide disagreement, if it comes, is the paper's limitation section.

### Sweep programme: the tool is in, nothing is launched (results of the planning turn, 2026-09-17)

Result half of the entry above, for the planning work itself — no fit ran, so these are tool and inventory facts, not science.

**Verified.** `uv run pytest -q tests/test_model_sweep.py` → 8 passed in 0.76 s: the three default grids and the int cast for d_w; per-dataset knobs (`--variant pdl018d --alpha-z 0.0036 --tile-cells 2048 --label-key … --epochs --patience`) reaching `TrainConfig` with the swept knob overriding the fixed one; run names equal to what is on disk (`sweep3_k0.1_s0`, `sweep3_k0_s2`, `dw2_s1`, `aw0.05_s0`); the skip filter and `--force`; `metric_row` on a full metrics.json, on an old one (the pre-rename `cycle.ceiling` key, `mirror`/`degeneracy` absent → `None`), and with the degeneracy pair at the top level instead of under `final`; `b_stability` returning 1.0 for a column permutation of the same B, dropping a single-seed value from `across_seeds`, and dropping shape-mismatched pairs in a d_w sweep.

**Smoke-tested against real artefacts.** `--param kappa --tag sweep3 --values 0.1 --seeds 0 --report-only` on the ovarian slide loaded `sweep3_k0.1_s0` and reproduced the handover's partial-isolation reading: `recon_lost0` −7.2594 vs `recon_lost1plus` −7.0421, `recon_degree_le2` −7.1315 — cells that lost edges to the prune still reconstruct better. `experiments/kappa_sweep_sweep3.json` was backed up before that call and restored after, and is back at its 18 runs.

**Not verified.** No multi-run report on a full grid (that loads 18 models and was not worth the GPU time while the plan is unapproved); no fit at any budget; the lung and FF per-fit estimates are extrapolations from single reference runs, the lung one from a run that shared the machine, so both carry roughly a factor of two of uncertainty. The degeneracy keys are read from two plausible places because the agent adding them has not documented where they land; if they land under a third key the sweep report will show `null` and needs a one-line fix.

**Scheduling dependency, flagged for the author.** If the programme launches before the §7.10 degeneracy metrics land in the per-run evaluation, all 141 runs will be written without them and will need a re-evaluation pass. Recommendation: land those metrics first.


### Transcript-flux β and per-cell κ_i from extranuclear transcript geometry (motivation, 2026-09-17)

**Why.** Today β_ij ∝ face_ij · exp(−d_ij/τ) tracks only centroid distance and
Voronoi face length, and κ is one global number swept on a grid. The
per-transcript nucleus flag (already the basis of `qc/nuclear_counts.npz`)
says which of a cell's transcripts are extranuclear, and their coordinates say
where they sit relative to the cell's own nucleus and to every neighbouring
nucleus. The doc-16 leak meter, which pooled this information to a type-by-type
table through sender-specific genes, failed its own checks (entry above); this
design uses the geometry directly and needs neither labels nor specific genes.
Agreed with the author 2026-09-17: build β first, derive κ_i from it, change
nothing in the model until the checks pass.

**How.** For each extranuclear transcript (q ≥ 20) assigned to cell i: distance
to i's own nucleus, distance to the nearest other nucleus, and that neighbour's
identity, restricted to i's pruned Voronoi neighbours. Per edge (i, j): the
flux F_ij = weighted count of i's extranuclear transcripts in the wall band
facing j, weighted by how much closer they lie to j's nucleus than to i's
(exact weighting and band width are the prototype's parameters, reported, with
a sensitivity check). β^T_ij = F_ij / Σ_j F_ij on connected cells; κ_i = Σ_j
F_ij / ℓ_i, an upper bound. Face length and centroid distance become covariates
that predict F_ij, not the kernel itself.

**Evaluated by (pre-registered).** (1) Homotypic null: along walls to same-type
neighbours deep inside homotypic regions the band should carry no *excess* —
its gene content should match the cell's own profile; along heterotypic walls
the banded transcripts should resemble the neighbour's type profile more than
the cell's own (cosine to type means, per edge class). (2) Replication: κ_i by
type and the β^T-vs-β_face relation must agree between the two GSE315411
sections of one core (Spearman over shared types ≥ 0.7 is the bar). (3) β^T
must correlate with, but not equal, the face-length β (edge-level Spearman
reported; identity would mean geometry adds nothing). (4) The κ_i distribution
must be a share: median well below 1, tail above 0.9 small, and its type
ordering compared with the per-type nuclear fractions of 2026-09-16 (TAFs
lowest) as the one external anchor. (5) Sensitivity to band width and weight
form: conclusions must not flip across a small grid of both.

**What is wished for.** A per-edge β and a per-cell κ_i measured from the slide
that pass (1)–(5); then the model change is a vector κ_i with a cap and the m
sweep {0, 0.5, 1, 1.5, 2}. A failed check is a finding: if the homotypic null
fails or the two sections disagree, β_face and the global κ sweep stand and the
reason is recorded here. CPU only; no fit is touched.

### Transcript-flux β and per-cell κ_i: results (2026-09-17)

*Delegated prototype (Opus), refuted (Sonnet), then extended with the power
analysis the refuter demanded; numbers verified against the JSONs by the
refuter.* Built and read on three slides, CPU only: 27 s to stream the ovarian
slide's 147.7M transcripts, ~35 s per slide including the power analysis.
Checks (2)–(5) pass. **Check (1) fails against the pre-registered bar — and the
power analysis run afterwards shows the bar was mis-specified, not the
estimator.**

**Parameters, fixed before the runs.** Nucleus reference = the vertex mean of
`nucleus_boundaries.parquet`. Admission = `overlaps_nucleus == 0`, `qv ≥ 20`,
gene in panel, host has a nucleus. Signed offset `s = d_own − d_nearest other
nucleus`, the nearest taken over the host's 40 µm-pruned Voronoi neighbours
that also have a nucleus. Weights `hard = 1[s>0]`, `ramp = clip((s+b)/2b)`,
`logistic = 1/(1+exp(−4s/b))`; primary b = 4 µm, ramp; the 3 × 3 grid
b ∈ {2,4,8} accumulated in one stream pass. `ℓ_i` = the bundle's assigned
total; τ = 20 µm for the face comparison. Gene content on 200k subsampled
directed edges (of 2.31M ovarian, ~0.40M per GSE core) with ≥ 10 banded
transcripts. No transcripts were subsampled. `hard` is band-invariant by
construction, so three of the nine grid rows coincide.

**Scale.** Ovarian 147.7M transcripts → 49.31M extranuclear q20 placed,
400,534 connected cells, 2.31M directed slots, 401,400/407,120 with a nucleus.
GSE `pdl018d` solo 237.5M → 12.40M placed, 68,601 of 69,422; dual 210.7M →
11.45M, 69,919 of 70,757. Every raw file existed.

**(1) Gene content — fails the bar; the bar was a 50 % bar.** Heterotypic
edges carry content whose cosine excess (neighbour − own) is −0.036 ovarian,
−0.056 GSE solo, −0.050 GSE dual, with the neighbour winning on 35 %, 27 %,
27 % of edges (n ≈ 21–23k, median band 22–24 transcripts). Read literally,
the band looks like the host: fail. **A power analysis on the same edges, with
the same counts and the same pooled profiles, planting band content as
`(1−p)ρ_host + p·ρ_neighbour`, says this reading was wrong.** The p = 0 world
does not score zero — it scores −0.046 (ovarian), −0.072 (solo), −0.063
(dual), because a cosine against L1-normalised pooled profiles is not centred.
The pre-registered bar (excess > 0) is only crossed at **p = 0.513, 0.501,
0.500**: it asked for a half-and-half band. Against the correct baseline the
observed values sit **16.9 σ, 29.4 σ and 25.7 σ above p = 0** and imply an
admixture of **p = 0.109 [0.092, 0.128]**, **0.112 [0.103, 0.122]** and
**0.107 [0.096, 0.118]**. The second moment, `frac_neighbour_gt_own`, implies
0.264 / 0.256 / 0.251 on the same edges — a factor 2.3 higher, so a single-p
two-profile mixture does not fit both moments and the honest statement is a
**bracket of 0.11–0.26**, not an interval. So: the check as written has no
power at the 10–15 % the model posits and full power only near 50 %; rebuilt
against its own null it separates 13 % from 0 at many sigma and points at an
admixture close to the geometric κ_i. The pre-registered verdict stands as a
fail; the inference the first draft drew from it — that the band is not
foreign — is withdrawn.

Two follow-ups, both recorded. The **within-cell control** (band vs the same
cell's deep extranuclear transcripts, both scored against the neighbour) gives
a difference-in-differences of −0.0089 [−0.0178, +0.0009] ovarian, −0.0042
[−0.0086, +0.0003] solo, −0.0056 [−0.0105, −0.0005] dual over 1000 edge
bootstraps: null on two slides, marginally negative on the third. **No power
curve was built for this statistic**, so by the argument above its null is not
yet interpretable and is not counted either way. The **circularity** the
pooled profiles carry — they already contain leaked content — was estimated by
deflating each type's profile by an assumed 13 % of its flux-weighted influx
and re-scoring: the observed excess moves by −0.0023 / −0.0047 / −0.0042 and
the planted p = 0.13 curve moves by almost exactly the same amount, so the
identifying contrast shifts by under 0.01 in p. Real, small, common-mode, and
conservative in direction.

**(2) Replication — passes.** Per-type median κ_i over the 32 shared classes
of the two sections of one core: **Spearman 0.843**, Pearson 0.873 (bar 0.7).
Multiciliated 0.176/0.178, Secretory 0.183/0.187, Neuroendocrine 0.175/0.182,
Pericytes and Neutrophils identical to four decimals; the only real
disagreements are B cells (0.133/0.069) and alveolar macrophages
(0.046/0.098). The β^T-vs-β_face relation replicates at 0.4945 vs 0.4952. The
doc-16 leak meter scored −0.085 on this same pair.

**(3) β^T vs β_face — passes.** Edge-level Spearman 0.485 ovarian (1.27M live
slots), 0.494 / 0.495 on the GSE pair: correlated, not identical.

**(4) κ_i is a share — passes; the external anchor agrees in sign.** Ovarian
median 0.130, IQR [0.064, 0.198], p95 0.303, no cell above 0.9 (GSE solo:
1.5e−5 of cells); GSE solo 0.126 [0.065, 0.196] p95 0.315; dual 0.127
[0.067, 0.196] p95 0.316. Spearman of per-type median κ against the
2026-09-16 pooled nuclear fractions is −0.465 (18 types), −0.269 (32), −0.493
(33): lower nuclear fraction, higher κ, on all three slides. TAFs, lowest
nuclear fraction 0.415, κ 0.146; stromal fibroblasts, highest 0.592, the
**lowest** κ of the 18 at 0.039 — the anchor the leak meter could not reproduce.

**(5) Sensitivity — passes.** Over the 3 × 3 grid ovarian median κ_i moves
0.076 → 0.171 and pooled 0.165 → 0.204; `frac κ_i > 0.9` ≤ 2e−5 in every cell
of the grid; Spearman(β^T, β_face) 0.374 → 0.536. The GSE pair tracks it and
solo/dual agreement holds at every point. No conclusion flips — but the band
sets the absolute level of κ_i to within a factor 2.3, so only the *ordering*
is band-free.

**What can be said about κ = 0.1, stated carefully.** κ_i is an **upper
bound**: every transcript nearer a neighbour's nucleus than its own is counted,
and nuclei are off-centre in elongated and nuclear-expansion-segmented cells,
so a slide that leaked nothing would still return κ_i > 0. Its level depends
on the band (median 0.076–0.171 across the grid). With those two caveats
carried, three slides across two tissues and two preservation methods put the
median at 0.126–0.130 and p95 at 0.30–0.32 with no tail at 1, and the
independent gene-content estimate brackets the admixture at 0.11–0.26. That is
**order-of-magnitude consistency** with spec 4.7's swept κ = 0.1 — enough to
say the operating point is not off by a factor of five in either direction, not
enough to call κ = 0.1 measured.

**Read.** A label-free, model-free per-cell geometric statistic that replicates
across sections, tracks the nuclear-fraction anchor, and is corroborated on gene
content by an independent estimate of the same order. Still missing: a
check-(1) design with power at 13 % by construction (type-specific genes,
scored against the simulated p = 0 null), a control that separates influx from
nucleus off-centring (shuffled nuclei / own polygon), and the per-edge
agreement between geometric κ_i and content-implied p. Until those exist, κ_i
stays an instrument: `equations.leakage_mix` keeps its scalar κ, `prepare`
keeps β_face. Module `discell/experiments/transcript_flux.py` (`--power`),
10 planted tests; artefacts `experiments/transcript_flux_{full,pdl018d}.json`
on the three datasets.

### Transcript-flux κ_i: the three decisive tests before adoption (motivation, 2026-09-17)

**Why.** The prototype (entry above) passed replication, share and sensitivity
and, once its gene-content check was scored against the correct null, implied
an admixture of ≈ 0.11 on three slides. Three things still separate "geometric
covariate" from "measured leak": the content check has power at 13 % only by
post-hoc simulation; the level of κ_i is not separated from nucleus
off-centring; and the two estimators have never been compared on the same
edges. Todo items 5.4–5.7. No model change is made in this round.

**How, evaluated by, pre-registered bars.**
- **5.4 (W-tf1) content check with power by design.** Score the per-edge band
  content on type-specific genes only (`leak_meter.specific_genes` at the 3×
  ratio, plus the top-N per type by fold-change so every type has genes),
  against the simulated `(1−p)ρ_host + p·ρ_nb` curve at the observed counts.
  Bar: the zero-crossing of "neighbour > own" must fall at p ≤ 0.15 on each
  slide; report the implied p with 2-SE and its agreement with the whole-panel
  estimate (0.11 [0.09, 0.13]). Failure = the crossing stays above 0.3, in which
  case the content statistic cannot be made decisive at these counts.
- **5.5 (W-tf2) off-centring control.** Recompute the signed offset with (a)
  nucleus positions shuffled among cells within 200 µm tiles (destroys the
  transcript–nucleus association, keeps the density field) and (b) the cell's
  own polygon centroid instead of its nucleus. Bar: the tile-shuffled κ_i gives
  the level a leak-free geometry would produce; the corrected κ_i = observed −
  shuffled must stay a share with median > 0 on all three slides and keep the
  type ordering (Spearman with the uncorrected ordering ≥ 0.8). Failure =
  corrected median ≤ 0.02, i.e. the level was geometry.
- **5.6 (W-tf7) per-edge agreement.** On heterotypic edges with ≥ 30 banded
  transcripts, the geometric flux share F_ij/ℓ_i against the content-implied
  admixture p_ij (from 5.4's statistic, per edge, binned by flux decile to
  average the multinomial noise). Bar: Spearman ≥ 0.5 across deciles on each
  slide and a calibration slope in [0.5, 2]. Failure = no monotone relation.
- **5.7 (W-tf9, W-tf8) moment fix and the DiD.** Host arm = the cell's
  extranuclear profile; bar: the two moments' implied p agree within their
  2-SE. DiD: build its power curve or drop the statistic; report which.

**What is wished for.** All three bars met → the model change of todo 5.8 (a
per-cell κ vector capped at 0.5, β^T and κ_i on the graph, `m` swept over
{0, 0.5, 1, 1.5, 2}) is built and one ovarian fit at m = 1 is compared with the
κ = 0.1 reference. Any failed bar is recorded and stops the adoption; the
instrument stays. CPU only, ovarian first, then both GSE sections.

### Transcript-flux κ_i: the three decisive tests — results (2026-09-17)

*Delegated (Opus), CPU only, 27–100 s per slide; key numbers spot-checked
against the artefacts. Written against the bars of the entry above; all three
fail.* Module `discell/experiments/transcript_flux.py` (`--decisive`), 7 new
planted tests (17 in the file, all passing); artefacts
`experiments/transcript_flux_decisive_{full,pdl018d}.json` on the three slides.
One streaming pass fills four distance references at once, so shuffled, polygon
and observed κ_i are measured on exactly the same transcripts.

**Parameters, fixed before the runs.** Primary band 4 µm, ramp.
`leak_meter.specific_genes` at ratio 3 (floor 200 nuclear counts), topped up to
N = 25 genes per type by nuclear-share fold change so no type is empty: 515 of
5,001 genes (ovarian), 898 / 896 (GSE). Shuffle tile 200 µm. 5.6 edge floor 30
banded transcripts. p grid {0, .02, .05, .10, .15, .20, .30, .50, 1}; DiD grid
{0, .05, .13, .30}; seed 0. All directed slots of the GSE cores and 1.0M of the
ovarian slide's 2.31M sampled for content.

**5.4 — fails, and the bar is refuted as a criterion, provably.** Zero-crossing
0.514 / 0.523 / 0.536 (bar ≤ 0.15). For a band drawn from `(1−p)ρ_host +
p ρ_nb` and scored by `cos(v, ρ_nb) − cos(v, ρ_host)`, the p = 0.5 mixture is
invariant under swapping the arms, which maps the excess to its own negative,
so its mean is exactly zero at p = 0.5 for any gene subset. Specificity
sharpens the curve; it cannot move the crossing (confirmed on planted worlds,
full panel and 25-gene subset). What specificity bought: p = 0.13 separates
from p = 0 at 9.1 / 13.0 / 10.0 σ on 1.7–2.9k edges (≈ 10× the information per
edge), implied admixture 0.160 [0.133, 0.184] / 0.162 [0.143, 0.178] /
0.101 [0.063, 0.126], consistent with the whole-panel 0.11 [0.09, 0.13]; the
second moment still says 0.23–0.27.

**5.5 — fails both bars; the polygon arm is the informative failure.** The
pre-registered tile shuffle permutes nucleus positions among cells, which makes
each cell's own nucleus a random point ~100 µm away: κ_shuffled 0.393 / 0.480 /
0.478, corrected κ_i −0.240 / −0.330 / −0.328 (negative for 84–85 % of cells),
ordering Spearman 0.32 / 0.40 / 0.15. That control is a destroy-everything
null, not an off-centring null. The other arm settles it: replacing the nucleus
by the cell's own polygon centroid moves κ_i only 0.1305 → 0.1173, 0.1263 →
0.1068, 0.1274 → 0.1081 — the off-centring share is 0.006 / 0.011 / 0.011,
ordering Spearman 0.70 / 0.76 / 0.80. A diagnostic beside the two arms,
permuting the nucleus-minus-centroid offset inside the tile, reproduces the
observed κ_i within 3 % (0.1267 / 0.1217 / 0.1223) at ordering Spearman
0.98 / 0.95 / 0.98. **κ_i's level is insensitive to the transcript-to-own-
nucleus relationship it is built from.**

**5.6 — fails; the decisive one.** 37,094 / 17,857 / 16,436 heterotypic edges
with ≥ 30 banded transcripts, ten flux-share deciles over a five-fold range.
The banded count rises with the flux share (37 → 85 ovarian) and the cosine
excess is count-dependent, so a single inversion curve manufactures a negative
relation (Spearman −0.89 / −0.42 / −0.69); each decile therefore gets its own
planted curve at its own counts (recorded as W-tf11). Count-matched:
**Spearman −0.539 / +0.685 / +0.539, slope −0.263 / +0.228 / +0.293** (bars
≥ 0.5 and [0.5, 2]); the Spearman sign is unstable across sample size
(ovarian +0.442 at 200k edges). Every slope is 4–10× below the calibration
band: a five-fold change in geometric flux share moves the content-implied p
by at most 0.05. **The geometric flux share does not say which edge carries
foreign content.** The same machinery recovers a planted per-edge admixture at
Spearman ≥ 0.5 and slope in [0.5, 2], so this is the slide's answer.

**5.7 — moments still disagree; the DiD has its power curve.** Host arm on
the pooled extranuclear profile: excess-implied p 0.136 [0.130, 0.143] /
0.102 [0.095, 0.109] / 0.087 [0.080, 0.094] vs fraction-implied 0.250 / 0.227 /
0.216, no 2-SE overlap; ratio still ~1.8–2.5. The DiD power curve (band at p,
core at 0) moves +0.0023 → +0.0085, +0.0107 → +0.0206, +0.0107 → +0.0179
between p = 0 and 0.13, separating at 2.0 / 3.8 / 2.8 σ, so the statistic has
power — but the observed DiD is −0.0108 [−0.0151, −0.0068], −0.0056 [−0.0089,
−0.0027], −0.0058 [−0.0093, −0.0024]: significantly negative on all three and
3–6 σ below the planted leak-free world. Wrong sign for influx; logged as an
open anomaly (W-tf10), counted for neither side.

**Read.** Three tests designed to separate "geometric covariate" from
"measured leak"; none makes the separation and one argues against it. κ_i
replicates across sections, tracks the nuclear-fraction anchor and sits at the
right order of magnitude, but its level does not depend on the nucleus (5.5)
and its per-edge value does not predict the content it stands for (5.6). The
content estimator cannot carry a crossing bar (5.4, proved) and does not fit a
single-p mixture on either host arm (5.7). **Todo 5.8 is not built.**
`equations.leakage_mix` keeps its scalar κ; `prepare` keeps β_face; the swept
global κ = 0.1 stands, now with an independent order-of-magnitude bracket
(0.09–0.25) behind it. κ_i and β^T stay instruments and reporting statistics.
What would reopen adoption is 5.6 alone: a per-edge relation with slope in
[0.5, 2] on at least two slides, using a content statistic that is not an
uncentred cosine — a per-edge multinomial mixture likelihood in p on the
specific genes, host arm from the same cell — plus an explanation of W-tf10.

### NMI drift check (todo 1.3, 2026-09-17): closed

Reloading `ablation_gat_type_only_s1` twice through `discell.model.degeneracy`
on CPU gives NMI 0.6538918662562847 both times, equal to the stored best value
to full precision; recon likewise. The NMI path is deterministic on the same
weights, so the 0.630-vs-0.658 mismatch on `reference_best` (trained 8 Sep) is a
code change between 8 and 11 September, not metric noise. Since no pre-settle
run will be quoted, nothing further is done; new fits are internally consistent.

### What does w buy once z is type-averaged? (todo 2.3, motivation, 2026-09-17)

**Why.** The degeneracy battery found that replacing each cell's z by its type
mean lands within 0.007–0.013 nats/count of a bare per-type profile lookup:
once z is type-averaged, w plus the leak mixture add almost nothing over
knowing t. That is the first reconstruction-side number on the α_w = 0.1
pinning and it decides how 2.2 (α_w = 0.05) should be read.

**How.** Post-hoc, CPU, on runs already on disk: the type_only seed triple
(α_w 0.1), the α_w study runs `alphaw_0.02/0.03/0.05/0.07` and `reference_best`
(type_z era, same α_z, seed 0), and `dw2_s0`, `dw8_s0`. For each, held-out
per-count recon under five decodes with ρ̄ and κ fixed: (a) full model; (b) z →
type mean, w posterior; (c) z → type mean, w → m_ψ(c,t) (context field only);
(d) z → type mean, w → m_ψ(c̄_t,t) (reference context: kills the context
dependence, keeps the per-type gauge offset); (e) per-type count profile. Read
the gaps (b)−(d) = what context-varying w buys, (d)−(e) = what the gauge offset
plus decoder nonlinearity buys, (a)−(b) = what per-cell z buys. Also the same
five with z kept and w → m_ψ(c̄_t,t), i.e. the context channel switched off with
z intact — the number that says what the response channel is worth to the
likelihood at each α_w.

**Evaluated by.** Pre-registered read: if (b)−(d) grows monotonically as α_w
falls (0.1 → 0.02) and is ≥ 0.01 nats/count at 0.05, the context channel
carries likelihood that 0.1 suppresses and 2.2 is worth running; if it is flat
near 0 at every α_w, w is a lookup-level covariate at every tested operating
point and 2.2's acceptance should weigh the z side only. Type_z-era runs are
compared only among themselves.

**What is wished for.** A single table, α_w × decode, from which the α_w = 0.05
question can be read before any new fit. Nothing is tuned; a null is recorded.

### What does w buy once z is type-averaged? (todo 2.3, results, 2026-09-17)

*Delegated (Opus), CPU only, ~1 min per run on 407k cells (60.4k held-out
seeds).* Ten runs on disk, six decodes each, ρ̄ and κ held fixed from the
forward pass, posterior means throughout; every decode goes through
`Trainer._decode_seeds`, which gained an optional w argument (planted test: fed
the cell's own μ_w it returns log_p to 1e-6). Artefact
`experiments/w_contribution.json`. Cross-check: decodes (a) and (b) reproduce
the §7.10 table to four decimals on the four shared runs.

| run | α_w | (a) full | (b) type-mean z | (c) + m_ψ(c,t) | (d) + m_ψ(c̄_t,t) | (e) type profile | (f) own z, ref w |
|---|---|---|---|---|---|---|---|
| alphaw_0.02 (type_z) | 0.02 | −7.2518 | −7.3597 | −7.3650 | −7.3842 | −7.3850 | −7.2713 |
| alphaw_0.03 (type_z) | 0.03 | −7.2510 | −7.3633 | −7.3669 | −7.3848 | −7.3850 | −7.2680 |
| alphaw_0.05 (type_z) | 0.05 | −7.2534 | −7.3690 | −7.3706 | −7.3856 | −7.3850 | −7.2653 |
| alphaw_0.07 (type_z) | 0.07 | −7.2586 | −7.3748 | −7.3752 | −7.3874 | −7.3850 | −7.2677 |
| reference_best (type_z) | 0.10 | −7.2586 | −7.3765 | −7.3769 | −7.3887 | −7.3850 | −7.2673 |
| type_only s0 | 0.10 | −7.2527 | −7.3782 | −7.3783 | −7.3869 | −7.3850 | −7.2585 |
| type_only s1 | 0.10 | −7.1924 | −7.3404 | −7.3409 | −7.3499 | −7.3531 | −7.1995 |
| type_only s2 | 0.10 | −7.2310 | −7.3780 | −7.3785 | −7.3877 | −7.3887 | −7.2377 |
| dw2_s0 | 0.10 | −7.2564 | −7.3788 | −7.3792 | −7.3866 | −7.3850 | −7.2617 |
| dw8_s0 | 0.10 | −7.2645 | −7.3745 | −7.3751 | −7.3858 | −7.3850 | −7.2719 |

Gaps (b)−(d) context-varying w / (d)−(e) gauge + nonlinearity / (a)−(b)
per-cell z / (a)−(f) context channel with z intact: 0.0244 / 0.0008 / 0.1080 /
0.0195 (α_w 0.02); 0.0215 / 0.0002 / 0.1123 / 0.0170 (0.03); 0.0166 / −0.0006 /
0.1156 / 0.0119 (0.05); 0.0126 / −0.0025 / 0.1162 / 0.0091 (0.07); 0.0122 /
−0.0037 / 0.1179 / 0.0087 (reference_best); type_only triple (b)−(d) 0.0087 /
0.0095 / 0.0097; dw2 0.0078, dw8 0.0113.

**The pre-registered read fires on the first branch.** (b)−(d) grows strictly
monotonically as α_w falls — 0.0122, 0.0126, 0.0166, 0.0215, 0.0244 over
0.1 → 0.02 — and is 0.0166 at 0.05, above the 0.01 bar. The context channel
carries held-out likelihood that α_w = 0.1 suppresses; 2.2 is worth running.
The level stays modest: at 0.05 the context-varying part of w is worth
0.017 nats/count against 0.116 for per-cell z; even at 0.02 it is a fifth of
the z channel.

**(d)−(e) is zero everywhere** (−0.0037 to +0.0032, no α_w trend): the
per-type gauge offset, the decoder nonlinearity and the leak mixture together
buy nothing measurable over an empirical per-type lookup. Everything w
contributes above the lookup is the context-varying part.

**(b)−(c) reads the pinning on the likelihood** for the first time: 0.0001–
0.0006 at α_w = 0.1 (posterior w and prior field decode identically — the
KL_w ≈ 0.002/dim result in nats per count), rising to 0.0053 at 0.02.

**Total recon does not move; the split does.** (a) spans 0.008 across the five
α_w values while its seed spread at fixed α_w is 0.06. As α_w falls, (a)−(b)
shrinks 0.118 → 0.108 by about what (b)−(d) gains. α_w decides which channel
carries the likelihood, not how much there is — so recon cannot adjudicate 2.2.

**Caveats.** Era: at α_w = 0.1, type_z gives (b)−(d) 0.0122 against the
type_only triple's 0.0087–0.0097 (offset 3× the seed spread of 0.0010), so the
trend is read inside type_z only and 0.05 under type_only should be expected
nearer 0.013–0.014. The four α_w runs are seed 0 only. Splits follow
`config.seed`, so the type_only triple is scored on three held-out sets; the
nine seed-0 runs share one. Budgets 500/40 except dw2/dw8 at 200/20, not
binding (best epochs 59–64). c̄_t is the training mean of c per type, isolated
flag included. Substitution, not retrain.

**How 2.2 is read.** α_w = 0.05 under type_only, 3 seeds, is accepted or
rejected on the w side — KL_w per dim off the floor, larger within-type-centred
context dependence of w, a second context axis reproducing across seeds (cosine
≥ 0.75, the 2026-09-14 pre-registration) — with the guards (mirror R², probe
ΔCE at floor, NMI and cycle_z inside the seed envelope) as veto, and this table
as the prior that the prize is ~0.013–0.014 nats/count, ~10 % of the z channel.
If the guards move at 0.05, the trade is not worth it.

### α_w = 0.05 under type_only, three seeds (todo 2.2, motivation, 2026-09-17)

**Why.** 2.3 says the context channel doubles its held-out contribution
between α_w 0.1 and 0.05 while total likelihood is unchanged, and the
2026-09-14 rank analysis says a second context axis becomes seed-stable at
0.05. Both were measured in the type_z era; the current architecture has never
been fitted below 0.1. **How.** `alphaw0.05_type_only_s{0,1,2}`: defaults
(type_only, κ 0.1, α_z 0.007, α_a 0.3, d_w 6), `--alpha-w 0.05`, 500 epochs /
patience 40, `--figures-every 500`, detached, one process per GPU; then
validate / atlas / report on each. **Evaluated by (pre-registered
2026-09-14, restated).** Accept as the new operating point only if, over the
three seeds: axis-2 share of cov(μ_w) ≥ 15 % with cross-seed axis-2 cosine
≥ 0.75; KL_w per dim above the 0.1 floor; the within-type-centred context
dependence of w larger than at 0.1; and the guards inside the type_only-triple
envelope — cycle_z ≥ 0.44, recon within ±0.06, probe ΔCE at floor, mirror R²
≤ 0.05, NMI ≥ 0.63; w-side battery rows (niche AUC, Moran) not below 0.1's.
Any guard outside the envelope on ≥ 2 seeds = reject, α_w stays 0.1.
**What is wished for.** A cleanly accepted 0.05 before the sweeps, or a clean
rejection with the numbers; nothing in between is tuned.

### Unattended queue for the 3–4 day window (motivation, 2026-09-17)

**Why.** The author is away 3–4 days; compute is not a constraint; every fit
is idempotent. Everything below is a sweep the programme already
pre-registered (docs/sweep_programme.md §1, §5; docs/todo.md §0, §6) or a
battery on a fit that exists. Nothing new is decided by the queue.

**What runs** (`scripts/queue_2026-09-17.sh`, two lanes, one per GPU, strictly
sequential per lane, resumable by relaunch; logs under `data/queue_logs/`):
1. waits for the three `alphaw0.05_type_only` fits (todo 2.2), then their full
   battery (validate incl. landmarks/matrix, atlas, transport, report) and the
   FF `reference_graphclust` battery (§§3–4 only; cluster labels);
2. degeneracy diagnostics on the 18 sweep3 κ runs (CPU, todo 1.2);
3. sweeps under tag `sweep3`, α_w legs first so they are useful whichever way
   2.2 goes, then κ, then d_w: ovarian α_w {0.02…0.3} (21 fits, 200/20);
   GSE core all three grids (51 fits, 200/20, α_z 0.0036, tile 2048); lung all
   three (51, 200/20, graphclust, α_z 0.004); FF all three (51, **500/40**,
   graphclust, α_z 0.0007, figures every 500). Grids are self-contained (the
   centre κ 0.1 / d_w 6 / α_w 0.1 is refitted under each param's run name; 6
   duplicate fits per dataset, accepted for a clean table);
4. one `--report-only` pass per (dataset, param) and the GSE cross-slide leg
   (every swept core checkpoint evaluated on the dual section).

Estimated 45 h (lane A) / 56 h (lane B) of wall clock. **Caveat, stated
before launch:** the κ and d_w legs run at the α_w = 0.1 centre; if 2.2
accepts 0.05, those two legs are rerun at the new centre and these serve as
the 0.1 comparison. Pass/fail rules for every leg are the programme's §7.10
battery reads and the pre-registered stability rule (sign and rank unchanged
across the grid, seed envelope excludes zero); read on return, nothing tuned.

### DNA-content cell-cycle label: analysis of the DAPI gating script (todo 3.5, 2026-09-17)

*Delegated (Opus), CPU only; script `scripts/test_cell_cycle.py` not modified;
artefacts `experiments/cell_cycle_dapi_analysis.{json,png}` on the ovarian
slide; area slope and R² spot-checked.* Item 3.5 opened on the hypothesis that
the 4N gate is contaminated by overlapping / stacked nuclei, correctable with
the nucleus and cell polygons. **Measured: the hypothesis is false as stated,
and the instrument fails for a deeper reason.**

*Overlap does not exist in 2D.* Exact pairwise intersection over all 414,533
nucleus rings: 741 nuclei (0.179 %) have any positive overlap, overlap area
1.28e-6 of nuclear area; among the 512 that reach the gate, 4N rate 0.656 vs
0.442 — a real enrichment on < 0.2 % of cells. A cell polygon contains part of
a foreign nucleus for 2.53 % of cells at ~0 area share. Xenium's nucleus masks
are a partition. The one material geometric effect is **multinucleate cells**:
2.9 % of cells carry ≥ 2 nucleus polygons, 97 % of them land in the 4N gate,
6.4 % of the 4N population. Axial stacking inside the ~5 µm section is
invisible to 2D polygons by construction.

*What integrated DAPI measures here is nuclear footprint area.* log dapi_sum on
log nucleus area: slope 1.035, R² 0.734; dapi_mean vs area r = −0.078. On top:
~12 % of every integral is un-subtracted background; median dapi_mean drifts
2.65× across 500 µm tiles (per-tile 4N fraction 0.04–0.75); 4N fraction by
segmentation route 0.860 boundary stain / 0.286 interior / 0.168 nucleus
expansion; per-type median dapi_mean spans 2.7×. In a pixel crop dapi_mean
rises with footprint, so section truncation dims and shrinks a nucleus
together and the integral amplifies the artefact.

*The gate.* Global 2-component GMM on the truncated integral: µ2N 6.7e5, µ4N
1.9e6, ratio 2.86 (log-space refit 3.89 — no second mode; the fit splits a
unimodal heavy-tailed body). The adaptive rule shrinks k to 0.34 and calls
2N 44.3 % / S 18.8 % / 4N 35.0 %. Highest-4N types: fallopian-tube epithelium
0.862 and ciliated epithelium 0.792, both MKI67 < 5 %. The G0/G1 area split is
a type split (within 2N, cell area vs G2M score r = −0.035).

*Prototyped corrections against the pre-registered bar (AUROC of the G2M score
for 4N vs 2N ≥ 0.70 in the four tumour types; MKI67 ratio ≥ 2):* script as-is
0.46/0.43/0.49/0.47 and 1.28/1.22/2.30/1.38; drop multinucleate ≈ unchanged;
per-type GMM best overall (ratio median 2.28, proliferative tumour AUROC 0.574,
MKI67 1.54); tile normalisation ≈ unchanged; area-regressed density best AUROC
(0.621) but inverts MKI67 (0.82–0.92), i.e. measures brightness, not content.
Median AUROC over 18 types 0.449–0.487 for every variant. Nothing clears the bar.

**Verdict: no DNA-content label from this slide; the blocker is physics, not
code** — through a single projected focus plane of a ~5 µm FFPE section the
sectioned fraction of each nucleus is unobserved and varies by more than the
twofold that is the whole 2N/4N signal. Third confirmation after A4 leg 1 and
A4 v2. Recorded, not tuned. Fix list for the script (bimodality refusal gate
via 1-vs-2 BIC, per-type × tile gating, exclusion of multinucleate / no-nucleus
/ expansion-route cells, background + flat-field correction in
`extract_nuclear_dapi`, within-type G0/G1 split, AUROC + MKI67 in place of
κ/ARI, lazy channel read instead of a 16 GB materialisation) carried in the
JSON. If an independent cycle target is still wanted, the FF slide (split-half
0.508/0.696) is the only candidate worth the six validation gates.

### DNA-content cell-cycle label on the fresh-frozen slide (motivation, 2026-09-17)

**Why.** On ovarian FFPE the DAPI integral is nuclear area with a 2.65× tile
drift and no second mode (entry above). The fresh-frozen slide differs in the
ways that matter: 8× depth with reliable marker scores (split-half S 0.51 /
G2M 0.70), larger cells (median area 88 vs 57 µm²), thicker high-quality
transcript layer (6.3 vs 4.3 µm), 1.4–2.7× brighter channels. Whether its DAPI
carries a 2N/4N signal is untested. `qc/nuclear_dapi.parquet` exists.

**How, six gates in order, each stop-if-fails, all pre-registered.**
1. Flat-field / tile correction of the nuclear DAPI mean (smooth per-tile
   gain from the nuclear-pixel median); gate: tile-median dapi_mean spread
   < 1.2× (ovarian 2.65×).
2. Background per tile from a pixel crop; gate: background < 3 % of the mean
   nuclear integral (ovarian 12 %).
3. Exclusions: cells without a nucleus polygon, multinucleate cells
   (nucleus_count ≠ 1), the nucleus-expansion segmentation route. Report
   counts.
4. Truncation: after steps 1–3, log dapi_sum on log nuclear area; gate: R² of
   the residual DNA-content estimate on area < 0.1 (ovarian 0.73). If the
   integral is still area, stop and say so; an explicit z-truncation model is
   the only route past this and is not attempted here.
5. Bimodality, per type on the log integral: 1-vs-2-component BIC margin and a
   dip test; gate: 2 components win clearly AND the component ratio lands in
   [1.8, 2.2] without tuning.
6. The bar: AUROC of the G2M marker score for the 4N vs 2N gate ≥ 0.70 within
   the top-4 MKI67 clusters, MKI67-positive ratio 4N/2N ≥ 2; correlation of the
   DAPI state with total transcripts reported beside it (ovarian 0.5–0.7 = the
   size confound; a cycle label should sit far below that).

**What is wished for.** A label that passes all six on FF becomes the
independent cycle target for the z-carries-cycle claim on that slide; any
failed gate is the finding and the label is not shipped. CPU only; the script
`scripts/test_cell_cycle.py` is not modified — a separate analysis module.

### Is the "joint" DAPI × Scanpy cycle label defensible? Depth-matched check (2026-09-17)

`docs/cell_cycle_comparison_report.md` recommends a joint consensus label
(4N gate AND Scanpy G2M) on the strength of higher marker positivity in the
joint class (MKI67 16.5 % vs 8.6 % in "2N but Scanpy G2M"). Checked on the
ovarian slide, mononucleate cells, the script's own global adaptive gate
reproduced (`scratchpad/cellcycle/obs_scored.parquet`): within Scanpy-G2M
cells the raw MKI67-positive rate is 0.244 (4N) vs 0.086 (2N) — but the 4N
gate is 49 % top-depth-decile cells and the 2N gate 7 %. **Within depth
deciles the two rates are equal** (decile 5: 0.072 vs 0.082; decile 8: 0.218
vs 0.195; decile 9: 0.353 vs 0.296); depth-matched MKI67 ratio 4N/2N = **1.13**.
Among Scanpy-G1 cells ("mitotic dropout" class) the mean G2M score is
identical for 4N and 2N in every decile (−0.017 to −0.023). Within the four
largest types the depth-partialled correlation of the 4N gate with the G2M
score is −0.014 to +0.093. The joint label's marker enrichment is the depth
axis of the DAPI gate (dapi_sum ≈ area ≈ counts), not DNA content; the
"leakage candidate" class (2N ∧ G2M) is small, shallow cells and the report's
niche-border enrichment claim is unsupported (A4 v2's gene-split fingerprint
found no transfer signature). Not adopted; the FF six-gate attempt (entry
above) is the remaining route.

### DNA-content cell-cycle label on the fresh-frozen slide (results, 2026-09-17)

*Delegated (Opus), CPU only; new module `discell/experiments/dapi_cycle.py`,
test `tests/test_experiments_dapi_cycle.py` (10 planted tests); artefacts
`experiments/dapi_cycle.{json,png}` on both slides (same instrument rerun on
ovarian for an exact comparison). The morphology image is read lazily in
512 px windows. `diptest` 0.11.0 was added to the venv and is not yet in the
dependency file.* The six pre-registered gates ran in order on
`xenium_prime_human_ovary_ff`. **Every thresholded gate fails, the first at
gate 1. No DNA-content label from the fresh-frozen slide.**

| gate | threshold | FF | ovarian (same instrument) |
|---|---|---|---|
| 1 flat-field: residual tile-median spread (p95/p5) after a degree-2 gain | < 1.2× | **1.55×** (raw 1.86×, 796 tiles) | 1.66× (raw 1.92×) |
| 2 background share of the nuclear integral | < 3 % | **5.0 %** (33.9–307 DN across tiles) | 3.0 % |
| 3 exclusions (no nucleus / multinucleate / expansion route) | report | 46.6k / 79.4k / 29.0k; 90.6 % kept | 5.7k / 17.5k / 8.2k; 93.7 % |
| 4 R² of the corrected integral on nuclear area | < 0.1 | **0.853**, slope 1.18 | 0.761, slope 1.09 |
| 5 bimodality per type: BIC margin > 10 AND dip p < 0.05 AND ratio ∈ [1.8, 2.2] | majority | **0 / 38** (BIC alone 36/38; dip 0/38; ratio median 2.94) | 0 / 24 (ratio 3.35) |
| 6 G2M AUROC 4N vs 2N in the top-4 MKI67 clusters / MKI67 ratio | ≥ 0.70 / ≥ 2 | **0.543** (0.50–0.64) / **1.59** | 0.547 / 1.55 |
| size confound: Spearman(integral, total counts) / (4N state, counts) | far below 0.5–0.7 | 0.734 / 0.573 | 0.653 / — |

**Reading.** The FF slide is better exactly where the motivation predicted —
marker split-half S 0.520 / G2M 0.692 against ovarian 0.216 / 0.533 — and
that makes the verdict stronger: measured against a more reliable G2M score
the DNA gate still separates at 0.54, and the corrected integral tracks
nuclear area harder than on FFPE (R² 0.85). The dip test does the job the
ovarian entry asked for: BIC alone would have licensed a 2-component split in
36 of 38 types that the dip says is one heavy-tailed body. Fourth confirmation
after A4 leg 1, A4 v2 and the ovarian gating analysis; the pipeline fixes were
all implemented and none moved the answer — the blocker is the projection
through one focus plane, not the pipeline. **The independent DNA-content cycle
target does not exist on either slide; the z-carries-cycle claim is argued on
the marker scores, with their reliability stated per slide.** Todo 3.5 closed.
Caveats: gate 6 scored on a 200k-cell subsample (smallest top-4 cluster 391
cells); gate-2 background = 10th pixel percentile of 512 px crops at 120 tile
centres, a different estimator from the ovarian entry's 12 %.

### Unattended queue: outcome (read 2026-09-21)

The queue of 2026-09-17 ran 24 h (16:01 on the 17th to 16:56 on the 18th),
about half the estimate, and finished. Done: the α_w = 0.05 seed batteries;
the FF reference battery (§§3–4 validate and report; **atlas failed** — the
landmark inventory is empty on `Cluster-N` labels and DBSCAN is fed zero
points, todo 1.8; **transport was killed, exit 137**, presumably OOM on the
1.16M-cell slide — both logged in `data/queue_logs/`); the degeneracy
diagnostics on the 18 sweep3 κ runs (todo 1.2 done); ovarian α_w (21 fits),
lung κ / d_w / α_w (51), FF κ / d_w / α_w (51, 500/40) with aggregate reports.
**Failed: every GSE core leg**, at the first fit, on an IndexError in the new
`type_degeneracy` probe — a type present only in held-out rows (Plasma cells,
n = 1 on the core) made `probe.classes_[argmax]` overflow. Fixed 2026-09-21
(the argmax over the k-wide log-probability is already a type index;
regression test added). The cross-slide leg therefore had nothing to evaluate.
A second instrument defect found while reading: the sweep report's
`cycle_r2_z` read `r2_mean_types`, the statistic issues M6 retired, so the
aggregate JSONs show cycle_z ≈ 0 where the pooled read is 0.42–0.49; fixed to
`r2_pooled` (mean-of-types kept as an extra column). Follow-up queue
`scripts/queue_2026-09-21_gse.sh` launched detached: GSE legs, report
regeneration on all datasets, GSE cross-slide.

### α_w = 0.05 under type_only, three seeds (todo 2.2, results, 2026-09-21)

`alphaw0.05_type_only_s{0,1,2}`, 500/40, full battery. Against the
pre-registered bars (reference triple at 0.1 in brackets):

| read | s0 | s1 | s2 | bar |
|---|---|---|---|---|
| recon | −7.2574 | −7.1810 | −7.2278 | within ±0.06 of (−7.2527 / −7.1924 / −7.2310) ✓ |
| NMI | **0.627** | **0.611** | 0.664 | ≥ 0.63 — **fails on 2 of 3** (ref 0.667 / 0.654 / 0.666) |
| cycle_z pooled | 0.460 | 0.496 | 0.438 | ≥ 0.44 — s2 marginally below (ref 0.440 / 0.499 / 0.465) |
| cycle_w | 0.008 | 0.010 | 0.002 | ≈ 0 ✓ |
| mirror R² | 0.039 | 0.040 | 0.042 | ≤ 0.05 ✓ (ref 0.044–0.046) |
| probe ΔCE | −0.007 | −0.007 | −0.015 | at floor ✓ |
| KL_w max/dim | 0.012 | 0.011 | 0.006 | above the 0.1 floor (0.001–0.009) ✓ |
| cov(μ_w) effective rank / axis-2 share | 4 / 0.39 | 3 / 0.36 | 3 / 0.25 | share ≥ 15 % ✓ (at 0.1: 0.02–0.17) |
| niche AUC w (z) | 0.815 (0.643) | 0.802 (0.639) | 0.781 (0.640) | not below 0.1's 0.768 (0.654) ✓ |
| Moran mean |I| w (z) | 0.554 (0.090) | 0.660 (0.090) | 0.567 (0.073) | ref 0.628 (0.061): mixed |

**Verdict: rejected per the pre-registered rule; α_w stays 0.1.** The
w side does what 2.3 predicted — the response channel opens (KL_w up 5–10×,
a robust second and third context axis, niche AUC of w up by 0.01–0.05) — but
the guard the rule names as a veto moves: NMI drops 0.04 on two seeds
(0.627, 0.611 against a 0.63 floor) and cycle_z slips below the bar on one.
The cross-seed axis-2 cosine was not computed (the atlas stores no loadings);
it is moot for the decision. Consistent with the type_z-era study: the
context channel's gain is paid for on the z side. Recorded, not tuned; the
ovarian α_w sweep below shows the same NMI slope over the whole grid.

### Three-dataset sweeps, first read (2026-09-21; pooled cycle R², 3 seeds per value, 200/20 except FF at 500/40)

Per-value means from `runs/sweep3_*/metrics.json` (the aggregate JSONs are
being regenerated with the pooled key). Guards: probe ΔCE at floor and
cycle_w ≤ 0.02 at every grid point on every slide; mirror R² 0.024–0.057
everywhere; I(z;t)/H(t) 0.76–0.85, within-type variance fraction 0.59–0.72,
type-mean-z recon gap 0.10–0.15 (ovarian, lung) and 0.035–0.059 (FF) — z is
not degenerate at any point of any grid.

**α_w** — *ovarian*: NMI rises monotonically with α_w, 0.604 (0.02) → 0.657
(0.1) → 0.676 (0.3); cycle_z 0.42–0.49 flat within seed spread; KL_w max
0.050 (0.02) → 0.007 (0.1) → 0.0002 (0.3); recon flat (−7.250 to −7.258).
*Lung*: NMI flat 0.659–0.667; cycle_z 0.37–0.44, noisy, no trend; KL_w
0.016 → 0.0001. *FF*: NMI 0.534 (0.02) → 0.607 (0.1) → 0.619 (0.2), the
same slope as ovarian and steeper; cycle_z flat 0.765–0.779. Read: the
NMI cost of a low α_w is real on the two ovarian slides and absent on lung;
above 0.1 nothing improves except NMI by ~0.01–0.02 while the w channel
closes entirely (KL_w → 0). The 0.1 operating point sits where the channel is
just closed and NMI has plateaued; the α_w = 0.05 rejection above is one
point on this curve.

**κ** — *ovarian*: recon plateau to 0.1 then monotone fall to −7.285 at 0.4
(sweep3 as before); NMI 0.64–0.67, cycle_z 0.42–0.46, mirror 0.050 → 0.037
falling with κ, KL_w max falls 0.008 → 0.001. *Lung*: recon −7.254 (0) →
−7.300 (0.4), the same shape with a larger fall; NMI flat 0.660–0.665;
cycle_z 0.38–0.43 no trend; mirror 0.031 → 0.024; type-mean recon gap falls
0.149 → 0.105 with κ (the leak channel absorbs part of what per-cell z
carried). *FF*: recon −7.315 (0.05–0.1) → −7.326 (0.4); NMI 0.59 (0) → 0.61
(≥ 0.05) flat after; cycle_z 0.77–0.78 then 0.749 at 0.4; recon gap 0.059 →
0.035. The envelope shape replicates on three slides: likelihood plateau then
decline past 0.1–0.2, disentanglement reads flat, mirror falling with κ.

**d_w** (lung, FF; ovarian on disk from 2026-09-15) — every read flat within
seed spread across {2, 3, 6, 8} on both slides (lung recon −7.256 to −7.260,
cycle_z 0.38–0.42; FF recon −7.315 to −7.317, cycle_z 0.76–0.78); KL_w spikes
at single points (lung d_w 3: 0.044) are the known checkpoint chase. d_w = 6
stands on three slides.

Not yet read: per-value B / shift-space stability and the per-type rows in
the regenerated reports; GSE core (running); cross-slide.

### Cycle target: continuous Scanpy scores, depth-neutral (todo 3.6, motivation, 2026-09-21)

**Decision (author, 2026-09-21).** After the DAPI analyses (three entries
above: overlap is not the mechanism, the integral is nuclear area, the FF
slide fails all six gates, and the joint label's marker enrichment is a depth
selection) the cycle target stays the Scanpy S / G2M marker score, used
**continuously**, never as a hard phase call. Redone correlation maps on the
scores (`figures/cell_cycle_score_correlation_per_celltype.{png,csv}`,
`scripts/cell_cycle/generate_score_correlation_maps.py`): integrated DAPI
correlates 0.3–0.7 with total counts in every type and 0.00–0.05 with
MKI67 / TOP2A once depth is partialled; the G2M score correlates 0.2–0.5 with
the same genes and is unchanged by partialling. The label-based map
(`cell_cycle_correlation_map_per_celltype.png`) and the comparison report's
joint-label recommendation are superseded.

**Why the target still needs work.** The G2M score is mildly anti-correlated
with total counts within type (−0.12 tumour, −0.28 smooth muscle, −0.36
stromal fibroblasts, −0.30 endothelium): an artefact of `normalize_total` +
`log1p` + `score_genes`' control-set draw, not biology. A depth-tilted target
rewards a latent that carries depth. The split-half reliability of the scores
is computed but not shown beside the R² it caps.

**How.** In `discell/model/cell_cycle.py::score_cell_cycle`: score on
library-normalised log counts against control genes matched on
expression bin (the Tirosh/Seurat construction, with enough bins and controls
that the control mean tracks depth the way the gene set does), or, if that is
insufficient, regress log depth out of the score within type. Evaluated by:
(1) within-type Spearman(score, log total counts) within ±0.05 on every type
with ≥ 2,000 cells on ovarian and lung; (2) split-half reliability not worse
than before (ovarian S 0.22 / G2M 0.52; FF 0.51 / 0.70); (3) the cycle read
on the pinned reference `ablation_gat_type_only_s1` and the lung reference
before/after: cycle_z, cycle_w, permuted floor, ℓ-baseline, linear reference
— pre-registered expectation: cycle_z changes by less than the seed spread
(±0.03), the ℓ-baseline moves toward 0 if it was not there, cycle_w stays at
the floor. **What is wished for:** a target that cannot be predicted from
depth, with its ceiling printed beside every number that uses it. A change
that moves cycle_z by more than the spread is a finding about the old target,
recorded, not tuned.

### Cycle target: depth-neutral scoring (todo 3.6, results, 2026-09-21)

*Delegated (Opus), CPU only; `discell/model/cell_cycle.py` (`depth_neutral`,
`score_cell_cycle(..., depth_neutral_target=False)` — old scoring stays the
default), new `discell/model/cycle_target.py` before/after CLI, 5 planted
tests; artefacts `experiments/cycle_target_<run>.json` on ovarian and lung;
headline numbers verified against the JSONs.* Against the three
pre-registered criteria: **(1) passed, (2) failed, (3) failed with a clean
attribution that rescues the expectation.**

**Where the tilt comes from.** Not the target sum, not the bin count. `log1p`
of library-normalised counts is concave, so the per-cell mean of any gene set
moves with sampling variance, i.e. depth; the control-gene mean is *more*
tilted than the marker mean (ovarian G2M 0.71 vs 0.43 within type), so the
correction overshoots and the score comes out anti-correlated with depth. The
matched-control construction cannot reach the ±0.05 bar (worst type
0.47–0.69 over `n_bins × ctrl_size`; 1:1 rank-matched controls centre the
median at 0 but leave 0.24–0.35). The pre-registered linear fallback is also
insufficient (Pearson zeroed, Spearman up to 0.41; binned mean subtraction
plateaus at 0.21–0.24) because at Xenium depth the score is a spike
distribution — 59 % of lung Cluster-6 (median 23 counts) share one "no marker
detected" value, itself a depth report. **What works:** a depth-conditional
normal-score transform within type (strata of ~500 cells by log counts, normal
score within stratum, ties broken at random under a fixed seed).

**(1) Depth neutrality — passed.** Worst within-type |Spearman(score, log
counts)|: ovarian 0.356 → 0.016 (G2M), 0.321 → 0.015 (S); lung 0.191 →
0.011, 0.252 → 0.009. Zero violations on 16 ovarian and 28 lung types.

**(2) Reliability not worse — failed, and that is the finding.** Split-half
S / G2M: ovarian 0.217 / 0.524 → 0.082 / 0.160; lung 0.213 / 0.279 → 0.078 /
0.111. Most of the old score's agreement with itself was depth: two marker
half-lists agreed because both read library size. The remainder is the honest
ceiling, and it is low.

**(3) Cycle read before / after** (pooled R²; `rank_control` = the same
transform with one depth stratum, isolating the transform from the depth
conditioning):

| read | ovarian `ablation_gat_type_only_s1` | lung `reference_graphclust` |
|---|---|---|
| cycle_z old / new / rank_control | 0.460 / **0.378** / 0.401 | 0.369 / **0.206** / 0.210 |
| cycle_w | 0.008 / 0.007 | −0.001 / 0.003 |
| permuted floor | −0.001 / −0.000 | −0.002 / −0.003 |
| ℓ-baseline | 0.001 / −0.001 | 0.000 / −0.001 |
| linear ref (50 PC) | 0.235 / 0.201 | 0.166 / 0.113 |

cycle_z moves by more than ±0.03, a finding about the old target, recorded not
tuned. The attribution: depth conditioning itself costs 0.022 (ovarian, inside
the spread) and 0.003 (lung); the rest — 0.059 and 0.159 — is the rank
transform plus tie-breaking, i.e. the discreteness of a marker score at
50–300 transcripts per cell. cycle_w at the floor under both; the ℓ-baseline
was already 0 on the four cycling types after per-type centring (the tilt
lived in the shallow non-cycling types the read excludes). The z-over-linear
ratio survives (1.96× → 1.88× ovarian, 2.22× → 1.82× lung).

**Status — author's call pending.** Old target stays the default; the new one
needs labels (pooled, it is worse than raw, 0.60) and one line in
`prepare.py` to reach a fit. If adopted, every cycle_z falls ~0.08 (ovarian) to
~0.16 (lung) with the ceiling printed beside it at 0.08–0.16; no guard or gate
changes sign.

**Decision (author, 2026-09-21).** The plain Scanpy score stays the target,
stated as an imperfect reference and not a gold standard: the claim is the
asymmetry (z predicts it, w does not) against the permuted floor, the depth
baseline (already 0 on the cycling types) and the split-half reliability as
ceiling. The depth-stratified rank target is kept as a robustness column, not
a method change — a transformed target would look like a workaround for an
issue the controls already show is absent. Kept from the exercise: the old
reliability ceiling was inflated by depth (0.52 → ~0.16 for G2M on ovarian),
to be quoted as a caveat beside every cycle R². Todo 3.6 closed.

### Atlas rewrite (todo 1.5 + 1.4, motivation, 2026-09-21)

**Why.** The w-programme atlas judges activity on the variance of varimax-
rotated coordinates (issues V10: a rank-2 w rotated onto 6 axes yields six
collinear coordinates that all pass), its stability metric reads the null
columns (V11), it reads raw w with the per-type gauge offset in it (V12), it
fails outright on cluster-labelled slides (the context-driver step feeds an
empty landmark inventory to DBSCAN), and its report section does not say what
a programme is. Every sweep run now on disk (ovarian, lung, FF, GSE core)
would be misread by it. Decided with the author 2026-09-21: annotate
programmes with pathways, never define pathways from programmes or test them
on the data they were fitted to.

**How.** (1) Activity by the effective rank r of cov(μ_w) on within-type
gauge-centred w (components ≥ 1 % of variance), varimax within the r-dim
subspace, each programme's variance share reported; (2) stability by shift-
space overlap of μ_w·B across seeds plus per-axis cross-seed cosines, the
matched-column correlation dropped; (3) all w reads gauge-centred per type
at the reference context (1.4), also in the report's per-type section, the
per-type ‖w‖ ranking withdrawn; (4) enrichment by a rank-based statistic on
the full loading vector against the hallmark sets on the expressed-panel
background, sets with < 5 panel genes untestable, BH per programme, label at
q ≤ 0.05 as now; (5) context drivers skip the landmark block when the
inventory is empty, so graphclust slides run; (6) three concordance reads:
label recurrence across seeds (same hallmark label in ≥ 2 of 3 seeds),
across slides, and κ-survival sweep-internal as already produced; (7) the
report section rewritten How / Evaluated by / What is wished for.

**Evaluated by.** Planted tests: rank-2 w in 6 dims → r = 2, two programmes;
a planted per-type offset does not change activity or labels; planted
sparse loadings recovered; a planted enriched set is labelled and a random
one is not; an empty landmark inventory does not crash. On real runs: the
pinned reference reads r = 2 with EMT recurring; the three seeds agree on
the label set; lung and FF run to completion.

**What is wished for.** Few, territorial, context-explained programmes
whose labels replicate across seeds and slides; a programme without a
replicating label is reported as unlabelled, not named.

### GSE315411 core: the three grids and the held-out section (results, 2026-09-21)

Follow-up queue `scripts/queue_2026-09-21_gse.sh`: 51 fits on the solo
`pdl018d` core (200/20, α_z 0.0036, tile 2048), every checkpoint evaluated
on the dual section (`runs/sweep3_*/crossslide/`), reports regenerated.
Pooled cycle R², 3 seeds per value.

**Same section.** α_w: recon flat −7.211 to −7.216; NMI 0.594 (0.02) → 0.618
(0.1) → 0.630 (0.2), the ovarian slope again; cycle_z 0.49–0.51 flat, cycle_w
≤ 0.016; KL_w max 0.017 → 0.0000 as α_w rises. κ: recon −7.210 (0) →
−7.215 (0.1) → −7.256 (0.4), plateau then fall; NMI 0.60–0.62, cycle_z
0.49–0.51, mirror 0.029 → 0.022, type-mean recon gap 0.162 → 0.109 with κ
(the fourth slide on which the leak channel absorbs part of what per-cell z
carried). d_w: everything flat across {2, 3, 6, 8}. Probe at floor and
I(z;t)/H(t) 0.73–0.77 everywhere. **The envelope shape now replicates on four
slides.**

**Held-out section (train section 11, evaluate section 10, same 35-class
vocabulary), the first genuinely out-of-sample read of the sweeps.** At every
grid point of every grid: reconstruction over all 64 dual tiles is 0.015–0.017
nats/count worse than the same-section best (a quarter of the ovarian seed
envelope); NMI −0.02 to −0.03; **cycle_z on the held-out section 0.505–0.527,
slightly above the same-section 0.49–0.51**, cycle_w ≤ 0.009; probe ΔCE at
its floor (−0.005 vs −0.009); mirror 0.033–0.043; I(z;t)/H(t) 0.76–0.79;
type-mean recon gap 0.108–0.157, falling with κ as on the training section.
No grid point behaves differently on the held-out section than on its own:
the section-to-section generalisation cost is a constant 0.015 nats/count
and 0.02–0.03 NMI, independent of κ, d_w and α_w. This is the strongest
stability statement the programme has: the disentanglement reads survive a
change of section at every point of the three grids.

### Atlas rewrite (todo 1.5 + 1.4, results, 2026-09-21)

*Delegated (Opus), one GPU; `discell/model/atlas.py`, the §6 and per-type-‖w‖
sections of `discell/model/report.py`, one empty-input guard in
`validate.py::landmark_inventory`, 12 planted tests (full suite 210 passed);
artefacts `runs/<run>/atlas/` on 8 runs over 3 slides; the pinned reference's
report regenerated; ranks, shares and recurrence spot-checked.* All seven
pre-registered points delivered; all planted and real-run checks pass.

**Points 1–2** (effective rank; shift-space stability) stood from the earlier
round and are now certified by plantings (rank-2 in 6 dims reads r = 2 where
the old read said 6/6; a permuted, sign-flipped basis reads cosine 1.0, an
orthogonal complement 0). **Point 3:** every read on gauge-centred w; the
report's per-type ‖w‖ ranking is withdrawn in place with the V12 numbers and
its offset-free replacement (within-type variance of a programme coordinate).
Deviation: the gauge is the within-type mean of μ_w, not m_ψ at the mean
context; any per-type constant is a valid gauge and no reported quantity
depends on the choice. **Point 4:** the top-50 hypergeometric is replaced by a
Mann–Whitney rank test on the full loading vector against the expressed-panel
background, BH per programme, q ≤ 0.05, rank AUC beside q, sign-blind. **Point
5:** the cluster-label crash had two sites (DBSCAN on an empty class inside
`landmark_inventory`; `np.stack([])` in the atlas); both guarded, the landmark
driver block dropped and recorded as a missing question. Lung and FF run.
**Point 6:** label-set recurrence across seeds (`--compare-runs`) and across
slides (`--compare-atlas`); κ-survival stays sweep-internal. **Point 7:** the
section is rewritten How / Evaluated by / What is wished for; a non-recurring
label ships as *unlabelled*.

**Ovarian seed triple (α_w = 0.1).** r = 2 of 6 on all three seeds; spectra
[0.95, 0.05], [0.62, 0.38], [0.94, 0.06]. Axis-1 cross-seed |cos| 0.93 / 0.93
/ 0.98, axis-2 0.64 / 0.83 / 0.40; shift-space overlap 0.67–0.93. Labels
recurring in ≥ 2 of 3: EMT and HYPOXIA (3/3), E2F_TARGETS and G2M_CHECKPOINT
(2/3). Dominant programme = the macrophage/stromal axis (F13A1, MRC1, TNXB,
KLF4) on every seed, Moran I 0.37–0.47, joint driver R² 0.92 with Φ the
largest partial (0.25–0.31) and landmarks ≈ 0; second = the matrix axis
(COMP, SFRP4, COL10A1, COL11A1), Moran 0.61–0.69. Pre-registration met. The
second axis does not reach |cos| ≳ 0.8 — it is the programme carrying 5 % of
w's variance on two seeds and 39 % on one — reported, not tuned.

**Cluster-labelled slides.** Lung r = 1 (spectrum [0.999, 0.001]), Moran
0.42, joint 0.87, EMT (q < 1e-4, AUC 0.68) on CCL19/ADAMDEC1/PLVAP/MMP9/CXCL9.
FF r = 3 ([0.55, 0.27, 0.18]), labels EMT, MYC_TARGETS_V1, EMT/HYPOXIA; its
second programme carries 27 % of w's variance but 71 % of the realised shift
(the two shares are reported separately for this reason). **Cross-slide
recurrence** (ovarian s1, lung, FF): EMT 3/3, HYPOXIA 2/3, KRAS_SIGNALING_UP
2/3.

**α_w = 0.05 triple — the cosines the 2.2 verdict lacked.** r = 4 / 3 / 3
against 2 at 0.1, and the extra axes reproduce: a third programme with the
same signature in all three fits (FOXL2, SFRP4, POSTN, GRIA2, WNT4, GREB1),
Moran I 0.84–0.88 (the most territorial programme seen), most modulated in
stromal fibroblasts and smooth muscle, Φ-driven (partial 0.50–0.57 vs
composition 0.02–0.09), and **unlabelled** (no hallmark at q ≤ 0.05 on 2 of 3
seeds). Axis-1 |cos| 0.88–0.97; the least stable pairing (0.29) is the matrix
axis. Answers "do the extra axes reproduce" in the affirmative; does not
reopen the α_w decision (its veto was NMI on the z side).

**κ-survival** unchanged and sweep-internal (0.29–0.43, flat over κ); not
regenerated under the new basis — it is now the one w-stability read not in
shift space, and the weakest of the three concordance reads.

### Three packages in parallel (motivation, 2026-09-21): transport clarity, x̃ decision, baseline survey

**Transport (todo 2.5) — priority.** The author's framing: transport is the
one experiment that exercises the whole system — z held fixed, the response
channel through m_ψ and B, the leak channel through β and κ — where every
other read isolates a subsystem; it must be right and clear. Current state:
counterfactual = Δ̂program + Δ̂leak with z fixed; k-means-on-composition
niches (K = 10) leave the supported tier near-empty (3/158 panels); the
extrapolation tier reads R² 0.099, slope 0.92, beats-both 100/155 on the
pinned reference; the model account minus the counterfactual is the
selection share (≈ 0.05); the κ-sensitivity companion was run on the
pre-correction object; the neighbour-dose experiment found Φ carries half or
more of w's context dependence, so a composition counterfactual that carries
each niche's real Φ mixes intervention with description; w reads must be
gauge-centred (V12; transport works on differences and is unaffected, to be
confirmed in code). **How:** (1) a plain-language statement of what is
predicted, from what, evaluated how, and what a pass is; (2) κ-sensitivity
rerun on the corrected counterfactual over the six sweep3 κ seeds; (3) a
Φ-held-fixed row (Φ at the receiver type's mean in both niches) beside the
composition-plus-real-Φ row; (4) annotation-defined niches (tumour rim /
core / stroma from the kNN-smoothed tumour fraction) so the supported tier
is populated; (5) per-panel calibration figures and a trust criterion per
panel; (6) the three-seed and cross-slide reads (lung, FF, GSE core) so the
whole-system claim carries envelopes. **Evaluated by:** on the pinned
reference the counterfactual must still beat both single channels in a
majority of panels with slope in [0.8, 1.2]; the Φ-fixed row states the
interventionable share; the κ trajectory of program / leak / total on the
corrected object is a κ-range, never a point; the supported tier under
annotation niches has ≥ 20 panels. **Wished for:** one figure and one table a
reader can follow without the code.

**x̃ (todo 2.1).** Powered planted world (fix V9: within-type thresholds,
excess FPR = victim − control, power gate raw AUROC ≥ 0.9 and raw excess ≥
0.1) comparing raw counts, z-probe without x̃, z-probe with x̃, and
leak-subtracted counts, three seeds; pass/fail per seed pre-registered in the
package; verdict use / option-only / drop.

**Baselines (todo 4.1).** Survey SIMVI, resolVI, MintFlow (+ scVIVA,
NicheCompass): inputs, outputs, scale, install, which of our metrics apply on
which dataset; comparison matrix; install plan; GPU-hour estimates; no
installs, no runs.

### Baselines survey (todo 4.1, findings, 2026-09-21)

*Delegated (Opus), no GPU, no installs; five `uv pip install --dry-run`
resolutions against the live env and web sources (URLs in the scratchpad
notes `scratchpad/baselines/dryruns.txt`).*

**Bracketing confirmed, and sharper than expected.** SIMVI (Dong & Kluger,
Nat Commun 2025; `simvi` 0.1.2) = our split without a leak channel:
intrinsic z + spatial-induced s, asymmetric regulariser, annotation-free, kNN
k = 10; largest published dataset ≈ 33k cells. resolVI (Ergen et al., bioRxiv
2025; inside scvi-tools) = our leak channel without a split: one latent, a
true / diffusion / background mixture with **per-cell mixture proportions**;
1.4M cells in < 6 h on a 3090. MintFlow (Lotfollahi lab, bioRxiv
2025.06.24.661094; `mintflow` 0.3.0) sits between: three latents (intrinsic,
incoming, outgoing), separate intrinsic and microenvironment-induced count
vectors, in-silico microenvironment perturbation — a counterfactual analogue
for the *program half* of our transport — but no contamination model and
**labels required**; published on Xenium 5K up to 337k cells. scVIVA and
NicheCompass are w-side-only (a niche-informed single latent; a niche
descriptor). So the z/w asymmetry rows are scorable in full on SIMVI and
MintFlow only, the contamination row on resolVI only, and no published method
occupies both axes — which is the paper's claim.

**Install.** `scvi-tools` 1.5.1 and `scviva-tools` 0.1.7 resolve into the
existing env with zero downgrades → resolVI and scVIVA need only an optional
extra. MintFlow would downgrade zarr 3.3 → 2.18 (breaks the tifffile image
path), anndata and pandas → own venv, mandatory; needs a wandb offline
decision. SIMVI pins `scvi-tools ≤ 0.16.2` (pytorch-lightning 1.5.8,
flax/jax) → own venv on python 3.10 with a tutorial-reproduction gate before
any DisCell slide. NicheCompass drags mlflow and a web stack → isolate.

**Matrix (method × metric).** Held-out recon: partial everywhere (different
likelihoods; needs one common unit or drops to an appendix row). Cycle
asymmetry, niche/Moran per latent, pseudotime, invariance probe: full on
SIMVI and MintFlow; one-sided on resolVI and scVIVA (a single latent — report
descriptively, never as a loss); NicheCompass n/a on the z side. Transport
analogue: MintFlow (program half only), SIMVI partial via per-gene spatial
effects. Per-cell contamination vs the κ grid and the transcript-flux median
0.13: **resolVI only**, as a distribution (it publishes no headline fraction).
Scale: resolVI everywhere; SIMVI is 12–35× above its published maximum on the
full slides — the GSE 69k core is its safe scale; MintFlow up to ovarian.

**Fairness, pre-registered.** Same tile split and fold map; same labels per
slide (table split by label-taking vs label-free); our pruned Delaunay graph,
and both graphs where a method insists on its own; d_z 20 / d_w 6 matched;
matched wall-clock; three seeds; every number through `validate.py` against
floor / ℓ-baseline / linear reference; empty cells read "n/a by construction";
baseline spatial latents gauge-centred (V12) and compared in shift space (V11).

**Order and cost (extrapolated, ± factor 2).** resolVI first (~12 GPU-h: GSE
core → ovarian → lung + FF) because it owns the row nothing else fills; SIMVI
second (~10–18 GPU-h) behind the tutorial gate, core before full slides;
MintFlow third (~20–26 GPU-h) for the counterfactual row; scVIVA and
NicheCompass last (~7–9 GPU-h). Total ≈ 50–65 GPU-h; stages 1–2 (~25 GPU-h)
deliver the bracketing claim.

**Caveats.** `08-validation-analyses_1.md` has no §8 in this revision — the
bracketing framing was reconstructed from the handover and the register. No
runtime figure is published for four of the five methods. MintFlow's own
baseline list (Supplementary Note 1) could not be retrieved. Dry-runs prove
resolvability, not importability.

### x̃ decision by powered planted world (todo 2.1, results, 2026-09-21)

*Delegated (Opus), GPU 1; `discell/applications/xtilde_gate.py` (paused patch
applied plus the ≥ 0.05 excess margin), the A4 planted leg retired into the
gate, 22 tests; artefacts `data/experiments_synthetic/xtilde_gate.{json,png}`;
six 400-epoch fits.*

**The world is powered — V9's defect was the plant and the threshold, not z.**
6,000 cells, 8 types, planted κ = 0.2; a programme taking a 25 % transcript
share of 12 mid-expressed genes in 30 % of the cells of two cycling types,
re-leaked through the true operator and resampled. Victims = non-cycling cells
in the top exposure tier (329–461 per seed); controls = non-cycling cells with
no planted neighbour. Thresholds within type at each type's control 0.7
quantile; statistic = excess FPR (victim − control), paired 500-draw stratified
bootstrap. Power gate (raw AUROC ≥ 0.9, raw excess ≥ 0.1) passes on all three
seeds at the first plant: AUROC 0.9997–1.0000, raw excess 0.265 / 0.370 /
0.379, control FPR 0.298–0.301 on every type (V9: 0.67, 0.11/0.00/0.02,
0.02–0.45).

**Four callers, three seeds (excess FPR; AUROC 0.999–1.000 throughout).** Raw
0.265 / 0.370 / 0.379 → z-probe without x̃ **0.106 / 0.177 / 0.118** → z-probe
with x̃ **0.088 / 0.140 / 0.024**; leak-subtracted counts with the model's ρ̄
0.202 / 0.300 / 0.286, with the true ρ̄ 0.206 / 0.297 / 0.281. Pre-registered
pass (excess lower by ≥ 0.05, paired CI > 0, AUROC within 0.02): z beats raw
3/3; x̃-z beats raw 3/3; counts-correction beats raw 3/3; **x̃-z beats plain z
1/3** (Δ +0.019 / +0.037 / +0.094; right sign every seed, two CIs cover 0).

**The amortisation-gap hypothesis is refuted as stated.** The per-cell z
carries only 31–48 % of raw's excess (mean 0.40, below the pre-registered 0.5
line on every seed): the amortised posterior mean does not simply inherit
what leaked into x_i; the penalty and the population-level z law remove most
of it. **The counts-level route sits on its own ceiling**: the model's ρ̄ and
the true ρ̄ give the same excess to three decimals, so subtracting a mean from
a multinomial draw removes about a quarter of the excess and no more.

**Verdict, as pre-registered: option-only.** x̃ helps in the built-for
direction on every seed but clears the bar on one of three, and the benefit
is the size of its known costs on the slide (NMI −0.017…−0.05, training
cycle_z −0.03, Moran-w down in 2/3). `subtract_leak` stays default off,
available for per-cell applications; spec-07 §7.13 stays parked. Agrees with
the near-neutral slide result of 2026-09-14 in a world where the truth is
known. **Caveat:** AUROC saturates at 1.0 on every arm, so the sensitivity half
of the rule never bound, and the world has no genuinely-cycling victim — the
doc-11 sensitivity-loss fail state is untestable by this design and remains
the live risk for per-cell z claims (A1/A2/A5).

### x̃ decisive follow-up: six seeds and an unsaturated plant (motivation, 2026-09-21)

**Why.** The author's position, recorded: x̃ is a clean implementation and a
loss on reconstruction is acceptable if the per-cell z becomes more likely to
be correct; if it improves trust in z it should be the default, and the κ
sweep is then rerun under it. The powered gate (entry above) left the
direction consistent (3/3) but the size unproven (margin cleared 1/3) and
tested specificity only (AUROC saturated at 1.0).

**How.** The same gate (`xtilde_gate.py`), (a) at **six seeds** with the
25 % plant, (b) at a **weaker plant** tuned so raw AUROC within the cycling
types sits near 0.85–0.92 (share lowered until the power gate is just
passed), six seeds, so the sensitivity half of the rule can bind; (c) a world
with **genuinely cycling victims** — a fraction of the non-cycling types'
cells given the plant without leaked neighbours — so the doc-11
sensitivity-loss fail state (does the corrected caller still see a real
cycling cell) is measurable as recall in those cells.

**Evaluated by (pre-registered).** Default-on if, over the 12 seeds of (a)+(b):
x̃-z lowers excess FPR vs plain z by ≥ 0.05 with the paired CI above zero on
≥ 8 of 12 with no reversal; within-cycling AUROC within 0.02 of plain z on
every seed; and in (c) recall on genuinely cycling victims within 0.03 of
plain z. Otherwise option-only stands. If default-on: `TrainConfig.subtract_leak
= True`, the register and spec §7.13 updated, and the κ sweep (6 × 3 seeds)
rerun on ovarian first, then the other three slides, with the type_only
α_w = 0.1 centre — the earlier sweeps become the x̃-off comparison.

### Transport clarity (todo 2.5, results, 2026-09-21)

*Delegated (Opus), one GPU; `discell/model/transport.py`, the §7 section of
`report.py`, 7 planted tests (full suite 239 passed, 1 skipped); artefacts
`runs/<run>/transport/` on 7 runs over 4 slides plus the six-point κ sweep
(`experiments/transport_kappa_sensitivity_v2.json`); table and figure
verified.* All six pre-registered How-steps delivered; all four bars met, one
marginally.

**What is predicted, plainly.** For one cell type and two neighbourhoods A
and B: the per-gene log-rate shift a cell of that type undergoes going from A
to B. Predicted from two channels added: the *program* channel (the context
prior m_ψ at B's mean context minus at A's, through the loadings B) and the
*leak* channel (κ times the difference in mean foreign influx; leakage comes
from the new neighbours, κ never changes inside a prediction). Held fixed:
the cell's intrinsic z. Evaluated on held-out tiles the model never saw,
against the observed depth-normalised mean shift, both sides mean-centred,
scored as R² against the zero-prediction null plus the calibration slope. The
"model account" additionally lets the type's intrinsic mix differ between
niches; its excess over the counterfactual is the *selection share*.

**The bars.** (i) Counterfactual beats both single channels in a majority with
slope in [0.8, 1.2] on the pinned reference: **100/155 (65 %), slope 0.92** —
reproduces the record exactly after a substantial rewrite. (ii) Φ-fixed row
states the interventionable share: **1.04** (per-panel median 0.99, IQR
0.94–1.06; seed/slide envelope 0.86–1.13). (iii) κ trajectory as a range:
leak 0.000 (κ = 0, sanity) → 0.084 (0.3), program 0.056 → 0.042, total
peaking at 0.086 (κ = 0.2), slope falling 0.99 → 0.60 and leaving the band at
κ ≥ 0.3 — **quotable range κ ∈ [0.05, 0.2]**. (iv) Supported tier ≥ 20 panels
under annotation niches: **71** (73 / 75 on the other seeds).

**Annotation niches.** Six ordered bands of the kNN-smoothed tumour fraction
(deep stroma → core; cuts 0.1/0.3/0.5/0.7/0.9 fixed before any result). Being
nested they share composition support, which k-means niches cannot;
handover limitation 9 closes for annotated slides. Supported tier 71 panels,
13 types, R² 0.068, slope 0.79 (0.88 / 0.88 on the other seeds — the one
marginal miss, recorded), beats both 42/71; best panel Tumor Cells rim → core
R² 0.322. On slides whose type names name no tumour the function raises and
composition niches are used and named (lung, FF, GSE core).

**The trust criterion — the main clarity gain.** Every panel now carries a
noise ceiling: the Spearman–Brown split-half reliability of the observed
shift, the largest R² any predictor could reach. Trusted = ceiling ≥ 0.5 on
≥ 100 genes. On ovarian the mean ceiling is **0.123**: 141 of 155
composition panels are essentially unmeasurable and the 0.099 headline is a
mean over mostly noise. On the **14 trusted panels the counterfactual reads
0.205 at slope 1.08, beating both channels in 13/14**, taking 34 % of what is
reachable. The ceiling is identical at all six κ, as it must be. **The
transport R² was never small because the model is weak; it was small because
most panels contain almost nothing measurable.** Never quote 0.099 without
0.205 beside it.

**The Φ question, answered against expectation.** Freezing Φ at the receiver
type's mean bites (the program channel moves by > 0.02 in 47/155 panels) yet
leaves the total counterfactual unchanged on every slide. Neighbour-dose
measured Φ's share of cell-to-cell variation within a type; transport asks
about differences of niche means, and Φ's cell-to-cell part averages out
inside a niche. The pre-registered worry that a composition counterfactual
carrying real Φ mixes intervention with description comes back **negative**:
the transported average effect is essentially interventionable. Shares > 1
in some reads are inside the envelope and read as "≈ 1".

**Envelopes and slides.** Ovarian triple 0.085 / 0.099 / 0.092, slope
0.91–0.97, beats-both a majority in all three; selection share 0.029–0.047
(about a third of an observed niche difference is which cells live there).
Lung 158 panels, 0.082, slope 0.84, 92/158. **FF now runs** — the exit-137
kill was the instrument holding per-cell rate matrices (23 GB on 1.16M
cells); it now accumulates per-(niche, type) means in the forward pass, ~3
min — and is the strongest read in the programme: **248 panels, R² 0.280,
beats both 223/248 (90 %), 140 trusted panels at 0.391**, ceiling 0.51 (the
deepest slide's observation is reliable); slope 1.25, just outside the band.
GSE core: ceiling 0.049, zero trusted panels — does not reach the noise floor;
its 0.062 is not a transport result.

**Deliverables:** `runs/ablation_gat_type_only_s1/transport/
transport_tumour-band_summary.png` (tier bars with the ceiling drawn, and the
per-panel biology-vs-contamination map with both tiers marked — a defect
fixed: the map used to show only extrapolation panels) and
`transport/transport_table.md` (four tiers × five predictors with every bar
underneath). Stale pre-correction `transport.json` files remain on unrelated
runs (`gat_sink_*`, `xtilde_*`, `alphaw0.05_*`, `wd0`) and must not be
compared to the new numbers.

### x̃ decisive follow-up (results, 2026-09-21)

*Delegated (Opus), GPU 1; `xtilde_gate.py` extended with a plant-share search,
a recall read-out on genuinely planted cells and the 12-seed rule; 36 tests;
artefacts `data/experiments_synthetic/xtilde_gate_{a,b,c}.{json,png}` and
`xtilde_gate_followup.{json,png}`; 15 seeds × 2 fits; verdict JSON verified.*

**Verdict: option-only, and now decisively.** Every clause of the
pre-registered rule fails: margin 3 of 12 (needs 8), 3 reversals (needs 0),
AUROC within 0.02 of plain z fails on 2 of 12, recall within 0.03 on 1 of 3.

**(a) Six seeds, 25 % plant, gate 6/6.** The three-seed worlds reproduce
exactly. Δ(z − x̃-z) +0.012 / +0.032 / +0.083 / **−0.028** / +0.093 / +0.060:
"right sign every seed" does not survive three more seeds. Counts-level
ceiling replicates 6/6 (model ρ̄ and true ρ̄ within 0.005).

**(b) Weakened plant — and a coupling the pre-registration could not know.**
Share search on seed 0 → 0.0562, raw AUROC 0.920, raw excess 0.094. Raw excess
reaches 0.1 only at AUROC ≈ 0.93, so "AUROC 0.85–0.92 while the gate passes"
is infeasible; the AUROC band was honoured and the arm run 6 % under the
excess line (full gate 2/6 seeds). **x̃ is worse than plain z on 3 of 6 seeds
with the CI above zero** (−0.050 / −0.081 / −0.057), and the z probe's own
AUROC collapses to 0.61–0.92 against raw's 0.82–0.96 with excess often
negative: z is blunting, not decontaminating. The saturated-plant benefit was
an artefact of a plant too strong to lose anything to.

**(c) Genuinely cycling victims — the doc-11 fail state fires.** 10 % of two
non-cycling types planted at exposure 0. Recall Δ(x̃-z − z) +0.014 / −0.060 /
**−0.221**; the counts-level arms keep raw's sensitivity (0.70–0.97). x̃ in the
encoder costs up to 22 points of recall on real cycling cells.

**Cost if flipped, restated (2026-09-14):** NMI −0.017 consistent, training
cycle_z −0.028, Moran-w −0.061 (down 2/3), planted-gate NMI −0.03…−0.05 at
κ = 0.2, plus a full κ-sweep rerun; benefit at the saturated plant mean
−0.042 excess (3/6 on the margin), at the unsaturated plant **+0.023 against
x̃**. The author's condition ("if it improves trust in z") is not met; in the
one world where the truth is known and the caller is not saturated, x̃
lowers trust in per-cell z. **`subtract_leak` stays default off; spec-07
§7.13 stays parked; no κ-sweep rerun.**

**Caveats.** (b)'s gate halves are not jointly satisfiable at any share (a
pre-registration defect, not a run defect); (c) ran 3 seeds at the weakened
share, not pre-registered; the share is a seed-0 calibration; GPU
nondeterminism moves the z arms by ≤ 0.013, so (a)'s seed-3 reversal is ~2×
noise while (b)'s are well above it. **Separately, and independently of x̃:
the z probe itself loses 0.25–0.38 of raw's recall on genuinely planted
cycling cells at an unsaturated plant.** That bears on every per-cell z
claim (A1/A2/A5) and is the finding to carry forward from this exercise.

### Transport at the distribution level: MMD read, pairwise and leave-one-niche-out (motivation, 2026-09-21)

**Why.** The mean-shift read (entry "Transport clarity") scores the predicted
per-gene difference of niche means. The author's question: transport is a
population moved from one context to another, so compare the transported
population with the population that was there, as distributions. Two
versions, both requested: **pairwise** (A → B, as the mean read pairs
niches) and **leave-one-niche-out** (every cell of type t *not* in A,
transported into A, compared with A's cells — if z is intrinsic, cells from
all contexts should land on A's population, and the pooled source gives more
samples). Also two readability additions to the mean read: fraction of the
noise ceiling as the headline beside raw R², and top-gene overlap (of the 50
genes predicted to rise most, how many are in the observed top 50).

**How.** For type t and target niche A: source cells S (niche B, or all
niches ≠ A), each kept at its own μ_z, given A's context (m_ψ at A's mean
context; Φ-fixed variant at the type mean) and A's leak source (κ · mean
influx of A), decoded to the probability vector p̂. Target cells T = held-out
cells of type t in A, represented by their raw normalised composition
(counts / ℓ). Distance: MMD² with a Gaussian kernel on the square-root
(Hellinger) map of the probability vectors, bandwidth = median pairwise
distance in T; energy distance reported beside it. References, all on the
same cells: **floor** = MMD² between two random halves of T (sampling noise);
**untransported** = MMD² between S decoded at its *own* contexts and T (what
transport must reduce); **type-mean** = MMD² between a single point (mean of
T) replicated and T (a degenerate predictor that ignores within-type spread);
**observed-source** = MMD² between S's raw compositions and T (the raw niche
difference). Score per panel: **gap closed** = (untransported − transported) /
(untransported − floor), clipped to [−1, 1]. Every model quantity from
training tiles; S and T from held-out tiles; sizes matched by subsampling to
min(|S|, |T|, 2,000). Panels: same type × niche pairs as the mean read (both
niche sources: composition k-means and tumour bands where defined).

**Evaluated by (pre-registered).** (1) Transported MMD² below untransported
on a majority of panels with a paired bootstrap CI excluding 0; (2) gap
closed reported with its distribution and its median; the pooled
leave-one-out version should have a tighter CI than the pairwise one on the
same target niche (that is what pooling buys) and a gap closed not below the
pairwise median by more than 0.1 — if pooling *hurts*, z is not context-
free in the way the pooled read assumes, and that is the finding; (3) the
transported cloud should not collapse onto the type mean: transported MMD²
to T must be below the type-mean predictor's on a majority of panels, else
the read is just the mean shift again; (4) agreement with the mean read:
Spearman across panels between gap closed and the mean read's counterfactual
R² ≥ 0.5, so the two instruments describe the same panels as good. Run on
the pinned ovarian reference, the two other seeds, and FF (the reliable
slide). **What is wished for:** a distribution read a reader can follow
without R², reported as "fraction of the niche gap closed", with the
leave-one-out version as the intrinsic-z test.

### Three packages (motivation, 2026-09-21, afternoon): handover refresh + text, ring-2 skip + κ-survival in shift space, article review

**Documentation.** `docs/handover.md` is from 2026-09-14 and predates
type_only being pinned, the atlas rewrite, transport clarity, the four-slide
sweeps with the held-out section, the leak-measurement negatives, the cycle
decision and the x̃ verdict. Rewrite it from the devlog and registers,
keeping its structure (what / operating point / how to run / results by
claim / retractions / limitations / verdict / open questions). Todo 3.3:
the depth qualifier on "z beats the linear reference 2×" in handover and
report text. Paper items A2–A7 of the revision list (model text only, no
numbers).

**Code.** Todo 1.6: under `type_only` the ring-2 encoder pass is unused —
run `posterior_z` on seeds ∪ ring1 only, keep type_z behaviour, prove the
loss and every diagnostic unchanged on a smoke fit, re-measure the halo
overhead at 4,096-cell tiles. κ-survival (`validate --sweep-tag`): move from
matched signature correlation to shift-space overlap + per-axis cosines,
consistent with the atlas; regenerate on sweep3 ovarian. Evaluated by:
identical metrics.json on a fixed-seed smoke fit before/after the skip; the
κ-survival table reproduces the atlas' cross-seed numbers at κ = 0.1.

**Articles.** Read the five baseline papers in `submission_paper/articles`
(SIMVI, resolVI, MintFlow, DisCoVR, Celcome — the last new to the project)
for experiment ideas applicable to DisCell and for how a comparison against
each could be made fair, given that none occupies both of our axes.
Deliverable: a ranked list of experiments (what, why, cost) and a comparison
design per method, with the metric each paper would accept as its own.

### Baseline articles reviewed (2026-09-21)

*Delegated (Opus), report-only: no code, no GPU, no installs. Five PDFs in
`submission_paper/articles/` read in full through `pdftotext -layout` with
page markers (scratch notes `scratchpad/articles/*_pg.txt`), plus
`discell-literature.bib`. Follows the survey entry "Baselines survey (todo
4.1)"; that entry's bracketing claim is confirmed by the primary sources and
sharpened in three places.*

**The bracketing claim survives contact with the papers, and one of them
supplies the argument for us.** Of the five, SIMVI and MintFlow occupy the
intrinsic/spatial axis only, resolVI the contamination axis only, DisCoVR is
the objective template with no spatial or contamination content at all, and
**Celcomen is not on our axes** — it disentangles *gene–gene* interaction
matrices (intra- vs inter-cellular) in an Ising-style energy model, not cell
latents, and has no contamination model. Celcomen stays in related work and
off the baseline list (see below). No published method occupies both axes.

**The confound is stated by no one, and is operationalised as a virtue by
MintFlow.** MintFlow's *only* real-data validation of its intrinsic /
microenvironment split is that "signalling genes" (any gene in any
ligand–receptor database) should receive a higher microenvironment-induced
share of their read counts than other genes (p. 5, Methods p. 39). Signalling
genes are, by construction, the genes expressed in the *neighbour*, so they
are exactly the counts most likely to be misassigned by segmentation. Their
validation criterion cannot distinguish a real microenvironment effect from
transcript transfer, and their only word on segmentation is one sentence of
limitation (p. 28). resolVI, which *does* model the transfer, never asks the
converse question: it runs niche differential expression and differential
colocalisation on corrected counts (liver cancer SPP1⁺ macrophages beside
SPP1⁺ tumour, pp. 10–11; colitis Bmp gradient, p. 13) without testing whether
the residual niche signal is leak, and in Methods (p. 21) disables the
scib PCR metric "as we expect that a majority of the variation is due to wrong
segmentation" — a strong, unverified claim in our favour. SIMVI names a
*different* confound (intrinsic-looking spatial structure from cell-type
colocalisation, p. 1; its positivity index, pp. 3, 5, 8) and says nothing
about transcripts. Celcomen and DisCoVR say nothing. **This is the
related-work paragraph: everyone in the split literature validates on the
neighbour-resembling half of the transcriptome, and nobody sweeps.**

**The one-figure head-to-head exists and is cheap.** MintFlow's per-cell
microenvironment score (Σ_g x^mic / Σ_g x), resolVI's per-cell α₁+α₂ diffusion
+background proportion, and our κ are *the same number with three different
names*: the fraction of a cell's counts not attributed to its own intrinsic
programme. MintFlow calls it signalling, resolVI calls it artefact, we
decline to call it either and sweep. Plotting the three on the same cells of
the same slide, against the transcript-flux median 0.13, is the paper's
central claim in one panel and needs only the two installs already scoped.

**Scale and fairness, corrected from the survey.** SIMVI's published maximum
is now pinned: MERFISH MTG 11,059 cells and STG 14,924 cells, Slide-seqV2,
Slide-tags tonsil, and the CosMx melanoma cohort — all far below our slides;
the GSE 69k core remains its safe scale and its k = 10 kNN graph is its own.
Celcomen's Xenium analysis is a **500 × 500 µm crop** of one slide (Methods,
p. 7), which is why it is not a scale-comparable baseline. resolVI's 1.4M
cells in under six hours on a 3090 stands. MintFlow's benchmark ran at
embedding size 10 because "some baselines … are not runnable" at its default
100 (p. 39) — so a matched-capacity comparison is what its own authors did,
and d_z = 20 / d_w = 6 against embedding 10 is defensible.

**Two instrument gaps this review opens.** (i) resolVI's **double-positive
metric** (mutually exclusive marker-gene pairs from a matched scRNA-seq
reference, Poisson-mixture threshold, Methods p. 21) is the one external
metric we can compute on our own slides with our own tile split *without
installing anything* — it scores raw counts, our x̃ = x − κℓρ̄ and the κ grid
on a criterion resolVI's own authors would accept, and it turns the parked
x̃ question into an externally-judged one. (ii) SIMVI's **axis benchmark**
(spatial effect must track the layered axis, not the orthogonal axis;
Kendall's τ true-positive vs false-positive, p. 5) maps directly onto our
tumour-band niches: w must track the band ordering and not the orthogonal
in-plane coordinate. Both are new instruments, both are small.

**Caveats.** All five read as text layers; figures were not inspected, so
every number quoted from a figure panel (MintFlow Fig. 1g/1h effect sizes,
SIMVI Fig. 2b/5d score bars, resolVI Fig. 2E/2F) is described, not
transcribed. Supplementary Notes are not in the PDFs: SIMVI's identifiability
proof (Note 1), its positivity formulation (Note 2) and parameter sweep (Note
3), MintFlow's method comparison (Note 1), identifiability proof (Note 2),
encoder/decoder architectures (Notes 3–4) and dataset descriptions (Note 6),
and Celcomen's proofs were **not read** — MintFlow's Supplementary Note 1 was
already recorded as unretrievable in the survey entry and remains so. No
runtime figure is published for SIMVI, MintFlow or Celcomen. Nothing here was
run; all cost figures are extrapolations carried over from the survey entry.

### Handover refreshed, 3.3, paper A2–A7 (2026-09-21)

*Delegated (Opus), documentation only.* `docs/handover.md` rewritten in place
(state 2026-09-21), same nine-section structure, from the devlog since
"Crystallisation" (2026-09-14) and the registers; every number with its
artefact path, negatives as findings; retired in §5 with reasons: per-type
‖w‖ ranking, "6 programmes", matched-column B stability, the pre-correction
κ-sensitivity object, the 2026-09-14 α_w = 0.05 candidate. **Todo 3.3
closed** (text only): the report's cycle section quotes the absolute read and
states the z / linear-reference ratio as a target-reliability property with
the depths (2.1 ovarian and lung FFPE, 0.96 GSE core, 0.90 FF). **Paper
A2–A7 applied**, model description only: budget-and-seed-selection paragraph;
the per-type translation gauge in the identifiability paragraph; type-level
counterfactual marked as exercised; implementation rows for type_only
sources (empirical case), the two budgets, seed selection, halo overhead
without numbers, int16 counts, a "built but not default" row; the choices
table split in two. Build clean, 19 pages, no undefined references. Sections
B and C of the revision list remain pending.

**Contradictions found by the rewrite, resolved here.** (1) The pinned
reference's cycle_z appears as 0.499 (metrics.json final, training-time
probe on the val split), 0.440 (doc-08 battery, block-CV subsample) and
0.460 (the 2026-09-21 cycle-target CLI, which collects latents over the
validation tiles with its own ridge) — three instruments on the same
weights; the report and handover quote the first two with their instrument
named, and the third is labelled as its own instrument in the cycle-target
entry. (2) `spec_deviations.md` gives α_a = 0.03 (closed-form era) and 0.3
(adversary) in one paragraph — both are correct for their invariance mode;
the paragraph now says so. (3) Lung `reference_graphclust` cycle_z is 0.347
in metrics.json and 0.369 in the cycle-target CLI — the same two-instrument
difference as (1). (4) todo 2.3's status line quoted type_z-era numbers as
current; corrected to name the era.

### L2 on w so that B carries the programme (motivation, 2026-09-21)

**Why (author).** The loadings B should be the informative object: a gene
programme is a column of B, and w says how far a cell moved along it. With
no penalty on w's magnitude the scale can sit in w (large w, small B) or in B
(small w, large B) — spec §7.12 names this rescaling gauge as the reason
σ_w is fixed, and issue V12 found a translation gauge on top of it (per-type
offsets 4–10× the within-type spread; coupled Adam weight decay moved the
offset *into* w, ‖B·mean_t w‖ 18–28 → 81–92). An L2 on the latent w itself,
not on parameters, would push magnitude into B and could make B more
interpretable and its columns more comparable across seeds.

**Questions to settle before any fit.** (1) What does an L2 on w add that
the KL(q(w)‖N(m_ψ, I)) does not — the KL already penalises w's *deviation*
from the prior mean, so an L2 on w penalises the prior mean itself (m_ψ's
output) as much as the deviation; is that the intended object, or should the
penalty be on E_batch[m_ψ(c,t)] per type (the V12 proposal), or on the
realised shift B·w in gene space? (2) Gauge: with σ_w fixed at 1 the rescaling
gauge is already pinned by the prior; what an L2 on w changes is the
*translation* gauge (where the per-type offset lands) — state which. (3) The
model already reads programmes gauge-centred at read time; what would a
training-time penalty change in the atlas' effective rank, shares, axis
cosines and labels? (4) Interaction with α_w: an L2 on w is a second pull
toward zero on top of a pull toward m_ψ; it could close the deviation
channel further or shrink the prior field.

**How.** Analysis first (the four questions, with the gauge algebra written
out), then a small pre-registered test on the ovarian slide: three variants
at one seed each, 200-epoch budget, `type_only` defaults — (a) L2 on the
sampled w (weight λ_w on E‖w‖² per cell), (b) L2 on the per-type mean of
m_ψ (the V12 penalty), (c) none (sweep3_k0.1_s0 as control) — at λ chosen so
the penalty is ~10 % of KL_w at initialisation; then the seed triple for the
variant that passes. **Evaluated by:** guards inside the seed envelope
(recon ±0.06, NMI ≥ 0.63, cycle_z ≥ 0.44, probe at floor, mirror ≤ 0.05);
‖mean_t w‖ per type falls toward the within-type spread (V12's numbers:
offset 4–10× the spread → ≤ 2×); atlas effective rank and the two programme
labels unchanged; **cross-seed axis-2 cosine improves** (0.40–0.83 at 0.1
today) — that is the "B more informative" claim made measurable; transport
counterfactual unmoved (it works on differences). **What is wished for:** a
penalty that fixes the translation gauge at training time without touching
the reads that already work; if the guards move or the axis cosines do not
improve, read-time centring stays and the penalty is recorded as tried.

### Ring-2 skip and κ-survival in shift space (2026-09-21)

*Delegated (Opus), GPU 1; `networks.py` forward, `train.py` `_to_device`,
`validate.py` companion; 55 tests across the five touched files pass, full
suite exit 0.*

**Ring-2 skip (todo 1.6, spec §4.5).** Under `type_only` the encoder runs on
seeds ∪ ring1; ring 2 stays in the tile (ring 1's ρ_j needs it for ρ̄) but is
a type/Φ lookup, as the spec says. `type_z` unchanged. Proof on the
fixed-seed synthetic smoke fit: 56 losses agree to 1.2e-7 relative (one
float32 ulp), every metrics.json number to ≤ 9.5e-7 absolute — float
tolerance, not bitwise, because the encoder GEMM's batch dimension changes
and re-blocks the matmul; the reparameterisation draw is still made at the
node count so the RNG stream is untouched. Planted test: ring-2 rows absent
under `type_only`, present under `type_z`; poking ring-2 counts moves ring-1
contexts only under `type_z`. **Halo at 4,096-cell tiles on ovarian
(`experiments/halo_overhead.json`): nodes encoded per step fall from +14.3 %
of seeds to +6.7 %, resident counts −6.6 %, seconds per epoch unchanged
within run-to-run noise** (whichever variant runs first is faster; the step
is decoder- and influx-bound over seeds). The saving is nodes and memory.

**κ-survival moved to shift space.** The producer of
`atlas_kappa_survival*.json` was no longer in the repo (an ad-hoc script);
it is now `validate --sweep-tag --analyses kappa_survival`, reusing the
atlas' own basis, cross-seed, label and recurrence functions so the two
cannot drift. Regenerated on sweep3 (18 runs). Against the pinned reference:
axis-1 |cos| 0.96 at κ ≤ 0.2 falling to 0.92 at κ = 0.3–0.4, shift overlap
0.90 → 0.85; sweep-internally 0.94–0.96 and 0.78–0.89, flat across the
ladder. **Hallmark labels are the most κ-stable object: EMT, HYPOXIA and
G2M_CHECKPOINT recur in all three seeds at every κ ≤ 0.3.** At κ = 0.1 the
companion reproduces the atlas triple (axis-1 0.933 / 0.955, overlap
0.80–0.95 vs the atlas' 0.93–0.98 and 0.67–0.93; sweep3 is a 200-epoch
budget against the atlas' 500, which is where axis 2 diverges). The retired
matched-signature numbers stay under `legacy` for one release; recomputed
they do not reproduce the previously published 0.29–0.43 (now 0.32–0.71)
and the old producer is gone — one more reason the metric is retired. The
old internal file's numbers were overwritten before carry-forward logic
existed and survive only in the 2026-09-21 devlog prose (recorded as the
agent's fault).

### Transport at the distribution level (results, 2026-09-21)

*Delegated (Opus), one GPU; `discell/model/transport.py` (`--read
distribution|both`), the §7 sub-block of `report.py`, 7 new planted tests
(14 in the file); artefacts `runs/<run>/transport/transport[_tumour-band]_
distribution.{json,png}` on the three ovarian seeds × two niche sources plus
FF `reference_graphclust`. The mean read reproduces every pinned number
exactly. Artefact and figure verified.*

**The construction, and one addition.** Source cells keep their own μ_z,
are given the target niche's mean context through the prior head and the
target's mean influx as the leak source, and are decoded; the cloud is
compared with the held-out cells living there as distributions (unbiased
MMD², Gaussian kernel on the Hellinger map, bandwidth = median pairwise
distance in the target; energy distance beside; floor / untransported /
type-mean / observed-source references; sizes matched to min(|S|,|T|,2000);
200-draw paired bootstrap). **As pre-registered the read is structurally
blind, and the reason is depth**: it compares predicted *rates* with raw
*multinomial* compositions, and the target's shot noise is ~500× the floor,
a near-constant offset in every distance; gap closed collapses to ~0
(median 0.004; an exactly correct prediction scores 0.11 in a planted
world). A **count-matched** companion — counts drawn from each predicted
rate at a depth drawn from the target — puts both sides on one geometry and
is reported beside the pre-registered columns, never instead. Numbers below
are count-matched unless said.

**Bars, pinned reference (158 pairwise / 62 leave-one-out panels).** (1)
transported below untransported **147/158**, CI excluding zero 116 — met
(pre-registered geometry 96/158). (2) median gap closed **0.45** pairwise,
**0.30** pooled; pooling gives the tighter CI in **107/158** at a cost of
0.08 (tolerance 0.1) — met. (3) transported below the type-mean predictor
**0/158** — **not met**. (4) Spearman with the mean read's counterfactual R²
**0.57** — met (0.44 as pre-registered, the one marginal miss there).

**Envelopes.** Pairwise median gap closed: ovarian 0.40 / 0.45 / 0.44
(composition niches) and 0.38 / 0.59 / 0.38 (tumour bands); **FF 0.44 with
264/264 panels improved and 250 CIs excluding zero** — the reliable slide is
unanimous, as on the mean read. Leave-one-out 0.29–0.36 throughout.

**Bar (3) fails, and that is the finding.** Count-matched, the type-mean
predictor — one mean composition resampled at matched depths — closes
0.90–1.00 of the gap on every leg. At Xenium depth the within-type-within-
niche spread is essentially shot noise, so a population read cannot see
per-cell structure: **the distribution check is the mean check in different
clothes**, exactly what bar (3) was pre-registered to detect. Claim that the
model moves a population's *location*, not that it reproduces its spread.

**Pooling buys what it was meant to.** The leave-one-out CI is tighter in a
majority on every leg (up to 185/264 on FF); the pooled gap closed is lower
by 0.02–0.09, inside tolerance but consistently signed: **z is context-free
enough that cells from every other niche land on the target nearly as well
as cells from one neighbouring niche, with a small residual cost.** The
"pooling hurts" fail state does not fire.

**Readability additions to the mean read.** (i) Fraction of the noise
ceiling is the headline with raw R² beside it: **0.34 on the trusted tier**
(0.81 all-panel is a ratio of means over unmeasurable panels and is not a
percentage of reachable signal), 0.51 tumour-band trusted, 0.55 FF.
(ii) Top-50 predicted-up gene overlap with the observed top 50: **7.3 of 50**
all-panel and **15.3 of 50** on the trusted tier against a chance level of
0.7–0.8 (9× and 22× chance); FF 16.8 and **21.1 of 50**. Both ship with the
chance level in every tier.

**Caveats.** The count-matched companion is an addition to a fixed
pre-registration (both variants ship; the pre-registered columns are the
record). Bootstrap duplicates bias absolute CIs slightly upward; comparisons
across panels are unaffected. The type-mean reference tests spread, not
location. GSE core not run (below its noise floor on the mean read). A
`numpy.multinomial` crash on pvals summing to 1 + 1 ulp was fixed mid-run.

### L2 on w (results, 2026-09-21)

*Delegated (Opus), one GPU; `elbo.py` (+`Weights.lambda_w` / `.w_penalty`),
`train.py` (the two fields and CLI), 9 planted tests; runs `wpen_a_s{0,1,2}`,
`wpen_b_s0`; artefacts `experiments/w_penalty{,_offsets,_shift}.json`.*

**The analysis predicts the negative before the fits.** Given the KL to
N(m_ψ, I), E‖w‖² = ‖m_ψ‖² + 2⟨m_ψ, μ_w − m_ψ⟩ + ‖μ_w − m_ψ‖² + Σσ²; the KL
already charges the last two, so at KL_w ≈ 0.002/dim an L2 on the sampled w
is an L2 on the prior field m_ψ plus a constant. The within/between split
E‖m_ψ‖² = Σ_t (n_t/n)‖m̄_t‖² + E‖m_ψ − m̄_t‖² makes variant (b) equal to (a)
with the context-varying channel exempted — the one channel 2.3 measured as
buying likelihood — so (b) is the right object on the algebra. Gauge: σ_w = 1
pins rescaling only through the deviation; at the pinned operating point the
offset channel is scale-free as well as translation-free (sharper than V12:
that is why a huge offset costs nothing). An L2 on B·w is gauge-invariant and
only shrinks the programme. **The atlas cannot move**: every read is on
Δ_i = B(w_i − w̄_t), exactly invariant to both gauges, so axis cosines are not
a quantity a gauge penalty can improve. Fits run as pre-registered anyway.

**Calibration.** λ·pen(init) = 0.10·α_w·KL_w(init) → λ_a 6.5e-4, λ_b
6.6e-3. Hazard: 97 % of E‖w‖² at init is the posterior variance the KL pins;
the rule is inert at init and ~27× the KL term at the measured end state.

**(a) L2 on w passes every guard on three seeds and fixes the gauge.** recon
−7.254 / −7.181 / −7.231, NMI 0.655–0.675, cycle_z 0.466 / 0.481 / 0.439,
probe below floor, mirror 0.044–0.046. Offset ratio ‖mean_t w‖ / within-type
spread median 2.40 → 1.72, ‖global mean w‖ 1.94 → 0.35, B's leading column
2.85 → 4.37 — magnitude did move into B. Transport unmoved (0.078 → 0.084,
slope 1.15, beats-both 101/155). Atlas: rank 2/2/3, dominant programme
identical (F13A1, MRC1, PLTP, KLF4), labels EMT / HYPOXIA / G2M. **Axis-2
cross-seed cosine did not improve**: 0.606 / 0.476 / 0.920 against the
control triple's 0.694 and the record's 0.40–0.83; one axis-1 pairing fell to
0.578. The realised within-type shift shrank 1.5× (6.17 → 4.21) — the
predicted tax on the working channel.

**(b) the per-type-mean penalty fails.** cycle_z 0.392, mirror 0.052, rank
collapsed to 1 ([0.9996, 0.0004]), the dominant macrophage/stromal programme
lost and replaced by a hypoxia/glycolysis axis, axis-1 vs the control triple
0.63–0.67, transport program_only 0.046 → 0.013 with slope 1.69. Mechanism,
from the gauge-invariant read: the realised within-type shift collapsed 6.17
→ 0.92 while the offset shift fell 12.7 → 0.98. Pinning m̄_t at zero leaves
the within-type scale unpinned and the optimiser took it; the excellent
offset ratio 1.16 is two numbers near zero.

**Verdict: drop as default; (a) recorded as an option (`--w-penalty w
--lambda-w 6.5e-4`), (b) rejected, (c) not run.** The success criterion was
a gauge-invariant quantity, so it could not fire; the fits add that the
penalty's cost is real (1.5× and 6.7× shrinkage of the working channel).
Read-time centring stays; V12's training-time clause is closed as not worth
closing — nothing downstream reads the gauge. **Caveats:** single λ per
variant (no ladder); (b) at one seed; (c) argued not measured; the record's
"α_w = 0.1 triple" cosines (0.64 / 0.83 / 0.40) do not reproduce exactly from
either `sweep3_k0.1_*` or `ablation_gat_type_only*` `programs.npy` (0.656 /
0.772 / 0.525 on the latter) — which runs back that line should be pinned.

### Baselines and article-derived experiments: three packages (motivation, 2026-09-21, evening)

**Author's direction.** Focus on the baselines and the experiments the article
review surfaced; install the baseline tools in `/home/rmolen/github/
DisCell-baselines` where a separate environment is needed (resolVI:
scvi-tools ≥ 1.3, pip; SIMVI: `pip install simvi`; MintFlow: `pip install
mintflow` with extra dependencies).

**Package 1 — analysis trio on existing runs (6b.1, 6b.3, 6b.4).**
*6b.1, MintFlow's criterion on our channels.* For every panel gene, the share
of its held-out expression shift attributed to (i) the response channel
B·Δm_ψ, (ii) the leak channel κ·Δρ̄, (iii) both, on the transport panels;
compare the distribution of shares for ligand–receptor genes (CellChatDB
union used by doc-09) against all other genes, across κ ∈ sweep3 and on the
pinned reference; composition-residualised per doc-09 §2. Pre-registered
read: if the leak channel alone gives LR genes a higher microenvironment
share than other genes (Mann–Whitney, effect size reported), MintFlow's
real-data validation criterion is reproducible from misassignment alone; if
only the response channel does, it is not. *6b.3, MIG / MIC.* y = niche
label (K = 10 composition; six tumour bands), I(y;z), I(y;w), I(w;z|y) by
kNN (Kraskov) and MINE on held-out tiles, within-type permutation floor; MIG
= (I(y;w) − I(y;z))/H(y), MIC = I(y;w)/(I(y;w)+I(y;z)); on the ovarian seed
triple, lung, FF. Expectation: MIC near 1, MIG well above the floor; the
z-niche † cell gets I(y;z) with its floor. *6b.4, SIMVI's axis test.*
Per-gene Kendall τ of the w-predicted shift (and of raw held-out shift, z-
probe shift, ℓ-baseline) against the ordered tumour-band index (true axis)
and against the orthogonal in-plane coordinate within the same cells (false
axis); true-positive and false-positive gene counts at |τ| thresholds as
SIMVI reports them. Expectation: w tracks the band ordering far above the
false axis; a symmetric result would say the niche reads are colocalisation.

**Package 2 — objective ablations (6b.5).** Four arms, ovarian, 3 seeds at
the sweep budget, type_only defaults: (i) drop the second KL copy
((1+ω) → 1); (ii) drop the intrinsic path (ω = 0); (iii) class-mean prior
m_ψ := mean_t μ_z-derived per-type constant in place of m_ψ(c,t) (DisCoVR's
prior); (iv) adversary on the reconstruction x̂ instead of μ_z. Read: the
full quadrant and guards vs the sweep3 κ = 0.1 triple. Expectation from spec
§6.2–6.3: (i) and (ii) cost z (cycle_z, NMI) — the terms are load-bearing;
(iii) closes the context channel (w rows fall); (iv) is DisCoVR's variance
reduction and may be neutral. Any arm that matches or beats the reference on
every read is a finding about the objective.

**Package 3 — baseline installs and smoke tests (4.2 stage 0).** resolVI via
scvi-tools as an optional extra of the project env (dry-run showed zero
downgrades); SIMVI in its own venv (its pin on old scvi-tools; python 3.10)
under `DisCell-baselines/simvi`; MintFlow in its own venv under
`DisCell-baselines/mintflow` (zarr < 3 conflicts with our image path).
Each: install, import, run the smallest tutorial to completion, then a
5,000-cell smoke fit on the GSE core exported as AnnData with our tile
split and our pruned graph attached; record versions, wall time, and what
the method emits (latents, corrected counts, per-cell mixture weights).
No full baseline fits yet — the fairness protocol (survey entry) governs
those and they are launched detached after the smoke tests pass.

### External criteria on our runs: MintFlow's signalling-gene share, MIG/MIC, SIMVI's axis test (results, 2026-09-21)

*Delegated (Opus), GPU 0; new module `discell/experiments/external_criteria.py`
(three subcommands, no model file touched), 22 planted tests; 13 runs;
artefacts `experiments/external_{signalling_share,mi_quadrant,axis_test}_
<run>.{json,png}`; headline effects and MIC values verified against the JSONs.*

**6b.1 — MintFlow's own validation criterion is reproducible from
misassignment alone; the strongest related-work result the programme has.**
Per panel gene the held-out log-rate shift is split (all sides centred) into
response |B·Δm_ψ|, leak |κ·Δρ̄| and remainder; shares summed over 158 ovarian
transport panels; the CellChatDB ligand ∪ receptor union (861 in-panel genes)
compared with the rest by Mann–Whitney with a rank-biserial effect. **The
leak channel separates LR genes from the rest at effect +0.23 to +0.27
(p ≤ 5e-26) at every κ from 0.05 to 0.4; the response channel manages +0.008
to +0.057 and is not significant at κ ≥ 0.2.** The leak effect is flat in κ
while the leak's mean share triples (0.040 → 0.203); the response effect
decays monotonically as κ rises (0.049 → 0.008) — the contamination channel
takes the LR signal off the biology channel. κ = 0 is exactly zero by
construction. FF reproduces it on 264 panels (leak +0.134, p = 3e-8;
response ns). Abundance-matched non-LR controls leave the leak effect at
+0.24. Everyone in the split literature validates on the neighbour-resembling
half of the transcriptome; this measures what that costs.

**6b.3 — MIG holds, raw MIC fails, and the floor explains both.** Ross-kNN
and MINE agree on all eight run × niche-source cells (ovarian seed triple ×
{K = 10 composition, six tumour bands}, lung, FF). Raw MIC is **0.44–0.60**,
never near 1; on FF raw MIG is negative (−0.10). But z's excess over the
within-type permutation floor is **+0.07 to +0.13 nats against w's +0.38 to
+0.51**: 86–89 % of I(y;z) is what any type-informative latent gets for
free because niches differ in composition. Floor-corrected, **MIG +0.13 to
+0.23 and MIC 0.79–0.87** (0.86–0.87 on the ordered bands) on every seed and
slide. I(w;z|y) 0.16–0.37 (kNN) / 0.10–0.29 (MINE). **The flagged z-niche †
cell is answered: it is type identity, not niche.** MIC without its floor is
uninterpretable on spatial niche labels and must never be quoted alone — a
correction to the metric as published, offered with the fix.

**6b.4 — SIMVI's axis test comes back asymmetric on all three seeds.** True
axis = six ordered tumour bands; false axis = the in-plane coordinate least
correlated with band index (y on ovarian), cut into six equal-count bins
within type on the same cells. w-predicted mean |τ| **0.918 / 0.847 / 0.933**
on the true axis vs **0.516 / 0.585 / 0.600** on the false; at |τ| ≥ 0.9 the
true/false panel counts are 8616/128, 10748/1532, 10069/736 (67× / 7.0× /
13.7×). w beats every reference row on the true axis (raw 0.46–0.47, z
0.62–0.70, ℓ 0.49–0.55). "Your niche reads are type colocalisation through
contiguity" is answered no. Caveat: the false axis is not an independence
null — every row scores 0.34–0.60 on it because slide geometry is
autocorrelated; the claim is the relative one SIMVI itself makes.

**Caveats.** 6b.1's share is of the *magnitude of a shift*, not of counts as
MintFlow computes it; only the LR-vs-other contrast is claimed. Doc-09's
exposure residualisation has no direct analogue within a single-type panel;
the composition control is an abundance-matched non-LR set (deviation
recorded; matched and unmatched agree to ±0.02). The unexplained share is
0.68–0.84 (consistent with the transport noise ceiling), so both shares are
small and only their contrast is read. 6b.3 capped at 20,000 held-out cells,
10 permutations; MINE small and a second opinion. 6b.4 pools 4–6 types per
run; seed s1 admits 6 and has the weakest ratio. One bug fixed mid-run:
sklearn's radius query rejects per-point radii on its fast path; both MI
estimators now use cKDTree (the 6b.1 and 6b.4 runs do not use that path).
Not done: 6b.1 on tumour-band niches; 6b.3 on FF bands (undefined).

### DAPI vs Scanpy on the fresh-frozen slide (2026-09-21)

*Delegated (Opus), CPU only; `scripts/cell_cycle/generate_score_correlation_
maps.py` run as-is on FF (→ `figures/cell_cycle_score_correlation_per_
celltype.{png,csv}` on FF), `dapi_cycle.bimodality` imported; scratch under
`scratchpad/cellcycle_ff/`; 250k-cell subsample per slide where the counts
matrix was needed; medians spot-checked.* The question: does 8× depth rescue
the marker score, and how does DAPI compare where it does?

**Scanpy: rescued by counts, not by chemistry.** Split-half reliability by
depth decile is one curve across both slides — FF G2M 0.13 (162 counts) →
0.89 (4,001); ovarian 0.02 (18) → 0.75 (760). At a matched 150–300 counts FF
is the *worse* slide (G2M 0.14 vs 0.45; S 0.10 vs 0.20), plausibly because
such cells are FF's bottom 9 % and ovarian's median. FF's pooled 0.69 vs
0.53 is composition: 91 % of FF cells exceed 300 counts against 34 % of
ovarian. External validity rises only slightly (depth-partialled G2M ~ MKI67
+0.17 → +0.24, ~ TOP2A +0.24 → +0.27; both are list members, so part of it is
self-correlation). **Reliability is to be quoted per depth stratum from
here on, not per slide**, and no wording should credit the fresh-frozen
chemistry.

**DAPI: the gate-level verdict stands; a per-cell signal exists and is too
small.** Depth-partialled DAPI ~ MKI67 / TOP2A on FF: median +0.028 /
+0.003, IQR inside ±0.04 — the ovarian 0.00–0.05 band, unmoved; raw DAPI ~
total counts is *worse* on FF (0.63 vs 0.48). Bimodality of the log integral
within type: 0/36 FF, 0/15 ovarian; BIC alone would split 35/36, the dip
refuses 35/36 (the one dip-positive cluster is 3.6 % MKI67+ with ratio 3.6 —
neither cycling nor 2N/4N); all four high-MKI67 FF clusters dip p ≥ 0.986.
MKI67+ Q4/Q1 across DAPI quartiles at fixed type and depth 1.03–1.63
(median 1.26) — the raw 2.1 was depth, the joint-label collapse again.

**New, and why this entry exists.** Within type *and* depth decile,
ρ(DAPI, G2M score) on FF is +0.21 median against ovarian's −0.03, and in the
two small high-MKI67 clusters it survives partialling depth *and* nuclear
area: Cluster-34 decile 8 ρ **+0.48** (DAPI density alone +0.45, area alone
+0.33), Cluster-36 decile 6 +0.40; G2M AUROC for DAPI-Q4 vs Q1 **0.86–0.89**
there — above the ≥ 0.70 bar the gate-level read missed at 0.54. It rises
monotonically with depth in all four FF clusters and shows the same rising
shape on ovarian's proliferative tumour (0.11 → 0.26, AUROC 0.45 → 0.65)
capped by that slide's depth; ≈ 0 in non-cycling types. **Reading: a faint
DNA-content signal is real and was hidden by the yardstick — below ~2,000
counts the G2M score is too noisy to detect it.** Unusable as a label:
confined to ~6.5k cells (0.5 % of the slide), invisible below median depth,
no MKI67 enrichment, no second mode to cut. Wording changes from "no signal"
to "a signal roughly an order of magnitude below the twofold a 2N/4N gate
needs, on top of the projection problem". Todo 3.5 stays closed.

**Caveats.** 250k-cell subsamples (pooled reliabilities reproduce the record
to ±0.02); the matched-depth band is matched on depth not cell quality; §4
on the raw integral, complementing gate 5; the two signal-carrying clusters
are 2,043 and 4,496 cells (~200–450 per decile, ρ SE ±0.08, 40 strata
untested for multiplicity); FF cluster identities are numbers, not names.

### Baseline tools installed (4.2 stage 0, 2026-09-21)

*Delegated (Opus), GPU 0 for smoke fits only; `discell/experiments/
export_for_baselines.py` (new), `pyproject.toml` optional extra `baselines`,
`/home/rmolen/github/DisCell-baselines/{resolvi,simvi,mintflow}` each with a
README and a smoke script; full suite 303 passed, 1 skipped. No full fits.*

**All three install and run; the survey's install plan holds.** resolVI via
scvi-tools 1.5.1 as the `baselines` extra of the project env, zero downgrades
by lock diff (21 additions; torch 2.13, zarr 3.3, anndata 0.13 untouched).
SIMVI in its own python 3.10 venv: `pip install simvi` resolves but does not
import (its scvi-tools ≤ 0.16.2 pin paired with a 2026 anndata), so the
2022 stack is pinned by hand (scvi-tools 0.16.1, anndata 0.8, lightning
1.5.10; torch 2.1.2+cu121 works on the 4090). MintFlow in its own python
3.11 venv (mintflow 0.3.0, torch 2.6.0+cu124, zarr 2.18) with two traps:
`mintflow[all]` upgrades torch and breaks the pyg wheels; `xarray_schema`
needs setuptools < 81. **wandb: off** (`flag_enable_wandb='False'` plus
`WANDB_MODE=offline`); nothing about the slides leaves the machine.

**Export.** `export_for_baselines.py` writes an .h5ad from `prepare.assemble`
so split and graph equal a DisCell fit: counts, spatial coords, the label that
becomes t, tile id and train/val flag, log depth, degree, a slice id, and the
pruned Delaunay as `obsp['discell_connectivities']` (symmetric binary) and
`obsp['discell_beta']` (directed weights). GSE core: 69,422 × 5,001, 35
types, 203,857 edges, 27 + 5 tiles; plus a contiguous 5,000-cell disc.
anndata 0.13's string index is unreadable by SIMVI's anndata 0.8 →
`read_h5ad_legacy.py` reads at h5py level.

**Smoke fits, 5,000-cell window.** resolVI 5 epochs in 4.4 s: one latent
(·, 10), corrected counts, per-cell mixture **0.831 true / 0.181 diffusion /
0.001 background** — a first contamination read of ≈ 0.18 against the
transcript-flux 0.13 and the κ grid (5 epochs; not quotable). SIMVI 5 epochs
in 3.0 s: intrinsic (·, 20) and interaction (·, 6) at matched d_z / d_w; no
corrected counts, no contamination. MintFlow 271 s per epoch (~60× resolVI;
consistent with 20–26 GPU-h): Z / S_in / S_out at width 100, Xint / Xmic
count matrices. **Graphs:** SIMVI takes our edge list directly (tested both
ways, same cost); resolVI insists on its own k = 10 spatial kNN (ours would
need `obsm['index_neighbor']` written by hand — not done); MintFlow insists
on a squidpy graph with no external hook — "both graphs" is a real row for
those two. Upstream defects recorded, not patched: SIMVI's full-batch GPU
path raises (use minibatches); MintFlow mutates its config dict in place.

**Caveats.** The published tutorial datasets were not run (resolVI's figshare
file returns HTTP 202 empty; SIMVI's notebook not fetched) — the full API path
was exercised on our window, so the tutorial gate is met in substance not
letter. MintFlow's microenvironment score is NaN on cells with < 5 counts;
filter before quoting. The loader's "serving zeros of width 384" line is the
bundle opening without Φ before `assemble` loads it (the FF entry recorded
the same trap); the GSE fits carry no "cells lack an image embedding"
warning, so they used Φ — checked 2026-09-21.

### Transport at the distribution level, second round: model-vs-model distance and matched twins (motivation, 2026-09-21)

**Why.** The first distribution read (6a.1) scored transported cells against
the target niche's *raw* counts. The author's objection: that comparison
includes the model's reconstruction error, which is the same for every
predictor and has nothing to do with transport. The read also could not
separate a transport that moves cells correctly from one that only moves the
mean, because at Xenium depth the type-mean predictor closed 0.9–1.0 of the
gap. Two additions, both pre-registered here before running.

**Read A — model-vs-model MMD.** Same panels (type × source niche → target
niche, pairwise and leave-one-niche-out), same Hellinger-map Gaussian-kernel
MMD², but the target cloud is now the target cells' *own decoded probability
vectors* (z and w from their posterior means, their own context and leak),
not their normalised counts. Both sides are then smooth model outputs; shot
noise disappears from both, so no count-matched companion is needed.
References unchanged: floor (two halves of the decoded target), untransported
(source cells decoded in their own niche), type-mean (decoded target mean
replicated), observed source. Headline is gap closed. What is wished for:
gap closed higher than the count-level read, and — the real question — the
type-mean predictor no longer at 1.0, because model-side spread is not shot
noise. If type-mean still closes ≈1.0 the within-niche spread of decoded
cells is itself tiny and the "spread" limit is in the model, not the data.

**Read B — matched twins.** For each target cell, its nearest source-niche
cell in z (Euclidean on μ_z, same type), transported into the target niche
and compared *cell to cell* with the target cell's own decoded vector
(Hellinger distance). Three references per panel: the untransported twin
(niche difference before correction), a *random* same-type source cell
transported (does matching on z matter at all), and the floor of two decoded
target cells that are z-nearest neighbours of each other within the target
niche. Report: median per-cell distance for each, and gap closed
(untransported − transported)/(untransported − floor); and the twin margin
(random − matched)/random. What is wished for: transported twin closer than
untransported (transport works per cell) *and* matched closer than random
(z carries per-cell information across niches). Failure of the second clause
while the first passes means the population read was all there ever was.

**HVG companion.** Both reads also on the top-1000 highly variable genes
(Scanpy `seurat` flavour on the training cells, renormalised on the subset),
because low-count genes only add noise to a 5k-gene Hellinger map. Reported
beside, never instead.

**Where.** Pinned reference `xenium_prime_ovarian_cancer_ffpe/runs/ablation_gat_type_only_s1`
(both niche sources) and the fresh-frozen `xenium_prime_human_ovary_ff/runs/reference_graphclust`.
Agent implements in `discell/model/transport.py`, reports in the run's
`transport/` folder and report section; no docs edited by the agent.

### Objective ablations, DisCoVR-style (6b.5, results, 2026-09-21)

*Delegated (Opus), one GPU; `elbo.py` (+`Weights.second_kl`), `train.py` (three optional `TrainConfig` fields and their CLI), `networks.py` (+`ClassMeanPrior`), one line of `validate.py`'s `load_run`, 11 planted tests in `tests/test_model_ablations.py`; runs `abl_{no_second_kl,no_path_b,class_mean_prior,adv_on_xhat}_s{0,1,2}`; artefact `experiments/objective_ablations.json`. Control `sweep3_k0.1_s{0,1,2}` at the same budget, whose transport read was run on s1/s2 to match.*

**Arm (i) is an identity, and that is the first finding.** At ω = 1, charging the z-KL once instead of `(1+ω)`-fold is *exactly* `α_z → α_z/2` — bit-for-bit, now a test. So the arm cannot speak to §6.2's argument, which is about the bound property, and what it measures is the α_z ladder. On that ladder the halved point is better on this slide: cycle_z 0.53/0.54/0.53 against the control triple's 0.41/0.47/0.47, recon −7.226/−7.156/−7.195 against −7.259, probe below its floor, transport unmoved. **The `(1+ω)` factor is defended by the derivation and by nothing in the read-outs; α_z = 0.007 is not the best point for z's continuous content.** No ovarian α_z ladder exists to place it — pre-registered as the follow-up.

**Arm (ii): path (b) is load-bearing, and the matched comparison is the one that shows it.** ω = 0 also sets the `(1+ω)` factor to 1, so (ii) is (i) with the intrinsic path removed and nothing else — and cycle_z falls 0.53 → 0.41/0.43/0.37, recon by 0.04 nats, with no seed overlap. Against the control alone the drop is mild (0.37–0.43 vs 0.41–0.47), because the control's second KL copy pushes the other way; reading (ii) against the control only would have understated the term. NMI, mirror, probe and transport are untouched: **(b) buys z's within-type continuous content specifically, exactly the pressure §6.3 says nothing else supplies.**

**Arm (iii): the DisCoVR prior closes the context channel and w goes feral.** `p(w|t) = N(μ_t, I)` with a learned per-type vector, posterior untouched. Transport counterfactual R² 0.078/0.100/0.097 → 0.052/0.059/0.051, slope 1.90/2.35/1.97, and **beats-both 94/105/98 → 0/0/0 out of 155**: the response channel no longer predicts a held-out shift better than either single-channel baseline, on any panel, on any seed. The mechanism is visible in the guards: with no context in the prior, w is a free per-cell latent and takes what z should hold — cycle_w 0.003/0.006/0.008 → **0.104/0.147/0.124**, NMI 0.64–0.66 → 0.60–0.62, KL_w up ~5×. The atlas fragments to rank 6 on all three seeds (variance fraction .35/.32/.15/.11/.05/.02 against the control's .998/.002), the dominant programme becomes proliferation (CENPA, TOP2A, CEP55, AURKB, BUB1) in place of the macrophage/stromal one (F13A1, MRC1, KLF4, TNXB), and cross-seed axis cosines fall to 0.26–0.69 with shift overlap 0.39–0.62. **`m_ψ(c, t)` is what makes w a response and not a second identity latent.**

**Arm (iv): the adversary on x̂ is not neutral — it defeats itself.** Heads on `log ρ = log_softmax(a(z)+Bw)` instead of μ_z: probe Δce **0.195/0.142/0.189** against a control of 0.013/−0.021/0.027 and a floor of −0.05, mirror R² 0.047–0.050 → **0.100/0.095/0.101**. Both invariance guards fail together, and the reason is structural: with ρ as the head input the encoder can satisfy the penalty by moving the decoder, so niche information stays in μ_z. w collapses onto its prior (KL_w ~ 0), the atlas is rank 1 with an unrecognisable programme and cross-seed cosines of 0.08–0.32, and transport falls to 0.060–0.071 with slope 1.6–2.3. The higher cycle_z (0.48–0.51) is the same effect read positively: z keeps what the penalty was meant to remove. **DisCoVR's variance-reduction argument does not transfer to a penalty whose target is the encoder's own latent.**

**Verdict: the objective survives. No arm matches or beats the reference on every read; three of four are decisively worse on the read they were built to stress.** Recorded as options, all default-off: `--no-second-kl` (documented as `α_z/2`), `--class-mean-prior`, `--adv-input xhat`. **Caveats:** one slide, one budget, one seed triple; arm (i) is confounded with α_z and the ladder that would deconfound it was not run; (ii) at ω = 0 only, no ω ladder; (iii)'s μ_t is learned rather than accumulated as DisCoVR's `E[z|y=k]` is, so it is the prior's *form* that is tested, not its estimator; (iv) keeps α_a = 0.3, calibrated for a μ_z head, and a head on ρ may want a different one. Atlas rank in the control triple is itself 1/2/3, so rank is a coarse read except where it moves to 6.

**Follow-up opened by arm (i): an α_z ladder.** {α_z/4, α_z/2, α_z, 2α_z} × 3 seeds on the ovarian core at 200/20, full battery. If α_z/2 holds its cycle_z gain with probe and mirror inside the envelope, the per-slide α_z rule (1/mean-count) gets a factor. Runs after the per-dataset `best` fits below.

### One `best` run per dataset (2026-09-21)

**Why.** The κ / d_w / α_w grids have now been read on four slides and land
on the same operating point everywhere: κ 0.1, d_w 6, α_w 0.1, α_a 0.3,
type-only attention sources, 500 epochs / patience 40, α_z = 1/mean-count
per slide. Every experiment from here on should be read on one named run per
dataset, so nobody has to remember that the ovarian pin is an "ablation" and
the FF pin is a "graphclust" reference. `runs/best` is that name.

| dataset | `runs/best` | how |
|---|---|---|
| ovarian FFPE | → `ablation_gat_type_only_s1` | symlink; seed 1 is the run every 2026-09 read (transport, atlas, x̃, cycle, external criteria) was made on, so it stays the one |
| ovary FF | → `reference_graphclust` | symlink; already at the operating point (type_only, 500/40, seed 0) |
| GSE315411 solo core | → `reference` | symlink; already at the operating point; `crossslide/` holds the held-out dual section |
| lung FFPE | new fit `best` | the old `reference_graphclust` predates type-only sources; refit seed 0 at the operating point, then validate / atlas / transport (both reads) / report |

Queue `scripts/queue_2026-09-21_best.sh` (GPU 1, detached, idempotent, logs
under `scripts/logs/best_2026-09-21/`). Ovarian seeds 0 and 2 of the same
configuration remain the envelope triple. **Open question carried from 6b.5:**
arm (i) showed α_z/2 raises cycle_z by ~0.08 on ovarian with guards intact;
if the α_z ladder confirms it, `best` is re-pinned on all four slides.

### Transport at the distribution level, second round (results, 2026-09-21)

*Delegated (Opus). `transport.py` (+`scores_model`, `twin_scores`, `hvg_mask`, `--read twins`, `--hvg N`), `report.py` (Read A / Read B blocks), 4 new tests (18 transport tests pass). Runs: ovarian `best` (kmeans and tumour-band niches), FF `best` (kmeans only; tumour bands need a tumour annotation the FF slide lacks). Count-level numbers regenerated bit-identical in every summary.*

**Read A, model-vs-model.** Median gap closed rises from 0.45 / 0.59 / 0.44 (count-matched; ovarian kmeans / ovarian band / FF) to **0.82 / 0.87 / 0.76** pairwise and 0.81 / 0.83 / 0.73 leave-one-out; improved and CI≠0 on every panel. **The type-mean predictor collapses**: −0.24 / −0.42 / −0.08 pairwise and ≈ −1 leave-one-out, beaten by the transported cloud on 158/158, 79/79, 260/264 panels. HVG-1000 companion moves gap closed by +0.02 and lifts type-mean towards 0; misses only in the two smallest FF HVG panels (n 110–144).

**Read B, matched twins.** Matched transported twin closer than a random same-type source cell in **501/501 panels** (median margin 0.48–0.61), closer than its untransported self in every panel, and at the floor (median 0.08–0.12 vs floor 0.08–0.11). Twin panels with gap ≤ 0 (12–22 per tier) are all degenerate denominators: the within-target z-NN distance exceeds the niche difference for that type, so there is nothing to remove; transported still beats both references there.

**Caveat that limits what these two reads say (main session's reading of the agent's deviation 1).** The target cells were decoded at the *niche-group mean* context and leak, the same `w_of[(niche, type)]` used for the transported source cells, not at their own posterior w. Both sides therefore share w exactly and differ only through z. Read A then measures whether the source type's z-cloud, decoded at the target's group w, matches the target's z-cloud decoded at the same w: a niche-invariance-of-z read plus the direction of the group-level w shift, not a test against what the target cells actually are. The type-mean collapse says the decoded within-niche spread from z is real, which is true and useful, but it was partly guaranteed by the construction. Read B's second clause (matched ≫ random) likewise follows from decoding being continuous in z once w is shared; its first clause (transported twin beats untransported twin) is the informative one and holds everywhere. **Fix, pre-registered here:** rerun both reads with the target side decoded at each target cell's own posterior μ_w and its own real context, so the target carries measured response, and report beside. If gap closed and twin margin survive, the reads stand; if they fall to the count-level numbers, the "spread" limit is shot noise after all and the model-side version was circular.

Paths: `runs/best/transport/transport{,_tumour-band}_{distribution,twins}.{json,png}` on both datasets; report sections "Read A — model-vs-model MMD", "Read B — matched twins".

### Phase change: model frozen; validation and analysis begin (author, 2026-09-21)

**Decision.** Model development stops here. What is frozen: the architecture
(z encoder on own counts + type; w posterior around `m_ψ(c,t)`; c from
type-only GATv2 ⊕ KRONOS Φ ⊕ isolated flag; decoder `a(z)+Bw` softmax; leak
mixture with κ and the shared-wall β), the objective (two-bound loss with the
`(1+ω)` factor, adversary α_a 0.3 on μ_z, α_w 0.1), the operating point
(κ 0.1, d_w 6, 500/40) and the rejected alternatives (x̃ input, L2 on w,
per-cell κ, class-mean prior, adversary on x̂, DAPI cycle labels,
depth-transformed cycle target). The one weight still allowed to move is
α_z: 6b.5 showed α_z/2 raises cycle_z with guards intact on ovarian, so the
ladder runs first and, if it wins on the pre-registered rule, `best` is
re-pinned on every slide before anything else runs. After that the only
changes to `discell/model` are evaluation code.

**Roles of the datasets from here.** Primary for validation and analysis:
GSE315411 (train the solo section, evaluate on the dual section — the only
genuinely out-of-sample read) and the fresh-frozen ovary (highest counts,
most cells, so the tightest estimates). Secondary, to show the same picture
elsewhere: ovarian FFPE (the development slide) and lung FFPE. Three seeds of
`best` on every dataset.

**Overnight programme (author away 2026-09-22).** Two detached, idempotent
lanes; every failed leg is recorded as a finding with its log, not retried
by hand.

*Lane 1 (DisCell, GPU 0):* (1) α_z ladder {¼, ½, 1, 2} × 3 seeds on the
ovarian core, 200/20, full battery; decision rule below. (2) 6a.6, transport
reads A and B with the target side decoded at each cell's own posterior μ_w
and real context, on ovarian and FF `best`. (3) `best` seeds 1 and 2 on GSE,
FF and lung (ovarian has its triple), each with validation, degeneracy,
atlas, transport (all reads), cross-slide on GSE, report. (4) Aggregate
per-dataset envelope tables.

*α_z decision rule (pre-registered):* a rung beats the control (α_z = 1/mean
count) if, over 3 seeds, mean pooled cycle_z is higher by more than the
control's seed range, AND probe ΔCE stays within its floor band, AND mirror
R² ≤ control max + 0.01, AND NMI ≥ control min − 0.01, AND cycle_w ≤ 0.02,
AND I(z;t)/H(t) within 0.05 of control. If exactly one rung qualifies, or
several and ½ is among them, the factor is adopted and every `best` fit in
(3) plus a refit of the four seed-0 runs uses it. Otherwise α_z stays.

*Lane 2 (baselines, GPU 1 after the lung `best` fit):* export all four
datasets with our split and graph; resolVI on all four; SIMVI and MintFlow on
GSE (solo train, dual evaluate) and FF; a shared evaluation of every baseline
latent on our battery (NMI, probe ΔCE vs floor, pooled cycle R², mirror R²,
degeneracy pair, recon where the tool decodes) so the comparison table has
one column per method. MintFlow on FF may not fit in memory; if so the log is
the finding and the run moves to another machine later.

**What "done" looks like on return:** one table per dataset with DisCell
(3 seeds) beside each baseline on the same battery; the GSE held-out column;
transport reads A/B with the honest target; the α_z verdict; every failure
listed with its log path.

### Lane 2 launched: baselines on all data (hand-off, 2026-09-21 evening)

*Delegated (Opus). Detached PID 1743182, log `scripts/logs/lane2_2026-09-22/queue.log`; the agent's full report is at `scripts/logs/lane2_2026-09-22/AGENT_REPORT.md`.* Built: `discell/experiments/baseline_battery.py` (one code path for DisCell and every baseline: NMI, probe ΔCE + floor, mirror R² against DisCell's c, pooled cycle R² of intrinsic and spatial latent, degeneracy pair, held-out recon per count with the type-profile line; 5 tests pass), runners for resolVI / SIMVI / MintFlow in `DisCell-baselines/`, exports of all five bundles under each `runs/best` configuration (FF is 1.16 M cells / 15.6 GB, not the ~700 k quoted earlier). DisCell column reproduces `metrics.json` on GSE `best` within eval-subsample noise; GSE held-out column already on disk (NMI 0.598, probe at floor, mirror 0.042, cycle_z 0.508, cycle_spatial 0.005).

Queue order (~25 h): DisCell columns → resolVI GSE + dual transfer → resolVI FF → SIMVI GSE / dual / FF 200k window → MintFlow GSE / dual / FF (wall-clock caps 5/3/6 h, truncated legs still write latents) → resolVI ovarian + lung → DisCell `best_s1/s2` columns if lane 1 has produced them. Findings already: SIMVI cannot transfer a fit (dual column is fit-on-section) and densifies the matrix (FF capped to a 200k-cell disc); resolVI and MintFlow use their own graphs; resolVI's second block is a mixture proportion, not a spatial latent. Stated asymmetry: baselines are fit on the whole slide as their papers do, so they see held-out counts unlabelled; a DisCell win is conservative. MintFlow's `predict()`+write step is the one not yet smoke-confirmed; check `smoke_mintflow.log` for DONE.

### Lane 1 launched: α_z ladder, honest-target transport, seed triples, envelopes (hand-off, 2026-09-21 evening)

*Delegated (Opus). Detached PID 1739280, log `scripts/logs/lane1_2026-09-22/queue.log`; full report `scripts/logs/lane1_2026-09-22/AGENT_REPORT.md`.* Built: `sweep.py --param alpha_z`; `transport.py --target-side {group,both}` (default both) with `collect_own_p` decoding each target cell at its own posterior μ_z, μ_w, context and influx, keys `*_own*` beside the group-w keys; report tables show both; `scripts/alpha_z_decision.py` (rule as pre-registered, with the open clauses fixed before reading: guards on the worst seed, probe within its own floor on every seed); `scripts/envelope_tables.py`; 27 transport/sweep tests pass.

Deviations recorded: the α_z control is `sweep3_k0.1_s{0,1,2}` (same point and budget), not refitted; the existing group-w keys reproduce within GPU float32 nondeterminism (max 1e-4 on secondary keys, all summaries to six decimals), not bit for bit. The earlier Read A prose had claimed the target was the cells' own posterior; corrected. Queue: ladder (3 rungs × 3 seeds, ~1 h) → decision → 6a.6 on ovarian (both niche sources) and FF → seed-0 refits and re-pin only if factor ≠ 1 → `best_s1/s2` on GSE, FF, lung (+ ovarian if moved) → per-seed battery, atlas cross-seed, transport, GSE cross-slide, report → envelope tables. Control as read by the rule: cycle_z 0.450, NMI min 0.636, mirror max 0.050, probe 0.006 vs floor −0.056, I(z;t)/H(t) 0.79.

Indicative own-target numbers from the smoke run (reduced settings, replaced by step 2): Read A median gap closed 0.72 own vs 0.89 group, type-mean −0.98; twin margin 0.50 own vs 0.59 group. The honest target costs something and collapses neither read.

**Lane 2 post-launch (17:20).** MintFlow runner now smoke-confirmed end to end (all three tools verified). First failure: the FF DisCell battery column OOMed (two Trainers over the 1.16 M-cell slide on one card); `baseline_battery.py` now reuses the loaded Trainer and frees tiles between sections. Repair running detached (PID 1746441, `scripts/logs/lane2_2026-09-22/repair_ff_column.log`, REPAIR_OK/FAILED verdict); GSE, dual, ovarian and lung columns unaffected. resolVI GSE fit running since 16:45.
FF repair REPAIR_OK (16:50): all five DisCell battery columns on disk, no FAILED markers. FF `best` on the shared battery: NMI 0.598, probe +0.002 (floor −0.015), mirror 0.056 (permuted 0.017), cycle_z 0.766 (spatial latent 0.002), I(z;t)/H(t) 0.818, recon −7.314 vs type-profile −7.375.

### Overnight lanes: results (2026-09-23, read on return)

**Lane 1 finished 00:05 on 2026-09-22, one failure (envelope tables, script bug, being repaired). Lane 2 finished 01:55, SIMVI and MintFlow legs failed (below).**

**α_z ladder: factor ½ adopted, `best` re-pinned on all four slides.** Control (α_z = 1/mean count, `sweep3_k0.1_s{0,1,2}`): cycle_z 0.450 (range 0.057), NMI 0.659, mirror 0.048, recon −7.256. Rungs, 3-seed means: ¼ → cycle_z 0.533, NMI 0.667, mirror 0.040, recon −7.195; ½ → cycle_z 0.527, NMI 0.663, mirror 0.040, recon −7.218; 2 → cycle_z 0.293, NMI 0.630, recon −7.289. Probe at its floor on every seed of every rung; cycle_w ≤ 0.009; I(z;t)/H(t) 0.76–0.80. ¼ and ½ both qualify on all six clauses; the rule adopts ½ when several qualify. Note the monotone direction: less KL pressure on z gives more continuous content and a better likelihood without the guards moving; ¼ is not worse than ½ on any read, so the ladder has not found the bottom. Recorded, not chased: α_z is frozen at ½ × 1/mean-count for the paper. New runs: `best_az0.5_s0` → `best`/`best_s0`, `best_s1`, `best_s2` on all four datasets, each with validation, degeneracy, atlas, transport (all reads, own-target included), GSE cross-slide, report. Old pins kept (`ablation_gat_type_only_s1`, `reference_graphclust`, `reference`, lung `best_pre_az0.5`).

**6a.6, honest target (target decoded at its own posterior μ_z, μ_w, context, influx), on the OLD pins.** Read A median gap closed pairwise / leave-one-out: ovarian kmeans 0.68 / 0.68 (group-w 0.82 / 0.81; count-matched 0.45 / 0.30); ovarian tumour band 0.83 / 0.61 (0.87 / 0.82); FF 0.72 / 0.65 (0.76 / 0.72). Type-mean predictor −0.16 to −0.49 pairwise, −1.0 leave-one-out; transported beats it on 157–264 of 158–264 panels. HVG-1000 within ±0.04. Read B twin margin own: 0.48 / 0.55 (ovarian), 0.53 / 0.55 (band), 0.45 / 0.48 (FF), against group-w 0.54–0.61. **Verdict: the honest target costs 0.04–0.14 of gap closed and 0.03–0.06 of twin margin; both reads stand, the type-mean collapse stands, so the "spread" was real and not the shared-w construction.** These are the numbers to quote, re-read on the new `best` triple by the lane (in each run's `transport/`).

**Lane 2, baselines.** resolVI ran on all five bundles (GSE solo 15 min, dual by transfer, FF 2.2 h, ovarian 38 min, lung 26 min). MintFlow GSE solo reached **5 of 50 epochs in the 5 h cap** (truncated, latents written); its dual leg skipped; MintFlow FF killed by host OOM after 10 min. **SIMVI failed on all three legs within seconds: CUDA OOM at 22 GB in the ZINB log-prob** (densified matrix, full-graph forward per minibatch). Repairs delegated 2026-09-23 morning; findings stand if they cannot be made to fit.

**Baseline battery, first read (resolVI columns; DisCell/best column is the OLD pin and is being recomputed — the `best_s1/s2` columns are the new α_z/2 runs).** Same picture on all five tables: resolVI's intrinsic latent has higher NMI (0.68–0.78 vs 0.60–0.67) and higher I(z;t)/H(t) (0.86–0.98 vs 0.77–0.85) but **fails the invariance guards everywhere**: probe ΔCE +0.02 to +0.22 against floors of −0.01 to −0.06 and mirror R² 0.13–0.21 against our 0.03–0.06 (permuted 0.014). Its within-type variance fraction is 0.20–0.26 against our 0.54–0.76: resolVI's z is close to a type label with little continuous content, and what continuous content it has carries the niche. Cycle R² of the intrinsic latent is comparable (GSE 0.50 vs 0.48–0.50; FF 0.68 vs 0.76–0.79; ovarian 0.50 vs 0.52; lung 0.35 vs 0.46). Its second block is a mixture proportion, not a spatial latent (spatial var fraction 0). MintFlow GSE (truncated): NMI 0.90, I(z;t)/H(t) 0.999, within-type var 0.007, cycle_z 0.035 — a 100-d intrinsic latent that is a type code and nothing else after 5 epochs; not a fair read of the method. On the GSE held-out section DisCell holds: NMI 0.598, probe at floor, mirror 0.042, cycle_z 0.508.

**Repairs, 2026-09-23 morning (Sonnet agent).** Envelope tables: `atlas.json` cross-seed `axis_cosine` lists contain literal nulls for some seed pairs (ovarian s2, FF s1/s2, GSE s1/s2); `scripts/envelope_tables.py` now skips them and footnotes which runs. Baseline tables: the `DisCell/best` column was the pre-re-pin run (α_z 0.007, seed 1) and has been replaced by `DisCell/best_s0`; resolVI/MintFlow columns byte-identical. Tables at `data/datasets/<ds>/experiments/{envelope_table,baseline_battery}.md`, combined `scratchpad/envelope_tables_all.md`.

**Envelope of the α_z/2 `best` triple (mean over 3 seeds; held-out GSE section in brackets).** Recon: ovarian −7.186, FF −7.316, lung −7.246, GSE −7.205 [−7.223]. NMI 0.656 / 0.630 / 0.665 / 0.638 [0.606]. Probe ΔCE within floor on every run; mirror 0.039 / 0.058 / 0.027 / 0.025 [0.041]. cycle_z pooled 0.521 / 0.783 / 0.463 / 0.445 [0.506], cycle_w ≤ 0.008 everywhere. I(z;t)/H(t) 0.81 / 0.84 / 0.74 / 0.77; within-type variance fraction 0.78 / 0.57 / 0.70 / 0.75. Read A own-target gap closed 0.674 / 0.591 / 0.721 / 0.553, type-mean −0.62 to −1.0; twin margin own 0.42 / 0.46 / 0.41 / 0.47. Atlas effective rank 1–3; cross-seed axis-1 cosine 0.96 (ovarian), 0.97 (lung), 0.69 (FF), 0.76 (GSE) — **the FF and GSE atlases are less seed-stable than the FFPE slides; watch item.** GSE mean-read transport has no trusted-tier panels (ceiling threshold not met at 69k cells): the mean read is not quotable there, the distribution reads are. FF cycle linear reference (0.854) exceeds cycle_z (0.783): at FF depth a linear read of the counts explains more cycle variance than z does — expected (z is 20-d and adversarially constrained) but must be stated when FF cycle numbers are quoted.

**Baseline battery, corrected DisCell column (best_s0, s1, s2 vs resolVI), the same story on all five tables.** resolVI: NMI +0.07 to +0.11 over us, I(z;t)/H(t) 0.86–0.98, within-type variance 0.20–0.26, probe ΔCE +0.02 to +0.22 above floor, mirror 0.13–0.21; its intrinsic latent is type-like and niche-leaky. DisCell: probe at floor on 14 of 15 columns (FF s2 +0.013 vs floor −0.013, ovarian s0 −0.027 below), mirror 0.025–0.060, within-type variance 0.54–0.77. Cycle R² of the intrinsic latent: GSE 0.50 vs 0.48–0.50, dual 0.49 vs 0.48, FF 0.68 vs 0.76–0.79, ovarian 0.50 vs 0.50–0.52, lung 0.35 vs 0.45–0.46. Held-out reconstruction per count: DisCell better by 0.01–0.10 nats on every table (GSE −7.206 vs −7.220; dual −7.221 vs −7.229; FF −7.312 vs −7.322; ovarian −7.217 vs −7.253; lung −7.234 vs −7.286). MintFlow GSE (5/50 epochs) recon −5.035 is on a different scale (its own likelihood), not comparable as written; flagged for the repair agent.

**Run archive (2026-09-23).** Ovarian FFPE `runs/` had ~200 directories from five weeks of development; 91 superseded ones moved (not deleted) to `runs/_archive/`: early adversary/weight calibration (`cal*`), debug probes (`probe_*`, `halo_*`, `test_v1`, `reference_k0.1*`), the first two κ sweeps (`sweep_k*`, `sweep2_k*`), single-seed α_w runs and the closed 2.2 triple, and rejected options (`xtilde`, `wpen_*`, `gat_sink`, `gat_type_only_wd*`). Kept: `best*`, `sweep3_*`, `ladder_az_*`, `dw*` (the only ovarian d_w envelope), `abl_*`, `ablation_gat_type_only*` + `reference_best` (type-only vs type-z comparison, old pin), `reference_graphclust`. Other datasets needed no cleaning. Devlog references to archived names remain valid history.

### Article-derived experiments, second pass on the frozen model (motivation, 2026-09-23)

Four legs launched in parallel, all pre-registered in todo §6b and the 2026-09-21 review entry; nothing here changes the model.

*Re-read of 6b.1 / 6b.3 / 6b.4 on the α_z/2 `best` triple* (ovarian, plus FF and GSE where the instrument runs). Wished for: the same verdicts as on the old pin — leak channel separates LR genes while w does not; floor-corrected MIG/MIC in the 0.8 band with z's niche excess ≤ 0.15; w true-axis |τ| ≥ 0.85 vs false ≤ 0.6. Any verdict that flips is a finding about α_z, not a re-tuning opportunity.

*6b.10, GO localisation of B* (Celcomen's takeaway). Rank-based enrichment of each programme's loadings against secreted / membrane / cytoplasmic / nuclear GO cellular-component sets, BH q ≤ 0.05, on the ovarian triple, and its κ-sensitivity across `sweep3_k*`. Wished for: the response programme enriched for secreted and membrane genes, and that enrichment stable across seeds; if it grows with κ the leak channel is feeding B and the atlas has a confound to state.

*6b.8, analytic-posterior planted world* (review B2). An additive two-latent count world with a known leak operator where the exact posterior mean of the intrinsic latent is closed-form; our amortised encoder is trained on it and the gap E[z|x] − μ_z(x) measured directly, as a function of κ mismatch and depth. Wished for: the gap small relative to the posterior width at matched κ and growing smoothly with mismatch. Synthetic only.

*6b.2 curation* (resolVI's takeaway). A Sonnet agent proposes mutually exclusive marker pairs and true-positive control pairs per dataset from public references, with sources; the author approves before anything is scored.

### GO localisation of B (6b.10, results, 2026-09-23)

*Delegated (Opus). `discell/experiments/go_localisation.py` reuses the atlas Mann–Whitney statistic on |loading| over expressed panel genes; sets are GO CC closures (extracellular, plasma membrane, cytoplasm, nucleus) from the current GOA human GAF (2026-05-21) and go-basic (2026-07-26), stored in `data/external/`. BH across sets × columns. 2 planted tests pass. Outputs `experiments/go_localisation.{json,md,png}` on all four datasets.*

**Ovarian FFPE triple, programme 0:** extracellular +0.12 to +0.13 (q ≤ 2e-6), plasma membrane +0.08 to +0.10 (q ≤ 4e-4), cytoplasm ≈ 0, nucleus −0.08 to −0.10 (q ≤ 3e-4). Same sign pattern on all seeds; spread 0.007 on 0.12. The B columns carrying the programme reproduce it, the others are flat. **Lung** +0.14 / +0.12 / −0.06 / −0.18 (q to 1e-19), **GSE** +0.14 / +0.11 / ns / −0.10. **FF ovary does not agree:** extracellular −0.09 on s0, +0.07 on s1, −0.03 on s2; effective ranks 1/2/3; programme 0 carries no significant hallmark either. On the healthy FF ovary there is no stable w programme for this read to be about (consistent with its cross-seed atlas cosine 0.69).

**κ-sensitivity (sweep3, 18 runs):** extracellular effect +0.117 at κ = 0, +0.129 at 0.4; slope +0.023 per unit κ, one seed-sd over the whole grid. Plasma membrane drifts down at high κ on seed 0 only. **The surface bias of B is present with the leak channel off and unchanged by it: the leak channel does not feed B.** Caveat stated: secreted/membrane genes are also the ones segmentation misassigns most, so the enrichment alone would not refute a leak story; the κ-flatness and the κ = 0 level do — the argument MintFlow's signalling-gene criterion lacks.

Verdict: rank-biserial 0.13 is a lean, not a separation: the response programme tends towards the cell surface. Reportable on three slides with FF as the stated exception.

### External criteria re-read on the α_z/2 triple (6b.1 / 6b.3 / 6b.4, results, 2026-09-23)

*Delegated (Opus). Twelve reads, all rc = 0, artefacts `experiments/external_*_best_s*.{json,png}` beside the old-pin ones. Model-free rows (raw observed, ℓ row) identical to the 2026-09-21 values to every digit, so the differences below are the model and nothing else.*

**6b.1 signalling-gene share — same conclusion, cleaner on ovarian.** Leak effect +0.245 / +0.284 / +0.245 (p ≤ 1e-29) on the three seeds against +0.256 before; response +0.041 / +0.041 / +0.032, now not significant at 0.05 on any seed (old pin p = 8.8e-3). FF: leak +0.122 (p 5e-7) vs response +0.080 (p 1e-3) — on FF both channels separate, the leak 1.5× more strongly; the old pin had response ns. "Leak alone does it" is exact on ovarian and becomes "leak does it far more strongly" on FF.

**6b.3 MIG/MIC with within-type floor — holds on ovarian and GSE, falls on FF.** Ovarian kmeans, floor-corrected MIC 0.789 / 0.768 / 0.788 (old 0.79–0.81), MIG +0.144 to +0.149; tumour bands 0.854 / 0.878 / 0.863; z's niche excess 0.12–0.13 vs w's 0.44–0.46; MINE agrees. GSE `best_s0` (no old counterpart): MIC fc 0.821 kNN / 0.713 MINE, MIG fc +0.154 / +0.097. **FF `best_s0`: MIC fc 0.664 / 0.698, MIG fc +0.052 / +0.057, outside the pre-registered 0.8 band (old pin 0.789 / 0.811).** Cause is on the w side: z's excess unchanged (+0.114 vs +0.103), **w's excess down 41 % (+0.226 vs +0.385)**. On the FF slide the α_z/2 run's w carries much less niche information than the old pin's did. One seed; FF `best_s1`/`best_s2` MI reads launched to separate slide from seed (`scripts/logs/rereads_2026-09-23/ff_mi_s12.log`). Consistent with the FF atlas instability (cosine 0.69) and its inconsistent GO read. Per the pre-registration a finding about α_z on FF, not a tuning opportunity.

**6b.4 SIMVI axis test — same conclusion, strengthened on every seed.** w true-axis |τ| 0.946 / 0.997 / 0.948 (old 0.918 / 0.847 / 0.933), false axis 0.53–0.60 flat; false-axis genes at |τ| ≥ 0.9 fall from 128 / 1532 / 736 to 15 / 50 / 231, ratio 609 / 352 / 43. The old weak seed is no longer weak. z row slightly down on s1 (0.696 → 0.609), the expected direction for a less-pressured z that does not absorb the spatial gradient.

### Preprocessing: what the Voronoi face actually measures (validation, 2026-09-23)

*Author question against the AISTATS preprocessing paragraph (App. B.2). No code changed; everything read off the stored bundles and `discell/preprocess/geometry.py`. Scripts and figures in `scratchpad/` (untracked): `face_anatomy.py`, `face_construction.py`, `face_rule.py`, `face_stats.py`, `sparse_limit.py`, `clip_binding.py`.*

**The clip applies to the Voronoi region, not to the segmentation polygon.** `build_voronoi_graph` intersects `owner[k]` with `Point(centroid).buffer(30um)`; the polygon is never clipped. Its only role in f_ij is placing the seed (polygon centre of mass). So the tessellation shapes are not the cell shapes, and a long thin cell and a round cell with the same centroid give the same region.

**f_ij is set by the flanking cells, not by the pair.** The face lies on the perpendicular bisector of i–j, cut at each end where a third cell becomes equidistant, so exactly `f_ij = |circumcentre(i,j,k1) - circumcentre(i,j,k2)|` over the two Delaunay triangles on edge ij. Verified on a 220 um interior patch, 2863 edges: **median discrepancy 9e-10 um**; the 10 failures sit at median radius 218 um in a 220 um patch (truncation artefacts, not real). Correlation of f_ij with the flanking circumcentre offset +0.717, with d_ij +0.098 — the pair's own separation barely matters.

**The face is always defined; contact is not.** pdl018d, 209,862 voronoi edges: face zero on **0.000%** of edges (median 7.21 um), while only **2.7% of those polygon pairs actually touch** (median gap 1.67 um) and `apposed_wall_um` is zero on **55.7%** (ovarian 47.2%). corr(apposed, gap) −0.496 / −0.507 vs corr(face, gap) −0.080 / +0.028 across the two slides. **This is the measured version of the B.2 argument** — a contact-based f would vanish on roughly half of all edges and what signal it carried would track the gap, i.e. density. Quote these rather than asserting it.

**The 30 um clip is nearly inert, and is a hard 60 um neighbourhood radius.** Slide-wide the median Voronoi region reaches only 10.1 um from its seed (p90 18.5, p99 42.1); the disc touches the region of **2.21% of 70,757 cells**, and 0.52% of faces exceed 30 um. In a 260 um interior patch it changed nothing (0/1214 cells, 0/3525 edges). Where it does bite it is exact: an edge survives only while d_ij < 2R, and `f_ij <= 2*sqrt(R^2 - (d_ij/2)^2)`. Checked on **2.2 M edges across GSE, ovarian and lung: zero edges with d_ij >= 60 um, zero faces above the chord ceiling**, max observed d_ij 59.93 / 59.84 / 59.88. **Consequence to state: the model's neighbourhood radius is 60 um set by the clip, and `DEFAULT_MAX_EDGE_UM = 100` never binds.** Sparsity inflates faces until the disc takes over (dense 6th-NN 10 um → median face 3.6 um; sparse 24 um → 14.1 um), then truncates them, then drops the edge; only 0.03% of cells end up isolated, median voronoi degree 6.

**Two corrections for the draft.** (1) The paragraph says clipping bounds faces; it also *deletes edges* (`length <= 0` is dropped), so 30 um is an adjacency cutoff too and any quoted mean degree is a function of it. (2) "Clipping each Voronoi cell" reads as a routine step applied everywhere — say it binds on ~2% of cells, at borders and gaps.

**Watch items, not acted on.** `apposed_wall_um` is in `DEFAULT_EDGE_METRICS` and nothing overrides it, but `prepare.from_dataset` reads only `shared_wall_um` and `centroid_dist_um`, so no polygon-contact quantity reaches the model — the B.2 claim holds today and is one refactor from silently failing. Sliver faces: 2.94% of edges under 0.5 um, 0.58% under 0.1 um, against a median of 7.2 um; harmless as weights, but phantom neighbours for any instrument that counts `voronoi_degree` unweighted rather than using the beta weights.

**Figures wanted for the appendix** (author's call, 2026-09-23): `scratchpad/face_anatomy.png` (polygons vs clipped regions vs the face; one edge zoomed with face, apposed wall and d_ij side by side; a border cell where the disc bites, 26% area cut) and `scratchpad/face_construction.png` (the polygons-discarded / bisector / cut-bisector construction, a dense and a sparse neighbourhood at one scale, and all 209,862 edges under the chord ceiling). Both need moving out of `scratchpad/` before they can be cited.

**FF MI follow-up on seeds 1 and 2 (2026-09-23): the FF drop is a dead context channel on one seed, not a slide effect.** Floor-corrected MIC kNN / MINE: s0 0.664 / 0.698, s1 0.758 / 0.830, **s2 0.000 / 0.001 with I(niche; w) = 0 exactly.** On `best_s2` the w channel never opened: `kl_w_per_dim` is 0.0000 on every dimension from epoch 4 to the end (s0 sits at 0.0003–0.008 throughout); the baseline battery's spatial-variance fraction for that run is undefined (w constant across cells). The prior network m_ψ(c,t) is ignoring c on that seed, so w = per-type constant and the model is a type-conditioned intrinsic autoencoder with a leak term. Its other reads did not flag it: NMI 0.640, probe at floor, cycle_z 0.79, recon −7.326 (the best of the three seeds), and transport Read A still closes 0.47 of the gap — **which means on FF the leak term alone accounts for roughly half of the between-niche shift**, a number worth keeping. Seed 1 is inside the band; seed 0 is below it (w excess 0.226) but alive. GSE `best_s2` shows a low spatial-variance fraction (0.038 vs 0.11 / 0.15) — its MI read is running to check for the same failure.

**Consequence for the freeze.** A dead w channel is a failed fit, like a diverged run, not a property of the model: it is detectable at epoch 4 from KL_w ≡ 0 and I(niche; w) = 0 on held-out cells. Proposed handling (evaluation code only): add `I(niche; w) > 0` / non-zero w variance across cells as a degeneracy guard in the battery and the sweep report, list `best_s2` on FF as a failed fit, and refit FF seed 3 to complete the triple. Awaiting the author's word before touching the pinned triple. The old FF pin and the sweep3 FF runs all have KL_w > 0 on at least one dimension, so this is the first observed instance on FF.

**Paper.** Method §latents sentence changed to "the response w carries how cells of that type shift with their surroundings" (was "what its environment did to it"); reason recorded in `revision_2026-09-17.md`. Build clean, 20 pages.

### Analytic-posterior planted world (6b.8, results, 2026-09-23)

*Delegated (Opus). `discell/experiments/planted_posterior.py`, 4 tests (posterior quadrature vs 4e6-draw importance sample). 45 fits, GPU 0, 64 min. `planted.fit_synthetic` gained `d_z`/`kappa` overrides with defaults unchanged (x̃-gate tests still pass).*

**Design.** DisCell's own generative model with w off: z ~ N(m_t, 0.5² I₂), ρ = softmax(zA), leak from the fit's own graph, multinomial counts; 3000 cells, 40 genes, κ_world 0.2. With ρ̄, κ, A known the posterior of z is 2-d and integrated exactly on a grid (boundary mass ≤ 1e-7). Exact numerically, not closed-form (multinomial–softmax is not conjugate), and it conditions on ρ̄, which the encoder is denied by design. μ_z is aligned to the truth by 5-fold cross-fitted maps: strict affine and a small MLP ("warped"); since `dec_a` is an MLP the warped gauge is the right one, and the affine reading is misleading (it *worsens* with training, 1.95 → 2.40 sd, while warped improves 0.88 → 0.82).

**Gap in posterior sd (warped, 3 seeds), κ mismatch × depth:** matched κ 0.55 / 0.83 / 1.48 at depth 100 / 300 / 1000; ±0.05 adds 0–5 %, ±0.10 adds 6–11 %; shallow U at every depth, minimised at matched κ (tie within 0.03 % at depth 1000). In the world's own z units the gap is flat at 0.13–0.14 across every cell: **the growth with depth is the posterior narrowing (sd 0.175 → 0.061), not the encoder degrading.** References: supervised MLP on the encoder's own inputs 0.61 / 0.95 / 1.61 (DisCell beats it at every depth); MLP given ρ̄ as well 0.53 / 0.80 / 1.35; linear oracle 1.40 / 2.50 / 4.53; prior mean 2.4 / 4.1 / 7.4.

**Verdict.** Grows smoothly with mismatch: yes. Small at matched κ: qualified yes — below a directly supervised amortiser of the same inputs, close to the ρ̄-informed one, far above the prior. Caveat: the oracles are regression references at finite capacity, not information-theoretic ceilings, so the safe claim is comparative. Outputs `data/datasets/synthetic_smoke/experiments/planted_posterior.{json,md,png}`.
**GSE seeds 1 and 2 MI check:** floor-corrected MIC 0.850 / 0.804, I(niche; w) 1.02 / 0.88, w excess 0.48 / 0.39 — the context channel is alive on all three GSE seeds; the low spatial-variance fraction on s2 was not a collapse. The dead channel is FF `best_s2` only.

### Dead context channel: guard and refit (motivation, 2026-09-23)

**Decision (author).** FF `best_s2` is a failed fit, not a model property; refit seed 3. No change to the objective (KL annealing on w would be the textbook remedy for a collapsed conditional prior, and it is exactly the tuning the freeze rules out). Protection is detection: (1) training-time detector — summed KL_w < 1e-5 in the first 20 epochs logs a warning and sets `dead_w_channel` in `metrics.json` (healthy runs sit at 3e-4 to 8e-3 at epoch 4, two orders of margin); (2) battery guard — I(niche; w) on held-out cells with a within-type permutation floor, plus the variance fraction of μ_w across cells, in the degeneracy block and sweep report; zero excess = "failed fit: dead context channel"; (3) queue rule — a flagged fit is refit with the next seed and the rate recorded (1 in 12 so far). Wished for: the guard flags exactly FF `best_s2` among the twelve `best_s*` runs; seed 3 opens the channel and lands inside the FF envelope (recon −7.31 to −7.33, NMI 0.62 to 0.64, cycle_z 0.76 to 0.80). If seed 3 also collapses the FF slide has a systematic problem with this prior and that becomes the finding.

### Collapse of the context channel: survey and a robustness fix (motivation, 2026-09-23)

**Survey.** Every run on disk with a KL_w history (329): 12 end with a dead context channel (summed KL_w < 1e-5), none recovered after dying early. FF 9 / 57 — **all three seeds at α_w 0.3, all three at 0.2, and 3 of 12 at the pinned α_w 0.1** (`best_s2`, `sweep3_k0.2_s1`, `sweep3_k0.05_s2`); ovarian 2 / 79 (one α_w 0.3 seed, one adversary-on-x̂ arm); lung 1 / 58 (α_w 0.3); GSE 0 / 57. The 2026-09-21 sweep note read "KL_w → 0 as α_w rises" as the channel closing softly; on FF it is a hard death, and the operating point sits at the edge of it on the headline slide. Detection (the guard now being built) catches it; it does not remove it.

**Decision (author, 2026-09-23).** Unfreeze this one thing: test the textbook remedies for a collapsed conditional prior, KL warm-up on w and free bits on w, alone and together. Robust beats quick; if the fix is adopted every pinned run is redone.

**Arms.** (a) control, α_w constant; (b) warm-up: α_w linearly from 0 to its target over the first 30 epochs, constant after — the converged objective is unchanged; (c) free bits: per-dimension KL_w charged only above λ = 0.05 nats (`max(KL_d − λ, 0)`), stationary objective, changed; (d) both. Every other weight and the architecture unchanged. **Where the channel dies:** FF at α_w 0.1 (rate 1/4) and α_w 0.3 (rate 3/3), 200/20, 4 seeds each → 4 arms × 2 α_w × 4 seeds = 32 FF fits. **Where it does not:** ovarian core at α_w 0.1, 3 seeds per arm, to see what the fix costs when nothing is broken.

**Evaluated by.** Collapse count per arm (KL_w at epoch 20 and end; I(niche; w) guard); the battery guards (probe at floor, mirror ≤ envelope, NMI, cycle_w ≤ 0.02, I(z;t)/H(t)); cycle_z; recon; KL_w level; transport Read A own-target gap closed on the ovarian runs.

**Decision rule.** An arm qualifies if it has **zero collapses at both FF α_w levels** and its ovarian guards and recon sit inside the control's seed envelope. Among qualifying arms prefer (b) warm-up alone, because the converged objective is identical to the pinned one and the paper's loss stays as written; take (c) or (d) only if (b) does not qualify. If no arm qualifies, fall back to per-slide α_w for FF (0.05, inside the grid) and report the collapse rate as a limitation. What is wished for: (b) qualifies, and on ovarian its numbers are indistinguishable from control.

### What the context should know about the centre cell: three query/prior ablations (motivation, 2026-09-23)

**Question (author).** Type enters the context twice: as the GAT query `embed(t_i)` and as an input to the prior `m_ψ(c_i, t_i)`. Should the environment be summarised without reference to the cell it surrounds, so the prior is "before knowing anything about the cell except its environment"? And could the query be the image instead, so that which neighbours matter depends on local tissue architecture rather than on the centre cell's label?

**Analysis before running.** Type in the query makes c depend on the centre cell (a hidden dependence the paper has to explain); removing it makes c a pure environment summary, and the type×environment interaction can still be learned downstream in m_ψ — expected cost small. Type in the prior is different: with α_w keeping the posterior on the prior, a type-free prior makes w identical for every type in a neighbourhood, and with one shared B every type receives the same expression shift. The atlas (effective rank 1–3) says B already carries one shared direction and type sets its amplitude and sign per type through m_ψ; a type-free prior keeps the direction and loses the per-type amplitude, which is exactly what the transport read scores. Φ is ego-masked, so an image query says nothing about the cell; it already enters c by concatenation, so the arm tests its value as a query specifically.

**Arms (each a default-off flag; pinned model bit-identical when off).** (i) `--query type_free`: constant learned query vector (no t). (ii) `--query image`: query = linear map of Φ_i (no t). (iii) `--prior-type-free`: m_ψ(c) only, t removed from the prior; posterior q(w|·) unchanged. Ovarian core, 3 seeds, 200/20, at whatever objective the collapse-fix test selects (queued behind it). **Evaluated by** the battery guards, cycle_z / cycle_w, transport mean read (fraction of ceiling, beats-both) and Read A / Read B own-target, atlas effective rank and cross-seed cosine, MIG/MIC with floor, and — for (ii) — whether attention weights vary with histology in a way that can be read (entropy of α_ij by type, correlation of α with Φ principal components). **Wished for / decision.** (i) indistinguishable from control → adopt (cleaner description at no cost). (iii) expected to lose transport per-type amplitude → stays an ablation; if it does not lose, w's definition in the paper changes to a universal response. (ii) is exploratory: adopt only if it beats control on transport with guards intact and the attention read is interpretable; otherwise report.

**8.10 built and queued (2026-09-23).** `--query {type,type_free,image}` and `--prior-type-free` (drop-in `TypeFreePrior` at `prior_w`, same pattern as `ClassMeanPrior`), `attention_read.py` (per-type attention entropy, α vs top-5 Φ PCs), 9 planted tests; 78 tests green across ablations/networks/train/transport/degeneracy. Queue `scripts/queue_2026-09-23_queryprior.sh` (PID 2480309) waits on the collapse-fix `DECISION.json`, then control + 3 arms × 3 seeds on ovarian with battery, transport (both niche sources), MIG/MIC, attention read, atlas with cross-seed comparison; read-out `scripts/queryprior_table.py` → `experiments/queryprior.{json,md}`. Known gap carried: `sweep.py` and `report.py` rebuild `DisCell` without the 6b.5 / 8.10 fields, so they cannot reload ablation runs (only `validate.load_run` is complete) — fix before any ablation run needs a report.

**8.9 built (2026-09-23).** `--w-warmup-epochs N` (α_w · min(1, epoch/N) on the KL_w term only, `alpha_w_eff` logged per epoch) and `--w-free-bits λ` (batch-mean per-dimension KL charged as max(KL_d − λ, 0), Kingma's form; raw KL_w still logged so the dead-channel guard reads the same quantity). `gaussian_kl` refactored through `gaussian_kl_per_dim`, bit-identical sum. 13 planted tests; full suite 332 passed. Grid queued (`scripts/queue_2026-09-23_wcollapse.sh`, read-out `scripts/wcollapse_table.py` → `DECISION.json`). **Memory correction:** an FF 200/20 fit is ~17 GB resident, an ovarian one > 6 GB, so FF fits need nearly a whole card; the queue gates on 18.5 / 9.5 GB free and takes whichever GPU is free (lane assignment dropped). No fit had started at hand-off because both GPUs were held by the seed-3 refit and the SIMVI repair.

**8.9 correction before any warm-up fit counts (2026-09-23).** During warm-up α_w ≈ 0, so the objective is not the pinned one: reconstruction is best there and worsens as α_w ramps in. Two consequences the agent flagged and I ordered fixed before reading: checkpoint selection must ignore epochs < N (no `best` from the warm-up era) and the early-stopping stale counter must not start until epoch ≥ N. Both gated on `w_warmup_epochs > 0`, planted tests, any warm-up fit made before the fix is invalidated and redone; `best.epoch` is reported per arm so a reader can see the warm-up arms were scored after re-convergence. The control fit already finished reproduces the pinned ovarian operating point (recon −7.224, NMI 0.670, mirror 0.040, cycle_z 0.505), so the baseline of the grid is sound.

**8.9 first FF read (2026-09-23 afternoon).** FF α_w 0.3 seed 0: control dead (KL_w 2e-11, I(niche;w) = 0, variance fraction 0). Warm-up arm: alive at both probes, end KL_w 2.0e-5, **I(niche;w) 1.17 with excess +0.49 over the within-type floor and variance fraction 0.36** — a working context channel, in the healthy ovarian band (control excess +0.44) — with better recon (−7.310 vs −7.312) and NMI (0.665 vs 0.646) than the control. One seed of four; free-bits arm next. Ovarian two seeds per arm: no collapses, recon identical to three decimals, NMI control 0.660 / free bits 0.671 / warm-up 0.629 / both 0.645.

**Interpretive correction, recorded for the paper.** KL_w is the divergence of q(w|·) from the *conditional* prior m_ψ(c,t); a posterior tracking an accurate context-varying prior field gives a small KL while w still varies strongly with niche, because the prior carries the variation. So a low KL_w is not a weak channel, and the summed-KL_w detector and the I(niche;w) guard can disagree in this thin-but-live regime; the guard is the meaningful read. A dead channel is the case where prior and posterior both ignore c: MI, floor and variance all exactly zero. Any sentence that reads low KL_w as a closed channel is wrong, including the 2026-09-21 sweep note "the w channel closes entirely (KL_w → 0)" — that note should be re-read as "the posterior sits on the prior", which is the intended operating regime.

**8.9 rule amendment (2026-09-23, made after FF α_w 0.3 seed 0 and before any other FF seed; disclosed as such).** Seed 0 at α_w 0.3: control dead by epoch 14; warm-up alive at both probes but decaying under full pressure to an end KL_w of 2.0e-5 (2× the threshold), guard healthy; free bits alive with KL_w rising at epoch 24 (1.1e-1), four orders above control, objective stationary so nothing to "survive after the ramp". The registered tie-break (prefer warm-up when both qualify) would reward the arm that ends nearest death. Amendment, per the author's standing instruction that robust beats quick: **when several arms qualify, prefer the arm whose minimum end-of-run KL_w over all FF seeds is farthest above the death threshold (log scale), provided its ovarian guards and recon are inside the control envelope; warm-up alone is preferred only if its margin is within one order of magnitude of the best.** Everything else in the rule stands. The margin is reported per arm and per seed regardless of who wins.
The read-out reports **both** verdicts, the original pre-registered rule's and the amended rule's, with an agreement flag; if they disagree that disagreement is a finding to be stated in the paper, and the amendment is labelled as such with its timing, never as pre-registered. The downstream queue (8.10) uses the amended verdict.

### Dead context channel: detector, guard and FF seed 3 (8.8, results, 2026-09-23)

*Delegated (Opus). `train.py`: `dead_w_channel` flag (ΣKL_w < 1e-5 within the first 20 epochs, WARNING logged, fit continues), written top-level in `metrics.json`. `degeneracy.py`: `w_channel_guard` — I(niche; μ_w) by kNN with a 5-fold within-type permutation floor (estimator and niches reused from `external_criteria` / `validate`) and the across-cell variance fraction of w; verdict "failed fit: dead context channel" when excess ≤ 0; rendered in `report.py` and merged into `sweep.py` rows. 24 tests pass. GPU 0 was full (collapse grid), so the guard sweep and the seed-3 battery ran on CPU.*

**Guard over the twelve pinned runs flags exactly FF `best_s2`** (I(niche;w) 0.000, floor 0.000, variance fraction 0.000). Live runs: excess +0.24 to +0.53 (floor sd 0.002–0.006, so ~50× noise), variance fraction 0.21–0.70. FF seeds 0/1: +0.24 / +0.40; ovarian +0.45 to +0.51; lung +0.43 to +0.53; GSE +0.31 to +0.47. The detector run post hoc returns true for `best_s2` only.

**FF `best_s3` is healthy and replaces `best_s2` in the triple.** Recon −7.3127, NMI 0.622, best epoch 164, ΣKL_w 3.3e-3 at epoch 4; guard excess +0.236, variance fraction 0.373 (beside seed 0's +0.239 / 0.373). Transport Read A 0.662 group / 0.630 own. FF envelope now over s0/s1/s3 (recon −7.3127 to −7.3117, Read A own 0.630–0.669). Baseline table column `DisCell/best_s3` added (spatial variance fraction 0.271 vs s2's 0.000). MI quadrant s3: w excess +0.231 kNN / +0.196 MINE — same band as s0 (+0.226), below s1 (+0.361); the FF w channel is consistently thinner than on the FFPE slides. Signalling share s3: response effect 0.033, p = 0.18 (s0: 0.080, p = 0.001), leak share as before — the FF response-channel significance is seed-dependent and should not be quoted as a finding; the leak channel's is not.

`best_s2` stays on disk as the documented failure. `runs/best` still points to `best_s0`.

**8.9 ovarian block complete (12/12, zero collapses) and rule amendment 2 (made after the ovarian block and after FF α_w 0.3 seed 0/1 arms, before the remaining FF arms; disclosed).** Three-seed ovarian envelope: recon inside the control envelope for every arm (−7.185 to −7.189 vs control −7.188); NMI control 0.665 [0.649, 0.675], warm-up 0.637 (every seed below the control minimum), free bits 0.673, both 0.646; cycle_w control 0.004–0.008, warm-up 0.003–0.016, free bits 0.004–0.023 (one seed over the 0.02 guard), both 0.007–0.024 (one seed over); I(niche;w) excess control 0.50, warm-up 0.64, free bits 0.42, both 0.59 — more KL_w is not more niche information, again. Every treatment arm fails at least one ovarian clause for a real reason: warm-up on NMI, free bits on cycle_w, the combination on both. If FF does not change this the registered fallback applies: per-slide α_w 0.05 for FF (the survey has 0/3 collapses at α_w 0.05 and 0.02 on FF) and the collapse rate reported as a limitation.

Knife-edge found by the agent: I(z;t)/H(t) "failed" the envelope by 9e-6 and 4e-5, four orders below its seed noise, because the envelope test had no tolerance and no direction. **Amendment 2:** (a) I(z;t)/H(t) is judged as in the α_z-ladder rule already on record — within 0.05 of the control mean; (b) every other envelope field passes if inside the control's [min, max] widened by one control seed-sd on each side, or if better in its named direction; (c) the absolute guards (cycle_w ≤ 0.02, probe within its floor band per seed) are unchanged. **Correction (same day, agent's check): amendment 2 does rescue an arm.** Warm-up's NMI deficit of 0.0115 below the control minimum is inside one control seed-sd (0.0139), and its I(z;t)/H(t) gap of 0.033 is inside the 0.05 tolerance, so under amendment 2 warm-up passes the ovarian clause completely; the free-bits arms still fail the unchanged cycle_w guard on one seed each. The record therefore reads: amendment 1 (after FF seed 0) demoted warm-up in favour of the arm with the larger KL_w margin; amendment 2 (after the ovarian block) restores warm-up as the only arm clearing the ovarian clause. Both moved the outcome, in opposite directions, and both were made after seeing data; the original rule's verdict is reported beside the amended one so a reader can follow the decision. Both the original and amended verdicts continue to be reported.

**8.9 queue defect found and fixed (2026-09-23, 15:45).** The DONE markers of the collapse-fix queue were keyed by step and run name only; FF and ovarian share run names (`wfix_<arm>_aw0.1_s<seed>`), so the ovarian block's markers would have made the FF α_w 0.1 block skip 12 of its 16 fits silently and the read-out would have run on n = 1 cells with no error. Markers are now dataset-qualified; the 55 existing ones were migrated by α_w value (unambiguous: no FF α_w 0.1 and no ovarian α_w 0.3 run existed). Recorded because it is the kind of failure that yields a plausible incomplete grid rather than an error, and any future queue of this shape must key markers by dataset. Also: one FF fit at epoch 134/200 was killed on restart before its progress was checked (~45 GPU-min lost, redone).

### Baseline repairs (4.2 stage 1, results, 2026-09-23; agent cut off by a session limit before finalising its report, `scripts/logs/lane2_2026-09-22/REPAIR_REPORT.md`, section A-FF unfinished)

**SIMVI.** The 22 GB OOM was SIMVI's ZINB log-probability on a densified full-graph forward; runner-side chunking fixed it (peak 8 GB). Ran on GSE solo (1.8 h), GSE dual fit-on-section, and an FF window of 100k cells (SIMVI cannot handle the full slide). Columns: GSE solo NMI 0.505, probe +0.032 vs floor −0.009, mirror 0.081 (ours 0.028), cycle_z 0.595 (ours 0.48–0.50), within-type variance 0.66, spatial latent cycle 0.035. GSE dual NMI 0.484, probe +0.024, mirror 0.069 (ours 0.040), cycle 0.610 (ours 0.483). **FF window: probe ΔCE +0.865 against a floor of −0.015, mirror 0.142 (ours 0.058), spatial-latent cycle R² 0.111** — on the fresh-frozen slide SIMVI's intrinsic latent predicts the niche almost perfectly and its spatial latent carries cell cycle; both channels fail the reads they exist for. Intrinsic cycle 0.774 matches ours (0.76–0.79). SIMVI is the nearest competitor on GSE (leaks a little more niche, a little more cycle) and fails clearly on FF.

**MintFlow.** The 5-of-50-epoch result was a runner setting: crop `width_window` 100 against MintFlow's own default 600 makes an epoch 3.7× slower (3,720 vs ~1,015 s); a full 50-epoch GSE fit is ~14 h at the default. The `pyg-lib` warning is a red herring (MintFlow uses its own sampler). The truncated model was transferred to the held-out section (column `MintFlow (transfer, 5/50 epochs)`: NMI 0.897, cycle 0.10, within-type variance 0.16 — a type code). **MintFlow on FF: host OOM, not GPU** — 956 M non-zeros, two CSR copies at 15 GB each plus MintFlow's own re-read and a dense COO build; ≥ 45 GB before training. Recorded; another machine or a runner that streams the matrix.

Pending: the 14-h MintFlow GSE fit at default width (author's call, after the collapse grid); SIMVI FF is done as a window and stays so.

**MintFlow full fit scheduled (author, 2026-09-23).** `scripts/queue_2026-09-23_mintflow_full.sh` (detached) waits for the collapse grid to release a GPU, then runs MintFlow on the GSE core at its default width 600 for 50 epochs (~14 h, cap 20 h), transfers to the dual section, and writes columns `MintFlow (50 epochs, w=600)` and `MintFlow (transfer, 50 epochs)`. Training-time comparison on record (500/40 DisCell early-stopped vs baselines' defaults): GSE — DisCell 1.3 min, resolVI 6 min, SIMVI 1.8 h, MintFlow ~14 h; FF — DisCell 31–72 min, resolVI 2.0 h, SIMVI 4.6 h on a 100k window, MintFlow infeasible on this host. Every fit records `minutes` (ours) or `train_s` + `peak_gpu_gb` (baselines).

### Correction to the Voronoi-face entry: the model prunes at 40 um (2026-09-23)

The entry above says "the model's neighbourhood radius is 60 um set by the clip, and `DEFAULT_MAX_EDGE_UM = 100` never binds." That is true of the **bundle** graph (`preprocess/geometry.py`, 100 um Delaunay prune, clip-limited to d < 60 um), not of the model. `model/prepare.py` applies a second prune at `DEFAULT_MAX_EDGE_UM = 40.0` when it builds the `ModelGraph`, and no pinned run overrides it. **The model's neighbourhood radius is 40 um**, as the manuscript states (§2.1, implementation table). The clip findings stand as measured on the bundle, and the implementation table's claim that the clip "removes faces only for pairs more than roughly 60 um apart, which the prune removes anyway" is confirmed by them (zero bundle edges at d >= 60 um across three slides). Caught while auditing the manuscript's citations against the code.

### Collapse remedy: the grid's verdict (8.9, results, 2026-09-24)

*44 fits (32 FF at α_w 0.3 and 0.1 × 4 seeds × 4 arms; 12 ovarian at α_w 0.1 × 3 seeds × 4 arms), 200/20, α_z at the pinned ½ factor. Read-out `data/datasets/xenium_prime_human_ovary_ff/experiments/wcollapse.md`; verdict `scripts/logs/wcollapse_2026-09-23/DECISION.json`. Queue finished 01:44.*

**Collapse counts (FF).** Control: 4/4 dead at α_w 0.3, 2/4 at the pinned 0.1 (6/8 — worse than the survey's 1/4). Warm-up, free bits, both: 0/8 each by the KL_w probe.

**But the KL_w probe is not the right one, and the I(niche; w) guard shows it.** Free bits on FF: KL_w 0.29 at α_w 0.3 and 0.24 at 0.1 — the most "open" channel by KL — with **I(niche; w) excess = 0.00 on all four seeds at 0.3 and 0.12 [0, 0.24] at 0.1**. Free bits lets q(w) wander within λ per dimension at no cost, so w varies from cell to cell without carrying anything about the niche: a channel open in divergence and dead in information. Warm-up on FF: KL_w 5e-5 at 0.3 and 1e-3 at 0.1 (thin by KL) with **excess +0.37 [0.23, 0.49] and +0.59 [0.51, 0.66]** — more niche information than the FF control's healthy seeds (+0.24 to +0.40) and in the ovarian band. The combination inherits warm-up's information (+0.34 / +0.57). This confirms the 2026-09-23 correction: low KL_w means the posterior sits on an informative prior; the guard, not KL_w, says whether the channel works.

**Ovarian (nothing broken there).** Recon identical across arms. Warm-up NMI 0.637 vs control 0.665 (inside the control envelope widened by one seed-sd), free bits 0.673, both 0.646. cycle_w guard (≤ 0.02): warm-up 3/3, free bits 2/3 (0.0227), both 2/3 (0.0238). I(niche;w) excess: control 0.50, warm-up 0.64, free bits 0.42, both 0.59.

**Rule.** Pre-registered and amended rules agree: **warm-up qualifies (0/8 collapses, ovarian inside envelope, all guards); free bits and the combination fail the ovarian cycle_w guard 2/3.** The amendment did not change the outcome. Had free bits qualified it would have won the amended margin (+4.2 vs +0.3 log10) — and it would have been the wrong choice, because that margin measured divergence, not information. The FF-margin amendment is therefore withdrawn as a criterion for future use; the guard excess is the quantity to compare.

**Costs to state.** Warm-up lowers NMI by 0.02–0.03 on both slides (FF 0.628 vs 0.649; ovarian 0.637 vs 0.665). Its converged objective is the pinned loss; the schedule is 30 epochs of linearly rising α_w on the KL_w term, with checkpoint selection and patience starting after the ramp.

**Decision pending the author (07:10):** adopt `--w-warmup-epochs 30` as default; refit `best_s0/s1/s2` on all four datasets at 500/40 with the full battery, cross-slide, reports, envelope and baseline tables, external reads; the paper's training paragraph gains the schedule. The 8.10 query/prior queue already runs with the flag (it read `DECISION.json` at 01:47).

### Warm-up on both KL terms (8.9b, motivation, 2026-09-24)

**Why (author).** The purpose of the fix is not only to stop dead runs but to leave z and w at least as good as before; a remedy that lowers NMI by 0.02–0.03 has not succeeded on that criterion. Warm-up on KL_w alone unbalances the two latents early: z is under full KL pressure from step one while w is nearly free, so within-type structure that belongs in z is partly taken by w. Annealing both KL terms together keeps the relative pressure as designed while letting both channels open before the penalties bite; the α_z ladder already showed that less early pressure on z gives more continuous content with guards intact.

**Arms.** control (have, from 8.9); warm-up on w only, 30 epochs (have); **warm-up on both KL_z and KL_w, 30 epochs** (new flag `--kl-warmup-epochs N`, scaling α_z on both copies of the z-divergence and α_w on the w-divergence by min(1, epoch/N); adversary, reconstruction and everything else untouched; checkpoint selection and patience start after the ramp as for 8.9). Where: FF at the pinned α_w 0.1, 4 seeds (the collapse setting that matters), and ovarian α_w 0.1, 3 seeds; 200/20, α_z at the ½ factor. Existing 8.9 control and w-only arms are reused (same budget, same α).

**Evaluated by / rule (author's criterion).** The both-KL arm is adopted if (i) 0 collapses on FF by the I(niche; w) guard, and (ii) on ovarian and FF no battery read (recon, NMI, probe, mirror, cycle_z, cycle_w, I(z;t)/H(t), I(niche;w) excess, Read A own-target gap closed) is worse than control beyond one control seed-sd, and at least one is better beyond it. If it fails (ii) where w-only warm-up also fails it, the decision falls to the author between w-only warm-up (NMI cost stated) and the per-slide α_w fallback. What is wished for: both-KL warm-up matches control on NMI and improves cycle_z, with 0 collapses.

**8.9b launched (2026-09-24 08:00).** `--kl-warmup-epochs N` (both z-divergence copies and the w-divergence ramped; mutually exclusive with `--w-warmup-epochs`; same checkpoint/patience/NMI-guard gate); 48 tests pass. Queue `scripts/queue_2026-09-24_wcollapse_b.sh`, read-out `scripts/wcollapse_b_table.py` → `experiments/wcollapse_b.md`, `DECISION_B.json`. Applied to the reused rows already: **w-only warm-up fails clause (ii)** — NMI worse than control by 1.7 sd (FF) and 2.0 sd (ovarian), I(z;t)/H(t) by 1.5 / 2.7 sd, ovarian cycle_w by 2.8 sd; better on mirror, probe, cycle_z, w excess, gap closed. Watch item: the first both-KL fit (ovarian s0) checkpointed at epoch 34, the first eligible after the ramp, and stopped at 54 — recon better than control (−7.198 vs −7.223) but the model had only four epochs at full KL weight before selection; the table counts checkpoints within 10 epochs of the ramp end per arm, and if every seed does this the guards on those checkpoints decide whether it matters.

### Query/prior ablations (8.10, results, 2026-09-24; run at the w-only warm-up objective, 200/20, ovarian, 3 seeds — to be re-read at the final objective)

*Queue finished 08:01. Table `data/datasets/xenium_prime_ovarian_cancer_ffpe/experiments/queryprior.md`.*

| arm | recon | NMI | mirror | cycle_z | cycle_w | Read A own gap | MIC fc (kmeans) | atlas rank |
|---|---|---|---|---|---|---|---|---|
| control (type query, type prior) | −7.187 | 0.652 | 0.039 | 0.523 | 0.010 | 0.746 | 0.872 | 2.7 |
| type-free query | −7.193 | 0.660 | 0.038 | 0.519 | 0.009 | 0.736 | 0.875 | 3.3 |
| image query | −7.187 | 0.639 | 0.037 | 0.535 | 0.012 | 0.746 | 0.766 | 3.0 |
| type-free prior | −7.186 | 0.657 | 0.037 | 0.533 | 0.009 | 0.749 | 0.816 | 3.0 |

**Type-free query: ADOPT by the pre-registered text** — every guard inside the control envelope, transport inside or above. c becomes a pure environment summary at no measurable cost; attention entropy 1.39 vs 1.37 nats. **Image query: report only** — NMI below the envelope (0.639), no transport gain, attention barely tracks Φ (max |corr| with Φ PCs 0.046 vs control 0.041); the image adds nothing as a query that it does not already add by concatenation. **Type-free prior: does NOT lose the transport read** (Read A own 0.749 vs 0.746, mean-read fraction of ceiling 0.805 vs 0.789, beats-both 108 vs 111), guards inside the envelope, cycle_z slightly up — the expected loss of per-type response amplitude did not appear on this slide. Its floor-corrected MIC is lower (0.816 vs 0.872, tumour bands 0.847 vs 0.865): with no type in the prior, w's niche information over its within-type floor is smaller, i.e. some of what looked like per-type response was the type-conditional offset. Atlas rank 3 on every seed vs 2–3.

**Reading.** On ovarian, the atlas is one shared programme with per-type amplitude, and the amplitude turns out not to matter for predicting held-out shifts. This makes the "universal response" definition of w defensible and simpler, but it is one slide, three seeds, a non-final objective, and it says nothing about FF or GSE, where the response is thinner. **Decision deferred** to the final-objective re-read on all four datasets; if it holds there, w is redefined as "how cells shift with their surroundings" without the "of that type" and the per-type gauge offset (V12) disappears with it. Type-free query is adopted now for the re-pin unless the re-read contradicts it.

**Decision (author, 2026-09-24): query and prior stay as they are.** Neither variant improved a read; a costless change is not a reason to alter a frozen model. Both are reported as ablations supporting the design: c may be summarised without the centre cell's type at no cost (so type in the query is not load-bearing), and removing type from the prior leaves held-out transport unchanged on ovarian while lowering floor-corrected MIC — the per-type amplitude of the response is not what predicts held-out shifts there. Type-free query is NOT adopted for the re-pin (supersedes the line above).

**On the warm-up unit (author's question).** The ramp is linear over N epochs and constant after; N = 30. An epoch is ~34 / 100 / 70 / 280 optimiser steps on GSE / ovarian / lung / FF (tile 2048 on GSE, 4096 elsewhere), so 30 epochs is ~1,000 to ~8,500 steps. Convergence tracks epochs more closely than steps (best epochs 35–60 on three slides, 84–164 on FF: 3× spread vs 8× in steps), so epochs stay the unit; the per-dataset step counts are stated. The checkpoint-at-ramp-end observation argues for a shorter ramp, not a different unit.

### R26 applies to every fine-margin decision of the last three days (2026-09-24, author's flag)

Review item R26: after early stopping the trainer evaluates the in-memory last-epoch model (patience epochs past the accepted checkpoint) and writes it as `metrics.json["final"]`; only recon, NMI and the degeneracy block are stored for the accepted checkpoint. **Every decision read-out reads `final`**: `wcollapse_table.py` (all fields, including recon and NMI), `wcollapse_b_table.py`, `alpha_z_decision.py`, `envelope_tables.py`, `sweep.py`. Post-hoc reads (I(niche;w) guard, transport, atlas, external criteria) load `best.pt` and are unaffected.

**Size of the effect, from history.jsonl (which logs the full battery at every evaluation epoch):** NMI moves by −0.02 to +0.02 between the best and the last epoch on the collapse-grid runs (ovarian warm-up s0 0.649 → 0.632; both-KL s0 0.664 → 0.644; FF control s0 0.636 → 0.659), the same size as the arm differences the verdicts rest on. Recomputed at the best epoch from history, ovarian NMI means are control 0.661, warm-up 0.651, both-KL 0.650, free bits 0.671, combination 0.642 — warm-up's deficit shrinks from 0.028 to 0.010; on FF warm-up 0.628 vs control 0.648 stays. Recon and mirror move by < 0.007 and < 0.003.

**Action.** A read-out-side `--at best` mode (history row at the accepted epoch) is being added to the four decision scripts, and the α_z ladder, 8.9, 8.9b and the envelope tables are re-read at the accepted checkpoint from existing histories — no refits, trainer untouched (another agent owns the R26 pipeline fix, todo 8.13). Verdicts are provisional until that re-read lands; any clause that flips is recorded. Once the pipeline fix lands, `final` will be at the checkpoint and the two modes coincide.

### Contact-based leakage kernel vs the Voronoi-face kernel, per section (motivation, 2026-09-24)

*Author request, adopting review R11: show in an appendix table how bad a kernel built on segmented-polygon contact would be, beside a metric for ours. Analysis only, on the stored bundles; nothing is fitted.*

R11 found that the manuscript's reason for not using polygon contact is wrong on these data. The data are not nuclear-expansion segmented: XOA segmented them multimodally, and only a few percent of cells were nuclear-expanded. The conclusion may still hold, for a different reason: segmented polygons rarely touch. The table tests that on every section, on the model's own edge set (Voronoi adjacency pruned at 40 um).

**Per section, from obs and the stored edge tables:**

- **Segmentation.** Shares of the XOA segmentation methods (`xenium_segmentation_method`).
- **Contact facts.** Share of adjacent pairs whose polygons touch; median polygon gap; share of edges with zero apposed wall at 1 um tolerance (a generous contact definition); Spearman rho of apposed wall with polygon gap.
- **Kernel comparison.** A contact kernel c_ij ∝ apposed_ij exp(−d_ij/tau), row-normalised exactly like beta, against beta itself (tau 20 um). Reads for each kernel:
  - share of edges with zero weight;
  - share of connected cells that would receive no leakage at all (an all-zero row);
  - that share in the sparsest and in the densest fifth of cells by local density (centroids within 20 um);
  - Spearman rho of a neighbour's within-row weight share with the polygon gap.

**Wished for:** the contact kernel leaves a large, density-dependent share of cells with no leakage and splits weight by polygon gap. The Voronoi kernel does neither.

**Also reported:** if the face kernel turns out to track density too, that is a finding against the paper's argument and goes in the table, not away from it. Raw face lengths do shrink where cells are small. The paper's defence rests on row normalisation, so the table reads the normalised weights.

### Contact-based leakage kernel vs the Voronoi-face kernel (results, 2026-09-24)

*Scripts `scratchpad/contact_vs_face.py`, `scratchpad/face_share_split.py`; outputs `scratchpad/contact_vs_face.json`, `face_share_split.json`. The table is in the manuscript as `tab:contact` (App. B, app:graph).*

**Segmentation, all five full sections.** 69–87% of cells were segmented by the interior 18S stain, 12–29% by the boundary cocktail and 1.0–2.5% by 5 um nuclear expansion. The manuscript's "under nuclear-expansion segmentation" was wrong for these data (review R11).

**Contact facts, on the model's edge set (Voronoi adjacency, d <= 40 um).**

- Polygons touch on 2.2–6.7% of adjacent pairs.
- Median polygon gap: 0.15–2.0 um.
- Even at 1 um tolerance, apposed wall is zero on **38–57%** of edges. It tracks the gap at Spearman −0.85 to −0.88.

**Kernel comparison.** Both kernels use the same decay and the same row normalisation.

- **The contact kernel leaves 3–11% of connected cells with no leakage at all.** That is 11–34% in the sparsest fifth of each section by local density, and 0.1–0.7% in the densest fifth.
- **The Voronoi kernel leaves 0% on every section**, and none of its edges has zero weight.
- Within-row weight share against polygon gap: contact −0.80 to −0.84; beta −0.48 to −0.55.

**Against us, and reported in the table.** Beta's shares still track the gap. The split shows why:

- The distance decay alone gives −0.49 to −0.62.
- The face alone gives −0.39 to −0.45.
- The full beta is no stronger than the decay alone on four of five sections, and only 0.01 stronger on ovary FF.

So beta grades by proximity through the decay, which is by design (nearer cells leak more). The face adds nothing to that.

**Verdict:** the wish holds in its sharp form. Contact **gates** leakage by density (a third of sparse cells receive none, dense cells almost all receive some), whereas beta gives every connected cell the same budget kappa. It does not hold in its soft form: "beta keeps density out of the within-neighbourhood split" is not true, because the decay puts proximity in on purpose. The manuscript text says exactly this.

### Re-read at the accepted checkpoint (R26): two decisions flip (2026-09-24)

*Delegated (Opus). `discell/experiments/at_best.py` returns the history row at `metrics["best"]["epoch"]` (matches `best` recon/NMI/I(z;t)/H(t) bit for bit on all 83 runs; matches post-hoc `degeneracy.json` on 49/62, rest within 5e-5); `--at {final,best}` on the four decision scripts, default unchanged (8/8 outputs byte-identical), `_at_best` siblings written. 5 tests. Note for the trainer fix: NMI re-evaluated on the same weights differs by up to 0.0026 run to run.*

**α_z ladder — flips: no rung qualifies at the checkpoint; by the rule α_z stays at 1/mean-count.** Clause 1 (cycle_z mean must beat the control mean by more than the control's seed range): at last epoch bar 0.507, ¼ 0.533, ½ 0.527 → pass; at best, control mean 0.440 with range 0.083 (seed 0 fell 0.413 → 0.391) → bar 0.523, ¼ 0.520, ½ 0.519 → **fail by 0.003–0.004**. The effect itself is unchanged (+0.08 cycle_z on every rung seed over every control seed but one); the clause fails because one control seed widened the range. Guards unchanged. A near miss under a conservative clause, but the pre-registered answer is "stay". **The re-pin to α_z/2 made on 2026-09-22 therefore rests on a last-epoch read and is provisional.**

**8.9 — flips on the KL-defined clause, not in substance.** At the checkpoint free bits' ovarian cycle_w breach disappears (seed 2 0.0227 → 0.0141), so free bits passes the ovarian clause and the KL-defined collapse count (0/8); the pre-registered order still picks warm-up; the (withdrawn) margin amendment would pick free bits; the two disagree. **But the information guard, not KL_w, defines a live channel (2026-09-24 morning entry): free bits has I(niche;w) excess 0 on 4/4 FF seeds at α_w 0.3 and on 2/4 at 0.1 — a dead channel by the read that matters. Free bits is out on that ground, and warm-up remains the only remedy whose channel carries niche information on every FF seed.** Warm-up's costs at the checkpoint: ovarian NMI −0.010 (0.651 vs 0.661, inside one seed-sd, no longer needing the widened band), FF NMI −0.020 (0.628 vs 0.648, unchanged); ovarian I(z;t)/H(t) 0.782 vs 0.816 (needs the 0.05 tolerance). Combination arm now fails cycle_w 2/3 and is out.

**8.9b — both remedies still fail the "nothing worse" criterion.** Both-KL warm-up: ovarian NMI now within 1 sd (−0.85 sd), cycle_w still worse (+1.9 sd; 0.0085 vs 0.0041, both far inside the 0.02 guard); FF seeds 0 and 2 pending. W-only: ovarian NMI same, FF NMI worse, FF cycle_w worse (+1.2 sd).

**Envelope tables at the checkpoint:** no verdict changes; GSE cycle_z 0.445 → 0.491; within-type variance fraction −0.05 on lung and GSE.

**Where this leaves the re-pin (author's decision needed).** Two coupled choices, both now to be made at the final objective with checkpoint reads: (a) α_z at 1 (pre-registered rule) or ½ (near-miss effect of +0.08 cycle_z); (b) w-only warm-up 30 (only live remedy) accepted with its NMI cost, or per-slide α_w for FF. Proposed: settle (a) inside the final configuration — 2 α_z × 3 seeds on ovarian and 2 × 4 on FF with `--w-warmup-epochs 30`, 200/20, read at the checkpoint (~4 GPU-h) — then re-pin once. The 8.10 ablations already ran at warm-up 30 and are unaffected by (a).

### Closing evaluation scored the wrong model: fixed (review R26, 2026-09-24)

**The bug.** After early stopping, `Trainer.fit` ran its closing `self.evaluate()` on the in-memory weights, not on the accepted checkpoint. Those weights are `patience` epochs past `best.pt`: a median of 20 epochs and at most 40 over 313 runs. `metrics.json["final"]` therefore described a model nobody uses. `calibrate._short_fit` had the same problem, since it calls `evaluate()` after `fit()`.

Affected readers of `final`:

- `sweep.py`: cycle, mirror, probe, degeneracy and recon gap in every sweep table.
- `envelope_tables.py`: every read except recon and NMI.
- `alpha_z_decision.py`: cycle, probe, mirror and degeneracy.
- `wcollapse_table.py` and `queryprior_table.py`: every read, including recon, NMI and KL_w.
- `report.py`; `crossslide.py` (own-section NMI).

The post-hoc tools that reload `best.pt` are unaffected: validation, degeneracy, transport, atlas and the baseline battery.

**How large, measured on existing runs.** `history.jsonl` holds the full evaluation at the accepted epoch, computed on the same weights that were saved to `best.pt`. Final minus accepted, over 313 runs:

| read | median abs error | p90 | max |
|---|---|---|---|
| probe ΔCE | 0.003 | 0.015 | 0.047 |
| cycle_z | 0.010 | 0.031 | 0.098 |
| NMI | 0.007 | 0.020 | 0.040 |
| recon (nats per count) | 0.002 | 0.007 | 0.023 |
| KL_w (nats) | 0.005 | 0.052 | **0.62** |

In the w-collapse legs (49 runs), 5 runs change collapse status at a 0.05-nat cut, 4 of them in warm-up arms. For example, ovarian klwarm30 s1 reads 0.012 at the accepted checkpoint and 0.149 at the last epoch. The α_z ladder's ¼-vs-½ call differed by 0.006 in cycle_z, inside this error; its ½-vs-control gap of 0.077 is outside it.

**The fix** (`model/train.py`):

- Before the closing evaluation, reload `best.pt` into `self.model`, and restore the covariance state if there is one. The closing metrics then describe the accepted checkpoint, and so does anything that keeps using the trainer afterwards, including calibration.
- `metrics.json` gains `final_epoch` (the epoch `final` describes) and `last_epoch`. **Runs made before the fix lack both keys**, which is how old runs are told apart from new ones.
- New test: `test_closing_evaluation_scores_the_accepted_checkpoint`. It forces the accepted epoch to be the first, lets training run on past it, and checks two things: the in-memory weights equal `best.pt`, and `final` equals the accepted epoch's history row. The test failed on the old code for the substantive reasons: weights differed, NMI 0.62 vs 0.57, probe −0.045 vs −0.024. It passes on the new code.
- The training-related suites (train, w-collapse, adversary, sweep, degeneracy guard, ablations, recovery, metrics) pass: 91 passed in 22 min.

**Handed to the other agent (author, 2026-09-24): steps 2 and 3.**

- **Re-read existing runs.** For a run without `final_epoch`, take the `history.jsonl` row at `best.epoch` in place of `final`. That row is the evaluation of `best.pt`'s weights. Recompute nothing.
- **Re-decide, in this order:**
  1. The w-collapse tables A and B, which carry the warm-up decision.
  2. The α_z ladder.
  3. Query/prior.
  4. The envelope and sweep tables.
- A verdict that changes is a finding, not a reason to retune.

### Final configuration frozen; re-pin launched (author's decision, 2026-09-24 ~10:00)

**Decision.** Stop tuning. The final configuration is the pinned model with two additions: `--w-warmup-epochs 30` (the only remedy whose context channel carries niche information on every FF seed; cost 0.01–0.02 NMI, stated) and α_z = ½ × 1/mean-count (kept: every rung seed beats every control seed but one by ~0.08 cycle_z with guards intact and better recon; the pre-registered clause fails by 0.003 at the checkpoint because one control seed widened the range — disclosed as a near miss, not claimed as a rule pass). Both-KL warm-up dropped (trades 0.01 NMI for 0.004 cycle_w, all inside guards). Query and prior unchanged. Everything else as frozen on 2026-09-21.

**Re-pin.** `best_s0/s1/s2` on all four datasets at 500/40 with the final configuration, run names `final_s<seed>`, `runs/best` → `final_s0`; validate, degeneracy (with the I(niche;w) guard), atlas with cross-seed comparison, transport all reads (tumour bands on ovarian), GSE cross-slide, reports; envelope tables, baseline DisCell columns, external criteria (6b.1/6b.3/6b.4) and GO localisation re-read; all tables at the accepted checkpoint (`--at best`; the R26 trainer fix is owned elsewhere and, once in, makes the two modes coincide). The α_z/2 runs (`best_az0.5_s0`, `best_s1/s2/s3`) stay on disk as the pre-warm-up generation. Failed fits (guard) are refit with the next seed and counted.

**From here the paper gets data.** Remaining model-side items are closed; open work is evaluation only: 6b.2 marker pairs (author approval pending), 6b.7 three-way contamination figure, 6b.9 MintFlow perturbation (after the full fit), handover refresh, paper sections B/C.

### Review R12: what the leakage term assumes, tested (motivation, 2026-09-24)

**Issue (review R12, writer's summary).** The leak mixture assumes (a) the same κ for every gene and (b) a leaked amount that scales with the receiver's depth (κ·ℓ_i), not the donor's. Whatever these miss lands in w, because z is forbidden to hold niche-predictable content. The wording fix is the writer's; the tests are ours.

**Test 1 — read-only, no model: does foreign transcript count track the receiver or the donor?** Using the transcript-flux assignment (extranuclear transcripts nearer a neighbour's nucleus than their own), regress each cell's foreign count on log ℓ_i (receiver depth) and on log Σ_j β_ij ℓ_j (face-weighted donor depth), jointly, within type, on ovarian FFPE and FF. Also the donor-density variant Σ_j β_ij ℓ_j / A_j with A_j the segmented area. Wished for: a clear winner. If the receiver coefficient dominates, assumption (b) holds and the wording fix suffices; if the donor term dominates, the depth-scaled arm below matters.

**Test 2 — sensitivity arms, run beside the default, never as defaults.** (i) `--kappa-mode depth`: κ_i = κ · clip(Σ_j β_ij ℓ_j / ℓ_i, 0, c_max) with c_max such that κ_i ≤ 0.5, nothing estimated; (ii) `--kappa-mode gene`: κ_g = κ · s_g / mean(s_g), s_g the per-gene extranuclear transcript share measured from coordinates (transcript-flux artefact), nothing estimated; mixture renormalised over genes. Note on the writer's proposal: multiplying β_ij by ℓ_j/ℓ_i and renormalising rows cancels ℓ_i and leaves κ·ℓ_i as the leaked amount — it changes which neighbours contribute, not how much leaks; (i) is the version that changes how much. Where: ovarian final configuration (warm-up 30, α_z ½), κ 0.1, 3 seeds, 200/20, at the accepted checkpoint. **Evaluated by** the battery guards, recon, transport (mean read and Read A own), the signalling-gene share of the leak and response channels, and the GO localisation lean of B. **What is wished for:** the response findings (transport, surface lean of B, w's LR share) survive all three leak forms — then "stable across κ" becomes "stable across κ and its form". If the surface lean of B drops under the gene-tilted arm, the flat κ was leaving membrane-gene leakage in w, and that is the finding.

**8.11 launched (10:18).** `scripts/queue_2026-09-24_final.sh`: flags reproduce each dataset's `best_s1` config plus `--w-warmup-epochs 30` (α_z GSE 0.0018, ovarian 0.0035, lung 0.002, FF 0.00035, checked at start); dead-channel refit with the next seed, logged in `DEAD_RUNS.tsv`; phase 2 re-points `runs/best`, regenerates envelope (at best), DisCell battery columns, external criteria, GO; then **re-scores every existing baseline column against the new context** so mirror R² is comparable across the table (no refits). **The R26 trainer fix has landed in the working tree** (other agent): GSE `final_s0` reports `final_epoch` = accepted epoch 39, so `final` and the checkpoint now coincide for new fits; the trainer file hash is logged per fit. GSE triple fitted and alive by 10:50; ETA for the whole queue ~18:00, GSE/dual baseline columns after the MintFlow fit exits.

### R12 test 1 — the foreign count does not scale with the donor (results, 2026-09-24)

*Delegated (Opus). `discell/experiments/r12_depth_test.py`, reusing `transcript_flux`'s streaming and hard leak assignment unchanged (FFPE per-cell κ_i reproduced exactly); 8 tests. Outputs `experiments/r12_depth_test.{json,md,png}` on ovarian FFPE (400k cells) and FF (1.11 M). Bootstrap over 200 µm tiles.*

**Verdict: assumption (b) survives as far as this test can see; a donor-scaled leak is ruled out.** With type fixed effects, log(1+F_i) on log ℓ_i (receiver) and log Σβ_ij ℓ_j (donor): FFPE receiver std coef +0.68 (partial R² 0.38), donor **−0.18** (0.04); FF receiver +0.67 (0.43), donor **−0.28** (0.12). The donor term is negative in every specification on both slides (density variant, nuclear-count depths, ramp weights, every segmentation group); a donor-scaled leak would make it positive. Receiver dominates by the pre-registered partial-R² ratio in 3 of 4 cells; FF primary is "no clear winner" only because its *negative* donor term is large.

**But F_i is not κ·ℓ_i in shape either.** Elasticity on ℓ_i 1.4 (FFPE) / 2.4 (FF), not 1; raw slope 0.24 / 0.20 with intercepts −21 / −120; F_i = 0 for 26 % of FFPE cells and its binned median is 0 below ℓ ≈ 45 (FFPE) / 200 (FF). Pattern receiver ↑, donor ↓, own area ↑, neighbour area ↓ is what a large cell beside small ones produces: the halfway line between nuclei falls inside the large cell and its own far-side transcripts are counted "foreign". So receiver dominance is mostly segmentation geometry, and F is an upper bound on leakage, not a leak count — consistent with 5.5 (κ_i level barely depends on nucleus position). **What this licenses in the paper:** state (b), say the position data rule out donor scaling, and do not claim they confirm receiver scaling. The depth-scaled sensitivity arm (test 2(i)) is now a robustness check only.

First FF κ_i bracket: hard median 0.080, pooled 0.133; ramp-4 median 0.117, pooled 0.150 — the same 0.1-ish bracket as FFPE.

### R12 test 1b — donor scaling at fixed area (motivation, 2026-09-24; writer's pushback, todo 8.14)

**Why.** Test 1 read a negative donor-depth coefficient as ruling out donor scaling. But segmentation geometry alone predicts receiver ↑ and donor ↓ (a large cell beside small ones counts its own far-side transcripts as foreign), and depth and area are strongly correlated, so the sign of the donor term unconditioned on area is not evidence about leakage. Test 1's unregistered check already showed the FFPE donor term turning positive (+0.15 standardised) once own and neighbour area entered. The "ruled out" wording was therefore not adopted by the writer; the appendix says only that position counts bound leakage from above.

**Test 1b.** Same outcome log(1+F_i), type fixed effects, both slides, joint regressors: log ℓ_i, log Σβ_ij ℓ_j, log A_i, log Σβ_ij A_j (donor area with the same β weighting). Variant: transcript density (ℓ/A) for receiver and donor in place of raw depth. Report the receiver depth–area and donor depth–area correlations. Tile bootstrap CIs as in test 1.

**Pre-registered threshold.** The quantity is the donor-depth elasticity at fixed areas (unstandardised log-log coefficient). Under F_i ≈ κ ℓ_i the receiver elasticity is 1 and the donor's 0; under a donor-scaled leak the donor's is of order 1. "Practically positive" = donor elasticity ≥ 0.25 (a quarter of the model's receiver elasticity). **Donor scaling is ruled out only if the donor-depth coefficient at fixed donor area is ≤ 0 and its 95 % CI upper bound is < 0.25**, on both slides. A coefficient in (0, 0.25) with CI excluding 0.25 is "small, not zero" and is reported as such; a CI covering 0.25 leaves it open and the depth-scaled sensitivity arm decides in practice.

**On "consistent with 5.5".** Todo 5.5 (transcript-flux per-cell κ_i, 2026-09-17) is marked failed because its per-edge slope against the model's leak prediction (0.2–0.3) missed the pre-registered [0.5, 2] band; that failure is exactly the finding that position-derived foreign counts do not track the model's leak quantity per edge. Its reported observation that the κ_i level barely depended on nucleus position is a valid negative result of a failed leg, and "consistent" means the geometry reading of test 1 agrees with it — not that 5.5 supports any leak model.

### R12 test 1b — donor scaling is NOT ruled out (results, 2026-09-24)

*Same agent and caches as test 1; `r12_depth_test.py` gained the two area-conditioned specs, the pre-registered classifier and two planted worlds; 15 tests pass. Results under `test_1b` in `experiments/r12_depth_test.{json,md,png}`.*

At fixed receiver and donor area, the donor-depth elasticity is **+0.36 [0.34, 0.38] on FFPE — above the 0.25 "practically positive" bar, whole CI above it — and +0.06 [0.05, 0.08] on FF — "small, not zero"**. Density variant gives the same classes (+0.38, +0.08). Receiver elasticity at fixed areas 0.53 (FFPE) / 1.43 (FF); own area +1.4 / +1.1, neighbour area −1.2 / −1.6 (the geometry pattern); depth–area correlation 0.70–0.72 receiver, 0.55–0.70 donor, which is why the unconditioned test 1 sign was uninformative. **Test 1's "ruled out" is withdrawn; the writer's appendix sentence (position counts bound leakage from above, identify neither scaling) stands.** Caveats: FF's added-variable plot is an inverted U, so +0.06 summarises a non-monotone relation; segmented area is a proxy for size (a planted geometry-only world with noisy areas gives −0.25, so FFPE's +0.36 argues against ruling donor scaling out without being a leak measurement); F is an upper bound and 26 % of FFPE cells have F = 0; at a true donor elasticity of 0 the rule's "≤ 0" clause makes "ruled out" vs "small, not zero" a coin flip, so FF's class is weak evidence either way.

**Consequence.** The depth-scaled κ_i sensitivity arm (test 2(i)) is no longer a formality: on FFPE the position data are compatible with a leak that scales partly with the donor. The arm's read on transport, guards and the surface lean of B is what the paper can say about it.

**R12 test 2, third arm added (author, 2026-09-24): depth by density.** The queued depth arm uses raw totals, κ_i = κ·clip(Σβ_ij ℓ_j / ℓ_i), which conflates a large cell with a dense one; the physical quantity behind donor scaling is transcript density at the shared face (β already carries face length). Arm `--kappa-mode density`: κ_i = κ · clip( Σ_j β_ij (ℓ_j/A_j) / (ℓ_i/A_i), 0, c_max ), same clip. Area: segmented cell area where the boundary was segmented; where segmentation is nucleus-expansion the cell area is a fixed dilation and carries no information, so there A = nuclear area (DAPI-derived) × the type-median cell/nucleus area ratio of boundary-segmented cells of that type; the fraction of expansion cells per type is reported. Same seeds, budget and reads as the other arms. Wished for: raw-depth and density arms agree; if they disagree, the size-versus-density distinction is itself a finding for the leak-term paragraph.

**8.9b complete (2026-09-24 10:59, last-epoch reads; at-checkpoint re-read agrees).** Both-KL warm-up: 0/4 FF collapses by the guard, but NMI worse than control beyond one sd on both slides and ovarian cycle_w worse; w-only warm-up likewise fails the "nothing worse" criterion (NMI, I(z;t)/H(t), ovarian cycle_w). Neither remedy is free; the author's decision of 10:00 (w-only warm-up 30, cost stated) stands. Full table `experiments/wcollapse_b.md`, `_at_best` sibling.

**R12 test 2 built and queued (2026-09-24 11:45); one amendment before any fit.** `--kappa-mode {global,depth,gene,density}`; s_g computed once from `transcripts.parquet` (`transcript_flux --gene-share`, cached `experiments/gene_extranuclear_share.npy`; median 0.39, ECM genes highest, κ_g 0.014–0.28 at κ 0.1, clip never binds); validate/transport/external_criteria now use each group's own κ so the reads are computed with the arm's leak term (unchanged under global; 19 planted tests; 3-epoch fit hash-identical before/after). Control = `wfix_warmup30_aw0.1_s{0,1,2}` (exact configuration, verified). **Amendment (author, before any arm fit):** the depth and density ratios average above one (mean κ_i 0.15 / 0.12 vs 0.1), which would confound "differently shaped leak" with "more leak"; the ratio is normalised to mean one over training cells, κ_i = κ·clip(r_i / mean(r_i), 0, c_max), nothing estimated. Known gaps carried: `report.py`/`sweep.py` do not pass `kappa_mode` (W-qp1 pattern); the R26 test in another lane is flaky under load (NMI re-evaluation noise ~0.003 vs a tight tolerance). ETA 20:00–24:00 sharing the cards with the re-pin.
**R12 test 2, normaliser final (12:09):** the scalar m is solved by bisection so that the post-clip mean of κ_i over connected training cells equals κ exactly (ovarian: depth m 1.55, 2.2 % of cells at the 0.5 cap; density m 1.18, 0.9 %; post-clip mean 0.1000 for both). All three arms now leak the same total as the default; only the shape differs. 43 ablation tests, 172 across the related files, global form hash-identical. Queue relaunched idempotently; no fit had started.

### α_w in units of 1/ℓ̄: the multiplier ladder (R19, motivation, 2026-09-24 13:50)

**Why (review R19, writer's brief).** α_w = 0.1 is 14× the bound-equivalent 1/ℓ̄ on ovarian, 25× on lung, 28× on GSE and **143× on FF**. KL_w is near zero by construction and every delivered w read is effectively the prior mean m_ψ(c,t) (the 2026-09-24 KL_w interpretation entry says the same from the other side). The 2026-09 α_w study rejected values below 0.1 on an NMI veto and, earlier, on a bistability traced to the w-mirror through neighbours' μ_z; type-only attention closed that route, and the sweep3 grids on all four datasets show no bistability at α_w 0.02–0.07 (recon seed spread ≤ 0.010) with KL_w opening to 0.19–0.33 on ovarian. The cost seen there is NMI, most on FF (0.50–0.56 vs 0.61). Lower α_w would also reduce the collapse pressure that motivated the warm-up (FF collapses 0/3 at α_w 0.02 and 0.05, 1/4 at 0.1, 3/3 at ≥ 0.2).

**Design.** α_w = m × 1/ℓ̄ with one multiplier m shared by all sections, ℓ̄ the mean total count of the training cells (fixed once per dataset; equals 1/(2α_z) at the pinned α_z). Ladder m ∈ {1, 2, 5, 10, 25}, plus the current α_w = 0.1 as the reference rung (m = 14.3 / 25 / 27.8 / 143 per dataset). Four datasets, 3 seeds, 200/20, the final configuration otherwise (warm-up 30 on w, α_z ½, type-only, κ 0.1). Plain runs first; a reverse-anneal arm (start at 0.1, anneal down to the rung) is added only if a rung misbehaves, and the standard warm-up is NOT used as that remedy. Every read at the accepted checkpoint (`--at best`).

**Guards (pre-registered; every seed, every dataset).** Recon seed spread within the reference rung's band; NMI ≥ 0.9 × its running max and within 0.02 of the reference rung; probe ΔCE within its floor; cycle_z inside the reference seed band; cycle_w ≤ 0.02; I(niche;w) guard alive; a w-side mirror (held-out R² of μ_w's within-type residual on neighbours' μ_z) not above the reference by more than one seed-sd.

**The read that decides (does the per-cell deviation d_i = μ_w,i − m_ψ(c_i,t_i) do held-out work?).** (i) KL_w per dimension; (ii) deviation→type NMI and deviation→cycle R² (must stay low: d is neither identity nor cycle); (iii) held-out reconstruction with w = μ_w minus with w = m_ψ, per count, on held-out cells — the deviation's held-out gain; (iv) the twin read's margin with w = μ_w vs w = m_ψ. **Rule.** Adopt the smallest m that passes every guard on every dataset AND whose held-out gain (iii) exceeds the reference rung's by more than the reference's seed range on at least the two primary datasets (GSE, FF). Otherwise α_w stays at 0.1 and the paper takes R19's reframing: w is the context regression m_ψ, and the per-cell channel is closed by design.

**Consequence if adopted.** The re-pin (8.11) is redone at the new α_w; the R18 table and the implementation row are written in 1/ℓ̄ units either way. Runs `aw_m<m>_s<seed>`, tag `awladder`.

### MintFlow, full fit (results, 2026-09-24 13:11)

50 epochs at MintFlow's default width 600 on the GSE core: 11.3 h, peak GPU 6.7 GB; transferred to the dual section. Columns `MintFlow (50 epochs, w=600)` and `MintFlow (transfer, 50 epochs)`. **The fair fit reads the same as the truncated one:** NMI 0.887 / 0.915, I(z;t)/H(t) ≈ 1, within-type variance fraction 0.003 / 0.004, intrinsic cycle R² 0.07 / 0.13, spatial-latent cycle 0.035 / 0.047, probe at floor, mirror 0.030 / 0.035. MintFlow's 100-d intrinsic latent is a type code with almost no continuous state; its microenvironment latent carries a little cycle. Against DisCell on GSE: NMI 0.89 vs 0.62–0.64, cycle_z 0.07 vs 0.48–0.50, within-type variance 0.003 vs 0.68–0.70. The comparison table for the paper is now complete on GSE (solo + held-out) for all three baselines; FF has resolVI and the SIMVI window; ovarian and lung have resolVI. Training time: DisCell 1.3 min vs MintFlow 11.3 h on the same core.

**Issue (B-mf1):** the MintFlow `recon (log-lik / count)` cell reads −5.035 identically for the truncated, the w=600 truncated and the full fit, and −5.018 for both transfers — a constant across different weights is not a likelihood read; the battery's MintFlow decoded-rate path must be inspected before that cell is quoted. Logged in issues.
**8.15 scope amendment (author, 2026-09-24 14:05, before any fit is read).** Staged: stage 1 = GSE core and ovarian FFPE, seeds 0 and 1, all six rungs (22 fits, ~1 h) → guards and deviation read → partial table. Stage 2, only on the author's word = FF, 2 seeds, the smallest stage-1-passing rung and the rung above it, plus the FF reference. Lung dropped from the ladder. Rule unchanged except that "every dataset" now means GSE and ovarian in stage 1 and GSE, ovarian and FF at the end.
**8.15 stage 1 running (14:25).** Stage-1 α_w values: GSE 0.0036 / 0.0072 / 0.018 / 0.036 / 0.09 (+ 0.1 ref); ovarian 0.007 / 0.014 / 0.035 / 0.07 / 0.175 (+ 0.1). **ℓ̄ discrepancy, for R18:** the mean total count of the training cells is 331 (GSE), 258 (ovarian), 1662 (FF), not the 278 / 143 / 1429 implied by 1/(2α_z); the pinned α_z values were set from an earlier per-slide mean (different cell filter / all cells vs connected training cells). The ladder uses ℓ̄ := 1/(2α_z) so that its multipliers are the ones quoted in the pre-registration; the R18 text must define ℓ̄ explicitly and state the pinned α_z in that unit rather than as "≈ 1/ℓ̄". Stage-1 table ETA ~16:00–16:30.

### The invariance probe re-graded per block, with a nonlinear grader (R20 + R22, motivation, 2026-09-24 15:00)

**Defect (R20).** `probe_delta_ce` scores one ridge regression by pooled squared error over v = [composition minus one column, 12 raw image PCs]. The PCs' variances sum to ~113 on ovarian against ≤ 1 for composition, so > 99 % of ΔCE is the image block; composition — the channel of the leakage confound — is effectively ungraded, and every invariance guard in every decision rule (α_z, collapse, query/prior, ladder) inherited this. **The adversary does not share the defect:** its heads use a categorical cross-entropy on composition and a soft cross-entropy on image-niche memberships, each against its own type baseline (elbo.py `adversary_terms`), so composition was penalised at full weight in training. Model and training untouched.

**Fix (evaluation only).** Per column k, gain_k = ½·log(MSE_k(type-only baseline) / MSE_k(probe)), the Gaussian cross-entropy gain with fitted variances (a conditional predictive V-information estimate, Xu et al. 2020); reported **per block** — composition and image — each with its own within-type permutation floor; the pooled number secondary. **A nonlinear grader (R22)** beside the ridge: the calibration MLP probe (`calibrate.mlp_probe_delta_ce`, 64→64, 300 Adam steps) on the same targets, split and floors. Both families, both blocks, at the accepted checkpoint (`best.pt`).

**Pre-registered guard.** A run passes invariance iff, for both probe families, the composition-block gain and the image-block gain are each within their floor band (|gain − floor| ≤ 2 × floor sd over the permutation draws) on held-out cells. Baselines are graded by the same rule on their stored intrinsic latents.

**Re-grade, no retraining:** pinned `best`/`final` triples on all four datasets (+ GSE dual), the sweep3 grids, the α_z ladder, the collapse grids (8.9, 8.9b), the query/prior and κ arms, the α_w ladder as its runs land, and the baselines (resolVI, SIMVI, MintFlow). **A verdict that changes on re-grading is a finding, not a re-tuning.** The α_w ladder table is read only with the per-block probe.

**Wished for.** DisCell's composition gain at floor everywhere (the adversary enforced it); resolVI's composition gap to us measured rather than assumed; the image block equal to today's ΔCE.
**Queue priority (15:35).** The κ-arm queue (3/9 fits done) is paused to give GPU 1 to the α_w ladder stage 1, which was being starved by the other queues' short-step lock churn; it resumes idempotently after stage 1. The ladder's courtesy window on foreign locks is lowered from 300 s to 90 s.

### Per-block probe, first read (8.16 interim, 2026-09-24 ~15:45): z carries a small, detectable amount of composition beyond type; the 2-sd guard is the wrong scale

*From `runs/*/validation/probe_blocks.json` and `experiments/probe_regrade/*.json` written so far (final triples on GSE, lung, ovarian, FF partial; resolVI on GSE, lung, ovarian; SIMVI and MintFlow on GSE).*

**Ridge, composition block, excess over the within-type permutation floor (nats per column, gain = ½ log MSE ratio):** DisCell final triples +0.0029/+0.0037/+0.0036 (GSE), +0.0025/+0.0024/+0.0026 (lung), +0.0044/+0.0042 (ovarian, two seeds so far), +0.0070/+0.0049/+0.0088 (FF). resolVI +0.0097 (GSE), +0.0122 (lung), +0.0258 (ovarian). SIMVI +0.0113 (GSE). MintFlow +0.0035 / −0.0020 (two fits, GSE). In variance terms exp(2·gain) − 1: DisCell's probe explains ~0.5–1.8 % of the within-type variance of the composition columns; resolVI 2–5 %. **MLP, composition:** DisCell +0.013–0.020 (GSE, lung, ovarian), +0.019–0.031 (FF), i.e. 3–6 %; resolVI +0.024 / +0.032 / +0.075 (5–16 %); SIMVI +0.026. Image block: DisCell ridge +0.001–0.007, MLP +0.006–0.023; resolVI +0.012–0.028 / +0.038–0.063.

**Reading.** (1) Composition information in z, given type, is small but real and reproducible across seeds and slides — the adversary reduced it to roughly a quarter to a half of resolVI's, not to zero; the pooled legacy probe could not see this because the image block dominated its scale. (2) The nonlinear grader finds 3–5× more than the ridge on every latent, as Elazar et al. predict. (3) FF carries the most (the slide where w is thinnest). (4) **The pre-registered pass rule, |excess| ≤ 2 × floor sd, is unusable:** the floor sd is 3e-5 to 1e-3 nats because the floor is estimated on tens of thousands of held-out cells, so any real excess is tens to hundreds of sd out and *every* latent fails, including MintFlow's near-type-code. That rule tests "exactly zero detectable dependence", which the probe's own sentence in the paper already says it cannot certify. The sd of the floor is estimation noise of the floor, not a tolerance for "negligible". I set that rule this morning; it was the wrong scale and is withdrawn before any decision uses it.

**Replacement scale, pre-registered now.** Report every excess in nats per column and as (i) the implied within-type variance fraction exp(2·excess) − 1 and (ii) a fraction of the *uncontrolled* reference — the α_a = 0 fit, which the spec names as the baseline the probe is to be compared against (ovarian `runs/_archive/cal_alpha_a_0`; refit α_a = 0 at the final configuration, one seed per dataset, cheap) — and of resolVI's excess on the same slide. **Guard:** a run passes invariance iff its composition-block excess is ≤ 25 % of the uncontrolled reference's on both graders, and its image-block excess likewise. The 25 % is a judgement, stated as such; the fractions are reported so a reader can apply their own. No decision rule is re-run until the α_a = 0 references exist.

### Residual composition in z: capacity, weighting or biology? (motivation, 2026-09-24 16:00)

**Question.** The per-block probe finds z carrying ~0.5–2 % (ridge) / 3–6 % (MLP) of within-type composition variance beyond type, a quarter to a half of resolVI's. Three sources: (1) adversary capacity (small heads, fixed steps; the MLP grader is stronger); (2) adversary weighting (head loss = composition CE + image-membership CE; if the image term dominates the gradient, composition was under-enforced; α_a itself was calibrated against the image-dominated pooled probe — note the 2026-09-01 calibration quoted "MLP-probe leak 14 % of uncontrolled", so an uncontrolled-relative MLP read was the original yardstick and the new ≤ 25 % guard is in the same units); (3) biology — intrinsic state co-varies with the niche through selection, which no adversary should remove.

**Diagnostic first (no training change).** Grade a fit on a planted world where z ⟂ niche by construction (the x̃-gate world, `discell/applications/planted.py`, the fit path `fit_synthetic` at the pinned adversary settings) with the per-block probe, ridge and MLP, against its own within-type floor and an α_a = 0 fit of the same world. Wished for: residual at floor → capacity is sufficient and the real-data residual is weighting or biology; residual present → capacity. Then a weighting check on the same world with the composition head weight raised (×3) — a planted world cannot have source (3), so whatever remains is (1) or (2).

**Proposed, not launched (author's word needed — training change):** an adversary ladder on ovarian, 2 seeds, final configuration: head steps ×2, head width ×2, composition weight ×3, and their combination; read by the per-block probe (fraction of uncontrolled), NMI, cycle_z, cycle_w, recon. Adopt only if the composition fraction falls with guards intact; otherwise the residual is reported as the ceiling with the planted-world evidence for which source it is. Post-hoc linear erasure of the composition-predictive subspace (LEACE-style) is an evaluation-side alternative to report, not a default.

### Planted-world probe: the residual is adversary capacity (results, 2026-09-24 17:00)

*Delegated (Opus). `discell/experiments/planted_probe.py` (`fit_planted`: `fit_synthetic`'s loop with the real adversary branch and the w warm-up; bit-identical to `fit_synthetic` in its own mode); 9 fits, 3 seeds × {α_a = 0, pinned, composition weight ×3}; per-block ridge and MLP probes at n_perm 5 and a 50-draw re-grade. Report `scripts/logs/planted_probe_2026-09-24/`.*

**In a world where z ⟂ niche given type by construction, the pinned adversary leaves composition in μ_z on every seed:** MLP +0.024 / +0.009 / +0.009 nats per column (4.8 / 1.8 / 1.8 % of within-type variance), 3.6–11 floor sd above floor; ridge at floor on one seed, +0.002–0.009 on two. This is 2–10 % of the same seed's α_a = 0 excess (passes the ≤ 25 % guard) and **the same size as the real-data residual (ridge 0.5–1.8 %, MLP 3–6 %)** — so the tissue residual needs no biology to explain it, though biology is not excluded there. Intrinsic signal is kept (held-out R² of the true z from μ_z 0.39–0.50, equal to the α_a = 0 fit). **Composition weight ×3** lowered the residual only on the seed with the worst one (0.024 → 0.006); the others were flat or up — by the pre-stated rule the reading is **capacity**, with a ceiling of ~0.006–0.014 MLP nats (1.3–2.7 %) that reweighting does not move. The true z reads at floor (11/12 cells within ±2 sd at 50 draws; the one outside is below floor), so the probe is unbiased.

**Consequence.** The "most but not all" sentence stands, and the residual is attributable to the adversary's heads not seeing what a stronger grader sees — not to biology, not to the weighting alone. The remedy, if wanted, is head capacity (steps, width, ensemble): the proposed ovarian adversary ladder (training change; author's word). Deviations recorded: `fit_synthetic` has no adversary (closed-form penalty hard-coded), so the loop was rebuilt locally; no composition-weight knob exists in the model — the ×3 arm used a local loss copy; 25 % of tiles held out; model sized as the synthetic (d_z 8).

**Hand-off, 2026-09-24 ~17:10 (author away).** Detached and self-completing: α_w ladder stage 1 (table → `experiments/awladder.md`, `scripts/logs/awladder_2026-09-24/`), final re-pin post-processing (`runs/best` → `final_s0`, envelope/baseline/external tables), per-block probe re-grade with α_a = 0 references (`experiments/probe_regrade/`, `scripts/logs/probe_regrade_2026-09-24/`), κ arms auto-resume after the ladder (`scripts/logs/r12_arms_2026-09-24/`). Agent reports land as `AGENT_REPORT.md` in each log folder. **Awaiting the author:** stage 2 of the ladder (FF), the adversary-capacity ladder (training change), the κ-arm red line, marker pairs, commit. No decision rule fires automatically from here: the ladder writes `DECISION_AW.json` but nothing consumes it; the re-pin re-points `runs/best` to `final_s0` as pre-registered this morning.

### Per-block probe re-grade, DisCell vs baselines (8.16 step 1, results, 2026-09-24 evening)

*Delegated (Opus). `metrics.probe_gain_per_block{,_mlp}`, `probe_blocks`, `probe_verdict`; `validate --analyses probe`; `experiments/probe_regrade.py`; α_a = 0 references `uncontrolled_s0` on all four datasets (200/20); 8 planted tests, 43 across related files; the ridge legacy block reproduces the in-trainer ΔCE to 1e-7, so the re-grade read identical inputs. Tables `experiments/probe_regrade_{final,baselines}.md` per section; full `scripts/logs/probe_regrade_2026-09-24/baselines_all_sections.md`.*

**Units:** excess over the within-type permutation floor, nats per column; implied within-type variance exp(2·excess) − 1; ·u = fraction of the α_a = 0 fit; ·r = fraction of resolVI.

| section | DisCell final (3 seeds) ridge comp / MLP comp | image ridge / MLP | resolVI comp ridge / MLP (·u) | SIMVI | MintFlow 50 ep |
|---|---|---|---|---|---|
| GSE solo | 0.7 % 0.41u / 3.5 % 0.69u | 0.27u / 0.40u | 1.18u / 0.97u | 1.38u / 1.01u | 0.43u / 0.10u |
| GSE dual | 1.0 % 0.53u / 3.9 % 0.67u | 0.28u / 0.42u | 0.99u / 1.05u | 1.23u / 0.94u | 0.45u / 0.13u |
| ovarian | 1.4 % 0.34u / 5.4 % 0.39u | 0.23u / 0.34u | 1.26u / 1.11u | — | — |
| lung | 0.5 % 0.24u / 2.7 % 0.43u | 0.13u / 0.16u | 1.18u / 1.02u | — | — |
| FF | 0.9 % 0.19u / 3.2 % 0.25u | 0.14u / 0.16u | 0.74u / 0.84u | 0.65u / 0.49u | — |

**Reading.** (1) The adversary removes 30–80 % of the composition information the uncontrolled model leaks, most on FF and lung, least on GSE under the MLP (0.67–0.69u). (2) **The uncontrolled DisCell leaks as much as resolVI** (0.8–1.1r on four sections) and more on FF (1.2–1.5r): the architecture does not remove niche from z by itself; the adversary does. (3) resolVI and SIMVI sit at or above uncontrolled on every block; MintFlow is near floor on the MLP because its latent is a type code. (4) DisCell as a fraction of resolVI: composition 0.21–0.53 (ridge), 0.30–0.71 (MLP). (5) Under the ≤ 25 %-of-uncontrolled guard, DisCell passes on FF seed 2 only (1/15 final runs); FF sits on the line (0.24–0.26u MLP comp); the binding block is always composition under the MLP.

**Grader choices signed off:** z and v standardised by training-cell moments; one MLP per block (a shared network let a composition fit leak into the image grade by 2.2 sd on a planted z); constant held-out columns dropped (GSE dual: 2). **Caveat to fix:** the uncontrolled references were 200/20 against 500/40 finals; on FF the reference stopped at epoch 54 vs 179 — the FF fractions depend on a shorter denominator. Uncontrolled refits at 500/40 ordered; single-reference denominators carry ~10–20 % seed noise, so a second uncontrolled seed on GSE and ovarian is ordered too.

**Paper sentence (unchanged in substance, now with numbers to come):** the adversary removes most but not all detectable composition information — 30–80 % of the uncontrolled leak, graded per block with linear and nonlinear probes; the remainder is the heads' capacity (planted-world entry), not biology; resolVI and SIMVI leak as much as an uncontrolled DisCell.

### α_w ladder, stage 1 (8.15, results, 2026-09-24 17:49)

*GSE and ovarian, seeds 0–1, rungs m ∈ {1, 2, 5, 10, 25} × 1/ℓ̄ plus the 0.1 reference; reads at the accepted checkpoint; `experiments/awladder.md`. No dead fits.*

**Rule outcome: no rung passes every guard on both slides; α_w stays at 0.1.** GSE: m10 passes every guard; m1 fails NMI (0.603 vs bar 0.613) and cycle_w (0.024); m2 NMI; m5 misses NMI by 0.004; m25 recon spread. Ovarian: every rung fails cycle_z and/or the w-mirror against the reference band, including m25 (α_w 0.175, *above* the reference), and m1 also cycle_w (0.027–0.028).

**The channel opens and does held-out work.** Held-out gain of the per-cell deviation (recon with μ_w minus with m_ψ, nats/count): GSE 0.0002 (ref) → 0.0009 (m10) → 0.0020 (m5) → 0.0060 (m2) → 0.0129 (m1); ovarian 0.0006 → 0.0009 → 0.0023 → 0.0094 → 0.0192. Deviation share of w's variance up to 21 % (GSE) / 4.7 % (ovarian) at m1; d → type NMI ≤ 0.16, d → cycle ≤ 0.05 (neither identity nor cycle); I(niche;w) excess unchanged across rungs. **So R19's per-cell channel is real and useful at the bound-equivalent weight; what it costs is 0.02–0.03 NMI and a doubling of cycle_w to just over the 0.02 guard on both slides at m1.**

**Weigh before accepting:** the ovarian reference band is the reused `wfix_warmup30_aw0.1_s{0,1}` pair with an implausibly narrow 2-seed spread (w-mirror sd 0.002, cycle_z sd 0.004), narrower than any rung's own spread; that m25, above the reference, fails the same clauses points at band width or at the reused fits, not at α_w. A fresh ovarian reference pair is ordered (15 min). Probe guard: rung `probe_blocks.json` not yet written; the reference itself fails `invariance_pass` on the composition blocks, so the switch can only add failures. ℓ̄ definition discrepancy recorded in `LADDER.json` (see 14:25 note). **Stage 2 (FF) not started; the author decides** whether to accept the rule ("keep 0.1, w is the context regression") or to weigh the held-out gain against an NMI/cycle_w guard that the ladder shows to be the least meaningful cost.
**Stage 1 re-rendered with a fresh ovarian reference pair (19:00).** The reused pair's tight band was chance: the fresh pair spans w-mirror 0.135–0.180, cycle_z 0.510–0.526, NMI 0.626–0.670. Under the fresh (and pooled) reference the ovarian cycle_z and w-mirror failures vanish; **the only clause blocking every rung below m25 is NMI, missed by 0.004–0.005 in the closest cases (m5 on GSE at 0.609 vs 0.613; m10 on ovarian at 0.623 vs 0.628)**; m25 fails GSE recon spread and shows no gain. The held-out gain of the deviation is unchanged and rises monotonically as α_w falls. The per-block probe (`invariance_pass`) fails on every run including the 0.1 reference, so it cannot separate rungs. **Verdict by the rule: no rung; α_w stays at 0.1.** For the author: the ladder's message is that a 5–10× multiplier (α_w ≈ 0.02–0.04) opens a per-cell channel that does real held-out work, at an NMI cost of 0.004–0.02 that sits on the pre-registered line and inside the reference's own seed spread. Accepting m5 or m10 would be a judgement against the letter of the rule, not against its evidence; declining keeps w as the context regression (R19's reframing). Stage 2 on FF is the natural next step if the author leans toward accepting.

### Final re-pin complete (8.11, results, 2026-09-24 17:42)

*12 fits, no dead channel, no failed step; `runs/best` → `final_s0` on all four datasets at 15:39; baseline columns re-scored against the final context. Envelopes at the accepted checkpoint, min / mean / max over 3 seeds (`experiments/envelope_table_at_best.md`):*

| | GSE core | ovary FF | ovarian FFPE | lung |
|---|---|---|---|---|
| recon (nats/count) | −7.210 / −7.205 / −7.201 | −7.322 / −7.315 / −7.311 | −7.220 / −7.181 / −7.140 | −7.256 / −7.243 / −7.230 |
| NMI | 0.618 / 0.626 / 0.633 | 0.609 / 0.628 / 0.647 | 0.644 / 0.651 / 0.658 | 0.663 / 0.665 / 0.667 |
| mirror R² | 0.030 | 0.054 | 0.040 | 0.026 |
| cycle_z pooled | 0.468 / 0.494 / 0.508 | 0.768 / 0.779 / 0.786 | 0.506 / 0.520 / 0.528 | 0.431 / 0.448 / 0.458 |
| cycle_w pooled | ≤ 0.013 | ≤ 0.004 | ≤ 0.018 | ≤ 0.006 |
| I(z;t)/H(t) | 0.77–0.82 | 0.82–0.83 | 0.78–0.84 | 0.76–0.78 |
| Read A own gap closed | 0.46 / 0.55 / 0.65 | 0.66 / 0.72 / 0.74 | 0.67 / 0.69 / 0.70 | 0.73 / 0.74 / 0.74 |
| twin margin own | 0.47 | 0.44 | 0.42 | 0.40 |
| atlas cross-seed cosine | **0.24 / 0.54 / 0.72** | 0.61 / 0.74 / 0.83 | 0.93 / 0.94 / 0.94 | 0.94 / 0.95 / 0.96 |

Against the α_z/2 pre-warm-up triple: recon within 0.005, NMI −0.01 to −0.03 (the stated warm-up cost), cycle_z within 0.02, transport reads within 0.03. **Watch item:** the GSE atlas is now unstable across seeds (axis-1 cosine 0.24–0.72, was 0.68–0.84); FF 0.61–0.83 (was 0.55–0.84). Warm-up opens w earlier and the GSE core (69k cells) may not pin one programme direction; the atlas on GSE should be read on the pooled triple or reported as unstable. Legacy pooled probe at floor everywhere; the per-block probe verdicts are in `probe_regrade_final.md` (8.16).

**8.15 stage 2 launched (author's steer, 2026-09-24 19:40): "avoid a dead latent; judge w by its own quality reads, with nuance".** Reading of stage 1 under that steer: at α_w 0.1 the per-cell part of w is closed (posterior on prior), while the context regression is alive (I(niche;w) 0.5 on every rung); at m1 the per-cell deviation is a fifth of w's variance and does held-out work but NMI and cycle_w give way; at m5–m10 (α_w 0.02–0.04) every guard stays inside the reference's own seed spread and the held-out gain is 3–10× the reference's. Stage 2: FF, rungs m5 and m10 plus the 0.1 reference, 2 seeds; and transport (`--read both --hvg 1000`) on the GSE and ovarian rung runs m5, m10 and the references, so the decision includes the transport headline. Decision quantity for w's quality, pre-stated: held-out gain of the deviation, Read A own-target gap closed, and I(niche;w) excess, against the guards NMI (within the reference seed spread), cycle_w ≤ 0.02, mirror, probe. Adoption remains the author's call on the completed table.

### Per-block probe re-grade, complete (8.16 step 2, results, 2026-09-24 20:00)

*Every decision input and every baseline graded at the accepted checkpoint against 500/40 α_a = 0 references (two seeds on GSE and ovarian, one on lung and FF). Tables `experiments/probe_regrade_*.md` per section; `scripts/logs/probe_regrade_2026-09-24/{AGENT_REPORT,baselines_all_sections,groups_summary}.md`. Legacy pooled ΔCE reproduced to 1e-7; 43 + 26 tests.*

**DisCell final, fraction of uncontrolled (3-seed range):** composition ridge / MLP — GSE 0.36–0.47 / 0.53–0.71; GSE dual 0.50–0.64 / 0.62–0.77; ovarian 0.31–0.55 / 0.34–0.56; lung 0.24–0.26 / 0.42–0.45; FF 0.18–0.19 / 0.23–0.26. Image blocks lower everywhere (0.08–0.60). **Guard (≤ 0.25 u on all four blocks): 0/3 seeds on four sections, 1/3 on FF, at the threshold.** In variance: composition 0.5–1.8 % (ridge), 2.7–6.4 % (MLP). Against resolVI: 0.19–0.60 (ridge), 0.25–0.83 (MLP). Uncontrolled DisCell ≈ resolVI on four sections, > resolVI on FF. SIMVI ≥ resolVI; MintFlow at floor under the MLP (type code).

**The finding that matters for what to do next: composition leakage is flat across every knob we have swept.** κ, α_w (sweep3 and the ladder), α_z, the collapse arms, query/prior and all nine leak-form arms sit at ridge 0.3–0.5 u / MLP 0.3–0.6 u on ovarian, within seed spread of each other. No existing hyperparameter moves it; only the adversary's capacity can (planted-world entry). On FF the collapse remedies pass 4/4 seeds and the control 1/4, but the difference is a single block at 0.21–0.27 u against a one-seed denominator — read as "at the threshold", not as a separation. Denominators carry 10–40 % noise (ovarian's two reference seeds differ by 40 %). No decision rule changes: the α_z and collapse verdicts are unchanged by the re-grade.

**Paper.** The invariance claim becomes: the adversary removes 30–80 % of the detectable composition information (per block, linear and nonlinear graders, against an uncontrolled fit of the same model), leaving 0.5–2 % (linear) to 3–6 % (nonlinear) of within-type composition variance in z; the remainder is the heads' capacity; uncontrolled DisCell leaks as much as resolVI. **The adversary-capacity ladder is the only lever left and is the author's call.**

Note: the author committed at 15:33 (`ea36e69`); the final probe code and all table/queue scripts of the afternoon are uncommitted on top.

### R12 test 2 — the leak's form does not move the response findings (results, 2026-09-24 20:14)

*Ovarian, final configuration, κ 0.1, 3 seeds per arm; control = the w-only warm-up runs; reads at the accepted checkpoint; `experiments/r12_arms.md`. All guards pass on every seed of every arm (no dead channel, I(niche;w) > 0, cycle_w ≤ 0.02, probe at floor). Post-clip mean κ_i = 0.1000 on the depth and density arms (normaliser 1.55 / 1.18; 2.1 % / 0.8 % of cells at the cap).*

| read | global (default) | depth | gene | density |
|---|---|---|---|---|
| recon | −7.181 | −7.197 | −7.187 | −7.190 |
| NMI | 0.651 | 0.656 | 0.637 | 0.651 |
| I(niche;w) excess | 0.64 | 0.61 | 0.62 | 0.61 |
| transport mean read, fraction of ceiling | 0.816 | 0.803 | 0.816 | 0.806 |
| Read A own gap closed | 0.714 | 0.729 | 0.735 | 0.727 |
| Read B (tumour band) own gap closed | 0.752 | 0.666 | 0.756 | 0.752 |
| response channel LR share (vs matched) | +0.12 | +0.10 | +0.11 | +0.11 |
| leak channel LR share (vs all other) | +0.26 | +0.26 | **+0.02** | +0.26 |
| GO programme 0: extracellular / membrane / nucleus | +0.12 / +0.12 / −0.12 | +0.12 / +0.11 / −0.11 | +0.11 / +0.10 / −0.11 | +0.12 / +0.11 / −0.11 |

**Verdict.** Every response finding — transport at both reads, the response channel's signalling-gene share, and the surface lean of B — is unchanged under a leak that scales with the donor's depth, with the donor's transcript density, or that favours extranuclear genes. "Stable across κ" becomes "stable across κ and its form". **The surface lean of B does not drop under the gene-tilted leak**, so it is not membrane-gene leakage that a flat κ left in w. The only read that moves is the leak channel's own LR share under the gene arm (+0.26 → +0.02), which is by construction: κ_g redistributes the leaked composition toward extranuclear genes and the leak channel no longer singles out LR genes — this is the check that the gene arm did what it says. Depth arm costs 0.016 nats of recon and lowers the tumour-band own gap (0.67 vs 0.75) with wide seed spread; density arm is indistinguishable from the default. **No arm approaches the "indefensible" line the author left unset; the default form stands, and the paper's leak-term paragraph can state assumptions (a) and (b) with these three sensitivity rows.**

**8.15 stage 2 interim (FF, 21:00):** m10 seed 0 NMI 0.548 against the FF reference 0.624 (bar 0.604) — a clear NMI failure on the slide where α_w is 143× bound-equivalent; the ladder agent was cut off by a session limit, its queue continues and writes `DECISION_AW.json` itself.

### Two training-side candidates, run in parallel before one re-pin (motivation, 2026-09-24 21:20)

**A. Adversary capacity (8.17; author: "definitely continue pushing").** The per-block probe leaves 0.5–2 % (ridge) / 3–6 % (MLP) of within-type composition variance in z on tissue and the same on a planted world where z ⟂ niche; reweighting alone does not move it; no swept hyperparameter moves it. Arms on ovarian at the final configuration, κ 0.1, 200/20, 2 seeds: control (6 head steps, width 64); steps ×2 (12); width ×2 (128); ensemble of 3 heads (encoder penalised on the mean excess); composition weight ×3 inside the head loss (new knob, default 1); steps ×2 + width ×2. **Evaluated by** the per-block probe (fraction of the 500/40 uncontrolled reference, ridge and MLP, composition and image), NMI, cycle_z, cycle_w, mirror, recon, I(niche;w) excess, transport Read A own gap. **Rule.** Adopt the cheapest arm whose MLP composition fraction falls below the control's by more than the control's seed range on both seeds with every guard inside the control envelope (one seed-sd) and recon not worse; if several, prefer steps ×2 (no architecture change), then width, then ensemble. **Wished for:** composition fraction toward ≤ 0.25 u with z untouched.

**B. Fixed false-positive floor (writer's proposal, todo 8.15b).** p_i = (1 − κ − η_i) ρ_i + κ ρ̄_i + η_i u, u even over the panel (stated approximation — genomic-DNA binding is not gene-uniform), η_i = λ_i / ℓ_i capped at 0.2, λ from the section's negative-control and genomic-control probe counts in the Xenium cell table (not the 10x summary; ovary FF's summary is internally inconsistent), genomic DNA counted. **Per-cell vs per-section decided by data:** if per-cell control counts correlate with segmented area (Spearman ≥ 0.3 within type), λ_i ∝ area with the section total fixed; else one λ per section. **Check:** with vs without on ovarian and GSE, κ 0.1, 3 seeds, final configuration; battery, transport, signalling share. **Expected outcome, written first: small changes** (≤ one seed-sd on every read); a larger change is a finding. Adopt if the check confirms "small" (it completes the model at no cost) — the decision is the author's on the table.

Both run now; whichever is adopted, the re-pin (8.11) is redone once with both.

**8.15b step 1 (control-probe rates, 21:31) and a rule defect.** λ per cell scaled to the panel: ovarian 6.57 (genomic control 6.30, neg-probe 0.25, neg-codeword 0.01) = 3.7 % of median counts; GSE core 1.73 (genomic 1.61) = 0.6 %. Deprecated codewords excluded (ovarian 22.9/cell). **The area rule as pre-registered (within-type Spearman ≥ 0.3) cannot pass on this data:** control counts are 97–99 % zeros, so even counts exactly proportional to area give Spearman 0.08 / 0.04; the observed values are 0.075 / 0.034. A Poisson regression of control counts on type + log area gives slope 0.98 (ovarian) and 0.65 (GSE): **ovarian false positives are proportional to area.** With one λ per section, 11.2 % of ovarian cells sit at the η = 0.2 cap (0.7 % on GSE); with area scaling 3.75 %. The queue launched with the per-section form per the letter of the rule; an ovarian area-scaled arm (`--fp-area`, 3 seeds) is added so the morning table shows both. Floor implemented as `mix + η(1/G − ρ)`, rows sum to 1, transport and the leak channel decode with the floor; 11 planted tests, 129 related tests pass. Fits queued behind the ladder's FF stage 2; read-out ~01:00.

**8.17 built and queued (21:41; relaunched 21:55 with a 600 s courtesy window on foreign locks, since the ladder's FF fits allocate ~4.5 min after locking).** Knobs `--adv-head-steps` (alias of the existing `--adv-steps`), `--adv-head-width` (alias of `--adv-hidden`), `--adv-ensemble K` (K head pairs, encoder penalised on the mean excess), `--adv-comp-weight c`; defaults bit-identical to a verbatim copy of the old code (terms, gradients, three trainer steps incl. Adam state; single-threaded — multithreaded CPU differs from itself by 6e-11). 29 tests. Arms `adv_{steps12,width128,ens3,comp3,steps12_width128}_s{0,1}` on ovarian, control = the warm-up pair (exact configuration); per run validate/probe, degeneracy, probe_regrade vs `uncontrolled500`, transport; head-step timing; `adv_table.py` applies the rule seed by seed (bars: MLP composition fraction below 0.436 / 0.306 for the two control seeds), preference steps12 → width128 → ens3, then comp3 and the combination by measured head time. Sensitivity against the fresh reference pair reported. ETA 01:00–02:00 on 2026-09-25.

**Hand-off 21:55 (author away).** Detached, parent PID 1: α_w ladder FF stage 2 (→ `DECISION_AW.json`), fp-floor check (→ `experiments/fp_floor.md`), adversary ladder (→ `experiments/adv_ladder.md`, `DECISION_ADV.json`). Morning read: the three tables decide the single remaining re-pin. Awaiting the author: α_w call (recommend keep 0.1 + R19 reframing), adoption of whichever of 8.17 / 8.15b pass, marker pairs, commit.

### Overnight results, 2026-09-25 morning: α_w ladder stage 2, adversary capacity, false-positive floor

**8.15 α_w ladder — complete; verdict: α_w stays at 0.1 (all three reference variants agree), R19's reframing adopted.** FF stage 2 (m5, m10 vs 0.1, seeds 0–1): NMI 0.545 / 0.547 vs 0.624 (−0.08), recon equal, cycle_w 0.010 / 0.005. On the slide where α_w 0.1 is 143× bound-equivalent, lowering it empties z's type structure. GSE/ovarian: only NMI blocks m5–m10, by 0.004–0.005. **Paper:** at the operating point w is the context regression m_ψ(c,t); the per-cell channel is closed by design; the ladder is the sensitivity analysis showing what opening it costs and buys (held-out gain 0.001–0.019 nats/count, deviation neither type nor cycle). `DECISION_AW.json`, `experiments/awladder.md`.

**8.17 adversary capacity — no arm qualifies by the rule; one arm is a consistent improvement.** MLP composition fraction of uncontrolled (2 seeds): control 0.50, **steps ×2 0.50 (no effect)**, width ×2 0.45, ensemble 3 0.43, **composition weight ×3 0.42**, steps+width 0.44. Only comp3 moves every probe block: ridge composition 0.405 → **0.235**, MLP image 0.29 → 0.23, ridge image 0.20 → 0.15; mirror 0.037 → **0.032**; Read A own gap 0.72 → **0.78**; NMI 0.647 (control 0.641); recon equal; cycle_z 0.519 vs 0.531 (the one read outside the control's narrow envelope); head time unchanged (×1.00). The rule's "falls by more than the control's seed range" bar was 0.13 because the reused control pair spans 0.44–0.57; against the fresh reference pair the same rule **adopts comp3**. **Reading:** more head capacity (steps) does nothing on tissue; wider heads and ensembles help a little; re-weighting composition against the image term is what moves composition leakage — so on tissue the residual was weighting, with the planted world's capacity ceiling underneath it (comp3 reached 0.42u, not 0.25u). **Follow-up launched:** comp3 and comp5 on a third ovarian seed and on GSE and FF (2 seeds each) vs their controls; adopt comp3 if it holds on the primaries with guards intact.

**8.15b false-positive floor — "small changes" held on GSE, not on ovarian.** λ = 3.7 % of median counts on ovarian (96 % genomic-DNA binding), 0.6 % on GSE. GSE: every read within one control seed-sd except NMI −0.008 (−1.8 sd). **Ovarian, per-section λ:** leak share of the between-niche shift 0.070 → 0.060 and response share 0.170 → **0.200** (+3.7 sd); leak channel LR share 0.237 → 0.173; Read A own gap 0.714 → 0.684 (−1 sd); mirror +0.003 (+2.3 sd), MLP image probe fraction 0.29 → 0.32; KL_w 0.95 on one seed (11 % of cells at the η cap). **Area-scaled λ_i:** same attribution shift (0.060 / 0.196), Read A unchanged (0.708), mirror +0.005 (+4.5 sd), cycle_z 0.517 vs 0.529 (−3 sd), KL_w normal. **Reading:** on the FFPE slide the floor re-attributes about one percentage point of the between-niche shift from the leak channel to the response channel and nudges the invariance reads the wrong way by 2–4 seed-sd (small in absolute terms: mirror 0.037 → 0.040–0.042). Not negligible, not large; and 96 % of λ rests on the even-spread assumption for genomic-DNA binding, which is the least justified part. **Recommendation: do not adopt as default; report as a sensitivity row** ("with a measured false-positive floor …") that answers the resolVI-background question with numbers. Author's call.

**Decisions (author, 2026-09-25 morning).** (1) α_w stays at 0.1; R19's reframing goes into the paper; the ladder is reported as the sensitivity analysis. (2) False-positive floor: **not adopted as default; kept as a reported sensitivity row** — with a measured floor (3.7 % of median counts on ovarian, 96 % genomic-DNA binding, even spread assumed) the leak share of the between-niche shift falls 0.070 → 0.060 and the response share rises 0.170 → 0.200, invariance reads move by 2–4 seed-sd (mirror 0.037 → 0.040), transport within one sd; on GSE (0.6 %) nothing moves. The flag `--fp-floor` (and `--fp-area`) stays default-off; runs `fp_s*`, `fp_area_s*` on ovarian and GSE are the sensitivity rows; the appendix leak-term paragraph gets this sentence beside the κ-form rows. (3) Adversary composition weight: confirmation running (third ovarian seed, GSE, FF; weights 3 and 5).
