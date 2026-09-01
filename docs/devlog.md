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
