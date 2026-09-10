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
