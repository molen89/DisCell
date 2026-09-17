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
