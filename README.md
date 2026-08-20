# DisCell

Preprocessing for spatial transcriptomics: raw counts, cell polygons, and
neighbour graphs with real geometric edge weights, ready to feed a model.

Built for 10x **Xenium** (Prime 5K).

```bash
uv sync --extra viz        # viz extra adds pyqt6 + tornado for interactive viewing
uv run python -m discell.paths     # create the data tree and print what is in it
```

## Where things live

Every input and output sits under one root, organised **by dataset**, so
commands need no path arguments. `discell/paths.py` is the single source of
truth; point `DISCELL_DATA` elsewhere to move the whole tree
(`export DISCELL_DATA=/scratch/discell`).

```
data/
  raw/          source cohorts -- SYMLINKS to wherever they were downloaded.
                Never copied: the Xenium cohort is 55 GB.
  interim/      unpacked per-sample outputs (interim/xenium -> .../extracted)
  models/       downloaded weights + marker metadata -- SHARED, one checkpoint
                embeds every slide
  datasets/
    <dataset_id>/
      dataset.json   provenance: platform, source path, cell count, µm/px
      bundle/        full.h5ad + full_polygons.parquet + full_edges_*.parquet
      embeddings/    full_v1.pt, kronos2_4ch.pt, ...
      figures/
      logs/
```

`raw` and `interim` are inputs, `datasets/` holds every output, and the whole
tree is gitignored.

### Why per-dataset

The two cohorts hold 46 candidate samples. In a flat tree an artefact's dataset
is only guessable from a filename prefix, so as soon as a second sample is
processed either the names collide and one silently overwrites the other, or a
run pairs one slide's bundle with another's embeddings. Neither failure raises.
Putting the dataset in the *path* makes both impossible, and frees the filename
to describe the variant instead — `full_v1.pt`, not
`ovarian_full_v1.pt`.

On top of that, artefacts are self-describing: every bundle carries
`uns["dataset_id"]` and every `.pt` a `dataset` key, and the loader refuses a
pair that disagrees rather than warning about zero-filled rows.

`dataset.json` records what identifies the slide once, and what varies per
bundle under `bundles`, so a 3k-cell dev bundle cannot overwrite what the full
one reported:

```json
{
  "dataset_id": "xenium_prime_ovarian_cancer_ffpe",
  "platform": "xenium",
  "source": "/…/10x_Xenium/extracted/Xenium_Prime_Ovarian_Cancer_FFPE",
  "microns_per_pixel": 0.2125,
  "modality": "cells",
  "bundles": {
    "full": {"n_cells": 407120, "n_features": 5101, "default_label": "cell_group"},
    "dev":  {"n_cells": 3000,   "n_features": 5101, "default_label": "cell_group"}
  }
}
```

A second export that disagrees on `platform`, `source` or `modality` is
refused — two samples landing on one id would leave the record describing slide
B while `bundle/` still held slide A. Paths are compared resolved, so the
symlinked `data/raw/…` and the real location count as the same source.

### Dataset ids

A `dataset_id` is the slug of the source sample directory, so every spelling of
one slide resolves to the same outputs:

```
Xenium_Prime_Ovarian_Cancer_FFPE                 -> xenium_prime_ovarian_cancer_ffpe
Xenium_Prime_Ovarian_Cancer_FFPE/outs            -> xenium_prime_ovarian_cancer_ffpe
Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip  -> xenium_prime_ovarian_cancer_ffpe
Xenium_Prime_Human_Lung_Cancer_FFPE_outs.zip     -> xenium_prime_human_lung_cancer_ffpe
```

Commands take `--dataset <id>`, infer it from `--sample <path>`, or — when only
one dataset has a bundle — infer it from the tree. They never guess between two:

```bash
uv run python -m discell.paths                        # what the tree holds
uv run python -m discell.main --embeddings full_v1    # dataset inferred
uv run python -m discell.main --dataset xenium_prime_ovarian_cancer_ffpe --graph contact
uv run python -m discell.preprocess --sample data/interim/xenium/Xenium_Prime_Human_Lung_Cancer_FFPE
```

Within a dataset, artefacts are named rather than pathed:
`--embeddings full_v1` resolves to that dataset's `embeddings/full_v1.pt`. A dataset can hold several bundle variants
(`--variant full`, `--variant tol2`) for different graph settings.

To add a cohort, symlink it in rather than copying:

```bash
ln -s /path/to/10x_Xenium data/raw/xenium
```

The one artefact deliberately kept *outside* this tree is the HVG panel
(`hvg_top_1000_square_008um.json`), which is cached next to the raw sample so it
can be handed to `--hvg-file` when processing a different sample with the same
gene panel.

The package splits along the artefacts: `preprocess` produces them, `data`
consumes them, and nothing in `data` imports from `preprocess`.

| module | holds |
|---|---|
| **`discell.preprocess`** | **the one preprocessing command** — raw slide to bundle, embeddings and figures |
| `discell.preprocess.xenium` | reading a slide: cells, boundaries, labels |
| `discell.preprocess.geometry` | polygons → neighbour graphs and edge metrics |
| `discell.preprocess.labels` | 10x's supplemental cell/gene/cluster annotations |
| `discell.preprocess.bundle` | writing a bundle |
| `discell.preprocess.crops` | per-cell image crops |
| `discell.preprocess.markers` | which protein each image channel is |
| `discell.preprocess.kronos` | per-cell image embeddings (v1/v2) |
| `discell.preprocess.embedding_qc` | does the embedding know anything? |
| `discell.preprocess.plotting` | slide-scale figures: graph, crops, comparisons |
| | |
| `discell.data.loader` | GPU-resident subgraph batches |
| `discell.data.batch` | what one batch carries — the model-facing contract |
| `discell.data.priors` | the fixed per-dataset arrays, and `dataset.constant` |
| `discell.data.bundle` | reading a bundle |
| `discell.data.embeddings` | reading a `.pt` of per-cell vectors |
| | |
| `discell.tiff` | the one place that touches `tifffile` |
| `discell.paths` | where artefacts live, keyed by dataset |
| `discell.main` | inspect one batch |

Only three entry points are runnable: `discell.preprocess` (produce),
`discell.main` (inspect) and `discell.examples.train_demo` (a worked example).
Everything else is a library.
---

## Quickstart — the whole pipeline

One command takes a raw slide to training batches. It writes into
`data/datasets/<dataset_id>/` and each stage finds its own inputs there.

```bash
S=data/interim/xenium/Xenium_Prime_Ovarian_Cancer_FFPE

uv run python -m discell.preprocess --sample $S --figures
uv run python -m discell.main --embeddings full_v1 --batch-cells 10   # inspect
```

```
DATASET  xenium_prime_ovarian_cancer_ffpe  (bundle 'full')

  [1/3] bundle     RUN   407,120 cells x 5,101 genes, 5 files, 1104 MB  [881.4s]
  [2/3] embed      RUN   407,120 x 384 -> full_v1.pt                    [664.3s]
  [3/3] figures    RUN   5 figure(s)                                    [312.7s]
```

**Every stage checks for its own output and skips when it is there**, so
re-running after a failure, or to add embeddings to an existing bundle, costs
nothing for the parts already done. `--force` redoes everything, `--force-from
embed` redoes that stage onward, `--only bundle` runs just the one.

The whole slide takes ~30 minutes, so run it detached — the console may close:

```bash
setsid nohup uv run python -m discell.preprocess --sample $S \
    > data/datasets/xenium_prime_ovarian_cancer_ffpe/logs/preprocess.log 2>&1 < /dev/null &
```

Common variations:

```bash
# a small bundle for iterating, with figures
uv run python -m discell.preprocess --sample $S --variant dev --max-cells 3000 --figures

# KRONOS2 instead of v1, named for the variant it is
uv run python -m discell.preprocess --sample $S --model v2 --embeddings-name kronos2_4ch

# a DAPI-only baseline over 2000 cells
uv run python -m discell.preprocess --sample $S --only embed \
    --channels 0 --limit 2000 --embeddings-name dapi_only

# a tighter contact graph kept beside the default one
uv run python -m discell.preprocess --sample $S --variant tol2 --contact-tolerance-um 2
```

---

## 1. Reading a slide — `discell.preprocess.xenium`

### Two graphs, because they answer different questions

Measured on `Xenium_Prime_Ovarian_Cancer_FFPE`, 407,120 cells, with a
2 µm contact tolerance:

| graph | edges | mean degree | isolated | median shared wall |
|---|---|---|---|---|
| `contact` | 719,440 | 3.53 | 6.6% | 0.00 µm |
| `voronoi` | 1,197,730 | 5.88 | 0.1% | 6.85 µm |

**`contact`** — cells whose boundaries touch, within a tolerance. Xenium
boundaries approach each other but almost never coincide exactly, so a strict
zero-distance rule collapses the graph entirely (mean degree 0.14, 88% isolated
on the lung slide). Even at 2 µm, mean degree is 3.53 where planar geometry
demands ~6, and 6.6% of cells touch nothing.

**`voronoi`** — an exact planar partition seeded from the centroids. Every
adjacent pair shares exactly one edge of positive length, so shared-wall length
is always defined. `--clip-radius-um` caps how far a cell claims territory.

Filter on `wall_dist_um` — or better `apposed_wall_um` — to recover contact
semantics from the tessellation.

### Edge metrics

Stored in `obsp` as symmetric sparse matrices, all in microns:

| key | meaning |
|---|---|
| `{graph}_connectivities` | adjacency |
| `{graph}_shared_wall_um` | length of shared boundary |
| `{graph}_centroid_dist_um` | centroid-to-centroid distance |
| `{graph}_wall_dist_um` | min distance between the **original** polygon boundaries |

`wall_dist_um` is measured between the real polygons in both graphs, so it keeps
physical meaning even when adjacency comes from the partition.

```python
from discell.data.bundle import load_bundle
from discell.preprocess.geometry import graph_edge_frame

adata = load_bundle("data/datasets/<id>/bundle", "full")
edges = graph_edge_frame(adata, "voronoi")          # tidy edge list
touching = edges[edges.wall_dist_um == 0]           # genuinely in contact
```

### Per-cell geometry in `obs`

`centroid_x_px/y_px`, `centroid_x_um/y_um`, `rep_x_px/y_px`, `area_um2`,
`perimeter_um`, `equiv_diameter_um`, `circularity`, `solidity`,
`contact_degree`, `voronoi_degree`.

**Use `rep_*`, not `centroid_*`, to anchor image crops.** A centroid can fall
outside its own outline for a sufficiently concave cell — 2 of 40,222 on the
mouse brain, at solidity ≈0.73. `rep_*` is guaranteed interior.

---

### Xenium specifics

Reads an extracted `outs` directory or a `*_outs.zip` — only the tabular members
are unpacked, since the morphology images and transcript tables are tens of GB
and not needed for graphs. Boundaries are long-form parquet in microns;
coordinates are converted to image pixels on load using `pixel_size` from
`experiment.xenium`, with `uns['microns_per_pixel']` converting back.

```bash
uv run python -m discell.preprocess --sample <outs-dir> --only bundle
```

### Xenium cells barely touch — use a tolerance

Segmentation is *measured* (97.8% from interior or boundary stains, 2.2% nucleus
expansion), so it might be expected to tile the tissue. It does not. Boundaries
come very close without coinciding — a quarter of neighbouring pairs sit within
**0.12 µm**, below the 0.2125 µm pixel — yet only 2.5% touch at exactly zero, so
exact-contact adjacency collapses. Tolerance recovers it (lung, 20k cells):

```
tol 0.0 um ->  1,448 edges  mean_deg 0.14  isolated 87.7%
tol 0.5 um -> 23,937 edges  mean_deg 2.39  isolated 12.7%
tol 1.0 um -> 25,960 edges  mean_deg 2.60  isolated 10.8%
tol 2.0 um -> 30,203 edges  mean_deg 3.02  isolated  7.6%
tol 5.0 um -> 39,139 edges  mean_deg 3.91  isolated  3.4%
```

`XENIUM_CONTACT_TOLERANCE_UM = 1.0` is therefore the default. The three variants
side by side, same sample:

```
                 variant  edges  mean_degree  isolated_%  shared_wall_um  centroid_um  touching_%
   contact (exact, 0 µm)   1448         0.14        87.7            0.00        10.22       100.0
contact (tolerance 1 µm)  25956         2.60        10.8            0.00         7.92         5.6
    voronoi (clip 30 µm)  58206         5.82         0.1            7.16        11.73         2.5
```

`shared_wall_um = 0.00` for **both** contact variants: on Xenium, touching pairs
meet at a point rather than along a wall, so shared-wall length carries no
information there. If you need it as a feature, Voronoi is the only variant that
provides it on this platform — which is what `apposed_wall_um` exists to fix.

At full scale the lung slide has 278,324 cells, 816,949 voronoi edges (mean
degree 5.87), and a contact graph of 344,751 edges at mean degree 2.48 with
10.8% isolated — still well short of a tessellation.

### Supplemental cell-type labels

Some Xenium datasets publish curated annotations as separate downloads, not
inside `outs.zip`. Drop them next to the outs and they are picked up
automatically:

```bash
set B https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun
curl -O $B/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_cell_groups.csv    # 16.5 MB
curl -O $B/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_gene_groups.csv    # 3.9 kB
```

`*_cell_groups.csv` (`cell_id,group,color`) becomes `obs['cell_group']` and
`obs['cell_group_color']`; when present it **outranks graphclust** as the default
label and its published palette drives plotting. `*_gene_groups.csv` becomes
`var['gene_group']`, with the many-to-many mapping in `uns['gene_groups']`.

On the ovarian sample: 18 named types over 406,611 of 407,120 cells.

Select with `--label-key cell_group`, or leave it to the default.

### Comparing graph definitions

`--figures compare` builds all three variants from a single load and renders
them side by side over a shared region, with the statistics table above.

```bash
uv run python -m discell.preprocess --sample <dir> --variant dev \
    --max-cells 20000 --figures compare
```

---

## 2. Figures — `--figures`

Pass `--figures` for the default set, or name the kinds you want:

```bash
uv run python -m discell.preprocess --sample <dir> --figures
uv run python -m discell.preprocess --sample <dir> --figures graph,compare
```

| kind | what it draws |
|---|---|
| `graph` | polygons filled by cell type, the neighbour graph with edge width and colour by apposed wall, over the tissue image and on plain background |
| `crops` | the exact 256 px window KRONOS is fed for one cell, channel by channel |
| `qc` | UMAP of the image embeddings coloured by cell type, with separation statistics |
| `compare` | the three graph definitions side by side over a shared region |

Default is `graph,crops,qc`; `compare` is opt-in because it rebuilds all three
graphs. The metric and every filter are encoded in each filename, so variants
never overwrite each other. Figures land in `data/datasets/<id>/figures/`.

The drawing itself lives in `discell.preprocess.plotting` and takes far more
options than the flag exposes — edge metric, colour map, region window, gap and
wall filters. Call `render()` directly when you want them:

```python
from discell.data.bundle import load_bundle
from discell.preprocess.plotting.render import render

adata = load_bundle("data/datasets/<id>/bundle", "full")
render(adata, sample_dir, out_dir, graph="contact", edge_metric="wall_dist_um",
       color_edges_by_metric=True, edge_cmap="magma", max_gap_um=0,
       region=(16800, 18200, 7500, 8900))
```

For an interactive pan/zoom view instead of a file, see §7.

## 3. Bundling a run — the `bundle` stage

A bundle is one reloadable analysis run: the counts, the polygons, both graphs
and every edge metric, plus a provenance record. Building the graphs is the
expensive part, so it is done once here and never again.

```bash
# whole slide, default tolerances -> data/datasets/<id>/bundle/full.*
uv run python -m discell.preprocess --sample <dir-or-archive> --only bundle

# a second bundle of the same slide with a tighter graph, kept side by side
uv run python -m discell.preprocess --sample <dir> --variant tol1 \
    --contact-tolerance-um 1.0 --wall-tolerance-um 0.5

# a small bundle for iterating
uv run python -m discell.preprocess --sample <dir> --variant dev --max-cells 20000
```

**Every edge the graph builder found is written.** Nothing is trimmed on the way
in: the metrics needed to filter a graph all travel with it, so a consumer can
take whatever subset it wants without the bundle having thrown the rest away.

Five files per bundle:

| file | holds |
|---|---|
| `<variant>.h5ad` | counts, `obs`, `obsm['spatial']`, both graphs in `obsp` |
| `<variant>_polygons.parquet` | cell id + polygon as WKB |
| `<variant>_edges_contact.parquet` | one row per contact edge, all metrics, µm |
| `<variant>_edges_voronoi.parquet` | the same for the Voronoi graph |
| `<variant>_params.json` | sample path, tolerances, counts, dataset id, modality |

`--variant` exists because one slide can yield several bundles — different graph
tolerances, or a small subset for iterating.

---

## 4. Image embeddings — the `embed` stage

[KRONOS](https://huggingface.co/MahmoodLab/KRONOS) is a multiplex-imaging
foundation model: it projects **each channel separately** and adds a per-marker
embedding, so it takes Xenium's 4-channel morphology stack directly. Both
releases are wired up and selected with `--model`:

| | v1 (`--model v1`) | v2 (`--model v2`) |
|---|---|---|
| architecture | ViT-S/16 | DINOv2 ViT-B/16 |
| dimension | 384 | 768 |
| markers identified by | integer ids | names, novel ones registrable |
| throughput | 637 cells/s | 122 cells/s |
| separation / 1-NN on ovarian | +0.4184 / 36.2% | +0.4166 / 36.2% |

They score the same, so **v1 is the default**: half the width, five times the
speed.

```bash
# v1, all four channels, whole slide -> data/datasets/<id>/embeddings/full_v1.pt
uv run python -m discell.preprocess --sample $S --only embed

# v2 -> full_v2.pt
uv run python -m discell.preprocess --sample $S --only embed --model v2

# a DAPI-only baseline, and a quick 2000-cell check
uv run python -m discell.preprocess --sample $S --only embed \
    --channels 0 --embeddings-name dapi_only
uv run python -m discell.preprocess --sample $S --only embed \
    --limit 2000 --embeddings-name probe
```

The output is named `<variant>_<model>` unless `--embeddings-name` says
otherwise. When a bundle already exists this stage reloads it rather than
re-reading the slide, which skips the graph build the image side has no use for.

The weights are gated: request access, then `--hf-token` or `HF_TOKEN`. v1
caches in `data/models/` and is shared by every dataset. **v2's checkpoint does
not** — the vendor's remote code calls `snapshot_download` without a `cache_dir`,
so it lands in `~/.cache/huggingface` regardless of `--cache-dir`.

**Markers.** Xenium's four channels map onto KRONOS's 177-marker vocabulary as
DAPI (exact), Na/K-ATPase and αSMA (antibody cocktails, named for their broadest
component), and 18S rRNA — for which the vocabulary has no RNA marker at all. v1
gives it an out-of-vocabulary id rather than borrowing another protein's; v2
registers it as a novel marker from statistics measured on the actual crops.

**Intensity scaling matters more than the model choice.** KRONOS expects
intensities in roughly `[0,1]` and its own loader divides by the dtype maximum.
Xenium morphology is uint16, so feeding raw values puts the z-score four orders
of magnitude out and every patch outside the pretraining distribution. Fixing
this moved v1 from +0.2724 / 24.1% to +0.4184 / 36.2% — a larger gain than
anything else tried. Files written since record `intensity_scaled: True`.

Each `.pt` holds `embeddings`, `cell_ids`, the marker set, and a `dataset` key
naming the slide it came from — which the dataloader checks.

### Does the embedding carry cell identity? — `--figures qc`

```bash
uv run python -m discell.preprocess --sample $S --figures qc --per-label 150
```

Reports within- vs between-class cosine and 1-NN label agreement, then plots the
UMAP. Both are computed on **centred** vectors: raw KRONOS embeddings sit in a
narrow cone (all pairwise cosines ≈ 0.995) that swamps the between-class
structure. The pairwise matrix is subsampled above 6000 cells — at 407k it would
be 0.7 TB — and says so when it does.

---

## 5. The dataloader — `discell.data.loader`

Serves subgraph batches for a model that predicts a cell from its neighbourhood.
Everything is held resident on the GPU, so a batch is an `index_select` with no
host-to-device copy.

A batch picks `batch_cells` **seeds**, pulls their 1-hop neighbours, and serves
only the directed **neighbour → seed** edges. Fields by the axis they are
aligned to:

| aligned to | fields | size |
|---|---|---|
| seeds | `image_embedding`, `seed_index`, `seed_composition` | fixed |
| nodes | `x`, `y`, `pos`, `node_ids`, `node_index` | varies |
| edges | `edge_index` (2, E), `edge_attr` (E, 3), `edge_id`, `into_j` | varies |

`x` is raw integer counts; `edge_attr` is `[centroid_dist_um, apposed_wall_um,
wall_dist_um]`. `pos` and `node_ids` are auxiliary — for plotting and debugging,
not model inputs. The spatial information a model should consume is already
reduced into `edge_attr`, and absolute slide coordinates would let a model
memorise position.

**A batch carries only what varies with it.** The fixed per-dataset arrays live
on `dataset.constant` and are read from there rather than copied onto every
batch:

| on `dataset.constant` | shape | meaning |
|---|---|---|
| `neighbour_composition` | (K, K) | mean neighbour-type mix **per cell type** |
| `niche` | (N, K) | **each cell's own** observed neighbour mix |
| `beta` | (E, 2) | row-stochastic smoothing weight over the whole graph |

`edge_id` is what connects the two: the row each batch edge occupies in the
whole-slide edge list, with `into_j` recording its direction, so any per-edge
array on the dataset can be gathered for a batch.

```python
beta = data.constant.edge_beta(batch)        # (n_edges,) aligned to edge_index
table = data.constant.neighbour_composition  # (K, K)
```

**The default graph is `voronoi`.** It is an exact planar partition, so every
edge has a positive shared face and `beta` is well defined. On `contact`,
Xenium's segmented polygons meet at a point rather than along a wall, which
leaves `shared_wall_um` zero on 98% of edges and `beta` all-zero for 95% of
cells.

In Python:

```python
from discell.data.loader import CellGraphDataset

data = CellGraphDataset.from_dataset(
    "xenium_prime_ovarian_cancer_ffpe",
    embeddings="full_v1",             # resolved inside that dataset
    batch_cells=64, resident="gpu",   # graph defaults to voronoi
)
for batch in data.batches():
    ...
```

Embeddings are checked against the bundle. A `.pt` recording a different dataset
raises, and so does one sharing no cell ids — previously that only warned about
zero-filled rows and trained on fabricated vectors.

---

## 6. Inspecting a batch — `discell.main`

Loads one batch, prints every field with its shape and meaning, runs eight
structural checks, and writes a four-panel figure.

```bash
uv run python -m discell.main --embeddings full_v1 --batch-cells 10
uv run python -m discell.main --dataset <id> --graph contact --image-scope nodes
uv run python -m discell.main --no-plot            # text only
```

The checks are assertions about batch structure, not statistics: every edge
points into a seed, no self-loops, endpoints are valid node rows, counts are raw
integers, labels are one-hot, composition rows sum to 1, `edge_attr` distances
match `pos`, and `constant.edge_beta` lines up with `edge_index`. The same
assertions run under pytest — see §8.

The figure combines label distribution, the batch graph in slide coordinates
(edge width = apposed wall, node colour = cell type), a cosine-distance matrix
over the image embeddings, and per-type neighbourhood composition bars —
alongside a whole-tissue panel placing the seeds on the slide.

### Training demo — `discell.examples.train_demo`

A minimal end-to-end sanity check that the batches train something:

```bash
uv run python -m discell.examples.train_demo --embeddings full_v1 --steps 300
```

---

## 7. Viewing over SSH

`discell.preprocess.plotting.show` opens an interactive pan/zoom view over a
bundle. It is a library call rather than a CLI — the figure options are too many
to flag usefully:

```python
# view.py
from discell import paths
from discell.data.bundle import load_bundle
from discell.preprocess.plotting.show import show_interactive

ds = paths.dataset("xenium_prime_ovarian_cancer_ffpe")
adata = load_bundle(ds.bundle_dir, "full")
show_interactive(adata, sample_dir=adata.uns["xenium_dir"], graph="voronoi",
                 label_key=adata.uns["default_label"],
                 edge_metric="apposed_wall_um", color_edges_by_metric=True)
```

Backend selection is automatic — QtAgg → TkAgg → WebAgg, with the display probed
via `xdpyinfo` first.

```bash
ssh -X user@host
uv run --extra viz python view.py
```

If X is slow over a WAN — likely at this figure size — skip X entirely by
passing `backend="WebAgg"`, then forward the port:

```bash
ssh -L 8988:localhost:8988 user@host
uv run --extra viz python view.py     # open http://localhost:8988
```

Two things that cost time to find:

- **TkAgg does not work under uv-managed CPython.** `import tkinter` succeeds,
  but `_tkinter` is statically linked and exposes no `__file__`, which
  matplotlib's `_tkagg` needs to locate Tcl/Tk. Hence `pyqt6` in the `viz` extra.
- **A dead `$DISPLAY` aborts Qt outright** rather than raising, so it cannot be
  caught — the display is probed with `xdpyinfo` first.

For the static PNGs, ImageMagick's `display` ships uncompressed pixmaps — about
156 MB per repaint at full slide. Always downsample:

```bash
display -resize 1800x data/datasets/<dataset_id>/figures/<name>.png
```

---

## 8. Tests

```bash
uv run --group dev pytest -q
```

The suite builds a real 1,500-cell bundle into a temporary `DISCELL_DATA` root
and asserts against it, so it covers the preprocessing entry point as well as
the loader. It skips rather than fails when the raw slide is not present, and
does not need the gated KRONOS weights — the batch contract holds whether the
image vectors are real or zero-filled.

| file | asserts |
|---|---|
| `tests/test_ingest.py` | polygons align with expression, representative points lie inside their outline, areas are plausible in microns, both graphs are symmetric, edge metrics are in range, Voronoi faces are strictly positive, WKB round-trips |
| `tests/test_batch_contract.py` | every field's shape and dtype, edges point into seeds, counts are raw, labels are one-hot and have no case-duplicate class, `edge_beta` matches a direct index and is row-stochastic, one epoch seeds every cell exactly once, residency changes nothing |
| `tests/test_preprocess.py` | the bundle writes all five files, params record the settings, the manifest is per-variant, a re-run skips |

---

## Notes on the data

- **Coordinates.** Polygons and `obsm['spatial']` are in full-resolution image
  pixels; `microns_per_pixel` comes from `experiment.xenium`.
- **Tissue images vary.** Tiled vs strip-based, uncompressed vs JPEG. Reading
  goes through `tifffile`; strip-based pages must be fully decoded, and are
  cached across figures in one run.
- **Node set mismatch.** A polygon exists per segmented object, but low-count
  cells are dropped from the filtered matrix. The loader keeps the intersection.
