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
      embeddings/    kronos1_v2scale.pt, kronos2_4ch.pt, ...
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
to describe the variant instead — `kronos1_v2scale.pt`, not
`ovarian_kronos1_v2scale.pt`.

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
uv run python -m discell.paths                   # what the tree holds
uv run python -m discell.main --embeddings kronos1_v2scale   # dataset inferred
uv run python -m discell.main --dataset xenium_prime_human_lung_cancer_ffpe --graph voronoi
uv run python -m discell.data.export --sample data/interim/xenium/Xenium_Prime_Human_Lung_Cancer_FFPE
```

Within a dataset, artefacts are named rather than pathed:
`--embeddings kronos1_v2scale` resolves to that dataset's
`embeddings/kronos1_v2scale.pt`. A dataset can hold several bundle variants
(`--variant full`, `--variant tol2`) for different graph settings.

To add a cohort, symlink it in rather than copying:

```bash
ln -s /path/to/10x_Xenium data/raw/xenium
```

The one artefact deliberately kept *outside* this tree is the HVG panel
(`hvg_top_1000_square_008um.json`), which is cached next to the raw sample so it
can be handed to `--hvg-file` when processing a different sample with the same
gene panel.

| module | holds |
|---|---|
| `discell.data.xenium` | reading a slide: cells, boundaries, labels, images |
| `discell.data.geometry` | polygons → neighbour graphs and edge metrics |
| `discell.data.export` | bundles: save and reload one analysis run |
| `discell.data.crops` | per-cell image crops |
| `discell.data.loader` | GPU-resident subgraph batches |
| `discell.images.kronos` | per-cell image embeddings (KRONOS v1/v2) |
| `discell.plotting.*` | figures |
| `discell.paths` | where artefacts live, keyed by dataset |
---

## Quickstart — the whole pipeline

Four commands take a raw slide to inspectable training batches. Each writes into
`data/datasets/<dataset_id>/`, and each later stage finds its inputs there
without being told where they are.

```bash
S=data/interim/xenium/Xenium_Prime_Ovarian_Cancer_FFPE

# 1. bundle: cells, polygons, both graphs, all edge metrics   (~15 min, 407k cells)
uv run python -m discell.data.export --sample $S

# 2. image embeddings: one KRONOS vector per cell             (~11 min on a GPU)
uv run python -m discell.images.kronos --sample $S --out kronos1.pt

# 3. does the embedding know anything? UMAP + separation stats
uv run python -m discell.images.embedding_qc --sample $S \
    --embeddings kronos1 --per-label 0 --out kronos1_umap.png

# 4. inspect what the dataloader serves
uv run python -m discell.main --embeddings kronos1 --batch-cells 10

uv run python -m discell.paths        # what the tree holds, any time
```

Stage 1 sets the dataset id from `--sample`; stages 2–4 infer it. Steps 1 and 2
are long — run them detached, since the console may close:

```bash
setsid nohup uv run python -m discell.data.export --sample $S \
    > data/datasets/xenium_prime_ovarian_cancer_ffpe/logs/export.log 2>&1 < /dev/null &
```

---

## 1. Reading a slide — `discell.data.xenium`

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
from discell.data.xenium import load_sample
from discell.data.geometry import graph_edge_frame

adata, sample_dir = load_sample("<sample dir>")
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

Reads an extracted Xenium `outs` directory or a `*_outs.zip` (only the tabular
members are extracted — the morphology images and transcript tables are tens of
GB and are not needed for graphs).

```bash
uv run python -m discell.data.xenium --self-test --sample <outs-dir>
uv run python -m discell.data.xenium --sample <outs-dir> --tessellation-report
uv run python -m discell.data.xenium --sample <outs-dir> --out cells.h5ad --edges-out edges.csv

# iterate on a 280k-cell slide using a contiguous central subset
uv run python -m discell.data.xenium --sample <outs-dir> --max-cells 20000
```

Differences handled on load: boundaries are long-form **parquet** (one row per
vertex) rather than GeoJSON, and coordinates are in **microns**. Geometry is
converted to image-pixel coordinates using `pixel_size` from `experiment.xenium`
so everything downstream is uniform, with `uns['microns_per_pixel']`
carrying the conversion back.

### Xenium cells barely touch — use a tolerance

Xenium segmentation is *measured* (`experiment.xenium` reports 97.8% from
interior or boundary stains, 2.2% nucleus expansion), so it might be expected to
tile the tissue. It does not. Measured on the Prime 5K lung sample, 20k cells:

```
graph          edges  mean_deg  isolated  wall_um  touching
contact        1,448      0.14     87.7%     0.00    100.0%
voronoi       58,200      5.82      0.1%     7.16      2.5%
```

Boundaries come very close without coinciding — a quarter of neighbouring pairs
sit within **0.12 µm**, below the 0.2125 µm pixel — yet only 2.5% touch at
exactly zero. So exact-contact adjacency collapses. Tolerance recovers it:

```
tol 0.0 um ->  1,448 edges  mean_deg 0.14  isolated 87.7%
tol 0.5 um -> 23,937 edges  mean_deg 2.39  isolated 12.7%
tol 1.0 um -> 25,960 edges  mean_deg 2.60  isolated 10.8%
tol 2.0 um -> 30,203 edges  mean_deg 3.02  isolated  7.6%
tol 5.0 um -> 39,139 edges  mean_deg 3.91  isolated  3.4%
```

At 1 µm the contact graph reaches mean degree 2.60, so
`XENIUM_CONTACT_TOLERANCE_UM = 1.0` is the default. A strict zero-distance rule
would leave 88% of cells isolated.

Both Xenium samples tested, and the lung sample at full scale:

| sample | cells | features | median cell | voronoi edges | voronoi deg |
|---|---|---|---|---|---|
| Human Lung Cancer FFPE (full) | 278,324 | 5,001 | 7.9 µm | 816,949 | 5.87 |
| Human Lung Cancer FFPE (20k subset) | 20,000 | 5,001 | 7.9 µm | 58,200 | 5.82 |
| Ovarian Cancer FFPE (20k subset) | 20,000 | 5,101 | 9.7 µm | 59,446 | 5.94 |

At full scale the contact graph (1 µm default tolerance) has 344,751 edges,
mean degree 2.48, 10.8% isolated — still well short of a tessellation.

### Supplemental cell-type labels

Some Xenium datasets publish curated annotations as separate downloads, not
inside `outs.zip`. Drop them next to the outs and they are picked up
automatically:

```bash
set B https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun
curl -O $B/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_cell_groups.csv    # 16.5 MB
curl -O $B/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_gene_groups.csv    # 3.9 kB
```

`*_cell_groups.csv` is `cell_id,group,color` and becomes `obs['cell_group']` plus
`obs['cell_group_color']`; when present it **outranks graphclust** as the default
label, and its published palette is used for plotting. `*_gene_groups.csv` is
`gene,group` and becomes `var['gene_group']`, with the full many-to-many mapping
in `uns['gene_groups']` (a marker gene may belong to several groups, so `var`
keeps the first).

On the ovarian sample this gives 18 named types over 406,611 of 407,124 cells:

```
Tumor Cells 103,607 · Smooth Muscle Cells 61,545 · Proliferative Tumor Cells 50,118
SOX2-OT+ Tumor Cells 39,355 · Tumor Associated Fibroblasts 39,332 · Macrophages 27,415
Stromal Associated Fibroblasts 24,604 · Stromal Associated Endothelial 18,249 ...
```

Plot with them via `--label-key cell_group` (or leave it to the default).

### Comparing graph definitions

`discell.plotting.compare_graphs` builds all three variants from a single load
and renders them side by side over a shared region, with a statistics table.

```bash
uv run python -m discell.plotting.compare_graphs --sample <dir> \
    --max-cells 20000 --both --zoom-um 400 --stats-out comparison.csv
```

On the Xenium lung sample:

```
                 variant  edges  mean_degree  isolated_%  shared_wall_um  centroid_um  touching_%
   contact (exact, 0 µm)   1448         0.14        87.7            0.00        10.22       100.0
contact (tolerance 1 µm)  25956         2.60        10.8            0.00         7.92         5.6
    voronoi (clip 30 µm)  58206         5.82         0.1            7.16        11.73         2.5
```

Note `shared_wall_um = 0.00` for **both** contact variants: on Xenium, touching
pairs meet at a point rather than along a wall, so shared-wall length carries no
information there and those panels are sized by centroid distance instead. If
you need shared-wall as a feature, the Voronoi graph is the only variant that
provides it on this platform.

---

## 2. Plotting — `discell.plotting.cell_graph`

Draws polygons filled by cluster, the neighbour graph with edge width
proportional to a geometric metric, and centroid nodes. Writes one figure over
the tissue image and one on plain background.

```bash
# full slide + a zoom panel, both with and without overlay
uv run python -m discell.plotting.cell_graph --sample <dir>

# a specific pixel window
uv run python -m discell.plotting.cell_graph --sample <dir> --region 16800 18200 7500 8900 --no-zoom
```

### Filtering edges, and showing how much is shared

```bash
# only pairs whose real polygons touch, coloured by how much boundary they share
uv run --extra viz python -m discell.plotting.cell_graph --show --touching --color-edges

# how far apart are neighbours that DON'T touch
... --show --edge-metric wall_dist_um --color-edges --edge-cmap magma

# contact graph, tolerant of 2um segmentation slop
... --show --graph contact --max-gap-um 2

# strong interfaces only, over the H&E
... --show --touching --min-wall-um 10 --color-edges --overlay

# same as a high-resolution file
... --touching --color-edges --region 16800 18200 7500 8900 --no-zoom
```

| flag | effect |
|---|---|
| `--touching` | shorthand for `--max-gap-um 0` |
| `--max-gap-um X` | keep edges whose original polygons are within X µm |
| `--min-wall-um X` | keep edges sharing at least X µm of boundary |
| `--max-centroid-um X` | drop edges longer than X µm |
| `--edge-metric` | `shared_wall_um` (default), `centroid_dist_um`, `wall_dist_um` |
| `--color-edges` | colour by the metric and add a colourbar |
| `--edge-cmap` | colormap, default `viridis` |
| `--graph` | `voronoi` (default) or `contact` |
| `--figsize-in` / `--dpi` | output size; 40in × 200dpi ≈ 8000 px |

Effect of the filters on mouse brain:

```
--touching        51,043 / 109,590 edges kept (46.6%)
--min-wall-um 8   82,085 / 109,590 edges kept (74.9%)
```

The metric and filter are encoded in the filename, so variants never overwrite
each other. Note the title reports the kept-fraction slide-wide, while the cell
and edge counts are for the rendered region.

---

## 3. Bundling a run — `discell.data.export`

A bundle is one reloadable analysis run: the counts, the polygons, both graphs
and every edge metric, plus a provenance record. Building the graphs is the
expensive part, so it is done once here and never again.

```bash
# whole slide, default tolerances -> data/datasets/<id>/bundle/full.*
uv run python -m discell.data.export --sample <dir-or-archive>

# a second bundle of the same slide with a tighter graph, kept side by side
uv run python -m discell.data.export --sample <dir> --variant tol1 \
    --contact-tolerance-um 1.0 --wall-tolerance-um 0.5

# bake an apposed-wall trim into the stored graph
uv run python -m discell.data.export --sample <dir> --variant strong --min-apposed-um 1.0

# a small bundle for iterating
uv run python -m discell.data.export --sample <dir> --variant dev --max-cells 20000
```

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

## 4. Image embeddings — `discell.images.kronos`

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
# v1, all four channels, whole slide -> data/datasets/<id>/embeddings/kronos1.pt
uv run python -m discell.images.kronos --sample $S --out kronos1.pt

# v2
uv run python -m discell.images.kronos --model v2 --sample $S --out kronos2.pt

# a DAPI-only baseline, and a quick 2000-cell check
uv run python -m discell.images.kronos --sample $S --out dapi_only.pt --channels 0
uv run python -m discell.images.kronos --sample $S --out probe.pt --limit 2000

# check crops, markers and shapes without loading the model
uv run python -m discell.images.kronos --sample $S --dry-run
```

The weights are gated: request access, then `--hf-token` or `HF_TOKEN`. They
cache once in `data/models/` and are shared by every dataset.

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

### Does the embedding carry cell identity? — `discell.images.embedding_qc`

```bash
# reuse saved embeddings, every cell, UMAP coloured by cell type
uv run python -m discell.images.embedding_qc --sample $S \
    --embeddings kronos1 --per-label 0 --out kronos1_umap.png

# embed a stratified subsample on the fly instead
uv run python -m discell.images.embedding_qc --sample $S --per-label 150
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

```bash
uv run python -m discell.data.loader --self-test
uv run python -m discell.data.loader --dataset <id> --batch-cells 64 \
    --graph contact --min-apposed-um 0.001
```

A batch picks `batch_cells` **seeds**, pulls their 1-hop neighbours, and by
default serves only the directed **neighbour → seed** edges. Fields by the axis
they are aligned to:

| aligned to | fields | size |
|---|---|---|
| seeds | `image_embedding`, `seed_index`, `seed_composition` | fixed |
| nodes | `x`, `y`, `pos`, `node_ids`, `node_index` | varies |
| edges | `edge_index` (2, E), `edge_attr` (E, 2) | varies |
| constant | `neighbour_composition` (K, K) | fixed |

`x` is raw integer counts; `edge_attr` is `[centroid_dist_um, apposed_wall_um]`;
`neighbour_composition` is the mean neighbour-type distribution per cell type,
computed once at construction. `pos` and `node_ids` are auxiliary — for plotting
and debugging, not model inputs. The spatial information a model should consume
is already reduced into `edge_attr`, and absolute slide coordinates would let a
model memorise position.

In Python:

```python
from discell.data.loader import CellGraphDataset

data = CellGraphDataset.from_dataset(
    "xenium_prime_ovarian_cancer_ffpe",
    embeddings="kronos1_v2scale",     # resolved inside that dataset
    graph="contact", batch_cells=64, resident="gpu",
)
for batch in data.torch_dataloader():
    ...
```

Embeddings are checked against the bundle. A `.pt` recording a different dataset
raises, and so does one sharing no cell ids — previously that only warned about
zero-filled rows and trained on fabricated vectors.

---

## 6. Inspecting a batch — `discell.main`

Loads one batch, prints every field with its shape and meaning, runs seven
structural checks, and writes a four-panel figure.

```bash
uv run python -m discell.main --embeddings kronos1_v2scale --batch-cells 10
uv run python -m discell.main --dataset <id> --graph voronoi --image-scope nodes
uv run python -m discell.main --no-plot            # text only
```

The checks are assertions about batch structure, not statistics: every edge
points into a seed, no self-loops, endpoints are valid node rows, counts are raw
integers, labels are one-hot, composition rows sum to 1, and `edge_attr`
distances match `pos`.

The figure combines label distribution, the batch graph in slide coordinates
(edge width = apposed wall, node colour = cell type), a cosine-distance matrix
over the image embeddings, and per-type neighbourhood composition bars —
alongside a whole-tissue panel placing the seeds on the slide.

### Training demo — `discell.examples.train_demo`

A minimal end-to-end sanity check that the batches train something:

```bash
uv run python -m discell.examples.train_demo --embeddings kronos1_v2scale --steps 300
```

---

## 7. Viewing over SSH

`--show` opens an interactive pan/zoom view. Backend selection is automatic:
QtAgg → TkAgg → WebAgg, with the display probed via `xdpyinfo` first.

```bash
ssh -X user@host
uv run --extra viz python -m discell.plotting.cell_graph --show
```

If X is slow over a WAN — likely at this figure size — skip X entirely:

```bash
ssh -L 8988:localhost:8988 user@host
uv run --extra viz python -m discell.plotting.cell_graph --show --backend WebAgg
# open http://localhost:8988
```

Two environment notes:

- **TkAgg does not work under uv-managed CPython.** `import tkinter` succeeds,
  but `_tkinter` is statically linked and exposes no `__file__`, which
  matplotlib's `_tkagg` needs to locate Tcl/Tk. Qt is used instead; that is why
  `pyqt6` is in the `viz` extra.
- **A dead `$DISPLAY` aborts Qt outright** rather than raising, so it cannot be
  caught. The display is probed with `xdpyinfo` before a backend is chosen, and
  selection falls back to WebAgg.

For the static PNGs, ImageMagick's `display` ships uncompressed pixmaps — about
156 MB per repaint at full slide. Always downsample:

```bash
display -resize 1800x data/datasets/<dataset_id>/figures/<name>.png
```

---

## Notes on the data

- **Coordinates.** Polygons and `obsm['spatial']` are in full-resolution image
  pixels; `microns_per_pixel` comes from `experiment.xenium`.
- **Tissue images vary.** Tiled vs strip-based, uncompressed vs JPEG. Reading
  goes through `tifffile`; strip-based pages must be fully decoded, and are
  cached across figures in one run.
- **Node set mismatch.** A polygon exists per segmented object, but low-count
  cells are dropped from the filtered matrix. The loader keeps the intersection.
