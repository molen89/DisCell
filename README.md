# DisCell

Preprocessing for spatial transcriptomics: raw counts, cell polygons, and
neighbour graphs with real geometric edge weights, ready to feed a model.

Supports 10x **Visium HD** (binned and segmented outputs) and **Xenium** (Prime
5K). Both platforms load into the same AnnData structure, so graphs, metrics and
plotting are platform-agnostic.

```bash
uv sync --extra viz        # viz extra adds pyqt6 + tornado for interactive viewing
```

| module | reads |
|---|---|
| `discell.data.visium_hd` | Visium HD binned outputs — bins × genes, HVG panels |
| `discell.data.segmented` | Visium HD segmented outputs — cells, polygons, graphs |
| `discell.data.xenium` | Xenium outs — cells, boundaries, graphs |
| `discell.plotting.cell_graph` | figures for either platform (auto-detected) |

---

## 1. Binned expression — `discell.data.visium_hd`

Reads a sample directory or a `*_binned_outputs.tar.gz`, extracts only the files
it needs, and returns raw counts subset to a set of highly variable genes.

`adata.X` is never modified: HVG selection normalises a throwaway copy, so counts
stay integer UMIs.

```bash
# end-to-end check on the default test sample
uv run python -m discell.data.visium_hd --self-test

# select 1000 HVGs, cache the panel next to the sample, write an .h5ad
uv run python -m discell.data.visium_hd --sample <dir-or-tar> --n-top-genes 1000 --out sample.h5ad

# reuse that panel on another sample -- skips HVG selection entirely
uv run python -m discell.data.visium_hd --sample <other> --hvg-file hvg_top_1000_square_008um.json
```

The panel file is how you pin **one shared gene axis across samples**, which is
what a model trained on more than one slide needs. Verified on two mouse brain
samples whose references differ (19,070 vs 33,696 genes): both resolve to an
identical 1000-gene axis, all 1000 present.

Useful options:

```bash
--bin-size square_002um       # 002/008/016; only 008 and 016 carry cluster labels
--matrix raw                  # every bin, not just tissue-covered ones
--hvg-flavor seurat_v3        # default; ranks directly on raw counts
--hvg-subsample 50000         # cap bins used for ranking (needed at 002um)
--on-missing pad              # zero-fill panel genes absent from this sample
--all-genes                   # keep all genes, flag the panel in var['highly_variable']
--var-names gene_id           # index var by Ensembl id instead of symbol
```

**Bin sizes.** `square_002um` has no `analysis/` directory, so no cluster labels
exist at that resolution. Labels come from 008 or 016 µm only.

---

## 2. Per-cell expression, polygons and graphs — `discell.data.segmented`

Reads `*_segmented_outputs.tar.gz`: per-cell counts, segmentation polygons,
cluster labels. Expression and geometry are matched by construction — the result
contains exactly the cells that have both, in one shared order, and
`uns['polygons']` indexes positionally into `obs`.

```bash
uv run python -m discell.data.segmented --self-test --sample <dir>

# write the cell x gene matrix and the edge table
uv run python -m discell.data.segmented --sample <dir> --out cells.h5ad --edges-out edges.csv

# nucleus outlines instead of expanded cell boundaries
uv run python -m discell.data.segmented --sample <dir> --polygon-kind nucleus

# looser tessellation, tolerant contact
uv run python -m discell.data.segmented --sample <dir> --clip-radius-um 50 --contact-tolerance-um 2
```

### Two graphs, because they answer different questions

Measured on `Visium_HD_Mouse_Brain`, 40,222 cells:

| graph | edges | mean degree | isolated | median shared wall |
|---|---|---|---|---|
| `contact` | 52,318 | 2.60 | 5.8% | 12.00 µm |
| `voronoi` | 109,590 | 5.45 | 0.4% | 14.72 µm |

**`contact`** — cells whose polygons actually touch. Faithful, but Space Ranger
cell boundaries are nucleus masks dilated outward by ~5.8 µm independently, so
they do not tile the plane. Mean degree 2.60 where planar geometry demands ~6,
and 2,351 cells touch nothing at all.

**`voronoi`** — an exact planar partition seeded from the centroids. Every
adjacent pair shares exactly one edge of positive length, so shared-wall length
is always defined. `--clip-radius-um` caps how far a cell claims territory.

Only **46.6%** of Voronoi edges join cells that physically touch. Filter on
`wall_dist_um` to recover contact semantics from the tessellation.

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
from discell.data.segmented import load_segmented_sample, graph_edge_frame

adata = load_segmented_sample("<sample dir>")
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

## 3. Xenium — `discell.data.xenium`

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
so everything downstream matches Visium HD, with `uns['microns_per_pixel']`
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

At 1 µm the contact graph reaches 2.60 — the same as Visium HD at *exact*
contact. `XENIUM_CONTACT_TOLERANCE_UM = 1.0` is therefore the default here,
unlike the Visium HD loader which defaults to 0.

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
uv run python -m discell.plotting.compare_graphs --sample <dir> --out-dir figures/ \
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

## 4. Plotting — `discell.plotting.cell_graph`

Draws polygons filled by cluster, the neighbour graph with edge width
proportional to a geometric metric, and centroid nodes. Writes one figure over
the tissue image and one on plain background.

```bash
# full slide + a zoom panel, both with and without overlay
uv run python -m discell.plotting.cell_graph --sample <dir> --out-dir figures/

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

## 4. Viewing over SSH

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
display -resize 1800x figures/<name>.png
```

---

## Notes on the data

- **Space Ranger "cell" polygons are not measured membranes.** They are nuclei
  dilated by ~5.8 µm (median; area ratio 6.16×, nucleus contained in its cell).
  H&E carries no membrane stain. This is why the polygons do not tessellate.
- **Coordinates.** Polygons and `obsm['spatial']` are in full-resolution image
  pixels; `microns_per_pixel` comes from the *segmented* `scalefactors_json.json`,
  which differs from the binned one and has no `bin_size_um`.
- **Tissue images vary.** Tiled vs strip-based, uncompressed vs JPEG. Reading
  goes through `tifffile`; strip-based pages must be fully decoded, and are
  cached across figures in one run.
- **Node set mismatch.** Space Ranger emits a polygon per segmented object but
  drops low-count cells from the filtered matrix. The loaders keep the
  intersection.
