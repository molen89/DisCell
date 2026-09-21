# GSE315411 — shared-label-space cell typing for the two Prime 5K slides

Produces **cell type names that are identical across both slides** of TMA PDLTMA006
(sections 10 and 11), so DisCell can train on one slide and evaluate on the other with
one type vocabulary. Received from the user 2026-09-15 and adapted; the record of what
was run and why is in `docs/devlog.md` ("Fourth intake: GSE315411").

```
00_build_query.py        two Xenium bundles  -> query_raw.h5ad          (CPU)
01_embed_cluster.py      joint scVI + Leiden -> query_emb.h5ad          (GPU)
02_transfer.py           scANVI + scArches   -> transfer_<tag>.csv      (GPU)  FAILED at this depth, kept as record
03_name_and_report.py    cluster naming from scANVI -> labels_<tag>.csv (CPU)  (record only)
02b_pseudobulk.py        cluster pseudobulk vs reference type means -> pseudobulk_<tag>.csv   (CPU)
03b_name_pseudobulk.py   two-reference lineage gate -> labels_pb.csv                        (CPU)  (candidate, superseded)
diag_cluster_markers.py  fold-change markers per cluster -> cluster_markers.csv             (CPU)
diag_canonical_markers.py canonical-marker table -> canonical_markers.csv                   (CPU)
diag_subcluster.py       split mixed clusters on the scVI latent -> subclusters.csv          (CPU)
03c_curate.py            curated_names.csv (unit -> name, evidence) -> labels_curated.csv   (CPU)  THE LABELS USED
04_write_cell_groups.py  <outs>/GSE315411_<slide>_cell_groups.csv per slide (--accept)      (CPU)  -> discell.preprocess
```

What was used (2026-09-15): stages 0, 1, then `02b` for both references, the
three `diag_*` scripts, `03c` with `curated_names.csv`, and `04 --tag curated --accept`.
The scANVI path (02/03) collapsed onto one class per reference (60 % "Alveolar
fibroblasts" / 97 % "CAP1") at ~100 transcripts per cell; its outputs stay on disk as
the record. See `docs/devlog.md` ("Fourth intake: GSE315411" and the two entries after it).

Working directory: `<GSE315411>/annotate/`; references in `annotate/references/`
(`download.sh` there fetches both from CELLxGENE as h5ad).

## Environment

`.venv/` here is a `uv venv --system-site-packages` overlay on the `gaston-mix` conda env
(scvi-tools 1.4.0, torch 2.5.1/cu124, scanpy 1.10.4) adding igraph, leidenalg, pyarrow.
Recreate with:

```bash
uv venv --python ~/mambaforge-pypy3/envs/gaston-mix/bin/python --system-site-packages .venv
uv pip install --python .venv/bin/python igraph leidenalg pyarrow
```

## Run

The label path as used, after `run_all.sh` has produced `query_emb.h5ad` (it also runs
the scANVI stages, which stop at the stage-4 guard):

```bash
P=.venv/bin/python; REF=<GSE315411>/annotate/references
$P 02b_pseudobulk.py --ref $REF/hlca_core.h5ad --ref-label-key ann_finest_level \
     --ref-lineage-keys ann_level_1 ann_level_2 ann_level_3 --tag hlca
$P 02b_pseudobulk.py --ref $REF/lungmap_cellref.h5ad --ref-label-key celltype_level3 \
     --ref-lineage-keys lineage_level1 lineage_level2 celltype_level1 --tag cellref
$P diag_cluster_markers.py; $P diag_canonical_markers.py
$P diag_subcluster.py --clusters 6 23 3 27 36 --resolution 0.3   # -> subclusters.csv
$P 03c_curate.py                       # applies curated_names.csv
$P 04_write_cell_groups.py --tag curated --accept
```


```bash
setsid nohup bash run_all.sh 1 </dev/null >run.log 2>&1 & disown   # GPU 1
```

Stages skip when their output exists. The CellRef cell-type column is read from
`annotate/references/cellref_label.txt` (written after inspecting the h5ad; default `celltype_level3`).

## Adaptations from the received scripts

- `00_build_query.py::assign_donors` — cell_stats files are Xenium Explorer selection
  exports with two `#` comment lines and a `Cell ID` column. Two of them (PDL026A and
  PDL085A, solo) have the first line *quoted* because the selection name ends in a space,
  which defeats pandas' `comment='#'`; `read_selection_export` strips the header lines by
  hand, and a file without a cell-id column now aborts instead of being skipped (the first
  attempt silently left 143k solo cells as `NA`). Solo bundle ↔ `*_prime_solo_*`, dual
  bundle ↔ `*_prime_V1_*` (verified 100 % id overlap; other variants 0 %).
- `02_transfer.py` — `--ref-gene-col feature_name --ref-use-raw` for CELLxGENE h5ads
  (Ensembl var index, raw counts in `.raw.X`).
- Budgets: stage 1 `--max-epochs 30` (scvi's heuristic gives 4 at 2.2 M cells); stage 2
  `--query-epochs 30` instead of 100 — both with early stopping.
- `04_write_cell_groups.py` is new: hands `cell_id, group, donor` to DisCell's
  curated-label ingest and fails if the two slides' vocabularies differ.

## Reading the report (`annotate/report_hlca.txt`)

- **Slide-imbalanced clusters** (one slide < 5 %): chemistry artefact of the dual run,
  not biology — reported, not used.
- **Composition JSD** between slides: serial sections of the same 17 cores, so ≈ 0;
  above ~0.05 the batch correction did not do its job.
- **Per-donor JSD**: one bad donor usually means a torn or folded core on one slide.
- **`Unknown_<cluster>`**: kept deliberately — HLCA/CellRef are adult, healthy lung; this is
  pediatric lung disease. An honest null beats a confident wrong label.
