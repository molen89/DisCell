# Reproducing the paper

The paper's numbers come from the reference fits made for the paper. This package was
checked on the reference fits and by refitting every section from scratch with the
released code. GPU refits
are not bitwise reproducible, so refitted values differ from the reference fits' at the
level of seed variation. What moves in the refits (the tumor axis on Lung FFPE, the NMI on
the TMA core), the image encoder and the results that are not reproduced in this release
are listed in the "Reproduction" section of the top-level `README.md`.

`expected/` holds the reference fits' own reads (frozen, `expected/SHA256SUMS`). The
reports call them "reference fits"; in their JSON they sit under the key `paper`. They are
not numbers copied from the paper's text, which rounds them and sometimes computes them on
another cell set.

One driver per section: `run.py` ingests the Xenium outputs, attaches the paper's lineage
labels, embeds the images (KRONOS2, your own access), fits the three final models
(κ = 0.1, seeds 0-2) and the rest of the κ grid (18 fits), computes the findings'
draws and the breakdown points of the three reported findings, the two checks and
the two members counted but not reported (M = 7, or 6 on the TMA sections), and compares
everything with `expected/` under the tolerances below.

```bash
pip install ".[image]" rdata                      # Python >= 3.11 for the image extra
bash reproduce/download.sh external data/         # CellChatDB
python reproduce/scripts/lr_genes.py data/CellChatDB.human.rda -o data/lr_genes.txt
bash reproduce/download.sh lung_ffpe data/
python reproduce/run.py --section lung_ffpe --data-root data/ --work work/lung_ffpe \
    --lr-genes data/lr_genes.txt
```

`--quick` runs one fit (κ = 0.1, seed 0) with 200 draws and no breakdown. Every stage
skips what is on disk, so a stopped run resumes. Two GPUs: run
`--stages ingest,embed` first, then `--stages fits,draws --part 0/2 --device cuda:0` and
`--part 1/2 --device cuda:1` side by side, then `--stages breakdown,compare`.
`<work>/report.md` is the result.

## Image encoder: KRONOS2

The reference fits embedded images with the first KRONOS release, which its authors have
closed to new access requests. The release supports KRONOS2 only
(`"embed": {"encoder": "kronos2"}` in every config), and every report says so in its first
line. The comparison with `expected/` is kept. A KRONOS2 refit of Ovarian FFPE gives the
same breakdown point for every finding. The other sections were not
refitted with KRONOS2, so for them a difference beyond the tolerances below is not by
itself a failure of the release. `--no-image` fits are the image-free alternative.

## Sections

| section | data | labels (`lineage`) | `alpha_z` | tile cells | large cells | M |
|---|---|---|---|---|---|---|
| `ovarian_ffpe` | 10x Xenium Prime FFPE Human Ovarian Cancer (XRrun) | 10x `cell_groups` -> `labels/ovarian_ffpe/lineage_map.csv` (SOX2-OT+ tumor cells -> Unassigned, cyst lining -> mesothelial) | 0.0035 | 4096 | zero | 7 |
| `lung_ffpe` | 10x Xenium Prime FFPE Human Lung Cancer | 10x `graphclust` -> `labels/lung_ffpe/lineage_map.csv` | 0.002 | 4096 | embed | 7 |
| `ovary_ff` | 10x Xenium Prime Fresh Frozen Human Ovary | 10x `graphclust` -> `labels/ovary_ff/lineage_map.csv` | 0.00035 | 4096 | embed | 7 |
| `tma_core` | GEO GSE315411, slide 11 (GSM9427181), core of donor PDL018D | in-house annotation, **not shipped** | 0.0018 | 2048 | embed | 6 |

Every other setting is the release default (`configs/<section>.json` lists them all, as the
final fits ran). κ grid 0, 0.05, 0.1, 0.2, 0.3, 0.4; seeds 0, 1, 2; κ = 0.1 is the final fit.
The Ovarian FF section needs about 24 GB of GPU memory and a large host RAM; its archive
(144 GB) is offered on the 10x dataset page only.

**The TMA core.** Its labels come from an in-house annotation (scVI, pseudobulk against
reference atlases, manual curation) that is not public, so only the three 10x sections can
be reproduced from public data. With the labels, pass `--cell-groups` (a CSV of
`cell_id, group, donor`) and `--label-dir` (its `lineage_map_proposed.csv` and
`*_reassignment_proposed.csv` tables).

**The serial section.** The serial section is never fitted: it is read through the
models fitted to the TMA core. `--stages serial` (TMA core only, after its fits) applies the
core's final and grid fits unchanged to the serial section (GSE315411 slide 10, sample
GSM9427182, `bash reproduce/download.sh tma_serial data/`, donor PDL018D): it ingests
and embeds slide 10, runs the held-out probe and the cross-section reads of the three
final fits and the findings' draws of all 18 fits on its cells, and compares the
probe with `expected/tma_serial` and the breakdown points with the reference fits' on
the serial section, in `<work>/serial/report.md`. Its labels are in-house too: pass
`--serial-cell-groups` and `--serial-label-dir`. `--part i/n` splits the draws; the last
part to finish (or a rerun with `--part 0/1`) writes the report.

## Variants

`--variant NAME` refits one of the two variants listed under `variants` in
`configs/<section>.json` into `<work>/variants/NAME/`, after the section's `ingest,embed`.
Each variant's `table` field says where its result appears in the paper.

* `no_adversary`: the final configuration with `alpha_a` 0, at κ = 0.1 and the listed
  seeds, the "DisCell, no adversary" points of Fig. 2 and Table S4. Stages `fits,reads`.
  `reads` writes `<fit>/reads.json` (probe, NMI of z and of the response, cell-cycle R² of z
  and w, relocation, the closing evaluation), and `--stages reads` without `--variant`
  does the same for the final fits, for comparison.
* `timing`: times the final configuration (seed 0) for Table S3, with 20 training epochs,
  one evaluation and peak GPU memory. Stage `timing`.

## Label maps: provenance

`labels/<section>/lineage_map.csv` maps each 10x source label to a lineage
(`source_label, lineage, kind, n_cells, share, evidence, note`). They were drafted from
per-cluster marker scores of the 10x labels (states and locations merged into their
lineage, developmentally distinct subtypes kept), and they are the files the final fits
used, unchanged. They are derived from the 10x datasets (CC BY 4.0, 10x Genomics), which
the paper cites. The Lung FFPE and Ovarian FF maps
key on 10x's `graphclust` cluster ids, so they hold only for the `analysis` output shipped
with those datasets.

## Tolerances

The tolerances were set for refits with the paper's own encoder; with KRONOS2 they bound
the encoder change as well (see above). GPU fits are not bitwise reproducible, so refits
are checked against the reference fits' three seeds: the seed ranges overlap and the means
differ by at most max(2 x the reference fits' seed range, a floor). The floors are the
drift of a same-configuration refit of the TMA core (recon_val 0.0026) and 1 point of
neighborhood information (probe `var_fraction`, the share 1 - e^(-2 excess), 0.01). For NMI
only the mean is checked, with a floor of 0.0175: repeated refits of one TMA-core seed
spread by up to that much, more than the reference fits' three seeds (0.0041), so the
ranges need not overlap. The best epoch must overlap the reference fits' range; every fit must
pass the w-channel guard (I(niche; w) above its within-type permutation floor, no dead
channel); every breakdown point (status and κ\*) must equal the reference fits'.

## Files

* `configs/` per-section settings; `labels/` lineage maps; `expected/` the reference
  fits' reads (frozen, `expected/SHA256SUMS`).
* `download.sh` public URLs; `scripts/lr_genes.py` the CellChatDB ligand-receptor union
  (CellChat is GPL-3.0; the database is downloaded, not shipped).
* `lineage.py` the label mapping; `run.py` the driver.

## The esophageal cohort (documented, not reproduced by this package)

The paper re-tests a published finding on two sections of an independent esophageal
adenocarcinoma cohort (Yates et al. 2025, *Cell Reports Medicine*; data on Zenodo, record
15341263, CC BY 4.0: `processed/Xenium.h5ad` and, per patient, the Xenium outputs with
`transcripts.parquet`). This package does not include the ingest for these data, so the
cohort is not refitted here. This section records what a reader needs to rebuild it with
the released API.

**Gates, set before any fit.** A section was used when it had at least 50,000 cells, a
median of about 100 counts per cell or more, and published labels that their marker genes
support. Of the six Xenium patients, P11 (141,505 cells, median 205 counts) and P12 (69,853
cells, median 312) passed. P4 is too small (22,398 cells), P8 too shallow (median 49), and
P10 and P13 are borderline in depth (medians 106 and 94) with 34% and 66% of their cells in
labels whose markers are not enriched. The marker check, a heuristic, compares each
(patient, label)'s mean of panel-present marker sets with the patient's mean and flags a
label whose own set is neither enriched 1.5-fold nor among its top two.

**Cells, coordinates, labels.** The published Cellpose cells and their `layers['counts']`
(4,999 genes: the Xenium Prime 5K panel without CYP2A6 and F2). `obsm['spatial']` and
`obs['area']` are in morphology-image pixels: µm = px x 0.2125 (without the conversion the
40 µm prune and τ = 20 µm would act at 4.7 times their scale). Labels are mapped to
lineages (Carcinoma -> `Tumour`; Epithelial and Epithelial (pre-cancerous) -> normal
epithelium; inflammatory CAF, Fibroblast, Adipose CAF -> Fibroblasts; TAM1, TAM2 ->
Macrophages; T, Treg -> T/NK cells; B; Endothelial; DC; Mast; Muscle -> smooth muscle;
Nerve/adrenal kept apart). On P11 the marker-inconsistent low-depth labels TAM1 (12,377
cells), Muscle (3,067) and Nerve/adrenal (934), 12% of its cells, are set to `Unassigned`,
which stays a training class and a neighbor and is never a metric target; P12 has no
flagged label. The graph is the release graph on the centroids (`dc.build_graph`).

**Three count versions** of the same cells, labels and graph, each fitted on its own:

1. *the authors' counts* (the headline): `layers['counts']`, every decoded transcript in
   the authors' cells, not filtered by quality;
2. *a filtered recount* from `transcripts.parquet`: gene codewords only
   (`codeword_category == "predesigned_gene"`), `qv >= 20`, on the same 4,999 genes. The
   Cellpose outlines are not in the record, so each transcript goes to the cell whose
   equal-area disc (centroid and area from the h5ad) contains it, the nearest such
   centroid where discs overlap. It matches the authors' counts at r = 0.992 (P11) and
   0.993 (P12) per cell and removes 10.1% and 14.0% low-quality transcripts;
3. *an unfiltered recount*: the same discs with every quality value (r = 0.993, 0.994).

**Fits.** Image-free (`Config(image_key=None)`), every other setting the release default,
`alpha_z` = 0.5 / median count of the version (P11 0.0024, 0.0028, 0.0026; P12 0.0016,
0.0019, 0.0017 for the three versions), the κ grid x seeds 0-2 (18 fits per version).

**The published finding and its finding rule.** Carcinoma cells
in the infiltrated CellCharter niches CC2 and CC4 (`obs['cluster_cellcharter']` levels 2
and 4) score higher on the EMT program cNMF3 than those in the carcinoma niche CC1; CC3 and
CC5-CC8 are on neither side. The authors' repository signature files do not reproduce their
published scores (cNMF3: r = 0.02), so the programs' gene sets were recovered from the
published scores themselves (r = 1.000), with equal weights:

* cNMF3 (EMT): ADAMTS12 ADAMTS2 AKT3 BNC2 CACNA1C CALCRL CCN4 CDH11 CDH2 CNTLN COL5A1
  COL5A2 DCLK1 EDIL3 ERG FAP FLI1 KLHL4 MAP1B MEF2C MEG3 MPDZ MRC2 MSN PDE3A PRKD1 PTPRM
  RBMS3 RUNX1T1 SLIT2 SULF1 TCF4 THBS2 ZEB1 ZEB2;
* cNMF4 (differentiated), tested CC3 against CC1: ABTB2 CBLB CD2AP CD47 CD55 CLDN18 CYP3A5
  DST DUOX2 EGFR FUT3 ITGA2 JUP KLF5 LAMB3 LAMC2 MET MXD1 MYO5B NLRC5 PLAUR PLEC PLSCR1 SAMD9
  SLC7A11 SMAD3 SPATS2L SYTL2 TCF7L2 THRB TJP1 TNFRSF21 TSPAN3 YWHAZ;
* cNMF1 (intermediate), tested CC1 against all other niches: ARF6 CANX CD59 EEF1G ELOB EPCAM
  H3F3B HMGA1 KRTCAP2 LDHA P4HB PDIA3 PDIA4 PGK1 RAC1 TSPAN8.

With the authors' own score (each gene's log1p(CP10K) scaled over the carcinoma cells of all
six patients, the mean over the program's genes) on their counts, the EMT difference reproduces: +0.150 on P11 and
+0.145 on P12. Each program gives two findings per fit, on the `Tumour` cells:

* *the score form* (the paper's reported reading): the same score computed on
  log1p(1e4 ρ_i), ρ_i the decoded clean profile at the posterior means
  (`DisCell.log_rho`), each gene scaled over the type's held-out cells of the fit, mean in
  the B niches minus mean in the A niches; tile-bootstrap intervals;
* *the alignment form*: the relocation alignment of `relocation_finding`
  projected on the program's genes, 2⟨o − s, u⟩⟨p, u⟩ / ⟨o, u⟩², with o the observed
  log-rate shift of held-out cells from A to B, s the spillover part, p the response part
  and u the indicator of the program's genes among the scored ones; half-tile subsampling.

**There are 13 findings** per section: the seven of the other sections
(`cycle_check`, `niche_information_check`, `moved_vs_type_mean`, `moved_vs_random_cell`,
`relocation_finding`, `signaling_gene_lean`, `tumor_axis`)
and the six program findings (`cnmf3_rho_score`, `cnmf4_rho_score`, `cnmf1_rho_score`,
`emt_cnmf3_alignment`, `cnmf4_cc3_alignment`, `cnmf1_cc1_alignment`), with the Bonferroni
correction over all 13. With the released API: fit the sweep, compute `fit_draws` for the
standard members, write each program finding's draws per fit with
`dc.sensitivity.finding_draws` (or `dc.sensitivity.write_draws`), and pass all thirteen to
`dc.sensitivity.breakdown` with `n_findings=13`; `dc.sensitivity.kappa_bar` gives the
bounds.

**What the paper reports.** The EMT difference in the score form holds beyond κ = 0.4 on
both patients in all three count versions, against model-based bounds of 0.36 to 0.40
(P11) and 0.27 to 0.29 (P12) and counts-only bounds of 0.47 to 0.49 and 0.41 to 0.44.
With the response held at its type mean as the intrinsic part, the response carries 40 to
44% of the shift on P11 and 4 to 5% on P12 at κ = 0.1. On P12 the response's own alignment
with the shift breaks at 0.05 to 0.2, and the signaling-gene lean breaks at 0.05 or 0.1 or
does not hold at κ = 0, while on P11 both hold beyond 0.4. The tumor axis holds beyond 0.4
on both patients.
