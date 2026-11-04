# DisCell

In imaging spatial transcriptomics, the counts assigned to a cell also contain transcripts
from its neighbors, because segmenting densely packed tissue is imperfect. This
**spillover** and a genuine response to the neighbors both make a cell look more like the
cells around it, and the counts alone cannot say how much of the resemblance comes from
each. A finding about how cells respond to their niche may therefore be an artifact of
segmentation.

DisCell describes each cell by its **intrinsic state** `z` (the cell's own properties,
which it would keep in any niche) and its **niche response** `w` (how it has shifted with
its surroundings), and adds spillover from its neighbors as a separate term, at a
**spillover fraction κ**. An adversary keeps the niche out of `z`, and a probe on held-out
cells measures what is left (the **neighborhood information**). The counts limit how much spillover
there can be, but never how little, so DisCell does not commit to one κ. It refits the
model over a grid of κ and reports two things. For each finding, its **breakdown point
κ\***: the smallest tested κ at which the finding no longer holds. For each section, the
**estimated upper bound κ̄**: the largest κ the counts allow.

DisCell is meant as a test that a niche finding must pass before it is interpreted. Fit the
sweep, read each finding's κ\* against the section's κ̄, and treat a finding that fails
within the range of spillover the counts allow as possibly an artifact of segmentation. The same fits give
additional readouts: the intrinsic state for downstream analysis, a per-gene split of each
gene's variation within a cell type into the parts due to the intrinsic state, the niche
response and spillover, and relocation (how a cell type would change in another niche, with
its spillover part separated).

DisCell needs counts, one position and one type label per cell, and optionally the
morphology image.

## Documentation

* [docs/model.md](https://github.com/molen89/DisCell/blob/main/docs/model.md): the model in plain terms, what the counts can tell about κ,
  and what a breakdown point means.
* [docs/use_cases.md](https://github.com/molen89/DisCell/blob/main/docs/use_cases.md): what DisCell can be used for, with the function to
  call for each, and its limits.
* [docs/code.md](https://github.com/molen89/DisCell/blob/main/docs/code.md): package layout, pipeline, where each equation lives, the
  paper's terms and their code names, the AnnData fields, determinism and tests.
* [examples/](https://github.com/molen89/DisCell/tree/main/examples/): a CPU quickstart on simulated data and a run on a public 10x Xenium
  section, as executed notebooks.

## Install

```bash
pip install discell                                        # from PyPI
pip install git+https://github.com/molen89/DisCell.git     # the latest code from GitHub
pip install "discell[image]"  # KRONOS2 image descriptors (Python >= 3.11)
pip install "discell[viz]"    # discell.pl plots (matplotlib)
```

From a clone of this repository, for the tests and the example notebooks:

```bash
pip install ".[dev]"          # pytest, ruff
pip install ".[docs]"         # run the example notebooks (matplotlib, Jupyter kernel)
```

A GPU is strongly recommended for real sections (`Config(device="cuda")`); a missing CUDA
device falls back to the CPU with a warning.

## The test: breakdown points along a κ sweep

```python
import discell as dc

adata = dc.read_xenium("Xenium_outs/", labels="auto")    # counts, centroids (µm), 10x cell groups
dc.build_graph(adata, "cell_group")

import discell.image                                      # optional, [image] extra
discell.image.embed(adata)                                # KRONOS2 -> .obsm['X_kronos_masked']

config = dc.Config(label_key="cell_group", image_key="X_kronos_masked", kappa=0.1, seed=0)
fits = dc.sensitivity.fit_sweep(adata, config, out_dir="fits")   # κ grid x seeds 0-2, resumable
lr_genes = open("lr_genes.txt").read().split()                   # optional, see below
draws = {}
for (kappa, seed), fitted in fits.items():
    draws[(kappa, seed)] = f"fits/k{kappa:g}_s{seed}/draws"
    dc.sensitivity.fit_draws(fitted, adata, lr_genes=lr_genes, out_dir=draws[(kappa, seed)])
table = dc.sensitivity.finding_breakdown(draws)                   # κ* of every finding
final = [fits[(0.1, seed)] for seed in (0, 1, 2)]
bound = dc.sensitivity.kappa_bar(adata, final)                   # κ̄, model-based and counts-only
verdicts = dc.sensitivity.decide(table, bound)                   # safe / not ruled out / undecided
```

`dc.sensitivity.KAPPA_GRID` is the paper's grid (0, 0.05, 0.1, 0.2, 0.3, 0.4), fitted
with three seeds per value. At each κ, every draw of a finding's interval picks three of
the fits with replacement, takes an independent resample of 200 µm tiles of tissue for each
pick and averages them. The interval runs from the α/(2m) to the 1 − α/(2m) quantile of
these averages, with α = 0.05 and m the number of findings tested on the section. A finding
holds at κ if its interval lies entirely on the side of zero of its κ = 0 estimate. κ\* is
the first grid value where it does not hold. A finding that fails at κ = 0 is "no finding",
and one that never fails is "above the grid". The draws are seeded per finding, so a rerun
gives the same result.

**The findings.** `discell.sensitivity.findings` computes the paper's findings on each fit.
Three are the findings the paper reports: the relocation finding
(`relocation_finding`), the signaling-gene lean (`signaling_gene_lean`,
needs a ligand-receptor gene list) and, where a tumor type exists, the tumor axis
(`tumor_axis`). Two are checks of the fit: the cell-cycle $R^2$ of `z` minus
that of `w` (`cycle_check`) and the niche information in `w` (`niche_information_check`).
Two more, comparisons of moved cells with the type's mean profile (`moved_vs_type_mean`) and
with a random cell (`moved_vs_random_cell`), were tested and count in the Bonferroni
correction, but the paper does not report them. Each member of the table carries its
`role` ("finding", "check" or "counted, not reported"). There are M = 7 on a
section with tumor cells and 6 without. Your own findings can be swept the same way
([docs/use_cases.md](https://github.com/molen89/DisCell/blob/main/docs/use_cases.md#test-how-robust-a-spatial-finding-is-to-spillover)).

**The bound κ̄.** Removing more of the neighbors' profile from a cell eventually drives
some gene's share below zero, which no clean profile can have. Pooled over the
connected cells of each type, with an exact Poisson upper limit and a Bonferroni correction
over genes, this gives a bound per type; the section's κ̄ is the largest over its types
(with at least 200 connected cells), the loosest valid choice. The model-based version takes
the neighbors' clean profiles from the fits (the paper: the three seeds at κ = 0.1),
the counts-only version their observed profiles. On the paper's sections the model-based κ̄
is 0.18, 0.38, 0.49, 0.32 and 0.29.

**The decision rule.** A finding that still holds at a tested κ at or above κ̄ *holds at
the bound* (`decide` returns "safe"): spillover of the modeled kind, of any size the counts
allow, does not explain it. One that breaks before any tested κ reaches κ̄ *breaks before
the bound* ("not ruled out") and should be treated as possibly an artifact. If κ̄ lies
above the tested range, a finding that holds throughout is *undecided*. κ\* neither
estimates κ nor certifies a finding, and κ̄ is not a measurement of the physical share of
misassigned transcripts. Both cover one form of spillover only: one fraction per section,
the same for every gene, reaching one ring of neighbors.

**Collapsed fits.** A fit whose `w` carries no niche information is flagged
(`model.summary()["dead_w_channel"]` or `["dead_context_channel"]` is True). `fit_sweep`
does not refit it for you: refit that grid point with another seed.

## Additional readouts

```python
model = fits[(0.1, 0)]
model.encode(adata)                                       # this fit's latents into adata
probe = dc.evaluate.heldout_probe(model, adata)           # neighborhood information
cycle = dc.evaluate.cell_cycle_r2(model, adata)           # cell-cycle R² of z and of w
split = dc.tl.gene_split(model, adata)                    # per gene: intrinsic / response / spillover
relocation = dc.tl.relocation(model, adata)               # a type moved between niches
```

`dc.Config` carries the paper's final configuration as defaults. `alpha_z="auto"` is
0.5 / ℓ̄, with ℓ̄ the section's median total count. The paper used the median on every
section except Ovarian FFPE, where ℓ̄ is 143 against a median of 178, so pass
`alpha_z=0.0035` to reproduce that section. The values used are in
`reproduce/configs/*.json`.

## Command line

```bash
discell ingest Xenium_outs/ -o section.h5ad                     # --labels CSV, --donors, --region
discell embed section.h5ad                                      # optional; KRONOS2; --large-cells zero|embed
discell fit section.h5ad --label-key cell_group --image-key X_kronos_masked -o fits/k0.1_s0
discell sweep section.h5ad --label-key cell_group --image-key X_kronos_masked -o fits
discell breakdown fits section.h5ad --lr-genes lr_genes.txt -o breakdown.json
discell relocation fits/k0.1_s0 section.h5ad -o relocation.json
```

`fit` and `sweep` build the graph when it is missing; `--config FILE.json` passes any
`Config` field. `sweep` and `breakdown` resume from what is on disk. `breakdown` prints
each finding's status and κ\* with its role in the paper, the section's κ̄
(model-based from the fits at κ = 0.1 by default; `--kappa-bar counts|none|VALUE`) and each
finding's verdict.

## KRONOS2 access (optional)

The image descriptor is computed by KRONOS2 (MahmoodLab; a gated model, license
CC BY-NC-ND 4.0, non-commercial academic use). Its weights are **never** shipped with
DisCell: request access at <https://huggingface.co/MahmoodLab/KRONOS2>, then log in with your
own account (`hf auth login`, or set `HF_TOKEN`). `discell.image.embed` downloads the model
(and, on first use, the public BioLinkBERT-large text encoder it uses to describe the 18S
channel, which is not in its marker vocabulary) into your Hugging Face cache and runs the
model's own code from the hub. Without the extra or the access it stops with a one-paragraph
message; DisCell then fits image-free (below).

**The first KRONOS release is not supported.** The paper's fits used it, but its authors
have closed it to new access requests, so this release supports KRONOS2 only (same crops
and mask, 768-d). A KRONOS2 refit of Ovarian FFPE gives the same breakdown point for every
finding ([Reproduction](#reproduction)).

## Image-free use

Leave out the descriptor: `Config(image_key=None)` (CLI: no `--image-key`). The niche
descriptor then carries the neighbor composition only, and the image half of the adversary
is dropped.

## Large cells

The image descriptor zeroes a 25 µm disk about each cell's center (the cell's own pixels),
over a 128 µm field. A cell larger than the disk stays partly visible: the default
`large_cells="embed"` embeds it anyway, as on three of the paper's four fitted sections;
`large_cells="zero"` gives such cells a zero descriptor, as on Ovarian FFPE.

## The AnnData contract

Input:

| where | what |
|---|---|
| `.X` (or `.layers[counts_layer]`) | raw counts, cells x genes |
| `.obs[label_key]` | cell types; missing values count as `Unassigned` (kept in training, never a metric target) |
| `.obsm['spatial']` | centroids in µm |
| `.obsm[image_key]` | optional per-cell image descriptor |

`dc.build_graph(adata, label_key)` adds the clipped-Voronoi cell graph and the spillover
kernel (`.obsp`, `.uns['discell']['graph']`). A fit writes `.obsm['X_discell_z']`,
`.obsm['X_discell_w']`, `.obsm['X_discell_mpsi']` (prior mean of `w`), the response loadings
`.varm['discell_B']` and its configuration and metrics in `.uns['discell']['fit']`. Readouts
store their records in `.uns['discell'][...]` (`dc.tl.record(adata, key)` reads one back).

## External tables (not bundled)

* Ligand-receptor genes for the signaling-gene lean: the paper used the ligand and receptor
  union of CellChatDB (`reproduce/scripts/lr_genes.py`, needs `pip install rdata`).

## Determinism

k-means and the PCA before it run single-threaded (pinned in code), so the graph, the
image clusters and the niches are the same on every machine. CPU fits are reproducible
run to run at one torch thread; GPU fits are not bitwise reproducible (scatter
non-determinism): refits agree within seed-to-seed variation, which is the tolerance
`reproduce/` checks against.

## Reproduction

The paper's numbers come from the reference fits made for the paper. This package was
checked against them in two ways. Its pieces were run on the reference fits and their
inputs, and every fitted section was refitted from scratch with the released code
(`reproduce/run.py`) and compared with the reference fits under the tolerances in
[`reproduce/README.md`](https://github.com/molen89/DisCell/blob/main/reproduce/README.md).
GPU fits are not bitwise reproducible, so a refit differs from the reference fits by about
as much as one seed differs from another.

### Where each result of the paper is in this release

| paper result | in this release |
|---|---|
| Ovarian FFPE, Lung FFPE, Ovarian FF, the TMA core, and the serial section, which is never fitted and is read through the core's fits | `reproduce/run.py --section ...`, `--stages serial`. The three 10x sections run from public data; the TMA core and the serial section need in-house labels that are not shipped |
| The κ sweep, 6 values x 3 seeds = 18 fits per fitted section | `dc.sensitivity.fit_sweep` |
| The breakdown points of Fig. 5 (findings in Table S1; M = 7, or 6 on the TMA sections) | `dc.sensitivity.fit_draws`, `dc.sensitivity.finding_breakdown` |
| The estimated bound κ̄, model-based and counts-only, and the colors of Fig. 5 | `dc.sensitivity.kappa_bar`, `dc.sensitivity.decide` |
| The separation of Fig. 2 and Table S4: neighborhood information, cell-cycle $R^2$ and NMI of `z` | `dc.evaluate.heldout_probe`, `dc.evaluate.cell_cycle_r2`, `Model.evaluate` |
| DisCell without the adversary (Fig. 2, Table S4) and the training cost (Table S3) | `reproduce/run.py --variant no_adversary`, `--variant timing` |
| The per-gene split of Figs. 3, 4 and S1 | `dc.tl.gene_split`, `dc.tl.gene_assignments`, `dc.tl.gene_parts`, `dc.pl.gene_map`, `dc.pl.gene_triangle` |
| Relocation (Fig. 6) | `dc.tl.relocation` |

The reference values in `reproduce/expected/` are the reference fits' own reads (for
example the NMI of each reference fit's `z`), not numbers copied from the paper's text,
so they can differ from a value the paper rounds or computes on another cell set.

**The breakdown points.** Run on the reference fits, the port gives every breakdown point
exactly.
The refits of every section used the same image encoder as the reference fits, the first
KRONOS release, embedded with the release code before it was removed. They reproduce every
breakdown point except the tumor axis on Lung FFPE, which breaks at 0.4 in the paper and
holds beyond 0.4 in the refit.

**The NMI on the TMA core.** The NMI of `z` with the type spans 0.615 to 0.619 in the
refits against 0.621 to 0.625 in the reference fits. Repeated refits of a single seed
spread by up to 0.0175, more than the reference fits' three seeds do (0.0041), so the NMI
is checked on its mean only, with that floor. The means differ by 0.0062.

**The image encoder.** The released code embeds images with KRONOS2. A KRONOS2 refit of
Ovarian FFPE gives the same breakdown point for every finding.
The other sections were not refitted with KRONOS2.

### Not reproduced in this release

* **The esophageal cohort.** Its data need an ingest that this package does not include.
  Its gates, count versions, finding rule and 13 findings are documented in
  [`reproduce/README.md`](https://github.com/molen89/DisCell/blob/main/reproduce/README.md#the-esophageal-cohort-documented-not-reproduced-by-this-package),
  enough to rebuild it with the released API.
* **The relocation of withheld cells of Fig. 7 (Tables S5–S7)**, run on Cellina's
  leave-one-out protocol: in each of nine tasks one cell population is removed before
  fitting, every method is refitted on the reduced data, and each predicts the withheld
  cells. It needs refits on reduced data, Cellina's software, and a step that computes the
  target niche's descriptor and influx at the withheld cells' positions, none of which this
  package includes. The protocol is in the supplement (S3.3).
* **The comparison methods** (resolVI, SIMVI, MintFlow, Cellina). They run in their own
  software environments, outside this package. The correlation and error of log fold
  changes on the 50 most changed genes (Cellina's benchmark metrics) and the
  ridge-regression reference are not computed by this package either.
* **The simulation studies** of the supplement (S5.1) beyond `discell.sim.simulate`, such
  as the control with two known findings.

## Citation

To be added on publication.

## License

The DisCell code is released under the BSD 3-Clause license (`LICENSE`). Third-party
data, models and gene lists, none of which is shipped here, are listed with their terms in
[`NOTICE`](https://github.com/molen89/DisCell/blob/main/NOTICE). In short: the 10x Xenium datasets are CC BY 4.0 (the
lineage maps in `reproduce/labels/` are derived from them), GSE315411 is under the terms
of its GEO submitters, KRONOS2 is CC BY-NC-ND 4.0 and its non-commercial terms bind anyone
who uses the image extra, CellChatDB is downloaded as data
only, and the Tirosh et al. cell-cycle genes are listed in `discell.sensitivity.findings`.
