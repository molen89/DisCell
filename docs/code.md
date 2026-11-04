# The code

This page is for people who want to run DisCell on their own data or read its code. It covers
the package layout, the pipeline with the exact function and command names, where each
equation of [model.md](model.md) lives, the paper's terms and their names in the code, the
AnnData fields DisCell reads and writes, determinism, and the tests.

* [Install](#install)
* [Package layout](#package-layout)
* [The pipeline](#the-pipeline)
* [Command line](#command-line)
* [Equations and where they live](#equations-and-where-they-live)
* [Paper terms and code names](#paper-terms-and-code-names)
* [The AnnData contract](#the-anndata-contract)
* [Configuration](#configuration)
* [Determinism](#determinism)
* [Tests](#tests)

## Install

```bash
pip install .                 # core
pip install ".[viz]"          # discell.pl plots (matplotlib)
pip install ".[image]"        # KRONOS2 image descriptor (Python >= 3.11, gated weights)
pip install ".[dev]"          # pytest, ruff
pip install ".[docs]"         # run the example notebooks (matplotlib, Jupyter kernel)
```

A GPU is strongly recommended for real sections. `Config(device="cuda")` is the default, and a
missing CUDA device falls back to the CPU with a warning.

## Package layout

```
src/discell/
  io/           read_xenium: Xenium output folder -> AnnData; the input contract
  graph/        clipped-Voronoi cell graph, spillover kernel beta, neighbor composition, tiles
  image/        optional KRONOS2 descriptor of the tissue around each cell ([image] extra)
  model/        Config, the networks, the objective, the trainer, Model and fit
  evaluate/     the held-out probe, NMI and other metrics, the w-channel guard, metric masks
  sensitivity/  the kappa sweep, tile bootstrap, breakdown points, the paper's findings
  tl/           readouts: relocation, the per-gene split, cross_section
  pl/           plots of the readouts ([viz] extra)
  sim.py        a tissue simulated from the model, with its truths
  cli.py        the `discell` command
reproduce/      configurations, label maps and a driver that reproduce the paper's sections
tests/          the test suite (CPU; a few tests need a GPU)
docs/, examples/   this documentation and the example notebooks
```

The core (`io`, `graph`, `image`, `model`, `evaluate`, `sensitivity`) fits and grades models.
The readouts (`tl`, `pl`) read only what a fit stores in the AnnData, plus the model where they
need its decoder (`relocation`, `cross_section`). `import discell as dc` exposes the main
entry points: `dc.read_xenium`, `dc.build_graph`, `dc.Config`, `dc.fit`, `dc.Model`,
`dc.KAPPA_GRID`, and the subpackages `dc.evaluate`, `dc.sensitivity`, `dc.tl`, `dc.pl`.

## The pipeline

| Step | Python | Command line |
|---|---|---|
| ingest | `dc.read_xenium` | `discell ingest` |
| embed (optional) | `discell.image.embed` | `discell embed` |
| graph | `dc.build_graph` | done by `fit` and `sweep` |
| fit | `dc.fit`, `dc.Model` | `discell fit` |
| sweep | `dc.sensitivity.fit_sweep` | `discell sweep` |
| breakdown | `dc.sensitivity.fit_draws`, `dc.sensitivity.finding_breakdown` | `discell breakdown` |
| bound and verdicts | `dc.sensitivity.kappa_bar`, `dc.sensitivity.decide` | printed by `discell breakdown` |
| per-gene split | `dc.tl.gene_split`, `dc.tl.gene_assignments`, `dc.tl.gene_parts`, `dc.pl.gene_map`, `dc.pl.gene_triangle` | |
| readouts | `dc.tl.*`, `dc.pl.*` | `discell relocation` |

The snippets below run as one script on a small simulated section, on a CPU, in about a
minute and a half. On real data, start from `dc.read_xenium` instead of `simulate` and drop the tiny
settings in `TINY`.

**Ingest.** `dc.read_xenium(outs)` reads a Xenium output folder: raw counts, centroids in µm
(from the segmented boundaries), 10x's clusterings, and a label table if you pass one
(`labels="cells.csv"` with a `cell_id` column; `"auto"` picks up a `*cell_groups.csv` in the
folder). `region_um=(x0, y0, x1, y1)` keeps a rectangle; `donors=[...]` keeps only the cells
of those donors (for example, cores of a tissue microarray). Without Xenium data, any AnnData that meets [the contract](#the-anndata-contract) works. The
simulation produces one:

```python
import tempfile
import discell as dc
from discell.sim import simulate

adata = simulate(n_cells=6000, n_genes=60, kappa=0.2, seed=0).to_anndata()
print(adata)
```

**Embed (optional).** `discell.image.embed(adata)` writes a 768-dimensional KRONOS2
descriptor of the tissue around each cell (cell masked) to `adata.obsm["X_kronos_masked"]`.
It needs the `[image]` extra, your own access to the gated weights, and the morphology image
of the Xenium folder. The simulation already ships an image-like descriptor in
`adata.obsm["X_image"]`.

**Graph.** `dc.build_graph(adata, label_key)` computes the clipped-Voronoi graph of the
centroids (if `.obsp` has none yet), prunes edges longer than 40 µm, and stores the spillover
kernel β and the neighbor composition. `fit` refuses to run without it.

```python
graph = dc.build_graph(adata, "cell_type")
print(adata.uns["discell"]["graph"]["n_edges"], "edges")
```

**Fit.** `dc.fit(adata, config, out_dir=None)` trains one model, writes its outputs into
`adata` and returns a `dc.Model`. With `out_dir` the model is saved there too.

```python
TINY = dict(device="cpu", tile_cells=1024, epochs=100, eval_every=5, w_warmup_epochs=10,
            patience=100, d_z=8, d_w=2, hidden=64, v_pcs=4)   # for a quick CPU demo only
config = dc.Config(label_key="cell_type", image_key="X_image", kappa=0.1, seed=0, **TINY)
work = tempfile.mkdtemp()
model = dc.fit(adata, config, out_dir=f"{work}/k0.1_s0")
print(model.summary()["best_epoch"], round(model.summary()["nmi"], 3))

same = dc.Model.load(f"{work}/k0.1_s0", device="cpu")   # later, or elsewhere
same.encode(adata)                                       # its latents back into adata
probe = dc.evaluate.heldout_probe(same, adata)           # neighborhood information per block
closing = same.evaluate(adata)                           # the fit's closing evaluation
print(round(probe["mlp"]["comp"]["var_fraction"], 3), closing["w_channel"]["dead_context_channel"])
```

**Sweep.** `dc.sensitivity.fit_sweep(adata, config, kappas, seeds, out_dir)` fits every
(κ, seed) pair into `out_dir/k<kappa>_s<seed>/`. Defaults: `dc.KAPPA_GRID` and seeds 0, 1, 2.
A fit already saved is loaded instead of refitted, so a stopped sweep resumes.

```python
fits = dc.sensitivity.fit_sweep(adata, config, kappas=(0.0, 0.2), seeds=(0, 1),
                                out_dir=f"{work}/sweep")
```

**Breakdown.** `dc.sensitivity.fit_draws(model, adata, out_dir=...)` computes the paper's
findings on one fit, with their tile-bootstrap draws, one `.npz` file per group of findings.
The groups are `cycle` (needs the S and G2M cell-cycle genes in the panel), `w_mi`,
`signaling` (the signaling-gene lean; needs `lr_genes=`), `axis` (needs a tumor type),
`relocation_mean` and `relocation_distribution` (relocation reads). A group that is not defined on a section is skipped.
`dc.sensitivity.finding_breakdown(dirs)` pools the fits and applies the breakdown rule
([model.md, section 8](model.md#8-the-sweep-and-breakdown-points)).
`dc.sensitivity.breakdown` does the same for findings you define yourself (see
[use_cases.md](use_cases.md#test-how-robust-a-spatial-finding-is-to-spillover)).

```python
dirs = {}
for (kappa, seed), fitted in fits.items():
    dirs[(kappa, seed)] = f"{work}/sweep/k{kappa:g}_s{seed}/draws"
    dc.sensitivity.fit_draws(fitted, adata, groups=("w_mi", "relocation_mean"), n=100,
                             min_cells=100, out_dir=dirs[(kappa, seed)])
table = dc.sensitivity.finding_breakdown(dirs, grid=(0.0, 0.2), seeds=(0, 1))
print({name: m["status"] for name, m in table["members"].items()})
```

The table holds, per finding, its pooled `trajectory` (estimate and interval at each
κ), `status` (`"breaks"`, `"above the grid"`, `"no finding"`, or `"gap at kappa = ..."` when a
fit is missing), `kappa_star`, `sign`, its `role` in the paper (`"finding"`,
`"check"` or `"counted, not reported"`) and its `paper_name`.

**Bound and verdicts.** `dc.sensitivity.kappa_bar(adata, models)` estimates the section's
upper bound on κ ([model.md, section 7](model.md#7-what-the-counts-can-tell-about-κ-an-upper-bound)):
`["model_based"]` from the fits passed (one value per fit, averaged), `["counts_only"]` from
the neighbors' observed profiles, each with the section value `kappa_bar` (the largest type
value over types with at least 200 connected cells), its `type`, the smallest type value
`min`, and a per-type table. `dc.sensitivity.decide(table, bound)` reads each κ\* against it
and returns `"safe"`, `"not ruled out"` or `"undecided"` per finding, the paper's "holds at
the bound", "breaks before the bound" and "undecided".

```python
bound = dc.sensitivity.kappa_bar(adata, [fits[(0.2, 0)], fits[(0.2, 1)]])
verdicts = dc.sensitivity.decide(table, bound)
print(round(bound["model_based"]["kappa_bar"], 3), {k: v["verdict"] for k, v in verdicts.items()})
```

**Readouts.** Each `dc.tl` function reads the fit's outputs from the AnnData, writes its
results back, and returns its record (also stored as JSON in `adata.uns["discell"][key]`;
`dc.tl.record(adata, key)` reads it back). Encode the fit you want to read first.

```python
model.encode(adata)
rel = dc.tl.relocation(model, adata, min_cells=100)
print(len(rel["pairs"]), "niche pairs")
```

`dc.tl.gene_split(model, adata)` splits each gene's within-type variation into intrinsic,
response and spillover shares on one fit, and `dc.tl.gene_assignments` reads the assignments of a sweep
([use_cases.md](use_cases.md#split-each-genes-variation-within-a-type)); they return their
arrays instead of storing them in the AnnData (`gene_split(..., path=...)` writes an `.npz`).
`dc.tl.cross_section(model, other_adata)` reads a model on another section.

**Plots** (`[viz]` extra). Each takes an optional `ax` and returns it.

```python
import matplotlib
matplotlib.use("Agg")

dc.pl.breakdown_trajectory(table, "niche_information_check")
```

## Command line

The `discell` command wraps the same steps. Every file it writes is named by an argument.

```bash
discell ingest Xenium_outs/ -o section.h5ad          # --labels CSV, --region X0 Y0 X1 Y1, --donors
discell embed section.h5ad                           # optional, [image] extra
discell fit section.h5ad --label-key lineage -o fits/k0.1_s0          # add --image-key X_kronos_masked
discell sweep section.h5ad --label-key lineage -o fits                # kappa grid x seeds 0-2
discell breakdown fits section.h5ad --lr-genes lr_genes.txt -o breakdown.json
discell relocation fits/k0.1_s0 section.h5ad -o relocation.json
```

`fit` and `sweep` build the graph when it is missing. `--config FILE.json` passes any `Config`
field (command-line flags override it). `sweep` and `breakdown` resume from what is on disk.
`breakdown` also estimates κ̄ (by default model-based, from the fits at κ = 0.1;
`--kappa-bar counts`, a number, or `none`) and prints each finding's verdict.
`discell <command> --help` lists every option.

## Equations and where they live

Equation names follow [model.md](model.md). Paths are under `src/discell/`.

| Model piece | Function | File |
|---|---|---|
| $z_i \sim \mathcal N(0, I)$, posterior $q(z \mid x, t)$ | `DisCell.posterior_z` | `model/networks.py` |
| prior mean $m_\psi(c_i, t_i)$, posterior $q(w \mid c, t, z, x)$ | `DisCell.prior_w`, `DisCell.enc_w` (in `DisCell.forward`) | `model/networks.py` |
| $\rho_i = \mathrm{softmax}(a(z_i) + Bw_i)$ | `DisCell.log_rho` (`dec_a`, `B`) | `model/networks.py` |
| influx $\bar\rho_i = \sum_j \beta_{ij}\rho_j$ (stop-gradient) | `foreign_influx` | `model/equations.py` |
| $p_i = (1-\kappa)\rho_i + \kappa\bar\rho_i$ | `spillover_mix` | `model/equations.py` |
| multinomial likelihood per count $\mathrm{Rec}_i$ | `multinomial_loglik` | `model/equations.py` |
| Gaussian KL terms | `gaussian_kl` | `model/equations.py` |
| kernel $\beta_{ij} \propto f_{ij} e^{-d_{ij}/\tau}$ | `smoothing_weights`, `graph_from_edges`, `build_graph` | `graph/kernel.py` |
| clipped-Voronoi neighbors, face lengths | `add_voronoi_graph` | `graph/voronoi.py` |
| neighbor composition $y_i$ | `composition` | `graph/kernel.py` |
| niche descriptor $c_i$ (GAT $\oplus\ \Phi_i \oplus$ isolated flag) | `GATv2`, `DisCell.context` | `model/networks.py` |
| image descriptor $\Phi_i$ (masked crop, KRONOS2) | `embed`, `ego_disc`, `kronos2_encoder` | `image/kronos.py`, `image/crops.py`, `image/kronos2.py` |
| objective $J$ (both paths, KL weights) | `discell_loss`, `Weights` | `model/objective.py` |
| adversary $\mathrm{Adv}_i$ and its heads | `adversary_terms`, `Adversary`, `Trainer._head_steps` | `model/objective.py`, `model/networks.py`, `model/trainer.py` |
| training loop, checkpoint rule, warm-up | `Trainer.fit`, `Trainer.step` | `model/trainer.py` |
| spatial tiles, two-hop halo, held-out split | `spatial_tiles`, `tile_batch`, `split_tiles` | `graph/tiles.py` |
| probe gain $\Gamma$, permutation baseline, neighborhood information | `probe_gain_per_block`, `probe_gain_per_block_mlp`, `probe_blocks`, `heldout_probe` | `evaluate/probe.py` |
| NMI of $z$ with the types | `z_type_nmi` | `evaluate/metrics.py` |
| w-channel guard (niche information in $w$) | `w_channel_guard`, `kl_w_is_dead` | `evaluate/guard.py` |
| cell-cycle $R^2$ of $z$ and $w$ | `cell_cycle_r2` | `evaluate/cycle.py` |
| estimated upper bound $\bar\kappa$ (pooled per type, Garwood limit, both versions) and the decision rule | `kappa_bar`, `type_bounds`, `poisson_upper`, `decide` | `sensitivity/upper_bound.py` |
| the κ sweep | `fit_sweep` | `sensitivity/sweep.py` |
| tile bootstrap, half-tile subsampling | `finding_draws`, `tile_draws`, `half_tile_weights` | `sensitivity/bootstrap.py` |
| pooling over fits, Bonferroni level, breakdown rule, κ\* | `pool`, `level`, `holds`, `breakdown_point`, `breakdown` | `sensitivity/breakdown.py` |
| the paper's findings | `fit_draws`, `finding_breakdown` | `sensitivity/findings.py` |
| relocation, noise ceiling, alignment | `relocation`, `score_shift`, `noise_ceiling`, `alignment` | `tl/relocate.py` |
| per-gene split: Shapley parts, shares $s_k$, gene rule, assignments, consensus and κ\* | `shapley`, `gene_split`, `gene_rule`, `assignments`, `gene_assignments`, `gene_parts` | `tl/genes.py` |
| the simulation | `simulate` | `sim.py` |

## Paper terms and code names

This table gives the code's name for each term of the paper.

| In the paper | In the code |
|---|---|
| intrinsic state $z_i$ | `.obsm["X_discell_z"]` (posterior mean), `mu_z`, `Config.d_z` |
| niche response $w_i$ | `.obsm["X_discell_w"]`, `mu_w`, `Config.d_w` |
| expected response $m_\psi(c_i, t_i)$ | `.obsm["X_discell_mpsi"]`, `prior_w` |
| gene programs, the columns of $B$ | `.varm["discell_B"]` |
| clean profile $\rho_i$ | `DisCell.log_rho` (its log) |
| the neighbors' influx $\bar\rho_i$ | `foreign_influx`, `rho_bar` |
| spillover fraction κ | `Config.kappa` |
| the κ grid {0, 0.05, 0.1, 0.2, 0.3, 0.4} | `dc.KAPPA_GRID` |
| intrinsic path, weight ω | `Config.omega` |
| adversary weight $\alpha_a$, composition weight $\lambda_y$ | `Config.alpha_a`, `Config.adv_comp_weight` |
| neighborhood information | `var_fraction` of the probe's composition block `comp` (`heldout_probe`) |
| cell-cycle $R^2$ | `dc.evaluate.cell_cycle_r2` (`cycle_r2_*_q90` in reproduction reports) |
| breakdown point κ\* | `kappa_star`; "above the grid" when it holds at every tested κ |
| estimated bound κ̄ (model-based, counts-only) | `dc.sensitivity.kappa_bar`: `["model_based"]`, `["counts_only"]` |
| holds at the bound / breaks before the bound / undecided | `dc.sensitivity.decide` (`"safe"`, `"not ruled out"`, `"undecided"`) |
| the per-gene split: shares of the intrinsic state, niche response and spillover; a gene's assignment and its κ\* | `dc.tl.gene_split` (`shares`, `assignment`), `dc.tl.gene_assignments` (`kappa_star`, `holds_beyond_kappa_bar`) |
| spillover-test genes and their source type | `spill_test`, `source` in a split |
| number of findings $M$ (Bonferroni) | `m_s` |
| the relocation finding | `relocation_finding` |
| moved cells against a random cell (counted in $M$, not reported) | `moved_vs_random_cell` |
| the signaling-gene lean (finding) | `signaling_gene_lean` (draw group `signaling`) |
| the tumor axis (finding) | `tumor_axis` |
| cell-cycle $R^2$ of $z$ minus that of $w$ (check) | `cycle_check` |
| niche information in $w$ (check) | `niche_information_check` |
| moved cells against the type's mean profile (counted in $M$, not reported) | `moved_vs_type_mean` |
| response part, spillover part and predicted shift of a relocation | `response`, `spillover` and `predicted` (`dc.tl.relocation`) |
| noise ceiling of a niche pair (S3.2) | `noise_ceiling` |

`discell.sensitivity.findings` holds these as `FINDINGS`, `CHECKS`, `COUNTED_NOT_REPORTED` and
`PAPER_NAMES`. A fit whose w carries no niche information is flagged by `dead_w_channel` and
`dead_context_channel` (`model.summary()`).

## The AnnData contract

**Input.**

| Field | Content |
|---|---|
| `.X` (or `.layers[counts_layer]`) | raw counts, cells x genes |
| `.obs[label_key]` | cell types. Missing values count as `Unassigned`: such cells stay in the graph and in training but are never graded. |
| `.obsm["spatial"]` | centroids in µm (or in pixels, with `.uns["discell"]["microns_per_pixel"]` set) |
| `.obsm[image_key]` | optional per-cell image descriptor (a zero row for a cell without one) |

**Written by `dc.read_xenium`.** `.obsm["spatial_px"]` (centroids in image pixels; the graph
is computed in this frame), `.obs["discell_covering_radius_um"]`, `.var["gene_id"]` and
`.var["feature_type"]`, one `.obs` column per 10x clustering (`graphclust`, ...) and per label
column, and `.uns["discell"]["xenium"]` (provenance).

**Written by `dc.build_graph`.**

| Field | Content |
|---|---|
| `.obsp["voronoi_connectivities"]`, `["voronoi_shared_wall_um"]`, `["voronoi_centroid_dist_um"]` | the unpruned face graph (computed if absent) |
| `.obsp["discell_connectivities"]` | pruned graph, undirected, 0/1 |
| `.obsp["discell_face_um"]`, `["discell_dist_um"]` | face length and centroid distance on kept edges |
| `.obsp["discell_beta"]` | spillover kernel β (rows = receiving cell; rows sum to 1) |
| `.obsm["discell_y"]` | neighbor composition $y$ (columns: sorted type names) |
| `.obs["discell_isolated"]` | cells with no edge after pruning |
| `.uns["discell"]["graph"]` | label key, type names, parameters, summary |

**Written by `discell.image.embed`.** `.obsm["X_kronos_masked"]` (768 columns),
`.obs["X_kronos_masked_embedded"]`, `.uns["discell"]["image"]` (the recipe).

**Written by `dc.fit` and `Model.encode`.**

| Field | Content |
|---|---|
| `.obsm["X_discell_z"]` | posterior mean of the intrinsic state z |
| `.obsm["X_discell_w"]` | posterior mean of the response w |
| `.obsm["X_discell_mpsi"]` | prior mean $m_\psi(c, t)$ of w: the response expected from the niche |
| `.varm["discell_B"]` | response loadings $B$, genes x `d_w` |
| `.uns["discell"]["fit"]` | κ, seed, `alpha_z`, best epoch, held-out reconstruction, NMI, guard flags, and the full configuration and metrics as JSON strings |

**Written by the readouts.**

| Function | Arrays | Record in `.uns["discell"]` |
|---|---|---|
| `tl.relocation` | `.obs["discell_niche"]` | `relocation` |
| `tl.cross_section` | the fit's fields, on the other section | `cross_section` |

Records are stored as JSON strings, so an `.h5ad` round trip keeps them whole.

**Saved model folder** (`Model.save`, `fit(out_dir=...)`, each sweep fit): `model.pt`
(weights, configuration, type and gene names), `config.json`, and for a fit `metrics.json`,
`history.jsonl`, `losses.jsonl`. `fit_draws` adds `draws/<group>.npz`.

## Configuration

`dc.Config` is a frozen dataclass whose defaults are the paper's settings. The fields you are
expected to set:

| Field | Default | Meaning |
|---|---|---|
| `label_key` | required | `.obs` column of cell types |
| `kappa` | 0.1 | assumed spillover fraction for this fit |
| `seed` | 0 | initialization, shuffle, held-out split, image PCA and clusters |
| `image_key` | None | `.obsm` key of the image descriptor; None fits the image-free model |
| `alpha_z` | `"auto"` | weight of the KL of z; `"auto"` = 0.5 / median total count |
| `alpha_a` | 0.3 | adversary weight; 0 gives the reference fit without it |
| `tile_cells` | 4096 | maximum cells per training tile |
| `epochs`, `patience` | 500, 40 | epoch budget; epochs without an accepted checkpoint before stopping |
| `device` | `"cuda"` | torch device |
| `counts_layer` | None | `.layers` key of the raw counts; None reads `.X` |

The other fields (`d_z`, `d_w`, network widths, optimizer and adversary settings, `alpha_w`,
the warm-up, the checkpoint guard) are documented in `model/config.py`. Changing them leaves the
configuration the paper evaluated. `reproduce/configs/*.json` lists every value used for the
paper's sections.

## Determinism

* The graph, the image clusters and every k-means (niches, guards) are deterministic: k-means
  and the PCA before it run single-threaded, pinned in code.
* CPU fits are reproducible run to run at one torch thread (`torch.set_num_threads(1)`).
  Multithreaded CPU training is not bitwise reproducible.
* GPU fits are not bitwise reproducible (scatter operations). Refits agree within the
  variation between seeds, which is why every finding is read over several seeds.
* Bootstrap draws use fixed seeds per finding group and fit seed, so a rerun of
  `fit_draws` on the same fit gives the same draws. The pooling over fits in
  `dc.sensitivity.breakdown` is seeded per finding, so it gives the same intervals too.

## Tests

```bash
pip install -e ".[dev,viz,docs]"
pytest -m "not gpu"                       # CPU suite
pytest -m "not gpu and not slow"          # skip the slower tests
pytest tests/test_docs.py                 # the Python snippets of these docs, and notebook 01
ruff check src tests reproduce docs
```

`tests/test_docs.py` runs every Python block of `docs/use_cases.md` and `docs/code.md` on the
CPU, checks that the example notebooks are stored with their outputs and without errors, and
(with the `[docs]` extra) executes notebook 01 from top to bottom. GPU tests are marked `gpu`.
`reproduce/` reproduces the paper's sections end to end ([reproduce/README.md](../reproduce/README.md)).
