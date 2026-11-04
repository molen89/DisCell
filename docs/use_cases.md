# What DisCell is for

DisCell fits one model per section and gives you, for every cell, an **intrinsic state z**
(the cell's own properties, which it would keep in any niche) and a **response w** (how it has shifted with its niche), with
**spillover** from neighboring cells modeled at an assumed **spillover fraction κ**. This page
lists what you can do with that, the function to call for each, and the limits that apply.
The model itself is explained in [model.md](model.md). Package layout and data formats are in
[code.md](code.md).

* [Separate a cell's own state from its niche response](#separate-a-cells-own-state-from-its-niche-response)
* [Test how robust a spatial finding is to spillover](#test-how-robust-a-spatial-finding-is-to-spillover)
* [Split each gene's variation within a type](#split-each-genes-variation-within-a-type)
* [Predict how a cell type shifts in another niche (relocation)](#predict-how-a-cell-type-shifts-in-another-niche-relocation)
* [Compare neighborhood information across methods with the probe](#compare-neighborhood-information-across-methods-with-the-probe)
* [Image-free use](#image-free-use)
* [Apply a fitted model to another section](#apply-a-fitted-model-to-another-section)
* [Limits](#limits)

The snippets run as written on a small simulated section on a CPU (about a minute and a half
in all). Run this setup first. On
real data, replace it with `dc.read_xenium` and your labels (see
[the public-section notebook](../examples/02_public_section.ipynb)) and leave the `Config`
sizes at their defaults.

```python
import numpy as np
import discell as dc
from discell.sim import simulate

adata = simulate(n_cells=6000, n_genes=60, kappa=0.2, seed=0).to_anndata()
dc.build_graph(adata, "cell_type")
TINY = dict(device="cpu", tile_cells=1024, epochs=100, eval_every=5, w_warmup_epochs=10,
            patience=100, d_z=8, d_w=2, hidden=64, v_pcs=4)   # for a quick CPU demo only
config = dc.Config(label_key="cell_type", image_key="X_image", kappa=0.1, **TINY)
```

## Separate a cell's own state from its niche response

**Question.** Is a difference I see between cells of one type a difference in what they are,
or in where they sit?

**Call.** `dc.fit(adata, config)`. It writes the intrinsic state to
`adata.obsm["X_discell_z"]`, the response to `adata.obsm["X_discell_w"]` and the response
expected from the niche alone to `adata.obsm["X_discell_mpsi"]`.

```python
model = dc.fit(adata, config)
z = adata.obsm["X_discell_z"]        # cells x 8 here (20 by default)
w = adata.obsm["X_discell_w"]        # cells x 2 here (6 by default)
```

Use z for what the cell is: clustering into states, cell-cycle phase, finer subtypes the labels
merged. Use w, compared within a type, for what the niche does to it. Both are ordinary
embeddings you can pass to scanpy (for example `sc.pp.neighbors(adata, use_rep="X_discell_z")`).

**Check the split.** The fit summary (`model.summary()`) reports the agreement of z with the labels (`nmi`) and the niche
information in w above its permutation baseline (`niche_information_check`).

## Test how robust a spatial finding is to spillover

**Question.** I see that fibroblasts near tumor cells express a gene program. Could that be
spillover from the tumor cells rather than a response?

The counts cannot answer this by estimating κ: they limit how much spillover there can be,
but never how little ([model.md, section 7](model.md#7-what-the-counts-can-tell-about-κ-an-upper-bound)).
DisCell refits at several assumed κ and reports, per finding, its **breakdown point κ\***:
the smallest tested κ at which the finding no longer holds. It also estimates the section's
**upper bound κ̄**, the largest spillover fraction the counts allow, and reads each κ\*
against it.

**Call.**

1. `dc.sensitivity.fit_sweep(adata, config, out_dir=...)` fits every κ of the grid
   (`dc.KAPPA_GRID` = 0, 0.05, 0.1, 0.2, 0.3, 0.4) at seeds 0, 1, 2 and saves each fit. It
   resumes from what is on disk.
2. `dc.sensitivity.fit_draws(model, adata, out_dir=...)` computes the paper's findings on
   one fit, with their spatial-bootstrap draws. Pass `lr_genes=` (a ligand-receptor gene list)
   to include the signaling-gene lean.
3. `dc.sensitivity.finding_breakdown(draws_dirs)` pools the fits and returns κ\* for every
   finding, with its `role` in the paper: one of the three reported findings, one
   of the two checks of the fit, or one of two comparisons of moved cells that are counted in
   the Bonferroni M but not reported (see [code.md](code.md#paper-terms-and-code-names)).
   `dc.pl.breakdown_trajectory(table, name)` plots one.
4. `dc.sensitivity.kappa_bar(adata, models)` estimates the bound, model-based from the fits
   you pass (the paper: the three seeds at κ = 0.1) and from the counts alone, per cell type
   and per section. `dc.sensitivity.decide(table, bound)` gives each finding's verdict
   (below).

```python
import tempfile

out = tempfile.mkdtemp()
fits = dc.sensitivity.fit_sweep(adata, config, kappas=(0.0, 0.2, 0.4), seeds=(0, 1),
                                out_dir=out)
dirs = {}
for (kappa, seed), fitted in fits.items():
    dirs[(kappa, seed)] = f"{out}/k{kappa:g}_s{seed}/draws"
    dc.sensitivity.fit_draws(fitted, adata, groups=("w_mi",), n=100, out_dir=dirs[(kappa, seed)])
table = dc.sensitivity.finding_breakdown(dirs, grid=(0.0, 0.2, 0.4), seeds=(0, 1))
for name, member in table["members"].items():
    print(name, member["status"], member["kappa_star"])

bound = dc.sensitivity.kappa_bar(adata, [fits[(0.2, 0)], fits[(0.2, 1)]])
print(round(bound["model_based"]["kappa_bar"], 3), round(bound["counts_only"]["kappa_bar"], 3))
for name, verdict in dc.sensitivity.decide(table, bound).items():
    print(name, verdict["verdict"])
```

The same from the command line: `discell sweep` then `discell breakdown`
([code.md](code.md#command-line)).

Each fit can also report the cell-cycle R² of its latents,
`dc.evaluate.cell_cycle_r2(model, adata)`: the held-out R² of the S and G2/M scores, centered
per type, from z and from w, on the most strongly cycling tenth of held-out cells. It is
the paper's measure of what the intrinsic state keeps (it is None on this simulation,
whose genes carry no cell-cycle markers).

**Your own finding.** Any signed number computed from one fit can be swept. Write it as a
function of per-cell weights (the bootstrap reweights the cells of resampled 200 µm tiles),
get its draws on every fit with `dc.sensitivity.finding_draws`, and pass them to
`dc.sensitivity.breakdown`. Here the finding is "type-0 cells have a larger response along
the first dimension of w than type-1 cells":

```python
def finding(fitted):
    fitted.encode(adata)                        # this fit's latents into adata
    w0 = adata.obsm["X_discell_w"][:, 0]
    t = adata.obs["cell_type"].to_numpy()
    a, b = t == "type0", t == "type1"

    def statistic(weights):
        return float(np.average(w0[a], weights=weights[a])
                     - np.average(w0[b], weights=weights[b]))

    return statistic


draws = {}
for (kappa, seed), fitted in fits.items():
    draws.setdefault(kappa, {})[seed] = dc.sensitivity.finding_draws(
        adata.obsm["spatial"], finding(fitted), n=200, seed=seed)
result = dc.sensitivity.breakdown({"type0_minus_type1": draws}, n_findings=1)
print(result["members"]["type0_minus_type1"]["status"])
```

This toy finding is only an example of the mechanics. A real finding must respect the
model's symmetries: each type's mean of w is arbitrary, and programs can rotate, so compare
within types and use rotation-free summaries (see
[model.md, section 9](model.md#9-what-else-is-not-identified)). If you test several findings,
pass them together so `n_findings` (the Bonferroni count) covers all of them.

**How to read κ\*.** It is a property of a finding, not an estimate of κ. κ\* = 0.2 means the
finding holds for an assumed κ up to 0.1 and could be spillover alone from 0.2 on. κ\* above
the grid means it needs a niche response even if 40% of every cell's counts were spillover.
The toy in [model.md, section 8](model.md#8-the-sweep-and-breakdown-points) shows three genes
with three different κ\* in one tissue.

Read κ\* against the section's bound κ̄ (the model-based value, as in the paper). The
verdicts of `decide` are the three outcomes of the paper's Fig. 5:

* **holds at the bound** (`"safe"`) if the finding still holds at a tested κ at or above κ̄:
  spillover of the modeled kind, of any size the counts allow, does not explain it;
* **breaks before the bound** (`"not ruled out"`) if it breaks before any tested κ reaches
  κ̄: spillover of that size is possible, so treat the finding as possibly an artifact of
  segmentation;
* **undecided** (`"undecided"`) if κ̄ lies above the tested range and the finding holds
  throughout.

κ̄ covers spillover of the modeled kind only (one fraction per section, the same for every
gene, one ring of neighbors), and it is not a measurement of how many transcripts were
misassigned. `discell breakdown` prints κ̄ and each verdict (`--kappa-bar model`, the
default, averages the fits at `--operating-kappa 0.1`).

## Split each gene's variation within a type

**Question.** In fibroblasts, does a gene vary because of what the cells are, because of
where they sit, or because their neighbors' transcripts spill into them?

**Call.**

1. `dc.tl.gene_split(model, adata)` divides each gene's within-type variation on the held-out
   cells into the shares of the intrinsic state, the niche response and spillover (exact
   Shapley values of the decoder's three inputs; [model.md, section 11](model.md#11-the-per-gene-split)),
   with tile-bootstrap intervals, and assigns the gene to a contribution when its lower limit
   exceeds 0.5 and it is at least twice each of the other two. Genes: those detected in at least 10% of the type's cells,
   and spillover-test genes that the type barely expresses but its neighbors do. Pass
   `path=` to keep the split on disk.
2. `dc.tl.gene_assignments(splits, kappa_bar=...)` takes the splits of a sweep
   (`{(kappa, seed): split}`): the assignment at κ = 0.1 needs two of three seeds, its κ\* is the
   smallest of 0.2, 0.3 and 0.4 at which it changes, and with the section's κ̄ each assignment
   is marked when it holds beyond the bound. `dc.pl.gene_triangle(assignments, cell_type)` places the
   type's genes by their three shares.
3. `dc.tl.gene_parts(model, adata, genes, cell_type, context_type=...)` gives each cell's
   three parts of a few genes across the whole section, and `dc.pl.gene_map(parts, gene)`
   maps them beside the observed share and the share of neighbors of the context type.

```python
split = dc.tl.gene_split(model, adata, min_cells=100)   # default 300 held-out cells per type
assignments = dc.tl.gene_assignments({(0.1, 0): split})  # a sweep: every (kappa, seed)
for cell_type, by_part in assignments["types"].items():
    print(cell_type, {part: len(rows) for part, rows in by_part.items()})
parts = dc.tl.gene_parts(model, adata, ["gene3"], "type0", context_type="type1")
print(parts["phi"].shape)                                # cells x genes x 3 parts
```

The split covers only what the model explains: counting noise is most of a gene's observed
variation within a type. It depends on the assumed κ: at a larger κ, more of a gene's
resemblance to the neighbors is given to spillover, which is why assignments are read along
the sweep and against κ̄.

## Predict how a cell type shifts in another niche (relocation)

**Question.** If these T cells sat in the tumor's niche instead of the stroma's, how would their
expression change, and how much of that change is response and how much spillover?

**Call.** `dc.tl.relocation(model, adata)`. For each niche pair (a type and two niches) it
predicts the per-gene shift, with a response part (`response`) and a spillover part
(`spillover`), scores both and the two decoded together (`predicted`) against the observed shift, and
reports each pair's noise ceiling. Niches are k-means clusters of the neighbor composition
(or pass your own with `niches=`).

```python
rel = dc.tl.relocation(model, adata, min_cells=100)   # default min_cells=500 for real sections
for tier, summary in rel["summary"].items():
    print(tier, summary["n_pairs"], round(summary["predicted"], 3),
          round(summary["noise_ceiling"], 3))
```

Read scores against the pair's ceiling, and prefer trusted pairs (ceiling at least 0.5). The
observed shift contains spillover, so this score does not show that the separation helps. In
the paper's test on cells withheld from training (Fig. 7), a ridge regression of the shifts
on neighbor composition had the lowest error on five of nine tasks, but it moves no cells and
keeps spillover in its prediction. The relocation finding
(`relocation_finding`) is one of the paper's findings, so a sweep gives it a breakdown point.

## Compare neighborhood information across methods with the probe

**Question.** Does my intrinsic or "cell-state" embedding still carry niche information?

**Call.** `dc.evaluate.heldout_probe(model, adata)` grades a DisCell fit on its held-out tiles.
To grade any other embedding with the same probes, call `dc.evaluate.probe_blocks` on your
latent, the types and the niche features; it fits a linear and a nonlinear probe on training
cells, scores held-out cells and reports the excess over a within-type permutation baseline,
the **neighborhood information**.

```python
probe = dc.evaluate.heldout_probe(model, adata)
print({g: round(probe[g]["comp"]["var_fraction"], 3) for g in ("ridge", "mlp")})

# any other embedding, graded the same way (here: PCA of log counts)
data = model.assemble(adata)
x = np.log1p(data.x.toarray())
other = (x - x.mean(0)) @ np.linalg.svd(x - x.mean(0), full_matrices=False)[2][:8].T
train = np.zeros(adata.n_obs, dtype=bool)
train[np.concatenate(data.train_tiles)] = True
rec = dc.evaluate.probe_blocks(other, data.t, data.v_block, data.vbar_t, train, ~train,
                               n_comp=data.n_comp)
print({g: round(rec[g]["comp"]["var_fraction"], 3) for g in ("ridge", "mlp")})
```

`model.assemble(adata)` gives the niche features (`v_block`: the neighbor composition, which
the paper's probe reads, plus image principal components when there is an image, which the
code scores as a second block), their type means and DisCell's spatial tile
split, so every method is graded on the same held-out cells. Compare with a DisCell fit
without the adversary (`Config(alpha_a=0)`) to see what the adversary removes. A probe at its
baseline bounds what that probe can detect; it does not prove independence.

## Image-free use

The image descriptor is optional. Leave it out with `Config(image_key=None)` (CLI: no
`--image-key`). The niche descriptor then carries the neighbor composition only, and the
image half of the adversary is dropped. Everything else works the same.

```python
free = dc.fit(adata, dc.Config(label_key="cell_type", image_key=None, **TINY))
print(dc.evaluate.heldout_probe(free, adata)["ridge"]["img"])       # None: no image block
```

With an image, DisCell uses KRONOS2 (`pip install "discell[image]"`, Python 3.11 or newer,
gated weights with your own Hugging Face access; non-commercial academic license). See the
[README](../README.md#kronos2-access-optional). `discell.image.embed(adata, encoder=fn)`
also accepts any function from image crops to vectors, if you want to use another model.

## Apply a fitted model to another section

`dc.tl.cross_section(model, other)` reads a fitted model on a section it was not fitted on
(for example a serial section). The other section must have the same genes in the same order
and the same type names. It writes the latents into the other AnnData, so every readout above
then works on it.

```python
# a second simulated tissue with the same genes and type names; its biology is drawn
# afresh, so expect lower scores than on a real serial section
other = simulate(n_cells=6000, n_genes=60, kappa=0.2, seed=1).to_anndata()
dc.build_graph(other, "cell_type")
reads = dc.tl.cross_section(model, other)
print(round(reads["nmi"], 3), round(reads["recon_all_tiles"], 3))
```

## Limits

* **One platform.** Every section in the paper is 10x Xenium. The model needs only counts,
  centroids and type labels (the image is optional), so it should apply to other platforms
  with segmented cells, imaging-based ones such as MERSCOPE or CosMx and sequencing-based
  ones such as Visium HD, but this is untested.
* **One form of spillover.** The sweep covers one fraction per section, from direct
  neighbors, the same for every gene. Spillover that is specific to some genes, or that
  spares the nucleus, is not covered, and the part of it the model cannot place lands in the
  response. A finding robust to the sweep can still come from intrinsic variation organized
  in space, such as a subclone filling one region, or from cells choosing their niche.
* **Type-level response at default weights.** At the default response weight
  ($\alpha_w = 0.1$), the response mostly follows its niche-expected prior mean: it describes
  how a type shifts with its niche, not how one cell deviates from it. A per-cell response
  built into a simulation was not detected.
* **Relocation.** On cells withheld from training (the paper's Fig. 7), a ridge regression of
  the shifts on neighbor composition had the lowest error on five of nine tasks, but it moves
  no cells and keeps spillover in its prediction. In simulation, where the spillover-free
  change is known, DisCell's response part recovered much more of it than the regression.
* **Collapsed fits.** A fit whose w carries no niche information is flagged
  (`model.summary()["dead_w_channel"]` or `["dead_context_channel"]`). Refit that grid point
  with another seed.
* **Labels matter.** z is anchored to the type labels you give. States or locations merged
  into one label can end up in either latent.
* **Compute.** Real sections need a GPU. A fit of a 280,000-cell section takes a few minutes
  on one RTX 4090. A full sweep is 18 fits.
* **The image model.** The paper's fits used the first KRONOS release, which is no longer
  available. This release uses KRONOS2. A KRONOS2 refit of Ovarian FFPE gives the same
  breakdown point for every finding. The other sections were not refitted with KRONOS2.
