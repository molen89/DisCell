# Examples

Two Jupyter notebooks, stored with their outputs so they read on GitHub without running them.

| Notebook | What it shows | Hardware | Runtime |
|---|---|---|---|
| [01_quickstart_simulated.ipynb](01_quickstart_simulated.ipynb) | The whole workflow on a simulated section with known spillover: fit, intrinsic state z versus response w, the probe's neighborhood information with and without the adversary, a small κ sweep with breakdown points, the section's upper bound κ̄ with the decision rule, and how to sweep a finding of your own. | CPU | about 2.5 minutes |
| [02_public_section.ipynb](02_public_section.ipynb) | A public 10x Xenium section (lung cancer FFPE, about 278,000 cells, 5,000 genes), image-free, one κ and one seed: ingest, lineage labels, graph, fit, probe, relocation. | one GPU (run on an RTX 4090); about 7 GB host memory | about 4 minutes, plus the 22.6 GB download |

To run them:

```bash
pip install ".[viz,docs]"            # from the repository root
jupyter lab examples/                # or any Jupyter front end (install one separately)
```

Notebook 02 needs the data first:

```bash
bash reproduce/download.sh lung_ffpe data/        # 10x Genomics, CC BY 4.0
```

It looks for the data in `../data/` relative to `examples/`. Set `DISCELL_DATA` to point
elsewhere. To try it on a smaller GPU, read only a region with
`dc.read_xenium(OUTS, region_um=(x0, y0, x1, y1))`.

The figures of notebook 02 are derived from the *Xenium Prime FFPE Human Lung Cancer* dataset
by 10x Genomics (CC BY 4.0).

`tests/test_docs.py` checks that both notebooks are stored executed and without errors, and
(with the `[docs]` extra) executes notebook 01 on the CPU.
