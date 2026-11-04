"""The per-gene split, its consensus assignments and the gene plots (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import discell as dc
from discell.sim import simulate
from discell.tl import genes as GS

SMALL = dict(label_key="cell_type", image_key="X_image", d_z=4, d_w=2, hidden=32,
             gat_dim=8, v_pcs=4, tile_cells=300, epochs=4, eval_every=2,
             w_warmup_epochs=2, patience=100, device="cpu", val_fraction=0.4)


def random_inputs(n=50, g=12, seed=0):
    rng = np.random.default_rng(seed)
    t = lambda *s: torch.as_tensor(rng.normal(size=s), dtype=torch.float64)  # noqa: E731
    rb = torch.as_tensor(rng.dirichlet(np.ones(g), size=n), dtype=torch.float64)
    rb_base = torch.as_tensor(rng.dirichlet(np.ones(g), size=n), dtype=torch.float64)
    return t(n, g), t(n, g), t(n, g), t(n, g), rb, rb_base


@pytest.mark.parametrize("kappa", [0.0, 0.1, 0.4])
def test_shapley_is_exact(kappa):
    a, a0, b, b0, rb, rb0 = random_inputs()
    phi, v, v0 = GS.shapley(a, a0, b, b0, rb, rb0, kappa)
    assert torch.allclose(phi.sum(-1), v - v0, atol=1e-12, rtol=0)
    if kappa == 0.0:
        assert phi[..., 2].abs().max() == 0          # no spillover part without spillover
    # a player at its baseline gets nothing
    phi, _, _ = GS.shapley(a0, a0, b, b0, rb, rb0, kappa)
    assert phi[..., 0].abs().max() < 1e-12
    # the value is the decoder's log p (the floor aside)
    if kappa > 0:
        p = (1 - kappa) * torch.softmax(a + b, -1) + kappa * rb
        assert torch.allclose(GS.value(a, b, rb, kappa), (p / p.sum(-1, keepdim=True)).log())


@pytest.fixture(scope="module")
def fitted():
    adata = simulate(n_cells=2400, n_genes=30, n_types=3, kappa=0.2, seed=5).to_anndata()
    dc.build_graph(adata, "cell_type")
    return dc.fit(adata, dc.Config(kappa=0.2, **SMALL)), adata


def test_gene_split_shares(fitted, tmp_path):
    model, adata = fitted
    split = dc.tl.gene_split(model, adata, n_boot=30, path=tmp_path / "split.npz")
    data = model.assemble(adata)
    held = np.zeros(adata.n_obs, dtype=bool)
    held[np.concatenate(data.val_tiles)] = True
    assert np.array_equal(split["n_cells"], np.bincount(data.t[held], minlength=3))
    assert split["checks"]["shapley_sum"] < 1e-12
    big = np.flatnonzero(split["n_cells"] >= GS.MIN_CELLS)
    assert len(big)
    point = split["shares"][big, 0]
    np.testing.assert_allclose(np.nansum(point, axis=-1), 1.0, atol=1e-5)
    assert (split["shares"][big, 1] <= split["shares"][big, 2] + 1e-6).all()
    # against the per-cell parts on the same held-out cells: s_k = Cov(phi_k, sum phi) / Var
    k = int(big[0])
    name = str(split["type_names"][k])
    genes = list(adata.var_names[:5])
    parts = dc.tl.gene_parts(model, adata, genes, name, data=data)
    phi = parts["phi"][held[parts["nodes"]]].astype(np.float64)
    total = phi.sum(-1)
    for c, _g in enumerate(genes):
        tc = total[:, c] - total[:, c].mean()
        expected = [np.mean((phi[:, c, j] - phi[:, c, j].mean()) * tc) / np.mean(tc ** 2)
                    for j in range(3)]
        np.testing.assert_allclose(split["shares"][k, 0, c], expected, atol=1e-5)
    # the file is read back instead of recomputed
    again = dc.tl.gene_split(model, adata, path=tmp_path / "split.npz")
    np.testing.assert_array_equal(again["shares"], split["shares"])
    assert again["kappa"] == split["kappa"] and again["checks"] == split["checks"]


def test_calls_rule():
    shares = np.full((1, 3, 4, 3), np.nan)
    shares[0, 0] = [[0.8, 0.1, 0.1], [0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.2, 0.35, 0.45]]
    shares[0, 1] = shares[0, 0] - 0.05
    shares[0, 2] = shares[0, 0] + 0.05
    assert GS.assignments(shares).tolist() == [[0, -1, 1, -1]]
    shares[0, 1, 2, 1] = 0.45                       # lower limit not above 0.5
    assert GS.assignments(shares)[0, 2] == -1


def fake_split(assignments_by_seed, n_genes=3, n_cells=400):
    shares = np.zeros((1, 3, n_genes, 3))
    for g, assignment in enumerate(assignments_by_seed):
        point = (np.array([0.34, 0.33, 0.33]) if assignment < 0
                 else np.eye(3)[assignment] * 0.8 + 0.1)
        shares[0, :, g] = [point, point - 0.05, point + 0.05]
    return {"shares": shares, "n_cells": np.array([n_cells]), "detect": np.ones((1, n_genes)),
            "own_rate": np.ones((1, n_genes)), "neigh_rate": np.ones((1, n_genes)),
            "sd": np.ones((1, n_genes)), "floored": np.zeros((1, n_genes)),
            "type_names": np.array(["A"]), "gene_names": np.array(["g0", "g1", "g2"])}


def test_gene_assignments_consensus_and_kappa_star():
    # gene 0: response at every kappa; gene 1: response until 0.2, spillover from 0.3;
    # gene 2: one seed of three only, so no consensus assignment
    per_kappa = {0.1: [[1, 1, 0], [1, 1, -1], [1, 1, -1]],
                 0.2: [[1, 1, 0]] * 3,
                 0.3: [[1, 2, 0]] * 3,
                 0.4: [[1, 2, 0]] * 3}
    splits = {(k, s): fake_split(rows[s]) for k, rows in per_kappa.items() for s in range(3)}
    out = dc.tl.gene_assignments(splits, kappa=0.1, kappa_bar=0.25)
    rows = {r["gene"]: r for r in out["types"]["A"]["response"]}
    assert set(rows) == {"g0", "g1"} and not out["types"]["A"]["intrinsic"]
    assert rows["g0"]["kappa_star"] is None and rows["g1"]["kappa_star"] == 0.3
    assert out["first_tested_at_or_above"] == 0.3
    assert rows["g0"]["holds_beyond_kappa_bar"] and not rows["g1"]["holds_beyond_kappa_bar"]
    assert out["counts"][0.3] == {"intrinsic": 1, "response": 1, "spillover": 1}


def test_gene_plots(fitted):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, adata = fitted
    name = sorted(set(adata.obs["cell_type"]))[0]
    parts = dc.tl.gene_parts(model, adata, ["gene0", "gene1"], name,
                             context_type=sorted(set(adata.obs["cell_type"]))[1])
    assert parts["phi"].shape == (len(parts["nodes"]), 2, 3)
    assert np.nanmax(parts["context"]) <= 1.0 + 1e-6
    axes = dc.pl.gene_map(parts, "gene1", shares=[0.2, 0.5, 0.3])
    assert len(axes) == 5 and all(ax.has_data() for ax in axes)
    plt.close(axes[0].figure)
    split = dc.tl.gene_split(model, adata, n_boot=10)
    assignments = dc.tl.gene_assignments({(0.2, 0): split}, kappa=0.2, kappa_bar=0.1)
    ax = dc.pl.gene_triangle(assignments, name)
    assert ax.has_data()
    plt.close(ax.figure)
