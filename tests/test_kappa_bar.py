"""The estimated upper bound kappa_bar and the decision rule (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import stats

import discell as dc
from discell.sensitivity import upper_bound as KB
from discell.sim import simulate

SMALL = dict(label_key="cell_type", image_key="X_image", d_z=4, d_w=2, hidden=32,
             gat_dim=8, v_pcs=4, tile_cells=300, epochs=4, eval_every=2,
             w_warmup_epochs=2, patience=100, device="cpu")


def test_poisson_upper_is_exact():
    for n in (0, 1, 5, 40, 300):
        for a in (0.05, 0.001):
            assert stats.poisson.cdf(n, KB.poisson_upper(n, a)) == pytest.approx(a, rel=1e-6)


def pooled(sim, rho_bar):
    """N and S = sum_i l_i rho_bar_i per type, over the connected cells."""
    conn = sim.graph.degrees > 0
    rows = np.flatnonzero(conn)
    onehot = sp.csr_matrix((np.ones(len(rows)), (sim.t[rows], rows)),
                           shape=(sim.n_types, len(sim.t)))
    return onehot @ sim.x.astype(float), onehot @ (sim.totals[:, None] * rho_bar)


@pytest.mark.parametrize("kappa", [0.05, 0.2, 0.4])
def test_bound_holds_for_the_planted_kappa(kappa):
    sim = simulate(n_cells=4000, n_genes=60, n_types=4, kappa=kappa, seed=3)
    N, S = pooled(sim, np.asarray(sim.graph.in_edges @ sim.rho_true))
    rows = KB.type_bounds(N, S)
    values = [r["kappa_bar"] for r in rows]
    assert all(v is not None for v in values)
    assert min(values) >= kappa                     # every type's bound covers the truth
    assert min(values) < 1.0                        # and the bound says something


@pytest.mark.parametrize("kappa", [0.05, 0.4])
def test_counts_only_bound_holds_for_the_planted_kappa(kappa):
    adata = simulate(n_cells=4000, n_genes=60, n_types=4, kappa=kappa, seed=3).to_anndata()
    dc.build_graph(adata, "cell_type")
    out = dc.sensitivity.kappa_bar(adata, min_cells=50)["counts_only"]
    assert kappa <= out["min"] <= out["kappa_bar"]


def test_eligibility_and_the_counts_only_limits():
    N = np.array([[0.0, 40.0, 5.0]])
    S = np.array([[10.0, 400.0, 100.0]])
    rows = KB.type_bounds(N, S)
    assert rows[0]["n_genes"] == 2                  # S < 25 is not eligible
    a = 0.05 / 2
    expected = min(KB.poisson_upper(40, a) / 400, KB.poisson_upper(5, a) / 100)
    assert rows[0]["kappa_bar"] == pytest.approx(expected)
    sd = np.array([[1.0, 100.0, 5.0]])              # gene 1: sd / S = 0.25 > 0.2
    rows = KB.type_bounds(N, S, sd)
    assert rows[0]["n_genes"] == 1 and rows[0]["gene"] == 2
    a = 0.05 / 2                                    # one gene, split over two limits
    lower = 100 - stats.norm.ppf(1 - a) * 5.0
    assert rows[0]["kappa_bar"] == pytest.approx(KB.poisson_upper(5, a) / lower)


@pytest.fixture(scope="module")
def section():
    adata = simulate(n_cells=1500, n_genes=40, kappa=0.2, seed=4).to_anndata()
    dc.build_graph(adata, "cell_type")
    return adata, dc.fit(adata, dc.Config(**SMALL))


def test_kappa_bar_on_a_fit(section):
    adata, model = section
    out = dc.sensitivity.kappa_bar(adata, [model], min_cells=50)
    co, mb = out["counts_only"], out["model_based"]
    assert co["kappa_bar"] is not None and mb["kappa_bar"] is not None
    assert co["min"] <= co["kappa_bar"] and mb["min"] <= mb["kappa_bar"]
    assert mb["fits"] == [[0.1, 0]] and len(mb["per_fit_section"]) == 1
    assert mb["per_fit_section"][0] == pytest.approx(mb["kappa_bar"])
    for name in out["types"]:
        r = mb["types"][name]
        if r["kappa_bar"] is not None:
            assert r["per_fit"] == [r["kappa_bar"]] and r["gene"] in adata.var_names
    # counts-only needs no model
    alone = dc.sensitivity.kappa_bar(adata, min_cells=50)
    assert alone["model_based"] is None
    assert alone["counts_only"]["kappa_bar"] == co["kappa_bar"]
    # a type below min_cells never sets the section value
    assert dc.sensitivity.kappa_bar(adata, min_cells=10**6)["counts_only"]["kappa_bar"] is None


def test_streamed_rho_bar_matches_the_resident_runner(section):
    import torch

    adata, model = section
    data = model.assemble(adata)
    graph = dc.graph.kernel.read_graph(adata)
    t = data.t
    conn = graph.degrees > 0
    S = KB._model_S(model, adata, t, conn, len(data.type_names))
    runner = model.runner(data)
    ref = np.zeros_like(S)
    with torch.no_grad():
        for batch in runner.train_batches + runner.val_batches:
            fwd = runner.network(**runner._forward_kwargs(batch), kappa=model.config.kappa,
                                 sample=False)
            n = batch["n_seeds"]
            nodes = batch["nodes"][:n]
            ell = batch["x"][:n].double().sum(1, keepdim=True)
            part = (ell * fwd.rho_bar.double()).numpy()
            for i, node in enumerate(nodes):
                if conn[node]:
                    ref[t[node]] += part[i]
    np.testing.assert_allclose(S, ref, rtol=1e-10, atol=1e-8)


def member(status, kappa_star=None):
    return {"status": status, "kappa_star": kappa_star}


@pytest.mark.parametrize("status, kappa_star, bound, verdict", [
    ("above the grid", None, 0.18, "safe"),           # Ovarian FFPE, every finding
    ("breaks", 0.4, 0.38, "not ruled out"),           # lung tumor axis
    ("breaks", 0.2, 0.49, "not ruled out"),           # FF signaling-gene lean
    ("above the grid", None, 0.49, "undecided"),      # FF relocation findings
    ("breaks", 0.3, 0.18, "safe"),                    # holds at 0.2 >= 0.18
    ("breaks", 0.2, 0.18, "not ruled out"),           # fails at the first value >= 0.18
    ("breaks", 0.2, 0.2, "not ruled out"),
    ("breaks", 0.3, 0.2, "safe"),
    ("no finding", None, 0.18, None),
    ("gap at kappa = 0.2", None, 0.18, None),
])
def test_decision_rule(status, kappa_star, bound, verdict):
    table = {"grid": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
             "members": {"f": member(status, kappa_star)}}
    out = dc.sensitivity.decide(table, bound)["f"]
    assert out["verdict"] == verdict


def test_decide_reads_a_kappa_bar_result():
    table = {"grid": [0.0, 0.1, 0.2], "members": {"f": member("above the grid")}}
    result = {"counts_only": {"kappa_bar": 0.5}, "model_based": {"kappa_bar": 0.15}}
    assert dc.sensitivity.decide(table, result)["f"]["verdict"] == "safe"
    assert dc.sensitivity.decide(table, {"counts_only": {"kappa_bar": 0.5},
                                         "model_based": None})["f"]["verdict"] == "undecided"
    with pytest.raises(ValueError):
        dc.sensitivity.decide(table, {"counts_only": {"kappa_bar": None}, "model_based": None})
