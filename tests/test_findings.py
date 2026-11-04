"""The paper's findings on a small simulated fit (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

import discell as dc
from discell.sensitivity import findings as F
from discell.sim import simulate

SMALL = dict(label_key="cell_type", image_key="X_image", d_z=4, d_w=2, hidden=32,
             gat_dim=8, v_pcs=4, tile_cells=300, epochs=4, eval_every=2,
             w_warmup_epochs=2, patience=100, device="cpu")
N = 20


def marked_adata(seed: int = 0):
    """A simulated section whose genes include the cycle markers and whose first
    type is a tumor type."""
    adata = simulate(n_cells=2400, n_genes=40, kappa=0.1, seed=seed).to_anndata()
    names = list(F.S_GENES[:12]) + list(F.G2M_GENES[:12]) + [f"gene{g}" for g in range(16)]
    adata.var_names = names
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    adata.obs["cell_type"] = np.where(labels == "type0", "Tumor", labels)
    dc.build_graph(adata, "cell_type")
    return adata


@pytest.fixture(scope="module")
def fitted():
    adata = marked_adata()
    return dc.fit(adata, dc.Config(**SMALL)), adata


def test_draw_seeds_follow_the_paper():
    assert F.draw_seed("cycle", 0) == 8190
    assert F.draw_seed("w_mi", 2) == 9192
    assert F.draw_seed("relocation_distribution", 1) == 14191


def test_cycle_scores_need_the_markers():
    adata = simulate(n_cells=300, n_genes=20, seed=1).to_anndata()
    assert F.cell_cycle_scores(adata.X, adata.var_names) is None
    marked = marked_adata()
    scores = F.cell_cycle_scores(marked.X, marked.var_names)
    assert scores.shape == (marked.n_obs, 2) and np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, F.cell_cycle_scores(marked.X, marked.var_names))


def test_cycling_set_takes_the_top_decile_of_held_out_cells():
    scores = np.stack([np.arange(100.0), np.zeros(100)], axis=1)
    held = np.arange(100) % 2 == 0
    eligible = np.ones(100, dtype=bool)
    eligible[98] = False
    out = F.cycling_set(scores, held, eligible)
    top = np.flatnonzero(out & held)
    assert len(top) == 5 and 98 not in top and top.min() == 88
    assert (np.flatnonzero(out & ~held) >= 88).all() and out[99]


def test_fit_draws_every_group(fitted, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "MIN_CYCLE_CELLS", 20)
    monkeypatch.setattr(F, "MIN_PAIR_GENES", 10)
    model, adata = fitted
    lr = list(adata.var_names[30:36])
    draws = F.fit_draws(model, adata, lr_genes=lr, n=N, out_dir=tmp_path, min_cells=20)
    flat = {k: v for members in draws.values() for k, v in members.items()}
    expected = {"cycle_check", "niche_information_check", "signaling_gene_lean",
                "relocation_finding", "moved_vs_type_mean", "moved_vs_random_cell"}
    assert expected <= set(flat)
    for name, entry in flat.items():
        assert np.isfinite(entry["estimate"]), name
        assert len(entry["draws"]) == N, name
    assert flat["signaling_gene_lean"]["method"] == "subsample"
    assert flat["niche_information_check"]["method"] == "percentile"
    for group in F.GROUPS:
        assert (tmp_path / f"{group}.npz").exists()
    # a second call reads the files instead of recomputing
    again = F.fit_draws(model, adata, lr_genes=lr, n=N, out_dir=tmp_path, min_cells=20)
    for group, members in draws.items():
        for name, entry in members.items():
            np.testing.assert_allclose(again[group][name]["draws"], entry["draws"])


def test_signaling_is_skipped_without_genes(fitted):
    model, adata = fitted
    out = F.fit_draws(model, adata, groups=("signaling", "w_mi"), n=N, min_cells=20)
    assert "signaling" not in out and "niche_information_check" in out["w_mi"]


def test_unknown_group_raises(fitted):
    model, adata = fitted
    with pytest.raises(ValueError):
        F.fit_draws(model, adata, groups=("marker",))


def test_finding_breakdown_over_a_small_sweep(tmp_path):
    adata = marked_adata(seed=2)
    cfg = dc.Config(**SMALL)
    kappas, seeds = (0.0, 0.2), (0, 1)
    models = dc.sensitivity.fit_sweep(adata, cfg, kappas=kappas, seeds=seeds,
                                      out_dir=tmp_path / "fits")
    dirs = {}
    for key, model in models.items():
        dirs[key] = tmp_path / "draws" / f"k{key[0]:g}_s{key[1]}"
        F.fit_draws(model, adata, groups=("w_mi", "relocation_mean"), n=N, out_dir=dirs[key],
                    min_cells=20)
    table = F.finding_breakdown(dirs, grid=kappas, seeds=seeds)
    assert set(table["members"]) == {"niche_information_check", "relocation_finding"}
    assert table["m_s"] == 2
    for member in table["members"].values():
        assert member["status"] in ("breaks", "above the grid", "no finding")
    assert table["members"]["niche_information_check"]["role"] == "check"
    assert table["members"]["relocation_finding"]["role"] == "finding"


def test_cell_cycle_r2_matches_the_cycle_check(fitted, monkeypatch):
    monkeypatch.setattr(F, "MIN_CYCLE_CELLS", 20)
    model, adata = fitted
    share = dc.evaluate.cell_cycle_r2(model, adata)
    point = F.fit_draws(model, adata, groups=("cycle",), n=N)["cycle"]["cycle_check"]
    assert share["n_held_out"] > 0 and np.isfinite([share["z"], share["w"]]).all()
    assert share["z"] - share["w"] == pytest.approx(point["estimate"], abs=1e-12)
    plain = simulate(n_cells=1200, n_genes=20, seed=1).to_anndata()
    dc.build_graph(plain, "cell_type")
    assert dc.evaluate.cell_cycle_r2(dc.fit(plain, dc.Config(**SMALL)), plain) is None


def test_finding_roles_follow_the_paper():
    assert len(F.FINDINGS) == 3 and len(F.CHECKS) == 2
    assert F.COUNTED_NOT_REPORTED == ("moved_vs_type_mean", "moved_vs_random_cell")
    members = set(F.FINDINGS) | set(F.CHECKS) | set(F.COUNTED_NOT_REPORTED)
    assert len(members) == 7
    assert set(F.PAPER_NAMES) == members
    assert set(F.POOL_ORDER) == members
    assert F.role("moved_vs_type_mean") == "counted, not reported"
    assert F.role("my_finding") is None
