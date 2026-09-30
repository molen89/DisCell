"""The context-latent grading (devlog 2026-09-30): a latent that carries
neighbour composition shows a positive composition gain and niche MI; the
same latent shuffled across cells sits at its floor; and DISCELL's mu_w,
graded here, reproduces the run's own w-channel guard (degeneracy.json)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import test_model_eval_mask as TE

from discell.experiments import context_grade as CG

synthetic_run = TE.synthetic_run


def _inputs(data):
    from discell.model.validate import niche_labels

    test = np.zeros(data.n_cells, dtype=bool)
    for tile in data.val_tiles:
        test[tile] = True
    return test, niche_labels(data, 4, 0)


def _read(data, latent, test, niche):
    return CG.grade(latent, data.t, ~test, test, data.v_block, data.vbar_t,
                    data.n_comp, niche, data.type_names, seed=0, n_perm=5)


def test_planted_composition_is_recovered_and_shuffled_is_not():
    import torch

    torch.set_num_threads(1)
    data = TE._world(TE.PLAIN)
    test, niche = _inputs(data)
    rng = np.random.default_rng(1)
    comp = data.v_block[:, :data.n_comp].astype(np.float64)
    planted = np.hstack([comp, rng.normal(size=(data.n_cells, 3))])
    planted += 0.05 * comp.std(axis=0).mean() * rng.normal(size=planted.shape)
    shuffled = planted[rng.permutation(data.n_cells)]

    got = _read(data, planted, test, niche)
    assert got["width"] == planted.shape[1]
    for family in ("ridge", "mlp"):
        block = got["probe"][family]["comp"]
        assert block["excess"] > 10 * max(block["floor_sd"], 1e-3), family
        assert block["var_fraction"] > 0.1, family
    assert got["niche_mi"]["w_niche_mi_excess"] > 0.1
    assert 0.0 <= got["nmi"] <= 1.0

    null = _read(data, shuffled, test, niche)
    for family in ("ridge", "mlp"):
        block = null["probe"][family]["comp"]
        assert abs(block["excess"]) < max(4 * block["floor_sd"], 0.01), family
    assert abs(null["niche_mi"]["w_niche_mi_excess"]) < 0.05


def test_resolvi_is_absent_not_zero(tmp_path):
    import anndata as ad

    latents = ad.AnnData(np.zeros((5, 1), dtype=np.float32))
    latents.obs_names = [f"cell_{i}" for i in range(5)]
    latents.obsm["X_resolvi"] = np.zeros((5, 3))
    latents.obsm["X_resolvi_mixture"] = np.full((5, 3), 1 / 3)
    latents.write_h5ad(tmp_path / "latents.h5ad")
    tool, key, rows, ctx = CG.load_context(tmp_path / "latents.h5ad")
    assert (tool, key, ctx) == ("resolvi", None, None)
    entry = CG.baseline_entry("unused", {
        "label": "resolVI", "entries": ["resolVI"],
        "latents": str(tmp_path / "latents.h5ad"), "split_run": "r",
        "split_from": "d", "record": "x", "intrinsic_probe": {}}, None, 5)
    assert entry["absent"] is True and "probe" not in entry


@pytest.mark.parametrize("synthetic_run", [(TE.WITH_U, False)], indirect=True)
def test_discell_mu_w_reproduces_the_w_guard(synthetic_run, monkeypatch):
    import torch

    from discell import paths
    from discell.model import degeneracy

    torch.set_num_threads(1)
    assert degeneracy.main(["--dataset", TE.DS, "--run", TE.RUN,
                            "--device", "cpu"]) == 0
    stored = json.loads((paths.dataset(TE.DS).root / "runs" / TE.RUN
                         / "degeneracy.json").read_text())["w_channel"]
    monkeypatch.setattr(CG, "RUNS", (TE.RUN,))
    assert CG.main(["niches", "--dataset", TE.DS]) == 0
    assert CG.main(["grade", "--dataset", TE.DS, "--only", "DISCELL"]) == 0
    out = json.loads((paths.dataset(TE.DS).root / "experiments"
                      / "context_grade.json").read_text())
    entry = out["methods"]["DISCELL"]["runs"][TE.RUN]
    assert entry["check_degeneracy"]["ok"]
    assert abs(entry["niche_mi"]["w_niche_mi_excess"]
               - stored["w_niche_mi_excess"]) <= CG.MI_TOL
    assert entry["niche_mi"]["n_cells"] == stored["n_cells"]
    assert entry["width"] == 2
