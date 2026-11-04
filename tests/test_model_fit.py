"""``discell.fit`` end to end on a small simulation, and the trainer's selection rules."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pytest
import torch

import discell as dc
from discell.evaluate import heldout_probe
from discell.model.trainer import Trainer
from discell.sim import simulate

SMALL = dict(label_key="cell_type", image_key="X_image", d_z=4, d_w=2, hidden=32,
             gat_dim=8, v_pcs=4, tile_cells=300, epochs=6, eval_every=2,
             w_warmup_epochs=2, patience=100, device="cpu")


def small_adata(seed: int = 0, unassigned: int = 0):
    adata = simulate(n_cells=1500, n_genes=30, kappa=0.1, seed=seed).to_anndata()
    if unassigned:
        labels = adata.obs["cell_type"].astype(object).to_numpy(copy=True)
        labels[:unassigned] = None
        adata.obs["cell_type"] = labels
    dc.build_graph(adata, "cell_type")
    return adata


@pytest.fixture(scope="module")
def fitted():
    adata = small_adata(unassigned=40)
    model = dc.fit(adata, dc.Config(**SMALL))
    return adata, model


def test_fit_writes_the_anndata_outputs(fitted, tmp_path):
    adata, model = fitted
    n = adata.n_obs
    assert adata.obsm["X_discell_z"].shape == (n, 4)
    assert adata.obsm["X_discell_w"].shape == (n, 2)
    assert adata.obsm["X_discell_mpsi"].shape == (n, 2)
    assert adata.varm["discell_B"].shape == (adata.n_vars, 2)
    fit = adata.uns["discell"]["fit"]
    assert fit["alpha_z"] == pytest.approx(0.5 / np.median(np.asarray(adata.X.sum(1))))
    assert fit["kmeans_threads"] == 1 and fit["best_epoch"] == model.metrics["best"]["epoch"]
    assert dc.Config.from_dict(json.loads(fit["config"])) == model.config
    adata.write_h5ad(tmp_path / "fit.h5ad")
    back = ad.read_h5ad(tmp_path / "fit.h5ad")
    assert np.array_equal(back.obsm["X_discell_z"], adata.obsm["X_discell_z"])
    assert back.uns["discell"]["fit"]["recon_val"] == fit["recon_val"]


def test_metrics_record_the_closing_evaluation_and_the_mask(fitted):
    _, model = fitted
    final = model.metrics["final"]
    assert np.isfinite(final["recon_val"]) and 0.0 <= final["nmi"] <= 1.0
    for key in ("probe", "mirror", "degeneracy", "recon_gap", "w_channel", "nmi_targets",
                "recon_val_targets", "kl_w_per_dim"):
        assert key in final
    assert final["eval_mask"]["n_cells_excluded"] == 40
    assert final["eval_mask"]["excluded_present"] == ["Unassigned"]
    assert "latents" not in final
    assert len(model.history) == 3 and model.losses and "adv_head_loss" in model.losses[0]


def test_every_cell_is_a_seed_exactly_once(fitted):
    adata, model = fitted
    latents = model.latents(model.assemble(adata))
    assert np.array_equal(np.sort(latents["nodes"]), np.arange(adata.n_obs))


def test_save_load_and_encode_round_trip(fitted, tmp_path):
    adata, model = fitted
    model.save(tmp_path / "m")
    loaded = dc.Model.load(tmp_path / "m", device="cpu")
    assert loaded.config == model.config and loaded.type_names == model.type_names
    assert loaded.metrics["best"]["epoch"] == model.metrics["best"]["epoch"]
    other = adata.copy()
    del other.obsm["X_discell_z"]
    loaded.encode(other)
    assert np.array_equal(other.obsm["X_discell_z"], adata.obsm["X_discell_z"])
    assert np.array_equal(other.varm["discell_B"], adata.varm["discell_B"])
    assert {p.name for p in (tmp_path / "m").iterdir()} >= {
        "model.pt", "config.json", "metrics.json", "history.jsonl", "losses.jsonl"}


def test_evaluate_reproduces_the_closing_evaluation(fitted):
    adata, model = fitted
    report = model.evaluate(adata)
    final = model.metrics["final"]
    assert report["recon_val"] == final["recon_val"] and report["nmi"] == final["nmi"]
    assert report["degeneracy"]["mi"] == pytest.approx(final["degeneracy"]["mi"], abs=1e-12)


def test_heldout_probe_grades_target_cells(fitted):
    adata, model = fitted
    record = heldout_probe(model, adata, n_perm=2)
    assert record["n_cells"] == adata.n_obs - 40
    assert record["ridge"]["img"] is not None and np.isfinite(record["mlp"]["comp"]["excess"])
    assert record["eval_mask"]["n_cells_excluded"] == 40


def test_encode_refuses_another_type_vocabulary(fitted):
    _, model = fitted
    other = simulate(n_cells=800, n_genes=30, n_types=3, seed=1).to_anndata()
    dc.build_graph(other, "cell_type")
    with pytest.raises(ValueError, match="types"):
        model.encode(other)


def test_image_free_fit():
    adata = small_adata(seed=2)
    model = dc.fit(adata, dc.Config(**{**SMALL, "image_key": None}))
    assert model.network.phi_dim == 0 and adata.uns["discell"]["fit"]["image_free"]
    assert model.metrics["final"]["probe"]["baseline_ce"] > 0
    assert all(s["adv_excess_phi"] == 0.0 for s in model.losses)




def test_uncontrolled_fit_builds_no_adversary():
    adata = small_adata(seed=3)
    model = dc.fit(adata, dc.Config(**{**SMALL, "alpha_a": 0.0, "epochs": 2}))
    assert "adv_head_loss" not in model.losses[0]


def test_same_seed_same_fit():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)     # multithreaded CPU torch is not reproducible
    try:
        runs = [dc.fit(small_adata(seed=4), dc.Config(**{**SMALL, "epochs": 2}))
                for _ in range(2)]
    finally:
        torch.set_num_threads(threads)
    assert [s["loss"] for s in runs[0].losses] == [s["loss"] for s in runs[1].losses]


def test_fit_refuses_inconsistent_inputs():
    adata = simulate(n_cells=500, n_genes=20, seed=5).to_anndata()
    with pytest.raises(KeyError, match="build_graph"):
        dc.fit(adata, dc.Config(**SMALL))
    dc.build_graph(adata, "cell_type")
    with pytest.raises(ValueError, match="unassigned_label"):
        dc.fit(adata, dc.Config(**{**SMALL, "unassigned_label": "unknown"}))
    with pytest.raises(KeyError, match="obsm"):
        dc.fit(adata, dc.Config(**{**SMALL, "image_key": "X_missing"}))


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present: no fallback")
def test_missing_cuda_falls_back_to_the_cpu_loudly():
    adata = small_adata(seed=6)
    with pytest.warns(RuntimeWarning, match="CUDA"):
        dc.fit(adata, dc.Config(**{**SMALL, "epochs": 2, "device": "cuda"}))


# -- the selection rules, with evaluate's reads planted ------------------------------

def run_planted(monkeypatch, recon, nmi=lambda epoch: 0.5, kl=None, **overrides):
    """A fit whose in-loop reads are replaced by planted functions of the epoch
    (eval_every = 1, so the i-th in-loop evaluation is epoch i). Returns the
    model and the network state at every evaluation."""
    real = Trainer.evaluate
    states: list = []

    def planted(self, final=False):
        out = real(self, final=final)
        if final:
            return out
        epoch = len(states)
        states.append({k: v.clone() for k, v in self.network.state_dict().items()})
        out["recon_val"], out["nmi"] = recon(epoch), nmi(epoch)
        if kl is not None:
            out["kl_w_per_dim"] = kl(epoch)
        return out

    monkeypatch.setattr(Trainer, "evaluate", planted)
    config = dc.Config(**{**SMALL, "eval_every": 1, **overrides})
    return dc.fit(small_adata(seed=7), config), states


def test_the_checkpoint_never_comes_from_the_warm_up(monkeypatch):
    model, _ = run_planted(monkeypatch, lambda e: 100.0 if e < 4 else -float(e),
                           w_warmup_epochs=4, epochs=8)
    assert model.metrics["best"]["epoch"] == 4


def test_patience_does_not_run_during_the_warm_up(monkeypatch):
    model, states = run_planted(monkeypatch, lambda e: -5.0, w_warmup_epochs=5,
                                epochs=12, patience=2)
    assert model.metrics["best"]["epoch"] == 5
    assert model.metrics["last_epoch"] == 7 and len(states) == 8


def test_without_warm_up_the_early_best_is_taken(monkeypatch):
    model, _ = run_planted(monkeypatch, lambda e: 100.0 + e if e < 2 else -float(e),
                           w_warmup_epochs=0, epochs=6)
    assert model.metrics["best"]["epoch"] == 1


def test_the_nmi_guard_rejects_a_collapsed_checkpoint(monkeypatch):
    model, _ = run_planted(monkeypatch, lambda e: float(e),
                           nmi=lambda e: 0.4 if e == 5 else 0.5, w_warmup_epochs=0,
                           epochs=6)
    assert model.metrics["best"]["epoch"] == 4


def test_the_closing_evaluation_scores_the_accepted_checkpoint(monkeypatch):
    model, states = run_planted(monkeypatch, lambda e: 1.0 if e == 2 else 0.0,
                                w_warmup_epochs=0, epochs=6)
    assert model.metrics["best"]["epoch"] == 2 and model.metrics["final_epoch"] == 2
    final_state = model.network.state_dict()
    assert all(torch.equal(final_state[k], states[2][k]) for k in final_state)
    assert not all(torch.equal(final_state[k], states[5][k]) for k in final_state)


def test_the_dead_w_detector_flags_the_fit(monkeypatch):
    model, _ = run_planted(monkeypatch, lambda e: float(e),
                           kl=lambda e: [0.0, 0.0] if e == 1 else [0.1, 0.1], epochs=4)
    assert model.metrics["dead_w_channel"] is True


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fit_on_the_gpu():
    adata = small_adata(seed=8)
    model = dc.fit(adata, dc.Config(**{**SMALL, "device": "cuda"}))
    assert next(model.network.parameters()).is_cuda
    assert np.isfinite(adata.uns["discell"]["fit"]["recon_val"])
