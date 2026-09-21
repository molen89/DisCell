"""The trainer end to end on a small simulation: two epochs, every artefact."""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

from discell.model.prepare import ModelData, spatial_tiles
from discell.model.synthetic import simulate
from discell.model.train import TrainConfig, Trainer


@pytest.fixture(scope="module")
def small_fit(tmp_path_factory, monkeypatch_session=None):
    sim = simulate(n_cells=1500, n_types=4, kappa=0.1, seed=0)
    k = sim.n_types
    t = sim.t.astype(np.int64)
    v = np.hstack([sim.graph.y[:, :-1], sim.phi[:, :4]]).astype(np.float32)
    vbar = np.stack([v[t == g].mean(axis=0) for g in range(k)])
    tiles = spatial_tiles(sim.positions, 300)
    data = ModelData(
        graph=sim.graph, x=sp.csr_matrix(sim.x), t=t,
        phi=sim.phi, positions=sim.positions, totals=sim.totals,
        median_counts=float(np.median(sim.totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=np.array([f"type{g}" for g in range(k)]),
        v_block=v, vbar_t=vbar,
        train_tiles=tiles[:-1], val_tiles=tiles[-1:],
    )
    config = TrainConfig(
        dataset="synthetic-smoke", kappa=0.1, d_z=6, d_w=2, hidden=32, gat_dim=8, epochs=4, eval_every=2, figures_every=4,
        patience=100, device="cpu", alpha_z=0.007, alpha_w=0.1, alpha_a=0.3,
        v_pcs=4, invariance="closed_form",   # synthetic data carries no e_phi
    )
    from discell import paths

    root = tmp_path_factory.mktemp("runs")
    real = paths.dataset
    paths.dataset = lambda _: type("D", (), {"root": root})()   # run dir only
    try:
        trainer = Trainer(config, data)
        summary = trainer.fit()
    finally:
        paths.dataset = real
    return trainer, summary


def test_fit_completes_with_finite_metrics(small_fit):
    _, summary = small_fit
    final = summary["final"]
    assert np.isfinite(final["recon_val"])
    assert 0.0 <= final["nmi"] <= 1.0
    assert np.isfinite(final["probe"]["delta_ce"])
    assert len(final["kl_w_per_dim"]) == 2


def test_run_directory_holds_the_record(small_fit):
    trainer, _ = small_fit
    names = {p.name for p in trainer.run_dir.iterdir()}
    assert "config.json" in names
    assert "metrics.json" in names
    assert "best.pt" in names
    history = [json.loads(line) for line in
               (trainer.run_dir / "history.jsonl").read_text().splitlines()]
    assert len(history) >= 2                      # one entry per evaluation
    assert {"epoch", "recon_val", "nmi", "mirror", "probe"} <= set(history[0])
    assert any(n.startswith("events.out.tfevents") for n in names)
    config = json.loads((trainer.run_dir / "config.json").read_text())
    assert config["kappa"] == 0.1


def test_config_round_trips(small_fit):
    trainer, _ = small_fit
    stored = json.loads((trainer.run_dir / "config.json").read_text())
    assert stored.pop("git")            # provenance is recorded...
    rebuilt = TrainConfig(**stored)     # ...and the config still round-trips
    assert rebuilt == trainer.config


def test_every_cell_is_a_seed_exactly_once(small_fit):
    trainer, _ = small_fit
    seeds = np.concatenate(
        [b["nodes"][:b["n_seeds"]]
         for b in trainer.train_batches + trainer.val_batches])
    assert len(seeds) == trainer.data.n_cells
    assert len(np.unique(seeds)) == trainer.data.n_cells


def test_matched_correlation_recovers_permuted_and_flipped_columns():
    from discell.model.sweep import matched_correlation

    rng = np.random.default_rng(7)
    b = rng.standard_normal((60, 3))
    scrambled = np.stack([-2.0 * b[:, 2], 0.5 * b[:, 0], b[:, 1]], axis=1)
    assert matched_correlation(b, scrambled) > 0.999
    assert matched_correlation(b, rng.standard_normal((60, 3))) < 0.35


def test_degeneracy_diagnostics_are_recorded(small_fit):
    """Spec 7.10's pair lands in metrics.json and history.jsonl."""
    trainer, summary = small_fit
    final = summary["final"]
    assert {"degeneracy", "recon_gap"} <= set(summary["best"])   # selected epoch
    degeneracy, gap = final["degeneracy"], final["recon_gap"]
    assert np.isfinite(degeneracy["mi_ratio"])
    assert 0.0 <= degeneracy["within_var_fraction"] <= 1.0
    assert len(degeneracy["within_var_fraction_per_dim"]) == trainer.config.d_z
    assert gap["recon"] == final["recon_val"]
    assert np.isfinite(gap["gap"]) and np.isfinite(gap["recon_type_profile"])
    assert abs(gap["recon"] - gap["recon_typemean_z"] - gap["gap"]) < 1e-12
    history = [json.loads(line) for line in
               (trainer.run_dir / "history.jsonl").read_text().splitlines()]
    assert {"degeneracy", "recon_gap"} <= set(history[0])


def test_type_mean_decode_reproduces_the_forward_at_the_cells_own_z(small_fit):
    """The substituted decode path is the model's own: fed the cell's own
    mu_z it returns log_p exactly, so a z constant within type gives a gap of
    exactly zero; fed something else it changes the decode."""
    import torch

    trainer, _ = small_fit
    batch = trainer.val_batches[0]
    n = batch["n_seeds"]
    with torch.no_grad():
        fwd = trainer.model(**trainer._forward_kwargs(batch),
                            kappa=trainer.config.kappa, sample=False)
        own = trainer._decode_seeds(fwd, fwd.mu_z[:n], n)
        other = trainer._decode_seeds(fwd, torch.zeros_like(fwd.mu_z[:n]), n)
    assert torch.allclose(own, fwd.log_p, atol=1e-5)
    assert not torch.allclose(other, fwd.log_p, atol=1e-3)
    swept = trainer._sweep([batch], want_log_p=True,
                           z_bar=torch.zeros(len(trainer.data.p_t),
                                             trainer.config.d_z))
    assert swept["log_p_typemean"].shape == swept["log_p"].shape == (n, trainer.data.x.shape[1])


def test_w_substitution_reproduces_the_forward_at_the_cells_own_w(small_fit):
    """The w arm of the substituted decode is the model's own: fed the cell's
    own mu_w (with its own mu_z) it returns log_p exactly, so a w substitution
    that plants each cell's own w has a gap of exactly zero; fed the prior mean
    or a constant it moves. rho_bar and kappa are untouched throughout."""
    import torch

    trainer, _ = small_fit
    batch = trainer.val_batches[0]
    n = batch["n_seeds"]
    with torch.no_grad():
        fwd = trainer.model(**trainer._forward_kwargs(batch),
                            kappa=trainer.config.kappa, sample=False)
        planted = trainer._decode_seeds(fwd, fwd.mu_z[:n], n, fwd.mu_w[:n])
        prior = trainer._decode_seeds(fwd, fwd.mu_z[:n], n,
                                      fwd.prior_mean_w[:n])
        zeroed = trainer._decode_seeds(fwd, fwd.mu_z[:n], n,
                                       torch.zeros_like(fwd.mu_w[:n]))
    assert torch.allclose(planted, fwd.log_p, atol=1e-6)
    assert not torch.allclose(zeroed, fwd.log_p, atol=1e-3)
    assert torch.isfinite(prior).all()


def test_w_contribution_decodes_are_ordered_and_planted(small_fit):
    """``degeneracy.w_contribution``: the six decodes of todo 2.3, with the
    reference-context w built from the per-type mean context. The full decode
    must equal the trainer's own held-out recon, and the substitutions may only
    lose likelihood relative to it."""
    from discell.model.degeneracy import w_contribution

    trainer, summary = small_fit
    out = w_contribution(trainer)
    recon = out["recon"]
    assert abs(recon["full"] - summary["final"]["recon_val"]) < 1e-9
    for key in ("typemean_z", "typemean_z_prior_w", "typemean_z_ref_w",
                "own_z_ref_w"):
        assert np.isfinite(recon[key]) and recon[key] <= recon["full"] + 1e-9
    # the lookup reference is not a substitution in the fitted model, so it is
    # only required to be finite (a short fit can be beaten by it)
    assert np.isfinite(recon["type_profile"])
    assert abs(out["gaps"]["b_minus_d"]
               - (recon["typemean_z"] - recon["typemean_z_ref_w"])) < 1e-12
