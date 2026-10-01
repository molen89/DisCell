"""The trainer end to end on a small simulation: two epochs, every artefact."""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

from discell.model.prepare import ModelData, spatial_tiles
from discell.model.synthetic import simulate
from discell.model.train import TrainConfig, Trainer


def _small_data() -> ModelData:
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
    return data


@pytest.fixture(scope="module")
def small_fit(tmp_path_factory, monkeypatch_session=None):
    data = _small_data()
    config = TrainConfig(
        dataset="synthetic-smoke", kappa=0.1, d_z=6, d_w=2, hidden=32, gat_dim=8, epochs=4, eval_every=2, figures_every=4,
        patience=100, device="cpu", alpha_z=0.007, alpha_w=0.1, alpha_a=0.3,
        v_pcs=4, invariance="closed_form",   # synthetic data carries no e_phi
        # the pre-final value, explicit: a 4-epoch fit inside the default
        # 30-epoch warm-up would never reach a checkpoint
        w_warmup_epochs=0,
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


# -- the dead-context-channel detector (2026-09-23, FF best_s2) -------------

def _history(kl_sums, d_w=6, eval_every=5):
    """A planted ``history.jsonl``: one record per evaluation epoch, each with
    *kl_sum* spread evenly over *d_w* dimensions."""
    return [{"epoch": eval_every * (i + 1) - 1,
             "kl_w_per_dim": [s / d_w] * d_w}
            for i, s in enumerate(kl_sums)]


def test_dead_w_channel_fires_on_a_planted_zero_kl_history():
    from discell.model.train import dead_w_channel

    assert dead_w_channel(_history([0.0] * 20)) is True
    # FF best_s2: exactly 0.0 on every dimension from epoch 4 to the end
    assert dead_w_channel(_history([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) is True


def test_dead_w_channel_is_quiet_on_a_live_history():
    from discell.model.train import dead_w_channel

    # the healthy band at the opening evaluations: summed KL_w 3e-4 .. 8e-3
    assert dead_w_channel(_history([3e-4, 1e-3, 5e-3, 8e-3])) is False
    assert dead_w_channel(_history([])) is False
    # a channel that opens after the detector's window is not the failure:
    # only the opening epochs are checked
    assert dead_w_channel(_history([0.0] * 20)[4:]) is False


def test_kl_w_is_dead_uses_the_sum_not_the_per_dimension_value():
    from discell.model.train import kl_w_is_dead

    # six dimensions each below the threshold, but alive in sum
    assert kl_w_is_dead([4e-6] * 6, epoch=4) is False
    assert kl_w_is_dead([1e-7] * 6, epoch=4) is True


def test_metrics_json_carries_the_dead_w_flag(small_fit):
    trainer, summary = small_fit
    assert summary["dead_w_channel"] is False      # the synthetic fit is alive
    written = json.loads((trainer.run_dir / "metrics.json").read_text())
    assert written["dead_w_channel"] is False


def test_closing_evaluation_scores_the_accepted_checkpoint(tmp_path, monkeypatch):
    """Review R26: after early stopping, ``final`` must describe ``best.pt``.

    The first evaluation is made to win by fiat, so training runs on past the
    accepted checkpoint before patience stops it. The model left in memory must
    then be the accepted one, and the closing metrics must be its metrics.
    """
    import torch

    from discell import paths

    data = _small_data()
    config = TrainConfig(
        dataset="synthetic-smoke", kappa=0.1, d_z=6, d_w=2, hidden=32, gat_dim=8,
        epochs=8, eval_every=1, figures_every=1000, patience=2, device="cpu",
        alpha_z=0.007, alpha_w=0.1, alpha_a=0.3, v_pcs=4, invariance="closed_form",
        w_warmup_epochs=0,          # the pre-final value: epoch 0 may be best
    )
    monkeypatch.setattr(paths, "dataset", lambda _: type("D", (), {"root": tmp_path})())
    trainer = Trainer(config, data)
    real_evaluate = trainer.evaluate
    calls = []

    def rigged_evaluate():
        report = real_evaluate()
        calls.append(report)
        if len(calls) == 1:
            report["recon_val"] += 1e6          # the first evaluation is the best
        return report

    trainer.evaluate = rigged_evaluate
    summary = trainer.fit()

    assert summary["best"]["epoch"] == 0
    assert summary["last_epoch"] > summary["best"]["epoch"]   # trained on past it
    assert summary["final_epoch"] == summary["best"]["epoch"]

    saved = torch.load(trainer.run_dir / "best.pt", map_location="cpu",
                       weights_only=False)["model"]
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value.cpu(), saved[name]), name

    history = {row["epoch"]: row for row in (
        json.loads(line) for line in
        (trainer.run_dir / "history.jsonl").read_text().splitlines())}
    accepted, final = history[0], summary["final"]
    assert final["nmi"] == pytest.approx(accepted["nmi"])
    assert final["mirror"]["r2"] == pytest.approx(accepted["mirror"]["r2"])
    assert final["probe"]["delta_ce"] == pytest.approx(accepted["probe"]["delta_ce"])
    assert final["kl_w_per_dim"] == pytest.approx(accepted["kl_w_per_dim"])

    stored = json.loads((trainer.run_dir / "metrics.json").read_text())
    assert stored["final_epoch"] == stored["best"]["epoch"]
    assert stored["last_epoch"] == summary["last_epoch"]


# -- the final configuration as the code default (2026-09-28) ----------------

#: every finalL_s0 run, and the fields in which the default TrainConfig is
#: allowed to differ from it: the run's name, the reference budget (500/40,
#: figures every 100 -- flags, like --epochs) and, on the GSE core, the data
#: variant and tile size (data knobs, not model settings)
FINAL_RUNS = {
    "gse315411_pdltma06_11_prime_solo": {"variant": "pdl018d", "tile_cells": 2048},
    "xenium_prime_ovarian_cancer_ffpe": {},
    "xenium_prime_human_lung_cancer_ffpe": {},
    "xenium_prime_human_ovary_ff": {},
}
#: the budget is the default now (author's decision 2026-09-28): only the
#: name and the figure cadence remain
BUDGET = {"run_name": "finalL_s0", "figures_every": 100}


@pytest.mark.parametrize("dataset", sorted(FINAL_RUNS))
def test_default_config_is_finalL_s0(dataset):
    """TrainConfig() with alpha_z and the label column resolved as a fit
    resolves them equals the dataset's finalL_s0 config.json in every field
    but the name, the figure cadence and (GSE) the data knobs -- the
    500/40 budget included."""
    import dataclasses

    from discell import paths
    from discell.model.train import resolve_alpha_z, resolve_label_key

    stored = paths.dataset(dataset).root / "runs" / "finalL_s0" / "config.json"
    if not stored.exists():
        pytest.skip(f"no {stored}")
    final = json.loads(stored.read_text())
    final.pop("git")
    default = resolve_label_key(TrainConfig(dataset=dataset,
                                            **FINAL_RUNS[dataset]))
    default = dataclasses.replace(default,
                                  alpha_z=resolve_alpha_z(dataset, None))
    got = dataclasses.asdict(default)
    # fields added after the finalL fits, at their bit-identical defaults
    assert set(got) - set(final) == {"no_image"} and got["no_image"] is False
    assert set(final) <= set(got)
    differs = {k: final[k] for k in final if got[k] != final[k]}
    assert differs == BUDGET


def test_old_records_reload_at_the_pre_final_defaults():
    """A stored config that predates a field ran at that field's old default:
    config_from_record must fill in the old value, never today's."""
    from discell.model.train import config_from_record

    old = config_from_record({"dataset": "x", "kappa": 0.2})
    assert (old.label_key, old.adv_comp_weight, old.w_warmup_epochs,
            old.alpha_z) == (None, 1.0, 0, 0.007)
    assert old.kappa == 0.2
    # recorded values are taken as recorded, whatever the defaults
    recorded = {"dataset": "x", "label_key": "lineage", "adv_comp_weight": 3.0,
                "w_warmup_epochs": 30, "alpha_z": 0.0018}
    assert config_from_record(recorded) == TrainConfig(**recorded)


def test_alpha_z_none_is_resolved_at_setup_and_recorded(tmp_path, monkeypatch):
    """alpha_z None: 1/2 / mean count of the connected training cells on a
    dataset without a pinned value, the pinned value on one with it; the
    float goes into config.json and the config round-trips."""
    from discell import paths
    from discell.model.train import ALPHA_Z_PINNED

    monkeypatch.setattr(paths, "dataset",
                        lambda _: type("D", (), {"root": tmp_path})())
    data = _small_data()
    rows = np.concatenate(data.train_tiles)
    rows = rows[data.graph.degrees[rows] > 0]
    planted = 0.5 / float(np.asarray(data.totals, dtype=np.float64)[rows].mean())
    base = dict(kappa=0.1, d_z=6, d_w=2, hidden=32, gat_dim=8, epochs=1,
                eval_every=1, figures_every=1000, patience=100, device="cpu",
                v_pcs=4, invariance="closed_form")
    trainer = Trainer(TrainConfig(dataset="synthetic-smoke", run_name="az",
                                  **base), data)
    assert trainer.config.alpha_z == pytest.approx(planted, rel=1e-12)
    trainer.fit()
    stored = json.loads((trainer.run_dir / "config.json").read_text())
    assert stored["alpha_z"] == trainer.config.alpha_z
    stored.pop("git")
    assert TrainConfig(**stored) == trainer.config
    # a pinned dataset takes its table value, not the counts
    ovarian = "xenium_prime_ovarian_cancer_ffpe"
    pinned = Trainer(TrainConfig(dataset=ovarian, run_name="az_pinned", **base),
                     data)
    assert pinned.config.alpha_z == ALPHA_Z_PINNED[ovarian] == 0.0035
    # a float is used as given
    given = Trainer(TrainConfig(dataset=ovarian, run_name="az_given",
                                alpha_z=0.007, **base), data)
    assert given.config.alpha_z == 0.007


def test_label_key_falls_back_to_the_bundle_default_with_a_warning(
        tmp_path, monkeypatch, caplog):
    import dataclasses
    import logging

    import h5py

    from discell import paths
    from discell.model.train import resolve_label_key

    monkeypatch.setattr(paths, "dataset",
                        lambda _: type("D", (), {"bundle_dir": tmp_path})())

    def bundle(columns, default_label):
        with h5py.File(tmp_path / "full.h5ad", "w") as f:
            obs = f.create_group("obs")
            for column in columns:
                obs.create_dataset(column, data=np.zeros(3))
            uns = f.create_group("uns")
            if default_label is not None:
                uns.create_dataset("default_label", data=default_label)

    config = TrainConfig(dataset="d")
    assert config.label_key == "lineage"
    bundle(["cell_group", "lineage"], "cell_group")
    assert resolve_label_key(config) == config          # the column is there
    bundle(["graphclust"], "graphclust")
    with caplog.at_level(logging.WARNING, logger="discell.model.train"):
        fallen = resolve_label_key(config)
    assert fallen == dataclasses.replace(config, label_key="graphclust")
    assert "falls back" in caplog.text
    bundle(["cell_group"], None)                        # the loader's own default
    assert resolve_label_key(config).label_key == "cell_group"
    # any other key, and None, is left to the loader
    for key in ("cell_group", "graphclust", None):
        other = TrainConfig(dataset="d", label_key=key)
        assert resolve_label_key(other) is other
