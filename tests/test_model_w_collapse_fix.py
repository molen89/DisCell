"""The two context-collapse remedies of 2026-09-23: KL warm-up and free bits.

Same contract as the 6b.5 ablations: **off** reproduces the pinned loss bit for
bit, and **on** changes exactly the one term it names, by an amount predicted in
closed form from the tile's own components.
"""

from __future__ import annotations

import dataclasses as dc

import numpy as np
import torch

from discell.model.elbo import Weights, discell_loss
from discell.model.equations import gaussian_kl_per_dim
from discell.model.train import TrainConfig
from tests.test_model_ablations import _forward


def _tile():
    _, tensors, batch, fwd = _forward(seed=0)
    return (fwd, tensors["x"][: batch.n_seeds], tensors["t"][: batch.n_seeds],
            batch.n_seeds)


# -- free bits --------------------------------------------------------------

def test_free_bits_off_is_the_pinned_loss_bit_for_bit():
    fwd, x, t, _ = _tile()
    w = Weights(omega=1.0, alpha_z=0.007, alpha_w=0.1)
    assert w.w_free_bits == 0.0
    a = discell_loss(fwd, x, t, weights=w).loss
    b = discell_loss(fwd, x, t, weights=dc.replace(w, w_free_bits=0.0)).loss
    assert a.item() == b.item()


def test_free_bits_below_lambda_charge_nothing_and_above_it_are_linear():
    """Per dimension: 0 below lambda, KL_k - lambda above -- checked against the
    tile's own per-dimension KL_w, at a lambda under, between and over them."""
    fwd, x, t, n = _tile()
    per_dim = gaussian_kl_per_dim(fwd.mu_w[:n], fwd.logvar_w[:n],
                                  fwd.prior_mean_w[:n], 0.0).mean(0)
    per_dim = per_dim.detach().numpy()
    assert (per_dim > 0).all()
    for lam in (0.0 + 1e-6, float(per_dim.min()) * 0.5,
                float(np.median(per_dim)), float(per_dim.max()) * 2.0):
        w = Weights(omega=1.0, alpha_z=0.007, alpha_w=0.1, w_free_bits=lam)
        terms = discell_loss(fwd, x, t, weights=w)
        planted = np.maximum(per_dim - lam, 0.0).sum()
        assert np.isclose(terms.kl_w_charged, planted, rtol=1e-5, atol=1e-7)
        # and above every dimension the term is gone entirely
        if lam > per_dim.max():
            assert terms.kl_w_charged == 0.0


def test_free_bits_move_only_the_kl_w_term_of_the_loss():
    fwd, x, t, _ = _tile()
    w = Weights(omega=1.0, alpha_z=0.007, alpha_w=0.1)
    off = discell_loss(fwd, x, t, weights=w)
    on = discell_loss(fwd, x, t, weights=dc.replace(w, w_free_bits=0.02))
    planted = w.alpha_w * (off.kl_w_charged - on.kl_w_charged)
    assert planted > 0
    assert np.isclose(off.loss.item() - on.loss.item(), planted,
                      rtol=1e-3, atol=1e-7)
    for key in ("recon_a", "recon_b", "kl_z", "kl_w", "w_penalty"):
        assert off.scalars()[key] == on.scalars()[key]


def test_free_bits_at_zero_lambda_leave_the_raw_kl_w_untouched():
    """kl_w -- what history.jsonl and the dead-channel guard read -- is always
    the raw summed KL, free bits on or off."""
    fwd, x, t, _ = _tile()
    w = Weights(omega=1.0, alpha_z=0.007, alpha_w=0.1)
    off = discell_loss(fwd, x, t, weights=w)
    on = discell_loss(fwd, x, t, weights=dc.replace(w, w_free_bits=0.05))
    assert on.kl_w == off.kl_w
    assert off.kl_w_charged == off.kl_w


def test_free_bits_keep_the_gradient_on_the_charged_dimensions_only():
    fwd, x, t, n = _tile()
    per_dim = gaussian_kl_per_dim(fwd.mu_w[:n], fwd.logvar_w[:n],
                                  fwd.prior_mean_w[:n], 0.0).mean(0)
    lam = float(np.median(per_dim.detach().numpy()))
    charged = (per_dim - lam).clamp(min=0.0).sum()
    grad, = torch.autograd.grad(charged, fwd.mu_w, retain_graph=True)
    dead = (per_dim.detach() <= lam)
    assert dead.any() and (~dead).any()
    assert grad[:n][:, dead].abs().sum().item() == 0.0
    assert grad[:n][:, ~dead].abs().sum().item() > 0.0


# -- warm-up ----------------------------------------------------------------

def test_warmup_off_is_the_constant_alpha_w_at_every_epoch():
    config = TrainConfig(dataset="x", alpha_w=0.1)
    assert config.w_warmup_epochs == 0
    for epoch in (0, 1, 15, 199):
        assert config.alpha_w_at(epoch) == config.alpha_w
        assert config.weights(epoch) == config.weights()
        assert config.weights(epoch).alpha_w == 0.1


def test_warmup_schedule_at_zero_half_full_and_double_n():
    n, alpha = 30, 0.3
    config = TrainConfig(dataset="x", alpha_w=alpha, w_warmup_epochs=n)
    assert config.alpha_w_at(0) == 0.0
    assert config.alpha_w_at(n // 2) == alpha * 0.5
    assert config.alpha_w_at(n) == alpha
    assert config.alpha_w_at(2 * n) == alpha
    # linear in between, and never above the target after it
    for epoch in range(n):
        assert np.isclose(config.alpha_w_at(epoch), alpha * epoch / n)
    for epoch in range(n, 3 * n):
        assert config.alpha_w_at(epoch) == alpha


def test_warmup_scales_exactly_the_kl_w_term_and_nothing_else():
    fwd, x, t, _ = _tile()
    config = TrainConfig(dataset="x", alpha_w=0.3, alpha_z=0.007,
                         w_warmup_epochs=30)
    full = discell_loss(fwd, x, t, weights=config.weights(30))
    half = discell_loss(fwd, x, t, weights=config.weights(15))
    planted = 0.5 * config.alpha_w * full.kl_w
    assert np.isclose(full.loss.item() - half.loss.item(), planted,
                      rtol=1e-3, atol=1e-7)
    for key in ("recon_a", "recon_b", "kl_z", "kl_w", "w_penalty"):
        assert full.scalars()[key] == half.scalars()[key]
    # epoch 0 charges no KL_w at all
    zero = discell_loss(fwd, x, t, weights=config.weights(0))
    assert np.isclose(full.loss.item() - zero.loss.item(),
                      config.alpha_w * full.kl_w, rtol=1e-3, atol=1e-7)


# -- the wiring -------------------------------------------------------------

def test_trainer_wiring_reaches_exactly_the_two_consumers():
    base = TrainConfig(dataset="x")
    assert (base.w_warmup_epochs, base.w_free_bits) == (0, 0.0)
    assert base.weights().w_free_bits == 0.0

    fb = TrainConfig(dataset="x", w_free_bits=0.05)
    assert fb.weights().w_free_bits == 0.05
    # free bits never touch the schedule
    assert fb.alpha_w_at(0) == fb.alpha_w

    wu = TrainConfig(dataset="x", alpha_w=0.1, w_warmup_epochs=30)
    # warm-up never touches Weights except through alpha_w
    assert wu.weights(0) == dc.replace(base.weights(), alpha_w=0.0)
    assert wu.weights(30) == base.weights()

    both = TrainConfig(dataset="x", alpha_w=0.1, w_warmup_epochs=30,
                       w_free_bits=0.05)
    assert both.weights(15) == dc.replace(base.weights(), alpha_w=0.05,
                                          w_free_bits=0.05)


def test_both_flags_together_compose_multiplicatively_on_the_charged_kl():
    fwd, x, t, n = _tile()
    per_dim = gaussian_kl_per_dim(fwd.mu_w[:n], fwd.logvar_w[:n],
                                  fwd.prior_mean_w[:n], 0.0).mean(0)
    lam = float(np.median(per_dim.detach().numpy()))
    config = TrainConfig(dataset="x", alpha_w=0.3, alpha_z=0.007,
                         w_warmup_epochs=30, w_free_bits=lam)
    both = discell_loss(fwd, x, t, weights=config.weights(15))
    neither = discell_loss(fwd, x, t, weights=dc.replace(
        config.weights(30), alpha_w=0.0, w_free_bits=0.0))
    planted = 0.5 * config.alpha_w * float(
        (per_dim - lam).clamp(min=0.0).sum().detach())
    assert np.isclose(both.loss.item() - neither.loss.item(), planted,
                      rtol=1e-3, atol=1e-7)


def test_the_cli_carries_both_flags_into_the_config():
    from discell.model.train import build_parser

    args = vars(build_parser().parse_args(
        ["--dataset", "d", "--w-warmup-epochs", "30", "--w-free-bits", "0.05"]))
    args.pop("quiet")
    config = TrainConfig(**args)
    assert config.w_warmup_epochs == 30 and config.w_free_bits == 0.05
    defaults = vars(build_parser().parse_args(["--dataset", "d"]))
    defaults.pop("quiet")
    assert TrainConfig(**defaults).weights() == TrainConfig(dataset="d").weights()


# -- end to end: a tiny fit with both flags on ------------------------------

def _smoke_fit(tmp_path, **config_kw):
    """The synthetic smoke fit of tests/test_model_train.py, 6 epochs."""
    import numpy as np
    import scipy.sparse as sp

    from discell import paths
    from discell.model.prepare import ModelData, spatial_tiles
    from discell.model.synthetic import simulate
    from discell.model.train import Trainer

    sim = simulate(n_cells=1500, n_types=4, kappa=0.1, seed=0)
    k, t = sim.n_types, sim.t.astype(np.int64)
    v = np.hstack([sim.graph.y[:, :-1], sim.phi[:, :4]]).astype(np.float32)
    tiles = spatial_tiles(sim.positions, 300)
    data = ModelData(
        graph=sim.graph, x=sp.csr_matrix(sim.x), t=t, phi=sim.phi,
        positions=sim.positions, totals=sim.totals,
        median_counts=float(np.median(sim.totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=np.array([f"type{g}" for g in range(k)]),
        v_block=v,
        vbar_t=np.stack([v[t == g].mean(axis=0) for g in range(k)]),
        train_tiles=tiles[:-1], val_tiles=tiles[-1:])
    defaults = dict(epochs=6, eval_every=2, patience=100)
    defaults.update(config_kw)
    config = TrainConfig(
        dataset="synthetic-smoke", kappa=0.1, d_z=6, d_w=2, hidden=32,
        gat_dim=8, figures_every=defaults["epochs"],
        device="cpu", alpha_z=0.007, alpha_w=0.1, alpha_a=0.3, v_pcs=4,
        invariance="closed_form", **defaults)
    real = paths.dataset
    paths.dataset = lambda _: type("D", (), {"root": tmp_path})()
    try:
        trainer = Trainer(config, data)
        trainer.fit()
    finally:
        paths.dataset = real
    return trainer


def test_a_real_fit_logs_the_warmed_up_alpha_w_and_records_both_flags(tmp_path):
    import json

    trainer = _smoke_fit(tmp_path, w_warmup_epochs=4, w_free_bits=0.05)
    run_dir = trainer.run_dir
    config = json.loads((run_dir / "config.json").read_text())
    assert config["w_warmup_epochs"] == 4 and config["w_free_bits"] == 0.05

    history = [json.loads(line) for line
               in (run_dir / "history.jsonl").read_text().splitlines()]
    schedule = {r["epoch"]: r["alpha_w_eff"] for r in history}
    assert schedule, "history carries no alpha_w_eff"
    for epoch, value in schedule.items():
        assert np.isclose(value, 0.1 * min(1.0, epoch / 4))
    # the trainer really stepped with the schedule, not with the constant
    assert trainer.config.weights(1).alpha_w == 0.1 * 0.25


def test_a_default_fit_logs_the_constant_alpha_w(tmp_path):
    import json

    trainer = _smoke_fit(tmp_path)
    history = [json.loads(line) for line
               in (trainer.run_dir / "history.jsonl").read_text().splitlines()]
    assert all(r["alpha_w_eff"] == 0.1 for r in history)
    assert all(r["alpha_z_eff"] == 0.007 for r in history)
    config = json.loads((trainer.run_dir / "config.json").read_text())
    assert config["w_warmup_epochs"] == 0 and config["w_free_bits"] == 0.0
    assert config["kl_warmup_epochs"] == 0


# -- the warm-up era is not eligible for the checkpoint ---------------------

def _fit_recording_selection(tmp_path, monkeypatch, recon_by_epoch,
                             nmi_by_epoch=lambda epoch: 0.5, **config_kw):
    """Run the smoke fit with `evaluate` replaced by a planted sequence.

    The plant makes the warm-up era look irresistible -- the best recon of the
    whole run sits inside it -- so a `best` from the warm-up era, or a patience
    counter started there, shows up as a failed assertion rather than as a
    judgement call about a real fit.
    """
    from discell.model.train import Trainer

    real = Trainer.evaluate
    seen = []

    def fake(self):
        out = real(self)
        seen.append(self.epoch)
        out["recon_val"] = recon_by_epoch(self.epoch, out["recon_val"])
        out["nmi"] = nmi_by_epoch(self.epoch)
        return out

    monkeypatch.setattr(Trainer, "evaluate", fake)
    trainer = _smoke_fit(tmp_path, **config_kw)
    return trainer, seen


def test_the_checkpoint_never_comes_from_the_warm_up_era(tmp_path, monkeypatch):
    """Best recon planted at epoch 1, inside a 4-epoch warm-up: `best` must
    still be a post-warm-up epoch."""
    import json

    trainer, _ = _fit_recording_selection(
        tmp_path, monkeypatch,
        lambda epoch, real: 100.0 if epoch < 4 else real,
        w_warmup_epochs=4, epochs=8, eval_every=1)
    summary = json.loads((trainer.run_dir / "metrics.json").read_text())
    assert summary["best"]["epoch"] >= 4, (
        "best came from the warm-up era: " + str(summary["best"]["epoch"]))
    assert summary["best"]["recon_val"] != 100.0


def test_patience_does_not_run_during_the_warm_up(tmp_path, monkeypatch):
    """A flat, never-improving recon through a warm-up longer than the patience
    window must not early-stop before the warm-up ends."""
    import json

    # patience 2 with eval_every 1 = 2 stale evaluations; the warm-up is 5 long,
    # so an ungated counter would stop at epoch 2, well inside it
    trainer, seen = _fit_recording_selection(
        tmp_path, monkeypatch, lambda epoch, real: -5.0,
        w_warmup_epochs=5, epochs=9, eval_every=1, patience=2)
    assert max(seen) >= 5, f"stopped inside the warm-up: reached epoch {max(seen)}"
    summary = json.loads((trainer.run_dir / "metrics.json").read_text())
    assert summary["best"]["epoch"] >= 5


def test_without_warm_up_selection_and_patience_are_unchanged(tmp_path,
                                                              monkeypatch):
    """The gate is inert at w_warmup_epochs = 0: the planted early best is
    taken, exactly as before the change."""
    import json

    trainer, _ = _fit_recording_selection(
        tmp_path, monkeypatch,
        lambda epoch, real: 100.0 if epoch < 4 else real,
        epochs=8, eval_every=1)
    summary = json.loads((trainer.run_dir / "metrics.json").read_text())
    assert summary["best"]["epoch"] < 4 and summary["best"]["recon_val"] == 100.0


# == warm-up on BOTH KL terms (8.9b pre-registration, 2026-09-24) ============
#
# --kl-warmup-epochs N: alpha_z_eff = alpha_z * min(1, epoch / N) on both
# copies of the z-divergence (the (1 + omega) factor stays) and alpha_w_eff
# likewise on the w-divergence; nothing else moves. Same contract as above:
# off is the pinned loss bit for bit, on is predicted in closed form.

def _pinned_weights(config):
    """The pinned objective's weights, built by hand from the constants --
    what TrainConfig.weights() returned before any warm-up existed."""
    return Weights(omega=config.omega, alpha_z=config.alpha_z,
                   alpha_w=config.alpha_w, alpha_a=config.alpha_a,
                   lambda_w=config.lambda_w, w_penalty=config.w_penalty,
                   second_kl=config.second_kl, w_free_bits=config.w_free_bits)


def test_kl_warmup_off_is_the_pinned_loss_bit_for_bit():
    fwd, x, t, _ = _tile()
    config = TrainConfig(dataset="x", alpha_z=0.00035, alpha_w=0.1)
    assert config.kl_warmup_epochs == 0 and config.warmup_epochs == 0
    pinned = discell_loss(fwd, x, t, weights=_pinned_weights(config)).loss
    for epoch in (0, 1, 15, 30, 199):
        assert config.alpha_z_at(epoch) == config.alpha_z
        assert config.alpha_w_at(epoch) == config.alpha_w
        assert config.weights(epoch) == _pinned_weights(config)
        loss = discell_loss(fwd, x, t, weights=config.weights(epoch)).loss
        assert torch.equal(loss, pinned)


def test_kl_warmup_schedule_at_zero_half_full_and_double_n_for_both_terms():
    n, az, aw = 30, 0.00035, 0.1
    config = TrainConfig(dataset="x", alpha_z=az, alpha_w=aw,
                         kl_warmup_epochs=n)
    assert config.warmup_epochs == n
    for alpha_at, target in ((config.alpha_z_at, az), (config.alpha_w_at, aw)):
        assert alpha_at(0) == 0.0
        assert alpha_at(n // 2) == target * 0.5
        assert alpha_at(n) == target
        assert alpha_at(2 * n) == target
        for epoch in range(n):
            assert np.isclose(alpha_at(epoch), target * epoch / n)
        for epoch in range(n, 3 * n):
            assert alpha_at(epoch) == target
    # the Weights the trainer steps with carry exactly these two values
    for epoch in (0, n // 2, n, 2 * n):
        w = config.weights(epoch)
        assert (w.alpha_z, w.alpha_w) == (config.alpha_z_at(epoch),
                                          config.alpha_w_at(epoch))


def test_kl_warmup_scales_exactly_the_two_kl_terms_and_nothing_else():
    fwd, x, t, _ = _tile()
    for omega in (1.0, 0.5):
        config = TrainConfig(dataset="x", omega=omega, alpha_z=0.05,
                             alpha_w=0.3, alpha_a=0.3, kl_warmup_epochs=30)
        # the Weights differ from the pinned ones in alpha_z and alpha_w only
        for epoch, f in ((0, 0.0), (15, 0.5), (30, 1.0), (60, 1.0)):
            assert config.weights(epoch) == dc.replace(
                _pinned_weights(config), alpha_z=f * config.alpha_z,
                alpha_w=f * config.alpha_w)
        full = discell_loss(fwd, x, t, weights=config.weights(30))
        half = discell_loss(fwd, x, t, weights=config.weights(15))
        zero = discell_loss(fwd, x, t, weights=config.weights(0))
        z_term = (1.0 + omega) * config.alpha_z * full.kl_z
        w_term = config.alpha_w * full.kl_w
        assert np.isclose(full.loss.item() - half.loss.item(),
                          0.5 * (z_term + w_term), rtol=1e-3, atol=1e-7)
        assert np.isclose(full.loss.item() - zero.loss.item(),
                          z_term + w_term, rtol=1e-3, atol=1e-7)
        # the z share on its own: both copies of the z-KL are halved, i.e. the
        # (1 + omega) factor stays and multiplies the warmed-up alpha_z
        z_only = discell_loss(fwd, x, t, weights=dc.replace(
            config.weights(30), alpha_z=0.5 * config.alpha_z))
        assert np.isclose(full.loss.item() - z_only.loss.item(),
                          0.5 * z_term, rtol=1e-3, atol=1e-7)
        # epoch 0 charges no KL at all: the loss is the two reconstructions
        assert np.isclose(zero.loss.item(),
                          -(zero.recon_a + omega * zero.recon_b),
                          rtol=1e-6, atol=1e-6)
        for key in ("recon_a", "recon_b", "kl_z", "kl_w", "kl_w_charged",
                    "w_penalty"):
            assert full.scalars()[key] == half.scalars()[key] \
                == zero.scalars()[key]


def test_kl_warmup_keeps_the_single_copy_under_the_second_kl_ablation():
    """With second_kl=False the z-KL is charged once; the warm-up scales that
    one copy and does not reintroduce the second."""
    fwd, x, t, _ = _tile()
    config = TrainConfig(dataset="x", alpha_z=0.05, alpha_w=0.3,
                         second_kl=False, kl_warmup_epochs=30)
    full = discell_loss(fwd, x, t, weights=config.weights(30))
    half = discell_loss(fwd, x, t, weights=config.weights(15))
    planted = 0.5 * (config.alpha_z * full.kl_z + config.alpha_w * full.kl_w)
    assert np.isclose(full.loss.item() - half.loss.item(), planted,
                      rtol=1e-3, atol=1e-7)


def test_kl_and_w_warmup_are_mutually_exclusive():
    import pytest

    from discell.model.train import main

    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainConfig(dataset="x", w_warmup_epochs=30, kl_warmup_epochs=30)
    # the CLI refuses before any data is touched
    with pytest.raises(ValueError, match="mutually exclusive"):
        main(["--dataset", "no-such-dataset", "--w-warmup-epochs", "30",
              "--kl-warmup-epochs", "30"])
    # each alone is fine, and the w-only warm-up never touches alpha_z
    wu = TrainConfig(dataset="x", alpha_z=0.007, w_warmup_epochs=30)
    assert wu.alpha_z_at(0) == 0.007 and wu.weights(0).alpha_z == 0.007
    assert TrainConfig(dataset="x", kl_warmup_epochs=30).warmup_epochs == 30


def test_the_cli_carries_the_kl_warmup_flag_into_the_config():
    from discell.model.train import build_parser

    args = vars(build_parser().parse_args(
        ["--dataset", "d", "--kl-warmup-epochs", "30"]))
    args.pop("quiet")
    config = TrainConfig(**args)
    assert config.kl_warmup_epochs == 30 and config.w_warmup_epochs == 0
    assert config.warmup_epochs == 30
    defaults = vars(build_parser().parse_args(["--dataset", "d"]))
    defaults.pop("quiet")
    assert TrainConfig(**defaults).kl_warmup_epochs == 0


def test_a_real_fit_steps_and_logs_both_warmed_up_alphas(tmp_path,
                                                         monkeypatch):
    """The trainer's own loss calls carry the schedule on both alphas, every
    history record logs alpha_z_eff and alpha_w_eff, and config.json records
    the flag."""
    import json

    import discell.model.train as train_module

    real_loss, stepped = train_module.discell_loss, []

    def recording(*args, **kwargs):
        stepped.append(kwargs["weights"])
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(train_module, "discell_loss", recording)
    trainer = _smoke_fit(tmp_path, kl_warmup_epochs=4)
    config = json.loads((trainer.run_dir / "config.json").read_text())
    assert config["kl_warmup_epochs"] == 4 and config["w_warmup_epochs"] == 0

    history = [json.loads(line) for line
               in (trainer.run_dir / "history.jsonl").read_text().splitlines()]
    assert history
    for r in history:
        f = min(1.0, r["epoch"] / 4)
        assert np.isclose(r["alpha_z_eff"], 0.007 * f)
        assert np.isclose(r["alpha_w_eff"], 0.1 * f)
    # the steps really used the ramp: epoch 0 charged no KL, and the ramp
    # reached the constants by the end
    alphas = sorted({(w.alpha_z, w.alpha_w) for w in stepped})
    assert alphas[0] == (0.0, 0.0)
    assert alphas[-1] == (0.007, 0.1)
    assert (0.007 * 0.5, 0.1 * 0.5) in alphas
    assert all(dc.replace(w, alpha_z=0.007, alpha_w=0.1)
               == trainer.config.weights() for w in stepped)


def test_the_checkpoint_never_comes_from_the_kl_warm_up_era(tmp_path,
                                                             monkeypatch):
    import json

    trainer, _ = _fit_recording_selection(
        tmp_path, monkeypatch,
        lambda epoch, real: 100.0 if epoch < 4 else real,
        kl_warmup_epochs=4, epochs=8, eval_every=1)
    summary = json.loads((trainer.run_dir / "metrics.json").read_text())
    assert summary["best"]["epoch"] >= 4, (
        "best came from the warm-up era: " + str(summary["best"]["epoch"]))
    assert summary["best"]["recon_val"] != 100.0


def test_patience_does_not_run_during_the_kl_warm_up(tmp_path, monkeypatch):
    import json

    trainer, seen = _fit_recording_selection(
        tmp_path, monkeypatch, lambda epoch, real: -5.0,
        kl_warmup_epochs=5, epochs=9, eval_every=1, patience=2)
    assert max(seen) >= 5, f"stopped inside the warm-up: reached epoch {max(seen)}"
    summary = json.loads((trainer.run_dir / "metrics.json").read_text())
    assert summary["best"]["epoch"] >= 5


def test_the_nmi_guard_reference_ignores_the_kl_warm_up_era(tmp_path,
                                                             monkeypatch):
    """NMI 1.0 planted inside the warm-up, 0.5 after it. Were the running max
    updated during the warm-up, every later epoch would fail the 0.9 guard and
    the run would end with no checkpoint (best epoch -1)."""
    import json

    trainer, _ = _fit_recording_selection(
        tmp_path, monkeypatch, lambda epoch, real: real,
        nmi_by_epoch=lambda epoch: 1.0 if epoch < 4 else 0.5,
        kl_warmup_epochs=4, epochs=8, eval_every=1)
    summary = json.loads((trainer.run_dir / "metrics.json").read_text())
    assert summary["best"]["epoch"] >= 4
    assert summary["best"]["nmi"] == 0.5
