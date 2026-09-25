"""The two metrics-package flags (devlog 2026-09-25): ``--phi-proj D`` and
``--time-only N``. Both default off, and off must be the pinned model."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from discell.model.networks import DisCell
from discell.model.train import TrainConfig, Trainer, build_parser

from tests.test_model_train import _small_data


def _model(seed: int = 0, **kw) -> DisCell:
    torch.manual_seed(seed)
    return DisCell(n_genes=30, n_types=4, phi_dim=16, median_counts=100.0,
                   d_z=6, d_w=2, hidden=32, gat_dim=8, **kw)


def test_phi_proj_off_is_the_pinned_model_bit_for_bit():
    pinned, off = _model(), _model(phi_proj=0)
    assert off.phi_proj is None
    a, b = pinned.state_dict(), off.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    # the next draw after construction is the same too: nothing was drawn
    assert torch.equal(torch.randn(3), (_model(), torch.randn(3))[1])


def test_phi_proj_puts_d_columns_of_phi_into_c():
    model = _model(phi_proj=4)
    assert model.phi_proj.weight.shape == (4, 16)
    n_ctx = 5
    t = torch.tensor([0, 1, 2, 3, 0, 1])
    phi = torch.randn(6, 16)
    c, _ = model.context(torch.zeros(6, 6), t, phi, torch.zeros(6, dtype=torch.bool),
                         torch.tensor([5, 4]), torch.tensor([0, 1]), n_ctx)
    assert c.shape == (n_ctx, 8 + 4 + 1)
    full, _ = _model().context(torch.zeros(6, 6), t, phi,
                               torch.zeros(6, dtype=torch.bool),
                               torch.tensor([5, 4]), torch.tensor([0, 1]), n_ctx)
    assert full.shape == (n_ctx, 8 + 16 + 1)
    # the checkpoint round-trips only into a model built with the same D
    _model(seed=1, phi_proj=4).load_state_dict(model.state_dict())
    with pytest.raises(RuntimeError):
        _model(seed=1).load_state_dict(model.state_dict())


def test_cli_parses_both_flags_and_defaults_are_off():
    args = vars(build_parser().parse_args(["--dataset", "x"]))
    assert args["phi_proj"] == 0 and args["time_only"] == 0
    args = vars(build_parser().parse_args(
        ["--dataset", "x", "--phi-proj", "32", "--time-only", "20"]))
    assert args["phi_proj"] == 32 and args["time_only"] == 20
    assert TrainConfig(dataset="x").phi_proj == 0


@pytest.fixture()
def run_root(tmp_path):
    from discell import paths

    real = paths.dataset
    paths.dataset = lambda _: type("D", (), {"root": tmp_path})()
    try:
        yield tmp_path
    finally:
        paths.dataset = real


def _config(**kw) -> TrainConfig:
    return TrainConfig(dataset="synthetic-smoke", kappa=0.1, d_z=6, d_w=2,
                       hidden=32, gat_dim=8, epochs=4, eval_every=2,
                       figures_every=4, patience=100, device="cpu", v_pcs=4,
                       invariance="closed_form", **kw)


def test_time_only_trains_without_evaluating_and_times_one_evaluation(run_root):
    trainer = Trainer(_config(run_name="timing"), _small_data())
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    calls = []
    real = trainer.evaluate
    trainer.evaluate = lambda: calls.append(1) or real()
    out = trainer.time_only(3)
    assert len(calls) == 1                      # one evaluation, after training
    assert out["n_epochs"] == 3 and len(out["epoch_s"]) == 3
    assert out["s_per_epoch"] > 0 and out["eval_s"] > 0
    assert out["peak_train_mib"] is None        # CPU: no GPU peak to read
    # the model was trained (parameters moved) but nothing was checkpointed
    after = trainer.model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    run_dir = run_root / "runs" / "timing"
    assert not (run_dir / "best.pt").exists()
    assert not (run_dir / "metrics.json").exists()
    assert json.loads((run_dir / "timing.json").read_text())["n_epochs"] == 3


def test_phi_proj_fits_end_to_end(run_root):
    trainer = Trainer(_config(run_name="proj", phi_proj=3), _small_data())
    summary = trainer.fit()
    assert np.isfinite(summary["best"]["recon_val"])
    saved = torch.load(run_root / "runs" / "proj" / "best.pt",
                       weights_only=False)
    assert saved["config"]["phi_proj"] == 3
    assert saved["model"]["phi_proj.weight"].shape == (3, trainer.data.phi.shape[1])
