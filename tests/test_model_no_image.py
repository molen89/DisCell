"""The no-image ablation (devlog 2026-10-01): ``--no-image`` drops Phi from
the context c, so c = GAT (+) isolated flag. Off must be the pinned model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from discell.model.networks import DisCell
from discell.model.train import TrainConfig, Trainer, build_parser

from tests.test_model_phi_proj_timing import _config, _model, run_root  # noqa: F401
from tests.test_model_train import _small_data


def _ctx(model, phi):
    t = torch.tensor([0, 1, 2, 3, 0, 1])
    return model.context(torch.zeros(6, 6), t, phi,
                         torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.bool),
                         torch.tensor([5, 4]), torch.tensor([0, 1]), 5)[0]


def test_no_image_off_is_the_pinned_model_bit_for_bit():
    pinned, off = _model(), _model(no_image=False)
    a, b = pinned.state_dict(), off.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    phi = torch.randn(6, 16)
    assert torch.equal(_ctx(pinned, phi), _ctx(off, phi))


def test_no_image_c_is_the_gat_and_the_flag_and_ignores_phi():
    model = _model(no_image=True)
    c = _ctx(model, torch.randn(6, 16))
    assert c.shape == (5, 8 + 1)
    assert torch.equal(c, _ctx(model, torch.randn(6, 16)))   # Phi is not read
    full = _ctx(_model(), torch.randn(6, 16))
    assert torch.equal(c[:, :8], full[:, :8])                 # same GAT part
    assert torch.equal(c[:, -1], full[:, -1])                 # same flag
    # the c-reading layers are sized without Phi
    assert model.prior_w[0].in_features == 8 + 1 + 4
    with pytest.raises(RuntimeError):
        _model().load_state_dict(model.state_dict())


def test_no_image_refuses_the_phi_reading_arms():
    with pytest.raises(ValueError):
        _model(no_image=True, phi_proj=4)
    with pytest.raises(ValueError):
        _model(no_image=True, query="image")


def test_cli_flag_and_default():
    assert vars(build_parser().parse_args(["--dataset", "x"]))["no_image"] is False
    assert vars(build_parser().parse_args(
        ["--dataset", "x", "--no-image"]))["no_image"] is True
    assert TrainConfig(dataset="x").no_image is False


def test_no_image_fits_end_to_end_and_reloads(run_root):  # noqa: F811
    trainer = Trainer(_config(run_name="noimg", no_image=True), _small_data())
    summary = trainer.fit()
    assert np.isfinite(summary["best"]["recon_val"])
    saved = torch.load(run_root / "runs" / "noimg" / "best.pt",
                       weights_only=False)
    assert saved["config"]["no_image"] is True
    assert saved["model"]["prior_w.0.weight"].shape[1] == 8 + 1 + 4
    reloaded = Trainer(TrainConfig(**saved["config"]), _small_data())
    reloaded.model.load_state_dict(saved["model"])
