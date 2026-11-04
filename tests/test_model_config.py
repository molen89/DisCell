"""The fit configuration: the paper's final configuration as defaults, validated."""

from __future__ import annotations

import dataclasses

import pytest

from discell import Config

#: the paper's final values; alpha_z and tile_cells are per section
FINAL = dict(kappa=0.1, d_z=20, d_w=6, hidden=256, gat_dim=32, heads=4, v_pcs=12, omega=1.0,
             alpha_w=0.1, w_warmup_epochs=30, alpha_a=0.3, adv_lr=2e-3, adv_steps=6,
             adv_hidden=64, adv_comp_weight=3.0, lr=1e-3, grad_clip=10.0, val_fraction=0.15,
             nmi_guard=0.9, eval_every=5, epochs=500, patience=40, tile_cells=4096, seed=0)


def test_defaults_are_the_final_configuration():
    cfg = Config(label_key="lineage")
    assert {k: getattr(cfg, k) for k in FINAL} == FINAL
    assert cfg.alpha_z == "auto" and cfg.image_key is None and cfg.image_free
    assert cfg.unassigned_label == "Unassigned" and cfg.device == "cuda"


def test_label_key_is_required():
    with pytest.raises(TypeError):
        Config()
    with pytest.raises(ValueError):
        Config(label_key="")


@pytest.mark.parametrize("field, value", [
    ("kappa", 1.0), ("kappa", -0.1), ("alpha_z", "median"), ("alpha_z", 0.0),
    ("alpha_a", -1.0), ("tile_cells", 0), ("epochs", 2.5), ("patience", True),
    ("val_fraction", 1.0), ("nmi_guard", 1.5), ("w_warmup_epochs", -1), ("lr", 0.0)])
def test_invalid_values_are_refused(field, value):
    with pytest.raises(ValueError):
        Config(label_key="lineage", **{field: value})


def test_alpha_z_auto_resolves_to_half_over_the_median_count():
    cfg = Config(label_key="lineage")
    assert cfg.resolve(279.0).alpha_z == pytest.approx(0.5 / 279.0)
    explicit = Config(label_key="lineage", alpha_z=0.0018)
    assert explicit.resolve(279.0) is explicit


def test_alpha_w_warm_up_is_linear_then_constant():
    cfg = Config(label_key="lineage")
    assert cfg.alpha_w_at(0) == 0.0
    assert cfg.alpha_w_at(15) == pytest.approx(0.05)
    assert cfg.alpha_w_at(30) == cfg.alpha_w_at(90) == 0.1
    assert Config(label_key="l", w_warmup_epochs=0).alpha_w_at(0) == 0.1


def test_round_trip_and_frozen():
    cfg = Config(label_key="lineage", kappa=0.2, image_key="X_image", alpha_z=0.002)
    assert Config.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.kappa = 0.3
