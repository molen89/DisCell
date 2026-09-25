"""Held-out reconstruction under the three decodes (metrics package item 2):
the full-posterior mode is the trainer's own recon_val, and the cross-check
modes are the trainer's recon_gap numbers."""

from __future__ import annotations

import numpy as np
import pytest

from discell.experiments import recon_modes as R
from discell.model import metrics as M

from tests.test_model_train import small_fit  # noqa: F401  (fixture)


def test_modes_reproduce_the_trainers_own_reads(small_fit):  # noqa: F811
    trainer, _ = small_fit
    report = trainer.evaluate()
    train = trainer._sweep(trainer.train_batches)
    t_train = trainer.data.t[train["nodes"]]
    k = len(trainer.data.p_t)
    z_bar = M.type_means(train["mu_z"], t_train, k)
    profile = R.type_log_profile(trainer.data.x[train["nodes"]], t_train, k)
    cells = R.per_cell_modes(trainer, trainer.val_batches, z_bar, profile)
    gap = report["recon_gap"]
    assert cells["full"].mean() == pytest.approx(report["recon_val"], abs=1e-5)
    assert cells["typemean_z"].mean() == pytest.approx(gap["recon_typemean_z"],
                                                       abs=1e-5)
    assert cells["type_profile"].mean() == pytest.approx(
        gap["recon_type_profile"], abs=1e-5)
    for mode in ("intrinsic", "context"):
        assert np.isfinite(cells[mode]).all()
        assert not np.allclose(cells[mode], cells["full"])
    n_val = sum(b["n_seeds"] for b in trainer.val_batches)
    assert all(len(v) == n_val for v in cells.values())


def test_summary_carries_every_mode_and_difference_with_cis(small_fit):  # noqa: F811
    trainer, _ = small_fit
    train = trainer._sweep(trainer.train_batches)
    t_train = trainer.data.t[train["nodes"]]
    k = len(trainer.data.p_t)
    cells = R.per_cell_modes(
        trainer, trainer.val_batches, M.type_means(train["mu_z"], t_train, k),
        R.type_log_profile(trainer.data.x[train["nodes"]], t_train, k))
    out = R.summarise(cells, np.asarray(trainer.data.positions), 50, 0)
    assert set(out["modes"]) == set(R.MODES) | {"typemean_z"}
    for entry in out["modes"].values():
        assert entry["ci95"][0] <= entry["recon"] <= entry["ci95"][1]
    d = out["differences"]["full-context"]
    assert d["value"] == pytest.approx(out["modes"]["full"]["recon"]
                                       - out["modes"]["context"]["recon"])
