"""The held-out-section protocol: a fit applied to another dataset's tiles."""

from __future__ import annotations

import numpy as np

from tests.test_model_train import small_fit  # noqa: F401  (fixture)


def test_apply_fit_on_own_data_matches_own_evaluate(small_fit):
    """Applied to the data it was trained on, the reads equal the trainer's
    own evaluate() (same seed, same tiles), and the all-tiles reconstruction
    is a finite per-count log-likelihood of the same order as the val one."""
    from discell.model.crossslide import apply_fit

    trainer, _ = small_fit
    own = trainer.evaluate()
    reads = apply_fit(trainer.config, trainer.model, trainer.data)
    assert reads["recon_val"] == own["recon_val"]
    assert reads["nmi"] == own["nmi"]
    assert reads["probe"]["delta_ce"] == own["probe"]["delta_ce"]
    assert np.isfinite(reads["recon_all_tiles"])
    assert abs(reads["recon_all_tiles"] - reads["recon_val"]) < 1.0
    assert reads["n_cells"] == trainer.data.n_cells
    assert reads["n_tiles"] == len(trainer.train_batches) + len(trainer.val_batches)
