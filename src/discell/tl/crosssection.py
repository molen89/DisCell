"""Apply a fitted model to another section (the held-out-section read)."""

from __future__ import annotations

import numpy as np

from discell.tl import _utils as U


def cross_section(model, adata) -> dict:
    """Read *model* on *adata*, a section it was not fitted on.

    *adata* carries a DisCell graph built with the model's label key, and the
    same type vocabulary and genes (in order) as the model: a mismatch is an
    error, never a remap. Its tiles are assembled under the model's settings;
    the latents and loadings are written into *adata* (as by ``Model.encode``,
    so every other readout then works on it), and the record holds the closing
    evaluation on *adata*'s held-out tiles (``Model.evaluate``) plus the
    reconstruction over every tile (``recon_all_tiles``: none of the section
    was trained on). Stored in ``.uns['discell']['cross_section']``.
    """
    data = model.assemble(adata)
    model.encode(adata, data=data)
    reads = model.evaluate(adata, data=data)
    runner = model.runner(data)
    every = runner.train_batches + runner.val_batches
    reads["recon_all_tiles"] = float(np.mean(runner.sweep(every, recon=True)["recon"]))
    reads["n_cells"] = int(data.n_cells)
    reads["n_tiles"] = len(every)
    U.store(adata, "cross_section", reads)
    return reads
