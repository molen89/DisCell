"""The cell-cycle R^2 of a latent: how much of the cell cycle it keeps within types.

The paper's measure of what the intrinsic state keeps (supplement S3.1): the
held-out R^2 of a ridge regression of the S and G2/M scores (Tirosh scoring),
centered per type, on the latent, over the tenth of held-out cells that cycle
most. The ridge is fitted on the training tiles' cells of the cycling set.
Spillover enters the scores, so the share shows where the model places the
cycle. The same computation, at unit weights, is the point estimate of the
``cycle_check`` (:mod:`discell.sensitivity.findings`).
"""

from __future__ import annotations

import numpy as np

from discell.evaluate.mask import metric_target_mask


def cell_cycle_r2(model, adata, *, data=None) -> dict | None:
    """``{"z": R^2, "w": R^2, "n_held_out": cells}`` of one fit on *adata*'s held-out
    tiles (Unassigned cells excluded); None when the panel lacks the cell-cycle
    genes (fewer than 10 S or G2/M markers). *data*: an assembly of *adata* made
    with :meth:`discell.Model.assemble`."""
    from discell.sensitivity.findings import _cycle_cells, _cycle_r2, cell_cycle_scores, cycling_set

    data = data if data is not None else model.assemble(adata)
    scores = cell_cycle_scores(data.x, data.gene_names)
    if scores is None:
        return None
    names = [str(v) for v in data.type_names]
    seed = model.config.seed
    lat = model.latents(data)                     # train tiles, then held-out tiles
    rows = lat["nodes"]
    n_train = sum(len(tile) for tile in data.train_tiles)
    train = np.arange(len(rows)) < n_train
    held = np.zeros(data.n_cells, dtype=bool)
    for tile in data.val_tiles:
        held[tile] = True
    q90 = cycling_set(scores, held, metric_target_mask(data.t, names,
                                                         (model.config.unassigned_label,)))
    q90 = q90[rows]
    out: dict = {}
    for key in ("mu_z", "mu_w"):
        cells = _cycle_cells(lat[key], data.t[rows], scores[rows], train, ~train, seed, q90)
        n = len(cells["rows_test"])
        out[key[3]] = _cycle_r2(cells, np.ones(n)) if n else float("nan")
        out["n_held_out"] = int(n)
    return out
