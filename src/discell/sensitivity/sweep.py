"""The kappa sweep: one fit per grid kappa and seed."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

from discell.model.config import Config
from discell.model.model import Model, fit
from discell.sensitivity.breakdown import KAPPA_GRID

#: The paper's seeds.
SEEDS = (0, 1, 2)


def run_dir(out_dir: str | Path, kappa: float, seed: int) -> Path:
    """Where :func:`fit_sweep` saves the fit at (*kappa*, *seed*)."""
    return Path(out_dir) / f"k{kappa:g}_s{seed}"


def fit_sweep(adata, config: Config, kappas: Sequence[float] = KAPPA_GRID,
              seeds: Sequence[int] = SEEDS, out_dir: str | Path | None = None
              ) -> dict[tuple[float, int], Model]:
    """Fit *config* at every kappa and seed; returns ``{(kappa, seed): Model}``.

    Everything but ``kappa`` and ``seed`` comes from *config*. With *out_dir*
    each fit is saved to ``<out_dir>/k<kappa>_s<seed>`` and a fit already saved
    there is loaded instead of refitted, so an interrupted sweep resumes. The
    ``.obsm`` outputs left in *adata* are those of the last fit;
    :meth:`Model.encode` writes any other fit's.
    """
    models: dict[tuple[float, int], Model] = {}
    for kappa in kappas:
        for seed in seeds:
            cfg = dataclasses.replace(config, kappa=float(kappa), seed=int(seed))
            where = None if out_dir is None else run_dir(out_dir, kappa, seed)
            if where is not None and (where / "model.pt").exists():
                models[(float(kappa), int(seed))] = Model.load(where, device=cfg.device)
                continue
            models[(float(kappa), int(seed))] = fit(adata, cfg, out_dir=where)
    return models
