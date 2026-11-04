"""Spatial tile resampling: the draws behind every breakdown-point interval.

Cells are grouped into square tiles (200 um by default) and whole tiles are
resampled, so the interval respects spatial correlation. A finding is a
statistic of per-cell weights; :func:`percentile_draws` evaluates it at unit
weights (the point estimate) and under tile draws with replacement, and
:func:`half_tile_weights` gives the half-tile subsamples without replacement
used for statistics that a with-replacement draw would bias (duplicated cells
correlate split halves).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

#: Side of the square tiles that are resampled (um).
TILE_UM = 200.0
#: Seed base of the paper's draws.
DRAW_SEED = 8190
#: Draws per fit in the paper (4,000 for the GPU-heavy findings).
N_DRAWS = 10_000


def tile_index(cells_xy: np.ndarray, tile_um: float = TILE_UM) -> tuple[np.ndarray, int]:
    """Contiguous tile id per cell (``tile_um`` squares) and the tile count."""
    xy = np.floor(np.asarray(cells_xy, dtype=np.float64)[:, :2] / tile_um).astype(np.int64)
    xy -= xy.min(axis=0)
    _, ids = np.unique(xy[:, 0] * (int(xy[:, 1].max()) + 1) + xy[:, 1], return_inverse=True)
    return ids.astype(np.int64), int(ids.max()) + 1


def tile_draws(tiles: np.ndarray, n_tiles: int, n: int, seed: int) -> Iterator[np.ndarray]:
    """Yield *n* per-cell weight vectors: tiles drawn with replacement, each
    cell weighted by its tile's multiplicity."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        mult = np.bincount(rng.integers(0, n_tiles, n_tiles),
                           minlength=n_tiles).astype(np.float64)
        yield mult[tiles]


def percentile_draws(positions: np.ndarray, statistic: Callable[[np.ndarray], float], n: int,
                     seed: int, tile_um: float = TILE_UM) -> tuple[float, np.ndarray]:
    """``statistic(weights)`` at unit weights and under *n* tile draws with replacement."""
    tiles, n_tiles = tile_index(positions, tile_um)
    point = float(statistic(np.ones(len(tiles))))
    draws = np.array([statistic(w) for w in tile_draws(tiles, n_tiles, n, seed)])
    return point, draws


def half_tile_weights(positions: np.ndarray, n: int, seed: int, tile_um: float = TILE_UM
                      ) -> tuple[np.ndarray, float]:
    """``(n, cells)`` 0/1 weights of half-tile subsamples without replacement,
    and the finite-population factor ``c = sqrt(m / (n_tiles - m))``."""
    tiles, n_tiles = tile_index(positions, tile_um)
    rng = np.random.default_rng(seed)
    m = n_tiles // 2
    out = np.zeros((n, len(tiles)), dtype=np.float32)
    for b in range(n):
        chosen = np.zeros(n_tiles, dtype=np.float32)
        chosen[rng.choice(n_tiles, m, replace=False)] = 1.0
        out[b] = chosen[tiles]
    return out, float(np.sqrt(m / (n_tiles - m)))


def finding_draws(positions: np.ndarray, statistic: Callable[[np.ndarray], float], *,
                   n: int = N_DRAWS, seed: int = DRAW_SEED, method: str = "percentile",
                   tile_um: float = TILE_UM) -> dict:
    """One fit's draws of a finding, in the form :func:`discell.sensitivity.breakdown` reads.

    *statistic* maps per-cell weights (``positions``' rows) to a number; its
    value at unit weights is the estimate. ``method="percentile"`` resamples
    tiles with replacement; ``"subsample"`` draws half-tile subsamples.
    Returns ``{"estimate", "draws", "method", "c"}``.
    """
    if method == "percentile":
        point, draws = percentile_draws(positions, statistic, n, seed, tile_um)
        return {"estimate": point, "draws": draws, "method": method, "c": 1.0}
    if method == "subsample":
        weights, c = half_tile_weights(positions, n, seed, tile_um)
        point = float(statistic(np.ones(len(positions))))
        draws = np.array([statistic(w.astype(np.float64)) for w in weights])
        return {"estimate": point, "draws": draws, "method": method, "c": c}
    raise ValueError(f"method must be 'percentile' or 'subsample', not {method!r}")
