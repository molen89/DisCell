"""The planted spill-over control (devlog 2026-10-01, RECOMB additions item 1)."""

import numpy as np
import pytest

from discell.experiments import planted_spillover as PS
from discell.model import transport as T
from discell.model.synthetic import simulate


def test_b_zero_none_and_all_false_are_the_old_world():
    a = simulate(n_cells=400, n_types=3, seed=3)
    b = simulate(n_cells=400, n_types=3, seed=3,
                 b_zero=np.zeros(60, dtype=bool))
    for field in ("x", "t", "z_true", "w_true", "B_true", "p_true", "phi"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_b_zero_removes_the_response_and_moves_no_draw():
    mask = np.zeros(60, dtype=bool)
    mask[:20] = True
    a = simulate(n_cells=400, n_types=3, seed=3)
    b = simulate(n_cells=400, n_types=3, seed=3, b_zero=mask)
    assert np.all(b.B_true[:, mask] == 0)
    assert np.array_equal(a.B_true[:, ~mask], b.B_true[:, ~mask])
    for field in ("t", "z_true", "w_true", "positions", "totals"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_spill_genes_fixed_and_sized():
    m = PS.spill_genes(0)
    assert m.sum() == PS.N_SPILL and np.array_equal(m, PS.spill_genes(0))


def test_r2_rows_is_score_shift():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(3, 30))
    pred = rng.normal(size=30)
    got = PS.r2_rows(obs, pred)
    for i in range(3):
        assert got[i] == pytest.approx(T.score_shift(pred, obs[i])["r2"])


def test_predicted_kappa_star():
    assert PS.predicted_kappa_star(0.1) == 0.1
    assert PS.predicted_kappa_star(0.2) == 0.2
    assert PS.predicted_kappa_star(0.15) == 0.2
    assert PS.predicted_kappa_star(0.0) == 0.0


def _entry(est, lo, hi, seeds):
    return {"estimate": est, "ci": [lo, hi], "per_seed": seeds}


def test_verdict_reads_breakdown_rule():
    from discell.experiments.breakdown import breakdown_point, GRID

    hold = _entry(0.2, 0.1, 0.3, [0.2, 0.2, 0.2])
    gone = _entry(0.01, -0.02, 0.04, [0.01, 0.0, 0.02])
    spill = {k: (hold if k < 0.1 else gone) for k in GRID}
    resp = {k: hold for k in GRID}
    table = {"members": {"spill_cf_minus_leak": breakdown_point(spill),
                         "response_cf_minus_leak": breakdown_point(resp)}}
    v = PS.verdict(table, 0.1)
    assert v["spill_tracks_kappa_true"] and v["response_survives_grid"]
    v = PS.verdict(table, 0.2)          # broke one grid point early
    assert not v["spill_tracks_kappa_true"]


class _Graph:
    def __init__(self, n):
        self.n_cells = n


class _Data:
    def __init__(self, x, totals, positions):
        self.x, self.totals, self.positions = x, totals, positions
        self.graph = _Graph(len(x))


def test_contrast_draws_exact_prediction():
    """A prediction equal to the observed shift has R^2 1; leak-only 0."""
    rng = np.random.default_rng(1)
    n, g = 400, 12
    x = rng.poisson(20, size=(n, g)).astype(float)
    totals = x.sum(1)
    positions = rng.uniform(0, 2000, size=(n, 2))
    data = _Data(x, totals, positions)
    rows_a, rows_b = np.arange(0, 200), np.arange(200, 400)
    rate = x / totals[:, None]
    obs = np.log(rate[rows_b].mean(0) + T.EPS) - np.log(rate[rows_a].mean(0)
                                                       + T.EPS)
    panels = [{"rows_a": rows_a, "rows_b": rows_b, "program": obs,
               "leak": np.zeros(g)}]
    sets = {"a": np.arange(g) < 6, "b": np.arange(g) >= 6}
    out, c = PS.contrast_draws(data, panels, sets, 20, 0)
    for m in sets:
        assert out[m][0] == pytest.approx(1.0)
        assert out[m].shape == (21,)
    assert c > 0
