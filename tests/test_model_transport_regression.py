"""The plain-regression reference for relocation (devlog 2026-10-01, RECOMB
additions item 2)."""

import numpy as np
import pytest
import scipy.sparse as sp

from discell.model import transport as T
from discell.model import transport_regression as R


def _linear(n=3000, p=4, g=7, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.dirichlet(np.ones(p), size=n)
    coef = rng.normal(size=(p, g))
    x = 2.0 + y @ coef + 0.01 * rng.normal(size=(n, g))
    return y, x, coef


def test_ridge_sparse_equals_dense():
    y, x, _ = _linear()
    x = np.clip(x, 0, None)
    x[x < 1.5] = 0.0
    d = R.ridge_gcv(y, x)
    s = R.ridge_gcv(y, sp.csr_matrix(x))
    assert s["lambda"] == pytest.approx(d["lambda"])
    assert np.allclose(s["coef"], d["coef"], atol=1e-8)
    assert np.allclose(s["intercept"], d["intercept"], atol=1e-8)


def test_ridge_matches_closed_form():
    y, x, _ = _linear()
    fit = R.ridge_gcv(y, x)
    yc, xc = y - y.mean(0), x - x.mean(0)
    want = np.linalg.solve(yc.T @ yc + fit["lambda"] * np.eye(y.shape[1]),
                           yc.T @ xc)
    assert np.allclose(fit["coef"], want, atol=1e-8)
    pred = fit["intercept"] + y @ fit["coef"]
    assert np.abs(pred - x).max() < 0.1


def test_rate_shift_is_log_ratio_of_predicted_means():
    fit = {"coef": np.array([[1.0, 0.0], [0.0, 2.0]]),
           "intercept": np.array([1.0, 1.0])}
    ya, yb = np.array([0.5, 0.5]), np.array([1.0, 0.0])
    got = R.predicted_shift(fit, "rate", ya, yb)
    assert got == pytest.approx(np.log([2.0, 1.0]) - np.log([1.5, 2.0]))
    assert R.predicted_shift(fit, "log", ya, yb) == pytest.approx(
        (yb - ya) @ fit["coef"])


def test_move_zero_shift_is_identity_and_normalised():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(9), size=5)
    for form in R.FORMS:
        assert np.allclose(R.move(p, np.zeros(9), form, 100.0), p)
        out = R.move(p, rng.normal(size=9), form, 100.0)
        assert np.allclose(out.sum(1), 1.0) and (out >= 0).all()


def test_read_a_target_itself_closes_the_gap():
    rng = np.random.default_rng(0)
    depths = np.full(300, 200)
    src = rng.multinomial(200, np.full(8, 1 / 8), size=300) / 200.0
    pv = np.array([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    tgt = rng.multinomial(200, pv, size=300) / 200.0
    good = rng.multinomial(200, pv, size=300) / 200.0
    out = R.read_a_scores({"untransported": src, "good": good}, tgt, depths,
                          np.random.default_rng(1))
    assert out["gap_closed"]["good"] > 0.8
    assert out["gap_closed"]["type_mean"] > 0.8


def test_of_ceiling_is_ratio_of_means():
    panels = [{"x": {"r2": 0.2}, "noise_ceiling": 0.5},
              {"x": {"r2": 0.4}, "noise_ceiling": 0.7}]
    got = R.of_ceiling(panels, "x")
    assert got["of_ceiling"] == pytest.approx(0.3 / 0.6)
    low = [{"x": {"r2": 0.01}, "noise_ceiling": 0.01}]
    assert np.isnan(R.of_ceiling(low, "x")["of_ceiling"])


def test_default_read_unchanged_and_regression_opt_in(monkeypatch):
    calls = []
    monkeypatch.setattr(T, "transport_check", lambda a: calls.append("mean"))
    monkeypatch.setattr(T, "distribution_check",
                        lambda a, twins=False: calls.append("dist"))
    monkeypatch.setattr(R, "regression_reference",
                        lambda a: calls.append("regression"))
    T.main(["--dataset", "d", "--run", "r", "--quiet"])
    assert calls == ["mean"]
    calls.clear()
    T.main(["--dataset", "d", "--run", "r", "--read", "regression", "--quiet"])
    assert calls == ["regression"]
    with pytest.raises(SystemExit):
        T.main(["--dataset", "d", "--run", "r", "--read", "regression",
                "--scored-cells", "heldout-tiles", "--quiet"])
