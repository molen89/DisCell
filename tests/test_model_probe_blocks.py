"""Planted-answer tests for the per-block invariance probe (review R20 + R22).

The niche block v is [composition (K-1 columns, variance << 1), image PCs
(variance ~10 each)]. The worlds plant which block a latent knows about and
check that each grader grades the block it should, whatever the columns'
scales are.

The block tests draw ``N_PERM = 20`` floor permutations: they test the planted
structure, not the rule's noise. With the default 5 draws the floor sd is
itself so noisy that an independent latent fails a block ~13 % of the time
(simulated, 300 draws; ~23 % for the two ridge blocks together).
"""

from __future__ import annotations

import numpy as np

from discell.model import metrics as M

N_PERM = 20


def _world(seed: int = 0, n: int = 12_000, k: int = 5, n_img: int = 12):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, k, n)
    alpha = rng.uniform(0.5, 2.0, (k, k))
    y = np.stack([rng.dirichlet(alpha[g]) for g in t])
    comp = y[:, :-1]
    img = 3.0 * rng.standard_normal((k, n_img))[t] \
        + 3.0 * rng.standard_normal((n, n_img))
    train = rng.random(n) < 0.7
    return rng, t, comp, img, train, ~train


def _vbar(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack([v[t == g].mean(axis=0) for g in range(t.max() + 1)])


def _composition_only_z(rng, t, comp):
    """z = the within-type composition residual, projected, plus noise dims."""
    resid = comp - _vbar(comp, t)[t]
    signal = resid @ rng.standard_normal((comp.shape[1], 2)) * 5.0
    return np.hstack([signal + 0.1 * rng.standard_normal(signal.shape),
                      rng.standard_normal((len(t), 4))])


def test_legacy_block_is_probe_delta_ce_exactly():
    rng, t, comp, img, train, test = _world(0)
    v = np.hstack([comp, img]).astype(np.float32)
    vbar = _vbar(v, t).astype(np.float32)
    z = rng.standard_normal((len(t), 6)).astype(np.float32)
    legacy = M.probe_delta_ce(z, t, v, vbar, train, test, seed=3)
    out = M.probe_gain_per_block(z, t, v, vbar, train, test,
                                 n_comp=comp.shape[1], seed=3)
    for key in ("delta_ce", "noise_floor", "baseline_ce"):
        assert np.isclose(out["legacy"][key], legacy[key], rtol=0, atol=1e-10)
    assert out["n_comp"] == 4 and out["n_img"] == 12
    assert len(out["comp"]["floor_draws"]) == 5


def test_composition_only_latent_fails_composition_passes_image():
    rng, t, comp, img, train, test = _world(1)
    v = np.hstack([comp, img])
    z = _composition_only_z(rng, t, comp)
    for grade in (M.probe_gain_per_block, M.probe_gain_per_block_mlp):
        out = grade(z, t, v, _vbar(v, t), train, test, n_comp=comp.shape[1],
                    n_perm=N_PERM)
        assert not out["comp"]["pass"], grade.__name__
        assert out["comp"]["excess"] > 0.1, grade.__name__
        assert out["img"]["pass"], (grade.__name__, out["img"])
    # the pooled legacy dCE is the image block's: it cannot see this leak
    ridge = M.probe_gain_per_block(z, t, v, _vbar(v, t), train, test,
                                   n_comp=comp.shape[1])
    assert abs(ridge["legacy"]["delta_ce"] - ridge["legacy"]["noise_floor"]) \
        < 0.01 * ridge["legacy"]["baseline_ce"]


def test_scaling_the_image_block_leaves_the_composition_grade_unchanged():
    rng, t, comp, img, train, test = _world(2)
    z = _composition_only_z(rng, t, comp)
    v = np.hstack([comp, img])
    v_big = np.hstack([comp, 100.0 * img])
    n_comp = comp.shape[1]
    for grade, tol in ((M.probe_gain_per_block, 1e-9),
                       (M.probe_gain_per_block_mlp, 1e-3)):
        a = grade(z, t, v, _vbar(v, t), train, test, n_comp=n_comp,
                  n_perm=N_PERM)
        b = grade(z, t, v_big, _vbar(v_big, t), train, test, n_comp=n_comp,
                  n_perm=N_PERM)
        for block in ("comp", "img"):
            for key in ("gain", "floor_mean", "floor_sd"):
                assert abs(a[block][key] - b[block][key]) < tol, \
                    (grade.__name__, block, key)
    # ... while the legacy pooled dCE moves with the image block's units
    assert b["legacy"]["baseline_ce"] > 1000 * a["legacy"]["baseline_ce"]


def test_nonlinear_dependence_is_caught_by_the_mlp_not_the_ridge():
    """Composition depends on the product of two latent dims (XOR-like):
    no linear read of (z, t) sees it, a small MLP does."""
    rng, t, comp, img, train, test = _world(3)
    z = rng.standard_normal((len(t), 4))
    product = z[:, 0] * z[:, 1]
    comp = comp + 0.05 * product[:, None] * rng.choice([-1.0, 1.0],
                                                       comp.shape[1])
    v = np.hstack([comp, img])
    ridge = M.probe_gain_per_block(z, t, v, _vbar(v, t), train, test,
                                   n_comp=comp.shape[1], n_perm=N_PERM)
    mlp = M.probe_gain_per_block_mlp(z, t, v, _vbar(v, t), train, test,
                                     n_comp=comp.shape[1], n_perm=N_PERM)
    assert ridge["comp"]["pass"], ridge["comp"]
    assert not mlp["comp"]["pass"], mlp["comp"]
    assert mlp["comp"]["excess"] > 10 * mlp["comp"]["floor_sd"]
    assert mlp["img"]["pass"] and ridge["img"]["pass"]


def test_invariance_guard_is_all_four_blocks():
    rng, t, comp, img, train, test = _world(4)
    v = np.hstack([comp, img])
    z = _composition_only_z(rng, t, comp)
    out = M.probe_blocks(z, t, v, _vbar(v, t), train, test,
                         n_comp=comp.shape[1], n_perm=N_PERM)
    assert set(out["pass"]) == {"ridge_comp", "ridge_img", "mlp_comp",
                                "mlp_img"}
    assert out["invariance_pass"] is False
    assert out["pass"]["ridge_img"] and out["pass"]["mlp_img"]


def test_a_column_constant_on_held_out_cells_is_not_graded():
    """A type that is no held-out cell's neighbour leaves a composition column
    that is zero on every held-out cell: it is dropped from the block, not
    graded as a ratio of rounding errors."""
    rng, t, comp, img, train, test = _world(5)
    comp = comp.copy()
    comp[test, 0] = 0.0
    v = np.hstack([comp, img])
    z = rng.standard_normal((len(t), 6))
    out = M.probe_gain_per_block(z, t, v, _vbar(v, t), train, test,
                                 n_comp=comp.shape[1], n_perm=N_PERM)
    assert out["comp"]["constant_cols"] == [0]
    assert out["comp"]["n_cols"] == comp.shape[1] - 1
    assert out["img"]["constant_cols"] == [] and out["img"]["n_cols"] == 12
    assert np.isfinite(out["comp"]["gain"]) and out["comp"]["pass"]
