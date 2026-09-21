"""Planted-answer tests for the spec 7.10 degeneracy diagnostics."""

from __future__ import annotations

import numpy as np

from discell.model import metrics as M


def _world(seed: int, n: int = 6000, k: int = 5):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, k, n)
    train = rng.random(n) < 0.7
    return rng, t, train, ~train


def test_z_equal_to_onehot_t_reads_degenerate():
    rng, t, train, test = _world(0)
    z = np.eye(t.max() + 1)[t] + 1e-3 * rng.standard_normal((len(t), t.max() + 1))
    out = M.type_degeneracy(z, t, train, test)
    assert out["mi_ratio"] > 0.97
    assert out["accuracy"] > 0.99
    assert out["within_var_fraction"] < 0.01


def test_z_independent_of_t_reads_blind():
    rng, t, train, test = _world(1)
    z = rng.standard_normal((len(t), 8))
    out = M.type_degeneracy(z, t, train, test)
    assert abs(out["mi_ratio"]) < 0.03
    assert out["within_var_fraction"] > 0.97
    assert len(out["within_var_fraction_per_dim"]) == 8


def test_variance_fraction_follows_the_planted_split():
    """Type means separated in one dim, pure noise in another: the trace
    fraction is the noise share, and the per-dim split is 0 / 1."""
    rng, t, train, test = _world(2)
    z = np.stack([10.0 * t, rng.standard_normal(len(t))], axis=1)
    out = M.type_degeneracy(z, t, train, test)
    per_dim = out["within_var_fraction_per_dim"]
    assert per_dim[0] < 0.01 and abs(per_dim[1] - 1.0) < 0.05
    expected = 1.0 / (1.0 + 100.0 * t.var())
    assert abs(out["within_var_fraction"] - expected) < 0.01


def test_type_means_fill_absent_types_with_the_global_mean():
    z = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]], dtype=np.float32)
    t = np.array([0, 0, 2])
    means = M.type_means(z, t, n_types=4)
    assert means.dtype == np.float32 and means.shape == (4, 2)
    np.testing.assert_allclose(means[0], [1.0, 1.0])
    np.testing.assert_allclose(means[2], [4.0, 4.0])
    np.testing.assert_allclose(means[1], z.mean(axis=0))
    np.testing.assert_allclose(means[3], z.mean(axis=0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _type_mean_decode(rates: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Stand-in for the trainer's substitution: every cell decoded at its
    type's mean latent. Averaging in the latent (here: log-rate) space is what
    ``Trainer._decode_seeds`` does with ``z_bar[t]``, so a latent that is
    constant within type leaves the decode untouched."""
    out = np.empty_like(rates)
    for g in np.unique(t):
        out[t == g] = rates[t == g].mean(axis=0)
    return out


def test_type_mean_decode_gap_is_zero_without_and_positive_with_per_cell_state():
    rng = np.random.default_rng(3)
    n, g, k = 3000, 40, 4
    t = rng.integers(0, k, n)
    type_logits = rng.standard_normal((k, g))[t]
    cell_logits = type_logits + rng.standard_normal((n, g))

    # world without per-cell state: the latent is constant within type, so the
    # substitution returns the same rates and the gap is exactly zero
    p_type = _softmax(type_logits)
    x = np.stack([rng.multinomial(200, p) for p in p_type]).astype(np.float64)
    substituted = _softmax(_type_mean_decode(type_logits, t))
    np.testing.assert_allclose(substituted, p_type)
    assert M.held_out_reconstruction(x, np.log(p_type)) \
        - M.held_out_reconstruction(x, np.log(substituted)) == 0.0

    # world with per-cell state: decoding at the cell's own rates beats the
    # type-level rates by a clear margin in nats per count
    p_cell = _softmax(cell_logits)
    x = np.stack([rng.multinomial(200, p) for p in p_cell]).astype(np.float64)
    gap = (M.held_out_reconstruction(x, np.log(p_cell))
           - M.held_out_reconstruction(x, np.log(
               _softmax(_type_mean_decode(cell_logits, t)))))
    assert gap > 0.1


def test_type_profile_reference_recovers_the_type_rates():
    """In a world where z IS t, the empirical type profile is the truth (up to
    sampling), so its held-out score matches the true type-rate score."""
    import scipy.sparse as sp

    rng = np.random.default_rng(4)
    n, g, k = 4000, 40, 4
    t = rng.integers(0, k, n)
    p_type = _softmax(rng.standard_normal((k, g)))
    x = np.stack([rng.multinomial(200, p_type[i]) for i in t]).astype(np.float64)
    train = np.arange(n) < 3000
    ref = M.type_profile_reconstruction(sp.csr_matrix(x[train]), t[train],
                                        x[~train], t[~train])
    truth = M.held_out_reconstruction(x[~train], np.log(p_type[t[~train]]))
    assert abs(ref - truth) < 0.01
    # a dense training matrix gives the same answer
    assert abs(M.type_profile_reconstruction(x[train], t[train], x[~train],
                                             t[~train]) - ref) < 1e-9


def test_type_degeneracy_survives_a_type_absent_from_training():
    """A type present only in held-out rows (GSE core: Plasma cells, n = 1)
    must not overflow the probe's class list (queue failure 2026-09-17)."""
    import numpy as np
    from discell.model.metrics import type_degeneracy
    rng = np.random.default_rng(0)
    n, k = 600, 5
    t = rng.integers(0, k - 1, n)          # types 0..3 in training
    t[-1] = k - 1                          # type 4 appears once, held out
    z = np.eye(k)[t] + 0.05 * rng.standard_normal((n, k))
    train = np.ones(n, bool); train[-100:] = False
    out = type_degeneracy(z, t, train, ~train, seed=0)
    assert np.isfinite(out["mi_ratio"]) and 0 <= out["accuracy"] <= 1
