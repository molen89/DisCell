"""Known-answer tests for the w-program atlas instruments (doc 08 section 6)."""

import numpy as np

from discell.model.atlas import hallmark_labels, varimax


def test_varimax_is_orthogonal_and_preserves_the_decomposition():
    rng = np.random.default_rng(0)
    loadings = rng.normal(size=(200, 4))
    rotation = varimax(loadings)
    assert np.allclose(rotation @ rotation.T, np.eye(4), atol=1e-8)
    u = rng.normal(size=(50, 4))
    # <u, L> per gene is invariant when both sides rotate together
    assert np.allclose(u @ loadings.T, (u @ rotation) @ (loadings @ rotation).T,
                       atol=1e-8)


def test_varimax_recovers_a_planted_sparse_structure():
    rng = np.random.default_rng(1)
    sparse = np.zeros((300, 3))
    sparse[:100, 0], sparse[100:200, 1], sparse[200:, 2] = 2.0, -1.5, 1.0
    mix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    mixed = sparse @ mix                          # rotated away from sparsity
    recovered = mixed @ varimax(mixed)
    # each recovered column concentrates on one planted block again
    share = np.abs(recovered).reshape(3, 100, 3).sum(axis=1)
    share /= share.sum(axis=1, keepdims=True)
    assert (share.max(axis=1) > 0.9).all()


def test_hallmark_labels_rank_the_planted_set_first():
    hallmarks = {"PLANTED": {f"G{i}" for i in range(30)},
                 "OTHER": {f"H{i}" for i in range(30)}}
    signature = [f"G{i}" for i in range(12)] + ["X1", "X2", "X3"]
    rows = hallmark_labels(signature, hallmarks, panel_size=2000)
    assert rows and rows[0]["hallmark"] == "PLANTED"
    assert rows[0]["p"] < 1e-6
