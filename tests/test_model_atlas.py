"""Known-answer tests for the w-program atlas instruments (doc 08 section 6).

The planted worlds are the pre-registration of the atlas rewrite (devlog,
2026-09-21): a rank-2 w in 6 dimensions must read as two programs, a
planted per-type offset must be invisible (issues V12), planted sparse
loadings must be recovered, a planted enriched set must be labelled and a
random one not, and an empty landmark inventory must not crash the build.
"""

import numpy as np
import pytest

from discell.model.atlas import (cross_seed, hallmark_labels,
                                 label_recurrence, significant_labels,
                                 varimax)


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


def _planted_rank_two(rng, n=6000, d_w=6, n_genes=400, n_types=4):
    """w on a 2-D subspace of d_w dims (variances 4 and 1) plus tiny noise."""
    coords = rng.normal(size=(n, 2)) * np.array([2.0, 1.0])
    basis, _ = np.linalg.qr(rng.normal(size=(d_w, 2)))
    mu_w = coords @ basis.T + 0.01 * rng.normal(size=(n, d_w))
    t = rng.integers(n_types, size=n)
    b_matrix = rng.normal(size=(n_genes, d_w))
    return mu_w, t, b_matrix, np.ones(n_genes, dtype=bool)


def test_planted_rank_two_in_six_dims_gives_two_programs():
    """V10: the rotated-coordinate-variance read called this 6/6 active."""
    from discell.model.atlas import centred_program_basis

    rng = np.random.default_rng(2)
    mu_w, t, b_matrix, expressed = _planted_rank_two(rng)
    u, loadings, info = centred_program_basis(mu_w, t, b_matrix, expressed)
    assert info["rank"] == 2
    assert u.shape == (len(mu_w), 2) and loadings.shape == (len(b_matrix), 2)
    assert abs(info["variance_fraction"][0] - 0.8) < 0.03
    assert abs(info["variance_fraction"][1] - 0.2) < 0.03
    assert abs(sum(info["variance_share"]) - 1.0) < 0.02
    assert min(info["variance_share"]) > 0.05
    # the decoder's realised shift is unchanged by the basis choice
    shift = (mu_w - mu_w.mean(axis=0)) @ b_matrix.T
    reconstructed = u @ loadings.T
    assert 1 - ((shift - reconstructed) ** 2).mean() / shift.var() > 0.99


def test_planted_per_type_offset_does_not_change_activity_or_loadings():
    """V12: the gauge offset is read as extra programs on raw w."""
    from discell.model.atlas import centred_program_basis
    from discell.model.lr_map import program_basis

    rng = np.random.default_rng(3)
    mu_w, t, b_matrix, expressed = _planted_rank_two(rng)
    offset = 10.0 * rng.normal(size=(t.max() + 1, mu_w.shape[1]))
    with_offset = mu_w + offset[t]
    # the offset would be read as extra programs on raw w ...
    assert program_basis(with_offset, b_matrix, expressed)[2]["rank"] > 2
    # ... and is invisible to the within-type-centred read
    u0, load0, info0 = centred_program_basis(mu_w, t, b_matrix, expressed)
    u1, load1, info1 = centred_program_basis(with_offset, t, b_matrix, expressed)
    assert info1["rank"] == info0["rank"] == 2
    assert np.allclose(info1["variance_share"], info0["variance_share"])
    assert np.allclose(np.abs(load1), np.abs(load0), atol=1e-6)
    assert np.allclose(np.abs(u1), np.abs(u0), atol=1e-6)


def test_planted_offset_does_not_change_the_hallmark_labels():
    from discell.model.atlas import centred_program_basis

    rng = np.random.default_rng(13)
    mu_w, t, b_matrix, expressed = _planted_rank_two(rng)
    genes = np.array([f"G{i}" for i in range(len(b_matrix))])
    b_matrix[:40, 0] += 6.0                      # a set loading on program 0
    hallmarks = {"PLANTED": set(genes[:40]),
                 "OTHER": set(genes[200:240])}
    offset = 10.0 * rng.normal(size=(t.max() + 1, mu_w.shape[1]))

    def labels(w):
        _, loadings, info = centred_program_basis(w, t, b_matrix, expressed)
        return [[h["hallmark"] for h in
                 hallmark_labels(loadings[:, k], genes, expressed, hallmarks)
                 if h["significant"]] for k in range(info["rank"])]

    plain, shifted = labels(mu_w), labels(mu_w + offset[t])
    assert {n for row in plain for n in row} == {"PLANTED"}
    assert plain == shifted


def test_cross_seed_reads_shared_span_as_one_and_orthogonal_as_zero():
    """V11: the invariant across seeds is the shift span, not B's columns."""
    rng = np.random.default_rng(4)
    full, _ = np.linalg.qr(rng.normal(size=(300, 4)))
    loadings = full[:, :2] * np.array([3.0, 1.0])
    same = loadings[:, ::-1] * np.array([1.0, -1.0])     # permuted, flipped
    out = cross_seed(loadings, same)
    assert [m["program"] for m in out["axis_cosine"]] == [1, 0]
    assert np.allclose([m["cosine"] for m in out["axis_cosine"]], 1.0)
    assert np.allclose(list(out["shift_overlap"].values()), 1.0)
    out = cross_seed(loadings, full[:, 2:])
    assert max(m["cosine"] for m in out["axis_cosine"]) < 1e-8
    assert np.allclose(list(out["shift_overlap"].values()), 0.0, atol=1e-8)


def test_hallmark_labels_find_the_planted_set_and_not_a_random_one():
    rng = np.random.default_rng(5)
    genes = np.array([f"G{i}" for i in range(2000)])
    expressed = np.ones(len(genes), dtype=bool)
    loading = rng.normal(size=len(genes)) * 0.1
    planted = set(genes[:30])
    loading[:30] += 3.0                        # the whole set loads, no cut
    hallmarks = {"PLANTED": planted,
                 **{f"RANDOM{j}": set(rng.choice(genes[30:], 30, replace=False))
                    for j in range(20)}}
    rows = hallmark_labels(loading, genes, expressed, hallmarks)
    assert rows[0]["hallmark"] == "PLANTED" and rows[0]["significant"]
    assert rows[0]["direction"] > 0
    assert not any(r["significant"] for r in rows[1:])


def test_hallmark_labels_are_blind_to_the_programs_arbitrary_sign():
    rng = np.random.default_rng(6)
    genes = np.array([f"G{i}" for i in range(1000)])
    expressed = np.ones(len(genes), dtype=bool)
    loading = rng.normal(size=len(genes)) * 0.1
    loading[:30] -= 3.0
    hallmarks = {"PLANTED": set(genes[:30]),
                 "OTHER": set(genes[500:530])}
    rows = hallmark_labels(loading, genes, expressed, hallmarks)
    assert rows[0]["hallmark"] == "PLANTED" and rows[0]["significant"]
    assert rows[0]["direction"] < 0          # orients the label, does not gate
    flipped = hallmark_labels(-loading, genes, expressed, hallmarks)
    assert flipped[0]["hallmark"] == "PLANTED"
    assert flipped[0]["q"] == pytest.approx(rows[0]["q"])


def test_untestable_sets_are_dropped_not_scored():
    genes = np.array([f"G{i}" for i in range(100)])
    expressed = np.ones(len(genes), dtype=bool)
    loading = np.zeros(len(genes)); loading[:4] = 5.0
    rows = hallmark_labels(loading, genes, expressed, {"TINY": set(genes[:4])})
    assert rows == []


def test_label_recurrence_counts_runs_not_programs():
    labels = {"s0": [["EMT"], ["EMT", "HYPOXIA"]],
              "s1": [["EMT"], []],
              "s2": [["COAGULATION"], []]}
    out = label_recurrence(labels)
    assert out["n_runs"] == 3
    assert out["counts"]["EMT"] == 2            # two runs, not three programs
    assert out["recurring"] == ["EMT"]
    assert out["label_sets"]["s0"] == ["EMT", "HYPOXIA"]


def test_significant_labels_reads_only_gated_hallmarks():
    atlas = {"programs": [{"hallmarks": [{"hallmark": "EMT", "significant": True},
                                         {"hallmark": "X", "significant": False}]},
                          {"hallmarks": []}]}
    assert significant_labels(atlas) == [["EMT"], []]


def test_an_empty_landmark_inventory_drops_the_block_instead_of_crashing():
    """Cluster-labelled slides have no landmark classes; the build must run."""
    from discell.model.atlas import landmark_distance_block

    positions = np.random.default_rng(7).normal(size=(200, 2)) * 100
    # what landmark_inventory returns on a graphclust slide: the interface
    # class is present but has no boundary cells, the rest never form
    empty = {"tumor_stroma_interface": {"constituents": np.array([], dtype=int)}}
    assert landmark_distance_block(positions, empty) is None
    assert landmark_distance_block(positions, {}) is None
    block, names = landmark_distance_block(
        positions, {**empty, "vasculature": {"constituents": np.arange(20)}})
    assert names == ["vasculature"] and block.shape == (200, 1)
    assert np.isfinite(block).all() and block.min() >= 0.0
