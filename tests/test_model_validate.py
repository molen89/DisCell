"""Known-answer tests for the doc-08 validation analyses.

Each analysis is checked on synthetic inputs where the allegiance is planted:
a latent built to encode the target must score high, a noise latent must sit
at the references. No trained model involved -- these certify the measuring
instruments, not the model.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from discell.model.validate import (
    analysis_landmarks, analysis_morans, analysis_niche, center_per_type,
    landmark_inventory, logistic_cv, morans_i, r2, ridge_cv)


def grid_graph(side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4-neighbour grid: positions plus undirected edge lists."""
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    positions = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    idx = np.arange(side * side).reshape(side, side)
    edge_i = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    edge_j = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    return positions, edge_i, edge_j


def row_normalised(edge_i, edge_j, n) -> sp.csr_matrix:
    adj = sp.coo_matrix(
        (np.ones(2 * len(edge_i)),
         (np.concatenate([edge_i, edge_j]), np.concatenate([edge_j, edge_i]))),
        shape=(n, n)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return sp.diags(1.0 / deg) @ adj


def test_morans_separates_smooth_field_from_noise():
    rng = np.random.default_rng(0)
    positions, edge_i, edge_j = grid_graph(40)
    weights = row_normalised(edge_i, edge_j, len(positions))
    smooth = np.sin(positions[:, 0] / 6.0) + np.cos(positions[:, 1] / 9.0)
    noise = rng.normal(size=len(positions))
    values = np.stack([smooth - smooth.mean(), noise - noise.mean()], axis=1)
    result = morans_i(values, weights, n_perms=200, seed=0)
    assert result["I"][0] > result["null_hi"][0] + 0.2   # decisively hot
    assert result["null_lo"][1] <= result["I"][1] <= result["null_hi"][1]


def test_center_per_type_zeroes_type_means():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 3, 500)
    values = rng.normal(size=(500, 4)) + t[:, None] * 5.0
    centred = center_per_type(values, t, np.ones(500, dtype=bool))
    for g in range(3):
        assert np.allclose(centred[t == g].mean(axis=0), 0.0, atol=1e-9)


def test_ridge_cv_recovers_planted_signal_and_floor_stays_cold():
    rng = np.random.default_rng(2)
    design = rng.normal(size=(2000, 5))
    target = design @ rng.normal(size=5) + 0.1 * rng.normal(size=2000)
    fold = rng.integers(0, 5, 2000)
    held = ridge_cv(design, target, fold)
    assert not np.isnan(held).any()
    assert r2(target, held) > 0.9
    shuffled = target[rng.permutation(2000)]
    assert abs(r2(shuffled, ridge_cv(design, shuffled, fold))) < 0.05


def test_logistic_cv_probabilities_are_aligned_and_normalised():
    rng = np.random.default_rng(3)
    labels = np.repeat([2, 5, 9], 300)          # non-contiguous class ids
    design = rng.normal(size=(900, 2)) + np.array(
        [[0, 0], [4, 0], [0, 4]]).repeat(300, axis=0)
    probs = logistic_cv(design, labels, rng.integers(0, 5, 900))
    assert probs.shape == (900, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (np.unique(labels)[probs.argmax(axis=1)] == labels).mean() > 0.95


def _stub_data(n_per_type: int, seed: int = 0):
    """Two ordinary types on a plane, plus the pieces the analyses touch."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_type
    positions = rng.uniform(0, 1000, size=(n, 2))
    t = np.repeat([0, 1], n_per_type)
    from scipy.spatial import cKDTree

    pairs = cKDTree(positions).query_pairs(25.0, output_type="ndarray")
    edge_i, edge_j = pairs[:, 0], pairs[:, 1]
    degrees = np.bincount(np.concatenate([edge_i, edge_j]), minlength=n)
    # niche structure: left / right halves have different "composition"
    y = np.stack([positions[:, 0] / 1000.0, 1.0 - positions[:, 0] / 1000.0],
                 axis=1) + 0.05 * rng.normal(size=(n, 2))
    graph = SimpleNamespace(n_cells=n, edge_i=edge_i, edge_j=edge_j,
                            degrees=degrees, y=y)
    return SimpleNamespace(
        graph=graph, t=t, positions=positions,
        totals=rng.integers(50, 300, n).astype(float),
        p_t=np.array([0.5, 0.5]),
        type_names=np.array(["Alpha Cells", "Beta Cells"]),
        gene_names=np.array([f"G{i}" for i in range(30)]),
        x=rng.poisson(1.0, size=(n, 30)).astype(float),
        cycle=None,
    ), rng


def test_niche_invariance_scores_the_planted_asymmetry():
    data, rng = _stub_data(3000)
    n = data.graph.n_cells
    # w encodes the niche-defining composition; z is pure noise
    latents = {"mu_w": data.graph.y + 0.02 * rng.normal(size=(n, 2)),
               "mu_z": rng.normal(size=(n, 4)),
               "fold": (data.positions[:, 1] // 200).astype(int) % 5}
    result = analysis_niche(data, latents, k_grid=(2,), max_cells=4000,
                            seed=0)
    pooled = result["K2"]["pooled_auc"]
    assert pooled["w"] > 0.95
    assert pooled["z"] < 0.6
    assert abs(pooled["floor"] - 0.5) < 0.1


def test_landmark_inventory_and_banded_regression():
    data, rng = _stub_data(3000)
    n = data.graph.n_cells
    # plant a structural type: 6 tight vessel blobs of 25 cells each
    centres = rng.uniform(100, 900, size=(6, 2))
    vessel_positions = np.repeat(centres, 25, axis=0) + rng.normal(
        0, 5.0, size=(150, 2))
    data.positions = np.vstack([data.positions, vessel_positions])
    data.t = np.concatenate([data.t, np.full(150, 2)])
    data.type_names = np.array(["Alpha Tumor Cells", "Beta Cells",
                                "Vessel Endothelial Cells"])
    data.p_t = np.array([0.45, 0.45, 0.1])
    data.totals = np.concatenate([data.totals,
                                  rng.integers(50, 300, 150).astype(float)])
    data.x = np.vstack([data.x, rng.poisson(1.0, size=(150, 30))])
    from scipy.spatial import cKDTree

    pairs = cKDTree(data.positions).query_pairs(25.0, output_type="ndarray")
    data.graph = SimpleNamespace(
        n_cells=n + 150, edge_i=pairs[:, 0], edge_j=pairs[:, 1],
        degrees=np.bincount(pairs.ravel(), minlength=n + 150),
        y=np.zeros((n + 150, 2)))

    classes = landmark_inventory(data)
    assert "vasculature" in classes           # merged endothelial types
    assert classes["vasculature"]["n_instances"] >= 5
    assert "tumor_stroma_interface" in classes

    distance = cKDTree(vessel_positions).query(data.positions)[0]
    u = np.log1p(np.minimum(distance, 500.0))
    latents = {"mu_w": np.stack(
                   [u + 0.05 * rng.normal(size=n + 150),
                    rng.normal(size=n + 150)], axis=1),
               "mu_z": rng.normal(size=(n + 150, 4)),
               "fold": (data.positions[:, 1] // 200).astype(int) % 5}
    result = analysis_landmarks(data, latents, b_matrix=rng.normal(
        size=(30, 2)), max_cells=5000, seed=0)
    vessel = result["classes"]["vasculature"]["pooled"]
    assert vessel["w"]["mid"] > 0.8          # the planted allegiance
    assert vessel["z"]["mid"] < 0.1
    assert abs(vessel["floor"]["mid"]) < 0.1
    assert "gene_signatures" in result


def test_morans_analysis_flags_collapsed_dims():
    data, rng = _stub_data(1500)
    n = data.graph.n_cells
    smooth = np.sin(data.positions[:, 0] / 60.0)
    latents = {"mu_w": np.stack(
                   [smooth, np.full(n, 3.14)], axis=1),   # dim 1 collapsed
               "mu_z": rng.normal(size=(n, 3)),
               "fold": np.zeros(n, dtype=int)}
    result = analysis_morans(data, latents, n_perms=100, seed=0)
    assert result["mu_w"]["collapsed"] == [False, True]
    assert result["mu_w"]["I"][0] > result["mu_w"]["null_hi"][0]
    hot_z = [i for i, (value, hi) in enumerate(
        zip(result["mu_z"]["I"], result["mu_z"]["null_hi"])) if value > hi]
    assert not hot_z                         # noise z inside the null band
