"""Known-answer tests for the doc-09 communication instruments.

Planted-truth checks on the pure pieces: exposure from the graph weights,
the composition residualisation, the depth-controlled association, and the
target-AUROC scoring. The full pipeline's falsification lives in its own
built-in controls (decoy exposure, receptor-negative receivers)."""

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from discell.model.communication import (
    exposure_midband, exposure_onehop, partial_association, residualise,
    target_auroc)


def test_exposure_onehop_matches_hand_computation():
    # 3 cells: 0 receives from 1 (beta .7) and 2 (beta .3)
    in_edges = sp.csr_matrix(np.array([[0.0, 0.7, 0.3],
                                       [1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0]]))
    data = SimpleNamespace(graph=SimpleNamespace(in_edges=in_edges))
    ligand = np.array([9.0, 2.0, 4.0])       # x_L / l per cell
    exposure = exposure_onehop(data, ligand)
    assert np.allclose(exposure, [0.7 * 2 + 0.3 * 4, 9.0, 2.0])


def test_exposure_midband_sees_shell_not_neighbours():
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 800, size=(4000, 2))
    ligand = np.zeros(4000)
    # a hot source at the centre; cells ~100 um away must read it, cells
    # adjacent to it (< 30 um) and far ones (> 200 um) must not
    centre = np.array([[400.0, 400.0]])
    positions[:50] = centre + rng.normal(0, 3, size=(50, 2))
    ligand[:50] = 5.0
    data = SimpleNamespace(positions=positions)
    field = exposure_midband(data, ligand)
    dist = np.hypot(*(positions - centre).T)
    mid = field[(dist > 80) & (dist < 120)].mean()
    far = field[dist > 250].mean()
    assert mid > 10 * max(far, 1e-9)


def test_residualise_removes_composition_signal():
    rng = np.random.default_rng(1)
    n = 5000
    y = rng.dirichlet(np.ones(3), size=n)
    # exposure driven by composition plus an independent within-composition part
    independent = rng.normal(size=n)
    exposure = 4.0 * y[:, 0] + 0.5 * independent
    data = SimpleNamespace(graph=SimpleNamespace(y=y))
    members = np.arange(n)
    e_tilde, ratio = residualise(exposure, data, members)
    assert 0.05 < ratio < 0.5                  # composition part removed
    assert abs(np.corrcoef(e_tilde, y[:, 0])[0, 1]) < 0.05
    assert np.corrcoef(e_tilde, independent)[0, 1] > 0.9


def test_partial_association_controls_depth():
    rng = np.random.default_rng(2)
    n = 4000
    log_l = rng.normal(size=n)
    e_tilde = rng.normal(size=n)
    x = np.stack([
        0.8 * e_tilde + 0.3 * rng.normal(size=n),   # true response
        2.0 * log_l + 0.3 * rng.normal(size=n),     # depth artefact
        rng.normal(size=n),                          # noise
    ], axis=1)
    assoc = partial_association(x, e_tilde, log_l)
    assert assoc[0] > 0.8
    assert abs(assoc[1]) < 0.05
    assert abs(assoc[2]) < 0.05


def test_target_auroc_orders_planted_program():
    rng = np.random.default_rng(3)
    scores = rng.random(500)
    targets = np.zeros(500, dtype=bool)
    targets[np.argsort(-scores)[:30]] = True   # targets = the top scores
    assert target_auroc(scores, targets) == 1.0
    shuffled = targets[rng.permutation(500)]
    assert 0.35 < target_auroc(scores, shuffled) < 0.65
