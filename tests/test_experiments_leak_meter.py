"""Planted worlds for the leak meter (`discell/experiments/leak_meter.py`).

Two gates, both on arrays only -- no bundle, no slide:

1. **Recovery.** A world where the document's assumptions hold exactly
   (genes specific against every other type, a receiver that expresses none
   of them, one rim share ``zeta`` shared by all genes, retention varying
   across genes) must return the planted ``c_rt = a_r b_t``, the planted
   ``zeta``, and per-cell ``kappa_i`` equal to the planted leaked share.
2. **The predicted bias.** Let the receiver express the sender's "specific"
   genes itself and the estimate must move *up* -- the document names this
   as why weak senders are upper bounds.

The lattice is three-coloured so no two neighbours share a type: then the
same-type entries (which no gene can measure) are genuinely absent, the
sender retention is not contaminated by its own leak, and the off-diagonal
table alone identifies ``a`` and ``b``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from discell.experiments import leak_meter as lm

N_SIDE = 48
GENES_PER_TYPE = 10
TYPES = 3
BASE_COUNTS = 2000.0
TAU_UM = 20.0
ZETA = 0.05
A_TRUE = np.array([1.0, 1.5, 0.7])       # receiver tendency
B_TRUE = np.array([1.0, 3.0, 0.3])       # sender tendency


def _eta(n_genes: int) -> np.ndarray:
    """Retention per gene and type, deliberately spread and != ``ZETA``."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.40, 0.85, size=(TYPES, n_genes))


def _lattice() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Square lattice, 4-connected, three-coloured by ``(x + 2y) % 3``."""
    xs, ys = np.meshgrid(np.arange(N_SIDE), np.arange(N_SIDE), indexing="ij")
    xs, ys = xs.ravel(), ys.ravel()
    index = (xs * N_SIDE + ys)
    edges = []
    for dx, dy in ((1, 0), (0, 1)):
        ok = (xs + dx < N_SIDE) & (ys + dy < N_SIDE)
        edges.append(np.stack([index[ok], ((xs + dx) * N_SIDE + (ys + dy))[ok]], axis=1))
    edge = np.concatenate(edges)
    positions = np.stack([xs, ys], axis=1).astype(np.float64) * 10.0
    types = (xs + 2 * ys) % 3
    return edge[:, 0], edge[:, 1], positions, types


def _plant(receiver_contamination: float = 0.0, seed: int = 0) -> tuple[lm.Slide, np.ndarray]:
    """A slide where every count is placed by hand; returns it and true kappa.

    ``receiver_contamination`` is the share of a receiver's own counts spent
    on the *sender* types' specific genes -- zero in the clean world.
    """
    edge_i, edge_j, positions, types = _lattice()
    n, g = len(types), TYPES * GENES_PER_TYPE
    eta = _eta(g)
    rng = np.random.default_rng(seed)

    # own profile: type t spends its budget on block t, plus an optional
    # equal-weight tail on every other block (the contamination).
    own = np.zeros((TYPES, g))
    for t in range(TYPES):
        block = slice(t * GENES_PER_TYPE, (t + 1) * GENES_PER_TYPE)
        own[t, block] = rng.uniform(0.5, 1.5, GENES_PER_TYPE)
        own[t] = own[t] / own[t].sum() * (1.0 - receiver_contamination)
        other = np.ones(g, dtype=bool)
        other[block] = False
        own[t, other] += receiver_contamination / other.sum()

    area = np.full(n, 100.0)
    weight = 1.0 * np.exp(-10.0 / TAU_UM)          # face 1 um, spacing 10 um
    face = np.ones(len(edge_i))
    dist = np.full(len(edge_i), 10.0)
    c_true = A_TRUE[:, None] * B_TRUE[None, :]

    # fixed point: the leaked mass depends on the neighbours' density, which
    # depends on their totals, which include what leaked into them.
    total = np.full(n, BASE_COUNTS)
    for _ in range(50):
        dens = total / area
        leaked = np.zeros((n, g))
        for src, dst in ((edge_i, edge_j), (edge_j, edge_i)):
            amount = c_true[types[dst], types[src]] * weight * dens[src]
            np.add.at(leaked, dst, amount[:, None] * own[types[src]])
        new_total = BASE_COUNTS + leaked.sum(axis=1)
        if np.allclose(new_total, total, rtol=1e-12):
            break
        total = new_total

    counts = BASE_COUNTS * own[types] + leaked
    nuclear = BASE_COUNTS * own[types] * eta[types] + ZETA * leaked
    kappa_true = leaked.sum(axis=1) / counts.sum(axis=1)
    slide = lm.Slide(
        counts=sp.csr_matrix(counts), nuclear=sp.csr_matrix(nuclear),
        type_index=types, type_names=np.array(["A", "B", "C"]),
        edge_i=edge_i, edge_j=edge_j, face_um=face, dist_um=dist,
        area_um2=area, positions_um=positions,
        has_nucleus=np.ones(n, dtype=bool))
    return slide, kappa_true


@pytest.fixture(scope="module")
def clean() -> tuple[lm.Slide, np.ndarray, dict]:
    slide, kappa_true = _plant()
    res = lm.measure(slide, min_counts=100.0, tau_um=TAU_UM, tile_cells=256)
    return slide, kappa_true, res


def test_specific_genes_are_the_planted_blocks(clean):
    _, _, res = clean
    for t, genes in res["specific"].items():
        assert set(genes.tolist()) == set(range(t * GENES_PER_TYPE, (t + 1) * GENES_PER_TYPE))


def test_recovers_the_planted_table_and_zeta(clean):
    _, _, res = clean
    c, zeta = res["table"]["c"], res["table"]["zeta"]
    truth = A_TRUE[:, None] * B_TRUE[None, :]
    off = ~np.eye(TYPES, dtype=bool)
    assert np.isfinite(c[off]).all()
    assert np.allclose(c[off], truth[off], rtol=0.05)
    assert np.allclose(zeta[off], ZETA, atol=0.02)


def test_factorisation_predicts_the_unmeasurable_same_type_entries(clean):
    _, _, res = clean
    fac = res["factorisation"]
    truth = A_TRUE[:, None] * B_TRUE[None, :]
    assert fac["rel_rms"] < 0.05
    # gauge: mean sender tendency is 1, so compare the product, not a or b.
    assert np.allclose(np.diag(fac["predicted"]), np.diag(truth), rtol=0.10)
    assert np.allclose(fac["b"] / fac["b"][0], B_TRUE / B_TRUE[0], rtol=0.05)


def test_per_cell_kappa_matches_the_planted_leak(clean):
    _, kappa_true, res = clean
    kappa = res["kappa"]
    assert np.isfinite(kappa).all()
    assert np.corrcoef(kappa, kappa_true)[0, 1] > 0.99
    assert np.sqrt(((kappa - kappa_true) ** 2).mean()) < 0.02
    assert abs(np.median(kappa) - np.median(kappa_true)) < 0.02


def test_receiver_expressing_the_specific_genes_biases_c_upward(clean):
    """The document's weak-sender caveat, planted: a receiver that makes the
    sender's genes itself pushes ``c_rt`` up, so the estimate is an upper
    bound rather than a measurement."""
    _, _, res_clean = clean
    slide, _ = _plant(receiver_contamination=0.05)
    res = lm.measure(slide, min_counts=100.0, tau_um=TAU_UM, tile_cells=256,
                     specific=res_clean["specific"])
    off = ~np.eye(TYPES, dtype=bool)
    assert (res["table"]["c"][off] > res_clean["table"]["c"][off]).all()
    assert np.median(res["kappa"]) > np.median(res_clean["kappa"])
