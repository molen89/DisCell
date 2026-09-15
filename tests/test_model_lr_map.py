"""Doc-09 section 8: the co-occurrence map on planted structure.

Three column/row pairs in a two-type toy tissue: one whose co-occurrence is
composition-mediated (both sides driven by y) -- hot in panel A, cold in
panel B; one genuine (a shared latent independent of y) -- hot in both; one
null -- cold in both. Plus the effective-rank row selection (issues V10).
"""

from __future__ import annotations

import numpy as np

from discell.model.lr_map import (benjamini_hochberg, cooccurrence,
                                  effective_programs, program_basis)


def _toy(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    y = rng.dirichlet(np.ones(3), n)
    shared = rng.normal(size=n)
    rows = np.stack([y[:, 0] * 3 + rng.normal(size=n) * 0.5,
                     shared + rng.normal(size=n) * 0.5,
                     rng.normal(size=n)], axis=1)
    cols = np.stack([y[:, 0] * 2 + rng.normal(size=n) * 0.5,
                     shared + rng.normal(size=n) * 0.5,
                     rng.normal(size=n)], axis=1)
    eligible = np.ones((2, 3), dtype=bool)
    return rows, cols, t, y, eligible


def test_spatial_shift_null_gives_the_same_verdicts_on_unstructured_fields():
    rows, cols, t, y, eligible = _toy()
    positions = np.random.default_rng(1).uniform(0, 5000, (len(t), 2))
    b = cooccurrence(rows, cols, t, y, eligible, n_perms=100, seed=0,
                     partial=True, positions=positions)
    qb = benjamini_hochberg(b["p"])
    assert qb[0, 0] > 0.05 and qb[1, 1] < 0.05 and qb[2, 2] > 0.05


def test_composition_mediated_pair_is_hot_in_a_and_cold_in_b():
    rows, cols, t, y, eligible = _toy()
    a = cooccurrence(rows, cols, t, y, eligible, n_perms=100, seed=0)
    b = cooccurrence(rows, cols, t, y, eligible, n_perms=100, seed=0, partial=True)
    qa, qb = benjamini_hochberg(a["p"]), benjamini_hochberg(b["p"])
    assert a["rho"][0, 0] > 0.5 and qa[0, 0] < 0.05          # composition pair, A
    assert abs(b["rho"][0, 0]) < 0.1 and qb[0, 0] > 0.05      # ... gone in B
    assert qa[1, 1] < 0.05 and qb[1, 1] < 0.05                # genuine pair, both
    assert b["rho"][1, 1] > 0.5
    assert qa[2, 2] > 0.05 and qb[2, 2] > 0.05                # null pair, neither


def test_ineligible_types_do_not_contribute():
    rows, cols, t, y, eligible = _toy()
    eligible[:, 2] = False                                    # column 2: no type
    out = cooccurrence(rows, cols, t, y, eligible, n_perms=20, seed=0)
    assert np.isnan(out["rho"][:, 2]).all() and out["n_cells"][2] == 0
    assert out["n_cells"][0] == len(t)


def test_effective_programs_use_the_rank_of_w_not_d_w():
    rng = np.random.default_rng(0)
    n, genes = 5000, 200
    latent = rng.normal(size=(n, 2)) * [3.0, 1.5]              # rank-2 w in 6 dims
    mix = rng.normal(size=(2, 6))
    mu_w = latent @ mix + rng.normal(size=(n, 6)) * 1e-3
    b_matrix = rng.normal(size=(genes, 6))
    rows, info = effective_programs(mu_w, b_matrix, np.ones(genes, dtype=bool), top=5)
    assert info["rank"] == 2
    assert {k for _, k, _ in rows} == {0, 1}
    assert len(rows) <= 20 and len({g for g, _, _ in rows}) == len(rows)


def test_program_coordinates_reproduce_the_realised_shift():
    rng = np.random.default_rng(0)
    n, genes = 4000, 150
    latent = rng.normal(size=(n, 2)) * [3.0, 1.5]
    mu_w = latent @ rng.normal(size=(2, 6)) + rng.normal(size=(n, 6)) * 1e-3
    b_matrix = rng.normal(size=(genes, 6))
    u, programs, info = program_basis(mu_w, b_matrix, np.ones(genes, dtype=bool))
    assert info["rank"] == 2 and u.shape == (n, 2) and programs.shape == (genes, 2)
    shift = (mu_w - mu_w.mean(axis=0)) @ b_matrix.T
    assert np.allclose(u @ programs.T, shift, atol=1e-2 * np.abs(shift).max())
