"""The label-independent cycling set (author's decision, devlog 2026-09-28).

The cycle reads pool over the top decile of the S + G2M score among a slide's
held-out cells instead of the top-4 MKI67 types of the label column. Planted
worlds check that the set is exactly a decile of the eligible held-out cells,
that it respects the Unassigned and held-out masks, that it does not move
when the labels do (the label-derived set does), and that the two sets give
different R^2 on a world built so that they should.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from discell.model import cell_cycle as CC
from discell.model import metrics as M


def _world(n: int = 20000, seed: int = 0):
    """Six types with planted cycling fractions; the score is the cycling
    intensity plus noise, z carries the intensity, w carries nothing."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 6, n)
    frac = np.array([0.4, 0.2, 0.05, 0.05, 0.02, 0.02])
    cycling = rng.random(n) < frac[t]
    a = np.where(cycling, rng.uniform(1.0, 3.0, n), 0.0)
    scores = np.stack([a + 0.5 * rng.standard_normal(n),
                       a + 0.5 * rng.standard_normal(n)], axis=1)
    z = np.hstack([a[:, None] + 0.05 * rng.standard_normal((n, 1)),
                   np.eye(6)[t], rng.standard_normal((n, 3))])
    w = rng.standard_normal((n, 4))
    held = rng.random(n) < 0.3
    return dict(t=t, scores=scores, z=z, w=w, held=held, cycling=cycling)


def test_the_set_is_exactly_a_decile_of_the_eligible_held_out_cells():
    w = _world()
    eligible = w["t"] != 5                            # type 5 plays Unassigned
    cells = CC.cycling_set(w["scores"], w["held"], eligible=eligible)
    candidates = w["held"] & eligible
    assert (cells & candidates).sum() == round(0.1 * candidates.sum())
    # masks respected: no excluded cell, held out or not
    assert not (cells & ~eligible).any()
    # the held-out part is the top of the candidates by S + G2M ...
    total = w["scores"].sum(axis=1)
    cut = total[cells & w["held"]].min()
    assert (total[candidates & ~cells] <= cut).all()
    # ... and a training cell is in iff it is eligible and reaches the cut
    train = ~w["held"]
    np.testing.assert_array_equal(cells[train],
                                  (eligible & (total >= cut))[train])
    # q moves the size, nothing else
    q80 = CC.cycling_set(w["scores"], w["held"], q=0.8, eligible=eligible)
    assert (q80 & candidates).sum() == round(0.2 * candidates.sum())
    assert (cells <= q80).all()


def test_ties_do_not_break_the_exact_count():
    scores = np.zeros((1000, 2))                      # every cell tied
    held = np.arange(1000) % 2 == 0
    cells = CC.cycling_set(scores, held)
    assert (cells & held).sum() == 50
    assert CC.cycling_set(scores[:5], held[:5]).sum() == 0   # round(0.3) = 0


def _slide(t, type_names, scores, held):
    """The fields ``slide_cycling_set`` reads off an assembled slide."""
    return SimpleNamespace(
        n_cells=len(t), t=t, type_names=np.asarray(type_names),
        val_tiles=[np.flatnonzero(held)],
        cycle={"s_score": scores[:, 0], "g2m_score": scores[:, 1]})


def test_the_decile_set_is_invariant_to_relabelling_the_label_set_is_not():
    w = _world()
    names = ["A", "B", "C", "D", "E", "Unassigned"]
    base = CC.slide_cycling_set(_slide(w["t"], names, w["scores"], w["held"]))
    # a bijective relabel (renamed, reordered) and a merge into lineages;
    # Unassigned stays Unassigned
    perm = np.array([3, 0, 4, 1, 2, 5])
    renamed = [f"type{g}" for g in range(5)] + ["Unassigned"]
    merged = np.array([0, 1, 2, 3, 0, 4])[w["t"]]
    for t_new, names_new in ((perm[w["t"]], renamed),
                             (merged, ["L0", "L1", "L2", "L3", "Unassigned"])):
        again = CC.slide_cycling_set(_slide(t_new, names_new, w["scores"],
                                            w["held"]))
        np.testing.assert_array_equal(again, base)
    # the retired set: the top-4 types by cycling fraction (MKI67 in the
    # trainer), Unassigned (the last label) skipped, covers 4 of the 5 types
    # before the merge and every lineage after it
    def label_set(t):
        k = t.max()
        frac = np.array([w["cycling"][t == g].mean() for g in range(k)])
        return np.isin(t, np.argsort(-frac)[:4])
    assigned = w["t"] != 5
    assert label_set(w["t"])[assigned].mean() < 0.9
    np.testing.assert_array_equal(label_set(merged), assigned)


def test_label_derived_and_decile_sets_give_different_r2():
    w = _world()
    train, test = ~w["held"], w["held"]
    label = M.cycle_r2(w["z"], w["t"], w["scores"], np.arange(4), train, test)
    cells = CC.cycling_set(w["scores"], w["held"])
    decile = M.cycle_r2(w["z"], w["t"], w["scores"], None, train, test,
                        cells=cells)
    assert abs(label["r2_pooled"] - decile["r2_pooled"]) > 0.1
    assert decile["r2_pooled"] > 0.2                  # z does carry cycle
    blind = M.cycle_r2(w["w"], w["t"], w["scores"], None, train, test,
                       cells=cells)
    assert abs(blind["r2_pooled"]) < 0.02
    # the decile read under a bijective relabel: same cells, same centring
    perm = np.array([3, 0, 4, 1, 2, 5])
    again = M.cycle_r2(w["z"], perm[w["t"]], w["scores"], None, train, test,
                       cells=cells)
    assert again["r2_pooled"] == pytest.approx(decile["r2_pooled"], abs=1e-12)


def test_cells_path_reproduces_the_type_path_and_ignores_cells_outside():
    w = _world()
    train, test = ~w["held"], w["held"]
    types = np.array([0, 2, 3])
    by_types = M.cycle_r2(w["z"], w["t"], w["scores"], types, train, test,
                          seed=3)
    by_cells = M.cycle_r2(w["z"], w["t"], w["scores"], None, train, test,
                          seed=3, cells=np.isin(w["t"], types))
    assert by_cells == by_types
    # what lies outside the set never enters the read
    cells = CC.cycling_set(w["scores"], w["held"])
    ref = M.cycle_r2(w["z"], w["t"], w["scores"], None, train, test,
                     seed=1, cells=cells)
    z = w["z"].copy()
    z[~cells] = 1e3 * np.random.default_rng(9).standard_normal(z[~cells].shape)
    scores = w["scores"].copy()
    scores[~cells] = -50.0
    out = M.cycle_r2(z, w["t"], scores, None, train, test, seed=1, cells=cells)
    assert out["r2_pooled"] == pytest.approx(ref["r2_pooled"], abs=1e-12)
    assert out["r2_permuted"] == pytest.approx(ref["r2_permuted"], abs=1e-12)


def test_q90_keys_read_the_pooled_r2():
    block = {"z": {"r2_pooled": 0.5}, "w": {"r2_pooled": 0.01},
             "linear_ref": {"r2_pooled": 0.4}}
    assert CC.q90_keys(block) == {"cycle_r2_z_q90": 0.5, "cycle_r2_w_q90": 0.01,
                                  "cycle_linear_q90": 0.4}
    assert CC.q90_keys(None) == {"cycle_r2_z_q90": None, "cycle_r2_w_q90": None,
                                 "cycle_linear_q90": None}

