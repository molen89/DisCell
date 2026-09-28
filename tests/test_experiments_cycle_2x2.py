"""Cycle read 2x2 (lineage relabel): the centring decides what the read sees.

Planted world: a "proliferative" STATE label encodes the cycle score and the
latent encodes only that state. With the state as a type (the read centred
per state) nothing is left to predict (R^2 ~ 0); with the state merged into
its lineage (centred per lineage) or uncentred, the latent predicts the score
through the state alone (R^2 ~ 1) -- the between-type circularity, visible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from discell.experiments import cycle_2x2 as C
from discell.model import metrics as M


def planted_state(n: int = 6000, seed: int = 0):
    rng = np.random.default_rng(seed)
    lineage = (rng.random(n) < 0.3).astype(np.int64)        # 0 tumour, 1 stroma
    prolif = (lineage == 0) & (rng.random(n) < 0.3)
    state = np.where(lineage == 1, 2, np.where(prolif, 0, 1))  # state as a type
    scores = prolif[:, None] * np.ones((1, 2)) + rng.normal(scale=0.05, size=(n, 2))
    latent = np.hstack([prolif[:, None] + rng.normal(scale=0.01, size=(n, 1)),
                        rng.normal(size=(n, 3))])
    train = rng.random(n) < 0.5
    return lineage, state, scores, latent, train


def test_state_label_that_encodes_the_score_is_invisible_within_type():
    lineage, state, scores, latent, train = planted_state()
    types = [0, 1, 2]
    within_state = C.grouped_r2(latent, state, types, None, scores, train)
    by_lineage = C.grouped_r2(latent, state, types, lineage, scores, train)
    uncentred = C.grouped_r2(latent, state, types, np.zeros_like(state),
                             scores, train)
    assert abs(within_state["r2_pooled"]) < 0.02
    assert by_lineage["r2_pooled"] > 0.95
    assert uncentred["r2_pooled"] > 0.95
    # the within-type permuted control stays at zero in every centring
    for r in (within_state, by_lineage, uncentred):
        assert abs(r["r2_permuted"]) < 0.02
    # and the same world read with the lineage as the set's own label
    lin = C.grouped_r2(latent, lineage, [0, 1], None, scores, train)
    lin_by_state = C.grouped_r2(latent, lineage, [0, 1], state, scores, train)
    assert lin["r2_pooled"] > 0.95 and abs(lin_by_state["r2_pooled"]) < 0.02
    # the between-type share says the same thing from the score alone
    assert C.between_share(scores, state) > 0.95
    assert C.between_share(scores, lineage) < C.between_share(scores, state)


def test_genuine_within_type_signal_survives_every_centring():
    lineage, state, _, _, train = planted_state(seed=1)
    rng = np.random.default_rng(2)
    u = rng.normal(size=len(state))
    offset = np.array([2.0, 0.0, -1.0])[state]
    scores = np.stack([u + offset, u + offset], 1) + rng.normal(scale=0.1, size=(len(u), 2))
    latent = np.hstack([u[:, None], offset[:, None], rng.normal(size=(len(u), 2))])
    for groups in (None, lineage, np.zeros_like(state)):
        r = C.grouped_r2(latent, state, [0, 1, 2], groups, scores, train)
        assert r["r2_pooled"] > 0.9


def test_protocol_path_is_cycle_r2_bit_for_bit():
    lineage, state, scores, latent, train = planted_state(seed=3)
    types = np.array([1, 0, 2])
    got = C.grouped_r2(latent, state, types, None, scores, train, seed=5)
    want = M.cycle_r2(latent, state, scores, types, train, ~train, seed=5)
    assert got["r2_pooled"] == want["r2_pooled"]
    assert got["r2_permuted"] == want["r2_permuted"]


def test_split_sums_exactly_to_the_pooled_r2():
    rng = np.random.default_rng(4)
    lineage, state, scores, latent, train = planted_state(seed=4)
    scores = scores + rng.normal(scale=0.3, size=scores.shape)
    dec = C.decompose(latent, lineage, [0, 1], scores, train, seed=0)
    ref = M.cycle_r2(latent, lineage, scores, np.array([0, 1]), train, ~train)
    assert dec["r2_pooled"] == pytest.approx(ref["r2_pooled"], abs=1e-10)
    total = sum(v["contribution"] for v in dec["by_type"].values())
    assert total == pytest.approx(dec["r2_pooled"], abs=1e-10)
    assert sum(v["share"] for v in dec["by_type"].values()) == pytest.approx(1.0)
    # the cycle signal lives in the tumour lineage only
    assert dec["by_type"][0]["contribution"] > 0.5
    assert abs(dec["by_type"][1]["contribution"]) < 0.02


def test_cycling_set_skips_unassigned_and_keeps_four():
    names = ["A", "Unassigned", "B", "C", "D", "E"]
    assert C.cycling_set([1, 0, 3, 2, 5, 4], names) == [0, 3, 2, 5]


def test_coverage_counts_heldout_cells_and_mki67():
    t = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    test = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    mki = np.array([1, 0, 1, 0, 0, 0, 1, 1], bool)
    scores = np.arange(16, dtype=float).reshape(8, 2)
    cov = C.coverage(t, [0, 1], test, mki, scores, t, t)
    assert cov["n_heldout"] == 4
    assert cov["share_heldout"] == pytest.approx(4 / 6)
    assert cov["share_heldout_mki67pos"] == pytest.approx(1.0)
    assert cov["mki67pos_fraction"] == pytest.approx(0.5)


# -- the whole 2x2 on a fake pair of encodings --------------------------------

def _fake_encode(n: int = 5000):
    """Two 'runs' over one planted slide: old labels = clusters with a
    proliferative-tumour and a proliferative-T cluster; new labels =
    lineages. The old model's z holds the state and the within-state cycle;
    the new one only half the within-state part. w is noise."""
    rng = np.random.default_rng(0)
    lineage = rng.choice(3, n, p=[0.6, 0.3, 0.1])  # Tumour, T/NK, Unassigned
    prolif = (lineage < 2) & (rng.random(n) < 0.25)
    old_t = np.select([(lineage == 0) & prolif, lineage == 0,
                       (lineage == 1) & prolif, lineage == 1], [0, 1, 3, 2], 4)
    u = rng.normal(size=n)
    scores = (np.stack([prolif * 1.0, prolif * 1.0], 1)
              + 0.5 * np.stack([u, u], 1) * prolif[:, None]
              + rng.normal(scale=0.3, size=(n, 2))).astype(np.float32)
    mki67 = rng.random(n) < np.where(prolif, 0.6, 0.05)
    totals = rng.poisson(200, n).astype(np.float32) + 1
    x_pcs = np.hstack([scores + rng.normal(scale=0.5, size=(n, 2)),
                       rng.normal(size=(n, 4))]).astype(np.float32)
    rows = rng.permutation(n)
    n_train = int(0.6 * n)
    z_old = np.hstack([prolif[:, None], u[:, None] * prolif[:, None],
                       rng.normal(size=(n, 3))])
    z_new = np.hstack([prolif[:, None], 0.5 * u[:, None] * prolif[:, None]
                       + rng.normal(scale=0.5, size=(n, 1)) * prolif[:, None],
                       rng.normal(size=(n, 3))])
    old_names = ["Cluster-A", "Cluster-B", "Cluster-C", "Cluster-D", "Unassigned"]
    new_names = ["T/NK cells", "Tumour", "Unassigned"]
    new_t = np.array([1, 0, 2])[lineage]

    def enc(run, t, names, z):
        k = len(names)
        frac = np.array([mki67[t == g].mean() for g in range(k)])
        types = C.cycling_set(np.argsort(-frac), names)
        train = np.arange(n) < n_train
        tr = t[rows]
        sc = scores[rows]
        w = rng.normal(size=(n, 2))
        cycle = {"types": types,
                 "z": M.cycle_r2(z[rows], tr, sc, np.array(types), train, ~train),
                 "w": M.cycle_r2(w[rows], tr, sc, np.array(types), train, ~train),
                 "linear_ref": M.cycle_r2(x_pcs[rows], tr, sc, np.array(types),
                                          train, ~train),
                 "lbaseline": M.cycle_r2(np.log(totals[rows])[:, None].astype(
                     np.float64), tr, sc, np.array(types), train, ~train)}
        return {"run": run, "label_key": None if run.startswith("final_")
                else "lineage", "seed": 0, "rows": rows, "n_train": n_train,
                "mu_z": z[rows], "mu_w": w[rows], "t": t, "type_names": names,
                "types": types, "scores": scores, "x_pcs": x_pcs,
                "totals": totals, "mki67": mki67,
                "battery": {"at_best": True, "epoch": 7, "cycle": cycle}}

    return {"final_s0": enc("final_s0", old_t, old_names, z_old),
            "finalL_s0": enc("finalL_s0", new_t, new_names, z_new)}


def test_whole_2x2_runs_and_reproduces_its_diagonal(monkeypatch):
    fake = _fake_encode()
    monkeypatch.setattr(C, "encode", lambda dataset, run, device: fake[run])
    rec = C.one_seed("fake", 0, "cpu")
    assert rec["sets"]["old"]["types"][:2] == ["Cluster-D", "Cluster-A"] or \
        set(rec["sets"]["old"]["types"][:2]) == {"Cluster-A", "Cluster-D"}
    assert "Unassigned" not in rec["sets"]["new"]["types"]
    assert all(c["abs_diff"] < 1e-12 for c in rec["checks"].values())
    r = rec["reads"]
    # the lineage read keeps the planted state: the model-free reference
    # rises on it, and falls back once the new set is centred per old type
    assert (r["counts|new|lineage|linear_ref"]["r2"]
            > r["counts|old|old|linear_ref"]["r2"])
    assert (r["counts|new|old|linear_ref"]["r2"]
            < r["counts|new|lineage|linear_ref"]["r2"])
    # lineage centring keeps the state: both models gain from it on the old set
    for m in C.MODELS:
        assert r[f"{m}|old|lineage|z"]["r2"] > r[f"{m}|old|old|z"]["r2"]
    # the planted model change: the new z has less within-state cycle
    assert r["new|old|old|z"]["r2"] < r["old|old|old|z"]["r2"]
    split = rec["new_set_split"]["new|z"]
    assert sum(v["contribution"] for v in split["by_lineage"].values()) == \
        pytest.approx(split["r2_pooled"], abs=1e-10)
    result = C._jsonable({"dataset": "fake", "per_seed": [rec, rec],
                          "summary": C.summarise([rec, rec])})
    json.dumps(result)
    md = C.dataset_markdown(result)
    assert "The 2×2" in md and "Tumour" in md
    assert "Cycle read 2×2" in C.combined_markdown([result])
