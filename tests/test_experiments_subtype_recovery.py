"""Subtype recovery (devlog 2026-09-28, part B): which channel reads a sublabel.

Planted lineage: three sublabels -- a majority reference (~72 %, like
"Tumor Cells" in the ovarian Tumour lineage), a *state* (~14 %) written into z
only, and a *location* (~14 %, spatial stripes) written into w and the
context prior mean only. The state must be called ``z ≫ w`` (and
``z ≫ mpsi``), the location ``w ≫ z`` (and ``mpsi ≫ z``), both verdicts met
with no contradiction. With the labels permuted every read must sit at its
floor.

One-vs-rest is what the call reads, so a channel that separates *another*
sublabel from this one earns part of this one's AUC (w lends the state AUC by
separating the location from it, in proportion to the location's share of
"the rest"); the ``DOMINANCE`` rule tolerates up to half of the strong
channel's excess.
"""

from __future__ import annotations

import numpy as np
import pytest

from discell.experiments import subtype_recovery as S

NAMES = ["ref", "state", "location"]
KINDS = {"ref": "lineage", "state": "state", "location": "location"}


def planted(n: int = 6000, seed: int = 0, sep: float = 3.0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 3000, size=(n, 2))
    location = (xy[:, 0] % 600) < 84
    state = ~location & (rng.random(n) < 1 / 6)
    y = np.where(location, 2, np.where(state, 1, 0))
    latents = {
        "z": np.hstack([sep * state[:, None] + rng.normal(size=(n, 1)),
                        rng.normal(size=(n, 3))]),
        "w": np.hstack([sep * location[:, None] + rng.normal(size=(n, 1)),
                        rng.normal(size=(n, 2))]),
        "mpsi": np.hstack([sep * location[:, None]
                           + rng.normal(scale=0.5, size=(n, 1)),
                           rng.normal(size=(n, 1))]),
    }
    latents["zw"] = np.hstack([latents["z"], latents["w"]])
    # held out = whole 300 um blocks, as the run's validation tiles are
    block = (xy[:, 0] // 300).astype(int) * 10 + (xy[:, 1] // 300).astype(int)
    test = np.isin(block, rng.choice(100, 30, replace=False))
    return latents, y, xy, test


@pytest.fixture(scope="module")
def graded():
    latents, y, xy, test = planted()
    return S.grade_group(latents, y, NAMES, ~test, test, xy, seed=0,
                         n_perm=5, n_boot=200)


def test_state_in_z_and_location_in_w_are_called_correctly(graded):
    calls = graded["calls"]
    assert calls["state"]["ridge|z-w"] == "z ≫ w"
    assert calls["state"]["ridge|z-mpsi"] == "z ≫ mpsi"
    assert calls["state"]["mlp|z-w"] == "z ≫ w"
    assert calls["location"]["ridge|z-w"] == "w ≫ z"
    assert calls["location"]["ridge|z-mpsi"] == "mpsi ≫ z"
    assert calls["location"]["mlp|z-w"] == "w ≫ z"
    for name in ("state", "location"):
        expected = S.expectation(name, KINDS[name], False)
        v = S.verdict(KINDS[name], expected, calls[name])
        assert v["met"] is True and v["contradiction"] is None, (name, v)
    # the reference has no registered expectation
    assert S.verdict("lineage", S.expectation("ref", "lineage", False),
                     calls["ref"])["met"] is None
    # the lineage read: [z, w] beats either channel alone, all beat the floor
    r = graded["reads"]
    ba = {k: r[k]["ba"]["estimate"] for k in r}
    assert ba["ridge|zw"] > max(ba["ridge|z"], ba["ridge|w"]) + 0.1
    for k in r:
        assert r[k]["ba"]["ci95"][0] > r[k]["floor"]["ba_mean"]
    assert graded["baseline"]["ba"] == pytest.approx(1 / 3)
    # the state's AUC from z is near perfect; its AUC from w is only what
    # w's separation of the location lends it (one-vs-rest), well below
    assert r["ridge|z"]["auc"]["state"]["estimate"] > 0.95
    assert r["ridge|w"]["auc"]["state"]["estimate"] < 0.65


def test_permuted_labels_sit_at_the_floor():
    latents, y, xy, test = planted(seed=1)
    y_perm = y[np.random.default_rng(9).permutation(len(y))]
    out = S.grade_group(latents, y_perm, NAMES, ~test, test, xy, seed=0,
                        n_perm=5, n_boot=200)
    # labels independent of every design: E[BA] = 1/K, E[AUC] = 1/2; the
    # tolerance is ~3 sd of the read at these class sizes (~250 held-out
    # cells for each minority sublabel)
    for key, r in out["reads"].items():
        assert abs(r["ba"]["estimate"] - r["floor"]["ba_mean"]) < 0.05, key
        assert abs(r["floor"]["ba_mean"] - 1 / 3) < 0.02, key
        for name in NAMES:
            assert abs(r["auc"][name]["estimate"] - 0.5) < 0.05, (key, name)
    for name, entry in out["calls"].items():
        for pair in ("ridge|z-w", "ridge|z-mpsi", "mlp|z-w"):
            assert entry[pair] == "neither", (name, pair, entry[pair])


def test_weighted_scores_match_sklearn_and_multiplicities():
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 500)
    pred = np.where(rng.random(500) < 0.6, y, rng.integers(0, 4, 500))
    s = S.scores(S.confusion(y, pred, 4))
    assert s["ba"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert s["macro_f1"] == pytest.approx(
        f1_score(y, pred, average="macro", labels=range(4)))
    score = np.round(rng.normal(size=500) + (y == 2), 1)          # with ties
    groups, n = S.auc_ranks(score)
    assert S.weighted_auc(groups, n, y == 2) == pytest.approx(
        roc_auc_score(y == 2, score))
    # integer weights = duplicated cells
    w = rng.integers(0, 3, 500).astype(float)
    rep = np.repeat(np.arange(500), w.astype(int))
    assert S.scores(S.confusion(y, pred, 4, w))["ba"] == pytest.approx(
        balanced_accuracy_score(y[rep], pred[rep]))
    assert S.weighted_auc(groups, n, y == 2, w) == pytest.approx(
        roc_auc_score(y[rep] == 2, score[rep]))


def test_groups_follow_the_applied_map():
    old = np.array(["A", "A2", "B", "C", "C", "Cyst"])
    lineage = np.array(["LA", "LA", "LB", "LC", "LC", "LCyst"])
    tables = {"pairs": {("A", "LA"), ("A2", "LA"), ("B", "LB"), ("C", "LC"),
                        ("Cyst", "LCyst")}}
    groups = S.build_groups(old, lineage, tables,
                            extra=(("LA + cyst", ("LA", "LCyst")),
                                   ("absent", ("LA", "LZ"))))
    assert [g["name"] for g in groups] == ["LA", "LA + cyst"]
    assert groups[0]["t_confounded"] is False and groups[1]["t_confounded"]
    with pytest.raises(ValueError, match="not in the applied map"):
        S.build_groups(np.array(["A", "X"]), np.array(["LA", "LA"]), tables)
    assert S.expectation(S.CYST_LABEL, "lineage", True) == S.T_CONFOUNDED
    assert S.expectation("TAF", "location", False) == "w/m_ψ"
    assert S.expectation("Prolif", "state", False) == "z"
    assert S.expectation("Cluster-1", "mixed", False) is None


def test_contradictions_are_flagged():
    rec = {"ridge|z": True, "ridge|w": True, "ridge|mpsi": False}
    state = S.verdict("state", "z", {"recovered": rec, "ridge|z-w": "both",
                                     "ridge|z-mpsi": "z ≫ mpsi"})
    assert state["met"] is False and "state read from μ_w" in state["contradiction"]
    loc = S.verdict("location", "w/m_ψ", {"recovered": rec,
                                          "ridge|z-w": "unresolved",
                                          "ridge|z-mpsi": "z ≫ mpsi"})
    assert loc["met"] is False and "niche in z" in loc["contradiction"]
    ok = S.verdict("location", "w/m_ψ", {"recovered": rec,
                                         "ridge|z-w": "both",
                                         "ridge|z-mpsi": "mpsi ≫ z"})
    assert ok["met"] is True and ok["contradiction"] is None


def test_summary_and_markdown_over_seeds(graded, tmp_path):
    def record(result, seed):
        subs = [{"name": n, "kind": KINDS[n], "n_train": 1, "n_test": 1,
                 "expectation": S.expectation(n, KINDS[n], False)}
                for n in NAMES]
        for e in subs:
            e["verdict"] = S.verdict(e["kind"], e["expectation"],
                                     result["calls"][e["name"]])
        group = {"name": "Planted", "lineages": ["Planted"],
                 "t_confounded": False, "sublabels": subs, "dropped": [],
                 "graded": True, **result}
        return {"dataset": "synthetic", "run": f"s{seed}", "seed": seed,
                "groups": [group]}

    summary = S.summarise([record(graded, 0), record(graded, 1)])
    rows = {s["name"]: s for s in summary["groups"][0]["sublabels"]}
    assert rows["state"]["calls"]["ridge|z-w"]["majority"] == "z ≫ w"
    assert rows["location"]["verdict"]["met"] is True
    # the reference is separated from the state by z and from the location
    # by w: both channels recover it one-vs-rest
    assert rows["ref"]["calls"]["ridge|z-w"]["majority"].startswith("both")
    text = S.markdown(summary)
    assert "| Planted | state | state |" in text and "met (2/2)" in text
    S.figure(summary, tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 0
