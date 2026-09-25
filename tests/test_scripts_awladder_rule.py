"""The alpha_w ladder's pre-registered rule, applied to planted read-outs.

``scripts/awladder_table.py`` is loaded by path (scripts/ is not a package).
Every cell is three planted seeds; the reference rung of every dataset is the
same, so each test moves one clause of one rung.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "awladder_table.py"


@pytest.fixture
def table():
    """A fresh module per test: set_scope rebinds its globals."""
    spec = importlib.util.spec_from_file_location("awladder_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(i: int, **over) -> dict:
    run = {"run": f"s{i}", "recon": -7.20 + 0.002 * i, "nmi": 0.63 + 0.002 * i,
           "cycle_z": 0.50 + 0.01 * i, "cycle_w": 0.01, "w_mirror": 0.19 + 0.002 * i,
           "gain": 0.0002 + 0.0001 * i, "nmi_running_max_ok": True,
           "probe_at_floor": True, "niche_alive": True}
    run.update({k: (v[i] if isinstance(v, tuple) else v) for k, v in over.items()})
    return run


def _cells(table, changes: dict) -> dict:
    """{(dataset, rung): cell}; *changes* maps (dataset, rung) -> overrides,
    or -> "missing" to leave one seed out."""
    cells = {}
    for ds in table.DATASETS:
        for rung in table.RUNGS_OF[ds]:
            over = changes.get((ds, rung), {})
            seeds = (0, 1) if over == "missing" else table.SEEDS
            runs = [_seed(i, **({} if over == "missing" else over)) for i in seeds]
            cells[(ds, rung)] = {"alpha_w": 0.0, "runs": runs,
                                 **table.summarise(runs)}
    return cells


def _winning(table, m: str) -> dict:
    """Rung *m* clears the gain clause on GSE and FF, nothing else moves."""
    return {(ds, m): {"gain": (0.01, 0.011, 0.012)} for ds in table.PRIMARY}


def test_the_smallest_passing_rung_is_adopted(table):
    changes = {**_winning(table, "m2"), **_winning(table, "m5")}
    ruling = table.decide(_cells(table, changes))
    assert ruling["per_rung"]["m1"]["status"] == "fail"      # no gain
    assert ruling["adopted_m"] == 2 and ruling["complete"]


def test_a_guard_failing_on_any_dataset_vetoes_the_rung(table):
    changes = {**_winning(table, "m2"), **_winning(table, "m5")}
    changes[(table.LU, "m2")] = {"cycle_w": (0.01, 0.03, 0.01)}
    ruling = table.decide(_cells(table, changes))
    assert ruling["per_rung"]["m2"]["guards"][table.LU]["cycle_w"] is False
    assert ruling["adopted_m"] == 5


def test_gain_must_beat_the_reference_by_its_seed_range(table):
    # +0.00025 over the reference mean, whose range is 0.0002: passes; the
    # same excess against a wider reference range does not
    changes = {(ds, "m1"): {"gain": (0.00045, 0.00055, 0.00065)}
               for ds in table.PRIMARY}
    assert table.decide(_cells(table, changes))["adopted_m"] == 1
    wide = dict(changes)
    for ds in table.PRIMARY:
        wide[(ds, table.REFERENCE)] = {"gain": (0.0, 0.0003, 0.0006)}
    ruling = table.decide(_cells(table, wide))
    assert ruling["adopted_m"] is None and "stays at 0.1" in ruling["verdict"]


def test_gain_on_lung_or_ovarian_alone_is_not_enough(table):
    changes = {(ds, "m1"): {"gain": (0.01, 0.011, 0.012)} for ds in (table.LU, table.OV)}
    changes[(table.GS, "m1")] = {"gain": (0.01, 0.011, 0.012)}
    assert table.decide(_cells(table, changes))["adopted_m"] is None


def test_an_incomplete_smaller_rung_leaves_the_rule_undecided(table, tmp_path):
    # m1 clears GSE's gain clause; its FF cell lacks a seed, so m1 is open
    changes = {**_winning(table, "m2"), (table.GS, "m1"): {"gain": (0.01, 0.011, 0.012)},
               (table.FF, "m1"): "missing"}
    ruling = table.decide(_cells(table, changes))
    assert ruling["per_rung"]["m1"]["status"] == "undecided"
    assert ruling["undecided"] and ruling["adopted_m"] is None
    assert not table.write_decision({'fresh': ruling}, {"datasets": {}}, tmp_path / "D.json")
    assert not (tmp_path / "D.json").exists()


def test_a_decided_rule_is_still_withheld_until_the_grid_is_complete(table, tmp_path):
    # m1 fails outright on GSE whatever FF's missing seed says, so m2 is the
    # answer already -- but the hand-off waits for every fit
    changes = {**_winning(table, "m2"), (table.FF, "m1"): "missing"}
    ruling = table.decide(_cells(table, changes))
    assert ruling["adopted_m"] == 2 and not ruling["complete"]
    assert not table.write_decision({'fresh': ruling}, {"datasets": {}}, tmp_path / "D.json")


def test_the_recon_clause_reads_the_seed_spread(table):
    # a rung whose seeds are all better than the reference but spread wider
    # than the reference band's width fails; a tight, shifted rung passes
    wide = {(table.OV, "m1"): {"recon": (-7.10, -7.00, -6.90)}}
    tight = {(table.OV, "m1"): {"recon": (-7.10, -7.099, -7.098)}}
    g_wide = table.decide(_cells(table, wide))["per_rung"]["m1"]["guards"][table.OV]
    g_tight = table.decide(_cells(table, tight))["per_rung"]["m1"]["guards"][table.OV]
    assert g_wide["recon_spread"] is False and g_tight["recon_spread"] is True


def _stage(table, ff_rungs=None):
    """The author's reduced scope: GSE and ovarian, seeds 0 1, every rung;
    with *ff_rungs*, stage 2 adds FF at the reference plus those rungs."""
    scope = {"seeds": [0, 1], "datasets": {table.GS: list(table.RUNGS),
                                           table.OV: list(table.RUNGS)}}
    if ff_rungs is not None:
        scope["datasets"][table.FF] = [table.REFERENCE] + ff_rungs
    table.set_scope(scope)


def test_stage_one_names_the_candidates_and_decides_nothing(table):
    _stage(table)
    changes = {(table.OV, "m1"): {"cycle_w": (0.01, 0.05)},
               (table.GS, "m2"): {"gain": (0.01, 0.011)}}
    ruling = table.decide(_cells(table, changes))
    assert ruling["per_rung"]["m1"]["status"] == "fail"
    assert ruling["stage2_candidates"]["guards_only"] == [2, 5]
    assert ruling["stage2_candidates"]["guards_and_gse_gain"] == [2, 5]
    assert ruling["adopted_m"] is None and "needs FF" in ruling["verdict"]
    assert not table.write_decision({'fresh': ruling}, {"datasets": {}}, Path("/nonexistent/D"))


def test_stage_two_decides_on_the_candidates(table, tmp_path):
    _stage(table, ff_rungs=["m2", "m5"])
    win = (0.01, 0.011)
    changes = {(table.OV, "m1"): {"cycle_w": (0.01, 0.05)},
               (table.GS, "m2"): {"gain": win}, (table.FF, "m2"): {"gain": win}}
    ruling = table.decide(_cells(table, changes))
    assert ruling["complete"] and ruling["adopted_m"] == 2
    ladder = {"convention": "alpha_z", "datasets": {
        ds: {"alpha_w": {f"m{m}": m * 0.001 for m in table.MULTIPLIERS},
             "lbar": 1000.0} for ds in (table.GS, table.LU, table.OV, table.FF)}}
    assert table.write_decision({'fresh': ruling}, ladder, tmp_path / "D.json")


def test_stage_two_with_both_candidates_failing_stays_open(table):
    _stage(table, ff_rungs=["m2", "m5"])
    # m10 clears GSE's gain clause, so only FF -- not run there -- decides it
    changes = {(table.OV, "m1"): {"cycle_w": (0.01, 0.05)},
               (table.FF, "m2"): {"cycle_w": (0.03, 0.03)},
               (table.FF, "m5"): {"cycle_w": (0.03, 0.03)},
               (table.GS, "m10"): {"gain": (0.01, 0.011)}}
    ruling = table.decide(_cells(table, changes))
    assert ruling["adopted_m"] is None and ruling["undecided"]
    assert "not run on FF" in ruling["verdict"]


def test_the_block_probe_replaces_the_legacy_probe_where_it_exists(table):
    # legacy pooled probe at its floor on every seed, but one seed's per-block
    # probe fails: the guard follows the per-block probe
    changes = {(table.OV, "m1"): {"probe_ok": (True, False, True),
                                  "probe_source": "blocks"},
               (table.GS, "m1"): {"gain": (0.01, 0.011, 0.012)},
               (table.FF, "m1"): {"gain": (0.01, 0.011, 0.012)}}
    ruling = table.decide(_cells(table, changes))
    assert ruling["per_rung"]["m1"]["guards"][table.OV]["probe"] is False
    assert ruling["per_rung"]["m1"]["guards"][table.LU]["probe"] is True
    assert ruling["adopted_m"] is None


def test_probe_blocks_are_read_from_the_run_directory(table, tmp_path):
    import json
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "probe_blocks.json").write_text(json.dumps(
        {"invariance_pass": False,
         "blocks": {"composition": {"excess": 0.01, "pass": True},
                    "image": {"ridge": {"excess": 0.2, "pass": False}}}}))
    got = table.probe_blocks(tmp_path)
    assert got["invariance_pass"] is False
    assert got["blocks"] == {"blocks/composition": {"excess": 0.01, "pass": True},
                             "blocks/image/ridge": {"excess": 0.2, "pass": False}}
    assert table.probe_blocks(tmp_path / "nothing") is None


def _with_reused(table, cells, ds, **over):
    """A reused reference pair on *ds* beside its fresh one."""
    runs = [_seed(i, **over) for i in table.SEEDS]
    cells[(ds, table.REUSED)] = {"alpha_w": 0.1, "runs": runs,
                                 "n_expected": len(runs), **table.summarise(runs)}
    return cells


def test_the_three_reference_envelopes_can_disagree(table):
    # fresh reference: wide w-mirror band; reused: tight; pooled in between
    cells = _cells(table, {(table.OV, table.REFERENCE): {"w_mirror": (0.13, 0.19, 0.16)},
                           (table.OV, "m1"): {"w_mirror": (0.145, 0.15, 0.148)},
                           (table.GS, "m1"): {"gain": (0.01, 0.011, 0.012)},
                           (table.FF, "m1"): {"gain": (0.01, 0.011, 0.012)}})
    _with_reused(table, cells, table.OV, w_mirror=(0.134, 0.136, 0.135))
    by = {v: table.decide(cells, v) for v in table.VARIANTS}
    wm = {v: by[v]["per_rung"]["m1"]["guards"][table.OV]["w_mirror"] for v in by}
    assert wm == {"fresh": True, "reused": False, "pooled": True}
    pooled = table.reference_variant(cells, table.OV, "pooled")
    assert pooled["n_expected"] == 6 and pooled["n_seeds"] == 6
    assert by["fresh"]["adopted_m"] == 1 and by["reused"]["adopted_m"] != 1


def test_a_dataset_with_one_kind_of_reference_uses_it_in_every_variant(table):
    cells = _cells(table, {})
    for v in table.VARIANTS:
        assert table.reference_variant(cells, table.GS, v) is cells[(table.GS, table.REFERENCE)]
    # only reused fits (FF: its aw_ref0.1 are links onto wfix): reused throughout
    cells[(table.FF, table.REFERENCE)] = {"alpha_w": 0.1, "runs": [], "n_expected": 0,
                                         **table.summarise([])}
    _with_reused(table, cells, table.FF)
    for v in table.VARIANTS:
        assert table.reference_variant(cells, table.FF, v) is cells[(table.FF, table.REUSED)]


def test_a_decision_needs_every_variant_to_agree_to_name_a_rung(table, tmp_path):
    import json
    cells = _cells(table, {(table.OV, table.REFERENCE): {"w_mirror": (0.13, 0.19, 0.16)},
                           (table.OV, "m1"): {"w_mirror": (0.145, 0.15, 0.148)},
                           (table.OV, "m2"): {"w_mirror": (0.12, 0.121, 0.122)},
                           **_winning(table, "m1"), **_winning(table, "m2")})
    _with_reused(table, cells, table.OV, w_mirror=(0.134, 0.136, 0.135))
    rulings = {v: table.decide(cells, v) for v in table.VARIANTS}
    ladder = {"convention": "alpha_z", "datasets": {
        ds: {"alpha_w": {f"m{m}": m * 0.001 for m in table.MULTIPLIERS},
             "lbar": 1000.0} for ds in (table.GS, table.LU, table.OV, table.FF)}}
    assert table.write_decision(rulings, ladder, tmp_path / "D.json")
    out = json.loads((tmp_path / "D.json").read_text())
    assert out["variants_agree"] is False and out["adopted_m"] is None
    assert out["by_reference"]["fresh"]["adopted_m"] == 1
    assert out["by_reference"]["reused"]["adopted_m"] == 2


def test_transport_reads_come_from_the_all_panel_tier_and_the_own_target(table, tmp_path):
    import json
    (tmp_path / "transport").mkdir()
    (tmp_path / "transport" / "transport.json").write_text(json.dumps(
        {"summary": {"extrapolation": {"n_panels": 8, "full_beats_both": 6,
                                       "counterfactual_of_ceiling": 0.4},
                     "extrapolation_trusted": {"n_panels": 2, "full_beats_both": 2,
                                               "counterfactual_of_ceiling": 0.9}}}))
    (tmp_path / "transport" / "transport_distribution.json").write_text(json.dumps(
        {"summary_model": {"pairwise": {"median_gap_closed": 0.9}},
         "summary_model_own": {"pairwise": {"median_gap_closed": 0.3}}}))
    got = table.transport_reads(tmp_path)
    assert got == {"readA_of_ceiling": 0.4, "readA_beats_both": 0.75,
                   "readA_own_gap": 0.3}
    assert table.transport_reads(tmp_path / "none") == {}
