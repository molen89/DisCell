"""The composition-weight confirmation's pre-stated rule on planted read-outs.

``scripts/adv_confirm_table.py`` is loaded by path (scripts/ is not a
package). Controls are planted per dataset; each test moves one clause.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "adv_confirm_table.py"


@pytest.fixture
def table():
    spec = importlib.util.spec_from_file_location("adv_confirm_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# control seeds: MLP comp mean 0.50, ridge comp mean 0.40 sd 0.05,
# NMI 0.64 +- 0.01, cycle_z 0.53 +- 0.01, recon -7.18 +- 0.02
CONTROL = [dict(mlp_comp_u=0.55, ridge_comp_u=0.45, nmi=0.65, cycle_z=0.54, recon=-7.16),
           dict(mlp_comp_u=0.45, ridge_comp_u=0.35, nmi=0.63, cycle_z=0.52, recon=-7.20)]
GOOD = [dict(mlp_comp_u=0.45, ridge_comp_u=0.30, nmi=0.64, cycle_z=0.53, recon=-7.18),
        dict(mlp_comp_u=0.40, ridge_comp_u=0.28, nmi=0.64, cycle_z=0.53, recon=-7.18)]


def _run(i: int, values: dict) -> dict:
    return {"run": f"r{i}", "dead_w_channel": False, "cycle_w_ok": True, **values}


def _cell(table, seeds: list[dict], expected: int | None = None) -> dict:
    out = table.summarise([_run(i, v) for i, v in enumerate(seeds)])
    out["expected"] = expected or len(seeds)
    out["missing"] = [f"r{i}" for i in range(len(seeds), out["expected"])]
    return out


def _cells(table, comp3=None, **over) -> dict:
    """Every dataset: planted control, comp3 = GOOD unless overridden."""
    cells = {}
    for ds, arms in table.ARMS.items():
        cells[ds] = {"control": _cell(table, CONTROL)}
        arm = over.get(ds, comp3 or GOOD)
        cells[ds]["comp3"] = None if arm is None else _cell(
            table, arm, expected=over.get(f"{ds}_expected"))
        if "comp5" in arms:
            cells[ds]["comp5"] = _cell(table, GOOD)
    return cells


def test_good_arm_on_every_dataset_is_confirmed(table):
    decision = table.decide(_cells(table))
    assert decision["confirmed"] is True and decision["status"] == "complete"
    assert decision["comp5_dose"]["passes_same_rule"] is True


def test_equal_to_control_is_not_confirmed(table):
    decision = table.decide(_cells(table, comp3=CONTROL))
    assert decision["confirmed"] is False
    assert all(v["clauses"]["mlp_below_mean"] is False
               for v in decision["comp3"].values())


def test_one_seed_at_the_control_mean_fails_a(table):
    arm = [dict(GOOD[0], mlp_comp_u=0.50), GOOD[1]]      # not strictly below
    decision = table.decide(_cells(table, **{table.FF: arm}))
    v = decision["comp3"][table.FF]
    assert v["clauses"]["mlp_below_mean"] is False and v["passes"] is False
    assert decision["confirmed"] is False
    assert decision["comp3"][table.GS]["passes"] is True


def test_ridge_within_one_control_sd_fails_b(table):
    # control ridge mean 0.40, sd 0.0707: mean 0.34 is below by 0.06 only
    arm = [dict(g, ridge_comp_u=0.34) for g in GOOD]
    v = table.decide(_cells(table, **{table.OV: arm}))["comp3"][table.OV]
    assert v["clauses"]["ridge_below_sd"] is False
    assert v["clauses"]["ridge_margin"] == pytest.approx(0.06)
    assert v["clauses"]["ridge_control_sd"] == pytest.approx(0.0707, abs=1e-4)


def test_guard_below_the_widened_envelope_fails_c(table):
    # cycle_z envelope [0.52 - 0.0141, 0.54 + 0.0141]
    arm = [dict(g, cycle_z=0.50) for g in GOOD]
    v = table.decide(_cells(table, **{table.GS: arm}))["comp3"][table.GS]
    assert v["clauses"]["guards_inside"] is False
    assert v["clauses"]["guards_outside"][0].startswith("cycle_z")
    inside = [dict(g, cycle_z=0.51) for g in GOOD]
    v = table.decide(_cells(table, **{table.GS: inside}))["comp3"][table.GS]
    assert v["clauses"]["guards_inside"] is True


def test_guard_above_the_envelope_passes_and_is_named(table):
    arm = [dict(g, nmi=0.70) for g in GOOD]
    v = table.decide(_cells(table, **{table.GS: arm}))["comp3"][table.GS]
    assert v["passes"] is True and v["clauses"]["guards_above_envelope"] == ["nmi"]


def test_missing_fit_is_pending_but_a_complete_failure_decides(table):
    decision = table.decide(_cells(table, **{table.OV: GOOD, f"{table.OV}_expected": 3}))
    assert decision["comp3"][table.OV]["passes"] is None
    assert decision["confirmed"] is None and decision["status"] == "incomplete"
    decision = table.decide(_cells(table, **{table.OV: GOOD, f"{table.OV}_expected": 3,
                                             table.FF: CONTROL}))
    assert decision["confirmed"] is False and decision["status"] == "incomplete"
    decision = table.decide(_cells(table, **{table.FF: None}))
    assert decision["comp3"][table.FF]["passes"] is None
