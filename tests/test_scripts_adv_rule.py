"""The adversary-capacity ladder's pre-registered rule on planted read-outs.

``scripts/adv_table.py`` is loaded by path (scripts/ is not a package). Each
cell is two planted seeds; each test moves one clause of one arm.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "adv_table.py"


@pytest.fixture
def table(monkeypatch):
    spec = importlib.util.spec_from_file_location("adv_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "timing", lambda: {
        "comp3": {"head_ms_median": 5.0}, "steps12_width128": {"head_ms_median": 12.0}})
    return module


def _seed(i: int, **over) -> dict:
    run = {"run": f"s{i}", "dead_w_channel": False, "cycle_w_ok": True,
           "mlp_comp_u": [0.56, 0.44][i], "nmi": 0.64 + 0.01 * i,
           "cycle_z": 0.53, "mirror_r2": 0.037, "cycle_w": 0.013,
           "w_niche_mi_excess": 0.63 + 0.03 * i, "read_a_own_gap_closed": 0.72,
           "recon": -7.18 + 0.02 * i, "best_epoch": 60}
    run.update({k: v[i] if isinstance(v, tuple) else v for k, v in over.items()})
    return run


def _cells(table, **arms) -> dict:
    cells = {"control": table.summarise([_seed(0), _seed(1)])}
    for arm in table.ARMS:
        if arm != "control":
            cells[arm] = table.summarise([_seed(0, **arms.get(arm, {})),
                                          _seed(1, **arms.get(arm, {}))])
    return cells


FALLS = {"mlp_comp_u": (0.40, 0.30)}      # control 0.56 / 0.44, range 0.12


def test_equal_to_control_qualifies_nowhere(table):
    decision = table.decide(_cells(table))
    assert decision["adopted"] is None and decision["status"] == "complete"
    assert all(v["clauses"]["probe_falls"] is False for v in decision["arms"].values())


def test_falls_beyond_range_on_both_seeds_is_adopted(table):
    decision = table.decide(_cells(table, width128=FALLS))
    assert decision["adopted"] == "width128"
    assert decision["arms"]["width128"]["clauses"]["probe_margins"] == pytest.approx([0.16, 0.14])


def test_one_seed_inside_the_range_fails(table):
    decision = table.decide(_cells(table, width128={"mlp_comp_u": (0.40, 0.33)}))
    assert decision["arms"]["width128"]["clauses"]["probe_falls"] is False
    assert decision["adopted"] is None


def test_a_guard_outside_the_envelope_fails(table):
    # NMI control 0.64 / 0.65, sd 0.0071: envelope floor 0.6329
    decision = table.decide(_cells(table, width128={**FALLS, "nmi": 0.62}))
    verdict = decision["arms"]["width128"]
    assert verdict["clauses"]["guards_outside"] == ["nmi"] and not verdict["qualifies"]


def test_better_than_the_envelope_passes(table):
    decision = table.decide(_cells(table, width128={**FALLS, "nmi": 0.70,
                                                    "cycle_w": 0.001}))
    assert decision["adopted"] == "width128"


def test_recon_below_the_envelope_fails(table):
    decision = table.decide(_cells(table, width128={**FALLS, "recon": -7.30}))
    assert decision["arms"]["width128"]["clauses"]["recon_not_worse"] is False


def test_absolute_cycle_w_guard(table):
    decision = table.decide(_cells(table, width128={**FALLS, "cycle_w_ok": (True, False)}))
    assert decision["arms"]["width128"]["clauses"]["absolute_guards"] is False


def test_preference_order_then_head_time(table):
    both = table.decide(_cells(table, ens3=FALLS, steps12=FALLS, comp3=FALLS))
    assert both["qualifying"] == ["steps12", "ens3", "comp3"]
    assert both["adopted"] == "steps12"
    unnamed = table.decide(_cells(table, steps12_width128=FALLS, comp3=FALLS))
    assert unnamed["qualifying"] == ["comp3", "steps12_width128"]


def test_missing_seed_is_pending(table):
    cells = _cells(table, width128=FALLS)
    cells["steps12"] = table.summarise([_seed(0, **FALLS)])
    decision = table.decide(cells)
    assert decision["arms"]["steps12"]["qualifies"] is None
    assert decision["status"] == "incomplete" and "steps12" in decision["pending"]
