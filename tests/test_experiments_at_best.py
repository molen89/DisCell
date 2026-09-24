"""The accepted checkpoint's battery is the history row at ``best["epoch"]`` (R26).

A tiny synthetic run directory: three evaluation records, the accepted one in
the middle, and a ``final`` block that -- as in every run trained before the
R26 fix -- describes the last evaluation, not the accepted one.
"""

from __future__ import annotations

import json

import pytest

from discell.experiments.at_best import battery_at_best, suffixed


def _record(epoch: int, recon: float, nmi: float, cycle_w: float) -> dict:
    """One evaluation, in the shape history.jsonl and metrics["final"] share."""
    return {"recon_val": recon, "nmi": nmi,
            "mirror": {"r2": 0.1 + epoch / 1000, "r2_permuted": 0.0},
            "probe": {"delta_ce": 0.002, "noise_floor": 0.003,
                      "baseline_ce": 1.0},
            "cycle": {"z": {"r2_pooled": 0.3}, "w": {"r2_pooled": cycle_w},
                      "linear_ref": {"r2_pooled": 0.4}},
            "degeneracy": {"mi_ratio": 0.8 - epoch / 1000,
                           "within_var_fraction": 0.7},
            "recon_gap": {"gap": 0.15},
            "kl_w_per_dim": [epoch * 1e-4, epoch * 2e-4]}


def _run(tmp_path, epochs=(4, 9, 14), best_epoch=9, extra_metrics=None):
    rows = {e: _record(e, -7.3 + e / 100, 0.6 + e / 1000, 0.001 * e)
            for e in (4, 9, 14)}
    with (tmp_path / "history.jsonl").open("w") as sink:
        for e in epochs:
            sink.write(json.dumps({"epoch": e, "step": 10 * e,
                                   "alpha_w_eff": 0.1, **rows[e]}) + "\n")
    best = rows[best_epoch]
    metrics = {"best": {"recon_val": best["recon_val"], "nmi": best["nmi"],
                        "epoch": best_epoch,
                        "degeneracy": best["degeneracy"]},
               "final": rows[14], **(extra_metrics or {})}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    return rows


def test_returns_the_history_row_at_the_best_epoch(tmp_path):
    rows = _run(tmp_path)
    block = battery_at_best(tmp_path)
    assert block["at_best"] is True
    assert block["epoch"] == 9
    # every key path the read-outs pull from `final`, at the accepted epoch
    assert block["recon_val"] == rows[9]["recon_val"]
    assert block["nmi"] == rows[9]["nmi"]
    assert block["mirror"]["r2"] == rows[9]["mirror"]["r2"]
    assert block["probe"]["delta_ce"] == rows[9]["probe"]["delta_ce"]
    assert block["probe"]["noise_floor"] == rows[9]["probe"]["noise_floor"]
    assert block["cycle"]["z"]["r2_pooled"] == rows[9]["cycle"]["z"]["r2_pooled"]
    assert block["cycle"]["w"]["r2_pooled"] == rows[9]["cycle"]["w"]["r2_pooled"]
    assert block["degeneracy"]["mi_ratio"] == rows[9]["degeneracy"]["mi_ratio"]
    assert sum(block["kl_w_per_dim"]) == pytest.approx(9 * 3e-4)
    # and not the last-epoch `final`
    assert block["cycle"]["w"]["r2_pooled"] != rows[14]["cycle"]["w"]["r2_pooled"]


def test_no_history_row_at_the_best_epoch_falls_back_to_final(tmp_path):
    rows = _run(tmp_path, epochs=(4, 14))
    block = battery_at_best(tmp_path)
    assert block["at_best"] is False
    assert "no history row" in block["at_best_reason"]
    assert block["cycle"]["w"]["r2_pooled"] == rows[14]["cycle"]["w"]["r2_pooled"]


def test_a_row_that_disagrees_with_best_is_not_taken_for_it(tmp_path):
    # the history of a later refit under the same name next to an older
    # metrics.json: same epoch number, different model
    _run(tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    metrics["best"]["recon_val"] += 0.01
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    block = battery_at_best(tmp_path)
    assert block["at_best"] is False
    assert "disagrees" in block["at_best_reason"]


def test_a_post_fix_final_is_already_at_best(tmp_path):
    # runs trained after the R26 fix re-evaluate best.pt for `final` and say
    # so in metrics["final_epoch"]; the fallback is then honest about it
    _run(tmp_path, epochs=(4, 14), extra_metrics={"final_epoch": 9})
    block = battery_at_best(tmp_path)
    assert block["at_best"] is True
    assert block["epoch"] == 9


def test_suffixed_only_when_reading_at_best(tmp_path):
    assert suffixed(tmp_path / "wcollapse", "final") == tmp_path / "wcollapse"
    assert suffixed(tmp_path / "wcollapse", "best") == tmp_path / "wcollapse_at_best"
    assert (suffixed(tmp_path / "DECISION.json", "best")
            == tmp_path / "DECISION_at_best.json")
