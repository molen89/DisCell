"""The 8.19 breakdown layer (``discell.experiments.breakdown``): pooling over
seeds, the Bonferroni level from m_s, and kappa* under the lean entry's rule
(devlog "8.19 revised to a lean version (author, 2026-09-29)")."""

from __future__ import annotations

import numpy as np
import pytest

from discell.experiments import breakdown as B
from discell.experiments.breakdown_draws import _weighted_median

GRID = B.GRID


def _entry(estimate, sd, n=4000, seed=0, method="percentile", c=1.0):
    rng = np.random.default_rng(seed)
    return {"estimate": float(estimate),
            "draws": estimate + sd * rng.standard_normal(n),
            "method": method, "c": c}


def _collected(means, sd=0.01, per_seed=None):
    """{kappa: {seed: {"m": entry}}} for one member whose seed mean at each
    grid point is *means*; *per_seed* offsets move single seeds."""
    out = {}
    for i, k in enumerate(GRID):
        out[k] = {}
        for s in range(3):
            off = (per_seed or {}).get((k, s), 0.0)
            out[k][s] = {"m": _entry(means[i] + off, sd, seed=100 * i + s)}
    return out


def _kstar(means, m_s=8, **kw):
    table = B.section_table("test", _collected(means, **kw), m_s=m_s)
    return table["members"]["m"]


def test_planted_breakdown_point():
    # a contrast that decays with kappa and loses its interval at 0.3
    m = _kstar([0.30, 0.25, 0.20, 0.12, 0.005, -0.05], sd=0.01)
    assert m["status"] == "breaks" and m["kappa_star"] == 0.3
    assert m["reason"] == "interval contains 0"
    assert m["sign"] == 1


def test_sign_flip_is_detected():
    # every interval excludes 0, but the sign turns at 0.2
    m = _kstar([0.3, 0.2, 0.1, -0.1, -0.2, -0.3], sd=0.005)
    assert m["status"] == "breaks" and m["kappa_star"] == 0.2
    assert m["reason"] == "a seed flips sign"


def test_single_seed_flip_breaks_even_when_the_pool_holds():
    # seed 2 alone crosses 0 at 0.05; the pooled interval stays clear of 0
    m = _kstar([0.3] * 6, sd=0.001,
               per_seed={(0.05, 2): -0.35, (0.05, 0): 0.2, (0.05, 1): 0.2})
    assert m["status"] == "breaks" and m["kappa_star"] == 0.05
    assert m["reason"] == "a seed flips sign"


def test_no_finding_at_kappa_zero():
    m = _kstar([0.001, 0.2, 0.2, 0.2, 0.2, 0.2], sd=0.05)
    assert m["status"] == "no finding" and m["kappa_star"] is None


def test_above_the_grid():
    m = _kstar([-0.5] * 6, sd=0.01)
    assert m["status"] == "above the grid" and m["sign"] == -1


def test_gap_is_not_survival():
    c = _collected([0.5] * 6)
    c[0.2] = {s: {} for s in range(3)}
    m = B.section_table("test", c, m_s=8)["members"]["m"]
    assert m["status"] == "gap at kappa = 0.2"


def test_bonferroni_level_from_m_s():
    assert B.level(8) == pytest.approx(1 - 0.05 / 8)
    assert B.level(9) == pytest.approx(1 - 0.05 / 9)
    assert B.FAMILY_SIZE == {"ovarian": 7, "lung": 7, "ff": 7, "gse": 6,
                             "gse_dual": 6}
    # a planted member whose interval excludes 0 at m = 1 but not at m = 50
    means = [0.02] * 6
    wide = B.section_table("t", _collected(means, sd=0.015), m_s=50)
    narrow = B.section_table("t", _collected(means, sd=0.015), m_s=1)
    assert narrow["members"]["m"]["status"] == "above the grid"
    assert wide["members"]["m"]["status"] == "no finding"
    lo50, hi50 = wide["members"]["m"]["trajectory"]["0"]["ci"]
    lo1, hi1 = narrow["members"]["m"]["trajectory"]["0"]["ci"]
    assert lo50 < lo1 and hi50 > hi1
    with pytest.raises(ValueError):
        B.level(0)


def test_pool_averages_draws_across_seeds():
    # three seeds, independent draws of sd 1: the pooled draws have sd 1/sqrt 3
    seeds = [_entry(v, 1.0, n=200_000, seed=s) for s, v in enumerate((1, 2, 3))]
    p = B.pool(seeds, alpha=0.05)
    assert p["estimate"] == pytest.approx(2.0)
    half = 0.5 * (p["ci"][1] - p["ci"][0])
    assert half == pytest.approx(1.96 / np.sqrt(3), rel=0.02)
    assert p["n_seeds"] == 3


def test_subsample_form_is_placed_on_the_point():
    seeds = [_entry(v, 1.0, n=100_000, seed=s, method="subsample", c=0.5)
             for s, v in enumerate((10.0, 10.0, 10.0))]
    for e in seeds:
        e["draws"] = e["draws"] + 3.0         # a shifted half-sample median
    p = B.pool(seeds, alpha=0.05)
    mid = 0.5 * (p["ci"][0] + p["ci"][1])
    assert mid == pytest.approx(10.0, abs=0.02)
    assert 0.5 * (p["ci"][1] - p["ci"][0]) == pytest.approx(
        0.5 * 1.96 / np.sqrt(3), rel=0.03)


def test_ranking_pairs_on_shared_draws():
    d = np.random.default_rng(0).normal(size=100)
    members = {"a": {"estimate": 2.0, "draws": 2 + d, "method": "percentile",
                     "c": 1.0},
               "b": {"estimate": 1.0, "draws": 1 + d, "method": "percentile",
                     "c": 1.0}}
    pair = B.ranking_pairs(members, ["a", "b"])["a - b"]
    assert pair["estimate"] == 1.0 and np.allclose(pair["draws"], 1.0)


def test_npz_round_trip(tmp_path):
    from discell.experiments.breakdown_draws import write_group

    path = tmp_path / "runs" / "r" / "breakdown" / "cycle.npz"
    write_group(path, {"x": (0.5, np.arange(5.0), "percentile", 1.0,
                             {"reproduces": True})}, {"run": "r"})
    got = B.read_draws(path)["x"]
    assert got["estimate"] == 0.5 and got["draws"].tolist() == list(range(5))


def test_weighted_median_is_the_resampled_median():
    rng = np.random.default_rng(1)
    for m in (5, 6, 31):
        v = rng.normal(size=m)
        w = rng.integers(0, 3, (40, m))
        ref = [np.median(np.repeat(v, wi)) if wi.sum() else np.nan for wi in w]
        assert np.allclose(_weighted_median(v, w), ref, equal_nan=True)


def test_a_point_some_seed_lacks_is_a_gap():
    c = _collected([0.5] * 6)
    del c[0.2][1]["m"]
    m = B.section_table("test", c, m_s=8)["members"]["m"]
    assert m["status"] == "gap at kappa = 0.2"


def test_marker_pairs_are_a_trajectory_not_a_family_member(tmp_path):
    """Author, 2026-09-30: the marker-pair contrast leaves the family (its
    draws on disk are not read) and is a trajectory readout from
    marker_pairs.json."""
    import json

    from discell.experiments.breakdown_draws import write_group

    run = tmp_path / "runs" / B.run_name(0.0, 0)
    for group, name in (("cycle", "cycle_asym_q90"),
                        ("marker", "marker_excl_minus_ctrl_dc")):
        write_group(B.draws_path(run, group),
                    {name: (0.5, np.arange(5.0), "percentile", 1.0, {})},
                    {"run": run.name})
    diff = {"exclusive": {"estimate": -0.002}, "control": {"estimate": 0.012}}
    (run / "marker_pairs.json").write_text(json.dumps(
        {"differences": {"decode_corrected-raw": diff}}))
    got = B.collect("ovarian", grid=(0.0,), seeds=(0,), root=tmp_path)
    assert set(got[0.0][0]) == {"cycle_asym_q90"}
    traj = B.trajectory_table("ovarian", grid=(0.0,), seeds=(0,),
                              root=tmp_path)
    e = traj["readouts"]["marker_excl_minus_ctrl_dc"]["0"]
    assert e["mean"] == pytest.approx(-0.014) and e["n"] == 1
    assert "marker_excl_minus_ctrl_dc" in traj["sources"]


def test_transport_cf_minus_program_is_a_trajectory(tmp_path):
    """Author, 2026-10-01: transport counterfactual - programme-only leaves
    the family (zero by construction at kappa = 0); its transport_mean.npz
    draws are not read, transport_cf_minus_leak still is, and the trajectory
    table reads it from transport.json's per-panel R^2."""
    import json

    from discell.experiments.breakdown_draws import write_group

    run = tmp_path / "runs" / B.run_name(0.05, 0)
    write_group(B.draws_path(run, "transport_mean"),
                {name: (0.5, np.arange(5.0), "subsample", 1.0, {})
                 for name in ("transport_cf_minus_program",
                              "transport_cf_minus_leak")},
                {"run": run.name})
    got = B.collect("lung", grid=(0.05,), seeds=(0,), root=tmp_path)
    assert set(got[0.05][0]) == {"transport_cf_minus_leak"}
    panels = [{"counterfactual": {"r2": 0.5}, "program_only": {"r2": 0.4},
               "leak_only": {"r2": 0.1}},
              {"counterfactual": {"r2": 0.3}, "program_only": {"r2": 0.3},
               "leak_only": {"r2": 0.0}}]
    (run / "transport").mkdir(parents=True)
    (run / "transport" / "transport.json").write_text(json.dumps(
        {"panels": panels}))
    traj = B.trajectory_table("lung", grid=(0.05,), seeds=(0,),
                              root=tmp_path)
    e = traj["readouts"]["transport_cf_minus_program"]["0.05"]
    assert e["mean"] == pytest.approx(0.05) and e["n"] == 1
    assert "transport_cf_minus_program" in traj["sources"]


def test_transport_members_use_all_panels():
    """Author, 2026-09-29: the transport members are scored on ALL panels,
    untrusted and supported-tier ones included."""
    from discell.experiments.breakdown_draws import (all_panel_contrasts,
                                                      select_panels)

    def panel(cf, prog, leak, trusted, overlap):
        return {"counterfactual": {"r2": cf}, "program_only": {"r2": prog},
                "leak_only": {"r2": leak}, "trusted": trusted,
                "overlap_flag": overlap}
    panels = [panel(0.5, 0.3, 0.1, True, True),
              panel(0.2, 0.1, 0.0, False, True),
              panel(0.4, 0.4, 0.2, False, False)]
    assert select_panels(panels) == panels
    got = all_panel_contrasts(panels)
    assert got["transport_cf_minus_program"] == pytest.approx(
        np.mean([0.5, 0.2, 0.4]) - np.mean([0.3, 0.1, 0.4]))
    assert got["transport_cf_minus_leak"] == pytest.approx(
        np.mean([0.5, 0.2, 0.4]) - np.mean([0.1, 0.0, 0.2]))
    assert all_panel_contrasts([]) == {}


def test_family_by_readout_availability_and_not_applicable(tmp_path):
    """Devlog 2026-10-01 ("Filling the breakdown figure's gaps"): the axis
    test joins lung and FF; the TMA sections carry it as not applicable (no
    tumour cells), with the reason in the table and the combined markdown;
    the dual reads every other headline member from its dual__ draws."""
    from discell.experiments.breakdown_draws import EVAL_GROUPS, write_group

    for sec in ("ovarian", "lung", "ff"):
        assert B.AXIS in B.FAMILY[sec] and sec not in B.NOT_APPLICABLE
    for sec in ("gse", "gse_dual"):
        assert B.AXIS not in B.FAMILY[sec]
        assert "tumour" in B.NOT_APPLICABLE[sec][B.AXIS]["reason"]
    assert set(B.FAMILY["gse_dual"]) == set(B.HEADLINE)
    assert set(B.DUAL_GROUPS) <= set(EVAL_GROUPS)
    # the dual's members come from dual__<eval>__<group>.npz only
    run = tmp_path / "runs" / B.run_name(0.0, 0)
    for group, name in (("cycle", "cycle_asym_q90"),
                        ("w_mi", "w_niche_mi_excess")):
        write_group(B.draws_path(run, group, B.GD),
                    {name: (0.5, np.arange(5.0), "percentile", 1.0, {})},
                    {"run": run.name})
    write_group(B.draws_path(run, "w_mi"),          # the core's own: not read
                {"w_niche_mi_excess": (9.0, np.arange(5.0), "percentile",
                                       1.0, {})}, {"run": run.name})
    got = B.collect("gse_dual", grid=(0.0,), seeds=(0,), root=tmp_path)
    assert got[0.0][0]["w_niche_mi_excess"]["estimate"] == 0.5
    assert set(got[0.0][0]) == {"cycle_asym_q90", "w_niche_mi_excess"}
    # the records and the combined markdown name the reason
    t_gse = B.section_table("gse_dual", got, grid=(0.0,), seeds=(0,))
    t_ov = B.section_table("ovarian", got, grid=(0.0,), seeds=(0,))
    assert t_gse["m_s"] == 6 and t_gse["not_applicable"] == {
        B.AXIS: B.NO_TUMOUR}
    assert t_ov["not_applicable"] == {} and B.AXIS in t_ov["family"]
    t_ov["members"][B.AXIS] = t_ov["members"]["cycle_asym_q90"]
    md = B.combined_markdown({"ovarian": t_ov, "gse_dual": t_gse})
    row = next(ln for ln in md.splitlines() if ln.startswith(f"| {B.AXIS}"))
    assert row.rstrip().endswith("| n/a |")
    assert B.NO_TUMOUR["reason"] in md
    assert B.NO_TUMOUR["reason"] in B.markdown(t_gse)
