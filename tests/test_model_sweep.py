"""The generalised sweep CLI: grids, run names, idempotence, aggregation.

No fit runs here -- these are the pure seams around ``fit_grid``/``report``
that decide which runs exist, what they are called, and how the finished ones
are read back. The naming test is the compatibility contract with the sweep3
and dw* runs already on disk.
"""

from __future__ import annotations


import numpy as np
import pytest

from discell.model import sweep


def test_default_grids_are_the_preregistered_ones():
    for param, expected in (("kappa", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]),
                            ("d_w", [2, 3, 6, 8]),
                            ("alpha_w", [0.03, 0.05, 0.07, 0.1, 0.2, 0.3])):
        args = sweep.parse_args(["--dataset", "ds", "--param", param])
        assert args.values == expected
        assert args.seeds == [0, 1, 2]
    # d_w is an int knob even when the CLI hands it floats
    args = sweep.parse_args(["--dataset", "ds", "--param", "d_w",
                             "--values", "2", "8"])
    assert args.values == [2, 8] and all(isinstance(v, int) for v in args.values)


def test_per_dataset_knobs_reach_the_fit_config():
    args = sweep.parse_args([
        "--dataset", "gse315411_pdltma06_11_prime_solo", "--variant",
        "pdl018d", "--param", "alpha_w", "--alpha-z", "0.0036",
        "--tile-cells", "2048", "--label-key", "graphclust",
        "--epochs", "300", "--patience", "30"])
    config = sweep.base_config(args, "aw0.05_s1", 1, 0.05)
    assert (config.variant, config.alpha_z, config.tile_cells) == (
        "pdl018d", 0.0036, 2048)
    assert (config.label_key, config.epochs, config.patience) == (
        "graphclust", 300, 30)
    assert config.alpha_w == 0.05 and config.seed == 1   # the swept knob wins
    assert config.kappa == 0.1 and config.d_w == 6       # the fixed ones hold


def test_run_names_match_the_runs_already_on_disk():
    assert sweep.run_name(0.1, 0, "sweep3") == "sweep3_k0.1_s0"
    assert sweep.run_name(0.0, 2, "sweep3") == "sweep3_k0_s2"
    assert sweep.run_name(2, 1, "", "d_w") == "dw2_s1"
    assert sweep.run_name(0.05, 0, "", "alpha_w") == "aw0.05_s0"


def _args(tmp_path, **over):
    argv = ["--dataset", "ds", "--seeds", "0", "1", "--values", "0.1", "0.2"]
    for key, value in over.items():
        argv += [f"--{key.replace('_', '-')}"] + (
            [] if value is True else [str(value)])
    return sweep.parse_args(argv)


def test_finished_runs_are_skipped_unless_forced(tmp_path):
    (tmp_path / "sweep_k0.1_s0").mkdir()
    (tmp_path / "sweep_k0.1_s0" / "metrics.json").write_text("{}")
    (tmp_path / "sweep_k0.2_s1").mkdir()          # started, never finished
    plan = sweep.planned_fits(_args(tmp_path), tmp_path)
    assert [name for name, _, _ in plan] == [
        "sweep_k0.2_s0", "sweep_k0.1_s1", "sweep_k0.2_s1"]
    assert plan[0][1:] == (0.2, 0)
    forced = sweep.planned_fits(_args(tmp_path, force=True), tmp_path)
    assert len(forced) == 4


def test_metric_row_reads_the_battery_and_tolerates_old_runs():
    full = {"best": {"recon_val": -7.25, "nmi": 0.66, "epoch": 59},
            "final": {"cycle": {"z": {"r2_pooled": 0.45, "r2_mean_types": 0.12},
                                "w": {"r2_pooled": 0.006},
                                "linear_ref": {"r2_pooled": 0.21}},
                      "mirror": {"r2": 0.048}, "probe": {"delta_ce": 0.006},
                      "degeneracy": {"tau": 0.12}, "recon_gap": 0.03}}
    row = sweep.metric_row("alpha_w", 0.05, 1, full)
    assert row["alpha_w"] == 0.05 and row["value"] == 0.05 and row["seed"] == 1
    assert (row["recon"], row["nmi"], row["epoch"]) == (-7.25, 0.66, 59)
    assert row["cycle_r2_z"] == 0.45 and row["cycle_r2_linear_ref"] == 0.21
    assert row["cycle_r2_z_mean_types"] == 0.12   # the retired statistic rides along
    assert row["degeneracy"] == {"tau": 0.12} and row["recon_gap"] == 0.03

    old = {"best": full["best"],
           "final": {"cycle": {"ceiling": {"r2_mean_types": 0.23}}}}
    row = sweep.metric_row("kappa", 0.1, 0, old)
    assert row["cycle_r2_linear_ref"] == 0.23       # pre-rename key
    assert row["mirror_r2"] is None and row["degeneracy"] is None
    assert row["cycle_r2_z"] is None and row["recon_gap"] is None

    # the degeneracy metrics may land at the top level instead of under final
    row = sweep.metric_row("kappa", 0.1, 0,
                           {"best": full["best"], "final": {},
                            "degeneracy": {"tau": 0.4}})
    assert row["degeneracy"] == {"tau": 0.4}


def test_b_stability_across_seeds_and_along_the_value_axis():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 3))
    loaded = {(0.1, 0): {"B": base}, (0.1, 1): {"B": base[:, ::-1]},
              (0.2, 0): {"B": rng.normal(size=(40, 3))}}
    out = sweep.b_stability(loaded, [0.1, 0.2], [0, 1], "kappa")
    # a column permutation is the same programme space: matching undoes it
    assert out["across_seeds"]["0.1"]["min"] == pytest.approx(1.0, abs=1e-6)
    assert "0.2" not in out["across_seeds"]       # one seed: no pair to make
    assert set(out["along_kappa"]) == {"k0.1_s0", "k0.1_s1", "k0.2_s0"}
    assert out["along_kappa"]["k0.1_s0"] == pytest.approx(1.0, abs=1e-6)
    assert out["along_kappa"]["k0.2_s0"] < 0.6    # unrelated B


def test_b_stability_drops_incomparable_shapes_in_a_d_w_sweep():
    rng = np.random.default_rng(1)
    loaded = {(2, 0): {"B": rng.normal(size=(40, 2))},
              (6, 0): {"B": rng.normal(size=(40, 6))}}
    out = sweep.b_stability(loaded, [2, 6], [0], "d_w")
    assert set(out["along_d_w"]) == {"dw2_s0"}    # d_w = 6 is not comparable
    assert out["across_seeds"] == {}              # one seed per value


def test_report_filename_keeps_the_cited_sweep3_name():
    assert sweep.report_filename("kappa", "sweep3") == "kappa_sweep_sweep3.json"
    assert sweep.report_filename("kappa", "sweep") == "kappa_sweep.json"
    assert sweep.report_filename("d_w", "") == "d_w_sweep_untagged.json"
