"""The top-decile cycle read in the trainer, and its re-read into old records.

One planted slide (synthetic counts, planted cycle scores, one class called
Unassigned) and one untrained model on CPU: ``Trainer.evaluate`` must carry
the top-decile block and its flat keys, and
``discell/experiments/cycle_reread.py`` must write exactly that block into a
run's ``metrics.json`` and exactly the battery's block into a battery column
-- after reproducing the stored label-derived read, and never over a record
that does not reproduce.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from discell.experiments import baseline_battery as B
from discell.experiments import cycle_reread as R
from discell.model import cell_cycle as CC

DS = "ds"


def _planted_data():
    import scipy.sparse as sp

    from discell.model.prepare import ModelData, spatial_tiles
    from discell.model.synthetic import simulate

    sim = simulate(n_cells=8000, n_types=4, kappa=0.1, seed=0)
    k, t = sim.n_types, sim.t.astype(np.int64)
    rng = np.random.default_rng(0)
    v = np.hstack([sim.graph.y[:, :-1], sim.phi[:, :4]]).astype(np.float32)
    tiles = spatial_tiles(sim.positions, 400)
    half = len(tiles) // 2
    return ModelData(
        graph=sim.graph, x=sp.csr_matrix(sim.x), t=t, phi=sim.phi,
        positions=sim.positions, totals=sim.totals,
        median_counts=float(np.median(sim.totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=np.array(["type0", "type1", "type2", "Unassigned"]),
        v_block=v, vbar_t=np.stack([v[t == g].mean(axis=0) for g in range(k)]),
        train_tiles=tiles[:half], val_tiles=tiles[half:],
        cycle={"s_score": rng.standard_normal(len(t)).astype(np.float32),
               "g2m_score": rng.standard_normal(len(t)).astype(np.float32),
               "cycling_types": [3, 0, 1, 2], "reliability": None,
               "x_pcs": rng.standard_normal((len(t), 5)).astype(np.float32)})


@pytest.fixture(scope="module")
def slide(tmp_path_factory):
    from discell import paths
    from discell.model.train import TrainConfig, Trainer

    root = tmp_path_factory.mktemp("ds")
    patch = pytest.MonkeyPatch()
    patch.setattr(paths, "dataset", lambda _: SimpleNamespace(root=root))
    data = _planted_data()
    config = TrainConfig(dataset=DS, run_name="r", kappa=0.1, d_z=6, d_w=2,
                         hidden=32, gat_dim=8, epochs=1, device="cpu",
                         alpha_z=0.007, v_pcs=4, invariance="closed_form")
    trainer = Trainer(config, data)
    report = trainer.evaluate()
    state = copy.deepcopy(trainer.model.state_dict())
    yield SimpleNamespace(root=root, data=data, config=config, report=report,
                          state=state)
    patch.undo()


def _trainer(slide):
    from discell.model.train import Trainer

    trainer = Trainer(slide.config, slide.data)
    trainer.model.load_state_dict(slide.state)
    trainer.model.eval()
    return trainer


def _roundtrip(obj):
    return json.loads(json.dumps(obj, default=str))


def _strip(block):
    return {k: v for k, v in block.items() if k != "reread"}


def test_the_trainer_reads_the_decile_set_of_non_unassigned_held_out_cells(slide):
    report, data = slide.report, slide.data
    block = report["cycle_q90"]
    held = np.zeros(data.n_cells, dtype=bool)
    held[np.concatenate(data.val_tiles)] = True
    assert block["n_heldout"] == round(0.1 * (held & (data.t != 3)).sum()) >= 200
    assert block["n_train"] >= 200
    assert set(block) >= {"z", "w", "linear_ref", "lbaseline"}
    assert np.isfinite(report["cycle_r2_z_q90"])
    assert report["cycle_r2_z_q90"] == block["z"]["r2_pooled"]
    assert report["cycle_r2_w_q90"] == block["w"]["r2_pooled"]
    assert report["cycle_linear_q90"] == block["linear_ref"]["r2_pooled"]
    assert report["cycle"]["types"] == [0, 1, 2]    # the retired read, kept
    # the arrays evaluate collected give the same block (the re-read's path)
    col = report["collected"]
    n_train = sum(len(tile) for tile in data.train_tiles)
    train = np.arange(len(col["rows"])) < n_train
    assert CC.trainer_cycle_q90(data, col["rows"], col["z"], col["w"], train,
                                seed=slide.config.seed) == block


def _run_on_disk(slide, run: str):
    """A finished run as the re-read finds it: config.json, a post-R26
    metrics.json without the new keys, and its DisCell battery column
    without cycle_q90."""
    from discell.model.validate import collect_latents

    run_dir = slide.root / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {**dataclasses.asdict(slide.config), "git": "x"}))
    final = {k: v for k, v in slide.report.items()
             if k not in ("collected", "cycle_q90") and not k.endswith("_q90")}
    (run_dir / "metrics.json").write_text(json.dumps(
        {"best": {"epoch": 4}, "final": final, "final_epoch": 4},
        default=float))
    trainer = _trainer(slide)
    latents = collect_latents(trainer, slide.data)
    data = slide.data
    test = np.zeros(data.n_cells, dtype=bool)
    test[np.concatenate(data.val_tiles)] = True
    cycle = B.cycle_inputs(data)
    c = np.random.default_rng(1).standard_normal((data.n_cells, 3))
    old = B.battery(latents["mu_z"], data.t, ~test, test, data.v_block,
                    data.vbar_t, c, spatial=latents["mu_w"],
                    cycle={k: cycle[k] for k in ("types", "scores")})
    new = B.battery(latents["mu_z"], data.t, ~test, test, data.v_block,
                    data.vbar_t, c, spatial=latents["mu_w"], cycle=cycle)
    exp = slide.root / "experiments"
    exp.mkdir(exist_ok=True)
    js = exp / "baseline_battery_lineage.json"
    entries = json.loads(js.read_text()) if js.exists() else {}
    entries[f"DisCell/{run}"] = {**old, "config": {"trained_on": DS}}
    js.write_text(json.dumps(entries, default=str))
    return run_dir, js, new["cycle_q90"]


def _patch_io(monkeypatch, slide, run_dir):
    from discell.model import prepare, validate

    monkeypatch.setattr(prepare, "assemble", lambda *a, **k: slide.data)
    monkeypatch.setattr(validate, "load_run", lambda ds, run, device, data=None: (
        slide.config, slide.data, _trainer(slide), run_dir, None))
    monkeypatch.setattr(R, "TABLES_LOCK", slide.root / "tables.lock")


def test_reread_writes_the_trainers_block_and_the_batterys_block(slide,
                                                                 monkeypatch):
    run_dir, js, battery_block = _run_on_disk(slide, "r1")
    _patch_io(monkeypatch, slide, run_dir)
    done = R.reread_run(DS, "r1", None, "cpu", force=False, cache={})
    assert set(done) == {"run", "metrics", "battery_lineage"}

    final = json.loads((run_dir / "metrics.json").read_text())["final"]
    assert _strip(final["cycle_q90"]) == _roundtrip(slide.report["cycle_q90"])
    for key in ("cycle_r2_z_q90", "cycle_r2_w_q90", "cycle_linear_q90"):
        assert final[key] == slide.report[key]
    assert max(final["cycle_q90"]["reread"]["stored_label_set_abs_diff"]
               .values()) < 1e-9
    entry = json.loads(js.read_text())["DisCell/r1"]
    assert _strip(entry["cycle_q90"]) == _roundtrip(battery_block)
    assert "top-decile set" in js.with_suffix(".md").read_text()

    # idempotent: nothing left to do, nothing loaded
    again = R.reread_run(DS, "r1", None, "cpu", force=False, cache={})
    assert again == {"run": "r1", "skipped": True}


def test_reread_refuses_a_record_that_does_not_reproduce(slide, monkeypatch):
    run_dir, js, _ = _run_on_disk(slide, "r2")
    _patch_io(monkeypatch, slide, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    metrics["final"]["cycle"]["z"]["r2_pooled"] += 0.1
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="not reproduced"):
        R.reread_run(DS, "r2", None, "cpu", force=False, cache={})
    assert "cycle_q90" not in json.loads(
        (run_dir / "metrics.json").read_text())["final"]


def test_reread_of_a_baseline_column_uses_the_slides_set_on_its_rows(
        slide, monkeypatch, tmp_path):
    data = slide.data
    rng = np.random.default_rng(2)
    rows = np.sort(rng.choice(data.n_cells, 6000, replace=False))
    z = rng.standard_normal((len(rows), 5))
    spatial = rng.standard_normal((len(rows), 2))
    test = np.zeros(data.n_cells, dtype=bool)
    test[np.concatenate(data.val_tiles)] = True
    cycle = B.cycle_inputs(data)
    c = rng.standard_normal((data.n_cells, 3))
    args = (z, data.t[rows], ~test[rows], test[rows], data.v_block[rows],
            data.vbar_t, c[rows])
    old = B.battery(*args, spatial=spatial, cycle={
        "types": cycle["types"], "scores": cycle["scores"][rows]})
    new = B.battery(*args, spatial=spatial, cycle={
        "types": cycle["types"], "scores": cycle["scores"][rows],
        "cells": cycle["cells"][rows], "x_pcs": cycle["x_pcs"][rows]})
    exp = tmp_path / "experiments"
    exp.mkdir()
    js = exp / "baseline_battery_lineage.json"
    js.write_text(json.dumps({"resolVI": old, "Unknown tool": old},
                             default=str))
    (tmp_path / "rv").mkdir()
    (tmp_path / "rv" / "latents.h5ad").touch()

    from discell import paths

    monkeypatch.setattr(paths, "dataset", lambda _: SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(R, "BRES", tmp_path)
    monkeypatch.setattr(R, "LATENTS", {DS: {"resolVI": "rv"}})
    monkeypatch.setattr(R, "TABLES_LOCK", tmp_path / "tables.lock")
    monkeypatch.setattr(B, "run_config", lambda *a: {"seed": 0})
    monkeypatch.setattr(B, "shared_side", lambda *a: (data, ~test, test, cycle))
    monkeypatch.setattr(B, "load_latents", lambda *a: (rows, z, spatial, {}))
    out = R.reread_baselines(DS, None, force=False)
    assert out["_lineage|Unknown tool"] == "no latents"
    entries = json.loads(js.read_text())
    assert _strip(entries["resolVI"]["cycle_q90"]) == _roundtrip(new["cycle_q90"])
    assert "cycle_q90" not in entries["Unknown tool"]


def test_reread_writes_the_held_out_section_records(slide, monkeypatch,
                                                    tmp_path):
    """--eval-dataset: the crossslide record's held_out_section and the
    section's battery DisCell column (here the section is the same planted
    slide, so the blocks must equal the trainer's and the battery's)."""
    from discell import paths

    run_dir, _, battery_block = _run_on_disk(slide, "r3")
    _patch_io(monkeypatch, slide, run_dir)
    roots = {DS: slide.root, "E": tmp_path}
    monkeypatch.setattr(paths, "dataset",
                        lambda name: SimpleNamespace(root=roots[name]))
    held = {k: v for k, v in slide.report.items()
            if k not in ("collected", "cycle_q90") and not k.endswith("_q90")}
    (run_dir / "crossslide").mkdir()
    cross = run_dir / "crossslide" / "E.json"
    cross.write_text(json.dumps({"held_out_section": held}, default=float))
    (tmp_path / "experiments").mkdir()
    eval_js = tmp_path / "experiments" / "baseline_battery_lineage.json"
    eval_js.write_text((slide.root / "experiments"
                        / "baseline_battery_lineage.json").read_text())

    done = R.reread_run(DS, "r3", "E", "cpu", force=False, cache={})
    assert {"metrics", "battery_lineage", "crossslide",
            "eval_battery_lineage"} <= set(done)
    section = json.loads(cross.read_text())["held_out_section"]
    assert _strip(section["cycle_q90"]) == _roundtrip(slide.report["cycle_q90"])
    assert section["cycle_r2_z_q90"] == slide.report["cycle_r2_z_q90"]
    entry = json.loads(eval_js.read_text())["DisCell/r3"]
    assert _strip(entry["cycle_q90"]) == _roundtrip(battery_block)
