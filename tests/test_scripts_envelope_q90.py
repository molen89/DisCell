"""Envelope tables: the top-decile cycle rows and where they are read from.

``scripts/envelope_tables.py`` is loaded by path (scripts/ is not a package).
A run trained after the R26 fix whose history row at the best epoch predates
the top-decile read takes it from ``metrics.json["final"]`` (the re-read of
``best.pt``); a pre-R26 run, whose ``final`` is another checkpoint, does not.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "envelope_tables.py"
Q90 = {"cycle_r2_z_q90": 0.41, "cycle_r2_w_q90": 0.012, "cycle_linear_q90": 0.33}


@pytest.fixture
def env(monkeypatch, tmp_path):
    # these runs carry no masked re-read (degeneracy.json battery): the
    # pre-rule reads, under the evaluation mask's switch
    monkeypatch.setenv("DISCELL_EVAL_INCLUDE_UNASSIGNED", "1")
    spec = importlib.util.spec_from_file_location("envelope_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.paths, "dataset",
                        lambda _: SimpleNamespace(root=tmp_path))
    return module


def _run(root: Path, name: str, final_epoch: int) -> None:
    d = root / "runs" / name
    d.mkdir(parents=True)
    cycle = {"z": {"r2_pooled": 0.6}, "w": {"r2_pooled": 0.02},
             "linear_ref": {"r2_pooled": 0.5}}
    best = {"epoch": 10, "recon_val": -1.0, "nmi": 0.7}
    (d / "metrics.json").write_text(json.dumps({
        "best": best, "final_epoch": final_epoch,
        "final": {"recon_val": -1.0, "nmi": 0.7, "cycle": cycle, **Q90}}))
    (d / "history.jsonl").write_text(json.dumps(
        {"epoch": 10, "recon_val": -1.0, "nmi": 0.7, "cycle": cycle}) + "\n")


def test_q90_rows_are_read_at_best_from_the_re_read_final(env, tmp_path):
    _run(tmp_path, "post", final_epoch=10)
    _run(tmp_path, "pre", final_epoch=25)
    post = env.run_record("ds", "post", at="best")
    assert (post["cycle_z_q90"], post["cycle_w_q90"], post["cycle_ref_q90"]) \
        == (0.41, 0.012, 0.33)
    assert post["cycle_z"] == 0.6                    # the retired row, kept
    pre = env.run_record("ds", "pre", at="best")
    assert pre["cycle_z_q90"] is None                # final is not best.pt
    assert env.run_record("ds", "pre", at="final")["cycle_z_q90"] == 0.41

    table = env.render("ds", [post])
    assert "| cycle R² (z, top-decile set) | 0.4100 |" in table
    assert "| cycle R² (z, pooled; label-derived set, retired) | 0.6000 |" in table
    assert "cycle_ref_q90" in env.HELD_OUT_ROWS
