"""`probe_regrade`'s reference options (lineage relabel, 2026-09-25).

At lineage labels every excess must be a fraction of the lineage-label
uncontrolled fits, never of the old-label ones; the baseline records of the
relabelled grade live in their own folder; and a scoped ``--reverdict`` must
leave every record outside its run patterns byte-for-byte untouched. The
defaults must stay what the queues already running rely on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from discell.experiments import probe_regrade as pr
from discell.model import metrics as M


def _record(strength: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n, k = 3000, 4
    t = rng.integers(0, k, n)
    y = np.stack([rng.dirichlet(np.ones(k)) for _ in t])
    comp = y[:, :-1]
    img = rng.standard_normal((k, 6))[t] + rng.standard_normal((n, 6))
    v = np.hstack([comp, img])
    vbar = np.stack([v[t == g].mean(axis=0) for g in range(k)])
    resid = v - vbar[t]
    z = np.hstack([resid @ rng.standard_normal((v.shape[1], 2)) * strength
                   + rng.standard_normal((n, 2)), rng.standard_normal((n, 3))])
    train = rng.random(n) < 0.7
    return M.probe_blocks(z, t, v, vbar, train, ~train, n_comp=comp.shape[1],
                          n_perm=2)


@pytest.fixture()
def section(tmp_path, monkeypatch):
    from discell import paths

    monkeypatch.setattr(paths, "DATASETS", tmp_path)
    runs = tmp_path / "ds" / "runs"

    def put(run: str, record: dict) -> None:
        path = runs / run / "validation" / "probe_blocks.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({**record, "run": run,
                                    "method": f"DisCell/{run}"}, default=float))

    put("uncontrolled500_s0", _record(10.0, 1))
    put("uncontrolledL_s0", _record(4.0, 2))
    put("uncontrolledL_s1", _record(4.0, 3))
    put("final_s0", _record(1.0, 4))
    put("finalL_s0", _record(1.0, 4))
    old = pr.baseline_path("ds", pr.RESOLVI)
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({**_record(6.0, 5), "run": None,
                               "method": pr.RESOLVI}, default=float))
    return runs


LINEAGE = pr.References(runs=("uncontrolledL_s0", "uncontrolledL_s1"),
                        run_200=None, baseline_dir="probe_regrade_lineage")


def test_defaults_are_the_old_references():
    assert pr.DEFAULT_REFS.runs == ("uncontrolled500_s0", "uncontrolled500_s1")
    assert pr.DEFAULT_REFS.run_200 == "uncontrolled_s0"
    assert pr.DEFAULT_REFS.baseline_dir == "probe_regrade"


def test_references_follow_the_override(section):
    found, resolvi, ref_200 = pr.references("ds", None, LINEAGE)
    assert sorted(r["run"] for r in found) == ["uncontrolledL_s0", "uncontrolledL_s1"]
    assert resolvi is None and ref_200 is None       # old-label records not read
    found, resolvi, _ = pr.references("ds", None)
    assert [r["run"] for r in found] == ["uncontrolled500_s0"]
    assert resolvi["method"] == pr.RESOLVI
    assert pr.baseline_path("ds", "SIMVI (x)", LINEAGE).parent.name == \
        "probe_regrade_lineage"


def test_scoped_reverdict_touches_only_its_runs(section):
    untouched = [section / r / "validation" / "probe_blocks.json"
                 for r in ("final_s0", "uncontrolled500_s0")]
    before = [p.read_bytes() for p in untouched]
    old_baseline = pr.baseline_path("ds", pr.RESOLVI).read_bytes()
    assert pr.main(["--dataset", "ds", "--reverdict", "--runs", "*L_s*",
                    "--references", "uncontrolledL_s0", "uncontrolledL_s1",
                    "--reference-200", "none", "--baseline-tag", "_lineage"]) == 0
    assert [p.read_bytes() for p in untouched] == before
    assert pr.baseline_path("ds", pr.RESOLVI).read_bytes() == old_baseline
    new = json.loads((section / "finalL_s0" / "validation"
                      / "probe_blocks.json").read_text())
    refs = [json.loads((section / r / "validation" / "probe_blocks.json").read_text())
            for r in ("uncontrolledL_s0", "uncontrolledL_s1")]
    mean = np.mean([r["ridge"]["comp"]["excess"] for r in refs])
    assert new["ridge"]["comp"]["fraction_of_uncontrolled"] == pytest.approx(
        new["ridge"]["comp"]["excess"] / mean)
    assert new["reference"]["uncontrolled"] == ["DisCell/uncontrolledL_s0",
                                                "DisCell/uncontrolledL_s1"]
    assert new["reference"]["uncontrolled200"] is None
