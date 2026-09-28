"""``scripts/archive_2026-09-25.py --apply`` on a planted data tree.

Loaded by path (scripts/ is not a package), with its ``DATA`` pointed at a
throwaway tree: a covered archive family moves, an aliased pair moves
together and still resolves, kept families and ``runs/best`` stay, an
uncovered run stays, and a kept link to an archived target refuses the
whole apply before anything moves.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "archive_2026-09-25.py"


@pytest.fixture
def archive(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("archive_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DATA", tmp_path / "data" / "datasets")
    monkeypatch.setattr(module, "BASELINES", tmp_path / "none")
    return module


def _run(runs: Path, name: str) -> None:
    (runs / name).mkdir(parents=True)
    (runs / name / "metrics.json").write_text("{}")
    (runs / name / "best.pt").write_bytes(b"x" * 1000)


def _tree(root: Path) -> Path:
    runs = root / "data" / "datasets" / "planted" / "runs"
    for name in ("finalL_s0", "sweepL_k0_s0", "uncontrolledL_s0", "final_s0",
                 "best_az0.5_s0", "sweep3_k0_s0", "dw2_s0", "reference_x"):
        _run(runs, name)
    (runs / "best").symlink_to("finalL_s0")
    (runs / "sweepL_k0.1_s0").symlink_to("finalL_s0")
    (runs / "best_s0").symlink_to("best_az0.5_s0")
    exp = runs.parent / "experiments"
    exp.mkdir()
    (exp / "envelope_table_ci.md").write_text("final_s0\n")
    (exp / "envelope_table.md").write_text("best_s0\n")
    return runs


def test_apply_moves_covered_families_and_keeps_the_rest(archive, tmp_path, capsys):
    runs = _tree(tmp_path)
    assert archive.main(["--apply"]) == 0
    moved = runs / "_archive"
    assert {p.name for p in moved.iterdir()} == {
        "final_s0", "best_az0.5_s0", "best_s0", archive.ARCHIVE_TSV}
    assert (moved / "best_s0" / "best.pt").exists()          # link resolves
    for kept in ("finalL_s0", "sweepL_k0_s0", "sweepL_k0.1_s0", "uncontrolledL_s0",
                 "dw2_s0", "reference_x", "sweep3_k0_s0"):   # sweep3: no table
        assert (runs / kept).exists()
    assert (runs / "best" / "best.pt").exists()
    assert (runs.parent / "experiments" / "envelope_table_ci.md").exists()
    rows = (moved / archive.ARCHIVE_TSV).read_text().splitlines()
    assert rows[0] == "run\tsize_bytes\tfamily\ttables"
    assert any(r.startswith("final_s0\t") and "envelope_table_ci.md" in r for r in rows)
    assert "Verified" in capsys.readouterr().out


def test_apply_refuses_when_a_kept_link_would_dangle(archive, tmp_path):
    runs = _tree(tmp_path)
    (runs / "best").unlink()
    (runs / "best").symlink_to("final_s0")                   # best -> archived
    with pytest.raises(SystemExit, match="REFUSED"):
        archive.main(["--apply"])
    assert not (runs / "_archive").exists()
    assert (runs / "final_s0").is_dir()
