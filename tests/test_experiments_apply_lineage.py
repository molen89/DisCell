"""The lineage relabel (`discell/experiments/apply_lineage.py`) on tiny bundles.

A synthetic dataset root with a proposed map, a per-cell reassignment file and
two bundle variants: the column written in place must be exactly the mapping
(author choices and per-cell targets included), nothing else in the file may
change (``uns['default_label']`` above all), and every inconsistency the
real tables could carry -- a missing choice, an unmapped label, a per-cell
class with a cell missing, two sections with different vocabularies -- must
stop the write rather than produce a column.
"""

from __future__ import annotations

import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from discell.experiments import apply_lineage as al

CELLS = [f"c{i}" for i in range(8)]
SOURCE = ["Tumor Cells", "Proliferative Tumor Cells", "SOX2-OT+ Tumor Cells",
          "Malignant Cells Lining Cyst", "Macrophages", None, "EC activated",
          "EC activated"]
MAP = pd.DataFrame({
    "source_label": ["Tumor Cells", "Proliferative Tumor Cells",
                     "SOX2-OT+ Tumor Cells", "Malignant Cells Lining Cyst",
                     "Macrophages", "Unassigned", "EC activated"],
    "lineage": ["Tumour", "Tumour", "SOX2-OT+ Tumor Cells",
                "Mesothelial-like cyst lining", "Macrophages", "Unassigned",
                "per-cell: state_reassignment_proposed.csv (EC subtypes)"],
    "kind": ["lineage", "state", "mixed", "lineage", "lineage", "lineage", "state"],
})
REASSIGN = pd.DataFrame({"cell_id": ["c6", "c7"],
                         "source_label": ["EC activated", "EC activated"],
                         "assigned_lineage": ["EC venous", "Lymphatic EC"],
                         "score": [0.5, 0.4], "margin": [0.1, 0.2],
                         "runner_up": ["Lymphatic EC", "EC venous"]})


def _bundle(path, cells, source):
    obs = pd.DataFrame({"cell_group": pd.Categorical(source),
                        "area_um2": np.arange(len(cells), dtype=float)},
                       index=pd.Index(cells))
    adata = ad.AnnData(X=np.ones((len(cells), 3), dtype=np.float32), obs=obs)
    adata.uns["default_label"] = "cell_group"
    adata.uns["label_columns"] = ["cell_group"]
    adata.write_h5ad(path)


def _root(tmp_path, name="ds", mapping=MAP, reassign=REASSIGN):
    root = tmp_path / name
    (root / "labels").mkdir(parents=True)
    (root / "bundle").mkdir()
    mapping.to_csv(root / "labels" / al.MAP_FILE, index=False)
    if reassign is not None:
        reassign.to_csv(root / "labels" / "state_reassignment_proposed.csv", index=False)
    _bundle(root / "bundle" / "full.h5ad", CELLS, SOURCE)
    _bundle(root / "bundle" / "core.h5ad", CELLS[4:], SOURCE[4:])   # a subset variant
    return root


CHOICES = {"sox2ot": "unassigned", "cyst": "mesothelial"}
EXPECTED = ["Tumour", "Tumour", "Unassigned", "Mesothelial-like cyst lining",
            "Macrophages", "Unassigned", "EC venous", "Lymphatic EC"]


def test_column_is_the_mapping_and_nothing_else_changes(tmp_path):
    root = _root(tmp_path)
    before = ad.read_h5ad(root / "bundle" / "full.h5ad")
    p = al.plan("ds", root, CHOICES)
    assert p.source_key == "cell_group"
    assert p.vocab == sorted({"Tumour", "Unassigned", "Mesothelial-like cyst lining",
                              "Macrophages", "EC venous", "Lymphatic EC"})
    record = al.write(p, CHOICES, None)

    after = ad.read_h5ad(root / "bundle" / "full.h5ad")
    assert list(after.obs["lineage"].astype(str)) == EXPECTED
    assert list(after.obs["lineage"].cat.categories) == p.vocab
    assert after.uns["default_label"] == "cell_group"          # left alone
    assert list(after.uns["label_columns"]) == ["cell_group"]
    pd.testing.assert_frame_equal(after.obs.drop(columns="lineage"), before.obs)
    np.testing.assert_array_equal(after.X, before.X)
    core = ad.read_h5ad(root / "bundle" / "core.h5ad")
    assert list(core.obs["lineage"].astype(str)) == EXPECTED[4:]
    # the core carries the full vocabulary as categories, its own classes as values
    assert list(core.obs["lineage"].cat.categories) == p.vocab

    csv = root / "labels" / al.APPLIED_CSV
    assert record["applied_csv_sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    stored = json.loads((root / "labels" / al.APPLIED_JSON).read_text())
    assert stored["choices"] == CHOICES and stored["source_key"] == "cell_group"
    applied = pd.read_csv(csv)
    row = applied.set_index(["source_label", "lineage"])
    assert row.loc[("SOX2-OT+ Tumor Cells", "Unassigned"), "rule"] == "choice:sox2ot=unassigned"
    assert row.loc[("EC activated", "EC venous"), "n_full"] == 1
    assert row.loc[("EC activated", "Lymphatic EC"), "n_core"] == 1
    # NaN folds onto Unassigned exactly as the loader does
    assert row.loc[("Unassigned", "Unassigned"), "n_full"] == 1


def test_rewrite_is_idempotent_and_reports_no_change(tmp_path):
    root = _root(tmp_path)
    al.write(al.plan("ds", root, CHOICES), CHOICES, None)
    again = al.plan("ds", root, CHOICES)
    assert again.changed == {"core": 0, "full": 0}
    al.write(again, CHOICES, None)
    assert list(ad.read_h5ad(root / "bundle" / "full.h5ad").obs["lineage"].astype(str)) == EXPECTED


def test_the_other_choices(tmp_path):
    root = _root(tmp_path)
    p = al.plan("ds", root, {"sox2ot": "tumour", "cyst": "tumour"})
    assert list(p.columns["full"][:4]) == ["Tumour"] * 4
    assert "Mesothelial-like cyst lining" not in p.vocab


@pytest.mark.parametrize("missing", ["sox2ot", "cyst"])
def test_a_choice_class_without_its_choice_is_an_error(tmp_path, missing):
    root = _root(tmp_path)
    with pytest.raises(ValueError, match=f"--{missing}"):
        al.plan("ds", root, {**CHOICES, missing: None})


def test_unmapped_label_and_uncovered_per_cell_class_are_errors(tmp_path):
    root = _root(tmp_path, mapping=MAP[MAP["source_label"] != "Macrophages"])
    with pytest.raises(ValueError, match="missing from the map"):
        al.plan("ds", root, CHOICES)
    root = _root(tmp_path, name="ds2", reassign=REASSIGN.iloc[:1])
    with pytest.raises(ValueError, match="no reassignment"):
        al.plan("ds2", root, CHOICES)
    wrong = REASSIGN.assign(source_label=["EC activated", "Macrophages"])
    root = _root(tmp_path, name="ds3", reassign=wrong)
    with pytest.raises(ValueError, match="disagree"):
        al.plan("ds3", root, CHOICES)


def test_pair_check_catches_a_vocabulary_or_class_set_mismatch(tmp_path):
    a = al.plan("a", _root(tmp_path, "a"), CHOICES)
    assert al.check_pair(a, al.plan("b", _root(tmp_path, "b"), CHOICES)) == []
    other = REASSIGN.assign(assigned_lineage=["EC venous", "EC general capillary"])
    c = al.plan("c", _root(tmp_path, "c", reassign=other), CHOICES)
    problems = al.check_pair(a, c)
    assert any("vocabularies differ" in s for s in problems)
    assert any("observed classes differ" in s for s in problems)


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    from discell import paths

    root = _root(tmp_path)
    monkeypatch.setattr(paths, "DATASETS", tmp_path)
    stamp = (root / "bundle" / "full.h5ad").stat().st_mtime_ns
    assert al.main(["--dataset", "ds", "--sox2ot", "unassigned", "--cyst",
                    "mesothelial", "--dry-run"]) == 0
    assert "dry run: nothing written" in capsys.readouterr().out
    assert (root / "bundle" / "full.h5ad").stat().st_mtime_ns == stamp
    assert not (root / "labels" / al.APPLIED_CSV).exists()
    assert "lineage" not in ad.read_h5ad(root / "bundle" / "full.h5ad").obs
