"""The preprocessing entry point: what it writes, and what it skips."""

from __future__ import annotations

import json


def test_bundle_stage_writes_the_whole_set(bundle):
    ds, variant = bundle
    for name in (f"{variant}.h5ad", f"{variant}_polygons.parquet",
                 f"{variant}_edges_contact.parquet",
                 f"{variant}_edges_voronoi.parquet", f"{variant}_params.json"):
        assert (ds.bundle_dir / name).exists(), name


def test_params_record_the_settings_used(bundle):
    ds, variant = bundle
    meta = json.loads((ds.bundle_dir / f"{variant}_params.json").read_text())
    assert meta["dataset_id"] == ds.dataset_id
    assert meta["n_cells"] > 0
    assert "contact" in meta["graphs"] and "voronoi" in meta["graphs"]
    # Nothing is trimmed on the way in.
    assert "edges_trimmed" not in meta["params"]


def test_manifest_keeps_facts_per_bundle(bundle):
    ds, variant = bundle
    record = ds.manifest()
    assert record["dataset_id"] == ds.dataset_id
    assert record["bundles"][variant]["n_cells"] > 0


def test_rerunning_skips_completed_stages(bundle, capsys):
    from discell.preprocess.main import main

    ds, variant = bundle
    before = ds.bundle(variant).stat().st_mtime
    assert main(["--sample", str(ds.manifest()["bundles"][variant]["source"]),
                 "--variant", variant, "--only", "bundle", "--quiet"]) == 0
    assert "SKIP" in capsys.readouterr().out
    assert ds.bundle(variant).stat().st_mtime == before
