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


def test_cell_groups_carry_optional_donor(tmp_path):
    """A TMA's cell_groups.csv can add a donor column; it survives the read so
    load_xenium_sample can attach it to obs next to cell_group."""
    from discell.preprocess.labels import read_cell_groups

    (tmp_path / "x_cell_groups.csv").write_text(
        "cell_id,group,donor\na-1,Macrophage,PDL018D\nb-1,AT2,PDL026A\n")
    frame = read_cell_groups(tmp_path)
    assert list(frame.index) == ["a-1", "b-1"]
    assert frame.loc["b-1", "group"] == "AT2"
    assert frame.loc["a-1", "donor"] == "PDL018D"
    assert read_cell_groups(tmp_path / "nowhere").empty


def test_cell_groups_carry_donor_through(tmp_path):
    """A TMA's ``*cell_groups.csv`` adds ``donor`` next to ``group``; the
    10x files have no such column and still read as before."""
    import pandas as pd

    from discell.preprocess.labels import read_cell_groups

    (tmp_path / "x_cell_groups.csv").write_text(
        "cell_id,group,donor\na,T cell,PDL001\nb,Fibroblast,PDL002\n")
    frame = read_cell_groups(tmp_path)
    assert list(frame.index) == ["a", "b"]
    assert frame.loc["b", "group"] == "Fibroblast"
    assert frame.loc["a", "donor"] == "PDL001"
    assert pd.isna(frame.reindex(["a", "zzz"]).loc["zzz", "donor"])

    (tmp_path / "x_cell_groups.csv").write_text("cell_id,group\na,T cell\n")
    assert "donor" not in read_cell_groups(tmp_path).columns


def test_donor_subset_needs_a_donor_column(bundle):
    """--donors is a TMA feature: on a slide whose cell_groups carry no donor
    column it must refuse rather than silently bundle every cell."""
    import pytest

    from discell.preprocess.main import main
    from discell.preprocess.xenium import XeniumError

    ds, variant = bundle
    with pytest.raises(XeniumError, match="donor"):
        main(["--sample", str(ds.manifest()["bundles"][variant]["source"]),
              "--variant", "donor-test", "--only", "bundle", "--max-cells", "1500",
              "--donors", "PDL000X", "--quiet"])


def _write_fake_xenium(root, donors=("A", "B", "C"), per_donor=9):
    """A tiny Xenium outs directory: square cells on a grid, one block of
    cells per donor, and a cell_groups.csv carrying group and donor."""
    import json

    import h5py
    import numpy as np
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    cells, verts = [], []
    side = int(np.sqrt(per_donor))
    for d_index, donor in enumerate(donors):
        for i in range(per_donor):
            cid = f"{donor.lower()}{i}-1"
            x0 = 20.0 * (i % side) + 1000.0 * d_index          # donors 1 mm apart
            y0 = 20.0 * (i // side)
            cells.append((cid, donor, "T1" if i % 2 else "T2"))
            for vx, vy in ((x0, y0), (x0 + 15, y0), (x0 + 15, y0 + 15), (x0, y0 + 15), (x0, y0)):
                verts.append((cid, vx, vy))
    pd.DataFrame(verts, columns=["cell_id", "vertex_x", "vertex_y"]).to_parquet(
        root / "cell_boundaries.parquet")
    (root / "experiment.xenium").write_text(json.dumps({"pixel_size": 0.2125,
                                                        "region_name": "fake"}))
    n, genes = len(cells), ["G1", "G2", "G3"]
    counts = np.random.default_rng(0).poisson(3.0, size=(n, len(genes))).astype(np.int32)
    with h5py.File(root / "cell_feature_matrix.h5", "w") as h:
        m = h.create_group("matrix")
        dense = counts.T                                        # 10x stores genes x cells, CSC
        nz = [(g, c, dense[g, c]) for c in range(n) for g in range(len(genes)) if dense[g, c]]
        m.create_dataset("data", data=np.array([v for _, _, v in nz], dtype=np.int32))
        m.create_dataset("indices", data=np.array([g for g, _, _ in nz], dtype=np.int64))
        indptr = np.cumsum([0] + [int((dense[:, c] != 0).sum()) for c in range(n)])
        m.create_dataset("indptr", data=indptr.astype(np.int64))
        m.create_dataset("shape", data=np.array([len(genes), n], dtype=np.int32))
        m.create_dataset("barcodes", data=np.array([c for c, _, _ in cells], dtype="S"))
        f = m.create_group("features")
        f.create_dataset("id", data=np.array([f"ENSG{i}" for i in range(len(genes))], dtype="S"))
        f.create_dataset("name", data=np.array(genes, dtype="S"))
        f.create_dataset("feature_type", data=np.array(["Gene Expression"] * len(genes), dtype="S"))
        f.create_dataset("genome", data=np.array(["GRCh38"] * len(genes), dtype="S"))
    pd.DataFrame([(c, g, d) for c, d, g in cells],
                 columns=["cell_id", "group", "donor"]).to_csv(root / "fake_cell_groups.csv", index=False)
    return root


def test_donor_subset_keeps_only_those_cores(tmp_path):
    """--donors keeps the named cores' cells (with their labels and donor
    column), builds graphs on them alone, and records the choice."""
    from discell.preprocess.xenium import load_xenium_sample

    sample = _write_fake_xenium(tmp_path / "Fake_Sample")
    everything = load_xenium_sample(sample, build_graphs=False)
    assert everything.n_obs == 27 and set(everything.obs["donor"]) == {"A", "B", "C"}

    subset = load_xenium_sample(sample, donors=["B", "C"])
    assert subset.n_obs == 18
    assert set(subset.obs["donor"]) == {"B", "C"}
    assert set(subset.obs["cell_group"]) == {"T1", "T2"}
    assert subset.uns["default_label"] == "cell_group"
    # geometry stayed aligned with the kept rows: every centroid lies in a kept block
    xs = subset.obsm["spatial_um"][:, 0]
    assert ((xs >= 1000) & (xs < 2100)).all()
    assert subset.obsp["voronoi_connectivities"].shape == (18, 18)
