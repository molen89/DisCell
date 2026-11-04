"""Xenium ingest and the clipped-Voronoi graph, on a generated fake Xenium directory."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

PIXEL_SIZE = 0.2125


def write_fake_xenium(root, donors=("A", "B", "C"), per_donor=9, control_probes=1):
    """A tiny Xenium outs directory: 15 um square cells on a 20 um grid, one block
    of cells per donor (1 mm apart), a cell_groups.csv with group and donor, and
    one control-probe feature that the reader must drop."""
    import h5py

    root.mkdir(parents=True, exist_ok=True)
    cells, verts = [], []
    side = int(np.sqrt(per_donor))
    for d_index, donor in enumerate(donors):
        for i in range(per_donor):
            cid = f"{donor.lower()}{i}-1"
            x0 = 20.0 * (i % side) + 1000.0 * d_index
            y0 = 20.0 * (i // side)
            cells.append((cid, donor, "T1" if i % 2 else "T2"))
            for vx, vy in ((x0, y0), (x0 + 15, y0), (x0 + 15, y0 + 15), (x0, y0 + 15),
                           (x0, y0)):
                verts.append((cid, vx, vy))
    pd.DataFrame(verts, columns=["cell_id", "vertex_x", "vertex_y"]).to_parquet(
        root / "cell_boundaries.parquet")
    (root / "experiment.xenium").write_text(json.dumps({"pixel_size": PIXEL_SIZE,
                                                        "region_name": "fake"}))
    n = len(cells)
    genes = ["G1", "G2", "G3"] + [f"NegControlProbe_{k}" for k in range(control_probes)]
    kinds = ["Gene Expression"] * 3 + ["Negative Control Probe"] * control_probes
    counts = np.random.default_rng(0).poisson(3.0, size=(n, len(genes))).astype(np.int32)
    with h5py.File(root / "cell_feature_matrix.h5", "w") as h:
        m = h.create_group("matrix")
        data, indices, indptr = [], [], [0]
        for c in range(n):                      # 10x stores genes x cells, CSC
            nz = np.flatnonzero(counts[c])
            data.extend(counts[c, nz])
            indices.extend(nz)
            indptr.append(len(data))
        m.create_dataset("data", data=np.asarray(data, dtype=np.int32))
        m.create_dataset("indices", data=np.asarray(indices, dtype=np.int64))
        m.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int64))
        m.create_dataset("shape", data=np.array([len(genes), n], dtype=np.int32))
        m.create_dataset("barcodes", data=np.array([c for c, _, _ in cells], dtype="S"))
        f = m.create_group("features")
        f.create_dataset("id", data=np.array([f"ENSG{i}" for i in range(len(genes))], dtype="S"))
        f.create_dataset("name", data=np.array(genes, dtype="S"))
        f.create_dataset("feature_type", data=np.array(kinds, dtype="S"))
        f.create_dataset("genome", data=np.array(["GRCh38"] * len(genes), dtype="S"))
    pd.DataFrame([(c, g, d) for c, d, g in cells],
                 columns=["cell_id", "group", "donor"]).to_csv(
        root / "fake_cell_groups.csv", index=False)
    return root, counts


@pytest.fixture()
def fake(tmp_path):
    return write_fake_xenium(tmp_path / "Fake_outs")


def test_reads_counts_genes_and_centroids(fake):
    from discell import read_xenium

    root, counts = fake
    adata = read_xenium(root)
    assert adata.n_obs == 27
    assert list(adata.var_names) == ["G1", "G2", "G3"]          # control probe dropped
    assert adata.var["gene_id"].tolist() == ["ENSG0", "ENSG1", "ENSG2"]
    assert adata.X.dtype == np.float32
    np.testing.assert_array_equal(adata.X.toarray(), counts[:, :3])
    assert adata.obs_names[0] == "a0-1"
    # 15 um squares: center of mass 7.5 um from the corner, in um and in pixels
    np.testing.assert_allclose(adata.obsm["spatial"][0], [7.5, 7.5], atol=1e-9)
    np.testing.assert_allclose(adata.obsm["spatial"][4], [27.5, 27.5], atol=1e-9)
    np.testing.assert_allclose(adata.obsm["spatial_px"] * PIXEL_SIZE, adata.obsm["spatial"],
                               rtol=0, atol=1e-9)
    assert adata.uns["discell"]["xenium"]["pixel_size_um"] == PIXEL_SIZE
    assert "microns_per_pixel" not in adata.uns["discell"]       # spatial is in um


def test_labels_attach_from_the_cell_groups_table(fake):
    from discell import read_xenium

    root, _ = fake
    adata = read_xenium(root)
    assert set(adata.obs["cell_group"]) == {"T1", "T2"}
    assert set(adata.obs["donor"]) == {"A", "B", "C"}
    assert adata.obs.loc["a1-1", "cell_group"] == "T1"
    bare = read_xenium(root, labels=None)
    assert "cell_group" not in bare.obs
    table = pd.DataFrame({"cell_id": ["a0-1", "b3-1"], "lineage": ["Epi", "Imm"]})
    custom = read_xenium(root, labels=table)
    assert custom.obs.loc["b3-1", "lineage"] == "Imm"
    assert custom.obs["lineage"].isna().sum() == 25


def test_donor_subset_keeps_only_those_cores(fake):
    from discell import read_xenium

    root, counts = fake
    subset = read_xenium(root, donors=["B", "C"])
    assert subset.n_obs == 18
    assert set(subset.obs["donor"]) == {"B", "C"}
    xs = subset.obsm["spatial"][:, 0]
    assert ((xs >= 1000) & (xs < 2100)).all()
    np.testing.assert_array_equal(subset.X.toarray(), counts[9:, :3])
    with pytest.raises(Exception, match="no cells from donors"):
        read_xenium(root, donors=["Z"])
    with pytest.raises(Exception, match="donor"):
        read_xenium(root, labels=None, donors=["A"])


def test_region_subset(fake):
    from discell import read_xenium

    root, _ = fake
    region = read_xenium(root, region_um=(0, 0, 30, 30))
    assert list(region.obs_names) == ["a0-1", "a1-1", "a3-1", "a4-1"]


def test_missing_files_raise(tmp_path):
    from discell.io.xenium import XeniumError, read_xenium

    with pytest.raises(XeniumError, match="cell_feature_matrix.h5"):
        read_xenium(tmp_path)


def test_cells_without_boundary_are_dropped(fake):
    from discell import read_xenium

    root, _ = fake
    frame = pd.read_parquet(root / "cell_boundaries.parquet")
    frame[frame["cell_id"] != "a2-1"].to_parquet(root / "cell_boundaries.parquet")
    adata = read_xenium(root)
    assert adata.n_obs == 26 and "a2-1" not in adata.obs_names


def test_voronoi_on_a_grid():
    """On a square grid the Voronoi regions are squares: faces = spacing, only the
    4-neighbors connect, and distances are the spacing."""
    from discell.graph.voronoi import voronoi_edges

    gx, gy = np.meshgrid(np.arange(4), np.arange(3), indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel()], 1) * 20.0   # exact grid: diagonals meet in a point
    ei, ej, face, dist = voronoi_edges(pos, clip_radius_um=30.0)
    assert len(ei) == 3 * 3 + 4 * 2                     # 9 horizontal + 8 vertical
    assert (ei < ej).all()
    np.testing.assert_allclose(dist, 20.0, atol=1e-4)
    faces = dict(zip(zip(ei, ej), face))
    # cells (1,1) and (2,1) (indices 4, 7): an interior face, the full 20 um
    np.testing.assert_allclose(faces[(4, 7)], 20.0, atol=1e-4)
    # cells (0,0) and (1,0) (0, 3): the face x = 10 runs from y = 10 out to the
    # 30 um disc, at y = -sqrt(30^2 - 10^2)
    np.testing.assert_allclose(faces[(0, 3)], 10.0 + np.sqrt(800.0), rtol=2e-3)


def test_voronoi_clip_cuts_long_faces():
    """Two cells 50 um apart: unclipped, their face is the whole bisector of the
    (unbounded) diagram; clipped at 30 um it is the chord 2*sqrt(30^2 - 25^2)."""
    from discell.graph.voronoi import voronoi_edges

    pos = np.array([[0.0, 0.0], [50.0, 0.0], [25.0, 400.0], [25.0, -400.0]])
    ei, ej, face, _ = voronoi_edges(pos, clip_radius_um=30.0, max_candidate_um=100.0)
    k = np.flatnonzero((ei == 0) & (ej == 1))
    assert len(k) == 1
    np.testing.assert_allclose(face[k], 2 * np.sqrt(30.0**2 - 25.0**2), rtol=2e-3)
    ei2, ej2, *_ = voronoi_edges(pos, clip_radius_um=10.0)
    assert not ((ei2 == 0) & (ej2 == 1)).any()          # discs of 10 um do not meet


def test_unit_frame_gives_the_same_graph_in_um():
    from discell.graph.voronoi import voronoi_edges

    rng = np.random.default_rng(0)
    pos_um = rng.uniform(0, 200, (150, 2))
    a = voronoi_edges(pos_um)
    b = voronoi_edges(pos_um / PIXEL_SIZE, unit_um=PIXEL_SIZE)
    common = set(zip(a[0], a[1])) & set(zip(b[0], b[1]))
    assert len(common) >= 0.98 * len(a[0])
    fa = dict(zip(zip(a[0], a[1]), a[2]))
    fb = dict(zip(zip(b[0], b[1]), b[2]))
    assert max(abs(fa[k] - fb[k]) for k in common) < 1e-6


def test_build_graph_straight_from_centroids(fake):
    import discell as dc

    root, _ = fake
    adata = dc.read_xenium(root)
    assert "voronoi_connectivities" not in adata.obsp
    graph = dc.build_graph(adata, "cell_group")
    assert "voronoi_connectivities" in adata.obsp
    assert adata.uns["discell"]["voronoi"]["spatial_key"] == "spatial_px"
    # 3 donors x a 3x3 grid of 4-neighbors, 20 um apart (donors 1 mm apart: no bridge)
    assert len(graph.edge_i) == 3 * 12
    np.testing.assert_allclose(adata.obsp["discell_dist_um"].data, 20.0, atol=1e-6)
    assert not graph.isolated.any()
    np.testing.assert_allclose(adata.obsm["discell_y"].sum(1), 1.0, atol=1e-6)


def test_graph_from_um_only_adata(fake):
    """An AnnData with only um centroids (no pixel frame) still builds."""
    import discell as dc

    root, _ = fake
    adata = dc.read_xenium(root)
    del adata.obsm["spatial_px"]
    graph = dc.build_graph(adata, "cell_group")
    assert adata.uns["discell"]["voronoi"]["unit_um"] == 1.0
    assert len(graph.edge_i) == 36


def test_read_10x_h5_gex_only_false(fake):
    from discell.io.xenium import read_10x_h5

    root, counts = fake
    full = read_10x_h5(root / "cell_feature_matrix.h5", gex_only=False)
    assert full.n_vars == 4
    np.testing.assert_array_equal(full.X.toarray(), counts)


#: the raw GSE315411 slide 11 outs directory (GSM9427181), if available locally
SLIDE11 = os.environ.get("DISCELL_SLIDE11", "")


@pytest.mark.slow
def test_slide11_pdl018d_cell_and_edge_counts():
    """The TMA core read from the raw slide: the paper's 69,422 cells and 205,929
    Voronoi edges."""
    from pathlib import Path

    import discell as dc
    from discell.graph.voronoi import add_voronoi_graph

    if not SLIDE11 or not Path(SLIDE11).is_dir():
        pytest.skip("raw GSE315411 slide 11 not available (set DISCELL_SLIDE11)")
    adata = dc.read_xenium(SLIDE11, labels=Path(SLIDE11) / "GSE315411_prime_solo_cell_groups.csv",
                           donors=["PDL018D"])
    assert adata.n_obs == 69422
    assert add_voronoi_graph(adata) == 205929
