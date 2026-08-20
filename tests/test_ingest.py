"""Geometry and units, from `xenium.self_test`."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def adata(bundle):
    from discell.data.bundle import load_bundle

    ds, variant = bundle
    return load_bundle(ds.bundle_dir, variant)


def test_expression_and_geometry_are_aligned(adata):
    polygons = adata.uns["polygons"]
    assert len(polygons) == adata.n_obs


def test_representative_points_lie_inside_their_polygon(adata):
    from shapely.geometry import Point

    reps = adata.obs[["rep_x_px", "rep_y_px"]].to_numpy()
    outside = [k for k in range(adata.n_obs)
               if adata.uns["polygons"][k].distance(Point(*reps[k])) > 1e-6]
    assert not outside


def test_cell_areas_are_plausible_in_microns(adata):
    # A Xenium cell is tens of um^2; a unit mix-up moves this by ~20x either way.
    assert 5 < adata.obs["area_um2"].median() < 2000


@pytest.mark.parametrize("graph", ["contact", "voronoi"])
def test_graphs_are_symmetric(adata, graph):
    conn = adata.obsp[f"{graph}_connectivities"]
    assert (conn != conn.T).nnz == 0


@pytest.mark.parametrize("graph", ["contact", "voronoi"])
def test_edge_metrics_are_in_range(adata, graph):
    from discell.preprocess.geometry import graph_edge_frame

    edges = graph_edge_frame(adata, graph)
    assert (edges["wall_dist_um"] >= 0).all()
    # Two boundaries cannot be further apart than the centroids they enclose.
    assert (edges["wall_dist_um"] <= edges["centroid_dist_um"] + 1e-6).all()
    assert (edges["apposed_wall_um"] >= 0).all()


def test_voronoi_faces_are_strictly_positive(adata):
    """The partition is exact, so every edge shares a face of real length.

    This is what makes voronoi the default graph: `beta` divides by it.
    """
    from discell.preprocess.geometry import graph_edge_frame

    assert (graph_edge_frame(adata, "voronoi")["shared_wall_um"] > 0).all()


def test_polygons_survive_the_bundle_round_trip(bundle, adata):
    import shapely

    ds, variant = bundle
    frame = __import__("pandas").read_parquet(
        ds.bundle_dir / f"{variant}_polygons.parquet")
    restored = shapely.from_wkb(frame["wkb"].to_numpy())
    assert list(map(str, frame["cell"])) == list(map(str, adata.obs_names))
    assert np.allclose([g.area for g in restored],
                       [p.area for p in adata.uns["polygons"]])
