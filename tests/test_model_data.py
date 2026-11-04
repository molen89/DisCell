"""Assembly of the fit arrays from an AnnData with a DisCell graph."""

from __future__ import annotations

import numpy as np
import pytest

from discell.graph import build_graph
from discell.model.data import assemble, soft_clusters


def test_assemble_with_an_image(adata):
    build_graph(adata, "lineage")
    data = assemble(adata, "lineage", "X_image", tile_cells=100, v_pcs=4, seed=0)
    k = len(data.type_names)
    assert data.v_block.shape == (adata.n_obs, k - 1 + 4)
    assert data.e_phi.shape == (adata.n_obs, k) and data.phibar_t.shape == (k, k)
    assert np.allclose(data.e_phi.sum(1), 1.0, atol=1e-5)
    assert data.vbar_t.shape == (k, k - 1 + 4)
    cells = np.concatenate(data.train_tiles + data.val_tiles)
    assert np.array_equal(np.sort(cells), np.arange(adata.n_obs))
    assert data.median_counts == float(np.median(data.totals))
    assert not data.image_free


def test_assemble_image_free(adata):
    build_graph(adata, "lineage")
    data = assemble(adata, "lineage", None, tile_cells=100)
    assert data.image_free and data.phi.shape == (adata.n_obs, 0)
    assert data.e_phi is None and data.phibar_t is None
    assert np.array_equal(data.v_block, data.graph.y[:, :-1])


def test_assemble_is_seeded(adata):
    build_graph(adata, "lineage")
    a = assemble(adata, "lineage", "X_image", tile_cells=100, v_pcs=4, seed=3)
    b = assemble(adata, "lineage", "X_image", tile_cells=100, v_pcs=4, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a.val_tiles, b.val_tiles))
    assert np.array_equal(a.v_block, b.v_block)


def test_assemble_refuses_a_missing_image_key_and_a_missing_graph(adata):
    with pytest.raises(KeyError):
        assemble(adata, "lineage", None)
    build_graph(adata, "lineage")
    with pytest.raises(KeyError):
        assemble(adata, "lineage", "X_missing")


def test_positions_in_pixels_are_scaled(adata):
    build_graph(adata, "lineage")
    um = assemble(adata, "lineage", None, tile_cells=100).positions
    adata.obsm["spatial"] = adata.obsm["spatial"] / 0.5
    adata.uns["discell"]["microns_per_pixel"] = 0.5
    assert np.allclose(assemble(adata, "lineage", None, tile_cells=100).positions, um)


def test_soft_clusters_are_deterministic_at_one_thread():
    pts = np.random.default_rng(0).normal(size=(500, 3))
    a, _ = soft_clusters(pts, 4, seed=1, n_threads=1)
    b, _ = soft_clusters(pts, 4, seed=1, n_threads=1)
    assert np.array_equal(a, b) and a.dtype == np.float32
    assert np.allclose(a.sum(1), 1.0, atol=1e-6)
