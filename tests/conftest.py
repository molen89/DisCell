"""Small generated fixtures: no data files."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def grid_adata(nx: int = 30, ny: int = 20, spacing_um: float = 12.0, n_types: int = 4,
               n_genes: int = 25, phi_dim: int = 16, seed: int = 0):
    """A jittered grid of cells with a 4-neighbor 'Voronoi' graph stored in .obsp."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    n = len(grid)
    pos = (grid * spacing_um + rng.normal(0, 1.0, (n, 2))).astype(np.float32)
    index = np.arange(n).reshape(nx, ny)
    pairs = np.concatenate([
        np.stack([index[:-1].ravel(), index[1:].ravel()], 1),
        np.stack([index[:, :-1].ravel(), index[:, 1:].ravel()], 1)])
    # a few long edges that the 40 um prune must drop
    pos[index[0, 0]] += 60.0
    ei, ej = pairs.min(1), pairs.max(1)
    dist = np.linalg.norm(pos[ei] - pos[ej], axis=1)
    face = rng.uniform(4.0, 10.0, len(ei))

    def sym(v):
        return sp.csr_matrix((np.r_[v, v], (np.r_[ei, ej], np.r_[ej, ei])), shape=(n, n))

    labels = np.array([f"type{g}" for g in rng.integers(0, n_types, n)], dtype=object)
    labels[rng.random(n) < 0.05] = None
    adata = ad.AnnData(X=sp.csr_matrix(rng.poisson(3.0, (n, n_genes)).astype(np.float32)))
    adata.obs_names = [f"cell{i}" for i in range(n)]
    adata.var_names = [f"gene{g}" for g in range(n_genes)]
    adata.obs["lineage"] = labels
    adata.obsm["spatial"] = pos
    adata.obsm["X_image"] = rng.normal(size=(n, phi_dim)).astype(np.float32)
    adata.obsp["voronoi_connectivities"] = sym(np.ones(len(ei), dtype=np.float32))
    adata.obsp["voronoi_shared_wall_um"] = sym(face)
    adata.obsp["voronoi_centroid_dist_um"] = sym(dist)
    return adata


@pytest.fixture()
def adata():
    return grid_adata()
