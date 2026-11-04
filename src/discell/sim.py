"""A tissue simulated from DisCell's own generative model, with its truths.

If the fit cannot recover a simulation of itself (types in ``z``, the
response loadings ``B``, the simulated spillover), nothing it says about real tissue
means anything. The simulation follows the model's story: types cluster in
space, ``z`` carries identity, ``w`` responds to the niche and ``kappa`` mixes
neighbors in, so spillover and signaling both make a cell resemble its
neighbors, the confound the kappa sweep exists to separate.

The graph goes through :func:`discell.graph.kernel.graph_from_edges`, the same
pruning and ``beta`` as a real section. :meth:`Simulation.to_anndata` gives the
input contract (with the Delaunay graph stored as the face graph, unit faces),
ready for :func:`discell.build_graph` and :func:`discell.fit`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from discell.graph.kernel import ModelGraph, graph_from_edges


@dataclass
class Simulation:
    """One synthetic tissue with every latent a fit is asked to recover."""

    graph: ModelGraph
    edge_i: np.ndarray         # unpruned Delaunay edges, i < j
    edge_j: np.ndarray
    positions: np.ndarray      # (N, 2) um
    t: np.ndarray              # (N,) type index
    x: np.ndarray              # (N, G) counts
    totals: np.ndarray         # (N,) library sizes
    phi: np.ndarray            # (N, d_phi) image-like niche descriptor
    z_true: np.ndarray         # (N, d_z)
    w_true: np.ndarray         # (N, d_w)
    B_true: np.ndarray         # (d_w, G)
    rho_true: np.ndarray       # (N, G) clean profiles
    p_true: np.ndarray         # (N, G) after the spillover mixture
    kappa: float

    @property
    def n_types(self) -> int:
        return int(self.t.max()) + 1

    def to_anndata(self, label_key: str = "cell_type", image_key: str = "X_image"):
        """The input contract: counts in ``.X``, types ``type<g>`` in
        ``.obs[label_key]``, centroids (um) in ``.obsm['spatial']``, ``phi`` in
        ``.obsm[image_key]`` and the unpruned graph in ``.obsp['voronoi_*']``
        (face length 1 on every edge, as the simulation's kernel uses)."""
        import anndata as ad

        n = len(self.t)
        ei, ej = self.edge_i, self.edge_j
        dist = np.linalg.norm(self.positions[ei] - self.positions[ej], axis=1)

        def symmetric(values: np.ndarray) -> sp.csr_matrix:
            return sp.csr_matrix((np.r_[values, values], (np.r_[ei, ej], np.r_[ej, ei])),
                                 shape=(n, n))

        adata = ad.AnnData(X=sp.csr_matrix(self.x))
        adata.obs_names = [f"cell{i}" for i in range(n)]
        adata.var_names = [f"gene{g}" for g in range(self.x.shape[1])]
        width = len(str(self.n_types - 1))
        adata.obs[label_key] = [f"type{g:0{width}d}" for g in self.t]
        adata.obsm["spatial"] = self.positions.astype(np.float32)
        adata.obsm[image_key] = self.phi
        adata.obsp["voronoi_connectivities"] = symmetric(np.ones(len(ei), dtype=np.float32))
        adata.obsp["voronoi_shared_wall_um"] = symmetric(np.ones(len(ei)))
        adata.obsp["voronoi_centroid_dist_um"] = symmetric(dist)
        return adata


def simulate(n_cells: int = 6000, n_genes: int = 60, n_types: int = 4,
             d_z: int = 4, d_w: int = 2, d_phi: int = 8, kappa: float = 0.2,
             mean_counts: float = 150.0, box_um: float = 1000.0, seed: int = 0,
             w_shift: np.ndarray | None = None,
             b_zero: np.ndarray | None = None) -> Simulation:
    """Simulate one tissue. The defaults give Xenium's ~13 um cell spacing.

    In generative order:

    * positions uniform; types cluster in space (a softened nearest-anchor
      map), without which ``y`` would carry nothing;
    * the graph: Delaunay, then the pipeline's pruning and ``beta`` (unit faces);
    * ``z ~ N(m_t, 0.5^2)``: identity, type-structured;
    * ``w = M y + noise``: the response is a linear read of the true niche;
    * ``log rho = z A + w B``, ``p = (1 - kappa) rho + kappa rho_bar``,
      ``x ~ Multinomial(l, p)``;
    * ``phi = P y + noise``: the image knows the niche, imperfectly.

    *w_shift* ``(N, d_w)``: a per-cell response beyond the niche, added to ``w``
    in the log-rate only (``w_true`` stays the niche response). *b_zero*
    ``(G,)`` bool: genes without a response (their columns of ``B`` are 0), so
    any niche dependence they show comes from the spillover alone. Neither moves a
    random draw.
    """
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, box_um, size=(n_cells, 2))

    anchors = rng.uniform(0, box_um, size=(3 * n_types, 2))
    anchor_type = np.arange(3 * n_types) % n_types
    gaps = np.linalg.norm(positions[:, None] - anchors[None], axis=-1)
    logits = -gaps / (0.05 * box_um) + rng.gumbel(size=gaps.shape)
    t = anchor_type[logits.argmax(axis=1)]

    simplices = Delaunay(positions).simplices
    pairs = np.unique(np.sort(np.vstack(
        [simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]]), axis=1), axis=0)
    dist = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    graph = graph_from_edges(pairs[:, 0], pairs[:, 1], np.ones(len(pairs)), dist, n_cells,
                             type_index=t, n_types=n_types)

    m_t = rng.normal(0.0, 1.5, size=(n_types, d_z))
    z = m_t[t] + 0.5 * rng.standard_normal((n_cells, d_z))
    w = graph.y @ rng.normal(0.0, 1.0, size=(n_types, d_w)) \
        + 0.3 * rng.standard_normal((n_cells, d_w))

    a_load = rng.normal(0.0, 1.0, size=(d_z, n_genes))
    b_load = rng.normal(0.0, 1.0, size=(d_w, n_genes))
    if b_zero is not None:
        b_load[:, np.asarray(b_zero, dtype=bool)] = 0.0
    w_rate = w if w_shift is None else w + w_shift
    logits_rho = z @ a_load + w_rate @ b_load
    rho = np.exp(logits_rho - logits_rho.max(axis=1, keepdims=True))
    rho /= rho.sum(axis=1, keepdims=True)

    rho_bar = graph.in_edges @ rho                      # rows sum to 1, or 0 if isolated
    p = (1.0 - kappa) * rho + kappa * rho_bar
    p /= p.sum(axis=1, keepdims=True)

    totals = np.maximum(rng.poisson(mean_counts, n_cells), 20)
    x = np.stack([rng.multinomial(totals[i], p[i]) for i in range(n_cells)])

    phi = graph.y @ rng.normal(0.0, 1.0, size=(n_types, d_phi)) \
        + 0.4 * rng.standard_normal((n_cells, d_phi))

    return Simulation(graph=graph, edge_i=pairs[:, 0], edge_j=pairs[:, 1],
                      positions=positions, t=t, x=x.astype(np.float32),
                      totals=totals.astype(np.float32), phi=phi.astype(np.float32),
                      z_true=z, w_true=w, B_true=b_load, rho_true=rho, p_true=p,
                      kappa=kappa)
