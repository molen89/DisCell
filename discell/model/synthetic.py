#!/usr/bin/env python3
"""Data simulated from DisCell's own generative model, truths attached.

The recovery gate: if the machinery cannot recover a simulation of itself --
types in ``z``, the response loadings ``B``, the planted leak -- then nothing
it says about real tissue means anything. Simulating from the *model's* story
(types cluster in space, ``z`` carries identity, ``w`` responds to the niche,
``kappa`` mixes neighbours in) also reproduces the central confound: leakage
and signalling both make a cell resemble its neighbours, which is what the
sweep exists to separate.

Everything graph-shaped goes through :mod:`discell.model.prepare`, so the
simulation and the fit share one code path for ``beta``, ``y`` and the rings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discell.model.prepare import ModelGraph, build_graph


@dataclass
class Simulation:
    """One synthetic tissue with every latent the fit is asked to recover."""

    graph: ModelGraph
    positions: np.ndarray      # (N, 2) um
    t: np.ndarray              # (N,) type index
    x: np.ndarray              # (N, G) counts
    totals: np.ndarray         # (N,) library sizes
    phi: np.ndarray            # (N, d_phi) image-like niche descriptor
    # the truths
    z_true: np.ndarray         # (N, d_z)
    w_true: np.ndarray         # (N, d_w)
    B_true: np.ndarray         # (d_w, G)
    rho_true: np.ndarray       # (N, G) clean compositions
    p_true: np.ndarray         # (N, G) after the leak mixture
    kappa: float

    @property
    def n_types(self) -> int:
        return int(self.t.max()) + 1


def simulate(n_cells: int = 6000, n_genes: int = 60, n_types: int = 4,
             d_z: int = 4, d_w: int = 2, d_phi: int = 8,
             kappa: float = 0.2, mean_counts: float = 150.0,
             box_um: float = 1000.0, seed: int = 0) -> Simulation:
    """Simulate one tissue. Defaults match Xenium's ~13 um cell spacing.

    The pieces, in the generative order of the spec:

    * positions uniform; **types cluster in space** (soft Voronoi of anchor
      points), because without that ``y`` carries nothing and neither leakage
      nor response would have anything to work with;
    * the graph is scipy Delaunay -> :func:`build_graph`, same pruning and
      ``beta`` as the real pipeline;
    * ``z ~ N(m_t, 0.5^2)`` -- identity, type-structured;
    * ``w = M y + noise`` -- the response is a linear read of the true niche;
    * ``log rho = z A + w B``, ``p = (1-kappa) rho + kappa rho_bar``,
      ``x ~ Multinomial(l, p)``;
    * ``phi = P y + noise`` -- the image knows the niche, imperfectly.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, box_um, size=(n_cells, 2))

    # spatially clustered types: nearest anchor, softened so borders mix
    anchors = rng.uniform(0, box_um, size=(3 * n_types, 2))
    anchor_type = np.arange(3 * n_types) % n_types
    gaps = np.linalg.norm(positions[:, None] - anchors[None], axis=-1)
    logits = -gaps / (0.05 * box_um) + rng.gumbel(size=gaps.shape)
    t = anchor_type[logits.argmax(axis=1)]

    from scipy.spatial import Delaunay

    simplices = Delaunay(positions).simplices
    pairs = np.unique(np.sort(np.vstack(
        [simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]]
    ), axis=1), axis=0)
    dist = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    graph = build_graph(pairs[:, 0], pairs[:, 1], np.ones(len(pairs)), dist,
                        n_cells, type_index=t, n_types=n_types)

    m_t = rng.normal(0.0, 1.5, size=(n_types, d_z))
    z = m_t[t] + 0.5 * rng.standard_normal((n_cells, d_z))
    w = graph.y @ rng.normal(0.0, 1.0, size=(n_types, d_w)) \
        + 0.3 * rng.standard_normal((n_cells, d_w))

    a_load = rng.normal(0.0, 1.0, size=(d_z, n_genes))
    b_load = rng.normal(0.0, 1.0, size=(d_w, n_genes))
    logits_rho = z @ a_load + w @ b_load
    rho = np.exp(logits_rho - logits_rho.max(axis=1, keepdims=True))
    rho /= rho.sum(axis=1, keepdims=True)

    rho_bar = graph.in_edges @ rho                      # rows sum to 1 or 0
    p = (1.0 - kappa) * rho + kappa * rho_bar
    p /= p.sum(axis=1, keepdims=True)                   # isolated -> kappa_i = 0

    totals = np.maximum(rng.poisson(mean_counts, n_cells), 20)
    x = np.stack([rng.multinomial(totals[i], p[i]) for i in range(n_cells)])

    phi = graph.y @ rng.normal(0.0, 1.0, size=(n_types, d_phi)) \
        + 0.4 * rng.standard_normal((n_cells, d_phi))

    return Simulation(graph=graph, positions=positions, t=t,
                      x=x.astype(np.float32), totals=totals.astype(np.float32),
                      phi=phi.astype(np.float32),
                      z_true=z, w_true=w, B_true=b_load,
                      rho_true=rho, p_true=p, kappa=kappa)
