#!/usr/bin/env python3
"""Planted worlds on the gate scaffold -- the shared adjudicator (doc 11).

``plant_programme``: a gene programme written into chosen cells' clean
compositions and re-mixed through the true leak operator.
``fit_synthetic``: one short DisCell fit on a synthetic tissue, posterior
means for every cell; the gate's calibration, nothing tuned per world.
"""

from __future__ import annotations

import numpy as np


def plant_programme(sim, planted: np.ndarray, genes: np.ndarray,
                    share: float, rng: np.random.Generator):
    """``rho_i <- (1 - share) rho_i + share * uniform(genes)`` for planted cells.

    A programme defined by the transcript share it takes, not by a log-fold:
    on the simulator's skewed compositions a log-fold on random genes moves
    almost nothing (issues V9, the under-powered plant), a share moves the
    same mass in every seed. Leak re-mixed through the true operator, counts
    resampled; ``sim`` is updated in place (x, rho_true, p_true) and returned.
    """
    rho = sim.rho_true.copy()
    programme = np.zeros(rho.shape[1])
    programme[genes] = 1.0 / len(genes)
    rho[planted] = (1.0 - share) * rho[planted] + share * programme
    rho_bar = sim.graph.in_edges @ rho
    p = (1.0 - sim.kappa) * rho + sim.kappa * rho_bar
    p /= p.sum(axis=1, keepdims=True).clip(min=1e-12)
    sim.x = np.stack([rng.multinomial(int(sim.totals[i]), p[i])
                      for i in range(len(sim.t))]).astype(np.float32)
    sim.rho_true, sim.p_true = rho, p
    return sim


def fit_synthetic(sim, epochs: int = 400, device: str = "cuda",
                  seed: int = 0, subtract_leak: bool = False) -> dict:
    """Returns {"z", "w", "b_matrix", "fold", "rho_bar"} for *sim* (all cells).

    ``rho_bar`` is the model's own foreign influx per cell (posterior means,
    the pass-1 encoding), the quantity the counts-level correction
    ``x~ = x - kappa l rho_bar`` needs.
    """
    import torch

    from discell.model.elbo import Weights, discell_loss
    from discell.model.equations import TypeCovariances
    from discell.model.networks import DisCell
    from discell.model.prepare import spatial_tiles, tile_batch

    torch.manual_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    genes, k = sim.x.shape[1], sim.n_types
    d_z, d_w = 8, 2
    model = DisCell(genes, k, sim.phi.shape[1],
                    median_counts=float(np.median(sim.totals)),
                    d_z=d_z, d_w=d_w, hidden=128, gat_dim=16,
                    subtract_leak=subtract_leak).to(device)
    cov = TypeCovariances(k, d_z, k - 1 + sim.phi.shape[1],
                          ema=0.05, min_count=100).to(device)
    weights = Weights(omega=1.0, alpha_z=0.007, alpha_w=0.1, alpha_a=0.02)
    p_t = torch.tensor(np.bincount(sim.t, minlength=k) / len(sim.t),
                       dtype=torch.float32, device=device)
    y_all = torch.tensor(sim.graph.y, dtype=torch.float32, device=device)
    phi_all = torch.tensor(sim.phi, device=device)
    batches = []
    for tile in spatial_tiles(sim.positions, 512):
        b = tile_batch(sim.graph, tile)
        batches.append((b.nodes, dict(
            x=torch.tensor(sim.x[b.nodes], device=device),
            t=torch.tensor(sim.t[b.nodes], device=device),
            phi=phi_all[torch.tensor(b.nodes, device=device)],
            isolated=torch.tensor(sim.graph.isolated[b.nodes], device=device),
            gat_src=torch.tensor(b.gat_src, device=device),
            gat_dst=torch.tensor(b.gat_dst, device=device),
            leak_src=torch.tensor(b.leak_src, device=device),
            leak_dst=torch.tensor(b.leak_dst, device=device),
            leak_beta=torch.tensor(b.leak_beta, dtype=torch.float32,
                                   device=device),
            n_seeds=b.n_seeds, n_context=b.n_context)))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for j in rng.permutation(len(batches)):
            nodes, tensors = batches[j]
            fwd = model(**tensors, kappa=sim.kappa)
            n = tensors["n_seeds"]
            seeds_t = torch.tensor(nodes[:n], device=device)
            terms = discell_loss(
                fwd, tensors["x"][:n], tensors["t"][:n], weights=weights,
                v=torch.cat([y_all[seeds_t][:, :-1], phi_all[seeds_t]], dim=-1),
                covariances=cov, p_t=p_t)
            optimiser.zero_grad()
            terms.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimiser.step()
        schedule.step()
    model.eval()
    n_cells = len(sim.t)
    z_hat = np.zeros((n_cells, d_z), dtype=np.float32)
    w_hat = np.zeros((n_cells, d_w), dtype=np.float32)
    rho_bar = np.zeros((n_cells, genes), dtype=np.float32)
    fold = np.zeros(n_cells, dtype=np.int64)
    with torch.no_grad():
        for j, (nodes, tensors) in enumerate(batches):
            fwd = model(**tensors, kappa=sim.kappa, sample=False)
            s = tensors["n_seeds"]
            z_hat[nodes[:s]] = fwd.mu_z[:s].cpu().numpy()
            w_hat[nodes[:s]] = fwd.mu_w[:s].cpu().numpy()
            rho_bar[nodes[:s]] = fwd.rho_bar[:s].cpu().numpy()
            fold[nodes[:s]] = j % 5
    return {"z": z_hat, "w": w_hat, "fold": fold, "rho_bar": rho_bar,
            "b_matrix": model.B.weight.detach().cpu().numpy()}
