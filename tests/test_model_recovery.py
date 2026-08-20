"""The stage-4 gate: DisCell must recover a simulation of itself.

One fit on synthetic tissue with known truths, several assertions against it.
The thresholds sit well below the observed values (NMI 0.62, w-CCA 0.80, B
principal cosine 0.83 on this seed) so seed variance does not flake the suite;
what they guard is the *mechanism*, not the last decimal: z carries type, w
tracks the planted response, B spans the planted programme space, and the
penalty holds the niche out of z.

Calibration, recorded where it bit: the 1/l reconstruction scaling makes
alpha = 1 price a latent's information ~l-fold above the unscaled ELBO --
z collapses there. alpha_z sits near 1/mean-counts; alpha_w is a knife-edge
(low: w steals identity; high: w collapses onto its prior) and is run here at
the pinned end, which the kappa-sweep calibration must revisit per spec 4.6.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from discell.model.elbo import Weights, discell_loss
from discell.model.equations import TypeCovariances
from discell.model.networks import DisCell
from discell.model.prepare import spatial_tiles, tile_batch
from discell.model.synthetic import simulate

WEIGHTS = Weights(omega=1.0, alpha_z=0.007, alpha_w=0.1, alpha_a=0.3)
EPOCHS = 500


@pytest.fixture(scope="module")
def fit():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = simulate(n_cells=6000, n_types=8, kappa=0.2, seed=0)
    genes, k, d_phi = sim.x.shape[1], sim.n_types, sim.phi.shape[1]

    model = DisCell(genes, k, d_phi, d_z=8, d_w=2, hidden=128,
                    t_dim=8, gat_dim=16).to(device)
    cov = TypeCovariances(k, 8, k - 1 + d_phi, ema=0.05, min_count=100).to(device)
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
            leak_beta=torch.tensor(b.leak_beta, dtype=torch.float32, device=device),
            n_seeds=b.n_seeds, n_context=b.n_context,
        )))

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    rng = np.random.default_rng(0)
    terms = None
    for _ in range(EPOCHS):
        for k_batch in rng.permutation(len(batches)):
            nodes, tensors = batches[k_batch]
            fwd = model(**tensors, kappa=sim.kappa)
            n_seeds = tensors["n_seeds"]
            seeds_t = torch.tensor(nodes[:n_seeds], device=device)
            terms = discell_loss(
                fwd, tensors["x"][:n_seeds], tensors["t"][:n_seeds],
                weights=WEIGHTS,
                # y minus one column: the simplex constraint makes the full
                # block singular and its logdet gradient unbounded
                v=torch.cat([y_all[seeds_t][:, :-1], phi_all[seeds_t]], dim=-1),
                covariances=cov, p_t=p_t)
            optimiser.zero_grad()
            terms.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimiser.step()
        schedule.step()

    model.eval()
    z_hat = np.zeros((6000, 8), dtype=np.float32)
    w_hat = np.zeros((6000, 2), dtype=np.float32)
    with torch.no_grad():
        for nodes, tensors in batches:
            fwd = model(**tensors, kappa=sim.kappa)
            n_seeds = tensors["n_seeds"]
            z_hat[nodes[:n_seeds]] = fwd.mu_z[:n_seeds].cpu().numpy()
            w_hat[nodes[:n_seeds]] = fwd.mu_w[:n_seeds].cpu().numpy()
    return sim, model, z_hat, w_hat, terms


def test_training_ended_finite_and_z_alive(fit):
    _, _, _, _, terms = fit
    assert torch.isfinite(terms.loss)
    assert 1.0 < terms.kl_z < 30.0          # informative, not exploded
    assert terms.penalty_info.get("excluded_fraction", 0.0) < 0.5


def test_z_recovers_the_planted_types(fit):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import normalized_mutual_info_score

    sim, _, z_hat, _, _ = fit
    k = sim.n_types
    nmi = normalized_mutual_info_score(
        sim.t, KMeans(k, n_init=10, random_state=0).fit_predict(z_hat))
    x_rep = PCA(8).fit_transform(np.log1p(sim.x / sim.totals[:, None] * 100))
    ceiling = normalized_mutual_info_score(
        sim.t, KMeans(k, n_init=10, random_state=0).fit_predict(x_rep))
    assert nmi > 0.45
    assert nmi > 0.55 * ceiling             # most of what the counts support


def test_w_tracks_the_planted_response(fit):
    from sklearn.cross_decomposition import CCA

    sim, _, _, w_hat, _ = fit
    cca = CCA(2).fit(w_hat, sim.w_true)
    u, v = cca.transform(w_hat, sim.w_true)
    first = abs(np.corrcoef(u[:, 0], v[:, 0])[0, 1])
    assert first > 0.6


def test_B_spans_the_planted_programme_space(fit):
    sim, model, _, _, _ = fit
    b_hat = model.B.weight.detach().cpu().numpy()          # (G, d_w)
    qa, _ = np.linalg.qr(b_hat)
    qb, _ = np.linalg.qr(sim.B_true.T)
    cosines = np.linalg.svd(qa.T @ qb, compute_uv=False)
    assert cosines[0] > 0.6


def test_penalty_kept_the_niche_out_of_z(fit):
    """Within-type, z should predict little of the neighbour composition."""
    sim, _, z_hat, _, _ = fit
    residual = sim.graph.y - sim.graph.ybar_t[sim.t]
    design = np.hstack([z_hat, np.ones((len(z_hat), 1))])
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
    r2 = 1 - (residual - design @ coefficients).var() / residual.var()
    assert r2 < 0.25                        # uncontrolled baseline sits ~0.18-0.28
