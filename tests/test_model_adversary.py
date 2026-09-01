"""The spec 4.6 escalation: heads, minimax wiring, zero-at-optimum."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from discell.model.elbo import adversary_terms
from discell.model.networks import Adversary, soft_cross_entropy

torch.manual_seed(0)


def test_soft_cross_entropy_matches_manual():
    target = torch.tensor([[0.2, 0.8]])
    log_pred = torch.log(torch.tensor([[0.5, 0.5]]))
    expected = -(0.2 * np.log(0.5) + 0.8 * np.log(0.5))
    assert float(soft_cross_entropy(target, log_pred)) == pytest.approx(expected)


def _toy(n=512, d_z=4, k=3, predictable=True, seed=0):
    rng = np.random.default_rng(seed)
    t = torch.tensor(rng.integers(0, k, n))
    z = torch.randn(n, d_z)
    if predictable:                       # y depends on z within type
        logits = torch.randn(d_z, k)
        y = torch.softmax(z @ logits * 2.0, dim=-1)
    else:                                 # y depends on t alone
        base = torch.softmax(torch.randn(k, k) * 2.0, dim=-1)
        y = base[t]
    ybar = torch.stack([y[t == g].mean(0) for g in range(k)])
    return z, t, y, ybar


def test_head_training_touches_only_head_parameters():
    z, t, y, ybar = _toy()
    z = z.requires_grad_(True)
    heads = Adversary(4, 3, 3, hidden=16)
    out = adversary_terms(heads, z, t, y, y, ybar, ybar)
    out.head_loss.backward()
    assert z.grad is None                             # detached path
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in heads.parameters())


def test_encoder_term_reaches_z():
    z, t, y, ybar = _toy()
    z = z.requires_grad_(True)
    heads = Adversary(4, 3, 3, hidden=16)
    out = adversary_terms(heads, z, t, y, y, ybar, ybar)
    out.encoder_term.backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0


def _train_heads(heads, z, t, y, ybar, steps=400):
    optimiser = torch.optim.Adam(heads.parameters(), lr=5e-3)
    for _ in range(steps):
        optimiser.zero_grad()
        adversary_terms(heads, z, t, y, y, ybar, ybar).head_loss.backward()
        optimiser.step()
    return adversary_terms(heads, z, t, y, y, ybar, ybar)


def test_zero_excess_when_z_carries_nothing_beyond_type():
    z, t, y, ybar = _toy(predictable=False, n=2048)
    out = _train_heads(Adversary(4, 3, 3, hidden=16), z, t, y, ybar)
    assert abs(out.excess_y) < 0.03                   # nothing to find


def test_positive_excess_when_z_predicts_the_niche():
    z, t, y, ybar = _toy(predictable=True, n=2048)
    out = _train_heads(Adversary(4, 3, 3, hidden=16), z, t, y, ybar)
    assert out.excess_y > 0.15                        # the leak is visible


def test_trainer_runs_in_adversary_mode(tmp_path):
    import scipy.sparse as sp

    from discell import paths
    from discell.model.prepare import ModelData, soft_clusters, spatial_tiles
    from discell.model.synthetic import simulate
    from discell.model.train import TrainConfig, Trainer

    sim = simulate(n_cells=1200, n_types=4, kappa=0.1, seed=0)
    k, t = sim.n_types, sim.t.astype(np.int64)
    v = np.hstack([sim.graph.y[:, :-1], sim.phi[:, :4]]).astype(np.float32)
    e_phi, _ = soft_clusters(sim.phi[:, :4], k)
    connected = sim.graph.degrees > 0
    data = ModelData(
        graph=sim.graph, x=sp.csr_matrix(sim.x), t=t, phi=sim.phi,
        positions=sim.positions, totals=sim.totals,
        median_counts=float(np.median(sim.totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=np.array([f"type{g}" for g in range(k)]),
        v_block=v,
        vbar_t=np.stack([v[(t == g) & connected].mean(0) for g in range(k)]),
        train_tiles=spatial_tiles(sim.positions, 300)[:-1],
        val_tiles=spatial_tiles(sim.positions, 300)[-1:],
        e_phi=e_phi,
        phibar_t=np.stack([e_phi[(t == g) & connected].mean(0) for g in range(k)]),
    )
    config = TrainConfig(dataset="adv-smoke", invariance="adversary",
                         alpha_a=0.05, kappa=0.1, d_z=6, d_w=2, hidden=32,
                         gat_dim=8, epochs=4, eval_every=2, figures_every=4,
                         patience=100, device="cpu", v_pcs=4)
    real = paths.dataset
    paths.dataset = lambda _: type("D", (), {"root": tmp_path})()
    try:
        trainer = Trainer(config, data)
        summary = trainer.fit()
    finally:
        paths.dataset = real
    assert np.isfinite(summary["final"]["recon_val"])
    assert trainer.adversary is not None
    # the heads were actually trained
    assert any(p.abs().sum() > 0 for p in trainer.adversary.parameters())
