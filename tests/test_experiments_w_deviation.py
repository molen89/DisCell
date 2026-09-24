"""The alpha_w ladder's deviation read (review R19) on planted worlds.

A real DisCell on the synthetic tissue, with the decoder's z path zeroed and
kappa = 0, so a cell's rate is ``softmax(b + B w)`` and nothing else. The
posterior mean is planted: ``mu_w = m_psi(c, t) + d`` with ``d`` fixed per
cell. Counts are then drawn from the model at a chosen true w, so the held-out
gain has a known sign:

* planted signal -- counts drawn at ``w = m_psi + d``, read at the same:
  the deviation carries the cell, gain > 0;
* no signal, posterior on its prior -- counts at ``m_psi``, ``d = 0``: the two
  decodes coincide, gain = 0;
* no signal, posterior off its prior -- counts at ``m_psi``, read with ``d``
  that has nothing to do with the counts: gain < 0, never credited.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from discell.experiments.w_deviation import (collect, deviation_read,
                                             within_type_mirror)
from discell.model.prepare import ModelData, spatial_tiles
from discell.model.synthetic import simulate
from discell.model.train import TrainConfig, Trainer


def _data() -> ModelData:
    sim = simulate(n_cells=1500, n_types=4, kappa=0.1, seed=0)
    k, t = sim.n_types, sim.t.astype(np.int64)
    v = np.hstack([sim.graph.y[:, :-1], sim.phi[:, :4]]).astype(np.float32)
    tiles = spatial_tiles(sim.positions, 300)
    return ModelData(
        graph=sim.graph, x=sp.csr_matrix(sim.x), t=t, phi=sim.phi,
        positions=sim.positions, totals=sim.totals,
        median_counts=float(np.median(sim.totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=np.array([f"type{g}" for g in range(k)]),
        v_block=v, vbar_t=np.stack([v[t == g].mean(axis=0) for g in range(k)]),
        train_tiles=tiles[:-2], val_tiles=tiles[-2:])


def _trainer(tmp_path, data: ModelData, state: dict | None = None) -> Trainer:
    from discell import paths

    config = TrainConfig(
        dataset="synthetic-smoke", kappa=0.0, d_z=6, d_w=2, hidden=32,
        gat_dim=8, epochs=1, eval_every=1, figures_every=1, device="cpu",
        alpha_z=0.007, alpha_w=0.1, alpha_a=0.3, v_pcs=4,
        invariance="closed_form")
    real = paths.dataset
    paths.dataset = lambda _: type("D", (), {"root": tmp_path})()
    try:
        trainer = Trainer(config, data)
    finally:
        paths.dataset = real
    if state is None:
        with torch.no_grad():
            last = [m for m in trainer.model.dec_a.modules()
                    if isinstance(m, torch.nn.Linear)][-1]
            last.weight.zero_()                        # rates ignore z
            trainer.model.B.weight.normal_(0.0, 1.0,   # w matters, visibly
                                           generator=torch.Generator().manual_seed(1))
    else:
        trainer.model.load_state_dict(state)
    trainer.model.eval()
    return trainer


def _plant(trainer: Trainer, d: np.ndarray) -> None:
    """``mu_w := m_psi(c, t) + d[cell]`` for every context node of a tile."""
    current: dict = {}
    kwargs_of, forward_of = trainer._forward_kwargs, trainer.model.forward

    def kwargs(batch):
        current["nodes"] = np.asarray(batch["nodes"])
        return kwargs_of(batch)

    def forward(*args, **kw):
        fwd = forward_of(*args, **kw)
        rows = current["nodes"][:fwd.mu_w.shape[0]]
        fwd.mu_w = fwd.prior_mean_w + torch.as_tensor(d[rows],
                                                      dtype=fwd.mu_w.dtype)
        return fwd

    trainer._forward_kwargs = kwargs
    trainer.model.forward = forward


def _redraw(trainer: Trainer, data: ModelData, seed: int) -> ModelData:
    """Counts drawn from the (planted) model's own decode at w = mu_w."""
    rates = np.zeros(data.x.shape, dtype=np.float64)
    with torch.no_grad():
        for batch in trainer.train_batches + trainer.val_batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch), kappa=0.0,
                                sample=False)
            n = batch["n_seeds"]
            rates[batch["nodes"][:n]] = trainer._decode_seeds(
                fwd, fwd.mu_z[:n], n, fwd.mu_w[:n]).exp().double().numpy()
    rng = np.random.default_rng(seed)
    depth = np.asarray(data.totals).round().astype(int).clip(min=20)
    x = np.stack([rng.multinomial(depth[i], rates[i] / rates[i].sum())
                  for i in range(len(depth))]).astype(np.float32)
    return dataclasses.replace(data, x=sp.csr_matrix(x),
                               totals=x.sum(axis=1))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wdev")
    data = _data()
    base = _trainer(tmp, data)
    state = {k: v.clone() for k, v in base.model.state_dict().items()}
    d = np.random.default_rng(7).normal(0.0, 1.0, (data.graph.n_cells, 2))

    def read(true_d: np.ndarray, read_d: np.ndarray) -> dict:
        gen = _trainer(tmp, data, state)
        _plant(gen, true_d)
        redrawn = _redraw(gen, data, seed=3)
        reader = _trainer(tmp, redrawn, state)
        _plant(reader, read_d)
        return deviation_read(reader)

    zero = np.zeros_like(d)
    return {"signal": read(d, d), "prior": read(zero, zero),
            "noise": read(zero, d), "data": data, "state": state, "tmp": tmp}


def test_a_planted_deviation_does_held_out_work(world):
    r = world["signal"]["recon"]
    assert r["gain"] > 0.05
    assert r["gain_frac_cells_positive"] > 0.8
    assert r["mu_w"] > r["prior_w"]


def test_a_posterior_on_its_prior_has_no_gain(world):
    r = world["prior"]["recon"]
    assert abs(r["gain"]) < 1e-9
    assert world["prior"]["deviation"]["mean_sq_norm"] < 1e-12
    assert np.isnan(world["prior"]["d_mirror"]["r2"])   # d = 0 explains nothing


def test_a_deviation_unrelated_to_the_counts_is_never_credited(world):
    assert world["noise"]["recon"]["gain"] < -0.05
    # and the planted world's gain is not the absence of that loss
    assert world["signal"]["recon"]["gain"] > abs(
        world["prior"]["recon"]["gain"]) + 0.05


def test_the_read_is_on_the_held_out_cells(world):
    data = world["data"]
    n_val = sum(len(tile) for tile in data.val_tiles)
    assert world["signal"]["n_val_cells"] == n_val
    assert world["signal"]["n_train_cells"] == data.graph.n_cells - n_val


def test_the_mu_w_decode_is_the_models_own_log_p(world):
    trainer = _trainer(world["tmp"], world["data"], world["state"])
    out = collect(trainer, trainer.val_batches)
    np.testing.assert_allclose(out["ll_mu_w"], out["ll_forward"], atol=1e-6)
    # an unplanted model's deviation is its encoder's, not zero
    assert np.abs(out["mu_w"] - out["prior_w"]).sum() > 0


def test_within_type_mirror_separates_a_planted_mirror_from_an_honest_w():
    rng = np.random.default_rng(0)
    n, k = 6000, 3
    t = rng.integers(0, k, n)
    offsets = rng.normal(0.0, 3.0, (k, 5))
    neighbour_z = rng.normal(size=(n, 5)) + offsets[t]
    train = rng.random(n) < 0.7
    mirror = (neighbour_z - offsets[t]) @ rng.normal(size=(5, 2)) \
        + 0.3 * rng.normal(size=(n, 2)) + offsets[t, :2]
    # an honest target: type offsets (shared with the design) plus noise --
    # within type there is nothing to find
    honest = offsets[t, :2] * 2.0 + rng.normal(size=(n, 2))
    planted = within_type_mirror(mirror, neighbour_z, t, train, ~train)
    null = within_type_mirror(honest, neighbour_z, t, train, ~train)
    assert planted["r2"] > 0.8 and planted["r2_floor"] < 0.02
    assert null["r2"] < 0.02 and abs(null["r2"] - null["r2_floor"]) < 0.02
