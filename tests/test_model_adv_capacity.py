"""The adversary-capacity knobs (8.17; devlog 2026-09-24 21:20, part A).

``--adv-head-steps`` (= ``adv_steps``), ``--adv-head-width`` (=
``adv_hidden``), ``--adv-ensemble K`` and ``--adv-comp-weight c``. Pinned
here: at the defaults the trainer reproduces the pre-knob code bit for bit (a
verbatim copy of it lives below), each knob reaches exactly its consumer and
nothing else, and an ensemble of one is the single head pair.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from discell import paths
from discell.model.elbo import adversary_terms, ensemble_adversary_terms
from discell.model.networks import Adversary, DisCell, soft_cross_entropy
from discell.model.train import TrainConfig, Trainer, build_parser


# -- the pre-knob code, verbatim (ea36e69) ---------------------------------

def pinned_adversary_terms(heads, mu_z, t, y, e_phi, ybar_t, phibar_t):
    """``elbo.adversary_terms`` before the knobs."""
    log_y, log_phi = heads(mu_z, t)
    base_y = soft_cross_entropy(y, ybar_t.clamp(min=1e-8).log()[t])
    base_phi = soft_cross_entropy(e_phi, phibar_t.clamp(min=1e-8).log()[t])
    excess_y = base_y - soft_cross_entropy(y, log_y)
    excess_phi = base_phi - soft_cross_entropy(e_phi, log_phi)
    encoder_term = (excess_y + excess_phi).mean()
    log_y_sg, log_phi_sg = heads(mu_z.detach(), t)
    head_loss = (soft_cross_entropy(y, log_y_sg)
                 + soft_cross_entropy(e_phi, log_phi_sg)).mean()
    return (encoder_term, head_loss, float(excess_y.mean().detach()),
            float(excess_phi.mean().detach()))


def pinned_step(trainer, batch):
    """``Trainer._step`` before the knobs (adversary branch, mu_z input)."""
    import dataclasses as dc

    from discell.model.elbo import discell_loss

    config = trainer.config
    kwargs = trainer._forward_kwargs(batch)
    fwd = trainer.model(**kwargs, kappa=config.kappa)
    n = batch["n_seeds"]
    weights = config.weights(trainer.epoch)
    adv_feat = fwd.mu_z[:n]
    enc, _, excess_y, excess_phi = pinned_adversary_terms(
        trainer.adversary, adv_feat, batch["t"][:n], batch["y_seed"],
        batch["ephi_seed"], trainer.ybar_t, trainer.phibar_t)
    terms = discell_loss(fwd, kwargs["x"][:n], batch["t"][:n],
                         weights=dc.replace(weights, alpha_a=0.0))
    loss = terms.loss + weights.alpha_a * enc
    trainer.optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), config.grad_clip)
    trainer.optimiser.step()
    z_frozen = adv_feat.detach()
    for _ in range(config.adv_steps):
        trainer.adversary_optimiser.zero_grad()
        log_y, log_phi = trainer.adversary(z_frozen, batch["t"][:n])
        head_loss = (soft_cross_entropy(batch["y_seed"], log_y)
                     + soft_cross_entropy(batch["ephi_seed"], log_phi)).mean()
        head_loss.backward()
        trainer.adversary_optimiser.step()
    return (float(terms.loss.detach()), float(enc.detach()),
            float(head_loss.detach()), excess_y, excess_phi)


# -- fixtures ------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    import scipy.sparse as sp

    from discell.model.prepare import ModelData, soft_clusters, spatial_tiles
    from discell.model.synthetic import simulate

    sim = simulate(n_cells=1200, n_types=4, kappa=0.1, seed=0)
    k, t = sim.n_types, sim.t.astype(np.int64)
    v = np.hstack([sim.graph.y[:, :-1], sim.phi[:, :4]]).astype(np.float32)
    e_phi, _ = soft_clusters(sim.phi[:, :4], k)
    connected = sim.graph.degrees > 0
    tiles = spatial_tiles(sim.positions, 300)
    return ModelData(
        graph=sim.graph, x=sp.csr_matrix(sim.x), t=t, phi=sim.phi,
        positions=sim.positions, totals=sim.totals,
        median_counts=float(np.median(sim.totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=np.array([f"type{g}" for g in range(k)]),
        v_block=v,
        vbar_t=np.stack([v[(t == g) & connected].mean(0) for g in range(k)]),
        train_tiles=tiles[:-1], val_tiles=tiles[-1:], e_phi=e_phi,
        phibar_t=np.stack([e_phi[(t == g) & connected].mean(0)
                           for g in range(k)]))


@pytest.fixture
def one_thread():
    """Bit-identity needs a fixed reduction order: multithreaded CPU
    index_add_ in the GAT differs run to run by ~1e-10 even for the pinned
    code against itself."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def make(data, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "dataset",
                        lambda _: SimpleNamespace(root=tmp_path))

    def build(**knobs) -> Trainer:
        config = TrainConfig(dataset="adv-capacity", invariance="adversary",
                             alpha_a=0.3, kappa=0.1, d_z=6, d_w=2, hidden=32,
                             gat_dim=8, epochs=4, eval_every=2,
                             figures_every=4, patience=100, device="cpu",
                             v_pcs=4, seed=3, **knobs)
        return Trainer(config, data)
    return build


def _toy(n=256, d_z=5, k=3, e=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, k, (n,), generator=g)
    z = torch.randn(n, d_z, generator=g)
    y = torch.softmax(torch.randn(n, k, generator=g), dim=-1)
    ephi = torch.softmax(torch.randn(n, e, generator=g), dim=-1)
    ybar = torch.stack([y[t == c].mean(0) for c in range(k)])
    phibar = torch.stack([ephi[t == c].mean(0) for c in range(k)])
    return z, t, y, ephi, ybar, phibar


def _grads(module):
    return [p.grad.clone() for p in module.parameters()]


def _same_state(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


# -- defaults are the pinned code, bit for bit ---------------------------------

def test_parser_defaults_are_the_pinned_values():
    args = vars(build_parser().parse_args(["--dataset", "x"]))
    args.pop("quiet")
    config = TrainConfig(**args)
    assert (config.adv_steps, config.adv_hidden, config.adv_ensemble,
            config.adv_comp_weight) == (6, 64, 1, 1.0)
    assert dataclasses.asdict(config) == dataclasses.asdict(TrainConfig(dataset="x"))


def test_adversary_terms_default_is_pinned_bitwise():
    z, t, y, ephi, ybar, phibar = _toy()
    torch.manual_seed(0)
    heads = Adversary(5, 3, 4, hidden=16)
    za = z.clone().requires_grad_(True)
    zb = z.clone().requires_grad_(True)
    new = adversary_terms(heads, za, t, y, ephi, ybar, phibar)
    enc, head, ey, ep = pinned_adversary_terms(heads, zb, t, y, ephi, ybar, phibar)
    assert torch.equal(new.encoder_term, enc) and torch.equal(new.head_loss, head)
    assert (new.excess_y, new.excess_phi) == (ey, ep)
    heads.zero_grad()
    new.encoder_term.backward()
    g_new, gz_new = _grads(heads), za.grad.clone()
    heads.zero_grad()
    enc.backward()
    assert all(torch.equal(a, b) for a, b in zip(g_new, _grads(heads)))
    assert torch.equal(gz_new, zb.grad)


def test_trainer_defaults_build_the_pinned_heads(make, data):
    """Same RNG stream: seed -> DisCell -> one Adversary, as before."""
    trainer = make()
    assert isinstance(trainer.adversary, Adversary)
    config = trainer.config
    torch.manual_seed(config.seed)
    DisCell(n_genes=data.x.shape[1], n_types=len(data.p_t),
            phi_dim=data.phi.shape[1], median_counts=data.median_counts,
            d_z=config.d_z, d_w=config.d_w, hidden=config.hidden,
            gat_dim=config.gat_dim, heads=config.heads,
            gat_sources=config.gat_sources)
    reference = Adversary(config.d_z, len(data.p_t), data.e_phi.shape[1],
                          hidden=64)
    assert _same_state(trainer.adversary, reference)


def test_trainer_step_defaults_bit_identical_to_pinned(make, one_thread):
    a, b = make(), make()
    assert _same_state(a.model, b.model) and _same_state(a.adversary, b.adversary)
    torch.manual_seed(11)
    got = []
    for batch in a.train_batches[:3]:
        terms, extras = a._step(batch)
        got.append((terms.loss.detach(), terms.penalty, extras["adv_head_loss"],
                    extras["adv_excess_y"], extras["adv_excess_phi"]))
    torch.manual_seed(11)
    want = [pinned_step(b, batch) for batch in b.train_batches[:3]]
    assert [tuple(float(x) for x in g) for g in got] == want
    assert _same_state(a.model, b.model)
    assert _same_state(a.adversary, b.adversary)
    for opt_a, opt_b in ((a.optimiser, b.optimiser),
                         (a.adversary_optimiser, b.adversary_optimiser)):
        for sa, sb in zip(opt_a.state.values(), opt_b.state.values()):
            assert all(torch.equal(sa[k], sb[k]) for k in ("exp_avg", "exp_avg_sq"))


# -- each knob reaches exactly its consumer ---------------------------------------

@pytest.mark.parametrize("flags, field, value", [
    (["--adv-head-steps", "12"], "adv_steps", 12),
    (["--adv-steps", "12"], "adv_steps", 12),
    (["--adv-head-width", "128"], "adv_hidden", 128),
    (["--adv-hidden", "128"], "adv_hidden", 128),
    (["--adv-ensemble", "3"], "adv_ensemble", 3),
    (["--adv-comp-weight", "3"], "adv_comp_weight", 3.0),
])
def test_each_flag_sets_exactly_its_field(flags, field, value):
    args = vars(build_parser().parse_args(["--dataset", "x"] + flags))
    args.pop("quiet")
    got = dataclasses.asdict(TrainConfig(**args))
    base = dataclasses.asdict(TrainConfig(dataset="x"))
    assert {k for k in got if got[k] != base[k]} == {field}
    assert got[field] == value


def test_ensemble_must_be_positive():
    with pytest.raises(ValueError, match="adv_ensemble"):
        TrainConfig(dataset="x", adv_ensemble=0)


def _count_steps(optimiser):
    counter = {"n": 0}
    step = optimiser.step

    def counted(*args, **kwargs):
        counter["n"] += 1
        return step(*args, **kwargs)
    optimiser.step = counted
    return counter


@pytest.mark.parametrize("steps", [6, 12])
def test_head_steps_knob_counts_head_updates_only(make, steps):
    trainer = make(adv_steps=steps)
    heads, model = _count_steps(trainer.adversary_optimiser), _count_steps(trainer.optimiser)
    trainer._step(trainer.train_batches[0])
    assert (heads["n"], model["n"]) == (steps, 1)


def _widths(adversary: Adversary) -> list[int]:
    """out_features of every Linear, head_y then head_phi."""
    return [m.out_features for head in (adversary.head_y, adversary.head_phi)
            for m in head.modules() if isinstance(m, torch.nn.Linear)]


def test_head_width_knob_sizes_only_the_heads(make):
    default, wide = make(), make(adv_hidden=128)
    n_types, e_phi = len(default.data.p_t), default.data.e_phi.shape[1]
    assert _widths(default.adversary) == [64, 64, n_types, 64, 64, e_phi]
    assert _widths(wide.adversary) == [128, 128, n_types, 128, 128, e_phi]
    assert _same_state(default.model, wide.model)       # model init untouched
    assert default.config.weights() == wide.config.weights()


def test_ensemble_knob_builds_k_independent_pairs(make):
    default, ens = make(), make(adv_ensemble=3)
    assert isinstance(ens.adversary, torch.nn.ModuleList) and len(ens.adversary) == 3
    assert _same_state(ens.adversary[0], default.adversary)  # same stream
    assert not _same_state(ens.adversary[0], ens.adversary[1])
    assert not _same_state(ens.adversary[1], ens.adversary[2])
    n_single = sum(p.numel() for p in default.adversary.parameters())
    n_opt = sum(p.numel() for g in ens.adversary_optimiser.param_groups
                for p in g["params"])
    assert n_opt == 3 * n_single
    assert _same_state(default.model, ens.model)
    before = [{k: v.clone() for k, v in m.state_dict().items()} for m in ens.adversary]
    ens._step(ens.train_batches[0])
    for member, old in zip(ens.adversary, before):
        assert any(not torch.equal(v, old[k]) for k, v in member.state_dict().items())


def test_ensemble_of_one_is_the_single_pair():
    z, t, y, ephi, ybar, phibar = _toy()
    torch.manual_seed(0)
    heads = Adversary(5, 3, 4, hidden=16)
    single = adversary_terms(heads, z, t, y, ephi, ybar, phibar, comp_weight=2.0)
    one = ensemble_adversary_terms([heads], z, t, y, ephi, ybar, phibar,
                                   comp_weight=2.0)
    assert torch.equal(one.encoder_term, single.encoder_term)
    assert torch.equal(one.head_loss, single.head_loss)
    assert (one.excess_y, one.excess_phi) == (single.excess_y, single.excess_phi)


def test_ensemble_one_trainer_equals_default_trainer(make, one_thread):
    a, b = make(), make(adv_ensemble=1)
    torch.manual_seed(5)
    ta, ea = a._step(a.train_batches[0])
    torch.manual_seed(5)
    tb, eb = b._step(b.train_batches[0])
    assert ea == eb and float(ta.loss.detach()) == float(tb.loss.detach())
    assert _same_state(a.model, b.model) and _same_state(a.adversary, b.adversary)


def test_ensemble_penalises_the_mean_and_sums_the_heads():
    z, t, y, ephi, ybar, phibar = _toy()
    torch.manual_seed(0)
    members = [Adversary(5, 3, 4, hidden=16) for _ in range(3)]
    each = [adversary_terms(h, z, t, y, ephi, ybar, phibar) for h in members]
    ens = ensemble_adversary_terms(members, z, t, y, ephi, ybar, phibar)
    assert float(ens.encoder_term) == pytest.approx(
        np.mean([float(m.encoder_term) for m in each]), rel=1e-6)
    assert float(ens.head_loss) == pytest.approx(
        sum(float(m.head_loss) for m in each), rel=1e-6)
    assert ens.excess_y == pytest.approx(np.mean([m.excess_y for m in each]))
    # K copies of one pair: the encoder sees the single pair, the heads K-fold
    clones = [members[0]] * 3
    rep = ensemble_adversary_terms(clones, z, t, y, ephi, ybar, phibar)
    assert float(rep.encoder_term) == pytest.approx(float(each[0].encoder_term), rel=1e-6)
    assert float(rep.head_loss) == pytest.approx(3 * float(each[0].head_loss), rel=1e-6)


def test_comp_weight_scales_composition_in_both_halves():
    z, t, y, ephi, ybar, phibar = _toy()
    torch.manual_seed(0)
    heads = Adversary(5, 3, 4, hidden=16)
    out = adversary_terms(heads, z, t, y, ephi, ybar, phibar, comp_weight=3.0)
    with torch.no_grad():
        log_y, log_phi = heads(z, t)
        ex_y = (soft_cross_entropy(y, ybar.log()[t])
                - soft_cross_entropy(y, log_y))
        ex_phi = (soft_cross_entropy(ephi, phibar.log()[t])
                  - soft_cross_entropy(ephi, log_phi))
        ce_y, ce_phi = soft_cross_entropy(y, log_y), soft_cross_entropy(ephi, log_phi)
    assert float(out.encoder_term) == pytest.approx(float((3 * ex_y + ex_phi).mean()), rel=1e-6)
    assert float(out.head_loss) == pytest.approx(float((3 * ce_y + ce_phi).mean()), rel=1e-6)
    assert out.excess_y == pytest.approx(float(ex_y.mean()), rel=1e-6)  # logged unweighted
    assert out.excess_phi == pytest.approx(float(ex_phi.mean()), rel=1e-6)


def test_comp_weight_reaches_the_trainer_head_loop(make, one_thread):
    """One head update at c = 3 equals a hand-made Adam step on
    (3 CE_y + CE_phi).mean(); the encoder half is handed c too."""
    import copy

    trainer = make(adv_comp_weight=3.0, adv_steps=1)
    batch = trainer.train_batches[0]
    n = batch["n_seeds"]
    with torch.no_grad():
        z = trainer.model(**trainer._forward_kwargs(batch),
                          kappa=0.1, sample=False).mu_z[:n]
    t, y, e = batch["t"][:n], batch["y_seed"], batch["ephi_seed"]
    hand = copy.deepcopy(trainer.adversary)
    opt = torch.optim.Adam(hand.parameters(), lr=trainer.config.adv_lr)
    log_y, log_phi = hand(z, t)
    (3.0 * soft_cross_entropy(y, log_y) + soft_cross_entropy(e, log_phi)).mean().backward()
    opt.step()
    trainer._head_steps(z, t, y, e)
    assert _same_state(trainer.adversary, hand)

    seen = {}
    import discell.model.train as train_module
    real = train_module.ensemble_adversary_terms

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)
    train_module.ensemble_adversary_terms = spy
    try:
        trainer._step(batch)
    finally:
        train_module.ensemble_adversary_terms = real
    assert seen["comp_weight"] == 3.0
