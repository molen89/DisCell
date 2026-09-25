"""The fixed false-positive floor (todo 8.15b; devlog 2026-09-24 21:20, B).

``p_i = (1 - kappa_i - eta_i) rho_i + kappa_i rho_bar_i + eta_i u``, u = 1/G,
``eta_i = min(lambda_i / l_i, 0.2)``. Planted: off is the pinned mixture bit
for bit; on with lambda = 0 is too, down to the trained weights; rows sum to
one under every leak form and match the formula in closed form; the cap
binds where lambda / l exceeds it; lambda is each control kind's per-feature
rate times the panel size.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from discell.model import fp_floor as F
from discell.model.equations import EPS, leakage_mix
from discell.model.networks import DisCell
from tests.test_model_ablations import _areas, _has_leak_edge, _leak_forward
from tests.test_model_elbo import tiny_tile

G = 25
SHARE = np.random.default_rng(5).uniform(0.05, 0.7, G)
LEAK_FORMS = ({}, {"kappa_mode": "depth", "kappa_ratio_mean": 1.2},
              {"kappa_mode": "gene", "kappa_gene_share": SHARE},
              {"kappa_mode": "density", "kappa_ratio_mean": 1.2,
               "area": _areas(10)})


def _fp_forward(eta=None, seed=3, area=None, **model_kw):
    """:func:`_leak_forward` with an optional eta: same init and noise draws."""
    torch.manual_seed(seed)
    _, tensors, batch = tiny_tile()
    torch.manual_seed(seed)
    model = DisCell(G, 3, 4, median_counts=100.0, d_z=4, d_w=2, hidden=16,
                    gat_dim=6, **model_kw)
    if area is not None:
        tensors = dict(tensors, area=area)
    if eta is not None:
        tensors = dict(tensors, eta=eta)
    torch.manual_seed(seed + 100)
    return model, tensors, batch, model(**tensors, kappa=0.1)


def _eta(n, seed=0):
    return torch.as_tensor(np.random.default_rng(seed).uniform(0.0, 0.2, n),
                           dtype=torch.float32)[:, None]


# -- off / lambda = 0: the pinned mixture ---------------------------------------

def test_off_is_the_pinned_mixture_bit_for_bit():
    for kw in LEAK_FORMS:
        _, _, batch, pinned = _leak_forward(**kw)
        _, _, _, off = _fp_forward(**kw)
        assert off.eta is None
        assert torch.equal(off.log_p, pinned.log_p), kw
        assert torch.equal(off.log_p_breve, pinned.log_p_breve), kw
    # the function itself: eta=None is the pre-floor expression, op for op
    n = batch.n_seeds
    rho, bar = pinned.log_rho[:n].exp(), pinned.rho_bar
    mix = 0.9 * rho + 0.1 * bar
    mix = mix / mix.sum(dim=-1, keepdim=True).clamp(min=EPS)
    assert torch.equal(leakage_mix(rho, bar, 0.1), mix.clamp(min=EPS).log())
    assert torch.equal(leakage_mix(rho, bar, 0.1, None),
                       leakage_mix(rho, bar, 0.1))


def test_on_with_lambda_zero_is_bit_identical_under_every_leak_form():
    for kw in LEAK_FORMS:
        _, _, batch, off = _fp_forward(**kw)
        _, _, _, zero = _fp_forward(eta=torch.zeros(batch.n_seeds, 1), **kw)
        assert torch.equal(zero.log_p, off.log_p), kw
        assert torch.equal(zero.log_p_breve, off.log_p_breve), kw
    # lambda = 0 is eta = 0 everywhere, a cell with no counts included
    assert np.array_equal(F.eta_from_lambda(0.0, [0.0, 3.0, 500.0]),
                          np.zeros(3))


# -- on: the formula, rows on the simplex ---------------------------------------

def test_rows_sum_to_one_under_every_leak_form():
    for kw in LEAK_FORMS:
        _, tensors, batch, fwd = _fp_forward(eta=_eta(7), **kw)
        n = batch.n_seeds
        for log_p in (fwd.log_p, fwd.log_p_breve):
            assert torch.allclose(log_p.exp().sum(-1), torch.ones(n),
                                  atol=1e-5), kw
            assert (log_p.exp() > 0).all(), kw


def test_on_is_the_preregistered_mixture_connected_and_isolated():
    """Planted: a connected seed decodes to (1 - kappa_i - eta_i) rho +
    kappa_i rho_bar + eta_i / G, an isolated one to (1 - eta_i) rho +
    eta_i / G, under the global and the per-cell (depth) forms."""
    for kw in ({}, {"kappa_mode": "depth", "kappa_ratio_mean": 1.2}):
        eta = _eta(7, seed=1)
        _, tensors, batch, fwd = _fp_forward(eta=eta, **kw)
        n = batch.n_seeds
        rho = fwd.log_rho[:n].exp().double()
        bar = fwd.rho_bar.double()
        k = (torch.full((n, 1), 0.1, dtype=torch.float64) if not kw
             else fwd.kappa_eff.double())
        e = eta.double()
        edge = _has_leak_edge(tensors, n)[:, None]
        assert (~edge).any() and edge.any()
        planted = torch.where(edge, (1 - k - e) * rho + k * bar + e / G,
                              (1 - e) * rho + e / G)
        assert torch.allclose(fwd.log_p.exp().double(), planted, atol=1e-6), kw
        # the floor is visible: every gene gets at least eta_i / G
        assert (fwd.log_p.exp() >= eta / G - 1e-7).all()


def test_eta_is_lambda_over_depth_capped_at_one_fifth():
    lam = 6.5
    totals = np.array([0.0, 1.0, 10.0, 32.5, 33.0, 100.0, 1000.0])
    got = F.eta_from_lambda(lam, totals)
    want = np.array([0.2, 0.2, 0.2, 0.2, lam / 33.0, lam / 100.0, lam / 1000.0])
    assert np.allclose(got, want, rtol=0, atol=1e-15)
    assert got.max() == F.ETA_CAP == 0.2
    assert (got[:4] == 0.2).all() and (got[4:] < 0.2).all()
    # the area form: proportional to A_i, the section total n * lambda fixed
    area = np.array([10.0, 20.0, 30.0, 40.0])
    lam_i = F.per_cell_lambda(lam, area, per_cell_area=True)
    assert np.isclose(lam_i.sum(), 4 * lam)
    assert np.allclose(lam_i / area, lam_i[0] / area[0])
    assert np.array_equal(F.per_cell_lambda(lam, area, per_cell_area=False),
                          np.full(4, lam))


def test_lambda_is_each_kind_s_per_feature_rate_times_the_panel():
    counts = {"control_probe_counts": np.array([0, 0, 1, 0.0]),       # 0.25
              "control_codeword_counts": np.array([0, 2, 0, 0.0]),    # 0.5
              "genomic_control_counts": np.array([1, 1, 0, 0.0])}     # 0.5
    n_features = {"Negative Control Probe": 40,
                  "Negative Control Codeword": 600,
                  "Genomic Control": 20, "Gene Expression": 5000}
    rec = F.section_lambda(counts, n_features, 5000)
    parts = rec["components"]
    assert np.isclose(parts["control_probe_counts"]["lambda"], 0.25 * 5000 / 40)
    assert np.isclose(parts["control_codeword_counts"]["lambda"],
                      0.5 * 5000 / 600)
    assert np.isclose(parts["genomic_control_counts"]["lambda"], 0.5 * 5000 / 20)
    assert np.isclose(rec["lambda"], 0.25 * 125 + 0.5 * 5000 / 600 + 125.0)


def test_area_rule_needs_every_section_at_the_threshold():
    def section(name, rho):
        return {"dataset": name, "variant": "v",
                "area_test": {"spearman": {"weighted_mean": rho}}}
    assert F.area_decision([section("a", 0.31), section("b", 0.30)])[
        "per_cell_area"]
    assert not F.area_decision([section("a", 0.9), section("b", 0.29)])[
        "per_cell_area"]
    # the statistic: a within-type rank correlation, cell-weighted over types
    rng = np.random.default_rng(0)
    area = rng.uniform(10, 100, 2000)
    t = np.repeat([0, 1], 1000)
    control = np.where(t == 0, area, -area) + rng.normal(0, 1e-3, 2000)
    rec = F.within_type_spearman(control, area, t)
    assert np.isclose(rec["weighted_mean"], 0.0, atol=1e-6)
    assert np.isclose(rec["per_type"][0], 1.0) and np.isclose(rec["per_type"][1], -1.0)


# -- the trainer: eta carried per seed, recorded, and the loss kept -------------

def _planted_controls(data, lam_counts=(0.02, 0.01, 0.2)):
    """A cell table for the synthetic slide: per-kind counts whose means are
    *lam_counts*, 40 / 600 / 20 features, areas 20-200."""
    n = data.x.shape[0]
    rng = np.random.default_rng(3)
    counts = {k: rng.poisson(m, n).astype(np.float64)
              for k, m in zip(F.CONTROL_KINDS, lam_counts)}
    return {"counts": counts, "area": rng.uniform(20.0, 200.0, n),
            "n_features": {"Negative Control Probe": 40,
                           "Negative Control Codeword": 600,
                           "Genomic Control": 20},
            "source": "planted"}


def _config(**kw):
    from discell.model.train import TrainConfig

    return TrainConfig(dataset="synthetic-smoke", kappa=0.1, d_z=6, d_w=2,
                       hidden=32, gat_dim=8, epochs=2, eval_every=1,
                       figures_every=1000, patience=100, device="cpu",
                       v_pcs=4, invariance="closed_form", **kw)


@pytest.fixture
def fp_env(tmp_path, monkeypatch):
    from discell import paths
    from tests.test_model_train import _small_data

    monkeypatch.setattr(paths, "dataset",
                        lambda _: type("D", (), {"root": tmp_path})())
    data = _small_data()
    controls = _planted_controls(data)
    calls = []

    def fake_read(dataset, variant):
        calls.append((dataset, variant))
        return controls

    monkeypatch.setattr(F, "read_cell_controls", fake_read)
    return tmp_path, data, controls, calls


def test_trainer_off_carries_no_eta_and_reads_no_cell_table(fp_env):
    from discell.model.train import Trainer

    _, data, _, calls = fp_env
    trainer = Trainer(_config(run_name="pinned"), data)
    assert trainer.fp_eta is None
    assert "eta" not in trainer.train_batches[0]
    assert "eta" not in Trainer._forward_kwargs(trainer.train_batches[0])
    assert not calls
    assert (trainer.config.fp_floor, trainer.config.fp_lambda) == (False, None)


def test_trainer_on_with_lambda_zero_trains_the_pinned_weights_bit_for_bit(fp_env):
    from discell.model.train import Trainer

    tmp_path, data, _, _ = fp_env
    # one thread: multithreaded CPU scatter-add backward reorders float sums,
    # so two pinned fits differ at 1e-9 in the GAT weights; single-threaded
    # they agree bit for bit, which is what makes this comparison exact
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        pinned = Trainer(_config(run_name="pinned"), data)
        a = pinned.fit()
        zero = Trainer(_config(run_name="fp_zero", fp_floor=True,
                               fp_lambda=0.0), data)
        assert np.array_equal(zero.fp_eta, np.zeros(data.x.shape[0]))
        b = zero.fit()
    finally:
        torch.set_num_threads(threads)
    assert a["final"]["recon_val"] == b["final"]["recon_val"]
    sa, sb = pinned.model.state_dict(), zero.model.state_dict()
    assert list(sa) == list(sb)                   # no buffer, no parameter added
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_trainer_on_carries_eta_per_seed_and_records_lambda(fp_env):
    from discell.model.train import TrainConfig, Trainer, build_parser

    tmp_path, data, controls, calls = fp_env
    args = vars(build_parser().parse_args(["--dataset", "x", "--fp-floor"]))
    args.pop("quiet")
    assert TrainConfig(**args).fp_floor and not TrainConfig(**args).fp_area

    trainer = Trainer(_config(run_name="fp_on", fp_floor=True), data)
    lam = F.section_lambda(controls["counts"], controls["n_features"],
                           data.x.shape[1])["lambda"]
    assert np.isclose(trainer.config.fp_lambda, lam)
    want = np.minimum(lam / data.totals, 0.2)
    assert np.allclose(trainer.fp_eta, want)
    assert trainer.config.fp_cap_share == float((want >= 0.2).mean())
    batch = trainer.train_batches[0]
    n = batch["n_seeds"]
    assert batch["eta"].shape == (n, 1)
    assert np.allclose(batch["eta"][:, 0].numpy(),
                       want[np.asarray(batch["nodes"])[:n]], atol=1e-7)
    # the decode path keeps the floor: own z and w decode to log_p exactly
    with torch.no_grad():
        fwd = trainer.model(**Trainer._forward_kwargs(batch), kappa=0.1,
                            sample=False)
        again = trainer._decode_seeds(fwd, fwd.mu_z[:n], n)
    assert fwd.eta is batch["eta"]
    # log_rho recomputed on the seeds alone: same numbers up to matmul order
    assert torch.allclose(again, fwd.log_p, atol=1e-6)
    no_floor = leakage_mix(trainer.model.log_rho(fwd.mu_z[:n], fwd.mu_w[:n])
                           .exp(), fwd.rho_bar, fwd.kappa_eff)
    assert not torch.allclose(again, no_floor, atol=1e-4)

    summary = trainer.fit()
    assert np.isfinite(summary["final"]["recon_val"])
    run_dir = tmp_path / "runs" / "fp_on"
    stored = json.loads((run_dir / "config.json").read_text())
    assert stored["fp_floor"] is True and stored["fp_area"] is False
    assert np.isclose(stored["fp_lambda"], lam)
    assert stored["fp_cap_share"] == trainer.config.fp_cap_share
    record = json.loads((run_dir / "fp_floor.json").read_text())
    assert record["eta_cap"] == 0.2 and np.isclose(record["lambda_used"], lam)
    # a reload rebuilds the same config, and with it the same eta
    rebuilt = TrainConfig(**{k: v for k, v in stored.items() if k != "git"})
    assert rebuilt == trainer.config
    assert np.array_equal(Trainer(rebuilt, data).fp_eta, trainer.fp_eta)


# -- the post-hoc reads decode with the same floor ------------------------------

def test_transport_mixtures_take_the_floor_and_stay_pinned_without_it():
    from types import SimpleNamespace

    from discell.model.transport import _decode_cells, group_eta, leak_rate

    rng = np.random.default_rng(2)
    rho = rng.dirichlet(np.ones(G))
    bar = rng.dirichlet(np.ones(G))
    k = 0.1
    assert np.array_equal(leak_rate(rho, bar, k), (1 - k) * rho + k * bar)
    assert group_eta({"rho": np.zeros((2, G))}, 1) is None
    assert group_eta({"eta": np.array([[0.05], [0.15]])}, 1) == 0.15
    got = leak_rate(rho, bar, k, 0.15)
    assert np.allclose(got, (1 - k - 0.15) * rho + k * bar + 0.15 / G)
    assert np.isclose(got.sum(), 1.0)

    model, tensors, batch, fwd = _fp_forward()
    trainer = SimpleNamespace(model=model.eval())
    n = batch.n_seeds
    mu_z, w = fwd.mu_z[:n].detach().numpy(), fwd.mu_w[:n].detach().numpy()
    eta = _eta(n, seed=4)
    with torch.no_grad():
        rho_t = model.log_rho(fwd.mu_z[:n], fwd.mu_w[:n]).exp()
    want = leakage_mix(rho_t, fwd.rho_bar, 0.1, eta).exp().numpy()
    got = _decode_cells(trainer, mu_z, w, fwd.rho_bar.numpy(), 0.1,
                        eta=eta[:, 0].numpy())
    assert np.array_equal(got, want)
    assert np.array_equal(_decode_cells(trainer, mu_z, w, fwd.rho_bar.numpy(),
                                        0.1),
                          leakage_mix(rho_t, fwd.rho_bar, 0.1).exp().numpy())
