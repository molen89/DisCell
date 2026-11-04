"""Planted-answer tests for the held-out reads: mask, NMI, probe, degeneracy, guard."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from discell.evaluate import guard as G
from discell.evaluate import mask as EM
from discell.evaluate import metrics as M
from discell.evaluate import probe as P
from discell.evaluate.mi import mi_discrete_continuous, within_type_permutation

# -- the evaluation mask ---------------------------------------------------------

NAMES = np.array(["B cells", "T cells", "Unassigned"])


def test_mask_excludes_the_named_type_case_insensitively():
    t = np.array([0, 1, 2, 2, 1])
    assert EM.metric_target_mask(t, NAMES).tolist() == [True, True, False, False, True]
    assert EM.metric_target_mask(t, NAMES, exclude=("unassigned",)).tolist() == \
        [True, True, False, False, True]
    assert EM.metric_target_mask(t, NAMES, exclude=()).all()
    assert EM.metric_target_mask(t, NAMES, exclude=("T cells",)).tolist() == \
        [True, False, True, True, False]


def test_nmi_inputs_compact_the_labels_and_are_inert_without_exclusions():
    z = np.arange(10.0)[:, None]
    t = np.array([0, 2, 1, 2, 0, 1, 2, 0, 1, 1])
    z_n, t_n = EM.nmi_inputs(z, t, NAMES)
    assert len(z_n) == 7 and set(t_n.tolist()) == {0, 1}
    z_all, t_all = EM.nmi_inputs(z, t, NAMES, exclude=())
    assert z_all is z and t_all is t
    names = np.array(["a", "b", "c"])
    assert EM.nmi_inputs(z, t, names)[0] is z


def test_mask_record_states_the_exclusion_and_the_thread_pin():
    record = EM.mask_record(NAMES, np.array([0, 2, 2]))
    assert record["excluded_types"] == ["Unassigned"]
    assert record["excluded_present"] == ["Unassigned"]
    assert record["n_cells_excluded"] == 2 and record["kmeans_threads"] == 1


# -- NMI -------------------------------------------------------------------------

def test_nmi_is_one_on_separated_types_and_deterministic():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 4, 2000)
    z = 10.0 * np.eye(4)[t] + 0.1 * rng.standard_normal((2000, 4))
    assert M.z_type_nmi(z, t, seed=0) == pytest.approx(1.0)
    noisy = rng.standard_normal((2000, 4))
    assert M.z_type_nmi(noisy, t, seed=1) == M.z_type_nmi(noisy, t, seed=1)
    assert M.z_type_nmi(noisy, t, seed=1) < 0.05


# -- the pooled and the per-block probe -------------------------------------------

def _world(seed: int = 0, n: int = 6000, k: int = 5, n_img: int = 8):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, k, n)
    alpha = rng.uniform(0.5, 2.0, (k, k))
    y = np.stack([rng.dirichlet(alpha[g]) for g in t])
    comp = y[:, :-1]
    img = 3.0 * rng.standard_normal((k, n_img))[t] + 3.0 * rng.standard_normal((n, n_img))
    train = rng.random(n) < 0.7
    return rng, t, comp, img, train, ~train


def _vbar(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack([v[t == g].mean(axis=0) for g in range(t.max() + 1)])


def _composition_only_z(rng, t, comp):
    resid = comp - _vbar(comp, t)[t]
    signal = resid @ rng.standard_normal((comp.shape[1], 2)) * 5.0
    return np.hstack([signal + 0.1 * rng.standard_normal(signal.shape),
                      rng.standard_normal((len(t), 4))])


def test_legacy_block_is_probe_delta_ce_exactly():
    rng, t, comp, img, train, test = _world(0)
    v = np.hstack([comp, img]).astype(np.float32)
    vbar = _vbar(v, t).astype(np.float32)
    z = rng.standard_normal((len(t), 6)).astype(np.float32)
    legacy = M.probe_delta_ce(z, t, v, vbar, train, test, seed=3)
    out = P.probe_gain_per_block(z, t, v, vbar, train, test, n_comp=comp.shape[1], seed=3)
    for key in ("delta_ce", "noise_floor", "baseline_ce"):
        assert out["legacy"][key] == pytest.approx(legacy[key], abs=1e-10)
    assert out["n_comp"] == 4 and out["n_img"] == 8
    assert len(out["comp"]["floor_draws"]) == 5


def test_composition_only_latent_is_caught_in_the_composition_block():
    rng, t, comp, img, train, test = _world(1)
    v = np.hstack([comp, img])
    z = _composition_only_z(rng, t, comp)
    out = P.probe_gain_per_block(z, t, v, _vbar(v, t), train, test, n_comp=comp.shape[1],
                                 n_perm=10)
    assert out["comp"]["excess"] > 0.1 and not out["comp"]["pass_2sd_legacy"]
    assert abs(out["img"]["excess"]) < 0.01
    # var_fraction is the share of within-type variance, 1 - exp(-2 excess), below 1
    e = out["comp"]["excess"]
    assert out["comp"]["var_fraction"] == pytest.approx(1 - np.exp(-2 * e), abs=1e-12)
    assert 0 < out["comp"]["var_fraction"] < 1


def test_a_column_constant_on_held_out_cells_is_not_graded():
    rng, t, comp, img, train, test = _world(5)
    comp = comp.copy()
    comp[test, 0] = 0.0
    v = np.hstack([comp, img])
    z = rng.standard_normal((len(t), 6))
    out = P.probe_gain_per_block(z, t, v, _vbar(v, t), train, test, n_comp=comp.shape[1])
    assert out["comp"]["constant_cols"] == [0]
    assert out["comp"]["n_cols"] == comp.shape[1] - 1
    assert np.isfinite(out["comp"]["gain"])


def test_probe_blocks_both_graders_and_an_image_free_block():
    rng, t, comp, img, train, test = _world(2, n=3000)
    z = rng.standard_normal((len(t), 4))
    out = P.probe_blocks(z, t, comp, _vbar(comp, t), train, test, n_comp=comp.shape[1],
                         n_perm=2)
    assert out["ridge"]["img"] is None and out["mlp"]["img"] is None
    assert set(out["pass_2sd_legacy"]) == {"ridge_comp", "mlp_comp"}
    assert out["mlp"]["family"] == "mlp" and np.isfinite(out["mlp"]["comp"]["excess"])
    with pytest.raises(ValueError):
        P.probe_blocks(z, t, comp, _vbar(comp, t), train, test, n_comp=comp.shape[1],
                       n_perm=1)


def test_mlp_probe_is_seeded_and_leaves_the_global_torch_rng_alone():
    rng = np.random.default_rng(0)
    design = rng.standard_normal((500, 3))
    target = design[:, :1] ** 2
    rows, test = np.arange(400), np.arange(400, 500)
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    a = P.mlp_fit_predict(design, target, rows, test, seed=4)
    after = torch.rand(1)
    assert torch.equal(before, after)
    assert np.array_equal(a, P.mlp_fit_predict(design, target, rows, test, seed=4))


@pytest.mark.slow
def test_nonlinear_dependence_is_caught_by_the_mlp_not_the_ridge():
    rng, t, comp, img, train, test = _world(3, n=12_000)
    z = rng.standard_normal((len(t), 4))
    product = z[:, 0] * z[:, 1]
    comp = comp + 0.05 * product[:, None] * rng.choice([-1.0, 1.0], comp.shape[1])
    v = np.hstack([comp, img])
    ridge = P.probe_gain_per_block(z, t, v, _vbar(v, t), train, test,
                                   n_comp=comp.shape[1], n_perm=20)
    mlp = P.probe_gain_per_block_mlp(z, t, v, _vbar(v, t), train, test,
                                     n_comp=comp.shape[1], n_perm=20)
    assert ridge["comp"]["pass_2sd_legacy"]
    assert mlp["comp"]["excess"] > 10 * mlp["comp"]["floor_sd"]


# -- degeneracy and reconstruction ------------------------------------------------

def _types(seed: int, n: int = 6000, k: int = 5):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, k, n)
    train = rng.random(n) < 0.7
    return rng, t, train, ~train


def test_z_equal_to_onehot_t_reads_degenerate():
    rng, t, train, test = _types(0)
    z = np.eye(t.max() + 1)[t] + 1e-3 * rng.standard_normal((len(t), t.max() + 1))
    out = M.type_degeneracy(z, t, train, test)
    assert out["mi_ratio"] > 0.97 and out["accuracy"] > 0.99
    assert out["within_var_fraction"] < 0.01


def test_z_independent_of_t_reads_blind():
    rng, t, train, test = _types(1)
    out = M.type_degeneracy(rng.standard_normal((len(t), 8)), t, train, test)
    assert abs(out["mi_ratio"]) < 0.03 and out["within_var_fraction"] > 0.97


def test_type_degeneracy_survives_a_type_absent_from_training():
    rng = np.random.default_rng(0)
    n, k = 600, 5
    t = rng.integers(0, k - 1, n)
    t[-1] = k - 1
    z = np.eye(k)[t] + 0.05 * rng.standard_normal((n, k))
    train = np.ones(n, bool)
    train[-100:] = False
    out = M.type_degeneracy(z, t, train, ~train, seed=0)
    assert np.isfinite(out["mi_ratio"]) and 0 <= out["accuracy"] <= 1


def test_type_means_fill_absent_types_with_the_global_mean():
    z = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]], dtype=np.float32)
    means = M.type_means(z, np.array([0, 0, 2]), n_types=4)
    assert means.dtype == np.float32
    np.testing.assert_allclose(means[0], [1.0, 1.0])
    np.testing.assert_allclose(means[1], z.mean(axis=0))


def test_type_profile_reference_recovers_the_type_rates_dense_or_sparse():
    rng = np.random.default_rng(4)
    n, g, k = 4000, 40, 4
    t = rng.integers(0, k, n)
    logits = rng.standard_normal((k, g))
    p_type = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    x = np.stack([rng.multinomial(200, p_type[i]) for i in t]).astype(np.float64)
    train = np.arange(n) < 3000
    ref = M.type_profile_reconstruction(sp.csr_matrix(x[train]), t[train], x[~train],
                                        t[~train])
    truth = M.held_out_reconstruction(x[~train], np.log(p_type[t[~train]]))
    assert abs(ref - truth) < 0.01
    sparse_test = M.type_profile_reconstruction(x[train], t[train],
                                                sp.csr_matrix(x[~train]), t[~train])
    assert sparse_test == pytest.approx(ref, abs=1e-9)


def test_mirror_reads_context_that_predicts_z_within_type():
    rng, t, _, _ = _types(2, n=3000)
    c = rng.standard_normal((3000, 5))
    z = c[:, :2] + 0.1 * rng.standard_normal((3000, 2)) + np.eye(5)[t][:, :2]
    out = M.mirror_r2(z, c, t)
    assert out["r2"] > 0.9 and abs(out["r2_permuted"]) < 0.05


# -- the dead-context-channel checks -----------------------------------------------

def _planted(n=3000, d_w=6, n_types=5, n_niches=10, collapsed=True, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, n_types, n)
    niche = rng.integers(0, n_niches, n)
    w = rng.normal(scale=5.0, size=(n_types, d_w))[t]
    if not collapsed:
        w = w + rng.normal(size=(n_niches, d_w))[niche] + 0.1 * rng.normal(size=(n, d_w))
    return w, niche, t


def test_collapsed_w_is_flagged():
    w, niche, t = _planted(collapsed=True, seed=1)
    out = G.w_channel_guard(w, niche, t, seed=1, perms=8)
    assert out["w_niche_mi"] == 0.0 and out["niche_information_check"] <= 0.0
    assert out["dead_context_channel"] is True and out["flag"] == G.DEAD_W_FLAG
    assert out["w_var_fraction_across_cells"] < 1e-9


def test_live_w_clears_its_floor_and_isolated_cells_are_dropped():
    w, niche, t = _planted(collapsed=False)
    out = G.w_channel_guard(w, niche, t, seed=0, perms=3)
    assert out["niche_information_check"] > 0.0 and out["flag"] is None
    assert 0.0 < out["w_var_fraction_across_cells"] < 1.0
    niche = niche.copy()
    niche[:1500] = -1
    assert G.w_channel_guard(w, niche, t, seed=0, perms=3)["n_cells"] == 1500


def test_kl_w_detector_uses_the_sum_over_the_opening_epochs():
    assert G.kl_w_is_dead([1e-6] * 6, epoch=4)
    assert not G.kl_w_is_dead([1e-6] * 6 + [1e-3], epoch=4)
    assert not G.kl_w_is_dead([0.0] * 6, epoch=G.DEAD_W_MAX_EPOCH)
    history = [{"epoch": 4, "kl_w_per_dim": [0.0] * 6},
               {"epoch": 9, "kl_w_per_dim": [0.1] * 6}]
    assert G.dead_w_channel(history)
    assert not G.dead_w_channel(history[1:])


def test_niche_labels_mark_isolated_cells_and_are_deterministic():
    rng = np.random.default_rng(0)
    y = rng.dirichlet(np.ones(4), 500)
    degrees = np.ones(500, dtype=np.int32)
    degrees[:7] = 0
    a = G.niche_labels(y, degrees, 5, seed=0)
    assert (a[:7] == -1).all() and (a[7:] >= 0).all()
    assert np.array_equal(a, G.niche_labels(y, degrees, 5, seed=0))


def test_mi_estimator_reads_zero_for_independence_and_more_for_dependence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 3000)
    independent = rng.standard_normal((3000, 2))
    dependent = independent + 2.0 * y[:, None]
    assert mi_discrete_continuous(independent, y, rng=np.random.default_rng(1)) < 0.02
    assert mi_discrete_continuous(dependent, y, rng=np.random.default_rng(1)) > 0.5
    t = rng.integers(0, 3, 3000)
    shuffled = within_type_permutation(y, t, np.random.default_rng(2))
    for g in range(3):
        assert np.array_equal(np.sort(shuffled[t == g]), np.sort(y[t == g]))
