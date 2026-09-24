"""Known-answer tests for the doc-08 section-7 transport instrument.

No trained model: the scorer, the pair ordering, the noise ceiling, the
band niches, the tier summary and the Phi-fixed channel are checked on
inputs where the answer is planted.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from discell.model.networks import DisCell
from discell.model.prepare import build_graph, tile_batch
from discell.model.train import Trainer
from discell.model.transport import (TUMOUR_BANDS, collect_channels,
                                     collect_own_p,
                                     distribution_scores,
                                     distribution_summary, noise_ceiling,
                                     pick_pairs, score_shift, tier_summary,
                                     restrict_renormalise, top_gene_overlap,
                                     tumour_band_labels, twin_scores,
                                     twin_summary)


def test_score_shift_perfect_prediction_and_planted_miscalibration():
    rng = np.random.default_rng(0)
    observed = rng.normal(size=400)
    perfect = score_shift(observed.copy(), observed)
    assert abs(perfect["r2"] - 1.0) < 1e-9
    assert abs(perfect["slope"] - 1.0) < 1e-9
    assert abs(perfect["corr"] - 1.0) < 1e-9
    # a per-panel constant (softmax normaliser, depth) is not charged
    shifted = score_shift(observed + 3.7, observed)
    assert abs(shifted["r2"] - 1.0) < 1e-9
    # right direction, twice the size: correlation 1, slope 1/2 and R^2
    # exactly 0 against the zero null (the size error costs the whole gain)
    twice = score_shift(2.0 * observed, observed)
    assert abs(twice["corr"] - 1.0) < 1e-9
    assert abs(twice["slope"] - 0.5) < 1e-9
    assert abs(twice["r2"]) < 1e-9
    # no prediction at all scores as the zero null
    flat = score_shift(np.zeros(400), observed)
    assert abs(flat["r2"]) < 1e-9
    assert np.isnan(flat["slope"])


def test_pick_pairs_orders_by_centroid_gap_and_skips_isolated():
    # three niches with planted compositions on a 3-simplex; niche 2 sits
    # far from both others, niches 0 and 1 are close
    comps = np.array([[0.6, 0.4, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    labels = np.repeat([0, 1, 2, -1], 20)
    y = np.vstack([comps[[0, 1, 2]].repeat(20, axis=0),
                   np.full((20, 3), 9.0)])          # isolated rows: garbage
    connected = labels >= 0
    pairs = pick_pairs(labels, y, connected)
    assert len(pairs) == 3
    assert pairs[-1] == (0, 1)                       # the closest pair last
    assert set(pairs[0]) in ({0, 2}, {1, 2})
    assert all(-1 not in p for p in pairs)


def test_noise_ceiling_is_one_for_a_noiseless_panel_and_low_for_pure_noise():
    rng = np.random.default_rng(0)
    n_genes = 200
    keep = np.ones(n_genes, dtype=bool)
    # noiseless: every cell of a niche carries that niche's exact profile,
    # so any split-half reproduces the same shift -> reliability 1
    rate_a = rng.uniform(1e-3, 1e-2, n_genes)
    rate_b = rate_a * np.exp(rng.normal(0, 0.5, n_genes))
    clean = sp.csr_matrix(np.vstack([np.tile(rate_a, (400, 1)),
                                     np.tile(rate_b, (400, 1))]))
    rows_a, rows_b = np.arange(400), 400 + np.arange(400)
    assert noise_ceiling(clean, rows_a, rows_b, keep,
                         np.random.default_rng(0)) > 0.99
    # pure noise: the two niches share one profile, so the observed shift
    # is sampling noise and the halves are uncorrelated -> ceiling ~ 0
    noisy = sp.csr_matrix(rng.lognormal(-6, 1.0, (800, n_genes)))
    assert noise_ceiling(noisy, rows_a, rows_b, keep,
                         np.random.default_rng(0)) < 0.2


def test_tumour_band_labels_are_ordered_bands_and_isolated_cells_drop_out():
    # a 1-D transect: tumour cells (type 0) on the left, stroma (type 1)
    # on the right, so the smoothed tumour fraction falls monotonically
    n = 600
    positions = np.column_stack([np.arange(n, dtype=float),
                                 np.zeros(n)])
    t = np.where(np.arange(n) < n // 2, 0, 1).astype(np.int64)
    degrees = np.ones(n, dtype=np.int64)
    degrees[:5] = 0                                   # planted isolated cells
    data = SimpleNamespace(
        type_names=["Tumor Cells", "Stroma"], t=t, positions=positions,
        graph=SimpleNamespace(n_cells=n, degrees=degrees))
    labels = tumour_band_labels(data, k=50)
    assert (labels[:5] == -1).all()
    band = labels[degrees > 0]
    # monotone along the transect and spanning core -> deep stroma
    assert (np.diff(band) <= 0).all()
    assert band.max() == len(TUMOUR_BANDS)
    assert band.min() == 0


def test_tier_summary_reports_the_bars_the_package_is_judged_on():
    def panel(cf, prog, leak, full, phi, ceiling, slope=1.0, trusted=True):
        return {"counterfactual": {"r2": cf, "slope": slope},
                "counterfactual_phi_fixed": {"r2": phi, "slope": slope},
                "program_only": {"r2": prog}, "leak_only": {"r2": leak},
                "program_phi_fixed": {"r2": phi}, "full": {"r2": full},
                "noise_ceiling": ceiling, "trusted": trusted}

    tier = [panel(0.2, 0.1, 0.1, 0.3, 0.1, 0.8),
            panel(0.05, 0.1, 0.02, 0.05, 0.01, 0.2, trusted=False)]
    s = tier_summary(tier)
    assert s["n_panels"] == 2 and s["n_trusted"] == 1
    # only the first panel beats both single channels
    assert s["full_beats_both"] == 1
    assert abs(s["counterfactual"] - 0.125) < 1e-12
    assert abs(s["selection_share"] - (0.175 - 0.125)) < 1e-12
    assert abs(s["interventionable_share"] - (0.055 / 0.125)) < 1e-12
    assert abs(s["noise_ceiling"] - 0.5) < 1e-12
    assert abs(s["counterfactual_of_ceiling"] - 0.25) < 1e-12


def _two_seed_tile(n_neighbours_a: int, n_neighbours_b: int):
    """Seeds 0 and 1 (type 0), each ringed by type-1 neighbours only; the
    two rings differ in size, so composition is identical and count is not."""
    ring_a = 2 + np.arange(n_neighbours_a)
    ring_b = 2 + n_neighbours_a + np.arange(n_neighbours_b)
    n = 2 + n_neighbours_a + n_neighbours_b
    ei = np.concatenate([np.zeros(n_neighbours_a, int),
                         np.ones(n_neighbours_b, int)])
    ej = np.concatenate([ring_a, ring_b])
    t = np.ones(n, dtype=np.int64)
    t[:2] = 0
    graph = build_graph(ei, ej, np.ones(len(ei)), np.full(len(ei), 10.0), n,
                        type_index=t, n_types=2)
    return graph, tile_batch(graph, np.array([0, 1])), t


def test_phi_fixed_channel_removes_the_image_and_keeps_composition():
    torch.manual_seed(0)
    genes, phi_dim = 25, 4
    graph, batch, t = _two_seed_tile(2, 5)
    model = DisCell(genes, 2, phi_dim, median_counts=50.0, d_z=3, d_w=2,
                    hidden=16, gat_dim=6, heads=2)
    rng = np.random.default_rng(1)
    tile = dict(
        nodes=batch.nodes,
        x=torch.tensor(rng.poisson(2.0, (len(batch.nodes), genes)),
                       dtype=torch.int16),
        t=torch.tensor(t[batch.nodes]),
        phi=torch.randn(len(batch.nodes), phi_dim),
        isolated=torch.tensor(graph.isolated[batch.nodes]),
        gat_src=torch.tensor(batch.gat_src), gat_dst=torch.tensor(batch.gat_dst),
        leak_src=torch.tensor(batch.leak_src), leak_dst=torch.tensor(batch.leak_dst),
        leak_beta=torch.tensor(batch.leak_beta, dtype=torch.float32),
        n_seeds=batch.n_seeds, n_context=batch.n_context)
    trainer = SimpleNamespace(model=model.eval(),
                              config=SimpleNamespace(kappa=0.1),
                              train_batches=[tile], val_batches=[],
                              _forward_kwargs=Trainer._forward_kwargs)

    # one group per seed, so a "group mean" is that seed's own value
    group = np.full(graph.n_cells, -1, dtype=np.int64)
    group[0], group[1] = 0, 1
    plain = collect_channels(trainer, group, 2)
    assert set(plain) == {"prior_w", "c", "rho", "rho_bar", "n"}
    assert plain["rho"].shape == plain["rho_bar"].shape == (2, genes)
    assert list(plain["n"]) == [1.0, 1.0]
    # real Phi differs between the two seeds, so does the prior mean
    assert not np.allclose(plain["prior_w"][0], plain["prior_w"][1])

    phi_by_type = rng.normal(size=(2, phi_dim)).astype(np.float32)
    fixed = collect_channels(trainer, group, 2, phi_by_type)
    assert set(fixed) == {"prior_w", "c", "n"}
    # same type, same composition (all type 1), Phi at the type mean: the
    # softmax GAT is count-blind, so the two prior means coincide exactly
    assert np.allclose(fixed["prior_w"][0], fixed["prior_w"][1], atol=1e-6)
    assert not np.allclose(fixed["prior_w"], plain["prior_w"])
    # the substituted Phi is the seeds' own type row, checked by hand
    kwargs = Trainer._forward_kwargs(tile)
    kwargs["phi"] = torch.as_tensor(phi_by_type)[kwargs["t"]]
    with torch.no_grad():
        by_hand = model(**kwargs, kappa=0.1, sample=False).prior_mean_w[:2]
    assert np.allclose(fixed["prior_w"], by_hand.numpy(), atol=1e-6)


def test_group_means_average_the_cells_of_a_group():
    """Two seeds in ONE group: the accumulator must return their mean."""
    torch.manual_seed(0)
    genes, phi_dim = 25, 4
    graph, batch, t = _two_seed_tile(2, 5)
    model = DisCell(genes, 2, phi_dim, median_counts=50.0, d_z=3, d_w=2,
                    hidden=16, gat_dim=6, heads=2)
    rng = np.random.default_rng(1)
    tile = dict(
        nodes=batch.nodes,
        x=torch.tensor(rng.poisson(2.0, (len(batch.nodes), genes)),
                       dtype=torch.int16),
        t=torch.tensor(t[batch.nodes]),
        phi=torch.randn(len(batch.nodes), phi_dim),
        isolated=torch.tensor(graph.isolated[batch.nodes]),
        gat_src=torch.tensor(batch.gat_src), gat_dst=torch.tensor(batch.gat_dst),
        leak_src=torch.tensor(batch.leak_src), leak_dst=torch.tensor(batch.leak_dst),
        leak_beta=torch.tensor(batch.leak_beta, dtype=torch.float32),
        n_seeds=batch.n_seeds, n_context=batch.n_context)
    trainer = SimpleNamespace(model=model.eval(),
                              config=SimpleNamespace(kappa=0.1),
                              train_batches=[tile], val_batches=[],
                              _forward_kwargs=Trainer._forward_kwargs)
    per_cell = collect_channels(trainer,
                                np.where(np.arange(graph.n_cells) < 2,
                                         np.arange(graph.n_cells), -1), 2)
    pooled = collect_channels(trainer,
                              np.where(np.arange(graph.n_cells) < 2, 0, -1), 1)
    assert pooled["n"][0] == 2.0
    for key in ("prior_w", "c", "rho", "rho_bar"):
        assert np.allclose(pooled[key][0], per_cell[key].mean(axis=0),
                           atol=1e-6)


# -- the distribution-level read (devlog 2026-09-21) -------------------------


def _simplex(rng, n, g, logits):
    """*n* draws around *logits* on the g-simplex, as probability vectors."""
    z = logits[None, :] + rng.normal(0, 0.25, (n, g))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_exact_transport_closes_the_gap_and_lands_on_the_floor():
    """A world where the transport is exact: the transported cells are
    fresh draws from the target's own law, so MMD^2 must fall to the
    sampling floor and gap closed must be ~1."""
    rng = np.random.default_rng(0)
    g = 30
    target_logits = rng.normal(0, 1.0, g)
    source_logits = target_logits + rng.normal(0, 1.5, g)   # a real gap
    p_target = _simplex(rng, 600, g, target_logits)
    p_trans = _simplex(rng, 600, g, target_logits)          # exact
    p_unt = _simplex(rng, 600, g, source_logits)
    s = distribution_scores(p_trans, p_unt, p_target, p_unt,
                            np.random.default_rng(1), n_boot=100)
    assert s["mmd2"]["transported"] < s["mmd2"]["untransported"]
    assert s["improves"] and s["ci_excludes_zero"]
    assert s["gap_closed"] > 0.9
    # "transported ~ floor": both are sampling noise on the same law
    assert abs(s["mmd2"]["transported"]) < 5 * abs(s["mmd2"]["floor"]) + 1e-6
    # the raw niche difference is the thing transport had to remove
    assert s["mmd2"]["observed_source"] > s["mmd2"]["transported"]


def test_a_constant_shift_predictor_fails_the_type_mean_bar():
    """Bar (3): a model that moves every cell to the same point reproduces
    the target MEAN and nothing else, so it must not beat the degenerate
    type-mean reference -- otherwise the read is just the mean shift."""
    rng = np.random.default_rng(0)
    g = 30
    target_logits = rng.normal(0, 1.0, g)
    p_target = _simplex(rng, 600, g, target_logits)
    p_unt = _simplex(rng, 600, g, target_logits + rng.normal(0, 1.5, g))
    # collapsed cloud: every transported cell at the target's mean
    p_trans = np.tile(p_target.mean(axis=0), (600, 1))
    s = distribution_scores(p_trans, p_unt, p_target, p_unt,
                            np.random.default_rng(1), n_boot=100)
    # it does help against no transport at all -- and still fails bar (3)
    assert s["improves"]
    assert not s["beats_type_mean"]
    assert s["gap_closed"] < 1.0


def test_pooling_tightens_the_ci_when_z_is_context_free():
    """A synthetic z with no context dependence: pooling every other niche
    into the source gives more cells on the binding side, so the paired
    bootstrap CI of transported - untransported must be tighter."""
    rng = np.random.default_rng(0)
    g = 30
    target_logits = rng.normal(0, 1.0, g)
    p_target = _simplex(rng, 1200, g, target_logits)        # the big side
    # one source niche: few cells, so the source is what binds the size match
    small = 150
    pw_trans = _simplex(rng, small, g, target_logits)
    pw_unt = _simplex(rng, small, g, target_logits + rng.normal(0, 1.2, g))
    # pooled: four niches' worth of the SAME context-free law
    pooled = 4 * small
    lo_trans = _simplex(rng, pooled, g, target_logits)
    lo_unt = _simplex(rng, pooled, g, target_logits + rng.normal(0, 1.2, g))
    pw = distribution_scores(pw_trans, pw_unt, p_target, pw_unt,
                             np.random.default_rng(2), n_boot=200)
    lo = distribution_scores(lo_trans, lo_unt, p_target, lo_unt,
                             np.random.default_rng(2), n_boot=200)
    assert pw["n"] == small and lo["n"] == pooled
    assert lo["ci_width"] < pw["ci_width"]


def test_size_matching_and_the_too_small_guard():
    rng = np.random.default_rng(0)
    g = 12
    p_target = _simplex(rng, 40, g, np.zeros(g))
    p_src = _simplex(rng, 300, g, np.ones(g))
    s = distribution_scores(p_src, p_src, p_target, p_src,
                            np.random.default_rng(0), n_boot=20)
    assert s["n"] == 40                       # matched to the smaller side
    tiny = distribution_scores(p_src[:5], p_src[:5], p_target[:5], p_src[:5],
                               np.random.default_rng(0), n_boot=20)
    assert tiny["insufficient"]


def test_distribution_summary_reports_the_bars():
    def panel(gap, improves, ci_zero, beats, width):
        return {"scores": {"insufficient": False, "gap_closed": gap,
                           "gap_closed_type_mean": 0.3,
                           "improves": improves,
                           "ci_excludes_zero": ci_zero,
                           "beats_type_mean": beats, "ci_width": width,
                           "mmd2": {"transported": 0.1, "untransported": 0.2,
                                    "floor": 0.01, "observed_source": 0.3},
                           "energy": {"transported": 0.1,
                                      "untransported": 0.2}}}
    tier = [panel(0.8, True, True, True, 0.02),
            panel(0.2, True, False, False, 0.04),
            {"scores": {"insufficient": True}}]
    s = distribution_summary(tier)
    assert s["n_panels"] == 2                 # the insufficient panel drops
    assert s["n_improved"] == 2 and s["n_ci_excludes_zero"] == 1
    assert s["n_beats_type_mean"] == 1
    assert abs(s["median_gap_closed"] - 0.5) < 1e-12
    assert abs(s["median_ci_width"] - 0.03) < 1e-12


def test_top_gene_overlap_is_scored_against_its_own_chance_level():
    rng = np.random.default_rng(0)
    observed = rng.normal(size=500)
    perfect = top_gene_overlap(observed.copy(), observed, k=50)
    assert perfect["overlap"] == 50 and perfect["fraction"] == 1.0
    assert abs(perfect["chance"] - 50 / 500) < 1e-12
    # a per-panel constant must not change the ranking
    assert top_gene_overlap(observed + 4.0, observed, k=50)["overlap"] == 50
    # an unrelated prediction sits at chance, within sampling slack
    noise = top_gene_overlap(rng.normal(size=500), observed, k=50)
    assert noise["overlap"] <= 15
    # fewer genes than k: k shrinks and chance becomes 1
    short = top_gene_overlap(observed[:20], observed[:20], k=50)
    assert short["k"] == 20 and short["chance"] == 1.0


def test_shot_noise_compresses_the_rate_variant_and_count_matching_fixes_it():
    """Why the count-matched companion exists.

    Target cells are multinomial draws at a realistic Xenium depth; the
    model predicts smooth rates. Even an EXACT prediction then sits far
    from the target cloud, because the target's own shot noise dominates
    the Hellinger geometry -- so gap closed is compressed towards zero for
    a reason that has nothing to do with transport. Drawing counts from
    the predictions at matched depths restores it."""
    rng = np.random.default_rng(0)
    g, n, depth = 200, 400, 300
    target_rates = rng.dirichlet(np.full(g, 0.3))
    # a realistic niche difference: a modest perturbation, not another slide
    source_rates = target_rates * np.exp(rng.normal(0, 0.3, g))
    source_rates = source_rates / source_rates.sum()
    counts = rng.multinomial(depth, target_rates, size=n).astype(float)
    p_target = counts / counts.sum(axis=1, keepdims=True)
    p_trans = np.tile(target_rates, (n, 1))       # an exact prediction
    p_unt = np.tile(source_rates, (n, 1))
    depths = np.full(n, depth)

    rate = distribution_scores(p_trans, p_unt, p_target, p_unt,
                               np.random.default_rng(1), n_boot=50)
    matched = distribution_scores(p_trans, p_unt, p_target, p_unt,
                                  np.random.default_rng(1), n_boot=50,
                                  count_depths=depths)
    assert not rate["count_matched"] and matched["count_matched"]
    # the exact prediction is far above the floor in the rate variant
    assert rate["mmd2"]["transported"] > 20 * abs(rate["mmd2"]["floor"])
    # and the companion recovers the transport the rate variant hides
    assert matched["gap_closed"] > rate["gap_closed"] + 0.2
    assert matched["gap_closed"] > 0.8


# -- second round: model-vs-model and matched twins (devlog 2026-09-21) ------


def test_model_vs_model_target_gives_a_closed_gap_and_a_beatable_type_mean():
    """Read A on a planted panel: the target cloud is a cloud of SMOOTH
    decoded vectors with real spread. A transport that reproduces that law
    must close ~1 of the gap, and the type-mean predictor -- one point
    replicated -- must NOT, because the spread it throws away is no longer
    shot noise."""
    rng = np.random.default_rng(0)
    g = 40
    target_logits = rng.normal(0, 1.0, g)
    p_target_model = _simplex(rng, 600, g, target_logits)   # decoded target
    p_trans = _simplex(rng, 600, g, target_logits)          # same law
    p_unt = _simplex(rng, 600, g, target_logits + rng.normal(0, 1.5, g))
    s = distribution_scores(p_trans, p_unt, p_target_model, p_unt,
                            np.random.default_rng(1), n_boot=100)
    assert not s["count_matched"]
    assert s["gap_closed"] > 0.9
    assert s["gap_closed_type_mean"] < 1.0
    assert s["beats_type_mean"]


def test_twin_read_matched_beats_random_when_z_carries_the_cell():
    """Read B: source and target cells come in exact z-pairs, and the
    decoded vector is a function of z. The matched twin must then land far
    closer than a random same-type source cell, and closer than the same
    twin left untransported."""
    rng = np.random.default_rng(0)
    g, n, d = 30, 120, 4
    z = rng.normal(size=(n, d))
    basis = rng.normal(size=(d, g))
    shift = rng.normal(0, 1.2, g)          # the niche difference

    def decode(zz, offset):
        lo = zz @ basis + offset[None, :]
        e = np.exp(lo - lo.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    perm = rng.permutation(n)              # twins are not the identity index
    z_src = z[perm]
    p_trans = decode(z_src, np.zeros(g))   # source cells in the target niche
    p_unt = decode(z_src, shift)           # ... left in their own niche
    p_target_model = decode(z, np.zeros(g))
    s = twin_scores(p_trans, p_unt, p_target_model, z_src, z,
                    np.random.default_rng(1), n_boot=100)
    assert not s["insufficient"] and s["n"] == n
    # the twin is the exact same cell, so the matched distance is ~0
    assert s["median"]["transported"] < 1e-6
    assert s["median"]["transported"] < 0.1 * s["median"]["random"]
    assert s["beats_random"] and s["beats_untransported"]
    assert s["twin_margin"] > 0.9
    assert s["gap_closed"] > 0.9
    assert s["ci_vs_random"][1] < 0.0 and s["ci_vs_untransported"][1] < 0.0


def test_twin_margin_vanishes_when_z_carries_nothing():
    """The failure mode the margin exists to detect: the decoded vector
    does not depend on z at all, so the matched twin is no better than a
    random source cell."""
    rng = np.random.default_rng(0)
    g, n, d = 30, 120, 4
    z = rng.normal(size=(n, d))
    z_src = rng.normal(size=(n, d))
    base = _simplex(rng, 1, g, rng.normal(0, 1.0, g))[0]
    p_trans = np.tile(base, (n, 1))
    p_unt = np.tile(_simplex(rng, 1, g, rng.normal(0, 1.5, g))[0], (n, 1))
    p_target_model = _simplex(rng, n, g, np.log(base + 1e-12))
    s = twin_scores(p_trans, p_unt, p_target_model, z_src, z,
                    np.random.default_rng(1), n_boot=100)
    assert abs(s["twin_margin"]) < 0.05


def test_twin_summary_and_hvg_restriction():
    panels = [{"twins": {"insufficient": False, "gap_closed": 0.8,
                         "twin_margin": 0.3, "median": {
                             "transported": 0.1, "untransported": 0.2,
                             "random": 0.15, "floor": 0.05},
                         "ci_untransported_excludes_zero": True,
                         "ci_random_excludes_zero": False}},
              {"twins": {"insufficient": False, "gap_closed": -0.2,
                         "twin_margin": -0.1, "median": {
                             "transported": 0.3, "untransported": 0.2,
                             "random": 0.25, "floor": 0.05},
                         "ci_untransported_excludes_zero": False,
                         "ci_random_excludes_zero": False}},
              {"twins": {"insufficient": True}}]
    sm = twin_summary(panels)
    assert sm["n_panels"] == 2
    assert sm["n_gap_closed_positive"] == 1
    assert abs(sm["median_gap_closed"] - 0.3) < 1e-9
    assert sm["n_twin_margin_positive"] == 1
    assert sm["n_ci_vs_untransported_excludes_zero"] == 1

    p = np.array([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]])
    q = restrict_renormalise(p, np.array([True, True, False]))
    assert np.allclose(q.sum(axis=1), 1.0)
    assert np.allclose(q[0], [1 / 3, 2 / 3])


def test_collect_own_p_returns_the_cells_own_decode_in_the_wanted_order():
    """The honest target of read 6a.6: ``p`` for chosen cells at their OWN
    posterior w, context and influx -- i.e. exactly ``Forward.log_p`` under
    ``sample=False``, gathered for a subset and in that subset's order."""
    torch.manual_seed(0)
    genes, phi_dim = 25, 4
    graph, batch, t = _two_seed_tile(2, 5)
    model = DisCell(genes, 2, phi_dim, median_counts=50.0, d_z=3, d_w=2,
                    hidden=16, gat_dim=6, heads=2)
    rng = np.random.default_rng(1)
    tile = dict(
        nodes=batch.nodes,
        x=torch.tensor(rng.poisson(2.0, (len(batch.nodes), genes)),
                       dtype=torch.int16),
        t=torch.tensor(t[batch.nodes]),
        phi=torch.randn(len(batch.nodes), phi_dim),
        isolated=torch.tensor(graph.isolated[batch.nodes]),
        gat_src=torch.tensor(batch.gat_src), gat_dst=torch.tensor(batch.gat_dst),
        leak_src=torch.tensor(batch.leak_src), leak_dst=torch.tensor(batch.leak_dst),
        leak_beta=torch.tensor(batch.leak_beta, dtype=torch.float32),
        n_seeds=batch.n_seeds, n_context=batch.n_context)
    trainer = SimpleNamespace(model=model.eval(),
                              config=SimpleNamespace(kappa=0.1),
                              train_batches=[tile], val_batches=[],
                              _forward_kwargs=Trainer._forward_kwargs)
    with torch.no_grad():
        by_hand = model(**Trainer._forward_kwargs(tile), kappa=0.1,
                        sample=False).log_p.exp().numpy()

    both = collect_own_p(trainer, np.array([0, 1]), graph.n_cells)
    assert both.shape == (2, genes)
    assert np.allclose(both, by_hand[:2], atol=1e-6)
    # rows follow *wanted*, not the cell index, and a subset is honoured
    assert np.allclose(collect_own_p(trainer, np.array([1, 0]),
                                     graph.n_cells), by_hand[[1, 0]],
                       atol=1e-6)
    assert np.allclose(collect_own_p(trainer, np.array([1]), graph.n_cells)[0],
                       by_hand[1], atol=1e-6)
    # it is a probability vector, and it is NOT the group-mean-w decode that
    # the first round used (the two seeds see different neighbourhoods)
    assert np.allclose(both.sum(axis=1), 1.0, atol=1e-5)
    assert not np.allclose(both[0], both[1], atol=1e-6)
