"""The double-positive instrument (6b.2) on a planted world with a known leak,
and the model-side collector against the forward pass it reads."""

from __future__ import annotations

import numpy as np
import pytest

from discell.experiments import marker_pairs as MP
from discell.model.synthetic import simulate

from tests.test_model_train import small_fit  # noqa: F401  (fixture)

KAPPA = 0.2
DEPTH = 150.0
#: gene layout: 0..3 marker of type 0..3 (strictly exclusive), 4/5 a control
#: pair co-expressed in type 0, 6/7 in type 1, 8.. background
N_GENES = 30
EXCL = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (0, 3)]
CTRL = [(4, 5), (6, 7)]


def planted_world(seed: int = 0, kappa: float = KAPPA):
    """Clustered types on the real graph builder; rho with strictly exclusive
    markers (share 0.1 in their own type, 0 elsewhere) and two control pairs
    (share 0.1 each in type 0 / type 1); the true leak p = (1-k) rho + k rho_bar."""
    sim = simulate(n_cells=4000, n_types=4, kappa=kappa, seed=seed)
    t = sim.t
    rho = np.full((len(t), N_GENES), 0.0)
    rho[:, 8:] = 1.0
    for g in range(4):
        rho[t == g, g] = 0.1 * (N_GENES - 8) / 0.5    # rescaled below
    rho[t == 0, 4] = rho[t == 0, 5] = rho[t == 0, 0]
    rho[t == 1, 6] = rho[t == 1, 7] = rho[t == 1, 1]
    rho /= rho.sum(axis=1, keepdims=True)
    rho_bar = np.asarray(sim.graph.in_edges @ rho)
    connected = rho_bar.sum(axis=1) > 0
    p = np.where(connected[:, None], (1 - kappa) * rho + kappa * rho_bar, rho)
    depth = np.full(len(t), DEPTH)
    rng = np.random.default_rng(seed)
    x = np.stack([rng.multinomial(int(DEPTH), row) for row in p]).astype(float)
    x_clean = np.stack([rng.multinomial(int(DEPTH), row) for row in rho]
                       ).astype(float)
    return dict(sim=sim, rho=rho, rho_bar=rho_bar, p=p, depth=depth, x=x,
                x_clean=x_clean)


def dp(on: np.ndarray, pairs) -> float:
    return float(np.mean([(on[:, a] & on[:, b]).mean() for a, b in pairs]))


def test_marker_shares_are_as_planted():
    w = planted_world()
    t = w["sim"].t
    assert np.allclose(w["rho"][t == 0, 0], w["rho"][t == 0, 4])
    assert (w["rho"][t != 1, 1] == 0).all()             # strictly exclusive
    assert w["rho"][t == 0, 0].mean() > 0.05


def test_true_kappa_puts_exclusive_dp_at_the_floor_control_unchanged():
    """The planted claim: under the true kappa and rho_bar the corrected
    decode removes every leak-induced double positive (floor = the leak-free
    world's, 0 here), while the control pairs keep their DP exactly."""
    w = planted_world()
    on = MP.on_calls(w["x"], w["depth"], w["rho_bar"], w["p"], KAPPA)
    clean = MP.on_calls(w["x_clean"], w["depth"], np.zeros_like(w["rho"]),
                        w["rho"], 0.0)
    floor_decode = dp(np.rint(w["depth"][:, None] * (1 - KAPPA) * w["rho"]) >= 1,
                      EXCL)
    assert floor_decode == 0.0 and dp(clean["raw"], EXCL) == 0.0
    # the leak is there to remove: raw and uncorrected decode carry it
    assert dp(on["raw"], EXCL) > 0.01
    assert dp(on["decode"], EXCL) > 0.01
    # ... and the corrected decode is at the floor
    assert dp(on["decode_corrected"], EXCL) == floor_decode
    # control pairs: every true co-expression is kept -- the corrected DP is
    # the leak-free world's, and inside the pairs' own types it is unchanged.
    # (Across all cells the uncorrected decode's control DP is higher: both
    # genes of a control pair leak together into other-type neighbours.)
    assert dp(on["decode_corrected"], CTRL) == pytest.approx(
        dp(clean["decode"], CTRL), abs=1e-12)
    t = w["sim"].t
    for (a, b), own in zip(CTRL, (0, 1)):
        keep = t == own
        assert dp(on["decode_corrected"][keep], [(a, b)]) == \
            dp(on["decode"][keep], [(a, b)])
    assert dp(on["decode"], CTRL) > dp(on["decode_corrected"], CTRL)
    # x~ lowers exclusive DP but leaves a Poisson ceiling above the floor:
    # it removes the mean leak, not the draw -- control inside its own type
    # within 2 % (relative)
    assert dp(on["xtilde"], EXCL) < 0.8 * dp(on["raw"], EXCL)
    assert dp(on["xtilde"], EXCL) > floor_decode
    for (a, b), own in zip(CTRL, (0, 1)):
        keep = t == own
        assert dp(on["xtilde"][keep], [(a, b)]) == pytest.approx(
            dp(on["raw"][keep], [(a, b)]), rel=0.02)
    # the headline ratio falls under both corrections
    for arm, ref in (("xtilde", "raw"), ("decode_corrected", "decode")):
        assert dp(on[arm], EXCL) / dp(on[arm], CTRL) < \
            dp(on[ref], EXCL) / dp(on[ref], CTRL)


def test_noiseless_world_xtilde_is_exact():
    """With expected counts (x = l p) x~ = l (1-k) rho exactly, so x~ also
    reaches the floor -- the gap in the sampled world is Poisson, not algebra."""
    w = planted_world()
    x = w["depth"][:, None] * w["p"]
    on = MP.on_calls(x, w["depth"], w["rho_bar"], w["p"], KAPPA)
    assert dp(on["raw"], EXCL) > 0.01
    assert dp(on["xtilde"], EXCL) == 0.0
    assert np.array_equal(on["xtilde"], on["decode_corrected"])


def test_kappa_zero_changes_nothing():
    w = planted_world()
    on = MP.on_calls(w["x"], w["depth"], w["rho_bar"], w["p"], 0.0)
    assert np.array_equal(on["xtilde"], on["raw"])
    assert np.array_equal(on["xtilde_ge1"], on["raw"])
    assert np.array_equal(on["decode_corrected"], on["decode"])


def test_unrounded_xtilde_drops_single_counts_whatever_the_leak():
    x = np.array([[1.0, 2.0, 1.0]])
    rho_bar = np.array([[1e-6, 1e-6, 0.0]])
    on = MP.on_calls(x, np.array([100.0]), rho_bar, np.full((1, 3), 0.1), 0.1)
    assert on["raw"].all() and on["xtilde"].all()
    assert on["xtilde_ge1"].tolist() == [[False, True, True]]


def test_score_reads_panels_ratio_and_paired_grading():
    w = planted_world()
    pairs = ([{"gene_a": f"g{a}", "gene_b": f"g{b}", "kind": "exclusive"}
              for a, b in EXCL]
             + [{"gene_a": f"g{a}", "gene_b": f"g{b}", "kind": "control"}
                for a, b in CTRL])
    genes = [f"g{i}" for i in range(N_GENES)]
    cells = {"x": w["x"], "depth": w["depth"], "rho_bar": w["rho_bar"],
             "p": w["p"]}
    out = MP.score(cells, KAPPA, pairs, genes, w["sim"].positions, n_boot=200)
    on = MP.on_calls(w["x"], w["depth"], w["rho_bar"], w["p"], KAPPA)
    for arm in MP.ARMS:
        e = out["arms"][arm]["exclusive"]["estimate"]
        c = out["arms"][arm]["control"]["estimate"]
        assert e == pytest.approx(dp(on[arm], EXCL))
        assert c == pytest.approx(dp(on[arm], CTRL))
        assert out["arms"][arm]["ratio"]["estimate"] == pytest.approx(e / c)
    # a perfect correction (true kappa, true rho_bar) removes the leak-made
    # control double positives too, so pooled control DP falls: the grading
    # calls it "specific, control also falls", never "as wished"
    g = out["grading"]["decode_corrected-decode"]
    assert g["exclusive_falls"] and g["specific"] and not g["control_holds"]
    assert MP.verdict([g, g, g]) == "specific, control also falls"
    assert out["grading"]["xtilde-raw"]["exclusive_falls"]
    held = dict(g, control_holds=True)
    assert MP.verdict([held] * 3) == "as wished"
    assert MP.verdict([held, held, dict(g, exclusive_falls=False)]) == \
        "no specific removal"
    assert len(out["pairs"]) == len(pairs)


def test_collector_matches_the_forward_pass(small_fit):  # noqa: F811
    """x, depth, rho_bar and p are the forward pass's own; the corrected
    decode is l (1 - kappa) rho on connected cells, l rho on isolated ones."""
    import torch

    trainer, _ = small_fit
    genes = np.array([0, 3, 7])
    cells = MP.collect_cells(trainer, trainer.val_batches, genes)
    n_val = sum(b["n_seeds"] for b in trainer.val_batches)
    assert len(cells["nodes"]) == n_val
    x_full = trainer.data.x[cells["nodes"]].toarray()
    assert np.array_equal(cells["x"], x_full[:, genes])
    assert np.allclose(cells["depth"], x_full.sum(axis=1))
    kappa = trainer.config.kappa
    rhos = []
    with torch.no_grad():
        for batch in trainer.val_batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch), kappa=kappa,
                                sample=False)
            rhos.append(fwd.log_rho[:batch["n_seeds"]].exp()[:, genes].numpy())
    rho = np.concatenate(rhos)
    corrected = cells["depth"][:, None] * np.clip(
        cells["p"] - kappa * cells["rho_bar"], 0, None)
    scale = np.where(cells["isolated"], 1.0, 1 - kappa)[:, None]
    assert np.allclose(corrected, cells["depth"][:, None] * scale * rho,
                       rtol=1e-4, atol=1e-6)
