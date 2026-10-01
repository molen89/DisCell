"""Follow-ups to the planted spill-over control (devlog 2026-10-01,
'RECOMB additions: results and follow-ups')."""

import numpy as np
import pytest

from discell.experiments import planted_followups as F
from discell.experiments import planted_spillover as PS


def _vectors(seed=0, g=20):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=g) for _ in range(6)]


def test_decompose_sums_to_the_original_contrast():
    obs, clean, true, prog, leak, pop = _vectors()
    terms = F.decompose(obs, clean, true, prog, leak, pop)
    original = (PS.r2_rows(obs[None], prog + leak)[0]
                - PS.r2_rows(obs[None], leak)[0])
    assert set(terms) == set(F.TERMS)
    assert sum(terms.values()) == pytest.approx(original)


def test_decompose_without_population_has_no_sampling_term():
    obs, clean, true, prog, leak, _ = _vectors(1)
    terms = F.decompose(obs, clean, true, prog, leak)
    assert terms["sampling"] == 0.0
    assert terms["penalty"] <= 0.0


def test_alignment_is_the_spill_cross_term_and_shift_invariant():
    obs, clean, true, prog, _, _ = _vectors(2)
    got = F.alignment_rows(obs[None] + 3.0, clean[None] - 1.0,
                           true[None] + 2.0, prog + 5.0)[0]
    o, c, t, p = (v - v.mean() for v in (obs, clean, true, prog))
    assert got == pytest.approx(2 * p @ (o - c) / (t @ t))
    assert F.alignment_rows(clean[None], clean[None], true[None], prog)[0] == 0.0


def test_alignment_is_zero_in_expectation_when_obs_is_clean_plus_noise():
    """The null the corrected contrast is built for: no spill, obs = clean
    + noise independent of the programme. The original is biased there."""
    rng = np.random.default_rng(3)
    clean = rng.normal(size=20)
    prog = 0.5 * clean + rng.normal(size=20)
    obs = clean[None] + rng.normal(size=(20000, 20))
    rep = np.repeat(clean[None], len(obs), 0)
    align = F.alignment_rows(obs, rep, rep, prog)
    assert abs(align.mean()) < 4 * align.std() / np.sqrt(len(align))
    original = PS.r2_rows(obs, prog)          # leak = 0
    assert original.mean() < -0.05


def test_overlap_guard_matches_transport_rule():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(200, 3))
    assert not F._overlap(a, a + 0.1)
    assert F._overlap(a, a + 10.0)


def test_summarise_ratios():
    panel = {"ceiling": {"observed": 0.5, "clean": 1.0}, "r2": {}}
    for gset in ("all", "spill", "response"):
        panel["r2"][gset] = {t: {m: 0.25 for m in F.METHODS}
                             for t in F.TARGETS}
    s = F.summarise([panel, panel])
    assert s["all"]["observed"]["regression_rate"]["of_ceiling"] == 0.5
    assert s["all"]["clean"]["discell_cf"]["of_ceiling"] == 0.25
    assert s["all"]["true_contaminated"]["discell_cf"]["of_ceiling"] is None


def test_noise_by_tiles_sees_a_programme_fitted_to_training_noise():
    """A programme equal to the training-tile cells' count noise correlates
    with e there and not on the held-out tiles."""
    from types import SimpleNamespace

    from discell.model.synthetic import simulate

    sim = simulate(n_cells=3000, n_types=2, kappa=0.0, seed=5)
    n = sim.t.size
    rng = np.random.default_rng(0)
    cells = rng.permutation(n)
    val = cells[: n // 2]
    a, b = cells[: n // 4], cells[n // 4: n // 2]          # val-tile cells
    ta, tb = cells[n // 2: 3 * n // 4], cells[3 * n // 4:]  # training tiles
    rate = sim.x / sim.totals[:, None]
    e_train = (np.log(rate[tb].mean(0) + 1e-8) - np.log(rate[ta].mean(0) + 1e-8)
               - F._mean_log_shift(sim.p_true, ta, tb))
    data = SimpleNamespace(graph=SimpleNamespace(n_cells=n),
                           val_tiles=[val])
    panel = {"rows_a": np.concatenate([a, ta]), "rows_b": np.concatenate([b, tb]),
             "program": e_train}
    out = F.noise_by_tiles(sim, data, [panel])
    assert out["train_tiles"][0] == pytest.approx(1.0)
    assert out["val_tiles"][0] < 0.9          # a different noise draw
