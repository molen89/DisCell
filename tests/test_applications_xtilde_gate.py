"""The x~ gate's read-out, against planted answers.

Every number here is known before the call: the thresholds are quantiles of
constructed control distributions, the FPRs are counted by hand, and the
V9 defect (one global threshold across types with different offsets) is
reproduced as a regression so it cannot come back.
"""

import numpy as np
import pytest

from discell.applications.xtilde_gate import (
    RECALL_MARGIN,
    TRUE_VICTIM_TYPES,
    bootstrap_excess,
    build_world,
    follow_up_verdict,
    recall_at_threshold,
    search_share,
    excess_fpr,
    gene_band,
    leak_subtracted_counts,
    power_gate,
    programme_score,
    read_out,
    verdict,
    within_type_threshold,
    z_probe,
)


def _two_type_world(victim_shift=(0.0, 0.0), offsets=(0.0, 100.0),
                    n_controls=200, n_victims=100):
    """Two types, control scores uniform on [offset, offset + 1).

    Victims are the same distribution shifted by *victim_shift*: with a shift
    of 0 the excess FPR is exactly 0 in expectation, whatever the offsets.
    """
    t, score, victims, controls = [], [], [], []
    for g, (off, shift) in enumerate(zip(offsets, victim_shift)):
        grid = (np.arange(n_controls) + 0.5) / n_controls
        t += [g] * n_controls
        score += list(off + grid)
        victims += [False] * n_controls
        controls += [True] * n_controls
        vgrid = (np.arange(n_victims) + 0.5) / n_victims
        t += [g] * n_victims
        score += list(off + vgrid + shift)
        victims += [True] * n_victims
        controls += [False] * n_victims
    return (np.array(t), np.array(score), np.array(victims), np.array(controls))


# -- within_type_threshold -------------------------------------------------

def test_threshold_is_the_control_quantile_of_each_type_separately():
    t, score, _, controls = _two_type_world()
    thr = within_type_threshold(score, t, controls, rate=0.3)
    # controls of type g are off + (k + 0.5)/200: the 0.7 quantile is off + 0.7
    assert set(thr) == {0, 1}
    assert thr[0] == pytest.approx(0.7, abs=2e-3)
    assert thr[1] == pytest.approx(100.7, abs=2e-3)


def test_threshold_calls_the_nominal_rate_of_controls_in_every_type():
    t, score, _, controls = _two_type_world()
    thr = within_type_threshold(score, t, controls, rate=0.3)
    for g in (0, 1):
        c = controls & (t == g)
        assert (score[c] > thr[g]).mean() == pytest.approx(0.3, abs=0.01)


# -- excess_fpr ------------------------------------------------------------

def test_excess_is_zero_when_victims_and_controls_share_a_distribution():
    t, score, victims, controls = _two_type_world(victim_shift=(0.0, 0.0))
    out = excess_fpr(score, t, victims, controls, rate=0.3)
    assert out["control_fpr"] == pytest.approx(0.3, abs=0.01)
    assert out["excess"] == pytest.approx(0.0, abs=0.02)
    assert out["n_victims"] == 200 and out["n_controls"] == 400


def test_a_known_victim_shift_gives_the_hand_counted_excess():
    # victims of both types shifted by +0.2: control threshold is off + 0.7,
    # so the victims above it are those with vgrid > 0.5 -> FPR 0.5.
    t, score, victims, controls = _two_type_world(victim_shift=(0.2, 0.2))
    out = excess_fpr(score, t, victims, controls, rate=0.3)
    assert out["victim_fpr"] == pytest.approx(0.5, abs=0.01)
    assert out["excess"] == pytest.approx(0.2, abs=0.02)


def test_pooling_is_weighted_by_victims_not_by_type():
    """Type 0 (300 victims, excess 0.7) and type 1 (100 victims, excess 0.0)."""
    grid = (np.arange(100) + 0.5) / 100
    t = np.array([0] * 400 + [1] * 200)
    score = np.concatenate([grid, np.tile(1.0 + grid, 3),       # type 0
                            grid, grid])                        # type 1
    controls = np.r_[[True] * 100, [False] * 300, [True] * 100, [False] * 100]
    victims = ~controls
    out = excess_fpr(score, t, victims, controls, rate=0.3)
    assert out["per_type"][0]["excess"] == pytest.approx(0.7, abs=0.01)
    assert out["per_type"][1]["excess"] == pytest.approx(0.0, abs=0.01)
    assert out["excess"] == pytest.approx((300 * 0.7 + 100 * 0.0) / 400, abs=0.01)


def test_types_below_min_cells_are_dropped_from_the_pool():
    t, score, victims, controls = _two_type_world(victim_shift=(0.2, 0.2),
                                                  n_victims=100)
    victims = victims & ((t == 0) | (np.cumsum(victims & (t == 1)) <= 5))
    out = excess_fpr(score, t, victims, controls, rate=0.3, min_cells=30)
    assert list(out["per_type"]) == [0]
    assert out["n_victims"] == 100


def test_empty_pool_returns_nan_rather_than_dividing_by_zero():
    t, score, _, controls = _two_type_world()
    out = excess_fpr(score, t, np.zeros(len(t), bool), controls)
    assert np.isnan(out["excess"]) and out["n_victims"] == 0


def test_within_type_thresholds_undo_the_v9_defect():
    """Per-type offsets, no leak effect: a global threshold invents an excess."""
    t, score, victims, controls = _two_type_world(victim_shift=(0.0, 0.0))
    assert excess_fpr(score, t, victims, controls)["excess"] == pytest.approx(0.0, abs=0.02)
    glob = np.quantile(score[controls], 0.7)                # one global cut
    control_fpr = [(score[controls & (t == g)] > glob).mean() for g in (0, 1)]
    assert control_fpr[0] == 0.0 and control_fpr[1] > 0.5   # the V9 pathology:
    # the nominal 30% is nowhere near 30% in either type, so "victim FPR" under
    # a global cut is a readout of the type offsets, not of the leak.


def test_excess_is_invariant_to_a_global_monotone_shift():
    t, score, victims, controls = _two_type_world(victim_shift=(0.2, 0.2))
    a = excess_fpr(score, t, victims, controls)["excess"]
    b = excess_fpr(3.0 * score + 7.0, t, victims, controls)["excess"]
    assert a == pytest.approx(b)


# -- bootstrap -------------------------------------------------------------

def test_bootstrap_draws_are_paired_across_arms():
    t, score, victims, controls = _two_type_world(victim_shift=(0.2, 0.2))
    draws = bootstrap_excess({"a": score, "b": 2.0 * score + 1.0}, t, victims,
                             controls, np.random.default_rng(0), n_boot=20)
    assert np.allclose(draws["a"], draws["b"])              # same rows, same cut
    assert draws["a"].std() > 0


def test_bootstrap_ci_covers_the_true_null():
    t, score, victims, controls = _two_type_world(victim_shift=(0.0, 0.0))
    draws = bootstrap_excess({"a": score}, t, victims, controls,
                             np.random.default_rng(1), n_boot=200)
    assert np.percentile(draws["a"], 2.5) < 0.0 < np.percentile(draws["a"], 97.5)


# -- power gate ------------------------------------------------------------

def test_power_gate_passes_exactly_at_its_thresholds():
    assert power_gate(0.9, 0.1)["passed"] is True
    assert power_gate(0.9 - 1e-9, 0.1)["passed"] is False
    assert power_gate(0.9, 0.1 - 1e-9)["passed"] is False
    assert power_gate(1.0, 0.0)["passed"] is False          # plant seen, leak not


# -- scores ----------------------------------------------------------------

def test_leak_subtraction_recovers_the_clean_rate_from_noiseless_counts():
    rng = np.random.default_rng(0)
    rho = rng.dirichlet(np.ones(10), size=50)
    rho_bar = rng.dirichlet(np.ones(10), size=50)
    kappa, ell = 0.2, 300.0
    x = ell * ((1 - kappa) * rho + kappa * rho_bar)
    assert np.allclose(leak_subtracted_counts(x, rho_bar, kappa),
                       ell * (1 - kappa) * rho, atol=1e-8)


def test_leak_subtraction_clips_at_zero():
    x = np.array([[1.0, 0.0]])
    out = leak_subtracted_counts(x, np.array([[0.0, 1.0]]), 0.5)
    assert out.min() == 0.0


def test_programme_score_is_blind_to_library_size():
    rng = np.random.default_rng(0)
    rho = rng.dirichlet(np.ones(20), size=40)
    genes = np.array([1, 5, 9])
    a = programme_score(300.0 * rho, genes, median=200.0)
    b = programme_score(900.0 * rho, genes, median=200.0)
    assert np.allclose(a, b)


def test_gene_band_respects_the_rank_window_and_widens_only_when_it_must():
    mean_rate = np.linspace(0.0, 1.0, 60)                   # already sorted
    band = gene_band(mean_rate, 12, (0.2, 0.5))
    assert band.tolist() == list(range(12, 30))             # 18 genes, no widening
    wide = gene_band(mean_rate, 24, (0.2, 0.5))             # band too narrow
    assert len(wide) >= 24 and set(band) <= set(wide)


def test_z_probe_recovers_a_target_that_lives_in_one_z_coordinate():
    rng = np.random.default_rng(0)
    t = np.repeat(np.arange(4), 250)
    cycling = t < 2
    mu_z = rng.standard_normal((len(t), 3))
    target = 2.0 * mu_z[:, 1] + 0.01 * rng.standard_normal(len(t))
    fold = np.tile(np.arange(5), len(t) // 5)
    score = z_probe(mu_z, t, target, cycling, fold)
    assert np.corrcoef(score[cycling], target[cycling])[0, 1] > 0.99
    assert np.corrcoef(score[~cycling], target[~cycling])[0, 1] > 0.99


# -- read_out / verdict ----------------------------------------------------

class _World:
    def __init__(self):
        t, score, victims, controls = _two_type_world(victim_shift=(0.2, 0.2))
        self.sim = type("S", (), {"t": t})()
        self.cycling = np.zeros(len(t), bool)
        self.cycling[:50] = True                # disjoint from victims/controls
        self.planted = np.zeros(len(t), bool)
        self.planted[:25] = True
        self.victims, self.controls = victims, controls
        self.genes = np.array([0, 1])
        self.exposure = np.zeros(len(t))
        self.true_victims = np.zeros(len(t), bool)
        self.score = score


def test_read_out_finds_no_difference_between_an_arm_and_its_own_copy():
    w = _World()
    scores = {"raw": w.score, "z_raw": w.score.copy()}
    # a perfect plant call inside the cycling cells, so the gate can pass
    for s in scores.values():
        s[:50] = np.where(w.planted[:50], 1e3, -1e3)
    out = read_out(w, scores, np.random.default_rng(0), n_boot=50)
    assert out["power"]["passed"] is True
    assert out["tests"]["z_raw_below_raw"]["excess_difference"] == pytest.approx(0.0)
    assert out["tests"]["z_raw_below_raw"]["passed"] is False
    assert out["tests"]["z_raw_below_raw"]["reversed"] is False
    assert out["tests"]["amortisation_gap"]["z_inherits_leak"] is True


def test_verdict_needs_two_of_three_and_no_reversal():
    def seed(passed, reversed_):
        return {"power": {"passed": True},
                "tests": {"H1_xtilde_encoder": {"passed": passed,
                                                "reversed": reversed_}}}
    assert verdict([seed(True, False), seed(True, False),
                    seed(False, False)])["H1_xtilde_encoder"]["supported"] is True
    assert verdict([seed(True, False), seed(True, False),
                    seed(False, True)])["H1_xtilde_encoder"]["supported"] is False
    assert verdict([seed(True, False), seed(False, False),
                    seed(False, False)])["H1_xtilde_encoder"]["supported"] is False


def test_a_reduction_below_the_margin_does_not_count_as_beating_raw():
    """Pre-registered: a caller beats raw only if excess is >= 0.05 lower.

    The z_raw arm here cuts the victim FPR from 0.50 to 0.47 (excess 0.20 ->
    0.17): a real, consistent reduction with the paired CI above 0, but
    smaller than the margin, so it must not pass: the margin is the binding
    constraint (the difference never goes negative, AUROC is kept).
    """
    w = _World()
    z = w.score.copy()
    # demote the 3 % of victims closest above each type's threshold
    for g in (0, 1):
        v = np.flatnonzero(w.victims & (w.sim.t == g))
        z[v[np.argsort(z[v])[50:53]]] -= 1.0
    out = read_out(w, {"raw": w.score, "z_raw": z},
                   np.random.default_rng(0), n_boot=200)
    test = out["tests"]["z_raw_below_raw"]
    assert test["excess_difference"] == pytest.approx(0.03, abs=0.005)
    assert test["difference_ci"][0] >= 0 and test["reversed"] is False
    assert test["auroc_kept"] is True
    assert test["passed"] is False


def test_a4_planted_leg_delegates_to_the_gate(monkeypatch):
    """A4's planted world is the gate now (V9): its pass is the gate's."""
    import discell.applications.a4_cycle as a4
    import discell.applications.xtilde_gate as gate

    stub = {"power": {"passed": True},
            "arms": {"raw": {"excess": 0.3, "auroc_cycling": 1.0},
                     "z_raw": {"excess": 0.1, "auroc_cycling": 1.0}},
            "tests": {"z_raw_below_raw": {"passed": True}}}
    monkeypatch.setattr(gate, "run_seed", lambda *a, **k: dict(stub))
    out = a4.planted_world(seed=0)
    assert out["pass"] is True
    assert out["raw"]["excess"] == 0.3 and out["z"]["excess"] == 0.1

    stub["tests"]["z_raw_below_raw"]["passed"] = False
    assert a4.planted_world(seed=0)["pass"] is False


def test_a4_planted_leg_reports_a_failed_power_gate_without_arms(monkeypatch):
    import discell.applications.a4_cycle as a4
    import discell.applications.xtilde_gate as gate

    monkeypatch.setattr(gate, "run_seed", lambda *a, **k: {
        "power": {"passed": False}, "arms": {}, "tests": {},
        "skipped": "power gate failed before the fits"})
    out = a4.planted_world(seed=0)
    assert out["pass"] is False and "raw" not in out


# -- (c) recall on genuinely planted cells ---------------------------------

def _recall_world(shift=0.5, n_true=80, offsets=(0.0, 100.0)):
    """Controls as in _two_type_world; *true victims* are a third group of the
    same types, shifted by *shift* -- planted cells with no leaked neighbour."""
    t, score, controls, true_victims = [], [], [], []
    for g, off in enumerate(offsets):
        grid = (np.arange(200) + 0.5) / 200
        t += [g] * 200
        score += list(off + grid)
        controls += [True] * 200
        true_victims += [False] * 200
        vgrid = (np.arange(n_true) + 0.5) / n_true
        t += [g] * n_true
        score += list(off + vgrid + shift)
        controls += [False] * n_true
        true_victims += [True] * n_true
    return (np.array(t), np.array(score), np.array(true_victims),
            np.array(controls))


def test_recall_is_the_hand_counted_fraction_above_the_control_threshold():
    # threshold is off + 0.7; planted cells sit at off + vgrid + 0.5, so the
    # called ones are those with vgrid > 0.2 -> recall 0.8, in both types.
    t, score, true_victims, controls = _recall_world(shift=0.5)
    out = recall_at_threshold(score, t, true_victims, controls, rate=0.3)
    assert out["recall"] == pytest.approx(0.8, abs=0.02)
    assert out["n_true_victims"] == 160
    assert set(out["per_type"]) == {0, 1}


def test_recall_is_one_when_the_plant_is_far_above_the_threshold():
    t, score, true_victims, controls = _recall_world(shift=10.0)
    assert recall_at_threshold(score, t, true_victims, controls)["recall"] == 1.0


def test_recall_uses_the_same_within_type_thresholds_as_the_fpr():
    """The per-type offsets must cancel here too, or (c) would read them."""
    t, score, true_victims, controls = _recall_world(shift=0.5)
    a = recall_at_threshold(score, t, true_victims, controls)["per_type"]
    assert a[0]["recall"] == pytest.approx(a[1]["recall"], abs=1e-9)


def test_recall_drops_types_below_min_cells_and_returns_nan_when_empty():
    t, score, true_victims, controls = _recall_world(shift=0.5, n_true=80)
    few = true_victims & ((t == 0) | (np.cumsum(true_victims & (t == 1)) <= 5))
    assert list(recall_at_threshold(score, t, few, controls)["per_type"]) == [0]
    empty = recall_at_threshold(score, t, np.zeros(len(t), bool), controls)
    assert np.isnan(empty["recall"]) and empty["n_true_victims"] == 0


def _carve_true_victims(w, per_type=50):
    """Move the *per_type* best-scoring victims of each type into the genuinely
    planted group: their scores (vgrid + 0.2, vgrid > 0.5) are all above that
    type's control threshold (0.7), so the base recall is exactly 1."""
    keep = np.zeros(len(w.score), bool)
    for g in (0, 1):
        v = np.flatnonzero(w.victims & (w.sim.t == g))
        keep[v[np.argsort(w.score[v])[-per_type:]]] = True
    w.true_victims, w.victims = keep, w.victims & ~keep
    return w


def test_read_out_reports_recall_per_arm_and_the_margin_against_plain_z():
    w = _carve_true_victims(_World(), per_type=30)
    z = w.score.copy()
    blunt = w.score.copy()
    blunt[w.true_victims] -= 1e3                     # calls none of them
    out = read_out(w, {"raw": w.score, "z_raw": z, "z_xtilde": blunt},
                   np.random.default_rng(0), n_boot=50)
    assert out["n_true_victims"] == 60
    assert out["arms"]["z_raw"]["recall"]["recall"] == 1.0
    assert out["arms"]["z_xtilde"]["recall"]["recall"] == 0.0
    test = out["tests"]["recall_z_xtilde_vs_z_raw"]
    assert test["delta"] < -RECALL_MARGIN and test["within_margin"] is False


def test_recall_within_the_margin_passes_even_when_it_falls_a_little():
    w = _carve_true_victims(_World(), per_type=50)
    z = w.score.copy()
    slightly = w.score.copy()
    slightly[np.flatnonzero(w.true_victims)[:2]] -= 1e3   # 2 of 100 lost
    out = read_out(w, {"raw": w.score, "z_raw": z, "z_xtilde": slightly},
                   np.random.default_rng(0), n_boot=50)
    test = out["tests"]["recall_z_xtilde_vs_z_raw"]
    assert test["delta"] == pytest.approx(-0.02, abs=1e-9)
    assert test["within_margin"] is True


# -- (c) the world ---------------------------------------------------------

def test_true_victims_are_non_cycling_unexposed_cells_of_the_named_types():
    w = build_world(0, share=0.1, true_victim_rate=0.1)
    t = w.sim.t
    assert w.true_victims.sum() > 0
    assert np.isin(t[w.true_victims], TRUE_VICTIM_TYPES).all()
    assert not w.cycling[w.true_victims].any()
    assert (w.exposure[w.true_victims] == 0).all()    # no planted cycling neighbour
    assert w.planted[w.true_victims].all()            # they carry the programme
    # the three groups are disjoint, and controls see no planted cell at all
    assert not (w.victims & w.true_victims).any()
    assert not (w.controls & (w.victims | w.true_victims | w.planted)).any()
    total = np.asarray(w.sim.graph.in_edges @ w.planted.astype(float)).ravel()
    assert (total[w.controls] == 0).all()


def test_the_arm_a_world_is_unchanged_when_no_true_victims_are_asked_for():
    a = build_world(1, share=0.1)
    b = build_world(1, share=0.1, true_victim_rate=0.0)
    assert not a.true_victims.any()
    assert np.array_equal(a.planted, b.planted) and np.array_equal(a.sim.x, b.sim.x)
    assert np.array_equal(a.genes, b.genes)


# -- (b) the plant-share search --------------------------------------------

def _stub_search(monkeypatch):
    """A world whose raw AUROC is a known increasing function of the share.

    planted cells score ``6 * share`` above the controls on a fixed uniform
    grid, so AUROC = 1 - (1 - 6 share)^2 / 2 and the raw excess is 1.2 share:
    both halves of the power gate move with the share, as they do in the real
    world, and the search's arithmetic can be checked without a simulation.
    """
    import discell.applications.xtilde_gate as gate

    n = 400
    t = np.repeat(np.arange(2), n // 2)
    planted = np.zeros(n, bool)
    planted[: n // 4] = True
    noise = (np.arange(n) % (n // 4) + 0.5) / (n // 4)       # uniform in [0, 1)
    gain, excess_per_share = 6.0, 1.2
    state = {}

    def fake_build_world(seed, share=0.25, n_genes=12, **kw):
        state["share"] = share
        return gate.World(sim=type("S", (), {"t": t, "x": np.zeros((n, 2)),
                                             "totals": np.ones(n)})(),
                          planted=planted, genes=np.array([0]),
                          cycling=np.ones(n, bool), exposure=np.zeros(n),
                          victims=np.zeros(n, bool), controls=np.zeros(n, bool),
                          true_victims=np.zeros(n, bool))

    monkeypatch.setattr(gate, "build_world", fake_build_world)
    monkeypatch.setattr(gate, "programme_score",
                        lambda x, genes, median=None: noise + gain * state["share"] * planted)
    monkeypatch.setattr(gate, "excess_fpr",
                        lambda *a, **k: {"excess": excess_per_share * state["share"]})
    return state


def test_share_search_halves_until_the_raw_auroc_leaves_saturation(monkeypatch):
    _stub_search(monkeypatch)
    out = search_share(0, start=0.25, target_auroc=0.92, refine=0)
    assert [row["share"] for row in out["trace"][:3]] == [0.25, 0.125, 0.0625]
    assert out["trace"][1]["auroc_raw"] >= 0.92        # 2 * 0.125 = 0.25 spread
    assert out["chosen"]["auroc_raw"] < 0.92
    assert out["share"] == out["chosen"]["share"]


def test_share_search_bisects_back_up_when_halving_breaks_the_power_gate(monkeypatch):
    """Halving overshoots: the refinement returns the largest share still below
    the target AUROC, and that share is where the gate is checked."""
    _stub_search(monkeypatch)
    coarse = search_share(0, start=0.25, target_auroc=0.92, refine=0)
    fine = search_share(0, start=0.25, target_auroc=0.92, refine=6)
    assert fine["share"] > coarse["share"]
    assert fine["chosen"]["auroc_raw"] < 0.92
    assert fine["chosen"]["auroc_raw"] > coarse["chosen"]["auroc_raw"]
    assert fine["chosen"]["passed"] is True            # excess back above 0.1
    assert len(fine["trace"]) == len(coarse["trace"]) + 6


def test_share_search_stops_at_the_first_share_when_the_world_starts_unsaturated(monkeypatch):
    _stub_search(monkeypatch)
    out = search_share(0, start=0.05, target_auroc=0.92, refine=6)
    assert len(out["trace"]) == 1 and out["share"] == 0.05


# -- the follow-up rule ----------------------------------------------------

def _ab_seed(passed, reversed_=False, auroc_kept=True):
    return {"tests": {"H1_xtilde_encoder": {"passed": passed, "reversed": reversed_,
                                            "auroc_kept": auroc_kept}}}


def _c_seed(delta):
    return {"tests": {"recall_z_xtilde_vs_z_raw": {"delta": delta,
                                                   "within_margin": delta >= -0.03}}}


def test_follow_up_rule_needs_eight_of_twelve_on_the_margin():
    ab = [_ab_seed(True)] * 8 + [_ab_seed(False)] * 4
    assert follow_up_verdict(ab, [_c_seed(-0.01)])["verdict"] == "default-on"
    seven = [_ab_seed(True)] * 7 + [_ab_seed(False)] * 5
    assert follow_up_verdict(seven, [_c_seed(-0.01)])["verdict"] == "option-only"


def test_follow_up_rule_is_vetoed_by_one_reversal_one_auroc_loss_or_lost_recall():
    ab = [_ab_seed(True)] * 8 + [_ab_seed(False)] * 4
    assert follow_up_verdict(ab[:-1] + [_ab_seed(False, reversed_=True)],
                             [_c_seed(-0.01)])["default_on"] is False
    assert follow_up_verdict(ab[:-1] + [_ab_seed(False, auroc_kept=False)],
                             [_c_seed(-0.01)])["default_on"] is False
    assert follow_up_verdict(ab, [_c_seed(-0.01), _c_seed(-0.05)])["default_on"] is False
    assert follow_up_verdict(ab, [])["default_on"] is False   # (c) never run


def test_follow_up_rule_does_not_punish_a_recall_gain():
    ab = [_ab_seed(True)] * 12
    out = follow_up_verdict(ab, [_c_seed(0.04)])
    assert out["recall_within_margin_every_seed"] is True and out["default_on"] is True
