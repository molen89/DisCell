"""Per-draw statistics of the 8.19 family on the eval-mask synthetic run:
every group writes its npz, the point estimates are the reads on disk
(where the synthetic run has them), and the opt-in twin cells of
``bootstrap.transport_dist_panels`` leave the default read unchanged."""

from __future__ import annotations

import json

import numpy as np
import pytest
import test_model_eval_mask as TE

from discell.experiments import breakdown as B

synthetic_run = TE.synthetic_run


@pytest.mark.parametrize("synthetic_run", [(TE.WITH_U, False)], indirect=True)
def test_groups_write_draws_and_reproduce(synthetic_run):
    import torch

    from discell import paths
    from discell.experiments import bootstrap as BS
    from discell.experiments import breakdown_draws as D
    from discell.model import transport as T
    from discell.model.validate import collect_latents, load_run, niche_labels

    torch.set_num_threads(1)
    assert T.main(["--dataset", TE.DS, "--run", TE.RUN, "--device", "cpu",
                   "--niches", "10", "--figures", "0", "--hvg", "50",
                   "--boot", "20", "--read", "both"]) == 0
    out = D.run(TE.DS, TE.RUN, ["cycle", "w_mi", "transport_mean",
                                "transport_dist"], n=40, device="cpu",
                boot_replayed=20)
    run_dir = paths.dataset(TE.DS).root / "runs" / TE.RUN
    # the synthetic slide has too few held-out cycling cells for the q90
    # ridge: that group writes nothing. The transport members are scored on
    # all panels (author 2026-09-29), so they exist and reproduce the read
    assert out["cycle"] == {}
    assert out["transport_mean"] and all(
        m["reproduces"] for m in out["transport_mean"].values())
    assert not B.draws_path(run_dir, "cycle").exists()
    for group in ("w_mi", "transport_dist"):
        members = B.read_draws(B.draws_path(run_dir, group))
        for m in members.values():
            assert np.isfinite(m["estimate"]) and len(m["draws"]) == 40
            assert m["method"] == "percentile"
    dist = out["transport_dist"]
    for key in ("readA_minus_typemean", "readB_twin_margin"):
        if key in dist:
            assert dist[key]["reproduces"] is True, (key, dist[key])
    # the default replay has no twin cells; the opt-in adds only them
    config, data, trainer, _, _ = load_run(TE.DS, TE.RUN, "cpu")
    labels = niche_labels(data, 10, config.seed)
    fold = collect_latents(trainer, data)["fold"]
    a = BS.transport_dist_panels(trainer, data, config, labels, fold, "cpu",
                                 20, n=5)
    b = BS.transport_dist_panels(trainer, data, config, labels, fold, "cpu",
                                 20, n=5, twin_cells=True)
    for pa, pb in zip(a["panels"], b["panels"]):
        assert "twins_own" not in pa and "twins_own" in pb
        if not pa["own"]["insufficient"]:
            assert json.dumps(pa["own"]["mmd2"]) == json.dumps(pb["own"]["mmd2"])


def test_draw_seeds_differ_by_run_seed_and_group():
    from discell.experiments.breakdown_draws import GROUPS, draw_seed

    seeds = {draw_seed(g, s) for g in GROUPS for s in range(3)}
    assert len(seeds) == 3 * len(GROUPS)


@pytest.mark.parametrize("synthetic_run", [(TE.WITH_U, False)], indirect=True)
def test_eval_dataset_reads_match_own_reads_on_the_same_section(synthetic_run):
    """The opt-in ``--eval-dataset`` variants (devlog 2026-10-01, the serial
    section's members): applied to the section the fit was trained on (the
    fixture's ``assemble`` answers every dataset id with the same data), the
    transport, twin, signalling-share and w-guard reads and every group's
    draws equal the run's own, and they land under
    ``runs/<run>/crossslide/<eval>/`` without touching the own reads."""
    import hashlib

    import torch

    from discell import paths
    from discell.experiments import breakdown_draws as D
    from discell.experiments import external_criteria as E
    from discell.model import degeneracy as DG
    from discell.model import transport as T
    from discell.model.validate import load_run

    torch.set_num_threads(1)
    alias = "evalmask_dual"
    run_dir = paths.dataset(TE.DS).root / "runs" / TE.RUN
    cross = run_dir / "crossslide" / alias
    common = ["--dataset", TE.DS, "--run", TE.RUN, "--device", "cpu",
              "--niches", "10", "--figures", "0", "--hvg", "50", "--boot",
              "20", "--read", "both"]
    assert T.main(common) == 0
    assert E.main(["signalling-share", "--dataset", TE.DS, "--run", TE.RUN,
                   "--device", "cpu"]) == 0

    def digest(root):
        return {p.relative_to(root).as_posix():
                hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*.json"))
                if "crossslide" not in p.parts}
    own_files = digest(paths.dataset(TE.DS).root)
    assert T.main(common + ["--eval-dataset", alias]) == 0
    assert E.main(["signalling-share", "--dataset", TE.DS, "--run", TE.RUN,
                   "--device", "cpu", "--eval-dataset", alias]) == 0
    assert DG.main(["--dataset", TE.DS, "--run", TE.RUN, "--device", "cpu",
                    "--eval-dataset", alias]) == 0
    assert digest(paths.dataset(TE.DS).root) == own_files   # nothing own moved

    def load(p):
        return json.loads(p.read_text())
    for name in ("transport.json", "transport_twins.json",
                 "transport_distribution.json"):
        a = load(run_dir / "transport" / name)
        b = load(cross / "transport" / name)
        assert b.pop("evaluated_on") == alias
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    a = load(paths.dataset(TE.DS).root / "experiments"
             / f"external_signalling_share_{TE.RUN}.json")
    b = load(cross / f"external_signalling_share_{TE.RUN}.json")
    assert b.pop("evaluated_on") == alias and a == b
    _, _, trainer, _, _ = load_run(TE.DS, TE.RUN, "cpu")
    own_guard = DG.w_channel_guard_from_trainer(trainer)
    assert load(cross / "degeneracy.json")["w_channel"] == json.loads(
        json.dumps(own_guard, default=float))
    # the draws: same estimates and draws, every point estimate reproduces
    # the cross read on disk, written to dual__<eval>__<group>.npz
    (run_dir / "degeneracy.json").write_text(json.dumps(
        {"w_channel": own_guard}, default=float))
    # (signalling: the synthetic slide has no panel with >= 100 kept genes,
    # so its draws are covered by the real-data smoke; the share read above)
    groups = ["w_mi", "transport_mean", "transport_dist"]
    own = D.run(TE.DS, TE.RUN, groups, n=40, device="cpu", boot_replayed=20)
    got = D.run(TE.DS, TE.RUN, groups, n=40, device="cpu", boot_replayed=20,
                eval_dataset=alias)
    for g in groups:
        assert got[g], g
        for k, v in got[g].items():
            assert v["reproduces"] is True, (g, k, v)
            assert v["estimate"] == own[g][k]["estimate"], (g, k)
        a = B.read_draws(B.draws_path(run_dir, g))
        b = B.read_draws(B.draws_path(run_dir, g, alias))
        assert set(a) == set(b)
        for k in a:
            assert np.array_equal(a[k]["draws"], b[k]["draws"],
                                  equal_nan=True), (g, k)
    with pytest.raises(ValueError):
        D.run(TE.DS, TE.RUN, ["axis"], n=4, device="cpu", eval_dataset=alias)
