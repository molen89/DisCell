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
    # ridge and no trusted transport panel: those groups write nothing
    assert out["cycle"] == {} and out["transport_mean"] == {}
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
