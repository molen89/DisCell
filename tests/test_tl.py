"""The readouts (``discell.tl``): numerics on constructed inputs, and every readout
end to end on a small CPU fit."""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

import discell as dc
from discell.sim import simulate
from discell.tl import _utils as U
from discell.tl import relocate as R

SMALL = dict(label_key="cell_type", image_key="X_image", d_z=4, d_w=3, hidden=32, gat_dim=8,
             v_pcs=4, tile_cells=400, epochs=6, eval_every=2, w_warmup_epochs=2,
             patience=100, device="cpu")


def section(seed: int = 0, n_cells: int = 3000, unassigned: int = 60):
    adata = simulate(n_cells=n_cells, n_genes=30, kappa=0.1, seed=seed).to_anndata()
    labels = adata.obs["cell_type"].astype(object).to_numpy(copy=True)
    labels[:unassigned] = None
    adata.obs["cell_type"] = labels
    dc.build_graph(adata, "cell_type")
    return adata


@pytest.fixture(scope="module")
def fitted():
    adata = section()
    model = dc.fit(adata, dc.Config(**SMALL))
    return adata, model


# -- numerics ----------------------------------------------------------------------



def test_alignment_is_the_program_part_of_the_r2_contrast():
    rng = np.random.default_rng(0)
    o, p, s = rng.normal(size=(3, 200)) + np.array([[1.0], [0.3], [-0.2]])
    cf = R.score_shift(p + s, o)["r2"]
    spill = R.score_shift(s, o)["r2"]
    oc, pc = o - o.mean(), p - p.mean()
    penalty = (pc ** 2).sum() / (oc ** 2).sum()
    assert cf - spill == pytest.approx(R.alignment(o, p, s) - penalty)


def test_noise_ceiling_reads_signal_and_noise():
    rng = np.random.default_rng(0)
    n, g = 400, 200
    profile_a, profile_b = rng.gamma(2.0, 1.0, g), rng.gamma(2.0, 1.0, g)
    x = np.vstack([rng.poisson(50 * profile_a, (n, g)), rng.poisson(50 * profile_b, (n, g))])
    rate = sp.csr_matrix(x / x.sum(axis=1, keepdims=True))
    rows_a, rows_b = np.arange(n), np.arange(n, 2 * n)
    keep = np.ones(g, bool)
    assert R.noise_ceiling(rate, rows_a, rows_b, keep, rng) > 0.95
    same = sp.csr_matrix(np.vstack([rng.poisson(50 * profile_a, (2 * n, g))]) / 1.0)
    assert R.noise_ceiling(same, rows_a, rows_b, keep, rng) < 0.2
    tiles = np.repeat(np.arange(8), n // 4)
    assert R.noise_ceiling(rate, rows_a, rows_b, keep, rng, split="tiles",
                           tile_of=tiles) > 0.9
    one_tile = np.zeros(2 * n, dtype=np.int64)
    assert np.isnan(R.noise_ceiling(rate, rows_a, rows_b, keep, rng, split="tiles",
                                    tile_of=one_tile))


# -- readouts on a fit -------------------------------------------------------------

def test_spatial_folds_follow_the_fit_tiles(fitted):
    adata, model = fitted
    data = model.assemble(adata)
    fold = dc.tl.spatial_folds(adata)
    for k, tile in enumerate(data.train_tiles + data.val_tiles):
        assert (fold[tile] == k % U.N_FOLDS).all()


def test_readouts_need_a_fit():
    adata = section(n_cells=300, unassigned=0)
    with pytest.raises(KeyError):
        dc.tl.record(adata, "relocation")


def test_decoded_shift():
    rho = np.array([0.0004, 0.4996, 0.5])
    bar_a, bar_b = np.array([0.0003, 0.5, 0.4997]), np.array([0.03, 0.47, 0.5])
    program = np.array([1.74, 0.0, 0.0])
    base = np.log(R._mixture(rho, bar_a, 0.1) + R.EPS)
    spillover = np.log(R._mixture(rho, bar_b, 0.1) + R.EPS) - base
    # with no response it is the spillover part, with no spillover the response alone
    assert R.decoded_shift(rho, bar_a, bar_b, 0 * program, 0.1) == pytest.approx(spillover)
    moved = rho * np.exp(program)
    moved /= moved.sum()
    assert R.decoded_shift(rho, rho, rho, program, 0.0) == pytest.approx(
        np.log(moved + R.EPS) - np.log(rho + R.EPS))
    # a gene with both parts large: adding them overshoots the decoded shift
    assert R.decoded_shift(rho, bar_a, bar_b, program, 0.1)[0] < program[0] + spillover[0]


def test_relocation_pairs_and_tiers(fitted):
    adata, model = fitted
    adata = adata.copy()
    rec = dc.tl.relocation(model, adata, n_niches=3, min_cells=40)
    pairs = rec["pairs"]
    assert pairs and all(p["type"] != "Unassigned" for p in pairs)
    for p in pairs:
        assert p["noise_ceiling"] <= 1.0 and p["n_genes"] <= adata.n_vars
        assert p["trusted"] == (p["noise_ceiling"] >= R.TRUST_CEILING
                                and p["n_genes"] >= R.TRUST_GENES)
        for key in R.PREDICTIONS:
            assert set(p[key]) == {"r2", "slope", "corr"}
    tiers = rec["summary"]
    assert sum(tiers[k]["n_pairs"] for k in ("supported", "extrapolation") if k in tiers) \
        == len(pairs)
    assert rec["alignment"] == pytest.approx(np.mean([p["alignment"] for p in pairs]))
    assert adata.obs["discell_niche"].max() == 2
    assert json.dumps(dc.tl.record(adata, "relocation")["pairs"]) == json.dumps(pairs)
    # deterministic, and given niches are used as given
    again = dc.tl.relocation(model, adata, n_niches=3, min_cells=40)
    assert json.dumps(again, sort_keys=True) == json.dumps(rec, sort_keys=True)
    given = dc.tl.relocation(model, adata, niches=adata.obs["discell_niche"].to_numpy(),
                             min_cells=40)
    assert given["niche_source"] == "given"
    assert json.dumps(given["pairs"]) == json.dumps(rec["pairs"])


def test_group_channels_reuses_a_given_runner(fitted):
    adata, model = fitted
    data = model.assemble(adata)
    group = np.where(data.t >= 0, data.t, -1).astype(np.int64)
    n_groups = int(group.max()) + 1
    fresh = R.group_channels(model, data, group, n_groups, context=True)
    reused = R.group_channels(model, data, group, n_groups, context=True,
                              runner=model.runner(data))
    assert fresh.keys() == reused.keys()
    for key in fresh:
        np.testing.assert_array_equal(fresh[key], reused[key])


def test_cross_section(fitted):
    adata, model = fitted
    other = section(seed=1, n_cells=1500)
    reads = dc.tl.cross_section(model, other)
    assert other.obsm["X_discell_w"].shape == (other.n_obs, 3)
    assert np.isfinite(reads["recon_all_tiles"]) and reads["n_cells"] == other.n_obs
    assert dc.tl.record(other, "cross_section")["n_tiles"] == reads["n_tiles"]
    wrong = section(seed=1, n_cells=1500)
    wrong.obs["cell_type"] = wrong.obs["cell_type"].astype(object).replace("type3", "type9")
    dc.build_graph(wrong, "cell_type")
    with pytest.raises(ValueError):
        dc.tl.cross_section(model, wrong)
