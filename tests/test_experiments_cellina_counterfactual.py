"""Cellina counterfactual adapter (discell/experiments/cellina_counterfactual.py)
and the niche-domain label (cellina_niche_domain.py)."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from discell.data.loader import UNASSIGNED
from discell.experiments import bootstrap as BS
from discell.experiments import cellina_counterfactual as CF
from discell.experiments import cellina_niche_domain as ND


def _planted_net(w_s_scale: float, g: int = 200, seed: int = 0):
    """p = softmax(W [z; s] + b): a known decoder whose s-response is planted."""
    gen = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(4, g)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(g, 4, generator=gen) * 0.6)
        lin.weight[:, 2:] *= w_s_scale
        lin.bias.copy_(torch.randn(g, generator=gen))
    return torch.nn.Sequential(lin, torch.nn.Softmax(dim=-1))


def _planted_slide(net, n_per: int = 2400, g: int = 200, seed: int = 0):
    """Two niches (x < 0: A, x > 0: B) of one type plus an Unassigned type.
    z is drawn from the same law in both niches and s is the niche's context,
    so the population shift the data show is exactly Cellina's rewiring
    counterfactual (A's cells, own z, B's s): the read must find it."""
    rng = np.random.default_rng(seed)
    n = 4 * n_per
    niche = np.repeat([0, 1, 0, 1], n_per)
    t = np.repeat([0, 0, 1, 1], n_per)
    xy = np.stack([np.where(niche == 0, -1, 1) * rng.uniform(10, 2000, n),
                   rng.uniform(0, 2000, n)], axis=1)
    tile = (np.floor((xy[:, 0] + 2000) / 250) * 16 + np.floor(xy[:, 1] / 250)).astype(int)
    _, tile = np.unique(tile, return_inverse=True)
    fold = tile % 4                                  # fold 0 = held-out tiles
    z = rng.normal(size=(n, 2)).astype(np.float32)
    s = (np.stack([niche, 1 - niche], 1) * 1.5
         + 0.1 * rng.normal(size=(n, 2))).astype(np.float32)
    p = CF.Decoder(net).decode(z, s).astype(np.float64)
    p /= p.sum(1, keepdims=True)
    depth = 3000
    x = sp.csr_matrix(rng.multinomial(depth, p).astype(np.float32))
    y = np.stack([1.0 - niche, niche.astype(float)], 1) + 0.02 * rng.random((n, 2))
    sl = {"labels": niche.astype(np.int64), "held_out": fold == 0, "t": t,
          "connected": np.ones(n, bool), "y": y,
          "x_rate": x.multiply(1.0 / depth).tocsr(), "tile_of": tile,
          "positions": xy, "names": ["Tcell", UNASSIGNED],
          "totals": np.full(n, float(depth)), "hvg": None}
    return sl, z, s


@pytest.fixture(scope="module")
def planted():
    net = _planted_net(1.0)
    sl, z, s = _planted_slide(net)
    return sl, z, s, net


def test_rewiring_counterfactual_recovers_the_planted_shift(planted):
    sl, z, s, net = planted
    res, boot = CF.mean_read(sl, z, s, CF.Decoder(net), seed=0)
    assert res["eval_mask"]["n_panels_dropped"] == 1        # Unassigned drawn, dropped
    assert [p["type"] for p in res["panels"]] == ["Tcell"]
    panel = res["panels"][0]
    assert panel["overlap_flag"] and panel["trusted"]
    assert panel["counterfactual"]["r2"] > 0.9
    assert 0.9 < panel["counterfactual_of_ceiling"] < 1.1
    assert abs(panel["counterfactual"]["slope"] - 1) < 0.1
    tier = res["summary"]["extrapolation_trusted"]
    assert tier["counterfactual_of_ceiling"] == pytest.approx(
        panel["counterfactual_of_ceiling"])
    assert tier["program_only"] is None and tier["full_beats_both"] is None
    assert panel["program_only"]["not_mapped"]
    ci = BS.transport_mean_bootstrap(boot, sl["x_rate"], sl["positions"], n=30)
    lo, hi = ci["reads"]["transport_of_ceiling_trusted"]["ci95"]
    assert lo < panel["counterfactual_of_ceiling"] < hi


def test_a_decoder_blind_to_s_predicts_no_shift(planted):
    sl, z, s, _ = planted
    res, _ = CF.mean_read(sl, z, s, CF.Decoder(_planted_net(0.0)), seed=0)
    panel = res["panels"][0]
    assert abs(panel["counterfactual"]["r2"]) < 1e-6        # zero prediction
    assert panel["noise_ceiling"] > 0.9                      # same data, same ceiling


def test_distribution_reads_close_the_gap_only_through_s(planted):
    sl, z, s, net = planted
    res, twins, ci = CF.distribution_read(sl, z, s, CF.Decoder(net), seed=0, boot=20,
                                          device="cpu", n_draws=20)
    assert len(res["pairwise"]) == 1 and res["eval_mask"]["n_pairwise_dropped"] == 1
    assert len(res["leave_one_out"]) == 2
    own = res["summary_model_own"]["pairwise"]
    assert own["median_gap_closed"] > 0.5
    assert res["summary_model"]["pairwise"]["median_gap_closed"] > 0.5
    assert twins["summary_own"]["pairwise"]["n_panels"] == 1
    assert set(ci) == {"readA_gap_group", "readA_gap_own", "readA_type_mean_own"}
    blind = CF.distribution_read(sl, z, s, CF.Decoder(_planted_net(0.0)), seed=0,
                                 boot=20, device="cpu", n_draws=20)[0]
    assert abs(blind["summary_model_own"]["pairwise"]["median_gap_closed"]) < 1e-6
    # the data-side draws do not depend on the model: raw-target floors agree
    assert [p["scores"]["mmd2"]["floor"] for p in blind["pairwise"]] == \
        [p["scores"]["mmd2"]["floor"] for p in res["pairwise"]]


def test_the_reads_are_repeatable_bit_for_bit(planted):
    sl, z, s, net = planted
    dec = CF.Decoder(net)
    assert np.array_equal(dec.decode(z[:500], s[:500]), dec.decode(z[:500], s[:500]))
    a = CF.mean_read(sl, z, s, dec, seed=0)[0]
    b = CF.mean_read(sl, z, s, CF.Decoder(net), seed=0)[0]
    assert a["panels"][0]["counterfactual"] == b["panels"][0]["counterfactual"]
    assert a["panels"][0]["noise_ceiling"] == b["panels"][0]["noise_ceiling"]


def test_the_decoder_reimplementation_is_scvi_decoderscvi():
    scvi_nn = pytest.importorskip("scvi.nn")
    torch.manual_seed(0)
    ref = scvi_nn.DecoderSCVI(16, 50, n_layers=2, n_hidden=32, inject_covariates=True,
                              use_batch_norm=True, use_layer_norm=False)
    with torch.no_grad():
        for m in ref.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.normal_()
                m.running_var.uniform_(0.5, 2.0)
                m.weight.normal_()
    ref.eval()
    x = torch.randn(64, 16)
    want = ref("gene", x, torch.zeros(64, 1))[0].detach().numpy()
    dec = CF.Decoder(CF.cellina_decoder(ref.state_dict()))
    got = dec.decode(x[:, :8].numpy(), x[:, 8:].numpy())
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dec.mean(x[:, :8].numpy(), x[:, 8:].numpy()),
                               want.mean(0), rtol=1e-5, atol=1e-8)


def test_niche_domain_fits_on_training_cells_and_isolated_get_their_own_class():
    rng = np.random.default_rng(0)
    y = np.vstack([rng.normal([0, 0], 0.1, (300, 2)), rng.normal([5, 5], 0.1, (300, 2)),
                   rng.normal([0, 9], 0.1, (100, 2))])
    train = np.r_[np.ones(600, bool), np.zeros(100, bool)]   # third blob: held out
    connected = np.ones(700, bool)
    connected[[5, 400]] = False
    centres = ND.fit_centres(y, connected, train, k=2, seed=0)
    labels = ND.assign(y, connected, centres)
    assert labels[5] == labels[400] == 2                     # isolated -> class K
    assert len(set(labels[:300][connected[:300]])) == 1
    assert labels[0] != labels[300]
    # the held-out blob did not get a centre of its own
    assert np.min(((centres - [0, 9]) ** 2).sum(1)) > 1.0
