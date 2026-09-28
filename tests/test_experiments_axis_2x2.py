"""Axis-test 2x2: planted band, two vocabularies, two planted models.

Synthetic slide: tumour probability rises along x, so the tumour band runs
along x and y is the orthogonal (false) axis. Old vocabulary: two tumour
states, a small "lining" patch that the old matcher calls tumour, and one
fibroblast class; lineage vocabulary: states merged, lining its own class.
Two planted models behind a fake trainer: w's prior mean follows the band
coordinate (``banded``) or the orthogonal one (``orthogonal``). The read
must see the plant on the true axis only for ``banded`` and on the false
axis only for ``orthogonal``, in every column; model-free rows must not move
with the model; the column must change which classes are read and which
cells are tumour; and a protocol column must be ``axis_test`` itself.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from discell.experiments import axis_2x2 as A
from discell.experiments import external_criteria as E
from discell.model.labels import is_tumour

OLD_NAMES = ["Tumor Cells", "Proliferative Tumor Cells",
             "Malignant Cells Lining Cyst", "Tumor Associated Fibroblasts"]
NEW_NAMES = ["Fibroblasts", "Mesothelial-like cyst lining", "Tumour"]
OLD_TO_NEW = np.array([2, 2, 1, 0])


class FakeModel:
    """Per-cell planted outputs, returned for a batch's seed nodes."""

    def __init__(self, prior_w, rho):
        self.prior_w = torch.as_tensor(prior_w, dtype=torch.float32)
        self.rho = torch.as_tensor(rho, dtype=torch.float32)

    def __call__(self, nodes, kappa, sample):
        idx = torch.as_tensor(nodes)
        rho = self.rho[idx]
        return SimpleNamespace(prior_mean_w=self.prior_w[idx], mu_w=self.prior_w[idx],
                               mu_z=torch.zeros(len(idx), 2), c=torch.zeros(len(idx), 1),
                               log_rho=rho.log(), rho_bar=rho)


class FakeTrainer:
    def __init__(self, model, tiles, n_train):
        self.model = model
        self.config = SimpleNamespace(kappa=0.1)
        batches = [{"nodes": tile, "n_seeds": len(tile)} for tile in tiles]
        self.train_batches, self.val_batches = batches[:n_train], batches[n_train:]

    @staticmethod
    def _forward_kwargs(batch):
        return {"nodes": batch["nodes"]}

    def _sweep(self, batches):
        nodes = np.concatenate([b["nodes"] for b in batches])
        w = self.model.prior_w[torch.as_tensor(nodes)].numpy()
        return {"nodes": nodes, "mu_z": np.zeros((len(nodes), 2)), "mu_w": w}


def planted_world(n: int = 100_000, n_genes: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1000, size=(n, 2))
    field = pos[:, 0] / 1000                       # the band coordinate
    tumour = rng.random(n) < field
    t_old = np.where(tumour, (rng.random(n) < 0.5).astype(np.int64), 3)
    lining = (~tumour) & (pos[:, 0] < 250) & (pos[:, 1] > 800) & (rng.random(n) < 0.5)
    t_old[lining] = 2
    t_new = OLD_TO_NEW[t_old]

    # tiles: a 20 x 20 grid, ids shuffled so held-out tiles (k % 5 == 0)
    # spread over the slide
    cell_tile = (pos[:, 0] // 50).astype(int) * 20 + (pos[:, 1] // 50).astype(int)
    order = rng.permutation(400)
    tiles = [np.flatnonzero(cell_tile == k) for k in order]
    tiles = [tile for tile in tiles if len(tile)]

    # counts: half the genes follow the band, the rest are flat; a type offset
    base = rng.normal(size=n_genes)
    slope = np.zeros(n_genes)
    slope[: n_genes // 2] = 1.5 * rng.choice([-1.0, 1.0], n_genes // 2)
    offset = rng.normal(scale=0.3, size=(len(OLD_NAMES), n_genes))
    logit = base[None] + field[:, None] * slope[None] + offset[t_old]
    rho = np.exp(logit - logit.max(axis=1, keepdims=True))
    rho /= rho.sum(axis=1, keepdims=True)
    depth = rng.poisson(150, n) + 20
    x = sp.csr_matrix(rng.poisson(depth[:, None] * rho).astype(np.float32))
    totals = np.asarray(x.sum(axis=1)).ravel()

    graph = SimpleNamespace(n_cells=n, degrees=np.ones(n, dtype=np.int64))
    n_train = int(0.85 * len(tiles))
    data = SimpleNamespace(graph=graph, positions=pos, x=x, totals=totals,
                           t=t_new, type_names=np.asarray(NEW_NAMES),
                           p_t=np.bincount(t_new) / n,
                           train_tiles=tiles[:n_train], val_tiles=tiles[n_train:])

    d_w = 12
    b_matrix = rng.normal(size=(n_genes, d_w))
    # every gene loads the planted coordinate clearly; the 11 noise
    # dimensions make the genes' orderings on the other axis independent
    b_matrix[:, 0] = rng.choice([-1.0, 1.0], n_genes) * (1 + np.abs(rng.normal(size=n_genes)))
    models = {}
    for name, coord in (("banded", field), ("orthogonal", pos[:, 1] / 1000)):
        prior_w = np.hstack([coord[:, None],
                             0.5 * rng.normal(size=(n, d_w - 1))])
        trainer = FakeTrainer(FakeModel(prior_w, rho), tiles, n_train)
        models[name] = (SimpleNamespace(kappa=0.1, seed=0), data, trainer,
                        None, b_matrix)
    labels = {"old": {"t": t_old, "names": OLD_NAMES,
                      "p_t": np.bincount(t_old) / n, "data": data,
                      "matcher": A.old_is_tumour},
              "new": {"t": t_new, "names": NEW_NAMES, "p_t": data.p_t,
                      "data": data, "matcher": is_tumour}}
    return {"data": data, "models": models, "labels": labels, "lining": lining}


@pytest.fixture(scope="module")
def world():
    w = planted_world()
    w["columns"] = {f"{a}/{b}": A.make_column(w["labels"][a], w["labels"][b])
                    for a, b in A.COLUMNS}
    w["reads"] = {(m, c): A.read_cell(model, col, "synthetic", m)
                  for m, model in w["models"].items()
                  for c, col in w["columns"].items()}
    return w


def test_matchers_split_the_two_vocabularies():
    assert [A.old_is_tumour(n) for n in OLD_NAMES] == [True, True, True, False]
    assert [is_tumour(n) for n in OLD_NAMES] == [True, True, False, False]
    assert [is_tumour(n) for n in NEW_NAMES] == [False, False, True]
    # the old matcher is for the old vocabulary only
    assert not A.old_is_tumour("Tumour")


def test_the_plant_is_read_on_its_own_axis_in_every_column(world):
    for c in world["columns"]:
        banded = world["reads"][("banded", c)]["pooled"]["w_predicted"]
        ortho = world["reads"][("orthogonal", c)]["pooled"]["w_predicted"]
        n = banded["n_gene_panels"]
        tb, to = banded["counts"]["0.9"], ortho["counts"]["0.9"]
        assert banded["mean_abs_tau_true"] > 0.95, c
        assert tb["true_axis"] >= 0.9 * n and tb["false_axis"] <= 0.1 * n, (c, tb)
        assert ortho["mean_abs_tau_false"] > 0.95, c
        assert to["false_axis"] >= 0.9 * n and to["true_axis"] <= 0.1 * n, (c, to)
        # the false axis is the orthogonal coordinate
        assert world["reads"][("banded", c)]["false_axis"]["coordinate"] == "y"


def test_model_free_rows_do_not_move_with_the_model(world):
    for c in world["columns"]:
        a, b = world["reads"][("banded", c)], world["reads"][("orthogonal", c)]
        assert A.same_rows(a, b), c
        assert not A.same_rows(a, b, rows=("w_predicted",)), c
    # and the planted count gradient is on the true axis of the raw read
    raw = world["reads"][("banded", "new/new")]["pooled"]["raw_observed"]["counts"]["0.9"]
    assert raw["true_axis"] > 3 * max(raw["false_axis"], 1)


def test_the_column_changes_the_classes_and_the_band(world):
    reads, cols = world["reads"], world["columns"]
    assert set(reads[("banded", "old/old")]["types"]) == {
        "Tumor Cells", "Proliferative Tumor Cells", "Tumor Associated Fibroblasts"}
    assert set(reads[("banded", "new/new")]["types"]) == {"Tumour", "Fibroblasts"}
    # the type set follows the first key, the band the second
    assert set(reads[("banded", "old/new")]["types"]) == set(reads[("banded", "old/old")]["types"])
    assert set(reads[("banded", "new/old")]["types"]) == set(reads[("banded", "new/new")]["types"])
    assert cols["old/new"]["tumour_classes"] == ["Tumour"]
    assert cols["new/old"]["tumour_classes"] == OLD_NAMES[:3]
    # the lining counts as tumour in the old band only: its field is higher
    # there, never lower, and the difference sits at the lining patch
    old_b, new_b = cols["old/old"]["band"], cols["new/new"]["band"]
    assert (old_b >= new_b).all()
    moved = old_b > new_b
    pos = world["data"].positions
    assert moved.sum() > 0
    assert (pos[moved, 0] < 400).mean() > 0.95 and (pos[moved, 1] > 700).mean() > 0.95


def test_a_protocol_column_is_axis_test_itself(world):
    """The data's own labels and matcher: read_cell == axis_test unpatched
    (only the loader stubbed)."""
    model = world["models"]["banded"]
    with mock.patch.object(E, "load_run", lambda *a, **k: model):
        direct = E.axis_test(argparse.Namespace(dataset="synthetic", run="banded",
                                                device="cpu"))
    assert json.dumps(direct, sort_keys=True) == json.dumps(
        world["reads"][("banded", "new/new")], sort_keys=True)


def test_stored_check_and_leave_one_out(world):
    read = world["reads"][("banded", "old/old")]
    stored = json.loads(json.dumps(read))           # a JSON round trip
    chk = A.against_stored(read, stored)
    assert chk["ok"] and chk["max_abs_diff_tau"] == 0.0
    other = world["reads"][("orthogonal", "old/old")]
    assert not A.against_stored(other, stored)["ok"]
    loo = A.leave_one_out(read)
    assert abs(sum(v["gene_share"] for v in loo.values()) - 1.0) < 1e-12
    pooled = read["pooled"]["w_predicted"]["counts"]["0.9"]
    for g, v in loo.items():
        own = read["types"][g]["rows"]["w_predicted"]["counts"]["0.9"]
        assert v["without"]["tp"] == pooled["true_axis"] - own["true_axis"]


def test_false_axis_alignment_separates_orthogonal_from_aligned(world):
    """y is orthogonal to the planted band: Tumour's y-bins carry no band
    spread; cutting along x (the band's own axis) is fully aligned. The
    relabel itself can bend a class's false axis: the lining patch (high y,
    low x) leaves Fibroblasts in the lineage column, so their top y-bin loses
    low-band cells and only that bin moves."""
    col = world["columns"]["new/new"]
    classes = ["Tumour", "Fibroblasts"]
    ortho = A.false_axis_alignment(world["data"], col, "y", classes)
    along = A.false_axis_alignment(world["data"], col, "x", classes)
    assert ortho["Tumour"]["band_range"] < 0.1
    assert abs(ortho["Tumour"]["spearman"]) < 0.05
    fib = np.array(ortho["Fibroblasts"]["mean_band_per_bin"])
    assert fib[-1] - fib[:-1].max() > 0.2 and np.ptp(fib[:-1]) < 0.1
    for c in classes:
        assert along[c]["tau_bins"] == 1.0 and along[c]["band_range"] > 2.0
        assert along[c]["spearman"] > 0.8


def test_composition_maps_the_vocabularies():
    t_old = np.array([0, 0, 1, 1, 2, 3, 3, 3])
    t_new = OLD_TO_NEW[t_old]
    rows = np.ones(len(t_old), dtype=bool)
    comp = A.composition(t_new, NEW_NAMES, t_old, OLD_NAMES, rows, ["Tumour"])
    assert comp["Tumour"] == {"Tumor Cells": 0.5, "Proliferative Tumor Cells": 0.5}
    back = A.composition(t_old, OLD_NAMES, t_new, NEW_NAMES, rows,
                         ["Malignant Cells Lining Cyst"])
    assert back == {"Malignant Cells Lining Cyst": {"Mesothelial-like cyst lining": 1.0}}
