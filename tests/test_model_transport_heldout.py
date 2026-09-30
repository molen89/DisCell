"""Transport scored on the model's held-out tiles only (devlog 2026-09-29,
"Transport scored on held-out tiles only"; ``--scored-cells heldout-tiles``).

(1) The default is the published read: no attribute, ``fold0`` and the CLI
    default give the same files at the same paths, and nothing is written
    under ``heldout_tiles/``.
(2) Under the variant every scored cell (observed shifts, ceilings, source
    and target clouds, the own-target decode) is in ``val_tiles``, every cell
    behind a model quantity (group means, the HVG set) is in ``train_tiles``,
    the Unassigned panels still leave the record, the files land under
    ``runs/<run>/heldout_tiles/`` and the published files are untouched --
    for transport.py, the bootstrap replay and the Cellina adapter.
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pytest
import scipy.sparse as sp
import test_model_eval_mask as TE

from discell.model import eval_mask as EM

DS, RUN, WITH_U = TE.DS, TE.RUN, TE.WITH_U
synthetic_run = TE.synthetic_run      # the eval-mask tests' synthetic run


def _masks(data):
    n = data.graph.n_cells
    val = np.zeros(n, dtype=bool)
    train = np.zeros(n, dtype=bool)
    val[np.concatenate(data.val_tiles)] = True
    train[np.concatenate(data.train_tiles)] = True
    assert not (val & train).any() and (val | train).all()
    return val, train


def _targs(**extra):
    return argparse.Namespace(dataset=DS, run=RUN, device="cpu", niches=3,
                              niche_source="kmeans", figures=0, hvg=50,
                              boot=20, target_side="both", **extra)


def _hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(
        p.read_bytes()).hexdigest() for p in sorted(root.rglob("*"))
        if p.is_file()}


class _Spy:
    """Records the cells every model-side and scored-side call receives."""

    def __init__(self, monkeypatch, module, bootstrap=None):
        self.model, self.scored = [], []
        T = module
        real = {k: getattr(T, k) for k in ("collect_channels", "noise_ceiling",
                                            "_cell_eta", "collect_own_p",
                                            "hvg_mask")}

        def channels(trainer, group, n_groups, phi_by_type=None):
            self.model.append(np.flatnonzero(np.asarray(group) >= 0))
            return real["collect_channels"](trainer, group, n_groups,
                                            phi_by_type)

        def ceiling(x_rate, rows_a, rows_b, *a, **k):
            self.scored += [np.asarray(rows_a), np.asarray(rows_b)]
            return real["noise_ceiling"](x_rate, rows_a, rows_b, *a, **k)

        def eta(trainer, rows):
            self.scored.append(np.asarray(rows))
            return real["_cell_eta"](trainer, rows)

        def own_p(trainer, wanted, n_cells):
            self.scored.append(np.asarray(wanted))
            return real["collect_own_p"](trainer, wanted, n_cells)

        def hvg(data, rows, *a, **k):
            self.model.append(np.asarray(rows))
            return real["hvg_mask"](data, rows, *a, **k)

        for name, fn in (("collect_channels", channels),
                         ("noise_ceiling", ceiling), ("_cell_eta", eta),
                         ("collect_own_p", own_p), ("hvg_mask", hvg)):
            monkeypatch.setattr(T, name, fn)

    def check(self, val, train):
        assert self.model and self.scored
        model = np.unique(np.concatenate(self.model))
        scored = np.unique(np.concatenate(self.scored))
        assert train[model].all(), "a model quantity used a held-out cell"
        assert val[scored].all(), "a scored cell lies in a training tile"
        return len(model), len(scored)


def test_split_cells_default_is_fold0_and_variant_is_the_tiles():
    from discell.model import transport as T

    class _D:
        pass

    data = _D()
    data.graph = _D()
    data.graph.n_cells = 10
    data.train_tiles = [np.array([0, 1, 2]), np.array([5, 6, 7, 8])]
    data.val_tiles = [np.array([3, 4]), np.array([9])]
    fold = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
    scored, model = T.split_cells(data, fold)
    assert np.array_equal(scored, fold == 0)
    assert np.array_equal(model, fold != 0)
    scored, model = T.split_cells(data, fold, "heldout-tiles")
    assert np.flatnonzero(scored).tolist() == [3, 4, 9]
    assert np.flatnonzero(model).tolist() == [0, 1, 2, 5, 6, 7, 8]
    with pytest.raises(ValueError):
        T.split_cells(data, fold, "tiles")
    from pathlib import Path
    assert T.out_root(Path("r")) == Path("r")        # the published root
    assert T.out_root(Path("r"), "heldout-tiles") == Path("r/heldout_tiles")


@pytest.fixture
def one_thread():
    """The reads are float32 torch on CPU; multithreaded reductions move the
    last bits between two calls of the same code, so a byte comparison runs
    on one thread with deterministic kernels."""
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("synthetic_run", [(WITH_U, False)], indirect=True)
def test_default_is_the_published_read(synthetic_run, one_thread):
    from discell import paths
    from discell.model import transport as T

    run_dir = paths.dataset(DS).root / "runs" / RUN
    outputs = {}
    for how in ("absent", "fold0", "cli"):
        if how == "cli":
            assert T.main(["--dataset", DS, "--run", RUN, "--device", "cpu",
                           "--niches", "3", "--figures", "0", "--hvg", "50",
                           "--boot", "20", "--read", "both"]) == 0
        else:
            args = (_targs() if how == "absent"
                    else _targs(scored_cells="fold0"))
            T.transport_check(args)
            T.distribution_check(args, twins=True)
        outputs[how] = _hashes(run_dir / "transport")
        assert not (run_dir / "heldout_tiles").exists()
    assert outputs["absent"] == outputs["fold0"] == outputs["cli"]
    assert {"transport.json", "transport_distribution.json",
            "transport_twins.json"} <= set(outputs["absent"])
    assert "scored_cells" not in (run_dir / "transport"
                                  / "transport.json").read_text()


@pytest.mark.parametrize("synthetic_run", [(WITH_U, False)], indirect=True)
def test_heldout_tiles_scores_val_tiles_and_fits_on_train_tiles(
        synthetic_run, monkeypatch):
    import json

    from discell import paths
    from discell.experiments import bootstrap as BS
    from discell.model import transport as T

    data, _ = synthetic_run
    val, train = _masks(data)
    run_dir = paths.dataset(DS).root / "runs" / RUN
    T.transport_check(_targs())                    # the published read first
    T.distribution_check(_targs(), twins=True)
    published = _hashes(run_dir / "transport")

    spy = _Spy(monkeypatch, T)
    args = _targs(scored_cells="heldout-tiles")
    mean = T.transport_check(args)
    dist = T.distribution_check(args, twins=True)
    n_model, n_scored = spy.check(val, train)
    assert n_scored > 0 and n_model > n_scored

    # the Unassigned mask still applies: its panels drawn, then dropped
    assert mean["panels"] and all(p["type"] != "Unassigned"
                                  for p in mean["panels"])
    assert all(p["type"] != "Unassigned" for v in ("pairwise", "leave_one_out")
               for p in dist[v])
    assert mean["eval_mask"]["excluded_types"] == ["Unassigned"]
    assert mean["eval_mask"]["n_panels_dropped"] >= 1
    rec = mean["scored_cells"]
    assert rec["scored_cells"] == "heldout-tiles"
    assert rec["scored_in_val_tiles"] == 1.0 and rec["model_in_val_tiles"] == 0.0
    assert rec["n_scored_pool"] == int(val.sum())

    # its own root; the published files are byte-for-byte what they were
    variant = run_dir / "heldout_tiles" / "transport"
    assert {"transport.json", "transport_distribution.json",
            "transport_twins.json"} <= set(_hashes(variant))
    assert _hashes(run_dir / "transport") == published
    twins = json.loads((variant / "transport_twins.json").read_text())
    assert twins["scored_cells"]["scored_cells"] == "heldout-tiles"

    # the half-tile CIs replay the variant: same split, own file, same panels
    boot_panels = []
    real_boot = BS.transport_mean_bootstrap

    def boot(panels, *a, **k):
        boot_panels.extend(panels)
        return real_boot(panels, *a, **k)

    monkeypatch.setattr(BS, "transport_mean_bootstrap", boot)
    # the replay's niches are the read's (K = 10 in production, 3 here)
    from discell.model import validate
    real_niches = validate.niche_labels
    monkeypatch.setattr(validate, "niche_labels",
                        lambda data, k, seed: real_niches(data, 3, seed))
    spy_b = _Spy(monkeypatch, T)
    before = (run_dir / "bootstrap_ci.json").exists()
    record = BS.run_bootstrap(DS, RUN, reads=("transport_mean", "transport_dist"),
                              n=20, device="cpu", boot_replayed=20,
                              scored_cells="heldout-tiles")
    spy_b.check(val, train)
    assert (run_dir / "bootstrap_ci.json").exists() == before
    assert (run_dir / "heldout_tiles" / "bootstrap_ci.json").exists()
    assert record["scored_cells"] == "heldout-tiles"
    rows = np.concatenate([np.concatenate([p["rows_a"], p["rows_b"]])
                           for p in boot_panels])
    assert val[rows].all()
    assert all(p["type"] != "Unassigned" for p in boot_panels), (
        [p["type"] for p in boot_panels])
    # the replay reproduces the variant's own point estimates
    assert record["reads"]["transport_of_ceiling"]["reproduces"] is True
    assert record["reads"]["transport_of_ceiling"]["panel_replay"][
        "max_abs_diff"] < 1e-6
    assert record["reads"]["readA_gap_own"]["reproduces"] is True
    with pytest.raises(ValueError):
        BS.run_bootstrap(DS, RUN, reads=("nmi",), n=5, device="cpu",
                         scored_cells="heldout-tiles")


@pytest.mark.parametrize("synthetic_run", [(WITH_U, False)], indirect=True)
def test_unassigned_panels_exist_under_the_switch_in_the_variant(
        synthetic_run, monkeypatch):
    """The world gives the variant Unassigned panels, so the drop tested
    above is a real drop."""
    from discell.model import transport as T

    monkeypatch.setenv(EM.INCLUDE_ENV, "1")
    mean = T.transport_check(_targs(scored_cells="heldout-tiles"))
    assert any(p["type"] == "Unassigned" for p in mean["panels"])


def test_sweep_mode_refuses_the_variant():
    from discell.model import transport as T

    with pytest.raises(SystemExit):
        T.main(["--dataset", DS, "--sweep-tag", "x",
                "--scored-cells", "heldout-tiles"])


def test_cellina_slide_splits_like_transport():
    from discell.experiments import cellina_counterfactual as CF

    class _D:
        pass

    rng = np.random.default_rng(0)
    n = 400
    data = _D()
    data.graph = _D()
    data.graph.n_cells = n
    data.graph.degrees = np.ones(n)
    data.graph.y = rng.random((n, 3))
    order = rng.permutation(n)
    tiles = np.array_split(order, 20)
    data.train_tiles, data.val_tiles = tiles[:17], tiles[17:]
    data.t = rng.integers(0, 2, n)
    data.x = sp.csr_matrix(rng.poisson(2.0, (n, 5)).astype(np.float32))
    data.totals = np.asarray(data.x.sum(1)).ravel()
    data.positions = rng.random((n, 2))
    data.type_names = np.array(["A", "Unassigned"])
    fold = np.zeros(n, dtype=np.int64)
    for k, tile in enumerate(data.train_tiles + data.val_tiles):
        fold[tile] = k % 5
    labels = rng.integers(0, 3, n)
    val, train = _masks(data)
    plain = CF.slide(data, labels, fold)
    assert np.array_equal(plain["held_out"], fold == 0)
    assert np.array_equal(plain["model"], fold != 0)
    sl = CF.slide(data, labels, fold, scored_cells="heldout-tiles")
    for k in range(3):
        for g in range(2):
            tr, te = CF._members(sl, k, g)
            assert train[tr].all() and val[te].all()
            assert len(tr) + len(te) == int(((labels == k) & (data.t == g)).sum())
            # without a model mask: the published complement, as before
            tr0, te0 = CF._members({k2: v for k2, v in plain.items()
                                    if k2 != "model"}, k, g)
            assert np.array_equal(tr0, CF._members(plain, k, g)[0])
            assert np.array_equal(te0, CF._members(plain, k, g)[1])
