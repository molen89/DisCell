"""The evaluation mask: Unassigned is a training class and a neighbour, never
a metric target (devlog 2026-09-28, author's decision).

(1) A slide with no Unassigned reads bit-identically with the mask on and off,
    for every read the mask touches -- each read run end to end on a small
    synthetic run on disk (best.pt, config, metrics, history), once as
    shipped and once under ``DISCELL_EVAL_INCLUDE_UNASSIGNED=1``.
(2) A planted slide whose Unassigned cells carry a spurious signal (their z
    scattered over the other types' clusters and loaded with the niche
    block): NMI and the probe move with the mask; cycle does not (the
    trainer's cycling set has always skipped Unassigned, so the cycle reads
    were already masked); and the composition, influx and forward pass the
    model sees are identical either way.
(3) On GSE ``finalL_s0`` (skipped when the run is not on disk): the in-trainer
    battery at the accepted checkpoint and the post-hoc reads that replay it
    agree under the mask.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from discell import paths
from discell.model import eval_mask as EM
from discell.model.prepare import ModelData, spatial_tiles
from discell.model.synthetic import simulate
from discell.model.train import TrainConfig, Trainer

DS, RUN = "evalmask_synthetic", "maskrun_s0"
PLAIN = ["Endothelial cells", "Macrophages", "Tumour", "pDC"]
WITH_U = ["Endothelial cells", "Tumour", "Unassigned", "pDC"]   # U in the middle
U = 2


# -- the helpers ----------------------------------------------------------------------

def test_switch_names_and_compaction(monkeypatch):
    monkeypatch.delenv(EM.INCLUDE_ENV, raising=False)
    names = np.array(WITH_U)
    assert EM.exclusions() == ("Unassigned",)
    assert EM.target_types(names).tolist() == [True, True, False, True]
    assert EM.is_excluded("unassigned") and not EM.is_excluded("Tumour")
    t = np.array([0, 2, 3, 1, 2])
    assert EM.metric_target_mask(t, names).tolist() == [True, False, True,
                                                        True, False]
    assert EM.metric_target_mask(names[t]).tolist() == [True, False, True,
                                                        True, False]
    assert EM.exclude_types([3, 2, 0], names).tolist() == [3, 0]
    # compaction: one index per target type, order kept
    assert EM.compact_types(np.array([0, 1, 3]), names).tolist() == [0, 1, 2]
    z = np.arange(10.0).reshape(5, 2)
    z_n, t_n = EM.nmi_inputs(z, t, names)
    assert t_n.tolist() == [0, 2, 1] and z_n.tolist() == z[[0, 2, 3]].tolist()
    # no excluded type: every helper hands its input back
    plain = np.array(PLAIN)
    assert EM.nmi_inputs(z, t, plain)[0] is z
    assert EM.compact_types(t, plain) is t
    # the switch restores the pre-rule reads
    monkeypatch.setenv(EM.INCLUDE_ENV, "1")
    assert EM.exclusions() == ()
    assert EM.metric_target_mask(t, names).all()
    assert EM.nmi_inputs(z, t, names)[0] is z
    assert EM.same_mask(None) and not EM.same_mask({"excluded_types":
                                                    ["Unassigned"]})
    monkeypatch.delenv(EM.INCLUDE_ENV)
    assert EM.same_mask({"excluded_types": ["unassigned"]})
    assert not EM.same_mask(None)


# -- a synthetic run on disk ------------------------------------------------------------

def _world(names, plant: bool = False, seed: int = 0) -> ModelData:
    sim = simulate(n_cells=4000, n_genes=120, n_types=4, d_phi=16, kappa=0.1,
                   seed=seed)
    t, k = sim.t.astype(np.int64), 4
    rng = np.random.default_rng(seed + 11)
    v = np.hstack([sim.graph.y[:, :-1], sim.phi[:, :4]]).astype(np.float32)
    tiles = spatial_tiles(sim.positions, 120)
    # a random 15 % of the tiles held out, as prepare.assemble draws them
    order = np.random.default_rng([seed, 2]).permutation(len(tiles))
    n_val = int(round(0.15 * len(tiles)))
    s_score = rng.normal(size=len(t)) + 0.5 * (t == 1)
    g2m = rng.normal(size=len(t))
    if plant:
        # the spurious signal lives in the Unassigned cells only
        s_score[t == U] += 3.0 * v[t == U, 0]
    cycle = {"s_score": s_score, "g2m_score": g2m,
             # Unassigned ranked first: the trainer's set must skip it
             "cycling_types": [U, 1, 0, 3], "reliability": None,
             "x_pcs": rng.normal(size=(len(t), 5))}
    return ModelData(
        graph=sim.graph, x=sp.csr_matrix(sim.x), t=t, phi=sim.phi,
        positions=sim.positions, totals=sim.totals,
        median_counts=float(np.median(sim.totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=np.array(names), v_block=v,
        vbar_t=np.stack([v[t == g].mean(axis=0) for g in range(k)]),
        train_tiles=[tiles[i] for i in order[n_val:]],
        val_tiles=[tiles[i] for i in order[:n_val]],
        gene_names=np.array([f"G{i}" for i in range(sim.x.shape[1])]),
        cycle=cycle)


def _config() -> TrainConfig:
    return TrainConfig(
        dataset=DS, run_name=RUN, kappa=0.1, d_z=6, d_w=2, hidden=32,
        gat_dim=8, epochs=1, eval_every=1, figures_every=1, device="cpu",
        alpha_z=0.007, alpha_w=0.1, alpha_a=0.3, v_pcs=4,
        invariance="closed_form", w_warmup_epochs=0, label_key="lineage")


def _plant_latents(monkeypatch, data: ModelData) -> None:
    """Replace the posterior mean of z (every read's source) by a planted
    one: the target types sit in tight clusters; Unassigned is a mixed bag
    across them, loaded with the niche block (a spurious probe signal)."""
    rng = np.random.default_rng(5)
    centres = rng.normal(scale=4.0, size=(4, 6))
    z = centres[data.t] + rng.normal(scale=0.3, size=(len(data.t), 6))
    bag = data.t == U
    z[bag] = (centres[rng.choice([0, 1, 3], bag.sum())]
              + rng.normal(scale=0.3, size=(bag.sum(), 6)))
    z[bag, :4] += 8.0 * (data.v_block[bag, :4] - data.v_block[bag, :4].mean(0))
    planted = z.astype(np.float32)
    real = Trainer._sweep

    def sweep(self, batches, want_log_p=False, z_bar=None):
        out = real(self, batches, want_log_p, z_bar)
        out["mu_z"] = planted[out["nodes"]]
        return out

    monkeypatch.setattr(Trainer, "_sweep", sweep)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch, request):
    """``(data, trainer)``: a random-init DisCell saved as a finished run
    under a throwaway data root, ``assemble`` answering with *data*."""
    names, plant = request.param
    monkeypatch.setattr(paths, "DATASETS", tmp_path)
    data = _world(names, plant)
    if plant:
        _plant_latents(monkeypatch, data)
    from discell.experiments import external_criteria as E
    from discell.model import (atlas, crossslide, degeneracy, prepare, sweep,
                               transport, validate)

    fake = lambda *a, **k: data                                  # noqa: E731
    for module in (prepare, validate, degeneracy, crossslide, sweep):
        monkeypatch.setattr(module, "assemble", fake)
    monkeypatch.setattr(atlas, "read_hallmarks", lambda panel: {})
    monkeypatch.setattr(E, "lr_gene_union", lambda panel: {"G1", "G5", "G9"})
    for module in (transport, E):
        monkeypatch.setattr(module, "MIN_CELLS", 40)
    monkeypatch.setattr(E, "MIN_BAND_CELLS", 5)
    config = _config()
    torch.manual_seed(0)
    trainer = Trainer(config, data)
    run_dir = trainer.run_dir
    # the record a pre-rule fit leaves: history / metrics unmasked
    EM.set_include_unassigned(True)
    try:
        report = trainer.evaluate()
    finally:
        EM.set_include_unassigned(None)
    report.pop("collected")
    torch.save({"model": trainer.model.state_dict(), "covariances": None,
                "config": dataclasses.asdict(config), "epoch": 0},
               run_dir / "best.pt")
    (run_dir / "config.json").write_text(json.dumps(
        dataclasses.asdict(config), default=str))
    best = {k: report[k] for k in ("recon_val", "nmi", "degeneracy",
                                   "recon_gap")}
    (run_dir / "metrics.json").write_text(json.dumps(
        {"best": {**best, "epoch": 0}, "final": report, "final_epoch": 0,
         "dead_w_channel": False}, default=float))
    (run_dir / "history.jsonl").write_text(json.dumps(
        {"epoch": 0, **report}, default=float) + "\n")
    # a baseline's stored latent (any intrinsic latent will do)
    import anndata as ad

    rng = np.random.default_rng(3)
    latents = ad.AnnData(np.zeros((data.n_cells, 1), dtype=np.float32))
    latents.obs_names = [f"cell_{i}" for i in range(data.n_cells)]
    latents.obsm["X_resolvi"] = rng.normal(size=(data.n_cells, 5))
    latents.obsm["X_resolvi_mixture"] = rng.dirichlet(np.ones(3), data.n_cells)
    (tmp_path / "base").mkdir()
    latents.write_h5ad(tmp_path / "base" / "latents.h5ad")
    yield data, trainer
    EM.set_include_unassigned(None)


def _subtype_patch(monkeypatch, data):
    """subtype_recovery's label tables for the synthetic lineages: Tumour
    splits in two by x position, Unassigned (when present) into two."""
    from discell.experiments import subtype_recovery as S

    lineage = np.asarray(data.type_names, dtype=str)[data.t]
    old = lineage.copy()
    left = np.arange(len(lineage)) % 2 == 0      # both sublabels in every tile
    old[(lineage == "Tumour") & left] = "Tumour-A"
    old[(lineage == "Tumour") & ~left] = "Tumour-B"
    old[(lineage == "Unassigned") & left] = "SOX2-OT+ Tumor Cells"
    pairs = set(zip(old.tolist(), lineage.tolist()))
    kinds = {"Tumour-A": "state", "Tumour-B": "location"}
    tables = {"source_key": "old", "obs_key": "lineage", "pairs": pairs,
              "kinds": kinds}
    monkeypatch.setattr(S, "label_tables", lambda ds: tables)
    monkeypatch.setattr(S, "old_labels", lambda *a: old)


def _all_reads(data, trainer, tmp_path, monkeypatch) -> dict:
    """Every read the mask touches, end to end, in queue order."""
    from discell.experiments import (baseline_battery, bootstrap, marker_pairs,
                                     probe_regrade, recon_modes,
                                     subtype_recovery, w_deviation)
    from discell.experiments import external_criteria as E
    from discell.model import atlas, degeneracy, sweep, transport, validate
    from discell.model.crossslide import apply_fit

    root = paths.dataset(DS).root
    run_dir = root / "runs" / RUN
    # every pass starts from the fitted run alone: no read of a previous pass
    # (e.g. a baseline's probe record) may feed this one
    import shutil
    keep = {"best.pt", "config.json", "metrics.json", "history.jsonl"}
    for item in run_dir.iterdir():
        if item.name not in keep:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    shutil.rmtree(root / "experiments", ignore_errors=True)
    for link in (root / "runs").iterdir():
        if link.is_symlink():
            link.unlink()
    out: dict = {}
    trainer.model.eval()
    report = trainer.evaluate()
    report.pop("collected")
    out["evaluate"] = report
    out["crossslide"] = apply_fit(trainer.config, trainer.model, data)
    args = argparse.Namespace(dataset=DS, device="cpu", n_perms=20,
                              analyses="morans,niche,probe,landmarks",
                              headline_k_only=True, max_cells=30_000,
                              force=True)
    out["validate"] = validate.run_analyses(args, RUN)
    out["probe_blocks"] = json.loads((run_dir / "validation"
                                      / "probe_blocks.json").read_text())
    out["kappa_survival_programs"] = validate._program_loadings(
        args, RUN, {})["loadings"]
    assert degeneracy.main(["--dataset", DS, "--run", RUN,
                            "--device", "cpu"]) == 0
    out["degeneracy"] = json.loads((run_dir / "degeneracy.json").read_text())
    assert probe_regrade.main(["--dataset", DS, "--run", RUN, "--force",
                               "--device", "cpu"]) == 0
    out["probe_regrade"] = json.loads((run_dir / "validation"
                                       / "probe_blocks.json").read_text())
    base = str(tmp_path / "base" / "latents.h5ad")
    assert probe_regrade.main(["--dataset", DS, "--run", RUN, "--force",
                               "--baseline-latents", base,
                               "--method", "resolVI"]) == 0
    out["probe_regrade_baseline"] = json.loads(
        (root / "experiments" / "probe_regrade" / "resolVI.json").read_text())
    out["atlas"] = atlas.build_atlas(argparse.Namespace(
        dataset=DS, run=RUN, device="cpu", n_perms=20, compare_runs=[],
        compare_atlas=[]))
    targs = argparse.Namespace(dataset=DS, run=RUN, device="cpu", niches=3,
                               niche_source="kmeans", figures=0, hvg=0,
                               boot=20, target_side="both")
    out["transport_mean"] = transport.transport_check(targs)
    out["transport_dist"] = transport.distribution_check(targs, twins=True)
    out["transport_twins"] = json.loads((run_dir / "transport"
                                         / "transport_twins.json").read_text())
    assert out["transport_mean"]["panels"], "the world must carry panels"
    out["recon_modes"] = recon_modes.recon_modes(DS, RUN, device="cpu",
                                                 n_boot=20)
    out["bootstrap"] = bootstrap.run_bootstrap(DS, RUN, n=20, device="cpu",
                                               boot_replayed=20)
    eargs = argparse.Namespace(dataset=DS, run=RUN, device="cpu", niches=3,
                               perms=2, niche_source="kmeans")
    out["mi_quadrant"] = E.mi_quadrant(eargs)
    share = E.signalling_share(eargs)
    out["signalling_share"] = {k: v for k, v in share.items()
                               if not k.startswith("_")}
    out["axis_test"] = E.axis_test(eargs)
    assert baseline_battery.main(["--dataset", DS, "--discell-run", RUN,
                                  "--config-run", RUN, "--device", "cpu"]) == 0
    assert baseline_battery.main(["--dataset", DS, "--method", "resolVI",
                                  "--latents", base, "--config-run", RUN]) == 0
    out["baseline_battery"] = json.loads((root / "experiments"
                                          / "baseline_battery.json").read_text())
    _subtype_patch(monkeypatch, data)
    out["subtype_recovery"] = subtype_recovery.grade_run(DS, RUN, "cpu",
                                                         n_boot=20, n_perm=2)
    pairs = [{"gene_a": "G0", "gene_b": "G1", "kind": "exclusive"},
             {"gene_a": "G2", "gene_b": "G3", "kind": "control"}]
    out["marker_pairs"] = marker_pairs.score_run(DS, RUN, pairs, "test",
                                                 device="cpu", n_boot=20)
    trainer.model.eval()
    out["w_deviation"] = w_deviation.deviation_read(trainer)
    link = root / "runs" / sweep.run_name(0.1, 0, "sweepT", "kappa")
    if not link.exists():
        link.symlink_to(RUN)
    out["sweep_report"] = sweep.report(sweep.parse_args(
        ["--dataset", DS, "--report-only", "--tag", "sweepT", "--param",
         "kappa", "--values", "0.1", "--seeds", "0"]))
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import envelope_tables
    out["envelope"] = envelope_tables.run_record(DS, RUN, "best")
    return out


#: keys that say how or when a read was made, not what it read
VOLATILE = {"eval_mask", "minutes", "source", "nmi_selection",
            "recon_val_selection", "recon_selection", "latents"}


def _canonical(obj):
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()
                if str(k) not in VOLATILE}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return "nan" if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _diff(a, b, where="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{where}/{k}: only on one side")
            else:
                out += _diff(a[k], b[k], f"{where}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} vs {len(b)} entries"]
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in _diff(x, y, f"{where}[{i}]")]
    return [] if a == b else [f"{where}: {a!r} vs {b!r}"]


READS = ("evaluate", "crossslide", "validate", "probe_blocks",
         "kappa_survival_programs", "degeneracy", "probe_regrade",
         "probe_regrade_baseline", "atlas", "transport_mean",
         "transport_dist", "transport_twins", "recon_modes", "bootstrap",
         "mi_quadrant", "signalling_share", "axis_test", "baseline_battery",
         "subtype_recovery", "marker_pairs", "w_deviation", "sweep_report",
         "envelope")


@pytest.mark.parametrize("synthetic_run", [(PLAIN, False)], indirect=True)
def test_a_slide_without_unassigned_reads_identically_with_the_mask_on_and_off(
        synthetic_run, tmp_path, monkeypatch):
    data, trainer = synthetic_run
    monkeypatch.delenv(EM.INCLUDE_ENV, raising=False)
    on = _canonical(_all_reads(data, trainer, tmp_path, monkeypatch))
    monkeypatch.setenv(EM.INCLUDE_ENV, "1")
    off = _canonical(_all_reads(data, trainer, tmp_path, monkeypatch))
    assert set(on) == set(off) == set(READS)
    problems = {read: _diff(on[read], off[read])[:5] for read in READS}
    problems = {k: v for k, v in problems.items() if v}
    assert not problems, problems
    # none of the comparisons is vacuous
    assert on["signalling_share"]["n_panels"] > 0
    assert on["axis_test"]["types"]
    assert any(g["graded"] for g in on["subtype_recovery"]["groups"]), (
        on["subtype_recovery"]["groups"])
    assert on["transport_dist"]["pairwise"] and on["atlas"]["programs"]
    assert on["mi_quadrant"]["sources"]["kmeans"]["n_cells"] > 0
    assert on["evaluate"]["nmi_targets"] == on["evaluate"]["nmi"]


@pytest.mark.parametrize("synthetic_run", [(WITH_U, True)], indirect=True)
def test_planted_unassigned_signal_moves_the_reads_and_not_the_model_inputs(
        synthetic_run, monkeypatch):
    data, trainer = synthetic_run
    y0, v0 = data.graph.y.copy(), data.v_block.copy()
    in_edges0 = data.graph.in_edges.copy()
    batch = trainer.val_batches[0]

    def forward():
        with torch.no_grad():
            fwd = trainer.model(**trainer._forward_kwargs(batch),
                                kappa=trainer.config.kappa, sample=False)
        return {k: getattr(fwd, k).numpy().copy()
                for k in ("rho_bar", "c", "log_p", "mu_w", "prior_mean_w")}

    trainer.model.eval()
    monkeypatch.delenv(EM.INCLUDE_ENV, raising=False)
    masked, fwd_masked = trainer.evaluate(), forward()
    monkeypatch.setenv(EM.INCLUDE_ENV, "1")
    unmasked, fwd_unmasked = trainer.evaluate(), forward()

    # the mixed bag drags NMI down and its niche load lifts the probe; the
    # mask takes both away
    assert masked["nmi_targets"] > unmasked["nmi_targets"] + 0.05
    assert masked["nmi"] == unmasked["nmi"]          # the selection score
    assert (unmasked["probe"]["delta_ce"]
            > masked["probe"]["delta_ce"] + 1e-3)
    assert masked["mirror"] != unmasked["mirror"]
    assert masked["degeneracy"] != unmasked["degeneracy"]
    assert masked["recon_val"] == unmasked["recon_val"]   # selection score
    assert masked["recon_val_targets"] != unmasked["recon_val_targets"]
    # cycle: Unassigned (ranked first) was never in the cycling set, so the
    # read is the same either way -- and its planted signal never enters
    assert U not in masked["cycle"]["types"]
    assert masked["cycle"] == unmasked["cycle"]
    # what the model sees does not move: composition, influx, forward pass
    assert np.array_equal(data.graph.y, y0) and np.array_equal(data.v_block, v0)
    assert (data.graph.in_edges != in_edges0).nnz == 0
    for key in fwd_masked:
        assert np.array_equal(fwd_masked[key], fwd_unmasked[key]), key
    # the selection pair of the fit loop is untouched by the mask
    assert masked["eval_mask"]["n_cells_excluded"] == int((data.t == U).sum()
                                                          ) > 0
    assert unmasked["eval_mask"]["excluded_types"] == []

    # the baseline battery: the same mask for every method's column
    from discell.experiments import baseline_battery as B

    columns = {}
    for flag in ("", "1"):
        monkeypatch.setenv(EM.INCLUDE_ENV, flag)
        assert B.main(["--dataset", DS, "--discell-run", RUN, "--config-run",
                       RUN, "--device", "cpu"]) == 0
        columns[flag] = json.loads((paths.dataset(DS).root / "experiments"
                                    / "baseline_battery.json").read_text())[
            f"DisCell/{RUN}"]
    assert columns[""]["nmi"] == pytest.approx(masked["nmi_targets"])
    assert columns["1"]["nmi"] == pytest.approx(unmasked["nmi_targets"])
    assert columns[""]["n_cells"] == data.n_cells - int((data.t == U).sum())
    assert columns[""]["probe"] != columns["1"]["probe"]


@pytest.mark.parametrize("synthetic_run", [(WITH_U, True)], indirect=True)
def test_transport_and_atlas_drop_unassigned_and_keep_the_other_panels(
        synthetic_run, tmp_path, monkeypatch):
    """The excluded type is never the moved type, and every other panel is
    bit-identical to the unmasked read (its random draws are preserved)."""
    from discell.model import atlas, transport

    data, trainer = synthetic_run
    targs = argparse.Namespace(dataset=DS, run=RUN, device="cpu", niches=3,
                               niche_source="kmeans", figures=0, hvg=0,
                               boot=20, target_side="both")
    reads = {}
    for flag in ("", "1"):
        monkeypatch.setenv(EM.INCLUDE_ENV, flag)
        reads[flag] = (_canonical(transport.transport_check(targs)),
                       _canonical(transport.distribution_check(targs)),
                       _canonical(atlas.build_atlas(argparse.Namespace(
                           dataset=DS, run=RUN, device="cpu", n_perms=20,
                           compare_runs=[], compare_atlas=[]))))
    (mean_m, dist_m, atlas_m), (mean_u, dist_u, atlas_u) = reads[""], reads["1"]
    assert all(p["type"] != "Unassigned" for p in mean_m["panels"])
    dropped = [p for p in mean_u["panels"] if p["type"] == "Unassigned"]
    assert dropped, "the world must give Unassigned panels"
    kept = [p for p in mean_u["panels"] if p["type"] != "Unassigned"]
    assert _diff(mean_m["panels"], kept) == []
    for version in ("pairwise", "leave_one_out"):
        kept = [p for p in dist_u[version] if p["type"] != "Unassigned"]
        assert _diff(dist_m[version], kept) == []
    assert "Unassigned" not in atlas_m["programs"][0]["type_activity"]
    assert "Unassigned" in atlas_u["programs"][0]["type_activity"]
    # subtype recovery: the Unassigned "lineage" (two old labels) is no group
    from discell.experiments import subtype_recovery

    _subtype_patch(monkeypatch, data)
    groups = {}
    for flag in ("", "1"):
        monkeypatch.setenv(EM.INCLUDE_ENV, flag)
        rec = subtype_recovery.grade_run(DS, RUN, "cpu", n_boot=20, n_perm=2)
        groups[flag] = {g["name"]: g for g in rec["groups"]}
    assert "Unassigned" in groups["1"] and "Unassigned" not in groups[""]
    assert _diff(_canonical(groups[""]["Tumour"]),
                 _canonical(groups["1"]["Tumour"])) == []


# -- (3) the in-trainer battery and the post-hoc battery on a real run -------------

GSE = "gse315411_pdltma06_11_prime_solo"
REAL = Path(__file__).resolve().parents[1] / "data" / "datasets"


@pytest.mark.skipif(not (Path(os.environ.get("DISCELL_DATA", REAL.parent))
                         / "datasets" / GSE / "runs" / "finalL_s0"
                         / "best.pt").exists(),
                    reason="GSE finalL_s0 is not on disk")
def test_in_trainer_and_post_hoc_batteries_agree_on_gse_finalL_s0(monkeypatch):
    from discell.experiments import bootstrap as B
    from discell.experiments import recon_modes as R
    from discell.model import degeneracy
    from discell.model import metrics as M
    from discell.model import validate

    monkeypatch.delenv(EM.INCLUDE_ENV, raising=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, data, trainer, _, _ = validate.load_run(GSE, "finalL_s0", device)
    names = data.type_names
    assert EM.excluded_type_ids(names).size == 1
    report = trainer.evaluate()                      # the in-trainer battery
    assert report["eval_mask"]["n_cells_excluded"] > 0
    seed = config.seed

    # the probe: validate's per-block record replays the in-trainer dCE
    latents = validate.collect_latents(trainer, data)
    blocks = validate.probe_blocks_for_run(data, latents["mu_z"], seed)
    legacy = blocks["ridge"]["legacy"]
    for key in ("delta_ce", "noise_floor", "baseline_ce"):
        assert legacy[key] == pytest.approx(report["probe"][key], rel=1e-5,
                                            abs=1e-7), key

    # NMI, cycle: the bootstrap's unit-weight statistics are the reads
    train = trainer._sweep(trainer.train_batches)
    val = trainer._sweep(trainer.val_batches, want_log_p=True)
    z_all = np.vstack([train["mu_z"], val["mu_z"]])
    w_all = np.vstack([train["mu_w"], val["mu_w"]])
    rows_all = np.concatenate([train["nodes"], val["nodes"]])
    t_all = data.t[rows_all]
    cells = B.nmi_cells(*EM.nmi_inputs(z_all, t_all, names), seed)
    assert B.nmi_statistic(cells, np.ones(len(cells["t"]))) == pytest.approx(
        report["nmi_targets"], abs=1e-10)
    train_mask = np.arange(len(rows_all)) < len(train["nodes"])
    scores = np.stack([data.cycle["s_score"], data.cycle["g2m_score"]],
                      axis=1)[rows_all]
    types = np.asarray(report["cycle"]["types"])
    for name, latent in (("z", z_all), ("w", w_all)):
        cyc = B.cycle_cells(latent, t_all, scores, types, train_mask,
                            ~train_mask, seed)
        assert B.cycle_statistic(cyc, np.ones(len(cyc["target"]))) == \
            pytest.approx(report["cycle"][name]["r2_pooled"], abs=1e-8)

    # the top-decile cycle rows (their bootstrap replay = the in-trainer read)
    from discell.model.cell_cycle import slide_cycling_set

    q90 = slide_cycling_set(data)[rows_all]
    assert not q90[~EM.metric_target_mask(t_all, names)].any()
    for name, latent, key in (("z", z_all, "cycle_r2_z_q90"),
                              ("w", w_all, "cycle_r2_w_q90")):
        cyc = B.cycle_cells(latent, t_all, scores, None, train_mask,
                            ~train_mask, seed, cells=q90)
        assert B.cycle_statistic(cyc, np.ones(len(cyc["target"]))) == \
            pytest.approx(report[key], abs=1e-8)

    # recon: the decodes of recon_modes on the target held-out cells are the
    # in-trainer recon_val_targets and recon_gap
    z_bar = M.type_means(train["mu_z"], data.t[train["nodes"]], len(data.p_t))
    profile = R.type_log_profile(data.x[train["nodes"]], data.t[train["nodes"]],
                                 len(data.p_t))
    modes = R.per_cell_modes(trainer, trainer.val_batches, z_bar, profile)
    on = EM.metric_target_mask(data.t[modes["nodes"]], names)
    gap = report["recon_gap"]
    assert modes["full"][on].mean() == pytest.approx(
        report["recon_val_targets"], abs=1e-5)
    assert modes["full"].mean() == pytest.approx(report["recon_val"], abs=1e-5)
    assert modes["typemean_z"][on].mean() == pytest.approx(
        gap["recon_typemean_z"], abs=1e-5)
    assert modes["type_profile"][on].mean() == pytest.approx(
        gap["recon_type_profile"], abs=1e-5)

    # the w-channel guard: the bootstrap's per-cell terms are its excess
    from discell.model.validate import niche_labels

    guard = degeneracy.w_channel_guard_from_trainer(trainer)
    niche = niche_labels(data, degeneracy.W_GUARD_NICHES, seed)
    keep = EM.metric_target_mask(data.t[val["nodes"]], names)
    vrows = val["nodes"][keep]
    wcells = B.w_mi_cells(val["mu_w"][keep], niche[vrows], data.t[vrows], seed)
    assert B.w_mi_statistic(wcells, np.ones(len(wcells["rows"]))) == \
        pytest.approx(guard["w_niche_mi_excess"], abs=1e-8)
    # the selection pair is the one the checkpoint was chosen on
    metrics = json.loads((REAL / GSE / "runs" / "finalL_s0" / "metrics.json")
                         .read_text()) if (REAL / GSE).exists() else None
    if metrics is not None:
        assert report["recon_val"] == pytest.approx(
            metrics["best"]["recon_val"], abs=1e-3)
        assert report["nmi"] == pytest.approx(metrics["best"]["nmi"], abs=5e-3)
