"""Reproduce one section of the paper: ingest, embed, fits, sweep, breakdown, compare.

::

    python reproduce/run.py --section lung_ffpe --data-root data/ --work work/lung \\
        --lr-genes work/lr_genes.txt [--quick] [--device cuda:0]

Stages (``--stages``, default all, in this order; each skips what is already on
disk, so an interrupted run resumes):

``ingest``     Xenium outs -> ``<work>/section.h5ad`` (lineage labels, graph)
``embed``      the ego-masked KRONOS2 descriptor (the ``[image]`` extra, your own access;
               the paper's numbers used KRONOS v1, see the README)
``fits``       the final fits (kappa = 0.1, seeds 0-2) and the rest of the kappa grid,
               ``<work>/fits/k<kappa>_s<seed>``
``draws``      the findings' tile draws of every fit, ``<fit>/draws/*.npz``
``breakdown``  kappa* of every member, ``<work>/breakdown.json``
``compare``    against ``reproduce/expected`` with the paper's tolerances,
               ``<work>/report.json`` and ``report.md``

``serial``     (TMA core only; not in the default stages) the core's final and grid fits
               applied unchanged to the serial section (GSE315411 slide 10, donor
               PDL018D): probe, cross-section reads and the findings' draws on
               its cells, its breakdown, compared with ``expected/tma_serial`` and
               ``breakdown_all.json["gse_dual"]``; in ``<work>/serial/``. Needs the core's
               fits in ``<work>/fits``, ``--serial-cell-groups`` and ``--serial-label-dir``.

``reads``      (not in the default stages) per final fit, ``<fit>/reads.json``: the
               probe, NMI of mu_z and mu_w with the type, cycle R^2 of mu_z and mu_w
               on the top-decile cycling set and the closing evaluation's reads

``--variant NAME``: one of the config-only refits listed under ``variants`` in
``configs/<section>.json``. ``no_adversary`` is the final configuration without the
adversary (Fig. 2 and Table S4), fitted at kappa = 0.1 and seeds 0-2 into
``<work>/variants/no_adversary/fits`` (stages ``fits,reads``; the section's
``section.h5ad`` must exist). ``timing`` times the final configuration (seed 0, Table
S3): N training epochs without evaluation (s per epoch excluding the first), one
evaluation, peak GPU memory, into ``<work>/variants/timing/timing.json`` (stage
``timing``).

``--quick``: one fit (kappa = 0.1, seed 0), 200 draws, no breakdown. ``--part i/n``
runs every n-th grid point of the ``fits`` and ``draws`` stages from i, so n
processes (one per GPU) share a sweep; run ``breakdown,compare`` after all of
them. The TMA core's labels are not public: pass ``--cell-groups`` and
``--label-dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lineage import lineage  # noqa: E402

log = logging.getLogger("reproduce")
STAGES = ("ingest", "embed", "fits", "draws", "breakdown", "compare")
#: the paper's image encoder (deprecated, closed to new access, not supported)
PAPER_ENCODER = "kronos-v1"
#: run only when asked for (``--stages serial``)
EXTRA_STAGES = ("serial", "reads", "timing")
QUICK_DRAWS = 200
#: refit drift of the TMA-core control (seed0 refitted), the floor of the
#: tolerance on every section; NMI: the largest same-seed spread over repeated
#: TMA-core refits (seed 1, 3 runs, 2026-10-06), above the control's 0.0115;
#: neighborhood information: 1 point of the share 1 - exp(-2 excess)
FLOORS = {"recon_val": 0.0026025772094726562, "nmi": 0.017464319134451123,
          "var_fraction": 0.01}
PROBE_BLOCKS = ("ridge_comp", "ridge_img", "mlp_comp", "mlp_img")


def load_section(name: str) -> dict:
    return json.loads((HERE / "configs" / f"{name}.json").read_text())


def grid_points(spec: dict, quick: bool) -> list[tuple[float, int]]:
    """The final fits first, then the rest of the kappa grid."""
    final = float(spec["final_kappa"])
    if quick:
        return [(final, int(spec["seeds"][0]))]
    points = [(final, int(s)) for s in spec["seeds"]]
    return points + [(float(k), int(s)) for k in spec["kappas"] if float(k) != final
                     for s in spec["seeds"]]


def fit_dir(work: Path, kappa: float, seed: int) -> Path:
    from discell.sensitivity.sweep import run_dir

    return run_dir(work / "fits", kappa, seed)


def write_atomic(adata, path: Path) -> None:
    tmp = path.with_suffix(".tmp.h5ad")
    adata.write_h5ad(tmp)
    tmp.replace(path)


# -- stages ----------------------------------------------------------------------

def stage_ingest(spec, args, path: Path) -> None:
    import pandas as pd

    import discell as dc

    if path.exists():
        log.info("ingest: %s exists", path)
        return
    outs = Path(args.outs) if args.outs else Path(args.data_root) / spec["outs"]
    lab = spec["labels"]
    cell_groups = args.cell_groups or lab["cell_groups"]
    label_map = (Path(args.label_dir) / "lineage_map_proposed.csv" if args.label_dir
                 else (HERE / lab["map"] if lab["map"] else None))
    reassign = (sorted(Path(args.label_dir).glob("*_reassignment_proposed.csv"))
                if args.label_dir else [])
    if label_map is None or (spec["donors"] and cell_groups is None):
        raise SystemExit(f"{spec['section']}: {lab.get('note', 'labels missing')}")
    t0 = time.time()
    adata = dc.read_xenium(outs, labels=cell_groups if cell_groups else None,
                           donors=spec["donors"])
    source = pd.Series(adata.obs[lab["source_key"]].astype(object).to_numpy(),
                       index=adata.obs_names)
    adata.obs["lineage"] = pd.Categorical(
        lineage(source, label_map, choices=lab["choices"], reassignments=reassign).to_numpy())
    if adata.n_obs != spec["n_cells"]:
        log.warning("ingest: %d cells, the paper's section has %d", adata.n_obs, spec["n_cells"])
    dc.build_graph(adata, "lineage")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(adata, path)
    log.info("ingest: %d cells, %d types in %.1f min -> %s", adata.n_obs,
             adata.obs["lineage"].nunique(), (time.time() - t0) / 60, path)


def stage_embed(spec, args, path: Path) -> None:
    import anndata as ad

    from discell.image import embed

    key = spec["embed"]["key"]
    adata = ad.read_h5ad(path)
    if key in adata.obsm:
        log.info("embed: %s present", key)
        return
    t0 = time.time()
    embed(adata, encoder=spec["embed"]["encoder"], key=key,
          large_cells=spec["embed"]["large_cells"], device=args.device)
    write_atomic(adata, path)
    log.info("embed: %s %s in %.1f min", key, adata.obsm[key].shape, (time.time() - t0) / 60)


def variant_of(spec, args) -> dict | None:
    """The ``variants`` entry named by ``--variant`` (None without it)."""
    name = getattr(args, "variant", None)
    if name is None:
        return None
    variants = spec.get("variants", {})
    if name not in variants:
        raise SystemExit(f"{spec['section']}: no variant {name!r} "
                         f"(configs/{spec['section']}.json lists {sorted(variants)})")
    return variants[name]


def config_of(spec, args):
    import discell as dc

    record = dict(spec["config"])
    if args.no_image:
        record["image_key"] = None
    variant = variant_of(spec, args)
    if variant is not None:
        record.update(variant.get("config", {}))
    return dc.Config(**record, device=args.device)


def fits_root(args) -> Path:
    """``<work>/fits``, or ``<work>/variants/<NAME>/fits`` with ``--variant``."""
    name = getattr(args, "variant", None)
    return args.work / "fits" if name is None else args.work / "variants" / name / "fits"


def my_points(spec, args) -> list[tuple[float, int]]:
    variant = variant_of(spec, args)
    if variant is not None:
        seeds = variant["seeds"][:1] if args.quick else variant["seeds"]
        points = [(float(spec["final_kappa"]), int(s)) for s in seeds]
    else:
        points = grid_points(spec, args.quick)
    i, n = (int(v) for v in args.part.split("/"))
    return [p for k, p in enumerate(points) if k % n == i]


def stage_fits(spec, args, adata) -> None:
    from discell.sensitivity.sweep import fit_sweep

    config = config_of(spec, args)
    for kappa, seed in my_points(spec, args):
        t0 = time.time()
        fit_sweep(adata, config, [kappa], [seed], out_dir=fits_root(args))
        log.info("fit kappa %g seed %d: %.1f min", kappa, seed, (time.time() - t0) / 60)


def fit_reads(model, adata) -> dict:
    """The per-fit reads of a variant (``reads`` stage)."""
    from discell.evaluate.cycle import cell_cycle_r2
    from discell.evaluate.mask import nmi_inputs
    from discell.evaluate.metrics import z_type_nmi
    from discell.evaluate.probe import heldout_probe

    data = model.assemble(adata)
    seed, names = model.config.seed, [str(v) for v in data.type_names]
    exclude = (model.config.unassigned_label,)
    lat = model.latents(data)                     # train tiles, then held-out tiles
    rows = lat["nodes"]
    t_rows = data.t[rows]
    out: dict = {"nmi_z_targets": z_type_nmi(*nmi_inputs(lat["mu_z"], t_rows, names, exclude),
                                             seed=seed),
                 "nmi_w_targets": z_type_nmi(*nmi_inputs(lat["mu_w"], t_rows, names, exclude),
                                             seed=seed)}
    del lat
    cycle = cell_cycle_r2(model, adata, data=data)
    if cycle is not None:
        out["cycle_r2_z_q90"], out["cycle_r2_w_q90"] = cycle["z"], cycle["w"]
    out["probe"] = heldout_probe(model, adata, data=data)
    return out


def stage_reads(spec, args, adata) -> None:
    from discell.model.model import Model
    from discell.sensitivity.sweep import run_dir

    for kappa, seed in my_points(spec, args):
        if kappa != float(spec["final_kappa"]):        # the grid's other fits: no reads
            continue
        where = run_dir(fits_root(args), kappa, seed)
        target = where / "reads.json"
        if target.exists():
            continue
        t0 = time.time()
        model = Model.load(where, device=args.device)
        reads = fit_reads(model, adata)
        m = model.metrics
        reads["fit"] = {"best": m["best"], "last_epoch": m["last_epoch"],
                        "minutes": m["minutes"], "dead_w_channel": m["dead_w_channel"],
                        **{k: m["final"][k] for k in ("recon_val", "recon_val_targets", "nmi",
                                                      "nmi_targets", "mirror", "w_channel")}}
        if abs(reads["nmi_z_targets"] - m["final"]["nmi_targets"]) > 1e-9:
            log.warning("reads %s: NMI of mu_z %.6f != the fit's nmi_targets %.6f", where,
                        reads["nmi_z_targets"], m["final"]["nmi_targets"])
        if not (where / "probe.json").exists():
            (where / "probe.json").write_text(json.dumps(reads["probe"], default=float))
        target.write_text(json.dumps(reads, default=float))
        log.info("reads kappa %g seed %d: %.1f min", kappa, seed, (time.time() - t0) / 60)




def gpu_neighbors() -> list[str]:
    """Other compute processes on the machine's GPUs (a timing is clean when empty)."""
    import os
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ["nvidia-smi unavailable"]
    mine = str(os.getpid())
    return [s.strip() for s in out.splitlines() if s.strip() and not s.startswith(mine + ",")]


def stage_timing(spec, args, adata) -> dict:
    """The cost of a fit (Table S3): N training epochs with no evaluation, the
    device synchronised around each (s per epoch excludes the first, warm-up
    epoch), then one closing evaluation timed on its own; peak allocated GPU
    memory during training (on top of the resident tiles) and during the
    evaluation; the network's parameters (the adversary not counted)."""
    import dataclasses

    import torch

    from discell.model.data import assemble
    from discell.model.trainer import Trainer, resolve_device

    spec_t = variant_of(spec, args)["timing"]
    target = args.work / "variants" / args.variant / "timing.json"
    if target.exists():
        return json.loads(target.read_text())
    config = dataclasses.replace(config_of(spec, args), seed=int(spec_t["seed"]))
    data = assemble(adata, config.label_key, config.image_key,
                    counts_layer=config.counts_layer, tile_cells=config.tile_cells,
                    val_fraction=config.val_fraction, v_pcs=config.v_pcs, seed=config.seed)
    config = config.resolve(data.median_counts)
    device = resolve_device(config.device)
    cuda = device.type == "cuda"
    before = gpu_neighbors() if cuda else []
    trainer = Trainer(config, data, device)
    n_params = int(sum(p.numel() for p in trainer.network.parameters()))

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    def peak():
        return torch.cuda.max_memory_allocated(device) / 2 ** 20 if cuda else None
    sync()
    resident = torch.cuda.memory_allocated(device) / 2 ** 20 if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    epoch_s = []
    for epoch in range(int(spec_t["epochs"])):
        sync()
        t0 = time.perf_counter()
        for index in trainer.rng.permutation(len(trainer.train_batches)):
            trainer.step(trainer.train_batches[index], epoch)
        trainer.schedule.step()
        sync()
        epoch_s.append(time.perf_counter() - t0)
        log.info("timing epoch %d: %.2f s", epoch, epoch_s[-1])
    train_peak = peak()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    trainer.evaluate()
    sync()
    eval_s = time.perf_counter() - t0
    steady = epoch_s[1:] if len(epoch_s) > 1 else epoch_s
    out = {"section": spec["section"], "n_epochs": len(epoch_s), "epoch_s": epoch_s,
           "s_per_epoch": float(np.mean(steady)), "s_per_epoch_all": float(np.mean(epoch_s)),
           "eval_s": eval_s, "resident_mib": resident, "peak_train_mib": train_peak,
           "peak_eval_mib": peak(), "n_params": n_params,
           "n_train_tiles": len(trainer.train_batches), "n_val_tiles": len(trainer.val_batches),
           "n_cells": int(data.n_cells), "phi_dim": int(data.phi.shape[1]),
           "device": str(device),
           "gpu": torch.cuda.get_device_name(device) if cuda else None,
           "gpu_neighbors_before": before,
           "gpu_neighbors_after": gpu_neighbors() if cuda else []}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1))
    log.info("timing: %.2f s/epoch, evaluation %.1f s, peak %.0f MiB", out["s_per_epoch"],
             eval_s, train_peak or 0)
    return out


def lr_genes(args) -> list[str] | None:
    if args.lr_genes:
        return Path(args.lr_genes).read_text().split()
    if args.cellchatdb:
        sys.path.insert(0, str(HERE / "scripts"))
        from lr_genes import cellchat_lr_genes

        return cellchat_lr_genes(args.cellchatdb)
    log.warning("no --lr-genes / --cellchatdb: the signaling member is left out, and the "
                "findings (and their Bonferroni level) are not the paper's")
    return None


def stage_draws(spec, args, adata) -> None:
    from discell.model.model import Model
    from discell.sensitivity.findings import fit_draws

    genes = lr_genes(args)
    for kappa, seed in my_points(spec, args):
        where = fit_dir(args.work, kappa, seed)
        t0 = time.time()
        model = Model.load(where, device=args.device)
        fit_draws(model, adata, lr_genes=genes, n=QUICK_DRAWS if args.quick else None,
                  out_dir=where / "draws")
        log.info("draws kappa %g seed %d: %.1f min", kappa, seed, (time.time() - t0) / 60)


def stage_breakdown(spec, args) -> None:
    from discell.sensitivity.findings import finding_breakdown

    dirs = {(k, s): fit_dir(args.work, k, s) / "draws" for k, s in grid_points(spec, False)}
    missing = [str(d) for d in dirs.values() if not d.exists()]
    if missing:
        raise SystemExit(f"breakdown: draws missing for {len(missing)} fits, e.g. {missing[0]}")
    table = finding_breakdown(dirs, grid=spec["kappas"], seeds=spec["seeds"])
    (args.work / "breakdown.json").write_text(json.dumps(table, indent=1, default=float))
    log.info("breakdown: %s", {k: v["status"] for k, v in table["members"].items()})


# -- compare -----------------------------------------------------------------------

def band(ours: list[float], ref: list[float], floor: float,
         require_overlap: bool = True) -> dict:
    """The refit tolerance: the seed ranges overlap
    (unless *require_overlap* is off) and the means differ by at most
    max(2 x the paper's range, *floor*). With one fit (quick): the value lies within
    that tolerance of the paper's range."""
    tol = max(2 * (max(ref) - min(ref)), floor)
    out = {"ours": ours, "paper": ref, "tolerance": tol}
    if len(ours) == 1:
        out["pass"] = bool(min(ref) - tol <= ours[0] <= max(ref) + tol)
    else:
        out["mean_diff"] = float(np.mean(ours) - np.mean(ref))
        out["ranges_overlap"] = bool(min(ours) <= max(ref) and min(ref) <= max(ours))
        out["pass"] = bool((out["ranges_overlap"] or not require_overlap)
                           and abs(out["mean_diff"]) <= tol)
    return out


def compare_breakdown(ours: dict, key: str) -> dict:
    """Every member's (status, kappa*) against ``breakdown_all.json[key]``."""
    paper = json.loads((HERE / "expected" / "breakdown_all.json").read_text())[key]
    members = {}
    for name, e in paper["members"].items():
        o = ours["members"].get(name)
        members[name] = {"paper": [e["status"], e["kappa_star"]],
                         "ours": None if o is None else [o["status"], o["kappa_star"]]}
        members[name]["same"] = members[name]["ours"] == members[name]["paper"]
    return {"m_s": [paper["m_s"], ours["m_s"]], "members": members,
            "pass": bool(paper["m_s"] == ours["m_s"]
                         and all(v["same"] for v in members.values()))}


def share(excess: float) -> float:
    """A probe excess as the share of within-type variance, 1 - exp(-2 excess)."""
    return float(-np.expm1(-2.0 * float(excess)))


def probe_reads(probes: list[dict], probe_ref: dict, seeds) -> dict:
    """The probe's var_fraction (the share 1 - exp(-2 excess)) per block against the
    paper's seeds. The expected records hold the earlier exp(2 excess) - 1, so the
    paper's share is recomputed from their excess."""
    out = {}
    for block in PROBE_BLOCKS:
        family, part = block.split("_")
        out[f"probe_{block}_var_fraction"] = band(
            [float(p[family][part]["var_fraction"]) for p in probes],
            [share(probe_ref[f"DisCell/seed{s}"][block]["excess"]) for s in seeds],
            FLOORS["var_fraction"])
    return out


def stage_compare(spec, args, adata) -> dict:
    from discell.evaluate.probe import heldout_probe
    from discell.model.model import Model

    expected = HERE / "expected" / spec["section"]
    seeds = [int(spec["seeds"][0])] if args.quick else [int(s) for s in spec["seeds"]]
    final = float(spec["final_kappa"])
    ref_all = [json.loads((expected / f"seed{s}" / "metrics.json").read_text())["best"]
               for s in spec["seeds"]]
    probe_ref = json.loads((expected / "probe_reference.json").read_text())
    ours, probes, guards = [], [], {}
    for s in seeds:
        where = fit_dir(args.work, final, s)
        ours.append(json.loads((where / "metrics.json").read_text()))
        cache = where / "probe.json"
        if not cache.exists():
            model = Model.load(where, device=args.device)
            cache.write_text(json.dumps(heldout_probe(model, adata), default=float))
        probes.append(json.loads(cache.read_text()))
    for k, s in grid_points(spec, args.quick):
        m = json.loads((fit_dir(args.work, k, s) / "metrics.json").read_text())
        guards[f"k{k:g}_s{s}"] = {
            "niche_information_check": m["final"]["w_channel"]["niche_information_check"],
            "dead": bool(m["dead_w_channel"] or m["final"]["w_channel"].get(
                "dead_context_channel", False))}
    report: dict = {"section": spec["section"], "quick": bool(args.quick),
                    "image_free": bool(args.no_image),
                    "encoder": None if args.no_image else spec["embed"]["encoder"],
                    "paper_encoder": PAPER_ENCODER, "seeds": seeds, "reads": {}}
    epochs = [int(m["best"]["epoch"]) for m in ours]
    paper_epochs = [int(r["epoch"]) for r in ref_all]
    overlap = bool(min(epochs) <= max(paper_epochs) and min(paper_epochs) <= max(epochs))
    report["reads"]["best_epoch"] = {"ours": epochs, "paper": paper_epochs,
                                     "ranges_overlap": overlap, "pass": overlap}
    for key in ("recon_val", "nmi"):
        # NMI moves more between refits of one seed than across the paper's seeds,
        # so only its mean is checked
        report["reads"][key] = band([float(m["best"][key]) for m in ours],
                                    [float(r[key]) for r in ref_all], FLOORS[key],
                                    require_overlap=key != "nmi")
    report["reads"].update(probe_reads(probes, probe_ref, spec["seeds"]))
    report["guard"] = {"fits": guards,
                       "pass": all(g["niche_information_check"] > 0 and not g["dead"]
                                   for g in guards.values())}
    bfile = args.work / "breakdown.json"
    if not args.quick and bfile.exists():
        report["breakdown"] = compare_breakdown(json.loads(bfile.read_text()),
                                                spec["breakdown_key"])
    checks = [v["pass"] for v in report["reads"].values()] + [report["guard"]["pass"]]
    if "breakdown" in report:
        checks.append(report["breakdown"]["pass"])
    report["pass"] = bool(all(checks))
    (args.work / "report.json").write_text(json.dumps(report, indent=1, default=float))
    (args.work / "report.md").write_text(markdown(report))
    return report


def stage_serial(spec, args) -> dict | None:
    """The core's fits on the serial section, with the release API only."""
    import anndata as ad
    import pandas as pd

    import discell as dc
    from discell.evaluate.probe import heldout_probe
    from discell.model.model import Model
    from discell.sensitivity.findings import finding_breakdown, fit_draws

    serial = spec.get("serial")
    if serial is None:
        raise SystemExit(f"{spec['section']} has no serial section")
    if not (args.serial_cell_groups and args.serial_label_dir):
        raise SystemExit("serial: pass --serial-cell-groups and --serial-label-dir "
                         "(the in-house slide-10 labels, not shipped)")
    work = args.work / "serial"
    work.mkdir(parents=True, exist_ok=True)
    path = work / "section.h5ad"
    if not path.exists():
        outs = (Path(args.serial_outs) if args.serial_outs
                else Path(args.data_root) / serial["outs"])
        labels = Path(args.serial_label_dir)
        t0 = time.time()
        adata = dc.read_xenium(outs, labels=args.serial_cell_groups, donors=spec["donors"])
        source = pd.Series(adata.obs[spec["labels"]["source_key"]].astype(object).to_numpy(),
                           index=adata.obs_names)
        adata.obs["lineage"] = pd.Categorical(lineage(
            source, labels / "lineage_map_proposed.csv", choices=spec["labels"]["choices"],
            reassignments=sorted(labels.glob("*_reassignment_proposed.csv"))).to_numpy())
        if adata.n_obs != serial["n_cells"]:
            log.warning("serial: %d cells, the paper's has %d", adata.n_obs, serial["n_cells"])
        dc.build_graph(adata, "lineage")
        write_atomic(adata, path)
        log.info("serial ingest: %d cells in %.1f min", adata.n_obs, (time.time() - t0) / 60)
    if not args.no_image:
        stage_embed(spec, args, path)
    adata = ad.read_h5ad(path)

    final = float(spec["final_kappa"])
    seeds = [int(spec["seeds"][0])] if args.quick else [int(s) for s in spec["seeds"]]
    for s in seeds:
        out = fit_dir(work, final, s)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "cross_section.json").exists():
            continue
        model = Model.load(fit_dir(args.work, final, s), device=args.device)
        data = model.assemble(adata)
        (out / "probe.json").write_text(json.dumps(heldout_probe(model, adata, data=data),
                                                   default=float))
        reads = dc.tl.cross_section(model, adata.copy())
        (out / "cross_section.json").write_text(json.dumps(reads, default=float))
        log.info("serial probe seed %d", s)
    genes = lr_genes(args)
    for kappa, seed in my_points(spec, args):
        t0 = time.time()
        model = Model.load(fit_dir(args.work, kappa, seed), device=args.device)
        fit_draws(model, adata, lr_genes=genes, n=QUICK_DRAWS if args.quick else None,
                  out_dir=fit_dir(work, kappa, seed) / "draws")
        log.info("serial draws kappa %g seed %d: %.1f min", kappa, seed,
                 (time.time() - t0) / 60)

    points = grid_points(spec, args.quick)
    if not all((fit_dir(work, k, s) / "draws").exists() for k, s in points):
        log.info("serial: draws of other parts missing; rerun with --part 0/1 to compare")
        return None
    probe_ref = json.loads((HERE / "expected" / serial["expected"]
                            / "probe_reference.json").read_text())
    probes = [json.loads((fit_dir(work, final, s) / "probe.json").read_text()) for s in seeds]
    report: dict = {"section": f"{spec['section']} serial ({serial['expected']})",
                    "quick": bool(args.quick), "image_free": bool(args.no_image),
                    "encoder": None if args.no_image else spec["embed"]["encoder"],
                    "paper_encoder": PAPER_ENCODER, "seeds": seeds,
                    "n_cells": int(adata.n_obs),
                    "reads": probe_reads(probes, probe_ref, spec["seeds"]),
                    "cross_section": {str(s): json.loads(
                        (fit_dir(work, final, s) / "cross_section.json").read_text())
                        for s in seeds}}
    checks = [v["pass"] for v in report["reads"].values()]
    if not args.quick:
        table = finding_breakdown({(k, s): fit_dir(work, k, s) / "draws" for k, s in points},
                                 grid=spec["kappas"], seeds=spec["seeds"])
        (work / "breakdown.json").write_text(json.dumps(table, indent=1, default=float))
        report["breakdown"] = compare_breakdown(table, serial["breakdown_key"])
        checks.append(report["breakdown"]["pass"])
    report["pass"] = bool(all(checks))
    (work / "report.json").write_text(json.dumps(report, indent=1, default=float))
    (work / "report.md").write_text(markdown(report))
    return report


def markdown(report: dict) -> str:
    lines = [f"# {report['section']}: reproduction report",
             "", f"Overall: **{'PASS' if report['pass'] else 'FAIL'}**"
             + (" (quick: one fit)" if report["quick"] else "")
             + (" (image-free: not the paper's configuration)" if report["image_free"] else "")
             + (f" (encoder {report['encoder']}; the paper's numbers are from "
                f"{report['paper_encoder']}, so image-dependent reads are expected to differ: "
                "see reproduce/README.md)"
                if report["encoder"] not in (None, report["paper_encoder"]) else ""),
             "", "| read | ours | reference fits | mean diff | tolerance | ranges overlap | pass |",
             "|---|---|---|---|---|---|---|"]

    def fmt(v):
        return ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v)
    for name, r in report["reads"].items():
        tol = f"{r['tolerance']:.4g}" if "tolerance" in r else ""
        diff = f"{r['mean_diff']:+.4g}" if "mean_diff" in r else ""
        overlap = {True: "yes", False: "NO", None: ""}[r.get("ranges_overlap")]
        lines.append(f"| {name} | {fmt(r['ours'])} | {fmt(r['paper'])} | {diff} | {tol} | "
                     f"{overlap} | {'yes' if r['pass'] else 'NO'} |")
    g = report.get("guard")
    if g is not None:
        lines += ["", "w-channel guard (I(niche; w) excess > 0, no dead channel) on every "
                  f"fit: {'yes' if g['pass'] else 'NO'}"]
    if "breakdown" in report:
        b = report["breakdown"]
        lines += ["", f"Breakdown points (m = {b['m_s'][1]}, reference fits {b['m_s'][0]}):",
                  "", "| member | ours | reference fits |", "|---|---|---|"]
        for name, v in b["members"].items():
            lines.append(f"| {name} | {v['ours']} | {v['paper']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--section", required=True,
                        choices=sorted(p.stem for p in (HERE / "configs").glob("*.json")))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data-root", default="data",
                        help="where download.sh put the data (default data/)")
    parser.add_argument("--outs", default=None, help="the Xenium outs directory itself")
    parser.add_argument("--cell-groups", default=None, help="TMA core: cell_id, group, donor CSV")
    parser.add_argument("--label-dir", default=None,
                        help="TMA core: lineage_map_proposed.csv + *_reassignment_proposed.csv")
    parser.add_argument("--serial-outs", default=None,
                        help="serial stage: slide-10 outs (default <data-root>/<config's>)")
    parser.add_argument("--serial-cell-groups", default=None,
                        help="serial stage: slide-10 cell_id, group, donor CSV")
    parser.add_argument("--serial-label-dir", default=None,
                        help="serial stage: slide-10 lineage map + reassignment tables")
    parser.add_argument("--lr-genes", default=None,
                        help="ligand-receptor genes, one per line (scripts/lr_genes.py)")
    parser.add_argument("--cellchatdb", default=None, help="CellChatDB.human.rda (needs rdata)")
    parser.add_argument("--variant", default=None,
                        help="a config-only refit from the section config's 'variants'")
    parser.add_argument("--stages", default=None,
                        help=f"default {','.join(STAGES)}; with --variant fits,reads "
                             "(timing for a timing variant)")
    parser.add_argument("--part", default="0/1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-image", action="store_true",
                        help="image-free fits (not the paper's configuration)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    spec = load_section(args.section)
    variant = variant_of(spec, args)
    if args.stages is None:
        args.stages = (",".join(STAGES) if variant is None
                       else "timing" if "timing" in variant else "fits,reads")
    stages = [s for s in args.stages.split(",") if s]
    unknown = set(stages) - set(STAGES) - set(EXTRA_STAGES)
    if unknown:
        parser.error(f"unknown stages {sorted(unknown)}")
    if variant is not None and not set(stages) <= {"fits", "reads", "timing"}:
        parser.error("--variant runs the stages fits and reads (or timing) only")
    if variant is not None and ("timing" in variant) != ("timing" in stages):
        parser.error(f"variant {args.variant}: stage timing "
                     + ("only" if "timing" in variant else "is for a timing variant"))
    args.work.mkdir(parents=True, exist_ok=True)
    path = args.work / "section.h5ad"
    if "ingest" in stages:
        stage_ingest(spec, args, path)
    if "embed" in stages and not args.no_image:
        stage_embed(spec, args, path)
    adata = None
    if set(stages) & {"fits", "draws", "compare", "reads", "timing"}:
        import anndata as ad

        adata = ad.read_h5ad(path)
    if "timing" in stages:
        stage_timing(spec, args, adata)
    if "fits" in stages:
        stage_fits(spec, args, adata)
    if "reads" in stages:
        stage_reads(spec, args, adata)
    if "draws" in stages:
        stage_draws(spec, args, adata)
    if "breakdown" in stages and not args.quick:
        stage_breakdown(spec, args)
    passed = True
    if "compare" in stages:
        report = stage_compare(spec, args, adata)
        print((args.work / "report.md").read_text())
        passed = report["pass"]
    if "serial" in stages:
        serial = stage_serial(spec, args)
        if serial is not None:
            print((args.work / "serial" / "report.md").read_text())
            passed = passed and serial["pass"]
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
