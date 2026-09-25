#!/usr/bin/env python3
"""The invariance probe re-graded per block, with a nonlinear grader (R20 + R22).

Devlog 2026-09-24 15:00. No retraining: a DisCell run is re-read from its
accepted checkpoint (``best.pt``, via :func:`validate.load_run`), a baseline
from its stored intrinsic latent, and both are graded by
:func:`metrics.probe_blocks` -- ridge and MLP, composition block and image
block, each against its own within-type permutation floor.

    # DisCell runs on their own section (several share one assembly)
    python -m discell.experiments.probe_regrade --dataset D --run final_s0 final_s1

    # DisCell runs on another section (the held-out-section protocol)
    python -m discell.experiments.probe_regrade --dataset D_dual \\
        --config-from D_solo --run final_s0

    # a baseline's stored latent, on the split and targets of --run
    python -m discell.experiments.probe_regrade --dataset D --run final_s0 \\
        --baseline-latents .../latents.h5ad --method resolVI

    # one table over whatever has been graded
    python -m discell.experiments.probe_regrade --dataset D --table final \\
        --runs 'final_s*' 'best_s*' --baselines

Artefacts (``probe_blocks.json``, same schema everywhere):

* own section   ``runs/<run>/validation/probe_blocks.json``
* other section ``<config-from>/runs/<run>/crossslide/probe_blocks_<D>.json``
* baseline      ``<D>/experiments/probe_regrade/<method>.json``
* table         ``<D>/experiments/probe_regrade_<group>.{md,json}``

The guard (devlog "Per-block probe, first read (8.16 interim)"): each
block's excess as a fraction of the uncontrolled reference's -- the seed mean
of the alpha_a = 0 fits at the final budget, ``uncontrolled500_s*``, of the
same dataset, graded on the same section (the earlier 200/20
``uncontrolled_s0`` is kept as ``fraction_of_uncontrolled200``) -- passes at
<= 25 %, and ``invariance_pass`` needs all four blocks;
fractions of resolVI's excess on the same slide are reported beside it. A
record graded before its reference exists has ``invariance_pass`` None;
``--reverdict`` re-applies the verdict to every record of a dataset
(section) without re-fitting any probe.

    python -m discell.experiments.probe_regrade --dataset D --reverdict

A DisCell run's node-ordered mu_z is cached beside its record
(``probe_latents[_<D>].npz``), so a re-grade (``--force``, another
``--n-perm``) skips the model. A record that exists is skipped without
``--force``: relaunching is safe.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from discell import paths
from discell.model import metrics as M

log = logging.getLogger("discell.probe_regrade")

#: the fields of a run's config that decide ``assemble``'s output
ASSEMBLY_KEYS = ("variant", "embeddings", "tile_cells", "phi_pca", "v_pcs",
                 "val_fraction", "seed", "label_key")
BLOCKS = M.GUARD_BLOCKS
#: the alpha_a = 0 fits at the final budget every excess is a fraction of
#: (their seed mean), the 200/20 one they replaced (kept for the record), and
#: the baseline beside them
REFERENCE_RUNS = ("uncontrolled500_s0", "uncontrolled500_s1")
REFERENCE_200 = "uncontrolled_s0"
RESOLVI = "resolVI"


def slug(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9=.+-]+", "_", method).strip("_")


def run_config(dataset: str, run: str) -> dict:
    path = paths.dataset(dataset).root / "runs" / run / "config.json"
    return json.loads(path.read_text())


def record_path(dataset: str, run: str, config_from: str | None) -> Path:
    if config_from is None or config_from == dataset:
        return (paths.dataset(dataset).root / "runs" / run / "validation"
                / "probe_blocks.json")
    return (paths.dataset(config_from).root / "runs" / run / "crossslide"
            / f"probe_blocks_{dataset}.json")


def baseline_path(dataset: str, method: str) -> Path:
    return (paths.dataset(dataset).root / "experiments" / "probe_regrade"
            / f"{slug(method)}.json")


def references(dataset: str, config_from: str | None) -> tuple:
    """([500/40 uncontrolled records], resolVI, 200/20 uncontrolled) for
    *dataset*'s section; missing ones are left out / None."""
    load = lambda p: json.loads(p.read_text()) if p.exists() else None
    refs = [load(record_path(dataset, run, config_from))
            for run in REFERENCE_RUNS]
    return ([r for r in refs if r is not None],
            load(baseline_path(dataset, RESOLVI)),
            load(record_path(dataset, REFERENCE_200, config_from)))


def write_atomic(path: Path, text: str) -> None:
    """Other tables read these records while queues write them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def latents_path(record: Path) -> Path:
    return record.with_name(record.name.replace("probe_blocks",
                                                "probe_latents")
                            .replace(".json", ".npz"))


class Assemblies:
    """``assemble`` once per (dataset, assembly key), keeping at most
    *capacity* (a held-out section needs both sections' assemblies)."""

    def __init__(self, capacity: int = 1) -> None:
        self.capacity, self.cache = capacity, {}

    def get(self, dataset: str, config: dict):
        from discell.model.prepare import assemble

        key = (dataset,) + tuple(config.get(k) for k in ASSEMBLY_KEYS)
        if key not in self.cache:
            while len(self.cache) >= self.capacity:    # free before building
                self.cache.pop(next(iter(self.cache)))
            started = time.time()
            self.cache[key] = assemble(
                dataset, config["variant"], config["embeddings"],
                tile_cells=config["tile_cells"], phi_pca=config.get("phi_pca"),
                v_pcs=config.get("v_pcs", 12),
                val_fraction=config["val_fraction"], seed=config["seed"],
                label_key=config.get("label_key"))
            log.info("assembled %s (seed %s) in %.0fs", dataset,
                     config["seed"], time.time() - started)
        return self.cache[key]


def encode(dataset: str, run: str, config_from: str | None, assemblies,
           device: str) -> np.ndarray:
    """Node-ordered mu_z of *run*'s accepted checkpoint on *dataset*'s cells."""
    import torch

    from discell.model.train import Trainer
    from discell.model.validate import collect_latents, load_run

    source = config_from or dataset
    cfg = run_config(source, run)
    data_a = assemblies.get(source, cfg)
    config, data_a, trainer, _, _ = load_run(source, run, device, data=data_a)
    if source != dataset:
        # the crossslide protocol: the trained model, the other section's
        # tiles (baseline_battery.discell_column does the same)
        data_b = assemblies.get(dataset, cfg)
        names_a = [str(x) for x in data_a.type_names]
        if names_a != [str(x) for x in data_b.type_names]:
            raise ValueError(f"type vocabularies differ: {source} / {dataset}")
        model = trainer.model
        trainer.train_batches, trainer.val_batches = [], []
        del trainer
        torch.cuda.empty_cache()
        trainer = Trainer(config, data_b)
        trainer.model = model.to(trainer.device).eval()
        data_a = data_b
    mu_z = collect_latents(trainer, data_a)["mu_z"].astype(np.float32)
    del trainer
    torch.cuda.empty_cache()
    return mu_z


def legacy_reference(dataset: str, run: str, config_from: str | None) -> dict:
    """The in-trainer probe dCE of the same weights, where one is on disk --
    the check that the re-grade read the accepted checkpoint."""
    from discell.experiments.at_best import battery_at_best

    source = config_from or dataset
    run_dir = paths.dataset(source).root / "runs" / run
    if source == dataset:
        row = battery_at_best(run_dir)
        return {"source": "history row at the accepted epoch (at_best)"
                if row.get("at_best") else "metrics final (not at best)",
                "at_best": bool(row.get("at_best")),
                **(row.get("probe") or {})}
    cross = run_dir / "crossslide" / f"{dataset}.json"
    if not cross.exists():
        return {}
    probe = json.loads(cross.read_text())["held_out_section"].get("probe", {})
    return {"source": f"crossslide/{dataset}.json", "at_best": True, **probe}


def grade_discell(dataset: str, runs: list[str], config_from: str | None,
                  n_perm: int, force: bool, device: str) -> int:
    """Grade each run; one that fails is logged and the rest go on. Returns
    the number of failures."""
    source = config_from or dataset
    assemblies = Assemblies(capacity=1 if source == dataset else 2)
    ordered = sorted(runs, key=lambda r: json.dumps(
        [run_config(source, r).get(k) for k in ASSEMBLY_KEYS]))
    failed = []
    for run in ordered:
        out = record_path(dataset, run, config_from)
        if out.exists() and not force:
            log.info("%s/%s: %s exists -- skipped", dataset, run, out.name)
            continue
        run_dir = paths.dataset(source).root / "runs" / run
        if not ((run_dir / "best.pt").exists()
                and (run_dir / "metrics.json").exists()):
            # best.pt is rewritten during a fit; metrics.json marks its end
            log.warning("%s/%s: not a finished fit -- skipped", source, run)
            continue
        try:
            grade_one(dataset, run, config_from, assemblies, n_perm, device,
                      out)
        except Exception:
            log.exception("%s/%s: FAILED", dataset, run)
            failed.append(run)
    if failed:
        log.error("%d run(s) failed: %s", len(failed), " ".join(failed))
    return len(failed)


def grade_one(dataset: str, run: str, config_from: str | None, assemblies,
              n_perm: int, device: str, out: Path) -> None:
    from discell.model.validate import probe_blocks_for_run

    source = config_from or dataset
    started = time.time()
    cfg = run_config(source, run)
    cache = latents_path(out)
    data = assemblies.get(dataset, cfg)
    if cache.exists() and len(np.load(cache)["mu_z"]) == data.n_cells:
        mu_z = np.load(cache)["mu_z"]
        log.info("%s/%s: cached mu_z", dataset, run)
    else:
        mu_z = encode(dataset, run, config_from, assemblies, device)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f".{cache.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, mu_z=mu_z)
        os.replace(tmp, cache)
        data = assemblies.get(dataset, cfg)
    record = probe_blocks_for_run(data, mu_z, seed=cfg["seed"], n_perm=n_perm)
    metrics = json.loads((paths.dataset(source).root / "runs" / run
                          / "metrics.json").read_text())
    record.update(dataset=dataset, run=run, method=f"DisCell/{run}",
                  trained_on=source, checkpoint="best.pt",
                  best_epoch=metrics["best"]["epoch"],
                  legacy_in_trainer=legacy_reference(dataset, run,
                                                     config_from),
                  minutes=(time.time() - started) / 60)
    refs, resolvi, ref_200 = references(dataset, config_from)
    if run in REFERENCE_RUNS:            # a reference counts itself in
        refs = [r for r in refs if r.get("run") != run] + [record]
    M.probe_verdict(record, refs, resolvi,
                    record if run == REFERENCE_200 else ref_200)
    write_atomic(out, json.dumps(record, indent=1, default=float))
    log.info("%s/%s: invariance_pass %s %s (%.1f min) -> %s", dataset,
             run, record["invariance_pass"], summary(record),
             record["minutes"], out)


def grade_baseline(dataset: str, run: str, config_from: str | None,
                   latents: str, method: str, n_perm: int,
                   force: bool) -> None:
    from discell.experiments.baseline_battery import load_latents

    out = baseline_path(dataset, method)
    if out.exists() and not force:
        log.info("%s '%s': exists -- skipped", dataset, method)
        return
    started = time.time()
    cfg = run_config(config_from or dataset, run)
    data = Assemblies().get(dataset, cfg)
    test = np.zeros(data.n_cells, dtype=bool)
    for tile in data.val_tiles:
        test[tile] = True
    tool = method.split("/")[0].split("_")[0].split(" ")[0].lower()
    rows, z, _, tool_cfg = load_latents(latents, tool)
    if rows.max() >= data.n_cells:
        raise ValueError("latents carry cell ids outside the dataset")
    record = M.probe_blocks(np.asarray(z, dtype=np.float64), data.t[rows],
                            data.v_block[rows], data.vbar_t, ~test[rows],
                            test[rows], n_comp=data.n_comp, seed=cfg["seed"],
                            n_perm=n_perm)
    battery = (paths.dataset(dataset).root / "experiments"
               / "baseline_battery.json")
    stored = (json.loads(battery.read_text()).get(method, {}).get("probe", {})
              if battery.exists() else {})
    record.update(dataset=dataset, run=None, method=method,
                  split_run=run, split_from=config_from or dataset,
                  latents=str(latents), d_intrinsic=int(z.shape[1]),
                  n_cells=int(len(rows)), n_heldout=int(test[rows].sum()),
                  tool_config=tool_cfg,
                  legacy_in_trainer={"source": "baseline_battery.json",
                                     **stored} if stored else {},
                  minutes=(time.time() - started) / 60)
    refs, resolvi, ref_200 = references(dataset, config_from)
    M.probe_verdict(record, refs, record if method == RESOLVI else resolvi,
                    ref_200)
    write_atomic(out, json.dumps(record, indent=1, default=str))
    log.info("%s '%s': invariance_pass %s %s -> %s", dataset, method,
             record["invariance_pass"], summary(record), out)


def reverdict(dataset: str, config_from: str | None) -> None:
    """Re-apply the guard to every record of *dataset*'s section: the DisCell
    runs (own section, or ``--config-from``'s runs on this one) and the
    baselines. Probe numbers are untouched."""
    refs, resolvi, ref_200 = references(dataset, config_from)
    runs_dir = paths.dataset(config_from or dataset).root / "runs"
    records = [record_path(dataset, p.name, config_from)
               for p in sorted(runs_dir.iterdir())
               if p.is_dir() and not p.is_symlink()]
    root = paths.dataset(dataset).root / "experiments" / "probe_regrade"
    records = [p for p in records if p.exists()] + (
        sorted(root.glob("*.json")) if root.exists() else [])
    for path in records:
        record = json.loads(path.read_text())
        is_resolvi = record.get("run") is None and record["method"] == RESOLVI
        M.probe_verdict(record, refs, record if is_resolvi else resolvi,
                        ref_200)
        write_atomic(path, json.dumps(record, indent=1, default=str))
    log.info("%s: re-verdicted %d records (500/40 uncontrolled references: "
             "%s; 200/20 %s; resolVI %s)", dataset, len(records),
             ", ".join(r["run"] for r in refs) or "MISSING",
             "present" if ref_200 else "missing",
             "present" if resolvi else "missing")


# -- tables -----------------------------------------------------------------

def summary(record: dict) -> str:
    frac = lambda x: "--" if x is None else f"{x:.2f}"
    return " ".join(f"{f}/{b} {record[f][b]['excess']:+.4f}"
                    f"(x{frac(record[f][b].get('fraction_of_uncontrolled'))}u)"
                    for f, b in BLOCKS) + f" invariance_pass {record['invariance_pass']}"


def _verdict(value) -> str:
    return "--" if value is None else ("pass" if value else "FAIL")


def _cell(block: dict) -> str:
    frac = lambda key: ("--" if block.get(key) is None
                        else f"{block[key]:.2f}")
    var = block.get("var_fraction", float(np.expm1(2 * block["excess"])))
    return (f"**{block['excess']:+.4f}** ({100 * var:.2f} %) "
            f"· {frac('fraction_of_uncontrolled')} u "
            f"({frac('fraction_of_uncontrolled200')} u200) · "
            f"{frac('fraction_of_resolvi')} r · {_verdict(block.get('pass'))}")


def _legacy_check(record: dict) -> str:
    ref = record.get("legacy_in_trainer") or {}
    if "delta_ce" not in ref:
        return "--"
    diff = record["ridge"]["legacy"]["delta_ce"] - ref["delta_ce"]
    return f"{diff:+.1e}" + ("" if ref.get("at_best", True) else " (ref not at best)")


def render(dataset: str, group: str, records: list[dict]) -> str:
    head = (["method / run", "held out"]
            + [f"{f} {b}" for f, b in BLOCKS]
            + ["invariance_pass (≤ 25 % of uncontrolled)",
               "2-sd rule (withdrawn)", "ridge pooled excess",
               "pooled, legacy: ΔCE / floor", "legacy vs in-trainer"])
    lines = [f"# Invariance probe re-graded per block -- {dataset}, group `{group}`",
             "",
             "Generated by `discell/experiments/probe_regrade.py` (devlog "
             "2026-09-24 15:00, R20 + R22). DisCell rows read the accepted "
             "checkpoint (`best.pt`); baselines their stored intrinsic latent "
             "on the same split and targets.",
             "",
             "Per column k of v, gain_k = ½ log(MSE_k(type mean) / "
             "MSE_k(probe)) on held-out cells (nats); a block's gain is the "
             "mean over its columns (composition: K−1 columns of y; image: "
             "12 PCs of Φ). Floor = the same with z permuted within type "
             "(mean over the permutation draws). Each cell: **excess** = "
             "gain − floor (nats per column), (the within-type variance "
             "fraction it implies, exp(2·excess) − 1) · the excess as a "
             "fraction of the uncontrolled fits' (α_a = 0 at the final "
             "500/40 budget, `uncontrolled500_s*`, seed mean, same section; "
             "`u`; in brackets against the earlier 200/20 `uncontrolled_s0`, "
             "`u200`, for the record) · as a fraction of resolVI's on the "
             "same slide (`r`) · pass iff ≤ 0.25 u. `invariance_pass` = "
             "all four blocks pass (devlog \"Per-block probe, first read "
             "(8.16 interim)\"; `--` = no uncontrolled reference yet). The "
             "withdrawn 15:00 rule (|excess| ≤ 2 floor sd) is shown beside "
             "it. The pooled, legacy ΔCE is `probe_delta_ce` recomputed on "
             "the same inputs (image-dominated); `legacy vs in-trainer` is "
             "its difference from the in-trainer value of the same weights "
             "(or the stored battery column), the check that the right "
             "checkpoint was read.",
             "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in records:
        rp = r["ridge"]["pooled"]
        lg = r["ridge"]["legacy"]
        lines.append("| " + " | ".join(
            [r["method"], f"{r.get('n_heldout') or r['ridge']['n_test']:,}"]
            + [_cell(r[f][b]) for f, b in BLOCKS]
            + [_verdict(r.get("invariance_pass")),
               _verdict(r.get("invariance_pass_2sd_legacy")),
               f"{rp['excess']:+.4f}",
               f"{lg['delta_ce']:+.4f} / {lg['noise_floor']:+.4f}",
               _legacy_check(r)]) + " |")
    n_perm = sorted({r["ridge"]["n_perm"] for r in records})
    lines += ["", f"Floor draws per block: {', '.join(map(str, n_perm))}; "
              f"probe fitted on ≤{M.MAX_EVAL_CELLS:,} training cells, graded "
              f"on ≤{M.MAX_EVAL_CELLS:,} held-out cells."]
    return "\n".join(lines) + "\n"


def collect(dataset: str, patterns: list[str], config_from: str | None,
            baselines: bool) -> list[dict]:
    source = config_from or dataset
    runs_dir = paths.dataset(source).root / "runs"
    names: dict = {}       # run directory -> the symlinked name, if any
    for p in sorted(runs_dir.iterdir()):
        if any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
            target = p.resolve().name
            if p.is_symlink():
                names[target] = p.name       # best_s0 -> best_az0.5_s0
            else:
                names.setdefault(target, None)
    records = []
    for name in sorted(names):
        path = record_path(dataset, name, config_from)
        if path.exists():
            record = json.loads(path.read_text())
            if names[name]:
                record["method"] += f" (= {names[name]})"
            records.append(record)
    if baselines:
        root = paths.dataset(dataset).root / "experiments" / "probe_regrade"
        for path in sorted(root.glob("*.json")) if root.exists() else []:
            records.append(json.loads(path.read_text()))
    return records


def table(dataset: str, group: str, patterns: list[str],
          config_from: str | None, baselines: bool) -> None:
    records = collect(dataset, patterns, config_from, baselines)
    if not records:
        log.warning("%s: nothing graded for group %s", dataset, group)
        return
    root = paths.dataset(dataset).root / "experiments"
    write_atomic(root / f"probe_regrade_{group}.md",
                 render(dataset, group, records))
    write_atomic(root / f"probe_regrade_{group}.json", json.dumps(
        {r["method"]: {"invariance_pass": r["invariance_pass"],
                       "pass": r["pass"],
                       "invariance_pass_2sd_legacy": r.get(
                           "invariance_pass_2sd_legacy"),
                       **{f"{f}_{b}": {k: r[f][b].get(k) for k in
                                       ("gain", "floor_mean", "floor_sd",
                                        "excess", "var_fraction",
                                        "excess_in_sd",
                                        "fraction_of_uncontrolled",
                                        "fraction_of_resolvi", "pass",
                                        "pass_2sd_legacy")}
                          for f, b in BLOCKS + (("ridge", "pooled"),
                                                ("mlp", "pooled"))},
                       "legacy": r["ridge"]["legacy"]}
         for r in records}, indent=1, default=float))
    log.info("wrote %s (%d rows)", root / f"probe_regrade_{group}.md",
             len(records))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--run", nargs="*", default=[],
                   help="DisCell run(s); with --baseline-latents, the run whose "
                        "split and targets grade the baseline")
    p.add_argument("--config-from", default=None,
                   help="dataset the run was trained on (held-out section)")
    p.add_argument("--baseline-latents", default=None)
    p.add_argument("--method", default=None, help="baseline column name")
    p.add_argument("--n-perm", type=int, default=5)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--table", default=None, metavar="GROUP",
                   help="render probe_regrade_<GROUP>.md from the records")
    p.add_argument("--runs", nargs="*", default=[],
                   help="--table: run-name patterns (fnmatch)")
    p.add_argument("--baselines", action="store_true",
                   help="--table: add the graded baselines")
    p.add_argument("--reverdict", action="store_true",
                   help="re-apply the guard to every record of the section")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.reverdict:
        reverdict(args.dataset, args.config_from)
        return 0
    if args.table:
        table(args.dataset, args.table, args.runs, args.config_from,
              args.baselines)
        return 0
    if args.baseline_latents:
        if not (args.method and len(args.run) == 1):
            p.error("--baseline-latents needs --method and one --run")
        grade_baseline(args.dataset, args.run[0], args.config_from,
                       args.baseline_latents, args.method, args.n_perm,
                       args.force)
        return 0
    if not args.run:
        p.error("give --run, --baseline-latents or --table")
    return int(grade_discell(args.dataset, args.run, args.config_from,
                             args.n_perm, args.force, args.device) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
