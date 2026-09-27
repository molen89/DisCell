#!/usr/bin/env python3
"""Pure-training timing per dataset (metrics package item 4, 2026-09-25).

One arm per call, at the dataset's final configuration (``runs/final_s0/
config.json``; the seed and schedule do not matter for a timing): N epochs of
``Trainer.time_only`` -- training steps only, no evaluation between them --
then one timed evaluation; s/epoch (first epoch excluded as warm-up), s per
evaluation, peak allocated GPU MiB.

Arms:
  ``phi``          the model as fitted (384-d Phi in c)
  ``phi_zeroed``   Phi replaced by zeros, the existing no-image option
                   (``calibrate``'s ``cal_no_phi``); compute-identical to
                   ``phi`` by construction -- it is the control for the
                   timing noise, not a cheaper model
  ``phi_dropped``  Phi given zero columns: c = [GAT, flag], the model without
                   the image channel at all (what "without Phi" costs)
  ``phiproj32``    the S53 arm, a learned 384 -> 32 projection (ovarian)

Usage::

    python scripts/timing_mode.py --dataset <id> --arm phi --epochs 20
    python scripts/timing_mode.py --tables     # timing.md per dataset + combined

``--ref-run R`` takes the configuration from ``runs/R`` instead of
``final_s0``; ``--tag T`` writes ``experiments/timing<T>/`` and
``timing<T>.md`` and reads the baselines from ``baseline_battery<T>.json``
(the lineage relabel: ``--ref-run finalL_s0 --tag _lineage``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from discell import paths

DATASETS = ("gse315411_pdltma06_11_prime_solo",
            "xenium_prime_human_lung_cancer_ffpe",
            "xenium_prime_ovarian_cancer_ffpe",
            "xenium_prime_human_ovary_ff")
SHORT = {"gse315411_pdltma06_11_prime_solo": "GSE",
         "xenium_prime_human_lung_cancer_ffpe": "lung",
         "xenium_prime_ovarian_cancer_ffpe": "ovarian",
         "xenium_prime_human_ovary_ff": "FF"}
ARMS = ("phi", "phi_zeroed", "phi_dropped", "phiproj32")
COMBINED = Path("scripts/logs/metrics_2026-09-25/timing_all.md")


def gpu_neighbours() -> list[str]:
    """Other compute processes on the visible GPU (a timing is only clean
    when this is empty)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20).stdout
    except Exception:
        return ["nvidia-smi unavailable"]
    import os

    mine = str(os.getpid())
    return [line.strip() for line in out.splitlines()
            if line.strip() and not line.startswith(mine + ",")]


def time_arm(dataset: str, arm: str, epochs: int, ref_run: str = "final_s0",
             tag: str = "") -> dict:
    from discell.model.prepare import assemble
    from discell.model.train import TrainConfig, Trainer

    ref = json.loads((paths.dataset(dataset).root / "runs" / ref_run
                      / "config.json").read_text())
    ref.pop("git", None)
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    config = TrainConfig(**{k: v for k, v in ref.items() if k in fields})
    config = dataclasses.replace(
        config, run_name=f"timing_{arm}",
        phi_proj=32 if arm == "phiproj32" else 0)
    data = assemble(dataset, config.variant, config.embeddings,
                    tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                    v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                    seed=config.seed, label_key=config.label_key)
    if arm == "phi_zeroed":
        data = dataclasses.replace(data, phi=np.zeros_like(data.phi))
    elif arm == "phi_dropped":
        data = dataclasses.replace(data, phi=np.zeros((len(data.phi), 0),
                                                      dtype=data.phi.dtype))
    before = gpu_neighbours()
    trainer = Trainer(config, data)
    n_params = int(sum(p.numel() for p in trainer.model.parameters()))
    out = trainer.time_only(epochs)
    out.update(arm=arm, n_params=n_params, config_from=f"runs/{ref_run}",
               gpu_neighbours_before=before, gpu_neighbours_after=gpu_neighbours())
    target = (paths.dataset(dataset).root / "experiments" / f"timing{tag}"
              / f"timing_{arm}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("s_per_epoch", "eval_s",
                                          "peak_train_mib", "peak_eval_mib")}))
    return out


def baselines(dataset: str, tag: str = "") -> list[tuple[str, dict]]:
    path = paths.dataset(dataset).root / "experiments" / f"baseline_battery{tag}.json"
    if not path.exists():
        return []
    out = []
    for name, entry in json.loads(path.read_text()).items():
        cfg = (entry or {}).get("config") or {}
        if cfg.get("train_s") is not None:
            out.append((name, cfg))
    return out


def _fmt(v, f="{:.1f}"):
    return "" if v is None else f.format(v)


def tables(tag: str = "", ref_run: str = "final_s0",
           combined_path: Path = COMBINED) -> None:
    combined = ["# Pure-training timing, every dataset (metrics package item 4)",
                "", "Generated by `scripts/timing_mode.py --tables`. DisCell "
                "arms: N training epochs with no evaluation (`--time-only`), "
                "s/epoch excludes the first (warm-up) epoch; one evaluation "
                "timed on its own; peak = torch max allocated during the "
                "training epochs (resident slide included) and during the "
                "evaluation. Baselines: the `train_s` / `peak_gpu_gb` their "
                "battery recorded for the whole fit.", ""]
    for dataset in DATASETS:
        root = paths.dataset(dataset).root / "experiments"
        rows = []
        for arm in ARMS:
            path = root / f"timing{tag}" / f"timing_{arm}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
        lines = [f"### {SHORT[dataset]} ({dataset})", "",
                 "| arm | epochs | s/epoch | s/evaluation | peak train MiB "
                 "| peak eval MiB | resident MiB | parameters | clean GPU |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            clean = not r["gpu_neighbours_before"] and not r["gpu_neighbours_after"]
            lines.append(
                f"| {r['arm']} | {r['n_epochs']} | {r['s_per_epoch']:.2f} "
                f"| {r['eval_s']:.1f} | {_fmt(r['peak_train_mib'], '{:.0f}')} "
                f"| {_fmt(r['peak_eval_mib'], '{:.0f}')} "
                f"| {_fmt(r['resident_mib'], '{:.0f}')} | {r['n_params']:,} "
                f"| {'yes' if clean else 'no: ' + '; '.join(r['gpu_neighbours_before'] + r['gpu_neighbours_after'])} |")
        if not rows:
            lines.append("| (no timing yet) | | | | | | | | |")
        lines += ["", "Baselines (recorded by the battery, whole fit):", "",
                  "| method | train_s | peak_gpu_gb | epochs |", "|---|---|---|---|"]
        for name, cfg in baselines(dataset, tag):
            epochs = (cfg.get("epochs_completed") or cfg.get("max_epochs")
                      or cfg.get("num_training_epochs") or "")
            lines.append(f"| {name} | {cfg['train_s']:.0f} | "
                         f"{_fmt(cfg.get('peak_gpu_gb'), '{:.2f}')} | {epochs} |")
        minutes = []
        prefix = ref_run.rsplit("_s", 1)[0]
        for s in range(3):
            m = root.parent / "runs" / f"{prefix}_s{s}" / "metrics.json"
            if m.exists():
                d = json.loads(m.read_text())
                minutes.append(f"{prefix}_s{s} {d['minutes']:.1f} min "
                               f"({d['last_epoch'] + 1} epochs, evaluation "
                               f"every 5)")
        if minutes:
            lines += ["", "DisCell's own recorded fits (wall clock incl. "
                      "evaluations and figures): " + "; ".join(minutes) + "."]
        table = "\n".join(lines) + "\n"
        (root / f"timing{tag}.md").write_text(table)
        print(f"wrote {root / f'timing{tag}.md'}")
        combined.append(table)
    combined_path = Path(combined_path)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text("\n".join(combined))
    print(f"wrote {combined_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--ref-run", default="final_s0",
                        help="run whose config.json is timed (default: %(default)s)")
    parser.add_argument("--tag", default="",
                        help="output suffix: experiments/timing<TAG>/, timing<TAG>.md; "
                             "baselines from baseline_battery<TAG>.json")
    parser.add_argument("--combined", default=str(COMBINED))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.tables:
        tables(args.tag, args.ref_run, args.combined)
        return 0
    if not (args.dataset and args.arm):
        parser.error("--dataset and --arm, or --tables")
    time_arm(args.dataset, args.arm, args.epochs, args.ref_run, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
