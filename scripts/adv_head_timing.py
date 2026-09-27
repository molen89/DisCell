#!/usr/bin/env python3
"""Head-step wall time of the adversary-capacity arms (8.17, devlog 2026-09-24 21:20 A).

One ovarian trainer at the control's configuration; per arm its head pairs
and head optimiser are swapped in (built as ``Trainer.__init__`` builds them)
and the trainer's own ``_step`` is timed, with ``_head_steps`` timed inside
it -- CUDA-synchronised on both sides. Arms interleave, the order rotating
per round, so a neighbour's load on the card spreads over all of them.
The fitted runs' own wall times are not comparable across arms: the control
fits are reused from an earlier queue, and every fit shared its card with
other queues.

    uv run python scripts/adv_head_timing.py [--rounds 5 --steps 20]

Writes ``<ovarian>/experiments/adv_head_timing.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from discell import paths
from discell.model.networks import Adversary
from discell.model.prepare import assemble
from discell.model.train import TrainConfig, Trainer

OV = "xenium_prime_ovarian_cancer_ffpe"
CONTROL = "wfix_warmup30_aw0.1_s0"
TMP_RUN = "_adv_head_timing_tmp"
#: arm -> TrainConfig knobs (the queue's flags)
ARMS = {"control": {},
        "steps12": {"adv_steps": 12},
        "width128": {"adv_hidden": 128},
        "ens3": {"adv_ensemble": 3},
        "comp3": {"adv_comp_weight": 3.0},
        "steps12_width128": {"adv_steps": 12, "adv_hidden": 128}}


def swap_heads(trainer: Trainer, config: TrainConfig) -> None:
    members = [Adversary(config.d_z, len(trainer.data.p_t),
                         trainer.data.e_phi.shape[1],
                         hidden=config.adv_hidden).to(trainer.device)
               for _ in range(config.adv_ensemble)]
    trainer.config = config
    trainer.adversary = (members[0] if len(members) == 1
                         else torch.nn.ModuleList(members))
    trainer.adversary_optimiser = torch.optim.Adam(
        trainer.adversary.parameters(), lr=config.adv_lr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--steps", type=int, default=20, help="model steps per arm per round")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default=str(paths.dataset(OV).root / "experiments"
                                        / "adv_head_timing.json"))
    args = p.parse_args(argv)

    runs = paths.dataset(OV).root / "runs"
    raw = json.loads((runs / CONTROL / "config.json").read_text())
    raw.pop("git", None)
    base = TrainConfig(**{**raw, "run_name": TMP_RUN})
    data = assemble(base.dataset, base.variant, base.embeddings,
                    tile_cells=base.tile_cells, phi_pca=base.phi_pca,
                    v_pcs=base.v_pcs, val_fraction=base.val_fraction,
                    seed=base.seed, label_key=base.label_key)
    trainer = Trainer(base, data)
    batches = trainer.train_batches[:args.steps]
    on_gpu = trainer.device.type == "cuda"
    sync = torch.cuda.synchronize if on_gpu else (lambda: None)

    head_ms: list[float] = []
    real_head_steps = trainer._head_steps

    def timed_head_steps(*a, **k):
        sync()
        t0 = time.perf_counter()
        out = real_head_steps(*a, **k)
        sync()
        head_ms.append(1e3 * (time.perf_counter() - t0))
        return out
    trainer._head_steps = timed_head_steps

    configs = {arm: dataclasses.replace(base, **knobs) for arm, knobs in ARMS.items()}
    samples = {arm: {"head_ms": [], "step_ms": []} for arm in ARMS}
    arms = list(ARMS)
    for arm in arms:                               # warm-up: kernels, allocator
        swap_heads(trainer, configs[arm])
        for batch in batches[:args.warmup]:
            trainer._step(batch)
    for r in range(args.rounds):
        for arm in arms[r % len(arms):] + arms[:r % len(arms)]:
            swap_heads(trainer, configs[arm])
            trainer._step(batches[0])               # first step after a swap
            for batch in batches:
                head_ms.clear()
                sync()
                t0 = time.perf_counter()
                trainer._step(batch)
                sync()
                samples[arm]["step_ms"].append(1e3 * (time.perf_counter() - t0))
                samples[arm]["head_ms"].append(head_ms[0])

    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip().splitlines()
    summary = {}
    for arm, s in samples.items():
        h, st = np.array(s["head_ms"]), np.array(s["step_ms"])
        summary[arm] = {"knobs": ARMS[arm], "n": int(len(h)),
                        "head_ms_median": float(np.median(h)),
                        "head_ms_q25": float(np.percentile(h, 25)),
                        "head_ms_q75": float(np.percentile(h, 75)),
                        "step_ms_median": float(np.median(st)),
                        "head_share_of_step": float(np.median(h / st))}
    for arm in summary:
        summary[arm]["head_ms_vs_control"] = (summary[arm]["head_ms_median"]
                                              / summary["control"]["head_ms_median"])
        summary[arm]["step_ms_vs_control"] = (summary[arm]["step_ms_median"]
                                              / summary["control"]["step_ms_median"])
    out = {"what": "wall time of Trainer._head_steps (all head updates of one "
                   "model step) and of the whole _step, ms, CUDA-synchronised; "
                   "medians over rounds x steps; arms interleaved",
           "dataset": OV, "config_from": CONTROL, "tile_cells": base.tile_cells,
           "rounds": args.rounds, "steps_per_round": len(batches),
           "device": (torch.cuda.get_device_name(trainer.device) if on_gpu
                      else "cpu"),
           "gpu_processes_at_end": smi,
           "when": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for arm, s in summary.items():
        print(f"{arm:18s} head {s['head_ms_median']:7.2f} ms "
              f"(x{s['head_ms_vs_control']:.2f})  step {s['step_ms_median']:7.2f} ms "
              f"(x{s['step_ms_vs_control']:.2f})  head share {s['head_share_of_step']:.2f}")
    tmp = runs / TMP_RUN
    if tmp.is_dir() and not any(tmp.iterdir()):
        tmp.rmdir()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
