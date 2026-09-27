#!/usr/bin/env python3
"""Read-out and decision for the adversary-capacity ladder (8.17, devlog 2026-09-24 21:20 A).

Ovarian core, final configuration (warm-up 30, alpha_z 0.0035, alpha_w 0.1,
kappa 0.1, d_w 6, type_only, 200/20), seeds 0 1, tag adv:

    control           ``wfix_warmup30_aw0.1_s*`` (6 head steps, width 64),
                      reused: the queue's preflight re-derives its config
    steps12           ``--adv-head-steps 12``
    width128          ``--adv-head-width 128``
    ens3              ``--adv-ensemble 3``
    comp3             ``--adv-comp-weight 3``
    steps12_width128  ``--adv-head-steps 12 --adv-head-width 128``

Per arm, the 2-seed mean [min, max] of: the per-block probe as a fraction
of the 500/40 uncontrolled reference (``uncontrolled500_s{0,1}``; ridge and
MLP, composition and image; ``validation/probe_blocks.json``), NMI,
cycle_z, cycle_w, mirror, recon (in-trainer battery at the accepted
checkpoint, R26), I(niche;w) excess (``degeneracy.json``), transport Read A
own-target gap closed (``transport/transport_distribution.json``), and the
head-step wall time (``experiments/adv_head_timing.json``, one benchmark
over all arms).

**The pre-registered rule**, applied mechanically: "Adopt the cheapest arm
whose MLP composition fraction falls below the control's by more than the
control's seed range on both seeds with every guard inside the control
envelope (one seed-sd) and recon not worse; if several, prefer steps x2 (no
architecture change), then width, then ensemble." Where the rule's words
leave a choice, it was fixed here before any arm had landed:

* *falls below ... on both seeds*: seed-paired (seed s fixes the data split,
  so arm and control seed s grade the same held-out cells):
  ``arm_s < control_s - (max - min of the control's two seeds)`` for s = 0, 1.
* *control envelope (one seed-sd)*: the control's [min, max] widened by one
  sample sd (ddof 1) on each side -- the 8.9 amendment 2(b) envelope --
  and a value better than the envelope in the read's direction passes.
  Guards and directions: NMI up, cycle_z up, mirror R2 down, cycle_w down,
  I(niche;w) excess up, Read A own gap closed up; each compared on the arm's
  2-seed mean. Absolute guards beside them, per seed: cycle_w <= 0.02 and
  no dead context channel.
* *recon not worse*: the arm's mean recon is not below that envelope.
* *cheapest ... prefer*: among qualifying arms the named order steps12,
  width128, ens3 decides; an arm the order does not name (comp3,
  steps12_width128) comes after them, cheaper head-step time first.

The image blocks and the ridge grader are reported, not gated (the rule
names the MLP composition fraction only). A sensitivity line re-applies the
same rule with the fresh reference pair ``aw_ref0.1_s{0,1}`` (the same
configuration, refitted 2026-09-24 for the alpha_w ladder) as the control;
it decides nothing.

    uv run python scripts/adv_table.py [--at best]

Writes ``<ovarian>/experiments/adv_ladder.{json,md}`` and
``scripts/logs/adv_ladder_2026-09-24/DECISION_ADV.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from discell.experiments.at_best import AT_CHOICES, HEADER, battery_at_best

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "datasets"
OV = "xenium_prime_ovarian_cancer_ffpe"
LOGS = REPO / "scripts" / "logs" / "adv_ladder_2026-09-24"
SEEDS = [0, 1]
ARMS = {"control": "wfix_warmup30_aw0.1_s{s}",
        "steps12": "adv_steps12_s{s}",
        "width128": "adv_width128_s{s}",
        "ens3": "adv_ens3_s{s}",
        "comp3": "adv_comp3_s{s}",
        "steps12_width128": "adv_steps12_width128_s{s}"}
ARM_FLAGS = {"control": "(reused wfix_warmup30_aw0.1_s*: 6 steps, width 64)",
             "steps12": "--adv-head-steps 12",
             "width128": "--adv-head-width 128",
             "ens3": "--adv-ensemble 3",
             "comp3": "--adv-comp-weight 3",
             "steps12_width128": "--adv-head-steps 12 --adv-head-width 128"}
SENSITIVITY_CONTROL = "aw_ref0.1_s{s}"
PREFERENCE = ["steps12", "width128", "ens3"]
DECISION_READ = "mlp_comp_u"
#: guard -> True if higher is better, False if lower is better
GUARDS = {"nmi": True, "cycle_z": True, "mirror_r2": False, "cycle_w": False,
          "w_niche_mi_excess": True, "read_a_own_gap_closed": True}
CYCLE_W_GUARD = 0.02

#: (key, label, format) in table order
READS = [
    ("mlp_comp_u", "**MLP composition, fraction of uncontrolled** (decides)", "{:.3f}"),
    ("ridge_comp_u", "ridge composition, fraction of uncontrolled", "{:.3f}"),
    ("mlp_img_u", "MLP image, fraction of uncontrolled", "{:.3f}"),
    ("ridge_img_u", "ridge image, fraction of uncontrolled", "{:.3f}"),
    ("mlp_comp_excess", "MLP composition excess (nats/column)", "{:+.4f}"),
    ("ridge_comp_excess", "ridge composition excess (nats/column)", "{:+.4f}"),
    ("nmi", "NMI(z; t)", "{:.3f}"),
    ("cycle_z", "cycle_z R² (pooled)", "{:.3f}"),
    ("cycle_w", "cycle_w R² (pooled)", "{:.4f}"),
    ("mirror_r2", "mirror R²", "{:.3f}"),
    ("recon", "recon (held-out, nats/count)", "{:.4f}"),
    ("w_niche_mi_excess", "I(niche; w) excess", "{:+.4f}"),
    ("read_a_own_gap_closed", "Read A own: median gap closed", "{:.3f}"),
    ("best_epoch", "accepted epoch", "{:.0f}"),
]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _dig(d, *keys, default=float("nan")):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return default if d is None else d


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def read_run(run: str, at: str = "best") -> dict | None:
    d = ROOT / OV / "runs" / run
    metrics = _load(d / "metrics.json")
    if metrics is None:
        return None
    config = _load(d / "config.json") or {}
    battery = battery_at_best(d) if at == "best" else metrics["final"]
    out = {"run": run,
           "knobs": {k: config.get(k) for k in ("adv_steps", "adv_hidden",
                                                "adv_ensemble", "adv_comp_weight")},
           "best_epoch": _dig(metrics, "best", "epoch"),
           "dead_w_channel": bool(metrics.get("dead_w_channel")),
           "minutes": metrics.get("minutes"),
           "recon": battery["recon_val"], "nmi": battery["nmi"],
           "mirror_r2": _dig(battery, "mirror", "r2"),
           "cycle_z": _dig(battery, "cycle", "z", "r2_pooled"),
           "cycle_w": _dig(battery, "cycle", "w", "r2_pooled")}
    if at == "best":
        out["at_best"] = battery["at_best"]
    out["cycle_w_ok"] = bool(_finite(out["cycle_w"])
                             and out["cycle_w"] <= CYCLE_W_GUARD)
    probe = _load(d / "validation" / "probe_blocks.json")
    if probe:
        for family in ("ridge", "mlp"):
            for block in ("comp", "img"):
                entry = probe[family][block]
                out[f"{family}_{block}_u"] = _dig(entry, "fraction_of_uncontrolled")
                out[f"{family}_{block}_excess"] = _dig(entry, "excess")
        out["probe_reference"] = _dig(probe, "reference", "uncontrolled", default=None)
        out["invariance_pass"] = probe.get("invariance_pass")
        out["probe_legacy_check"] = (
            probe["ridge"]["legacy"]["delta_ce"]
            - probe["legacy_in_trainer"]["delta_ce"]
            if "delta_ce" in (probe.get("legacy_in_trainer") or {}) else None)
    guard = _load(d / "degeneracy.json") or {}
    out["w_niche_mi_excess"] = _dig(guard, "w_channel", "w_niche_mi_excess")
    dist = _load(d / "transport" / "transport_distribution.json")
    if dist:
        out["read_a_own_gap_closed"] = _dig(dist, "summary_model_own",
                                            "pairwise", "median_gap_closed")
    return out


def summarise(runs: list[dict]) -> dict:
    out = {"runs": [r["run"] for r in runs], "n_seeds": len(runs),
           "dead_w_channel": sum(r["dead_w_channel"] for r in runs),
           "cycle_w_ok": sum(r["cycle_w_ok"] for r in runs),
           "invariance_pass": sum(bool(r.get("invariance_pass")) for r in runs),
           "per_seed": {r["run"]: r for r in runs}}
    for key, _, _ in READS:
        vals = [r[key] for r in runs if _finite(r.get(key))]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                        "max": float(max(vals)), "n": len(vals),
                        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "by_seed": [r.get(key) for r in runs]}
    return out


def collect(at: str, arms: dict) -> dict:
    cells = {}
    for arm, template in arms.items():
        runs = [r for r in (read_run(template.format(s=s), at) for s in SEEDS)
                if r is not None]
        cells[arm] = summarise(runs) if runs else None
    return cells


def timing() -> dict:
    t = _load(ROOT / OV / "experiments" / "adv_head_timing.json")
    return (t or {}).get("arms", {})


def inside_envelope(value: float, control: dict, key: str,
                    higher_is_better: bool) -> bool:
    entry = control[key]
    lo, hi = entry["min"] - entry["sd"], entry["max"] + entry["sd"]
    if lo <= value <= hi:
        return True
    return value > hi if higher_is_better else value < lo


def judge(arm: dict | None, control: dict | None) -> dict:
    """The rule's clauses for one arm; None = not yet decidable."""
    clauses, notes = {}, []
    complete = (arm is not None and control is not None
                and arm["n_seeds"] == len(SEEDS) == control["n_seeds"])
    # (1) MLP composition, seed-paired, beyond the control's seed range
    ctl = (control or {}).get(DECISION_READ)
    got = (arm or {}).get(DECISION_READ)
    if not (complete and ctl and got and ctl["n"] == got["n"] == len(SEEDS)):
        clauses["probe_falls"] = None
        notes.append("MLP composition: seeds missing")
    else:
        span = ctl["max"] - ctl["min"]
        margins = [c - a for a, c in zip(got["by_seed"], ctl["by_seed"])]
        clauses["probe_falls"] = all(m > span for m in margins)
        clauses["probe_margins"] = margins
        clauses["probe_control_range"] = span
        notes.append("MLP comp below control by "
                     + ", ".join(f"{m:+.3f}" for m in margins)
                     + f" (needs > {span:.3f} on both seeds)")
    # (2) guards inside the widened envelope, or better
    bad, missing = [], []
    for key, hib in GUARDS.items():
        if not complete or key not in arm or key not in control \
                or arm[key]["n"] < len(SEEDS):
            missing.append(key)
        elif not inside_envelope(arm[key]["mean"], control, key, hib):
            bad.append(key)
    clauses["guards_inside_envelope"] = (None if missing else not bad)
    clauses["guards_outside"] = bad
    notes.append("guards: " + ("missing " + ", ".join(missing) if missing
                               else "outside on " + ", ".join(bad) if bad
                               else "all inside the control envelope"))
    # (3) absolute per-seed guards
    if complete:
        clauses["absolute_guards"] = (arm["cycle_w_ok"] == arm["n_seeds"]
                                      and arm["dead_w_channel"] == 0)
        notes.append(f"cycle_w <= {CYCLE_W_GUARD} on {arm['cycle_w_ok']}/"
                     f"{arm['n_seeds']}, dead channels {arm['dead_w_channel']}")
    else:
        clauses["absolute_guards"] = None
    # (4) recon not worse: not below the widened envelope
    if complete and "recon" in arm and "recon" in control:
        clauses["recon_not_worse"] = inside_envelope(arm["recon"]["mean"],
                                                     control, "recon", True)
        notes.append(f"recon {arm['recon']['mean'] - control['recon']['mean']:+.4f} "
                     "vs control mean")
    else:
        clauses["recon_not_worse"] = None
    keys = ("probe_falls", "guards_inside_envelope", "absolute_guards",
            "recon_not_worse")
    values = [clauses[k] for k in keys]
    qualifies = None if any(v is None for v in values) else all(values)
    return {"qualifies": qualifies, "clauses": clauses, "notes": notes}


def decide(cells: dict, control_key: str = "control") -> dict:
    control = cells.get(control_key)
    heads = timing()
    arms = [a for a in ARMS if a != "control"]
    verdicts = {a: judge(cells.get(a), control) for a in arms}
    qualifying = [a for a in arms if verdicts[a]["qualifies"]]
    rest = sorted((a for a in qualifying if a not in PREFERENCE),
                  key=lambda a: heads.get(a, {}).get("head_ms_median", float("inf")))
    order = [a for a in PREFERENCE if a in qualifying] + rest
    pending = [a for a in arms if verdicts[a]["qualifies"] is None]
    adopted = order[0] if order else None
    status = "incomplete" if pending else "complete"
    if adopted:
        reading = (f"adopt {adopted} ({ARM_FLAGS[adopted]}): its MLP composition "
                   "fraction falls below the control's by more than the "
                   "control's seed range on both seeds, every guard inside "
                   "the control envelope, recon not worse")
    elif pending:
        reading = "no arm qualifies yet; pending: " + ", ".join(pending)
    else:
        reading = ("no arm qualifies: the adversary stays at 6 steps / width "
                   "64 / one pair / weight 1, and the residual is reported "
                   "as the ceiling with the planted-world evidence")
    return {"adopted": adopted, "status": status, "qualifying": order,
            "pending": pending, "reading": reading, "arms": verdicts}


def _fmt(entry: dict | None, fmt: str) -> str:
    if not entry:
        return "--"
    text = (f"{fmt.format(entry['mean'])} [{fmt.format(entry['min'])}, "
            f"{fmt.format(entry['max'])}]")
    return text + ("" if entry["n"] == len(SEEDS) else f" (n={entry['n']})")


def render(cells: dict, decision: dict, sensitivity: dict, at: str) -> str:
    heads = timing()
    arms = list(ARMS)
    lines = ["# Adversary-capacity ladder (8.17): ovarian, final configuration, "
             "κ 0.1, 200/20, seeds 0 1" + (f" -- {HEADER}" if at == "best" else ""),
             "",
             "Cells: 2-seed mean [min, max]. Probe fractions: each block's "
             "excess over its within-type floor as a fraction of the 500/40 "
             "uncontrolled fits' mean (`uncontrolled500_s0`, `_s1`). Control = "
             "`wfix_warmup30_aw0.1_s*` (reused, config re-derived by the "
             "queue's preflight). Rule and its fixed readings: "
             "`scripts/adv_table.py` docstring.", "",
             "| arm | flags |", "|---|---|"]
    lines += [f"| {a} | `{ARM_FLAGS[a]}` |" for a in arms]
    lines += ["", "| read | " + " | ".join(arms) + " |",
              "|---" * (len(arms) + 1) + "|"]
    for key, label, fmt in READS:
        lines.append(f"| {label} | " + " | ".join(
            _fmt((cells[a] or {}).get(key), fmt) for a in arms) + " |")
    lines.append("| head-step wall time, ms per model step (× control) | " + " | ".join(
        f"{heads[a]['head_ms_median']:.1f} (×{heads[a]['head_ms_vs_control']:.2f})"
        if a in heads else "--" for a in arms) + " |")
    lines.append("| whole step, ms (× control) | " + " | ".join(
        f"{heads[a]['step_ms_median']:.1f} (×{heads[a]['step_ms_vs_control']:.2f})"
        if a in heads else "--" for a in arms) + " |")
    for key, label in (("n_seeds", "seeds landed"),
                       ("dead_w_channel", "dead KL_w channel (trainer)"),
                       ("cycle_w_ok", f"cycle_w ≤ {CYCLE_W_GUARD}"),
                       ("invariance_pass", "invariance_pass (≤ 0.25 u, all blocks)")):
        lines.append(f"| {label} | " + " | ".join(
            "--" if cells[a] is None else str(cells[a][key]) for a in arms) + " |")
    lines += ["", "## Per seed", ""]
    per = ["mlp_comp_u", "ridge_comp_u", "mlp_img_u", "ridge_img_u", "nmi",
           "cycle_z", "cycle_w", "mirror_r2", "recon", "w_niche_mi_excess",
           "read_a_own_gap_closed"]
    lines += ["| run | best epoch | " + " | ".join(per) + " | legacy ΔCE check |",
              "|---" * (len(per) + 3) + "|"]
    for arm in arms:
        for run, r in ((cells[arm] or {}).get("per_seed") or {}).items():
            chk = r.get("probe_legacy_check")
            lines.append(f"| {run} | {r['best_epoch']} | " + " | ".join(
                "--" if not _finite(r.get(k)) else f"{r[k]:.4g}" for k in per)
                + f" | {'--' if chk is None else f'{chk:+.1e}'} |")
    lines += ["", "## The pre-registered rule", "",
              f"**{decision['reading']}** (status: {decision['status']}).", "",
              "| arm | qualifies | MLP comp falls | guards in envelope | absolute guards | recon not worse | notes |",
              "|---|---|---|---|---|---|---|"]
    show = lambda v: "--" if v is None else ("yes" if v else "**no**")
    for arm, v in decision["arms"].items():
        c = v["clauses"]
        lines.append(f"| {arm} | {show(v['qualifies'])} | {show(c['probe_falls'])} | "
                     f"{show(c['guards_inside_envelope'])} | {show(c['absolute_guards'])} | "
                     f"{show(c['recon_not_worse'])} | {'; '.join(v['notes'])} |")
    lines += ["", "Sensitivity (decides nothing): the same rule with the fresh "
              "reference pair `aw_ref0.1_s*` as the control -- "
              f"{sensitivity['reading']} (status: {sensitivity['status']}; "
              "qualifying: " + (", ".join(sensitivity["qualifying"]) or "none") + ")."]
    fell_back = [run for c in cells.values() if c for run, r in c["per_seed"].items()
                 if r.get("at_best") is False]
    missing = [ARMS[a].format(s=s) for a in arms for s in SEEDS
               if ARMS[a].format(s=s) not in ((cells[a] or {}).get("per_seed") or {})]
    lines += ["", "Fallbacks to the last epoch: " + (", ".join(fell_back) or "none")
              + ". Missing fits: " + (", ".join(missing) or "none") + "."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--at", choices=AT_CHOICES, default="best")
    parser.add_argument("--out", default=str(ROOT / OV / "experiments" / "adv_ladder"))
    args = parser.parse_args(argv)
    cells = collect(args.at, ARMS)
    decision = decide(cells)
    alt = dict(cells)
    alt["control"] = collect(args.at, {"control": SENSITIVITY_CONTROL})["control"]
    sensitivity = decide(alt)
    text = render(cells, decision, sensitivity, args.at)
    print(text)
    stem = Path(args.out)
    if args.at != "best":
        stem = stem.with_name(f"{stem.name}_at_{args.at}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = {"test": "8.17 adversary-capacity ladder (devlog 2026-09-24 21:20 A)",
               "dataset": OV, "seeds": SEEDS, "arms": ARM_FLAGS,
               "read_at": HEADER if args.at == "best" else "final",
               "cells": cells, "head_timing": timing(), "decision": decision,
               "sensitivity_fresh_control": sensitivity}
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=1, default=float))
    stem.with_suffix(".md").write_text(text)
    if args.at == "best":
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "DECISION_ADV.json").write_text(json.dumps(
            {"written": time.strftime("%Y-%m-%d %H:%M:%S"),
             "rule": ("adopt the cheapest arm whose MLP composition fraction "
                      "falls below the control's by more than the control's "
                      "seed range on both seeds with every guard inside the "
                      "control envelope (one seed-sd) and recon not worse; if "
                      "several, prefer steps x2, then width, then ensemble"),
             "readings_fixed_before_data": {
                 "falls_below": "seed-paired: arm_s < control_s - control range, s = 0, 1",
                 "envelope": "control [min, max] widened by one sample sd each "
                             "side; better than the envelope in the read's "
                             "direction passes; arm 2-seed mean",
                 "guards": {k: ("higher" if v else "lower") + " is better"
                            for k, v in GUARDS.items()},
                 "absolute": f"cycle_w <= {CYCLE_W_GUARD} and no dead channel, per seed",
                 "recon": "arm mean not below the widened control envelope",
                 "preference": PREFERENCE + ["then unnamed arms by head-step time"]},
             "control": [ARMS["control"].format(s=s) for s in SEEDS],
             "adopted": decision["adopted"], "status": decision["status"],
             "reading": decision["reading"], "qualifying": decision["qualifying"],
             "arms": decision["arms"],
             "sensitivity_fresh_control": {
                 "control": [SENSITIVITY_CONTROL.format(s=s) for s in SEEDS],
                 "adopted": sensitivity["adopted"],
                 "qualifying": sensitivity["qualifying"],
                 "reading": sensitivity["reading"]}},
            indent=1, default=float))
    print(f"wrote {stem.with_suffix('.json')}, {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
