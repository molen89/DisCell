#!/usr/bin/env python3
"""The read-out for 8.9b, warm-up on both KL terms (devlog 2026-09-24).

Three arms side by side at the pinned alpha_w 0.1 -- the 8.9 control and
w-only warm-up (tag ``wfix``, reused, not refitted) and the new both-KL
warm-up (tag ``wfixb``, arm ``klwarm30``) -- on FF (seeds 0-3) and ovarian
(seeds 0-2), and the 8.9b rule applied mechanically to BOTH remedies:

(i)  0 FF collapses by the I(niche; w) guard: ``w_niche_mi_excess > 0`` on
     every FF seed;
(ii) on both datasets no read is worse than control by more than one control
     seed-sd (arm mean vs control mean, sample sd over the control seeds),
     and at least one read is better by more than one sd.

Reads and the direction that counts as better::

    recon +, nmi +, probe_excess - (probe dCE minus its own noise floor),
    mirror_r2 -, cycle_z +, cycle_w -, mi_ratio (none), w_niche_mi_excess +,
    own_gap_closed + (ovarian only: FF has no transport read)

``mi_ratio`` = I(z;t)/H(t) has no better direction (devlog: "neither end is
the target"), so a move beyond one sd either way counts as worse and it can
never count as better. "At least one better" is read over both datasets
together; the stricter reading (at least one better on EACH dataset) is
reported beside it as ``qualifies_strict``.

    uv run python scripts/wcollapse_b_table.py [--at best]

``--at best`` reads the in-trainer fields at the accepted checkpoint instead
of the last-epoch ``final`` (review R26); the I(niche;w) guard and transport
already come from best.pt. Outputs go to ``*_at_best`` siblings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from wcollapse_table import FF, OV, ROOT, read_run

from discell.experiments.at_best import AT_CHOICES, HEADER, suffixed

REPO = Path(__file__).resolve().parents[1]
DECISION_PATH = REPO / "scripts" / "logs" / "wcollapse_b_2026-09-24" / "DECISION_B.json"
OUT_STEM = ROOT / FF / "experiments" / "wcollapse_b"

#: arm -> run-name template at alpha_w 0.1
ARMS = {"control": "wfix_control_aw0.1_s{s}",
        "warmup30": "wfix_warmup30_aw0.1_s{s}",
        "klwarm30": "wfixb_klwarm30_aw0.1_s{s}"}
REMEDIES = ["warmup30", "klwarm30"]
SEEDS = {FF: [0, 1, 2, 3], OV: [0, 1, 2]}
#: read -> +1 higher is better, -1 lower is better, 0 no direction
READS = {"recon": +1, "nmi": +1, "probe_excess": -1, "mirror_r2": -1,
         "cycle_z": +1, "cycle_w": -1, "mi_ratio": 0,
         "w_niche_mi_excess": +1, "own_gap_closed": +1}
#: reads that exist on only one dataset (FF runs carry no transport read)
ONLY_ON = {"own_gap_closed": OV}
#: shown for context, never part of the rule
INFO = ["best_epoch", "last_epoch", "kl_w", "probe_delta_ce",
        "probe_noise_floor"]


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and v == v


def load_run(dataset: str, run: str, at: str = "final") -> dict | None:
    out = read_run(dataset, run, at)
    if out is None:
        return None
    config = json.loads((ROOT / dataset / "runs" / run / "config.json").read_text())
    out["kl_warmup_epochs"] = config.get("kl_warmup_epochs", 0)
    out["warmup_epochs"] = max(out["w_warmup_epochs"], out["kl_warmup_epochs"])
    out["has_degeneracy"] = (ROOT / dataset / "runs" / run
                             / "degeneracy.json").exists()
    return out


def cell(runs: list[dict]) -> dict:
    """Per-read seed mean / sample sd / min / max, plus the guard counts."""
    out = {"runs": [r["run"] for r in runs], "n_seeds": len(runs),
           "niche_guard_alive": sum(_finite(r.get("w_niche_mi_excess"))
                                    and r["w_niche_mi_excess"] > 0 for r in runs),
           "niche_guard_read": sum(_finite(r.get("w_niche_mi_excess"))
                                   for r in runs),
           "degeneracy_flag_dead": sum(bool(r.get("w_guard_dead")) for r in runs),
           "kl_w_collapsed": sum(r["collapsed"] for r in runs),
           "cycle_w_ok": sum(r["cycle_w_ok"] for r in runs),
           "probe_at_floor": sum(r["probe_at_floor"] for r in runs),
           "best_epoch_within_10_of_warmup_end": sum(
               1 for r in runs if r["warmup_epochs"]
               and r["best_epoch"] < r["warmup_epochs"] + 10),
           "per_seed": {r["run"]: {k: r.get(k) for k in
                                   list(READS) + INFO if k in r}
                        for r in runs}}
    if any("at_best" in r for r in runs):
        out["at_best"] = [r["at_best"] for r in runs]
    for field in list(READS) + INFO:
        vals = [r[field] for r in runs if _finite(r.get(field))]
        if vals:
            out[field] = {"mean": float(np.mean(vals)),
                          "sd": float(np.std(vals, ddof=1)) if len(vals) > 1
                          else 0.0,
                          "min": float(min(vals)), "max": float(max(vals)),
                          "n": len(vals)}
    return out


def collect(at: str = "final") -> dict:
    cells = {}
    for dataset, seeds in SEEDS.items():
        for arm, template in ARMS.items():
            runs = [r for r in (load_run(dataset, template.format(s=s), at)
                                for s in seeds) if r is not None]
            cells[(dataset, arm)] = cell(runs) if runs else None
    return cells


def compare(arm_cell: dict, ctrl: dict, dataset: str, n_expected: int) -> dict:
    """Clause (ii) on one dataset: every read's arm-minus-control shift in
    units of the control seed-sd, and the verdict per read."""
    reads, missing = {}, []
    if arm_cell["n_seeds"] < n_expected:
        missing.append(f"arm has {arm_cell['n_seeds']}/{n_expected} seeds")
    for field, direction in READS.items():
        if ONLY_ON.get(field, dataset) != dataset:
            continue
        a, c = arm_cell.get(field), ctrl.get(field)
        if (a is None or c is None or a["n"] < n_expected
                or c["n"] < n_expected):
            missing.append(field)
            continue
        delta, sd = a["mean"] - c["mean"], c["sd"]
        if direction == 0:
            verdict = "worse" if abs(delta) > sd else "same"
        else:
            signed = direction * delta
            verdict = ("worse" if signed < -sd else
                       "better" if signed > sd else "same")
        reads[field] = {"arm_mean": a["mean"], "control_mean": c["mean"],
                        "control_sd": sd, "delta": delta,
                        "delta_in_sd": (delta / sd) if sd > 0 else None,
                        "direction": {1: "higher", -1: "lower", 0: "none"}[direction],
                        "verdict": verdict}
    return {"reads": reads, "missing": missing,
            "worse": [f for f, r in reads.items() if r["verdict"] == "worse"],
            "better": [f for f, r in reads.items() if r["verdict"] == "better"]}


def decide(cells: dict) -> dict:
    ruling = {}
    for arm in REMEDIES:
        reasons, undecided = [], []
        ff = cells.get((FF, arm))
        # (i) the I(niche; w) guard on every FF seed
        if ff is None or ff["niche_guard_read"] < len(SEEDS[FF]):
            n = 0 if ff is None else ff["niche_guard_read"]
            clause_i = None
            undecided.append(f"FF guard read on {n}/{len(SEEDS[FF])} seeds")
        else:
            collapses = ff["niche_guard_read"] - ff["niche_guard_alive"]
            clause_i = collapses == 0
            reasons.append(f"(i) FF I(niche;w) collapses {collapses}/"
                           f"{len(SEEDS[FF])}")
        # (ii) per dataset against control
        per_dataset, worse, better, better_each = {}, [], [], []
        for dataset in (FF, OV):
            a, c = cells.get((dataset, arm)), cells.get((dataset, "control"))
            if a is None or c is None:
                undecided.append(f"{dataset}: arm or control missing")
                continue
            cmp_ = compare(a, c, dataset, len(SEEDS[dataset]))
            per_dataset[dataset] = cmp_
            if cmp_["missing"]:
                undecided.append(f"{dataset}: missing " + ", ".join(cmp_["missing"]))
            worse += [f"{dataset}:{f}" for f in cmp_["worse"]]
            better += [f"{dataset}:{f}" for f in cmp_["better"]]
            better_each.append(bool(cmp_["better"]))
        # a worse read settles clause (ii) as failed even on a partial grid;
        # otherwise it waits for every read
        if worse:
            clause_ii = False
        elif undecided:
            clause_ii = None
        else:
            clause_ii = bool(better)
        reasons.append("(ii) worse beyond 1 sd: "
                       + (", ".join(worse) if worse else "none")
                       + "; better beyond 1 sd: "
                       + (", ".join(better) if better else "none"))
        if clause_i is False or clause_ii is False:
            qualifies = False
        elif clause_i is None or clause_ii is None:
            qualifies = None
        else:
            qualifies = True
        strict = (None if qualifies is None else
                  bool(qualifies and len(better_each) == 2 and all(better_each)))
        ruling[arm] = {"qualifies": qualifies, "qualifies_strict": strict,
                       "clause_i_ff_zero_collapses": clause_i,
                       "clause_ii_envelope": clause_ii,
                       "worse": worse, "better": better,
                       "reasons": reasons, "undecided": undecided,
                       "per_dataset": per_dataset}
    return ruling


def _fmt(entry: dict | None) -> str:
    if not entry:
        return "--"
    return f"{entry['mean']:.4g} ± {entry['sd']:.2g}"


def render(cells: dict, ruling: dict, at: str = "final") -> str:
    lines = ["# 8.9b: warm-up on both KL terms (tags wfix / wfixb, alpha_w 0.1)",
             "", "Cells: seed mean ± sample sd. `control` and `warmup30` are "
             "the 8.9 runs, reused; `klwarm30` = `--kl-warmup-epochs 30`.",
             "Rule (8.9b, mechanical): (i) I(niche;w) excess > 0 on every FF "
             "seed; (ii) on both datasets no read worse than control by more "
             "than one control seed-sd, at least one better by more than one "
             "sd. mi_ratio has no direction: beyond 1 sd either way = worse.",
             ""]
    if at == "best":
        lines[0] += f" -- {HEADER}"
        fell_back = [run for c in cells.values() if c
                     for run, seed in zip(c["runs"], c["at_best"]) if not seed]
        lines += ["recon, NMI, probe, mirror, cycle, KL_w and I(z;t)/H(t) "
                  "are the history row at the accepted epoch; the I(niche;w) "
                  "guard and transport are from best.pt as before. Last-epoch "
                  "fallbacks: " + (", ".join(fell_back) if fell_back else "none")
                  + ".", ""]
    cols = list(READS) + ["best_epoch", "last_epoch", "kl_w"]
    for dataset in (FF, OV):
        lines += [f"## {dataset}", "",
                  "| arm | n | I(niche;w)>0 | KL_w-collapsed | cycle_w≤0.02 | "
                  "probe at floor | " + " | ".join(cols) + " |",
                  "|---" * (len(cols) + 6) + "|"]
        for arm in ARMS:
            c = cells.get((dataset, arm))
            if c is None:
                lines.append(f"| {arm} | 0 |" + " -- |" * (len(cols) + 4))
                continue
            lines.append(
                f"| {arm} | {c['n_seeds']} | {c['niche_guard_alive']}/"
                f"{c['niche_guard_read']} | {c['kl_w_collapsed']} | "
                f"{c['cycle_w_ok']} | {c['probe_at_floor']} | "
                + " | ".join(_fmt(c.get(f)) for f in cols) + " |")
        lines.append("")
        lines += [f"### {dataset}: shift vs control in control seed-sd", "",
                  "| arm | " + " | ".join(r for r in READS
                                         if ONLY_ON.get(r, dataset) == dataset)
                  + " |",
                  "|---" * (1 + sum(ONLY_ON.get(r, dataset) == dataset
                                    for r in READS)) + "|"]
        for arm in REMEDIES:
            cmp_ = ruling[arm]["per_dataset"].get(dataset)
            row = [arm]
            for field in READS:
                if ONLY_ON.get(field, dataset) != dataset:
                    continue
                r = (cmp_ or {}).get("reads", {}).get(field)
                if r is None:
                    row.append("--")
                    continue
                z = r["delta_in_sd"]
                mark = {"worse": " **worse**", "better": " *better*",
                        "same": ""}[r["verdict"]]
                row.append(("n/a" if z is None else f"{z:+.2f}") + mark)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Verdict", ""]
    for arm in REMEDIES:
        v = ruling[arm]
        state = {True: "QUALIFIES", False: "fails", None: "undecided"}[v["qualifies"]]
        lines.append(f"- **{arm}**: {state} -- " + "; ".join(v["reasons"]))
        if v["qualifies"] is not None:
            lines.append(f"  - strict reading (a better read on each dataset): "
                         f"{'qualifies' if v['qualifies_strict'] else 'fails'}")
        if v["undecided"]:
            lines.append("  - pending: " + "; ".join(v["undecided"]))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(OUT_STEM))
    parser.add_argument("--decision", default=str(DECISION_PATH))
    parser.add_argument("--at", choices=AT_CHOICES, default="final",
                        help="where the in-trainer reads come from: the "
                             "last-epoch `final` (default) or the accepted "
                             "checkpoint (R26); `best` writes *_at_best files")
    args = parser.parse_args(argv)

    cells = collect(args.at)
    ruling = decide(cells)
    text = render(cells, ruling, args.at)
    print(text)

    table = {dataset: {arm: cells.get((dataset, arm)) for arm in ARMS}
             for dataset in (FF, OV)}
    stem = suffixed(args.out, args.at)
    stem.parent.mkdir(parents=True, exist_ok=True)
    read_at = {"read_at": HEADER} if args.at == "best" else {}
    stem.with_suffix(".json").write_text(json.dumps(
        {"rule": "8.9b (devlog 2026-09-24)", "reads": READS,
         "cells": table, "decision": ruling, **read_at}, indent=1,
        default=float))
    stem.with_suffix(".md").write_text(text)

    undecided = [a for a in REMEDIES if ruling[a]["qualifies"] is None]
    decision = {"klwarm30_qualifies": ruling["klwarm30"]["qualifies"],
                "warmup30_qualifies": ruling["warmup30"]["qualifies"],
                "klwarm30_qualifies_strict": ruling["klwarm30"]["qualifies_strict"],
                "warmup30_qualifies_strict": ruling["warmup30"]["qualifies_strict"],
                "complete": not undecided, "undecided": undecided,
                "klwarm30_flags": "--kl-warmup-epochs 30",
                "warmup30_flags": "--w-warmup-epochs 30",
                "table": {arm: {k: ruling[arm][k] for k in
                                ("clause_i_ff_zero_collapses",
                                 "clause_ii_envelope", "worse", "better",
                                 "reasons", "undecided", "per_dataset")}
                          for arm in REMEDIES}, **read_at}
    path = suffixed(args.decision, args.at)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NaN is not valid strict JSON; nothing here should be NaN, so fail loudly
    path.write_text(json.dumps(decision, indent=1, allow_nan=False))
    print(f"wrote {stem.with_suffix('.json')}, {stem.with_suffix('.md')}, {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
