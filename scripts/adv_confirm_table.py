#!/usr/bin/env python3
"""Read-out and decision for the composition-weight confirmation (8.17 follow-up, tag advc).

Devlog 2026-09-25 morning, 8.17: "Follow-up launched: comp3 and comp5 on a
third ovarian seed and on GSE and FF (2 seeds each) vs their controls; adopt
comp3 if it holds on the primaries with guards intact." Final configuration
(warm-up 30, kappa 0.1, d_w 6, type_only, alpha_w 0.1, alpha_a 0.3, 200/20,
each dataset's alpha_z), the only change ``--adv-comp-weight c``:

    GSE core  control  aw_ref0.1_s{0,1}                          (reused)
              comp3    advc_comp3_s{0,1}   comp5  advc_comp5_s{0,1}
    FF        control  wfix_warmup30_aw0.1_s{0,1}                (reused; the
                       FF reference fits, runs/aw_ref0.1_s* link to them)
              comp3    advc_comp3_s{0,1}
    ovarian   control  wfix_warmup30_aw0.1_s{0,1,2} + aw_ref0.1_s{0,1}
                       (reused; same configuration, pooled: 5 fits)
              comp3    adv_comp3_s{0,1,2}  (s0 s1 from the 8.17 ladder)
              comp5    adv_comp5_s{0,1,2}

Per dataset x arm, the seed mean [min, max] of: the per-block probe as a
fraction of that dataset's 500/40 uncontrolled reference
(``uncontrolled500_s*``; ridge and MLP, composition and image), NMI, cycle_z,
cycle_w, mirror, recon (in-trainer battery at the accepted checkpoint, R26),
I(niche;w) excess, transport Read A own-target gap closed.

**The pre-stated rule** (brief, 2026-09-25): comp3 is confirmed if on GSE and
FF (the primaries) and ovarian (a) its MLP composition fraction is below the
control's mean on every seed, (b) its ridge composition fraction is below the
control's by more than the control's seed-sd on the mean, and (c) NMI,
cycle_z and recon sit inside the control [min, max] widened by one sd; comp5
is reported for dose. Readings fixed here before any advc fit landed:

* (a) every arm seed ``s``: ``arm_s < mean(control)``; strict.
* (b) ``mean(arm) < mean(control) - sd(control)``, sample sd (ddof 1) over the
  control's fits.
* (c) the arm's seed mean against ``[min - sd, max + sd]`` of the control;
  all three reads are higher-is-better (recon is a held-out log-likelihood
  per count), and a value above the envelope passes (the 8.9 amendment 2(b)
  and ``adv_table.py`` reading); the table says when that happens.
* ovarian control: the five fits of the one configuration pooled (the reused
  warm-up triple and the fresh pair, which the ladder's preflight showed
  identical); the rule re-applied against each subset alone is reported
  as sensitivity and decides nothing.
* cycle_w, mirror, I(niche;w) excess, Read A own gap, the image blocks,
  cycle_w <= 0.02 and dead channels are reported, not gated.
* A dataset's verdict is None until every listed fit has landed; comp3 is
  confirmed only if all three datasets pass, not confirmed as soon as one
  complete dataset fails.

    uv run python scripts/adv_confirm_table.py [--at best]

Writes ``<ovarian>/experiments/adv_confirm.{json,md}`` and (with ``--at
best``) ``scripts/logs/adv_confirm_2026-09-25/DECISION_ADVC.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adv_table import READS, _dig, _finite, _load, summarise  # noqa: E402

from discell.experiments.at_best import AT_CHOICES, HEADER, battery_at_best  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "datasets"
GS = "gse315411_pdltma06_11_prime_solo"
OV = "xenium_prime_ovarian_cancer_ffpe"
FF = "xenium_prime_human_ovary_ff"
LOGS = REPO / "scripts" / "logs" / "adv_confirm_2026-09-25"
LABEL = {GS: "GSE core", FF: "FF", OV: "ovarian"}
#: the datasets the confirmation needs, in decision order (primaries first)
DECIDING = [GS, FF, OV]
_WFIX = [f"wfix_warmup30_aw0.1_s{s}" for s in (0, 1, 2)]
_FRESH = [f"aw_ref0.1_s{s}" for s in (0, 1)]
ARMS = {
    GS: {"control": _FRESH,
         "comp3": ["advc_comp3_s0", "advc_comp3_s1"],
         "comp5": ["advc_comp5_s0", "advc_comp5_s1"]},
    FF: {"control": _WFIX[:2],
         "comp3": ["advc_comp3_s0", "advc_comp3_s1"]},
    OV: {"control": _WFIX + _FRESH,
         "comp3": [f"adv_comp3_s{s}" for s in (0, 1, 2)],
         "comp5": [f"adv_comp5_s{s}" for s in (0, 1, 2)]},
}
#: ovarian control subsets, re-applied as sensitivity (decide nothing)
SENSITIVITY = {"warm-up triple": _WFIX, "fresh pair": _FRESH}
ARM_FLAGS = {"control": "(reused; --adv-comp-weight 1)",
             "comp3": "--adv-comp-weight 3", "comp5": "--adv-comp-weight 5"}
#: guard clause (c): all higher-is-better
GUARDS = ("nmi", "cycle_z", "recon")
CYCLE_W_GUARD = 0.02
RULE = ("comp3 is confirmed if on GSE and FF (the primaries) and ovarian its "
        "MLP composition fraction is below the control's mean on every seed, "
        "ridge composition below by more than the control's seed-sd on the "
        "mean, and NMI, cycle_z, recon inside the control [min, max] widened "
        "by one sd; comp5 reported for dose")


def read_run(dataset: str, run: str, at: str = "best") -> dict | None:
    d = ROOT / dataset / "runs" / run
    metrics = _load(d / "metrics.json")
    if metrics is None:
        return None
    config = _load(d / "config.json") or {}
    battery = battery_at_best(d) if at == "best" else metrics["final"]
    out = {"run": run, "resolved": d.resolve().name,
           "adv_comp_weight": config.get("adv_comp_weight", 1.0),
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
    guard = _load(d / "degeneracy.json") or {}
    out["w_niche_mi_excess"] = _dig(guard, "w_channel", "w_niche_mi_excess")
    dist = _load(d / "transport" / "transport_distribution.json")
    if dist:
        out["read_a_own_gap_closed"] = _dig(dist, "summary_model_own",
                                            "pairwise", "median_gap_closed")
    return out


def cell(dataset: str, runs: list[str], at: str) -> dict | None:
    got = [r for r in (read_run(dataset, run, at) for run in runs) if r is not None]
    if not got:
        return None
    out = summarise(got)
    out["expected"] = len(runs)
    out["missing"] = [run for run in runs if run not in out["per_seed"]]
    return out


def collect(at: str) -> dict:
    return {ds: {arm: cell(ds, runs, at) for arm, runs in arms.items()}
            for ds, arms in ARMS.items()}


def envelope(control: dict, key: str) -> tuple[float, float]:
    entry = control[key]
    return entry["min"] - entry["sd"], entry["max"] + entry["sd"]


def judge(arm: dict | None, control: dict | None) -> dict:
    """The rule's clauses for one arm on one dataset; None = not decidable."""
    clauses, notes = {}, []
    complete = (arm is not None and control is not None
                and not arm["missing"] and not control["missing"])
    have = lambda c, k: (c is not None and k in c
                         and c[k]["n"] == c["expected"])
    # (a) MLP composition below the control's mean on every seed
    if complete and have(arm, "mlp_comp_u") and have(control, "mlp_comp_u"):
        mean = control["mlp_comp_u"]["mean"]
        margins = [mean - v for v in arm["mlp_comp_u"]["by_seed"]]
        clauses["mlp_below_mean"] = all(m > 0 for m in margins)
        clauses["mlp_margins"] = margins
        notes.append("MLP comp below control mean " + f"{mean:.3f} by "
                     + ", ".join(f"{m:+.3f}" for m in margins))
    else:
        clauses["mlp_below_mean"] = None
        notes.append("MLP comp: fits or reads missing")
    # (b) ridge composition: mean below by more than the control's sd
    if complete and have(arm, "ridge_comp_u") and have(control, "ridge_comp_u"):
        ctl, got = control["ridge_comp_u"], arm["ridge_comp_u"]
        margin = ctl["mean"] - got["mean"]
        clauses["ridge_below_sd"] = margin > ctl["sd"]
        clauses["ridge_margin"] = margin
        clauses["ridge_control_sd"] = ctl["sd"]
        notes.append(f"ridge comp mean below by {margin:+.3f} "
                     f"(needs > {ctl['sd']:.3f})")
    else:
        clauses["ridge_below_sd"] = None
        notes.append("ridge comp: fits or reads missing")
    # (c) NMI, cycle_z, recon inside the widened control envelope (or above)
    outside, above, missing = [], [], []
    for key in GUARDS:
        if not (complete and have(arm, key) and have(control, key)):
            missing.append(key)
            continue
        lo, hi = envelope(control, key)
        value = arm[key]["mean"]
        if value < lo:
            outside.append(f"{key} {value:.4f} < {lo:.4f}")
        elif value > hi:
            above.append(key)
    clauses["guards_inside"] = None if missing else not outside
    clauses["guards_outside"] = outside
    clauses["guards_above_envelope"] = above
    notes.append("guards: " + ("missing " + ", ".join(missing) if missing
                               else "below envelope on " + "; ".join(outside)
                               if outside else "NMI, cycle_z, recon inside")
                 + (" (above the envelope, passes: " + ", ".join(above) + ")"
                    if above else ""))
    values = [clauses[k] for k in ("mlp_below_mean", "ridge_below_sd",
                                   "guards_inside")]
    passes = (False if any(v is False for v in values)
              else None if any(v is None for v in values) else True)
    return {"passes": passes, "clauses": clauses, "notes": notes}


def combine(verdicts: dict) -> bool | None:
    values = list(verdicts.values())
    if any(v["passes"] is False for v in values):
        return False
    return None if any(v["passes"] is None for v in values) else True


def decide(cells: dict) -> dict:
    comp3 = {ds: judge(cells[ds].get("comp3"), cells[ds]["control"])
             for ds in DECIDING}
    comp5 = {ds: judge(cells[ds].get("comp5"), cells[ds]["control"])
             for ds in DECIDING if "comp5" in ARMS[ds]}
    confirmed = combine(comp3)
    pending = [LABEL[ds] for ds, v in comp3.items() if v["passes"] is None]
    failed = [LABEL[ds] for ds, v in comp3.items() if v["passes"] is False]
    if confirmed:
        reading = ("comp3 confirmed (--adv-comp-weight 3): on GSE, FF and "
                   "ovarian its MLP composition fraction is below the "
                   "control's mean on every seed, ridge composition below by "
                   "more than the control's seed-sd, NMI / cycle_z / recon "
                   "inside the widened control envelope")
    elif confirmed is False:
        reading = ("comp3 NOT confirmed: the rule fails on "
                   + ", ".join(failed)
                   + (f" (still pending: {', '.join(pending)})" if pending else ""))
    else:
        reading = "comp3 not yet decidable; pending: " + ", ".join(pending)
    dose = combine(comp5)
    return {"confirmed": confirmed,
            "status": "incomplete" if pending else "complete",
            "reading": reading, "comp3": comp3,
            "comp5_dose": {"passes_same_rule": dose, "datasets": comp5}}


def sensitivity(at: str, cells: dict) -> dict:
    out = {}
    for name, runs in SENSITIVITY.items():
        control = cell(OV, runs, at)
        out[name] = {"control": runs, "cell": control,
                     **{arm: judge(cells[OV].get(arm), control)
                        for arm in ("comp3", "comp5")}}
    return out


def _fmt(entry: dict | None, fmt: str, expected: int) -> str:
    if not entry:
        return "--"
    text = (f"{fmt.format(entry['mean'])} [{fmt.format(entry['min'])}, "
            f"{fmt.format(entry['max'])}]")
    return text + ("" if entry["n"] == expected else f" (n={entry['n']})")


def _show(v) -> str:
    return "--" if v is None else ("yes" if v else "**no**")


def render(cells: dict, decision: dict, sens: dict, at: str) -> str:
    lines = ["# Composition-weight confirmation (8.17 follow-up, tag advc)"
             + (f" -- {HEADER}" if at == "best" else ""), "",
             "Final configuration, κ 0.1, 200/20; the arms differ from their "
             "control only in `--adv-comp-weight`. Cells: seed mean [min, max]. "
             "Probe fractions: each block's excess over its within-type floor "
             "as a fraction of that dataset's 500/40 uncontrolled fits "
             "(`uncontrolled500_s*`). Rule and its fixed readings: "
             "`scripts/adv_confirm_table.py` docstring.", "",
             f"**{decision['reading']}** (status: {decision['status']}).", ""]
    for ds in DECIDING:
        arms = list(ARMS[ds])
        lines += [f"## {LABEL[ds]} (`{ds}`)", "",
                  "| arm | fits |", "|---|---|"]
        lines += [f"| {a} `{ARM_FLAGS[a]}` | " + ", ".join(
            f"`{r}`" for r in ARMS[ds][a]) + " |" for a in arms]
        lines += ["", "| read | " + " | ".join(arms) + " |",
                  "|---" * (len(arms) + 1) + "|"]
        for key, label, fmt in READS:
            lines.append(f"| {label.replace(' (decides)', '')} | " + " | ".join(
                _fmt((cells[ds][a] or {}).get(key), fmt, len(ARMS[ds][a]))
                for a in arms) + " |")
        for key, label in (("n_seeds", "fits landed"),
                           ("dead_w_channel", "dead KL_w channel (trainer)"),
                           ("cycle_w_ok", f"cycle_w ≤ {CYCLE_W_GUARD}"),
                           ("invariance_pass", "invariance_pass (≤ 0.25 u, all blocks)")):
            lines.append(f"| {label} | " + " | ".join(
                "--" if cells[ds][a] is None
                else f"{cells[ds][a][key]}/{len(ARMS[ds][a])}"
                for a in arms) + " |")
        lines += ["", "| arm | passes | (a) MLP comp < control mean, every seed | "
                  "(b) ridge comp < mean − sd | (c) NMI, cycle_z, recon in envelope | notes |",
                  "|---|---|---|---|---|---|"]
        for arm, v in (("comp3", decision["comp3"].get(ds)),
                       ("comp5 (dose)", decision["comp5_dose"]["datasets"].get(ds))):
            if v is None:
                continue
            c = v["clauses"]
            lines.append(f"| {arm} | {_show(v['passes'])} | {_show(c['mlp_below_mean'])} | "
                         f"{_show(c['ridge_below_sd'])} | {_show(c['guards_inside'])} | "
                         f"{'; '.join(v['notes'])} |")
        lines.append("")
    dose = decision["comp5_dose"]["passes_same_rule"]
    lines += ["## Dose", "",
              "comp5 under the same rule on GSE and ovarian (decides nothing): "
              + ("passes on both" if dose else "--" if dose is None
                 else "fails on " + ", ".join(
                     LABEL[ds] for ds, v in decision["comp5_dose"]["datasets"].items()
                     if v["passes"] is False)) + ".", ""]
    for ds in (GS, OV):
        c3, c5 = (cells[ds].get("comp3") or {}), (cells[ds].get("comp5") or {})
        ctl = cells[ds]["control"] or {}
        parts = []
        for key in ("mlp_comp_u", "ridge_comp_u", "nmi", "cycle_z", "recon"):
            vals = [x.get(key, {}).get("mean") if x else None for x in (ctl, c3, c5)]
            parts.append(f"{key} " + " → ".join("--" if v is None else f"{v:.3f}"
                                              for v in vals))
        lines.append(f"* {LABEL[ds]} (weight 1 → 3 → 5): " + "; ".join(parts))
    lines += ["", "## Ovarian sensitivity: each control subset alone (decides nothing)", "",
              "| control | fits | comp3 passes | comp5 passes | comp3 notes |",
              "|---|---|---|---|---|"]
    for name, s in sens.items():
        lines.append(f"| {name} | {', '.join(s['control'])} | {_show(s['comp3']['passes'])} | "
                     f"{_show(s['comp5']['passes'])} | {'; '.join(s['comp3']['notes'])} |")
    per = ["mlp_comp_u", "ridge_comp_u", "mlp_img_u", "ridge_img_u", "nmi",
           "cycle_z", "cycle_w", "mirror_r2", "recon", "w_niche_mi_excess",
           "read_a_own_gap_closed"]
    lines += ["", "## Per fit", "",
              "| dataset | arm | run | best epoch | " + " | ".join(per) + " |",
              "|---" * (len(per) + 4) + "|"]
    fell_back, missing = [], []
    for ds in DECIDING:
        for arm in ARMS[ds]:
            c = cells[ds][arm]
            missing += [f"{LABEL[ds]}/{r}" for r in (ARMS[ds][arm] if c is None
                                                     else c["missing"])]
            for run, r in ((c or {}).get("per_seed") or {}).items():
                if r.get("at_best") is False:
                    fell_back.append(f"{LABEL[ds]}/{run}")
                name = run if r["resolved"] == run else f"{run} (→ {r['resolved']})"
                lines.append(f"| {LABEL[ds]} | {arm} | {name} | {r['best_epoch']} | "
                             + " | ".join("--" if not _finite(r.get(k)) else f"{r[k]:.4g}"
                                          for k in per) + " |")
    lines += ["", "Fallbacks to the last epoch: " + (", ".join(fell_back) or "none")
              + ". Missing fits: " + (", ".join(missing) or "none") + "."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--at", choices=AT_CHOICES, default="best")
    parser.add_argument("--out", default=str(ROOT / OV / "experiments" / "adv_confirm"))
    args = parser.parse_args(argv)
    cells = collect(args.at)
    decision = decide(cells)
    sens = sensitivity(args.at, cells)
    text = render(cells, decision, sens, args.at)
    print(text)
    stem = Path(args.out)
    if args.at != "best":
        stem = stem.with_name(f"{stem.name}_at_{args.at}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = {"test": "8.17 follow-up: composition-weight confirmation (tag advc)",
               "rule": RULE, "arms": {LABEL[ds]: arms for ds, arms in ARMS.items()},
               "flags": ARM_FLAGS,
               "read_at": HEADER if args.at == "best" else "final",
               "cells": {LABEL[ds]: c for ds, c in cells.items()},
               "decision": decision, "sensitivity_ovarian_controls": sens}
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=1, default=float))
    stem.with_suffix(".md").write_text(text)
    if args.at == "best":
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "DECISION_ADVC.json").write_text(json.dumps(
            {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": RULE,
             "readings_fixed_before_data": {
                 "a_mlp": "every arm seed s: arm_s < mean(control), strict",
                 "b_ridge": "mean(arm) < mean(control) - sd(control), ddof 1",
                 "c_guards": "arm mean of NMI, cycle_z, recon inside control "
                             "[min - sd, max + sd]; above the envelope passes "
                             "(all three higher-is-better)",
                 "ovarian_control": "the 5 fits pooled; each subset alone is "
                                    "sensitivity only",
                 "combine": "confirmed iff GSE, FF and ovarian all pass; a "
                            "dataset is undecided until all its fits landed",
                 "not_gated": "cycle_w, mirror, I(niche;w) excess, Read A own "
                              "gap, image blocks, cycle_w <= 0.02, dead channels"},
             "controls": {LABEL[ds]: ARMS[ds]["control"] for ds in DECIDING},
             "confirmed": decision["confirmed"], "status": decision["status"],
             "reading": decision["reading"],
             "comp3": {LABEL[ds]: v for ds, v in decision["comp3"].items()},
             "comp5_dose": {"passes_same_rule": decision["comp5_dose"]["passes_same_rule"],
                            "datasets": {LABEL[ds]: v for ds, v in
                                         decision["comp5_dose"]["datasets"].items()}},
             "sensitivity_ovarian_controls": {
                 name: {"control": s["control"], "comp3": s["comp3"]["passes"],
                        "comp5": s["comp5"]["passes"]} for name, s in sens.items()}},
            indent=1, default=float))
    print(f"wrote {stem.with_suffix('.json')}, {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
