#!/usr/bin/env python3
"""The read-out for the w-collapse grid (devlog pre-registration, 2026-09-23).

Per arm x alpha_w x dataset: the collapse count (summed KL_w at epoch 20 and at
the end, below ``DEAD_W_KL_SUM``) plus the seed mean/min/max of every read the
pre-registration names, then the decision rule applied mechanically.

    uv run python scripts/wcollapse_table.py [--tag wfix] [--at best]

``--at best`` reads the in-trainer fields at the accepted checkpoint instead
of the last-epoch ``final`` (review R26). The collapse probes stay on the
trajectory's epoch-20 and last records, as the pre-registration defines them.
Outputs go to ``*_at_best`` siblings.

Runs re-graded by ``discell/experiments/probe_regrade.py`` (R20 + R22,
``validation/probe_blocks.json``, always the accepted checkpoint) add the
per-block probe columns -- each block's excess over its floor -- and the
count of seeds passing the four-block invariance guard (each block's excess
<= 25 % of the uncontrolled alpha_a = 0 fit's; the table shows the fractions,
the per-run records the excess in nats too). They are reported, not used: the
decision rule and its hand-off still read the pooled, legacy probe as
pre-registered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from discell.experiments.at_best import (AT_CHOICES, HEADER, battery_at_best,
                                        suffixed)
from discell.model.train import DEAD_W_KL_SUM

ROOT = Path(__file__).resolve().parents[1] / "data" / "datasets"
FF = "xenium_prime_human_ovary_ff"
OV = "xenium_prime_ovarian_cancer_ffpe"
ARMS = ["control", "warmup30", "fb0.05", "warmup30+fb0.05"]
#: the CLI flags each arm implies -- the hand-off other queued jobs read
ARM_FLAGS = {"control": "",
             "warmup30": "--w-warmup-epochs 30",
             "fb0.05": "--w-free-bits 0.05",
             "warmup30+fb0.05": "--w-warmup-epochs 30 --w-free-bits 0.05"}
#: where the downstream job waits for the verdict
DECISION_PATH = (Path(__file__).resolve().parents[1] / "scripts" / "logs"
                 / "wcollapse_2026-09-23" / "DECISION.json")
#: the pre-registration's collapse probes: the epoch-20 record (histories are
#: written every 5 epochs, so epoch 19 is "epoch 20") and the last record
EPOCH_20 = 19
CYCLE_W_GUARD = 0.02


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def read_run(dataset: str, run: str, at: str = "final") -> dict | None:
    """Every number the table needs for one fit, or None if it has not landed.

    *at* = "best" takes the in-trainer fields from the accepted checkpoint's
    history row; the collapse probes and the post-hoc files are unchanged.
    """
    d = ROOT / dataset / "runs" / run
    metrics = _load(d / "metrics.json")
    if metrics is None:
        return None
    history = [json.loads(line) for line in
               (d / "history.jsonl").read_text().splitlines() if line.strip()]
    by_epoch = {int(r["epoch"]): r for r in history}
    kl_at = lambda r: float(np.sum(r["kl_w_per_dim"])) if r else float("nan")
    early, last = by_epoch.get(EPOCH_20), (history[-1] if history else None)
    final = battery_at_best(d) if at == "best" else metrics["final"]
    out = {
        "run": run,
        "kl_w_epoch20": kl_at(early),
        "kl_w_end": kl_at(last),
        "recon": final["recon_val"],
        "nmi": final["nmi"],
        "mirror_r2": final["mirror"]["r2"],
        "probe_delta_ce": final["probe"]["delta_ce"],
        "probe_noise_floor": final["probe"]["noise_floor"],
        # pooled, the convention of scripts/envelope_tables.py
        "cycle_z": final.get("cycle", {}).get("z", {}).get("r2_pooled",
                                                          float("nan")),
        "cycle_w": final.get("cycle", {}).get("w", {}).get("r2_pooled",
                                                          float("nan")),
        "kl_w": float(np.sum(final["kl_w_per_dim"])),
        "mi_ratio": final["degeneracy"]["mi_ratio"],
        "alpha_w_eff_end": (last or {}).get("alpha_w_eff", float("nan")),
        # the epoch the checkpoint came from, and the last epoch trained. With
        # a warm-up, best_epoch must sit clear of w_warmup_epochs: it is the
        # reader's evidence that the arm was scored after re-convergence, not
        # a few evaluations after its warm-up ended.
        "best_epoch": metrics.get("best", {}).get("epoch", float("nan")),
        "last_epoch": int(last["epoch"]) if last else float("nan"),
        "w_warmup_epochs": json.loads(
            (d / "config.json").read_text()).get("w_warmup_epochs", 0),
    }
    if at == "best":
        out["at_best"] = final["at_best"]
    out["collapsed_epoch20"] = bool(out["kl_w_epoch20"] < DEAD_W_KL_SUM)
    out["collapsed_end"] = bool(out["kl_w_end"] < DEAD_W_KL_SUM)
    out["collapsed"] = out["collapsed_epoch20"] or out["collapsed_end"]
    # how far the probe sits above its own within-type permutation floor; the
    # pre-registration compares it to the control's envelope, not to a pinned
    # absolute threshold, so this is reported, not thresholded
    out["probe_excess"] = out["probe_delta_ce"] - out["probe_noise_floor"]
    out["cycle_w_ok"] = bool(out["cycle_w"] <= CYCLE_W_GUARD)
    # "the probe sits at its floor" = |delta_ce| <= |noise_floor| on that same
    # run, per seed -- the convention of scripts/alpha_z_decision.py and of the
    # devlog throughout. Amendment 2 (c) keeps it an absolute guard.
    out["probe_at_floor"] = bool(abs(out["probe_delta_ce"])
                                 <= abs(out["probe_noise_floor"]))

    blocks = _load(d / "validation" / "probe_blocks.json")
    if blocks:
        for family, block in BLOCKS:
            entry = blocks[family][block]
            out[f"probe_{family}_{block}_excess"] = entry["excess"]
            if entry.get("fraction_of_uncontrolled") is not None:
                out[f"probe_{family}_{block}_frac"] = entry[
                    "fraction_of_uncontrolled"]
        if blocks.get("invariance_pass") is not None:
            out["invariance_pass"] = bool(blocks["invariance_pass"])

    guard = _load(d / "degeneracy.json") or {}
    w_channel = guard.get("w_channel") or {}
    out["w_niche_mi"] = w_channel.get("w_niche_mi", float("nan"))
    out["w_niche_mi_excess"] = w_channel.get("w_niche_mi_excess", float("nan"))
    out["w_guard_dead"] = w_channel.get("dead_context_channel")

    dist = _load(d / "transport" / "transport_distribution.json")
    if dist:
        out["own_gap_closed"] = (dist["summary_model_own"]["pairwise"]
                                 ["median_gap_closed"])
    return out


#: the per-block probe (R20 + R22): (family, block)
BLOCKS = (("ridge", "comp"), ("ridge", "img"), ("mlp", "comp"), ("mlp", "img"))
FIELDS = ["best_epoch", "last_epoch", "recon", "nmi", "probe_delta_ce",
          "probe_noise_floor", "probe_excess", "mirror_r2", "cycle_z",
          "cycle_w", "kl_w", "w_niche_mi_excess", "mi_ratio", "own_gap_closed"
          ] + [f"probe_{f}_{b}_frac" for f, b in BLOCKS]
#: column headers where the field name alone would mislead
LABELS = {"probe_delta_ce": "probe_delta_ce (pooled, legacy)",
          "probe_noise_floor": "probe_noise_floor (pooled, legacy)",
          "probe_excess": "probe_excess (pooled, legacy)",
          **{f"probe_{f}_{b}_frac": f"probe {b}/{f} ÷ uncontrolled"
             for f, b in BLOCKS}}
# best_epoch/last_epoch are shown but never compared across arms: the decision
# rule's envelope clause names its own fields in `decide`.


def summarise(runs: list[dict]) -> dict:
    """Seed mean/min/max of every field, plus the collapse counts."""
    out = {"n_seeds": len(runs),
           "seeds_done": [r["run"] for r in runs],
           "collapsed_epoch20": sum(r["collapsed_epoch20"] for r in runs),
           "collapsed_end": sum(r["collapsed_end"] for r in runs),
           "collapsed": sum(r["collapsed"] for r in runs),
           "w_guard_dead": sum(bool(r.get("w_guard_dead")) for r in runs),
           "cycle_w_ok": sum(r["cycle_w_ok"] for r in runs),
           "probe_at_floor": sum(r["probe_at_floor"] for r in runs),
           # the four-block guard, over the seeds that have been re-graded
           "invariance_pass": sum(bool(r.get("invariance_pass")) for r in runs),
           "invariance_graded": sum("invariance_pass" in r for r in runs),
           # with a warm-up, how many seeds took their checkpoint from an epoch
           # that is not merely the first eligible one
           "best_epoch_at_warmup_edge": sum(
               1 for r in runs if r.get("w_warmup_epochs")
               and r["best_epoch"] < r["w_warmup_epochs"] + 10)}
    for field in FIELDS:
        vals = [r[field] for r in runs if field in r and r[field] == r[field]]
        if vals:
            out[field] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                          "max": float(max(vals)), "n": len(vals),
                          # sample sd; the widening half-width of amendment 2(b)
                          "sd": float(np.std(vals, ddof=1)) if len(vals) > 1
                          else 0.0}
    return out


def collect(tag: str, at: str = "final") -> dict:
    cells = {}
    for arm in ARMS:
        for dataset, alphas, seeds in ((FF, ["0.3", "0.1"], [0, 1, 2, 3]),
                                       (OV, ["0.1"], [0, 1, 2])):
            for aw in alphas:
                runs = [r for r in
                        (read_run(dataset, f"{tag}_{arm}_aw{aw}_s{s}", at)
                         for s in seeds) if r is not None]
                if runs:
                    cells[(dataset, aw, arm)] = {"runs": runs,
                                                 **summarise(runs)}
    return cells


def ff_margin(cells: dict, arm: str) -> float:
    """log10 of the arm's WORST FF end-of-run KL_w over the death threshold.

    The amended tie-break (coordinator, 2026-09-23 -- made **after** FF alpha_w
    0.3 seed 0 and before the other FF seeds, so it is an amendment, not part
    of the pre-registration): among qualifying arms prefer the one whose
    minimum end-of-run KL_w across all FF seeds sits farthest above 1e-5 on a
    log scale. Taking the minimum over both alpha_w levels is the conservative
    reading of "all FF seeds": an arm is only as robust as its worst seed.

    NaN when the arm has no FF run yet. Negative means that worst seed is
    below the threshold, i.e. the arm collapsed somewhere and fails clause 1
    regardless.
    """
    ends = [r["kl_w_end"] for aw in ("0.3", "0.1")
            for r in (cells.get((FF, aw, arm)) or {}).get("runs", [])
            if r["kl_w_end"] == r["kl_w_end"]]
    if not ends:
        return float("nan")
    return float(np.log10(max(min(ends), 1e-300) / DEAD_W_KL_SUM))


#: amendment 2(a): I(z;t)/H(t) is judged as in the alpha_z-ladder rule already
#: on record (devlog 2026-09-22) -- within this of the control mean, no
#: direction -- rather than against the seed envelope.
MI_RATIO_TOLERANCE = 0.05


def inside_envelope(value: float, control: dict, field: str,
                    higher_is_better: bool | None) -> bool:
    """Amendment 2(b): inside the control's [min, max] **widened by one control
    seed-sd on each side**, or better in the field's named direction.

    The widening exists because with three seeds a bare min/max is a very
    unstable envelope: before it, two arms failed on I(z;t)/H(t) by 9e-6 and
    4e-5, four orders below that quantity's own seed noise.
    ``higher_is_better=None`` means no direction, so only the widened band
    passes.
    """
    if control is None or field not in control:
        return False
    entry = control[field]
    sd = entry.get("sd", 0.0)
    lo, hi = entry["min"] - sd, entry["max"] + sd
    if lo <= value <= hi:
        return True
    if higher_is_better is None:
        return False
    return value > hi if higher_is_better else value < lo


def decide(cells: dict) -> dict:
    """The pre-registered rule, applied mechanically.

    An arm qualifies if it has zero collapses at both FF alpha_w levels and its
    ovarian guards and recon sit inside the control's seed envelope. Among the
    qualifying arms prefer warm-up alone, then free bits, then both.
    """
    control_ov = cells.get((OV, "0.1", "control"))
    verdict = {}
    for arm in ARMS:
        clauses, reasons = {}, []
        ff_cells = [cells.get((FF, aw, arm)) for aw in ("0.3", "0.1")]
        if any(c is None for c in ff_cells):
            clauses["ff_zero_collapses"] = None
            reasons.append("FF grid incomplete")
        else:
            total = sum(c["collapsed"] for c in ff_cells)
            clauses["ff_zero_collapses"] = total == 0
            reasons.append(f"FF collapses {total}/"
                           f"{sum(c['n_seeds'] for c in ff_cells)}")
        ov = cells.get((OV, "0.1", arm))
        if ov is None or control_ov is None:
            clauses["ovarian_inside_envelope"] = None
            reasons.append("ovarian arm or control missing")
        elif arm == "control":
            clauses["ovarian_inside_envelope"] = True
            reasons.append("ovarian: control is its own envelope")
        else:
            #: field -> which direction counts as "better than the widened
            #: envelope"; None = no direction, the band alone. probe_delta_ce
            #: is NOT here: amendment 2(c) keeps it an absolute per-seed guard.
            checks = {"recon": True, "nmi": True, "mirror_r2": False,
                      "cycle_z": True, "own_gap_closed": True}
            bad = [f for f, hib in checks.items()
                   if f in ov and not inside_envelope(ov[f]["mean"],
                                                      control_ov, f, hib)]
            # amendment 2(a): I(z;t)/H(t) against the control mean, not the band
            if "mi_ratio" in ov and "mi_ratio" in control_ov:
                if abs(ov["mi_ratio"]["mean"]
                       - control_ov["mi_ratio"]["mean"]) > MI_RATIO_TOLERANCE:
                    bad.append("mi_ratio")
            # amendment 2(c): absolute per-seed guards, unchanged
            guards_ok = (ov["cycle_w_ok"] == ov["n_seeds"]
                         and ov["probe_at_floor"] == ov["n_seeds"]
                         and ov["collapsed"] == 0)
            clauses["ovarian_inside_envelope"] = (not bad) and guards_ok
            reasons.append("ovarian outside envelope on " + ", ".join(bad)
                           if bad else "ovarian inside envelope")
            if not guards_ok:
                reasons.append(f"ovarian guards: cycle_w "
                               f"{ov['cycle_w_ok']}/{ov['n_seeds']} ok, probe "
                               f"{ov['probe_at_floor']}/{ov['n_seeds']} at "
                               f"floor, {ov['collapsed']} collapses")
        values = list(clauses.values())
        verdict[arm] = {
            "clauses": clauses, "reasons": reasons,
            "qualifies": all(v is True for v in values),
            "undecided": any(v is None for v in values)}
    # Two selections are reported, never one. The pre-registration preferred
    # (b) warm-up outright among qualifying arms; the 2026-09-23 amendment
    # replaces that with a robustness tie-break on the FF margin, keeping a
    # bounded preference for warm-up because its converged objective is the
    # pinned one. Reporting both keeps an amendment made after seeing FF
    # alpha_w 0.3 seed 0 from quietly overwriting what the pre-registration
    # would have chosen: if they disagree, that disagreement is a finding.
    margins = {a: ff_margin(cells, a) for a in ARMS}
    qualifying = [a for a in ARMS if verdict[a]["qualifies"] and a != "control"]

    #: the original rule: fixed preference order among the qualifying arms
    preference = ["warmup30", "fb0.05", "warmup30+fb0.05"]
    original = next((a for a in preference if a in qualifying), None)
    original_reason = ("pre-registered preference order "
                       f"({' > '.join(preference)})" if original
                       else "no arm qualifies")

    amended, amended_reason = None, "no arm qualifies"
    if qualifying:
        scored = [a for a in qualifying if margins[a] == margins[a]]
        if not scored:
            amended = original
            amended_reason = "no FF margin yet; falls back to the original rule"
        else:
            best = max(scored, key=lambda a: margins[a])
            amended, amended_reason = best, (
                f"largest FF margin ({margins[best]:+.2f} log10 above 1e-5)")
            if ("warmup30" in scored
                    and margins["warmup30"] >= margins[best] - 1.0):
                amended = "warmup30"
                amended_reason = (
                    f"warm-up kept: its margin {margins['warmup30']:+.2f} is "
                    f"within one order of the best ({margins[best]:+.2f}, "
                    f"{best}), and its converged objective is the pinned one")
    # the downstream job consumes the amended rule
    selected, selection_reason = amended, amended_reason
    undecided = [a for a in ARMS if verdict[a]["undecided"]]
    return {"arms": verdict, "selected": selected,
            "selection_reason": selection_reason,
            "selected_original": original,
            "original_reason": original_reason,
            "selected_amended": amended,
            "amended_reason": amended_reason,
            "rules_agree": original == amended,
            "ff_margin_log10": margins,
            "rule": "pre-registered clauses + 2026-09-23 FF-margin amendment",
            "fallback": None if selected or undecided else
            "no arm qualifies: fall back to per-slide alpha_w = 0.05 on FF and "
            "report the collapse rate as a limitation",
            "undecided": undecided}


def write_decision(ruling: dict, path: Path, force: bool = False) -> bool:
    """The hand-off another queued job waits on: the arm and its CLI flags.

    Withheld while any arm is undecided -- a job that acted on a half-finished
    grid would be fitting the wrong arm -- unless *force* says otherwise.
    """
    if ruling["undecided"] and not force:
        print(f"decision withheld ({path}): grid incomplete for "
              + ", ".join(ruling["undecided"]))
        return False
    selected = ruling["selected"]
    payload = {"selected_arm": selected,
               "flags": ARM_FLAGS.get(selected, ""),
               "qualifying": [a for a in ARMS if ruling["arms"][a]["qualifies"]],
               # both verdicts travel with the hand-off; selected_arm/flags are
               # the amended rule's, which is what the downstream job consumes
               "selected_arm_preregistered": ruling.get("selected_original"),
               "selected_arm_amended": ruling.get("selected_amended"),
               "rules_agree": ruling.get("rules_agree"),
               "preregistered_reason": ruling.get("original_reason"),
               "amended_reason": ruling.get("amended_reason"),
               "selection_reason": ruling.get("selection_reason"),
               # NaN is not valid strict JSON and a downstream job parses
               # this file, so an arm with no FF run yet is null, not NaN
               "ff_margin_log10": {a: (None if m != m else round(m, 3)) for a, m
                                   in (ruling.get("ff_margin_log10") or {}).items()},
               "rule": ruling.get("rule"),
               "fallback": ruling["fallback"],
               "undecided": ruling["undecided"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    print(f"wrote {path}: {json.dumps(payload)}")
    return True


#: the pre-registered fallback is "per-slide alpha_w for FF (0.05, inside the
#: grid)". These existing survey runs are its evidence: if FF at alpha_w 0.05
#: and 0.02 does not collapse, the fallback has somewhere to fall back to.
#: They are alpha_z 0.0007, not the 0.00035 of this grid, so they are evidence
#: about the fallback, not rows of the grid -- reported separately for that.
FALLBACK_RUNS = {"0.05": [f"sweep3_aw0.05_s{s}" for s in (0, 1, 2)],
                 "0.02": [f"sweep3_aw0.02_s{s}" for s in (0, 1, 2)]}


def fallback_survey(at: str = "final") -> list[dict]:
    """Collapse counts for the FF low-alpha_w survey runs behind the fallback."""
    rows = []
    for aw in sorted(FALLBACK_RUNS):
        runs = [r for r in (read_run(FF, n, at) for n in FALLBACK_RUNS[aw])
                if r is not None]
        if not runs:
            rows.append({"alpha_w": aw, "n": 0})
            continue
        rows.append({"alpha_w": aw, "n": len(runs),
                     "collapsed": sum(r["collapsed"] for r in runs),
                     "collapsed_epoch20": sum(r["collapsed_epoch20"]
                                              for r in runs),
                     "collapsed_end": sum(r["collapsed_end"] for r in runs),
                     "kl_w_end_min": min(r["kl_w_end"] for r in runs),
                     "recon_mean": float(np.mean([r["recon"] for r in runs])),
                     "nmi_mean": float(np.mean([r["nmi"] for r in runs])),
                     "runs": [r["run"] for r in runs]})
    return rows


def _fmt(cell: dict, field: str) -> str:
    if field not in cell:
        return "--"
    e = cell[field]
    return f"{e['mean']:.4g} [{e['min']:.4g}, {e['max']:.4g}]"


def render(cells: dict, ruling: dict, at: str = "final") -> str:
    lines = ["# The w-collapse grid (tag wfix, pre-registered 2026-09-23)", "",
             "Cells are seed mean [min, max]; collapse = summed KL_w < "
             f"{DEAD_W_KL_SUM:g} at epoch {EPOCH_20 + 1} or at the end.", "",
             "`FF margin` is log10 of the arm's **worst** end-of-run KL_w over",
             "all FF seeds, divided by the threshold -- the tie-break added by",
             "the 2026-09-23 amendment. It is an **amendment**, adopted after",
             "FF alpha_w 0.3 seed 0 was seen and before the other FF seeds; it",
             "is not part of the original pre-registration.", ""]
    if at == "best":
        lines[0] += f" -- {HEADER}"
        fell_back = [r["run"] for c in cells.values() for r in c["runs"]
                     if not r["at_best"]]
        lines[2:2] = [
            "recon, NMI, probe, mirror, cycle, KL_w and I(z;t)/H(t) are the "
            "history row at the accepted epoch. The collapse probes (epoch-20 "
            "and last record), the FF margin, and the I(niche;w) guard and "
            "transport reads (from best.pt) are unchanged. Last-epoch "
            "fallbacks: " + (", ".join(fell_back) if fell_back else "none")
            + ".", ""]
    margins = {a: ff_margin(cells, a) for a in ARMS}
    lines += ["| arm | FF margin (log10 above 1e-5) |", "|---|---|"]
    for arm in ARMS:
        m = margins[arm]
        lines.append(f"| {arm} | " + ("--" if m != m else f"{m:+.2f}") + " |")
    lines.append("")
    cols = FIELDS
    for dataset, aw in ((FF, "0.3"), (FF, "0.1"), (OV, "0.1")):
        lines += [f"## {dataset}, alpha_w {aw}", "",
                  "| arm | n | collapse e20/end | "
                  + " | ".join(LABELS.get(c, c) for c in cols)
                  + " | invariance guard (pass/graded) |",
                  "|---" * (len(cols) + 4) + "|"]
        for arm in ARMS:
            cell = cells.get((dataset, aw, arm))
            if cell is None:
                lines.append(f"| {arm} | 0 | -- |" + " -- |" * (len(cols) + 1))
                continue
            lines.append(
                f"| {arm} | {cell['n_seeds']} | "
                f"{cell['collapsed_epoch20']}/{cell['collapsed_end']} | "
                + " | ".join(_fmt(cell, c) for c in cols)
                + f" | {cell['invariance_pass']}/{cell['invariance_graded']} |")
        lines.append("")
    lines += ["## The decision rule", ""]
    for arm in ARMS:
        v = ruling["arms"][arm]
        state = ("QUALIFIES" if v["qualifies"] else
                 "undecided" if v["undecided"] else "fails")
        lines.append(f"- **{arm}**: {state} -- " + "; ".join(v["reasons"]))
    lines += ["", "### Fallback evidence: FF at low alpha_w (existing survey)",
              "",
              "The pre-registered fallback is per-slide alpha_w = 0.05 on FF.",
              "These `sweep3` runs are at alpha_z 0.0007, **not** this grid's",
              "0.00035, so they are evidence about the fallback, not grid rows.",
              "",
              "| alpha_w | n | collapses e20/end | worst end KL_w | recon | NMI |",
              "|---|---|---|---|---|---|"]
    for row in fallback_survey(at):
        if not row["n"]:
            lines.append(f"| {row['alpha_w']} | 0 | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {row['alpha_w']} | {row['n']} | "
            f"{row['collapsed_epoch20']}/{row['collapsed_end']} | "
            f"{row['kl_w_end_min']:.3g} | {row['recon_mean']:.4f} | "
            f"{row['nmi_mean']:.4f} |")

    pending = ruling["undecided"]
    show = lambda a: a or ("none yet -- grid incomplete" if pending else "none")
    lines += ["", "### Both verdicts", "",
              "| rule | selects | reason |", "|---|---|---|",
              f"| pre-registered | {show(ruling.get('selected_original'))} | "
              f"{ruling.get('original_reason', '')} |",
              f"| amended (2026-09-23) | {show(ruling.get('selected_amended'))} | "
              f"{ruling.get('amended_reason', '')} |", ""]
    if pending:
        lines.append("Both are provisional: the grid is incomplete.")
    elif ruling.get("rules_agree"):
        lines.append("**The two rules agree.** The amendment did not change "
                     "the outcome.")
    else:
        lines.append(
            "**The two rules DISAGREE**: the pre-registration selects "
            f"`{show(ruling.get('selected_original'))}`, the amendment "
            f"`{show(ruling.get('selected_amended'))}`. The amendment was made "
            "after FF alpha_w 0.3 seed 0 was seen, so this disagreement is a "
            "finding and must be reported as one, not resolved silently.")
    lines += ["", f"**Downstream uses the amended rule: {show(ruling['selected'])}**"]
    lines.append("Rule: " + ruling.get("rule", ""))
    if ruling["fallback"]:
        lines.append(f"Fallback: {ruling['fallback']}")
    if ruling["undecided"]:
        lines.append("Undecided (fits still missing): "
                     + ", ".join(ruling["undecided"]))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", default="wfix")
    parser.add_argument("--out", default=None,
                        help="output stem; default the FF experiments dir")
    parser.add_argument("--decision", default=None,
                        help=f"hand-off file; default {DECISION_PATH}")
    parser.add_argument("--at", choices=AT_CHOICES, default="final",
                        help="where the in-trainer reads come from: the "
                             "last-epoch `final` (default) or the accepted "
                             "checkpoint (R26); `best` writes *_at_best files")
    parser.add_argument("--force-decision", action="store_true",
                        help="write the hand-off even while the grid is "
                             "incomplete (a downstream job reads it, so by "
                             "default an undecided rule writes nothing)")
    args = parser.parse_args(argv)

    cells = collect(args.tag, args.at)
    ruling = decide(cells)
    text = render(cells, ruling, args.at)
    print(text)

    stem = suffixed(Path(args.out) if args.out else (
        ROOT / FF / "experiments" / "wcollapse"), args.at)
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tag": args.tag, "dead_w_kl_sum": DEAD_W_KL_SUM,
               "epoch_20_record": EPOCH_20,
               "fallback_survey": fallback_survey(args.at),
               "cells": [{"dataset": d, "alpha_w": a, "arm": m, **c}
                         for (d, a, m), c in cells.items()],
               "decision": ruling}
    if args.at == "best":
        payload["read_at"] = HEADER
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=1,
                                                    default=float))
    stem.with_suffix(".md").write_text(text)
    print(f"wrote {stem.with_suffix('.json')} and {stem.with_suffix('.md')}")
    write_decision(ruling, suffixed(Path(args.decision) if args.decision
                                    else DECISION_PATH, args.at),
                   force=args.force_decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
