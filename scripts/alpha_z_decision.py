#!/usr/bin/env python3
"""The alpha_z ladder's verdict, by the rule pre-registered in the devlog
("Phase change: model frozen", 2026-09-21).

A rung beats the control (alpha_z = 1/mean count) if, over its 3 seeds:

1. mean pooled ``cycle_z`` is higher than the control's by more than the
   control's own seed range (max - min);
2. probe ``delta_ce`` stays within its floor band;
3. mirror R^2 <= control max + 0.01;
4. NMI >= control min - 0.01;
5. ``cycle_w`` <= 0.02;
6. ``I(z;t)/H(t)`` within 0.05 of the control's.

If exactly one rung qualifies, or several and 1/2 is among them, the factor
is adopted; otherwise alpha_z stays (factor 1).

Operationalisation of the clauses the sentence leaves open, fixed here
before the numbers were read and recorded in the output file:

- clauses 3, 5 read the rung's WORST seed (max), clause 4 its worst (min):
  a guard is intact only if it is intact on every seed;
- clauses 1, 6 read the 3-seed mean, as the sentence says;
- "within its floor band" is ``|delta_ce| <= |noise_floor|`` per seed, the
  permuted-z floor of that same run, and is required of every seed: that is
  what "the probe sits at its floor" has meant throughout the devlog.

Usage::

    python scripts/alpha_z_decision.py --dataset <id> --tag ladder_az \\
        --base 0.007 --factors 0.25 0.5 1 2 --seeds 0 1 2
"""

# ``--at best`` reads cycle, mirror, probe and I(z;t)/H(t) at the accepted
# checkpoint instead of the last-epoch ``final`` (review R26); NMI and recon
# were always read there. The output goes to an ``*_at_best`` sibling. (Kept
# out of the docstring above: that is the rule text recorded in the output.)
#
# Runs re-graded by discell/experiments/probe_regrade.py (R20 + R22,
# validation/probe_blocks.json, always the accepted checkpoint) add the
# per-block probe: each block's excess over its floor as a fraction of the
# uncontrolled (alpha_a = 0) fit's (rung mean; the excess in nats is in the
# JSON) and the seeds passing the four-block guard (<= 25 % of uncontrolled).
# Reported beside the rule, not used by it: clause 2 still reads the pooled,
# legacy probe.

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from discell import paths
from discell.experiments.at_best import (AT_CHOICES, HEADER, battery_at_best,
                                        suffixed)
from discell.model.sweep import run_name

CLAUSES = ("cycle_z_beats_control", "probe_at_floor", "mirror_ok", "nmi_ok",
           "cycle_w_ok", "mi_ratio_ok")
#: the per-block probe (R20 + R22): (family, block)
BLOCKS = (("ridge", "comp"), ("ridge", "img"), ("mlp", "comp"), ("mlp", "img"))


def read_run(run_dir, at: str = "final") -> dict | None:
    """The six instruments of one fit, from wherever the run wrote them.

    *at* = "best" takes the in-trainer block from the accepted checkpoint's
    history row instead of the last-epoch ``final``.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    final = (battery_at_best(run_dir) if at == "best"
             else metrics.get("final")) or {}
    cycle = final.get("cycle") or {}
    degeneracy = final.get("degeneracy")
    if degeneracy is None and (run_dir / "degeneracy.json").exists():
        degeneracy = json.loads(
            (run_dir / "degeneracy.json").read_text()).get("degeneracy")
    probe = final.get("probe") or {}
    blocks_path = run_dir / "validation" / "probe_blocks.json"
    blocks = (json.loads(blocks_path.read_text()) if blocks_path.exists()
              else None)
    return {
        **({f"probe_{f}_{b}_excess": blocks[f][b]["excess"]
            for f, b in BLOCKS} if blocks else {}),
        **({f"probe_{f}_{b}_frac": blocks[f][b]["fraction_of_uncontrolled"]
            for f, b in BLOCKS
            if blocks and blocks[f][b].get("fraction_of_uncontrolled")
            is not None}),
        **({"invariance_pass": bool(blocks["invariance_pass"])}
           if blocks and blocks.get("invariance_pass") is not None else {}),
        **({"at_best": final["at_best"]} if at == "best" else {}),
        "run": run_dir.name,
        "cycle_z": (cycle.get("z") or {}).get("r2_pooled"),
        "cycle_w": (cycle.get("w") or {}).get("r2_pooled"),
        "nmi": metrics["best"]["nmi"],
        "nmi_final": final.get("nmi"),
        "recon": metrics["best"]["recon_val"],
        "mirror_r2": (final.get("mirror") or {}).get("r2"),
        "probe_delta_ce": probe.get("delta_ce"),
        "probe_noise_floor": probe.get("noise_floor"),
        "mi_ratio": (degeneracy or {}).get("mi_ratio"),
    }


def rung_stats(rows: list[dict]) -> dict:
    """Mean / min / max / range of every instrument over a rung's seeds."""
    out: dict = {"n_seeds": len(rows), "runs": [r["run"] for r in rows]}
    for key in ("cycle_z", "cycle_w", "nmi", "nmi_final", "recon",
                "mirror_r2", "probe_delta_ce", "probe_noise_floor",
                "mi_ratio") + tuple(f"probe_{f}_{b}_{k}" for f, b in BLOCKS
                                    for k in ("excess", "frac")):
        values = [r[key] for r in rows if r.get(key) is not None]
        if not values:
            out[key] = None
            continue
        out[key] = {"mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "range": float(np.max(values) - np.min(values)),
                    "values": [float(v) for v in values]}
    out["invariance_pass_per_seed"] = [r["invariance_pass"] for r in rows
                                       if "invariance_pass" in r]
    out["probe_at_floor_per_seed"] = [
        bool(abs(r["probe_delta_ce"]) <= abs(r["probe_noise_floor"]))
        for r in rows
        if r.get("probe_delta_ce") is not None
        and r.get("probe_noise_floor") is not None]
    return out


def judge(rung: dict, control: dict) -> dict:
    """The six clauses, each as a bool with the numbers that decided it."""
    def val(stats, key, stat):
        entry = stats.get(key)
        return None if entry is None else entry[stat]

    checks = {
        "cycle_z_beats_control": (
            val(rung, "cycle_z", "mean") is not None
            and val(rung, "cycle_z", "mean")
            > val(control, "cycle_z", "mean") + val(control, "cycle_z",
                                                    "range")),
        "probe_at_floor": bool(rung["probe_at_floor_per_seed"])
        and all(rung["probe_at_floor_per_seed"]),
        "mirror_ok": (val(rung, "mirror_r2", "max") is not None
                      and val(rung, "mirror_r2", "max")
                      <= val(control, "mirror_r2", "max") + 0.01),
        "nmi_ok": (val(rung, "nmi", "min") is not None
                   and val(rung, "nmi", "min")
                   >= val(control, "nmi", "min") - 0.01),
        "cycle_w_ok": (val(rung, "cycle_w", "max") is not None
                       and val(rung, "cycle_w", "max") <= 0.02),
        "mi_ratio_ok": (val(rung, "mi_ratio", "mean") is not None
                        and val(control, "mi_ratio", "mean") is not None
                        and abs(val(rung, "mi_ratio", "mean")
                                - val(control, "mi_ratio", "mean")) <= 0.05),
    }
    return {"checks": checks, "qualifies": all(checks.values())}


def decide(qualifying: list[float]) -> tuple[float, str]:
    """The adoption rule: exactly one rung, or several with 1/2 among them."""
    beats = [f for f in qualifying if f != 1.0]
    if len(beats) == 1:
        return beats[0], f"exactly one rung qualifies (x{beats[0]:g})"
    if len(beats) > 1 and 0.5 in beats:
        return 0.5, ("several rungs qualify and 1/2 is among them "
                     f"({', '.join('x%g' % f for f in beats)})")
    if len(beats) > 1:
        return 1.0, ("several rungs qualify but 1/2 is not among them "
                     f"({', '.join('x%g' % f for f in beats)}) -- alpha_z "
                     "stays")
    return 1.0, "no rung qualifies -- alpha_z stays"


def render(table: dict, control_factor: float) -> str:
    rows = [("factor", "alpha_z", "seeds", "cycle_z (mean)", "cycle_w (max)",
             "NMI (min)", "mirror (max)", "probe dCE / floor (pooled, legacy)",
             "I(z;t)/H(t)", "recon (mean)", "clauses",
             "probe comp ÷ uncontrolled r/m (mean)", "img r/m (mean)",
             "invariance guard")]
    for key in sorted(table, key=float):
        entry = table[key]
        stats = entry["stats"]

        def fmt(name, stat, spec="{:.4f}"):
            e = stats.get(name)
            return "-" if e is None else spec.format(e[stat])
        checks = entry.get("judgement", {}).get("checks", {})
        marks = ("control" if float(key) == control_factor else
                 "".join("+" if checks.get(c) else "-" for c in CLAUSES))
        rows.append((
            f"x{float(key):g}", f"{entry['alpha_z']:g}",
            str(stats["n_seeds"]), fmt("cycle_z", "mean"),
            fmt("cycle_w", "max"), fmt("nmi", "min"), fmt("mirror_r2", "max"),
            f"{fmt('probe_delta_ce', 'mean')} / "
            f"{fmt('probe_noise_floor', 'mean')}",
            fmt("mi_ratio", "mean"), fmt("recon", "mean", "{:.4f}"), marks,
            f"{fmt('probe_ridge_comp_frac', 'mean', '{:.2f}')} / "
            f"{fmt('probe_mlp_comp_frac', 'mean', '{:.2f}')}",
            f"{fmt('probe_ridge_img_frac', 'mean', '{:.2f}')} / "
            f"{fmt('probe_mlp_img_frac', 'mean', '{:.2f}')}",
            f"{sum(stats['invariance_pass_per_seed'])}/"
            f"{len(stats['invariance_pass_per_seed'])}"))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
             for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines.append("")
    lines.append("clause order: " + ", ".join(CLAUSES))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset",
                        default="xenium_prime_ovarian_cancer_ffpe")
    parser.add_argument("--tag", default="ladder_az")
    parser.add_argument("--base", type=float, default=0.007,
                        help="the control alpha_z (1/mean count)")
    parser.add_argument("--factors", type=float, nargs="*",
                        default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--control-factor", type=float, default=1.0)
    parser.add_argument("--control-runs", nargs="*", default=None,
                        help="run names to use as the control rung instead "
                             "of the ladder's own (the sweep3 k0.1 fits are "
                             "the alpha_z = base control at 200/20)")
    parser.add_argument("--out", default="alpha_z_decision.json")
    parser.add_argument("--at", choices=AT_CHOICES, default="final",
                        help="where the in-trainer reads come from: the "
                             "last-epoch `final` (default) or the accepted "
                             "checkpoint (R26); `best` writes *_at_best files")
    args = parser.parse_args(argv)

    runs_dir = paths.dataset(args.dataset).root / "runs"
    table: dict = {}
    missing: list[str] = []
    fell_back: list[str] = []
    for factor in args.factors:
        value = args.base * factor
        if (factor == args.control_factor and args.control_runs):
            names = list(args.control_runs)
        else:
            names = [run_name(value, seed, args.tag, "alpha_z")
                     for seed in args.seeds]
        rows = []
        for name in names:
            row = read_run(runs_dir / name, args.at)
            if row is None:
                missing.append(name)
            else:
                rows.append(row)
                if row.get("at_best") is False:
                    fell_back.append(name)
        if rows:
            table[f"{factor:g}"] = {"factor": factor, "alpha_z": value,
                                    "stats": rung_stats(rows)}

    control_key = f"{args.control_factor:g}"
    out_path = suffixed(paths.dataset(args.dataset).root / "experiments"
                        / args.out, args.at)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if control_key not in table:
        payload = {"factor": 1.0, "qualifying": [], "table": table,
                   "missing": missing,
                   "reason": "the control rung is missing -- alpha_z stays"}
        out_path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload["reason"]))
        return 1

    control = table[control_key]["stats"]
    qualifying: list[float] = []
    for key, entry in table.items():
        if key == control_key:
            continue
        entry["judgement"] = judge(entry["stats"], control)
        if entry["judgement"]["qualifies"]:
            qualifying.append(entry["factor"])
    factor, reason = decide(sorted(qualifying))
    payload = {"factor": factor, "qualifying": sorted(qualifying),
               "reason": reason, "rule": __doc__, "dataset": args.dataset,
               "tag": args.tag, "base_alpha_z": args.base,
               "control_runs": table[control_key]["stats"]["runs"],
               "missing": missing, "table": table}
    if args.at == "best":
        payload["read_at"] = HEADER
        payload["fell_back_to_final"] = fell_back
        print(f"{HEADER}; last-epoch fallbacks: "
              + (", ".join(fell_back) if fell_back else "none"))
    out_path.write_text(json.dumps(payload, indent=2))
    print(render(table, args.control_factor))
    print(f"\nqualifying: {sorted(qualifying) or 'none'}")
    print(f"verdict: factor {factor:g} -- {reason}")
    if missing:
        print(f"missing runs: {', '.join(missing)}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
