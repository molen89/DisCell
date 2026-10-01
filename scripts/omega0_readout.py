#!/usr/bin/env python3
"""READOUT.md of the omega = 0 ablation (scripts/queue_2026-09-30_omega0.sh).

omega0L_s{0,1,2} against finalL_s{0,1,2} per section: mean and seed range
of each read, and, against the prediction set in advance (devlog "omega = 0
ablation (motivation, 2026-09-30)"), which read moved beyond finalL's seed
range. Every value comes from a file the queue's reads wrote:

  NMI z                 runs/<R>/degeneracy.json  nmi_targets
  NMI mu_w              <ROOT>/nmi_w/<D>__<R>.json nmi_w_targets
  cycle R^2 z, w (q90)  degeneracy.json battery.cycle_r2_{z,w}_q90
  I(niche; w) excess    degeneracy.json w_channel.w_niche_mi_excess
  within-type share     degeneracy.json w_channel.w_var_fraction_across_cells
  probe comp MLP share  validation/probe_blocks.json mlp.comp.excess,
                        printed as 1 - exp(-2 excess) in %
  held-out recon        degeneracy.json recon_val_targets

    python scripts/omega0_readout.py <ROOT>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DATASETS = (("gse315411_pdltma06_11_prime_solo", "GSE core"),
            ("xenium_prime_ovarian_cancer_ffpe", "ovarian FFPE"),
            ("xenium_prime_human_lung_cancer_ffpe", "lung FFPE"),
            ("xenium_prime_human_ovary_ff", "FF"))
ARMS = ("finalL", "omega0L")
SEEDS = (0, 1, 2)
# key, label, predicted direction (+1 rises, -1 falls, 0 changes, None: none set)
READS = (("nmi_z", "NMI z", -1),
         ("nmi_w", "NMI mu_w", +1),
         ("cyc_z", "cycle R2 z (q90)", -1),
         ("cyc_w", "cycle R2 w (q90)", +1),
         ("share_w", "within-type share of w's variance", 0),
         ("mi_w", "I(niche; w) excess", None),
         ("probe", "probe comp MLP share (%)", None),
         ("recon", "held-out recon", None))


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def reads(root: Path, ds: str, run: str) -> dict:
    d = Path(f"data/datasets/{ds}/runs/{run}")
    g = load(d / "degeneracy.json") or {}
    b, w = g.get("battery") or {}, g.get("w_channel") or {}
    pb = ((load(d / "validation" / "probe_blocks.json") or {}).get("mlp") or {}).get("comp") or {}
    nw = load(root / "nmi_w" / f"{ds}__{run}.json") or {}
    m = load(d / "metrics.json") or {}
    ex = pb.get("excess")
    return {"nmi_z": g.get("nmi_targets"), "nmi_w": nw.get("nmi_w_targets"),
            "cyc_z": b.get("cycle_r2_z_q90"), "cyc_w": b.get("cycle_r2_w_q90"),
            "share_w": w.get("w_var_fraction_across_cells"),
            "mi_w": w.get("w_niche_mi_excess"),
            "probe": None if ex is None else 100 * (1 - math.exp(-2 * ex)),
            "recon": g.get("recon_val_targets"),
            "_best_epoch": (m.get("best") or {}).get("epoch"),
            "_last_epoch": m.get("last_epoch"), "_minutes": m.get("minutes"),
            "_dead": m.get("dead_w_channel") or w.get("dead_context_channel"),
            "_nmi_check": (nw.get("check") or {}).get("ok"),
            "_omega": (load(d / "config.json") or {}).get("omega")}


def summary(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)}


def fmt(s, n=3):
    if s is None:
        return "--"
    return f"{s['mean']:.{n}f} ({s['min']:.{n}f} to {s['max']:.{n}f})" + ("" if s["n"] == 3 else f" [n={s['n']}]")


def main(argv) -> int:
    root = Path(argv[1])
    L = ["# omega = 0 ablation: omega0L against finalL", "",
         "Queue `scripts/queue_2026-09-30_omega0.sh`; written by `scripts/omega0_readout.py`. "
         "Final configuration with `--omega 0` (intrinsic path dropped; the (1+omega) "
         "z-KL factor reads 1), seeds 0-2 at finalL's splits, reads at best.pt under the "
         "Unassigned mask. Cells: 3-seed mean (min to max).", "",
         "**Prediction set in advance** (devlog \"omega = 0 ablation (motivation, 2026-09-30)\"): "
         "NMI z falls, NMI mu_w rises, cycle R2 of w rises and of z falls, the within-type "
         "share of w's variance changes. If none moves beyond finalL's seed range the claim "
         "is weakened to \"argued, not shown to matter at this configuration\".", "",
         "Two criteria are shown, since the devlog does not say which \"beyond the seed range\" "
         "means: **mean** = omega0L's 3-seed mean outside finalL's [min, max]; **disjoint** = "
         "omega0L's whole range outside finalL's range (the stricter).", ""]
    table, verdicts = {}, []
    for ds, name in DATASETS:
        per = {a: [reads(root, ds, f"{a}_s{s}") for s in SEEDS] for a in ARMS}
        table[ds] = per
        L += [f"## {name} (`{ds}`)", "",
              "| read | finalL | omega0L | mean beyond | disjoint | predicted | verdict |",
              "|---|---|---|---|---|---|---|"]
        for key, label, pred in READS:
            f, o = summary([r[key] for r in per["finalL"]]), summary([r[key] for r in per["omega0L"]])
            n = 2 if key == "probe" else 4 if key in ("mi_w", "recon", "cyc_w") else 3
            if f is None or o is None:
                L.append(f"| {label} | {fmt(f, n)} | {fmt(o, n)} | -- | -- | -- | not read |")
                continue
            beyond = o["mean"] < f["min"] or o["mean"] > f["max"]
            disjoint = o["max"] < f["min"] or o["min"] > f["max"]
            direction = +1 if o["mean"] > f["max"] else -1 if o["mean"] < f["min"] else 0
            ptxt = {1: "rises", -1: "falls", 0: "changes", None: "none set"}[pred]
            if pred is None:
                v = "moved" if beyond else "within range"
            elif not beyond:
                v = "within range"
            elif pred == 0 or direction == pred:
                v = "as predicted" + (" (disjoint)" if disjoint else " (mean only)")
            else:
                v = "OPPOSITE to prediction" + (" (disjoint)" if disjoint else " (mean only)")
            if pred is not None:
                verdicts.append((name, label, beyond, disjoint, v))
            L.append(f"| {label} | {fmt(f, n)} | {fmt(o, n)} | {'yes' if beyond else 'no'} | "
                     f"{'yes' if disjoint else 'no'} | {ptxt} | {v} |")
        L += ["", "Fits: " + "; ".join(
            f"{a}_s{s} best/last epoch {r['_best_epoch']}/{r['_last_epoch']}, "
            f"{(r['_minutes'] or 0):.1f} min, omega {r['_omega']}"
            + (", **DEAD w channel**" if r["_dead"] else "")
            + ("" if r["_nmi_check"] in (True, None) else ", NMI z check FAILED")
            for a in ARMS for s, r in zip(SEEDS, per[a]) if r["_best_epoch"] is not None), ""]

    moved = [v for v in verdicts if v[2]]
    L += ["## Against the prediction", ""]
    if not any(v[2] for v in verdicts):
        L.append("No predicted read moved beyond finalL's seed range on any section: by the "
                 "rule set in advance, the claim that the intrinsic path is load-bearing is "
                 "weakened to \"argued, not shown to matter at this configuration\".")
    else:
        for name, label, beyond, disjoint, v in moved:
            L.append(f"* {name}: {label} -- {v}")
        L += ["", f"{sum(1 for v in moved if v[3])} of {len(verdicts)} predicted "
              f"(section, read) pairs moved with disjoint ranges, {len(moved)} by the mean; "
              f"{sum(1 for v in moved if 'OPPOSITE' in v[4])} of the moves are opposite to "
              "the prediction."]
    missing = [f"{ds}/{a}_s{s}" for ds, _ in DATASETS for a in ARMS for s in SEEDS
               if table[ds][a][s]["nmi_z"] is None]
    failed = sorted(p.name for p in (root / "failed").glob("*")) if (root / "failed").exists() else []
    dead = (root / "DEAD_RUNS.tsv").read_text().strip() if (root / "DEAD_RUNS.tsv").exists() else ""
    ctl = load(root / "control.json")
    L += ["", "## Records", "",
          f"* runs without reads: {', '.join(missing) or 'none'}",
          f"* failed steps: {', '.join(failed) or 'none'} (logs in logs/<step>.log)",
          "* dead fits (recorded, never refitted):" + (f"\n```\n{dead}\n```" if dead else " none")]
    if ctl:
        L.append(f"* code-drift control (GSE finalL_s0 refitted with today's code as "
                 f"`{ctl.get('run')}`): {ctl.get('text')}")
    (root / "READOUT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
