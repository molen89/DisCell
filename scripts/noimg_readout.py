#!/usr/bin/env python3
"""READOUT.md of the no-image ablation (scripts/queue_2026-10-01_noimg.sh).

noimgL_s{0,1,2} (the final configuration with Phi removed from the context
c, ``--no-image``) against finalL_s{0,1,2} per section: mean and seed range
of each read, "beyond finalL's seed range" yes/no, and the prediction set in
advance (devlog "No-image ablation at the final configuration (motivation,
2026-10-01; author)"): removing the image costs most on FF (niche
information in w, relocation, programme stability) and least on the TMA
core; if nothing moves beyond finalL's seed range, the image is optional at
this configuration. Every value comes from a file the queue's reads wrote:

  NMI z, cycle R^2 z (q90)   runs/<R>/degeneracy.json nmi_targets, battery.*
  I(niche; w) excess         degeneracy.json w_channel.w_niche_mi_excess
  within-type share of w     degeneracy.json w_channel.w_var_fraction_across_cells
  probe comp MLP share       validation/probe_blocks.json mlp.comp.excess
                             (1 - exp(-2 excess), in %)
  held-out recon             degeneracy.json recon_val_targets
  transport                  transport/transport.json summary.<tier>.
                             counterfactual_of_ceiling (cell-split ceiling;
                             extrapolation_trusted = the published headline,
                             extrapolation = all panels)
  atlas                      atlas/atlas.json: rank; cross_seed programme-0
                             axis cosine and shift overlap (mean over the other
                             two seeds of the same arm); programme 0's driver
                             partial R^2 (composition, Phi PCs) and hallmark
                             labels; label_recurrence.recurring
  noimgL vs finalL axis      |cos| of programme 0 of noimgL_s<k> with its best
                             match in finalL_s<k> (atlas.cross_seed on the two
                             programs.npy, all genes; computed here)

    python scripts/noimg_readout.py <ROOT>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

DATASETS = (("gse315411_pdltma06_11_prime_solo", "GSE core (TMA)"),
            ("xenium_prime_ovarian_cancer_ffpe", "ovarian FFPE"),
            ("xenium_prime_human_lung_cancer_ffpe", "lung FFPE"),
            ("xenium_prime_human_ovary_ff", "FF"))
ARMS = ("finalL", "noimgL")
SEEDS = (0, 1, 2)
# key, label, digits, predicted direction (-1 falls; None: none set). The
# prediction names three reads: niche information in w, relocation and
# programme stability -- each predicted to fall, most on FF, least on GSE.
READS = (("mi_w", "I(niche; w) excess", 4, -1),
         ("tr_trusted", "transport, fraction of ceiling (trusted tier)", 3, -1),
         ("tr_all", "transport, fraction of ceiling (all panels)", 3, -1),
         ("atlas_cos", "atlas cross-seed programme-0 cosine", 3, -1),
         ("atlas_overlap", "atlas cross-seed shift overlap", 3, -1),
         ("nmi_z", "NMI z", 3, None),
         ("cyc_z", "cycle R2 z (q90)", 3, None),
         ("share_w", "within-type share of w's variance", 3, None),
         ("probe", "probe comp MLP share (%)", 2, None),
         ("recon", "held-out recon", 4, None),
         ("atlas_rank", "atlas rank", 2, None),
         ("drv_comp", "programme 0 driver partial R2: composition", 3, None),
         ("drv_phi", "programme 0 driver partial R2: Phi PCs", 3, None))


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def run_dir(ds: str, run: str) -> Path:
    return Path(f"data/datasets/{ds}/runs/{run}")


def reads(ds: str, run: str) -> dict:
    d = run_dir(ds, run)
    g = load(d / "degeneracy.json") or {}
    b, w = g.get("battery") or {}, g.get("w_channel") or {}
    pb = ((load(d / "validation" / "probe_blocks.json") or {}).get("mlp") or {}).get("comp") or {}
    ex = pb.get("excess")
    t = ((load(d / "transport" / "transport.json") or {}).get("summary")) or {}
    a = load(d / "atlas" / "atlas.json") or {}
    cos, over = [], []
    for other in (a.get("cross_seed") or {}).values():
        for e in other.get("axis_cosine") or []:   # envelope_tables.atlas_row's read
            if e and e.get("program") == 0:
                cos.append(e["cosine"])
        so = other.get("shift_overlap") or {}
        if so.get("this_inside_other") is not None:
            over.append(so["this_inside_other"])
    p0 = (a.get("programs") or [{}])[0]
    part = (p0.get("drivers") or {}).get("partial") or {}
    m = load(d / "metrics.json") or {}

    def ceil(tier):
        v = (t.get(tier) or {}).get("counterfactual_of_ceiling")
        return v if v is not None and math.isfinite(v) else None

    return {"mi_w": w.get("w_niche_mi_excess"),
            "tr_trusted": ceil("extrapolation_trusted"), "tr_all": ceil("extrapolation"),
            "atlas_cos": float(np.mean(cos)) if cos else None,
            "atlas_overlap": float(np.mean(over)) if over else None,
            "nmi_z": g.get("nmi_targets"), "cyc_z": b.get("cycle_r2_z_q90"),
            "share_w": w.get("w_var_fraction_across_cells"),
            "probe": None if ex is None else 100 * (1 - math.exp(-2 * ex)),
            "recon": g.get("recon_val_targets"),
            "atlas_rank": a.get("rank"),
            "drv_comp": part.get("composition_y"), "drv_phi": part.get("phi_pcs"),
            "_hallmarks0": [h["hallmark"] for h in p0.get("hallmarks") or []
                            if h.get("significant")],
            "_recurring": (a.get("label_recurrence") or {}).get("recurring"),
            "_n_trusted": (t.get("extrapolation_trusted") or {}).get("n_panels"),
            "_best_epoch": (m.get("best") or {}).get("epoch"),
            "_last_epoch": m.get("last_epoch"), "_minutes": m.get("minutes"),
            "_dead": m.get("dead_w_channel") or w.get("dead_context_channel"),
            "_no_image": (load(d / "config.json") or {}).get("no_image", False)}


def cross_arm_cosine(ds: str, seed: int):
    """|cos| of noimgL_s<seed>'s programme 0 with its best match in finalL_s<seed>."""
    from discell.model.atlas import cross_seed

    a = run_dir(ds, f"noimgL_s{seed}") / "atlas" / "programs.npy"
    f = run_dir(ds, f"finalL_s{seed}") / "atlas" / "programs.npy"
    if not (a.exists() and f.exists()):
        return None
    e = cross_seed(np.load(a), np.load(f))["axis_cosine"][0]
    return None if e is None else e["cosine"]


def summary(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)}


def fmt(s, n=3):
    if s is None:
        return "--"
    return (f"{s['mean']:.{n}f} ({s['min']:.{n}f} to {s['max']:.{n}f})"
            + ("" if s["n"] == 3 else f" [n={s['n']}]"))


def main(argv) -> int:
    root = Path(argv[1])
    L = ["# No-image ablation: noimgL against finalL", "",
         "Queue `scripts/queue_2026-10-01_noimg.sh`; written by `scripts/noimg_readout.py`. "
         "Final configuration with `--no-image` (Phi removed from the context: c = GAT over "
         "neighbour composition + the isolated flag; the adversary's e_Phi target unchanged), "
         "seeds 0-2 at finalL's splits, reads at best.pt under the Unassigned mask. "
         "Cells: 3-seed mean (min to max).", "",
         "**Prediction set in advance** (devlog \"No-image ablation at the final configuration "
         "(motivation, 2026-10-01; author)\"): removing the image costs most on FF (niche "
         "information in w, relocation, programme stability) and least on the TMA core. If "
         "nothing moves beyond finalL's seed range, the image is optional at this "
         "configuration, and that is reported as found.", "",
         "\"Beyond finalL's seed range\": **mean** = noimgL's 3-seed mean outside finalL's "
         "[min, max]; **disjoint** = noimgL's whole range outside finalL's range (the stricter). "
         "The three predicted reads are mapped to I(niche; w) excess (niche information in w), "
         "transport fraction of ceiling (relocation) and the atlas cross-seed programme-0 "
         "cosine / shift overlap (programme stability), each predicted to fall.", ""]
    table, verdicts, cost = {}, [], {}
    for ds, name in DATASETS:
        per = {a: [reads(ds, f"{a}_s{s}") for s in SEEDS] for a in ARMS}
        table[ds] = per
        L += [f"## {name} (`{ds}`)", "",
              "| read | finalL | noimgL | change of mean | beyond finalL's range (mean) "
              "| disjoint | predicted | verdict |",
              "|---|---|---|---|---|---|---|---|"]
        cost[name] = []
        for key, label, n, pred in READS:
            f = summary([r[key] for r in per["finalL"]])
            o = summary([r[key] for r in per["noimgL"]])
            if f is None or o is None:
                L.append(f"| {label} | {fmt(f, n)} | {fmt(o, n)} | -- | -- | -- | -- | not read |")
                continue
            beyond = o["mean"] < f["min"] or o["mean"] > f["max"]
            disjoint = o["max"] < f["min"] or o["min"] > f["max"]
            direction = +1 if o["mean"] > f["max"] else -1 if o["mean"] < f["min"] else 0
            delta = o["mean"] - f["mean"]
            ptxt = {-1: "falls", None: "none set"}[pred]
            if pred is None:
                v = "moved" if beyond else "within range"
            elif not beyond:
                v = "within range"
            elif direction == pred:
                v = "as predicted" + (" (disjoint)" if disjoint else " (mean only)")
            else:
                v = "OPPOSITE to prediction" + (" (disjoint)" if disjoint else " (mean only)")
            if pred is not None:
                verdicts.append((name, label, beyond, disjoint, v))
                rel = delta / abs(f["mean"]) if f["mean"] else float("nan")
                cost[name].append((label, rel, beyond and direction == pred))
            L.append(f"| {label} | {fmt(f, n)} | {fmt(o, n)} | {delta:+.{n}f} | "
                     f"{'yes' if beyond else 'no'} | {'yes' if disjoint else 'no'} | "
                     f"{ptxt} | {v} |")
        xs = [cross_arm_cosine(ds, s) for s in SEEDS]
        xs_txt = ", ".join("--" if x is None else f"{x:.3f}" for x in xs)
        L += ["", f"noimgL_s<k> vs finalL_s<k>, programme-0 axis |cos| (same seed, all genes): "
              f"{xs_txt}. For scale: finalL's own cross-seed cosine above.", "",
              "Programme 0's significant hallmark labels per seed, and the labels recurring "
              "across the arm's seeds:", ""]
        for a in ARMS:
            for s, r in zip(SEEDS, per[a]):
                L.append(f"* {a}_s{s}: {', '.join(r['_hallmarks0']) or 'none'}"
                         + (f" (trusted transport panels: {r['_n_trusted']})"
                            if r["_n_trusted"] is not None else ""))
            rec = next((r["_recurring"] for r in per[a] if r["_recurring"] is not None), None)
            L.append(f"* {a} recurring (>= 2 of 3 seeds): "
                     + ("--" if rec is None else ", ".join(rec) or "none"))
        L += ["", "Fits: " + "; ".join(
            f"{a}_s{s} best/last epoch {r['_best_epoch']}/{r['_last_epoch']}, "
            f"{(r['_minutes'] or 0):.1f} min, no_image {r['_no_image']}"
            + (", **DEAD w channel**" if r["_dead"] else "")
            for a in ARMS for s, r in zip(SEEDS, per[a]) if r["_best_epoch"] is not None), ""]

    L += ["## Against the prediction", ""]
    moved = [v for v in verdicts if v[2]]
    if not moved:
        L.append("No predicted read moved beyond finalL's seed range on any section: by the "
                 "rule set in advance, the image is optional at this configuration.")
    else:
        for name, label, _, _, v in moved:
            L.append(f"* {name}: {label} -- {v}")
        L += ["", f"{sum(1 for v in moved if v[3])} of {len(verdicts)} predicted "
              f"(section, read) pairs moved with disjoint ranges, {len(moved)} by the mean; "
              f"{sum(1 for v in moved if 'OPPOSITE' in v[4])} of the moves are opposite "
              "to the prediction."]
    L += ["", "Ordering (\"most on FF, least on the TMA core\"): per section, the predicted "
          "reads' relative change of the mean (noimgL - finalL) / |finalL|, and how many "
          "fell beyond finalL's range.", "",
          "| section | " + " | ".join(lbl for lbl, _, _, p in
                                      [(r[1], 0, 0, r[3]) for r in READS] if p is not None)
          + " | fell beyond range |",
          "|---|" + "---|" * (sum(1 for r in READS if r[3] is not None) + 1)]
    for name, rows in cost.items():
        by = {lbl: (rel, hit) for lbl, rel, hit in rows}
        cells = [("--" if lbl not in by else f"{100 * by[lbl][0]:+.1f}%")
                 for _, lbl, _, p in READS if p is not None]
        L.append(f"| {name} | " + " | ".join(cells)
                 + f" | {sum(1 for _, _, h in rows if h)} of {len(rows)} |")
    missing = [f"{ds}/{a}_s{s}" for ds, _ in DATASETS for a in ARMS for s in SEEDS
               if table[ds][a][s]["nmi_z"] is None]
    failed = (sorted(p.name for p in (root / "failed").glob("*"))
              if (root / "failed").exists() else [])
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
