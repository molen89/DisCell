#!/usr/bin/env python3
"""READOUT.md of the RECOMB additions queue (devlog "RECOMB story map
approved; three additions (motivation, 2026-10-01; author)", items 1-2):
the results against the predictions fixed in that entry.

    uv run python scripts/recomb_additions_readout.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("scripts/logs/recomb_additions_2026-10-01")
SPILL = Path("data/datasets/synthetic_smoke/experiments/planted_spillover.json")
DATASETS = {"gse315411_pdltma06_11_prime_solo": "GSE core",
            "xenium_prime_ovarian_cancer_ffpe": "ovarian FFPE",
            "xenium_prime_human_lung_cancer_ffpe": "lung FFPE",
            "xenium_prime_human_ovary_ff": "ovarian FF"}
KEYS = ("discell", "regression_log", "regression_rate")


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _f(v, d=3):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def _rng(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "–", None
    return (f"{np.mean(v):.3f} [{min(v):.3f}, {max(v):.3f}]"
            + ("" if len(v) == 3 else f" (n={len(v)})"), (min(v), max(v)))


def spill_section() -> list[str]:
    res = _load(SPILL)
    lines = ["## 1. Planted spill-over positive control", "",
             "**Prediction (fixed in advance):** the spill-only contrast's κ* "
             "is the first grid point ≥ κ_true (0.1 → 0.1, 0.2 → 0.2); the "
             "genuine response contrast survives the grid. If κ* does not "
             "track κ_true, report it as found and present the breakdown "
             "point as a lower-power check.", ""]
    if res is None:
        return lines + ["Not available: `planted_spillover.json` is missing "
                        "(see the queue log)."]
    from discell.experiments.breakdown import _kstar

    lines += ["| κ_true | fits | spill-only κ* | predicted | as predicted | "
              "genuine κ* | survives the grid |", "|---|---|---|---|---|---|---|"]
    for kt, w in res["worlds"].items():
        v = w["verdict"]
        m = w["table"]["members"]
        sp = m.get("spill_cf_minus_leak")
        re = m.get("response_cf_minus_leak")
        if float(kt) == 0:
            pred, ok = "no finding (null world)", v.get("spill_null_no_finding")
        else:
            pred = f"breaks at {v['predicted_spill_kappa_star']:g}"
            ok = v.get("spill_tracks_kappa_true")
        lines.append(
            f"| {kt} | {w['n_fits']}/18 | {_kstar(sp) if sp else '–'} | {pred} "
            f"| {'yes' if ok else 'no'} | {_kstar(re) if re else '–'} | "
            f"{'yes' if v.get('response_survives_grid') else 'no'} |")
    hits = [w["verdict"].get("spill_tracks_kappa_true")
            for kt, w in res["worlds"].items() if float(kt) > 0]
    surv = [w["verdict"].get("response_survives_grid")
            for w in res["worlds"].values()]
    lines += ["", f"**Read:** spill-only κ* tracks κ_true in "
              f"{sum(bool(h) for h in hits)}/{len(hits)} planted worlds; the "
              f"genuine contrast survives the grid in "
              f"{sum(bool(s) for s in surv)}/{len(surv)} worlds (null "
              "included). Details and per-κ intervals: "
              "`data/datasets/synthetic_smoke/experiments/planted_spillover.md`."]
    return lines


def regression_section() -> list[str]:
    from discell import paths

    lines = ["", "## 2. Plain-regression reference for relocation", "",
             "**No outcome pre-judged** (devlog): if the regression matches "
             "DISCELL on relocation, DISCELL's advantage lies in the separation "
             "(z clean, spill-over modelled), not in relocation accuracy.", "",
             "Fraction of ceiling, 3-seed mean [min, max] over finalL_s0–s2. "
             "'all panels' / 'trusted' are the published tiers (extrapolation / "
             "extrapolation_trusted). Regression `log` = as specified (log1p at "
             "the median depth); `rate` = on the read's own scale.", "",
             "| section | tier | DISCELL | regression (log) | regression (rate) "
             "| best regression vs DISCELL |", "|---|---|---|---|---|---|"]
    readas = []
    for ds, short in DATASETS.items():
        t = _load(paths.dataset(ds).root / "experiments"
                  / "transport_regression_reference.json")
        if not t or not t.get("runs"):
            lines.append(f"| {short} | – | not available | | | |")
            continue
        runs = t["runs"]
        for tier, label in (("extrapolation", "all panels"),
                            ("extrapolation_trusted", "trusted")):
            cells, ranges = [], {}
            for k in KEYS:
                txt, rg = _rng([(r["mean_read"].get(tier) or {}).get(k)
                                for r in runs.values()])
                cells.append(txt)
                ranges[k] = rg
            d = ranges["discell"]
            best = max((ranges[k] for k in KEYS[1:] if ranges[k]),
                       key=lambda r: r[1], default=None)
            if d is None or best is None:
                verdict = "–"
            elif best[0] > d[1]:
                verdict = "regression higher (ranges disjoint)"
            elif best[1] < d[0]:
                verdict = "DISCELL higher (ranges disjoint)"
            else:
                verdict = "ranges overlap"
            lines.append(f"| {short} | {label} | " + " | ".join(cells)
                         + f" | {verdict} |")
        for run, r in runs.items():
            a = r["read_a"]
            readas.append((short, run, a))
    lines += ["", "### Read A (pairwise panels)", "",
              "Count level on raw target cells (the regression moves raw "
              "source compositions; DISCELL's like-for-like number is its "
              "count-matched read). DISCELL's published Read A scores against "
              "its own decode of the target cells, on which a model-free "
              "predictor cannot be scored; shown for orientation only. Cells: "
              "median gap closed (minus the type-mean predictor's).", "",
              "| section | run | regression (log) | regression (rate) | "
              "DISCELL count-matched | DISCELL published (own) |",
              "|---|---|---|---|---|---|"]
    for short, run, a in readas:
        def gm(e):
            return (f"{_f(e.get('median_gap_closed'))} "
                    f"({_f(e.get('minus_type_mean'))})")
        lines.append(f"| {short} | {run} | {gm(a['regression_log'])} | "
                     f"{gm(a['regression_rate'])} | "
                     f"{gm(a['discell_count_matched'])} | "
                     f"{gm(a['discell_own_published'])} |")
    lines += ["", "Per-dataset tables: `data/datasets/<id>/experiments/"
              "transport_regression_reference.md`; per run: "
              "`runs/<run>/transport/regression_reference.json`."]
    return lines


def main() -> int:
    failed = sorted(p.name for p in (ROOT / "failed").glob("*")) \
        if (ROOT / "failed").exists() else []
    done = len(list((ROOT / "done").glob("*"))) if (ROOT / "done").exists() else 0
    lines = ["# RECOMB additions 1–2: readout", "",
             "Spec and predictions: devlog \"RECOMB story map approved; three "
             "additions (motivation, 2026-10-01; author)\", items 1 and 2. "
             f"Queue: {done} steps done, {len(failed)} failed"
             + (f" ({', '.join(failed)})" if failed else "") + ".", ""]
    lines += spill_section()
    lines += regression_section()
    (ROOT / "READOUT.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "READOUT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
