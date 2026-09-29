#!/usr/bin/env python3
"""Published transport read vs the held-out-tiles sensitivity row.

Devlog "Transport scored on held-out tiles only (sensitivity row; motivation,
2026-09-29; author chose option b)". Per dataset x seed (DisCell finalL_s*)
and per dataset (Cellina's counterfactual, mirroring finalL_s0): the mean
read's fraction of ceiling (extrapolation tier, trusted and all panels) with
its half-tile 95 % CI, the cells the panels score, and a flag for whether the
held-out-tiles value falls outside the published read's CI. Read A (own
target, tile CI) and the own-target twin margin are listed beside, unflagged.

Reads only: ``runs/<run>/{transport/,bootstrap_ci.json}`` against
``runs/<run>/heldout_tiles/{transport/,bootstrap_ci.json}`` (and the same
under Cellina's ``<ds>_lineage_cf``). Missing files print as ``--``.

    python scripts/transport_heldout_table.py --out <md path>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("gse315411_pdltma06_11_prime_solo", "xenium_prime_human_ovary_ff",
            "xenium_prime_ovarian_cancer_ffpe",
            "xenium_prime_human_lung_cancer_ffpe")
SHORT = {"gse315411_pdltma06_11_prime_solo": "GSE solo",
         "xenium_prime_human_ovary_ff": "ovary FF",
         "xenium_prime_ovarian_cancer_ffpe": "ovarian FFPE",
         "xenium_prime_human_lung_cancer_ffpe": "lung FFPE"}
RUNS = ("finalL_s0", "finalL_s1", "finalL_s2")
CELLINA = Path("/home/rmolen/github/DisCell-baselines/results/cellina")
TIERS = (("transport_of_ceiling_trusted", "extrapolation_trusted", "trusted"),
         ("transport_of_ceiling", "extrapolation", "all panels"))


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def read(root: Path) -> dict | None:
    """The numbers of one read (a run directory or its heldout_tiles/)."""
    mean = _load(root / "transport" / "transport.json")
    ci = (_load(root / "bootstrap_ci.json") or {}).get("reads") or {}
    if mean is None:
        return None
    out = {"panels": len(mean.get("panels", [])),
           "scored_cells": mean.get("scored_cells")}
    # cells the kept panels score: each (niche, type) group's n_test once
    groups = {}
    for p in mean.get("panels", []):
        for niche, n in zip(p["pair"], p["n_test"]):
            groups[(niche, p["type"])] = n
    out["n_cells_panels"] = int(sum(groups.values()))
    for key, tier, _ in TIERS:
        summary = (mean.get("summary") or {}).get(tier) or {}
        entry = ci.get(key) or {}
        out[key] = {"value": summary.get("counterfactual_of_ceiling"),
                    "n_panels": summary.get("n_panels", 0),
                    "ci95": entry.get("ci95"), "n_cells": entry.get("n_cells"),
                    "reproduces": entry.get("reproduces")}
    entry = ci.get("readA_gap_own") or {}
    out["readA_gap_own"] = {"value": entry.get("estimate"),
                            "ci95": entry.get("ci95")}
    twins = _load(root / "transport" / "transport_twins.json") or {}
    out["twin_margin_own"] = (((twins.get("summary_own") or {}).get("pairwise")
                               or {}).get("median_twin_margin"))
    return out


def _num(v) -> bool:
    return v is not None and not math.isnan(v)


def fmt(v, ci=None) -> str:
    if not _num(v):
        return "--"
    text = f"{v:.3f}"
    if ci and all(_num(c) for c in ci):
        text += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return text


def outside(published: dict | None, heldout: dict | None, key: str) -> str:
    """Is the held-out-tiles value outside the published read's CI?"""
    if not published or not heldout:
        return "--"
    ci, v = published[key]["ci95"], heldout[key]["value"]
    if not ci or not _num(v) or not all(_num(c) for c in ci):
        return "--"
    return "**OUTSIDE**" if (v < ci[0] or v > ci[1]) else "inside"


def rows() -> list[dict]:
    out = []
    for ds in DATASETS:
        runs = REPO / "data" / "datasets" / ds / "runs"
        for run in RUNS:
            out.append({"dataset": ds, "method": "DisCell", "run": run,
                        "published": read(runs / run),
                        "heldout": read(runs / run / "heldout_tiles")})
        cdir = CELLINA / f"{ds}_lineage_cf"
        out.append({"dataset": ds, "method": "Cellina", "run": "cf (mirrors finalL_s0)",
                    "published": read(cdir),
                    "heldout": read(cdir / "heldout_tiles")})
    return out


def render(table: list[dict]) -> str:
    lines = ["# Transport: published read vs held-out tiles only", "",
             ("Devlog 2026-09-29, \"Transport scored on held-out tiles only\" "
             "(author's option b). *published*: fold 0 of the prepare tiles "
             "scored, folds 1-4 for the model quantities (~85 % of fold 0 in "
             "the model's training tiles). *held-out tiles*: only cells of the "
             "model's held-out tiles (val_tiles) scored, every model quantity "
             "from its training tiles. Same niches, panels rule, trusted tier, "
             "cell-split ceiling, half-tile CIs and Unassigned mask. Fraction "
             "of ceiling = mean read, extrapolation tier; 95 % half-tile CI in "
             "brackets; *flag* = the held-out-tiles value outside the published "
             "CI (the pre-set rule: outside on a primary section -> the author "
             "considers switching the headline read). *cells* = cells the kept "
             "panels score (each niche x type group once)."), ""]
    for key, _, label in TIERS:
        lines += [f"## Fraction of ceiling, {label}", "",
                  ("| dataset | method | run | published | panels | cells | "
                   "held-out tiles | panels | cells | flag |"),
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in table:
            p, h = r["published"], r["heldout"]
            cells = []
            for x in (p, h):
                if x is None:
                    cells += ["--", "--", "--"]
                else:
                    cells += [fmt(x[key]["value"], x[key]["ci95"]),
                              str(x[key]["n_panels"]), str(x["n_cells_panels"])]
            lines.append(f"| {SHORT[r['dataset']]} | {r['method']} | {r['run']} | "
                         + " | ".join(cells) + f" | {outside(p, h, key)} |")
        lines.append("")
    lines += ["## Read A (own target, tile CI) and twin margin (own target)", "",
              ("| dataset | method | run | Read A published | Read A held-out "
               "tiles | twin margin published | twin margin held-out tiles |"),
              "|---|---|---|---|---|---|---|"]
    for r in table:
        p, h = r["published"] or {}, r["heldout"] or {}
        a = [fmt((x.get("readA_gap_own") or {}).get("value"),
                 (x.get("readA_gap_own") or {}).get("ci95")) for x in (p, h)]
        t = [fmt(x.get("twin_margin_own")) for x in (p, h)]
        lines.append(f"| {SHORT[r['dataset']]} | {r['method']} | {r['run']} | "
                     f"{a[0]} | {a[1]} | {t[0]} | {t[1]} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="markdown path (a .json with the numbers beside)")
    args = parser.parse_args(argv)
    table = rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(table))
    args.out.with_suffix(".json").write_text(json.dumps(table, indent=2,
                                                        default=float))
    print(args.out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
