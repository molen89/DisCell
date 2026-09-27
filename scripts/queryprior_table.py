#!/usr/bin/env python3
"""The read-out for the query/prior ablations (devlog pre-registration, 2026-09-23).

Per arm (control, query type_free, query image, prior type-free) over seeds
0 1 2 on the ovarian core: the seed mean/min/max of every read the
pre-registration names -- the battery guards, cycle_z / cycle_w, the transport
mean read (fraction of ceiling, beats-both), Read A / Read B own-target,
atlas effective rank and cross-seed cosine, MIG / MIC with the floor, and the
attention read -- then the decision text applied mechanically.

    uv run python scripts/queryprior_table.py [--tag qp]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data" / "datasets"
OV = "xenium_prime_ovarian_cancer_ffpe"
ARMS = ["control", "query_type_free", "query_image", "prior_type_free"]
ARM_FLAGS = {"control": "", "query_type_free": "--query type_free",
             "query_image": "--query image",
             "prior_type_free": "--prior-type-free"}
SEEDS = [0, 1, 2]
CYCLE_W_GUARD = 0.02
#: the transport block the reads are taken from: the honest target
BLOCK = "extrapolation"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _dig(d, *keys, default=float("nan")):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def read_run(run: str) -> dict | None:
    """Every number one fit contributes, or None when it has not landed."""
    d = ROOT / OV / "runs" / run
    metrics = _load(d / "metrics.json")
    if metrics is None:
        return None
    final = metrics["final"]
    out = {
        "run": run,
        "recon": final["recon_val"],
        "nmi": final["nmi"],
        "mirror_r2": _dig(final, "mirror", "r2"),
        "probe_delta_ce": _dig(final, "probe", "delta_ce"),
        "probe_noise_floor": _dig(final, "probe", "noise_floor"),
        "cycle_z": _dig(final, "cycle", "z", "r2_pooled"),
        "cycle_w": _dig(final, "cycle", "w", "r2_pooled"),
        "kl_w": float(np.sum(final["kl_w_per_dim"])),
        "mi_ratio": _dig(final, "degeneracy", "mi_ratio"),
    }
    out["probe_excess"] = out["probe_delta_ce"] - out["probe_noise_floor"]
    out["cycle_w_ok"] = bool(out["cycle_w"] <= CYCLE_W_GUARD)

    guard = _load(d / "degeneracy.json") or {}
    out["w_guard_dead"] = _dig(guard, "w_channel", "dead_context_channel",
                               default=None)

    # -- transport: the mean read, then the own-target distribution read ----
    for tag, stem in (("a", "transport"), ("b", "transport_tumour-band")):
        mean = _load(d / "transport" / f"{stem}.json")
        if mean:
            block = _dig(mean, "summary", BLOCK, default={})
            out[f"read_{tag}_of_ceiling"] = block.get("counterfactual_of_ceiling",
                                                      float("nan"))
            out[f"read_{tag}_beats_both"] = block.get("full_beats_both",
                                                      float("nan"))
            out[f"read_{tag}_n_panels"] = block.get("n_panels", float("nan"))
        dist = _load(d / "transport" / f"{stem}_distribution.json")
        if dist:
            out[f"read_{tag}_own_gap_closed"] = _dig(
                dist, "summary_model_own", "pairwise", "median_gap_closed")
            out[f"read_{tag}_own_beats_type_mean"] = _dig(
                dist, "summary_model_own", "pairwise", "n_beats_type_mean")

    # -- the atlas: effective rank and the cross-seed agreement -------------
    atlas = _load(d / "atlas" / "atlas.json")
    if atlas:
        out["atlas_rank"] = atlas.get("rank", float("nan"))
        cosines = [m["cosine"] for other in atlas.get("cross_seed", {}).values()
                   for m in other.get("axis_cosine", []) if m]
        out["atlas_cross_seed_cosine"] = (float(np.mean(cosines)) if cosines
                                          else float("nan"))
        overlaps = [_dig(o, "shift_overlap", "this_inside_other")
                    for o in atlas.get("cross_seed", {}).values()]
        overlaps = [v for v in overlaps if v == v]
        out["atlas_shift_overlap"] = (float(np.mean(overlaps)) if overlaps
                                      else float("nan"))

    # -- MIG / MIC with the floor (external_criteria mi-quadrant) ----------
    mi = _load(ROOT / OV / "experiments" / f"external_mi_quadrant_{run}.json")
    if mi:
        knn = _dig(mi, "sources", "kmeans", "knn", default={})
        out["MIG"] = knn.get("MIG", float("nan"))
        out["MIC"] = knn.get("MIC", float("nan"))
        out["MIG_floor_corrected"] = knn.get("MIG_floor_corrected",
                                             float("nan"))

    # -- the attention read (arm (ii)'s, run on every arm as its reference) -
    attn = _load(d / "attention_read.json")
    if attn:
        out["attn_entropy"] = attn.get("entropy_mean", float("nan"))
        spread = [abs(v) for v in attn.get("entropy_by_type", {}).values()
                  for v in [v["entropy_mean"]]]
        out["attn_entropy_type_spread"] = (float(max(spread) - min(spread))
                                           if spread else float("nan"))
        corrs = [abs(v) for v in attn.get("alpha_vs_dst_phi_pc", []) if v == v]
        out["attn_alpha_phi_corr_max"] = float(max(corrs)) if corrs else float("nan")
    return out


FIELDS = ["recon", "nmi", "probe_delta_ce", "probe_noise_floor",
          "probe_excess", "mirror_r2", "cycle_z", "cycle_w", "kl_w",
          "mi_ratio", "read_a_of_ceiling", "read_a_beats_both",
          "read_a_own_gap_closed", "read_b_of_ceiling",
          "read_b_own_gap_closed", "atlas_rank", "atlas_cross_seed_cosine",
          "atlas_shift_overlap", "MIG", "MIC", "MIG_floor_corrected",
          "attn_entropy", "attn_entropy_type_spread",
          "attn_alpha_phi_corr_max"]


def summarise(runs: list[dict]) -> dict:
    out = {"n_seeds": len(runs), "seeds_done": [r["run"] for r in runs],
           "cycle_w_ok": sum(r["cycle_w_ok"] for r in runs),
           "w_guard_dead": sum(bool(r.get("w_guard_dead")) for r in runs)}
    for field in FIELDS:
        vals = [r[field] for r in runs if field in r and r[field] == r[field]]
        if vals:
            out[field] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                          "max": float(max(vals)), "n": len(vals)}
    return out


def collect(tag: str) -> dict:
    cells = {}
    for arm in ARMS:
        runs = [r for r in (read_run(f"{tag}_{arm}_s{s}") for s in SEEDS)
                if r is not None]
        if runs:
            cells[arm] = {"runs": runs, **summarise(runs)}
    return cells


def inside_envelope(value: float, control: dict, field: str,
                    higher_is_better: bool | None) -> bool:
    """Inside the control's seed envelope, or (if a direction is named) better
    than all of it. ``higher_is_better=None`` demands strictly inside."""
    if control is None or field not in control:
        return False
    lo, hi = control[field]["min"], control[field]["max"]
    if lo <= value <= hi:
        return True
    if higher_is_better is None:
        return False
    return value > hi if higher_is_better else value < lo


def beats(value: float, control: dict, field: str) -> bool | None:
    """Strictly above every control seed."""
    if control is None or field not in control or value != value:
        return None
    return bool(value > control[field]["max"])


#: the guards an arm must hold to be "indistinguishable" / "intact";
#: value = the direction that counts as better than the envelope
GUARDS = {"recon": True, "nmi": True, "mirror_r2": False,
          "probe_delta_ce": False, "cycle_z": True, "mi_ratio": None}
#: what the transport read scores -- the per-type amplitude arm (iii) should lose
TRANSPORT = ["read_a_of_ceiling", "read_a_own_gap_closed",
             "read_b_own_gap_closed"]


def guard_report(arm: dict, control: dict) -> tuple[bool | None, list[str]]:
    missing = [f for f in GUARDS if f not in arm]
    if missing or arm is None:
        return None, [f"missing {', '.join(missing)}"]
    bad = [f for f, hib in GUARDS.items()
           if not inside_envelope(arm[f]["mean"], control, f, hib)]
    ok = (not bad) and arm["cycle_w_ok"] == arm["n_seeds"]
    reasons = ([f"outside the control envelope on {', '.join(bad)}"] if bad
               else ["every battery guard inside the control envelope"])
    if arm["cycle_w_ok"] != arm["n_seeds"]:
        reasons.append(f"cycle_w guard {arm['cycle_w_ok']}/{arm['n_seeds']}")
    return ok, reasons


def decide(cells: dict) -> dict:
    """The pre-registration's decision text, applied mechanically."""
    control = cells.get("control")
    out = {}
    for arm in ARMS[1:]:
        cell = cells.get(arm)
        if cell is None or control is None:
            out[arm] = {"verdict": "undecided",
                        "reasons": ["arm or control not fitted yet"]}
            continue
        guards_ok, reasons = guard_report(cell, control)
        transport_in = {f: inside_envelope(cell[f]["mean"], control, f, True)
                        for f in TRANSPORT if f in cell}
        transport_up = {f: beats(cell[f]["mean"], control, f)
                        for f in TRANSPORT if f in cell}
        incomplete = (cell["n_seeds"] < len(SEEDS)
                      or control["n_seeds"] < len(SEEDS)
                      or guards_ok is None or not transport_in)
        entry = {"guards_ok": guards_ok, "reasons": reasons,
                 "transport_inside_or_better": transport_in,
                 "transport_beats_control": transport_up,
                 "n_seeds": cell["n_seeds"]}
        if incomplete:
            entry["verdict"] = "undecided"
            entry["reasons"] = reasons + ["grid incomplete "
                                          f"({cell['n_seeds']}/{len(SEEDS)} seeds"
                                          " here, "
                                          f"{control['n_seeds']}/{len(SEEDS)}"
                                          " control)"]
        elif arm == "query_type_free":
            # "(i) indistinguishable from control -> adopt"
            same = guards_ok and all(transport_in.values())
            entry["verdict"] = "ADOPT" if same else "keep the pinned query"
            entry["reasons"] = reasons + [
                "transport inside/above the control envelope on "
                + ", ".join(f for f, v in transport_in.items() if v)
                if any(transport_in.values()) else
                "transport below the control envelope"]
        elif arm == "query_image":
            # "adopt only if it beats control on transport with guards intact
            #  and the attention read is interpretable; otherwise report"
            wins = [f for f, v in transport_up.items() if v]
            if guards_ok and wins:
                entry["verdict"] = ("ADOPT IF the attention read is "
                                    "interpretable (a human call)")
            else:
                entry["verdict"] = "report only"
            entry["reasons"] = reasons + [
                ("beats every control seed on " + ", ".join(wins)) if wins
                else "does not beat the control on any transport read",
                "attention read: entropy "
                f"{cell.get('attn_entropy', {}).get('mean', float('nan')):.3f} "
                "nats, |corr(alpha, Phi PC)| max "
                f"{cell.get('attn_alpha_phi_corr_max', {}).get('mean', float('nan')):.3f}"
                " (control: entropy "
                f"{control.get('attn_entropy', {}).get('mean', float('nan')):.3f}"
                ", corr "
                f"{control.get('attn_alpha_phi_corr_max', {}).get('mean', float('nan')):.3f})"]
        else:   # prior_type_free
            # "expected to lose transport per-type amplitude -> stays an
            #  ablation; if it does not lose, w's definition in the paper
            #  changes to a universal response"
            lost = [f for f, v in transport_in.items() if not v]
            if lost:
                entry["verdict"] = ("stays an ablation: loses the per-type "
                                    "amplitude on " + ", ".join(lost))
            else:
                entry["verdict"] = ("DOES NOT LOSE the transport read -- w's "
                                    "definition in the paper becomes a "
                                    "universal response"
                                    + ("" if guards_ok else
                                       ", but the guards are not intact"))
            entry["reasons"] = reasons
        out[arm] = entry
    return out


def _cell(arm: dict, field: str, fmt: str = "{:.4g}") -> str:
    if field not in arm:
        return "--"
    v = arm[field]
    return (fmt + " [" + fmt + ", " + fmt + "]").format(v["mean"], v["min"],
                                                        v["max"])


TABLES = [
    ("Battery guards", ["recon", "nmi", "probe_delta_ce", "probe_noise_floor",
                        "mirror_r2", "mi_ratio"]),
    ("Cycle", ["cycle_z", "cycle_w", "kl_w"]),
    ("Transport (mean read, %s block; own-target distribution read)" % BLOCK,
     ["read_a_of_ceiling", "read_a_beats_both", "read_a_own_gap_closed",
      "read_b_of_ceiling", "read_b_own_gap_closed"]),
    ("Atlas and MIG/MIC", ["atlas_rank", "atlas_cross_seed_cosine",
                           "atlas_shift_overlap", "MIG", "MIC",
                           "MIG_floor_corrected"]),
    ("Attention read", ["attn_entropy", "attn_entropy_type_spread",
                        "attn_alpha_phi_corr_max"]),
]


def render(cells: dict, ruling: dict) -> str:
    lines = ["# Query / prior ablations -- ovarian core, 3 seeds, 200/20", "",
             "Cells are `seed mean [min, max]`. Arms: "
             + ", ".join(f"`{a}` (`{ARM_FLAGS[a] or 'pinned'}`)" for a in ARMS),
             ""]
    for title, cols in TABLES:
        lines += [f"## {title}", "",
                  "| arm | seeds | " + " | ".join(cols) + " |",
                  "|" + "---|" * (len(cols) + 2)]
        for arm in ARMS:
            cell = cells.get(arm)
            if cell is None:
                lines.append(f"| {arm} | 0 | " + " | ".join("--" for _ in cols)
                             + " |")
                continue
            lines.append(f"| {arm} | {cell['n_seeds']} | "
                         + " | ".join(_cell(cell, c) for c in cols) + " |")
        lines.append("")
    lines += ["## The decision text, applied mechanically", ""]
    for arm in ARMS[1:]:
        v = ruling[arm]
        lines.append(f"- **{arm}**: {v['verdict']} -- "
                     + "; ".join(v.get("reasons", [])))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", default="qp")
    parser.add_argument("--out", default=None,
                        help="output stem; default the ovarian experiments dir")
    args = parser.parse_args(argv)

    cells = collect(args.tag)
    ruling = decide(cells)
    text = render(cells, ruling)
    print(text)

    stem = Path(args.out) if args.out else (
        ROOT / OV / "experiments" / "queryprior")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(
        {"tag": args.tag, "dataset": OV, "transport_block": BLOCK,
         "arms": {a: {k: v for k, v in c.items()} for a, c in cells.items()},
         "decision": ruling}, indent=1, default=float))
    stem.with_suffix(".md").write_text(text)
    print(f"wrote {stem.with_suffix('.json')} and {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
