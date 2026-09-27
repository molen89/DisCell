#!/usr/bin/env python3
"""The read-out of the fixed false-positive floor check (todo 8.15b; devlog
2026-09-24 21:20, "Two training-side candidates", part B).

With vs without ``--fp-floor`` at the final configuration, kappa 0.1, 200/20,
seeds 0 1 2, on two sections (plus, on ovarian, the area arm fp_area_s{0,1,2},
``--fp-floor --fp-area``: lambda_i proportional to segmented area with the
section total fixed; coordinator 2026-09-24 21:50, the Spearman rule having
no power on zero-inflated control counts):

    ovarian  without: wfix_warmup30_aw0.1_s{0,1,2} (the 8.9 w-only warm-up
             runs, this exact configuration, reused); with: fp_s{0,1,2}
    GSE core without: aw_ref0.1_s{0,1} (the alpha_w ladder's reference rung,
             reused) + fp_control_s2 (the same configuration, seed 2, fitted
             here); with: fp_s{0,1,2}

Per section and arm, the 3-seed mean [min, max] of recon, NMI, the per-block
probe (fraction of the uncontrolled500 reference: ridge / MLP x composition /
image), mirror, cycle_z, cycle_w, I(niche; w) excess, Read A own-target gap
closed (all genes and HVG), the 6b.1 LR-gene effect of the leak and response
shares (and their mean shares), and KL_w. A read is flagged when the with-arm
mean moves from the without-arm mean by more than one without-arm seed-sd
(ddof 1). The entry's expected outcome, written first: small changes (<= one
seed-sd on every read); a larger change is a finding. No adoption rule is
applied here -- the decision is the author's.

    uv run python scripts/fp_table.py [--at best]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from discell.experiments.at_best import AT_CHOICES, HEADER, battery_at_best

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "datasets"
OV = "xenium_prime_ovarian_cancer_ffpe"
GS = "gse315411_pdltma06_11_prime_solo"
SECTIONS = {
    OV: {"label": "ovarian (full)",
         "without": ["wfix_warmup30_aw0.1_s0", "wfix_warmup30_aw0.1_s1",
                     "wfix_warmup30_aw0.1_s2"],
         "with": ["fp_s0", "fp_s1", "fp_s2"],
         "area": ["fp_area_s0", "fp_area_s1", "fp_area_s2"]},
    GS: {"label": "GSE core (pdl018d)",
         "without": ["aw_ref0.1_s0", "aw_ref0.1_s1", "fp_control_s2"],
         "with": ["fp_s0", "fp_s1", "fp_s2"]},
}
ARMS = ("without", "with", "area")
#: column headers: control | per-section lambda | lambda_i proportional to area
ARM_LABELS = {"without": "control", "with": "per-section λ", "area": "area λ_i"}
PROBE = [(f, b) for f in ("ridge", "mlp") for b in ("comp", "img")]

#: (key, label, format) in table order
READS = [
    ("recon", "recon (held-out, nats/count)", "{:.4f}"),
    ("nmi", "NMI(z; t)", "{:.3f}"),
    *[(f"probe_{f}_{b}", f"probe {f} {b}: fraction of uncontrolled500",
       "{:.3f}") for f, b in PROBE],
    ("mirror_r2", "mirror R²", "{:.3f}"),
    ("cycle_z", "cycle_z R² (pooled)", "{:.3f}"),
    ("cycle_w", "cycle_w R² (pooled)", "{:.4f}"),
    ("w_niche_mi_excess", "I(niche; w) excess", "{:+.4f}"),
    ("read_a_own_gap_closed", "Read A own: median gap closed", "{:.3f}"),
    ("read_a_own_gap_closed_hvg", "Read A own (HVG 1000): median gap closed",
     "{:.3f}"),
    ("leak_lr_effect", "leak share, LR vs matched (rank-biserial)", "{:+.3f}"),
    ("response_lr_effect", "response share, LR vs matched (rank-biserial)",
     "{:+.3f}"),
    ("mean_share_leak", "mean leak share of the shift", "{:.3f}"),
    ("mean_share_response", "mean response share of the shift", "{:.3f}"),
    ("kl_w", "KL_w (summed, nats)", "{:.4f}"),
    ("best_epoch", "accepted epoch", "{:.0f}"),
]
#: reads that describe the floor itself, not a comparison
FLOOR_READS = [("fp_lambda", "λ (expected false positives per cell, panel)",
                "{:.3f}"),
               ("fp_cap_share", "share of cells at the η = 0.2 cap", "{:.4f}")]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _dig(d, *keys, default=float("nan")):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def read_run(dataset: str, run: str, at: str = "best") -> dict | None:
    """Every number one fit contributes, or None when it has not landed."""
    d = ROOT / dataset / "runs" / run
    metrics = _load(d / "metrics.json")
    if metrics is None:
        return None
    config = _load(d / "config.json") or {}
    battery = battery_at_best(d) if at == "best" else metrics["final"]
    out = {"run": run, "fp_floor": bool(config.get("fp_floor", False)),
           "fp_area": bool(config.get("fp_area", False)),
           "fp_lambda": config.get("fp_lambda"),
           "fp_cap_share": config.get("fp_cap_share"),
           "seed": config.get("seed"),
           "best_epoch": _dig(metrics, "best", "epoch"),
           "dead_w_channel": bool(metrics.get("dead_w_channel")),
           "recon": battery["recon_val"], "nmi": battery["nmi"],
           "mirror_r2": _dig(battery, "mirror", "r2"),
           "cycle_z": _dig(battery, "cycle", "z", "r2_pooled"),
           "cycle_w": _dig(battery, "cycle", "w", "r2_pooled"),
           "kl_w": float(np.sum(battery["kl_w_per_dim"]))}
    if at == "best":
        out["at_best"] = battery["at_best"]
    probe = _load(d / "validation" / "probe_blocks.json") or {}
    for f, b in PROBE:
        out[f"probe_{f}_{b}"] = _dig(probe, f, b, "fraction_of_uncontrolled")
    out["w_niche_mi_excess"] = _dig(_load(d / "degeneracy.json") or {},
                                    "w_channel", "w_niche_mi_excess")
    dist = _load(d / "transport" / "transport_distribution.json")
    if dist:
        out["read_a_own_gap_closed"] = _dig(dist, "summary_model_own",
                                            "pairwise", "median_gap_closed")
        out["read_a_own_gap_closed_hvg"] = _dig(
            dist, "summary_model_own_hvg", "pairwise", "median_gap_closed")
    share = _load(ROOT / dataset / "experiments"
                  / f"external_signalling_share_{run}.json")
    if share:
        for channel in ("leak", "response"):
            out[f"{channel}_lr_effect"] = _dig(share, "tests", channel,
                                               "abundance_matched", "effect")
            out[f"mean_share_{channel}"] = _dig(share, "mean_share", channel)
    return out


def summarise(runs: list[dict], keys) -> dict:
    out = {"runs": [r["run"] for r in runs], "n_seeds": len(runs),
           "dead_w_channel": sum(r["dead_w_channel"] for r in runs),
           "per_seed": {r["run"]: r for r in runs}}
    if any("at_best" in r for r in runs):
        out["at_best"] = [r.get("at_best") for r in runs]
    for key in keys:
        vals = [r[key] for r in runs if _finite(r.get(key))]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                        "max": float(max(vals)), "n": len(vals),
                        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1
                        else None}
    return out


def compare(without: dict, with_: dict) -> dict:
    """Per read: the move of the with-arm mean in without-arm seed-sds, and
    the flag (> 1 sd). None where either side or the sd is missing."""
    out = {}
    for key, _, _ in READS:
        a, b = without.get(key), with_.get(key)
        if not a or not b or not a.get("sd"):
            out[key] = {"delta": (b["mean"] - a["mean"]) if a and b else None,
                        "in_sd": None, "flag": None}
            continue
        delta = b["mean"] - a["mean"]
        out[key] = {"delta": delta, "sd_without": a["sd"],
                    "in_sd": delta / a["sd"], "flag": bool(abs(delta) > a["sd"])}
    return out


def collect(at: str) -> dict:
    keys = [k for k, _, _ in READS + FLOOR_READS]
    sections = {}
    for ds, spec in SECTIONS.items():
        arms = {}
        for arm in ARMS:
            runs = [r for r in (read_run(ds, run, at)
                                for run in spec.get(arm, []))
                    if r is not None]
            arms[arm] = summarise(runs, keys) if runs else None
        cmps, moved = {}, {}
        for arm in ARMS[1:]:
            if arm not in spec:
                continue
            c = (compare(arms["without"], arms[arm])
                 if arms["without"] and arms[arm] else None)
            cmps[arm] = c
            moved[arm] = sorted(k for k, v in (c or {}).items() if v["flag"])
        sections[ds] = {"label": spec["label"], "runs": spec, "arms": arms,
                        "comparison": cmps.get("with"),
                        "comparison_area": cmps.get("area"),
                        "moved_beyond_one_sd": moved.get("with", []),
                        "moved_beyond_one_sd_area": moved.get("area")}
    return sections


def _fmt(entry: dict | None, fmt: str, n_want: int = 3) -> str:
    if not entry:
        return "--"
    text = (f"{fmt.format(entry['mean'])} [{fmt.format(entry['min'])}, "
            f"{fmt.format(entry['max'])}]")
    return text + ("" if entry["n"] == n_want else f" (n={entry['n']})")


def render(sections: dict, at: str) -> str:
    lines = ["# The fixed false-positive floor, with vs without (todo 8.15b)",
             ""]
    if at == "best":
        lines[0] += f" -- {HEADER}"
    lines += ["p_i = (1 − κ − η_i) ρ_i + κ ρ̄_i + η_i u, u = 1/G, "
              "η_i = min(λ/ℓ_i, 0.2), one λ per section from the Xenium cell "
              "table (negative-control probe + negative-control codeword + "
              "genomic control, each per feature × G). Final configuration, "
              "κ 0.1, 200/20. Cells: 3-seed mean [min, max]. Δ/sd: the move "
              "of the with-arm mean in without-arm seed-sds (ddof 1); ⚑ = "
              "more than one sd. Expected (written first): small changes, "
              "≤ one seed-sd on every read; a larger change is a finding. No "
              "adoption rule is applied here.", ""]
    for ds, sec in sections.items():
        arms = sec["arms"]
        shown = [a for a in ARMS if a in sec["runs"]]
        lines += [f"## {sec['label']} (`{ds}`)", "",
                  "; ".join(f"{ARM_LABELS[a]}: {', '.join(sec['runs'][a])}"
                            for a in shown), ""]
        for arm in shown[1:]:
            w = arms.get(arm) or {}
            for key, label, fmt in FLOOR_READS:
                if w.get(key):
                    lines.append(f"- {ARM_LABELS[arm]}: {label}: "
                                 f"{fmt.format(w[key]['mean'])}")
        cmps = {"with": sec["comparison"] or {},
                "area": sec.get("comparison_area") or {}}
        head = "| read | " + " | ".join(
            [ARM_LABELS["without"]] + [f"{ARM_LABELS[a]} | Δ/sd"
                                       for a in shown[1:]]) + " |"
        lines += ["", head, "|---" * (1 + 1 + 2 * len(shown[1:])) + "|"]
        for key, label, fmt in READS:
            cells = [_fmt((arms["without"] or {}).get(key), fmt)]
            for arm in shown[1:]:
                c = cmps[arm].get(key) or {}
                cells += [_fmt((arms[arm] or {}).get(key), fmt),
                          ("--" if c.get("in_sd") is None
                           else f"{c['in_sd']:+.2f}")
                          + (" ⚑" if c.get("flag") else "")]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        for arm, key in (("with", "moved_beyond_one_sd"),
                         ("area", "moved_beyond_one_sd_area")):
            if arm in shown:
                moved = sec.get(key) or []
                lines.append(f"Moved by more than one control seed-sd "
                             f"({ARM_LABELS[arm]}): "
                             + (", ".join(moved) if moved else "none") + ".")
        missing = [r for arm in shown for r in sec["runs"][arm]
                   if r not in ((arms[arm] or {}).get("per_seed") or {})]
        dead = [r for arm in shown for r, v in
                ((arms[arm] or {}).get("per_seed") or {}).items()
                if v["dead_w_channel"]]
        lines += ["Missing fits: " + (", ".join(missing) if missing else "none")
                  + ". Dead KL_w channel (trainer): "
                  + (", ".join(dead) if dead else "none") + ".", ""]
        per = ["recon", "nmi", "probe_mlp_comp", "cycle_z", "w_niche_mi_excess",
               "read_a_own_gap_closed", "leak_lr_effect", "kl_w"]
        lines += ["| run | floor | best epoch | " + " | ".join(per) + " |",
                  "|---" * (len(per) + 3) + "|"]
        for arm in shown:
            for run, r in ((arms[arm] or {}).get("per_seed") or {}).items():
                floor = ("area" if r["fp_area"] else "section"
                         if r["fp_floor"] else "off")
                lines.append(f"| {run} | {floor} | "
                             f"{r['best_epoch']} | " + " | ".join(
                                 "--" if not _finite(r.get(k)) else
                                 f"{r[k]:.4g}" for k in per) + " |")
        lines.append("")
    if at == "best":
        fell_back = [run for sec in sections.values()
                     for arm in ARMS if sec["arms"][arm]
                     for run, ok in zip(sec["arms"][arm]["runs"],
                                        sec["arms"][arm].get("at_best", []))
                     if not ok]
        lines += ["In-trainer reads are the history row at the accepted epoch; "
                  "probe, degeneracy, transport and 6b.1 are post-hoc on "
                  "best.pt. Fallbacks to the last epoch: "
                  + (", ".join(fell_back) if fell_back else "none") + "."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--at", choices=AT_CHOICES, default="best")
    parser.add_argument("--out", default=str(ROOT / OV / "experiments"
                                             / "fp_floor"))
    args = parser.parse_args(argv)
    sections = collect(args.at)
    text = render(sections, args.at)
    print(text)
    stem = Path(args.out)
    if args.at != "best":
        stem = stem.with_name(f"{stem.name}_at_{args.at}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(
        {"test": "todo 8.15b, devlog 2026-09-24 21:20 part B",
         "read_at": HEADER if args.at == "best" else "final",
         "flag_rule": "|mean_with - mean_without| > sd_without (ddof 1)",
         "sections": sections}, indent=1, default=float))
    stem.with_suffix(".md").write_text(text)
    print(f"wrote {stem.with_suffix('.json')}, {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
