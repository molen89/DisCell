#!/usr/bin/env python3
"""The read-out for review R12 Test 2, the leak-form sensitivity arms (devlog 2026-09-24).

Three forms of the leak coefficient on the ovarian core at the final
configuration (warm-up 30, alpha_z 1/2, kappa 0.1, 200/20), seeds 0 1 2:

    global  one kappa (the pinned form). The w-only warm-up runs of 8.9,
            ``wfix_warmup30_aw0.1_s*``, ARE this configuration (config.json
            identical but run name and seed) and are reused, not refitted.
    depth   ``--kappa-mode depth``: kappa_i = kappa * clip(r_i / m, 0, 5),
            r_i = sum_j beta_ij l_j / l_i, m solved by bisection so the
            post-clip mean leak fraction over the connected training cells
            is kappa (amendments 1-2)
    gene    ``--kappa-mode gene``:  kappa_g = min(kappa * s_g / mean(s_g), 0.9)
    density ``--kappa-mode density``: the same with r_i = sum_j beta_ij
            (l_j/A_j) / (l_i/A_i), A the cell area (nucleus-expansion cells:
            nuclear area x the type-median cell/nucleus ratio); third arm,
            added by the author the same day

Per arm, the three-seed mean [min, max] of every read the entry names: the
battery guards (recon, NMI, probe vs its floor, mirror, cycle_z, cycle_w,
I(niche;w) excess), transport (mean read: fraction of ceiling and beats-both;
Read A own-target gap closed; the tumour-band Read B beside), the LR-gene
effect of the leak and response channels (6b.1), and the GO extracellular /
plasma-membrane / nucleus effect of atlas programme 0 (6b.10). The entry
states what is wished for; no rule is applied here, the table is the finding.

    uv run python scripts/r12_table.py [--at best]

``--at best`` (the default) reads the in-trainer battery at the accepted
checkpoint (review R26: the reused control runs predate the closing-evaluation
fix, the new arms do not, and only the checkpoint read compares them);
``--at final`` writes ``*_at_final`` siblings.
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
ARMS = {"global": "wfix_warmup30_aw0.1_s{s}",
        "depth": "r12_depth_s{s}",
        "gene": "r12_gene_s{s}",
        "density": "r12_density_s{s}"}
ARM_FLAGS = {"global": "(reused 8.9 w-only warm-up runs)",
             "depth": "--kappa-mode depth", "gene": "--kappa-mode gene",
             "density": "--kappa-mode density"}
SEEDS = [0, 1, 2]
BLOCK = "extrapolation"       #: the transport tier read (all panels), as in 8.x
GO_SETS = ("extracellular", "plasma_membrane", "nucleus")
CYCLE_W_GUARD = 0.02

#: (key, label, format) in table order
READS = [
    ("recon", "recon (held-out, nats/count)", "{:.4f}"),
    ("nmi", "NMI(z; t)", "{:.3f}"),
    ("probe_delta_ce", "probe ΔCE", "{:.4f}"),
    ("probe_noise_floor", "probe floor", "{:.4f}"),
    ("probe_excess", "probe ΔCE − floor", "{:+.4f}"),
    ("mirror_r2", "mirror R²", "{:.3f}"),
    ("cycle_z", "cycle_z R² (pooled)", "{:.3f}"),
    ("cycle_w", "cycle_w R² (pooled)", "{:.4f}"),
    ("mi_ratio", "I(z;t)/H(t)", "{:.3f}"),
    ("w_niche_mi_excess", "I(niche; w) excess", "{:+.4f}"),
    ("read_a_of_ceiling", "Read A mean: fraction of ceiling", "{:.3f}"),
    ("read_a_of_ceiling_trusted", "Read A mean: fraction of ceiling (trusted)", "{:.3f}"),
    ("read_a_beats_both", "Read A mean: beats-both (share of panels)", "{:.3f}"),
    ("read_a_own_gap_closed", "Read A own: median gap closed", "{:.3f}"),
    ("read_b_of_ceiling", "Read B (tumour band) mean: fraction of ceiling", "{:.3f}"),
    ("read_b_beats_both", "Read B mean: beats-both (share of panels)", "{:.3f}"),
    ("read_b_own_gap_closed", "Read B own: median gap closed", "{:.3f}"),
    ("leak_lr_effect", "leak share, LR vs matched (rank-biserial)", "{:+.3f}"),
    ("response_lr_effect", "response share, LR vs matched (rank-biserial)", "{:+.3f}"),
    ("leak_lr_effect_all", "leak share, LR vs all other", "{:+.3f}"),
    ("response_lr_effect_all", "response share, LR vs all other", "{:+.3f}"),
    ("mean_share_leak", "mean leak share of the shift", "{:.3f}"),
    ("mean_share_response", "mean response share of the shift", "{:.3f}"),
    ("go_p0_extracellular", "GO programme 0: extracellular", "{:+.3f}"),
    ("go_p0_plasma_membrane", "GO programme 0: plasma membrane", "{:+.3f}"),
    ("go_p0_nucleus", "GO programme 0: nucleus", "{:+.3f}"),
    ("kappa_i_mean_train", "mean κ_i over connected training cells", "{:.4f}"),
    ("kappa_i_cap_share", "share of training cells at the κ_i = 0.5 cap", "{:.4f}"),
    ("kappa_ratio_mean", "normaliser m (clip-aware, amendment 2)", "{:.4f}"),
    ("best_epoch", "accepted epoch", "{:.0f}"),
    ("kl_w", "KL_w (summed, nats)", "{:.4f}"),
]


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


def read_run(run: str, at: str = "best") -> dict | None:
    """Every number one fit contributes, or None when it has not landed."""
    d = ROOT / OV / "runs" / run
    metrics = _load(d / "metrics.json")
    if metrics is None:
        return None
    config = _load(d / "config.json") or {}
    battery = battery_at_best(d) if at == "best" else metrics["final"]
    out = {
        "run": run,
        "kappa_mode": config.get("kappa_mode", "global"),
        "kappa_gene_source": config.get("kappa_gene_source"),
        "best_epoch": _dig(metrics, "best", "epoch"),
        "final_epoch_key": "final_epoch" in metrics,
        "dead_w_channel": bool(metrics.get("dead_w_channel")),
        "recon": battery["recon_val"],
        "nmi": battery["nmi"],
        "mirror_r2": _dig(battery, "mirror", "r2"),
        "probe_delta_ce": _dig(battery, "probe", "delta_ce"),
        "probe_noise_floor": _dig(battery, "probe", "noise_floor"),
        "cycle_z": _dig(battery, "cycle", "z", "r2_pooled"),
        "cycle_w": _dig(battery, "cycle", "w", "r2_pooled"),
        "mi_ratio": _dig(battery, "degeneracy", "mi_ratio"),
        "kl_w": float(np.sum(battery["kl_w_per_dim"])),
    }
    if at == "best":
        out["at_best"] = battery["at_best"]
    form = _load(d / f"kappa_{out['kappa_mode']}.json")
    if form:
        out["kappa_ratio_mean"] = form.get("ratio_mean_used")
        out["kappa_i_mean_train"] = _dig(form, "ratio", "kappa_i_mean_train")
        out["kappa_i_cap_share"] = _dig(form, "ratio", "kappa_i_cap_share_train")
        area = form.get("area") or {}
        if area:
            out["expansion_fraction"] = area.get("expansion_fraction")
            out["expansion_fraction_by_type"] = {
                v.get("type", g): v["expansion_fraction"]
                for g, v in area.get("by_type", {}).items()}
    out["probe_excess"] = out["probe_delta_ce"] - out["probe_noise_floor"]
    out["probe_at_floor"] = bool(abs(out["probe_delta_ce"])
                                 <= abs(out["probe_noise_floor"]))
    out["cycle_w_ok"] = bool(out["cycle_w"] <= CYCLE_W_GUARD)

    guard = _load(d / "degeneracy.json") or {}
    out["w_niche_mi_excess"] = _dig(guard, "w_channel", "w_niche_mi_excess")
    out["w_guard_dead"] = _dig(guard, "w_channel", "dead_context_channel",
                               default=None)

    for tag, stem in (("a", "transport"), ("b", "transport_tumour-band")):
        mean = _load(d / "transport" / f"{stem}.json")
        if mean:
            block = _dig(mean, "summary", BLOCK, default={})
            n = block.get("n_panels", 0)
            out[f"read_{tag}_of_ceiling"] = block.get("counterfactual_of_ceiling",
                                                      float("nan"))
            out[f"read_{tag}_beats_both"] = (block["full_beats_both"] / n
                                             if n else float("nan"))
            out[f"read_{tag}_n_panels"] = n
            out[f"read_{tag}_of_ceiling_trusted"] = _dig(
                mean, "summary", f"{BLOCK}_trusted", "counterfactual_of_ceiling")
        dist = _load(d / "transport" / f"{stem}_distribution.json")
        if dist:
            out[f"read_{tag}_own_gap_closed"] = _dig(
                dist, "summary_model_own", "pairwise", "median_gap_closed")

    share = _load(ROOT / OV / "experiments"
                  / f"external_signalling_share_{run}.json")
    if share:
        for channel in ("leak", "response"):
            out[f"{channel}_lr_effect"] = _dig(share, "tests", channel,
                                               "abundance_matched", "effect")
            out[f"{channel}_lr_effect_all"] = _dig(share, "tests", channel,
                                                   "all_other", "effect")
            out[f"mean_share_{channel}"] = _dig(share, "mean_share", channel)

    go = _load(ROOT / OV / "experiments" / f"go_localisation_{run}.json")
    table = _dig(go, "runs", run, "programmes", default=None) if go else None
    if table:
        rows = {r["set"]: r for r in table[0]}
        for name in GO_SETS:
            out[f"go_p0_{name}"] = _dig(rows, name, "effect")
            out[f"go_p0_{name}_q"] = _dig(rows, name, "q")
    return out


def cell(runs: list[dict]) -> dict:
    out = {"runs": [r["run"] for r in runs], "n_seeds": len(runs),
           "dead_w_channel": sum(r["dead_w_channel"] for r in runs),
           "w_guard_dead": sum(bool(r.get("w_guard_dead")) for r in runs),
           "niche_guard_alive": sum(_finite(r.get("w_niche_mi_excess"))
                                    and r["w_niche_mi_excess"] > 0 for r in runs),
           "cycle_w_ok": sum(r["cycle_w_ok"] for r in runs),
           "probe_at_floor": sum(r["probe_at_floor"] for r in runs),
           "per_seed": {r["run"]: r for r in runs}}
    if any("at_best" in r for r in runs):
        out["at_best"] = [r.get("at_best") for r in runs]
    for key, _, _ in READS:
        vals = [r[key] for r in runs if _finite(r.get(key))]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                        "max": float(max(vals)), "n": len(vals)}
    return out


def collect(at: str) -> dict:
    cells = {}
    for arm, template in ARMS.items():
        runs = [r for r in (read_run(template.format(s=s), at) for s in SEEDS)
                if r is not None]
        cells[arm] = cell(runs) if runs else None
    return cells


def _fmt(entry: dict | None, fmt: str) -> str:
    if not entry:
        return "--"
    text = (f"{fmt.format(entry['mean'])} [{fmt.format(entry['min'])}, "
            f"{fmt.format(entry['max'])}]")
    return text + ("" if entry["n"] == len(SEEDS) else f" (n={entry['n']})")


def render(cells: dict, at: str) -> str:
    lines = ["# Review R12 Test 2: the leak coefficient's form (ovarian, "
             "final configuration, κ 0.1)", ""]
    if at == "best":
        lines[0] += f" -- {HEADER}"
    lines += ["Cells: 3-seed mean [min, max]. `global` = the 8.9 w-only "
              "warm-up runs (`wfix_warmup30_aw0.1_s*`), reused as the control "
              "(identical configuration); `depth` = `--kappa-mode depth`, "
              "κ_i = κ·clip(r_i / m, 0, 5) ≤ 0.5 with r_i = "
              "Σ_j β_ij ℓ_j / ℓ_i and m solved so the post-clip mean leak "
              "fraction over the connected training cells is κ; `gene` = "
              "`--kappa-mode gene`, κ_g = min(κ·s_g/mean s_g, 0.9), s_g the "
              "measured per-gene extranuclear share; `density` = "
              "`--kappa-mode density`, the same normalised clip on r_i = "
              "Σ_j β_ij (ℓ_j/A_j) / "
              "(ℓ_i/A_i), 0, 5), A the segmented cell area (nucleus-expansion "
              "cells: nuclear area × the type-median cell/nucleus ratio of "
              "non-expansion cells). Transport tier: "
              f"`{BLOCK}` (all panels). LR effects: 6b.1, abundance-matched "
              "unless marked. GO: rank-biserial effect of |loading| of atlas "
              "programme 0.", ""]
    if at == "best":
        fell_back = [run for c in cells.values() if c
                     for run, ok in zip(c["runs"], c.get("at_best", [])) if not ok]
        lines += ["In-trainer reads are the history row at the accepted epoch; "
                  "the guard, transport, 6b.1 and GO reads are post-hoc on "
                  "best.pt. Fallbacks to the last epoch: "
                  + (", ".join(fell_back) if fell_back else "none") + ".", ""]
    arms = list(ARMS)
    lines += ["| guard counts | " + " | ".join(arms) + " |",
              "|---" * (len(arms) + 1) + "|"]
    for key, label in (("n_seeds", "seeds landed"),
                       ("dead_w_channel", "dead KL_w channel (trainer)"),
                       ("niche_guard_alive", "I(niche;w) excess > 0"),
                       ("cycle_w_ok", f"cycle_w ≤ {CYCLE_W_GUARD}"),
                       ("probe_at_floor", "probe at its floor")):
        lines.append(f"| {label} | " + " | ".join(
            "--" if cells[a] is None else str(cells[a][key]) for a in arms) + " |")
    lines += ["", "| read | " + " | ".join(arms) + " |",
              "|---" * (len(arms) + 1) + "|"]
    for key, label, fmt in READS:
        lines.append(f"| {label} | " + " | ".join(
            _fmt((cells[a] or {}).get(key), fmt) for a in arms) + " |")
    lines += ["", "## Per seed", ""]
    per = ["recon", "nmi", "cycle_z", "w_niche_mi_excess", "read_a_of_ceiling",
           "read_a_own_gap_closed", "leak_lr_effect", "response_lr_effect",
           "go_p0_plasma_membrane", "go_p0_extracellular"]
    lines += ["| run | κ form | best epoch | " + " | ".join(per) + " |",
              "|---" * (len(per) + 3) + "|"]
    for arm in arms:
        for run, r in ((cells[arm] or {}).get("per_seed") or {}).items():
            lines.append(f"| {run} | {r['kappa_mode']} | {r['best_epoch']} | "
                         + " | ".join("--" if not _finite(r.get(k)) else
                                      f"{r[k]:.4g}" for k in per) + " |")
    ref = next((r for r in ((cells.get("density") or {}).get("per_seed")
                            or {}).values() if r.get("expansion_fraction_by_type")),
               None)
    if ref:
        lines += ["", "Density arm, nucleus-expansion cells (area from the "
                  f"nucleus): {ref['expansion_fraction']:.2%} of cells; by type: "
                  + ", ".join(f"{k} {v:.1%}" for k, v in sorted(
                      ref["expansion_fraction_by_type"].items(),
                      key=lambda kv: -kv[1]))]
    missing = [ARMS[a].format(s=s) for a in arms for s in SEEDS
               if ARMS[a].format(s=s) not in ((cells[a] or {}).get("per_seed") or {})]
    lines += ["", "Missing fits: " + (", ".join(missing) if missing else "none")]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--at", choices=AT_CHOICES, default="best",
                        help="in-trainer reads at the accepted checkpoint "
                             "(default, R26) or the last-epoch `final`")
    parser.add_argument("--out", default=str(ROOT / OV / "experiments"
                                             / "r12_arms"))
    args = parser.parse_args(argv)
    cells = collect(args.at)
    text = render(cells, args.at)
    print(text)
    stem = Path(args.out)
    if args.at != "best":
        stem = stem.with_name(f"{stem.name}_at_{args.at}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(
        {"test": "review R12 Test 2 (devlog 2026-09-24)", "arms": ARM_FLAGS,
         "dataset": OV, "seeds": SEEDS, "transport_tier": BLOCK,
         "read_at": HEADER if args.at == "best" else "final",
         "cells": cells}, indent=1, default=float))
    stem.with_suffix(".md").write_text(text)
    print(f"wrote {stem.with_suffix('.json')}, {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
