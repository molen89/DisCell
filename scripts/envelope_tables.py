#!/usr/bin/env python3
"""Per-dataset envelope tables over the three `best` seeds (todo 8.6).

One markdown table per dataset: every instrument of the battery as
min / mean / max over the seed triple, so the paper quotes an envelope and
never a single fit. GSE315411 also gets a held-out column -- the dual
section, the only genuinely out-of-sample read.

Everything is read off disk; nothing is fitted or re-evaluated here. A run
that never produced an artefact leaves its cell empty rather than the table
missing.

Usage::

    python scripts/envelope_tables.py
    python scripts/envelope_tables.py --dataset <id> --runs best_s0 best_s1
    python scripts/envelope_tables.py --at best

``--at best`` reads the in-trainer battery (probe, mirror, cycle, KL_w,
I(z;t)/H(t), within-type variance, recon gap) at the accepted checkpoint
instead of the last-epoch ``final`` (review R26). Recon and NMI were always
read there, and the held-out, transport and atlas reads come from best.pt.
Outputs go to ``*_at_best`` siblings.

Where a run has been re-graded by ``discell/experiments/probe_regrade.py``
(``validation/probe_blocks.json``; on the held-out section
``crossslide/probe_blocks_<section>.json``), the per-block probe rows are
filled: each block's excess over its within-type permutation floor, the
same as a fraction of the uncontrolled (alpha_a = 0) fit's, and the fraction
of seeds passing the guard (all four blocks <= 25 % of uncontrolled). The old
ΔCE rows stay, labelled "pooled, legacy".

Where ``discell/experiments/recon_modes.py`` has run (``recon_modes.json``;
on the held-out section ``recon_modes_<section>.json``), the intrinsic-only
and context-only decodes and the type-profile reference are filled beside
the full-posterior reconstruction (metrics package item 2).

``--ci`` (metrics package item 1) adds a column with the 200 um
tile-bootstrap 95 % interval of each read, read from
``bootstrap_ci.json`` (``discell/experiments/bootstrap.py``) and, for the
reconstruction rows, from ``recon_modes.json``. The column is the envelope of
the per-seed intervals -- lowest lower bound, highest upper bound over the
seeds that carry one -- so it spans seed and tile variation at once. A ``†``
marks a read whose recomputed point estimate did not reproduce the stored
one within the bootstrap's tolerance on some seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from discell import paths
from discell.experiments.at_best import (AT_CHOICES, HEADER, battery_at_best,
                                        suffixed)

DATASETS = ("xenium_prime_ovarian_cancer_ffpe",
            "xenium_prime_human_ovary_ff",
            "xenium_prime_human_lung_cancer_ffpe",
            "gse315411_pdltma06_11_prime_solo")
DEFAULT_RUNS = ("best_s0", "best_s1", "best_s2")
CROSS_SLIDE = {"gse315411_pdltma06_11_prime_solo":
               "gse315411_pdltma06_10_prime_dual"}

#: (row label, key in the per-run record, format)
ROWS = (
    ("held-out reconstruction (nats/count)", "recon", "{:.4f}"),
    ("held-out recon, intrinsic-only decode (posterior z, w = m_ψ)",
     "recon_intrinsic", "{:.4f}"),
    ("held-out recon, context-only decode (type-mean z, w = m_ψ)",
     "recon_context", "{:.4f}"),
    ("held-out recon, type-profile reference", "recon_type_profile",
     "{:.4f}"),
    ("NMI vs the type labels", "nmi", "{:.4f}"),
    ("probe ΔCE (pooled, legacy)", "probe_delta_ce", "{:.4f}"),
    ("probe floor (pooled, legacy)", "probe_noise_floor", "{:.4f}"),
    ("probe excess, composition / ridge (nats, R20)", "probe_ridge_comp_excess",
     "{:+.4f}"),
    ("probe excess, image / ridge (nats, R20)", "probe_ridge_img_excess",
     "{:+.4f}"),
    ("probe excess, composition / MLP (nats, R22)", "probe_mlp_comp_excess",
     "{:+.4f}"),
    ("probe excess, image / MLP (nats, R22)", "probe_mlp_img_excess",
     "{:+.4f}"),
    ("probe excess ÷ uncontrolled, composition / ridge", "probe_ridge_comp_frac",
     "{:.2f}"),
    ("probe excess ÷ uncontrolled, image / ridge", "probe_ridge_img_frac",
     "{:.2f}"),
    ("probe excess ÷ uncontrolled, composition / MLP", "probe_mlp_comp_frac",
     "{:.2f}"),
    ("probe excess ÷ uncontrolled, image / MLP", "probe_mlp_img_frac",
     "{:.2f}"),
    ("invariance guard, all four blocks ≤ 25 % of uncontrolled (fraction "
     "of seeds)", "invariance_pass", "{:.2f}"),
    ("mirror R²", "mirror_r2", "{:.4f}"),
    ("cycle R² (z, pooled)", "cycle_z", "{:.4f}"),
    ("cycle R² (w, pooled)", "cycle_w", "{:.4f}"),
    ("cycle R² (linear reference)", "cycle_ref", "{:.4f}"),
    ("KL_w (sum over dims)", "kl_w", "{:.4f}"),
    ("I(z;t)/H(t)", "mi_ratio", "{:.4f}"),
    ("within-type variance fraction of z", "within_var_fraction", "{:.4f}"),
    ("type-mean-z reconstruction gap", "recon_gap", "{:.4f}"),
    ("I(niche; w) excess over within-type floor (nats)", "w_niche_mi_excess",
     "{:+.4f}"),
    ("transport, mean read: fraction of ceiling (trusted)",
     "transport_of_ceiling_trusted", "{:.3f}"),
    ("transport, mean read: fraction of ceiling (all panels)",
     "transport_of_ceiling", "{:.3f}"),
    ("Read A gap closed, group-w target (pairwise median)",
     "readA_gap_group", "{:.3f}"),
    ("Read A gap closed, own-w target (pairwise median)",
     "readA_gap_own", "{:.3f}"),
    ("Read A type-mean predictor, own-w target",
     "readA_type_mean_own", "{:.3f}"),
    ("Read B twin margin, group-w target (pairwise median)",
     "twin_margin_group", "{:.3f}"),
    ("Read B twin margin, own-w target (pairwise median)",
     "twin_margin_own", "{:.3f}"),
    ("atlas effective rank", "atlas_rank", "{:.1f}"),
    ("atlas cross-seed axis-1 cosine", "atlas_axis1_cosine", "{:.3f}"),
)

HELD_OUT_ROWS = {
    "recon": "recon", "nmi": "nmi", "probe_delta_ce": "probe_delta_ce",
    "probe_noise_floor": "probe_noise_floor", "mirror_r2": "mirror_r2",
    "cycle_z": "cycle_z", "cycle_w": "cycle_w", "cycle_ref": "cycle_ref",
    "kl_w": "kl_w",
    **{k: k for k in ("recon_intrinsic", "recon_context",
                      "recon_type_profile")},
    **{k: k for k in ("probe_ridge_comp_excess", "probe_ridge_img_excess",
                      "probe_mlp_comp_excess", "probe_mlp_img_excess",
                      "probe_ridge_comp_frac", "probe_ridge_img_frac",
                      "probe_mlp_comp_frac", "probe_mlp_img_frac",
                      "invariance_pass")}}


def probe_blocks_row(path: Path) -> dict:
    """The per-block probe (``probe_blocks.json``, R20 + R22) when the run has
    been re-graded: each block's excess over its floor, the same as a
    fraction of the uncontrolled fit's, and the guard as 1/0 so that the
    envelope's mean is the fraction of seeds passing (absent until the
    uncontrolled reference exists)."""
    blocks = _load(path)
    if not blocks:
        return {}
    out = {}
    if blocks.get("invariance_pass") is not None:
        out["invariance_pass"] = float(blocks["invariance_pass"])
    for family in ("ridge", "mlp"):
        for block in ("comp", "img"):
            entry = blocks[family][block]
            out[f"probe_{family}_{block}_excess"] = entry["excess"]
            if entry.get("fraction_of_uncontrolled") is not None:
                out[f"probe_{family}_{block}_frac"] = entry[
                    "fraction_of_uncontrolled"]
    return out


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


#: recon_modes.json mode -> envelope key
RECON_MODES = {"full": "recon", "intrinsic": "recon_intrinsic",
               "context": "recon_context", "type_profile": "recon_type_profile"}


def recon_modes_row(path: Path) -> tuple[dict, dict]:
    """The three decodes and the reference (values), and their tile CIs."""
    modes = (_load(path) or {}).get("modes") or {}
    values, cis = {}, {}
    for mode, key in RECON_MODES.items():
        if mode in modes:
            if key != "recon":           # the full mode IS best.recon_val
                values[key] = modes[mode]["recon"]
            cis[key] = {"ci95": modes[mode]["ci95"], "reproduces": True}
    return values, cis


def bootstrap_row(path: Path) -> dict:
    """``{key: {ci95, reproduces}}`` from ``bootstrap_ci.json``."""
    reads = (_load(path) or {}).get("reads") or {}
    return {key: {"ci95": entry["ci95"],
                  "reproduces": entry.get("reproduces") is not False}
            for key, entry in reads.items()}


def _cycle(block: dict, latent: str):
    entry = (block or {}).get(latent) or {}
    return entry.get("r2_pooled")


def evaluation_row(final: dict) -> dict:
    """The battery of one evaluation block (same shape on both sections)."""
    cycle = final.get("cycle") or {}
    return {
        "recon": final.get("recon_val"),
        "nmi": final.get("nmi"),
        "probe_delta_ce": (final.get("probe") or {}).get("delta_ce"),
        "probe_noise_floor": (final.get("probe") or {}).get("noise_floor"),
        "mirror_r2": (final.get("mirror") or {}).get("r2"),
        "cycle_z": _cycle(cycle, "z"), "cycle_w": _cycle(cycle, "w"),
        "cycle_ref": (_cycle(cycle, "linear_ref")
                      if _cycle(cycle, "linear_ref") is not None
                      else _cycle(cycle, "ceiling")),
        "kl_w": (float(np.sum(final["kl_w_per_dim"]))
                 if final.get("kl_w_per_dim") else None),
    }


def transport_row(run_dir: Path) -> dict:
    """The three transport reads, from whichever files the run has."""
    out: dict = {}
    mean = _load(run_dir / "transport" / "transport.json")
    if mean:
        summary = mean.get("summary") or {}
        for key, tier in (("transport_of_ceiling_trusted",
                           "extrapolation_trusted"),
                          ("transport_of_ceiling", "extrapolation")):
            value = (summary.get(tier) or {}).get("counterfactual_of_ceiling")
            out[key] = (None if value is None or not np.isfinite(value)
                        else float(value))
    dist = _load(run_dir / "transport" / "transport_distribution.json")
    if dist:
        for key, block in (("readA_gap_group", "summary_model"),
                           ("readA_gap_own", "summary_model_own")):
            out[key] = ((dist.get(block) or {}).get("pairwise") or {}).get(
                "median_gap_closed")
        out["readA_type_mean_own"] = (
            (dist.get("summary_model_own") or {}).get("pairwise") or {}).get(
                "median_gap_closed_type_mean")
    twins = _load(run_dir / "transport" / "transport_twins.json")
    if twins:
        for key, block in (("twin_margin_group", "summary"),
                           ("twin_margin_own", "summary_own")):
            out[key] = ((twins.get(block) or {}).get("pairwise") or {}).get(
                "median_twin_margin")
    return out


def atlas_row(run_dir: Path) -> dict:
    atlas = _load(run_dir / "atlas" / "atlas.json")
    if not atlas:
        return {}
    had_none = False
    cosines = []
    for other in (atlas.get("cross_seed") or {}).values():
        for entry in other.get("axis_cosine", []):
            if entry is None:
                had_none = True
                continue
            if entry.get("program") == 0:
                cosines.append(entry["cosine"])
    return {"atlas_rank": atlas.get("rank"),
            "atlas_axis1_cosine": (float(np.mean(cosines)) if cosines
                                   else None),
            "atlas_none_entries": had_none}


def run_record(dataset: str, run: str, at: str = "final") -> dict | None:
    """Everything the tables read for one fit; *at* = "best" takes the
    in-trainer block from the accepted checkpoint's history row."""
    run_dir = paths.dataset(dataset).root / "runs" / run
    metrics = _load(run_dir / "metrics.json")
    if metrics is None:
        return None
    final = (battery_at_best(run_dir) if at == "best"
             else metrics.get("final")) or {}
    record = evaluation_row(final)
    record["recon"] = metrics["best"]["recon_val"]     # the selected epoch
    record["nmi"] = metrics["best"]["nmi"]
    degeneracy = final.get("degeneracy")
    recon_gap = final.get("recon_gap")
    post_hoc = _load(run_dir / "degeneracy.json") or {}
    degeneracy = degeneracy or post_hoc.get("degeneracy") or {}
    recon_gap = recon_gap or post_hoc.get("recon_gap") or {}
    record["mi_ratio"] = degeneracy.get("mi_ratio")
    record["within_var_fraction"] = degeneracy.get("within_var_fraction")
    record["recon_gap"] = recon_gap.get("gap")
    record["w_niche_mi_excess"] = (post_hoc.get("w_channel") or {}).get(
        "w_niche_mi_excess")
    modes, record["ci"] = recon_modes_row(run_dir / "recon_modes.json")
    record.update(modes)
    record["ci"].update(bootstrap_row(run_dir / "bootstrap_ci.json"))
    record.update(probe_blocks_row(run_dir / "validation"
                                   / "probe_blocks.json"))
    record.update(transport_row(run_dir))
    record.update(atlas_row(run_dir))
    record["run"] = run
    if at == "best":
        record["at_best"] = final["at_best"]
    eval_ds = CROSS_SLIDE.get(dataset)
    if eval_ds:
        cross = _load(run_dir / "crossslide" / f"{eval_ds}.json")
        if cross:
            record["held_out"] = evaluation_row(
                cross.get("held_out_section") or {})
            record["held_out"].update(probe_blocks_row(
                run_dir / "crossslide" / f"probe_blocks_{eval_ds}.json"))
        held_modes, _ = recon_modes_row(run_dir / f"recon_modes_{eval_ds}.json")
        if held_modes:
            record.setdefault("held_out", {}).update(held_modes)
    return record


def envelope(records: list[dict], key: str) -> tuple:
    values = [r[key] for r in records
              if r.get(key) is not None and np.isfinite(r[key])]
    if not values:
        return (None, None, None, 0)
    return (float(np.min(values)), float(np.mean(values)),
            float(np.max(values)), len(values))


def ci_cell(records: list[dict], key: str, fmt: str) -> str:
    """Envelope of the per-seed tile CIs: ``[min lo, max hi] (n)``."""
    entries = [r["ci"][key] for r in records if key in (r.get("ci") or {})]
    bounds = [e["ci95"] for e in entries
              if e["ci95"] and all(b is not None and np.isfinite(b)
                                   for b in e["ci95"])]
    if not bounds:
        return ""
    lo = min(b[0] for b in bounds)
    hi = max(b[1] for b in bounds)
    mark = "" if all(e["reproduces"] for e in entries) else " †"
    return f"[{fmt.format(lo)}, {fmt.format(hi)}] ({len(bounds)}){mark}"


def render(dataset: str, records: list[dict], ci: bool = False) -> str:
    held = [r["held_out"] for r in records if r.get("held_out")]
    head = ["metric", "min", "mean", "max", "n"]
    if ci:
        head += ["tile 95 % CI, seed envelope (n)"]
    if held:
        head += ["held-out mean", "held-out n"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    for label, key, fmt in ROWS:
        lo, mean, hi, n = envelope(records, key)
        cells = [label] + (["", "", "", "0"] if n == 0 else
                           [fmt.format(lo), fmt.format(mean),
                            fmt.format(hi), str(n)])
        if ci:
            cells.append(ci_cell(records, key, fmt))
        if held:
            hkey = HELD_OUT_ROWS.get(key)
            hlo, hmean, hhi, hn = (envelope(held, hkey) if hkey
                                   else (None, None, None, 0))
            cells += ["" if hn == 0 else fmt.format(hmean), str(hn)]
        lines.append("| " + " | ".join(cells) + " |")
    runs = ", ".join(f"`{r['run']}`" for r in records)
    note = (f"\nRuns: {runs}. Columns are the min / mean / max over the seed "
            f"triple; `n` is how many of the three runs carry that "
            f"instrument.")
    if held:
        note += (f" The held-out column is the {CROSS_SLIDE[dataset]} "
                 f"section, evaluated with the weights fitted on this one.")
    if ci:
        note += (" The CI column is the envelope of the per-seed 200 um "
                 "tile-bootstrap 95 % intervals (1000 draws; conditional on "
                 "the fitted probes, clustering and model; "
                 "`discell/experiments/bootstrap.py`, reconstruction rows from "
                 "`recon_modes.json`); † = the recomputed point estimate did "
                 "not reproduce the stored one on some seed.")
    if any("at_best" in r for r in records):
        fell_back = [r["run"] for r in records if not r["at_best"]]
        note += (f" In-trainer rows: {HEADER}; last-epoch fallbacks: "
                 + (", ".join(f"`{r}`" for r in fell_back) if fell_back
                    else "none") + ".")
    none_runs = [r["run"] for r in records if r.get("atlas_none_entries")]
    if none_runs:
        note += (f" Some atlas cross-seed axis_cosine entries were null and "
                 f"were skipped for: {', '.join(f'`{r}`' for r in none_runs)}.")
    return f"### {dataset}\n\n" + "\n".join(lines) + "\n" + note + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    parser.add_argument("--combined", default="scratchpad/envelope_tables_all.md")
    parser.add_argument("--at", choices=AT_CHOICES, default="final",
                        help="where the in-trainer reads come from: the "
                             "last-epoch `final` (default) or the accepted "
                             "checkpoint (R26); `best` writes *_at_best files")
    parser.add_argument("--ci", action="store_true",
                        help="add the tile-bootstrap CI column (from "
                             "bootstrap_ci.json / recon_modes.json); writes "
                             "*_ci files")
    args = parser.parse_args(argv)

    combined = ["# Envelope tables over the `best` seed triples"
                + (f" -- {HEADER}" if args.at == "best" else ""),
                "", "Generated by `scripts/envelope_tables.py`; every number "
                "is read off disk.", ""]
    for dataset in args.datasets:
        records = []
        for run in args.runs:
            record = run_record(dataset, run, args.at)
            if record is None:
                print(f"{dataset}: no metrics.json for {run} -- skipped")
                continue
            records.append(record)
        if not records:
            print(f"{dataset}: no runs found, no table written")
            continue
        table = render(dataset, records, args.ci)
        out = suffixed(paths.dataset(dataset).root / "experiments"
                       / ("envelope_table_ci.md" if args.ci
                          else "envelope_table.md"), args.at)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table)
        print(f"wrote {out}")
        combined.append(table)
    path = suffixed(args.combined, args.at)
    if args.ci:
        path = path.with_name(path.stem + "_ci" + path.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(combined))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
