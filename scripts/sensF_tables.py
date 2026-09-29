#!/usr/bin/env python3
"""Sensitivity rows at the final configuration (tag sensF).

Devlog 2026-09-28, "Author's directions on the handover audit", item 3:
nothing is quoted from pre-final settings. The kappa-form arms, the
false-positive floor, the alpha_w multiplier ladder and the adversary arms
are rerun at the final configuration (500/40, lineage labels, adversary
composition weight 3, w-only warm-up 30, the pinned alpha_z, kappa 0.1,
alpha_w 0.1, d_w 6, alpha_a 0.3, type_only) by
``scripts/queue_2026-09-28_sens_final.sh``, beside the controls
``finalL_s{0,1,2}`` (reused, never refit).

Each family's per-run reads come from its pre-final read-out script
(``r12_table``, ``fp_table``, ``awladder_table``, ``adv_table``: the same
files, keys and labels), so a row means what it meant there. Per read: the
control's 3-seed mean [min, max]; per arm its mean [min, max] and the move
of its mean in control seed-sds (ddof 1), flagged beyond one sd (the
``fp_table`` convention). A second table per section gives the 200 um
tile-bootstrap 95 % intervals (``bootstrap_ci.json``, ``recon_modes.json``;
the envelope over seeds, ``envelope_tables.ci_cell``) for the reads the
bootstrap covers. Nothing is adopted here: these are sensitivity rows. For
information only, the alpha_w family also carries the R19 ladder's
pre-registered guards and the adversary family the 8.17 rule's clauses,
both re-applied against finalL.

    uv run python scripts/sensF_tables.py [--families alpha_w adversary]

Writes ``<ovarian>/experiments/sensF_<family>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adv_table  # noqa: E402
import awladder_table  # noqa: E402
import envelope_tables  # noqa: E402
import fp_table  # noqa: E402
import r12_table  # noqa: E402
from discell.experiments.at_best import HEADER, battery_at_best  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "datasets"
GS = "gse315411_pdltma06_11_prime_solo"
OV = "xenium_prime_ovarian_cancer_ffpe"
FF = "xenium_prime_human_ovary_ff"
SHORT = {GS: "GSE core (pdl018d)", OV: "ovarian FFPE", FF: "ovary FF"}
OUT_DIR = ROOT / OV / "experiments"
CONTROL = "finalL_s{s}"
CONTROL_SEEDS = (0, 1, 2)
CYCLE_W_GUARD = 0.02
QUEUE = "scripts/queue_2026-09-28_sens_final.sh"

#: alpha_w = m / l-bar, l-bar = 1 / (2 alpha_z) at the pinned alpha_z (the
#: R19 ladder's convention, scripts/logs/awladder_2026-09-24/LADDER.json)
ALPHA_Z = {GS: 0.0018, OV: 0.0035, FF: 0.00035}
AW_RUNGS = {GS: (1, 2, 5, 10, 25), OV: (1, 2, 5, 10, 25), FF: (5, 10)}


def alpha_w(ds: str, m: int) -> float:
    return round(2.0 * m * ALPHA_Z[ds], 10)


def _fmt4(reads):
    """awladder_table.COLUMNS carry no format."""
    return [(k, label, "{:.4g}") for k, label in reads]


FAMILIES = {
    "kappa_form": {
        "title": "κ form (review R12 test 2)",
        "seeds": (0, 1, 2),
        "sections": {OV: {
            "depth": ("sensF_kdepth_s{s}", "--kappa-mode depth"),
            "gene": ("sensF_kgene_s{s}", "--kappa-mode gene"),
            "density": ("sensF_kdensity_s{s}", "--kappa-mode density")}},
        "reads": r12_table.READS,
        "read": lambda ds, run: r12_table.read_run(run, "best"),
        "per_seed": ["recon", "nmi", "cycle_z", "w_niche_mi_excess",
                     "read_a_of_ceiling", "read_a_own_gap_closed",
                     "read_b_own_gap_closed", "leak_lr_effect",
                     "response_lr_effect", "go_p0_extracellular",
                     "go_p0_plasma_membrane"],
        "note": "κ_i = κ·clip(r_i/m, 0, 5) ≤ 0.5 with m solved so the post-clip "
                "mean leak fraction over connected training cells is κ (depth: "
                "r_i = Σβ_ij ℓ_j/ℓ_i; density: the same on ℓ/A); gene: "
                "κ_g = min(κ·s_g/mean s_g, 0.9). The GO rows are atlas "
                "programme 0 (per-run `go_localisation_<run>.json`; the "
                "controls' own per-run files, same statistic as "
                "`go_localisation_lineage`).",
    },
    "fp_floor": {
        "title": "fixed false-positive floor (todo 8.15b)",
        "seeds": (0, 1, 2),
        "sections": {
            OV: {"per-section λ": ("sensF_fp_s{s}", "--fp-floor"),
                 "area λ_i": ("sensF_fparea_s{s}", "--fp-floor --fp-area")},
            GS: {"per-section λ": ("sensF_fp_s{s}", "--fp-floor")}},
        "reads": [(k, label.replace("uncontrolled500", "uncontrolledL"), fmt)
                  for k, label, fmt in fp_table.READS + fp_table.FLOOR_READS],
        "read": lambda ds, run: fp_table.read_run(ds, run, "best"),
        "per_seed": ["recon", "nmi", "probe_mlp_comp", "cycle_z",
                     "w_niche_mi_excess", "read_a_own_gap_closed",
                     "leak_lr_effect", "mean_share_leak",
                     "mean_share_response", "kl_w", "fp_cap_share"],
        "note": "p_i = (1 − κ − η_i) ρ_i + κ ρ̄_i + η_i/G, η_i = min(λ_i/ℓ_i, "
                "0.2), λ from the section's negative-control and "
                "genomic-control counts (per section, or ∝ segmented area "
                "with the section total fixed).",
    },
    "alpha_w": {
        "title": "α_w multiplier ladder (review R19)",
        "seeds": (0, 1),
        "sections": {ds: {f"m{m}": (f"sensF_awm{m}_s{{s}}",
                                    f"--alpha-w {alpha_w(ds, m):g}")
                          for m in rungs}
                     for ds, rungs in AW_RUNGS.items()},
        "reads": _fmt4(awladder_table.COLUMNS),
        "read": None,                       # set below: needs the rung's alpha_w
        "per_seed": ["recon", "nmi", "cycle_z", "cycle_w", "kl_w",
                     "niche_excess", "w_mirror", "gain", "dev_nmi",
                     "readA_own_gap"],
        "note": "α_w = m / ℓ̄, ℓ̄ = 1/(2 α_z) at the pinned α_z; the control "
                "(α_w 0.1) is m = 27.8 (GSE), 14.3 (ovarian), 143 (FF). "
                "Deviation reads (w-mirror, deviation NMI/cycle, held-out "
                "gain): `w_deviation.json` on every run, controls included.",
    },
    "adversary": {
        "title": "adversary capacity and composition weight (8.17)",
        "seeds": (0, 1),
        "sections": {OV: {
            "steps12": ("sensF_advsteps12_s{s}", "--adv-head-steps 12"),
            "width128": ("sensF_advwidth128_s{s}", "--adv-head-width 128"),
            "ens3": ("sensF_advens3_s{s}", "--adv-ensemble 3"),
            "comp1": ("sensF_advcomp1_s{s}",
                      "--adv-comp-weight 1 (the pre-final default)"),
            "comp5": ("sensF_advcomp5_s{s}", "--adv-comp-weight 5")}},
        "reads": [(k, label.replace("**", "").replace(" (decides)", ""), fmt)
                  for k, label, fmt in adv_table.READS],
        "read": lambda ds, run: adv_table.read_run(run, "best"),
        "per_seed": ["mlp_comp_u", "ridge_comp_u", "mlp_img_u", "ridge_img_u",
                     "nmi", "cycle_z", "mirror_r2", "recon",
                     "read_a_own_gap_closed"],
        "note": "Every arm keeps the final composition weight 3 unless it "
                "names another; head-step timing is not re-measured.",
    },
}

#: bootstrap-covered reads for the CI table: (label, envelope key, format)
CI_KEYS = ("recon", "nmi", "probe_ridge_comp_frac", "probe_mlp_comp_frac",
           "probe_ridge_img_frac", "probe_mlp_img_frac", "cycle_z", "cycle_w",
           "w_niche_mi_excess", "transport_of_ceiling",
           "transport_of_ceiling_trusted", "readA_gap_own")
CI_ROWS = [(label, key, fmt) for label, key, fmt in envelope_tables.ROWS
           if key in CI_KEYS]
#: the all-panel fraction-of-ceiling reads: kept in the JSON, not rendered
#: (author's decision 2026-09-28: the headline is the trusted tier under the
#: cell-split ceiling, as in the envelope tables)
ALL_PANEL_CEILING = {"transport_of_ceiling", "read_a_of_ceiling",
                     "read_b_of_ceiling", "readA_of_ceiling",
                     "readB_of_ceiling"}


# -- reading ----------------------------------------------------------------------

def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _write(path: Path, text: str) -> None:
    """Atomic: queue lanes may refresh the partial tables at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def reader(family: str, ds: str, arm: str | None):
    """The family's read_run for one dataset and arm (None = the control)."""
    if family != "alpha_w":
        return FAMILIES[family]["read"]
    aw = 0.1 if arm is None else alpha_w(ds, int(arm[1:]))
    return lambda ds_, run: awladder_table.read_run(ds_, run, aw, "best")


def guard_record(ds: str, run: str) -> dict | None:
    """Guard fields read the same way for every family."""
    d = ROOT / ds / "runs" / run
    metrics = _load(d / "metrics.json")
    if metrics is None:
        return None
    battery = battery_at_best(d)
    w = (_load(d / "degeneracy.json") or {}).get("w_channel") or {}
    probe = _load(d / "validation" / "probe_blocks.json") or {}
    cycle_w = ((battery.get("cycle") or {}).get("w") or {}).get("r2_pooled")
    return {"run": run,
            "dead_w_channel": bool(metrics.get("dead_w_channel")),
            "guard_dead": w.get("dead_context_channel"),
            "niche_excess": w.get("w_niche_mi_excess"),
            "cycle_w_ok": _finite(cycle_w) and cycle_w <= CYCLE_W_GUARD,
            "invariance_pass": probe.get("invariance_pass"),
            "at_best": bool(battery.get("at_best", True))}


def config_diff(ds: str, run: str, seed: int) -> list[str] | None:
    """Keys where the run's config.json differs from finalL_s<seed>'s."""
    runs = ROOT / ds / "runs"
    new, ref = _load(runs / run / "config.json"), _load(
        runs / CONTROL.format(s=seed) / "config.json")
    if new is None or ref is None:
        return None
    return sorted(k for k in set(new) | set(ref)
                  if k not in ("run_name", "seed", "git")
                  and new.get(k, "<absent>") != ref.get(k, "<absent>"))


def summarise(records: list[dict], keys) -> dict:
    out = {"runs": [r["run"] for r in records], "n_seeds": len(records)}
    for key in keys:
        vals = [r[key] for r in records if _finite(r.get(key))]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                        "max": float(max(vals)), "n": len(vals),
                        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1
                        else None}
    return out


def move(control: dict, arm: dict, key: str) -> dict:
    """The arm mean's move in control seed-sds; flag beyond one sd."""
    a, b = control.get(key), arm.get(key)
    if not a or not b:
        return {"delta": None, "in_sd": None, "flag": None}
    delta = b["mean"] - a["mean"]
    if not a.get("sd"):
        return {"delta": delta, "in_sd": None, "flag": None}
    return {"delta": delta, "in_sd": delta / a["sd"],
            "flag": bool(abs(delta) > a["sd"])}


def read_group(family: str, ds: str, arm: str | None, template: str,
               seeds) -> dict:
    """Records, summaries, guards and CI records of one seed group."""
    spec = FAMILIES[family]
    read = reader(family, ds, arm)
    names = [template.format(s=s) for s in seeds]
    records = [r for r in (read(ds, n) for n in names) if r is not None]
    guards = [g for g in (guard_record(ds, n) for n in names) if g is not None]
    cis = [c for c in (envelope_tables.run_record(ds, n, "best") for n in names)
           if c is not None]
    keys = [k for k, _, _ in spec["reads"]]
    out = {"names": names, "records": records, "guards": guards, "ci_records": cis,
           "summary": summarise(records, keys)}
    if arm is not None:
        out["config_diff"] = {n: config_diff(ds, n, s)
                              for n, s in zip(names, seeds)}
    return out


def collect(family: str) -> dict:
    spec = FAMILIES[family]
    sections = {}
    for ds, arms in spec["sections"].items():
        control = read_group(family, ds, None, CONTROL, CONTROL_SEEDS)
        groups = {arm: read_group(family, ds, arm, template, spec["seeds"])
                  for arm, (template, _) in arms.items()}
        moves = {arm: {k: move(control["summary"], g["summary"], k)
                       for k, _, _ in spec["reads"]}
                 for arm, g in groups.items()}
        sections[ds] = {"control": control, "arms": groups, "moves": moves}
        if family == "alpha_w":
            sections[ds]["ladder_guards"] = ladder_guards(ds, control, groups)
        if family == "adversary":
            sections[ds]["rule_817"] = rule_817(ds, groups)
    return sections


# -- the families' own reads, for information ---------------------------------------

def _aw_cell(records: list[dict], n_expected: int, aw: float) -> dict:
    return {"alpha_w": aw, "runs": records, "n_expected": n_expected,
            "names": [r["run"] for r in records],
            **awladder_table.summarise(records)}


def ladder_guards(ds: str, control: dict, groups: dict) -> dict:
    """The R19 ladder's guards and gain clause, rung vs the finalL triple."""
    awladder_table.SEEDS = FAMILIES["alpha_w"]["seeds"]   # the rungs' seeds
    ref = _aw_cell(control["records"], len(CONTROL_SEEDS), 0.1)
    out = {}
    for arm, g in groups.items():
        cell = _aw_cell(g["records"], len(FAMILIES["alpha_w"]["seeds"]),
                        alpha_w(ds, int(arm[1:])))
        out[arm] = {"guards": awladder_table.guards(cell, ref),
                    "gain": awladder_table.gain_clause(cell, ref)}
    return out


def rule_817(ds: str, groups: dict) -> dict:
    """The 8.17 rule's clauses, seed-paired against finalL_s{0,1}."""
    seeds = FAMILIES["adversary"]["seeds"]
    adv_table.SEEDS = list(seeds)
    control = [r for r in (adv_table.read_run(CONTROL.format(s=s), "best")
                           for s in seeds) if r is not None]
    ctl = adv_table.summarise(control) if control else None
    return {arm: adv_table.judge(adv_table.summarise(g["records"])
                                 if g["records"] else None, ctl)
            for arm, g in groups.items()}


# -- rendering ------------------------------------------------------------------------

def _cell(entry: dict | None, fmt: str, n_want: int) -> str:
    if not entry:
        return "--"
    text = (f"{fmt.format(entry['mean'])} [{fmt.format(entry['min'])}, "
            f"{fmt.format(entry['max'])}]")
    return text + ("" if entry["n"] == n_want else f" (n={entry['n']})")


def _count(guards: list[dict], test) -> str:
    return f"{sum(bool(test(g)) for g in guards)}/{len(guards)}" if guards else "--"


def render_section(family: str, ds: str, sec: dict) -> list[str]:
    spec = FAMILIES[family]
    arms = spec["sections"][ds]
    names = list(arms)
    n_arm = len(spec["seeds"])
    ctl = sec["control"]
    L = [f"## {SHORT[ds]} (`{ds}`)", "",
         "Control: " + ", ".join(f"`{n}`" for n in ctl["names"]) + ". Arms: "
         + "; ".join(f"**{a}** `{flags}` ({', '.join(sec['arms'][a]['names'])})"
                     for a, (_, flags) in arms.items()) + ".", ""]
    diffs = []
    for a in names:
        keys = sorted({k for v in sec["arms"][a]["config_diff"].values() if v
                       for k in v})
        landed = [v for v in sec["arms"][a]["config_diff"].values() if v is not None]
        if landed:
            diffs.append(f"{a}: " + (", ".join(keys) if keys else "nothing"))
    L += ["Config check (keys where an arm's config.json differs from "
          "`finalL_s<seed>`'s, run name/seed/git aside): "
          + ("; ".join(diffs) if diffs else "no arm fit has landed") + ".", ""]
    # guard counts
    L += ["| guard counts | control | " + " | ".join(names) + " |",
          "|---" * (len(names) + 2) + "|"]
    groups = [ctl] + [sec["arms"][a] for a in names]
    for label, test in (
            ("seeds landed", None),
            ("dead KL_w channel (trainer)", lambda g: g["dead_w_channel"]),
            ("dead context channel (guard)", lambda g: g["guard_dead"]),
            ("I(niche; w) excess > 0", lambda g: _finite(g["niche_excess"])
             and g["niche_excess"] > 0),
            (f"cycle_w ≤ {CYCLE_W_GUARD}", lambda g: g["cycle_w_ok"]),
            ("invariance_pass (every probe block ≤ 0.25 u)",
             lambda g: g["invariance_pass"])):
        if test is None:
            cells = [f"{len(g['guards'])}/{len(g['names'])}" for g in groups]
        else:
            cells = [_count(g["guards"], test) for g in groups]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    # the reads
    L += ["", "| read | control | " + " | ".join(f"{a} | Δ/sd" for a in names)
          + " |", "|---" * (2 + 2 * len(names)) + "|"]
    for key, label, fmt in spec["reads"]:
        if key in ALL_PANEL_CEILING:
            continue
        cells = [_cell(ctl["summary"].get(key), fmt, len(CONTROL_SEEDS))]
        for a in names:
            m = sec["moves"][a][key]
            cells += [_cell(sec["arms"][a]["summary"].get(key), fmt, n_arm),
                      ("--" if m["in_sd"] is None else f"{m['in_sd']:+.2f}")
                      + (" ⚑" if m["flag"] else "")]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    for a in names:
        moved = [label for key, label, _ in spec["reads"]
                 if key not in ALL_PANEL_CEILING and sec["moves"][a][key]["flag"]]
        L.append(f"- Moved by more than one control seed-sd, **{a}**: "
                 + (", ".join(moved) if moved else "none") + ".")
    # tile-bootstrap intervals
    L += ["", "### Tile-bootstrap 95 % intervals", "",
          "200 µm tiles, 1000 draws, conditional on the fitted objects "
          "(`discell/experiments/bootstrap.py`; reconstruction from "
          "`recon_modes.json`). Cell: the envelope of the per-seed intervals "
          "[lowest lower, highest upper] (seeds carrying one); † = a "
          "recomputed point did not reproduce the stored one on some seed.", "",
          "| read | control | " + " | ".join(names) + " |",
          "|---" * (len(names) + 2) + "|"]
    for label, key, fmt in CI_ROWS:
        if key in ALL_PANEL_CEILING:
            continue
        cells = [envelope_tables.ci_cell(g["ci_records"], key, fmt) or "--"
                 for g in groups]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    # family extras
    if family == "alpha_w":
        L += render_ladder_guards(ds, sec["ladder_guards"])
    if family == "adversary":
        L += render_rule_817(sec["rule_817"])
    # per seed
    per = [k for k in spec["per_seed"] if k not in ALL_PANEL_CEILING]
    L += ["", "### Per seed", "",
          "| run | best epoch | " + " | ".join(per) + " |",
          "|---" * (len(per) + 2) + "|"]
    for g in groups:
        for r in g["records"]:
            L.append(f"| {r['run']} | {r.get('best_epoch', '--')} | " + " | ".join(
                "--" if not _finite(r.get(k)) else f"{r[k]:.4g}" for k in per)
                + " |")
    missing = [n for g in groups for n in g["names"]
               if n not in {r["run"] for r in g["records"]}]
    fell_back = [g_["run"] for g in groups for g_ in g["guards"] if not g_["at_best"]]
    L += ["", "Missing fits: " + (", ".join(missing) if missing else "none")
          + ". Fallbacks to the last epoch: "
          + (", ".join(fell_back) if fell_back else "none") + ".", ""]
    return L


def render_ladder_guards(ds: str, guards: dict) -> list[str]:
    primary = ds in awladder_table.PRIMARY
    cols = awladder_table.GUARDS
    L = ["", "### The R19 ladder's guards, re-applied against finalL "
         "(information; nothing is re-decided)", "",
         "Reference = the finalL triple (α_w 0.1); rung seeds 0 1. Guard "
         "definitions: `scripts/awladder_table.py` docstring. The probe guard "
         "is `invariance_pass` per seed, which the control itself may fail "
         "(guard counts above). Gain clause: rung mean held-out gain minus "
         "the control's, against the control's seed range"
         + (" (a clause on this primary dataset)." if primary
            else " (reported; not a clause on this dataset)."), "",
         "| rung | " + " | ".join(cols) + " | gain − control gain | bar | gain clause |",
         "|---" * (len(cols) + 4) + "|"]
    for arm, e in guards.items():
        g, gc = e["guards"], e["gain"]
        L.append(f"| {arm} | " + " | ".join(awladder_table._mark(g.get(k))
                                          for k in cols) + " | "
                 + (f"{gc['excess']:+.5f} | {gc['bar']:.5f} | "
                    if gc.get("pass") is not None else "-- | -- | ")
                 + awladder_table._mark(gc.get("pass")) + " |")
    return L


def render_rule_817(rule: dict) -> list[str]:
    show = lambda v: "--" if v is None else ("yes" if v else "no")
    L = ["", "### The 8.17 rule's clauses, re-applied against finalL_s0/s1 "
         "(information; weight 3 is adopted and nothing is re-decided)", "",
         "Seed-paired MLP composition fall beyond the control's seed range, "
         "guards inside the control envelope widened by one sd, absolute "
         "guards, recon not worse (`scripts/adv_table.py` docstring).", "",
         "| arm | all clauses | MLP comp falls | guards in envelope | absolute "
         "guards | recon not worse | notes |", "|---|---|---|---|---|---|---|"]
    for arm, v in rule.items():
        c = v["clauses"]
        L.append(f"| {arm} | {show(v['qualifies'])} | {show(c['probe_falls'])} | "
                 f"{show(c['guards_inside_envelope'])} | "
                 f"{show(c['absolute_guards'])} | {show(c['recon_not_worse'])} | "
                 f"{'; '.join(v['notes'])} |")
    return L


def render(family: str, sections: dict) -> str:
    spec = FAMILIES[family]
    seeds = " ".join(str(s) for s in spec["seeds"])
    L = [f"# Sensitivity rows at the final configuration: {spec['title']} "
         f"-- {HEADER}", "",
         "Final configuration (500/40, lineage labels, adversary composition "
         "weight 3, w-only warm-up 30, the pinned α_z, κ 0.1, α_w 0.1, d_w 6, "
         "α_a 0.3, type_only) but for each arm's own flag; tag sensF, "
         f"`{QUEUE}`. Control = `finalL_s{{0,1,2}}` (reused, never refit). "
         f"Arm seeds {seeds}; each arm seed shares its data split with the "
         "control of the same seed. Cells: seed mean [min, max]. Δ/sd: the "
         "arm mean's move in control seed-sds (ddof 1); ⚑ = more than one sd. "
         "Probe fractions are taken against `uncontrolledL_s{0,1}`. "
         "Sensitivity rows: no rule is applied and nothing is adopted. The "
         "pre-final tables (200/20, cell-type labels, weight 1) stay beside "
         "this one, labelled pre-final; their runs are in `runs/_archive/`.",
         "", spec["note"], ""]
    for ds, sec in sections.items():
        L += render_section(family, ds, sec)
    return "\n".join(L) + "\n"


def _jsonable(sections: dict) -> dict:
    out = {}
    for ds, sec in sections.items():
        entry = {"control": {k: v for k, v in sec["control"].items()
                             if k != "ci_records"},
                 "arms": {a: {k: v for k, v in g.items() if k != "ci_records"}
                          for a, g in sec["arms"].items()},
                 "moves": sec["moves"]}
        entry["ci"] = {name: {key: envelope_tables.ci_cell(g["ci_records"], key, fmt)
                              for _, key, fmt in CI_ROWS}
                       for name, g in [("control", sec["control"])]
                       + list(sec["arms"].items())}
        for extra in ("ladder_guards", "rule_817"):
            if extra in sec:
                entry[extra] = sec[extra]
        out[ds] = entry
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--families", nargs="+", default=list(FAMILIES),
                        choices=list(FAMILIES))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    for family in args.families:
        sections = collect(family)
        text = render(family, sections)
        if not args.quiet:
            print(text)
        stem = Path(args.out_dir) / f"sensF_{family}"
        _write(stem.with_suffix(".md"), text)
        _write(stem.with_suffix(".json"), json.dumps(
            {"family": family, "read_at": HEADER, "queue": QUEUE,
             "control": CONTROL, "control_seeds": list(CONTROL_SEEDS),
             "arm_seeds": list(FAMILIES[family]["seeds"]),
             "arms": {ds: {a: flags for a, (_, flags) in arms.items()}
                      for ds, arms in FAMILIES[family]["sections"].items()},
             "flag_rule": "|mean_arm - mean_control| > sd_control (ddof 1)",
             "sections": _jsonable(sections)}, indent=1, default=float))
        print(f"wrote {stem.with_suffix('.md')}, {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
