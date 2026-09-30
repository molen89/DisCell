#!/usr/bin/env python3
"""The breakdown point kappa* of every signed contrast, per section (todo 8.19).

Spec: devlog "Breakdown-point nulls decided (author, 2026-09-29)", "8.19
breakdown table: classification and family (author, 2026-09-29;
motivation)" (pooling, interval) and "8.19 revised to a lean version
(author, 2026-09-29)" (the family and the rule, which replace the earlier
entry's); method section 2.8, Definition 1. The family of each section
and every member's formula are in
``scripts/logs/breakdown_2026-09-29/FAMILY.md``.

**Input.** Per fit, the per-draw statistics that
:mod:`discell.experiments.breakdown_draws` writes to
``runs/<run>/breakdown/<group>.npz``: for every member its point estimate
(the statistic at unit weights) and ``n`` draws, each draw resampling the
200 um tiles of that fit's scored cells (``bootstrap.tile_index``),
conditional on the fitted objects.

**Pooling (fixed by the devlog).** The interval at a grid point is for the
3-seed mean. Draw b of the pool is the mean of draw b of the three seeds'
fits; each seed's draws come from its own generator, so the tiles are
resampled independently within each seed. Two interval forms, as in
``bootstrap.py``:

``percentile``  the [a/2, 1 - a/2] percentiles of the pooled draws;
``subsample``   half-tile subsampling (the transport R^2 members): each
                seed's draws are centred on their own median and scaled by
                its sqrt(m/(n-m)), the scaled deviations are averaged over
                seeds, and their percentiles are placed on the pooled point
                (``bootstrap.subsample_interval``'s reverse form).

The level is two-sided ``1 - 0.05 / m_s`` with ``m_s`` the section's family
size (:data:`FAMILY_SIZE`, the counts of FAMILY.md).

**kappa* (the lean entry's rule).** Null 0 for every member. A contrast
*holds* at a grid kappa if every seed's estimate has the sign of the
pooled kappa = 0 estimate and the pooled interval excludes 0. kappa* is the
first grid kappa where it fails: "no finding" if it fails at kappa = 0,
"above the grid" if it never fails. A grid point without the member (no
seed carries it) before any failure stops the scan: "gap at kappa" (never
read as survival). Every member is judged on all three seeds: a grid point
that some seed lacks counts as a gap (author, 2026-09-29: the transport
members are scored on all panels, so every section has them on 3 seeds).

**Rankings.** :func:`ranking_pairs` turns the members of a ranking into its
pairwise differences on the same draws (each then gets its own kappa*). No
member of the current families is a ranking (FAMILY.md).

**Trajectory table (no kappa*).** Magnitudes and the Moran's I of mu_z and
mu_w, only from reads that already exist on every sweep fit: 3-seed mean
and [min, max] per grid point.

Usage::

    python -m discell.experiments.breakdown --section ovarian
    python -m discell.experiments.breakdown --all      # + the combined table

Writes ``data/datasets/<ds>/experiments/breakdown{,_trajectory}.{json,md}``
per section (the dual: under its own dataset) and, with ``--all``,
``scripts/logs/breakdown_2026-09-29/breakdown_all{,_trajectory}.{json,md}``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

log = logging.getLogger("discell.experiments.breakdown")

GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
SEEDS = (0, 1, 2)
FAMILY_ALPHA = 0.05
OV = "xenium_prime_ovarian_cancer_ffpe"
LU = "xenium_prime_human_lung_cancer_ffpe"
FF = "xenium_prime_human_ovary_ff"
GS = "gse315411_pdltma06_11_prime_solo"
GD = "gse315411_pdltma06_10_prime_dual"
#: section -> (dataset the fits live in, dataset read, dual?)
SECTIONS = {"ovarian": (OV, OV, False), "lung": (LU, LU, False),
            "ff": (FF, FF, False), "gse": (GS, GS, False),
            "gse_dual": (GS, GD, True)}
#: m_s, the lean family's sizes (devlog "8.19 revised to a lean version";
#: FAMILY.md): 8 headline contrasts, + the axis test on ovarian; the dual
#: carries the cycle asymmetry only
FAMILY_SIZE = {"ovarian": 9, "lung": 8, "ff": 8, "gse": 8, "gse_dual": 1}
#: the draw groups of a section (breakdown_draws.GROUPS); the dual: cycle only
GROUPS = ("cycle", "w_mi", "marker", "signalling", "axis", "transport_mean",
          "transport_dist")
DUAL_GROUPS = ("cycle",)
LOG_ROOT = Path("scripts/logs/breakdown_2026-09-29")


def run_name(kappa: float, seed: int) -> str:
    """The fit at a grid point: kappa = 0.1 is finalL (sweepL_k0.1 links it)."""
    return (f"finalL_s{seed}" if np.isclose(kappa, 0.1)
            else f"sweepL_k{kappa:g}_s{seed}")


def draws_path(run_dir: Path, group: str, eval_dataset: str | None = None
               ) -> Path:
    stem = group if eval_dataset is None else f"dual__{eval_dataset}__{group}"
    return Path(run_dir) / "breakdown" / f"{stem}.npz"


def level(m_s: int, alpha: float = FAMILY_ALPHA) -> float:
    """The two-sided Bonferroni level ``1 - alpha / m_s``."""
    if m_s < 1:
        raise ValueError("a family has at least one member")
    return 1.0 - alpha / m_s


# --------------------------------------------------------------------------
# reading the per-fit draws
# --------------------------------------------------------------------------


def read_draws(path: Path) -> dict:
    """``{member: {"estimate", "draws", "method", "c"}}`` from one npz."""
    with np.load(path, allow_pickle=False) as f:
        names = [str(x) for x in f["members"]]
        out = {}
        for i, name in enumerate(names):
            out[name] = {"estimate": float(f["estimate"][i]),
                         "draws": np.asarray(f["draws"][i], dtype=np.float64),
                         "method": str(f["method"][i]),
                         "c": float(f["c"][i])}
    return out


def collect(section: str, grid: Sequence[float] = GRID,
            seeds: Sequence[int] = SEEDS, root: Path | None = None) -> dict:
    """``{kappa: {seed: {member: entry}}}`` from every fit's draw files."""
    from discell import paths

    ds, read_ds, dual = SECTIONS[section]
    root = Path(root) if root else paths.dataset(ds).root
    groups = DUAL_GROUPS if dual else GROUPS
    out: dict = {}
    for kappa in grid:
        out[kappa] = {}
        for seed in seeds:
            run_dir = root / "runs" / run_name(kappa, seed)
            members: dict = {}
            for group in groups:
                p = draws_path(run_dir, group, read_ds if dual else None)
                if p.exists():
                    members.update(read_draws(p))
            out[kappa][seed] = members
    return out


# --------------------------------------------------------------------------
# pooling and the interval
# --------------------------------------------------------------------------


def pool(per_seed: Sequence[Mapping], alpha: float) -> dict | None:
    """The 3-seed pooled point and two-sided ``1 - alpha`` interval of one
    member at one grid point, from each seed's ``{estimate, draws, method,
    c}`` (seeds without a finite estimate are left out and counted)."""
    entries = [e for e in per_seed
               if e is not None and np.isfinite(e["estimate"])]
    if not entries:
        return None
    methods = {e["method"] for e in entries}
    if len(methods) != 1:
        raise ValueError(f"seeds disagree on the interval form: {methods}")
    method = methods.pop()
    n = min(len(e["draws"]) for e in entries)
    point = float(np.mean([e["estimate"] for e in entries]))
    if method == "percentile":
        stack = np.stack([e["draws"][:n] for e in entries])
    elif method == "subsample":
        stack = np.stack([e["c"] * (e["draws"][:n]
                                    - np.nanmedian(e["draws"][:n]))
                          for e in entries])
    else:
        raise ValueError(f"unknown interval form {method!r}")
    ok = np.isfinite(stack).all(axis=0)
    pooled = stack[:, ok].mean(axis=0)
    if len(pooled) < 2:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(pooled, [100 * alpha / 2,
                                        100 * (1 - alpha / 2)])
        if method == "subsample":
            lo, hi = point + lo, point + hi
    return {"estimate": point, "ci": [float(lo), float(hi)],
            "per_seed": [float(e["estimate"]) for e in entries],
            "n_seeds": len(entries), "n_draws": int(ok.sum()),
            "method": method}


def contains_zero(ci: Sequence[float]) -> bool:
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return True                 # an undefined interval excludes nothing
    return bool(lo <= 0.0 <= hi)


def holds(entry: dict | None, sign: int) -> bool:
    """Every seed has *sign* and the pooled interval excludes 0."""
    return (entry is not None and not contains_zero(entry["ci"])
            and all(int(np.sign(v)) == sign for v in entry["per_seed"]))


def breakdown_point(trajectory: Mapping[float, dict | None],
                    grid: Sequence[float] = GRID) -> dict:
    """kappa* of one member from its pooled grid points (the lean entry's
    rule; module docstring)."""
    first = trajectory.get(grid[0])
    if first is None:
        return {"status": "missing at kappa = 0", "kappa_star": None}
    sign = int(np.sign(first["estimate"]))
    if not holds(first, sign):
        return {"status": "no finding", "kappa_star": None, "sign": sign,
                "reason": ("interval contains 0" if contains_zero(first["ci"])
                           else "seeds disagree in sign")}
    for kappa in grid[1:]:
        entry = trajectory.get(kappa)
        if entry is None:
            return {"status": f"gap at kappa = {kappa:g}", "kappa_star": None,
                    "sign": sign}
        if not holds(entry, sign):
            return {"status": "breaks", "kappa_star": float(kappa),
                    "sign": sign,
                    "reason": ("interval contains 0" if contains_zero(entry["ci"])
                               else "a seed flips sign")}
    return {"status": "above the grid", "kappa_star": None, "sign": sign}


def ranking_pairs(members: Mapping[str, dict], names: Sequence[str]) -> dict:
    """The pairwise differences ``a - b`` of a ranking's members, on the
    same draws of one fit (the members of one read share their tile draws)."""
    out = {}
    for a, b in itertools.combinations(names, 2):
        ea, eb = members[a], members[b]
        if ea["method"] != eb["method"]:
            raise ValueError("a ranking's members share one interval form")
        n = min(len(ea["draws"]), len(eb["draws"]))
        out[f"{a} - {b}"] = {"estimate": ea["estimate"] - eb["estimate"],
                             "draws": ea["draws"][:n] - eb["draws"][:n],
                             "method": ea["method"], "c": ea["c"]}
    return out


def section_table(section: str, collected: Mapping | None = None,
                  m_s: int | None = None, grid: Sequence[float] = GRID,
                  seeds: Sequence[int] = SEEDS) -> dict:
    """Every member of a section: pooled points and intervals per grid
    point, and kappa*."""
    collected = collect(section, grid, seeds) if collected is None else collected
    m_s = FAMILY_SIZE[section] if m_s is None else m_s
    alpha = FAMILY_ALPHA / m_s
    names = sorted({m for k in collected.values() for s in k.values()
                    for m in s})
    members = {}
    for name in names:
        traj = {}
        for kappa in grid:
            entry = pool([collected[kappa][s].get(name)
                          for s in collected[kappa]], alpha)
            # all seeds or nothing: a point some seed lacks is a gap
            traj[kappa] = (entry if entry is not None
                           and entry["n_seeds"] == len(collected[kappa])
                           else None)
        members[name] = {"trajectory": {f"{k:g}": v for k, v in traj.items()},
                         **breakdown_point(traj, grid)}
    found = len(names)
    return {"section": section,
            "dataset": SECTIONS.get(section, (None,) * 3)[1],
            "m_s": m_s, "level": level(m_s), "alpha_per_side": alpha / 2,
            "grid": list(grid), "seeds": list(seeds),
            "members_found": found, "family_complete": found == m_s,
            "members": members}


# --------------------------------------------------------------------------
# the trajectory-only readouts (no kappa*)
# --------------------------------------------------------------------------


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _moran(record, latent):
    """Within-type Moran's I of a latent, averaged over dimensions weighted
    by within-type variance (the appendix table's convention)."""
    m = ((record or {}).get("morans") or {}).get(latent)
    if not m:
        return None
    i, v = np.asarray(m["I"], float), np.asarray(m["var_per_dim"], float)
    return float((i * v).sum() / v.sum()) if v.sum() > 0 else None


TRAJECTORY_SOURCES = {
    "transport_of_ceiling_trusted": "runs/<run>/transport/transport.json "
    "summary.extrapolation_trusted.counterfactual_of_ceiling",
    "transport_of_ceiling": "same, summary.extrapolation",
    "w_within_type_var_fraction": "degeneracy.json w_channel."
    "w_var_fraction_across_cells (tr Cov(w|t) / tr Cov(w))",
    "w_within_type_var": "the same fraction x w_total_var",
    "moran_mu_z": "validation/validation.json morans.mu_z, variance-weighted",
    "moran_mu_w": "validation/validation.json morans.mu_w, variance-weighted",
    "atlas_effective_rank": "atlas/programs.npy columns",
    "kappa_survival_overlap": "experiments/atlas_kappa_survival.json, shift "
    "overlap vs finalL_s0 (mean of both directions)",
}


def trajectory_values(run_dir: Path, dataset_root: Path, run: str) -> dict:
    """The trajectory readouts of one fit, each from the file named in
    :data:`TRAJECTORY_SOURCES`."""
    out: dict = {}
    tr = _load(run_dir / "transport" / "transport.json") or {}
    summ = tr.get("summary") or {}
    for tier, key in (("extrapolation_trusted", "transport_of_ceiling_trusted"),
                      ("extrapolation", "transport_of_ceiling")):
        out[key] = (summ.get(tier) or {}).get("counterfactual_of_ceiling")
    deg = (_load(run_dir / "degeneracy.json") or {}).get("w_channel") or {}
    if "w_var_fraction_across_cells" in deg:
        out["w_within_type_var_fraction"] = deg["w_var_fraction_across_cells"]
        out["w_within_type_var"] = (deg["w_var_fraction_across_cells"]
                                    * deg["w_total_var"])
    val = _load(run_dir / "validation" / "validation.json")
    out["moran_mu_z"] = _moran(val, "mu_z")
    out["moran_mu_w"] = _moran(val, "mu_w")
    prog = run_dir / "atlas" / "programs.npy"
    out["atlas_effective_rank"] = (float(np.load(prog).shape[1])
                                   if prog.exists() else None)
    surv = ((_load(dataset_root / "experiments" / "atlas_kappa_survival.json")
             or {}).get("survival") or {})
    alias = (f"sweepL_k0.1_{run.split('_')[-1]}" if run.startswith("finalL_")
             else run)
    ov = (surv.get(alias) or {}).get("shift_overlap")
    out["kappa_survival_overlap"] = (0.5 * (ov["this_inside_other"]
                                            + ov["other_inside_this"])
                                     if ov else None)
    return {k: (float(v) if v is not None and np.isfinite(v) else None)
            for k, v in out.items()}


def trajectory_table(section: str, grid: Sequence[float] = GRID,
                     seeds: Sequence[int] = SEEDS, root: Path | None = None
                     ) -> dict:
    from discell import paths

    ds, read_ds, dual = SECTIONS[section]
    root = Path(root) if root else paths.dataset(ds).root
    rows: dict = {}
    if not dual:                   # no trajectory readout is read on the dual
        for kappa in grid:
            for seed in seeds:
                run = run_name(kappa, seed)
                for key, value in trajectory_values(root / "runs" / run, root,
                                                    run).items():
                    rows.setdefault(key, {}).setdefault(f"{kappa:g}",
                                                        []).append(value)
    table = {}
    for key, per in rows.items():
        table[key] = {}
        for k, vals in per.items():
            v = [x for x in vals if x is not None]
            table[key][k] = ({"mean": float(np.mean(v)), "min": float(min(v)),
                              "max": float(max(v)), "n": len(v)} if v
                             else None)
    return {"section": section, "dataset": read_ds, "grid": list(grid),
            "note": "trajectory only, no kappa* (devlog 2026-09-29): point "
                    "estimate = 3-seed mean, [min, max] over seeds",
            "sources": TRAJECTORY_SOURCES, "readouts": table}


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def _fmt(v, d=3):
    return "–" if v is None or not np.isfinite(v) else f"{v:+.{d}f}"


def _kstar(entry):
    if entry["status"] == "breaks":
        return f"{entry['kappa_star']:g} ({entry['reason']})"
    return entry["status"]


def markdown(table: dict) -> str:
    grid = table["grid"]
    lines = [f"# Breakdown points — {table['section']} ({table['dataset']})",
             "", f"Family size m_s = {table['m_s']}; members with draws on "
             f"disk: {table['members_found']}. Two-sided level 1 − 0.05/m_s = "
             f"{table['level']:.5f}. Cells: 3-seed mean [interval]. Null 0; "
             "holds at κ if every seed keeps the κ = 0 sign and the interval excludes 0; κ* = first κ where it fails (devlog, lean entry).", ""]
    if not table["family_complete"]:
        lines += [f"**Incomplete:** {table['members_found']} of "
                  f"{table['m_s']} members have draws; the level still uses "
                  "m_s.", ""]
    lines += ["| member | " + " | ".join(f"κ = {k:g}" for k in grid)
              + " | κ* |", "|---" * (len(grid) + 2) + "|"]
    for name, m in table["members"].items():
        cells = []
        for k in grid:
            e = m["trajectory"][f"{k:g}"]
            cells.append("–" if e is None else
                         f"{_fmt(e['estimate'])} [{_fmt(e['ci'][0])}, "
                         f"{_fmt(e['ci'][1])}]")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {_kstar(m)} |")
    return "\n".join(lines) + "\n"


def trajectory_markdown(table: dict) -> str:
    grid = table["grid"]
    lines = [f"# Trajectory readouts — {table['section']} ({table['dataset']})",
             "", table["note"] + ".", "",
             "| readout | " + " | ".join(f"κ = {k:g}" for k in grid) + " |",
             "|---" * (len(grid) + 1) + "|"]
    for key, per in table["readouts"].items():
        cells = []
        for k in grid:
            e = per.get(f"{k:g}")
            cells.append("–" if e is None else
                         f"{e['mean']:.3f} [{e['min']:.3f}, {e['max']:.3f}]"
                         + ("" if e["n"] == 3 else f" (n={e['n']})"))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "Sources:"] + [f"- `{k}`: {v}" for k, v in
                                 table["sources"].items()]
    return "\n".join(lines) + "\n"


def combined_markdown(tables: Mapping[str, dict]) -> str:
    lines = ["# Breakdown points, all sections (todo 8.19)", "",
             "κ* per member and section (Bonferroni within each section, "
             "m_s in the header; · = not in that section's family). Details: "
             "each section's `experiments/breakdown.md`.", ""]
    sections = list(tables)
    names = sorted({n for t in tables.values() for n in t["members"]})
    lines += ["| member | " + " | ".join(
        f"{s} (m={tables[s]['m_s']})" for s in sections) + " |",
        "|---" * (len(sections) + 1) + "|"]
    for name in names:
        cells = [(_kstar(tables[s]["members"][name])
                  if name in tables[s]["members"] else "·") for s in sections]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_section(section: str) -> tuple[dict, dict]:
    from discell import paths

    table = section_table(section)
    traj = trajectory_table(section)
    out = paths.dataset(SECTIONS[section][1]).root / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "breakdown.json").write_text(json.dumps(table, indent=1))
    (out / "breakdown.md").write_text(markdown(table))
    if traj["readouts"]:
        (out / "breakdown_trajectory.json").write_text(json.dumps(traj,
                                                                  indent=1))
        (out / "breakdown_trajectory.md").write_text(trajectory_markdown(traj))
    log.info("%s: %d/%d members -> %s", section, table["members_found"],
             table["m_s"], out / "breakdown.md")
    return table, traj


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--section", choices=list(SECTIONS), action="append",
                        default=[])
    parser.add_argument("--all", action="store_true",
                        help="every section, plus the combined tables")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    sections = list(SECTIONS) if args.all else args.section
    if not sections:
        parser.error("--section or --all")
    tables, trajs = {}, {}
    for s in sections:
        tables[s], trajs[s] = write_section(s)
    if args.all:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        (LOG_ROOT / "breakdown_all.json").write_text(json.dumps(tables,
                                                                indent=1))
        (LOG_ROOT / "breakdown_all.md").write_text(combined_markdown(tables))
        (LOG_ROOT / "breakdown_all_trajectory.json").write_text(
            json.dumps(trajs, indent=1))
        (LOG_ROOT / "breakdown_all_trajectory.md").write_text("\n".join(
            trajectory_markdown(t) for t in trajs.values() if t["readouts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
