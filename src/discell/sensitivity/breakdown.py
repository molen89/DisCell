"""Breakdown points: the smallest tested spillover fraction at which a finding fails.

Input per finding: for every grid kappa and seed, one fit's draws
``{"estimate", "draws", "method", "c"}`` (:func:`discell.sensitivity.finding_draws`).

**Pooling.** At each grid point the interval is for the mean over seeds, by a
two-level bootstrap. Each draw picks the seeds' fits with replacement, takes an
independent tile draw of each pick and averages them, so the interval covers the
spread between fits as well as over the tissue. ``percentile``: a pick's value is
its fit's tile draw. ``subsample`` (half-tile subsamples): a pick's value is its
fit's estimate plus its draw centered on the draws' median and scaled by its
``c``. The choices come from one generator per finding, so a result is
reproducible.

**Level.** Two-sided ``1 - alpha / M`` with ``M`` the number of findings (Bonferroni
over the findings, ``alpha = 0.05``).

**kappa\\*.** Null 0. A finding *holds* at a grid kappa if the pooled interval
lies entirely on the side of 0 of the pooled kappa = 0 estimate. kappa\\* is the
first grid kappa where it fails: ``"no finding"`` if it fails
at kappa = 0, ``"above the grid"`` if it never fails. A grid point that some
seed lacks, before any failure, stops the scan (``"gap at kappa = ..."``); it is
never read as survival.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

#: The paper's kappa grid.
KAPPA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
FINDINGS_ALPHA = 0.05


def level(m_s: int, alpha: float = FINDINGS_ALPHA) -> float:
    """The two-sided Bonferroni level ``1 - alpha / m_s``."""
    if m_s < 1:
        raise ValueError("there is at least one finding")
    return 1.0 - alpha / m_s


def pool(per_seed: Sequence[Mapping | None], alpha: float,
         rng: np.random.Generator | None = None) -> dict | None:
    """The seed-pooled point and two-sided ``1 - alpha`` interval of one finding
    at one grid point (seeds without a finite estimate are left out and counted).
    The interval is the two-level bootstrap of the module doc; *rng* fixes its
    choices (default: seed 0)."""
    entries = [e for e in per_seed if e is not None and np.isfinite(e["estimate"])]
    if not entries:
        return None
    methods = {e["method"] for e in entries}
    if len(methods) != 1:
        raise ValueError(f"seeds disagree on the interval form: {methods}")
    method = methods.pop()
    if method not in ("percentile", "subsample"):
        raise ValueError(f"unknown interval form {method!r}")
    n = min(len(e["draws"]) for e in entries)
    est = np.array([float(e["estimate"]) for e in entries])

    def values(e):
        d = np.asarray(e["draws"][:n], dtype=np.float64)
        return (e["estimate"] + e["c"] * (d - np.nanmedian(d))
                if method == "subsample" else d)

    stack = np.stack([values(e) for e in entries])            # (seeds, draws)
    rng = np.random.default_rng(0) if rng is None else rng
    k = len(entries)
    pick = rng.integers(0, k, size=(n, k))                     # fits, with replacement
    draw = rng.integers(0, n, size=(n, k))                     # an independent draw per pick
    pooled = stack[pick, draw].mean(axis=1)
    pooled = pooled[np.isfinite(pooled)]
    if len(pooled) < 2:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(pooled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(est.mean()), "ci": [float(lo), float(hi)],
            "per_seed": [float(x) for x in est],
            "n_seeds": k, "n_draws": int(len(pooled)), "method": method}


def contains_zero(ci: Sequence[float]) -> bool:
    """True when the interval contains 0 or is undefined."""
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return True
    return bool(lo <= 0.0 <= hi)


def holds(entry: dict | None, sign: int) -> bool:
    """The pooled interval lies entirely on the side of 0 given by *sign*."""
    if entry is None or sign == 0:
        return False
    lo, hi = entry["ci"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    return bool(lo > 0) if sign > 0 else bool(hi < 0)


def _fail_reason(entry: dict) -> str:
    return ("interval contains 0" if contains_zero(entry["ci"])
            else "interval on the other side of 0")


def breakdown_point(trajectory: Mapping[float, dict | None],
                    grid: Sequence[float] = KAPPA_GRID) -> dict:
    """kappa\\* of one finding from its pooled grid points (module doc)."""
    first = trajectory.get(grid[0])
    if first is None:
        return {"status": "missing at kappa = 0", "kappa_star": None}
    sign = int(np.sign(first["estimate"]))
    if not holds(first, sign):
        return {"status": "no finding", "kappa_star": None, "sign": sign,
                "reason": _fail_reason(first)}
    for kappa in grid[1:]:
        entry = trajectory.get(kappa)
        if entry is None:
            return {"status": f"gap at kappa = {kappa:g}", "kappa_star": None, "sign": sign}
        if not holds(entry, sign):
            return {"status": "breaks", "kappa_star": float(kappa), "sign": sign,
                    "reason": _fail_reason(entry)}
    return {"status": "above the grid", "kappa_star": None, "sign": sign}


def ranking_pairs(members: Mapping[str, dict], names: Sequence[str]) -> dict:
    """The pairwise differences ``a - b`` of a ranking's members, on the same
    draws of one fit (each difference then gets its own kappa\\*)."""
    out = {}
    for a, b in itertools.combinations(names, 2):
        ea, eb = members[a], members[b]
        if ea["method"] != eb["method"]:
            raise ValueError("a ranking's members share one interval form")
        n = min(len(ea["draws"]), len(eb["draws"]))
        out[f"{a} - {b}"] = {"estimate": ea["estimate"] - eb["estimate"],
                             "draws": np.asarray(ea["draws"][:n]) - np.asarray(eb["draws"][:n]),
                             "method": ea["method"], "c": ea["c"]}
    return out


def breakdown(draws: Mapping[str, Mapping[float, Mapping[int, dict | None]]], *,
              grid: Sequence[float] | None = None, seeds: Sequence[int] | None = None,
              n_findings: int | None = None, alpha: float = FINDINGS_ALPHA,
              order: Sequence[str] | None = None) -> dict:
    """kappa\\* of every finding.

    Parameters
    ----------
    draws
        ``{finding: {kappa: {seed: {"estimate", "draws", "method", "c"}}}}``;
        a missing seed (or None) is a gap at that grid point.
    grid
        The kappa grid in order (default: every kappa in *draws*, sorted).
    seeds
        The seeds every grid point must have (default: every seed in *draws*).
    n_findings
        ``m`` of the Bonferroni level (default: the number of findings).
    alpha
        Level over all findings.
    order
        The findings in the order that fixes their bootstrap choices (default: sorted).

    Returns ``{"m_s", "level", "alpha_per_side", "grid", "seeds", "members"}``;
    each member has its pooled ``trajectory`` (keyed ``f"{kappa:g}"``),
    ``status`` (``"breaks"``, ``"above the grid"``, ``"no finding"``, ``"gap at
    kappa = ..."``), ``kappa_star`` and ``sign``.
    """
    if grid is None:
        grid = sorted({float(k) for per in draws.values() for k in per})
    if seeds is None:
        seeds = sorted({int(s) for per in draws.values() for k in per.values() for s in k})
    m_s = len(draws) if n_findings is None else int(n_findings)
    a = alpha / m_s
    members = {}
    for i, name in enumerate(sorted(draws) if order is None else order):
        rng = np.random.default_rng(8190 + i)    # the bootstrap's choices, fixed per finding
        traj = {}
        for kappa in grid:
            at = draws[name].get(kappa, {})
            entry = pool([at.get(s) for s in seeds], a, rng)
            traj[kappa] = entry if entry is not None and entry["n_seeds"] == len(seeds) else None
        members[name] = {"trajectory": {f"{k:g}": v for k, v in traj.items()},
                         **breakdown_point(traj, grid)}
    return {"m_s": m_s, "level": level(m_s, alpha), "alpha_per_side": a / 2,
            "grid": [float(k) for k in grid], "seeds": [int(s) for s in seeds],
            "members": members}


def read_draws(path: str | Path) -> dict:
    """``{member: {"estimate", "draws", "method", "c"}}`` from one draw file (npz)."""
    with np.load(path, allow_pickle=False) as f:
        names = [str(x) for x in f["members"]]
        return {name: {"estimate": float(f["estimate"][i]),
                       "draws": np.asarray(f["draws"][i], dtype=np.float64),
                       "method": str(f["method"][i]), "c": float(f["c"][i])}
                for i, name in enumerate(names)}


def write_draws(path: str | Path, members: Mapping[str, dict], meta: dict | None = None) -> Path:
    """Write one fit's finding draws (the layout :func:`read_draws` reads).

    Members with fewer draws are padded with NaN, which :func:`pool` drops.
    """
    path = Path(path)
    names = list(members)
    n = max(len(members[k]["draws"]) for k in names)
    draws = np.full((len(names), n), np.nan)
    for i, k in enumerate(names):
        draws[i, :len(members[k]["draws"])] = members[k]["draws"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, members=np.array(names),
             estimate=np.array([members[k]["estimate"] for k in names], float), draws=draws,
             method=np.array([members[k]["method"] for k in names]),
             c=np.array([members[k]["c"] for k in names], float),
             meta=np.array(json.dumps(meta or {}, default=float)))
    return path
