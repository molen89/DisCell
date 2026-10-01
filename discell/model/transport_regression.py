#!/usr/bin/env python3
"""A plain-regression reference for the relocation reads (devlog "RECOMB
story map approved; three additions (motivation, 2026-10-01; author)",
item 2): "why not just regress?".

On exactly the panels of the published transport reads of a run (the mean
read's (type, niche A -> niche B) panels and the distribution read's
pairwise panels: k-means niches on y, fold 0 scored, folds 1-4 model side,
the same cell thresholds and Unassigned mask), each panel's shift is
predicted by a **ridge regression of the type's expression on neighbour
composition** y, fitted on the type's model-side cells (connected, in a
niche) from the raw, contaminated counts. No DISCELL quantity enters: no
latent, no decoder, no kappa; only the bundle, the run's split and niches.

Two forms of the regression (both reported; neither is tuned to the read):

``log``   as specified: per-cell ``log1p(s x / l)`` (s = the section's median
          total count) on [y, 1]; the predicted shift is ``(ybar_B - ybar_A)
          b`` on that scale. It compresses low-rate genes relative to the
          read's log mean-rate scale.
``rate``  on the read's own scale: per-cell ``x / l`` on [y, 1]; the predicted
          shift is ``log(max(a + ybar_B b, MIN_RATE)) - log(max(a + ybar_A b,
          MIN_RATE))``, the log ratio of predicted niche-mean rates (the mean
          of a linear prediction is the prediction at the mean y).

ybar_X is the mean y of the panel type's model-side cells in niche X (the
cells the model's group means come from). The ridge penalty is one per type,
chosen by GCV over a grid (sufficient statistics; with ~K features and
thousands of cells it is close to least squares).

**Mean read.** ``transport.score_shift`` R^2 against the observed held-out
shift on the panel's kept genes, with the panel's stored noise ceiling and
trust flag; fraction of ceiling = mean R^2 / mean ceiling over a tier
(``transport.tier_summary``'s ratio): the published tiers ``extrapolation``
("all panels" in the tables) and ``extrapolation_trusted`` ("trusted"),
plus literally all panels and all trusted panels.

**Read A** (pairwise panels). The published Read A
(``summary_model_own``) scores against the target cells *as DISCELL decodes
them*, a DISCELL quantity, so a model-free predictor cannot be scored on it.
The regression is scored on the count-level form instead: source = the
held-out cells of the type in niche A (at most ``SIZE_CAP``), untransported =
their raw compositions, transported = the same compositions moved by the
regression's per-gene shift (``rate``: times exp(shift); ``log``: on the
log1p scale) and renormalised, target = the raw compositions of the held-out
cells in niche B. MMD^2 on the Hellinger map, bandwidth from the target,
floor = two halves of the target, type-mean = the target's mean composition
drawn at target depths (as the count-matched read); gap closed = (unt -
trans) / (unt - floor), clipped to [-1, 1]. DISCELL's like-for-like number
is its count-matched read on the same raw targets
(``summary_count_matched``); its published own-target read is printed
beside it, labelled.

Writes ``runs/<run>/transport/regression_reference.json``. Opt-in only:
``python -m discell.model.transport --dataset <id> --run <run> --read
regression`` (the default reads are unchanged), and the per-dataset table
``python -m discell.model.transport_regression compare --dataset <id>``
-> ``experiments/transport_regression_reference.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from discell.model import eval_mask as EM
from discell.model import transport as T

log = logging.getLogger("discell.model.transport_regression")

FORMS = ("log", "rate")
RIDGE_GRID = np.logspace(-6, 2, 17)      #: x trace(X'X)/p
SEED_OFFSET = 41                         #: Read A's own generator stream
RUNS = ("finalL_s0", "finalL_s1", "finalL_s2")


# --------------------------------------------------------------------------
# the regression
# --------------------------------------------------------------------------


def ridge_gcv(y: np.ndarray, x, grid: Sequence[float] = RIDGE_GRID) -> dict:
    """Multi-output ridge of *x* (cells x genes, sparse or dense) on [y, 1],
    intercept unpenalised, one penalty for all genes chosen by GCV.

    Works from sufficient statistics only, so a type with 10^5 cells and
    5k sparse genes costs one sparse-dense product. Returns ``coef``
    (features x genes), ``intercept`` (genes,), the penalty and its df."""
    import scipy.sparse as sp

    n, p = y.shape
    ybar = y.mean(axis=0)
    yc = y - ybar
    xtx = yc.T @ yc
    if sp.issparse(x):
        xbar = np.asarray(x.mean(axis=0)).ravel()
        xty = np.asarray((x.T @ y).T) - n * np.outer(ybar, xbar)
        ss = np.asarray(x.multiply(x).sum(axis=0)).ravel() - n * xbar ** 2
    else:
        x = np.asarray(x, dtype=np.float64)
        xbar = x.mean(axis=0)
        xty = yc.T @ (x - xbar)
        ss = ((x - xbar) ** 2).sum(axis=0)
    d, v = np.linalg.eigh(xtx)
    d = np.clip(d, 0.0, None)
    proj = v.T @ xty                                    # p x G
    scale = max(float(np.trace(xtx)) / p, 1e-12)
    best = None
    for g in grid:
        lam = g * scale
        shrink = 1.0 / (d + lam)
        coef_e = shrink[:, None] * proj
        # RSS = ss - 2 b'X'Y + b'X'X b, in the eigenbasis
        rss = float((ss - 2 * (coef_e * proj).sum(0)
                     + (d[:, None] * coef_e ** 2).sum(0)).sum())
        df = float((d * shrink).sum()) + 1.0            # + the intercept
        gcv = rss / (n * max(1.0 - df / n, 1e-12) ** 2)
        if best is None or gcv < best[0]:
            best = (gcv, lam, df, coef_e)
    _, lam, df, coef_e = best
    coef = v @ coef_e
    return {"coef": coef, "intercept": xbar - ybar @ coef, "lambda": lam,
            "df": df, "n": int(n)}


def fit_type(y_rows: np.ndarray, rate_rows, form: str, scale: float) -> dict:
    """The ridge of one type's cells for one *form*."""
    if form == "log":
        x = rate_rows.copy()
        x.data = np.log1p(scale * x.data)
    elif form == "rate":
        x = rate_rows
    else:
        raise ValueError(f"form must be one of {FORMS}, not {form!r}")
    return ridge_gcv(np.asarray(y_rows, dtype=np.float64), x)


def predicted_shift(fit: dict, form: str, ybar_a: np.ndarray,
                    ybar_b: np.ndarray) -> np.ndarray:
    """The per-gene shift A -> B, on the form's scale (module docstring)."""
    if form == "log":
        return (ybar_b - ybar_a) @ fit["coef"]
    pa = fit["intercept"] + ybar_a @ fit["coef"]
    pb = fit["intercept"] + ybar_b @ fit["coef"]
    return (np.log(np.maximum(pb, T.MIN_RATE))
            - np.log(np.maximum(pa, T.MIN_RATE)))


def move(p_source: np.ndarray, shift: np.ndarray, form: str,
         scale: float) -> np.ndarray:
    """Raw source compositions moved by a per-gene *shift*, renormalised."""
    p = np.asarray(p_source, dtype=np.float64)
    if form == "log":
        out = np.expm1(np.log1p(scale * p) + shift[None, :]) / scale
    else:
        out = p * np.exp(shift)[None, :]
    out = np.clip(out, 0.0, None)
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------
# Read A, count level
# --------------------------------------------------------------------------


def read_a_scores(clouds: dict, p_target: np.ndarray, depths: np.ndarray,
                  rng: np.random.Generator, device: str = "cpu",
                  cap: int = T.SIZE_CAP) -> dict:
    """Gap closed of every predictor cloud in *clouds* (``untransported``
    plus one or more transported clouds, all rows the same source cells)
    against the raw target, with the type-mean reference and the floor
    (``transport.distribution_scores``' estimator, no bootstrap)."""
    import torch

    n_s, n_t = len(clouds["untransported"]), len(p_target)
    m = int(min(n_s, n_t, cap))
    if m < 20:
        return {"n": m, "insufficient": True}
    idx_s = rng.choice(n_s, m, replace=False)
    idx_t = rng.choice(n_t, m, replace=False)
    target = np.asarray(p_target, dtype=np.float64)[idx_t]
    dep = np.maximum(np.asarray(depths, dtype=np.int64)[
        rng.integers(0, len(depths), m)], 1)
    mean = target.mean(axis=0)
    mean = mean / mean.sum()
    draw = rng.multinomial(dep, mean).astype(np.float64)
    type_mean = draw / draw.sum(axis=1, keepdims=True).clip(min=1.0)

    def to_t(a):
        return torch.as_tensor(T._hellinger(a), dtype=torch.float32,
                               device=device)

    with torch.no_grad():
        y = to_t(target)
        d_yy = torch.cdist(y, y)
        off = d_yy[~torch.eye(m, dtype=torch.bool, device=d_yy.device)]
        sigma = float(off.median())
        gamma = 1.0 / (2.0 * max(sigma, 1e-8) ** 2)
        kyy = torch.exp(-gamma * d_yy.pow(2))
        half = m // 2
        perm = torch.as_tensor(rng.permutation(m), device=y.device)
        y1, y2 = y[perm[:half]], y[perm[half:2 * half]]
        mmd = {"floor": T._mmd2(T._kernel_parts(y1, y1, gamma),
                                T._kernel_parts(y2, y2, gamma),
                                T._kernel_parts(y1, y2, gamma))}
        named = {k: np.asarray(v)[idx_s] for k, v in clouds.items()}
        named["type_mean"] = type_mean
        for name, cloud in named.items():
            x = to_t(cloud)
            mmd[name] = T._mmd2(T._kernel_parts(x, x, gamma), kyy,
                                T._kernel_parts(x, y, gamma))
    denom = mmd["untransported"] - mmd["floor"]

    def gap(name):
        if abs(denom) <= 1e-12:
            return float("nan")
        return float(np.clip((mmd["untransported"] - mmd[name]) / denom,
                             -1.0, 1.0))
    return {"n": m, "insufficient": False, "bandwidth": sigma, "mmd2": mmd,
            "gap_closed": {k: gap(k) for k in named if k != "untransported"}}


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------


def load_split(dataset: str, run: str):
    """``(config, data, run_dir)``: the run's configuration and its assembled
    data, as ``validate.load_run`` builds them, without the model."""
    import torch

    from discell import paths
    from discell.model.prepare import assemble
    from discell.model.train import config_from_record

    run_dir = paths.dataset(dataset).root / "runs" / run
    payload = torch.load(run_dir / "best.pt", map_location="cpu",
                         weights_only=False)
    config = config_from_record(payload["config"])
    del payload
    data = assemble(dataset, config.variant, config.embeddings,
                    tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                    v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                    seed=config.seed, label_key=config.label_key)
    return config, data, run_dir


def folds(data) -> np.ndarray:
    """``validate.collect_latents``' fold, without the forward pass."""
    from discell.model.validate import N_FOLDS

    fold = np.zeros(data.graph.n_cells, dtype=np.int64)
    for k, tile in enumerate(data.train_tiles + data.val_tiles):
        fold[tile] = k % N_FOLDS
    return fold


def of_ceiling(panels: list[dict], key: str) -> dict:
    """``tier_summary``'s fraction: mean R^2 / mean ceiling (NaN under 0.05)."""
    if not panels:
        return {"n_panels": 0}
    r2 = float(np.mean([p[key]["r2"] for p in panels]))
    ceiling = float(np.mean([p["noise_ceiling"] for p in panels]))
    return {"n_panels": len(panels), "mean_r2": r2, "noise_ceiling": ceiling,
            "of_ceiling": r2 / ceiling if ceiling >= 0.05 else float("nan")}


def tiers(panels: list[dict]) -> dict:
    return {"extrapolation": [p for p in panels if p["overlap_flag"]],
            "extrapolation_trusted": [p for p in panels
                                      if p["overlap_flag"] and p["trusted"]],
            "all": list(panels),
            "all_trusted": [p for p in panels if p["trusted"]]}


def regression_reference(args: argparse.Namespace) -> dict:
    from discell.model.validate import niche_labels

    config, data, run_dir = load_split(args.dataset, args.run)
    out_dir = run_dir / "transport"
    stored = json.loads((out_dir / "transport.json").read_text())
    stored_dist = json.loads((out_dir / "transport_distribution.json")
                             .read_text())
    fold = folds(data)
    scored, model_side = T.split_cells(data, fold, "fold0")
    connected = data.graph.degrees > 0
    labels = niche_labels(data, args.niches, config.seed)
    n_types = len(data.p_t)
    names = [str(n) for n in data.type_names]
    y = np.asarray(data.graph.y, dtype=np.float64)
    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    scale = float(data.median_counts)

    # one ridge per (type, form), on the type's model-side cells in a niche
    fits: dict = {}
    for g in range(n_types):
        rows = np.flatnonzero(connected & model_side & (labels >= 0)
                              & (data.t == g))
        if len(rows) < T.MIN_CELLS:
            continue
        for form in FORMS:
            fits[(g, form)] = fit_type(y[rows], x_rate[rows], form, scale)

    def side_rows(niche, g):
        base = connected & (data.t == g) & (labels == niche)
        return (np.flatnonzero(base & model_side),
                np.flatnonzero(base & scored))

    # -- the mean read: the stored panels, replayed --------------------------
    by_key = {(tuple(p["pair"]), p["type"]): p for p in stored["panels"]}
    panels, mismatches = [], []
    for niche_a, niche_b in T.pick_pairs(labels, data.graph.y, connected):
        for g in range(n_types):
            tr_a, te_a = side_rows(niche_a, g)
            tr_b, te_b = side_rows(niche_b, g)
            if (min(len(tr_a), len(tr_b)) < T.MIN_CELLS
                    or min(len(te_a), len(te_b)) < T.MIN_CELLS // 5):
                continue
            if EM.is_excluded(names[g]):
                continue
            ref = by_key.get(((int(niche_a), int(niche_b)), names[g]))
            obs_a = np.asarray(x_rate[te_a].mean(axis=0)).ravel()
            obs_b = np.asarray(x_rate[te_b].mean(axis=0)).ravel()
            keep = (obs_a > T.MIN_RATE) & (obs_b > T.MIN_RATE)
            observed = np.log(obs_b[keep] + T.EPS) - np.log(obs_a[keep] + T.EPS)
            same = (ref is not None
                    and ref["n_train"] == [len(tr_a), len(tr_b)]
                    and ref["n_test"] == [len(te_a), len(te_b)]
                    and ref["n_genes"] == int(keep.sum()))
            if not same:
                mismatches.append({"pair": [int(niche_a), int(niche_b)],
                                   "type": names[g]})
                continue
            ybar_a, ybar_b = y[tr_a].mean(axis=0), y[tr_b].mean(axis=0)
            panel = {"pair": [int(niche_a), int(niche_b)], "type": names[g],
                     "n_genes": int(keep.sum()),
                     "overlap_flag": ref["overlap_flag"],
                     "trusted": ref["trusted"],
                     "noise_ceiling": ref["noise_ceiling"],
                     "discell": ref["counterfactual"]}
            for form in FORMS:
                shift = predicted_shift(fits[(g, form)], form, ybar_a, ybar_b)
                panel[f"regression_{form}"] = T.score_shift(shift[keep],
                                                           observed)
            panels.append(panel)
    missing = len(stored["panels"]) - len(panels)
    if mismatches or missing:
        raise RuntimeError(f"panels do not reproduce the published read: "
                           f"{len(mismatches)} mismatched, {missing} missing "
                           f"({mismatches[:3]})")

    mean_read = {}
    for name, tier in tiers(panels).items():
        mean_read[name] = {k: of_ceiling(tier, k) for k in
                           ("discell", *(f"regression_{f}" for f in FORMS))}

    # -- Read A, count level, the published pairwise panels --------------------
    rng = np.random.default_rng([config.seed, SEED_OFFSET])
    read_a = []
    for entry in stored_dist["pairwise"]:
        niche_a, niche_b = entry["pair"]
        g = names.index(entry["type"])
        tr_a, te_a = side_rows(niche_a, g)
        tr_b, te_b = side_rows(niche_b, g)
        if [len(te_a), len(te_b)] != [entry["n_source"], entry["n_target"]]:
            raise RuntimeError(f"Read A panel {entry['pair']} {entry['type']} "
                               "does not reproduce the published cell counts")
        src = te_a[rng.permutation(len(te_a))[:T.SIZE_CAP]]
        p_src = np.asarray(x_rate[src].todense())
        ybar_a, ybar_b = y[tr_a].mean(axis=0), y[tr_b].mean(axis=0)
        clouds = {"untransported": p_src}
        for form in FORMS:
            shift = predicted_shift(fits[(g, form)], form, ybar_a, ybar_b)
            clouds[f"regression_{form}"] = move(p_src, shift, form, scale)
        tgt = te_b[rng.permutation(len(te_b))[:T.SIZE_CAP]]
        scores = read_a_scores(clouds, np.asarray(x_rate[tgt].todense()),
                               np.asarray(data.totals)[tgt], rng, args.device)
        read_a.append({"pair": [int(niche_a), int(niche_b)],
                       "type": entry["type"], "scores": scores,
                       "discell_count_matched": entry["scores_count_matched"]
                       .get("gap_closed"),
                       "discell_own": entry["scores_model_own"]
                       .get("gap_closed")})
        log.info("Read A %s %d->%d: %s", entry["type"][:18], niche_a, niche_b,
                 {k: round(v, 3) for k, v in
                  scores.get("gap_closed", {}).items()})

    ok = [p for p in read_a if not p["scores"]["insufficient"]]

    def med(values):
        v = np.asarray(values, dtype=float)
        return float(np.nanmedian(v)) if len(v) else float("nan")
    tm = med([p["scores"]["gap_closed"]["type_mean"] for p in ok])
    read_a_summary = {"n_panels": len(ok), "median_gap_closed_type_mean": tm}
    for form in FORMS:
        gc = med([p["scores"]["gap_closed"][f"regression_{form}"] for p in ok])
        read_a_summary[f"regression_{form}"] = {
            "median_gap_closed": gc, "minus_type_mean": gc - tm}
    cm = (stored_dist.get("summary_count_matched") or {}).get("pairwise") or {}
    own = (stored_dist.get("summary_model_own") or {}).get("pairwise") or {}
    read_a_summary["discell_count_matched"] = {
        "median_gap_closed": cm.get("median_gap_closed"),
        "median_gap_closed_type_mean": cm.get("median_gap_closed_type_mean"),
        "minus_type_mean": (cm["median_gap_closed"]
                            - cm["median_gap_closed_type_mean"])
        if "median_gap_closed_type_mean" in cm else None}
    read_a_summary["discell_own_published"] = {
        "median_gap_closed": own.get("median_gap_closed"),
        "median_gap_closed_type_mean": own.get("median_gap_closed_type_mean"),
        "minus_type_mean": (own["median_gap_closed"]
                            - own["median_gap_closed_type_mean"])
        if "median_gap_closed_type_mean" in own else None,
        "note": "target = the cells as DISCELL decodes them; not a target a "
                "model-free predictor can be scored on"}

    result = {"run": args.run, "dataset": args.dataset,
              "spec": "devlog 'RECOMB story map approved; three additions', "
                      "item 2",
              "niches": args.niches, "forms": list(FORMS),
              "ridge": {f"{names[g]}|{form}": {k: fit[k] for k in
                                               ("lambda", "df", "n")}
                        for (g, form), fit in fits.items()},
              "eval_mask": stored.get("eval_mask"),
              "mean_read": mean_read, "read_a": read_a_summary,
              "panels": panels, "read_a_panels": read_a}
    (out_dir / "regression_reference.json").write_text(
        json.dumps(result, indent=1, default=float))
    log.info("wrote %s", out_dir / "regression_reference.json")
    for name in ("extrapolation", "extrapolation_trusted"):
        log.info("fraction of ceiling [%s]: %s", name,
                 {k: round(v.get("of_ceiling", float("nan")), 3)
                  for k, v in mean_read[name].items()})
    return result


# --------------------------------------------------------------------------
# the per-dataset comparison
# --------------------------------------------------------------------------


def _f(v, d=3) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def compare(dataset: str, runs: Sequence[str] = RUNS) -> dict:
    from discell import paths

    root = paths.dataset(dataset).root
    rows = {}
    for run in runs:
        p = root / "runs" / run / "transport" / "regression_reference.json"
        if p.exists():
            rows[run] = json.loads(p.read_text())
    keys = ("discell", *(f"regression_{f}" for f in FORMS))
    table = {"dataset": dataset, "runs": {}}
    for run, r in rows.items():
        table["runs"][run] = {
            "mean_read": {t: {k: r["mean_read"][t][k].get("of_ceiling")
                              for k in keys}
                          | {"n_panels": r["mean_read"][t]["discell"]
                             .get("n_panels", 0)}
                          for t in r["mean_read"]},
            "read_a": r["read_a"]}
    lines = [f"# Plain-regression reference for relocation — {dataset}", "",
             "Same panels as DISCELL's published transport reads (finalL, "
             "fold 0 scored). Regression: ridge of the type's expression on "
             "neighbour composition, model-side cells, raw counts; `log` = "
             "log1p at the median depth (as specified), `rate` = depth-"
             "normalised rate (the read's own scale). Fraction of ceiling = "
             "mean R² / mean noise ceiling over the tier. Published tiers: "
             "'all panels' = extrapolation, 'trusted' = extrapolation_trusted.",
             "", "## Mean read: fraction of ceiling", "",
             "| run | tier | panels | DISCELL | regression (log) | "
             "regression (rate) |", "|---|---|---|---|---|---|"]
    for run, r in table["runs"].items():
        for tier in ("extrapolation", "extrapolation_trusted", "all",
                     "all_trusted"):
            v = r["mean_read"].get(tier) or {}
            lines.append(f"| {run} | {tier} | {v.get('n_panels', 0)} | "
                         f"{_f(v.get('discell'))} | "
                         f"{_f(v.get('regression_log'))} | "
                         f"{_f(v.get('regression_rate'))} |")
    lines += ["", "## Read A (pairwise panels): median gap closed, and minus "
              "the type-mean predictor's", "",
              "Count level (raw target cells, no DISCELL quantity): the "
              "regression moves the source cells' raw compositions; DISCELL's "
              "like-for-like number is its count-matched read. DISCELL's "
              "published Read A (own target = the target cells as DISCELL "
              "decodes them) is printed last; a model-free predictor cannot be "
              "scored on that target.", "",
              "| run | panels | regression (log) | regression (rate) | "
              "type-mean (regression read) | DISCELL count-matched | "
              "type-mean (count-matched) | DISCELL published (own) |",
              "|---|---|---|---|---|---|---|---|"]
    for run, r in table["runs"].items():
        a = r["read_a"]
        cm, own = a["discell_count_matched"], a["discell_own_published"]

        def gm(e):
            return (f"{_f(e.get('median_gap_closed'))} "
                    f"({_f(e.get('minus_type_mean'), 3)})")
        lines.append(
            f"| {run} | {a['n_panels']} | {gm(a['regression_log'])} | "
            f"{gm(a['regression_rate'])} | "
            f"{_f(a['median_gap_closed_type_mean'])} | {gm(cm)} | "
            f"{_f(cm.get('median_gap_closed_type_mean'))} | {gm(own)} |")
    lines.append("")
    lines.append("Cells: median gap closed (median gap closed minus the "
                 "type-mean predictor's median, the breakdown member's form). "
                 "Moving a raw composition on the log1p scale (`log`) lifts "
                 "zero counts off zero by expm1(shift)/s, which a "
                 "multiplicative move (`rate`) does not.")
    out = root / "experiments" / "transport_regression_reference"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(table, indent=1,
                                                   default=float))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    log.info("wrote %s", out.with_suffix(".md"))
    return table


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--dataset", required=True)
    c.add_argument("--runs", nargs="*", default=list(RUNS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    compare(args.dataset, args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
