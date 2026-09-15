#!/usr/bin/env python3
"""Doc-09 section 8: the LR co-occurrence map (SIMVI fig-6h style) with its
composition control.

Panel A: rows = top +-10 B-loadings per effective-rank w program (issues
V10), row value per cell <w_i, B_g>; columns = gate-zero ligand-receptor
pairs ranked by Var(exposure) x receiver prevalence, column value per cell
LR_i = R_i x E_i (depth-normalised log1p receptor x one-hop exposure); entry
= Spearman within receiver type, Fisher-z pooled, validation tiles; null =
within-type permutation of the LR score, BH per map. Panel B: the same after
rank-transforming and ridge-residualising both sides on neighbour
composition y within type. Descriptive only -- no communication claim.

Usage::

    python -m discell.model.lr_map --dataset <id> --run ablation_gat_type_only_s1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from typing import Sequence

import numpy as np
from scipy.stats import rankdata

from discell.model.validate import collect_latents, load_run

log = logging.getLogger("discell.model.lr_map")

N_PAIRS = 30
TOP_LOADINGS = 10
MIN_RECEPTOR_POSITIVE = 500      #: receptor-positive validation cells per type
MIN_ELIGIBLE_TYPES = 2           #: a column needs two receiver types to pool
N_PERMS = 200
VAR_FRACTION = 0.01              #: effective-rank threshold on cov(mu_w) (V10)
Q_THRESHOLD = 0.05


def program_basis(mu_w: np.ndarray, b_matrix: np.ndarray,
                  expressed: np.ndarray):
    """The effective-rank programs of w (issues V10): top-r principal
    directions of cov(mu_w) carrying >= VAR_FRACTION of the variance,
    varimax-rotated within that subspace on the expressed genes.

    Returns per-cell program coordinates ``u`` (N, r) -- standardised PC
    scores under the rotation, so ``u @ programs.T`` is the realised context
    shift on the effective subspace -- the gene loadings ``programs`` (G, r)
    and ``{"rank", "variance_fraction"}``.
    """
    from discell.model.atlas import varimax

    centred = mu_w - mu_w.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(centred.T))
    order = np.argsort(-evals)
    evals, evecs = evals[order].clip(min=0), evecs[:, order]
    frac = evals / max(evals.sum(), 1e-12)
    r = max(1, int((frac >= VAR_FRACTION).sum()))
    scale = np.sqrt(evals[:r]) + 1e-12
    loadings = b_matrix @ (evecs[:, :r] * scale)                 # effect scale
    rotation = varimax(loadings[expressed]) if r > 1 else np.eye(1)
    programs = loadings @ rotation
    u = (centred @ evecs[:, :r] / scale) @ rotation
    return u, programs, {"rank": r, "variance_fraction": frac.round(4).tolist()}


def effective_programs(mu_w: np.ndarray, b_matrix: np.ndarray,
                       expressed: np.ndarray, top: int = TOP_LOADINGS):
    """Top +-*top* loadings per effective-rank program, deduplicated:
    ``[(gene, program, sign), ...]`` and the rank info (used for labels)."""
    _, programs, info = program_basis(mu_w, b_matrix, expressed)
    r = info["rank"]
    rows, seen = [], set()
    for k in range(r):
        col = np.where(expressed, programs[:, k], 0.0)
        for g, sign in [(g, +1) for g in np.argsort(-col)[:top]] + \
                       [(g, -1) for g in np.argsort(col)[:top]]:
            if int(g) not in seen:
                seen.add(int(g))
                rows.append((int(g), k, sign))
    return rows, info


def receptor_clean_rates(trainer, gene_idx: np.ndarray) -> np.ndarray:
    """The model's clean rates rho_g (posterior means) for a few genes, every
    cell, in dataset order -- the receptor side of the LR score (section
    8.2b: raw 0-2 counts per cell starve the Spearman; mildly circular)."""
    import torch

    chunks, nodes = [], []
    with torch.no_grad():
        for batch in trainer.train_batches + trainer.val_batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch),
                                kappa=trainer.config.kappa, sample=False)
            n = batch["n_seeds"]
            chunks.append(fwd.log_rho[:n][:, torch.as_tensor(gene_idx, device=fwd.log_rho.device)]
                          .exp().cpu().numpy())
            nodes.append(batch["nodes"][:n])
    nodes = np.concatenate(nodes)
    out = np.zeros((len(nodes), len(gene_idx)), dtype=np.float32)
    out[nodes] = np.concatenate(chunks)
    return out


def _residualise(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Every column of *values* minus its ridge fit on [y, 1] (one type)."""
    design = np.hstack([y - y.mean(axis=0), np.ones((len(y), 1))])
    gram = design.T @ design + 1e-3 * np.eye(design.shape[1])
    return values - design @ np.linalg.solve(gram, design.T @ values)


def _standardise(m: np.ndarray) -> np.ndarray:
    m = m - m.mean(axis=0)
    return m / (m.std(axis=0) + 1e-12)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    flat = p.ravel()
    order = np.argsort(flat)
    ranked = flat[order] * len(flat) / (np.arange(len(flat)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty_like(flat)
    out[order] = q
    return out.reshape(p.shape)


def _shift_index(positions: np.ndarray, rng, min_um: float = 500.0) -> np.ndarray:
    """A Moran-preserving null for one type: every cell takes the value of
    the same-type cell nearest to its position displaced by one random
    vector (|delta| >= *min_um*), so the shuffled field keeps its spatial
    smoothness but loses its alignment with the row field. A plain
    permutation destroys the smoothness and is anti-conservative for two
    smooth fields (the A4/V8 lesson)."""
    from scipy.spatial import cKDTree

    angle = rng.uniform(0, 2 * np.pi)
    radius = rng.uniform(min_um, 2 * min_um)
    delta = radius * np.array([np.cos(angle), np.sin(angle)])
    return cKDTree(positions).query(positions + delta)[1]


def cooccurrence(rows: np.ndarray, cols: np.ndarray, t: np.ndarray,
                 y: np.ndarray, eligible: np.ndarray, n_perms: int = N_PERMS,
                 seed: int = 0, partial: bool = False,
                 positions: np.ndarray | None = None) -> dict:
    """Within-type Spearman between every row and every column variable,
    Fisher-z pooled over the types *eligible* for each column
    ``(n_types, n_cols)``, with a within-type permutation null on the
    (processed) column scores -- a spatial shift (``_shift_index``) when
    *positions* are given, a plain permutation otherwise. *partial*:
    rank-transform, then residualise both sides on y within type (panel B).
    Types are ``np.unique(t)``.

    Returns ``{"rho", "p", "z_null_sd", "n_cells"}`` -- rho/p ``(n_rows, n_cols)``.
    """
    rng = np.random.default_rng(seed)
    types = np.unique(t)
    z_obs = np.zeros((rows.shape[1], cols.shape[1]))
    z_null = np.zeros((n_perms,) + z_obs.shape)
    weight = np.zeros(cols.shape[1])
    n_cells = np.zeros(cols.shape[1], dtype=int)
    for i, g in enumerate(types):
        members = np.flatnonzero(t == g)
        n = len(members)
        if n < 10 or not eligible[i].any():
            continue
        r = np.apply_along_axis(rankdata, 0, rows[members])
        c = np.apply_along_axis(rankdata, 0, cols[members])
        if partial:
            r, c = _residualise(r, y[members]), _residualise(c, y[members])
        r, c = _standardise(r), _standardise(c)
        w = (n - 3.0) * eligible[i]                          # per column
        z_obs += w[None, :] * np.arctanh(np.clip(r.T @ c / n, -0.999999, 0.999999))
        for k in range(n_perms):
            perm = (rng.permutation(n) if positions is None
                    else _shift_index(positions[members], rng))
            z_null[k] += w[None, :] * np.arctanh(
                np.clip(r.T @ c[perm] / n, -0.999999, 0.999999))
        weight += w
        n_cells += (n * eligible[i]).astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = z_obs / weight[None, :]
        z_null = z_null / weight[None, None, :]
    # the permutations calibrate the null's scale; the tail is read from a
    # Gaussian at that scale so BH over a 1000-cell map is not capped by
    # 1/(n_perms + 1) (a single true survivor in panel B could never clear
    # BH on empirical p-values alone). The empirical p is kept for reference.
    from scipy.stats import norm

    sd = z_null.std(axis=0)
    p_empirical = (1.0 + (np.abs(z_null) >= np.abs(z)[None]).sum(axis=0)) / (n_perms + 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 2.0 * norm.sf(np.abs(z) / sd)
    p = np.where(weight[None, :] > 0, p, np.nan)
    return {"rho": np.tanh(z), "p": p, "p_empirical": p_empirical,
            "z_null_sd": sd, "n_cells": n_cells}


def receptor_score(x_rate, gene_of: dict, subunits: list[str],
                   median_counts: float) -> tuple[np.ndarray, np.ndarray]:
    """Depth-normalised log1p receptor expression (mean over subunits) and
    the receptor-positive mask (every subunit detected)."""
    cols = [np.asarray(x_rate[:, gene_of[s]].todense()).ravel() for s in subunits]
    score = np.mean([np.log1p(c * median_counts) for c in cols], axis=0)
    positive = np.all([c > 0 for c in cols], axis=0)
    return score, positive


def build_map(args: argparse.Namespace) -> dict:
    from discell.model.communication import (exposure_onehop, load_databases,
                                             residualise, sender_receiver_types)

    started = time.time()
    config, data, trainer, run_dir, b_matrix = load_run(args.dataset, args.run,
                                                        args.device)
    latents = collect_latents(trainer, data)
    gene_names = np.asarray([str(g) for g in data.gene_names])
    gene_of = {g: k for k, g in enumerate(gene_names)}
    expressed = np.asarray((data.x > 0).mean(axis=0)).ravel() >= 0.01
    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsc()
    names = [str(n) for n in data.type_names]

    # rows: the effective-rank program coordinates (section 8.2b: per-gene
    # rows at rank 2 are duplicates within program-sign blocks), labelled by
    # each program's top +/- loadings
    row_values, programs, rank_info = program_basis(latents["mu_w"], b_matrix,
                                                    expressed)
    gene_rows, _ = effective_programs(latents["mu_w"], b_matrix, expressed, top=4)
    rows = []
    for k in range(rank_info["rank"]):
        up = [str(gene_names[g]) for g, kk, sgn in gene_rows if kk == k and sgn > 0]
        down = [str(gene_names[g]) for g, kk, sgn in gene_rows if kk == k and sgn < 0]
        rows.append({"program": k, "up": up, "down": down,
                     "label": f"P{k}: {'/'.join(up[:3])} ↔ {'/'.join(down[:3])}"})

    # columns: gate-zero pairs, LR_i = R_i x E_i. Ranked by the variance of
    # the composition-RESIDUALISED exposure (doc-09 section 2: raw Var(E) is
    # sender proximity) x receiver prevalence; one column per ligand (pairs
    # sharing a ligand share E and differ only in the receptor factor).
    pairs, _ = load_databases(set(gene_names))
    sender_receiver_types(data, x_rate, pairs)
    pairs = [p for p in pairs if p["sender_types"] and p["receiver_types"]]
    counts = np.bincount(data.t, minlength=len(names))
    connected = data.graph.degrees > 0
    rng = np.random.default_rng(config.seed)
    receptor_genes = sorted({gene_of[sub] for pair in pairs
                             for sub in pair["receptor_subunits"]})
    rho = receptor_clean_rates(trainer, np.array(receptor_genes))    # (N, n_rec)
    rho_col = {g: rho[:, j] for j, g in enumerate(receptor_genes)}
    for pair in pairs:
        ligand = np.asarray(x_rate[:, gene_of[pair["ligand"]]].todense()).ravel()
        pair["exposure"] = exposure_onehop(data, ligand)
        _, pair["receptor_positive"] = receptor_score(
            x_rate, gene_of, pair["receptor_subunits"], data.median_counts)
        pair["receptor"] = np.mean([np.log1p(rho_col[gene_of[sub]] * data.median_counts)
                                    for sub in pair["receptor_subunits"]], axis=0)
        pair["lr"] = pair["receptor"] * pair["exposure"]
        spread, weight = 0.0, 0
        for g in pair["receiver_types"]:
            members = np.flatnonzero((data.t == g) & connected)
            if len(members) > 20_000:
                members = np.sort(rng.choice(members, 20_000, replace=False))
            e_tilde, _ = residualise(pair["exposure"], data, members)
            spread += e_tilde.var() * len(members); weight += len(members)
        prevalence = counts[pair["receiver_types"]].sum() / data.graph.n_cells
        pair["rank_score"] = float(spread / max(weight, 1) * prevalence)
    best_per_ligand: dict = {}
    for pair in sorted(pairs, key=lambda p: -p["rank_score"]):
        best_per_ligand.setdefault(pair["ligand"], pair)
    pairs = list(best_per_ligand.values())
    # validation tiles only, connected cells; types eligible per column when
    # they hold >= MIN_RECEPTOR_POSITIVE receptor-positive validation cells;
    # pairs with no eligible type cannot be scored and leave the ranking
    val = np.zeros(data.graph.n_cells, dtype=bool)
    val[np.concatenate(data.val_tiles)] = True
    val &= data.graph.degrees > 0
    t_val = data.t[val]
    types = np.unique(t_val)

    def eligible_types(pair):
        return [(pair["receptor_positive"] & val & (data.t == g)).sum()
                >= MIN_RECEPTOR_POSITIVE for g in types]
    pairs = [p for p in pairs if sum(eligible_types(p)) >= MIN_ELIGIBLE_TYPES]
    pairs = sorted(pairs, key=lambda p: -p["rank_score"])[:args.pairs]
    pairs.sort(key=lambda p: (p["pathway"], -p["rank_score"]))   # grouped
    eligible = np.stack([eligible_types(p) for p in pairs], axis=1)
    cols = np.stack([pair["lr"] for pair in pairs], axis=1)[val]
    y = data.graph.y[val].astype(np.float64)
    # two nulls per panel: the doc's within-type permutation (the map's
    # significance; anti-conservative for two smooth fields) and the
    # Moran-preserving spatial shift (the stringent control: is the
    # co-occurrence more than two smooth fields at the domain scale?)
    # the ladder (section 8.2b): A' uncontrolled over all validation cells
    # (the SIMVI-comparable view), A within type, B composition-partialled
    one_type = np.zeros(len(t_val), dtype=int)
    rungs = (("A_prime", one_type, np.ones((1, len(pairs)), dtype=bool), False),
             ("A", t_val, eligible, False),
             ("B", t_val, eligible, True))
    panels = {}
    for key, strata, elig, partial in rungs:
        out = cooccurrence(row_values[val], cols, strata, y, elig,
                           n_perms=args.n_perms, seed=config.seed, partial=partial)
        out["q"] = benjamini_hochberg(np.nan_to_num(out["p"], nan=1.0))
        shift = cooccurrence(row_values[val], cols, strata, y, elig,
                             n_perms=args.n_perms, seed=config.seed, partial=partial,
                             positions=data.positions[val])
        out["q_shift"] = benjamini_hochberg(np.nan_to_num(shift["p"], nan=1.0))
        panels[key] = out
    survivors_b = [(rows[i]["label"], pairs[j]["name"],
                    float(panels["B"]["rho"][i, j]), float(panels["B"]["q"][i, j]))
                   for i, j in zip(*np.where(panels["B"]["q"] <= Q_THRESHOLD))]
    result = {
        "run": args.run, "kappa": config.kappa, "n_perms": args.n_perms,
        "null": {"q": "within-type permutation of the LR score (doc 09 section 8)",
                 "q_shift": "within-type Moran-preserving spatial shift, 500-1000 um"},
        "seconds": round(time.time() - started, 1),
        "effective_rank": rank_info,
        "receptor_side": "model clean rates rho_g (softmax(a(z) + Bw)), log1p x "
                         "median depth -- mildly circular, descriptive map",
        "rows": rows,
        "columns": [{"name": p["name"], "pathway": p["pathway"],
                     "ligand": p["ligand"], "receptors": p["receptor_subunits"],
                     "secreted": bool(p["secreted"]),
                     "n_eligible_types": int(eligible[:, j].sum()),
                     "n_cells": int(panels["A"]["n_cells"][j])}
                    for j, p in enumerate(pairs)],
        "panel_A_prime": {"rho": panels["A_prime"]["rho"].tolist(),
                          "q": panels["A_prime"]["q"].tolist(),
                          "n_significant": int((panels["A_prime"]["q"] <= Q_THRESHOLD).sum()),
                          "q_shift": panels["A_prime"]["q_shift"].tolist(),
                          "n_significant_shift": int((panels["A_prime"]["q_shift"] <= Q_THRESHOLD).sum()),
                          "max_abs_rho": float(np.nanmax(np.abs(panels["A_prime"]["rho"])))},
        "panel_A": {"rho": panels["A"]["rho"].tolist(), "q": panels["A"]["q"].tolist(),
                    "n_significant": int((panels["A"]["q"] <= Q_THRESHOLD).sum()),
                    "q_shift": panels["A"]["q_shift"].tolist(),
                    "n_significant_shift": int((panels["A"]["q_shift"] <= Q_THRESHOLD).sum()),
                    "max_abs_rho": float(np.nanmax(np.abs(panels["A"]["rho"])))},
        "panel_B": {"rho": panels["B"]["rho"].tolist(), "q": panels["B"]["q"].tolist(),
                    "n_significant": int((panels["B"]["q"] <= Q_THRESHOLD).sum()),
                    "q_shift": panels["B"]["q_shift"].tolist(),
                    "n_significant_shift": int((panels["B"]["q_shift"] <= Q_THRESHOLD).sum()),
                    "max_abs_rho": float(np.nanmax(np.abs(panels["B"]["rho"]))),
                    "survivors": survivors_b},
        "caption": ("Top: uncontrolled, w's programs co-occur with curated ligand-"
                    "receptor axes - the context field carries recognisable biology. "
                    "Middle: within type, most of it is type composition. Bottom: after "
                    "composition control, the density collapses - the map measures "
                    "where pairs and programs co-locate, not signalling. Methods that "
                    "show only the top panel are reporting composition; none of the "
                    "three rungs carries a spatial-autocorrelation-aware null "
                    "(boxed entries would survive it)."),
    }
    out_dir = run_dir / "communication"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "lr_map.json").write_text(json.dumps(result, indent=1))
    map_figure(result, out_dir / "lr_map.png")
    log.info("A' %d / A %d / B %d significant entries of %d (permutation null); "
             "%d / %d / %d under the spatial-shift null; %.0f s",
             result["panel_A_prime"]["n_significant"], result["panel_A"]["n_significant"],
             result["panel_B"]["n_significant"], panels["A"]["q"].size,
             result["panel_A_prime"]["n_significant_shift"],
             result["panel_A"]["n_significant_shift"],
             result["panel_B"]["n_significant_shift"], result["seconds"])
    return result


def map_figure(result: dict, path, q_threshold: float = Q_THRESHOLD) -> None:
    """The ladder: A' uncontrolled, A within type, B composition-partialled.
    Rows = effective-rank programs, columns grouped by pathway; n.s. cells
    (permutation null) grey; cells surviving the spatial-shift null boxed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows, cols = result["rows"], result["columns"]
    row_labels = [r["label"][:60] for r in rows]
    col_labels = [f"{c['ligand']}→{'+'.join(c['receptors'])} [{c['pathway']}]"
                  for c in cols]
    rungs = (("A_prime", "A′: uncontrolled (pooled over all cells)"),
             ("A", "A: within receiver type"),
             ("B", "B: within type, composition-partialled"))
    limit = float(np.nanpercentile(np.abs(np.array(result["panel_A_prime"]["rho"])), 98)) or 0.1
    fig, axes = plt.subplots(len(rungs), 1,
                             figsize=(2.5 + 0.4 * len(cols),
                                      1.2 + len(rungs) * (0.9 + 0.45 * len(rows))),
                             sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    for ax, (key, title) in zip(np.atleast_1d(axes), rungs):
        panel = result[f"panel_{key}"]
        rho, q = np.array(panel["rho"]), np.array(panel["q"])
        shown = np.ma.masked_where(~(q <= q_threshold), rho)
        image = ax.imshow(shown, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_title(f"{title} — {panel['n_significant']} of {q.size} entries at q ≤ "
                     f"{q_threshold} (permutation null; grey = n.s.); "
                     f"{panel['n_significant_shift']} survive the spatial-shift null "
                     f"(boxed); max |ρ| = {panel['max_abs_rho']:.2f}", fontsize=7)
        for i, j in zip(*np.where(np.array(panel["q_shift"]) <= q_threshold)):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.0))
    axes[-1].set_xticks(range(len(cols)))
    axes[-1].set_xticklabels(col_labels, rotation=90, fontsize=5)
    fig.colorbar(image, ax=list(axes), shrink=0.6, pad=0.02, label="pooled Spearman ρ")
    fig.suptitle(f"{result['run']}: w-program × LR co-occurrence ladder (validation "
                 f"tiles; effective rank {result['effective_rank']['rank']}; receptor "
                 "side = model clean rates)\n"
                 + "\n".join(textwrap.wrap(result["caption"], 170)), fontsize=7)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument("--n-perms", type=int, default=N_PERMS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    build_map(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
