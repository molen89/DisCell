#!/usr/bin/env python3
"""A4 -- decontaminated cycle call, redesigned (doc 11 v1, A4 redesign).

Contrast population: post-mitotic types stratified by neighbour-cycle
exposure (the leak victims sit at proliferative-niche edges). Real-data leg:
the gene-split fingerprint -- contamination transfers transcripts, homophily
transfers state; disjoint halves A/B of the cycle set give
delta = corr(own_A, nbr_A) - corr(own_A, nbr_B): ~0 under homophily, > 0
under leakage, and one-hop only (ring-1 vs ring-2). DAPI kept group-level.
Primary adjudicator: the planted world (cycle-like program + leak on the
gate scaffold; phase-label recovery z vs raw).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from discell import paths
from discell.model.cell_cycle import G2M_GENES, S_GENES
from discell.model.validate import collect_latents, load_run

log = logging.getLogger("discell.applications.a4_cycle")
N_SPLITS = 40
N_PERMS = 200
MIN_TYPE_CELLS = 2000


def log_norm(x, totals, median):
    """log1p(x / l * median) as CSR."""
    xn = sp.csr_matrix(x.multiply(1.0 / np.clip(totals, 1, None)[:, None]) * median)
    xn.data = np.log1p(xn.data)
    return xn


def set_score(xn, idx):
    """Mean log-normalised expression over a gene set, per cell."""
    return np.asarray(xn[:, idx].mean(axis=1)).ravel()


def ring_operators(data):
    """Row-normalised 1-hop and exact 2-hop neighbour averaging operators."""
    n = data.graph.n_cells
    i, j = data.graph.edge_i, data.graph.edge_j
    adj = sp.coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    two = (adj @ adj).tocsr()
    two.setdiag(0)
    two = two - two.multiply(adj)          # exact ring 2: drop 1-hop pairs
    two.data[:] = 1.0
    two.eliminate_zeros()

    def rownorm(m):
        deg = np.asarray(m.sum(axis=1)).ravel()
        inv = np.where(deg > 0, 1.0 / np.clip(deg, 1e-12, None), 0.0)
        return sp.diags(inv) @ m

    return rownorm(adj), rownorm(two)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def gene_split_delta(xn, cycle_idx, r1, r2, t, members, rng, n_splits=N_SPLITS):
    """delta_k = corr(own_A, nbr_k(A)) - corr(own_A, nbr_k(B)) per ring k,
    within type over *members*, averaged over random disjoint halves."""
    per_split = {"ring1": [], "ring2": []}
    for _ in range(n_splits):
        perm = rng.permutation(cycle_idx)
        half = len(perm) // 2
        own_a, own_b = set_score(xn, perm[:half]), set_score(xn, perm[half:])
        for key, op in (("ring1", r1), ("ring2", r2)):
            nbr_a, nbr_b = np.asarray(op @ own_a).ravel(), np.asarray(op @ own_b).ravel()
            deltas = []
            for g in np.unique(t[members]):
                rows = members[t[members] == g]
                deltas.append(corr(own_a[rows], nbr_a[rows]) - corr(own_a[rows], nbr_b[rows]))
            per_split[key].append(float(np.mean(deltas)))
    out = {}
    for key, vals in per_split.items():
        vals = np.array(vals)
        out[key] = {"mean": float(vals.mean()),
                    "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
    out["leak_detected"] = bool(out["ring1"]["ci"][0] > 0 and out["ring1"]["mean"] > out["ring2"]["mean"])
    return out


def probe_axes(mu_z, t, scores, cycling, train):
    """beta_S/beta_G2M fitted within the cycling types (training folds);
    returns a function projecting any cells onto max(z.beta_S, z.beta_G2M)."""
    betas, means = {}, {}
    for g in cycling:
        fit = np.flatnonzero((t == g) & train)
        z_fit = mu_z[fit] - mu_z[fit].mean(axis=0)
        gram = z_fit.T @ z_fit + 1e-3 * np.eye(z_fit.shape[1])
        betas[g] = np.linalg.solve(gram, z_fit.T @ (scores[fit] - scores[fit].mean(axis=0)))
        means[g] = mu_z[fit].mean(axis=0)
    beta = np.mean(list(betas.values()), axis=0)
    mean = np.mean(list(means.values()), axis=0)
    return lambda z: ((z - mean) @ beta).max(axis=1)


def rate_matched(strength, raw_call, t_members):
    call = np.zeros(len(strength), dtype=bool)
    for g in np.unique(t_members):
        rows = np.flatnonzero(t_members == g)
        k = int(raw_call[rows].sum())
        if k:
            call[rows[np.argsort(-strength[rows])[:k]]] = True
    return call


def stratified_gap(call, exposure, t_members):
    """Positive rate in the top exposure quartile minus the bottom, per type, averaged."""
    gaps = []
    for g in np.unique(t_members):
        rows = np.flatnonzero(t_members == g)
        lo, hi = np.percentile(exposure[rows], [25, 75])
        top, bot = rows[exposure[rows] >= hi], rows[exposure[rows] <= lo]
        if len(top) > 50 and len(bot) > 50:
            gaps.append(call[top].mean() - call[bot].mean())
    return float(np.mean(gaps)) if gaps else float("nan")


def planted_world(seed=0, device="cuda", epochs=400):
    """Cycle-like program + leak on the gate scaffold; recovery z vs raw."""
    from sklearn.metrics import roc_auc_score
    from discell.applications.planted import fit_synthetic
    from discell.model.synthetic import simulate
    rng = np.random.default_rng(seed)
    sim = simulate(n_cells=6000, n_types=8, kappa=0.2, seed=seed)
    cycling_types = [0, 1]
    in_cycling = np.isin(sim.t, cycling_types)
    planted = in_cycling & (rng.random(len(sim.t)) < 0.3)
    genes = rng.choice(sim.x.shape[1], 12, replace=False)
    log_rho = np.log(sim.rho_true.clip(min=1e-12))
    log_rho[np.ix_(planted, genes)] += 1.0
    rho = np.exp(log_rho - log_rho.max(axis=1, keepdims=True))
    rho /= rho.sum(axis=1, keepdims=True)
    rho_bar = sim.graph.in_edges @ rho
    p = (1 - sim.kappa) * rho + sim.kappa * rho_bar
    p /= p.sum(axis=1, keepdims=True).clip(min=1e-12)
    sim.x = np.stack([rng.multinomial(int(sim.totals[i]), p[i]) for i in range(len(sim.t))]).astype(np.float32)
    sim.rho_true, sim.p_true = rho, p
    fit = fit_synthetic(sim, epochs=epochs, device=device, seed=seed)
    totals = sim.x.sum(axis=1)
    xn = np.log1p(sim.x / np.clip(totals, 1, None)[:, None] * np.median(totals))
    raw = xn[:, genes].mean(axis=1)
    train = fit["fold"] != 0
    project = probe_axes(fit["z"], sim.t, np.stack([raw, raw], axis=1), cycling_types, train)
    zscore = project(fit["z"])
    exposure = np.asarray(sim.graph.in_edges @ planted.astype(float)).ravel()
    victims = ~in_cycling & (exposure > np.percentile(exposure[~in_cycling], 75))
    out = {"n_planted": int(planted.sum()), "n_victims": int(victims.sum())}
    for name, score in (("raw", raw), ("z", zscore)):
        thr = np.percentile(score[in_cycling], 70)      # planted rate = 30%
        out[name] = {"auroc_cycling": float(roc_auc_score(planted[in_cycling], score[in_cycling])),
                     "victim_fpr": float((score[victims] > thr).mean()),
                     "control_fpr": float((score[~in_cycling & ~victims] > thr).mean())}
    out["pass"] = bool(out["z"]["victim_fpr"] < out["raw"]["victim_fpr"] and out["z"]["auroc_cycling"] >= out["raw"]["auroc_cycling"] - 0.02)
    return out


def quartile_rates(call, exposure, t_members):
    """Positive rate per exposure quartile, computed within type and averaged."""
    rates = np.zeros((0, 4))
    for g in np.unique(t_members):
        rows = np.flatnonzero(t_members == g)
        q = np.searchsorted(np.percentile(exposure[rows], [25, 50, 75]), exposure[rows], side="right")
        rates = np.vstack([rates, [call[rows][q == k].mean() if (q == k).any() else np.nan for k in range(4)]])
    return np.nanmean(rates, axis=0)


def a4(args) -> dict:
    import pandas as pd
    from scipy.stats import ks_2samp
    config, data, trainer, run_dir, _ = load_run(args.dataset, args.run, args.device)
    latents = collect_latents(trainer, data)
    rng = np.random.default_rng(0)
    cyc, t = data.cycle, data.t
    names = [str(n) for n in data.type_names]
    counts = np.bincount(t, minlength=len(names))
    cycling = [g for g in cyc["cycling_types"] if "nassigned" not in names[g]][:4]
    post = [g for g in range(len(names)) if g not in cycling and "nassigned" not in names[g] and counts[g] >= MIN_TYPE_CELLS]
    connected = data.graph.degrees > 0
    members = np.flatnonzero(np.isin(t, post) & connected)
    cyc_members = np.flatnonzero(np.isin(t, cycling) & connected)
    log.info("post-mitotic population: %d cells in %d types; cycling contrast: %d cells", len(members), len(post), len(cyc_members))
    # exposure = beta-weighted neighbour cycle score; calls: raw phase != G1, z rate-matched per type
    scores = np.stack([cyc["s_score"], cyc["g2m_score"]], axis=1)
    exposure = np.asarray(data.graph.in_edges @ scores.sum(axis=1)).ravel()
    raw_call = cyc["phase"] != 0
    train = latents["fold"] != 0
    strength = probe_axes(latents["mu_z"], t, scores, cycling, train)(latents["mu_z"])
    z_call = np.zeros(len(t), dtype=bool)
    z_call[members] = rate_matched(strength[members], raw_call[members], t[members])
    out = {"run": args.run, "population": {"post_mitotic": [names[g] for g in post], "cycling": [names[g] for g in cycling],
                                            "n_post": int(len(members)), "n_cycling": int(len(cyc_members)),
                                            "raw_positive_rate": float(raw_call[members].mean())}}
    # stratified gaps (top vs bottom exposure quartile) with a within-type exposure-shuffle band
    tm, em = t[members], exposure[members]
    calls = {"raw": raw_call[members], "z": z_call[members]}
    strat = {}
    for name, call in calls.items():
        null = []
        for _ in range(N_PERMS):
            shuffled = em.copy()
            for g in post:
                rows = np.flatnonzero(tm == g)
                shuffled[rows] = shuffled[rng.permutation(rows)]
            null.append(stratified_gap(call, shuffled, tm))
        strat[name] = {"gap": stratified_gap(call, em, tm),
                       "null_band": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                       "quartile_rates": quartile_rates(call, em, tm).tolist(),
                       "per_type": {names[g]: stratified_gap(call[tm == g], em[tm == g], tm[tm == g]) for g in post}}
    strat["raw_exceeds_band"] = bool(strat["raw"]["gap"] > strat["raw"]["null_band"][1])
    strat["z_below_raw"] = bool(strat["z"]["gap"] < strat["raw"]["gap"])
    out["stratified"] = strat
    # DAPI, group level: both-positive / raw-only / neither, dapi_sum standardised within type
    qc = pd.read_parquet(paths.dataset(args.dataset).root / "qc" / "nuclear_dapi.parquet")
    assert len(qc) == len(t), "DAPI table must be bundle-aligned"
    dapi = qc["dapi_sum"].to_numpy(dtype=float)
    std = np.full(len(t), np.nan)
    for g in post:
        rows = members[tm == g]
        ok = rows[np.isfinite(dapi[rows])]
        std[ok] = (dapi[ok] - np.median(dapi[ok])) / (dapi[ok].std() + 1e-12)
    groups = {"both": raw_call & z_call, "raw_only": raw_call & ~z_call, "neither": ~raw_call & ~z_call}
    vals = {k: std[members][m[members] & np.isfinite(std[members])] for k, m in groups.items()}
    out["dapi"] = {k: {"n": int(len(v)), "median": float(np.median(v)) if len(v) else float("nan")} for k, v in vals.items()}
    for k in ("both", "raw_only"):
        ks = ks_2samp(vals[k], vals["neither"])
        out["dapi"][k].update(ks_vs_neither=float(ks.statistic), p=float(ks.pvalue))
    # gene-split fingerprint on both populations (post-mitotic = claim, cycling = contrast)
    gene_names = [str(g) for g in data.gene_names]
    cycle_idx = np.array([gene_names.index(g) for g in S_GENES + G2M_GENES if g in gene_names])
    xn = log_norm(data.x, data.totals, data.median_counts)
    r1, r2 = ring_operators(data)
    out["fingerprint"] = {"post_mitotic": gene_split_delta(xn, cycle_idx, r1, r2, t, members, rng),
                          "cycling": gene_split_delta(xn, cycle_idx, r1, r2, t, cyc_members, rng)}
    log.info("gene-split delta post-mitotic: %s", out["fingerprint"]["post_mitotic"])
    # planted worlds, three seeds
    out["planted"] = [planted_world(seed=s, device=args.device) for s in range(3)]
    out["planted_pass"] = bool(all(p["pass"] for p in out["planted"]))
    # verdict lines, pre-registered (doc 11 A4 redesign)
    verdict = []
    fp = out["fingerprint"]["post_mitotic"]
    verdict.append("gene-split fingerprint (post-mitotic): " + ("leak signature present (ring-1 delta > 0, > ring-2)" if fp["leak_detected"] else "no contamination signature (delta ring-1 CI covers 0 or not one-hop)"))
    if strat["raw_exceeds_band"]:
        verdict.append("raw calls track neighbour-cycle exposure beyond the shuffle band; z " + ("rejects part of it (gap smaller)" if strat["z_below_raw"] else "does NOT reduce the gap"))
    else:
        verdict.append("raw calls do not track exposure beyond the shuffle band: leak-induced cycle false-positives are rare at kappa=0.1 on this slide (pre-registered honest outcome)")
    d = out["dapi"]
    verdict.append("DAPI (group level, weak proxy): both-positive median %.2f (KS %.2f), raw-only %.2f (KS %.2f), neither 0 by construction" % (d["both"]["median"], d["both"]["ks_vs_neither"], d["raw_only"]["median"], d["raw_only"]["ks_vs_neither"]))
    verdict.append("planted world (primary adjudicator): %d/3 seeds pass; victim FPR raw %.2f vs z %.2f, AUROC raw %.2f vs z %.2f (seed means)" % (
        sum(p["pass"] for p in out["planted"]), *[float(np.mean([p[k][m] for p in out["planted"]])) for k, m in (("raw", "victim_fpr"), ("z", "victim_fpr"), ("raw", "auroc_cycling"), ("z", "auroc_cycling"))]))
    out["verdict"] = verdict
    out_dir = run_dir / "applications"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "a4_cycle.json").write_text(json.dumps(out, indent=2))
    a4_figure(out, vals, out_dir / "a4_cycle.png")
    for line in verdict:
        log.info("VERDICT %s", line)
    return out


def a4_figure(out, dapi_vals, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    ax = axes[0]
    for name, c in (("raw", "tab:red"), ("z", "tab:blue")):
        ax.plot(range(4), out["stratified"][name]["quartile_rates"], "o-", color=c, label="%s call (gap %.3f)" % (name, out["stratified"][name]["gap"]))
    lo, hi = out["stratified"]["raw"]["null_band"]
    ax.set_title("positive rate vs neighbour-cycle exposure\n(post-mitotic types; raw shuffle band [%.3f, %.3f])" % (lo, hi), fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Q1 low", "Q2", "Q3", "Q4 high"]); ax.set_ylabel("call rate"); ax.legend(fontsize=8)
    ax = axes[1]
    for k, (pop, c) in enumerate((("post_mitotic", "tab:blue"), ("cycling", "tab:gray"))):
        for j, ring in enumerate(("ring1", "ring2")):
            r = out["fingerprint"][pop][ring]
            x = k * 2.5 + j
            ax.bar(x, r["mean"], color=c, alpha=0.9 if j == 0 else 0.5)
            ax.plot([x, x], r["ci"], color="k")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([0, 1, 2.5, 3.5]); ax.set_xticklabels(["post r1", "post r2", "cyc r1", "cyc r2"])
    ax.set_title("gene-split delta = corr(own_A, nbr_A) - corr(own_A, nbr_B)\n(leak: > 0 and one-hop; homophily: ~0)", fontsize=9)
    ax = axes[2]
    keys = [k for k in ("both", "raw_only", "neither") if len(dapi_vals[k])]
    ax.boxplot([dapi_vals[k] for k in keys], showfliers=False)
    ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels(["%s\nn=%d" % (k, out["dapi"][k]["n"]) for k in keys], fontsize=8)
    ax.set_title("nuclear DAPI, standardised within type\n(S/G2M expected high; raw-only ~ neither if contamination)", fontsize=9)
    ax = axes[3]
    for j, (k, m, lab) in enumerate((("raw", "victim_fpr", "victim FPR raw"), ("z", "victim_fpr", "victim FPR z"), ("raw", "auroc_cycling", "AUROC raw"), ("z", "auroc_cycling", "AUROC z"))):
        v = [p[k][m] for p in out["planted"]]
        ax.bar(j, np.mean(v), color="tab:red" if k == "raw" else "tab:blue", alpha=0.8)
        ax.scatter([j] * len(v), v, color="k", s=10, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["victim FPR\nraw", "victim FPR\nz", "AUROC\nraw", "AUROC\nz"], fontsize=8)
    ax.set_title("planted world (3 seeds): victim FPR lower + AUROC kept = pass\n%d/3 pass" % sum(p["pass"] for p in out["planted"]), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    a4(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
