#!/usr/bin/env python3
"""Three external criteria, taken from the baseline papers, run on our runs.

Package 1 of the devlog entry "Baselines and article-derived experiments:
three packages" (2026-09-21). Nothing here trains anything; every read is on
finished runs, on held-out tiles, with a permutation floor beside it.

**6b.1 -- MintFlow's signalling-gene criterion** (their p. 5 / Methods
p. 39: genes in any ligand-receptor database should take a higher
microenvironment-induced share of their counts than other genes). Our
analogue, on the transport panels: per panel (one cell type, one ordered
niche pair) the observed held-out log-rate shift is split into a *response*
part (the context prior through the loadings, ``B (m_psi(B) - m_psi(A))``),
a *leak* part (``kappa`` times the change in foreign influx) and an
unexplained remainder; each gene's share of each part is accumulated over
panels. The pre-registered read compares the LR-gene distribution of each
share with the non-LR distribution (Mann-Whitney, rank-biserial effect
size). If the LEAK share alone separates them, MintFlow's only real-data
validation is reproducible from misassignment; if only the RESPONSE share
does, it is not.

**6b.3 -- DisCoVR's MIG / MIC.** y = niche label (k-means composition
niches, and the ordered tumour bands where the slide is annotated).
``I(y;z)``, ``I(y;w)`` and ``I(w;z|y)`` on held-out tiles by a kNN
(Kraskov/Ross) estimator and by a small MINE, each with a within-type
permutation floor; ``H(y)`` empirical.
``MIG = (I(y;w) - I(y;z)) / H(y)``, ``MIC = I(y;w) / (I(y;w) + I(y;z))``.

**6b.4 -- SIMVI's true-axis / false-axis test** (their p. 5). Per gene, the
Kendall tau of a predicted per-band shift against the ordered tumour-band
index (the true axis) and against equal-count bins of the in-plane
coordinate least correlated with the band index (the false axis), on the
same cells. Rows: the w-predicted shift, the raw held-out shift, a z row
and a depth (l) row. TP / FP gene counts at |tau| thresholds.

Usage::

    python -m discell.experiments.external_criteria signalling-share \\
        --dataset <id> --run <run> [--run <run> ...]
    python -m discell.experiments.external_criteria mi-quadrant \\
        --dataset <id> --run <run> [--niche-source kmeans|tumour-band]
    python -m discell.experiments.external_criteria axis-test \\
        --dataset <id> --run <run>

Outputs land in ``data/datasets/<ds>/experiments/external_*.json`` (plus a
``.png`` per experiment).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from discell import paths
from discell.model.transport import (EPS, MIN_CELLS, MIN_RATE, TUMOUR_BANDS,
                                     collect_channels, pick_pairs,
                                     tumour_band_labels)
from discell.model.validate import collect_latents, load_run, niche_labels

log = logging.getLogger("discell.experiments.external_criteria")

EXTERNAL = Path("data/external")
MI_K = 5                  #: neighbours for the Kraskov / Ross estimators
MI_CELLS = 20_000         #: cap on held-out cells entering an MI estimate
N_PERM = 10               #: within-type permutations for each MI floor
MINE_STEPS = 600
MINE_HIDDEN = 128
TAU_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
MIN_BAND_CELLS = 200
N_FALSE_BINS = len(TUMOUR_BANDS) + 1


# ===========================================================================
# 6b.1  MintFlow's signalling-gene criterion
# ===========================================================================

def lr_gene_union(panel: set[str]) -> set[str]:
    """Every CellChatDB ligand or receptor subunit present in the panel.

    MintFlow's criterion is "any gene in any ligand-receptor database", so
    this is the union with NO NicheNet target gate -- deliberately wider
    than doc-09's 618 scored pairs, which additionally require >= 20 in-panel
    targets. Provenance is doc-09's: ``data/external/CellChatDB.human.rda``.
    """
    import rdata

    db = rdata.read_rda(EXTERNAL / "CellChatDB.human.rda")["CellChatDB.human"]
    inter, complexes = db["interaction"], db["complex"]

    def subunits(name: str) -> list[str]:
        if name in complexes.index:
            return [s for s in complexes.loc[name] if isinstance(s, str) and s]
        return [name]

    genes: set[str] = set()
    for _, row in inter.iterrows():
        for name in (row["ligand"], row["receptor"]):
            genes.update(subunits(name))
    return {g for g in genes if g in panel}


def accumulate_shares(panels: Sequence[dict], n_genes: int) -> dict:
    """Per-gene share of the observed shift taken by each channel.

    Each panel contributes, for the genes it scores, ``|response|``,
    ``|leak|`` and ``|observed - response - leak|`` (all three vectors
    mean-centred first, as ``transport.score_shift`` centres them: the
    softmax normaliser and depth are per-panel constants and must not be
    charged to a channel). Summing the three over panels and dividing gives
    a per-gene share in [0, 1] that is a share of the *magnitude* of the
    observed shift -- the closest analogue of MintFlow's share of counts
    that a shift-level instrument admits.

    ``both`` is ``(response + leak) / total``, i.e. one minus the
    unexplained share, and is reported as the third column of the criterion.
    """
    acc = np.zeros((3, n_genes))
    hits = np.zeros(n_genes)
    for panel in panels:
        keep = panel["keep"]
        def c(v):
            return v - v.mean()
        resp, leak, obs = c(panel["response"]), c(panel["leak"]), c(panel["observed"])
        acc[0, keep] += np.abs(resp)
        acc[1, keep] += np.abs(leak)
        acc[2, keep] += np.abs(obs - resp - leak)
        hits[keep] += 1
    total = acc.sum(axis=0)
    ok = (hits > 0) & (total > 1e-12)
    out = np.full((3, n_genes), np.nan)
    out[:, ok] = acc[:, ok] / total[ok]
    return {"response": out[0], "leak": out[1], "unexplained": out[2],
            "both": out[0] + out[1], "n_panels_per_gene": hits, "scored": ok}


def mann_whitney(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Mann-Whitney U with the rank-biserial effect size.

    ``effect = 2 * AUC - 1``: +1 means every *a* (the LR genes) exceeds
    every *b*, 0 means no separation. Reported always beside p, because on
    ~5,000 genes a p-value alone says nothing about size."""
    from scipy.stats import mannwhitneyu

    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"n_lr": int(len(a)), "n_other": int(len(b)), "u": float("nan"),
                "p": float("nan"), "effect": float("nan"),
                "median_lr": float("nan"), "median_other": float("nan")}
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    auc = float(u) / (len(a) * len(b))
    return {"n_lr": int(len(a)), "n_other": int(len(b)), "u": float(u),
            "p": float(p), "effect": float(2 * auc - 1),
            "median_lr": float(np.median(a)),
            "median_other": float(np.median(b))}


def abundance_matched_other(is_lr: np.ndarray, scored: np.ndarray,
                            mean_rate: np.ndarray, rng,
                            n_bins: int = 20) -> np.ndarray:
    """A non-LR comparison set matched to the LR genes on expression level.

    Doc-09's load-bearing residualisation (E-tilde = E - ridge(E ~ y)) has no
    direct analogue here: a transport panel is *within one cell type* and its
    predictor IS neighbour composition, so there is nothing left to partial
    out. What the residualisation protects against -- a signalling gene
    scoring high because it marks a type that is simply more abundant on one
    side -- is instead handled by matching: LR genes are more highly
    expressed than the panel average, and both the leak and the response
    channel are better estimated in well-expressed genes. Sampling the
    non-LR set to the LR set's log-mean-rate histogram removes that route.
    This is a DEVIATION from doc-09 section 2 and is reported as one.
    """
    lr = np.flatnonzero(scored & is_lr)
    other = np.flatnonzero(scored & ~is_lr)
    if len(lr) == 0 or len(other) == 0:
        return other
    lo = np.log(np.maximum(mean_rate, MIN_RATE))
    edges = np.quantile(lo[scored], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins_lr = np.digitize(lo[lr], edges[1:-1])
    bins_ot = np.digitize(lo[other], edges[1:-1])
    picked = []
    for b in range(n_bins):
        want = int((bins_lr == b).sum())
        pool = other[bins_ot == b]
        if want == 0 or len(pool) == 0:
            continue
        picked.append(rng.choice(pool, min(want, len(pool)), replace=False)
                      if len(pool) > want else pool)
    return np.concatenate(picked) if picked else other


def signalling_share(args: argparse.Namespace) -> dict:
    """6b.1 on one run: build the panels, accumulate shares, run the test."""
    config, data, trainer, run_dir, b_matrix = load_run(
        args.dataset, args.run, args.device)
    latents = collect_latents(trainer, data)
    connected = data.graph.degrees > 0
    source = args.niche_source
    try:
        labels = (tumour_band_labels(data) if source == "tumour-band"
                  else niche_labels(data, args.niches, config.seed))
    except ValueError as exc:                # no tumour-annotated type
        log.warning("%s -- falling back to composition niches", exc)
        source = "kmeans"
        labels = niche_labels(data, args.niches, config.seed)
    held_out = latents["fold"] == 0
    kappa = config.kappa
    n_types = len(data.p_t)
    n_niches = int(labels.max()) + 1
    model_rows = connected & ~held_out & (labels >= 0)
    group = np.full(data.graph.n_cells, -1, dtype=np.int64)
    group[model_rows] = labels[model_rows] * n_types + data.t[model_rows]
    channels = collect_channels(trainer, group, n_niches * n_types)

    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    mean_rate = np.asarray(x_rate[connected].mean(axis=0)).ravel()
    names = [str(n) for n in data.type_names]
    built = []
    for niche_a, niche_b in pick_pairs(labels, data.graph.y, connected):
        for g in range(n_types):
            rows = {}
            ok = True
            for niche, side in ((niche_a, "A"), (niche_b, "B")):
                tr = np.flatnonzero(connected & ~held_out & (data.t == g)
                                    & (labels == niche))
                te = np.flatnonzero(connected & held_out & (data.t == g)
                                    & (labels == niche))
                if len(tr) < MIN_CELLS or len(te) < MIN_CELLS // 5:
                    ok = False
                    break
                rows[side] = te
            if not ok:
                continue
            gid_a = niche_a * n_types + g
            gid_b = niche_b * n_types + g
            rho_a, bar_a = channels["rho"][gid_a], channels["rho_bar"][gid_a]
            rho_b, bar_b = channels["rho"][gid_b], channels["rho_bar"][gid_b]
            response = b_matrix @ (channels["prior_w"][gid_b]
                                   - channels["prior_w"][gid_a])
            leak = (np.log((1 - kappa) * rho_a + kappa * bar_b + EPS)
                    - np.log((1 - kappa) * rho_a + kappa * bar_a + EPS))
            obs_a = np.asarray(x_rate[rows["A"]].mean(axis=0)).ravel()
            obs_b = np.asarray(x_rate[rows["B"]].mean(axis=0)).ravel()
            keep = np.flatnonzero((obs_a > MIN_RATE) & (obs_b > MIN_RATE))
            if len(keep) < 100:
                continue
            built.append({"keep": keep, "response": response[keep],
                          "leak": leak[keep],
                          "observed": (np.log(obs_b[keep] + EPS)
                                       - np.log(obs_a[keep] + EPS)),
                          "type": names[g],
                          "pair": (int(niche_a), int(niche_b))})

    gene_names = np.asarray([str(g) for g in data.gene_names])
    shares = accumulate_shares(built, len(gene_names))
    lr = lr_gene_union(set(gene_names.tolist()))
    is_lr = np.isin(gene_names, sorted(lr))
    rng = np.random.default_rng(config.seed)
    matched = abundance_matched_other(is_lr, shares["scored"], mean_rate, rng)

    result = {"dataset": args.dataset, "run": args.run, "kappa": float(kappa),
              "niche_source": source, "n_panels": len(built),
              "n_genes_scored": int(shares["scored"].sum()),
              "n_lr_in_panel": int(is_lr.sum()), "tests": {}}
    for channel in ("response", "leak", "both"):
        v = shares[channel]
        lr_v = v[shares["scored"] & is_lr]
        result["tests"][channel] = {
            "all_other": mann_whitney(lr_v, v[shares["scored"] & ~is_lr]),
            "abundance_matched": mann_whitney(lr_v, v[matched])}
    result["mean_share"] = {c: float(np.nanmean(shares[c][shares["scored"]]))
                            for c in ("response", "leak", "unexplained")}
    result["_shares"] = shares
    result["_is_lr"] = is_lr
    return result


# ===========================================================================
# 6b.3  DisCoVR's MIG / MIC
# ===========================================================================

def mi_discrete_continuous(x: np.ndarray, y: np.ndarray, k: int = MI_K,
                           rng=None) -> float:
    """I(discrete y ; continuous vector x) by Ross (2014), in nats.

    The discrete-continuous member of the Kraskov family: for each point,
    the distance to its k-th neighbour WITHIN its own class sets a ball, and
    the count of all points in that ball supplies the correction.
    ``I = psi(N) - <psi(N_c)> + psi(k) - <psi(m)>``. Ties are broken by a
    tiny jitter (latent coordinates are continuous, but duplicated rows do
    occur on capped subsamples)."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x = np.asarray(x, float)
    x = x.reshape(len(x), -1)
    rng = np.random.default_rng(0) if rng is None else rng
    x = x + 1e-10 * rng.standard_normal(x.shape) * (x.std(axis=0) + 1e-12)
    y = np.asarray(y)
    n = len(x)
    full = cKDTree(x)
    radius = np.zeros(n)
    n_class = np.zeros(n)
    k_eff = np.zeros(n)
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        n_class[rows] = len(rows)
        kk = min(k, len(rows) - 1)
        if kk < 1:
            radius[rows], k_eff[rows] = np.inf, np.nan
            continue
        d = cKDTree(x[rows]).query(x[rows], k=kk + 1, p=np.inf)[0][:, kk]
        radius[rows], k_eff[rows] = d, kk
    good = np.isfinite(radius) & np.isfinite(k_eff)
    # scipy takes a per-point radius vector; sklearn's radius_neighbors
    # rejects one on its fast path (the whole point of the Ross estimator
    # is that every point has its OWN ball)
    m = np.asarray(full.query_ball_point(x[good], r=radius[good] * (1 - 1e-12),
                                         p=np.inf, return_length=True))
    value = (digamma(n) - digamma(n_class[good]).mean()
             + digamma(k_eff[good]).mean() - digamma(np.maximum(m, 1)).mean())
    return float(max(value, 0.0))


def mi_continuous(x: np.ndarray, y: np.ndarray, k: int = MI_K,
                  rng=None) -> float:
    """I(x;y) for two continuous vectors, Kraskov estimator 1, in nats."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, float).reshape(len(x), -1)
    y = np.asarray(y, float).reshape(len(y), -1)
    x = x + 1e-10 * rng.standard_normal(x.shape) * (x.std(axis=0) + 1e-12)
    y = y + 1e-10 * rng.standard_normal(y.shape) * (y.std(axis=0) + 1e-12)
    n = len(x)
    k = min(k, n - 1)
    if k < 1:
        return 0.0
    joint = np.hstack([x, y])
    eps = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, k]
    counts = []
    for v in (x, y):
        counts.append(np.asarray(cKDTree(v).query_ball_point(
            v, r=eps * (1 - 1e-12), p=np.inf, return_length=True)) - 1)
    nx, ny = counts
    value = (digamma(k) + digamma(n)
             - digamma(np.maximum(nx, 0) + 1).mean()
             - digamma(np.maximum(ny, 0) + 1).mean())
    return float(max(value, 0.0))


def cmi_knn(w: np.ndarray, z: np.ndarray, y: np.ndarray, k: int = MI_K,
            rng=None) -> float:
    """I(w;z|y) for discrete y: the p(y)-weighted mean of within-class MI."""
    total, n = 0.0, len(y)
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        if len(rows) < 4 * k:
            continue
        total += (len(rows) / n) * mi_continuous(w[rows], z[rows], k, rng)
    return float(total)


def mine(x: np.ndarray, y: np.ndarray, steps: int = MINE_STEPS,
         hidden: int = MINE_HIDDEN, seed: int = 0, device: str = "cpu") -> float:
    """Donsker-Varadhan lower bound on I(x;y) from a small MLP statistic.

    Deliberately small and short: this is the *second opinion* on the kNN
    number, not an independent headline. The bound is evaluated on a held-out
    half of the rows the estimator was trained on, so an over-fitted critic
    reads low rather than high -- the conservative direction for a bound."""
    import torch

    torch.manual_seed(seed)
    dev = torch.device(device)
    x = torch.as_tensor(np.asarray(x, np.float32).reshape(len(x), -1), device=dev)
    y = torch.as_tensor(np.asarray(y, np.float32).reshape(len(y), -1), device=dev)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = (y - y.mean(0)) / (y.std(0) + 1e-6)
    n = len(x)
    perm = torch.randperm(n, device=dev)
    fit, evl = perm[: n // 2], perm[n // 2:]
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1] + y.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = min(512, len(fit))

    def bound(rows):
        idx = rows[torch.randint(len(rows), (batch,), device=dev)]
        shuf = rows[torch.randint(len(rows), (batch,), device=dev)]
        joint = net(torch.cat([x[idx], y[idx]], 1))
        marg = net(torch.cat([x[idx], y[shuf]], 1))
        return joint.mean() - (torch.logsumexp(marg, 0)
                               - np.log(len(marg)))

    for _ in range(steps):
        opt.zero_grad()
        (-bound(fit)).backward()
        opt.step()
    with torch.no_grad():
        value = float(np.mean([float(bound(evl)) for _ in range(20)]))
    return max(value, 0.0)


def one_hot(y: np.ndarray) -> np.ndarray:
    ks = np.unique(y)
    return (y[:, None] == ks[None, :]).astype(np.float64)


def within_type_permutation(y: np.ndarray, t: np.ndarray, rng) -> np.ndarray:
    """Shuffle the niche label among cells of the same type.

    The floor this builds is the MI a latent can get from cell TYPE alone:
    niches differ in composition, so a type-informative latent is niche-
    informative for free. Everything above this floor is niche information
    that type identity does not already carry."""
    out = y.copy()
    for g in np.unique(t):
        rows = np.flatnonzero(t == g)
        out[rows] = y[rng.permutation(rows)]
    return out


def mi_quadrant(args: argparse.Namespace) -> dict:
    """6b.3 on one run: the {I(y;z), I(y;w), I(w;z|y)} table with floors."""
    config, data, trainer, run_dir, _ = load_run(
        args.dataset, args.run, args.device)
    latents = collect_latents(trainer, data)
    connected = data.graph.degrees > 0
    out = {"dataset": args.dataset, "run": args.run, "kappa": float(config.kappa),
           "sources": {}}
    rng = np.random.default_rng(config.seed)

    sources = ["kmeans"]
    try:
        bands = tumour_band_labels(data)
        sources.append("tumour-band")
    except ValueError as exc:
        bands = None
        out["tumour_band_note"] = str(exc)

    for source in sources:
        labels = (bands if source == "tumour-band"
                  else niche_labels(data, args.niches, config.seed))
        rows = np.flatnonzero(connected & (labels >= 0)
                              & (latents["fold"] == 0))   # held-out tiles
        if len(rows) > MI_CELLS:
            rows = np.sort(rng.choice(rows, MI_CELLS, replace=False))
        y, t = labels[rows], data.t[rows]
        z, w = latents["mu_z"][rows], latents["mu_w"][rows]
        p = np.bincount(y - y.min()) / len(y)
        p = p[p > 0]
        h_y = float(-(p * np.log(p)).sum())

        entry = {"n_cells": int(len(rows)), "n_niches": int(len(np.unique(y))),
                 "H_y_nats": h_y, "knn": {}, "mine": {}}
        knn = {"I_yz": mi_discrete_continuous(z, y, rng=rng),
               "I_yw": mi_discrete_continuous(w, y, rng=rng),
               "I_wz_given_y": cmi_knn(w, z, y, rng=rng)}
        floors = {"I_yz": [], "I_yw": []}
        for _ in range(args.perms):
            y_perm = within_type_permutation(y, t, rng)
            floors["I_yz"].append(mi_discrete_continuous(z, y_perm, rng=rng))
            floors["I_yw"].append(mi_discrete_continuous(w, y_perm, rng=rng))
        entry["knn"] = dict(knn)
        entry["knn_floor"] = {k: float(np.mean(v)) for k, v in floors.items()}
        entry["knn_floor_sd"] = {k: float(np.std(v)) for k, v in floors.items()}

        yh = one_hot(y)
        m_yz = mine(z, yh, seed=config.seed, device=args.device)
        m_yw = mine(w, yh, seed=config.seed, device=args.device)
        # I(w;z|y) = I(w;(z,y)) - I(w;y)
        m_cmi = max(mine(w, np.hstack([z, yh]), seed=config.seed,
                         device=args.device) - m_yw, 0.0)
        entry["mine"] = {"I_yz": m_yz, "I_yw": m_yw, "I_wz_given_y": m_cmi}
        y_perm = within_type_permutation(y, t, rng)
        ph = one_hot(y_perm)
        entry["mine_floor"] = {
            "I_yz": mine(z, ph, seed=config.seed, device=args.device),
            "I_yw": mine(w, ph, seed=config.seed, device=args.device)}

        for est in ("knn", "mine"):
            iz, iw = entry[est]["I_yz"], entry[est]["I_yw"]
            fz, fw = entry[f"{est}_floor"]["I_yz"], entry[f"{est}_floor"]["I_yw"]
            entry[est]["MIG"] = float((iw - iz) / h_y) if h_y > 0 else float("nan")
            entry[est]["MIC"] = float(iw / (iw + iz)) if (iw + iz) > 1e-9 else float("nan")
            entry[est]["MIG_floor_corrected"] = float(
                (max(iw - fw, 0) - max(iz - fz, 0)) / h_y) if h_y > 0 else float("nan")
            entry[est]["I_yz_excess_over_floor"] = float(iz - fz)
            entry[est]["I_yw_excess_over_floor"] = float(iw - fw)
        out["sources"][source] = entry
    return out


# ===========================================================================
# 6b.4  SIMVI's true-axis / false-axis test
# ===========================================================================

def kendall_tau_rows(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Kendall tau-b of every column of *values* against *order*.

    *values* is ``(n_bands, n_genes)``; ``order`` is the band index. With
    a handful of bands the O(n^2) pair loop is vectorised over genes and
    costs nothing. Ties in *values* are handled (tau-b); ``order`` is a
    strict ranking by construction."""
    values = np.asarray(values, float)
    n = len(order)
    num = np.zeros(values.shape[1])
    ties_v = np.zeros(values.shape[1])
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            s_o = np.sign(order[j] - order[i])
            if s_o == 0:
                continue
            n_pairs += 1
            d = values[j] - values[i]
            num += s_o * np.sign(d)
            ties_v += (d == 0)
    denom = np.sqrt(np.maximum(n_pairs - ties_v, 0) * n_pairs)
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)


def equal_count_bins(v: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))[1:-1]
    return np.digitize(v, edges)


def axis_counts(tau_true: np.ndarray, tau_false: np.ndarray) -> dict:
    """SIMVI's read-out: gene counts above |tau| on each axis, and the ratio."""
    out = {}
    for thr in TAU_THRESHOLDS:
        tp = int((np.abs(tau_true) >= thr).sum())
        fp = int((np.abs(tau_false) >= thr).sum())
        out[f"{thr:g}"] = {"true_axis": tp, "false_axis": fp,
                           "ratio": float(tp / fp) if fp else float("inf"),
                           "excess": tp - fp}
    return out


def axis_test(args: argparse.Namespace) -> dict:
    """6b.4 on one run: four predictor rows x (true axis, false axis)."""
    config, data, trainer, run_dir, b_matrix = load_run(
        args.dataset, args.run, args.device)
    latents = collect_latents(trainer, data)
    connected = data.graph.degrees > 0
    bands = tumour_band_labels(data)          # raises on unannotated slides
    held_out = latents["fold"] == 0
    n_types = len(data.p_t)
    n_bands = int(bands.max()) + 1

    # the false axis: the in-plane coordinate least correlated with the band
    # index over the cells that carry a band, binned to the same number of
    # equal-count bins. Reported by name so the reader knows which it was.
    from scipy.stats import spearmanr
    have = connected & (bands >= 0)
    rho_x = abs(float(spearmanr(data.positions[have, 0], bands[have])[0]))
    rho_y = abs(float(spearmanr(data.positions[have, 1], bands[have])[0]))
    axis_dim = 0 if rho_x <= rho_y else 1
    false_field = data.positions[:, axis_dim]

    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    x_counts = data.x.tocsr()
    names = [str(n) for n in data.type_names]
    result = {"dataset": args.dataset, "run": args.run,
              "kappa": float(config.kappa), "n_bands": n_bands,
              "false_axis": {"coordinate": "xy"[axis_dim],
                             "spearman_with_band": {"x": rho_x, "y": rho_y}},
              "types": {}, "pooled": {}}

    pooled = {row: {"true": [], "false": []}
              for row in ("w_predicted", "raw_observed", "z_row", "l_row")}
    for g in range(n_types):
        members = connected & (data.t == g)
        if members.sum() < MIN_BAND_CELLS * n_bands:
            continue
        # equal-count false bins WITHIN this type, so both axes cut the same
        # cells into the same number of groups of the same sizes
        fb = np.full(len(bands), -1, dtype=np.int64)
        fb[members] = equal_count_bins(false_field[members], N_FALSE_BINS)

        rows = {}
        for axis, lab in (("true", bands), ("false", fb)):
            n_groups = int(lab[members].max()) + 1
            group = np.full(data.graph.n_cells, -1, dtype=np.int64)
            model_rows = members & ~held_out & (lab >= 0)
            group[model_rows] = lab[model_rows]
            ch = collect_channels(trainer, group, n_groups)
            keep_band = []
            prog, zrow, raw, lrow = [], [], [], []
            for b in range(n_groups):
                te = np.flatnonzero(members & held_out & (lab == b))
                if ch["n"][b] < MIN_CELLS or len(te) < MIN_BAND_CELLS:
                    continue
                keep_band.append(b)
                prog.append(b_matrix @ ch["prior_w"][b])
                # the model's own decontaminated log rate MINUS the w path:
                # what the intrinsic latent contributes to this band's mean
                zrow.append(np.log(ch["rho"][b] + EPS) - prog[-1])
                raw.append(np.log(np.asarray(
                    x_rate[te].mean(axis=0)).ravel() + EPS))
                # depth / density row: the same means WITHOUT depth
                # normalisation, i.e. what you see if only library size and
                # cell density move
                lrow.append(np.log(np.asarray(
                    x_counts[te].mean(axis=0)).ravel() + EPS))
            if len(keep_band) < 4:
                rows = {}
                break
            order = np.array(keep_band, float)
            keep_gene = np.all(np.exp(np.array(raw)) > MIN_RATE, axis=0)
            rows[axis] = {
                "w_predicted": kendall_tau_rows(np.array(prog)[:, keep_gene], order),
                "raw_observed": kendall_tau_rows(np.array(raw)[:, keep_gene], order),
                "z_row": kendall_tau_rows(np.array(zrow)[:, keep_gene], order),
                "l_row": kendall_tau_rows(np.array(lrow)[:, keep_gene], order),
                "n_bands_used": len(keep_band), "n_genes": int(keep_gene.sum())}
        if "true" not in rows or "false" not in rows:
            continue
        n_common = min(rows["true"]["n_genes"], rows["false"]["n_genes"])
        entry = {"n_bands_true": rows["true"]["n_bands_used"],
                 "n_bands_false": rows["false"]["n_bands_used"],
                 "n_genes": n_common, "rows": {}}
        for row in ("w_predicted", "raw_observed", "z_row", "l_row"):
            tt, tf = rows["true"][row], rows["false"][row]
            if len(tt) != len(tf):            # different gene masks per axis
                m = min(len(tt), len(tf))
                tt, tf = tt[:m], tf[:m]
            entry["rows"][row] = {
                "counts": axis_counts(tt, tf),
                "mean_abs_tau_true": float(np.abs(tt).mean()),
                "mean_abs_tau_false": float(np.abs(tf).mean())}
            pooled[row]["true"].append(tt)
            pooled[row]["false"].append(tf)
        result["types"][names[g]] = entry

    for row, d in pooled.items():
        if not d["true"]:
            continue
        tt = np.concatenate(d["true"])
        tf = np.concatenate(d["false"])
        result["pooled"][row] = {
            "n_gene_panels": int(len(tt)), "counts": axis_counts(tt, tf),
            "mean_abs_tau_true": float(np.abs(tt).mean()),
            "mean_abs_tau_false": float(np.abs(tf).mean())}
    return result


# ===========================================================================
# figures + CLI
# ===========================================================================

def share_figure(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shares, is_lr = result["_shares"], result["_is_lr"]
    ok = shares["scored"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, channel in zip(axes, ("response", "leak", "both")):
        v = shares[channel]
        bins = np.linspace(0, 1, 41)
        ax.hist(v[ok & ~is_lr], bins=bins, density=True, alpha=0.55,
                label=f"other (n={int((ok & ~is_lr).sum())})", color="0.6")
        ax.hist(v[ok & is_lr], bins=bins, density=True, alpha=0.55,
                label=f"LR (n={int((ok & is_lr).sum())})", color="#c9662a")
        t = result["tests"][channel]["all_other"]
        ax.set_title(f"{channel}: effect {t['effect']:+.3f}, p={t['p']:.1e}",
                     fontsize=8)
        ax.set_xlabel("share of |observed shift|")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("density")
    fig.suptitle(f"{result['run']} (kappa {result['kappa']}), "
                 f"{result['n_panels']} panels, {result['niche_source']} niches",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def axis_figure(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in ("w_predicted", "raw_observed", "z_row", "l_row")
            if r in result["pooled"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.38
    xs = np.arange(len(rows))
    thr = f"{TAU_THRESHOLDS[1]:g}"
    tp = [result["pooled"][r]["counts"][thr]["true_axis"] for r in rows]
    fp = [result["pooled"][r]["counts"][thr]["false_axis"] for r in rows]
    ax.bar(xs - width / 2, tp, width, label="true axis (band index)",
           color="#2a6ec9")
    ax.bar(xs + width / 2, fp, width,
           label=f"false axis ({result['false_axis']['coordinate']})",
           color="#c9662a")
    ax.set_xticks(xs)
    ax.set_xticklabels(rows, fontsize=8)
    ax.set_ylabel(f"gene-panels with |tau| >= {thr}")
    ax.set_title(f"{result['run']}: SIMVI axis test", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("signalling-share", "mi-quadrant", "axis-test"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--run", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--niches", type=int, default=10)
        p.add_argument("--tag", default=None)
        if name == "signalling-share":
            p.add_argument("--niche-source", default="kmeans",
                           choices=["kmeans", "tumour-band"])
        if name == "mi-quadrant":
            p.add_argument("--perms", type=int, default=N_PERM)
    args = parser.parse_args(argv)

    out_dir = paths.dataset(args.dataset).root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.run
    if args.command == "signalling-share":
        result = signalling_share(args)
        share_figure(result, out_dir / f"external_signalling_share_{tag}.png")
        for k in ("_shares", "_is_lr"):
            result.pop(k)
        stem = f"external_signalling_share_{tag}"
    elif args.command == "mi-quadrant":
        result = mi_quadrant(args)
        stem = f"external_mi_quadrant_{tag}"
    else:
        result = axis_test(args)
        axis_figure(result, out_dir / f"external_axis_test_{tag}.png")
        stem = f"external_axis_test_{tag}"
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2,
                                                     default=float))
    log.info("wrote %s", out_dir / f"{stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
