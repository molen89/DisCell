"""The paper's findings: each fit's tile draws of its findings and checks.

Every member is a signed finding with null 0, computed from one fitted model on
one section; :func:`fit_draws` returns, per member, the point estimate and its
draws in the form :func:`discell.sensitivity.breakdown` pools. As everywhere in
:mod:`discell.sensitivity.bootstrap`, a draw resamples the 200 um tiles of the
cells a read is graded on and is **conditional on the fitted objects** (ridge
fits, kNN balls, pairs, model predictions held at the point estimate's).

The paper (Table S1 of its supplement) reports three findings (:data:`FINDINGS`) and two
checks of the fit (:data:`CHECKS`). Two more members, comparisons of moved cells with
the type's mean profile and with a random cell (:data:`COUNTED_NOT_REPORTED`), were
tested, so they count in the Bonferroni ``M``, but they are not reported, because their
scores compare each model with its own decode. ``M = 7`` on a section with tumor
bands, 6 without. :data:`PAPER_NAMES` gives each member's name in the paper.

Members, by draw group (a group's draws share one generator,
``DRAW_SEED + 1000 * index + fit seed``):

``cycle`` (check)
    ``cycle_check``: within-type R^2 of the S / G2M cell-cycle scores from
    ``mu_z`` minus from ``mu_w``, ridge fitted on the training tiles, graded on
    the held-out top decile of the cycling score (Unassigned excluded). Needs
    the Tirosh S and G2M markers in the panel.
``w_mi`` (check)
    ``niche_information_check``: ``I(niche; mu_w)`` minus its within-type
    permutation floor on the held-out tiles (the w-channel guard).
``signaling`` (finding: the signaling-gene lean)
    ``signaling_gene_lean``: rank-biserial effect of the per-gene
    niche-response share, ligand-receptor genes against every other scored
    gene, over the relocation pairs. Needs a ligand-receptor gene list
    (*lr_genes*; the paper used the CellChatDB ligand and receptor union).
    Half-tile subsampling.
``axis`` (finding: the tumor axis)
    ``tumor_axis``: pooled ``mean |tau|`` of the predicted
    niche response along the tumor/stroma bands minus along a false axis.
    Needs a tumor-annotated type with enough cells per band.
``relocation_mean`` (finding: relocation beyond its spillover part)
    ``relocation_finding``: over every relocation
    pair, the mean of ``2 <o - s, p> / ||o||^2``, the part of
    ``R^2(response + spillover) - R^2(spillover)`` the response part earns.
    Half-tile subsampling.
``relocation_distribution``
    ``moved_vs_random_cell`` (counted, not reported): median over pairs of the
    margin of a cell's z-nearest moved cell over a random moved cell;
    ``moved_vs_type_mean`` (counted, not reported): median over pairs
    of the distribution gap closed by moving cells into the other niche (MMD^2
    against the target cells' own decodes) minus the type-mean predictor's.

The findings of a section are every member defined on it (six on a section without
tumor bands, seven with).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from discell.evaluate.guard import W_GUARD_MAX_CELLS, W_GUARD_PERMS, niche_labels
from discell.evaluate.mask import mask_record, metric_target_mask, target_types
from discell.evaluate.metrics import _subsample_rows
from discell.evaluate.mi import ross_terms, within_type_permutation
from discell.model.data import NUM_THREADS
from discell.model.equations import spillover_mix
from discell.sensitivity.bootstrap import (
    DRAW_SEED,
    N_DRAWS,
    TILE_UM,
    half_tile_weights,
    percentile_draws,
    tile_index,
)
from discell.sensitivity.breakdown import KAPPA_GRID, breakdown, read_draws, write_draws
from discell.tl.labels import is_tumor
from discell.tl.relocate import EPS, MIN_CELLS, MIN_RATE, N_NICHES, group_channels, niche_pairs

log = logging.getLogger("discell.sensitivity.findings")

#: Draw groups, in the order that fixes their draw seeds (the paper's; "marker"
#: holds a seed slot of a readout that left the findings).
SEED_ORDER = ("cycle", "w_mi", "marker", "signaling", "axis", "relocation_mean",
              "relocation_distribution")
GROUPS = ("cycle", "w_mi", "signaling", "axis", "relocation_mean", "relocation_distribution")
#: Fewer draws for the groups that recompute per-gene shifts on the GPU.
N_DRAWS_HEAVY = 4_000
HEAVY = ("signaling", "axis", "relocation_mean", "relocation_distribution")
#: The findings the paper reports (Table S1), in its order.
FINDINGS = ("relocation_finding", "signaling_gene_lean",
            "tumor_axis")
#: Checks of the fit.
CHECKS = ("cycle_check", "niche_information_check")
#: Tested, so counted in the Bonferroni M, but not reported: their scores compare each
#: model with its own decode.
COUNTED_NOT_REPORTED = ("moved_vs_type_mean", "moved_vs_random_cell")
#: Each member's name in the paper.
PAPER_NAMES = {
    "relocation_finding": "the relocation finding",
    "signaling_gene_lean": "the signaling-gene lean",
    "tumor_axis": "the tumor axis",
    "cycle_check": "cell-cycle R^2 of z minus that of w",
    "niche_information_check": "niche information in w",
    "moved_vs_type_mean": "moved cells against the type's mean profile",
    "moved_vs_random_cell": "moved cells against a random cell",
}
#: The members in the order that fixes their bootstrap choices in the breakdown table
#: (the paper's), so that a member's interval does not depend on its name.
POOL_ORDER = ("tumor_axis", "cycle_check", "moved_vs_type_mean", "moved_vs_random_cell",
              "signaling_gene_lean", "relocation_finding", "niche_information_check")
#: Cells per side after size matching in the distribution read.
SIZE_CAP = 2000
#: Paired draws of the distribution read whose random stream fixes each pair's cells.
BOOT_REPLAYED = 200
#: Tumor bands: cuts of the kNN-smoothed tumor fraction (k = 50).
TUMOR_BANDS = (0.1, 0.3, 0.5, 0.7, 0.9)
MIN_BAND_CELLS = 200
N_FALSE_BINS = len(TUMOR_BANDS) + 1
#: The cycling set: the top decile of S + G2M among held-out cells.
CYCLING_QUANTILE = 0.9

#: Seurat ``cc.genes.updated.2019`` (Tirosh et al. 2016, updated).
S_GENES = (
    "MCM5 PCNA TYMS FEN1 MCM7 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 "
    "CENPU HELLS RFC2 POLR1B NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 "
    "MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 "
    "CLSPN POLA1 CHAF1B MRPL36 E2F8").split()
G2M_GENES = (
    "HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 "
    "TMPO CENPF TACC3 PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E "
    "TUBB4B GTSE1 KIF20B HJURP CDCA3 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 "
    "NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 "
    "CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA").split()
MIN_CYCLE_GENES = 10
#: Training and held-out cycling cells a cycle read needs.
MIN_CYCLE_CELLS = 200
#: Scored genes a signaling pair needs.
MIN_PAIR_GENES = 100


def draw_seed(group: str, fit_seed: int) -> int:
    """The draw generator's seed of *group* on the fit with seed *fit_seed*."""
    return DRAW_SEED + 1000 * SEED_ORDER.index(group) + int(fit_seed)


# -- cell-cycle scores ---------------------------------------------------------

def _score_genes(x_log, var_names, gene_list, ctrl_size: int, n_bins: int = 25,
                 random_state: int = 0) -> np.ndarray:
    """Tirosh gene-set score (Scanpy's ``score_genes`` with its defaults): mean
    of the set minus the mean of control genes drawn per expression bin."""
    import pandas as pd

    rs = np.random.RandomState(random_state)
    names = pd.Index(var_names)
    gene_list = pd.Index(gene_list).intersection(names)
    obs_avg = pd.Series(np.asarray(x_log.mean(axis=0), dtype=np.float64).ravel(),
                        index=names.astype("string"))
    obs_avg = obs_avg[np.isfinite(obs_avg)]
    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items
    control = pd.Index([], dtype="string")
    for cut in np.unique(obs_cut.loc[gene_list]):
        r_genes = obs_cut[obs_cut == cut].index
        if ctrl_size < len(r_genes):
            r_genes = r_genes.to_series().sample(ctrl_size, random_state=rs).index
        control = control.union(r_genes.difference(gene_list))

    def mean(genes):
        cols = names.get_indexer(genes)
        return np.asarray(x_log[:, cols].sum(axis=1), dtype=np.float64).ravel() / len(cols)
    return mean(gene_list) - mean(control)


def cell_cycle_scores(x, gene_names: Sequence[str]) -> np.ndarray | None:
    """``(N, 2)`` S and G2M scores of raw counts *x* (Tirosh scoring after total
    normalization to the median depth and log1p); None when the panel has
    fewer than 10 markers of either list."""
    import scipy.sparse as sp

    names = [str(g) for g in gene_names]
    s_hits = [g for g in S_GENES if g in names]
    g2m_hits = [g for g in G2M_GENES if g in names]
    if len(s_hits) < MIN_CYCLE_GENES or len(g2m_hits) < MIN_CYCLE_GENES:
        return None
    x = sp.csr_matrix(x, dtype=np.float32, copy=True)
    counts = np.asarray(x.sum(axis=1)).ravel()
    target = np.median(counts[counts > 0])
    scale = np.where(counts > 0, counts / target, 1.0).astype(np.float32)
    x = sp.diags(1.0 / scale).astype(np.float32) @ x
    x.data = np.log1p(x.data)
    ctrl = min(len(s_hits), len(g2m_hits))
    return np.stack([_score_genes(x, names, s_hits, ctrl),
                     _score_genes(x, names, g2m_hits, ctrl)], axis=1)


def cycling_set(scores: np.ndarray, held_out: np.ndarray, eligible: np.ndarray,
                q: float = CYCLING_QUANTILE) -> np.ndarray:
    """The cells of the cycle read (bool): the top ``1 - q`` of the eligible
    held-out cells by S + G2M (ties by row order), plus every eligible training
    cell scoring at least the lowest of them."""
    total = np.asarray(scores, dtype=np.float64).sum(axis=1)
    candidates = np.flatnonzero(held_out & eligible)
    k = int(round((1.0 - q) * len(candidates)))
    out = np.zeros(len(total), dtype=bool)
    if k == 0:
        return out
    top = candidates[np.argsort(-total[candidates], kind="stable")[:k]]
    out[top] = True
    out |= ~held_out & eligible & (total >= total[top[-1]])
    return out


def _cycle_cells(latent, t, scores, train, test, seed, cells) -> dict:
    rng = np.random.default_rng(seed)
    rows_train = np.flatnonzero(train & cells)
    rows_test = np.flatnonzero(test & cells)
    if len(rows_train) < MIN_CYCLE_CELLS or len(rows_test) < MIN_CYCLE_CELLS:
        return {"rows_test": rows_test[:0]}
    rows_train = _subsample_rows(rows_train, rng)
    rows_test = _subsample_rows(rows_test, rng)
    latent = latent.astype(np.float64)
    target = scores.astype(np.float64)
    for g in np.unique(t[cells]):
        members = np.flatnonzero((t == g) & cells)
        latent[members] -= latent[members].mean(axis=0)
        target[members] -= target[members].mean(axis=0)
    design = np.hstack([latent[rows_train], np.ones((len(rows_train), 1))])
    coef = np.linalg.solve(design.T @ design + 1e-3 * np.eye(design.shape[1]),
                           design.T @ target[rows_train])
    held = np.hstack([latent[rows_test], np.ones((len(rows_test), 1))])
    return {"rows_test": rows_test, "residual": target[rows_test] - held @ coef,
            "target": target[rows_test]}


def _cycle_r2(cells: dict, weights: np.ndarray) -> float:
    if weights.sum() <= 0:
        return float("nan")
    w = np.repeat(weights, cells["target"].shape[1])

    def var(a):
        a = a.ravel()
        mean = np.dot(w, a) / w.sum()
        return np.dot(w, (a - mean) ** 2) / w.sum()
    return float(1.0 - var(cells["residual"]) / max(var(cells["target"]), 1e-12))


# -- one fit's shared quantities -----------------------------------------------

class _Fit:
    """The sweeps, folds and niches every group of one fit reads."""

    def __init__(self, model, adata, data, n_threads, exclude, min_cells):
        self.model, self.adata, self.config = model, adata, model.config
        self.data = data if data is not None else model.assemble(adata)
        self.runner = model.runner(self.data)
        self.device = self.runner.device
        self.exclude, self.min_cells = tuple(exclude), int(min_cells)
        self.names = [str(v) for v in self.data.type_names]
        self.train = self.runner.sweep(self.runner.train_batches)
        self.val = self.runner.sweep(self.runner.val_batches)
        d = self.data
        n = d.n_cells
        self.fold = np.zeros(n, dtype=np.int64)
        for k, tile in enumerate(d.train_tiles + d.val_tiles):
            self.fold[tile] = k % 5
        self.mu_z = np.zeros((n, self.train["mu_z"].shape[1]), dtype=np.float32)
        self.prior_w = np.zeros((n, self.train["prior_w"].shape[1]), dtype=np.float32)
        for sw in (self.train, self.val):
            self.mu_z[sw["nodes"]] = sw["mu_z"]
            self.prior_w[sw["nodes"]] = sw["prior_w"]
        self.connected = d.graph.degrees > 0
        self.niches = niche_labels(d.graph.y, d.graph.degrees, N_NICHES, self.config.seed,
                                   n_threads=n_threads)
        self.positions = np.asarray(d.positions, dtype=np.float64)
        self.b_matrix = model.network.B.weight.detach().cpu().numpy()
        self._x_rate = None

    @property
    def x_rate(self):
        if self._x_rate is None:
            d = self.data
            self._x_rate = d.x.multiply(1.0 / d.totals.clip(min=1.0)[:, None]).tocsr()
        return self._x_rate

    def group_rows(self) -> tuple[dict, dict]:
        """Model-side (folds 1-4) and scored (fold 0) rows of every (niche, type)
        group with at least ``min_cells`` and ``min_cells // 5`` cells."""
        d, labels = self.data, self.niches
        train_rows, test_rows = {}, {}
        for k in range(int(labels.max()) + 1):
            for g in range(len(self.names)):
                where = self.connected & (d.t == g) & (labels == k)
                tr = np.flatnonzero(where & (self.fold != 0))
                te = np.flatnonzero(where & (self.fold == 0))
                if len(tr) >= self.min_cells and len(te) >= self.min_cells // 5:
                    train_rows[(k, g)], test_rows[(k, g)] = tr, te
        return train_rows, test_rows

    def channels(self, context: bool = False) -> dict:
        d, labels = self.data, self.niches
        n_types = len(self.names)
        rows = self.connected & (self.fold != 0) & (labels >= 0)
        group = np.full(d.n_cells, -1, dtype=np.int64)
        group[rows] = labels[rows] * n_types + d.t[rows]
        # reuse the resident tiles: a second runner doubles the GPU memory (ovary FF)
        return group_channels(self.model, d, group, (int(labels.max()) + 1) * n_types,
                              context=context, runner=self.runner)

    def shift_parts(self, ch: dict, a: int, b: int, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(response, spillover)`` per gene of type *g* moved from niche *a* to *b*."""
        n_types, kappa = len(self.names), self.config.kappa
        ga, gb = a * n_types + g, b * n_types + g
        response = self.b_matrix @ (ch["prior_w"][gb] - ch["prior_w"][ga])
        rho_a = ch["rho"][ga]
        spill = (np.log((1 - kappa) * rho_a + kappa * ch["rho_bar"][gb] + EPS)
                 - np.log((1 - kappa) * rho_a + kappa * ch["rho_bar"][ga] + EPS))
        return response, spill

    def mean_rate(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.x_rate[rows].mean(axis=0)).ravel()


# -- the groups ----------------------------------------------------------------

def _group_cycle(fit: _Fit, n: int, seed: int) -> dict:
    d = fit.data
    scores = cell_cycle_scores(d.x, d.gene_names)
    if scores is None:
        return {}
    held = np.zeros(d.n_cells, dtype=bool)
    for tile in d.val_tiles:
        held[tile] = True
    q90 = cycling_set(scores, held, metric_target_mask(d.t, fit.names, fit.exclude))
    train, val = fit.train, fit.val
    rows = np.concatenate([train["nodes"], val["nodes"]])
    tr = np.arange(len(rows)) < len(train["nodes"])
    t_rows = d.t[rows]
    cz = _cycle_cells(np.vstack([train["mu_z"], val["mu_z"]]), t_rows, scores[rows], tr, ~tr,
                      fit.config.seed, q90[rows])
    cw = _cycle_cells(np.vstack([train["mu_w"], val["mu_w"]]), t_rows, scores[rows], tr, ~tr,
                      fit.config.seed, q90[rows])
    if not len(cz["rows_test"]):
        return {}
    point, draws = percentile_draws(fit.positions[rows[cz["rows_test"]]],
                                    lambda w: _cycle_r2(cz, w) - _cycle_r2(cw, w), n, seed)
    return {"cycle_check": _member(point, draws)}


def _group_w_mi(fit: _Fit, n: int, seed: int) -> dict:
    d, val = fit.data, fit.val
    on = metric_target_mask(d.t[val["nodes"]], fit.names, fit.exclude)
    vrows = val["nodes"][on]
    w = np.asarray(val["mu_w"][on], dtype=np.float64)
    niche, t = fit.niches[vrows], d.t[vrows]
    rng = np.random.default_rng(fit.config.seed)
    rows = np.flatnonzero(niche >= 0)
    if len(rows) > W_GUARD_MAX_CELLS:
        rows = np.sort(rng.choice(rows, W_GUARD_MAX_CELLS, replace=False))
    y, t_rows, w_rows = niche[rows], t[rows], w[rows]
    real = ross_terms(w_rows, y, rng=rng)
    floors = [ross_terms(w_rows, within_type_permutation(y, t_rows, rng), rng=rng)
              for _ in range(W_GUARD_PERMS)]

    def mi(terms, weights):
        ok = np.isfinite(terms)
        total = weights[ok].sum()
        return (max(float(np.dot(weights[ok], terms[ok]) / total), 0.0) if total > 0
                else float("nan"))

    def stat(weights):
        return mi(real, weights) - float(np.mean([mi(f, weights) for f in floors]))
    point, draws = percentile_draws(fit.positions[vrows[rows]], stat, n, seed)
    return {"niche_information_check": _member(point, draws)}


def _subsample_means(fit: _Fit, pair_rows: list[tuple[np.ndarray, np.ndarray]],
                     keeps: list[np.ndarray], n: int, seed: int, with_point: bool,
                     chunk: int = 250):
    """Per pair, yield ``(lo, hi, log shift)`` blocks: the observed log-rate
    shift ``log mean_B - log mean_A`` on the kept genes under half-tile
    subsamples of every scored cell any pair uses (with *with_point*, row 0 is
    the unit-weight point). Also returns the subsample weights' scale ``c``."""
    import torch

    cells = np.unique(np.concatenate([np.concatenate(r) for r in pair_rows]))
    pos = np.full(fit.data.n_cells, -1, dtype=np.int64)
    pos[cells] = np.arange(len(cells))
    weights, c = half_tile_weights(fit.positions[cells], n, seed)
    dev = fit.device
    total = n + int(with_point)

    def blocks(j):
        rows, keep = pair_rows[j], keeps[j]
        xs = [torch.as_tensor(np.asarray(fit.x_rate[r][:, keep].todense()),
                              dtype=torch.float32, device=dev) for r in rows]
        for lo in range(0, total, chunk):
            hi = min(lo + chunk, total)
            means = []
            for x, r in zip(xs, rows):
                if with_point:      # row 0 = unit weights, row i = subsample i - 1
                    w = weights[max(lo - 1, 0):hi - 1, pos[r]]
                    if lo == 0:
                        w = np.vstack([np.ones((1, len(r)), np.float32), w])
                else:
                    w = weights[lo:hi, pos[r]]
                w = torch.as_tensor(w, device=dev)
                means.append(((w @ x) / w.sum(dim=1, keepdim=True).clamp(min=1e-12)).double())
            yield lo, hi, torch.log(means[1] + EPS) - torch.log(means[0] + EPS)

    def empty(j):
        e = ((weights[:, pos[pair_rows[j][0]]].sum(1) == 0)
             | (weights[:, pos[pair_rows[j][1]]].sum(1) == 0))
        return np.concatenate([[False], e]) if with_point else e
    return blocks, empty, c


def _group_signaling(fit: _Fit, lr_genes: Iterable[str], n: int, seed: int) -> dict:
    import torch
    from scipy.stats import mannwhitneyu

    d = fit.data
    ch = fit.channels()
    train_rows, test_rows = fit.group_rows()
    keep_type = target_types(fit.names, fit.exclude)
    built = []
    for a, b in niche_pairs(fit.niches, d.graph.y, fit.connected):
        for g in range(len(fit.names)):
            if not keep_type[g] or (a, g) not in train_rows or (b, g) not in train_rows:
                continue
            ra, rb = test_rows[(a, g)], test_rows[(b, g)]
            obs_a, obs_b = fit.mean_rate(ra), fit.mean_rate(rb)
            keep = np.flatnonzero((obs_a > MIN_RATE) & (obs_b > MIN_RATE))
            if len(keep) < MIN_PAIR_GENES:
                continue
            response, spill = fit.shift_parts(ch, a, b, g)
            built.append({"keep": keep, "response": response[keep], "spillover": spill[keep],
                          "rows": (ra, rb),
                          "observed": np.log(obs_b[keep] + EPS) - np.log(obs_a[keep] + EPS)})
    if not built:
        return {}
    n_genes = len(d.gene_names)
    acc = np.zeros((3, n_genes))
    hits = np.zeros(n_genes)
    for p in built:
        resp, spillover, obs = (v - v.mean()
                                for v in (p["response"], p["spillover"], p["observed"]))
        acc[0, p["keep"]] += np.abs(resp)
        acc[1, p["keep"]] += np.abs(spillover)
        acc[2, p["keep"]] += np.abs(obs - resp - spillover)
        hits[p["keep"]] += 1
    total = acc.sum(axis=0)
    scored = (hits > 0) & (total > 1e-12)
    share = np.full(n_genes, np.nan)
    share[scored] = acc[0, scored] / total[scored]
    genes = np.asarray([str(g) for g in d.gene_names])
    is_lr = np.isin(genes, sorted(set(lr_genes) & set(genes.tolist())))
    lr_v, other_v = share[scored & is_lr], share[scored & ~is_lr]
    if len(lr_v) < 3 or len(other_v) < 3:
        return {}
    u, _ = mannwhitneyu(lr_v, other_v, alternative="two-sided")
    point = 2.0 * float(u) / (len(lr_v) * len(other_v)) - 1.0

    blocks, _, c = _subsample_means(fit, [p["rows"] for p in built],
                                    [p["keep"] for p in built], n, seed, with_point=False)
    dev = fit.device
    draw_acc = torch.zeros((3, n, n_genes), dtype=torch.float64, device=dev)
    for j, p in enumerate(built):
        keep = torch.as_tensor(p["keep"], device=dev)
        resp = torch.as_tensor(p["response"] - p["response"].mean(), device=dev)
        spillover = torch.as_tensor(p["spillover"] - p["spillover"].mean(), device=dev)
        for lo, hi, obs in blocks(j):
            obs = obs - obs.mean(dim=1, keepdim=True)
            valid = torch.isfinite(obs).all(dim=1)
            obs = torch.where(valid[:, None], obs, torch.zeros_like(obs))
            vmask = valid[:, None].double()
            draw_acc[0, lo:hi][:, keep] += vmask * resp.abs()[None, :]
            draw_acc[1, lo:hi][:, keep] += vmask * spillover.abs()[None, :]
            draw_acc[2, lo:hi][:, keep] += vmask * (obs - resp - spillover).abs()
    sc = torch.as_tensor(scored, device=dev)
    shares = draw_acc[0][:, sc] / draw_acc.sum(dim=0)[:, sc].clamp(min=1e-12)
    lr = torch.as_tensor(is_lr[scored], device=dev)
    ranks = torch.argsort(torch.argsort(shares, dim=1), dim=1).double() + 1.0
    n1, n2 = float(lr.sum()), float((~lr).sum())
    draws = (2.0 * (ranks[:, lr].sum(dim=1) - n1 * (n1 + 1) / 2.0) / (n1 * n2) - 1.0)
    return {"signaling_gene_lean":
            _member(point, draws.cpu().numpy(), "subsample", c)}


def _group_relocation_mean(fit: _Fit, n: int, seed: int) -> dict:
    import torch

    d = fit.data
    ch = fit.channels()
    train_rows, test_rows = fit.group_rows()
    keep_type = target_types(fit.names, fit.exclude)
    pairs = []
    for a, b in niche_pairs(fit.niches, d.graph.y, fit.connected):
        for g in range(len(fit.names)):
            if not keep_type[g] or (a, g) not in train_rows or (b, g) not in train_rows:
                continue
            ra, rb = test_rows[(a, g)], test_rows[(b, g)]
            keep = (fit.mean_rate(ra) > MIN_RATE) & (fit.mean_rate(rb) > MIN_RATE)
            response, spill = fit.shift_parts(ch, a, b, g)
            pairs.append({"rows": (ra, rb), "keep": keep, "program": response[keep],
                           "spillover": spill[keep]})
    if not pairs:
        return {}
    blocks, empty, c = _subsample_means(fit, [p["rows"] for p in pairs],
                                        [p["keep"] for p in pairs], n, seed, with_point=True)
    dev = fit.device
    align = np.full((n + 1, len(pairs)), np.nan)
    for j, p in enumerate(pairs):
        preds = {k: torch.as_tensor(v - v.mean(), device=dev)
                 for k, v in (("program", p["program"]), ("spillover", p["spillover"]))}
        for lo, hi, obs in blocks(j):
            oc = obs - obs.mean(dim=1, keepdim=True)
            den = (oc ** 2).sum(dim=1).clamp(min=1e-12)
            align[lo:hi, j] = (2.0 * ((oc - preds["spillover"][None, :])
                                      * preds["program"][None, :])
                               .sum(dim=1) / den).cpu().numpy()
        align[empty(j), j] = np.nan
    values = np.nanmean(align, axis=1)
    return {"relocation_finding": _member(values[0], values[1:], "subsample", c)}


def _gap(u, t, floor):
    u, t = np.asarray(u, dtype=np.float64), np.asarray(t, dtype=np.float64)
    denom = u - floor
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((u - t) / np.where(np.abs(denom) > 1e-12, denom, np.nan), -1.0, 1.0)


def _hellinger(p):
    return np.sqrt(np.clip(np.asarray(p, dtype=np.float64), 0.0, None))


def _hellinger_rows(a, b):
    return np.linalg.norm(_hellinger(a) - _hellinger(b), axis=1) / np.sqrt(2.0)


def _weighted_mmd_parts(kxx, row_kxy, weights):
    """``sxx - 2 sxy`` of the unbiased MMD^2 with the source side weighted:
    ``sxx`` pairs distinct cells only, so a cell drawn twice is not paired with
    its own copy."""
    import torch

    wm = weights.sum(dim=1)
    w2 = (weights ** 2).sum(dim=1)
    quad = ((weights @ kxx) * weights).sum(dim=1)
    diag = (weights ** 2) @ kxx.diagonal()
    sxx = (quad - diag) / (wm ** 2 - w2).clamp(min=1e-12)
    sxy = (weights @ row_kxy) / wm.clamp(min=1e-12)
    part = sxx - 2.0 * sxy
    return torch.where((wm ** 2 - w2) > 0, part, torch.full_like(part, float("nan")))


def _mmd2(kxx, kyy, kxy) -> float:
    m, n = kxx.shape[0], kyy.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    sxx = (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
    syy = (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
    return float(sxx + syy - 2.0 * kxy.mean())


def _mmd_draws(p_trans, p_unt, p_target, rows, rng, device, weights_of, chunk=250) -> dict:
    """MMD^2 of the moved, unmoved and type-mean decodes against
    the target decodes (size-matched subsample, median-heuristic Gaussian
    kernel on the Hellinger map, split-half floor), and the first two under
    every tile draw of the source cells."""
    import torch

    n_s, n_t = len(p_trans), len(p_target)
    m = int(min(n_s, n_t, SIZE_CAP))
    if m < 20:
        return {"insufficient": True}
    idx_s = rng.choice(n_s, m, replace=False)
    idx_t = rng.choice(n_t, m, replace=False)

    def to_t(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
    target = np.asarray(p_target)[idx_t]
    y = to_t(_hellinger(target))
    xs = {"moved": to_t(_hellinger(np.asarray(p_trans)[idx_s])),
          "unmoved": to_t(_hellinger(np.asarray(p_unt)[idx_s])),
          "type_mean": to_t(_hellinger(np.tile(target.mean(axis=0), (m, 1))))}
    weights = torch.as_tensor(weights_of(np.asarray(rows)[idx_s]), dtype=torch.float64,
                              device=device)
    out = {"insufficient": False, "mmd2": {}, "draws": {}}

    def kernel(a, b, gamma):
        return torch.exp(-gamma * torch.cdist(a, b).pow(2))
    with torch.no_grad():
        d_yy = torch.cdist(y, y)
        sigma = float(d_yy[~torch.eye(m, dtype=torch.bool, device=device)].median())
        gamma = 1.0 / (2.0 * max(sigma, 1e-8) ** 2)
        kyy = torch.exp(-gamma * d_yy.pow(2))
        syy = float((kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)))
        for name, x in xs.items():
            kxx, kxy = kernel(x, x, gamma), kernel(x, y, gamma)
            out["mmd2"][name] = _mmd2(kxx, kyy, kxy)
            if name != "type_mean":
                kd, rd = kxx.double(), kxy.double().mean(dim=1)
                out["draws"][name] = torch.cat(
                    [_weighted_mmd_parts(kd, rd, weights[i:i + chunk])
                     for i in range(0, len(weights), chunk)]).cpu().numpy() + syy
        half = m // 2
        perm = torch.as_tensor(rng.permutation(m), device=device)
        y1, y2 = y[perm[:half]], y[perm[half:2 * half]]
        out["mmd2"]["floor"] = _mmd2(kernel(y1, y1, gamma), kernel(y2, y2, gamma),
                                     kernel(y1, y2, gamma))
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Median of *values* repeated by each row of integer *weights*; NaN for an
    empty draw."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    cum = np.cumsum(np.rint(weights[:, order]).astype(np.int64), axis=1)
    total = cum[:, -1]
    out = np.full(len(weights), np.nan)
    for i in np.flatnonzero(total > 0):
        lo = np.searchsorted(cum[i], (total[i] - 1) // 2, side="right")
        hi = np.searchsorted(cum[i], total[i] // 2, side="right")
        out[i] = 0.5 * (v[lo] + v[hi])
    return out


def _group_relocation_distribution(fit: _Fit, n: int, seed: int) -> dict:
    import torch


    d, config, net = fit.data, fit.config, fit.model.network
    dev = fit.device
    n_types = len(fit.names)
    ch = fit.channels(context=True)
    train_rows, test_rows = fit.group_rows()
    kappa = config.kappa

    with torch.no_grad():
        def niche_w(k, g):
            onehot = torch.zeros(1, n_types, device=dev)
            onehot[0, g] = 1.0
            c = torch.as_tensor(ch["c"][k * n_types + g][None, :], dtype=torch.float32,
                                device=dev)
            return net.prior_w(torch.cat([c, onehot], dim=-1)).cpu().numpy()[0]
        w_of = {(k, g): (niche_w(k, g), ch["rho_bar"][k * n_types + g]) for (k, g) in train_rows}

    def decode(mu_z, w, rho_bar, chunk=4096):
        out = []
        with torch.no_grad():
            for s in range(0, len(mu_z), chunk):
                z = torch.as_tensor(mu_z[s:s + chunk], dtype=torch.float32, device=dev)
                ww = torch.as_tensor(w[s:s + chunk], dtype=torch.float32, device=dev)
                bar = torch.as_tensor(rho_bar[s:s + chunk], dtype=torch.float32, device=dev)
                out.append(spillover_mix(net.log_rho(z, ww).exp(), bar, kappa).exp().cpu().numpy())
        return np.vstack(out)

    # each pair's source and target cells come from the read's own random
    # stream, advanced as the full distribution read advances it
    rng = np.random.default_rng(config.seed)

    def advance(n_s, n_t):
        m = int(min(n_s, n_t, SIZE_CAP))
        if m < 20:
            return
        rng.choice(n_s, m, replace=False)
        rng.choice(n_t, m, replace=False)
        rng.permutation(m)
        for _ in range(BOOT_REPLAYED):
            rng.integers(0, m, m)
            rng.integers(0, m, m)

    keep_type = target_types(fit.names, fit.exclude)
    raw = []
    for a, b in niche_pairs(fit.niches, d.graph.y, fit.connected):
        for g in range(n_types):
            if (a, g) not in train_rows or (b, g) not in train_rows:
                continue
            src = test_rows[(a, g)]
            rows = src[rng.permutation(len(src))[:SIZE_CAP]]
            tgt = test_rows[(b, g)]
            tgt = tgt[rng.permutation(len(tgt))[:SIZE_CAP]]
            advance(len(rows), len(tgt))
            raw.append({"g": g, "a": a, "b": b, "rows": rows, "tgt": tgt})
    raw = [r for r in raw if keep_type[r["g"]]]
    if not raw:
        return {}
    universe = np.unique(np.concatenate([r["rows"] for r in raw]))
    pos = np.full(d.n_cells, -1, dtype=np.int64)
    pos[universe] = np.arange(len(universe))
    tiles, n_tiles = tile_index(fit.positions[universe])
    draw_rng = np.random.default_rng(seed)
    mults = np.stack([np.bincount(draw_rng.integers(0, n_tiles, n_tiles),
                                  minlength=n_tiles)[tiles] for _ in range(n)]).astype(np.int32)

    def weights_of(cells):
        return mults[:, pos[cells]]

    # the target cells as the model decodes them (own z, w, context and influx)
    wanted = np.unique(np.concatenate([r["tgt"] for r in raw]))
    where = np.full(d.n_cells, -1, dtype=np.int64)
    where[wanted] = np.arange(len(wanted))
    own_p = None
    runner = fit.runner
    with torch.no_grad():
        for batch in runner.train_batches + runner.val_batches:
            k = batch["n_seeds"]
            take = np.flatnonzero(where[batch["nodes"][:k]] >= 0)
            if not len(take):
                continue
            fwd = net(**runner._forward_kwargs(batch), kappa=kappa, sample=False)
            p = fwd.log_p[:k].exp()[torch.as_tensor(take, device=dev)].float().cpu().numpy()
            if own_p is None:
                own_p = np.zeros((len(wanted), p.shape[1]), dtype=np.float32)
            own_p[where[batch["nodes"][:k][take]]] = p

    pt, pm, dt, dm, twins = [], [], [], [], []
    for r in raw:
        rows, tgt, g = r["rows"], r["tgt"], r["g"]
        w_b, bar_b = w_of[(r["b"], g)]
        w_a, bar_a = w_of[(r["a"], g)]
        z = fit.mu_z[rows]
        p_trans = decode(z, np.tile(w_b, (len(rows), 1)), np.tile(bar_b, (len(rows), 1)))
        p_unt = decode(z, np.tile(w_a, (len(rows), 1)), np.tile(bar_a, (len(rows), 1)))
        p_tgt = own_p[where[tgt]].astype(np.float64)
        s = _mmd_draws(p_trans, p_unt, p_tgt, rows, np.random.default_rng(config.seed + 101),
                       dev, weights_of)
        if not s["insufficient"]:
            mm, u = s["mmd2"], s["draws"]["unmoved"]
            pt.append(float(_gap(mm["unmoved"], mm["moved"], mm["floor"])))
            pm.append(float(_gap(mm["unmoved"], mm["type_mean"], mm["floor"])))
            dt.append(_gap(u, s["draws"]["moved"], mm["floor"]))
            dm.append(_gap(u, np.full(len(u), mm["type_mean"]), mm["floor"]))
        if len(tgt) >= 20 and len(rows) >= 20:
            zt = fit.mu_z[tgt].astype(np.float64)
            dist = np.linalg.norm(zt[:, None, :] - z.astype(np.float64)[None, :, :], axis=-1)
            twin = np.argmin(dist, axis=1)
            rand = np.random.default_rng(config.seed + 103).integers(0, len(rows), len(tgt))
            twins.append({"tgt": tgt, "d_tr": _hellinger_rows(p_trans[twin], p_tgt),
                          "d_rd": _hellinger_rows(p_trans[rand], p_tgt)})
    out = {}
    if pt:
        point = float(np.nanmedian(pt) - np.nanmedian(pm))
        draws = np.nanmedian(np.stack(dt, 1), 1) - np.nanmedian(np.stack(dm, 1), 1)
        out["moved_vs_type_mean"] = _member(point, draws)
    if twins:
        tgt = np.unique(np.concatenate([t["tgt"] for t in twins]))
        tiles_t, n_tiles_t = tile_index(fit.positions[tgt])
        where_t = np.full(d.n_cells, -1, dtype=np.int64)
        where_t[tgt] = np.arange(len(tgt))
        rng_t = np.random.default_rng(seed + 7)
        mults_t = np.stack([np.bincount(rng_t.integers(0, n_tiles_t, n_tiles_t),
                                        minlength=n_tiles_t)[tiles_t] for _ in range(n)])
        pts, drs = [], []
        for t in twins:
            med_tr, med_rd = np.median(t["d_tr"]), np.median(t["d_rd"])
            pts.append((med_rd - med_tr) / med_rd if med_rd > 1e-12 else np.nan)
            w = mults_t[:, where_t[t["tgt"]]]
            wt, wr = _weighted_median(t["d_tr"], w), _weighted_median(t["d_rd"], w)
            with np.errstate(invalid="ignore", divide="ignore"):
                drs.append(np.where(w.sum(1) > 0,
                                    (wr - wt) / np.where(wr > 1e-12, wr, np.nan), np.nan))
        out["moved_vs_random_cell"] = _member(float(np.nanmedian(pts)),
                                           np.nanmedian(np.stack(drs, 1), 1))
    return out


def tumor_bands(t: np.ndarray, type_names: Sequence[str], positions: np.ndarray,
                 degrees: np.ndarray, k: int = 50) -> np.ndarray | None:
    """Bands of the kNN-smoothed tumor fraction (cuts :data:`TUMOR_BANDS`), -1
    for isolated cells; None when no type is tumor-annotated."""
    from scipy.spatial import cKDTree


    tumor = [g for g, name in enumerate(type_names) if is_tumor(str(name))]
    if not tumor:
        return None
    mask = np.isin(t, tumor).astype(np.float64)
    k = min(k, len(t) - 1)
    neighbors = cKDTree(positions).query(positions, k=k + 1)[1]
    labels = np.digitize(mask[neighbors].mean(axis=1), TUMOR_BANDS).astype(np.int64)
    labels[degrees == 0] = -1
    return labels


def _kendall_tau_rows(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Kendall tau-b of every column of *values* against the strict ranking *order*."""
    values = np.asarray(values, float)
    num = np.zeros(values.shape[1])
    ties = np.zeros(values.shape[1])
    n_pairs = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            s = np.sign(order[j] - order[i])
            if s == 0:
                continue
            n_pairs += 1
            diff = values[j] - values[i]
            num += s * np.sign(diff)
            ties += diff == 0
    denom = np.sqrt(np.maximum(n_pairs - ties, 0) * n_pairs)
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)


def _group_axis(fit: _Fit, n: int, seed: int) -> dict:
    from scipy.stats import spearmanr

    d = fit.data
    bands = tumor_bands(d.t, fit.names, fit.positions, d.graph.degrees)
    if bands is None:
        return {}
    held_out = fit.fold == 0
    n_bands = int(bands.max()) + 1
    have = fit.connected & (bands >= 0)
    rho_x = abs(float(spearmanr(fit.positions[have, 0], bands[have])[0]))
    rho_y = abs(float(spearmanr(fit.positions[have, 1], bands[have])[0]))
    false_field = fit.positions[:, 0 if rho_x <= rho_y else 1]
    keep_type = target_types(fit.names, fit.exclude)
    specs = []
    for g in np.flatnonzero(keep_type):
        members = fit.connected & (d.t == g)
        if members.sum() < MIN_BAND_CELLS * n_bands:
            continue
        fb = np.full(len(bands), -1, dtype=np.int64)
        v = false_field[members]
        fb[members] = np.digitize(v, np.quantile(v, np.linspace(0, 1, N_FALSE_BINS + 1))[1:-1])
        per = {}
        for axis, lab in (("true", bands), ("false", fb)):
            keep_band, band_rows, raw = [], [], []
            for b in range(int(lab[members].max()) + 1):
                model_rows = np.flatnonzero(members & ~held_out & (lab == b))
                te = np.flatnonzero(members & held_out & (lab == b))
                if len(model_rows) < fit.min_cells or len(te) < MIN_BAND_CELLS:
                    continue
                keep_band.append(b)
                band_rows.append(model_rows)
                raw.append(np.log(fit.mean_rate(te) + EPS))
            if len(keep_band) < 4:
                per = {}
                break
            per[axis] = {"order": np.array(keep_band, float), "rows": band_rows,
                         "genes": np.all(np.exp(np.array(raw)) > MIN_RATE, axis=0)}
        if "true" in per and "false" in per:
            common = per["true"]["genes"] & per["false"]["genes"]
            per["true"]["genes"] = per["false"]["genes"] = common
            specs.append(per)
    if not specs:
        return {}
    universe = np.unique(np.concatenate([r for s in specs for a in s.values() for r in a["rows"]]))
    pos = np.full(d.n_cells, -1, dtype=np.int64)
    pos[universe] = np.arange(len(universe))
    prior = fit.prior_w.astype(np.float64)

    def stat(weights):
        tt_all, tf_all = [], []
        for s in specs:
            taus = {}
            for axis in ("true", "false"):
                a = s[axis]
                prog = []
                for r in a["rows"]:
                    w = weights[pos[r]]
                    if w.sum() <= 0:
                        return float("nan")
                    prog.append(fit.b_matrix @ (w @ prior[r] / w.sum()))
                taus[axis] = _kendall_tau_rows(np.array(prog)[:, a["genes"]], a["order"])
            m = min(len(taus["true"]), len(taus["false"]))
            tt_all.append(taus["true"][:m])
            tf_all.append(taus["false"][:m])
        return float(np.abs(np.concatenate(tt_all)).mean()
                     - np.abs(np.concatenate(tf_all)).mean())
    point, draws = percentile_draws(fit.positions[universe], stat, n, seed)
    return {"tumor_axis": _member(point, draws)}


def _member(estimate, draws, method: str = "percentile", c: float = 1.0) -> dict:
    return {"estimate": float(estimate), "draws": np.asarray(draws, dtype=np.float64),
            "method": method, "c": float(c)}


# -- public --------------------------------------------------------------------

def fit_draws(model, adata, *, groups: Sequence[str] = GROUPS,
              lr_genes: Iterable[str] | None = None, n: int | None = None,
              out_dir: str | Path | None = None, data=None,
              n_threads: int = NUM_THREADS, min_cells: int = MIN_CELLS) -> dict:
    """Every member's draws on one fit: ``{group: {member: {"estimate", "draws",
    "method", "c"}}}`` (a group none of whose members is defined maps to ``{}``).

    *model* is a fitted :class:`discell.Model`; *adata* the section it is read
    on (the fitted section, or another with the same types and genes). Draws
    run on the model's device. *lr_genes*: the ligand-receptor genes of the
    signaling member (skipped when None). *n*: draws per group (default
    10,000, or 4,000 for :data:`HEAVY`). *out_dir*: each group is written to
    ``<out_dir>/<group>.npz`` (:func:`discell.sensitivity.write_draws`), and a
    group already there is read instead of recomputed. *n_threads*: of the
    niche k-means. *min_cells*: model-side cells per niche and type of a
    relocation pair. The draws run with single-threaded BLAS.
    """
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown groups {sorted(unknown)}; choose from {GROUPS}")
    exclude = (model.config.unassigned_label,)
    fit = None
    out: dict = {}
    for group in groups:
        path = None if out_dir is None else Path(out_dir) / f"{group}.npz"
        if path is not None and path.exists():
            out[group] = read_draws(path)
            continue
        if group == "signaling" and lr_genes is None:
            log.info("signaling member skipped: no ligand-receptor genes given")
            continue
        if fit is None:
            fit = _Fit(model, adata, data, n_threads, exclude, min_cells)
        n_g = n or (N_DRAWS_HEAVY if group in HEAVY else N_DRAWS)
        seed = draw_seed(group, model.config.seed)
        # thousands of small vector reductions: threaded BLAS only adds overhead
        with threadpool_limits(limits=1, user_api="blas"):
            if group == "cycle":
                members = _group_cycle(fit, n_g, seed)
            elif group == "w_mi":
                members = _group_w_mi(fit, n_g, seed)
            elif group == "signaling":
                members = _group_signaling(fit, lr_genes, n_g, seed)
            elif group == "axis":
                members = _group_axis(fit, n_g, seed)
            elif group == "relocation_mean":
                members = _group_relocation_mean(fit, n_g, seed)
            else:
                members = _group_relocation_distribution(fit, n_g, seed)
        if not members:
            log.info("%s: no member defined on this fit", group)
        out[group] = members
        if path is not None:
            meta = {"group": group, "kappa": float(model.config.kappa),
                    "seed": int(model.config.seed), "n_draws": n_g, "draw_seed": seed,
                    "tile_um": TILE_UM, "n_threads": int(n_threads),
                    "min_cells": int(min_cells),
                    "eval_mask": mask_record(fit.names, fit.data.t, exclude)}
            if members:
                write_draws(path, members, meta)
            else:
                _write_empty(path, meta)
    return out


def _write_empty(path: Path, meta: dict) -> None:
    """A group without members on this fit, recorded so a rerun skips it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, members=np.array([], dtype=str), estimate=np.zeros(0), draws=np.zeros((0, 0)),
             method=np.array([], dtype=str), c=np.zeros(0), meta=np.array(json.dumps(meta)))


def finding_breakdown(draws_dirs, *, members: Sequence[str] | None = None,
                     grid: Sequence[float] = KAPPA_GRID, seeds: Sequence[int] = (0, 1, 2),
                     alpha: float = 0.05) -> dict:
    """kappa\\* of every finding from the draw files of a sweep.

    *draws_dirs* maps ``(kappa, seed)`` to the directory :func:`fit_draws`
    wrote. *members*: the findings (default: every member found on the kappa = 0
    fits); their number is the Bonferroni ``M``. Each member of the result also
    carries its ``role`` (``"finding"``, ``"check"`` or ``"counted, not reported"``;
    None for a member outside the paper's findings) and its ``paper_name``.
    """
    per: dict = {}
    for (kappa, seed), where in draws_dirs.items():
        found = {}
        for path in sorted(Path(where).glob("*.npz")):
            found.update(read_draws(path))
        for name, entry in found.items():
            per.setdefault(name, {}).setdefault(float(kappa), {})[int(seed)] = entry
    if members is None:
        at0 = {name for name, by_k in per.items() if float(grid[0]) in by_k}
        members = sorted(at0)
    draws = {name: per.get(name, {}) for name in members}
    order = [m for m in POOL_ORDER if m in draws] + sorted(set(draws) - set(POOL_ORDER))
    table = breakdown(draws, grid=[float(k) for k in grid], seeds=list(seeds),
                      n_findings=len(members), alpha=alpha, order=order)
    for name, member in table["members"].items():
        member["role"] = role(name)
        member["paper_name"] = PAPER_NAMES.get(name)
    return table


def role(member: str) -> str | None:
    """``"finding"``, ``"check"`` or ``"counted, not reported"`` for a member of
    the paper's findings, None otherwise."""
    if member in FINDINGS:
        return "finding"
    if member in CHECKS:
        return "check"
    if member in COUNTED_NOT_REPORTED:
        return "counted, not reported"
    return None


__all__ = ["CHECKS", "COUNTED_NOT_REPORTED", "FINDINGS", "GROUPS",
           "PAPER_NAMES", "cell_cycle_scores", "cycling_set", "draw_seed", "finding_breakdown",
           "fit_draws", "role", "tumor_bands"]
