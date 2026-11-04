"""Relocation, the mean read: does the model predict how a type changes between niches?

Niches are k-means clusters of the neighbor composition ``y`` (data, never the
latents). A *niche pair* is one cell type and two niches (A, B). The model's
quantities are group means over the training folds (folds 1 to 4 of the fit's
spatial tiles), the observed shift is read on fold 0, per gene, on the
log scale:

* ``observed``: ``log`` mean observed share in B minus in A;
* ``response``: ``B_g . (mean m_psi in B - mean m_psi in A)``, the niche
  response of the same cells moved to B (``z`` fixed);
* ``spillover``: the spillover mixture with B's foreign influx minus with A's
  (``(1 - kappa) rho_A + kappa rho_bar``), the cell's own clean profile fixed;
* ``predicted``: the two together, decoded (:func:`decoded_shift`), A's clean
  profile with B's response and B's influx against A. Spillover mixes in on the
  rate scale, so this is not the sum of the two parts; ``full`` is the model's
  account of the two actual populations (it also lets the type's own mix differ);

Each prediction is scored by :func:`score_shift` against the observed shift on
the genes above ``MIN_RATE`` in both niches. A pair's ``noise_ceiling`` is the
split-half reliability of the observed shift (Spearman-Brown), the largest R²
any predictor can reach; ``ceiling_tiles`` splits by tile instead of by cell.
A pair is *trusted* when its ceiling is at least 0.5 on at least 100 genes.
``overlap_flag`` marks extrapolation (the niches share no composition support
for the type along their gap); with k-means niches nearly every pair is
extrapolation. ``alignment`` is ``2 <o - s, p> / ||o||^2`` (centered vectors):
the part of ``R²(response + spillover) - R²(spillover)`` that the response
earns, without the never-positive penalty ``||p||^2 / ||o||^2``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from discell.evaluate.guard import niche_labels
from discell.evaluate.mask import EXCLUDED_TYPES, mask_record, target_types
from discell.model.data import NUM_THREADS
from discell.tl import _utils as U

N_NICHES = 10
MIN_CELLS = 500          #: model-side cells per niche and type (scored side: a fifth)
MIN_RATE = 1e-5          #: genes below this mean rate in either niche are not scored
N_SPLITS = 4             #: split-halves averaged for the noise ceiling
TRUST_CEILING = 0.5
TRUST_GENES = 100
TOP_GENES = 50
EPS = 1e-8
#: below this ceiling a ratio to it is not reported
MIN_CEILING = 0.05
PREDICTIONS = ("predicted", "full", "response", "spillover")


def score_shift(prediction: np.ndarray, observed: np.ndarray) -> dict:
    """R² (against the zero prediction, no fitted slope), calibration slope and
    correlation of a predicted per-gene shift. Both sides are centered: depth and
    the softmax normalizer enter as per-pair constants."""
    p = prediction - prediction.mean()
    o = observed - observed.mean()
    slope = float(np.polyfit(p, o, 1)[0]) if p.std() > 1e-9 else float("nan")
    ss = 1.0 - ((o - p) ** 2).sum() / max((o ** 2).sum(), 1e-12)
    return {"r2": float(ss), "slope": slope, "corr": float(np.corrcoef(p, o)[0, 1])}


def alignment(observed: np.ndarray, response: np.ndarray, spillover: np.ndarray) -> float:
    """``2 <o - s, p> / ||o||^2`` on centered vectors (module doc)."""
    o, p, s = (v - v.mean() for v in (observed, response, spillover))
    return float(2.0 * ((o - s) * p).sum() / max((o ** 2).sum(), 1e-12))


def top_gene_overlap(prediction: np.ndarray, observed: np.ndarray, k: int = TOP_GENES) -> dict:
    """Of the *k* genes predicted to rise most, how many are in the observed top
    *k* (both centered); ``chance = k / G`` always beside it."""
    k = int(min(k, len(observed)))
    if k == 0:
        return {"k": 0, "overlap": 0, "fraction": float("nan"), "chance": float("nan")}
    p = prediction - prediction.mean()
    o = observed - observed.mean()
    both = set(np.argsort(p)[-k:].tolist()) & set(np.argsort(o)[-k:].tolist())
    return {"k": k, "overlap": len(both), "fraction": len(both) / k,
            "chance": k / len(observed)}


def _tile_halves(rows_a, rows_b, tile_of, rng, tries: int = 100):
    tiles = np.unique(tile_of[np.concatenate([rows_a, rows_b])])
    if min(len(np.unique(tile_of[rows])) for rows in (rows_a, rows_b)) < 2:
        return None
    for _ in range(tries):
        first = rng.permutation(tiles)[:len(tiles) // 2]
        masks = [np.isin(tile_of[rows], first) for rows in (rows_a, rows_b)]
        if all(m.any() and not m.all() for m in masks):
            return [[rows[m], rows[~m]] for rows, m in zip((rows_a, rows_b), masks)]
    return None


def noise_ceiling(x_rate, rows_a: np.ndarray, rows_b: np.ndarray, keep: np.ndarray,
                  rng: np.random.Generator, n_splits: int = N_SPLITS, split: str = "cells",
                  tile_of: np.ndarray | None = None) -> float:
    """The split-half reliability of the observed shift, ``2r / (1 + r)``.

    ``split="cells"``: random halves of each niche's cells. ``"tiles"``: a
    random half of the pair's tiles on both sides (noise a tile's cells share
    lands in one half; NaN when the tiles cannot separate both niches).
    Half-sample rates are floored at ``MIN_RATE``; 0 when ``r <= 0``.
    """
    if split not in ("cells", "tiles"):
        raise ValueError(f"split must be 'cells' or 'tiles', not {split!r}")
    if split == "tiles" and tile_of is None:
        raise ValueError("split='tiles' needs tile_of")
    half_shifts = []
    for _ in range(n_splits):
        if split == "tiles":
            halves = _tile_halves(rows_a, rows_b, tile_of, rng)
            if halves is None:
                return float("nan")
        else:
            halves = []
            for rows in (rows_a, rows_b):
                order = rng.permutation(len(rows))
                halves.append([rows[order[:len(rows) // 2]], rows[order[len(rows) // 2:]]])
        for side in (0, 1):
            rate_a = np.asarray(x_rate[halves[0][side]].mean(axis=0)).ravel()
            rate_b = np.asarray(x_rate[halves[1][side]].mean(axis=0)).ravel()
            half_shifts.append(np.log(np.maximum(rate_b[keep], MIN_RATE))
                               - np.log(np.maximum(rate_a[keep], MIN_RATE)))
    r = float(np.mean([np.corrcoef(half_shifts[2 * i], half_shifts[2 * i + 1])[0, 1]
                       for i in range(n_splits)]))
    return float(2 * r / (1 + r)) if r > 0 else 0.0


def niche_pairs(labels: np.ndarray, y: np.ndarray, connected: np.ndarray) -> list[tuple]:
    """Every niche pair, most composition-distinct (centroid distance) first."""
    ks = [k for k in np.unique(labels) if k >= 0]
    centroids = {k: y[connected & (labels == k)].mean(axis=0) for k in ks}
    scored = sorted(((np.linalg.norm(centroids[a] - centroids[b]), a, b)
                     for i, a in enumerate(ks) for b in ks[i + 1:]), reverse=True)
    return [(a, b) for _, a, b in scored]


def group_channels(model, data, group: np.ndarray, n_groups: int, context: bool = False,
                   runner=None) -> dict:
    """Per-group means of ``m_psi`` (``prior_w``), the clean rates ``rho`` and the
    foreign influx ``rho_bar`` over the cells with ``group >= 0`` (posterior
    means; float64 sums). *context*: also the mean context vector ``c``. *runner*:
    an existing ``model.runner(data)`` to reuse (its tiles are already on the
    device)."""
    import torch

    runner = runner if runner is not None else model.runner(data)
    network = runner.network
    network.eval()
    keys = ("prior_w", "rho", "rho_bar")
    if context:
        keys = keys + ("c",)
    acc = counts = None
    with torch.no_grad():
        for batch in runner.train_batches + runner.val_batches:
            kwargs = runner._forward_kwargs(batch)
            fwd = network(**kwargs, kappa=model.config.kappa, sample=False)
            n = batch["n_seeds"]
            gid = torch.as_tensor(np.asarray(group)[batch["nodes"][:n]],
                                  device=fwd.mu_w.device, dtype=torch.long)
            take = gid >= 0
            gid = gid[take]
            values = {"prior_w": fwd.prior_mean_w[:n][take],
                      "rho": fwd.log_rho[:n].exp()[take], "rho_bar": fwd.rho_bar[take],
                      "c": fwd.c[:n][take]}
            if acc is None:
                counts = torch.zeros(n_groups, dtype=torch.float64, device=gid.device)
                acc = {k: torch.zeros(n_groups, values[k].shape[1], dtype=torch.float64,
                                      device=gid.device) for k in keys}
            counts.index_add_(0, gid, torch.ones_like(gid, dtype=torch.float64))
            for k in keys:
                acc[k].index_add_(0, gid, values[k].double())
    denom = counts.clamp(min=1.0)[:, None]
    return {k: (v / denom).cpu().numpy() for k, v in acc.items()}


def _mixture(rho, rho_bar, kappa):
    return (1 - kappa) * rho + kappa * rho_bar


def decoded_shift(rho_a, bar_a, bar_b, response, kappa):
    """The predicted shift of A's cells moved to B: A's clean profile with B's response
    and B's influx, against A. Adding the response and spillover parts instead overshoots
    genes where both are large."""
    moved = rho_a * np.exp(response)
    moved = moved / moved.sum()
    return np.log(_mixture(moved, bar_b, kappa) + EPS) - np.log(_mixture(rho_a, bar_a, kappa) + EPS)


def tier_summary(tier: list[dict]) -> dict:
    """Means over a tier's pairs: every prediction's R², the noise ceilings and
    the predicted shift's fraction of them, the top-gene overlap, the median slopes,
    the alignment, and the number of pairs where the predicted shift beats both
    single parts."""
    s: dict = {key: float(np.mean([p[key]["r2"] for p in tier])) for key in PREDICTIONS}
    s["n_pairs"] = len(tier)
    s["n_trusted"] = int(sum(p["trusted"] for p in tier))
    s["noise_ceiling"] = float(np.mean([p["noise_ceiling"] for p in tier]))
    s["fraction_of_ceiling"] = (float(s["predicted"] / s["noise_ceiling"])
                                if s["noise_ceiling"] >= MIN_CEILING else float("nan"))
    s["top_gene_overlap"] = float(np.mean([p["top_gene_overlap"]["fraction"] for p in tier]))
    s["top_gene_chance"] = float(np.mean([p["top_gene_overlap"]["chance"] for p in tier]))
    s["median_slope"] = float(np.median([p["predicted"]["slope"] for p in tier]))
    s["predicted_beats_both"] = int(sum(
        p["predicted"]["r2"] > max(p["response"]["r2"], p["spillover"]["r2"])
        for p in tier))
    s["selection_share"] = float(s["full"] - s["predicted"])
    s["alignment"] = float(np.mean([p["alignment"] for p in tier]))
    tiled = [p for p in tier if np.isfinite(p["ceiling_tiles"])]
    s["n_pairs_tiles"] = len(tiled)
    s["n_trusted_tiles"] = int(sum(p["trusted_tiles"] for p in tier))
    s["ceiling_tiles"] = (float(np.mean([p["ceiling_tiles"] for p in tiled]))
                          if tiled else float("nan"))
    s["fraction_of_ceiling_tiles"] = (
        float(np.mean([p["predicted"]["r2"] for p in tiled]) / s["ceiling_tiles"])
        if s["ceiling_tiles"] >= MIN_CEILING else float("nan"))
    return s


def relocation(model, adata, *, n_niches: int = N_NICHES, niches: np.ndarray | None = None,
               min_cells: int = MIN_CELLS, data=None, n_threads: int = NUM_THREADS,
               exclude: Iterable[str] = EXCLUDED_TYPES) -> dict:
    """The mean relocation read of *model* on *adata* (module doc).

    *niches*: per-cell niche labels (``-1`` = none) instead of k-means on
    ``y`` with *n_niches* clusters (single-threaded by default). A pair needs
    *min_cells* model-side and ``min_cells // 5`` scored cells per niche. *data*: an
    assembly from ``model.assemble(adata)``. Pairs of an excluded type are
    drawn (so every other pair's ceiling draws stay the same) and then
    dropped. Writes ``.obs['discell_niche']`` and the record
    ``.uns['discell']['relocation']``: ``pairs``, ``summary`` per tier
    (``supported`` / ``extrapolation``, each also ``_trusted`` and
    ``_trusted_tiles``), ``alignment`` (mean over all pairs) and the mask.
    """
    exclude = tuple(exclude)
    data = data if data is not None else model.assemble(adata)
    config = model.config
    g = data.graph
    connected = g.degrees > 0
    if niches is None:
        labels = niche_labels(g.y, g.degrees, n_niches, config.seed, n_threads=n_threads)
        source = "kmeans"
    else:
        labels = np.asarray(niches, dtype=np.int64)
        source = "given"
    tile_of = np.full(data.n_cells, -1, dtype=np.int64)
    fold = np.zeros(data.n_cells, dtype=np.int64)
    for k, tile in enumerate(data.train_tiles + data.val_tiles):
        tile_of[tile] = k
        fold[tile] = k % U.N_FOLDS
    scored, model_side = fold == 0, fold != 0
    rng = np.random.default_rng(config.seed)
    rng_tiles = np.random.default_rng([config.seed, 1])

    n_types = len(data.type_names)
    n_groups = (int(labels.max()) + 1) * n_types
    model_rows = connected & model_side & (labels >= 0)
    group = np.full(data.n_cells, -1, dtype=np.int64)
    group[model_rows] = labels[model_rows] * n_types + data.t[model_rows]
    ch = group_channels(model, data, group, n_groups)
    b_matrix = model.network.B.weight.detach().cpu().numpy()
    kappa = config.kappa
    x_rate = data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    names = [str(n) for n in data.type_names]
    keep_type = target_types(names, exclude)

    pairs = []
    for niche_a, niche_b in niche_pairs(labels, g.y, connected):
        for t in range(n_types):
            members = {}
            for niche, side in ((niche_a, "A"), (niche_b, "B")):
                where = connected & (data.t == t) & (labels == niche)
                train_rows = np.flatnonzero(where & model_side)
                test_rows = np.flatnonzero(where & scored)
                if len(train_rows) < min_cells or len(test_rows) < min_cells // 5:
                    break
                members[side] = (train_rows, test_rows)
            if len(members) < 2:
                continue
            y_a, y_b = g.y[members["A"][0]], g.y[members["B"][0]]
            gap_vec = y_b.mean(0) - y_a.mean(0)
            gap = np.linalg.norm(gap_vec)
            unit = gap_vec / max(gap, 1e-12)
            spread = 0.5 * (float((y_a @ unit).std()) + float((y_b @ unit).std()))
            ga, gb = niche_a * n_types + t, niche_b * n_types + t
            rho_a, bar_a, rho_b, bar_b = ch["rho"][ga], ch["rho_bar"][ga], ch["rho"][gb], \
                ch["rho_bar"][gb]
            response = b_matrix @ (ch["prior_w"][gb] - ch["prior_w"][ga])
            base = np.log(_mixture(rho_a, bar_a, kappa) + EPS)
            full = np.log(_mixture(rho_b, bar_b, kappa) + EPS) - base
            spillover = np.log(_mixture(rho_a, bar_b, kappa) + EPS) - base

            obs_a = np.asarray(x_rate[members["A"][1]].mean(axis=0)).ravel()
            obs_b = np.asarray(x_rate[members["B"][1]].mean(axis=0)).ravel()
            keep = (obs_a > MIN_RATE) & (obs_b > MIN_RATE)
            observed = np.log(obs_b[keep] + EPS) - np.log(obs_a[keep] + EPS)
            predictions = {"predicted": decoded_shift(rho_a, bar_a, bar_b, response, kappa),
                           "full": full, "response": response, "spillover": spillover}
            pair = {"niches": [int(niche_a), int(niche_b)], "type": names[t],
                     "n_model": [len(members[s][0]) for s in "AB"],
                     "n_scored": [len(members[s][1]) for s in "AB"],
                     "n_genes": int(keep.sum()), "overlap_flag": bool(gap > 3 * spread)}
            pair.update({k: score_shift(v[keep], observed) for k, v in predictions.items()})
            pair["noise_ceiling"] = noise_ceiling(x_rate, members["A"][1], members["B"][1],
                                                   keep, rng)
            pair["top_gene_overlap"] = top_gene_overlap(predictions["predicted"][keep],
                                                         observed)
            pair["alignment"] = alignment(observed, response[keep], spillover[keep])
            pair["trusted"] = bool(pair["noise_ceiling"] >= TRUST_CEILING
                                    and pair["n_genes"] >= TRUST_GENES)
            cf = pair["predicted"]["r2"]
            pair["fraction_of_ceiling"] = (float(cf / pair["noise_ceiling"])
                                            if pair["noise_ceiling"] >= MIN_CEILING
                                            else float("nan"))
            pair["ceiling_tiles"] = noise_ceiling(x_rate, members["A"][1], members["B"][1],
                                                   keep, rng_tiles, split="tiles",
                                                   tile_of=tile_of)
            pair["trusted_tiles"] = bool(pair["ceiling_tiles"] >= TRUST_CEILING
                                          and pair["n_genes"] >= TRUST_GENES)
            pair["fraction_of_ceiling_tiles"] = (
                float(cf / pair["ceiling_tiles"]) if pair["ceiling_tiles"] >= MIN_CEILING
                else float("nan"))
            if keep_type[t]:
                pairs.append(pair)

    tiers = {"supported": [p for p in pairs if not p["overlap_flag"]],
             "extrapolation": [p for p in pairs if p["overlap_flag"]]}
    for name in ("supported", "extrapolation"):
        tiers[f"{name}_trusted"] = [p for p in tiers[name] if p["trusted"]]
        tiers[f"{name}_trusted_tiles"] = [p for p in tiers[name] if p["trusted_tiles"]]
    rec = {"kappa": float(kappa), "niche_source": source, "n_niches": int(labels.max()) + 1,
           "min_cells": int(min_cells),
           "pairs": pairs,
           "summary": {name: tier_summary(tier) for name, tier in tiers.items() if tier},
           "alignment": float(np.mean([p["alignment"] for p in pairs])) if pairs
           else float("nan"),
           "eval_mask": mask_record(names, data.t, exclude)}
    adata.obs["discell_niche"] = labels
    U.store(adata, "relocation", rec)
    return rec
