"""The estimated upper bound kappa_bar on a section's spillover fraction, and the
decision rule that reads each finding's breakdown point against it.

Removing more of the neighbors' profile from a cell eventually drives some
gene's share below zero, which no clean profile can have, so the counts
bound kappa from above (the paper's Proposition 1). One cell's bound is too
noisy to use, so it is pooled over the connected cells ``C_t`` of each type
``t`` (supplement S2.2)::

    N_tg = sum_{i in C_t} x_ig,   S_tg = sum_{i in C_t} l_i rho_bar_ig,
    kappa_bar_t = min_g U(N_tg) / S_tg

over the type's genes with ``S_tg >= 25`` (at least 10 spilled counts at
kappa = 0.4), where ``U`` is the exact (Garwood) one-sided Poisson upper limit
at level ``a = 0.05 / G_t``, Bonferroni over the type's ``G_t`` genes, so that a
gene with few counts cannot set the bound by chance.

* **model-based**: ``rho_bar_i`` from a fit, the forward pass at the posterior
  means over every tile. Each fit gives its own ``kappa_bar_t``; the type's
  value is their mean (the paper: the three seeds at kappa = 0.1).
* **counts-only**: ``rho_bar_i = sum_j beta_ij x_j / l_j``, the neighbors'
  observed profiles; no model. ``S`` then carries the counting noise of the
  neighbors, so it enters through a normal lower confidence limit
  ``S - z sd(S)``, with ``sd(S)^2 = sum_j (sum_{i in C_t} l_i beta_ij / l_j)^2
  x_jg``; genes also need ``sd(S) / S <= 0.2``, and the level ``0.05 / G_t`` is
  split evenly between the two limits.

A section's ``kappa_bar`` is the largest type value over the types with at least
200 connected cells (unassigned cells count as a type). It is the loosest of the
type bounds, so it rules out as little spillover as the data permit; a finding
that holds beyond it holds beyond every type's estimate. The smallest type
value is reported too. Both versions bound spillover of the model's form only:
one fraction per section, the same for every gene, from one ring of neighbors.
It is not a measurement of the physical share of misassigned transcripts.

:func:`decide` reads a breakdown table against a section's ``kappa_bar`` by the
paper's rule (Sec. 2.4): a finding is **safe** if it still holds at a tested
kappa at or above ``kappa_bar``; **not ruled out** (possibly an artifact) if it
fails by that value; **undecided** if ``kappa_bar`` lies above the tested range
and the finding holds throughout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy import stats

from discell.graph.kernel import read_graph
from discell.io.anndata import counts_matrix, fold_labels
from discell.model.trainer import Runner, resolve_device, tile_tensors

ALPHA = 0.05
#: Genes need S_tg >= this (at least 10 spilled counts at kappa = 0.4).
MIN_S = 25.0
#: Counts-only: genes need sd(S) / S <= this.
MAX_REL_SD = 0.2
#: A type enters the section value with at least this many connected cells.
MIN_CELLS = 200
SAFE, NOT_RULED_OUT, UNDECIDED = "safe", "not ruled out", "undecided"


def poisson_upper(n, a: float) -> np.ndarray:
    """The exact (Garwood) one-sided upper limit at level *a* of a Poisson mean
    from its count *n*: ``chi2_{1-a, 2(n+1)} / 2``."""
    return stats.chi2.ppf(1.0 - a, 2.0 * (np.asarray(n, dtype=np.float64) + 1.0)) / 2.0


def type_bounds(N: np.ndarray, S: np.ndarray, sd: np.ndarray | None = None, *,
                alpha: float = ALPHA, min_s: float = MIN_S,
                max_rel_sd: float = MAX_REL_SD) -> list[dict]:
    """Per type (rows of the ``(K, G)`` sums *N* and *S*): ``kappa_bar``, the
    binding gene (column index), the number of eligible genes and the 10th
    percentile and median over them of the point ratio ``N / S``. *sd*: the
    standard deviation of *S* for the counts-only version (None: model-based)."""
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        eligible = S >= min_s
        if sd is not None:
            eligible &= sd / S <= max_rel_sd
        for k in range(N.shape[0]):
            g = np.flatnonzero(eligible[k])
            row = {"kappa_bar": None, "gene": None, "n_genes": int(len(g)),
                   "q10": None, "median": None}
            if len(g):
                a = alpha / len(g) if sd is None else alpha / (2 * len(g))
                s = S[k, g] if sd is None else S[k, g] - stats.norm.ppf(1 - a) * sd[k, g]
                v = np.where(s > 0, poisson_upper(N[k, g], a) / np.where(s > 0, s, 1.0),
                             np.inf)
                i = int(np.argmin(v))
                point = N[k, g] / S[k, g]
                row.update(kappa_bar=float(v[i]) if np.isfinite(v[i]) else None,
                           gene=int(g[i]), q10=float(np.percentile(point, 10)),
                           median=float(np.percentile(point, 50)))
            out.append(row)
    return out


def _model_S(model, adata, t: np.ndarray, connected: np.ndarray, n_types: int,
             device=None) -> np.ndarray:
    """``(K, G)`` sums of ``l_i rho_bar_i`` over each type's connected cells, from
    one fit's forward pass at the posterior means, one tile at a time."""
    import torch

    data = model.assemble(adata)
    dev = resolve_device(device or model.config.device)
    net = model.network.to(dev).eval()
    S = torch.zeros((n_types, len(model.gene_names)), dtype=torch.float64, device=dev)
    t_dev = torch.as_tensor(t, device=dev)
    seen = np.zeros(data.n_cells, dtype=bool)
    with torch.no_grad():
        for tile in data.train_tiles + data.val_tiles:
            batch = tile_tensors(data, tile, dev)
            n = batch["n_seeds"]
            nodes = np.asarray(batch["nodes"][:n])
            seen[nodes] = True
            fwd = net(**Runner._forward_kwargs(batch), kappa=model.config.kappa, sample=False)
            ok = torch.as_tensor(connected[nodes], device=dev)
            if not ok.any():
                continue
            ell = batch["x"][:n].double().sum(dim=1, keepdim=True)
            idx = torch.as_tensor(nodes, device=dev)[ok]
            S.index_add_(0, t_dev[idx], (ell * fwd.rho_bar.double())[ok])
            del batch, fwd
    if not seen.all():
        raise AssertionError("tiles must cover every cell")
    return S.cpu().numpy()


def _summary(rows: Mapping[str, dict], n_connected: Mapping[str, int],
             min_cells: int) -> dict:
    shown = {n: r for n, r in rows.items()
             if n_connected[n] >= min_cells and r["kappa_bar"] is not None}
    if not shown:
        return {"kappa_bar": None, "type": None, "min": None, "min_type": None}
    hi = max(shown, key=lambda n: shown[n]["kappa_bar"])
    lo = min(shown, key=lambda n: shown[n]["kappa_bar"])
    return {"kappa_bar": shown[hi]["kappa_bar"], "type": hi,
            "min": shown[lo]["kappa_bar"], "min_type": lo}


def kappa_bar(adata, models: Sequence = (), *, counts_layer: str | None = None,
              alpha: float = ALPHA, min_s: float = MIN_S, max_rel_sd: float = MAX_REL_SD,
              min_cells: int = MIN_CELLS, device: str | None = None) -> dict:
    """The estimated upper bound on *adata*'s spillover fraction (module doc).

    *adata* carries the graph of :func:`discell.build_graph`; its labels, graph
    and counts define the types, the connected cells and ``N``. *models*: fitted
    :class:`discell.Model` s for the model-based version (the paper: the three
    seeds at kappa = 0.1; the same types and genes as *adata*); without them only
    the counts-only version is computed. *device*: of the forward passes
    (default: each model's).

    Returns ``{"counts_only": {...}, "model_based": {...} or None, "types",
    "n_connected", "params"}``. Each version holds the section value
    ``kappa_bar`` (the largest type value among types with at least *min_cells*
    connected cells), its ``type``, the smallest such value ``min`` and its
    ``min_type``, and per type ``types[name]`` (``kappa_bar``, binding ``gene``,
    ``n_genes``, and the 10th percentile and median of the point ratios). The
    model-based version adds per type the per-fit values ``per_fit`` and their
    ``range``, the section value of each fit (``per_fit_section``, the paper's
    "range over fits") and the fits' ``(kappa, seed)``.
    """
    meta = adata.uns.get("discell", {}).get("graph")
    if meta is None:
        raise KeyError("no DisCell graph in adata: run discell.build_graph first")
    graph = read_graph(adata)
    t, names = fold_labels(adata.obs[meta["label_key"]], meta["unassigned_label"])
    names = [str(v) for v in names]
    K = len(names)
    genes = np.asarray([str(g) for g in adata.var_names])
    x = counts_matrix(adata, counts_layer).astype(np.float64)
    ell = np.asarray(x.sum(axis=1)).ravel()
    B = sp.csr_matrix(graph.in_edges, dtype=np.float64)
    connected = graph.degrees > 0
    rows = np.flatnonzero(connected)
    onehot = sp.csr_matrix((np.ones(len(rows)), (t[rows], rows)), shape=(K, len(t)))
    N = np.asarray((onehot @ x).todense())
    # counts-only: S^C = sum_i l_i sum_j beta_ij x_j / l_j over each type's connected cells
    weight = sp.csr_matrix((ell[rows], (t[rows], rows)), shape=(K, len(t)))
    C = (weight @ B) @ sp.diags(1.0 / np.maximum(ell, 1.0))
    S_c = np.asarray((C @ x).todense())
    sd_c = np.sqrt(np.asarray((C.power(2) @ x).todense()))
    n_conn = {n: int(c) for n, c in zip(names, np.bincount(t[rows], minlength=K))}
    params = {"alpha": alpha, "min_s": min_s, "max_rel_sd": max_rel_sd,
              "min_cells": min_cells}

    def named(per_type):
        return {n: {**r, "gene": None if r["gene"] is None else str(genes[r["gene"]])}
                for n, r in zip(names, per_type)}

    counts_rows = named(type_bounds(N, S_c, sd_c, alpha=alpha, min_s=min_s,
                                    max_rel_sd=max_rel_sd))
    out = {"counts_only": {**_summary(counts_rows, n_conn, min_cells), "types": counts_rows},
           "model_based": None, "types": names, "n_connected": n_conn, "params": params}
    if not models:
        return out
    fits, per_fit = [], []
    for model in models:
        if list(model.type_names) != names or list(model.gene_names) != list(genes):
            raise ValueError("a model's types or genes differ from adata's")
        S_m = _model_S(model, adata, t, connected, K, device=device)
        per_fit.append(named(type_bounds(N, S_m, alpha=alpha, min_s=min_s)))
        fits.append([float(model.config.kappa), int(model.config.seed)])
    model_rows = {}
    for n in names:
        values = [f[n]["kappa_bar"] for f in per_fit]
        if any(v is None for v in values):
            model_rows[n] = {"kappa_bar": None, "per_fit": values, "range": None}
            continue
        binding = [f[n]["gene"] for f in per_fit]
        model_rows[n] = {
            "kappa_bar": float(np.mean(values)), "per_fit": values,
            "range": [min(values), max(values)],
            "gene": max(set(binding), key=binding.count), "genes": binding,
            "n_genes": [f[n]["n_genes"] for f in per_fit],
            "q10": float(np.mean([f[n]["q10"] for f in per_fit])),
            "median": float(np.mean([f[n]["median"] for f in per_fit]))}
    section = [_summary(f, n_conn, min_cells)["kappa_bar"] for f in per_fit]
    out["model_based"] = {**_summary(model_rows, n_conn, min_cells), "types": model_rows,
                          "per_fit_section": section, "fits": fits}
    return out


def section_value(bound) -> float:
    """A section's ``kappa_bar`` from a number or a :func:`kappa_bar` result (the
    model-based version when it was computed, else the counts-only one)."""
    if isinstance(bound, Mapping):
        version = bound.get("model_based") or bound["counts_only"]
        bound = version["kappa_bar"]
    if bound is None or not np.isfinite(bound):
        raise ValueError("no kappa_bar: no type has enough connected cells and genes")
    return float(bound)


def verdict(member: Mapping, kappa_bar: float, grid: Sequence[float]) -> str | None:
    """The paper's rule for one breakdown-table member (module doc); None for a
    member that is no finding at kappa = 0 or has a gap in its sweep."""
    status = member["status"]
    reached = [k for k in grid if k >= kappa_bar - 1e-12]
    if status == "above the grid":
        return SAFE if reached else UNDECIDED
    if status != "breaks":
        return None
    if reached and member["kappa_star"] > reached[0] + 1e-12:
        return SAFE
    return NOT_RULED_OUT


def decide(table: Mapping, kappa_bar) -> dict:
    """Each member's verdict against a section's estimated bound.

    *table*: a breakdown table (:func:`discell.sensitivity.breakdown` or
    :func:`discell.sensitivity.findings_breakdown`). *kappa_bar*: the section's
    value, a number or a :func:`kappa_bar` result. Returns ``{member:
    {"verdict", "status", "kappa_star", "first_tested_at_or_above"}}``, with
    verdict ``"safe"``, ``"not ruled out"``, ``"undecided"`` or None (no finding
    at kappa = 0, or a gap in the sweep).
    """
    bound = section_value(kappa_bar)
    grid = [float(k) for k in table["grid"]]
    reached = [k for k in grid if k >= bound - 1e-12]
    return {name: {"verdict": verdict(m, bound, grid), "status": m["status"],
                   "kappa_star": m["kappa_star"],
                   "first_tested_at_or_above": reached[0] if reached else None}
            for name, m in table["members"].items()}


__all__ = ["ALPHA", "MAX_REL_SD", "MIN_CELLS", "MIN_S", "NOT_RULED_OUT", "SAFE", "UNDECIDED",
           "decide", "kappa_bar", "poisson_upper", "section_value", "type_bounds", "verdict"]
