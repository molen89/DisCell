"""The per-gene split: each gene's variation within a cell type, divided into the parts
due to the intrinsic state, the niche response and spillover (supplement S3.5).

For a held-out cell ``i`` of type ``t``, the decoder's three inputs are its intrinsic
logits ``a(z_i)``, its response logits ``B w_i`` (both at the posterior means) and its
neighbors' influx ``rho_bar_i``; the value is ``v = log p_g``. Setting each input to
the cell's value or the type's mean (the baseline: the mean ``mu_z`` and ``mu_w`` over
the type's held-out cells and the mean ``rho_bar`` over the connected ones), in all
eight combinations, gives exact Shapley values ``phi_z, phi_w, phi_rho_bar`` that sum
to ``v - v(baseline)``; ``phi_rho_bar = 0`` at kappa = 0. Part ``k``'s share of the
gene's within-type variation is ``s_k = Cov_t(phi_k, v) / Var_t(v)`` over the type's
held-out cells, so the shares sum to one; intervals are 95 % percentiles of a 200 um
tile bootstrap (the baseline held fixed). The split covers only what the model
explains: counting noise is most of the observed within-type variation.

Genes, per type with at least 300 held-out cells (unassigned cells are never a
target), as fixed in advance in the paper: with a within-type SD of ``v`` of at least
0.1 and at most 1 % of the type's cells at the decoder's floor, either *detected* in at
least 10 % of the type's cells, or a *spillover-test* gene, below 10 % of the rate of
the type that expresses it most (its source) and with a neighbor rate at least twice
its own.

A fit **assignments** part ``k`` for a gene when the interval of ``s_k`` lies above 0.5 and
``s_k`` is at least twice the next share. :func:`gene_assignments` takes the fits of a
sweep: a gene enters a type's population when it passes the gene rule in two of three
seeds at the operating kappa, its assignment is the part called in at least two of three
seeds, and its breakdown point ``kappa*`` is the smallest of 0.2, 0.3 and 0.4 at
which that consensus assignment changes. Read against a section's estimated bound (see
:func:`discell.sensitivity.kappa_bar`), a assignment *holds beyond kappa_bar* when it is
unchanged at the first tested kappa at or above it.

``gene_parts`` gives the per-cell parts of a few genes over every cell of one type,
the input of :func:`discell.pl.gene_map`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from discell.evaluate.mask import metric_target_mask
from discell.model.equations import EPS
from discell.model.trainer import Runner, resolve_device, tile_tensors
from discell.sensitivity.bootstrap import tile_draws, tile_index

PARTS = ("intrinsic", "response", "spillover")
N_BOOT = 200
MIN_CELLS = 300           #: held-out target cells a type needs
MIN_DETECT = 0.10         #: detected genes: share of the type's cells with a count
MIN_SD = 0.10             #: within-type SD of log p_g (nats)
MAX_FLOORED = 0.01        #: share of the type's cells at the decoder's floor
SOURCE_FRAC = 0.10        #: spillover-test genes: own rate below this x the source's
NEIGHBOR_RATIO = 2.0     #: ... and a neighbor rate at least this x the own rate
CALL_LO = 0.5             #: a assignment needs the share's lower limit above this
CALL_RATIO = 2.0          #: ... and the share at least this x the next share
CONSENSUS = 2             #: seeds of three that must agree
UP = (0.2, 0.3, 0.4)      #: the tested kappa at which a assignment may change
CHUNK = 1024              #: cells per Shapley block


# -- the Shapley decomposition ----------------------------------------------------

def value(a: torch.Tensor, b: torch.Tensor, rho_bar: torch.Tensor, kappa: float
          ) -> torch.Tensor:
    """``log p`` from intrinsic logits *a*, response logits *b* and influx *rho_bar*
    (the decoder of :func:`discell.model.equations.spillover_mix` without its floor;
    an isolated cell, ``rho_bar = 0``, gets ``log rho``)."""
    log_rho = torch.log_softmax(a + b, dim=-1)
    if kappa == 0.0:
        return log_rho
    total = (1.0 - kappa) + kappa * rho_bar.sum(dim=-1, keepdim=True)
    return torch.logaddexp(np.log(1.0 - kappa) + log_rho,
                           np.log(kappa) + rho_bar.log()) - total.log()


def shapley(a_own, a_base, b_own, b_base, rb_own, rb_base, kappa: float):
    """``(phi (n, G, 3), v_full, v_empty)``: exact Shapley values of the three inputs
    (intrinsic, response, influx), each at its own value or its baseline."""
    inputs = ((a_own, a_base), (b_own, b_base), (rb_own, rb_base))
    v = {}
    for s in range(8):
        bits = tuple((s >> k) & 1 for k in range(3))
        v[bits] = value(*(inputs[k][0] if bits[k] else inputs[k][1] for k in range(3)), kappa)
    weight = {0: 1.0 / 3.0, 1: 1.0 / 6.0, 2: 1.0 / 3.0}     # |S|! (2 - |S|)! / 3!
    phi = []
    for k in range(3):
        acc = torch.zeros_like(v[(1, 1, 1)])
        for bits, val in v.items():
            if bits[k]:
                continue
            on = list(bits)
            on[k] = 1
            acc += weight[sum(bits)] * (v[tuple(on)] - val)
        phi.append(acc)
    return torch.stack(phi, dim=-1), v[(1, 1, 1)], v[(0, 0, 0)]


def _seeds(model, data, tiles, keep, device):
    """Per tile, the forward pass at the posterior means and the rows of its seeds
    with *keep*: ``(nodes, mu_z, mu_w, rho_bar (float64), log_p)``."""
    net = model.network.to(device).eval()
    with torch.no_grad():
        for tile in tiles:
            batch = tile_tensors(data, tile, device)
            n = batch["n_seeds"]
            nodes = np.asarray(batch["nodes"][:n])
            sel = np.flatnonzero(keep[nodes])
            if len(sel) == 0:
                continue
            fwd = net(**Runner._forward_kwargs(batch), kappa=model.config.kappa, sample=False)
            st = torch.as_tensor(sel, device=device)
            yield (nodes[sel], fwd.mu_z[:n][st], fwd.mu_w[:n][st],
                   fwd.rho_bar[st].double(), fwd.log_p[:n][st])
            del batch, fwd


def _baseline(model, data, held_target, n_types, device) -> dict:
    """Per type: logits of the mean ``mu_z`` and ``mu_w`` over its held-out cells and
    the mean ``rho_bar`` over the connected ones (float64, ``(K, G)``)."""
    net = model.network
    sz = torch.zeros((n_types, model.config.d_z), dtype=torch.float64, device=device)
    sw = torch.zeros((n_types, model.config.d_w), dtype=torch.float64, device=device)
    srb = torch.zeros((n_types, len(model.gene_names)), dtype=torch.float64, device=device)
    n = torch.zeros(n_types, dtype=torch.float64, device=device)
    n_conn = torch.zeros(n_types, dtype=torch.float64, device=device)
    t = torch.as_tensor(data.t, device=device)
    for nodes, mz, mw, rb, _ in _seeds(model, data, data.val_tiles, held_target, device):
        tt = t[torch.as_tensor(nodes, device=device)]
        conn = (rb.sum(dim=1) > 0).double()
        sz.index_add_(0, tt, mz.double())
        sw.index_add_(0, tt, mw.double())
        srb.index_add_(0, tt, rb * conn[:, None])
        n.index_add_(0, tt, torch.ones_like(conn))
        n_conn.index_add_(0, tt, conn)
    with torch.no_grad():
        z = (sz / n.clamp(min=1)[:, None]).float()
        w = (sw / n.clamp(min=1)[:, None]).float()
        return {"a": net.dec_a(z).double(), "b": net.B(w).double(),
                "rb": srb / n_conn.clamp(min=1)[:, None], "n": n.cpu().numpy()}


def _rates(data, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the own rate ``x / l`` and the beta-weighted neighbor rate (dense)."""
    x = sp.csr_matrix(data.x)
    rate = (sp.diags(1.0 / np.clip(np.asarray(data.totals, float), 1.0, None)) @ x).tocsr()
    neigh = data.graph.in_edges[rows] @ rate
    return np.asarray(rate[rows].todense()), np.asarray(neigh.todense())


# -- one fit ------------------------------------------------------------------------

def gene_split(model, adata, *, data=None, n_boot: int = N_BOOT, seed: int | None = None,
               device: str | None = None, path: str | Path | None = None,
               min_cells: int = MIN_CELLS) -> dict:
    """The per-gene split of one fit on *adata*'s held-out cells (module doc).

    *data*: an assembly of *adata* made with :meth:`discell.Model.assemble`. *seed*: of
    the tile bootstrap (default the fit's seed). *device*: of the decodes (default the
    model's). *path*: an ``.npz`` file; when it exists it is read instead, otherwise
    the split is written there (so a sweep's splits resume). *min_cells*: held-out
    cells a type needs for shares (the paper: 300).

    Returns arrays over types ``K`` and genes ``G``: ``shares`` ``(K, 3, G, 3)`` (point,
    2.5 % and 97.5 % limits; parts in :data:`PARTS` order), ``n_cells`` (held-out
    target cells), ``detect``, ``own_rate``, ``neigh_rate``, ``sd`` (of ``log p_g``),
    ``floored``, the gene rule ``detected`` and ``spill_test``, ``source`` (per gene,
    the type with the largest own rate), ``assignment`` (``-1`` none, else the index of the
    called part), plus ``type_names``, ``gene_names``, ``kappa``, ``seed`` and the
    exactness ``checks`` (``shapley_sum``: the largest ``|sum_k phi_k - (v - v0)|``;
    ``log_p``: the largest difference to the forward pass where unfloored).
    """
    if path is not None and Path(path).exists():
        return read_split(path)
    data = data if data is not None else model.assemble(adata)
    dev = resolve_device(device or model.config.device)
    kappa = float(model.config.kappa)
    names = [str(v) for v in data.type_names]
    K, G = len(names), len(model.gene_names)
    target = metric_target_mask(data.t, names, (model.config.unassigned_label,))
    held = np.zeros(data.n_cells, dtype=bool)
    held[np.concatenate(data.val_tiles)] = True
    held_target = held & target
    nodes_all = np.flatnonzero(held_target)
    t_held = data.t[nodes_all]
    n_cells = np.bincount(t_held, minlength=K)
    big = np.flatnonzero(n_cells >= min_cells)          # only these types get shares
    slot = np.full(K, -1, dtype=np.int64)
    slot[big] = np.arange(len(big))
    base = _baseline(model, data, held_target, K, dev)

    # the tile bootstrap over the held-out target cells: one weight row per draw
    tiles, n_tiles = tile_index(np.asarray(data.positions, float)[nodes_all])
    W = np.vstack([np.ones((1, len(nodes_all)))]
                  + list(tile_draws(tiles, n_tiles, n_boot, model.config.seed
                                    if seed is None else seed))).astype(np.float32)
    col = np.full(data.n_cells, -1, dtype=np.int64)
    col[nodes_all] = np.arange(len(nodes_all))
    W = torch.as_tensor(W, device=dev)
    D = W.shape[0]
    # weighted sums per type: u = v - v0, u^2, phi_k, phi_k u
    acc = torch.zeros((len(big), 2 + 6, D, G), dtype=torch.float64, device=dev)
    floored = torch.zeros((K, G), dtype=torch.float64, device=dev)
    worst = {"shapley_sum": 0.0, "log_p": 0.0}
    t_dev = torch.as_tensor(data.t, device=dev)
    for nodes, mz, mw, rb, log_p in _seeds(model, data, data.val_tiles, held_target, dev):
        net = model.network
        for lo in range(0, len(nodes), CHUNK):
            sl = slice(lo, lo + CHUNK)
            nd = nodes[sl]
            tt = t_dev[torch.as_tensor(nd, device=dev)]
            with torch.no_grad():
                a_own, b_own = net.dec_a(mz[sl]).double(), net.B(mw[sl]).double()
            phi, v, v0 = shapley(a_own, base["a"][tt], b_own, base["b"][tt], rb[sl],
                                 base["rb"][tt], kappa)
            worst["shapley_sum"] = max(worst["shapley_sum"],
                                       float((phi.sum(-1) - (v - v0)).abs().max()))
            free = v > np.log(EPS)
            if free.any():
                worst["log_p"] = max(worst["log_p"],
                                     float((v - log_p[sl].double())[free].abs().max()))
            u = v - v0
            floored.index_add_(0, tt, (~free).double())
            for k in torch.unique(tt).tolist():
                if slot[k] < 0:
                    continue
                m = tt == k
                w = W[:, torch.as_tensor(col[nd], device=dev)[m]].double()
                uk, pk, a_k = u[m], phi[m], acc[slot[k]]
                a_k[0] += w @ uk
                a_k[1] += w @ (uk * uk)
                for j in range(3):
                    a_k[2 + j] += w @ pk[..., j]
                    a_k[5 + j] += w @ (pk[..., j] * uk)
            del phi, v, v0, u
    shares = np.full((K, 3, G, 3), np.nan, dtype=np.float32)
    sd = np.full((K, G), np.nan)
    for k in big:
        a_k = acc[slot[k]]
        tot = W[:, torch.as_tensor(np.flatnonzero(t_held == k), device=dev)].double().sum(1)
        mean = a_k / tot[None, :, None]
        var = mean[1] - mean[0] ** 2                                          # (D, G)
        cov = torch.stack([mean[5 + j] - mean[2 + j] * mean[0] for j in range(3)], -1)
        d = (cov / var[..., None]).cpu().numpy()                              # (D, G, 3)
        sd[k] = torch.sqrt(var[0].clamp(min=0)).cpu().numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            shares[k] = np.stack([d[0], np.nanpercentile(d[1:], 2.5, axis=0),
                                  np.nanpercentile(d[1:], 97.5, axis=0)])
    del acc
    # rates and detection over the same cells
    own, neigh = _rates(data, nodes_all)
    x_held = sp.csr_matrix(data.x)[nodes_all]
    detect = np.zeros((K, G))
    own_rate = np.zeros((K, G))
    neigh_rate = np.zeros((K, G))
    for k in np.flatnonzero(n_cells):
        r = t_held == k
        # count / n, not a sparse mean: a gene detected in exactly 10% must pass the rule
        detect[k] = np.asarray((x_held[r] > 0).sum(axis=0)).ravel() / r.sum()
        own_rate[k] = own[r].mean(axis=0)
        neigh_rate[k] = neigh[r].mean(axis=0)
    floored = floored.cpu().numpy() / np.maximum(n_cells, 1)[:, None]
    out = {"type_names": np.asarray(names), "gene_names": np.asarray(model.gene_names),
           "kappa": kappa, "seed": int(model.config.seed), "shares": shares,
           "n_cells": n_cells, "detect": detect, "own_rate": own_rate,
           "neigh_rate": neigh_rate, "sd": sd, "floored": floored, "min_cells": int(min_cells),
           "checks": {k: float(v) for k, v in worst.items()}}
    out.update(gene_rule(out))
    out["assignment"] = assignments(shares)
    if path is not None:
        write_split(path, out)
    return out


def gene_rule(split: Mapping) -> dict:
    """``detected``, ``spill_test`` (``(K, G)`` bool) and ``source`` (``(G,)``) of one
    fit's split (module doc)."""
    big = np.asarray(split["n_cells"]) >= int(split.get("min_cells", MIN_CELLS))
    own, neigh = np.asarray(split["own_rate"]), np.asarray(split["neigh_rate"])
    common = ((np.asarray(split["sd"]) >= MIN_SD) & (np.asarray(split["floored"]) <= MAX_FLOORED)
              & big[:, None])
    source = np.argmax(np.where(big[:, None], own, -np.inf), axis=0)
    own_source = own[source, np.arange(own.shape[1])]
    spill = (common & (own < SOURCE_FRAC * own_source[None]) & (neigh >= NEIGHBOR_RATIO * own)
             & (np.arange(own.shape[0])[:, None] != source[None]))
    return {"detected": common & (np.asarray(split["detect"]) >= MIN_DETECT),
            "spill_test": spill, "source": source}


def assignments(shares: np.ndarray) -> np.ndarray:
    """``(K, G)`` assignment of one fit from its ``(K, 3, G, 3)`` shares: the part whose
    lower limit exceeds 0.5 and whose share is at least twice the next; -1 if none."""
    point, lo = shares[:, 0], shares[:, 1]
    out = np.full(point.shape[:-1], -1, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        for k in range(3):
            nxt = np.max(point[..., [j for j in range(3) if j != k]], axis=-1)
            out[(lo[..., k] > CALL_LO) & (point[..., k] >= CALL_RATIO * nxt)] = k
    return out


def write_split(path, split: Mapping) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in split.items() if k != "checks"}
    np.savez_compressed(path, checks=np.array(json.dumps(split["checks"])), **arrays)
    return path


def read_split(path) -> dict:
    import json

    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    out["checks"] = json.loads(str(out["checks"]))
    for k in ("kappa",):
        out[k] = float(out[k])
    out["seed"] = int(out["seed"])
    if "min_cells" in out:
        out["min_cells"] = int(out["min_cells"])
    return out


# -- a sweep ------------------------------------------------------------------------

def gene_assignments(splits: Mapping[tuple[float, int], Mapping], *, kappa: float = 0.1,
               kappa_bar: float | None = None, grid: Sequence[float] | None = None) -> dict:
    """The consensus assignments of a sweep's splits (module doc).

    *splits*: ``{(kappa, seed): gene_split(...)}`` over a sweep (the paper: six kappa x
    three seeds). *kappa*: the operating value whose fits set the population and the
    assignments. *kappa_bar*: a section's estimated bound (a number); with it each assignment gets
    ``holds_beyond_kappa_bar``. *grid*: the tested kappa (default: those in *splits*).

    Returns ``{"types": {type: {part: [gene rows]}}, "kappa", "kappa_bar",
    "first_tested_at_or_above", "counts", "arrays"}``; a gene row holds the gene, its
    seed-mean share of the called part, the three seed-mean shares, the rule that
    admitted it (``detected`` and/or ``spill_test``, with the source type), its
    ``kappa_star`` (None above 0.4) and, with *kappa_bar*, ``holds_beyond_kappa_bar``.
    ``arrays`` has the ``(K, G)`` population, consensus assignment, kappa* and seed-mean
    shares ``(K, G, 3)``.
    """
    seeds = sorted({s for k, s in splits if abs(k - kappa) < 1e-9})
    if not seeds:
        raise ValueError(f"no split at kappa = {kappa:g}")
    grid = sorted({float(k) for k, _ in splits}) if grid is None else [float(k) for k in grid]
    first = splits[(kappa, seeds[0])]
    names = [str(v) for v in first["type_names"]]
    genes = [str(g) for g in first["gene_names"]]
    need = min(CONSENSUS, len(seeds))
    at = [splits[(kappa, s)] for s in seeds]
    rules = [gene_rule(f) for f in at]
    kept = np.sum([np.asarray(f["n_cells"]) >= int(f.get("min_cells", MIN_CELLS))
                   for f in at], axis=0) >= need
    detected = np.sum([r["detected"] for r in rules], axis=0) >= need
    spill = np.sum([r["spill_test"] for r in rules], axis=0) >= need
    source = rules[0]["source"]
    population = (detected | spill) & kept[:, None]

    def consensus(k):
        cs = np.stack([assignments(np.asarray(splits[(k, s)]["shares"])) for s in seeds])
        out = np.full(cs.shape[1:], -1, dtype=np.int64)
        for j in range(3):
            out[(cs == j).sum(axis=0) >= need] = j
        return np.where(population, out, -1)

    cons = {k: consensus(k) for k in grid if all((k, s) in splits for s in seeds)}
    assignment = cons[min(cons, key=lambda k: abs(k - kappa))]
    star = np.full(assignment.shape, np.inf)
    for k in reversed([k for k in UP if k in cons]):
        star[cons[k] != assignment] = k
    star[assignment < 0] = np.nan
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # types without shares: NaN
        mean_shares = np.nanmean([np.asarray(f["shares"])[:, 0] for f in at], axis=0)
    cut = None
    if kappa_bar is not None:
        cut = next((k for k in grid if k >= kappa_bar - 1e-12), None)
    types = {}
    for t in np.flatnonzero(kept):
        by_part = {}
        for j, part in enumerate(PARTS):
            rows = []
            for g in np.flatnonzero(assignment[t] == j):
                row = {"gene": genes[g], "share": float(mean_shares[t, g, j]),
                       "shares": [float(v) for v in mean_shares[t, g]],
                       "detected": bool(detected[t, g]), "spill_test": bool(spill[t, g]),
                       "source": names[source[g]] if spill[t, g] else None,
                       "kappa_star": None if np.isinf(star[t, g]) else float(star[t, g])}
                if kappa_bar is not None:
                    row["holds_beyond_kappa_bar"] = bool(cut is not None and star[t, g] > cut)
                rows.append(row)
            by_part[part] = sorted(rows, key=lambda r: (-r["share"], r["gene"]))
        types[names[t]] = by_part
    counts = {k: {p: int((c == j).sum()) for j, p in enumerate(PARTS)} for k, c in cons.items()}
    return {"types": types, "kappa": float(kappa), "kappa_bar": kappa_bar,
            "first_tested_at_or_above": cut, "counts": counts,
            "arrays": {"population": population, "assignment": assignment, "kappa_star": star,
                       "mean_shares": mean_shares, "type_names": names, "gene_names": genes}}


# -- per-cell parts for maps --------------------------------------------------------

def gene_parts(model, adata, genes: Sequence[str], cell_type: str, *,
               context_type: str | None = None, data=None, device: str | None = None) -> dict:
    """The per-cell parts of *genes* over every cell of *cell_type* (training and
    held-out tiles), against the same baseline as :func:`gene_split`.

    Returns ``nodes`` (cell rows), ``phi`` ``(n, len(genes), 3)`` (natural log of the
    fold change each part contributes against an average cell of the type), ``counts``
    and ``totals`` of those cells, ``positions`` of every cell, ``context`` (the
    beta-weighted share of *context_type* neighbors per cell; NaN for isolated cells;
    None without *context_type*), and the ``genes``, ``cell_type`` and ``kappa``.
    """
    data = data if data is not None else model.assemble(adata)
    dev = resolve_device(device or model.config.device)
    names = [str(v) for v in data.type_names]
    if cell_type not in names:
        raise KeyError(f"no type {cell_type!r}")
    k = names.index(cell_type)
    gidx = np.array([list(model.gene_names).index(g) for g in genes])
    target = metric_target_mask(data.t, names, (model.config.unassigned_label,))
    held = np.zeros(data.n_cells, dtype=bool)
    held[np.concatenate(data.val_tiles)] = True
    base = _baseline(model, data, held & target, len(names), dev)
    kappa = float(model.config.kappa)
    keep = data.t == k
    out_nodes, out_phi = [], []
    net = model.network
    for nodes, mz, mw, rb, _ in _seeds(model, data, data.train_tiles + data.val_tiles, keep, dev):
        with torch.no_grad():
            a_own, b_own = net.dec_a(mz).double(), net.B(mw).double()
        n = len(nodes)
        phi, _, _ = shapley(a_own, base["a"][[k] * n], b_own, base["b"][[k] * n], rb,
                            base["rb"][[k] * n], kappa)
        out_nodes.append(nodes)
        out_phi.append(phi[:, torch.as_tensor(gidx, device=dev)].float().cpu().numpy())
    order = np.argsort(np.concatenate(out_nodes))
    nodes = np.concatenate(out_nodes)[order]
    x = sp.csr_matrix(data.x)
    context = None
    if context_type is not None:
        c = names.index(context_type)
        inn = data.graph.in_edges.tocsr()[nodes]
        rows = np.repeat(np.arange(len(nodes)), np.diff(inn.indptr))
        share = np.bincount(rows, weights=inn.data * (data.t[inn.indices] == c),
                            minlength=len(nodes))
        tot = np.asarray(inn.sum(axis=1)).ravel()
        context = np.where(tot > 0, share / np.where(tot > 0, tot, 1.0), np.nan)
    return {"nodes": nodes, "phi": np.concatenate(out_phi)[order],
            "counts": np.asarray(x[nodes][:, gidx].todense(), dtype=np.float32),
            "totals": np.asarray(data.totals, np.float32)[nodes],
            "positions": np.asarray(data.positions, np.float32), "context": context,
            "genes": list(genes), "cell_type": cell_type, "context_type": context_type,
            "kappa": kappa}


__all__ = ["CALL_LO", "CALL_RATIO", "CONSENSUS", "N_BOOT", "PARTS", "UP", "assignments",
           "gene_assignments", "gene_parts", "gene_rule", "gene_split", "read_split", "shapley",
           "value", "write_split"]
