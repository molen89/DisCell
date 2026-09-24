#!/usr/bin/env python3
"""The amortisation gap against an exactly computable posterior (todo 6b.8).

**The world.** DisCell's own generative model with the response channel
switched off, so the log-rate is the sum of exactly two terms: an intrinsic
latent and a leak term built from the neighbours.

    z_i ~ N(m_{t_i}, sigma^2 I_2)            intrinsic, d_z = 2
    rho_i = softmax(z_i A)                   clean composition, A known
    rho_bar_i = sum_j beta_ij rho_j          the true leak operator on the
                                             Delaunay graph (model.prepare)
    p_i propto (1 - kappa) rho_i + kappa rho_bar_i
    x_i ~ Multinomial(l_i, p_i),  l_i ~ Poisson(depth)

Nothing here is outside the model class the fit assumes: ``dec_a`` is an MLP
that contains ``z A``, and the leak mixture is the fit's own. So a gap
measured on this world is a gap of *inference*, not of model mismatch --
which is the point. ``w`` is set to zero in the world (the model still
carries its d_w = 2 channel, as it would on real tissue).

**Why the posterior is exact.** With ``rho_bar_i``, ``kappa``, ``A``,
``m_t``, ``sigma`` and ``l_i`` all known, the posterior of the intrinsic
latent

    log p(z | x_i, t_i, rho_bar_i) = -||z - m_{t_i}||^2 / 2 sigma^2
                                     + sum_g x_ig log p_ig(z) + const

is a two-dimensional density. It has no elementary closed form -- the
multinomial-softmax pair is not conjugate -- but two dimensions is small
enough that it is computed *exactly to numerical precision* by quadrature:
a Laplace fit (coarse grid, then Newton) gives the location and scale, a
129 x 129 dense grid over +/- 9 Laplace sd in whitened coordinates gives the
normalising constant, mean and variance. ``posterior_moments`` asserts the
grid's boundary mass is negligible, and the unit test checks it against a
long importance sample. The alternative the brief offered -- a conjugate
Gaussian observation, where ``E[z|x]`` is literally linear -- was rejected
because DisCell's decoder is multinomial: feeding it Gaussian observations
would measure model mismatch on top of the amortisation gap.

**What is compared.** ``mu_z(x_i, t_i)`` from the trained encoder against
``E[z | x_i, t_i, rho_bar_i]``. ``z`` is identified only up to an invertible
affine map (prior N(0, I), MLP decoder), so ``mu_z`` is first put in the
world's basis by a 5-fold cross-fitted affine least-squares map -- six
parameters, fitted on the same cells. The per-cell gap is reported in units
of the posterior width,

    g_i = || mu_hat_i - E[z|x_i] || / || sd(z | x_i) ||

Because ``dec_a`` is an MLP, ``z`` is in fact identified only up to a smooth
invertible reparameterisation, not just an affine one, so a second reading --
``discell_warped`` -- puts ``mu_z`` in the world's basis with a cross-fitted
MLP instead. That cannot cheat: ``mu_z`` is two-dimensional, so a regression
from it can only use information the encoder actually kept. ``discell`` is
the strict affine reading, ``discell_warped`` the gauge-free one.

**References.** Three amortisers are fitted *supervised* against the same
target, all 5-fold cross-fitted. ``linear_oracle`` (the brief's ridge
regression) and ``mlp_oracle`` read exactly the encoder's inputs,
``encode_counts(x)`` and the type one-hot; ``mlp_oracle_rhobar`` also reads
``log rho_bar_i``, which the encoder is structurally denied -- spec 7.13
says the ``kappa l rho_bar`` bias cannot be removed by any function of
``x`` alone. The distance between the last two is therefore the part of the
gap that is unreachable rather than unlearned, and ``prior_mean`` (the type
mean, knowing no counts at all) is the floor of the scale.

**Axes.** kappa mismatch (model kappa minus world kappa) in
{-0.1, -0.05, 0, +0.05, +0.1} x depth in {100, 300, 1000} x 3 seeds.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m discell.experiments.planted_posterior \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np

from discell import paths

log = logging.getLogger("discell.experiments.planted_posterior")

N_CELLS = 3000
N_GENES = 40
N_TYPES = 4
D_Z = 2                     # the world's intrinsic latent; also the model's
D_PHI = 8
SIGMA_Z = 0.5               # within-type spread of z, as in model.synthetic
TYPE_SPREAD = 1.5           # m_t ~ N(0, TYPE_SPREAD^2)
BOX_UM = 700.0              # 3000 cells here matches the 6000/1000um density
KAPPA_WORLD = 0.2
DEPTHS = (100, 300, 1000)
MISMATCH = (-0.1, -0.05, 0.0, 0.05, 0.1)
EPOCHS = 300
GRID_N = 129                # quadrature nodes per whitened dimension
GRID_HALF = 9.0             # half-width in Laplace sd
FOLDS = 5
RIDGE = 1e-2


# -- the world -------------------------------------------------------------

def build_world(seed: int, depth: int, kappa: float = KAPPA_WORLD):
    """One tissue from the two-term world above, as a ``Simulation``.

    Reuses :func:`discell.model.prepare.build_graph` so the leak operator the
    world uses is edge-for-edge the one the fit uses.
    """
    from scipy.spatial import Delaunay

    from discell.model.prepare import build_graph
    from discell.model.synthetic import Simulation

    rng = np.random.default_rng(1000 + seed)
    positions = rng.uniform(0, BOX_UM, size=(N_CELLS, 2))
    anchors = rng.uniform(0, BOX_UM, size=(3 * N_TYPES, 2))
    anchor_type = np.arange(3 * N_TYPES) % N_TYPES
    gaps = np.linalg.norm(positions[:, None] - anchors[None], axis=-1)
    logits = -gaps / (0.05 * BOX_UM) + rng.gumbel(size=gaps.shape)
    t = anchor_type[logits.argmax(axis=1)]

    simplices = Delaunay(positions).simplices
    pairs = np.unique(np.sort(np.vstack(
        [simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]]
    ), axis=1), axis=0)
    dist = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    graph = build_graph(pairs[:, 0], pairs[:, 1], np.ones(len(pairs)), dist,
                        N_CELLS, type_index=t, n_types=N_TYPES)

    m_t = rng.normal(0.0, TYPE_SPREAD, size=(N_TYPES, D_Z))
    z = m_t[t] + SIGMA_Z * rng.standard_normal((N_CELLS, D_Z))
    a_load = rng.normal(0.0, 1.0, size=(D_Z, N_GENES))
    logits_rho = z @ a_load
    rho = np.exp(logits_rho - logits_rho.max(axis=1, keepdims=True))
    rho /= rho.sum(axis=1, keepdims=True)

    rho_bar = np.asarray(graph.in_edges @ rho)
    p = (1.0 - kappa) * rho + kappa * rho_bar
    p /= p.sum(axis=1, keepdims=True)

    totals = np.maximum(rng.poisson(depth, N_CELLS), 20)
    x = np.stack([rng.multinomial(totals[i], p[i]) for i in range(N_CELLS)])
    phi = graph.y @ rng.normal(0.0, 1.0, size=(N_TYPES, D_PHI)) \
        + 0.4 * rng.standard_normal((N_CELLS, D_PHI))

    sim = Simulation(graph=graph, positions=positions, t=t,
                     x=x.astype(np.float32), totals=totals.astype(np.float32),
                     phi=phi.astype(np.float32), z_true=z,
                     w_true=np.zeros((N_CELLS, 1)),
                     B_true=np.zeros((1, N_GENES)),
                     rho_true=rho, p_true=p, kappa=kappa)
    return sim, {"A": a_load, "m_t": m_t, "rho_bar": rho_bar,
                 "sigma_z": SIGMA_Z}


# -- the exact posterior ---------------------------------------------------

def _log_posterior(z, truth_A, m, rho_bar, kappa, x, sigma):
    """Unnormalised log p(z | x, t, rho_bar), broadcasting over grid nodes.

    *z* is (..., d_z); *m*, *rho_bar*, *x* carry the cell axis and broadcast.
    """
    import torch

    logits = z @ truth_A
    rho = torch.softmax(logits, dim=-1)
    p = (1.0 - kappa) * rho + kappa * rho_bar
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    prior = -((z - m) ** 2).sum(dim=-1) / (2.0 * sigma ** 2)
    return prior + (x * p.clamp_min(1e-30).log()).sum(dim=-1)


def posterior_moments(sim, truth, kappa: float | None = None,
                      device: str = "cuda", chunk: int = 64,
                      grid_n: int = GRID_N, grid_half: float = GRID_HALF):
    """Exact ``E[z | x_i, t_i, rho_bar_i]`` and ``sd`` for every cell.

    Laplace fit (coarse grid then Newton) sets the quadrature frame; the
    moments come from a dense ``grid_n^2`` grid over ``+/- grid_half`` Laplace
    sd in whitened coordinates. Returns (mean, sd, diagnostics).
    """
    import torch

    device = device if torch.cuda.is_available() else "cpu"
    kappa = float(sim.kappa if kappa is None else kappa)
    dt = torch.float64
    A = torch.tensor(truth["A"], dtype=dt, device=device)
    m = torch.tensor(truth["m_t"][sim.t], dtype=dt, device=device)
    rho_bar = torch.tensor(truth["rho_bar"], dtype=dt, device=device)
    x = torch.tensor(np.asarray(sim.x, dtype=np.float64), dtype=dt, device=device)
    sigma = float(truth["sigma_z"])

    # isolated cells have a zero rho_bar row, so the mixture renormalises to
    # rho: the world's own convention, reproduced here exactly.
    n = len(sim.t)
    mean = torch.zeros(n, D_Z, dtype=dt, device=device)
    sd = torch.zeros(n, D_Z, dtype=dt, device=device)
    edge_mass = torch.zeros(n, dtype=dt, device=device)

    # coarse grid, shared across cells, in the prior's own coordinates
    lin = torch.linspace(-4.5 * sigma, 4.5 * sigma, 91, dtype=dt, device=device)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    coarse = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)   # (C, 2)
    # fine grid in whitened coordinates
    ulin = torch.linspace(-grid_half, grid_half, grid_n, dtype=dt, device=device)
    uy, ux = torch.meshgrid(ulin, ulin, indexing="ij")
    unit = torch.stack([ux.reshape(-1), uy.reshape(-1)], dim=-1)     # (Q, 2)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        m_c, rb_c, x_c = m[s:e, None], rho_bar[s:e, None], x[s:e, None]
        lp = _log_posterior(m_c + coarse[None], A, m_c, rb_c, kappa, x_c, sigma)
        z0 = m[s:e] + coarse[lp.argmax(dim=-1)]
        # Newton polish with an exact Hessian (autograd, per cell)
        for _ in range(8):
            zv = z0.clone().requires_grad_(True)
            f = _log_posterior(zv, A, m[s:e], rho_bar[s:e], kappa, x[s:e], sigma)
            g = torch.autograd.grad(f.sum(), zv, create_graph=True)[0]
            rows = [torch.autograd.grad(g[:, k].sum(), zv, retain_graph=True)[0]
                    for k in range(D_Z)]
            H = torch.stack(rows, dim=1).detach()
            g = g.detach()
            step = torch.linalg.solve(-H, g.unsqueeze(-1)).squeeze(-1)
            z0 = (z0 + step).detach()
            if float(step.abs().max()) < 1e-10:
                break
        # the quadrature frame only has to bracket the posterior, so the
        # Laplace covariance is symmetrised and its eigenvalues floored and
        # capped at the prior's -- a cell where the mode search lands on a
        # non-concave point then widens the grid instead of failing.
        cov = torch.linalg.inv(-H)
        cov = 0.5 * (cov + cov.transpose(1, 2))
        ev, Q = torch.linalg.eigh(cov)
        ev = ev.clamp(1e-12, 4.0 * sigma ** 2)
        cov = Q @ torch.diag_embed(ev) @ Q.transpose(1, 2)
        L = torch.linalg.cholesky(cov)
        zq = z0[:, None] + (unit[None] @ L.transpose(1, 2))           # (c, Q, 2)
        lq = _log_posterior(zq, A, m_c, rb_c, kappa, x_c, sigma)
        wq = torch.softmax(lq, dim=-1)
        mu = (wq[..., None] * zq).sum(dim=1)
        var = (wq[..., None] * (zq - mu[:, None]) ** 2).sum(dim=1)
        mean[s:e], sd[s:e] = mu, var.clamp_min(0).sqrt()
        border = torch.zeros(grid_n, grid_n, dtype=torch.bool, device=device)
        border[0], border[-1], border[:, 0], border[:, -1] = True, True, True, True
        edge_mass[s:e] = wq[:, border.reshape(-1)].sum(dim=-1)

    worst = float(edge_mass.max())
    if worst > 1e-6:
        raise RuntimeError(f"quadrature grid too narrow: boundary mass {worst:.2e}")
    return (mean.cpu().numpy(), sd.cpu().numpy(),
            {"max_boundary_mass": worst,
             "mean_posterior_sd": float(sd.mean())})


# -- amortisers and the gap ------------------------------------------------

def _folds(n: int, seed: int, k: int = FOLDS) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n) % k


def _cross_fit_linear(features: np.ndarray, target: np.ndarray,
                      fold: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Out-of-fold ridge predictions of *target* from *features* (+ intercept)."""
    f = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    out = np.zeros_like(target)
    eye = ridge * np.eye(f.shape[1])
    eye[-1, -1] = 0.0
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        beta = np.linalg.solve(f[tr].T @ f[tr] + eye, f[tr].T @ target[tr])
        out[te] = f[te] @ beta
    return out


def _cross_fit_mlp(features: np.ndarray, target: np.ndarray, fold: np.ndarray,
                   device: str = "cuda", epochs: int = 400,
                   hidden: int = 128, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions of a small supervised MLP -- the ceiling."""
    import torch
    from torch import nn

    device = device if torch.cuda.is_available() else "cpu"
    f = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(target, dtype=torch.float32, device=device)
    out = np.zeros_like(target)
    for k in np.unique(fold):
        torch.manual_seed(seed * 100 + int(k))
        tr = torch.tensor(fold != k, device=device)
        te = torch.tensor(fold == k, device=device)
        net = nn.Sequential(nn.Linear(f.shape[1], hidden), nn.SiLU(),
                            nn.Linear(hidden, hidden), nn.SiLU(),
                            nn.Linear(hidden, target.shape[1])).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ((net(f[tr]) - y[tr]) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            out[fold == k] = net(f[te]).cpu().numpy()
    return out


def gap(pred: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> dict:
    """``||pred - E[z|x]|| / ||sd(z|x)||`` per cell, summarised."""
    d = np.linalg.norm(pred - mean, axis=1)
    g = d / np.linalg.norm(sd, axis=1)
    return {"mean": float(g.mean()), "median": float(np.median(g)),
            "p90": float(np.percentile(g, 90)),
            # the same gap in the world's own z units: the sd-normalised
            # column grows with depth partly because the posterior narrows,
            # and this says how much of that is the encoder getting worse.
            "abs_mean": float(d.mean())}


def encoder_features(sim) -> np.ndarray:
    """Exactly what ``enc_z`` reads: ``encode_counts(x)`` and the type one-hot."""
    import torch

    from discell.model.networks import encode_counts

    feat = encode_counts(torch.tensor(np.asarray(sim.x, dtype=np.float32)),
                         float(np.median(sim.totals))).numpy()
    onehot = np.eye(N_TYPES, dtype=np.float32)[sim.t]
    return np.concatenate([feat, onehot], axis=1)


@dataclass
class Cell:
    mismatch: float
    depth: int
    seed: int


#: the world and its exact posterior depend on (seed, depth) only, not on the
#: kappa the *model* is told, so the quadrature is done once per world.
_WORLDS: dict[tuple[int, int], tuple] = {}


def world_and_posterior(seed: int, depth: int, device: str = "cuda"):
    key = (seed, depth)
    if key not in _WORLDS:
        sim, truth = build_world(seed, depth)
        mean, sd, diag = posterior_moments(sim, truth, device=device)
        _WORLDS.clear()                      # one world at a time is enough
        _WORLDS[key] = (sim, truth, mean, sd, diag)
    return _WORLDS[key]


def run_cell(cell: Cell, device: str = "cuda", epochs: int = EPOCHS) -> dict:
    """One (mismatch, depth, seed): build, fit, integrate, measure."""
    from discell.applications.planted import fit_synthetic

    t0 = time.time()
    sim, truth, mean, sd, diag = world_and_posterior(cell.seed, cell.depth,
                                                     device=device)
    fold = _folds(len(sim.t), seed=cell.seed)
    fit = fit_synthetic(sim, epochs=epochs, device=device, seed=cell.seed,
                        d_z=D_Z, kappa=sim.kappa + cell.mismatch)

    features = encoder_features(sim)
    rows = {
        "discell": gap(_cross_fit_linear(fit["z"], mean, fold), mean, sd),
        "discell_warped": gap(_cross_fit_mlp(fit["z"], mean, fold, device=device,
                                             seed=cell.seed + 7), mean, sd),
        "linear_oracle": gap(_cross_fit_linear(features, mean, fold), mean, sd),
        "mlp_oracle": gap(_cross_fit_mlp(features, mean, fold, device=device,
                                         seed=cell.seed), mean, sd),
        # the same supervised MLP with the true rho_bar handed to it: the
        # encoder is structurally denied this input, so the distance between
        # this arm and mlp_oracle is the part of the gap no function of
        # (x, t) can close (spec 7.13).
        "mlp_oracle_rhobar": gap(_cross_fit_mlp(
            np.concatenate([features,
                            np.log(np.clip(truth["rho_bar"], 1e-8, None)
                                   ).astype(np.float32)], axis=1),
            mean, fold, device=device, seed=cell.seed + 13), mean, sd),
        "prior_mean": gap(truth["m_t"][sim.t], mean, sd),
    }
    spread = float(np.linalg.norm(mean - mean.mean(axis=0), axis=1).mean()
                   / np.linalg.norm(sd, axis=1).mean())
    out = {"mismatch": cell.mismatch, "depth": cell.depth, "seed": cell.seed,
           "kappa_world": float(sim.kappa),
           "kappa_model": float(sim.kappa + cell.mismatch),
           "gaps": rows, "posterior": diag,
           "signal_to_posterior_sd": spread,
           "seconds": round(time.time() - t0, 1)}
    log.info("%s", json.dumps(out))
    return out


# -- report ----------------------------------------------------------------

def summarise(rows: list[dict]) -> dict:
    """3-seed mean and range per (mismatch, depth) per amortiser."""
    table = {}
    for r in rows:
        key = f"{r['mismatch']:+.2f}|{r['depth']}"
        table.setdefault(key, {"mismatch": r["mismatch"], "depth": r["depth"],
                               "seeds": [], "arms": {}})
        table[key]["seeds"].append(r["seed"])
        table[key].setdefault("abs", {})
        table[key].setdefault("posterior_sd", []).append(
            r["posterior"]["mean_posterior_sd"])
        for arm, g in r["gaps"].items():
            table[key]["arms"].setdefault(arm, []).append(g["mean"])
            table[key]["abs"].setdefault(arm, []).append(g["abs_mean"])
    for entry in table.values():
        entry["posterior_sd"] = float(np.mean(entry["posterior_sd"]))
        for field in ("arms", "abs"):
            entry[field] = {
                arm: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                      "max": float(np.max(v)), "n_seeds": len(v)}
                for arm, v in entry[field].items()}
    return table


def verdict(table: dict, arm: str = "discell_warped") -> dict:
    """Pre-registered wish: small at matched kappa, growing smoothly with it.

    Scored on *arm*. ``discell_warped`` is the reading the wish is about --
    the strict affine gauge mixes the encoder's error with how nonlinearly
    its latent happens to be warped, which training changes on its own.
    """
    out = {}
    for depth in DEPTHS:
        at = {m: table[f"{m:+.2f}|{depth}"]["arms"][arm]["mean"]
              for m in MISMATCH if f"{m:+.2f}|{depth}" in table}
        if len(at) != len(MISMATCH):
            continue
        matched = at[0.0]
        wings = [at[m] for m in MISMATCH if m != 0.0]
        out[str(depth)] = {
            "matched": matched,
            "min_is_at_matched": bool(matched <= min(at.values()) + 1e-9),
            "monotone_each_side": bool(
                at[-0.1] >= at[-0.05] >= matched and at[0.1] >= at[0.05] >= matched),
            "max_wing_over_matched": float(max(wings) / matched)
            if matched > 0 else float("nan"),
            "gap_in_sd": at,
        }
    return out


def write_report(rows: list[dict], table: dict, out_dir, args) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"design": {
        "n_cells": N_CELLS, "n_genes": N_GENES, "n_types": N_TYPES,
        "d_z": D_Z, "sigma_z": SIGMA_Z, "kappa_world": KAPPA_WORLD,
        "depths": list(DEPTHS), "mismatch": list(MISMATCH),
        "epochs": args.epochs, "seeds": args.seeds,
        "quadrature": {"grid": [GRID_N, GRID_N], "half_width_sd": GRID_HALF},
    }, "runs": rows, "table": table,
        "verdict": verdict(table, "discell_warped"),
        "verdict_affine_gauge": verdict(table, "discell")}
    (out_dir / "planted_posterior.json").write_text(json.dumps(payload, indent=2))

    arms = ("discell", "discell_warped", "mlp_oracle", "mlp_oracle_rhobar",
            "linear_oracle", "prior_mean")
    lines = ["# Planted-posterior amortisation gap (todo 6b.8)", "",
             f"Gap ||mu_z - E[z|x]|| / ||sd(z|x)||, {len(args.seeds)}-seed mean "
             "(min-max), kappa_world = 0.2.", ""]
    for arm in arms:
        lines += [f"## {arm}", "",
                  "| kappa mismatch | " + " | ".join(f"depth {d}" for d in DEPTHS) + " |",
                  "|---|" + "---|" * len(DEPTHS)]
        for m in MISMATCH:
            cells = []
            for d in DEPTHS:
                e = table.get(f"{m:+.2f}|{d}", {}).get("arms", {}).get(arm)
                cells.append(f"{e['mean']:.3f} ({e['min']:.3f}-{e['max']:.3f})"
                             if e else "-")
            lines.append(f"| {m:+.2f} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## discell, absolute units (world z, prior sd 0.5)", "",
              "| kappa mismatch | " + " | ".join(f"depth {d}" for d in DEPTHS) + " |",
              "|---|" + "---|" * len(DEPTHS)]
    for m in MISMATCH:
        cells = []
        for d in DEPTHS:
            e = table.get(f"{m:+.2f}|{d}", {}).get("abs", {}).get("discell")
            cells.append(f"{e['mean']:.3f}" if e else "-")
        lines.append(f"| {m:+.2f} | " + " | ".join(cells) + " |")
    lines += ["",
              "Mean posterior sd per depth: " + ", ".join(
                  f"{d}: {table[f'+0.00|{d}']['posterior_sd']:.4f}"
                  for d in DEPTHS if f"+0.00|{d}" in table), "",
              "## Verdict (gauge-free reading, discell_warped)", "", "```",
              json.dumps(payload["verdict"], indent=2), "```", "",
              "## Verdict (strict affine gauge, discell)", "", "```",
              json.dumps(payload["verdict_affine_gauge"], indent=2), "```", ""]
    (out_dir / "planted_posterior.md").write_text("\n".join(lines))

    depths = [d for d in DEPTHS if any(f"{m:+.2f}|{d}" in table for m in MISMATCH)]

    def series(d, arm, field):
        return [table.get(f"{m:+.2f}|{d}", {}).get("arms", {}).get(arm, {}).get(field)
                for m in MISMATCH]

    # row 1: every arm on one scale. row 2: the gauge-free DisCell reading on
    # its own scale -- the pre-registered wish is a shallow U around matched
    # kappa, and it is invisible next to the prior-mean baseline.
    fig, axes = plt.subplots(2, len(depths), figsize=(4 * len(depths), 6.4),
                             squeeze=False)
    for row, shown in enumerate((arms, ("discell_warped",))):
        for ax, d in zip(axes[row], depths):
            for arm in shown:
                y, lo, hi = (series(d, arm, f) for f in ("mean", "min", "max"))
                if any(v is None for v in y):
                    continue
                ax.plot(MISMATCH, y, marker="o", label=arm)
                ax.fill_between(MISMATCH, lo, hi, alpha=0.15)
            ax.set_xlabel("model kappa - world kappa")
            ax.axvline(0.0, color="0.7", lw=0.8, zorder=0)
            ax.set_title(f"depth {d}" if row == 0
                         else f"depth {d} -- discell_warped only")
    axes[0][0].set_ylabel("gap / posterior sd")
    axes[1][0].set_ylabel("gap / posterior sd")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "planted_posterior.png", dpi=160)
    plt.close(fig)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    parser.add_argument("--mismatch", type=float, nargs="+", default=list(MISMATCH))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="synthetic_smoke")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    rows = []
    for depth in args.depths:
        for seed in args.seeds:
            for mismatch in args.mismatch:
                rows.append(run_cell(Cell(mismatch, depth, seed),
                                     device=args.device, epochs=args.epochs))
    table = summarise(rows)
    out_dir = paths.dataset(args.dataset).root / "experiments"
    payload = write_report(rows, table, out_dir, args)
    log.info("verdict %s", json.dumps(payload["verdict"], indent=2))
    log.info("wrote %s", out_dir / "planted_posterior.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
