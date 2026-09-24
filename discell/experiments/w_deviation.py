#!/usr/bin/env python3
"""The deviation read of the alpha_w multiplier ladder (review R19).

Pre-registered in the devlog, "alpha_w in units of 1/l-bar: the multiplier
ladder (R19, motivation, 2026-09-24 13:50)". The question: does the per-cell
deviation ``d_i = mu_w,i - m_psi(c_i, t_i)`` do held-out work, or is every
delivered w read the prior mean ``m_psi``? All reads come from ``best.pt``,
the accepted checkpoint. Held-out cells are the seeds of the validation
tiles, the cells behind ``recon_val``.

* (i)   KL_w per dimension: the history row at the best epoch, beside the
        same quantity recomputed here on the held-out cells (a check that
        this read and the trainer read one checkpoint).
* (ii)  ``d`` must be neither identity nor cycle: NMI(k-means(d), type) on
        held-out cells (the ``z_type_nmi`` protocol) and the cycle R^2 of
        ``d`` (pooled, the protocol of the trainer's ``cycle_z``: fit on
        training cells, scored on held-out cells).
* (iii) the held-out gain: per-count reconstruction of held-out cells
        decoded with ``w = mu_w`` minus decoded with ``w = m_psi(c, t)``. ``z``
        stays ``mu_z``; the foreign influx and kappa stay as the forward pass
        left them (the ``degeneracy.w_contribution`` convention). The
        ``w = mu_w`` decode is the model's own, so its number is
        ``recon_val`` at the checkpoint (recorded as a check).
* the w-side mirror (the ladder's guard): held-out ridge R^2 of mu_w's
        within-type residual on the neighbours' mu_z (``N_i = sum_j beta_ij
        mu_z_j``, the leak graph's weights), with the within-type permutation
        floor; the same read with ``d`` as the target beside it.
* (iv)  the twin margin at ``w = mu_w`` vs ``w = m_psi``: NOT computed. The
        transport twin read decodes its targets at the niche-group mean
        ``m_psi`` or at the own ``mu_w`` (``--read twins``), never at the
        cell's own ``m_psi(c_i, t_i)``, and running it costs a transport pass
        per fit; the entry says so instead of carrying a number.

Writes ``runs/<run>/w_deviation.json``.

Usage::

    python -m discell.experiments.w_deviation --dataset <id> --run <run> [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

import numpy as np

from discell.model import metrics as M

log = logging.getLogger("discell.experiments.w_deviation")

TWIN_SKIPPED = ("not computed: the transport twin read decodes targets at the "
                "niche-group mean m_psi or at the own mu_w, never at the cell's "
                "own m_psi(c_i, t_i), and costs a transport pass per fit")


def collect(trainer, batches: list[dict]) -> dict:
    """Per seed of *batches*, in eval mode (posterior means, ``sample=False``).

    Returns ``nodes``, ``t``, ``mu_z``, ``mu_w``, ``prior_w`` (= m_psi(c, t)),
    per-dimension ``kl_w`` (the trainer's formula), and three per-cell
    per-count log-likelihoods: ``ll_mu_w`` (decoded at ``w = mu_w``),
    ``ll_prior_w`` (at ``w = m_psi``) and ``ll_forward`` (the forward pass's
    own ``log_p``, which equals ``ll_mu_w`` on an unmodified model).
    """
    import torch

    model = trainer.model
    was_training = model.training
    model.eval()
    keys = ("nodes", "mu_z", "mu_w", "prior_w", "kl_w", "ll_mu_w",
            "ll_prior_w", "ll_forward")
    out: dict = {k: [] for k in keys}

    def per_count(x, log_p):
        return ((x * log_p).sum(dim=-1)
                / x.sum(dim=-1).clamp(min=1.0)).cpu().numpy()

    with torch.no_grad():
        for batch in batches:
            fwd = model(**trainer._forward_kwargs(batch),
                        kappa=trainer.config.kappa, sample=False)
            n = batch["n_seeds"]
            x = batch["x"][:n].float()
            mu_z = fwd.mu_z[:n]
            out["nodes"].append(np.asarray(batch["nodes"][:n]))
            out["mu_z"].append(mu_z.cpu().numpy())
            out["mu_w"].append(fwd.mu_w[:n].cpu().numpy())
            out["prior_w"].append(fwd.prior_mean_w[:n].cpu().numpy())
            out["kl_w"].append((0.5 * (-fwd.logvar_w[:n] + fwd.logvar_w[:n].exp()
                                       + (fwd.mu_w[:n] - fwd.prior_mean_w[:n]) ** 2
                                       - 1.0)).cpu().numpy())
            out["ll_mu_w"].append(per_count(
                x, trainer._decode_seeds(fwd, mu_z, n, fwd.mu_w[:n])))
            out["ll_prior_w"].append(per_count(
                x, trainer._decode_seeds(fwd, mu_z, n, fwd.prior_mean_w[:n])))
            out["ll_forward"].append(per_count(x, fwd.log_p))
    if was_training:
        model.train()
    return {k: np.concatenate(v) for k, v in out.items()}


def neighbour_mean(in_edges, values: np.ndarray, n_cells: int,
                   nodes: np.ndarray) -> np.ndarray:
    """``N_i = sum_j beta_ij v_j`` over the leak graph; *values* are given for
    *nodes* (every cell once), the result is indexed by cell id."""
    full = np.zeros((n_cells, values.shape[1]), dtype=np.float64)
    full[nodes] = values
    return np.asarray(in_edges @ full)


def within_type_mirror(target: np.ndarray, design: np.ndarray, t: np.ndarray,
                       train: np.ndarray, test: np.ndarray,
                       seed: int = 0) -> dict:
    """Held-out ridge R^2 of *target*'s within-type residual on *design*'s,
    against the within-type permutation floor.

    Both blocks are centred per type over the rows used (train and test);
    the floor permutes the centred design within each type and refits. The
    shape of ``metrics.mirror_r2``, held out as ``metrics.w_mirror_delta_r2``
    is (ridge 1e-3, at most ``MAX_EVAL_CELLS`` rows per side).
    """
    rng = np.random.default_rng(seed)
    rows_train = M._subsample_rows(np.flatnonzero(train), rng)
    rows_test = M._subsample_rows(np.flatnonzero(test), rng)
    if len(rows_train) < 50 or len(rows_test) < 50:
        return {"r2": float("nan"), "r2_floor": float("nan"),
                "n_train": int(len(rows_train)), "n_test": int(len(rows_test))}
    keep = np.concatenate([rows_train, rows_test])
    y = np.asarray(target, dtype=np.float64)[keep].copy()
    x = np.asarray(design, dtype=np.float64)[keep].copy()
    t_keep = t[keep]
    permuted = np.empty_like(x)
    for g in np.unique(t_keep):
        members = np.flatnonzero(t_keep == g)
        y[members] -= y[members].mean(axis=0)
        x[members] -= x[members].mean(axis=0)
        permuted[members] = x[members[rng.permutation(len(members))]]
    n_tr = len(rows_train)

    def held_out_r2(block: np.ndarray) -> float:
        d_tr = np.hstack([block[:n_tr], np.ones((n_tr, 1))])
        gram = d_tr.T @ d_tr + 1e-3 * np.eye(d_tr.shape[1])
        coef = np.linalg.solve(gram, d_tr.T @ y[:n_tr])
        d_te = np.hstack([block[n_tr:], np.ones((len(block) - n_tr, 1))])
        resid = y[n_tr:] - d_te @ coef
        total = ((y[n_tr:] - y[n_tr:].mean(axis=0)) ** 2).sum()
        if total <= 1e-12:             # a constant target: nothing to explain
            return float("nan")
        return float(1.0 - (resid ** 2).sum() / total)

    return {"r2": held_out_r2(x), "r2_floor": held_out_r2(permuted),
            "n_train": int(n_tr), "n_test": int(len(rows_test))}


def cycling_types(data) -> np.ndarray | None:
    """The trainer's cycle protocol: the MKI67-ranked cycling types without
    "Unassigned", the first four (``Trainer.evaluate``)."""
    if data.cycle is None:
        return None
    names = [str(n) for n in data.type_names]
    return np.array([g for g in data.cycle["cycling_types"]
                     if "nassigned" not in names[g]][:4])


def deviation_read(trainer) -> dict:
    """Every number of the read for one loaded model; no IO."""
    data, seed = trainer.data, trainer.config.seed
    train = collect(trainer, trainer.train_batches)
    val = collect(trainer, trainer.val_batches)

    # (iii) the held-out gain
    recon_mu = float(val["ll_mu_w"].mean())
    recon_prior = float(val["ll_prior_w"].mean())
    per_cell_gain = val["ll_mu_w"] - val["ll_prior_w"]
    recon = {"mu_w": recon_mu, "prior_w": recon_prior,
             "gain": recon_mu - recon_prior,
             "gain_median_cell": float(np.median(per_cell_gain)),
             "gain_frac_cells_positive": float((per_cell_gain > 0).mean()),
             # the model's own decode; equals mu_w on an unmodified model
             "forward": float(val["ll_forward"].mean())}

    d_val = val["mu_w"] - val["prior_w"]
    t_val = data.t[val["nodes"]]
    # rows in the trainer's evaluate() order: training seeds, then held-out
    rows_all = np.concatenate([train["nodes"], val["nodes"]])
    t_all = data.t[rows_all]
    train_mask = np.zeros(len(rows_all), dtype=bool)
    train_mask[:len(train["nodes"])] = True
    d_all = np.vstack([train["mu_w"] - train["prior_w"], d_val])
    w_all = np.vstack([train["mu_w"], val["mu_w"]])

    total_var = float(val["mu_w"].var(axis=0).sum())
    deviation = {
        # (ii) neither identity nor cycle
        "nmi_type": M.z_type_nmi(d_val, t_val, seed=seed),
        "cycle": None, "cycle_w_check": None,
        "mean_sq_norm": float((d_val ** 2).sum(axis=1).mean()),
        # the share of mu_w's held-out variance that is deviation
        "var_share_of_mu_w": float(d_val.var(axis=0).sum()
                                   / max(total_var, 1e-12)),
    }
    types = cycling_types(data)
    if types is not None:
        scores = np.stack([data.cycle["s_score"], data.cycle["g2m_score"]],
                          axis=1)[rows_all]
        deviation["cycle"] = M.cycle_r2(d_all, t_all, scores, types,
                                        train_mask, ~train_mask, seed=seed)
        # mu_w through the same protocol = the battery's cycle_w at the
        # checkpoint: the check that both reads saw one model
        deviation["cycle_w_check"] = M.cycle_r2(
            w_all, t_all, scores, types, train_mask, ~train_mask,
            seed=seed)["r2_pooled"]

    # the w-side mirror: neighbours' mu_z within type, held out
    n_cells = data.graph.n_cells
    neighbour_z = neighbour_mean(data.graph.in_edges,
                                 np.vstack([train["mu_z"], val["mu_z"]]),
                                 n_cells, rows_all)[rows_all]
    connected = data.graph.degrees[rows_all] > 0
    mirror_train, mirror_test = train_mask & connected, ~train_mask & connected
    w_mirror = within_type_mirror(w_all, neighbour_z, t_all, mirror_train,
                                  mirror_test, seed=seed)
    d_mirror = within_type_mirror(d_all, neighbour_z, t_all, mirror_train,
                                  mirror_test, seed=seed)

    return {"n_val_cells": int(len(val["nodes"])),
            "n_train_cells": int(len(train["nodes"])),
            "recon": recon,
            "kl_w_per_dim_recomputed": val["kl_w"].mean(axis=0).tolist(),
            "deviation": deviation,
            "w_mirror": w_mirror, "d_mirror": d_mirror,
            "twin_margin": {"skipped": TWIN_SKIPPED}}


def main(argv: Sequence[str] | None = None) -> int:
    from discell.experiments.at_best import battery_at_best
    from discell.model.degeneracy import load_trainer

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    trainer, run_dir, epoch = load_trainer(args.dataset, args.run, args.device)
    at_best = battery_at_best(run_dir)
    out = {"run": args.run, "dataset": args.dataset, "epoch": epoch,
           "alpha_w": trainer.config.alpha_w, "alpha_z": trainer.config.alpha_z,
           "seed": trainer.config.seed,
           # (i) from the history row at the accepted epoch
           "kl_w_per_dim": at_best.get("kl_w_per_dim"),
           "kl_w_sum": float(np.sum(at_best.get("kl_w_per_dim") or [np.nan])),
           "history_at_best": bool(at_best.get("at_best")),
           **deviation_read(trainer)}
    out["recon"]["recon_val_at_best"] = at_best.get("recon_val")
    (run_dir / "w_deviation.json").write_text(
        json.dumps(out, indent=2, default=float))
    r, dv = out["recon"], out["deviation"]
    log.info("%s (epoch %d, alpha_w %g): recon mu_w %.5f  m_psi %.5f  gain "
             "%+.5f (recon_val at best %.5f) | KL_w %.4g | d: NMI %.3f  cycle "
             "R2 %s  var share %.3f | w-mirror %.4f (floor %.4f)  d-mirror %.4f "
             "(floor %.4f)", args.run, epoch, out["alpha_w"], r["mu_w"],
             r["prior_w"], r["gain"], r["recon_val_at_best"] or float("nan"),
             out["kl_w_sum"], dv["nmi_type"],
             "--" if dv["cycle"] is None else f"{dv['cycle']['r2_pooled']:.4f}",
             dv["var_share_of_mu_w"], out["w_mirror"]["r2"],
             out["w_mirror"]["r2_floor"], out["d_mirror"]["r2"],
             out["d_mirror"]["r2_floor"])
    log.info("wrote %s", run_dir / "w_deviation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
