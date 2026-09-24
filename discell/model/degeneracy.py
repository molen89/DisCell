#!/usr/bin/env python3
"""Post-hoc degeneracy diagnostics (spec 7.10) for a finished run.

Two reads of "is z just t?": ``I(mu_z; t)/H(t)`` from a held-out logistic
probe with the within-type share of z's variance, and the held-out
reconstruction lost when every cell's z is replaced by its type's mean z
(w, the leak mixture and kappa unchanged). Fits from 2026-09-17 on record
both in ``metrics.json`` through ``Trainer.evaluate``; this module runs that
same evaluation from ``best.pt`` for earlier runs and writes
``runs/<run>/degeneracy.json``.

Usage::

    python -m discell.model.degeneracy --dataset <id> --run <run> [--device cpu]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from typing import Sequence

import numpy as np

from discell import paths
from discell.model import metrics as M
from discell.model.prepare import assemble
from discell.model.train import TrainConfig, Trainer

log = logging.getLogger("discell.model.degeneracy")

# ---------------------------------------------------------------------------
# The dead-context-channel guard (2026-09-23, FF best_s2)
# ---------------------------------------------------------------------------
# ``type_degeneracy`` above asks "is z just t?". This asks the mirror question
# of the other channel -- "is w just a per-type constant?" -- which the rest of
# the battery does not see: on FF ``best_s2`` the prior network m_psi(c, t)
# ignored c, w came out constant within each type, and recon, NMI, the probe
# and the cycle reads were all healthy (recon -7.326, the best of the three
# seeds). Two reads, both on held-out cells:
#
#   ``w_niche_mi``  I(niche label ; mu_w) by the Ross kNN estimator, with the
#                   within-type permutation floor beside it -- the same
#                   estimator and the same floor as 6b.3's MI quadrant, so the
#                   numbers are comparable with ``external_criteria mi-quadrant``.
#                   A per-type-constant w scores exactly its floor.
#   ``w_var_fraction_across_cells``  the share of w's total variance that
#                   varies across cells *within* a type, tr Cov(w|t) / tr Cov(w)
#                   (law of total variance). The per-type gauge offset mu_t is
#                   not identified (issues V12), so the between-type part of
#                   w's variance is gauge, not response: a dead channel reads 0
#                   here however large ||w|| is.
#
#: niches for the guard's label: K = 10 k-means on neighbour composition, the
#: same construction ``validate.niche_labels`` gives 6b.3 and the niche analysis
W_GUARD_NICHES = 10
W_GUARD_PERMS = 5           #: within-type permutations behind the floor
W_GUARD_MAX_CELLS = 20_000  #: cap on held-out cells for the kNN estimator
#: the flag: no niche information above what cell type already carries
DEAD_W_FLAG = "failed fit: dead context channel"


def w_channel_guard(w: np.ndarray, niche: np.ndarray, t: np.ndarray,
                    seed: int = 0, perms: int = W_GUARD_PERMS,
                    max_cells: int = W_GUARD_MAX_CELLS) -> dict:
    """The two guard reads on held-out arrays; no model state, no IO.

    *w* (n, d_w) posterior means, *niche* the k-means niche label (negative =
    isolated, dropped), *t* the cell type. Flagged when the MI excess over the
    within-type permutation floor is <= 0.
    """
    from discell.experiments.external_criteria import (
        mi_discrete_continuous, within_type_permutation)

    w = np.asarray(w, dtype=np.float64)
    niche, t = np.asarray(niche), np.asarray(t)
    rng = np.random.default_rng(seed)

    # the variance read uses every held-out cell (an exact O(n) pass)
    total = w.var(axis=0).sum()
    across = np.zeros(w.shape[1])
    for g in np.unique(t):
        members = t == g
        across += members.mean() * w[members].var(axis=0)
    var_fraction = float(across.sum() / max(total, 1e-12))

    rows = np.flatnonzero(niche >= 0)
    if len(rows) > max_cells:
        rows = np.sort(rng.choice(rows, max_cells, replace=False))
    y, t_rows, w_rows = niche[rows], t[rows], w[rows]
    mi = mi_discrete_continuous(w_rows, y, rng=rng)
    floor = [mi_discrete_continuous(w_rows,
                                    within_type_permutation(y, t_rows, rng),
                                    rng=rng)
             for _ in range(perms)]
    excess = float(mi - np.mean(floor))
    return {"w_niche_mi": float(mi),
            "w_niche_mi_floor": float(np.mean(floor)),
            "w_niche_mi_floor_sd": float(np.std(floor)),
            "w_niche_mi_excess": excess,
            "w_var_fraction_across_cells": var_fraction,
            "w_total_var": float(total),
            "n_cells": int(len(rows)), "n_niches": int(len(np.unique(y))),
            "n_perms": int(perms),
            "dead_context_channel": bool(excess <= 0.0),
            "flag": DEAD_W_FLAG if excess <= 0.0 else None}


def w_channel_guard_from_trainer(trainer, niches: int = W_GUARD_NICHES) -> dict:
    """``w_channel_guard`` on one fitted run's held-out (validation) cells."""
    from discell.model.validate import niche_labels

    seed = trainer.config.seed
    sweep = trainer._sweep(trainer.val_batches)
    rows = sweep["nodes"]
    labels = niche_labels(trainer.data, niches, seed)
    out = w_channel_guard(sweep["mu_w"], labels[rows], trainer.data.t[rows],
                          seed=seed)
    out["niches"] = int(niches)
    return out


def load_trainer(dataset: str, run: str, device: str):
    """``validate.load_run`` with *device* honoured for the resident tiles too.

    ``load_run`` builds the Trainer from the checkpoint's own ``device``
    ("cuda" for every pinned run), which would park the whole slide on a GPU
    even when the caller asked for the CPU.
    """
    import torch

    run_dir = paths.dataset(dataset).root / "runs" / run
    payload = torch.load(run_dir / "best.pt", map_location="cpu",
                         weights_only=False)
    payload["config"].setdefault("gat_sources", "type_z")   # pre-field era
    payload["config"].setdefault("subtract_leak", False)
    payload["config"].setdefault("gat_sink", False)
    config = dataclasses.replace(TrainConfig(**payload["config"]), device=device)
    data = assemble(dataset, config.variant, config.embeddings,
                    tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                    v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                    seed=config.seed, label_key=config.label_key)
    trainer = Trainer(config, data)
    trainer.model.load_state_dict(payload["model"])
    trainer.model.eval()
    return trainer, run_dir, int(payload["epoch"])


def w_contribution(trainer) -> dict:
    """Held-out per-count recon under the todo-2.3 decodes (devlog 2026-09-17).

    All decodes reuse ``Trainer._decode_seeds``: ``rho_bar`` and ``kappa`` stay
    exactly as the forward pass left them, posterior means throughout, only
    ``z`` and ``w`` move.

    ``full``                 (a) the model as fitted;
    ``typemean_z``           (b) z -> type mean of mu_z over training cells;
    ``typemean_z_prior_w``   (c) b, and w -> m_psi(c, t), the context field;
    ``typemean_z_ref_w``     (d) c, at the reference context: w -> m_psi(c_bar_t, t);
    ``type_profile``         (e) the empirical per-type count profile;
    ``own_z_ref_w``          (f) z kept, w -> m_psi(c_bar_t, t).

    ``c_bar_t`` is the mean context vector of the training cells of type *t*
    (the per-type gauge offset with the context dependence switched off,
    devlog "The w gauge", issue V12).
    """
    import torch

    config = trainer.config
    n_types = len(trainer.data.p_t)
    train = trainer._sweep(trainer.train_batches)
    t_train = trainer.data.t[train["nodes"]]
    z_bar = torch.tensor(M.type_means(train["mu_z"], t_train, n_types),
                         device=config.device)
    c_bar = torch.tensor(M.type_means(train["c"], t_train, n_types),
                         device=config.device)
    with torch.no_grad():
        onehot = torch.eye(n_types, device=config.device)
        w_ref = trainer.model.prior_w(torch.cat([c_bar, onehot], dim=-1))

    keys = ("full", "typemean_z", "typemean_z_prior_w", "typemean_z_ref_w",
            "own_z_ref_w")
    logs: dict[str, list] = {k: [] for k in keys}
    trainer.model.eval()
    with torch.no_grad():
        for batch in trainer.val_batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch),
                                kappa=config.kappa, sample=False)
            n = batch["n_seeds"]
            t_seed = batch["t"][:n]
            zb, wr = z_bar[t_seed], w_ref[t_seed]
            logs["full"].append(fwd.log_p.cpu().numpy())
            logs["typemean_z"].append(
                trainer._decode_seeds(fwd, zb, n).cpu().numpy())
            logs["typemean_z_prior_w"].append(
                trainer._decode_seeds(fwd, zb, n,
                                      fwd.prior_mean_w[:n]).cpu().numpy())
            logs["typemean_z_ref_w"].append(
                trainer._decode_seeds(fwd, zb, n, wr).cpu().numpy())
            logs["own_z_ref_w"].append(
                trainer._decode_seeds(fwd, fwd.mu_z[:n], n, wr).cpu().numpy())
    trainer.model.train()

    val_nodes = np.concatenate([b["nodes"][:b["n_seeds"]]
                                for b in trainer.val_batches])
    x_val = np.vstack([trainer.data.x[b["nodes"][:b["n_seeds"]]].toarray()
                       for b in trainer.val_batches])
    recon = {k: M.held_out_reconstruction(x_val, np.concatenate(v))
             for k, v in logs.items()}
    recon["type_profile"] = M.type_profile_reconstruction(
        trainer.data.x[train["nodes"]], t_train, x_val,
        trainer.data.t[val_nodes])
    gaps = {
        "b_minus_d": recon["typemean_z"] - recon["typemean_z_ref_w"],
        "d_minus_e": recon["typemean_z_ref_w"] - recon["type_profile"],
        "a_minus_b": recon["full"] - recon["typemean_z"],
        "a_minus_f": recon["full"] - recon["own_z_ref_w"],
        "b_minus_c": recon["typemean_z"] - recon["typemean_z_prior_w"],
    }
    return {"recon": recon, "gaps": gaps, "n_val_cells": int(len(val_nodes)),
            "config": {k: getattr(config, k) for k in
                       ("alpha_w", "alpha_z", "d_w", "d_z", "kappa", "seed",
                        "gat_sources", "epochs", "patience", "subtract_leak")}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--niches", type=int, default=W_GUARD_NICHES,
                        help="k-means niches behind the w-channel guard's "
                             "I(niche; mu_w) read")
    parser.add_argument("--w-contribution", action="store_true",
                        help="todo 2.3: the alpha_w x decode table; merges into "
                             "experiments/w_contribution.json instead of "
                             "writing degeneracy.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    trainer, run_dir, epoch = load_trainer(args.dataset, args.run, args.device)
    if args.w_contribution:
        out_path = (paths.dataset(args.dataset).root / "experiments"
                    / "w_contribution.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = json.loads(out_path.read_text()) if out_path.exists() else {}
        entry = w_contribution(trainer)
        entry["epoch"] = epoch
        table[args.run] = entry
        out_path.write_text(json.dumps(table, indent=2))
        r, g = entry["recon"], entry["gaps"]
        log.info("%s (epoch %d): a %.4f  b %.4f  c %.4f  d %.4f  e %.4f  "
                 "f %.4f | (b)-(d) %.4f  (d)-(e) %.4f  (a)-(b) %.4f",
                 args.run, epoch, r["full"], r["typemean_z"],
                 r["typemean_z_prior_w"], r["typemean_z_ref_w"],
                 r["type_profile"], r["own_z_ref_w"],
                 g["b_minus_d"], g["d_minus_e"], g["a_minus_b"])
        return 0
    report = trainer.evaluate()
    guard = w_channel_guard_from_trainer(trainer, args.niches)
    out = {"run": args.run, "epoch": epoch, "recon_val": report["recon_val"],
           "nmi": report["nmi"], "degeneracy": report["degeneracy"],
           "w_channel": guard,
           "recon_gap": report["recon_gap"]}
    (run_dir / "degeneracy.json").write_text(json.dumps(out, indent=2))
    d, g = out["degeneracy"], out["recon_gap"]
    log.info("%s (epoch %d): I/H %.3f  within-var %.3f  recon %.4f  "
             "type-mean z %.4f  gap %.4f  type profile %.4f",
             args.run, epoch, d["mi_ratio"], d["within_var_fraction"],
             g["recon"], g["recon_typemean_z"], g["gap"], g["recon_type_profile"])
    log.info("%s w channel: I(niche;w) %.4f  floor %.4f  excess %+.4f  "
             "across-cell var fraction %.4f%s",
             args.run, guard["w_niche_mi"], guard["w_niche_mi_floor"],
             guard["w_niche_mi_excess"], guard["w_var_fraction_across_cells"],
             f"  -- {guard['flag']}" if guard["flag"] else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
