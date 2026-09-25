#!/usr/bin/env python3
"""Held-out reconstruction as a model-comparison score (metrics package, item 2).

Devlog "Metrics package for the final tables (motivation, 2026-09-25 07:30)",
review R24. The early-stopping score is the autoencoding one: each held-out
cell decoded from its OWN posterior, so it reads the cell's counts twice
(encoder in, likelihood out). Beside it, per count and on the same held-out
cells, from the accepted checkpoint (``best.pt``):

``full``         posterior z, posterior w -- the score that selected the
                 checkpoint (with the NMI guard); must equal
                 ``metrics.json best.recon_val`` (asserted, 1e-3).
``intrinsic``    posterior z, w = m_psi(c, t), the prior mean: the context
                 channel predicted from the niche, not read off the cell.
``context``      z = the type's mean mu_z over TRAINING cells, w = m_psi(c,
                 t): nothing of the cell's own counts is used. What remains
                 is t, the cell's context c (neighbour types through the GAT
                 and its own ego-masked image embedding Phi_i) and the
                 foreign influx rho_bar (the neighbours' decodes).
``type_profile`` the empirical per-type count profile of the training cells
                 (``metrics.type_profile_reconstruction``): the reference.

In every decode the foreign influx and kappa stay as the forward pass left
them (``Trainer._decode_seeds``), posterior means throughout. Also recorded,
as a cross-check only: ``typemean_z`` (type-mean z, posterior w), the decode
behind ``recon_gap`` in metrics.json.

Each mode carries a 200 um tile-bootstrap 95 % CI over the held-out cells,
and so do the differences between modes (same draws). Per-cell values go to
``recon_modes_cells.npz`` for paired comparisons across runs.

With ``--eval-dataset`` the fit is applied to another section (the
cross-slide protocol of ``discell.model.crossslide``): every tile of that
section is held out, type means and the type profile come from the trained
section's training cells, and ``full`` must equal the ``recon_all_tiles`` of
``crossslide/<eval>.json`` when that file exists.

Usage::

    python -m discell.experiments.recon_modes --dataset <id> --run <run>
    python -m discell.experiments.recon_modes --dataset <id> --run <run> \\
        --eval-dataset gse315411_pdltma06_10_prime_dual

Writes ``runs/<run>/recon_modes.json`` (``recon_modes_<eval>.json``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

import numpy as np

log = logging.getLogger("discell.experiments.recon_modes")

MODES = ("full", "intrinsic", "context", "type_profile")
#: tolerance of the full-posterior reproduction of best.recon_val (nats/count)
REPRODUCE_TOL = 1e-3
DIFFERENCES = (("full", "intrinsic"), ("full", "context"),
               ("intrinsic", "context"), ("context", "type_profile"),
               ("full", "type_profile"))


def type_log_profile(x_train, t_train: np.ndarray, n_types: int) -> np.ndarray:
    """``log`` of the pooled per-type training profile, as
    ``metrics.type_profile_reconstruction`` builds it (floor 1e-8)."""
    import scipy.sparse as sp

    onehot = sp.csr_matrix((np.ones(len(t_train)),
                            (t_train, np.arange(len(t_train)))),
                           shape=(n_types, len(t_train)))
    pooled = np.asarray((onehot @ x_train).todense() if sp.issparse(x_train)
                        else onehot @ x_train, dtype=np.float64)
    return np.log((pooled / pooled.sum(axis=1, keepdims=True).clip(min=1.0)
                   ).clip(min=1e-8))


def per_cell_modes(trainer, batches: list[dict], z_bar: np.ndarray,
                   log_profile: np.ndarray) -> dict:
    """Per-cell per-count log-likelihood of every mode over *batches*' seeds."""
    import torch

    device = trainer.device
    zb = torch.as_tensor(z_bar, dtype=torch.float32, device=device)
    lp = torch.as_tensor(log_profile, dtype=torch.float32, device=device)
    out = {k: [] for k in ("nodes",) + MODES + ("typemean_z",)}
    trainer.model.eval()
    with torch.no_grad():
        for batch in batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch),
                                kappa=trainer.config.kappa, sample=False)
            n = batch["n_seeds"]
            x = batch["x"][:n].float()
            totals = x.sum(dim=1).clamp(min=1.0)
            t = batch["t"][:n]
            score = lambda log_p: ((x * log_p).sum(dim=1) / totals).cpu().numpy()
            prior_w = fwd.prior_mean_w[:n]
            out["nodes"].append(np.asarray(batch["nodes"][:n]))
            out["full"].append(score(fwd.log_p))
            out["intrinsic"].append(score(trainer._decode_seeds(
                fwd, fwd.mu_z[:n], n, prior_w)))
            out["context"].append(score(trainer._decode_seeds(
                fwd, zb[t], n, prior_w)))
            out["typemean_z"].append(score(trainer._decode_seeds(fwd, zb[t], n)))
            out["type_profile"].append(score(lp[t]))
    return {k: np.concatenate(v) for k, v in out.items()}


def summarise(cells: dict, positions: np.ndarray, n_boot: int, seed: int) -> dict:
    """Means per mode and the pairwise differences, with tile CIs."""
    from discell.experiments.bootstrap import tile_bootstrap, weighted_mean

    keys = MODES + ("typemean_z",)

    def stat(values, weights):
        means = {k: weighted_mean(values[k], weights) for k in keys}
        means.update({f"{a}-{b}": means[a] - means[b] for a, b in DIFFERENCES})
        return means

    boot = tile_bootstrap(positions[cells["nodes"]], cells, stat, n_boot,
                          seed=seed)
    reads = boot["reads"]
    return {"modes": {k: {"recon": reads[k]["estimate"], "ci95": reads[k]["ci95"],
                          "se": reads[k]["se"]} for k in keys},
            "differences": {f"{a}-{b}": {"value": reads[f"{a}-{b}"]["estimate"],
                                         "ci95": reads[f"{a}-{b}"]["ci95"]}
                            for a, b in DIFFERENCES},
            "bootstrap": {k: boot[k] for k in ("n_boot", "n_tiles", "n_cells",
                                               "tile_um", "seed")}}


def recon_modes(dataset: str, run: str, eval_dataset: str | None = None,
                device: str = "cuda", n_boot: int = 1000, seed: int = 0) -> dict:
    import torch

    from discell import paths
    from discell.model import metrics as M
    from discell.model.prepare import assemble
    from discell.model.train import Trainer
    from discell.model.validate import load_run

    config, data, trainer, run_dir, _ = load_run(dataset, run, device)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    n_types = len(data.p_t)
    train = trainer._sweep(trainer.train_batches)
    t_train = data.t[train["nodes"]]
    z_bar = M.type_means(train["mu_z"], t_train, n_types)
    log_profile = type_log_profile(data.x[train["nodes"]], t_train, n_types)
    checks: dict = {}
    if eval_dataset is None:
        cells = per_cell_modes(trainer, trainer.val_batches, z_bar, log_profile)
        positions = np.asarray(data.positions, dtype=np.float64)
        stored = metrics["best"]["recon_val"]
        gap = metrics["best"].get("recon_gap") or {}
        checks["typemean_z_vs_best_recon_gap"] = gap.get("recon_typemean_z")
        checks["type_profile_vs_best_recon_gap"] = gap.get("recon_type_profile")
        stored_name = "metrics.json best.recon_val"
        out_path = run_dir / "recon_modes.json"
        cells_path = run_dir / "recon_modes_cells.npz"
    else:
        data_b = assemble(eval_dataset, config.variant, config.embeddings,
                          tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                          v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                          seed=config.seed, label_key=config.label_key)
        if [str(n) for n in data.type_names] != [str(n) for n in data_b.type_names]:
            raise ValueError(f"type vocabularies differ: {dataset} / {eval_dataset}")
        model = trainer.model
        trainer.train_batches, trainer.val_batches = [], []
        del trainer
        torch.cuda.empty_cache()
        trainer = Trainer(config, data_b)
        trainer.model = model.to(trainer.device).eval()
        cells = per_cell_modes(trainer, trainer.train_batches + trainer.val_batches,
                               z_bar, log_profile)
        positions = np.asarray(data_b.positions, dtype=np.float64)
        cross = run_dir / "crossslide" / f"{eval_dataset}.json"
        stored = (json.loads(cross.read_text())["held_out_section"]
                  ["recon_all_tiles"] if cross.exists() else None)
        stored_name = f"crossslide/{eval_dataset}.json recon_all_tiles"
        out_path = run_dir / f"recon_modes_{eval_dataset}.json"
        cells_path = run_dir / f"recon_modes_cells_{eval_dataset}.npz"

    full = float(cells["full"].mean())
    if stored is not None:
        diff = abs(full - float(stored))
        assert diff <= REPRODUCE_TOL, (
            f"{run}: full-posterior recon {full:.5f} != {stored_name} "
            f"{stored:.5f} (|diff| {diff:.2e} > {REPRODUCE_TOL})")
        checks["full_vs_stored"] = {"stored": float(stored), "abs_diff": diff,
                                    "source": stored_name}
    else:
        checks["full_vs_stored"] = {"stored": None, "source": stored_name,
                                    "note": "no stored value to check against"}
    for key, mode in (("typemean_z_vs_best_recon_gap", "typemean_z"),
                      ("type_profile_vs_best_recon_gap", "type_profile")):
        if checks.get(key) is not None:
            checks[key] = {"stored": float(checks[key]),
                           "abs_diff": abs(float(cells[mode].mean())
                                           - float(checks[key]))}
    result = {"run": run, "dataset": dataset,
              "evaluated_on": eval_dataset or dataset,
              "section": "held-out section, every tile" if eval_dataset
              else "validation tiles (held out)",
              "epoch": metrics["best"]["epoch"],
              "selection": "checkpoint selected on `full` (held-out recon, "
                           "own posterior) jointly with the NMI guard",
              "n_cells": int(len(cells["nodes"])),
              **summarise(cells, positions, n_boot, seed),
              "checks": checks,
              "definitions": {
                  "full": "posterior mu_z, posterior mu_w",
                  "intrinsic": "posterior mu_z, w = m_psi(c, t)",
                  "context": "z = type mean of mu_z over training cells, "
                             "w = m_psi(c, t); no use of the cell's counts",
                  "type_profile": "pooled per-type training profile",
                  "typemean_z": "type-mean z, posterior w (recon_gap's "
                                "decode; cross-check only)",
                  "all": "per-count log-likelihood, rho_bar and kappa as the "
                         "forward pass left them"}}
    out_path.write_text(json.dumps(result, indent=2))
    np.savez_compressed(cells_path, **cells)
    m = result["modes"]
    log.info("%s on %s: full %.4f  intrinsic %.4f  context %.4f  type profile "
             "%.4f  (stored %s) -> %s", run, eval_dataset or dataset,
             m["full"]["recon"], m["intrinsic"]["recon"], m["context"]["recon"],
             m["type_profile"]["recon"], stored, out_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--eval-dataset", default=None,
                        help="apply the fit to this section (cross-slide)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    recon_modes(args.dataset, args.run, args.eval_dataset, args.device,
                args.n_boot, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
