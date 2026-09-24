#!/usr/bin/env python3
"""The DisCell battery, computed identically for DisCell and for a baseline.

One code path produces every column of the comparison table (todo 8.5). The
inputs a column needs are only: an *intrinsic* latent, optionally a *spatial*
latent, and (where the tool decodes) held-out decoded rates. Everything else
-- which cells are held out, which label is ``t``, the niche block ``v``, the
cell-cycle scores, the cycling types -- comes from
:func:`discell.model.prepare.assemble` under the pinned run's configuration,
so a baseline is scored on exactly the cells and targets DisCell is scored on.

    # the DisCell column (also caches the context c the mirror read needs)
    python -m discell.experiments.baseline_battery --dataset D --discell-run best

    # a baseline column
    python -m discell.experiments.baseline_battery --dataset D \
        --method resolvi --latents .../latents.h5ad

    # the held-out section: assemble D under the run of another dataset
    python -m discell.experiments.baseline_battery --dataset D_dual \
        --config-from D_solo --config-run best --discell-run best

Artefacts: ``data/datasets/<D>/experiments/baseline_battery[_<tag>].json``
(one entry per method, merged across calls) and the markdown table beside it.

**What the columns mean.** Every read is the one ``Trainer.evaluate`` makes,
on the same functions in :mod:`discell.model.metrics`:

``nmi``            k-means over the intrinsic latent against the labels --
                   does the latent still know cell type at all.
``probe_dce``      held-out excess skill of (latent, t) at predicting the
                   niche block v, against the per-type mean; ``probe_floor``
                   is the same with the latent permuted within type. A latent
                   that has been cleaned of niche sits at its floor.
``mirror_r2``      within-type R^2 of the intrinsic latent on DisCell's
                   context vector c -- the mirror attractor. The same c is
                   used for every method, so this is a like-for-like read of
                   how much neighbourhood the latent carries.
``cycle_z``        within-type, held-out ridge R^2 of the continuous S/G2M
                   scores from the intrinsic latent (pooled over the cycling
                   types), with the within-type permuted control.
``cycle_spatial``  the same from the tool's spatial / microenvironment latent:
                   it should sit at the floor, because cycle is intrinsic.
``mi_ratio``,      the degeneracy pair: a held-out probe lower bound on
``within_var``     I(z;t)/H(t), and tr Cov(z|t)/tr Cov(z).
``recon``          mean per-count held-out log-likelihood of the decoded
                   rates, with the empirical per-type profile as the
                   reference line. Empty for tools that do not decode.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from discell.model import metrics as M

log = logging.getLogger("discell.baseline_battery")

#: cap on held-out cells used for the reconstruction read
MAX_RECON_CELLS = 40_000


# -- the battery itself -----------------------------------------------------

def battery(z: np.ndarray, t: np.ndarray, train: np.ndarray, test: np.ndarray,
            v_block: np.ndarray, vbar_t: np.ndarray, c: np.ndarray,
            spatial: np.ndarray | None = None, cycle: dict | None = None,
            seed: int = 0, n_comp: int | None = None) -> dict:
    """Every read of the battery from plain arrays; no IO, no model state.

    All arrays are indexed by *the cells this method produced a latent for*;
    ``train`` / ``test`` are boolean masks over the same rows. *n_comp*, the
    width of v_block's composition block, adds the per-block probe
    (``probe_blocks``, both graders, R20 + R22).
    """
    z = np.asarray(z, dtype=np.float64)
    out = {
        "n_cells": int(len(z)), "n_heldout": int(test.sum()),
        "d_intrinsic": int(z.shape[1]),
        "d_spatial": int(spatial.shape[1]) if spatial is not None else 0,
        "nmi": M.z_type_nmi(z, t, seed=seed),
        "mirror": M.mirror_r2(z, np.asarray(c, dtype=np.float64), t, seed=seed),
        "probe": M.probe_delta_ce(z, t, v_block, vbar_t, train, test, seed=seed),
        "degeneracy": M.type_degeneracy(z, t, train, test, seed=seed),
    }
    if n_comp is not None:
        out["probe_blocks"] = M.probe_blocks(z, t, v_block, vbar_t, train,
                                             test, n_comp=n_comp, seed=seed)
    if spatial is not None and spatial.shape[1] > 0:
        s = np.asarray(spatial, dtype=np.float64)
        out["spatial_var_fraction"] = float(
            s.var(axis=0).sum() / max(s.var(axis=0).sum() + z.var(axis=0).sum(),
                                      1e-12))
    if cycle is not None:
        types = np.asarray(cycle["types"])
        scores = np.asarray(cycle["scores"], dtype=np.float64)
        out["cycle"] = {
            "types": types.tolist(),
            "z": M.cycle_r2(z, t, scores, types, train, test, seed=seed),
        }
        if spatial is not None and spatial.shape[1] > 0:
            out["cycle"]["spatial"] = M.cycle_r2(
                np.asarray(spatial, dtype=np.float64), t, scores, types,
                train, test, seed=seed)
    return out


def reconstruction(x, t: np.ndarray, train: np.ndarray, rows: np.ndarray,
                   rate: np.ndarray | None) -> dict:
    """Held-out per-count log-likelihood on *rows* plus the profile reference.

    *rate* is the tool's decoded expectation on those rows (any positive
    scale: it is renormalised to a probability vector per cell, which is what
    :func:`held_out_reconstruction` consumes). ``None`` for a tool that does
    not decode -- only the reference line is then returned.
    """
    x_test = np.asarray(x[rows].todense() if hasattr(x[rows], "todense")
                        else x[rows], dtype=np.float64)
    train_rows = np.flatnonzero(train)
    out = {"n_cells": int(len(rows)),
           "recon_type_profile": M.type_profile_reconstruction(
               x[train_rows], t[train_rows], x_test, t[rows])}
    if rate is not None:
        rate = np.asarray(rate, dtype=np.float64)
        log_p = np.log((rate / rate.sum(axis=1, keepdims=True).clip(min=1e-12)
                        ).clip(min=1e-12))
        out["recon"] = M.held_out_reconstruction(x_test, log_p)
    return out


# -- assembling the shared side ---------------------------------------------

def run_config(dataset: str, run: str) -> dict:
    from discell import paths

    path = paths.dataset(dataset).root / "runs" / run / "config.json"
    return json.loads(path.read_text())


def shared_side(dataset: str, config: dict):
    """``assemble`` under *config* plus the masks and targets every column uses."""
    from discell.model.prepare import assemble

    data = assemble(dataset, config["variant"], config["embeddings"],
                    tile_cells=config["tile_cells"],
                    phi_pca=config.get("phi_pca"), v_pcs=config.get("v_pcs", 12),
                    val_fraction=config["val_fraction"], seed=config["seed"],
                    label_key=config.get("label_key"))
    n = data.n_cells
    test = np.zeros(n, dtype=bool)
    for tile in data.val_tiles:
        test[tile] = True
    cycle = None
    if data.cycle is not None:
        names = [str(x) for x in data.type_names]
        types = np.array([g for g in data.cycle["cycling_types"]
                          if "nassigned" not in names[g]][:4])
        cycle = {"types": types,
                 "scores": np.stack([data.cycle["s_score"],
                                     data.cycle["g2m_score"]], axis=1)}
    return data, ~test, test, cycle


def context_path(dataset: str, run: str, tag: str) -> Path:
    from discell import paths

    root = paths.dataset(dataset).root / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"baseline_context_{run}{tag}.npz"


def load_context(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run the DisCell column first "
            "(--discell-run), it caches the context c the mirror read needs")
    return np.load(path)["c"]


# -- the DisCell column -----------------------------------------------------

def discell_column(dataset: str, run: str, config_from: str | None,
                   device: str = "cuda") -> tuple[dict, np.ndarray]:
    """The battery on a DisCell run's own latents; also returns the context c.

    *config_from* names the dataset the run was trained on when *dataset* is a
    held-out section (the crossslide protocol: the trained weights, the
    training run's configuration, the other section's cells).
    """
    import torch

    from discell.model.train import Trainer
    from discell.model.validate import load_run

    source = config_from or dataset
    config, data_a, trainer_a, _, _ = load_run(source, run, device=device)
    if source == dataset:
        data, train, test, cycle = data_a, None, None, None
        n = data.n_cells
        test = np.zeros(n, dtype=bool)
        for tile in data.val_tiles:
            test[tile] = True
        train = ~test
        names = [str(x) for x in data.type_names]
        cycle = None
        if data.cycle is not None:
            types = np.array([g for g in data.cycle["cycling_types"]
                              if "nassigned" not in names[g]][:4])
            cycle = {"types": types,
                     "scores": np.stack([data.cycle["s_score"],
                                         data.cycle["g2m_score"]], axis=1)}
    else:
        data, train, test, cycle = shared_side(dataset, run_config(source, run))
        names_a = [str(x) for x in data_a.type_names]
        names_b = [str(x) for x in data.type_names]
        if names_a != names_b:
            raise ValueError("type vocabularies differ between "
                             f"{source} and {dataset}")

    # `load_run` already built a Trainer, and `Trainer.__init__` moves every
    # tile to the GPU -- on the 1.16 M-cell slide a second Trainer over the
    # same data is a second copy of the dataset on the card and OOMs. Reuse
    # the one we have; for a held-out section, free its tiles first.
    if source == dataset:
        trainer = trainer_a
        trainer.model = trainer.model.eval()
    else:
        model = trainer_a.model
        trainer_a.train_batches, trainer_a.val_batches = [], []
        del data_a, trainer_a
        torch.cuda.empty_cache()
        trainer = Trainer(config, data)
        trainer.model = model.to(trainer.device).eval()
    every = trainer.train_batches + trainer.val_batches
    swept = trainer._sweep(every)
    order = np.argsort(swept["nodes"])
    z = swept["mu_z"][order]
    w = swept["mu_w"][order]
    c = swept["c"][order]

    reads = battery(z, data.t, train, test, data.v_block, data.vbar_t, c,
                    spatial=w, cycle=cycle, seed=config.seed,
                    n_comp=data.n_comp)
    reads["kl_z_mean"] = float(np.mean(swept["kl_z"]))
    reads["kl_w_mean"] = float(np.mean(swept["kl_w"].sum(axis=1)))

    rows = np.flatnonzero(test)
    if len(rows) > MAX_RECON_CELLS:
        rows = np.sort(np.random.default_rng(0).choice(rows, MAX_RECON_CELLS,
                                                       replace=False))
    val = trainer._sweep(trainer.val_batches, want_log_p=True)
    keep = np.isin(val["nodes"], rows)
    x_test = np.asarray(data.x[val["nodes"][keep]].todense(), dtype=np.float64)
    train_rows = np.flatnonzero(train)
    reads["reconstruction"] = {
        "n_cells": int(keep.sum()),
        "recon": M.held_out_reconstruction(x_test, val["log_p"][keep]),
        "recon_type_profile": M.type_profile_reconstruction(
            data.x[train_rows], data.t[train_rows], x_test,
            data.t[val["nodes"][keep]])}
    reads["config"] = {"tool": "DisCell", "run": run, "trained_on": source,
                       "kappa": config.kappa, "d_z": config.d_z,
                       "d_w": config.d_w, "alpha_z": config.alpha_z,
                       "seed": config.seed}
    return reads, c


# -- a baseline column ------------------------------------------------------

#: obsm keys each tool writes, as (intrinsic, second latent). resolVI's
#: second block is its per-cell *mixture proportion* (true / diffusion /
#: background), not a spatial latent: it occupies the same slot so the
#: "is the second channel free of cell-intrinsic state?" reads still apply.
LATENT_KEYS = {
    "resolvi": ("X_resolvi", "X_resolvi_mixture"),
    "simvi": ("X_simvi_intrinsic", "X_simvi_spatial"),
    "mintflow": ("X_mintflow_intrinsic", "X_mintflow_micro_in"),
}


def load_latents(path: str | Path, tool: str | None = None):
    """(rows, intrinsic, spatial, config) from a tool's ``latents.h5ad``."""
    import anndata as ad

    adata = ad.read_h5ad(path)
    rows = np.array([int(str(s).split("_")[-1]) for s in adata.obs_names],
                    dtype=np.int64)
    keys = LATENT_KEYS.get(tool or "", (None, None))
    if keys[0] not in adata.obsm:                   # recognise the tool by its
        keys = next((v for v in LATENT_KEYS.values()   # own obsm keys instead
                     if v[0] in adata.obsm), (sorted(adata.obsm)[0], None))
    spatial = adata.obsm[keys[1]] if keys[1] in adata.obsm else None
    intrinsic_key = keys[0]
    cfg_path = Path(path).parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return rows, np.asarray(adata.obsm[intrinsic_key]), \
        (None if spatial is None else np.asarray(spatial)), cfg


def baseline_column(dataset: str, method: str, latents: str, config: dict,
                    c: np.ndarray, data, train: np.ndarray, test: np.ndarray,
                    cycle: dict | None, seed: int = 0) -> dict:
    tool = method.split("/")[0].split("_")[0]
    rows, z, spatial, cfg = load_latents(latents, tool)
    if rows.max() >= data.n_cells:
        raise ValueError("latents carry cell ids outside the dataset: the "
                         "export and the battery disagree about the split")
    sub_cycle = None
    if cycle is not None:
        sub_cycle = {"types": cycle["types"], "scores": cycle["scores"][rows]}
    reads = battery(z, data.t[rows], train[rows], test[rows],
                    data.v_block[rows], data.vbar_t, c[rows],
                    spatial=spatial, cycle=sub_cycle, seed=seed,
                    n_comp=data.n_comp)

    decoded = Path(latents).parent / "decoded_heldout.npz"
    rate, dec_rows = None, None
    if decoded.exists():
        blob = np.load(decoded)
        dec_rows, rate = blob["rows"], blob["rate"]
    if dec_rows is None:
        dec_rows = rows[test[rows]]
        if len(dec_rows) > MAX_RECON_CELLS:
            dec_rows = np.sort(np.random.default_rng(0).choice(
                dec_rows, MAX_RECON_CELLS, replace=False))
    reads["reconstruction"] = reconstruction(data.x, data.t, train, dec_rows,
                                             rate)
    reads["config"] = cfg
    return reads


# -- artefacts --------------------------------------------------------------

ROWS = [
    ("cells scored / held out", lambda r: f"{r['n_cells']:,} / {r['n_heldout']:,}"),
    ("intrinsic dim", lambda r: str(r["d_intrinsic"])),
    ("spatial dim", lambda r: str(r["d_spatial"] or "--")),
    ("NMI(z, type)", lambda r: f"{r['nmi']:.3f}"),
    ("probe dCE (pooled, legacy)", lambda r: f"{r['probe']['delta_ce']:+.3f}"),
    ("probe floor (pooled, legacy)",
     lambda r: f"{r['probe']['noise_floor']:+.3f}"),
    ("probe gain comp/ridge", lambda r: _gain(r, "ridge", "comp")),
    ("probe gain img/ridge", lambda r: _gain(r, "ridge", "img")),
    ("probe gain comp/mlp", lambda r: _gain(r, "mlp", "comp")),
    ("probe gain img/mlp", lambda r: _gain(r, "mlp", "img")),
    ("invariance guard (all four blocks)",
     lambda r: "pass" if r["probe_blocks"]["invariance_pass"] else "FAIL"),
    ("mirror R2", lambda r: f"{r['mirror']['r2']:.3f}"),
    ("mirror R2 (permuted)", lambda r: f"{r['mirror']['r2_permuted']:.3f}"),
    ("cycle R2, intrinsic", lambda r: _cyc(r, "z", "r2_pooled")),
    ("cycle R2, permuted", lambda r: _cyc(r, "z", "r2_permuted")),
    ("cycle R2, spatial", lambda r: _cyc(r, "spatial", "r2_pooled")),
    ("I(z;t)/H(t)", lambda r: f"{r['degeneracy']['mi_ratio']:.3f}"),
    ("within-type var fraction",
     lambda r: f"{r['degeneracy']['within_var_fraction']:.3f}"),
    ("spatial var fraction",
     lambda r: f"{r['spatial_var_fraction']:.3f}" if "spatial_var_fraction" in r
     else "--"),
    ("recon (log-lik / count)",
     lambda r: f"{r['reconstruction']['recon']:.3f}"
     if "recon" in r.get("reconstruction", {}) else "no decoder"),
    ("type-profile reference",
     lambda r: f"{r['reconstruction']['recon_type_profile']:.3f}"),
]


def _gain(reads: dict, family: str, block: str) -> str:
    """``gain (floor mean +- sd) pass|FAIL`` of one probe block, in nats."""
    b = reads["probe_blocks"][family][block]
    return (f"{b['gain']:+.4f} (floor {b['floor_mean']:+.4f} ± "
            f"{b['floor_sd']:.4f}) {'pass' if b['pass'] else 'FAIL'}")


def _cyc(reads: dict, which: str, field: str) -> str:
    block = (reads.get("cycle") or {}).get(which)
    if not block or not np.isfinite(block.get(field, np.nan)):
        return "--"
    return f"{block[field]:.3f}"


def markdown(entries: dict, dataset: str, tag: str) -> str:
    methods = list(entries)
    lines = [f"# Baseline battery -- {dataset}{tag}", "",
             "One column per method, every read computed by "
             "`discell/experiments/baseline_battery.py` on the same held-out "
             "cells, the same label key and the same cycle scores.", "",
             "| read | " + " | ".join(methods) + " |",
             "|---|" + "---|" * len(methods)]
    for name, fmt in ROWS:
        cells = []
        for m in methods:
            try:
                cells.append(fmt(entries[m]))
            except Exception:
                cells.append("--")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Configurations:", ""]
    for m in methods:
        cfg = entries[m].get("config", {})
        lines.append(f"* **{m}** -- " + ", ".join(
            f"{k}={v}" for k, v in cfg.items()
            if k in ("tool", "run", "trained_on", "graph", "max_epochs",
                     "num_training_epochs", "epochs_completed", "truncated",
                     "subsampled", "transfer_mode", "train_s", "peak_gpu_gb",
                     "n_latent", "n_intrinsic", "n_spatial", "kappa", "d_z",
                     "d_w", "seed")))
    return "\n".join(lines) + "\n"


def artefacts(dataset: str, tag: str):
    from discell import paths

    root = paths.dataset(dataset).root / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"baseline_battery{tag}.json", \
        root / f"baseline_battery{tag}.md"


def merge_and_write(dataset: str, tag: str, method: str, reads: dict) -> None:
    js, md = artefacts(dataset, tag)
    entries = json.loads(js.read_text()) if js.exists() else {}
    entries[method] = reads
    js.write_text(json.dumps(entries, indent=2, default=str))
    md.write_text(markdown(entries, dataset, tag))
    log.info("wrote %s and %s (%d columns)", js, md, len(entries))


# -- CLI --------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config-from", default=None,
                   help="dataset whose run configuration and weights apply "
                        "(the held-out-section protocol)")
    p.add_argument("--config-run", default="best")
    p.add_argument("--discell-run", default=None,
                   help="evaluate this DisCell run's own latents")
    p.add_argument("--method", default=None, help="column name for --latents")
    p.add_argument("--latents", default=None)
    p.add_argument("--tag", default="", help="suffix for the artefact names")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source = args.config_from or args.dataset
    ctx = context_path(args.dataset, args.config_run,
                       "" if source == args.dataset else f"_from_{source}")

    if args.discell_run:
        reads, c = discell_column(args.dataset, args.discell_run,
                                  args.config_from, device=args.device)
        np.savez_compressed(ctx, c=c.astype(np.float32))
        log.info("cached context %s %s", ctx, c.shape)
        merge_and_write(args.dataset, args.tag,
                        args.method or f"DisCell/{args.discell_run}", reads)
        return 0

    if not (args.method and args.latents):
        p.error("give either --discell-run, or --method with --latents")
    config = run_config(source, args.config_run)
    data, train, test, cycle = shared_side(args.dataset, config)
    c = load_context(ctx)
    if len(c) != data.n_cells:
        raise ValueError(f"cached context has {len(c)} rows, dataset has "
                         f"{data.n_cells}")
    reads = baseline_column(args.dataset, args.method, args.latents, config,
                            c, data, train, test, cycle, seed=config["seed"])
    merge_and_write(args.dataset, args.tag, args.method, reads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
