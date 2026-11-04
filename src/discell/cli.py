"""The ``discell`` command line: thin subcommands over the Python API.

::

    discell ingest OUTS -o section.h5ad [--labels CSV] [--donors D ...]
    discell embed section.h5ad [--image TIFF] [--large-cells embed|zero]   # [image] extra
    discell fit section.h5ad --label-key lineage -o fits/k0.1_s0 [--kappa 0.1 --seed 0 ...]
    discell sweep section.h5ad --label-key lineage -o fits/ [--kappas ... --seeds ...]
    discell breakdown fits/ section.h5ad [--lr-genes lr.txt] [-o breakdown.json]
                      [--kappa-bar model|counts|none|VALUE]
    discell relocation fits/k0.1_s0 section.h5ad -o relocation.json

``fit`` and ``sweep`` build the graph for ``--label-key`` when the AnnData has
none for it. ``--config FILE.json`` passes any :class:`discell.Config` field;
flags given on the command line override it. Every file the commands write is
named by an argument; nothing is written next to the inputs unless asked.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger("discell.cli")

#: Config fields exposed as flags (the others through --config).
FLAGS = {"kappa": float, "seed": int, "image_key": str, "alpha_z": str, "alpha_a": float,
         "tile_cells": int, "epochs": int, "patience": int, "device": str,
         "counts_layer": str, "unassigned_label": str}


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def _read(path):
    import anndata as ad

    return ad.read_h5ad(path)


def _config(args):
    from discell.model.config import Config

    record = json.loads(Path(args.config).read_text()) if args.config else {}
    for name in FLAGS:
        value = getattr(args, name, None)
        if value is not None:
            record[name] = value
    if args.label_key is not None:
        record["label_key"] = args.label_key
    if "label_key" not in record:
        raise SystemExit("discell: --label-key (or label_key in --config) is required")
    alpha_z = record.get("alpha_z")
    if isinstance(alpha_z, str) and alpha_z != "auto":
        record["alpha_z"] = float(alpha_z)
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(record) - known
    if unknown:
        raise SystemExit(f"discell: unknown config fields {sorted(unknown)}")
    return Config(**record)


def _ensure_graph(adata, config) -> None:
    from discell.graph.kernel import build_graph

    meta = adata.uns.get("discell", {}).get("graph")
    if (meta is None or meta.get("label_key") != config.label_key
            or meta.get("unassigned_label") != config.unassigned_label):
        build_graph(adata, config.label_key, unassigned_label=config.unassigned_label)


# -- subcommands ---------------------------------------------------------------

def cmd_ingest(args) -> int:
    from discell.io.xenium import read_xenium

    labels = None if args.labels == "none" else args.labels
    region = tuple(args.region) if args.region else None
    adata = read_xenium(args.outs, labels=labels, clusters=not args.no_clusters,
                        donors=args.donors, donor_key=args.donor_key, region_um=region)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes -> {args.out}")
    return 0


def cmd_embed(args) -> int:
    from discell.image import embed

    adata = _read(args.adata)
    embed(adata, args.image, key=args.key, large_cells=args.large_cells, device=args.device,
          batch_size=args.batch_size, cache_dir=args.cache_dir)
    out = args.out or args.adata
    adata.write_h5ad(out)
    print(f"{args.key}: {adata.obsm[args.key].shape} -> {out}")
    return 0


def cmd_fit(args) -> int:
    from discell.model.model import fit

    adata = _read(args.adata)
    config = _config(args)
    _ensure_graph(adata, config)
    model = fit(adata, config, out_dir=args.out)
    if args.write_h5ad:
        adata.write_h5ad(args.write_h5ad)
    s = model.summary()
    print(f"best epoch {s['best_epoch']}, recon_val {s['recon_val']:.4f}, NMI {s['nmi']:.4f},"
          f" I(niche; w) excess {s['niche_information_check']:.4f} -> {args.out}")
    return 0


def cmd_sweep(args) -> int:
    from discell.sensitivity.breakdown import KAPPA_GRID
    from discell.sensitivity.sweep import SEEDS, fit_sweep, run_dir

    adata = _read(args.adata)
    config = _config(args)
    _ensure_graph(adata, config)
    kappas = _floats(args.kappas) if args.kappas else list(KAPPA_GRID)
    seeds = _ints(args.seeds) if args.seeds else list(SEEDS)
    models = fit_sweep(adata, config, kappas, seeds, out_dir=args.out)
    for (kappa, seed), model in models.items():
        best = (model.metrics or {}).get("best", {}).get("epoch")
        print(f"kappa {kappa:g} seed {seed}: best epoch {best} -> "
              f"{run_dir(args.out, kappa, seed)}")
    return 0


def _fit_dirs(root: Path) -> dict[tuple[float, int], Path]:
    """``{(kappa, seed): directory}`` of every saved fit under *root*."""
    import torch

    found = {}
    for where in sorted(p.parent for p in Path(root).glob("*/model.pt")):
        cfg = torch.load(where / "model.pt", map_location="cpu", weights_only=True)["config"]
        found[(float(cfg["kappa"]), int(cfg["seed"]))] = where
    return found


def cmd_breakdown(args) -> int:
    from discell.model.model import Model
    from discell.sensitivity.breakdown import KAPPA_GRID
    from discell.sensitivity.findings import GROUPS, finding_breakdown, fit_draws
    from discell.sensitivity.upper_bound import decide

    adata = _read(args.adata)
    lr = None
    if args.lr_genes:
        lr = [g.strip() for g in Path(args.lr_genes).read_text().split() if g.strip()]
    groups = args.groups.split(",") if args.groups else list(GROUPS)
    fits = _fit_dirs(args.fits)
    if not fits:
        raise SystemExit(f"discell: no saved fits under {args.fits}")
    dirs = {}
    for (kappa, seed), where in sorted(fits.items()):
        dirs[(kappa, seed)] = where / "draws"
        model = None
        if not all((where / "draws" / f"{g}.npz").exists() for g in groups
                   if g != "signaling" or lr is not None):
            model = Model.load(where, device=args.device)
        if model is None:           # every draw file is already there
            continue
        _ensure_graph(adata, model.config)
        fit_draws(model, adata, groups=groups, lr_genes=lr, n=args.n,
                  out_dir=where / "draws")
        print(f"kappa {kappa:g} seed {seed}: draws -> {where / 'draws'}")
    grid = _floats(args.kappas) if args.kappas else sorted({k for k, _ in fits})
    seeds = _ints(args.seeds) if args.seeds else sorted({s for _, s in fits})
    if args.kappas is None and tuple(grid) != KAPPA_GRID:
        log.info("kappa grid read from the fits: %s", grid)
    result = finding_breakdown(dirs, grid=grid, seeds=seeds)
    bound = _kappa_bar(args, adata, fits)
    if bound is not None:
        result["kappa_bar"] = bound
        result["decision"] = decide(result, bound["value"])
    text = json.dumps(result, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(f"M = {result['m_s']} findings (Bonferroni)")
    if bound is not None:
        print(f"kappa_bar = {bound['value']:.3f} ({bound['source']})")
    for name, member in result["members"].items():
        star = "" if member["kappa_star"] is None else f" kappa* = {member['kappa_star']:g}"
        role = member.get("role")
        note = "" if role is None else f"  [{role}]"
        verdict = "" if bound is None else result["decision"][name]["verdict"]
        print(f"{name}: {member['status']}{star}{note}" + (f"  -> {verdict}" if verdict else ""))
    return 0


def _kappa_bar(args, adata, fits) -> dict | None:
    """The section's estimated bound for ``breakdown`` (``--kappa-bar``)."""
    from discell.model.model import Model
    from discell.sensitivity.upper_bound import kappa_bar

    choice = args.kappa_bar
    if choice == "none":
        return None
    try:
        return {"value": float(choice), "source": "given"}
    except ValueError:
        pass
    models = []
    if choice == "model":
        models = [Model.load(where, device=args.device)
                  for (kappa, _), where in sorted(fits.items())
                  if abs(kappa - args.operating_kappa) < 1e-9]
        if not models:
            log.warning("no fits at kappa = %g: kappa_bar from the counts only",
                        args.operating_kappa)
    elif choice != "counts":
        raise SystemExit(f"discell: --kappa-bar must be model, counts, none or a number, "
                         f"not {choice!r}")
    config = models[0].config if models else Model.load(next(iter(fits.values()))).config
    _ensure_graph(adata, config)
    record = kappa_bar(adata, models, device=args.device)
    version = record["model_based"] if models else record["counts_only"]
    if version["kappa_bar"] is None:
        log.warning("no type has enough connected cells for kappa_bar")
        return None
    source = (f"model-based, mean over {len(models)} fits at kappa = {args.operating_kappa:g}"
              if models else "counts only")
    return {"value": version["kappa_bar"], "source": source, "type": version["type"],
            "record": record}


def cmd_relocation(args) -> int:
    from discell import tl
    from discell.model.model import Model

    adata = _read(args.adata)
    model = Model.load(args.fit, device=args.device)
    _ensure_graph(adata, model.config)
    model.encode(adata)
    out = tl.relocation(model, adata)
    Path(args.out).write_text(json.dumps(out, indent=1, default=_jsonable))
    if args.write_h5ad:
        adata.write_h5ad(args.write_h5ad)
    print(f"relocation -> {args.out}")
    return 0


def _jsonable(value):
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# -- parser --------------------------------------------------------------------

def _config_flags(p) -> None:
    p.add_argument("--label-key", help=".obs column of cell types (required)")
    p.add_argument("--config", help="JSON file of discell.Config fields")
    for name, kind in FLAGS.items():
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=kind, default=None)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="discell", description=__doc__.split("\n")[0])
    top.add_argument("-v", "--verbose", action="store_true", help="log progress")
    sub = top.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Xenium output directory -> h5ad")
    p.add_argument("outs")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--labels", default="auto",
                   help="cell_id + label columns CSV; 'auto' = a *cell_groups.csv in OUTS; "
                        "'none'")
    p.add_argument("--donors", nargs="+", default=None, help="keep only these donors")
    p.add_argument("--donor-key", default="donor")
    p.add_argument("--region", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                   help="keep cells in this box (um)")
    p.add_argument("--no-clusters", action="store_true", help="skip the 10x clusterings")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("embed", help="per-cell KRONOS2 descriptor ([image] extra)")
    p.add_argument("adata")
    p.add_argument("--image", default=None,
                   help="morphology OME-TIFF (default: morphology_focus_0000 of the outs)")
    p.add_argument("-o", "--out", default=None, help="h5ad to write (default: in place)")
    p.add_argument("--key", default="X_kronos_masked")
    p.add_argument("--large-cells", choices=("embed", "zero"), default="embed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cache-dir", default=None, help="Hugging Face cache for the weights")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("fit", help="one fit")
    p.add_argument("adata")
    p.add_argument("-o", "--out", required=True, help="directory of the saved model")
    p.add_argument("--write-h5ad", default=None, help="also write the AnnData with the outputs")
    _config_flags(p)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("sweep", help="fits over the kappa grid and seeds (resumable)")
    p.add_argument("adata")
    p.add_argument("-o", "--out", required=True, help="directory of k<kappa>_s<seed> fits")
    p.add_argument("--kappas", default=None, help="comma list (default: the paper's grid)")
    p.add_argument("--seeds", default=None, help="comma list (default 0,1,2)")
    _config_flags(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("breakdown", help="finding draws of every fit, then kappa*")
    p.add_argument("fits", help="directory of a sweep")
    p.add_argument("adata")
    p.add_argument("-o", "--out", default=None, help="JSON of the breakdown table")
    p.add_argument("--lr-genes", default=None,
                   help="ligand-receptor genes, whitespace-separated (signaling member)")
    p.add_argument("--groups", default=None, help="comma list of draw groups (default all)")
    p.add_argument("--n", type=int, default=None, help="draws per group")
    p.add_argument("--kappas", default=None, help="the grid (default: the fits' kappas)")
    p.add_argument("--seeds", default=None, help="the seeds (default: the fits' seeds)")
    p.add_argument("--kappa-bar", default="model",
                   help="the section's estimated bound kappa_bar to read kappa* against: "
                        "'model' (from the fits at --operating-kappa), 'counts' (counts only), "
                        "a number, or 'none'")
    p.add_argument("--operating-kappa", type=float, default=0.1,
                   help="the fits the model-based kappa_bar is averaged over (default 0.1)")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_breakdown)

    p = sub.add_parser("relocation", help="cell types moved between niches -> JSON")
    p.add_argument("fit", help="directory of one saved fit")
    p.add_argument("adata")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--write-h5ad", default=None)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_relocation)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
