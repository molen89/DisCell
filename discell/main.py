#!/usr/bin/env python3
"""Inspect what the dataloader serves.

Builds the dataset, pulls one batch, and prints every tensor with its shape,
dtype, device and -- the part that matters -- which axis it is aligned to.
Nothing else: no model, no training, no plots. Those live in
:mod:`discell.examples.train_demo` and :mod:`discell.plotting`.

Usage::

    python -m discell.main                          # dataset inferred when only one is bundled
    python -m discell.main --dataset xenium_prime_ovarian_cancer_ffpe --embeddings kronos1_v2scale
    python -m discell.main --batch-cells 10 --image-scope nodes
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from discell.data.loader import CellGraphDataset
from discell import paths

log = logging.getLogger("discell.main")

RULE = "=" * 78


def describe_batch(batch, dataset) -> None:
    """Print one batch field by field, grouped by the axis it follows."""
    groups = (
        ("NODE-ALIGNED  (seeds + their neighbours; length varies per batch)", (
            ("x", batch.x, "raw integer counts"),
            ("y", batch.y, "one-hot cell type"),
            ("pos", batch.pos, "centroid microns - auxiliary, for plotting"),
            ("node_index", batch.node_index, "row in the full dataset"),
        )),
        ("EDGE-ALIGNED  (neighbour -> seed; length varies per batch)", (
            ("edge_index", batch.edge_index, "(2, n_edges) local node positions"),
            ("edge_attr", batch.edge_attr, f"{batch.edge_attr_names}"),
        )),
        ("SEED-ALIGNED  (fixed at batch_cells)", (
            ("seed_index", batch.seed_index, "which node rows are seeds"),
            ("image_embedding", batch.image_embedding,
             f"KRONOS, scope={dataset.image_scope}"),
            ("seed_composition", batch.seed_composition, "each seed's type prior"),
        )),
        ("CONSTANT", (
            ("neighbour_composition", batch.neighbour_composition,
             "mean neighbour-type vector per type"),
        )),
    )
    for title, fields in groups:
        print(f"\n  {title}")
        for name, tensor, note in fields:
            if tensor is None:
                print(f"    {name:<22} (absent)")
                continue
            shape = str(tuple(tensor.shape))
            dtype = str(tensor.dtype).replace("torch.", "")
            print(f"    {name:<22} {shape:>16}  {dtype:<8}  {note}")

    print(f"\n  node_ids               list[{len(batch.node_ids)}]  e.g. {batch.node_ids[:2]}")
    print(f"  device                 {batch.x.device}")


def check_invariants(batch) -> None:
    """The properties a consumer is entitled to rely on."""
    src, dst = batch.edge_index[0], batch.edge_index[1]
    seeds = set(batch.seed_index.tolist())
    checks = [
        ("every edge points INTO a seed",
         all(int(v) in seeds for v in dst.tolist())),
        ("no self-loops", bool((src != dst).all())),
        ("edge endpoints are valid node rows",
         int(batch.edge_index.max()) < batch.n_nodes),
        ("counts are raw integers",
         bool(torch.allclose(batch.x, batch.x.round())) and float(batch.x.min()) >= 0),
        ("labels are one-hot", bool((batch.y.sum(1) == 1).all())),
        ("seed_composition rows sum to 1",
         bool(torch.allclose(batch.seed_composition.sum(1),
                             torch.ones_like(batch.seed_composition[:, 0]), atol=1e-5))),
        ("edge_attr distance matches pos",
         bool(torch.allclose((batch.pos[dst] - batch.pos[src]).norm(dim=1),
                             batch.edge_attr[:, 0], atol=1e-2))),
    ]
    print()
    for label, ok in checks:
        print(f"    [{'ok' if ok else 'FAIL'}] {label}")

def plot_batch(index, batch, dataset, out_dir: str, overlay: bool = False,
               dpi: int = 260) -> None:
    """Render the four inspection panels for this batch."""
    from discell.plotting.batch_view import plot_batch as render

    path = render(batch, dataset, Path(out_dir) / f"batch_{index}.png",
                  dpi=dpi, overlay=overlay)
    print(f"  wrote {path}")


def run(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    resident = (None if args.resident == "none"
                else args.resident if args.resident != "auto"
                else ("gpu" if device == "cuda" else None))

    ds, variant = paths.resolve_bundle(args)
    out_dir = Path(args.out_dir) if args.out_dir else ds.figures

    started = time.time()
    dataset = CellGraphDataset.from_dataset(
        ds, variant,
        graph=args.graph,
        batch_cells=args.batch_cells,
        label_key=args.label_key,
        edge_mode=args.edge_mode,
        image_scope=args.image_scope,
        min_apposed_um=args.min_apposed_um,
        embeddings=args.embeddings,
        embedding_dim=args.embedding_dim,
        as_torch=True,
        device=device,
        resident=resident,
        counts_dtype=args.counts_dtype,
        seed=args.seed,
    )

    print(RULE)
    print(f"DATASET  {ds.dataset_id}  (bundle {variant!r})")
    print(RULE)
    print(f"  {dataset.adata.n_obs:,} cells x {len(dataset.gene_index):,} genes, "
          f"{dataset.n_types} types")
    print(f"  {len(dataset.edge_i):,} edges in the {args.graph!r} graph "
          f"(edge_mode={args.edge_mode})")
    print(f"  {len(dataset)} batches of {args.batch_cells} seed cells")
    covered = 100 * dataset.embedding_mask.mean()
    print(f"  image embeddings: {covered:.1f}% real, dim {dataset.embeddings.shape[1]}")
    if dataset.resident is not None:
        held = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
        print(f"  resident on {dataset.resident}"
              + (f", {held:.2f} GB held" if held else ""))
    print(f"  ready in {time.time() - started:.1f}s")

    for index, batch in enumerate(dataset.batches(shuffle=not args.no_shuffle)):
        print()
        print(RULE)
        print(f"BATCH {index}   {batch.n_seeds} seeds -> {batch.n_nodes} nodes, "
              f"{batch.edge_index.shape[1]} edges")
        print(RULE)
        describe_batch(batch, dataset)
        check_invariants(batch)

        if not args.no_plot:
            plot_batch(index, batch, dataset, out_dir, not args.no_overlay, args.dpi)

        break   # one batch is enough to inspect the contract



    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    paths.add_dataset_args(parser)
    parser.add_argument("--graph", default="contact")
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--batch-cells", type=int, default=10)
    parser.add_argument("--edge-mode", default="to_seed",
                        choices=("to_seed", "induced", "star"))
    parser.add_argument("--image-scope", default="seeds", choices=("seeds", "nodes"))
    parser.add_argument("--min-apposed-um", type=float, default=None)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--resident", default="auto",
                        choices=("auto", "gpu", "cuda", "cpu", "none"))
    parser.add_argument("--counts-dtype", default="float32",
                        choices=("float32", "float16"))
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--no-overlay", action="store_true",
                        help="skip the morphology image behind the tissue map")
    parser.add_argument("--out-dir", default=None,
                        help="default: this dataset's figures directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
