#!/usr/bin/env python3
"""A small graph model trained on the dataloader, as a worked example.

Not the model you want -- it exists to show the data contract end to end and to
give a floor to beat. The useful number is the comparison against the
neighbourhood-composition baseline, which predicts a cell's type from its
neighbours' types alone and uses no counts, geometry or morphology.

Usage::

    python -m discell.examples.train_demo --name ovarian_full
    python -m discell.examples.train_demo --name ovarian_full \\
        --embeddings embeddings/ovarian_kronos_all.pt --steps 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from discell.data.loader import CellGraphDataset, TorchBatch

log = logging.getLogger("discell.examples.train_demo")


class EdgeAwareLayer(nn.Module):
    """One message-passing step whose messages are modulated by edge geometry.

    Each edge carries ``[centroid_dist_um, apposed_wall_um]``. Those are turned
    into a scalar gate, so a neighbour sharing a long wall at close range counts
    for more than a distant one that merely shares Voronoi territory. That is the
    whole reason the geometry is carried through the pipeline.
    """

    def __init__(self, dim: int, edge_dim: int = 2) -> None:
        super().__init__()
        self.message = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(edge_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.update = nn.Linear(2 * dim, dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        weight = torch.sigmoid(self.gate(edge_attr))          # (n_edges, 1)
        messages = self.message(h)[src] * weight              # (n_edges, dim)

        pooled = torch.zeros_like(h)
        pooled.index_add_(0, dst, messages)
        counts = torch.zeros(h.shape[0], 1, device=h.device)
        counts.index_add_(0, dst, torch.ones_like(weight))
        pooled = pooled / counts.clamp(min=1.0)               # mean, not sum

        return F.relu(self.update(torch.cat([h, pooled], dim=1)))


class CellTypeModel(nn.Module):
    """Counts + neighbourhood geometry + image embedding -> cell type."""

    def __init__(self, n_genes: int, n_types: int, image_dim: int,
                 dim: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(n_genes, dim), nn.ReLU())
        self.layers = nn.ModuleList(EdgeAwareLayer(dim) for _ in range(layers))
        # Image features exist only for seed cells, so they join at the head
        # rather than being message-passed with the rest.
        self.image = nn.Sequential(nn.Linear(image_dim, dim), nn.ReLU())
        self.head = nn.Linear(2 * dim, n_types)

    def forward(self, batch: TorchBatch) -> torch.Tensor:
        # log1p is the one transform applied to the raw counts, here rather than
        # in the loader so what is stored on disk stays raw.
        h = self.encode(torch.log1p(batch.x))
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_attr)
        seeds = h[batch.seed_index]
        image = self.image(batch.image_embedding)
        return self.head(torch.cat([seeds, image], dim=1))


def composition_baseline(dataset: CellGraphDataset, batch: TorchBatch,
                         device: str) -> torch.Tensor:
    """Predict a seed's type from its neighbours' types alone.

    Uses only the precomputed K x K table -- no counts, no image, no geometry
    beyond who is adjacent. Any real model should beat this.
    """
    composition = dataset.composition_tensor(device)          # (K, K)
    src, dst = batch.edge_index[0], batch.edge_index[1]
    neighbour_types = torch.zeros_like(batch.y)
    neighbour_types.index_add_(0, dst, batch.y[src])
    seed_profile = neighbour_types[batch.seed_index]           # (n_seeds, K)
    seed_profile = seed_profile / seed_profile.sum(1, keepdim=True).clamp(min=1)
    # Score each candidate type by how well its expected neighbourhood matches.
    return seed_profile @ composition.T


def run(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.manual_seed(args.seed)

    print("=" * 74)
    print("1. LOAD BUNDLE")
    print("=" * 74)
    started = time.time()
    dataset = CellGraphDataset(
        args.bundle_dir, args.name,
        graph=args.graph,
        batch_cells=args.batch_cells,
        min_apposed_um=args.min_apposed_um,
        embeddings=args.embeddings,
        embedding_dim=args.embedding_dim,
        as_torch=True,
        device=device,
        # Everything resident on the GPU: batching becomes an index_select with
        # no host-to-device copy, ~3x the throughput of the transfer path.
        resident=None if args.resident == "none" else args.resident if args.resident != "auto" else (
            "gpu" if device == "cuda" else None),
        counts_dtype=args.counts_dtype,
        seed=args.seed,
    )
    print(f"   {dataset.adata.n_obs:,} cells x {len(dataset.gene_index):,} genes")
    print(f"   {len(dataset.edge_i):,} edges in the {args.graph!r} graph")
    print(f"   {dataset.n_types} cell types, {len(dataset)} batches of {args.batch_cells}")
    print(f"   image embeddings: {'real for ' if args.embeddings else 'none, zeros of width '}"
          f"{int(dataset.embedding_mask.sum()) if args.embeddings else dataset.embeddings.shape[1]}"
          f"{' cells' if args.embeddings else ''}")
    if dataset.resident is not None:
        import torch as _t
        print(f"   resident on {dataset.resident}: "
              f"{_t.cuda.memory_allocated() / 1e9:.2f} GB held" if device == "cuda"
              else f"   resident on {dataset.resident}")
    print(f"   loaded in {time.time() - started:.1f}s on {device}")

    print()
    print("=" * 74)
    print("2. NEIGHBOURHOOD COMPOSITION  (fixed, returned at construction)")
    print("=" * 74)
    top = dataset.composition_support.sort_values(ascending=False).head(5).index
    print(dataset.neighbour_composition.loc[top, top].round(3).to_string())
    print(f"\n   full table: {dataset.neighbour_composition.shape}, "
          f"as a tensor via dataset.composition_tensor()")

    print()
    print("=" * 74)
    print("3. A BATCH")
    print("=" * 74)
    batch = next(iter(dataset.batches(shuffle=False)))
    print(f"   {batch!r}\n")
    for name, tensor in (("x            ", batch.x), ("edge_index   ", batch.edge_index),
                         ("edge_attr    ", batch.edge_attr), ("y            ", batch.y),
                         ("image_embed  ", batch.image_embedding),
                         ("seed_index   ", batch.seed_index),
                         ("seed_comp    ", batch.seed_composition),
                         ("neigh_comp   ", batch.neighbour_composition)):
        fixed = "fixed" if name.strip() in (
            "image_embed", "seed_index", "seed_comp", "neigh_comp") else "varies"
        print(f"   {name} {str(tuple(tensor.shape)):>18}  {str(tensor.dtype):<14} {fixed}")
    print(f"\n   auxiliary (not model input):")
    print(f"   pos           {str(tuple(batch.pos.shape)):>18}  "
          f"{str(batch.pos.dtype):<14} microns, for plotting")
    print(f"\n   edge_attr columns: {batch.edge_attr_names}")
    print(f"   counts are raw: max={batch.x.max():.0f}, integral="
          f"{bool(torch.allclose(batch.x, batch.x.round()))}")

    print()
    print("=" * 74)
    print("4. TRAIN A SMALL EDGE-AWARE GNN")
    print("=" * 74)
    model = CellTypeModel(
        n_genes=len(dataset.gene_index), n_types=dataset.n_types,
        image_dim=dataset.embeddings.shape[1], dim=args.dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   {model.__class__.__name__}: {n_params:,} parameters on {device}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    started, losses = time.time(), []
    for step, batch in enumerate(dataset.batches(shuffle=True)):
        if step >= args.steps:
            break
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.seed_labels())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(float(loss))
        if step % max(args.steps // 6, 1) == 0:
            print(f"   step {step:>4}  loss {np.mean(losses[-20:]):.4f}")
    print(f"   {args.steps} steps in {time.time() - started:.1f}s")

    print()
    print("=" * 74)
    print("5. EVALUATE AGAINST THE COMPOSITION BASELINE")
    print("=" * 74)
    model.eval()
    hits = base_hits = total = 0
    with torch.no_grad():
        for step, batch in enumerate(dataset.batches(shuffle=True)):
            if step >= args.eval_batches:
                break
            truth = batch.seed_labels()
            hits += int((model(batch).argmax(1) == truth).sum())
            base_hits += int((composition_baseline(dataset, batch, device).argmax(1)
                              == truth).sum())
            total += int(truth.numel())
    majority = float(dataset.labels.sum(0).max() / dataset.labels.shape[0])
    print(f"   cells evaluated          {total:,}")
    print(f"   majority-class           {100 * majority:5.1f}%")
    print(f"   composition baseline     {100 * base_hits / total:5.1f}%   "
          f"(neighbour types only)")
    print(f"   this model               {100 * hits / total:5.1f}%   "
          f"(counts + geometry + image)")
    print("\n   A demo, not a result: {} steps on one sample, no train/test split."
          .format(args.steps))
    if not args.embeddings:
        print("   Image embeddings are zeros here -- rerun with --embeddings once")
        print("   KRONOS access lands to see what the morphology adds.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle-dir", default="bundles")
    parser.add_argument("--name", default="ovarian_full")
    parser.add_argument("--graph", default="contact")
    parser.add_argument("--batch-cells", type=int, default=128)
    parser.add_argument("--resident", default="auto",
                        choices=("auto", "gpu", "cuda", "cpu", "none"),
                        help="hold counts/labels/embeddings on this device; "
                             "auto uses the GPU when one is available")
    parser.add_argument("--counts-dtype", default="float32",
                        choices=("float32", "float16"))
    parser.add_argument("--min-apposed-um", type=float, default=None)
    parser.add_argument("--embeddings", default=None, help="KRONOS .pt / .npz")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
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
