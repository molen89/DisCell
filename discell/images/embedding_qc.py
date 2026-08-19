#!/usr/bin/env python3
"""Does the image embedding know anything about cell type?

Takes a balanced sample of cells per label, embeds their image crops with
KRONOS, and projects to UMAP coloured by label. If the embedding carries
morphological signal that tracks cell identity, the classes separate; if it does
not, the plot is one cloud and that is worth knowing before wiring the
embeddings into a model.

Sampling is **stratified** -- N cells per label rather than N at random -- so
rare types are visible and the picture is not dominated by whatever is most
abundant.

Embeddings are **centred** before UMAP. Raw KRONOS vectors sit in a narrow cone
(all pairwise cosines ~0.995), which swamps the between-class structure;
subtracting the mean moves same-class similarity from +0.001 to +0.175 over
other-class on the lung sample.

Usage::

    python -m discell.images.embedding_qc --sample <dir> --per-label 100
    python -m discell.images.embedding_qc --sample <dir> --label-key cell_group \\
        --per-label 150 --out umap.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("discell.images.embedding_qc")

DEFAULT_PER_LABEL = 100


def stratified_sample(
    labels: np.ndarray, per_label: int, rng: np.random.Generator,
    min_label_count: int = 10,
) -> np.ndarray:
    """Up to *per_label* cell indices for each label with enough members."""
    picked: list[np.ndarray] = []
    counts = pd.Series(labels).value_counts()
    for label, total in counts.items():
        if total < min_label_count:
            log.info("  skipping %r: only %d cells", label, total)
            continue
        members = np.flatnonzero(labels == label)
        take = min(per_label, len(members))
        picked.append(rng.choice(members, take, replace=False))
    return np.sort(np.concatenate(picked))


#: Cells used for the pairwise separation statistics. The cosine matrix is
#: quadratic -- at 407k cells it would be 0.7 TB -- so the report is computed on
#: a stratified subsample and says so.
MAX_PAIRWISE_CELLS = 6000


def separation_report(embeddings: np.ndarray, labels: np.ndarray,
                      max_cells: int = MAX_PAIRWISE_CELLS,
                      seed: int = 0) -> dict:
    """Within- vs between-class cosine, and 1-NN agreement, on centred vectors.

    Centring happens on the *full* set before any subsampling, so the mean that
    is removed is the real one; only the pairwise matrix is subsampled.
    """
    centred = embeddings - embeddings.mean(axis=0)
    subsampled = len(labels) > max_cells
    if subsampled:
        rng = np.random.default_rng(seed)
        pick = stratified_sample(labels, max(max_cells // max(len(set(labels)), 1), 1),
                                 rng, min_label_count=1)
        centred, labels = centred[pick], labels[pick]

    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True).clip(1e-9)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, np.nan)
    same = labels[:, None] == labels[None, :]
    nearest = np.nanargmax(similarity, axis=1)
    n_classes = len(set(labels))
    return {
        "n_cells": len(labels),
        "n_classes": n_classes,
        "subsampled": subsampled,
        "cosine_same": float(np.nanmean(similarity[same])),
        "cosine_other": float(np.nanmean(similarity[~same])),
        "separation": float(np.nanmean(similarity[same]) - np.nanmean(similarity[~same])),
        "nn_agreement": float(np.mean(labels[nearest] == labels)),
        "chance": 1.0 / n_classes,
    }


def run_umap(embeddings: np.ndarray, seed: int = 0, n_neighbors: int = 15,
             min_dist: float = 0.1) -> np.ndarray:
    """UMAP on centred, standardised embeddings."""
    import umap

    centred = embeddings - embeddings.mean(axis=0)
    scaled = centred / centred.std(axis=0).clip(1e-9)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric="cosine", random_state=seed)
    started = time.time()
    coords = reducer.fit_transform(scaled)
    log.info("UMAP on %s in %.1fs", scaled.shape, time.time() - started)
    return np.asarray(coords)


def plot_umap(coords: np.ndarray, labels: np.ndarray, out_path: Path,
              palette: dict | None = None, title: str = "",
              subtitle: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = pd.Series(labels).value_counts().index.tolist()
    palette = dict(palette or {})
    cmap = plt.get_cmap("tab20" if len(order) <= 20 else "turbo")
    for k, label in enumerate(order):
        colour = palette.get(label)
        # Fill any label the shipped palette does not cover, and reject anything
        # matplotlib cannot read as a colour.
        if not isinstance(colour, (str, tuple)):
            palette[label] = (cmap(k % 20) if len(order) <= 20
                              else cmap(k / max(len(order) - 1, 1)))

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(13, 10), dpi=200)
    counts = {label: int((labels == label).sum()) for label in order}

    if len(labels) > 50_000:
        # Draw once in shuffled order: class-by-class would let whichever class
        # is plotted last blanket the ones before it, which at this density
        # invents structure that is not there.
        rng = np.random.default_rng(0)
        shuffled = rng.permutation(len(labels))
        colours = np.array([palette[label] for label in labels[shuffled]], dtype=object)
        ax.scatter(coords[shuffled, 0], coords[shuffled, 1],
                   s=0.6, alpha=0.35, c=list(colours), linewidths=0, rasterized=True)
        handles = [Line2D([], [], marker="o", linestyle="", markersize=7,
                          markerfacecolor=palette[label], markeredgecolor="none",
                          label=f"{label} ({counts[label]:,})") for label in order]
    else:
        for label in order:
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.85,
                       color=palette.get(label, "#999999"), linewidths=0.2,
                       edgecolors="white", label=f"{label} ({counts[label]:,})")
        handles = None
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False, ncol=1 if len(order) <= 26 else 2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", required=True)
    parser.add_argument("--model", default="v1", choices=("v1", "v2", "kronos", "kronos2"),
                        help="which image model to embed with when not reusing a .pt")
    parser.add_argument("--label-key", default=None, help="obs column; default is the sample's")
    parser.add_argument("--per-label", type=int, default=DEFAULT_PER_LABEL,
                        help="cells sampled per class; 0 uses every cell")
    parser.add_argument("--min-label-count", type=int, default=10)
    parser.add_argument("--channels", default="0,1,2,3")
    parser.add_argument("--patch-px", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--embeddings", default=None,
                        help="reuse a saved .pt instead of embedding again")
    parser.add_argument("--save-embeddings", default=None)
    parser.add_argument("--out", default="figures/kronos_umap.png")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    from discell.images.kronos import (
        XENIUM_MARKERS,
        describe_markers,
        embed_cells,
        fetch_marker_metadata,
        load_kronos,
        load_marker_metadata,
        resolve_markers,
        write_embeddings,
    )
    from discell.plotting.cell_graph import find_tissue_image, load_any_sample

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    rng = np.random.default_rng(args.seed)

    # graphs are irrelevant to embedding and cost ~4 min on a full slide
    adata, sample_dir = load_any_sample(args.sample, 30.0, args.max_cells,
                                        build_graphs=False)
    image_path = find_tissue_image(sample_dir)
    if image_path is None:
        raise SystemExit(f"no image under {sample_dir}")

    label_key = args.label_key or adata.uns.get("default_label")
    if label_key not in adata.obs:
        raise SystemExit(f"label {label_key!r} not in obs; have "
                         f"{list(adata.uns.get('label_columns', []))}")
    labels_all = (pd.Series(adata.obs[label_key]).astype(object)
                  .where(pd.Series(adata.obs[label_key]).notna(), "unassigned")
                  .astype(str).to_numpy())

    if args.per_label > 0:
        cells = stratified_sample(labels_all, args.per_label, rng, args.min_label_count)
    else:
        cells = np.arange(len(labels_all))
        log.info("Using every cell (--per-label 0)")
    labels = labels_all[cells]
    print(f"\nSample : {sample_dir.name}")
    print(f"Label  : {label_key}  ({len(set(labels))} classes)")
    print(f"Cells  : {len(cells):,} "
          f"({'every cell' if args.per_label <= 0 else f'{args.per_label} per class, stratified'})")

    if args.embeddings:
        payload = torch.load(args.embeddings, weights_only=False)
        model_name = str(payload.get("model", "kronos"))
        stored_markers = list(payload.get("marker_names", []))
        stored = np.asarray(payload["cell_ids"]).astype(str)
        lookup = pd.Series(np.arange(len(stored)), index=pd.Index(stored))
        position = lookup.reindex(pd.Index(adata.obs_names[cells].astype(str)))
        keep = position.notna().to_numpy()
        embeddings = payload["embeddings"].float().numpy()[position[keep].astype(int)]
        cells, labels = cells[keep], labels[keep]
        print(f"Reused {len(embeddings):,} embeddings from {args.embeddings}")
    else:
        if args.model in ("v2", "kronos2"):
            raise SystemExit(
                "embed with `python -m discell.images.kronos --model v2 --out emb2.pt` "
                "first, then pass --embeddings emb2.pt here")
        channels = [int(c) for c in args.channels.split(",")]
        specs = [m for m in XENIUM_MARKERS if m.channel in channels]
        metadata = load_marker_metadata(fetch_marker_metadata())
        markers = resolve_markers(specs, metadata)
        print()
        print(describe_markers(markers))
        model, precision, _dim = load_kronos()
        device = args.device if torch.cuda.is_available() else "cpu"
        embeddings, cell_ids = embed_cells(
            adata, image_path, markers, model, precision, cells=list(cells),
            patch_px=args.patch_px, batch_size=args.batch_size, device=device,
        )
        if args.save_embeddings:
            write_embeddings(args.save_embeddings, embeddings, cell_ids, meta={
                "label_key": label_key, "labels": list(map(str, labels)),
            })

    print()
    report = separation_report(embeddings, labels, seed=args.seed)
    print("Separation on centred embeddings:")
    print(f"  cosine  same-class {report['cosine_same']:+.4f}")
    print(f"  cosine other-class {report['cosine_other']:+.4f}")
    print(f"  separation         {report['separation']:+.4f}")
    print(f"  1-NN agreement     {100 * report['nn_agreement']:.1f}%  "
          f"(chance {100 * report['chance']:.1f}%, "
          f"{report['nn_agreement'] / report['chance']:.1f}x)")
    if report["subsampled"]:
        print(f"  [pairwise stats on a stratified {report['n_cells']:,}-cell subsample; "
              f"the full matrix would be {len(labels)**2 * 4 / 1e12:.1f} TB]")

    coords = run_umap(embeddings, args.seed, args.n_neighbors, args.min_dist)

    palette = None
    colour_col = f"{label_key}_color"
    if colour_col in adata.obs:
        # A categorical .astype(str) leaves missing entries as float NaN under
        # pandas 3.0, which reaches matplotlib as a colour and raises there.
        supplied = pd.Series(adata.obs[colour_col]).astype(object)
        supplied = supplied.where(supplied.notna(), "").astype(str).to_numpy()
        palette = {}
        for label, colour in zip(labels_all, supplied):
            if colour.startswith("#"):
                palette.setdefault(label, colour)
        missing = sorted(set(labels) - set(palette))
        if missing:
            log.info("no shipped colour for %d label(s): %s", len(missing), missing[:4])
            palette = None if len(missing) == len(set(labels)) else palette
        if palette:
            log.info("Using the palette shipped with %s", label_key)

    tag = {"kronos": "KRONOS v1", "kronos2": "KRONOS2"}.get(
        locals().get("model_name", "kronos"), locals().get("model_name", "KRONOS"))
    markers = locals().get("stored_markers") or []
    subtitle = (f"{tag} {embeddings.shape[1]}-d"
                + (f" · {len(markers)} ch ({', '.join(markers)})" if markers else "")
                + f" · {len(embeddings):,} cells · "
                f"1-NN {100 * report['nn_agreement']:.1f}% vs "
                f"{100 * report['chance']:.1f}% chance"
                + (f" (on {report['n_cells']:,})" if report["subsampled"] else ""))
    plot_umap(coords, labels, Path(args.out), palette,
              title=f"{sample_dir.name} — image embedding UMAP, coloured by {label_key}",
              subtitle=subtitle)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
