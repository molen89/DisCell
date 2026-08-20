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

    python -m discell.preprocess.embedding_qc --sample <dir> --per-label 100
    python -m discell.preprocess.embedding_qc --sample <dir> --label-key cell_group \\
        --per-label 150 --out umap.png
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd


log = logging.getLogger("discell.preprocess.embedding_qc")

DEFAULT_PER_LABEL = 100

#: Below this, the pairwise statistics are noise rather than a measurement.
MIN_OVERLAP_CELLS = 50


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
