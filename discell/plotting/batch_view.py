#!/usr/bin/env python3
"""Four views of a single dataloader batch, for inspection.

Each panel answers one question about what the loader just served:

1. **label distribution** -- which cell types are in the batch, split by whether
   a node is a seed or context
2. **the batch in space** -- nodes at their real micron coordinates, edge width
   by shared wall, node colour by expression magnitude
3. **image-embedding similarity** -- do the seeds look alike to KRONOS
4. **neighbourhood, prior vs observed** -- the dataset-wide expectation for each
   seed's type against what this batch actually contains

Colours follow the reference palette: categorical slots 1-2 for the two-series
comparisons, the blue sequential ramp for magnitude, and the blue-red diverging
pair for cosine, which is signed and needs a neutral midpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("discell.plotting.batch_view")

#: Mirrors the loader: cells with no annotation are an explicit class.
UNASSIGNED = "unassigned"

# Reference palette, used unchanged.
SERIES_1 = "#2a78d6"      # blue   - prior / seeds
SERIES_2 = "#eb6834"      # orange - observed / context
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: Same ramp truncated for **discrete marks**. The full ramp's lightest steps sit
#: within a hair of the surface, so a low-count node effectively disappears; the
#: palette's ordinal rule requires starting no lighter than step 250 (2.06:1).
MARK_BLUE = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#ec835a", "#d03b3b", "#7d1d1d"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def type_palette(dataset) -> dict:
    """Colour per cell type.

    Prefers the palette published with the annotation (Xenium's cell_groups.csv
    ships one), so these figures agree with 10x's own and with the embedding
    UMAP. Only when none exists does it fall back to the reference categorical
    slots -- and past eight types to a continuous map, which is a knowingly weak
    encoding that the direct labels and per-tile titles have to carry.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    names = list(dataset.type_names)
    palette: dict[str, object] = {}

    colour_col = f"{dataset.label_key}_color"
    if colour_col in dataset.adata.obs:
        supplied = pd.Series(dataset.adata.obs[colour_col]).astype(object)
        supplied = supplied.where(supplied.notna(), "").astype(str).to_numpy()
        labels = pd.Series(dataset.adata.obs[dataset.label_key]).astype(object)
        labels = labels.where(labels.notna(), UNASSIGNED).astype(str).to_numpy()
        for label, colour in zip(labels, supplied):
            if colour.startswith("#"):
                palette.setdefault(label, colour)

    missing = [n for n in names if n not in palette]
    if missing:
        if len(names) <= 8:
            slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
            for k, name in enumerate(missing):
                palette[name] = slots[k % len(slots)]
        else:
            cmap = plt.get_cmap("turbo")
            for k, name in enumerate(missing):
                palette[name] = cmap(k / max(len(missing) - 1, 1))
    return palette


def _node_types(batch, dataset):
    """Type name per node, and the colour for each."""
    names = list(dataset.type_names)
    indices = batch.y.argmax(1).cpu().numpy()
    return [names[i] for i in indices], indices


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)
    ax.title.set_color(INK)


def _short(name: str, width: int = 22) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def _panel_labels(ax, batch, type_names) -> None:
    """Which types are present, seeds versus context."""
    y = batch.y.cpu().numpy()
    seed_rows = batch.seed_index.cpu().numpy()
    is_seed = np.zeros(len(y), dtype=bool)
    is_seed[seed_rows] = True
    types = y.argmax(1)

    present = sorted(set(types.tolist()),
                     key=lambda t: -(types == t).sum())
    seeds = np.array([((types == t) & is_seed).sum() for t in present])
    context = np.array([((types == t) & ~is_seed).sum() for t in present])

    positions = np.arange(len(present))
    width = 0.38
    ax.bar(positions - width / 2 - 0.01, seeds, width, color=SERIES_1,
           label="seed", zorder=3)
    ax.bar(positions + width / 2 + 0.01, context, width, color=SERIES_2,
           label="context (neighbour)", zorder=3)
    for x, value in zip(positions - width / 2 - 0.01, seeds):
        if value:
            ax.text(x, value, str(int(value)), ha="center", va="bottom",
                    fontsize=7, color=INK_2)
    for x, value in zip(positions + width / 2 + 0.01, context):
        if value:
            ax.text(x, value, str(int(value)), ha="center", va="bottom",
                    fontsize=7, color=INK_2)

    ax.set_xticks(positions)
    ax.set_xticklabels([_short(type_names[t], 18) for t in present],
                       rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("cells", color=INK_2, fontsize=8)
    ax.set_title("1 · label distribution", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)


def _tissue_bounds(dataset):
    """Whole-tissue extent in microns -- fixed, independent of the batch."""
    positions = dataset.positions_um
    return (float(positions[:, 0].min()), float(positions[:, 0].max()),
            float(positions[:, 1].min()), float(positions[:, 1].max()))


def _tissue_image(dataset, bounds, mpp, max_px=4000):
    """The morphology image over the whole tissue, in micron extent."""
    from pathlib import Path as _Path

    from discell.plotting.cell_graph import (
        XENIUM_DEFAULT_CHANNELS, find_tissue_image, is_xenium_morphology,
        read_image_window,
    )

    directory = dataset.adata.uns.get("xenium_dir") or dataset.adata.uns.get("segmented_dir")
    if not directory:
        return None
    path = find_tissue_image(_Path(directory))
    if path is None:
        return None
    x0, x1, y0, y1 = bounds
    channels = XENIUM_DEFAULT_CHANNELS if is_xenium_morphology(path) else None
    try:
        window, _ = read_image_window(
            path, int(x0 / mpp), int(y0 / mpp), int(x1 / mpp), int(y1 / mpp),
            max_px=max_px, channels=channels,
        )
    except Exception as exc:                       # missing channel file, etc.
        log.warning("no tissue overlay (%s): %s", type(exc).__name__, exc)
        return None
    return window


def _panel_graph_grid(axes, batch, dataset, palette) -> int:
    """One tile per seed: its neighbourhood in seed-relative microns.

    Re-centring each seed keeps the real local geometry -- distances and angles
    are untouched -- while making it legible. At whole-tissue scale a 10 um edge
    is sub-pixel, which is why the tissue map is a separate figure.
    """
    from matplotlib.collections import LineCollection

    pos = batch.pos.cpu().numpy()
    src = batch.edge_index[0].cpu().numpy()
    dst = batch.edge_index[1].cpu().numpy()
    wall = batch.edge_attr[:, 1].cpu().numpy()
    seed_rows = batch.seed_index.cpu().numpy()
    names, _ = _node_types(batch, dataset)

    scale = np.percentile(wall, 95) if len(wall) else 1.0
    # One radius for every tile, so tile-to-tile distances are comparable.
    radius = 1.0
    for row in seed_rows:
        for a in src[dst == row]:
            radius = max(radius, float(np.linalg.norm(pos[a] - pos[row])))
    radius *= 1.25

    for tile, (ax, row) in enumerate(zip(axes, seed_rows)):
        incoming = src[dst == row]
        origin = pos[row]
        if len(incoming):
            offsets = pos[incoming] - origin
            ax.add_collection(LineCollection(
                [[(0.0, 0.0), tuple(o)] for o in offsets],
                linewidths=0.6 + 3.4 * np.clip(wall[dst == row] / (scale or 1), 0, 1),
                colors=INK_2, alpha=0.55, zorder=2))
            ax.scatter(offsets[:, 0], offsets[:, 1], s=90,
                       c=[palette[names[a]] for a in incoming],
                       edgecolors=SURFACE, linewidths=1.0, zorder=3)
        ax.scatter([0], [0], s=210, c=[palette[names[row]]],
                   edgecolors=INK, linewidths=2.0, zorder=4)

        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{_short(names[row], 20)}  ·  deg {len(incoming)}",
                     fontsize=7, loc="left", color=INK)
        _style(ax)

    for ax in axes[len(seed_rows):]:
        ax.set_axis_off()
    return int(round(radius))


def _panel_tissue(ax, batch, dataset, fig, palette, overlay: bool = True,
                  seeds_only: bool = True) -> None:
    """Where this batch sits on the slide, over the morphology image."""
    pos = batch.pos.cpu().numpy()
    seed_rows = batch.seed_index.cpu().numpy()
    names, _ = _node_types(batch, dataset)
    mpp = float(dataset.adata.uns.get("microns_per_pixel", 1.0))

    x0, x1, y0, y1 = _tissue_bounds(dataset)
    if overlay:
        image = _tissue_image(dataset, (x0, x1, y0, y1), mpp)
        if image is not None:
            ax.imshow(image, extent=(x0, x1, y1, y0), zorder=0)

    is_seed = np.zeros(len(pos), dtype=bool)
    is_seed[seed_rows] = True
    colours = [palette[n] for n in names]

    if not seeds_only:
        ax.scatter(pos[~is_seed, 0], pos[~is_seed, 1], s=26,
                   c=[colours[i] for i in np.flatnonzero(~is_seed)],
                   edgecolors=SURFACE, linewidths=0.5, alpha=0.95, zorder=3,
                   label="context (neighbour)")
    # Seeds only by default: a neighbour sits ~10 um from its seed, which is
    # sub-pixel across an 11 mm slide, so context marks land on top of their own
    # seed and add nothing but clutter. The graph tiles are where they are read.
    ax.scatter(pos[is_seed, 0], pos[is_seed, 1], s=150,
               c=[colours[i] for i in np.flatnonzero(is_seed)],
               edgecolors="#ffffff", linewidths=2.0, zorder=4, label="seed")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    # adjustable="box": the tissue keeps its true proportions and the axes
    # shrinks to fit. "datalim" would pad the limits instead, floating a small
    # map in whitespace and breaking the fixed extent.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (µm)", color=INK_2, fontsize=9)
    ax.set_ylabel("y (µm)", color=INK_2, fontsize=9)
    kind = "seed centroids" if seeds_only else "all centroids"
    ax.set_title(f"{kind} on the tissue  ·  {x1 - x0:,.0f} × {y1 - y0:,.0f} µm"
                 f"  ·  fixed extent", fontsize=11, loc="left")
    _style(ax)


def _panel_cosine(ax, batch, dataset, fig, type_names) -> None:
    """Cosine between seed image embeddings, on centred vectors."""
    from matplotlib.colors import (LinearSegmentedColormap, Normalize,
                                   TwoSlopeNorm)

    embeddings = batch.image_embedding.cpu().numpy()
    seed_rows = batch.seed_index.cpu().numpy()
    if embeddings.shape[0] == batch.n_nodes:        # image_scope='nodes'
        embeddings = embeddings[seed_rows]

    if not np.any(embeddings):
        ax.text(0.5, 0.5, "no image embeddings\n(run with --embeddings)",
                ha="center", va="center", color=INK_2, fontsize=9)
        ax.set_axis_off()
        ax.set_title("3 · image-embedding cosine", fontsize=10, loc="left")
        return

    # Raw KRONOS vectors sit in a narrow cone -- every pair reads ~0.995 and the
    # panel would be uniformly one colour. Remove the dataset mean first.
    centred = embeddings - dataset.embeddings.mean(axis=0, keepdims=True)
    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True).clip(1e-9)
    similarity = unit @ unit.T

    off_diagonal = similarity[~np.eye(len(similarity), dtype=bool)]
    if off_diagonal.size == 0:
        # A single seed has no pair to compare against.
        ax.text(0.5, 0.5, "one seed in the batch\nno pair to compare",
                ha="center", va="center", color=INK_2, fontsize=9)
        ax.set_axis_off()
        ax.set_title("3 · image-embedding cosine between seeds",
                     fontsize=10, loc="left")
        return
    low, high = float(off_diagonal.min()), float(off_diagonal.max())

    # The diagonal is 1.0 by construction and would pin the top of any scale,
    # so it is masked out rather than allowed to flatten the real contrast.
    shown = similarity.copy().astype(float)
    np.fill_diagonal(shown, np.nan)

    if low < -0.05 and high > 0.05:
        # Genuinely signed: diverging with a neutral zero.
        limit = max(abs(low), abs(high))
        cmap = LinearSegmentedColormap.from_list("div_br", DIVERGING)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        note = "signed"
    else:
        # All one sign and clustered: this is magnitude, so a single-hue ramp
        # over the observed range shows structure that a zero-anchored
        # diverging scale would compress into one flat colour.
        cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
        norm = Normalize(vmin=low, vmax=high)
        note = f"range {low:.2f}–{high:.2f}"
    cmap = cmap.copy()
    cmap.set_bad(SURFACE)

    # aspect='auto' so the matrix fills its slot; a square heatmap would
    # leave most of this column empty.
    image = ax.imshow(shown, cmap=cmap, norm=norm, aspect="auto")
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    bar.set_label(f"cosine, centred ({note})", color=INK_2, fontsize=8)
    bar.ax.tick_params(colors=INK_2, labelsize=7)
    bar.outline.set_visible(False)

    labels = [_short(type_names[int(t)], 16)
              for t in batch.y[seed_rows].argmax(1).cpu().numpy()]
    if len(labels) <= 24:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        # Past ~24 seeds per-cell labels overlap into a smear; the matrix still
        # reads as structure, and the label distribution panel carries the types.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{len(labels)} seeds (labels omitted)",
                      color=INK_2, fontsize=8)
    ax.set_title("3 · image-embedding cosine between seeds", fontsize=10, loc="left")
    _style(ax)


def _panel_neighbourhood(axes, batch, type_names) -> None:
    """Dataset prior for each seed's type against what this batch contains."""
    y = batch.y.cpu().numpy()
    seed_rows = batch.seed_index.cpu().numpy()
    src = batch.edge_index[0].cpu().numpy()
    dst = batch.edge_index[1].cpu().numpy()
    prior = batch.seed_composition.cpu().numpy()
    seed_types = y[seed_rows].argmax(1)

    # Observed: for each seed, the normalised type histogram of its own inbound
    # neighbours, then averaged across the seeds of each type.
    observed = np.zeros_like(prior)
    for k, row in enumerate(seed_rows):
        incoming = src[dst == row]
        if len(incoming):
            observed[k] = y[incoming].mean(axis=0)

    shown = 0
    for cell_type in sorted(set(seed_types.tolist()),
                            key=lambda t: -(seed_types == t).sum()):
        if shown >= len(axes):
            break
        ax = axes[shown]
        members = seed_types == cell_type
        prior_row = prior[members].mean(axis=0)
        observed_row = observed[members].mean(axis=0)

        keep = np.argsort(-(prior_row + observed_row))[:6]
        positions = np.arange(len(keep))
        width = 0.38
        ax.bar(positions - width / 2 - 0.01, prior_row[keep], width,
               color=SERIES_1, label="dataset prior", zorder=3)
        ax.bar(positions + width / 2 + 0.01, observed_row[keep], width,
               color=SERIES_2, label="this batch", zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([_short(type_names[t], 14) for t in keep],
                           rotation=40, ha="right", fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_title(f"{_short(type_names[cell_type], 26)}  "
                     f"(n={int(members.sum())})", fontsize=8, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        if shown == 0:
            ax.legend(frameon=False, fontsize=7, labelcolor=INK_2)
            ax.set_ylabel("fraction of neighbours", color=INK_2, fontsize=7)
        _style(ax)
        shown += 1

    for ax in axes[shown:]:
        ax.set_axis_off()


def _type_legend(fig, palette, names, ncol=6):
    """Shared legend so identity is never colour-in-a-vacuum."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="", markersize=8,
                      markerfacecolor=palette[n], markeredgecolor=INK_2,
                      markeredgewidth=0.5, label=_short(n, 26))
               for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=8, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.01))


def plot_batch(batch, dataset, out_path: str | Path = "figures/batch.png",
               dpi: int = 220, overlay: bool = True, max_tiles: int = 16,
               index: int | None = None) -> Path:
    """One combined figure: neighbourhood tiles, tissue map, and the summaries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if index is not None:
        out_path = Path(out_path)
        out_path = out_path.parent / f"batch_{index}.png"

    palette = type_palette(dataset)
    names, _ = _node_types(batch, dataset)
    present = sorted(set(names), key=lambda n: -names.count(n))
    type_names = list(dataset.type_names)

    n_tiles = min(batch.n_seeds, max_tiles)
    cols = int(np.ceil(np.sqrt(n_tiles)))
    rows = int(np.ceil(n_tiles / cols))

    fig = plt.figure(figsize=(30, 19), dpi=dpi, facecolor=SURFACE)
    outer = fig.add_gridspec(2, 1, height_ratios=[2.05, 1], hspace=0.24)

    # Top: the per-seed graph tiles beside the tissue map.
    top = outer[0].subgridspec(1, 2, width_ratios=[1.05, 1], wspace=0.10)
    tiles = top[0, 0].subgridspec(rows, cols, hspace=0.42, wspace=0.16)
    radius = _panel_graph_grid(
        [fig.add_subplot(tiles[i, j]) for i in range(rows) for j in range(cols)],
        batch, dataset, palette)
    _panel_tissue(fig.add_subplot(top[0, 1]), batch, dataset, fig, palette,
                  overlay=overlay, seeds_only=True)

    # Bottom: composition and embedding summaries.
    lower = outer[1].subgridspec(1, 5, wspace=0.40,
                                 width_ratios=[1.35, 1.1, 1, 1, 1])
    _panel_labels(fig.add_subplot(lower[0, 0]), batch, type_names)
    _panel_cosine(fig.add_subplot(lower[0, 1]), batch, dataset, fig, type_names)
    _panel_neighbourhood([fig.add_subplot(lower[0, j]) for j in (2, 3, 4)],
                         batch, type_names)

    truncated = ("" if batch.n_seeds <= max_tiles
                 else f"  ·  tiles show {max_tiles} of {batch.n_seeds} seeds")
    fig.suptitle(
        f"batch: {batch.n_seeds} seeds → {batch.n_nodes} nodes, "
        f"{batch.edge_index.shape[1]} edges  ·  colour = {dataset.label_key}  ·  "
        f"tiles ±{radius} µm, edge width = apposed wall{truncated}",
        fontsize=14, color=INK, x=0.01, ha="left")
    _type_legend(fig, palette, present, ncol=7)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path
