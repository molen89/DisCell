#!/usr/bin/env python3
"""Compare neighbour-graph definitions on one sample, side by side.

Builds three graphs from a single load so they are directly comparable:

``contact_exact``
    polygons touching at zero distance
``contact_tol``
    polygons within ``--tolerance-um`` of each other
``voronoi``
    exact planar partition of the centroids, clipped to ``--clip-radius-um``

Writes one multi-panel figure over a shared region plus a statistics table, and
optionally a full-resolution figure per variant.

Xenium boundaries are segmented independently and are dilated
nuclei that collide, so exact contact already yields mean degree ~2.6. Xenium
boundaries stop just short of each other -- a quarter of neighbouring pairs sit
within 0.12 um, under the 0.2125 um pixel -- so exact contact collapses to ~0.14
and a tolerance is required.

Usage::

    python -m discell.preprocess.plotting.compare_graphs --sample <dir> --out-dir figures/
    python -m discell.preprocess.plotting.compare_graphs --sample <dir> --max-cells 20000 --overlay
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from discell.preprocess.geometry import (
    DEFAULT_CLIP_RADIUS_UM,
    DEFAULT_MAX_EDGE_UM,
    DEFAULT_WALL_TOLERANCE_UM,
    _store_graph,
    add_apposed_wall,
    build_contact_graph,
    build_voronoi_graph,
    graph_edge_frame,
)
from discell.tiff import find_tissue_image
from discell.tiff import read_image_window
from discell.preprocess.plotting.cell_graph import plot_cell_graph


log = logging.getLogger("discell.preprocess.plotting.compare_graphs")

DEFAULT_TOLERANCE_UM = 1.0


def build_variants(
    adata,
    tolerance_um: float = DEFAULT_TOLERANCE_UM,
    clip_radius_um: float = DEFAULT_CLIP_RADIUS_UM,
    max_edge_um: float = DEFAULT_MAX_EDGE_UM,
    wall_tolerance_um: float | None = DEFAULT_WALL_TOLERANCE_UM,
) -> list[tuple[str, str]]:
    """Build and store the three variants. Returns ``[(prefix, title), ...]``.

    Each gets ``apposed_wall_um`` measured on the real polygons, so a panel can
    be drawn or filtered by how much membrane two cells actually share rather
    than by the tessellation wall, which is a property of the partition.
    """
    polys = adata.uns["polygons"]
    cents = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    mpp = float(adata.uns["microns_per_pixel"])

    variants: list[tuple[str, str]] = []

    log.info("Building contact graph at exact contact")
    started = time.time()
    graph = build_contact_graph(polys, cents, mpp, 0.0, max_edge_um)
    _store_graph(adata, graph, "contact_exact")
    variants.append(("contact_exact", "contact (exact, 0 µm)"))
    log.info("  %.1fs", time.time() - started)

    log.info("Building contact graph at %.2f um tolerance", tolerance_um)
    started = time.time()
    graph = build_contact_graph(polys, cents, mpp, tolerance_um, max_edge_um)
    _store_graph(adata, graph, "contact_tol")
    variants.append(("contact_tol", f"contact (tolerance {tolerance_um:g} µm)"))
    log.info("  %.1fs", time.time() - started)

    log.info("Building Voronoi graph, clip %.0f um", clip_radius_um)
    started = time.time()
    graph, _ = build_voronoi_graph(polys, cents, mpp, clip_radius_um, max_edge_um)
    _store_graph(adata, graph, "voronoi_cmp")
    variants.append(("voronoi_cmp", f"voronoi (clip {clip_radius_um:g} µm)"))
    log.info("  %.1fs", time.time() - started)

    if wall_tolerance_um is not None:
        for prefix, _ in variants:
            add_apposed_wall(adata, prefix, wall_tolerance_um, polys)

    return variants


def statistics(adata, variants: Sequence[tuple[str, str]]):
    """Per-variant summary, as a DataFrame."""
    import pandas as pd

    rows = []
    for prefix, title in variants:
        info = adata.uns[f"{prefix}_graph"]
        edges = graph_edge_frame(adata, prefix)
        degrees = np.asarray(adata.obsp[f"{prefix}_connectivities"].sum(axis=1)).ravel()
        rows.append({
            "variant": title,
            "edges": int(info["n_edges"]),
            "mean_degree": round(float(info["mean_degree"]), 2),
            "median_degree": int(np.median(degrees)),
            "isolated_%": round(100 * float(info["isolated_fraction"]), 1),
            "shared_wall_um": round(float(edges["shared_wall_um"].median()), 2) if len(edges) else 0.0,
            "centroid_um": round(float(edges["centroid_dist_um"].median()), 2) if len(edges) else 0.0,
            "touching_%": round(100 * float((edges["wall_dist_um"] == 0).mean()), 1) if len(edges) else 0.0,
        })
    return pd.DataFrame(rows)


def densest_region(adata, span_um: float) -> tuple[float, float, float, float]:
    """A window of *span_um* centred on the densest patch of tissue."""
    cents = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    mpp = float(adata.uns["microns_per_pixel"])
    half = (span_um / mpp) / 2
    counts, xedges, yedges = np.histogram2d(cents[:, 0], cents[:, 1], bins=40)
    bx, by = np.unravel_index(np.argmax(counts), counts.shape)
    cx = 0.5 * (xedges[bx] + xedges[bx + 1])
    cy = 0.5 * (yedges[by] + yedges[by + 1])
    return (cx - half, cx + half, cy - half, cy + half)


def render_comparison(
    adata,
    sample_dir: Path,
    out_dir: Path,
    variants: Sequence[tuple[str, str]],
    region: tuple[float, float, float, float],
    label_key: str = "cluster",
    overlay: bool = False,
    color_edges: bool = True,
    edge_metric: str = "apposed_wall_um",
    min_apposed_um: float | None = None,
    figsize_in: float = 16.0,
    dpi: int = 150,
) -> Path:
    """One row of panels, one per variant, over a shared region."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1 = region
    aspect = (y1 - y0) / max(x1 - x0, 1e-9)
    mpp = float(adata.uns["microns_per_pixel"])
    sample_id = adata.uns.get("sample", {}).get("sample_id", sample_dir.name)

    image = None
    if overlay:
        image_path = find_tissue_image(sample_dir)
        if image_path is not None:
            log.info("Reading background image %s", image_path.name)
            image, _ = read_image_window(
                image_path, int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)), max_px=4000
            )

    n = len(variants)
    fig, axes = plt.subplots(
        1, n, figsize=(figsize_in * n, figsize_in * aspect + 1.5), dpi=dpi
    )
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, (prefix, title) in zip(axes, variants):
        if image is not None:
            ax.imshow(image, extent=(x0, x1, y1, y0), zorder=1)

        # A metric that is zero throughout would draw every edge at minimum
        # width and the bottom of the colormap. That happens with
        # shared_wall_um on a contact graph -- on Xenium touching pairs meet at
        # a point, not along a wall. Falling back to centroid distance keeps the
        # panel legible but INVERTS the reading: thicker then means further
        # apart, so the panel says so in its title.
        key = f"{prefix}_{edge_metric}"
        stored = adata.obsp[key] if key in adata.obsp else None
        usable = stored is not None and stored.nnz and float(np.median(stored.data)) > 0
        metric = edge_metric if usable else "centroid_dist_um"
        note = "" if usable else (
            f"  ({edge_metric.replace('_um', '')} is 0 here — sized by CENTROID "
            f"DISTANCE, so thicker = further apart)")

        drawn = plot_cell_graph(
            adata, ax, graph=prefix, label_key=label_key, region=region,
            polygon_alpha=0.30 if overlay else 0.65,
            edge_color="#00e5ff" if overlay else "#111111",
            edge_alpha=0.9, node_size=10.0, max_linewidth=3.5,
            color_edges_by_metric=color_edges, edge_metric=metric,
            min_apposed_um=min_apposed_um,
        )
        info = adata.uns[f"{prefix}_graph"]
        ax.set_title(
            f"{title}\n{drawn['edges']:,} edges here · mean degree "
            f"{info['mean_degree']:.2f} · isolated {100 * info['isolated_fraction']:.1f}%"
            f"{note}",
            fontsize=15, pad=12,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    bar_px = 100.0 / mpp
    bx, by = x0 + 0.05 * (x1 - x0), y1 - 0.06 * (y1 - y0)
    axes[0].plot([bx, bx + bar_px], [by, by], color="black" if not overlay else "white",
                 linewidth=4, zorder=6)
    axes[0].text(bx + bar_px / 2, by - 0.015 * (y1 - y0), "100 µm", ha="center",
                 va="bottom", fontsize=12, zorder=6,
                 color="black" if not overlay else "white")

    fig.suptitle(
        f"{sample_id} — neighbour graph definitions — {adata.n_obs:,} cells, "
        f"field of view {(x1 - x0) * mpp:,.0f} µm · node colour = {label_key} · "
        f"edge width/colour per panel (see subtitles)",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    suffix = "overlay" if overlay else "plain"
    path = out_dir / f"graph_comparison_{suffix}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Wrote %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path
