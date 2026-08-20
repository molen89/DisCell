#!/usr/bin/env python3
"""Render cell polygons and their neighbour graph over the tissue image.

Draws three layers, any of which can be turned off:

* **polygons** -- each cell filled by its cluster colour
* **edges** -- neighbour graph, line width proportional to shared wall length
* **nodes** -- cell centroids, coloured by cluster

Two variants are produced by default: one over the H&E/tissue image, one on a
plain background. Both are written at high resolution so they can be panned and
zoomed in an image viewer rather than re-rendered per region.

The tissue image is read through ``tifffile``, which handles the tiled,
strip-based and JPEG-compressed BigTIFFs in this cohort without decoding the
whole slide when only a window is needed.

Usage::

    python -m discell.preprocess.plotting.cell_graph --sample <dir> --out-dir figures/
    python -m discell.preprocess.plotting.cell_graph --sample <dir> --region 8000 9000 3000 4000
"""

from __future__ import annotations

import logging

import numpy as np

from discell.data.loader import DEFAULT_GRAPH


log = logging.getLogger("discell.preprocess.plotting.cell_graph")

DEFAULT_DPI = 200
DEFAULT_FIGSIZE_IN = 40.0  # 40in x 200dpi = 8000 px on the long edge

#: Shared-wall length (um) mapped to the thickest edge line.
WALL_REF_UM = 20.0

#: Value of each edge metric (um) that maps to the thickest line. Metrics not
#: listed fall back to the 95th percentile of whatever is being drawn.
METRIC_REF_UM = {
    "shared_wall_um": WALL_REF_UM,
    "apposed_wall_um": WALL_REF_UM,
    "centroid_dist_um": 30.0,
    "wall_dist_um": 10.0,
}

EDGE_METRICS = ("shared_wall_um", "apposed_wall_um", "centroid_dist_um", "wall_dist_um")



def cluster_colors(adata, label_key: str = "cluster") -> tuple[dict, np.ndarray]:
    """Map each cluster to a colour, preferring the palette 10x shipped."""
    import matplotlib.pyplot as plt
    import pandas as pd

    # Not every cell is necessarily assigned a cluster -- Xenium's clusters.csv
    # is short of cells.parquet by a few hundred rows -- and under pandas 3.0 a
    # categorical .astype(str) leaves those as float NaN rather than "nan".
    series = pd.Series(adata.obs[label_key]).astype(object)
    labels = series.where(series.notna(), "unassigned").astype(str).to_numpy()
    categories = sorted(
        set(labels), key=lambda s: (len(s), s)  # Cluster-2 before Cluster-10
    )

    # Prefer a palette shipped with the labels: the graphclust export
    # carries 'cluster_color', Xenium's supplemental cell_groups.csv carries
    # 'cell_group_color'.
    palette: dict[str, str] = {}
    for colour_key in (f"{label_key}_color", "cluster_color"):
        if colour_key not in adata.obs:
            continue
        supplied = pd.Series(adata.obs[colour_key]).astype(object)
        supplied = supplied.where(supplied.notna(), "").astype(str).to_numpy()
        for label, colour in zip(labels, supplied):
            if colour and colour not in ("nan", ""):
                palette.setdefault(label, colour)
        break

    missing = [c for c in categories if c not in palette]
    if missing:
        # tab20 wraps past 20 categories; use a continuous map when there are more.
        cmap = plt.get_cmap("tab20" if len(missing) <= 20 else "turbo")
        for k, category in enumerate(missing):
            palette[category] = (
                cmap(k % 20) if len(missing) <= 20 else cmap(k / max(len(missing) - 1, 1))
            )

    return palette, labels


def plot_cell_graph(
    adata,
    ax,
    graph: str = DEFAULT_GRAPH,
    label_key: str = "cluster",
    polygon_alpha: float = 0.55,
    edge_color: str = "#111111",
    edge_alpha: float = 0.55,
    node_size: float = 6.0,
    max_linewidth: float = 3.0,
    region: tuple[float, float, float, float] | None = None,
    edge_metric: str = "shared_wall_um",
    color_edges_by_metric: bool = False,
    edge_cmap: str = "viridis",
    max_gap_um: float | None = None,
    min_wall_um: float | None = None,
    max_centroid_um: float | None = None,
    min_apposed_um: float | None = None,
) -> dict:
    """Draw polygons, neighbour edges and nodes onto *ax* in pixel coordinates.

    Edge line width is proportional to *edge_metric*, and with
    *color_edges_by_metric* the same value also drives colour, which is the
    readable way to show how much two cells share rather than merely that they
    are adjacent.

    The filters select which edges survive, all in microns:

    ``max_gap_um``
        keep only edges whose *original* polygons are within this distance.
        ``0`` keeps only cells that physically touch -- 46.6% of Voronoi edges
        on the mouse brain.
    ``min_wall_um``
        drop edges sharing less than this much boundary.
    ``max_centroid_um``
        drop edges longer than this between centroids.

    Returns a dict describing what was drawn, including the LineCollection under
    ``"edge_collection"`` so a colourbar can be attached.
    """
    from matplotlib.collections import LineCollection, PolyCollection

    polys = adata.uns["polygons"]
    cents = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    palette, labels = cluster_colors(adata, label_key)

    if region is not None:
        x0, x1, y0, y1 = region
        keep = (
            (cents[:, 0] >= x0) & (cents[:, 0] <= x1)
            & (cents[:, 1] >= y0) & (cents[:, 1] <= y1)
        )
    else:
        keep = np.ones(len(cents), dtype=bool)
    keep_idx = np.flatnonzero(keep)
    remap = -np.ones(len(cents), dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))
    log.info("  drawing %d of %d cells", len(keep_idx), len(cents))

    drawn = {"cells": len(keep_idx), "edges": 0}

    verts, facecolors = [], []
    for k in keep_idx:
        poly = polys[k]
        if poly.is_empty:
            continue
        verts.append(np.asarray(poly.exterior.coords))
        facecolors.append(palette.get(labels[k], "#999999"))
    ax.add_collection(
        PolyCollection(
            verts,
            facecolors=facecolors,
            edgecolors="#333333",
            linewidths=0.25,
            alpha=polygon_alpha,
            zorder=2,
        )
    )

    from discell.preprocess.geometry import graph_edge_frame

    if edge_metric not in EDGE_METRICS:
        raise ValueError(f"edge_metric must be one of {EDGE_METRICS}")

    edges = graph_edge_frame(adata, graph)
    n_before = len(edges)
    if max_gap_um is not None:
        edges = edges[edges["wall_dist_um"] <= max_gap_um]
    if min_wall_um is not None:
        edges = edges[edges["shared_wall_um"] >= min_wall_um]
    if max_centroid_um is not None:
        edges = edges[edges["centroid_dist_um"] <= max_centroid_um]
    if min_apposed_um is not None:
        if "apposed_wall_um" not in edges:
            log.warning("min_apposed_um ignored: no apposed wall stored for %r", graph)
        else:
            # Trims edges whose cells are adjacent but share no real membrane
            # -- the case that arises whenever the graph was built with a
            # looser tolerance than the wall was measured with.
            edges = edges[edges["apposed_wall_um"] >= min_apposed_um]
    drawn["edges_before_filter"] = n_before
    drawn["edges_after_filter"] = len(edges)
    if n_before and len(edges) < n_before:
        log.info(
            "  edge filter kept %d of %d edges (%.1f%%)",
            len(edges), n_before, 100 * len(edges) / n_before,
        )

    i = edges["i"].to_numpy()
    j = edges["j"].to_numpy()
    values = edges[edge_metric].to_numpy()
    both = keep[i] & keep[j]
    i, j, values = i[both], j[both], values[both]

    if len(i):
        reference = METRIC_REF_UM.get(edge_metric) or float(np.percentile(values, 95))
        widths = np.clip(values / max(reference, 1e-9), 0.15, 1.0) * max_linewidth
        segments = np.stack([cents[i], cents[j]], axis=1)

        if color_edges_by_metric:
            collection = LineCollection(
                segments, linewidths=widths, array=values, cmap=edge_cmap,
                alpha=edge_alpha, zorder=3,
            )
            collection.set_clim(float(values.min()), float(np.percentile(values, 99)))
        else:
            collection = LineCollection(
                segments, linewidths=widths, colors=edge_color,
                alpha=edge_alpha, zorder=3,
            )
        ax.add_collection(collection)
        drawn["edges"] = len(i)
        drawn["edge_collection"] = collection
        drawn["metric"] = edge_metric
        drawn["metric_median"] = float(np.median(values))
        drawn["wall_um_median"] = float(np.median(edges["shared_wall_um"].to_numpy()[both]))

    ax.scatter(
        cents[keep_idx, 0],
        cents[keep_idx, 1],
        s=node_size,
        c=[palette.get(labels[k], "#999999") for k in keep_idx],
        edgecolors="#000000",
        linewidths=0.2,
        zorder=4,
    )

    if region is not None:
        ax.set_xlim(region[0], region[1])
        ax.set_ylim(region[3], region[2])  # image convention: y grows downward
    else:
        pad = 0.01 * max(np.ptp(cents[:, 0]), np.ptp(cents[:, 1]))
        ax.set_xlim(cents[:, 0].min() - pad, cents[:, 0].max() + pad)
        ax.set_ylim(cents[:, 1].max() + pad, cents[:, 1].min() - pad)
    ax.set_aspect("equal")
    return drawn


def _legend_handles(
    adata,
    label_key: str,
    edge_metric: str = "shared_wall_um",
    show_width_key: bool = True,
):
    from matplotlib.lines import Line2D

    palette, labels = cluster_colors(adata, label_key)
    counts = {c: int((labels == c).sum()) for c in palette}
    order = sorted(palette, key=lambda c: -counts.get(c, 0))
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=12,
               markerfacecolor=palette[c], markeredgecolor="#000000",
               label=f"{c}  (n={counts.get(c, 0)})")
        for c in order
    ]
    if show_width_key:
        handles.append(Line2D([], [], linestyle="none", label=""))
        reference = METRIC_REF_UM.get(edge_metric, WALL_REF_UM)
        short = edge_metric.replace("_um", "").replace("_", " ")
        for fraction in (0.25, 0.5, 1.0):
            handles.append(
                Line2D([], [], color="#111111",
                       linewidth=np.clip(fraction, 0.15, 1.0) * 3.0,
                       label=f"{short} {reference * fraction:g} µm")
            )
    return handles
