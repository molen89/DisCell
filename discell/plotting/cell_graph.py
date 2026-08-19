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

    python -m discell.plotting.cell_graph --sample <dir> --out-dir figures/
    python -m discell.plotting.cell_graph --sample <dir> --region 8000 9000 3000 4000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("discell.plotting.cell_graph")

DEFAULT_DPI = 200
DEFAULT_FIGSIZE_IN = 40.0  # 40in x 200dpi = 8000 px on the long edge
DEFAULT_GRAPH = "voronoi"

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


def find_tissue_image(sample_dir: Path) -> Path | None:
    """Locate a background image: Visium HD tissue image or Xenium morphology."""
    patterns = (
        "*_tissue_image.btf", "*_tissue_image.tif", "*_tissue_image.tiff",
        # Xenium: channel 0000 is DAPI, which reads best under the polygons.
        "morphology_focus/morphology_focus_0000.ome.tif",
        "morphology_focus/morphology_focus_*.ome.tif",
        "morphology.ome.tif",
    )
    for pattern in patterns:
        hits = sorted(sample_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


#: Full decoded pages for strip-based/compressed images, which have no random
#: access. Keyed by path; holds one slide (~1.4 GB for the mouse brain) so the
#: several figures of one run decode it once.
_PAGE_CACHE: dict[str, np.ndarray] = {}


def clear_image_cache() -> None:
    _PAGE_CACHE.clear()


def _to_uint8_rgb(window: np.ndarray) -> np.ndarray:
    """Normalise an arbitrary image window to 8-bit RGB for display.

    Xenium morphology images are uint16 with most of the range unused, so a
    percentile stretch is needed for anything to be visible.
    """
    if window.ndim == 3 and window.shape[-1] in (3, 4):
        rgb = window[..., :3]
    elif window.ndim == 2:
        rgb = window
    else:  # unexpected layout: collapse to the first plane
        rgb = np.asarray(window).reshape(-1, *window.shape[-2:])[0]

    if rgb.dtype != np.uint8:
        finite = rgb[np.isfinite(rgb)] if np.issubdtype(rgb.dtype, np.floating) else rgb
        if finite.size:
            low, high = np.percentile(finite, (1.0, 99.5))
        else:
            low, high = 0.0, 1.0
        if high <= low:
            high = low + 1.0
        rgb = np.clip((rgb.astype(np.float32) - low) / (high - low), 0, 1)
        rgb = (rgb * 255).astype(np.uint8)

    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    return rgb


#: Xenium morphology_focus channel names, in file order.
XENIUM_CHANNELS = ("DAPI", "ATP1A1/CD45/E-Cadherin", "18S", "alphaSMA/Vimentin")

#: Default composite for Xenium morphology: boundary in red, nuclei in green,
#: interior RNA in blue. DAPI alone is the sparsest of the four and a poor
#: backdrop; the boundary channel carries the tissue architecture.
XENIUM_DEFAULT_CHANNELS = (1, 0, 2)


def is_xenium_morphology(path: Path) -> bool:
    return "morphology" in Path(path).name


def _compose_rgb(planes: Sequence[np.ndarray]) -> np.ndarray:
    """Map 1-4 stretched planes onto RGB.

    One plane is greyscale; two or three fill R, G, B in order; a fourth is
    blended into all three as luminance so it lightens rather than tints.
    """
    planes = [p.astype(np.float32) for p in planes]
    stretched = []
    for plane in planes:
        low, high = np.percentile(plane, (1.0, 99.5))
        if high <= low:
            high = low + 1.0
        stretched.append(np.clip((plane - low) / (high - low), 0, 1))

    if len(stretched) == 1:
        rgb = np.stack([stretched[0]] * 3, axis=-1)
    else:
        rgb = np.zeros((*stretched[0].shape, 3), dtype=np.float32)
        for k, plane in enumerate(stretched[:3]):
            rgb[..., k] = plane
        if len(stretched) >= 4:
            rgb = np.clip(rgb + 0.4 * stretched[3][..., None], 0, 1)
    return (rgb * 255).astype(np.uint8)


def read_image_window(
    path: Path,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    max_px: int = 12000,
    channel: int = 0,
    channels: Sequence[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Read the window ``[y0:y1, x0:x1]`` of a large TIFF, as 8-bit RGB.

    Handles the three layouts in this cohort: Visium HD's single-level tiled or
    strip-based RGB, and Xenium's pyramidal multi-channel OME-TIFF, whose series
    is ``(C, Y, X)`` -- slicing that as if it were 2-D silently returns an empty
    array. When the file carries a pyramid, the coarsest level that still meets
    *max_px* is used rather than decimating full-resolution pixels.

    Returns ``(rgb, scale)`` with *scale* the factor applied to the requested
    window.
    """
    import tifffile

    started = time.time()
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        axes = series.axes
        levels = list(series.levels)

        def yx(shape: tuple) -> tuple[int, int]:
            if "Y" in axes and "X" in axes:
                return shape[axes.index("Y")], shape[axes.index("X")]
            return shape[0], shape[1]

        height, width = yx(levels[0].shape)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"empty image window x[{x0}:{x1}] y[{y0}:{y1}] against {width}x{height}"
            )

        # Pick the finest pyramid level whose view of this window fits max_px.
        want = max(x1 - x0, y1 - y0)
        chosen, factor = 0, 1.0
        for index, level in enumerate(levels):
            level_h, level_w = yx(level.shape)
            level_factor = width / max(level_w, 1)
            if want / level_factor <= max_px:
                chosen, factor = index, level_factor
                break
            chosen, factor = index, level_factor
        if len(levels) > 1:
            log.info("  using pyramid level %d of %d (1/%.0f scale)",
                     chosen, len(levels), factor)

        page = handle.pages[0]
        tiled = getattr(page, "is_tiled", False)
        sx0, sx1 = int(x0 / factor), int(np.ceil(x1 / factor))
        sy0, sy1 = int(y0 / factor), int(np.ceil(y1 / factor))

        if tiled or len(levels) > 1:
            import zarr

            store = handle.aszarr(series=0, level=chosen)
            arr = zarr.open(store, mode="r")
            if arr.ndim == 3 and axes.startswith("C"):
                # Multi-channel OME (Xenium): the channels live in sibling files
                # but tifffile resolves them through this one handle.
                wanted = list(channels) if channels else [channel]
                wanted = [c for c in wanted if 0 <= c < arr.shape[0]]
                if not wanted:
                    raise ValueError(f"no valid channel in {channels or channel} "
                                     f"for an image with {arr.shape[0]} channels")
                planes = [np.asarray(arr[c, sy0:sy1, sx0:sx1]) for c in wanted]
                window = _compose_rgb(planes) if len(planes) > 1 else planes[0]
                if len(planes) > 1:
                    log.info("  composite channels %s -> RGB", wanted)
            elif arr.ndim == 3:
                window = np.asarray(arr[sy0:sy1, sx0:sx1, :])
            else:
                window = np.asarray(arr[sy0:sy1, sx0:sx1])
            store.close()
        else:
            # Strip-based pages have no random access, so the whole page must be
            # decoded; cache it for the other figures in this run.
            key = str(path)
            if key not in _PAGE_CACHE:
                log.info("  page is not tiled -- decoding full page (%dx%d)", width, height)
                _PAGE_CACHE[key] = page.asarray()
            else:
                log.info("  reusing cached decode of %s", path.name)
            window = _PAGE_CACHE[key][sy0:sy1, sx0:sx1]

    window = _to_uint8_rgb(window)

    scale = 1.0 / factor
    longest = max(window.shape[0], window.shape[1])
    if longest > max_px:
        step = int(np.ceil(longest / max_px))
        window = window[::step, ::step]
        scale /= step
    log.info("  read image window in %.1fs, shape=%s", time.time() - started, window.shape)
    return window, scale


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

    # Prefer a palette shipped with the labels: Visium HD's graphclust geojson
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
    draw_polygons: bool = True,
    draw_edges: bool = True,
    draw_nodes: bool = True,
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
    import scipy.sparse as sp
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

    if draw_polygons:
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

    if draw_edges:
        from discell.data.segmented import graph_edge_frame

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

    if draw_nodes:
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


def render(
    adata,
    sample_dir: Path,
    out_dir: Path,
    graph: str = DEFAULT_GRAPH,
    label_key: str = "cluster",
    region: tuple[float, float, float, float] | None = None,
    figsize_in: float = DEFAULT_FIGSIZE_IN,
    dpi: int = DEFAULT_DPI,
    tag: str = "full",
    image_max_px: int = 12000,
    interpolation: str | None = None,
    channels: Sequence[int] | None = None,
    draw_polygons: bool = True,
    draw_edges: bool = True,
    draw_nodes: bool = True,
    edge_metric: str = "shared_wall_um",
    color_edges_by_metric: bool = False,
    edge_cmap: str = "viridis",
    max_gap_um: float | None = None,
    min_wall_um: float | None = None,
    max_centroid_um: float | None = None,
    min_apposed_um: float | None = None,
) -> list[Path]:
    """Write the overlay and no-overlay figures. Returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cents = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if region is None:
        region = (
            float(cents[:, 0].min()), float(cents[:, 0].max()),
            float(cents[:, 1].min()), float(cents[:, 1].max()),
        )
    x0, x1, y0, y1 = region
    aspect = (y1 - y0) / max(x1 - x0, 1e-9)

    mpp = adata.uns.get("microns_per_pixel", 1.0)
    info = adata.uns.get(f"{graph}_graph", {})
    sample_id = adata.uns.get("sample", {}).get("sample_id", sample_dir.name)
    written: list[Path] = []

    for overlay in (False, True):
        image = None
        if overlay:
            image_path = find_tissue_image(sample_dir)
            if image_path is None:
                log.warning("No tissue image in %s -- skipping overlay figure", sample_dir)
                continue
            log.info("Reading tissue image %s", image_path.name)
            picked = channels
            if picked is None and is_xenium_morphology(image_path):
                picked = XENIUM_DEFAULT_CHANNELS
            image, _ = read_image_window(
                image_path, int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)),
                max_px=image_max_px, channels=picked,
            )

        width_in = figsize_in
        height_in = max(4.0, figsize_in * aspect)
        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
        fig.patch.set_facecolor("white")
        _resolution_note = None

        if overlay:
            source_px = max(int(x1 - x0), int(y1 - y0))
            mode = interpolation or "nearest"
            ax.imshow(image, extent=(x0, x1, y1, y0), interpolation=mode, zorder=1)
            _resolution_note = (source_px, max(image.shape[0], image.shape[1]))
            polygon_alpha, edge_alpha, edge_color = 0.30, 0.75, "#00e5ff"
        else:
            ax.set_facecolor("white")
            polygon_alpha, edge_alpha, edge_color = 0.70, 0.55, "#111111"

        drawn = plot_cell_graph(
            adata, ax, graph=graph, label_key=label_key, region=region,
            polygon_alpha=polygon_alpha, edge_alpha=edge_alpha, edge_color=edge_color,
            node_size=6.0 if len(cents) > 5000 else 40.0,
            max_linewidth=3.0,
            edge_metric=edge_metric, color_edges_by_metric=color_edges_by_metric,
            edge_cmap=edge_cmap, max_gap_um=max_gap_um, min_wall_um=min_wall_um,
            max_centroid_um=max_centroid_um, min_apposed_um=min_apposed_um,
            draw_polygons=draw_polygons, draw_edges=draw_edges, draw_nodes=draw_nodes,
        )

        span_um = (x1 - x0) * mpp
        filters = []
        if max_gap_um is not None:
            filters.append("touching only" if max_gap_um == 0 else f"gap ≤ {max_gap_um:g} µm")
        if min_wall_um is not None:
            filters.append(f"wall ≥ {min_wall_um:g} µm")
        if max_centroid_um is not None:
            filters.append(f"centroid ≤ {max_centroid_um:g} µm")
        kept = ""
        if filters:
            before = drawn.get("edges_before_filter", 0)
            after = drawn.get("edges_after_filter", 0)
            kept = (f" · filter: {', '.join(filters)} "
                    f"({after:,}/{before:,} = {100 * after / max(before, 1):.0f}% kept)")

        short = edge_metric.replace("_um", "").replace("_", " ")
        ax.set_title(
            f"{sample_id} — {graph} graph — {drawn['cells']:,} cells, {drawn['edges']:,} edges\n"
            f"edge width{' and colour' if color_edges_by_metric else ''} ∝ {short} "
            f"(median {drawn.get('metric_median', float('nan')):.1f} µm) · "
            f"node colour = {label_key} · field of view {span_um:,.0f} µm · "
            f"mean degree {info.get('mean_degree', float('nan')):.2f}{kept}",
            fontsize=22, pad=24,
        )
        ax.set_xlabel("x (full-res pixels)", fontsize=16)
        ax.set_ylabel("y (full-res pixels)", fontsize=16)
        ax.legend(
            handles=_legend_handles(adata, label_key, edge_metric,
                                    show_width_key=not color_edges_by_metric),
            loc="upper right", fontsize=14, framealpha=0.9, ncol=1,
        )
        if color_edges_by_metric and drawn.get("edge_collection") is not None:
            bar = fig.colorbar(drawn["edge_collection"], ax=ax, fraction=0.025, pad=0.01)
            bar.set_label(f"{short} (µm)", fontsize=16)
            bar.ax.tick_params(labelsize=13)

        # Scale bar, 100 um.
        bar_px = 100.0 / mpp
        bx, by = x0 + 0.03 * (x1 - x0), y1 - 0.05 * (y1 - y0)
        ax.plot([bx, bx + bar_px], [by, by], color="black", linewidth=5, zorder=6)
        ax.text(bx + bar_px / 2, by - 0.012 * (y1 - y0), "100 µm",
                ha="center", va="bottom", fontsize=16, zorder=6)

        if _resolution_note is not None:
            # Measured from the laid-out axes, not the figure width: margins and
            # the legend can take half of it, and an optimistic estimate would
            # hide exactly the downsampling this is meant to surface.
            fig.canvas.draw()
            box = ax.get_window_extent()
            source_px, read_px = _resolution_note
            per_source = max(box.width, box.height) / max(source_px, 1)
            log.info(
                "  %.0f um field: %d source px -> %d read -> %.0f axes px "
                "(%.2f screen px per source px; source is %.3f um/px)",
                (x1 - x0) * mpp, source_px, read_px, max(box.width, box.height),
                per_source, mpp * max(source_px / max(read_px, 1), 1.0),
            )
            if per_source < 1.0:
                log.warning("  figure is coarser than the image (%.2fx): raise --dpi or "
                            "--figsize-in, or shrink --region", per_source)

        # Encode the metric and filter in the name so variants do not overwrite.
        parts = [sample_id, graph, tag]
        if edge_metric != "shared_wall_um":
            parts.append(edge_metric.replace("_um", ""))
        if color_edges_by_metric:
            parts.append("coloured")
        if max_gap_um is not None:
            parts.append("touching" if max_gap_um == 0 else f"gap{max_gap_um:g}")
        if min_wall_um is not None:
            parts.append(f"wall{min_wall_um:g}")
        if min_apposed_um is not None:
            parts.append(f"app{min_apposed_um:g}")
        parts.append("overlay" if overlay else "plain")
        path = out_dir / ("_".join(parts) + ".png")
        started = time.time()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("Wrote %s (%.1f MB) in %.1fs",
                 path.name, path.stat().st_size / 1e6, time.time() - started)
        written.append(path)

    return written


def build_parser() -> argparse.ArgumentParser:
    from discell.data.visium_hd import DEFAULT_DATA_ROOT

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", default=str(DEFAULT_DATA_ROOT / "Visium_HD_Mouse_Brain"))
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--graph", default=DEFAULT_GRAPH, choices=("voronoi", "contact"))
    parser.add_argument("--label-key", default="cluster")
    parser.add_argument("--region", nargs=4, type=float, default=None,
                        metavar=("X0", "X1", "Y0", "Y1"),
                        help="pixel window to render; default is the whole slide")
    parser.add_argument("--zoom-um", type=float, default=400.0,
                        help="also render a zoom panel this many um wide")
    parser.add_argument("--no-zoom", action="store_true")
    parser.add_argument("--figsize-in", type=float, default=DEFAULT_FIGSIZE_IN)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--clip-radius-um", type=float, default=30.0)
    parser.add_argument("--show", action="store_true",
                        help="open an interactive pan/zoom view instead of writing PNGs; "
                             "uses a window under ssh -X, else serves in a browser")
    parser.add_argument("--backend", default=None,
                        help="force a matplotlib backend (QtAgg/TkAgg/WebAgg); "
                             "default picks the first that loads")
    parser.add_argument("--port", type=int, default=8988, help="WebAgg port")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Xenium only: use the N cells nearest the tissue centre")

    edges = parser.add_argument_group("edge metric and filtering")
    edges.add_argument("--edge-metric", default="shared_wall_um", choices=EDGE_METRICS,
                       help="which per-edge quantity drives line width (and colour)")
    edges.add_argument("--color-edges", action="store_true",
                       help="colour edges by --edge-metric and add a colourbar, "
                            "so you see how much each pair shares")
    edges.add_argument("--edge-cmap", default="viridis")
    edges.add_argument("--max-gap-um", type=float, default=None,
                       help="keep only edges whose original polygons are within this "
                            "distance; 0 keeps only cells that physically touch")
    edges.add_argument("--touching", action="store_true",
                       help="shorthand for --max-gap-um 0")
    edges.add_argument("--min-wall-um", type=float, default=None,
                       help="drop edges sharing less boundary than this (shared_wall_um)")
    edges.add_argument("--min-apposed-um", type=float, default=None,
                       help="drop edges whose cells share less apposed membrane than this; "
                            "use >0 to trim adjacency that has no real contact")
    edges.add_argument("--max-centroid-um", type=float, default=None,
                       help="drop edges longer than this between centroids")
    edges.add_argument("--contact-tolerance-um", type=float, default=None,
                       help="graph adjacency: how close polygons must be to be neighbours "
                            "(default 0 on Visium HD, 1 on Xenium)")
    edges.add_argument("--wall-tolerance-um", type=float, default=None,
                       help="apposed_wall_um: how close two membranes must run to count as "
                            "shared wall (default 1). Independent of adjacency.")
    parser.add_argument("--overlay", action="store_true", help="--show over the tissue image")

    layers = parser.add_argument_group("image channels and layers")
    layers.add_argument("--channels", default=None,
                        help="image channels to composite, e.g. '1,0,2' -> R,G,B. "
                             "Xenium morphology defaults to 1,0,2 (boundary/DAPI/18S); "
                             "a 4th channel is blended in as luminance")
    layers.add_argument("--no-polygons", action="store_true", help="hide cell outlines")
    layers.add_argument("--no-edges", action="store_true", help="hide graph edges")
    layers.add_argument("--no-nodes", action="store_true", help="hide centroid markers")
    layers.add_argument("--image-only", action="store_true",
                        help="hide every overlay: inspect the image alone")
    layers.add_argument("--image-max-px", type=int, default=12000,
                        help="longest edge read from the image; caps which pyramid "
                             "level is used, so raise it to keep full resolution")
    layers.add_argument("--interpolation", default=None,
                        choices=("nearest", "antialiased", "bilinear", "bicubic"),
                        help="imshow interpolation; default nearest at >=1:1, else antialiased")

    parser.add_argument("--quiet", action="store_true")
    return parser


def display_reachable(timeout_s: float = 5.0) -> bool:
    """Whether ``$DISPLAY`` names an X server we can actually talk to.

    Worth probing rather than trusting: if ``$DISPLAY`` is set but the server is
    unreachable -- a stale ``ssh -X`` session, forwarding refused -- Qt does not
    raise, it prints "could not load the Qt platform plugin" and *aborts the
    process*. A try/except cannot recover from that, so the check has to happen
    before the backend is chosen.
    """
    import os
    import shutil
    import subprocess

    display = os.environ.get("DISPLAY")
    if not display:
        return False
    probe = shutil.which("xdpyinfo")
    if probe is None:
        return True  # can't tell; let the toolkit try
    try:
        done = subprocess.run(
            [probe, "-display", display], capture_output=True, timeout=timeout_s
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return done.returncode == 0


def select_backend(preferred: str | None = None, webagg_port: int = 8988) -> str:
    """Pick an interactive matplotlib backend that actually loads here.

    Ordering matters on this setup. ``import tkinter`` succeeds under
    uv-managed CPython, but ``_tkinter`` is statically linked and exposes no
    ``__file__``, so matplotlib's ``_tkagg`` extension cannot locate Tcl/Tk and
    TkAgg raises ImportError at load time. Qt is the dependable X11 choice;
    WebAgg needs no display at all and is used whenever no X server is reachable.
    """
    import os

    import matplotlib

    display = os.environ.get("DISPLAY") if display_reachable() else None
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates += ["QtAgg", "TkAgg", "WebAgg"]
    seen: set[str] = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]

    rejected: dict[str, str] = {}
    for backend in ordered:
        if backend != "WebAgg" and not display:
            rejected[backend] = (
                "no reachable X server "
                f"(DISPLAY={os.environ.get('DISPLAY') or 'unset'})"
            )
            continue
        try:
            if backend == "WebAgg":
                matplotlib.rcParams["webagg.port"] = webagg_port
                matplotlib.rcParams["webagg.open_in_browser"] = False
            matplotlib.use(backend, force=True)
        except Exception as exc:  # missing toolkit, broken extension, no Tcl/Tk
            rejected[backend] = f"{type(exc).__name__}: {str(exc).splitlines()[-1][:110]}"
            continue
        for name, why in rejected.items():
            log.info("  backend %s unavailable (%s)", name, why)
        log.info("Using interactive backend: %s", backend)
        return backend

    raise RuntimeError(
        "No interactive backend available:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in rejected.items())
        + "\nInstall the viz extra with: uv sync --extra viz"
    )


def show_interactive(
    adata,
    sample_dir: Path,
    graph: str = DEFAULT_GRAPH,
    label_key: str = "cluster",
    region: tuple[float, float, float, float] | None = None,
    overlay: bool = False,
    backend: str | None = None,
    webagg_port: int = 8988,
    edge_metric: str = "shared_wall_um",
    color_edges_by_metric: bool = False,
    edge_cmap: str = "viridis",
    max_gap_um: float | None = None,
    min_wall_um: float | None = None,
    max_centroid_um: float | None = None,
    min_apposed_um: float | None = None,
    channels: Sequence[int] | None = None,
    draw_polygons: bool = True,
    draw_edges: bool = True,
    draw_nodes: bool = True,
    image_max_px: int = 6000,
    interpolation: str | None = None,
) -> None:
    """Open a pan/zoomable view, over ``ssh -X`` or in a browser.

    Vector layers are re-rasterised by the toolkit on every zoom, so this stays
    responsive where panning a 8000x10000 PNG in an image viewer does not.
    """
    import os

    chosen = select_backend(backend, webagg_port)
    import matplotlib.pyplot as plt

    cents = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if region is None:
        region = (
            float(cents[:, 0].min()), float(cents[:, 0].max()),
            float(cents[:, 1].min()), float(cents[:, 1].max()),
        )
    fig, ax = plt.subplots(figsize=(16, 12))
    if overlay:
        image_path = find_tissue_image(sample_dir)
        if image_path is not None:
            x0, x1, y0, y1 = region
            picked = channels
            if picked is None and is_xenium_morphology(image_path):
                picked = XENIUM_DEFAULT_CHANNELS
            image, _ = read_image_window(
                image_path, int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)),
                max_px=image_max_px, channels=picked,
            )
            source_px = max(int(x1 - x0), int(y1 - y0))
            log.info("  %.0f um field: %d source px -> %d read (source is %.4f um/px)",
                     (x1 - x0) * float(adata.uns.get("microns_per_pixel", 1.0)),
                     source_px, max(image.shape[0], image.shape[1]),
                     float(adata.uns.get("microns_per_pixel", 1.0)))
            ax.imshow(image, extent=(x0, x1, y1, y0),
                      interpolation=interpolation or "nearest", zorder=1)

    drawn = plot_cell_graph(
        adata, ax, graph=graph, label_key=label_key, region=region,
        polygon_alpha=0.35 if overlay else 0.7,
        edge_color="#00e5ff" if overlay else "#111111",
        node_size=8.0,
        edge_metric=edge_metric, color_edges_by_metric=color_edges_by_metric,
        edge_cmap=edge_cmap, max_gap_um=max_gap_um, min_wall_um=min_wall_um,
        max_centroid_um=max_centroid_um, min_apposed_um=min_apposed_um,
        draw_polygons=draw_polygons, draw_edges=draw_edges, draw_nodes=draw_nodes,
    )
    short = edge_metric.replace("_um", "").replace("_", " ")
    kept = ""
    if drawn.get("edges_after_filter") != drawn.get("edges_before_filter"):
        kept = (f" — {drawn['edges_after_filter']:,}/{drawn['edges_before_filter']:,} "
                f"edges kept")
    ax.set_title(f"{adata.uns.get('sample', {}).get('sample_id', '')} — {graph} graph{kept}\n"
                 f"scroll/drag to zoom; edge width ∝ {short}")
    if color_edges_by_metric and drawn.get("edge_collection") is not None:
        bar = fig.colorbar(drawn["edge_collection"], ax=ax, fraction=0.03, pad=0.01)
        bar.set_label(f"{short} (µm)")
    plt.tight_layout()

    if chosen == "WebAgg":
        print(
            f"\nServing on http://localhost:{webagg_port}\n"
            f"  If this is a remote host, forward the port from your laptop:\n"
            f"    ssh -L {webagg_port}:localhost:{webagg_port} "
            f"{os.environ.get('USER', 'user')}@<this-host>\n"
            f"  then open http://localhost:{webagg_port}\n"
        )
    else:
        print(f"\nOpening {chosen} window on DISPLAY={os.environ.get('DISPLAY', '<unset>')}\n")
    plt.show()


def load_any_sample(
    sample_path: str,
    clip_radius_um: float,
    max_cells: int | None = None,
    contact_tolerance_um: float | None = None,
    wall_tolerance_um: float | None = None,
    build_graphs: bool = True,
):
    """Load a sample from either platform, detected from what is on disk.

    Returns ``(adata, sample_dir)``. Both loaders produce the same structure, so
    everything downstream is platform-agnostic. Passing ``None`` for either
    tolerance keeps that loader's own default, which differs by platform.
    """
    path = Path(sample_path).expanduser().resolve()
    is_xenium = (
        (path / "cell_boundaries.parquet").exists()
        or (path / "experiment.xenium").exists()
        or bool(list(path.glob("*/cell_boundaries.parquet")))
        or path.suffix == ".zip"
    )
    extra = {"build_graphs": build_graphs}
    if contact_tolerance_um is not None:
        extra["contact_tolerance_um"] = contact_tolerance_um
    if wall_tolerance_um is not None:
        extra["wall_tolerance_um"] = wall_tolerance_um

    if is_xenium:
        from discell.data.xenium import load_xenium_sample, locate_xenium_dir

        log.info("Detected Xenium output")
        sample_dir = locate_xenium_dir(path)
        adata = load_xenium_sample(
            sample_dir, clip_radius_um=clip_radius_um, max_cells=max_cells, **extra
        )
        return adata, sample_dir

    from discell.data.segmented import load_segmented_sample
    from discell.data.visium_hd import resolve_sample

    log.info("Detected Visium HD output")
    sample_dir, _, _ = resolve_sample(sample_path)
    adata = load_segmented_sample(sample_path, clip_radius_um=clip_radius_um, **extra)
    return adata, sample_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )

    adata, sample_dir = load_any_sample(
        args.sample, args.clip_radius_um, args.max_cells,
        args.contact_tolerance_um, args.wall_tolerance_um,
    )
    # Xenium has no shipped palette and names its labels 'graphclust'.
    label_key = args.label_key
    if label_key not in adata.obs:
        label_key = adata.uns.get("default_label", label_key)
        log.info("Using label key %r", label_key)
    args.label_key = label_key
    out_dir = Path(args.out_dir)

    region = tuple(args.region) if args.region else None
    max_gap_um = 0.0 if args.touching else args.max_gap_um
    channels = ([int(c) for c in args.channels.split(",") if c.strip() != ""]
                if args.channels else None)
    layer_kwargs = dict(
        channels=channels,
        image_max_px=args.image_max_px,
        interpolation=args.interpolation,
        draw_polygons=not (args.no_polygons or args.image_only),
        draw_edges=not (args.no_edges or args.image_only),
        draw_nodes=not (args.no_nodes or args.image_only),
    )
    edge_kwargs = dict(
        edge_metric=args.edge_metric,
        color_edges_by_metric=args.color_edges,
        edge_cmap=args.edge_cmap,
        max_gap_um=max_gap_um,
        min_wall_um=args.min_wall_um,
        max_centroid_um=args.max_centroid_um,
        min_apposed_um=args.min_apposed_um,
    )

    if args.show:
        show_interactive(adata, sample_dir, graph=args.graph, label_key=args.label_key,
                         region=region, overlay=args.overlay, backend=args.backend,
                         webagg_port=args.port, **edge_kwargs, **layer_kwargs)
        return 0

    written = render(adata, sample_dir, out_dir, graph=args.graph,
                     label_key=args.label_key, region=region,
                     figsize_in=args.figsize_in, dpi=args.dpi, tag="full",
                     **edge_kwargs, **layer_kwargs)

    if not args.no_zoom and region is None:
        # Centre the zoom on the densest patch, which is the most informative.
        cents = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        mpp = adata.uns["microns_per_pixel"]
        half = (args.zoom_um / mpp) / 2
        counts, xedges, yedges = np.histogram2d(cents[:, 0], cents[:, 1], bins=40)
        bx, by = np.unravel_index(np.argmax(counts), counts.shape)
        cx = 0.5 * (xedges[bx] + xedges[bx + 1])
        cy = 0.5 * (yedges[by] + yedges[by + 1])
        log.info("Zoom centred on densest patch at (%.0f, %.0f) px", cx, cy)
        written += render(
            adata, sample_dir, out_dir, graph=args.graph, label_key=args.label_key,
            region=(cx - half, cx + half, cy - half, cy + half),
            figsize_in=min(args.figsize_in, 24.0), dpi=args.dpi, tag="zoom",
            **edge_kwargs, **layer_kwargs,
        )

    print("\nWrote:")
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
