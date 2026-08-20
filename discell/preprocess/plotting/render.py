#!/usr/bin/env python3
"""Write cell-graph figures to disk.

:func:`discell.preprocess.plotting.cell_graph.plot_cell_graph` draws onto an axes; this
manages the figure, the morphology overlay, the scale bar and the filename.

The metric and every filter are encoded in the name, so variants never
overwrite each other, and the dataset is carried by the directory::

    python -m discell.preprocess.plotting.render --sample <dir>
    python -m discell.preprocess.plotting.render --sample <dir> --touching --color-edges
    python -m discell.preprocess.plotting.render --sample <dir> --graph contact --min-apposed-um 1.0
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from discell.tiff import find_tissue_image
from discell.tiff import XENIUM_DEFAULT_CHANNELS, is_xenium_morphology, read_image_window
from discell.preprocess.plotting.cell_graph import (
    DEFAULT_DPI,
    DEFAULT_FIGSIZE_IN,
    DEFAULT_GRAPH,
    _legend_handles,
    plot_cell_graph,
)

log = logging.getLogger("discell.preprocess.plotting.render")

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
    channels: Sequence[int] | None = None,
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
                max_px=12000, channels=picked,
            )

        width_in = figsize_in
        height_in = max(4.0, figsize_in * aspect)
        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
        fig.patch.set_facecolor("white")

        if overlay:
            source_px = max(int(x1 - x0), int(y1 - y0))
            ax.imshow(image, extent=(x0, x1, y1, y0), interpolation="nearest", zorder=1)
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

        # Encode the metric and filter in the name so variants do not overwrite.
        # The dataset is carried by the directory, not the filename.
        parts = [graph, tag]
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
