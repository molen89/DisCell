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

    python -m discell.plotting.compare_graphs --sample <dir> --out-dir figures/
    python -m discell.plotting.compare_graphs --sample <dir> --max-cells 20000 --overlay
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from discell.data.geometry import (
    _store_graph,
    build_contact_graph,
    build_voronoi_graph,
    graph_edge_frame,
)
from discell.data.xenium import find_tissue_image
from discell.plotting.cell_graph import plot_cell_graph, read_image_window

from discell import paths

log = logging.getLogger("discell.plotting.compare_graphs")

DEFAULT_TOLERANCE_UM = 1.0
DEFAULT_CLIP_RADIUS_UM = 30.0
DEFAULT_MAX_EDGE_UM = 100.0


def build_variants(
    adata,
    tolerance_um: float = DEFAULT_TOLERANCE_UM,
    clip_radius_um: float = DEFAULT_CLIP_RADIUS_UM,
    max_edge_um: float = DEFAULT_MAX_EDGE_UM,
) -> list[tuple[str, str]]:
    """Build and store the three variants. Returns ``[(prefix, title), ...]``."""
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

        # Contact graphs can have a shared wall of zero throughout -- on Xenium
        # touching pairs meet at a point, not along a wall -- which would render
        # every edge at minimum width and the bottom of the colormap, i.e.
        # invisible. Fall back to centroid distance for those panels.
        wall = adata.obsp[f"{prefix}_shared_wall_um"]
        wall_median = float(np.median(wall.data)) if wall.nnz else 0.0
        metric = "shared_wall_um" if wall_median > 0 else "centroid_dist_um"
        note = "" if wall_median > 0 else "  (shared wall is 0 — sized by centroid distance)"

        drawn = plot_cell_graph(
            adata, ax, graph=prefix, label_key=label_key, region=region,
            polygon_alpha=0.30 if overlay else 0.65,
            edge_color="#00e5ff" if overlay else "#111111",
            edge_alpha=0.9, node_size=10.0, max_linewidth=3.5,
            color_edges_by_metric=color_edges, edge_metric=metric,
            draw_nodes=True,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out-dir", default=None,
                        help="default: this sample's dataset figures directory")
    parser.add_argument("--tolerance-um", type=float, default=DEFAULT_TOLERANCE_UM)
    parser.add_argument("--clip-radius-um", type=float, default=DEFAULT_CLIP_RADIUS_UM)
    parser.add_argument("--max-edge-um", type=float, default=DEFAULT_MAX_EDGE_UM)
    parser.add_argument("--wall-tolerance-um", type=float, default=1.0,
                        help="apposed_wall_um tolerance; independent of --tolerance-um")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--label-key", default="cluster")
    parser.add_argument("--zoom-um", type=float, default=400.0)
    parser.add_argument("--region", nargs=4, type=float, default=None,
                        metavar=("X0", "X1", "Y0", "Y1"))
    parser.add_argument("--both", action="store_true",
                        help="write plain and overlay versions")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--figsize-in", type=float, default=16.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--stats-out", default=None, help="write the table to .csv")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from discell.data.xenium import load_sample

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )

    # Graphs are rebuilt here with controlled parameters, so skip the loader's.
    adata, sample_dir = load_sample(
        args.sample, args.clip_radius_um, args.max_cells,
        wall_tolerance_um=args.wall_tolerance_um,
    )
    label_key = args.label_key
    if label_key not in adata.obs:
        label_key = adata.uns.get("default_label", label_key)
        log.info("Using label key %r", label_key)

    ds = paths.dataset_for(sample_dir).ensure()
    out_dir = Path(args.out_dir) if args.out_dir else ds.figures
    log.info("Dataset %s -> %s", ds.dataset_id, out_dir)

    variants = build_variants(adata, args.tolerance_um, args.clip_radius_um, args.max_edge_um)

    table = statistics(adata, variants)
    print()
    print(table.to_string(index=False))
    print()
    if args.stats_out:
        stats_path = Path(args.stats_out)
        if stats_path.parent == Path("."):
            stats_path = out_dir / stats_path.name
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(stats_path, index=False)
        log.info("Wrote %s", stats_path)

    region = tuple(args.region) if args.region else densest_region(adata, args.zoom_um)
    log.info("Region: x %.0f..%.0f  y %.0f..%.0f px", *region)

    modes = [False, True] if args.both else [args.overlay]
    written = []
    for overlay in modes:
        written.append(render_comparison(
            adata, sample_dir, out_dir, variants, region,
            label_key=label_key, overlay=overlay,
            figsize_in=args.figsize_in, dpi=args.dpi,
        ))

    print("Wrote:")
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
