#!/usr/bin/env python3
"""Fixed-size image windows centred on individual cells.

Two entry points:

:func:`crop_cell` / :func:`crop_cells`
    raw arrays for a model -- always the same pixel shape, read at pyramid
    level 0 (0.2125 um/px on Xenium), zero-padded at the slide edge
:func:`plot_cell` / :func:`plot_cell_grid`
    the same window rendered for inspection, optionally with the cell outline,
    its neighbours and the graph edges drawn on top

The window is ``half_um`` in every direction from the cell's anchor point, so a
default of 10 um gives a 20x20 um field -- 94x94 px at Xenium's 0.2125 um/px
HD. Because that differs by platform, pass ``out_size`` to resample every crop
to one shape before it reaches a model.

Usage::

    python -m discell.data.crops --sample <dir> --cell 1234 --half-um 10 --out crop.png
    python -m discell.data.crops --sample <dir> --grid 24 --half-um 10 --graph voronoi
    python -m discell.data.crops --sample <dir> --export crops.npz --limit 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from discell import paths

log = logging.getLogger("discell.data.crops")

DEFAULT_HALF_UM = 10.0
DEFAULT_ANCHOR = "centroid"


@dataclass
class CellCrop:
    """One fixed-size window around a cell."""

    image: np.ndarray                # (H, W, C)
    cell_index: int
    cell_id: str
    center_px: tuple[float, float]
    bounds_px: tuple[int, int, int, int]   # x0, y0, x1, y1 in slide coordinates
    half_um: float
    microns_per_pixel: float
    channels: tuple[int, ...]
    padded: bool                     # window ran past the slide edge

    @property
    def size_px(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]

    def __repr__(self) -> str:
        return (
            f"CellCrop({self.cell_id}, {self.image.shape}, "
            f"{2 * self.half_um:g}um, {self.microns_per_pixel:.4f}um/px"
            f"{', padded' if self.padded else ''})"
        )


def image_shape(path: Path) -> tuple[int, int, int]:
    """``(height, width, n_channels)`` of level 0."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        axes, shape = series.axes, series.levels[0].shape
        height = shape[axes.index("Y")] if "Y" in axes else shape[0]
        width = shape[axes.index("X")] if "X" in axes else shape[1]
        if "C" in axes:
            channels = shape[axes.index("C")]
        elif "S" in axes:
            channels = shape[axes.index("S")]
        else:
            channels = 1
    return height, width, channels


def read_raw_window(
    path: Path, x0: int, y0: int, x1: int, y1: int, channels: Sequence[int] | None = None
) -> np.ndarray:
    """Read ``[y0:y1, x0:x1]`` at level 0 without stretching, as ``(H, W, C)``.

    Unlike the plotting reader this keeps the native dtype and does no percentile
    normalisation -- a model should see the real values, and any stretch should
    be a documented preprocessing step rather than a side effect of reading.
    """
    import tifffile
    import zarr

    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        axes = series.axes
        store = handle.aszarr(series=0, level=0)
        arr = zarr.open(store, mode="r")
        if arr.ndim == 3 and axes.startswith("C"):
            wanted = list(channels) if channels is not None else list(range(arr.shape[0]))
            planes = [np.asarray(arr[c, y0:y1, x0:x1]) for c in wanted]
            window = np.stack(planes, axis=-1)
        elif arr.ndim == 3:
            window = np.asarray(arr[y0:y1, x0:x1, :])
            if channels is not None:
                window = window[..., list(channels)]
        else:
            window = np.asarray(arr[y0:y1, x0:x1])[..., None]
        store.close()
    return window


def _anchor_point(adata, index: int, anchor: str) -> tuple[float, float]:
    if anchor == "representative" and "rep_x_px" in adata.obs:
        # Guaranteed inside the outline; matters for the rare concave cell whose
        # centre of mass falls outside it.
        return (float(adata.obs["rep_x_px"].iloc[index]),
                float(adata.obs["rep_y_px"].iloc[index]))
    point = adata.obsm["spatial"][index]
    return float(point[0]), float(point[1])


def resolve_cell(adata, cell: int | str) -> int:
    """Accept a positional index or a cell id."""
    if isinstance(cell, (int, np.integer)):
        return int(cell)
    matches = np.flatnonzero(np.asarray(adata.obs_names, dtype=str) == str(cell))
    if not len(matches):
        raise KeyError(f"cell {cell!r} not found")
    return int(matches[0])


def crop_cell(
    adata,
    image_path: str | Path,
    cell: int | str,
    half_um: float = DEFAULT_HALF_UM,
    channels: Sequence[int] | None = None,
    out_size: int | None = None,
    anchor: str = DEFAULT_ANCHOR,
    pad_value: float = 0.0,
) -> CellCrop:
    """Fixed-size window centred on one cell.

    The size in pixels is derived once from *half_um* and the slide's
    microns-per-pixel, so every crop from a sample has identical shape. Windows
    that run past the slide edge are padded rather than clipped, which keeps that
    guarantee for cells near the tissue border.
    """
    image_path = Path(image_path)
    index = resolve_cell(adata, cell)
    mpp = float(adata.uns["microns_per_pixel"])
    height, width, _ = image_shape(image_path)

    size = int(round(2 * half_um / mpp))
    cx, cy = _anchor_point(adata, index, anchor)
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x1, y1 = x0 + size, y0 + size

    # Clamp for reading, then pad back so the shape never varies.
    rx0, ry0 = max(0, x0), max(0, y0)
    rx1, ry1 = min(width, x1), min(height, y1)
    padded = (rx0, ry0, rx1, ry1) != (x0, y0, x1, y1)

    if rx1 <= rx0 or ry1 <= ry0:
        raise ValueError(f"cell {index} at ({cx:.0f},{cy:.0f}) lies outside the image")

    window = read_raw_window(image_path, rx0, ry0, rx1, ry1, channels)
    if padded:
        window = np.pad(
            window,
            ((ry0 - y0, y1 - ry1), (rx0 - x0, x1 - rx1), (0, 0)),
            mode="constant", constant_values=pad_value,
        )

    if out_size is not None and window.shape[0] != out_size:
        from scipy.ndimage import zoom

        factor = out_size / window.shape[0]
        window = zoom(window, (factor, factor, 1), order=1)
        window = window[:out_size, :out_size]

    used = tuple(channels) if channels is not None else tuple(range(window.shape[-1]))
    return CellCrop(
        image=window,
        cell_index=index,
        cell_id=str(adata.obs_names[index]),
        center_px=(cx, cy),
        bounds_px=(x0, y0, x1, y1),
        half_um=half_um,
        microns_per_pixel=mpp,
        channels=used,
        padded=padded,
    )


#: Edge of the image region decoded at once when cropping in bulk. The morphology
#: images are 1024x1024 JPEG2000 tiles: one 256px window straddles up to four of
#: them and costs ~63ms, while a whole tile decodes in ~5.6ms. With ~208 cells
#: per tile, decoding a large block once and cutting every crop out of RAM turns
#: the per-cell cost into a rounding error.
DEFAULT_BLOCK_PX = 4096


def iter_crop_blocks(
    adata,
    image_path: str | Path,
    cells: Sequence[int],
    half_um: float = DEFAULT_HALF_UM,
    channels: Sequence[int] | None = None,
    out_size: int | None = None,
    anchor: str = DEFAULT_ANCHOR,
    block_px: int = DEFAULT_BLOCK_PX,
    pad_value: float = 0.0,
):
    """Yield ``(cell_indices, crops)`` block by block, decoding each region once.

    Cells are grouped by the image block their window falls in; each block is
    read in one call and every crop is then sliced out of memory. Crops that run
    past the slide edge are padded exactly as :func:`crop_cell` does, so the
    output is identical -- only far faster.
    """
    image_path = Path(image_path)
    mpp = float(adata.uns["microns_per_pixel"])
    height, width, _ = image_shape(image_path)
    size = int(round(2 * half_um / mpp))

    cells = np.asarray(list(cells), dtype=np.int64)
    if anchor == "representative" and "rep_x_px" in adata.obs:
        cx = adata.obs["rep_x_px"].to_numpy()[cells]
        cy = adata.obs["rep_y_px"].to_numpy()[cells]
    else:
        points = np.asarray(adata.obsm["spatial"], dtype=np.float64)[cells]
        cx, cy = points[:, 0], points[:, 1]

    x0 = np.round(cx - size / 2).astype(np.int64)
    y0 = np.round(cy - size / 2).astype(np.int64)

    block_ids = (y0 // block_px) * (width // block_px + 2) + (x0 // block_px)
    order = np.argsort(block_ids, kind="stable")

    for _, group in _grouped(block_ids[order], order):
        gx0, gy0 = x0[group], y0[group]
        # One region covering every window in this block, clipped to the slide.
        rx0, ry0 = max(0, int(gx0.min())), max(0, int(gy0.min()))
        rx1 = min(width, int(gx0.max()) + size)
        ry1 = min(height, int(gy0.max()) + size)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        region = read_raw_window(image_path, rx0, ry0, rx1, ry1, channels)

        crops = []
        for k in range(len(group)):
            sx0, sy0 = int(gx0[k]) - rx0, int(gy0[k]) - ry0
            window = region[max(0, sy0) : sy0 + size, max(0, sx0) : sx0 + size]
            pad_y = (max(0, -sy0), max(0, size - window.shape[0] - max(0, -sy0)))
            pad_x = (max(0, -sx0), max(0, size - window.shape[1] - max(0, -sx0)))
            if pad_y != (0, 0) or pad_x != (0, 0):
                window = np.pad(window, (pad_y, pad_x, (0, 0)),
                                mode="constant", constant_values=pad_value)
            if out_size is not None and window.shape[0] != out_size:
                from scipy.ndimage import zoom

                factor = out_size / window.shape[0]
                window = zoom(window, (factor, factor, 1), order=1)[:out_size, :out_size]
            crops.append(window)
        yield cells[group], np.stack(crops)


def _grouped(sorted_keys: np.ndarray, values: np.ndarray):
    """Split *values* wherever *sorted_keys* changes."""
    if len(sorted_keys) == 0:
        return
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    for chunk in np.split(np.arange(len(sorted_keys)), boundaries):
        yield sorted_keys[chunk[0]], values[chunk]


def crop_cells(
    adata,
    image_path: str | Path,
    cells: Sequence[int | str] | None = None,
    half_um: float = DEFAULT_HALF_UM,
    channels: Sequence[int] | None = None,
    out_size: int | None = None,
    anchor: str = DEFAULT_ANCHOR,
    stack: bool = True,
):
    """Crop many cells. Returns a stacked ``(n, H, W, C)`` array, or the list."""
    cells = list(range(adata.n_obs)) if cells is None else list(cells)
    started = time.time()
    crops = [
        crop_cell(adata, image_path, c, half_um, channels, out_size, anchor)
        for c in cells
    ]
    n_padded = sum(c.padded for c in crops)
    log.info(
        "Cropped %d cells at %gum (%s px) in %.1fs%s",
        len(crops), 2 * half_um, "x".join(map(str, crops[0].size_px)) if crops else "?",
        time.time() - started,
        f", {n_padded} padded at the slide edge" if n_padded else "",
    )
    if not stack:
        return crops
    return np.stack([c.image for c in crops]), crops


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------


def _stretch(window: np.ndarray) -> np.ndarray:
    """Percentile stretch to 8-bit RGB, for display only."""
    from discell.plotting.cell_graph import _compose_rgb

    planes = [window[..., k] for k in range(window.shape[-1])]
    return _compose_rgb(planes)


def plot_cell(
    adata,
    image_path: str | Path,
    cell: int | str,
    half_um: float = DEFAULT_HALF_UM,
    channels: Sequence[int] | None = None,
    anchor: str = DEFAULT_ANCHOR,
    graph: str | None = "voronoi",
    label_key: str | None = None,
    draw_polygon: bool = True,
    draw_neighbours: bool = True,
    draw_edges: bool = True,
    ax=None,
    title: bool = True,
):
    """Render one cell's window, optionally with its outline and neighbours."""
    import matplotlib.pyplot as plt

    crop = crop_cell(adata, image_path, cell, half_um, channels, anchor=anchor)
    x0, y0, x1, y1 = crop.bounds_px
    index = crop.cell_index

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.imshow(_stretch(crop.image), extent=(x0, x1, y1, y0), interpolation="nearest",
              zorder=1)

    polys = adata.uns.get("polygons")
    cents = np.asarray(adata.obsm["spatial"], dtype=float)
    label_key = label_key or adata.uns.get("default_label")

    neighbours: list[int] = []
    if graph and f"{graph}_connectivities" in adata.obsp:
        neighbours = list(adata.obsp[f"{graph}_connectivities"][index].indices)

    if draw_neighbours and polys is not None:
        for j in neighbours:
            if x0 <= cents[j, 0] <= x1 or True:  # clip happens via the axes limits
                xs, ys = polys[j].exterior.xy
                ax.plot(xs, ys, lw=0.8, color="#00e5ff", alpha=0.85, zorder=3)
    if draw_edges and neighbours:
        for j in neighbours:
            ax.plot([cents[index, 0], cents[j, 0]], [cents[index, 1], cents[j, 1]],
                    lw=0.8, color="#ffd400", alpha=0.9, zorder=4)
    if draw_polygon and polys is not None:
        xs, ys = polys[index].exterior.xy
        ax.plot(xs, ys, lw=1.8, color="#ff2d55", zorder=5)
    ax.plot(*crop.center_px, marker="+", ms=8, mew=1.4, color="white", zorder=6)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        label = ""
        if label_key and label_key in adata.obs:
            label = f"\n{adata.obs[label_key].iloc[index]}"
        ax.set_title(f"{crop.cell_id}  ({2 * half_um:g} µm, {crop.size_px[0]}px)"
                     f"{label}", fontsize=7)
    return ax, crop


def plot_cell_grid(
    adata,
    image_path: str | Path,
    cells: Sequence[int | str],
    half_um: float = DEFAULT_HALF_UM,
    channels: Sequence[int] | None = None,
    cols: int = 8,
    graph: str | None = "voronoi",
    label_key: str | None = None,
    **kwargs,
):
    """A montage of per-cell windows."""
    import matplotlib.pyplot as plt

    cells = list(cells)
    rows = int(np.ceil(len(cells) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.35 * rows), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(cells):]:
        ax.axis("off")
    for ax, cell in zip(axes, cells):
        plot_cell(adata, image_path, cell, half_um, channels, graph=graph,
                  label_key=label_key, ax=ax, **kwargs)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", required=True)
    parser.add_argument("--half-um", type=float, default=DEFAULT_HALF_UM,
                        help="window half-width in microns (10 -> a 20x20um field)")
    parser.add_argument("--cell", default=None, help="cell index or id")
    parser.add_argument("--grid", type=int, default=None, help="montage of N random cells")
    parser.add_argument("--channels", default=None, help="e.g. '1,0,2'")
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR,
                        choices=("centroid", "representative"))
    parser.add_argument("--graph", default="voronoi", help="'none' to hide the graph")
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--no-polygon", action="store_true")
    parser.add_argument("--no-neighbours", action="store_true")
    parser.add_argument("--no-edges", action="store_true")
    parser.add_argument("--out", default="crop.png",
                        help="filename inside this sample's dataset figures dir, "
                             "or a full path")
    parser.add_argument("--export", default=None,
                        help="write crops for many cells to .npz (image + cell ids)")
    parser.add_argument("--limit", type=int, default=None, help="cells to export")
    parser.add_argument("--out-size", type=int, default=None,
                        help="resample every crop to this pixel size")
    parser.add_argument("--max-cells", type=int, default=20000)
    parser.add_argument("--clip-radius-um", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from discell.data.xenium import find_tissue_image, load_sample
    from discell.plotting.cell_graph import XENIUM_DEFAULT_CHANNELS, is_xenium_morphology

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )

    adata, sample_dir = load_sample(args.sample, args.clip_radius_um, args.max_cells)
    ds = paths.dataset_for(sample_dir).ensure()
    out_path = (Path(args.out) if Path(args.out).parent != Path(".")
                else ds.figures / Path(args.out).name)
    image_path = find_tissue_image(sample_dir)
    if image_path is None:
        raise SystemExit(f"no image found under {sample_dir}")
    log.info("Image: %s", image_path.name)

    channels = ([int(c) for c in args.channels.split(",")] if args.channels
                else (list(XENIUM_DEFAULT_CHANNELS) if is_xenium_morphology(image_path)
                      else None))
    graph = None if args.graph in ("none", "None", "") else args.graph
    rng = np.random.default_rng(args.seed)

    if args.export:
        cells = list(range(adata.n_obs))
        if args.limit:
            cells = sorted(rng.choice(len(cells), min(args.limit, len(cells)), replace=False))
        stack, crops = crop_cells(adata, image_path, cells, args.half_um, channels,
                                  args.out_size, args.anchor)
        np.savez_compressed(
            args.export,
            images=stack,
            cell_ids=np.asarray([c.cell_id for c in crops]),
            half_um=args.half_um,
            microns_per_pixel=crops[0].microns_per_pixel,
            channels=np.asarray(crops[0].channels),
        )
        size = Path(args.export).stat().st_size / 1e6
        print(f"\nWrote {args.export}: {stack.shape} {stack.dtype}, {size:.1f} MB")
        return 0

    kwargs = dict(draw_polygon=not args.no_polygon,
                  draw_neighbours=not args.no_neighbours,
                  draw_edges=not args.no_edges)

    if args.grid:
        cells = sorted(rng.choice(adata.n_obs, min(args.grid, adata.n_obs), replace=False))
        fig = plot_cell_grid(adata, image_path, cells, args.half_um, channels,
                             graph=graph, label_key=args.label_key, **kwargs)
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        print(f"\nWrote {out_path}: {len(cells)} cells at {2 * args.half_um:g} µm")
        return 0

    cell = args.cell if args.cell is not None else int(rng.integers(adata.n_obs))
    cell = int(cell) if str(cell).lstrip("-").isdigit() else cell
    fig, ax = plt.subplots(figsize=(5, 5), dpi=220)
    _, crop = plot_cell(adata, image_path, cell, args.half_um, channels,
                        anchor=args.anchor, graph=graph, label_key=args.label_key,
                        ax=ax, **kwargs)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"\n{crop!r}\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
