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

    from discell.preprocess.crops import crop_cell, iter_crop_blocks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from discell.tiff import image_shape, read_raw_window

log = logging.getLogger("discell.preprocess.crops")

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


def _anchor_point(adata, index: int, anchor: str) -> tuple[float, float]:
    if anchor == "representative" and "rep_x_px" in adata.obs:
        # Guaranteed inside the outline; matters for the rare concave cell whose
        # centre of mass falls outside it.
        return (float(adata.obs["rep_x_px"].iloc[index]),
                float(adata.obs["rep_y_px"].iloc[index]))
    point = adata.obsm["spatial"][index]
    return float(point[0]), float(point[1])


def crop_cell(
    adata,
    image_path: str | Path,
    cell: int,
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
    index = int(cell)
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
