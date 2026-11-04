"""Fixed-size image windows centered on cells, with the ego disc zeroed.

A crop covers ``2 * half_um`` micrometres, cut at the slide's level-0 pixels
(``round(2 * half_um / mpp)`` px), zero-padded where it runs past the slide
edge, resampled to ``out_size`` px (bilinear, ``scipy.ndimage.zoom`` order 1)
and, for ``mask='ego'``, zeroed inside a disc of ``mask_radius_um`` about its
center. The disc is fixed and round, so the hole is identical for every cell
and carries no information about the one removed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

#: Ways to hide part of a crop: the whole window, or the ego disc zeroed.
MASK_MODES = ("none", "ego")
#: Radius of the ego disc. On the ovarian slide (407,120 cells) the radius a
#: disc at the centroid needs to contain the cell runs p50 6.3 um, p99 14.3,
#: p99.9 18.3, max 36.8; 25 um covers all but six, which are not embedded.
DEFAULT_MASK_RADIUS_UM = 25.0
#: Edge of the image region decoded at once. One region is read per block and
#: every crop in it sliced from memory.
DEFAULT_BLOCK_PX = 4096


def ego_disc(out_size: int, radius_um: float, field_um: float) -> np.ndarray:
    """``(out_size, out_size)`` bool, True where the crop is kept (outside the disc)."""
    center = (out_size - 1) / 2.0
    radius_px = radius_um / (field_um / out_size)
    y, x = np.ogrid[:out_size, :out_size]
    return ((x - center) ** 2 + (y - center) ** 2) > radius_px ** 2


def _shape(image) -> tuple[int, int]:
    if isinstance(image, np.ndarray):
        return image.shape[0], image.shape[1]
    from discell.image.tiff import image_shape

    height, width, _ = image_shape(image)
    return height, width


def _read(image, x0: int, y0: int, x1: int, y1: int, channels) -> np.ndarray:
    if isinstance(image, np.ndarray):
        window = image[y0:y1, x0:x1]
        return window if channels is None else window[..., list(channels)]
    from discell.image.tiff import read_raw_window

    return read_raw_window(image, x0, y0, x1, y1, channels)


def iter_crop_blocks(image: str | Path | np.ndarray, centers_px: np.ndarray,
                     microns_per_pixel: float, cells: Sequence[int] | None = None, *,
                     half_um: float = 64.0, channels: Sequence[int] | None = None,
                     out_size: int | None = 256, block_px: int = DEFAULT_BLOCK_PX,
                     pad_value: float = 0.0, mask: str = "ego",
                     mask_radius_um: float = DEFAULT_MASK_RADIUS_UM
                     ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(cell_indices, crops)`` block by block; crops are ``(B, H, W, C)``.

    *image* is a level-0 OME-TIFF path or an in-memory ``(H, W, C)`` array;
    *centers_px* are the crop centers in its pixels (``(N, 2)``, x then y);
    *cells* selects rows of *centers_px* (default all). Crops keep the image
    dtype. Cells are grouped by the image block their window starts in, so
    the output order is by block, not by *cells*.
    """
    if mask not in MASK_MODES:
        raise ValueError(f"mask must be one of {list(MASK_MODES)}, not {mask!r}")
    height, width = _shape(image)
    size = int(round(2 * half_um / float(microns_per_pixel)))
    edge = out_size or size
    keep = ego_disc(edge, mask_radius_um, 2 * half_um) if mask == "ego" else None

    centers = np.asarray(centers_px, dtype=np.float64)
    cells = (np.arange(len(centers)) if cells is None
             else np.asarray(list(cells), dtype=np.int64))
    x0 = np.round(centers[cells, 0] - size / 2).astype(np.int64)
    y0 = np.round(centers[cells, 1] - size / 2).astype(np.int64)
    block_ids = (y0 // block_px) * (width // block_px + 2) + (x0 // block_px)
    order = np.argsort(block_ids, kind="stable")
    sorted_ids = block_ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, len(order)]

    for a, b in zip(starts[:-1], starts[1:]):
        group = order[a:b]
        if len(group) == 0:
            continue
        gx0, gy0 = x0[group], y0[group]
        rx0, ry0 = max(0, int(gx0.min())), max(0, int(gy0.min()))
        rx1, ry1 = min(width, int(gx0.max()) + size), min(height, int(gy0.max()) + size)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        region = _read(image, rx0, ry0, rx1, ry1, channels)
        crops = []
        for k in range(len(group)):
            sx0, sy0 = int(gx0[k]) - rx0, int(gy0[k]) - ry0
            window = region[max(0, sy0): sy0 + size, max(0, sx0): sx0 + size]
            pad_y = (max(0, -sy0), max(0, size - window.shape[0] - max(0, -sy0)))
            pad_x = (max(0, -sx0), max(0, size - window.shape[1] - max(0, -sx0)))
            if pad_y != (0, 0) or pad_x != (0, 0):
                window = np.pad(window, (pad_y, pad_x, (0, 0)), mode="constant",
                                constant_values=pad_value)
            if out_size is not None and window.shape[0] != out_size:
                from scipy.ndimage import zoom

                factor = out_size / window.shape[0]
                window = zoom(window, (factor, factor, 1), order=1)[:out_size, :out_size]
            if keep is not None:
                window = np.where(keep[..., None], window, 0).astype(window.dtype)
            crops.append(window)
        yield cells[group], np.stack(crops)
