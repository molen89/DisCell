#!/usr/bin/env python3
"""Reading Xenium morphology TIFFs.

The one place that touches ``tifffile``. Two readers, because the callers want
different things and conflating them is how a model ends up trained on
percentile-stretched pixels:

:func:`read_raw_window`
    native dtype, no normalisation -- what a model should see
:func:`read_image_window`
    pyramid-aware, percentile-stretched 8-bit RGB -- what a figure should show

Both the preprocessing side (crops, plotting) and the consumer side
(:mod:`discell.plotting.batch_view`) import from here, so neither has to reach
across into the other.

``tifffile`` handles the tiled, strip-based and JPEG-compressed BigTIFFs in this
cohort without decoding a whole slide to read one window.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("discell.tiff")

#: Default composite for Xenium morphology: boundary in red, nuclei in green,
#: interior RNA in blue. DAPI alone is the sparsest of the four and a poor
#: backdrop; the boundary channel carries the tissue architecture.
XENIUM_DEFAULT_CHANNELS = (1, 0, 2)

def find_tissue_image(sample_dir: str | Path) -> Path | None:
    """Locate the morphology image for a sample.

    Takes a string too: ``uns["xenium_dir"]`` comes back from the h5ad as one.
    """
    sample_dir = Path(sample_dir)
    patterns = (
        # Channel 0000 is DAPI, which reads best under the polygons.
        "morphology_focus/morphology_focus_0000.ome.tif",
        "morphology_focus/morphology_focus_*.ome.tif",
        "morphology.ome.tif",
    )
    for pattern in patterns:
        hits = sorted(sample_dir.glob(pattern))
        if hits:
            return hits[0]
    return None

def is_xenium_morphology(path: Path) -> bool:
    return "morphology" in Path(path).name


# -- raw access: native dtype, for models ---------------------------------


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


# -- display access: stretched 8-bit RGB, for figures ---------------------


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
    channels: Sequence[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Read the window ``[y0:y1, x0:x1]`` of a pyramidal OME-TIFF, as 8-bit RGB.

    The coarsest pyramid level that still meets *max_px* is used, rather than
    decimating full-resolution pixels. Returns ``(rgb, scale)``.
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

        sx0, sx1 = int(x0 / factor), int(np.ceil(x1 / factor))
        sy0, sy1 = int(y0 / factor), int(np.ceil(y1 / factor))

        import zarr

        store = handle.aszarr(series=0, level=chosen)
        arr = zarr.open(store, mode="r")
        # Xenium morphology is (C, Y, X): the channels live in sibling files,
        # but tifffile resolves them through this one handle. Slicing it as 2-D
        # silently returns an empty array.
        wanted = [c for c in (channels or [0]) if 0 <= c < arr.shape[0]]
        if not wanted:
            raise ValueError(f"no valid channel in {channels} for an image "
                             f"with {arr.shape[0]} channels")
        planes = [np.asarray(arr[c, sy0:sy1, sx0:sx1]) for c in wanted]
        window = _compose_rgb(planes)
        if len(planes) > 1:
            log.info("  composite channels %s -> RGB", wanted)
        store.close()

    scale = 1.0 / factor
    longest = max(window.shape[0], window.shape[1])
    if longest > max_px:
        step = int(np.ceil(longest / max_px))
        window = window[::step, ::step]
        scale /= step
    log.info("  read image window in %.1fs, shape=%s", time.time() - started, window.shape)
    return window, scale
