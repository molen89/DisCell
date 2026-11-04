"""Reading Xenium morphology OME-TIFFs at level 0, in the native dtype.

No normalization happens here: a model should see the real values, and any
stretch is a documented step of :mod:`discell.image.kronos`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def find_morphology(outs: str | Path) -> Path | None:
    """The morphology image of a Xenium output directory.

    ``morphology_focus/morphology_focus_0000.ome.tif`` first: on Xenium Prime its
    OME metadata names the sibling channel files, so the one handle reads all
    four channels.
    """
    outs = Path(outs)
    for pattern in ("morphology_focus/morphology_focus_0000.ome.tif",
                    "morphology_focus/morphology_focus_*.ome.tif", "morphology.ome.tif"):
        hits = sorted(outs.glob(pattern))
        if hits:
            return hits[0]
    return None


def image_shape(path: str | Path) -> tuple[int, int, int]:
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


def read_raw_window(path: str | Path, x0: int, y0: int, x1: int, y1: int,
                    channels: Sequence[int] | None = None) -> np.ndarray:
    """``[y0:y1, x0:x1]`` of level 0 as ``(H, W, C)`` in the native dtype."""
    import tifffile
    import zarr

    with tifffile.TiffFile(path) as handle:
        axes = handle.series[0].axes
        store = handle.aszarr(series=0, level=0)
        arr = zarr.open(store, mode="r")
        if arr.ndim == 3 and axes.startswith("C"):
            wanted = list(channels) if channels is not None else list(range(arr.shape[0]))
            window = np.stack([np.asarray(arr[c, y0:y1, x0:x1]) for c in wanted], axis=-1)
        elif arr.ndim == 3:
            window = np.asarray(arr[y0:y1, x0:x1, :])
            if channels is not None:
                window = window[..., list(channels)]
        else:
            window = np.asarray(arr[y0:y1, x0:x1])[..., None]
        store.close()
    return window
