"""Pixel scaling shared by the image encoders."""

from __future__ import annotations

import numpy as np


def intensity_scaling_factor(dtype) -> float:
    """Divisor that puts raw pixels on KRONOS2's training scale (about [0, 1]).

    The KRONOS loaders divide by the dtype maximum: uint8 255, other unsigned
    65535, float 400. Xenium morphology is uint16.
    """
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return 255.0
    if np.issubdtype(dtype, np.unsignedinteger):
        return 65535.0
    if np.issubdtype(dtype, np.floating):
        return 400.0
    raise ValueError(f"no scaling factor for dtype {dtype!r}")
