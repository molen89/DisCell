#!/usr/bin/env python3
"""Render the exact crop that KRONOS is fed for one cell.

Shows each morphology channel separately, the RGB composite, and the same
window with the cell's own polygon and its neighbours' centroids drawn on top --
which is what makes the field of view concrete: at the default 256 px / 0.2125
um/px the crop is 54.4 um across and contains a median of 21 other cells, so the
embedding sees the neighbourhood, not just the cell.

Usage::

    python -m discell.preprocess.plotting.crop_view --sample <dir>
    python -m discell.preprocess.plotting.crop_view --sample <dir> --cell 12345 --patch-px 256
    python -m discell.preprocess.plotting.crop_view --sample <dir> --half-um 10   # tight crop
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from discell.preprocess.crops import crop_cell
from discell.preprocess.kronos import XENIUM_MARKERS

log = logging.getLogger("discell.preprocess.plotting.crop_view")


def _stretch(plane: np.ndarray) -> np.ndarray:
    """Percentile stretch to 8-bit. Xenium morphology is uint16 with most of the
    range unused, so raw values render as near-black."""
    low, high = np.percentile(plane, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    return np.clip((plane.astype(np.float32) - low) / (high - low), 0, 1)


def render_crop(adata, image_path: Path, index: int, out_path: Path,
                half_um: float, channels: Sequence[int], dpi: int = 200) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    crop = crop_cell(adata, image_path, index, half_um=half_um, channels=list(channels))
    image, (x0, y0, x1, y1) = crop.image, crop.bounds_px
    mpp = crop.microns_per_pixel
    names = {m.channel: m.name for m in XENIUM_MARKERS}

    fig, axes = plt.subplots(1, len(channels) + 2, figsize=(3.1 * (len(channels) + 2), 3.6),
                             dpi=dpi)
    for k, channel in enumerate(channels):
        axes[k].imshow(_stretch(image[..., k]), cmap="gray")
        axes[k].set_title(f"ch{channel} — {names.get(channel, '?')}", fontsize=10)

    # The composite the plotting code uses: boundary red, nuclei green, RNA blue.
    rgb = np.zeros((*image.shape[:2], 3), dtype=np.float32)
    for slot, channel in enumerate((1, 0, 2)):
        if channel in channels:
            rgb[..., slot] = _stretch(image[..., list(channels).index(channel)])
    axes[len(channels)].imshow(rgb)
    axes[len(channels)].set_title("composite", fontsize=10)

    ax = axes[len(channels) + 1]
    ax.imshow(rgb)
    polys = adata.uns.get("polygons")
    cents = np.asarray(adata.obsm["spatial"], dtype=float)
    inside = np.flatnonzero((cents[:, 0] >= x0) & (cents[:, 0] < x1)
                            & (cents[:, 1] >= y0) & (cents[:, 1] < y1))
    ax.scatter(cents[inside, 0] - x0, cents[inside, 1] - y0, s=14, c="#ffd166",
               edgecolors="black", linewidths=0.4, zorder=3)
    ax.scatter([cents[index, 0] - x0], [cents[index, 1] - y0], s=90, c="#ef476f",
               edgecolors="white", linewidths=1.2, zorder=4)
    if polys is not None:
        xs, ys = polys[index].exterior.xy
        ax.add_patch(MplPolygon(np.column_stack([np.asarray(xs) - x0, np.asarray(ys) - y0]),
                                fill=False, edgecolor="#ef476f", linewidth=2.0, zorder=5))
    ax.set_title(f"{len(inside) - 1} other cells in view", fontsize=10)

    bar = 10.0 / mpp
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        a.plot([6, 6 + bar], [image.shape[0] - 8] * 2, color="white", linewidth=3)
    axes[0].text(6, image.shape[0] - 12, "10 µm", color="white", fontsize=8, va="bottom")

    fig.suptitle(
        f"what KRONOS sees for cell {crop.cell_id} — {2 * half_um:.1f} µm field, "
        f"{image.shape[0]}×{image.shape[1]} px, {len(channels)} channels "
        f"({mpp:.4f} µm/px)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n{crop!r}")
    print(f"model input would be: (batch, {len(channels)}, {image.shape[0]}, {image.shape[1]})")
    print(f"other cells inside the crop: {len(inside) - 1}")
    print(f"Wrote {out_path}")
    return out_path
