"""Per-cell image descriptors from KRONOS2, with the ego disc zeroed.

:func:`embed` crops every cell from the four Xenium morphology channels: 256 px
crops at 0.5 um/px (a 128 um field) centered on the cell, a 25 um disc about the
center zeroed, zero padding at the slide edge, intensities divided by the dtype
maximum. A cell reaching beyond the disc (covering radius
``.obs['discell_covering_radius_um']``, from :func:`discell.read_xenium`, above the
radius; mostly elongated smooth-muscle cells) stays partly visible in its own
field and is embedded like any other (``large_cells="embed"``, the default).
``large_cells="zero"`` gives such cells a zero row instead, as the paper's
ovarian FFPE descriptor did for its 6. The encoder is KRONOS2
(:mod:`discell.image.kronos2`): 768-d, markers by name, the recipe of the
authors' KRONOS2 refit of Ovarian FFPE.

KRONOS v1 is not supported. The paper's image descriptor used KRONOS v1
(ViT-S/16, 384-d), which its authors have deprecated and closed to new access
requests, so this release supports KRONOS2 only and does not offer an exact
reproduction of the paper's image results. A KRONOS2 refit of Ovarian FFPE gave
the same breakdown point for every finding (supplement S4); the other sections were
not refitted with KRONOS2.

KRONOS2's weights are gated (CC BY-NC-ND 4.0) and never ship with this package:
use your own Hugging Face login (``hf auth login``, ``HF_TOKEN`` or ``hf_token=``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from discell.image.crops import DEFAULT_BLOCK_PX, DEFAULT_MASK_RADIUS_UM, iter_crop_blocks
from discell.image.markers import intensity_scaling_factor

log = logging.getLogger("discell.image.kronos")

#: The input edge must be a multiple of the ViT patch size.
PATCH_MULTIPLE = 16
#: The paper's recipe.
DEFAULT_PATCH_PX = 256
DEFAULT_TARGET_MPP = 0.5
DEFAULT_KEY = "X_kronos_masked"

ACCESS_MESSAGE = (
    "discell.image needs KRONOS2 ({missing}). Install the image extra with "
    "`pip install 'discell[image]'`. KRONOS2 is gated and its weights are not "
    "distributed with DisCell: request access at https://huggingface.co/MahmoodLab/KRONOS2 "
    "(licence CC BY-NC-ND 4.0), then log in with `hf auth login` or set HF_TOKEN. "
    "(KRONOS v1, which the paper's image descriptor used, is deprecated by its authors and "
    "closed to new access requests, and is not supported.) Without an image descriptor "
    "DisCell fits image-free (Config(image_key=None))."
)


def _centers_px(adata, microns_per_pixel: float) -> np.ndarray:
    if "spatial_px" in adata.obsm:
        return np.asarray(adata.obsm["spatial_px"], dtype=np.float64)
    pos = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if adata.uns.get("discell", {}).get("microns_per_pixel") is not None:
        return pos                                # already pixels
    return pos / float(microns_per_pixel)


def embed(adata, image_path: str | Path | np.ndarray | None = None, *,
          encoder: str | Callable[[np.ndarray], np.ndarray] = "kronos2",
          channels: Sequence[int] = (0, 1, 2, 3),
          patch_px: int = DEFAULT_PATCH_PX, target_mpp: float = DEFAULT_TARGET_MPP,
          mask: str = "ego", mask_radius_um: float = DEFAULT_MASK_RADIUS_UM,
          cells: Sequence[int] | None = None, key: str = DEFAULT_KEY,
          microns_per_pixel: float | None = None, batch_size: int = 128,
          block_px: int = DEFAULT_BLOCK_PX, device: str = "cuda",
          hf_token: str | None = None, cache_dir: str | Path | None = None,
          checkpoint: str | Path | None = None, large_cells: str = "embed",
          revision: str | None = None) -> np.ndarray:
    """Embed every cell's masked crop and write ``.obsm[key]``.

    Parameters
    ----------
    adata
        Cells with centroids (``.obsm['spatial_px']`` from :func:`discell.read_xenium`,
        or ``.obsm['spatial']`` in um).
    image_path
        The level-0 morphology OME-TIFF (default: ``morphology_focus_0000.ome.tif``
        of the Xenium directory :func:`discell.read_xenium` read), or an
        in-memory ``(H, W, C)`` array.
    encoder
        ``"kronos2"`` (default; 768-d), or a function from ``(B, C, P, P)`` float32
        crops already divided by the dtype maximum to ``(B, D)`` (for tests or
        another model).
    patch_px, target_mpp
        Crop edge in pixels and its resolution: the field is ``patch_px * target_mpp`` um.
    mask, mask_radius_um
        ``"ego"`` zeroes a disc of this radius about the center; ``"none"`` keeps
        the whole window.
    cells
        Row indices to embed (default all); the others get zero rows.
    large_cells
        With ``mask="ego"``: ``"embed"`` (default) embeds cells larger than the
        disc, partly visible; ``"zero"`` gives them a zero row (as the paper's
        ovarian FFPE descriptor did).
    microns_per_pixel
        Level-0 pixel size (default ``.uns['discell']['xenium']['pixel_size_um']``).
    checkpoint
        A local copy of the KRONOS2 hub repository (a directory).
    revision
        KRONOS2 hub revision (default the one the paper's KRONOS2 refit used).

    Returns the ``(N, D)`` float32 embedding; also sets
    ``.uns['discell']['image']`` (the recipe) and ``.obs[key + '_embedded']``.
    """
    if patch_px % PATCH_MULTIPLE:
        raise ValueError(f"patch_px must be a multiple of {PATCH_MULTIPLE}")
    meta = adata.uns.setdefault("discell", {})
    xenium = meta.get("xenium", {})
    if microns_per_pixel is None:
        microns_per_pixel = xenium.get("pixel_size_um")
        if microns_per_pixel is None:
            raise ValueError("pass microns_per_pixel (the image's level-0 pixel size)")
    if image_path is None:
        from discell.image.tiff import find_morphology

        image_path = find_morphology(xenium.get("path", "")) if xenium.get("path") else None
        if image_path is None:
            raise ValueError("pass image_path: no morphology image found")
    channels = [int(c) for c in channels]

    centers = _centers_px(adata, microns_per_pixel)
    name, names, novel = "custom", None, None
    if not callable(encoder):
        if encoder != "kronos2":
            # KRONOS v1 (the paper's descriptor) is deprecated and closed to new
            # access requests; this release supports KRONOS2 only
            raise ValueError(f"encoder {encoder!r} is not supported; use 'kronos2' "
                             "or a function")
        from discell.image import kronos2

        name = "kronos2"
        names = kronos2.marker_names(channels)
        revision = revision or kronos2.KRONOS2_REVISION
        model = kronos2.load_kronos2(hf_token, cache_dir, checkpoint, revision, device)
        novel = kronos2.novel_marker_stats(image_path, centers, microns_per_pixel,
                                           channels, names, patch_px=patch_px)
        kronos2.register_novel_markers(model, novel)
        encoder = kronos2.kronos2_encoder(model, names, device)

    if large_cells not in ("zero", "embed"):
        raise ValueError(f"large_cells must be 'zero' or 'embed', not {large_cells!r}")
    selected = np.arange(adata.n_obs) if cells is None else np.asarray(list(cells), np.int64)
    if mask == "ego" and large_cells == "zero":
        if "discell_covering_radius_um" in adata.obs:
            radius = adata.obs["discell_covering_radius_um"].to_numpy()[selected]
            too_large = radius > mask_radius_um
            if too_large.any():
                log.info("%d cells are larger than the %g um disc and get a zero row",
                         int(too_large.sum()), mask_radius_um)
            selected = selected[~too_large]
        else:
            log.warning("no .obs['discell_covering_radius_um']: cannot check that every "
                        "cell fits inside the %g um disc", mask_radius_um)

    half_um = patch_px * target_mpp / 2
    rows: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    started = time.time()
    for indices, crops in iter_crop_blocks(
            image_path, centers, microns_per_pixel, selected, half_um=half_um,
            channels=channels, out_size=patch_px, block_px=block_px, mask=mask,
            mask_radius_um=mask_radius_um):
        scale = intensity_scaling_factor(crops.dtype)
        for start in range(0, len(indices), batch_size):
            piece = crops[start:start + batch_size]
            array = np.ascontiguousarray(piece.transpose(0, 3, 1, 2), dtype=np.float32)
            outputs.append(np.asarray(encoder(array / scale), dtype=np.float32))
            rows.append(indices[start:start + batch_size])
    if not outputs:
        raise ValueError("no cell could be cropped")
    values = np.concatenate(outputs)
    out = np.zeros((adata.n_obs, values.shape[1]), dtype=np.float32)
    done = np.concatenate(rows)
    out[done] = values
    embedded = np.zeros(adata.n_obs, dtype=bool)
    embedded[done] = True
    adata.obsm[key] = out
    adata.obs[f"{key}_embedded"] = embedded
    meta["image"] = {
        "key": key, "model": name,
        "channels": channels, "patch_px": int(patch_px), "target_mpp": float(target_mpp),
        "field_um": float(patch_px * target_mpp), "mask": mask,
        "mask_radius_um": float(mask_radius_um) if mask == "ego" else None,
        "large_cells": large_cells if mask == "ego" else None,
        "slide_microns_per_pixel": float(microns_per_pixel),
        "marker_names": names,
        "novel_marker_stats": None if novel is None else {k: list(v) for k, v in novel.items()},
        "revision": revision if name == "kronos2" else None,
        "n_embedded": int(embedded.sum()), "minutes": round((time.time() - started) / 60, 2),
    }
    return out
