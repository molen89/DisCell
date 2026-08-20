#!/usr/bin/env python3
"""Per-cell image embeddings from KRONOS (MahmoodLab).

KRONOS projects **each channel separately** and adds a per-marker embedding, so
it takes any number of channels as long as each is identified -- a natural fit
for Xenium's 4-channel morphology stack, with a caveat about marker identity;
see :data:`XENIUM_MARKERS`.

The call contract, read from the source rather than the vendor README, whose
minimal example omits ``marker_ids`` even though the model asserts on it::

    batch = (batch - mean[None, :, None, None]) / std[None, :, None, None]
    patch_emb, marker_emb, token_emb = model(batch, marker_ids=marker_ids)

``batch`` is ``(B, n_markers, P, P)`` with P a multiple of 16; ``marker_ids`` is
a list of per-sample LongTensors; ``mean``/``std`` come from KRONOS's
``marker_metadata.csv``. The weights are gated -- pass ``--hf-token`` or set
``HF_TOKEN``.

Driven by :mod:`discell.preprocess.main`; :func:`embed_sample` is the entry
point and dispatches between the two model versions.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from discell import paths
from discell.preprocess.crops import DEFAULT_MASK_RADIUS_UM, MASK_MODES
from discell.preprocess.markers import (
    MarkerSpec,
    describe_markers,
    fetch_marker_metadata,
    intensity_scaling_factor,
    load_marker_metadata,
    resolve_markers,
)

log = logging.getLogger("discell.preprocess.kronos")

#: KRONOS ViT patch size; the input edge must be a multiple of this.
KRONOS_PATCH_MULTIPLE = 16

#: Default input edge in pixels, matching the KRONOS tutorials. At Xenium's
#: 0.2125 um/px this is a 54.4 um field: a cell (p99 extent 13.1 um) sits
#: centrally with roughly one cell-width of surrounding tissue on each side, and
#: no resampling is needed.
DEFAULT_PATCH_PX = 256






#: An id outside KRONOS's trained range (4-498 of num_markers=512). Marker
#: embeddings are deterministic sincos of the id, so an untrained id yields a
#: vector the model never associated with any protein. That is preferable to
#: borrowing a real marker's id, which would actively assert the channel is a
#: protein it is not.
OUT_OF_VOCAB_MARKER_ID = 505

#: Xenium Prime morphology_focus channels, mapped against KRONOS's real
#: 177-marker vocabulary (checked, not guessed).
#:
#: Channel 0 is a clean match. Channels 1 and 3 are antibody *cocktails* imaged
#: together -- every component is in the vocabulary, but KRONOS allows only one
#: marker identity per channel, so the default names the broadest component and
#: the alternatives are listed for override. Channel 2 is ribosomal RNA and the
#: vocabulary contains no RNA markers at all, so it gets an out-of-vocabulary id.
#:
#:   ch1 alternatives: CD45 (256, immune), E-CADHERIN (328, epithelial)
#:   ch3 alternatives: VIMENTIN (496, mesenchymal)
XENIUM_MARKERS: tuple[MarkerSpec, ...] = (
    MarkerSpec(0, "DAPI", exact=True),
    # Na/K-ATPase == ATP1A1; the pan-membrane component, present on all cells,
    # where CD45 and E-Cadherin are lineage-restricted.
    MarkerSpec(1, "NAKATP", exact=False),
    MarkerSpec(2, "18S", exact=False),      # rRNA: nothing comparable in vocabulary
    MarkerSpec(3, "A-SMA", exact=False),    # alphaSMA/Vimentin cocktail
)










def load_kronos(
    checkpoint: str = "hf_hub:MahmoodLab/kronos",
    cache_dir: str | Path = paths.MODELS,
    hf_token: str | None = None,
):
    """Load KRONOS. Returns ``(model, precision, embedding_dim)``."""
    from kronos import create_model_from_pretrained

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cfg = {"model_type": "vits16", "token_overlap": False}
    return create_model_from_pretrained(
        checkpoint_path=checkpoint, cfg_path=None, hf_auth_token=token,
        cache_dir=str(cache_dir), cfg=cfg,
    )


def _embed_blocks(adata, image_path, channels, cells, forward, *, half_um: float,
                  out_size: int | None, batch_size: int, block_px: int,
                  anchor: str = "centroid", mask: str = "none",
                  mask_radius_um: float = DEFAULT_MASK_RADIUS_UM,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Crop every cell and run *forward* over the batches.

    *forward* takes ``(B, C, H, W)`` float32 already divided by the dtype
    maximum, and returns ``(B, dim)``. Crops arrive block by block: each image
    region is decoded once and every window in it sliced from RAM, which is
    ~700x the per-cell path.
    """
    from discell.preprocess.crops import iter_crop_blocks

    cells = list(range(adata.n_obs)) if cells is None else list(cells)
    cell_names = np.asarray(adata.obs_names, dtype=str)
    outputs: list[np.ndarray] = []
    kept: list[str] = []
    started, done = time.time(), 0

    for indices, crops in iter_crop_blocks(
        adata, image_path, cells, half_um=half_um, channels=list(channels),
        out_size=out_size, anchor=anchor, block_px=block_px,
        mask=mask, mask_radius_um=mask_radius_um,
    ):
        for start in range(0, len(indices), batch_size):
            piece = crops[start : start + batch_size]
            # (B, H, W, C) -> (B, C, H, W), scaled to the [0, 1] range both
            # models were trained on.
            array = np.ascontiguousarray(piece.transpose(0, 3, 1, 2), dtype=np.float32)
            outputs.append(forward(array / intensity_scaling_factor(piece.dtype)))
            kept.extend(cell_names[indices[start : start + batch_size]])
        done += len(indices)
        rate = done / max(time.time() - started, 1e-9)
        log.info("  %d/%d cells (%.0f cells/s, eta %.1f min)",
                 done, len(cells), rate, (len(cells) - done) / max(rate, 1e-9) / 60)

    embeddings = np.concatenate(outputs, axis=0)
    log.info("Embedded %d cells -> %s in %.1f min", len(kept), embeddings.shape,
             (time.time() - started) / 60)
    return embeddings, np.asarray(kept, dtype=str)


def embed_cells(
    adata, image_path: str | Path, markers: Sequence[MarkerSpec], model, precision,
    cells: Sequence[int] | None = None, patch_px: int = DEFAULT_PATCH_PX,
    batch_size: int = 32, device: str = "cuda", half_um: float | None = None,
    block_px: int = 4096, mask: str = "none",
    mask_radius_um: float = DEFAULT_MASK_RADIUS_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed cells with KRONOS v1. Returns ``(embeddings, cell_ids)``."""
    import torch

    if patch_px % KRONOS_PATCH_MULTIPLE:
        raise ValueError(f"patch_px must be a multiple of {KRONOS_PATCH_MULTIPLE}")

    mpp = float(adata.uns["microns_per_pixel"])
    # Either take patch_px pixels natively, or a fixed micron field resampled.
    half, out_size = ((patch_px * mpp) / 2, None) if half_um is None else (half_um, patch_px)
    ids = [m.marker_id if m.resolved else i for i, m in enumerate(markers)]
    if any(not m.resolved for m in markers):
        log.warning("Using positional fallback ids for unresolved markers: %s", ids)
    mean = torch.tensor([m.mean for m in markers], dtype=precision, device=device)
    std = torch.tensor([m.std for m in markers], dtype=precision, device=device)
    model.eval().to(device)
    log.info("Crop: %d px, %.1f um field (%.4f um/px)%s, mask=%s",
             patch_px, 2 * half, 2 * half / patch_px,
             "" if out_size is None else " [resampled]", mask)

    def forward(array: np.ndarray) -> np.ndarray:
        batch = torch.from_numpy(array).to(device=device, dtype=precision)
        batch = (batch - mean[None, :, None, None]) / std[None, :, None, None]
        marker_ids = [torch.tensor(ids, device=device)] * batch.shape[0]
        with torch.no_grad():
            patch_emb, _marker, _token = model(batch, marker_ids=marker_ids)
        return patch_emb.float().cpu().numpy()

    return _embed_blocks(adata, image_path, [m.channel for m in markers], cells,
                         forward, half_um=half, out_size=out_size,
                         batch_size=batch_size, block_px=block_px,
                         mask=mask, mask_radius_um=mask_radius_um)


def write_embeddings(path: str | Path, embeddings: np.ndarray, cell_ids: np.ndarray,
                     meta: dict | None = None) -> Path:
    """Write in the layout :func:`discell.data.embeddings.load_embeddings` reads.

    ``.pt`` (the default) stores a torch tensor; ``.npz`` is still accepted for
    interoperability with tools that would rather not import torch.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".pt", ".pth"):
        torch.save(
            {
                "embeddings": torch.from_numpy(embeddings.astype(np.float32)),
                "cell_ids": list(map(str, cell_ids)),
                **(meta or {}),
            },
            path,
        )
    else:
        np.savez_compressed(path, embeddings=embeddings.astype(np.float32),
                            cell_ids=cell_ids, **(meta or {}))
    log.info("Wrote %s (%s, %.1f MB)", path, embeddings.shape,
             path.stat().st_size / 1e6)
    return path


def _embed_v1(adata, image_path, channels, cells, *, mask, mask_radius_um, patch_px, half_um, batch_size,
              block_px, device, cache_dir, hf_token):
    """KRONOS v1: markers by integer id, caller applies mean/std."""
    specs = [m for m in XENIUM_MARKERS if m.channel in channels]
    if len(specs) != len(channels):
        raise ValueError(f"unknown channel in {channels}; known: "
                         f"{[m.channel for m in XENIUM_MARKERS]}")
    fetched = fetch_marker_metadata(cache_dir=cache_dir, hf_token=hf_token)
    markers = resolve_markers(specs, load_marker_metadata(str(fetched) if fetched else None))

    log.info("KRONOS v1 markers:\n%s", describe_markers(markers))
    if any(not m.exact for m in markers):
        log.warning("channels flagged APPROXIMATE are antibody cocktails or a "
                    "non-protein (rRNA) channel; KRONOS assigns one marker identity "
                    "per channel, so no faithful id exists for them")
    if not any(m.resolved for m in markers):
        log.warning("no marker_metadata.csv, so ids fall back to channel position and "
                    "normalisation is a no-op -- embeddings will not match a canonical run")

    model, precision, dim = load_kronos(cache_dir=cache_dir, hf_token=hf_token)
    log.info("KRONOS loaded: precision=%s, embedding_dim=%s", precision, dim)
    embeddings, cell_ids = embed_cells(
        adata, image_path, markers, model, precision, cells=cells,
        patch_px=patch_px, batch_size=batch_size, device=device,
        half_um=half_um, block_px=block_px, mask=mask,
        mask_radius_um=mask_radius_um,
    )
    return embeddings, cell_ids, {
        "model": "kronos1",
        "channels": [m.channel for m in markers],
        "marker_names": [m.name for m in markers],
        "marker_ids": [m.marker_id for m in markers],
    }


def _embed_v2(adata, image_path, channels, cells, *, mask, mask_radius_um, patch_px,
              half_um, batch_size, block_px, device, cache_dir, hf_token,
              drop_unknown_markers):
    """KRONOS2: markers by name, normalisation inside the model."""
    from discell.preprocess.kronos2 import (
        NOVEL_MARKERS, PREFERRED_DAPI, XENIUM_MARKER_NAMES,
        describe_markers as describe_v2, embed_cells_v2, load_kronos2,
        measure_channel_stats, register_novel_markers, resolve_marker_names,
    )
    import torch

    names = [XENIUM_MARKER_NAMES[c] for c in channels]
    resolved = resolve_marker_names(names, cache_dir, hf_token)
    log.info("KRONOS2 markers:\n%s", describe_v2(names, resolved))

    # v2's marker embedder raises on an unknown name -- the documented default
    # statistics cover preprocess() only. Dropping the channel is the honest
    # option; substituting a vocabulary name would assert the channel is a
    # protein it is not.
    unknown = [(c, n) for c, n in zip(channels, names) if resolved.get(n) is False]
    registrable = [n for _, n in unknown if n in NOVEL_MARKERS]
    unusable = [(c, n) for c, n in unknown if n not in NOVEL_MARKERS]
    if unusable and drop_unknown_markers:
        keep = [(c, n) for c, n in zip(channels, names) if (c, n) not in unusable]
        channels, names = [c for c, _ in keep], [n for _, n in keep]
        log.warning("dropped %s", ", ".join(f"channel {c} ({n})" for c, n in unusable))
    elif unusable:
        raise ValueError(
            f"markers not in the vocabulary and not registrable: "
            f"{[n for _, n in unusable]}; pass --drop-unknown-markers to skip them")

    device = device if torch.cuda.is_available() else "cpu"
    model, dim = load_kronos2(cache_dir, hf_token, device)
    log.info("KRONOS2 loaded: embedding_dim=%d", dim)

    registered = []
    if registrable:
        mpp = float(adata.uns["microns_per_pixel"])
        stats = measure_channel_stats(adata, image_path, channels, names,
                                      half_um=(patch_px * mpp) / 2)
        registered = register_novel_markers(model, names, stats)
        for name in registered:
            mean, std = stats[name]
            log.info("  registered %s: mean=%.5f std=%.5f (measured on this sample)",
                     name, mean, std)

    embeddings, cell_ids = embed_cells_v2(
        adata, image_path, names, channels, model, cells=cells,
        patch_px=patch_px, batch_size=batch_size, device=device,
        block_px=block_px, preferred_dapi=PREFERRED_DAPI, half_um=half_um,
        mask=mask, mask_radius_um=mask_radius_um,
    )
    return embeddings, cell_ids, {
        "model": "kronos2",
        "channels": channels,
        "marker_names": names,
        "in_vocabulary": [bool(resolved.get(n)) for n in names],
        "registered_novel": registered,
    }


def embed_sample(
    adata, image_path: str | Path, model: str = "v1",
    channels: Sequence[int] = (0, 1, 2, 3), cells: Sequence[int] | None = None,
    patch_px: int = DEFAULT_PATCH_PX, half_um: float | None = None,
    batch_size: int = 128, block_px: int = 4096, device: str = "cuda",
    cache_dir: str | Path = paths.MODELS, hf_token: str | None = None,
    drop_unknown_markers: bool = False, mask: str = "none",
    mask_radius_um: float = DEFAULT_MASK_RADIUS_UM,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Embed a sample's cells. Returns ``(embeddings, cell_ids, meta)``.

    *meta* is what :func:`write_embeddings` stamps into the file so a stale
    artefact stays identifiable: model version, channels, marker identities and
    the resolution the crops were cut at.
    """
    if mask not in MASK_MODES:
        raise ValueError(f"mask must be one of {list(MASK_MODES)}, not {mask!r}")
    image_path = Path(image_path)
    channels = [int(c) for c in channels]
    cells = list(range(adata.n_obs)) if cells is None else list(cells)
    mpp = float(adata.uns["microns_per_pixel"])
    field_um = patch_px * mpp if half_um is None else 2 * half_um
    log.info("Embedding %d of %d cells from %s: %d px, %.1f um field "
             "(%.4f um/px), %d channels, mask=%s",
             len(cells), adata.n_obs, image_path.name, patch_px, field_um,
             field_um / patch_px, len(channels), mask)

    shared = dict(mask=mask, mask_radius_um=mask_radius_um, patch_px=patch_px,
                  half_um=half_um, batch_size=batch_size, block_px=block_px,
                  device=device, cache_dir=cache_dir, hf_token=hf_token)
    if model in ("v2", "kronos2"):
        out = _embed_v2(adata, image_path, channels, cells,
                        drop_unknown_markers=drop_unknown_markers, **shared)
    else:
        out = _embed_v1(adata, image_path, channels, cells, **shared)

    embeddings, cell_ids, meta = out
    return embeddings, cell_ids, {
        "patch_px": patch_px,
        # The resolution the crops were actually cut at, which is what makes an
        # embedding comparable to another -- not the slide's native um/px.
        "microns_per_pixel": field_um / patch_px,
        "slide_microns_per_pixel": mpp,
        "field_um": field_um,
        "mask": mask,
        "mask_radius_um": mask_radius_um if mask == "ego" else None,
        # Recorded so a stale file is identifiable later: everything embedded
        # before the intensity fix fed raw uint16 to a model expecting [0, 1].
        "intensity_scaled": True,
        "pretrained": True,
        **meta,
    }
