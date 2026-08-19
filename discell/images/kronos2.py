#!/usr/bin/env python3
"""Per-cell image embeddings from KRONOS2 (MahmoodLab).

The successor to :mod:`discell.images.kronos`, and a different contract:

=====================  ==============================  ==========================
                       KRONOS (v1)                     KRONOS2 (v2)
=====================  ==============================  ==========================
architecture           marker-aware ViT-S/16           marker-aware DINOv2 ViT-B/16
embedding              384-d                           768-d
marker identity        integer ``marker_id``           marker **name** (string)
vocabulary             177 markers                     288 markers
normalisation          caller applies mean/std         ``model.preprocess`` does it
out-of-vocabulary      borrow an unused id             documented default stats
loading                ``create_model_from_pretrained``  ``AutoModel`` + remote code
=====================  ==============================  ==========================

Passing markers by name removes v1's most awkward step: there is no id to look
up and no risk of silently claiming a channel is a protein it is not. Names are
matched on a separator-insensitive key (``HLA-DR`` == ``hla_dr``), exactly, with
no fuzzy aliasing -- so an unknown marker falls back to default statistics
rather than being conflated with a similar one.

Both weights are gated; request access at
https://huggingface.co/MahmoodLab/KRONOS2 and authenticate as for v1.

Usage::

    python -m discell.images.kronos --model v2 --sample <dir> --out emb2.pt
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("discell.images.kronos2")

#: Reference tissue-patch size from the model card.
DEFAULT_PATCH_PX = 256

#: v2 resolves markers by name, so the Xenium channels are given the vocabulary
#: spelling directly. Checked against the shipped marker_metadata.csv:
#: DAPI, NAKATP and A-SMA all resolve; 18S does not appear (the vocabulary holds
#: no RNA species), and falls back to the documented default statistics.
XENIUM_MARKER_NAMES = ("DAPI", "NAKATP", "18S", "A-SMA")

#: Channel treated as the nuclear stain by ``model.preprocess``.
PREFERRED_DAPI = "DAPI"


#: 18S rRNA is absent from the 288-marker vocabulary, and v2 rejects unknown
#: markers outright. Rather than drop the channel or borrow another protein's
#: identity, it is registered as a *novel marker* through the model's own
#: ``register_additional_markers`` hook. A novel marker must reuse an existing
#: compartment/family: ``cytoplasm`` because ribosomal RNA is predominantly
#: cytoplasmic and 10x describes this as the interior-RNA stain, and ``qc_mask``
#: ("derived mask/pseudo-channel") because it is not a protein and no protein
#: family honestly fits.
NOVEL_MARKERS = {
    "18S": {
        "compartment": "cytoplasm",
        "family": "qc_mask",
        "compartment_desc": "cytoplasmic",
        "family_desc": "Derived mask/pseudo-channel",
        "marker_full_name": "18S ribosomal RNA (interior RNA stain)",
    },
}


def register_novel_markers(model, names: Sequence[str], stats: dict,
                           out_csv: str | Path = "model_assets/kronos2_novel.csv") -> list:
    """Register out-of-vocabulary channels so v2 accepts all of them.

    *stats* maps marker name to ``(mean, std)`` measured on the actual crops --
    the model's own docs specify the caller computes these before registering,
    since a novel marker has no pretraining statistics to fall back on.
    """
    import pandas as pd

    novel = [n for n in names if n in NOVEL_MARKERS]
    if not novel:
        return []

    rows = []
    for name in novel:
        mean, std = stats.get(name, (0.0254762, 0.0398634))   # documented defaults
        rows.append({"marker_name": name, **NOVEL_MARKERS[name],
                     "mean": float(mean), "std": float(std), "pretraining": "no"})
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    model.register_additional_markers(str(out_csv))
    for row in rows:
        log.info("registered novel marker %s (%s/%s) mean=%.5f std=%.5f",
                 row["marker_name"], row["compartment"], row["family"],
                 row["mean"], row["std"])
    return novel


def measure_channel_stats(adata, image_path, channels, names, half_um,
                          n_cells: int = 512, seed: int = 0) -> dict:
    """Mean/std per channel on scaled crops, for novel-marker registration."""
    from discell.data.crops import iter_crop_blocks
    from discell.images.kronos import intensity_scaling_factor

    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(adata.n_obs, min(n_cells, adata.n_obs), replace=False))
    collected = []
    for _, crops in iter_crop_blocks(adata, image_path, picked, half_um=half_um,
                                     channels=list(channels)):
        collected.append(crops.astype(np.float32) / intensity_scaling_factor(crops.dtype))
    stack = np.concatenate(collected, axis=0)
    return {name: (float(stack[..., k].mean()), float(stack[..., k].std()) or 1.0)
            for k, name in enumerate(names)}


def load_kronos2(cache_dir: str | Path = "./model_assets", hf_token: str | None = None,
                 device: str = "cuda"):
    """Load KRONOS2. Returns ``(model, embedding_dim)``."""
    import torch
    from transformers import AutoModel

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    started = time.time()
    model = AutoModel.from_pretrained(
        "MahmoodLab/KRONOS2", trust_remote_code=True,
        cache_dir=str(cache_dir), token=token,
    )
    model.eval().to(device)
    dim = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 768)
    log.info("KRONOS2 loaded in %.1fs (embedding dim %d) on %s",
             time.time() - started, dim, device)
    return model, dim


def resolve_marker_names(names: Sequence[str], cache_dir: str | Path = "./model_assets",
                         hf_token: str | None = None) -> dict:
    """Report which marker names the pretraining vocabulary knows."""
    import sys

    from huggingface_hub import hf_hub_download
    import pandas as pd

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        csv = hf_hub_download("MahmoodLab/KRONOS2", "marker_metadata.csv",
                              cache_dir=str(cache_dir), token=token)
        utils = hf_hub_download("MahmoodLab/KRONOS2", "marker_utils.py",
                                cache_dir=str(cache_dir), token=token)
    except Exception as exc:
        log.warning("could not check the marker vocabulary (%s)", type(exc).__name__)
        return {name: None for name in names}

    sys.path.insert(0, str(Path(utils).parent))
    from marker_utils import marker_match_key  # noqa: E402

    known = {marker_match_key(str(n)) for n in pd.read_csv(csv)["marker_name"]}
    return {name: marker_match_key(name) in known for name in names}


def describe_markers(names: Sequence[str], resolved: dict) -> str:
    lines = [f"{'ch':>3} {'marker':<12} {'in vocabulary':<14} note"]
    for channel, name in enumerate(names):
        status = {True: "yes", False: "no", None: "unchecked"}[resolved.get(name)]
        note = "" if resolved.get(name) else "falls back to default marker statistics"
        lines.append(f"{channel:>3} {name:<12} {status:<14} {note}")
    return "\n".join(lines)


def embed_cells_v2(
    adata,
    image_path: str | Path,
    marker_names: Sequence[str],
    channels: Sequence[int],
    model,
    cells: Sequence[int] | None = None,
    patch_px: int = DEFAULT_PATCH_PX,
    batch_size: int = 32,
    device: str = "cuda",
    block_px: int = 4096,
    anchor: str = "centroid",
    preferred_dapi: str = PREFERRED_DAPI,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed cells with KRONOS2. Returns ``(embeddings, cell_ids)``."""
    import torch

    from discell.data.crops import iter_crop_blocks

    mpp = float(adata.uns["microns_per_pixel"])
    half = (patch_px * mpp) / 2
    log.info("Crop: %d px, %.1f um field (%.4f um/px), markers %s",
             patch_px, 2 * half, mpp, list(marker_names))

    cells = list(range(adata.n_obs)) if cells is None else list(cells)
    cell_names = np.asarray(adata.obs_names, dtype=str)
    marker_names = list(marker_names)

    outputs: list[np.ndarray] = []
    kept: list[str] = []
    started, done = time.time(), 0
    for indices, crops in iter_crop_blocks(
        adata, image_path, cells, half_um=half, channels=channels,
        out_size=None, anchor=anchor, block_px=block_px,
    ):
        for start in range(0, len(indices), batch_size):
            piece = crops[start : start + batch_size]
            # (B, H, W, C) -> (B, C, H, W); preprocess wants numpy and does the
            # marker-aware z-score itself, so no mean/std is applied here.
            from discell.images.kronos import intensity_scaling_factor

            scaling = intensity_scaling_factor(piece.dtype)
            array = np.ascontiguousarray(piece.transpose(0, 3, 1, 2),
                                         dtype=np.float32) / scaling
            prepared = model.preprocess(array, marker_names,
                                        preferred_dapi=preferred_dapi)
            batch = torch.as_tensor(np.asarray(prepared), dtype=torch.float32,
                                    device=device)
            with torch.inference_mode():
                features = model(batch, marker_names)
            outputs.append(features.float().cpu().numpy())
            kept.extend(cell_names[indices[start : start + batch_size]])

        done += len(indices)
        elapsed = max(time.time() - started, 1e-9)
        log.info("  %d/%d cells (%.0f cells/s, eta %.1f min)",
                 done, len(cells), done / elapsed,
                 (len(cells) - done) / max(done / elapsed, 1e-9) / 60)

    embeddings = np.concatenate(outputs, axis=0)
    log.info("Embedded %d cells -> %s in %.1f min", len(kept), embeddings.shape,
             (time.time() - started) / 60)
    return embeddings, np.asarray(kept, dtype=str)
