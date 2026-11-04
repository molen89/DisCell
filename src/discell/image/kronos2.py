"""KRONOS2 (MahmoodLab), the default image encoder.

KRONOS2 is a marker-aware DINOv2 ViT-B/16 giving a 768-d CLS embedding. Unlike
KRONOS v1 it takes each channel's marker by **name** and z-scores the channels
itself (``model.preprocess``). The recipe is the one of the authors' KRONOS2 refit
of Ovarian FFPE: the four Xenium morphology
channels named ``DAPI``, ``NAKATP``, ``18S``, ``A-SMA`` (``DAPI`` as the nuclear
stain), crops as for v1 (256 px at 0.5 um/px, 25 um disc zeroed, divided by the
dtype maximum). ``18S`` rRNA is not in KRONOS2's marker vocabulary, so it is
registered as a novel marker (compartment ``cytoplasm``, family ``qc_mask``)
with a mean and standard deviation measured on the slide: 512 random cells
(seed 0), unmasked 256 px crops at the slide's native resolution.

Registering a novel marker makes the model encode its description with
BioLinkBERT-large (public, downloaded once by the model's own code). The weights
are gated (CC BY-NC-ND 4.0) and never ship with DisCell: request access at
https://huggingface.co/MahmoodLab/KRONOS2 and use your own Hugging Face login.
Loading runs the model's own code from the hub (``trust_remote_code``).
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from discell.image.crops import iter_crop_blocks
from discell.image.markers import intensity_scaling_factor

log = logging.getLogger("discell.image.kronos2")

KRONOS2_REPO = "MahmoodLab/KRONOS2"
#: The hub revision the authors' KRONOS2 refit used.
KRONOS2_REVISION = "e2a5473206c02d9973885b6445ef783f7e082951"
KRONOS2_DIM = 768

#: Marker name of each Xenium Prime ``morphology_focus`` channel, as KRONOS2 reads it.
XENIUM_MARKER_NAMES: dict[int, str] = {0: "DAPI", 1: "NAKATP", 2: "18S", 3: "A-SMA"}
#: Channel treated as the nuclear stain by ``model.preprocess``.
PREFERRED_DAPI = "DAPI"

#: Out-of-vocabulary channels registered as novel markers. A novel marker must
#: reuse an existing compartment and family: ``cytoplasm`` since ribosomal RNA is
#: mostly cytoplasmic (10x's interior-RNA stain), ``qc_mask`` ("derived
#: mask/pseudo-channel") since it is not a protein.
NOVEL_MARKERS: dict[str, dict[str, str]] = {
    "18S": {
        "compartment": "cytoplasm",
        "family": "qc_mask",
        "compartment_desc": "cytoplasmic",
        "family_desc": "Derived mask/pseudo-channel",
        "marker_full_name": "18S ribosomal RNA (interior RNA stain)",
    },
}
#: Cells and seed of the novel-marker statistics.
STATS_CELLS = 512
STATS_SEED = 0

_NEEDS = ("transformers", "omegaconf", "huggingface_hub", "tifffile", "zarr")


def require_kronos2() -> None:
    """Raise the one-paragraph access message when the extra is missing."""
    from importlib.util import find_spec

    from discell.image.kronos import ACCESS_MESSAGE

    missing = [m for m in _NEEDS if find_spec(m) is None]
    if missing:
        raise ImportError(ACCESS_MESSAGE.format(missing=", ".join(missing) + " not installed"))


def load_kronos2(hf_token: str | None = None, cache_dir: str | Path | None = None,
                 checkpoint: str | Path | None = None, revision: str = KRONOS2_REVISION,
                 device: str = "cuda"):
    """KRONOS2 in eval mode on *device*.

    *checkpoint* is a local copy of the hub repository (a directory); otherwise
    it is downloaded at *revision* with the user's own access.
    """
    require_kronos2()
    from transformers import AutoModel

    if checkpoint is None:
        from huggingface_hub import snapshot_download

        from discell.image.kronos import ACCESS_MESSAGE

        try:
            checkpoint = snapshot_download(
                KRONOS2_REPO, revision=revision, token=hf_token,
                cache_dir=None if cache_dir is None else str(cache_dir),
                ignore_patterns=["demo_image/*"])
        except Exception as exc:          # gated, unauthenticated, offline, ...
            raise PermissionError(ACCESS_MESSAGE.format(
                missing=f"could not download {KRONOS2_REPO}: {type(exc).__name__}")) from exc
    model = AutoModel.from_pretrained(str(checkpoint), trust_remote_code=True)
    return model.eval().to(device)


def marker_names(channels: Sequence[int]) -> list[str]:
    unknown = [c for c in channels if c not in XENIUM_MARKER_NAMES]
    if unknown:
        raise ValueError(f"unknown channel(s) {unknown}; known: {sorted(XENIUM_MARKER_NAMES)}")
    return [XENIUM_MARKER_NAMES[c] for c in channels]


def novel_marker_stats(image, centers_px: np.ndarray, microns_per_pixel: float,
                       channels: Sequence[int], names: Sequence[str],
                       patch_px: int = 256, n_cells: int = STATS_CELLS,
                       seed: int = STATS_SEED) -> dict[str, tuple[float, float]]:
    """``{name: (mean, std)}`` of each novel-marker channel on scaled crops.

    *n_cells* random rows of *centers_px* (all cells of the section), unmasked
    *patch_px* crops at the slide's native resolution, divided by the dtype maximum.
    """
    novel = [(c, n) for c, n in zip(channels, names) if n in NOVEL_MARKERS]
    if not novel:
        return {}
    n_obs = len(centers_px)
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(n_obs, min(n_cells, n_obs), replace=False))
    collected = []
    for _, crops in iter_crop_blocks(image, centers_px, microns_per_pixel, picked,
                                     half_um=patch_px * microns_per_pixel / 2,
                                     channels=[c for c, _ in novel], out_size=None,
                                     mask="none"):
        collected.append(crops.astype(np.float32) / intensity_scaling_factor(crops.dtype))
    stack = np.concatenate(collected, axis=0)
    return {name: (float(stack[..., k].mean()), float(stack[..., k].std()) or 1.0)
            for k, (_, name) in enumerate(novel)}


def register_novel_markers(model, stats: dict[str, tuple[float, float]]) -> list[str]:
    """Register the channels of *stats* with KRONOS2 as novel markers."""
    import pandas as pd

    if not stats:
        return []
    rows = [{"marker_name": name, **NOVEL_MARKERS[name], "mean": float(mean),
             "std": float(std), "pretraining": "no"} for name, (mean, std) in stats.items()]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "novel_markers.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        model.register_additional_markers(str(path))
    for row in rows:
        log.info("registered novel marker %s (%s/%s) mean=%.5f std=%.5f", row["marker_name"],
                 row["compartment"], row["family"], row["mean"], row["std"])
    return list(stats)


def kronos2_encoder(model, names: Sequence[str], device: str = "cuda",
                    preferred_dapi: str = PREFERRED_DAPI
                    ) -> Callable[[np.ndarray], np.ndarray]:
    """A function ``(B, C, P, P)`` float32 in [0, 1] -> ``(B, 768)`` running KRONOS2."""
    import torch

    names = list(names)

    def forward(array: np.ndarray) -> np.ndarray:
        # preprocess does the marker-aware z-score itself
        prepared = model.preprocess(array, names, preferred_dapi=preferred_dapi)
        batch = torch.as_tensor(np.asarray(prepared), dtype=torch.float32, device=device)
        with torch.inference_mode():
            return model(batch, names).float().cpu().numpy()

    return forward
