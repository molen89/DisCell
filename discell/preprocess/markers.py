#!/usr/bin/env python3
"""Which protein each image channel is, as KRONOS needs to be told.

KRONOS adds a per-marker embedding to every channel, so a channel is only
interpreted correctly if its marker identity is right. Xenium's four channels
map onto the 177-marker vocabulary imperfectly -- see :data:`XENIUM_MARKERS` in
:mod:`discell.preprocess.kronos` -- and this module holds the lookup, the
normalisation statistics that ship with the weights, and the intensity scaling
both model versions expect.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from discell import paths

log = logging.getLogger("discell.preprocess.markers")

#: An id outside KRONOS's trained range (4-498 of num_markers=512). Marker
#: embeddings are deterministic sincos of the id, so an untrained id yields a
#: vector the model never associated with any protein -- preferable to
#: borrowing a real marker's id, which would assert the channel is a protein it
#: is not.
OUT_OF_VOCAB_MARKER_ID = 505


def intensity_scaling_factor(dtype) -> float:
    """Divisor that puts raw pixels on the scale KRONOS was trained on.

    Both KRONOS releases expect intensities in roughly [0, 1] -- the shipped
    per-marker means are ~0.005-0.08 -- and their own loader divides by the
    dtype maximum (uint8 -> 255, unsigned -> 65535, float -> 400) before the
    marker z-score. Xenium morphology is uint16, so feeding raw values makes the
    z-score four orders of magnitude too large and puts every patch far outside
    the pretraining distribution.
    """
    if dtype == np.uint8:
        return 255.0
    if np.issubdtype(dtype, np.unsignedinteger):
        return 65535.0
    if np.issubdtype(dtype, np.floating):
        return 400.0
    raise ValueError(f"no scaling factor for dtype {dtype!r}")


@dataclass(frozen=True)
class MarkerSpec:
    """One imaging channel, as KRONOS needs to see it."""

    channel: int
    name: str
    marker_id: int | None = None
    mean: float = 0.0
    std: float = 1.0
    exact: bool = True   # False when the channel is not a clean single marker

    @property
    def resolved(self) -> bool:
        return self.marker_id is not None


def fetch_marker_metadata(
    repo: str = "MahmoodLab/kronos",
    cache_dir: str | Path = paths.MODELS,
    hf_token: str | None = None,
) -> Path | None:
    """Download ``marker_metadata.csv`` from the KRONOS repo.

    It ships alongside the weights, so once access is approved there is no
    separate file to chase. Returns None if the repo is still gated.
    """
    from huggingface_hub import hf_hub_download

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        return Path(hf_hub_download(repo, filename="marker_metadata.csv",
                                    cache_dir=str(cache_dir), token=token))
    except Exception as exc:
        log.warning("Could not fetch marker_metadata.csv (%s): %s",
                    type(exc).__name__, str(exc).splitlines()[0][:120])
        return None


def load_marker_metadata(path: str | Path | None) -> pd.DataFrame | None:
    """Read KRONOS's ``marker_metadata.csv``.

    Expected columns: ``marker_name, marker_id, marker_mean, marker_std``. The
    file ships with the gated KRONOS release, so it is optional here.
    """
    if path is None:
        return None
    frame = pd.read_csv(path)
    needed = {"marker_name", "marker_id"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"{path} must contain {sorted(needed)}; found {list(frame.columns)}")
    frame["marker_name"] = frame["marker_name"].astype(str).str.upper()
    return frame.set_index("marker_name")


def resolve_markers(
    specs: Sequence[MarkerSpec], metadata: pd.DataFrame | None
) -> list[MarkerSpec]:
    """Fill marker_id/mean/std from the metadata table, warning on misses."""
    out: list[MarkerSpec] = []
    for spec in specs:
        if metadata is None:
            out.append(spec)
            continue
        key = spec.name.upper()
        if key not in metadata.index:
            log.warning(
                "marker %r (channel %d) is not in KRONOS's vocabulary -- using the "
                "out-of-vocabulary id %d rather than borrowing another protein's",
                spec.name, spec.channel, OUT_OF_VOCAB_MARKER_ID,
            )
            out.append(MarkerSpec(channel=spec.channel, name=spec.name,
                                  marker_id=OUT_OF_VOCAB_MARKER_ID,
                                  mean=0.0, std=1.0, exact=False))
            continue
        row = metadata.loc[key]
        out.append(MarkerSpec(
            channel=spec.channel, name=spec.name,
            marker_id=int(row["marker_id"]),
            mean=float(row.get("marker_mean", 0.0)),
            std=float(row.get("marker_std", 1.0)) or 1.0,
            exact=spec.exact,
        ))
    return out


def describe_markers(markers: Sequence[MarkerSpec]) -> str:
    lines = [f"{'ch':>3} {'marker':<12} {'id':>5} {'mean':>10} {'std':>10}  note"]
    for m in markers:
        note = "" if m.exact else "APPROXIMATE (cocktail or non-protein channel)"
        ident = str(m.marker_id) if m.resolved else "-"
        lines.append(f"{m.channel:>3} {m.name:<12} {ident:>5} {m.mean:>10.3f} "
                     f"{m.std:>10.3f}  {note}")
    return "\n".join(lines)
