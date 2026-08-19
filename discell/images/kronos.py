#!/usr/bin/env python3
"""Per-cell image embeddings from KRONOS (MahmoodLab).

KRONOS is a multiplex-imaging foundation model: it projects **each channel
separately** and adds a per-marker embedding, so it takes an arbitrary number of
channels as long as each one is identified by a ``marker_id``. That makes it a
natural fit for Xenium's 4-channel morphology stack, with one important caveat
about marker identity -- see :data:`XENIUM_MARKERS` below.

Contract, read from the source rather than the README (whose minimal example
omits ``marker_ids`` even though the model asserts on it)::

    model, precision, embedding_dim = create_model_from_pretrained(
        checkpoint_path="hf_hub:MahmoodLab/kronos", cache_dir=..., hf_auth_token=...)
    batch = (batch - mean[None, :, None, None]) / std[None, :, None, None]
    patch_emb, marker_emb, token_emb = model(batch, marker_ids=marker_ids)

* ``batch`` is ``(B, n_markers, P, P)``; P must be a multiple of 16
* ``marker_ids`` is a list of per-sample LongTensors of length ``n_markers``
* per-marker ``mean``/``std`` come from KRONOS's ``marker_metadata.csv``

**The weights are gated.** Request access at
https://huggingface.co/MahmoodLab/KRONOS (institutional email required), then
pass ``--hf-token`` or set ``HF_TOKEN``. Without it this module still builds
crops, marker ids and batches, and can run an architecture-only model so the
plumbing is testable.

Output is written in the format :func:`discell.data.loader.load_embeddings`
expects, so it drops straight into the dataloader.

Usage::

    python -m discell.images.kronos --sample <dir> --out embeddings.npz --limit 2000
    python -m discell.images.kronos --sample <dir> --out emb.npz --patch-px 256 --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("discell.images.kronos")

#: KRONOS ViT patch size; the input edge must be a multiple of this.
KRONOS_PATCH_MULTIPLE = 16

#: Default input edge in pixels, matching the KRONOS tutorials. At Xenium's
#: 0.2125 um/px this is a 54.4 um field: a cell (p99 extent 13.1 um) sits
#: centrally with roughly one cell-width of surrounding tissue on each side, and
#: no resampling is needed.
DEFAULT_PATCH_PX = 256


def intensity_scaling_factor(dtype) -> float:
    """Divisor that puts raw pixels on the scale KRONOS was trained on.

    Both KRONOS releases expect intensities in roughly [0, 1] -- the shipped
    per-marker means are ~0.005-0.08 -- and their own loader divides by the
    dtype maximum (uint8 -> 255, unsigned -> 65535, float -> 400) before the
    marker z-score. Xenium morphology is uint16, so feeding raw values makes the
    z-score four orders of magnitude too large and puts every patch far outside
    the pretraining distribution.
    """
    import numpy as _np

    if dtype == _np.uint8:
        return 255.0
    if _np.issubdtype(dtype, _np.unsignedinteger):
        return 65535.0
    if _np.issubdtype(dtype, _np.floating):
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


def fetch_marker_metadata(
    repo: str = "MahmoodLab/kronos",
    cache_dir: str | Path = "./model_assets",
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


def load_kronos(
    checkpoint: str = "hf_hub:MahmoodLab/kronos",
    cache_dir: str | Path = "./model_assets",
    hf_token: str | None = None,
    pretrained: bool = True,
):
    """Load KRONOS. Returns ``(model, precision, embedding_dim)``.

    With ``pretrained=False`` the architecture is built with random weights,
    which is useful for checking shapes without the gated download.
    """
    from kronos import create_model, create_model_from_pretrained

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cfg = {"model_type": "vits16", "token_overlap": False}
    if not pretrained:
        # create_model builds from config only; create_model_from_pretrained
        # would try to torch.load(None) and fail.
        log.warning("Building KRONOS with RANDOM weights -- embeddings are meaningless")
        return create_model(checkpoint_path=None, cfg_path=None, cfg=cfg)
    return create_model_from_pretrained(
        checkpoint_path=checkpoint, cfg_path=None, hf_auth_token=token,
        cache_dir=str(cache_dir), cfg=cfg,
    )


def embed_cells(
    adata,
    image_path: str | Path,
    markers: Sequence[MarkerSpec],
    model,
    precision,
    cells: Sequence[int] | None = None,
    patch_px: int = DEFAULT_PATCH_PX,
    batch_size: int = 32,
    device: str = "cuda",
    half_um: float | None = None,
    anchor: str = "centroid",
    block_px: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed cells in batches. Returns ``(embeddings, cell_ids)``."""
    import torch

    from discell.data.crops import DEFAULT_BLOCK_PX, iter_crop_blocks

    if patch_px % KRONOS_PATCH_MULTIPLE:
        raise ValueError(f"patch_px must be a multiple of {KRONOS_PATCH_MULTIPLE}")

    channels = [m.channel for m in markers]
    mpp = float(adata.uns["microns_per_pixel"])
    # Either take patch_px pixels natively, or a fixed micron field resampled.
    if half_um is None:
        half = (patch_px * mpp) / 2
        out_size = None
    else:
        half, out_size = half_um, patch_px
    log.info("Crop: %d px, %.1f um field (%.4f um/px)%s",
             patch_px, 2 * half, mpp, "" if out_size is None else " [resampled]")

    ids = [m.marker_id if m.resolved else i for i, m in enumerate(markers)]
    if any(not m.resolved for m in markers):
        log.warning("Using positional fallback ids for unresolved markers: %s", ids)
    mean = torch.tensor([m.mean for m in markers], dtype=precision, device=device)
    std = torch.tensor([m.std for m in markers], dtype=precision, device=device)

    cells = list(range(adata.n_obs)) if cells is None else list(cells)
    model.eval().to(device)
    cell_names = np.asarray(adata.obs_names, dtype=str)

    outputs: list[np.ndarray] = []
    kept: list[str] = []
    started, done = time.time(), 0
    # Crops come back block by block: each image region is decoded once and every
    # window in it is sliced from RAM, which is ~700x the per-cell path.
    for indices, crops in iter_crop_blocks(
        adata, image_path, cells, half_um=half, channels=channels,
        out_size=out_size, anchor=anchor, block_px=block_px,
    ):
        for start in range(0, len(indices), batch_size):
            piece = crops[start : start + batch_size]
            # (B, H, W, C) -> (B, C, H, W)
            scaling = intensity_scaling_factor(piece.dtype)
            array = np.ascontiguousarray(piece.transpose(0, 3, 1, 2),
                                         dtype=np.float32) / scaling
            batch = torch.from_numpy(array).to(device=device, dtype=precision)
            batch = (batch - mean[None, :, None, None]) / std[None, :, None, None]
            marker_ids = [torch.tensor(ids, device=device) for _ in range(batch.shape[0])]

            with torch.no_grad():
                patch_emb, _marker_emb, _token_emb = model(batch, marker_ids=marker_ids)
            outputs.append(patch_emb.float().cpu().numpy())
            kept.extend(cell_names[indices[start : start + batch_size]])

        done += len(indices)
        elapsed = max(time.time() - started, 1e-9)
        log.info("  %d/%d cells (%.0f cells/s, eta %.1f min)",
                 done, len(cells), done / elapsed,
                 (len(cells) - done) / max(done / elapsed, 1e-9) / 60)

    embeddings = np.concatenate(outputs, axis=0)
    log.info("Embedded %d cells -> %s in %.1f min (%.0f cells/s)",
             len(kept), embeddings.shape, (time.time() - started) / 60,
             len(kept) / max(time.time() - started, 1e-9))
    return embeddings, np.asarray(kept, dtype=str)


def write_embeddings(path: str | Path, embeddings: np.ndarray, cell_ids: np.ndarray,
                     meta: dict | None = None) -> Path:
    """Write in the layout :func:`discell.data.loader.load_embeddings` reads.

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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="v1", choices=("v1", "v2", "kronos", "kronos2"),
                        help="v1 = KRONOS (384-d, marker ids); "
                             "v2 = KRONOS2 (768-d, marker names)")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", default="kronos_embeddings.pt",
                        help="output path; .pt stores torch tensors, .npz numpy")
    parser.add_argument("--channels", default="0,1,2,3",
                        help="image channels to embed; '0' is a DAPI-only baseline")
    parser.add_argument("--patch-px", type=int, default=DEFAULT_PATCH_PX,
                        help="input edge in pixels; must be a multiple of 16")
    parser.add_argument("--half-um", type=float, default=None,
                        help="fix the field in microns and resample to --patch-px "
                             "instead of cropping natively")
    parser.add_argument("--marker-metadata", default=None,
                        help="KRONOS marker_metadata.csv (ships with the gated release)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-px", type=int, default=4096,
                        help="image block decoded at once when cropping in bulk")
    parser.add_argument("--limit", type=int, default=None, help="embed only N cells")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="limit to the N cells nearest the tissue centre; "
                             "default is every cell")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--cache-dir", default="./model_assets")
    parser.add_argument("--drop-unknown-markers", action="store_true",
                        help="v2: skip channels that are neither in the vocabulary "
                             "nor registrable, instead of failing")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="architecture only, random weights: checks plumbing")
    parser.add_argument("--dry-run", action="store_true",
                        help="report crops, markers and shapes without running the model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _run_v2(args, adata, sample_dir, image_path) -> int:
    """KRONOS2 path: markers by name, normalisation inside the model."""
    import numpy as _np

    from discell.images.kronos2 import (
        NOVEL_MARKERS, PREFERRED_DAPI, XENIUM_MARKER_NAMES,
        describe_markers as describe_v2, embed_cells_v2, load_kronos2,
        measure_channel_stats, register_novel_markers, resolve_marker_names,
    )

    channels = [int(c) for c in args.channels.split(",")]
    names = [XENIUM_MARKER_NAMES[c] for c in channels]
    resolved = resolve_marker_names(names, args.cache_dir, args.hf_token)

    print(f"\nModel : KRONOS2 (marker-aware DINOv2 ViT-B/16)")
    print(f"Image : {image_path.name}")
    print(describe_v2(names, resolved))

    # v2's marker embedder raises on an unknown name -- the documented default
    # statistics cover preprocess() only. Dropping the channel is the honest
    # option; substituting a vocabulary name would assert the channel is a
    # protein it is not.
    unknown = [(c, n) for c, n in zip(channels, names) if resolved.get(n) is False]
    registrable = [n for _, n in unknown if n in NOVEL_MARKERS]
    unusable = [(c, n) for c, n in unknown if n not in NOVEL_MARKERS]
    if unusable and args.drop_unknown_markers:
        keep = [(c, n) for c, n in zip(channels, names)
                if (c, n) not in unusable]
        channels, names = [c for c, _ in keep], [n for _, n in keep]
        print("\n  NOTE: dropped " + ", ".join(f"channel {c} ({n})" for c, n in unusable))
    elif unusable:
        raise SystemExit(
            f"markers not in the vocabulary and not registrable: "
            f"{[n for _, n in unusable]}; pass --drop-unknown-markers to skip them")
    if registrable:
        print(f"\n  {', '.join(registrable)} not in the vocabulary -- registering as "
              f"novel marker(s) via the model's own hook, keeping all "
              f"{len(names)} channels.")

    rng = _np.random.default_rng(args.seed)
    cells = list(range(adata.n_obs))
    if args.limit:
        cells = sorted(rng.choice(len(cells), min(args.limit, len(cells)), replace=False))

    mpp = float(adata.uns["microns_per_pixel"])
    print(f"\nCells : {len(cells):,} of {adata.n_obs:,}")
    print(f"Crop  : {args.patch_px}px, {args.patch_px * mpp:.1f} um field, "
          f"{len(names)} channels")

    if args.dry_run:
        print("\ndry run: model not loaded")
        return 0

    import torch as _torch

    device = args.device if _torch.cuda.is_available() else "cpu"
    model, dim = load_kronos2(args.cache_dir, args.hf_token, device)
    print(f"\nKRONOS2 loaded: embedding_dim={dim}")

    registered = []
    if registrable:
        stats = measure_channel_stats(adata, image_path, channels, names,
                                      half_um=(args.patch_px * mpp) / 2)
        registered = register_novel_markers(model, names, stats)
        for name in registered:
            mean, std = stats[name]
            print(f"  registered {name}: mean={mean:.5f} std={std:.5f} "
                  f"(measured on this sample's crops)")

    embeddings, cell_ids = embed_cells_v2(
        adata, image_path, names, channels, model, cells=cells,
        patch_px=args.patch_px, batch_size=args.batch_size, device=device,
        block_px=args.block_px, preferred_dapi=PREFERRED_DAPI,
    )
    write_embeddings(args.out, embeddings, cell_ids, meta={
        "model": "kronos2",
        "patch_px": args.patch_px,
        "microns_per_pixel": mpp,
        "channels": channels,
        "marker_names": names,
        "in_vocabulary": [bool(resolved.get(n)) for n in names],
        "registered_novel": registered,
        "pretrained": True,
    })
    print(f"\nWrote {args.out}: {embeddings.shape}")
    print(f"Use with: CellGraphDataset(..., embeddings='{args.out}')")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from discell.plotting.cell_graph import find_tissue_image, load_any_sample

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )

    # graphs are irrelevant to embedding and cost ~4 min on a full slide
    adata, sample_dir = load_any_sample(args.sample, 30.0, args.max_cells,
                                        build_graphs=False)
    image_path = find_tissue_image(sample_dir)
    if image_path is None:
        raise SystemExit(f"no image under {sample_dir}")

    if args.model in ("v2", "kronos2"):
        return _run_v2(args, adata, sample_dir, image_path)

    wanted = [int(c) for c in args.channels.split(",")]
    specs = [m for m in XENIUM_MARKERS if m.channel in wanted]
    if len(specs) != len(wanted):
        raise SystemExit(f"unknown channel in {wanted}; known: "
                         f"{[m.channel for m in XENIUM_MARKERS]}")
    metadata_path = args.marker_metadata
    if metadata_path is None and not args.no_pretrained:
        # Ships in the KRONOS repo, so fetch it with the same credentials.
        fetched = fetch_marker_metadata(cache_dir=args.cache_dir, hf_token=args.hf_token)
        metadata_path = str(fetched) if fetched else None
    markers = resolve_markers(specs, load_marker_metadata(metadata_path))

    print(f"\nImage: {image_path.name}")
    print(describe_markers(markers))
    if any(not m.exact for m in markers):
        print("\n  NOTE: channels flagged APPROXIMATE are antibody cocktails or a\n"
              "  non-protein (rRNA) channel. KRONOS assigns one marker identity per\n"
              "  channel, so no faithful id exists for them.")
    if not any(m.resolved for m in markers):
        print("\n  NOTE: no marker_metadata.csv given, so ids fall back to channel\n"
              "  position and normalisation is a no-op. Embeddings will not match\n"
              "  a canonical KRONOS run.")

    rng = np.random.default_rng(args.seed)
    cells = list(range(adata.n_obs))
    if args.limit:
        cells = sorted(rng.choice(len(cells), min(args.limit, len(cells)), replace=False))

    mpp = float(adata.uns["microns_per_pixel"])
    field = args.patch_px * mpp if args.half_um is None else 2 * args.half_um
    print(f"\nCells: {len(cells):,} of {adata.n_obs:,}")
    print(f"Crop : {args.patch_px}px, {field:.1f} um field, {len(markers)} channels")

    if args.dry_run:
        from discell.data.crops import DEFAULT_BLOCK_PX, iter_crop_blocks

        crop = crop_cell(adata, image_path, cells[0],
                         half_um=(args.patch_px * mpp) / 2 if args.half_um is None
                         else args.half_um,
                         channels=[m.channel for m in markers],
                         out_size=None if args.half_um is None else args.patch_px)
        print(f"\nexample crop: {crop!r}")
        print(f"model input would be: ({args.batch_size}, {len(markers)}, "
              f"{crop.image.shape[0]}, {crop.image.shape[1]})")
        return 0

    model, precision, embedding_dim = load_kronos(
        cache_dir=args.cache_dir, hf_token=args.hf_token,
        pretrained=not args.no_pretrained,
    )
    print(f"\nKRONOS loaded: precision={precision}, embedding_dim={embedding_dim}")

    device = args.device if args.device != "cuda" or __import__("torch").cuda.is_available() else "cpu"
    embeddings, cell_ids = embed_cells(
        adata, image_path, markers, model, precision, cells=cells,
        patch_px=args.patch_px, batch_size=args.batch_size, device=device,
        half_um=args.half_um, block_px=args.block_px,
    )
    write_embeddings(args.out, embeddings, cell_ids, meta={
        "patch_px": args.patch_px,
        "microns_per_pixel": mpp,
        "channels": [m.channel for m in markers],
        "marker_names": [m.name for m in markers],
        "marker_ids": [m.marker_id for m in markers],
        "pretrained": not args.no_pretrained,
    })
    print(f"\nWrote {args.out}: {embeddings.shape}")
    print(f"Use with: CellGraphDataset(..., embeddings='{args.out}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
