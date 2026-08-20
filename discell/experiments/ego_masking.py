#!/usr/bin/env python3
"""What is a per-cell image embedding actually reading?

The dataloader hands a model an image vector per cell alongside its
neighbourhood graph. The vector is cut from a patch far wider than the cell, so
before trusting it we need to know whether it describes *the cell* or *the
tissue around it* -- and if the latter, whether that is anything more than the
homophily the graph already carries.

Three arms over identical patches, plus a resolution control:

===============  ===========================================================
``unmasked``     the whole patch
``ego``          a fixed-radius disk at the anchor is zeroed
``ego_only``     everything outside the cell's own polygon is zeroed
``ego_only_hi``  the same, cut at the slide's native resolution
===============  ===========================================================

The disk is *fixed* and *round*: round so there is no orientation artefact,
fixed so the hole is identical for every cell and therefore says nothing about
the one removed. Masking by the cell's own polygon would do the opposite -- the
silhouette it leaves is the most type-informative feature the cell has, which is
why that is arm three rather than arm two. Cells too large for the disk are
dropped from *every* arm, so the hole always covers its cell completely.

``ego_only_hi`` exists because at 0.5 um/px a median cell spans ~17 px, barely
one ViT patch token; without it a low ``ego_only`` score cannot be told apart
from simply not having given the model enough pixels.

Each arm's embedding is scored the same way: multinomial logistic regression
predicting cell type, macro one-vs-rest AUC, on a **spatial** split. The split
has to be spatial -- neighbouring cells' patches overlap almost completely, and
their labels are each other's features, so a random split leaks on both counts.

The number every arm is measured against is ``AUC(type | neighbour composition)``:
cell type predicted from the *labels* of the neighbours alone, no image at all.

============================  =============================================
``AUC(unmasked) - AUC(ego)``  what masking actually buys
``AUC(ego_only)``             what the cell carries on its own
``AUC(ego)`` vs the baseline  whether the masked patch knows anything beyond
                              the neighbourhood being type-homogeneous
============================  =============================================

If ``AUC(ego)`` sits at the baseline, the masked patch is reading homophily and
nothing else -- expected, legitimate, and already handled by conditioning on the
neighbour types. If it sits clearly above, the cell leaves an imprint on its
surroundings that survives its own removal.

Usage::

    python -m discell.experiments.ego_masking --dataset <id>
    python -m discell.experiments.ego_masking --cells 0        # whole slide
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from discell import paths
from discell.preprocess.crops import DEFAULT_MASK_RADIUS_UM

log = logging.getLogger("discell.experiments.ego_masking")

#: KRONOS's own reference patch size, from its model card.
PATCH_PX = 256

#: Resolution the crops are cut at. The slide is 0.2125 um/px and KRONOS's
#: worked example runs at 0.37, so the native scale is already finer than
#: anything the model saw; 0.5 backs further off and buys a 128 um field, which
#: is what keeps the mask down to 12% of the patch.
TARGET_MPP = 0.5

#: Cells per type in the subsample. The whole slide is 407k and every arm has to
#: be embedded separately; this is ample for an AUC comparison and keeps a full
#: sweep inside an hour. ``--cells 0`` runs the lot.
DEFAULT_CELLS = 60_000

#: Side of the blocks the spatial split is cut into.
DEFAULT_TILE_UM = 1000.0


@dataclass(frozen=True)
class Arm:
    name: str
    model: str
    mask: str
    target_mpp: float | None       # None = the slide's native resolution
    note: str

    @property
    def stem(self) -> str:
        return f"egomask_{self.name}"


ARMS: tuple[Arm, ...] = (
    Arm("unmasked_v1", "v1", "none", TARGET_MPP, "whole patch"),
    Arm("ego_v1", "v1", "ego", TARGET_MPP, "fixed disk zeroed"),
    Arm("ego_only_v1", "v1", "ego_only", TARGET_MPP, "outside the polygon zeroed"),
    Arm("ego_only_hi_v1", "v1", "ego_only", None, "as above, native resolution"),
    Arm("unmasked_v2", "v2", "none", TARGET_MPP, "whole patch"),
    Arm("ego_v2", "v2", "ego", TARGET_MPP, "fixed disk zeroed"),
    Arm("ego_only_v2", "v2", "ego_only", TARGET_MPP, "outside the polygon zeroed"),
)


# --------------------------------------------------------------------------
# cell selection and the split
# --------------------------------------------------------------------------


def selectable(adata, mask_radius_um: float) -> np.ndarray:
    """Cells the disk can fully cover, as a boolean over the cell axis.

    A cell wider than the hole would show through it, which is the one thing the
    mask must not allow. They are dropped from every arm, not just the masked
    one, so all four are scored over the same cells.
    """
    from discell.preprocess.crops import covering_radius_um

    radius = covering_radius_um(adata)
    keep = radius <= mask_radius_um
    if not keep.all():
        log.info("Excluding %d of %d cells (%.4f%%) larger than the %g um disk "
                 "(largest %.1f um)", int((~keep).sum()), len(keep),
                 100 * (~keep).mean(), mask_radius_um, radius.max())
    return keep


def stratified(labels: np.ndarray, total: int, rng) -> np.ndarray:
    """Up to ``total`` cells, spread evenly over the label space.

    Even rather than proportional: the rare types are the ones an AUC is most
    uncertain about, and the abundant ones have samples to spare.
    """
    types = np.unique(labels)
    per_type = max(1, total // len(types))
    picked = []
    for t in types:
        rows = np.flatnonzero(labels == t)
        take = min(per_type, len(rows))
        picked.append(rng.choice(rows, take, replace=False))
    return np.sort(np.concatenate(picked))


def spatial_split(positions_um: np.ndarray, tile_um: float, margin_um: float,
                  test_fraction: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """``(train, test)`` boolean masks over contiguous tissue blocks.

    A random per-cell split would be meaningless here twice over: adjacent
    cells' patches overlap almost entirely, so the same pixels would sit on both
    sides; and the baseline predicts a cell's type from its neighbours' labels,
    which a random split puts straight into the training set.

    Cells within *margin_um* of a tile edge are dropped rather than assigned, so
    no patch straddles the boundary.
    """
    tiles = np.floor(positions_um / tile_um).astype(np.int64)
    keys = np.unique(tiles, axis=0)
    is_test = rng.random(len(keys)) < test_fraction
    lookup = {tuple(k): bool(v) for k, v in zip(keys, is_test)}
    test_tile = np.array([lookup[tuple(t)] for t in tiles])

    offset = positions_um - tiles * tile_um
    interior = ((offset > margin_um) & (offset < tile_um - margin_um)).all(axis=1)
    log.info("Spatial split: %d tiles of %g um, %d test; %d of %d cells dropped "
             "within %g um of a tile edge", len(keys), tile_um, int(is_test.sum()),
             int((~interior).sum()), len(tiles), margin_um)
    return (~test_tile) & interior, test_tile & interior


# --------------------------------------------------------------------------
# what the mask actually does, drawn
# --------------------------------------------------------------------------


def plot_mask_examples(adata, image_path, out_path: Path, mask_radius_um: float,
                       label_key: str = "cell_group", dpi: int = 130) -> Path:
    """Each arm over the smallest, median and largest cells, outline drawn on.

    The point of the figure is the bottom two rows: the largest cell the disk
    still covers, and the largest cell on the slide, which it does not. Seeing
    the outline against the hole is the only honest check that the mask is doing
    what the argument for it assumes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    from discell.preprocess.crops import covering_radius_um, iter_crop_blocks

    radius = covering_radius_um(adata)
    order = np.argsort(radius)
    fits = np.flatnonzero(radius <= mask_radius_um)
    rows = [
        (int(order[0]), "smallest"),
        (int(order[len(order) // 2]), "median"),
        (int(fits[np.argmax(radius[fits])]), f"largest that fits the {mask_radius_um:g} um disk"),
        (int(order[-1]), "largest on the slide -- excluded"),
    ]

    mpp = float(adata.uns["microns_per_pixel"])
    half = PATCH_PX * TARGET_MPP / 2
    size = int(round(2 * half / mpp))
    scale = PATCH_PX / size
    centres = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    modes = ("none", "ego", "ego_only")

    crops: dict = {}
    for mode in modes:
        crops[mode] = {}
        for idx, block in iter_crop_blocks(
                adata, image_path, [c for c, _ in rows], half_um=half,
                channels=[0, 1, 2, 3], out_size=PATCH_PX, mask=mode,
                mask_radius_um=mask_radius_um):
            for k, crop in zip(idx, block):
                crops[mode][int(k)] = crop

    fig, axes = plt.subplots(len(rows), len(modes),
                             figsize=(3.0 * len(modes), 3.15 * len(rows)))
    for r, (cell, caption) in enumerate(rows):
        xy = np.asarray(adata.uns["polygons"][cell].exterior.coords)
        x0 = round(centres[cell][0] - size / 2)
        y0 = round(centres[cell][1] - size / 2)
        local = (xy - (x0, y0)) * scale
        for c, mode in enumerate(modes):
            ax = axes[r, c]
            window = crops[mode][cell].astype(np.float32)
            # boundary / DAPI / interior-RNA, the composite used elsewhere
            rgb = np.stack([window[..., 1], window[..., 0], window[..., 2]], -1)
            lit = rgb[rgb > 0]
            lo, hi = np.percentile(lit, (1, 99.5)) if lit.size else (0.0, 1.0)
            ax.imshow(np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1))
            ax.plot(local[:, 0], local[:, 1], color="#ffd400", lw=1.1)
            if mode == "ego":
                ax.add_patch(Circle(((PATCH_PX - 1) / 2, (PATCH_PX - 1) / 2),
                                    mask_radius_um / TARGET_MPP, fill=False,
                                    color="#00e5ff", lw=0.9, ls="--"))
            ax.set_xlim(0, PATCH_PX); ax.set_ylim(PATCH_PX, 0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(mode, fontsize=10)
        axes[r, 0].set_ylabel(
            f"{caption}\nr={radius[cell]:.1f} um, {adata.obs['area_um2'].iloc[cell]:.0f} um2\n"
            f"{str(adata.obs[label_key].iloc[cell])[:24]}",
            fontsize=7)
    fig.suptitle(
        f"{PATCH_PX} px at {TARGET_MPP} um/px = {2 * half:g} um field   |   "
        f"ego disk {mask_radius_um:g} um radius (dashed), cell outline in yellow",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


# --------------------------------------------------------------------------
# embedding and scoring
# --------------------------------------------------------------------------


def embed_arm(arm: Arm, adata, image_path, dataset, cells: np.ndarray,
              args) -> Path:
    """Embed one arm, or return the file if it is already there."""
    from discell.preprocess.kronos import embed_sample, write_embeddings

    target = dataset.embeddings_dir / f"{arm.stem}.pt"
    if target.exists() and not args.force:
        log.info("[%s] reusing %s", arm.name, target.name)
        return target

    half_um = (PATCH_PX * arm.target_mpp / 2) if arm.target_mpp else None
    started = time.time()
    embeddings, cell_ids, meta = embed_sample(
        adata, image_path, model=arm.model, channels=[0, 1, 2, 3],
        cells=cells, patch_px=PATCH_PX, half_um=half_um,
        mask=arm.mask, mask_radius_um=args.mask_radius_um,
        batch_size=args.batch_size, block_px=args.block_px,
        device=args.device, cache_dir=args.cache_dir, hf_token=args.hf_token,
    )
    write_embeddings(target, embeddings, cell_ids, meta={
        "dataset": dataset.dataset_id, "experiment": "ego_masking",
        "arm": arm.name, **meta,
    })
    log.info("[%s] %s in %.1f min", arm.name, embeddings.shape,
             (time.time() - started) / 60)
    return target


def score(features: np.ndarray, labels: np.ndarray, train: np.ndarray,
          test: np.ndarray, seed: int, max_train: int) -> dict:
    """Macro one-vs-rest AUC and accuracy from a multinomial logistic fit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(train)
    if len(rows) > max_train:
        rows = np.sort(rng.choice(rows, max_train, replace=False))
    test_rows = np.flatnonzero(test)

    scaler = StandardScaler().fit(features[rows])
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(scaler.transform(features[rows]), labels[rows])

    probabilities = model.predict_proba(scaler.transform(features[test_rows]))
    truth = labels[test_rows]
    # Restrict to classes the fit actually saw; a class absent from train has no
    # column and roc_auc_score would refuse the whole matrix.
    present = np.isin(truth, model.classes_)
    return {
        "auc": float(roc_auc_score(truth[present], probabilities[present],
                                   multi_class="ovr", average="macro",
                                   labels=model.classes_)),
        "accuracy": float(accuracy_score(truth[present],
                                         model.classes_[probabilities[present].argmax(1)])),
        "n_train": int(len(rows)),
        "n_test": int(present.sum()),
        "n_classes": int(len(model.classes_)),
    }


def run(args: argparse.Namespace) -> int:
    from discell.data.embeddings import load_embeddings
    from discell.data.loader import CellGraphDataset
    from discell.tiff import find_tissue_image

    dataset = paths.dataset(args.dataset).ensure()
    rng = np.random.default_rng(args.seed)

    # One open, shared by the crops and the baseline: the loader holds the
    # bundle it read, and its label space is the one a model would train on --
    # NaN already folded, no case-duplicate class.
    log.info("Opening %s / %s", dataset.dataset_id, args.variant)
    data = CellGraphDataset.from_dataset(dataset, args.variant, graph=args.graph,
                                         label_key=args.label_key)
    adata = data.adata
    image_path = find_tissue_image(adata.uns["xenium_dir"])
    if image_path is None:
        raise SystemExit(f"no morphology image for {dataset.dataset_id}")

    labels_all = data.type_names[data.type_index]

    figures = dataset.root / "experiments"
    plot_mask_examples(adata, image_path, figures / "ego_masking_examples.png",
                       args.mask_radius_um, label_key=data.label_key)

    fits = selectable(adata, args.mask_radius_um)
    pool = np.flatnonzero(fits)
    cells = (pool if args.cells == 0 else
             pool[stratified(labels_all[pool], args.cells, rng)])
    log.info("Scoring %d cells over %d types", len(cells), len(np.unique(labels_all[cells])))

    positions = np.asarray(adata.obsm["spatial_um"], dtype=np.float64)[cells]
    field_um = PATCH_PX * TARGET_MPP
    train, test = spatial_split(positions, args.tile_um, field_um / 2,
                                args.test_fraction, rng)
    labels = labels_all[cells]

    # -- the baseline: type from the neighbours' types, no image at all -------
    niche = data.constant.niche.numpy()[cells]
    results = {"neighbour_composition": {
        **score(niche, labels, train, test, args.seed, args.max_train),
        "note": "no image -- cell type from the neighbours' labels alone",
    }}
    log.info("baseline AUC(type | neighbour composition) = %.4f",
             results["neighbour_composition"]["auc"])

    # -- the arms -------------------------------------------------------------
    for arm in ARMS:
        if args.arms and arm.name not in args.arms:
            continue
        path = embed_arm(arm, adata, image_path, dataset, cells, args)
        vectors, found = load_embeddings(path, np.asarray(adata.obs_names, dtype=str)[cells],
                                         dataset=dataset.dataset_id)
        if not found.all():
            log.warning("[%s] %d cells missing from the file", arm.name, int((~found).sum()))
        results[arm.name] = {**score(vectors, labels, train & found, test & found,
                                     args.seed, args.max_train),
                             **{k: v for k, v in asdict(arm).items() if k != "name"},
                             "dim": int(vectors.shape[1])}
        log.info("[%s] AUC %.4f", arm.name, results[arm.name]["auc"])

    report(results, dataset, args)
    return 0


def report(results: dict, dataset, args) -> None:
    import pandas as pd

    frame = pd.DataFrame(results).T
    out = dataset.root / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "ego_masking.csv")
    (out / "ego_masking.json").write_text(json.dumps(
        {"results": results, "args": {k: str(v) for k, v in vars(args).items()},
         "patch_px": PATCH_PX, "target_mpp": TARGET_MPP}, indent=2))

    base = results.get("neighbour_composition", {}).get("auc")
    print()
    print("=" * 78)
    print(f"EGO MASKING  {dataset.dataset_id}")
    print("=" * 78)
    print(f"  {'arm':<18} {'AUC':>7} {'vs baseline':>12} {'acc':>7} {'dim':>5}  note")
    for name, row in results.items():
        delta = "" if base is None or name == "neighbour_composition" else f"{row['auc'] - base:+.4f}"
        print(f"  {name:<18} {row['auc']:>7.4f} {delta:>12} {row['accuracy']:>7.3f} "
              f"{row.get('dim', '-'):>5}  {row.get('note', '')}")
    print()
    for model in ("v1", "v2"):
        full, ego = results.get(f"unmasked_{model}"), results.get(f"ego_{model}")
        if full and ego:
            print(f"  {model}: masking costs {full['auc'] - ego['auc']:+.4f} AUC; "
                  f"masked patch sits {ego['auc'] - base:+.4f} above the "
                  f"neighbour-composition baseline")
    print(f"\n  wrote {out / 'ego_masking.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default=paths.DEFAULT_VARIANT)
    parser.add_argument("--graph", default=None, help="for the baseline; default the loader's")
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS,
                        help="stratified subsample size; 0 for every cell")
    parser.add_argument("--arms", nargs="*", default=None,
                        help=f"subset of {[a.name for a in ARMS]}")
    parser.add_argument("--mask-radius-um", type=float, default=DEFAULT_MASK_RADIUS_UM)
    parser.add_argument("--tile-um", type=float, default=DEFAULT_TILE_UM)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--max-train", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-px", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=str(paths.MODELS))
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--force", action="store_true", help="re-embed every arm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from discell.data.loader import DEFAULT_GRAPH

    args = build_parser().parse_args(argv)
    args.graph = args.graph or DEFAULT_GRAPH
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
