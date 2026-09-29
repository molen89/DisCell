#!/usr/bin/env python3
"""The niche label Cellina's domain adversary reads (devlog "Cellina: three
additions (motivation, 2026-09-29; author)", addition 1).

K = 10 k-means on neighbour composition ``y``, the construction of
:func:`discell.model.validate.niche_labels` (``KMeans(K, n_init=4,
random_state=seed)``, fitted on at most 100,000 rows drawn by
``default_rng(seed)``, predicted for every connected cell), with two changes
the author asked for: the fit sees **training cells only** (the run's train
tiles), and isolated cells (degree 0) get their own class ``K`` so every cell
has a label. A held-out section (``--config-from``) is labelled with the
fitted section's centres (nearest centre = ``KMeans.predict``).

k-means depends on the BLAS thread count (issue T-omp): run at
OMP_NUM_THREADS=8, as every niche-label read is. Writes
``<out>/<dataset>.npz`` (``cell_id``, ``label``, ``centres``, ``classes``) and
``<out>/<dataset>.json`` (class counts, split counts, thread count).

    python -m discell.experiments.cellina_niche_domain --dataset D
    python -m discell.experiments.cellina_niche_domain --dataset D_dual \\
        --config-from D_solo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("discell.cellina_niche_domain")

K = 10
MAX_FIT = 100_000          #: validate.niche_labels' subsample cap
OUT = Path("/home/rmolen/github/DisCell-baselines/data/niche_domain")


def assemble_like(dataset: str, config: dict):
    """``assemble`` under a run's configuration (as baseline_battery does)."""
    from discell.model.prepare import assemble

    return assemble(dataset, config["variant"], config["embeddings"],
                    tile_cells=config["tile_cells"],
                    phi_pca=config.get("phi_pca"), v_pcs=config.get("v_pcs", 12),
                    val_fraction=config["val_fraction"], seed=config["seed"],
                    label_key=config.get("label_key"))


def train_mask(data) -> np.ndarray:
    train = np.zeros(data.graph.n_cells, dtype=bool)
    for tile in data.train_tiles:
        train[tile] = True
    return train


def fit_centres(y: np.ndarray, connected: np.ndarray, train: np.ndarray,
                k: int = K, seed: int = 0) -> np.ndarray:
    """k-means centres on the connected TRAINING cells (niche_labels' recipe)."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(connected & train)
    fit_rows = rows if len(rows) <= MAX_FIT else np.sort(
        rng.choice(rows, MAX_FIT, replace=False))
    return KMeans(k, n_init=4, random_state=seed).fit(y[fit_rows]).cluster_centers_


def assign(y: np.ndarray, connected: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Nearest centre for connected cells, class ``len(centres)`` for isolated."""
    labels = np.full(len(y), len(centres), dtype=np.int64)
    rows = np.flatnonzero(connected)
    for a in range(0, len(rows), 50_000):
        chunk = rows[a:a + 50_000]
        d = ((y[chunk, None, :] - centres[None]) ** 2).sum(-1)
        labels[chunk] = d.argmin(1)
    return labels


def main(argv=None) -> int:
    from discell.experiments.baseline_battery import run_config

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--config-from", default=None,
                   help="label a held-out section with this dataset's centres")
    p.add_argument("--config-run", default="finalL_s0")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source = args.config_from or args.dataset
    config = run_config(source, args.config_run)
    data = assemble_like(args.dataset, config)
    y = np.asarray(data.graph.y, dtype=np.float64)
    connected = data.graph.degrees > 0
    train = train_mask(data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if source == args.dataset:
        centres = fit_centres(y, connected, train, K, config["seed"])
        classes = None
    else:
        blob = np.load(out / f"{source}.npz")
        centres, classes = blob["centres"], blob["classes"]
    labels = assign(y, connected, centres)
    if classes is None:                    # the fitted section's vocabulary
        classes = np.unique(labels)
    counts = {str(c): int((labels == c).sum()) for c in range(K + 1)}
    record = {
        "dataset": args.dataset, "centres_from": source,
        "config_run": args.config_run, "k": K, "seed": config["seed"],
        "isolated_class": K, "classes": [int(c) for c in classes],
        "n_cells": int(len(labels)), "n_isolated": int((~connected).sum()),
        "fit_cells": int((connected & train).sum()) if source == args.dataset
        else None,
        "class_counts": counts,
        "class_counts_train": {str(c): int(((labels == c) & train).sum())
                               for c in range(K + 1)},
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "recipe": "validate.niche_labels (KMeans k=10, n_init=4, random_state="
                  "seed, <=100k fit rows) fitted on connected TRAINING cells; "
                  "isolated cells = class 10"}
    np.savez_compressed(out / f"{args.dataset}.npz",
                        cell_id=np.arange(len(labels), dtype=np.int64),
                        label=labels, centres=centres, classes=classes)
    (out / f"{args.dataset}.json").write_text(json.dumps(record, indent=2))
    log.info("%s: %s", args.dataset, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
