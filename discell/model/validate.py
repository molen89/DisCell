#!/usr/bin/env python3
"""Latent-validation analyses (doc 08): the allegiance claim, post hoc.

Every analysis probes a target with a *known* allegiance from BOTH latents
and reports the asymmetry against reference rows: *floor* (within-type target
permutation), *l-baseline* (the probe from ``log l`` alone), and, for
expression-derived targets, the 50-PC *ceiling*. Everything is fitted within
type with spatial-block CV over the training tiles -- random CV is invalid
for spatial targets, autocorrelation leaks the answer through neighbours.

Analyses: per-dimension Moran's I (section 3), niche invariance (section 4),
distance-to-landmark regression (section 2), and the allegiance matrix
(section 5). Outputs land under ``runs/<run>/validation/``.

Usage::

    python -m discell.model.validate --dataset <id> --run reference_best
    python -m discell.model.validate --dataset <id> --sweep-tag sweep2 \\
        --analyses morans,niche,landmarks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from discell import paths
from discell.model.prepare import ModelData, assemble
from discell.model.train import TrainConfig, Trainer

log = logging.getLogger("discell.model.validate")

N_FOLDS = 5
MAX_CELLS_PER_TYPE = 30_000
MIN_CELLS_PER_TYPE = 1_000
DIST_CAP_UM = 500.0
#: near = one-hop composition echo (sanity); mid = the evidence; far = expected
#: failure for both latents. The 30-50 um strip is a deliberate buffer.
BANDS = {"near": (0.0, 30.0), "mid": (50.0, 300.0), "far": (300.0, 500.0)}
#: a latent dim whose variance is below this fraction of its latent's largest
#: per-dim variance is treated as collapsed (hatched, not counted)
COLLAPSED_VAR_FRACTION = 1e-3


# -- shared infrastructure --------------------------------------------------

def load_run(dataset: str, run: str, device: str = "cuda"):
    """Rebuild (config, data, trainer-with-best-weights) for a finished run."""
    import torch

    from discell.model.networks import DisCell

    run_dir = paths.dataset(dataset).root / "runs" / run
    payload = torch.load(run_dir / "best.pt", map_location="cpu",
                         weights_only=False)
    config = TrainConfig(**payload["config"])
    data = assemble(dataset, config.variant, config.embeddings,
                    tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                    v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                    seed=config.seed, label_key=config.label_key)
    device = device if torch.cuda.is_available() else "cpu"
    model = DisCell(n_genes=data.x.shape[1], n_types=len(data.p_t),
                    phi_dim=data.phi.shape[1],
                    median_counts=data.median_counts, d_z=config.d_z,
                    d_w=config.d_w, hidden=config.hidden,
                    gat_dim=config.gat_dim, heads=config.heads).to(device)
    model.load_state_dict(payload["model"])
    trainer = Trainer(config, data)
    trainer.model = model.eval()
    b_matrix = payload["model"]["B.weight"].numpy()
    return config, data, trainer, run_dir, b_matrix


def collect_latents(trainer: Trainer, data: ModelData) -> dict:
    """Posterior means for every cell (sample=False) plus spatial-block folds.

    Folds reuse the tiles as contiguous blocks: tile k -> fold k mod N_FOLDS,
    so every fold is a union of whole tiles and no tile straddles folds.
    """
    swept = trainer._sweep(trainer.train_batches + trainer.val_batches)
    order = np.argsort(swept["nodes"])
    n = data.graph.n_cells
    fold = np.zeros(n, dtype=np.int64)
    for k, tile in enumerate(data.train_tiles + data.val_tiles):
        fold[tile] = k % N_FOLDS
    rows = swept["nodes"][order]
    assert len(rows) == n, "tiles must partition the cells exactly once"
    return {"mu_z": swept["mu_z"][order], "mu_w": swept["mu_w"][order],
            "fold": fold}


def center_per_type(values: np.ndarray, t: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
    """Subtract the per-type mean over ``mask`` cells; zero elsewhere."""
    out = np.zeros_like(values, dtype=np.float64)
    for g in np.unique(t[mask]):
        members = mask & (t == g)
        out[members] = values[members] - values[members].mean(axis=0)
    return out


def ridge_cv(design: np.ndarray, target: np.ndarray, fold: np.ndarray,
             lam: float = 1e-3) -> np.ndarray:
    """Held-out predictions from per-fold ridge fits (fold-train z-scoring)."""
    prediction = np.full(len(target), np.nan)
    for f in np.unique(fold):
        train, test = fold != f, fold == f
        mean = design[train].mean(axis=0)
        scale = design[train].std(axis=0) + 1e-12
        d_train = np.hstack([(design[train] - mean) / scale,
                             np.ones((train.sum(), 1))])
        gram = d_train.T @ d_train + lam * np.eye(d_train.shape[1])
        coef = np.linalg.solve(gram, d_train.T @ target[train])
        d_test = np.hstack([(design[test] - mean) / scale,
                            np.ones((test.sum(), 1))])
        prediction[test] = d_test @ coef
    return prediction


def logistic_cv(design: np.ndarray, labels: np.ndarray,
                fold: np.ndarray, seed: int = 0) -> np.ndarray:
    """Held-out class probabilities, class-weighted, aligned to sorted labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(labels)
    probs = np.zeros((len(labels), len(classes)))
    for f in np.unique(fold):
        train, test = fold != f, fold == f
        scaler = StandardScaler().fit(design[train])
        clf = LogisticRegression(max_iter=500, class_weight="balanced",
                                 random_state=seed)
        clf.fit(scaler.transform(design[train]), labels[train])
        block = clf.predict_proba(scaler.transform(design[test]))
        for j, c in enumerate(clf.classes_):        # a fold may miss a class
            probs[test, np.searchsorted(classes, c)] = block[:, j]
    # rows renormalised: contiguous niches can hide a class from one fold's
    # training half, and multiclass AUC requires probabilities summing to 1
    return probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(1.0 - (target - prediction).var()
                 / max(target.var(), 1e-12))


def macro_auc(target: np.ndarray, probs: np.ndarray) -> float:
    """Macro one-vs-rest AUC; the two-class case needs the 1-d form."""
    from sklearn.metrics import roc_auc_score

    if probs.shape[1] == 2:
        return float(roc_auc_score(target == np.unique(target)[1],
                                   probs[:, 1]))
    return float(roc_auc_score(target, probs, multi_class="ovr",
                               average="macro"))


# -- section 3: per-dimension Moran's I -------------------------------------

def row_normalised_graph(data: ModelData) -> tuple[sp.csr_matrix, np.ndarray]:
    """Binary symmetric adjacency over connected cells, row-normalised."""
    n = data.graph.n_cells
    i, j = data.graph.edge_i, data.graph.edge_j
    adj = sp.coo_matrix(
        (np.ones(2 * len(i)), (np.concatenate([i, j]),
                               np.concatenate([j, i]))),
        shape=(n, n)).tocsr()
    degrees = np.asarray(adj.sum(axis=1)).ravel()
    connected = degrees > 0
    inv = np.zeros(n)
    inv[connected] = 1.0 / degrees[connected]
    return sp.diags(inv) @ adj, connected


def morans_i(values: np.ndarray, weights: sp.csr_matrix,
             n_perms: int = 1000, seed: int = 0) -> dict:
    """I per column of ``values`` (already centred), with a permutation null.

    With row-normalised W the n/sum(W) factor is 1, so
    ``I_d = (v_d' W v_d) / (v_d' v_d)``. The null shuffles the centred values
    over cells (one shared permutation per iteration, valid per dim).
    """
    rng = np.random.default_rng(seed)
    denom = (values ** 2).sum(axis=0).clip(min=1e-12)
    observed = (values * (weights @ values)).sum(axis=0) / denom
    null = np.empty((n_perms, values.shape[1]))
    for p in range(n_perms):
        shuffled = values[rng.permutation(len(values))]
        null[p] = (shuffled * (weights @ shuffled)).sum(axis=0) / denom
    lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
    return {"I": observed.tolist(), "null_lo": lo.tolist(),
            "null_hi": hi.tolist()}


def analysis_morans(data: ModelData, latents: dict, n_perms: int,
                    seed: int) -> dict:
    weights, connected = row_normalised_graph(data)
    weights = weights[connected][:, connected]
    out = {}
    for label in ("mu_z", "mu_w"):
        values = center_per_type(latents[label], data.t, connected)[connected]
        variances = values.var(axis=0)
        result = morans_i(values, weights, n_perms=n_perms, seed=seed)
        result["var_per_dim"] = variances.tolist()
        result["collapsed"] = (variances < COLLAPSED_VAR_FRACTION
                               * variances.max()).tolist()
        out[label] = result
    out["mean_abs_I"] = {
        label: float(np.mean(np.abs(np.array(out[label]["I"])[
            ~np.array(out[label]["collapsed"])])))
        for label in ("mu_z", "mu_w")}

    # section 3.2 triage for hot z dims: a dim that regresses on y is
    # leakage; one that tracks the cycle scores is intrinsic-spatial
    # biology (benign). In-sample by design -- this is a flag, not a metric.
    z_centred = center_per_type(latents["mu_z"], data.t, connected)[connected]
    y_centred = center_per_type(data.graph.y, data.t, connected)[connected]
    gram = y_centred.T @ y_centred + 1e-3 * np.eye(y_centred.shape[1])
    fitted = y_centred @ np.linalg.solve(gram, y_centred.T @ z_centred)
    triage = {"y_r2": (1.0 - (z_centred - fitted).var(axis=0)
                       / z_centred.var(axis=0).clip(min=1e-12)).tolist()}
    if data.cycle is not None:
        scores = np.stack([data.cycle["s_score"],
                           data.cycle["g2m_score"]], axis=1)
        s_centred = center_per_type(scores, data.t, connected)[connected]
        z_unit = z_centred / z_centred.std(axis=0).clip(min=1e-12)
        s_unit = s_centred / s_centred.std(axis=0).clip(min=1e-12)
        triage["cycle_abs_corr"] = np.abs(
            z_unit.T @ s_unit / len(z_unit)).max(axis=1).tolist()
    out["mu_z"]["triage"] = triage
    return out


def morans_figure(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.6))
    offset = 0
    for label, colour in (("mu_z", "#3b6fb6"), ("mu_w", "#c9662a")):
        entry = result[label]
        xs = np.arange(len(entry["I"])) + offset
        bars = ax.bar(xs, entry["I"], color=colour, width=0.8,
                      label=f"{label} (mean |I| "
                            f"{result['mean_abs_I'][label]:.3f})")
        for bar, collapsed in zip(bars, entry["collapsed"]):
            if collapsed:
                bar.set_hatch("///"); bar.set_alpha(0.35)
        ax.fill_between(xs, entry["null_lo"], entry["null_hi"],
                        color="0.6", alpha=0.4, zorder=3,
                        label="permutation null (95%)"
                        if label == "mu_z" else None)
        offset += len(entry["I"]) + 2
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel("Moran's I (per-type centred)")
    ax.set_xlabel("latent dimension (z block | w block; hatched = collapsed)")
    ax.legend(fontsize=8)
    ax.set_title("spatial autocorrelation per latent dimension", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -- section 4: niche invariance of z ---------------------------------------

def niche_labels(data: ModelData, k: int, seed: int) -> np.ndarray:
    """k-means niches over neighbour composition y; -1 for isolated cells.

    Labels come from data (y), never from the latents -- anything else is
    circular. Fitted on a subsample, predicted for every connected cell.
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    connected = data.graph.degrees > 0
    rows = np.flatnonzero(connected)
    fit_rows = rows if len(rows) <= 100_000 else np.sort(
        rng.choice(rows, 100_000, replace=False))
    model = KMeans(k, n_init=4, random_state=seed).fit(data.graph.y[fit_rows])
    labels = np.full(data.graph.n_cells, -1, dtype=np.int64)
    labels[rows] = model.predict(data.graph.y[rows])
    return labels


def analysis_niche(data: ModelData, latents: dict, k_grid: Sequence[int],
                   max_cells: int, seed: int) -> dict:
    """Multinomial ``mu -> niche`` per eligible type; the z column should sit
    at max(floor, l-baseline) while w clears ~0.8 macro AUC."""
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed)
    log_l = np.log(data.totals.clip(min=1.0))[:, None]
    out: dict = {}
    for k in k_grid:
        labels = niche_labels(data, k, seed)
        per_type: dict = {}
        for g in range(len(data.p_t)):
            members = np.flatnonzero((data.t == g) & (labels >= 0))
            counts = np.bincount(labels[members], minlength=k)
            kept = np.flatnonzero(counts >= 500)
            if len(kept) < 2:
                continue
            members = members[np.isin(labels[members], kept)]
            if len(members) > max_cells:
                members = np.sort(rng.choice(members, max_cells,
                                             replace=False))
            target = labels[members]
            fold = latents["fold"][members]
            if len(np.unique(fold)) < 2:
                continue
            floor_target = target.copy()
            rng.shuffle(floor_target)               # within type by design
            designs = {"z": latents["mu_z"][members],
                       "w": latents["mu_w"][members],
                       "lbaseline": log_l[members]}
            scores: dict = {"n_cells": int(len(members)),
                            "n_niches": int(len(kept))}
            for name, design in designs.items():
                probs = logistic_cv(design, target, fold, seed=seed)
                scores[name] = {
                    "auc": macro_auc(target, probs),
                    "balanced_acc": float(balanced_accuracy_score(
                        target, np.unique(target)[probs.argmax(axis=1)]))}
            probs = logistic_cv(latents["mu_z"][members], floor_target,
                                fold, seed=seed)
            scores["floor"] = {"auc": macro_auc(floor_target, probs)}
            per_type[str(data.type_names[g])] = scores
        weights = np.array([v["n_cells"] for v in per_type.values()], float)
        pooled = {name: float(np.average(
            [v[name]["auc"] for v in per_type.values()], weights=weights))
            for name in ("z", "w", "lbaseline", "floor")} if per_type else {}
        out[f"K{k}"] = {"per_type": per_type, "pooled_auc": pooled}
    return out


# -- section 2: distance-to-landmark regression -----------------------------

def _dbscan_instances(positions: np.ndarray, members: np.ndarray,
                      eps_um: float, min_samples: int,
                      max_cells: int | None = None):
    """DBSCAN instances over one candidate set; window on instance size."""
    from sklearn.cluster import DBSCAN

    clusters = DBSCAN(eps=eps_um, min_samples=min_samples).fit_predict(
        positions[members])
    kept, kept_ids = [], []
    for c in range(clusters.max() + 1):
        instance = members[clusters == c]
        if len(instance) < 20 or (max_cells and len(instance) > max_cells):
            continue
        kept.append(instance)
        kept_ids.append(np.full(len(instance), len(kept) - 1))
    if not kept:
        return None, None
    return np.concatenate(kept), np.concatenate(kept_ids)


def landmark_inventory(data: ModelData, eps_um: float = 40.0,
                       min_samples: int = 8) -> dict:
    """Landmark classes for this slide (redefined after the map diagnosis;
    motivations and success checks in the devlog).

    - ``vasculature``: every endothelial type merged, plus pericytes iff
      they demonstrably wrap the vessels (>=50% within 30 um of an
      endothelial cell -- decided by the measured number, logged).
    - smooth muscle restricted to **compact** instances (20-500 cells):
      arteriole rings stay, anatomical wall sheets drop out.
    - interface at **tissue scale**: kNN-smoothed tumour fraction (k = 50),
      compartments = field > 0.5, boundary = graph edges crossing the
      smoothed compartments. A 1-hop definition is degenerate on this
      interleaved slide (57.8% boundary, near-share 84%).
    Instances < 20 cells and classes < 5 instances are dropped throughout;
    everything derives from ``t`` and positions only, same data status as y.
    """
    from scipy.spatial import cKDTree

    names = [str(n) for n in data.type_names]
    classes: dict = {}

    endo_types = [g for g, name in enumerate(names) if "Endothelial" in name]
    vessel_members = np.flatnonzero(np.isin(data.t, endo_types))
    vessel_types = list(endo_types)
    pericyte_types = [g for g, name in enumerate(names) if "Pericyte" in name]
    if len(vessel_members) and pericyte_types:
        pericytes = np.flatnonzero(np.isin(data.t, pericyte_types))
        near_vessel = cKDTree(data.positions[vessel_members]).query(
            data.positions[pericytes])[0] <= 30.0
        log.info("pericyte co-location: %.0f%% within 30 um of endothelium "
                 "-> %s in vasculature", 100 * near_vessel.mean(),
                 "included" if near_vessel.mean() >= 0.5 else "NOT included")
        if near_vessel.mean() >= 0.5:
            vessel_members = np.sort(np.concatenate([vessel_members,
                                                     pericytes]))
            vessel_types += pericyte_types
    candidates = [("vasculature", vessel_members, vessel_types, None)]
    for g, name in enumerate(names):
        if "Smooth Muscle" in name:
            candidates.append((f"{name} (compact)",
                               np.flatnonzero(data.t == g), [g], 500))
    for class_name, members, types, size_cap in candidates:
        constituents, instance_of = _dbscan_instances(
            data.positions, members, eps_um, min_samples, size_cap)
        if constituents is None or instance_of.max() + 1 < 5:
            log.info("landmark class dropped (too few instances): %s",
                     class_name)
            continue
        classes[class_name] = {
            "constituents": constituents, "instance_of": instance_of,
            "exclude_types": types,
            "n_instances": int(instance_of.max()) + 1}

    # interface: smooth the tumour indicator over ~tissue scale first
    tumour_types = [g for g, name in enumerate(names)
                    if "Tumor Cells" in name or "Malignant" in name]
    is_tumour = np.isin(data.t, tumour_types).astype(np.float64)
    k = min(50, data.graph.n_cells - 1)
    neighbours = cKDTree(data.positions).query(data.positions, k=k + 1)[1]
    compartment = is_tumour[neighbours].mean(axis=1) > 0.5
    across = compartment[data.graph.edge_i] != compartment[data.graph.edge_j]
    boundary = np.zeros(data.graph.n_cells, dtype=bool)
    boundary[data.graph.edge_i[across]] = True
    boundary[data.graph.edge_j[across]] = True
    connected = data.graph.degrees > 0
    log.info("compartment interface: %.1f%% of connected cells are boundary "
             "(kNN-smoothed compartments, k=%d)",
             100 * boundary.sum() / max(connected.sum(), 1), k)
    classes["tumor_stroma_interface"] = {
        "constituents": np.flatnonzero(boundary), "instance_of": None,
        "exclude_types": [], "n_instances": None}
    for name, entry in classes.items():
        log.info("landmark class %s: %d constituent cells%s", name,
                 len(entry["constituents"]),
                 "" if entry["n_instances"] is None
                 else f" in {entry['n_instances']} instances")
    return classes


def landmark_map_figure(data: ModelData, classes: dict, path: Path) -> None:
    """Where each landmark class sits, and what its distance field means.

    Top row: constituents over the tissue, coloured by DBSCAN instance --
    sheets, specks and genuine vessel-like structures are immediately
    distinguishable. Bottom row: every cell coloured by its band under
    d(i, L): near (echo), buffer, mid (the evidence), far, out-of-cap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    show = np.sort(rng.choice(data.graph.n_cells,
                              min(120_000, data.graph.n_cells),
                              replace=False))
    pos = data.positions
    cmap = ListedColormap(["#d73027", "#fdae61", "#2c7fb8", "#c6dbef",
                           "#f0f0f0"])
    norm = BoundaryNorm([0, 30, 50, 300, 500, 1e9], cmap.N)
    fig, axes = plt.subplots(2, len(classes),
                             figsize=(3.9 * len(classes), 7.4),
                             squeeze=False)
    scatter = None
    for c, (name, entry) in enumerate(classes.items()):
        cons = entry["constituents"]
        ax = axes[0, c]
        ax.scatter(pos[show, 0], pos[show, 1], s=0.3, color="0.88",
                   rasterized=True)
        if entry["instance_of"] is not None:
            ax.scatter(pos[cons, 0], pos[cons, 1], s=0.8,
                       c=entry["instance_of"] % 20, cmap="tab20",
                       rasterized=True)
        else:
            ax.scatter(pos[cons, 0], pos[cons, 1], s=0.4, color="#7b3294",
                       rasterized=True)
        ax.set_title(f"{name[:30]}\n{len(cons)} cells"
                     + (f", {entry['n_instances']} instances"
                        if entry["n_instances"] else ""), fontsize=8)
        distance = cKDTree(pos[cons]).query(pos[show])[0]
        ax = axes[1, c]
        scatter = ax.scatter(pos[show, 0], pos[show, 1], c=distance, s=0.4,
                             cmap=cmap, norm=norm, rasterized=True)
        share = [(distance <= 30).mean(), ((distance > 50)
                                           & (distance <= 300)).mean(),
                 (distance > 500).mean()]
        ax.set_title(f"near {share[0]:.0%} | mid {share[1]:.0%} | "
                     f"beyond cap {share[2]:.0%}", fontsize=8)
    for ax in axes.ravel():
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    bar = fig.colorbar(scatter, ax=axes[1, :], orientation="horizontal",
                       fraction=0.05, pad=0.04,
                       ticks=[15, 40, 175, 400, 750])
    bar.ax.set_xticklabels(["near <30", "buffer", "mid 50-300", "far -500",
                            "beyond cap"], fontsize=7)
    fig.suptitle("landmark inventory and the distance target d(i, L)",
                 fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analysis_landmarks(data: ModelData, latents: dict, b_matrix: np.ndarray,
                       max_cells: int, seed: int,
                       map_path: Path | None = None) -> dict:
    """Banded distance regression per landmark class, both latents, plus the
    direction analysis (theta cosines, gene signatures) on the w side."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    log_l = np.log(data.totals.clip(min=1.0))[:, None]
    classes = landmark_inventory(data)
    if map_path is not None:
        landmark_map_figure(data, classes, map_path)
    gene_names = (np.asarray(data.gene_names)
                  if data.gene_names is not None else None)
    out: dict = {"classes": {}}
    thetas: dict = {}
    for name, entry in classes.items():
        distance = cKDTree(data.positions[entry["constituents"]]).query(
            data.positions)[0]
        target_all = np.log1p(np.minimum(distance, DIST_CAP_UM))
        constituent = np.zeros(data.graph.n_cells, dtype=bool)
        constituent[entry["constituents"]] = True
        eligible = (distance <= DIST_CAP_UM) & ~constituent \
            & ~np.isin(data.t, entry["exclude_types"])

        per_type: dict = {}
        theta_by_type: dict = {}
        for g in range(len(data.p_t)):
            type_rows = eligible & (data.t == g)
            if type_rows.sum() < MIN_CELLS_PER_TYPE:
                continue
            bands: dict = {"n_cells": int(type_rows.sum())}
            # fit *within* each band: the mid-band question is "does the
            # latent order cells by distance where one hop cannot see the
            # landmark" -- a pooled 0-500 fit scored band-locally punishes
            # the global slope, not that question (all-negative R2 with the
            # floor at zero was the tell)
            for band, (lo, hi) in BANDS.items():
                members = np.flatnonzero(type_rows & (distance > lo)
                                         & (distance <= hi))
                if len(members) < 500:
                    continue
                if len(members) > max_cells:
                    members = np.sort(rng.choice(members, max_cells,
                                                 replace=False))
                target = target_all[members]
                fold = latents["fold"][members]
                if len(np.unique(fold)) < 2:
                    continue
                floor_target = target.copy()
                rng.shuffle(floor_target)
                for probe_name, design in (
                        ("w", latents["mu_w"][members]),
                        ("z", latents["mu_z"][members]),
                        ("lbaseline", log_l[members]),
                        # y-baseline: raw one-hop composition. The landmark
                        # sets are type-defined, so any w-signal this matches
                        # is definitional (composition reads the target), not
                        # induced expression response
                        ("ybaseline", data.graph.y[members])):
                    bands.setdefault(probe_name, {})[band] = r2(
                        target, ridge_cv(design, target, fold))
                bands.setdefault("floor", {})[band] = r2(
                    floor_target, ridge_cv(latents["mu_w"][members],
                                           floor_target, fold))
                if band == "mid":
                    # the direction, from the evidence band: full-fit ridge
                    # on raw w so B @ theta-hat lives in decoder space
                    w_members = latents["mu_w"][members]
                    centred = w_members - w_members.mean(axis=0)
                    gram = centred.T @ centred + 1e-3 * np.eye(
                        centred.shape[1])
                    theta = np.linalg.solve(
                        gram, centred.T @ (target - target.mean()))
                    theta_by_type[str(data.type_names[g])] = (
                        theta / max(np.linalg.norm(theta), 1e-12))
            if len(bands) > 1:
                per_type[str(data.type_names[g])] = bands

        pooled = {}
        for probe_name in ("w", "z", "lbaseline", "ybaseline", "floor"):
            for band in BANDS:
                values = [(v[probe_name][band], v["n_cells"])
                          for v in per_type.values()
                          if band in v.get(probe_name, {})]
                if values:
                    pooled.setdefault(probe_name, {})[band] = float(
                        np.average([v for v, _ in values],
                                   weights=[n for _, n in values]))
        # one theta per class: cell-weighted mean of per-type unit vectors
        if theta_by_type:
            stack = np.stack(list(theta_by_type.values()))
            mean_theta = stack.mean(axis=0)
            mean_theta /= max(np.linalg.norm(mean_theta), 1e-12)
            thetas[name] = mean_theta
            consistency = [float(stack[a] @ stack[b])
                           for a in range(len(stack))
                           for b in range(a + 1, len(stack))]
        else:
            consistency = []
        out["classes"][name] = {
            "n_instances": entry["n_instances"],
            "n_constituents": int(len(entry["constituents"])),
            "per_type": per_type, "pooled": pooled,
            "theta_cross_type_cosine": (float(np.mean(consistency))
                                        if consistency else None)}

    # cosines between classes + gene signatures (valid within this model)
    keys = list(thetas)
    out["theta_cosines"] = {
        f"{a} ~ {b}": float(thetas[a] @ thetas[b])
        for i, a in enumerate(keys) for b in keys[i + 1:]}
    if gene_names is not None:
        # B rows for never-expressed genes (control/mutant/viral probes) are
        # unconstrained by any gradient -- mask to expressed genes or the
        # signature is decoder noise
        prevalence = np.asarray((data.x > 0).mean(axis=0)).ravel()
        expressed = prevalence >= 0.01
        out["gene_signatures"] = {}
        for name, theta in thetas.items():
            loading = np.where(expressed, b_matrix @ theta, 0.0)
            order = np.argsort(loading)
            out["gene_signatures"][name] = {
                "high": [(str(gene_names[i]), float(loading[i]))
                         for i in order[-15:][::-1]],
                "low": [(str(gene_names[i]), float(loading[i]))
                        for i in order[:15]]}
    return out


def landmark_figure(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(result["classes"])
    fig, axes = plt.subplots(1, max(len(classes), 1),
                             figsize=(4.2 * max(len(classes), 1), 3.4),
                             squeeze=False)
    styles = {"w": ("#c9662a", "-"), "z": ("#3b6fb6", "-"),
              "lbaseline": ("0.4", "--"), "ybaseline": ("#7b3294", "--"),
              "floor": ("0.6", ":")}
    xs = np.arange(len(BANDS))
    for ax, name in zip(axes[0], classes):
        pooled = result["classes"][name]["pooled"]
        for probe_name, (colour, ls) in styles.items():
            values = [pooled.get(probe_name, {}).get(band, np.nan)
                      for band in BANDS]
            ax.plot(xs, values, marker="o", color=colour, ls=ls,
                    label=probe_name)
        ax.set_xticks(xs, list(BANDS))
        ax.set_title(name[:34], fontsize=8)
        ax.set_ylabel("held-out R²", fontsize=8)
        ax.axhline(0, color="0.3", lw=0.6)
        ax.legend(fontsize=7)
    fig.suptitle("distance-to-landmark by band (mid band is the evidence)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -- section 5: the allegiance matrix ---------------------------------------

def cycle_row(data: ModelData, latents: dict, max_cells: int,
              seed: int) -> dict:
    """S/G2M from both latents under this doc's conventions (block CV,
    within cycling types), with floor, l-baseline and 50-PC ceiling."""
    rng = np.random.default_rng(seed)
    cyc = data.cycle
    names = [str(n) for n in data.type_names]
    types = [g for g in cyc["cycling_types"] if "nassigned" not in names[g]][:4]
    scores = np.stack([cyc["s_score"], cyc["g2m_score"]], axis=1)
    members = np.flatnonzero(np.isin(data.t, types)
                             & (data.graph.degrees > 0))
    if len(members) > max_cells:
        members = np.sort(rng.choice(members, max_cells, replace=False))
    target = center_per_type(scores[members], data.t[members],
                             np.ones(len(members), dtype=bool))
    fold = latents["fold"][members]
    log_l = np.log(data.totals.clip(min=1.0))[:, None]
    designs = {"z": latents["mu_z"][members], "w": latents["mu_w"][members],
               "ceiling": cyc["x_pcs"][members],
               "lbaseline": log_l[members]}
    out = {}
    for name, design in designs.items():
        design = center_per_type(design, data.t[members],
                                 np.ones(len(members), dtype=bool))
        held = np.stack([ridge_cv(design, target[:, k], fold)
                         for k in range(2)], axis=1)
        out[name] = float(np.mean([r2(target[:, k], held[:, k])
                                   for k in range(2)]))
    floor_target = target.copy()
    for g in types:
        sel = np.flatnonzero(data.t[members] == g)
        floor_target[sel] = target[sel[rng.permutation(len(sel))]]
    held = np.stack([ridge_cv(designs["z"], floor_target[:, k], fold)
                     for k in range(2)], axis=1)
    out["floor"] = float(np.mean([r2(floor_target[:, k], held[:, k])
                                  for k in range(2)]))
    return out


def allegiance_matrix(results: dict, pseudo: dict, path: Path) -> dict:
    """Rows = targets with known allegiance, columns = mu_z / mu_w; the
    expected pattern is block-diagonal, every cold cell certified."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    niche = results["niche"][next(iter(results["niche"]))]["pooled_auc"]
    mid = {probe: results["landmarks"]["classes"][name]["pooled"]
           .get(probe, {}).get("mid")
           for name in list(results["landmarks"]["classes"])[:1]
           for probe in ("z", "w", "lbaseline", "floor")}
    mid_all = {probe: float(np.mean([
        entry["pooled"][probe]["mid"]
        for entry in results["landmarks"]["classes"].values()
        if "mid" in entry["pooled"].get(probe, {})]))
        for probe in ("z", "w", "lbaseline", "floor")}
    cycle = results["cycle_row"]
    matrix = {
        "S/G2M score (intrinsic)": {
            "z": cycle["z"], "w": cycle["w"], "floor": cycle["floor"],
            "lbaseline": cycle["lbaseline"], "ceiling": cycle["ceiling"],
            "unit": "R²"},
        "niche label (spatial)": {
            "z": niche["z"], "w": niche["w"], "floor": niche["floor"],
            "lbaseline": niche["lbaseline"], "unit": "macro AUC"},
        "mid-band landmark distance (spatial)": {
            "z": mid_all["z"], "w": mid_all["w"], "floor": mid_all["floor"],
            "lbaseline": mid_all["lbaseline"], "unit": "R²"},
        "pseudotime tissue-gradient (existing)": {
            "z": pseudo["z"]["niche_r2"], "w": pseudo["w"]["niche_r2"],
            "unit": "niche R² (type-partialled)"},
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rows = list(matrix)
    cells = np.zeros((len(rows), 2))
    for r, row in enumerate(rows):
        entry = matrix[row]
        floor = entry.get("floor", 0.0)
        span = max(abs(entry["z"] - floor), abs(entry["w"] - floor), 1e-9)
        cells[r] = [(entry["z"] - floor) / span, (entry["w"] - floor) / span]
    ax.imshow(cells, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for r, row in enumerate(rows):
        entry = matrix[row]
        for c, col in enumerate(("z", "w")):
            refs = " ".join(f"{k[0]}={entry[k]:.2f}" for k in
                            ("floor", "lbaseline", "ceiling") if k in entry)
            ax.text(c, r, f"{entry[col]:.2f}\n[{refs}]", ha="center",
                    va="center", fontsize=7,
                    color="white" if cells[r, c] < 0.6 else "black")
    ax.set_xticks([0, 1], ["mu_z", "mu_w"])
    ax.set_yticks(range(len(rows)),
                  [f"{r} ({matrix[r]['unit']})" for r in rows], fontsize=7)
    ax.set_title("allegiance matrix: each target hot in exactly one column",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    matrix["mid_band_first_class_only"] = mid
    return matrix


# -- orchestration ----------------------------------------------------------

def run_analyses(args: argparse.Namespace, run: str) -> dict:
    config, data, trainer, run_dir, b_matrix = load_run(
        args.dataset, run, args.device)
    out_dir = run_dir / "validation"
    out_dir.mkdir(exist_ok=True)
    latents = collect_latents(trainer, data)
    wanted = args.analyses.split(",")
    # partial reruns update their sections and keep the rest of the record
    results: dict = {}
    if (out_dir / "validation.json").exists():
        results = json.loads((out_dir / "validation.json").read_text())
    results.update({"run": run, "kappa": config.kappa, "seed": config.seed})

    if "morans" in wanted or "matrix" in wanted:
        results["morans"] = analysis_morans(data, latents, args.n_perms,
                                            config.seed)
        morans_figure(results["morans"], out_dir / "morans_i.png")
        log.info("moran mean |I|: %s", results["morans"]["mean_abs_I"])
    if "niche" in wanted or "matrix" in wanted:
        k_grid = (10,) if args.headline_k_only else (10, 6, 15)
        results["niche"] = analysis_niche(data, latents, k_grid,
                                          args.max_cells, config.seed)
        log.info("niche pooled AUC (K=10): %s",
                 results["niche"]["K10"]["pooled_auc"])
    if "landmarks" in wanted or "matrix" in wanted:
        results["landmarks"] = analysis_landmarks(
            data, latents, b_matrix, args.max_cells, config.seed,
            map_path=out_dir / "landmark_map.png")
        landmark_figure(results["landmarks"], out_dir / "landmark_bands.png")
    if "matrix" in wanted:
        from discell.model.report import pseudotime_cross_check

        results["cycle_row"] = cycle_row(data, latents, args.max_cells,
                                         config.seed)
        rows = np.arange(data.graph.n_cells)
        pseudo = pseudotime_cross_check(
            latents["mu_z"], latents["mu_w"], data.graph.y, data.t,
            data.positions, data.graph.edge_i, data.graph.edge_j, rows,
            out_dir / "pseudotime_tissue.png", seed=config.seed)
        results["pseudotime"] = pseudo
        results["matrix"] = allegiance_matrix(
            results, pseudo, out_dir / "allegiance_matrix.png")

    (out_dir / "validation.json").write_text(
        json.dumps(results, indent=2, default=float))
    log.info("wrote %s", out_dir / "validation.json")
    return results


def sweep_companion(args: argparse.Namespace) -> None:
    """The per-kappa line plot: run the cheap analyses over a sweep tag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from discell.model.sweep import run_name

    rows = []
    for seed in args.seeds:
        for kappa in args.kappas:
            run = run_name(kappa, seed, args.sweep_tag)
            run_dir = paths.dataset(args.dataset).root / "runs" / run
            cached = run_dir / "validation" / "validation.json"
            if cached.exists() and not args.force:
                rows.append(json.loads(cached.read_text()))
                continue
            if not (run_dir / "best.pt").exists():
                log.warning("missing %s -- skipped", run)
                continue
            rows.append(run_analyses(args, run))

    def series(picker):
        by_kappa: dict = {}
        for row in rows:
            try:
                by_kappa.setdefault(row["kappa"], []).append(picker(row))
            except (KeyError, TypeError):
                continue
        return by_kappa

    panels = {
        "Moran mean |I| (z)": series(lambda r: r["morans"]["mean_abs_I"]["mu_z"]),
        "Moran mean |I| (w)": series(lambda r: r["morans"]["mean_abs_I"]["mu_w"]),
        "niche AUC (z)": series(
            lambda r: r["niche"]["K10"]["pooled_auc"]["z"]),
        "niche AUC (w)": series(
            lambda r: r["niche"]["K10"]["pooled_auc"]["w"]),
        "mid-band R² (z)": series(lambda r: float(np.mean(
            [c["pooled"]["z"]["mid"]
             for c in r["landmarks"]["classes"].values()]))),
        "mid-band R² (w)": series(lambda r: float(np.mean(
            [c["pooled"]["w"]["mid"]
             for c in r["landmarks"]["classes"].values()]))),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (title_z, title_w) in zip(axes, [
            ("Moran mean |I| (z)", "Moran mean |I| (w)"),
            ("niche AUC (z)", "niche AUC (w)"),
            ("mid-band R² (z)", "mid-band R² (w)")]):
        for title, colour in ((title_z, "#3b6fb6"), (title_w, "#c9662a")):
            data_k = panels[title]
            kappas = sorted(data_k)
            mean = [float(np.mean(data_k[k])) for k in kappas]
            lo = [float(np.min(data_k[k])) for k in kappas]
            hi = [float(np.max(data_k[k])) for k in kappas]
            ax.plot(kappas, mean, marker="o", color=colour,
                    label=title.split(" (")[0] + " " + title[-2])
            ax.fill_between(kappas, lo, hi, color=colour, alpha=0.2)
        ax.set_xlabel("kappa")
        ax.set_title(title_z.split(" (")[0], fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("allegiance vs kappa (seed envelopes)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = paths.dataset(args.dataset).root / "experiments"
    fig.savefig(out / f"validation_{args.sweep_tag}_kappa.png", dpi=130)
    plt.close(fig)
    (out / f"validation_{args.sweep_tag}.json").write_text(
        json.dumps(rows, indent=2, default=float))
    log.info("wrote %s", out / f"validation_{args.sweep_tag}.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", default=None,
                        help="single run to validate (headline mode)")
    parser.add_argument("--sweep-tag", default=None,
                        help="validate a whole sweep tag instead")
    parser.add_argument("--kappas", type=float, nargs="*",
                        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--analyses", default="morans,niche,landmarks,matrix")
    parser.add_argument("--max-cells", type=int, default=MAX_CELLS_PER_TYPE)
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--headline-k-only", action="store_true",
                        help="skip the K=6/15 niche robustness grid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.sweep_tag:
        sweep_companion(args)
    elif args.run:
        run_analyses(args, args.run)
    else:
        parser.error("one of --run / --sweep-tag is required")
    return 0


if __name__ == "__main__":
    sys.exit(main())
