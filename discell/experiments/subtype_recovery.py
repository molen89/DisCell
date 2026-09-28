#!/usr/bin/env python3
"""Subtype recovery: which channel reads the sublabels the lineage relabel merged.

Pre-registered in the devlog, "Two additions before the paper numbers
(motivation, 2026-09-28)", part B. The lineage relabel merged *states*
(proliferative, VEGFA+, inflammatory, activated, myofibroblast, ...) and
*locations* (tumour- / stroma-associated fibroblasts and endothelium) into
their lineages; the old labels are now held-out targets. A state is intrinsic
and should be read from z; a location is context and should be read from w
(or the context prior mean m_psi(c, t)); neither from the other channel.

Per lineage that absorbed >= 2 old labels (``labels/lineage_map_applied``;
for lung / FF the old ``graphclust`` clusters of a lineage), on the run's own
split -- training-tile seeds fit, validation-tile seeds (held out) grade --
a probe from each channel to the old sublabel:

``z``     mu_z, the posterior mean of the intrinsic latent
``w``     mu_w, the posterior mean of the response latent
``mpsi``  m_psi(c, t), the context prior mean (within a lineage t is
          constant, so this is a function of the context c alone)
``zw``    [mu_z, mu_w]

Graders. **ridge** (primary): the ridge of ``metrics.probe_delta_ce``
(intercept, penalty 1e-3) on the one-hot sublabel -- a multinomial
least-squares probe -- on a design standardised over the training rows.
**mlp** (z and w): ``calibrate.mlp_fit_predict`` on the same one-hot target
and standardised design. Each grader is fitted twice on all training cells of
the lineage: *balanced* (rows weighted by inverse sublabel frequency; for
the MLP, minibatches from every sublabel resampled to one size, at most
``MAX_TRAIN_PER_CLASS``), whose argmax gives balanced accuracy, macro-F1 and
per-sublabel recall; and *natural* (unweighted), whose column k scores
sublabel k against the rest of the lineage as it occurs -- the one-vs-rest
AUC. (A balanced fit's column k is tuned against rare sublabels the
held-out "rest" barely contains; its held-out one-vs-rest AUC can fall below
1/2.) Held-out cells are all graded.

Reads per probe: balanced accuracy, macro-F1, per-sublabel one-vs-rest
recall (argmax) and one-vs-rest AUC; the within-lineage permutation floor
(the sublabels permuted within the lineage, separately among training and
held-out cells, the probe refit, ``N_PERM`` draws); the frequency baseline
(the training majority sublabel for every cell: BA = 1/K); 95 %
tile-bootstrap intervals (``bootstrap.tile_bootstrap``, conditional on the
fitted probes) for every read and for the paired differences z - w and
z - m_psi on the same draws.

Which channel recovers a sublabel (the call; not pre-registered, fixed here
before the first real read). It rests on the AUC, not the recall: among
sublabels a channel cannot separate, the argmax is decided by fit noise, so a
sublabel's recall can land anywhere between 0 and 1 while the lineage's BA
and the AUC stay put. A channel *recovers* a sublabel when its AUC's 95 %
interval lies above the channel's floor mean. For a pair (a, b):
``a ≫ b`` when AUC_a - AUC_b has its interval above 0 and b's excess over
its floor is at most ``DOMINANCE`` (half) of a's; ``both`` when both recover
without dominance (in the summary ``both (a > b)`` when the difference's
interval also excludes 0); ``neither`` when neither recovers; ``unresolved``
when one recovers and dominance is not established. The verdict against the
expectation uses the ridge calls: a state is met when ``z ≫ w``; a location
when ``w ≫ z`` or ``mpsi ≫ z``. Contradictions (both findings,
pre-registered): a location read from z (z recovers it and no w-side
channel dominates -- niche in z); a state read from w (w or m_psi recovers
it without z dominating -- over-merging feeding the context regression).

The mesothelial cyst lining is its own lineage under the applied map
(``cyst=mesothelial``), so no within-lineage probe contains it. On ovarian it
is read in one declared cross-lineage group, the curators' malignant set
(Tumour + cyst lining), flagged ``t_confounded``: every encoder (enc_z,
enc_w, m_psi) takes onehot(t) as input, so there any channel can read the
cyst lining through t and the expectation "cyst lining from z" is not
testable on these runs.

Usage::

    python -m discell.experiments.subtype_recovery --dataset <id> \\
        --run finalL_s0 finalL_s1 finalL_s2 [--device cuda] [--force]

Writes ``runs/<run>/subtype_recovery.json`` per run (kept without
``--force``, so a relaunch only adds what is missing) and
``experiments/subtype_recovery.{json,md,png}`` over the runs given
(3-seed mean [min, max]).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

log = logging.getLogger("discell.experiments.subtype_recovery")

CHANNELS = ("z", "w", "mpsi", "zw")
MLP_CHANNELS = ("z", "w")
DISPLAY = {"z": "μ_z", "w": "μ_w", "mpsi": "m_ψ", "zw": "[μ_z, μ_w]"}
#: the channel pairs a call is made for, (grader, a, b)
PAIRS = (("ridge", "z", "w"), ("ridge", "z", "mpsi"), ("mlp", "z", "w"))
N_PERM = 5
N_BOOT = 1000
RIDGE_PENALTY = 1e-3
#: a sublabel enters its lineage's probe with at least this many training /
#: held-out cells (otherwise listed under ``dropped``)
MIN_TRAIN, MIN_TEST = 20, 10
MAX_TRAIN_PER_CLASS = 10_000
MLP_THREADS = 2
#: "a ≫ b": b's excess over its floor is at most this share of a's
DOMINANCE = 0.5
EXPECTATION = {"state": "z", "location": "w/m_ψ"}
CYST_LABEL = "Malignant Cells Lining Cyst"
T_CONFOUNDED = "z (t-confounded: untestable)"
#: declared cross-lineage groups, (name, lineages); see the module docstring
EXTRA_GROUPS = {
    "xenium_prime_ovarian_cancer_ffpe": (
        ("Tumour + cyst lining (curators' malignant)",
         ("Tumour", "Mesothelial-like cyst lining")),),
}
#: lung / FF are informational (pre-registration): the verdict is reported,
#: not scored
INFORMATIONAL = ("xenium_prime_human_lung_cancer_ffpe",
                 "xenium_prime_human_ovary_ff")
SHORT = {"xenium_prime_human_ovary_ff": "FF",
         "xenium_prime_human_lung_cancer_ffpe": "lung",
         "gse315411_pdltma06_11_prime_solo": "GSE",
         "gse315411_pdltma06_10_prime_dual": "GSE dual",
         "xenium_prime_ovarian_cancer_ffpe": "ovarian"}


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def label_tables(dataset: str) -> dict:
    """The applied map (old label -> lineage pairs, obs keys) and each old
    label's ``kind`` (from the proposal the applied map was built from)."""
    from discell import paths

    root = paths.dataset(dataset).root / "labels"
    meta = json.loads((root / "lineage_map_applied.json").read_text())
    with open(root / meta.get("applied_csv", "lineage_map_applied.csv")) as fh:
        pairs = {(r["source_label"], r["lineage"]) for r in csv.DictReader(fh)}
    with open(root / "lineage_map_proposed.csv") as fh:
        kinds = {r["source_label"]: r["kind"] for r in csv.DictReader(fh)}
    return {"source_key": meta["source_key"], "obs_key": meta["obs_key"],
            "pairs": pairs, "kinds": kinds}


def fold_unassigned(values) -> np.ndarray:
    """The loader's rule (``CellGraphDataset._build_labels``): NaN becomes the
    panel's own spelling of Unassigned."""
    import pandas as pd

    from discell.data.loader import UNASSIGNED

    series = pd.Series(values).astype(object)
    spellings = {str(v).casefold(): str(v) for v in series.dropna().unique()}
    return series.where(series.notna(),
                        spellings.get(UNASSIGNED.casefold(), UNASSIGNED)
                        ).astype(str).to_numpy()


def old_labels(dataset: str, variant: str, lineage_names: np.ndarray,
               tables: dict) -> np.ndarray:
    """The old label of every cell, in the bundle's (= the model data's) cell
    order; the bundle's lineage column must equal the run's type labels."""
    import anndata as ad

    from discell import paths

    ds = paths.dataset(dataset)
    adata = ad.read_h5ad(ds.bundle_dir / f"{variant}.h5ad", backed="r")
    try:
        old = fold_unassigned(adata.obs[tables["source_key"]].to_numpy())
        lineage = fold_unassigned(adata.obs[tables["obs_key"]].to_numpy())
    finally:
        adata.file.close()
    if len(lineage) != len(lineage_names) or not np.array_equal(
            lineage, np.asarray(lineage_names, dtype=str)):
        raise ValueError(f"{dataset}/{variant}: bundle column "
                         f"{tables['obs_key']!r} does not match the run's labels")
    return old


def build_groups(old: np.ndarray, lineage: np.ndarray, tables: dict,
                 extra: Sequence = ()) -> list[dict]:
    """Every lineage with >= 2 old labels among its cells, plus the declared
    cross-lineage groups. Each observed (old, lineage) pair must be in the
    applied map."""
    seen = set(zip(old.tolist(), lineage.tolist()))
    unknown = seen - tables["pairs"]
    if unknown:
        raise ValueError(f"pairs not in the applied map: {sorted(unknown)[:5]}")
    groups = []
    for lin in sorted(set(lineage.tolist())):
        subs = sorted(set(old[lineage == lin].tolist()))
        if len(subs) >= 2:
            groups.append({"name": lin, "lineages": [lin],
                           "t_confounded": False})
    for name, lins in extra:
        if all(l in set(lineage.tolist()) for l in lins):
            groups.append({"name": name, "lineages": list(lins),
                           "t_confounded": True})
    return groups


def expectation(label: str, kind: str | None, t_confounded: bool) -> str | None:
    if label == CYST_LABEL:
        return T_CONFOUNDED if t_confounded else "z"
    return EXPECTATION.get(kind or "")


# --------------------------------------------------------------------------
# graders and scores (model-free)
# --------------------------------------------------------------------------


def _standardise(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (train - mean) / sd, (test - mean) / sd


def ridge_scores(x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, k: int, balanced: bool) -> np.ndarray:
    """Multinomial least-squares ridge on the one-hot sublabel: (n_test, K)
    scores. *balanced* weights the rows by inverse sublabel frequency (the
    argmax then targets balanced accuracy); unweighted, column k is the
    one-vs-rest least-squares probe of sublabel k against the rest as it
    occurs (least squares decouples across target columns)."""
    a, b = _standardise(np.asarray(x_train, np.float64),
                        np.asarray(x_test, np.float64))
    if balanced:
        counts = np.bincount(y_train, minlength=k).astype(np.float64)
        weight = (len(y_train) / (k * np.maximum(counts, 1.0)))[y_train]
    else:
        weight = np.ones(len(y_train))
    d = np.hstack([a, np.ones((len(a), 1))])
    target = np.eye(k)[y_train]
    gram = (d * weight[:, None]).T @ d + RIDGE_PENALTY * np.eye(d.shape[1])
    coef = np.linalg.solve(gram, (d * weight[:, None]).T @ target)
    return np.hstack([b, np.ones((len(b), 1))]) @ coef


def mlp_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
               k: int, seed: int, balanced: bool) -> np.ndarray:
    """``calibrate.mlp_fit_predict`` on the one-hot sublabel: (n_test, K)
    scores. *balanced*: minibatches drawn from every sublabel resampled to
    one size (at most ``MAX_TRAIN_PER_CLASS``); otherwise from the training
    rows as they occur."""
    from discell.model.calibrate import mlp_fit_predict

    a, b = _standardise(np.asarray(x_train, np.float64),
                        np.asarray(x_test, np.float64))
    if balanced:
        rng = np.random.default_rng([seed, 7])
        counts = np.bincount(y_train, minlength=k)
        size = int(min(counts.max(), MAX_TRAIN_PER_CLASS))
        rows = np.concatenate([
            rng.choice(np.flatnonzero(y_train == c), size,
                       replace=counts[c] < size)
            for c in range(k) if counts[c]])
    else:
        rows = np.arange(len(a))
    design = np.vstack([a, b]).astype(np.float32)
    target = np.vstack([np.eye(k)[y_train], np.zeros((len(b), k))])
    test_rows = np.arange(len(a), len(design))
    import torch

    # a 64-wide net on 2048-row batches: more intra-op threads only wait on
    # each other when the machine is shared
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, MLP_THREADS))
    try:
        out = mlp_fit_predict(design, target, rows, test_rows, seed)
    finally:
        torch.set_num_threads(threads)
    return np.asarray(out, dtype=np.float64)


def confusion(y: np.ndarray, pred: np.ndarray, k: int,
              weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted confusion matrix, rows = truth, columns = prediction."""
    return np.bincount(y * k + pred, weights=weights,
                       minlength=k * k).reshape(k, k)


def scores(cm: np.ndarray) -> dict:
    """Balanced accuracy, macro-F1 and per-class recall from a confusion
    matrix; a class absent from the truth is left out of both means."""
    truth, called = cm.sum(axis=1), cm.sum(axis=0)
    hit = np.diag(cm)
    present = truth > 0
    recall = np.where(present, hit / np.where(present, truth, 1.0), np.nan)
    precision = np.where(called > 0, hit / np.where(called > 0, called, 1.0), 0.0)
    rec0 = np.nan_to_num(recall)
    denom = precision + rec0
    f1 = np.where(denom > 0, 2 * precision * rec0 / np.where(denom > 0, denom, 1.0),
                  0.0)
    if not present.any():
        return {"ba": float("nan"), "macro_f1": float("nan"), "recall": recall}
    return {"ba": float(recall[present].mean()),
            "macro_f1": float(f1[present].mean()), "recall": recall}


def auc_ranks(score: np.ndarray) -> tuple[np.ndarray, int]:
    """Rank group of every score (ties share one), for :func:`weighted_auc`."""
    _, groups = np.unique(score, return_inverse=True)
    return groups.ravel(), int(groups.max()) + 1


def weighted_auc(groups: np.ndarray, n_groups: int, positive: np.ndarray,
                 weights: np.ndarray | None = None) -> float:
    """One-vs-rest AUC (Mann-Whitney, ties counted half) under per-cell
    weights; NaN without both a positive and a negative."""
    w = np.ones(len(groups)) if weights is None else weights
    pos = np.bincount(groups, weights=w * positive, minlength=n_groups)
    neg = np.bincount(groups, weights=w * ~positive, minlength=n_groups)
    below = np.cumsum(neg) - neg
    den = pos.sum() * neg.sum()
    if den <= 0:
        return float("nan")
    return float((pos * (below + 0.5 * neg)).sum() / den)


def _call(rec_a: bool, rec_b: bool, e_a: float, e_b: float,
          diff_ci: Sequence[float], a: str, b: str) -> str:
    lo, hi = diff_ci
    if not rec_a and not rec_b:
        return "neither"
    if lo > 0 and e_b <= DOMINANCE * e_a:
        return f"{a} ≫ {b}"
    if hi < 0 and e_a <= DOMINANCE * e_b:
        return f"{b} ≫ {a}"
    if rec_a and rec_b:
        return "both"
    return "unresolved"


def _floor_read(y_te: np.ndarray, balanced: np.ndarray,
                natural: np.ndarray, k: int) -> dict:
    out = scores(confusion(y_te, balanced.argmax(axis=1), k))
    out["auc"] = np.array([weighted_auc(*auc_ranks(natural[:, c]), y_te == c)
                           for c in range(k)])
    return out


def grade_group(latents: Mapping[str, np.ndarray], y: np.ndarray,
                names: Sequence[str], train: np.ndarray, test: np.ndarray,
                positions: np.ndarray, seed: int = 0, n_perm: int = N_PERM,
                n_boot: int = N_BOOT,
                mlp_channels: Sequence[str] = MLP_CHANNELS) -> dict:
    """Every read of one group; no IO, no model.

    *latents*: channel -> (n, d) over the group's cells; *y*: sublabel index
    per cell (0..K-1, *names* in that order); *train* / *test*: masks over
    the cells; *positions*: (n, 2) um, for the tile bootstrap of the
    held-out cells.
    """
    from discell.experiments.bootstrap import tile_bootstrap

    y = np.asarray(y, dtype=np.int64)
    k = len(names)
    rng = np.random.default_rng(seed)
    train_rows, test_rows = np.flatnonzero(train), np.flatnonzero(test)
    y_tr, y_te = y[train_rows], y[test_rows]
    probes = [("ridge", ch) for ch in latents] + [
        ("mlp", ch) for ch in mlp_channels if ch in latents]

    def fit(grader, ch, labels_tr, balanced):
        x = latents[ch]
        if grader == "ridge":
            return ridge_scores(x[train_rows], labels_tr, x[test_rows], k,
                                balanced)
        return mlp_scores(x[train_rows], labels_tr, x[test_rows], k, seed,
                          balanced)

    # balanced fits -> argmax -> BA, macro-F1, recall; natural fits -> each
    # column's one-vs-rest AUC
    preds = {f"{g}|{ch}": fit(g, ch, y_tr, True).argmax(axis=1)
             for g, ch in probes}
    ranks = {}
    for g, ch in probes:
        natural = fit(g, ch, y_tr, False)
        ranks[f"{g}|{ch}"] = [auc_ranks(natural[:, c]) for c in range(k)]
    positive = [y_te == c for c in range(k)]
    # the floor: sublabels permuted within the group, separately among the
    # training and the held-out cells; the probe refit and graded on them
    floors = {key: [] for key in preds}
    for _ in range(n_perm):
        p_tr = y_tr[rng.permutation(len(y_tr))]
        p_te = y_te[rng.permutation(len(y_te))]
        for g, ch in probes:
            floors[f"{g}|{ch}"].append(_floor_read(
                p_te, fit(g, ch, p_tr, True), fit(g, ch, p_tr, False), k))

    majority = int(np.bincount(y[np.flatnonzero(train)], minlength=k).argmax())
    base = scores(confusion(y_te, np.full(len(y_te), majority), k))

    def statistic(_values, weights):
        out = {}
        per = {}
        for key, p in preds.items():
            per[key] = scores(confusion(y_te, p, k, weights))
            per[key]["auc"] = [weighted_auc(g_, n_, positive[c], weights)
                               for c, (g_, n_) in enumerate(ranks[key])]
        for key, s in per.items():
            out[f"{key}|ba"] = s["ba"]
            out[f"{key}|macro_f1"] = s["macro_f1"]
            for c in range(k):
                out[f"{key}|recall|{c}"] = s["recall"][c]
                out[f"{key}|auc|{c}"] = s["auc"][c]
        for g, a, b in PAIRS:
            ka, kb = f"{g}|{a}", f"{g}|{b}"
            if ka not in per or kb not in per:
                continue
            out[f"{g}|{a}-{b}|ba"] = per[ka]["ba"] - per[kb]["ba"]
            for c in range(k):
                for m in ("recall", "auc"):
                    out[f"{g}|{a}-{b}|{m}|{c}"] = per[ka][m][c] - per[kb][m][c]
        return out

    boot = tile_bootstrap(positions[test_rows], None, statistic, n=n_boot,
                          seed=seed)["reads"]

    def est(key):
        return {"estimate": boot[key]["estimate"], "ci95": boot[key]["ci95"]}

    reads = {}
    for key in preds:
        fl = floors[key]
        reads[key] = {
            "ba": est(f"{key}|ba"), "macro_f1": est(f"{key}|macro_f1"),
            "recall": {names[c]: est(f"{key}|recall|{c}") for c in range(k)},
            "auc": {names[c]: est(f"{key}|auc|{c}") for c in range(k)},
            "floor": {"ba": [f["ba"] for f in fl],
                      "macro_f1": [f["macro_f1"] for f in fl],
                      "ba_mean": float(np.mean([f["ba"] for f in fl])),
                      "macro_f1_mean": float(np.mean([f["macro_f1"] for f in fl])),
                      **{m: {names[c]: [float(f[m][c]) for f in fl]
                             for c in range(k)} for m in ("recall", "auc")}}}
    differences = {}
    for g, a, b in PAIRS:
        if f"{g}|{a}" not in preds or f"{g}|{b}" not in preds:
            continue
        differences[f"{g}|{a}-{b}"] = {
            "ba": est(f"{g}|{a}-{b}|ba"),
            **{m: {names[c]: est(f"{g}|{a}-{b}|{m}|{c}") for c in range(k)}
               for m in ("recall", "auc")}}

    calls = {}
    for name in names:
        rec, exc = {}, {}
        for key, r in reads.items():
            floor = float(np.nanmean(r["floor"]["auc"][name]))
            rec[key] = bool(r["auc"][name]["ci95"][0] > floor)
            exc[key] = float(r["auc"][name]["estimate"] - floor)
        entry = {"recovered": rec, "auc_excess": exc}
        for g, a, b in PAIRS:
            ka, kb = f"{g}|{a}", f"{g}|{b}"
            if ka in reads and kb in reads:
                entry[f"{g}|{a}-{b}"] = _call(
                    rec[ka], rec[kb], exc[ka], exc[kb],
                    differences[f"{g}|{a}-{b}"]["auc"][name]["ci95"], a, b)
        calls[name] = entry

    return {"n_train": int(len(train_rows)), "n_test": int(len(test_rows)),
            "k": k, "names": list(names),
            "baseline": {"majority": names[majority], "ba": base["ba"],
                         "macro_f1": base["macro_f1"]},
            "reads": reads, "differences": differences, "calls": calls,
            "n_boot": int(n_boot), "n_perm": int(n_perm), "seed": int(seed)}


def verdict(kind: str | None, expected: str | None, call: dict) -> dict:
    """The ridge calls against the pre-registered expectation."""
    if expected is None:
        return {"expected": None, "met": None, "contradiction": None}
    if expected == T_CONFOUNDED:
        return {"expected": expected, "met": None, "contradiction": None}
    rec = call["recovered"]
    zw, zm = call["ridge|z-w"], call["ridge|z-mpsi"]
    if expected == "z":          # a state, or the cyst lining within a lineage
        met = zw == "z ≫ w"
        read_from = [DISPLAY[ch] for ch, flag in (
            ("w", rec["ridge|w"] and zw != "z ≫ w"),
            ("mpsi", rec["ridge|mpsi"] and zm != "z ≫ mpsi")) if flag]
        return {"expected": expected, "met": bool(met),
                "contradiction": (f"state read from {' and '.join(read_from)} "
                                  "(over-merging feeding the context regression)"
                                  if read_from else None)}
    met = zw == "w ≫ z" or zm == "mpsi ≫ z"
    from_z = rec["ridge|z"] and zw != "w ≫ z" and zm != "mpsi ≫ z"
    return {"expected": expected, "met": bool(met),
            "contradiction": "location read from z (niche in z)" if from_z else None}



# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------


def encode(dataset: str, run: str, device: str) -> dict:
    """Posterior means and m_psi(c, t) of the accepted checkpoint over the
    trainer's rows (training-tile seeds, then validation-tile seeds).
    ``degeneracy.load_trainer`` (``validate.load_run`` with *device* honoured
    for the resident tiles too, as ``w_deviation`` loads), so a slide too
    large for the free GPU memory can be encoded on the CPU."""
    import torch

    from discell.model.degeneracy import load_trainer

    trainer, _, _ = load_trainer(dataset, run, device)
    config, data = trainer.config, trainer.data
    train = trainer._sweep(trainer.train_batches)
    val = trainer._sweep(trainer.val_batches)
    out = {"config": config, "rows": np.concatenate([train["nodes"], val["nodes"]]),
           "n_train": len(train["nodes"]),
           "z": np.vstack([train["mu_z"], val["mu_z"]]),
           "w": np.vstack([train["mu_w"], val["mu_w"]]),
           "mpsi": np.vstack([train["prior_w"], val["prior_w"]]),
           "lineage": np.asarray(data.type_names, dtype=str)[data.t],
           "positions": np.asarray(data.positions, dtype=np.float64)[:, :2]}
    trainer.train_batches = trainer.val_batches = []
    del trainer, data
    gc.collect()
    torch.cuda.empty_cache()
    return out


def grade_run(dataset: str, run: str, device: str = "cuda",
              n_boot: int = N_BOOT, n_perm: int = N_PERM) -> dict:
    enc = encode(dataset, run, device)
    config = enc["config"]
    tables = label_tables(dataset)
    if config.label_key != tables["obs_key"]:
        raise ValueError(f"{run} was trained on {config.label_key!r}, the map "
                         f"writes {tables['obs_key']!r}")
    old_all = old_labels(dataset, config.variant, enc["lineage"], tables)
    rows = enc["rows"]
    old, lineage = old_all[rows], enc["lineage"][rows]
    is_train = np.arange(len(rows)) < enc["n_train"]
    positions = enc["positions"][rows]
    latents = {"z": enc["z"], "w": enc["w"], "mpsi": enc["mpsi"],
               "zw": np.hstack([enc["z"], enc["w"]])}
    groups = build_groups(old_all, enc["lineage"], tables,
                          EXTRA_GROUPS.get(dataset, ()))
    records = []
    for gi, group in enumerate(groups):
        member = np.isin(lineage, group["lineages"])
        subs = sorted(set(old[member].tolist()))
        kept, dropped = [], []
        for s in subs:
            n_tr = int((member & is_train & (old == s)).sum())
            n_te = int((member & ~is_train & (old == s)).sum())
            entry = {"name": s, "kind": tables["kinds"].get(s),
                     "n_train": n_tr, "n_test": n_te}
            entry["expectation"] = expectation(s, entry["kind"],
                                               group["t_confounded"])
            (kept if n_tr >= MIN_TRAIN and n_te >= MIN_TEST else dropped
             ).append(entry)
        record = {**group, "sublabels": kept, "dropped": dropped,
                  "graded": len(kept) >= 2}
        if record["graded"]:
            names = [e["name"] for e in kept]
            cells = np.flatnonzero(member & np.isin(old, names))
            y = np.searchsorted(names, old[cells])
            result = grade_group({ch: v[cells] for ch, v in latents.items()},
                                 y, names, is_train[cells], ~is_train[cells],
                                 positions[cells], seed=config.seed * 1000 + gi,
                                 n_perm=n_perm, n_boot=n_boot)
            for e in kept:
                e["verdict"] = verdict(e["kind"], e["expectation"],
                                       result["calls"][e["name"]])
            record.update(result)
            r = result["reads"]
            log.info("%s | %s (K=%d, %d held out): BA z %.3f  w %.3f  mpsi "
                     "%.3f  zw %.3f | floor z %.3f", run, group["name"],
                     len(names), result["n_test"], r["ridge|z"]["ba"]["estimate"],
                     r["ridge|w"]["ba"]["estimate"],
                     r["ridge|mpsi"]["ba"]["estimate"],
                     r["ridge|zw"]["ba"]["estimate"],
                     r["ridge|z"]["floor"]["ba_mean"])
        else:
            log.info("%s | %s: fewer than 2 sublabels with >= %d/%d cells",
                     run, group["name"], MIN_TRAIN, MIN_TEST)
        records.append(record)
    return {"dataset": dataset, "run": run, "seed": int(config.seed),
            "variant": config.variant, "label_key": config.label_key,
            "source_key": tables["source_key"],
            "n_heldout": int((~is_train).sum()),
            "settings": {"min_train": MIN_TRAIN, "min_test": MIN_TEST,
                         "max_train_per_class": MAX_TRAIN_PER_CLASS,
                         "ridge_penalty": RIDGE_PENALTY, "n_perm": n_perm,
                         "n_boot": n_boot, "dominance": DOMINANCE},
            "groups": records}


# --------------------------------------------------------------------------
# across seeds
# --------------------------------------------------------------------------


def _mmm(values) -> dict:
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not v:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"),
                "n": 0}
    return {"mean": float(np.mean(v)), "min": min(v), "max": max(v), "n": len(v)}


def _majority(calls: list[str]) -> str:
    if not calls:
        return "n/a"
    top, count = Counter(calls).most_common(1)[0]
    return top if count * 2 > len(calls) else "mixed"


def _qualified(group: dict, name: str, key: str) -> str:
    """A seed's call; ``both`` carries the direction when the paired AUC
    difference's interval excludes 0 (e.g. ``both (w > z)``)."""
    call = group["calls"][name][key]
    if call != "both":
        return call
    a, b = key.split("|")[1].split("-")
    lo, hi = group["differences"][key]["auc"][name]["ci95"]
    if lo > 0:
        return f"both ({a} > {b})"
    if hi < 0:
        return f"both ({b} > {a})"
    return call


def _majority_qualified(calls: list[str]) -> str:
    """Majority of the base calls; a ``both`` majority keeps a direction
    only when most seeds carry that same direction."""
    base = _majority([c.split(" (")[0] for c in calls])
    if base == "both":
        top, count = Counter(calls).most_common(1)[0]
        if top != "both" and top.startswith("both") and count * 2 > len(calls):
            return top
    return base


def summarise(records: list[dict]) -> dict:
    """Per group and sublabel, the seed mean [min, max] of every read, the
    call per seed and its majority, and the verdict count."""
    dataset = records[0]["dataset"]
    out = {"dataset": dataset, "runs": [r["run"] for r in records],
           "informational": dataset in INFORMATIONAL, "groups": []}
    names = []
    for r in records:
        for g in r["groups"]:
            if g["name"] not in names:
                names.append(g["name"])
    for gname in names:
        per_seed = [g for r in records for g in r["groups"] if g["name"] == gname]
        graded = [g for g in per_seed if g["graded"]]
        entry = {"name": gname, "t_confounded": per_seed[0]["t_confounded"],
                 "lineages": per_seed[0]["lineages"],
                 "n_seeds_graded": len(graded),
                 "dropped": sorted({d["name"] for g in per_seed
                                    for d in g["dropped"]}),
                 "reads": {}, "sublabels": []}
        if not graded:
            out["groups"].append(entry)
            continue
        entry["n_test"] = _mmm([g["n_test"] for g in graded])
        entry["baseline_ba"] = _mmm([g["baseline"]["ba"] for g in graded])
        entry["baseline_macro_f1"] = _mmm([g["baseline"]["macro_f1"]
                                           for g in graded])
        for key in graded[0]["reads"]:
            rs = [g["reads"][key] for g in graded]
            entry["reads"][key] = {
                m: {**_mmm([x[m]["estimate"] for x in rs]),
                    "ci95_envelope": [min(x[m]["ci95"][0] for x in rs),
                                      max(x[m]["ci95"][1] for x in rs)],
                    "floor": _mmm([x["floor"][f"{m}_mean"] for x in rs])}
                for m in ("ba", "macro_f1")}
        entry["differences_ba"] = {
            key: {**_mmm([g["differences"][key]["ba"]["estimate"]
                          for g in graded]),
                  "ci95_envelope": [
                      min(g["differences"][key]["ba"]["ci95"][0] for g in graded),
                      max(g["differences"][key]["ba"]["ci95"][1] for g in graded)]}
            for key in graded[0]["differences"]}
        subs = []
        for g in graded:
            for s in g["sublabels"]:
                if s["name"] not in [x["name"] for x in subs]:
                    subs.append(s)
        for s in sorted(subs, key=lambda e: e["name"]):
            name = s["name"]
            seeds = [g for g in graded if name in g["names"]]
            row = {"name": name, "kind": s["kind"],
                   "expectation": s["expectation"], "n_seeds": len(seeds),
                   "n_test": _mmm([next(e["n_test"] for e in g["sublabels"]
                                        if e["name"] == name) for g in seeds]),
                   "recall": {}, "auc": {}, "floor_recall": {},
                   "floor_auc": {}, "calls": {}, "verdict": {}}
            for key in seeds[0]["reads"]:
                for m in ("recall", "auc"):
                    row[m][key] = _mmm([g["reads"][key][m][name]["estimate"]
                                        for g in seeds])
                    row[m][key]["ci95_envelope"] = [
                        min(g["reads"][key][m][name]["ci95"][0] for g in seeds),
                        max(g["reads"][key][m][name]["ci95"][1] for g in seeds)]
                    row[f"floor_{m}"][key] = _mmm([np.nanmean(
                        g["reads"][key]["floor"][m][name]) for g in seeds])
            for key in [f"{g_}|{a}-{b}" for g_, a, b in PAIRS]:
                per = [_qualified(g, name, key) for g in seeds
                       if g["calls"][name].get(key) is not None]
                row["calls"][key] = {"per_seed": per,
                                     "majority": _majority_qualified(per)}
            verdicts = [next(e["verdict"] for e in g["sublabels"]
                             if e["name"] == name) for g in seeds]
            row["verdict"] = {
                "expected": s["expectation"],
                "met_seeds": sum(bool(v["met"]) for v in verdicts),
                "n_seeds": len(verdicts),
                "met": (None if s["expectation"] in (None, T_CONFOUNDED)
                        else sum(bool(v["met"]) for v in verdicts) * 2
                        > len(verdicts)),
                "contradictions": [v["contradiction"] for v in verdicts],
                "contradiction": _majority([v["contradiction"] or "none"
                                            for v in verdicts])}
            entry["sublabels"].append(row)
        out["groups"].append(entry)
    return out


def _f(v, d: int = 2) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def _mm(m: dict, d: int = 2) -> str:
    if not m or not m.get("n"):
        return "–"
    if m["n"] == 1:
        return _f(m["mean"], d)
    return f"{_f(m['mean'], d)} [{_f(m['min'], d)}, {_f(m['max'], d)}]"


def markdown(summary: dict) -> str:
    ds = summary["dataset"]
    L = [f"# Subtype recovery — {SHORT.get(ds, ds)} (`{ds}`)", "",
         f"Runs: {', '.join(summary['runs'])}. Pre-registration: devlog "
         "\"Two additions before the paper numbers (motivation, 2026-09-28)\", "
         "part B. Held-out = validation-tile seeds of each run; values are the "
         "seed mean [min, max]; CI = envelope of the per-seed 95 % tile-"
         "bootstrap intervals (conditional on the fitted probes).", ""]
    if summary["informational"]:
        L += ["**Informational dataset** (old graphclust clusters within "
              "lineage): verdicts reported, not scored.", ""]
    L += ["Channels: μ_z, μ_w, m_ψ(c,t), [μ_z, μ_w]. Grader: class-balanced "
          "multinomial ridge (primary), MLP for μ_z and μ_w. Floor: sublabels "
          f"permuted within the lineage, {N_PERM} draws. Frequency baseline: "
          "the training-majority sublabel (BA = 1/K). A channel *recovers* a "
          "sublabel when the CI of its one-vs-rest AUC is above its floor mean; "
          "`a ≫ b` when the paired AUC difference CI is above 0 and b's excess "
          f"over its floor is ≤ {DOMINANCE:g}× a's; `both (a > b)` when both "
          "recover and the difference CI excludes 0. Calls are the majority "
          "over seeds; verdicts use the ridge calls.", ""]
    L += ["## Balanced accuracy per lineage", "",
          "| lineage | K | held out | ridge μ_z | ridge μ_w | ridge m_ψ | "
          "ridge [μ_z,μ_w] | MLP μ_z | MLP μ_w | floor (ridge μ_z) | "
          "frequency | Δ(z−w) ridge |", "|---|---|---|---|---|---|---|---|---|"
          "---|---|---|"]
    for g in summary["groups"]:
        if not g["n_seeds_graded"]:
            continue
        r = g["reads"]
        tag = " †" if g["t_confounded"] else ""
        L.append(f"| {g['name']}{tag} | {len(g['sublabels'])} | "
                 f"{_mm(g['n_test'], 0)} | " +
                 " | ".join(_mm(r[k]["ba"]) for k in
                            ("ridge|z", "ridge|w", "ridge|mpsi", "ridge|zw",
                             "mlp|z", "mlp|w")) +
                 f" | {_mm(r['ridge|z']['ba']['floor'])} | "
                 f"{_mm(g['baseline_ba'])} | "
                 f"{_mm(g['differences_ba']['ridge|z-w'])} |")
    L += ["", "## Macro-F1 per lineage", "",
          "| lineage | ridge μ_z | ridge μ_w | ridge m_ψ | ridge [μ_z,μ_w] | "
          "MLP μ_z | MLP μ_w | floor (ridge μ_z) | frequency |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g in summary["groups"]:
        if not g["n_seeds_graded"]:
            continue
        r = g["reads"]
        L.append(f"| {g['name']} | " +
                 " | ".join(_mm(r[k]["macro_f1"]) for k in
                            ("ridge|z", "ridge|w", "ridge|mpsi", "ridge|zw",
                             "mlp|z", "mlp|w")) +
                 f" | {_mm(r['ridge|z']['macro_f1']['floor'])} | "
                 f"{_mm(g['baseline_macro_f1'])} |")
    L += ["", "## BA 95 % CI (envelope over seeds)", "",
          "| lineage | ridge μ_z | ridge μ_w | ridge m_ψ | ridge [μ_z,μ_w] | "
          "MLP μ_z | MLP μ_w | Δ(z−w) ridge | Δ(z−m_ψ) ridge |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g in summary["groups"]:
        if not g["n_seeds_graded"]:
            continue
        r = g["reads"]
        ci = lambda c: f"[{_f(c[0])}, {_f(c[1])}]"
        L.append(f"| {g['name']} | " +
                 " | ".join(ci(r[k]["ba"]["ci95_envelope"]) for k in
                            ("ridge|z", "ridge|w", "ridge|mpsi", "ridge|zw",
                             "mlp|z", "mlp|w")) +
                 f" | {ci(g['differences_ba']['ridge|z-w']['ci95_envelope'])} | "
                 f"{ci(g['differences_ba']['ridge|z-mpsi']['ci95_envelope'])} |")
    L += ["", "## Per sublabel: one-vs-rest AUC and the call", "",
          "AUC of the sublabel's probe score against the rest of its lineage "
          "(threshold-free; the call's basis). Floor ≈ 0.5.", "",
          "| lineage | sublabel | kind | held out | ridge μ_z | ridge μ_w | "
          "ridge m_ψ | ridge [μ_z,μ_w] | MLP μ_z | MLP μ_w | call z vs w | "
          "call z vs m_ψ | call MLP z vs w | expected | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = ("ridge|z", "ridge|w", "ridge|mpsi", "ridge|zw", "mlp|z", "mlp|w")
    for g in summary["groups"]:
        for s in g["sublabels"]:
            c = s["calls"]
            v = s["verdict"]
            if v["expected"] is None:
                verdict_s = "—"
            elif v["expected"] == T_CONFOUNDED:
                verdict_s = "untestable (t-confounded)"
            else:
                verdict_s = (f"{'met' if v['met'] else 'not met'} "
                             f"({v['met_seeds']}/{v['n_seeds']})")
                seen = Counter(c for c in v["contradictions"] if c)
                for text, count in seen.items():
                    note = f"{text} ({count}/{v['n_seeds']})"
                    verdict_s += (f"; **{note}**" if count * 2 > v["n_seeds"]
                                  else f"; {note}")
            if summary["informational"] and v["expected"] not in (None, T_CONFOUNDED):
                verdict_s = f"({verdict_s})"
            L.append(f"| {g['name']} | {s['name']} | {s['kind'] or '–'} | "
                     f"{_mm(s['n_test'], 0)} | " +
                     " | ".join(_mm(s["auc"].get(k, {})) for k in keys) +
                     f" | {c['ridge|z-w']['majority']} | "
                     f"{c['ridge|z-mpsi']['majority']} | "
                     f"{c['mlp|z-w']['majority']} | {v['expected'] or '—'} | "
                     f"{verdict_s} |")
    L += ["", "## Per sublabel: one-vs-rest recall (argmax of the probe)", "",
          "Recall of a sublabel among sublabels a channel cannot separate is "
          "decided by the argmax's tie-breaking; read it beside the AUC.", "",
          "| lineage | sublabel | ridge μ_z | ridge μ_w | ridge m_ψ | "
          "ridge [μ_z,μ_w] | MLP μ_z | MLP μ_w | floor μ_z / μ_w |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g in summary["groups"]:
        for s in g["sublabels"]:
            fl = (f"{_f(s['floor_recall']['ridge|z']['mean'])} / "
                  f"{_f(s['floor_recall']['ridge|w']['mean'])}")
            L.append(f"| {g['name']} | {s['name']} | " +
                     " | ".join(_mm(s["recall"].get(k, {})) for k in keys) +
                     f" | {fl} |")
    dropped = [(g["name"], g["dropped"]) for g in summary["groups"] if g["dropped"]]
    ungraded = [g["name"] for g in summary["groups"] if not g["n_seeds_graded"]]
    L += ["", f"Sublabels with < {MIN_TRAIN} training or < {MIN_TEST} held-out "
          "cells on some seed are left out of that seed's probe:"]
    L += [f"- {n}: {', '.join(d)}" for n, d in dropped] or ["- none"]
    if ungraded:
        L += ["", "Lineages with ≥ 2 old labels but < 2 gradable sublabels: "
              + ", ".join(ungraded)]
    if any(g["t_confounded"] for g in summary["groups"]):
        L += ["", "† cross-lineage group: the cyst lining is its own type t, an "
              "input to enc_z, enc_w and m_ψ, so every channel can read it "
              "through t; the cyst-lining expectation is untestable here."]
    return "\n".join(L) + "\n"


def figure(summary: dict, path: Path) -> None:
    """One-vs-rest AUC above the floor per sublabel, ridge μ_z / μ_w / m_ψ,
    seed mean with min-max whiskers, one panel per graded lineage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in summary["groups"] if g["n_seeds_graded"]]
    if not groups:
        return
    colours = {"ridge|z": "#2a78d6", "ridge|w": "#eb6834", "ridge|mpsi": "#1baf7a"}
    widths = [max(2, len(g["sublabels"])) for g in groups]
    ncol = min(3, len(groups))
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow),
                             squeeze=False)
    for ax, g in zip(axes.ravel(), groups):
        subs = g["sublabels"]
        x = np.arange(len(subs))
        bw = 0.26
        for j, key in enumerate(colours):
            fl = [s["floor_auc"][key]["mean"] for s in subs]
            mean = np.subtract([s["auc"][key]["mean"] for s in subs], fl)
            lo = np.subtract([s["auc"][key]["min"] for s in subs], fl)
            hi = np.subtract([s["auc"][key]["max"] for s in subs], fl)
            err = np.array([np.subtract(mean, lo), np.subtract(hi, mean)])
            ax.bar(x + (j - 1) * bw, mean, bw * 0.92, color=colours[key],
                   label=DISPLAY[key.split("|")[1]], yerr=err,
                   error_kw={"elinewidth": 0.8, "capsize": 0, "ecolor": "#52514e"})
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s['name'][:22]}\n[{s['kind'] or '?'}]"
                            for s in subs], rotation=35, ha="right", fontsize=7)
        ax.set_title(g["name"] + (" †" if g["t_confounded"] else ""), fontsize=9)
        ax.set_ylabel("AUC − floor", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color="#e5e4df", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    ds = summary["dataset"]
    fig.suptitle(f"Subtype recovery, {SHORT.get(ds, ds)}: one-vs-rest AUC "
                 "above the within-lineage floor (ridge; seed mean, min-max)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def main(argv: Sequence[str] | None = None) -> int:
    from discell import paths

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--force", action="store_true",
                        help="re-grade runs whose record exists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    root = paths.dataset(args.dataset).root
    records = []
    for run in args.run:
        path = root / "runs" / run / "subtype_recovery.json"
        if path.exists() and not args.force:
            log.info("%s: kept %s", run, path)
            records.append(json.loads(path.read_text()))
            continue
        record = _jsonable(grade_run(args.dataset, run, args.device,
                                     args.n_boot, args.n_perm))
        path.write_text(json.dumps(record, indent=1))
        log.info("%s: wrote %s", run, path)
        records.append(record)
    summary = _jsonable(summarise(records))
    out = root / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "subtype_recovery.json").write_text(json.dumps(summary, indent=1))
    (out / "subtype_recovery.md").write_text(markdown(summary))
    figure(summary, out / "subtype_recovery.png")
    log.info("wrote %s", out / "subtype_recovery.{json,md,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
