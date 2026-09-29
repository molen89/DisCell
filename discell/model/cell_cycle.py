#!/usr/bin/env python3
"""Cell-cycle scores from the counts, for the z-disentanglement read.

Tirosh marker-set scoring via scanpy -- the same thing Seurat does -- using the
``cc.genes.updated.2019`` lists intersected with the panel. Two cautions carried
from the architect's note, both encoded here rather than remembered:

* at Xenium depth (~50-300 tx/cell) the hard ``phase`` label is mostly
  "no signal detected -> G1", so quantitative reads use the **continuous**
  S/G2M scores; the phase label is kept for colouring only;
* probing cycle in a post-mitotic type just measures noise, so types are
  ranked by MKI67-positive fraction and analyses restrict to the top ranks.

The mild circularity -- scores computed from the same x that z encodes -- is
the point, not a flaw: the question is whether z *retains* that axis of x.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("discell.model.cell_cycle")

#: Seurat cc.genes.updated.2019 (Tirosh et al. 2016, updated).
S_GENES = (
    "MCM5 PCNA TYMS FEN1 MCM7 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 "
    "CENPU HELLS RFC2 POLR1B NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 "
    "MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 "
    "CLSPN POLA1 CHAF1B MRPL36 E2F8"
).split()

G2M_GENES = (
    "HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 "
    "TMPO CENPF TACC3 PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E "
    "TUBB4B GTSE1 KIF20B HJURP CDCA3 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 "
    "NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 "
    "CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA"
).split()

PHASES = ("G1", "S", "G2M")

#: Fewer panel hits than this per list and the scores are not worth attaching.
MIN_GENES_PER_LIST = 10


#: depth-conditional transform: cells per log-depth stratum (todo 3.6).
DEPTH_BIN_CELLS = 500


def depth_neutral(score: np.ndarray, totals: np.ndarray,
                  groups: np.ndarray | None = None, seed: int = 0,
                  cells_per_bin: int = DEPTH_BIN_CELLS) -> np.ndarray:
    """Normal scores of *score* within strata of log depth (todo 3.6).

    Each group (a cell type, when ``groups`` is given) is cut into strata of
    ~``cells_per_bin`` cells by log total counts and the score is replaced by
    its normal score *within the stratum*, so nothing about a cell's depth can
    be read from its target. Ties -- at Xenium depth most of a shallow type
    shares one "no marker detected" score, and that value is itself a depth
    report -- are broken at random under *seed*: the only construction that
    reaches the pre-registered Spearman bar. The price is that the tied mass
    becomes noise, which the reliability printed beside every R^2 measures.
    """
    rng = np.random.default_rng(seed)
    from scipy.stats import norm, rankdata

    log_depth = np.log(np.asarray(totals, dtype=np.float64).clip(min=1.0))
    if groups is None:
        groups = np.zeros(len(score), dtype=np.int64)
    out = np.zeros(len(score), dtype=np.float64)
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        n_bins = max(1, len(members) // cells_per_bin)
        order = np.argsort(log_depth[members], kind="stable")
        stratum = np.empty(len(members), dtype=np.int64)
        stratum[order] = np.minimum(
            (np.arange(len(members)) * n_bins) // len(members), n_bins - 1)
        for k in range(n_bins):
            rows = members[stratum == k]
            if len(rows) < 5:
                continue
            ranks = rankdata(score[rows] + rng.normal(scale=1e-9, size=len(rows)),
                             method="ordinal")
            out[rows] = norm.ppf((ranks - 0.5) / len(rows))
    return out.astype(np.float32)


def score_cell_cycle(x, gene_names: np.ndarray, t: np.ndarray | None = None,
                     depth_neutral_target: bool = False,
                     seed: int = 0) -> dict | None:
    """Tirosh scores over a counts matrix. Returns None when the panel lacks
    the markers (e.g. synthetic data).

    ``{"s_score", "g2m_score" (N,) float32, "phase" (N,) int in {0:G1,1:S,
    2:G2M}, "n_genes" (used per list)}``. Scoring runs on a scratch AnnData so
    the caller's counts stay raw.

    With *depth_neutral_target* the returned scores are the depth-conditional
    normal scores of :func:`depth_neutral`, stratified within *t* when it is
    given (pooled otherwise, which is markedly weaker -- pass *t*). The raw
    Scanpy scores stay under ``s_score_raw`` / ``g2m_score_raw`` and the phase
    label is always the raw one. ``reliability`` follows the returned target;
    ``reliability_raw`` is always the raw-score split half.
    """
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    names = [str(g) for g in gene_names]
    s_hits = [g for g in S_GENES if g in names]
    g2m_hits = [g for g in G2M_GENES if g in names]
    if len(s_hits) < MIN_GENES_PER_LIST or len(g2m_hits) < MIN_GENES_PER_LIST:
        log.info("cell cycle skipped: only %d S / %d G2M markers in the panel",
                 len(s_hits), len(g2m_hits))
        return None

    scratch = ad.AnnData(X=x.copy(), var=pd.DataFrame(index=names))
    sc.pp.normalize_total(scratch)
    sc.pp.log1p(scratch)
    sc.tl.score_genes_cell_cycle(scratch, s_genes=s_hits, g2m_genes=g2m_hits)
    phase = np.array([PHASES.index(p) for p in scratch.obs["phase"]],
                     dtype=np.int8)

    totals = np.asarray(x.sum(axis=1), dtype=np.float64).ravel()
    raw = {label: scratch.obs[key].to_numpy(dtype=np.float32)
           for label, key in (("s", "S_score"), ("g2m", "G2M_score"))}
    target = ({label: depth_neutral(v, totals, t, seed=seed)
               for label, v in raw.items()} if depth_neutral_target else raw)

    # split-half reliability: score each half-list separately and correlate.
    # This is the ceiling on the ceiling -- scores this noisy cannot be
    # predicted better than they agree with themselves. Computed on the raw
    # halves and, when the target is transformed, on the transformed halves,
    # since the transform is part of what the probe has to predict.
    reliability_raw, reliability = {}, {}
    for label, hits in (("s", s_hits), ("g2m", g2m_hits)):
        half = len(hits) // 2
        sc.tl.score_genes(scratch, hits[:half], score_name="_a")
        sc.tl.score_genes(scratch, hits[half:], score_name="_b")
        a = scratch.obs["_a"].to_numpy(dtype=np.float64)
        b = scratch.obs["_b"].to_numpy(dtype=np.float64)
        reliability_raw[label] = float(np.corrcoef(a, b)[0, 1])
        if depth_neutral_target:
            reliability[label] = float(np.corrcoef(
                depth_neutral(a, totals, t, seed=seed),
                depth_neutral(b, totals, t, seed=seed + 1))[0, 1])
        else:
            reliability[label] = reliability_raw[label]
    log.info("cycle score split-half reliability: S %.3f, G2M %.3f "
             "(raw S %.3f, G2M %.3f)", reliability["s"], reliability["g2m"],
             reliability_raw["s"], reliability_raw["g2m"])
    log.info("cell cycle: %d S + %d G2M markers; phases G1 %.1f%% / S %.1f%% "
             "/ G2M %.1f%%", len(s_hits), len(g2m_hits),
             *(100 * (phase == k).mean() for k in range(3)))
    return {"s_score": target["s"], "g2m_score": target["g2m"],
            "s_score_raw": raw["s"], "g2m_score_raw": raw["g2m"],
            "phase": phase, "reliability": reliability,
            "reliability_raw": reliability_raw,
            "depth_neutral": bool(depth_neutral_target),
            "n_genes": (len(s_hits), len(g2m_hits))}


def cycling_type_ranking(x, gene_names: np.ndarray, t: np.ndarray,
                         n_types: int) -> np.ndarray:
    """Types ordered by MKI67-positive fraction, most proliferative first."""
    names = [str(g) for g in gene_names]
    if "MKI67" not in names:
        return np.argsort(-np.bincount(t, minlength=n_types))
    positive = (x[:, names.index("MKI67")] > 0)
    positive = np.asarray(positive.todense()).ravel() if hasattr(positive, "todense") \
        else np.asarray(positive).ravel()
    fractions = np.array([positive[t == g].mean() if (t == g).any() else 0.0
                          for g in range(n_types)])
    return np.argsort(-fractions)


#: the label-independent cycling set (author's decision, devlog 2026-09-28):
#: the top decile of the S + G2M score among a slide's held-out cells. It
#: replaces the label-derived "top-4 MKI67 types" set in every battery; that
#: set survives only in ``discell/experiments/cycle_2x2.py``, for the record.
CYCLING_QUANTILE = 0.9


def cycling_set(scores: np.ndarray, held_out_mask: np.ndarray,
                q: float = CYCLING_QUANTILE,
                eligible: np.ndarray | None = None) -> np.ndarray:
    """The cells of the cycle reads, chosen from the scores alone (bool, N).

    *scores* (N, 2) S and G2M scores; a cell's cycling score is their sum.
    The candidates are the *eligible* cells inside *held_out_mask* (eligible:
    every cell by default; the reads pass every cell but Unassigned). The set
    holds exactly ``round((1 - q) * n)`` of those n candidates, the highest
    scoring (ties broken by row order), plus every eligible cell outside
    *held_out_mask* that scores at least the lowest of them -- so the ridge a
    read fits on the training cells sees the score range it is graded on.
    No label enters.
    """
    scores = np.asarray(scores, dtype=np.float64)
    total = scores.sum(axis=1) if scores.ndim == 2 else scores
    held = np.asarray(held_out_mask, dtype=bool)
    eligible = (np.ones(len(total), dtype=bool) if eligible is None
                else np.asarray(eligible, dtype=bool))
    candidates = np.flatnonzero(held & eligible)
    k = int(round((1.0 - q) * len(candidates)))
    out = np.zeros(len(total), dtype=bool)
    if k == 0:
        return out
    top = candidates[np.argsort(-total[candidates], kind="stable")[:k]]
    out[top] = True
    out |= ~held & eligible & (total >= total[top[-1]])
    return out


def cycle_eligible(t: np.ndarray, type_names) -> np.ndarray:
    """Cells a cycle read may take as targets: every cell but Unassigned
    (author's decision 2026-09-28) -- the shared evaluation mask."""
    from discell.model import eval_mask as EM

    return EM.metric_target_mask(np.asarray(t), type_names)


def slide_cycling_set(data) -> np.ndarray:
    """:func:`cycling_set` of one assembled slide (node-indexed): held out =
    its validation tiles, Unassigned excluded. Every read of that slide --
    the trainer's, the battery's for each method, validate's -- restricts
    this one mask to the rows it scores."""
    held = np.zeros(data.n_cells, dtype=bool)
    for tile in data.val_tiles:
        held[tile] = True
    scores = np.stack([data.cycle["s_score"], data.cycle["g2m_score"]], axis=1)
    return cycling_set(scores, held,
                       eligible=cycle_eligible(data.t, data.type_names))


def cycle_q90_reads(designs: dict, t: np.ndarray, scores: np.ndarray,
                    train: np.ndarray, test: np.ndarray, cells: np.ndarray,
                    seed: int = 0) -> dict:
    """``metrics.cycle_r2`` of every design (name -> (N, d)) on the cycling
    set *cells*, centred per label within the set, plus the set's size."""
    from discell.model import metrics as M

    out = {"q": CYCLING_QUANTILE, "n_train": int((cells & train).sum()),
           "n_heldout": int((cells & test).sum())}
    for name, design in designs.items():
        out[name] = M.cycle_r2(design, t, scores, None, train, test,
                               seed=seed, cells=cells)
    return out


def trainer_cycle_q90(data, rows: np.ndarray, z: np.ndarray, w: np.ndarray,
                      train: np.ndarray, seed: int = 0) -> dict:
    """The in-trainer cycle reads on the top-decile set (``Trainer.evaluate``
    and its re-read, ``discell/experiments/cycle_reread.py``): z, w, the
    50-PC linear reference on counts and the log-depth baseline, on *rows*
    of *data* (training rows where *train*, held out elsewhere)."""
    cyc = data.cycle
    return cycle_q90_reads(
        {"z": z, "w": w, "linear_ref": cyc["x_pcs"][rows],
         "lbaseline": np.log(data.totals[rows].clip(min=1.0)
                             )[:, None].astype(np.float64)},
        data.t[rows], np.stack([cyc["s_score"], cyc["g2m_score"]],
                               axis=1)[rows],
        train, ~train, slide_cycling_set(data)[rows], seed=seed)


def q90_keys(block: dict | None) -> dict:
    """The flat ``metrics.json`` keys of a :func:`cycle_q90_reads` block."""
    pooled = lambda name: ((block or {}).get(name) or {}).get("r2_pooled")
    return {"cycle_r2_z_q90": pooled("z"), "cycle_r2_w_q90": pooled("w"),
            "cycle_linear_q90": pooled("linear_ref")}


def expression_pcs(x, n_components: int = 50, seed: int = 0) -> np.ndarray:
    """Top PCs of log-normalised counts -- the probe ceiling's design matrix.

    The identical ridge probe run from these against the same scores is the
    ceiling on any latent: z cannot retain more cycle than the counts carry.
    """
    import scanpy as sc
    from sklearn.decomposition import TruncatedSVD

    import anndata as ad
    import pandas as pd

    scratch = ad.AnnData(X=x.copy(), var=pd.DataFrame(
        index=[str(k) for k in range(x.shape[1])]))
    sc.pp.normalize_total(scratch)
    sc.pp.log1p(scratch)
    svd = TruncatedSVD(n_components, random_state=seed)
    return svd.fit_transform(scratch.X).astype(np.float32)
