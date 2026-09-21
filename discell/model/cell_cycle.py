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
