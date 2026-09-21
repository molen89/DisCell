#!/usr/bin/env python3
"""The counterfactual transport check (doc 08 section 7).

Model-predicted between-niche expression shifts must match empirically
observed shifts on held-out tiles. This validates the surviving
counterfactual claim -- average effects under ``do(c')`` at composition
level (per-cell counterfactuals are out of scope, doc-10 cut).

Per (type, niche pair), on the log-rate scale:

- program channel:  ``<B_g, mean m_psi(B,t) - mean m_psi(A,t)>``
- leak channel:     ``log`` of the kappa-mixed mean rates minus the
  program-only version -- i.e. the shift the foreign influx adds on top
- observed:         difference of depth-normalised mean expression of
  held-out type-t cells between the niches (same log transform)

References per panel: zero-prediction null, program-only, leak-only. The
claim needs the full model to beat both single-channel versions. The
per-gene program-vs-leak split of each observed niche difference is the
headline figure. Guards: composition-overlap check per pair (else
extrapolation -- flagged, not reported), near-zero genes excluded, counts
reported.

A companion row holds Phi at the receiver type's mean in both niches
(``*_phi_fixed``): the program channel then answers to neighbour
composition alone, the part of the context an intervention can set; its
share of the counterfactual's R² is the interventionable share (the
neighbour-dose experiment, devlog 2026-09-16, motivates the row).

Every panel also carries a ``noise_ceiling``: the split-half reliability
of the OBSERVED shift itself, Spearman-Brown corrected. No predictor can
score above it, so it says which panels are worth reading at all and what
fraction of the reachable signal the counterfactual actually takes.

Niches come from ``--niche-source``: ``kmeans`` on neighbour composition
(doc-08 §4.1's data-defined fallback, the default) or ``tumour-band`` --
nested bands of the kNN-smoothed tumour fraction, which is the field
``validate.landmark_inventory`` already builds for the interface
landmark. Band niches are ordered and overlapping, so unlike k-means
niches they can populate the supported (interpolation) tier.

Usage::

    python -m discell.model.transport --dataset <id> --run <run>
    python -m discell.model.transport --dataset <id> --run <run> \
        --niche-source tumour-band
    python -m discell.model.transport --dataset <id> --run <run> \
        --read distribution
    python -m discell.model.transport --dataset <id> --sweep-tag sweep3 \
        --seeds 0 --sweep-out transport_kappa_sensitivity_v2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

import numpy as np

from discell.model.validate import (collect_latents, load_run, niche_labels)

log = logging.getLogger("discell.model.transport")

MIN_CELLS = 500
MIN_RATE = 1e-5          #: genes below this mean rate in the type are excluded
N_PAIRS = 4              #: most composition-distinct niche pairs
N_SPLITS = 4             #: cell splits averaged for the per-panel noise ceiling
TRUST_CEILING = 0.5      #: a panel is trusted if its observed shift is this reliable
TRUST_GENES = 100        #: ... and scored on at least this many genes
TOP_GENES = 50            #: size of the predicted-up / observed-up gene lists
TUMOUR_BANDS = (0.1, 0.3, 0.5, 0.7, 0.9)  #: cuts on the smoothed tumour field
EPS = 1e-8


def collect_channels(trainer, group: np.ndarray, n_groups: int,
                     phi_by_type: np.ndarray | None = None) -> dict:
    """Per-GROUP mean prior mean, decontaminated rate and foreign influx.

    *group* is one integer per cell (``-1`` = belongs to no panel); the
    forward pass accumulates sums into ``n_groups`` rows and divides at
    the end. Only group means are ever used downstream, and the per-cell
    rate matrices are ``n_cells x n_genes`` float32 -- 23 GB on the 1.16M
    cell FF slide, which is what killed the earlier run (issues, watch
    2026-09-21). Accumulating instead keeps the instrument at
    ``n_groups x n_genes``.

    With *phi_by_type* ``(K, phi_dim)`` every node's Phi is replaced by
    its type's mean before the forward pass, so ``m_psi`` answers to
    neighbour composition (and t) only. Only the prior mean is collected
    then: rho and rho_bar are the real neighbours' and come from the
    plain pass.
    """
    import torch

    keys = ("prior_w", "c") if phi_by_type is not None else (
        "prior_w", "c", "rho", "rho_bar")
    acc: dict = {}
    counts = None
    with torch.no_grad():
        for batch in trainer.train_batches + trainer.val_batches:
            kwargs = trainer._forward_kwargs(batch)
            if phi_by_type is not None:
                kwargs["phi"] = torch.as_tensor(
                    phi_by_type, device=kwargs["phi"].device,
                    dtype=kwargs["phi"].dtype)[kwargs["t"]]
            fwd = trainer.model(**kwargs, kappa=trainer.config.kappa,
                                sample=False)
            n = batch["n_seeds"]
            device = fwd.prior_mean_w.device
            gid = torch.as_tensor(np.asarray(group)[batch["nodes"][:n]],
                                  device=device, dtype=torch.long)
            take = gid >= 0
            gid = gid[take]
            values = {"prior_w": fwd.prior_mean_w[:n][take],
                      "c": fwd.c[:n][take],
                      "rho": fwd.log_rho[:n].exp()[take],
                      "rho_bar": fwd.rho_bar[take]}
            if counts is None:
                counts = torch.zeros(n_groups, dtype=torch.float64,
                                     device=device)
                for k in keys:
                    acc[k] = torch.zeros(n_groups, values[k].shape[1],
                                         dtype=torch.float64, device=device)
            counts.index_add_(0, gid, torch.ones_like(gid,
                                                      dtype=torch.float64))
            for k in keys:
                acc[k].index_add_(0, gid, values[k].double())
    if counts is None:                       # no batches at all
        return {k: np.zeros((n_groups, 0)) for k in keys} | {
            "n": np.zeros(n_groups)}
    denom = counts.clamp(min=1.0)[:, None]
    out = {k: (v / denom).cpu().numpy() for k, v in acc.items()}
    out["n"] = counts.cpu().numpy()
    return out


def score_shift(prediction: np.ndarray, observed: np.ndarray) -> dict:
    """Held-out R², calibration slope and correlation of a predicted
    per-gene log-rate shift against the observed one.

    Both sides are centred: the softmax normaliser and depth enter as
    per-panel constants and must not be charged to the model. R² is
    against the zero-prediction null (no fitted slope), so a prediction of
    the right direction but wrong size is penalised -- the slope says
    which."""
    p = prediction - prediction.mean()
    o = observed - observed.mean()
    slope = float(np.polyfit(p, o, 1)[0]) if p.std() > 1e-9 else float("nan")
    ss = 1.0 - ((o - p) ** 2).sum() / max((o ** 2).sum(), 1e-12)
    return {"r2": float(ss), "slope": slope,
            "corr": float(np.corrcoef(p, o)[0, 1])}


def top_gene_overlap(prediction: np.ndarray, observed: np.ndarray,
                     k: int = TOP_GENES) -> dict:
    """Of the *k* genes predicted to rise most, how many are in the observed
    top *k* -- the mean read without R².

    Both sides are centred first, as in ``score_shift``: the per-panel
    constant would otherwise pick the same k genes on both sides for a
    trivial reason. ``chance`` is ``k / G``, the overlap a random ranking
    gives, and is reported beside the count always -- an overlap is only
    readable against it."""
    k = int(min(k, len(observed)))
    if k == 0:
        return {"k": 0, "overlap": 0, "fraction": float("nan"),
                "chance": float("nan")}
    p = prediction - prediction.mean()
    o = observed - observed.mean()
    top_p = set(np.argsort(p)[-k:].tolist())
    top_o = set(np.argsort(o)[-k:].tolist())
    return {"k": k, "overlap": int(len(top_p & top_o)),
            "fraction": float(len(top_p & top_o) / k),
            "chance": float(k / len(observed))}


def noise_ceiling(x_rate, rows_a: np.ndarray, rows_b: np.ndarray,
                  keep: np.ndarray, rng: np.random.Generator,
                  n_splits: int = N_SPLITS) -> float:
    """The largest R² any predictor of this panel's observed shift can get.

    The observed shift is a noisy estimate: it is a difference of two
    sample means over a few hundred held-out cells. Split the cells of
    each niche in half and form the shift twice; the correlation of the
    two halves is ``v / (v + 2σ²)`` for signal variance ``v`` and
    per-half noise ``2σ²``, and its Spearman-Brown correction
    ``2r/(1+r) = v / (v + σ²)`` is exactly the reliability of the
    full-sample shift -- which is the ceiling of the R² the scorer
    reports. Half-sample rates are floored at ``MIN_RATE`` (the gene mask
    guarantees the full sample clears it), which can only push the
    estimate down, so the number is a conservative ceiling."""
    half_shifts = []
    for _ in range(n_splits):
        halves = []
        for rows in (rows_a, rows_b):
            order = rng.permutation(len(rows))
            halves.append([rows[order[:len(rows) // 2]],
                           rows[order[len(rows) // 2:]]])
        for side in (0, 1):
            rate_a = np.asarray(x_rate[halves[0][side]].mean(axis=0)).ravel()
            rate_b = np.asarray(x_rate[halves[1][side]].mean(axis=0)).ravel()
            half_shifts.append(
                np.log(np.maximum(rate_b[keep], MIN_RATE))
                - np.log(np.maximum(rate_a[keep], MIN_RATE)))
    corrs = [float(np.corrcoef(half_shifts[2 * i], half_shifts[2 * i + 1])[0, 1])
             for i in range(n_splits)]
    r = float(np.mean(corrs))
    return float(2 * r / (1 + r)) if r > 0 else 0.0


def tumour_band_labels(data, k: int = 50) -> np.ndarray:
    """Ordered niches along the tumour/stroma axis; -1 for isolated cells.

    The field is the kNN-smoothed tumour fraction (k = 50) that
    ``validate.landmark_inventory`` uses for the interface landmark --
    built from ``t`` and positions only, the same data status as ``y``,
    and equally free of the latents. Cutting it at ``TUMOUR_BANDS`` gives
    deep stroma → stroma → interface → rim → core: niches that are
    *nested along one axis*, so neighbouring bands genuinely share
    composition support. k-means niches cannot do this -- distinct
    clusters are composition-disjoint by construction, which is why the
    supported tier is empty with them (handover limitation 9)."""
    from scipy.spatial import cKDTree

    names = [str(n) for n in data.type_names]
    tumour_types = [g for g, name in enumerate(names)
                    if "Tumor Cells" in name or "Malignant" in name]
    if not tumour_types:
        # cluster-labelled slide: no type name says "tumour", so there is
        # no annotation axis to band. The caller must use composition
        # niches and say so -- silently returning one band would look like
        # a supported tier that is really a single niche.
        raise ValueError(
            "no tumour-annotated type on this slide ({}); annotation niches "
            "are undefined here -- use --niche-source kmeans and report the "
            "niches as composition-defined".format(
                ", ".join(names[:4]) + " ..."))
    is_tumour = np.isin(data.t, tumour_types).astype(np.float64)
    k = min(k, data.graph.n_cells - 1)
    neighbours = cKDTree(data.positions).query(data.positions, k=k + 1)[1]
    field = is_tumour[neighbours].mean(axis=1)
    labels = np.digitize(field, TUMOUR_BANDS).astype(np.int64)
    labels[data.graph.degrees == 0] = -1
    return labels


def pick_pairs(labels: np.ndarray, y: np.ndarray, connected: np.ndarray,
               n_pairs: int = N_PAIRS) -> list[tuple[int, int]]:
    """Every niche pair, most composition-distinct first.

    All pairs are evaluated and the per-(pair, type) overlap guard decides
    reportability -- selecting only the most distinct pairs up front is
    self-defeating (they are exactly the ones the guard flags)."""
    ks = [k for k in np.unique(labels) if k >= 0]
    centroids = {k: y[connected & (labels == k)].mean(axis=0) for k in ks}
    scored = sorted(((np.linalg.norm(centroids[a] - centroids[b]), a, b)
                     for i, a in enumerate(ks) for b in ks[i + 1:]),
                    reverse=True)
    return [(a, b) for _, a, b in scored]


def tier_summary(tier: list[dict]) -> dict:
    """The headline numbers of one tier of panels.

    Means over panels of every channel's R², the median calibration slope,
    the count of panels where the counterfactual beats both single
    channels (doc-08 7.4's requirement), the mean noise ceiling with the
    share of it the counterfactual takes, and the interventionable share
    -- the Phi-fixed counterfactual's R² as a fraction of the real-Phi
    one, i.e. how much of the transported shift answers to the part of the
    context an intervention could actually set."""
    summary = {key: float(np.mean([p[key]["r2"] for p in tier]))
               for key in ("counterfactual", "full", "program_only",
                           "leak_only", "counterfactual_phi_fixed",
                           "program_phi_fixed")}
    summary["n_panels"] = len(tier)
    summary["n_trusted"] = int(sum(p["trusted"] for p in tier))
    summary["noise_ceiling"] = float(np.mean([p["noise_ceiling"]
                                              for p in tier]))
    # undefined when there is no reachable signal at all -- a ratio against
    # a zero ceiling is not a percentage, it is a division by noise
    summary["counterfactual_of_ceiling"] = float(
        summary["counterfactual"] / summary["noise_ceiling"]
        ) if summary["noise_ceiling"] >= 0.05 else float("nan")
    summary["interventionable_share"] = float(
        summary["counterfactual_phi_fixed"]
        / max(summary["counterfactual"], 1e-12))
    summary["top_gene_overlap"] = float(np.mean(
        [p["top_gene_overlap"]["fraction"] for p in tier
         if p.get("top_gene_overlap")])) if all(
             p.get("top_gene_overlap") for p in tier) else float("nan")
    summary["top_gene_chance"] = float(np.mean(
        [p["top_gene_overlap"]["chance"] for p in tier
         if p.get("top_gene_overlap")])) if all(
             p.get("top_gene_overlap") for p in tier) else float("nan")
    summary["median_slope"] = float(np.median(
        [p["counterfactual"]["slope"] for p in tier]))
    summary["median_slope_phi_fixed"] = float(np.median(
        [p["counterfactual_phi_fixed"]["slope"] for p in tier]))
    # doc 7.4: the claim rides on the counterfactual total
    summary["full_beats_both"] = int(sum(
        p["counterfactual"]["r2"] > max(p["program_only"]["r2"],
                                        p["leak_only"]["r2"])
        for p in tier))
    summary["selection_share"] = float(summary["full"]
                                       - summary["counterfactual"])
    return summary


def summary_figure(results: dict, path) -> None:
    """The section-7 headline in one figure: tier bars + the per-panel
    program-vs-leak attribution map (how much of each niche difference is
    biology vs contamination).

    The bars come from the *headline tier* -- the one with the most panels,
    which is extrapolation under composition niches and supported under
    annotation niches. The map carries every panel, the two tiers marked
    apart, so no tier is silently invisible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = results["panels"]
    if not panels:
        return
    tier_name = max(("supported", "extrapolation"),
                    key=lambda n: results.get("summary", {}).get(
                        n, {}).get("n_panels", 0))
    summary = results.get("summary", {}).get(tier_name, {})
    if not summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    keys = ("counterfactual", "counterfactual_phi_fixed", "full",
            "program_only", "leak_only")
    axes[0].bar(range(5), [summary.get(k, float("nan")) for k in keys],
                color=("#2c7fb8", "#9ecae1", "#7fb8d4", "#41ab5d", "#c9662a"),
                width=0.6)
    axes[0].set_xticks(range(5), ("counter-\nfactual", "counterfactual\n"
                                  "Φ fixed", "model\naccount",
                                  "program\nonly", "leak\nonly"),
                       fontsize=8)
    ceiling = summary.get("noise_ceiling")
    if ceiling:
        axes[0].axhline(ceiling, color="0.3", lw=1.0, ls="--")
        axes[0].text(4.4, ceiling, "noise ceiling", fontsize=7, ha="right",
                     va="bottom", color="0.3")
    axes[0].set_ylabel(f"mean held-out R² ({tier_name} tier)", fontsize=8)
    axes[0].set_title(
        f"both channels required: counterfactual beats both in "
        f"{summary.get('full_beats_both', '?')}/{summary.get('n_panels', '?')}"
        f" panels\nmedian calibration slope "
        f"{summary.get('median_slope', float('nan')):.2f}"
        f" | interventionable share "
        f"{summary.get('interventionable_share', float('nan')):.2f}",
        fontsize=9)

    full = np.array([p["counterfactual"]["r2"] for p in panels])
    for flag, marker, label in ((False, "o", "supported"),
                                (True, "^", "extrapolation")):
        take = np.array([p["overlap_flag"] == flag for p in panels])
        if not take.any():
            continue
        prog = np.array([p["program_only"]["r2"] for p in panels])[take]
        leak = np.array([p["leak_only"]["r2"] for p in panels])[take]
        sc = axes[1].scatter(prog.clip(min=0), leak.clip(min=0), marker=marker,
                             s=20 + 300 * full[take].clip(min=0),
                             c=full[take], cmap="viridis", alpha=0.7,
                             vmin=float(full.min()), vmax=float(full.max()),
                             edgecolors="0.4", linewidths=0.4, label=label)
    axes[1].legend(fontsize=7, loc="lower right", title="tier",
                   title_fontsize=7)
    lim = max(1e-3, max(max(p["program_only"]["r2"] for p in panels),
                        max(p["leak_only"]["r2"] for p in panels))) * 1.1
    axes[1].plot([0, lim], [0, lim], color="0.6", lw=0.8, ls="--")
    axes[1].set_xlabel("program-channel R² (biology)", fontsize=8)
    axes[1].set_ylabel("leak-channel R² (contamination)", fontsize=8)
    axes[1].set_title("per (type, niche-pair): what explains the observed "
                      "shift\nabove the line = contamination-dominated",
                      fontsize=9)
    for p in sorted(panels, key=lambda q: -q["counterfactual"]["r2"])[:4]:
        axes[1].annotate(p["type"][:14],
                         (max(p["program_only"]["r2"], 0),
                          max(p["leak_only"]["r2"], 0)), fontsize=6)
    plt.colorbar(sc, ax=axes[1], fraction=0.04,
                 label="counterfactual R²")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def transport_check(args: argparse.Namespace) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, data, trainer, run_dir, b_matrix = load_run(
        args.dataset, args.run, args.device)
    latents = collect_latents(trainer, data)
    connected = data.graph.degrees > 0
    source = getattr(args, "niche_source", "kmeans")
    labels = (tumour_band_labels(data) if source == "tumour-band"
              else niche_labels(data, args.niches, config.seed))
    rng = np.random.default_rng(config.seed)
    held_out = latents["fold"] == 0            # spatial-block held-out tiles
    kappa = config.kappa

    # model quantities are read from the TRAINING folds, per (niche, type);
    # the forward pass accumulates those group means directly (memory)
    n_types = len(data.p_t)
    n_niches = int(labels.max()) + 1
    n_groups = n_niches * n_types
    model_rows = connected & ~held_out & (labels >= 0)
    group = np.full(data.graph.n_cells, -1, dtype=np.int64)
    group[model_rows] = labels[model_rows] * n_types + data.t[model_rows]
    channels = collect_channels(trainer, group, n_groups)
    # Phi at the receiver type's mean (over connected cells, as in the
    # neighbour-dose instrument) in BOTH niches: m_psi's composition-only
    # response
    phi_by_type = np.stack([data.phi[connected & (data.t == g)].mean(axis=0)
                            for g in range(n_types)])
    prior_w_phi_fixed = collect_channels(
        trainer, group, n_groups, phi_by_type)["prior_w"]

    x_rate = data.x.multiply(
        1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()

    pairs = pick_pairs(labels, data.graph.y, connected)
    names = [str(n) for n in data.type_names]
    results: dict = {"run": args.run, "kappa": kappa,
                     "niche_source": source, "panels": []}
    curves: dict = {}          # per panel: (counterfactual, observed) per gene
    out_dir = run_dir / "transport"
    out_dir.mkdir(exist_ok=True)
    stem = "transport" if source == "kmeans" else f"transport_{source}"

    for niche_a, niche_b in pairs:
        for g in range(len(names)):
            members = {}
            ok = True
            for niche, side in ((niche_a, "A"), (niche_b, "B")):
                train_rows = np.flatnonzero(connected & ~held_out
                                            & (data.t == g)
                                            & (labels == niche))
                test_rows = np.flatnonzero(connected & held_out
                                           & (data.t == g)
                                           & (labels == niche))
                if len(train_rows) < MIN_CELLS or len(test_rows) < MIN_CELLS // 5:
                    ok = False
                    break
                members[side] = (train_rows, test_rows)
            if not ok:
                continue

            # composition-overlap guard: prior means are extrapolation when
            # the niches share no composition support for this type. The
            # comparison is 1-D along the gap direction -- summed per-dim
            # spreads overstate the relevant support
            y_a = data.graph.y[members["A"][0]]
            y_b = data.graph.y[members["B"][0]]
            gap_vec = y_b.mean(0) - y_a.mean(0)
            gap = np.linalg.norm(gap_vec)
            unit = gap_vec / max(gap, 1e-12)
            spread = 0.5 * (float((y_a @ unit).std())
                            + float((y_b @ unit).std()))
            overlap_flag = bool(gap > 3 * spread)

            gid_a, gid_b = niche_a * n_types + g, niche_b * n_types + g

            def mean_channels(gid):
                return (channels["rho"][gid], channels["rho_bar"][gid],
                        channels["prior_w"][gid])

            rho_a, bar_a, mpsi_a = mean_channels(gid_a)
            rho_b, bar_b, mpsi_b = mean_channels(gid_b)

            # predicted shifts, log-rate scale, from TRAINING folds. Same
            # type on both sides, so the per-type offset gauge of w
            # (issues V12) cancels here; rho is gauge-invariant by
            # construction (a(z) + B w is what the gauge leaves fixed)
            program = b_matrix @ (mpsi_b - mpsi_a)
            program_phi_fixed = b_matrix @ (prior_w_phi_fixed[gid_b]
                                            - prior_w_phi_fixed[gid_a])
            full = (np.log((1 - kappa) * rho_b + kappa * bar_b + EPS)
                    - np.log((1 - kappa) * rho_a + kappa * bar_a + EPS))
            leak_only = (np.log((1 - kappa) * rho_a + kappa * bar_b + EPS)
                         - np.log((1 - kappa) * rho_a + kappa * bar_a + EPS))

            # observed shift on HELD-OUT tiles, depth-normalised
            obs_a = np.asarray(x_rate[members["A"][1]].mean(axis=0)).ravel()
            obs_b = np.asarray(x_rate[members["B"][1]].mean(axis=0)).ravel()
            keep = (obs_a > MIN_RATE) & (obs_b > MIN_RATE)
            observed = np.log(obs_b[keep] + EPS) - np.log(obs_a[keep] + EPS)

            def score(prediction):
                return score_shift(prediction[keep], observed)

            panel = {"pair": (int(niche_a), int(niche_b)),
                     "type": names[g],
                     "n_train": [len(members[s][0]) for s in "AB"],
                     "n_test": [len(members[s][1]) for s in "AB"],
                     "n_genes": int(keep.sum()),
                     "overlap_flag": overlap_flag,
                     # the doc-7.2 counterfactual: context response + new
                     # neighbours' influx, the cell's own z held fixed
                     "counterfactual": score(program + leak_only),
                     # the model's account of the actual populations -- also
                     # lets the type's intrinsic mix differ across niches;
                     # its excess over the counterfactual measures selection
                     "full": score(full),
                     "program_only": score(program),
                     "leak_only": score(leak_only),
                     # the same counterfactual with Phi held at the type
                     # mean in both niches: composition-only response +
                     # the new neighbours' influx
                     "counterfactual_phi_fixed": score(
                         program_phi_fixed + leak_only),
                     "program_phi_fixed": score(program_phi_fixed),
                     # split-half reliability of the observed shift: no
                     # predictor can beat it, so it says whether a low R²
                     # is the model's fault or the panel's
                     "noise_ceiling": noise_ceiling(
                         x_rate, members["A"][1], members["B"][1], keep, rng),
                     # the mean read without R²: of the 50 genes predicted
                     # to rise most, how many the data also puts in its top
                     # 50 (chance = 50 / n_genes, always beside it)
                     "top_gene_overlap": top_gene_overlap(
                         (program + leak_only)[keep], observed),
                     "zero_null_r2": 0.0}
            # trust criterion (one line a reader can apply): the observed
            # shift must be a reliable target at all (>= TRUST_CEILING of
            # its variance is signal) and the panel must have enough genes
            # for the score to mean anything. Untrusted panels are carried
            # in the JSON and excluded from the headline means.
            panel["trusted"] = bool(
                panel["noise_ceiling"] >= TRUST_CEILING
                and panel["n_genes"] >= TRUST_GENES)
            panel["counterfactual_of_ceiling"] = float(
                panel["counterfactual"]["r2"] / panel["noise_ceiling"]
                ) if panel["noise_ceiling"] >= 0.05 else float("nan")
            # how much of the counterfactual survives when Phi is frozen:
            # the panel's interventionable share
            panel["interventionable_share"] = float(
                panel["counterfactual_phi_fixed"]["r2"]
                / panel["counterfactual"]["r2"]) if (
                    panel["counterfactual"]["r2"] > 1e-6) else float("nan")
            results["panels"].append(panel)
            curves[(niche_a, niche_b, g)] = (
                (program + leak_only)[keep], observed)

    # per-gene calibration of the best-predicted panels -- the object scored
    # (the counterfactual), both sides centred as in the score, tier named
    ranked = sorted(results["panels"],
                    key=lambda p: -p["counterfactual"]["r2"])
    for panel in ranked[:args.figures]:
        niche_a, niche_b = panel["pair"]
        g = names.index(panel["type"])
        pred, obs = curves[(niche_a, niche_b, g)]
        pred, obs = pred - pred.mean(), obs - obs.mean()
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.scatter(pred, obs, s=2, alpha=0.4, rasterized=True)
        lims = np.percentile(np.concatenate([pred, obs]), [1, 99])
        ax.plot(lims, lims, color="0.4", lw=0.8, ls="--", label="1:1")
        ax.plot(lims, panel["counterfactual"]["slope"] * lims, color="#c9662a",
                lw=0.8, label=f"fit, slope {panel['counterfactual']['slope']:.2f}")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel("counterfactual log-rate shift (program + leak, z fixed)")
        ax.set_ylabel("observed shift (held-out tiles)")
        ax.set_title(f"{panel['type'][:26]} | niche {niche_a}->{niche_b} "
                     f"({'extrapolation' if panel['overlap_flag'] else 'supported'})\n"
                     f"counterfactual R² {panel['counterfactual']['r2']:.2f} "
                     f"(prog {panel['program_only']['r2']:.2f}, "
                     f"leak {panel['leak_only']['r2']:.2f}, "
                     f"Φ fixed {panel['counterfactual_phi_fixed']['r2']:.2f})"
                     f" | ceiling {panel['noise_ceiling']:.2f}"
                     f" ({'trusted' if panel['trusted'] else 'untrusted'})",
                     fontsize=7)
        fig.tight_layout()
        prefix = "" if source == "kmeans" else f"{source}_"
        fig.savefig(out_dir / f"{prefix}pair{niche_a}-{niche_b}_type{g}.png",
                    dpi=130)
        plt.close(fig)

    # two tiers: with data-defined (k-means) niches, distinct pairs are
    # composition-disjoint BY CONSTRUCTION, so the interpolation tier is
    # structurally near-empty and the informative regime is extrapolation --
    # named as such, never blended (the guard's purpose)
    results["summary"] = {}
    tiers = {"supported": [p for p in results["panels"]
                           if not p["overlap_flag"]],
             "extrapolation": [p for p in results["panels"]
                               if p["overlap_flag"]]}
    # the same two tiers restricted to panels whose observed shift is a
    # reliable target at all (the trust criterion) -- reported beside, never
    # instead of, the all-panel numbers
    for tier_name in ("supported", "extrapolation"):
        tiers[f"{tier_name}_trusted"] = [p for p in tiers[tier_name]
                                         if p["trusted"]]
    for name, tier in tiers.items():
        if not tier:
            continue
        results["summary"][name] = tier_summary(tier)
        summary = results["summary"][name]
        log.info("transport [%s]: %d panels | mean R² counterfactual %.3f "
                 "(Φ fixed %.3f -> interventionable share %.2f), model "
                 "account %.3f, program %.3f, leak %.3f | slope %.2f | "
                 "counterfactual beats both %d/%d",
                 name, summary["n_panels"], summary["counterfactual"],
                 summary["counterfactual_phi_fixed"],
                 summary["interventionable_share"], summary["full"],
                 summary["program_only"], summary["leak_only"],
                 summary["median_slope"], summary["full_beats_both"],
                 summary["n_panels"])
        log.info("transport [%s]: noise ceiling %.3f (%d/%d panels trusted) "
                 "-> the counterfactual takes %.0f%% of what is reachable",
                 name, summary["noise_ceiling"], summary["n_trusted"],
                 summary["n_panels"],
                 100 * summary["counterfactual_of_ceiling"])

    summary_figure(results, out_dir / f"{stem}_summary.png")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(results, indent=2, default=float))
    log.info("wrote %s", out_dir / f"{stem}.json")
    return results


# ---------------------------------------------------------------------------
# The distribution-level read (devlog "Transport at the distribution level",
# 2026-09-21). The mean read above scores a difference of niche MEANS. This
# one moves a population: source cells of type t are decoded at the target
# niche's context and leak source, and the transported cloud is compared with
# the cells that actually live there, as distributions.
# ---------------------------------------------------------------------------

SIZE_CAP = 2000          #: cells per side after size matching
N_BOOT = 200             #: paired bootstrap draws for the transported-untransported CI


def _hellinger(p: np.ndarray) -> np.ndarray:
    """Square-root map of a probability vector: Euclidean distance on it is
    the Hellinger distance (up to sqrt 2), which is the natural geometry for
    compositions and keeps the kernel from being dominated by the few
    highest-rate genes."""
    return np.sqrt(np.clip(np.asarray(p, dtype=np.float64), 0.0, None))


def _kernel_parts(x, y, gamma):
    import torch
    return torch.exp(-gamma * torch.cdist(x, y).pow(2))


def _mmd2(kxx, kyy, kxy) -> float:
    """Unbiased MMD^2: the within-sample terms drop their diagonals, so the
    estimator is zero in expectation when the two samples come from the same
    law (that is what makes the floor meaningful)."""
    m, n = kxx.shape[0], kyy.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    sxx = (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
    syy = (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
    return float(sxx + syy - 2.0 * kxy.mean())


def _energy(x, y) -> float:
    """Energy distance ``2E|x-y| - E|x-x'| - E|y-y'|`` on the same map --
    a kernel-free second opinion, reported beside MMD^2 and never instead."""
    import torch
    dxy = torch.cdist(x, y).mean()
    dxx = torch.cdist(x, x).mean()
    dyy = torch.cdist(y, y).mean()
    return float(2 * dxy - dxx - dyy)


def distribution_scores(p_transported: np.ndarray, p_untransported: np.ndarray,
                        p_target: np.ndarray, p_source_obs: np.ndarray,
                        rng: np.random.Generator, device: str = "cpu",
                        n_boot: int = N_BOOT, cap: int = SIZE_CAP,
                        count_depths: np.ndarray | None = None) -> dict:
    """One distribution panel, given the four clouds on the simplex.

    *p_transported* and *p_untransported* are the SAME source cells decoded
    at the target niche's context/leak and at their own; *p_source_obs* is
    those cells' raw normalised counts; *p_target* is the held-out cells that
    actually live in the target niche.

    Sizes are matched by subsampling every cloud to ``min(|S|, |T|, cap)``
    -- MMD^2's sampling behaviour depends on the sample sizes, so a source
    with more cells than the target would otherwise score differently for a
    reason that has nothing to do with transport. (This is also exactly what
    the pooled leave-one-out version buys: a source that was smaller than
    the target stops being the binding side.)

    With *count_depths* (the target cells' library sizes) every MODEL cloud
    -- transported, untransported and the type-mean point -- is first turned
    into a multinomial draw at a depth drawn from that vector. This is the
    **count-matched companion**, and it exists because the read as
    pre-registered compares smooth predicted rates against raw multinomial
    compositions: on a 5k panel at a few hundred counts per cell the shot
    noise of the target is far larger than any difference between two rate
    clouds, so the rate variant's MMD^2 is dominated by a term that is the
    same for every predictor and its gap closed is compressed towards zero.
    Both variants are reported; neither replaces the other.

    References, all against the same target cells: **floor** (two halves of
    the target -- pure sampling noise), **untransported**, **type-mean** (a
    single point, the target's mean composition, replicated: a predictor
    that gets the mean right and the spread wrong) and **observed-source**
    (the raw niche difference, no model). The headline is **gap closed** =
    (untransported - transported) / (untransported - floor).
    """
    import torch

    n_s = len(p_transported)
    n_t = len(p_target)
    m = int(min(n_s, n_t, cap))
    if m < 20:
        return {"n": m, "insufficient": True}
    idx_s = rng.choice(n_s, m, replace=False)
    idx_t = rng.choice(n_t, m, replace=False)

    p_transported = np.asarray(p_transported)[idx_s]
    p_untransported = np.asarray(p_untransported)[idx_s]
    p_source_obs = np.asarray(p_source_obs)[idx_s]
    p_target = np.asarray(p_target)[idx_t]
    type_mean = p_target.mean(axis=0)
    if count_depths is not None:
        depths = np.asarray(count_depths, dtype=np.int64)
        depths = np.maximum(depths[rng.integers(0, len(depths), m)], 1)

        def as_counts(p):
            # float64 throughout and the last cell absorbing the residue:
            # numpy's multinomial rejects pvals whose leading entries sum
            # to more than 1 by a single ulp, which a 5k-gene softmax can
            p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
            p = p / p.sum(axis=-1, keepdims=True)
            p[..., -1] = np.clip(1.0 - p[..., :-1].sum(axis=-1), 0.0, 1.0)
            draw = rng.multinomial(depths, p).astype(np.float64)
            return draw / draw.sum(axis=1, keepdims=True).clip(min=1.0)
        p_transported = as_counts(p_transported)
        p_untransported = as_counts(p_untransported)
        type_mean_cloud = as_counts(np.tile(type_mean, (m, 1)))
    else:
        type_mean_cloud = np.tile(type_mean, (m, 1))
    idx_s = np.arange(m)
    idx_t = np.arange(m)

    def to_t(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)

    y = to_t(_hellinger(p_target))
    xt = to_t(_hellinger(p_transported))
    xu = to_t(_hellinger(p_untransported))
    xo = to_t(_hellinger(p_source_obs))
    # the target's MEAN composition, replicated (and, in the count-matched
    # companion, resampled at matched depths): the degenerate predictor
    # bar (3) exists to rule out
    xm = to_t(_hellinger(type_mean_cloud))

    with torch.no_grad():
        # bandwidth: median pairwise distance WITHIN the target, so the
        # kernel's scale is set by the population being matched, never by
        # the predictions being judged
        d_yy = torch.cdist(y, y)
        off = d_yy[~torch.eye(m, dtype=torch.bool, device=d_yy.device)]
        sigma = float(off.median())
        gamma = 1.0 / (2.0 * max(sigma, 1e-8) ** 2)

        kyy = torch.exp(-gamma * d_yy.pow(2))
        parts = {}
        for name, x in (("transported", xt), ("untransported", xu),
                        ("type_mean", xm), ("observed_source", xo)):
            kxx = _kernel_parts(x, x, gamma)
            kxy = _kernel_parts(x, y, gamma)
            parts[name] = (kxx, kxy)
        mmd = {name: _mmd2(kxx, kyy, kxy)
               for name, (kxx, kxy) in parts.items()}
        # floor: two random halves of the TARGET against each other
        half = m // 2
        perm = torch.as_tensor(rng.permutation(m), device=y.device)
        y1, y2 = y[perm[:half]], y[perm[half:2 * half]]
        mmd["floor"] = _mmd2(_kernel_parts(y1, y1, gamma),
                             _kernel_parts(y2, y2, gamma),
                             _kernel_parts(y1, y2, gamma))
        energy = {name: _energy(x, y) for name, x in
                  (("transported", xt), ("untransported", xu),
                   ("type_mean", xm), ("observed_source", xo))}
        energy["floor"] = _energy(y1, y2)

        # paired bootstrap of transported - untransported over CELLS. The
        # two estimates share the target sample and the source cells, so the
        # target-target terms cancel exactly in the difference and only the
        # source-source and cross terms are resampled.
        kt_xx, kt_xy = parts["transported"]
        ku_xx, ku_xy = parts["untransported"]
        diffs = []
        for _ in range(n_boot):
            i = torch.as_tensor(rng.integers(0, m, m), device=y.device)
            j = torch.as_tensor(rng.integers(0, m, m), device=y.device)

            def half_mmd(kxx, kxy):
                sub = kxx[i][:, i]
                sxx = (sub.sum() - sub.diagonal().sum()) / (m * (m - 1))
                return sxx - 2.0 * kxy[i][:, j].mean()
            diffs.append(float(half_mmd(kt_xx, kt_xy)
                               - half_mmd(ku_xx, ku_xy)))
    lo, hi = (float(np.percentile(diffs, 2.5)),
              float(np.percentile(diffs, 97.5)))

    denom = mmd["untransported"] - mmd["floor"]
    gap = (float(np.clip((mmd["untransported"] - mmd["transported"]) / denom,
                         -1.0, 1.0)) if abs(denom) > 1e-12 else float("nan"))
    gap_type_mean = (float(np.clip(
        (mmd["untransported"] - mmd["type_mean"]) / denom, -1.0, 1.0))
        if abs(denom) > 1e-12 else float("nan"))
    return {"n": m, "insufficient": False, "bandwidth": sigma,
            "count_matched": count_depths is not None,
            "mmd2": mmd, "energy": energy, "gap_closed": gap,
            "gap_closed_type_mean": gap_type_mean,
            "improves": bool(mmd["transported"] < mmd["untransported"]),
            "ci_excludes_zero": bool(hi < 0.0),
            "beats_type_mean": bool(mmd["transported"] < mmd["type_mean"]),
            "delta": float(mmd["transported"] - mmd["untransported"]),
            "ci": [lo, hi], "ci_width": float(hi - lo)}


def distribution_summary(panels: list[dict], key: str = "scores") -> dict:
    """The four pre-registered bars over a list of distribution panels.

    *key* picks the variant: ``scores`` is the read exactly as
    pre-registered (predicted rates against raw compositions),
    ``scores_count_matched`` the companion that puts both sides on the same
    sampling geometry."""
    ok = [p for p in panels
          if key in p and not p[key].get("insufficient")]
    if not ok:
        return {"n_panels": 0}
    gaps = np.array([p[key]["gap_closed"] for p in ok], dtype=float)
    return {
        "n_panels": len(ok),
        # bar (1)
        "n_improved": int(sum(p[key]["improves"] for p in ok)),
        "n_ci_excludes_zero": int(sum(p[key]["ci_excludes_zero"]
                                      for p in ok)),
        # bar (2)
        "median_gap_closed": float(np.nanmedian(gaps)),
        "mean_gap_closed": float(np.nanmean(gaps)),
        "q25_gap_closed": float(np.nanpercentile(gaps, 25)),
        "q75_gap_closed": float(np.nanpercentile(gaps, 75)),
        "median_ci_width": float(np.median(
            [p[key]["ci_width"] for p in ok])),
        # bar (3)
        "n_beats_type_mean": int(sum(p[key]["beats_type_mean"]
                                     for p in ok)),
        "median_gap_closed_type_mean": float(np.nanmedian(
            [p[key]["gap_closed_type_mean"] for p in ok])),
        "median_mmd2_transported": float(np.median(
            [p[key]["mmd2"]["transported"] for p in ok])),
        "median_mmd2_untransported": float(np.median(
            [p[key]["mmd2"]["untransported"] for p in ok])),
        "median_mmd2_floor": float(np.median(
            [p[key]["mmd2"]["floor"] for p in ok])),
        "median_mmd2_observed_source": float(np.median(
            [p[key]["mmd2"]["observed_source"] for p in ok])),
        "median_energy_transported": float(np.median(
            [p[key]["energy"]["transported"] for p in ok])),
        "median_energy_untransported": float(np.median(
            [p[key]["energy"]["untransported"] for p in ok])),
    }


def _decode_cells(trainer, mu_z: np.ndarray, w: np.ndarray,
                  rho_bar: np.ndarray, kappa: float,
                  chunk: int = 4096) -> np.ndarray:
    """``p`` for cells kept at their OWN z, given *w* and *rho_bar* per row.

    Exactly the decode path of ``Trainer._decode_seeds`` -- the prior head's
    w through B, the intrinsic programme through a(z), the leak mixed in
    probability space at the model's own kappa -- with the context and the
    influx substituted rather than read off the cell's real neighbours.
    """
    import torch

    from discell.model.equations import leakage_mix

    device = next(trainer.model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(mu_z), chunk):
            sl = slice(start, start + chunk)
            z = torch.as_tensor(mu_z[sl], dtype=torch.float32, device=device)
            ww = torch.as_tensor(np.asarray(w)[sl], dtype=torch.float32,
                                 device=device)
            bar = torch.as_tensor(np.asarray(rho_bar)[sl], dtype=torch.float32,
                                  device=device)
            rho = trainer.model.log_rho(z, ww).exp()
            out.append(leakage_mix(rho, bar, kappa).exp().cpu().numpy())
    return np.vstack(out)


def _niche_w(trainer, c_mean: np.ndarray, type_index: int,
             n_types: int) -> np.ndarray:
    """``m_psi`` at a niche's mean context for one type: the context the
    transported cells are given."""
    import torch

    device = next(trainer.model.parameters()).device
    with torch.no_grad():
        onehot = torch.zeros(1, n_types, device=device)
        onehot[0, type_index] = 1.0
        c = torch.as_tensor(np.asarray(c_mean)[None, :], dtype=torch.float32,
                            device=device)
        return trainer.model.prior_w(
            torch.cat([c, onehot], dim=-1)).cpu().numpy()[0]


def distribution_check(args: argparse.Namespace,
                       twins: bool = False) -> dict:
    """The distribution read, both versions, on one run.

    Pairwise: source = held-out cells of type t in niche A, target = niche B
    (the same panels the mean read pairs). Leave-one-niche-out: source =
    every held-out cell of type t that is NOT in A, transported into A. If z
    is intrinsic, cells from every context should land on A's population,
    and the pooled source stops the source side being the binding one for
    the size match.
    """
    from scipy.stats import spearmanr

    config, data, trainer, run_dir, b_matrix = load_run(
        args.dataset, args.run, args.device)
    latents = collect_latents(trainer, data)
    mu_z = latents["mu_z"]
    connected = data.graph.degrees > 0
    source_kind = getattr(args, "niche_source", "kmeans")
    labels = (tumour_band_labels(data) if source_kind == "tumour-band"
              else niche_labels(data, args.niches, config.seed))
    rng = np.random.default_rng(config.seed)
    held_out = latents["fold"] == 0
    kappa = config.kappa
    n_types = len(data.p_t)
    n_niches = int(labels.max()) + 1
    n_groups = n_niches * n_types

    model_rows = connected & ~held_out & (labels >= 0)
    group = np.full(data.graph.n_cells, -1, dtype=np.int64)
    group[model_rows] = labels[model_rows] * n_types + data.t[model_rows]
    channels = collect_channels(trainer, group, n_groups)

    x_rate = data.x.multiply(
        1.0 / data.totals.clip(min=1.0)[:, None]).tocsr()
    names = [str(n) for n in data.type_names]
    device = str(next(trainer.model.parameters()).device)
    n_hvg = int(getattr(args, "hvg", 0) or 0)
    hvg = (hvg_mask(data, np.flatnonzero(connected & ~held_out),
                    n_hvg, seed=config.seed) if n_hvg else None)

    def on_hvg(a):
        return restrict_renormalise(a, hvg)

    # which (niche, type) cells exist in enough numbers on both sides
    train_rows: dict = {}
    test_rows: dict = {}
    for k in range(n_niches):
        for g in range(n_types):
            tr = np.flatnonzero(connected & ~held_out & (data.t == g)
                                & (labels == k))
            te = np.flatnonzero(connected & held_out & (data.t == g)
                                & (labels == k))
            if len(tr) >= MIN_CELLS and len(te) >= MIN_CELLS // 5:
                train_rows[(k, g)] = tr
                test_rows[(k, g)] = te

    # the niche context/leak each group supplies, once
    w_of: dict = {}
    for (k, g) in train_rows:
        gid = k * n_types + g
        w_of[(k, g)] = (_niche_w(trainer, channels["c"][gid], g, n_types),
                        channels["rho_bar"][gid])

    def score_panel(src_rows: np.ndarray, src_niche: np.ndarray,
                    target: tuple[int, int]) -> dict:
        """Decode *src_rows* into *target* (and at their own niches) and
        score. ``src_niche`` gives each source cell's own niche."""
        keep = rng.permutation(len(src_rows))[:SIZE_CAP]
        rows, own = src_rows[keep], src_niche[keep]
        g = target[1]
        w_a, bar_a = w_of[target]
        w_to = np.tile(w_a, (len(rows), 1))
        bar_to = np.tile(bar_a, (len(rows), 1))
        w_own = np.stack([w_of[(int(k), g)][0] for k in own])
        bar_own = np.stack([w_of[(int(k), g)][1] for k in own])
        p_trans = _decode_cells(trainer, mu_z[rows], w_to, bar_to, kappa)
        p_unt = _decode_cells(trainer, mu_z[rows], w_own, bar_own, kappa)
        tgt = test_rows[target]
        tgt = tgt[rng.permutation(len(tgt))[:SIZE_CAP]]
        p_target = np.asarray(x_rate[tgt].todense())
        p_source_obs = np.asarray(x_rate[rows].todense())
        depths = np.asarray(data.totals)[tgt]
        out = {
            # the count-level read, unchanged (same rng stream as before, so
            # the numbers already read stay identical)
            "scores": distribution_scores(p_trans, p_unt, p_target,
                                          p_source_obs, rng, device=device,
                                          n_boot=args.boot),
            "scores_count_matched": distribution_scores(
                p_trans, p_unt, p_target, p_source_obs,
                np.random.default_rng(config.seed), device=device,
                n_boot=args.boot, count_depths=depths)}
        # -- second round. Dedicated generators, so nothing above shifts.
        seed2 = config.seed + 1
        p_tgt_model = _decode_cells(
            trainer, mu_z[tgt], np.tile(w_a, (len(tgt), 1)),
            np.tile(bar_a, (len(tgt), 1)), kappa)
        out["scores_model"] = distribution_scores(
            p_trans, p_unt, p_tgt_model, p_source_obs,
            np.random.default_rng(seed2), device=device, n_boot=args.boot)
        if hvg is not None:
            out["scores_model_hvg"] = distribution_scores(
                on_hvg(p_trans), on_hvg(p_unt), on_hvg(p_tgt_model),
                on_hvg(p_source_obs), np.random.default_rng(seed2 + 1),
                device=device, n_boot=args.boot)
        if twins:
            out["twins"] = twin_scores(
                p_trans, p_unt, p_tgt_model, mu_z[rows], mu_z[tgt],
                np.random.default_rng(seed2 + 2), n_boot=args.boot)
            if hvg is not None:
                out["twins_hvg"] = twin_scores(
                    on_hvg(p_trans), on_hvg(p_unt), on_hvg(p_tgt_model),
                    mu_z[rows], mu_z[tgt],
                    np.random.default_rng(seed2 + 2), n_boot=args.boot)
        return out

    results: dict = {"run": args.run, "kappa": kappa,
                     "niche_source": source_kind,
                     "pairwise": [], "leave_one_out": []}

    pairs = pick_pairs(labels, data.graph.y, connected)
    for niche_a, niche_b in pairs:
        for g in range(n_types):
            if (niche_a, g) not in train_rows or (niche_b, g) not in train_rows:
                continue
            src = test_rows[(niche_a, g)]
            panel = score_panel(src, np.full(len(src), niche_a),
                                (niche_b, g))
            scores, matched = panel["scores"], panel["scores_count_matched"]
            results["pairwise"].append(
                {"pair": [int(niche_a), int(niche_b)], "type": names[g],
                 "target": int(niche_b), "n_source": int(len(src)),
                 "n_target": int(len(test_rows[(niche_b, g)])), **panel})
            log.info("distribution pairwise %s %d->%d: gap closed %.2f "
                     "(count-matched %.2f; mmd2 %.4g vs %.4g, floor %.4g)",
                     names[g][:18], niche_a, niche_b,
                     scores.get("gap_closed", float("nan")),
                     matched.get("gap_closed", float("nan")),
                     scores.get("mmd2", {}).get("transported", float("nan")),
                     scores.get("mmd2", {}).get("untransported", float("nan")),
                     scores.get("mmd2", {}).get("floor", float("nan")))
            log.info("    model-vs-model gap closed %.2f (type-mean %.2f)"
                     "%s", panel["scores_model"].get("gap_closed", float("nan")),
                     panel["scores_model"].get("gap_closed_type_mean",
                                               float("nan")),
                     ("" if "twins" not in panel else
                      " | twins gap %.2f margin %.2f" % (
                          panel["twins"].get("gap_closed", float("nan")),
                          panel["twins"].get("twin_margin", float("nan")))))

    for (niche_a, g) in sorted(train_rows):
        others = [k for k in range(n_niches)
                  if k != niche_a and (k, g) in train_rows]
        if not others:
            continue
        src = np.concatenate([test_rows[(k, g)] for k in others])
        own = np.concatenate([np.full(len(test_rows[(k, g)]), k)
                              for k in others])
        panel = score_panel(src, own, (niche_a, g))
        scores, matched = panel["scores"], panel["scores_count_matched"]
        results["leave_one_out"].append(
            {"target": int(niche_a), "type": names[g],
             "n_source": int(len(src)), "n_source_niches": len(others),
             "n_target": int(len(test_rows[(niche_a, g)])), **panel})
        log.info("distribution leave-one-out %s ->%d: gap closed %.2f "
                 "(count-matched %.2f; %d source niches)", names[g][:18],
                 niche_a, scores.get("gap_closed", float("nan")),
                 matched.get("gap_closed", float("nan")), len(others))

    results["summary"] = {
        "pairwise": distribution_summary(results["pairwise"]),
        "leave_one_out": distribution_summary(results["leave_one_out"])}
    results["summary_count_matched"] = {
        "pairwise": distribution_summary(results["pairwise"],
                                         "scores_count_matched"),
        "leave_one_out": distribution_summary(results["leave_one_out"],
                                              "scores_count_matched")}
    for suffix in ("", "_hvg"):
        key = f"scores_model{suffix}"
        summary = {"pairwise": distribution_summary(results["pairwise"], key),
                   "leave_one_out": distribution_summary(
                       results["leave_one_out"], key)}
        if summary["pairwise"].get("n_panels"):
            results[f"summary_model{suffix}"] = summary

    # bar (2), second half: pooling should TIGHTEN the CI on the same target
    loo_by = {(p["target"], p["type"]): p for p in results["leave_one_out"]}

    def pooling(key: str) -> dict:
        paired = []
        for p in results["pairwise"]:
            q = loo_by.get((p["target"], p["type"]))
            if q is None or p[key].get("insufficient") or \
                    q[key].get("insufficient"):
                continue
            paired.append({"target": p["target"], "type": p["type"],
                           "source": p["pair"][0],
                           "pairwise_ci_width": p[key]["ci_width"],
                           "loo_ci_width": q[key]["ci_width"],
                           "pairwise_gap": p[key]["gap_closed"],
                           "loo_gap": q[key]["gap_closed"]})
        return {
            "n_compared": len(paired),
            "n_loo_tighter": int(sum(d["loo_ci_width"] < d["pairwise_ci_width"]
                                     for d in paired)),
            "median_pairwise_ci_width": float(np.median(
                [d["pairwise_ci_width"] for d in paired]))
            if paired else float("nan"),
            "median_loo_ci_width": float(np.median(
                [d["loo_ci_width"] for d in paired]))
            if paired else float("nan"),
            "median_gap_drop": float(np.median(
                [d["pairwise_gap"] - d["loo_gap"] for d in paired]))
            if paired else float("nan"),
            "panels": paired}

    results["pooling"] = pooling("scores")
    results["pooling_count_matched"] = pooling("scores_count_matched")
    for suffix in ("", "_hvg"):
        if f"summary_model{suffix}" in results:
            results[f"pooling_model{suffix}"] = pooling(
                f"scores_model{suffix}")

    # bar (4): do the two instruments call the same panels good?
    stem = "transport" if source_kind == "kmeans" else f"transport_{source_kind}"
    mean_path = run_dir / "transport" / f"{stem}.json"

    def agreement(key: str) -> dict:
        if not mean_path.exists():
            return {"available": False}
        mean = json.loads(mean_path.read_text())
        by_key = {(tuple(p["pair"]), p["type"]): p for p in mean["panels"]}
        xs, ys = [], []
        for p in results["pairwise"]:
            q = by_key.get((tuple(p["pair"]), p["type"]))
            if q is None or p[key].get("insufficient"):
                continue
            xs.append(p[key]["gap_closed"])
            ys.append(q["counterfactual"]["r2"])
        if len(xs) < 5:
            return {"available": False}
        rho, pval = spearmanr(xs, ys)
        return {"available": True, "n": len(xs), "spearman": float(rho),
                "p_value": float(pval)}

    results["agreement_with_mean_read"] = agreement("scores")
    results["agreement_with_mean_read_count_matched"] = agreement(
        "scores_count_matched")
    for suffix in ("", "_hvg"):
        if f"summary_model{suffix}" in results:
            results[f"agreement_with_mean_read_model{suffix}"] = agreement(
                f"scores_model{suffix}")

    out_dir = run_dir / "transport"
    out_dir.mkdir(exist_ok=True)
    if twins:
        twin_res = {"run": args.run, "kappa": kappa,
                    "niche_source": source_kind,
                    "n_hvg": int(hvg.sum()) if hvg is not None else 0,
                    "pairwise": [], "leave_one_out": []}
        for version in ("pairwise", "leave_one_out"):
            for p in results[version]:
                row = {k: v for k, v in p.items()
                       if not k.startswith("scores")}
                twin_res[version].append(row)
        twin_res["summary"] = {
            "pairwise": twin_summary(twin_res["pairwise"]),
            "leave_one_out": twin_summary(twin_res["leave_one_out"])}
        if hvg is not None:
            twin_res["summary_hvg"] = {
                "pairwise": twin_summary(twin_res["pairwise"], "twins_hvg"),
                "leave_one_out": twin_summary(twin_res["leave_one_out"],
                                              "twins_hvg")}
        twin_figure(twin_res, out_dir / f"{stem}_twins.png")
        (out_dir / f"{stem}_twins.json").write_text(
            json.dumps(twin_res, indent=2, default=float))
        log.info("wrote %s", out_dir / f"{stem}_twins.json")
        for name, sm in twin_res["summary"].items():
            if sm.get("n_panels"):
                log.info("twins [%s]: %d panels | gap closed>0 in %d "
                         "(median %.2f) | margin>0 in %d (median %.3f)",
                         name, sm["n_panels"], sm["n_gap_closed_positive"],
                         sm["median_gap_closed"],
                         sm["n_twin_margin_positive"],
                         sm["median_twin_margin"])
    distribution_figure(results, out_dir / f"{stem}_distribution.png")
    (out_dir / f"{stem}_distribution.json").write_text(
        json.dumps(results, indent=2, default=float))
    log.info("wrote %s", out_dir / f"{stem}_distribution.json")
    return results


def distribution_figure(results: dict, path) -> None:
    """Per-panel gap closed, pairwise vs leave-one-out, with the floor (gap
    closed = 1 by construction) and the type-mean reference marked.

    Both variants are drawn: solid = the count-matched companion (the two
    clouds on the same sampling geometry), faded = the read exactly as
    pre-registered (predicted rates against raw compositions)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    def series(version, key):
        return [p for p in results[version]
                if key in p and not p[key].get("insufficient")]

    drew = False
    for key, alpha, tag in (("scores_count_matched", 1.0, "count-matched"),
                            ("scores", 0.35, "as pre-registered")):
        for version, colour, label in (
                ("pairwise", "#2c7fb8", "pairwise A→B"),
                ("leave_one_out", "#c9662a", "leave-one-niche-out")):
            rows = series(version, key)
            if not rows:
                continue
            drew = True
            gaps = np.sort([p[key]["gap_closed"] for p in rows])
            axes[0].plot(np.linspace(0, 1, len(gaps)), gaps, lw=1.4,
                         color=colour, alpha=alpha,
                         label=f"{label}, {tag} (n={len(gaps)})")
    if not drew:
        plt.close(fig)
        return
    tm = [p["scores_count_matched"]["gap_closed_type_mean"]
          for v in ("pairwise", "leave_one_out")
          for p in series(v, "scores_count_matched")]
    tm_med = float(np.nanmedian(tm)) if tm else float("nan")
    if tm:
        axes[0].axhline(tm_med, color="0.45", lw=1.6, ls=":",
                        label=f"type-mean predictor (median {tm_med:.2f})")
    axes[0].axhline(1.0, color="0.3", lw=1.0, ls="--",
                    label="floor (sampling noise)")
    axes[0].axhline(0.0, color="0.7", lw=0.8)
    axes[0].set_xlabel("panels, sorted", fontsize=8)
    axes[0].set_ylabel("fraction of the niche gap closed", fontsize=8)
    axes[0].legend(fontsize=6, loc="upper left")
    axes[0].set_title("how much of the distance to the target population\n"
                      "the transport removes (type-mean reference "
                      f"{tm_med:.2f}, count-matched)", fontsize=9)

    key = "scores_count_matched"
    lo_by = {(p["target"], p["type"]): p
             for p in series("leave_one_out", key)}
    xs, ys = [], []
    for p in series("pairwise", key):
        q = lo_by.get((p["target"], p["type"]))
        if q is not None:
            xs.append(p[key]["gap_closed"])
            ys.append(q[key]["gap_closed"])
    if xs:
        axes[1].scatter(xs, ys, s=18, alpha=0.7, c="#41ab5d",
                        edgecolors="0.4", linewidths=0.4)
        lim = [min(min(xs), min(ys)) - 0.05, max(max(xs), max(ys)) + 0.05]
        axes[1].plot(lim, lim, color="0.6", lw=0.8, ls="--")
        axes[1].set_xlim(lim)
        axes[1].set_ylim(lim)
    pool = results.get("pooling_count_matched", {})
    axes[1].set_xlabel("gap closed, pairwise source (count-matched)",
                       fontsize=8)
    axes[1].set_ylabel("gap closed, pooled leave-one-out source", fontsize=8)
    axes[1].set_title(
        "does pooling every other context hurt?\n"
        "tighter CI in {}/{} panels | median gap drop {:.2f}".format(
            pool.get("n_loo_tighter", "?"), pool.get("n_compared", "?"),
            pool.get("median_gap_drop", float("nan"))), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Second round (devlog "Transport at the distribution level, second round",
# 2026-09-21): the model-vs-model target cloud, the matched-twin per-cell
# read, and the highly-variable-gene companion for both.
# ---------------------------------------------------------------------------


def hvg_mask(data, rows: np.ndarray, n_top: int = 1000,
             cap: int = 200_000, seed: int = 0) -> np.ndarray:
    """Boolean gene mask: Scanpy's ``seurat`` HVGs on the TRAINING cells.

    Counts are depth-normalised and log1p-ed first, exactly the input the
    flavour expects. Low-count genes only add noise to a 5k-gene Hellinger
    map; restricting the probability vectors to these genes (and
    renormalising) is the companion read, never the replacement."""
    import anndata
    import scanpy as sc

    rows = np.asarray(rows)
    if len(rows) > cap:
        rows = np.random.default_rng(seed).choice(rows, cap, replace=False)
    ad = anndata.AnnData(X=data.x[rows].astype(np.float32))
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    n_top = int(min(n_top, ad.n_vars))
    sc.pp.highly_variable_genes(ad, flavor="seurat", n_top_genes=n_top)
    mask = np.asarray(ad.var["highly_variable"].to_numpy(), dtype=bool)
    log.info("HVG companion: %d/%d genes on %d training cells",
             int(mask.sum()), len(mask), len(rows))
    return mask


def restrict_renormalise(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Probability vectors kept on *mask*'s genes and renormalised."""
    q = np.asarray(p, dtype=np.float64)[:, mask]
    return q / q.sum(axis=1, keepdims=True).clip(min=1e-12)


def _hellinger_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Hellinger distance between two arrays of probability
    vectors (0 = identical, 1 = disjoint support)."""
    return (np.linalg.norm(_hellinger(a) - _hellinger(b), axis=1)
            / np.sqrt(2.0))


def twin_scores(p_trans: np.ndarray, p_unt: np.ndarray,
                p_target_model: np.ndarray, z_src: np.ndarray,
                z_tgt: np.ndarray, rng: np.random.Generator,
                n_boot: int = N_BOOT) -> dict:
    """The matched-twin read for one panel, cell to cell.

    For every target cell, its nearest source cell in mu_z (Euclidean, same
    type, from the source niche(s)) is its *twin*. The twin transported into
    the target niche is compared with the target cell's OWN decoded vector
    by Hellinger distance. Three references per cell: the same twin
    *untransported* (decoded in its own niche -- the niche difference before
    correction), a *random* same-type source cell transported (does matching
    on z buy anything at all), and the *floor*, the target cell's z-nearest
    OTHER target cell, both decoded in the target niche (the distance two
    genuinely similar cells of the target population are apart under the
    model).

    ``gap_closed`` = (untransported - transported) / (untransported - floor)
    on the medians; ``twin_margin`` = (random - transported) / random. Paired
    bootstraps over cells for (transported - untransported) and
    (transported - random)."""
    n_t, n_s = len(z_tgt), len(z_src)
    if n_t < 20 or n_s < 20:
        return {"n": int(min(n_t, n_s)), "insufficient": True}
    d = np.linalg.norm(np.asarray(z_tgt, dtype=np.float64)[:, None, :]
                       - np.asarray(z_src, dtype=np.float64)[None, :, :],
                       axis=-1)
    twin = np.argmin(d, axis=1)
    rand = rng.integers(0, n_s, n_t)
    dtt = np.linalg.norm(np.asarray(z_tgt, dtype=np.float64)[:, None, :]
                         - np.asarray(z_tgt, dtype=np.float64)[None, :, :],
                         axis=-1)
    np.fill_diagonal(dtt, np.inf)
    other = np.argmin(dtt, axis=1)

    d_tr = _hellinger_rows(p_trans[twin], p_target_model)
    d_un = _hellinger_rows(p_unt[twin], p_target_model)
    d_rd = _hellinger_rows(p_trans[rand], p_target_model)
    d_fl = _hellinger_rows(p_target_model[other], p_target_model)

    med = {"transported": float(np.median(d_tr)),
           "untransported": float(np.median(d_un)),
           "random": float(np.median(d_rd)),
           "floor": float(np.median(d_fl))}
    denom = med["untransported"] - med["floor"]
    gap = (float(np.clip((med["untransported"] - med["transported"]) / denom,
                         -1.0, 1.0)) if abs(denom) > 1e-12 else float("nan"))
    margin = (float((med["random"] - med["transported"]) / med["random"])
              if med["random"] > 1e-12 else float("nan"))

    b_un, b_rd = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n_t, n_t)
        b_un.append(float(np.median(d_tr[i] - d_un[i])))
        b_rd.append(float(np.median(d_tr[i] - d_rd[i])))
    ci_un = [float(np.percentile(b_un, 2.5)), float(np.percentile(b_un, 97.5))]
    ci_rd = [float(np.percentile(b_rd, 2.5)), float(np.percentile(b_rd, 97.5))]
    return {
        "n": int(n_t), "n_source": int(n_s), "insufficient": False,
        "median": med,
        "gap_closed": gap, "twin_margin": margin,
        "delta_untransported": float(np.median(d_tr - d_un)),
        "delta_random": float(np.median(d_tr - d_rd)),
        "ci_vs_untransported": ci_un, "ci_vs_random": ci_rd,
        "beats_untransported": bool(med["transported"]
                                    < med["untransported"]),
        "beats_random": bool(med["transported"] < med["random"]),
        "ci_untransported_excludes_zero": bool(ci_un[1] < 0.0),
        "ci_random_excludes_zero": bool(ci_rd[1] < 0.0),
        "median_z_twin_distance": float(np.median(d[np.arange(n_t), twin])),
    }


def twin_summary(panels: list[dict], key: str = "twins") -> dict:
    """Summary across twin panels: the two pre-registered clauses."""
    ok = [p[key] for p in panels
          if key in p and not p[key].get("insufficient")]
    if not ok:
        return {"n_panels": 0}
    gaps = np.array([s["gap_closed"] for s in ok], dtype=float)
    marg = np.array([s["twin_margin"] for s in ok], dtype=float)
    return {
        "n_panels": len(ok),
        "n_gap_closed_positive": int(np.sum(gaps > 0)),
        "frac_gap_closed_positive": float(np.mean(gaps > 0)),
        "median_gap_closed": float(np.nanmedian(gaps)),
        "q25_gap_closed": float(np.nanpercentile(gaps, 25)),
        "q75_gap_closed": float(np.nanpercentile(gaps, 75)),
        "n_twin_margin_positive": int(np.sum(marg > 0)),
        "frac_twin_margin_positive": float(np.mean(marg > 0)),
        "median_twin_margin": float(np.nanmedian(marg)),
        "n_ci_vs_untransported_excludes_zero": int(sum(
            s["ci_untransported_excludes_zero"] for s in ok)),
        "n_ci_vs_random_excludes_zero": int(sum(
            s["ci_random_excludes_zero"] for s in ok)),
        "median_distance_transported": float(np.median(
            [s["median"]["transported"] for s in ok])),
        "median_distance_untransported": float(np.median(
            [s["median"]["untransported"] for s in ok])),
        "median_distance_random": float(np.median(
            [s["median"]["random"] for s in ok])),
        "median_distance_floor": float(np.median(
            [s["median"]["floor"] for s in ok])),
    }


def twin_figure(results: dict, path) -> None:
    """Sorted gap-closed curves (pairwise and leave-one-out) beside sorted
    twin margins, full panel and HVG companion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    drew = False
    for key, alpha, tag in (("twins", 1.0, "all genes"),
                            ("twins_hvg", 0.45, "HVG 1000")):
        for version, colour, label in (
                ("pairwise", "#2c7fb8", "pairwise A→B"),
                ("leave_one_out", "#c9662a", "leave-one-niche-out")):
            rows = [p[key] for p in results.get(version, [])
                    if key in p and not p[key].get("insufficient")]
            if not rows:
                continue
            drew = True
            for ax, field in ((axes[0], "gap_closed"),
                              (axes[1], "twin_margin")):
                v = np.sort([r[field] for r in rows])
                ax.plot(np.linspace(0, 1, len(v)), v, lw=1.4, color=colour,
                        alpha=alpha, label=f"{label}, {tag} (n={len(v)})")
    if not drew:
        plt.close(fig)
        return
    for ax, title, ylab in (
            (axes[0], "matched twin: fraction of the per-cell niche gap "
                      "closed\n(untransported − transported)/(untransported "
                      "− floor)", "gap closed, per panel"),
            (axes[1], "does matching on z buy anything?\ntwin margin "
                      "(random − matched)/random", "twin margin, per panel")):
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_xlabel("panels, sorted", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
    axes[0].axhline(1.0, color="0.3", lw=1.0, ls="--")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sweep_sensitivity(args: argparse.Namespace) -> dict:
    """Doc-08 section 7.5: the transport summary of every run of a kappa
    sweep, one model resident at a time. Keyed by run name, each entry the
    run's two-tier summary plus its kappa and seed."""
    import gc

    import torch

    from discell import paths
    from discell.model.sweep import run_name

    out = paths.dataset(args.dataset).root / "experiments" / args.sweep_out
    table: dict = {}
    for seed in args.seeds:
        for kappa in args.kappas:
            run = run_name(kappa, seed, args.sweep_tag)
            run_dir = paths.dataset(args.dataset).root / "runs" / run
            if not (run_dir / "best.pt").exists():
                log.warning("missing %s -- skipped", run)
                continue
            results = transport_check(
                argparse.Namespace(**{**vars(args), "run": run}))
            table[run] = {"kappa": kappa, "seed": seed, **results["summary"]}
            del results
            gc.collect()
            torch.cuda.empty_cache()
            out.write_text(json.dumps(table, indent=2, default=float))
    log.info("wrote %s", out)
    return table


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", default=None)
    parser.add_argument("--sweep-tag", default=None,
                        help="run every model of a kappa sweep instead")
    parser.add_argument("--kappas", type=float, nargs="*",
                        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--sweep-out",
                        default="transport_kappa_sensitivity.json",
                        help="file name under experiments/ (sweep mode)")
    parser.add_argument("--niches", type=int, default=10)
    parser.add_argument("--niche-source", default="kmeans",
                        choices=("kmeans", "tumour-band"),
                        help="k-means on composition, or ordered bands of "
                             "the kNN-smoothed tumour fraction")
    parser.add_argument("--figures", type=int, default=8)
    parser.add_argument("--read", default="mean",
                        choices=("mean", "distribution", "twins", "both"),
                        help="mean-shift read (default), the distribution-"
                             "level MMD read (which always also produces "
                             "the model-vs-model variant), the matched-twin "
                             "per-cell read, or both")
    parser.add_argument("--hvg", type=int, default=0,
                        help="also run the new reads on this many Scanpy "
                             "seurat HVGs (0 disables)")
    parser.add_argument("--boot", type=int, default=N_BOOT,
                        help="paired bootstrap draws (distribution read)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.sweep_tag:
        sweep_sensitivity(args)
    elif args.run:
        if args.read in ("mean", "both"):
            transport_check(args)
        if args.read in ("distribution", "twins", "both"):
            distribution_check(args, twins=args.read in ("twins", "both"))
    else:
        parser.error("one of --run / --sweep-tag is required")
    return 0


if __name__ == "__main__":
    sys.exit(main())
