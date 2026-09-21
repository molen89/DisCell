#!/usr/bin/env python3
"""The w-program atlas (doc 08 section 6): the positive w story.

One row per *effective* program of w. The per-type mean of w is a gauge
(``(a(z) - B mu_t, w + mu_t)`` is the same model; issues V12), so w is
centred within type first; the programs are the r principal directions of
cov(w) carrying >= 1% of its trace (issues V10), varimax-rotated within
that subspace in gene space (``lr_map.program_basis``, shared). The
remaining d_w - r directions carry no variance and are reported as null
directions, not as programs. The rotation is applied to the effect
decomposition ``<w, B> = sum_k u_k L_k`` (u whitened coordinates, L the
effect-scale loadings), so the decoder's output is invariant under it.

Per program: variance share (in w and in the realised shift B w), gene
signature, a rank-based MSigDB hallmark label (Mann-Whitney on the full
loading vector against the expressed-panel background, BH per program,
q <= 0.05), tissue territory with Moran's I, type-partialled context
drivers (y, Phi-PCs, and landmark distances where the slide has landmarks
-- the section-2.5 collinearity logic), per-type within-type activity.
Concordance: per-axis cross-seed cosines and shift-space overlap with
``--compare-runs`` (issues V11; the matched-column B correlation is
dropped), hallmark-label recurrence across seeds (``--compare-runs``) and
across slides (``--compare-atlas``, gene panels need not agree), and
kappa-survival, which is sweep-internal (``validate --sweep-tag``) and
pointed at from the output.

Usage::

    python -m discell.model.atlas --dataset <id> --run ablation_gat_type_only_s1 \
        --compare-runs ablation_gat_type_only ablation_gat_type_only_s2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from discell.model.lr_map import VAR_FRACTION, program_basis
from discell.model.validate import (center_per_type, collect_latents,
                                    landmark_inventory, load_run, morans_i,
                                    ridge_cv, r2, row_normalised_graph)

log = logging.getLogger("discell.model.atlas")

HALLMARKS_GMT = Path("data/external/msigdb_hallmarks_h.all.v2023.2.Hs.symbols.gmt")
TOP_GENES = 15


def varimax(loadings: np.ndarray, max_iter: int = 100,
            tol: float = 1e-8) -> np.ndarray:
    """The rotation matrix R maximising the varimax criterion of ``L @ R``."""
    p, k = loadings.shape
    rotation = np.eye(k)
    variance = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        gradient = loadings.T @ (
            rotated ** 3 - rotated @ np.diag((rotated ** 2).sum(axis=0)) / p)
        u, s, vt = np.linalg.svd(gradient)
        rotation = u @ vt
        if s.sum() < variance * (1 + tol):
            break
        variance = s.sum()
    return rotation


def centred_program_basis(mu_w: np.ndarray, t: np.ndarray,
                          b_matrix: np.ndarray, expressed: np.ndarray):
    """Effective-rank programs of w read on within-type-centred w.

    Centring removes the per-type offset gauge (issues V12) before the
    spectrum of cov(w) is read; ``program_basis`` (lr_map, shared) then
    keeps the r directions with >= VAR_FRACTION of the trace and varimax-
    rotates them in gene space (issues V10). Programs are ordered by their
    share of w's variance, signed so the largest expressed loading is
    positive. Returns whitened coordinates ``u`` (N, r), effect-scale
    loadings (G, r) and ``{"rank", "variance_fraction", "variance_share",
    "shift_share"}`` -- the eigen-spectrum of cov(w), each program's share
    of w's variance and of the realised shift's variance on the subspace.
    """
    centred = center_per_type(mu_w, t, np.ones(len(t), dtype=bool))
    u, loadings, info = program_basis(centred, b_matrix, expressed)
    # u is whitened and uncorrelated, so program k occupies the w-direction
    # a_k = cov(u_k, w) with variance ||a_k||^2
    axes = u.T @ centred / len(u)                              # (r, d_w)
    share = (axes ** 2).sum(axis=1) / max(centred.var(axis=0).sum(), 1e-12)
    order = np.argsort(-share)
    u, loadings, share = u[:, order], loadings[:, order], share[order]
    sign = np.sign(loadings[expressed][
        np.abs(loadings[expressed]).argmax(axis=0), np.arange(len(share))])
    u, loadings = u * sign, loadings * sign
    shift = (loadings[expressed] ** 2).sum(axis=0)
    info["variance_share"] = share.round(4).tolist()
    info["shift_share"] = (shift / max(shift.sum(), 1e-12)).round(4).tolist()
    return u, loadings, info


def cross_seed(loadings: np.ndarray, other: np.ndarray) -> dict:
    """Agreement of two runs' program loadings (rows = the same genes).

    ``axis_cosine``: |cos| of each program of ``loadings`` with its best
    one-to-one match in ``other`` (None when ``other`` has fewer programs).
    ``shift_overlap``: the fraction of this run's realised-shift variance
    lying inside the other run's program span, and the reverse -- the
    invariant object across seeds (issues V11), where matched B columns
    read the null directions as instability.
    """
    from scipy.optimize import linear_sum_assignment

    def unit(m):
        return m / np.linalg.norm(m, axis=0, keepdims=True).clip(min=1e-12)

    cosines = np.abs(unit(loadings).T @ unit(other))            # (r, r')
    rows, cols = linear_sum_assignment(-cosines)
    matched = [None] * loadings.shape[1]
    for i, j in zip(rows, cols):
        matched[i] = {"program": int(j), "cosine": float(cosines[i, j])}

    def inside(a, b):
        q, _ = np.linalg.qr(b)
        return float(((q.T @ a) ** 2).sum() / (a ** 2).sum())

    return {"axis_cosine": matched,
            "shift_overlap": {"this_inside_other": inside(loadings, other),
                              "other_inside_this": inside(other, loadings)}}


def read_hallmarks(panel: set[str]) -> dict[str, set[str]]:
    """MSigDB hallmark sets intersected with the panel; empty sets dropped."""
    sets: dict[str, set[str]] = {}
    for line in HALLMARKS_GMT.read_text().splitlines():
        name, _url, *genes = line.split("\t")
        hits = set(genes) & panel
        if len(hits) >= 5:
            sets[name.removeprefix("HALLMARK_")] = hits
    return sets


def hallmark_labels(loading: np.ndarray, gene_names: np.ndarray,
                    expressed: np.ndarray, hallmarks: dict[str, set],
                    top: int = 3) -> list[dict]:
    """Rank-based hallmark enrichment of one program's *full* loading vector.

    The statistic is a two-sided Mann-Whitney U on the ranks of |loading|
    over the expressed panel: set members against the rest of the panel.
    No top-k cut is taken, so the label does not depend on an arbitrary
    signature length, and the background is the expressed panel, not the
    genome. Sets with < 5 panel genes are untestable and were dropped by
    ``read_hallmarks``. BH is applied across the sets tested for this
    program; a label ships only at q <= 0.05. ``direction`` is the mean
    signed loading of the members (the program's sign is itself arbitrary,
    so it orients the label, it does not gate it).
    """
    from scipy.stats import mannwhitneyu

    panel = gene_names[expressed]
    values = np.abs(loading[expressed])
    signed = loading[expressed]
    index = {g: i for i, g in enumerate(panel)}
    rows = []
    for name, members in hallmarks.items():
        idx = np.array([index[g] for g in members if g in index], dtype=int)
        if len(idx) < 5 or len(idx) >= len(panel):
            continue
        mask = np.zeros(len(panel), dtype=bool)
        mask[idx] = True
        stat = mannwhitneyu(values[mask], values[~mask],
                            alternative="two-sided", method="asymptotic")
        # AUC in [0, 1]: > 0.5 means the set's loadings are larger than the
        # panel's, the direction of interest; < 0.5 is a depleted set
        auc = float(stat.statistic) / (mask.sum() * (~mask).sum())
        rows.append({"hallmark": name, "n_panel": int(mask.sum()),
                     "auc": auc, "p": float(stat.pvalue),
                     "direction": float(signed[mask].mean())})
    rows.sort(key=lambda r: r["p"])
    m = len(rows)
    for i, row in enumerate(rows):
        row["q"] = min(row["p"] * m / (i + 1), 1.0)
    for i in range(m - 2, -1, -1):
        rows[i]["q"] = min(rows[i]["q"], rows[i + 1]["q"])
    for row in rows:
        row["significant"] = bool(row["q"] <= 0.05 and row["auc"] > 0.5)
    return [r for r in rows if r["auc"] > 0.5][:top]


def label_recurrence(label_sets: dict[str, list[list[str]]],
                     minimum: int = 2) -> dict:
    """How often a hallmark label recurs across runs (seeds or slides).

    ``label_sets`` maps a run name to its per-program list of significant
    hallmark names. A label recurs when it is significant in at least
    *minimum* of the runs (on any program -- program order is not shared
    across fits, only the label set is). Returns the per-label run counts,
    the recurring labels, and the per-run label sets for the record.
    """
    per_run = {run: sorted({name for names in programs for name in names})
               for run, programs in label_sets.items()}
    counts: dict[str, int] = {}
    for names in per_run.values():
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    return {"n_runs": len(per_run),
            "minimum": minimum,
            "label_sets": per_run,
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "recurring": sorted(n for n, c in counts.items() if c >= minimum)}


def significant_labels(atlas: dict) -> list[list[str]]:
    """Per-program significant hallmark names of a built atlas."""
    return [[h["hallmark"] for h in p["hallmarks"] if h.get("significant")]
            for p in atlas["programs"]]


def landmark_distance_block(positions: np.ndarray, classes: dict):
    """Log distance to the nearest constituent of each landmark class.

    ``None`` when the slide has no landmark class with constituents. The
    inventory is defined from annotated type names, so a cluster-labelled
    slide ("Cluster 7") yields classes that are empty or absent; an empty
    block is not an empty driver but an unaskable question, and feeding it
    to DBSCAN/cKDTree is what made the atlas fail on graphclust slides.
    Returns ``(block (N, n_classes), names)``.
    """
    from scipy.spatial import cKDTree

    usable = {name: entry for name, entry in classes.items()
              if len(entry["constituents"]) > 0}
    if not usable:
        return None
    block = np.stack([
        np.log1p(np.minimum(
            cKDTree(positions[entry["constituents"]]).query(positions)[0],
            500.0))
        for entry in usable.values()], axis=1)
    return block, list(usable)


def context_drivers(u_k: np.ndarray, blocks: dict[str, np.ndarray],
                    fold: np.ndarray, rows: np.ndarray) -> dict:
    """Held-out R2 of one program from each context block, marginal + joint.

    Marginal R2s overlap (section 2.5 collinearity -- vessel density itself
    varies rim to core); the joint row bounds the total, the partials name
    the unique shares.
    """
    target = u_k[rows]
    out = {"marginal": {}, "partial": {}}
    for name, block in blocks.items():
        out["marginal"][name] = r2(target, ridge_cv(block[rows], target,
                                                    fold[rows]))
    joint = np.hstack(list(blocks.values()))
    out["joint"] = r2(target, ridge_cv(joint[rows], target, fold[rows]))
    for name in blocks:
        rest = np.hstack([b for n, b in blocks.items() if n != name])
        out["partial"][name] = out["joint"] - r2(
            target, ridge_cv(rest[rows], target, fold[rows]))
    return out


def build_atlas(args: argparse.Namespace) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, data, trainer, run_dir, b_matrix = load_run(
        args.dataset, args.run, args.device)
    latents = collect_latents(trainer, data)
    rng = np.random.default_rng(config.seed)
    gene_names = np.asarray([str(g) for g in data.gene_names])
    prevalence = np.asarray((data.x > 0).mean(axis=0)).ravel()
    expressed = prevalence >= 0.01
    hallmarks = read_hallmarks(set(gene_names[expressed]))

    # -- canonical basis: effective rank of within-type-centred w, varimax
    # within the r-dim subspace (V10, V12) ---------------------------------
    u, programs, info = centred_program_basis(latents["mu_w"], data.t,
                                              b_matrix, expressed)
    r, d_w = info["rank"], b_matrix.shape[1]
    log.info("effective rank %d of %d (eigen-fractions %s); program shares %s",
             r, d_w, info["variance_fraction"], info["variance_share"])

    # shared context blocks for the drivers
    connected = data.graph.degrees > 0
    weights, _ = row_normalised_graph(data)
    weights_cc = weights[connected][:, connected]
    u_centred = center_per_type(u, data.t, connected)
    moran = morans_i(u_centred[connected], weights_cc,
                     n_perms=args.n_perms, seed=config.seed)

    rng_phi = np.random.default_rng(0)
    fit_rows = rng_phi.choice(len(data.phi), min(len(data.phi), 50_000),
                              replace=False)
    from sklearn.decomposition import PCA
    phi_pcs = PCA(12, random_state=0).fit(
        data.phi[fit_rows]).transform(data.phi).astype(np.float64)
    landmarks = landmark_distance_block(data.positions,
                                        landmark_inventory(data))
    # type-partialled, like the target: a program's drivers are the cell's
    # context within its type, not its type identity via homophilous y
    everyone = np.ones(len(data.t), dtype=bool)
    blocks = {"composition_y": data.graph.y.astype(np.float64),
              "phi_pcs": phi_pcs}
    if landmarks is None:
        classes: list[str] = []
        log.warning("no landmark class has constituents on this slide "
                    "(cluster labels?) -- the landmark driver block is dropped")
    else:
        blocks["landmark_distances"], classes = landmarks
    blocks = {n: center_per_type(b, data.t, everyone) for n, b in blocks.items()}

    sample = np.flatnonzero(connected)
    if len(sample) > 30_000:
        sample = np.sort(rng.choice(sample, 30_000, replace=False))

    out_dir = run_dir / "atlas"
    out_dir.mkdir(exist_ok=True)
    for stale in out_dir.glob("program_*.png"):
        stale.unlink()
    np.save(out_dir / "programs.npy", programs.astype(np.float32))
    names = [str(n) for n in data.type_names]
    atlas: dict = {"run": args.run, "d_w": d_w, "rank": r,
                   "rank_var_fraction": VAR_FRACTION,
                   "variance_fraction": info["variance_fraction"],
                   "landmark_classes": classes,
                   "driver_blocks": list(blocks),
                   "kappa_survival": "see experiments/atlas_kappa_survival*.json "
                                     "(validate --sweep-tag; doc-08 section 6.5)",
                   "cross_seed": {}, "programs": []}
    for other in args.compare_runs:
        path = run_dir.parent / other / "atlas" / "programs.npy"
        if not path.exists():
            log.warning("no atlas for %s -- build it first; skipped", other)
            continue
        other_programs = np.load(path)
        atlas["cross_seed"][other] = cross_seed(programs[expressed],
                                                other_programs[expressed])
        atlas["cross_seed"][other]["rank"] = int(other_programs.shape[1])
        log.info("vs %s: %s", other, atlas["cross_seed"][other])
    show = np.sort(rng.choice(np.flatnonzero(connected),
                              min(120_000, int(connected.sum())),
                              replace=False))
    for k in range(r):
        entry: dict = {"program": k, "active": True,
                       "variance_share": info["variance_share"][k],
                       "shift_share": info["shift_share"][k],
                       "moran_I": moran["I"][k],
                       "moran_null_hi": moran["null_hi"][k]}
        loading = np.where(expressed, programs[:, k], 0.0)
        top = np.argsort(-np.abs(loading))
        entry["signature_high"] = [(str(gene_names[i]), float(loading[i]))
                                   for i in top[:TOP_GENES]
                                   if loading[i] > 0][:TOP_GENES]
        entry["signature_low"] = [(str(gene_names[i]), float(loading[i]))
                                  for i in top[:3 * TOP_GENES]
                                  if loading[i] < 0][:TOP_GENES]
        entry["hallmarks"] = hallmark_labels(programs[:, k], gene_names,
                                             expressed, hallmarks)
        entry["drivers"] = context_drivers(u[:, k], blocks,
                                           latents["fold"], sample)
        per_type_var = {names[g]: float(u[data.t == g, k].var())
                        for g in range(len(names))}
        total = sum(per_type_var.values()) or 1.0
        entry["type_activity"] = dict(sorted(
            ((n, v / total) for n, v in per_type_var.items()),
            key=lambda kv: -kv[1])[:8])

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6),
                                 gridspec_kw={"width_ratios": [1.3, 1, 1]})
        sc = axes[0].scatter(data.positions[show, 0], data.positions[show, 1],
                             c=u[show, k], cmap="RdBu_r", s=0.5,
                             vmin=np.percentile(u[show, k], 2),
                             vmax=np.percentile(u[show, k], 98),
                             rasterized=True)
        axes[0].set_aspect("equal"); axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].set_title(f"program {k} ({100 * entry['variance_share']:.0f}% "
                          f"of w variance) territory "
                          f"(Moran I {entry['moran_I']:.2f})", fontsize=9)
        plt.colorbar(sc, ax=axes[0], fraction=0.04)
        genes = entry["signature_high"][:10] + entry["signature_low"][:5]
        axes[1].barh(range(len(genes)), [v for _, v in genes], height=0.7)
        axes[1].set_yticks(range(len(genes)),
                           [g for g, _ in genes], fontsize=6)
        top_hm = next((h for h in entry["hallmarks"] if h["significant"]), None)
        label = top_hm["hallmark"] if top_hm else "unlabelled (no set at q<=0.05)"
        axes[1].set_title(f"signature | hallmark: {label[:28]}", fontsize=8)
        marg = entry["drivers"]["marginal"]
        part = entry["drivers"]["partial"]
        xs = np.arange(len(marg))
        axes[2].bar(xs - 0.2, list(marg.values()), width=0.4,
                    label="marginal R²")
        axes[2].bar(xs + 0.2, [part[n] for n in marg], width=0.4,
                    label="partial R²")
        axes[2].set_xticks(xs, [n[:12] for n in marg], fontsize=7)
        axes[2].axhline(0, color="0.3", lw=0.6)
        axes[2].set_title(f"drivers (joint {entry['drivers']['joint']:.2f})",
                          fontsize=8)
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"program_{k}.png", dpi=130)
        plt.close(fig)
        atlas["programs"].append(entry)
        log.info("program %d: share %.3f, Moran %.2f, hallmark %s, joint R² %.2f",
                 k, entry["variance_share"], entry["moran_I"], label,
                 entry["drivers"]["joint"])

    # -- concordance (3 of the 3 reads; the third, kappa-survival, is
    # sweep-internal and pointed at below). Labels are compared as sets:
    # program order and gene panels need not agree, so the same function
    # serves seeds (same slide) and slides (different panels).
    label_sets = {args.run: significant_labels(atlas)}
    for other in list(args.compare_runs) + list(args.compare_atlas):
        path = (run_dir.parent / other / "atlas" / "atlas.json"
                if other in args.compare_runs else Path(other))
        if not path.exists():
            log.warning("no atlas.json at %s -- skipped for label recurrence",
                        path)
            continue
        other_atlas = json.loads(path.read_text())
        label_sets[other_atlas.get("run", str(path))] = significant_labels(
            other_atlas)
    atlas["label_recurrence"] = label_recurrence(label_sets)
    log.info("label recurrence over %d runs: %s",
             atlas["label_recurrence"]["n_runs"],
             atlas["label_recurrence"]["counts"])

    (out_dir / "atlas.json").write_text(json.dumps(atlas, indent=2,
                                                   default=float))
    log.info("wrote %s", out_dir / "atlas.json")
    return atlas


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--n-perms", type=int, default=500)
    parser.add_argument("--compare-runs", nargs="*", default=[],
                        help="other runs of this dataset whose atlas exists; "
                             "cross-seed loading cosines and shift overlap")
    parser.add_argument("--compare-atlas", nargs="*", default=[],
                        help="paths to atlas.json files of OTHER slides; "
                             "label recurrence only (gene panels differ)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    build_atlas(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
